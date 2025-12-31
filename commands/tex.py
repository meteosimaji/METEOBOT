import asyncio
import io
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import discord
from discord.ext import commands
from PIL import Image

from utils import defer_interaction


DEFAULT_DPI = 300
DEFAULT_COMPILE_TIMEOUT_S = 15
DEFAULT_RASTER_TIMEOUT_S = 10
DEFAULT_MAX_BYTES = 7_800_000
DEFAULT_AUTOWRAP = True

MIN_DPI = 72
MAX_DPI = 600
MIN_TIMEOUT_S = 3
MAX_TIMEOUT_S = 60
MIN_MAX_BYTES = 100_000
MAX_MAX_BYTES = 8_000_000
DEFAULT_MAX_PAGES = 3
MIN_MAX_PAGES = 1
MAX_MAX_PAGES = 4
DISCORD_ATTACHMENT_LIMIT = 10

MAX_LATEX_CHARS = 20_000


@dataclass(frozen=True)
class RenderResult:
    png_white_bytes: list[bytes]
    png_transparent_bytes: list[bytes]
    pdf_bytes: Optional[bytes]
    used_engine: str
    pdf_omitted_for_size: bool = False


class LatexRenderError(RuntimeError):
    pass


def _which(*names: str) -> Optional[str]:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def _is_full_document(src: str) -> bool:
    return "\\documentclass" in src


def _assert_supported_document(src: str) -> None:
    r"""Reject document classes that won't compile under Tectonic/XeTeX.

    A common failure case is `\documentclass[...uplatex...]{jsarticle}`, which is a
    pLaTeX/upLaTeX workflow incompatible with Tectonic. Suggest bxjsarticle instead.
    """

    m = re.search(r"\\documentclass\s*(\[[^\]]*\])?\{([^}]+)\}", src, flags=re.IGNORECASE)
    if not m:
        return
    options = (m.group(1) or "").lower()
    cls = (m.group(2) or "").lower().strip()

    unsupported_classes = {"jsarticle", "jsbook", "jsreport"}
    if cls in unsupported_classes:
        raise LatexRenderError(
            f"`{cls}` is a pLaTeX/upLaTeX document class and is not supported under Tectonic. "
            "Switch to `\\documentclass{bxjsarticle}` (XeTeX-compatible) before rendering."
        )

    if "uplatex" in options or "platex" in options:
        raise LatexRenderError(
            "The `uplatex`/`platex` options are not supported under Tectonic. "
            "Switch to `\\documentclass{bxjsarticle}` (XeTeX-compatible) before rendering."
        )


def _looks_like_raw_body(src: str) -> bool:
    # If the user is already using LaTeX environments/delimiters, don't auto-wrap.
    if "\\begin{" in src or "\\end{" in src:
        return True
    if "\\[" in src or "\\]" in src or "\\(" in src or "\\)" in src:
        return True
    if "$$" in src or "\\section" in src or "\\subsection" in src:
        return True
    # Inline math $...$ (ignore escaped \$)
    tmp = re.sub(r"\\\$", "", src)
    if tmp.count("$") >= 2:
        return True
    return False


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", text))


def _wrap_expression(src: str) -> str:
    s = src.strip()
    if not s:
        return r"\[\;\]"
    if _looks_like_raw_body(s):
        return s
    # Single-line, no obvious LaTeX structure => interpret as a math expression.
    return r"\[\displaystyle " + s + r"\]"


def _reject_dangerous_tex(src: str) -> None:
    """Best-effort safety filter.

    This bot can be called indirectly by /ask (LLM tool). Without isolation, TeX can
    potentially read local files (\\input, \\openin, etc.). We block common I/O macros.

    Note: This is not a perfect sandbox. The real safety boundary should be OS-level.
    """

    # Allow \usepackage etc, but block obvious file I/O and command execution primitives.
    banned = [
        r"\\write18",
        r"\\input\b",
        r"\\include\b",
        r"\\includegraphics\b",
        r"\\openin",
        r"\\openout",
        r"\\read",
        r"\\write\b",
        r"\\file\b",
        r"\\catcode",
        r"\\usepackage\s*\{shellesc\}",
        r"\\immediate\s*\\write",
        r"\\pdfobj",
        r"\\pdfliteral",
        r"\\directlua",
        r"\\special",
    ]
    for pat in banned:
        if re.search(pat, src, flags=re.IGNORECASE):
            raise LatexRenderError(
                "Security filter: forbidden TeX primitive detected (file I/O / execution). "
                "If you truly need advanced TeX features, run them locally instead of on the bot."
            )


def _default_template_xetex(body: str) -> str:
    return (
        textwrap.dedent(
            r"""
            \documentclass[preview,border=2pt]{standalone}

            % --- Math ---
            \usepackage{amsmath,amssymb,mathtools}
            \usepackage{physics}
            \usepackage{siunitx}
            \usepackage{mhchem}
            \usepackage{bm}

            % --- Graphics / circuits ---
            \usepackage{graphicx}
            \usepackage{tikz}
            \usetikzlibrary{arrows.meta,calc,positioning,decorations.pathmorphing}
            \usepackage{circuitikz}

            % --- Colors & background ---
            \usepackage{xcolor}
            \nopagecolor

            % --- Unicode / Japanese ---
            \usepackage{fontspec}
            \usepackage{xeCJK}
            \defaultfontfeatures{Ligatures=TeX}

            % Latin font
            \IfFontExistsTF{TeX Gyre Termes}{\setmainfont{TeX Gyre Termes}}{}

            % CJK font fallbacks (use what's installed)
            \IfFontExistsTF{Noto Sans CJK JP}{\setCJKmainfont{Noto Sans CJK JP}}{%
              \IfFontExistsTF{Noto Serif CJK JP}{\setCJKmainfont{Noto Serif CJK JP}}{%
                \IfFontExistsTF{IPAexGothic}{\setCJKmainfont{IPAexGothic}}{%
                  \IfFontExistsTF{IPAMincho}{\setCJKmainfont{IPAMincho}}{%
                    \setCJKmainfont{FandolSong-Regular}% final forced fallback
                  }
                }
              }
            }

            % Make output crisp
            \linespread{1.0}

            \begin{document}
            {body}
            \end{document}
            """
        )
        .strip()
        .replace("{body}", body)
        + "\n"
    )


def _default_text_template_xetex(body: str) -> str:
    return (
        textwrap.dedent(
            r"""
            \documentclass[border=6pt]{standalone}

            \usepackage{xcolor}
            \usepackage{fontspec}
            \usepackage{xeCJK}
            \usepackage{amsmath,amssymb}
            \defaultfontfeatures{Ligatures=TeX}

            % Latin font
            \IfFontExistsTF{TeX Gyre Termes}{\setmainfont{TeX Gyre Termes}}{}

            % CJK font fallbacks (use what's installed)
            \IfFontExistsTF{Noto Sans CJK JP}{\setCJKmainfont{Noto Sans CJK JP}}{%
              \IfFontExistsTF{Noto Serif CJK JP}{\setCJKmainfont{Noto Serif CJK JP}}{%
                \IfFontExistsTF{IPAexGothic}{\setCJKmainfont{IPAexGothic}}{%
                  \IfFontExistsTF{IPAMincho}{\setCJKmainfont{IPAMincho}}{%
                    \setCJKmainfont{FandolSong-Regular}% final forced fallback
                  }
                }
              }
            }

            \linespread{1.05}
            \setlength{\parindent}{0pt}
            \setlength{\parskip}{6pt}
            \nopagecolor

            \begin{document}
            {body}
            \end{document}
            """
        )
        .strip()
        .replace("{body}", body)
        + "\n"
    )


def _pick_engine() -> tuple[str, str]:
    """Return (engine_kind, engine_path_or_name).

    Only Tectonic is allowed because it supports untrusted mode for safer input.
    """
    if _which("tectonic"):
        return "tectonic", "tectonic"
    raise LatexRenderError(
        "Tectonic is required for /tex (untrusted mode). Install 'tectonic' and ensure it is on PATH."
    )


def _run_subprocess(cmd: list[str], timeout_s: int, cwd: Path) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
            text=True,
        )
    except subprocess.TimeoutExpired as e:
        raise LatexRenderError(f"LaTeX compile timed out ({timeout_s}s).") from e


def _compile_to_pdf(tex_source: str, workdir: Path, timeout_s: int) -> tuple[Path, str]:
    engine_kind, engine = _pick_engine()

    tex_path = workdir / "input.tex"
    tex_path.write_text(tex_source, encoding="utf-8")

    if engine_kind == "tectonic":
        # Use --untrusted for safer processing of LLM-provided input.
        # Also force untrusted via env var as a mild extra layer.
        env = os.environ.copy()
        env["TECTONIC_UNTRUSTED_MODE"] = "1"
        cmd = [
            engine,
            "-X",
            "compile",
            "--untrusted",
            "--outdir",
            str(workdir),
            str(tex_path),
        ]
        try:
            p = subprocess.run(
                cmd,
                cwd=str(workdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
                text=True,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            raise LatexRenderError(f"Tectonic timed out ({timeout_s}s).") from e

        if p.returncode != 0:
            # Compatibility fallback: some older tectonic builds don't support `-X compile`.
            out = p.stdout or ""
            if "unknown" in out.lower() and "-x" in out.lower():
                cmd2 = [engine, "--outdir", str(workdir), str(tex_path)]
                try:
                    p2 = subprocess.run(
                        cmd2,
                        cwd=str(workdir),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        timeout=timeout_s,
                        check=False,
                        text=True,
                        env=env,
                    )
                except subprocess.TimeoutExpired as e:
                    raise LatexRenderError(f"Tectonic timed out ({timeout_s}s).") from e
                if p2.returncode != 0:
                    raise LatexRenderError(_summarize_tex_output(p2.stdout))
            else:
                raise LatexRenderError(_summarize_tex_output(out))
        pdf_path = workdir / "input.pdf"
        if not pdf_path.exists():
            # Tectonic may output texput.pdf if input is '-' but we use a filename.
            alt = workdir / "texput.pdf"
            if alt.exists():
                pdf_path = alt
            else:
                raise LatexRenderError("Compilation succeeded but PDF was not produced (unexpected).")
        return pdf_path, "tectonic"

    raise LatexRenderError("Reached unexpected engine branch (only tectonic is supported).")


def _summarize_tex_output(output: str, max_chars: int = 1500) -> str:
    if not output:
        return "LaTeX failed (no output captured)."
    # Pull the first explicit TeX error (starts with '!'). If absent, return tail.
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("!"):
            snippet = "\n".join(lines[i : i + 8])
            return snippet[:max_chars]
    return "\n".join(lines[-20:])[:max_chars]


def _pdf_to_png_via_gs(
    pdf_path: Path,
    out_prefix: Path,
    dpi: int,
    timeout_s: int,
    max_pages: int,
    *,
    device: str = "pngalpha",
) -> list[Path]:
    gs = _which("gs", "gswin64c", "gswin32c")
    if not gs:
        raise LatexRenderError("Ghostscript not found (needed for PDF→PNG). Install 'gs'.")

    out_pattern = out_prefix.with_name(out_prefix.stem + "-%03d.png")
    cmd = [
        gs,
        "-q",
        "-dBATCH",
        "-dNOPAUSE",
        "-dSAFER",
        "-dFirstPage=1",
        f"-dLastPage={max_pages}",
        f"-sDEVICE={device}",
        f"-r{dpi}",
        "-dTextAlphaBits=4",
        "-dGraphicsAlphaBits=4",
        "-dUseCropBox",
        f"-sOutputFile={out_pattern}",
        str(pdf_path),
    ]
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
        text=True,
    )
    outputs = sorted(out_prefix.parent.glob(out_prefix.stem + "-" + "[0-9][0-9][0-9]" + ".png"))
    if p.returncode != 0 or not outputs:
        raise LatexRenderError("Ghostscript failed to rasterize PDF to PNG.\n" + _summarize_tex_output(p.stdout))
    return outputs


def _encode_png_with_limit(img: Image.Image, max_bytes: int) -> tuple[bytes, Image.Image]:
    """Encode an image to PNG, downscaling if necessary to fit the size limit."""

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    while buf.tell() > max_bytes and img.width > 16 and img.height > 16:
        scale = (max_bytes / buf.tell()) ** 0.5
        new_w = max(16, int(img.width * scale))
        new_h = max(16, int(img.height * scale))
        img = img.resize((new_w, new_h), resample=Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), img


def _encode_dual_png_with_limit(rgba_img: Image.Image, max_bytes: int) -> tuple[bytes, bytes]:
    """Encode transparent and white PNGs with matched dimensions.

    If the white variant requires extra downscaling to fit `max_bytes`, retry both
    encodes using the same reduced size so the toggle view swaps images seamlessly.
    """

    candidate = rgba_img
    while True:
        transparent_bytes, scaled_rgba = _encode_png_with_limit(candidate, max_bytes)

        white_img = _flatten_to_white(scaled_rgba)
        white_bytes, scaled_white = _encode_png_with_limit(white_img, max_bytes)

        if scaled_white.size != scaled_rgba.size:
            # Keep both variants at the same resolution; retry from the smaller size.
            candidate = scaled_white.convert("RGBA")
            continue

        return transparent_bytes, white_bytes


def _flatten_to_white(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    background = Image.new("RGB", img.size, "white")
    background.paste(img, mask=img.getchannel("A"))
    return background


def render_latex_to_png_pdf(
    src: str,
    dpi: int,
    compile_timeout_s: int,
    raster_timeout_s: int,
    max_bytes: int,
    max_pages: int,
) -> RenderResult:
    """Render LaTeX -> (transparent + white PNGs, PDF)."""
    _reject_dangerous_tex(src)

    with tempfile.TemporaryDirectory(prefix="latexbot_") as td:
        workdir = Path(td)

        if _is_full_document(src):
            _assert_supported_document(src)
            tex_source = src.strip() + "\n"
        else:
            s = src.strip()
            # Accept user shorthand: [ ... ]  -> \[ ... \]
            if s.startswith("[") and s.endswith("]") and not s.startswith(r"\["):
                s = r"\[" + s[1:-1].strip() + r"\]"
            explicit_env = _looks_like_raw_body(s)
            contains_cjk = _contains_cjk(s)
            autowrap = _bool_env("LATEXBOT_AUTOWRAP", DEFAULT_AUTOWRAP)
            if contains_cjk and not explicit_env:
                tex_source = _default_text_template_xetex(body=s)
            else:
                if "\n" in s and not explicit_env:
                    raise LatexRenderError(
                        "Multi-line input must use an explicit math environment. "
                        "Wrap it like: \\[\\begin{aligned} ... \\end{aligned}\\] and align equals with `&`, "
                        "or provide a full LaTeX document (\\documentclass ...)."
                    )

                if not explicit_env and not autowrap:
                    raise LatexRenderError(
                        "No explicit math delimiters found and LATEXBOT_AUTOWRAP=0. Wrap your expression with $...$ or \\[...\\], "
                        "or provide a full LaTeX document (\\documentclass ...), or re-enable auto-wrap with LATEXBOT_AUTOWRAP=1."
                    )

                body = _wrap_expression(s) if (not explicit_env and autowrap) else s
                tex_source = _default_template_xetex(body=body)

        pdf_path, engine = _compile_to_pdf(tex_source, workdir=workdir, timeout_s=compile_timeout_s)

        tmp_png = workdir / "render"
        page_pngs = _pdf_to_png_via_gs(
            pdf_path,
            tmp_png,
            dpi=dpi,
            timeout_s=raster_timeout_s,
            max_pages=max_pages,
            device="pngalpha",
        )

        png_white_bytes: list[bytes] = []
        png_transparent_bytes: list[bytes] = []
        for page_path in page_pngs:
            with Image.open(page_path) as im:
                rgba_img = im.convert("RGBA")

            transparent_bytes, white_bytes = _encode_dual_png_with_limit(rgba_img, max_bytes)

            png_transparent_bytes.append(transparent_bytes)
            png_white_bytes.append(white_bytes)

        pdf_bytes: Optional[bytes] = None
        pdf_omitted_for_size = False
        if pdf_path.exists():
            pdf_size = pdf_path.stat().st_size
            if pdf_size <= max_bytes:
                pdf_bytes = pdf_path.read_bytes()
            else:
                pdf_omitted_for_size = True

        return RenderResult(
            png_white_bytes=png_white_bytes,
            png_transparent_bytes=png_transparent_bytes,
            pdf_bytes=pdf_bytes,
            used_engine=engine,
            pdf_omitted_for_size=pdf_omitted_for_size,
        )


def _int_env(var: str, default: int, *, min_val: int, max_val: int) -> int:
    raw = os.environ.get(var)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return max(min_val, min(max_val, parsed))


def _bool_env(var: str, default: bool) -> bool:
    raw = os.environ.get(var)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


async def _send_response(
    ctx: commands.Context,
    *,
    content: Optional[str] = None,
    embed: Optional[discord.Embed] = None,
    embeds: Optional[list[discord.Embed]] = None,
    files: Optional[list[discord.File]] = None,
    mention_author: bool = False,
) -> discord.Message:
    if embed is not None and embeds is not None:
        raise ValueError("Pass either embed or embeds, not both.")

    send_kwargs = {}
    if content is not None:
        send_kwargs["content"] = content
    if embed is not None:
        send_kwargs["embed"] = embed
    if embeds is not None:
        send_kwargs["embeds"] = embeds
    if files is not None:
        send_kwargs["files"] = files

    if ctx.interaction:
        return await ctx.interaction.followup.send(**send_kwargs)

    return await ctx.reply(**send_kwargs, mention_author=mention_author)


def _build_preview_embed(
    *,
    white_files: list[str],
    transparent_files: list[str],
    pdf_omitted_for_size: bool,
    max_bytes: int,
    color: int = 0xE67E22,
) -> discord.Embed:
    embed = discord.Embed(
        title="TeX Render",
        description=(
            "White preview is shown as the main image; transparent preview is the thumbnail. "
            "All white + transparent pages are attached for download."
        ),
        color=color,
    )

    if len(white_files) > 1:
        embed.add_field(
            name="Pages",
            value=(
                f"Showing page 1 preview. Attachments include {len(white_files)} pages "
                "for both backgrounds."
            ),
            inline=False,
        )

    if pdf_omitted_for_size:
        embed.add_field(
            name="PDF",
            value=f"PDF attachment omitted (over ~{max_bytes:,} bytes).",
            inline=False,
        )

    embed.set_image(url=f"attachment://{white_files[0]}")
    embed.set_thumbnail(url=f"attachment://{transparent_files[0]}")
    return embed


class Tex(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.hybrid_command(
        name="tex",
        description="Render LaTeX to PNG (transparent + white) and PDF.",
        help=(
            "Render LaTeX to crisp PNGs with both transparent and white backgrounds (+ PDF).\n"
            "- If you paste a full document (contains \\documentclass), it compiles as-is.\n"
            "- Otherwise include math delimiters ($...$, \\[...\\]); single-line input auto-wraps by default (disable with LATEXBOT_AUTOWRAP=0).\n"
            "Tip: TikZ/CircuiTikZ and Japanese text are supported when Tectonic is installed."
        ),
        usage="<latex>",
        extras={
            "category": "Tools",
            "pro": "Requires `tectonic` (untrusted mode) and Ghostscript on the server for PNG output.",
            "destination": "Render LaTeX into PNG images with both white and transparent backgrounds plus the PDF for download.",
            "plus": "Both white and transparent previews are attached; full documents compile as-is and auto-wrap is on by default (set LATEXBOT_AUTOWRAP=0 to require delimiters).",
        },
    )
    async def tex(self, ctx: commands.Context, *, arg: str):
        await defer_interaction(ctx)

        # Limits: keep it sane for Discord + compilation.
        arg = (arg or "").strip()
        if not arg:
            empty_embed = discord.Embed(
                title="🧪 No LaTeX Provided",
                description=(
                    "Add something to render, e.g., `/tex \\[\\frac{a}{b}\\]`. Single-line math auto-wraps by default."
                ),
                color=0xE74C3C,
            )
            await _send_response(
                ctx,
                content="No LaTeX provided.",
                embed=empty_embed,
                mention_author=False,
            )
            return
        if len(arg) > MAX_LATEX_CHARS:
            too_long = discord.Embed(
                title="🧪 LaTeX Too Long",
                description=f"Please trim your input to {MAX_LATEX_CHARS:,} characters or less.",
                color=0xE74C3C,
            )
            await _send_response(
                ctx,
                content="LaTeX input exceeds the limit.",
                embed=too_long,
                mention_author=False,
            )
            return

        dpi = _int_env(
            "LATEXBOT_DPI",
            DEFAULT_DPI,
            min_val=MIN_DPI,
            max_val=MAX_DPI,
        )
        compile_timeout_s = _int_env(
            "LATEXBOT_COMPILE_TIMEOUT_S",
            DEFAULT_COMPILE_TIMEOUT_S,
            min_val=MIN_TIMEOUT_S,
            max_val=MAX_TIMEOUT_S,
        )
        raster_timeout_s = _int_env(
            "LATEXBOT_RASTER_TIMEOUT_S",
            DEFAULT_RASTER_TIMEOUT_S,
            min_val=MIN_TIMEOUT_S,
            max_val=MAX_TIMEOUT_S,
        )
        max_bytes = _int_env(
            "LATEXBOT_MAX_BYTES",
            DEFAULT_MAX_BYTES,
            min_val=MIN_MAX_BYTES,
            max_val=MAX_MAX_BYTES,
        )
        max_pages = _int_env(
            "LATEXBOT_MAX_PAGES",
            DEFAULT_MAX_PAGES,
            min_val=MIN_MAX_PAGES,
            max_val=MAX_MAX_PAGES,
        )

        # Render off the event loop.
        try:
            result = await asyncio.to_thread(
                render_latex_to_png_pdf,
                arg,
                dpi,
                compile_timeout_s,
                raster_timeout_s,
                max_bytes,
                max_pages,
            )
        except LatexRenderError as e:
            msg = str(e).strip()
            if len(msg) > 1900:
                msg = msg[:1900] + "…"
            error_embed = discord.Embed(
                title="⚠️ TeX Render Failed",
                description=f"```\n{msg}\n```",
                color=0xE74C3C,
            )
            await _send_response(
                ctx,
                content=f"TeX render failed:\n```\n{msg}\n```",
                embed=error_embed,
                mention_author=False,
            )
            return
        except Exception as e:
            error_embed = discord.Embed(
                title="⚠️ Unexpected TeX Error",
                description=f"```\n{repr(e)}\n```",
                color=0xE74C3C,
            )
            await _send_response(
                ctx,
                content=f"Unexpected TeX error:\n```\n{repr(e)}\n```",
                embed=error_embed,
                mention_author=False,
            )
            return

        if len(result.png_white_bytes) != len(result.png_transparent_bytes):
            error_msg = "Internal error: rendered PNG variant counts differ."
            await _send_response(
                ctx,
                content=error_msg,
                embed=discord.Embed(title="⚠️ TeX Render Failed", description=error_msg, color=0xE74C3C),
            )
            return

        attachment_budget = len(result.png_white_bytes) * 2 + (1 if result.pdf_bytes else 0)
        if attachment_budget > DISCORD_ATTACHMENT_LIMIT:
            over_embed = discord.Embed(
                title="📎 Attachment Limit Hit",
                description=(
                    f"This render would need {attachment_budget} files, which exceeds Discord's limit of {DISCORD_ATTACHMENT_LIMIT}.\n"
                    "Try reducing the page count to stay within the cap."
                ),
                color=0xE67E22,
            )
            await _send_response(
                ctx,
                content=over_embed.description,
                embed=over_embed,
                mention_author=False,
            )
            return

        files: list[discord.File] = []
        try:
            png_white_files: list[str] = []
            png_transparent_files: list[str] = []
            for idx, (white_data, transparent_data) in enumerate(
                zip(result.png_white_bytes, result.png_transparent_bytes), start=1
            ):
                white_name = "render-white.png" if len(result.png_white_bytes) == 1 else f"render-white-{idx}.png"
                transparent_name = (
                    "render-transparent.png"
                    if len(result.png_transparent_bytes) == 1
                    else f"render-transparent-{idx}.png"
                )

                white_io = io.BytesIO(white_data)
                white_io.seek(0)
                files.append(discord.File(fp=white_io, filename=white_name))
                png_white_files.append(white_name)

                transparent_io = io.BytesIO(transparent_data)
                transparent_io.seek(0)
                files.append(discord.File(fp=transparent_io, filename=transparent_name))
                png_transparent_files.append(transparent_name)

            if result.pdf_bytes:
                pdf_io = io.BytesIO(result.pdf_bytes)
                pdf_io.seek(0)
                files.append(discord.File(fp=pdf_io, filename="render.pdf"))

            if not png_white_files or not png_transparent_files:
                error_embed = discord.Embed(
                    title="⚠️ TeX Render Failed",
                    description="Internal error: render completed but no previews were produced.",
                    color=0xE74C3C,
                )
                await _send_response(
                    ctx,
                    content=error_embed.description,
                    embed=error_embed,
                    mention_author=False,
                )
                return

            embed = _build_preview_embed(
                white_files=png_white_files,
                transparent_files=png_transparent_files,
                pdf_omitted_for_size=result.pdf_omitted_for_size,
                max_bytes=max_bytes,
            )

            await _send_response(
                ctx,
                embed=embed,
                files=files,
                mention_author=False,
            )
        finally:
            # discord.File keeps open handles on Windows; ensure close.
            for f in files:
                try:
                    f.close()
                except Exception:
                    pass


async def setup(bot: commands.Bot):
    await bot.add_cog(Tex(bot))
