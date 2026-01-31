from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
import tempfile
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import discord
from discord.ext import commands
import yt_dlp

from utils import ASK_ERROR_TAG, BOT_PREFIX, defer_interaction, humanize_delta, safe_reply, tag_error_text

log = logging.getLogger(__name__)

# ---- defaults (tuned for Discord attachment limits) ----

# Safety cap. Actual per-guild cap is ctx.guild.filesize_limit.
DEFAULT_MAX_BYTES = 200 * 1024 * 1024  # 200MiB
DEFAULT_TIMEOUT_S = 120
DEFAULT_MAX_CONCURRENT = 2
DEFAULT_LOG_TAIL = 60
DEFAULT_WORK_ROOT = "data/savevideo"
DEFAULT_DNS_TIMEOUT_S = 2
DEFAULT_ERROR_TEXT_MAX = 1900
DISCORD_CONTENT_LIMIT = 4000
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

ERROR_PATTERNS: dict[str, re.Pattern[str]] = {
    "unsupported_url": re.compile(r"(unsupported\s+url|no\s+video\s+formats)", re.I),
    "forbidden": re.compile(r"(403|forbidden|access\s+denied)", re.I),
    "rate_limited": re.compile(r"(429|rate\s*limit|too\s+many\s+requests)", re.I),
    "private": re.compile(r"(private|unavailable|not\s+available|deleted)", re.I),
    "login_required": re.compile(r"(login\s*required|sign\s*in|confirm\s+your\s+age)", re.I),
    "geo_restricted": re.compile(r"(not\s+available\s+in\s+your\s+country|geo\s*restricted)", re.I),
    "ffmpeg_missing": re.compile(r"(ffmpeg|avconv).*(not\s+found|not\s+installed)", re.I),
    "max_filesize": re.compile(r"(max(filesize)?|file\s+is\s+too\s+large|max\-filesize)", re.I),
}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _env_str(name: str) -> str | None:
    raw = os.getenv(name)
    raw = raw.strip() if isinstance(raw, str) else None
    return raw or None


def _parse_cookies_from_browser(
    spec: str | None,
) -> tuple[str] | tuple[str, str | None, str | None, str | None] | None:
    """Parse CLI-like cookies-from-browser into the tuple expected by yt-dlp."""
    spec = (spec or "").strip()
    if not spec:
        return None

    parts = spec.split(":")
    head = parts[0].strip()
    profile = parts[1].strip() if len(parts) >= 2 and parts[1].strip() else None
    container = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else None

    if "+" in head:
        browser, keyring = head.split("+", 1)
        browser = browser.strip()
        keyring = keyring.strip() or None
    else:
        browser, keyring = head.strip(), None

    if profile is None and keyring is None and container is None:
        return (browser,)

    if container is not None and keyring is None:
        return (browser, profile, None, container)

    if keyring is not None and container is None:
        return (browser, profile, keyring)

    if keyring is not None and container is not None:
        return (browser, profile, keyring, container)

    return (browser, profile, None, None)


def _is_blocked_ip_literal(host: str) -> bool:
    """Block obvious SSRF targets (localhost, RFC1918, link-local, etc.) for IP literals."""

    try:
        ip = ip_address(host)
    except ValueError:
        return host.lower() in {"localhost", "ip6-localhost"}

    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
    )


async def _resolve_host_ips_async(host: str) -> set[str]:
    addrs: set[str] = set()
    infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    for family, _type, _proto, _canon, sockaddr in infos:
        if family == socket.AF_INET:
            addrs.add(sockaddr[0])
        elif family == socket.AF_INET6:
            addrs.add(sockaddr[0])
    return addrs


async def _is_safe_direct_url_async(url: str, *, dns_timeout_s: int) -> bool:
    """Allow only public http(s) targets."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in {"http", "https"}:
        return False

    host = parsed.hostname
    if not host:
        return False

    if _is_blocked_ip_literal(host):
        return False

    try:
        ips = await asyncio.wait_for(_resolve_host_ips_async(host), timeout=float(dns_timeout_s))
    except Exception:
        return False

    for ip_s in ips:
        if _is_blocked_ip_literal(ip_s):
            return False

    return True


def _pick_downloaded_file(download_dir: Path) -> Path | None:
    files = [path for path in download_dir.iterdir() if path.is_file()]
    if not files:
        return None
    # prefer mp4 if present, else the largest file.
    mp4s = [path for path in files if path.suffix.lower() == ".mp4"]
    candidates = mp4s or files
    return max(candidates, key=lambda path: path.stat().st_size)


class YTDLLogger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def debug(self, message: str) -> None:
        self._append(message)

    def info(self, message: str) -> None:
        self._append(message)

    def warning(self, message: str) -> None:
        self._append(message)

    def error(self, message: str) -> None:
        self._append(message)

    def _append(self, message: str) -> None:
        if message:
            self.lines.append(str(message))

    def tail(self, count: int) -> str:
        if count <= 0:
            return ""
        return "\n".join(self.lines[-count:])


@dataclass(frozen=True)
class SaveConfig:
    max_bytes: int
    timeout_s: int
    max_concurrent: int
    log_tail_lines: int
    work_root: Path
    dns_timeout_s: int
    user_agent: str
    cookies_file: str | None
    cookies_from_browser: tuple[str] | tuple[str, str | None, str | None, str | None] | None
    impersonate: str | None

    @staticmethod
    def from_env() -> "SaveConfig":
        return SaveConfig(
            max_bytes=_env_int("SAVEVIDEO_MAX_BYTES", DEFAULT_MAX_BYTES),
            timeout_s=_env_int("SAVEVIDEO_TIMEOUT_S", DEFAULT_TIMEOUT_S),
            max_concurrent=_env_int("SAVEVIDEO_MAX_CONCURRENT", DEFAULT_MAX_CONCURRENT),
            log_tail_lines=_env_int("SAVEVIDEO_LOG_TAIL_LINES", DEFAULT_LOG_TAIL),
            work_root=Path(os.getenv("SAVEVIDEO_WORK_DIR", DEFAULT_WORK_ROOT)),
            dns_timeout_s=_env_int("SAVEVIDEO_DNS_TIMEOUT_S", DEFAULT_DNS_TIMEOUT_S),
            user_agent=os.getenv("SAVEVIDEO_USER_AGENT", DEFAULT_USER_AGENT),
            cookies_file=_env_str("SAVEVIDEO_COOKIES_FILE"),
            cookies_from_browser=_parse_cookies_from_browser(
                _env_str("SAVEVIDEO_COOKIES_FROM_BROWSER")
            ),
            impersonate=_env_str("SAVEVIDEO_IMPERSONATE"),
        )


def _human_size(value: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return f"{size:.1f}GB"


def _truncate_text(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    suffix = "...(truncated)"
    keep = max(max_len - len(suffix), 0)
    return f"{text[:keep]}{suffix}"


def _assemble_error_message(*, base: str, tail: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    tail = tail.strip()
    if not tail:
        return _truncate_text(base, max_len)

    tail_prefix = "\n```text\n"
    tail_suffix = "\n```"
    trunc_suffix = "\n...(truncated)"

    headroom = max_len - len(base) - len(tail_prefix) - len(tail_suffix)
    if headroom <= 0:
        return _truncate_text(base, max_len)

    if len(tail) <= headroom:
        return f"{base}{tail_prefix}{tail}{tail_suffix}"

    trimmed_tail = tail[: max(headroom - len(trunc_suffix), 0)]
    return f"{base}{tail_prefix}{trimmed_tail}{trunc_suffix}{tail_suffix}"


def _cap_error_message(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""

    tagged = tag_error_text(text)
    if len(tagged) <= max_len:
        return tagged

    cleaned = tagged.replace(ASK_ERROR_TAG, "").rstrip()
    suffix = f"\n{ASK_ERROR_TAG}"
    available = max_len - len(suffix)
    if available <= 0:
        return ASK_ERROR_TAG[:max_len]

    truncated = _truncate_text(cleaned, available)
    return f"{truncated}{suffix}"


def _extra_headers(url: str, user_agent: str) -> dict[str, str]:
    headers = {"User-Agent": user_agent}
    host = urlparse(url).hostname or ""
    if "tiktok.com" in host:
        # cobaltもここを強めにやってる。Refererが無いと弾かれることがある。
        headers["Referer"] = "https://www.tiktok.com/"
    if host.endswith("x.com") or host.endswith("twitter.com"):
        headers["Referer"] = "https://x.com/"
    return headers


def _base_ydl_opts(
    *,
    logger: YTDLLogger,
    work_dir: Path,
    max_bytes: int,
    headers: dict[str, str],
    cookies_file: str | None,
    cookies_from_browser: tuple[str] | tuple[str, str | None, str | None, str | None] | None,
    socket_timeout: int,
    impersonate: str | None,
) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "noplaylist": True,
        "restrictfilenames": True,
        "logger": logger,
        "max_filesize": max_bytes,
        "merge_output_format": "mp4",
        "http_headers": headers,
        "outtmpl": {"default": str(work_dir / "%(title).200B-%(id)s.%(ext)s")},
        # ffmpegがあるなら勝手に良い感じにマルチスレ化されるが、断片DLの並列数を少し上げる。
        "concurrent_fragment_downloads": 4,
        "socket_timeout": socket_timeout,
        "retries": 3,
        "fragment_retries": 3,
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file
    if cookies_from_browser:
        # e.g. "chrome", "firefox" (yt-dlpの仕様)
        opts["cookiesfrombrowser"] = cookies_from_browser
    if impersonate:
        opts["impersonate"] = impersonate
    return opts


def _probe_info(
    url: str,
    logger: YTDLLogger,
    headers: dict[str, str],
    cookies_file: str | None,
    cookies_from_browser: tuple[str] | tuple[str, str | None, str | None, str | None] | None,
    impersonate: str | None,
) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "noplaylist": True,
        "logger": logger,
        "skip_download": True,
        "http_headers": headers,
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = cookies_from_browser
    if impersonate:
        opts["impersonate"] = impersonate
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if not isinstance(info, dict):
        raise RuntimeError("yt-dlp returned unexpected info")
    return info


def _estimate_format_size(fmt: dict[str, Any], duration_s: float | None) -> int | None:
    size = fmt.get("filesize") or fmt.get("filesize_approx")
    if isinstance(size, (int, float)) and size > 0:
        return int(size)

    # bitrate (tbr) is in Kbps
    tbr = fmt.get("tbr")
    if duration_s and isinstance(tbr, (int, float)) and tbr > 0:
        return int(float(duration_s) * float(tbr) * 1000.0 / 8.0)

    # fallback: vbr/abr
    vbr = fmt.get("vbr")
    abr = fmt.get("abr")
    if duration_s and isinstance(vbr, (int, float)) and isinstance(abr, (int, float)):
        br = float(vbr) + float(abr)
        if br > 0:
            return int(float(duration_s) * br * 1000.0 / 8.0)

    return None


def _is_mp4_h264ish_video(fmt: dict[str, Any]) -> bool:
    if (fmt.get("vcodec") or "none") == "none":
        return False
    ext = (fmt.get("ext") or "").lower()
    if ext != "mp4":
        return False
    vcodec = (fmt.get("vcodec") or "").lower()
    # avc1 (H.264) が最強に互換性高い。hevc (hvc1/hev1) も一応OK。
    return (
        vcodec.startswith("avc1")
        or ("h264" in vcodec)
        or vcodec.startswith("hvc1")
        or vcodec.startswith("hev1")
    )


def _is_m4a_aacish_audio(fmt: dict[str, Any]) -> bool:
    if (fmt.get("acodec") or "none") == "none":
        return False
    if (fmt.get("vcodec") or "none") != "none":
        return False
    ext = (fmt.get("ext") or "").lower()
    if ext not in {"m4a", "mp4"}:
        return False
    acodec = (fmt.get("acodec") or "").lower()
    return acodec.startswith("mp4a") or ("aac" in acodec)


def _iter_mp4_candidates(
    info: dict[str, Any],
    *,
    max_bytes: int,
    max_height: int | None,
) -> Iterable[tuple[str, dict[str, Any], dict[str, Any] | None, int | None]]:
    """Yield candidates as (format_spec, vfmt, afmt, est_size).

    - format_spec is either "id" (muxed) or "vid+aud".
    - vfmt is the chosen video format dict (for muxed it is the whole format).
    - afmt is None for muxed, otherwise audio format dict.
    - est_size may be None if we can't estimate.
    """

    duration = info.get("duration")
    duration_s = float(duration) if isinstance(duration, (int, float)) and duration > 0 else None

    formats: list[dict[str, Any]] = [f for f in (info.get("formats") or []) if isinstance(f, dict)]

    # Partition formats.
    muxed: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    audios: list[dict[str, Any]] = []

    for f in formats:
        # Ignore story/unknown formats.
        if not f.get("format_id"):
            continue

        if _is_mp4_h264ish_video(f) and (f.get("acodec") or "none") != "none":
            # muxed mp4 with audio
            muxed.append(f)
        elif _is_mp4_h264ish_video(f):
            videos.append(f)
        elif _is_m4a_aacish_audio(f):
            audios.append(f)

    # Apply max_height filter.
    if max_height is not None:
        muxed = [f for f in muxed if (f.get("height") or 0) <= max_height]
        videos = [f for f in videos if (f.get("height") or 0) <= max_height]

    # Sort with a bias toward higher res / higher bitrate.
    def _vkey(f: dict[str, Any]) -> tuple[int, float]:
        return (int(f.get("height") or 0), float(f.get("tbr") or 0.0))

    def _akey(f: dict[str, Any]) -> float:
        return float(f.get("abr") or f.get("tbr") or 0.0)

    muxed.sort(key=_vkey, reverse=True)
    videos.sort(key=_vkey, reverse=True)
    audios.sort(key=_akey, reverse=True)

    # Limit combinatorics.
    videos = videos[:30]
    audios = audios[:10]

    # 1) muxed candidates
    for f in muxed:
        est = _estimate_format_size(f, duration_s)
        yield (str(f["format_id"]), f, None, est)

    # 2) video+audio candidates
    for v in videos:
        for a in audios:
            est_v = _estimate_format_size(v, duration_s)
            est_a = _estimate_format_size(a, duration_s)
            est = est_v + est_a if isinstance(est_v, int) and isinstance(est_a, int) else None
            yield (f"{v['format_id']}+{a['format_id']}", v, a, est)


def _rank_candidate(vfmt: dict[str, Any], afmt: dict[str, Any] | None, est_size: int | None) -> tuple:
    """Higher is better."""
    height = int(vfmt.get("height") or 0)
    fps = float(vfmt.get("fps") or 0.0)
    tbr = float(vfmt.get("tbr") or 0.0)

    # codec preference: avc1 > h264-other > hevc
    vcodec = (vfmt.get("vcodec") or "").lower()
    codec_score = 0
    if vcodec.startswith("avc1"):
        codec_score = 3
    elif "h264" in vcodec:
        codec_score = 2
    elif vcodec.startswith("hvc1") or vcodec.startswith("hev1"):
        codec_score = 1

    audio_score = 0
    if afmt is None:
        acodec = (vfmt.get("acodec") or "").lower()
    else:
        acodec = (afmt.get("acodec") or "").lower()
    if acodec.startswith("mp4a") or "aac" in acodec:
        audio_score = 1

    # Prefer known-under-limit sizes; unknown sizes get a penalty.
    size_known = 1 if isinstance(est_size, int) else 0

    # Bigger est_size is usually better quality *if* still under limit, but we don't want to hug the limit.
    # We'll use a small nudge rather than a main axis.
    size_nudge = int(est_size or 0)

    return (codec_score, audio_score, size_known, height, fps, tbr, size_nudge)


def _pick_best_formats(
    info: dict[str, Any],
    *,
    max_bytes: int,
    max_height: int | None,
) -> list[tuple[str, dict[str, Any], dict[str, Any] | None, int | None]]:
    """Return candidates in best-first order.

    We only keep candidates whose *estimated* size is under max_bytes (with a small safety margin).
    Unknown-size candidates are kept as a last resort.
    """

    safety = int(max_bytes * 0.97)

    in_limit: list[tuple[str, dict[str, Any], dict[str, Any] | None, int | None]] = []
    unknown: list[tuple[str, dict[str, Any], dict[str, Any] | None, int | None]] = []

    seen: set[str] = set()
    for spec, vfmt, afmt, est in _iter_mp4_candidates(info, max_bytes=max_bytes, max_height=max_height):
        if spec in seen:
            continue
        seen.add(spec)
        if isinstance(est, int):
            if est <= safety:
                in_limit.append((spec, vfmt, afmt, est))
        else:
            unknown.append((spec, vfmt, afmt, est))

    in_limit.sort(key=lambda t: _rank_candidate(t[1], t[2], t[3]), reverse=True)
    unknown.sort(key=lambda t: _rank_candidate(t[1], t[2], t[3]), reverse=True)

    # Try known-good candidates first, then unknown.
    return in_limit + unknown


def _download_with_spec(
    url: str,
    *,
    format_spec: str,
    max_bytes: int,
    logger: YTDLLogger,
    work_dir: Path,
    headers: dict[str, str],
    cookies_file: str | None,
    cookies_from_browser: tuple[str] | tuple[str, str | None, str | None, str | None] | None,
    socket_timeout: int,
    impersonate: str | None,
) -> tuple[dict[str, Any], Path]:
    opts = _base_ydl_opts(
        logger=logger,
        work_dir=work_dir,
        max_bytes=max_bytes,
        headers=headers,
        cookies_file=cookies_file,
        cookies_from_browser=cookies_from_browser,
        socket_timeout=socket_timeout,
        impersonate=impersonate,
    )
    opts["format"] = format_spec

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    if not isinstance(info, dict):
        raise RuntimeError("yt-dlp returned unexpected info")

    chosen = _pick_downloaded_file(work_dir)
    if chosen is None:
        raise RuntimeError("Download completed but no files were written")

    return info, chosen


def _classify_error(message: str) -> str:
    for key, pattern in ERROR_PATTERNS.items():
        if pattern.search(message):
            return key
    return "unknown"


class Save(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.cfg = SaveConfig.from_env()
        self._sem = asyncio.Semaphore(self.cfg.max_concurrent)

    @commands.hybrid_command(
        name="save",
        description="Save a video from a public URL and attach it as mp4.",
        help=(
            "Download a video from a public URL (TikTok, YouTube, etc.) with yt-dlp "
            "and attach it to Discord. The bot auto-selects an mp4-compatible format "
            "that fits the guild's upload limit.\n\n"
            "**Usage**: `/save <url>`\n\n"
            "Optional: `/save <url> max_height:<int>`\n\n"
            "Env knobs: SAVEVIDEO_MAX_BYTES, SAVEVIDEO_TIMEOUT_S, SAVEVIDEO_MAX_CONCURRENT, "
            "SAVEVIDEO_LOG_TAIL_LINES, SAVEVIDEO_WORK_DIR, SAVEVIDEO_DNS_TIMEOUT_S, "
            "SAVEVIDEO_USER_AGENT, SAVEVIDEO_COOKIES_FILE, SAVEVIDEO_COOKIES_FROM_BROWSER, "
            "SAVEVIDEO_IMPERSONATE\n\n"
            f"Prefix: `{BOT_PREFIX}save <url>`"
        ),
        extras={
            "category": "Tools",
            "pro": (
                "Fetches a single video with yt-dlp, blocks private/localhost targets, "
                "and auto-fits the result under the Discord upload limit."
            ),
        },
    )
    async def save(self, ctx: commands.Context, url: str, max_height: int | None = None) -> None:
        url = (url or "").strip()
        if not url:
            await safe_reply(
                ctx,
                tag_error_text("Provide a video URL to save."),
                mention_author=False,
                ephemeral=True,
            )
            return

        # SSRF hardening: resolve hostnames and block private ranges.
        ok = await _is_safe_direct_url_async(url, dns_timeout_s=self.cfg.dns_timeout_s)

        if not ok:
            await safe_reply(
                ctx,
                tag_error_text("That URL is not allowed. Use a public http/https link."),
                mention_author=False,
                ephemeral=True,
            )
            return

        # slash/interaction response
        await defer_interaction(ctx)

        guild = ctx.guild
        # discord.py exposes the *effective* cap based on boost tier.
        max_upload_bytes = int(getattr(ctx, "filesize_limit", 0) or 0)
        if max_upload_bytes <= 0:
            max_upload_bytes = self.cfg.max_bytes

        max_bytes = min(max_upload_bytes, self.cfg.max_bytes)

        async with self._sem:
            logger = YTDLLogger()
            headers = _extra_headers(url, self.cfg.user_agent)
            host = urlparse(url).hostname or ""
            impersonate = self.cfg.impersonate
            if impersonate is None and (
                "tiktok.com" in host or host.endswith("x.com") or host.endswith("twitter.com")
            ):
                impersonate = "chrome"

            try:
                self.cfg.work_root.mkdir(parents=True, exist_ok=True)

                with tempfile.TemporaryDirectory(prefix="savevideo_", dir=self.cfg.work_root) as tmp_dir:
                    tmp_path = Path(tmp_dir)

                    info = await asyncio.wait_for(
                        asyncio.to_thread(
                            _probe_info,
                            url,
                            logger,
                            headers,
                            self.cfg.cookies_file,
                            self.cfg.cookies_from_browser,
                            impersonate,
                        ),
                        timeout=float(self.cfg.timeout_s),
                    )

                    # yt-dlp can still return a playlist-like object in some edge cases.
                    if isinstance(info.get("entries"), list):
                        await safe_reply(
                            ctx,
                            tag_error_text(
                                "This URL resolves to multiple items (playlist/carousel). "
                                "This command currently supports a single video."
                            ),
                            mention_author=False,
                            ephemeral=True,
                        )
                        return

                    candidates = _pick_best_formats(info, max_bytes=max_bytes, max_height=max_height)
                    if not candidates:
                        raise RuntimeError("No mp4-compatible formats found")

                    last_error: Exception | None = None
                    download_info: dict[str, Any] | None = None
                    download_path: Path | None = None
                    chosen_spec: str | None = None
                    chosen_est: int | None = None

                    # Try a few best candidates first.
                    for spec, vfmt, afmt, est in candidates[:8]:
                        try:
                            chosen_spec = spec
                            chosen_est = est if isinstance(est, int) else None
                            download_info, download_path = await asyncio.wait_for(
                                asyncio.to_thread(
                                    _download_with_spec,
                                    url,
                                    format_spec=spec,
                                    max_bytes=max_bytes,
                                    logger=logger,
                                    work_dir=tmp_path,
                                    headers=headers,
                                    cookies_file=self.cfg.cookies_file,
                                    cookies_from_browser=self.cfg.cookies_from_browser,
                                    socket_timeout=max(10, int(self.cfg.timeout_s)),
                                    impersonate=impersonate,
                                ),
                                timeout=float(self.cfg.timeout_s),
                            )
                            break
                        except Exception as exc:
                            last_error = exc
                            log.debug("save download failed for %s: %s", spec, exc)

                    if download_path is None or download_info is None:
                        raise last_error or RuntimeError("Download failed")

                    file_size = download_path.stat().st_size
                    if file_size > max_upload_bytes:
                        raise RuntimeError("max-filesize: exceeds Discord upload limit")

                    title = str(download_info.get("title") or info.get("title") or "Saved video")
                    duration = download_info.get("duration", info.get("duration"))
                    duration_text = humanize_delta(duration) if isinstance(duration, (int, float)) else "?"
                    res_width = download_info.get("width")
                    res_height = download_info.get("height")
                    resolution = (
                        f"{res_width}x{res_height}"
                        if isinstance(res_width, int) and isinstance(res_height, int)
                        else "unknown"
                    )
                    webpage_url = download_info.get("webpage_url") or info.get("webpage_url") or url

                    embed = discord.Embed(
                        title="💾 Video saved",
                        description=f"[{title}]({webpage_url})",
                        color=0x2ECC71,
                    )
                    embed.add_field(name="Duration", value=duration_text, inline=True)
                    embed.add_field(name="Resolution", value=resolution, inline=True)
                    embed.add_field(name="Size", value=_human_size(file_size), inline=True)
                    if chosen_spec:
                        embed.add_field(name="Format", value=chosen_spec, inline=False)
                    if chosen_est is not None:
                        embed.add_field(name="Est. size", value=_human_size(chosen_est), inline=True)
                    embed.add_field(name="Discord limit", value=_human_size(max_upload_bytes), inline=True)

                    file_obj = discord.File(download_path, filename="video.mp4")
                    await safe_reply(ctx, embed=embed, file=file_obj, mention_author=False)

            except yt_dlp.utils.DownloadError as exc:
                await self._handle_error(ctx, url, exc, logger, max_upload_bytes)
            except asyncio.TimeoutError:
                await safe_reply(
                    ctx,
                    tag_error_text(
                        f"Download timed out after {self.cfg.timeout_s}s. "
                        "Try a shorter clip, or increase SAVEVIDEO_TIMEOUT_S."
                    ),
                    mention_author=False,
                    ephemeral=True,
                )
            except Exception as exc:
                await self._handle_error(ctx, url, exc, logger, max_upload_bytes)

    async def _handle_error(
        self,
        ctx: commands.Context,
        url: str,
        exc: Exception,
        logger: YTDLLogger,
        max_upload_bytes: int,
    ) -> None:
        raw_error = f"{exc}"
        kind = _classify_error(raw_error)

        reason = {
            "unsupported_url": "Unsupported URL or no compatible formats.",
            "forbidden": "Access denied (403). The site may block bots.",
            "rate_limited": "Rate limited (429). Wait a bit and retry.",
            "private": "The video looks private, deleted, or unavailable.",
            "login_required": "Login is required (or age/anti-bot verification).",
            "geo_restricted": "Geo-restricted. The content is not available from this region.",
            "ffmpeg_missing": "ffmpeg is missing on the host. Install ffmpeg to merge streams.",
            "max_filesize": f"The video exceeds the Discord upload limit ({_human_size(max_upload_bytes)}).",
            "unknown": "Extraction failed. The site may require cookies or a newer yt-dlp.",
        }.get(kind, "Extraction failed.")

        tail = logger.tail(self.cfg.log_tail_lines)
        base = f"{reason}\nURL: {url}\nRaw error: {_truncate_text(raw_error, 500)}"
        message = _assemble_error_message(
            base=base,
            tail=tail,
            max_len=DEFAULT_ERROR_TEXT_MAX,
        )

        await safe_reply(
            ctx,
            _cap_error_message(message, DISCORD_CONTENT_LIMIT),
            mention_author=False,
            ephemeral=True,
        )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Save(bot))
