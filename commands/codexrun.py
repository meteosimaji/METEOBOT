from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from typing import Any

import discord
from discord.ext import commands

from utils import BOT_PREFIX, defer_interaction, safe_reply, tag_error_embed, tag_error_text

log = logging.getLogger(__name__)

DEFAULT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "files_changed": {"type": "array", "items": {"type": "string"}},
        "tests_run": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["summary", "files_changed"],
    "additionalProperties": True,
}

SAFE_ENV_KEYS = {
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "PATH",
    "PYTHONHOME",
    "PYTHONPATH",
    "SHELL",
    "TMPDIR",
    "USER",
    "VIRTUAL_ENV",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
}
SAFE_ADD_DIR_ROOTS = [
    Path(raw.strip()).resolve()
    for raw in (os.getenv("CODEX_ALLOWED_ROOTS") or "").split(",")
    if raw.strip()
]
ALLOWED_PATH_PREFIXES = [
    raw.strip().strip("/").replace("\\", "/")
    for raw in (os.getenv("CODEX_ALLOWED_PATHS") or "").split(",")
    if raw.strip()
]


def _now_run_id() -> str:
    return secrets.token_hex(8)


def _parse_spec(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    if raw.startswith("{") and raw.endswith("}"):
        return json.loads(raw)
    return {"prompt": raw}


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _build_safe_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in SAFE_ENV_KEYS or key.startswith("CODEX_"):
            env[key] = value
    return env


def _path_is_allowed(path: Path, allowed_paths: list[Path]) -> bool:
    if not allowed_paths:
        return True
    return any(_is_under(path, allowed) for allowed in allowed_paths)


def _relpath_allowed(rel_path: str, allowed_prefixes: Iterable[str]) -> bool:
    if not allowed_prefixes:
        return True
    norm = rel_path.strip().lstrip("./").replace("\\", "/")
    for prefix in allowed_prefixes:
        if norm == prefix:
            return True
        if prefix and norm.startswith(prefix.rstrip("/") + "/"):
            return True
    return False


async def _run_proc(
    args: list[str],
    cwd: Path,
    timeout_s: int,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        with contextlib.suppress(Exception):
            proc.kill()
        return 124, "", f"timeout after {timeout_s}s: {' '.join(args)}"

    out = (out_b or b"").decode("utf-8", "replace")
    err = (err_b or b"").decode("utf-8", "replace")
    return int(proc.returncode or 0), out, err


@dataclass
class CodexConfig:
    codex_bin: str
    repo_root: Path
    runs_root: Path
    sandbox: str
    model: str | None
    timeout_s: int
    keep_worktree: bool

    @staticmethod
    def from_env(repo_root: Path) -> "CodexConfig":
        return CodexConfig(
            codex_bin=os.getenv("CODEX_BIN", "codex"),
            repo_root=repo_root,
            runs_root=Path(os.getenv("CODEX_RUNS_DIR", "data/codex_runs")),
            sandbox=os.getenv("CODEX_SANDBOX", "workspace-write"),
            model=(os.getenv("CODEX_MODEL") or "").strip() or None,
            timeout_s=int(os.getenv("CODEX_TIMEOUT_S", "900")),
            keep_worktree=(os.getenv("CODEX_KEEP_WORKTREE", "false").lower() in {"1", "true", "yes"}),
        )


class CodexRun(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self._lock = asyncio.Lock()

    async def _ensure_owner(self, ctx: commands.Context) -> bool:
        try:
            if await self.bot.is_owner(ctx.author):
                return True
        except Exception:
            pass
        await safe_reply(ctx, "Owner専用コマンドだよ。", ephemeral=True)
        return False

    @commands.hybrid_command(  # type: ignore[arg-type]
        name="codex",
        description="Run Codex CLI in an isolated workspace and return a patch.",
        help=(
            "Run Codex CLI in an isolated git clone and return the diff as an attachment.\n\n"
            "**Usage**: `/codex <prompt or JSON spec>`\n"
            "**Examples**: `/codex Fix the README typos`\n"
            f"`{BOT_PREFIX}codex {{\"prompt\": \"update docs\"}}`"
        ),
        extras={
            "category": "Tools",
            "pro": "Owner-only: runs Codex CLI in a temporary clone and returns the git diff.",
        },
    )
    async def codex(self, ctx: commands.Context, *, spec: str) -> None:
        if not await self._ensure_owner(ctx):
            return

        await defer_interaction(ctx)

        try:
            data = _parse_spec(spec)
        except Exception as exc:
            emb = tag_error_embed(
                discord.Embed(title="Spec parse error", description=str(exc), color=0xFF0000)
            )
            await safe_reply(ctx, embed=emb, ephemeral=True)
            return

        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            await safe_reply(ctx, "promptが空だよ。", ephemeral=True)
            return

        repo = Path(data.get("repo") or ".").resolve()
        cfg = CodexConfig.from_env(repo)

        allowed_roots = (os.getenv("CODEX_ALLOWED_ROOTS") or "").strip()
        if allowed_roots:
            ok = False
            for raw_root in [root.strip() for root in allowed_roots.split(",") if root.strip()]:
                root_path = Path(raw_root).resolve()
                if _is_under(repo, root_path):
                    ok = True
                    break
            if not ok:
                await safe_reply(ctx, "そのrepoパスは許可されてないよ。", ephemeral=True)
                return

        try:
            cfg.runs_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            emb = tag_error_embed(
                discord.Embed(
                    title="Runs dir error",
                    description=f"Failed to create runs directory: {exc}",
                    color=0xFF0000,
                )
            )
            await safe_reply(ctx, embed=emb, ephemeral=True)
            return

        run_id = _now_run_id()
        worktree_dir = cfg.runs_root / f"worktree_{run_id}"
        schema_path = cfg.runs_root / f"schema_{run_id}.json"
        last_msg_path = cfg.runs_root / f"last_{run_id}.json"
        jsonl_path = cfg.runs_root / f"events_{run_id}.jsonl"
        diff_path = cfg.runs_root / f"patch_{run_id}.diff"

        sandbox = str(data.get("sandbox") or cfg.sandbox)
        model = str(data.get("model") or cfg.model or "")
        timeout_s = int(data.get("timeout_s") or cfg.timeout_s)
        keep_worktree = bool(data.get("keep_worktree", cfg.keep_worktree))
        full_auto_raw = data.get("full_auto", None)
        full_auto = True if full_auto_raw is None else bool(full_auto_raw)
        ask_for_approval = str(data.get("ask_for_approval") or "").strip()
        add_dirs = data.get("add_dir") or []
        if isinstance(add_dirs, str):
            add_dirs = [add_dirs]
        if not isinstance(add_dirs, list):
            add_dirs = []

        schema_obj = data.get("output_schema") or DEFAULT_SCHEMA
        try:
            schema_path.write_text(json.dumps(schema_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            emb = tag_error_embed(
                discord.Embed(
                    title="Schema write error",
                    description=f"Failed to write schema file: {exc}",
                    color=0xFF0000,
                )
            )
            await safe_reply(ctx, embed=emb, ephemeral=True)
            return

        env = _build_safe_env()

        async with self._lock:
            if not (repo / ".git").exists():
                await safe_reply(
                    ctx,
                    "repo直下に .git が見つからない。git cloneした作業ツリーでやるのが安全。",
                    ephemeral=True,
                )
                return

            code, _, err = await _run_proc(
                [
                    "git",
                    "clone",
                    "--no-hardlinks",
                    "--no-tags",
                    "--depth",
                    "1",
                    str(repo),
                    str(worktree_dir),
                ],
                cwd=repo,
                timeout_s=60,
                env=env,
            )
            if code != 0:
                emb = tag_error_embed(
                    discord.Embed(title="git clone failed", description=err[:3500], color=0xFF0000)
                )
                await safe_reply(ctx, embed=emb, ephemeral=True)
                return

            try:
                cmd: list[str] = [cfg.codex_bin, "exec", "--json"]
                cmd += ["--output-schema", str(schema_path)]
                cmd += ["--output-last-message", str(last_msg_path)]
                if full_auto:
                    cmd += ["--full-auto"]
                else:
                    cmd += ["--sandbox", sandbox]
                if ask_for_approval:
                    cmd += ["--ask-for-approval", ask_for_approval]
                if model:
                    cmd += ["--model", model]
                for raw_dir in add_dirs:
                    add_dir_path = Path(raw_dir).resolve()
                    if SAFE_ADD_DIR_ROOTS:
                        if not any(_is_under(add_dir_path, root) for root in SAFE_ADD_DIR_ROOTS):
                            await safe_reply(
                                ctx,
                                "add_dir must be under CODEX_ALLOWED_ROOTS.",
                                ephemeral=True,
                            )
                            return
                    elif not _is_under(add_dir_path, repo):
                        await safe_reply(
                            ctx,
                            "add_dir must be under the repo unless CODEX_ALLOWED_ROOTS is set.",
                            ephemeral=True,
                        )
                        return
                    cmd += ["--add-dir", str(add_dir_path)]

                try:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        prompt,
                        cwd=str(worktree_dir),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=env,
                    )
                except FileNotFoundError:
                    await safe_reply(ctx, "codex binary not found. Check CODEX_BIN.", ephemeral=True)
                    return
                except OSError as exc:
                    await safe_reply(ctx, f"Failed to start codex: {exc}", ephemeral=True)
                    return

                try:
                    out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
                except asyncio.TimeoutError:
                    with contextlib.suppress(Exception):
                        proc.kill()
                    out_b, err_b = b"", f"timeout after {timeout_s}s".encode()

                out = (out_b or b"").decode("utf-8", "replace")
                err = (err_b or b"").decode("utf-8", "replace")
                jsonl_path.write_text(out, encoding="utf-8")

                if proc.returncode not in (0, None):
                    emb = tag_error_embed(
                        discord.Embed(
                            title="codex exec failed",
                            description=(err[-3500:] if err else "no stderr"),
                            color=0xFF0000,
                        )
                    )
                    await safe_reply(ctx, embed=emb, ephemeral=True)
                    return

                add_warning = ""
                code, _, add_err = await _run_proc(
                    ["git", "-C", str(worktree_dir), "add", "-N", "."],
                    cwd=worktree_dir,
                    timeout_s=60,
                    env=env,
                )
                if code != 0:
                    add_warning = add_err or "git add -N failed"

                code, diff_out, diff_err = await _run_proc(
                    ["git", "-C", str(worktree_dir), "diff", "--no-color"],
                    cwd=worktree_dir,
                    timeout_s=60,
                    env=env,
                )
                if code != 0:
                    diff_out = ""
                    diff_err = diff_err or "git diff failed"
                if add_warning:
                    diff_err = f"{add_warning}; {diff_err}" if diff_err else add_warning

                code, names_out, _ = await _run_proc(
                    ["git", "-C", str(worktree_dir), "diff", "--name-only"],
                    cwd=worktree_dir,
                    timeout_s=60,
                    env=env,
                )
                if code == 0 and names_out.strip():
                    allowed_paths = [worktree_dir]
                    for line in names_out.splitlines():
                        rel_path = line.strip()
                        if not rel_path:
                            continue
                        if not _relpath_allowed(rel_path, ALLOWED_PATH_PREFIXES):
                            await safe_reply(
                                ctx,
                                tag_error_text(
                                    "Diff touches files outside allowed paths. Aborting."
                                ),
                                ephemeral=True,
                            )
                            return
                        rel_path_obj = Path(rel_path)
                        if rel_path_obj.is_absolute():
                            continue
                        full_path = (worktree_dir / rel_path_obj).resolve()
                        if not _path_is_allowed(full_path, allowed_paths):
                            await safe_reply(
                                ctx,
                                tag_error_text(
                                    "Diff touches files outside allowed paths. Aborting."
                                ),
                                ephemeral=True,
                            )
                            return

                diff_path.write_text(diff_out, encoding="utf-8")

                last_obj: dict[str, Any] = {}
                try:
                    last_obj = json.loads(last_msg_path.read_text(encoding="utf-8"))
                except Exception:
                    last_obj = {}

                summary = (last_obj.get("summary") or "").strip() or "done"
                files_changed = last_obj.get("files_changed")
                if not isinstance(files_changed, list):
                    files_changed = []

                sandbox_label = "full-auto" if full_auto else sandbox
                desc = f"Sandbox: `{sandbox_label}`\n"
                if model:
                    desc += f"Model: `{model}`\n"
                if ask_for_approval:
                    desc += f"Approval: `{ask_for_approval}`\n"
                if files_changed:
                    desc += "Files:\n" + "\n".join(
                        f"- `{item}`" for item in files_changed[:30]
                    ) + ("\n..." if len(files_changed) > 30 else "")
                if diff_err:
                    desc += f"\nDiff warning: {diff_err}"

                emb = discord.Embed(title="Codex run complete", description=desc[:3900], color=0x00AAFF)
                emb.add_field(name="Summary", value=summary[:1024] or "ok", inline=False)

                files: list[discord.File] = []
                if diff_out.strip():
                    files.append(
                        discord.File(
                            fp=io.BytesIO(diff_out.encode("utf-8")),
                            filename=f"patch_{run_id}.diff",
                        )
                    )
                files.append(
                    discord.File(
                        fp=io.BytesIO(
                            json.dumps(last_obj, ensure_ascii=False, indent=2).encode("utf-8")
                        ),
                        filename=f"last_{run_id}.json",
                    )
                )
                files.append(
                    discord.File(
                        fp=io.BytesIO(out.encode("utf-8")),
                        filename=f"events_{run_id}.jsonl",
                    )
                )

                if ctx.interaction:
                    await ctx.interaction.followup.send(embed=emb, files=files)
                else:
                    await ctx.reply(embed=emb, files=files, mention_author=False)
            finally:
                if not keep_worktree:
                    with contextlib.suppress(Exception):
                        shutil.rmtree(worktree_dir)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(CodexRun(bot))
