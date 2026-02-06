from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import re
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable
from urllib.parse import urlparse

import aiohttp
import discord
from discord.ext import commands

from utils import BOT_PREFIX, defer_interaction, safe_reply, tag_error_embed, tag_error_text

log = logging.getLogger(__name__)

DEFAULT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "files_changed": {"type": "array", "items": {"type": "string"}},
        "tests_run": {"type": ["array", "null"], "items": {"type": "string"}},
        "notes": {"type": ["string", "null"]},
    },
    "required": ["summary", "files_changed", "tests_run", "notes"],
    "additionalProperties": False,
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


def _copy_repo_to_worktree(repo: Path, worktree_dir: Path, runs_root: Path) -> None:
    ignore_names = [".git", "cookies.json", "cookies.txt"]
    if _is_under(runs_root, repo):
        rel_runs_root = runs_root.resolve().relative_to(repo.resolve())
        if rel_runs_root.parts:
            ignore_names.append(rel_runs_root.parts[0])
    worktree_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        repo,
        worktree_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(*ignore_names),
    )


def _normalize_no_git_diff(
    diff_text: str,
    snapshot_name: str,
    worktree_name: str,
) -> str:
    snapshot_prefix = f"{snapshot_name}/"
    worktree_prefix = f"{worktree_name}/"
    normalized: list[str] = []
    for line in diff_text.splitlines():
        updated = line
        if line.startswith(("diff --git ", "--- ", "+++ ")):
            updated = (
                updated.replace(f"a/{snapshot_prefix}", "a/")
                .replace(f"b/{worktree_prefix}", "b/")
            )
        elif line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            updated = updated.replace(snapshot_prefix, "").replace(worktree_prefix, "")
        elif line.startswith("Binary files "):
            updated = (
                updated.replace(f"a/{snapshot_prefix}", "a/")
                .replace(f"b/{worktree_prefix}", "b/")
                .replace(snapshot_prefix, "")
                .replace(worktree_prefix, "")
            )
        normalized.append(updated)
    if diff_text.endswith("\n"):
        return "\n".join(normalized) + "\n"
    return "\n".join(normalized)


def _parse_github_repo(value: str) -> tuple[str, str] | None:
    raw = (value or "").strip()
    if not raw:
        return None
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", raw):
        owner, repo = raw.split("/", 1)
        return owner, repo
    parsed = urlparse(raw)
    if not parsed.netloc:
        return None
    host = parsed.netloc.lower()
    if not host.endswith("github.com"):
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None
    owner = parts[0]
    repo = parts[1]
    if repo.endswith(".git"):
        repo = repo[: -len(".git")]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner):
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
        return None
    return owner, repo


def _github_remote_url(owner: str, repo: str) -> str:
    return f"https://github.com/{owner}/{repo}.git"


def _get_github_token() -> str:
    return (os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or "").strip()


async def _github_api_json(
    method: str,
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "METEOBOT-codexrun",
    }
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.request(method, url, json=payload) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"GitHub API {resp.status}: {text[:2000]}")
            if not text.strip():
                return {}
            return await resp.json()


def _make_askpass_script(path: Path) -> None:
    content = """#!/bin/sh
case "$1" in
  *Username*)
    echo "x-access-token"
    ;;
  *Password*)
    echo "${GITHUB_TOKEN}"
    ;;
  *)
    echo ""
    ;;
esac
"""
    path.write_text(content, encoding="utf-8")
    try:
        os.chmod(path, 0o700)
    except Exception:
        pass


async def _git(
    args: list[str],
    cwd: Path,
    timeout_s: int = 300,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    return await _run_proc(["git", *args], cwd=cwd, timeout_s=timeout_s, env=env)


async def _create_github_pr_from_worktree(
    *,
    worktree_dir: Path,
    owner: str,
    repo: str,
    token: str,
    base_branch: str | None,
    title: str,
    body: str,
    draft: bool,
    run_id: str,
    env_base: dict[str, str],
) -> str:
    askpass = worktree_dir / f".git_askpass_{run_id}.sh"
    _make_askpass_script(askpass)
    env_git = dict(env_base)
    env_git.update(
        {
            "GITHUB_TOKEN": token,
            "GIT_ASKPASS": str(askpass),
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    remote_url = _github_remote_url(owner, repo)
    code, out, err = await _git(["remote"], cwd=worktree_dir, env=env_git)
    if code != 0:
        raise RuntimeError(err or "git remote failed")
    remotes = {line.strip() for line in out.splitlines() if line.strip()}
    if "origin" in remotes:
        await _git(["remote", "set-url", "origin", remote_url], cwd=worktree_dir, env=env_git)
    else:
        await _git(["remote", "add", "origin", remote_url], cwd=worktree_dir, env=env_git)
    if not base_branch:
        try:
            info = await _github_api_json(
                "GET",
                f"https://api.github.com/repos/{owner}/{repo}",
                token,
                payload=None,
            )
            base_branch = str(info.get("default_branch") or "").strip() or "main"
        except Exception:
            base_branch = "main"
    await _git(["fetch", "--depth", "1", "origin", base_branch], cwd=worktree_dir, env=env_git)
    branch = f"codex/{run_id}"
    code, _, err = await _git(
        ["checkout", "-B", branch, f"origin/{base_branch}"],
        cwd=worktree_dir,
        env=env_git,
    )
    if code != 0:
        raise RuntimeError(err or "git checkout failed")
    code, _, err = await _git(["add", "-A"], cwd=worktree_dir, env=env_git)
    if code != 0:
        raise RuntimeError(err or "git add failed")
    code, status_out, _ = await _git(["status", "--porcelain"], cwd=worktree_dir, env=env_git)
    if code != 0:
        raise RuntimeError("git status failed")
    if not status_out.strip():
        raise RuntimeError("no changes to commit")
    code, _, err = await _git(
        [
            "-c",
            "user.name=METEOBOT Codex",
            "-c",
            "user.email=codex@users.noreply.github.com",
            "commit",
            "-m",
            title,
        ],
        cwd=worktree_dir,
        env=env_git,
    )
    if code != 0:
        raise RuntimeError(err or "git commit failed")
    code, _, err = await _git(
        ["push", "-u", "origin", branch],
        cwd=worktree_dir,
        env=env_git,
        timeout_s=300,
    )
    if code != 0:
        raise RuntimeError(err or "git push failed")
    pr = await _github_api_json(
        "POST",
        f"https://api.github.com/repos/{owner}/{repo}/pulls",
        token,
        payload={
            "title": title,
            "head": branch,
            "base": base_branch,
            "body": body,
            "draft": bool(draft),
        },
    )
    pr_url = str(pr.get("html_url") or "").strip()
    if not pr_url:
        raise RuntimeError("PR created but URL missing")
    return pr_url


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
        await safe_reply(ctx, "This command is owner-only.", ephemeral=True)
        return False

    @commands.hybrid_command(  # type: ignore[arg-type]
        name="codex",
        description="Run Codex CLI in an isolated workspace and return a patch.",
        help=(
            "Run Codex CLI in an isolated workspace and return the diff as an attachment. "
            "Optionally opens a GitHub PR when configured.\n\n"
            "**Usage**: `/codex <prompt or JSON spec>`\n"
            "**Examples**: `/codex Fix the README typos`\n"
            f"`{BOT_PREFIX}codex {{\"prompt\": \"update docs\"}}`"
        ),
        extras={
            "category": "Tools",
            "pro": (
                "Owner-only: runs Codex CLI in a temporary workspace, returns the git diff, "
                "and can open a GitHub PR when configured."
            ),
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
            await safe_reply(ctx, "The prompt is empty.", ephemeral=True)
            return

        repo = Path(data.get("repo") or ".").resolve()
        cfg = CodexConfig.from_env(repo)

        try:
            runs_root = cfg.runs_root
            if not runs_root.is_absolute():
                runs_root = (repo / runs_root).resolve()
            cfg.runs_root = runs_root
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

        allowed_roots = (os.getenv("CODEX_ALLOWED_ROOTS") or "").strip()
        if allowed_roots:
            ok = False
            for raw_root in [root.strip() for root in allowed_roots.split(",") if root.strip()]:
                root_path = Path(raw_root).resolve()
                if _is_under(repo, root_path):
                    ok = True
                    break
            if not ok:
                await safe_reply(ctx, "That repo path is not allowed.", ephemeral=True)
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
        github_repo_raw = str(data.get("github_repo") or os.getenv("CODEX_GITHUB_REPO") or "").strip()
        github_repo = _parse_github_repo(github_repo_raw) if github_repo_raw else None
        github_pr_enabled = bool(data.get("github_pr", False))
        github_base = str(data.get("github_base") or os.getenv("CODEX_GITHUB_BASE") or "").strip() or None
        github_draft = bool(data.get("github_draft", False))

        async with self._lock:
            local_has_git = (repo / ".git").exists()
            work_has_git = False
            snapshot_dir: Path | None = None
            if local_has_git:
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
                work_has_git = True
            elif github_repo:
                owner, repo_name = github_repo
                code, _, err = await _run_proc(
                    [
                        "git",
                        "clone",
                        "--no-tags",
                        "--depth",
                        "1",
                        _github_remote_url(owner, repo_name),
                        str(worktree_dir),
                    ],
                    cwd=cfg.runs_root,
                    timeout_s=120,
                    env=env,
                )
                if code != 0:
                    emb = tag_error_embed(
                        discord.Embed(
                            title="github clone failed",
                            description=(err or "git clone failed")[:3500],
                            color=0xFF0000,
                        )
                    )
                    await safe_reply(ctx, embed=emb, ephemeral=True)
                    return
                work_has_git = True
            else:
                snapshot_dir = cfg.runs_root / f"snapshot_{run_id}"
                try:
                    _copy_repo_to_worktree(repo, worktree_dir, cfg.runs_root)
                    _copy_repo_to_worktree(repo, snapshot_dir, cfg.runs_root)
                except Exception as exc:
                    emb = tag_error_embed(
                        discord.Embed(
                            title="workspace copy failed",
                            description=str(exc),
                            color=0xFF0000,
                        )
                    )
                    await safe_reply(ctx, embed=emb, ephemeral=True)
                    return

            try:
                cmd: list[str] = [cfg.codex_bin, "exec", "--json"]
                cmd += ["--output-schema", str(schema_path)]
                cmd += ["--output-last-message", str(last_msg_path)]
                if not work_has_git:
                    cmd += ["--skip-git-repo-check"]
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
                diff_args: list[str]
                diff_name_args: list[str]
                diff_cwd = worktree_dir
                name_status = False
                if work_has_git:
                    code, _, add_err = await _run_proc(
                        ["git", "-C", str(worktree_dir), "add", "-N", "."],
                        cwd=worktree_dir,
                        timeout_s=60,
                        env=env,
                    )
                    if code != 0:
                        add_warning = add_err or "git add -N failed"

                    diff_args = ["git", "-C", str(worktree_dir), "diff", "--no-color"]
                    diff_name_args = [
                        "git",
                        "-C",
                        str(worktree_dir),
                        "diff",
                        "--name-only",
                    ]
                else:
                    if snapshot_dir is None:
                        await safe_reply(
                            ctx,
                            tag_error_text("Snapshot directory is missing."),
                            ephemeral=True,
                        )
                        return
                    diff_cwd = cfg.runs_root
                    name_status = True
                    diff_args = [
                        "git",
                        "diff",
                        "--no-index",
                        "--relative",
                        "--no-color",
                        snapshot_dir.name,
                        worktree_dir.name,
                    ]
                    diff_name_args = [
                        "git",
                        "diff",
                        "--no-index",
                        "--name-status",
                        "--relative",
                        snapshot_dir.name,
                        worktree_dir.name,
                    ]

                code, diff_out, diff_err = await _run_proc(
                    diff_args,
                    cwd=diff_cwd,
                    timeout_s=60,
                    env=env,
                )
                ok_codes = (0,) if work_has_git else (0, 1)
                if code not in ok_codes:
                    diff_out = ""
                    diff_err = diff_err or "git diff failed"
                elif diff_out and not work_has_git and snapshot_dir:
                    diff_out = _normalize_no_git_diff(
                        diff_out,
                        snapshot_name=snapshot_dir.name,
                        worktree_name=worktree_dir.name,
                    )
                if add_warning:
                    diff_err = f"{add_warning}; {diff_err}" if diff_err else add_warning

                code, names_out, _ = await _run_proc(
                    diff_name_args,
                    cwd=diff_cwd,
                    timeout_s=60,
                    env=env,
                )
                if code in ok_codes and names_out.strip():
                    allowed_paths = [worktree_dir]
                    for line in names_out.splitlines():
                        rel_path = line.strip()
                        if not rel_path:
                            continue
                        if name_status:
                            parts = [part for part in rel_path.split("\t") if part]
                            rel_paths = [part for part in parts[1:] if part != "/dev/null"]
                        else:
                            rel_paths = [rel_path]
                        for rel_entry in rel_paths:
                            rel_path_obj = Path(rel_entry)
                            if snapshot_dir:
                                for prefix in (snapshot_dir.name + "/", worktree_dir.name + "/"):
                                    if rel_entry.startswith(prefix):
                                        rel_entry = rel_entry[len(prefix) :]
                                        rel_path_obj = Path(rel_entry)
                                        break
                            if rel_path_obj.is_absolute():
                                await safe_reply(
                                    ctx,
                                    tag_error_text("Absolute paths in diff are not allowed."),
                                    ephemeral=True,
                                )
                                return
                            if not _relpath_allowed(rel_entry, ALLOWED_PATH_PREFIXES):
                                await safe_reply(
                                    ctx,
                                    tag_error_text(
                                        "Diff touches files outside allowed paths. Aborting."
                                    ),
                                    ephemeral=True,
                                )
                                return
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

                pr_url = ""
                if github_pr_enabled:
                    token = _get_github_token()
                    if not token:
                        pr_url = "(github_pr requested but GITHUB_TOKEN/GH_TOKEN is not set)"
                    elif not work_has_git:
                        pr_url = (
                            "(github_pr requested but workspace is not a git repo; "
                            "set github_repo to enable cloning)"
                        )
                    else:
                        gh = github_repo
                        if not gh:
                            try:
                                code, url_out, _ = await _git(
                                    ["remote", "get-url", "origin"],
                                    cwd=worktree_dir,
                                    env=env,
                                )
                                if code == 0:
                                    gh = _parse_github_repo(url_out.strip())
                            except Exception:
                                gh = None
                        if not gh:
                            pr_url = "(github_pr requested but github_repo is missing)"
                        else:
                            owner, repo_name = gh
                            pr_title = str(data.get("github_title") or "").strip() or f"codex: {summary}"
                            pr_body = str(data.get("github_body") or "").strip()
                            if not pr_body:
                                tests_run = last_obj.get("tests_run")
                                if not isinstance(tests_run, list):
                                    tests_run = []
                                notes = (last_obj.get("notes") or "").strip()
                                pr_body_lines = [
                                    f"Prompt: {prompt}",
                                    "",
                                    f"Summary: {summary}",
                                ]
                                if files_changed:
                                    pr_body_lines += [
                                        "",
                                        "Files:",
                                        *[f"- {item}" for item in files_changed[:80]],
                                    ]
                                if tests_run:
                                    pr_body_lines += [
                                        "",
                                        "Tests:",
                                        *[f"- {item}" for item in tests_run[:80]],
                                    ]
                                if notes:
                                    pr_body_lines += ["", "Notes:", notes]
                                pr_body = "\n".join(pr_body_lines)
                            try:
                                pr_url = await _create_github_pr_from_worktree(
                                    worktree_dir=worktree_dir,
                                    owner=owner,
                                    repo=repo_name,
                                    token=token,
                                    base_branch=github_base,
                                    title=pr_title,
                                    body=pr_body,
                                    draft=github_draft,
                                    run_id=run_id,
                                    env_base=env,
                                )
                            except Exception as exc:
                                pr_url = f"(failed to create PR: {exc})"

                sandbox_label = "full-auto" if full_auto else sandbox
                desc = f"Sandbox: `{sandbox_label}`\n"
                if model:
                    desc += f"Model: `{model}`\n"
                if ask_for_approval:
                    desc += f"Approval: `{ask_for_approval}`\n"
                desc += f"Repo: `{repo}`\n"
                if github_repo_raw:
                    desc += f"GitHub: `{github_repo_raw}`\n"
                if pr_url:
                    desc += f"PR: {pr_url}\n"
                if files_changed:
                    desc += "Files:\n" + "\n".join(
                        f"- `{item}`" for item in files_changed[:30]
                    ) + ("\n..." if len(files_changed) > 30 else "")
                if diff_err:
                    desc += f"\n\nWarning: {diff_err[:1200]}"

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
                    if snapshot_dir:
                        with contextlib.suppress(Exception):
                            shutil.rmtree(snapshot_dir)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(CodexRun(bot))
