import os
import sys
import asyncio
from pathlib import Path
from types import SimpleNamespace

import discord
from discord.ext import commands


sys.path.append(str(Path(__file__).resolve().parents[1]))

from commands import ask as ask_module  # noqa: E402


def _make_ask() -> ask_module.Ask:
    os.environ.setdefault("OPENAI_TOKEN", "test-token")
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix="!", intents=intents)
    return ask_module.Ask(bot)


def test_collect_container_file_links_includes_existing_container_files(tmp_path: Path) -> None:
    async def _run() -> None:
        ask = _make_ask()
        ask._repo_root = tmp_path
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        ask._iter_container_file_citations = lambda _outputs: [
            {
                "container_id": "cntr_1",
                "file_id": "file_new",
                "filename": "new.gif",
                "path": "/mnt/data/new.gif",
            }
        ]
        ask._extract_container_ids = lambda _outputs: ["cntr_1"]

        async def _list_container_files_page(container_id: str, *, after: str | None = None, limit: int = 100) -> tuple[list[dict[str, object]], str | None, bool]:
            assert container_id == "cntr_1"
            assert after is None
            _ = limit
            return (
                [
                    {
                        "id": "file_old",
                        "path": "/mnt/data/old.gif",
                        "source": "assistant",
                        "bytes": 128,
                    },
                    {
                        "id": "file_new",
                        "path": "/mnt/data/new.gif",
                        "source": "assistant",
                        "bytes": 256,
                    },
                ],
                None,
                False,
            )

        ask._list_container_files_page = _list_container_files_page

        async def _download_container_file(*, container_id: str, file_id: str, dest_path: Path) -> tuple[int, str]:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(file_id.encode("utf-8"))
            return len(file_id), "image/gif"

        ask._download_container_file = _download_container_file

        async def _register_download(path: Path, *, filename: str, expires_s: int, keep_file: bool) -> str:
            assert path.exists()
            assert expires_s > 0
            assert keep_file is True
            return f"https://example.test/{filename}"

        ask.register_download = _register_download
        ask._load_ask_workspace_manifest = lambda _workspace_dir: {}
        ask._write_ask_workspace_manifest = lambda _workspace_dir, _manifest: None
        ask._load_link_context = lambda _ctx: []
        ask._prune_link_context = lambda entries: entries
        ask._write_link_context = lambda _ctx, _entries: None

        entries, notes = await ask._collect_container_file_links(
            ctx=SimpleNamespace(),
            workspace_dir=workspace_dir,
            outputs=[],
        )

        assert notes == []
        assert [entry["filename"] for entry in entries] == ["new.gif", "old.gif"]

    asyncio.run(_run())


def test_collect_container_file_links_uses_fallback_container_id(tmp_path: Path) -> None:
    async def _run() -> None:
        ask = _make_ask()
        ask._repo_root = tmp_path
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        ask._iter_container_file_citations = lambda _outputs: []
        ask._extract_container_ids = lambda _outputs: []

        seen_container_ids: list[str] = []

        async def _list_container_files_page(container_id: str, *, after: str | None = None, limit: int = 100) -> tuple[list[dict[str, object]], str | None, bool]:
            seen_container_ids.append(container_id)
            assert after is None
            _ = limit
            return (
                [
                    {
                        "id": "file_existing",
                        "path": "/mnt/data/existing.txt",
                        "source": "assistant",
                        "bytes": 10,
                    }
                ],
                None,
                False,
            )

        ask._list_container_files_page = _list_container_files_page

        async def _download_container_file(*, container_id: str, file_id: str, dest_path: Path) -> tuple[int, str]:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(file_id.encode("utf-8"))
            return 10, "text/plain"

        ask._download_container_file = _download_container_file
        ask.register_download = (
            lambda path, *, filename, expires_s, keep_file: asyncio.sleep(
                0,
                result=f"https://example.test/{filename}",
            )
        )
        ask._load_ask_workspace_manifest = lambda _workspace_dir: {}
        ask._write_ask_workspace_manifest = lambda _workspace_dir, _manifest: None
        ask._load_link_context = lambda _ctx: []
        ask._prune_link_context = lambda entries: entries
        ask._write_link_context = lambda _ctx, _entries: None

        entries, notes = await ask._collect_container_file_links(
            ctx=SimpleNamespace(),
            workspace_dir=workspace_dir,
            outputs=[],
            fallback_container_id="cntr_fallback",
        )

        assert notes == []
        assert seen_container_ids == ["cntr_fallback"]
        assert len(entries) == 1
        assert entries[0]["filename"] == "existing.txt"

    asyncio.run(_run())


def test_collect_container_file_links_prioritizes_placeholder_matches_over_limit(tmp_path: Path) -> None:
    async def _run() -> None:
        ask = _make_ask()
        ask._repo_root = tmp_path
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        ask._iter_container_file_citations = lambda _outputs: []
        ask._extract_container_ids = lambda _outputs: ["cntr_1"]

        async def _list_container_files_page(_container_id: str, *, after: str | None = None, limit: int = 100) -> tuple[list[dict[str, object]], str | None, bool]:
            assert after is None
            _ = limit
            return (
                [
                    {"id": "f1", "path": "/mnt/data/a.txt", "source": "assistant", "bytes": 10},
                    {"id": "f2", "path": "/mnt/data/b.txt", "source": "assistant", "bytes": 10},
                    {"id": "f3", "path": "/mnt/data/c.txt", "source": "assistant", "bytes": 10},
                    {"id": "f4", "path": "/mnt/data/d.txt", "source": "assistant", "bytes": 10},
                    {"id": "f5", "path": "/mnt/data/e.txt", "source": "assistant", "bytes": 10},
                    {"id": "f6", "path": "/mnt/data/target.gif", "source": "assistant_output", "bytes": 10},
                ],
                None,
                False,
            )

        ask._list_container_files_page = _list_container_files_page

        async def _download_container_file(*, container_id: str, file_id: str, dest_path: Path) -> tuple[int, str]:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(file_id.encode("utf-8"))
            return 10, "application/octet-stream"

        ask._download_container_file = _download_container_file
        ask.register_download = (
            lambda path, *, filename, expires_s, keep_file: asyncio.sleep(
                0,
                result=f"https://example.test/{filename}",
            )
        )
        ask._load_ask_workspace_manifest = lambda _workspace_dir: {}
        ask._write_ask_workspace_manifest = lambda _workspace_dir, _manifest: None
        ask._load_link_context = lambda _ctx: []
        ask._prune_link_context = lambda entries: entries
        ask._write_link_context = lambda _ctx, _entries: None

        original_max = ask_module.ASK_CONTAINER_FILE_MAX_COUNT
        ask_module.ASK_CONTAINER_FILE_MAX_COUNT = 5
        try:
            entries, notes = await ask._collect_container_file_links(
                ctx=SimpleNamespace(),
                workspace_dir=workspace_dir,
                outputs=[],
                preferred_link_keys={"/mnt/data/target.gif"},
            )
        finally:
            ask_module.ASK_CONTAINER_FILE_MAX_COUNT = original_max

        assert any(entry["filename"] == "target.gif" for entry in entries)
        assert any("Only the first" in note for note in notes)

    asyncio.run(_run())


def test_collect_container_file_links_finds_placeholder_in_paginated_results(tmp_path: Path) -> None:
    async def _run() -> None:
        ask = _make_ask()
        ask._repo_root = tmp_path
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        ask._iter_container_file_citations = lambda _outputs: []
        ask._extract_container_ids = lambda _outputs: ["cntr_1"]

        calls: list[str | None] = []

        async def _list_container_files_page(
            _container_id: str,
            *,
            after: str | None = None,
            limit: int = 100,
        ) -> tuple[list[dict[str, object]], str | None, bool]:
            _ = limit
            calls.append(after)
            if after is None:
                return (
                    [
                        {"id": "f1", "path": "/mnt/data/a.txt", "bytes": 10},
                        {"id": "f2", "path": "/mnt/data/b.txt", "bytes": 10},
                    ],
                    "f2",
                    True,
                )
            return (
                [
                    {"id": "f3", "path": "/mnt/data/brownian_motion.gif", "bytes": 10},
                ],
                None,
                False,
            )

        ask._list_container_files_page = _list_container_files_page

        async def _download_container_file(*, container_id: str, file_id: str, dest_path: Path) -> tuple[int, str]:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(file_id.encode("utf-8"))
            return 10, "application/octet-stream"

        ask._download_container_file = _download_container_file
        ask.register_download = (
            lambda path, *, filename, expires_s, keep_file: asyncio.sleep(
                0,
                result=f"https://example.test/{filename}",
            )
        )
        ask._load_ask_workspace_manifest = lambda _workspace_dir: {}
        ask._write_ask_workspace_manifest = lambda _workspace_dir, _manifest: None
        ask._load_link_context = lambda _ctx: []
        ask._prune_link_context = lambda entries: entries
        ask._write_link_context = lambda _ctx, _entries: None

        entries, notes = await ask._collect_container_file_links(
            ctx=SimpleNamespace(),
            workspace_dir=workspace_dir,
            outputs=[],
            preferred_link_keys={"/mnt/data/brownian_motion.gif"},
        )

        assert calls == [None, "f2"]
        assert any(entry["filename"] == "brownian_motion.gif" for entry in entries)
        assert notes == []

    asyncio.run(_run())
