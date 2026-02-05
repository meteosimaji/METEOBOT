import os
import sys
import asyncio
from pathlib import Path

import discord
from discord.ext import commands


sys.path.append(str(Path(__file__).resolve().parents[1]))

from commands import ask as ask_module  # noqa: E402


def _make_ask() -> ask_module.Ask:
    os.environ.setdefault("OPENAI_TOKEN", "test-token")
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix="!", intents=intents)
    return ask_module.Ask(bot)


def test_state_store_persists_and_loads(tmp_path: Path) -> None:
    ask = _make_ask()
    ask._ask_state_store_path = tmp_path / "ask_conversations.json"
    ask.bot.ai_last_response_id = {"1:10": "resp_A"}  # type: ignore[attr-defined]
    ask._ask_state_links = {"1:20": "1:10"}

    ask._save_ask_state_store()

    reloaded = _make_ask()
    reloaded._ask_state_store_path = ask._ask_state_store_path
    reloaded._load_ask_state_store()

    assert reloaded.bot.ai_last_response_id == {"1:10": "resp_A"}  # type: ignore[attr-defined]
    assert reloaded._ask_state_links == {"1:20": "1:10"}


def test_state_store_corruption_auto_resets(tmp_path: Path) -> None:
    ask = _make_ask()
    ask._ask_state_store_path = tmp_path / "ask_conversations.json"
    ask._ask_state_store_path.write_text("{ broken json", encoding="utf-8")

    ask._load_ask_state_store()

    assert ask.bot.ai_last_response_id == {}  # type: ignore[attr-defined]
    assert ask._ask_state_links == {}
    assert ask._ask_state_store_path.exists()
    backups = list(tmp_path.glob("ask_conversations.corrupt-*.json"))
    assert backups, "Corrupted state file should be backed up"


def test_clear_response_state_keeps_links(tmp_path: Path) -> None:
    ask = _make_ask()
    ask._ask_state_store_path = tmp_path / "ask_conversations.json"
    ask.bot.ai_last_response_id = {"1:10": "resp_A"}  # type: ignore[attr-defined]
    ask._ask_state_links = {"1:20": "1:10"}

    removed = ask._clear_response_state("1:20")

    assert removed is True
    assert ask.bot.ai_last_response_id == {}  # type: ignore[attr-defined]
    assert ask._ask_state_links == {"1:20": "1:10"}

    reloaded = _make_ask()
    reloaded._ask_state_store_path = ask._ask_state_store_path
    reloaded._load_ask_state_store()
    assert reloaded._ask_state_links == {"1:20": "1:10"}


def test_parse_memory_control_text() -> None:
    parse = ask_module.Ask._parse_memory_control_text
    assert parse("ask link #123") == ("link", "#123")
    assert parse("ask share 456") == ("link", "456")
    assert parse("ask unlink") == ("unlink", None)
    assert parse("ask unshare") == ("unlink", None)
    assert parse("unlink") == ("unlink", None)
    assert parse("unshare") == ("unlink", None)
    assert parse("link #123") == ("link", "#123")
    assert parse("share 123") == ("link", "123")
    assert parse("ask about link behavior") == ("", None)


def test_link_confirm_view_buttons() -> None:
    async def _build_labels() -> list[str]:
        view = ask_module._LinkConfirmView(author_id=1)
        return [getattr(child, "label", "") for child in view.children]

    labels = asyncio.run(_build_labels())
    assert "Link" in labels
    assert "Cancel" in labels
