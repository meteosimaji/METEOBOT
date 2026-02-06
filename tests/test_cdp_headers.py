import json
import os
import sys
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


def test_parse_cdp_headers_env_ignores_invalid_json(monkeypatch) -> None:
    monkeypatch.setenv("ASK_BROWSER_CDP_HEADERS_JSON", "{bad")
    assert ask_module._parse_cdp_headers_env() == {}


def test_parse_cdp_headers_env_accepts_string_map(monkeypatch) -> None:
    monkeypatch.setenv(
        "ASK_BROWSER_CDP_HEADERS_JSON",
        json.dumps({"x-openclaw-relay-token": "abc", "X-Test": "1", "n": 2}),
    )
    headers = ask_module._parse_cdp_headers_env()
    assert headers == {"x-openclaw-relay-token": "abc", "X-Test": "1"}


def test_ask_loads_cdp_headers(monkeypatch) -> None:
    monkeypatch.setenv("ASK_BROWSER_CDP_HEADERS_JSON", json.dumps({"X-Test": "token"}))
    ask = _make_ask()
    assert ask._cdp_headers == {"X-Test": "token"}
