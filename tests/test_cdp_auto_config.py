import asyncio
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


def test_build_local_cdp_url_formats_ipv4() -> None:
    assert ask_module._build_local_cdp_url(host="127.0.0.1", port=9222) == "http://127.0.0.1:9222"


def test_build_local_cdp_url_formats_ipv6() -> None:
    assert ask_module._build_local_cdp_url(host="::1", port=9222) == "http://[::1]:9222"


def test_operator_ensure_local_cdp_uses_env_url(monkeypatch) -> None:
    monkeypatch.setenv("ASK_BROWSER_CDP_URL", "http://127.0.0.1:9222")
    ask = _make_ask()
    cdp_url, error = asyncio.run(ask._operator_ensure_local_cdp(state_key="1:2"))
    assert cdp_url == "http://127.0.0.1:9222"
    assert error is None


def test_operator_ensure_local_cdp_returns_disabled_when_auto_off(monkeypatch) -> None:
    monkeypatch.delenv("ASK_BROWSER_CDP_URL", raising=False)
    ask = _make_ask()
    monkeypatch.setattr(ask_module, "ASK_BROWSER_CDP_AUTO_CONFIG", False)
    cdp_url, error = asyncio.run(ask._operator_ensure_local_cdp(state_key="1:2"))
    assert cdp_url is None
    assert error == "cdp_auto_config_disabled"


def test_operator_ensure_local_cdp_skips_when_auto_launch_disabled(monkeypatch) -> None:
    monkeypatch.delenv("ASK_BROWSER_CDP_URL", raising=False)
    ask = _make_ask()
    monkeypatch.setattr(ask_module, "ASK_BROWSER_CDP_AUTO_CONFIG", True)
    monkeypatch.setattr(ask_module, "ASK_BROWSER_CDP_AUTO_LAUNCH", False)
    cdp_url, error = asyncio.run(ask._operator_ensure_local_cdp(state_key="1:2"))
    assert cdp_url is None
    assert error == "cdp_auto_launch_disabled"


def test_operator_ensure_local_cdp_rejects_unsafe_host(monkeypatch) -> None:
    monkeypatch.delenv("ASK_BROWSER_CDP_URL", raising=False)
    ask = _make_ask()
    monkeypatch.setattr(ask_module, "ASK_BROWSER_CDP_AUTO_CONFIG", True)
    monkeypatch.setattr(ask_module, "ASK_BROWSER_CDP_AUTO_LAUNCH", True)
    monkeypatch.setattr(ask_module, "ASK_BROWSER_CDP_AUTO_HOST", "0.0.0.0")
    cdp_url, error = asyncio.run(ask._operator_ensure_local_cdp(state_key="1:2"))
    assert cdp_url is None
    assert error == "cdp_auto_host_unsafe"
