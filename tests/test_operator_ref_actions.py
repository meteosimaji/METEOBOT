import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import discord
from aiohttp import web
from discord.ext import commands

sys.path.append(str(Path(__file__).resolve().parents[1]))

from commands import ask as ask_module  # noqa: E402


def _make_ask() -> ask_module.Ask:
    os.environ.setdefault("OPENAI_TOKEN", "test-token")
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix="!", intents=intents)
    return ask_module.Ask(bot)


class _DummyRequest:
    def __init__(self, token: str, payload: dict[str, object]) -> None:
        self.match_info = {"token": token}
        self._payload = payload

    async def json(self) -> dict[str, object]:
        return self._payload


class _DummyAgent:
    class _DummyPage:
        viewport_size = {"width": 1200, "height": 800}

        async def evaluate(self, _script: str) -> dict[str, int]:
            return {"width": 1200, "height": 800}

    def __init__(self) -> None:
        self.page = self._DummyPage()

    def is_started(self) -> bool:
        return True

    async def act(self, action: dict[str, object]) -> dict[str, object]:
        return {"ok": True, "observation": {"url": "https://example.com", "title": "Example"}}


def _parse_response_json(response: web.Response) -> dict[str, object]:
    body = response.body.decode("utf-8") if response.body else "{}"
    return json.loads(body)


def test_operator_allows_click_ref_actions() -> None:
    ask = _make_ask()
    ask._get_operator_session = lambda token: SimpleNamespace(state_key="1:2", owner_id=123)  # type: ignore[method-assign]
    ask._get_browser_lock_for_state_key = lambda state_key: asyncio.Lock()  # type: ignore[method-assign]
    ask._ensure_operator_browser_started = (  # type: ignore[method-assign]
        lambda **kwargs: asyncio.sleep(0, result=(_DummyAgent(), None))
    )

    request = _DummyRequest(
        "token",
        {"action": {"type": "click_ref", "ref": "e12", "ref_generation": 7}},
    )

    response = asyncio.run(ask._operator_handle_action(request))
    data = _parse_response_json(response)

    assert response.status == 200
    assert data["ok"] is True


def test_operator_rejects_missing_ref_generation() -> None:
    ask = _make_ask()
    ask._get_operator_session = lambda token: SimpleNamespace(state_key="1:2", owner_id=123)  # type: ignore[method-assign]

    request = _DummyRequest(
        "token",
        {"action": {"type": "click_ref", "ref": "e12"}},
    )

    response = asyncio.run(ask._operator_handle_action(request))
    data = _parse_response_json(response)

    assert response.status == 400
    assert data["error"] == "missing_ref_generation"


def test_operator_observation_includes_ref_and_size_metadata() -> None:
    ask = _make_ask()

    class _ObsAgent(_DummyAgent):
        async def observe(self):
            return SimpleNamespace(
                url="https://example.com",
                title="Example",
                ref_generation=3,
                refs=[{"ref": "e1", "bbox": {"x": 1, "y": 2, "width": 10, "height": 20}}],
            )

    data = asyncio.run(ask._operator_observation(_ObsAgent()))
    assert data["ref_generation"] == 3
    assert isinstance(data["refs"], list)
    assert data["viewport_css"]["width"] == 1200
    assert data["screenshot_px"]["height"] == 800
