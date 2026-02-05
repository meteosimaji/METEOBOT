import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import discord
from discord.ext import commands

sys.path.append(str(Path(__file__).resolve().parents[1]))

from commands import hack as hack_module  # noqa: E402


class _FakeRole:
    def __init__(self, name: str, admin: bool, members: list[Any]) -> None:
        self.name = name
        self.permissions = SimpleNamespace(administrator=admin)
        self.members = members


def _make_cog() -> hack_module.HackCommand:
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix="!", intents=intents)
    return hack_module.HackCommand(bot)


def test_hack_without_member_shows_empty_holders_message() -> None:
    cog = _make_cog()
    captured: list[tuple[str, bool]] = []

    async def _capture_reply(ctx: Any, content: str, **kwargs: Any) -> None:
        captured.append((content, bool(kwargs.get("ephemeral"))))

    original_safe_reply = hack_module.safe_reply
    hack_module.safe_reply = _capture_reply  # type: ignore[assignment]

    guild = SimpleNamespace(
        id=1,
        roles=[],
    )
    ctx = cast(
        commands.Context[Any],
        SimpleNamespace(
            interaction=object(),
            author=SimpleNamespace(id=hack_module.HACK_OWNER_ID),
            guild=guild,
        ),
    )

    try:
        hack_callback = cast(Any, cog.hack.callback)
        asyncio.run(hack_callback(cog, ctx, None))
    finally:
        hack_module.safe_reply = original_safe_reply  # type: ignore[assignment]

    assert captured == [
        ("No members currently have the protected `/hack` administrator role.", True)
    ]


def test_hack_without_member_lists_current_holders() -> None:
    cog = _make_cog()
    captured: list[tuple[str, bool]] = []

    async def _capture_reply(ctx: Any, content: str, **kwargs: Any) -> None:
        captured.append((content, bool(kwargs.get("ephemeral"))))

    original_safe_reply = hack_module.safe_reply
    hack_module.safe_reply = _capture_reply  # type: ignore[assignment]

    member_a = SimpleNamespace(id=10, mention="<@10>")
    member_b = SimpleNamespace(id=11, mention="<@11>")
    role = _FakeRole(hack_module.HACK_ROLE_NAME, True, [member_a, member_b])
    guild = SimpleNamespace(
        id=1,
        roles=[role],
    )
    ctx = cast(
        commands.Context[Any],
        SimpleNamespace(
            interaction=object(),
            author=SimpleNamespace(id=hack_module.HACK_OWNER_ID),
            guild=guild,
        ),
    )

    try:
        hack_callback = cast(Any, cog.hack.callback)
        asyncio.run(hack_callback(cog, ctx, None))
    finally:
        hack_module.safe_reply = original_safe_reply  # type: ignore[assignment]

    assert len(captured) == 1
    assert captured[0][1] is True
    assert "Current protected `/hack` administrator role holders:" in captured[0][0]
    assert "- <@10> (`10`)" in captured[0][0]
    assert "- <@11> (`11`)" in captured[0][0]
