import os
import sys
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import discord
from discord.ext import commands

sys.path.append(str(Path(__file__).resolve().parents[1]))

from commands import ask as ask_module  # noqa: E402


def _make_ask() -> ask_module.Ask:
    os.environ.setdefault("OPENAI_TOKEN", "test-token")
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix="!", intents=intents)
    return ask_module.Ask(bot)


def _make_ctx() -> SimpleNamespace:
    return SimpleNamespace(
        guild=SimpleNamespace(id=1),
        channel=SimpleNamespace(id=2),
        author=SimpleNamespace(guild_permissions=SimpleNamespace(administrator=True)),
        interaction=None,
        message=None,
    )


def _register_moderation_commands(bot: commands.Bot) -> None:
    @commands.command(name="hack", extras={"category": "Moderation"})
    async def hack_cmd(ctx: commands.Context, *, arg: str = "") -> None:
        return None

    @commands.command(name="unhack", extras={"category": "Moderation"})
    async def unhack_cmd(ctx: commands.Context, *, arg: str = "") -> None:
        return None

    bot.add_command(hack_cmd)
    bot.add_command(unhack_cmd)


def test_bot_invoke_blocks_hack_and_unhack() -> None:
    ask = _make_ask()
    _register_moderation_commands(ask.bot)

    async def _always_can_run(ctx: commands.Context, command: commands.Command) -> tuple[bool, str]:
        return True, ""

    ask._can_run_command = _always_can_run  # type: ignore[method-assign]

    async def _run() -> None:
        ctx = cast(commands.Context[Any], _make_ctx())
        hack_result = await ask._function_router(ctx, "bot_invoke", {"name": "hack", "arg": "@user"})
        unhack_result = await ask._function_router(ctx, "bot_invoke", {"name": "unhack", "arg": ""})

        assert isinstance(hack_result, dict)
        assert isinstance(unhack_result, dict)

        assert hack_result["ok"] is False
        assert hack_result["error"] == "restricted_for_llm"
        assert unhack_result["ok"] is False
        assert unhack_result["error"] == "restricted_for_llm"

    asyncio.run(_run())


def test_bot_commands_marks_hack_and_unhack_as_llm_blocked() -> None:
    ask = _make_ask()
    _register_moderation_commands(ask.bot)

    async def _always_can_run(ctx: commands.Context, command: commands.Command) -> tuple[bool, str]:
        return True, ""

    ask._can_run_command = _always_can_run  # type: ignore[method-assign]

    async def _run() -> None:
        ctx = cast(commands.Context[Any], _make_ctx())
        hack_info = await ask._function_router(ctx, "bot_commands", {"name": "hack"})
        unhack_info = await ask._function_router(ctx, "bot_commands", {"name": "unhack"})

        assert isinstance(hack_info, dict)
        assert isinstance(unhack_info, dict)

        assert hack_info["llm_can_invoke"] is False
        assert hack_info["llm_blocked_reason"] == "blocked_command"
        assert unhack_info["llm_can_invoke"] is False
        assert unhack_info["llm_blocked_reason"] == "blocked_command"

    asyncio.run(_run())
