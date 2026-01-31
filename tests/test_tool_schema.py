import os
import sys
from pathlib import Path

import discord
from discord.ext import commands


sys.path.append(str(Path(__file__).resolve().parents[1]))

from commands import ask as ask_module  # noqa: E402


def test_browser_tool_schema_strict() -> None:
    tools = ask_module.Ask._build_browser_tools()
    issues = ask_module._validate_strict_tool_schemas(tools)
    assert issues == []


def test_bot_tool_schema_strict() -> None:
    os.environ.setdefault("OPENAI_TOKEN", "test-token")
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix="!", intents=intents)

    @bot.command()
    async def ping(ctx: commands.Context) -> None:
        return None

    ask = ask_module.Ask(bot)
    tools = [*ask._build_bot_tools(), *ask._build_browser_tools()]
    issues = ask_module._validate_strict_tool_schemas(tools)
    assert issues == []
