import logging
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


def test_build_tools_for_ask_includes_all_current_tools() -> None:
    os.environ.setdefault("OPENAI_TOKEN", "test-token")
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix="!", intents=intents)

    @bot.command()
    async def ping(ctx: commands.Context) -> None:
        return None

    ask = ask_module.Ask(bot)
    tools = ask._build_tools_for_action(
        action="ask",
        tool_profile="lite",
        container_config={"type": "auto", "memory_limit": "4g"},
    )

    signatures = [ask._tool_signature(tool) for tool in tools]
    assert len(signatures) == len(set(signatures))

    builtin_types = {str(tool.get("type") or "") for tool in tools if str(tool.get("type") or "") != "function"}
    assert {"web_search", "code_interpreter", "shell"}.issubset(builtin_types)

    function_names = {
        str(tool.get("name") or "")
        for tool in tools
        if str(tool.get("type") or "") == "function"
    }
    assert {
        "bot_commands",
        "bot_invoke",
        "discord_fetch_message",
        "discord_list_attachments",
        "discord_read_attachment",
        "browser",
    }.issubset(function_names)


def test_build_tools_for_ask_auto_includes_new_builder_tools() -> None:
    os.environ.setdefault("OPENAI_TOKEN", "test-token")
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix="!", intents=intents)

    @bot.command()
    async def ping(ctx: commands.Context) -> None:
        return None

    class _AskWithExtraTool(ask_module.Ask):
        @staticmethod
        def _build_extra_tools() -> list[dict[str, object]]:
            return [
                {
                    "type": "function",
                    "name": "extra_tool",
                    "description": "extra tool",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                }
            ]

    ask = _AskWithExtraTool(bot)
    tools = ask._build_tools_for_action(
        action="ask",
        tool_profile="lite",
        container_config={"type": "auto", "memory_limit": "4g"},
    )
    function_names = {
        str(tool.get("name") or "")
        for tool in tools
        if str(tool.get("type") or "") == "function"
    }
    assert "extra_tool" in function_names


def test_declared_tool_builder_errors_are_logged(caplog) -> None:
    os.environ.setdefault("OPENAI_TOKEN", "test-token")
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix="!", intents=intents)

    class _AskWithBrokenBuilder(ask_module.Ask):
        @staticmethod
        def _build_broken_tools() -> list[dict[str, object]]:
            raise RuntimeError("boom")

    ask = _AskWithBrokenBuilder(bot)
    with caplog.at_level(logging.WARNING):
        tools = ask._build_declared_function_tools()

    assert isinstance(tools, list)
    assert "Ask tool builder '_build_broken_tools' failed." in caplog.text


def test_missing_tools_for_action_are_logged(caplog) -> None:
    os.environ.setdefault("OPENAI_TOKEN", "test-token")
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix="!", intents=intents)
    ask = ask_module.Ask(bot)

    expected_tools = [{"type": "function", "name": "x"}]
    assembled_tools: list[dict[str, object]] = []
    with caplog.at_level(logging.ERROR):
        ask._log_missing_tools_for_action(
            action="ask",
            tool_profile="lite",
            expected_tools=expected_tools,
            assembled_tools=assembled_tools,
        )

    assert "Missing tools detected for action=ask profile=lite" in caplog.text
