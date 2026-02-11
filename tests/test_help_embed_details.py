import sys
from pathlib import Path

import discord
from discord.ext import commands

sys.path.append(str(Path(__file__).resolve().parents[1]))

from commands.help import make_command_help_embed  # noqa: E402


def test_make_command_help_embed_contains_detailed_fields() -> None:
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix="!", intents=intents)

    @commands.command(
        name="demohelp",
        description="Demo description",
        help=(
            "Demo long help.\\n\\n"
            "**Usage**: `/demohelp <target>`\\n"
            "**Examples**: `/demohelp ping`\\n"
            "`!demohelp ping`"
        ),
        usage="<target>",
        extras={
            "category": "Utility",
            "destination": "Demo destination",
            "plus": "Demo plus",
            "pro": "Demo pro",
        },
    )
    async def demohelp(ctx: commands.Context, target: str, note: str = "") -> None:
        return None

    bot.add_command(demohelp)

    embed = make_command_help_embed(bot, "demohelp")
    assert embed is not None

    fields = {field.name: field.value for field in embed.fields}
    assert fields["Destination"] == "Demo destination"
    assert fields["Plus"] == "Demo plus"
    assert fields["Pro"] == "Demo pro"
    assert fields["Usage"] == "/demohelp <target>"
    assert "target (string, required)" in fields["Arg (bot_invoke)"]
    assert "note (string, optional)" in fields["Arg (bot_invoke)"]
    assert "`/demohelp ping`" in fields["Examples"]
    assert "`!demohelp ping`" in fields["Examples"]
