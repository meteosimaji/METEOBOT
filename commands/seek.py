from __future__ import annotations

import logging
from typing import Optional

from discord.ext import commands

from music import get_player
from utils import BOT_PREFIX, defer_interaction, ensure_voice, format_timestamp, tag_error_text

log = logging.getLogger(__name__)


def parse_timestamp(ts: str) -> Optional[float]:
    """Parse a timestamp like ``1:23`` and return seconds.

    A leading ``+`` or ``-`` sign is preserved to allow relative seeks.
    Invalid formats return ``None``.
    """
    ts = ts.strip()
    sign = 1
    if ts.startswith("+"):
        ts = ts[1:]
    elif ts.startswith("-"):
        sign = -1
        ts = ts[1:]

    if not ts:
        return None

    parts = ts.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None

    secs = 0
    for p in nums:
        secs = secs * 60 + p
    return float(secs) * sign


class Seek(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="seek",
        description="Jump to a timestamp in the current track",
        help=(
            "Jump to a specific point in the currently playing song.\n"
            "Use `mm:ss` for an absolute position or prefix with `+`/`-` to move relative.\n\n"
            "**Usage**: `/seek <time>`\n"
            "**Examples**: `/seek 1:30`  `/seek +30`\n"
            f"`{BOT_PREFIX}seek -10`"
        ),
        extras={
            "category": "Music",
            "pro": "Seeks within the current track. Relative seeks use + or -.",
        },
    )
    async def seek(self, ctx: commands.Context, *, timestamp: str | None = None) -> None:
        # timestamp is optional to make it LLM-invokable (single optional arg).
        if ctx.guild is None:
            return await ctx.reply(
                tag_error_text("This command can only be used in a server."), mention_author=False
            )
        await defer_interaction(ctx)
        if not await ensure_voice(ctx):
            return

        if not timestamp:
            return await ctx.reply(
                tag_error_text("Give me a timestamp like `1:23` or `+30`."),
                mention_author=False,
            )

        player = get_player(self.bot, ctx.guild)
        if not player.voice or (not player.voice.is_playing() and not player.voice.is_paused()) or not player.current:
            return await ctx.reply(tag_error_text("Nothing is playing."), mention_author=False)

        secs = parse_timestamp(timestamp)
        if secs is None:
            return await ctx.reply(tag_error_text("Invalid timestamp."), mention_author=False)

        relative = timestamp.strip().startswith(("+", "-"))
        target = (player.get_position() + secs) if relative else secs
        if target < 0:
            target = 0.0

        duration = player.current.duration if player.current else 0.0
        if duration and target >= duration:
            await player.skip()
            return await ctx.reply("Skipped to end.", mention_author=False)

        success = await player.seek(target)
        if not success:
            return await ctx.reply(tag_error_text("Failed to seek."), mention_author=False)
        await ctx.reply(f"Seeking to {format_timestamp(target)}", mention_author=False)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Seek(bot))
