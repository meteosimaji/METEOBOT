from __future__ import annotations

import logging

from discord.ext import commands

from music import get_player
from utils import BOT_PREFIX, defer_interaction

log = logging.getLogger(__name__)


class Leave(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="leave",
        description="Leave the current voice channel",
        help=(
            "Disconnect from voice and clear the current music state.\n\n"
            "**Usage**: `/leave`\n"
            f"`{BOT_PREFIX}leave`"
        ),
        extras={
            "category": "Music",
            "pro": "Disconnects from voice and resets the player for this server.",
        },
    )
    async def leave(self, ctx: commands.Context) -> None:
        if ctx.guild is None:
            return await ctx.reply("This command can only be used in a server.", mention_author=False)
        await defer_interaction(ctx)

        player = get_player(self.bot, ctx.guild)
        if not player.voice or not player.voice.is_connected():
            return await ctx.reply("I'm not connected to voice.", mention_author=False)

        try:
            player.queue.clear()
            player.current = None
            player.offset = 0.0
            player.started_at = 0.0
            await player.cleanup()
        except Exception:
            log.exception("Failed to leave voice")
            return await ctx.reply("Failed to disconnect from voice.", mention_author=False)

        await ctx.reply("Left the voice channel.", mention_author=False)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Leave(bot))
