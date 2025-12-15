from __future__ import annotations

import logging

from discord.ext import commands

from music import get_player
from utils import defer_interaction, ensure_voice, BOT_PREFIX

log = logging.getLogger(__name__)


class Stop(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="stop",
        description="Stop playback and clear the queue",
        help=(
            "Stop playing and clear the queue, leaving the channel if auto-leave is enabled.\n\n"
            "**Usage**: `/stop`\n"
            f"`{BOT_PREFIX}stop`"
        ),
        extras={
            "category": "Music",
            "pro": "Stops playback, clears queue, and disconnects if auto-leave is enabled.",
        },
    )
    async def stop(self, ctx: commands.Context) -> None:
        if ctx.guild is None:
            return await ctx.reply("This command can only be used in a server.", mention_author=False)
        await defer_interaction(ctx)
        if not await ensure_voice(ctx):
            return

        player = get_player(self.bot, ctx.guild)
        if not player.voice or (not player.voice.is_playing() and not player.voice.is_paused()):
            return await ctx.reply("Nothing is playing.", mention_author=False)
        await player.stop()
        log.info("%s stopped playback", ctx.author)
        await ctx.reply("Stopped and cleared queue.", mention_author=False)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Stop(bot))
