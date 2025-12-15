from __future__ import annotations

import logging

from discord.ext import commands

from music import get_player
from utils import defer_interaction, ensure_voice, BOT_PREFIX

log = logging.getLogger(__name__)


class Pause(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="pause",
        description="Pause the current track",
        help=(
            "Pause playback of the current song. You must be in the same voice channel as the bot.\n\n"
            "**Usage**: `/pause`\n"
            f"`{BOT_PREFIX}pause`"
        ),
        extras={
            "category": "Music",
            "pro": "Pauses playback and keeps your place so you can /resume later.",
        },
    )
    async def pause(self, ctx: commands.Context) -> None:
        if ctx.guild is None:
            return await ctx.reply("This command can only be used in a server.", mention_author=False)
        await defer_interaction(ctx)
        if not await ensure_voice(ctx):
            return

        player = get_player(self.bot, ctx.guild)
        if not player.voice or (not player.voice.is_playing() and not player.voice.is_paused()):
            return await ctx.reply("Nothing is playing.", mention_author=False)
        if player.voice.is_paused():
            return await ctx.reply("Playback is already paused.", mention_author=False)

        await player.pause()
        log.info("%s paused playback", ctx.author)
        await ctx.reply("Paused.", mention_author=False)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Pause(bot))
