from __future__ import annotations

import logging

from discord.ext import commands

from music import get_player
from utils import defer_interaction, ensure_voice, BOT_PREFIX

log = logging.getLogger(__name__)


class Skip(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="skip",
        description="Skip the current track",
        help=(
            "Skip over the song that's currently playing. You need to share a voice channel with the bot.\n\n"
            "**Usage**: `/skip`\n"
            f"`{BOT_PREFIX}skip`"
        ),
        extras={
            "category": "Music",
            "pro": "Instantly jumps to the next track in the queue.",
        },
    )
    async def skip(self, ctx: commands.Context) -> None:
        if ctx.guild is None:
            return await ctx.reply("This command can only be used in a server.", mention_author=False)
        await defer_interaction(ctx)
        if not await ensure_voice(ctx):
            return

        player = get_player(self.bot, ctx.guild)
        if not player.voice or (not player.voice.is_playing() and not player.voice.is_paused()):
            return await ctx.reply("Nothing is playing.", mention_author=False)
        await player.skip()
        log.info("%s skipped track", ctx.author)
        await ctx.reply("Skipped.", mention_author=False)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Skip(bot))
