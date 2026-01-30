from __future__ import annotations
# mypy: ignore-errors

import logging

from discord.ext import commands

from music import get_player
from utils import BOT_PREFIX, defer_interaction, ensure_voice, tag_error_text

log = logging.getLogger(__name__)


class Bye(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="bye",
        description="Stop playback, clear the queue, and leave voice",
        help=(
            "Stop playing, clear the queue, and disconnect from the voice channel.\n\n"
            "**Usage**: `/bye`\n"
            f"`{BOT_PREFIX}bye`"
        ),
        extras={
            "category": "Music",
            "pro": "Always leaves the voice channel after clearing playback, regardless of Auto Leave.",
        },
    )
    async def bye(self, ctx: commands.Context) -> None:
        if ctx.guild is None:
            return await ctx.reply(
                tag_error_text("This command can only be used in a server."), mention_author=False
            )
        await defer_interaction(ctx)
        if not await ensure_voice(ctx):
            return

        player = get_player(self.bot, ctx.guild)
        if not player.voice or not player.voice.is_connected():
            return await ctx.reply(tag_error_text("I'm not in voice right now."), mention_author=False)
        if (
            not player.voice.is_playing()
            and not player.voice.is_paused()
            and not player.queue
            and not player.current
        ):
            return await ctx.reply(tag_error_text("Nothing to stop."), mention_author=False)

        await player.stop()
        log.info("%s stopped playback and left voice", ctx.author)
        await ctx.reply("Stopped, cleared the queue, and left voice.", mention_author=False)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Bye(bot))
