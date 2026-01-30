from __future__ import annotations
# mypy: ignore-errors

import logging

from discord.ext import commands

from music import get_player
from utils import BOT_PREFIX, defer_interaction, ensure_voice, tag_error_text

log = logging.getLogger(__name__)


class Resume(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="resume",
        description="Resume the paused track",
        help=(
            "Resume playback of the song currently on pause. You must be in the same voice channel as the bot.\n\n"
            "**Usage**: `/resume`\n"
            f"`{BOT_PREFIX}resume`"
        ),
        extras={
            "category": "Music",
            "pro": "Restarts the paused track from the exact spot it stopped.",
        },
    )
    async def resume(self, ctx: commands.Context) -> None:
        if ctx.guild is None:
            return await ctx.reply(
                tag_error_text("This command can only be used in a server."), mention_author=False
            )
        await defer_interaction(ctx)
        if not await ensure_voice(ctx):
            return

        player = get_player(self.bot, ctx.guild)
        if not player.voice or (not player.voice.is_playing() and not player.voice.is_paused()):
            return await ctx.reply(tag_error_text("Nothing is playing."), mention_author=False)
        if not player.voice.is_paused():
            return await ctx.reply(tag_error_text("Playback isn't paused."), mention_author=False)

        await player.resume()
        log.info("%s resumed playback", ctx.author)
        await ctx.reply("Resumed.", mention_author=False)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Resume(bot))
