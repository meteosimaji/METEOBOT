from __future__ import annotations

import logging

from discord.ext import commands

from music import VoiceConnectionError, get_player
from utils import BOT_PREFIX, defer_interaction

log = logging.getLogger(__name__)


class Join(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="join",
        description="Join your voice channel",
        help=(
            "Make the bot join the voice channel you're currently in.\n\n"
            "**Usage**: `/join`\n"
            f"`{BOT_PREFIX}join`"
        ),
        extras={
            "category": "Music",
            "pro": "Joins your current voice channel without queueing anything.",
        },
    )
    async def join(self, ctx: commands.Context) -> None:
        if ctx.guild is None:
            return await ctx.reply("This command can only be used in a server.", mention_author=False)
        await defer_interaction(ctx)

        if not getattr(ctx.author, "voice", None) or not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.reply("Join a voice channel first.", mention_author=False)

        player = get_player(self.bot, ctx.guild)
        try:
            await player.join(ctx.author.voice.channel)  # type: ignore[arg-type]
        except (VoiceConnectionError, Exception):
            log.exception("Failed to join voice")
            return await ctx.reply("Couldn't join your voice channel (check permissions/connectivity).", mention_author=False)

        await ctx.reply("Joined your voice channel.", mention_author=False)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Join(bot))
