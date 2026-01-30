from __future__ import annotations
# mypy: ignore-errors

import logging

import discord
from discord.ext import commands

from music import get_player, progress_bar
from utils import BOT_PREFIX, defer_interaction, format_timestamp, humanize_delta, tag_error_text

log = logging.getLogger(__name__)


class Now(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="now",
        description="Display the song that's currently playing",
        help=(
            "Show details about the current track including title, a progress bar and elapsed time.\n\n"
            "**Usage**: `/now`\n"
            f"`{BOT_PREFIX}now`"
        ),
        extras={
            "category": "Music",
            "pro": (
                "Display details about the currently playing track, including a progress bar, "
                "elapsed/total time and duration when available."
            ),
        },
    )
    async def now(self, ctx: commands.Context) -> None:
        if ctx.guild is None:
            return await ctx.reply(
                tag_error_text("This command can only be used in a server."), mention_author=False
            )
        await defer_interaction(ctx)
        player = get_player(self.bot, ctx.guild)
        if not player.current or not player.voice:
            return await ctx.reply(tag_error_text("Nothing is playing."), mention_author=False)
        position = player.get_position()
        bar, pct = progress_bar(position, player.current.duration)
        embed = discord.Embed(
            title="🎵 Now Playing",
            description=f"[{player.current.title}]({player.current.page_url})",
            color=0x1DB954,
        )
        time_txt = format_timestamp(position)
        if player.current.duration:
            time_txt += f"/{format_timestamp(player.current.duration)}"
        embed.add_field(name="Progress", value=f"`{bar}` {pct:4.1f}% `{time_txt}`", inline=False)
        if player.current.duration:
            embed.set_footer(text=humanize_delta(player.current.duration))
        await ctx.reply(embed=embed, mention_author=False)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Now(bot))
