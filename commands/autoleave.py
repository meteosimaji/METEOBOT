from __future__ import annotations

import discord
from discord.ext import commands

from music import get_player
from utils import BOT_PREFIX, defer_interaction, safe_reply, tag_error_text


class AutoLeave(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="autoleave",
        description="Toggle Auto Leave (disconnect when no one is listening)",
        help=(
            "Turn Auto Leave on/off or view the current setting. "
            "Use `ON`, `OFF`, or `NOW` (case-insensitive). "
            "Leaving the argument empty is treated the same as `NOW`.\n\n"
            "**Usage**: `/autoleave [ON|OFF|NOW]`\n"
            f"`{BOT_PREFIX}autoleave [ON|OFF|NOW]`"
        ),
        extras={
            "category": "Music",
            "pro": (
                "Auto Leave will disconnect when the bot is idle or alone in voice. "
                "Passing no value shows the current state, ON enables, OFF disables. "
                "LLMs should prefer uppercase tokens."
            ),
        },
    )
    async def autoleave(self, ctx: commands.Context, state: str | None = None) -> None:
        if ctx.guild is None:
            return await safe_reply(
                ctx, tag_error_text("This command can only be used in a server."), mention_author=False
            )
        await defer_interaction(ctx)

        player = get_player(self.bot, ctx.guild)

        normalized = (state or "now").strip().lower()
        if normalized not in {"on", "off", "now"}:
            return await safe_reply(
                ctx,
                tag_error_text("Choose **ON**, **OFF**, or **NOW** (case-insensitive)."),
                mention_author=False,
            )

        embed = discord.Embed(title="⚙️ Auto Leave", color=0x1DB954)
        current = "ON" if player.auto_leave else "OFF"

        if normalized == "now":
            embed.description = f"Auto Leave is currently **{current}**."
            return await ctx.reply(embed=embed, mention_author=False)

        desired_on = normalized == "on"
        if player.auto_leave == desired_on:
            embed.description = (
                f"Auto Leave is already **{current}**. No changes were made."
            )
            return await ctx.reply(embed=embed, mention_author=False)

        player.auto_leave = desired_on
        if desired_on:
            player._request_auto_leave_check()
            player.request_lonely_auto_leave(delay=0.0)
        else:
            player._cancel_auto_leave_task()

        embed.description = f"Auto Leave set to **{'ON' if desired_on else 'OFF'}**."
        await ctx.reply(embed=embed, mention_author=False)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(AutoLeave(bot))
