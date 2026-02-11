from __future__ import annotations
# mypy: ignore-errors

import discord
from discord.ext import commands

from music import get_player
from utils import BOT_PREFIX, defer_interaction, safe_reply, tag_error_text


class LoopMode(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="loopmode",
        description="Set or view loop mode (none/track/queue)",
        help=(
            "Switch between loop modes or view the current setting. "
            "Accepts `OFF`, `TRACK`, `QUEUE`, or `NOW` (case-insensitive). "
            "`ON` is accepted as an alias for `QUEUE`. Leaving the argument empty shows the current mode.\n\n"
            "**Usage**: `/loopmode [OFF|TRACK|QUEUE|NOW]`\n"
            f"`{BOT_PREFIX}loopmode [OFF|TRACK|QUEUE|NOW]`"
        ),
        extras={
            "category": "Music",
            "destination": "Set repeat mode for playback (track, queue, or off).",
            "plus": "Use TRACK/QUEUE/OFF (ON maps to QUEUE); omitting the argument shows the current loop mode.",
            "pro": (
                "Sets loop to repeat the current track, the whole queue, or turn looping off. "
                "Shows the current mode when no value is provided. `ON` aliases to `QUEUE`. "
                "LLMs should pass uppercase tokens."
            ),
        },
    )
    async def loopmode(self, ctx: commands.Context, state: str | None = None) -> None:
        if ctx.guild is None:
            return await safe_reply(
                ctx, tag_error_text("This command can only be used in a server."), mention_author=False
            )
        await defer_interaction(ctx)

        player = get_player(self.bot, ctx.guild)

        normalized = (state or "now").strip().lower()
        choices = {"off": "none", "none": "none", "queue": "queue", "on": "queue", "track": "track", "now": "now"}
        if normalized not in choices:
            return await safe_reply(
                ctx,
                tag_error_text(
                    "Choose **OFF**, **TRACK**, **QUEUE**, or **NOW** (case-insensitive). `ON` is accepted for queue looping."
                ),
                mention_author=False,
            )

        embed = discord.Embed(title="🔁 Loop Mode", color=0x1DB954)
        current = player.loop.capitalize()

        if normalized == "now":
            embed.description = f"Loop is currently **{current}**."
            return await ctx.reply(embed=embed, mention_author=False)

        desired_mode = choices[normalized]
        if player.loop == desired_mode:
            embed.description = (
                f"Loop is already **{current}**. No changes were made."
            )
            return await ctx.reply(embed=embed, mention_author=False)

        player.loop = desired_mode
        embed.description = f"Loop set to **{player.loop.capitalize()}**."
        await ctx.reply(embed=embed, mention_author=False)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(LoopMode(bot))
