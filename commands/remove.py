from __future__ import annotations

import logging

import discord
from discord.ext import commands

from music import get_player
from utils import BOT_PREFIX, defer_interaction, safe_reply, ensure_voice, sanitize

log = logging.getLogger(__name__)


class Remove(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="remove",
        description="Remove a recently added track by addition order",
        help=(
            "Delete the most recently added song by default. Provide a number to remove that many steps "
            "back in the addition order (1 = latest, 2 = the one before that, and so on). "
            "This follows the order songs were added, not the current queue order, so looping won't change the count.\n\n"
            "**Usage**: `/remove [steps]`\n"
            f"`{BOT_PREFIX}remove [steps]`"
        ),
        extras={
            "category": "Music",
            "pro": (
                "Counts by addition order rather than queue position. Looping or reordering won't change the index. "
                "LLMs should pass a positive integer (default 1) to drop a recent addition before it plays."
            ),
        },
    )
    async def remove(self, ctx: commands.Context, steps: int | None = None) -> None:
        """Remove the Nth most recent pending track (1 = latest)."""
        if ctx.guild is None:
            return await safe_reply(ctx, "This command can only be used in a server.", mention_author=False)
        await defer_interaction(ctx)
        if not await ensure_voice(ctx):
            return

        count = steps if steps is not None else 1
        if count <= 0:
            return await safe_reply(ctx, "Provide a positive number (1 = most recent addition).", mention_author=False)

        player = get_player(self.bot, ctx.guild)
        track = await player.remove_recent_add(count)
        if not track:
            recent = list(player.added_tracks)[-8:][::-1]  # latest first
            hint = ""
            if recent:
                lines = [f"{i+1}. {sanitize(t.title)}" for i, t in enumerate(recent)]
                hint = "\n\nRecent additions (latest first):\n" + "\n".join(lines)
            return await safe_reply(
                ctx,
                "Couldn't find that recent addition. It may have already finished, or there are fewer pending songs."
                + hint,
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
            )

        log.info("%s removed track #%d from recent additions: %s", ctx.author, count, track.title)
        await safe_reply(
            ctx,
            f"Removed **{sanitize(track.title)}** (#{count} from the most recent additions).",
            mention_author=False,
            allowed_mentions=discord.AllowedMentions.none(),
        )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Remove(bot))
