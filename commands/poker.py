from __future__ import annotations

import logging

from discord.ext import commands

from utils import BOT_PREFIX, defer_interaction, safe_reply, tag_error_text

log = logging.getLogger(__name__)


class Poker(commands.Cog):
    """Create simajilord poker room links."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="poker",
        description="Create a heads-up NLHE room link on simajilord.com.",
        usage="<ranked:true|false>",
        help=(
            "Create a heads-up NLHE room URL for simajilord.com.\n"
            "Ranked mode disables in-game GTO button by design.\n\n"
            "**Usage**: `/poker ranked:<true|false>`\n"
            "**Examples**: `/poker ranked:false`\n"
            f"`{BOT_PREFIX}poker ranked:true`"
        ),
        extras={
            "category": "Games",
            "destination": "Create a heads-up NLHE room URL that opens on simajilord.com.",
            "plus": "Fast room creation from Discord with identity-bound access tokens.",
            "pro": "Supports ranked/casual metadata so ranked rooms can disable GTO reference during matches.",
        },
    )
    async def poker(self, ctx: commands.Context, *, ranked: bool = False) -> None:
        await defer_interaction(ctx)
        ask_cog = self.bot.get_cog("Ask")
        if ask_cog is None or not hasattr(ask_cog, "create_poker_room_link"):
            await safe_reply(
                ctx,
                tag_error_text("Poker service is unavailable because Ask cog is not loaded."),
                mention_author=False,
                ephemeral=True,
            )
            return
        try:
            payload = await ask_cog.create_poker_room_link(ctx, ranked=ranked)
        except Exception:
            log.exception("Failed to create poker room link")
            await safe_reply(
                ctx,
                tag_error_text("Failed to create poker room link."),
                mention_author=False,
                ephemeral=True,
            )
            return

        url = str(payload.get("url") or "")
        if not url:
            await safe_reply(
                ctx,
                tag_error_text("Failed to issue poker room link."),
                mention_author=False,
                ephemeral=True,
            )
            return

        mode = "Ranked" if ranked else "Casual"
        gto = "disabled" if ranked else "available"
        await safe_reply(
            ctx,
            f"🎴 **{mode} HU NLHE room**\n{url}\nGTO button: **{gto}**",
            mention_author=False,
        )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Poker(bot))
