import logging
from datetime import datetime, timedelta, timezone

import discord
from discord.ext import commands

from utils import BOT_PREFIX, defer_interaction, error_embed, humanize_delta

log = logging.getLogger(__name__)

PROGRESS_BLOCKS = 20  # Mini bar with 20 segments


def day_progress_bar(seconds_today: float) -> tuple[str, float]:
    """Progress bar for today starting from midnight."""
    ratio = max(0.0, min(1.0, seconds_today / 86400))
    filled = int(ratio * PROGRESS_BLOCKS)
    empty = PROGRESS_BLOCKS - filled
    return "▰" * filled + "▱" * empty, ratio * 100


class Uptime(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    async def _reply(self, ctx: commands.Context, **kwargs) -> None:
        if ctx.interaction:
            if ctx.interaction.response.is_done():
                await ctx.interaction.followup.send(**kwargs)
            else:
                await ctx.interaction.response.send_message(**kwargs)
        else:
            await ctx.send(**kwargs)

    @commands.hybrid_command(
        name="uptime",
        description="Show how long the bot has been running",
        help=(
            "Display the bot's uptime and a progress bar for today.\n\n"
            "**Usage**: `/uptime`\n"
            "**Examples**: `/uptime`\n"
            f"`{BOT_PREFIX}uptime`"
        ),
        extras={
            "category": "Utility",
            "pro": (
                "Shows how long the bot has been running since its last restart. "
                "A progress bar illustrates how much of the current day has passed "
                "in the configured timezone."
            ),
        },
    )
    async def uptime(self, ctx: commands.Context) -> None:
        await defer_interaction(ctx)
        try:
            launch_time: datetime = getattr(self.bot, "launch_time", datetime.now(timezone.utc))
            if launch_time.tzinfo is None:
                launch_time = launch_time.replace(tzinfo=timezone.utc)
            now_utc = datetime.now(timezone.utc)
            delta = now_utc - launch_time
            total_seconds = int(delta.total_seconds())

            human = humanize_delta(total_seconds)
            days = total_seconds // 86400
            rem = total_seconds % 86400
            hours = rem // 3600
            rem %= 3600
            minutes = rem // 60
            seconds = rem % 60

            ofs = 9  # Default: JST
            if ctx.guild:
                ofs = int(getattr(self.bot, "guild_tz", {}).get(str(ctx.guild.id), 9))
            tz = timezone(timedelta(hours=ofs))
            local_now = now_utc.astimezone(tz)
            midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
            seconds_today = (local_now - midnight).total_seconds()
            bar, pct = day_progress_bar(seconds_today)

            embed = discord.Embed(
                title="⏱️ Bot Uptime",
                description="Here's how long I've been running!",
                color=0x42A5F5,
                timestamp=now_utc,
            )

            embed.add_field(
                name="⏳ Uptime",
                value=f"{human}\n`{days}d {hours}h {minutes}m {seconds}s`",
                inline=True,
            )

            ts = int(launch_time.timestamp())
            embed.add_field(
                name="📅 Started",
                value=f"<t:{ts}:F>\n<t:{ts}:R>",
                inline=True,
            )

            embed.add_field(
                name="🕰️ Timezone",
                value=f"UTC{ofs:+d}:00",
                inline=True,
            )

            embed.add_field(
                name="🌅 Day Progress",
                value=f"`{bar}` {pct:4.1f}%",
                inline=False,
            )

            embed.set_footer(text="Crafted with care ✨")

            await self._reply(ctx, embed=embed)
        except Exception:
            log.exception("uptime command failed")
            await self._reply(ctx, embed=error_embed(desc="Failed to get uptime"))


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Uptime(bot))
