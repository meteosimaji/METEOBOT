import asyncio
import logging
import time
from datetime import datetime, timezone

import discord
from discord.ext import commands
from utils import BOT_PREFIX, defer_interaction, tag_error_embed

log = logging.getLogger(__name__)

class Ping(commands.Cog):
    """Simple ping command."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(  # type: ignore[arg-type]
        name="ping",
        description="Check the bot's responsiveness with style and speed!",
        help=(
            "Gauge API, Gateway and WebSocket latency to see how quickly I respond.\n\n"
            "**Usage**: `/ping`\n"
            "**Examples**: `/ping`\n"
            f"`{BOT_PREFIX}ping`"
        ),
        extras={
            "category": "Utility",
            "destination": "Measure bot/API/Gateway latency and heartbeat delay.",
            "plus": "Shows separate timings so you can distinguish Discord API delay from websocket heartbeat delay.",
            "pro": (
                "A short deferral measures API latency, a bare GET /gateway call "
                "reports Gateway latency, and the heartbeat delay provides WebSocket "
                "latency. Compare the three numbers to pinpoint slowdowns."
            ),
        },
    )
    async def ping(self, ctx: commands.Context) -> None:
        try:
            async def _measure_gateway() -> float | None:
                t = time.perf_counter()
                try:
                    await self.bot.http.request(discord.http.Route("GET", "/gateway"))
                    return (time.perf_counter() - t) * 1000
                except Exception:
                    return None

            gateway_task = asyncio.create_task(_measure_gateway())

            start = time.perf_counter()
            await defer_interaction(ctx)
            api_latency = (time.perf_counter() - start) * 1000
            gateway_latency = await gateway_task
            ws_latency = self.bot.latency * 1000

            now_utc = datetime.now(timezone.utc)
            embed = discord.Embed(
                title="\U0001F3D3 Bot Ping",
                description="Here's how fast I can respond!",
                color=0xFFC0CB,
                timestamp=now_utc,
            )
            embed.add_field(
                name="\U0001F4BB API Latency", value=f"{api_latency:.0f} ms", inline=True
            )
            embed.add_field(
                name="\U0001F310 Gateway",
                value=f"{gateway_latency:.0f} ms" if gateway_latency is not None else "N/A",
                inline=True,
            )
            embed.add_field(
                name="\U0001F4E1 WebSocket", value=f"{ws_latency:.0f} ms", inline=True
            )
            embed.set_footer(text="Crafted with care ✨")

            if ctx.interaction:
                await ctx.interaction.followup.send(embed=embed)
            else:
                await ctx.reply(embed=embed, mention_author=False)
        except Exception:
            log.exception("Failed to execute ping command")
            error_embed = discord.Embed(
                title="\u26A0\ufe0f Ping Failed",
                description="An error occurred while measuring latency.",
                color=0xFF0000,
            )
            error_embed = tag_error_embed(error_embed)
            try:
                if ctx.interaction:
                    await ctx.interaction.followup.send(embed=error_embed, ephemeral=True)
                else:
                    await ctx.reply(embed=error_embed, mention_author=False)
            except Exception:
                pass

async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Ping(bot))
