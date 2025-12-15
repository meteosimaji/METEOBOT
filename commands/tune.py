from __future__ import annotations

import logging

from discord.ext import commands

from music import get_player
from utils import defer_interaction, ensure_voice, BOT_PREFIX

log = logging.getLogger(__name__)


def _parse_two_floats(values: str) -> tuple[float, float] | None:
    parts = (values or "").strip().split()
    if len(parts) != 2:
        return None
    try:
        a = float(parts[0])
        b = float(parts[1])
    except ValueError:
        return None
    return a, b


class Tune(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="tune",
        description="Adjust music playback speed and pitch",
        help=(
            "Change the playback speed and pitch of the current track.\n"
            "Provide two numbers: `<speed> <pitch>` (each 0.5–2.0).\n\n"
            "**Usage**: `/tune <speed> <pitch>`\n"
            "**Examples**: `/tune 1.2 0.8`\n"
            f"`{BOT_PREFIX}tune 1.5 1.5`"
        ),
        extras={
            "category": "Music",
            "pro": (
                "Fine-tune playback speed and pitch. Implemented as a single string argument "
                "so the LLM can invoke it via bot_invoke."
            ),
        },
    )
    async def tune(self, ctx: commands.Context, *, values: str | None = None) -> None:
        # values is optional to satisfy LLM invocation (single optional arg)
        if ctx.guild is None:
            return await ctx.reply("This command can only be used in a server.", mention_author=False)
        await defer_interaction(ctx)
        if not await ensure_voice(ctx):
            return

        if not values:
            return await ctx.reply("Give me two numbers like `1.2 0.9` (speed pitch).", mention_author=False)

        parsed = _parse_two_floats(values)
        if not parsed:
            return await ctx.reply("Invalid format. Use: `<speed> <pitch>` (e.g. `1.2 0.9`).", mention_author=False)

        speed, pitch = parsed
        if not 0.5 <= speed <= 2.0:
            return await ctx.reply("Speed must be between 0.5× and 2.0×.", mention_author=False)
        if not 0.5 <= pitch <= 2.0:
            return await ctx.reply("Pitch must be between 0.5× and 2.0×.", mention_author=False)

        player = get_player(self.bot, ctx.guild)
        if not player.voice or not player.current:
            return await ctx.reply("Nothing is playing.", mention_author=False)

        position = player.get_position()
        old_speed, old_pitch = player.speed, player.pitch
        player.speed = speed
        player.pitch = pitch
        success = await player.seek(position)
        if not success:
            player.speed = old_speed
            player.pitch = old_pitch
            return await ctx.reply("Failed to retune.", mention_author=False)

        await ctx.reply(f"Set speed to {speed:.2f}× and pitch to {pitch:.2f}×.", mention_author=False)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Tune(bot))
