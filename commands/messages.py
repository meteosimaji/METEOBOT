"""Command to fetch recent messages in the current channel."""

from __future__ import annotations

import logging
from typing import Iterable

import discord
from discord.ext import commands

from utils import BOT_PREFIX, defer_interaction, sanitize

log = logging.getLogger(__name__)


def _format_messages(messages: Iterable[discord.Message], *, limit: int = 3900) -> str:
    lines: list[str] = []
    total_length = 0
    truncated = False

    for msg in messages:
        snippet = sanitize(msg.content.strip()) if msg.content else ""
        if not snippet:
            details: list[str] = []
            if msg.attachments:
                details.append(f"attachments: {len(msg.attachments)}")
            if msg.embeds:
                details.append(f"embeds: {len(msg.embeds)}")
            snippet = f"[{', '.join(details)}]" if details else "[no content]"
        snippet = snippet.replace("\n", " ").replace("\r", " ")
        snippet = " ".join(snippet.split())
        if len(snippet) > 180:
            snippet = snippet[:177] + "…"
        ts = int(msg.created_at.timestamp())
        author = (
            sanitize(msg.author.display_name)
            if hasattr(msg.author, "display_name")
            else sanitize(str(msg.author))
        )
        line = f"• <t:{ts}:t> {author}: {snippet}"
        projected = total_length + len(line) + (1 if lines else 0)
        if projected > limit:
            truncated = True
            break

        lines.append(line)
        total_length = projected

    if truncated:
        suffix = "…(truncated)"
        projected = total_length + len(suffix) + (1 if lines else 0)
        if projected <= limit:
            lines.append(suffix)

    return "\n".join(lines)


class Messages(commands.Cog):
    """Fetch recent messages from the current channel."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="messages",
        description="Show recent messages in this channel.",
        help=(
            "Retrieve the latest messages from the current channel. Provide a number "
            "to change how many are shown (defaults to 10, min 1 / max 50).\n\n"
            "**Usage**: `/messages [count]`\n"
            "**Examples**: `/messages`, `/messages 5`\n"
            f"`{BOT_PREFIX}messages 15`"
        ),
        extras={
            "category": "Utility",
            "pro": (
                "If the count argument is missing the command shows 10 messages by "
                "default. The count is clamped to a maximum of 50 and long outputs "
                "are trimmed to fit Discord's embed limits."
            ),
        },
    )
    async def messages(
        self, ctx: commands.Context, count: int | None = None
    ) -> None:  # type: ignore[override]
        await defer_interaction(ctx)
        try:
            raw_amount = count if count is not None else 10
            amount = max(1, min(50, int(raw_amount)))

            if not hasattr(ctx.channel, "history"):
                if ctx.interaction and ctx.interaction.response.is_done():
                    await ctx.interaction.followup.send(
                        "This channel type is not supported.",
                        ephemeral=True,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                elif ctx.interaction:
                    await ctx.interaction.response.send_message(
                        "This channel type is not supported.",
                        ephemeral=True,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                else:
                    await ctx.reply(
                        "This channel type is not supported.",
                        mention_author=False,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                return

            history = [m async for m in ctx.channel.history(limit=amount)]
            history.reverse()

            if not history:
                content = "No recent messages found."
            else:
                content = _format_messages(history)

            embed = discord.Embed(
                title="\U0001F4DD Recent Messages",
                description=content,
                color=0x42A5F5,
            )
            embed.set_footer(text=f"Showing {len(history)} message(s)")

            if ctx.interaction:
                if ctx.interaction.response.is_done():
                    await ctx.interaction.followup.send(
                        embed=embed,
                        ephemeral=True,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                else:
                    await ctx.interaction.response.send_message(
                        embed=embed,
                        ephemeral=True,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
            else:
                await ctx.reply(
                    embed=embed,
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
        except discord.Forbidden:
            log.exception("Missing permissions to fetch channel history")
            message = "I don't have permission to read messages in this channel."
            if ctx.interaction:
                if ctx.interaction.response.is_done():
                    await ctx.interaction.followup.send(
                        message, ephemeral=True, allowed_mentions=discord.AllowedMentions.none()
                    )
                else:
                    await ctx.interaction.response.send_message(
                        message, ephemeral=True, allowed_mentions=discord.AllowedMentions.none()
                    )
            else:
                await ctx.reply(
                    message, mention_author=False, allowed_mentions=discord.AllowedMentions.none()
                )
        except discord.HTTPException:
            log.exception("Discord API rejected messages response")
            message = "Discord rejected the response. Please try a smaller count."
            if ctx.interaction:
                if ctx.interaction.response.is_done():
                    await ctx.interaction.followup.send(
                        message, ephemeral=True, allowed_mentions=discord.AllowedMentions.none()
                    )
                else:
                    await ctx.interaction.response.send_message(
                        message, ephemeral=True, allowed_mentions=discord.AllowedMentions.none()
                    )
            else:
                await ctx.reply(
                    message, mention_author=False, allowed_mentions=discord.AllowedMentions.none()
                )
        except Exception:
            log.exception("Failed to fetch messages")
            message = "Could not fetch messages right now."
            if ctx.interaction:
                if ctx.interaction.response.is_done():
                    await ctx.interaction.followup.send(
                        message, ephemeral=True, allowed_mentions=discord.AllowedMentions.none()
                    )
                else:
                    await ctx.interaction.response.send_message(
                        message, ephemeral=True, allowed_mentions=discord.AllowedMentions.none()
                    )
            else:
                await ctx.reply(
                    message, mention_author=False, allowed_mentions=discord.AllowedMentions.none()
                )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Messages(bot))
