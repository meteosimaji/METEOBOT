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
            "Show a numbered list of recent additions when no number is provided. "
            "Provide a number to remove that many steps back in the addition order "
            "(1 = latest, 2 = the one before that, and so on). "
            "You can also remove by stable ID shown in the list (e.g., A12). "
            "This follows the order songs were added, not the current queue order, so looping won't change the count.\n\n"
            "**Usage**: `/remove [steps]`\n"
            f"`{BOT_PREFIX}remove [steps]`"
        ),
        extras={
            "category": "Music",
            "pro": (
                "Counts by addition order rather than queue position. Looping or reordering won't change the index. "
                "Call without a number to show a numbered list (with IDs), then pass a number or ID to remove."
            ),
        },
    )
    async def remove(self, ctx: commands.Context, steps: str | None = None) -> None:
        """Remove the Nth most recent pending track (1 = latest)."""
        if ctx.guild is None:
            return await safe_reply(ctx, "This command can only be used in a server.", mention_author=False)
        await defer_interaction(ctx)
        if not await ensure_voice(ctx):
            return

        player = get_player(self.bot, ctx.guild)
        if steps is None:
            recent = list(player.added_tracks)[::-1]
            if not recent:
                return await safe_reply(ctx, "No recent additions to remove.", mention_author=False)

            header = "Recent additions (latest first). Use `/remove <number>` or `/remove A<id>` to delete:\n"
            lines = []
            for idx, track in enumerate(recent, start=1):
                title = sanitize(track.title)
                url = track.page_url or ""
                add_id = f"A{track.add_id}" if track.add_id is not None else "A?"
                line = f"{idx}. [{add_id}] {title}"
                if url:
                    line += f" ({url})"
                lines.append(line)

            max_len = 1900
            chunks: list[str] = []
            buf = header
            for line in lines:
                if len(buf) + len(line) + 1 > max_len and buf:
                    chunks.append(buf)
                    buf = ""
                buf += ("" if not buf else "\n") + line
            if buf:
                chunks.append(buf)

            first = True
            for chunk in chunks:
                if first:
                    await safe_reply(
                        ctx,
                        chunk,
                        mention_author=False,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                    first = False
                    continue
                if ctx.interaction:
                    await ctx.interaction.followup.send(
                        chunk,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                else:
                    await ctx.send(
                        chunk,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
            return

        raw_steps = steps.strip()
        if not raw_steps:
            return await safe_reply(ctx, "Provide a number or ID (e.g., 1 or A12).", mention_author=False)

        add_id = None
        count = None
        if raw_steps.upper().startswith("A") and raw_steps[1:].isdigit():
            add_id = int(raw_steps[1:])
        elif raw_steps.isdigit():
            count = int(raw_steps)

        if add_id is None and count is None:
            return await safe_reply(ctx, "Provide a number or ID (e.g., 1 or A12).", mention_author=False)

        if count is not None and count <= 0:
            return await safe_reply(ctx, "Provide a positive number (1 = most recent addition).", mention_author=False)

        if add_id is not None:
            track = await player.remove_by_add_id(add_id)
        else:
            track = await player.remove_recent_add(count or 1)
        if not track:
            recent = list(player.added_tracks)[-8:][::-1]  # latest first
            hint = ""
            if recent:
                lines = []
                for i, t in enumerate(recent):
                    add_id_label = f"A{t.add_id}" if t.add_id is not None else "A?"
                    line = f"{i+1}. [{add_id_label}] {sanitize(t.title)}"
                    if t.page_url:
                        line += f" ({t.page_url})"
                    lines.append(line)
                hint = "\n\nRecent additions (latest first):\n" + "\n".join(lines)
            return await safe_reply(
                ctx,
                "Couldn't find that recent addition. It may have already finished, or there are fewer pending songs."
                + hint,
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
            )

        add_label = f"A{track.add_id}" if track.add_id is not None else "A?"
        log.info("%s removed track %s from recent additions: %s", ctx.author, add_label, track.title)
        replay_hint = f"Re-add with `/play {track.page_url}`." if track.page_url else ""
        await safe_reply(
            ctx,
            f"Removed **{sanitize(track.title)}** ({add_label}). "
            f"{replay_hint}".strip(),
            mention_author=False,
            allowed_mentions=discord.AllowedMentions.none(),
        )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Remove(bot))
