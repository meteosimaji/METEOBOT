from __future__ import annotations

import logging

import discord
from discord.ext import commands

from music import get_player
from utils import BOT_PREFIX, defer_interaction, safe_reply, ensure_voice, sanitize, tag_error_text

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
            "You can also remove by stable ID shown in the list (e.g., A12), "
            "or provide multiple numbers/IDs separated by commas. "
            "This follows the order songs were added, not the current queue order, so looping won't change the count.\n\n"
            "**Usage**: `/remove [steps]`\n"
            f"`{BOT_PREFIX}remove [steps]`"
        ),
        extras={
            "category": "Music",
            "pro": (
                "Counts by addition order rather than queue position. Looping or reordering won't change the index. "
                "Call without a number to show a numbered list (with IDs), then pass a number or ID to remove. "
                "You can also pass a comma-separated list like A12,3,A14."
            ),
        },
    )
    async def remove(self, ctx: commands.Context, steps: str | None = None) -> None:
        """Remove the Nth most recent pending track (1 = latest)."""
        if ctx.guild is None:
            return await safe_reply(
                ctx, tag_error_text("This command can only be used in a server."), mention_author=False
            )
        await defer_interaction(ctx)
        if not await ensure_voice(ctx):
            return

        def _clamp_line(text: str, limit: int) -> str:
            if len(text) <= limit:
                return text
            if limit <= 3:
                return text[:limit]
            return text[: limit - 3] + "..."

        player = get_player(self.bot, ctx.guild)
        if steps is None:
            recent = list(player.added_tracks)[::-1]
            if not recent:
                return await safe_reply(
                    ctx, tag_error_text("No recent additions to remove."), mention_author=False
                )

            header = "Recent additions (latest first). Use `/remove <number>` or `/remove A<id>` to delete:\n"
            lines = []
            for idx, track in enumerate(recent, start=1):
                title = sanitize(track.title)
                url = track.page_url or ""
                add_id = f"A{track.add_id}" if track.add_id is not None else "A?"
                line = f"{idx}. [{add_id}] {title}"
                if url:
                    line += f" (<{url}>)"
                lines.append(_clamp_line(line, 800))

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
                    await ctx.reply(
                        chunk,
                        mention_author=False,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
            return

        raw_steps = steps.strip()
        if not raw_steps:
            return await safe_reply(
                ctx,
                tag_error_text("Provide a number or ID (e.g., 1 or A12)."),
                mention_author=False,
            )

        parts = [part.strip() for part in raw_steps.split(",") if part.strip()]
        if not parts:
            return await safe_reply(
                ctx,
                tag_error_text("Provide a number or ID (e.g., 1 or A12)."),
                mention_author=False,
            )
        if len(parts) > 25:
            return await safe_reply(
                ctx,
                "Too many targets. Please remove 25 or fewer items at a time.",
                mention_author=False,
            )

        recent_snapshot: list | None = None
        if any(part.isdigit() for part in parts):
            recent_snapshot = list(player.added_tracks)[::-1]

        targets: list[tuple[int, str]] = []
        not_found: list[str] = []
        duplicate_targets: list[str] = []
        seen_targets: set[int] = set()
        for part in parts:
            if part.upper().startswith("A") and part[1:].isdigit():
                add_id = int(part[1:])
                if add_id in seen_targets:
                    duplicate_targets.append(part)
                else:
                    seen_targets.add(add_id)
                    targets.append((add_id, part))
                continue
            if part.isdigit():
                count = int(part)
                if count <= 0:
                    return await safe_reply(
                        ctx,
                        "Provide a positive number (1 = most recent addition).",
                        mention_author=False,
                    )
                if recent_snapshot is None or count > len(recent_snapshot):
                    not_found.append(part)
                    continue
                target_track = recent_snapshot[count - 1]
                if target_track.add_id is None:
                    not_found.append(part)
                    continue
                add_id = target_track.add_id
                if add_id in seen_targets:
                    duplicate_targets.append(part)
                else:
                    seen_targets.add(add_id)
                    targets.append((add_id, part))
                continue
            return await safe_reply(
                ctx,
                "Provide numbers or IDs (e.g., 1 or A12). You can separate multiple with commas.",
                mention_author=False,
            )

        removed: list[tuple[str, str]] = []
        for add_id, label in targets:
            track = await player.remove_by_add_id(add_id)
            if not track:
                not_found.append(label)
                continue
            add_label = f"A{track.add_id}" if track.add_id is not None else "A?"
            log.info("%s removed track %s from recent additions: %s", ctx.author, add_label, track.title)
            url = f" — <{track.page_url}>" if track.page_url else ""
            removed.append((sanitize(track.title), f"{add_label}{url}"))

        if not removed:
            recent = list(player.added_tracks)[-8:][::-1]  # latest first
            hint = ""
            if recent:
                lines = []
                for i, t in enumerate(recent):
                    add_id_label = f"A{t.add_id}" if t.add_id is not None else "A?"
                    line = f"{i+1}. [{add_id_label}] {sanitize(t.title)}"
                    if t.page_url:
                        line += f" (<{t.page_url}>)"
                    lines.append(_clamp_line(line, 800))
                hint = "\n\nRecent additions (latest first):\n" + "\n".join(lines)
            message = (
                "Couldn't find that recent addition. It may have already finished, or there are fewer pending songs."
                + hint
            )
            if duplicate_targets:
                message += "\n\nDuplicates ignored: " + ", ".join(duplicate_targets)
            return await safe_reply(
                ctx,
                message,
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
            )

        removed_lines = [f"- **{title}** ({details})" for title, details in removed]
        message = "Removed:\n" + "\n".join(removed_lines)
        if not_found:
            message += "\n\nCouldn't find: " + ", ".join(not_found)
        if duplicate_targets:
            message += "\n\nDuplicates ignored: " + ", ".join(duplicate_targets)

        max_len = 1900
        chunks: list[str] = []
        buf = ""
        for line in message.split("\n"):
            line = _clamp_line(line, max_len)
            next_line = line if not buf else f"{buf}\n{line}"
            if len(next_line) > max_len and buf:
                chunks.append(buf)
                buf = line
            else:
                buf = next_line
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
                await ctx.reply(
                    chunk,
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions.none(),
                )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Remove(bot))
