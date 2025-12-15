"""Bulk message deletion command."""

from __future__ import annotations

import asyncio
import contextlib
import re
import shlex
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional

import discord
from discord import app_commands
from discord.ext import commands

from utils import BOT_PREFIX, defer_interaction, safe_reply, strip_markdown

# Accepts:
# - https://discord.com/channels/<guild>/<channel>/<message>
# - https://ptb.discord.com/channels/<guild>/<channel>/<message>
# - https://discordapp.com/channels/<guild>/<channel>/<message>
# - wrapped in <...> (Discord auto-format)
MESSAGE_LINK_RE = re.compile(
    r"https?://(?:ptb\.|canary\.)?discord(?:app)?\.com/channels/"
    r"(?P<guild>\d+)/(?P<channel>\d+)/(?P<message>\d+)(?:/)?(?:\?.*)?$"
)

FORCE_TOKENS = {
    "--force",
    "-f",
    "force",
    "force=true",
    "force=1",
    "include_pinned",
    "include-pinned",
    "pinned=true",
}


@dataclass
class PurgeQuery:
    """Query parameters for the purge command."""

    channel: discord.abc.Messageable
    amount: Optional[int] = None
    user: Optional[discord.Member] = None
    start_id: Optional[int] = None
    end_id: Optional[int] = None
    phrase: Optional[str] = None
    include_pinned: bool = False


class ConfirmView(discord.ui.View):
    def __init__(self, author_id: int) -> None:
        super().__init__(timeout=30)
        self.author_id = author_id
        self.result: bool | None = None

    def disable_all_items(self) -> None:
        for item in self.children:
            item.disabled = True

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author_id:
            await interaction.response.send_message(
                "Only the command invoker can confirm.", ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="Delete", style=discord.ButtonStyle.danger)
    async def confirm(
        self, interaction: discord.Interaction, _: discord.ui.Button
    ) -> None:
        self.result = True
        self.disable_all_items()
        await interaction.response.edit_message(view=self)
        self.stop()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
        self.result = False
        self.disable_all_items()
        await interaction.response.edit_message(view=self)
        self.stop()


def _clean_token(tok: str) -> str:
    tok = tok.strip()
    if tok.startswith("<") and tok.endswith(">"):
        tok = tok[1:-1]
    return tok.strip()


def _jump_link(guild_id: int, channel_id: int, message_id: int) -> str:
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"


async def parse_query(ctx: commands.Context, tokens: List[str]) -> PurgeQuery:
    if not isinstance(ctx.channel, discord.abc.Messageable):
        raise ValueError("This channel type is not supported.")

    channel: discord.abc.Messageable = ctx.channel
    amount: Optional[int] = None
    user: Optional[discord.Member] = None
    msg_ids: List[int] = []
    phrase_parts: List[str] = []
    include_pinned = False

    link_channel_id: Optional[int] = None

    for raw in tokens:
        tok = _clean_token(raw)

        if tok.lower() in FORCE_TOKENS:
            include_pinned = True
            continue

        m = MESSAGE_LINK_RE.fullmatch(tok)
        if m:
            cid = int(m.group("channel"))
            mid = int(m.group("message"))
            if link_channel_id is None:
                link_channel_id = cid
            elif link_channel_id != cid:
                raise ValueError(
                    "Message links must point to the same channel for a range purge."
                )
            msg_ids.append(mid)

            if ctx.guild:
                ch = ctx.guild.get_channel(cid)
                if ch is None and hasattr(ctx.guild, "get_thread"):
                    ch = ctx.guild.get_thread(cid)  # type: ignore[attr-defined]
                if isinstance(ch, discord.abc.Messageable):
                    channel = ch
            continue

        if tok.isdigit():
            if len(tok) >= 17:  # snowflake-ish
                msg_ids.append(int(tok))
                continue
            if amount is not None:
                raise ValueError("Too many numbers. Provide a single amount.")
            amount = int(tok)
            continue

        # User?
        try:
            maybe_user = await commands.MemberConverter().convert(ctx, tok)
            if isinstance(maybe_user, discord.Member):
                user = maybe_user
                continue
        except Exception:
            pass

        # Channel?
        try:
            ch = await commands.GuildChannelConverter().convert(ctx, tok)
            if isinstance(ch, discord.abc.Messageable):
                channel = ch
                continue
        except Exception:
            pass

        # Thread (some forks expose this converter; safe to try)
        thread_conv = getattr(commands, "ThreadConverter", None)
        if thread_conv is not None:
            try:
                th = await thread_conv().convert(ctx, tok)  # type: ignore[misc]
                if isinstance(th, discord.abc.Messageable):
                    channel = th
                    continue
            except Exception:
                pass

        phrase_parts.append(tok)

    if len(msg_ids) > 2:
        raise ValueError("Too many message IDs/links. Provide at most two for a range.")

    phrase = " ".join(phrase_parts).strip() or None
    start_id = msg_ids[0] if msg_ids else None
    end_id = msg_ids[1] if len(msg_ids) > 1 else None

    return PurgeQuery(
        channel=channel,
        amount=amount,
        user=user,
        start_id=start_id,
        end_id=end_id,
        phrase=phrase,
        include_pinned=include_pinned,
    )


def compile_phrase(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(strip_markdown(phrase)).replace(r"\*", ".*")
    return re.compile(escaped, flags=re.IGNORECASE)


def _matches(m: discord.Message, query: PurgeQuery, pat: Optional[re.Pattern[str]]) -> bool:
    if m.pinned and not query.include_pinned:
        return False
    if query.user and m.author.id != query.user.id:
        return False
    if pat and not pat.search(strip_markdown(m.content)):
        return False
    return True


async def _collect_targets(
    ctx: commands.Context, query: PurgeQuery
) -> tuple[list[int], Counter[int], dict[int, str]]:
    """
    Collect message IDs to delete, and per-author counts.

    Returns:
      - ids: list[int]
      - counts: Counter[author_id]
      - mentions: dict[author_id] -> mention string
    """
    channel = query.channel
    pat = compile_phrase(query.phrase) if query.phrase else None

    ids: list[int] = []
    counts: Counter[int] = Counter()
    mentions: dict[int, str] = {}

    def record(m: discord.Message) -> None:
        ids.append(m.id)
        aid = m.author.id
        counts[aid] += 1
        mentions.setdefault(aid, getattr(m.author, "mention", f"<@{aid}>"))

    invoked_id = ctx.message.id if ctx.message else None

    # Range (two IDs)
    if query.start_id and query.end_id:
        start = await channel.fetch_message(query.start_id)  # type: ignore[attr-defined]
        end = await channel.fetch_message(query.end_id)  # type: ignore[attr-defined]

        if start.created_at > end.created_at:
            start, end = end, start

        candidates: list[discord.Message] = [start]
        async for m in channel.history(
            after=start, before=end, oldest_first=True, limit=None  # type: ignore[attr-defined]
        ):
            candidates.append(m)
        candidates.append(end)

        for m in candidates:
            if invoked_id and m.id == invoked_id:
                continue
            if _matches(m, query, pat):
                record(m)
        return ids, counts, mentions

    # From link -> next N (window) (include start)
    if query.start_id and query.amount:
        start = await channel.fetch_message(query.start_id)  # type: ignore[attr-defined]
        window: list[discord.Message] = [start]
        if query.amount > 1:
            async for m in channel.history(
                after=start,
                oldest_first=True,
                limit=query.amount - 1,  # window size
            ):
                window.append(m)

        for m in window:
            if invoked_id and m.id == invoked_id:
                continue
            if _matches(m, query, pat):
                record(m)
        return ids, counts, mentions

    # From link -> now (unbounded) (include start)
    if query.start_id:
        start = await channel.fetch_message(query.start_id)  # type: ignore[attr-defined]
        if not (invoked_id and start.id == invoked_id) and _matches(start, query, pat):
            record(start)
        async for m in channel.history(after=start, oldest_first=True, limit=None):
            if invoked_id and m.id == invoked_id:
                continue
            if _matches(m, query, pat):
                record(m)
        return ids, counts, mentions

    # Amount-based (latest N matching)
    if query.amount is not None:
        before = (
            ctx.message
            if (ctx.message and getattr(channel, "id", None) == ctx.channel.id)
            else None
        )
        async for m in channel.history(limit=None, before=before):
            if invoked_id and m.id == invoked_id:
                continue
            if _matches(m, query, pat):
                record(m)
                if len(ids) >= query.amount:
                    break
        return ids, counts, mentions

    # Otherwise: purge everything matching filters in the channel (unbounded)
    async for m in channel.history(limit=None):
        if invoked_id and m.id == invoked_id:
            continue
        if _matches(m, query, pat):
            record(m)
    return ids, counts, mentions


def _describe_query(ctx: commands.Context, data: PurgeQuery, total: int) -> str:
    ch = data.channel
    if data.start_id and data.end_id:
        return f"This will purge **{total}** message(s) in the selected range."
    if data.start_id and data.amount:
        return (
            f"This will remove **{total}** message(s) from the chosen message onward "
            f"(next {data.amount} scanned)."
        )
    if data.start_id:
        return f"This will remove **{total}** message(s) from the chosen message up to now."
    if data.user and data.amount is not None:
        return f"This will remove the last **{total}** message(s) from {data.user.mention}."
    if data.user:
        return f"This will remove every message from {data.user.mention}."
    if data.phrase and data.amount is not None:
        return f"This will remove the last **{total}** message(s) containing `{data.phrase}`."
    if data.phrase:
        return f"This will remove every message containing `{data.phrase}`."
    if data.amount is not None and ch != ctx.channel:
        return f"This will remove the last **{total}** message(s) in {ch.mention}."
    if data.amount is not None:
        return f"This will remove the last **{total}** message(s)."
    return f"This will remove **{total}** message(s) from {ch.mention}."


async def _delete_by_ids(
    channel: discord.abc.Messageable, message_ids: list[int], *, reason: str | None = None
) -> int:
    """Delete IDs efficiently: bulk for <14d (<=100 per call), individual otherwise."""
    now = discord.utils.utcnow()
    cutoff = now - timedelta(days=14)

    recent: list[int] = []
    old: list[int] = []
    for mid in message_ids:
        created = discord.utils.snowflake_time(mid)
        if created > cutoff:
            recent.append(mid)
        else:
            old.append(mid)

    deleted = 0

    delete_messages = getattr(channel, "delete_messages", None)
    if callable(delete_messages):
        for i in range(0, len(recent), 100):
            chunk = [discord.Object(id=x) for x in recent[i : i + 100]]
            try:
                await delete_messages(chunk, reason=reason)
                deleted += len(chunk)
            except Exception:
                # fallback to per-message deletes for this chunk
                for obj in chunk:
                    try:
                        pm = getattr(channel, "get_partial_message", None)
                        if callable(pm):
                            await pm(obj.id).delete(reason=reason)
                        else:
                            msg = await channel.fetch_message(obj.id)  # type: ignore[attr-defined]
                            await msg.delete(reason=reason)
                        deleted += 1
                    except Exception:
                        pass
    else:
        old.extend(recent)

    for mid in old:
        try:
            pm = getattr(channel, "get_partial_message", None)
            if callable(pm):
                await pm(mid).delete(reason=reason)
            else:
                msg = await channel.fetch_message(mid)  # type: ignore[attr-defined]
                await msg.delete(reason=reason)
            deleted += 1
        except Exception:
            pass

    return deleted


class Purge(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="purge",
        description="Bulk delete messages using flexible filters",
        help=(
            "Delete messages by count, range, user, keyword or link.\n\n"
            "**Usage**: `/purge <filters>`\n"
            "**Examples**:\n"
            "`/purge 25`\n"
            "`/purge https://discord.com/.../123 https://discord.com/.../456`\n"
            "`/purge https://discord.com/.../123 50`\n"
            "`/purge https://discord.com/.../123`\n"
            "`/purge @User 30`\n"
            "`/purge \"spam*\"`\n"
            "`/purge #general 100`\n"
            "`/purge --force 100` (includes pinned)\n"
            f"`{BOT_PREFIX}purge @User`"
        ),
        extras={
            "category": "Moderation",
            "pro": (
                "Combine filters such as amount, user, channel, message links or "
                "wildcard phrases to target specific messages. The bot summarises "
                "the matches and asks for confirmation for ranges or larger deletions."
            ),
        },
    )
    @commands.has_guild_permissions(manage_messages=True)
    @app_commands.checks.has_permissions(manage_messages=True)
    @app_commands.default_permissions(manage_messages=True)
    @app_commands.guild_only()
    async def purge(self, ctx: commands.Context, *, query: str | None = None) -> None:
        await defer_interaction(ctx)

        if ctx.guild is None:
            return await safe_reply(ctx, "Use this in a server text channel.", mention_author=False)

        try:
            tokens = shlex.split(query or "")
        except ValueError:
            return await safe_reply(
                ctx,
                "Invalid quotes in the query. If you use phrases, wrap them like: `/purge \"spam*\"`",
                ephemeral=True,
                mention_author=False,
            )

        if not tokens:
            return await safe_reply(
                ctx, "Specify an amount or filter.", ephemeral=True, mention_author=False
            )

        try:
            data = await parse_query(ctx, tokens)
        except ValueError as e:
            return await safe_reply(ctx, str(e), ephemeral=True, mention_author=False)

        channel = data.channel
        if not isinstance(channel, discord.abc.Messageable):
            return await safe_reply(ctx, "Channel not found.", ephemeral=True, mention_author=False)

        try:
            target_ids, counts, mentions = await _collect_targets(ctx, data)
        except discord.Forbidden:
            return await safe_reply(
                ctx,
                "I don't have permission to read message history in that channel.",
                ephemeral=True,
                mention_author=False,
            )
        except discord.NotFound:
            return await safe_reply(
                ctx,
                "One of the referenced messages couldn't be found.",
                ephemeral=True,
                mention_author=False,
            )
        except Exception:
            return await safe_reply(
                ctx,
                "Failed to build a purge list. Check links/IDs and permissions.",
                ephemeral=True,
                mention_author=False,
            )

        if data.amount is not None:
            target_ids = target_ids[: data.amount]

        if not target_ids:
            return await safe_reply(
                ctx,
                "No messages matched the query.",
                ephemeral=True,
                mention_author=False,
            )

        total = len(target_ids)
        is_range = data.start_id is not None and data.end_id is not None
        needs_confirm = is_range or total >= 30

        intro = _describe_query(ctx, data, total)

        lines: list[str] = []
        if len(counts) > 1:
            lines.append("**By author**")
            top = counts.most_common(10)
            for aid, c in top:
                lines.append(f"- {mentions.get(aid, f'<@{aid}>')}: {c}")
            remaining = len(counts) - len(top)
            if remaining > 0:
                lines.append(f"- …and {remaining} more")
            lines.append("")
        lines.append(f"**Total**: {total}")
        if data.include_pinned:
            lines.append("*include_pinned=true*")

        if ctx.guild and data.start_id and data.end_id:
            ch_id = getattr(channel, "id", None)
            if isinstance(ch_id, int):
                a = _jump_link(ctx.guild.id, ch_id, data.start_id)
                b = _jump_link(ctx.guild.id, ch_id, data.end_id)
                lines.append("")
                lines.append("**Range**")
                lines.append(f"- {a}")
                lines.append(f"- {b}")

        if needs_confirm:
            embed = discord.Embed(
                title="❗ Confirm purge",
                color=0xED4245,
                description="\n".join([intro, "", *lines, "", "Shall I proceed?"]),
            )
            view = ConfirmView(ctx.author.id)
            prompt = await safe_reply(
                ctx, embed=embed, view=view, ephemeral=True, mention_author=False
            )
            await view.wait()

            if not view.result:
                if ctx.message:
                    with contextlib.suppress(Exception):
                        await ctx.message.delete()
                with contextlib.suppress(Exception):
                    await prompt.delete()
                return

            if ctx.message:
                with contextlib.suppress(Exception):
                    await ctx.message.delete()
            with contextlib.suppress(Exception):
                await prompt.delete()

        deleted = await _delete_by_ids(
            channel,
            target_ids,
            reason=f"Purge invoked by {ctx.author} ({ctx.author.id})",
        )

        if ctx.interaction:
            now = int(discord.utils.utcnow().timestamp())
            msg = await ctx.interaction.followup.send(
                f"✅ Purge complete — {deleted} message(s)\n"
                f"<t:{now}> · <t:{now}:R> · by {ctx.author.mention}",
                ephemeral=True,
            )
            await asyncio.sleep(5)
            with contextlib.suppress(Exception):
                await msg.delete()
        else:
            await ctx.send(f"Deleted {deleted} message(s).", delete_after=5)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Purge(bot))
