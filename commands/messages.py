"""Command to fetch recent messages in the current channel."""

from __future__ import annotations

import contextlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import discord
from discord.ext import commands

from utils import BOT_PREFIX, defer_interaction, sanitize, tag_error_text

log = logging.getLogger(__name__)
LINK_RE = re.compile(r"https?://", re.IGNORECASE)
MENTION_ID_RE = re.compile(r"<@!?(\d+)>")
MESSAGE_LINK_RE = re.compile(
    r"https?://(?:ptb\.|canary\.)?discord(?:app)?\.com/channels/"
    r"(?:@me|\d+)/\d+/(?P<message_id>\d+)",
    re.IGNORECASE,
)


@dataclass
class MessageQuery:
    limit: int = 50
    scan_limit: int | None = None
    keywords: list[str] = field(default_factory=list)
    from_ids: set[int] = field(default_factory=set)
    from_names: set[str] = field(default_factory=set)
    mention_ids: set[int] = field(default_factory=set)
    mention_names: set[str] = field(default_factory=set)
    has_filters: set[str] = field(default_factory=set)
    pinned: bool | None = None
    after: datetime | None = None
    before: datetime | None = None
    during: datetime | None = None
    after_id: int | None = None
    before_id: int | None = None


def _parse_date(value: str) -> datetime | None:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc)


def _normalize_user_token(raw: str) -> tuple[int | None, str | None]:
    if not raw:
        return None, None
    raw = raw.strip("<>.,)")
    raw = raw.lstrip("@")
    match = MENTION_ID_RE.match(raw)
    if match:
        return int(match.group(1)), None
    if raw.isdigit():
        return int(raw), None
    return None, raw.casefold()


def _parse_message_id(value: str) -> int | None:
    cleaned = value.strip("<>.,)")
    match = MESSAGE_LINK_RE.match(cleaned)
    if match:
        return int(match.group("message_id"))
    if cleaned.isdigit():
        return int(cleaned)
    return None


def _parse_query(raw: str | None) -> MessageQuery:
    query = MessageQuery()
    if not raw:
        return query

    tokens = raw.split()
    for index, token in enumerate(tokens):
        if index == 0 and token.isdigit():
            query.limit = int(token)
            continue

        if ":" not in token:
            query.keywords.append(token)
            continue

        key, value = token.split(":", 1)
        key = key.casefold()
        value = value.strip()

        if not value:
            continue

        if key == "from":
            user_id, name = _normalize_user_token(value)
            if user_id is not None:
                query.from_ids.add(user_id)
            elif name:
                query.from_names.add(name)
            continue

        if key == "mentions":
            user_id, name = _normalize_user_token(value)
            if user_id is not None:
                query.mention_ids.add(user_id)
            elif name:
                query.mention_names.add(name)
            continue

        if key == "has":
            query.has_filters.add(value.casefold())
            continue

        if key == "scan" and value.isdigit():
            query.scan_limit = int(value)
            continue

        if key == "before_id":
            message_id = _parse_message_id(value)
            if message_id is not None:
                query.before_id = message_id
            continue

        if key == "after_id":
            message_id = _parse_message_id(value)
            if message_id is not None:
                query.after_id = message_id
            continue

        if key in {"before", "after", "during"}:
            parsed = _parse_date(value)
            if parsed is None:
                continue
            if key == "before":
                query.before = parsed
            elif key == "after":
                query.after = parsed
            else:
                query.during = parsed
            continue

        if key == "pinned":
            lowered = value.casefold()
            if lowered in {"true", "yes", "1"}:
                query.pinned = True
            elif lowered in {"false", "no", "0"}:
                query.pinned = False
            continue

        if key == "in":
            continue

        query.keywords.append(token)

    return query


def _attachment_has_type(
    attachment: discord.Attachment, *, prefixes: tuple[str, ...], extensions: tuple[str, ...]
) -> bool:
    content_type = (attachment.content_type or "").casefold()
    if any(content_type.startswith(prefix) for prefix in prefixes):
        return True
    suffix = Path(getattr(attachment, "filename", "")).suffix.casefold()
    return bool(suffix and suffix in extensions)


def _message_has_filter(message: discord.Message, has_filters: set[str]) -> bool:
    if not has_filters:
        return True

    content = message.content or ""
    attachments = message.attachments or []
    embeds = message.embeds or []

    def _embed_has_link(embed: discord.Embed) -> bool:
        if LINK_RE.search(embed.url or ""):
            return True
        if LINK_RE.search(embed.description or ""):
            return True
        for field in getattr(embed, "fields", []):
            if LINK_RE.search(field.name or "") or LINK_RE.search(field.value or ""):
                return True
        return False

    def _match_has(filter_name: str) -> bool:
        if filter_name in {"link", "links"}:
            if LINK_RE.search(content):
                return True
            return any(_embed_has_link(embed) for embed in embeds)
        if filter_name in {"embed", "embeds"}:
            return bool(embeds)
        if filter_name in {"file", "files"}:
            return bool(attachments)
        if filter_name in {"image", "images"}:
            if any(
                _attachment_has_type(
                    att,
                    prefixes=("image/",),
                    extensions=(".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"),
                )
                for att in attachments
            ):
                return True
            return any(getattr(getattr(embed, "image", None), "url", None) for embed in embeds)
        if filter_name in {"video", "videos"}:
            return any(
                _attachment_has_type(
                    att,
                    prefixes=("video/",),
                    extensions=(".mp4", ".mov", ".mkv", ".webm", ".avi"),
                )
                for att in attachments
            )
        if filter_name in {"sound", "audio", "sounds"}:
            return any(
                _attachment_has_type(
                    att,
                    prefixes=("audio/",),
                    extensions=(".mp3", ".wav", ".flac", ".ogg", ".m4a"),
                )
                for att in attachments
            )
        if filter_name in {"poll", "polls"}:
            return bool(getattr(message, "poll", None))
        return False

    return all(_match_has(name) for name in has_filters)


def _message_matches_query(message: discord.Message, query: MessageQuery) -> bool:
    if query.from_ids or query.from_names:
        author = message.author
        author_id = getattr(author, "id", None)
        author_name = (
            getattr(author, "display_name", None)
            or getattr(author, "name", None)
            or ""
        ).casefold()
        if query.from_ids and author_id not in query.from_ids:
            return False
        if query.from_names and author_name not in query.from_names:
            if not any(name in author_name for name in query.from_names):
                return False

    if query.mention_ids or query.mention_names:
        mentions = message.mentions or []
        mention_ids = {u.id for u in mentions}
        mention_names = {
            (getattr(u, "display_name", None) or getattr(u, "name", "")).casefold()
            for u in mentions
        }
        if query.mention_ids and not (mention_ids & query.mention_ids):
            return False
        if query.mention_names and not (mention_names & query.mention_names):
            if not any(
                target in mention
                for target in query.mention_names
                for mention in mention_names
            ):
                return False

    if query.pinned is not None and message.pinned != query.pinned:
        return False

    created_at = message.created_at
    if query.after and created_at < query.after:
        return False
    if query.before and created_at >= query.before:
        return False
    if query.during:
        target_date = query.during.date()
        if created_at.astimezone(timezone.utc).date() != target_date:
            return False

    if query.keywords:
        text = (message.content or "").casefold()
        if not all(keyword.casefold() in text for keyword in query.keywords):
            return False

    return _message_has_filter(message, query.has_filters)


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
            "Retrieve and filter messages in the current channel. Provide a number "
            "to change how many are shown (defaults to 50, min 1 / max 50). You can also "
            "use search filters like from:, mentions:, has:, before:, after:, during:, "
            "before_id:, after_id:, pinned:true/false, and scan: (max history to scan). "
            "during: uses the server timezone (set via /settime) and ignores before_id:/after_id. "
            "Text outside filters is "
            "treated as search keywords.\n\n"
            "**Usage**: `/messages [query]`\n"
            "**Examples**: `/messages`, `/messages 5`, `/messages おはよう from:1234`, "
            "`/messages has:link before:2026-01-01`, `/messages scan:1000 has:image`, "
            "`/messages before_id:1234567890 scan:2000`\n"
            f"`{BOT_PREFIX}messages 15 from:@user`"
        ),
        extras={
            "category": "Utility",
            "pro": (
                "If no count is provided the command shows 50 messages by default. "
                "The count is clamped to a maximum of 50. Query filters like from:, "
                "mentions:, has:, before:, after:, during:, before_id:, after_id:, "
                "pinned:true/false, and scan: are supported. Long outputs are trimmed to "
                "fit Discord's embed limits."
            ),
        },
    )
    async def messages(
        self, ctx: commands.Context, query: str | None = None
    ) -> None:  # type: ignore[override]
        await defer_interaction(ctx)
        try:
            parsed = _parse_query(query)
            amount = max(1, min(50, int(parsed.limit)))

            if not hasattr(ctx.channel, "history"):
                error_message = tag_error_text("This channel type is not supported.")
                if ctx.interaction and ctx.interaction.response.is_done():
                    await ctx.interaction.followup.send(
                        error_message,
                        ephemeral=True,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                elif ctx.interaction:
                    await ctx.interaction.response.send_message(
                        error_message,
                        ephemeral=True,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                else:
                    await ctx.reply(
                        error_message,
                        mention_author=False,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                return

            if parsed.during:
                tz = timezone.utc
                if ctx.guild is not None:
                    with contextlib.suppress(Exception):
                        from cogs.settime import get_guild_offset

                        tz = timezone(timedelta(hours=get_guild_offset(self.bot, ctx.guild.id)))
                local_start = datetime.combine(parsed.during.date(), datetime.min.time(), tzinfo=tz)
                parsed.after = local_start.astimezone(timezone.utc)
                parsed.before = (local_start + timedelta(days=1)).astimezone(timezone.utc)
                parsed.during = None
                parsed.after_id = None
                parsed.before_id = None

            use_filters = bool(
                parsed.keywords
                or parsed.from_ids
                or parsed.from_names
                or parsed.mention_ids
                or parsed.mention_names
                or parsed.has_filters
                or parsed.pinned is not None
                or parsed.after
                or parsed.before
                or parsed.after_id
                or parsed.before_id
                or parsed.scan_limit
            )
            scan_limit = amount
            if use_filters:
                requested_scan = parsed.scan_limit or 0
                scan_limit = min(5000, max(amount, 200, requested_scan))

            history_kwargs: dict[str, object] = {"limit": scan_limit}
            if parsed.after_id:
                history_kwargs["after"] = discord.Object(id=parsed.after_id)
            elif parsed.after:
                history_kwargs["after"] = parsed.after
            if parsed.before_id:
                history_kwargs["before"] = discord.Object(id=parsed.before_id)
            elif parsed.before:
                history_kwargs["before"] = parsed.before

            history = [m async for m in ctx.channel.history(**history_kwargs)]
            history.reverse()

            if use_filters:
                history = [m for m in history if _message_matches_query(m, parsed)]

            if history:
                selected = history[-amount:]
                ctx.recent_message_links = [
                    {"url": m.jump_url, "kind": "message_link"} for m in selected if m.jump_url
                ]
            else:
                selected = []

            if not selected:
                content = "No recent messages found."
            else:
                content = _format_messages(selected)

            embed = discord.Embed(
                title="\U0001F4DD Recent Messages",
                description=content,
                color=0x42A5F5,
            )
            embed.set_footer(text=f"Showing {len(selected)} message(s)")

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
            message = tag_error_text("I don't have permission to read messages in this channel.")
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
            message = tag_error_text("Discord rejected the response. Please try a smaller count.")
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
            message = tag_error_text("Could not fetch messages right now.")
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
