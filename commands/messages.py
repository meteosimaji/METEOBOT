"""Command to fetch recent messages in the current channel."""

from __future__ import annotations

import contextlib
import difflib
import logging
import re
import shlex
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
ROLE_ID_RE = re.compile(r"<@&(\d+)>")
MESSAGE_LINK_RE = re.compile(
    r"https?://(?:ptb\.|canary\.)?discord(?:app)?\.com/channels/"
    r"(?:@me|\d+)/\d+/(?P<message_id>\d+)",
    re.IGNORECASE,
)
CHANNEL_LINK_RE = re.compile(
    r"https?://(?:ptb\.|canary\.)?discord(?:app)?\.com/channels/"
    r"(?:@me|\d+)/(?P<channel_id>\d+)",
    re.IGNORECASE,
)
CHANNEL_MENTION_RE = re.compile(r"<#(\d+)>")


@dataclass
class MessageQuery:
    limit: int = 50
    scan_limit: int | None = None
    keywords: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)
    from_ids: set[int] = field(default_factory=set)
    from_names: set[str] = field(default_factory=set)
    exclude_from_ids: set[int] = field(default_factory=set)
    exclude_from_names: set[str] = field(default_factory=set)
    mention_ids: set[int] = field(default_factory=set)
    mention_names: set[str] = field(default_factory=set)
    exclude_mention_ids: set[int] = field(default_factory=set)
    exclude_mention_names: set[str] = field(default_factory=set)
    mention_role_ids: set[int] = field(default_factory=set)
    exclude_mention_role_ids: set[int] = field(default_factory=set)
    role_ids: set[int] = field(default_factory=set)
    role_names: set[str] = field(default_factory=set)
    exclude_role_ids: set[int] = field(default_factory=set)
    exclude_role_names: set[str] = field(default_factory=set)
    has_filters: set[str] = field(default_factory=set)
    exclude_has_filters: set[str] = field(default_factory=set)
    pinned: bool | None = None
    author_is_bot: bool | None = None
    exclude_author_is_bot: bool | None = None
    after: datetime | None = None
    before: datetime | None = None
    during: datetime | None = None
    after_is_date: bool = False
    before_is_date: bool = False
    during_is_date: bool = False
    after_id: int | None = None
    before_id: int | None = None
    in_channel_ids: set[int] = field(default_factory=set)
    in_channel_names: list[str] = field(default_factory=list)
    in_provided: bool = False
    exclude_in_channel_ids: set[int] = field(default_factory=set)
    exclude_in_channel_names: list[str] = field(default_factory=list)
    exclude_in_provided: bool = False
    scope: str | None = None
    scope_provided: bool = False
    scope_invalid: str | None = None
    server_ids: set[int] = field(default_factory=set)
    server_names: list[str] = field(default_factory=list)
    server_provided: bool = False
    scope_category_ids: set[int] = field(default_factory=set)
    scope_category_names: list[str] = field(default_factory=list)


def _parse_date(value: str) -> tuple[datetime | None, bool]:
    if value.isdigit():
        try:
            timestamp = int(value)
            if len(value) >= 13:
                timestamp = timestamp // 1000
            return datetime.fromtimestamp(timestamp, tz=timezone.utc), False
        except (OverflowError, OSError, ValueError):
            return None, False
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None, False
    return parsed.replace(tzinfo=timezone.utc), True


def _extract_mention_id(raw: str, *, mention_prefixes: tuple[str, ...]) -> int | None:
    if not raw:
        return None
    cleaned = raw.strip().strip("<>.,)")
    if cleaned.startswith("<@") and cleaned.endswith(">"):
        inner = cleaned[2:-1]
        for prefix in mention_prefixes:
            if prefix and inner.startswith(prefix):
                inner = inner[len(prefix) :]
                break
        if inner.isdigit():
            return int(inner)
    cleaned = cleaned.lstrip("@")
    for prefix in mention_prefixes:
        if prefix:
            cleaned = cleaned.lstrip(prefix)
    if cleaned.isdigit():
        return int(cleaned)
    return None


def _normalize_user_token(raw: str) -> tuple[int | None, str | None]:
    if not raw:
        return None, None
    mention_id = _extract_mention_id(raw, mention_prefixes=("!",))
    if mention_id is None:
        mention_id = _extract_mention_id(raw, mention_prefixes=("",))
    if mention_id is not None:
        return mention_id, None
    cleaned = raw.strip("<>.,)")
    cleaned = cleaned.lstrip("@")
    return None, cleaned.casefold()


def _normalize_role_token(raw: str) -> tuple[int | None, str | None]:
    if not raw:
        return None, None
    stripped = raw.strip()
    match = ROLE_ID_RE.match(stripped)
    if match:
        return int(match.group(1)), None
    cleaned = raw.strip("<>.,)")
    if cleaned.startswith("@&") or cleaned.startswith("&"):
        role_id = cleaned.lstrip("@&")
        if role_id.isdigit():
            return int(role_id), None
    cleaned = cleaned.lstrip("@")
    cleaned = cleaned.lstrip("&")
    return None, cleaned.casefold()


def _parse_role_mention_id(raw: str) -> int | None:
    if not raw:
        return None
    raw = raw.strip("<>.,)")
    match = ROLE_ID_RE.match(raw)
    if match:
        return int(match.group(1))
    if raw.startswith("@&") or raw.startswith("&"):
        cleaned = raw.lstrip("@&")
        if cleaned.isdigit():
            return int(cleaned)
    return None


def _parse_message_id(value: str) -> int | None:
    cleaned = value.strip("<>.,)")
    match = MESSAGE_LINK_RE.match(cleaned)
    if match:
        return int(match.group("message_id"))
    if cleaned.isdigit():
        return int(cleaned)
    return None


def _parse_channel_id(value: str) -> int | None:
    raw = value.strip().strip("<>.,)")
    mention_match = CHANNEL_MENTION_RE.match(raw)
    if mention_match:
        return int(mention_match.group(1))
    cleaned = raw.strip("<>")
    if cleaned.startswith("#"):
        cleaned = cleaned[1:]
    link_match = CHANNEL_LINK_RE.match(cleaned)
    if link_match:
        return int(link_match.group("channel_id"))
    if cleaned.isdigit():
        return int(cleaned)
    return None


def _normalize_channel_name(raw: str) -> str:
    token = (raw or "").strip()
    if token.startswith("#"):
        token = token[1:]
    return token.casefold()


def _normalize_category_name(raw: str) -> str:
    return (raw or "").strip().casefold()


def _parse_query(raw: str | None) -> MessageQuery:
    query = MessageQuery()
    if not raw:
        return query

    tokens = shlex.split(raw, comments=False, posix=True)
    for index, token in enumerate(tokens):
        if index == 0 and token.isdigit():
            query.limit = int(token)
            continue

        if ":" not in token:
            query.keywords.append(token)
            continue

        key, value = token.split(":", 1)
        key = key.casefold()
        negated = False
        if key.startswith("!"):
            negated = True
            key = key[1:]
        value = value.strip()

        if not value:
            continue

        if key == "from":
            role_id = _parse_role_mention_id(value)
            if role_id is not None:
                if negated:
                    query.exclude_role_ids.add(role_id)
                else:
                    query.role_ids.add(role_id)
                continue
            user_id, name = _normalize_user_token(value)
            if user_id is not None:
                if negated:
                    query.exclude_from_ids.add(user_id)
                else:
                    query.from_ids.add(user_id)
            elif name:
                if negated:
                    query.exclude_from_names.add(name)
                else:
                    query.from_names.add(name)
            continue

        if key == "mentions":
            role_id = _parse_role_mention_id(value)
            if role_id is not None:
                if negated:
                    query.exclude_mention_role_ids.add(role_id)
                else:
                    query.mention_role_ids.add(role_id)
                continue
            user_id, name = _normalize_user_token(value)
            if user_id is not None:
                if negated:
                    query.exclude_mention_ids.add(user_id)
                else:
                    query.mention_ids.add(user_id)
            elif name:
                if negated:
                    query.exclude_mention_names.add(name)
                else:
                    query.mention_names.add(name)
            continue

        if key == "role":
            role_id, name = _normalize_role_token(value)
            if role_id is not None:
                if negated:
                    query.exclude_role_ids.add(role_id)
                else:
                    query.role_ids.add(role_id)
            elif name:
                if negated:
                    query.exclude_role_names.add(name)
                else:
                    query.role_names.add(name)
            continue

        if key == "has":
            if negated:
                query.exclude_has_filters.add(value.casefold())
            else:
                query.has_filters.add(value.casefold())
            continue

        if key == "bot":
            lowered = value.casefold()
            if lowered in {"true", "yes", "1"}:
                if negated:
                    query.exclude_author_is_bot = True
                else:
                    query.author_is_bot = True
            elif lowered in {"false", "no", "0"}:
                if negated:
                    query.exclude_author_is_bot = False
                else:
                    query.author_is_bot = False
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
            parsed, is_date = _parse_date(value)
            if parsed is None:
                continue
            if key == "before":
                query.before = parsed
                query.before_is_date = is_date
            elif key == "after":
                query.after = parsed
                query.after_is_date = is_date
            else:
                query.during = parsed
                query.during_is_date = is_date
            continue

        if key == "pinned":
            lowered = value.casefold()
            if lowered in {"true", "yes", "1"}:
                query.pinned = True
            elif lowered in {"false", "no", "0"}:
                query.pinned = False
            continue

        if key == "keyword":
            if negated:
                query.exclude_keywords.append(value)
            else:
                query.keywords.append(value)
            continue

        if key == "in":
            if negated:
                query.exclude_in_provided = True
            else:
                query.in_provided = True
            parts = [p.strip() for p in value.split(",") if p.strip()]
            for part in parts:
                channel_id = _parse_channel_id(part)
                if channel_id is not None:
                    if negated:
                        query.exclude_in_channel_ids.add(channel_id)
                    else:
                        query.in_channel_ids.add(channel_id)
                else:
                    if negated:
                        query.exclude_in_channel_names.append(part)
                    else:
                        query.in_channel_names.append(part)
            continue

        if key == "scope":
            query.scope_provided = True
            scope_value = value.casefold()
            if scope_value in {"all", "global"}:
                query.scope = scope_value
                continue
            if scope_value.startswith("category="):
                query.scope = "category"
                raw_category = scope_value.split("category=", 1)[1].strip()
                if raw_category.isdigit():
                    query.scope_category_ids.add(int(raw_category))
                elif raw_category:
                    query.scope_category_names.append(raw_category)
                continue
            query.scope_invalid = value
            continue

        if key == "server":
            query.server_provided = True
            parts = [p.strip() for p in value.split(",") if p.strip()]
            for part in parts:
                if part.isdigit():
                    query.server_ids.add(int(part))
                else:
                    query.server_names.append(part)
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


def _message_has_any_filter(message: discord.Message, has_filters: set[str]) -> bool:
    if not has_filters:
        return False
    return any(_message_has_filter(message, {name}) for name in has_filters)


def _keyword_matches_single(text: str, keyword: str) -> bool:
    if not keyword:
        return True
    if "*" not in keyword:
        return keyword in text
    pattern = re.escape(keyword).replace(r"\*", ".*")
    return bool(re.search(pattern, text, flags=re.DOTALL))


def _keyword_matches(text: str, keyword: str) -> bool:
    if "|" in keyword:
        parts = [part for part in keyword.split("|") if part]
        if not parts:
            return False
        return any(_keyword_matches_single(text, part) for part in parts)
    return _keyword_matches_single(text, keyword)


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

    if query.exclude_from_ids or query.exclude_from_names:
        author = message.author
        author_id = getattr(author, "id", None)
        author_name = (
            getattr(author, "display_name", None)
            or getattr(author, "name", None)
            or ""
        ).casefold()
        if query.exclude_from_ids and author_id in query.exclude_from_ids:
            return False
        if query.exclude_from_names and author_name in query.exclude_from_names:
            return False
        if query.exclude_from_names and any(name in author_name for name in query.exclude_from_names):
            return False

    if query.mention_ids or query.mention_names or query.mention_role_ids:
        mentions = message.mentions or []
        mention_ids = {u.id for u in mentions}
        mention_names = {
            (getattr(u, "display_name", None) or getattr(u, "name", "")).casefold()
            for u in mentions
        }
        mention_role_ids = {r.id for r in getattr(message, "role_mentions", []) or []}
        if query.mention_ids and not (mention_ids & query.mention_ids):
            return False
        if query.mention_names and not (mention_names & query.mention_names):
            if not any(
                target in mention
                for target in query.mention_names
                for mention in mention_names
            ):
                return False
        if query.mention_role_ids and not (mention_role_ids & query.mention_role_ids):
            return False

    if query.exclude_mention_ids or query.exclude_mention_names or query.exclude_mention_role_ids:
        mentions = message.mentions or []
        mention_ids = {u.id for u in mentions}
        mention_names = {
            (getattr(u, "display_name", None) or getattr(u, "name", "")).casefold()
            for u in mentions
        }
        mention_role_ids = {r.id for r in getattr(message, "role_mentions", []) or []}
        if query.exclude_mention_ids and (mention_ids & query.exclude_mention_ids):
            return False
        if query.exclude_mention_names and (mention_names & query.exclude_mention_names):
            return False
        if query.exclude_mention_names and any(
            target in mention
            for target in query.exclude_mention_names
            for mention in mention_names
            ):
                return False
        if query.exclude_mention_role_ids and (mention_role_ids & query.exclude_mention_role_ids):
            return False

    has_role_filters = bool(
        query.role_ids
        or query.role_names
        or query.exclude_role_ids
        or query.exclude_role_names
    )
    member = None
    if has_role_filters:
        member = message.author if isinstance(message.author, discord.Member) else None
        if member is None:
            guild = getattr(message, "guild", None)
            if guild is not None:
                member = guild.get_member(getattr(message.author, "id", 0))
        if member is None:
            return False

    if query.role_ids or query.role_names:
        if member is None:
            return False
        member_roles = getattr(member, "roles", []) or []
        role_ids = {getattr(role, "id", None) for role in member_roles}
        role_names = {getattr(role, "name", "").casefold() for role in member_roles}
        if query.role_ids and not (role_ids & query.role_ids):
            return False
        if query.role_names and not (role_names & query.role_names):
            if not any(name in role_name for name in query.role_names for role_name in role_names):
                return False

    if query.exclude_role_ids or query.exclude_role_names:
        if member is not None:
            member_roles = getattr(member, "roles", []) or []
            role_ids = {getattr(role, "id", None) for role in member_roles}
            role_names = {getattr(role, "name", "").casefold() for role in member_roles}
            if query.exclude_role_ids and (role_ids & query.exclude_role_ids):
                return False
            if query.exclude_role_names and (role_names & query.exclude_role_names):
                return False
            if query.exclude_role_names and any(
                name in role_name for name in query.exclude_role_names for role_name in role_names
            ):
                return False

    if query.author_is_bot is not None and message.author.bot != query.author_is_bot:
        return False

    if query.exclude_author_is_bot is not None and message.author.bot == query.exclude_author_is_bot:
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
        if not all(_keyword_matches(text, keyword.casefold()) for keyword in query.keywords):
            return False

    if query.exclude_keywords:
        text = (message.content or "").casefold()
        if any(_keyword_matches(text, keyword.casefold()) for keyword in query.exclude_keywords):
            return False

    if query.exclude_has_filters and _message_has_any_filter(message, query.exclude_has_filters):
        return False

    return _message_has_filter(message, query.has_filters)


def _format_messages(
    messages: Iterable[discord.Message],
    *,
    limit: int = 3900,
    show_channel: bool = False,
    show_guild: bool = False,
) -> str:
    lines: list[str] = []
    total_length = 0
    truncated = False

    for msg in messages:
        channel_label = ""
        if show_channel:
            channel = getattr(msg, "channel", None)
            channel_name = getattr(channel, "name", None) or "unknown-channel"
            if show_guild:
                guild = getattr(getattr(channel, "guild", None), "name", None) or "unknown-server"
                channel_label = f"[{sanitize(guild)} #{sanitize(channel_name)}] "
            else:
                channel_label = f"[#{sanitize(channel_name)}] "
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
        line = f"• {channel_label}<t:{ts}:t> {author}: {snippet}"
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
            "use search filters like from:, mentions:, role:, has:, keyword:, bot:, before:, after:, during:, "
            "before_id:, after_id:, in:, scope:, pinned:true/false, and scan: (max history to scan; "
            "defaults to all history when filters are used, so scan: is optional). Prefix filters with "
            "! to exclude matches (e.g. !from:, !mentions:, !role:, !has:, !keyword:, !bot:, !in:). "
            "during: uses the server timezone (set via /settime) and ignores before_id:/after_id. "
            "in: accepts channel mentions/IDs/links across servers; plain channel names "
            "only resolve within the current server and may require disambiguation. "
            "scope: supports all, global, or category=<id|name> to scan across multiple "
            "channels (use scope: or in:, not both; !in: may be combined with scope: to "
            "exclude channels). Add server:<id|name> with scope:all "
            "or scope:category to target a specific server by ID or name. Multi-channel results include "
            "the channel (and server for scope:global) next to each message. "
            "Text outside filters is "
            "treated as search keywords (supports * wildcard, | for OR, and quoted phrases).\n\n"
            "**Usage**: `/messages [query]`\n"
            "**Examples**: `/messages`, `/messages 5`, `/messages おはよう from:1234`, "
            "`/messages has:link before:2026-01-01`, `/messages scan:1000 has:image`, "
            "`/messages before_id:1234567890 scan:2000`, `/messages in:<#1234567890>`, "
            "`/messages !in:<#1234567890> keyword:err*`, `/messages from:<@&1234567890>`, "
            "`/messages mentions:<@&1234567890>`, `/messages bot:true role:mods`, "
            "`/messages scope:all`\n"
            f"`{BOT_PREFIX}messages 15 from:@user`"
        ),
        extras={
            "category": "Utility",
            "pro": (
                "If no count is provided the command shows 50 messages by default. "
                "The count is clamped to a maximum of 50. Query filters like from:, "
            "mentions:, role:, has:, keyword:, bot:, before:, after:, during:, before_id:, after_id:, "
            "in:, scope:, pinned:true/false, and scan: are supported (in: accepts multiple values "
            "separated by commas; from:/mentions: accept role mentions like <@&id>). Use scope: or "
            "in:, not both (but !in: can be combined "
            "with scope: to exclude channels). Prefix filters with ! to exclude matches "
            "(for example !from:, !mentions:, !role:, !has:, !keyword:, !bot:, !in:). Keyword "
            "matching supports * wildcards, | for OR, and quoted phrases; text outside filters is "
            "treated as keyword search too. "
            "When filters are used without scan:, the command scans all available history, so "
            "scan: is optional. Use server:<id|name> with scope:all or scope:category to target a "
            "specific server. Long outputs are trimmed to fit Discord's embed limits."
        ),
        },
    )
    async def messages(
        self, ctx: commands.Context, query: str | None = None
    ) -> None:  # type: ignore[override]
        await defer_interaction(ctx)
        try:
            try:
                parsed = _parse_query(query)
            except ValueError:
                error_message = tag_error_text(
                    "Invalid query: please check for an unclosed quote."
                )
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
            amount = max(1, min(50, int(parsed.limit)))

            tz = timezone.utc
            if ctx.guild is not None:
                with contextlib.suppress(Exception):
                    from cogs.settime import get_guild_offset

                    tz = timezone(timedelta(hours=get_guild_offset(self.bot, ctx.guild.id)))

            if parsed.before and parsed.before_is_date:
                local_before = datetime.combine(
                    parsed.before.date(), datetime.min.time(), tzinfo=tz
                )
                parsed.before = local_before.astimezone(timezone.utc)

            if parsed.after and parsed.after_is_date:
                local_after = datetime.combine(
                    parsed.after.date(), datetime.min.time(), tzinfo=tz
                )
                parsed.after = local_after.astimezone(timezone.utc)

            if parsed.during:
                if parsed.during_is_date:
                    local_date = parsed.during.date()
                else:
                    local_date = parsed.during.astimezone(tz).date()
                local_start = datetime.combine(local_date, datetime.min.time(), tzinfo=tz)
                parsed.after = local_start.astimezone(timezone.utc)
                parsed.before = (local_start + timedelta(days=1)).astimezone(timezone.utc)
                parsed.during = None
                parsed.after_id = None
                parsed.before_id = None

            has_role_filters = bool(
                parsed.role_ids
                or parsed.role_names
                or parsed.exclude_role_ids
                or parsed.exclude_role_names
            )
            if has_role_filters and not self.bot.intents.members:
                resolve_errors = [
                    "Role filters may be incomplete because the bot lacks the Members intent."
                ]
            else:
                resolve_errors = []

            if parsed.scope and (parsed.in_channel_ids or parsed.in_channel_names):
                error_message = tag_error_text(
                    "Use scope: or in: filters, not both in the same query (use !in: with scope: "
                    "to exclude channels)."
                )
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

            if parsed.scope_provided and parsed.scope is None:
                error_message = tag_error_text(
                    "Unknown scope value. Use scope:all, scope:global, or scope:category=<id|name>."
                )
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

            if parsed.in_provided and not (parsed.in_channel_ids or parsed.in_channel_names):
                error_message = tag_error_text(
                    "in: requires at least one channel mention, ID, link, or name."
                )
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

            if parsed.exclude_in_provided and not (
                parsed.exclude_in_channel_ids or parsed.exclude_in_channel_names
            ):
                error_message = tag_error_text(
                    "!in: requires at least one channel mention, ID, link, or name."
                )
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

            target_guild: discord.Guild | None = None
            if parsed.server_provided:
                if parsed.scope not in {"all", "category"}:
                    error_message = tag_error_text(
                        "server: can only be used with scope:all or scope:category."
                    )
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

                guild_candidates: list[discord.Guild] = []
                if parsed.server_ids:
                    for guild_id in parsed.server_ids:
                        guild = self.bot.get_guild(guild_id)
                        if guild is not None:
                            guild_candidates.append(guild)
                if parsed.server_names:
                    name_targets = {name.casefold() for name in parsed.server_names}
                    for guild in self.bot.guilds:
                        if guild.name.casefold() in name_targets:
                            guild_candidates.append(guild)

                unique_candidates = list({guild.id: guild for guild in guild_candidates}.values())
                if not unique_candidates:
                    error_message = tag_error_text(
                        "server: did not match any servers I can access."
                    )
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
                if len(unique_candidates) > 1:
                    shown = ", ".join(f"{guild.name} ({guild.id})" for guild in unique_candidates[:3])
                    error_message = tag_error_text(
                        "server: matched multiple servers. Use an ID to disambiguate. "
                        f"Examples: {shown}"
                    )
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
                target_guild = next(iter(unique_candidates))

            if parsed.in_channel_names or parsed.exclude_in_channel_names:
                if ctx.guild is None:
                    error_message = tag_error_text(
                        "in: or !in: with a channel name only works inside a server. "
                        "Use a channel mention (<#...>), ID, or link."
                    )
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

                searchable_channels: list[discord.abc.GuildChannel] = []
                searchable_channels.extend(getattr(ctx.guild, "text_channels", []))
                searchable_channels.extend(getattr(ctx.guild, "threads", []))

                name_index: dict[str, list[discord.abc.GuildChannel]] = {}
                for channel in searchable_channels:
                    channel_name = getattr(channel, "name", None)
                    if not channel_name:
                        continue
                    name_index.setdefault(channel_name.casefold(), []).append(channel)

                def _resolve_channel_names(
                    raw_names: list[str],
                    *,
                    target_ids: set[int],
                    label: str,
                ) -> list[str]:
                    unresolved: list[str] = []
                    ambiguous: list[tuple[str, list[discord.abc.GuildChannel]]] = []

                    for raw_name in raw_names:
                        normalized = _normalize_channel_name(raw_name)
                        if not normalized:
                            unresolved.append(raw_name)
                            continue
                        matches = name_index.get(normalized, [])
                        if len(matches) == 1:
                            target_ids.add(matches[0].id)
                        elif len(matches) == 0:
                            unresolved.append(raw_name)
                        else:
                            ambiguous.append((raw_name, matches))

                    parts: list[str] = []

                    if ambiguous:
                        shown: list[str] = []
                        for raw_name, matches in ambiguous[:3]:
                            candidates: list[str] = []
                            for channel in matches[:5]:
                                mention = getattr(channel, "mention", f"<#{channel.id}>")
                                category = getattr(getattr(channel, "category", None), "name", None)
                                candidates.append(f"{mention} ({category})" if category else mention)
                            shown.append(f"{raw_name} -> {', '.join(candidates)}")
                        parts.append(
                            f"Ambiguous {label}: channel name(s). Use a channel mention/ID/link to "
                            f"disambiguate. {' | '.join(shown)}"
                        )

                    if unresolved:
                        all_names = sorted(name_index.keys())
                        suggestions: list[str] = []
                        for raw_name in unresolved[:3]:
                            normalized = _normalize_channel_name(raw_name)
                            if not normalized:
                                continue
                            close = difflib.get_close_matches(normalized, all_names, n=3, cutoff=0.6)
                            for candidate in close:
                                channels = name_index.get(candidate, [])
                                if channels:
                                    suggestions.append(
                                        getattr(channels[0], "mention", f"<#{channels[0].id}>")
                                    )
                        uniq: list[str] = []
                        for suggestion in suggestions:
                            if suggestion not in uniq:
                                uniq.append(suggestion)
                            if len(uniq) >= 5:
                                break
                        if uniq:
                            parts.append(
                                f"Unknown {label}: channel name(s): "
                                f"{', '.join(unresolved[:5])}. Did you mean: {', '.join(uniq)}"
                            )
                        else:
                            parts.append(
                                f"Unknown {label}: channel name(s): "
                                f"{', '.join(unresolved[:5])}."
                            )

                    return parts

                in_parts = _resolve_channel_names(
                    parsed.in_channel_names,
                    target_ids=parsed.in_channel_ids,
                    label="in",
                )
                exclude_parts = _resolve_channel_names(
                    parsed.exclude_in_channel_names,
                    target_ids=parsed.exclude_in_channel_ids,
                    label="!in",
                )

                if in_parts or exclude_parts:
                    error_message = tag_error_text(" ".join(in_parts + exclude_parts))
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

            if parsed.scope == "category":
                if ctx.guild is None:
                    error_message = tag_error_text(
                        "scope:category only works inside a server. Use scope:global or in: instead."
                    )
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

                categories: list[discord.CategoryChannel] = []
                if parsed.scope_category_ids:
                    for category_id in parsed.scope_category_ids:
                        category = ctx.guild.get_channel(category_id)
                        if isinstance(category, discord.CategoryChannel):
                            categories.append(category)
                    if not categories:
                        error_message = tag_error_text("scope:category did not match any categories.")
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
                elif parsed.scope_category_names:
                    normalized_targets = {_normalize_category_name(name) for name in parsed.scope_category_names}
                    for category in ctx.guild.categories:
                        if _normalize_category_name(category.name) in normalized_targets:
                            categories.append(category)
                    if not categories:
                        error_message = tag_error_text("scope:category did not match any categories.")
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
                else:
                    error_message = tag_error_text(
                        "scope:category requires a category ID (scope:category=123) or name."
                    )
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

            use_filters = bool(
                parsed.keywords
                or parsed.exclude_keywords
                or parsed.from_ids
                or parsed.from_names
                or parsed.exclude_from_ids
                or parsed.exclude_from_names
                or parsed.mention_ids
                or parsed.mention_names
                or parsed.exclude_mention_ids
                or parsed.exclude_mention_names
                or parsed.mention_role_ids
                or parsed.exclude_mention_role_ids
                or parsed.role_ids
                or parsed.role_names
                or parsed.exclude_role_ids
                or parsed.exclude_role_names
                or parsed.has_filters
                or parsed.exclude_has_filters
                or parsed.pinned is not None
                or parsed.author_is_bot is not None
                or parsed.exclude_author_is_bot is not None
                or parsed.after
                or parsed.before
                or parsed.after_id
                or parsed.before_id
                or parsed.scan_limit
                or parsed.in_channel_ids
                or parsed.in_channel_names
                or parsed.exclude_in_channel_ids
                or parsed.exclude_in_channel_names
                or parsed.scope
            )
            scan_limit = amount
            if use_filters:
                requested_scan = parsed.scan_limit or 0
                if parsed.scan_limit is None:
                    scan_limit = None
                else:
                    scan_limit = max(amount, 200, requested_scan)

            history_kwargs: dict[str, object] = {"limit": scan_limit}
            if parsed.after_id:
                history_kwargs["after"] = discord.Object(id=parsed.after_id)
            elif parsed.after:
                history_kwargs["after"] = parsed.after
            if parsed.before_id:
                history_kwargs["before"] = discord.Object(id=parsed.before_id)
            elif parsed.before:
                history_kwargs["before"] = parsed.before

            target_channels: list[discord.abc.Messageable] = []
            if parsed.scope:
                def _guild_candidates(guild: discord.Guild) -> list[discord.abc.GuildChannel]:
                    candidates: list[discord.abc.GuildChannel] = []
                    candidates.extend(getattr(guild, "text_channels", []))
                    candidates.extend(getattr(guild, "forum_channels", []))
                    candidates.extend(getattr(guild, "threads", []))
                    return candidates

                async def _member_for_guild(guild: discord.Guild) -> discord.Member | None:
                    member = guild.get_member(ctx.author.id)
                    if member is None:
                        with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                            member = await guild.fetch_member(ctx.author.id)
                    return member

                async def _bot_member_for_guild(guild: discord.Guild) -> discord.Member | None:
                    bot_member = guild.me or guild.get_member(self.bot.user.id if self.bot.user else 0)
                    if bot_member is None and self.bot.user is not None:
                        with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                            bot_member = await guild.fetch_member(self.bot.user.id)
                    return bot_member

                if parsed.scope == "all":
                    guild = target_guild or ctx.guild
                    if guild is None:
                        resolve_errors.append("scope:all requires a server context.")
                    else:
                        member = await _member_for_guild(guild)
                        if member is None:
                            resolve_errors.append("You are not a member of this server.")
                        else:
                            bot_member = await _bot_member_for_guild(guild)
                            for channel in _guild_candidates(guild):
                                if not hasattr(channel, "history"):
                                    continue
                                user_perms = channel.permissions_for(member)
                                if not user_perms.view_channel or not user_perms.read_message_history:
                                    continue
                                if bot_member is not None:
                                    bot_perms = channel.permissions_for(bot_member)
                                    if not bot_perms.view_channel or not bot_perms.read_message_history:
                                        continue
                                target_channels.append(channel)
                elif parsed.scope == "global":
                    if target_guild is not None:
                        resolve_errors.append("server: cannot be used with scope:global.")
                        target_channels = []
                        warnings = resolve_errors
                        if warnings and not target_channels:
                            error_message = tag_error_text(" ".join(warnings))
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
                    for guild in self.bot.guilds:
                        member = await _member_for_guild(guild)
                        if member is None:
                            continue
                        bot_member = await _bot_member_for_guild(guild)
                        for channel in _guild_candidates(guild):
                            if not hasattr(channel, "history"):
                                continue
                            user_perms = channel.permissions_for(member)
                            if not user_perms.view_channel or not user_perms.read_message_history:
                                continue
                            if bot_member is not None:
                                bot_perms = channel.permissions_for(bot_member)
                                if not bot_perms.view_channel or not bot_perms.read_message_history:
                                    continue
                            target_channels.append(channel)
                elif parsed.scope == "category":
                    guild = target_guild or ctx.guild
                    if guild is None:
                        resolve_errors.append("scope:category requires a server context.")
                    else:
                        member = await _member_for_guild(guild)
                        if member is None:
                            resolve_errors.append("You are not a member of this server.")
                        else:
                            bot_member = await _bot_member_for_guild(guild)
                            category_channels: list[discord.abc.GuildChannel] = []
                            name_targets = {
                                _normalize_category_name(name)
                                for name in parsed.scope_category_names
                            }
                            for category in guild.categories:
                                if (
                                    parsed.scope_category_ids
                                    and category.id not in parsed.scope_category_ids
                                ):
                                    continue
                                if name_targets and _normalize_category_name(category.name) not in name_targets:
                                    continue
                                category_channels.extend(category.channels)
                                for thread in getattr(guild, "threads", []):
                                    parent = getattr(thread, "parent", None)
                                    if parent and getattr(parent, "category_id", None) == category.id:
                                        category_channels.append(thread)
                            for channel in category_channels:
                                if not hasattr(channel, "history"):
                                    continue
                                user_perms = channel.permissions_for(member)
                                if not user_perms.view_channel or not user_perms.read_message_history:
                                    continue
                                if bot_member is not None:
                                    bot_perms = channel.permissions_for(bot_member)
                                    if not bot_perms.view_channel or not bot_perms.read_message_history:
                                        continue
                                target_channels.append(channel)
            elif parsed.in_channel_ids:
                for channel_id in sorted(parsed.in_channel_ids):
                    channel = self.bot.get_channel(channel_id)
                    if channel is None:
                        with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                            channel = await self.bot.fetch_channel(channel_id)
                    if channel is None:
                        resolve_errors.append(f"<#{channel_id}>: channel not found.")
                        continue
                    if not hasattr(channel, "history"):
                        resolve_errors.append(
                            f"{getattr(channel, 'mention', f'<#{channel_id}>')}: channel does not support messages."
                        )
                        continue
                    guild = getattr(channel, "guild", None)
                    if guild is None:
                        resolve_errors.append(
                            f"{getattr(channel, 'mention', f'<#{channel_id}>')}: direct messages are not supported."
                        )
                        continue
                    member = guild.get_member(ctx.author.id)
                    if member is None:
                        with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                            member = await guild.fetch_member(ctx.author.id)
                    if member is None:
                        resolve_errors.append(
                            f"{getattr(channel, 'mention', f'<#{channel_id}>')}: you are not a member of {guild.name}."
                        )
                        continue
                    user_perms = channel.permissions_for(member)
                    if not user_perms.view_channel or not user_perms.read_message_history:
                        resolve_errors.append(
                            f"{getattr(channel, 'mention', f'<#{channel_id}>')}: you don't have permission to read messages."
                        )
                        continue
                    bot_member = guild.me or guild.get_member(self.bot.user.id if self.bot.user else 0)
                    if bot_member is None and self.bot.user is not None:
                        with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                            bot_member = await guild.fetch_member(self.bot.user.id)
                    if bot_member is not None:
                        bot_perms = channel.permissions_for(bot_member)
                        if not bot_perms.view_channel or not bot_perms.read_message_history:
                            resolve_errors.append(
                                f"{getattr(channel, 'mention', f'<#{channel_id}>')}: I don't have permission to read messages."
                            )
                            continue
                    target_channels.append(channel)
            else:
                if parsed.scope_provided or parsed.in_provided:
                    error_message = tag_error_text("No channels matched the provided scope or in: filter.")
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
                target_channels.append(ctx.channel)

            if parsed.exclude_in_channel_ids:
                target_channels = [
                    channel
                    for channel in target_channels
                    if getattr(channel, "id", None) not in parsed.exclude_in_channel_ids
                ]

            warnings = resolve_errors
            if warnings and not target_channels:
                error_message = tag_error_text(" ".join(warnings))
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

            if not target_channels:
                error_message = tag_error_text("No channels are available to search.")
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

            history: list[discord.Message] = []
            for channel in target_channels:
                if not hasattr(channel, "history"):
                    continue
                matches: list[discord.Message] = []
                scanned = 0
                try:
                    async for message in channel.history(**history_kwargs):
                        scanned += 1
                        if scan_limit is not None and scanned > scan_limit:
                            break
                        if use_filters and not _message_matches_query(message, parsed):
                            continue
                        matches.append(message)
                        if len(matches) >= amount:
                            break
                except (discord.Forbidden, discord.NotFound, discord.HTTPException):
                    channel_id = getattr(channel, "id", "unknown")
                    mention = getattr(channel, "mention", f"<#{channel_id}>")
                    resolve_errors.append(f"{mention}: could not fetch messages.")
                    continue
                history.extend(matches)
            history.sort(key=lambda m: m.created_at)

            if resolve_errors and not history:
                error_message = tag_error_text(" ".join(resolve_errors))
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
                show_channel = bool(
                    parsed.scope
                    or parsed.in_channel_ids
                    or parsed.in_channel_names
                )
                show_guild = parsed.scope == "global"
                content = _format_messages(
                    selected,
                    show_channel=show_channel,
                    show_guild=show_guild,
                )
            if warnings:
                warning_text = " ".join(warnings)
                if len(warning_text) > 500:
                    warning_text = warning_text[:497] + "..."
                content = f"{content}\n\n⚠️ {warning_text}"

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
