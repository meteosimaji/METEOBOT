import asyncio
import importlib
import logging
import os
import re
import socket
import warnings
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
import ipaddress
import json
from datetime import datetime, timedelta, timezone

import discord
from discord.ext import commands
from PIL import Image as PILImage, ImageOps, UnidentifiedImageError, features as PILFeatures

from utils import BOT_PREFIX, defer_interaction, safe_reply, tag_error_embed, tag_error_text

HEIF_ENABLED = False
try:  # Optional HEIC/HEIF support
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_ENABLED = True
except Exception:
    HEIF_ENABLED = False

# Safety guard against decompression bombs in untrusted images
PILImage.MAX_IMAGE_PIXELS = 4096 * 4096 * 4
DecompressionBombError = getattr(PILImage, "DecompressionBombError", OSError)
DecompressionBombWarning = getattr(PILImage, "DecompressionBombWarning", Warning)

log = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 50 * 1024 * 1024  # 50MB per input image (Images API limit)
MAX_IMAGE_COUNT = 16  # Images API accepts up to 16 input images
MAX_VIDEO_REFERENCES = 1
MAX_VIDEO_BYTES = 8 * 1024 * 1024
DEFAULT_MODEL = "sora-2"
DEFAULT_SECONDS = "8"
DEFAULT_SIZE = "1280x720"
LIMITS_PATH = Path(".data/video_limits.json")
USER_DAILY_LIMIT = 1
GUILD_WEEKLY_LIMIT = 2
GLOBAL_DAILY_LIMIT = 2
ALLOWED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/heic",
    "image/heif",
}
URL_PATTERN = re.compile(r"https?://[^\s<>]+")
VIDEO_ID_PATTERN = re.compile(r"video_[a-zA-Z0-9]+")
VIDEO_URL_PATTERN = re.compile(r"https?://[^\s<>]*video_[a-zA-Z0-9]+[^\s<>]*")
MODEL_OPTION_PATTERN = re.compile(r"\bmodel:[^\s]+\b", re.IGNORECASE)
SECONDS_OPTION_PATTERN = re.compile(r"\bseconds:(4|8|12)\b")
SIZE_OPTION_PATTERN = re.compile(r"\bsize:(\d{3,4}x\d{3,4})\b", re.IGNORECASE)
MESSAGE_LINK_PATTERN = re.compile(
    r"https?://(?:(?:ptb|canary)\.)?discord(?:app)?\.com/channels/"
    r"(?P<guild_id>\d+|@me)/(?P<channel_id>\d+)/(?P<message_id>\d+)(?:\?.*)?$"
)
BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]
ALLOWED_VIDEO_SIZES = {"1280x720", "720x1280"}

_openai_module = importlib.import_module("openai")
OpenAI = getattr(_openai_module, "OpenAI")
AsyncOpenAI = getattr(_openai_module, "AsyncOpenAI", None)
BadRequestError = getattr(_openai_module, "BadRequestError", ())


def _summarize_prompt_for_embed(prompt: str, max_len: int = 300) -> str:
    """Replace URLs with clickable labels (using filenames/hosts) and trim."""

    def _format_url(url: str) -> str:
        url = url.rstrip(") ,.;&")
        parsed = urlparse(url)
        filename = Path(parsed.path.rstrip("/")).name or parsed.netloc or "link"
        label = re.sub(r"[\[\]\(\)]", "", filename.split("?")[0] or "link")
        return f"[{label}]({url})"

    replaced = URL_PATTERN.sub(lambda m: _format_url(m.group(0)), prompt).strip()
    if not replaced:
        return "[prompt hidden]"

    if len(replaced) <= max_len:
        return replaced

    parts: list[str] = []
    length = 0
    for token in re.split(r"(\s+)", replaced):
        if not token:
            continue
        token_len = len(token)
        if length + token_len > max_len:
            if not parts:
                return replaced[: max_len - 1].rstrip() + "…"
            break
        parts.append(token)
        length += token_len

    return "".join(parts).rstrip() + "…"


def _normalize_discord_cdn_url(url: str) -> str:
    """Strip transform query params from Discord CDN/preview URLs when safe."""

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    # Only normalize Discord CDN/preview URLs; leave other hosts intact to avoid
    # breaking signed or transformation-required links.
    if host not in {"media.discordapp.net", "cdn.discordapp.com"}:
        return url

    query_dict = dict(parse_qsl(parsed.query, keep_blank_values=True))

    # Signed/expiring URLs must be left untouched; modifying the query can
    # invalidate the signature and yield 404s even when the asset exists.
    if any(key in query_dict for key in ("ex", "is", "hm")):
        return url

    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key not in {"format", "quality", "width", "height"}
    ]

    parsed = parsed._replace(query=urlencode(query_pairs, doseq=True))
    return urlunparse(parsed)


def _with_discord_cdn_format(
    url: str, image_format: str, *, prefer_media: bool = False
) -> str | None:
    """Return a Discord CDN URL with a forced format, when safe to modify."""

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in {"media.discordapp.net", "cdn.discordapp.com"}:
        return None

    query_dict = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if any(key in query_dict for key in ("ex", "is", "hm")):
        return None

    if prefer_media and host != "media.discordapp.net":
        parsed = parsed._replace(netloc="media.discordapp.net")

    if query_dict.get("format") == image_format and not prefer_media:
        return None

    query_dict["format"] = image_format
    return urlunparse(parsed._replace(query=urlencode(query_dict, doseq=True)))


def _describe_header(data: bytes) -> str:
    if not data:
        return "empty"
    header = data[:12]
    hex_part = header.hex(" ")
    ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in header)
    return f"{hex_part} | {ascii_part}"


def _extract_remix_target(prompt: str) -> tuple[str | None, str]:
    """Extract a single remix video ID and return a cleaned prompt."""
    matches = VIDEO_ID_PATTERN.findall(prompt)
    if not matches:
        return None, prompt
    if len(set(matches)) > 1:
        raise ValueError("Multiple video IDs detected. Provide only one for remix.")
    remix_id = matches[0]
    cleaned = VIDEO_URL_PATTERN.sub("", prompt)
    cleaned = re.sub(rf"\b{re.escape(remix_id)}\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return remix_id, cleaned


def _extract_video_options(prompt: str) -> tuple[str, str, str, str, list[str]]:
    model = DEFAULT_MODEL
    seconds = DEFAULT_SECONDS
    size = DEFAULT_SIZE
    notes: list[str] = []

    if MODEL_OPTION_PATTERN.search(prompt):
        notes.append("Model selection is disabled; using sora-2.")
        prompt = MODEL_OPTION_PATTERN.sub("", prompt)

    seconds_match = SECONDS_OPTION_PATTERN.search(prompt)
    if seconds_match:
        seconds = seconds_match.group(1)
        prompt = SECONDS_OPTION_PATTERN.sub("", prompt)

    size_match = SIZE_OPTION_PATTERN.search(prompt)
    if size_match:
        candidate = size_match.group(1).lower()
        if candidate in ALLOWED_VIDEO_SIZES:
            size = candidate
        else:
            notes.append(
                f"Requested size {candidate} isn't supported; using {DEFAULT_SIZE}."
            )
        prompt = SIZE_OPTION_PATTERN.sub("", prompt)

    prompt = re.sub(r"\s+", " ", prompt).strip()
    return model, seconds, size, prompt, notes


def _parse_size(size: str) -> tuple[int, int]:
    try:
        width_str, height_str = size.lower().split("x", 1)
        return int(width_str), int(height_str)
    except Exception as exc:
        raise ValueError(f"Invalid size format: {size}") from exc


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_day_key(now: datetime) -> str:
    return now.date().isoformat()


def _utc_week_start(now: datetime) -> datetime:
    days_since_sunday = (now.weekday() + 1) % 7
    start = now - timedelta(days=days_since_sunday)
    return start.replace(hour=0, minute=0, second=0, microsecond=0)


def _format_utc(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _extract_video_error(video: object) -> str | None:
    for attr in ("error", "last_error", "reason"):
        value = getattr(video, attr, None)
        if not value:
            continue
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            message = value.get("message") or value.get("error") or value.get("reason")
            if message:
                return str(message)
            return json.dumps(value, ensure_ascii=False)
        return str(value)
    return None


def _resize_reference_image(
    img: PILImage.Image, target: tuple[int, int]
) -> PILImage.Image:
    """Resize with letterboxing to avoid cropping."""
    resample = getattr(PILImage, "Resampling", PILImage).LANCZOS
    fitted = ImageOps.contain(img, target, method=resample)
    canvas = PILImage.new("RGB", target, (0, 0, 0))
    x = (target[0] - fitted.width) // 2
    y = (target[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


class Video(commands.Cog):
    """Generate videos with Sora."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        token = os.getenv("OPENAI_TOKEN")
        if not token:
            log.warning("OPENAI_TOKEN is not set. Add it to your .env")
        if AsyncOpenAI is not None:
            self.client = AsyncOpenAI(api_key=token)
            self._async_client = True
        else:
            self.client = OpenAI(api_key=token)
            self._async_client = False
        self._limit_lock = asyncio.Lock()

    async def _load_limits(self) -> dict[str, dict[str, dict[str, object]]]:
        if not LIMITS_PATH.exists():
            return {"users": {}, "guilds": {}, "global": {}}
        try:
            raw = LIMITS_PATH.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception:
            log.warning("Failed to read %s; starting fresh.", LIMITS_PATH)
            return {"users": {}, "guilds": {}, "global": {}}
        if not isinstance(data, dict):
            return {"users": {}, "guilds": {}, "global": {}}
        data.setdefault("users", {})
        data.setdefault("guilds", {})
        data.setdefault("global", {})
        return data  # type: ignore[return-value]

    async def _save_limits(self, data: dict[str, dict[str, dict[str, object]]]) -> None:
        LIMITS_PATH.parent.mkdir(parents=True, exist_ok=True)
        LIMITS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    async def _check_and_consume_limits(self, ctx: commands.Context) -> str | None:
        now = _utc_now()
        day_key = _utc_day_key(now)
        next_day = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = _utc_week_start(now)
        week_key = week_start.date().isoformat()
        next_week = week_start + timedelta(days=7)

        user_id = str(ctx.author.id)
        guild_id = str(ctx.guild.id) if ctx.guild else None

        async with self._limit_lock:
            data = await self._load_limits()
            users = data.setdefault("users", {})
            guilds = data.setdefault("guilds", {})
            global_limits = data.setdefault("global", {})

            errors: list[str] = []
            user_entry = users.get(user_id, {})
            user_day = user_entry.get("day")
            user_count = int(user_entry.get("count", 0) or 0)
            if user_day != day_key:
                user_count = 0
            if user_count >= USER_DAILY_LIMIT:
                errors.append(
                    "You already used /video today (per-user daily limit across all servers). "
                    f"Next reset: {_format_utc(next_day)}."
                )

            if guild_id:
                guild_entry = guilds.get(guild_id, {})
                guild_week = guild_entry.get("week_start")
                count = int(guild_entry.get("count", 0) or 0)
                if guild_week != week_key:
                    count = 0
                if count >= GUILD_WEEKLY_LIMIT:
                    errors.append(
                        "This server has reached its weekly /video limit "
                        f"({GUILD_WEEKLY_LIMIT} per week shared across users). "
                        f"Next reset: {_format_utc(next_week)}."
                    )

            global_day = global_limits.get("day")
            global_count = int(global_limits.get("count", 0) or 0)
            if global_day != day_key:
                global_count = 0
            if global_count >= GLOBAL_DAILY_LIMIT:
                errors.append(
                    "Global /video daily limit reached "
                    f"({GLOBAL_DAILY_LIMIT} per day across all servers). "
                    f"Next reset: {_format_utc(next_day)}."
                )

            if errors:
                return "\n".join(f"• {line}" for line in errors)

            users[user_id] = {
                "day": day_key,
                "count": user_count + 1,
                "last_used": now.isoformat(),
            }
            if guild_id:
                guild_entry = guilds.get(guild_id, {})
                if guild_entry.get("week_start") != week_key:
                    guild_entry = {"week_start": week_key, "count": 0}
                guild_entry["count"] = int(guild_entry.get("count", 0) or 0) + 1
                guilds[guild_id] = guild_entry

            if global_limits.get("day") != day_key:
                global_limits = {"day": day_key, "count": 0}
            global_limits["count"] = int(global_limits.get("count", 0) or 0) + 1
            global_limits["day"] = day_key
            data["global"] = global_limits

            await self._save_limits(data)

        return None

    async def _videos_create(self, **kwargs):
        if self._async_client:
            return await self.client.videos.create(**kwargs)
        return await asyncio.to_thread(self.client.videos.create, **kwargs)

    async def _videos_remix(self, remix_id: str, *, prompt: str):
        remix_method = getattr(self.client.videos, "remix", None)
        if remix_method:
            if self._async_client:
                try:
                    return await remix_method(video_id=remix_id, prompt=prompt)
                except TypeError:
                    return await remix_method(remix_id, prompt=prompt)
            try:
                return await asyncio.to_thread(remix_method, video_id=remix_id, prompt=prompt)
            except TypeError:
                return await asyncio.to_thread(remix_method, remix_id, prompt=prompt)
        raise RuntimeError(
            "Your OpenAI SDK does not support video remix. Update the openai package."
        )

    async def _videos_retrieve(self, video_id: str):
        if self._async_client:
            return await self.client.videos.retrieve(video_id)
        return await asyncio.to_thread(self.client.videos.retrieve, video_id)

    async def _videos_create_and_poll(self, **kwargs):
        create_and_poll = getattr(self.client.videos, "create_and_poll", None)
        if create_and_poll:
            if self._async_client:
                return await create_and_poll(**kwargs)
            return await asyncio.to_thread(create_and_poll, **kwargs)
        video = await self._videos_create(**kwargs)
        return await self._videos_poll(video.id)

    async def _videos_poll(self, video_id: str):
        delay_s = 8
        video = await self._videos_retrieve(video_id)
        while getattr(video, "status", None) in {"queued", "in_progress"}:
            await asyncio.sleep(delay_s)
            video = await self._videos_retrieve(video_id)
        return video

    async def _download_content(self, video_id: str, *, variant: str):
        if self._async_client:
            content = await self.client.videos.download_content(video_id, variant=variant)
        else:
            content = await asyncio.to_thread(
                self.client.videos.download_content, video_id, variant=variant
            )
        if hasattr(content, "read"):
            return content.read()
        if hasattr(content, "iter_bytes"):
            return b"".join(content.iter_bytes())
        if isinstance(content, bytes):
            return content
        if hasattr(content, "write_to_file"):
            tmp_path = Path(f".tmp_{video_id}_{variant}")
            content.write_to_file(tmp_path)
            data = tmp_path.read_bytes()
            try:
                tmp_path.unlink()
            except Exception:
                pass
            return data
        raise RuntimeError("Unexpected download content type")

    async def _gather_images(
        self, ctx: commands.Context, prompt: str
    ) -> tuple[list[tuple[bytes, str]], list[str], bool, str]:
        """Collect input images from ask-injected context, attachments, replies, and URLs."""

        notes: list[str] = []
        images: list[tuple[bytes, str]] = []
        had_candidates = False
        seen_attachment_ids: set[int] = set()
        seen_urls: set[str] = set()
        seen_message_links: set[int] = set()
        used_prompt_urls: list[str] = []

        MAX_DECODE_DIM = 4096  # Safety guard against overly large inputs during decode

        def _normalize_image_bytes(
            data: bytes,
            source: str,
            filename_hint: str,
            content_type_hint: str = "",
        ) -> tuple[bytes, str] | None:
            lower_hint = filename_hint.lower()

            # Quick detection for obvious HTML responses to improve user-facing notes.
            if data[:6].lower().startswith(b"<html") or data[:1] == b"<":
                notes.append(f"Skipped {source}: URL returned HTML, not an image.")
                return None

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", DecompressionBombWarning)
                    img = PILImage.open(BytesIO(data))
                    img.load()
            except (
                UnidentifiedImageError,
                OSError,
                DecompressionBombError,
                DecompressionBombWarning,
            ):
                log.warning(
                    "Image decode failed for %s (content_type=%s size=%s header=%s).",
                    source,
                    content_type_hint or "unknown",
                    len(data) if data else 0,
                    _describe_header(data),
                )
                is_heic = lower_hint.endswith((".heic", ".heif")) or content_type_hint in {
                    "image/heic",
                    "image/heif",
                }
                if is_heic and not HEIF_ENABLED:
                    notes.append(
                        f"Skipped {source}: HEIC/HEIF isn't supported on this host (install pillow-heif). "
                        "Convert to JPG/PNG and re-upload."
                    )
                else:
                    notes.append(
                        f"Skipped {source}: couldn't decode as an image. "
                        "If this came from a Discord preview/CDN, download it and re-upload the actual file."
                    )
                return None

            try:
                if getattr(img, "is_animated", False) and getattr(img, "n_frames", 1) > 1:
                    notes.append(f"Skipped {source}: animated images aren't supported. Export a still frame.")
                    return None
            except Exception:
                pass

            try:
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass

            has_alpha = (
                img.mode in ("RGBA", "LA")
                or (img.mode == "P" and "transparency" in getattr(img, "info", {}))
            )

            if has_alpha:
                img = img.convert("RGBA")
            else:
                img = img.convert("RGB")

            if max(img.size) > MAX_DECODE_DIM:
                resample = getattr(PILImage, "Resampling", PILImage).LANCZOS
                img.thumbnail((MAX_DECODE_DIM, MAX_DECODE_DIM), resample)

            out = BytesIO()
            stem = Path(filename_hint or "image").stem or "image"
            if has_alpha:
                img.save(out, format="PNG", optimize=True)
                new_name = f"{stem}.png"
            else:
                img.save(out, format="JPEG", quality=95, optimize=True, progressive=True)
                new_name = f"{stem}.jpg"

            new_data = out.getvalue()
            if len(new_data) > MAX_IMAGE_BYTES:
                notes.append(
                    f"Skipped {source}: normalized image exceeded {MAX_IMAGE_BYTES // (1024 * 1024)}MB."
                )
                return None

            return new_data, new_name

        async def _read_attachment(att: discord.Attachment, source: str) -> None:
            att_id = getattr(att, "id", None)
            if att_id is not None and att_id in seen_attachment_ids:
                return
            if att_id is not None:
                seen_attachment_ids.add(att_id)

            size = getattr(att, "size", None)
            if size is not None and size > MAX_IMAGE_BYTES:
                notes.append(
                    f"Skipped {source}: attachment is larger than {MAX_IMAGE_BYTES // (1024 * 1024)}MB."
                )
                return

            if len(images) >= MAX_VIDEO_REFERENCES:
                notes.append(
                    f"Skipped {source}: video references are limited to {MAX_VIDEO_REFERENCES} image."
                )
                return
            if len(images) >= MAX_IMAGE_COUNT:
                notes.append(
                    f"Skipped {source}: reached the {MAX_IMAGE_COUNT}-image limit for edits."
                )
                return

            content_type = (att.content_type or "").split(";", 1)[0].lower()
            if not content_type:
                ext = Path(getattr(att, "filename", "")).suffix.lower()
                ext_map = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".webp": "image/webp",
                    ".heic": "image/heic",
                    ".heif": "image/heif",
                }
                content_type = ext_map.get(ext, "")

            if content_type not in ALLOWED_IMAGE_TYPES:
                notes.append(f"Skipped {source}: unsupported file type.")
                return

            try:
                data = await att.read()
            except Exception:
                notes.append(f"Skipped {source}: couldn't read the attachment.")
                return

            if not data:
                notes.append(f"Skipped {source}: attachment was empty.")
                return

            if len(data) > MAX_IMAGE_BYTES:
                notes.append(
                    f"Skipped {source}: attachment is larger than {MAX_IMAGE_BYTES // (1024 * 1024)}MB."
                )
                return
            filename_hint = getattr(att, "filename", "image") or "image"
            normalized = _normalize_image_bytes(data, source, filename_hint, content_type)
            if not normalized:
                return

            norm_data, norm_name = normalized
            images.append((norm_data, norm_name))

        def _iter_attachments() -> list[tuple[discord.Attachment, str]]:
            att_list: list[tuple[discord.Attachment, str]] = []

            if getattr(ctx, "message", None) is not None and ctx.message:
                for att in ctx.message.attachments:
                    att_list.append((att, "your message attachment"))

                ref = ctx.message.reference
                if ref and ref.resolved and isinstance(ref.resolved, discord.Message):
                    for att in ref.resolved.attachments:
                        att_list.append((att, "the replied message attachment"))

            for att in getattr(ctx, "ai_images", []) or []:
                if isinstance(att, discord.Attachment):
                    att_list.append((att, "an ask attachment"))

            interaction = getattr(ctx, "interaction", None)
            for att in getattr(interaction, "attachments", []) or []:
                att_list.append((att, "your slash attachment"))

            return att_list

        attachments = _iter_attachments()
        if attachments:
            had_candidates = True

        for att, source in attachments:
            await _read_attachment(att, source)

        # URLs embedded in the prompt
        url_matches = [match.rstrip("),.;&") for match in URL_PATTERN.findall(prompt)]
        if url_matches:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                loop = asyncio.get_running_loop()

                async def _is_blocked_host(hostname: str) -> bool:
                    try:
                        ip_obj = ipaddress.ip_address(hostname)
                        return any(ip_obj in net for net in BLOCKED_NETWORKS)
                    except ValueError:
                        pass

                    try:
                        addr_infos = await loop.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
                    except Exception:
                        return True

                    for info in addr_infos:
                        sockaddr = info[4]
                        if not sockaddr:
                            continue
                        ip_str = sockaddr[0]
                        try:
                            ip_obj = ipaddress.ip_address(ip_str)
                        except ValueError:
                            return True
                        if any(ip_obj in net for net in BLOCKED_NETWORKS):
                            return True
                    return False

                ext_content_type_map = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".webp": "image/webp",
                    ".heic": "image/heic",
                    ".heif": "image/heif",
                }

                for raw_url in url_matches:
                    # Discord attachments sometimes include a trailing '&' (and similar
                    # punctuation) when copied out of embeds or message content. These
                    # suffixes break the signed CDN query string and lead to HTML/403
                    # bodies that Pillow cannot decode. Strip common trailing noise
                    # before normalization/fetching.
                    url = raw_url
                    normalized_url = _normalize_discord_cdn_url(url)
                    if normalized_url in seen_urls:
                        continue
                    seen_urls.add(normalized_url)

                    message_match = MESSAGE_LINK_PATTERN.match(url)
                    if message_match:
                        had_candidates = True
                        try:
                            message_id = int(message_match.group("message_id"))
                            channel_id = int(message_match.group("channel_id"))
                        except ValueError:
                            notes.append(f"Skipped message link (invalid ID): {url}")
                            continue

                        if message_id in seen_message_links:
                            continue
                        seen_message_links.add(message_id)

                        channel = self.bot.get_channel(channel_id)
                        if channel is None:
                            try:
                                channel = await self.bot.fetch_channel(channel_id)
                            except Exception:
                                notes.append(f"Skipped message link (can't access channel): {url}")
                                continue

                        if not hasattr(channel, "fetch_message"):
                            notes.append(
                                f"Skipped message link (channel doesn't allow reading messages): {url}"
                            )
                            continue

                        try:
                            linked_message = await channel.fetch_message(message_id)
                        except (discord.Forbidden, discord.NotFound):
                            notes.append(f"Skipped message link (can't read message): {url}")
                            continue
                        except Exception:
                            notes.append(f"Skipped message link (failed to read message): {url}")
                            continue

                        if not linked_message.attachments:
                            notes.append(f"Skipped message link (no attachments to edit): {url}")
                            continue

                        for att in linked_message.attachments:
                            await _read_attachment(att, f"message link {url}")
                        used_prompt_urls.append(url)
                        continue

                    if len(images) >= MAX_VIDEO_REFERENCES:
                        notes.append(
                            f"Skipped URL {normalized_url}: video references are limited to {MAX_VIDEO_REFERENCES} image."
                        )
                        continue
                    if len(images) >= MAX_IMAGE_COUNT:
                        notes.append(
                            f"Skipped URL {normalized_url}: reached the {MAX_IMAGE_COUNT}-image limit for edits."
                        )
                        continue

                    parsed = urlparse(normalized_url)
                    if parsed.scheme != "https":
                        notes.append("Skipped URL (HTTPS required): " + normalized_url)
                        continue

                    hostname = parsed.hostname
                    if not hostname:
                        notes.append("Skipped URL (invalid host): " + normalized_url)
                        continue

                    if await _is_blocked_host(hostname):
                        notes.append("Skipped URL (host blocked): " + normalized_url)
                        continue

                    try:
                        redirect_statuses = {301, 302, 303, 307, 308}
                        current_url = normalized_url
                        redirect_hops = 0
                        attempted_png_fallback = False
                        used_png_fallback = False
                        attempted_retry = False
                        while True:
                            resp = await session.get(current_url, allow_redirects=False)
                            try:
                                if resp.status in redirect_statuses:
                                    location = resp.headers.get("Location")
                                    if not location:
                                        notes.append(
                                            f"Skipped URL (redirect missing Location): {current_url}"
                                        )
                                        break
                                    current_url = urljoin(current_url, location)
                                    parsed_redirect = urlparse(current_url)
                                    if parsed_redirect.scheme != "https":
                                        notes.append("Skipped URL (HTTPS required): " + current_url)
                                        break
                                    redirect_host = parsed_redirect.hostname
                                    if not redirect_host or await _is_blocked_host(redirect_host):
                                        notes.append("Skipped URL (host blocked): " + current_url)
                                        break
                                    redirect_hops += 1
                                    if redirect_hops > 3:
                                        notes.append(f"Skipped URL (too many redirects): {normalized_url}")
                                        break
                                    continue

                                if resp.status != 200:
                                    notes.append(f"Skipped URL ({resp.status}): {current_url}")
                                    break

                                parsed_final = urlparse(current_url)
                                png_fallback_url = _with_discord_cdn_format(
                                    current_url, "png", prefer_media=True
                                )

                                content_type = (resp.headers.get("Content-Type", "") or "").split(
                                    ";", 1
                                )[0].lower()
                                ext_type_hint = ext_content_type_map.get(
                                    Path(parsed_final.path).suffix.lower(), ""
                                )
                                is_discord_cdn = parsed_final.hostname in {
                                    "cdn.discordapp.com",
                                    "media.discordapp.net",
                                }
                                if (
                                    is_discord_cdn
                                    or ext_type_hint in ALLOWED_IMAGE_TYPES
                                    or content_type in ALLOWED_IMAGE_TYPES
                                ):
                                    had_candidates = True
                                if content_type not in ALLOWED_IMAGE_TYPES:
                                    if not content_type and not is_discord_cdn and ext_type_hint not in ALLOWED_IMAGE_TYPES:
                                        notes.append(f"Skipped URL (missing content type): {current_url}")
                                        break
                                    if content_type and not is_discord_cdn and ext_type_hint not in ALLOWED_IMAGE_TYPES:
                                        notes.append(f"Skipped URL (not an image): {current_url}")
                                        break
                                    if not content_type and is_discord_cdn:
                                        notes.append(
                                            f"Attempting decode despite missing content type: {current_url}"
                                        )
                                    elif content_type and is_discord_cdn:
                                        notes.append(
                                            f"Attempting decode despite content type {content_type}: {current_url}"
                                        )
                                    elif ext_type_hint in ALLOWED_IMAGE_TYPES:
                                        notes.append(
                                            f"Attempting decode based on file extension despite content type {content_type or 'missing'}: {current_url}"
                                        )

                                if (
                                    content_type == "image/webp"
                                    and not PILFeatures.check("webp")
                                    and png_fallback_url
                                    and not attempted_png_fallback
                                ):
                                    attempted_png_fallback = True
                                    used_png_fallback = True
                                    current_url = png_fallback_url
                                    continue

                                content_length = resp.headers.get("Content-Length")
                                content_length_val: int | None = None
                                try:
                                    if content_length is not None:
                                        content_length_val = int(content_length)
                                    if content_length_val is not None and content_length_val > MAX_IMAGE_BYTES:
                                        notes.append(
                                            f"Skipped URL: file exceeded {MAX_IMAGE_BYTES // (1024 * 1024)}MB"
                                        )
                                        break
                                except ValueError:
                                    pass

                                data = await resp.content.read(MAX_IMAGE_BYTES + 1)
                                if len(data) > MAX_IMAGE_BYTES:
                                    notes.append(
                                        f"Skipped URL: file exceeded {MAX_IMAGE_BYTES // (1024 * 1024)}MB"
                                    )
                                    break

                                name_hint = Path(parsed_final.path).name or "image"
                                normalized = _normalize_image_bytes(
                                    data,
                                    f"URL {current_url}",
                                    name_hint,
                                    content_type or ext_type_hint,
                                )
                                if not normalized:
                                    if (
                                        png_fallback_url
                                        and not attempted_png_fallback
                                        and png_fallback_url != current_url
                                    ):
                                        attempted_png_fallback = True
                                        used_png_fallback = True
                                        current_url = png_fallback_url
                                        continue
                                    if (
                                        not attempted_retry
                                        and content_length_val is not None
                                        and len(data) < content_length_val
                                    ):
                                        attempted_retry = True
                                        continue
                                    break

                                norm_data, norm_name = normalized
                                images.append((norm_data, norm_name))
                                if used_png_fallback:
                                    notes.append(
                                        "Used Discord CDN PNG fallback "
                                        f"(orig: {normalized_url}, used: {current_url})."
                                    )
                                used_prompt_urls.append(url)
                                break
                            finally:
                                resp.release()
                    except Exception as exc:
                        notes.append(
                            f"Skipped URL (download failed): {normalized_url} "
                            f"({type(exc).__name__}: {str(exc)[:120]})"
                        )

        if used_prompt_urls:
            cleaned_prompt = prompt
            for used_url in sorted(set(used_prompt_urls), key=len, reverse=True):
                cleaned_prompt = re.sub(re.escape(used_url), "", cleaned_prompt)
            cleaned_prompt = re.sub(r"\s+", " ", cleaned_prompt).strip()
        else:
            cleaned_prompt = prompt

        return images, notes, had_candidates, cleaned_prompt

    @commands.hybrid_command(
        name="video",
        description="Generate a video from a text prompt using Sora.",
        help=(
            "Generate a short video with Sora and wait for completion. Attach an image, include a public HTTPS image URL, or paste a Discord message link with an attachment to use it as the first frame (video edit). "
            "Include a Sora video ID (video_...) or link to remix an existing video.\n\n"
            "Prompts work best when they describe shot type, subject, action, setting, and lighting.\n\n"
            "**Usage**: `/video <prompt>`\n"
            "**Options**: `seconds` = `4`, `8`, or `12`, `size` = `720x1280` or `1280x720`.\n"
            "**Examples**: `/video Wide tracking shot of a teal coupe driving through a desert highway, heat ripples visible, hard sun overhead.`\n"
            "`/video Close-up of a steaming coffee cup on a wooden table, morning light through blinds, soft depth of field.`\n"
            "`/video video_abc123 Shift the color palette to teal, sand, and rust, with a warm backlight.`\n"
            "`/video seconds:12 size:720x1280 A cinematic drone shot over a misty rainforest at sunrise.`\n"
            f"`{BOT_PREFIX}video a cozy cabin in falling snow at night` (with an attachment to use as the first frame)\n\n"
            "Bot defaults: model `sora-2`, size `1280x720`, seconds `8`. Reference images are resized to the target size"
            " with letterboxing. Results are asynchronous and can take a few minutes.\n\n"
            "Limits: global usage is capped at 2 videos per day across all servers; each user can run /video once per day "
            "across all servers; each server can run /video twice per week shared across users. Weekly limits reset at "
            "Sunday 00:00 UTC."
        ),
        extras={
            "category": "AI",
            "destination": (
                "Generate or remix videos with Sora from prompts, reference images, or a video ID "
                "(daily global, daily per-user, and weekly per-server limits apply)."
            ),
            "plus": (
                "Attach/URL/link an image to use it as the first frame (auto-resized to target size), "
                "or include a video_... ID to remix. Optional tokens: seconds:12, size:720x1280."
                " HTTPS URLs only."
            ),
            "pro": "Uses the Video API with Sora (sora-2), supports remixing, attachments, reply images, ask-injected images, HTTPS URLs, and message links that contain attachments.",
        },
    )
    async def video(self, ctx: commands.Context, *, prompt: str) -> None:
        if not prompt or not prompt.strip():
            return await safe_reply(
                ctx,
                tag_error_text(
                    "Tell me what to create first. Describe shot type, subject, action, setting, and lighting."
                ),
                ephemeral=True,
                mention_author=False,
            )

        prompt = prompt.strip()
        try:
            remix_id, prompt = _extract_remix_target(prompt)
        except ValueError as exc:
            return await safe_reply(
                ctx,
                tag_error_text(str(exc)),
                ephemeral=True,
                mention_author=False,
            )
        if remix_id and not prompt:
            return await safe_reply(
                ctx,
                tag_error_text("Remix requests need a prompt describing the change."),
                ephemeral=True,
                mention_author=False,
            )
        model, seconds, size, prompt, option_notes = _extract_video_options(prompt)
        await defer_interaction(ctx)

        images, notes, had_candidates, cleaned_prompt = await self._gather_images(ctx, prompt)
        if cleaned_prompt:
            prompt = cleaned_prompt
        if option_notes:
            notes.extend(option_notes)
        display_prompt = _summarize_prompt_for_embed(prompt)

        if had_candidates and not images and notes and not remix_id:
            error_embed = discord.Embed(
                title="\u26a0\ufe0f No usable images to reference",
                description=(
                    "All provided image inputs were skipped or couldn't be decoded, "
                    "so I won't generate a video from this request."
                ),
                color=0xFF0000,
            )
            error_embed = tag_error_embed(error_embed)
            note_text = "\n".join(notes)
            error_embed.add_field(name="Notes", value=note_text[:1024], inline=False)
            await safe_reply(ctx, embed=error_embed, ephemeral=True, mention_author=False)
            return

        if not getattr(self.client, "api_key", None):
            return await safe_reply(
                ctx,
                tag_error_text("OPENAI_TOKEN is not set, so I can't generate videos right now."),
                ephemeral=True,
                mention_author=False,
            )

        limit_error = await self._check_and_consume_limits(ctx)
        if limit_error:
            return await safe_reply(
                ctx,
                tag_error_text("⏳ Video usage limit reached:\n" + limit_error),
                ephemeral=True,
                mention_author=False,
            )

        status_embed = discord.Embed(
            title=(
                "🎬 Remixing video…"
                if remix_id
                else "🎬 Generating video from image…"
                if images
                else "🎬 Generating video…"
            ),
            description=display_prompt,
            color=0x5865F2,
        )
        status_msg: discord.Message | None = None
        try:
            if ctx.interaction:
                status_msg = await ctx.interaction.followup.send(embed=status_embed, wait=True)
            else:
                status_msg = await ctx.reply(embed=status_embed, mention_author=False)
        except Exception:
            status_msg = None

        try:
            if remix_id and images:
                conflict_embed = discord.Embed(
                    title="\u26a0\ufe0f Can't mix video remix and image reference",
                    description=(
                        "Please provide either a remix video ID/link or a reference image, not both."
                    ),
                    color=0xFF0000,
                )
                conflict_embed = tag_error_embed(conflict_embed)
                if notes:
                    conflict_embed.add_field(
                        name="Notes",
                        value="\n".join(notes)[:1024],
                        inline=False,
                    )
                if status_msg:
                    await status_msg.edit(embed=conflict_embed, attachments=[])
                else:
                    await safe_reply(
                        ctx,
                        embed=conflict_embed,
                        ephemeral=True,
                        mention_author=False,
                    )
                return

            request_kwargs = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "seconds": seconds,
            }

            if remix_id:
                video = await self._videos_remix(remix_id, prompt=prompt)
                if getattr(video, "status", None) in {"queued", "in_progress"}:
                    video = await self._videos_poll(video.id)
            elif images:
                if len(images) > 1:
                    notes.append(
                        "Using the first image as the video reference frame; extra images were ignored."
                    )
                img_data, img_name = images[0]
                try:
                    with PILImage.open(BytesIO(img_data)) as img_check:
                        target = _parse_size(size)
                        resized = _resize_reference_image(img_check, target)
                        out = BytesIO()
                        resized.save(out, format="PNG", optimize=True)
                        img_data = out.getvalue()
                        notes.append(
                            f"Resized reference image to {size} with letterboxing."
                        )
                except Exception:
                    notes.append(
                        "Couldn't resize the reference image; Sora may reject mismatched dimensions."
                    )

                buf = BytesIO(img_data)
                buf.name = img_name or "reference.png"
                request_kwargs["input_reference"] = buf

                video = await self._videos_create_and_poll(**request_kwargs)
            else:
                video = await self._videos_create_and_poll(**request_kwargs)
            status = getattr(video, "status", None)
            if status != "completed":
                error_detail = _extract_video_error(video)
                if error_detail:
                    notes.append(f"Provider error: {error_detail}")
                raise RuntimeError(
                    f"Video creation failed with status {status}"
                    + (f": {error_detail}" if error_detail else "")
                )

            video_bytes = await self._download_content(video.id, variant="video")
            max_upload_bytes = (
                ctx.guild.filesize_limit if getattr(ctx, "guild", None) else MAX_VIDEO_BYTES
            )
            if len(video_bytes) > max_upload_bytes:
                notes.append(
                    "Video output exceeded the Discord upload limit. Try shorter duration, smaller size, or use sora-2 for faster, lighter renders."
                )
                thumbnail_bytes = await self._download_content(video.id, variant="thumbnail")
                thumb_file = discord.File(BytesIO(thumbnail_bytes), filename="thumbnail.webp")
                embed = discord.Embed(
                    title="🎬 Video Ready (too large to upload)",
                    description=display_prompt,
                    color=0x5865F2,
                )
                embed.set_image(url="attachment://thumbnail.webp")
                embed.add_field(name="Video ID", value=video.id, inline=False)
                embed.add_field(
                    name="Settings",
                    value=f"Model: {getattr(video, 'model', model)}\nSize: {getattr(video, 'size', size)}\nSeconds: {getattr(video, 'seconds', seconds)}",
                    inline=False,
                )
                if notes:
                    embed.add_field(name="Notes", value="\n".join(notes)[:1024], inline=False)
                if status_msg:
                    await status_msg.edit(embed=embed, attachments=[thumb_file])
                    return
                if ctx.interaction:
                    await ctx.interaction.followup.send(embed=embed, file=thumb_file, ephemeral=False)
                else:
                    await ctx.reply(embed=embed, file=thumb_file, mention_author=False)
                return

            video_file = discord.File(BytesIO(video_bytes), filename="video.mp4")
            embed = discord.Embed(
                title="🎬 Video Generated",
                description=display_prompt,
                color=0x5865F2,
            )
            embed.set_footer(text="Crafted with care ✨")
            embed.add_field(name="Video ID", value=video.id, inline=False)
            embed.add_field(
                name="Settings",
                value=f"Model: {getattr(video, 'model', model)}\nSize: {getattr(video, 'size', size)}\nSeconds: {getattr(video, 'seconds', seconds)}",
                inline=False,
            )
            if notes:
                embed.add_field(name="Notes", value="\n".join(notes)[:1024], inline=False)

            if status_msg:
                await status_msg.edit(embed=embed, attachments=[video_file])
                return
            if ctx.interaction:
                await ctx.interaction.followup.send(embed=embed, file=video_file, ephemeral=False)
            else:
                await ctx.reply(embed=embed, file=video_file, mention_author=False)
        except Exception as exc:
            log.exception("Failed to generate video")
            description = "An error occurred while generating the video. Try again later."
            if isinstance(exc, RuntimeError) and "Video creation failed with status" in str(exc):
                description = (
                    "The video provider reported a failed job status. "
                    "Try adjusting the prompt or reference image and try again."
                )
            if isinstance(exc, BadRequestError):
                detail = getattr(exc, "message", str(exc))
                description = f"OpenAI rejected the request: {detail}"
                description += (
                    "\nSora blocks copyrighted characters/music, real people, and content not suitable for minors. "
                    "Input images with human faces may be rejected. "
                    "If you used a reference image, ensure it is PNG/JPEG/WEBP and matches the target size."
                )
            error_embed = discord.Embed(
                title="\u26A0\ufe0f Video Failed",
                description=description,
                color=0xFF0000,
            )
            error_embed = tag_error_embed(error_embed)
            if notes:
                note_text = "\n".join(notes)
                error_embed.add_field(name="Notes", value=note_text[:1024], inline=False)
            if status_msg:
                await status_msg.edit(embed=error_embed, attachments=[])
            else:
                await safe_reply(ctx, embed=error_embed, ephemeral=True, mention_author=False)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Video(bot))
