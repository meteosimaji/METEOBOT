import asyncio
import base64
import importlib
import logging
import os
import re
import socket
from io import BytesIO
import warnings
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
import ipaddress

import discord
from discord.ext import commands
from PIL import Image as PILImage, ImageOps, UnidentifiedImageError

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

from utils import BOT_PREFIX, defer_interaction, safe_reply

log = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 50 * 1024 * 1024  # 50MB per input image (Images API limit)
MAX_IMAGE_COUNT = 16  # Images API accepts up to 16 input images
ALLOWED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/heic",
    "image/heif",
}
URL_PATTERN = re.compile(r"https?://[^\s<>]+")
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
    """Prefer original CDN assets over preview variants."""

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

    if host == "media.discordapp.net":
        parsed = parsed._replace(netloc="cdn.discordapp.com")

    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key not in {"format", "quality", "width", "height"}
    ]

    parsed = parsed._replace(query=urlencode(query_pairs, doseq=True))
    return urlunparse(parsed)

_openai_module = importlib.import_module("openai")
OpenAI = getattr(_openai_module, "OpenAI")
AsyncOpenAI = getattr(_openai_module, "AsyncOpenAI", None)
BadRequestError = getattr(_openai_module, "BadRequestError", ())
OPENAI_OMIT = getattr(_openai_module, "omit", None)


class Image(commands.Cog):
    """Generate images from prompts."""

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

    async def _images_generate(self, **kwargs):
        if self._async_client:
            return await self.client.images.generate(**kwargs)
        return await asyncio.to_thread(self.client.images.generate, **kwargs)

    async def _images_edit(self, **kwargs):
        if self._async_client:
            return await self.client.images.edit(**kwargs)
        return await asyncio.to_thread(self.client.images.edit, **kwargs)

    async def _gather_images(
        self, ctx: commands.Context, prompt: str
    ) -> tuple[list[tuple[bytes, str]], list[str], bool]:
        """Collect input images from ask-injected context, attachments, replies, and URLs."""

        notes: list[str] = []
        images: list[tuple[bytes, str]] = []
        had_candidates = False
        seen_attachment_ids: set[int] = set()
        seen_urls: set[str] = set()
        seen_message_links: set[int] = set()

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
            if len(new_data) >= MAX_IMAGE_BYTES:
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
            if size is not None and size >= MAX_IMAGE_BYTES:
                notes.append(
                    f"Skipped {source}: attachment is larger than {MAX_IMAGE_BYTES // (1024 * 1024)}MB."
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

            if len(data) >= MAX_IMAGE_BYTES:
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

            return att_list

        attachments = _iter_attachments()
        if attachments:
            had_candidates = True

        for att, source in attachments:
            await _read_attachment(att, source)

        # URLs embedded in the prompt
        url_matches = URL_PATTERN.findall(prompt)
        if url_matches:
            had_candidates = True
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
                    url = raw_url.rstrip("),.;&")
                    normalized_url = _normalize_discord_cdn_url(url)
                    if normalized_url in seen_urls:
                        continue
                    seen_urls.add(normalized_url)

                    message_match = MESSAGE_LINK_PATTERN.match(url)
                    if message_match:
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

                                content_length = resp.headers.get("Content-Length")
                                try:
                                    if content_length is not None and int(content_length) >= MAX_IMAGE_BYTES:
                                        notes.append(
                                            f"Skipped URL: file exceeded {MAX_IMAGE_BYTES // (1024 * 1024)}MB"
                                        )
                                        break
                                except ValueError:
                                    pass

                                data = await resp.content.read(MAX_IMAGE_BYTES + 1)
                                if len(data) >= MAX_IMAGE_BYTES:
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
                                    break

                                norm_data, norm_name = normalized
                                images.append((norm_data, norm_name))
                                break
                            finally:
                                resp.release()
                    except Exception as exc:
                        notes.append(
                            f"Skipped URL (download failed): {normalized_url} "
                            f"({type(exc).__name__}: {str(exc)[:120]})"
                        )

        return images, notes, had_candidates

    @commands.hybrid_command(
        name="image",
        description="Generate an image from a text prompt using GPT Image.",
        cooldown_after_parsing=True,
        cooldown=commands.CooldownMapping.from_cooldown(1, 10, commands.BucketType.user),
        help=(
            "Generate or edit a 1024x1024 image with gpt-image-1.5. Attach images, drop public HTTPS image URLs, or paste Discord message links with attachments in the prompt to edit; first image is the base, the rest are references (up to 16 inputs total, each under 50MB).\n\n"
            "**Usage**: `/image <prompt>`\n"
            "**Examples**: `/image cozy cabin in the snow at night`, `/image this image https://... add a crown`, `/image use the image in this message link ... and add fireworks`\n"
            f"`{BOT_PREFIX}image add a neon crown to this character` (with an attachment)"
            "\nHEIC/HEIF uploads are accepted and converted internally when pillow-heif is available."
        ),
        extras={
            "category": "AI",
            "destination": "Generate or edit images from prompts, attachments, URLs, or message links.",
            "plus": "Edits when you attach/URL/link an image (first is base, others references); otherwise generates. HTTPS URLs only.",
            "pro": "Uses the Images API with gpt-image-1.5, supports attachments, reply images, ask-injected images, HTTPS URLs, and message links that contain attachments.",
        },
    )
    async def image(self, ctx: commands.Context, *, prompt: str) -> None:
        if not prompt or not prompt.strip():
            return await safe_reply(ctx, "Tell me what to draw first.", ephemeral=True, mention_author=False)

        prompt = prompt.strip()
        await defer_interaction(ctx)

        images, notes, had_candidates = await self._gather_images(ctx, prompt)
        display_prompt = _summarize_prompt_for_embed(prompt)

        if had_candidates and not images and notes:
            error_embed = discord.Embed(
                title="\u26a0\ufe0f No usable images to edit",
                description=(
                    "All provided image inputs were skipped or couldn't be decoded, "
                    "so I won't generate a new image for this request."
                ),
                color=0xFF0000,
            )
            note_text = "\n".join(notes)
            error_embed.add_field(name="Notes", value=note_text[:1024], inline=False)
            await safe_reply(ctx, embed=error_embed, ephemeral=True, mention_author=False)
            return

        status_embed = discord.Embed(
            title="🎨 Editing image…" if images else "🎨 Generating image…",
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

        if not getattr(self.client, "api_key", None):
            return await safe_reply(
                ctx,
                "OPENAI_TOKEN is not set, so I can't generate images right now.",
                ephemeral=True,
                mention_author=False,
            )

        try:
            request_kwargs = {
                "model": "gpt-image-1.5",
                "prompt": prompt,
                "size": "1024x1024",
                "output_format": "png",
                "stream": True,
                "partial_images": 3,
            }
            if OPENAI_OMIT is not None:
                request_kwargs["response_format"] = OPENAI_OMIT

            def _file_from_bytes(data: bytes, name: str) -> tuple[discord.File, str]:
                buf = BytesIO(data)
                buf.seek(0)
                file_obj = discord.File(buf, filename=name)
                return file_obj, f"attachment://{name}"

            partials: list[bytes] = []

            async def _send_partial(image_bytes: bytes, idx: int | None, editing: bool) -> None:
                partials.append(image_bytes)
                if not status_msg:
                    return
                partial_file, partial_url = _file_from_bytes(
                    image_bytes, f"partial{idx if idx is not None else len(partials)}.png"
                )
                partial_embed = discord.Embed(
                    title="🎨 Editing image… (partial)" if editing else "🎨 Generating image… (partial)",
                    description=display_prompt,
                    color=0x5865F2,
                )
                partial_embed.set_image(url=partial_url)
                try:
                    await status_msg.edit(embed=partial_embed, attachments=[partial_file])
                except Exception:
                    return

            async def _extract_final_and_partials():
                final_bytes: bytes | None = None

                # Use edits when at least one image is available; otherwise, fall back to generates with streaming.
                if images:
                    def _build_edit_kwargs(streaming: bool) -> dict:
                        edit_kwargs = {
                            "model": "gpt-image-1.5",
                            "prompt": prompt,
                            "image": [],
                            "size": "1024x1024",
                            "output_format": "png",
                        }
                        if streaming:
                            edit_kwargs["stream"] = True
                            edit_kwargs["partial_images"] = 3
                        if OPENAI_OMIT is not None:
                            edit_kwargs["response_format"] = OPENAI_OMIT

                        for idx, (data, name) in enumerate(images):
                            buf = BytesIO(data)
                            buf.name = name or f"image{idx}.png"
                            edit_kwargs.setdefault("image", []).append(buf)

                        return edit_kwargs

                    if self._async_client:
                        try:
                            edit_stream = await self.client.images.edit(**_build_edit_kwargs(streaming=True))
                            async for event in edit_stream:
                                evt_type = getattr(event, "type", "") or ""
                                b64_val = (
                                    getattr(event, "b64_json", None)
                                    or getattr(event, "image_base64", None)
                                    or getattr(event, "data", None)
                                )
                                if not b64_val:
                                    continue
                                try:
                                    image_bytes = base64.b64decode(b64_val)
                                except Exception:
                                    continue
                                if "partial" in evt_type:
                                    partial_idx = getattr(event, "partial_image_index", None)
                                    await _send_partial(image_bytes, partial_idx, editing=True)
                                final_bytes = image_bytes
                        except Exception as stream_exc:
                            log.warning("Streaming image edit failed; falling back to non-streaming.", exc_info=stream_exc)

                    if final_bytes is None:
                        edit_request = _build_edit_kwargs(streaming=False)
                        result = await self._images_edit(**edit_request)
                        data_items = getattr(result, "data", None)
                        if data_items is None and isinstance(result, dict):
                            data_items = result.get("data")
                        data_items = data_items or []
                        if not data_items:
                            raise RuntimeError("No image returned")
                        data = data_items[0]
                        b64_val = getattr(data, "b64_json", None) or getattr(data, "image_base64", None) or (
                            data.get("b64_json") if isinstance(data, dict) else None
                        )
                        if not b64_val:
                            raise RuntimeError("No image returned")
                        final_bytes = base64.b64decode(b64_val)
                elif self._async_client:
                    stream = await self.client.images.generate(**request_kwargs)
                    async for event in stream:
                        evt_type = getattr(event, "type", "") or ""
                        b64_val = (
                            getattr(event, "b64_json", None)
                            or getattr(event, "image_base64", None)
                            or getattr(event, "data", None)
                        )
                        if not b64_val:
                            continue
                        try:
                            image_bytes = base64.b64decode(b64_val)
                        except Exception:
                            continue
                        if "partial" in evt_type:
                            partial_idx = getattr(event, "partial_image_index", None)
                            await _send_partial(image_bytes, partial_idx, editing=False)
                        final_bytes = image_bytes
                else:
                    # Fallback: non-streaming generation path
                    request_copy = dict(request_kwargs)
                    request_copy.pop("stream", None)
                    request_copy.pop("partial_images", None)
                    result = await self._images_generate(**request_copy)
                    data_items = getattr(result, "data", None)
                    if data_items is None and isinstance(result, dict):
                        data_items = result.get("data")
                    data_items = data_items or []
                    if not data_items:
                        raise RuntimeError("No image returned")
                    data = data_items[0]
                    b64_val = getattr(data, "b64_json", None) or getattr(data, "image_base64", None) or (
                        data.get("b64_json") if isinstance(data, dict) else None
                    )
                    if not b64_val:
                        raise RuntimeError("No image returned")
                    final_bytes = base64.b64decode(b64_val)
                return final_bytes, partials

            final_bytes, partials = await _extract_final_and_partials()
            if not final_bytes:
                raise RuntimeError("No image returned")

            final_file, final_url = _file_from_bytes(final_bytes, "image.png")

            embed = discord.Embed(
                title="🖼️ Image Edited" if images else "🖼️ Image Generated",
                description=prompt,
                color=0x5865F2,
            )
            embed.set_image(url=final_url)
            embed.set_footer(text="Crafted with care ✨")

            if notes:
                note_text = "\n".join(notes)
                embed.add_field(name="Notes", value=note_text[:1024], inline=False)

            if status_msg:
                try:
                    await status_msg.edit(embed=embed, attachments=[final_file])
                    return
                except Exception:
                    pass

            # Fallback if we couldn't edit the status message
            if ctx.interaction:
                await ctx.interaction.followup.send(embed=embed, file=final_file, ephemeral=False)
            else:
                await ctx.reply(embed=embed, file=final_file, mention_author=False)
        except Exception as exc:
            log.exception("Failed to generate image")
            description = "An error occurred while generating the image. Try again later."
            if isinstance(exc, BadRequestError):
                detail = getattr(exc, "message", str(exc))
                description = f"OpenAI rejected the request: {detail}"
                if "invalid_image_file" in str(getattr(exc, "code", "")) or "Invalid image file" in detail:
                    description += (
                        "\nMake sure your images are PNG, JPEG/JPG, or WEBP, under 50MB, and still accessible. "
                        "Message links must point to a message with downloadable image attachments."
                        "\nIf you pasted a Discord CDN/preview link, download the image and re-upload it here as a PNG/JPEG/WEBP file."
                    )
            error_embed = discord.Embed(
                title="\u26A0\ufe0f Image Failed",
                description=description,
                color=0xFF0000,
            )
            if notes:
                note_text = "\n".join(notes)
                error_embed.add_field(name="Notes", value=note_text[:1024], inline=False)
            try:
                if status_msg:
                    await status_msg.edit(embed=error_embed, attachments=[])
                else:
                    await safe_reply(ctx, embed=error_embed, ephemeral=True, mention_author=False)
            except Exception:
                return


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Image(bot))
