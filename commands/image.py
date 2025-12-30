import asyncio
import base64
import importlib
import logging
import os
import re
import socket
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
import ipaddress

import discord
from discord.ext import commands

from utils import BOT_PREFIX, defer_interaction, safe_reply

log = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 50 * 1024 * 1024  # 50MB per input image (Images API limit)
MAX_IMAGE_COUNT = 16  # Images API accepts up to 16 input images
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
URL_PATTERN = re.compile(r"https?://[^\s<>]+")
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
    """Hide long URLs and trim prompt text for cleaner embeds."""
    cleaned = URL_PATTERN.sub("[link]", prompt).strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1].rstrip() + "…"
    return cleaned or "[prompt hidden]"

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
    ) -> tuple[list[tuple[bytes, str]], list[str]]:
        """Collect input images from ask-injected context, attachments, replies, and URLs."""

        notes: list[str] = []
        images: list[tuple[bytes, str]] = []
        seen_attachment_ids: set[int] = set()
        seen_urls: set[str] = set()

        async def _read_attachment(att: discord.Attachment, source: str) -> None:
            att_id = getattr(att, "id", None)
            if att_id is not None and att_id in seen_attachment_ids:
                return
            if att_id is not None:
                seen_attachment_ids.add(att_id)

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

            filename = getattr(att, "filename", "image.png") or "image.png"
            images.append((data, filename))

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

        for att, source in _iter_attachments():
            await _read_attachment(att, source)

        # URLs embedded in the prompt
        url_matches = URL_PATTERN.findall(prompt)
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

                for raw_url in url_matches:
                    url = raw_url.rstrip("),.;")
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    if len(images) >= MAX_IMAGE_COUNT:
                        notes.append(
                            f"Skipped URL {url}: reached the {MAX_IMAGE_COUNT}-image limit for edits."
                        )
                        continue

                    parsed = urlparse(url)
                    if parsed.scheme != "https":
                        notes.append("Skipped URL (HTTPS required): " + url)
                        continue

                    hostname = parsed.hostname
                    if not hostname:
                        notes.append("Skipped URL (invalid host): " + url)
                        continue

                    if await _is_blocked_host(hostname):
                        notes.append("Skipped URL (host blocked): " + url)
                        continue

                    try:
                        async with session.get(url, allow_redirects=False) as resp:
                            if resp.status != 200:
                                notes.append(f"Skipped URL ({resp.status}): {url}")
                                continue

                            content_type = (resp.headers.get("Content-Type", "") or "").split(
                                ";", 1
                            )[0].lower()
                            if content_type not in ALLOWED_IMAGE_TYPES:
                                notes.append(f"Skipped URL (not an image): {url}")
                                continue

                            content_length = resp.headers.get("Content-Length")
                            try:
                                if content_length is not None and int(content_length) >= MAX_IMAGE_BYTES:
                                    notes.append(
                                        f"Skipped URL: file exceeded {MAX_IMAGE_BYTES // (1024 * 1024)}MB"
                                    )
                                    continue
                            except ValueError:
                                pass

                            data = await resp.content.read(MAX_IMAGE_BYTES + 1)
                            if len(data) >= MAX_IMAGE_BYTES:
                                notes.append(
                                    f"Skipped URL: file exceeded {MAX_IMAGE_BYTES // (1024 * 1024)}MB"
                                )
                                continue

                            name = Path(parsed.path).name or "image.png"
                            images.append((data, name))
                    except Exception:
                        notes.append(f"Skipped URL (download failed): {url}")

        return images, notes

    @commands.hybrid_command(
        name="image",
        description="Generate an image from a text prompt using GPT Image.",
        cooldown_after_parsing=True,
        cooldown=commands.CooldownMapping.from_cooldown(1, 10, commands.BucketType.user),
        help=(
            "Generate or edit a 1024x1024 image with gpt-image-1.5. Attach images or drop public HTTPS image URLs in the prompt to edit; first image is the base, the rest are references (up to 16 inputs total, each under 50MB).\n\n"
            "**Usage**: `/image <prompt>`\n"
            "**Examples**: `/image cozy cabin in the snow at night`, `/image this image https://... add a crown`\n"
            f"`{BOT_PREFIX}image add a neon crown to this character` (with an attachment)"
        ),
        extras={
            "category": "AI",
            "destination": "Generate or edit images from prompts, attachments, or URLs.",
            "plus": "Edits when you attach/URL an image (first is base, others references); otherwise generates. HTTPS URLs only.",
            "pro": "Uses the Images API with gpt-image-1.5, supports attachments, reply images, ask-injected images, and HTTPS URLs.",
        },
    )
    async def image(self, ctx: commands.Context, *, prompt: str) -> None:
        if not prompt or not prompt.strip():
            return await safe_reply(ctx, "Tell me what to draw first.", ephemeral=True, mention_author=False)

        prompt = prompt.strip()
        await defer_interaction(ctx)

        images, notes = await self._gather_images(ctx, prompt)
        display_prompt = _summarize_prompt_for_embed(prompt)

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
                description = f"OpenAI rejected the request: {getattr(exc, 'message', str(exc))}"
            error_embed = discord.Embed(
                title="\u26A0\ufe0f Image Failed",
                description=description,
                color=0xFF0000,
            )
            try:
                if status_msg:
                    await status_msg.edit(embed=error_embed, attachments=[])
                else:
                    await safe_reply(ctx, embed=error_embed, ephemeral=True, mention_author=False)
            except Exception:
                return


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Image(bot))
