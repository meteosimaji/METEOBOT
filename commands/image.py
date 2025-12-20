import asyncio
import base64
import importlib
import logging
import os
from io import BytesIO

import discord
from discord.ext import commands

from utils import BOT_PREFIX, defer_interaction, safe_reply

log = logging.getLogger(__name__)

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

    @commands.hybrid_command(
        name="image",
        description="Generate an image from a text prompt using GPT Image.",
        cooldown_after_parsing=True,
        cooldown=commands.CooldownMapping.from_cooldown(1, 10, commands.BucketType.user),
        help=(
            "Create a 1024x1024 image from your prompt using gpt-image-1.5.\n\n"
            "**Usage**: `/image <prompt>`\n"
            "**Examples**: `/image cozy cabin in the snow at night`\n"
            f"`{BOT_PREFIX}image futuristic city skyline at sunrise`"
        ),
        extras={
            "category": "AI",
            "destination": "Turn your prompt into a generated image.",
            "plus": "Generates a single 1024x1024 PNG using gpt-image-1.5.",
            "pro": "Uses the Images API with gpt-image-1.5 and returns the image as an attachment.",
        },
    )
    async def image(self, ctx: commands.Context, *, prompt: str) -> None:
        if not prompt or not prompt.strip():
            return await safe_reply(ctx, "Tell me what to draw first.", ephemeral=True, mention_author=False)

        prompt = prompt.strip()
        await defer_interaction(ctx)

        status_embed = discord.Embed(
            title="🎨 Generating image…",
            description=prompt,
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

            async def _extract_final_and_partials():
                final_bytes: bytes | None = None
                partials: list[bytes] = []
                if self._async_client:
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
                            partials.append(image_bytes)
                        final_bytes = image_bytes
                else:
                    # Fallback: non-streaming path
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

            def _file_from_bytes(data: bytes, name: str) -> tuple[discord.File, str]:
                buf = BytesIO(data)
                buf.seek(0)
                file_obj = discord.File(buf, filename=name)
                return file_obj, f"attachment://{name}"

            # Show the latest partial if available
            if status_msg and partials:
                last_partial = partials[-1]
                partial_file, partial_url = _file_from_bytes(last_partial, "partial.png")
                partial_embed = discord.Embed(
                    title="🎨 Generating image… (partial)",
                    description=prompt,
                    color=0x5865F2,
                )
                partial_embed.set_image(url=partial_url)
                try:
                    await status_msg.edit(embed=partial_embed, attachments=[partial_file])
                except Exception:
                    pass

            final_file, final_url = _file_from_bytes(final_bytes, "image.png")

            embed = discord.Embed(
                title="🖼️ Image Generated",
                description=prompt,
                color=0x5865F2,
            )
            embed.set_image(url=final_url)
            embed.set_footer(text="Crafted with care ✨")

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
