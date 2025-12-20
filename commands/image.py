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

        if not getattr(self.client, "api_key", None):
            return await safe_reply(
                ctx,
                "OPENAI_TOKEN is not set, so I can't generate images right now.",
                ephemeral=True,
                mention_author=False,
            )

        try:
            result = await self._images_generate(
                model="gpt-image-1.5",
                prompt=prompt,
                size="1024x1024",
                response_format="b64_json",
            )
            data_items = getattr(result, "data", None) or []
            if not data_items:
                raise RuntimeError("No image returned")
            data = data_items[0]
            b64 = getattr(data, "b64_json", None)
            if not b64:
                raise RuntimeError("No image returned")

            image_bytes = base64.b64decode(b64)
            buf = BytesIO(image_bytes)
            buf.seek(0)
            file = discord.File(buf, filename="image.png")

            embed = discord.Embed(
                title="🖼️ Image Generated",
                description=prompt,
                color=0x5865F2,
            )
            embed.set_image(url="attachment://image.png")
            embed.set_footer(text="Model: gpt-image-1.5 • Crafted with care ✨")

            if ctx.interaction:
                await ctx.interaction.followup.send(embed=embed, file=file, ephemeral=False)
            else:
                await ctx.reply(embed=embed, file=file, mention_author=False)
        except Exception:
            log.exception("Failed to generate image")
            error_embed = discord.Embed(
                title="\u26A0\ufe0f Image Failed",
                description="An error occurred while generating the image. Try again later.",
                color=0xFF0000,
            )
            try:
                await safe_reply(ctx, embed=error_embed, ephemeral=True, mention_author=False)
            except Exception:
                return


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Image(bot))
