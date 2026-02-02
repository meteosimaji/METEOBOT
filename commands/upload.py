from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import uuid

import discord
from discord.ext import commands

from commands import ask as ask_module
from utils import defer_interaction, safe_reply, tag_error_text

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = REPO_ROOT / "data" / "uploads"


class Upload(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    def _get_ask_cog(self) -> ask_module.Ask | None:
        ask_cog = self.bot.get_cog("Ask")
        if isinstance(ask_cog, ask_module.Ask):
            return ask_cog
        return None

    @staticmethod
    def _attachment_filename(att: discord.Attachment) -> str:
        name = (att.filename or "").strip()
        return Path(name).name or "upload"

    async def _resolve_mnt_data_link(
        self, ask_cog: ask_module.Ask, ctx: commands.Context, path: str
    ) -> dict[str, Any] | None:
        entries = ask_cog._prune_link_context(ask_cog._load_link_context(ctx))
        for entry in reversed(entries):
            entry_path = entry.get("path")
            if isinstance(entry_path, str) and entry_path == path:
                return entry
        return None

    @commands.hybrid_command(
        name="upload",
        description="Upload a file to simajilord.com/save and return a temporary link.",
        help=(
            "Provide a /mnt/data path from /ask outputs or attach a file, and the bot returns "
            "a temporary simajilord.com/save link you can share or use with /play."
        ),
        usage="<path?>",
        extras={
            "category": "Tools",
            "pro": (
                "Accepts a /mnt/data path from /ask outputs or file attachments, then returns a "
                "temporary simajilord.com/save link. Links expire automatically."
            ),
        },
    )
    async def upload(self, ctx: commands.Context, *, path: str | None = None) -> None:
        await defer_interaction(ctx)
        ask_cog = self._get_ask_cog()
        if ask_cog is None:
            return await safe_reply(
                ctx,
                tag_error_text("Upload is unavailable because the Ask cog is not loaded."),
                ephemeral=True,
                mention_author=False,
            )

        path = (path or "").strip()
        attachments: list[discord.Attachment] = []
        if ctx.message and ctx.message.attachments:
            attachments.extend(ctx.message.attachments)
        if getattr(ctx, "interaction", None) and getattr(ctx.interaction, "attachments", None):
            attachments.extend(list(ctx.interaction.attachments))

        if path:
            if not path.startswith("/mnt/data/"):
                return await safe_reply(
                    ctx,
                    tag_error_text("Only /mnt/data/... paths are supported. Attach a file instead."),
                    ephemeral=True,
                    mention_author=False,
                )
            match = await self._resolve_mnt_data_link(ask_cog, ctx, path)
            if not match:
                return await safe_reply(
                    ctx,
                    tag_error_text("No upload link found for that /mnt/data path yet."),
                    ephemeral=True,
                    mention_author=False,
                )
            url = match.get("url")
            if not isinstance(url, str) or not url:
                return await safe_reply(
                    ctx,
                    tag_error_text("Upload link is unavailable for that path."),
                    ephemeral=True,
                    mention_author=False,
                )
            embed = discord.Embed(
                title="📎 Upload Link",
                description=f"Download URL: {url}",
                color=0x5865F2,
            )
            expires_at = match.get("link_expires_at")
            if isinstance(expires_at, str) and expires_at:
                embed.add_field(name="Expires", value=expires_at, inline=False)
            embed.add_field(name="Path", value=path, inline=False)
            await ctx.reply(embed=embed, mention_author=False)
            return

        if not attachments:
            return await safe_reply(
                ctx,
                tag_error_text("Provide a /mnt/data path or attach a file."),
                ephemeral=True,
                mention_author=False,
            )

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        results: list[tuple[str, str]] = []
        for att in attachments:
            if att.size and att.size > ask_module.MAX_ATTACHMENT_DOWNLOAD_BYTES:
                await safe_reply(
                    ctx,
                    tag_error_text(
                        f"{att.filename or 'attachment'} exceeds the size limit "
                        f"({ask_module.MAX_ATTACHMENT_DOWNLOAD_BYTES} bytes)."
                    ),
                    mention_author=False,
                )
                continue
            filename = self._attachment_filename(att)
            dest_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
            try:
                await att.save(dest_path)
            except Exception:
                log.exception("Failed to save attachment for upload.")
                await safe_reply(
                    ctx,
                    tag_error_text(f"Failed to download {filename}."),
                    mention_author=False,
                )
                continue

            link = await ask_cog.register_download(
                dest_path,
                filename=filename,
                expires_s=ask_module.ASK_CONTAINER_FILE_LINK_TTL_S,
                keep_file=False,
            )
            if not link:
                await safe_reply(
                    ctx,
                    tag_error_text(f"Failed to create a link for {filename}."),
                    mention_author=False,
                )
                continue

            created_at = datetime.now(timezone.utc)
            expires_at = created_at + timedelta(seconds=ask_module.ASK_CONTAINER_FILE_LINK_TTL_S)
            entry = {
                "id": ask_cog._next_link_context_id(
                    ask_cog._prune_link_context(ask_cog._load_link_context(ctx))
                ),
                "created_at": created_at.isoformat(),
                "url": link,
                "filename": filename,
                "path": None,
                "bytes": dest_path.stat().st_size,
                "content_type": (att.content_type or "").split(";", 1)[0] if att.content_type else None,
                "source": "upload_command",
                "link_expires_at": expires_at.isoformat(),
                "file_expires_at": expires_at.isoformat(),
                "local_path": str(dest_path.relative_to(REPO_ROOT)),
            }
            ask_cog._append_link_context(ctx, entry)
            results.append((filename, link))

        if not results:
            return

        lines = [f"- {name}: {url}" for name, url in results]
        embed = discord.Embed(
            title="📎 Uploaded Files",
            description="\n".join(lines),
            color=0x5865F2,
        )
        await ctx.reply(embed=embed, mention_author=False)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Upload(bot))
