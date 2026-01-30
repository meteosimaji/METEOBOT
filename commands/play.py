from __future__ import annotations
# mypy: ignore-errors

import asyncio
import logging
import re
from collections.abc import AsyncIterator
import os
import socket
from ipaddress import ip_address
from urllib.error import URLError
from urllib.parse import urlparse

import discord
from discord.ext import commands
import yt_dlp

from music import Track, VoiceConnectionError, get_player, resolve_track
from utils import BOT_PREFIX, defer_interaction, humanize_delta, safe_reply, tag_error_text

log = logging.getLogger(__name__)

SUPPORTED_AUDIO = {"wav", "flac", "mp3", "m4a", "aac", "ogg", "opus"}
SUPPORTED_VIDEO = {"mp4", "mkv", "webm", "mov", "mka"}
SUPPORTED = SUPPORTED_AUDIO | SUPPORTED_VIDEO

MIME_EXT_MAP = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/flac": "flac",
    "audio/aac": "aac",
    "audio/ogg": "ogg",
    "audio/opus": "opus",
    "video/mp4": "mp4",
    "video/x-matroska": "mkv",
    "video/webm": "webm",
    "video/quicktime": "mov",
    "audio/x-matroska": "mka",
}


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _is_local_path(s: str) -> bool:
    # Windows drive (C:\), UNC (\\server\share), absolute posix (/), or relative ./ ../
    if _is_url(s):
        return False
    if s.startswith(("./", "../", "/")):
        return True
    if re.match(r"^[a-zA-Z]:[\\/]", s):
        return True
    if s.startswith("\\\\"):
        return True
    return False


def _is_blocked_host(host: str) -> bool:
    try:
        ip = ip_address(host)
    except ValueError:
        return host.lower() in {"localhost", "ip6-localhost"}
    return ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast


def _is_safe_direct_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.hostname
    if not host:
        return False
    return not _is_blocked_host(host)


def _guess_name_and_ext(src: str) -> tuple[str, str]:
    path = src
    try:
        if _is_url(src):
            path = urlparse(src).path or src
    except Exception:
        pass
    name = os.path.basename(path) or "file"
    _, dot, ext = name.rpartition(".")
    return name, (ext.lower() if dot else "")


def _attachment_to_src(att: discord.Attachment) -> tuple[str, str]:
    name = att.filename or "attachment"
    _, dot, ext = name.rpartition(".")
    if not ext and att.content_type:
        mime = att.content_type.split(";")[0].lower()
        ext = MIME_EXT_MAP.get(mime, "")
    return att.url, (ext.lower() if ext else "")


async def _ensure_joined(ctx: commands.Context) -> tuple[bool, object]:
    """Join the author's voice channel and return (ok, player|reason)."""
    if not getattr(ctx.author, "voice", None) or not getattr(ctx.author.voice, "channel", None):
        return False, "Join a voice channel first."

    player = get_player(ctx.bot, ctx.guild)  # type: ignore[arg-type]
    try:
        await player.join(ctx.author.voice.channel)  # type: ignore[arg-type]
        if not player.voice or not player.voice.is_connected():
            return False, "Voice connection failed (disconnected immediately). Try again or check your network/firewall."
    except (VoiceConnectionError, asyncio.TimeoutError):
        return False, "Couldn't join your voice channel."
    return True, player


class Play(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="play",
        description="Play a song, playlist or media file",
        help=(
            "Queue music from a link (YouTube, Spotify, Niconico) or a search phrase. "
            "You can also attach media files (prefix command) or pass a direct file URL.\n\n"
            "**Usage**: `/play <query or url>`\n"
            "**Examples**: `/play never gonna give you up`\n"
            "`/play https://youtu.be/dQw4w9WgXcQ`\n"
            f"`{BOT_PREFIX}play` with file attachments\n"
            f"`{BOT_PREFIX}play https://example.com/file.mp3`\n"
            "Supported direct files: wav, flac, mp3, m4a/aac, ogg/opus, mp4/mkv/webm/mov/mka"
        ),
        extras={
            "category": "Music",
            "pro": (
                "Supports URLs from YouTube, Spotify and Niconico as well as "
                "plain search terms. Prefix usage can also attach media files; "
                "video is decoded as audio-only. Playlist URLs are expanded with yt-dlp. "
                "LLM/tooling can invoke /play with a single text argument (query or URL)."
            ),
        },
    )
    async def play(
        self,
        ctx: commands.Context,
        *,
        source: str | None = None,
    ) -> None:
        """Play music from a search term or URL (LLM/tooling must supply one text arg)."""
        if ctx.guild is None:
            return await safe_reply(
                ctx, tag_error_text("This command can only be used in a server."), mention_author=False
            )
        await defer_interaction(ctx)

        # Attachments (prefix or slash) count as file candidates.
        attachments: list[discord.Attachment] = []
        if ctx.message and ctx.message.attachments:
            attachments.extend(ctx.message.attachments)
        if getattr(ctx, "interaction", None) and getattr(ctx.interaction, "attachments", None):
            attachments.extend(list(ctx.interaction.attachments))

        has_attachments = bool(attachments)

        # Treat URLs as direct files only when they include an extension
        is_direct_file = False
        is_local_path = False
        title_ext = ("", "")
        if source:
            title_ext = _guess_name_and_ext(source)
            name, ext = title_ext
            is_local_path = _is_local_path(source)
            is_direct_file = _is_url(source) and (ext in SUPPORTED)

        if not source and not has_attachments:
            return await safe_reply(
                ctx,
                tag_error_text("Provide a song name, URL or attach a file."),
                ephemeral=True,
                mention_author=False,
            )

        # File-mode: attachments, direct file URLs, or (admin-only) local paths.
        if has_attachments or is_local_path or is_direct_file:
            candidates: list[tuple[str, str, str]] = []
            rejected: list[str] = []
            policy_rejects: list[str] = []
            direct_rejects: list[str] = []

            if source and (is_local_path or is_direct_file):
                name, ext = title_ext
                if is_local_path and not (
                    hasattr(ctx.author, "guild_permissions")
                    and ctx.author.guild_permissions.administrator
                ):
                    rejected.append(name or "local file")
                    policy_rejects.append(name or "local file")
                elif is_direct_file and not _is_safe_direct_url(source):
                    rejected.append(name or "direct file")
                    direct_rejects.append(name or "direct file")
                else:
                    candidates.append((name, source, ext))

            for att in attachments:
                url, ext = _attachment_to_src(att)
                candidates.append((att.filename or "attachment", url, ext))

            # Deduplicate by src URL/path.
            if candidates:
                seen: set[str] = set()
                unique: list[tuple[str, str, str]] = []
                for title, src, ext in candidates:
                    if src in seen:
                        continue
                    seen.add(src)
                    unique.append((title, src, ext))
                candidates = unique

            if not candidates and not rejected:
                return await safe_reply(
                    ctx,
                    tag_error_text("Attach a file or provide a URL/path first."),
                    mention_author=False,
                )

            accepted: list[tuple[str, str]] = []
            for title, src, ext in candidates:
                if ext in SUPPORTED:
                    accepted.append((title, src))
                else:
                    rejected.append(title)

            if not accepted:
                if direct_rejects:
                    return await safe_reply(
                        ctx,
                        tag_error_text("🚫 Direct URLs to private or local addresses are not allowed."),
                        mention_author=False,
                    )
                if policy_rejects and set(rejected) == set(policy_rejects):
                    return await safe_reply(
                        ctx,
                        tag_error_text("🚫 Local file paths are **admin-only** for security."),
                        mention_author=False,
                    )
                return await safe_reply(
                    ctx,
                    tag_error_text(
                        "❌ This file type isn't supported.\n"
                        "Direct files: wav/flac/mp3/m4a/aac/ogg/opus/mp4/mkv/webm/mov/mka\n"
                        "Tip: YouTube/Spotify/Niconico URLs are not direct links, so just paste them as-is."
                    ),
                    mention_author=False,
                )

            ok, res = await _ensure_joined(ctx)
            if not ok:
                return await safe_reply(ctx, tag_error_text(str(res)), ephemeral=True, mention_author=False)
            player = res  # type: ignore[assignment]

            first: Track | None = None
            count = 0
            await player.start_playlist()
            try:
                for title, src in accepted:
                    track = Track(url=src, title=title, duration=0.0, page_url=src)
                    await player.add(track, from_playlist=True)
                    count += 1
                    if first is None:
                        first = track
            finally:
                await player.end_playlist()

            remove_tip = (
                f"Undo a mistaken add with /remove or `{BOT_PREFIX}remove`; "
                "use /remove 1 for the latest track."
            )
            if count == 1 and first:
                embed = discord.Embed(
                    title="🎵 Added to Queue",
                    description=(
                        f"[{first.title}]({first.page_url})" if _is_url(first.page_url) else f"{first.title}"
                    ),
                    color=0x1DB954,
                )
                add_id = f"A{first.add_id}" if first.add_id is not None else "A?"
                embed.add_field(name="ID", value=add_id)
                embed.set_footer(text=remove_tip)
                await ctx.reply(embed=embed, mention_author=False)
            else:
                head = first.title if first else "file"
                embed = discord.Embed(
                    title="🎞️ Added multiple to Queue",
                    description=f"Queued **{count}** file(s) starting with **{head}**",
                    color=0x1DB954,
                )
                embed.set_footer(text=remove_tip)
                await ctx.reply(embed=embed, mention_author=False)

            if rejected:
                ext_rejected = [r for r in rejected if r not in policy_rejects + direct_rejects]
                if ext_rejected:
                    await safe_reply(
                        ctx,
                        tag_error_text("⚠️ Skipped unsupported: " + ", ".join(ext_rejected)),
                        mention_author=False,
                    )
                if policy_rejects:
                    await safe_reply(
                        ctx,
                        tag_error_text("🚫 Skipped local path (admin-only): " + ", ".join(policy_rejects)),
                        mention_author=False,
                    )
                if direct_rejects:
                    await safe_reply(
                        ctx,
                        tag_error_text("🚫 Skipped unsafe direct URL: " + ", ".join(direct_rejects)),
                        mention_author=False,
                    )
            return

        # Search / page URL mode => yt-dlp resolve
        await self._queue_from_resolve(ctx, source)

    async def _queue_from_resolve(self, ctx: commands.Context, source: str | None) -> None:
        if not source:
            return await safe_reply(
                ctx, tag_error_text("Provide a song name or URL."), ephemeral=True, mention_author=False
            )

        ok, res = await _ensure_joined(ctx)
        if not ok:
            return await safe_reply(ctx, tag_error_text(str(res)), ephemeral=True, mention_author=False)
        player = res  # type: ignore[assignment]

        # resolve_track may return Track or AsyncIterator[Track] for playlists.
        try:
            result = await resolve_track(source)
        except yt_dlp.utils.DownloadError:
            log.exception("yt-dlp extraction failed")
            return await safe_reply(
                ctx,
                tag_error_text(
                    "No results found. Try adding 'MV' or the official artist name, "
                    "or paste the exact video URL if you have it."
                ),
                ephemeral=True,
                mention_author=False,
            )
        except (URLError, socket.gaierror, ConnectionError, TimeoutError):
            log.exception("Network error during yt-dlp extraction")
            return await safe_reply(
                ctx,
                tag_error_text(
                    "The source service is unreachable; try again later. If you have the exact URL, "
                    "you can paste it to avoid search misses."
                ),
                ephemeral=True,
                mention_author=False,
            )
        except Exception:
            log.exception("yt-dlp extraction failed")
            return await safe_reply(
                ctx,
                tag_error_text("Failed to fetch that track. If this is a specific video, try the direct URL."),
                ephemeral=True,
                mention_author=False,
            )

        log.info("%s requested %s", ctx.author, source)

        if isinstance(result, AsyncIterator):
            await player.start_playlist()
            count = 0
            first: Track | None = None
            message: discord.Message | None = None
            try:
                async for track in result:  # type: ignore[assignment]
                    await player.add(track, from_playlist=True)
                    count += 1
                    if count == 1:
                        first = track
                        embed = discord.Embed(
                            title="🎵 Added to Queue",
                            description=f"[{track.title}]({track.page_url})",
                            color=0x1DB954,
                        )
                        if track.duration:
                            embed.add_field(name="Duration", value=humanize_delta(track.duration))
                        add_id = f"A{track.add_id}" if track.add_id is not None else "A?"
                        embed.add_field(name="ID", value=add_id)
                        if track.related:
                            related_lines = [
                                f"R{i + 1}. [{item['title']}]({item['url']})"
                                for i, item in enumerate(track.related[:3])
                            ]
                            if related_lines:
                                embed.add_field(name="Related", value="\n".join(related_lines), inline=False)
                        embed.set_footer(
                            text=(
                                f"To cancel this add, run /remove or `{BOT_PREFIX}remove`; "
                                "use /remove 1 for the latest track. "
                                "If it's wrong, use a Related URL with /play to lock the exact video."
                            )
                        )
                        message = await ctx.reply(embed=embed, mention_author=False)
            finally:
                await player.end_playlist()
            if count == 0:
                return await safe_reply(
                    ctx, tag_error_text("No tracks found."), ephemeral=True, mention_author=False
                )
            if count > 1 and message and first:
                embed = discord.Embed(
                    title="🎵 Playlist Added",
                    description=f"{count} tracks starting with [{first.title}]({first.page_url})",
                    color=0x1DB954,
                )
                embed.set_footer(
                    text=(
                        f"Undo mistaken additions with /remove or `{BOT_PREFIX}remove`; "
                        "use /remove 1 for the most recent track."
                    )
                )
                await message.edit(embed=embed)
            return

        # Single track
        track = result  # type: ignore[assignment]
        await player.add(track)
        embed = discord.Embed(
            title="🎵 Added to Queue",
            description=f"[{track.title}]({track.page_url})",
            color=0x1DB954,
        )
        if track.duration:
            embed.add_field(name="Duration", value=humanize_delta(track.duration))
        add_id = f"A{track.add_id}" if track.add_id is not None else "A?"
        embed.add_field(name="ID", value=add_id)
        if track.related:
            related_lines = [
                f"R{i + 1}. [{item['title']}]({item['url']})" for i, item in enumerate(track.related[:3])
            ]
            if related_lines:
                embed.add_field(name="Related", value="\n".join(related_lines), inline=False)
        embed.set_footer(
            text=(
                f"Need to cancel? Run /remove or `{BOT_PREFIX}remove`; "
                "use /remove 1 for the latest track. "
                "If it's wrong, use a Related URL with /play to lock the exact video."
            )
        )
        await ctx.reply(embed=embed, mention_author=False)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Play(bot))
