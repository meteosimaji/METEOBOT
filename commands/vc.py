from __future__ import annotations
# mypy: ignore-errors

import logging

import discord
from discord.ext import commands

from music import VoiceConnectionError, get_player
from utils import BOT_PREFIX, defer_interaction, safe_reply, tag_error_text

log = logging.getLogger(__name__)


class VC(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="vc",
        description="Join, move, or leave a voice channel.",
        help=(
            "Join the caller's voice channel, move to a specified channel, or leave.\n\n"
            "**Usage**: `/vc` (join your current voice channel)\n"
            "**Usage**: `/vc <channel>` (join/move to a specific voice channel)\n"
            f"`{BOT_PREFIX}vc`\n"
            f"`{BOT_PREFIX}vc #voice-channel`\n"
            f"`{BOT_PREFIX}vc 123456789012345678`\n"
        ),
        extras={
            "category": "Music",
            "pro": (
                "Use without arguments to join your voice channel. Provide a channel or ID "
                "to move the bot between voice channels; if already in that channel, "
                "calling /vc again disconnects while holding the queue for 30 seconds."
            ),
        },
    )
    async def vc(
        self,
        ctx: commands.Context,
        channel: discord.VoiceChannel | discord.StageChannel | None = None,
        channel_id: str | None = None,
    ) -> None:
        if ctx.guild is None:
            return await safe_reply(
                ctx,
                tag_error_text("This command can only be used in a server."),
                mention_author=False,
            )
        await defer_interaction(ctx)

        target = channel
        if target is None and channel_id is not None:
            channel_id = channel_id.strip()
            if not channel_id.isdigit():
                return await safe_reply(
                    ctx,
                    tag_error_text("Please provide a numeric channel ID."),
                    mention_author=False,
                )
            resolved = ctx.guild.get_channel(int(channel_id))
            if isinstance(resolved, (discord.VoiceChannel, discord.StageChannel)):
                target = resolved
            elif resolved is None:
                return await safe_reply(
                    ctx,
                    tag_error_text("That channel ID could not be found."),
                    mention_author=False,
                )
            else:
                channel_type_names = {
                    discord.ChannelType.text: "text",
                    discord.ChannelType.news: "news",
                    discord.ChannelType.forum: "forum",
                    discord.ChannelType.category: "category",
                    discord.ChannelType.private_thread: "private thread",
                    discord.ChannelType.public_thread: "public thread",
                    discord.ChannelType.news_thread: "news thread",
                }
                channel_type = channel_type_names.get(
                    getattr(resolved, "type", None),
                    str(getattr(resolved, "type", "unknown")),
                )
                return await safe_reply(
                    ctx,
                    tag_error_text(
                        f"This is a {channel_type} channel. Please provide a voice or stage channel ID."
                    ),
                    mention_author=False,
                )

        author_voice = getattr(getattr(ctx.author, "voice", None), "channel", None)
        if target is None:
            if author_voice is None:
                return await safe_reply(
                    ctx,
                    tag_error_text("Join a voice channel first or pass a channel/ID."),
                    mention_author=False,
                )
            target = author_voice

        player = get_player(self.bot, ctx.guild)
        if player.voice and player.voice.is_connected():
            current = getattr(player.voice, "channel", None)
        else:
            current = None

        if current and current.id == target.id:
            await player.hold_disconnect(delay=30.0)
            log.info("%s left voice via /vc (hold)", ctx.author)
            embed = discord.Embed(
                title="👋 Left Voice Channel",
                description="Paused playback and kept the queue for 30 seconds.",
                color=0xE74C3C,
            )
            return await safe_reply(ctx, embed=embed, mention_author=False)

        try:
            await player.join(target)
        except VoiceConnectionError:
            return await safe_reply(
                ctx,
                tag_error_text("Couldn't join that voice channel."),
                mention_author=False,
            )

        resumed = await player.resume_after_hold()
        if current and current.id != target.id:
            log.info("%s moved voice to %s", ctx.author, target.id)
            description = f"Moved to {target.mention}."
            if resumed and player.current:
                description += " Resumed playback."
            embed = discord.Embed(
                title="➡️ Moved Voice Channel",
                description=description,
                color=0x3498DB,
            )
            return await safe_reply(ctx, embed=embed, mention_author=False)
        log.info("%s joined voice %s", ctx.author, target.id)
        description = f"Joined {target.mention}."
        if resumed and player.current:
            description += " Resumed playback."
        embed = discord.Embed(
            title="✅ Joined Voice Channel",
            description=description,
            color=0x2ECC71,
        )
        return await safe_reply(ctx, embed=embed, mention_author=False)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(VC(bot))
