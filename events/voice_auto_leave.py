from __future__ import annotations

# mypy: ignore-errors
import asyncio
import logging

import discord
from discord.ext import commands

from events import EventInfo
from music import get_player

log = logging.getLogger(__name__)


EVENT_INFO = EventInfo(
    name="voice_auto_leave",
    destination="Leave voice automatically when only bots remain in the voice channel.",
    plus="Respects the Auto Leave toggle and triggers even while music is playing.",
    category="Music",
)


class VoiceAutoLeave(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self._cleanup_tasks: dict[int, asyncio.Task] = {}

    def cog_unload(self) -> None:
        for task in self._cleanup_tasks.values():
            task.cancel()
        self._cleanup_tasks.clear()

    def _spawn_cleanup(self, player, guild_id: int) -> None:
        existing = self._cleanup_tasks.pop(guild_id, None)
        if existing:
            existing.cancel()

        async def _run() -> None:
            await asyncio.sleep(10)
            player.sync_voice_client()
            if player.voice and player.voice.is_connected():
                return
            await player.cleanup()

        task = asyncio.create_task(_run())
        self._cleanup_tasks[guild_id] = task

        def _done(t: asyncio.Task) -> None:
            try:
                t.result()
            except asyncio.CancelledError:
                return
            except Exception:
                log.exception("voice_auto_leave cleanup task failed")
            finally:
                if self._cleanup_tasks.get(guild_id) is t:
                    self._cleanup_tasks.pop(guild_id, None)

        task.add_done_callback(_done)

    @commands.Cog.listener()
    async def on_voice_state_update(
        self,
        member: discord.Member | discord.User,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        if member.bot:
            if self.bot.user and member.id == self.bot.user.id:
                if before.channel is not None and after.channel is None:
                    player = get_player(self.bot, before.channel.guild)
                    player.sync_voice_client()
                    player.last_channel_id = None
                    player.ignore_after = True
                    self._spawn_cleanup(player, before.channel.guild.id)
            return

        channel = before.channel or after.channel
        if channel is None:
            return

        player = get_player(self.bot, channel.guild)
        player.sync_voice_client()
        if not player.voice or not player.voice.channel:
            return

        voice_channel_id = player.voice.channel.id
        affected_channels = {c.id for c in (before.channel, after.channel) if c}
        if voice_channel_id not in affected_channels:
            return

        player.request_lonely_auto_leave(delay=10.0)


async def setup(bot: commands.Bot) -> None:
    if not hasattr(bot, "events"):
        bot.events = []
    bot.events.append(EVENT_INFO)
    await bot.add_cog(VoiceAutoLeave(bot))
