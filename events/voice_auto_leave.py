from __future__ import annotations

import discord
from discord.ext import commands

from events import EventInfo
from music import get_player


EVENT_INFO = EventInfo(
    name="voice_auto_leave",
    destination="Leave voice automatically when only bots remain in the voice channel.",
    plus="Respects the Auto Leave toggle and triggers even while music is playing.",
    category="Music",
)


class VoiceAutoLeave(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.Cog.listener()
    async def on_voice_state_update(
        self,
        member: discord.Member | discord.User,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        if member.bot:
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
