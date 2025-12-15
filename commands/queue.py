from __future__ import annotations

import asyncio
import logging
from typing import Optional

import discord
from discord.ext import commands

from music import get_player, progress_bar
from utils import format_timestamp, defer_interaction, BOT_PREFIX

log = logging.getLogger(__name__)


def _short(text: str, limit: int = 96) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ControlView(discord.ui.View):
    def __init__(self, player) -> None:
        super().__init__(timeout=600)  # auto-stop after 10 minutes
        self.player = player
        self.message: Optional[discord.Message] = None
        self.task: Optional[asyncio.Task] = None
        self.update_labels()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Ensure only listeners in the same voice channel can press controls."""
        user = interaction.user
        if not user or not getattr(user, "voice", None) or not user.voice or not user.voice.channel:
            await interaction.response.send_message("Join a voice channel first.", ephemeral=True)
            return False
        if not self.player.voice or not getattr(self.player.voice, "channel", None):
            await interaction.response.send_message("I'm not in voice right now.", ephemeral=True)
            return False
        if user.voice.channel.id != self.player.voice.channel.id:
            await interaction.response.send_message("Use the controls from my current voice channel.", ephemeral=True)
            return False
        return True

    async def start(self, message: discord.Message) -> None:
        """Begin automatic queue updates."""
        self.message = message
        self.task = asyncio.create_task(self.auto_update())

    async def auto_update(self) -> None:
        # Editing every second is spicy; 2s is friendlier to rate limits.
        delay = 2.0
        try:
            while True:
                await asyncio.sleep(delay)
                if not self.message or self.is_finished():
                    break
                if not self.player.voice or not self.player.voice.is_connected():
                    break
                self.update_labels()
                try:
                    await self.message.edit(embed=make_queue_embed(self.player), view=self)
                    delay = 2.0
                except discord.HTTPException as exc:
                    if exc.status == 429:
                        retry_after = getattr(exc, "retry_after", None)
                        if retry_after is None:
                            retry_after = exc.response.headers.get("Retry-After", 0)
                        try:
                            delay = max(float(retry_after), 2.0)
                        except (TypeError, ValueError):
                            delay = 2.0
                        continue
                    break
                except discord.NotFound:
                    break
        except asyncio.CancelledError:
            pass
        finally:
            self.task = None

    def stop(self) -> None:
        super().stop()
        if self.task and not self.task.done():
            self.task.cancel()

    async def on_timeout(self) -> None:  # pragma: no cover
        self.stop()

    def update_labels(self) -> None:
        self.btn_autoleave.label = f"Auto Leave: {'On' if self.player.auto_leave else 'Off'}"
        self.btn_loop.label = f"Loop: {self.player.loop.capitalize()}"
        self.btn_pause.label = "Resume" if self.player.voice and self.player.voice.is_paused() else "Pause"
        self.btn_speed.label = f"Speed {self.player.speed:.1f}×"
        self.btn_pitch.label = f"Pitch {self.player.pitch:.1f}×"
        self.btn_remove.disabled = not bool(self.player.queue)

    @discord.ui.button(label="Auto Leave", style=discord.ButtonStyle.secondary)
    async def btn_autoleave(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not interaction.response.is_done():
            await interaction.response.defer()
        self.player.auto_leave = not self.player.auto_leave
        log.info("%s toggled auto leave to %s", interaction.user, self.player.auto_leave)
        self.update_labels()
        try:
            await interaction.message.edit(embed=make_queue_embed(self.player), view=self)
        except discord.NotFound:
            pass

    @discord.ui.button(label="Loop", style=discord.ButtonStyle.secondary)
    async def btn_loop(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not interaction.response.is_done():
            await interaction.response.defer()
        if self.player.loop == "none":
            self.player.loop = "track"
        elif self.player.loop == "track":
            self.player.loop = "queue"
        else:
            self.player.loop = "none"
        log.info("%s set loop mode to %s", interaction.user, self.player.loop)
        self.update_labels()
        try:
            await interaction.message.edit(embed=make_queue_embed(self.player), view=self)
        except discord.NotFound:
            pass

    @discord.ui.button(label="Pause", style=discord.ButtonStyle.secondary)
    async def btn_pause(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not interaction.response.is_done():
            await interaction.response.defer()
        if self.player.voice and self.player.voice.is_paused():
            await self.player.resume()
        else:
            await self.player.pause()
        log.info("%s pressed pause", interaction.user)
        self.update_labels()
        try:
            await interaction.message.edit(embed=make_queue_embed(self.player), view=self)
        except discord.NotFound:
            pass

    @discord.ui.button(label="Speed", style=discord.ButtonStyle.secondary)
    async def btn_speed(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(SpeedModal(self))

    @discord.ui.button(label="Pitch", style=discord.ButtonStyle.secondary)
    async def btn_pitch(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(PitchModal(self))

    @discord.ui.button(label="Skip", style=discord.ButtonStyle.secondary)
    async def btn_skip(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not interaction.response.is_done():
            await interaction.response.defer()
        await self.player.skip()
        log.info("%s skipped track via control view", interaction.user)
        self.update_labels()
        try:
            await interaction.message.edit(embed=make_queue_embed(self.player), view=self)
        except discord.NotFound:
            pass

    @discord.ui.button(label="Stop", style=discord.ButtonStyle.danger)
    async def btn_stop(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not interaction.response.is_done():
            await interaction.response.defer()
        await self.player.stop()
        log.info("%s stopped playback via control view", interaction.user)
        self.update_labels()
        try:
            await interaction.message.edit(embed=make_queue_embed(self.player), view=self)
        except discord.NotFound:
            pass
        self.stop()

    @discord.ui.button(label="Remove", style=discord.ButtonStyle.danger)
    async def btn_remove(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self.player.queue:
            return await interaction.response.send_message("Queue is empty.", ephemeral=True)
        view = RemoveView(self.player, self)
        log.info("%s opened remove view", interaction.user)
        await interaction.response.send_message("Select a track to remove:", view=view, ephemeral=True)


class SpeedModal(discord.ui.Modal):
    def __init__(self, control: "ControlView") -> None:
        super().__init__(title="Adjust Speed")
        self.control = control
        self.player = control.player
        self.value = discord.ui.TextInput(label="Speed (0.5–2.0)", default=f"{self.player.speed:.2f}")
        self.add_item(self.value)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        try:
            speed = float(self.value.value)
        except ValueError:
            return await interaction.response.send_message("That's not a number.", ephemeral=True)
        if not 0.5 <= speed <= 2.0:
            return await interaction.response.send_message("Try a value between 0.5× and 2.0×.", ephemeral=True)

        pos = self.player.get_position()
        old_speed = self.player.speed
        self.player.speed = speed
        err = ""
        try:
            ok = await self.player.seek(pos)
        except Exception as exc:  # pragma: no cover
            ok = False
            err = str(exc)
        if not ok:
            self.player.speed = old_speed
            return await interaction.response.send_message(f"Failed to apply speed: {err or 'seek failed'}", ephemeral=True)

        self.control.update_labels()
        if self.control.message:
            await self.control.message.edit(embed=make_queue_embed(self.player), view=self.control)
        await interaction.response.send_message(f"Speed set to {speed:.2f}×.", ephemeral=True)


class PitchModal(discord.ui.Modal):
    def __init__(self, control: "ControlView") -> None:
        super().__init__(title="Adjust Pitch")
        self.control = control
        self.player = control.player
        self.value = discord.ui.TextInput(label="Pitch (0.5–2.0)", default=f"{self.player.pitch:.2f}")
        self.add_item(self.value)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        try:
            pitch = float(self.value.value)
        except ValueError:
            return await interaction.response.send_message("That's not a number.", ephemeral=True)
        if not 0.5 <= pitch <= 2.0:
            return await interaction.response.send_message("Try a value between 0.5× and 2.0×.", ephemeral=True)

        pos = self.player.get_position()
        old_pitch = self.player.pitch
        self.player.pitch = pitch
        err = ""
        try:
            ok = await self.player.seek(pos)
        except Exception as exc:  # pragma: no cover
            ok = False
            err = str(exc)
        if not ok:
            self.player.pitch = old_pitch
            return await interaction.response.send_message(f"Failed to apply pitch: {err or 'seek failed'}", ephemeral=True)

        self.control.update_labels()
        if self.control.message:
            await self.control.message.edit(embed=make_queue_embed(self.player), view=self.control)
        await interaction.response.send_message(f"Pitch set to {pitch:.2f}×.", ephemeral=True)


class RemoveView(discord.ui.View):
    def __init__(self, player, control: ControlView) -> None:
        super().__init__(timeout=60)
        self.player = player
        self.control = control
        options = [
            discord.SelectOption(label=_short(f"{i+1}. {t.title}"), value=str(i))
            for i, t in enumerate(list(player.queue)[:25])
        ]
        self.select = discord.ui.Select(placeholder="Track to remove", options=options)
        self.select.callback = self.select_callback
        self.add_item(self.select)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        user = interaction.user
        if not user or not getattr(user, "voice", None) or not user.voice or not user.voice.channel:
            await interaction.response.send_message("Join a voice channel first.", ephemeral=True)
            return False
        if not self.player.voice or not getattr(self.player.voice, "channel", None):
            await interaction.response.send_message("I'm not in voice right now.", ephemeral=True)
            return False
        if user.voice.channel.id != self.player.voice.channel.id:
            await interaction.response.send_message("Use the controls from my current voice channel.", ephemeral=True)
            return False
        return True

    async def select_callback(self, interaction: discord.Interaction):
        index = int(self.select.values[0])
        async with self.player._add_lock:
            track = self.player.remove_at(index)
        if not track:
            await interaction.response.send_message("Invalid selection.", ephemeral=True)
            return
        await interaction.response.send_message(f"Removed **{track.title}**", ephemeral=True)
        self.control.update_labels()
        if self.control.message:
            await self.control.message.edit(embed=make_queue_embed(self.player), view=self.control)
        self.stop()


def make_queue_embed(player) -> discord.Embed:
    embed = discord.Embed(title="🎶 Music Queue", color=0x1DB954)
    if player.current:
        pos = player.get_position()
        bar, pct = progress_bar(pos, player.current.duration)
        time_txt = format_timestamp(pos)
        if player.current.duration:
            time_txt += f"/{format_timestamp(player.current.duration)}"
        embed.add_field(
            name="Now Playing",
            value=f"**{player.current.title}**\n`{bar}` {pct:4.1f}% `{time_txt}`",
            inline=False,
        )

    footer_parts: list[str] = []
    if not player.queue:
        embed.description = "Queue is empty"
    else:
        q = list(player.queue)
        lines = [f"{i+1}. {_short(t.title, 80)}" for i, t in enumerate(q[:10])]
        embed.add_field(name="Up Next", value="\n".join(lines), inline=False)
        if len(q) > 10:
            footer_parts.append(f"...and {len(q)-10} more")

    footer_parts.append(f"Loop: {player.loop.capitalize()}")
    footer_parts.append(f"Auto Leave: {'On' if player.auto_leave else 'Off'}")
    footer_parts.append(f"Speed: {player.speed:.1f}×")
    footer_parts.append(f"Pitch: {player.pitch:.1f}×")
    embed.set_footer(text=" | ".join(footer_parts))
    return embed


class Queue(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="queue",
        description="Display and control the music queue",
        help=(
            "Show the upcoming songs in an embed with buttons to control playback. "
            "The panel auto-updates while the queue changes.\n\n"
            "**Usage**: `/queue`\n"
            f"`{BOT_PREFIX}queue`"
        ),
        extras={
            "category": "Music",
            "pro": (
                "List songs waiting to play with interactive controls (pause/resume, loop, skip, stop, remove). "
                "Also provides modals to adjust speed and pitch."
            ),
        },
    )
    async def queue(self, ctx: commands.Context) -> None:
        if ctx.guild is None:
            return await ctx.reply("This command can only be used in a server.", mention_author=False)
        await defer_interaction(ctx)
        player = get_player(self.bot, ctx.guild)
        if not player.voice:
            return await ctx.reply("Nothing is playing.", mention_author=False)
        view = ControlView(player)
        embed = make_queue_embed(player)
        msg = await ctx.reply(embed=embed, view=view, mention_author=False)
        await view.start(msg)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Queue(bot))
