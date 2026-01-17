from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Coroutine
from urllib.parse import urlparse, parse_qs

import discord
from discord.ext import commands
import yt_dlp

log = logging.getLogger(__name__)

# yt-dlp options: keep it quiet and extract audio URL without downloading.
YTDL_OPTS: dict[str, Any] = {
    "format": "bestaudio/best",
    "quiet": True,
    "nocheckcertificate": True,
}

# ffmpeg reconnect for flaky sources; decode audio only.
FFMPEG_OPTS: dict[str, str] = {
    "before_options": "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
    "options": "-vn",
}

PROGRESS_BLOCKS = 20


def progress_bar(position: float, duration: float) -> tuple[str, float]:
    """Return a unicode progress bar and percentage (0-100)."""
    if duration <= 0:
        return "▱" * PROGRESS_BLOCKS, 0.0
    ratio = max(0.0, min(1.0, position / duration))
    filled = int(ratio * PROGRESS_BLOCKS)
    empty = PROGRESS_BLOCKS - filled
    return "▰" * filled + "▱" * empty, ratio * 100.0


def _normalize_related_url(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = str(raw).strip()
    if not raw:
        return None
    if raw.startswith(("http://", "https://")):
        return raw
    if raw.startswith("//"):
        return f"https:{raw}"
    if raw.startswith("www.youtube.com/") or raw.startswith("youtube.com/"):
        return f"https://{raw}"
    if raw.startswith("youtu.be/"):
        return f"https://{raw}"
    if raw.startswith("/watch?v="):
        return f"https://www.youtube.com{raw}"
    if raw.startswith("watch?v="):
        return f"https://www.youtube.com/{raw}"
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", raw):
        return f"https://www.youtube.com/watch?v={raw}"
    return None


def _entry_url(entry: dict[str, Any]) -> str | None:
    raw = entry.get("webpage_url") or entry.get("url") or entry.get("id")
    normalized = _normalize_related_url(raw)
    if normalized:
        return normalized
    if raw and urlparse(str(raw)).scheme:
        return str(raw)
    return None


def _is_youtube_video_url(url: str | None) -> bool:
    if not url:
        return False
    return (
        "youtube.com/watch" in url
        or "youtu.be/" in url
        or "youtube.com/shorts/" in url
        or "youtube.com/live/" in url
    )


def _build_related_from_entries(
    entries: list[dict[str, Any]], selected_url: str | None, youtube_only: bool
) -> list[dict[str, Any]]:
    related: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        url = _entry_url(entry)
        if not url or (selected_url and url == selected_url):
            continue
        if youtube_only and not _is_youtube_video_url(url):
            continue
        title = str(entry.get("title") or "").strip()
        if not title:
            continue
        duration = entry.get("duration")
        related.append(
            {
                "title": title,
                "url": url,
                "duration": int(duration) if isinstance(duration, (int, float)) else None,
                "uploader": entry.get("uploader") or entry.get("channel"),
            }
        )
        if len(related) >= 5:
            break
    return related


def _chain_atempo(tempo: float) -> list[str]:
    """Return a list of atempo filters that stays within ffmpeg's 0.5–2.0 bounds."""

    filters: list[str] = []
    t = float(tempo)
    while t > 2.0:
        filters.append("atempo=2.0")
        t /= 2.0
    while 0 < t < 0.5:
        filters.append("atempo=0.5")
        t /= 0.5
    if abs(t - 1.0) > 1e-6:
        filters.append(f"atempo={t:.6f}")
    return filters


def _format_headers_arg(headers: dict[str, str] | None) -> str | None:
    """Format http_headers for ffmpeg -headers option, filtering sensitive values."""

    if not headers:
        return None

    filtered: list[tuple[str, str]] = []
    for key, value in headers.items():
        if not value:
            continue
        lower = key.lower()
        if lower in {"cookie", "authorization"}:
            continue
        safe_value = str(value).replace("\r", " ").replace("\n", " ")
        filtered.append((key, safe_value))

    if not filtered:
        return None

    payload = "\r\n".join(f"{k}: {v}" for k, v in filtered) + "\r\n"
    safe_payload = payload.replace('"', r'\"')
    return f'-headers "{safe_payload}"'


@dataclass
class Track:
    url: str          # direct media URL for ffmpeg
    title: str
    duration: float   # seconds (0 if unknown)
    page_url: str     # webpage URL for display
    add_id: int | None = None
    related: list[dict[str, Any]] | None = None
    http_headers: dict[str, str] | None = None


class VoiceConnectionError(Exception):
    """Raised when connecting to a voice channel fails."""


class MusicPlayer:
    """Per-guild music player with a simple queue and ffmpeg playback.

    Design constraints:
    - Keep it safe: no file reads, no shell, no voicevox/TTS.
    - Keep it debuggable: explicit logging and simple state.
    - Keep it "LLM-friendly": commands should be callable with 0 or 1 arg.
    """

    def __init__(self, bot: commands.Bot, guild: discord.Guild):
        self.bot = bot
        self.guild = guild

        self.queue: deque[Track] = deque()
        self.added_tracks: deque[Track] = deque()
        self.voice: discord.VoiceClient | None = None
        self.current: Track | None = None
        self.last_removed: Track | None = None
        self.next_add_id: int = 1

        # Loop modes: none | track | queue
        self.loop: str = "none"
        self.auto_leave: bool = True

        # Playback timing
        self.offset: float = 0.0
        self.started_at: float = 0.0
        self.ignore_after: bool = False

        # Audio transforms
        self.speed: float = 1.0   # 0.5-2.0
        self.pitch: float = 1.0   # 0.5-2.0

        self.history: deque[Track] = deque(maxlen=50)

        # Playlist coordination:
        # While a playlist is loading, we want tracks appended by the loader
        # to be the "tail", and any external additions to be buffered until
        # the playlist finishes.
        self._add_lock = asyncio.Lock()
        self._play_lock = asyncio.Lock()
        self._wait_buf: deque[Track] = deque()
        self._tail_idx: int | None = None
        self._playlist_loading: bool = False

        self.last_channel_id: int | None = None
        self._connect_lock = asyncio.Lock()
        self._auto_leave_task: asyncio.Task | None = None

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> None:
        task = asyncio.create_task(coro)

        def _done(t: asyncio.Task) -> None:
            try:
                t.result()
            except Exception:  # pragma: no cover
                log.exception("background task failed")

        task.add_done_callback(_done)

    def sync_voice_client(self) -> None:
        """Adopt an existing guild voice_client if we lost reference."""

        vc = getattr(self.guild, "voice_client", None)
        if vc and vc.is_connected():
            if not self.voice or not self.voice.is_connected():
                self.voice = vc
            ch = getattr(vc, "channel", None)
            if ch:
                self.last_channel_id = ch.id

    async def _safe_voice_play(self, source: discord.AudioSource, *, after) -> None:
        """Play on the voice client, retrying once if it disconnects mid-call."""

        self.sync_voice_client()
        if not self.voice or not self.voice.is_connected():
            self.voice = None
            if not await self.ensure_connected():
                raise VoiceConnectionError("Not connected to voice (ensure_connected failed)")

        try:
            self.voice.play(source, after=after)  # type: ignore[union-attr]
            return
        except discord.ClientException as exc:
            if "Not connected to voice" not in str(exc):
                raise

        log.warning("Not connected at voice.play(); reconnecting once and retrying")
        self.voice = None
        if not await self.ensure_connected():
            raise VoiceConnectionError("Not connected to voice after retry")
        self.voice.play(source, after=after)  # type: ignore[union-attr]

    async def join(self, channel: discord.VoiceChannel) -> None:
        async with self._connect_lock:
            self.last_channel_id = channel.id

            # If we still hold a disconnected VoiceClient, drop it to avoid stale state.
            if self.voice and not self.voice.is_connected():
                try:
                    await self.voice.disconnect(force=True)
                except Exception:
                    pass
                self.voice = None

            # Already connected to the correct channel.
            if (
                self.voice
                and self.voice.channel
                and self.voice.channel.id == channel.id
                and self.voice.is_connected()
            ):
                return

            # Connected elsewhere: move.
            if self.voice and self.voice.is_connected():
                await self.voice.move_to(channel)
                log.info("Moved to voice channel %s", channel.id)
                return

            try:
                self.voice = await channel.connect(reconnect=True, timeout=15.0, self_deaf=True)
            except (asyncio.TimeoutError, discord.DiscordException) as exc:
                log.exception("Failed to connect to voice channel %s", channel.id)
                raise VoiceConnectionError(f"Failed to connect to voice channel {channel.id}") from exc

            log.info("Joined voice channel %s", channel.id)
            self._request_auto_leave_check()

    def create_source(self, track: Track, *, seek: float = 0.0) -> discord.AudioSource:
        """Create a Discord AudioSource from a Track.

        Speed & pitch are applied via ffmpeg filters.

        To apply pitch=P and speed=S independently, we:
        - shift pitch by changing sample rate: asetrate=48000*P
          (this also changes speed by factor P)
        - resample back: aresample=48000
        - correct tempo to get final speed S: atempo = S / P
        """
        if seek > 0:
            self.offset = seek
        else:
            self.offset = 0.0
        self.started_at = time.monotonic()

        before_parts = [FFMPEG_OPTS["before_options"]]
        header_arg = _format_headers_arg(track.http_headers)
        if header_arg:
            before_parts.append(header_arg)
        before_opts = " ".join(before_parts)
        if seek > 0:
            before_opts += f" -ss {seek}"

        filters: list[str] = []

        # Pitch shift (and tempo compensation)
        if self.pitch != 1.0:
            # Normalize to 48k so the pitch factor is accurate regardless of source rate
            filters.append("aresample=48000")
            filters.append(f"asetrate=48000*{self.pitch}")
            filters.append("aresample=48000")

        tempo = self.speed
        if self.pitch != 1.0:
            tempo = self.speed / self.pitch

        if tempo != 1.0:
            # ffmpeg atempo supports 0.5-2.0 per instance; chain if needed.
            filters.extend(_chain_atempo(tempo))

        options = FFMPEG_OPTS["options"]
        if filters:
            options += f' -filter:a "{",".join(filters)}"'

        source = discord.FFmpegPCMAudio(track.url, before_options=before_opts, options=options)

        # sanity check: ensure ffmpeg actually starts and produces data
        proc = getattr(source, "_process", None)
        if proc is not None and proc.poll() is not None:
            err = b""
            if proc.stderr:
                try:
                    err = proc.stderr.read()
                except Exception:
                    pass
            msg = f"FFmpeg failed to start for {track.page_url}"
            log.error("%s: %s", msg, err.decode(errors="ignore"))
            raise RuntimeError(msg)

        return discord.PCMVolumeTransformer(source)

    async def start_playlist(self) -> None:
        async with self._add_lock:
            self._playlist_loading = True
            self._tail_idx = len(self.queue) - 1 if self.queue else None
            self._wait_buf.clear()

    async def end_playlist(self) -> None:
        async with self._add_lock:
            if self._tail_idx is None:
                idx = 0
            else:
                idx = self._tail_idx + 1

            for t in self._wait_buf:
                self.queue.insert(idx, t)  # deque.insert exists in Python 3.5+
                idx += 1
            self._wait_buf.clear()
            self._playlist_loading = False
            self._tail_idx = None

        # If nothing is playing, kick off playback.
        if not self.voice or not self.voice.is_playing():
            await self.ensure_connected()
            self._spawn(self.play_next())

    async def add(self, track: Track, *, from_playlist: bool = False) -> None:
        if from_playlist:
            async with self._add_lock:
                if track.add_id is None:
                    track.add_id = self.next_add_id
                    self.next_add_id += 1
                log.info("Queued track (playlist): %s (%s)", track.title, track.page_url)
                self.queue.append(track)
                self._tail_idx = len(self.queue) - 1
                self.added_tracks.append(track)
            if not self.voice or not self.voice.is_playing():
                self._spawn(self.play_next())
            return

        async with self._add_lock:
            if track.add_id is None:
                track.add_id = self.next_add_id
                self.next_add_id += 1
            if self._playlist_loading:
                self._wait_buf.append(track)
                self.added_tracks.append(track)
                log.info("Buffered external track: %s (%s)", track.title, track.page_url)
                return

            if self._tail_idx is not None:
                idx = self._tail_idx + 1
                self.queue.insert(idx, track)
                self._tail_idx += 1
            else:
                self.queue.append(track)
            self.added_tracks.append(track)

        if not self.voice or not self.voice.is_playing():
            self._spawn(self.play_next())

    def get_position(self) -> float:
        if not self.voice or not self.current:
            return 0.0
        if self.voice.is_paused():
            return self.offset
        if self.voice.is_playing():
            return max(0.0, (time.monotonic() - self.started_at) * self.speed + self.offset)
        return self.offset

    async def play_next(self) -> None:
        async with self._play_lock:
            if self.voice and not self.voice.is_connected():
                log.warning("VoiceClient exists but is disconnected; resetting and reconnecting")
                self.voice = None
                if not await self.ensure_connected():
                    return
            if self.voice is None:
                if not await self.ensure_connected():
                    return
            self._cancel_auto_leave_task()
            if self.voice.is_playing() or self.voice.is_paused():
                return

            track: Track | None = None
            if self.loop == "track" and self.current:
                track = self.current
            else:
                try:
                    track = self.queue.popleft()
                    self.current = track
                except IndexError:
                    track = None

            if not track:
                log.info("Queue ended")
                if self.current and self.loop != "track":
                    self.current = None
                self._request_auto_leave_check()
                return

            # Note: we intentionally keep currently playing tracks in
            # ``added_tracks`` so /remove can still "undo" an immediately
            # started track (when the queue was empty and playback began
            # right away). We discard finished/skipped tracks elsewhere.

            log.info("Now playing: %s (%s)", track.title, track.page_url)
            try:
                source = self.create_source(track)
            except RuntimeError:
                log.error("Failed to create source for %s", track.page_url)
                self._spawn(self.play_next())
                return

            def _after(e: Exception | None) -> None:
                loop = self.bot.loop
                loop.call_soon_threadsafe(self._spawn, self.after_play(e))

            try:
                await self._safe_voice_play(source, after=_after)
            except VoiceConnectionError:
                log.error("Failed to start playback: voice connection unavailable")
                return

    async def after_play(self, error: Exception | None) -> None:
        if self.ignore_after:
            self.ignore_after = False
            return
        if error:
            log.exception("Music playback error: %s", error)

        # Once a track ends naturally, it is no longer a "recent addition" that
        # /remove should target. Exception: loop=track keeps replaying the same
        # current track, so keep it removable until the user changes modes.
        if self.current and self.loop != "track":
            self._discard_by_identity(self.added_tracks, self.current)

        if self.current:
            log.info("Finished playing: %s", self.current.title)
            self.history.append(self.current)

        if self.loop == "queue" and self.current:
            self.queue.append(self.current)

        self.offset = 0.0
        self.started_at = 0.0
        await self.play_next()

    async def ensure_connected(self) -> bool:
        if self.voice and self.voice.is_connected():
            return True
        if self.last_channel_id:
            ch = self.guild.get_channel(self.last_channel_id)
            if isinstance(ch, discord.VoiceChannel):
                try:
                    await self.join(ch)
                    return True
                except Exception:  # pragma: no cover
                    log.debug("ensure_connected failed", exc_info=True)
        return False

    async def pause(self) -> None:
        if self.voice and self.voice.is_playing():
            self.offset += (time.monotonic() - self.started_at) * self.speed
            self.voice.pause()
            log.info("Playback paused")

    async def resume(self) -> None:
        if self.voice and self.voice.is_paused():
            self.started_at = time.monotonic()
            self.voice.resume()
            log.info("Playback resumed")

    async def stop(self) -> None:
        self.queue.clear()
        self.added_tracks.clear()
        self._wait_buf.clear()
        self._playlist_loading = False
        self._tail_idx = None
        self.current = None
        self.offset = 0.0
        self.started_at = 0.0
        if self.voice:
            self.ignore_after = True
            self.voice.stop()
        else:
            self.ignore_after = False
        await self.cleanup(reset_ignore_after=False)
        log.info("Playback stopped, cleared queue, and cleaned up voice")

    async def skip(self) -> None:
        if self.voice and (self.voice.is_playing() or self.voice.is_paused()):
            if self.current:
                # Skipping means we should no longer treat the current track as a
                # "recent addition". Also, if loop=track, we must clear current
                # or play_next() would simply replay it.
                self._discard_by_identity(self.added_tracks, self.current)
                self.history.append(self.current)
                if self.loop == "queue":
                    self.queue.append(self.current)
                if self.loop == "track":
                    self.current = None
            self.offset = 0.0
            self.started_at = 0.0
            self.ignore_after = True
            self.voice.stop()
            await self.play_next()
            log.info("Track skipped")

    def remove_at(self, index: int) -> Track | None:
        q = self.queue
        if index < 0 or index >= len(q):
            return None
        q.rotate(-index)
        try:
            track = q.popleft()
        except IndexError:
            track = None
        q.rotate(index)

        if track and self._tail_idx is not None and index <= self._tail_idx:
            self._tail_idx -= 1
            if self._tail_idx < 0:
                self._tail_idx = None
        if track:
            self._discard_by_identity(self.added_tracks, track)
        return track

    async def seek(self, position: float) -> bool:
        if not (self.current and self.voice):
            return False
        duration = self.current.duration
        if duration > 0:
            position = max(0.0, min(position, duration))
        else:
            position = max(0.0, position)

        try:
            source = self.create_source(self.current, seek=position)
        except RuntimeError as exc:
            log.error("Seek failed for %s: %s", self.current.page_url, exc)
            return False

        self.ignore_after = True
        self.voice.stop()

        def _after_seek(e: Exception | None) -> None:
            loop = self.bot.loop
            loop.call_soon_threadsafe(self._spawn, self.after_play(e))

        try:
            await self._safe_voice_play(source, after=_after_seek)
            return True
        except Exception:
            log.exception("Seek failed during playback restart")
            return False

    async def cleanup(self, *, reset_ignore_after: bool = True) -> None:
        self._cancel_auto_leave_task()
        if self.voice:
            try:
                await self.voice.disconnect()
            except Exception:
                pass
            self.voice = None
            log.info("Disconnected from voice channel")
        self.queue.clear()
        self.added_tracks.clear()
        self.current = None
        self.offset = 0.0
        self.started_at = 0.0
        if reset_ignore_after:
            self.ignore_after = False
        self.last_channel_id = None
        # Keep the player instance so existing views continue to operate.
        self._wait_buf.clear()
        self._playlist_loading = False
        self._tail_idx = None

    def _cancel_auto_leave_task(self) -> None:
        if self._auto_leave_task:
            self._auto_leave_task.cancel()
            self._auto_leave_task = None

    @staticmethod
    def _discard_by_identity(dq: deque[Track], target: Track) -> bool:
        for i, t in enumerate(dq):
            if t is target:
                dq.rotate(-i)
                dq.popleft()
                dq.rotate(i)
                return True
        return False

    def _remove_from_queue(self, track: Track) -> bool:
        q = self.queue
        for idx, t in enumerate(q):
            if t is track:
                q.rotate(-idx)
                q.popleft()
                q.rotate(idx)
                if self._tail_idx is not None and idx <= self._tail_idx:
                    self._tail_idx -= 1
                    if self._tail_idx < 0:
                        self._tail_idx = None
                return True
        return False

    def _remove_from_wait_buf(self, track: Track) -> bool:
        if not self._wait_buf:
            return False
        for idx, t in enumerate(self._wait_buf):
            if t is track:
                self._wait_buf.rotate(-idx)
                self._wait_buf.popleft()
                self._wait_buf.rotate(idx)
                return True
        return False

    async def remove_recent_add(self, n_back: int = 1) -> Track | None:
        """Remove the Nth most recent added (pending) track, counting from 1."""
        if n_back <= 0:
            return None
        cancel_current = False
        async with self._add_lock:
            if n_back > len(self.added_tracks):
                return None
            target = self.added_tracks[-n_back]
            if target is self.current:
                cancel_current = True
                self._discard_by_identity(self.added_tracks, target)
                # Clear current so play_next() won't replay it (especially under loop=track).
                self.current = None
            else:
                removed = self._remove_from_queue(target) or self._remove_from_wait_buf(target)
                if not removed:
                    return None
                self._discard_by_identity(self.added_tracks, target)

        self.last_removed = target

        if cancel_current:
            # Cancel the current track without treating it as "skipped":
            # no history push, no loop queue re-add.
            try:
                self.sync_voice_client()
                if self.voice and (self.voice.is_playing() or self.voice.is_paused()):
                    self.offset = 0.0
                    self.started_at = 0.0
                    self.ignore_after = True
                    self.voice.stop()
            except Exception:
                pass
            await self.play_next()
            return target

        return target

    async def remove_by_add_id(self, add_id: int) -> Track | None:
        """Remove a pending track by its stable add_id."""
        if add_id <= 0:
            return None
        cancel_current = False
        async with self._add_lock:
            target = None
            for track in self.added_tracks:
                if track.add_id == add_id:
                    target = track
                    break
            if target is None:
                return None
            if target is self.current:
                cancel_current = True
                self._discard_by_identity(self.added_tracks, target)
                self.current = None
            else:
                removed = self._remove_from_queue(target) or self._remove_from_wait_buf(target)
                if not removed:
                    return None
                self._discard_by_identity(self.added_tracks, target)

        self.last_removed = target

        if cancel_current:
            try:
                self.sync_voice_client()
                if self.voice and (self.voice.is_playing() or self.voice.is_paused()):
                    self.offset = 0.0
                    self.started_at = 0.0
                    self.ignore_after = True
                    self.voice.stop()
            except Exception:
                pass
            await self.play_next()
            return target

        return target

    def request_lonely_auto_leave(self, *, delay: float = 10.0) -> None:
        """Schedule an auto-leave when no non-bot listeners remain.

        Intended for voice-state updates while music is playing; validates
        channel membership again after ``delay`` before disconnecting.
        """

        if not self.auto_leave:
            return

        self.sync_voice_client()
        if not self.voice or not self.voice.is_connected():
            return

        channel = getattr(self.voice, "channel", None)
        members = getattr(channel, "members", []) if channel else []
        if any(not getattr(m, "bot", False) for m in members):
            return

        self._cancel_auto_leave_task()

        async def _lonely_leave() -> None:
            me = asyncio.current_task()
            try:
                await asyncio.sleep(delay)
                self.sync_voice_client()
                if not self.auto_leave:
                    return
                if not self.voice or not self.voice.is_connected():
                    return
                channel = getattr(self.voice, "channel", None)
                members = getattr(channel, "members", []) if channel else []
                if any(not getattr(m, "bot", False) for m in members):
                    return
                log.info("Auto leave triggered: channel %s has no listeners", getattr(channel, "id", "?"))
                # Prevent after_play from restarting playback/reconnect churn
                self.ignore_after = True
                try:
                    if self.voice:
                        self.voice.stop()
                except Exception:
                    pass
                await self.cleanup(reset_ignore_after=False)
            finally:
                if self._auto_leave_task is me:
                    self._auto_leave_task = None

        self._auto_leave_task = asyncio.create_task(_lonely_leave())

    def _request_auto_leave_check(self, *, delay: float = 90.0) -> None:
        """Schedule an auto-leave if idle and allowed.

        Leaves after ``delay`` seconds only when:
        - auto_leave is enabled
        - connected and not playing/paused
        - queue is empty
        - no non-bot members remain in the channel (or channel missing)
        """

        self._cancel_auto_leave_task()

        if not self.auto_leave:
            return

        if not self.voice or not self.voice.is_connected():
            return

        if self.voice.is_playing() or self.voice.is_paused():
            return

        if self.queue:
            return

        async def _auto_leave_loop() -> None:
            me = asyncio.current_task()
            try:
                while True:
                    await asyncio.sleep(delay)
                    if self._playlist_loading:
                        continue
                    if not self.auto_leave:
                        return
                    if not self.voice or not self.voice.is_connected():
                        return
                    if self.voice.is_playing() or self.voice.is_paused():
                        return
                    if self.queue:
                        return
                    channel = getattr(self.voice, "channel", None)
                    members = getattr(channel, "members", []) if channel else []
                    non_bots = [m for m in members if not getattr(m, "bot", False)]
                    if non_bots:
                        continue
                    await self.cleanup()
                    return
            finally:
                if self._auto_leave_task is me:
                    self._auto_leave_task = None

        self._auto_leave_task = asyncio.create_task(_auto_leave_loop())


players: dict[int, MusicPlayer] = {}


def get_player(bot: commands.Bot, guild: discord.Guild | int) -> MusicPlayer:
    if isinstance(guild, int):
        guild_id = guild
        guild_obj = bot.get_guild(guild_id)
        if guild_obj is None:
            raise ValueError(f"Unknown guild ID: {guild_id}")
    else:
        guild_id = guild.id
        guild_obj = guild

    if guild_id not in players:
        players[guild_id] = MusicPlayer(bot, guild_obj)
    player = players[guild_id]
    player.sync_voice_client()
    return player


async def yt_search(query: str) -> Track:
    """Fetch track information using yt-dlp in a thread."""

    def extract() -> Track:
        with yt_dlp.YoutubeDL(YTDL_OPTS) as ydl:
            info = ydl.extract_info(query, download=False)
            search_related: list[dict[str, Any]] = []
            if "entries" in info:
                entries = info.get("entries") or []
                if not entries:
                    raise yt_dlp.utils.DownloadError("No results")
                info = entries[0]
                entry_url = _entry_url(info)
                youtube_only = any(
                    _is_youtube_video_url(_entry_url(candidate))
                    for candidate in entries
                    if isinstance(candidate, dict)
                )
                if youtube_only:
                    for candidate in entries:
                        candidate_url = _entry_url(candidate)
                        if _is_youtube_video_url(candidate_url):
                            info = candidate
                            entry_url = candidate_url
                            break
                search_related = _build_related_from_entries(entries, entry_url, youtube_only)
                if _is_youtube_video_url(entry_url) and not info.get("related_videos"):
                    refreshed = ydl.extract_info(entry_url, download=False)
                    if "entries" not in refreshed:
                        info = refreshed
            headers = info.get("http_headers") or None
            related: list[dict[str, Any]] = []
            if search_related:
                related = search_related
            else:
                for entry in (info.get("related_videos") or []):
                    if not isinstance(entry, dict):
                        continue
                    title = str(entry.get("title") or "").strip()
                    url = _normalize_related_url(entry.get("webpage_url") or entry.get("url") or entry.get("id"))
                    if not title or not url:
                        continue
                    duration = entry.get("duration")
                    related.append(
                        {
                            "title": title,
                            "url": url,
                            "duration": int(duration) if isinstance(duration, (int, float)) else None,
                            "uploader": entry.get("uploader") or entry.get("channel"),
                        }
                    )
                    if len(related) >= 5:
                        break
            return Track(
                url=info["url"],
                title=info.get("title", "Unknown"),
                duration=info.get("duration", 0) or 0,
                page_url=info.get("webpage_url", query),
                related=related or None,
                http_headers=headers,
            )

    return await asyncio.to_thread(extract)


async def yt_search_results(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return search results without queuing."""

    def extract() -> list[dict[str, Any]]:
        with yt_dlp.YoutubeDL(YTDL_OPTS) as ydl:
            info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
        entries = info.get("entries") or []
        results: list[dict[str, Any]] = []
        youtube_only = any(
            _is_youtube_video_url(_entry_url(entry)) for entry in entries if isinstance(entry, dict)
        )
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            url = _entry_url(entry)
            if not url:
                continue
            if youtube_only and not _is_youtube_video_url(url):
                continue
            title = str(entry.get("title") or "").strip()
            if not title:
                continue
            duration = entry.get("duration")
            results.append(
                {
                    "title": title,
                    "url": url,
                    "duration": int(duration) if isinstance(duration, (int, float)) else None,
                    "uploader": entry.get("uploader") or entry.get("channel"),
                }
            )
            if len(results) >= limit:
                break
        if not results:
            raise yt_dlp.utils.DownloadError("No results")
        return results

    return await asyncio.to_thread(extract)


async def iter_playlist(url: str) -> AsyncIterator[Track]:
    """Yield tracks from a YouTube playlist one by one (flat extraction first)."""

    def extract_urls() -> list[str]:
        opts = {**YTDL_OPTS, "extract_flat": "in_playlist"}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            entries = info.get("entries", []) or []
            out: list[str] = []
            for e in entries:
                u = (e or {}).get("url")
                if u:
                    out.append(str(u))
            return out

    entry_urls = await asyncio.to_thread(extract_urls)
    for entry_url in entry_urls:
        # yt-dlp sometimes returns just an ID when extract_flat is enabled.
        if not urlparse(entry_url).scheme:
            entry_url = f"https://www.youtube.com/watch?v={entry_url}"
        try:
            track = await yt_search(entry_url)
        except Exception:  # pragma: no cover
            log.debug("failed to extract playlist entry %s", entry_url, exc_info=True)
            continue
        yield track


async def _multi_search(query: str) -> Track:
    """Try multiple services for a plain text query."""
    services = ["ytsearch", "scsearch"]
    last_error: Exception | None = None
    for service in services:
        try:
            return await yt_search(f"{service}:{query}")
        except Exception as exc:  # pragma: no cover
            last_error = exc
            log.debug("search failed on %s", service, exc_info=True)
    if last_error:
        raise last_error
    raise yt_dlp.utils.DownloadError(f"No results found for {query!r}")


async def resolve_track(query: str) -> Track | AsyncIterator[Track]:
    """Resolve a track from a URL or search query.

    URLs are handled directly. Spotify links are converted into a YouTube
    search using extracted metadata. Playlist URLs return an async iterator.
    Plain text queries are searched across multiple services.
    """
    parsed = urlparse(query)
    if parsed.scheme and parsed.netloc:
        if "spotify" in parsed.netloc:
            def extract_spotify() -> str:
                with yt_dlp.YoutubeDL(YTDL_OPTS) as ydl:
                    info = ydl.extract_info(query, download=False)
                title = info.get("track") or info.get("title") or ""
                artist = info.get("artist") or info.get("uploader") or ""
                return f"{title} {artist}".strip()

            search_q = await asyncio.to_thread(extract_spotify)
            return await _multi_search(search_q)

        if "nicovideo" in parsed.netloc:
            return await yt_search(query)

        if "youtube" in parsed.netloc or "youtu.be" in parsed.netloc:
            qs = parse_qs(parsed.query)
            if parsed.path == "/playlist" or ("list" in qs and "v" not in qs):
                return iter_playlist(query)

        return await yt_search(query)

    return await _multi_search(query)
