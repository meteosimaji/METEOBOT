from __future__ import annotations

import asyncio
import logging
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


@dataclass
class Track:
    url: str          # direct media URL for ffmpeg
    title: str
    duration: float   # seconds (0 if unknown)
    page_url: str     # webpage URL for display


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

        before_opts = FFMPEG_OPTS["before_options"]
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
                log.info("Queued track (playlist): %s (%s)", track.title, track.page_url)
                self.queue.append(track)
                self._tail_idx = len(self.queue) - 1
                self.added_tracks.append(track)
            if not self.voice or not self.voice.is_playing():
                self._spawn(self.play_next())
            return

        async with self._add_lock:
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
                if self.auto_leave and not self._playlist_loading:
                    await self.cleanup()
                else:
                    self._request_auto_leave_check()
                return

            self._discard_by_identity(self.added_tracks, track)

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
                self.history.append(self.current)
                if self.loop == "queue":
                    self.queue.append(self.current)
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
        async with self._add_lock:
            if n_back > len(self.added_tracks):
                return None
            target = self.added_tracks[-n_back]
            removed = self._remove_from_queue(target) or self._remove_from_wait_buf(target)
            if not removed:
                return None
            self._discard_by_identity(self.added_tracks, target)
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
            if "entries" in info:
                entries = info.get("entries") or []
                if not entries:
                    raise yt_dlp.utils.DownloadError("No results")
                info = entries[0]
            return Track(
                url=info["url"],
                title=info.get("title", "Unknown"),
                duration=info.get("duration", 0) or 0,
                page_url=info.get("webpage_url", query),
            )

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
