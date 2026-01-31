from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

from cogs.settime import get_guild_offset
from utils import BOT_PREFIX, defer_interaction, safe_reply, sanitize, tag_error_text

log = logging.getLogger(__name__)

DATA_PATH = Path("data/say_schedules.json")
TIME_RE = re.compile(r"^(?P<hour>\d{1,2}):(?P<minute>\d{2})$")
IN_RE = re.compile(r"^in:(?P<value>\d+)(?P<unit>[smhd])$")
WEEKDAYS = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}
MAX_SCHEDULES_PER_GUILD = int(os.getenv("SAY_MAX_SCHEDULES_PER_GUILD", "50"))
MAX_SCHEDULES_PER_USER = int(os.getenv("SAY_MAX_SCHEDULES_PER_USER", "10"))
MAX_SCHEDULES_PER_CHANNEL = int(os.getenv("SAY_MAX_SCHEDULES_PER_CHANNEL", "20"))
ALLOW_EVERYONE = os.getenv("SAY_ALLOW_EVERYONE", "").lower() in {"1", "true", "yes"}
MIN_REPEAT_MINUTES = int(os.getenv("SAY_MIN_REPEAT_MINUTES", "5"))


@dataclass(frozen=True)
class ScheduleTime:
    hour: int
    minute: int
    day_shift: int = 0

    def key(self) -> str:
        return f"{self.hour:02d}:{self.minute:02d}+{self.day_shift}"


@dataclass
class SaySchedule:
    schedule_id: str
    guild_id: int
    channel_id: int
    author_id: int
    message: str
    times: list[ScheduleTime]
    days: list[int]
    created_at: str
    last_sent: dict[str, str] = field(default_factory=dict)
    run_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schedule_id": self.schedule_id,
            "guild_id": self.guild_id,
            "channel_id": self.channel_id,
            "author_id": self.author_id,
            "message": self.message,
            "times": [
                {"hour": t.hour, "minute": t.minute, "day_shift": t.day_shift} for t in self.times
            ],
            "days": self.days,
            "created_at": self.created_at,
            "last_sent": self.last_sent,
            "run_at": self.run_at,
        }

    @staticmethod
    def from_dict(data: dict[str, object]) -> SaySchedule:
        times_raw = data.get("times")
        times: list[ScheduleTime] = []
        if isinstance(times_raw, list):
            for entry in times_raw:
                if not isinstance(entry, dict):
                    continue
                hour = int(entry.get("hour", 0))
                minute = int(entry.get("minute", 0))
                day_shift = int(entry.get("day_shift", 0))
                times.append(ScheduleTime(hour=hour, minute=minute, day_shift=day_shift))
        days_raw = data.get("days")
        days = [int(day) for day in days_raw] if isinstance(days_raw, list) else []
        last_sent = data.get("last_sent")
        return SaySchedule(
            schedule_id=str(data.get("schedule_id", "")),
            guild_id=int(data.get("guild_id", 0)),
            channel_id=int(data.get("channel_id", 0)),
            author_id=int(data.get("author_id", 0)),
            message=str(data.get("message", "")),
            times=times,
            days=days,
            created_at=str(data.get("created_at", "")),
            last_sent=last_sent if isinstance(last_sent, dict) else {},
            run_at=str(data.get("run_at")) if data.get("run_at") else None,
        )


def _split_payload(raw: str) -> tuple[str, str | None]:
    parts = raw.split("--", 1)
    if len(parts) == 1:
        return raw.strip(), None
    return parts[0].strip(), parts[1].strip()


def _parse_schedule_spec(
    raw: str,
) -> tuple[list[ScheduleTime], list[int], list[str], timedelta | None, bool]:
    tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]
    times: list[ScheduleTime] = []
    days: set[int] = set()
    errors: list[str] = []
    has_everyday = False
    has_schedule_token = False
    in_delay: timedelta | None = None

    for token in tokens:
        if token in {"everyday", "daily", "every"}:
            has_everyday = True
            has_schedule_token = True
            continue
        if token in WEEKDAYS:
            days.add(WEEKDAYS[token])
            has_schedule_token = True
            continue
        match_in = IN_RE.match(token)
        if match_in:
            value = int(match_in.group("value"))
            unit = match_in.group("unit")
            if value <= 0:
                errors.append("in: value must be positive")
                continue
            if unit == "s":
                in_delay = timedelta(seconds=value)
            elif unit == "m":
                in_delay = timedelta(minutes=value)
            elif unit == "h":
                in_delay = timedelta(hours=value)
            elif unit == "d":
                in_delay = timedelta(days=value)
            has_schedule_token = True
            continue
        match = TIME_RE.match(token)
        if match:
            hour = int(match.group("hour"))
            minute = int(match.group("minute"))
            day_shift = 0
            if hour == 24 and minute == 0:
                hour = 0
                day_shift = 1
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                errors.append(f"Invalid time: {token}")
                continue
            times.append(ScheduleTime(hour=hour, minute=minute, day_shift=day_shift))
            has_schedule_token = True
            continue
        errors.append(f"Unknown token: {token}")

    if has_everyday:
        days.clear()

    return times, sorted(days), errors, in_delay, has_schedule_token


def _format_days(days: list[int]) -> str:
    if not days:
        return "everyday"
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return ", ".join(names[day] for day in days)


def _format_times(times: list[ScheduleTime]) -> str:
    rendered = []
    for t in times:
        label = f"{t.hour:02d}:{t.minute:02d}"
        if t.day_shift:
            label += " (+1d)"
        rendered.append(label)
    return ", ".join(rendered)


class Say(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._schedules: dict[str, SaySchedule] = {}
        self._load_schedules()

    async def cog_load(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run_loop())

    async def cog_unload(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    def _load_schedules(self) -> None:
        if not DATA_PATH.exists():
            return
        try:
            raw = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            log.exception("Failed to read say schedules")
            return
        if not isinstance(raw, list):
            return
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            schedule = SaySchedule.from_dict(entry)
            if schedule.schedule_id:
                self._schedules[schedule.schedule_id] = schedule

    def _save_schedules(self) -> None:
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = [schedule.to_dict() for schedule in self._schedules.values()]
        tmp_path = DATA_PATH.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(DATA_PATH)

    async def _run_loop(self) -> None:
        await self.bot.wait_until_ready()
        while not self.bot.is_closed():
            try:
                await self._tick()
            except Exception:
                log.exception("Say schedule loop failed")
            now = datetime.now(timezone.utc)
            wait = 60 - now.second - (now.microsecond / 1_000_000)
            await asyncio.sleep(max(1.0, wait))

    async def _tick(self) -> None:
        if not self._schedules:
            return
        to_remove: list[str] = []
        async with self._lock:
            for schedule in list(self._schedules.values()):
                if await self._process_schedule(schedule):
                    to_remove.append(schedule.schedule_id)
            for schedule_id in to_remove:
                self._schedules.pop(schedule_id, None)
            self._save_schedules()

    async def _process_schedule(self, schedule: SaySchedule) -> bool:
        guild = self.bot.get_guild(schedule.guild_id)
        if guild is None:
            return False
        channel = self.bot.get_channel(schedule.channel_id)
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            return False
        now_utc = datetime.now(timezone.utc)
        if schedule.run_at:
            try:
                run_at = datetime.fromisoformat(schedule.run_at)
            except ValueError:
                return True
            if now_utc < run_at:
                return False
            try:
                await channel.send(sanitize(schedule.message))
            except Exception:
                log.exception("Failed to send scheduled say message")
                return False
            return True

        ofs = get_guild_offset(self.bot, guild.id)
        tz = timezone(timedelta(hours=ofs))
        now_local = datetime.now(tz)
        for time_entry in schedule.times:
            if now_local.hour != time_entry.hour or now_local.minute != time_entry.minute:
                continue
            effective_day = (now_local.weekday() - time_entry.day_shift) % 7
            if schedule.days and effective_day not in schedule.days:
                continue
            time_key = time_entry.key()
            date_key = now_local.date().isoformat()
            if schedule.last_sent.get(time_key) == date_key:
                continue
            try:
                await channel.send(sanitize(schedule.message))
            except Exception:
                log.exception("Failed to send scheduled say message")
                return False
            schedule.last_sent[time_key] = date_key
        return False

    def _can_manage_schedules(self, ctx: commands.Context) -> bool:
        if not ctx.guild or not isinstance(ctx.author, discord.Member):
            return False
        if ALLOW_EVERYONE:
            return True
        return ctx.author.guild_permissions.manage_messages or ctx.author.guild_permissions.administrator

    def _schedule_limits_ok(self, ctx: commands.Context) -> tuple[bool, str]:
        guild_id = ctx.guild.id if ctx.guild else 0
        channel_id = ctx.channel.id if ctx.channel else 0
        user_id = ctx.author.id
        guild_count = sum(1 for s in self._schedules.values() if s.guild_id == guild_id)
        channel_count = sum(1 for s in self._schedules.values() if s.channel_id == channel_id)
        user_count = sum(1 for s in self._schedules.values() if s.author_id == user_id)
        if guild_count >= MAX_SCHEDULES_PER_GUILD:
            return False, "This server has reached the schedule limit."
        if channel_count >= MAX_SCHEDULES_PER_CHANNEL:
            return False, "This channel has reached the schedule limit."
        if user_count >= MAX_SCHEDULES_PER_USER:
            return False, "You have reached your schedule limit."
        return True, ""

    @commands.hybrid_command(  # type: ignore[arg-type]
        name="say",
        description="Send a message now or schedule recurring posts.",
        help=(
            "Send a message immediately or schedule recurring announcements with a compact spec.\n\n"
            "**Usage**: `/say <message> --<times>,<days>`\n"
            "**Examples**: `/say おはよう --7:30,everyday`\n"
            "`/say 自動募集 VC1にあつまって！--12:00,24:00,monday`\n"
            "`/say 30分後に通知 --in:30m`\n"
            f"`{BOT_PREFIX}say list`\n"
            f"`{BOT_PREFIX}say delete <id>`"
        ),
        extras={
            "category": "Utility",
            "pro": (
                "Append `--HH:MM` times and weekday tokens (mon/tue/...) to schedule recurring posts. "
                "Use `everyday` for daily repeats, or `in:30m` for a one-time delay."
            ),
        },
    )
    @app_commands.guild_only()
    async def say(self, ctx: commands.Context, *, text: str) -> None:
        await defer_interaction(ctx)
        if ctx.guild is None:
            await safe_reply(
                ctx,
                tag_error_text("Please use /say inside a server."),
                mention_author=False,
                ephemeral=True,
            )
            return
        raw_text = (text or "").strip()
        if not raw_text:
            await safe_reply(
                ctx,
                tag_error_text("Provide a message to send."),
                mention_author=False,
                ephemeral=True,
            )
            return
        lowered = raw_text.lower()
        if lowered == "list":
            await self._handle_list(ctx)
            return
        if lowered.startswith("delete "):
            await self._handle_delete(ctx, lowered.split("delete ", 1)[1].strip())
            return
        if not self._can_manage_schedules(ctx):
            await safe_reply(
                ctx,
                tag_error_text("You need Manage Messages permission to use /say."),
                mention_author=False,
                ephemeral=True,
            )
            return
        message, spec = _split_payload(raw_text)
        if spec is None:
            await ctx.send(sanitize(message))
            if ctx.interaction:
                await safe_reply(ctx, "Message sent.", ephemeral=True)
            return
        if not spec:
            await safe_reply(
                ctx,
                tag_error_text("Schedule spec is empty. Example: `--7:30,everyday`."),
                mention_author=False,
                ephemeral=True,
            )
            return
        times, days, errors, in_delay, has_schedule_token = _parse_schedule_spec(spec)
        if errors and not has_schedule_token:
            await ctx.send(sanitize(raw_text))
            if ctx.interaction:
                await safe_reply(ctx, "Message sent.", ephemeral=True)
            return
        if errors:
            await safe_reply(
                ctx,
                tag_error_text("Invalid schedule: " + "; ".join(errors)),
                mention_author=False,
                ephemeral=True,
            )
            return
        if in_delay and (times or days):
            await safe_reply(
                ctx,
                tag_error_text("`in:` cannot be combined with recurring schedules."),
                mention_author=False,
                ephemeral=True,
            )
            return
        if not in_delay and not times:
            await safe_reply(
                ctx,
                tag_error_text("Include at least one time (e.g., `--7:30,everyday`)."),
                mention_author=False,
                ephemeral=True,
            )
            return
        if times:
            if MIN_REPEAT_MINUTES > 1:
                for t in times:
                    if t.minute % MIN_REPEAT_MINUTES != 0:
                        await safe_reply(
                            ctx,
                            tag_error_text(
                                f"Minimum repeat interval is {MIN_REPEAT_MINUTES} minutes."
                            ),
                            mention_author=False,
                            ephemeral=True,
                        )
                        return
        ok, reason = self._schedule_limits_ok(ctx)
        if not ok:
            await safe_reply(ctx, tag_error_text(reason), mention_author=False, ephemeral=True)
            return
        schedule_id = uuid.uuid4().hex[:12]
        created_at = datetime.now(timezone.utc).isoformat()
        run_at = None
        if in_delay:
            run_at = (datetime.now(timezone.utc) + in_delay).isoformat()
        schedule = SaySchedule(
            schedule_id=schedule_id,
            guild_id=ctx.guild.id,
            channel_id=ctx.channel.id,
            author_id=ctx.author.id,
            message=message,
            times=times,
            days=days,
            created_at=created_at,
            run_at=run_at,
        )
        async with self._lock:
            self._schedules[schedule_id] = schedule
            self._save_schedules()
        if run_at:
            summary = (
                "Scheduled one-time announcement created.\n"
                f"Run at: `{run_at}`"
            )
        else:
            summary = (
                "Scheduled announcement created.\n"
                f"Times: `{_format_times(times)}`\n"
                f"Days: `{_format_days(days)}`"
            )
        await safe_reply(ctx, summary, mention_author=False, ephemeral=True)

    async def _handle_list(self, ctx: commands.Context) -> None:
        if not self._can_manage_schedules(ctx):
            await safe_reply(
                ctx,
                tag_error_text("You need Manage Messages permission to list schedules."),
                mention_author=False,
                ephemeral=True,
            )
            return
        guild_id = ctx.guild.id if ctx.guild else 0
        items = [s for s in self._schedules.values() if s.guild_id == guild_id]
        if not items:
            await safe_reply(ctx, "No schedules found.", mention_author=False, ephemeral=True)
            return
        lines = []
        for schedule in items[:25]:
            if schedule.run_at:
                detail = f"once at {schedule.run_at}"
            else:
                detail = f"{_format_times(schedule.times)} / {_format_days(schedule.days)}"
            msg = schedule.message
            if len(msg) > 40:
                msg = msg[:37] + "..."
            lines.append(f"`{schedule.schedule_id}` • {detail} • {msg}")
        await safe_reply(ctx, "\n".join(lines), mention_author=False, ephemeral=True)

    async def _handle_delete(self, ctx: commands.Context, schedule_id: str) -> None:
        if not schedule_id:
            await safe_reply(
                ctx,
                tag_error_text("Provide a schedule id to delete."),
                mention_author=False,
                ephemeral=True,
            )
            return
        async with self._lock:
            schedule = self._schedules.get(schedule_id)
            if not schedule:
                await safe_reply(
                    ctx,
                    tag_error_text("Schedule not found."),
                    mention_author=False,
                    ephemeral=True,
                )
                return
            if schedule.author_id != ctx.author.id and not self._can_manage_schedules(ctx):
                await safe_reply(
                    ctx,
                    tag_error_text("You can only delete your own schedules."),
                    mention_author=False,
                    ephemeral=True,
                )
                return
            self._schedules.pop(schedule_id, None)
            self._save_schedules()
        await safe_reply(ctx, "Schedule deleted.", mention_author=False, ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Say(bot))
