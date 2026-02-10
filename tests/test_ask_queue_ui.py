import asyncio
import sys
from pathlib import Path
from types import MethodType

sys.path.append(str(Path(__file__).resolve().parents[1]))

from commands import ask as ask_module  # noqa: E402


def test_queue_fraction_excludes_running_task() -> None:
    position, total = ask_module._queue_fraction(
        queued_index=1,
        queued_count=2,
    )

    assert (position, total) == (1, 2)


def test_queue_fraction_clamps_negative_inputs() -> None:
    position, total = ask_module._queue_fraction(
        queued_index=-4,
        queued_count=-1,
    )

    assert (position, total) == (1, 1)


def test_queue_start_embed_uses_english_processing_copy() -> None:
    embed = ask_module.Ask._build_queue_start_embed(None)

    assert embed.title == "▶️ /ask processing started"
    assert embed.description == "Your request is now running."


def test_queue_embed_clarifies_waiting_only_fraction() -> None:
    embed = ask_module.Ask._build_queue_embed(None, position=1, total=2)

    assert embed.fields[0].name == "In queue"
    assert embed.fields[0].value == "1 / 2 (waiting only)"


def test_queue_embed_status_shows_running_count_when_present() -> None:
    embed = ask_module.Ask._build_queue_embed(None, position=1, total=2, running_count=1)

    assert embed.fields[1].name == "Status"
    assert embed.fields[1].value == "Waiting (1 running task)"


def test_processing_started_notice_handles_none_reply_without_error() -> None:
    class _DummyAsk:
        pass

    dummy = _DummyAsk()
    scheduled: list[object] = []

    async def _reply(self, ctx: object, **kwargs: object) -> None:
        return None

    def _schedule(self, message: object, *, delay: int) -> None:
        scheduled.append((message, delay))

    dummy._reply = MethodType(_reply, dummy)
    dummy._build_queue_start_embed = MethodType(ask_module.Ask._build_queue_start_embed, dummy)
    dummy._schedule_message_delete = MethodType(_schedule, dummy)

    asyncio.run(ask_module.Ask._send_processing_started_notice(dummy, object()))

    assert scheduled == []
