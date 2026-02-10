from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from commands.ask import run_responses_agent  # noqa: E402


class _DummyStream:
    def __init__(self, events: list[Any]) -> None:
        self._events = events

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


def test_run_responses_agent_emits_reasoning_delta_from_stream_event() -> None:
    events: list[dict[str, Any]] = []

    async def _event_cb(evt: dict[str, Any]) -> None:
        events.append(evt)

    response = types.SimpleNamespace(id="resp_1", output=[])

    async def _responses_stream(**_: Any) -> _DummyStream:
        return _DummyStream(stream_events.copy())

    stream_events = [
        types.SimpleNamespace(type="response.reasoning_summary_text.delta", delta="step-1"),
        types.SimpleNamespace(type="response.completed", response=response),
    ]

    async def _run() -> None:
        resp, all_outputs, error = await run_responses_agent(
            responses_create=lambda **_: response,
            responses_stream=_responses_stream,
            model="gpt-5.2-mini",
            input_items=[{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            tools=[],
            event_cb=_event_cb,
        )
        assert error is None
        assert resp is response
        assert all_outputs == []

    asyncio.run(_run())

    assert any(
        evt.get("type") == "model_reasoning_delta" and evt.get("delta") == "step-1"
        for evt in events
    )
