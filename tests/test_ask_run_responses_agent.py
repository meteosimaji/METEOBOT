from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from commands.ask import (
    _extract_response_refusal,
    _normalize_compaction_output_items,
    run_responses_agent,
)  # noqa: E402


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
            model="gpt-5-mini-2025-08-07",
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




def test_run_responses_agent_passes_prompt_cache_request_fields() -> None:
    captured: dict[str, Any] = {}

    async def _responses_create(**kwargs: Any):
        captured.update(kwargs)
        return types.SimpleNamespace(id="resp_cache", output=[])

    async def _responses_stream(**_: Any):
        return None

    async def _run() -> None:
        resp, all_outputs, error = await run_responses_agent(
            responses_create=_responses_create,
            responses_stream=_responses_stream,
            model="gpt-5-mini-2025-08-07",
            input_items=[{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            tools=[],
            prompt_cache_retention="24h",
            prompt_cache_key="ask:v2:standard",
        )
        assert error is None
        assert getattr(resp, "id", "") == "resp_cache"
        assert all_outputs == []

    asyncio.run(_run())

    assert captured["prompt_cache_retention"] == "24h"
    assert captured["prompt_cache_key"] == "ask:v2:standard"


def test_run_responses_agent_works_without_stream_handler() -> None:
    response = types.SimpleNamespace(id="resp_no_stream", output=[])

    async def _responses_create(**_: Any):
        return response

    async def _run() -> None:
        resp, all_outputs, error = await run_responses_agent(
            responses_create=_responses_create,
            responses_stream=None,
            model="gpt-5-mini-2025-08-07",
            input_items=[{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            tools=[],
        )
        assert error is None
        assert resp is response
        assert all_outputs == []

    asyncio.run(_run())


def test_normalize_compaction_output_items_strips_transient_fields() -> None:
    raw = [
        {
            "id": "msg_1",
            "status": "completed",
            "role": "user",
            "content": [{"type": "input_text", "text": "hi"}],
        },
        {"type": "compaction", "payload": {"opaque": True}},
    ]

    normalized = _normalize_compaction_output_items(raw)

    assert len(normalized) == 2
    assert "id" not in normalized[0]
    assert "status" not in normalized[0]
    assert normalized[0]["role"] == "user"


def test_extract_response_refusal_supports_message_level_refusal() -> None:
    response = types.SimpleNamespace(
        output=[
            types.SimpleNamespace(
                type="message",
                refusal="I'm sorry, I cannot assist with that request.",
                content=[],
            )
        ]
    )

    refusal = _extract_response_refusal(response)

    assert refusal == "I'm sorry, I cannot assist with that request."
