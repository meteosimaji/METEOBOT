from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from commands.ask import Ask  # noqa: E402


class _DummyChannel:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.should_fail = should_fail

    async def send(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        if self.should_fail:
            raise RuntimeError("send failed")
        return {"ok": True, "kwargs": kwargs}


class _DummyCtx:
    def __init__(self, channel: _DummyChannel) -> None:
        self.channel = channel


class _FakeAsk:
    async def _reply(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def _send_with_retry(self, send_fn, **kwargs: Any) -> object:
        return await send_fn(**kwargs)


def test_reply_required_refuses_public_fallback_for_ephemeral() -> None:
    channel = _DummyChannel()
    ctx = _DummyCtx(channel)
    fake = _FakeAsk()

    async def _run() -> None:
        with pytest.raises(RuntimeError, match="ephemeral"):
            await Ask._reply_required(fake, ctx, ephemeral=True, content="secret")

    asyncio.run(_run())

    assert channel.calls == []


def test_reply_required_uses_channel_fallback_when_non_ephemeral() -> None:
    channel = _DummyChannel()
    ctx = _DummyCtx(channel)
    fake = _FakeAsk()

    async def _run() -> object:
        return await Ask._reply_required(fake, ctx, content="hello")

    out = asyncio.run(_run())

    assert isinstance(out, dict)
    assert channel.calls and channel.calls[0].get("content") == "hello"


def test_reply_required_propagates_channel_fallback_failure() -> None:
    channel = _DummyChannel(should_fail=True)
    ctx = _DummyCtx(channel)
    fake = _FakeAsk()

    async def _run() -> None:
        with pytest.raises(RuntimeError, match="send failed"):
            await Ask._reply_required(fake, ctx, content="hello")

    asyncio.run(_run())

    assert len(channel.calls) == 1
