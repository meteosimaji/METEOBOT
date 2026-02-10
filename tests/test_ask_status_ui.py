import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from commands import ask as ask_module  # noqa: E402


class _DummyCtx:
    interaction = None

    async def reply(self, **kwargs):  # pragma: no cover - not used in tests
        return None


def test_thinking_label_keeps_content_without_180_char_clipping() -> None:
    ui = ask_module._AskStatusUI(_DummyCtx())
    ui._thinking_text = "a" * 500

    label = ui._thinking_label(3)

    assert label.startswith("thinking (turn 3): ")
    assert label.endswith("a" * 500)
    assert "..." not in label


def test_thinking_label_caps_at_2000_chars() -> None:
    ui = ask_module._AskStatusUI(_DummyCtx())
    ui._thinking_text = "x" * 2001

    label = ui._thinking_label(1)

    assert label.startswith("thinking (turn 1): ")
    assert label.endswith("...")
    assert len(label.split(": ", 1)[1]) == 2000


def test_reasoning_delta_keeps_accumulated_text_with_2000_cap() -> None:
    ui = ask_module._AskStatusUI(_DummyCtx())

    async def _run() -> str:
        await ui.emit({"type": "turn_start", "turn": 1})
        await ui.emit({"type": "model_reasoning_delta", "turn": 1, "delta": "abc"})
        await ui.emit({"type": "model_reasoning_delta", "turn": 1, "delta": "def"})
        await ui.emit({"type": "model_reasoning_delta", "turn": 1, "delta": "z" * 2500})
        return ui._thinking_text

    out = asyncio.run(_run())

    assert out.endswith("...")
    assert len(out) == 2000


def test_refusal_delta_caps_at_2000_chars() -> None:
    ui = ask_module._AskStatusUI(_DummyCtx())

    async def _run() -> str:
        await ui.emit({"type": "turn_start", "turn": 1})
        await ui.emit({"type": "model_refusal_delta", "turn": 1, "delta": "y" * 2500})
        return ui._thinking_text

    out = asyncio.run(_run())

    assert out.startswith("refusal: ")
    assert out.endswith("...")
    assert len(out) == 2000


def test_status_ui_uses_custom_reply_func() -> None:
    calls: list[dict[str, object]] = []

    async def _custom_reply(ctx: object, **kwargs: object) -> None:
        calls.append({"ctx": ctx, **kwargs})
        return None

    ui = ask_module._AskStatusUI(_DummyCtx(), reply_func=_custom_reply)

    async def _run() -> None:
        await ui.start()

    asyncio.run(_run())

    assert calls
    assert calls[0].get("embed") is not None
