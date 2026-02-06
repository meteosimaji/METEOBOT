import asyncio
import sys
from pathlib import Path
from typing import Any, cast
from unittest import mock

sys.path.append(str(Path(__file__).resolve().parents[1]))

from commands._browser_agent import BrowserAgent, BrowserObservation, RefEntry


def _entry(ref: str, *, role: str, name: str, nth: int | None) -> RefEntry:
    return RefEntry(
        ref=ref,
        role=role,
        name=name,
        nth=nth,
        mode="aria",
        selector=None,
        bbox=None,
        frame_name=None,
        frame_url=None,
    )


def test_flat_ref_snapshot_uses_one_based_nth_labels() -> None:
    entries = [
        _entry("e1", role="textbox", name="Email", nth=0),
        _entry("e2", role="textbox", name="Email", nth=1),
    ]

    snapshot = BrowserAgent._format_ref_snapshot(entries)

    assert '(nth=1)' in snapshot
    assert '(nth=2)' in snapshot
    assert '(nth=0)' not in snapshot


def test_tree_ref_snapshot_preserves_hierarchy_and_only_labels_operable_refs() -> None:
    aria_snapshot = """- document \"Root\"
  - dialog \"Login\"
    - textbox \"Email\" [ref=e1]
    - textbox \"Email\" [ref=e2]
"""
    entries = [
        _entry("e1", role="textbox", name="Email", nth=0),
        _entry("e2", role="textbox", name="Email", nth=1),
    ]

    snapshot = BrowserAgent._format_ref_snapshot_tree(aria_snapshot, entries)
    lines = snapshot.splitlines()

    assert lines[0] == '- document "Root"'
    assert lines[1] == '  - dialog "Login"'
    assert lines[2] == '    - textbox "Email" [ref=e1] (nth=1)'
    assert lines[3] == '    - textbox "Email" [ref=e2] (nth=2)'
    assert "[ref=" not in lines[0]
    assert "[ref=" not in lines[1]


def test_tree_ref_snapshot_handles_mixed_indent_width_and_keeps_context_role_siblings() -> None:
    aria_snapshot = """- document "Root"
    - region "Checkout"
        - heading "Payment"
        - group "Card"
		- textbox "Card number" [ref=e9]
"""
    entries = [
        _entry("e9", role="textbox", name="Card number", nth=None),
    ]

    snapshot = BrowserAgent._format_ref_snapshot_tree(aria_snapshot, entries)
    lines = snapshot.splitlines()

    assert '- document "Root"' in lines
    assert '  - region "Checkout"' in lines
    assert '    - heading "Payment"' in lines
    assert '    - group "Card"' in lines
    assert '    - textbox "Card number" [ref=e9]' in lines


def test_build_ref_entries_with_fallback_uses_clickable_when_primary_times_out() -> None:
    agent = BrowserAgent()

    fallback_entries = [
        RefEntry(
            ref="c1",
            role="button",
            name="Submit",
            nth=None,
            mode="role",
            selector="button#submit",
            bbox={"x": 1.0, "y": 2.0, "width": 10.0, "height": 20.0},
            frame_name=None,
            frame_url="https://example.com",
        )
    ]

    async def fake_safe_page_read(page, label, func, *, default, max_retries=2, timeout_s=None):
        if label == "ref_entries":
            return [], 0, "TimeoutError:", False
        if label == "ref_entries_fallback":
            return fallback_entries, 0, None, False
        return default, 0, None, False

    agent._safe_page_read = fake_safe_page_read  # type: ignore[method-assign]

    entries, ref_error, ref_error_raw, ref_degraded, retry_count, nav_race = asyncio.run(
        agent._build_ref_entries_with_fallback(page=None, aria_ref_snapshot=None)  # type: ignore[arg-type]
    )

    assert len(entries) == 1
    assert entries[0].bbox is not None
    assert ref_error == "ref_entries_degraded_fallback_clickable"
    assert ref_error_raw == "TimeoutError:"
    assert ref_degraded is True
    assert retry_count == 0
    assert nav_race is False


def test_build_ref_entries_with_fallback_passes_configured_timeouts() -> None:
    agent = BrowserAgent()
    recorded: dict[str, float | None] = {}

    async def fake_safe_page_read(page, label, func, *, default, max_retries=2, timeout_s=None):
        recorded[label] = timeout_s
        if label == "ref_entries":
            return [], 0, "TimeoutError:", False
        if label == "ref_entries_fallback":
            return [], 0, "TimeoutError:", False
        return default, 0, None, False

    agent._safe_page_read = fake_safe_page_read  # type: ignore[method-assign]

    asyncio.run(agent._build_ref_entries_with_fallback(page=None, aria_ref_snapshot=None))  # type: ignore[arg-type]

    assert recorded["ref_entries"] is not None
    assert recorded["ref_entries_fallback"] is not None


def test_observe_reuses_last_good_refs_when_current_ref_extraction_fails() -> None:
    agent = BrowserAgent()

    class _DummyPage:
        def is_closed(self) -> bool:
            return False

    page = _DummyPage()
    agent._page = page  # type: ignore[assignment]
    agent._active_tab_id = "tab1"
    agent._page_ids = {cast(Any, page): "tab1"}

    last_good = BrowserObservation(
        url="https://example.com",
        title="Example",
        aria="",
        ref_generation=5,
        ref_snapshot='- button "OK" [ref=e1]',
        refs=[{"ref": "e1", "role": "button", "name": "OK", "bbox": {"x": 1, "y": 1, "width": 10, "height": 10}}],
        ok=True,
    )
    agent._last_good_observation_by_tab["tab1"] = last_good

    async def fake_observe_page(_page):
        return BrowserObservation(
            url="https://example.com",
            title="Example",
            aria="",
            ref_generation=6,
            ref_snapshot="",
            refs=[],
            ok=True,
            title_error=None,
            aria_error=None,
            ref_error="ref_entries_timeout",
            ref_error_raw="TimeoutError:",
            ref_degraded=True,
            nav_race=False,
            last_good_used=False,
            retry_count=0,
            error=None,
            timestamp=0.0,
        )

    agent._observe_page = fake_observe_page  # type: ignore[method-assign]

    observation = asyncio.run(agent.observe())

    assert observation.last_good_used is True
    assert observation.ref_generation == 5
    assert observation.refs == last_good.refs
    assert agent._ref_generation_by_tab["tab1"] == 5
    assert agent._refs_by_tab["tab1"]["e1"].ref == "e1"


def test_round_bbox_uses_half_up_rounding_for_ties() -> None:
    rounded = BrowserAgent._round_bbox(
        {"x": 10.25, "y": 10.85, "width": 10.15, "height": 10.05}
    )

    assert rounded == (10.3, 10.9, 10.2, 10.1)


def test_round_bbox_returns_none_for_invalid_bbox_values() -> None:
    invalid_bbox: dict[str, Any] = {"x": "bad", "y": 1, "width": 1, "height": 1}
    rounded = BrowserAgent._round_bbox(cast(dict[str, float], invalid_bbox))

    assert rounded is None


def test_ref_entries_materially_changed_ignores_small_bbox_jitter() -> None:
    previous = {
        "e1": RefEntry(
            ref="e1",
            role="button",
            name="Run",
            nth=None,
            mode="aria",
            selector="#run",
            bbox={"x": 10.04, "y": 20.04, "width": 30.04, "height": 40.04},
            frame_name=None,
            frame_url="https://example.com",
        )
    }
    current = [
        RefEntry(
            ref="e1",
            role="button",
            name="Run",
            nth=None,
            mode="aria",
            selector="#run",
            bbox={"x": 10.03, "y": 20.03, "width": 30.03, "height": 40.03},
            frame_name=None,
            frame_url="https://example.com",
        )
    ]

    assert BrowserAgent._ref_entries_materially_changed(previous, current) is False


def test_ref_entries_materially_changed_detects_field_change() -> None:
    previous = {
        "e1": RefEntry(
            ref="e1",
            role="button",
            name="Run",
            nth=None,
            mode="aria",
            selector="#run",
            bbox={"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0},
            frame_name=None,
            frame_url="https://example.com",
        )
    }
    current = [
        RefEntry(
            ref="e1",
            role="button",
            name="Run now",
            nth=None,
            mode="aria",
            selector="#run",
            bbox={"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0},
            frame_name=None,
            frame_url="https://example.com",
        )
    ]

    assert BrowserAgent._ref_entries_materially_changed(previous, current) is True


def test_detect_playwright_version_falls_back_to_unknown_when_missing() -> None:
    with mock.patch("commands._browser_agent.importlib.metadata.version", side_effect=Exception("boom")):
        with mock.patch("commands._browser_agent.playwright.__version__", "", create=True):
            version, source = BrowserAgent._detect_playwright_version()

    assert version == "unknown"
    assert source == "fallback"
