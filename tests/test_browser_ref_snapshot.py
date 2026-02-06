import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from commands._browser_agent import BrowserAgent, RefEntry


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
