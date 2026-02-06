import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from commands import ask as ask_module  # noqa: E402


def test_cdp_policy_allows_localhost() -> None:
    assert ask_module._is_remote_cdp_url_allowed("http://127.0.0.1:9222") is True
    assert ask_module._is_remote_cdp_url_allowed("ws://localhost:18792/cdp") is True


def test_cdp_policy_blocks_remote_by_default(monkeypatch) -> None:
    monkeypatch.setattr(ask_module, "ASK_BROWSER_CDP_ALLOW_REMOTE", False)
    assert ask_module._is_remote_cdp_url_allowed("ws://10.0.0.2:9222") is False
    assert ask_module._is_remote_cdp_url_allowed("http://example.com:9222") is False


def test_cdp_policy_requires_tls_when_remote_enabled(monkeypatch) -> None:
    monkeypatch.setattr(ask_module, "ASK_BROWSER_CDP_ALLOW_REMOTE", True)
    assert ask_module._is_remote_cdp_url_allowed("ws://example.com:9222") is False
    assert ask_module._is_remote_cdp_url_allowed("http://example.com:9222") is False
    assert ask_module._is_remote_cdp_url_allowed("wss://example.com/cdp") is True
    assert ask_module._is_remote_cdp_url_allowed("https://example.com/json/version") is True
