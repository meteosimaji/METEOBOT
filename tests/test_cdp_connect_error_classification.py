import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from commands import ask as ask_module  # noqa: E402


def test_classify_cdp_timeout() -> None:
    assert ask_module._classify_cdp_connect_error(asyncio.TimeoutError()) == "cdp_connect_timeout"


def test_classify_cdp_auth_error() -> None:
    assert (
        ask_module._classify_cdp_connect_error(Exception("401 Unauthorized during handshake"))
        == "cdp_auth_failed"
    )


def test_classify_cdp_dns_error() -> None:
    assert ask_module._classify_cdp_connect_error(Exception("Name or service not known")) == "cdp_dns_failed"


def test_classify_cdp_refused_error() -> None:
    assert (
        ask_module._classify_cdp_connect_error(Exception("ECONNREFUSED 127.0.0.1:18792"))
        == "cdp_connection_refused"
    )


def test_classify_cdp_handshake_error() -> None:
    assert ask_module._classify_cdp_connect_error(Exception("WebSocket handshake failed")) == "cdp_handshake_failed"
