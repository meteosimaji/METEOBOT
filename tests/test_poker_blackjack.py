from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp.test_utils import make_mocked_request

from web.poker.service import PokerService


def _resolver(token: str):
    if token == "ok":
        return {"user_id": "1", "nickname": "n", "avatar_url": "", "ranked": False}
    if token == "ranked":
        return {"user_id": "2", "nickname": "r", "avatar_url": "", "ranked": True}
    if token == "ranked2":
        return {"user_id": "3", "nickname": "r2", "avatar_url": "", "ranked": True}
    return None


def _json_payload(resp) -> dict:
    return json.loads(resp.text)


def test_blackjack_invalid_bet_rejected(tmp_path: Path) -> None:
    svc = PokerService(tmp_path, _resolver)
    req = make_mocked_request("POST", "/poker/api/blackjack/play?token=ok")

    async def _json():
        return {"bet": 2500, "tip_mode": False}

    req.json = _json  # type: ignore[method-assign]

    resp = asyncio.run(svc.handle_blackjack(req))
    assert resp.status == 400


def test_gto_api_rejects_ranked_table(tmp_path: Path) -> None:
    svc = PokerService(tmp_path, _resolver)
    room = svc._room("abc", ranked=True)
    svc._join_if_needed(room.table, {"user_id": "2", "nickname": "r", "avatar_url": ""})
    req = make_mocked_request("GET", "/poker/api/gto/abc?token=ranked", match_info={"room_id": "abc"})
    resp = asyncio.run(svc.handle_gto(req))
    assert resp.status in {400, 403}


def test_ranked_report_updates_mmr_and_non_decreasing_rp(tmp_path: Path) -> None:
    svc = PokerService(tmp_path, _resolver)
    room = svc._room("ranked-room", ranked=True)
    svc._join_if_needed(room.table, {"user_id": "2", "nickname": "r", "avatar_url": ""})
    svc._join_if_needed(room.table, {"user_id": "3", "nickname": "r2", "avatar_url": ""})

    req = make_mocked_request("POST", "/poker/api/ranked/report/ranked-room?token=ranked", match_info={"room_id": "ranked-room"})

    async def _json():
        return {"winner_user_id": "2"}

    req.json = _json  # type: ignore[method-assign]
    resp = asyncio.run(svc.handle_ranked_report(req))
    assert resp.status == 200
    payload = _json_payload(resp)
    assert payload["winner"]["mmr"] > 1500
    assert payload["loser"]["mmr"] < 1500
    assert payload["winner"]["rp"] >= 0
    assert payload["loser"]["rp"] >= 0
