from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from web.poker.engine import start_hand
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

    req.json = _json  # type: ignore[assignment,method-assign]

    resp = asyncio.run(svc.handle_blackjack(req))
    assert resp.status == 400


def test_blackjack_tip_is_non_zero_for_typical_win(tmp_path: Path) -> None:
    svc = PokerService(tmp_path, _resolver)
    svc._rng.seed(1)
    req = make_mocked_request("POST", "/poker/api/blackjack/play?token=ok")

    async def _json():
        return {"bet": 1000, "tip_mode": True}

    req.json = _json  # type: ignore[assignment,method-assign]
    for _ in range(30):
        resp = asyncio.run(svc.handle_blackjack(req))
        payload = _json_payload(resp)
        if payload["result"] == "win":
            assert payload["tip_paid"] == 100
            return
    raise AssertionError("expected at least one win to validate tip payout")


def test_gto_api_rejects_ranked_table(tmp_path: Path) -> None:
    svc = PokerService(tmp_path, _resolver)
    room = svc._room("abc", ranked=True)
    svc._join_if_needed(room.table, {"user_id": "2", "nickname": "r", "avatar_url": ""})
    req = make_mocked_request("GET", "/poker/api/gto/abc?token=ranked", match_info={"room_id": "abc"})
    resp = asyncio.run(svc.handle_gto(req))
    assert resp.status in {400, 403}


def test_ranked_report_requires_finished_hand_and_prevents_duplicate(tmp_path: Path) -> None:
    svc = PokerService(tmp_path, _resolver)
    room = svc._room("ranked-room", ranked=True)
    svc._join_if_needed(room.table, {"user_id": "2", "nickname": "r", "avatar_url": ""})
    svc._join_if_needed(room.table, {"user_id": "3", "nickname": "r2", "avatar_url": ""})

    req = make_mocked_request("POST", "/poker/api/ranked/report/ranked-room?token=ranked", match_info={"room_id": "ranked-room"})

    async def _json_not_finished():
        return {"winner_user_id": "2"}

    req.json = _json_not_finished  # type: ignore[assignment,method-assign]
    not_finished_resp = asyncio.run(svc.handle_ranked_report(req))
    assert not_finished_resp.status == 400

    start_hand(room.table, svc._rng)
    room.table.winner = 0
    room.table.street = "showdown"

    async def _json_finished():
        return {"winner_user_id": room.table.players[0].user_id, "hand_no": room.table.hand_no}

    req.json = _json_finished  # type: ignore[assignment,method-assign]
    ok_resp = asyncio.run(svc.handle_ranked_report(req))
    assert ok_resp.status == 200

    duplicate_resp = asyncio.run(svc.handle_ranked_report(req))
    assert duplicate_resp.status == 409


def test_replay_hides_hole_cards_until_showdown(tmp_path: Path) -> None:
    svc = PokerService(tmp_path, _resolver)
    room = svc._room("replay-room", ranked=False)
    svc._join_if_needed(room.table, {"user_id": "1", "nickname": "n", "avatar_url": ""})
    svc._join_if_needed(room.table, {"user_id": "4", "nickname": "x", "avatar_url": ""})
    start_hand(room.table, svc._rng)

    req = make_mocked_request("GET", "/poker/api/replay/replay-room?token=ok", match_info={"room_id": "replay-room"})
    pre_resp = asyncio.run(svc.handle_replay(req))
    pre_payload = _json_payload(pre_resp)
    assert all(not p["hole"] for p in pre_payload["players"])

    room.table.street = "showdown"
    post_resp = asyncio.run(svc.handle_replay(req))
    post_payload = _json_payload(post_resp)
    assert all(len(p["hole"]) == 2 for p in post_payload["players"])


def test_blackjack_double_uses_winning_session_only_once(tmp_path: Path) -> None:
    svc = PokerService(tmp_path, _resolver)
    svc._rng.seed(2)
    play_req = make_mocked_request("POST", "/poker/api/blackjack/play?token=ok")

    async def _play_json():
        return {"bet": 1000, "tip_mode": False}

    play_req.json = _play_json  # type: ignore[assignment,method-assign]

    session_id = 0
    for _ in range(50):
        payload = _json_payload(asyncio.run(svc.handle_blackjack(play_req)))
        if payload["result"] == "win":
            session_id = int(payload["double_up_session_id"])
            break
    assert session_id > 0

    double_req = make_mocked_request("POST", "/poker/api/blackjack/double?token=ok")

    async def _double_json():
        return {"session_id": session_id}

    double_req.json = _double_json  # type: ignore[assignment,method-assign]
    first = asyncio.run(svc.handle_blackjack_double(double_req))
    assert first.status == 200
    second = asyncio.run(svc.handle_blackjack_double(double_req))
    assert second.status == 409


def test_private_state_exposes_only_viewer_hole_cards(tmp_path: Path) -> None:
    svc = PokerService(tmp_path, _resolver)
    room = svc._room("ui-room", ranked=False)
    svc._join_if_needed(room.table, {"user_id": "1", "nickname": "n", "avatar_url": ""})
    svc._join_if_needed(room.table, {"user_id": "4", "nickname": "x", "avatar_url": ""})
    start_hand(room.table, svc._rng)

    mine = svc._private_state_for_user(room.table, "1")
    theirs = svc._private_state_for_user(room.table, "4")

    assert mine.get("my_seat") in {0, 1}
    assert theirs.get("my_seat") in {0, 1}
    assert mine["my_hole"] != theirs["my_hole"]
    assert len(mine["my_hole"]) == 2
    assert len(theirs["my_hole"]) == 2


def test_bot_room_auto_seats_bot_player(tmp_path: Path) -> None:
    svc = PokerService(tmp_path, _resolver)
    room = svc._room("bot-abcd1234", ranked=False)
    svc._join_if_needed(room.table, {"user_id": "1", "nickname": "n", "avatar_url": ""})
    assert len(room.table.players) == 2
    assert any(p.user_id.startswith("bot:") for p in room.table.players)
    assert room.table.hand_no == 1
    assert room.table.street == "preflop"


def test_create_bot_room_api_returns_room_url(tmp_path: Path) -> None:
    svc = PokerService(tmp_path, _resolver)
    req = make_mocked_request("POST", "/poker/api/room/create-bot?token=ok")
    resp = asyncio.run(svc.handle_create_bot_room(req))
    payload = _json_payload(resp)

    assert resp.status == 200
    assert payload["ok"] is True
    assert str(payload["room_id"]).startswith("bot-")
    assert str(payload["url"]).startswith(f"/poker/room/{payload['room_id']}?token=")


def test_state_api_returns_private_view_with_legal_actions(tmp_path: Path) -> None:
    svc = PokerService(tmp_path, _resolver)
    room = svc._room("state-room", ranked=False)
    svc._join_if_needed(room.table, {"user_id": "1", "nickname": "n", "avatar_url": ""})
    svc._join_if_needed(room.table, {"user_id": "4", "nickname": "x", "avatar_url": ""})
    start_hand(room.table, svc._rng)

    req = make_mocked_request("GET", "/poker/api/state/state-room?token=ok", match_info={"room_id": "state-room"})
    resp = asyncio.run(svc.handle_state(req))
    payload = _json_payload(resp)

    assert payload["ok"] is True
    state = payload["state"]
    assert len(state["my_hole"]) == 2
    assert "actions" in state["legal"]
    assert isinstance(state["recent_actions"], list)


def test_idle_room_warning_shows_within_five_minutes(tmp_path: Path) -> None:
    svc = PokerService(tmp_path, _resolver)
    room = svc._room("idle-warning-room", ranked=False)
    svc._join_if_needed(room.table, {"user_id": "1", "nickname": "n", "avatar_url": ""})
    room.idle_delete_at_s = svc._now_s() + 120

    state = svc._private_state_for_user(room.table, "1")
    warning = state.get("idle_warning")
    assert isinstance(warning, dict)
    assert warning.get("active") is True
    assert warning.get("can_extend") is True


def test_idle_room_extend_api_resets_delete_timer(tmp_path: Path) -> None:
    svc = PokerService(tmp_path, _resolver)
    room = svc._room("extend-room", ranked=False)
    svc._join_if_needed(room.table, {"user_id": "1", "nickname": "n", "avatar_url": ""})
    room.idle_delete_at_s = svc._now_s() + 60
    before = room.idle_delete_at_s

    req = make_mocked_request("POST", "/poker/api/room/extend/extend-room?token=ok", match_info={"room_id": "extend-room"})
    resp = asyncio.run(svc.handle_room_extend(req))
    payload = _json_payload(resp)

    assert resp.status == 200
    assert payload["ok"] is True
    assert room.idle_delete_at_s > before


def test_idle_room_expired_is_deleted_and_not_recreated(tmp_path: Path) -> None:
    svc = PokerService(tmp_path, _resolver)
    room = svc._room("expired-room", ranked=False)
    room.idle_delete_at_s = svc._now_s() - 1

    svc._expire_idle_rooms()
    assert "expired-room" not in svc._rooms

    try:
        svc._room("expired-room", ranked=False)
    except web.HTTPGone as exc:
        assert "room_expired" in exc.text
    else:
        raise AssertionError("expected room_expired")
