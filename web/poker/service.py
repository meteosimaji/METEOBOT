from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sqlite3
from typing import Any, Callable

from aiohttp import WSMsgType, web

from .engine import (
    GameRuleError,
    PlayerState,
    TableConfig,
    TableState,
    apply_action,
    gto_recommendations,
    legal_actions,
    replay_scores,
    start_hand,
)

IdentityResolver = Callable[[str], dict[str, Any] | None]
VIRTUAL_BOT_POPULATION = 10000
MATCHMAKING_BOT_WAIT_S = 15
ROOM_IDLE_DELETE_S = 30 * 60
ROOM_IDLE_WARNING_S = 5 * 60


@dataclass
class Room:
    table: TableState
    sockets: set[web.WebSocketResponse]
    lock: asyncio.Lock
    last_client_action_id_by_user: dict[str, int]
    socket_user_ids: dict[web.WebSocketResponse, str]
    idle_delete_at_s: float


class PokerService:
    def __init__(self, data_root: Path, identity_resolver: IdentityResolver) -> None:
        self._rooms: dict[str, Room] = {}
        self._expired_room_ids: set[str] = set()
        self._rng = random.Random()
        self._identity_resolver = identity_resolver
        self._db_path = data_root / "poker.sqlite3"
        data_root.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS coin_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    delta INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS blackjack_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    bet INTEGER NOT NULL,
                    result TEXT NOT NULL,
                    tip_mode INTEGER NOT NULL,
                    tip_paid INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS poker_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    room_id TEXT NOT NULL,
                    hand_no INTEGER NOT NULL,
                    street TEXT NOT NULL,
                    seat INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    off_tree INTEGER NOT NULL,
                    gto_freq REAL NOT NULL,
                    gto_evloss_bb REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS rating_state (
                    user_id TEXT PRIMARY KEY,
                    mmr REAL NOT NULL,
                    rp INTEGER NOT NULL,
                    wcp INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS matchmaking_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    queued_at TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS ranked_hand_reports (
                    room_id TEXT NOT NULL,
                    hand_no INTEGER NOT NULL,
                    winner_user_id TEXT NOT NULL,
                    loser_user_id TEXT NOT NULL,
                    reported_by_user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (room_id, hand_no)
                )
                """
            )
            try:
                con.execute("ALTER TABLE blackjack_sessions ADD COLUMN double_up_used INTEGER NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass

    def _room(self, room_id: str, *, ranked: bool) -> Room:
        if room_id in self._expired_room_ids:
            raise web.HTTPGone(text="room_expired")
        self._expire_idle_rooms()
        room = self._rooms.get(room_id)
        if room:
            if ranked and not room.table.ranked:
                room.table.ranked = True
            return room
        table = TableState(room_id=room_id, config=TableConfig(), ranked=ranked)
        room = Room(
            table=table,
            sockets=set(),
            lock=asyncio.Lock(),
            last_client_action_id_by_user={},
            socket_user_ids={},
            idle_delete_at_s=self._now_s() + ROOM_IDLE_DELETE_S,
        )
        self._rooms[room_id] = room
        return room

    @staticmethod
    def _now_s() -> float:
        return datetime.now(timezone.utc).timestamp()

    def _has_started_any_hand(self, room: Room) -> bool:
        return room.table.hand_no > 0

    def _idle_warning_payload(self, room: Room) -> dict[str, Any] | None:
        if self._has_started_any_hand(room):
            return None
        remaining_s = int(max(0.0, room.idle_delete_at_s - self._now_s()))
        if remaining_s > ROOM_IDLE_WARNING_S:
            return None
        return {
            "active": True,
            "remaining_seconds": remaining_s,
            "message": "This room will be deleted soon due to inactivity. Press Extend to keep it for another 30 minutes.",
            "can_extend": True,
            "extend_seconds": ROOM_IDLE_DELETE_S,
        }

    def _expire_idle_rooms(self) -> None:
        now_s = self._now_s()
        expired_ids = [
            room_id
            for room_id, room in self._rooms.items()
            if not self._has_started_any_hand(room) and room.idle_delete_at_s <= now_s
        ]
        for room_id in expired_ids:
            room = self._rooms.pop(room_id, None)
            if room is None:
                continue
            self._expired_room_ids.add(room_id)
            for ws in list(room.sockets):
                if ws.closed:
                    continue
                asyncio.create_task(ws.send_json({"type": "error", "error": "room_expired"}))
                asyncio.create_task(ws.close(code=1001, message=b"room_expired"))

    async def handle_lobby(self, request: web.Request) -> web.Response:
        self._expire_idle_rooms()
        return web.Response(text=_LOBBY_HTML, content_type="text/html")

    async def handle_room(self, request: web.Request) -> web.Response:
        self._expire_idle_rooms()
        token = request.query.get("token", "")
        profile = self._identity_resolver(token)
        if not profile:
            raise web.HTTPUnauthorized(text="invalid_token")
        room_id = request.match_info.get("room_id", "")
        ranked = bool(profile.get("ranked", False))
        room = self._room(room_id, ranked=ranked)
        room.table.ranked = ranked
        return web.Response(text=_room_html(room_id, token, ranked), content_type="text/html")

    async def handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        self._expire_idle_rooms()
        room_id = request.match_info.get("room_id", "")
        token = request.query.get("token", "")
        profile = self._identity_resolver(token)
        if not profile:
            raise web.HTTPUnauthorized(text="invalid_token")

        ranked = bool(profile.get("ranked", False))
        room = self._room(room_id, ranked=ranked)
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        room.sockets.add(ws)
        room.socket_user_ids[ws] = str(profile["user_id"])

        async with room.lock:
            self._join_if_needed(room.table, profile)
            await self._broadcast(room, ack_action_id=None)

        user_id = str(profile["user_id"])
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                payload = json.loads(msg.data)
                action = str(payload.get("action") or "")
                amount = int(payload.get("amount") or 0)
                client_action_id = int(payload.get("client_action_id") or 0)

                async with room.lock:
                    last_seen = room.last_client_action_id_by_user.get(user_id, 0)
                    if client_action_id and client_action_id <= last_seen:
                        await ws.send_json(
                            {
                                "type": "ack",
                                "ack_action_id": last_seen,
                                "server_seq": room.table.server_seq,
                                "duplicate": True,
                            }
                        )
                        continue
                    seat = self._seat_for_user(room.table, user_id)
                    if seat is None:
                        continue
                    if action == "start":
                        if len(room.table.players) == 2 and room.table.street in {"waiting", "showdown"}:
                            start_hand(room.table, self._rng)
                    elif action == "next_hand":
                        if room.table.street == "showdown":
                            start_hand(room.table, self._rng)
                    elif action == "gto_reveal":
                        if room.table.ranked:
                            await ws.send_json({"type": "error", "error": "gto_disabled_in_ranked"})
                        else:
                            room.table.gto_viewer_user_id = user_id
                    elif action in {"fold", "check", "call", "bet", "raise", "raise_to", "allin"}:
                        try:
                            rec = apply_action(room.table, seat, action, amount)
                        except GameRuleError as exc:
                            await ws.send_json({"type": "error", "error": str(exc)})
                        else:
                            self._persist_action(room.table.room_id, rec)
                    self._run_bot_until_human_turn(room)
                    if client_action_id:
                        room.last_client_action_id_by_user[user_id] = client_action_id
                    await self._broadcast(room, ack_action_id=client_action_id or None)
        finally:
            room.sockets.discard(ws)
            room.socket_user_ids.pop(ws, None)
        return ws

    async def handle_gto(self, request: web.Request) -> web.Response:
        self._expire_idle_rooms()
        token = request.query.get("token", "")
        profile = self._identity_resolver(token)
        if not profile:
            raise web.HTTPUnauthorized(text="invalid_token")
        room_id = request.match_info.get("room_id", "")
        ranked = bool(profile.get("ranked", False))
        room = self._room(room_id, ranked=ranked)
        user_id = str(profile["user_id"])
        seat = self._seat_for_user(room.table, user_id)
        if seat is None:
            return web.json_response({"ok": False, "error": "not_seated"}, status=403)
        try:
            payload = gto_recommendations(room.table, seat)
        except GameRuleError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        room.table.gto_viewer_user_id = user_id
        room.table.server_seq += 1
        await self._broadcast(room, ack_action_id=None)
        return web.json_response(
            {
                "ok": True,
                "ranked": room.table.ranked,
                "gto": payload,
                "gto_method": "heuristic_blueprint_approximation",
                "is_equilibrium_proven": False,
            }
        )

    async def handle_replay(self, request: web.Request) -> web.Response:
        self._expire_idle_rooms()
        token = request.query.get("token", "")
        profile = self._identity_resolver(token)
        if not profile:
            raise web.HTTPUnauthorized(text="invalid_token")
        room_id = request.match_info.get("room_id", "")
        ranked = bool(profile.get("ranked", False))
        room = self._room(room_id, ranked=ranked)
        replay = replay_scores(room.table)
        reveal_hole_cards = room.table.street == "showdown"
        players = [
            {
                "user_id": p.user_id,
                "nickname": p.nickname,
                "hole": [str(card) for card in p.hole] if reveal_hole_cards else [],
            }
            for p in room.table.players
        ]
        return web.json_response(
            {
                "ok": True,
                "room_id": room_id,
                "hand_no": room.table.hand_no,
                "seed_commit": room.table.seed_commit,
                "seed_reveal": room.table.seed_reveal if room.table.street == "showdown" else "",
                "players": players,
                "replay": replay,
            }
        )

    async def handle_state(self, request: web.Request) -> web.Response:
        self._expire_idle_rooms()
        token = request.query.get("token", "")
        profile = self._identity_resolver(token)
        if not profile:
            raise web.HTTPUnauthorized(text="invalid_token")
        room_id = request.match_info.get("room_id", "")
        ranked = bool(profile.get("ranked", False))
        room = self._room(room_id, ranked=ranked)
        user_id = str(profile["user_id"])
        return web.json_response({"ok": True, "state": self._private_state_for_user(room.table, user_id)})

    async def handle_ranked_report(self, request: web.Request) -> web.Response:
        self._expire_idle_rooms()
        token = request.query.get("token", "")
        profile = self._identity_resolver(token)
        if not profile:
            raise web.HTTPUnauthorized(text="invalid_token")
        if not bool(profile.get("ranked", False)):
            return web.json_response({"ok": False, "error": "ranked_only"}, status=400)

        room_id = request.match_info.get("room_id", "")
        room = self._rooms.get(room_id)
        if room is None:
            return web.json_response({"ok": False, "error": "room_not_found"}, status=404)

        reporter_user_id = str(profile["user_id"])
        player_ids = [p.user_id for p in room.table.players]
        if reporter_user_id not in player_ids:
            return web.json_response({"ok": False, "error": "reporter_not_in_room"}, status=403)
        if len(player_ids) != 2:
            return web.json_response({"ok": False, "error": "need_two_players"}, status=400)
        if room.table.street != "showdown":
            return web.json_response({"ok": False, "error": "hand_not_finished"}, status=400)
        if room.table.winner is None:
            return web.json_response({"ok": False, "error": "tie_hand_no_winner"}, status=400)

        body = await request.json()
        expected_hand_no = int(body.get("hand_no") or room.table.hand_no)
        if expected_hand_no != room.table.hand_no:
            return web.json_response({"ok": False, "error": "hand_no_mismatch"}, status=400)

        winner_user_id = room.table.players[room.table.winner].user_id
        requested_winner = str(body.get("winner_user_id") or winner_user_id)
        if requested_winner != winner_user_id:
            return web.json_response({"ok": False, "error": "winner_mismatch"}, status=400)
        loser_user_id = next(uid for uid in player_ids if uid != winner_user_id)

        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as con:
            try:
                con.execute(
                    "INSERT INTO ranked_hand_reports (room_id, hand_no, winner_user_id, loser_user_id, reported_by_user_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (room_id, room.table.hand_no, winner_user_id, loser_user_id, reporter_user_id, now),
                )
            except sqlite3.IntegrityError:
                return web.json_response({"ok": False, "error": "already_reported"}, status=409)

        winner_state = self._load_rating_state(winner_user_id)
        loser_state = self._load_rating_state(loser_user_id)

        expected_w = 1.0 / (1.0 + 10 ** ((loser_state["mmr"] - winner_state["mmr"]) / 400.0))
        expected_l = 1.0 - expected_w
        k = 24.0
        winner_state["mmr"] += k * (1.0 - expected_w)
        loser_state["mmr"] += k * (0.0 - expected_l)
        winner_state["rp"] += 10
        loser_state["rp"] += 3

        all_human_ratings = self._all_mmr_values(exclude_bots=True)
        winner_state["wcp"] = self._compute_wcp(winner_state["mmr"], all_human_ratings)
        loser_state["wcp"] = self._compute_wcp(loser_state["mmr"], all_human_ratings)

        self._save_rating_state(winner_user_id, winner_state)
        self._save_rating_state(loser_user_id, loser_state)
        return web.json_response(
            {
                "ok": True,
                "winner": {"user_id": winner_user_id, **winner_state},
                "loser": {"user_id": loser_user_id, **loser_state},
                "hand_no": room.table.hand_no,
            }
        )

    async def handle_ranked_leaderboard(self, request: web.Request) -> web.Response:
        self._expire_idle_rooms()
        with sqlite3.connect(self._db_path) as con:
            rows = con.execute(
                "SELECT user_id, mmr, rp, wcp FROM rating_state ORDER BY mmr DESC LIMIT 100"
            ).fetchall()
        return web.json_response(
            {
                "ok": True,
                "leaderboard": [
                    {"user_id": r[0], "mmr": float(r[1]), "rp": int(r[2]), "wcp": int(r[3])}
                    for r in rows
                ],
            }
        )

    async def handle_matchmaking_enqueue(self, request: web.Request) -> web.Response:
        self._expire_idle_rooms()
        token = request.query.get("token", "")
        profile = self._identity_resolver(token)
        if not profile:
            raise web.HTTPUnauthorized(text="invalid_token")
        body = await request.json()
        mode = str(body.get("mode") or "quick").lower()
        if mode not in {"quick", "ranked"}:
            return web.json_response({"ok": False, "error": "bad_mode"}, status=400)
        user_id = str(profile["user_id"])
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as con:
            con.execute(
                "UPDATE matchmaking_queue SET active=0 WHERE user_id=? AND active=1",
                (user_id,),
            )
            con.execute(
                "INSERT INTO matchmaking_queue (user_id, mode, queued_at, active) VALUES (?, ?, ?, 1)",
                (user_id, mode, now),
            )
            row = con.execute(
                "SELECT id, user_id FROM matchmaking_queue WHERE mode=? AND active=1 AND user_id != ? ORDER BY id ASC LIMIT 1",
                (mode, user_id),
            ).fetchone()
            if row is not None:
                con.execute("UPDATE matchmaking_queue SET active=0 WHERE id IN (?, ?)", (int(row[0]), con.execute("SELECT last_insert_rowid()").fetchone()[0]))
                room_id = f"mm-{secrets_token(self._rng)}"
                return web.json_response(
                    {
                        "ok": True,
                        "matched": True,
                        "room_id": room_id,
                        "opponent_user_id": str(row[1]),
                        "mode": mode,
                    }
                )
        return web.json_response({"ok": True, "matched": False, "mode": mode})

    async def handle_matchmaking_poll(self, request: web.Request) -> web.Response:
        self._expire_idle_rooms()
        token = request.query.get("token", "")
        profile = self._identity_resolver(token)
        if not profile:
            raise web.HTTPUnauthorized(text="invalid_token")
        user_id = str(profile["user_id"])
        mode = str((request.query.get("mode") or "quick")).lower()
        if mode not in {"quick", "ranked"}:
            return web.json_response({"ok": False, "error": "bad_mode"}, status=400)

        now_dt = datetime.now(timezone.utc)
        with sqlite3.connect(self._db_path) as con:
            row = con.execute(
                "SELECT id, queued_at FROM matchmaking_queue WHERE user_id=? AND mode=? AND active=1 ORDER BY id DESC LIMIT 1",
                (user_id, mode),
            ).fetchone()
            if row is None:
                return web.json_response({"ok": True, "queued": False, "matched": False})
            queued_at = datetime.fromisoformat(str(row[1]))
            waited = max(0.0, (now_dt - queued_at).total_seconds())
            if waited >= MATCHMAKING_BOT_WAIT_S:
                con.execute("UPDATE matchmaking_queue SET active=0 WHERE id=?", (int(row[0]),))
                room_id = f"bot-{secrets_token(self._rng)}"
                return web.json_response(
                    {
                        "ok": True,
                        "queued": False,
                        "matched": True,
                        "bot": True,
                        "room_id": room_id,
                        "mode": mode,
                        "waited_s": waited,
                    }
                )
        return web.json_response({"ok": True, "queued": True, "matched": False})

    async def handle_blackjack(self, request: web.Request) -> web.Response:
        self._expire_idle_rooms()
        token = request.query.get("token", "")
        profile = self._identity_resolver(token)
        if not profile:
            raise web.HTTPUnauthorized(text="invalid_token")
        user_id = str(profile["user_id"])
        body = await request.json()
        bet = int(body.get("bet") or 0)
        tip_mode = bool(body.get("tip_mode", False))
        if bet <= 0 or bet > 2000 or bet % 100 != 0:
            return web.json_response({"ok": False, "error": "invalid_bet"}, status=400)

        deck = [value for value in range(1, 14)] * 4
        self._rng.shuffle(deck)
        player = [deck.pop(), deck.pop()]
        dealer = [deck.pop(), deck.pop()]
        while _bj_value(player) < 17:
            player.append(deck.pop())
        while _bj_value(dealer) < 17:
            dealer.append(deck.pop())

        pv = _bj_value(player)
        dv = _bj_value(dealer)
        if pv > 21:
            result = "lose"
            delta = -bet
        elif dv > 21 or pv > dv:
            result = "win"
            delta = bet
        elif pv == dv:
            result = "push"
            delta = 0
        else:
            result = "lose"
            delta = -bet

        tip_paid = 0
        if result == "win" and tip_mode:
            tip_raw = (bet * 5 + 99) // 100
            tip_paid = ((tip_raw + 99) // 100) * 100
            delta -= tip_paid

        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as con:
            cur = con.execute(
                "INSERT INTO blackjack_sessions (user_id, bet, result, tip_mode, tip_paid, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, bet, result, int(tip_mode), tip_paid, now),
            )
            session_row_id = cur.lastrowid
            session_id = int(session_row_id) if session_row_id is not None else 0
            con.execute(
                "INSERT INTO coin_ledger (user_id, delta, reason, created_at) VALUES (?, ?, ?, ?)",
                (user_id, delta, f"blackjack:{result}", now),
            )

        return web.json_response(
            {
                "ok": True,
                "result": result,
                "delta": delta,
                "tip_paid": tip_paid,
                "player": player,
                "dealer": dealer,
                "can_double_up": result == "win",
                "double_up_session_id": session_id if result == "win" else None,
            }
        )

    async def handle_blackjack_double(self, request: web.Request) -> web.Response:
        self._expire_idle_rooms()
        token = request.query.get("token", "")
        profile = self._identity_resolver(token)
        if not profile:
            raise web.HTTPUnauthorized(text="invalid_token")
        body = await request.json()
        session_id = int(body.get("session_id") or 0)
        user_id = str(profile["user_id"])
        if session_id <= 0:
            return web.json_response({"ok": False, "error": "invalid_session_id"}, status=400)

        with sqlite3.connect(self._db_path) as con:
            row = con.execute(
                "SELECT bet, tip_paid, result, COALESCE(double_up_used, 0) FROM blackjack_sessions WHERE id=? AND user_id=?",
                (session_id, user_id),
            ).fetchone()
            if row is None:
                return web.json_response({"ok": False, "error": "session_not_found"}, status=404)
            bet = int(row[0])
            tip_paid = int(row[1])
            result = str(row[2])
            double_up_used = int(row[3])
            if result != "win":
                return web.json_response({"ok": False, "error": "session_not_winning"}, status=400)
            if double_up_used:
                return web.json_response({"ok": False, "error": "double_up_already_used"}, status=409)

            amount = bet - tip_paid
            win = bool(self._rng.randint(0, 1))
            delta = amount if win else -amount
            now = datetime.now(timezone.utc).isoformat()
            con.execute("UPDATE blackjack_sessions SET double_up_used=1 WHERE id=?", (session_id,))
            con.execute(
                "INSERT INTO coin_ledger (user_id, delta, reason, created_at) VALUES (?, ?, ?, ?)",
                (user_id, delta, f"blackjack:double_up:{session_id}", now),
            )

        return web.json_response({"ok": True, "win": win, "delta": delta, "amount": amount, "session_id": session_id})

    async def handle_rankings(self, request: web.Request) -> web.Response:
        self._expire_idle_rooms()
        with sqlite3.connect(self._db_path) as con:
            top_coin = con.execute(
                "SELECT user_id, COALESCE(SUM(delta),0) AS total FROM coin_ledger GROUP BY user_id ORDER BY total DESC LIMIT 20"
            ).fetchall()
            top_tip = con.execute(
                "SELECT user_id, COALESCE(SUM(tip_paid),0) AS total_tip, COUNT(*) AS cnt FROM blackjack_sessions WHERE tip_mode=1 GROUP BY user_id ORDER BY total_tip DESC LIMIT 20"
            ).fetchall()
        return web.json_response(
            {
                "ok": True,
                "coin": [{"user_id": r[0], "total": int(r[1])} for r in top_coin],
                "tip": [{"user_id": r[0], "tip_total": int(r[1]), "count": int(r[2])} for r in top_tip],
            }
        )

    async def handle_create_bot_room(self, request: web.Request) -> web.Response:
        self._expire_idle_rooms()
        token = request.query.get("token", "")
        profile = self._identity_resolver(token)
        if not profile:
            raise web.HTTPUnauthorized(text="invalid_token")
        room_id = f"bot-{secrets_token(self._rng)}"
        self._room(room_id, ranked=False)
        return web.json_response(
            {
                "ok": True,
                "room_id": room_id,
                "url": f"/poker/room/{room_id}?token={token}",
            }
        )

    async def handle_room_extend(self, request: web.Request) -> web.Response:
        self._expire_idle_rooms()
        token = request.query.get("token", "")
        profile = self._identity_resolver(token)
        if not profile:
            raise web.HTTPUnauthorized(text="invalid_token")
        room_id = request.match_info.get("room_id", "")
        room = self._rooms.get(room_id)
        if room is None:
            if room_id in self._expired_room_ids:
                return web.json_response({"ok": False, "error": "room_expired"}, status=410)
            return web.json_response({"ok": False, "error": "room_not_found"}, status=404)
        if self._has_started_any_hand(room):
            return web.json_response({"ok": False, "error": "room_already_started"}, status=400)
        warning = self._idle_warning_payload(room)
        if warning is None:
            return web.json_response({"ok": False, "error": "not_in_warning_window"}, status=400)
        room.idle_delete_at_s = self._now_s() + ROOM_IDLE_DELETE_S
        room.table.server_seq += 1
        await self._broadcast(room, ack_action_id=None)
        return web.json_response({"ok": True, "delete_after_seconds": ROOM_IDLE_DELETE_S})

    def _persist_action(self, room_id: str, rec: Any) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as con:
            con.execute(
                "INSERT INTO poker_actions (room_id, hand_no, street, seat, action, amount, off_tree, gto_freq, gto_evloss_bb, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    room_id,
                    int(rec.hand_no),
                    str(rec.street),
                    int(rec.seat),
                    str(rec.action),
                    int(rec.amount),
                    1 if bool(rec.off_tree) else 0,
                    float(rec.gto_freq),
                    float(rec.gto_evloss_bb),
                    now,
                ),
            )

    def _load_rating_state(self, user_id: str) -> dict[str, float | int]:
        with sqlite3.connect(self._db_path) as con:
            row = con.execute("SELECT mmr, rp, wcp FROM rating_state WHERE user_id=?", (user_id,)).fetchone()
        if row is None:
            return {"mmr": 1500.0, "rp": 0, "wcp": 0}
        return {"mmr": float(row[0]), "rp": int(row[1]), "wcp": int(row[2])}

    def _save_rating_state(self, user_id: str, state: dict[str, float | int]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as con:
            con.execute(
                "INSERT INTO rating_state (user_id, mmr, rp, wcp, updated_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET mmr=excluded.mmr, rp=excluded.rp, wcp=excluded.wcp, updated_at=excluded.updated_at",
                (user_id, float(state["mmr"]), int(state["rp"]), int(state["wcp"]), now),
            )

    def _all_mmr_values(self, *, exclude_bots: bool) -> list[float]:
        query = "SELECT mmr, user_id FROM rating_state"
        with sqlite3.connect(self._db_path) as con:
            rows = con.execute(query).fetchall()
        vals = []
        for mmr, user_id in rows:
            if exclude_bots and str(user_id).startswith("bot:"):
                continue
            vals.append(float(mmr))
        return vals

    def _compute_wcp(self, mmr: float, all_human_ratings: list[float]) -> int:
        if not all_human_ratings:
            p = 0.5
        else:
            lower = sum(1 for value in all_human_ratings if value <= mmr)
            p = lower / len(all_human_ratings)
        n_display = max(1, len(all_human_ratings) + VIRTUAL_BOT_POPULATION)
        return int(p * n_display)

    def _join_if_needed(self, table: TableState, profile: dict[str, Any]) -> None:
        user_id = str(profile["user_id"])
        if any(player_state.user_id == user_id for player_state in table.players):
            return
        if len(table.players) >= 2:
            return
        stack = table.config.initial_stack_bb * table.config.start_bb
        table.players.append(
            PlayerState(
                user_id=user_id,
                nickname=str(profile.get("nickname") or "player"),
                avatar_url=str(profile.get("avatar_url") or ""),
                stack=stack,
            )
        )
        if table.room_id.startswith("bot-") and len(table.players) == 1:
            table.players.append(
                PlayerState(
                    user_id=f"bot:{table.room_id}",
                    nickname="MeteoBot",
                    avatar_url="",
                    stack=stack,
                )
            )
        if table.room_id.startswith("bot-") and len(table.players) == 2 and table.street == "waiting" and table.hand_no == 0:
            start_hand(table, self._rng)

    def _is_bot_turn(self, table: TableState) -> bool:
        if table.street not in {"preflop", "flop", "turn", "river"}:
            return False
        if table.to_act not in {0, 1}:
            return False
        return table.players[table.to_act].user_id.startswith("bot:")

    def _bot_action(self, table: TableState) -> tuple[str, int]:
        seat = table.to_act
        me = table.players[seat]
        opp = table.players[1 - seat]
        to_call = max(0, opp.committed - me.committed)
        if to_call > 0:
            if to_call >= me.stack:
                return "allin", 0
            return "call", 0
        return "check", 0

    def _run_bot_until_human_turn(self, room: Room) -> None:
        while self._is_bot_turn(room.table):
            action, amount = self._bot_action(room.table)
            try:
                rec = apply_action(room.table, room.table.to_act, action, amount)
            except GameRuleError:
                break
            self._persist_action(room.table.room_id, rec)

    def _seat_for_user(self, table: TableState, user_id: str) -> int | None:
        for seat, player_state in enumerate(table.players):
            if player_state.user_id == user_id:
                return seat
        return None

    def _private_state_for_user(self, table: TableState, user_id: str) -> dict[str, Any]:
        state = table.public()
        room = self._rooms.get(table.room_id)
        if room is not None:
            idle_warning = self._idle_warning_payload(room)
            if idle_warning:
                state["idle_warning"] = idle_warning
        seat = self._seat_for_user(table, user_id)
        hand_actions = [
            {
                "street": record.street,
                "seat": record.seat,
                "action": record.action,
                "amount": record.amount,
                "to_call": record.to_call,
                "off_tree": record.off_tree,
            }
            for record in table.action_log
            if record.hand_no == table.hand_no
        ]
        state["recent_actions"] = hand_actions[-10:]
        if seat is None:
            return state
        state["my_seat"] = seat
        state["my_hole"] = [str(card) for card in table.players[seat].hole]
        state["legal"] = legal_actions(table, seat)
        if table.street == "showdown":
            state["showdown_hands"] = {
                str(idx): [str(card) for card in player.hole] for idx, player in enumerate(table.players)
            }
        return state

    async def _broadcast(self, room: Room, *, ack_action_id: int | None) -> None:
        table = room.table
        for ws in list(room.sockets):
            if ws.closed:
                room.sockets.discard(ws)
                room.socket_user_ids.pop(ws, None)
                continue
            user_id = room.socket_user_ids.get(ws, "")
            payload = {
                "type": "state",
                "state": self._private_state_for_user(table, user_id),
                "ack_action_id": ack_action_id,
                "server_seq": table.server_seq,
            }
            await ws.send_json(payload)


def register_poker_routes(app: web.Application, service: PokerService) -> None:
    app.router.add_get("/poker", service.handle_lobby)
    app.router.add_get("/poker/room/{room_id}", service.handle_room)
    app.router.add_get("/poker/ws/{room_id}", service.handle_ws)
    app.router.add_get("/poker/api/gto/{room_id}", service.handle_gto)
    app.router.add_get("/poker/api/replay/{room_id}", service.handle_replay)
    app.router.add_get("/poker/api/state/{room_id}", service.handle_state)
    app.router.add_post("/poker/api/ranked/report/{room_id}", service.handle_ranked_report)
    app.router.add_get("/poker/api/ranked/leaderboard", service.handle_ranked_leaderboard)
    app.router.add_post("/poker/api/matchmaking/enqueue", service.handle_matchmaking_enqueue)
    app.router.add_get("/poker/api/matchmaking/poll", service.handle_matchmaking_poll)
    app.router.add_post("/poker/api/blackjack/play", service.handle_blackjack)
    app.router.add_post("/poker/api/blackjack/double", service.handle_blackjack_double)
    app.router.add_get("/poker/api/rankings", service.handle_rankings)
    app.router.add_post("/poker/api/room/extend/{room_id}", service.handle_room_extend)
    app.router.add_post("/poker/api/room/create-bot", service.handle_create_bot_room)


def _bj_value(cards: list[int]) -> int:
    total = 0
    aces = 0
    for card in cards:
        if card == 1:
            aces += 1
            total += 11
        else:
            total += min(card, 10)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total


def secrets_token(rng: random.Random) -> str:
    return f"{rng.getrandbits(32):08x}"


_LOBBY_HTML = """<!doctype html><html><body style='margin:0;background:radial-gradient(circle at 15% 10%,#334155 0,#020617 55%);color:#fff;font-family:Inter,system-ui,sans-serif;min-height:100vh;display:grid;place-items:center'>
<div style='text-align:center;padding:32px;border:1px solid rgba(255,255,255,.2);border-radius:20px;background:rgba(15,23,42,.65);backdrop-filter:blur(8px)'>
<h1 style='margin:0 0 10px;font-size:clamp(28px,4vw,44px);letter-spacing:.08em'>METEOBOT <span style='color:#38bdf8'>POKER</span></h1>
<p style='margin:0;opacity:.8'>Discord の <b>/poker</b> コマンドで専用ルーム URL を作成してください。</p>
</div></body></html>"""


def _room_html(room_id: str, token: str, ranked: bool) -> str:
    return f"""<!doctype html>
<html lang='ja'>
<head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width,initial-scale=1'/>
<title>Meteo Poker {room_id}</title>
<style>
:root {{
  --night:#05060d;
  --ink:#0e1627;
  --glass:rgba(8, 13, 24, 0.62);
  --edge:rgba(165, 186, 224, 0.3);
  --soft:#bfd2f4;
  --accent:#8ad8ff;
  --ruby:#ff4c6a;
  --emerald:#2ce09f;
  --shadow:0 24px 80px rgba(0, 0, 0, 0.5);
}}
* {{ box-sizing:border-box }}
body {{
  margin:0;
  font-family: Inter, "Noto Sans JP", system-ui, sans-serif;
  color:#eff6ff;
  min-height:100vh;
  background:
    radial-gradient(900px 600px at 15% -8%, rgba(83,142,255,.28), transparent 72%),
    radial-gradient(1000px 700px at 80% -16%, rgba(255,83,140,.2), transparent 76%),
    linear-gradient(180deg, #11192a 0%, #090d17 58%, var(--night) 100%);
  overflow-x:hidden;
}}
body::before {{
  content:"";
  position:fixed;
  inset:0;
  background-image: radial-gradient(rgba(255,255,255,.08) 1px, transparent 1px);
  background-size: 3px 3px;
  opacity:.08;
  pointer-events:none;
}}
.wrap {{ max-width:1280px; margin:0 auto; padding:22px; display:grid; gap:18px; }}
.hud {{
  display:flex;
  gap:12px;
  flex-wrap:wrap;
  align-items:center;
  justify-content:space-between;
  background:linear-gradient(120deg, rgba(12,23,42,.9), rgba(20,31,54,.76));
  border:1px solid var(--edge);
  box-shadow:var(--shadow);
  border-radius:18px;
  padding:13px 16px;
}}
.brand {{font-size:1.12rem;letter-spacing:.12em;font-weight:800}}
.brand small {{display:block;opacity:.8;color:var(--soft);font-weight:500;letter-spacing:.05em}}
.chip {{
  display:inline-flex;
  align-items:center;
  gap:8px;
  border-radius:999px;
  padding:7px 12px;
  border:1px solid var(--edge);
  background:rgba(7,15,30,.72);
}}
.chip b {{color:#f8fbff}}
.grid {{display:grid;grid-template-columns:2.1fr 1fr;gap:16px}}
@media (max-width:1020px) {{ .grid {{grid-template-columns:1fr}} }}
.panel {{
  position:relative;
  border-radius:20px;
  border:1px solid var(--edge);
  background:var(--glass);
  box-shadow:var(--shadow);
  backdrop-filter: blur(10px);
  padding:16px;
}}
.table {{
  min-height:560px;
  overflow:hidden;
  background: radial-gradient(circle at 50% 33%, rgba(8,26,42,.85), rgba(6,10,18,.95));
}}
.stage {{
  position:relative;
  width:100%;
  height:100%;
  min-height:520px;
  perspective:1200px;
}}
.table-3d {{
  position:absolute;
  inset:30px 24px;
  border-radius:180px;
  background: radial-gradient(circle at 50% 40%, rgba(40,53,76,.95), rgba(11,16,30,.98));
  transform: rotateX(14deg);
  border:2px solid rgba(168, 199, 255, .28);
  box-shadow: inset 0 -18px 30px rgba(0,0,0,.42), 0 25px 40px rgba(0,0,0,.55);
}}
.table-3d::before {{
  content:"";
  position:absolute;
  inset:20px;
  border-radius:140px;
  background: radial-gradient(circle at 50% 35%, rgba(31, 109, 119, .2), rgba(5, 12, 18, .65));
  border:1px solid rgba(133,177,255,.25);
}}
.table-3d::after {{
  content:"";
  position:absolute;
  inset:0;
  border-radius:180px;
  box-shadow: inset 0 0 80px rgba(92,173,255,.22);
}}
.orbit {{
  position:absolute;
  inset:60px;
  border-radius:150px;
  border:1px dashed rgba(138,216,255,.2);
  animation: spin 18s linear infinite;
}}
.orbit::before, .orbit::after {{
  content:"";
  position:absolute;
  width:12px;
  height:12px;
  border-radius:50%;
  background:radial-gradient(circle, #dbf4ff, #61c8ff);
  box-shadow:0 0 16px rgba(124,214,255,.8);
}}
.orbit::before {{top:-6px;left:20%}}
.orbit::after {{bottom:-6px;right:16%}}
@keyframes spin {{ from {{transform:rotate(0deg)}} to {{transform:rotate(360deg)}} }}
.seat {{
  position:absolute;
  width:min(45%, 360px);
  padding:11px;
  border-radius:16px;
  border:1px solid rgba(156,197,255,.24);
  background:linear-gradient(130deg, rgba(11,20,37,.95), rgba(9,16,29,.65));
  z-index:4;
  transition:transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}}
.seat.active {{
  border-color:rgba(138,216,255,.8);
  box-shadow:0 0 26px rgba(98,190,255,.35);
  transform:translateY(-3px);
}}
#seat0 {{ top:68px; left:46px; }}
#seat1 {{ bottom:48px; right:52px; }}
.meta {{display:flex;justify-content:space-between;gap:8px;color:var(--soft);font-size:.85rem}}
.cards {{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px}}
.card {{
  width:62px;
  height:92px;
  border-radius:12px;
  display:grid;
  place-items:center;
  font-size:1.65rem;
  font-weight:800;
  color:#0e1422;
  background:linear-gradient(160deg,#fff,#d6e7ff 70%);
  border:1px solid rgba(126,162,220,.45);
  box-shadow:0 10px 20px rgba(0,0,0,.35);
  transform-style:preserve-3d;
  animation:cardEnter .35s ease;
}}
.card::before {{
  content:"";
  position:absolute;
  inset:0;
  border-radius:12px;
  background:linear-gradient(130deg,rgba(255,255,255,.58),transparent 50%);
  pointer-events:none;
}}
.card.red {{ color:#e11d48; }}
.card.back {{
  background:linear-gradient(140deg,#1f2f48,#0a1220);
  color:#d7e7ff;
  font-size:.7rem;
  letter-spacing:.16em;
  border-color:rgba(160,196,255,.28);
}}
@keyframes cardEnter {{ from {{transform:translateY(8px) rotateX(15deg);opacity:.4}} to {{transform:none;opacity:1}} }}
.board {{
  position:absolute;
  left:50%;
  top:50%;
  transform:translate(-50%,-50%);
  z-index:5;
  width:min(640px,80%);
  text-align:center;
}}
.board .cards {{justify-content:center;min-height:98px}}
.pot-core {{
  margin:0 auto 12px;
  width:140px;
  height:140px;
  border-radius:50%;
  background:radial-gradient(circle at 35% 30%, rgba(165,224,255,.28), rgba(24,40,69,.92));
  border:1px solid rgba(158,191,255,.4);
  box-shadow:inset 0 0 26px rgba(100,156,255,.3), 0 0 25px rgba(66,136,240,.2);
  display:grid;
  place-items:center;
  animation: breathe 3.2s ease-in-out infinite;
}}
@keyframes breathe {{ 0%,100% {{transform:scale(1)}} 50% {{transform:scale(1.03)}} }}
.pot-core strong {{display:block;font-size:1.6rem}}
.pot-core span {{font-size:.72rem;color:var(--soft);letter-spacing:.16em}}
.stat {{display:flex;justify-content:center;gap:10px;flex-wrap:wrap;margin-bottom:10px}}
.stat span {{
  border-radius:999px;
  padding:6px 11px;
  border:1px solid rgba(158,191,255,.35);
  background:rgba(7,14,26,.66);
}}
.chips-decor {{position:absolute;right:80px;top:102px;z-index:6;display:flex;gap:10px;align-items:flex-end;pointer-events:none;}}
.stack {{
  width:24px;
  border-radius:999px;
  background:linear-gradient(180deg,#9ec7ff,#3e5c9a);
  box-shadow:0 8px 14px rgba(0,0,0,.35);
  animation:floatChip 3s ease-in-out infinite;
}}
.stack:nth-child(1) {{height:38px}}
.stack:nth-child(2) {{height:58px;animation-delay:-.8s}}
.stack:nth-child(3) {{height:48px;animation-delay:-1.4s}}
@keyframes floatChip {{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-4px)}}}}
.actions {{display:flex;gap:8px;flex-wrap:wrap}}
button {{
  border:1px solid rgba(165,186,224,.35);
  border-radius:12px;
  padding:10px 12px;
  color:#e7f1ff;
  background:linear-gradient(150deg, rgba(19,30,51,.95), rgba(10,18,32,.78));
  cursor:pointer;
  transition:all .16s ease;
}}
button:hover {{transform:translateY(-1px);border-color:var(--accent);box-shadow:0 6px 14px rgba(85,173,255,.18)}}
button.primary {{background:linear-gradient(120deg,#2e8bff,#2664ff)}}
button.warn {{border-color:rgba(255,95,124,.5);color:#ffd3db}}
input {{
  flex:1;
  min-width:130px;
  border-radius:12px;
  border:1px solid rgba(165,186,224,.35);
  background:rgba(6,12,24,.92);
  color:#eff6ff;
  padding:10px;
}}
#log {{max-height:350px;overflow:auto;display:grid;gap:6px;scrollbar-width:thin}}
.log-item {{
  padding:8px 10px;
  border-radius:11px;
  border:1px solid rgba(165,186,224,.22);
  background:linear-gradient(130deg, rgba(9,17,30,.92), rgba(11,21,37,.58));
  font-size:.88rem;
}}
.muted {{opacity:.82;color:var(--soft)}}
</style>
</head>
<body>
<div class='wrap'>
  <div class='hud'>
    <div class='brand'>METEOBOT TEXAS HOLDEM<small>room {room_id}</small></div>
    <div class='chip'>Mode: <b>{'RANKED' if ranked else 'CASUAL'}</b></div>
    <div class='chip'>GTO: <b>{'LOCKED' if ranked else 'ENABLED'}</b></div>
    <div class='chip'>Theme: <b>Neo Monochrome</b></div>
  </div>
  <div class='grid'>
    <section class='panel table'>
      <div class='stage'>
        <div class='table-3d'></div>
        <div class='orbit'></div>
        <div class='chips-decor'><div class='stack'></div><div class='stack'></div><div class='stack'></div></div>
        <div id='seat0' class='seat'><div class='meta'><b id='seat0Name'>Seat0</b><span id='seat0Stack'>0</span></div><div id='seat0Cards' class='cards'></div></div>
        <div id='seat1' class='seat'><div class='meta'><b id='seat1Name'>Seat1</b><span id='seat1Stack'>0</span></div><div id='seat1Cards' class='cards'></div></div>
        <div class='board'>
          <div class='pot-core'><div><span>POT</span><strong id='pot'>0</strong></div></div>
          <div class='stat'>
            <span>Street <b id='street'>waiting</b></span>
            <span>To Act <b id='toAct'>-</b></span>
          </div>
          <div id='boardCards' class='cards'></div>
          <div class='muted' style='margin-top:10px'>Winner: <b id='winner'>-</b> / Reason: <b id='reason'>-</b></div>
        </div>
      </div>
    </section>

    <aside class='panel'>
      <div class='actions'>
        <button class='primary' onclick="send('start')">Start Hand</button>
        <button onclick="send('next_hand')">Next Hand</button>
        <button class='warn' onclick="send('fold')">Fold</button>
        <button onclick="send('check')">Check</button>
        <button onclick="send('call')">Call</button>
      </div>
      <div class='actions' style='margin-top:8px'>
        <input id='raiseTo' type='number' step='100' min='0' placeholder='raise_to amount'>
        <button onclick='sendRaise()'>Raise To</button>
        <button onclick="send('allin')">All-in</button>
      </div>
      <div class='actions' style='margin-top:8px'>
        <button onclick='showGto()'>GTO</button>
        <button onclick='showReplay()'>Replay</button>
        <button onclick='refreshState()'>Refresh API</button>
        <button onclick='createBotRoom()'>Play vs Bot</button>
      </div>
      <div id='idleWarning' class='log-item' style='display:none;margin-top:10px'>
        <div id='idleWarningText'></div>
        <div class='actions' style='margin-top:8px'>
          <button id='idleExtendBtn' onclick='extendRoom()'>Extend 30 minutes</button>
        </div>
      </div>
      <p class='muted'>My seat: <b id='mySeat'>-</b> / Legal: <span id='legal'>-</span></p>
      <div id='log'></div>
    </aside>
  </div>
</div>

<script>
let cid = 0;
const SUIT = {{s:'♠', h:'♥', d:'♦', c:'♣'}};
const isRed = (card) => card.endsWith('h') || card.endsWith('d');
const $ = (id) => document.getElementById(id);
const ws = new WebSocket((location.protocol==='https:'?'wss://':'ws://') + location.host + '/poker/ws/{room_id}?token={token}');

function cardNode(card, hidden) {{
  const div = document.createElement('div');
  if (hidden) {{
    div.className = 'card back';
    div.textContent = 'METEO';
    return div;
  }}
  const rank = card.slice(0, -1);
  const suit = SUIT[card.slice(-1)] || '?';
  div.className = 'card' + (isRed(card) ? ' red' : '');
  div.textContent = `${{rank}}${{suit}}`;
  return div;
}}

function paintCards(nodeId, cards, hiddenCount=0) {{
  const node = $(nodeId);
  node.innerHTML = '';
  if (!cards.length && hiddenCount===0) {{
    node.appendChild(cardNode('', true));
    return;
  }}
  if (hiddenCount > 0) {{
    for (let i=0;i<hiddenCount;i++) node.appendChild(cardNode('', true));
    return;
  }}
  cards.forEach((card)=> node.appendChild(cardNode(card, false)));
}}

function actionLabel(item) {{
  const amount = item.amount ? ` ${{item.amount}}` : '';
  const extra = item.off_tree ? ' ⚠off-tree' : '';
  return `[${{item.street}}] seat${{item.seat}} ${{item.action}}${{amount}}${{extra}}`;
}}

function renderState(st) {{
  $('street').textContent = st.street || '-';
  $('pot').textContent = String(st.pot ?? '-');
  $('toAct').textContent = String(st.to_act ?? '-');
  $('mySeat').textContent = String(st.my_seat ?? '-');
  $('winner').textContent = st.winner === null || st.winner === undefined ? '-' : `seat${{st.winner}}`;
  $('reason').textContent = st.winner_reason || '-';
  $('legal').textContent = (st.legal?.actions || []).map((a)=>a.action + (a.amount?`(${{a.amount}})`:'' )).join(', ') || '-';

  const idleWarning = st.idle_warning;
  if (idleWarning?.active) {{
    const sec = Math.max(0, Number(idleWarning.remaining_seconds || 0));
    const min = Math.floor(sec / 60);
    const rem = sec % 60;
    const message = idleWarning.message || 'This room will be deleted soon due to inactivity.';
    $('idleWarningText').textContent = `${{message}} (${{min}}:${{String(rem).padStart(2, '0')}})`;
    $('idleWarning').style.display = 'block';
    $('idleExtendBtn').disabled = idleWarning.can_extend === false;
  }} else {{
    $('idleWarning').style.display = 'none';
  }}

  const players = st.players || [];
  [0,1].forEach((seat)=>{{
    const p = players.find((x)=>x.seat===seat);
    if (!p) return;
    $(`seat${{seat}}Name`).textContent = `${{p.nickname}}${{st.my_seat===seat?' (YOU)':''}}`;
    $(`seat${{seat}}Stack`).textContent = `stack ${{p.stack}} | commit ${{p.committed}}`;
    const showdown = st.showdown_hands?.[String(seat)] || [];
    const visible = st.my_seat===seat ? (st.my_hole || []) : showdown;
    paintCards(`seat${{seat}}Cards`, visible, visible.length ? 0 : 2);
    $(`seat${{seat}}`).classList.toggle('active', st.to_act===seat);
  }});

  paintCards('boardCards', st.board || []);
  const logs = st.recent_actions || [];
  $('log').innerHTML = logs.length ? logs.slice().reverse().map((item)=>`<div class='log-item'>${{actionLabel(item)}}</div>`).join('') : "<div class='log-item muted'>No actions yet.</div>";
}}

ws.onmessage = (e) => {{
  const j = JSON.parse(e.data);
  if (j.type === 'error') {{ alert(j.error); return; }}
  if (j.type !== 'state') return;
  renderState(j.state || {{}});
}};

function send(action, amount) {{
  cid += 1;
  ws.send(JSON.stringify({{action, amount: amount || 0, client_action_id: cid}}));
}}
function sendRaise() {{
  const n = parseInt($('raiseTo').value || '0', 10);
  send('raise_to', Number.isFinite(n) ? n : 0);
}}

async function showGto() {{
  const gtoResp = await fetch('/poker/api/gto/{room_id}?token={token}');
  alert(JSON.stringify(await gtoResp.json(), null, 2));
}}
async function showReplay() {{
  const replayResp = await fetch('/poker/api/replay/{room_id}?token={token}');
  alert(JSON.stringify(await replayResp.json(), null, 2));
}}
async function refreshState() {{
  const stateResp = await fetch('/poker/api/state/{room_id}?token={token}');
  const data = await stateResp.json();
  if (data?.state) renderState(data.state);
}}

async function extendRoom() {{
  const resp = await fetch('/poker/api/room/extend/{room_id}?token={token}', {{method: 'POST'}});
  const data = await resp.json();
  if (!data?.ok) {{
    alert(data?.error || 'Failed to extend room lifetime.');
    return;
  }}
  alert('Room lifetime extended by 30 minutes.');
}}

async function createBotRoom() {{
  const resp = await fetch('/poker/api/room/create-bot?token={token}', {{method: 'POST'}});
  const data = await resp.json();
  if (!data?.ok || !data?.url) {{
    alert(data?.error || 'Failed to create bot room.');
    return;
  }}
  location.href = data.url;
}}
</script>
</body>
</html>"""
