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
    replay_scores,
    start_hand,
)

IdentityResolver = Callable[[str], dict[str, Any] | None]
VIRTUAL_BOT_POPULATION = 10000
MATCHMAKING_BOT_WAIT_S = 15


@dataclass
class Room:
    table: TableState
    sockets: set[web.WebSocketResponse]
    lock: asyncio.Lock
    last_client_action_id_by_user: dict[str, int]


class PokerService:
    def __init__(self, data_root: Path, identity_resolver: IdentityResolver) -> None:
        self._rooms: dict[str, Room] = {}
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

    def _room(self, room_id: str, *, ranked: bool) -> Room:
        room = self._rooms.get(room_id)
        if room:
            if ranked and not room.table.ranked:
                room.table.ranked = True
            return room
        table = TableState(room_id=room_id, config=TableConfig(), ranked=ranked)
        room = Room(table=table, sockets=set(), lock=asyncio.Lock(), last_client_action_id_by_user={})
        self._rooms[room_id] = room
        return room

    async def handle_lobby(self, request: web.Request) -> web.Response:
        return web.Response(text=_LOBBY_HTML, content_type="text/html")

    async def handle_room(self, request: web.Request) -> web.Response:
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
                    if client_action_id:
                        room.last_client_action_id_by_user[user_id] = client_action_id
                    await self._broadcast(room, ack_action_id=client_action_id or None)
        finally:
            room.sockets.discard(ws)
        return ws

    async def handle_gto(self, request: web.Request) -> web.Response:
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
        token = request.query.get("token", "")
        profile = self._identity_resolver(token)
        if not profile:
            raise web.HTTPUnauthorized(text="invalid_token")
        room_id = request.match_info.get("room_id", "")
        ranked = bool(profile.get("ranked", False))
        room = self._room(room_id, ranked=ranked)
        replay = replay_scores(room.table)
        players = [
            {
                "user_id": p.user_id,
                "nickname": p.nickname,
                "hole": [str(card) for card in p.hole],
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

    async def handle_ranked_report(self, request: web.Request) -> web.Response:
        token = request.query.get("token", "")
        profile = self._identity_resolver(token)
        if not profile:
            raise web.HTTPUnauthorized(text="invalid_token")
        if not bool(profile.get("ranked", False)):
            return web.json_response({"ok": False, "error": "ranked_only"}, status=400)
        body = await request.json()
        winner_user_id = str(body.get("winner_user_id") or "")
        room_id = request.match_info.get("room_id", "")
        room = self._rooms.get(room_id)
        if room is None:
            return web.json_response({"ok": False, "error": "room_not_found"}, status=404)
        player_ids = [p.user_id for p in room.table.players]
        if winner_user_id not in player_ids:
            return web.json_response({"ok": False, "error": "winner_not_in_room"}, status=400)
        loser_ids = [uid for uid in player_ids if uid != winner_user_id]
        if not loser_ids:
            return web.json_response({"ok": False, "error": "need_two_players"}, status=400)
        loser_user_id = loser_ids[0]
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
            }
        )

    async def handle_ranked_leaderboard(self, request: web.Request) -> web.Response:
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
            tip_paid = ((bet * 5 + 99) // 100 // 100) * 100
            delta -= tip_paid

        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as con:
            con.execute(
                "INSERT INTO blackjack_sessions (user_id, bet, result, tip_mode, tip_paid, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, bet, result, int(tip_mode), tip_paid, now),
            )
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
            }
        )

    async def handle_blackjack_double(self, request: web.Request) -> web.Response:
        token = request.query.get("token", "")
        profile = self._identity_resolver(token)
        if not profile:
            raise web.HTTPUnauthorized(text="invalid_token")
        body = await request.json()
        amount = int(body.get("amount") or 0)
        if amount <= 0:
            return web.json_response({"ok": False, "error": "invalid_amount"}, status=400)
        win = bool(self._rng.randint(0, 1))
        delta = amount if win else -amount
        user_id = str(profile["user_id"])
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as con:
            con.execute(
                "INSERT INTO coin_ledger (user_id, delta, reason, created_at) VALUES (?, ?, ?, ?)",
                (user_id, delta, "blackjack:double_up", now),
            )
        return web.json_response({"ok": True, "win": win, "delta": delta})

    async def handle_rankings(self, request: web.Request) -> web.Response:
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

    def _seat_for_user(self, table: TableState, user_id: str) -> int | None:
        for seat, player_state in enumerate(table.players):
            if player_state.user_id == user_id:
                return seat
        return None

    async def _broadcast(self, room: Room, *, ack_action_id: int | None) -> None:
        table = room.table
        payload = {
            "type": "state",
            "state": table.public(),
            "ack_action_id": ack_action_id,
            "server_seq": table.server_seq,
        }
        for ws in list(room.sockets):
            if ws.closed:
                room.sockets.discard(ws)
                continue
            await ws.send_json(payload)


def register_poker_routes(app: web.Application, service: PokerService) -> None:
    app.router.add_get("/poker", service.handle_lobby)
    app.router.add_get("/poker/room/{room_id}", service.handle_room)
    app.router.add_get("/poker/ws/{room_id}", service.handle_ws)
    app.router.add_get("/poker/api/gto/{room_id}", service.handle_gto)
    app.router.add_get("/poker/api/replay/{room_id}", service.handle_replay)
    app.router.add_post("/poker/api/ranked/report/{room_id}", service.handle_ranked_report)
    app.router.add_get("/poker/api/ranked/leaderboard", service.handle_ranked_leaderboard)
    app.router.add_post("/poker/api/matchmaking/enqueue", service.handle_matchmaking_enqueue)
    app.router.add_get("/poker/api/matchmaking/poll", service.handle_matchmaking_poll)
    app.router.add_post("/poker/api/blackjack/play", service.handle_blackjack)
    app.router.add_post("/poker/api/blackjack/double", service.handle_blackjack_double)
    app.router.add_get("/poker/api/rankings", service.handle_rankings)


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


_LOBBY_HTML = """<!doctype html><html><body style='background:#111;color:#fff;font-family:sans-serif'>
<h1>simajilord Poker</h1>
<p>Discordの /poker コマンドでルームURLを発行してください。</p>
</body></html>"""


def _room_html(room_id: str, token: str, ranked: bool) -> str:
    return f"""<!doctype html>
<html><body style='background:#111;color:#fff;font-family:sans-serif'>
<h2>HU NLHE Room {room_id}</h2>
<p>mode: {'ranked' if ranked else 'casual'} / GTO: {'disabled' if ranked else 'enabled'}</p>
<div id='s' style='white-space:pre-wrap'></div>
<div>
<button onclick=send('start')>start</button>
<button onclick=send('next_hand')>next_hand</button>
<button onclick=send('fold')>fold</button>
<button onclick=send('check')>check</button>
<button onclick=send('call')>call</button>
<button onclick="send('raise_to', parseInt(prompt('raise_to?')||'0',10))">raise_to</button>
<button onclick="fetch('/poker/api/gto/{room_id}?token={token}').then(r=>r.json()).then(j=>alert(JSON.stringify(j,null,2)))">GTO</button>
<button onclick="fetch('/poker/api/replay/{room_id}?token={token}').then(r=>r.json()).then(j=>alert(JSON.stringify(j,null,2)))">Replay</button>
</div>
<script>
let cid = 0;
const ws = new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/poker/ws/{room_id}?token={token}');
ws.onmessage = (e) => {{ const j = JSON.parse(e.data); document.getElementById('s').textContent = JSON.stringify(j, null, 2); }};
function send(action, amount) {{
  cid += 1;
  ws.send(JSON.stringify({{action, amount: amount||0, client_action_id: cid}}));
}}
</script>
</body></html>"""
