from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from itertools import combinations
import random
from typing import Any

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}
STREETS = {"preflop", "flop", "turn", "river"}


@dataclass(frozen=True)
class Card:
    rank: str
    suit: str

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"


@dataclass
class PlayerState:
    user_id: str
    nickname: str
    avatar_url: str
    stack: int
    hole: list[Card] = field(default_factory=list)
    committed: int = 0
    acted: bool = False
    folded: bool = False
    all_in: bool = False


@dataclass
class ActionRecord:
    hand_no: int
    street: str
    seat: int
    action: str
    amount: int
    to_call: int
    off_tree: bool
    gto_freq: float
    gto_evloss_bb: float


@dataclass
class TableConfig:
    initial_stack_bb: int = 100
    start_sb: int = 100
    start_bb: int = 200
    blind_up_every_hands: int = 10


@dataclass
class TableState:
    room_id: str
    config: TableConfig
    ranked: bool = False
    players: list[PlayerState] = field(default_factory=list)
    button_index: int = 0
    hand_no: int = 0
    sb: int = 100
    bb: int = 200
    street: str = "waiting"
    board: list[Card] = field(default_factory=list)
    pot: int = 0
    deck: list[Card] = field(default_factory=list)
    to_act: int = 0
    min_raise_to: int = 0
    current_bet: int = 0
    winner: int | None = None
    winner_reason: str = ""
    action_log: list[ActionRecord] = field(default_factory=list)
    server_seq: int = 0
    seed_commit: str = ""
    seed_reveal: str = ""
    gto_viewer_user_id: str | None = None

    def public(self) -> dict[str, Any]:
        players = []
        for idx, p in enumerate(self.players):
            players.append(
                {
                    "seat": idx,
                    "user_id": p.user_id,
                    "nickname": p.nickname,
                    "avatar_url": p.avatar_url,
                    "stack": p.stack,
                    "committed": p.committed,
                    "acted": p.acted,
                    "folded": p.folded,
                    "all_in": p.all_in,
                }
            )
        return {
            "room_id": self.room_id,
            "ranked": self.ranked,
            "hand_no": self.hand_no,
            "sb": self.sb,
            "bb": self.bb,
            "street": self.street,
            "board": [str(c) for c in self.board],
            "pot": self.pot,
            "to_act": self.to_act,
            "min_raise_to": self.min_raise_to,
            "current_bet": self.current_bet,
            "winner": self.winner,
            "winner_reason": self.winner_reason,
            "server_seq": self.server_seq,
            "seed_commit": self.seed_commit,
            "seed_reveal": self.seed_reveal if self.street == "showdown" else "",
            "gto_viewer_user_id": self.gto_viewer_user_id,
            "players": players,
        }


class GameRuleError(ValueError):
    pass


def make_deck(rng: random.Random) -> list[Card]:
    deck = [Card(r, s) for r in RANKS for s in SUITS]
    for i in range(len(deck) - 1, 0, -1):
        j = rng.randint(0, i)
        deck[i], deck[j] = deck[j], deck[i]
    return deck


def _level_multiplier(hand_no: int, every: int) -> int:
    if every <= 0:
        return 1
    return 2 ** ((max(1, hand_no) - 1) // every)


def start_hand(table: TableState, rng: random.Random) -> None:
    if len(table.players) != 2:
        raise GameRuleError("heads_up_only")
    table.hand_no += 1
    mult = _level_multiplier(table.hand_no, table.config.blind_up_every_hands)
    table.sb = table.config.start_sb * mult
    table.bb = table.config.start_bb * mult
    table.board = []
    table.pot = 0
    table.street = "preflop"
    table.winner = None
    table.winner_reason = ""
    table.current_bet = table.bb
    table.min_raise_to = table.bb * 2
    table.seed_reveal = f"{table.room_id}:{table.hand_no}:{rng.random():.24f}"
    table.seed_commit = sha256(table.seed_reveal.encode("utf-8")).hexdigest()
    table.deck = make_deck(rng)
    table.action_log = [r for r in table.action_log if r.hand_no != table.hand_no]
    table.gto_viewer_user_id = None

    table.button_index = 0 if table.hand_no % 2 == 1 else 1
    sb_idx = table.button_index
    bb_idx = 1 - sb_idx

    for player_state in table.players:
        player_state.hole = [table.deck.pop(), table.deck.pop()]
        player_state.committed = 0
        player_state.acted = False
        player_state.folded = False
        player_state.all_in = False

    _post_blind(table, table.players[sb_idx], table.sb)
    _post_blind(table, table.players[bb_idx], table.bb)
    table.to_act = sb_idx
    table.server_seq += 1


def _post_blind(table: TableState, player_state: PlayerState, amount: int) -> None:
    if amount >= player_state.stack:
        posted = player_state.stack
        player_state.committed += posted
        player_state.stack = 0
        player_state.all_in = True
        table.pot += posted
        return
    player_state.stack -= amount
    player_state.committed += amount
    table.pot += amount


def _collect_committed_for_new_street(table: TableState) -> None:
    # Pot is incremented when chips are committed, so only the per-street trackers reset here.
    for player_state in table.players:
        player_state.committed = 0
        player_state.acted = False


def _post_all_in(table: TableState, player_state: PlayerState) -> None:
    committed = player_state.stack
    player_state.committed += committed
    table.pot += committed
    player_state.stack = 0
    player_state.all_in = True


def _put_chips(table: TableState, player_state: PlayerState, amount: int) -> None:
    if amount < 0:
        raise GameRuleError("negative_amount")
    if amount >= player_state.stack:
        _post_all_in(table, player_state)
        return
    player_state.stack -= amount
    player_state.committed += amount
    table.pot += amount


def _active_indexes(table: TableState) -> list[int]:
    return [i for i, player_state in enumerate(table.players) if not player_state.folded]


def _other(seat: int) -> int:
    return 1 - seat


def _street_cards(street: str) -> int:
    return {"flop": 3, "turn": 1, "river": 1}.get(street, 0)


def _next_street(table: TableState) -> None:
    if table.street == "preflop":
        table.street = "flop"
    elif table.street == "flop":
        table.street = "turn"
    elif table.street == "turn":
        table.street = "river"
    else:
        _resolve_showdown(table)
        return

    draw = _street_cards(table.street)
    for _ in range(draw):
        table.board.append(table.deck.pop())
    _collect_committed_for_new_street(table)
    table.current_bet = 0
    table.min_raise_to = table.bb
    table.to_act = _other(table.button_index)


def _should_advance(table: TableState) -> bool:
    active_players = [player_state for player_state in table.players if not player_state.folded]
    if len(active_players) <= 1:
        return True
    if all(player_state.all_in for player_state in active_players):
        return True
    return all(player_state.acted for player_state in active_players) and active_players[0].committed == active_players[1].committed


def _raise_sizes_for_state(table: TableState) -> list[int]:
    if table.street == "preflop":
        return [int(table.bb * 2.5), int(table.bb * 3), int(table.bb * 4)]
    base_pot = max(table.pot, table.bb)
    return [
        ((base_pot // 3 + 99) // 100) * 100,
        (((base_pot * 2) // 3 + 99) // 100) * 100,
        ((base_pot + 99) // 100) * 100,
    ]


def gto_recommendations(table: TableState, seat: int) -> dict[str, Any]:
    if table.ranked:
        raise GameRuleError("gto_disabled_in_ranked")
    if table.street not in STREETS:
        raise GameRuleError("hand_not_running")
    if seat not in {0, 1}:
        raise GameRuleError("bad_seat")
    player_state = table.players[seat]
    opponent_state = table.players[_other(seat)]
    to_call = max(0, opponent_state.committed - player_state.committed)
    sizes = _raise_sizes_for_state(table)
    actions: list[dict[str, Any]] = []
    if to_call == 0:
        actions.append({"action": "check", "frequency": 0.35, "ev": 0.0})
    else:
        actions.append({"action": "fold", "frequency": 0.20, "ev": -to_call / max(table.bb, 1)})
        actions.append({"action": "call", "frequency": 0.35, "ev": -0.05})
    raise_freq = 0.45 / max(1, len(sizes))
    for target in sizes:
        if target <= player_state.committed:
            continue
        actions.append(
            {
                "action": "raise_to",
                "amount": target,
                "frequency": raise_freq,
                "ev": 0.15 - abs(target - (table.pot + to_call)) / max(table.bb * 20, 1),
            }
        )
    actions.append({"action": "allin", "amount": player_state.committed + player_state.stack, "frequency": 0.05, "ev": -0.2})
    actions.sort(key=lambda item: float(item["ev"]), reverse=True)
    return {
        "street": table.street,
        "to_call": to_call,
        "actions": actions,
        "off_tree_policy": "linear_interpolation",
    }


def legal_actions(table: TableState, seat: int) -> dict[str, Any]:
    if table.street not in STREETS:
        return {"actions": []}
    if seat != table.to_act:
        return {"actions": []}
    player_state = table.players[seat]
    opponent_state = table.players[_other(seat)]
    if player_state.folded or player_state.all_in:
        return {"actions": []}

    to_call = max(0, opponent_state.committed - player_state.committed)
    can_raise_to = max(table.min_raise_to, player_state.committed + to_call + table.bb)
    all_in_to = player_state.committed + player_state.stack
    min_raise_to = min(can_raise_to, all_in_to)
    raise_targets = sorted({
        min(all_in_to, target)
        for target in _raise_sizes_for_state(table)
        if target > player_state.committed
    })
    if min_raise_to > player_state.committed:
        raise_targets.append(min_raise_to)
    raise_targets = sorted({target for target in raise_targets if target > player_state.committed and target <= all_in_to})

    actions: list[dict[str, Any]] = []
    if to_call == 0:
        actions.append({"action": "check"})
    else:
        actions.append({"action": "fold"})
        actions.append({"action": "call", "amount": to_call})
    for target in raise_targets:
        if target == all_in_to:
            continue
        actions.append({"action": "raise_to", "amount": target})
    actions.append({"action": "allin", "amount": all_in_to})
    return {
        "to_call": to_call,
        "min_raise_to": min_raise_to,
        "all_in_to": all_in_to,
        "actions": actions,
    }


def apply_action(table: TableState, seat: int, action: str, amount: int = 0) -> ActionRecord:
    if table.street not in STREETS:
        raise GameRuleError("hand_not_running")
    if seat != table.to_act:
        raise GameRuleError("not_your_turn")

    player_state = table.players[seat]
    opponent_state = table.players[_other(seat)]
    if player_state.folded or player_state.all_in:
        raise GameRuleError("player_inactive")

    to_call = max(0, opponent_state.committed - player_state.committed)
    recommendation = gto_recommendations(table, seat) if not table.ranked else {"actions": []}
    off_tree = False
    gto_freq = 0.0
    gto_evloss_bb = 0.0

    action_street = table.street
    if action == "fold":
        player_state.folded = True
        table.winner = _other(seat)
        _award_fold(table)
    elif action == "check":
        if to_call != 0:
            raise GameRuleError("cannot_check")
        player_state.acted = True
    elif action == "call":
        _put_chips(table, player_state, to_call)
        player_state.acted = True
    elif action in {"bet", "raise", "raise_to", "allin"}:
        target = player_state.committed + player_state.stack if action == "allin" else amount
        if target % 100 != 0:
            raise GameRuleError("must_be_100_multiple")
        if target <= player_state.committed:
            raise GameRuleError("raise_target_too_small")
        full_raise = max(table.bb, table.current_bet - opponent_state.committed)
        if target < table.min_raise_to and target < player_state.committed + player_state.stack:
            raise GameRuleError("below_min_raise")
        delta = target - player_state.committed
        _put_chips(table, player_state, delta)
        previous_bet = table.current_bet
        table.current_bet = max(table.current_bet, player_state.committed)
        last_raise = player_state.committed - opponent_state.committed
        if last_raise >= full_raise:
            table.min_raise_to = player_state.committed + last_raise
            opponent_state.acted = False
        player_state.acted = True
        if player_state.committed < previous_bet + full_raise and player_state.all_in:
            off_tree = True
    else:
        raise GameRuleError("unknown_action")

    if recommendation.get("actions"):
        best_ev = max(float(item.get("ev", 0.0)) for item in recommendation["actions"])
        chosen_candidates: list[dict[str, Any]] = []
        for item in recommendation["actions"]:
            if item.get("action") != ("raise_to" if action in {"raise", "bet", "raise_to"} else action):
                continue
            if item.get("action") == "raise_to":
                rec_amount = int(item.get("amount") or 0)
                if rec_amount == amount:
                    chosen_candidates = [item]
                    break
                if abs(rec_amount - amount) <= 100:
                    chosen_candidates.append(item)
            else:
                chosen_candidates = [item]
                break
        if chosen_candidates:
            chosen = max(chosen_candidates, key=lambda item: float(item.get("ev", 0.0)))
            gto_freq = float(chosen.get("frequency", 0.0))
            gto_evloss_bb = max(0.0, best_ev - float(chosen.get("ev", 0.0)))
        else:
            off_tree = True
            gto_evloss_bb = max(0.15, best_ev)

    table.to_act = _other(seat)
    if _should_advance(table):
        if len(_active_indexes(table)) == 1:
            table.winner = _active_indexes(table)[0]
            _award_fold(table)
        elif all(player_state.all_in for player_state in table.players if not player_state.folded):
            _runout_and_showdown(table)
        else:
            _next_street(table)

    table.server_seq += 1
    rec = ActionRecord(
        hand_no=table.hand_no,
        street=action_street,
        seat=seat,
        action=action,
        amount=amount,
        to_call=to_call,
        off_tree=off_tree,
        gto_freq=gto_freq,
        gto_evloss_bb=gto_evloss_bb,
    )
    table.action_log.append(rec)
    return rec


def replay_scores(table: TableState) -> dict[str, Any]:
    hand_actions = [rec for rec in table.action_log if rec.hand_no == table.hand_no]
    if not hand_actions:
        return {"A": 100.0, "B": 100.0, "actions": []}

    scored: list[dict[str, Any]] = []
    a_vals: list[float] = []
    b_vals: list[float] = []
    for rec in hand_actions:
        sa = max(0.0, min(100.0, rec.gto_freq * 100.0 - rec.gto_evloss_bb * 10.0))
        sb = max(0.0, min(100.0, 100.0 - (rec.gto_evloss_bb * 8.0 + (20.0 if rec.off_tree else 0.0))))
        a_vals.append(sa)
        b_vals.append(sb)
        scored.append(
            {
                "seat": rec.seat,
                "street": rec.street,
                "action": rec.action,
                "amount": rec.amount,
                "off_tree": rec.off_tree,
                "score_A": round(sa, 2),
                "score_B": round(sb, 2),
            }
        )
    return {
        "A": round(sum(a_vals) / len(a_vals), 2),
        "B": round(sum(b_vals) / len(b_vals), 2),
        "actions": scored,
    }


def _award_fold(table: TableState) -> None:
    _collect_committed_for_new_street(table)
    if table.winner is None:
        return
    table.winner_reason = "fold"
    table.players[table.winner].stack += table.pot
    table.pot = 0
    table.street = "showdown"


def _runout_and_showdown(table: TableState) -> None:
    while len(table.board) < 5:
        table.board.append(table.deck.pop())
    _resolve_showdown(table)


def _resolve_showdown(table: TableState) -> None:
    _collect_committed_for_new_street(table)
    rank0 = best_rank(table.players[0].hole + table.board)
    rank1 = best_rank(table.players[1].hole + table.board)
    if rank0 > rank1:
        table.winner = 0
        table.winner_reason = hand_rank_name(rank0[0])
        table.players[0].stack += table.pot
    elif rank1 > rank0:
        table.winner = 1
        table.winner_reason = hand_rank_name(rank1[0])
        table.players[1].stack += table.pot
    else:
        split = table.pot // 2
        table.players[0].stack += split
        table.players[1].stack += table.pot - split
        table.winner = None
        table.winner_reason = "split_pot"
    table.pot = 0
    table.street = "showdown"


def hand_rank_name(rank_index: int) -> str:
    names = {
        0: "high_card",
        1: "one_pair",
        2: "two_pair",
        3: "three_of_a_kind",
        4: "straight",
        5: "flush",
        6: "full_house",
        7: "four_of_a_kind",
        8: "straight_flush",
    }
    return names.get(rank_index, "unknown")


def best_rank(cards: list[Card]) -> tuple[int, list[int]]:
    return max(rank_five(list(combo)) for combo in combinations(cards, 5))


def rank_five(cards: list[Card]) -> tuple[int, list[int]]:
    ranks = sorted((RANK_VALUE[card.rank] for card in cards), reverse=True)
    counts: dict[int, int] = {}
    for rank in ranks:
        counts[rank] = counts.get(rank, 0) + 1
    by_count = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    flush = len({card.suit for card in cards}) == 1
    unique = sorted(set(ranks), reverse=True)
    straight_high = 0
    if len(unique) == 5 and unique[0] - unique[4] == 4:
        straight_high = unique[0]
    if unique == [14, 5, 4, 3, 2]:
        straight_high = 5

    if flush and straight_high:
        return 8, [straight_high]
    if by_count[0][1] == 4:
        four = by_count[0][0]
        kicker = max(rank for rank in ranks if rank != four)
        return 7, [four, kicker]
    if by_count[0][1] == 3 and by_count[1][1] == 2:
        return 6, [by_count[0][0], by_count[1][0]]
    if flush:
        return 5, ranks
    if straight_high:
        return 4, [straight_high]
    if by_count[0][1] == 3:
        trip = by_count[0][0]
        kickers = sorted([rank for rank in ranks if rank != trip], reverse=True)
        return 3, [trip, *kickers]
    if by_count[0][1] == 2 and by_count[1][1] == 2:
        hi_pair = max(by_count[0][0], by_count[1][0])
        lo_pair = min(by_count[0][0], by_count[1][0])
        kicker = max(rank for rank in ranks if rank not in {hi_pair, lo_pair})
        return 2, [hi_pair, lo_pair, kicker]
    if by_count[0][1] == 2:
        pair = by_count[0][0]
        kickers = sorted([rank for rank in ranks if rank != pair], reverse=True)
        return 1, [pair, *kickers]
    return 0, ranks
