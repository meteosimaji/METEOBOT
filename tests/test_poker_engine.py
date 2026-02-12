from __future__ import annotations

import random

from web.poker.engine import Card, GameRuleError, PlayerState, TableConfig, TableState, apply_action, best_rank, gto_recommendations, replay_scores, start_hand


def _table(*, ranked: bool = False) -> TableState:
    return TableState(
        room_id="r1",
        ranked=ranked,
        config=TableConfig(initial_stack_bb=100, start_sb=100, start_bb=200, blind_up_every_hands=10),
        players=[
            PlayerState(user_id="u1", nickname="a", avatar_url="", stack=20_000),
            PlayerState(user_id="u2", nickname="b", avatar_url="", stack=20_000),
        ],
    )


def test_start_hand_posts_blinds_and_deals() -> None:
    table = _table()
    start_hand(table, random.Random(1))
    assert table.street == "preflop"
    assert table.pot == 300
    assert len(table.players[0].hole) == 2
    assert len(table.players[1].hole) == 2
    assert table.seed_commit


def test_apply_action_turn_validation() -> None:
    table = _table()
    start_hand(table, random.Random(2))
    wrong_seat = 1 - table.to_act
    try:
        apply_action(table, wrong_seat, "fold")
    except GameRuleError as exc:
        assert str(exc) == "not_your_turn"
    else:
        raise AssertionError("expected error")


def test_best_rank_flush_beats_straight() -> None:
    flush_cards = [Card("A", "h"), Card("J", "h"), Card("7", "h"), Card("4", "h"), Card("2", "h"), Card("K", "c"), Card("Q", "d")]
    straight_cards = [Card("9", "c"), Card("8", "d"), Card("7", "s"), Card("6", "h"), Card("5", "c"), Card("2", "d"), Card("A", "s")]
    assert best_rank(flush_cards) > best_rank(straight_cards)


def test_ranked_disables_gto_recommendation() -> None:
    table = _table(ranked=True)
    start_hand(table, random.Random(7))
    try:
        gto_recommendations(table, table.to_act)
    except GameRuleError as exc:
        assert str(exc) == "gto_disabled_in_ranked"
    else:
        raise AssertionError("expected ranked gto rejection")


def test_replay_scores_produced_after_actions() -> None:
    table = _table()
    start_hand(table, random.Random(3))
    seat = table.to_act
    apply_action(table, seat, "call")
    apply_action(table, table.to_act, "check")
    scores = replay_scores(table)
    assert "A" in scores and "B" in scores
    assert isinstance(scores["actions"], list)


def test_fold_payout_uses_single_pot_and_action_street_recorded() -> None:
    table = _table()
    start_hand(table, random.Random(10))
    seat = table.to_act
    rec = apply_action(table, seat, "fold")
    winner = table.players[table.winner or 0]
    assert table.street == "showdown"
    assert winner.stack == 20_100
    assert rec.street == "preflop"


def test_commits_are_kept_in_pot_across_streets() -> None:
    table = _table()
    start_hand(table, random.Random(11))
    apply_action(table, table.to_act, "call")
    apply_action(table, table.to_act, "check")
    assert table.street == "flop"
    assert table.pot == 400
