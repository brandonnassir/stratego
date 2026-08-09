"""Derived event content and within-ply ordering.

Covers `09_public_event_and_replay_schema.md` sections 5-10 and 17.
"""

import json

import pytest

from stratego.engine.constants import BLUE, RED
from stratego.engine.events import (
    EVENT_BEHAVIOR,
    EVENT_COMBAT,
    EVENT_GAME_END,
    EVENT_IDENTITY_REVEAL,
    EVENT_MOVE,
)
from stratego.engine.random_play import play_random_game
from tests.helpers import make_position, piece_at, play

# Ordering defined by `09_...` section 17.
CATEGORY_ORDER = {
    EVENT_MOVE: 0,
    EVENT_IDENTITY_REVEAL: 1,
    EVENT_COMBAT: 2,
    EVENT_BEHAVIOR: 3,
    EVENT_GAME_END: 4,
}

BEHAVIOR_ORDER = {
    "threat": 0,
    "evade": 1,
    "declined_attack": 2,
    "protect": 3,
    "was_protected": 4,
}


def test_a_quiet_move_emits_only_a_move_event():
    state = make_position(
        red={"e3": "captain", "a1": "flag"}, blue={"j9": "scout", "j10": "flag"}
    )
    events = play(state, "e3 e4")[0]
    assert [event["event_type"] for event in events] == [EVENT_MOVE]


def test_move_event_fields():
    state = make_position(
        red={"e3": "captain", "a1": "flag"}, blue={"e4": "marshal", "j10": "flag"}
    )
    target = piece_at(state, "e4")
    mover = piece_at(state, "e3")
    events = play(state, "e3 e4")[0]
    move = events[0]

    assert move["event_type"] == EVENT_MOVE
    assert move["ply"] == 1
    assert move["player"] == "red"
    assert move["distance"] == 1
    assert move["is_attack"] is True
    assert move["source"] == mover.starting_square
    assert move["destination"] == target.starting_square
    assert "piece_type" not in move


def test_scout_move_event_records_the_distance():
    state = make_position(
        red={"a1": "scout", "j1": "flag"}, blue={"j9": "scout", "j10": "flag"}
    )
    events = play(state, "a1 a5")[0]
    assert events[0]["distance"] == 4


def test_event_categories_appear_in_the_documented_order():
    for seed in range(8):
        state, _ = play_random_game(seed)
        by_ply: dict[int, list[dict]] = {}
        for event in state.events:
            by_ply.setdefault(event["ply"], []).append(event)
        for ply, events in by_ply.items():
            keys = [CATEGORY_ORDER[event["event_type"]] for event in events]
            assert keys == sorted(keys), f"ply {ply} of game {seed}"


def test_identity_reveals_are_ordered_by_stable_piece_identifier():
    """Ordering uses the canonical `(owner, setup_slot)` identifier.

    The engine packs that pair into an integer with red before blue, so the
    comparison is made on the identifier rather than on its display string.
    """
    from stratego.engine.pieces import piece_id_from_name

    for seed in range(8):
        state, _ = play_random_game(seed)
        by_ply: dict[int, list[int]] = {}
        for event in state.events:
            if event["event_type"] == EVENT_IDENTITY_REVEAL:
                by_ply.setdefault(event["ply"], []).append(
                    piece_id_from_name(event["piece_id"])
                )
        for ply, identifiers in by_ply.items():
            assert identifiers == sorted(identifiers), f"ply {ply} of game {seed}"


def test_behaviour_events_are_ordered_by_type_then_actor():
    for seed in range(8):
        state, _ = play_random_game(seed)
        by_ply: dict[int, list[tuple]] = {}
        for event in state.events:
            if event["event_type"] == EVENT_BEHAVIOR:
                by_ply.setdefault(event["ply"], []).append(
                    (BEHAVIOR_ORDER[event["behavior_type"]], event["actor_piece_id"])
                )
        for ply, keys in by_ply.items():
            assert keys == sorted(keys), f"ply {ply} of game {seed}"


def test_combat_event_fields():
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e4": "marshal", "j10": "flag"},
        acting_player=RED,
    )
    events = play(state, "e3 e4")[0]
    combat = next(event for event in events if event["event_type"] == EVENT_COMBAT)
    assert combat["outcome"] == "defender_survives"
    assert combat["attacker_type"] == "captain"
    assert combat["defender_type"] == "marshal"
    assert combat["flag_captured"] is False


def test_flag_capture_marks_the_combat_and_ends_the_game():
    state = make_position(
        red={"e3": "captain"}, blue={"e4": "flag", "j10": "scout"}, acting_player=RED
    )
    events = play(state, "e3 e4")[0]
    combat = next(event for event in events if event["event_type"] == EVENT_COMBAT)
    assert combat["flag_captured"] is True

    end = events[-1]
    assert end["event_type"] == EVENT_GAME_END
    assert end["result"] == "red_win"
    assert end["winner"] == "red"
    assert end["terminal_reason"] == "flag_capture"


def test_game_end_event_reports_a_draw_without_a_winner():
    state = make_position(
        red={"a1": "flag", "a3": "captain"},
        blue={"j10": "flag", "a4": "bomb", "j9": "bomb"},
        acting_player=RED,
    )
    events = play(state, "a3 a4")[0]
    end = events[-1]
    assert end["event_type"] == EVENT_GAME_END
    assert end["winner"] is None
    assert end["result"] == "draw"
    assert end["terminal_reason"] == "both_no_legal_move_draw"


def test_exactly_one_game_end_event_per_finished_game():
    for seed in range(8):
        state, _ = play_random_game(seed)
        ends = [event for event in state.events if event["event_type"] == EVENT_GAME_END]
        assert len(ends) == 1
        assert ends[-1] is state.events[-1]


def test_every_event_is_json_serialisable():
    state, _ = play_random_game(4)
    text = json.dumps(state.events)
    assert json.loads(text) == state.events


def test_ply_numbers_increase_and_match_the_move_count():
    state, _ = play_random_game(6)
    plies = [event["ply"] for event in state.events]
    assert plies == sorted(plies)
    assert max(plies) == state.total_moves


def test_reveal_is_emitted_only_when_knowledge_actually_changes():
    state = make_position(
        red={"e3": "captain", "d3": "scout", "a1": "flag"},
        blue={"e4": "marshal", "j10": "flag", "j9": "scout"},
        acting_player=RED,
    )
    first = play(state, "e3 e4")[0]
    assert len([e for e in first if e["event_type"] == EVENT_IDENTITY_REVEAL]) == 2

    # The marshal is already known to red; attacking it again reveals only the
    # newly involved red scout.
    second = play(state, "j9 i9", "d3 d4", "e4 d4")[-1]
    reveals = [e for e in second if e["event_type"] == EVENT_IDENTITY_REVEAL]
    assert [event["owner"] for event in reveals] == ["red"]
