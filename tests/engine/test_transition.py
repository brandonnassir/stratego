"""Atomic state transitions and the illegal-action guarantee.

Covers `03_game_engine_spec.md` section 10 and section 9 of the Phase Two
instructions.
"""

import pytest

from stratego.engine.actions import encode_action
from stratego.engine.constants import BLUE, RED
from stratego.engine.legal_moves import legal_actions
from stratego.engine.random_play import play_random_game, play_random_game_to_ply
from stratego.engine.state import state_fingerprint
from stratego.engine.transition import (
    IllegalActionError,
    TerminalStateError,
    apply_action,
)
from tests.helpers import make_position, nonterminal_state, piece_at, play, square


def test_ordinary_move_updates_occupancy_and_moved_status():
    state = make_position(red={"e3": "captain"}, blue={"a10": "flag"}, acting_player=RED)
    captain = piece_at(state, "e3")
    assert not captain.has_moved

    play(state, "e3 e4")

    assert state.board[square("e3")] is None
    assert state.board[square("e4")] == captain.piece_id
    assert captain.current_square == square("e4")
    assert captain.has_moved


def test_acting_player_alternates():
    state = make_position(
        red={"e3": "captain", "a1": "flag"}, blue={"e7": "captain", "a10": "flag"}
    )
    assert state.acting_player == RED
    play(state, "e3 e4")
    assert state.acting_player == BLUE
    play(state, "e7 e6")
    assert state.acting_player == RED


def test_move_counters_advance():
    state = make_position(
        red={"e3": "captain", "a1": "flag"}, blue={"e7": "captain", "a10": "flag"}
    )
    assert state.total_moves == 0 and state.battleless_moves == 0
    play(state, "e3 e4")
    assert state.total_moves == 1 and state.battleless_moves == 1
    play(state, "e7 e6")
    assert state.total_moves == 2 and state.battleless_moves == 2


def test_recent_move_history_records_the_move():
    state = make_position(
        red={"e3": "captain", "a1": "flag"}, blue={"e7": "captain", "a10": "flag"}
    )
    play(state, "e3 e4")
    latest = state.recent_moves[-1]
    assert latest.ply == 1
    assert latest.player == RED
    assert latest.source == square("e3")
    assert latest.destination == square("e4")
    assert latest.destination_had_opponent is False
    assert latest.target_piece_id is None


def test_recent_move_history_is_bounded_to_sixteen_plies():
    state, _ = play_random_game(3)
    assert len(state.recent_moves) <= 16


@pytest.mark.parametrize(
    "illegal",
    [
        ("e3", "d4"),  # diagonal
        ("e3", "e5"),  # two squares for a non-Scout
        ("e3", "e3"),  # no movement
        ("e4", "e5"),  # not the acting player's piece
        ("a1", "a2"),  # immovable flag
        ("b5", "c5"),  # onto a lake
    ],
)
def test_illegal_action_leaves_the_entire_state_unchanged(illegal):
    state = make_position(
        red={"e3": "captain", "a1": "flag", "b5": "miner"},
        blue={"e4": "captain", "a10": "flag"},
        acting_player=RED,
    )
    before = state_fingerprint(state)

    source, destination = (square(name) for name in illegal)
    with pytest.raises(IllegalActionError):
        apply_action(state, encode_action(source, destination))

    assert state_fingerprint(state) == before


def test_illegal_action_after_many_plies_leaves_state_unchanged():
    state = nonterminal_state(60, first_seed=21)
    before = state_fingerprint(state)
    legal = set(legal_actions(state))
    illegal = next(action for action in range(10_000) if action not in legal)

    with pytest.raises(IllegalActionError):
        apply_action(state, illegal)

    assert state_fingerprint(state) == before


def test_transition_on_a_terminal_state_is_refused():
    state = make_position(
        red={"e3": "scout"}, blue={"e4": "flag", "j10": "captain"}, acting_player=RED
    )
    play(state, "e3 e4")
    assert state.terminal
    with pytest.raises(TerminalStateError):
        apply_action(state, encode_action(square("e4"), square("e5")))


def test_terminal_state_reports_no_legal_actions():
    state = make_position(
        red={"e3": "scout"}, blue={"e4": "flag", "j10": "captain"}, acting_player=RED
    )
    play(state, "e3 e4")
    assert legal_actions(state) == []


def test_attack_records_the_target_in_recent_history():
    state = make_position(
        red={"e3": "marshal"}, blue={"e4": "captain", "a10": "flag"}, acting_player=RED
    )
    defender = piece_at(state, "e4")
    play(state, "e3 e4")
    latest = state.recent_moves[-1]
    assert latest.destination_had_opponent is True
    assert latest.target_piece_id == defender.piece_id


def test_action_history_matches_the_moves_played():
    state = make_position(
        red={"e3": "captain", "a1": "flag"}, blue={"e7": "captain", "a10": "flag"}
    )
    play(state, "e3 e4", "e7 e6")
    assert state.action_history == [
        encode_action(square("e3"), square("e4")),
        encode_action(square("e7"), square("e6")),
    ]


def test_capture_ply_is_recorded():
    state = make_position(
        red={"e3": "marshal"}, blue={"e4": "captain", "a10": "flag"}, acting_player=RED
    )
    defender = piece_at(state, "e4")
    play(state, "e3 e4")
    assert defender.capture_ply == 1


def test_supplying_a_precomputed_legal_list_matches_recomputing_it():
    state = make_position(
        red={"e3": "captain", "a1": "flag"}, blue={"e7": "captain", "a10": "flag"}
    )
    other = make_position(
        red={"e3": "captain", "a1": "flag"}, blue={"e7": "captain", "a10": "flag"}
    )
    action = encode_action(square("e3"), square("e4"))

    apply_action(state, action)
    apply_action(other, action, legal=legal_actions(other))

    assert state_fingerprint(state) == state_fingerprint(other)
