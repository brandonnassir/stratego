"""Win, loss and draw conditions, and the battleless counter.

Covers `04_engine_validation_plan.md` sections 9 and 10, and sections 10 and 11
of the Phase Two instructions.
"""

import pytest

from stratego.engine.constants import (
    BLUE,
    EVALUATION_RULES,
    NOT_TERMINAL,
    RED,
    TERMINAL_ABSOLUTE_MOVE_LIMIT_DRAW,
    TERMINAL_BATTLELESS_MOVE_LIMIT_DRAW,
    TERMINAL_BOTH_NO_LEGAL_MOVE_DRAW,
    TERMINAL_FLAG_CAPTURE,
    TERMINAL_OPPONENT_NO_LEGAL_MOVE,
    TERMINAL_REASONS,
    RulesConfig,
    TRAINING_RULES,
)
from stratego.engine.legal_moves import has_legal_action
from tests.helpers import make_position, play, square


def test_terminal_reason_labels_are_exactly_the_documented_set():
    assert set(TERMINAL_REASONS) == {
        "flag_capture",
        "opponent_no_legal_move",
        "both_no_legal_move_draw",
        "battleless_move_limit_draw",
        "absolute_move_limit_draw",
        "not_terminal",
    }


def test_fresh_game_is_not_terminal():
    state = make_position(
        red={"e3": "captain", "a1": "flag"}, blue={"e7": "captain", "a10": "flag"}
    )
    assert not state.terminal
    assert state.terminal_reason == NOT_TERMINAL
    assert state.winner is None


def test_red_flag_capture():
    state = make_position(
        red={"e3": "scout", "a1": "flag"},
        blue={"e4": "flag", "j10": "bomb"},
        acting_player=RED,
    )
    play(state, "e3 e4")
    assert state.terminal_reason == TERMINAL_FLAG_CAPTURE
    assert state.winner == RED
    assert state.effective_score_for(RED) == 1.0
    assert state.effective_score_for(BLUE) == 0.0


def test_blue_flag_capture():
    state = make_position(
        red={"e3": "flag", "a1": "bomb"},
        blue={"e4": "scout", "j10": "flag"},
        acting_player=BLUE,
    )
    play(state, "e4 e3")
    assert state.terminal_reason == TERMINAL_FLAG_CAPTURE
    assert state.winner == BLUE


def test_opponent_left_with_no_legal_move_loses():
    # After blue's move red owns only a flag and bombs, so red cannot move.
    state = make_position(
        red={"a1": "flag", "a2": "bomb", "b1": "bomb"},
        blue={"j10": "flag", "j9": "captain"},
        acting_player=BLUE,
    )
    assert not has_legal_action(state, RED)
    play(state, "j9 i9")
    assert state.terminal_reason == TERMINAL_OPPONENT_NO_LEGAL_MOVE
    assert state.winner == BLUE
    assert state.result_for(BLUE) == 1.0
    assert state.result_for(RED) == -1.0


def test_both_players_without_a_legal_move_is_a_draw():
    # Red's only movable piece dies attacking a bomb; afterwards neither side
    # owns a movable piece.
    state = make_position(
        red={"a1": "flag", "a3": "captain"},
        blue={"j10": "flag", "a4": "bomb", "j9": "bomb"},
        acting_player=RED,
    )
    play(state, "a3 a4")
    assert state.terminal_reason == TERMINAL_BOTH_NO_LEGAL_MOVE_DRAW
    assert state.winner is None
    assert state.is_draw
    assert state.result_for(RED) == 0.0
    assert state.effective_score_for(RED) == 0.5


def shuffle_position(rules=TRAINING_RULES, battleless_moves=0):
    """Two lone pieces that can shuffle back and forth without ever fighting."""
    return make_position(
        red={"a1": "captain", "j1": "flag"},
        blue={"a10": "captain", "j10": "flag"},
        acting_player=RED,
        rules=rules,
        battleless_moves=battleless_moves,
    )


def shuffle(state, count):
    """Play `count` battleless plies as an A-B-A-B shuffle.

    The move is derived from the current board rather than from a counter, so
    the helper can be called repeatedly on the same state.
    """
    for _ in range(count):
        if state.terminal:
            return
        if state.acting_player == RED:
            move = "a1 a2" if state.board[square("a1")] is not None else "a2 a1"
        else:
            move = "a10 a9" if state.board[square("a10")] is not None else "a9 a10"
        play(state, move)


def test_training_draw_at_one_hundred_battleless_moves():
    state = shuffle_position(TRAINING_RULES)
    shuffle(state, 99)
    assert state.battleless_moves == 99
    assert not state.terminal

    shuffle(state, 1)
    assert state.battleless_moves == 100
    assert state.terminal_reason == TERMINAL_BATTLELESS_MOVE_LIMIT_DRAW
    assert state.is_draw


def test_evaluation_draw_at_two_hundred_battleless_moves():
    state = shuffle_position(EVALUATION_RULES)
    shuffle(state, 199)
    assert not state.terminal
    shuffle(state, 1)
    assert state.battleless_moves == 200
    assert state.terminal_reason == TERMINAL_BATTLELESS_MOVE_LIMIT_DRAW


def test_battleless_counter_increments_on_every_non_combat_move():
    state = shuffle_position()
    for expected in range(1, 11):
        shuffle(state, 1)
        assert state.battleless_moves == expected


def test_combat_resets_the_battleless_counter_then_it_climbs_again():
    state = make_position(
        red={"e3": "marshal", "a1": "flag"},
        blue={"e4": "captain", "j10": "flag", "j9": "scout"},
        acting_player=RED,
        battleless_moves=55,
    )
    play(state, "e3 e4")
    assert state.battleless_moves == 0
    play(state, "j9 i9")
    assert state.battleless_moves == 1


def test_absolute_move_limit_draw():
    rules = RulesConfig(battleless_move_limit=10_000, absolute_move_limit=12)
    state = shuffle_position(rules)
    shuffle(state, 11)
    assert not state.terminal
    shuffle(state, 1)
    assert state.total_moves == 12
    assert state.terminal_reason == TERMINAL_ABSOLUTE_MOVE_LIMIT_DRAW
    assert state.is_draw


def test_result_requires_a_terminal_state():
    state = shuffle_position()
    with pytest.raises(ValueError):
        state.result_for(RED)
