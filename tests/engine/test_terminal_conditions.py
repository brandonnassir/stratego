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
from stratego.engine.legal_moves import (
    generate_actions_for_player,
    has_legal_action,
)
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


# ---------------------------------------------------------------------------
# Terminal-condition precedence (`02_project_ruleset.md` section 9A)
#
#   1. flag_capture
#   2. opponent_no_legal_move
#   3. both_no_legal_move_draw
#   4. battleless_move_limit_draw
#   5. absolute_move_limit_draw
# ---------------------------------------------------------------------------

# Blue owns only a Flag and a Bomb, so blue never has a legal move.
IMMOBILE_BLUE = {"a10": "flag", "b10": "bomb"}


def test_no_legal_move_victory_outranks_the_battleless_draw():
    """Level 2 beats level 4 when one move satisfies both."""
    state = make_position(
        red={"a1": "captain", "j1": "flag"},
        blue=dict(IMMOBILE_BLUE),
        acting_player=RED,
        rules=TRAINING_RULES,
        battleless_moves=TRAINING_RULES.battleless_move_limit - 1,
    )
    assert not has_legal_action(state, BLUE)

    play(state, "a1 a2")

    # Both conditions really are satisfied by this single move.
    assert state.battleless_moves == TRAINING_RULES.battleless_move_limit
    assert not has_legal_action(state, BLUE)
    # Precedence resolves it as a win, not a draw.
    assert state.terminal_reason == TERMINAL_OPPONENT_NO_LEGAL_MOVE
    assert state.winner == RED
    assert not state.is_draw


def test_no_legal_move_victory_outranks_the_absolute_move_limit_draw():
    """Level 2 beats level 5 when one move satisfies both."""
    rules = RulesConfig(battleless_move_limit=10_000, absolute_move_limit=12)
    state = make_position(
        red={"a1": "captain", "j1": "flag"},
        blue=dict(IMMOBILE_BLUE),
        acting_player=RED,
        rules=rules,
        total_moves=rules.absolute_move_limit - 1,
    )

    play(state, "a1 a2")

    assert state.total_moves == rules.absolute_move_limit
    assert not has_legal_action(state, BLUE)
    assert state.terminal_reason == TERMINAL_OPPONENT_NO_LEGAL_MOVE
    assert state.winner == RED


def test_mutual_stalemate_outranks_the_absolute_move_limit_draw():
    """Level 3 beats level 5; both are draws, but the reason must be correct."""
    rules = RulesConfig(battleless_move_limit=10_000, absolute_move_limit=12)
    state = make_position(
        red={"a1": "flag", "a3": "captain"},
        blue={"j10": "flag", "a4": "bomb", "j9": "bomb"},
        acting_player=RED,
        rules=rules,
        total_moves=rules.absolute_move_limit - 1,
    )

    play(state, "a3 a4")  # the captain dies on the bomb, stranding both sides

    assert state.total_moves == rules.absolute_move_limit
    assert not has_legal_action(state, RED)
    assert not has_legal_action(state, BLUE)
    assert state.terminal_reason == TERMINAL_BOTH_NO_LEGAL_MOVE_DRAW
    assert state.is_draw


def test_flag_capture_outranks_every_other_condition():
    """Level 1 beats levels 2 and 5 simultaneously."""
    rules = RulesConfig(battleless_move_limit=10_000, absolute_move_limit=12)
    state = make_position(
        red={"e3": "scout", "a1": "bomb"},
        blue={"e4": "flag", "j10": "bomb"},
        acting_player=RED,
        rules=rules,
        total_moves=rules.absolute_move_limit - 1,
    )

    play(state, "e3 e4")

    assert state.total_moves == rules.absolute_move_limit
    assert not has_legal_action(state, BLUE)
    assert state.terminal_reason == TERMINAL_FLAG_CAPTURE
    assert state.winner == RED


def test_draw_limits_still_apply_when_both_players_can_move():
    """Levels 4 and 5 are unaffected when the mobility question does not arise."""
    state = shuffle_position(TRAINING_RULES)
    shuffle(state, 100)
    assert state.terminal_reason == TERMINAL_BATTLELESS_MOVE_LIMIT_DRAW

    rules = RulesConfig(battleless_move_limit=10_000, absolute_move_limit=12)
    state = shuffle_position(rules)
    shuffle(state, 12)
    assert state.terminal_reason == TERMINAL_ABSOLUTE_MOVE_LIMIT_DRAW


def test_battleless_limit_can_never_coincide_with_a_capture_or_stalemate():
    """The two remaining precedence collisions are structurally unreachable.

    `02_project_ruleset.md` section 9A records both consequences:

    - any combat resets the no-battle counter to zero, so a capture and
      `battleless_move_limit_draw` cannot occur on the same move;
    - a player cannot strand itself with a non-combat move, because the square
      it just vacated is always available to move back into, so
      `both_no_legal_move_draw` always follows a combat move and therefore also
      cannot coincide with the battleless limit.
    """
    # Combat resets the counter even when it was one short of the threshold.
    state = make_position(
        red={"e3": "marshal", "a1": "flag"},
        blue={"e4": "captain", "j10": "flag", "j9": "scout"},
        acting_player=RED,
        rules=TRAINING_RULES,
        battleless_moves=TRAINING_RULES.battleless_move_limit - 1,
    )
    play(state, "e3 e4")
    assert state.battleless_moves == 0
    assert not state.terminal

    # A non-combat move always leaves the mover able to step back, so the mover
    # can never strand itself.
    state = make_position(
        red={"a1": "captain", "j1": "flag"},
        blue={"a10": "captain", "j10": "flag"},
        acting_player=RED,
        rules=TRAINING_RULES,
    )
    play(state, "a1 a2")
    assert has_legal_action(state, RED)
    destinations = {
        action % 100 for action in generate_actions_for_player(state, RED)
    }
    assert square("a1") in destinations
