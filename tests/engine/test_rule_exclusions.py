"""The two deliberately excluded competitive rules must stay excluded.

Covers `04_engine_validation_plan.md` section 11 and `02_project_ruleset.md`
section 2.
"""

import pytest

from stratego.engine.actions import encode_action
from stratego.engine.constants import (
    RED,
    TERMINAL_BATTLELESS_MOVE_LIMIT_DRAW,
    RulesConfig,
)
from stratego.engine.legal_moves import legal_actions
from tests.helpers import make_position, play, square


def test_rules_configuration_refuses_to_enable_the_excluded_rules():
    with pytest.raises(ValueError, match="deliberately"):
        RulesConfig(two_square_rule_enabled=True)
    with pytest.raises(ValueError, match="deliberately"):
        RulesConfig(continuous_chasing_rule_enabled=True)


def test_default_configurations_have_both_rules_disabled():
    from stratego.engine.constants import EVALUATION_RULES, TRAINING_RULES

    for rules in (TRAINING_RULES, EVALUATION_RULES):
        assert rules.two_square_rule_enabled is False
        assert rules.continuous_chasing_rule_enabled is False


def test_repeated_back_and_forth_movement_stays_legal():
    """The two-square rule is excluded, so A-B-A-B never becomes illegal."""
    rules = RulesConfig(battleless_move_limit=10_000, absolute_move_limit=10_000)
    state = make_position(
        red={"a1": "captain", "j1": "flag"},
        blue={"a10": "captain", "j10": "flag"},
        acting_player=RED,
        rules=rules,
    )
    for repetition in range(20):
        assert encode_action(square("a1"), square("a2")) in legal_actions(state)
        play(state, "a1 a2", "a10 a9")
        assert encode_action(square("a2"), square("a1")) in legal_actions(state)
        play(state, "a2 a1", "a9 a10")
        assert not state.terminal, f"terminated during repetition {repetition}"

    assert state.total_moves == 80


def test_repeated_back_and_forth_only_ends_through_the_draw_counter():
    state = make_position(
        red={"a1": "captain", "j1": "flag"},
        blue={"a10": "captain", "j10": "flag"},
        acting_player=RED,
    )
    while not state.terminal:
        if state.acting_player == RED:
            source, destination = (
                ("a1", "a2") if state.board[square("a1")] is not None else ("a2", "a1")
            )
        else:
            source, destination = (
                ("a10", "a9") if state.board[square("a10")] is not None else ("a9", "a10")
            )
        play(state, f"{source} {destination}")

    assert state.terminal_reason == TERMINAL_BATTLELESS_MOVE_LIMIT_DRAW
    assert state.total_moves == 100


def test_a_repeating_chase_is_never_rejected():
    """The continuous-chasing rule is excluded.

    Red's captain chases blue's captain up and down the a-file. The position
    after every second pair of plies repeats exactly, which the competitive
    chasing rule would forbid.
    """
    rules = RulesConfig(battleless_move_limit=10_000, absolute_move_limit=10_000)
    state = make_position(
        red={"a1": "captain", "j1": "flag"},
        blue={"a3": "captain", "j10": "flag"},
        acting_player=RED,
        rules=rules,
    )
    seen_positions = []
    for _ in range(12):
        # Red steps towards blue; blue retreats; then both step back.
        play(state, "a1 a2")
        assert encode_action(square("a3"), square("a4")) in legal_actions(state)
        play(state, "a3 a4")
        play(state, "a2 a1")
        assert encode_action(square("a4"), square("a3")) in legal_actions(state)
        play(state, "a4 a3")
        seen_positions.append(tuple(state.board))

    # Every recorded position is the same, and the engine never objected.
    assert len(set(seen_positions)) == 1
    assert not state.terminal
