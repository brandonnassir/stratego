"""Privileged belief-learning targets stay separate from the observation.

Covers `03_game_engine_spec.md` section 15, `06_observation_v2_127ch.md`
section 15 and `08_internal_state_spec.md` section 16.
"""

import inspect

from stratego.engine import observation as observation_module
from stratego.engine.constants import BLUE, PIECE_TYPE_NAMES, RED, opponent_of
from stratego.engine.observation import belief_target, build_observation
from tests.helpers import make_position, nonterminal_state, piece_at, play


def test_belief_target_lists_every_hidden_opponent_piece():
    state = nonterminal_state(70)
    targets = belief_target(state, RED)
    expected = {
        record.piece_id
        for record in state.pieces_of(BLUE)
        if record.alive and not record.known_to(RED)
    }
    from stratego.engine.pieces import piece_id_from_name

    assert {piece_id_from_name(item["piece_id"]) for item in targets} == expected


def test_belief_target_carries_square_and_true_type():
    state = make_position(
        red={"e3": "captain", "a1": "flag"}, blue={"e7": "marshal", "j10": "flag"}
    )
    targets = belief_target(state, RED)
    by_square = {item["square"]: item["true_type"] for item in targets}
    assert by_square[piece_at(state, "e7").current_square] == "marshal"
    assert by_square[piece_at(state, "j10").current_square] == "flag"


def test_belief_target_excludes_already_known_pieces():
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e4": "marshal", "j10": "flag"},
        acting_player=RED,
    )
    play(state, "e3 e4")  # reveals the marshal
    targets = belief_target(state, RED)
    assert all(item["true_type"] != "marshal" for item in targets)


def test_belief_target_excludes_own_and_captured_pieces():
    state = nonterminal_state(120)
    targets = belief_target(state, RED)
    from stratego.engine.pieces import piece_id_from_name

    for item in targets:
        record = state.pieces[piece_id_from_name(item["piece_id"])]
        assert record.owner == BLUE
        assert record.alive


def test_belief_target_is_deterministically_ordered():
    state = nonterminal_state(85)
    first = belief_target(state, RED)
    second = belief_target(state, RED)
    assert first == second
    assert [item["piece_id"] for item in first] == sorted(
        item["piece_id"] for item in first
    )


def test_both_observers_receive_their_own_targets():
    state = nonterminal_state(65)
    red_targets = belief_target(state, RED)
    blue_targets = belief_target(state, BLUE)
    assert {item["piece_id"] for item in red_targets}.isdisjoint(
        {item["piece_id"] for item in blue_targets}
    )


def test_observation_builder_does_not_call_the_belief_target_function():
    """A structural guard against the two paths ever being wired together."""
    source = inspect.getsource(observation_module.build_observation)
    assert "belief_target" not in source


def test_observation_and_belief_target_are_separate_products():
    state = nonterminal_state(95)
    observation = build_observation(state, RED)
    targets = belief_target(state, RED)
    assert observation.shape == (127, 10, 10)
    assert isinstance(targets, list)
    # Nothing in the observation encodes a hidden true type: every hidden
    # opponent square is marked only by the generic occupancy plane.
    from stratego.engine.coordinates import to_perspective
    from stratego.engine.observation import (
        CH_HIDDEN_OPPONENT_OCCUPANCY,
        CH_KNOWN_OPPONENT_IDENTITY,
    )

    for item in targets:
        row, column = divmod(to_perspective(item["square"], RED), 10)
        assert observation[CH_HIDDEN_OPPONENT_OCCUPANCY, row, column] == 1.0
        assert (
            observation[
                CH_KNOWN_OPPONENT_IDENTITY : CH_KNOWN_OPPONENT_IDENTITY + 12, row, column
            ].sum()
            == 0.0
        )
        assert item["true_type"] in PIECE_TYPE_NAMES
