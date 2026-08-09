"""Action encoding and legal-list / legal-mask agreement.

Covers `04_engine_validation_plan.md` section 13 and section 7 of the Phase Two
instructions.
"""

import numpy as np
import pytest

from stratego.engine.actions import (
    ACTION_SPACE_SIZE,
    action_from_perspective,
    action_to_perspective,
    decode_action,
    encode_action,
)
from stratego.engine.constants import BLUE, NUM_SQUARES, RED
from stratego.engine.coordinates import to_perspective
from stratego.engine.legal_moves import legal_action_mask, legal_actions
from tests.helpers import nonterminal_state
from tests.helpers import make_position


def test_action_space_is_exactly_ten_thousand():
    assert ACTION_SPACE_SIZE == 10_000
    assert NUM_SQUARES * NUM_SQUARES == 10_000


def test_every_action_identifier_round_trips():
    seen = set()
    for source in range(NUM_SQUARES):
        for destination in range(NUM_SQUARES):
            action = encode_action(source, destination)
            assert action == 100 * source + destination
            assert decode_action(action) == (source, destination)
            seen.add(action)
    assert len(seen) == ACTION_SPACE_SIZE
    assert min(seen) == 0 and max(seen) == ACTION_SPACE_SIZE - 1


def test_encoding_rejects_out_of_range_squares():
    for bad in (-1, 100, 1000):
        with pytest.raises(ValueError):
            encode_action(bad, 0)
        with pytest.raises(ValueError):
            encode_action(0, bad)
    for bad in (-1, ACTION_SPACE_SIZE):
        with pytest.raises(ValueError):
            decode_action(bad)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("ply", [0, 5, 40, 120])
def test_legal_list_and_mask_agree(seed, ply):
    state = nonterminal_state(ply, first_seed=seed)
    actions = legal_actions(state)
    mask = legal_action_mask(state)

    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask.dtype == np.uint8
    assert set(np.flatnonzero(mask).tolist()) == set(actions)
    assert int(mask.sum()) == len(actions)


@pytest.mark.parametrize("seed", range(8))
def test_legal_list_has_no_duplicates_and_is_sorted(seed):
    state = nonterminal_state(33, first_seed=seed)
    actions = legal_actions(state)
    assert actions == sorted(actions)
    assert len(actions) == len(set(actions))


def test_illegal_actions_are_masked():
    state = make_position(
        red={"e3": "captain", "a1": "flag"}, blue={"e7": "captain", "a10": "flag"}
    )
    mask = legal_action_mask(state)
    legal = set(legal_actions(state))
    # Sample the whole space rather than only the legal entries.
    for action in range(0, ACTION_SPACE_SIZE, 37):
        assert bool(mask[action]) == (action in legal)


def test_terminal_state_has_an_all_zero_mask():
    from tests.helpers import play

    state = make_position(
        red={"e3": "scout"}, blue={"e4": "flag", "j10": "captain"}, acting_player=RED
    )
    play(state, "e3 e4")
    assert legal_actions(state) == []
    assert int(legal_action_mask(state).sum()) == 0


@pytest.mark.parametrize("player", [RED, BLUE])
def test_perspective_action_mapping_round_trips(player):
    for source in range(0, NUM_SQUARES, 7):
        for destination in range(0, NUM_SQUARES, 11):
            action = encode_action(source, destination)
            normalized = action_to_perspective(action, player)
            assert action_from_perspective(normalized, player) == action
            assert decode_action(normalized) == (
                to_perspective(source, player),
                to_perspective(destination, player),
            )


def test_normalized_mask_is_a_permutation_of_the_absolute_mask():
    state = nonterminal_state(30, first_seed=5)
    actions = legal_actions(state)
    normalized = {action_to_perspective(action, state.acting_player) for action in actions}
    assert len(normalized) == len(actions)
    assert {
        action_from_perspective(action, state.acting_player) for action in normalized
    } == set(actions)
