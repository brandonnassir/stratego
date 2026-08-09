"""Randomized hidden-identity permutation tests.

Covers `07_observation_validation_matrix.md` section 11,
`09_public_event_and_replay_schema.md` section 16 and
`04_engine_validation_plan.md` section 21.6.

The full 100,000-trial acceptance gate runs in
`scripts/run_phase2_validation.py`; this module runs a smaller seeded sample so
a regression shows up in the ordinary test run.
"""

import random

import numpy as np
import pytest

from stratego.engine.constants import BLUE, IMMOVABLE_TYPES, RED
from stratego.engine.observation import belief_target, build_observation
from stratego.engine.permutation import (
    belief_targets_differ,
    compare_public_surfaces,
    hidden_opponent_piece_ids,
    permutation_is_valid,
    permute_hidden_identities,
    public_surface,
)
from tests.helpers import nonterminal_state

SAMPLE_PLIES = (6, 25, 60, 120, 220)


def test_hidden_pieces_are_exactly_the_unrevealed_living_opponents():
    state = nonterminal_state(80)
    hidden = set(hidden_opponent_piece_ids(state, RED))
    expected = {
        record.piece_id
        for record in state.pieces_of(BLUE)
        if record.alive and not record.known_to(RED)
    }
    assert hidden == expected
    # Every captured piece was revealed by the combat that captured it.
    assert all(record.known_to(RED) for record in state.pieces_of(BLUE) if not record.alive)


def test_permutation_preserves_the_hidden_type_multiset():
    state = nonterminal_state(70)
    rng = random.Random(0)
    permuted, info = permute_hidden_identities(state, RED, rng)
    assert info["valid"]

    hidden = hidden_opponent_piece_ids(state, RED)
    before = sorted(state.pieces[piece_id].true_type for piece_id in hidden)
    after = sorted(permuted.pieces[piece_id].true_type for piece_id in hidden)
    assert before == after


def test_permutation_never_makes_a_moved_piece_immovable():
    state = nonterminal_state(140)
    rng = random.Random(1)
    for _ in range(50):
        permuted, info = permute_hidden_identities(state, RED, rng)
        assert info["valid"]
        for record in permuted.pieces:
            if record.true_type in IMMOVABLE_TYPES:
                assert not record.has_moved


def test_permutation_leaves_known_and_own_pieces_untouched():
    state = nonterminal_state(90)
    permuted, _ = permute_hidden_identities(state, RED, random.Random(2))
    hidden = set(hidden_opponent_piece_ids(state, RED))
    for record, other in zip(state.pieces, permuted.pieces):
        if record.piece_id not in hidden:
            assert record.true_type == other.true_type


@pytest.mark.parametrize("ply", SAMPLE_PLIES)
@pytest.mark.parametrize("observer", [RED, BLUE])
def test_public_surface_is_invariant_under_hidden_permutation(ply, observer):
    state = nonterminal_state(ply)
    baseline = public_surface(state, observer)
    rng = random.Random(ply * 31 + observer)

    for _ in range(40):
        permuted, info = permute_hidden_identities(state, observer, rng)
        assert info["valid"]
        mismatches = compare_public_surfaces(baseline, public_surface(permuted, observer))
        assert sum(mismatches.values()) == 0, mismatches


def test_every_observation_channel_survives_permutation():
    state = nonterminal_state(110)
    baseline = build_observation(state, RED)
    rng = random.Random(7)
    for _ in range(60):
        permuted, _ = permute_hidden_identities(state, RED, rng)
        candidate = build_observation(permuted, RED)
        differing = [
            channel
            for channel in range(127)
            if not np.array_equal(baseline[channel], candidate[channel])
        ]
        assert differing == []


def scripted_behaviour_state(behavior_type):
    """A scripted position holding a live event of the requested type.

    Blue keeps several hidden pieces of different types so the permutation has
    something to shuffle.
    """
    from tests.helpers import make_position, play

    blue_extra = {"i9": "marshal", "j9": "scout", "h9": "miner", "g9": "sergeant"}

    if behavior_type == "threat":
        state = make_position(
            red={"e3": "captain", "a1": "flag"},
            blue={"e5": "colonel", "j10": "flag", **blue_extra},
            acting_player=RED,
        )
        play(state, "e3 e4")
        return state

    if behavior_type == "declined_attack":
        state = make_position(
            red={"e3": "captain", "c3": "miner", "a1": "flag"},
            blue={"e4": "colonel", "j10": "flag", **blue_extra},
            acting_player=RED,
        )
        play(state, "c3 c4")
        return state

    # evade, protect and was_protected all start from a blue threat on e3.
    state = make_position(
        red={"e3": "captain", "c3": "miner", "a1": "flag"},
        blue={"e5": "colonel", "j10": "flag", **blue_extra},
        acting_player=BLUE,
    )
    play(state, "e5 e4")
    play(state, "e3 e2" if behavior_type == "evade" else "c3 d3")
    return state


@pytest.mark.parametrize(
    "behavior_type", ["threat", "evade", "declined_attack", "protect", "was_protected"]
)
def test_behaviour_channels_survive_permutation(behavior_type):
    """Validation-matrix section 8.9, one case per behaviour type."""
    state = scripted_behaviour_state(behavior_type)
    assert any(
        key[1] == behavior_type for key in state.behavior_memory
    ), f"the scripted position did not record a {behavior_type} event"

    block = {"threat": 0, "evade": 1, "declined_attack": 2, "protect": 3, "was_protected": 4}[
        behavior_type
    ]
    channels = [68 + 4 * block + offset for offset in range(4)] + [
        88 + 4 * block + offset for offset in range(4)
    ]

    baseline = build_observation(state, RED)[channels]
    rng = random.Random(11)
    for _ in range(40):
        permuted, _ = permute_hidden_identities(state, RED, rng)
        assert np.array_equal(build_observation(permuted, RED)[channels], baseline)


def test_belief_targets_are_a_positive_control():
    """Privileged targets are expected to change; that is what makes the
    invariance of the public surface meaningful."""
    state = nonterminal_state(100)
    rng = random.Random(13)
    changed_trials = 0
    for _ in range(40):
        permuted, info = permute_hidden_identities(state, RED, rng)
        if info["changed"]:
            changed_trials += 1
            assert belief_targets_differ(state, permuted, RED)
    assert changed_trials > 0


def test_belief_target_squares_and_identifiers_are_unchanged_by_permutation():
    state = nonterminal_state(100)
    permuted, _ = permute_hidden_identities(state, RED, random.Random(17))
    original = belief_target(state, RED)
    other = belief_target(permuted, RED)
    assert [item["piece_id"] for item in original] == [item["piece_id"] for item in other]
    assert [item["square"] for item in original] == [item["square"] for item in other]


def test_permutation_validity_checker_rejects_an_impossible_assignment():
    state = nonterminal_state(150)
    hidden = hidden_opponent_piece_ids(state, RED)
    moved = [piece_id for piece_id in hidden if state.pieces[piece_id].has_moved]
    assert moved, "the sampled position should contain a moved hidden piece"

    types = [state.pieces[piece_id].true_type for piece_id in hidden]
    from stratego.engine.constants import BOMB

    forced = list(types)
    forced[hidden.index(moved[0])] = BOMB
    assert not permutation_is_valid(state, hidden, forced)
