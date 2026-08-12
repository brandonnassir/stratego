"""Colour symmetry: the property `model_contract_v2` exists to buy.

Covers the Phase 6 Agent 1 gate `symmetry_regression`.

The instrument is the mirrored-game pair already accepted in Phase 2
(`tests/observation/test_perspective.py`): a game and its colour-swapped,
180-degree-rotated twin, advanced through mirrored actions. Reusing it rather
than hand-building a rotated `GameState` matters -- the twin here is a genuinely
reachable position produced by the frozen engine, so nothing in this file
depends on a test fixture's idea of what a legal state looks like.

What v2 changes
---------------
Under v1 the two equivalent roles already received identical *observations*;
what they did not share was the action space, so the same network reading the
same normalized board still had to pick out of two differently-numbered move
lists. :func:`test_the_v1_absolute_frame_would_have_broken_the_symmetry` is the
negative control for exactly that, and it is the reason this file can claim the
migration achieved something rather than merely preserved something.
"""

from __future__ import annotations

import numpy as np
import pytest

from stratego.engine.constants import BLUE, RED
from stratego.engine.legal_moves import legal_action_mask, legal_actions
from stratego.engine.observation import build_observation
from stratego.evaluation.policy import build_policy_input
from stratego.model.action_frame import (
    absolute_legal_actions_to_model,
    absolute_legal_mask_to_model,
    model_action_to_absolute,
)
from stratego.model.policy_adapter import greedy_action
from stratego.model.tokenization import (
    observation_batch_from_numpy,
    tokenize_numpy_observation,
)

from tests.observation.test_perspective import (
    mirror_action,
    mirrored_games,
    play_mirrored,
)

#: Plies at which the pair is compared. Includes 0 (both setups untouched) and
#: several mid-game points, so the comparison covers positions with revealed
#: pieces, behavioural memory and a populated recent-move window.
PLIES = (0, 1, 6, 25, 60)


def mirrored_pair(seed: int, plies: int, attempts: int = 40):
    """A non-terminal mirrored game pair advanced `plies` plies.

    The twin's first player is blue, so after the same number of plies the two
    games have the same ply count and *opposite* acting colours -- which is the
    correspondence the whole file is about.

    Random play sometimes ends a game before the requested ply. Rather than
    skipping such a case -- which would silently drop a comparison -- the search
    walks to the next seed, so every parameter combination contributes a real
    position. Failing to find one is an error, not a quiet pass.
    """
    for offset in range(attempts):
        candidate_seed = seed + 1000 * offset
        original, twin = mirrored_games(candidate_seed)
        play_mirrored(original, twin, plies, candidate_seed)
        if not original.terminal and not twin.terminal:
            return original, twin
    raise AssertionError(
        f"no non-terminal mirrored pair at ply {plies} within {attempts} seeds from {seed}"
    )


def _assert_roles_correspond(original, twin) -> None:
    """Guard the instrument: equal ply counts, opposite acting colours."""
    assert original.total_moves == twin.total_moves
    assert original.battleless_moves == twin.battleless_moves
    assert {original.acting_player, twin.acting_player} == {RED, BLUE}
    assert not original.terminal and not twin.terminal


# ---------------------------------------------------------------------------
# Identical inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("plies", PLIES)
def test_equivalent_roles_receive_identical_normalized_observations(seed, plies):
    original, twin = mirrored_pair(seed, plies)
    _assert_roles_correspond(original, twin)

    first = build_observation(original, original.acting_player)
    second = build_observation(twin, twin.acting_player)
    assert np.array_equal(first, second)
    # And identically after tokenization, since that is what the model reads.
    assert np.array_equal(
        tokenize_numpy_observation(first).numpy(), tokenize_numpy_observation(second).numpy()
    )


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("plies", PLIES)
def test_equivalent_roles_receive_identical_normalized_legal_masks(seed, plies):
    original, twin = mirrored_pair(seed, plies)
    _assert_roles_correspond(original, twin)

    first = absolute_legal_mask_to_model(
        legal_action_mask(original), original.acting_player
    )
    second = absolute_legal_mask_to_model(legal_action_mask(twin), twin.acting_player)
    assert np.array_equal(first, second)

    assert absolute_legal_actions_to_model(
        legal_actions(original), original.acting_player
    ) == absolute_legal_actions_to_model(legal_actions(twin), twin.acting_player)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("plies", PLIES)
def test_the_absolute_legal_sets_are_mirror_images_but_not_equal(seed, plies):
    """The engine frames genuinely differ; only the model frame collapses them."""
    original, twin = mirrored_pair(seed, plies)
    first = set(legal_actions(original))
    second = set(legal_actions(twin))
    assert {mirror_action(action) for action in first} == second
    assert first != second


# ---------------------------------------------------------------------------
# Identical decisions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("plies", PLIES)
def test_the_same_network_picks_the_same_normalized_move_in_both_positions(
    seed, plies, model
):
    """One deterministic model, two mirrored roles, one strategic decision.

    The absolute actions differ (they are mirror images); the *normalized*
    action, which is what the network actually emitted, is identical.
    """
    original, twin = mirrored_pair(seed, plies)
    _assert_roles_correspond(original, twin)

    chosen = []
    for state in (original, twin):
        player = state.acting_player
        observation = build_observation(state, player)
        logits = model.forward_observation(
            observation_batch_from_numpy(observation)
        ).policy_logits[0]
        legal = absolute_legal_actions_to_model(legal_actions(state), player)
        chosen.append((greedy_action(logits, legal), player))

    (first_action, first_player), (second_action, second_player) = chosen
    assert first_action == second_action

    first_absolute = model_action_to_absolute(first_action, first_player)
    second_absolute = model_action_to_absolute(second_action, second_player)
    assert mirror_action(first_absolute) == second_absolute
    assert first_absolute in legal_actions(original)
    assert second_absolute in legal_actions(twin)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("plies", (0, 6, 25))
def test_the_policy_adapter_makes_mirrored_decisions_end_to_end(seed, plies, greedy_policy):
    """The same claim, through the real `decide` path and its validation."""
    original, twin = mirrored_pair(seed, plies)
    _assert_roles_correspond(original, twin)

    selected = []
    for state in (original, twin):
        request = build_policy_input(
            state,
            policy=greedy_policy.ref,
            policy_seed=4242,
            requirements=greedy_policy.requirements,
        )
        result = greedy_policy.decide_checked(request)
        selected.append(result)

    first, second = selected
    # Same normalized decision, mirrored absolute moves.
    assert first.diagnostics["model_action_id"] == second.diagnostics["model_action_id"]
    assert mirror_action(first.selected_action_id) == second.selected_action_id
    assert first.selected_action_id != second.selected_action_id


# ---------------------------------------------------------------------------
# Negative control
# ---------------------------------------------------------------------------


def test_the_v1_absolute_frame_would_have_broken_the_symmetry(model):
    """The migration bought something: selecting in absolute squares does not.

    Replays the retired v1 selection rule -- identical logits, but chosen over
    the engine's *absolute* legal identifiers -- and requires that it disagrees.
    Without this, every assertion above would still pass on an implementation
    that never converted anything, because both frames coincide for red.
    """
    disagreements = 0
    compared = 0
    for seed in range(6):
        for plies in (0, 6, 25):
            original, twin = mirrored_pair(seed, plies)
            picks = []
            for state in (original, twin):
                observation = build_observation(state, state.acting_player)
                logits = model.forward_observation(
                    observation_batch_from_numpy(observation)
                ).policy_logits[0]
                # v1: the model's 10,000 outputs indexed in absolute squares.
                picks.append(greedy_action(logits, legal_actions(state)))
            compared += 1
            if mirror_action(picks[0]) != picks[1]:
                disagreements += 1

    assert compared > 0
    assert disagreements > 0, (
        "the v1 absolute-frame rule agreed with the mirror everywhere, so this "
        "control proves nothing about the v2 migration"
    )
