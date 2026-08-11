"""Model-level hidden-information safety.

Covers Phase 5 gates 13 (`no_privileged_input_reachable`) and the fast slice of
14 (`hidden_information_10000_zero_mismatch`).

The trial
---------
Take a real position, permute the true types of the opponent pieces the acting
player cannot legally know, and require that **everything the model sees and
everything the model says** is identical, while the privileged belief target --
the positive control -- differs. If any model output moved, the model read
something it was not entitled to.

The full 10,000-trial audit runs in `scripts/run_phase5.py`. This module runs a
smaller sample on every test run, plus the structural audits that do not need a
sample at all.
"""

from __future__ import annotations

import gc
import random

import numpy as np
import pytest
import torch

from stratego.engine.legal_moves import legal_action_mask, legal_actions
from stratego.engine.observation import build_observation
from stratego.engine.permutation import permute_hidden_identities
from stratego.engine.state import GameState
from stratego.evaluation.policy import PolicyInput, PolicyRequirements, build_policy_input
from stratego.model.tokenization import tokenize_numpy_observation
from stratego.training.belief_targets import dense_belief_target

from ..helpers import nonterminal_state

#: Plies the fast sample draws from. The full audit in the script uses more.
SAMPLE_PLIES = (15, 30, 55, 85, 125)
SAMPLE_TRIALS_PER_PLY = 24


def _trial_states(ply: int, seed: int):
    """A position and a hidden-identity permutation of it, or `None` if unusable."""
    state = nonterminal_state(ply)
    observer = state.acting_player
    twin, info = permute_hidden_identities(state, observer, random.Random(seed))
    if not info["valid"] or not info["changed"]:
        return None
    return state, twin, observer, info


def test_a_sample_of_permutation_trials_moves_nothing_the_model_can_see(model, greedy_policy):
    trials = 0
    skipped = 0
    mismatches = {
        "observation": 0,
        "legal_actions": 0,
        "policy_logits": 0,
        "value_logits": 0,
        "belief_logits": 0,
        "greedy_action": 0,
        "positive_control": 0,
    }

    for ply in SAMPLE_PLIES:
        for offset in range(SAMPLE_TRIALS_PER_PLY):
            trial = _trial_states(ply, seed=1000 * ply + offset)
            if trial is None:
                skipped += 1
                continue
            state, twin, observer, info = trial
            trials += 1

            original_observation = build_observation(state, observer)
            twin_observation = build_observation(twin, observer)
            if not np.array_equal(original_observation, twin_observation):
                mismatches["observation"] += 1
            if legal_actions(state) != legal_actions(twin):
                mismatches["legal_actions"] += 1

            with torch.no_grad():
                first = model(tokenize_numpy_observation(original_observation))
                second = model(tokenize_numpy_observation(twin_observation))
            if not torch.equal(first.policy_logits, second.policy_logits):
                mismatches["policy_logits"] += 1
            if not torch.equal(first.value_logits, second.value_logits):
                mismatches["value_logits"] += 1
            if not torch.equal(first.belief_logits, second.belief_logits):
                mismatches["belief_logits"] += 1

            first_decision = greedy_policy.decide_checked(
                build_policy_input(
                    state,
                    policy=greedy_policy.ref,
                    policy_seed=5,
                    requirements=greedy_policy.requirements,
                )
            )
            second_decision = greedy_policy.decide_checked(
                build_policy_input(
                    twin,
                    policy=greedy_policy.ref,
                    policy_seed=5,
                    requirements=greedy_policy.requirements,
                )
            )
            if first_decision.selected_action_id != second_decision.selected_action_id:
                mismatches["greedy_action"] += 1
            if first_decision.diagnostics != second_decision.diagnostics:
                mismatches["greedy_action"] += 1

            # Positive control: the privileged truth *must* have changed, or the
            # trial proved nothing.
            original_labels, _ = dense_belief_target(state, observer)
            twin_labels, _ = dense_belief_target(twin, observer)
            if np.array_equal(original_labels, twin_labels):
                mismatches["positive_control"] += 1
            assert info["hidden_pieces"] >= 2

    assert trials >= 80, f"only {trials} usable trials ({skipped} skipped)"
    assert mismatches == dict.fromkeys(mismatches, 0)


def test_the_underlying_hidden_types_really_did_change():
    """Guards the guard: a permutation that changed nothing would prove nothing."""
    trial = _trial_states(45, seed=7)
    assert trial is not None
    state, twin, observer, _ = trial
    original = [
        record.true_type
        for record in state.pieces
        if record.owner != observer and record.alive and not record.known_to(observer)
    ]
    permuted = [
        record.true_type
        for record in twin.pieces
        if record.owner != observer and record.alive and not record.known_to(observer)
    ]
    assert original != permuted
    assert sorted(original) == sorted(permuted)  # a permutation, not a substitution


# ---------------------------------------------------------------------------
# Structural audit: nothing privileged is reachable at all
# ---------------------------------------------------------------------------


def test_the_policy_only_requests_observer_safe_products(greedy_policy, sampling_policy):
    for policy in (greedy_policy, sampling_policy):
        requirements = policy.requirements
        assert requirements.observation is True
        assert requirements.legal_action_mask is True
        assert requirements.public_view is False
        assert requirements.public_events is False
        assert requirements.public_setup is False


def test_the_policy_input_carries_nothing_privileged(greedy_policy):
    state = nonterminal_state(50)
    request = build_policy_input(
        state,
        policy=greedy_policy.ref,
        policy_seed=1,
        requirements=greedy_policy.requirements,
    )
    assert isinstance(request, PolicyInput)
    assert request.public_view is None
    assert request.public_events is None
    assert request.public_setup is None
    for name in vars(request):
        value = getattr(request, name)
        assert not isinstance(value, GameState)


def test_no_game_state_is_reachable_from_the_policy_object_graph(greedy_policy):
    """Walk the adapter's object graph and prove no privileged object is in it."""
    seen: set[int] = set()
    frontier = [greedy_policy]
    privileged_found = []

    while frontier:
        item = frontier.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, GameState):
            privileged_found.append(item)
            continue
        if type(item).__name__ == "PieceRecord":
            privileged_found.append(item)
            continue
        if isinstance(item, (str, bytes, int, float, bool, type(None), np.ndarray)):
            continue
        if isinstance(item, torch.Tensor):
            continue
        if isinstance(item, dict):
            frontier.extend(list(item.keys()) + list(item.values()))
            continue
        if isinstance(item, (list, tuple, set, frozenset)):
            frontier.extend(item)
            continue
        attributes = getattr(item, "__dict__", None)
        if attributes:
            frontier.extend(attributes.values())
        if len(seen) > 20_000:  # pragma: no cover - the graph is far smaller
            pytest.fail("object graph unexpectedly large; the audit would be unreliable")

    assert privileged_found == []


def test_the_observation_handed_to_the_model_is_read_only(greedy_policy):
    state = nonterminal_state(40)
    request = build_policy_input(
        state,
        policy=greedy_policy.ref,
        policy_seed=1,
        requirements=greedy_policy.requirements,
    )
    assert request.observation.flags.writeable is False
    assert request.legal_action_mask.flags.writeable is False
    # And a decision does not mutate what it was given.
    before = request.observation.copy()
    greedy_policy.decide_checked(request)
    assert np.array_equal(request.observation, before)


def test_a_decision_reads_only_the_declared_products(greedy_policy):
    """A request built without the observation must fail loudly, not silently."""
    state = nonterminal_state(40)
    request = build_policy_input(
        state,
        policy=greedy_policy.ref,
        policy_seed=1,
        requirements=PolicyRequirements(observation=False, legal_action_mask=True),
    )
    from stratego.evaluation.policy import PolicyContractError

    with pytest.raises(PolicyContractError):
        greedy_policy.decide(request)


def test_the_legality_mask_is_the_engines_own_product(greedy_policy):
    state = nonterminal_state(60)
    actions = legal_actions(state)
    request = build_policy_input(
        state,
        policy=greedy_policy.ref,
        policy_seed=1,
        requirements=greedy_policy.requirements,
        legal=actions,
    )
    assert np.array_equal(request.legal_action_mask, legal_action_mask(state, actions))


def test_garbage_collection_finds_no_state_referenced_by_the_model(model):
    """A last check that no engine state is pinned by a model attribute."""
    referents = gc.get_referents(model.__dict__)
    assert not any(isinstance(item, GameState) for item in referents)
