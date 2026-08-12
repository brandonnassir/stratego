"""Masking, ties, precision, non-finite values and the engine's final authority.

Covers Phase 5 gates 11 (`legality_edge_cases_pass`) and 12
(`engine_illegal_action_guard_preserved`).

The single rule under test: **the adapter never knowingly returns an illegal
action, and never repairs a failure into a legal move.** Every case below either
produces the right legal action or raises. None of them silently degrades to a
random or first-legal choice.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from stratego.engine.constants import ACTION_SPACE_SIZE
from stratego.engine.legal_moves import legal_action_mask, legal_actions
from stratego.engine.snapshot import create_snapshot, snapshot_to_json
from stratego.engine.transition import IllegalActionError, apply_action
from stratego.evaluation.policy import PolicyRequirements, build_policy_input
from stratego.model.policy_adapter import (
    NeuralPolicyError,
    categorical_action,
    greedy_action,
    legal_actions_from_mask,
    usable_logits,
    validate_legality,
)

from ..helpers import nonterminal_state
from .conftest import StubOutputPolicy, crafted_policy_logits


def _logits(background: float = 0.0) -> torch.Tensor:
    return torch.full((ACTION_SPACE_SIZE,), background, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def test_the_highest_raw_logit_being_illegal_does_not_matter():
    legal = [5, 77, 1234]
    logits = _logits(-20.0)
    logits[9999] = 500.0  # illegal and enormous
    logits[77] = 1.0
    assert greedy_action(logits, legal) == 77


def test_all_of_the_largest_logits_being_illegal_does_not_matter():
    legal = [10, 20, 30]
    logits = _logits(-1.0)
    for illegal in (1, 2, 3, 4, 5000, 9999):
        logits[illegal] = 100.0 + illegal
    logits[20] = -0.5  # the best of a bad legal bunch
    assert greedy_action(logits, legal) == 20


def test_exactly_one_legal_action_is_always_that_action():
    logits = _logits(0.0)
    logits[4321] = -1e30  # even a terrible score is still the only option
    assert greedy_action(logits, [4321]) == 4321
    assert categorical_action(logits, [4321], random.Random(0))[0] == 4321


def test_tied_legal_maxima_resolve_to_the_lowest_action_identifier():
    legal = [900, 12, 4500, 120]
    logits = _logits(-5.0)
    for action in legal:
        logits[action] = 7.5  # an exact tie across all four
    assert greedy_action(logits, legal) == 12
    # The tie-break is a property of the values, not of the input list order.
    assert greedy_action(logits, list(reversed(legal))) == 12


def test_extreme_finite_logits_are_usable():
    legal = [1, 2, 3]
    logits = _logits(0.0)
    logits[1] = torch.finfo(torch.float32).max
    logits[2] = -torch.finfo(torch.float32).max
    logits[3] = 0.0
    assert greedy_action(logits, legal) == 1
    action, probabilities = categorical_action(logits, legal, random.Random(1))
    assert action in legal
    assert abs(sum(probabilities) - 1.0) < 1e-9


def test_float16_logits_are_accepted_and_compared_correctly():
    legal = [7, 8, 9]
    logits = torch.full((ACTION_SPACE_SIZE,), -1.0, dtype=torch.float16)
    logits[8] = 2.0
    assert usable_logits(logits, legal).dtype == torch.float32
    assert greedy_action(logits, legal) == 8


def test_a_float16_overflow_on_a_legal_action_is_rejected_not_widened():
    legal = [7, 8, 9]
    logits = torch.full((ACTION_SPACE_SIZE,), -1.0, dtype=torch.float16)
    logits[8] = float("inf")  # 70000.0 in float16 lands here too
    with pytest.raises(NeuralPolicyError, match="non-finite"):
        greedy_action(logits, legal)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_logit_on_a_legal_action_raises(bad):
    legal = [11, 22, 33]
    logits = _logits(0.0)
    logits[22] = bad
    with pytest.raises(NeuralPolicyError, match="non-finite"):
        greedy_action(logits, legal)
    with pytest.raises(NeuralPolicyError, match="non-finite"):
        categorical_action(logits, legal, random.Random(0))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_logit_on_an_illegal_action_is_ignored(bad):
    """Illegal entries are never read, so their values cannot matter."""
    legal = [11, 22, 33]
    logits = _logits(0.0)
    logits[9998] = bad
    logits[33] = 1.0
    assert greedy_action(logits, legal) == 33


def test_an_all_negative_infinity_row_is_rejected_rather_than_guessed():
    legal = [1, 2]
    logits = torch.full((ACTION_SPACE_SIZE,), float("-inf"))
    with pytest.raises(NeuralPolicyError):
        greedy_action(logits, legal)


# ---------------------------------------------------------------------------
# Malformed legality products
# ---------------------------------------------------------------------------


def test_an_empty_legality_product_raises():
    with pytest.raises(NeuralPolicyError, match="empty"):
        validate_legality([])
    with pytest.raises(NeuralPolicyError, match="empty"):
        greedy_action(_logits(), [])


def test_an_empty_dense_mask_raises():
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.uint8)
    assert legal_actions_from_mask(mask) == ()
    with pytest.raises(NeuralPolicyError):
        validate_legality([5], mask)


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros(9_999, dtype=np.uint8),  # too short
        np.zeros(10_001, dtype=np.uint8),  # too long
        np.zeros((100, 100), dtype=np.uint8),  # right size, wrong rank
    ],
)
def test_a_malformed_dense_mask_raises(mask):
    with pytest.raises(NeuralPolicyError):
        legal_actions_from_mask(mask)


def test_a_mask_holding_values_other_than_zero_or_one_raises():
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.uint8)
    mask[5] = 2
    with pytest.raises(NeuralPolicyError):
        legal_actions_from_mask(mask)


def test_disagreeing_legality_products_raise():
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.uint8)
    mask[[5, 6, 7]] = 1
    validate_legality([7, 5, 6], mask)  # order does not matter
    with pytest.raises(NeuralPolicyError, match="disagree"):
        validate_legality([5, 6], mask)
    with pytest.raises(NeuralPolicyError, match="disagree"):
        validate_legality([5, 6, 7, 8], mask)


def test_duplicate_and_out_of_range_action_identifiers_raise():
    with pytest.raises(NeuralPolicyError, match="duplicate"):
        validate_legality([4, 4, 5])
    with pytest.raises(NeuralPolicyError, match="outside"):
        validate_legality([10_000])
    with pytest.raises(NeuralPolicyError, match="outside"):
        validate_legality([-1])


def test_a_wrongly_shaped_policy_row_raises():
    with pytest.raises(NeuralPolicyError):
        usable_logits(torch.zeros(9_999), [1])
    with pytest.raises(NeuralPolicyError):
        usable_logits(torch.zeros(4, ACTION_SPACE_SIZE), [1])
    with pytest.raises(NeuralPolicyError, match="floating"):
        usable_logits(torch.zeros(ACTION_SPACE_SIZE, dtype=torch.int32), [1])


# ---------------------------------------------------------------------------
# The Phase 3 sampler regression, kept permanent
# ---------------------------------------------------------------------------


class _FixedDraw(random.Random):
    """A generator whose `random()` returns a chosen value. Nothing else changes."""

    def __init__(self, value: float):
        super().__init__(0)
        self._value = value
        self.draws = 0

    def random(self) -> float:
        self.draws += 1
        return self._value


@pytest.mark.parametrize("draw", [0.0, 1e-300, 0.5, 1.0 - 2**-53, 0.9999999999999999])
def test_the_sampler_stays_inside_the_legal_set_for_every_extreme_draw(draw):
    """The permanent regression from the Phase 3 Gumbel failure.

    Phase 3's Gumbel-max sampler could draw `u == 0`, produce `+inf` noise, add
    it to the `-inf` illegal fill to get `NaN`, and `argmax` ranks `NaN` first --
    so it chose an action the engine had declared illegal. This sampler indexes
    into the legal list instead of arg-maxing over 10,000 entries, so an extreme
    draw cannot name an illegal action at all. These cases pin that.
    """
    legal = [3, 17, 250, 9_999]
    logits = _logits(0.0)
    logits[3] = 30.0  # a near-degenerate distribution: one action holds ~all mass
    rng = _FixedDraw(draw)
    action, probabilities = categorical_action(logits, legal, rng)
    assert action in legal
    assert rng.draws == 1  # exactly one draw per decision
    assert all(np.isfinite(probabilities))


def test_the_phase_3_gumbel_guard_is_still_in_place():
    """The Phase 3 fix itself must not regress while Phase 5 depends on the story."""
    from stratego.training.representative_model import _gumbel_noise

    noise = _gumbel_noise((4096, 512), torch.device("cpu"), None)
    assert bool(torch.isfinite(noise).all())


def test_underflowing_probabilities_do_not_produce_an_illegal_choice():
    """One action dominates so hard that the others underflow to exactly zero."""
    legal = [1, 2, 3]
    logits = _logits(0.0)
    logits[1] = 0.0
    logits[2] = -1e30
    logits[3] = -1e30
    action, probabilities = categorical_action(logits, legal, random.Random(5))
    assert action == 1
    assert probabilities[1] == 0.0 and probabilities[2] == 0.0


def test_the_sampler_draws_exactly_once_per_decision():
    legal = list(range(0, 500, 7))
    logits = torch.randn(ACTION_SPACE_SIZE, generator=torch.Generator().manual_seed(3))
    rng = _FixedDraw(0.73)
    categorical_action(logits, legal, rng)
    assert rng.draws == 1


def test_the_same_seed_gives_the_same_sample_and_different_seeds_spread():
    legal = list(range(0, 1000, 3))
    logits = torch.randn(ACTION_SPACE_SIZE, generator=torch.Generator().manual_seed(11))
    first = categorical_action(logits, legal, random.Random(99))[0]
    again = categorical_action(logits, legal, random.Random(99))[0]
    assert first == again
    spread = {categorical_action(logits, legal, random.Random(seed))[0] for seed in range(60)}
    assert len(spread) > 1  # it is genuinely stochastic, not a constant
    assert spread <= set(legal)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal is not available on this machine"
)
def test_the_sampler_runs_on_metal_logits():
    """Regression, Phase 6 Agent 5.

    The cumulative sum is walked in float64 for exactness, and Metal has no
    float64 dtype, so `.to(torch.float64)` on an MPS tensor raises `TypeError`.
    Phase 5 only ever sampled from a CPU model, so the seeded-categorical policy
    could not run on the device it was meant to run on. The widening now names
    the CPU explicitly; this test is what stops it from drifting back.
    """
    legal = list(range(0, 1000, 3))
    logits = torch.randn(ACTION_SPACE_SIZE, generator=torch.Generator().manual_seed(11))
    on_metal = logits.to("mps")
    action, probabilities = categorical_action(on_metal, legal, random.Random(99))
    assert action in legal
    assert abs(sum(probabilities) - 1.0) < 1e-9
    # Same weights, same seed, same draw: the device must not change the choice.
    assert action == categorical_action(logits, legal, random.Random(99))[0]


# ---------------------------------------------------------------------------
# The engine remains the final authority
# ---------------------------------------------------------------------------


def test_the_engine_rejects_an_illegal_action_loudly_and_stays_inert():
    """If an illegal action ever reached the engine, it must raise and change nothing."""
    state = nonterminal_state(40)
    legal = set(legal_actions(state))
    illegal = next(action for action in range(ACTION_SPACE_SIZE) if action not in legal)
    before = snapshot_to_json(create_snapshot(state, include_history=True))

    with pytest.raises(IllegalActionError):
        apply_action(state, illegal)

    after = snapshot_to_json(create_snapshot(state, include_history=True))
    assert before == after  # inert: no piece moved, no ply counted, no event emitted


def test_a_policy_that_returns_an_illegal_action_is_caught_by_the_contract(model):
    """The adapter cannot do this, so the guard is proven with a stub that does."""
    state = nonterminal_state(40)
    legal = legal_actions(state)
    illegal = next(action for action in range(ACTION_SPACE_SIZE) if action not in set(legal))

    policy = StubOutputPolicy(model, crafted_policy_logits(0), mode="greedy")
    request = build_policy_input(
        state,
        policy=policy.ref,
        policy_seed=3,
        requirements=PolicyRequirements(observation=True, legal_action_mask=True),
    )
    from stratego.evaluation.policy import PolicyContractError, validate_policy_result

    forged = policy.result(request, illegal)
    with pytest.raises(PolicyContractError, match="illegal"):
        validate_policy_result(forged, request)


def test_the_adapter_never_substitutes_a_move_when_the_model_fails(model):
    """A broken model produces an exception, not a quietly reasonable move."""
    state = nonterminal_state(40)
    broken = torch.full((ACTION_SPACE_SIZE,), float("nan"))
    policy = StubOutputPolicy(model, broken, mode="greedy")
    request = build_policy_input(
        state,
        policy=policy.ref,
        policy_seed=3,
        requirements=PolicyRequirements(observation=True, legal_action_mask=True),
    )
    with pytest.raises(NeuralPolicyError):
        policy.decide(request)


def test_a_decision_matches_the_engine_mask_at_every_ply_of_a_short_game(greedy_policy):
    """A hundred real decisions, each checked against the engine's own mask."""
    state = nonterminal_state(20)
    for _ in range(100):
        if state.terminal:
            break
        legal = legal_actions(state)
        mask = legal_action_mask(state, legal)
        request = build_policy_input(
            state,
            policy=greedy_policy.ref,
            policy_seed=17,
            requirements=greedy_policy.requirements,
            legal=legal,
        )
        chosen = greedy_policy.decide_checked(request).selected_action_id
        assert mask[chosen] == 1
        apply_action(state, chosen, legal=legal)
