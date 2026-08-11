"""CPU/Metal, precision and batch equivalence.

Covers Phase 5 gates 19 (`cpu_mps_float32_equivalence_pass`), 20
(`mps_float16_finite_and_equivalent`) and 21 (`batch_equivalence_pass`).

Predeclared tolerances
----------------------
========================  ==========  ==========
Comparison                atol        rtol
========================  ==========  ==========
float32 across devices    1e-4        1e-4
float16                   5e-2        5e-2
========================  ==========  ==========

These are the Phase 5 instruction's starting policy, used unchanged. They are
declared here and in `scripts/run_phase5.py`, which records the *measured*
maximum errors separately for policy logits, value probabilities and belief
logits so a later phase can tighten them against data rather than taste.

Greedy agreement is asserted exactly only on **crafted-margin** examples, where
one legal action leads by far more than any tolerance. Natural-corpus agreement
is measured and reported instead of asserted exactly, because an untrained
network produces near-ties whose ordering a 1e-7 kernel difference can flip --
hiding that behind a loose tolerance would be the dishonest version.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stratego.engine.observation import build_observation
from stratego.engine.legal_moves import legal_actions
from stratego.model.contract import value_probabilities
from stratego.model.integration_model import build_integration_model
from stratego.model.policy_adapter import greedy_action
from stratego.model.tokenization import observation_to_tokens, tokenize_numpy_observation

from ..helpers import nonterminal_state
from .conftest import TEST_SEED, deterministic_observation

FLOAT32_TOLERANCE = {"atol": 1e-4, "rtol": 1e-4}
FLOAT16_TOLERANCE = {"atol": 5e-2, "rtol": 5e-2}

mps_available = torch.backends.mps.is_available()
requires_mps = pytest.mark.skipif(not mps_available, reason="MPS is not available on this machine")


def _corpus_tokens(count: int = 8) -> torch.Tensor:
    """Real engine observations, tokenized once and reused across devices."""
    plies = (12, 24, 36, 48, 60, 72, 84, 96)[:count]
    observations = [
        build_observation(state, state.acting_player)
        for state in (nonterminal_state(ply) for ply in plies)
    ]
    return tokenize_numpy_observation(observations)


def _max_errors(reference: torch.Tensor, other: torch.Tensor) -> tuple[float, float]:
    """Maximum absolute and relative error between two CPU float32 tensors."""
    reference = reference.detach().to("cpu", torch.float32)
    other = other.detach().to("cpu", torch.float32)
    absolute = (reference - other).abs()
    relative = absolute / reference.abs().clamp(min=1e-12)
    return float(absolute.max()), float(relative.max())


# ---------------------------------------------------------------------------
# Batch equivalence, on CPU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [8, 64, 256])
def test_a_position_gives_the_same_answer_alone_and_inside_a_batch(model, batch):
    """The same observation, evaluated alone and padded into a larger batch."""
    single = deterministic_observation(seed=21, batch=1)
    filler = deterministic_observation(seed=22, batch=batch - 1)
    stacked = torch.cat([single, filler], dim=0)

    with torch.no_grad():
        alone = model.forward_observation(single)
        together = model.forward_observation(stacked)

    row = together.row(0)
    assert torch.allclose(alone.policy_logits, row.policy_logits, **FLOAT32_TOLERANCE)
    assert torch.allclose(alone.value_logits, row.value_logits, **FLOAT32_TOLERANCE)
    assert torch.allclose(alone.belief_logits, row.belief_logits, **FLOAT32_TOLERANCE)


@pytest.mark.parametrize("batch", [8, 64, 256])
def test_the_selected_action_does_not_depend_on_the_batch(model, batch):
    state = nonterminal_state(40)
    legal = legal_actions(state)
    observation = build_observation(state, state.acting_player)

    single = tokenize_numpy_observation(observation)
    filler = observation_to_tokens(deterministic_observation(seed=23, batch=batch - 1))
    stacked = torch.cat([single, filler], dim=0)

    with torch.no_grad():
        alone = model(single)
        together = model(stacked)

    assert greedy_action(alone.policy_logits[0], legal) == greedy_action(
        together.policy_logits[0], legal
    )


def test_every_row_of_a_batch_matches_its_own_single_evaluation(model):
    """Not just row zero: a batching bug could touch any row."""
    observations = deterministic_observation(seed=24, batch=6)
    with torch.no_grad():
        batched = model.forward_observation(observations)
        for index in range(observations.shape[0]):
            alone = model.forward_observation(observations[index : index + 1])
            assert torch.allclose(
                alone.policy_logits, batched.row(index).policy_logits, **FLOAT32_TOLERANCE
            )
            assert torch.allclose(
                alone.value_logits, batched.row(index).value_logits, **FLOAT32_TOLERANCE
            )


# ---------------------------------------------------------------------------
# Cross-device equivalence
# ---------------------------------------------------------------------------


def _compare_devices(dtype: torch.dtype, tolerance: dict) -> dict:
    """Run identical weights and inputs on CPU float32 and MPS `dtype`."""
    tokens = _corpus_tokens()
    cpu_model = build_integration_model(seed=TEST_SEED, device="cpu", dtype=torch.float32)
    mps_model = build_integration_model(seed=TEST_SEED, device="mps", dtype=dtype)

    with torch.no_grad():
        cpu_out = cpu_model(tokens)
        mps_out = mps_model(tokens.to("mps", dtype))

    mps_cpu = mps_out.detached_cpu()
    assert mps_cpu.all_finite(), "Metal produced a non-finite output"

    policy_error = _max_errors(cpu_out.policy_logits, mps_cpu.policy_logits)
    value_error = _max_errors(
        value_probabilities(cpu_out.value_logits), value_probabilities(mps_cpu.value_logits)
    )
    belief_error = _max_errors(cpu_out.belief_logits, mps_cpu.belief_logits)

    assert torch.allclose(cpu_out.value_logits, mps_cpu.value_logits, **tolerance)
    assert torch.allclose(cpu_out.belief_logits, mps_cpu.belief_logits, **tolerance)
    assert torch.allclose(cpu_out.policy_logits, mps_cpu.policy_logits, **tolerance)

    return {
        "policy_logits": policy_error,
        "value_probabilities": value_error,
        "belief_logits": belief_error,
        "cpu": cpu_out,
        "mps": mps_cpu,
    }


@requires_mps
def test_cpu_and_metal_float32_agree_within_the_declared_tolerance():
    errors = _compare_devices(torch.float32, FLOAT32_TOLERANCE)
    for name in ("policy_logits", "value_probabilities", "belief_logits"):
        absolute, _ = errors[name]
        assert np.isfinite(absolute)


@requires_mps
def test_metal_float16_is_finite_and_agrees_within_the_declared_tolerance():
    errors = _compare_devices(torch.float16, FLOAT16_TOLERANCE)
    for name in ("policy_logits", "value_probabilities", "belief_logits"):
        absolute, _ = errors[name]
        assert np.isfinite(absolute)


@requires_mps
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_a_crafted_margin_selects_the_same_action_on_every_device(dtype):
    """With a decisive margin, device and precision cannot change the choice."""
    state = nonterminal_state(40)
    legal = legal_actions(state)
    tokens = tokenize_numpy_observation(build_observation(state, state.acting_player))

    cpu_model = build_integration_model(seed=TEST_SEED, device="cpu", dtype=torch.float32)
    mps_model = build_integration_model(seed=TEST_SEED, device="mps", dtype=dtype)
    with torch.no_grad():
        cpu_logits = cpu_model(tokens).policy_logits[0].to("cpu", torch.float32)
        mps_logits = mps_model(tokens.to("mps", dtype)).policy_logits[0].to("cpu", torch.float32)

    for target in legal[:: max(1, len(legal) // 5)]:
        margin = torch.zeros(10_000)
        margin[target] = 100.0  # far larger than any float16 rounding
        assert greedy_action(cpu_logits + margin, legal) == target
        assert greedy_action(mps_logits + margin, legal) == target


@requires_mps
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_natural_greedy_agreement_is_measured_rather_than_assumed(dtype):
    """Reports the natural agreement rate; near-ties are expected, not hidden.

    The fixture is untrained, so many legal actions score within a few ulps of
    each other. A kernel difference can reorder those legitimately. Asserting
    exact agreement here would either be flaky or would need a tolerance so wide
    it stopped meaning anything, so the rate is measured and printed instead.
    """
    cpu_model = build_integration_model(seed=TEST_SEED, device="cpu", dtype=torch.float32)
    mps_model = build_integration_model(seed=TEST_SEED, device="mps", dtype=dtype)

    agreements, total = 0, 0
    for ply in (20, 35, 50, 65, 80, 95):
        state = nonterminal_state(ply)
        legal = legal_actions(state)
        tokens = tokenize_numpy_observation(build_observation(state, state.acting_player))
        with torch.no_grad():
            cpu_logits = cpu_model(tokens).policy_logits[0].to("cpu", torch.float32)
            mps_logits = mps_model(tokens.to("mps", dtype)).policy_logits[0].to(
                "cpu", torch.float32
            )
        total += 1
        agreements += int(greedy_action(cpu_logits, legal) == greedy_action(mps_logits, legal))

    rate = agreements / total
    print(f"natural greedy agreement, {dtype}: {agreements}/{total} = {rate:.3f}")
    assert total == 6
    assert rate >= 0.5  # a hard floor, not a strength claim


@requires_mps
def test_the_legal_action_set_is_identical_regardless_of_device():
    """Legality comes from the engine, so no device can change it."""
    for ply in (25, 50, 75):
        state = nonterminal_state(ply)
        assert legal_actions(state) == legal_actions(state)


@requires_mps
@pytest.mark.parametrize("batch", [8, 64])
def test_batch_equivalence_holds_on_metal_too(batch):
    mps_model = build_integration_model(seed=TEST_SEED, device="mps", dtype=torch.float32)
    single = deterministic_observation(seed=31, batch=1).to("mps")
    stacked = torch.cat([single, deterministic_observation(seed=32, batch=batch - 1).to("mps")])
    with torch.no_grad():
        alone = mps_model.forward_observation(single).detached_cpu()
        together = mps_model.forward_observation(stacked).detached_cpu().row(0)
    assert torch.allclose(alone.policy_logits, together.policy_logits, **FLOAT32_TOLERANCE)
    assert torch.allclose(alone.value_logits, together.value_logits, **FLOAT32_TOLERANCE)


def test_the_declared_tolerances_are_the_documented_ones():
    assert FLOAT32_TOLERANCE == {"atol": 1e-4, "rtol": 1e-4}
    assert FLOAT16_TOLERANCE == {"atol": 5e-2, "rtol": 5e-2}
