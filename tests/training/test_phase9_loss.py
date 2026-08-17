"""Phase 9 Agent 5: the frozen objective, checked against the assignment.

Every reference here is written from `05_AGENT_5_PPO_TRAINER_AND_RESUME.md`
and the common contract, in plain Python, rather than by calling the module
under test. A test that reuses the implementation's own arithmetic proves only
that the module is self-consistent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from stratego.model.contract import BELIEF_IGNORE_INDEX, POLICY_LOGIT_COUNT, ModelOutputs
from stratego.model.losses import belief_loss as frozen_belief_loss
from stratego.training import phase9_loss as pl
from stratego.training.phase9_contract import (
    BELIEF_LOSS_WEIGHT,
    PPO_CLIP_EPSILON,
    VALUE_LOSS_WEIGHT,
)

SQUARES = 100
BELIEF_CLASSES = 12


# ---------------------------------------------------------------------------
# A deterministic synthetic batch
# ---------------------------------------------------------------------------


def make_batch(size=6, legal_count=7, seed=20260817, eligible=None):
    """A batch whose every field is under the test's control.

    Real examples come from a sealed rollout; here the point is the
    arithmetic, so the distributions are built explicitly and the stored
    behavior probabilities are rounded to float32 exactly as storage would.
    """
    rng = np.random.default_rng(seed)
    legal_mask = np.zeros((size, POLICY_LOGIT_COUNT), dtype=bool)
    behavior = np.zeros((size, POLICY_LOGIT_COUNT), dtype=np.float32)
    actions = np.zeros(size, dtype=np.int64)
    for row in range(size):
        columns = np.sort(
            rng.choice(POLICY_LOGIT_COUNT, size=legal_count, replace=False)
        )
        legal_mask[row, columns] = True
        logits = rng.normal(size=legal_count)
        probabilities = np.exp(logits - logits.max())
        probabilities = (probabilities / probabilities.sum()).astype(np.float32)
        behavior[row, columns] = probabilities
        actions[row] = int(columns[rng.integers(legal_count)])
    if eligible is None:
        eligible = np.array([row % 2 == 0 for row in range(size)], dtype=bool)
    targets = rng.dirichlet(np.ones(3), size=size).astype(np.float32)
    belief_target = np.full((size, SQUARES), BELIEF_IGNORE_INDEX, dtype=np.int64)
    belief_mask = np.zeros((size, SQUARES), dtype=bool)
    for row in range(size):
        squares = rng.choice(SQUARES, size=4, replace=False)
        belief_target[row, squares] = rng.integers(0, BELIEF_CLASSES, size=4)
        belief_mask[row, squares] = True
    return {
        "legal_mask": torch.from_numpy(legal_mask),
        "behavior_probabilities": torch.from_numpy(behavior),
        "sampled_action_model": torch.from_numpy(actions),
        "behavior_action_probability": torch.from_numpy(
            behavior[np.arange(size), actions]
        ),
        "standardized_advantage": torch.from_numpy(
            rng.normal(size=size).astype(np.float32)
        ),
        "ppo_eligible": torch.from_numpy(np.asarray(eligible, dtype=bool)),
        "wdl_target": torch.from_numpy(targets),
        "belief_target": torch.from_numpy(belief_target),
        "belief_mask": torch.from_numpy(belief_mask),
    }


def make_outputs(size=6, seed=99, requires_grad=False):
    generator = torch.Generator().manual_seed(seed)
    return ModelOutputs(
        policy_logits=torch.randn(
            size, POLICY_LOGIT_COUNT, generator=generator, requires_grad=requires_grad
        ),
        value_logits=torch.randn(size, 3, generator=generator, requires_grad=requires_grad),
        belief_logits=torch.randn(
            size, SQUARES, BELIEF_CLASSES, generator=generator, requires_grad=requires_grad
        ),
    )


def evaluate(outputs=None, batch=None, kl_beta=0.005, entropy_coefficient=0.005, **overrides):
    batch = dict(batch or make_batch())
    batch.update(overrides)
    return pl.phase9_batch_loss(
        outputs or make_outputs(size=int(batch["legal_mask"].shape[0])),
        kl_beta=kl_beta,
        entropy_coefficient=entropy_coefficient,
        **batch,
    )


# ---------------------------------------------------------------------------
# Reference arithmetic, written from the assignment
# ---------------------------------------------------------------------------


def reference_legal_softmax(logits_row, legal_row):
    columns = np.flatnonzero(legal_row)
    values = np.asarray(logits_row, dtype=np.float64)[columns]
    shifted = values - values.max()
    exponentials = np.exp(shifted)
    return columns, exponentials / exponentials.sum()


def reference_ppo(outputs, batch):
    """`-E[min(r*A, clip(r, 0.8, 1.2)*A)]` over the eligible subset."""
    logits = outputs.policy_logits.detach().numpy()
    legal = batch["legal_mask"].numpy()
    actions = batch["sampled_action_model"].numpy()
    behavior = batch["behavior_action_probability"].numpy()
    advantage = batch["standardized_advantage"].numpy()
    eligible = batch["ppo_eligible"].numpy()
    terms = []
    for row in range(logits.shape[0]):
        if not eligible[row]:
            continue
        columns, probabilities = reference_legal_softmax(logits[row], legal[row])
        theta = float(probabilities[list(columns).index(int(actions[row]))])
        ratio = theta / float(behavior[row])
        clipped = min(max(ratio, 1.0 - PPO_CLIP_EPSILON), 1.0 + PPO_CLIP_EPSILON)
        terms.append(
            min(ratio * float(advantage[row]), clipped * float(advantage[row]))
        )
    return -float(np.mean(terms)) if terms else 0.0


def reference_kl(outputs, batch):
    """`sum_a pi_b ln(pi_b / pi_theta)` over the legal set, meaned."""
    logits = outputs.policy_logits.detach().numpy()
    legal = batch["legal_mask"].numpy()
    behavior = batch["behavior_probabilities"].numpy()
    rows = []
    for row in range(logits.shape[0]):
        columns, theta = reference_legal_softmax(logits[row], legal[row])
        stored = behavior[row][columns].astype(np.float64)
        total = 0.0
        for index in range(len(columns)):
            if stored[index] > 0:
                total += stored[index] * math.log(stored[index] / theta[index])
        rows.append(total)
    return float(np.mean(rows))


def reference_entropy(outputs, batch):
    logits = outputs.policy_logits.detach().numpy()
    legal = batch["legal_mask"].numpy()
    rows = []
    for row in range(logits.shape[0]):
        _columns, theta = reference_legal_softmax(logits[row], legal[row])
        rows.append(-float(np.sum(theta * np.log(theta))))
    return float(np.mean(rows))


def reference_value(outputs, batch):
    logits = outputs.value_logits.detach().numpy().astype(np.float64)
    targets = batch["wdl_target"].numpy().astype(np.float64)
    rows = []
    for row in range(logits.shape[0]):
        shifted = logits[row] - logits[row].max()
        log_probabilities = shifted - np.log(np.exp(shifted).sum())
        rows.append(-float(np.sum(targets[row] * log_probabilities)))
    return float(np.mean(rows))


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------


def test_ppo_matches_the_frozen_objective():
    outputs, batch = make_outputs(), make_batch()
    loss = evaluate(outputs, batch)
    assert float(loss.ppo) == pytest.approx(reference_ppo(outputs, batch), abs=1e-5)


def test_ppo_ignores_every_ineligible_example():
    """The advantage filter narrows the policy gradient and nothing else."""
    outputs, batch = make_outputs(), make_batch()
    before = evaluate(outputs, batch)
    altered = dict(batch)
    advantage = batch["standardized_advantage"].clone()
    ineligible = ~batch["ppo_eligible"]
    advantage[ineligible] = advantage[ineligible] + 17.0
    altered["standardized_advantage"] = advantage
    after = evaluate(outputs, altered)
    assert float(after.ppo) == pytest.approx(float(before.ppo), abs=1e-7)
    # ...while the four all-decision terms are untouched by an advantage at all.
    assert float(after.value) == pytest.approx(float(before.value), abs=1e-7)
    assert float(after.belief) == pytest.approx(float(before.belief), abs=1e-7)
    assert float(after.kl) == pytest.approx(float(before.kl), abs=1e-7)


def test_ineligible_examples_still_carry_value_belief_kl_entropy_gradient():
    """An ineligible decision is not an excluded decision."""
    size = 4
    outputs = make_outputs(size=size, requires_grad=True)
    batch = make_batch(size=size, eligible=np.zeros(size, dtype=bool))
    loss = evaluate(outputs, batch)
    assert float(loss.ppo.detach()) == 0.0
    assert loss.ppo_examples == 0
    loss.total.backward()
    assert bool(torch.isfinite(outputs.policy_logits.grad).all())
    assert float(outputs.value_logits.grad.abs().sum()) > 0.0
    assert float(outputs.belief_logits.grad.abs().sum()) > 0.0
    # The policy head still receives KL and entropy gradient, just no PPO.
    assert float(outputs.policy_logits.grad.abs().sum()) > 0.0


def test_empty_eligible_subset_keeps_the_graph_connected():
    size = 4
    outputs = make_outputs(size=size, requires_grad=True)
    batch = make_batch(size=size, eligible=np.zeros(size, dtype=bool))
    loss = evaluate(outputs, batch)
    assert loss.ppo.requires_grad
    assert float(loss.ppo.detach()) == 0.0
    assert loss.clip_fraction == 0.0


def test_clip_fraction_counts_the_eligible_subset_only():
    outputs, batch = make_outputs(), make_batch()
    loss = evaluate(outputs, batch)
    logits = outputs.policy_logits.detach().numpy()
    legal = batch["legal_mask"].numpy()
    actions = batch["sampled_action_model"].numpy()
    behavior = batch["behavior_action_probability"].numpy()
    eligible = batch["ppo_eligible"].numpy()
    expected = 0
    for row in range(logits.shape[0]):
        if not eligible[row]:
            continue
        columns, probabilities = reference_legal_softmax(logits[row], legal[row])
        ratio = float(probabilities[list(columns).index(int(actions[row]))]) / float(
            behavior[row]
        )
        expected += int(abs(ratio - 1.0) > PPO_CLIP_EPSILON)
    assert loss.ppo_clipped == expected
    assert loss.ppo_examples == int(eligible.sum())


def test_clipping_bounds_the_surrogate():
    """A ratio driven far above 1.2 stops improving the objective."""
    size = 2
    batch = make_batch(size=size, eligible=np.ones(size, dtype=bool))
    batch["standardized_advantage"] = torch.ones(size)
    outputs = make_outputs(size=size)
    boosted = outputs.policy_logits.detach().clone()
    boosted[torch.arange(size), batch["sampled_action_model"]] += 50.0
    mild = ModelOutputs(
        policy_logits=boosted,
        value_logits=outputs.value_logits,
        belief_logits=outputs.belief_logits,
    )
    loss = evaluate(mild, batch)
    # With A = +1 and r >> 1.2 every term is clipped to 1.2, so -E[.] = -1.2.
    assert float(loss.ppo) == pytest.approx(-(1.0 + PPO_CLIP_EPSILON), abs=1e-4)
    assert loss.clip_fraction == 1.0


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def test_illegal_logits_change_no_loss_and_no_gradient():
    size = 4
    batch = make_batch(size=size)
    outputs = make_outputs(size=size, requires_grad=True)
    baseline = evaluate(outputs, batch)
    baseline.total.backward()
    baseline_grad = outputs.policy_logits.grad.detach().clone()

    tampered = make_outputs(size=size, requires_grad=True)
    with torch.no_grad():
        tampered.policy_logits.copy_(outputs.policy_logits.detach())
        tampered.value_logits.copy_(outputs.value_logits.detach())
        tampered.belief_logits.copy_(outputs.belief_logits.detach())
        illegal = ~batch["legal_mask"]
        tampered.policy_logits[illegal] = 500.0
    altered = evaluate(tampered, batch)
    altered.total.backward()

    assert float(altered.total.detach()) == pytest.approx(
        float(baseline.total.detach()), abs=1e-6
    )
    assert float(altered.kl.detach()) == pytest.approx(float(baseline.kl.detach()), abs=1e-6)
    assert float(altered.entropy.detach()) == pytest.approx(
        float(baseline.entropy.detach()), abs=1e-6
    )
    illegal = ~batch["legal_mask"]
    assert float(tampered.policy_logits.grad[illegal].abs().max()) == 0.0
    assert torch.allclose(
        tampered.policy_logits.grad[batch["legal_mask"]],
        baseline_grad[batch["legal_mask"]],
        atol=1e-6,
    )


def test_illegal_realized_action_is_refused():
    batch = make_batch()
    actions = batch["sampled_action_model"].clone()
    row = 0
    illegal = int(np.flatnonzero(~batch["legal_mask"][row].numpy())[0])
    actions[row] = illegal
    with pytest.raises(pl.Phase9LossError, match="illegal under the batch's mask"):
        evaluate(batch=batch, sampled_action_model=actions)


def test_behavior_mass_on_an_illegal_action_is_refused():
    batch = make_batch()
    behavior = batch["behavior_probabilities"].clone()
    row = 0
    illegal = int(np.flatnonzero(~batch["legal_mask"][row].numpy())[0])
    behavior[row, illegal] = 0.5
    with pytest.raises(pl.Phase9LossError, match="illegal action"):
        evaluate(batch=batch, behavior_probabilities=behavior)


def test_dense_matrix_and_stored_scalar_must_agree():
    batch = make_batch()
    scalar = batch["behavior_action_probability"].clone()
    scalar[0] = float(scalar[0]) * 0.5
    with pytest.raises(pl.Phase9LossError, match="disagree"):
        evaluate(batch=batch, behavior_action_probability=scalar)


def test_non_positive_denominator_is_refused():
    batch = make_batch()
    behavior = batch["behavior_probabilities"].clone()
    scalar = batch["behavior_action_probability"].clone()
    row, action = 0, int(batch["sampled_action_model"][0])
    behavior[row, action] = 0.0
    scalar[row] = 0.0
    # Renormalize so the row still sums to one and only the denominator is bad.
    behavior[row] = behavior[row] / behavior[row].sum()
    behavior[row, action] = 0.0
    with pytest.raises(pl.Phase9LossError):
        evaluate(
            batch=batch,
            behavior_probabilities=behavior,
            behavior_action_probability=scalar,
        )


def test_behavior_row_must_sum_to_one():
    batch = make_batch()
    behavior = batch["behavior_probabilities"].clone() * 0.5
    scalar = batch["behavior_action_probability"].clone() * 0.5
    with pytest.raises(pl.Phase9LossError, match="sums to 1"):
        evaluate(
            batch=batch,
            behavior_probabilities=behavior,
            behavior_action_probability=scalar,
        )


# ---------------------------------------------------------------------------
# Value, belief
# ---------------------------------------------------------------------------


def test_value_is_soft_categorical_cross_entropy():
    outputs, batch = make_outputs(), make_batch()
    loss = evaluate(outputs, batch)
    assert float(loss.value) == pytest.approx(reference_value(outputs, batch), abs=1e-5)


def test_value_reduces_to_hard_cross_entropy_on_a_one_hot_target():
    size = 3
    outputs = make_outputs(size=size)
    batch = make_batch(size=size)
    one_hot = torch.zeros(size, 3)
    classes = torch.tensor([0, 1, 2])
    one_hot[torch.arange(size), classes] = 1.0
    loss = evaluate(outputs, batch, wdl_target=one_hot)
    expected = torch.nn.functional.cross_entropy(outputs.value_logits, classes)
    assert float(loss.value) == pytest.approx(float(expected), abs=1e-6)


def test_value_target_outside_the_simplex_is_refused():
    batch = make_batch()
    targets = batch["wdl_target"].clone()
    targets[0] = targets[0] * 2.0
    with pytest.raises(pl.Phase9LossError, match="simplex tolerance"):
        evaluate(batch=batch, wdl_target=targets)


def test_belief_is_the_frozen_phase8_loss():
    outputs, batch = make_outputs(), make_batch()
    loss = evaluate(outputs, batch)
    expected = frozen_belief_loss(
        outputs.belief_logits, batch["belief_target"], batch["belief_mask"]
    )
    assert float(loss.belief) == pytest.approx(float(expected), abs=1e-7)


# ---------------------------------------------------------------------------
# KL and entropy
# ---------------------------------------------------------------------------


def test_kl_matches_the_frozen_direction():
    outputs, batch = make_outputs(), make_batch()
    loss = evaluate(outputs, batch)
    assert float(loss.kl) == pytest.approx(reference_kl(outputs, batch), abs=1e-5)


def test_kl_direction_is_not_symmetric():
    """`D(pi_b || pi_theta)` is the frozen direction, not the other one."""
    outputs, batch = make_outputs(), make_batch()
    forward = float(evaluate(outputs, batch).kl)
    logits = outputs.policy_logits.detach().numpy()
    legal = batch["legal_mask"].numpy()
    behavior = batch["behavior_probabilities"].numpy()
    reverse_rows = []
    for row in range(logits.shape[0]):
        columns, theta = reference_legal_softmax(logits[row], legal[row])
        stored = behavior[row][columns].astype(np.float64)
        reverse_rows.append(float(np.sum(theta * np.log(theta / stored))))
    reverse = float(np.mean(reverse_rows))
    assert forward != pytest.approx(reverse, abs=1e-3)


def test_kl_is_zero_when_the_learner_is_the_behavior_policy():
    size = 3
    batch = make_batch(size=size)
    logits = torch.log(batch["behavior_probabilities"].clamp(min=1e-30))
    outputs = ModelOutputs(
        policy_logits=logits,
        value_logits=torch.zeros(size, 3),
        belief_logits=torch.zeros(size, SQUARES, BELIEF_CLASSES),
    )
    loss = evaluate(outputs, batch)
    assert float(loss.kl) == pytest.approx(0.0, abs=1e-6)
    assert float(loss.ratio_mean) == pytest.approx(1.0, abs=1e-5)


def test_zero_behavior_probability_contributes_zero_not_nan():
    size = 2
    batch = make_batch(size=size)
    behavior = batch["behavior_probabilities"].clone()
    columns = np.flatnonzero(batch["legal_mask"][0].numpy())
    victim = int(columns[0]) if int(columns[0]) != int(
        batch["sampled_action_model"][0]
    ) else int(columns[1])
    freed = float(behavior[0, victim])
    behavior[0, victim] = 0.0
    # Give the freed mass to another legal entry so the row still sums to one.
    other = int(columns[-1]) if int(columns[-1]) != victim else int(columns[-2])
    behavior[0, other] = behavior[0, other] + freed
    batch = dict(batch)
    batch["behavior_probabilities"] = behavior
    batch["behavior_action_probability"] = behavior[
        torch.arange(size), batch["sampled_action_model"]
    ]
    loss = evaluate(batch=batch)
    assert loss.all_finite()


def test_entropy_matches_the_reference_and_is_differentiable():
    size = 4
    outputs = make_outputs(size=size, requires_grad=True)
    batch = make_batch(size=size)
    loss = evaluate(outputs, batch)
    assert float(loss.entropy.detach()) == pytest.approx(
        reference_entropy(outputs, batch), abs=1e-5
    )
    loss.entropy.backward()
    assert float(outputs.policy_logits.grad.abs().sum()) > 0.0


def test_entropy_of_a_uniform_legal_row_is_log_legal_count():
    size, legal_count = 2, 7
    batch = make_batch(size=size, legal_count=legal_count)
    logits = torch.zeros(size, POLICY_LOGIT_COUNT)
    outputs = ModelOutputs(
        policy_logits=logits,
        value_logits=torch.zeros(size, 3),
        belief_logits=torch.zeros(size, SQUARES, BELIEF_CLASSES),
    )
    loss = evaluate(outputs, batch)
    assert float(loss.entropy) == pytest.approx(math.log(legal_count), abs=1e-5)
    assert float(loss.entropy_normalized) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_total_is_the_frozen_weighted_sum():
    beta, coefficient = 0.02, 0.003
    outputs, batch = make_outputs(), make_batch()
    loss = evaluate(outputs, batch, kl_beta=beta, entropy_coefficient=coefficient)
    expected = (
        float(loss.ppo)
        + VALUE_LOSS_WEIGHT * float(loss.value)
        + BELIEF_LOSS_WEIGHT * float(loss.belief)
        + beta * float(loss.kl)
        - coefficient * float(loss.entropy)
    )
    assert float(loss.total) == pytest.approx(expected, abs=1e-6)


def test_entropy_is_subtracted_so_a_larger_coefficient_lowers_the_loss():
    outputs, batch = make_outputs(), make_batch()
    low = evaluate(outputs, batch, entropy_coefficient=0.001)
    high = evaluate(outputs, batch, entropy_coefficient=0.005)
    assert float(high.total) < float(low.total)


def test_negative_beta_or_coefficient_is_refused():
    with pytest.raises(pl.Phase9LossError, match="kl_beta"):
        evaluate(kl_beta=-0.1)
    with pytest.raises(pl.Phase9LossError, match="entropy_coefficient"):
        evaluate(entropy_coefficient=-0.1)


def test_loss_semantics_states_the_frozen_weights():
    semantics = pl.loss_semantics()
    assert semantics["value"]["weight"] == VALUE_LOSS_WEIGHT
    assert semantics["belief"]["weight"] == BELIEF_LOSS_WEIGHT
    assert semantics["ppo"]["clip_epsilon"] == PPO_CLIP_EPSILON
    assert "pi_b || pi_theta" in semantics["kl"]["direction"]


# ---------------------------------------------------------------------------
# The frame reconciliation
# ---------------------------------------------------------------------------


class _StubExample:
    def __init__(self, learner_side, actions, probabilities):
        self.learner_side = learner_side
        self.behavior_legal_actions = tuple(actions)
        self.behavior_legal_probabilities = tuple(probabilities)
        self.game_id = "stub"
        self.decision_index = 0


def test_behavior_matrix_uses_the_frozen_frame_converter():
    from stratego.engine.constants import BLUE, RED
    from stratego.model.action_frame import absolute_action_to_model

    actions = (11, 202, 3003)
    probabilities = (0.2, 0.3, 0.5)
    for player in (RED, BLUE):
        matrix = pl.behavior_probability_matrix(
            [_StubExample(player, actions, probabilities)]
        )
        for action, probability in zip(actions, probabilities):
            column = absolute_action_to_model(action, player)
            assert matrix[0, column] == pytest.approx(probability, abs=1e-7)
        assert int((matrix[0] > 0).sum()) == len(actions)


def test_behavior_matrix_refuses_misaligned_runs():
    with pytest.raises(pl.Phase9LossError, match="probabilities"):
        pl.behavior_probability_matrix([_StubExample(0, (1, 2, 3), (0.5, 0.5))])
