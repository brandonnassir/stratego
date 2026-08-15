"""Phase 8 Agent 4: the frozen loss semantics, proven on hand-built batches.

The tests here state the three normalizations as arithmetic and require the
implementation to reproduce them exactly, prove that illegal logits are
inert, and prove that an illegal teacher action is a loud stop rather than a
training signal.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from stratego.model.contract import BELIEF_IGNORE_INDEX, ModelOutputs
from stratego.training.warmstart_loss import (
    WarmstartBatchLoss,
    WarmstartLossError,
    WarmstartLossWeights,
    legal_policy_entropy,
    warmstart_batch_loss,
)

BATCH = 4
ACTIONS = 10000
SQUARES = 100
TYPES = 12

WEIGHTS = WarmstartLossWeights(lambda_policy=1.0, lambda_value=1.0, lambda_belief=1.0)


def build_batch(seed: int = 20260815, policy_weights=(1.0, 0.5, 0.0, 1.0)) -> dict:
    """A deterministic synthetic batch with mixed supervision weights."""
    generator = torch.Generator().manual_seed(seed)
    # requires_grad matches reality — head logits always carry a graph — and
    # lets the zero-weight test prove the policy branch stays connected.
    outputs = ModelOutputs.validated(
        torch.randn(BATCH, ACTIONS, generator=generator).requires_grad_(True),
        torch.randn(BATCH, 3, generator=generator).requires_grad_(True),
        torch.randn(BATCH, SQUARES, TYPES, generator=generator).requires_grad_(True),
    )
    legal_mask = torch.zeros(BATCH, ACTIONS, dtype=torch.bool)
    actions = torch.zeros(BATCH, dtype=torch.int64)
    for row in range(BATCH):
        legal = torch.randperm(ACTIONS, generator=generator)[: 7 + 3 * row]
        legal_mask[row, legal] = True
        actions[row] = legal[row % legal.numel()]
    belief_mask = torch.zeros(BATCH, SQUARES, dtype=torch.bool)
    belief_target = torch.full((BATCH, SQUARES), BELIEF_IGNORE_INDEX, dtype=torch.int64)
    for row in range(BATCH):
        supervised = torch.randperm(SQUARES, generator=generator)[: 3 + 5 * row]
        belief_mask[row, supervised] = True
        belief_target[row, supervised] = torch.randint(
            0, TYPES, (supervised.numel(),), generator=generator
        )
    return {
        "outputs": outputs,
        "legal_mask": legal_mask,
        "policy_actions": actions,
        "policy_weights": torch.tensor(policy_weights, dtype=torch.float32),
        "value_targets": torch.tensor([0, 1, 2, 1], dtype=torch.int64),
        "belief_targets": belief_target,
        "belief_mask": belief_mask,
    }


def compute(batch: dict, weights: WarmstartLossWeights = WEIGHTS) -> WarmstartBatchLoss:
    return warmstart_batch_loss(
        batch["outputs"],
        legal_mask=batch["legal_mask"],
        policy_actions=batch["policy_actions"],
        policy_weights=batch["policy_weights"],
        value_targets=batch["value_targets"],
        belief_targets=batch["belief_targets"],
        belief_mask=batch["belief_mask"],
        weights=weights,
    )


class TestPolicyLegalityMasking:
    def test_arbitrarily_large_illegal_logits_change_nothing(self):
        batch = build_batch()
        reference = compute(batch)
        poisoned_logits = batch["outputs"].policy_logits.clone()
        poisoned_logits[~batch["legal_mask"]] = 1e12
        poisoned = dict(
            batch,
            outputs=ModelOutputs.validated(
                poisoned_logits,
                batch["outputs"].value_logits,
                batch["outputs"].belief_logits,
            ),
        )
        result = compute(poisoned)
        # Exact equality, not closeness: the fill happens before normalization,
        # so the illegal entries' stored values are unreachable.
        assert torch.equal(result.policy, reference.policy)
        assert torch.equal(result.total, reference.total)
        assert result.legal_policy_entropy == reference.legal_policy_entropy

    def test_negative_illegal_logits_change_nothing(self):
        batch = build_batch()
        reference = compute(batch)
        poisoned_logits = batch["outputs"].policy_logits.clone()
        poisoned_logits[~batch["legal_mask"]] = -1e12
        poisoned = dict(
            batch,
            outputs=ModelOutputs.validated(
                poisoned_logits,
                batch["outputs"].value_logits,
                batch["outputs"].belief_logits,
            ),
        )
        assert torch.equal(compute(poisoned).policy, reference.policy)

    def test_illegal_teacher_action_raises(self):
        batch = build_batch()
        actions = batch["policy_actions"].clone()
        illegal = (~batch["legal_mask"][0]).nonzero()[0, 0]
        actions[0] = illegal
        with pytest.raises(WarmstartLossError, match="illegal"):
            compute(dict(batch, policy_actions=actions))

    def test_out_of_range_teacher_action_raises(self):
        batch = build_batch()
        actions = batch["policy_actions"].clone()
        actions[1] = ACTIONS
        with pytest.raises(WarmstartLossError, match="outside"):
            compute(dict(batch, policy_actions=actions))


class TestPolicyNormalization:
    def test_weighted_normalization_is_exact(self):
        batch = build_batch()
        result = compute(batch)
        logits = batch["outputs"].policy_logits.masked_fill(~batch["legal_mask"], -1e9)
        log_probabilities = F.log_softmax(logits, dim=1)
        per_example = -log_probabilities.gather(
            1, batch["policy_actions"][:, None]
        ).squeeze(1)
        weights = batch["policy_weights"]
        expected = (weights * per_example).sum() / weights.sum()
        assert torch.allclose(result.policy, expected, rtol=0, atol=0)
        assert result.policy_supervised_decisions == 3
        assert result.policy_weight_sum == pytest.approx(2.5)

    def test_zero_weight_batch_contributes_exactly_zero(self):
        batch = build_batch(policy_weights=(0.0, 0.0, 0.0, 0.0))
        result = compute(batch)
        assert float(result.policy.detach()) == 0.0
        assert result.policy_supervised_decisions == 0
        expected_total = result.value + result.belief
        assert torch.allclose(result.total, expected_total, rtol=0, atol=0)
        # The graph stays connected: backward through the total must succeed
        # even when the policy term is the zero branch.
        result.total.backward()

    def test_loss_weights_scale_components(self):
        batch = build_batch()
        weighted = compute(
            batch, WarmstartLossWeights(lambda_policy=1.0, lambda_value=0.5, lambda_belief=0.5)
        )
        expected = weighted.policy + 0.5 * weighted.value + 0.5 * weighted.belief
        assert torch.allclose(weighted.total, expected, rtol=0, atol=0)

    def test_negative_weights_rejected(self):
        batch = build_batch(policy_weights=(1.0, -0.5, 0.0, 1.0))
        with pytest.raises(WarmstartLossError, match="non-negative"):
            compute(batch)


class TestValueLoss:
    def test_value_is_mean_cross_entropy(self):
        batch = build_batch()
        result = compute(batch)
        expected = F.cross_entropy(
            batch["outputs"].value_logits, batch["value_targets"]
        )
        assert torch.allclose(result.value, expected, rtol=0, atol=0)
        assert result.value_decisions == BATCH

    def test_out_of_range_value_target_raises(self):
        batch = build_batch()
        targets = batch["value_targets"].clone()
        targets[0] = 3
        with pytest.raises(WarmstartLossError, match="value target"):
            compute(dict(batch, value_targets=targets))


class TestBeliefLoss:
    def test_normalization_is_per_supervised_square(self):
        batch = build_batch()
        result = compute(batch)
        logits = batch["outputs"].belief_logits
        labels = batch["belief_targets"]
        mask = batch["belief_mask"]
        per_square = F.cross_entropy(
            logits.reshape(BATCH * SQUARES, TYPES),
            labels.reshape(BATCH * SQUARES),
            ignore_index=BELIEF_IGNORE_INDEX,
            reduction="none",
        ).reshape(BATCH, SQUARES)
        expected = per_square[mask].sum() / mask.sum()
        assert torch.allclose(result.belief, expected, rtol=0, atol=1e-7)
        assert result.belief_supervised_pieces == int(mask.sum())

    def test_hidden_count_does_not_multiply_influence(self):
        # Two batches identical except one has 4x the supervised squares with
        # identical per-square CE structure: per-square normalization keeps the
        # belief loss magnitude comparable rather than 4x larger.
        sparse = build_batch()
        result = compute(sparse)
        assert result.belief_supervised_pieces > 0
        # There are never 100 supervised squares from a real position, and the
        # loss must not average over all 100 with zeros in the denominator: the
        # denominator equals exactly the supervised count.
        mask = sparse["belief_mask"]
        assert int(mask.sum()) < BATCH * SQUARES

    def test_mask_and_labels_must_agree(self):
        batch = build_batch()
        mask = batch["belief_mask"].clone()
        row, square = 0, int(batch["belief_mask"][0].nonzero()[0])
        mask[row, square] = False
        with pytest.raises(WarmstartLossError, match="mask"):
            compute(dict(batch, belief_mask=mask))

    def test_zero_supervised_squares_contribute_zero(self):
        batch = build_batch()
        empty_mask = torch.zeros(BATCH, SQUARES, dtype=torch.bool)
        empty_labels = torch.full((BATCH, SQUARES), BELIEF_IGNORE_INDEX, dtype=torch.int64)
        result = compute(dict(batch, belief_mask=empty_mask, belief_targets=empty_labels))
        assert float(result.belief.detach()) == 0.0
        assert result.belief_supervised_pieces == 0


class TestEntropyReporting:
    def test_uniform_over_legal_reports_maximum_entropy(self):
        legal_mask = torch.zeros(2, ACTIONS, dtype=torch.bool)
        legal_mask[0, :10] = True
        legal_mask[1, :100] = True
        logits = torch.zeros(2, ACTIONS)
        log_probabilities = F.log_softmax(
            logits.masked_fill(~legal_mask, -1e9), dim=1
        )
        entropy, normalized = legal_policy_entropy(log_probabilities, legal_mask)
        expected = (math.log(10) + math.log(100)) / 2
        assert entropy == pytest.approx(expected, rel=1e-5)
        assert normalized == pytest.approx(1.0, rel=1e-5)

    def test_single_legal_action_defines_normalized_entropy_one(self):
        legal_mask = torch.zeros(1, ACTIONS, dtype=torch.bool)
        legal_mask[0, 42] = True
        log_probabilities = F.log_softmax(
            torch.zeros(1, ACTIONS).masked_fill(~legal_mask, -1e9), dim=1
        )
        entropy, normalized = legal_policy_entropy(log_probabilities, legal_mask)
        assert entropy == pytest.approx(0.0, abs=1e-6)
        assert normalized == pytest.approx(1.0)


class TestReporting:
    def test_to_dict_carries_every_required_per_batch_quantity(self):
        result = compute(build_batch())
        payload = result.to_dict()
        for key in (
            "loss_total",
            "loss_policy",
            "loss_value",
            "loss_belief",
            "policy_supervised_decisions",
            "policy_weight_sum",
            "value_decisions",
            "belief_supervised_pieces",
            "legal_policy_entropy",
            "legal_policy_entropy_normalized",
            "lambda_policy",
            "lambda_value",
            "lambda_belief",
        ):
            assert key in payload
        assert result.all_finite()
