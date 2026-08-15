"""Phase 8 Agent 4: the frozen per-batch warm-start training loss.

Specification sources:

- `04_AGENT_4_TRAINER_AND_RESUME.md` ("Loss implementation", "Policy legality
  masking")
- `00_PHASE_8_SEQUENCE_AND_COMMON_CONTRACT.md` sections 16 and 18 (target
  semantics, per-head normalization)
- Agent 1's frozen `loss_semantics()` in
  :mod:`stratego.training.warmstart_contract`

Composition over reimplementation
---------------------------------
The three heads reuse the frozen primitives of :mod:`stratego.model.losses`
wherever one exists, because "loss normalization exact" is easiest to prove
about code that *is* the frozen definition rather than a copy of it:

```text
policy   masked_policy_log_probabilities (fill -1e9), then the Agent 1
         weighted normalization sum(w_i * CE_i) / sum(w_i)
value    stratego.model.losses.value_loss — mean CE over the batch
belief   stratego.model.losses.belief_loss — per supervised square
```

The policy normalization cannot reuse `stratego.model.losses.policy_loss`
directly (that is an unweighted mean), so this module reimplements exactly the
weighted form Agent 1 froze, on top of the same masked log-probabilities.

Illegal actions cannot reach the loss twice over: the masked log-softmax
replaces every illegal logit with the frozen `-1e9` fill *before* normalizing,
so their stored values never matter, and a teacher action that is itself
illegal under the batch's mask raises rather than training on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.contract import (
    POLICY_LOGIT_COUNT,
    VALUE_CLASS_COUNT,
    ModelContractError,
    ModelOutputs,
)
from ..model.losses import (
    belief_loss,
    masked_policy_log_probabilities,
    value_loss,
)

#: The per-batch loss semantics of the Phase 8 warm start. A change to any
#: normalization, masking or combination rule is a new version after review.
WARMSTART_LOSS_VERSION = "warmstart_loss_v1"


class WarmstartLossError(RuntimeError):
    """A batch violated the frozen loss contract. Never repaired, always raised."""


@dataclass(frozen=True)
class WarmstartLossWeights:
    """The frozen loss combination `L = lp*L_policy + lv*L_value + lb*L_belief`."""

    lambda_policy: float
    lambda_value: float
    lambda_belief: float

    def __post_init__(self) -> None:
        for name in ("lambda_policy", "lambda_value", "lambda_belief"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise WarmstartLossError(f"{name} must be a number, got {value!r}")
            if not value >= 0.0:
                raise WarmstartLossError(f"{name} must be >= 0, got {value!r}")

    def to_dict(self) -> dict:
        return {
            "lambda_policy": float(self.lambda_policy),
            "lambda_value": float(self.lambda_value),
            "lambda_belief": float(self.lambda_belief),
        }


@dataclass(frozen=True)
class WarmstartBatchLoss:
    """One batch's loss components plus every per-batch supervision count.

    The four loss entries stay tensors (the total carries the graph); the
    counts and diagnostics are plain numbers, detached at construction, so a
    metrics consumer can hold the structure without holding the graph.
    """

    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    belief: torch.Tensor
    weights: WarmstartLossWeights

    batch_size: int
    policy_supervised_decisions: int
    policy_weight_sum: float
    value_decisions: int
    belief_supervised_pieces: int

    #: Mean Shannon entropy (nats) of the legal-action distribution, and the
    #: same normalized by `ln(legal_count)` (a one-legal-action row contributes
    #: 1.0 by convention: the forced distribution is uniform over its support).
    legal_policy_entropy: float
    legal_policy_entropy_normalized: float

    def all_finite(self) -> bool:
        return all(
            bool(torch.isfinite(component).all())
            for component in (self.total, self.policy, self.value, self.belief)
        )

    def to_dict(self) -> dict:
        return {
            "loss_total": float(self.total.detach()),
            "loss_policy": float(self.policy.detach()),
            "loss_value": float(self.value.detach()),
            "loss_belief": float(self.belief.detach()),
            "batch_size": self.batch_size,
            "policy_supervised_decisions": self.policy_supervised_decisions,
            "policy_weight_sum": self.policy_weight_sum,
            "value_decisions": self.value_decisions,
            "belief_supervised_pieces": self.belief_supervised_pieces,
            "legal_policy_entropy": self.legal_policy_entropy,
            "legal_policy_entropy_normalized": self.legal_policy_entropy_normalized,
            **self.weights.to_dict(),
        }


def _require_shape(tensor: torch.Tensor, shape: tuple, name: str) -> None:
    if tuple(tensor.shape) != shape:
        raise WarmstartLossError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def check_teacher_actions_legal(
    policy_actions: torch.Tensor, legal_mask: torch.Tensor
) -> None:
    """Raise unless every teacher action is legal under the batch's mask.

    The corpus audits already proved recorded actions legal, so a violation
    here means the batch was assembled wrong (frame mix-up, index slip) — a
    stop condition, never something to mask away.
    """
    actions = policy_actions.to(torch.int64)
    if bool(((actions < 0) | (actions >= POLICY_LOGIT_COUNT)).any()):
        raise WarmstartLossError(
            f"a policy target action is outside 0..{POLICY_LOGIT_COUNT - 1}"
        )
    chosen_legal = legal_mask.to(torch.bool).gather(1, actions[:, None]).squeeze(1)
    if not bool(chosen_legal.all()):
        illegal = int((~chosen_legal).sum())
        raise WarmstartLossError(
            f"{illegal} teacher action(s) are illegal under the supplied legality "
            "mask; refusing to train on them"
        )


def legal_policy_entropy(
    log_probabilities: torch.Tensor, legal_mask: torch.Tensor
) -> tuple:
    """`(mean_entropy_nats, mean_normalized_entropy)` of one batch.

    Both are computed over legal entries only. Normalization divides by
    `ln(legal_count)`; a row with exactly one legal action is defined as 1.0
    (its forced distribution is uniform over its one-element support).
    """
    mask = legal_mask.to(torch.bool)
    probabilities = log_probabilities.exp()
    contributions = torch.where(
        mask, -probabilities * log_probabilities, torch.zeros_like(probabilities)
    )
    entropy = contributions.sum(dim=1)
    legal_counts = mask.sum(dim=1).to(entropy.dtype)
    max_entropy = torch.log(legal_counts)
    normalized = torch.where(
        legal_counts > 1, entropy / max_entropy.clamp(min=1e-12), torch.ones_like(entropy)
    )
    return float(entropy.mean()), float(normalized.mean())


def warmstart_batch_loss(
    outputs: ModelOutputs,
    *,
    legal_mask: torch.Tensor,
    policy_actions: torch.Tensor,
    policy_weights: torch.Tensor,
    value_targets: torch.Tensor,
    belief_targets: torch.Tensor,
    belief_mask: torch.Tensor,
    weights: WarmstartLossWeights,
) -> WarmstartBatchLoss:
    """The frozen Phase 8 batch loss over one `WarmstartBatch`'s targets.

    ```text
    L_policy = sum_i(w_i * CE_i) / sum_i(w_i)     (0 when the weights sum to 0)
    L_value  = mean CE over the batch
    L_belief = supervised-square CE sum / max(supervised squares, 1)
    L        = lambda_policy*L_policy + lambda_value*L_value
               + lambda_belief*L_belief
    ```

    Every tensor is validated against the batch implied by `outputs`; the
    teacher actions must be legal; weights must be finite and non-negative;
    value targets must be in class range. All violations raise.
    """
    batch = outputs.batch_size
    _require_shape(legal_mask, (batch, POLICY_LOGIT_COUNT), "legal_mask")
    _require_shape(policy_actions, (batch,), "policy_actions")
    _require_shape(policy_weights, (batch,), "policy_weights")
    _require_shape(value_targets, (batch,), "value_targets")

    weight = policy_weights.to(torch.float32)
    if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
        raise WarmstartLossError("policy weights must be finite and non-negative")

    targets = value_targets.to(torch.int64)
    if bool(((targets < 0) | (targets >= VALUE_CLASS_COUNT)).any()):
        raise WarmstartLossError(
            f"a value target is outside 0..{VALUE_CLASS_COUNT - 1}"
        )

    check_teacher_actions_legal(policy_actions, legal_mask)

    try:
        log_probabilities = masked_policy_log_probabilities(
            outputs.policy_logits, legal_mask
        )
    except ModelContractError as error:
        raise WarmstartLossError(str(error)) from error

    actions = policy_actions.to(torch.int64)
    per_example_ce = -log_probabilities.gather(1, actions[:, None]).squeeze(1)
    weighted_sum = (weight * per_example_ce).sum()
    weight_sum = weight.sum()
    # Branch-free zero-weight rule: when every weight is zero the numerator is
    # exactly zero too, so dividing by 1 instead of 0 yields the frozen
    # "contributes L_policy = 0" without detaching the graph.
    denominator = torch.where(weight_sum > 0, weight_sum, torch.ones_like(weight_sum))
    policy = weighted_sum / denominator

    try:
        value = value_loss(outputs.value_logits, targets)
        belief = belief_loss(outputs.belief_logits, belief_targets, belief_mask)
    except ModelContractError as error:
        raise WarmstartLossError(str(error)) from error

    total = (
        weights.lambda_policy * policy
        + weights.lambda_value * value
        + weights.lambda_belief * belief
    )
    entropy, normalized_entropy = legal_policy_entropy(
        log_probabilities.detach(), legal_mask
    )
    return WarmstartBatchLoss(
        total=total,
        policy=policy,
        value=value,
        belief=belief,
        weights=weights,
        batch_size=batch,
        policy_supervised_decisions=int((weight > 0).sum()),
        policy_weight_sum=float(weight_sum.detach()),
        value_decisions=batch,
        belief_supervised_pieces=int(belief_mask.to(torch.bool).sum()),
        legal_policy_entropy=entropy,
        legal_policy_entropy_normalized=normalized_entropy,
    )


def loss_semantics_summary() -> dict:
    """The serializable statement of this module's frozen semantics."""
    return {
        "loss_version": WARMSTART_LOSS_VERSION,
        "policy": (
            "masked_policy_log_probabilities (illegal fill -1e9), per-example "
            "CE at the teacher action, sum(w_i*CE_i)/sum(w_i); zero weight sum "
            "contributes exactly 0"
        ),
        "value": "stratego.model.losses.value_loss — mean categorical CE",
        "belief": (
            "stratego.model.losses.belief_loss — supervised-square CE sum / "
            "max(supervised squares, 1)"
        ),
        "combination": (
            "lambda_policy*L_policy + lambda_value*L_value + lambda_belief*L_belief"
        ),
        "teacher_action_rule": "an illegal teacher action raises WarmstartLossError",
        "illegal_logit_rule": (
            "illegal logits are replaced by the frozen -1e9 fill before "
            "normalization, so their stored values cannot affect the loss"
        ),
    }


__all__ = [
    "WARMSTART_LOSS_VERSION",
    "WarmstartBatchLoss",
    "WarmstartLossError",
    "WarmstartLossWeights",
    "check_teacher_actions_legal",
    "legal_policy_entropy",
    "loss_semantics_summary",
    "warmstart_batch_loss",
]
