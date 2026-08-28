"""Phase 17 Agent 2: the move objective, with a per-row value mask.

Specification sources: Agent 2 instruction sections 5 and 6, common contract
sections 4 and 9.

Why this file exists at all
---------------------------
`stratego.training.phase9_loss.phase9_batch_loss` is the accepted objective and
is not edited. Two Phase 17 requirements cannot be expressed through it:

```text
1  the value term is meaned over EVERY row and has no per-row loss mask, so a
   row whose W/D/L target should carry less weight -- or none -- cannot be in
   the batch at all. That is precisely why Phase 16 left partial emission off.
2  common contract section 4 disables the marginal belief auxiliary loss. The
   accepted objective's 0.25 belief weight is a module-level constant.
```

So the Phase 17 objective is assembled here from the accepted *components* --
`behavior_kl_per_row`, `legal_entropy_per_row`, `validate_behavior_matrix`,
`masked_policy_log_probabilities` -- with a masked value term written to reduce
to the accepted one. :func:`assert_value_loss_reduces` proves that reduction on
real numbers rather than asserting it in a docstring.

The belief head is structurally absent, not weighted to zero
------------------------------------------------------------
`belief_logits` are never read, so the head receives no gradient at all. That
is a stronger statement than a 0.0 coefficient -- a coefficient can be edited,
a term that is not in the graph cannot be re-enabled by a number -- and it is
what `test_move_loss_and_trainer` checks by looking for a gradient on
`belief_output.*` after a real backward pass.

Names, kept apart
-----------------
`c_H * H(pi_theta)` is an entropy BONUS. `beta * D_KL(pi_b || pi_theta)` is a
behavior-KL penalty in the FORWARD direction. The paper has neither: it has a
reverse KL to a magnet policy. All three are different objects and this module
never reports one under another's name.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ...model.contract import (
    POLICY_LOGIT_COUNT,
    VALUE_CLASS_COUNT,
    ModelContractError,
    ModelOutputs,
)
from ...model.losses import masked_policy_log_probabilities
from ..phase9_contract import (
    BEHAVIOR_LOG_EPSILON,
    PPO_CLIP_EPSILON,
    VALUE_LOSS_WEIGHT,
)
from ..phase9_loss import (
    BEHAVIOR_SUM_TOLERANCE,
    Phase9LossError,
    behavior_kl_per_row,
    legal_entropy_per_row,
    soft_value_loss,
    validate_behavior_matrix,
)
from .move_contract import (
    BELIEF_LOSS_WEIGHT,
    MOVE_LOSS_VERSION,
    PHASE9_ACCEPTED_BELIEF_LOSS_WEIGHT,
    Phase17MoveError,
)


class Phase17LossError(Phase17MoveError):
    """A Phase 17 minibatch is outside the move objective's contract."""


# ---------------------------------------------------------------------------
# The masked value term
# ---------------------------------------------------------------------------


def masked_soft_value_loss(
    value_logits: torch.Tensor,
    wdl_targets: torch.Tensor,
    row_weight: torch.Tensor,
) -> torch.Tensor:
    """Weighted categorical cross-entropy against the soft W/D/L targets.

    `sum_i w_i * CE_i / sum_i w_i`. With `w = 1` everywhere this is exactly the
    accepted `soft_value_loss`, which is the whole compatibility claim and is
    checked by :func:`assert_value_loss_reduces`.

    A batch whose weights are all zero contributes exactly zero, branch-free,
    so the graph stays connected and the other terms are unaffected.
    """
    if value_logits.dim() != 2 or value_logits.shape[1] != VALUE_CLASS_COUNT:
        raise Phase17LossError(
            f"value logits must be [B, {VALUE_CLASS_COUNT}], got "
            f"{tuple(value_logits.shape)}"
        )
    if tuple(wdl_targets.shape) != tuple(value_logits.shape):
        raise Phase17LossError(
            f"WDL targets must match the value logits {tuple(value_logits.shape)}, "
            f"got {tuple(wdl_targets.shape)}"
        )
    if tuple(row_weight.shape) != (value_logits.shape[0],):
        raise Phase17LossError(
            f"the value row weight must be [B] = [{value_logits.shape[0]}], got "
            f"{tuple(row_weight.shape)}"
        )
    targets = wdl_targets.to(torch.float32)
    if not bool(torch.isfinite(targets).all()) or bool((targets < 0).any()):
        raise Phase17LossError("a WDL target is negative or non-finite")
    sums = targets.sum(dim=1)
    if bool((sums - 1.0).abs().max() > BEHAVIOR_SUM_TOLERANCE):
        worst = float((sums - 1.0).abs().max())
        raise Phase17LossError(
            f"a WDL target row sums to 1 only within {worst:.3e}, outside the "
            f"{BEHAVIOR_SUM_TOLERANCE} simplex tolerance"
        )
    weight = row_weight.to(torch.float32)
    if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
        raise Phase17LossError("a value row weight is negative or non-finite")

    log_probabilities = F.log_softmax(value_logits.to(torch.float32), dim=1)
    per_row = -(targets * log_probabilities).sum(dim=1)
    total = weight.sum()
    denominator = torch.where(total > 0, total, torch.ones_like(total))
    return (weight * per_row).sum() / denominator


def assert_value_loss_reduces(
    value_logits: torch.Tensor, wdl_targets: torch.Tensor, *, tolerance: float = 1e-6
) -> dict:
    """Prove the masked term IS the accepted term when every weight is 1."""
    ones = torch.ones(value_logits.shape[0], dtype=torch.float32)
    ours = float(masked_soft_value_loss(value_logits, wdl_targets, ones).detach())
    theirs = float(soft_value_loss(value_logits, wdl_targets).detach())
    difference = abs(ours - theirs)
    if difference > tolerance:
        raise Phase17LossError(
            "the Phase 17 masked value term no longer reduces to the accepted "
            f"soft_value_loss: {ours!r} vs {theirs!r}"
        )
    return {
        "rows": int(value_logits.shape[0]),
        "phase17_masked": ours,
        "phase9_accepted": theirs,
        "difference": difference,
        "tolerance": tolerance,
        "reduces_to_accepted": True,
    }


# ---------------------------------------------------------------------------
# The batch result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase17BatchLoss:
    """One minibatch's Phase 17 objective, every component reported separately."""

    total: torch.Tensor
    ppo: torch.Tensor
    value: torch.Tensor
    kl: torch.Tensor
    entropy: torch.Tensor

    kl_beta: float
    entropy_coefficient: float

    batch_size: int
    ppo_examples: int
    ppo_clipped: int
    clip_fraction: float
    ratio_mean: float
    ratio_min: float
    ratio_max: float
    advantage_abs_mean: float
    entropy_normalized: float
    value_rows_weighted: float
    boundary_rows: int

    @property
    def components(self) -> tuple:
        return (self.total, self.ppo, self.value, self.kl, self.entropy)

    def all_finite(self) -> bool:
        return all(bool(torch.isfinite(item).all()) for item in self.components)

    def to_dict(self) -> dict:
        return {
            "loss_version": MOVE_LOSS_VERSION,
            "loss_total": float(self.total.detach()),
            "loss_ppo": float(self.ppo.detach()),
            "loss_value": float(self.value.detach()),
            "loss_belief": 0.0,
            "behavior_kl": float(self.kl.detach()),
            "policy_entropy": float(self.entropy.detach()),
            "kl_beta": float(self.kl_beta),
            "entropy_coefficient": float(self.entropy_coefficient),
            "value_weight": VALUE_LOSS_WEIGHT,
            "belief_weight": BELIEF_LOSS_WEIGHT,
            "belief_term_present": False,
            "batch_size": int(self.batch_size),
            "ppo_examples": int(self.ppo_examples),
            "ppo_clipped": int(self.ppo_clipped),
            "clip_fraction": float(self.clip_fraction),
            "ratio_mean": float(self.ratio_mean),
            "ratio_min": float(self.ratio_min),
            "ratio_max": float(self.ratio_max),
            "advantage_abs_mean": float(self.advantage_abs_mean),
            "policy_entropy_normalized": float(self.entropy_normalized),
            "value_rows_weighted": float(self.value_rows_weighted),
            "boundary_rows": int(self.boundary_rows),
        }


def phase17_batch_loss(
    outputs: ModelOutputs,
    *,
    legal_mask: torch.Tensor,
    sampled_action_model: torch.Tensor,
    behavior_action_probability: torch.Tensor,
    behavior_probabilities: torch.Tensor,
    standardized_advantage: torch.Tensor,
    ppo_eligible: torch.Tensor,
    wdl_target: torch.Tensor,
    kl_beta: float,
    entropy_coefficient: float,
    value_row_weight: "torch.Tensor | None" = None,
    boundary_rows: int = 0,
) -> Phase17BatchLoss:
    """The Phase 17 move objective over one minibatch of learner decisions.

    ```text
    r_t      = pi_theta(a_t|s_t) / pi_b(a_t|s_t)
    L_PPO    = -mean_eligible[min(r*A, clip(r, 0.8, 1.2)*A)]
    L_value  = weighted CE against the soft WDL lambda target
    D_KL     = mean_batch D_KL(pi_b || pi_theta) over legal actions   (FORWARD)
    H        = mean_batch legal-softmax entropy of pi_theta           (BONUS)
    L        = L_PPO + 0.5*L_value + beta*D_KL - c_H*H
    ```

    Two populations, and confusing them is the mistake this shape prevents:
    `L_PPO` sees the advantage-filtered subset; the value, KL and entropy terms
    see every row of the minibatch.
    """
    batch = int(outputs.batch_size)
    if value_row_weight is None:
        value_row_weight = torch.ones(batch, dtype=torch.float32)
    for tensor, shape, name in (
        (legal_mask, (batch, POLICY_LOGIT_COUNT), "legal_mask"),
        (behavior_probabilities, (batch, POLICY_LOGIT_COUNT), "behavior_probabilities"),
        (sampled_action_model, (batch,), "sampled_action_model"),
        (behavior_action_probability, (batch,), "behavior_action_probability"),
        (standardized_advantage, (batch,), "standardized_advantage"),
        (ppo_eligible, (batch,), "ppo_eligible"),
        (wdl_target, (batch, VALUE_CLASS_COUNT), "wdl_target"),
        (value_row_weight, (batch,), "value_row_weight"),
    ):
        if tuple(tensor.shape) != shape:
            raise Phase17LossError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
    if kl_beta < 0.0 or not np.isfinite(kl_beta):
        raise Phase17LossError(f"kl_beta must be finite and >= 0, got {kl_beta!r}")
    if entropy_coefficient < 0.0 or not np.isfinite(entropy_coefficient):
        raise Phase17LossError(
            f"entropy_coefficient must be finite and >= 0, got {entropy_coefficient!r}"
        )

    try:
        validate_behavior_matrix(
            behavior_probabilities,
            legal_mask,
            sampled_action_model,
            behavior_action_probability,
        )
    except Phase9LossError as error:
        raise Phase17LossError(str(error)) from error

    advantage = standardized_advantage.to(torch.float32)
    if not bool(torch.isfinite(advantage).all()):
        raise Phase17LossError("a standardized advantage is non-finite")

    try:
        log_probabilities = masked_policy_log_probabilities(
            outputs.policy_logits, legal_mask
        )
    except ModelContractError as error:
        raise Phase17LossError(str(error)) from error

    actions = sampled_action_model.to(torch.int64)
    log_theta = log_probabilities.gather(1, actions[:, None]).squeeze(1)
    log_behavior = torch.log(
        behavior_action_probability.to(torch.float32).clamp(min=BEHAVIOR_LOG_EPSILON)
    )
    ratio = torch.exp(log_theta - log_behavior)

    eligible = ppo_eligible.to(torch.bool)
    weight = eligible.to(torch.float32)
    eligible_count = weight.sum()
    clipped_ratio = ratio.clamp(1.0 - PPO_CLIP_EPSILON, 1.0 + PPO_CLIP_EPSILON)
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    denominator = torch.where(
        eligible_count > 0, eligible_count, torch.ones_like(eligible_count)
    )
    ppo = -(weight * surrogate).sum() / denominator

    value = masked_soft_value_loss(outputs.value_logits, wdl_target, value_row_weight)
    kl_rows = behavior_kl_per_row(log_probabilities, behavior_probabilities, legal_mask)
    kl = kl_rows.mean()
    entropy_rows = legal_entropy_per_row(log_probabilities, legal_mask)
    entropy = entropy_rows.mean()

    # The belief term is structurally absent: `outputs.belief_logits` is never
    # read, so the head receives no gradient. Common contract section 4.
    total = (
        ppo
        + VALUE_LOSS_WEIGHT * value
        + float(kl_beta) * kl
        - float(entropy_coefficient) * entropy
    )

    with torch.no_grad():
        detached_ratio = ratio.detach()
        clipped = ((detached_ratio - 1.0).abs() > PPO_CLIP_EPSILON) & eligible
        clipped_count = int(clipped.sum())
        eligible_total = int(eligible.sum())
        selected = detached_ratio[eligible]
        legal_counts = legal_mask.to(torch.bool).sum(dim=1).to(torch.float32)
        maximum = torch.log(legal_counts.clamp(min=1.0))
        normalized = torch.where(
            legal_counts > 1,
            entropy_rows.detach() / maximum.clamp(min=1e-12),
            torch.ones_like(entropy_rows),
        )
        return Phase17BatchLoss(
            total=total,
            ppo=ppo,
            value=value,
            kl=kl,
            entropy=entropy,
            kl_beta=float(kl_beta),
            entropy_coefficient=float(entropy_coefficient),
            batch_size=batch,
            ppo_examples=eligible_total,
            ppo_clipped=clipped_count,
            clip_fraction=(clipped_count / eligible_total if eligible_total else 0.0),
            ratio_mean=float(selected.mean()) if eligible_total else 0.0,
            ratio_min=float(selected.min()) if eligible_total else 0.0,
            ratio_max=float(selected.max()) if eligible_total else 0.0,
            advantage_abs_mean=float(advantage.detach().abs().mean()),
            entropy_normalized=float(normalized.mean()),
            value_rows_weighted=float(value_row_weight.to(torch.float32).sum()),
            boundary_rows=int(boundary_rows),
        )


def loss_semantics() -> dict:
    return {
        "loss_version": MOVE_LOSS_VERSION,
        "objective": (
            "L = L_PPO + 0.5*L_value + beta*D_KL(pi_b || pi_theta) - c_H*H(pi_theta)"
        ),
        "value_term": "per-row weighted; reduces to the accepted soft_value_loss at w = 1",
        "belief_term": "structurally absent; the head receives no gradient",
        "belief_weight": BELIEF_LOSS_WEIGHT,
        "phase9_accepted_belief_weight": PHASE9_ACCEPTED_BELIEF_LOSS_WEIGHT,
        "kl_direction": "FORWARD, D_KL(pi_behavior || pi_current), over the legal set",
        "entropy_term": "a BONUS subtracted from the loss; not a KL, not the paper's magnet",
        "populations": {
            "L_PPO": "the advantage-filtered subset",
            "value / KL / entropy": "every row of the minibatch",
        },
        "accepted_components_reused": [
            "stratego.training.phase9_loss.behavior_kl_per_row",
            "stratego.training.phase9_loss.legal_entropy_per_row",
            "stratego.training.phase9_loss.validate_behavior_matrix",
            "stratego.model.contract.masked_policy_log_probabilities",
        ],
        "accepted_objective_not_edited": "stratego.training.phase9_loss.phase9_batch_loss",
    }


__all__ = [
    "Phase17BatchLoss",
    "Phase17LossError",
    "assert_value_loss_reduces",
    "loss_semantics",
    "masked_soft_value_loss",
    "phase17_batch_loss",
]
