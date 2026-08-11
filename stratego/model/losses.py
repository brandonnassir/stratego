"""Placeholder multi-head losses for the Phase 5 autograd connectivity check.

Specification source: Phase 5 single-agent instructions, section 5.5.

.. warning::

   **These are connectivity probes, not a training objective.** Phase 5 is
   authorised to run exactly one controlled backward pass to prove that the
   shared encoder and all three heads receive gradients. The weights
   :data:`DEFAULT_VALUE_WEIGHT` and :data:`DEFAULT_BELIEF_WEIGHT` are not tuned
   and carry no claim; Phase 6 owns the real objective, its weighting and any
   auxiliary terms.

The combined form is

.. math::

    L = L_{policy} + \\lambda_v L_{value} + \\lambda_b L_{belief}.

Nothing here reads privileged state. Targets arrive as tensors that a caller
built elsewhere -- for belief, from
:mod:`stratego.training.belief_targets`, which is not importable from this
package's own imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .contract import (
    BELIEF_IGNORE_INDEX,
    POLICY_LOGIT_COUNT,
    ModelContractError,
    validate_belief_logits,
    validate_policy_logits,
    validate_value_logits,
)

#: Placeholder head weights. Untuned by construction; see the module warning.
DEFAULT_VALUE_WEIGHT = 1.0
DEFAULT_BELIEF_WEIGHT = 1.0

#: Finite stand-in for `-inf` when masking illegal actions inside a *loss*.
#:
#: Inference masks with true `-inf`, which is exact and harmless because nothing
#: differentiates it. In a loss, `-inf` entries make `log_softmax` produce
#: `-inf - -inf = NaN` gradients as soon as a row is dominated by masked
#: entries, so the loss path uses a large finite penalty instead. The difference
#: is numerically negligible (`exp(-1e9)` underflows to exactly zero in float32)
#: and it keeps every gradient finite.
MASKED_LOGIT_FILL = -1e9


def masked_policy_log_probabilities(
    policy_logits: torch.Tensor, legal_mask: "torch.Tensor | None" = None
) -> torch.Tensor:
    """`log_softmax` over the legal actions only. Returns `[B, 10000]`."""
    validate_policy_logits(policy_logits)
    logits = policy_logits.to(torch.float32)
    if legal_mask is not None:
        if legal_mask.shape != policy_logits.shape:
            raise ModelContractError(
                f"legal mask {tuple(legal_mask.shape)} does not match policy logits "
                f"{tuple(policy_logits.shape)}"
            )
        mask = legal_mask.to(torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ModelContractError("a legality mask row has no legal action at all")
        logits = logits.masked_fill(~mask, MASKED_LOGIT_FILL)
    return F.log_softmax(logits, dim=1)


def policy_loss(
    policy_logits: torch.Tensor,
    target_actions: torch.Tensor,
    legal_mask: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Negative log-likelihood of `target_actions` under the masked policy.

    `target_actions` is `int64[B]` holding engine action identifiers, so the
    target index is used directly with no remapping.
    """
    if target_actions.dim() != 1 or target_actions.shape[0] != policy_logits.shape[0]:
        raise ModelContractError(
            f"target actions must be int64[B={policy_logits.shape[0]}], got "
            f"{tuple(target_actions.shape)}"
        )
    targets = target_actions.to(torch.int64)
    if bool(((targets < 0) | (targets >= POLICY_LOGIT_COUNT)).any()):
        raise ModelContractError("a target action identifier is outside 0..9999")
    log_probabilities = masked_policy_log_probabilities(policy_logits, legal_mask)
    if legal_mask is not None:
        chosen_is_legal = legal_mask.to(torch.bool).gather(1, targets[:, None]).squeeze(1)
        if not bool(chosen_is_legal.all()):
            raise ModelContractError(
                "a policy target action is illegal under the supplied legality mask"
            )
    return -log_probabilities.gather(1, targets[:, None]).squeeze(1).mean()


def value_loss(value_logits: torch.Tensor, target_classes: torch.Tensor) -> torch.Tensor:
    """Categorical cross-entropy over WIN/DRAW/LOSS.

    Categorical rather than a scalar regression on purpose: the contract keeps
    three outcomes, and a draw is a real Stratego result rather than the midpoint
    between a win and a loss.
    """
    validate_value_logits(value_logits)
    if target_classes.dim() != 1 or target_classes.shape[0] != value_logits.shape[0]:
        raise ModelContractError(
            f"target classes must be int64[B={value_logits.shape[0]}], got "
            f"{tuple(target_classes.shape)}"
        )
    return F.cross_entropy(value_logits.to(torch.float32), target_classes.to(torch.int64))


def belief_loss(
    belief_logits: torch.Tensor,
    target_labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked per-square cross-entropy over the 12 piece types.

    `target_labels` is `int64[B, 100]` with `BELIEF_IGNORE_INDEX` on unsupervised
    squares; `loss_mask` is `bool[B, 100]`, true on exactly the supervised ones.
    Both are required, and they must agree -- the redundancy is deliberate, so a
    caller that builds one of them wrongly fails loudly instead of quietly
    training the belief head on empty squares.

    Normalisation is per supervised square, so a position with three hidden
    pieces and one with thirty contribute comparable per-square gradients. A
    batch with no supervised square at all returns a real zero that still carries
    a gradient path, which keeps the loss composable.
    """
    validate_belief_logits(belief_logits)
    batch, squares, types = belief_logits.shape
    if tuple(target_labels.shape) != (batch, squares):
        raise ModelContractError(
            f"belief labels must be int64[{batch}, {squares}], got {tuple(target_labels.shape)}"
        )
    if tuple(loss_mask.shape) != (batch, squares):
        raise ModelContractError(
            f"belief mask must be bool[{batch}, {squares}], got {tuple(loss_mask.shape)}"
        )

    labels = target_labels.to(torch.int64)
    mask = loss_mask.to(torch.bool)
    if not bool(torch.equal(mask, labels != BELIEF_IGNORE_INDEX)):
        raise ModelContractError(
            "belief mask and belief labels disagree: the mask must be true exactly "
            f"where labels differ from {BELIEF_IGNORE_INDEX}"
        )
    supervised = labels[mask]
    if supervised.numel() and bool(((supervised < 0) | (supervised >= types)).any()):
        raise ModelContractError(f"a supervised belief label is outside 0..{types - 1}")

    per_square = F.cross_entropy(
        belief_logits.to(torch.float32).reshape(batch * squares, types),
        labels.reshape(batch * squares),
        ignore_index=BELIEF_IGNORE_INDEX,
        reduction="none",
    ).reshape(batch, squares)
    # `ignore_index` already zeroes the excluded squares; multiplying by the mask
    # states the same thing explicitly and is what the mask test asserts against.
    masked = per_square * mask.to(per_square.dtype)
    supervised_count = mask.sum()
    return masked.sum() / supervised_count.clamp(min=1).to(masked.dtype)


@dataclass(frozen=True)
class MultiHeadLoss:
    """The combined loss and its three components, all scalar tensors."""

    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    belief: torch.Tensor
    value_weight: float
    belief_weight: float

    def all_finite(self) -> bool:
        return all(
            bool(torch.isfinite(component).all())
            for component in (self.total, self.policy, self.value, self.belief)
        )

    def to_dict(self) -> dict:
        return {
            "total": float(self.total.detach()),
            "policy": float(self.policy.detach()),
            "value": float(self.value.detach()),
            "belief": float(self.belief.detach()),
            "value_weight": self.value_weight,
            "belief_weight": self.belief_weight,
        }


def multi_head_loss(
    outputs,
    *,
    target_actions: torch.Tensor,
    legal_mask: "torch.Tensor | None",
    target_value_classes: torch.Tensor,
    belief_labels: torch.Tensor,
    belief_mask: torch.Tensor,
    value_weight: float = DEFAULT_VALUE_WEIGHT,
    belief_weight: float = DEFAULT_BELIEF_WEIGHT,
) -> MultiHeadLoss:
    """`L = L_policy + lambda_v * L_value + lambda_b * L_belief`.

    `outputs` is a :class:`~stratego.model.contract.ModelOutputs`.
    """
    policy = policy_loss(outputs.policy_logits, target_actions, legal_mask)
    value = value_loss(outputs.value_logits, target_value_classes)
    belief = belief_loss(outputs.belief_logits, belief_labels, belief_mask)
    total = policy + value_weight * value + belief_weight * belief
    return MultiHeadLoss(
        total=total,
        policy=policy,
        value=value,
        belief=belief,
        value_weight=float(value_weight),
        belief_weight=float(belief_weight),
    )


__all__ = [
    "DEFAULT_BELIEF_WEIGHT",
    "DEFAULT_VALUE_WEIGHT",
    "MASKED_LOGIT_FILL",
    "MultiHeadLoss",
    "belief_loss",
    "masked_policy_log_probabilities",
    "multi_head_loss",
    "policy_loss",
    "value_loss",
]
