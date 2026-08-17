"""Phase 9 Agent 5: the frozen RL objective, and nothing else.

Specification sources:

- `05_AGENT_5_PPO_TRAINER_AND_RESUME.md` ("PPO loss", "Value and belief
  losses", "Behavior KL", "Entropy", "Total loss")
- `00_PHASE_9_SEQUENCE_AND_COMMON_CONTRACT.md` ("PPO and damping", "Full loss
  and common optimizer constraints")
- Agent 1's `phase9_contract`: the clip epsilon, the loss weights, the KL
  direction/target/controller, the entropy endpoints and the log floor are all
  frozen there. This module *consumes* them; it restates no constant, so a
  tuned weight would have to be tuned where Agent 1's contract digest sees it.

The objective, verbatim:

```text
L = L_PPO + 0.5*L_value + 0.25*L_belief + beta*D_KL(pi_b || pi_theta)
    - c_H * H(pi_theta)
```

Populations
-----------
Two different populations, and confusing them is the mistake this module is
shaped to prevent:

```text
L_PPO                    advantage-filtered learner decisions (ppo_eligible)
L_value/L_belief/KL/H    every learner decision of the minibatch
```

The advantage filter narrows the *policy gradient only*. Value, belief, KL and
entropy see every learner-side example regardless of `ppo_eligible`, which is
why the eligibility flag is applied as a mask inside the PPO term rather than
by filtering the batch.

Opponent decisions never appear at all: Agent 4's train order is built from
learner sequences, so a rule, stress or historical-opponent decision is not a
member of the universe a minibatch is drawn from. That is a structural zero,
not a weight of zero — and :func:`phase9_batch_loss` still refuses a batch that
claims otherwise, because a structural guarantee nobody checks is a comment.

Frames
------
The network scores perspective-normalized squares; `trajectory_v1` stored the
behavior distribution ascending in the engine's *absolute* frame. The KL term
needs the two side by side, so :func:`behavior_probability_matrix` is the one
place they are reconciled — the same reconciliation Agent 3 made when it stored
the distribution, walked in the opposite direction. Nothing downstream of it
has to know that blue's frames differ at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..model.action_frame import absolute_action_to_model
from ..model.contract import (
    POLICY_LOGIT_COUNT,
    VALUE_CLASS_COUNT,
    ModelContractError,
    ModelOutputs,
)
from ..model.losses import belief_loss, masked_policy_log_probabilities
from .phase9_contract import (
    BEHAVIOR_LOG_EPSILON,
    BELIEF_LOSS_WEIGHT,
    PPO_CLIP_EPSILON,
    VALUE_LOSS_WEIGHT,
)
from .trajectory import PROBABILITY_SUM_TOLERANCE

#: Sum-to-one slack for a stored behavior distribution, inherited from
#: `trajectory_v1` — the tolerance those probabilities were validated under
#: when they were written.
BEHAVIOR_SUM_TOLERANCE = PROBABILITY_SUM_TOLERANCE


class Phase9LossError(RuntimeError):
    """A batch violated the frozen objective's preconditions. Never repaired."""


# ---------------------------------------------------------------------------
# Absolute -> model frame, vectorized
# ---------------------------------------------------------------------------


def _model_frame_tables() -> dict:
    """`{player: int64[10000]}` mapping absolute action -> model action.

    Built by calling the frozen public converter for every action of every
    player exactly once, so this table cannot drift from the mapping the rest
    of the system uses: if `absolute_action_to_model` ever changed, this table
    would change with it.
    """
    tables = {}
    for player in (0, 1):
        tables[player] = np.fromiter(
            (
                absolute_action_to_model(action, player)
                for action in range(POLICY_LOGIT_COUNT)
            ),
            dtype=np.int64,
            count=POLICY_LOGIT_COUNT,
        )
    return tables


_TO_MODEL_FRAME = _model_frame_tables()


def model_frame_table(player: int) -> np.ndarray:
    """The absolute -> model action table of one player, as an int64 array."""
    table = _TO_MODEL_FRAME.get(int(player))
    if table is None:
        raise Phase9LossError(f"learner side {player!r} is not a player")
    return table


def behavior_probability_matrix(examples) -> np.ndarray:
    """Dense `float32[B, 10000]` of `pi_b` in the model frame.

    One row per example, zero everywhere except the stored legal set. The
    entries are the stored float32 bytes themselves, moved — never recomputed,
    never renormalized: the sealed rollout is the authority on what the
    behavior policy did, exactly as it is for the PPO denominator.

    Built here, in whatever process assembles the batch, because it is pure
    array work over data the example already carries.
    """
    items = list(examples)
    if not items:
        raise Phase9LossError("cannot build a behavior matrix from no examples")
    matrix = np.zeros((len(items), POLICY_LOGIT_COUNT), dtype=np.float32)
    for row, example in enumerate(items):
        table = model_frame_table(example.learner_side)
        actions = np.asarray(example.behavior_legal_actions, dtype=np.int64)
        probabilities = np.asarray(
            example.behavior_legal_probabilities, dtype=np.float32
        )
        if actions.shape != probabilities.shape:
            raise Phase9LossError(
                f"{example.game_id} ply {example.decision_index}: "
                f"{actions.size} legal actions carry {probabilities.size} "
                "probabilities"
            )
        matrix[row, table[actions]] = probabilities
    return matrix


# ---------------------------------------------------------------------------
# The components
# ---------------------------------------------------------------------------


def soft_value_loss(
    value_logits: torch.Tensor, wdl_targets: torch.Tensor
) -> torch.Tensor:
    """Categorical cross-entropy against Agent 4's soft W/D/L lambda targets.

    `-sum_c Y_c * log softmax(z)_c`, meaned over the batch. Soft rather than
    hard-class because `Y_t = (1-lambda_V) P_{t+1} + lambda_V Y_{t+1}` is a
    distribution over three real outcomes; collapsing it to an argmax would
    discard exactly the blend the frozen target was defined to carry.
    """
    if value_logits.dim() != 2 or value_logits.shape[1] != VALUE_CLASS_COUNT:
        raise Phase9LossError(
            f"value logits must be [B, {VALUE_CLASS_COUNT}], got "
            f"{tuple(value_logits.shape)}"
        )
    if tuple(wdl_targets.shape) != tuple(value_logits.shape):
        raise Phase9LossError(
            f"WDL targets must match the value logits {tuple(value_logits.shape)}, "
            f"got {tuple(wdl_targets.shape)}"
        )
    targets = wdl_targets.to(torch.float32)
    if not bool(torch.isfinite(targets).all()) or bool((targets < 0).any()):
        raise Phase9LossError("a WDL target is negative or non-finite")
    sums = targets.sum(dim=1)
    if bool((sums - 1.0).abs().max() > BEHAVIOR_SUM_TOLERANCE):
        worst = float((sums - 1.0).abs().max())
        raise Phase9LossError(
            f"a WDL target row sums to 1 only within {worst:.3e}, outside the "
            f"{BEHAVIOR_SUM_TOLERANCE} simplex tolerance"
        )
    log_probabilities = F.log_softmax(value_logits.to(torch.float32), dim=1)
    return -(targets * log_probabilities).sum(dim=1).mean()


def behavior_kl_per_row(
    log_probabilities: torch.Tensor,
    behavior_probabilities: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """`D_KL(pi_b || pi_theta)` per decision, over the legal set.

    The Agent 1-frozen direction: the behavior policy is the reference and the
    learner is the argument, so a learner that abandons mass the behavior
    policy placed somewhere is penalized. Entries where `pi_b` is exactly zero
    contribute exactly zero — the float32 storage can round a very negative
    logit to 0.0, and `0 * log 0` must be the limit, not a NaN.
    """
    mask = legal_mask.to(torch.bool)
    behavior = behavior_probabilities.to(torch.float32)
    positive = behavior > 0
    log_behavior = torch.log(behavior.clamp(min=BEHAVIOR_LOG_EPSILON))
    contribution = behavior * (log_behavior - log_probabilities)
    return torch.where(
        positive & mask, contribution, torch.zeros_like(contribution)
    ).sum(dim=1)


def legal_entropy_per_row(
    log_probabilities: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    """`H(pi_theta)` per decision in nats, over the legal set, differentiable.

    Phase 8's `legal_policy_entropy` reports the same quantity as a float for
    metrics. Here entropy is a *term of the loss* and has to carry a gradient,
    so it is recomputed rather than reused.
    """
    mask = legal_mask.to(torch.bool)
    probabilities = log_probabilities.exp()
    contribution = -probabilities * log_probabilities
    return torch.where(mask, contribution, torch.zeros_like(contribution)).sum(dim=1)


# ---------------------------------------------------------------------------
# The batch result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase9BatchLoss:
    """One minibatch's frozen objective, every component reported separately.

    The six loss entries stay tensors (the total carries the graph); the
    diagnostics are plain numbers detached at construction, so a metrics
    consumer can hold this object without holding the graph.
    """

    total: torch.Tensor
    ppo: torch.Tensor
    value: torch.Tensor
    belief: torch.Tensor
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
    belief_supervised_squares: int
    entropy_normalized: float

    @property
    def components(self) -> tuple:
        return (self.total, self.ppo, self.value, self.belief, self.kl, self.entropy)

    def all_finite(self) -> bool:
        return all(bool(torch.isfinite(item).all()) for item in self.components)

    def to_dict(self) -> dict:
        return {
            "loss_total": float(self.total.detach()),
            "loss_ppo": float(self.ppo.detach()),
            "loss_value": float(self.value.detach()),
            "loss_belief": float(self.belief.detach()),
            "behavior_kl": float(self.kl.detach()),
            "policy_entropy": float(self.entropy.detach()),
            "kl_beta": float(self.kl_beta),
            "entropy_coefficient": float(self.entropy_coefficient),
            "value_weight": VALUE_LOSS_WEIGHT,
            "belief_weight": BELIEF_LOSS_WEIGHT,
            "batch_size": int(self.batch_size),
            "ppo_examples": int(self.ppo_examples),
            "ppo_clipped": int(self.ppo_clipped),
            "clip_fraction": float(self.clip_fraction),
            "ratio_mean": float(self.ratio_mean),
            "ratio_min": float(self.ratio_min),
            "ratio_max": float(self.ratio_max),
            "advantage_abs_mean": float(self.advantage_abs_mean),
            "belief_supervised_squares": int(self.belief_supervised_squares),
            "policy_entropy_normalized": float(self.entropy_normalized),
        }


def validate_behavior_matrix(
    behavior_probabilities: torch.Tensor,
    legal_mask: torch.Tensor,
    sampled_action_model: torch.Tensor,
    behavior_action_probability: torch.Tensor,
) -> None:
    """Every way the stored distribution and the batch could disagree.

    Each check is a different mix-up: mass outside the legal set means the two
    frames were reconciled wrongly; a row that does not sum to one means the
    stored distribution is not a distribution; a realized-action entry that
    differs from the separately stored scalar means the dense matrix and the
    PPO denominator came from different decisions. None of them is repairable
    at training time.
    """
    mask = legal_mask.to(torch.bool)
    behavior = behavior_probabilities.to(torch.float32)
    if behavior.shape != mask.shape:
        raise Phase9LossError(
            f"behavior matrix {tuple(behavior.shape)} does not match the legal "
            f"mask {tuple(mask.shape)}"
        )
    if not bool(torch.isfinite(behavior).all()) or bool((behavior < 0).any()):
        raise Phase9LossError("a stored behavior probability is negative or non-finite")
    outside = behavior.masked_fill(mask, 0.0)
    if bool((outside > 0).any()):
        rows = int((outside > 0).any(dim=1).sum())
        raise Phase9LossError(
            f"{rows} row(s) place behavior probability on an illegal action; the "
            "stored legal set and the model-frame mask describe different positions"
        )
    sums = behavior.sum(dim=1)
    if bool((sums - 1.0).abs().max() > BEHAVIOR_SUM_TOLERANCE):
        worst = float((sums - 1.0).abs().max())
        raise Phase9LossError(
            f"a stored behavior row sums to 1 only within {worst:.3e}, outside "
            f"the {BEHAVIOR_SUM_TOLERANCE} tolerance it was written under"
        )
    actions = sampled_action_model.to(torch.int64)
    if bool(((actions < 0) | (actions >= POLICY_LOGIT_COUNT)).any()):
        raise Phase9LossError("a sampled action identifier is outside 0..9999")
    legal_choice = mask.gather(1, actions[:, None]).squeeze(1)
    if not bool(legal_choice.all()):
        illegal = int((~legal_choice).sum())
        raise Phase9LossError(
            f"{illegal} realized action(s) are illegal under the batch's mask; "
            "refusing to train on them"
        )
    stored = behavior.gather(1, actions[:, None]).squeeze(1)
    scalar = behavior_action_probability.to(torch.float32)
    if not bool(torch.equal(stored, scalar)):
        worst = float((stored - scalar).abs().max())
        raise Phase9LossError(
            "the dense behavior matrix and the stored realized-action "
            f"probability disagree by up to {worst:.3e}; they must be the same "
            "float32 bytes from the same decision"
        )
    if bool((scalar <= 0).any()) or not bool(torch.isfinite(scalar).all()):
        bad = int(((scalar <= 0) | ~torch.isfinite(scalar)).sum())
        raise Phase9LossError(
            f"{bad} PPO denominator(s) are not finite and positive"
        )


def phase9_batch_loss(
    outputs: ModelOutputs,
    *,
    legal_mask: torch.Tensor,
    sampled_action_model: torch.Tensor,
    behavior_action_probability: torch.Tensor,
    behavior_probabilities: torch.Tensor,
    standardized_advantage: torch.Tensor,
    ppo_eligible: torch.Tensor,
    wdl_target: torch.Tensor,
    belief_target: torch.Tensor,
    belief_mask: torch.Tensor,
    kl_beta: float,
    entropy_coefficient: float,
) -> Phase9BatchLoss:
    """The frozen Phase 9 objective over one minibatch of learner decisions.

    ```text
    r_t      = pi_theta(a_t|s_t) / pi_b(a_t|s_t)
    L_PPO    = -mean_eligible[min(r*A, clip(r, 0.8, 1.2)*A)]
    L_value  = mean CE against the soft WDL lambda target
    L_belief = supervised-square CE (the frozen Phase 8 belief loss)
    D_KL     = mean_batch D_KL(pi_b || pi_theta) over legal actions
    H        = mean_batch legal-softmax entropy of pi_theta
    L        = L_PPO + 0.5*L_value + 0.25*L_belief + beta*D_KL - c_H*H
    ```

    Every logit is masked before normalization, so an illegal action can
    neither receive probability nor contribute a gradient. A minibatch with no
    PPO-eligible decision contributes `L_PPO = 0` — branch-free, so the graph
    stays connected and the remaining four terms are unaffected.
    """
    batch = int(outputs.batch_size)
    for tensor, shape, name in (
        (legal_mask, (batch, POLICY_LOGIT_COUNT), "legal_mask"),
        (behavior_probabilities, (batch, POLICY_LOGIT_COUNT), "behavior_probabilities"),
        (sampled_action_model, (batch,), "sampled_action_model"),
        (behavior_action_probability, (batch,), "behavior_action_probability"),
        (standardized_advantage, (batch,), "standardized_advantage"),
        (ppo_eligible, (batch,), "ppo_eligible"),
        (wdl_target, (batch, VALUE_CLASS_COUNT), "wdl_target"),
    ):
        if tuple(tensor.shape) != shape:
            raise Phase9LossError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
    if kl_beta < 0.0 or not np.isfinite(kl_beta):
        raise Phase9LossError(f"kl_beta must be finite and >= 0, got {kl_beta!r}")
    if entropy_coefficient < 0.0 or not np.isfinite(entropy_coefficient):
        raise Phase9LossError(
            f"entropy_coefficient must be finite and >= 0, got {entropy_coefficient!r}"
        )

    validate_behavior_matrix(
        behavior_probabilities,
        legal_mask,
        sampled_action_model,
        behavior_action_probability,
    )
    advantage = standardized_advantage.to(torch.float32)
    if not bool(torch.isfinite(advantage).all()):
        raise Phase9LossError("a standardized advantage is non-finite")

    try:
        log_probabilities = masked_policy_log_probabilities(
            outputs.policy_logits, legal_mask
        )
    except ModelContractError as error:
        raise Phase9LossError(str(error)) from error

    actions = sampled_action_model.to(torch.int64)
    log_theta = log_probabilities.gather(1, actions[:, None]).squeeze(1)
    log_behavior = torch.log(
        behavior_action_probability.to(torch.float32).clamp(min=BEHAVIOR_LOG_EPSILON)
    )
    ratio = torch.exp(log_theta - log_behavior)

    eligible = ppo_eligible.to(torch.bool)
    weight = eligible.to(torch.float32)
    eligible_count = weight.sum()
    clipped_ratio = ratio.clamp(
        1.0 - PPO_CLIP_EPSILON, 1.0 + PPO_CLIP_EPSILON
    )
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    # Branch-free empty-subset rule: with no eligible decision the numerator is
    # exactly zero, so dividing by 1 instead of 0 yields the frozen "L_PPO = 0"
    # without detaching the graph.
    denominator = torch.where(
        eligible_count > 0, eligible_count, torch.ones_like(eligible_count)
    )
    ppo = -(weight * surrogate).sum() / denominator

    try:
        belief = belief_loss(outputs.belief_logits, belief_target, belief_mask)
    except ModelContractError as error:
        raise Phase9LossError(str(error)) from error
    value = soft_value_loss(outputs.value_logits, wdl_target)

    kl_rows = behavior_kl_per_row(log_probabilities, behavior_probabilities, legal_mask)
    kl = kl_rows.mean()
    entropy_rows = legal_entropy_per_row(log_probabilities, legal_mask)
    entropy = entropy_rows.mean()

    total = (
        ppo
        + VALUE_LOSS_WEIGHT * value
        + BELIEF_LOSS_WEIGHT * belief
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
        return Phase9BatchLoss(
            total=total,
            ppo=ppo,
            value=value,
            belief=belief,
            kl=kl,
            entropy=entropy,
            kl_beta=float(kl_beta),
            entropy_coefficient=float(entropy_coefficient),
            batch_size=batch,
            ppo_examples=eligible_total,
            ppo_clipped=clipped_count,
            clip_fraction=(
                clipped_count / eligible_total if eligible_total else 0.0
            ),
            ratio_mean=float(selected.mean()) if eligible_total else 0.0,
            ratio_min=float(selected.min()) if eligible_total else 0.0,
            ratio_max=float(selected.max()) if eligible_total else 0.0,
            advantage_abs_mean=float(advantage.detach().abs().mean()),
            belief_supervised_squares=int(belief_mask.to(torch.bool).sum()),
            entropy_normalized=float(normalized.mean()),
        )


def loss_semantics() -> dict:
    """The serializable statement of this module's frozen semantics."""
    return {
        "objective": (
            "L = L_PPO + 0.5*L_value + 0.25*L_belief + beta*D_KL(pi_b||pi_theta) "
            "- c_H*H(pi_theta)"
        ),
        "ppo": {
            "population": "learner decisions with ppo_eligible=True only",
            "ratio": (
                "exp(log pi_theta(a) - log max(pi_b(a), 1e-12)); the denominator "
                "is the stored float32 probability of the realized action"
            ),
            "clip_epsilon": PPO_CLIP_EPSILON,
            "advantage": "the per-iteration standardized advantage from Agent 4",
            "empty_subset": "L_PPO = 0, computed branch-free so the graph survives",
            "clip_fraction": "|r - 1| > 0.20 over the eligible subset",
        },
        "value": {
            "population": "every learner decision of the minibatch",
            "form": "-sum_c Y_c log softmax(z)_c against the WDL lambda target",
            "weight": VALUE_LOSS_WEIGHT,
        },
        "belief": {
            "population": "every learner decision of the minibatch",
            "form": (
                "stratego.model.losses.belief_loss — the accepted Phase 8 "
                "hidden-only supervised-square cross-entropy"
            ),
            "weight": BELIEF_LOSS_WEIGHT,
        },
        "kl": {
            "population": "every learner decision of the minibatch",
            "direction": "D_KL(pi_b || pi_theta) over the legal set",
            "zero_mass_rule": "pi_b == 0 contributes exactly 0, never NaN",
            "storage_rule": (
                "the stored float32 distribution is used as written — never "
                "recomputed and never renormalized"
            ),
        },
        "entropy": {
            "population": "every learner decision of the minibatch",
            "form": "legal-softmax entropy in nats, differentiable",
            "sign": "subtracted, so a higher coefficient rewards exploration",
        },
        "masking": (
            "illegal logits are filled with -1e9 before log_softmax, so an "
            "illegal action can neither hold probability nor pass a gradient"
        ),
        "frames": (
            "the stored behavior distribution is absolute-frame ascending and "
            "is mapped into the model frame by the frozen "
            "absolute_action_to_model converter"
        ),
        "refusals": [
            "behavior mass on an illegal action",
            "a behavior row outside the sum-to-one tolerance",
            "a realized action illegal under the mask",
            "a dense-matrix entry that differs from the stored scalar",
            "a non-positive or non-finite PPO denominator",
            "a non-finite advantage",
            "a WDL target outside the simplex",
        ],
    }


__all__ = [
    "BEHAVIOR_SUM_TOLERANCE",
    "Phase9BatchLoss",
    "Phase9LossError",
    "behavior_kl_per_row",
    "behavior_probability_matrix",
    "legal_entropy_per_row",
    "loss_semantics",
    "model_frame_table",
    "phase9_batch_loss",
    "soft_value_loss",
    "validate_behavior_matrix",
]
