"""Phase 8 Agent 3: the three frozen pre-training baselines.

Specification sources:

- `00_PHASE_8_SEQUENCE_AND_COMMON_CONTRACT.md` section 17 (baselines frozen
  before training) and section 27 (game-level bootstrap)
- Agent 1's `warmstart_eval_v1` contract
  (:func:`stratego.training.warmstart_contract.evaluation_contract`)

The three baselines
-------------------
```text
policy   uniform over legal actions          CE = ln(legal_count)
value    one constant W/D/L train prior      CE = -ln(prior[target])
belief   observable unresolved-inventory     CE = -ln(U[t] / sum(U))
         marginal, per hidden piece
```

Everything here is arithmetic over sufficient statistics; nothing touches the
corpus, the engine, or a model. The belief marginal's inputs `U` are the
observer's *observable* unresolved counts — the same numbers the observation's
channels 56-67 carry — so the baseline is a function of exactly the
information the model input already exposes, never of privileged truth. The
privileged true type appears only where any cross-entropy needs its target.

Aggregation follows `warmstart_eval_v1` exactly: policy metrics weight each
example by its frozen supervision weight and normalize by the weight sum
(matching the training-loss normalization); value metrics are per selected
decision; belief metrics are per supervised hidden piece. Confidence
intervals resample *games*, with model-vs-baseline pairing left to the shared
index matrix.
"""

from __future__ import annotations

import numpy as np

from ..engine.constants import NUM_PIECE_TYPES, PIECE_COUNTS
from ..model.contract import VALUE_CLASS_COUNT, VALUE_CLASS_ORDER
from .warmstart_contract import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_REPLICATES,
    METRIC_LOG_EPSILON,
    WARMSTART_EVAL_VERSION,
)

#: Initial opponent inventory per piece type, as a vector indexed by type.
INITIAL_TYPE_COUNTS = np.array(
    [PIECE_COUNTS[piece_type] for piece_type in range(NUM_PIECE_TYPES)], dtype=np.int64
)

#: Observation channel block holding the normalized unresolved inventory.
UNRESOLVED_INVENTORY_CHANNEL = 56


class WarmstartBaselineError(RuntimeError):
    """A baseline was fitted or evaluated outside its frozen contract."""


def _safe_log(values: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(values, METRIC_LOG_EPSILON))


# ---------------------------------------------------------------------------
# Policy baseline: uniform over legal actions
# ---------------------------------------------------------------------------


def uniform_policy_metrics(legal_counts, weights) -> dict:
    """Weighted uniform-legal CE and expected top-1 over supervised examples.

    `legal_counts` and `weights` are per-example arrays over the
    policy-supervised population (weight > 0). The weighting mirrors the
    training-loss normalization, as `warmstart_eval_v1` freezes it.
    """
    counts = np.asarray(legal_counts, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if counts.shape != weight.shape:
        raise WarmstartBaselineError("legal counts and weights must align")
    if counts.size == 0:
        raise WarmstartBaselineError("the policy-supervised population is empty")
    if np.any(counts < 1):
        raise WarmstartBaselineError("a legal count below 1 is impossible")
    if np.any(weight <= 0):
        raise WarmstartBaselineError("the policy population must have weight > 0")
    total_weight = float(weight.sum())
    return {
        "cross_entropy": float((weight * np.log(counts)).sum() / total_weight),
        "expected_top1_accuracy": float((weight / counts).sum() / total_weight),
        "examples": int(counts.size),
        "weight_sum": total_weight,
    }


# ---------------------------------------------------------------------------
# Value baseline: one constant train-fitted W/D/L prior
# ---------------------------------------------------------------------------


def fit_value_prior(class_counts) -> tuple:
    """The frozen constant W/D/L distribution from train class counts."""
    counts = np.asarray(class_counts, dtype=np.int64)
    if counts.shape != (VALUE_CLASS_COUNT,):
        raise WarmstartBaselineError(
            f"class counts must have shape ({VALUE_CLASS_COUNT},), got {counts.shape}"
        )
    if counts.sum() <= 0:
        raise WarmstartBaselineError("cannot fit a value prior from zero examples")
    if np.any(counts < 0):
        raise WarmstartBaselineError("class counts cannot be negative")
    return tuple(float(value) for value in counts / counts.sum())


def value_prior_metrics(class_counts, prior) -> dict:
    """CE, Brier and accuracy of a constant prior against class counts.

    The predicted class of a constant distribution is fixed: the argmax with
    ties broken toward the lowest class index, exactly the frozen tie-break.
    """
    counts = np.asarray(class_counts, dtype=np.int64)
    probabilities = np.asarray(prior, dtype=np.float64)
    if counts.shape != (VALUE_CLASS_COUNT,) or probabilities.shape != (VALUE_CLASS_COUNT,):
        raise WarmstartBaselineError("class counts and prior must both have length 3")
    total = int(counts.sum())
    if total == 0:
        raise WarmstartBaselineError("cannot evaluate a prior on zero examples")
    log_prior = _safe_log(probabilities)
    cross_entropy = float(-(counts * log_prior).sum() / total)
    # Brier for a constant prediction: sum_k (p_k - onehot_k)^2, averaged with
    # the class frequencies as weights over the one-hot targets.
    per_class_brier = ((probabilities[None, :] - np.eye(VALUE_CLASS_COUNT)) ** 2).sum(axis=1)
    brier = float((counts * per_class_brier).sum() / total)
    predicted = int(np.argmax(probabilities))
    return {
        "cross_entropy": cross_entropy,
        "brier": brier,
        "accuracy": float(counts[predicted] / total),
        "predicted_class": VALUE_CLASS_ORDER[predicted],
        "examples": total,
        "class_frequencies": [float(value) for value in counts / total],
    }


# ---------------------------------------------------------------------------
# Belief baseline: observable unresolved-inventory marginal
# ---------------------------------------------------------------------------


def unresolved_counts_from_observation(observation: np.ndarray) -> np.ndarray:
    """The observer's unresolved opponent inventory, read off the model input.

    Channels 56-67 carry `U_T / N_T` as constant planes; multiplying back by
    the initial counts recovers the integer inventory. Reading it from the
    observation (rather than the privileged state) makes the baseline's
    observability claim literal: its inputs are bytes the model already sees.
    """
    planes = np.asarray(observation)
    if planes.ndim != 3:
        raise WarmstartBaselineError(f"expected a [C,10,10] observation, got {planes.shape}")
    normalized = planes[
        UNRESOLVED_INVENTORY_CHANNEL : UNRESOLVED_INVENTORY_CHANNEL + NUM_PIECE_TYPES, 0, 0
    ]
    counts = np.rint(normalized.astype(np.float64) * INITIAL_TYPE_COUNTS).astype(np.int64)
    if np.any(counts < 0) or np.any(counts > INITIAL_TYPE_COUNTS):
        raise WarmstartBaselineError("decoded unresolved counts are out of range")
    return counts


def belief_marginal(unresolved_counts) -> np.ndarray:
    """`p(type) = U_t / sum(U)` for one position's hidden opponent pieces."""
    counts = np.asarray(unresolved_counts, dtype=np.float64)
    if counts.shape != (NUM_PIECE_TYPES,):
        raise WarmstartBaselineError(
            f"unresolved counts must have shape ({NUM_PIECE_TYPES},), got {counts.shape}"
        )
    total = counts.sum()
    if total <= 0:
        raise WarmstartBaselineError("no unresolved pieces: the marginal is undefined")
    return counts / total


def belief_marginal_statistics(unresolved_counts, true_types) -> dict:
    """Per-position CE sum, top-1 hits and piece count for the belief prior.

    `true_types` are the privileged labels of that position's supervised
    pieces; they enter only as CE/accuracy *targets*. Predicted type is the
    marginal's argmax with ties broken toward the lowest type index.
    """
    marginal = belief_marginal(unresolved_counts)
    types = np.asarray(true_types, dtype=np.int64)
    if types.size == 0:
        return {"cross_entropy_sum": 0.0, "top1_hits": 0, "pieces": 0}
    if np.any((types < 0) | (types >= NUM_PIECE_TYPES)):
        raise WarmstartBaselineError("a true type index is out of range")
    predicted = int(np.argmax(marginal))
    return {
        "cross_entropy_sum": float(-_safe_log(marginal[types]).sum()),
        "top1_hits": int((types == predicted).sum()),
        "pieces": int(types.size),
    }


# ---------------------------------------------------------------------------
# Game-level bootstrap
# ---------------------------------------------------------------------------


def bootstrap_ratio_interval(
    numerators,
    denominators,
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    chunk: int = 500,
) -> dict:
    """Percentile CI of `sum(num)/sum(den)` under game resampling.

    `numerators`/`denominators` hold one entry per game — the game's summed
    contribution to the metric's numerator and denominator — so resampling
    rows resamples whole games, which is the frozen correlation-honest unit.
    The index matrix is drawn exactly as `warmstart_eval_v1` states
    (`default_rng(seed).integers(0, n_games, size=(replicates, n_games))`),
    generated in row chunks so ten thousand replicates never materialize at
    once.
    """
    numerator = np.asarray(numerators, dtype=np.float64)
    denominator = np.asarray(denominators, dtype=np.float64)
    if numerator.shape != denominator.shape or numerator.ndim != 1:
        raise WarmstartBaselineError("per-game numerators and denominators must align")
    games = numerator.size
    if games == 0:
        raise WarmstartBaselineError("cannot bootstrap zero games")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(replicates), dtype=np.float64)
    produced = 0
    while produced < replicates:
        rows = min(int(chunk), int(replicates) - produced)
        indices = rng.integers(0, games, size=(rows, games))
        sampled_denominator = denominator[indices].sum(axis=1)
        sampled_denominator[sampled_denominator == 0] = np.nan
        estimates[produced : produced + rows] = (
            numerator[indices].sum(axis=1) / sampled_denominator
        )
        produced += rows
    lower = float(np.nanpercentile(estimates, (1.0 - confidence) / 2.0 * 100.0))
    upper = float(np.nanpercentile(estimates, (1.0 + confidence) / 2.0 * 100.0))
    point = float(numerator.sum() / denominator.sum())
    return {
        "point": point,
        "lower": lower,
        "upper": upper,
        "confidence": float(confidence),
        "replicates": int(replicates),
        "games": int(games),
        "seed": int(seed),
        "eval_version": WARMSTART_EVAL_VERSION,
    }


__all__ = [
    "INITIAL_TYPE_COUNTS",
    "UNRESOLVED_INVENTORY_CHANNEL",
    "WarmstartBaselineError",
    "belief_marginal",
    "belief_marginal_statistics",
    "bootstrap_ratio_interval",
    "fit_value_prior",
    "uniform_policy_metrics",
    "unresolved_counts_from_observation",
    "value_prior_metrics",
]
