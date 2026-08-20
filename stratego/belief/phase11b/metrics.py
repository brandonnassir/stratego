"""Phase 11B shared metrics: CE, the count baseline, `R_CE`, top-1.

Specification source: `00_PHASE_11B_OVERVIEW.md` ("Shared Metrics") and
`01_AGENT_1_ATTACHED_BELIEF_HEAD.md` ("Evaluation").

One denominator, computed once
------------------------------
`R_CE = CE_candidate / CE_remaining_count_baseline`. The denominator is the
accepted `remaining_count_belief_v1` arithmetic — `c[r] * mask[r]`,
normalized — recomputed here directly from the corpus's stored public
arrays rather than re-entered by hand, and :func:`baseline_probabilities`
asserts the invariant the accepted module asserts: the true rank always has
positive mass, so the baseline CE is always finite.

Every candidate in the sprint divides by the *same* denominator on the
*same* development pieces, which is what makes the leaderboard a
comparison rather than four unrelated numbers.

The primary probability convention is the accepted one
------------------------------------------------------
The candidate's 12-vector is the raw float64 softmax of its logits: no
masking, no epsilon, full simplex. That is how the Phase 11 head was
measured and how the accepted sampler consumes a belief, so it is how a
replacement must be measured if `0.975` is to mean anything.

:func:`projected_probabilities` additionally renormalizes a candidate onto
the publicly legal support. It is reported as a **diagnostic only** — it is
a different interface contract, and mixing it into the headline number
would compare a masked candidate against an unmasked reference.
"""

from __future__ import annotations

import numpy as np

from .contract import CORPUS_STRATA, RANK_COUNT, Phase11BError

#: The metric-set identity every Phase 11B report names.
METRICS_VERSION = "phase11b_metrics_v1"

#: The accepted predictive baseline this sprint divides by.
BASELINE_VERSION = "remaining_count_belief_v1"

#: The Phase 11 head's development-set reference point, for orientation
#: only. `00_PHASE_11B_OVERVIEW.md`: "a reference point, not a hard gate".
PHASE11_HEAD_REFERENCE_R_CE = 0.975


class Phase11BMetricsError(Phase11BError):
    """A metric could not be computed, or violated one of its invariants."""


def piece_strata(data: dict) -> np.ndarray:
    """`int8[M]` — each hidden piece's behaviour stratum index."""
    counts = np.diff(np.asarray(data["piece_offset"], dtype=np.int64))
    return np.repeat(np.asarray(data["stratum"], dtype=np.int8), counts)


def piece_samples(data: dict) -> np.ndarray:
    """`int64[M]` — each hidden piece's sample row."""
    counts = np.diff(np.asarray(data["piece_offset"], dtype=np.int64))
    return np.repeat(np.arange(int(data["samples"]), dtype=np.int64), counts)


def piece_games(data: dict) -> np.ndarray:
    """`int64[M]` — each hidden piece's game ordinal, the bootstrap unit."""
    return np.asarray(data["game_ordinal"], dtype=np.int64)[piece_samples(data)]


def baseline_probabilities(data: dict) -> np.ndarray:
    """`float64[M, 12]` — the accepted remaining-count baseline per piece.

    `q[r] = c[r] * mask[r] / sum_r' c[r'] * mask[r']`, with `c` the stored
    public remaining inventory of the piece's decision and `mask` the
    stored public legal-rank mask of the piece.
    """
    counts = np.asarray(data["remaining_counts"], dtype=np.float64)[piece_samples(data)]
    mask = np.asarray(data["legal_rank_mask"], dtype=np.float64)
    weights = counts * mask
    totals = weights.sum(axis=1, keepdims=True)
    if not np.all(totals > 0):
        raise Phase11BMetricsError("a hidden piece has no publicly admissible rank")
    return weights / totals


def projected_probabilities(probabilities: np.ndarray, data: dict) -> np.ndarray:
    """A candidate's vector renormalized onto the publicly legal support.

    Diagnostic only. Zero mass on a rank the public state excludes, mass
    rescaled over the rest.
    """
    counts = np.asarray(data["remaining_counts"], dtype=np.float64)[piece_samples(data)]
    support = (counts > 0) & np.asarray(data["legal_rank_mask"], dtype=bool)
    projected = np.asarray(probabilities, dtype=np.float64) * support
    totals = projected.sum(axis=1, keepdims=True)
    # A candidate can in principle put no mass at all on the legal support.
    # Falling back to the baseline there keeps the diagnostic finite and
    # says so in the returned count rather than silently smoothing.
    empty = (totals <= 0).ravel()
    if empty.any():
        projected[empty] = baseline_probabilities(data)[empty]
        totals = projected.sum(axis=1, keepdims=True)
    return projected / totals


def cross_entropy(probabilities: np.ndarray, true_rank: np.ndarray) -> np.ndarray:
    """`float64[M]` — per-piece `-log p[true_rank]`."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != RANK_COUNT:
        raise Phase11BMetricsError(
            f"probabilities must be [M, {RANK_COUNT}], got {probabilities.shape}"
        )
    rows = np.arange(probabilities.shape[0])
    mass = probabilities[rows, np.asarray(true_rank, dtype=np.int64)]
    if np.any(mass <= 0):
        # Only a candidate can do this; the baseline cannot. Clamping at the
        # smallest positive float64 keeps one pathological row from erasing
        # the whole metric, and `zero_mass_rows` reports how often it happened.
        mass = np.maximum(mass, np.finfo(np.float64).tiny)
    return -np.log(mass)


def _bootstrap_interval(
    values: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple:
    """A percentile bootstrap of `R_CE`, resampling whole games."""
    unique, index = np.unique(groups, return_inverse=True)
    order = np.argsort(index, kind="stable")
    sorted_index = index[order]
    bounds = np.searchsorted(sorted_index, np.arange(len(unique) + 1))
    sums = np.add.reduceat(values[order], bounds[:-1])
    base_sums = np.add.reduceat(baseline[order], bounds[:-1])
    sizes = np.diff(bounds).astype(np.float64)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(resamples, len(unique)))
    numerator = sums[draws].sum(axis=1)
    denominator = base_sums[draws].sum(axis=1)
    ratios = numerator / denominator
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def evaluate(
    probabilities: np.ndarray,
    data: dict,
    *,
    baseline: "np.ndarray | None" = None,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 20260819,
) -> dict:
    """The shared metric block for one candidate on one stored split.

    `probabilities` is `[M, 12]` in the corpus's own piece order.
    """
    true_rank = np.asarray(data["true_rank"], dtype=np.int64)
    pieces = int(data["pieces"])
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (pieces, RANK_COUNT):
        raise Phase11BMetricsError(
            f"probabilities are {probabilities.shape}, expected {(pieces, RANK_COUNT)}"
        )
    if baseline is None:
        baseline = baseline_probabilities(data)

    candidate_ce = cross_entropy(probabilities, true_rank)
    baseline_ce = cross_entropy(baseline, true_rank)
    correct = probabilities.argmax(axis=1) == true_rank
    baseline_correct = baseline.argmax(axis=1) == true_rank

    overall_ce = float(candidate_ce.mean())
    overall_baseline = float(baseline_ce.mean())
    strata = piece_strata(data)
    per_stratum = {}
    for index, name in enumerate(CORPUS_STRATA):
        selection = strata == index
        if not selection.any():
            continue
        stratum_ce = float(candidate_ce[selection].mean())
        stratum_baseline = float(baseline_ce[selection].mean())
        per_stratum[name] = {
            "pieces": int(selection.sum()),
            "ce": stratum_ce,
            "baseline_ce": stratum_baseline,
            "r_ce": stratum_ce / stratum_baseline,
            "top1": float(correct[selection].mean()),
            "baseline_top1": float(baseline_correct[selection].mean()),
        }

    lower, upper = _bootstrap_interval(
        candidate_ce,
        baseline_ce,
        piece_games(data),
        resamples=int(bootstrap_resamples),
        seed=int(bootstrap_seed),
    )
    projected = projected_probabilities(probabilities, data)
    projected_ce = float(cross_entropy(projected, true_rank).mean())
    return {
        "metrics_version": METRICS_VERSION,
        "baseline_version": BASELINE_VERSION,
        "split": data.get("split"),
        "samples": int(data["samples"]),
        "pieces": pieces,
        "ce": overall_ce,
        "baseline_ce": overall_baseline,
        "r_ce": overall_ce / overall_baseline,
        "r_ce_ci95": [lower, upper],
        "top1": float(correct.mean()),
        "baseline_top1": float(baseline_correct.mean()),
        "zero_mass_rows": int(
            (probabilities[np.arange(pieces), true_rank] <= 0).sum()
        ),
        "strata": per_stratum,
        "worst_stratum": (
            max(per_stratum, key=lambda name: per_stratum[name]["r_ce"])
            if per_stratum
            else None
        ),
        "best_stratum": (
            min(per_stratum, key=lambda name: per_stratum[name]["r_ce"])
            if per_stratum
            else None
        ),
        "diagnostic_projected_ce": projected_ce,
        "diagnostic_projected_r_ce": projected_ce / overall_baseline,
    }


def paired_comparison(
    left: np.ndarray,
    right: np.ndarray,
    data: dict,
    *,
    resamples: int = 4000,
    seed: int = 20260819,
) -> dict:
    """A paired game-bootstrap of `CE(left) - CE(right)` on the same pieces.

    Two candidates scored on the same positions are far more comparable
    than their two marginal intervals suggest, because most of the variance
    is the position mix and cancels. This resamples whole games — the
    clustering unit — and reports the paired difference, which is the
    honest way to ask whether one candidate is really better than another.

    A negative difference means `left` has the lower cross-entropy.
    """
    true_rank = np.asarray(data["true_rank"], dtype=np.int64)
    difference = cross_entropy(left, true_rank) - cross_entropy(right, true_rank)
    groups = piece_games(data)
    unique, index = np.unique(groups, return_inverse=True)
    order = np.argsort(index, kind="stable")
    bounds = np.searchsorted(index[order], np.arange(len(unique) + 1))
    sums = np.add.reduceat(difference[order], bounds[:-1])
    sizes = np.diff(bounds).astype(np.float64)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(int(resamples), len(unique)))
    means = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    lower, upper = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    return {
        "ce_difference": float(difference.mean()),
        "ce_difference_ci95": [lower, upper],
        "left_lower_ce": bool(upper < 0.0),
        "distinguishable": bool(upper < 0.0 or lower > 0.0),
        "bootstrap_unit": "game",
        "games": int(len(unique)),
        "resamples": int(resamples),
    }


def uniform_reference(data: dict) -> dict:
    """The uninformed reference: a flat 12-way vector on every piece.

    Not a gate and not a competitor — a sanity floor, so a candidate that
    scores worse than "know nothing at all" is visible immediately.
    """
    pieces = int(data["pieces"])
    flat = np.full((pieces, RANK_COUNT), 1.0 / RANK_COUNT, dtype=np.float64)
    return evaluate(flat, data, bootstrap_resamples=200)


__all__ = [
    "BASELINE_VERSION",
    "METRICS_VERSION",
    "PHASE11_HEAD_REFERENCE_R_CE",
    "Phase11BMetricsError",
    "baseline_probabilities",
    "cross_entropy",
    "evaluate",
    "paired_comparison",
    "piece_games",
    "piece_samples",
    "piece_strata",
    "projected_probabilities",
    "uniform_reference",
]
