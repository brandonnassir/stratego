"""Phase 15 Agent 1 section 11: the metric block.

Specification source: `01_AGENT_1_BELIEF_HEAD_TRAINING.md` section 11.

```text
cross-entropy / NLL          Brier score
remaining-count baseline CE  expected calibration error
R_CE                         maximum calibration error
top-1 accuracy               raw vs calibrated
```

One denominator, computed once
------------------------------
`R_CE = CE_candidate / CE_remaining_count_baseline`. The denominator is the
accepted `remaining_count_belief_v1` arithmetic — `c[r] * mask[r]`,
normalized — recomputed from the corpus's own stored public arrays rather
than re-entered by hand. Every model compared in this phase divides by the
same denominator on the same development pieces, which is what makes a
comparison a comparison.

The probability convention is the accepted one
-----------------------------------------------
The candidate's 12-vector is the raw float64 softmax of its logits: no
masking, no epsilon, full simplex. That is how the accepted Phase 11 head
was measured and how the accepted sampler consumes a belief, so it is how a
replacement must be measured.

Calibration is measured on the top-1 confidence
------------------------------------------------
ECE and MCE bin pieces by `max_r p[r]` and compare mean confidence with
observed top-1 accuracy inside each bin — the standard multi-class
reliability construction. Temperature scaling cannot change top-1 labels,
so a calibrated model's accuracy is identical and only the confidence
axis moves.
"""

from __future__ import annotations

import numpy as np

from ...setups.families import FAMILY_KEYS
from .contract import (
    CORPUS_COLORS,
    GAME_BANDS,
    OPPONENTS,
    OPPONENT_CLASS,
    POLICY_SOURCES,
    RANK_COUNT,
    SETUP_SOURCES,
    game_band,
    Phase15Error,
)

#: The metric-set identity every Phase 15 report names.
METRICS_VERSION = "phase15_belief_metrics_v1"

#: The accepted predictive baseline this phase divides by.
BASELINE_VERSION = "remaining_count_belief_v1"

#: Reliability bins for ECE/MCE. Equal width on `[0, 1]`.
CALIBRATION_BINS = 15


class Phase15MetricsError(Phase15Error):
    """A metric could not be computed, or violated one of its invariants."""


# ---------------------------------------------------------------------------
# Piece-level views of a stored split
# ---------------------------------------------------------------------------


def piece_counts(data: dict) -> np.ndarray:
    return np.diff(np.asarray(data["piece_offset"], dtype=np.int64))


def piece_samples(data: dict) -> np.ndarray:
    """`int64[M]` — each hidden piece's sample row."""
    return np.repeat(np.arange(int(data["samples"]), dtype=np.int64), piece_counts(data))


def piece_games(data: dict) -> np.ndarray:
    """`int64[M]` — each hidden piece's game ordinal, the bootstrap unit."""
    return np.asarray(data["game_ordinal"], dtype=np.int64)[piece_samples(data)]


def piece_field(data: dict, name: str) -> np.ndarray:
    """A per-sample label array broadcast to per-piece."""
    return np.asarray(data[name])[piece_samples(data)]


def piece_bands(data: dict) -> np.ndarray:
    """`int8[M]` — each hidden piece's early/middle/late band index."""
    index = {name: position for position, name in enumerate(GAME_BANDS)}
    per_sample = np.asarray(
        [index[game_band(int(value))] for value in np.asarray(data["total_moves"])],
        dtype=np.int8,
    )
    return per_sample[piece_samples(data)]


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
        raise Phase15MetricsError("a hidden piece has no publicly admissible rank")
    return weights / totals


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------


def cross_entropy(probabilities: np.ndarray, true_rank: np.ndarray) -> np.ndarray:
    """`float64[M]` — per-piece `-log p[true_rank]`, the NLL."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != RANK_COUNT:
        raise Phase15MetricsError(
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


def brier(probabilities: np.ndarray, true_rank: np.ndarray) -> np.ndarray:
    """`float64[M]` — the multi-class Brier score of each piece.

    `sum_r (p[r] - 1[r == true])^2`, the standard multi-class form, which
    ranges over `[0, 2]`.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(probabilities.shape[0]), np.asarray(true_rank, dtype=np.int64)] = 1.0
    return ((probabilities - onehot) ** 2).sum(axis=1)


def calibration_error(
    probabilities: np.ndarray, true_rank: np.ndarray, *, bins: int = CALIBRATION_BINS
) -> dict:
    """Expected and maximum calibration error over top-1 confidence."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == np.asarray(true_rank, dtype=np.int64)).astype(np.float64)
    total = confidence.size
    if total == 0:  # pragma: no cover - guarded by callers
        raise Phase15MetricsError("no pieces to calibrate over")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    # `right=True` on all but the first bin, so confidence exactly 1.0 lands
    # in the last bin rather than falling outside every one of them.
    index = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, int(bins) - 1)
    expected = 0.0
    maximum = 0.0
    table = []
    for position in range(int(bins)):
        selection = index == position
        count = int(selection.sum())
        if not count:
            continue
        mean_confidence = float(confidence[selection].mean())
        accuracy = float(correct[selection].mean())
        gap = abs(mean_confidence - accuracy)
        expected += (count / total) * gap
        maximum = max(maximum, gap)
        table.append(
            {
                "bin": position,
                "lower": float(edges[position]),
                "upper": float(edges[position + 1]),
                "pieces": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "gap": gap,
            }
        )
    return {
        "bins": int(bins),
        "expected_calibration_error": float(expected),
        "maximum_calibration_error": float(maximum),
        "reliability": table,
    }


def _bootstrap_r_ce(
    values: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> "list[float]":
    """A percentile bootstrap of `R_CE`, resampling whole games."""
    unique, index = np.unique(groups, return_inverse=True)
    order = np.argsort(index, kind="stable")
    bounds = np.searchsorted(index[order], np.arange(len(unique) + 1))
    sums = np.add.reduceat(values[order], bounds[:-1])
    base_sums = np.add.reduceat(baseline[order], bounds[:-1])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(int(resamples), len(unique)))
    ratios = sums[draws].sum(axis=1) / base_sums[draws].sum(axis=1)
    return [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))]


def _block(
    probabilities: np.ndarray,
    baseline: np.ndarray,
    true_rank: np.ndarray,
    selection: np.ndarray,
) -> dict:
    """The metric block of one slice of the pieces."""
    candidate_ce = cross_entropy(probabilities[selection], true_rank[selection])
    baseline_ce = cross_entropy(baseline[selection], true_rank[selection])
    correct = probabilities[selection].argmax(axis=1) == true_rank[selection]
    baseline_correct = baseline[selection].argmax(axis=1) == true_rank[selection]
    mean_ce = float(candidate_ce.mean())
    mean_baseline = float(baseline_ce.mean())
    calibration = calibration_error(probabilities[selection], true_rank[selection])
    return {
        "pieces": int(selection.sum()),
        "ce": mean_ce,
        "nll": mean_ce,
        "baseline_ce": mean_baseline,
        "r_ce": mean_ce / mean_baseline,
        "top1": float(correct.mean()),
        "baseline_top1": float(baseline_correct.mean()),
        "brier": float(brier(probabilities[selection], true_rank[selection]).mean()),
        "baseline_brier": float(
            brier(baseline[selection], true_rank[selection]).mean()
        ),
        "expected_calibration_error": calibration["expected_calibration_error"],
        "maximum_calibration_error": calibration["maximum_calibration_error"],
    }


def breakdowns(
    probabilities: np.ndarray, baseline: np.ndarray, true_rank: np.ndarray, data: dict
) -> dict:
    """Every section 11 breakdown, over the same pieces."""
    result: dict = {}
    dimensions = (
        ("observer_color", piece_field(data, "observer_color"), CORPUS_COLORS),
        ("observer_source", piece_field(data, "observer_model"), POLICY_SOURCES),
        ("opponent", piece_field(data, "opponent"), OPPONENTS),
        ("setup_source", piece_field(data, "setup_source"), SETUP_SOURCES),
        ("opponent_setup_family", piece_field(data, "opponent_family"), FAMILY_KEYS),
        ("game_band", piece_bands(data), GAME_BANDS),
    )
    for name, codes, labels in dimensions:
        codes = np.asarray(codes)
        block = {}
        for position, label in enumerate(labels):
            selection = codes == position
            if not selection.any():
                continue
            block[label] = _block(probabilities, baseline, true_rank, selection)
        result[name] = block

    classes = np.asarray(piece_field(data, "opponent"))
    opponent_class = {}
    for label in sorted(set(OPPONENT_CLASS.values())):
        members = [
            position
            for position, name in enumerate(OPPONENTS)
            if OPPONENT_CLASS[name] == label
        ]
        selection = np.isin(classes, members)
        if selection.any():
            opponent_class[label] = _block(
                probabilities, baseline, true_rank, selection
            )
    result["opponent_class"] = opponent_class
    return result


def evaluate(
    probabilities: np.ndarray,
    data: dict,
    *,
    baseline: "np.ndarray | None" = None,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 20260824,
    with_breakdowns: bool = True,
) -> dict:
    """The full section 11 metric block for one model on one stored split.

    `probabilities` is `[M, 12]` in the corpus's own piece order, and
    `data` must have been loaded with `labels=True`.
    """
    if "true_rank" not in data:
        raise Phase15MetricsError(
            "evaluate needs the privileged labels; load the split with labels=True"
        )
    true_rank = np.asarray(data["true_rank"], dtype=np.int64)
    pieces = int(data["pieces"])
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (pieces, RANK_COUNT):
        raise Phase15MetricsError(
            f"probabilities are {probabilities.shape}, expected {(pieces, RANK_COUNT)}"
        )
    if not np.isfinite(probabilities).all():
        raise Phase15MetricsError("a probability is not finite")
    if baseline is None:
        baseline = baseline_probabilities(data)

    candidate_ce = cross_entropy(probabilities, true_rank)
    baseline_ce = cross_entropy(baseline, true_rank)
    overall = _block(probabilities, baseline, true_rank, np.ones(pieces, dtype=bool))
    calibration = calibration_error(probabilities, true_rank)
    block = {
        "metrics_version": METRICS_VERSION,
        "baseline_version": BASELINE_VERSION,
        "split": data.get("split"),
        "samples": int(data["samples"]),
        "games": int(data["games"]),
        **overall,
        "r_ce_ci95": _bootstrap_r_ce(
            candidate_ce,
            baseline_ce,
            piece_games(data),
            resamples=int(bootstrap_resamples),
            seed=int(bootstrap_seed),
        ),
        "zero_mass_rows": int(
            (probabilities[np.arange(pieces), true_rank] <= 0).sum()
        ),
        "probability_sum_max_deviation": float(
            np.abs(probabilities.sum(axis=1) - 1.0).max()
        ),
        "reliability": calibration["reliability"],
    }
    if with_breakdowns:
        block["breakdowns"] = breakdowns(probabilities, baseline, true_rank, data)
    return block


def paired_comparison(
    left: np.ndarray,
    right: np.ndarray,
    data: dict,
    *,
    resamples: int = 4000,
    seed: int = 20260824,
) -> dict:
    """A paired game-bootstrap of `CE(left) - CE(right)` on the same pieces.

    Two models scored on the same positions are far more comparable than
    their two marginal intervals suggest, because most of the variance is
    the position mix and cancels. A negative difference means `left` has the
    lower cross-entropy.
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

    Not a gate and not a competitor — a sanity floor, so a model that
    scores worse than "know nothing at all" is visible immediately.
    """
    pieces = int(data["pieces"])
    flat = np.full((pieces, RANK_COUNT), 1.0 / RANK_COUNT, dtype=np.float64)
    return evaluate(flat, data, bootstrap_resamples=200, with_breakdowns=False)


__all__ = [
    "BASELINE_VERSION",
    "CALIBRATION_BINS",
    "METRICS_VERSION",
    "Phase15MetricsError",
    "baseline_probabilities",
    "breakdowns",
    "brier",
    "calibration_error",
    "cross_entropy",
    "evaluate",
    "paired_comparison",
    "piece_bands",
    "piece_field",
    "piece_games",
    "piece_samples",
    "uniform_reference",
]
