"""Phase 11 Agent 2: the privileged belief evaluator.

Specification sources:

- `02_AGENT_2_BELIEF_EVALUATOR_BASELINES_VALIDATION.md` sections 1 and 6
- Agent 1's `phase11_belief_metrics_v1`

This is the only Phase 11 module that reads a true rank, and it reads one
only from a truth shard that a completed replay wrote after every learned
and baseline vector already existed. It scores; it returns numbers; it has
no path back into inference, into the sampler, or into any threshold. That
one-way property is what makes the validation evidence admissible.

Aggregation, exactly as frozen
------------------------------
```text
per event    ce, top1, brier, entropy, true-rank probability, confidence
per case     unweighted mean over the case's events, both colour games pooled
overall      unweighted mean over case aggregates (equal case weight)
delta        per-case learned minus baseline, then the mean of those
R_CE         mean(case ce_learned) / mean(case ce_baseline)
ECE          pooled events, 15 equal-width bins — never a case mean
CI           10,000-replicate case percentile bootstrap, 95%, one
             domain-separated PCG64 stream per metric token
```

Cases with no events
--------------------
A case contributes an aggregate only if it has at least one prediction
event. A game in which the observer never acts — the opponent moves first
and ends the game immediately — produces none, and a case whose *both*
games do that has no defined mean. Such cases are excluded from the case
mean and from resampling, and counted in `cases_without_events`. This is
arithmetic, not selection: the rule is fixed here, applies identically to
every metric and every stratum, and removes no case for anything it
scored.
"""

from __future__ import annotations

import math

import numpy as np

from ..training.phase11_contract import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_REPLICATES,
    ECE_SPECIFICATION,
    LOG_PROBABILITY_FLOOR,
    OVERALL_METRIC_TOKENS,
    Phase11ContractError,
    RANK_COUNT,
    RANK_NAMES,
    progress_bucket,
)
from ..training.phase11_seed import (
    OPPONENT_STRATA,
    SETUP_SOURCES,
    bootstrap_stream_seed,
)
from .statistics import quantile

#: The evaluator identity recorded on every artifact.
EVALUATOR_VERSION = "phase11_belief_evaluator_v1"

#: The report-only diagnostic slices, in the frozen order.
SLICE_KEYS = (
    "opponent_stratum",
    "observer_color",
    "progress_bucket",
    "piece_moved",
    "true_rank",
    "opponent_setup_source",
)

_ECE_BINS = int(ECE_SPECIFICATION["bins"])


class Phase11EvaluatorError(Phase11ContractError):
    """A metric could not be computed, or an input failed its checks."""


# ---------------------------------------------------------------------------
# Per-event scoring
# ---------------------------------------------------------------------------


def score_matrix(probabilities: np.ndarray, true_rank: np.ndarray) -> dict:
    """The frozen per-event metrics for one `[n, 12]` probability block."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != RANK_COUNT:
        raise Phase11EvaluatorError(
            f"probabilities have shape {probabilities.shape}, expected (n, {RANK_COUNT})"
        )
    true_rank = np.asarray(true_rank, dtype=np.int64)
    if true_rank.shape != (probabilities.shape[0],):
        raise Phase11EvaluatorError("true ranks and probabilities disagree in length")
    if true_rank.size and (true_rank.min() < 0 or true_rank.max() >= RANK_COUNT):
        raise Phase11EvaluatorError("a true rank index is outside 0..11")

    rows = np.arange(probabilities.shape[0])
    true_probability = probabilities[rows, true_rank]
    floored = np.maximum(true_probability, LOG_PROBABILITY_FLOOR)
    ce = -np.log(floored)
    argmax = probabilities.argmax(axis=1)  # first occurrence wins ties
    top1 = (argmax == true_rank).astype(np.float64)
    onehot = np.zeros_like(probabilities)
    onehot[rows, true_rank] = 1.0
    brier = ((probabilities - onehot) ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0)
    entropy = -terms.sum(axis=1)
    confidence = probabilities.max(axis=1)
    return {
        "ce": ce,
        "top1": top1,
        "brier": brier,
        "entropy": entropy,
        "true_rank_probability": true_probability,
        "confidence": confidence,
        "log_floor_events": int((true_probability < LOG_PROBABILITY_FLOOR).sum()),
        "argmax": argmax,
    }


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray) -> dict:
    """The frozen pooled-event ECE: 15 equal-width bins, equal event weight."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    total = confidence.size
    if total == 0:
        return {"ece": float("nan"), "events": 0, "bins": []}
    index = np.minimum((confidence * _ECE_BINS).astype(np.int64), _ECE_BINS - 1)
    index = np.maximum(index, 0)
    ece = 0.0
    bins = []
    for bin_index in range(_ECE_BINS):
        members = index == bin_index
        count = int(members.sum())
        if count == 0:
            bins.append(
                {
                    "bin": bin_index,
                    "lower": bin_index / _ECE_BINS,
                    "upper": (bin_index + 1) / _ECE_BINS,
                    "events": 0,
                    "confidence": None,
                    "accuracy": None,
                }
            )
            continue
        mean_confidence = float(confidence[members].mean())
        accuracy = float(correct[members].mean())
        ece += (count / total) * abs(accuracy - mean_confidence)
        bins.append(
            {
                "bin": bin_index,
                "lower": bin_index / _ECE_BINS,
                "upper": (bin_index + 1) / _ECE_BINS,
                "events": count,
                "confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {"ece": float(ece), "events": total, "bins": bins}


# ---------------------------------------------------------------------------
# The scored event table
# ---------------------------------------------------------------------------


class Phase11ScoredEvents:
    """Every scored prediction event of one bank, in one columnar table."""

    def __init__(self, columns: dict, case_ids: "list[str]") -> None:
        self.columns = columns
        self.case_ids = list(case_ids)
        self.case_count = len(case_ids)
        sizes = {name: len(value) for name, value in columns.items()}
        if len(set(sizes.values())) > 1:
            raise Phase11EvaluatorError(f"ragged event table: {sizes}")
        self.events = next(iter(sizes.values())) if sizes else 0

    # -- case aggregation -------------------------------------------------

    def case_means(self, values: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
        """`(case_index, case_mean)` over the cases that have events."""
        case = self.columns["case_index"]
        totals = np.bincount(case, weights=values, minlength=self.case_count)
        counts = np.bincount(case, minlength=self.case_count)
        present = np.nonzero(counts)[0]
        return present, totals[present] / counts[present]

    def case_event_counts(self) -> np.ndarray:
        return np.bincount(self.columns["case_index"], minlength=self.case_count)

    def mask_for(self, **selectors) -> np.ndarray:
        mask = np.ones(self.events, dtype=bool)
        for name, value in selectors.items():
            mask &= self.columns[name] == value
        return mask


def build_scored_events(rows: "list[dict]") -> Phase11ScoredEvents:
    """Assemble the columnar table from per-game scored blocks.

    Each block is one game's arrays plus its case/stratum identity. The
    order of blocks is the caller's deterministic game order; nothing in a
    metric depends on it, which the audit path re-checks by shuffling.
    """
    case_ids: list[str] = []
    case_lookup: dict[str, int] = {}
    for row in rows:
        if row["case_id"] not in case_lookup:
            case_lookup[row["case_id"]] = len(case_ids)
            case_ids.append(row["case_id"])

    columns: dict[str, list] = {name: [] for name in (
        "case_index",
        "stratum_index",
        "source_index",
        "observer_index",
        "bucket_index",
        "piece_moved",
        "true_rank",
        "ce_learned",
        "ce_baseline",
        "top1_learned",
        "top1_baseline",
        "brier_learned",
        "brier_baseline",
        "entropy_learned",
        "entropy_baseline",
        "true_rank_probability_learned",
        "true_rank_probability_baseline",
        "confidence_learned",
        "confidence_baseline",
    )}
    for row in rows:
        size = row["true_rank"].size
        if size == 0:
            continue
        columns["case_index"].append(np.full(size, case_lookup[row["case_id"]], dtype=np.int32))
        columns["stratum_index"].append(
            np.full(size, OPPONENT_STRATA.index(row["opponent_stratum"]), dtype=np.int8)
        )
        columns["source_index"].append(
            np.full(size, SETUP_SOURCES.index(row["opponent_setup_source"]), dtype=np.int8)
        )
        columns["observer_index"].append(
            np.full(size, 0 if row["observer_color"] == "red" else 1, dtype=np.int8)
        )
        columns["bucket_index"].append(row["bucket_index"])
        columns["piece_moved"].append(row["piece_moved"])
        columns["true_rank"].append(row["true_rank"])
        for side in ("learned", "baseline"):
            scores = row[side]
            columns[f"ce_{side}"].append(scores["ce"])
            columns[f"top1_{side}"].append(scores["top1"])
            columns[f"brier_{side}"].append(scores["brier"])
            columns[f"entropy_{side}"].append(scores["entropy"])
            columns[f"true_rank_probability_{side}"].append(scores["true_rank_probability"])
            columns[f"confidence_{side}"].append(scores["confidence"])

    # An empty column must keep its dtype: `case_index` is an index, and a
    # float64 zero-length array cannot be bincounted.
    empty_dtype = {
        "case_index": np.int32,
        "stratum_index": np.int8,
        "source_index": np.int8,
        "observer_index": np.int8,
        "bucket_index": np.int8,
        "piece_moved": np.uint8,
        "true_rank": np.int64,
    }
    assembled = {
        name: (
            np.concatenate(values)
            if values
            else np.zeros(0, dtype=empty_dtype.get(name, np.float64))
        )
        for name, values in columns.items()
    }
    return Phase11ScoredEvents(assembled, case_ids)


# ---------------------------------------------------------------------------
# The frozen case bootstrap
# ---------------------------------------------------------------------------


def _percentile_interval(replicates: np.ndarray) -> dict:
    ordered = np.sort(np.asarray(replicates, dtype=np.float64))
    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    values = ordered.tolist()
    return {
        "lower": quantile(values, alpha),
        "upper": quantile(values, 1.0 - alpha),
        "replicates": int(ordered.size),
        "confidence": BOOTSTRAP_CONFIDENCE,
    }


def _resample_index(seed: int, case_count: int, replicates: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, case_count, size=(replicates, case_count))


def bootstrap_mean(
    case_values: np.ndarray, bank: str, metric_token: str, *, replicates: int = BOOTSTRAP_REPLICATES
) -> dict:
    """Percentile CI of the mean of case aggregates, on its own stream."""
    case_values = np.asarray(case_values, dtype=np.float64)
    if case_values.size == 0:
        return {
            "point": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "cases": 0,
            "metric_token": metric_token,
        }
    seed = bootstrap_stream_seed(bank, metric_token)
    index = _resample_index(seed, case_values.size, replicates)
    statistics = case_values[index].mean(axis=1)
    interval = _percentile_interval(statistics)
    interval.update(
        {
            "point": float(case_values.mean()),
            "cases": int(case_values.size),
            "metric_token": metric_token,
            "stream_seed": seed,
        }
    )
    return interval


def bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    bank: str,
    metric_token: str,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    """Percentile CI of `mean(numerator) / mean(denominator)`.

    Both case aggregates are resampled *together* under one index draw, as
    the frozen ratio rule requires: the ratio is recomputed inside every
    replicate, never assembled from two independent resamplings.
    """
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if numerator.shape != denominator.shape:
        raise Phase11EvaluatorError("ratio aggregates disagree in length")
    if numerator.size == 0:
        return {
            "point": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "cases": 0,
            "metric_token": metric_token,
        }
    seed = bootstrap_stream_seed(bank, metric_token)
    index = _resample_index(seed, numerator.size, replicates)
    statistics = numerator[index].mean(axis=1) / denominator[index].mean(axis=1)
    interval = _percentile_interval(statistics)
    interval.update(
        {
            "point": float(numerator.mean() / denominator.mean()),
            "cases": int(numerator.size),
            "metric_token": metric_token,
            "stream_seed": seed,
        }
    )
    return interval


# ---------------------------------------------------------------------------
# Overall and slice metrics
# ---------------------------------------------------------------------------


def _aligned_case_values(table: Phase11ScoredEvents, names, mask=None):
    """Case aggregates of several columns over one common case set."""
    if mask is None:
        subset = table.columns
        case = table.columns["case_index"]
    else:
        subset = {name: table.columns[name][mask] for name in names}
        case = table.columns["case_index"][mask]
    counts = np.bincount(case, minlength=table.case_count)
    present = np.nonzero(counts)[0]
    values = {}
    for name in names:
        totals = np.bincount(
            case, weights=np.asarray(subset[name], dtype=np.float64), minlength=table.case_count
        )
        values[name] = totals[present] / counts[present]
    return present, values


def overall_metrics(table: Phase11ScoredEvents, bank: str, *, token_suffix: str = "") -> dict:
    """The frozen overall metric block, with its intervals."""
    names = (
        "ce_learned",
        "ce_baseline",
        "top1_learned",
        "top1_baseline",
        "brier_learned",
        "brier_baseline",
        "entropy_learned",
        "entropy_baseline",
        "true_rank_probability_learned",
        "true_rank_probability_baseline",
    )
    present, case_values = _aligned_case_values(table, names)
    deltas = {
        "ce_delta": case_values["ce_learned"] - case_values["ce_baseline"],
        "top1_delta": case_values["top1_learned"] - case_values["top1_baseline"],
        "brier_delta": case_values["brier_learned"] - case_values["brier_baseline"],
    }

    def token(name: str) -> str:
        return f"{name}{token_suffix}"

    metrics = {
        name: bootstrap_mean(values, bank, token(name))
        for name, values in list(case_values.items()) + list(deltas.items())
    }
    metrics["r_ce"] = bootstrap_ratio(
        case_values["ce_learned"], case_values["ce_baseline"], bank, token("r_ce")
    )
    missing = [name for name in OVERALL_METRIC_TOKENS if name not in metrics]
    if missing:
        raise Phase11EvaluatorError(f"metric block is missing {missing}")

    learned_ece = expected_calibration_error(
        table.columns["confidence_learned"], table.columns["top1_learned"]
    )
    baseline_ece = expected_calibration_error(
        table.columns["confidence_baseline"], table.columns["top1_baseline"]
    )
    counts = table.case_event_counts()
    return {
        "events": int(table.events),
        "cases_with_events": int(present.size),
        "cases_without_events": int((counts == 0).sum()),
        "metrics": metrics,
        "ece_learned": learned_ece,
        "ece_baseline": baseline_ece,
    }


def slice_metrics(table: Phase11ScoredEvents, bank: str) -> dict:
    """Every required diagnostic slice.

    Stratum slices aggregate cases exactly as the overall metrics do,
    because a gate reads them (Gate D's per-stratum `R_CE`). Every other
    slice is the frozen report-only pooled-event mean.
    """
    slices: dict = {}

    stratum_block = {}
    for index, stratum in enumerate(OPPONENT_STRATA):
        mask = table.columns["stratum_index"] == index
        stratum_block[stratum] = _stratum_metrics(table, bank, stratum, mask)
    slices["opponent_stratum"] = stratum_block

    slices["observer_color"] = {
        color: _pooled(table, table.columns["observer_index"] == index)
        for index, color in enumerate(("red", "blue"))
    }
    slices["progress_bucket"] = {
        bucket: _pooled(table, table.columns["bucket_index"] == index)
        for index, bucket in enumerate(("early", "middle", "late"))
    }
    slices["piece_moved"] = {
        label: _pooled(table, table.columns["piece_moved"] == value)
        for label, value in (("unmoved", 0), ("moved", 1))
    }
    slices["true_rank"] = {
        name: _pooled(table, table.columns["true_rank"] == index)
        for index, name in enumerate(RANK_NAMES)
    }
    slices["opponent_setup_source"] = {
        source: _pooled(table, table.columns["source_index"] == index)
        for index, source in enumerate(SETUP_SOURCES)
    }
    if tuple(slices) != SLICE_KEYS:
        raise Phase11EvaluatorError("diagnostic slices drifted from the frozen list")
    return slices


def _stratum_metrics(table, bank: str, stratum: str, mask: np.ndarray) -> dict:
    names = ("ce_learned", "ce_baseline", "top1_learned", "top1_baseline",
             "brier_learned", "brier_baseline")
    present, case_values = _aligned_case_values(table, names, mask)
    suffix = f"|st={stratum}"
    block = {
        "events": int(mask.sum()),
        "cases_with_events": int(present.size),
        "ce_learned": bootstrap_mean(case_values["ce_learned"], bank, f"ce_learned{suffix}"),
        "ce_baseline": bootstrap_mean(case_values["ce_baseline"], bank, f"ce_baseline{suffix}"),
        "ce_delta": bootstrap_mean(
            case_values["ce_learned"] - case_values["ce_baseline"], bank, f"ce_delta{suffix}"
        ),
        "top1_delta": bootstrap_mean(
            case_values["top1_learned"] - case_values["top1_baseline"],
            bank,
            f"top1_delta{suffix}",
        ),
        "brier_delta": bootstrap_mean(
            case_values["brier_learned"] - case_values["brier_baseline"],
            bank,
            f"brier_delta{suffix}",
        ),
        "r_ce": bootstrap_ratio(
            case_values["ce_learned"], case_values["ce_baseline"], bank, f"r_ce{suffix}"
        ),
        "ece_learned": expected_calibration_error(
            table.columns["confidence_learned"][mask], table.columns["top1_learned"][mask]
        ),
        "ece_baseline": expected_calibration_error(
            table.columns["confidence_baseline"][mask], table.columns["top1_baseline"][mask]
        ),
    }
    block["pooled"] = _pooled(table, mask)
    return block


def _pooled(table: Phase11ScoredEvents, mask: np.ndarray) -> dict:
    """A report-only pooled-event slice: no bootstrap, no gate."""
    events = int(mask.sum())
    if events == 0:
        return {"events": 0}
    block = {"events": events}
    for name in (
        "ce_learned",
        "ce_baseline",
        "top1_learned",
        "top1_baseline",
        "brier_learned",
        "brier_baseline",
        "entropy_learned",
        "entropy_baseline",
        "true_rank_probability_learned",
        "true_rank_probability_baseline",
    ):
        block[name] = float(np.asarray(table.columns[name])[mask].mean())
    block["r_ce"] = (
        block["ce_learned"] / block["ce_baseline"] if block["ce_baseline"] > 0 else float("nan")
    )
    block["top1_delta"] = block["top1_learned"] - block["top1_baseline"]
    block["brier_delta"] = block["brier_learned"] - block["brier_baseline"]
    block["ece_learned"] = expected_calibration_error(
        table.columns["confidence_learned"][mask], table.columns["top1_learned"][mask]
    )["ece"]
    return block


def all_finite(payload) -> "list[str]":
    """Every non-finite numeric leaf of a metric block, by path."""
    findings: list[str] = []

    def walk(node, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            if not math.isfinite(float(node)):
                findings.append(path)

    walk(payload, "")
    return findings


__all__ = [
    "EVALUATOR_VERSION",
    "Phase11EvaluatorError",
    "Phase11ScoredEvents",
    "SLICE_KEYS",
    "all_finite",
    "bootstrap_mean",
    "bootstrap_ratio",
    "build_scored_events",
    "expected_calibration_error",
    "overall_metrics",
    "score_matrix",
    "slice_metrics",
]
