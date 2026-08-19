"""Phase 11 Agent 5: independent recomputation from the primitive rows.

Specification source: `05_AGENT_5_INTEGRATED_VALIDATION_FREEZE.md`,
"Independent recomputation" — CE learned/baseline/ratio, the top-1 delta,
the Brier delta, ECE, the per-stratum CE ratios and *all* bootstrap
intervals, recomputed from primitive rows.

Why this module imports almost nothing
--------------------------------------
An audit that calls the thing it audits proves only that the call
succeeded. This module therefore imports **no** `phase11_*` module: not
`phase11_evaluator` (whose aggregation it re-derives), not
`phase11_seed` (whose stream derivation it re-derives), not
`phase11_contract` (whose constants it restates). It reads the recorded
shards with NumPy and rebuilds every quantity from the frozen written
specification.

The frozen constants below are restatements of Agent 1's, and
`test_phase11_recompute.py` asserts each one equals the contract's live
value. If a threshold, bin count, seed or floor ever moves in the
contract, that test fails here — which is the point: the two paths must
disagree loudly rather than drift together.

What "independent" means numerically
------------------------------------
- The per-event scores take a different arithmetic route: cross-entropy
  comes from a log-sum-exp of the raw logits rather than from a softmax
  followed by a log, and the Brier score is expanded to
  `sum(p^2) - 2*p[true] + 1` rather than summed over a one-hot difference.
- Case aggregation and every replicate mean are computed with
  :func:`math.fsum` over Python floats, not NumPy pairwise summation.
- The quantile is re-implemented from the frozen linear-interpolation
  rule.
- The resampling *index* is the frozen statistic itself — a PCG64 draw
  from a domain-separated seed — so it is re-derived (seed and draw) but
  not replaced. Replacing it would recompute a different interval, not the
  same one.

Agreement is therefore expected at float64 rounding level, and the caller
compares against a 1e-9 tolerance.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

#: The Agent 5 independent-recomputation path.
RECOMPUTE_VERSION = "phase11_independent_recompute_v1"

#: Agreement tolerance. The two paths differ only by summation order, so
#: any real disagreement is orders of magnitude above this.
RECOMPUTE_TOLERANCE = 1e-9

# --- restated frozen constants (asserted equal to the contract in tests) ---

_RANKS = 12
_LOG_FLOOR = 1e-12
_ECE_BINS = 15
_REPLICATES = 10_000
_CONFIDENCE = 0.95
_IDENTITY_VERSION = "phase11_identity_v1"
_PERSON = b"strat-b11"
_BOOTSTRAP_DOMAIN = "bootstrap"
_BOOTSTRAP_DOMAIN_ROOT = 2026081901
_BANK_BOOTSTRAP_ROOT = {"validation": 2026081907, "test": 2026081908}

#: The ten per-event columns the overall block aggregates, and the six a
#: stratum block aggregates. Restated from the frozen metrics document.
_OVERALL_METRIC_NAMES = (
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
_STRATUM_METRIC_NAMES = (
    "ce_learned",
    "ce_baseline",
    "top1_learned",
    "top1_baseline",
    "brier_learned",
    "brier_baseline",
)

_STRATA = (
    "phase9_selfplay",
    "phase8_anchor",
    "strategic_rule",
    "tactical_rule",
    "basic_rule",
    "information_miser",
    "scout_rush",
    "miner_rush",
)


class Phase11RecomputeError(RuntimeError):
    """The independent path could not rebuild a quantity."""


# ---------------------------------------------------------------------------
# Re-derived seeds and quantiles
# ---------------------------------------------------------------------------


def independent_seed(domain: str, *parts) -> int:
    """The Phase 11 stream seed, re-derived from the frozen written rule.

    Colon-joined `identity version : domain : domain root : parts`, blake2b
    with an 8-byte digest under the `strat-b11` personalization, taken as a
    big-endian integer shifted right one bit to 63 bits.
    """
    payload = ":".join(
        [_IDENTITY_VERSION, domain, str(_BOOTSTRAP_DOMAIN_ROOT), *[str(p) for p in parts]]
    )
    digest = hashlib.blake2b(payload.encode(), digest_size=8, person=_PERSON).digest()
    return int.from_bytes(digest, "big") >> 1


def independent_bootstrap_seed(bank: str, metric_token: str) -> int:
    if bank not in _BANK_BOOTSTRAP_ROOT:
        raise Phase11RecomputeError(f"no bootstrap root for bank {bank!r}")
    return independent_seed(
        _BOOTSTRAP_DOMAIN, _BANK_BOOTSTRAP_ROOT[bank], bank, metric_token
    )


def independent_quantile(sorted_values, probability: float) -> float:
    """The frozen linear-interpolation quantile, re-implemented."""
    count = len(sorted_values)
    if count == 0:
        raise Phase11RecomputeError("quantile of an empty sample")
    if count == 1:
        return float(sorted_values[0])
    position = probability * (count - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


# ---------------------------------------------------------------------------
# Per-event scores, by a different arithmetic route
# ---------------------------------------------------------------------------


def independent_learned_scores(logits: np.ndarray, true_rank: np.ndarray) -> dict:
    """Learned per-event scores straight from the recorded float32 logits.

    Cross-entropy never forms a probability first: it is
    `logsumexp(z) - z[true]`, floored the same way the frozen formula
    floors the probability. The probability vector is formed separately for
    Brier, entropy and confidence.
    """
    logits = np.asarray(logits, dtype=np.float64)
    true_rank = np.asarray(true_rank, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[1] != _RANKS:
        raise Phase11RecomputeError(f"logits have shape {logits.shape}")
    rows = np.arange(logits.shape[0])

    shift = logits.max(axis=1)
    exponentials = np.exp(logits - shift[:, None])
    total = exponentials.sum(axis=1)
    log_partition = np.log(total) + shift
    log_probability = logits[rows, true_rank] - log_partition
    # The frozen floor is on the probability, so it becomes a ceiling on
    # the negative log at -ln(1e-12).
    ce = np.minimum(-log_probability, -math.log(_LOG_FLOOR))

    probabilities = exponentials / total[:, None]
    true_probability = probabilities[rows, true_rank]
    brier = (probabilities**2).sum(axis=1) - 2.0 * true_probability + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.where(
            probabilities > 0.0, probabilities * np.log(probabilities), 0.0
        ).sum(axis=1)
    argmax = probabilities.argmax(axis=1)
    return {
        "ce": ce,
        "top1": (argmax == true_rank).astype(np.float64),
        "brier": brier,
        "entropy": entropy,
        "true_rank_probability": true_probability,
        "confidence": probabilities.max(axis=1),
    }


def independent_baseline_scores(
    counts: np.ndarray, masks: np.ndarray, true_rank: np.ndarray
) -> dict:
    """`remaining_count_belief_v1` per-event scores, rebuilt from counts.

    The baseline vector is the masked remaining inventory renormalized —
    stated here from the frozen baseline document and not imported, so a
    change to the baseline implementation shows up as a disagreement.
    """
    counts = np.asarray(counts, dtype=np.float64)
    masks = np.asarray(masks, dtype=np.float64)
    true_rank = np.asarray(true_rank, dtype=np.int64)
    if counts.shape != masks.shape:
        raise Phase11RecomputeError("counts and masks disagree in shape")
    weights = counts * masks
    totals = weights.sum(axis=1)
    if (totals <= 0.0).any():
        raise Phase11RecomputeError("a baseline row has no legal mass")
    probabilities = weights / totals[:, None]
    rows = np.arange(probabilities.shape[0])
    true_probability = probabilities[rows, true_rank]
    ce = -np.log(np.maximum(true_probability, _LOG_FLOOR))
    brier = (probabilities**2).sum(axis=1) - 2.0 * true_probability + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.where(
            probabilities > 0.0, probabilities * np.log(probabilities), 0.0
        ).sum(axis=1)
    argmax = probabilities.argmax(axis=1)
    return {
        "ce": ce,
        "top1": (argmax == true_rank).astype(np.float64),
        "brier": brier,
        "entropy": entropy,
        "true_rank_probability": true_probability,
        "confidence": probabilities.max(axis=1),
    }


# ---------------------------------------------------------------------------
# Case aggregation, the bootstrap, and ECE
# ---------------------------------------------------------------------------


def independent_case_means(values, case_index, case_count: int):
    """`(present case indices, case means)` computed with `math.fsum`.

    Both colour games of a logical case land in the same bucket, which is
    the frozen resampling unit: aggregate within the case, then resample
    cases.
    """
    buckets: list[list[float]] = [[] for _ in range(case_count)]
    for value, case in zip(values.tolist(), case_index.tolist()):
        buckets[int(case)].append(float(value))
    present = [index for index, bucket in enumerate(buckets) if bucket]
    means = [math.fsum(buckets[index]) / len(buckets[index]) for index in present]
    return present, means


def _replicate_index(seed: int, case_count: int, replicates: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0, case_count, size=(replicates, case_count)
    )


def independent_bootstrap_mean(case_values, bank: str, metric_token: str) -> dict:
    """The percentile interval of the mean of case aggregates."""
    values = [float(value) for value in case_values]
    if not values:
        return {
            "point": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "cases": 0,
            "metric_token": metric_token,
        }
    seed = independent_bootstrap_seed(bank, metric_token)
    index = _replicate_index(seed, len(values), _REPLICATES)
    count = float(len(values))
    statistics = sorted(
        math.fsum(values[position] for position in index[replicate].tolist()) / count
        for replicate in range(_REPLICATES)
    )
    alpha = (1.0 - _CONFIDENCE) / 2.0
    return {
        "point": math.fsum(values) / count,
        "lower": independent_quantile(statistics, alpha),
        "upper": independent_quantile(statistics, 1.0 - alpha),
        "cases": len(values),
        "metric_token": metric_token,
        "stream_seed": seed,
    }


def independent_bootstrap_ratio(
    numerator, denominator, bank: str, metric_token: str
) -> dict:
    """The percentile interval of `mean(numerator) / mean(denominator)`.

    One index draw feeds both aggregates and the ratio is recomputed inside
    every replicate — the frozen paired rule.
    """
    top = [float(value) for value in numerator]
    bottom = [float(value) for value in denominator]
    if len(top) != len(bottom):
        raise Phase11RecomputeError("ratio aggregates disagree in length")
    if not top:
        return {
            "point": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "cases": 0,
            "metric_token": metric_token,
        }
    seed = independent_bootstrap_seed(bank, metric_token)
    index = _replicate_index(seed, len(top), _REPLICATES)
    statistics = []
    for replicate in range(_REPLICATES):
        row = index[replicate].tolist()
        statistics.append(
            math.fsum(top[position] for position in row)
            / math.fsum(bottom[position] for position in row)
        )
    statistics.sort()
    alpha = (1.0 - _CONFIDENCE) / 2.0
    return {
        "point": math.fsum(top) / math.fsum(bottom),
        "lower": independent_quantile(statistics, alpha),
        "upper": independent_quantile(statistics, 1.0 - alpha),
        "cases": len(top),
        "metric_token": metric_token,
        "stream_seed": seed,
    }


def independent_ece(confidence, correct) -> float:
    """The frozen 15-bin equal-width pooled ECE, re-implemented."""
    confidence = [float(value) for value in confidence]
    correct = [float(value) for value in correct]
    total = len(confidence)
    if total == 0:
        return float("nan")
    sums = [0.0] * _ECE_BINS
    hits = [0.0] * _ECE_BINS
    counts = [0] * _ECE_BINS
    for value, hit in zip(confidence, correct):
        index = int(value * _ECE_BINS)
        if index >= _ECE_BINS:
            index = _ECE_BINS - 1
        if index < 0:
            index = 0
        sums[index] += value
        hits[index] += hit
        counts[index] += 1
    error = 0.0
    for index in range(_ECE_BINS):
        if counts[index] == 0:
            continue
        mean_confidence = sums[index] / counts[index]
        accuracy = hits[index] / counts[index]
        error += (counts[index] / total) * abs(accuracy - mean_confidence)
    return float(error)


# ---------------------------------------------------------------------------
# The full independent pass over a prediction store
# ---------------------------------------------------------------------------


def recompute_bank(root, manifest: dict, bank: str, *, shard_reader=None) -> dict:
    """Every required quantity, rebuilt from the recorded primitive rows.

    `shard_reader` lets the caller supply the accepted reader (paths are a
    diagnostic, not identity, so locating a file is not the thing under
    audit); the arithmetic on the arrays it returns is entirely this
    module's.
    """
    if shard_reader is None:
        raise Phase11RecomputeError(
            "recompute_bank needs a shard_reader; the independent path audits "
            "arithmetic, not file layout"
        )

    case_lookup: dict[str, int] = {}
    for entry in manifest["games_index"]:
        if entry["case_id"] not in case_lookup:
            case_lookup[entry["case_id"]] = len(case_lookup)

    metric_names = (
        "ce",
        "top1",
        "brier",
        "entropy",
        "true_rank_probability",
        "confidence",
    )
    # Per-game blocks, concatenated once. Accumulating Python floats
    # instead would cost ~1 GB of boxed objects on the validation bank.
    chunks: dict[str, list] = {"case_index": [], "stratum_index": []}
    for name in metric_names:
        chunks[f"{name}_learned"] = []
        chunks[f"{name}_baseline"] = []

    for entry in manifest["games_index"]:
        public, truth = shard_reader(entry["game_id"])
        true_rank = np.asarray(truth, dtype=np.int64)
        size = int(true_rank.size)
        if size == 0:
            continue
        offsets = np.asarray(public["event_offset"], dtype=np.int64)
        counts = np.asarray(public["remaining_counts"], dtype=np.float64)
        masks = np.asarray(public["legal_rank_mask"], dtype=np.float64)
        # Expand the per-decision inventory to per-event rows.
        expanded = np.empty((size, _RANKS), dtype=np.float64)
        for decision in range(offsets.size - 1):
            start, stop = int(offsets[decision]), int(offsets[decision + 1])
            if stop > start:
                expanded[start:stop] = counts[decision]
        learned = independent_learned_scores(public["belief_logits"], true_rank)
        baseline = independent_baseline_scores(expanded, masks, true_rank)
        chunks["case_index"].append(
            np.full(size, case_lookup[entry["case_id"]], dtype=np.int64)
        )
        chunks["stratum_index"].append(
            np.full(size, _STRATA.index(entry["opponent_stratum"]), dtype=np.int64)
        )
        for name in metric_names:
            chunks[f"{name}_learned"].append(learned[name])
            chunks[f"{name}_baseline"].append(baseline[name])

    def _joined(name: str, dtype):
        values = chunks[name]
        return (
            np.concatenate(values).astype(dtype, copy=False)
            if values
            else np.zeros(0, dtype=dtype)
        )

    case_index = _joined("case_index", np.int64)
    stratum_index = _joined("stratum_index", np.int64)
    case_count = len(case_lookup)
    arrays = {
        name: _joined(name, np.float64)
        for name in chunks
        if name not in ("case_index", "stratum_index")
    }
    del chunks

    overall = _metric_block(arrays, case_index, case_count, bank, mask=None, suffix="")
    overall["ece_learned"] = independent_ece(
        arrays["confidence_learned"], arrays["top1_learned"]
    )
    overall["ece_baseline"] = independent_ece(
        arrays["confidence_baseline"], arrays["top1_baseline"]
    )
    overall["events"] = int(case_index.size)

    strata = {}
    for index, name in enumerate(_STRATA):
        mask = stratum_index == index
        block = _metric_block(
            arrays, case_index, case_count, bank, mask=mask, suffix=f"|st={name}"
        )
        block["ece_learned"] = independent_ece(
            arrays["confidence_learned"][mask], arrays["top1_learned"][mask]
        )
        block["events"] = int(mask.sum())
        strata[name] = block

    return {
        "recompute_version": RECOMPUTE_VERSION,
        "bank": bank,
        "events": int(case_index.size),
        "cases": case_count,
        "overall": overall,
        "strata": strata,
    }


def _metric_block(arrays, case_index, case_count, bank, *, mask, suffix: str) -> dict:
    """One metric block: case aggregates, their intervals, and the ratio."""
    selected_index = case_index if mask is None else case_index[mask]
    aggregates = {}
    present_reference = None
    names = _STRATUM_METRIC_NAMES if suffix else _OVERALL_METRIC_NAMES
    for name in names:
        values = arrays[name] if mask is None else arrays[name][mask]
        present, means = independent_case_means(values, selected_index, case_count)
        if present_reference is None:
            present_reference = present
        elif present != present_reference:
            raise Phase11RecomputeError(
                f"case sets disagree between metrics in block {suffix or 'overall'!r}"
            )
        aggregates[name] = means

    block: dict = {"cases_with_events": len(present_reference or [])}
    deltas = {
        "ce_delta": [
            learned - base
            for learned, base in zip(aggregates["ce_learned"], aggregates["ce_baseline"])
        ],
        "top1_delta": [
            learned - base
            for learned, base in zip(
                aggregates["top1_learned"], aggregates["top1_baseline"]
            )
        ],
        "brier_delta": [
            learned - base
            for learned, base in zip(
                aggregates["brier_learned"], aggregates["brier_baseline"]
            )
        ],
    }
    for name, values in list(aggregates.items()) + list(deltas.items()):
        block[name] = independent_bootstrap_mean(values, bank, f"{name}{suffix}")
    block["r_ce"] = independent_bootstrap_ratio(
        aggregates["ce_learned"], aggregates["ce_baseline"], bank, f"r_ce{suffix}"
    )
    return block


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _deviation(primary, independent, seen_nan_pairs=None) -> float:
    """Absolute deviation, with NaN handled rather than silently dropped.

    `max()` ignores a NaN argument that arrives second, so a NaN deviation
    would vanish from the worst-case rollup. Two NaNs agree (an empty slice
    has no mean on either path); one NaN is a total disagreement.
    """
    left, right = float(primary), float(independent)
    left_missing, right_missing = math.isnan(left), math.isnan(right)
    if left_missing and right_missing:
        if seen_nan_pairs is not None:
            seen_nan_pairs.append(1)
        return 0.0
    if left_missing or right_missing:
        return float("inf")
    return abs(left - right)


def compare_interval(primary: dict, independent: dict, seen_nan_pairs=None) -> dict:
    """Absolute deviations of one interval's point, lower and upper."""
    return {
        key: _deviation(primary[key], independent[key], seen_nan_pairs)
        for key in ("point", "lower", "upper")
    }


def compare_blocks(
    primary_overall: dict,
    primary_strata: dict,
    independent: dict,
    *,
    tolerance: float = RECOMPUTE_TOLERANCE,
) -> dict:
    """Compare every recomputed quantity against the accepted evaluator."""
    deviations: dict = {}
    worst = 0.0
    nan_pairs: list = []

    for name, block in primary_overall["metrics"].items():
        if name not in independent["overall"]:
            raise Phase11RecomputeError(f"the independent pass is missing {name!r}")
        detail = compare_interval(block, independent["overall"][name], nan_pairs)
        deviations[f"overall.{name}"] = detail
        worst = max(worst, max(detail.values()))
        # An empty sample carries no stream (nothing was resampled).
        primary_seed = block.get("stream_seed")
        independent_seed_value = independent["overall"][name].get("stream_seed")
        if primary_seed != independent_seed_value:
            raise Phase11RecomputeError(
                f"the bootstrap stream seed of {name!r} was not reproduced: "
                f"{primary_seed} != {independent_seed_value}"
            )

    ece_deviation = _deviation(
        primary_overall["ece_learned"]["ece"],
        independent["overall"]["ece_learned"],
        nan_pairs,
    )
    deviations["overall.ece_learned"] = {"ece": ece_deviation}
    worst = max(worst, ece_deviation)
    ece_baseline_deviation = _deviation(
        primary_overall["ece_baseline"]["ece"],
        independent["overall"]["ece_baseline"],
        nan_pairs,
    )
    deviations["overall.ece_baseline"] = {"ece": ece_baseline_deviation}
    worst = max(worst, ece_baseline_deviation)

    for stratum, block in primary_strata.items():
        reference = independent["strata"][stratum]
        for name in (
            "ce_learned",
            "ce_baseline",
            "ce_delta",
            "top1_delta",
            "brier_delta",
            "r_ce",
        ):
            detail = compare_interval(block[name], reference[name], nan_pairs)
            deviations[f"stratum.{stratum}.{name}"] = detail
            worst = max(worst, max(detail.values()))
        stratum_ece = _deviation(
            block["ece_learned"]["ece"], reference["ece_learned"], nan_pairs
        )
        deviations[f"stratum.{stratum}.ece_learned"] = {"ece": stratum_ece}
        worst = max(worst, stratum_ece)

    return {
        "recompute_version": RECOMPUTE_VERSION,
        "quantities_compared": len(deviations),
        "both_nan_comparisons": len(nan_pairs),
        "max_deviation": worst,
        "tolerance": tolerance,
        "within_tolerance": worst <= tolerance,
        "deviations": deviations,
    }


__all__ = [
    "Phase11RecomputeError",
    "RECOMPUTE_TOLERANCE",
    "RECOMPUTE_VERSION",
    "compare_blocks",
    "compare_interval",
    "independent_baseline_scores",
    "independent_bootstrap_mean",
    "independent_bootstrap_ratio",
    "independent_bootstrap_seed",
    "independent_case_means",
    "independent_ece",
    "independent_learned_scores",
    "independent_quantile",
    "independent_seed",
    "recompute_bank",
]
