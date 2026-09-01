"""Paired non-inferiority statistics for the Phase 18 Gate G1 control.

The frozen design lives in
`reports/phase18/phase18_phase8_reproduction_contract_v1.json`; this module
implements it and invents nothing. Two facts drive the whole file.

**A paired delta needs one shared resample, not two independent ones.** The
candidate and the accepted checkpoint are scored on the *same* sealed test
games and the *same* setup pairs, so a replicate draws one index vector and
applies it to both arms. Resampling the arms separately would throw the
pairing away and inflate the interval by roughly the between-game variance
that the pairing was built to remove.

**The two metric families keep their own accepted dialect.** Head metrics are
ratios of summed per-game numerators to summed per-game denominators, so they
follow `warmstart_baselines.bootstrap_ratio_interval`: the `warmstart_eval_v1`
index draw, chunked at 500 rows, and `numpy.nanpercentile` endpoints. Play
metrics are means over paired units, so they follow
`statistics.bootstrap_interval` and take their endpoints from that module's
own `quantile`, which exists precisely so an interval cannot shift with a
library's default interpolation. Each delta is therefore expressed in the same
statistical dialect as the single-arm interval it sits next to.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from stratego.evaluation.statistics import quantile

#: The contract's frozen bootstrap size and confidence.
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95

#: Row chunk for the index draw. The generator is consumed as one stream, so
#: the chunk changes peak memory and never the result.
INDEX_CHUNK = 500

#: The margin's direction, named after the contract's own field suffixes.
#: `delta_max` metrics are better when lower (a cross-entropy ratio, a Brier
#: score) and the margin caps the delta from above. `delta_min` metrics are
#: better when higher (a top-1 rate, an effective win rate) and the margin
#: floors the delta from below.
DIRECTION_DELTA_MAX = "delta_max"
DIRECTION_DELTA_MIN = "delta_min"
DIRECTIONS = (DIRECTION_DELTA_MAX, DIRECTION_DELTA_MIN)

RATIO_METHOD = "paired_game_cluster_ratio_bootstrap"
UNIT_METHOD = "paired_unit_difference_bootstrap"


class NonInferiorityError(ValueError):
    """A malformed comparison. Always raised, never silently repaired."""


@dataclass(frozen=True)
class DeltaInterval:
    """One paired delta and its two-sided bootstrap interval.

    `delta` is always candidate minus reference, so its sign reads the same
    way for every metric: negative means the candidate scored lower.
    """

    candidate: float
    reference: float
    delta: float
    lower: float
    upper: float
    confidence: float
    replicates: int
    seed: int
    resampling_unit: str
    sample_size: int
    method: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MarginVerdict:
    """Whether one delta clears its frozen margin.

    The decision reads the bound, never the point estimate: a delta on the
    good side of the margin whose interval crosses it has not demonstrated
    non-inferiority.
    """

    metric: str
    direction: str
    margin: float
    delta: float
    deciding_bound: float
    deciding_bound_name: str
    non_inferior: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _index_blocks(size: int, replicates: int, seed: int, chunk: int = INDEX_CHUNK):
    """The frozen `warmstart_eval_v1` index draw, yielded in row blocks."""
    if size < 1:
        raise NonInferiorityError("cannot bootstrap an empty sample")
    if replicates < 1:
        raise NonInferiorityError(f"replicates must be at least 1, got {replicates}")
    generator = np.random.default_rng(int(seed))
    produced = 0
    while produced < replicates:
        rows = min(int(chunk), int(replicates) - produced)
        yield generator.integers(0, size, size=(rows, size))
        produced += rows


def _aligned(name: str, *arrays) -> tuple:
    prepared = [np.asarray(array, dtype=np.float64) for array in arrays]
    shapes = {array.shape for array in prepared}
    if len(shapes) != 1:
        raise NonInferiorityError(f"{name}: misaligned inputs {sorted(shapes)}")
    if prepared[0].ndim != 1:
        raise NonInferiorityError(f"{name}: expected one dimension per arm")
    if prepared[0].size == 0:
        raise NonInferiorityError(f"{name}: no observations to resample")
    return tuple(prepared)


def paired_ratio_delta(
    candidate_numerators,
    candidate_denominators,
    reference_numerators,
    reference_denominators,
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    chunk: int = INDEX_CHUNK,
) -> DeltaInterval:
    """Cluster bootstrap of `ratio(candidate) - ratio(reference)` over games.

    Every array carries one entry per game, in the same game order for both
    arms. One index vector per replicate resamples whole games and scores both
    arms on it, which is what makes the interval paired.
    """
    candidate_numerator, candidate_denominator, reference_numerator, reference_denominator = _aligned(
        "paired_ratio_delta",
        candidate_numerators,
        candidate_denominators,
        reference_numerators,
        reference_denominators,
    )
    games = candidate_numerator.size
    if candidate_denominator.sum() == 0 or reference_denominator.sum() == 0:
        raise NonInferiorityError("paired_ratio_delta: an arm has a zero denominator")

    estimates = np.empty(int(replicates), dtype=np.float64)
    produced = 0
    for indices in _index_blocks(games, replicates, seed, chunk=chunk):
        rows = indices.shape[0]
        candidate_bottom = candidate_denominator[indices].sum(axis=1)
        reference_bottom = reference_denominator[indices].sum(axis=1)
        candidate_bottom[candidate_bottom == 0] = np.nan
        reference_bottom[reference_bottom == 0] = np.nan
        estimates[produced : produced + rows] = (
            candidate_numerator[indices].sum(axis=1) / candidate_bottom
            - reference_numerator[indices].sum(axis=1) / reference_bottom
        )
        produced += rows

    candidate_point = float(candidate_numerator.sum() / candidate_denominator.sum())
    reference_point = float(reference_numerator.sum() / reference_denominator.sum())
    tail = (1.0 - confidence) / 2.0
    return DeltaInterval(
        candidate=candidate_point,
        reference=reference_point,
        delta=candidate_point - reference_point,
        lower=float(np.nanpercentile(estimates, tail * 100.0)),
        upper=float(np.nanpercentile(estimates, (1.0 - tail) * 100.0)),
        confidence=float(confidence),
        replicates=int(replicates),
        seed=int(seed),
        resampling_unit="game",
        sample_size=int(games),
        method=RATIO_METHOD,
    )


def paired_unit_delta(
    candidate_scores,
    reference_scores,
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    chunk: int = INDEX_CHUNK,
) -> DeltaInterval:
    """Paired bootstrap of the mean per-unit difference over setup pairs.

    Each array holds one paired-unit score per setup pair, in the same pair
    order for both arms. A replicate resamples whole units and scores both
    arms on the same draw.
    """
    candidate, reference = _aligned(
        "paired_unit_delta", candidate_scores, reference_scores
    )
    units = candidate.size
    differences = candidate - reference

    means = np.empty(int(replicates), dtype=np.float64)
    produced = 0
    for indices in _index_blocks(units, replicates, seed, chunk=chunk):
        rows = indices.shape[0]
        means[produced : produced + rows] = differences[indices].mean(axis=1)
        produced += rows

    means.sort()
    ordered = means.tolist()
    tail = (1.0 - confidence) / 2.0
    return DeltaInterval(
        candidate=float(candidate.mean()),
        reference=float(reference.mean()),
        delta=float(differences.mean()),
        lower=quantile(ordered, tail),
        upper=quantile(ordered, 1.0 - tail),
        confidence=float(confidence),
        replicates=int(replicates),
        seed=int(seed),
        resampling_unit="paired_unit",
        sample_size=int(units),
        method=UNIT_METHOD,
    )


def assess_margin(
    metric: str, interval: DeltaInterval, *, margin: float, direction: str
) -> MarginVerdict:
    """Decide one frozen margin from the interval's relevant bound."""
    if direction not in DIRECTIONS:
        raise NonInferiorityError(
            f"{metric}: unknown direction {direction!r}; expected one of {DIRECTIONS}"
        )
    if direction == DIRECTION_DELTA_MAX:
        bound, name = interval.upper, "upper"
        non_inferior = bound <= margin
    else:
        bound, name = interval.lower, "lower"
        non_inferior = bound >= margin
    return MarginVerdict(
        metric=metric,
        direction=direction,
        margin=float(margin),
        delta=interval.delta,
        deciding_bound=float(bound),
        deciding_bound_name=name,
        non_inferior=bool(non_inferior),
    )
