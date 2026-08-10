"""Effective win rate, paired confidence intervals, and league ratings.

Specification sources:

- `02_project_ruleset.md` section 9 (result values)
- Phase 4 common contract ("Primary evaluation metric")
- Phase 4 Agent 3 instructions ("Statistics", "League rating")

The headline metric
-------------------
Effective win rate, `EWR = (W + 0.5 D) / N`. Wins, draws and losses are always
reported beside it, because a 0.5 built from all draws and a 0.5 built from equal
wins and losses describe very different opponents and the ratio alone cannot
tell them apart.

Why the paired unit is the resampling unit
------------------------------------------
Phase 4 evaluates in colour-swapped pairs: the same two setups, the same two
policies, played once with the candidate as red and once as blue. Those two
games are *not* independent observations. They share a board, so a setup that
happens to be a deathtrap depresses whichever policy is unlucky enough to defend
it in both games, and they share a first-move assignment, so the pairing is what
cancels the first-move advantage rather than averaging over it.

Bootstrapping individual games would therefore treat 2N correlated games as 2N
independent ones and report an interval that is too narrow -- exactly the failure
mode the pairing was designed to remove. Every interval in this module resamples
whole paired units. A unit contributes one number, the mean of its two games, so
a unit's score lives in `{0, 0.25, 0.5, 0.75, 1.0}`.

Ordering
--------
Every aggregation sorts its rows by `match_id` before doing anything, so a
summary is a function of the result *set* and not of the order the workers
returned it in. `tests/evaluation/test_statistics.py` asserts this by shuffling.

Errored matches
---------------
A quarantined match (a policy broke its contract) has no winner and no score. It
is counted, never scored, and by default it makes summarisation raise: silently
dropping broken games would turn a systematic policy bug into a slightly smaller
sample. Pass `allow_policy_errors=True` to acknowledge them explicitly.
"""

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..engine.constants import BLUE, PLAYER_NAMES, RED
from .match_runner import (
    ERROR_ILLEGAL_ACTION,
    RESULT_DRAW,
    RESULT_LOSS,
    RESULT_WIN,
    MatchResult,
    _rules_payload,
)
from .match_spec import PairedUnit, rules_token
from .policy import PolicyRef

STATISTICS_VERSION = "evaluation_statistics_v1"

#: Defaults for a final report. Unit tests may use far fewer resamples; the
#: acceptance run and Agent 4's calibration should not.
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260403
DEFAULT_CONFIDENCE = 0.95

#: Resamples processed per NumPy block. Bounds peak memory at roughly
#: `block * units * 8` bytes rather than `resamples * units * 8`.
_BOOTSTRAP_BLOCK = 1_000

BOOTSTRAP_METHOD = "paired_unit_percentile_bootstrap"
BOOTSTRAP_ENGINE = "numpy_pcg64"

LEAGUE_METHOD = "bradley_terry_mm_elo_scale"
DEFAULT_ELO_ANCHOR = 1500.0
DEFAULT_ELO_SCALE = 400.0
DEFAULT_BT_PRIOR_DRAWS = 1.0
DEFAULT_BT_TOLERANCE = 1e-12
DEFAULT_BT_MAX_ITERATIONS = 10_000


class StatisticsError(ValueError):
    """Raised when a result set cannot be summarised as asked."""


# ---------------------------------------------------------------------------
# Effective win rate
# ---------------------------------------------------------------------------


def effective_win_rate(wins: int, draws: int, losses: int) -> float:
    """`(W + 0.5 D) / N`. Raises on an empty sample rather than returning 0.5."""
    # Negatives are checked first: they can sum to zero and would otherwise be
    # reported as an empty sample.
    if min(wins, draws, losses) < 0:
        raise StatisticsError("win, draw and loss counts must be non-negative")
    games = wins + draws + losses
    if games <= 0:
        raise StatisticsError("effective win rate is undefined for zero games")
    return (wins + 0.5 * draws) / games


@dataclass(frozen=True)
class OutcomeCounts:
    """Win/draw/loss counts, plus the errored games that carry no outcome."""

    wins: int = 0
    draws: int = 0
    losses: int = 0
    errors: int = 0

    @property
    def games(self) -> int:
        """Scored games. Errored matches are deliberately excluded."""
        return self.wins + self.draws + self.losses

    @property
    def effective_win_rate(self) -> float:
        return effective_win_rate(self.wins, self.draws, self.losses)

    def to_dict(self) -> dict:
        payload = {
            "games": self.games,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "errors": self.errors,
        }
        payload["effective_win_rate"] = self.effective_win_rate if self.games else None
        return payload

    @staticmethod
    def from_results(results: "Iterable[MatchResult]") -> "OutcomeCounts":
        wins = draws = losses = errors = 0
        for row in results:
            if row.errored:
                errors += 1
            elif row.candidate_result == RESULT_WIN:
                wins += 1
            elif row.candidate_result == RESULT_DRAW:
                draws += 1
            elif row.candidate_result == RESULT_LOSS:
                losses += 1
            else:  # pragma: no cover -- MatchResult validates its label
                raise StatisticsError(f"unknown result label {row.candidate_result!r}")
        return OutcomeCounts(wins=wins, draws=draws, losses=losses, errors=errors)


# ---------------------------------------------------------------------------
# Paired evaluation units
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedUnitScore:
    """One paired unit's contribution: the mean of its two colour-swapped games."""

    paired_unit_id: str
    candidate: str
    opponent: str
    setup_pair_id: int
    replicate: int
    red_score: float
    blue_score: float

    @property
    def score(self) -> float:
        """The unit's single observation -- the candidate's mean over both colours."""
        return (self.red_score + self.blue_score) / 2.0

    def to_dict(self) -> dict:
        return {
            "paired_unit_id": self.paired_unit_id,
            "candidate": self.candidate,
            "opponent": self.opponent,
            "setup_pair_id": self.setup_pair_id,
            "replicate": self.replicate,
            "candidate_as_red": self.red_score,
            "candidate_as_blue": self.blue_score,
            "unit_score": self.score,
        }


def ordered_results(results: "Iterable[MatchResult]") -> tuple[MatchResult, ...]:
    """Rows sorted by `match_id`.

    Every public aggregation starts here, which is what makes a summary a
    function of the result set rather than of arrival order.
    """
    return tuple(sorted(results, key=lambda row: row.match_id))


def detect_result_problems(results: "Iterable[MatchResult]") -> list[str]:
    """Structural problems in a raw result set.

    Covers the "missing/duplicate pair detection" the instructions require:

    - the same `match_id` reported twice (a schedule run twice, or a worker
      returning a shard twice);
    - a paired unit with only one of its two games (a truncated run);
    - a paired unit whose two rows claim the same colour, or disagree about the
      policies, setup pair or replicate they belong to.
    """
    rows = ordered_results(results)
    problems: list[str] = []

    seen: dict[str, MatchResult] = {}
    for row in rows:
        if row.match_id in seen:
            problems.append(f"duplicate match_id {row.match_id}")
        seen[row.match_id] = row

    grouped: dict[str, list[MatchResult]] = {}
    for row in seen.values():
        grouped.setdefault(row.paired_unit_id, []).append(row)

    for unit_id in sorted(grouped):
        members = grouped[unit_id]
        colors = sorted(row.candidate_color for row in members)
        if len(members) != 2:
            problems.append(
                f"paired unit {unit_id} has {len(members)} game(s), expected 2 "
                f"(colours present: {[PLAYER_NAMES[c] for c in colors]})"
            )
            continue
        if colors != [RED, BLUE]:
            problems.append(
                f"paired unit {unit_id} has colour assignments "
                f"{[PLAYER_NAMES[c] for c in colors]}, expected one red and one blue"
            )
        first, second = members
        for attribute in ("candidate_policy_id", "opponent_policy_id", "setup_pair_id", "replicate"):
            if getattr(first, attribute) != getattr(second, attribute):
                problems.append(
                    f"paired unit {unit_id}: the two games disagree on {attribute} "
                    f"({getattr(first, attribute)!r} != {getattr(second, attribute)!r})"
                )
    return problems


def build_paired_units(
    results: "Iterable[MatchResult]", *, allow_policy_errors: bool = False
) -> tuple[PairedUnitScore, ...]:
    """Collapse rows into paired-unit observations.

    Raises on any structural problem, so a bad result set cannot silently produce
    a confident-looking interval over half the data.
    """
    rows = ordered_results(results)
    problems = detect_result_problems(rows)
    if problems:
        listed = "\n  ".join(problems[:20])
        more = "" if len(problems) <= 20 else f"\n  ... and {len(problems) - 20} more"
        raise StatisticsError(f"result set is not cleanly paired:\n  {listed}{more}")

    errored = [row for row in rows if row.errored]
    if errored and not allow_policy_errors:
        first = errored[0]
        raise StatisticsError(
            f"{len(errored)} match(es) ended in a policy error and carry no result "
            f"(first: {first.match_id}, {first.policy_error_category}, "
            f"{first.policy_error_policy}). Fix the policy, or pass "
            "allow_policy_errors=True to summarise the remainder and report the "
            "excluded matches explicitly."
        )

    grouped: dict[str, list[MatchResult]] = {}
    for row in rows:
        grouped.setdefault(row.paired_unit_id, []).append(row)

    units: list[PairedUnitScore] = []
    for unit_id, members in grouped.items():
        if any(row.errored for row in members):
            # A unit is only an observation if both its games produced a score;
            # half a unit would reintroduce the colour bias the pairing removes.
            continue
        by_color = {row.candidate_color: row for row in members}
        units.append(
            PairedUnitScore(
                paired_unit_id=unit_id,
                candidate=members[0].candidate.token,
                opponent=members[0].opponent.token,
                setup_pair_id=members[0].setup_pair_id,
                replicate=members[0].replicate,
                red_score=float(by_color[RED].candidate_score),
                blue_score=float(by_color[BLUE].candidate_score),
            )
        )
    units.sort(key=lambda unit: unit.paired_unit_id)
    return tuple(units)


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def quantile(sorted_values: "Sequence[float]", probability: float) -> float:
    """Linear-interpolation quantile of an already-sorted sample.

    Implemented here rather than taken from NumPy so the interval endpoints are
    defined by this module and cannot shift with a library's default
    interpolation method.
    """
    if not sorted_values:
        raise StatisticsError("quantile of an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise StatisticsError(f"probability must be in [0, 1], got {probability}")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


@dataclass(frozen=True)
class ConfidenceInterval:
    """A two-sided interval and the parameters that produced it."""

    lower: float
    upper: float
    confidence: float
    method: str
    resamples: int | None = None
    seed: int | None = None
    engine: str | None = None
    resampling_unit: str = "paired_unit"
    sample_size: int = 0

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def excludes(self, value: float) -> bool:
        return value < self.lower or value > self.upper

    def to_dict(self) -> dict:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "method": self.method,
            "resamples": self.resamples,
            "seed": self.seed,
            "engine": self.engine,
            "resampling_unit": self.resampling_unit,
            "sample_size": self.sample_size,
            "width": self.width,
        }


def bootstrap_interval(
    values: "Sequence[float]",
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
    resampling_unit: str = "paired_unit",
) -> ConfidenceInterval:
    """Percentile bootstrap over `values`, reproducible from `seed`.

    `values` must already be one number per independent observation. For paired
    evaluation that means one per paired unit -- see
    :func:`paired_bootstrap_interval`, which is the entry point callers should
    normally use.

    Draws come from NumPy's PCG64 generator, seeded explicitly, and the resamples
    are generated in blocks so peak memory does not grow with `resamples`.
    Block size does not affect the result: the generator is consumed as one
    stream, so the same seed produces the same draws whatever the blocking.
    """
    sample = [float(value) for value in values]
    if not sample:
        raise StatisticsError("bootstrap of an empty sample")
    if resamples < 1:
        raise StatisticsError(f"resamples must be at least 1, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise StatisticsError(f"confidence must be in (0, 1), got {confidence}")

    if len(sample) == 1:
        # Every resample is the same single observation; the interval is a point.
        only = sample[0]
        return ConfidenceInterval(
            lower=only,
            upper=only,
            confidence=confidence,
            method=BOOTSTRAP_METHOD,
            resamples=resamples,
            seed=seed,
            engine=BOOTSTRAP_ENGINE,
            resampling_unit=resampling_unit,
            sample_size=1,
        )

    observations = np.asarray(sample, dtype=np.float64)
    size = observations.size
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    filled = 0
    while filled < resamples:
        block = min(_BOOTSTRAP_BLOCK, resamples - filled)
        indices = generator.integers(0, size, size=(block, size))
        means[filled : filled + block] = observations[indices].mean(axis=1)
        filled += block

    means.sort()
    ordered = means.tolist()
    tail = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        lower=quantile(ordered, tail),
        upper=quantile(ordered, 1.0 - tail),
        confidence=confidence,
        method=BOOTSTRAP_METHOD,
        resamples=resamples,
        seed=seed,
        engine=BOOTSTRAP_ENGINE,
        resampling_unit=resampling_unit,
        sample_size=size,
    )


def paired_bootstrap_interval(
    units: "Sequence[PairedUnitScore]",
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> ConfidenceInterval:
    """The project's confidence interval for an effective win rate.

    Resamples whole paired units, so the two colour-swapped games of a unit are
    always drawn or dropped together.
    """
    if not units:
        raise StatisticsError("paired bootstrap needs at least one complete paired unit")
    return bootstrap_interval(
        [unit.score for unit in units],
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        resampling_unit="paired_unit",
    )


def normal_interval(
    values: "Sequence[float]",
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resampling_unit: str = "paired_unit",
) -> ConfidenceInterval:
    """Normal-approximation interval over the same observations.

    Reported beside the bootstrap as a sanity check, never as the headline: the
    unit score is a five-valued discrete variable, and near 0 or 1 the normal
    approximation runs off the end of `[0, 1]` while the bootstrap does not.
    """
    sample = [float(value) for value in values]
    if len(sample) < 2:
        raise StatisticsError("a normal interval needs at least two observations")
    mean = sum(sample) / len(sample)
    variance = sum((value - mean) ** 2 for value in sample) / (len(sample) - 1)
    standard_error = math.sqrt(variance / len(sample))
    # 95% -> 1.96. Derived rather than hard-coded so a different confidence works.
    z = _normal_quantile(1.0 - (1.0 - confidence) / 2.0)
    return ConfidenceInterval(
        lower=mean - z * standard_error,
        upper=mean + z * standard_error,
        confidence=confidence,
        method="normal_approximation",
        resampling_unit=resampling_unit,
        sample_size=len(sample),
    )


def _normal_quantile(probability: float) -> float:
    """Inverse standard normal CDF via bisection on `math.erf`.

    Accurate to ~1e-12 and dependency-free; the interval it feeds is a secondary
    cross-check, so speed is irrelevant.
    """
    if not 0.0 < probability < 1.0:
        raise StatisticsError(f"probability must be in (0, 1), got {probability}")
    low, high = -12.0, 12.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if 0.5 * (1.0 + math.erf(middle / math.sqrt(2.0))) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


# ---------------------------------------------------------------------------
# Descriptive summaries
# ---------------------------------------------------------------------------


def ply_summary(results: "Iterable[MatchResult]") -> dict:
    """Mean, median and range of game length over scored games."""
    plies = sorted(row.plies for row in ordered_results(results) if row.scored)
    if not plies:
        return {"games": 0, "mean": None, "median": None, "minimum": None, "maximum": None,
                "total": 0}
    return {
        "games": len(plies),
        "mean": sum(plies) / len(plies),
        "median": quantile(plies, 0.5),
        "minimum": plies[0],
        "maximum": plies[-1],
        "total": sum(plies),
    }


def terminal_reason_frequencies(results: "Iterable[MatchResult]") -> dict:
    """How each game ended, as counts and shares, in descending frequency."""
    rows = ordered_results(results)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.terminal_reason] = counts.get(row.terminal_reason, 0) + 1
    total = len(rows)
    # Sort by count, then reason, so the table is stable for equal counts.
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "games": total,
        "counts": {reason: count for reason, count in ordered},
        "shares": {reason: count / total for reason, count in ordered},
    }


def color_split(results: "Iterable[MatchResult]") -> dict:
    """Candidate performance as red and as blue, reported separately.

    Under `color_swap_same_board` a matchup has one game of each colour per
    paired unit, so a large gap between the two halves is evidence about
    first-move advantage or a colour-dependent bug, not sampling noise.
    """
    rows = ordered_results(results)
    payload: dict[str, Any] = {}
    for color in (RED, BLUE):
        subset = [row for row in rows if row.candidate_color == color]
        counts = OutcomeCounts.from_results(subset)
        payload[PLAYER_NAMES[color]] = counts.to_dict()
        payload[PLAYER_NAMES[color]]["moves_first"] = bool(
            subset and subset[0].first_player == color
        )
    red = payload["red"].get("effective_win_rate")
    blue = payload["blue"].get("effective_win_rate")
    payload["difference_red_minus_blue"] = (
        None if red is None or blue is None else red - blue
    )
    return payload


def setup_pair_stratification(
    results: "Iterable[MatchResult]", *, include_table: bool = False
) -> dict:
    """Per-setup-pair breakdown, summarised.

    The full per-pair table is off by default: a 1,024-pair league would add a
    thousand rows per matchup to a report for very little signal. The summary
    still answers the question the stratification is for -- whether the result
    rests on a handful of pathological boards -- by reporting how the per-pair
    effective win rate is distributed and which pairs sit at the extremes.
    """
    rows = ordered_results(results)
    grouped: dict[int, list[MatchResult]] = {}
    for row in rows:
        grouped.setdefault(row.setup_pair_id, []).append(row)

    per_pair: dict[int, dict] = {}
    for pair_id in sorted(grouped):
        counts = OutcomeCounts.from_results(grouped[pair_id])
        per_pair[pair_id] = counts.to_dict()

    rates = sorted(
        (entry["effective_win_rate"], pair_id)
        for pair_id, entry in per_pair.items()
        if entry["effective_win_rate"] is not None
    )
    payload: dict[str, Any] = {
        "setup_pairs": len(per_pair),
        "games_per_pair_minimum": min((entry["games"] for entry in per_pair.values()), default=0),
        "games_per_pair_maximum": max((entry["games"] for entry in per_pair.values()), default=0),
    }
    if rates:
        values = [rate for rate, _ in rates]
        payload.update(
            {
                "pair_effective_win_rate_mean": sum(values) / len(values),
                "pair_effective_win_rate_median": quantile(values, 0.5),
                "pair_effective_win_rate_minimum": values[0],
                "pair_effective_win_rate_maximum": values[-1],
                "worst_pairs": [pair_id for _, pair_id in rates[:5]],
                "best_pairs": [pair_id for _, pair_id in rates[-5:]][::-1],
                "pairs_candidate_won_outright": sum(1 for value in values if value == 1.0),
                "pairs_candidate_lost_outright": sum(1 for value in values if value == 0.0),
            }
        )
    if include_table:
        payload["table"] = per_pair
    return payload


def unit_score_histogram(units: "Sequence[PairedUnitScore]") -> dict:
    """Counts of the five possible paired-unit scores.

    A unit score is the mean of two games each worth 0, 0.5 or 1, so it can only
    be one of five values. The histogram distinguishes "the candidate splits every
    pair" from "the candidate sweeps half and loses half", which have the same
    effective win rate and very different meanings.
    """
    labels = {0.0: "0.0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1.0: "1.0"}
    counts = {label: 0 for label in labels.values()}
    for unit in units:
        if unit.score not in labels:
            raise StatisticsError(
                f"paired unit {unit.paired_unit_id} scored {unit.score}, which is not one "
                "of the five values two games can average to"
            )
        counts[labels[unit.score]] += 1
    return counts


# ---------------------------------------------------------------------------
# Matchup summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchupSummary:
    """Everything Phase 4 reports about one candidate/opponent matchup."""

    candidate: str
    opponent: str
    counts: OutcomeCounts
    paired_units: int
    effective_win_rate: float
    interval: ConfidenceInterval
    normal: "ConfidenceInterval | None"
    colors: Mapping[str, Any]
    plies: Mapping[str, Any]
    terminal_reasons: Mapping[str, Any]
    setup_pairs: Mapping[str, Any]
    unit_scores: Mapping[str, int]
    policy_errors: int = 0
    statistics_version: str = STATISTICS_VERSION

    @property
    def separated_from_even(self) -> bool:
        """Whether the interval excludes 0.5 -- the tier-separation question."""
        return self.interval.excludes(0.5)

    def to_dict(self) -> dict:
        return {
            "statistics_version": self.statistics_version,
            "candidate": self.candidate,
            "opponent": self.opponent,
            "games": self.counts.games,
            "wins": self.counts.wins,
            "draws": self.counts.draws,
            "losses": self.counts.losses,
            "policy_errors": self.policy_errors,
            "paired_units": self.paired_units,
            "effective_win_rate": self.effective_win_rate,
            "confidence_interval": self.interval.to_dict(),
            "normal_interval": None if self.normal is None else self.normal.to_dict(),
            "separated_from_even": self.separated_from_even,
            "color_split": dict(self.colors),
            "plies": dict(self.plies),
            "terminal_reasons": dict(self.terminal_reasons),
            "setup_pair_stratification": dict(self.setup_pairs),
            "paired_unit_score_histogram": dict(self.unit_scores),
        }


def summarize_matchup(
    results: "Iterable[MatchResult]",
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
    allow_policy_errors: bool = False,
    include_setup_table: bool = False,
) -> MatchupSummary:
    """Summarise one matchup. Every row must name the same candidate and opponent."""
    rows = ordered_results(results)
    if not rows:
        raise StatisticsError("summarize_matchup was given no results")
    matchups = {row.matchup for row in rows}
    if len(matchups) != 1:
        raise StatisticsError(
            f"summarize_matchup received {len(matchups)} matchups; group the rows first "
            "(see summarize_run)"
        )

    counts = OutcomeCounts.from_results(rows)
    units = build_paired_units(rows, allow_policy_errors=allow_policy_errors)
    if not units:
        raise StatisticsError(
            f"matchup {rows[0].matchup} produced no complete paired unit to resample"
        )

    scores = [unit.score for unit in units]
    return MatchupSummary(
        candidate=rows[0].candidate.token,
        opponent=rows[0].opponent.token,
        counts=counts,
        paired_units=len(units),
        effective_win_rate=counts.effective_win_rate,
        interval=paired_bootstrap_interval(
            units, resamples=resamples, seed=seed, confidence=confidence
        ),
        normal=(normal_interval(scores, confidence=confidence) if len(scores) > 1 else None),
        colors=color_split(rows),
        plies=ply_summary(rows),
        terminal_reasons=terminal_reason_frequencies(rows),
        setup_pairs=setup_pair_stratification(rows, include_table=include_setup_table),
        unit_scores=unit_score_histogram(units),
        policy_errors=counts.errors,
    )


def group_by_matchup(
    results: "Iterable[MatchResult]",
) -> dict[str, tuple[MatchResult, ...]]:
    """Rows grouped by `candidate vs opponent`, each group sorted by `match_id`."""
    grouped: dict[str, list[MatchResult]] = {}
    for row in ordered_results(results):
        grouped.setdefault(row.matchup, []).append(row)
    return {key: tuple(value) for key, value in sorted(grouped.items())}


def summarize_run(
    results: "Iterable[MatchResult]",
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
    allow_policy_errors: bool = False,
    include_setup_table: bool = False,
    league: bool = True,
) -> dict:
    """Per-matchup summaries plus the run-level totals and league ratings.

    The bootstrap seed is offset per matchup by a hash of the matchup name rather
    than by its position, so adding a matchup to a run cannot change any other
    matchup's interval.
    """
    rows = ordered_results(results)
    if not rows:
        raise StatisticsError("summarize_run was given no results")

    grouped = group_by_matchup(rows)
    summaries: dict[str, dict] = {}
    for matchup, subset in grouped.items():
        summaries[matchup] = summarize_matchup(
            subset,
            resamples=resamples,
            seed=matchup_seed(seed, matchup),
            confidence=confidence,
            allow_policy_errors=allow_policy_errors,
            include_setup_table=include_setup_table,
        ).to_dict()

    counts = OutcomeCounts.from_results(rows)
    payload: dict[str, Any] = {
        "statistics_version": STATISTICS_VERSION,
        "matches": len(rows),
        "matchups": len(grouped),
        "paired_units": len({row.paired_unit_id for row in rows}),
        "policies": sorted(
            {row.candidate.token for row in rows} | {row.opponent.token for row in rows}
        ),
        "policy_errors": counts.errors,
        "illegal_policy_actions": sum(
            1 for row in rows if row.policy_error_category == ERROR_ILLEGAL_ACTION
        ),
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "engine": BOOTSTRAP_ENGINE,
            "resamples": resamples,
            "base_seed": seed,
            "confidence": confidence,
            "resampling_unit": "paired_unit",
        },
        "terminal_reasons": terminal_reason_frequencies(rows),
        "plies": ply_summary(rows),
        "per_matchup": summaries,
        "per_opponent": summarize_per_opponent(rows),
        "problems": detect_result_problems(rows),
    }
    if league:
        payload["league"] = bradley_terry_ratings(rows).to_dict()
    return payload


def matchup_seed(base_seed: int, matchup: str) -> int:
    """Per-matchup bootstrap seed, derived from the matchup's name.

    Position-independent by design: two runs that contain the same matchup give
    it the same seed even if the rest of the run differs.
    """
    digest = hashlib.blake2b(
        f"{int(base_seed)}:{matchup}".encode(), digest_size=8, person=b"strat-bst"
    ).digest()
    return int.from_bytes(digest, "big") >> 1


def summarize_per_opponent(results: "Iterable[MatchResult]") -> dict:
    """Each policy's record, pooled over every opponent it faced.

    A policy appears as candidate in some rows and as opponent in others, so both
    sides of every game are counted: the opponent's outcome is the mirror of the
    candidate's.
    """
    rows = ordered_results(results)
    tallies: dict[str, dict[str, int]] = {}

    def bump(token: str, key: str) -> None:
        entry = tallies.setdefault(token, {"wins": 0, "draws": 0, "losses": 0, "errors": 0})
        entry[key] += 1

    for row in rows:
        if row.errored:
            bump(row.candidate.token, "errors")
            bump(row.opponent.token, "errors")
            continue
        if row.candidate_result == RESULT_WIN:
            bump(row.candidate.token, "wins")
            bump(row.opponent.token, "losses")
        elif row.candidate_result == RESULT_LOSS:
            bump(row.candidate.token, "losses")
            bump(row.opponent.token, "wins")
        else:
            bump(row.candidate.token, "draws")
            bump(row.opponent.token, "draws")

    payload: dict[str, Any] = {}
    for token in sorted(tallies):
        entry = tallies[token]
        counts = OutcomeCounts(
            wins=entry["wins"], draws=entry["draws"], losses=entry["losses"],
            errors=entry["errors"],
        )
        payload[token] = counts.to_dict()
    return payload


# ---------------------------------------------------------------------------
# League rating
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeagueRatings:
    """Bradley-Terry strengths on an Elo-like scale.

    A convenience ranking only. The project's success metric is effective win
    rate with a paired confidence interval; a single rating number cannot
    express "these two are statistically indistinguishable", which is exactly the
    question Phase 4's tier gate asks.
    """

    method: str
    ratings: Mapping[str, float]
    strengths: Mapping[str, float]
    games: Mapping[str, int]
    iterations: int
    converged: bool
    tolerance: float
    prior_draws: float
    anchor: float
    scale: float

    @property
    def ranking(self) -> tuple[str, ...]:
        """Policies strongest first; ties broken by token for determinism."""
        return tuple(
            token
            for token, _ in sorted(self.ratings.items(), key=lambda item: (-item[1], item[0]))
        )

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "ratings": dict(self.ratings),
            "strengths": dict(self.strengths),
            "games": dict(self.games),
            "ranking": list(self.ranking),
            "iterations": self.iterations,
            "converged": self.converged,
            "tolerance": self.tolerance,
            "prior_draws": self.prior_draws,
            "elo_anchor": self.anchor,
            "elo_scale": self.scale,
        }


def pairwise_table(results: "Iterable[MatchResult]") -> dict:
    """Scored games and score totals for each unordered pair of policies.

    Keys are `"a|b"` with `a < b` lexicographically, so the table does not depend
    on which side happened to be scheduled as candidate.
    """
    rows = ordered_results(results)
    table: dict[str, dict[str, float]] = {}
    for row in rows:
        if not row.scored:
            continue
        first, second = sorted((row.candidate.token, row.opponent.token))
        key = f"{first}|{second}"
        entry = table.setdefault(
            key, {"games": 0.0, f"score_{first}": 0.0, f"score_{second}": 0.0}
        )
        candidate_score = float(row.candidate_score)
        entry["games"] += 1.0
        entry[f"score_{row.candidate.token}"] += candidate_score
        entry[f"score_{row.opponent.token}"] += 1.0 - candidate_score
    return table


def bradley_terry_ratings(
    results: "Iterable[MatchResult]",
    *,
    prior_draws: float = DEFAULT_BT_PRIOR_DRAWS,
    tolerance: float = DEFAULT_BT_TOLERANCE,
    max_iterations: int = DEFAULT_BT_MAX_ITERATIONS,
    anchor: float = DEFAULT_ELO_ANCHOR,
    scale: float = DEFAULT_ELO_SCALE,
) -> LeagueRatings:
    """Bradley-Terry maximum likelihood by the MM algorithm.

    The model is `P(i beats j) = p_i / (p_i + p_j)`; a draw counts as half a win
    to each side, which is the standard treatment for chess-like data and is what
    makes the ratings agree in spirit with effective win rate.

    The MM update is

    ```text
    p_i  <-  W_i / sum_j  n_ij / (p_i + p_j)
    ```

    where `W_i` is `i`'s total score and `n_ij` the games between `i` and `j`.
    Iteration is deterministic: policies are processed in sorted token order,
    strengths are renormalised to a geometric mean of 1 after every sweep, and
    the loop stops on a fixed tolerance or a fixed iteration cap.

    `prior_draws` adds that many virtual drawn games to each pair actually
    played. Without it the likelihood has no finite maximum when a policy wins or
    loses everything -- and Agent 2 measured `random_legal` at an effective win
    rate of 0.013 against `strategic_rule_based`, close enough to a sweep that a
    slightly smaller sample would produce one. The prior is reported so the
    ratings are never mistaken for unregularised estimates.
    """
    if prior_draws < 0:
        raise StatisticsError(f"prior_draws must be non-negative, got {prior_draws}")
    if tolerance <= 0:
        raise StatisticsError(f"tolerance must be positive, got {tolerance}")

    rows = [row for row in ordered_results(results) if row.scored]
    if not rows:
        raise StatisticsError("league ratings need at least one scored game")

    tokens = sorted({row.candidate.token for row in rows} | {row.opponent.token for row in rows})
    if len(tokens) < 2:
        raise StatisticsError("league ratings need at least two policies")
    index = {token: position for position, token in enumerate(tokens)}
    size = len(tokens)

    pair_games = [[0.0] * size for _ in range(size)]
    scores = [0.0] * size
    played = [0] * size
    for row in rows:
        a = index[row.candidate.token]
        b = index[row.opponent.token]
        candidate_score = float(row.candidate_score)
        pair_games[a][b] += 1.0
        pair_games[b][a] += 1.0
        scores[a] += candidate_score
        scores[b] += 1.0 - candidate_score
        played[a] += 1
        played[b] += 1

    # Virtual drawn games, added only between policies that actually met.
    if prior_draws:
        for a in range(size):
            for b in range(size):
                if a != b and pair_games[a][b] > 0.0:
                    pair_games[a][b] += prior_draws
                    scores[a] += prior_draws / 2.0

    strengths = [1.0] * size
    iterations = 0
    converged = False
    for iterations in range(1, max_iterations + 1):
        updated = list(strengths)
        for a in range(size):
            denominator = 0.0
            for b in range(size):
                if a == b or pair_games[a][b] == 0.0:
                    continue
                denominator += pair_games[a][b] / (strengths[a] + strengths[b])
            if denominator == 0.0 or scores[a] == 0.0:
                # Unreachable with a positive prior; kept so a caller who sets
                # prior_draws=0 gets a clear failure instead of a division error.
                raise StatisticsError(
                    f"policy {tokens[a]} has no comparable games (score {scores[a]}); "
                    "Bradley-Terry needs prior_draws > 0 for a disconnected or "
                    "undefeated policy"
                )
            updated[a] = scores[a] / denominator

        # Renormalise to a geometric mean of 1: the model is scale-invariant, so
        # this fixes the gauge and makes the iteration comparable between runs.
        log_mean = sum(math.log(value) for value in updated) / size
        updated = [value / math.exp(log_mean) for value in updated]

        shift = max(
            abs(new - old) / max(old, 1e-300) for new, old in zip(updated, strengths)
        )
        strengths = updated
        if shift < tolerance:
            converged = True
            break

    ratings = {
        token: anchor + scale * math.log10(strengths[index[token]]) for token in tokens
    }
    return LeagueRatings(
        method=LEAGUE_METHOD,
        ratings=ratings,
        strengths={token: strengths[index[token]] for token in tokens},
        games={token: played[index[token]] for token in tokens},
        iterations=iterations,
        converged=converged,
        tolerance=tolerance,
        prior_draws=prior_draws,
        anchor=anchor,
        scale=scale,
    )


# ---------------------------------------------------------------------------
# Synthetic results, for statistical validation
# ---------------------------------------------------------------------------


def synthetic_results(
    outcomes: "Sequence[tuple[float, float]]",
    *,
    candidate: str = "candidate@1.0.0",
    opponent: str = "opponent@1.0.0",
    plies: int = 100,
    terminal_reason: str = "flag_capture",
) -> tuple[MatchResult, ...]:
    """Build a result table with known outcomes, for testing the statistics.

    Each entry is one paired unit as `(candidate_as_red, candidate_as_blue)` with
    each score in `{0.0, 0.5, 1.0}`. The rows are structurally valid -- real
    paired-unit identifiers, one red and one blue per unit, distinct match
    identifiers -- so they exercise the same code path as measured results
    without playing a game.
    """
    candidate_ref = PolicyRef.from_token(candidate)
    opponent_ref = PolicyRef.from_token(opponent)
    rows: list[MatchResult] = []
    for pair_id, scores in enumerate(outcomes):
        unit = PairedUnit(
            candidate=candidate_ref, opponent=opponent_ref, setup_pair_id=pair_id
        )
        for spec, score in zip(unit.matches, scores):
            if score not in (0.0, 0.5, 1.0):
                raise StatisticsError(f"a game score must be 0.0, 0.5 or 1.0, got {score}")
            if score == 0.5:
                winner: int | None = None
                label = RESULT_DRAW
            elif score == 1.0:
                winner = spec.candidate_color
                label = RESULT_WIN
            else:
                winner = spec.opponent_color
                label = RESULT_LOSS
            rows.append(
                MatchResult(
                    match_id=spec.match_id,
                    paired_unit_id=spec.paired_unit_id,
                    suite_version=spec.suite_version,
                    pairing_mode=spec.pairing_mode,
                    candidate_policy_id=candidate_ref.policy_id,
                    candidate_policy_version=candidate_ref.policy_version,
                    opponent_policy_id=opponent_ref.policy_id,
                    opponent_policy_version=opponent_ref.policy_version,
                    candidate_color=spec.candidate_color,
                    setup_pair_id=spec.setup_pair_id,
                    setup_bank_version=spec.setup_bank_version,
                    replicate=spec.replicate,
                    root_seed=spec.root_seed,
                    candidate_seed=spec.candidate_seed,
                    opponent_seed=spec.opponent_seed,
                    rules=rules_token(spec.rules),
                    rules_payload=_rules_payload(spec.rules),
                    red_setup="?" * 40,
                    blue_setup="?" * 40,
                    first_player=spec.first_player,
                    winner=winner,
                    draw=winner is None,
                    candidate_result=label,
                    candidate_score=score,
                    terminal_reason=terminal_reason,
                    plies=plies,
                    decisions=plies,
                    replay_digest=f"synthetic-{spec.match_id}",
                )
            )
    return tuple(rows)


def summary_to_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=float)


__all__ = [
    "BOOTSTRAP_ENGINE",
    "BOOTSTRAP_METHOD",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE",
    "LEAGUE_METHOD",
    "STATISTICS_VERSION",
    "ConfidenceInterval",
    "LeagueRatings",
    "MatchupSummary",
    "OutcomeCounts",
    "PairedUnitScore",
    "StatisticsError",
    "bootstrap_interval",
    "bradley_terry_ratings",
    "build_paired_units",
    "color_split",
    "detect_result_problems",
    "effective_win_rate",
    "group_by_matchup",
    "matchup_seed",
    "normal_interval",
    "ordered_results",
    "paired_bootstrap_interval",
    "pairwise_table",
    "ply_summary",
    "quantile",
    "setup_pair_stratification",
    "summarize_matchup",
    "summarize_per_opponent",
    "summarize_run",
    "summary_to_json",
    "synthetic_results",
    "terminal_reason_frequencies",
    "unit_score_histogram",
]
