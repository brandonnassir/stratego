"""Phase 10 Agent 1: the frozen acceptance gates, statistics and classification.

Specification sources:

- `00_PHASE_10_SEQUENCE_AND_COMMON_CONTRACT.md` ("Candidate-selection score",
  "Final acceptance gates", "Statistics")
- `01_AGENT_1_CONTRACT_SEEDS_BANKS_ACCEPTANCE.md` ("Freeze acceptance/statistics")

Everything is recomputed from primitives
----------------------------------------
The common contract's rule is explicit: *gate booleans and selection scores
must be recomputed from primitive recorded outcomes*. So every function here
takes per-case, per-game scores — the smallest thing a game produces — and
derives the effective win rates, the deltas, the intervals, the gate
booleans and the classification itself. Nothing accepts a pre-summarized
rate, because a summary is exactly where a mistake becomes invisible.

Inequality semantics
--------------------
The contract mixes strict and non-strict thresholds, and the difference
decides real cases, so both are named rather than implied:

```text
EWR >= 0.49         non-strict   at_least()
paired 95% LB > 0.47    strict   above()
```

Every gate below states which it uses, and
`tests/training/test_phase10_acceptance.py` pins each threshold with a
negative test one representable step on the failing side.

Bootstrap
---------
Intervals come from the already-accepted project bootstrap
(:func:`stratego.evaluation.statistics.bootstrap_interval`): a paired-unit
percentile bootstrap over NumPy's PCG64 with this module's frozen replicate
count and confidence, seeded per matchup by
:func:`stratego.training.phase10_seed.bootstrap_stream_seed`. The resampling
unit is the logical case, so a case's two colour games are always drawn or
dropped together, and a learned-minus-neutral difference is resampled as one
per-case number so the pairing is preserved exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..evaluation.statistics import bootstrap_interval
from .phase10_contract import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_REPLICATES,
    CANDIDATE_IDS,
    GATE_A,
    GATE_B,
    GATE_C,
    GATE_D,
    HARD_GATE_IDS,
    MATCHUP_BASIC,
    MATCHUP_LEARNED_VS_NEUTRAL,
    MATCHUP_PHASE8_ANCHOR,
    MATCHUP_RANDOM,
    MATCHUP_STRATEGIC,
    MATCHUP_TACTICAL,
    SELECTION_SCORE_WEIGHTS,
    VALIDATION_BASIC_MIN_EWR,
    VALIDATION_RANDOM_MIN_EWR,
    DIVERSITY_THRESHOLDS,
)
from .phase10_seed import bootstrap_stream_seed

#: The neutral reference an unbiased direct comparison would produce.
DIRECT_REFERENCE = 0.5


class Phase10AcceptanceError(ValueError):
    """Raised when an acceptance computation is given malformed inputs."""


def at_least(value: float, threshold: float) -> bool:
    """Non-strict `>=`, named so a gate cannot silently become strict."""
    return float(value) >= float(threshold)


def above(value: float, threshold: float) -> bool:
    """Strict `>`, named so a gate cannot silently become non-strict."""
    return float(value) > float(threshold)


# ---------------------------------------------------------------------------
# Primitive outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchupOutcomes:
    """Primitive per-case game scores for one matchup, one bank.

    `learned_games[i]` is `(game 0 score, game 1 score)` of case `i` from the
    evaluated selector's perspective — game 0 is the selector playing Red,
    game 1 the selector playing Blue, per the frozen colour pairing. A game
    score is 1.0 for a win, 0.5 for a draw and 0.0 for a loss.

    `neutral_games` carries the `Phase9+neutral_v1` arm on the *same* cases
    for matchups 2-6, and is `None` for the direct learned-vs-neutral
    matchup, which has no second arm by construction.
    """

    token: str
    case_ids: "tuple[str, ...]"
    learned_games: "tuple[tuple[float, float], ...]"
    neutral_games: "tuple[tuple[float, float], ...] | None" = None

    def __post_init__(self) -> None:
        if len(self.case_ids) != len(self.learned_games):
            raise Phase10AcceptanceError(
                f"{self.token}: {len(self.case_ids)} case ids for "
                f"{len(self.learned_games)} learned case results"
            )
        if not self.learned_games:
            raise Phase10AcceptanceError(f"{self.token}: no cases")
        for label, games in (("learned", self.learned_games), ("neutral", self.neutral_games)):
            if games is None:
                continue
            if len(games) != len(self.case_ids):
                raise Phase10AcceptanceError(
                    f"{self.token}: {label} arm covers {len(games)} cases, "
                    f"expected {len(self.case_ids)}"
                )
            for index, pair in enumerate(games):
                if len(pair) != 2:
                    raise Phase10AcceptanceError(
                        f"{self.token}: {label} case {index} has {len(pair)} games, "
                        "expected exactly the two colour-paired games"
                    )
                for score in pair:
                    if score not in (0.0, 0.5, 1.0):
                        raise Phase10AcceptanceError(
                            f"{self.token}: {label} case {index} has game score "
                            f"{score!r}; expected 1.0, 0.5 or 0.0"
                        )

    @property
    def case_count(self) -> int:
        return len(self.case_ids)

    def learned_case_scores(self) -> "tuple[float, ...]":
        return tuple((pair[0] + pair[1]) / 2.0 for pair in self.learned_games)

    def neutral_case_scores(self) -> "tuple[float, ...]":
        if self.neutral_games is None:
            raise Phase10AcceptanceError(f"{self.token}: this matchup has no neutral arm")
        return tuple((pair[0] + pair[1]) / 2.0 for pair in self.neutral_games)

    def learned_color_scores(self, game_index: int) -> "tuple[float, ...]":
        return tuple(pair[game_index] for pair in self.learned_games)

    def paired_differences(self) -> "tuple[float, ...]":
        neutral = self.neutral_case_scores()
        return tuple(
            learned - baseline
            for learned, baseline in zip(self.learned_case_scores(), neutral)
        )


def effective_win_rate(scores) -> float:
    """The mean of per-unit scores. The only EWR definition Phase 10 uses."""
    values = [float(score) for score in scores]
    if not values:
        raise Phase10AcceptanceError("effective win rate of an empty sample")
    return sum(values) / len(values)


def interval(values, bank: str, token: str) -> dict:
    """The frozen paired-unit percentile bootstrap interval of one quantity."""
    result = bootstrap_interval(
        values,
        resamples=BOOTSTRAP_REPLICATES,
        seed=bootstrap_stream_seed(bank, token),
        confidence=BOOTSTRAP_CONFIDENCE,
        resampling_unit="phase10_logical_case",
    )
    payload = result.to_dict()
    payload["bank"] = bank
    payload["token"] = token
    return payload


def matchup_summary(outcomes: MatchupOutcomes, bank: str) -> dict:
    """Everything derivable from one matchup's primitives, and nothing more."""
    learned = outcomes.learned_case_scores()
    summary = {
        "token": outcomes.token,
        "bank": bank,
        "case_count": outcomes.case_count,
        "learned_ewr": effective_win_rate(learned),
        "learned_red_ewr": effective_win_rate(outcomes.learned_color_scores(0)),
        "learned_blue_ewr": effective_win_rate(outcomes.learned_color_scores(1)),
        "learned_interval": interval(learned, bank, f"{outcomes.token}:learned"),
    }
    if outcomes.neutral_games is not None:
        neutral = outcomes.neutral_case_scores()
        differences = outcomes.paired_differences()
        summary.update(
            {
                "neutral_ewr": effective_win_rate(neutral),
                "neutral_interval": interval(neutral, bank, f"{outcomes.token}:neutral"),
                "delta": effective_win_rate(differences),
                "delta_interval": interval(
                    differences, bank, f"{outcomes.token}:delta"
                ),
            }
        )
    return summary


def summarize_matchups(outcomes, bank: str) -> dict:
    """Matchup summaries keyed by token, recomputed from primitives."""
    summaries = {}
    for entry in outcomes:
        if entry.token in summaries:
            raise Phase10AcceptanceError(f"duplicate matchup token {entry.token!r}")
        summaries[entry.token] = matchup_summary(entry, bank)
    return summaries


def _require(summaries: dict, token: str) -> dict:
    try:
        return summaries[token]
    except KeyError:
        raise Phase10AcceptanceError(f"missing matchup {token!r}") from None


# ---------------------------------------------------------------------------
# Candidate selection (Agent 5, validation bank only)
# ---------------------------------------------------------------------------


def delta_direct(summaries: dict) -> float:
    """`Delta_D = EWR(learned selector vs neutral selector) - 0.5`."""
    return _require(summaries, MATCHUP_LEARNED_VS_NEUTRAL)["learned_ewr"] - DIRECT_REFERENCE


def selection_score(summaries: dict) -> dict:
    """`S10` and its four components, recomputed from primitives."""
    components = {
        "delta_direct": delta_direct(summaries),
        "delta_strategic": _require(summaries, MATCHUP_STRATEGIC)["delta"],
        "delta_tactical": _require(summaries, MATCHUP_TACTICAL)["delta"],
        "delta_phase8_anchor": _require(summaries, MATCHUP_PHASE8_ANCHOR)["delta"],
    }
    score = sum(
        SELECTION_SCORE_WEIGHTS[name] * value for name, value in components.items()
    )
    return {"s10": score, "components": components, "weights": dict(SELECTION_SCORE_WEIGHTS)}


def validation_guards(summaries: dict) -> dict:
    """The two validation point guards. Guards, never score components."""
    random_ewr = _require(summaries, MATCHUP_RANDOM)["learned_ewr"]
    basic_ewr = _require(summaries, MATCHUP_BASIC)["learned_ewr"]
    checks = {
        "random_overall": at_least(random_ewr, VALIDATION_RANDOM_MIN_EWR),
        "basic": at_least(basic_ewr, VALIDATION_BASIC_MIN_EWR),
    }
    return {
        "random_ewr": random_ewr,
        "basic_ewr": basic_ewr,
        "random_min": VALIDATION_RANDOM_MIN_EWR,
        "basic_min": VALIDATION_BASIC_MIN_EWR,
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def tie_break_key(candidate: dict) -> tuple:
    """The frozen tie-break key, ordered so `max` picks the winner.

    Reads, in order: higher S10, higher Delta_Strategic, higher Delta_D,
    higher normalized family entropy, higher effective base diversity, then
    lexicographically smaller candidate id — the last inverted by negating
    the ordering so a plain `max` respects it.
    """
    candidate_id = candidate["candidate_id"]
    if candidate_id not in CANDIDATE_IDS:
        raise Phase10AcceptanceError(
            f"unknown candidate {candidate_id!r}; expected one of {list(CANDIDATE_IDS)}"
        )
    return (
        float(candidate["s10"]),
        float(candidate["delta_strategic"]),
        float(candidate["delta_direct"]),
        float(candidate["normalized_family_entropy"]),
        float(candidate["effective_base_diversity"]),
        tuple(-ord(character) for character in candidate_id),
    )


def select_winner(candidates) -> dict:
    """The single frozen winner among the eligible candidates.

    Ineligible candidates never win, whatever their score; if none is
    eligible the result says so and the phase's FAIL rule applies.
    """
    eligible = [entry for entry in candidates if entry.get("eligible", False)]
    ranked = sorted(eligible, key=tie_break_key, reverse=True)
    return {
        "candidate_count": len(list(candidates)),
        "eligible_count": len(eligible),
        "ranking": [entry["candidate_id"] for entry in ranked],
        "winner": ranked[0]["candidate_id"] if ranked else None,
        "no_eligible_candidate": not eligible,
        "outcome": "FAIL" if not eligible else "SELECTED",
    }


# ---------------------------------------------------------------------------
# The eight hard gates (Agent 7, test bank)
# ---------------------------------------------------------------------------


def gate_a(summaries: dict) -> dict:
    """Gate A — direct learned-v-neutral non-inferiority."""
    direct = _require(summaries, MATCHUP_LEARNED_VS_NEUTRAL)
    ewr = direct["learned_ewr"]
    lower = direct["learned_interval"]["lower"]
    ordinary = {
        "ewr_ok": at_least(ewr, GATE_A["ordinary"]["ewr_min"]),
        "lb_ok": above(lower, GATE_A["ordinary"]["lb_min"]),
    }
    improved = {
        "ewr_ok": at_least(ewr, GATE_A["improved"]["ewr_min"]),
        "lb_ok": above(lower, GATE_A["improved"]["lb_min"]),
    }
    return {
        "gate": "A",
        "ewr": ewr,
        "lower_bound": lower,
        "thresholds": {
            "ordinary": dict(GATE_A["ordinary"]),
            "improved": dict(GATE_A["improved"]),
        },
        "ordinary_checks": ordinary,
        "improved_checks": improved,
        "pass": all(ordinary.values()),
        "improved": all(improved.values()),
    }


def league_delta(summaries: dict) -> dict:
    """`Delta_L` and the per-case league differences it is resampled over."""
    weights = GATE_B["league_weights"]
    tokens = {
        "delta_strategic": MATCHUP_STRATEGIC,
        "delta_tactical": MATCHUP_TACTICAL,
        "delta_phase8_anchor": MATCHUP_PHASE8_ANCHOR,
    }
    components = {name: _require(summaries, token)["delta"] for name, token in tokens.items()}
    return {
        "delta_l": sum(weights[name] * value for name, value in components.items()),
        "components": components,
        "weights": dict(weights),
    }


def league_case_differences(outcomes: dict) -> "tuple[float, ...]":
    """The per-case weighted league difference, for the Gate B interval.

    The three strong-opponent matchups run on the identical case list, so the
    weighted combination is itself a per-case quantity and can be resampled as
    one paired unit — which is what makes the Gate B interval a statement
    about `Delta_L` rather than about three separate deltas.
    """
    weights = GATE_B["league_weights"]
    tokens = {
        "delta_strategic": MATCHUP_STRATEGIC,
        "delta_tactical": MATCHUP_TACTICAL,
        "delta_phase8_anchor": MATCHUP_PHASE8_ANCHOR,
    }
    per_matchup = {}
    reference_cases = None
    for name, token in tokens.items():
        entry = outcomes[token]
        if reference_cases is None:
            reference_cases = entry.case_ids
        elif entry.case_ids != reference_cases:
            raise Phase10AcceptanceError(
                f"{token} runs on a different case list than the other league "
                "matchups; Delta_L cannot be paired"
            )
        per_matchup[name] = entry.paired_differences()
    return tuple(
        sum(weights[name] * per_matchup[name][index] for name in tokens)
        for index in range(len(reference_cases))
    )


def gate_b(summaries: dict, league_differences, bank: str) -> dict:
    """Gate B — strong-opponent league non-inferiority.

    Takes the per-case weighted league differences rather than the raw
    matchups, so the quantity the interval is built over is visible at the
    call site; :func:`league_case_differences` is the composer
    :func:`evaluate_acceptance` uses to produce them from primitives.
    """
    league = league_delta(summaries)
    differences = tuple(float(value) for value in league_differences)
    if not differences:
        raise Phase10AcceptanceError("Gate B needs at least one league case")
    league_interval = interval(differences, bank, "league:delta_l")
    lower = league_interval["lower"]
    checks = {
        "delta_l_ok": at_least(league["delta_l"], GATE_B["delta_l_min"]),
        "lb_ok": above(lower, GATE_B["lb_min"]),
    }
    significant = {
        "delta_l_ok": above(league["delta_l"], GATE_B["significant"]["delta_l_min"]),
        "lb_ok": above(lower, GATE_B["significant"]["lb_min"]),
    }
    return {
        "gate": "B",
        "delta_l": league["delta_l"],
        "components": league["components"],
        "weights": league["weights"],
        "interval": league_interval,
        "thresholds": {
            "delta_l_min": GATE_B["delta_l_min"],
            "lb_min": GATE_B["lb_min"],
            "significant": dict(GATE_B["significant"]),
        },
        "checks": checks,
        "significant_checks": significant,
        "pass": all(checks.values()),
        "significantly_positive": all(significant.values()),
    }


def gate_c(summaries: dict) -> dict:
    """Gate C — individual strong-opponent guards."""
    checks = {}
    observed = {}
    for token in GATE_C["opponents"]:
        lower = _require(summaries, token)["delta_interval"]["lower"]
        observed[token] = lower
        checks[token] = above(lower, GATE_C["lb_min"])
    return {
        "gate": "C",
        "lower_bounds": observed,
        "threshold": GATE_C["lb_min"],
        "checks": checks,
        "pass": all(checks.values()),
    }


def gate_d(summaries: dict) -> dict:
    """Gate D — easy-opponent guards, overall, per colour, and paired."""
    random_entry = _require(summaries, MATCHUP_RANDOM)
    basic_entry = _require(summaries, MATCHUP_BASIC)
    checks = {
        "random_overall": at_least(random_entry["learned_ewr"], GATE_D["random_overall_min"]),
        "random_red": at_least(random_entry["learned_red_ewr"], GATE_D["random_red_min"]),
        "random_blue": at_least(random_entry["learned_blue_ewr"], GATE_D["random_blue_min"]),
        "basic": at_least(basic_entry["learned_ewr"], GATE_D["basic_min"]),
    }
    for token in GATE_D["paired_opponents"]:
        lower = _require(summaries, token)["delta_interval"]["lower"]
        checks[f"{token}_paired_lb"] = above(lower, GATE_D["paired_lb_min"])
    return {
        "gate": "D",
        "random_overall": random_entry["learned_ewr"],
        "random_red": random_entry["learned_red_ewr"],
        "random_blue": random_entry["learned_blue_ewr"],
        "basic": basic_entry["learned_ewr"],
        "paired_lower_bounds": {
            token: _require(summaries, token)["delta_interval"]["lower"]
            for token in GATE_D["paired_opponents"]
        },
        "thresholds": {
            key: value for key, value in GATE_D.items() if key not in ("gate", "name")
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def gate_e(diversity_report: dict) -> dict:
    """Gate E — every diversity threshold, over the final mixed distribution."""
    thresholds = DIVERSITY_THRESHOLDS
    observed = {
        "normalized_family_entropy": float(diversity_report["min_normalized_family_entropy"]),
        "effective_families": float(diversity_report["min_effective_families"]),
        "family_probability_min": float(diversity_report["min_family_probability"]),
        "family_probability_max": float(diversity_report["max_family_probability"]),
        "within_family_base_entropy": float(
            diversity_report["min_within_family_normalized_base_entropy"]
        ),
        "conditional_base_probability_max": float(
            diversity_report["max_conditional_base_probability"]
        ),
    }
    checks = {
        "normalized_family_entropy": at_least(
            observed["normalized_family_entropy"],
            thresholds["normalized_family_entropy_min"],
        ),
        "effective_families": at_least(
            observed["effective_families"], thresholds["effective_families_min"]
        ),
        "family_probability_min": at_least(
            observed["family_probability_min"], thresholds["family_probability_min"]
        ),
        "family_probability_max": at_least(
            thresholds["family_probability_max"], observed["family_probability_max"]
        ),
        "within_family_base_entropy": at_least(
            observed["within_family_base_entropy"],
            thresholds["within_family_normalized_base_entropy_min"],
        ),
        "conditional_base_probability_max": at_least(
            thresholds["max_conditional_base_probability"],
            observed["conditional_base_probability_max"],
        ),
    }
    return {
        "gate": "E",
        "observed": observed,
        "thresholds": dict(thresholds),
        "scope": "worst case over every candidate, colour and split",
        "checks": checks,
        "pass": all(checks.values()),
    }


def gate_f(correctness_report: dict) -> dict:
    """Gate F — correctness and information safety, all counts zero."""
    checks = {
        name: int(correctness_report.get(name, -1)) == 0
        for name in GATE_F_COUNTERS
    }
    return {
        "gate": "F",
        "counters": {name: correctness_report.get(name) for name in GATE_F_COUNTERS},
        "checks": checks,
        "pass": all(checks.values()),
    }


#: The Gate F counters, each of which must be exactly zero. A missing
#: counter reads as -1 and fails, so an omission can never pass silently.
GATE_F_COUNTERS = (
    "illegal_setups",
    "inventory_errors",
    "stranded_sampled_setups",
    "split_leakage",
    "provenance_mismatch",
    "hidden_opponent_selector_inputs",
    "illegal_neural_moves",
    "non_finite_selector_outputs",
    "inference_failures",
)


def gate_g(reproducibility_report: dict) -> dict:
    """Gate G — reproducibility of the whole selection chain."""
    checks = {
        "same_base": bool(reproducibility_report.get("same_base")),
        "same_reflection": bool(reproducibility_report.get("same_reflection")),
        "same_perturbation": bool(reproducibility_report.get("same_perturbation")),
        "same_final_fingerprint": bool(
            reproducibility_report.get("same_final_fingerprint")
        ),
        "worker_order_independent": bool(
            reproducibility_report.get("worker_order_independent")
        ),
        "process_restart_independent": bool(
            reproducibility_report.get("process_restart_independent")
        ),
    }
    return {"gate": "G", "checks": checks, "pass": all(checks.values())}


def gate_h(preservation_report: dict) -> dict:
    """Gate H — exact Phase 9 preservation."""
    from .phase10_contract import GATE_H as EXPECTED

    checks = {
        "checkpoint_sha256": preservation_report.get("checkpoint_sha256")
        == EXPECTED["checkpoint_sha256"],
        "model_state_digest": preservation_report.get("model_state_digest")
        == EXPECTED["model_state_digest"],
        "parameters": preservation_report.get("parameters") == EXPECTED["parameters"],
        "c1_optimizer_steps_zero": preservation_report.get("c1_optimizer_steps") == 0,
    }
    return {
        "gate": "H",
        "observed": {key: preservation_report.get(key) for key in checks},
        "expected": dict(EXPECTED),
        "checks": checks,
        "pass": all(checks.values()),
    }


def classify(gates: dict) -> str:
    """The frozen final classification of a completed Phase 10 evaluation.

    `BLOCKED` is never produced here: it is the verdict when the evidence
    needed to evaluate a gate could not be established at all, which is a
    statement about the run rather than about these numbers.
    """
    missing = [name for name in HARD_GATE_IDS if name not in gates]
    if missing:
        raise Phase10AcceptanceError(f"missing gate results: {missing}")
    if not all(gates[name]["pass"] for name in HARD_GATE_IDS):
        return "FAIL"
    if gates["A"].get("improved") and gates["B"].get("significantly_positive"):
        return "PASS-IMPROVED"
    return "PASS-NONINFERIOR"


def evaluate_acceptance(
    outcomes: dict,
    *,
    bank: str,
    diversity_report: dict,
    correctness_report: dict,
    reproducibility_report: dict,
    preservation_report: dict,
) -> dict:
    """Every hard gate and the final classification, from primitives alone."""
    summaries = summarize_matchups(outcomes.values(), bank)
    gates = {
        "A": gate_a(summaries),
        "B": gate_b(summaries, league_case_differences(outcomes), bank),
        "C": gate_c(summaries),
        "D": gate_d(summaries),
        "E": gate_e(diversity_report),
        "F": gate_f(correctness_report),
        "G": gate_g(reproducibility_report),
        "H": gate_h(preservation_report),
    }
    classification = classify(gates)
    return {
        "bank": bank,
        "matchups": summaries,
        "gates": gates,
        "hard_gates_all_pass": all(gates[name]["pass"] for name in HARD_GATE_IDS),
        "gates_true": sum(1 for name in HARD_GATE_IDS if gates[name]["pass"]),
        "gates_total": len(HARD_GATE_IDS),
        "classification": classification,
    }


__all__ = [
    "DIRECT_REFERENCE",
    "GATE_F_COUNTERS",
    "MatchupOutcomes",
    "Phase10AcceptanceError",
    "above",
    "at_least",
    "classify",
    "delta_direct",
    "effective_win_rate",
    "evaluate_acceptance",
    "gate_a",
    "gate_b",
    "gate_c",
    "gate_d",
    "gate_e",
    "gate_f",
    "gate_g",
    "gate_h",
    "interval",
    "league_case_differences",
    "league_delta",
    "matchup_summary",
    "select_winner",
    "selection_score",
    "summarize_matchups",
    "tie_break_key",
    "validation_guards",
]
