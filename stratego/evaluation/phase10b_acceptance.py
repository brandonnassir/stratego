"""Optional Phase 10B: validation scoring, checkpoint selection and hard gates.

Specification source: `OPTIONAL_PHASE_10B_SETUP_CONDITIONED_FINE_TUNING_AGENT.md`
sections 17-19 and 21-22.

Everything here is recomputed
-----------------------------
Every score, delta, interval, gate boolean and classification is a pure
function of primitive recorded per-game rows and the frozen thresholds. No
function reads a previously stored verdict, and none can be handed a summary
in place of the outcomes it summarizes. That is what makes an independent
re-check of a Phase 10B result cheap: rerun this module over the stored rows.

Report-only never rescues a gate
--------------------------------
Diagnostics may accompany a gate row; they never enter the boolean.
"""

from __future__ import annotations

from ..training.phase10b_contract import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_REPLICATES,
    FINAL_GATES,
    HARD_GATE_IDS,
    MATCHUP_BASIC,
    MATCHUP_DIRECT,
    MATCHUP_NEUTRAL,
    MATCHUP_PHASE8,
    MATCHUP_RANDOM,
    MATCHUP_STRATEGIC,
    MATCHUP_TACTICAL,
    PAIRED_DELTA_MATCHUPS,
    STRONG_OPPONENT_WEIGHTS,
    VALIDATION_ELIGIBILITY,
    VALIDATION_SCORE_WEIGHTS,
    VALIDATION_TIE_BREAK,
    Phase10BContractError,
    validation_score,
)
from ..training.phase10b_seed import bootstrap_seed
from .phase10b_eval import (
    ARM_BASELINE,
    ARM_CANDIDATE,
    Phase10BEvalError,
    case_means,
    color_split,
    expected_win_rate,
    safety_counters,
)
from .statistics import bootstrap_interval

#: The delta term each score weight is computed from.
SCORE_TERM_MATCHUP = {
    "delta_direct": MATCHUP_DIRECT,
    "delta_neutral": MATCHUP_NEUTRAL,
    "delta_strategic": MATCHUP_STRATEGIC,
    "delta_tactical": MATCHUP_TACTICAL,
    "delta_phase8": MATCHUP_PHASE8,
}


class Phase10BAcceptanceError(Phase10BContractError):
    """Raised when an acceptance computation is given something it cannot score."""


def _cell(rows_by_cell: dict, arm: str, matchup: str) -> list:
    try:
        return rows_by_cell[(arm, matchup)]
    except KeyError:
        raise Phase10BAcceptanceError(
            f"no rows for cell ({arm!r}, {matchup!r})"
        ) from None


def paired_differences(rows_by_cell: dict, matchup: str) -> dict:
    """`{case_id: candidate mean - Phase 9 mean}` for one paired matchup."""
    candidate = case_means(_cell(rows_by_cell, ARM_CANDIDATE, matchup))
    baseline = case_means(_cell(rows_by_cell, ARM_BASELINE, matchup))
    if set(candidate) != set(baseline):
        raise Phase10BAcceptanceError(
            f"{matchup}: the two arms were not evaluated on the same logical cases"
        )
    return {case_id: candidate[case_id] - baseline[case_id] for case_id in candidate}


def head_to_head_margins(rows_by_cell: dict, matchup: str) -> dict:
    """`{case_id: candidate mean - 0.5}` for one head-to-head matchup."""
    return {
        case_id: value - 0.5
        for case_id, value in case_means(_cell(rows_by_cell, ARM_CANDIDATE, matchup)).items()
    }


def interval(values, *, bank_version: str, token: str, replicates: int = BOOTSTRAP_REPLICATES):
    """The frozen paired-unit percentile bootstrap over per-case observations."""
    ordered = [values[key] for key in sorted(values)]
    return bootstrap_interval(
        ordered,
        resamples=int(replicates),
        seed=bootstrap_seed(bank_version, token),
        confidence=BOOTSTRAP_CONFIDENCE,
        resampling_unit="logical_case",
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validation_report(rows_by_cell: dict, *, bank_version: str, iteration: int,
                      behavior_kl: "float | None" = None,
                      replicates: int = BOOTSTRAP_REPLICATES) -> dict:
    """The complete frozen validation result of one scheduled checkpoint."""
    deltas: dict = {}
    per_matchup: dict = {}

    for term, matchup in SCORE_TERM_MATCHUP.items():
        if matchup in (MATCHUP_DIRECT, MATCHUP_NEUTRAL):
            observations = head_to_head_margins(rows_by_cell, matchup)
            candidate_ewr = expected_win_rate(_cell(rows_by_cell, ARM_CANDIDATE, matchup))
            baseline_ewr = None
        else:
            observations = paired_differences(rows_by_cell, matchup)
            candidate_ewr = expected_win_rate(_cell(rows_by_cell, ARM_CANDIDATE, matchup))
            baseline_ewr = expected_win_rate(_cell(rows_by_cell, ARM_BASELINE, matchup))
        point = sum(observations.values()) / len(observations)
        deltas[term] = point
        per_matchup[matchup] = {
            "term": term,
            "candidate_ewr": candidate_ewr,
            "baseline_ewr": baseline_ewr,
            "delta": point,
            "cases": len(observations),
            "interval": interval(
                observations,
                bank_version=bank_version,
                token=f"validation|it={iteration}|{matchup}",
                replicates=replicates,
            ).to_dict(),
        }

    guards: dict = {}
    for matchup in (MATCHUP_RANDOM, MATCHUP_BASIC):
        guards[matchup] = {
            "candidate_ewr": expected_win_rate(_cell(rows_by_cell, ARM_CANDIDATE, matchup)),
            "baseline_ewr": expected_win_rate(_cell(rows_by_cell, ARM_BASELINE, matchup)),
            "color_split": color_split(_cell(rows_by_cell, ARM_CANDIDATE, matchup)),
        }

    counters: dict = {}
    for (arm, matchup), rows in sorted(rows_by_cell.items()):
        counters[f"{arm}|{matchup}"] = safety_counters(rows)
    total_errors = sum(
        entry["errored_games"] + entry["missing_scores"] for entry in counters.values()
    )

    neutral_ewr = per_matchup[MATCHUP_NEUTRAL]["candidate_ewr"]
    eligibility = {
        "random_ewr": guards[MATCHUP_RANDOM]["candidate_ewr"],
        "random_ewr_min": VALIDATION_ELIGIBILITY["random_ewr_min"],
        "random_pass": guards[MATCHUP_RANDOM]["candidate_ewr"]
        >= VALIDATION_ELIGIBILITY["random_ewr_min"],
        "basic_ewr": guards[MATCHUP_BASIC]["candidate_ewr"],
        "basic_ewr_min": VALIDATION_ELIGIBILITY["basic_ewr_min"],
        "basic_pass": guards[MATCHUP_BASIC]["candidate_ewr"]
        >= VALIDATION_ELIGIBILITY["basic_ewr_min"],
        "neutral_rollback_ewr": neutral_ewr,
        "neutral_rollback_ewr_min": VALIDATION_ELIGIBILITY["neutral_rollback_ewr_min"],
        "neutral_rollback_pass": neutral_ewr
        >= VALIDATION_ELIGIBILITY["neutral_rollback_ewr_min"],
        "no_evaluation_errors": total_errors == 0,
    }
    eligibility["eligible"] = bool(
        eligibility["random_pass"]
        and eligibility["basic_pass"]
        and eligibility["neutral_rollback_pass"]
        and eligibility["no_evaluation_errors"]
    )

    return {
        "iteration": int(iteration),
        "bank_version": bank_version,
        "score": validation_score(deltas),
        "score_weights": dict(VALIDATION_SCORE_WEIGHTS),
        "deltas": deltas,
        "per_matchup": per_matchup,
        "guards": guards,
        "eligibility": eligibility,
        "behavior_kl": None if behavior_kl is None else float(behavior_kl),
        "safety_counters": counters,
    }


def select_checkpoint(reports) -> dict:
    """Apply the frozen eligibility filter and tie-break; return the winner.

    Selection reads validation only. It cannot change a learning rate, a
    population mix, a selector, a PPO threshold, an iteration count or an
    entropy schedule, because none of those are inputs here.
    """
    entries = list(reports)
    if not entries:
        raise Phase10BAcceptanceError("no scheduled validation reports to select from")
    eligible = [entry for entry in entries if entry["eligibility"]["eligible"]]
    if not eligible:
        return {
            "selected": None,
            "result": "FAIL",
            "reason": (
                "no scheduled checkpoint satisfied the frozen validation "
                "eligibility rule; the accepted Phase 9 checkpoint remains the "
                "only accepted move model"
            ),
            "considered": [entry["iteration"] for entry in entries],
            "eligible": [],
            "tie_break": list(VALIDATION_TIE_BREAK),
        }

    def key(entry):
        kl = entry.get("behavior_kl")
        return (
            -float(entry["score"]),
            -float(entry["deltas"]["delta_direct"]),
            -float(entry["deltas"]["delta_neutral"]),
            -float(entry["deltas"]["delta_strategic"]),
            float("inf") if kl is None else float(kl),
            int(entry["iteration"]),
        )

    ranked = sorted(eligible, key=key)
    winner = ranked[0]
    return {
        "selected": winner["iteration"],
        "result": "SELECTED",
        "score": winner["score"],
        "deltas": dict(winner["deltas"]),
        "behavior_kl": winner.get("behavior_kl"),
        "considered": [entry["iteration"] for entry in entries],
        "eligible": [entry["iteration"] for entry in eligible],
        "ranking": [
            {
                "iteration": entry["iteration"],
                "score": entry["score"],
                "delta_direct": entry["deltas"]["delta_direct"],
                "delta_neutral": entry["deltas"]["delta_neutral"],
                "delta_strategic": entry["deltas"]["delta_strategic"],
                "behavior_kl": entry.get("behavior_kl"),
            }
            for entry in ranked
        ],
        "tie_break": list(VALIDATION_TIE_BREAK),
        "unique": len(ranked) == 1
        or key(ranked[0]) != key(ranked[1]),
    }


# ---------------------------------------------------------------------------
# Final gates
# ---------------------------------------------------------------------------


def final_report(
    rows_by_cell: dict,
    *,
    bank_version: str,
    training_counters: dict,
    belief: dict,
    upstream: dict,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    """Recompute every hard gate from primitive rows and frozen thresholds."""
    measurements: dict = {}

    for matchup in (MATCHUP_DIRECT, MATCHUP_NEUTRAL):
        observations = head_to_head_margins(rows_by_cell, matchup)
        rows = _cell(rows_by_cell, ARM_CANDIDATE, matchup)
        ewr = expected_win_rate(rows)
        band = interval(
            observations, bank_version=bank_version, token=f"final|{matchup}",
            replicates=replicates,
        )
        measurements[matchup] = {
            "candidate_ewr": ewr,
            "cases": len(observations),
            "margin_interval": band.to_dict(),
            "ewr_lower_bound": band.lower + 0.5,
            "ewr_upper_bound": band.upper + 0.5,
            "color_split": color_split(rows),
        }

    for matchup in PAIRED_DELTA_MATCHUPS:
        observations = paired_differences(rows_by_cell, matchup)
        point = sum(observations.values()) / len(observations)
        band = interval(
            observations, bank_version=bank_version, token=f"final|{matchup}",
            replicates=replicates,
        )
        measurements[matchup] = {
            "candidate_ewr": expected_win_rate(_cell(rows_by_cell, ARM_CANDIDATE, matchup)),
            "baseline_ewr": expected_win_rate(_cell(rows_by_cell, ARM_BASELINE, matchup)),
            "delta": point,
            "cases": len(observations),
            "delta_interval": band.to_dict(),
            "candidate_color_split": color_split(_cell(rows_by_cell, ARM_CANDIDATE, matchup)),
        }

    # Gate C's composite is resampled jointly over the same logical cases, so
    # the three strong-opponent deltas move together exactly as they do in the
    # data rather than being combined after three independent bootstraps.
    per_matchup_differences = {
        matchup: paired_differences(rows_by_cell, matchup)
        for matchup in STRONG_OPPONENT_WEIGHTS
    }
    case_ids = set.intersection(*(set(values) for values in per_matchup_differences.values()))
    if len(case_ids) != len(next(iter(per_matchup_differences.values()))):
        raise Phase10BAcceptanceError(
            "the strong-opponent matchups were not evaluated on identical cases"
        )
    composite = {
        case_id: sum(
            weight * per_matchup_differences[matchup][case_id]
            for matchup, weight in STRONG_OPPONENT_WEIGHTS.items()
        )
        for case_id in case_ids
    }
    composite_point = sum(composite.values()) / len(composite)
    composite_band = interval(
        composite, bank_version=bank_version, token="final|delta_L", replicates=replicates
    )
    measurements["delta_L"] = {
        "weights": dict(STRONG_OPPONENT_WEIGHTS),
        "point": composite_point,
        "cases": len(composite),
        "interval": composite_band.to_dict(),
    }

    gates: dict = {}

    threshold = FINAL_GATES["gate_a_direct_adaptation"]
    direct = measurements[MATCHUP_DIRECT]
    gates["gate_a_direct_adaptation"] = {
        "observed": {
            "ewr": direct["candidate_ewr"],
            "paired_lower_bound": direct["ewr_lower_bound"],
        },
        "threshold": dict(threshold),
        "pass": bool(
            direct["candidate_ewr"] >= threshold["ewr_min"]
            and direct["ewr_lower_bound"] > threshold["paired_lower_bound_min"]
        ),
    }

    threshold = FINAL_GATES["gate_b_neutral_rollback"]
    neutral = measurements[MATCHUP_NEUTRAL]
    gates["gate_b_neutral_rollback"] = {
        "observed": {
            "ewr": neutral["candidate_ewr"],
            "paired_lower_bound": neutral["ewr_lower_bound"],
        },
        "threshold": dict(threshold),
        "pass": bool(
            neutral["candidate_ewr"] >= threshold["ewr_min"]
            and neutral["ewr_lower_bound"] > threshold["paired_lower_bound_min"]
        ),
    }

    threshold = FINAL_GATES["gate_c_strong_composite"]
    gates["gate_c_strong_composite"] = {
        "observed": {
            "point": composite_point,
            "lower_bound": composite_band.lower,
        },
        "threshold": dict(threshold),
        "pass": bool(
            composite_point >= threshold["point_min"]
            and composite_band.lower > threshold["lower_bound_min"]
        ),
    }

    threshold = FINAL_GATES["gate_d_individual_regression"]
    per_opponent = {
        matchup: measurements[matchup]["delta_interval"]["lower"]
        for matchup in STRONG_OPPONENT_WEIGHTS
    }
    gates["gate_d_individual_regression"] = {
        "observed": per_opponent,
        "threshold": dict(threshold),
        "pass": bool(
            all(value > threshold["lower_bound_min"] for value in per_opponent.values())
        ),
    }

    threshold = FINAL_GATES["gate_e_easy_opponents"]
    random_cell = measurements[MATCHUP_RANDOM]
    basic_cell = measurements[MATCHUP_BASIC]
    random_split = random_cell["candidate_color_split"]
    easy_observed = {
        "random_overall": random_cell["candidate_ewr"],
        "random_red": random_split["red"],
        "random_blue": random_split["blue"],
        "basic": basic_cell["candidate_ewr"],
        "random_delta_lower_bound": random_cell["delta_interval"]["lower"],
        "basic_delta_lower_bound": basic_cell["delta_interval"]["lower"],
    }
    gates["gate_e_easy_opponents"] = {
        "observed": easy_observed,
        "threshold": dict(threshold),
        "pass": bool(
            easy_observed["random_overall"] >= threshold["random_overall_min"]
            and easy_observed["random_red"] >= threshold["random_red_min"]
            and easy_observed["random_blue"] >= threshold["random_blue_min"]
            and easy_observed["basic"] >= threshold["basic_min"]
            and easy_observed["random_delta_lower_bound"]
            > threshold["paired_lower_bound_min"]
            and easy_observed["basic_delta_lower_bound"]
            > threshold["paired_lower_bound_min"]
        ),
    }

    threshold = FINAL_GATES["gate_f_training_safety"]
    safety_observed = {
        "hard_kl_violations": int(training_counters.get("kl_hard_limit_breaches", 0)),
        "hard_clip_fraction_violations": int(
            training_counters.get("clip_fraction_hard_limit_breaches", 0)
        ),
        "nonfinite_losses": int(training_counters.get("non_finite_losses", 0)),
        "nonfinite_gradients": int(training_counters.get("non_finite_gradients", 0)),
        "optimizer_corruption": int(training_counters.get("checkpoint_errors", 0))
        + int(training_counters.get("non_finite_parameters", 0)),
        "illegal_training_actions": int(training_counters.get("illegal_targets", 0))
        + int(training_counters.get("data_mismatches", 0)),
    }
    gates["gate_f_training_safety"] = {
        "observed": safety_observed,
        "threshold": dict(threshold),
        "pass": bool(
            safety_observed["hard_kl_violations"] <= threshold["hard_kl_violations_max"]
            and safety_observed["hard_clip_fraction_violations"]
            <= threshold["hard_clip_fraction_violations_max"]
            and safety_observed["nonfinite_losses"] <= threshold["nonfinite_losses_max"]
            and safety_observed["nonfinite_gradients"]
            <= threshold["nonfinite_gradients_max"]
            and safety_observed["optimizer_corruption"]
            <= threshold["optimizer_corruption_max"]
            and safety_observed["illegal_training_actions"]
            <= threshold["illegal_training_actions_max"]
        ),
    }

    threshold = FINAL_GATES["gate_g_belief_preservation"]
    belief_observed = {
        "ce_ratio": belief.get("ce_ratio"),
        "top1_degradation": belief.get("top1_degradation"),
        "candidate_ce": belief.get("candidate_ce"),
        "phase9_ce": belief.get("phase9_ce"),
        "candidate_top1": belief.get("candidate_top1"),
        "phase9_top1": belief.get("phase9_top1"),
        "benchmark": belief.get("benchmark"),
    }
    gates["gate_g_belief_preservation"] = {
        "observed": belief_observed,
        "threshold": dict(threshold),
        "pass": bool(
            belief_observed["ce_ratio"] is not None
            and belief_observed["top1_degradation"] is not None
            and belief_observed["ce_ratio"] <= threshold["ce_ratio_max"]
            and belief_observed["top1_degradation"] <= threshold["top1_degradation_max"]
        ),
    }

    threshold = FINAL_GATES["gate_h_upstream_preservation"]
    unchanged = {
        name: bool(entry.get("unchanged"))
        for name, entry in sorted(upstream.get("artifacts", {}).items())
    }
    gates["gate_h_upstream_preservation"] = {
        "observed": {
            "artifacts": unchanged,
            "checked": len(unchanged),
        },
        "threshold": dict(threshold),
        "pass": bool(unchanged) and all(unchanged.values()),
    }

    missing = sorted(set(HARD_GATE_IDS) - set(gates))
    if missing:
        raise Phase10BAcceptanceError(f"gates not computed: {missing}")

    return {
        "bank_version": bank_version,
        "measurements": measurements,
        "gates": gates,
        "gates_total": len(HARD_GATE_IDS),
        "gates_passed": sum(1 for gate in gates.values() if gate["pass"]),
        "bootstrap": {
            "method": "paired-unit percentile bootstrap over logical cases",
            "replicates": int(replicates),
            "confidence": BOOTSTRAP_CONFIDENCE,
        },
        "classification": classify(gates),
    }


def classify(gates: dict) -> str:
    """`PASS-CANDIDATE` only when every hard gate passes; otherwise `FAIL`.

    `BLOCKED` is never returned from here: it is a statement about provenance
    and integrity that the harness decides before any gate is computed.
    """
    missing = sorted(set(HARD_GATE_IDS) - set(gates))
    if missing:
        raise Phase10BAcceptanceError(f"cannot classify without gates {missing}")
    return "PASS-CANDIDATE" if all(gates[name]["pass"] for name in HARD_GATE_IDS) else "FAIL"


__all__ = [
    "Phase10BAcceptanceError",
    "SCORE_TERM_MATCHUP",
    "classify",
    "final_report",
    "head_to_head_margins",
    "interval",
    "paired_differences",
    "select_checkpoint",
    "validation_report",
]
