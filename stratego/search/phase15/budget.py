"""Phase 15 Agent 2 section 13: Stage C, choosing the budget.

Specification source: `02_AGENT_2_SEARCH_IMPLEMENTATION.md` sections 7, 13, 14.

One variable at a time
----------------------
Section 5 forbids tuning `beta`, candidate count, depth and world count
simultaneously, and section 13 asks for the three instructed presets on the
same boards and seeds. So this module plays a *ladder*, not a grid: the same
pairing, the same board list, the same per-decision seeds, three points that
differ only in the accepted preset that names them.

The STRONG gate
---------------
Section 7 allows one `STRONG` configuration, and only under a condition:

```text
MEDIUM shows a useful improvement   AND   MEDIUM stays under the human-play
                                          latency budget
```

:func:`strong_gate` evaluates exactly that from the measured ladder and
returns a verdict with its reasons. It is a gate, not a recommendation
engine: when it refuses, no STRONG configuration is built, and when it
allows one, the depth still comes from a short latency pilot inside the
instructed 10-12 range rather than from a search over it.

Cheapest, not largest
---------------------
Section 13's selection rule is explicit that a larger budget is not stronger
merely by being larger. :func:`select_budget` therefore walks the ladder from
the *cheapest* rung upward and stops at the first one that is not
meaningfully behind the strongest observed rung and that does not regress the
named weakness pack — so the burden of proof is on the expensive rung.
"""

from __future__ import annotations

import numpy as np

from .contract import (
    ACCEPTABLE_MOVE_SECONDS,
    LADDER_PRESET_NAMES,
    MEANINGFUL_EWR_GAIN,
    PREFERRED_MOVE_SECONDS,
    STRONG_DEPTH_RANGE,
    STRONG_PRESET_NAME,
    Phase15SearchError,
    preset as preset_of,
)

#: The budget-ladder identity.
BUDGET_VERSION = "phase15_budget_ladder_v1"

#: How much a rung's weakness-pack EWR may fall below the cheapest rung's
#: before it counts as a regression rather than noise. Same scale as the
#: engineering margin, halved because the pack is a subset of the games.
WEAKNESS_REGRESSION_TOLERANCE = 0.05


class Phase15BudgetError(Phase15SearchError):
    """A budget ladder could not be built or read."""


def ladder_points(summaries: dict, pairing_id: str, presets=LADDER_PRESET_NAMES) -> dict:
    """The measured rungs of one pairing's ladder, cheapest first."""
    points = {}
    for name in presets:
        entry = summaries.get(f"{pairing_id}|{name}")
        if entry is None:
            continue
        config = preset_of(name)
        points[name] = {
            "preset_id": name,
            "worlds": config.worlds,
            "rollout_depth": config.rollout_depth,
            "max_root_candidates": config.max_root_candidates,
            "games": entry["games"],
            "ewr": entry["ewr"],
            "min_opponent_ewr": (entry["min_opponent"] or {}).get("ewr"),
            "min_family_ewr": (entry["min_family"] or {}).get("ewr"),
            "weakness_pack_family_ewr": entry["weakness_pack_family_ewr"],
            "weakness_pack_opponent_ewr": entry["weakness_pack_opponent_ewr"],
            "search_seconds_per_game": entry["search_seconds_per_game"],
            "median_seconds_per_move": entry.get("median_seconds_per_move"),
            "p95_seconds_per_move": entry.get("p95_seconds_per_move"),
            "max_seconds_per_move": entry.get("max_seconds_per_move"),
            "mean_c1_forwards_per_game": entry.get("mean_c1_forwards_per_game"),
            "move_change_rate": entry["move_change_rate"],
            "fallback_rate": entry["fallback_rate"],
            "paired_vs_direct": entry.get("paired_vs_direct"),
        }
    if not points:
        raise Phase15BudgetError(f"no ladder rung was measured for {pairing_id!r}")
    return points


def ladder_analysis(points: dict, paired: "dict | None" = None) -> dict:
    """Section 13's four recorded quantities, rung by rung."""
    ordered = [name for name in LADDER_PRESET_NAMES if name in points]
    rows = []
    for index, name in enumerate(ordered):
        rung = points[name]
        row = dict(rung)
        if index:
            previous = points[ordered[index - 1]]
            added_seconds = rung["search_seconds_per_game"] - previous["search_seconds_per_game"]
            gained = (rung["ewr"] or 0.0) - (previous["ewr"] or 0.0)
            row["added_search_seconds_per_game"] = round(added_seconds, 4)
            row["ewr_gain_over_previous"] = round(gained, 5)
            row["ewr_gain_per_added_search_second"] = (
                round(gained / added_seconds, 6) if added_seconds > 0 else None
            )
            row["cost_multiple_over_previous"] = (
                round(rung["search_seconds_per_game"] / previous["search_seconds_per_game"], 3)
                if previous["search_seconds_per_game"]
                else None
            )
        else:
            row["added_search_seconds_per_game"] = None
            row["ewr_gain_over_previous"] = None
            row["ewr_gain_per_added_search_second"] = None
            row["cost_multiple_over_previous"] = None
        row["human_play"] = human_play_verdict(rung)
        if paired:
            row["paired_vs_cheapest"] = paired.get(name)
        rows.append(row)
    return {"budget_version": BUDGET_VERSION, "rungs": rows, "order": ordered}


def human_play_verdict(rung: dict) -> dict:
    """Section 7's two latency lines, applied to one rung."""
    median = rung.get("median_seconds_per_move")
    p95 = rung.get("p95_seconds_per_move")
    worst = rung.get("max_seconds_per_move")
    verdict = "unmeasured"
    if median is not None:
        if (p95 or median) <= PREFERRED_MOVE_SECONDS:
            verdict = "comfortable"
        elif (worst or p95 or median) <= ACCEPTABLE_MOVE_SECONDS:
            verdict = "acceptable"
        else:
            verdict = "impractical"
    return {
        "verdict": verdict,
        "median_seconds_per_move": median,
        "p95_seconds_per_move": p95,
        "max_seconds_per_move": worst,
        "preferred_ceiling": PREFERRED_MOVE_SECONDS,
        "acceptable_ceiling": ACCEPTABLE_MOVE_SECONDS,
    }


def select_budget(points: dict, *, margin: float = MEANINGFUL_EWR_GAIN) -> dict:
    """The cheapest rung that is not meaningfully behind the strongest.

    "Strongest observed" is the highest EWR on the ladder, whichever rung it
    lands on — the rule has no preference for the expensive end and does not
    assume the ladder is monotone.
    """
    ordered = [name for name in LADDER_PRESET_NAMES if name in points]
    if not ordered:
        raise Phase15BudgetError("the ladder has no rung to select from")
    best_name = max(ordered, key=lambda name: points[name]["ewr"] or 0.0)
    best = points[best_name]
    cheapest = points[ordered[0]]
    reasons = []
    selected = None
    for name in ordered:
        rung = points[name]
        behind = (best["ewr"] or 0.0) - (rung["ewr"] or 0.0)
        weakness = rung.get("weakness_pack_family_ewr")
        baseline = cheapest.get("weakness_pack_family_ewr")
        regresses = (
            weakness is not None
            and baseline is not None
            and weakness < baseline - WEAKNESS_REGRESSION_TOLERANCE
        )
        practical = human_play_verdict(rung)["verdict"] != "impractical"
        entry = {
            "preset_id": name,
            "ewr": rung["ewr"],
            "behind_best": round(behind, 5),
            "within_margin": behind <= margin,
            "weakness_pack_family_ewr": weakness,
            "regresses_weakness_pack": bool(regresses),
            "human_play": human_play_verdict(rung)["verdict"],
        }
        reasons.append(entry)
        if selected is None and behind <= margin and not regresses and practical:
            selected = name
    if selected is None:  # pragma: no cover - the best rung always qualifies
        selected = best_name
    return {
        "selected_preset": selected,
        "strongest_observed_preset": best_name,
        "strongest_observed_ewr": best["ewr"],
        "margin": margin,
        "weakness_regression_tolerance": WEAKNESS_REGRESSION_TOLERANCE,
        "rungs": reasons,
        "rule": (
            "walk the ladder from the cheapest rung and stop at the first that is "
            "within the engineering margin of the strongest observed rung, does not "
            "regress the weakness pack, and is not impractical for human play"
        ),
    }


def maximum_strength_mode(points: dict, selected: str, *, margin: float = MEANINGFUL_EWR_GAIN) -> dict:
    """Section 14 point 5: keep a slower mode only if it buys an observed gain."""
    ordered = [name for name in LADDER_PRESET_NAMES if name in points]
    candidates = [
        name
        for name in ordered
        if human_play_verdict(points[name])["verdict"] != "impractical"
    ]
    if not candidates:  # pragma: no cover - TINY is never impractical
        return {"mode": selected, "reason": "no rung is practical for human play"}
    strongest = max(candidates, key=lambda name: points[name]["ewr"] or 0.0)
    gain = (points[strongest]["ewr"] or 0.0) - (points[selected]["ewr"] or 0.0)
    return {
        "mode": strongest,
        "selected_preset": selected,
        "observed_gain_over_selected": round(gain, 5),
        "buys_a_useful_gain": gain > 0,
        "inside_engineering_margin": abs(gain) <= margin,
        "within_latency_ceiling": (
            human_play_verdict(points[strongest])["verdict"] != "impractical"
        ),
        "note": (
            "the strongest observed practical rung. When the gain sits inside the "
            "engineering margin this names the strongest configuration observed, "
            "not a rung proven stronger."
        ),
    }


def strong_gate(points: dict, *, margin: float = MEANINGFUL_EWR_GAIN) -> dict:
    """Section 7's condition for trying one STRONG configuration."""
    if "MEDIUM" not in points:
        return {
            "allowed": False,
            "reason": "MEDIUM was not measured, so its condition cannot be met",
        }
    medium = points["MEDIUM"]
    cheaper = [points[name] for name in ("TINY", "SMALL") if name in points]
    best_cheaper = max((rung["ewr"] or 0.0) for rung in cheaper) if cheaper else 0.0
    improvement = (medium["ewr"] or 0.0) - best_cheaper
    verdict = human_play_verdict(medium)
    useful = improvement >= margin
    practical = verdict["verdict"] != "impractical"
    allowed = bool(useful and practical)
    return {
        "allowed": allowed,
        "medium_ewr": medium["ewr"],
        "best_cheaper_ewr": round(best_cheaper, 5),
        "improvement_over_cheaper": round(improvement, 5),
        "useful_improvement_required": margin,
        "shows_useful_improvement": bool(useful),
        "medium_human_play": verdict,
        "stays_under_latency_budget": bool(practical),
        "depth_range_if_allowed": list(STRONG_DEPTH_RANGE),
        "preset_if_allowed": STRONG_PRESET_NAME,
        "reason": (
            "section 7 allows one STRONG configuration only after MEDIUM shows a "
            "useful improvement and stays below the human-play latency budget"
            if allowed
            else (
                "MEDIUM did not show a useful improvement over the cheaper rungs"
                if not useful
                else "MEDIUM is already past the human-play latency budget"
            )
        ),
    }


def strong_depth_pilot(latencies: dict) -> dict:
    """Choose one STRONG depth from a short latency pilot, not a grid search."""
    if not latencies:
        raise Phase15BudgetError("a STRONG depth pilot needs measured latencies")
    low, high = STRONG_DEPTH_RANGE
    eligible = {
        int(depth): value
        for depth, value in latencies.items()
        if low <= int(depth) <= high
    }
    if not eligible:
        raise Phase15BudgetError(
            f"no piloted depth lies in the instructed range {low}-{high}"
        )
    affordable = {
        depth: value
        for depth, value in eligible.items()
        if value.get("p95_seconds", float("inf")) <= ACCEPTABLE_MOVE_SECONDS
    }
    chosen = max(affordable) if affordable else min(eligible)
    return {
        "chosen_depth": int(chosen),
        "range": list(STRONG_DEPTH_RANGE),
        "piloted": {str(depth): value for depth, value in sorted(eligible.items())},
        "rule": (
            "the deepest piloted depth whose p95 stays inside the 5 s ceiling; the "
            "shallowest piloted depth if none does"
        ),
    }


__all__ = [
    "BUDGET_VERSION",
    "Phase15BudgetError",
    "WEAKNESS_REGRESSION_TOLERANCE",
    "human_play_verdict",
    "ladder_analysis",
    "ladder_points",
    "maximum_strength_mode",
    "select_budget",
    "strong_depth_pilot",
    "strong_gate",
]
