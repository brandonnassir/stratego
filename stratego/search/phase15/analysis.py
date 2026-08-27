"""Phase 15 Agent 2 sections 12-14: reading the match pack.

Specification source: `02_AGENT_2_SEARCH_IMPLEMENTATION.md` sections 12, 13, 14.

Effective win rate, and what it is not
--------------------------------------
`EWR` here is the mean of the accepted engine's own `effective_score_for` over
the games of a slice — one number per arm per slice, nothing modelled and
nothing adjusted. Section 12 is explicit that no significance claim is
required, and none is made: every table carries its own `games` count, and
:func:`paired_delta` reports the standard error of the paired difference so a
reader can see at a glance which gaps the pack can and cannot resolve.

Paired, because the pack is paired
-----------------------------------
Every arm played the same board list. The informative comparison is therefore
the *paired* one — mean over boards of (arm score - reference score on the
same board) — which removes the board-to-board variance that dominates an
unpaired difference of two 120-game means. The unpaired means are reported
too, because they are what a reader expects to see, but the paired delta is
what section 14's decision order is applied to.

The worst stratum
-----------------
Section 14 asks for worst-stratum strength and specific weight on the
aggressive, unusual, Scout, Miner/Bomb and Flag-structure packs. Both are
computed here from the same rows: `min_opponent_ewr` and
`min_family_ewr` over slices with at least `MIN_SLICE_GAMES` games, and a
named `weakness_pack` average over the families section 14 calls out.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .contract import (
    MATCH_FAMILY_KEYS,
    MATCH_OPPONENTS,
    MEANINGFUL_EWR_GAIN,
    Phase15SearchError,
    pairing as pairing_of,
)

#: The analysis identity.
ANALYSIS_VERSION = "phase15_match_analysis_v1"

#: Two systems whose median move times fall in the same bucket are treated
#: as equally fast when breaking a strength tie. 50 ms is well below what a
#: human notices and well above the run-to-run noise of one measurement.
LATENCY_TIEBREAK_BUCKET_SECONDS = 0.05

#: A slice with fewer games than this is reported but never allowed to be the
#: worst stratum: one or two games is a coin, not a weakness.
MIN_SLICE_GAMES = 4

#: Section 14's "aggressive, unusual, Scout, Miner/Bomb, Flag-structure"
#: emphasis, as the family keys the pack actually carries.
WEAKNESS_PACK_FAMILIES = (
    "aggressive_high_rank_front",
    "irregular_high_entropy",
    "scout_forward_information",
    "miner_forward",
    "high_bomb_placement",
    "distributed_bomb_defense",
    "corner_flag_fortress",
    "near_corner_flag_fortress",
)

#: Section 14's "aggressive and unusual" opponents.
WEAKNESS_PACK_OPPONENTS = (
    "stress_scout_rush",
    "stress_miner_rush",
    "stress_berserker",
    "stress_chaos",
)


class Phase15AnalysisError(Phase15SearchError):
    """A match pack could not be analysed."""


def _ewr(rows) -> "float | None":
    if not rows:
        return None
    return float(np.mean([float(row["effective_score"]) for row in rows]))


def _wdl(rows) -> dict:
    counts = {"win": 0, "draw": 0, "loss": 0}
    for row in rows:
        counts[row["outcome"]] += 1
    return counts


def _slice(rows, key: str) -> dict:
    grouped: dict = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return {
        name: {
            "games": len(entries),
            "ewr": round(_ewr(entries), 5),
            **_wdl(entries),
        }
        for name, entries in sorted(grouped.items())
    }


def _minimum(sliced: dict) -> dict:
    eligible = {
        name: entry for name, entry in sliced.items() if entry["games"] >= MIN_SLICE_GAMES
    }
    if not eligible:
        return {"name": None, "ewr": None, "games": 0}
    name = min(eligible, key=lambda key: eligible[key]["ewr"])
    return {"name": name, "ewr": eligible[name]["ewr"], "games": eligible[name]["games"]}


def _latency(rows) -> dict:
    values = [
        float(row["seconds_per_player_move"])
        for row in rows
        if row.get("seconds_per_player_move") is not None
    ]
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_seconds_per_move": round(float(array.mean()), 5),
        "median_seconds_per_move": round(float(np.median(array)), 5),
        "p95_seconds_per_move": round(float(np.percentile(array, 95)), 5),
        "max_seconds_per_move": round(float(array.max()), 5),
    }


def arm_summary(rows: "list[dict]", move_seconds: "dict | None" = None) -> dict:
    """Every section 12 quantity for one arm's games."""
    if not rows:
        raise Phase15AnalysisError("an arm summary was requested for no games")
    counts = _wdl(rows)
    by_opponent = _slice(rows, "opponent")
    by_family = _slice(rows, "player_family_key")
    changes = [
        float(row["move_change_rate"])
        for row in rows
        if row.get("move_change_rate") is not None
    ]
    fallbacks = sum(int(row["fallbacks"]) for row in rows)
    decisions = sum(int(row["player_decisions"]) for row in rows)
    weakness_families = [
        row for row in rows if row["player_family_key"] in WEAKNESS_PACK_FAMILIES
    ]
    weakness_opponents = [
        row for row in rows if row["opponent"] in WEAKNESS_PACK_OPPONENTS
    ]
    report = {
        "arm_id": rows[0]["arm_id"],
        "move_model": rows[0]["move_model"],
        "provider": rows[0]["provider"],
        "preset_id": rows[0]["preset_id"],
        "games": len(rows),
        "wins": counts["win"],
        "draws": counts["draw"],
        "losses": counts["loss"],
        "ewr": round(_ewr(rows), 5),
        "ewr_by_opponent": by_opponent,
        "ewr_by_opponent_class": _slice(rows, "opponent_class"),
        "ewr_by_color": _slice(rows, "player_color"),
        "ewr_by_setup_source": _slice(rows, "setup_source"),
        "ewr_by_family": by_family,
        "min_opponent": _minimum(by_opponent),
        "min_family": _minimum(by_family),
        "weakness_pack_family_ewr": (
            round(_ewr(weakness_families), 5) if weakness_families else None
        ),
        "weakness_pack_family_games": len(weakness_families),
        "weakness_pack_opponent_ewr": (
            round(_ewr(weakness_opponents), 5) if weakness_opponents else None
        ),
        "weakness_pack_opponent_games": len(weakness_opponents),
        "move_change_rate": round(float(np.mean(changes)), 5) if changes else None,
        "player_decisions": decisions,
        "fallbacks": fallbacks,
        "fallback_rate": round(fallbacks / decisions, 6) if decisions else None,
        "search_seconds_per_game": round(
            float(np.mean([float(row["player_seconds"]) for row in rows])), 4
        ),
        "mean_c1_forwards_per_game": round(
            float(np.mean([int(row["c1_forwards"]) for row in rows])), 1
        ),
        "mean_c1_forwards_per_move": (
            round(
                sum(int(row["c1_forwards"]) for row in rows)
                / sum(int(row["player_decisions"]) for row in rows),
                2,
            )
            if decisions
            else None
        ),
        "mean_plies": round(float(np.mean([int(row["plies"]) for row in rows])), 1),
        **_latency(rows),
    }
    if move_seconds:
        pooled = np.asarray(
            [value for board in move_seconds.values() for value in board],
            dtype=np.float64,
        )
        if pooled.size:
            report["move_latency"] = {
                "decisions": int(pooled.size),
                "mean": round(float(pooled.mean()), 5),
                "median": round(float(np.median(pooled)), 5),
                "p95": round(float(np.percentile(pooled, 95)), 5),
                "p99": round(float(np.percentile(pooled, 99)), 5),
                "max": round(float(pooled.max()), 5),
            }
    return report


def paired_delta(rows: "list[dict]", reference: "list[dict]") -> dict:
    """Mean over shared boards of `arm - reference`, with its standard error."""
    left = {row["board_id"]: float(row["effective_score"]) for row in rows}
    right = {row["board_id"]: float(row["effective_score"]) for row in reference}
    shared = sorted(set(left) & set(right))
    if not shared:
        return {"boards": 0, "delta": None, "standard_error": None}
    differences = np.asarray([left[board] - right[board] for board in shared])
    standard_error = (
        float(differences.std(ddof=1) / math.sqrt(len(shared))) if len(shared) > 1 else None
    )
    return {
        "boards": len(shared),
        "delta": round(float(differences.mean()), 5),
        "standard_error": (
            round(standard_error, 5) if standard_error is not None else None
        ),
        "wins": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "losses": int((differences < 0).sum()),
    }


def analyse_pack(results: "list[dict]") -> dict:
    """Every arm's summary plus its paired delta against its own direct arm."""
    by_arm: dict = defaultdict(list)
    seconds_by_arm: dict = defaultdict(dict)
    fallback_reasons: dict = defaultdict(lambda: defaultdict(int))
    for entry in results:
        row = entry["row"]
        key = (row["arm_id"], row["preset_id"])
        by_arm[key].append(row)
        seconds_by_arm[key][row["board_id"]] = entry.get("move_seconds") or []
        for reason, count in (entry.get("fallback_reasons") or {}).items():
            fallback_reasons[key][reason] += int(count)

    summaries = {}
    for key, rows in by_arm.items():
        arm_id, preset_id = key
        summary = arm_summary(rows, seconds_by_arm[key])
        summary["fallback_reasons"] = dict(fallback_reasons[key])
        summaries[f"{arm_id}|{preset_id}"] = summary

    for key, rows in by_arm.items():
        arm_id, preset_id = key
        target = pairing_of(arm_id) if arm_id in _known_pairings() else None
        if target is None or target.kind == "direct":
            continue
        reference_key = (f"{target.move_model}_direct", "direct")
        reference = by_arm.get(reference_key)
        if reference is None:
            continue
        summaries[f"{arm_id}|{preset_id}"]["paired_vs_direct"] = paired_delta(
            rows, reference
        )
    return summaries


def _known_pairings() -> set:
    from .contract import PAIRINGS_BY_ID

    return set(PAIRINGS_BY_ID)


def system_matrix(summaries: dict, *, preset_id: str) -> dict:
    """Section 14's matrix: the four complete systems, side by side."""
    from .contract import COMBINED_PAIRING_IDS

    matrix = {}
    for pairing_id in COMBINED_PAIRING_IDS:
        key = f"{pairing_id}|{preset_id}"
        entry = summaries.get(key)
        if entry is None:
            continue
        target = pairing_of(pairing_id)
        direct = summaries.get(f"{target.move_model}_direct|direct") or {}
        matrix[pairing_id] = {
            "move_model": target.move_model,
            "belief_model": target.provider,
            "direct_ewr": direct.get("ewr"),
            "search_ewr": entry["ewr"],
            "paired_delta_vs_direct": (entry.get("paired_vs_direct") or {}).get("delta"),
            "paired_standard_error": (entry.get("paired_vs_direct") or {}).get(
                "standard_error"
            ),
            "worst_opponent": entry["min_opponent"],
            "worst_family": entry["min_family"],
            "weakness_pack_family_ewr": entry["weakness_pack_family_ewr"],
            "weakness_pack_opponent_ewr": entry["weakness_pack_opponent_ewr"],
            "median_seconds_per_move": entry.get("median_seconds_per_move"),
            "p95_seconds_per_move": entry.get("p95_seconds_per_move"),
            "fallback_rate": entry["fallback_rate"],
            "move_change_rate": entry["move_change_rate"],
            "games": entry["games"],
        }
    return matrix


def select_system(matrix: dict, *, margin: float = MEANINGFUL_EWR_GAIN) -> dict:
    """Section 14's decision order, applied to the matrix.

    1. reject a system with an integrity failure (handled by the gate, which
       must have passed before any of these games were played);
    2. prefer better overall and worst-stratum match strength;
    3. give specific weight to the aggressive/unusual/Scout/Miner/Flag pack;
    4. on an effective tie, prefer lower latency and the simpler pairing;
    5. keep a maximum-strength mode when the slower one buys an observed gain.
    """
    if not matrix:
        raise Phase15AnalysisError("no complete system was played")

    def score(entry: dict) -> float:
        # Overall strength, the worst stratum and the named weakness pack,
        # in the order section 14 lists them. Weights are a stated reading of
        # that order, not a tuned quantity.
        overall = entry["search_ewr"] or 0.0
        worst = (entry["worst_opponent"] or {}).get("ewr")
        worst_family = (entry["worst_family"] or {}).get("ewr")
        weakness = entry["weakness_pack_family_ewr"]
        parts = [(0.5, overall)]
        for weight, value in ((0.2, worst), (0.15, worst_family), (0.15, weakness)):
            if value is not None:
                parts.append((weight, value))
        total = sum(weight for weight, _ in parts)
        return sum(weight * value for weight, value in parts) / total

    ranked = sorted(matrix.items(), key=lambda item: score(item[1]), reverse=True)
    best_id, best = ranked[0]
    contenders = [
        (pairing_id, entry)
        for pairing_id, entry in ranked
        if abs(score(entry) - score(best)) <= margin
    ]
    # Effective tie: prefer lower latency, then the simpler pairing (the one
    # whose belief specialist shares its move model's backbone), then the
    # composite. Latency is bucketed first: two systems whose median move
    # times differ by a few milliseconds are equally fast to a human, and
    # letting such a difference decide the selection would be spurious
    # precision dressed up as a rule.
    def tiebreak(item):
        pairing_id, entry = item
        latency = entry.get("median_seconds_per_move") or 0.0
        bucket = round(latency / LATENCY_TIEBREAK_BUCKET_SECONDS)
        simple = 0 if pairing_id in ("p18_b18", "p24_b24") else 1
        return (bucket, simple, -score(entry))

    selected_id, selected = min(contenders, key=tiebreak) if contenders else (best_id, best)
    return {
        "selected": selected_id,
        "selected_entry": selected,
        "ranked": [
            {"pairing_id": pairing_id, "composite": round(score(entry), 5), **entry}
            for pairing_id, entry in ranked
        ],
        "contenders_within_margin": [pairing_id for pairing_id, _ in contenders],
        "margin": margin,
        "rule": (
            "composite = 0.5*overall EWR + 0.2*worst opponent + 0.15*worst family "
            "+ 0.15*weakness-pack family EWR; ties inside the engineering margin "
            f"broken by median latency bucketed to {LATENCY_TIEBREAK_BUCKET_SECONDS}s, "
            "then the simpler pairing, then the composite"
        ),
    }


__all__ = [
    "ANALYSIS_VERSION",
    "LATENCY_TIEBREAK_BUCKET_SECONDS",
    "MIN_SLICE_GAMES",
    "Phase15AnalysisError",
    "WEAKNESS_PACK_FAMILIES",
    "WEAKNESS_PACK_OPPONENTS",
    "analyse_pack",
    "arm_summary",
    "paired_delta",
    "select_system",
    "system_matrix",
]
