"""Phase 15 belief-mixture pilot: the two stages and how they are read.

Specification source: the operator's brief of 2026-08-25.

Stage 1 is a *cheap position diagnostic*
-----------------------------------------
Every arm answers the same 120 replayed questions under the same fixed
seed, so a difference between two arms is a difference between two decision
procedures rather than between two samples. Nothing is played.

The reference the whole stage turns on is the **oracle at the same rung**.
Because root candidates are chosen by P24's policy alone — the belief
provider has no say in which moves are considered — every arm and the
oracle evaluate *the same candidate set* at every position. That makes the
oracle's Q-vector a common yardstick and

.. code-block:: text

    oracle_q_regret(arm) = max_a Q_oracle(a) - Q_oracle(a_chosen by arm)

well defined at every position, not merely where the arms happen to agree.
It is a strictly finer instrument than agreement: two arms can disagree
with the oracle equally often while one of them loses far less by doing so.

Stage 2 is a *tiny match confirmation*, and only runs if Stage 1 earns it
--------------------------------------------------------------------------
The selection rule is stated before the numbers are seen: an intermediate
lambda must beat *both* endpoints on oracle regret, and beat them by more
than the paired standard error of the regret difference. If it does not,
the honest answer is that there is no useful mixture, and Stage 2 is not
run at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ...engine.legal_moves import legal_actions
from .mixture import (
    MIXTURE_DECISION_SEED,
    MIXTURE_LAMBDAS,
    MIXTURE_REFERENCE_PRESET,
    MIXTURE_STAGE1_PRESET,
    MIXTURE_VERSION,
    Phase15MixtureError,
    build_mixture_bundle,
    lambda_token,
)

#: Arm names Stage 1 reports. The four references first, then the sweep.
ORACLE_ARM = "oracle_LARGE"
B24_MEDIUM_ARM = "b24_MEDIUM"
B24_LARGE_ARM = "b24_LARGE"
COUNT_LARGE_ARM = "count_LARGE"


def mix_arm_name(lam: float) -> str:
    return f"mix_{lambda_token(lam)}_LARGE"


def endpoint_arms(lambdas=MIXTURE_LAMBDAS) -> "tuple[str, str]":
    """`(count-only endpoint, B24-only endpoint)` of the sweep."""
    ordered = sorted(float(value) for value in lambdas)
    return mix_arm_name(ordered[0]), mix_arm_name(ordered[-1])


def interior_arms(lambdas=MIXTURE_LAMBDAS) -> "list[str]":
    ordered = sorted(float(value) for value in lambdas)
    return [mix_arm_name(value) for value in ordered[1:-1]]


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------


def build_stage1_arms(
    models,
    *,
    lambdas=MIXTURE_LAMBDAS,
    stage1_preset: str = MIXTURE_STAGE1_PRESET,
    reference_preset: str = MIXTURE_REFERENCE_PRESET,
    device: str = "cpu",
) -> dict:
    """Every arm Stage 1 decides with, built once, keyed by arm name.

    The three accepted reference arms come from the frozen
    :func:`~stratego.search.phase15.systems.build_engine`; only the mixture
    arms come from this pilot's own builder.
    """
    from .systems import build_engine

    arms = {
        ORACLE_ARM: build_engine(
            "p24_oracle", models, stage1_preset, production=False, device=device
        ),
        B24_MEDIUM_ARM: build_engine(
            "p24_b24", models, reference_preset, device=device
        ),
        B24_LARGE_ARM: build_engine("p24_b24", models, stage1_preset, device=device),
        COUNT_LARGE_ARM: build_engine(
            "p24_remaining_count", models, stage1_preset, device=device
        ),
    }
    for lam in lambdas:
        arms[mix_arm_name(lam)] = build_mixture_bundle(
            models, lam, stage1_preset, device=device
        )
    return arms


def _oracle_reference(decision) -> dict:
    """The oracle's Q and S per candidate action, plus its own choice."""
    q_by_action = {
        int(candidate.absolute_action_id): float(candidate.q_value)
        for candidate in decision.candidates
    }
    s_by_action = {
        int(candidate.absolute_action_id): float(candidate.score)
        for candidate in decision.candidates
    }
    return {
        "selected_action_id": int(decision.selected_action_id),
        "q_by_action": q_by_action,
        "s_by_action": s_by_action,
        "best_q": max(q_by_action.values()),
        "best_s": max(s_by_action.values()),
        "candidates": len(q_by_action),
    }


def run_stage1(
    models,
    states,
    *,
    lambdas=MIXTURE_LAMBDAS,
    stage1_preset: str = MIXTURE_STAGE1_PRESET,
    reference_preset: str = MIXTURE_REFERENCE_PRESET,
    seed: int = MIXTURE_DECISION_SEED,
    device: str = "cpu",
    progress=None,
) -> "list[dict]":
    """One row per (position, arm). `states` is `(row, state, plan)` triples.

    The oracle runs first at each position so its Q-vector is available as
    the reference for every other arm at that same position, and the two
    B24 references run before the sweep so a mixture row can carry its
    disagreement with both without a second pass.
    """
    from ..phase12.contract import Phase12SearchError

    arms = build_stage1_arms(
        models,
        lambdas=lambdas,
        stage1_preset=stage1_preset,
        reference_preset=reference_preset,
        device=device,
    )
    ordered = (
        [ORACLE_ARM, B24_MEDIUM_ARM, B24_LARGE_ARM, COUNT_LARGE_ARM]
        + [mix_arm_name(lam) for lam in lambdas]
    )

    rows: list[dict] = []
    for index, (position, state, plan) in enumerate(states):
        legal = set(legal_actions(state))
        reference: dict = {}
        medium_action: "int | None" = None
        large_action: "int | None" = None
        for arm in ordered:
            bundle = arms[arm]
            started = time.perf_counter()
            failure = ""
            decision = None
            try:
                decision = bundle.engine.choose_action(state, seed=seed)
            except Phase12SearchError as error:  # pragma: no cover - a real defect
                failure = type(error).__name__
            elapsed = time.perf_counter() - started

            if decision is None:
                rows.append(
                    {
                        "position_id": position["position_id"],
                        "arm": arm,
                        "preset_id": bundle.config.preset_id,
                        "ply": int(position["ply"]),
                        "unresolved": int(position["unresolved"]),
                        "legal_actions": len(legal),
                        "action_id": -1,
                        "legal": 0,
                        "search_error": failure,
                        "seconds": round(elapsed, 5),
                    }
                )
                continue

            action = int(decision.selected_action_id)
            if arm == ORACLE_ARM:
                reference = _oracle_reference(decision)
            if arm == B24_MEDIUM_ARM:
                medium_action = action
            if arm == B24_LARGE_ARM:
                large_action = action

            q_by_action = reference.get("q_by_action") or {}
            s_by_action = reference.get("s_by_action") or {}
            in_candidates = action in q_by_action
            rows.append(
                {
                    "position_id": position["position_id"],
                    "arm": arm,
                    "preset_id": bundle.config.preset_id,
                    "ply": int(position["ply"]),
                    "unresolved": int(position["unresolved"]),
                    "legal_actions": len(legal),
                    "action_id": action,
                    "direct_action_id": int(decision.direct_action_id),
                    "move_changed": int(bool(decision.move_changed)),
                    "legal": int(action in legal),
                    "search_error": "",
                    "seconds": round(elapsed, 5),
                    "c1_forwards": int(decision.c1_forwards),
                    "unique_worlds": int(decision.unique_worlds),
                    "worlds_requested": int(decision.worlds_requested),
                    "candidates": len(decision.candidates),
                    "matches_oracle": (
                        int(action == reference["selected_action_id"])
                        if reference
                        else ""
                    ),
                    "action_in_oracle_candidates": int(in_candidates),
                    "oracle_q": (
                        round(q_by_action[action], 6) if in_candidates else ""
                    ),
                    "oracle_q_best": (
                        round(reference["best_q"], 6) if reference else ""
                    ),
                    "oracle_q_regret": (
                        round(reference["best_q"] - q_by_action[action], 6)
                        if in_candidates
                        else ""
                    ),
                    "oracle_s_regret": (
                        round(reference["best_s"] - s_by_action[action], 6)
                        if in_candidates
                        else ""
                    ),
                    "matches_b24_medium": (
                        "" if medium_action is None else int(action == medium_action)
                    ),
                    "matches_b24_large": (
                        "" if large_action is None else int(action == large_action)
                    ),
                }
            )
        if progress is not None:
            progress(index + 1, len(states), len(rows))
    return rows


# ---------------------------------------------------------------------------
# Reading Stage 1
# ---------------------------------------------------------------------------


def _mean(values) -> "float | None":
    values = [value for value in values if value != "" and value is not None]
    return round(float(np.mean(values)), 5) if values else None


def summarize_stage1(rows: "list[dict]") -> dict:
    """Section-by-section, one entry per arm."""
    by_arm: dict[str, list] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    report = {}
    for arm, entries in by_arm.items():
        seconds = np.asarray([float(row["seconds"]) for row in entries])
        regrets = [
            float(row["oracle_q_regret"])
            for row in entries
            if row.get("oracle_q_regret") not in ("", None)
        ]
        report[arm] = {
            "decisions": len(entries),
            "preset_id": entries[0].get("preset_id"),
            "oracle_agreement": _mean([row.get("matches_oracle") for row in entries]),
            "oracle_q_regret_mean": (
                round(float(np.mean(regrets)), 6) if regrets else None
            ),
            "oracle_q_regret_median": (
                round(float(np.median(regrets)), 6) if regrets else None
            ),
            "oracle_q_regret_p90": (
                round(float(np.percentile(regrets, 90)), 6) if regrets else None
            ),
            "oracle_q_regret_n": len(regrets),
            "oracle_s_regret_mean": _mean(
                [row.get("oracle_s_regret") for row in entries]
            ),
            "disagreement_with_b24_medium": (
                None
                if _mean([row.get("matches_b24_medium") for row in entries]) is None
                else round(
                    1.0 - _mean([row.get("matches_b24_medium") for row in entries]), 5
                )
            ),
            "disagreement_with_b24_large": (
                None
                if _mean([row.get("matches_b24_large") for row in entries]) is None
                else round(
                    1.0 - _mean([row.get("matches_b24_large") for row in entries]), 5
                )
            ),
            "move_change_rate_vs_direct": _mean(
                [row.get("move_changed") for row in entries]
            ),
            "legal_decision_rate": _mean([row.get("legal") for row in entries]),
            "illegal_decisions": sum(1 for row in entries if not int(row.get("legal", 0))),
            "search_errors": sum(1 for row in entries if row.get("search_error")),
            "action_outside_oracle_candidates": sum(
                1
                for row in entries
                if row.get("action_in_oracle_candidates") in (0, "0")
            ),
            "mean_seconds": round(float(seconds.mean()), 5),
            "median_seconds": round(float(np.median(seconds)), 5),
            "p95_seconds": round(float(np.percentile(seconds, 95)), 5),
            "mean_c1_forwards": _mean([row.get("c1_forwards") for row in entries]),
            "mean_unique_worlds": _mean([row.get("unique_worlds") for row in entries]),
        }
    return report


def paired_regret_delta(rows: "list[dict]", arm: str, reference: str) -> dict:
    """`arm` minus `reference` oracle Q-regret, paired position by position.

    Negative is better: the arm loses less value than the reference does
    against a perfect-belief searcher. The standard error is the paired
    one — the same positions, the same oracle, the same candidate set — so
    it is far tighter than two independent means would give.
    """
    mine = {
        row["position_id"]: float(row["oracle_q_regret"])
        for row in rows
        if row["arm"] == arm and row.get("oracle_q_regret") not in ("", None)
    }
    theirs = {
        row["position_id"]: float(row["oracle_q_regret"])
        for row in rows
        if row["arm"] == reference and row.get("oracle_q_regret") not in ("", None)
    }
    shared = sorted(set(mine) & set(theirs))
    if not shared:
        return {"positions": 0, "delta": None, "standard_error": None}
    differences = np.asarray([mine[key] - theirs[key] for key in shared])
    error = (
        float(differences.std(ddof=1) / np.sqrt(len(differences)))
        if len(differences) > 1
        else 0.0
    )
    return {
        "positions": len(shared),
        "delta": round(float(differences.mean()), 6),
        "standard_error": round(error, 6),
        "better_positions": int((differences < 0).sum()),
        "worse_positions": int((differences > 0).sum()),
        "tied_positions": int((differences == 0).sum()),
    }


def reference_comparisons(rows: "list[dict]", *, lambdas=MIXTURE_LAMBDAS) -> dict:
    """Every arm's paired regret against the two arms whose EWR we already know.

    This is the comparison that decides whether the metric is usable at all.
    B24 at MEDIUM scores 0.9333 in match play and B24 at LARGE scores 0.8583 —
    a 0.075 gap that is the whole reason this pilot exists. If position-level
    oracle regret cannot separate *those two*, it cannot be used to find a
    mixture that closes the gap, and the sweep below is measuring noise.
    """
    arms = [ORACLE_ARM, B24_MEDIUM_ARM, B24_LARGE_ARM, COUNT_LARGE_ARM] + [
        mix_arm_name(lam) for lam in lambdas
    ]
    floor = None
    for row in rows:
        if row["arm"] == ORACLE_ARM and row.get("oracle_q_regret") not in ("", None):
            floor = floor or []
            floor.append(float(row["oracle_q_regret"]))
    floor_mean = round(float(np.mean(floor)), 6) if floor else None
    return {
        "oracle_regret_floor": floor_mean,
        "floor_note": (
            "the oracle selects by S = Q + beta*log(pi + epsilon), not by argmax "
            "Q, so its own Q-regret is the floor of this metric rather than zero; "
            "beta is frozen and identical across arms, so the floor is common"
        ),
        "vs_b24_large": {
            arm: paired_regret_delta(rows, arm, B24_LARGE_ARM)
            for arm in arms
            if arm != B24_LARGE_ARM
        },
        "vs_b24_medium": {
            arm: paired_regret_delta(rows, arm, B24_MEDIUM_ARM)
            for arm in arms
            if arm != B24_MEDIUM_ARM
        },
        "the_gap_the_pilot_exists_to_explain": paired_regret_delta(
            rows, B24_LARGE_ARM, B24_MEDIUM_ARM
        ),
    }


def select_lambda(
    rows: "list[dict]", summary: dict, *, lambdas=MIXTURE_LAMBDAS
) -> dict:
    """The stated rule, applied to Stage 1.

    An interior lambda is selected only when it beats **both** endpoints on
    paired oracle Q-regret by more than the paired standard error of that
    same difference. Anything less is inside the noise of 120 positions and
    is reported as "no useful mixture" rather than dressed up as a winner.
    """
    count_end, b24_end = endpoint_arms(lambdas)
    interior = interior_arms(lambdas)
    comparisons = {}
    for arm in interior:
        comparisons[arm] = {
            "vs_count_endpoint": paired_regret_delta(rows, arm, count_end),
            "vs_b24_endpoint": paired_regret_delta(rows, arm, b24_end),
        }

    ranked = sorted(
        (arm for arm in interior if summary.get(arm, {}).get("oracle_q_regret_mean") is not None),
        key=lambda arm: summary[arm]["oracle_q_regret_mean"],
    )
    findings = []
    selected = None
    for arm in ranked:
        entry = comparisons[arm]
        beats = []
        for label, delta in entry.items():
            value, error = delta.get("delta"), delta.get("standard_error")
            if value is None:
                beats.append(False)
                continue
            beats.append(value < 0.0 and abs(value) > (error or 0.0))
        if all(beats):
            selected = arm
            break
        findings.append(
            f"{arm}: regret vs count endpoint "
            f"{entry['vs_count_endpoint']['delta']:+.4f}"
            f" (se {entry['vs_count_endpoint']['standard_error']:.4f}), "
            f"vs B24 endpoint {entry['vs_b24_endpoint']['delta']:+.4f}"
            f" (se {entry['vs_b24_endpoint']['standard_error']:.4f})"
        )

    return {
        "rule": (
            "select an interior lambda only if its paired oracle Q-regret beats "
            "both endpoints by more than the paired standard error of that "
            "difference; otherwise report no useful mixture"
        ),
        "count_endpoint_arm": count_end,
        "b24_endpoint_arm": b24_end,
        "interior_arms": interior,
        "comparisons": comparisons,
        "ranked_by_regret": ranked,
        "selected_arm": selected,
        "selected_lambda": (
            None
            if selected is None
            else next(
                float(lam) for lam in lambdas if mix_arm_name(lam) == selected
            )
        ),
        "stage2_authorized": selected is not None,
        "findings": findings,
    }


def check_endpoint_identity(rows: "list[dict]", *, lambdas=MIXTURE_LAMBDAS) -> dict:
    """`lambda = 1` must reproduce the frozen `p24_b24` arm at the same rung.

    The endpoint is built through the mixture code path — two providers, an
    addition and a normalization — so this is a real check of that path
    rather than a tautology: if the ordinal derivation, the marginal keys or
    the normalization were wrong, the two would part company here.
    """
    _count_end, b24_end = endpoint_arms(lambdas)
    mine = {row["position_id"]: row["action_id"] for row in rows if row["arm"] == b24_end}
    theirs = {
        row["position_id"]: row["action_id"]
        for row in rows
        if row["arm"] == B24_LARGE_ARM
    }
    shared = sorted(set(mine) & set(theirs))
    differing = [key for key in shared if mine[key] != theirs[key]]
    return {
        "passed": not differing and bool(shared),
        "endpoint_arm": b24_end,
        "frozen_arm": B24_LARGE_ARM,
        "positions": len(shared),
        "differing_positions": len(differing),
        "findings": (
            []
            if not differing
            else [
                f"the lambda=1 endpoint chose a different action at "
                f"{len(differing)} of {len(shared)} positions"
            ]
        ),
    }


def check_shared_candidate_set(rows: "list[dict]") -> dict:
    """Every arm's action must lie in the oracle's candidate set.

    The property that makes oracle regret comparable at all: candidates come
    from P24's root policy, which no belief provider can influence. If this
    ever failed, the regret column would be measuring two different things.
    """
    outside = [
        (row["arm"], row["position_id"])
        for row in rows
        if row.get("action_in_oracle_candidates") in (0, "0")
    ]
    return {
        "passed": not outside,
        "decisions": len(rows),
        "outside": len(outside),
        "findings": [f"{arm} at {position}" for arm, position in outside[:6]],
    }


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------


def analyse_stage2(entries: "list[dict]", reference_arm: str) -> dict:
    """EWR, paired deltas, worst opponent, latency and game length per arm."""
    from .analysis import arm_summary, paired_delta

    by_arm: dict[str, list] = {}
    seconds_by_arm: dict[str, dict] = {}
    fallbacks_by_arm: dict[str, dict] = {}
    for entry in entries:
        row = entry["row"]
        key = row["arm_id"] + "|" + row["preset_id"]
        by_arm.setdefault(key, []).append(row)
        seconds_by_arm.setdefault(key, {})[row["board_id"]] = (
            entry.get("move_seconds") or []
        )
        for reason, count in (entry.get("fallback_reasons") or {}).items():
            bucket = fallbacks_by_arm.setdefault(key, {})
            bucket[reason] = bucket.get(reason, 0) + int(count)

    reference_rows = by_arm.get(reference_arm, [])
    report = {}
    for key, rows in by_arm.items():
        summary = arm_summary(rows, seconds_by_arm.get(key, {}))
        report[key] = {
            "arm": key,
            "games": summary["games"],
            "wins": summary["wins"],
            "draws": summary["draws"],
            "losses": summary["losses"],
            "ewr": summary["ewr"],
            "paired_vs_reference": (
                paired_delta(rows, reference_rows) if key != reference_arm else None
            ),
            "worst_opponent": summary["min_opponent"],
            "ewr_by_opponent": summary["ewr_by_opponent"],
            "worst_family": summary["min_family"],
            "pack_latency": {
                "median_seconds_per_move": summary.get("median_seconds_per_move"),
                "p95_seconds_per_move": summary.get("p95_seconds_per_move"),
                "max_seconds_per_move": summary.get("max_seconds_per_move"),
                "note": "measured under process contention, not idle",
            },
            "fallbacks": summary["fallbacks"],
            "fallback_rate": summary["fallback_rate"],
            "fallback_reasons": fallbacks_by_arm.get(key, {}),
            "player_decisions": summary["player_decisions"],
            "mean_plies": summary["mean_plies"],
        }
    return report


def decide_stage2(rungs: dict, *, medium_arm: str, large_arm: str, mix_arm: str) -> dict:
    """Adopt the mixture only if it gives back a meaningful part of -0.075."""
    medium = rungs.get(medium_arm) or {}
    large = rungs.get(large_arm) or {}
    mix = rungs.get(mix_arm) or {}
    large_gap = (large.get("ewr") or 0.0) - (medium.get("ewr") or 0.0)
    mix_gap = (mix.get("ewr") or 0.0) - (medium.get("ewr") or 0.0)
    recovered = None
    if large_gap < 0:
        recovered = round((mix_gap - large_gap) / abs(large_gap), 4)
    paired = (mix.get("paired_vs_reference") or {}) if mix else {}
    clean = (
        mix.get("fallbacks") == 0
        and not (mix.get("fallback_reasons") or {})
    )
    adopt = bool(
        recovered is not None
        and recovered >= 0.5
        and paired.get("delta") is not None
        and clean
    )
    return {
        "rule": (
            "adopt the mixture for deeper search only if it recovers at least "
            "half of LARGE's regression against MEDIUM without introducing "
            "correctness or fallback problems"
        ),
        "medium_ewr": medium.get("ewr"),
        "large_ewr": large.get("ewr"),
        "mixture_ewr": mix.get("ewr"),
        "large_gap_vs_medium": round(large_gap, 5),
        "mixture_gap_vs_medium": round(mix_gap, 5),
        "fraction_of_regression_recovered": recovered,
        "mixture_fallbacks": mix.get("fallbacks"),
        "correctness_clean": clean,
        "adopt_mixture_for_deeper_search": adopt,
        "recommendation": (
            "LARGE + mixture for a deeper-search mode" if adopt else "keep MEDIUM + B24"
        ),
    }


__all__ = [
    "B24_LARGE_ARM",
    "MixTask",
    "B24_MEDIUM_ARM",
    "COUNT_LARGE_ARM",
    "MIXTURE_VERSION",
    "ORACLE_ARM",
    "Phase15MixtureError",
    "analyse_stage2",
    "build_stage1_arms",
    "check_endpoint_identity",
    "check_reference_arms_reproduce",
    "check_shared_candidate_set",
    "decide_stage2",
    "endpoint_arms",
    "interior_arms",
    "mix_arm_name",
    "paired_regret_delta",
    "reference_comparisons",
    "run_stage1",
    "run_stage2_pack",
    "run_stage2_task",
    "select_lambda",
    "summarize_stage1",
]


# ---------------------------------------------------------------------------
# Running Stage 2
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MixTask:
    """One (arm, preset, board) unit of Stage 2 work.

    `lam` is `None` for a frozen arm and a mixture weight for a mixture arm;
    it is the only thing that distinguishes the two paths, and the frozen
    arms take the accepted builder untouched.
    """

    arm_id: str
    preset_name: str
    board_id: str
    lam: "float | None" = None
    keep_moves: bool = True

    @property
    def key(self) -> tuple:
        return (self.arm_id, self.preset_name, self.board_id)


def _stage2_system(task: MixTask):
    """One assembled system, cached per worker exactly as the accepted pack does."""
    from . import execution
    from .contract import pairing as pairing_of
    from .systems import build_engine

    state = execution._STATE
    cache_key = (task.arm_id, task.preset_name)
    bundle = state["systems"].get(cache_key)
    if bundle is None:
        if task.lam is None:
            target = pairing_of(task.arm_id)
            bundle = build_engine(
                target,
                state["models"],
                task.preset_name,
                production=target.kind != "diagnostic",
                device=state["device"],
            )
        else:
            bundle = build_mixture_bundle(
                state["models"],
                float(task.lam),
                task.preset_name,
                device=state["device"],
            )
            if bundle.pairing.pairing_id != task.arm_id:
                raise Phase15MixtureError(
                    f"task names arm {task.arm_id!r} but lambda {task.lam} builds "
                    f"{bundle.pairing.pairing_id!r}"
                )
        state["systems"][cache_key] = bundle
    return bundle


def run_stage2_task(task: MixTask) -> dict:
    """Play one board with one arm. The accepted `run_task`, with one branch.

    The body is deliberately the accepted
    :func:`stratego.search.phase15.execution.run_task`'s body: same seat,
    same `play_board`, same row shape. Only system *resolution* differs,
    because a mixture arm is not in the frozen pairing table.
    """
    import time as _time

    from . import execution
    from .matchplay import play_board
    from .systems import build_seat

    bundle = _stage2_system(task)
    plan = execution._plan(task.board_id)
    seat = build_seat(bundle, execution._STATE["owners"])
    started = _time.perf_counter()
    record = play_board(
        plan,
        seat,
        execution._STATE["owners"],
        preset_id=task.preset_name,
        keep_moves=task.keep_moves,
    )
    row = record.row()
    row["preset_id"] = task.preset_name
    row["wall_seconds"] = round(_time.perf_counter() - started, 4)
    if task.keep_moves:
        row["actions"] = [int(move["action_id"]) for move in record.moves]
    return {
        "row": row,
        "move_seconds": [round(value, 6) for value in record.move_seconds],
        "fallback_reasons": dict(getattr(seat, "fallbacks", {}) or {}),
        "probe": None,
    }


def run_stage2_pack(
    tasks: "list[MixTask]",
    *,
    root: str = ".",
    device: str = "cpu",
    workers: int = 8,
    progress=None,
) -> "list[dict]":
    """Every Stage 2 task, in pack order, over `workers` processes.

    All three arms run in one pack so their latency is measured under the
    same contention; comparing a median move time from a quiet pack against
    one from a busy pack would be comparing the machines, not the arms.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from . import execution

    tasks = list(tasks)
    if not tasks:
        return []
    if int(workers) <= 1:
        execution._worker_init(str(root), device, False, configure_threads=False)
        results = []
        for index, task in enumerate(tasks):
            results.append(run_stage2_task(task))
            if progress is not None:
                progress(index + 1, len(tasks), results[-1])
        return results

    results: "list[dict | None]" = [None] * len(tasks)
    completed = 0
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=execution._worker_init,
        initargs=(str(root), device, False),
    ) as pool:
        futures = {
            pool.submit(run_stage2_task, task): index
            for index, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            completed += 1
            if progress is not None:
                progress(completed, len(tasks), results[index])
    missing = [index for index, value in enumerate(results) if value is None]
    if missing:  # pragma: no cover - a future that neither returned nor raised
        raise Phase15MixtureError(f"{len(missing)} Stage 2 tasks produced no result")
    return results


def check_reference_arms_reproduce(fresh: "list[dict]", stored: "list[dict]") -> dict:
    """A fresh reference arm must reproduce its deeper-pilot row exactly.

    The two frozen arms — MEDIUM + B24 and LARGE + B24 — were already played
    on these boards with these seeds. Replaying them proves the mixture pack
    is running the same code under the same conditions, which is what lets a
    paired delta be read as a belief effect rather than a pack effect.
    """
    stored_by_key = {
        (row["arm_id"], row["preset_id"], row["board_id"]): row for row in stored
    }
    findings = []
    compared = 0
    for row in fresh:
        key = (row["arm_id"], row["preset_id"], row["board_id"])
        reference = stored_by_key.get(key)
        if reference is None:
            continue
        compared += 1
        for field in ("outcome", "effective_score", "plies"):
            if row.get(field) != reference.get(field):
                findings.append(
                    f"{key}: {field} is {row.get(field)!r}, the stored row has "
                    f"{reference.get(field)!r}"
                )
        if (row.get("actions") or []) != (reference.get("actions") or []):
            findings.append(f"{key}: the played action sequence differs")
    return {
        "passed": not findings and compared > 0,
        "games_compared": compared,
        "findings": findings[:6],
    }
