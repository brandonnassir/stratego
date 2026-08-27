"""Phase 16 Agent 2 Stages 2 and 5-6: the match pack, the probe, the caps.

Specification source: `02_AGENT_2_STOCHASTIC_SEARCH.md` sections 4, 5, 6.

Boards
------
Stage 2 prefers Agent 1's `phase16_benchmark_v1` (through its frozen handoff
document, digests re-verified) and falls back — as the brief declares — to a
fresh 60-board balanced set drawn exactly per the Phase 15 Stage C rules:
one board per (opponent x setup source x colour) cell, from the accepted
library's `validation` split, through the accepted orientation gate, at a
fresh ordinal no Phase 15 pack used. The fallback pack is named
`phase16_agent02_interim_pack_v1`.

Pairing
-------
Every arm plays the same board list with the same accepted seed streams; the
paired comparison against the `tau = tau_r = 0` control is therefore a
difference between two decision procedures on one position, exactly as the
accepted Phase 15 packs read theirs. The analysis is the accepted
`analyse_stage2` from the mixture pilot, reached by import.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..phase15.boards import BOARD_CELLS, Phase15BoardPlan, board_plan
from ..phase15.contract import parse_board_id
from ..phase15.mixture_pilot import analyse_stage2  # reuse by import
from .contract import (
    CONTROL_ARM,
    FALLBACK_TAU,
    FALLBACK_TAU_R,
    INTERIM_PACK_ORDINAL,
    INTERIM_PACK_VERSION,
    MEASUREMENT_HANDOFF_ARTIFACT,
    PROBE_GAMES_PER_OPPONENT,
    PROBE_OPPONENTS,
    PROBE_ORDINAL_BASE,
    PROBE_VERSION,
    ROLLOUT_TOP_P,
    STAGE2_EWR_MARGIN,
    STOCHASTIC_PAIRING,
    Phase16StochasticError,
    arm_name,
)
from .stochastic import (
    StochasticArm,
    StochasticSeat,
    build_stochastic_bundle,
    sample_move,
)


class Phase16MatchError(Phase16StochasticError):
    """A Stage 2 pack could not be built or run."""


# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------


def interim_pack_plans(sources=None) -> "list[Phase15BoardPlan]":
    """The fallback pack: 60 fresh boards, one per Stage C cell, ordinal 2.

    Ordinal 2 collides with nothing: Phase 15 Stage B used ordinals 0-1, the
    Stage C / deeper-pilot pack used ordinal 0, the Phase 15 diagnostic pack
    100-114, the Phase 16 position pack 200+, and the probe 300+.
    """
    from ..phase15.boards import Phase15MatchSetupSources

    sources = Phase15MatchSetupSources() if sources is None else sources
    plans = []
    for cell in BOARD_CELLS:
        opponent, source, color = cell
        plans.append(
            board_plan(
                opponent,
                source,
                color,
                INTERIM_PACK_ORDINAL,
                sources,
                cell_index=BOARD_CELLS.index(cell),
            )
        )
    return plans


def build_interim_manifest(plans: "list[Phase15BoardPlan]", *, generated_utc: str) -> dict:
    import hashlib

    boards = []
    for plan in plans:
        row = plan.describe()
        row["red_setup"] = list(plan.red_setup)
        row["blue_setup"] = list(plan.blue_setup)
        boards.append(row)
    payload = {
        "artifact": INTERIM_PACK_VERSION,
        "generated_utc": generated_utc,
        "rules": (
            "one board per (opponent x setup source x colour) cell, the Phase 15 "
            "Stage C construction unchanged, at fresh ordinal "
            f"{INTERIM_PACK_ORDINAL}"
        ),
        "board_count": len(boards),
        "boards": boards,
    }
    payload["manifest_digest"] = hashlib.sha256(
        json.dumps(boards, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def resolve_stage2_boards(root: "Path | str" = ".") -> dict:
    """Agent 1's benchmark if its handoff has landed and verifies; else the
    declared fallback. Returns `{pack_name, board_ids, source, detail}`."""
    root = Path(root)
    handoff_path = root / "reports/phase16" / f"{MEASUREMENT_HANDOFF_ARTIFACT}.json"
    if handoff_path.is_file():
        try:
            handoff = json.loads(handoff_path.read_text())
            if handoff.get("artifact") != MEASUREMENT_HANDOFF_ARTIFACT:
                raise Phase16MatchError(
                    f"{handoff_path} is not a {MEASUREMENT_HANDOFF_ARTIFACT} document"
                )
            benchmark = handoff.get("benchmark") or {}
            manifest_path = root / benchmark["manifest_path"]
            payload = manifest_path.read_bytes()
            import hashlib

            observed = hashlib.sha256(payload).hexdigest()
            expected = benchmark.get("manifest_sha256")
            if expected and observed != expected:
                raise Phase16MatchError(
                    f"{manifest_path} hashes to {observed}, the handoff records "
                    f"{expected}"
                )
            manifest = json.loads(payload)
            boards = manifest.get("boards") or []
            board_ids = [row["board_id"] for row in boards]
            subset = benchmark.get("quick_subset_board_ids") or manifest.get(
                "quick_subset_board_ids"
            )
            if subset:
                board_ids = list(subset)
                pack_name = f"{benchmark.get('pack_name', 'phase16_benchmark_v1')}|quick_subset"
            else:
                pack_name = benchmark.get("pack_name", "phase16_benchmark_v1")
            for identifier in board_ids:
                parse_board_id(identifier)
            return {
                "pack_name": pack_name,
                "board_ids": board_ids,
                "source": "agent1_handoff",
                "detail": {
                    "handoff_path": str(handoff_path),
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": observed,
                    "boards": len(board_ids),
                },
            }
        except (KeyError, OSError, ValueError, Phase16MatchError) as error:
            detail = f"handoff present but unusable ({error}); using the declared fallback"
        else:  # pragma: no cover - unreachable
            detail = ""
    else:
        detail = "Agent 1's handoff has not landed; using the declared fallback"
    plans = interim_pack_plans()
    return {
        "pack_name": INTERIM_PACK_VERSION,
        "board_ids": [plan.board_id for plan in plans],
        "source": "interim_fallback",
        "detail": {"note": detail, "boards": len(plans)},
    }


# ---------------------------------------------------------------------------
# Running the pack
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StochTask:
    """One (arm, preset, board) unit of Stage 2 / probe work."""

    tau: float
    tau_r: float
    preset_name: str
    board_id: str
    top_p: float = ROLLOUT_TOP_P
    pairing_id: str = STOCHASTIC_PAIRING
    keep_moves: bool = True

    @property
    def arm_id(self) -> str:
        return arm_name(self.tau, self.tau_r)

    @property
    def key(self) -> tuple:
        return (self.arm_id, self.preset_name, self.board_id)


def _stage2_system(task: StochTask):
    """One assembled stochastic system, cached per worker.

    The cache and the loaded models live in the accepted Phase 15 worker
    state (`stratego.search.phase15.execution._STATE`), exactly as the
    accepted mixture pilot did — one loader, one owner set, one plan cache
    for every pack this agent runs.
    """
    from ..phase15 import execution

    state = execution._STATE
    cache_key = ("phase16", task.arm_id, task.preset_name)
    bundle = state["systems"].get(cache_key)
    if bundle is None:
        arm = StochasticArm(
            task.tau, task.tau_r, top_p=task.top_p, pairing_id=task.pairing_id
        )
        bundle = build_stochastic_bundle(
            state["models"], arm, task.preset_name, device=state["device"]
        )
        state["systems"][cache_key] = bundle
    return bundle


def run_stage2_task(task: StochTask) -> dict:
    """Play one board with one arm. The accepted task body with a
    stochastic seat in the player's chair."""
    from ..phase15 import execution
    from ..phase15.matchplay import play_board

    bundle = _stage2_system(task)
    plan = execution._plan(task.board_id)
    arm = StochasticArm(task.tau, task.tau_r, top_p=task.top_p, pairing_id=task.pairing_id)
    seat = StochasticSeat(arm, bundle, owners=execution._STATE["owners"])
    started = time.perf_counter()
    record = play_board(
        plan,
        seat,
        execution._STATE["owners"],
        preset_id=task.preset_name,
        keep_moves=task.keep_moves,
    )
    row = record.row()
    row["arm_id"] = seat.arm_id
    row["tau"] = float(task.tau)
    row["tau_r"] = float(task.tau_r)
    row["top_p"] = float(task.top_p)
    row["preset_id"] = task.preset_name
    row["wall_seconds"] = round(time.perf_counter() - started, 4)
    row["sampled_move_changes"] = int(seat.sampled_changes)
    if task.keep_moves:
        row["actions"] = [int(move["action_id"]) for move in record.moves]
    return {
        "row": row,
        "move_seconds": [round(value, 6) for value in record.move_seconds],
        "fallback_reasons": dict(seat.fallbacks or {}),
    }


def run_stage2_pack(
    tasks: "list[StochTask]",
    *,
    root: str = ".",
    device: str = "cpu",
    workers: int = 8,
    progress=None,
) -> "list[dict]":
    """Every task, in pack order, over `workers` processes.

    `workers=1` runs in-process through the same code path — the accepted
    serial-reference pattern.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from ..phase15 import execution

    tasks = list(tasks)
    if not tasks:
        return []
    if int(workers) <= 1:
        execution._worker_init(str(root), device, True, configure_threads=False)
        results = []
        for index, task in enumerate(tasks):
            results.append(run_stage2_task(task))
            if progress is not None:
                progress(index + 1, len(tasks), results[-1])
        return results

    from ..phase15.loaders import ensure_phase9_anchor

    ensure_phase9_anchor(root=str(root))
    results: "list[dict | None]" = [None] * len(tasks)
    completed = 0
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=execution._worker_init,
        initargs=(str(root), device, True),
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
        raise Phase16MatchError(f"{len(missing)} Stage 2 tasks produced no result")
    return results


def analyse_pack(entries: "list[dict]", *, reference_preset: str) -> dict:
    """The accepted mixture-pilot analysis, keyed `arm|preset`, with the
    tau=0 control at `reference_preset` as the paired reference."""
    return analyse_stage2(entries, f"{CONTROL_ARM}|{reference_preset}")


# ---------------------------------------------------------------------------
# The predeclared selection (brief section 4)
# ---------------------------------------------------------------------------


def select_configuration(
    stage2_by_arm: dict,
    stage1_summary: dict,
    *,
    selection_preset: str = "MEDIUM",
    margin: float = STAGE2_EWR_MARGIN,
) -> dict:
    """Among arms with EWR within `margin` of the control at MEDIUM, the
    lowest Stage 1 repeat rate at MEDIUM wins; ties break to lower tau then
    lower tau_r; if none qualifies, the named fallback arm if *it* is within
    the margin, else keep argmax."""
    control_key = f"{CONTROL_ARM}|{selection_preset}"
    control = stage2_by_arm.get(control_key) or {}
    control_ewr = control.get("ewr")
    if control_ewr is None:
        raise Phase16MatchError(
            f"the control has no Stage 2 EWR at {selection_preset}; selection "
            "cannot be applied"
        )
    arms = {}
    for key, entry in stage2_by_arm.items():
        arm, _, preset = key.partition("|")
        if preset != selection_preset or arm == CONTROL_ARM:
            continue
        stage1 = (stage1_summary.get("arms") or {}).get(f"{arm}|{selection_preset}") or {}
        arms[arm] = {
            "ewr": entry.get("ewr"),
            "delta_vs_control": (
                None if entry.get("ewr") is None else round(entry["ewr"] - control_ewr, 5)
            ),
            "paired_vs_control": entry.get("paired_vs_reference"),
            "repeat_rate": stage1.get("repeat_rate"),
            "tau": stage1.get("tau"),
            "tau_r": stage1.get("tau_r"),
            "games": entry.get("games"),
        }
    qualifiers = {
        arm: entry
        for arm, entry in arms.items()
        if entry["ewr"] is not None
        and entry["ewr"] >= control_ewr - float(margin) - 1e-12
        and entry["repeat_rate"] is not None
    }
    selected = None
    reason = ""
    if qualifiers:
        selected = min(
            qualifiers,
            key=lambda arm: (
                qualifiers[arm]["repeat_rate"],
                qualifiers[arm]["tau"],
                qualifiers[arm]["tau_r"],
            ),
        )
        reason = (
            f"lowest Stage 1 repeat rate at {selection_preset} among arms with EWR "
            f"within {margin} of the control"
        )
    else:
        fallback = arm_name(FALLBACK_TAU, FALLBACK_TAU_R)
        entry = arms.get(fallback)
        if (
            entry is not None
            and entry["ewr"] is not None
            and entry["ewr"] >= control_ewr - float(margin) - 1e-12
        ):
            selected = fallback
            reason = (
                "no arm qualified outright; the brief's named fallback "
                f"(tau={FALLBACK_TAU}, tau_r={FALLBACK_TAU_R}) is within the margin"
            )
        else:
            reason = (
                "no viable stochastic mode: no arm, including the named fallback, "
                "kept EWR within the margin of the argmax control — argmax is kept"
            )
    result = {
        "rule": (
            f"among arms with EWR within {margin} of the control at "
            f"{selection_preset}, select the one with the lowest Stage 1 repeat "
            f"rate (ties: lower tau, then lower tau_r); if none qualifies, select "
            f"tau={FALLBACK_TAU}, tau_r={FALLBACK_TAU_R} if it is within {margin}; "
            "else report no-viable-stochastic-mode and keep argmax"
        ),
        "selection_preset": selection_preset,
        "margin": float(margin),
        "control_arm": CONTROL_ARM,
        "control_ewr": control_ewr,
        "arms": arms,
        "qualifiers": sorted(qualifiers),
        "selected_arm": selected if selected is not None else CONTROL_ARM,
        "stochastic_mode_viable": selected is not None,
        "reason": reason,
    }
    if selected is not None:
        result["selected_tau"] = arms[selected]["tau"]
        result["selected_tau_r"] = arms[selected]["tau_r"]
    else:
        result["selected_tau"] = 0.0
        result["selected_tau_r"] = 0.0
    return result


# ---------------------------------------------------------------------------
# The repeat-encounter probe (brief section 5)
# ---------------------------------------------------------------------------


def probe_plan(opponent: str, index: int, sources=None) -> Phase15BoardPlan:
    """A fresh board for probe game `index` against `opponent`.

    Sources rotate with the index, colours alternate, ordinals 300+ so no
    other pack shares a board.
    """
    from ..phase15.boards import Phase15MatchSetupSources
    from ..phase15.contract import MATCH_SETUP_SOURCES

    sources = Phase15MatchSetupSources() if sources is None else sources
    source = MATCH_SETUP_SOURCES[int(index) % len(MATCH_SETUP_SOURCES)]
    color = ("red", "blue")[int(index) % 2]
    return board_plan(
        opponent, source, color, PROBE_ORDINAL_BASE + int(index), sources
    )


def probe_tasks(
    arms: "list[StochasticArm]",
    *,
    preset: str,
    opponents=PROBE_OPPONENTS,
    games: int = PROBE_GAMES_PER_OPPONENT,
) -> "list[StochTask]":
    tasks = []
    for arm in arms:
        for opponent in opponents:
            for index in range(int(games)):
                plan = probe_plan(opponent, index)
                tasks.append(
                    StochTask(
                        tau=arm.tau,
                        tau_r=arm.tau_r,
                        preset_name=preset,
                        board_id=plan.board_id,
                        top_p=arm.top_p,
                        pairing_id=arm.pairing_id,
                        keep_moves=False,
                    )
                )
    return tasks


def analyse_probe(entries: "list[dict]") -> dict:
    """Per-index EWR trend per arm, plus a least-squares slope.

    Fixed bots cannot adapt, so this is a weak proxy by construction — the
    caller's report must say so — and no significance claim is made.
    """
    by_arm: dict[str, list] = {}
    for entry in entries:
        row = entry["row"]
        by_arm.setdefault(row["arm_id"], []).append(row)
    report = {}
    for arm, rows in sorted(by_arm.items()):
        by_index: dict[int, list] = {}
        for row in rows:
            index = int(row["ordinal"]) - PROBE_ORDINAL_BASE
            by_index.setdefault(index, []).append(float(row["effective_score"]))
        indices = sorted(by_index)
        series = [
            {
                "game_index": index,
                "games": len(by_index[index]),
                "ewr": round(float(np.mean(by_index[index])), 4),
            }
            for index in indices
        ]
        xs = np.asarray(
            [index for index in indices for _ in by_index[index]], dtype=np.float64
        )
        ys = np.asarray(
            [score for index in indices for score in by_index[index]], dtype=np.float64
        )
        slope = None
        if len(xs) > 1 and float(np.var(xs)) > 0:
            slope = round(float(np.polyfit(xs, ys, 1)[0]), 6)
        halves = None
        if indices:
            midpoint = (max(indices) + 1) / 2
            first = [score for index in indices if index < midpoint for score in by_index[index]]
            second = [score for index in indices if index >= midpoint for score in by_index[index]]
            if first and second:
                halves = {
                    "first_half_ewr": round(float(np.mean(first)), 4),
                    "first_half_games": len(first),
                    "second_half_ewr": round(float(np.mean(second)), 4),
                    "second_half_games": len(second),
                }
        report[arm] = {
            "games": len(rows),
            "ewr": round(float(np.mean([row["effective_score"] for row in rows])), 4),
            "per_index": series,
            "ewr_slope_per_game_index": slope,
            "halves": halves,
        }
    return {
        "artifact": PROBE_VERSION,
        "note": (
            "fixed bots cannot adapt across games, so a flat trend here is a weak "
            "proxy only; the real adaptation test is the operator series"
        ),
        "arms": report,
    }


# ---------------------------------------------------------------------------
# Idle latency and caps (brief section 6)
# ---------------------------------------------------------------------------


def measure_idle_latency(
    models,
    replayed,
    arm: StochasticArm,
    *,
    presets=("TINY", "MEDIUM"),
    device: str = "cpu",
) -> dict:
    """Single-process idle latency of full varied-mode decisions.

    One decision = one (possibly sampled-rollout) search plus one softmax
    draw, timed together, on replayed diagnostic positions — the accepted
    Phase 15 idle-measurement pattern. Must be run on an otherwise idle
    machine; the caller records that context.
    """
    from ..phase15.contract import search_seed_for
    from .stochastic import move_rng, rollout_seed_for_arm
    from .engine import Phase16StochasticEngine

    profiles = {}
    for preset in presets:
        bundle = build_stochastic_bundle(models, arm, preset, device=device)
        timings, forwards = [], []
        sampled_away = 0
        for position, state, _plan in replayed:
            identifier = position["position_id"]
            ply = int(position["ply"])
            seed = search_seed_for(identifier, ply)
            started = time.perf_counter()
            if isinstance(bundle.engine, Phase16StochasticEngine):
                decision = bundle.engine.choose_action(
                    state,
                    seed=seed,
                    rollout_seed=rollout_seed_for_arm(arm, identifier, ply),
                )
            else:
                decision = bundle.engine.choose_action(state, seed=seed)
            action, record = sample_move(
                decision, arm.tau, move_rng(arm, identifier, ply)
            )
            timings.append(time.perf_counter() - started)
            forwards.append(decision.c1_forwards)
            if record.get("changed_from_argmax"):
                sampled_away += 1
            del action
        array = np.asarray(timings, dtype=np.float64)
        profiles[preset] = {
            "preset_id": preset,
            "decisions": len(timings),
            "median_seconds_per_move": round(float(np.median(array)), 5),
            "p95_seconds_per_move": round(float(np.percentile(array, 95)), 5),
            "max_seconds_per_move": round(float(array.max()), 5),
            "mean_c1_forwards": round(float(np.mean(forwards)), 1),
            "sampled_away_from_argmax": sampled_away,
        }
    return profiles


def decide_time_caps(idle_profiles: dict, phase15_caps: dict, phase15_idle: dict) -> dict:
    """The predeclared cap rule.

    Sampling adds one softmax draw per decision, so the frozen Phase 15 caps
    are kept unless the measured idle p95 exceeds the Phase 15 idle p95 by
    more than 10%, in which case the cap becomes `min(3.5 x p95, 5.0)` —
    the Phase 15 headroom rule under the Phase 15 ceiling — and the change
    is flagged.
    """
    from .contract import ACCEPTABLE_MOVE_SECONDS

    caps = {}
    findings = []
    for preset, profile in idle_profiles.items():
        p95 = float(profile["p95_seconds_per_move"])
        reference_p95 = float((phase15_idle.get(preset) or {}).get("p95", p95))
        keep = p95 <= reference_p95 * 1.10
        if keep:
            caps[preset] = float(phase15_caps[preset])
        else:
            caps[preset] = round(min(3.5 * p95, ACCEPTABLE_MOVE_SECONDS), 2)
            findings.append(
                f"{preset}: idle p95 {p95:.3f}s exceeds the Phase 15 idle p95 "
                f"{reference_p95:.3f}s by more than 10%; cap re-derived"
            )
    return {
        "rule": (
            "keep the frozen Phase 15 caps unless idle p95 grew by more than 10%, "
            "then min(3.5 x p95, 5.0)"
        ),
        "caps_seconds": caps,
        "changed": bool(findings),
        "findings": findings,
    }


__all__ = [
    "Phase16MatchError",
    "StochTask",
    "analyse_pack",
    "analyse_probe",
    "build_interim_manifest",
    "decide_time_caps",
    "interim_pack_plans",
    "measure_idle_latency",
    "probe_plan",
    "probe_tasks",
    "resolve_stage2_boards",
    "run_stage2_pack",
    "run_stage2_task",
    "select_configuration",
]
