"""Phase 16 Agent 2 Stage 1: position diagnostics for the temperature grid.

Specification source: `02_AGENT_2_STOCHASTIC_SEARCH.md` section 3.

The design, and why the searches are shared
-------------------------------------------
Every arm answers the same 120 fresh replayed positions under the accepted
fixed world seed (`DECISION_SEED`, the Phase 15 Stage A seed), so the worlds
are identical across arms and replays and the argmax control is constant
across replays *by construction* — its repeat rate is 1.0 because it is the
same frozen decision sixteen times, which is exactly what "argmax control =
1.0" means in the brief.

The sixteen "reseeded replays" reseed the two *sampling* streams only:

- move draws: `move_sample_seed(tau, tau_r, position_id, ply, replay)`;
- rollout draws: `rollout_sample_seed(tau_r, top_p, position_id, ply, replay)`.

The rollout stream is keyed by the rollout configuration and not by `tau`,
so every move-temperature arm at the same `tau_r` shares the same sixteen
underlying searches. A Stage 1 difference between two `tau` arms is then a
pure move-sampling effect on identical score vectors, and the grid costs
`1 + 16` searches per (position, budget) instead of `8 x 16`.

Oracle Q-regret, exactly as the mixture pilot read it
------------------------------------------------------
The oracle runs once per (position, budget) at the same rung and seed. Root
candidates come from P24's policy alone, so every arm and the oracle evaluate
the same candidate set (asserted per decision), which makes
``max_a Q_oracle(a) - Q_oracle(a chosen)`` well defined everywhere. The
regret column is read as *excess over the common beta-floor* — the oracle's
own regret under its S-selection — exactly as
`reports/phase15/agent_02_mixture_report.md` reads it, with the machinery
reached by import.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ...engine.legal_moves import legal_actions
from ..phase15.contract import (
    MATCH_LIBRARY_SPLIT,
    MATCH_OPPONENTS,
    MATCH_SETUP_SOURCES,
)
from ..phase15.mixture_pilot import _oracle_reference  # reuse by import (brief section 3)
from ..phase15.positions import (
    POSITIONS_PER_GAME,
    DiagnosticPosition,
    materialize_positions,
    play_for_positions,
)
from .contract import (
    CONTROL_ARM,
    DECISION_SEED,
    MOVE_TAUS,
    POSITION_ORDINAL_BASE_16,
    POSITION_PACK_VERSION,
    REGRET_EXCESS_MARGIN,
    ROLLOUT_TAUS,
    ROLLOUT_TOP_P,
    STAGE1_REPLAYS,
    STAGE1_VERSION,
    STAGE_BUDGETS,
    STOCHASTIC_PAIRING,
    Phase16StochasticError,
)
from .stochastic import StochasticArm, build_stochastic_bundle, move_rng, sample_move

#: The oracle reference's arm label in the Stage 1 rows.
ORACLE_ARM_16 = "oracle_reference"

#: The Stage 1 observer: positions are harvested from games played by the
#: *direct* selected move model, never by a search arm (Phase 15 rule).
POSITION_OBSERVER = "p24"

#: How many harvest games the generator plans initially; it extends past
#: this deterministically if short games under-fill the target.
POSITION_GAMES = 30

#: The pack size the brief asks for.
POSITION_TARGET = 120


class Phase16DiagnosticsError(Phase16StochasticError):
    """A Stage 1 diagnostic could not be generated or run."""


# ---------------------------------------------------------------------------
# Fresh positions (Phase 15 pattern, new ordinals, single observer)
# ---------------------------------------------------------------------------


def position_cells_16(games: int = POSITION_GAMES) -> "list[tuple[str, str, str, str, int]]":
    """`(observer, opponent, source, colour, ordinal)` cells, ordinals 200+.

    The Phase 15 rotation pattern with one observer — P24, the selected
    system's move model — so all thirty games rotate through the ten
    opponents three times, the three sources and both colours stay balanced,
    and the ordinals cannot collide with any Phase 15 pack (Stage B used
    0-1, the Phase 15 diagnostic pack 100-114).
    """
    cells = []
    for game in range(int(games)):
        opponent = MATCH_OPPONENTS[game % len(MATCH_OPPONENTS)]
        source = MATCH_SETUP_SOURCES[game % len(MATCH_SETUP_SOURCES)]
        color = ("red", "blue")[game % 2]
        cells.append(
            (POSITION_OBSERVER, opponent, source, color, POSITION_ORDINAL_BASE_16 + game)
        )
    return cells


def generate_positions_16(
    owners: dict,
    *,
    target: int = POSITION_TARGET,
    per_game: int = POSITIONS_PER_GAME,
    sources=None,
    library_split: str = MATCH_LIBRARY_SPLIT,
    progress=None,
) -> "list[DiagnosticPosition]":
    """The fresh Stage 1 pack: exactly `target` replayable positions.

    Reuses the accepted Phase 15 harvest (`play_for_positions`) and board
    construction unchanged. If the planned games under-fill the target
    (short games yield fewer eligible decisions), further cells are added
    deterministically — same rotation, next ordinals — and the first
    `target` positions in generation order are kept.
    """
    from ..phase15.boards import Phase15MatchSetupSources, board_plan

    sources = Phase15MatchSetupSources() if sources is None else sources
    positions: list[DiagnosticPosition] = []
    game = 0
    hard_cap = POSITION_GAMES * 4
    while len(positions) < int(target):
        if game >= hard_cap:  # pragma: no cover - would need pathological games
            raise Phase16DiagnosticsError(
                f"{game} harvest games produced only {len(positions)} positions"
            )
        cells = position_cells_16(game + 1)
        observer, opponent, source, color, ordinal = cells[game]
        plan = board_plan(
            opponent, source, color, ordinal, sources, library_split=library_split
        )
        found = play_for_positions(plan, observer, owners, per_game=per_game)
        positions.extend(found)
        game += 1
        if progress is not None:
            progress(game, len(positions))
    return positions[: int(target)]


def build_position_manifest_16(
    positions: "list[DiagnosticPosition]", *, generated_utc: str, **extra
) -> dict:
    """The Phase 16 position manifest, digest-bound like the Phase 15 one."""
    import hashlib
    import json

    from ...belief.phase15.orientation import ORIENTATION_RULE, ORIENTATION_RULE_VERSION
    from ..phase15.positions import MIN_PLY, MIN_UNRESOLVED

    rows = [position.describe() for position in positions]
    by_opponent: dict[str, int] = {}
    for position in positions:
        by_opponent[position.opponent] = by_opponent.get(position.opponent, 0) + 1
    payload = {
        "artifact": POSITION_PACK_VERSION,
        "generated_utc": generated_utc,
        "orientation_rule_version": ORIENTATION_RULE_VERSION,
        "orientation_rule": ORIENTATION_RULE,
        "eligibility": {"min_ply": MIN_PLY, "min_unresolved": MIN_UNRESOLVED},
        "positions_per_game": POSITIONS_PER_GAME,
        "position_ordinal_base": POSITION_ORDINAL_BASE_16,
        "observer_model": POSITION_OBSERVER,
        "library_split": MATCH_LIBRARY_SPLIT,
        "note": (
            "fresh Phase 16 Stage 1 positions; the Phase 15 pattern with one "
            "observer (P24, the selected system) and ordinals 200+, so no "
            "Phase 15 board or position is reused"
        ),
        "positions": rows,
        "position_count": len(rows),
        "balance_by_opponent": dict(sorted(by_opponent.items())),
        **extra,
    }
    payload["manifest_digest"] = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


# ---------------------------------------------------------------------------
# The Stage 1 grid on one chunk of positions
# ---------------------------------------------------------------------------


def stage1_engines(
    models,
    *,
    budgets=STAGE_BUDGETS,
    rollout_taus=ROLLOUT_TAUS,
    top_p: float = ROLLOUT_TOP_P,
    pairing_id: str = STOCHASTIC_PAIRING,
    device: str = "cpu",
) -> dict:
    """Every engine Stage 1 needs, built once, keyed `(budget, kind)`.

    `kind` is `'oracle'`, or a rollout-temperature float. The oracle is
    built `production=False` through the accepted builder — the same
    offline-diagnostic path the Phase 15 stages used.
    """
    from ..phase15.systems import build_engine

    move_model = pairing_id.split("_", 1)[0]
    engines: dict = {}
    for budget in budgets:
        engines[(budget, "oracle")] = build_engine(
            f"{move_model}_oracle", models, budget, production=False, device=device
        )
        for tau_r in rollout_taus:
            arm = StochasticArm(0.0, float(tau_r), top_p=top_p, pairing_id=pairing_id)
            engines[(budget, float(tau_r))] = build_stochastic_bundle(
                models, arm, budget, device=device
            )
    return engines


def run_stage1_positions(
    models,
    replayed,
    *,
    budgets=STAGE_BUDGETS,
    taus=MOVE_TAUS,
    rollout_taus=ROLLOUT_TAUS,
    replays: int = STAGE1_REPLAYS,
    top_p: float = ROLLOUT_TOP_P,
    pairing_id: str = STOCHASTIC_PAIRING,
    seed: int = DECISION_SEED,
    device: str = "cpu",
    engines: "dict | None" = None,
    progress=None,
) -> "list[dict]":
    """The full grid on `replayed` `(row, state, plan)` triples.

    One row per (budget, arm, position, replay) plus one oracle-reference
    row per (budget, position). The candidate-set identity between every
    arm and the oracle is asserted per decision, not assumed.
    """
    from .stochastic import rollout_seed_for_arm

    if engines is None:
        engines = stage1_engines(
            models,
            budgets=budgets,
            rollout_taus=rollout_taus,
            top_p=top_p,
            pairing_id=pairing_id,
            device=device,
        )
    rows: list[dict] = []
    for index, (position, state, _plan) in enumerate(replayed):
        position_id = position["position_id"]
        ply = int(position["ply"])
        legal = set(legal_actions(state))
        for budget in budgets:
            # 1. the oracle reference at the same rung and seed
            oracle_bundle = engines[(budget, "oracle")]
            started = time.perf_counter()
            oracle_decision = oracle_bundle.engine.choose_action(state, seed=seed)
            oracle_seconds = time.perf_counter() - started
            reference = _oracle_reference(oracle_decision)
            floor = float(reference["best_q"]) - float(
                reference["q_by_action"][reference["selected_action_id"]]
            )
            rows.append(
                {
                    "position_id": position_id,
                    "preset_id": budget,
                    "arm": ORACLE_ARM_16,
                    "replay": 0,
                    "ply": ply,
                    "unresolved": int(position["unresolved"]),
                    "action_id": int(reference["selected_action_id"]),
                    "oracle_q_regret": round(floor, 6),
                    "legal": int(reference["selected_action_id"] in legal),
                    "seconds": round(oracle_seconds, 5),
                    "candidates": int(reference["candidates"]),
                }
            )

            # 2. the underlying searches, one per rollout configuration
            decisions_by_tau_r: dict[float, list] = {}
            for tau_r in rollout_taus:
                bundle = engines[(budget, float(tau_r))]
                if float(tau_r) == 0.0:
                    started = time.perf_counter()
                    base = bundle.engine.choose_action(state, seed=seed)
                    elapsed = time.perf_counter() - started
                    decisions_by_tau_r[0.0] = [(base, elapsed)] * int(replays)
                else:
                    arm0 = StochasticArm(0.0, float(tau_r), top_p=top_p, pairing_id=pairing_id)
                    bucket = []
                    for replay in range(int(replays)):
                        rollout_seed = rollout_seed_for_arm(arm0, position_id, ply, replay)
                        started = time.perf_counter()
                        decision = bundle.engine.choose_action(
                            state, seed=seed, rollout_seed=rollout_seed
                        )
                        bucket.append((decision, time.perf_counter() - started))
                    decisions_by_tau_r[float(tau_r)] = bucket

            # the shared-candidate property that makes regret comparable
            oracle_candidates = set(reference["q_by_action"])
            for tau_r, bucket in decisions_by_tau_r.items():
                for decision, _elapsed in bucket[:1] if tau_r == 0.0 else bucket:
                    observed = {
                        int(candidate.absolute_action_id)
                        for candidate in decision.candidates
                    }
                    if observed != oracle_candidates:
                        raise Phase16DiagnosticsError(
                            f"{position_id} @ {budget}: tau_r={tau_r} evaluated a "
                            "different candidate set than the oracle; regret is "
                            "not comparable"
                        )

            control_action = int(decisions_by_tau_r[0.0][0][0].selected_action_id)

            # 3. every arm, every replay: one seeded move draw
            for tau_r in rollout_taus:
                bucket = decisions_by_tau_r[float(tau_r)]
                for tau in taus:
                    arm = StochasticArm(
                        float(tau), float(tau_r), top_p=top_p, pairing_id=pairing_id
                    )
                    for replay in range(int(replays)):
                        decision, search_seconds = bucket[replay]
                        rng = move_rng(arm, position_id, ply, replay)
                        action, record = sample_move(decision, arm.tau, rng)
                        regret = ""
                        if action in reference["q_by_action"]:
                            regret = round(
                                float(reference["best_q"])
                                - float(reference["q_by_action"][action]),
                                6,
                            )
                        rows.append(
                            {
                                "position_id": position_id,
                                "preset_id": budget,
                                "arm": arm.arm_id,
                                "tau": float(tau),
                                "tau_r": float(tau_r),
                                "replay": int(replay),
                                "ply": ply,
                                "unresolved": int(position["unresolved"]),
                                "action_id": int(action),
                                "argmax_action_id": int(decision.selected_action_id),
                                "direct_action_id": int(decision.direct_action_id),
                                "changed_from_argmax": int(
                                    bool(record["changed_from_argmax"])
                                ),
                                "move_changed_vs_direct": int(
                                    int(action) != int(decision.direct_action_id)
                                ),
                                "matches_control": int(int(action) == control_action),
                                "matches_oracle": int(
                                    int(action) == int(reference["selected_action_id"])
                                ),
                                "oracle_q_regret": regret,
                                "legal": int(action in legal),
                                "search_seconds": round(float(search_seconds), 5),
                                "c1_forwards": int(decision.c1_forwards),
                                "unique_worlds": int(decision.unique_worlds),
                                "candidates": len(decision.candidates),
                            }
                        )
        if progress is not None:
            progress(index + 1, len(replayed), len(rows))
    return rows


# ---------------------------------------------------------------------------
# Parallel execution (the accepted worker pattern, Phase 16 state)
# ---------------------------------------------------------------------------

_STAGE1_STATE: dict = {}


def stage1_worker_init(
    root: str,
    device: str,
    manifest_path: str,
    budgets,
    taus,
    rollout_taus,
    replays: int,
    top_p: float,
    pairing_id: str,
    seed: int,
) -> None:
    """Load one worker's own frozen stack and engines, exactly once.

    `torch.set_num_threads` is process-local and sufficient; nothing here
    touches `os.environ` (Phase 15 defect 0).
    """
    import json as _json

    import torch

    torch.set_num_threads(1)
    from ..phase15.loaders import load_all

    models = load_all(root=root, device=device, with_anchor=False)
    manifest = _json.loads(Path(manifest_path).read_text())
    _STAGE1_STATE.update(
        {
            "models": models,
            "manifest": manifest,
            "budgets": tuple(budgets),
            "taus": tuple(float(value) for value in taus),
            "rollout_taus": tuple(float(value) for value in rollout_taus),
            "replays": int(replays),
            "top_p": float(top_p),
            "pairing_id": str(pairing_id),
            "seed": int(seed),
            "device": device,
            "engines": stage1_engines(
                models,
                budgets=tuple(budgets),
                rollout_taus=tuple(float(value) for value in rollout_taus),
                top_p=float(top_p),
                pairing_id=str(pairing_id),
                device=device,
            ),
        }
    )


def stage1_worker_chunk(bounds: "tuple[int, int]") -> "list[dict]":
    """Run the grid on positions `[start, stop)` of the worker's manifest."""
    start, stop = bounds
    state = _STAGE1_STATE
    subset = dict(state["manifest"])
    subset["positions"] = state["manifest"]["positions"][start:stop]
    replayed = materialize_positions(subset)
    return run_stage1_positions(
        state["models"],
        replayed,
        budgets=state["budgets"],
        taus=state["taus"],
        rollout_taus=state["rollout_taus"],
        replays=state["replays"],
        top_p=state["top_p"],
        pairing_id=state["pairing_id"],
        seed=state["seed"],
        device=state["device"],
        engines=state["engines"],
    )


def run_stage1_pack(
    manifest_path: "str | Path",
    *,
    root: str = ".",
    device: str = "cpu",
    budgets=STAGE_BUDGETS,
    taus=MOVE_TAUS,
    rollout_taus=ROLLOUT_TAUS,
    replays: int = STAGE1_REPLAYS,
    top_p: float = ROLLOUT_TOP_P,
    pairing_id: str = STOCHASTIC_PAIRING,
    seed: int = DECISION_SEED,
    workers: int = 8,
    chunk_size: int = 4,
    progress=None,
) -> "list[dict]":
    """The whole Stage 1 grid over `workers` processes, rows in pack order."""
    import json as _json
    from concurrent.futures import ProcessPoolExecutor, as_completed

    manifest = _json.loads(Path(manifest_path).read_text())
    count = len(manifest["positions"])
    bounds = [
        (start, min(start + int(chunk_size), count))
        for start in range(0, count, int(chunk_size))
    ]
    init_args = (
        str(root),
        device,
        str(manifest_path),
        tuple(budgets),
        tuple(taus),
        tuple(rollout_taus),
        int(replays),
        float(top_p),
        str(pairing_id),
        int(seed),
    )
    if int(workers) <= 1:
        stage1_worker_init(*init_args)
        rows: list[dict] = []
        for index, chunk in enumerate(bounds):
            rows.extend(stage1_worker_chunk(chunk))
            if progress is not None:
                progress(index + 1, len(bounds), len(rows))
        return rows

    results: "list[list | None]" = [None] * len(bounds)
    completed = 0
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=stage1_worker_init,
        initargs=init_args,
    ) as pool:
        futures = {
            pool.submit(stage1_worker_chunk, chunk): index
            for index, chunk in enumerate(bounds)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            completed += 1
            if progress is not None:
                progress(completed, len(bounds), sum(len(r) for r in results if r))
    rows = []
    for chunk_rows in results:
        if chunk_rows is None:  # pragma: no cover - a future that vanished
            raise Phase16DiagnosticsError("a Stage 1 chunk produced no result")
        rows.extend(chunk_rows)
    return rows


# ---------------------------------------------------------------------------
# Reading Stage 1
# ---------------------------------------------------------------------------


def _entropy_nats(counts: "dict[int, int]") -> float:
    total = sum(counts.values())
    probabilities = np.asarray([count / total for count in counts.values()])
    return float(-np.sum(probabilities * np.log(probabilities)))


def summarize_stage1(rows: "list[dict]") -> dict:
    """Per (arm, preset): the brief's three diagnostics plus regret excess."""
    oracle_floor: dict[str, list] = {}
    for row in rows:
        if row["arm"] == ORACLE_ARM_16:
            oracle_floor.setdefault(row["preset_id"], []).append(
                float(row["oracle_q_regret"])
            )
    floors = {
        preset: round(float(np.mean(values)), 6) for preset, values in oracle_floor.items()
    }

    by_key: dict[tuple, list] = {}
    for row in rows:
        if row["arm"] == ORACLE_ARM_16:
            continue
        by_key.setdefault((row["arm"], row["preset_id"]), []).append(row)

    report: dict = {"oracle_regret_floor_by_preset": floors, "arms": {}}
    for (arm, preset), entries in sorted(by_key.items()):
        by_position: dict[str, list] = {}
        for row in entries:
            by_position.setdefault(row["position_id"], []).append(row)
        repeat_rates, entropies, distinct = [], [], []
        for position_rows in by_position.values():
            counts: dict[int, int] = {}
            for row in position_rows:
                counts[int(row["action_id"])] = counts.get(int(row["action_id"]), 0) + 1
            repeat_rates.append(max(counts.values()) / len(position_rows))
            entropies.append(_entropy_nats(counts))
            distinct.append(len(counts))
        regrets = [
            float(row["oracle_q_regret"])
            for row in entries
            if row.get("oracle_q_regret") not in ("", None)
        ]
        regret_mean = round(float(np.mean(regrets)), 6) if regrets else None
        floor = floors.get(preset)
        report["arms"][f"{arm}|{preset}"] = {
            "arm": arm,
            "preset_id": preset,
            "tau": entries[0].get("tau"),
            "tau_r": entries[0].get("tau_r"),
            "positions": len(by_position),
            "replays_per_position": len(entries) // max(len(by_position), 1),
            "decisions": len(entries),
            "repeat_rate": round(float(np.mean(repeat_rates)), 5),
            "played_move_entropy_nats": round(float(np.mean(entropies)), 5),
            "mean_distinct_actions": round(float(np.mean(distinct)), 3),
            "agreement_with_tau0": round(
                float(np.mean([row["matches_control"] for row in entries])), 5
            ),
            "oracle_agreement": round(
                float(np.mean([row["matches_oracle"] for row in entries])), 5
            ),
            "move_change_rate_vs_direct": round(
                float(np.mean([row["move_changed_vs_direct"] for row in entries])), 5
            ),
            "sampled_away_from_argmax_rate": round(
                float(np.mean([row["changed_from_argmax"] for row in entries])), 5
            ),
            "oracle_q_regret_mean": regret_mean,
            "oracle_q_regret_excess_over_floor": (
                None
                if regret_mean is None or floor is None
                else round(regret_mean - floor, 6)
            ),
            "oracle_q_regret_n": len(regrets),
            "illegal_decisions": sum(1 for row in entries if not int(row["legal"])),
            "median_search_seconds_contended": round(
                float(np.median([row["search_seconds"] for row in entries])), 5
            ),
            "mean_c1_forwards": round(
                float(np.mean([row["c1_forwards"] for row in entries])), 1
            ),
        }
    return report


def apply_stage1_filter(
    summary: dict,
    *,
    budgets=STAGE_BUDGETS,
    margin: float = REGRET_EXCESS_MARGIN,
) -> dict:
    """The predeclared filter (brief section 3), applied per arm.

    An arm survives if its mean oracle Q-regret excess is within `margin` of
    the tau=0 control's **at every Stage 1 budget** — the conservative
    reading, declared before the numbers were seen. The control survives by
    definition. The full grid is reported regardless of the filter.
    """
    arms = summary.get("arms", {})
    arm_ids = sorted({entry["arm"] for entry in arms.values()})
    verdicts = {}
    for arm in arm_ids:
        checks = {}
        survives = True
        for budget in budgets:
            mine = arms.get(f"{arm}|{budget}", {})
            control = arms.get(f"{CONTROL_ARM}|{budget}", {})
            excess = mine.get("oracle_q_regret_excess_over_floor")
            control_excess = control.get("oracle_q_regret_excess_over_floor")
            if excess is None or control_excess is None:
                survives = False
                checks[budget] = {"available": False}
                continue
            delta = round(excess - control_excess, 6)
            passed = delta <= float(margin) + 1e-12
            checks[budget] = {
                "available": True,
                "excess": excess,
                "control_excess": control_excess,
                "delta_vs_control": delta,
                "within_margin": passed,
            }
            survives = survives and passed
        if arm == CONTROL_ARM:
            survives = True
        verdicts[arm] = {"survives": survives, "budgets": checks}
    return {
        "rule": (
            "an arm survives if its mean oracle Q-regret excess is within "
            f"+{margin} of the tau=0 control at every Stage 1 budget; the "
            "control survives by definition; the full grid is reported regardless"
        ),
        "margin": float(margin),
        "control_arm": CONTROL_ARM,
        "verdicts": verdicts,
        "survivors": [arm for arm, entry in verdicts.items() if entry["survives"]],
    }


__all__ = [
    "ORACLE_ARM_16",
    "POSITION_GAMES",
    "POSITION_OBSERVER",
    "POSITION_TARGET",
    "Phase16DiagnosticsError",
    "apply_stage1_filter",
    "build_position_manifest_16",
    "generate_positions_16",
    "materialize_positions",
    "position_cells_16",
    "run_stage1_pack",
    "run_stage1_positions",
    "stage1_engines",
    "stage1_worker_chunk",
    "stage1_worker_init",
    "summarize_stage1",
]
