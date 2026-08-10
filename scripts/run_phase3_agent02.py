#!/usr/bin/env python3
"""Phase 3 Agent 2 acceptance harness.

Runs the multiprocess/shared-memory gates that are too slow for the ordinary
pytest run and writes `reports/phase_3_data/agent_02_shared_memory_scaling.json`
plus `reports/phase_3_data/agent_02_shared_memory_scaling_raw.csv`:

- >= 25,000 cross-process environment steps compared, one by one, against
  independently stepped frozen reference games;
- >= 5,000 reset events distributed across workers, each with a full
  neighbouring-slot isolation check;
- the worker failure surface: a killed worker, a hung worker and a worker that
  raises;
- the CPU scaling screen over workers 4/6/8/10/12 and environments
  256/512/1,024/1,536/2,048, followed by longer runs of the best configurations;
- the automated test suite summary.

No PyTorch, no Metal, no inference: this is the central-processing-unit path
only, as specified.

Usage:

    python scripts/run_phase3_agent02.py                 # full acceptance run
    python scripts/run_phase3_agent02.py --quick         # fast smoke run
    python scripts/run_phase3_agent02.py --skip-pytest   # measurements only
    python scripts/run_phase3_agent02.py --skip-benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.engine.constants import (  # noqa: E402
    IMPLEMENTATION_VERSION,
    OBSERVATION_VERSION,
    RULES_VERSION,
    TRAINING_RULES,
    RulesConfig,
)
from stratego.engine.legal_moves import legal_action_mask, legal_actions  # noqa: E402
from stratego.engine.observation import build_observation  # noqa: E402
from stratego.engine.transition import apply_action  # noqa: E402
from stratego.training.batch_simulation import (  # noqa: E402
    BATCH_INTERFACE_VERSION,
    NO_ACTING_PLAYER,
)
from stratego.training.shared_buffers import (  # noqa: E402
    SHARED_BUFFER_VERSION,
    SKIP_ACTION,
    STATUS_ACTIVE,
    STATUS_TERMINAL,
    SharedEnvironmentBuffers,
    buffer_nbytes,
    max_resident_bytes,
)
from stratego.training.worker_pool import (  # noqa: E402
    THREAD_LIMIT_VARIABLES,
    WORKER_POOL_VERSION,
    WorkerCrashError,
    WorkerFaultError,
    WorkerPool,
    WorkerTimeoutError,
    collect_finished,
    select_action,
)
from tests.training.differential import reference_game  # noqa: E402

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_3_data"
DEFAULT_OUTPUT = DATA_DIRECTORY / "agent_02_shared_memory_scaling.json"
DEFAULT_RAW_OUTPUT = DATA_DIRECTORY / "agent_02_shared_memory_scaling_raw.csv"

WORKER_COUNTS = (4, 6, 8, 10, 12)
ENVIRONMENT_COUNTS = (256, 512, 1024, 1536, 2048)

LOGICAL_CORES = os.cpu_count() or 1


# ---------------------------------------------------------------------------
# Stage 1: cross-process equivalence
# ---------------------------------------------------------------------------


def compare_row(buffers, slot: int, reference, generation: int) -> list[str]:
    """Compare one shared-memory row against an independent reference game.

    The reference side never touches `stratego.training`: it is a `GameState`
    built by the frozen engine and advanced with `apply_action`. Only the seed
    derivation is shared, which is the definition of "deterministically seeded"
    rather than a shortcut.
    """
    problems: list[str] = []
    if int(buffers.environment_id[slot]) != slot:
        problems.append("environment_id differs")
    if int(buffers.generation[slot]) != generation:
        problems.append("generation differs")
    if int(buffers.ply[slot]) != reference.total_moves:
        problems.append("ply differs")
    if int(buffers.battleless_moves[slot]) != reference.battleless_moves:
        problems.append("battleless counter differs")
    if bool(buffers.terminal[slot]) != reference.terminal:
        problems.append("terminal flag differs")

    expected_acting = NO_ACTING_PLAYER if reference.terminal else reference.acting_player
    if int(buffers.acting_player[slot]) != expected_acting:
        problems.append("acting player differs")

    if reference.terminal:
        if int(buffers.status[slot]) != STATUS_TERMINAL:
            problems.append("status is not terminal")
        if buffers.legal_mask[slot].any():
            problems.append("terminal slot published a non-empty legality mask")
        if buffers.observations[slot].any():
            problems.append("terminal slot published an observation")
        return problems

    if int(buffers.status[slot]) != STATUS_ACTIVE:
        problems.append("status is not active")
    if not np.array_equal(buffers.observations[slot], build_observation(reference)):
        problems.append("observation differs")
    legal = legal_actions(reference)
    if not np.array_equal(buffers.legal_mask[slot], legal_action_mask(reference, legal)):
        problems.append("dense legal mask differs")
    if int(buffers.legal_count[slot]) != len(legal):
        problems.append("legal count differs")
    return problems


def compare_outcome(reported: dict, reference, generation: int) -> list[str]:
    """Compare a collected terminal outcome against the reference game."""
    problems: list[str] = []
    if reported["generation"] != generation:
        problems.append("outcome generation differs")
    if reported["terminal_reason"] != reference.terminal_reason:
        problems.append("terminal reason differs")
    if reported["winner"] != reference.winner:
        problems.append("winner differs")
    if reported["is_draw"] != reference.is_draw:
        problems.append("draw flag differs")
    if reported["total_moves"] != reference.total_moves:
        problems.append("final ply differs")
    if reported["result_for_red"] != reference.result_for(0):
        problems.append("red result differs")
    if reported["result_for_blue"] != reference.result_for(1):
        problems.append("blue result differs")
    return problems


def equivalence_run(
    *,
    num_environments: int,
    num_workers: int,
    root_seed: int,
    target_steps: int,
    auto_reset: bool,
    rules: RulesConfig = TRAINING_RULES,
    max_details: int = 5,
) -> dict:
    """Drive a real worker pool and independent reference games in lockstep."""
    started = time.perf_counter()
    pool = WorkerPool(
        num_environments,
        num_workers,
        root_seed=root_seed,
        rules=rules,
        step_timeout=600.0,
    )
    pool.start()
    buffers = pool.buffers

    generations = {slot: 0 for slot in range(num_environments)}
    references = {
        slot: reference_game(root_seed, slot, 0, rules) for slot in range(num_environments)
    }
    episodes = np.zeros(num_environments, dtype=np.int64)

    mismatches = 0
    details: list[dict] = []
    steps = 0
    batch_steps = 0
    completed = 0
    resets = 0
    action_checks = 0
    terminal_reasons: Counter = Counter()

    def note(slot: int, problems: list[str], stage: str) -> None:
        nonlocal mismatches
        if not problems:
            return
        mismatches += len(problems)
        if len(details) < max_details:
            details.append(
                {
                    "stage": stage,
                    "slot": slot,
                    "environment_id": int(buffers.environment_id[slot]),
                    "generation": generations[slot],
                    "ply": references[slot].total_moves,
                    "action_history": list(references[slot].action_history)[-16:],
                    "problems": problems,
                }
            )

    try:
        # The first publish is a comparison point too: it proves a worker
        # process reproduces the slot its identity seeds.
        for slot in range(num_environments):
            note(slot, compare_row(buffers, slot, references[slot], 0), "initial_publish")

        while steps < target_steps:
            actions = pool.select_actions(buffers.actions)

            # The coordinator picks from a dense mask; the reference picks from
            # a legal-action list. Requiring the two to agree proves the mask
            # carries the same legality the engine generated *and* that both
            # sides are about to make the same transition.
            for slot in range(num_environments):
                reference = references[slot]
                if reference.terminal:
                    expected = SKIP_ACTION
                else:
                    expected = select_action(
                        root_seed,
                        slot,
                        generations[slot],
                        reference.total_moves,
                        legal_actions(reference),
                    )
                action_checks += 1
                if int(actions[slot]) != expected:
                    note(
                        slot,
                        [
                            f"selected action differs: shared mask gave "
                            f"{int(actions[slot])}, reference list gave {expected}"
                        ],
                        "action_selection",
                    )

            pool.step(auto_reset=auto_reset)
            batch_steps += 1

            for slot in range(num_environments):
                if int(actions[slot]) >= 0:
                    apply_action(references[slot], int(actions[slot]))
                    steps += 1

            # Terminal results are read from the persistent `last_*` fields, so
            # an immediate reset cannot lose them.
            reported = {
                outcome["environment_id"]: outcome
                for outcome in collect_finished(buffers, episodes)
            }
            finished = [
                slot for slot in range(num_environments) if references[slot].terminal
            ]
            for slot in finished:
                completed += 1
                terminal_reasons[references[slot].terminal_reason] += 1
                outcome = reported.pop(slot, None)
                if outcome is None:
                    note(slot, ["finished game was not reported to the coordinator"], "terminal")
                    continue
                note(slot, compare_outcome(outcome, references[slot], generations[slot]), "terminal")
            for slot in sorted(reported):
                note(slot, ["an outcome was reported for a game that did not finish"], "terminal")

            if not auto_reset and finished:
                # Exercise the coordinator-driven reset policy: the terminal
                # rows stay published until the coordinator asks for a reset.
                for slot in finished:
                    note(
                        slot,
                        compare_row(buffers, slot, references[slot], generations[slot]),
                        "terminal_row",
                    )
                pool.request_reset(finished)
                pool.step(apply_actions=False, auto_reset=False)
                batch_steps += 1

            for slot in finished:
                generations[slot] += 1
                references[slot] = reference_game(root_seed, slot, generations[slot], rules)
                resets += 1

            for slot in range(num_environments):
                note(
                    slot,
                    compare_row(buffers, slot, references[slot], generations[slot]),
                    "post_step",
                )
    finally:
        pool.shutdown()

    elapsed = time.perf_counter() - started
    return {
        "num_environments": num_environments,
        "num_workers": num_workers,
        "root_seed": root_seed,
        "auto_reset": auto_reset,
        "cross_process_steps": steps,
        "batch_steps": batch_steps,
        "action_selection_checks": action_checks,
        "row_comparisons": batch_steps * num_environments + num_environments,
        "games_completed": completed,
        "resets": resets,
        "terminal_reason_counts": dict(sorted(terminal_reasons.items())),
        "equivalence_mismatches": mismatches,
        "mismatch_details": details,
        "elapsed_seconds": elapsed,
    }


def equivalence_stage(plan: list[dict]) -> dict:
    runs = []
    totals = {
        "cross_process_steps": 0,
        "equivalence_mismatches": 0,
        "row_comparisons": 0,
        "action_selection_checks": 0,
        "games_completed": 0,
        "resets": 0,
    }
    reasons: Counter = Counter()
    details: list[dict] = []
    for entry in plan:
        report = equivalence_run(**entry)
        runs.append(report)
        for key in totals:
            totals[key] += report[key]
        reasons.update(report["terminal_reason_counts"])
        details.extend(report["mismatch_details"])
        print(
            f"  {report['num_workers']:>2} workers / "
            f"{report['num_environments']:>4} envs: "
            f"{report['cross_process_steps']:>6} steps, "
            f"{report['equivalence_mismatches']} mismatches, "
            f"{report['games_completed']} games, "
            f"{report['elapsed_seconds']:6.1f}s",
            flush=True,
        )
    return {
        **totals,
        "terminal_reason_counts": dict(sorted(reasons.items())),
        "runs": runs,
        "mismatch_details": details,
    }


# ---------------------------------------------------------------------------
# Stage 2: reset isolation across workers
# ---------------------------------------------------------------------------


def reset_stage(
    *,
    num_environments: int,
    num_workers: int,
    root_seed: int,
    rounds: int,
    per_round: int,
    advance_steps: int = 2,
    stagger_spread: int = 211,
    rules: RulesConfig = TRAINING_RULES,
    max_details: int = 10,
) -> dict:
    """Reset slots on every worker while the rest of the batch stands still."""
    started = time.perf_counter()
    pool = WorkerPool(
        num_environments,
        num_workers,
        root_seed=root_seed,
        rules=rules,
        step_timeout=600.0,
    )
    pool.start()
    buffers = pool.buffers

    stride = max(1, num_environments // per_round)
    events = 0
    mismatches = 0
    generation_errors = 0
    isolation_checks = 0
    natural_resets = 0
    details: list[dict] = []
    trajectory_keys: set[tuple[int, int]] = {
        (slot, 0) for slot in range(num_environments)
    }
    workers_touched: set[int] = set()
    ply_spreads: list[int] = []

    def note(slot: int, problems: list[str], stage: str) -> None:
        nonlocal mismatches
        if not problems:
            return
        mismatches += len(problems)
        if len(details) < max_details:
            details.append(
                {
                    "stage": stage,
                    "slot": slot,
                    "generation": int(buffers.generation[slot]),
                    "problems": problems,
                }
            )

    try:
        # Put the slots at substantially different plies before any reset is
        # requested. A bulk-synchronous batch otherwise advances in lockstep,
        # which would make every isolation check compare slots that are all at
        # the same point of the same-length game. The dense action vector
        # already means "skip this slot" for a negative entry, so holding a slot
        # back needs no new mechanism. A coprime stride spreads the targets out
        # instead of producing a tight ramp.
        targets = (np.arange(num_environments) * 37) % stagger_spread
        for index in range(int(targets.max())):
            pool.select_actions(buffers.actions)
            buffers.actions[targets <= index] = SKIP_ACTION
            pool.step(auto_reset=True)
        stagger_ply_spread = int(buffers.ply.max() - buffers.ply.min())

        for round_index in range(rounds):
            # Keep the batch at mixed plies rather than letting it converge, so
            # every isolation check compares genuinely different positions.
            for _ in range(advance_steps):
                pool.select_actions(buffers.actions)
                report = pool.step(auto_reset=True)
                natural_resets += report.resets
            ply_spreads.append(int(buffers.ply.max() - buffers.ply.min()))

            selected = sorted(
                (round_index + index * stride) % num_environments
                for index in range(per_round)
            )
            selected = sorted(set(selected))
            untouched = [
                slot for slot in range(num_environments) if slot not in set(selected)
            ]
            workers_touched.update(int(buffers.worker_id[slot]) for slot in selected)

            before_rows = buffers.snapshot_rows(untouched)
            before_generation = buffers.generation.copy()
            before_environment = buffers.environment_id.copy()

            pool.request_reset(selected)
            pool.step(apply_actions=False, auto_reset=False)

            # Every field of every neighbouring slot must be byte-identical.
            # `publish_sequence` is the one exception: every slot advances it on
            # every phase, which is exactly how a stale buffer is detected.
            differing = [
                name
                for name in buffers.rows_equal(before_rows, untouched)
                if name != "publish_sequence"
            ]
            isolation_checks += len(untouched)
            if differing:
                note(
                    untouched[0],
                    [f"neighbouring slots changed during a reset: {differing}"],
                    "isolation",
                )

            for slot in selected:
                events += 1
                problems: list[str] = []
                generation = int(buffers.generation[slot])
                if generation != int(before_generation[slot]) + 1:
                    generation_errors += 1
                    problems.append("generation did not increment exactly once")
                if int(buffers.environment_id[slot]) != int(before_environment[slot]):
                    generation_errors += 1
                    problems.append("environment_id changed across a reset")
                key = (slot, generation)
                if key in trajectory_keys:
                    generation_errors += 1
                    problems.append(f"trajectory key {key} was reused")
                trajectory_keys.add(key)
                if int(buffers.ply[slot]) != 0:
                    problems.append("reset slot did not restart the ply counter")
                if int(buffers.battleless_moves[slot]) != 0:
                    problems.append("reset slot kept a battleless counter")
                if int(buffers.terminal[slot]) != 0:
                    problems.append("reset slot is still terminal")
                if int(buffers.status[slot]) != STATUS_ACTIVE:
                    problems.append("reset slot is not active")
                if int(buffers.legal_count[slot]) <= 0:
                    problems.append("reset slot has no legal action")

                # The decisive check: the slot now holds exactly the game its
                # new generation seeds, rebuilt independently by this process.
                expected = reference_game(root_seed, slot, generation, rules)
                if not np.array_equal(
                    buffers.observations[slot], build_observation(expected)
                ):
                    problems.append("reset slot is not the game its new generation seeds")
                if not np.array_equal(
                    buffers.legal_mask[slot], legal_action_mask(expected)
                ):
                    problems.append("reset slot published the wrong legality mask")
                if int(buffers.acting_player[slot]) != expected.acting_player:
                    problems.append("reset slot does not start with the first player")
                note(slot, problems, "fresh")
    finally:
        pool.shutdown()

    return {
        "reset_events": events,
        "reset_mismatches": mismatches,
        "generation_errors": generation_errors,
        "reset_isolation_checks": isolation_checks,
        "natural_resets_during_stage": natural_resets,
        "distinct_trajectory_keys": len(trajectory_keys),
        "workers_touched": sorted(workers_touched),
        "mean_ply_spread": float(np.mean(ply_spreads)) if ply_spreads else 0.0,
        "max_ply_spread": max(ply_spreads) if ply_spreads else 0,
        "stagger_ply_spread": stagger_ply_spread,
        "reset_problem_details": details,
        "num_environments": num_environments,
        "num_workers": num_workers,
        "elapsed_seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# Stage 3: worker failure surface
# ---------------------------------------------------------------------------


def failure_stage(*, root_seed: int = 8080) -> dict:
    """Deliberately break workers and require a clear infrastructure error."""
    results: dict = {"cases": [], "deadlocks": 0}

    def record(name: str, passed: bool, message: str) -> None:
        results["cases"].append(
            {"case": name, "detected": passed, "error": message[:400]}
        )

    # -- a worker process is killed outright ------------------------------
    pool = WorkerPool(64, 4, root_seed=root_seed, step_timeout=30.0)
    pool.start()
    try:
        pool.select_actions(pool.buffers.actions)
        pool.step()
        killed = pool.kill_worker(2)
        try:
            for _ in range(3):
                pool.select_actions(pool.buffers.actions)
                pool.step()
        except WorkerCrashError as error:
            record("killed_worker", True, f"pid {killed}: {error}")
        else:
            record("killed_worker", False, "the coordinator kept stepping")
    finally:
        pool.shutdown(timeout=10.0)

    # -- a worker stops responding ----------------------------------------
    pool = WorkerPool(64, 4, root_seed=root_seed + 1, step_timeout=2.0)
    pool.start()
    try:
        pool.stall_worker(1, 60.0)
        deadline = time.perf_counter() + 30.0
        try:
            pool.select_actions(pool.buffers.actions)
            pool.step()
        except WorkerTimeoutError as error:
            record("hung_worker", True, str(error))
        else:
            record("hung_worker", False, "the coordinator returned from a hung phase")
        if time.perf_counter() > deadline:
            results["deadlocks"] += 1
    finally:
        for worker_id in range(4):
            try:
                pool.kill_worker(worker_id)
            except Exception:  # noqa: BLE001 - teardown of a broken pool
                pass
        pool.shutdown(timeout=10.0)

    # -- a worker raises ---------------------------------------------------
    pool = WorkerPool(32, 2, root_seed=root_seed + 2, step_timeout=20.0)
    pool.start()
    results["worker_thread_limits"] = sorted(set(pool.worker_thread_limits.values()))
    results["thread_limit_variables"] = list(THREAD_LIMIT_VARIABLES)
    try:
        pool._sequence += 1
        for worker in pool._workers:
            worker.connection.send({"kind": "not-a-command", "sequence": pool._sequence})
        try:
            pool._await_replies(20.0, stage="step")
        except WorkerFaultError as error:
            record("worker_exception", True, str(error))
        else:
            record("worker_exception", False, "a malformed command was accepted")
    finally:
        pool.shutdown(timeout=10.0)

    results["worker_failure_detection_passed"] = all(
        case["detected"] for case in results["cases"]
    )
    return results


# ---------------------------------------------------------------------------
# Stage 4: CPU scaling benchmark
# ---------------------------------------------------------------------------


def benchmark_config(
    *,
    num_workers: int,
    num_environments: int,
    root_seed: int,
    seconds: float,
    warmup_steps: int,
    min_steps: int,
    label: str,
    rules: RulesConfig = TRAINING_RULES,
) -> dict:
    """Measure one (workers, environments) point of the CPU-only pipeline.

    The action policy is the cheap deterministic legal-action selection, so no
    model inference is involved and what is measured is observation building,
    legality generation, stepping, reset, shared-memory transport and worker
    synchronisation.
    """
    pool = WorkerPool(
        num_environments,
        num_workers,
        root_seed=root_seed,
        rules=rules,
        step_timeout=600.0,
    )
    error = ""
    try:
        pool.start()
        for _ in range(warmup_steps):
            pool.select_actions(pool.buffers.actions)
            pool.step()

        latencies: list[float] = []
        policy_seconds = 0.0
        wait_seconds = 0.0
        dispatch_seconds = 0.0
        straggler_seconds = 0.0
        worker_busy_seconds = 0.0
        worker_cpu_seconds = 0.0
        stepped = 0
        observation_builds = 0
        terminals = 0
        resets = 0

        coordinator_cpu_start = time.process_time()
        wall_start = time.perf_counter()
        steps = 0
        while steps < min_steps or time.perf_counter() - wall_start < seconds:
            policy_start = time.perf_counter()
            # The policy writes straight into the shared action buffer: the
            # coordinator's output never becomes a Python object either.
            pool.select_actions(pool.buffers.actions)
            policy_seconds += time.perf_counter() - policy_start

            report = pool.step(auto_reset=True)
            latencies.append(report.wall_seconds)
            wait_seconds += report.wait_seconds
            dispatch_seconds += report.dispatch_seconds
            straggler_seconds += report.straggler_seconds
            worker_busy_seconds += report.worker_busy_seconds
            worker_cpu_seconds += report.worker_cpu_seconds
            stepped += report.stepped
            observation_builds += report.observation_builds
            terminals += report.terminals
            resets += report.resets
            steps += 1

        wall = time.perf_counter() - wall_start
        coordinator_cpu = time.process_time() - coordinator_cpu_start
        startup_seconds = pool.startup_seconds
        shared_bytes = pool.buffers.nbytes
    except Exception as failure:  # noqa: BLE001 - a failed point must be recorded
        error = f"{type(failure).__name__}: {failure}"
        pool.shutdown(timeout=15.0)
        return {
            "label": label,
            "num_workers": num_workers,
            "num_environments": num_environments,
            "error": error,
            "positions_per_second": 0.0,
        }

    totals = pool.shutdown(timeout=30.0)
    latency = np.array(latencies)
    worker_rss = totals.get("worker_max_rss_bytes", 0)
    coordinator_rss = max_resident_bytes()

    return {
        "label": label,
        "num_workers": num_workers,
        "num_environments": num_environments,
        "environments_per_worker": num_environments / num_workers,
        "measured_steps": int(latency.size),
        "measured_seconds": wall,
        "startup_seconds": startup_seconds,
        "positions": stepped,
        "positions_per_second": stepped / wall,
        # One state transition per position in this layer; reported separately
        # because Agent 5's pipeline may evaluate several positions per move.
        "transitions_per_second": stepped / wall,
        "observation_builds": observation_builds,
        "observation_builds_per_second": observation_builds / wall,
        "mean_step_seconds": float(latency.mean()),
        "p50_step_seconds": float(np.percentile(latency, 50)),
        "p95_step_seconds": float(np.percentile(latency, 95)),
        "max_step_seconds": float(latency.max()),
        "microseconds_per_position": 1e6 * wall / max(stepped, 1),
        "coordinator_wait_fraction": wait_seconds / wall,
        "coordinator_policy_fraction": policy_seconds / wall,
        "coordinator_dispatch_fraction": dispatch_seconds / wall,
        "barrier_wait_fraction": straggler_seconds / wall,
        "worker_active_fraction": worker_busy_seconds / (wall * num_workers),
        "coordinator_cpu_seconds": coordinator_cpu,
        "worker_cpu_seconds": worker_cpu_seconds,
        "cpu_cores_busy": (worker_cpu_seconds + coordinator_cpu) / wall,
        "cpu_utilization": (worker_cpu_seconds + coordinator_cpu) / (wall * LOGICAL_CORES),
        "shared_memory_bytes": shared_bytes,
        "coordinator_max_rss_bytes": coordinator_rss,
        "worker_max_rss_bytes": worker_rss,
        "total_memory_bytes": shared_bytes + coordinator_rss + worker_rss,
        "terminals": terminals,
        "resets": resets,
        "error": "",
    }


def benchmark_stage(
    *,
    worker_counts: tuple[int, ...],
    environment_counts: tuple[int, ...],
    screen_seconds: float,
    long_seconds: float,
    long_configurations: int,
    root_seed: int,
    warmup_steps: int,
    min_steps: int,
) -> dict:
    """Short screen of every feasible pair, then longer runs of the best few."""
    screening: list[dict] = []
    for num_environments in environment_counts:
        for num_workers in worker_counts:
            if num_environments < num_workers:  # pragma: no cover - never true here
                continue
            result = benchmark_config(
                num_workers=num_workers,
                num_environments=num_environments,
                root_seed=root_seed,
                seconds=screen_seconds,
                warmup_steps=warmup_steps,
                min_steps=min_steps,
                label="screen",
            )
            screening.append(result)
            print(
                f"  screen {num_workers:>2}w x {num_environments:>4}e: "
                f"{result['positions_per_second']:>9,.0f} pos/s"
                + (f"  ERROR {result['error']}" if result["error"] else ""),
                flush=True,
            )

    ranked = sorted(
        (entry for entry in screening if not entry["error"]),
        key=lambda entry: entry["positions_per_second"],
        reverse=True,
    )
    measured: list[dict] = []
    for entry in ranked[:long_configurations]:
        result = benchmark_config(
            num_workers=entry["num_workers"],
            num_environments=entry["num_environments"],
            root_seed=root_seed + 1,
            seconds=long_seconds,
            warmup_steps=warmup_steps * 2,
            min_steps=min_steps,
            label="measured",
        )
        measured.append(result)
        print(
            f"  measure {result['num_workers']:>2}w x "
            f"{result['num_environments']:>4}e: "
            f"{result['positions_per_second']:>9,.0f} pos/s over "
            f"{result['measured_seconds']:.0f}s"
            + (f"  ERROR {result['error']}" if result["error"] else ""),
            flush=True,
        )

    candidates = [entry for entry in measured if not entry["error"]] or ranked
    best = max(candidates, key=lambda entry: entry["positions_per_second"], default=None)
    return {
        "screening_results": screening,
        "measured_results": measured,
        "best": best,
        "errors": [entry for entry in screening + measured if entry["error"]],
    }


# ---------------------------------------------------------------------------
# Stage 5: automated test suite summary
# ---------------------------------------------------------------------------


def pytest_stage() -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    output = completed.stdout + completed.stderr

    def count(label: str) -> int:
        match = re.search(rf"(\d+) {label}", output)
        return int(match.group(1)) if match else 0

    passed = count("passed")
    failed = count("failed")
    skipped = count("skipped")
    xfailed = count("xfailed")
    errors = count("error")

    return {
        "total": passed + failed + skipped + xfailed + errors,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "expected_failures": xfailed,
        "errors": errors,
        "seconds": elapsed,
        "exit_code": completed.returncode,
        "failure_lines": [line for line in output.splitlines() if line.startswith("FAILED")],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

FILES_CREATED = [
    "stratego/training/shared_buffers.py",
    "stratego/training/worker_pool.py",
    "tests/training/test_shared_buffers.py",
    "tests/training/test_worker_pool.py",
    "scripts/run_phase3_agent02.py",
    "reports/phase_3_data/agent_02_shared_memory_scaling.json",
    "reports/phase_3_data/agent_02_shared_memory_scaling_raw.csv",
]

FILES_MODIFIED = [
    "stratego/training/batch_simulation.py",
    "reports/phase_3_implementation_report.md",
]

CSV_COLUMNS = [
    "label",
    "num_workers",
    "num_environments",
    "environments_per_worker",
    "measured_steps",
    "measured_seconds",
    "startup_seconds",
    "positions",
    "positions_per_second",
    "transitions_per_second",
    "observation_builds_per_second",
    "microseconds_per_position",
    "mean_step_seconds",
    "p50_step_seconds",
    "p95_step_seconds",
    "max_step_seconds",
    "coordinator_wait_fraction",
    "coordinator_policy_fraction",
    "coordinator_dispatch_fraction",
    "barrier_wait_fraction",
    "worker_active_fraction",
    "coordinator_cpu_seconds",
    "worker_cpu_seconds",
    "cpu_cores_busy",
    "cpu_utilization",
    "shared_memory_bytes",
    "coordinator_max_rss_bytes",
    "worker_max_rss_bytes",
    "total_memory_bytes",
    "terminals",
    "resets",
    "error",
]


def write_raw_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--screen-seconds", type=float, default=4.0)
    parser.add_argument("--long-seconds", type=float, default=45.0)
    parser.add_argument("--long-configurations", type=int, default=3)
    parser.add_argument("--reset-rounds", type=int, default=160)
    parser.add_argument("--reset-per-round", type=int, default=32)
    parser.add_argument("--stagger-spread", type=int, default=211)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    options = parser.parse_args()

    # (environments, workers, seed, target env steps, coordinator reset policy).
    # Small batches run many batch steps and therefore finish many games; large
    # batches exercise the transport at the Phase 3 engineering scale.
    equivalence_plan = [
        {
            "num_environments": 32,
            "num_workers": 4,
            "root_seed": 9001,
            "target_steps": 8_000,
            "auto_reset": True,
        },
        {
            "num_environments": 128,
            "num_workers": 8,
            "root_seed": 9002,
            "target_steps": 12_000,
            "auto_reset": False,
        },
        {
            "num_environments": 256,
            "num_workers": 12,
            "root_seed": 9003,
            "target_steps": 10_000,
            "auto_reset": True,
        },
    ]
    worker_counts = WORKER_COUNTS
    environment_counts = ENVIRONMENT_COUNTS
    reset_environments, reset_workers = 256, 8

    if options.quick:
        equivalence_plan = [
            {
                "num_environments": 16,
                "num_workers": 4,
                "root_seed": 9001,
                "target_steps": 600,
                "auto_reset": True,
            },
            {
                "num_environments": 32,
                "num_workers": 4,
                "root_seed": 9002,
                "target_steps": 600,
                "auto_reset": False,
            },
        ]
        worker_counts = (4, 8)
        environment_counts = (256, 512)
        options.screen_seconds = 1.5
        options.long_seconds = 3.0
        options.long_configurations = 1
        options.reset_rounds = 4
        options.reset_per_round = 8
        reset_environments, reset_workers = 64, 4
        options.stagger_spread = 17

    started = time.perf_counter()

    print("stage: cross-process equivalence", flush=True)
    equivalence = equivalence_stage(equivalence_plan)

    print("stage: reset isolation", flush=True)
    reset = reset_stage(
        num_environments=reset_environments,
        num_workers=reset_workers,
        root_seed=9101,
        rounds=options.reset_rounds,
        per_round=options.reset_per_round,
        stagger_spread=options.stagger_spread,
    )
    print(
        f"  {reset['reset_events']} reset events across workers "
        f"{reset['workers_touched']}, {reset['reset_mismatches']} mismatches, "
        f"{reset['reset_isolation_checks']} isolation checks, "
        f"{reset['elapsed_seconds']:.1f}s",
        flush=True,
    )

    print("stage: worker failure surface", flush=True)
    failure = failure_stage()
    for case in failure["cases"]:
        print(f"  {case['case']}: {'detected' if case['detected'] else 'NOT DETECTED'}", flush=True)

    if options.skip_benchmark:
        benchmark = {
            "screening_results": [],
            "measured_results": [],
            "best": None,
            "errors": [],
        }
    else:
        print("stage: CPU scaling benchmark", flush=True)
        benchmark = benchmark_stage(
            worker_counts=worker_counts,
            environment_counts=environment_counts,
            screen_seconds=options.screen_seconds,
            long_seconds=options.long_seconds,
            long_configurations=options.long_configurations,
            root_seed=9201,
            warmup_steps=3,
            min_steps=8,
        )

    if options.skip_pytest:
        tests = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "expected_failures": 0,
            "errors": 0,
            "seconds": 0.0,
            "exit_code": None,
            "failure_lines": [],
        }
    else:
        print("stage: pytest", flush=True)
        tests = pytest_stage()
        print(f"  {tests['passed']} passed, {tests['failed']} failed", flush=True)

    elapsed = time.perf_counter() - started

    best = benchmark["best"]
    minimum_steps = 600 if options.quick else 25_000
    minimum_resets = 24 if options.quick else 5_000
    passed = (
        equivalence["cross_process_steps"] >= minimum_steps
        and equivalence["equivalence_mismatches"] == 0
        and reset["reset_events"] >= minimum_resets
        and reset["reset_mismatches"] == 0
        and reset["generation_errors"] == 0
        and failure["worker_failure_detection_passed"]
        and failure["deadlocks"] == 0
        and not benchmark["errors"]
        and (options.skip_benchmark or best is not None)
        and (options.skip_pytest or (tests["failed"] == 0 and tests["exit_code"] == 0))
    )

    # A one-slot block is the cheapest way to report the exact layout the run
    # used without hard-coding it in two places.
    reference_buffers = SharedEnvironmentBuffers.create(1)
    try:
        shapes = reference_buffers.shapes()
        shapes = {name: ["N"] + shape[1:] for name, shape in shapes.items()}
        dtypes = reference_buffers.dtypes()
        field_documentation = reference_buffers.field_documentation()
    finally:
        reference_buffers.close()
        reference_buffers.unlink()

    metrics = {
        "agent": "agent_02_shared_memory_cpu",
        "status": "PASS" if passed else "FAIL",
        "implementation_version": IMPLEMENTATION_VERSION,
        "batch_interface_version": BATCH_INTERFACE_VERSION,
        "shared_buffer_version": SHARED_BUFFER_VERSION,
        "worker_pool_version": WORKER_POOL_VERSION,
        "rules_version": RULES_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "logical_cores": LOGICAL_CORES,
        "start_method": "spawn",
        "quick_mode": options.quick,
        "worker_counts": list(worker_counts),
        "environment_counts": list(environment_counts),
        "shared_buffer_shapes": shapes,
        "shared_buffer_dtypes": dtypes,
        "shared_buffer_fields": field_documentation,
        "shared_buffer_bytes_by_environment_count": {
            str(count): buffer_nbytes(count) for count in environment_counts
        },
        # Equivalence
        "cross_process_steps": equivalence["cross_process_steps"],
        "equivalence_mismatches": equivalence["equivalence_mismatches"],
        "equivalence_mismatch_details": equivalence["mismatch_details"],
        "equivalence_row_comparisons": equivalence["row_comparisons"],
        "equivalence_action_selection_checks": equivalence["action_selection_checks"],
        "equivalence_games_completed": equivalence["games_completed"],
        "equivalence_natural_resets": equivalence["resets"],
        "equivalence_runs": equivalence["runs"],
        "terminal_reason_counts": equivalence["terminal_reason_counts"],
        # Reset isolation
        "reset_events": reset["reset_events"],
        "reset_mismatches": reset["reset_mismatches"],
        "reset_generation_errors": reset["generation_errors"],
        "reset_isolation_checks": reset["reset_isolation_checks"],
        "reset_distinct_trajectory_keys": reset["distinct_trajectory_keys"],
        "reset_workers_touched": reset["workers_touched"],
        "reset_mean_ply_spread": reset["mean_ply_spread"],
        "reset_max_ply_spread": reset["max_ply_spread"],
        "reset_stagger_ply_spread": reset["stagger_ply_spread"],
        "reset_problem_details": reset["reset_problem_details"],
        "reset_stage_configuration": {
            "num_environments": reset["num_environments"],
            "num_workers": reset["num_workers"],
            "rounds": options.reset_rounds,
            "per_round": options.reset_per_round,
        },
        # Failure surface
        "worker_failure_detection_passed": failure["worker_failure_detection_passed"],
        "worker_failure_cases": failure["cases"],
        "thread_limit_variables": failure["thread_limit_variables"],
        "worker_thread_limits_observed": failure["worker_thread_limits"],
        # Benchmark
        "screening_results": benchmark["screening_results"],
        "measured_results": benchmark["measured_results"],
        "best_cpu_configuration": (
            None
            if best is None
            else {
                "num_workers": best["num_workers"],
                "num_environments": best["num_environments"],
                "label": best["label"],
                "measured_seconds": best["measured_seconds"],
                "measured_steps": best["measured_steps"],
            }
        ),
        "best_cpu_positions_per_second": (
            0.0 if best is None else best["positions_per_second"]
        ),
        "best_cpu_transitions_per_second": (
            0.0 if best is None else best["transitions_per_second"]
        ),
        "best_cpu_observation_builds_per_second": (
            0.0 if best is None else best["observation_builds_per_second"]
        ),
        "best_cpu_coordinator_wait_fraction": (
            0.0 if best is None else best["coordinator_wait_fraction"]
        ),
        "best_cpu_barrier_wait_fraction": (
            0.0 if best is None else best["barrier_wait_fraction"]
        ),
        "best_cpu_worker_active_fraction": (
            0.0 if best is None else best["worker_active_fraction"]
        ),
        "best_cpu_utilization": 0.0 if best is None else best["cpu_utilization"],
        "memory_peak_bytes": max(
            [max_resident_bytes()]
            + [
                entry["total_memory_bytes"]
                for entry in benchmark["screening_results"] + benchmark["measured_results"]
                if not entry["error"]
            ]
        ),
        "coordinator_peak_rss_bytes": max_resident_bytes(),
        "deadlocks": failure["deadlocks"],
        "errors": [entry["error"] for entry in benchmark["errors"]],
        # Tests
        "test_total": tests["total"],
        "test_passed": tests["passed"],
        "test_failed": tests["failed"],
        "test_skipped": tests["skipped"],
        "test_expected_failures": tests["expected_failures"],
        "test_errors": tests["errors"],
        "test_exit_code": tests["exit_code"],
        "test_seconds": tests["seconds"],
        "test_failure_lines": tests["failure_lines"],
        "elapsed_seconds": elapsed,
        "files_created": FILES_CREATED,
        "files_modified": FILES_MODIFIED,
    }

    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    write_raw_csv(
        options.raw_output,
        benchmark["screening_results"] + benchmark["measured_results"],
    )

    print(f"\nstatus: {metrics['status']}")
    print(f"cross-process steps: {metrics['cross_process_steps']}")
    print(f"reset events: {metrics['reset_events']}")
    if best is not None:
        print(
            f"best CPU configuration: {best['num_workers']} workers x "
            f"{best['num_environments']} environments = "
            f"{best['positions_per_second']:,.0f} positions/second"
        )
    print(f"elapsed: {elapsed:.1f}s")
    print(f"wrote {options.output.relative_to(REPOSITORY_ROOT)}")
    print(f"wrote {options.raw_output.relative_to(REPOSITORY_ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
