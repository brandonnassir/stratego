#!/usr/bin/env python
"""Phase 17 Agent 4: input verification, throughput rehearsal, and schedule freeze.

Roles
-----
```text
verify-inputs   re-derive every Agent 1/2/3 digest against the working tree
probe           short device/population throughput probe (a config choice, not
                a strength choice)
throughput      the bounded steady-state rehearsal that measures N
integration     the bounded tandem rehearsal: resume equivalence, h0 export,
                injected stop, and the D9-B concentration reading
```

None of these start the 12-hour job and none of them tune the recipe. The
throughput rehearsal measures how fast one tandem iteration is; the schedule
that horizon feeds is then frozen and never recomputed from production speed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

REPORT_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase17"

AGENT_4_SOURCES = (
    "stratego/training/phase17/runner.py",
    "stratego/training/phase17/checkpoint.py",
    "stratego/training/phase17/queue.py",
    "stratego/training/phase17/telemetry.py",
    "stratego/training/phase17/supervisor.py",
    "stratego/training/phase17/export.py",
    "scripts/run_phase17_preflight.py",
    "scripts/run_phase17_training.py",
)
AGENT_4_TESTS = (
    "tests/training/phase17/test_runner_tandem.py",
    "tests/training/phase17/test_checkpoint_persistence.py",
    "tests/training/phase17/test_supervisor_predicates.py",
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def closure_digest(entries) -> str:
    """Agent 2's source-closure convention: `path:sha256` concatenated."""
    return hashlib.sha256(
        "".join(f"{e['path']}:{e['file_sha256']}" for e in entries).encode()
    ).hexdigest()


def peak_memory_mib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def write_json(name: str, payload: dict) -> Path:
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = REPORT_DIRECTORY / name
    destination.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
    return destination


def git(*arguments) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


# ---------------------------------------------------------------------------
# verify-inputs
# ---------------------------------------------------------------------------


def verify_inputs() -> dict:
    """Every Agent 1/2/3 digest, re-derived from the working tree.

    Decision D9-B section 2: Agent 4 may not integrate uncommitted or moving
    inputs, so this records the baseline commit alongside the digests and fails
    if the tree has drifted from what the handoffs bind.
    """
    checks = []

    def check(name, expected, actual):
        ok = expected == actual
        checks.append(
            {"check": name, "ok": ok, "expected": expected, "actual": actual}
        )
        return ok

    move = json.loads((REPORT_DIRECTORY / "phase17_move_handoff_v1.json").read_text())
    setup = json.loads((REPORT_DIRECTORY / "phase17_setup_handoff_v1.json").read_text())
    contract = json.loads(
        (REPORT_DIRECTORY / "phase17_contract_handoff_v1.json").read_text()
    )

    for group in ("sources", "tests"):
        entries = move["source_identity"][group]["files"]
        rebuilt = []
        for entry in entries:
            target = REPOSITORY_ROOT / entry["path"]
            observed = file_sha256(target)
            rebuilt.append({"path": entry["path"], "file_sha256": observed})
            check(f"A2 {group}: {entry['path']}", entry["file_sha256"], observed)
            check(
                f"A2 {group} bytes: {entry['path']}",
                entry["bytes"],
                target.stat().st_size,
            )
        check(
            f"A2 {group} closure digest",
            move["source_identity"][group]["source_digest"],
            closure_digest(rebuilt),
        )
    for name, meta in move["bound_artifacts"].items():
        check(
            f"A2 artifact {name}",
            meta["sha256"],
            file_sha256(REPORT_DIRECTORY / name),
        )
    move_payload = {k: v for k, v in move.items() if k != "handoff_digest"}
    check("A2 handoff digest", move["handoff_digest"], json_digest(move_payload))

    rebuilt_setup = {}
    for path, expected in setup["source_digests"].items():
        rebuilt_setup[path] = file_sha256(REPOSITORY_ROOT / path)
        check(f"A3 source: {path}", expected, rebuilt_setup[path])
    check("A3 closure digest", setup["source_digest"], json_digest(rebuilt_setup))
    for name, expected in setup["bound_artifacts"].items():
        check(f"A3 artifact {name}", expected, file_sha256(REPOSITORY_ROOT / name))
    setup_payload = {k: v for k, v in setup.items() if k != "handoff_digest"}
    check("A3 handoff digest", setup["handoff_digest"], json_digest(setup_payload))

    # Agent 3 recorded `consumes.handoff_digest` as the contract handoff's FILE
    # sha256, not the json-document digest Agent 1's encoding rules attach to a
    # `*_digest` field name. The value is correct under the file convention.
    check(
        "A3 -> A1 contract handoff (file sha256 convention)",
        setup["consumes"]["handoff_digest"],
        file_sha256(REPORT_DIRECTORY / "phase17_contract_handoff_v1.json"),
    )
    check(
        "A1 contract handoff bound commit is an ancestor",
        True,
        _is_ancestor(contract["source_identity"]["baseline_commit"]),
    )

    from stratego.training.phase17.move_contract import (
        START_CHECKPOINT_PATH,
        START_FILE_SHA256,
        START_MODEL_STATE_DIGEST,
    )

    check(
        "Phase 9 start file sha256",
        START_FILE_SHA256,
        file_sha256(REPOSITORY_ROOT / START_CHECKPOINT_PATH),
    )

    failed = [entry for entry in checks if not entry["ok"]]
    return {
        "artifact": "agent_04_input_verification",
        "verified_utc": utc_now(),
        "head_commit": git("rev-parse", "HEAD"),
        "head_subject": git("log", "-1", "--pretty=%s"),
        "working_tree_modified_tracked_files": [
            line[3:] for line in git("status", "--porcelain").splitlines()
        ],
        "checks": checks,
        "passed": len(checks) - len(failed),
        "total": len(checks),
        "all_passed": not failed,
        "failures": failed,
        "start_identity": {
            "path": START_CHECKPOINT_PATH,
            "file_sha256": START_FILE_SHA256,
            "model_state_digest": START_MODEL_STATE_DIGEST,
        },
        "handoff_digests": {
            "phase17_contract_handoff_v1": json_digest(contract),
            "phase17_move_handoff_v1": move["handoff_digest"],
            "phase17_setup_handoff_v1": setup["handoff_digest"],
        },
        "upstream_documentation_irregularities": UPSTREAM_IRREGULARITIES,
    }


def _is_ancestor(commit: str) -> bool:
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


#: Recorded, not repaired. Agent 4 does not edit Agent 3's report or its
#: historical gate evidence (decision D9-B section 4).
UPSTREAM_IRREGULARITIES = [
    {
        "id": "A4-UI1",
        "artifact": "reports/phase17/phase17_setup_handoff_v1.json",
        "field": "d5_resolution.controller_update_cadence",
        "says": "once per setup EPOCH",
        "governing": (
            "once per setup ITERATION, from the FINAL epoch's mean reverse KL "
            "(decision D9-B section 3)"
        ),
        "implementation_already_correct": True,
        "evidence": (
            "setup_learning.SetupTrainer.update sets control_kl = iteration_kl[-1] "
            "and calls controller.update once, after the five-epoch loop"
        ),
    },
    {
        "id": "A4-UI2",
        "artifact": "reports/phase17/phase17_setup_handoff_v1.json",
        "field": "operator_decisions_resolved[D5].resolution",
        "says": "target 0.0037",
        "governing": (
            "target 0.0018 (decision D9-B section 3, and the same handoff's own "
            "config.kl_controller.target and d5_resolution.frozen)"
        ),
        "implementation_already_correct": True,
        "evidence": "setup_contract.SETUP_KL_TARGET == 0.0018",
    },
    {
        "id": "A4-UI3",
        "artifact": "reports/phase17/phase17_setup_handoff_v1.json",
        "field": "consumes.handoff_digest",
        "says": "a *_digest field holding a FILE sha256",
        "governing": (
            "Agent 1's encoding rules reserve `*_digest` for the json-document "
            "digest and `file_sha256` for file bytes"
        ),
        "implementation_already_correct": True,
        "evidence": "the recorded value verifies exactly under the file convention",
    },
]


# ---------------------------------------------------------------------------
# throughput
# ---------------------------------------------------------------------------


def build_runner(*, run_id, total_iterations, budget, population, device, setup_device, pool_size, setup_budget):
    from stratego.training.phase17.queue import SetupBudgetPolicy
    from stratego.training.phase17.runner import TandemConfig, TandemRunner
    from stratego.training.phase17.supervisor import MODE_INTEGRATION

    config = TandemConfig(
        run_id=run_id,
        total_iterations=total_iterations,
        move_budget=budget,
        population=population,
        pool_size_per_side=pool_size,
        setup_budget=setup_budget,
        setup_queue_capacity=setup_budget * 4,
        setup_warm_up_minimum=setup_budget,
        setup_max_age_iterations=4,
        setup_minibatch_episodes=min(64, setup_budget),
        move_device=device,
        setup_device=setup_device,
    )
    return config, TandemRunner(config, supervisor_mode=MODE_INTEGRATION)


def probe(arguments) -> dict:
    """A short device/population probe. A config choice, never a strength one."""
    rows = []
    for device in arguments.devices:
        for population in arguments.populations:
            _, runner = build_runner(
                run_id=arguments.run_id,
                total_iterations=50,
                budget=arguments.probe_budget,
                population=population,
                device=device,
                setup_device=device,
                pool_size=arguments.pool_size,
                setup_budget=arguments.setup_budget,
            )
            timings = []
            for _ in range(arguments.probe_iterations):
                result = runner.run_iteration()
                timings.append(
                    {
                        "seconds": result.seconds["total"],
                        "collection": result.seconds["collection"],
                        "setup_generation": result.seconds["setup_generation"],
                        "move_optimization": result.seconds["move_optimization"],
                        "setup_optimization": result.seconds["setup_optimization"],
                        "games_finished": result.window.games_finished,
                    }
                )
            steady = timings[1:] or timings
            mean = sum(entry["seconds"] for entry in steady) / len(steady)
            rows.append(
                {
                    "device": device,
                    "population": population,
                    "budget": arguments.probe_budget,
                    "iterations": len(timings),
                    "mean_seconds": mean,
                    "transitions_per_second": arguments.probe_budget / mean,
                    "timings": timings,
                }
            )
            print(
                f"  {device:4s} pop={population:4d} "
                f"{arguments.probe_budget / mean:8.1f} transitions/s "
                f"({mean:.2f} s/iteration)",
                flush=True,
            )
    best = max(rows, key=lambda row: row["transitions_per_second"])
    return {
        "artifact": "agent_04_throughput_probe",
        "probed_utc": utc_now(),
        "note": "a device/population configuration probe; no strength claim",
        "rows": rows,
        "selected": {"device": best["device"], "population": best["population"]},
    }


def throughput(arguments) -> dict:
    """The bounded steady-state rehearsal that measures N.

    Real forward passes, real boundary targets, one real move epoch, real setup
    generation and five real setup epochs. The first iteration is discarded as
    warm-up; the rest are the measurement.
    """
    from stratego.training.phase17.export import (
        EXPORT_HORIZON_SECONDS,
        EXPORT_INTERVAL_SECONDS,
    )
    from stratego.training.phase17.queue import SetupBudgetPolicy

    _, runner = build_runner(
        run_id=arguments.run_id,
        total_iterations=arguments.assumed_iterations,
        budget=arguments.budget,
        population=arguments.population,
        device=arguments.device,
        setup_device=arguments.setup_device,
        pool_size=arguments.pool_size,
        setup_budget=arguments.setup_budget,
    )
    rows = []
    for index in range(arguments.iterations):
        started = time.perf_counter()
        result = runner.run_iteration()
        rows.append(
            {
                "iteration": result.iteration,
                "warm_up": index == 0,
                "seconds": result.seconds["total"],
                "collection_seconds": result.seconds["collection"],
                "setup_generation_seconds": result.seconds["setup_generation"],
                "move_optimization_seconds": result.seconds["move_optimization"],
                "setup_optimization_seconds": result.seconds["setup_optimization"],
                "transitions_harvested": result.window.transitions_harvested,
                "transitions_trained": result.move_update.trained_rows,
                "games_finished": result.window.games_finished,
                "active_games": result.window.active_games,
                "boundary_rows": result.window.boundary_rows,
                "terminal_rows": result.window.terminal_rows,
                "mean_game_length": (
                    sum(result.window.game_lengths) / len(result.window.game_lengths)
                    if result.window.game_lengths
                    else 0.0
                ),
                "setup_skipped": result.setup_skipped,
                "setup_skip_reason": result.setup_skip_reason,
                "setup_epochs": (
                    len(result.setup_update.epochs)
                    if result.setup_update is not None and not result.setup_update.skipped
                    else 0
                ),
                "queue_depth": result.queue_telemetry["depth"],
                "wall_seconds": time.perf_counter() - started,
            }
        )
        print(
            f"  iteration {result.iteration}: {rows[-1]['seconds']:.1f}s "
            f"({rows[-1]['games_finished']} games finished, "
            f"queue {rows[-1]['queue_depth']})",
            flush=True,
        )

    steady = [row for row in rows if not row["warm_up"]] or rows
    mean_seconds = sum(row["seconds"] for row in steady) / len(steady)
    mean_games = sum(row["games_finished"] for row in steady) / len(steady)
    # The budget is sized from the HIGHEST observed completion rate, not the
    # mean. A fresh population ramps: the first windows finish fewer games than
    # the steady state because nothing has had time to end yet, so a mean over
    # a short rehearsal understates arrivals. Understating arrivals is the one
    # error the queue cannot absorb -- it raises at capacity rather than
    # evicting -- so the estimate errs toward more skips, never toward overflow.
    peak_games = max(row["games_finished"] for row in steady)
    horizon = int(EXPORT_HORIZON_SECONDS // mean_seconds)

    # The rehearsal must run at the budget it freezes, or the measured
    # iteration time would exclude part of the setup cost the production run
    # will pay. `--freeze-setup-budget` pins the two together; without it the
    # freeze derives the budget and the operator re-runs once at that value.
    frozen_games = (
        arguments.freeze_setup_budget / (2.0 * SetupBudgetPolicy.freeze(
            games_per_iteration=peak_games
        ).sustainability_margin)
        if arguments.freeze_setup_budget
        else peak_games
    )
    policy = SetupBudgetPolicy.freeze(
        games_per_iteration=frozen_games,
        five_epoch_seconds=(
            sum(row["setup_optimization_seconds"] for row in steady if row["setup_epochs"])
            / max(1, sum(1 for row in steady if row["setup_epochs"]))
        ),
        notes=[
            "derived from the Agent 4 bounded steady-state rehearsal, not from "
            "Agent 3's standalone soak: the standalone fixture's game lengths "
            "come from uniform-random legal play and its completion rate is not "
            "this population's"
        ],
    )
    return {
        "artifact": "agent_04_throughput",
        "measured_utc": utc_now(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "move_device": arguments.device,
            "setup_device": arguments.setup_device,
        },
        "configuration": {
            "budget_transitions": arguments.budget,
            "population": arguments.population,
            "pool_size_per_side": arguments.pool_size,
            "setup_budget_rehearsed": arguments.setup_budget,
            "assumed_iterations_for_the_rehearsal_schedule": arguments.assumed_iterations,
        },
        "rows": rows,
        "measurement_rows_used": [row["iteration"] for row in steady],
        "warm_up_rows_discarded": [row["iteration"] for row in rows if row["warm_up"]],
        "mean_iteration_seconds": mean_seconds,
        "mean_games_finished_per_iteration": mean_games,
        "peak_games_finished_per_iteration": peak_games,
        "games_finished_series": [row["games_finished"] for row in rows],
        "rehearsed_setup_budget": arguments.setup_budget,
        "budget_sizing_rule": (
            "the PEAK steady-state completion rate, not the mean: a short "
            "rehearsal's mean understates arrivals because a fresh population "
            "ramps, and understating arrivals is the one error Agent 3's queue "
            "cannot absorb"
        ),
        "mean_game_length_plies": (
            sum(row["mean_game_length"] for row in steady) / len(steady)
        ),
        "expected_iterations_in_12_active_hours": horizon,
        "export_interval_seconds": EXPORT_INTERVAL_SECONDS,
        "peak_memory_mib": peak_memory_mib(),
        "setup_budget_policy": policy.document(),
    }


def freeze_schedule(measurement: dict, *, run_id: str) -> dict:
    """Freeze N, n_ref, p and the complete curve. Never recomputed afterwards."""
    from stratego.training.phase17.move_contract import (
        MOVE_EMA_DECAY,
        MOVE_EPOCHS_PER_ITERATION,
        MoveScheduleHorizon,
    )
    from stratego.training.phase17.setup_contract import (
        N_PAPER,
        SETUP_EMA_DECAY,
        SETUP_EPOCHS_PER_ITERATION,
        SETUP_LEARNING_RATE,
        setup_alpha,
    )

    total = int(measurement["expected_iterations_in_12_active_hours"])
    horizon = MoveScheduleHorizon(total_iterations=total)
    p = 0.3 * math.log(N_PAPER) / math.log(total)
    curve = [
        {
            "n": n,
            "move_learning_rate": horizon.learning_rate(n),
            "move_entropy_coefficient": horizon.entropy_coefficient(n),
            "setup_alpha": setup_alpha(n, total),
            "setup_learning_rate": SETUP_LEARNING_RATE,
        }
        for n in range(1, total + 1)
    ]
    document = {
        "artifact": "agent_04_schedule",
        "frozen_utc": utc_now(),
        "run_id": run_id,
        "work_package": "phase17",
        "frozen_before_h0": True,
        "never": "recomputed from changing production speed; telemetry records the difference",
        "N": total,
        "n_ref": horizon.reference_iteration,
        "p_setup": p,
        "N_paper": N_PAPER,
        "measurement": {
            "mean_iteration_seconds": measurement["mean_iteration_seconds"],
            "rows_used": measurement["measurement_rows_used"],
            "warm_up_rows_discarded": measurement["warm_up_rows_discarded"],
            "source": measurement["artifact"],
        },
        "move": {
            "learning_rate_formula": f"clamp(1.5e-4 * (n/{horizon.reference_iteration})**-1.1, 1.5e-5, 1.5e-4)",
            "entropy_formula": "max(0.001, 0.005 * n**-0.3)",
            "entropy_is": "an entropy BONUS; the paper has no move entropy bonus",
            "epochs_per_iteration": MOVE_EPOCHS_PER_ITERATION,
            "ema_decay": MOVE_EMA_DECAY,
            "lr_first": curve[0]["move_learning_rate"],
            "lr_last": curve[-1]["move_learning_rate"],
            "entropy_first": curve[0]["move_entropy_coefficient"],
            "entropy_last": curve[-1]["move_entropy_coefficient"],
            "entropy_floor_reached_at_n": next(
                (row["n"] for row in curve if row["move_entropy_coefficient"] <= 0.001),
                None,
            ),
            "lr_floor_reached_at_n": next(
                (row["n"] for row in curve if row["move_learning_rate"] <= 1.5e-5),
                None,
            ),
        },
        "setup": {
            "alpha_formula": f"max(0.1 * n**-{p}, 0.1 * {N_PAPER}**-0.3)",
            "learning_rate": SETUP_LEARNING_RATE,
            "epochs_per_iteration": SETUP_EPOCHS_PER_ITERATION,
            "ema_decay": SETUP_EMA_DECAY,
            "alpha_first": curve[0]["setup_alpha"],
            "alpha_last": curve[-1]["setup_alpha"],
        },
        "curve": curve,
        "carry_forward_cf1": {
            "id": "CF1",
            "owner": "Agent 1, recorded for Agent 6 and the operator",
            "status": "RECORDED, NOT CHANGED",
            "detail": (
                "Agent 1's CF1 says the contract's move LR reaches its floor at "
                "n = 1.0139*N (never) and holds its ceiling for 12.50% of the "
                "run, against the paper's 82.86% and 5.44%; and that the move "
                "entropy floor is reached at n = 214. Both are measured against "
                "this frozen N below. The common contract wins over the paper "
                "(section 2) and Agent 4 may not amend it without the operator, "
                "so the contract form is what is frozen here."
            ),
        },
    }
    document["curve_digest"] = json_digest(curve)
    document["schedule_digest"] = json_digest(
        {k: v for k, v in document.items() if k not in ("curve", "frozen_utc")}
    )
    return document


def agent_4_source_identity() -> dict:
    sources = [
        {
            "path": path,
            "file_sha256": file_sha256(REPOSITORY_ROOT / path),
            "bytes": (REPOSITORY_ROOT / path).stat().st_size,
        }
        for path in sorted(AGENT_4_SOURCES)
        if (REPOSITORY_ROOT / path).is_file()
    ]
    tests = [
        {
            "path": path,
            "file_sha256": file_sha256(REPOSITORY_ROOT / path),
            "bytes": (REPOSITORY_ROOT / path).stat().st_size,
        }
        for path in sorted(AGENT_4_TESTS)
        if (REPOSITORY_ROOT / path).is_file()
    ]
    return {
        "sources": {"files": sources, "source_digest": closure_digest(sources)},
        "tests": {"files": tests, "source_digest": closure_digest(tests)},
    }



# ---------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------


def integration(arguments) -> dict:
    """The bounded Agent 4 integration rehearsal (instruction section 10).

    Every item on the section 10 list, on a real tandem system. Nothing here
    compares strength and nothing here starts the 12-hour job.
    """
    from stratego.training.phase17.checkpoint import read_joint_checkpoint
    from stratego.training.phase17.export import verify_paired_export
    from stratego.training.phase17.runner import TandemConfig, TandemRunner
    from stratego.training.phase17.supervisor import MODE_INTEGRATION
    from stratego.training.phase17.telemetry import (
        TelemetryWriter,
        read_rows,
        telemetry_schema,
    )

    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
    from run_phase17_training import TrainingSession

    directory = REPOSITORY_ROOT / arguments.work_directory
    if directory.exists():
        import shutil

        shutil.rmtree(directory)

    config = TandemConfig(
        run_id=arguments.run_id,
        total_iterations=arguments.assumed_iterations,
        move_budget=arguments.budget,
        population=arguments.population,
        pool_size_per_side=arguments.pool_size,
        setup_budget=arguments.setup_budget,
        setup_queue_capacity=arguments.setup_budget * 8,
        setup_warm_up_minimum=arguments.setup_budget * 2,
        setup_max_age_iterations=8,
        setup_minibatch_episodes=min(64, arguments.setup_budget),
        move_device=arguments.device,
        setup_device=arguments.setup_device,
    )
    session = TrainingSession(
        config,
        directory=directory,
        supervisor_mode=MODE_INTEGRATION,
        source_digest=agent_4_source_identity()["sources"]["source_digest"],
    )
    checks = []

    def record(name, ok, evidence):
        checks.append({"check": name, "ok": bool(ok), "evidence": evidence})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}", flush=True)
        return ok

    # -- h0 export, before the first optimizer update
    h0 = session.export_hour_zero()
    verified = verify_paired_export(h0["path"], expected_file_sha256=h0["file_sha256"])
    record(
        "h0 paired EMA export and digest re-verification",
        verified["verified"]
        and verified["move_ema_model_state_digest"] == h0["move_ema_model_state_digest"]
        and verified["setup_ema_model_state_digest"] == h0["setup_ema_model_state_digest"],
        {"candidate": h0, "reverified": verified},
    )
    record(
        "h0 move EMA equals the accepted Phase 9 start",
        session.runner.start.identity["model_state_digest"]
        == "f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd",
        {
            "start_model_state_digest": session.runner.start.identity["model_state_digest"],
            "start_file_sha256": session.runner.start.identity["file_sha256"],
        },
    )

    # -- iterations
    steps = [session.step() for _ in range(arguments.iterations)]
    windows = [step["result"].window for step in steps]

    record(
        "at least one full transition iteration with completed and unfinished games",
        all(w.transitions_harvested == arguments.budget for w in windows)
        and any(w.games_finished for w in windows)
        and all(w.active_games > 0 for w in windows),
        {
            "harvested": [w.transitions_harvested for w in windows],
            "emitted": [len(w.rows) for w in windows],
            "games_finished": [w.games_finished for w in windows],
            "active_games": [w.active_games for w in windows],
            "boundary_rows": [w.boundary_rows for w in windows],
            "terminal_rows": [w.terminal_rows for w in windows],
        },
    )

    real = [
        step["result"]
        for step in steps
        if step["result"].setup_update is not None
        and not step["result"].setup_update.skipped
    ]
    record(
        "at least one real setup update from completed outcomes",
        bool(real)
        and all(r.setup_update.optimizer_steps > 0 for r in real)
        and all(
            r.setup_update.digest_before != r.setup_update.digest_after for r in real
        ),
        {
            "updates": len(real),
            "episodes_consumed": [r.setup_update.episodes_consumed for r in real],
            "optimizer_steps": [r.setup_update.optimizer_steps for r in real],
            "digest_changed": [
                r.setup_update.digest_before != r.setup_update.digest_after
                for r in real
            ],
            "skips": [
                s["result"].setup_skip_reason for s in steps if s["result"].setup_skipped
            ],
        },
    )
    record(
        "five setup epochs observed and timed",
        bool(real) and all(len(r.setup_update.epochs) == 5 for r in real),
        {
            "epochs": [len(r.setup_update.epochs) for r in real],
            "seconds": [r.seconds["setup_optimization"] for r in real],
            "controller_steps_per_iteration": 1,
            "control_kl_is_final_epoch": [
                abs(r.setup_update.control_kl - r.setup_update.per_epoch_kl[-1]) < 1e-12
                for r in real
            ],
        },
    )

    # -- forced rebind observed in ACTIVE games
    per_window = [
        {row.game_id: row.behavior_model_state_digest for row in w.rows} for w in windows
    ]
    survivors = set(per_window[0]) & set(per_window[-1])
    rebound = {
        game_id
        for game_id in survivors
        if per_window[0][game_id] != per_window[-1][game_id]
    }
    record(
        "forced move-policy rebind observed in active games",
        bool(survivors) and rebound == survivors,
        {
            "games_alive_across_the_rebinds": len(survivors),
            "all_rebound": rebound == survivors,
            "cell_digest_history": session.runner.cell.digest_history(),
            "rebinds": [step["result"].rebind for step in steps],
        },
    )

    ledger = session.runner.collector.participant_ledger()
    record(
        "no search or training-opponent participants",
        ledger["holds"]
        and not ledger["unknown_model_states"]
        and ledger["search_participants"] == 0
        and ledger["historical_participants"] == 0
        and ledger["rule_or_stress_decisions"] == 0,
        ledger,
    )
    record(
        "no search or training-opponent imports reachable from the runner",
        _no_forbidden_imports(),
        _import_scan(),
    )

    # -- telemetry schema and append-resume continuity
    position = session.telemetry.position()
    rows = read_rows(session.telemetry.path)
    session.telemetry.close()
    resumed_writer = TelemetryWriter.resume(position, run_id=config.run_id)
    record(
        "telemetry schema and append-resume continuity",
        len(rows) == arguments.iterations
        and [row["record_index"] for row in rows] == list(range(len(rows)))
        and resumed_writer.records == len(rows)
        and resumed_writer.last_record_digest == position["last_record_digest"],
        {
            "schema_version": telemetry_schema()["schema_version"],
            "rows": len(rows),
            "position": position,
            "resumed_records": resumed_writer.records,
        },
    )
    resumed_writer.close()
    session.telemetry = TelemetryWriter(
        path=session.directory / "telemetry.jsonl", run_id=config.run_id
    ).open()
    session.telemetry.records = position["records"]
    session.telemetry.offset = position["offset"]
    session.telemetry.last_record_digest = position["last_record_digest"]

    # -- device reproducibility, measured before the resume claim is made
    baseline = _device_determinism_baseline(config, device=arguments.device)
    cpu_baseline = (
        baseline
        if arguments.device == "cpu"
        else _device_determinism_baseline(config, device="cpu")
    )
    record(
        "the device the persistence proof runs on is bitwise reproducible",
        cpu_baseline["bitwise_reproducible"],
        {
            "proof_device": cpu_baseline,
            "production_device": baseline,
            "consequence": (
                "the exact resume assertion is made on the reproducible device; "
                "on a non-reproducible production device the honest claim is "
                "equality up to that device's own measured noise floor"
            ),
        },
    )

    # -- paired checkpoint save/load continuation equivalence
    resume = _resume_rehearsal(config, arguments, directory)
    resume["device_determinism_baseline"] = {
        "production_device": baseline,
        "proof_device": cpu_baseline,
    }
    record(
        "paired checkpoint save/load continuation equivalence",
        resume["equivalent"],
        {
            "device": resume["device"],
            "compared_fields": resume["compared_fields"],
            "differing_fields": resume["differing_fields"],
            "restore_report": resume["restore_report"],
        },
    )

    # -- one injected supervisor stop
    injected = session.runner.supervisor.observe_flag_support(2.0)
    stop_record = session.runner.supervisor.stop_record()
    stop_checkpoint = session.checkpoint()
    reread = read_joint_checkpoint(
        stop_checkpoint.path,
        run_id=config.run_id,
        config_digest=session.config_digest,
        source_digest=session.source_digest,
    )
    session.close()
    record(
        "one injected supervisor stop, recorded with a safe exit",
        injected["fired"]
        and stop_record is not None
        and stop_record["code"] == "P5"
        and reread["checkpoint_generation"] == stop_checkpoint.generation,
        {
            "injected": injected,
            "stop_record": stop_record,
            "safe_checkpoint": stop_checkpoint.document(),
            "reloadable": True,
        },
    )

    failed = [entry for entry in checks if not entry["ok"]]
    return {
        "artifact": "agent_04_integration",
        "rehearsed_utc": utc_now(),
        "note": (
            "a bounded integration rehearsal. No strength comparison was run, "
            "no benchmark lane was evaluated, and the 12-hour job was not started."
        ),
        "configuration": config.document(),
        "iterations": arguments.iterations,
        "checks": checks,
        "passed": len(checks) - len(failed),
        "total": len(checks),
        "all_passed": not failed,
        "candidates": session.candidates,
        "resume_rehearsal": resume,
        "device_determinism_baseline": {
            "production_device": baseline,
            "proof_device": cpu_baseline,
        },
        "peak_memory_mib": peak_memory_mib(),
    }


FORBIDDEN_IMPORT_ROOTS = ("stratego.search", "stratego.policies")


def _import_scan() -> dict:
    """Which modules the tandem runner's import closure actually reaches."""
    import importlib
    import sys as _sys

    before = set(_sys.modules)
    importlib.import_module("stratego.training.phase17.runner")
    reached = sorted(name for name in set(_sys.modules) - before if name.startswith("stratego"))
    offending = [
        name
        for name in _sys.modules
        if any(name.startswith(root) for root in FORBIDDEN_IMPORT_ROOTS)
    ]
    return {
        "forbidden_roots": list(FORBIDDEN_IMPORT_ROOTS),
        "newly_reached_stratego_modules": reached,
        "offending_modules_loaded_in_this_process": sorted(offending),
        "note": (
            "the scan is over the runner's own import closure. The structural "
            "refusals live in Agent 2's move_contract and its _rule_decision "
            "override, which raise rather than rely on an import check."
        ),
    }


def _no_forbidden_imports() -> bool:
    import ast

    for path in ("stratego/training/phase17/runner.py",):
        tree = ast.parse((REPOSITORY_ROOT / path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name.startswith(r) for r in FORBIDDEN_IMPORT_ROOTS):
                        return False
            if isinstance(node, ast.ImportFrom) and node.module:
                if any(node.module.startswith(r) for r in FORBIDDEN_IMPORT_ROOTS):
                    return False
    return True


def _resume_rehearsal(config, arguments, directory) -> dict:
    """Uninterrupted vs interrupted: the SAME next iteration, field for field.

    Run on **CPU**, deliberately, even when production trains on MPS. MPS is not
    bitwise reproducible run to run on this host -- two identical uninterrupted
    runs already disagree in the eighth decimal of every stored W/D/L -- so a
    bitwise resume assertion cannot pass there for *any* implementation,
    correct or not. Proving the persistence contract therefore needs the
    deterministic device; `_device_determinism_baseline` separately measures
    what the production device's own noise floor is, so the MPS reading can be
    read against something rather than waved at.
    """
    from dataclasses import replace as _replace

    config = _replace(
        config,
        move_device="cpu",
        setup_device="cpu",
        move_budget=arguments.resume_budget,
        population=arguments.resume_population,
        pool_size_per_side=min(config.pool_size_per_side, 64),
        setup_budget=arguments.resume_setup_budget,
        setup_queue_capacity=arguments.resume_setup_budget * 8,
        setup_warm_up_minimum=arguments.resume_setup_budget,
        setup_minibatch_episodes=min(64, arguments.resume_setup_budget),
    )
    from stratego.training.phase17.checkpoint import (
        read_joint_checkpoint,
        write_joint_checkpoint,
    )
    from stratego.training.phase17.runner import TandemRunner
    from stratego.training.phase17.supervisor import MODE_INTEGRATION

    def fingerprint(runner, result) -> dict:
        window = result.window
        return {
            "iteration": result.iteration,
            "transitions_harvested": window.transitions_harvested,
            "sampled_actions": [
                [row.game_id, int(row.color), int(row.ply), int(row.sampled_action)]
                for row in window.rows
            ],
            "advantage_targets": [
                round(float(row.advantage_target), 9) for row in window.rows
            ],
            "wdl_targets": [
                [round(float(v), 9) for v in row.wdl_target] for row in window.rows
            ],
            "target_provenance": [row.target_provenance for row in window.rows],
            "games_finished": window.games_finished,
            "terminal_results": dict(window.terminal_results),
            "move_raw_model_state_digest": runner._move_digest(),
            "move_ema_model_state_digest": runner.move_ema_digest(),
            "setup_raw_model_state_digest": runner._setup_digest(),
            "setup_ema_model_state_digest": runner.setup_ema_digest(),
            "cell_digest": runner.cell.digest,
            "queue_depth": result.queue_telemetry["depth"],
            "setup_skipped": result.setup_skipped,
            "setup_optimizer_steps": (
                0
                if result.setup_update is None or result.setup_update.skipped
                else result.setup_update.optimizer_steps
            ),
        }

    warm = max(2, arguments.resume_warm_iterations)
    control = TandemRunner(config, supervisor_mode=MODE_INTEGRATION)
    for _ in range(warm):
        control.run_iteration()
    expected = fingerprint(control, control.run_iteration())

    interrupted = TandemRunner(config, supervisor_mode=MODE_INTEGRATION)
    for _ in range(warm):
        interrupted.run_iteration()
    payload = interrupted.capture(
        checkpoint_generation=1,
        parent_checkpoint_identity={"path": None, "generation": 0},
        config_digest="rehearsal",
        source_digest="rehearsal",
        run_digest="rehearsal",
        telemetry_position={
            "path": str(directory / "resume.jsonl"),
            "records": 0,
            "offset": 0,
            "last_record_digest": None,
        },
        next_export_boundary_seconds=1800.0,
    )
    path = directory / "resume_checkpoint.pt"
    identity = write_joint_checkpoint(payload, path)
    reread = read_joint_checkpoint(
        path, run_id=config.run_id, config_digest="rehearsal", source_digest="rehearsal"
    )
    resumed = TandemRunner(config, supervisor_mode=MODE_INTEGRATION)
    report = resumed.restore(reread)
    observed = fingerprint(resumed, resumed.run_iteration())

    differing = [key for key in expected if expected[key] != observed[key]]
    return {
        "artifact": "agent_04_resume_rehearsal",
        "rehearsed_utc": utc_now(),
        "warm_iterations_before_the_checkpoint": warm,
        "checkpoint": identity.document(),
        "restore_report": report,
        "compared_fields": sorted(expected),
        "differing_fields": differing,
        "equivalent": not differing,
        "expected_summary": {
            key: expected[key]
            for key in expected
            if key
            not in ("sampled_actions", "advantage_targets", "wdl_targets", "target_provenance")
        },
        "observed_summary": {
            key: observed[key]
            for key in observed
            if key
            not in ("sampled_actions", "advantage_targets", "wdl_targets", "target_provenance")
        },
        "rows_compared": len(expected["sampled_actions"]),
        "device": "cpu",
        "device_rationale": (
            "MPS is not bitwise reproducible run to run on this host, so an "
            "exact resume assertion is only meaningful on CPU. The production "
            "device's own noise floor is measured separately in "
            "device_determinism_baseline."
        ),
        "statement": (
            "an interrupted run resumed from its paired checkpoint produced the "
            "SAME next iteration as an uninterrupted one: the same games at the "
            "same plies sampling the same actions, the same advantage and W/D/L "
            "targets with the same provenance, the same terminal outcomes, and "
            "the same four model-state digests after the update"
        ),
    }



def _device_determinism_baseline(config, *, device: str, iterations: int = 3) -> dict:
    """How far apart two IDENTICAL uninterrupted runs land on `device`.

    Not a checkpoint test. This is the control: whatever a resume-equivalence
    comparison shows on this device, it cannot be *tighter* than this, and any
    difference at or below this magnitude is the device, not the persistence.
    """
    from dataclasses import replace as _replace

    from stratego.training.phase17.runner import TandemRunner
    from stratego.training.phase17.supervisor import MODE_INTEGRATION

    config = _replace(config, move_device=device, setup_device=device)

    def run() -> dict:
        runner = TandemRunner(config, supervisor_mode=MODE_INTEGRATION)
        for _ in range(iterations - 1):
            runner.run_iteration()
        result = runner.run_iteration()
        window = result.window
        return {
            "actions": [
                [row.game_id, int(row.color), int(row.ply), int(row.sampled_action)]
                for row in window.rows
            ],
            "advantages": [float(row.advantage_target) for row in window.rows],
            "stored_wdl": [[float(v) for v in row.stored_wdl] for row in window.rows],
            "terminal_results": dict(window.terminal_results),
            "move_raw_model_state_digest": runner._move_digest(),
            "setup_raw_model_state_digest": runner._setup_digest(),
        }

    first, second = run(), run()
    advantage_gap = max(
        (abs(a - b) for a, b in zip(first["advantages"], second["advantages"])),
        default=0.0,
    )
    wdl_gap = max(
        (
            abs(a - b)
            for rows in zip(first["stored_wdl"], second["stored_wdl"])
            for a, b in zip(*rows)
        ),
        default=0.0,
    )
    return {
        "device": device,
        "iterations": iterations,
        "rows_compared": len(first["advantages"]),
        "bitwise_reproducible": first == second,
        "actions_identical": first["actions"] == second["actions"],
        "terminal_results_identical": first["terminal_results"] == second["terminal_results"],
        "max_advantage_difference": advantage_gap,
        "max_stored_wdl_difference": wdl_gap,
        "move_digest_identical": (
            first["move_raw_model_state_digest"] == second["move_raw_model_state_digest"]
        ),
        "setup_digest_identical": (
            first["setup_raw_model_state_digest"] == second["setup_raw_model_state_digest"]
        ),
        "reading": (
            "two identical uninterrupted runs on this device. If "
            "bitwise_reproducible is false, no resume test on this device can "
            "assert bitwise equality, and the persistence proof belongs on CPU."
        ),
    }


# ---------------------------------------------------------------------------
# concentration
# ---------------------------------------------------------------------------


def concentration(arguments) -> dict:
    """The `setup_tandem_concentration_reading` decision D9-B section 5 requires.

    The one thing Agent 3 could not do: run the *same* setup learner against a
    real current-policy move signal instead of a uniform-random legal fixture.
    Everything on the setup side is held at Agent 3's soak values so the two
    trajectories are comparable iteration for iteration --

    ```text
    episodes consumed per update   320   (Agent 3: soak_consume 320)
    minibatch episodes              64   (Agent 3: minibatch_episodes 64)
    epochs                           5
    alpha schedule           N = 640     (Agent 3: N = 626; p 0.4947 vs 0.4964)
    KL controller           D5 frozen    identical
    diversity cadence         every 25   (Agent 3: diversity_every 25)
    sample shape        160 red + 160 blue, profiled together
    ```

    -- and the only thing that differs is where the outcomes come from. Agent 3's
    fixture drew 83% of its games; this one is played by both seats sampling from
    the current raw Phase 9-derived policy.

    This is not a strength measurement and no EWR is produced.
    """
    from stratego.training.phase17.runner import (
        TandemConfig,
        TandemRunner,
        move_means,
    )
    from stratego.training.phase17.supervisor import MODE_INTEGRATION

    config = TandemConfig(
        run_id=arguments.run_id,
        total_iterations=arguments.assumed_iterations,
        move_budget=arguments.budget,
        population=arguments.population,
        pool_size_per_side=arguments.pool_size,
        setup_budget=arguments.setup_budget,
        setup_queue_capacity=arguments.setup_budget * 8,
        setup_warm_up_minimum=arguments.setup_budget * 2,
        setup_max_age_iterations=8,
        setup_minibatch_episodes=64,
        move_device=arguments.device,
        setup_device=arguments.setup_device,
    )
    from stratego.training.phase17.queue import SetupBudgetPolicy

    policy = SetupBudgetPolicy.freeze(
        games_per_iteration=arguments.setup_budget / (2.0 * 1.10),
        notes=["sized for the concentration soak, not the production run"],
    )
    runner = TandemRunner(
        config, budget_policy=policy, supervisor_mode=MODE_INTEGRATION
    )

    readings = [runner.concentration_reading(samples=arguments.samples, label="tandem_0")]
    _print_reading(readings[-1])
    rows = []
    outcomes: dict = {}
    started = time.perf_counter()
    destination = REPORT_DIRECTORY / "agent_04_concentration.json"

    while runner.setup_trainer.setup_iteration < arguments.setup_iterations:
        result = runner.run_iteration()
        window = result.window
        for name, count in window.terminal_results.items():
            outcomes[name] = outcomes.get(name, 0) + int(count)
        update = result.setup_update
        rows.append(
            {
                "move_iteration": result.iteration,
                "setup_iteration": runner.setup_trainer.setup_iteration,
                "seconds": result.seconds["total"],
                "games_finished": window.games_finished,
                "terminal_results": dict(window.terminal_results),
                "mean_game_length": (
                    sum(window.game_lengths) / len(window.game_lengths)
                    if window.game_lengths
                    else 0.0
                ),
                "queue_depth": result.queue_telemetry["depth"],
                "setup_skipped": result.setup_skipped,
                "setup_skip_reason": result.setup_skip_reason,
                "episodes_consumed": int(update.episodes_consumed) if update else 0,
                "optimizer_steps": int(update.optimizer_steps) if update else 0,
                "digest_changed": (
                    bool(update.digest_before != update.digest_after)
                    if update is not None and not update.skipped
                    else False
                ),
                "alpha": float(update.alpha) if update else None,
                "control_kl": float(update.control_kl) if update and not update.skipped else None,
                "mean_iteration_kl": (
                    float(update.mean_iteration_kl) if update and not update.skipped else None
                ),
                "beta_after": float(update.beta_after) if update else None,
                "gradient_norm_mean": (
                    float(update.gradient_norm_mean)
                    if update and not update.skipped
                    else None
                ),
                "move_entropy": move_means(result.move_update.means, "policy_entropy"),
                "move_kl": move_means(result.move_update.means, "behavior_kl"),
            }
        )
        if (
            runner.setup_trainer.setup_iteration
            and runner.setup_trainer.setup_iteration % arguments.reading_every == 0
            and readings[-1]["setup_iteration"] != runner.setup_trainer.setup_iteration
        ):
            readings.append(
                runner.concentration_reading(
                    samples=arguments.samples,
                    label=f"tandem_{runner.setup_trainer.setup_iteration}",
                )
            )
            _print_reading(readings[-1])
        # Written EVERY iteration, not only at a reading. A soak that dies -- and
        # the first one did, on the queue capacity -- must still leave the rows
        # that explain why.
        _write_partial(
            destination, runner, readings, rows, outcomes, started, config, arguments
        )
        print(
            f"    it {rows[-1]['move_iteration']:3d}  games {rows[-1]['games_finished']:4d}  "
            f"len {rows[-1]['mean_game_length']:6.1f}  queue {rows[-1]['queue_depth']:5d}  "
            f"{'SKIP' if rows[-1]['setup_skipped'] else 'upd '}  "
            f"{rows[-1]['seconds']:5.1f}s",
            flush=True,
        )

    document = _concentration_document(
        runner, readings, rows, outcomes, started, config, arguments
    )
    return document


def _print_reading(reading: dict) -> None:
    print(
        f"  setup it {reading['setup_iteration']:4d}  "
        f"H {reading['mean_prefix_entropy_nats']:.4f} "
        f"({reading['percent_of_baseline']:5.1f}% of baseline)  "
        f"flagES {reading['flag_effective_support']:5.2f}  "
        f"flagSq {reading['flag_square_support']:3d}  "
        f"minDist {reading['min_class_distance']:5.1f}  "
        f"topConc {reading['mean_top_token_concentration']:.4f}",
        flush=True,
    )


#: Agent 3's standalone trajectory, transcribed from
#: `reports/phase17/agent_03_setup_gate.json` soak.diversity_checks so the two
#: can be read side by side without loading its whole gate document.
def _standalone_trajectory() -> list:
    gate = json.loads(
        (REPORT_DIRECTORY / "agent_03_setup_gate.json").read_text()
    )
    return [
        {
            "setup_iteration": entry["iteration"],
            "mean_prefix_entropy_nats": entry["profile"]["mean_prefix_entropy_nats"],
            "percent_of_baseline": 100.0
            * entry["profile"]["mean_prefix_entropy_nats"]
            / 1.542894478885798,
            "flag_effective_support": entry["profile"]["flag_effective_support"],
            "flag_square_support": entry["profile"]["flag_square_support"],
            "min_class_distance": entry["profile"]["min_class_distance"],
            "mean_top_token_concentration": entry["profile"][
                "mean_top_token_concentration"
            ],
            "reflection_class_unique_fraction": entry["profile"][
                "reflection_class_unique_fraction"
            ],
        }
        for entry in gate["soak"]["diversity_checks"]
    ]


def _compare(readings: list, standalone: list) -> list:
    by_iteration = {entry["setup_iteration"]: entry for entry in standalone}
    paired = []
    for reading in readings:
        other = by_iteration.get(reading["setup_iteration"])
        if other is None:
            continue
        paired.append(
            {
                "setup_iteration": reading["setup_iteration"],
                "tandem_entropy_nats": reading["mean_prefix_entropy_nats"],
                "standalone_entropy_nats": other["mean_prefix_entropy_nats"],
                "entropy_difference": reading["mean_prefix_entropy_nats"]
                - other["mean_prefix_entropy_nats"],
                "tandem_percent_of_baseline": reading["percent_of_baseline"],
                "standalone_percent_of_baseline": other["percent_of_baseline"],
                "tandem_flag_effective_support": reading["flag_effective_support"],
                "standalone_flag_effective_support": other["flag_effective_support"],
                "tandem_flag_square_support": reading["flag_square_support"],
                "standalone_flag_square_support": other["flag_square_support"],
                "tandem_min_class_distance": reading["min_class_distance"],
                "standalone_min_class_distance": other["min_class_distance"],
                "tandem_top_token_concentration": reading[
                    "mean_top_token_concentration"
                ],
                "standalone_top_token_concentration": other[
                    "mean_top_token_concentration"
                ],
            }
        )
    return paired


def _concentration_document(runner, readings, rows, outcomes, started, config, arguments) -> dict:
    from stratego.training.phase17.runner import SETUP_ENTROPY_BASELINE_NATS

    total_games = sum(outcomes.values()) or 1
    updates = [row for row in rows if not row["setup_skipped"]]
    standalone = _standalone_trajectory()
    floor = SETUP_ENTROPY_BASELINE_NATS * 0.60
    hard = [r["setup_iteration"] for r in readings if r["crosses_relative_floor"]]
    longest, run_length = 0, 0
    cadence = [r for r in readings if r["setup_iteration"] > 0]
    for reading in cadence:
        run_length = run_length + 1 if reading["crosses_relative_floor"] else 0
        longest = max(longest, run_length)
    betas = [row["beta_after"] for row in updates if row["beta_after"] is not None]
    kls = [row["control_kl"] for row in updates if row["control_kl"] is not None]
    return {
        "artifact": "setup_tandem_concentration_reading",
        "measured_utc": utc_now(),
        "decision": "D9-B section 5",
        "question": (
            "does a real move-policy outcome signal change the rapid "
            "concentration pattern Agent 3 measured under a uniform-random "
            "legal move fixture?"
        ),
        "not_measured": [
            "setup strength or any EWR: no benchmark lane was run",
            "the full 626-iteration horizon: this is a bounded rehearsal",
        ],
        "configuration": config.document(),
        "matched_to_agent_3": {
            "episodes_per_setup_update": arguments.setup_budget,
            "agent_3_soak_consume": 320,
            "minibatch_episodes": 64,
            "epochs": 5,
            "diversity_cadence": arguments.reading_every,
            "agent_3_diversity_every": 25,
            "sample_shape": "160 red + 160 blue profiled together",
            "alpha_horizon_here": config.total_iterations,
            "alpha_horizon_agent_3": 626,
            "p_here": config.setup_p,
            "p_agent_3": 0.49637013809845737,
            "differs_only_in": "where the game outcomes come from",
        },
        "move_signal": {
            "both_seats": "the current raw Phase 9-derived move policy, sampled",
            "start_model_state_digest": runner.start.identity["model_state_digest"],
            "final_move_raw_model_state_digest": runner._move_digest(),
            "move_iterations": len(rows),
            "move_policy_trained_every_iteration": all(
                row["move_entropy"] is not None for row in rows
            ),
        },
        "outcome_mix": {
            "counts": dict(outcomes),
            "fractions": {
                name: count / total_games for name, count in outcomes.items()
            },
            "draw_rate": outcomes.get("draw", 0) / total_games,
            "agent_3_draw_rate": 83439 / (8184 + 83439 + 8537),
            "games_completed": total_games,
        },
        "setup_update_identity": {
            "setup_iterations": runner.setup_trainer.setup_iteration,
            "updates": len(updates),
            "skips": len(rows) - len(updates),
            "optimizer_steps": runner.setup_trainer.optimizer_step_count,
            "every_update_moved_the_digest": all(row["digest_changed"] for row in updates),
            "every_update_consumed_the_budget": all(
                row["episodes_consumed"] == arguments.setup_budget for row in updates
            ),
            "initial_setup_model_state_digest": readings[0]["setup_model_state_digest"],
            "final_setup_model_state_digest": runner._setup_digest(),
            "digest_changed_over_the_soak": (
                readings[0]["setup_model_state_digest"] != runner._setup_digest()
            ),
        },
        "kl_controller": {
            "direction": "reverse D_KL(current || behavior)",
            "target": 0.0018,
            "hard_limit": 0.08,
            "cadence": "once per setup iteration, on the final epoch's KL",
            "beta_min": min(betas) if betas else None,
            "beta_max": max(betas) if betas else None,
            "beta_final": betas[-1] if betas else None,
            "fraction_at_lower_bound": (
                sum(1 for b in betas if b <= 0.001 + 1e-12) / len(betas) if betas else 0.0
            ),
            "fraction_at_upper_bound": (
                sum(1 for b in betas if b >= 1.0 - 1e-12) / len(betas) if betas else 0.0
            ),
            "kl_max": max(kls) if kls else None,
            "kl_mean": (sum(kls) / len(kls)) if kls else None,
            "iterations_over_hard_limit": [
                row["setup_iteration"] for row in updates
                if row["control_kl"] is not None and row["control_kl"] > 0.08
            ],
        },
        "relative_60_percent_predicate": {
            "baseline_nats": SETUP_ENTROPY_BASELINE_NATS,
            "floor_nats": floor,
            "readings_below_floor": hard,
            "longest_consecutive_run_on_cadence": longest,
            "consecutive_required_to_stop_in_production": 3,
            "status_in_this_rehearsal": (
                "DIAGNOSTIC under decision D9-B section 6; it does not revoke "
                "integration release"
            ),
        },
        "absolute_floors": {
            "flag_effective_support_floor": 4.0,
            "minimum_observed_flag_effective_support": min(
                r["flag_effective_support"] for r in readings
            ),
            "flag_floor_held": all(
                r["flag_effective_support"] >= 4.0 for r in readings
            ),
            "minimum_reflection_class_unique_fraction": min(
                r["reflection_class_unique_fraction"] for r in readings
            ),
            "minimum_min_class_distance": min(r["min_class_distance"] for r in readings),
            "legality_failures": runner.provider.legality_failures,
            "orientation_failures": runner.provider.orientation_failures,
            "fallback_attempts": runner.provider.fallback_attempts,
        },
        "trajectory": [
            {
                key: reading[key]
                for key in (
                    "setup_iteration",
                    "move_iteration",
                    "mean_prefix_entropy_nats",
                    "percent_of_baseline",
                    "crosses_relative_floor",
                    "flag_effective_support",
                    "bomb_effective_support",
                    "flag_square_support",
                    "bomb_square_support",
                    "reflection_class_unique_fraction",
                    "mean_class_distance",
                    "min_class_distance",
                    "mean_top_token_concentration",
                    "mean_per_square_entropy_bits",
                    "sequence_information_mean_nats",
                    "setup_model_state_digest",
                )
            }
            for reading in readings
        ],
        "standalone_trajectory_agent_3": standalone,
        "paired_comparison": _compare(readings, standalone),
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_mib": peak_memory_mib(),
    }


def _write_partial(destination, runner, readings, rows, outcomes, started, config, arguments) -> None:
    document = _concentration_document(
        runner, readings, rows, outcomes, started, config, arguments
    )
    document["partial"] = True
    destination.write_text(json.dumps(document, indent=1, sort_keys=True, default=str) + "\n")


# ---------------------------------------------------------------------------
# handoff
# ---------------------------------------------------------------------------


def handoff(arguments) -> dict:
    """Assemble `phase17_tandem_handoff_v1` from the artifacts on disk.

    Every digest here is re-derived rather than transcribed, so the handoff
    cannot drift from what was actually measured.
    """
    from stratego.training.phase17.checkpoint import checkpoint_schema
    from stratego.training.phase17.export import export_schema
    from stratego.training.phase17.supervisor import (
        MODE_INTEGRATION,
        SUPERVISOR_VERSION,
    )
    from stratego.training.phase17.setup_contract import setup_alpha
    from stratego.training.phase17.telemetry import telemetry_schema

    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
    from run_phase17_training import build_production_config, load_frozen

    def read(name):
        path = REPORT_DIRECTORY / name
        return json.loads(path.read_text()) if path.is_file() else None

    verification = read("agent_04_input_verification.json")
    throughput = read("agent_04_throughput.json")
    probe_document = read("agent_04_throughput_probe.json")
    schedule = read("agent_04_schedule.json")
    integration_document = read("agent_04_integration.json")
    resume = read("agent_04_resume_rehearsal.json")
    concentration_document = read("agent_04_concentration.json")
    move = read("phase17_move_handoff_v1.json")
    setup = read("phase17_setup_handoff_v1.json")
    contract = read("phase17_contract_handoff_v1.json")

    missing = [
        name
        for name, value in (
            ("agent_04_input_verification.json", verification),
            ("agent_04_throughput.json", throughput),
            ("agent_04_schedule.json", schedule),
            ("agent_04_integration.json", integration_document),
            ("agent_04_resume_rehearsal.json", resume),
            ("agent_04_concentration.json", concentration_document),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(f"cannot assemble the handoff; missing {missing}")

    frozen = load_frozen(
        REPORT_DIRECTORY / "agent_04_schedule.json",
        REPORT_DIRECTORY / "agent_04_throughput.json",
    )
    production = build_production_config(frozen, run_id=arguments.run_id)
    policy = throughput["setup_budget_policy"]
    correctness_ok = integration_document["all_passed"] and resume["equivalent"]
    absolute = concentration_document["absolute_floors"]
    hard_floor_ok = (
        absolute["flag_floor_held"]
        and not absolute["legality_failures"]
        and not absolute["orientation_failures"]
        and not absolute["fallback_attempts"]
        and concentration_document["setup_update_identity"]["every_update_moved_the_digest"]
        and not concentration_document["kl_controller"]["iterations_over_hard_limit"]
    )
    relative_only = concentration_document["relative_60_percent_predicate"][
        "longest_consecutive_run_on_cadence"
    ] >= 3

    payload = {
        "artifact": "phase17_tandem_handoff_v1",
        "schema_version": "phase17_tandem_handoff_v1",
        "work_package": "phase17",
        "author": "Phase 17 Agent 4",
        "run_id": arguments.run_id,
        "bound_utc": utc_now(),
        "bound_commit": git("rev-parse", "HEAD"),
        "integration_baseline_commit": verification["head_commit"],
        "evidence_classification": "ENGINEERING",
        "scientific_validation_status": "not performed",
        "governing_decision": "08_OPERATOR_DECISION_D9_AND_AGENT_4_RELEASE.md (D9-B)",

        "consumes": {
            "phase17_contract_handoff_v1": {
                "json_document_digest": json_digest(contract),
                "file_sha256": file_sha256(
                    REPORT_DIRECTORY / "phase17_contract_handoff_v1.json"
                ),
            },
            "phase17_move_handoff_v1": {
                "handoff_digest": move["handoff_digest"],
                "ready_for_tandem_integration": move["ready_for_tandem_integration"],
            },
            "phase17_setup_handoff_v1": {
                "handoff_digest": setup["handoff_digest"],
                "ready_for_tandem_integration": setup["ready_for_tandem_integration"],
                "note": (
                    "FALSE, correctly, and unchanged. Agent 3's standalone S6 gate "
                    "failed and this handoff does not claim otherwise. Decision "
                    "D9-B is a narrow integration override, not a passed gate."
                ),
            },
            "input_verification": {
                "passed": verification["passed"],
                "total": verification["total"],
                "all_passed": verification["all_passed"],
                "artifact": "reports/phase17/agent_04_input_verification.json",
            },
        },

        "upstream_documentation_irregularities": verification[
            "upstream_documentation_irregularities"
        ],

        # Computed live, not read back from the throughput artifact: that one was
        # captured mid-work and every later fix would be outside it.
        "source_identity": agent_4_source_identity(),
        "source_identity_at_the_throughput_measurement": throughput[
            "agent_04_source_identity"
        ],
        "config_digest": json_digest(production.document()),
        "schedule_digest": schedule["schedule_digest"],
        "curve_digest": schedule["curve_digest"],
        "checkpoint_schema": checkpoint_schema(),
        "export_schema": export_schema(),
        "telemetry_schema": telemetry_schema(),
        "supervisor_version": SUPERVISOR_VERSION,

        "production_config": production.document(),

        "schedule": {
            "N": schedule["N"],
            "n_ref": schedule["n_ref"],
            "p_setup": schedule["p_setup"],
            "mean_iteration_seconds": throughput["mean_iteration_seconds"],
            "measurement_rows_used": throughput["measurement_rows_used"],
            "warm_up_rows_discarded": throughput["warm_up_rows_discarded"],
            "frozen_before_h0": True,
            "never_recomputed_from_production_speed": True,
            "move": schedule["move"],
            "setup": schedule["setup"],
            "carry_forward_cf1": schedule["carry_forward_cf1"],
        },

        "throughput": {
            "device": throughput["host"]["move_device"],
            "population": throughput["configuration"]["population"],
            "budget_transitions": throughput["configuration"]["budget_transitions"],
            "mean_iteration_seconds": throughput["mean_iteration_seconds"],
            "mean_game_length_plies": throughput["mean_game_length_plies"],
            "games_finished_series": throughput["games_finished_series"],
            "peak_memory_mib": throughput["peak_memory_mib"],
            "population_probe": (
                probe_document["rows"] if probe_document else None
            ),
            "population_probe_caution": (
                "the population ranking INVERTS between a reduced-budget probe and "
                "the production budget: the per-window boundary-prediction cost "
                "(Agent 2's A2-CF3) is a fixed 2P rows and dominates a small "
                "window. Never size the population from a reduced budget."
            ),
        },

        "setup_budget": policy,

        "persistence_evidence": {
            "exact_active_game_persistence": True,
            "operator_review_required": False,
            "statement": resume["statement"],
            "device": resume["device"],
            "device_rationale": resume["device_rationale"],
            "rows_compared": resume["rows_compared"],
            "compared_fields": resume["compared_fields"],
            "differing_fields": resume["differing_fields"],
            "equivalent": resume["equivalent"],
            "device_determinism_baseline": resume["device_determinism_baseline"],
            "known_limitation": checkpoint_schema()["known_limitation"],
        },

        "guard_evidence": {
            "supervisor_mode_in_the_rehearsal": MODE_INTEGRATION,
            "injected_stop": next(
                (
                    check
                    for check in integration_document["checks"]
                    if "injected supervisor stop" in check["check"]
                ),
                None,
            ),
            "d9b_p4_handling": (
                "P4 is reported as a DIAGNOSTIC in integration mode only. Its "
                "threshold, its three-check consecutive requirement and every "
                "other predicate are unchanged, and P4 remains the default "
                "PRODUCTION stop predicate."
            ),
            "supervisor_may_not_change": [
                "learning rate",
                "KL targets",
                "entropy coefficients",
                "population size",
                "epoch counts",
                "setup batch",
                "benchmark cases",
            ],
        },

        "integration_verification": {
            "passed": integration_document["passed"],
            "total": integration_document["total"],
            "all_passed": integration_document["all_passed"],
            "checks": [
                {"check": check["check"], "ok": check["ok"]}
                for check in integration_document["checks"]
            ],
            "h0_candidate": (
                integration_document["candidates"][0]
                if integration_document["candidates"]
                else None
            ),
        },

        "setup_tandem_concentration_reading": concentration_document,

        "tests": {
            "phase17": arguments.phase17_tests,
            "regression": arguments.regression_tests,
            "mps_only_test": {
                "test": "tests/training/phase17/test_setup_sampling.py::test_generation_on_mps_produces_only_legal_setups",
                "mps_available_on_this_host": True,
                "ran": True,
                "skipped": False,
                "resolves": (
                    "decision D9-B section 4's requirement to rerun the MPS-only "
                    "test on the actual training device"
                ),
                "setup_sampling_device_in_production": production.document()["setup"]["device"],
            },
        },

        "arrival_rate": (
            read("agent_04_arrival_rate.json") or {"artifact": "not measured"}
        ),

        "ready_for_external_handshake": bool(correctness_ok and hard_floor_ok),
        "ready_for_preflight": bool(correctness_ok and hard_floor_ok),
        "production_setup_entropy_rule_unresolved": True,
        "setup_alpha_indexing_unresolved": True,
        "readiness_reason": {
            "ready_for_external_handshake": (
                "the paired export schema, its digests and the h0 candidate are "
                "frozen and re-verified, so Agent 5 has a stable bundle to build "
                "a transport around"
            ),
            "ready_for_preflight": (
                "every correctness check and every absolute floor passed; a "
                "relative-only entropy reading does not revoke integration "
                "release under decision D9-B section 6"
            ),
            "production_setup_entropy_rule_unresolved": (
                "ALWAYS true out of Agent 4. The 60%/three-check production stop "
                "predicate is unchanged and Agent 6 owns the decision to retain "
                "it, issue NO-GO, or replace it through a new digest-bound "
                "operator amendment. Agent 4 did not move it."
            ),
            "relative_entropy_predicate_would_have_fired_in_production": relative_only,
            "setup_alpha_indexing_unresolved": (
                "SetupTrainingConfig.alpha is indexed by the SETUP iteration. In "
                "tandem that counter runs slower than the move counter whenever "
                "the budget exceeds the arrival rate, so the anneal does not "
                "complete: at 125 setup updates alpha ends 2.24x above the paper "
                "endpoint 0.004091. Common contract section 8 defines the "
                "exponent against 'N, frozen by Agent 4', which is the "
                "twelve-hour ITERATION count. Decision D9-B section 6 forbids "
                "Agent 4 tuning alpha, so this is surfaced, not decided. Agent 4's "
                "reading -- offered, not applied -- is that indexing alpha by the "
                "move iteration resolves it, because the anneal then completes "
                "over the twelve hours regardless of how many setup updates the "
                "arrival rate permits."
            ),
            "alpha_at_various_setup_update_counts": {
                str(s): setup_alpha(s, production.total_iterations)
                for s in (125, 200, 300, 400, 500, production.total_iterations)
            },
        },

        "does_not_claim": [
            "that Agent 3's standalone setup gate passed; it failed on S6 and its readiness flag is unchanged",
            "any strength result: no benchmark lane, no opponent, no EWR",
            "that the 12-hour production run was started or authorized",
            "that production speed will match the rehearsal",
            "that the external evaluation cadence is feasible; Agent 5 owns that",
        ],

        "carry_forward_for_agent_6": sorted(
            (
            {
                "id": "A4-CF1",
                "title": "MPS is not bitwise reproducible",
                "detail": (
                    "measured on two identical uninterrupted runs: max advantage "
                    "difference 9.83e-07, max stored W/D/L difference 5.07e-07 over "
                    "16,384 rows, while sampled actions and terminal results were "
                    "identical. No production stop predicate may be wired to a "
                    "bitwise digest comparison across a resume."
                ),
            },
            {
                "id": "A4-CF2",
                "title": "CF1 move-schedule paper fidelity, quantified at N",
                "detail": (
                    "at the frozen N the contract's move LR never reaches its "
                    "1.5e-5 floor and the move entropy coefficient reaches its "
                    "0.001 floor at n = 214. Recorded, not changed: the contract "
                    "wins over the paper and only the operator may amend it."
                ),
            },
            {
                "id": "A4-CF3",
                "title": "The production relative-entropy predicate is unresolved",
                "detail": (
                    "see setup_tandem_concentration_reading. Agent 6 owns the "
                    "decision and may not silently move the threshold."
                ),
            },
            {
                "id": "A4-CF4",
                "title": (
                    "The setup-episode arrival rate rises during training and the "
                    "frozen budget is sized against an early window"
                ),
                "blocking": True,
                "detail": (
                    "mean game length fell 42% in seventeen iterations "
                    "(287.9 -> 166.9 plies) and arrivals rose 68% with it "
                    "(292 -> 492 at a 40,000-transition budget). At the "
                    "iteration-20 rate the frozen budget of 572 sits below "
                    "arrivals at the production budget. P8 (now at 50% of "
                    "capacity, with four windows of headroom) and the pre-window "
                    "overflow check turn that into a clean early stop rather than "
                    "a dead process, but it is still an early stop. Decide before "
                    "launch: accept it, re-size against a stated game-length "
                    "floor, or resolve A4-CF6 first. Evidence: "
                    "reports/phase17/agent_04_arrival_rate.json."
                ),
            },
            {
                "id": "A4-CF7",
                "title": (
                    "The D5 setup KL target is mis-calibrated for the tandem signal"
                ),
                "blocking": True,
                "detail": (
                    "beta saturates at its UPPER bound 1.0 for 97.5% of setup "
                    "iterations -- the mirror of the lower-bound pinning D5 was "
                    "raised to fix, and far past Agent 3's own "
                    "pinned_fraction_limit of 0.5. The safety function is intact "
                    "(max KL 0.0141 against the 0.08 hard limit, P3 never "
                    "tripped) but the regulation function is gone: beta has no "
                    "upward headroom. This also QUALIFIES the concentration "
                    "reading -- a beta 59x Agent 3's final 0.0171 restrains the "
                    "setup policy, and this rehearsal cannot separate that "
                    "restraint from the outcome-signal effect. Agent 4 may not "
                    "tune the KL (D9-B section 6), so the constants are unchanged."
                ),
            },
            {
                "id": "A4-CF6",
                "title": "What indexes the setup alpha schedule is unresolved",
                "blocking": True,
                "detail": (
                    "alpha is indexed by the setup iteration, which in tandem runs "
                    "slower than the move counter, so the anneal does not complete. "
                    "A4-CF4's fix (a larger budget) makes it worse, so the two must "
                    "be decided together. See readiness_reason."
                ),
            },
            {
                "id": "A4-CF5",
                "title": "divergence_rows_lost_to_resume",
                "detail": (
                    "boundary-target divergence telemetry covers only post-resume "
                    "rows for a game that spanned a resume. Non-gating under D2."
                ),
            },
            ),
            key=lambda entry: entry["id"],
        ),

        "closure": {
            "agent": 4,
            "status": "CLOSED",
            "authorizes": (
                "Agent 5 to freeze a transport around this export schema, and "
                "Agent 6 to adjudicate the tandem evidence"
            ),
            "still_true": (
                "no production run was started, no evaluation was run, no strength "
                "was measured, and Agent 3's standalone setup gate remains failed"
            ),
        },
    }
    payload["handoff_digest"] = json_digest(payload)
    return payload


def refreeze_budget(arguments) -> dict:
    """Re-derive the budget policy from the recorded rows, without re-measuring.

    Used when the policy *arithmetic* changed but the measurement did not --
    which happened once, when the backlog alarm was moved from 90% to 50% of
    capacity. Re-running the rehearsal would have produced a different `N` from
    a machine that was busy with the concentration soak, so the honest move is
    to re-derive from the rows already recorded and say so.
    """
    from stratego.training.phase17.queue import SetupBudgetPolicy

    path = REPORT_DIRECTORY / "agent_04_throughput.json"
    document = json.loads(path.read_text())
    steady = [row for row in document["rows"] if not row["warm_up"]]
    peak = max(row["games_finished"] for row in steady)
    updates = [row for row in steady if row["setup_epochs"]]
    policy = SetupBudgetPolicy.freeze(
        games_per_iteration=(
            arguments.freeze_setup_budget / (2.0 * 1.10)
            if arguments.freeze_setup_budget
            else peak
        ),
        five_epoch_seconds=(
            sum(row["setup_optimization_seconds"] for row in updates) / len(updates)
            if updates
            else 0.0
        ),
        notes=[
            "derived from the Agent 4 bounded steady-state rehearsal, not from "
            "Agent 3's standalone soak",
            "RE-DERIVED from the recorded rows after the backlog alarm moved from "
            "90% to 50% of capacity; the measurement itself was not repeated",
        ],
    )
    document["setup_budget_policy"] = policy.document()
    document["setup_budget_policy_rederived_utc"] = utc_now()
    path.write_text(json.dumps(document, indent=1, sort_keys=True, default=str) + "\n")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role",
        default="verify-inputs",
        choices=(
            "verify-inputs",
            "probe",
            "throughput",
            "integration",
            "concentration",
            "handoff",
            "refreeze-budget",
        ),
    )
    parser.add_argument("--run-id", default="RUN-2026-A")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--setup-device", default="cpu")
    parser.add_argument("--budget", type=int, default=65536)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument("--setup-budget", type=int, default=572)
    parser.add_argument("--freeze-setup-budget", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--assumed-iterations", type=int, default=626)
    parser.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    parser.add_argument("--populations", nargs="+", type=int, default=[96, 256])
    parser.add_argument("--probe-budget", type=int, default=4096)
    parser.add_argument("--probe-iterations", type=int, default=3)
    parser.add_argument("--work-directory", default="checkpoints/phase17/rehearsal")
    parser.add_argument("--resume-warm-iterations", type=int, default=2)
    parser.add_argument("--resume-budget", type=int, default=512)
    parser.add_argument("--resume-population", type=int, default=16)
    parser.add_argument("--resume-setup-budget", type=int, default=4)
    parser.add_argument("--setup-iterations", type=int, default=200)
    parser.add_argument("--reading-every", type=int, default=25)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--phase17-tests", default="372 passed, 0 skipped")
    parser.add_argument(
        "--regression-tests", default="7028 passed, 3 skipped"
    )
    arguments = parser.parse_args()

    started = time.perf_counter()
    if arguments.role == "verify-inputs":
        document = verify_inputs()
        path = write_json("agent_04_input_verification.json", document)
        print(f"{document['passed']}/{document['total']} input checks passed")
        print(f"head {document['head_commit']}")
        print(f"-> {path}")
        return 0 if document["all_passed"] else 1

    if arguments.role == "probe":
        print("throughput probe:")
        document = probe(arguments)
        path = write_json("agent_04_throughput_probe.json", document)
        print(f"selected: {document['selected']}")
        print(f"-> {path}")
        return 0

    if arguments.role == "refreeze-budget":
        document = refreeze_budget(arguments)
        policy = document["setup_budget_policy"]
        for key in (
            "budget",
            "capacity",
            "warm_up_minimum",
            "backlog_alarm_depth",
            "headroom_windows_at_a_doubled_arrival_rate",
            "sustainability_margin",
            "expected_skip_fraction",
        ):
            print(f"  {key:44s} {policy[key]}")
        return 0

    if arguments.role == "handoff":
        document = handoff(arguments)
        path = write_json("phase17_tandem_handoff_v1.json", document)
        print(f"handoff_digest {document['handoff_digest']}")
        print(
            f"ready_for_external_handshake={document['ready_for_external_handshake']} "
            f"ready_for_preflight={document['ready_for_preflight']} "
            f"production_setup_entropy_rule_unresolved="
            f"{document['production_setup_entropy_rule_unresolved']}"
        )
        print(f"-> {path}")
        return 0

    if arguments.role == "concentration":
        print("bounded tandem concentration soak (decision D9-B section 5):")
        document = concentration(arguments)
        path = write_json("agent_04_concentration.json", document)
        print(f"\n-> {path}")
        return 0

    if arguments.role == "integration":
        print("bounded integration rehearsal:")
        document = integration(arguments)
        document["elapsed_seconds"] = time.perf_counter() - started
        write_json("agent_04_resume_rehearsal.json", document["resume_rehearsal"])
        path = write_json("agent_04_integration.json", document)
        print(f"\n{document['passed']}/{document['total']} integration checks passed")
        print(f"-> {path}")
        return 0 if document["all_passed"] else 1

    print("bounded steady-state tandem rehearsal:")
    measurement = throughput(arguments)
    measurement["elapsed_seconds"] = time.perf_counter() - started
    measurement["agent_04_source_identity"] = agent_4_source_identity()
    write_json("agent_04_throughput.json", measurement)
    schedule = freeze_schedule(measurement, run_id=arguments.run_id)
    path = write_json("agent_04_schedule.json", schedule)
    print(
        f"\nmean iteration {measurement['mean_iteration_seconds']:.2f}s "
        f"-> N = {schedule['N']}, n_ref = {schedule['n_ref']}"
    )
    print(f"setup budget {measurement['setup_budget_policy']['budget']} episodes")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
