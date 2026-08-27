#!/usr/bin/env python
"""Phase 16 Agent 3 runner: training loop v2 and the 3x6-hour shootout.

Roles, in execution order:

```text
decompose   Phase 14's collection/training split       (read-only, no lock)
gates       the section-3 correctness gates          (smoke run takes the lock)
throughput  Phase 14's collector vs this one         (lock; the gate-3 reference)
train       one 6-hour arm                           (lock; --arm)
evaluate    score an arm's exported hours            (lock; --arm)
candidate   freeze phase16_recipe_candidate_v1
report      render reports/phase16/agent_03_report.md
```

Every heavy role takes `checkpoints/phase16/COMPUTE_LOCK.json` (overview
section 5) and refuses to start while another agent's live pid holds it;
`--wait-lock N` polls for up to N minutes instead of refusing. A 6-hour arm
checkpoints every ten minutes and resumes by re-running the same command; an
evaluation pack appends one JSON line per finished game and resumes the same
way.

Nothing here starts a run longer than six hours. Section 5's `stop_rule` is
the only authority on what happens after the shootout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

REPORT_ROOT = REPOSITORY_ROOT / "reports/phase16"
AGENT_ROOT = REPORT_ROOT / "agent03"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints/phase16"
LOCK_PATH = CHECKPOINT_ROOT / "COMPUTE_LOCK.json"

GATES_PATH = REPORT_ROOT / "agent_03_gates.json"
THROUGHPUT_PATH = REPORT_ROOT / "agent_03_throughput.json"
RUN_CONFIG_PATH = REPORT_ROOT / "agent_03_run_configs.json"
CURVES_PATH = REPORT_ROOT / "agent_03_hour_curves.json"
CANDIDATE_PATH = CHECKPOINT_ROOT / "phase16_recipe_candidate_v1.json"
REPORT_PATH = REPORT_ROOT / "agent_03_report.md"
SUITE_PATH = REPORT_ROOT / "agent_03_full_suite.json"

AGENT_NUMBER = 3


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message: str) -> None:
    print(f"[{_utc()}] {message}", flush=True)


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    log(f"wrote {path}")
    return path


def _read(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing prerequisite artifact: {path}")
    return json.loads(path.read_text())


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def acquire_lock(task: str, expected_hours: float, *, wait_minutes: float = 0.0) -> bool:
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + float(wait_minutes) * 60
    while True:
        if LOCK_PATH.is_file():
            try:
                held = json.loads(LOCK_PATH.read_text())
            except ValueError:
                held = {}
            pid = held.get("pid")
            if pid is not None and _pid_alive(pid) and int(pid) != os.getpid():
                if time.time() < deadline:
                    log(
                        f"COMPUTE_LOCK held by agent {held.get('agent')} pid {pid} "
                        f"({held.get('task')}); waiting…"
                    )
                    time.sleep(60)
                    continue
                log(
                    f"COMPUTE_LOCK held by agent {held.get('agent')} pid {pid} "
                    f"({held.get('task')}); refusing to co-run heavy compute"
                )
                return False
            log("stale COMPUTE_LOCK (pid gone); replacing it")
        LOCK_PATH.write_text(
            json.dumps(
                {
                    "agent": AGENT_NUMBER,
                    "task": task,
                    "started_utc": _utc(),
                    "expected_hours": float(expected_hours),
                    "pid": os.getpid(),
                },
                indent=2,
            )
            + "\n"
        )
        return True


def release_lock() -> None:
    if not LOCK_PATH.is_file():
        return
    try:
        held = json.loads(LOCK_PATH.read_text())
    except ValueError:
        held = {}
    if int(held.get("pid", -1)) == os.getpid():
        LOCK_PATH.unlink()
        log("released COMPUTE_LOCK")


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def role_gates(arguments) -> int:
    """The four section-3 correctness gates, in order, with evidence.

    A skipped gate carries forward whatever was recorded for it before, rather
    than overwriting a real result with "skipped". Re-running this to fill in
    one gate must not silently erase the other three -- the smoke run in
    particular costs twenty minutes and the compute lock.
    """
    from stratego.training.phase16 import contract as C
    from stratego.training.phase16 import targets as T

    previous = (
        json.loads(GATES_PATH.read_text()).get("gates", {})
        if GATES_PATH.is_file()
        else {}
    )

    def carried(name: str, note: str) -> dict:
        earlier = previous.get(name)
        if earlier and earlier.get("pass") is not None:
            kept = dict(earlier)
            kept["note"] = f"{kept.get('note') or ''} (carried forward: {note})".strip()
            return kept
        return {"pass": None, "note": note}

    results: dict = {
        "artifact": "phase16_agent03_gates_v1",
        "written_utc": _utc(),
        "contract_digest": C.contract_digest(),
        "gates": {},
    }

    # ---- gate 2 first: it is free and it gates the meaning of everything else
    log("gate window_edge_invariant: windowed targets vs whole-game targets")
    import numpy as np

    invariants = []
    rng = np.random.default_rng(20260826)
    for trial in range(24):
        count = int(rng.integers(20, 90))
        predictions = []
        for _ in range(count):
            draw = rng.random(3)
            predictions.append(tuple(float(v) for v in draw / draw.sum()))
        boundaries = sorted(
            rng.choice(range(1, count), size=min(3, count - 1), replace=False).tolist()
        )
        result = ["red_win", "blue_win", "draw"][trial % 3]
        player = trial % 2
        invariants.append(T.window_edge_invariant(predictions, result, player, boundaries))
    results["gates"]["window_edge_invariant"] = {
        "trials": len(invariants),
        "all_hold": all(entry["holds"] for entry in invariants),
        "max_advantage_difference": max(
            entry["max_advantage_difference"] for entry in invariants
        ),
        "max_wdl_difference": max(entry["max_wdl_difference"] for entry in invariants),
        "tolerance": T.INVARIANT_TOLERANCE,
        "minimum_windows": min(entry["windows"] for entry in invariants),
        "pass": all(entry["holds"] for entry in invariants)
        and min(entry["windows"] for entry in invariants) >= 3,
    }

    # ---- gate 1: the fixed-seed smoke run
    if arguments.skip_smoke:
        results["gates"]["smoke_run"] = carried("smoke_run", "skipped by --skip-smoke")
    else:
        if not acquire_lock("agent3 smoke run", 0.5, wait_minutes=arguments.wait_lock):
            return 2
        try:
            results["gates"]["smoke_run"] = _smoke_run(arguments)
        finally:
            release_lock()

    # ---- gate 3: collection throughput against Phase 14's collector
    if THROUGHPUT_PATH.is_file():
        throughput = json.loads(THROUGHPUT_PATH.read_text())
        results["gates"]["collection_throughput"] = throughput["gate"]
    else:
        results["gates"]["collection_throughput"] = {
            "pass": None,
            "note": f"run --role throughput first; {THROUGHPUT_PATH} is absent",
        }

    # ---- gate 4: the full suite
    if arguments.skip_suite:
        results["gates"]["full_pytest"] = carried("full_pytest", "skipped by --skip-suite")
    else:
        results["gates"]["full_pytest"] = _full_suite()

    decided = [entry for entry in results["gates"].values() if entry.get("pass") is not None]
    results["all_pass"] = bool(decided) and all(entry["pass"] for entry in decided)
    results["undecided"] = [
        name for name, entry in results["gates"].items() if entry.get("pass") is None
    ]
    _write(GATES_PATH, results)
    log(f"gates: all_pass={results['all_pass']} undecided={results['undecided']}")
    return 0 if results["all_pass"] else 1


def _smoke_run(arguments) -> dict:
    """A short fixed-seed run: completes, checkpoints, resumes, and is bit-identical.

    The bit-identity claim is the sharp one and is checked the only way that
    means anything: the *same* window of rows is replayed through a fresh
    trainer on CPU, and every parameter must match to the last bit.
    """
    import copy

    import torch

    from stratego.training.phase16 import contract as C
    from stratego.training.phase16.checkpoint import (
        build_payload,
        load_starting_model,
        read as read_checkpoint,
        restore,
        save as save_payload,
    )
    from stratego.training.phase16.collector import WindowCollector
    from stratego.training.phase16.population import HistoricalPool
    from stratego.training.phase16.setups import build_setup_source
    from stratego.training.phase16.snapshots import bind_anchor, participants_for
    from stratego.training.phase16.trainer import WindowTrainer

    minutes = float(arguments.smoke_minutes)
    config = C.ARM_B.replace(
        arm_id="smoke",
        population=int(arguments.smoke_population),
        window_decisions=int(arguments.smoke_window),
        minibatch_size=int(arguments.minibatch),
        device=arguments.device,
        collection_device=arguments.collection_device,
    )
    log(
        f"smoke: population={config.population} window={config.window_decisions} "
        f"device={config.device} budget={minutes} min"
    )
    started = time.time()
    model = load_starting_model(device=config.device, root=REPOSITORY_ROOT)
    trainer = WindowTrainer(config, model, device=config.device)
    source = build_setup_source(config.setups, root=REPOSITORY_ROOT)
    participants = participants_for(
        model,
        identity="CURRENT",
        device=config.collection_device,
        historical=bind_anchor(model, identity="P24", device=config.collection_device),
        inference_batch_shape=config.inference_batch_shape,
    )
    collector = WindowCollector(
        config, participants, setup_source=source, pool=HistoricalPool("P24")
    )

    windows = []
    kept_rows = None
    deadline = started + minutes * 60
    while time.time() < deadline:
        window = collector.collect_window(should_continue=lambda: time.time() < deadline)
        if not window.rows:
            break
        if kept_rows is None:
            # A slice, not the window: the bit-identity claim is about the
            # update path, and four minibatches exercise every part of it. Deep
            # copying a production window would cost another 4 GB and minutes
            # of CPU for nothing the smaller sample does not already show.
            kept = int(arguments.replay_rows)
            kept_rows = [copy.deepcopy(row) for row in window.rows[:kept]]
        update = trainer.train_window(window.rows, iteration=window.iteration)
        windows.append(
            {
                "iteration": window.iteration,
                "collection": window.summary(),
                "optimization": {
                    key: value
                    for key, value in update.summary().items()
                    if not isinstance(value, (dict, list))
                },
            }
        )
        log(
            f"smoke window {window.iteration}: rows={len(window.rows)} "
            f"plies/s={window.plies_per_second:.1f} steps={update.steps} "
            f"kl={update.means.get('mean_behavior_kl', float('nan')):.4f}"
        )
        window.rows.clear()
        participants = participants_for(
            model,
            identity="CURRENT",
            device=config.collection_device,
            historical=collector.participants.historical,
            inference_batch_shape=config.inference_batch_shape,
        )
        collector.rebind(participants)

    if not windows:
        return {"pass": False, "note": "the smoke run completed no window"}

    # ---- checkpoint and resume
    scratch = CHECKPOINT_ROOT / "arms" / "smoke"
    payload = build_payload(
        config=config,
        model=model,
        optimizer=trainer.optimizer,
        ema=trainer.ema,
        trainer_state=trainer.trainer_state(),
        collector_state=collector.state(),
        clock={"elapsed_seconds": time.time() - started, "started_utc": _utc()},
    )
    written = save_payload(payload, scratch / "hot.pt")
    reread = read_checkpoint(scratch / "hot.pt")
    fresh_model = load_starting_model(device="cpu", root=REPOSITORY_ROOT)
    fresh = WindowTrainer(config.replace(device="cpu"), fresh_model, device="cpu")
    resumed = restore(
        reread, config=config, model=fresh_model, optimizer=fresh.optimizer, ema=fresh.ema
    )

    # ---- bit-identity of one update, on CPU, from identical inputs
    left_model = load_starting_model(device="cpu", root=REPOSITORY_ROOT)
    right_model = load_starting_model(device="cpu", root=REPOSITORY_ROOT)
    cpu_config = config.replace(device="cpu")
    left = WindowTrainer(cpu_config, left_model, device="cpu")
    right = WindowTrainer(cpu_config, right_model, device="cpu")
    left.train_window([copy.deepcopy(row) for row in kept_rows], iteration=1)
    right.train_window([copy.deepcopy(row) for row in kept_rows], iteration=1)
    left_state, right_state = left_model.state_dict(), right_model.state_dict()
    identical = set(left_state) == set(right_state) and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )
    differing = [
        name for name in left_state if not torch.equal(left_state[name], right_state[name])
    ]

    return {
        "pass": bool(
            identical
            and resumed["model_state_digest"] == payload["model_state_digest"]
            and len(windows) >= 1
        ),
        "minutes": round((time.time() - started) / 60, 2),
        "windows": len(windows),
        "window_rows": [entry["collection"]["rows"] for entry in windows],
        "checkpoint": written,
        "resume_model_state_digest": resumed["model_state_digest"],
        "resume_matches": resumed["model_state_digest"] == payload["model_state_digest"],
        "cpu_rerun_bit_identical": bool(identical),
        "cpu_rerun_differing_tensors": differing[:5],
        "replayed_rows": len(kept_rows or []),
        "telemetry": windows,
    }


def _full_suite() -> dict:
    log("gate full_pytest: running the whole suite")
    started = time.time()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    payload = {
        "pass": completed.returncode == 0,
        "returncode": completed.returncode,
        "summary": tail,
        "minutes": round((time.time() - started) / 60, 2),
    }
    _write(SUITE_PATH, payload)
    return payload


# ---------------------------------------------------------------------------
# throughput
# ---------------------------------------------------------------------------


def role_throughput(arguments) -> int:
    """Gate 3: Phase 14's collector and this one, on this machine, back to back.

    Phase 14's own run state is never touched: the reference is measured by
    running the accepted collector into a scratch root under `data/phase16`,
    which is additive and untracked, and deleting nothing.
    """
    if not acquire_lock("agent3 throughput reference", 0.6, wait_minutes=arguments.wait_lock):
        return 2
    try:
        payload = _throughput(arguments)
    finally:
        release_lock()
    _write(THROUGHPUT_PATH, payload)
    log(f"throughput gate: pass={payload['gate']['pass']}")
    return 0 if payload["gate"]["pass"] else 1


def _throughput(arguments) -> dict:
    import shutil

    from stratego.training.phase14_checkpoint import Phase14SnapshotResolver
    from stratego.training.phase14_collector import (
        collect_iteration,
        resolve_pool_participants,
    )
    from stratego.training.phase14_pool import ActivePool, HistoricalArchive
    from stratego.training.phase14_setup_source import Phase14SetupSource
    from stratego.training.phase16 import contract as C
    from stratego.training.phase16.checkpoint import load_starting_model
    from stratego.training.phase16.collector import WindowCollector
    from stratego.training.phase16.population import HistoricalPool
    from stratego.training.phase16.setups import build_setup_source
    from stratego.training.phase16.snapshots import bind_anchor, participants_for

    games = int(arguments.throughput_games)
    population = int(arguments.smoke_population)
    device = arguments.collection_device
    scratch = REPOSITORY_ROOT / "data/phase16/agent03_throughput_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)

    # ---- the Phase 16 window collector
    log(f"throughput: phase16 window collector, population={population}, device={device}")
    model = load_starting_model(device=device, root=REPOSITORY_ROOT)
    config = C.ARM_B.replace(
        arm_id="throughput",
        population=population,
        window_decisions=int(arguments.throughput_decisions),
        minibatch_size=int(arguments.minibatch),
        device=device,
        collection_device=device,
    )
    collector = WindowCollector(
        config,
        participants_for(
            model,
            identity="CURRENT",
            device=device,
            historical=bind_anchor(model, identity="P24", device=device),
            inference_batch_shape=config.inference_batch_shape,
        ),
        setup_source=build_setup_source(config.setups, root=REPOSITORY_ROOT),
        pool=HistoricalPool("P24"),
    )
    window = collector.collect_window()
    ours = {
        "collector": "phase16_window_collector_v1",
        "population": population,
        "plies_advanced": window.plies_advanced,
        "seconds": round(window.seconds, 2),
        "plies_per_second": round(window.plies_per_second, 2),
        "learner_decisions": window.learner_decisions,
        "games_finished": window.games_finished,
    }
    log(f"throughput: phase16 = {ours['plies_per_second']} plies/s")

    # ---- the accepted Phase 14 collector, into a scratch root
    #
    # The reference is *measured*, not quoted: the same machine, the same
    # moment, the same games-in-flight. Phase 14's own run state is never
    # touched -- this writes into `data/phase16/agent03_throughput_scratch`
    # and removes it afterwards.
    log(f"throughput: phase14 collector reference, {games} games, device={device}")
    pool = ActivePool.for_archive(HistoricalArchive())
    behavior_path = REPOSITORY_ROOT / pool.checkpoint_for("P9")["path"]
    if not behavior_path.is_file():
        return {
            "artifact": "phase16_agent03_throughput_v1",
            "written_utc": _utc(),
            "device": device,
            "phase16": ours,
            "gate": {
                "pass": None,
                "note": f"the Phase 14 anchor {behavior_path} is absent; no reference measurable",
            },
        }
    resolver = Phase14SnapshotResolver(device=device, inference_batch_shape=64)
    behavior = resolver.bind(
        behavior_path,
        logical_identity="B0001",
        policy_token="phase14_behavior_v1|B0001",
        expected_sha256=pool.checkpoint_for("P9")["sha256"],
    )
    participants = resolve_pool_participants(
        pool, behavior=behavior, device=device, inference_batch_shape=64, resolver=resolver
    )
    summary = collect_iteration(
        scratch,
        1,
        participants,
        setup_source=Phase14SetupSource.build(),
        segment="main",
        pool=pool,
        games_in_flight=population,
        limit=games,
        seal=False,
    )
    theirs = {
        "collector": "phase14_collector_v1",
        "games_in_flight": population,
        "games_collected": summary["games_collected"],
        "plies": summary["total_plies"],
        "seconds": round(summary["seconds"], 2),
        "plies_per_second": round(summary["total_plies"] / summary["seconds"], 2)
        if summary["seconds"]
        else 0.0,
    }
    log(f"throughput: phase14 = {theirs['plies_per_second']} plies/s")
    shutil.rmtree(scratch, ignore_errors=True)

    ratio = (
        ours["plies_per_second"] / theirs["plies_per_second"]
        if theirs["plies_per_second"]
        else 0.0
    )
    return {
        "artifact": "phase16_agent03_throughput_v1",
        "written_utc": _utc(),
        "device": device,
        "phase16": ours,
        "phase14": theirs,
        "gate": {
            "rule": "phase16 plies/s within 2x of Phase 14's on this machine",
            "ratio_phase16_over_phase14": round(ratio, 4),
            "pass": bool(ratio >= 0.5),
            "note": (
                "the Phase 14 reference is measured here, on this machine, into a "
                "scratch root; the open Phase 14 run state is never read or written"
            ),
        },
    }


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def role_train(arguments) -> int:
    from stratego.training.phase16 import contract as C
    from stratego.training.phase16.runner import ArmRunner

    config = C.arm(arguments.arm)
    if arguments.population:
        config = config.replace(population=int(arguments.population))
    if arguments.window:
        config = config.replace(window_decisions=int(arguments.window))
    if arguments.minibatch:
        config = config.replace(minibatch_size=int(arguments.minibatch))
    config = config.replace(
        device=arguments.device, collection_device=arguments.collection_device
    )
    hours = float(arguments.hours)
    if hours > C.ARM_HOURS:
        raise SystemExit(
            f"no run in this instruction exceeds {C.ARM_HOURS} hours; refusing {hours}"
        )
    if not acquire_lock(f"agent3 train {config.arm_id}", hours, wait_minutes=arguments.wait_lock):
        return 2
    try:
        runner = ArmRunner(
            config,
            root=REPOSITORY_ROOT,
            telemetry_root=AGENT_ROOT,
            hours=hours,
            device=arguments.device,
            collection_device=arguments.collection_device,
        )
        built = runner.build(resume=not arguments.no_resume)
        log(f"train {config.arm_id}: {json.dumps(built['resumed'] or {}, sort_keys=True)}")
        configs = json.loads(RUN_CONFIG_PATH.read_text()) if RUN_CONFIG_PATH.is_file() else {}
        configs[config.arm_id] = runner.run_config()
        _write(RUN_CONFIG_PATH, configs)

        def _progress(row: dict) -> None:
            optimization = row["optimization"]
            log(
                f"{config.arm_id} it={row['iteration']} h={row['elapsed_hours']:.3f} "
                f"rows={row['collection']['rows']} "
                f"plies/s={row['collection']['plies_per_second']:.1f} "
                f"steps={optimization['optimizer_steps']} "
                f"lr={optimization['learning_rate']:.2e} "
                f"cH={optimization['entropy_coefficient']:.4f} "
                f"kl={optimization.get('mean_behavior_kl', float('nan')):.4f} "
                f"wall={row['iteration_wall_seconds']:.0f}s"
            )

        summary = runner.run(progress=_progress)
    finally:
        release_lock()
    _write(REPORT_ROOT / f"agent_03_{config.arm_id}_run.json", summary)
    log(
        f"train {config.arm_id}: {summary['windows']} windows, "
        f"{summary['optimizer_steps']} steps, {summary['clock']['elapsed_hours']:.3f} h"
    )
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def role_evaluate(arguments) -> int:
    """Score one arm's exported hours on the quick subset and the adversarial stratum."""
    from stratego.evaluation.phase16 import baseline, benchmark
    from stratego.evaluation.phase16.contract import (
        ARM_ADVERSARIAL_BOTH,
        QUICK_SUBSET_NAME,
    )
    from stratego.evaluation.phase16.runner import (
        Task16,
        normalize_seat_spec,
        resolve_subset,
        run_pack16,
    )
    from stratego.search.phase15.analysis import arm_summary
    from stratego.training.phase16.seat import provider_spec

    arm_id = arguments.arm
    storage = CHECKPOINT_ROOT / "arms" / arm_id
    identities = sorted(storage.glob("hour_*_identity.json"))
    if not identities:
        raise SystemExit(f"no exported hours under {storage}; run --role train first")

    bench_manifest = benchmark.load_benchmark_manifest(root=REPOSITORY_ROOT)
    # Score the *full* pack and read the predeclared quick subset out of it.
    #
    # Section 4 names the 60-board quick subset, and that stays the decision
    # instrument: `adopt_recipe` and `setups_causal` are applied to it and to
    # nothing else. But quick60 is a subset of the full 120 (ordinal 0 of every
    # cell), so scoring the full pack yields the briefed reading *and* a
    # higher-powered one from the same games, and a pack hour costs ~10 seconds
    # rather than the minutes that made a subset worth having. The same argument
    # covers the two adversarial strata the decision does not use: they are what
    # separates "adversarial setups hurt because the opponent plays them" from
    # "...because I have to play them", which is exactly what a reader of
    # `setups_causal` will want to know.
    quick_boards = set(resolve_subset(bench_manifest, QUICK_SUBSET_NAME))
    bench_boards = resolve_subset(bench_manifest, None)
    base_manifest = baseline.load_baseline_manifest(root=REPOSITORY_ROOT)
    strata: dict = {}
    for row in base_manifest["boards"]:
        strata.setdefault(row["setup_source"], []).append(row["board_id"])
    adversarial_boards = [board for boards in strata.values() for board in boards]
    log(
        f"evaluate {arm_id}: {len(identities)} hours x ({len(bench_boards)} benchmark "
        f"[{len(quick_boards)} of them the predeclared quick subset] + "
        f"{len(adversarial_boards)} adversarial across {len(strata)} strata) boards"
    )

    if not acquire_lock(f"agent3 evaluate {arm_id}", 1.5, wait_minutes=arguments.wait_lock):
        return 2
    curves = json.loads(CURVES_PATH.read_text()) if CURVES_PATH.is_file() else {}
    try:
        rows_by_hour = {}
        for path in identities:
            identity = json.loads(path.read_text())
            if identity.get("skipped"):
                continue
            hour = int(identity["hour"])
            weights = Path(identity["weights_path"])
            if not weights.is_absolute():
                weights = REPOSITORY_ROOT / weights
            label = f"{arm_id}_h{hour:02d}"
            spec = normalize_seat_spec(
                provider_spec(
                    str(weights),
                    arm_id=label,
                    expected_sha256=identity.get("weights_sha256"),
                )
            )
            out_path = AGENT_ROOT / f"{label}_games.jsonl"
            tasks = [Task16(spec, "direct", board) for board in bench_boards]
            tasks += [Task16(spec, "direct", board) for board in adversarial_boards]
            started = time.time()
            results = run_pack16(
                tasks,
                root=str(REPOSITORY_ROOT),
                device=arguments.eval_device,
                workers=int(arguments.workers),
                out_path=out_path,
            )
            rows = [entry["row"] for entry in results]
            bench_rows = [row for row in rows if row["board_id"].startswith("phase16_benchmark")]
            quick_rows = [row for row in bench_rows if row["board_id"] in quick_boards]
            by_stratum = {
                name: [row for row in rows if row["board_id"] in set(boards)]
                for name, boards in strata.items()
            }
            decision_rows = by_stratum[ARM_ADVERSARIAL_BOTH]
            entry = {
                "arm": arm_id,
                "hour": hour,
                "weights_sha256": identity.get("weights_sha256"),
                "model_state_digest": identity.get("model_state_digest"),
                "source": identity.get("source"),
                "iteration": identity.get("iteration"),
                "optimizer_step": identity.get("optimizer_step"),
                # the two the predeclared rules read, and nothing else
                "benchmark": {
                    "pack": bench_manifest["artifact"],
                    "manifest_digest": bench_manifest["manifest_digest"],
                    "subset": QUICK_SUBSET_NAME,
                    "games": len(quick_rows),
                    **arm_summary(quick_rows, {}),
                },
                "adversarial": {
                    "pack": base_manifest["artifact"],
                    "stratum": ARM_ADVERSARIAL_BOTH,
                    "games": len(decision_rows),
                    **arm_summary(decision_rows, {}),
                },
                # higher-powered secondary readings from the same games; the
                # decision rules never look at these
                "benchmark_full": {
                    "pack": bench_manifest["artifact"],
                    "manifest_digest": bench_manifest["manifest_digest"],
                    "subset": "full",
                    "games": len(bench_rows),
                    **arm_summary(bench_rows, {}),
                },
                "adversarial_strata": {
                    name: {
                        "pack": base_manifest["artifact"],
                        "stratum": name,
                        "games": len(stratum_rows),
                        **arm_summary(stratum_rows, {}),
                    }
                    for name, stratum_rows in sorted(by_stratum.items())
                    if stratum_rows
                },
                "minutes": round((time.time() - started) / 60, 2),
            }
            rows_by_hour[str(hour)] = entry
            log(
                f"{label}: quick60 {entry['benchmark'].get('ewr')} "
                f"full120 {entry['benchmark_full'].get('ewr')} "
                f"adv_both {entry['adversarial'].get('ewr')} "
                f"({entry['minutes']} min)"
            )
        curves[arm_id] = rows_by_hour
        _write(CURVES_PATH, curves)
    finally:
        release_lock()
    return 0


# ---------------------------------------------------------------------------
# decompose
# ---------------------------------------------------------------------------

#: Phase 14's per-iteration telemetry. Read-only, always.
PHASE14_TELEMETRY = Path(
    "/Volumes/Brandon_Washington/stratego_phase14/logs/phase14_telemetry.jsonl"
)
DECOMPOSITION_PATH = REPORT_ROOT / "agent_03_phase14_decomposition.json"
PHASE14_GAMES_PER_ITERATION = 2048


def role_decompose(arguments) -> int:
    """Split Phase 14's iterations into collection and training, from its own log.

    The report claims the window collector's value is *pinned iteration sizing*
    rather than collection speed. That claim rests on where Phase 14's hours
    actually went, so it is derived here from that run's own telemetry rather
    than asserted -- and it is derived reproducibly, so a reader can recompute
    it instead of trusting a number.

    Strictly read-only: the file is opened for reading with the standard library,
    no `stratego.training.phase14_*` module is imported (each drags in torch and
    some of them open checkpoints), and nothing under the Phase 14 run root is
    written. The repository freeze is about modification, and none happens here.
    """
    import statistics

    source = Path(arguments.telemetry or PHASE14_TELEMETRY)
    if not source.is_file():
        log(f"no Phase 14 telemetry at {source}; nothing to decompose")
        return 1
    rows = []
    with open(source) as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    rows.sort(key=lambda row: row["collection"]["iteration"])

    previous_hours = 0.0
    previous_step = 0
    series = []
    for row in rows:
        collection, training, clock = row["collection"], row["training"], row["clock"]
        games_per_second = collection.get("games_per_second")
        mean_length = collection.get("mean_game_length")
        if games_per_second is None or mean_length is None:
            continue
        hours = clock["elapsed_hours"]
        step = training.get("global_optimizer_step") or 0
        iteration_seconds = (hours - previous_hours) * 3600.0
        collection_seconds = PHASE14_GAMES_PER_ITERATION / games_per_second
        training_seconds = iteration_seconds - collection_seconds
        steps = step - previous_step
        series.append(
            {
                "iteration": collection["iteration"],
                "elapsed_hours": round(hours, 4),
                "iteration_seconds": round(iteration_seconds, 1),
                "collection_seconds": round(collection_seconds, 1),
                "training_seconds": round(training_seconds, 1),
                "training_share": round(training_seconds / iteration_seconds, 4)
                if iteration_seconds > 0
                else None,
                "collection_plies_per_second": round(games_per_second * mean_length, 1),
                "mean_game_length": round(mean_length, 1),
                "optimizer_steps": steps,
                "seconds_per_step": round(training_seconds / steps, 4) if steps > 0 else None,
            }
        )
        previous_hours, previous_step = hours, step

    if not series:
        log(f"{source} carried no usable iteration records")
        return 1

    first, last = series[:5], series[-5:]

    def mean(entries, key):
        usable = [entry for entry in entries if entry[key] is not None]
        return sum(entry[key] for entry in usable) / len(usable)

    total_iteration = sum(entry["iteration_seconds"] for entry in series)
    total_collection = sum(entry["collection_seconds"] for entry in series)
    rates = [entry["collection_plies_per_second"] for entry in series]
    document = {
        "artifact": "phase16_agent03_phase14_iteration_decomposition_v1",
        "written_utc": _utc(),
        "source": {
            "path": str(source),
            "access": "read-only; no Phase 14 module imported, nothing written",
            "records": len(series),
            "note": (
                "derived from the run's own per-iteration telemetry: collection "
                f"seconds = {PHASE14_GAMES_PER_ITERATION} games / games_per_second, "
                "training seconds = the remainder of the iteration's elapsed time"
            ),
        },
        "whole_run": {
            "hours": round(total_iteration / 3600.0, 2),
            "collection_hours": round(total_collection / 3600.0, 2),
            "training_hours": round((total_iteration - total_collection) / 3600.0, 2),
            "collection_share": round(total_collection / total_iteration, 4),
            "training_share": round(
                (total_iteration - total_collection) / total_iteration, 4
            ),
        },
        "collection_plies_per_second": {
            "min": min(rates),
            "median": round(statistics.median(rates), 1),
            "max": max(rates),
            "first5": round(mean(first, "collection_plies_per_second"), 1),
            "last5": round(mean(last, "collection_plies_per_second"), 1),
        },
        "growth": {
            key: {
                "first5": round(mean(first, field), digits),
                "last5": round(mean(last, field), digits),
            }
            for key, field, digits in (
                ("mean_game_length", "mean_game_length", 1),
                ("training_share", "training_share", 4),
                ("seconds_per_optimizer_step", "seconds_per_step", 4),
            )
        },
        "series": series,
    }
    document["growth"].update(
        {
            "iteration_minutes": {
                "first5": round(mean(first, "iteration_seconds") / 60, 1),
                "last5": round(mean(last, "iteration_seconds") / 60, 1),
            },
            "collection_minutes": {
                "first5": round(mean(first, "collection_seconds") / 60, 1),
                "last5": round(mean(last, "collection_seconds") / 60, 1),
            },
            "training_minutes": {
                "first5": round(mean(first, "training_seconds") / 60, 1),
                "last5": round(mean(last, "training_seconds") / 60, 1),
            },
            "optimizer_steps_per_iteration": {
                "first5": round(mean(first, "optimizer_steps")),
                "last5": round(mean(last, "optimizer_steps")),
            },
        }
    )
    _write(DECOMPOSITION_PATH, document)
    whole = document["whole_run"]
    log(
        f"phase14: {whole['hours']} h = collection {whole['collection_hours']} h "
        f"({whole['collection_share']:.0%}) + training {whole['training_hours']} h "
        f"({whole['training_share']:.0%}); collection plies/s median "
        f"{document['collection_plies_per_second']['median']}"
    )
    return 0


# ---------------------------------------------------------------------------
# candidate
# ---------------------------------------------------------------------------


def role_candidate(arguments) -> int:
    """Apply section 5's predeclared rules and freeze the winner, or STOP."""
    from stratego.training.phase16 import contract as C
    from stratego.training.phase16.analysis import decide_recipe, horizon_evidence

    curves = _read(CURVES_PATH)
    configs = _read(RUN_CONFIG_PATH)
    decision = decide_recipe(curves, configs)
    document = {
        "artifact": "phase16_recipe_candidate_v1",
        "agent": "agent_03",
        "phase": "phase_16",
        "written_utc": _utc(),
        "contract_digest": C.contract_digest(),
        "decision_rules": dict(C.DECISION_RULES),
        "horizon_evidence": horizon_evidence(REPOSITORY_ROOT),
        **decision,
    }
    _write(CANDIDATE_PATH, document)
    log(f"candidate: {decision['verdict']}")
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def role_report(arguments) -> int:
    from stratego.training.phase16.report_text import render_report

    text = render_report(
        gates=json.loads(GATES_PATH.read_text()) if GATES_PATH.is_file() else {},
        throughput=json.loads(THROUGHPUT_PATH.read_text())
        if THROUGHPUT_PATH.is_file()
        else {},
        curves=json.loads(CURVES_PATH.read_text()) if CURVES_PATH.is_file() else {},
        configs=json.loads(RUN_CONFIG_PATH.read_text()) if RUN_CONFIG_PATH.is_file() else {},
        candidate=json.loads(CANDIDATE_PATH.read_text()) if CANDIDATE_PATH.is_file() else {},
        telemetry_root=AGENT_ROOT,
        report_root=REPORT_ROOT,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text)
    log(f"wrote {REPORT_PATH} ({len(text.splitlines())} lines)")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


ROLES = {
    "decompose": role_decompose,
    "gates": role_gates,
    "throughput": role_throughput,
    "train": role_train,
    "evaluate": role_evaluate,
    "candidate": role_candidate,
    "report": role_report,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=sorted(ROLES))
    parser.add_argument("--arm", default="b_damped")
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--collection-device", dest="collection_device", default="mps")
    parser.add_argument("--eval-device", dest="eval_device", default="cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--population", type=int, default=0)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--smoke-minutes", dest="smoke_minutes", type=float, default=20.0)
    parser.add_argument("--smoke-population", dest="smoke_population", type=int, default=96)
    parser.add_argument("--smoke-window", dest="smoke_window", type=int, default=8192)
    parser.add_argument("--replay-rows", dest="replay_rows", type=int, default=2048)
    parser.add_argument("--telemetry", default=None)
    parser.add_argument("--throughput-games", dest="throughput_games", type=int, default=96)
    parser.add_argument(
        "--throughput-decisions", dest="throughput_decisions", type=int, default=8192
    )
    parser.add_argument("--wait-lock", dest="wait_lock", type=float, default=0.0)
    parser.add_argument("--no-resume", dest="no_resume", action="store_true")
    parser.add_argument("--skip-smoke", dest="skip_smoke", action="store_true")
    parser.add_argument("--skip-suite", dest="skip_suite", action="store_true")
    arguments = parser.parse_args(argv)
    return ROLES[arguments.role](arguments)


if __name__ == "__main__":
    raise SystemExit(main())
