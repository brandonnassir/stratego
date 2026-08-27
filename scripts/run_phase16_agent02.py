#!/usr/bin/env python
"""Phase 16 Agent 2 runner: stochastic search, staged and resumable.

Roles, in execution order:

```text
boundary    read-only process-boundary check         (light)
positions   fresh 120-position Stage 1 pack          (lock)
stage1      the temperature grid on one budget       (lock; --budget)
stage2      the paired match pack on one budget      (lock; --budget)
probe       repeat-encounter probe                   (lock; --preset)
latency     idle-machine caps for the varied modes   (idle machine only)
candidate   freeze phase16_stochastic_candidate_v1
report      render reports/phase16/agent_02_report.md
```

Every heavy role takes `checkpoints/phase16/COMPUTE_LOCK.json` (overview
section 5) and refuses to start while another agent's live pid holds it;
`--wait-lock N` polls for up to N minutes instead of refusing. Stage 2 and
the probe append one JSON line per finished game, so an interrupted pack
resumes by re-running the same command.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

REPORT_ROOT = REPOSITORY_ROOT / "reports/phase16"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints/phase16"
LOCK_PATH = CHECKPOINT_ROOT / "COMPUTE_LOCK.json"

BOUNDARY_PATH = REPORT_ROOT / "agent_02_process_boundary.json"
POSITION_MANIFEST_PATH = REPORT_ROOT / "agent_02_position_manifest.json"
STAGE1_GRID_PATH = REPORT_ROOT / "agent_02_stage1_grid.json"
STAGE2_BOARDS_PATH = REPORT_ROOT / "agent_02_stage2_boards.json"
INTERIM_MANIFEST_PATH = REPORT_ROOT / "agent_02_interim_pack_manifest.json"
STAGE2_JSONL = REPORT_ROOT / "agent_02_stage2_games.jsonl"
STAGE2_PACK_PATH = REPORT_ROOT / "agent_02_stage2_pack.json"
PROBE_JSONL = REPORT_ROOT / "agent_02_probe_games.jsonl"
PROBE_PATH = REPORT_ROOT / "agent_02_probe.json"
LATENCY_PATH = REPORT_ROOT / "agent_02_idle_latency.json"
REPORT_PATH = REPORT_ROOT / "agent_02_report.md"
SUITE_PATH = REPORT_ROOT / "agent_02_full_suite.json"
CLI_PATH = REPORT_ROOT / "agent_02_cli_verification.json"


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}")
    return path


def _read(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing prerequisite artifact: {path}")
    return json.loads(path.read_text())


def _write_csv(path: Path, rows: "list[dict]") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {path} ({len(rows)} rows)")
    return path


def _progress(label: str, quiet: bool):
    if quiet:
        return None
    started = time.time()

    def report(done: int, total: int, payload=None) -> None:
        elapsed = time.time() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (total - done) / rate if rate > 0 else float("nan")
        print(
            f"  [{label}] {done}/{total} ({elapsed/60:.1f} min elapsed, "
            f"~{remaining/60:.1f} min left)",
            flush=True,
        )

    return report


# ---------------------------------------------------------------------------
# The compute lock (overview section 5)
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
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
                    print(
                        f"COMPUTE_LOCK held by agent {held.get('agent')} pid {pid} "
                        f"({held.get('task')}); waiting…",
                        flush=True,
                    )
                    time.sleep(60)
                    continue
                print(
                    f"COMPUTE_LOCK held by agent {held.get('agent')} pid {pid} "
                    f"({held.get('task')}); refusing to co-run heavy compute"
                )
                return False
            print("stale COMPUTE_LOCK (pid gone); replacing it")
        LOCK_PATH.write_text(
            json.dumps(
                {
                    "agent": 2,
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
    if LOCK_PATH.is_file():
        try:
            held = json.loads(LOCK_PATH.read_text())
        except ValueError:
            held = {}
        if held.get("pid") == os.getpid():
            LOCK_PATH.unlink()
            print("released COMPUTE_LOCK")


# ---------------------------------------------------------------------------
# boundary
# ---------------------------------------------------------------------------


def role_boundary(args) -> dict:
    import subprocess

    listing = subprocess.run(
        ["ps", "-axo", "pid,pcpu,pmem,time,command"], capture_output=True, text=True
    ).stdout.splitlines()
    interesting = [
        " ".join(line.split())
        for line in listing
        if ("phase14" in line or "phase15" in line or "phase16" in line)
        and "ps -axo" not in line
        and "run_phase16_agent02.py --role boundary" not in line
    ]
    phase14 = [
        line
        for line in interesting
        if "phase14" in line and "phase14_dashboard" not in line
    ]
    payload = {
        "agent": "agent_02",
        "artifact": "phase16_agent02_process_boundary_v1",
        "phase": "phase_16",
        "checked_utc": _utc(),
        "cpu_count": os.cpu_count(),
        "load_average": dict(zip(("1m", "5m", "15m"), os.getloadavg())),
        "matching_processes": interesting,
        "phase14_learner_or_evaluator_running": bool(phase14),
        "phase14_processes": phase14,
        "method": "read-only `ps` inspection; no signal, no control file, no run-state write",
        "note": (
            "the read-only Phase 14 dashboard may hold port 8714 and is not "
            "competition; Phase 16 Agent 1 runs in parallel and is coordinated "
            "through checkpoints/phase16/COMPUTE_LOCK.json, never signalled"
        ),
        "status": "engineering_deliverable_not_a_strength_claim",
        "scientific_validation_status": "not performed",
        "verdict": "clear_to_run" if not phase14 else "phase14_activity_detected",
    }
    _write(BOUNDARY_PATH, payload)
    return payload


# ---------------------------------------------------------------------------
# positions
# ---------------------------------------------------------------------------


def role_positions(args) -> dict:
    if POSITION_MANIFEST_PATH.is_file() and not args.force:
        print(f"{POSITION_MANIFEST_PATH} exists; --force to regenerate")
        return _read(POSITION_MANIFEST_PATH)
    if not acquire_lock("agent02_positions", 0.3, wait_minutes=args.wait_lock):
        raise SystemExit(2)
    try:
        import torch

        torch.set_num_threads(max(1, args.threads))
        from stratego.search.phase15.loaders import load_all
        from stratego.search.phase15.matchplay import build_owners
        from stratego.search.phase16.diagnostics import (
            build_position_manifest_16,
            generate_positions_16,
        )

        models = load_all(root=REPOSITORY_ROOT, device=args.device, with_anchor=True)
        owners = build_owners(models, device=args.device)
        started = time.time()
        positions = generate_positions_16(
            owners,
            progress=(
                None
                if args.quiet
                else lambda games, found: print(
                    f"  [positions] game {games}: {found} positions", flush=True
                )
            ),
        )
        manifest = build_position_manifest_16(
            positions,
            generated_utc=_utc(),
            generation_minutes=round((time.time() - started) / 60, 2),
        )
        _write(POSITION_MANIFEST_PATH, manifest)
        return manifest
    finally:
        release_lock()


# ---------------------------------------------------------------------------
# stage1
# ---------------------------------------------------------------------------


def _stage1_rows_path(budget: str) -> Path:
    return REPORT_ROOT / f"agent_02_stage1_{budget.lower()}_rows.csv"


def _load_stage1_rows() -> "list[dict]":
    from stratego.search.phase16.contract import STAGE_BUDGETS

    rows: list[dict] = []
    for budget in STAGE_BUDGETS:
        path = _stage1_rows_path(budget)
        if not path.is_file():
            continue
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                for key in (
                    "replay",
                    "ply",
                    "unresolved",
                    "action_id",
                    "legal",
                    "c1_forwards",
                ):
                    if key in row and row[key] != "":
                        row[key] = int(float(row[key]))
                for key in (
                    "matches_control",
                    "matches_oracle",
                    "move_changed_vs_direct",
                    "changed_from_argmax",
                    "unique_worlds",
                    "candidates",
                ):
                    if key in row and row[key] not in ("", None):
                        row[key] = int(float(row[key]))
                for key in ("tau", "tau_r", "search_seconds", "seconds"):
                    if key in row and row[key] not in ("", None):
                        row[key] = float(row[key])
                rows.append(row)
    return rows


def _rebuild_stage1_grid() -> dict:
    from stratego.search.phase16.contract import (
        REGRET_EXCESS_MARGIN,
        STAGE1_REPLAYS,
        STAGE1_VERSION,
    )
    from stratego.search.phase16.diagnostics import (
        apply_stage1_filter,
        summarize_stage1,
    )

    rows = _load_stage1_rows()
    if not rows:
        raise SystemExit("no Stage 1 rows found; run --role stage1 first")
    budgets = sorted({row["preset_id"] for row in rows})
    summary = summarize_stage1(rows)
    verdict = apply_stage1_filter(summary, budgets=budgets, margin=REGRET_EXCESS_MARGIN)
    manifest = _read(POSITION_MANIFEST_PATH)
    payload = {
        "artifact": STAGE1_VERSION,
        "agent": "agent_02",
        "phase": "phase_16",
        "generated_utc": _utc(),
        "position_pack": manifest["artifact"],
        "position_manifest_digest": manifest["manifest_digest"],
        "positions": manifest["position_count"],
        "replays_per_arm_per_position": STAGE1_REPLAYS,
        "world_seed": "DECISION_SEED (the accepted Phase 15 Stage A seed), fixed",
        "budgets_completed": budgets,
        "summary": summary,
        "filter": verdict,
        "latency_note": (
            "Stage 1 search_seconds are measured under worker contention and are "
            "not latency claims; idle caps come from the latency role"
        ),
        "status": "engineering_deliverable_not_a_strength_claim",
        "scientific_validation_status": "not performed",
    }
    _write(STAGE1_GRID_PATH, payload)
    return payload


def role_stage1(args) -> dict:
    from stratego.search.phase16.contract import STAGE_BUDGETS

    budgets = [args.budget] if args.budget else list(STAGE_BUDGETS)
    manifest = _read(POSITION_MANIFEST_PATH)
    del manifest
    for budget in budgets:
        path = _stage1_rows_path(budget)
        if path.is_file() and not args.force:
            print(f"{path} exists; --force to re-run {budget}")
            continue
        expected = 0.3 if budget == "TINY" else 1.5
        if not acquire_lock(f"agent02_stage1_{budget}", expected, wait_minutes=args.wait_lock):
            raise SystemExit(2)
        try:
            from stratego.search.phase16.diagnostics import run_stage1_pack

            rows = run_stage1_pack(
                POSITION_MANIFEST_PATH,
                root=str(REPOSITORY_ROOT),
                device=args.device,
                budgets=(budget,),
                workers=args.workers,
                progress=_progress(f"stage1 {budget}", args.quiet),
            )
            _write_csv(path, rows)
        finally:
            release_lock()
    return _rebuild_stage1_grid()


# ---------------------------------------------------------------------------
# stage2
# ---------------------------------------------------------------------------


def _stage2_boards() -> dict:
    if STAGE2_BOARDS_PATH.is_file():
        return _read(STAGE2_BOARDS_PATH)
    from stratego.search.phase16.matchpack import (
        build_interim_manifest,
        interim_pack_plans,
        resolve_stage2_boards,
    )

    resolved = resolve_stage2_boards(REPOSITORY_ROOT)
    if resolved["source"] == "interim_fallback":
        manifest = build_interim_manifest(interim_pack_plans(), generated_utc=_utc())
        _write(INTERIM_MANIFEST_PATH, manifest)
        resolved["detail"]["manifest_path"] = str(INTERIM_MANIFEST_PATH)
        resolved["detail"]["manifest_digest"] = manifest["manifest_digest"]
    resolved["resolved_utc"] = _utc()
    _write(STAGE2_BOARDS_PATH, resolved)
    return resolved


def _stage2_arms() -> "list[tuple[float, float]]":
    from stratego.search.phase16.contract import (
        CONTROL_ARM,
        FALLBACK_TAU,
        FALLBACK_TAU_R,
        arm_name,
        parse_arm_name,
    )

    grid = _read(STAGE1_GRID_PATH)
    survivors = list(grid["filter"]["survivors"])
    if CONTROL_ARM not in survivors:
        survivors = [CONTROL_ARM] + survivors
    # The brief's named fallback arm must have Stage 2 numbers even if it
    # failed the Stage 1 filter, or the selection rule's fallback branch
    # could never be evaluated.
    fallback = arm_name(FALLBACK_TAU, FALLBACK_TAU_R)
    if fallback not in survivors:
        survivors.append(fallback)
    return [parse_arm_name(arm) for arm in survivors]


def _done_keys(path: Path) -> set:
    done = set()
    if path.is_file():
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)["row"]
                done.add((row["arm_id"], row["preset_id"], row["board_id"]))
    return done


def _append_jsonl(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> "list[dict]":
    entries = []
    if path.is_file():
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


def _rebuild_stage2_pack() -> dict:
    from stratego.search.phase16.contract import STAGE2_VERSION
    from stratego.search.phase16.matchpack import analyse_pack

    boards = _read(STAGE2_BOARDS_PATH)
    entries = _read_jsonl(STAGE2_JSONL)
    if not entries:
        raise SystemExit("no Stage 2 games recorded yet")
    presets = sorted({entry["row"]["preset_id"] for entry in entries})
    analysis = {
        preset: analyse_pack(
            [entry for entry in entries if entry["row"]["preset_id"] == preset],
            reference_preset=preset,
        )
        for preset in presets
    }
    payload = {
        "artifact": STAGE2_VERSION,
        "agent": "agent_02",
        "phase": "phase_16",
        "generated_utc": _utc(),
        "pack_name": boards["pack_name"],
        "boards_source": boards["source"],
        "boards": len(boards["board_ids"]),
        "games_recorded": len(entries),
        "presets_completed": presets,
        "analysis_by_preset": analysis,
        "pack_latency_note": (
            "pack move times are contention-inflated (~1.8x, Phase 15 measured); "
            "latency claims come only from the idle latency role"
        ),
        "status": "engineering_deliverable_not_a_strength_claim",
        "scientific_validation_status": "not performed",
    }
    if "MEDIUM" in analysis:
        expected = len(boards["board_ids"])
        complete = all(
            entry.get("games") == expected for entry in analysis["MEDIUM"].values()
        )
        if complete:
            grid = _read(STAGE1_GRID_PATH)
            from stratego.search.phase16.matchpack import select_configuration

            payload["selection"] = select_configuration(
                analysis["MEDIUM"], grid["summary"]
            )
        else:
            payload["selection_pending"] = (
                "MEDIUM is incomplete: not every arm has played every board; "
                "re-run --role stage2 --budget MEDIUM to finish"
            )
    _write(STAGE2_PACK_PATH, payload)
    return payload


def role_stage2(args) -> dict:
    from stratego.search.phase16.contract import STAGE_BUDGETS
    from stratego.search.phase16.matchpack import StochTask, run_stage2_pack

    boards = _stage2_boards()
    arms = _stage2_arms()
    budgets = [args.budget] if args.budget else list(STAGE_BUDGETS)
    done = _done_keys(STAGE2_JSONL)
    tasks = []
    for preset in budgets:
        for tau, tau_r in arms:
            for board_id in boards["board_ids"]:
                task = StochTask(tau=tau, tau_r=tau_r, preset_name=preset, board_id=board_id)
                if task.key not in done:
                    tasks.append(task)
    print(
        f"stage2: {len(arms)} arms x {len(boards['board_ids'])} boards x "
        f"{budgets}; {len(done)} games already recorded, {len(tasks)} to play"
    )
    if tasks:
        hours = sum(0.05 if task.preset_name == "TINY" else 0.2 for task in tasks) / max(
            args.workers, 1
        )
        if not acquire_lock(
            f"agent02_stage2_{'_'.join(budgets)}", round(hours, 2), wait_minutes=args.wait_lock
        ):
            raise SystemExit(2)
        try:
            progress = _progress("stage2", args.quiet)

            def record(done_count, total, entry):
                _append_jsonl(STAGE2_JSONL, entry)
                if progress is not None and (
                    done_count % 5 == 0 or done_count == total
                ):
                    progress(done_count, total)

            run_stage2_pack(
                tasks,
                root=str(REPOSITORY_ROOT),
                device=args.device,
                workers=args.workers,
                progress=record,
            )
        finally:
            release_lock()
    return _rebuild_stage2_pack()


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def role_probe(args) -> dict:
    from stratego.search.phase16.contract import CONTROL_ARM, parse_arm_name
    from stratego.search.phase16.matchpack import (
        analyse_probe,
        probe_tasks,
        run_stage2_pack,
    )
    from stratego.search.phase16.stochastic import StochasticArm

    pack = _read(STAGE2_PACK_PATH)
    selection = pack.get("selection")
    if not selection:
        raise SystemExit("stage2 MEDIUM selection not available; run stage2 first")
    selected = selection["selected_arm"]
    arm_ids = [selected] if selected == CONTROL_ARM else [selected, CONTROL_ARM]
    arms = [StochasticArm(*parse_arm_name(arm)) for arm in arm_ids]
    preset = args.preset or "MEDIUM"
    tasks = [
        task
        for task in probe_tasks(arms, preset=preset)
        if task.key not in _done_keys(PROBE_JSONL)
    ]
    print(f"probe: arms {arm_ids} at {preset}; {len(tasks)} games to play")
    if tasks:
        hours = len(tasks) * (0.05 if preset == "TINY" else 0.2) / max(args.workers, 1)
        if not acquire_lock("agent02_probe", round(hours, 2), wait_minutes=args.wait_lock):
            raise SystemExit(2)
        try:
            progress = _progress("probe", args.quiet)

            def record(done_count, total, entry):
                _append_jsonl(PROBE_JSONL, entry)
                if progress is not None:
                    progress(done_count, total)

            run_stage2_pack(
                tasks,
                root=str(REPOSITORY_ROOT),
                device=args.device,
                workers=args.workers,
                progress=record,
            )
        finally:
            release_lock()
    entries = _read_jsonl(PROBE_JSONL)
    payload = analyse_probe(entries)
    payload.update(
        {
            "agent": "agent_02",
            "phase": "phase_16",
            "generated_utc": _utc(),
            "preset_id": preset,
            "selected_arm": selected,
            "control_arm": CONTROL_ARM,
            "games_recorded": len(entries),
            "status": "engineering_deliverable_not_a_strength_claim",
            "scientific_validation_status": "not performed",
        }
    )
    _write(PROBE_PATH, payload)
    return payload


# ---------------------------------------------------------------------------
# benchscore — the adversarial delta, once Agent 1's handoff has landed
# ---------------------------------------------------------------------------

BENCHSCORE_JSONL = REPORT_ROOT / "agent_02_benchscore_games.jsonl"
BENCHSCORE_PATH = REPORT_ROOT / "agent_02_adversarial_delta.json"


def role_benchscore(args) -> dict:
    """Brief section 4, last bullet: when Agent 1's adversarial pack exists,
    score the selected arm on its opponent-side adversarial boards and report
    the paired delta vs the deterministic control on the same boards.

    Consumes Agent 1's *delivered* artifacts (handoff-named manifests and its
    scoring runner) — never its work-in-progress: this role refuses to run
    until `phase16_measurement_handoff_v1.json` exists and its digests verify.
    """
    handoff_path = REPORT_ROOT / "phase16_measurement_handoff_v1.json"
    if not handoff_path.is_file():
        print(
            "Agent 1's phase16_measurement_handoff_v1.json has not landed; "
            "re-run `.venv/bin/python scripts/run_phase16_agent02.py --role "
            "benchscore --workers 10` once it exists"
        )
        return {"skipped": "handoff not landed"}
    handoff = json.loads(handoff_path.read_text())
    from stratego.search.phase16.contract import (
        CONTROL_ARM,
        MEASUREMENT_HANDOFF_ARTIFACT,
        parse_arm_name,
    )

    if handoff.get("artifact") != MEASUREMENT_HANDOFF_ARTIFACT:
        raise SystemExit(f"{handoff_path} is not a {MEASUREMENT_HANDOFF_ARTIFACT} document")

    from stratego.evaluation.phase16.baseline import load_baseline_manifest
    from stratego.evaluation.phase16.contract import ARM_ADVERSARIAL_OPPONENT
    from stratego.evaluation.phase16.runner import Task16, run_pack16

    pack = _read(STAGE2_PACK_PATH)
    selection = pack.get("selection")
    if not selection:
        raise SystemExit("stage2 MEDIUM selection not available; run stage2 first")
    selected = selection["selected_arm"]
    arms = [selected] if selected == CONTROL_ARM else [selected, CONTROL_ARM]
    baseline = load_baseline_manifest(root=REPOSITORY_ROOT)
    board_ids = [
        row["board_id"]
        for row in baseline["boards"]
        if row.get("setup_source") == ARM_ADVERSARIAL_OPPONENT
    ]
    if not board_ids:
        raise SystemExit("the baseline manifest carries no adversarial_opponent boards")
    preset = args.preset or "MEDIUM"

    def spec(arm):
        tau, tau_r = parse_arm_name(arm)
        return {
            "factory": "stratego.search.phase16.stochastic:benchmark_seat_factory",
            "kwargs": {"tau": tau, "tau_r": tau_r},
            "arm_id": arm,
        }

    tasks = [
        Task16(
            seat_spec=(
                spec(arm)["factory"],
                json.dumps(spec(arm)["kwargs"], sort_keys=True),
                arm,
            ),
            preset_name=preset,
            board_id=board_id,
        )
        for arm in arms
        for board_id in board_ids
    ]
    hours = len(tasks) * (0.05 if preset == "TINY" else 0.2) / max(args.workers, 1)
    if not acquire_lock("agent02_benchscore", round(hours, 2), wait_minutes=args.wait_lock):
        raise SystemExit(2)
    try:
        progress = _progress("benchscore", args.quiet)
        results = run_pack16(
            tasks,
            root=str(REPOSITORY_ROOT),
            device=args.device,
            workers=args.workers,
            out_path=BENCHSCORE_JSONL,
            progress=(
                None
                if progress is None
                else lambda done_count, total, _entry: (
                    progress(done_count, total) if done_count % 5 == 0 else None
                )
            ),
        )
    finally:
        release_lock()

    from stratego.search.phase15.analysis import arm_summary, paired_delta

    rows_by_arm: dict = {}
    seconds_by_arm: dict = {}
    for entry in results:
        row = entry["row"]
        rows_by_arm.setdefault(row["arm_id"], []).append(row)
        seconds_by_arm.setdefault(row["arm_id"], {})[row["board_id"]] = entry.get(
            "move_seconds"
        ) or []
    control_rows = rows_by_arm.get(CONTROL_ARM, [])
    report = {}
    for arm, rows in rows_by_arm.items():
        summary = arm_summary(rows, seconds_by_arm.get(arm, {}))
        report[arm] = {
            "games": summary["games"],
            "ewr": summary["ewr"],
            "wins": summary["wins"],
            "draws": summary["draws"],
            "losses": summary["losses"],
            "paired_vs_control_same_boards": (
                paired_delta(rows, control_rows) if arm != CONTROL_ARM else None
            ),
            "worst_opponent": summary["min_opponent"],
            "fallbacks": summary["fallbacks"],
        }
    payload = {
        "agent": "agent_02",
        "phase": "phase_16",
        "generated_utc": _utc(),
        "pack_name": baseline["artifact"],
        "baseline_manifest_digest": baseline["manifest_digest"],
        "adversarial_library_digest": baseline.get("adversarial_library_digest"),
        "arm": ARM_ADVERSARIAL_OPPONENT,
        "preset_id": preset,
        "boards": len(board_ids),
        "selected_arm": selected,
        "control_arm": CONTROL_ARM,
        "scores": report,
        "note": (
            "opponent-side adversarial setups from Agent 1's frozen pack; the "
            "delta is paired on identical boards; no significance claim"
        ),
        "status": "engineering_deliverable_not_a_strength_claim",
        "scientific_validation_status": "not performed",
    }
    _write(BENCHSCORE_PATH, payload)
    return payload


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------


def role_latency(args) -> dict:
    import torch

    torch.set_num_threads(max(1, args.threads))
    from stratego.search.phase15.candidate import load_candidate as load_phase15_candidate
    from stratego.search.phase16.contract import parse_arm_name
    from stratego.search.phase16.diagnostics import materialize_positions
    from stratego.search.phase16.matchpack import decide_time_caps, measure_idle_latency
    from stratego.search.phase16.stochastic import StochasticArm
    from stratego.search.phase15.loaders import load_all

    pack = _read(STAGE2_PACK_PATH)
    selection = pack.get("selection")
    if not selection:
        raise SystemExit("stage2 MEDIUM selection not available; run stage2 first")
    arm = StochasticArm(*parse_arm_name(selection["selected_arm"]))
    manifest = _read(POSITION_MANIFEST_PATH)
    subset = dict(manifest)
    subset["positions"] = manifest["positions"][: args.positions or 40]
    replayed = materialize_positions(subset)
    models = load_all(root=REPOSITORY_ROOT, device=args.device, with_anchor=False)
    profiles = measure_idle_latency(models, replayed, arm, device=args.device)
    phase15 = load_phase15_candidate(
        REPOSITORY_ROOT / "checkpoints/phase15/phase15_search_candidate_v1.json"
    )
    pilot = phase15["latency"]["single_process_pilot"]
    phase15_idle = {
        "TINY": {"p95": pilot["selected_preset"]["p95_seconds_per_move"]},
        "MEDIUM": {"p95": pilot["maximum_strength_preset"]["p95_seconds_per_move"]},
    }
    phase15_caps = {
        "TINY": phase15["time_caps_seconds"]["selected_search"],
        "MEDIUM": phase15["time_caps_seconds"]["maximum_strength"],
    }
    caps = decide_time_caps(profiles, phase15_caps, phase15_idle)
    payload = {
        "agent": "agent_02",
        "phase": "phase_16",
        "generated_utc": _utc(),
        "selected_arm": arm.arm_id,
        "measured_on": (
            f"one process, {max(1, args.threads)} torch thread(s), idle machine; "
            "full varied-mode decisions (search + one softmax draw) on "
            f"{len(replayed)} replayed diagnostic positions"
        ),
        "idle_profiles": profiles,
        "phase15_idle_reference": phase15_idle,
        "phase15_caps_reference": phase15_caps,
        "cap_decision": caps,
        "mode_caps_seconds": {
            "varied_fast": caps["caps_seconds"].get("TINY"),
            "varied_strength": caps["caps_seconds"].get("MEDIUM"),
        },
        "status": "engineering_deliverable_not_a_strength_claim",
        "scientific_validation_status": "not performed",
    }
    _write(LATENCY_PATH, payload)
    return payload


# ---------------------------------------------------------------------------
# candidate
# ---------------------------------------------------------------------------


def role_candidate(args) -> dict:
    import platform

    import torch

    from stratego.search.phase16.candidate import (
        DEFAULT_CANDIDATE_PATH_16,
        build_candidate_record_16,
        write_candidate_16,
    )
    from stratego.search.phase16.contract import (
        DOMAIN_ROOTS_16,
        STOCHASTIC_IDENTITY_VERSION,
        STOCHASTIC_MASTER_SEED,
        parse_arm_name,
    )
    from stratego.search.phase16.stochastic import StochasticArm

    grid = _read(STAGE1_GRID_PATH)
    pack = _read(STAGE2_PACK_PATH)
    latency = _read(LATENCY_PATH)
    probe = json.loads(PROBE_PATH.read_text()) if PROBE_PATH.is_file() else None
    selection = pack["selection"]
    arm = StochasticArm(*parse_arm_name(selection["selected_arm"]))

    stage1_headline = {
        "pack_name": grid["position_pack"],
        "position_manifest_digest": grid["position_manifest_digest"],
        "positions": grid["positions"],
        "replays": grid["replays_per_arm_per_position"],
        "filter": grid["filter"],
        "selected_arm_summary": {
            budget: (grid["summary"]["arms"] or {}).get(
                f"{selection['selected_arm']}|{budget}"
            )
            for budget in grid["budgets_completed"]
        },
    }
    stage2_headline = {
        "pack_name": pack["pack_name"],
        "boards_source": pack["boards_source"],
        "boards": pack["boards"],
        "games_recorded": pack["games_recorded"],
        "selection": selection,
        "per_preset_selected_vs_control": {
            preset: {
                key: {
                    "ewr": entry.get("ewr"),
                    "games": entry.get("games"),
                    "paired_vs_reference": entry.get("paired_vs_reference"),
                }
                for key, entry in analysis.items()
                if key.startswith((selection["selected_arm"], "stoch_t000_r000"))
            }
            for preset, analysis in pack["analysis_by_preset"].items()
        },
    }
    probe_headline = None
    if probe:
        probe_headline = {
            "artifact": probe.get("artifact"),
            "preset_id": probe.get("preset_id"),
            "games_recorded": probe.get("games_recorded"),
            "arms": {
                armid: {
                    "ewr": entry.get("ewr"),
                    "games": entry.get("games"),
                    "ewr_slope_per_game_index": entry.get("ewr_slope_per_game_index"),
                    "halves": entry.get("halves"),
                }
                for armid, entry in (probe.get("arms") or {}).items()
            },
            "note": probe.get("note"),
        }

    known_limitations = [
        "machine packs cannot measure adaptation resistance: every Stage 1/2/probe "
        "opponent is a fixed policy that cannot learn the player's habits, so the "
        "unpredictability these numbers buy is only measurable in the operator "
        "exam (Agent 1's protocol, Agent 5's exam)",
        "a compact engineering pack: 60 paired boards per arm per budget in Stage "
        "2; no significance claim is made anywhere",
        "the repeat-encounter probe uses fixed bots and is a weak proxy by "
        "construction; a flat trend there does not demonstrate adaptation "
        "resistance",
        "Stage 1 shares searches across move-temperature arms at the same tau_r "
        "by design (paired draws on identical score vectors); arm rows are "
        "correlated across tau, which pairing exploits and independence-based "
        "readings must not assume",
        "with sampled rollouts, deduplicated duplicate worlds share one sampled "
        "rollout (the accepted dedup is kept byte-identical); this changes Q "
        "variance, not support",
        "no scientific validation phase was performed; this is an engineering "
        "selection",
    ]
    record = build_candidate_record_16(
        arm=arm,
        time_caps={
            "varied_strength": latency["mode_caps_seconds"]["varied_strength"],
            "varied_fast": latency["mode_caps_seconds"]["varied_fast"],
        },
        idle_latency=latency["idle_profiles"],
        stage1=stage1_headline,
        stage2=stage2_headline,
        selection=selection,
        probe=probe_headline,
        seed_streams={
            "identity_version": STOCHASTIC_IDENTITY_VERSION,
            "personalization": "strat-p16s",
            "master_seed": STOCHASTIC_MASTER_SEED,
            "domain_roots": DOMAIN_ROOTS_16,
            "world_seed": (
                "unchanged: the accepted Phase 15 search_seed_for(board_id, ply)"
            ),
            "move_draw": "move_sample_seed(tau, tau_r, identifier, ply, replay)",
            "rollout_draw": "rollout_sample_seed(tau_r, top_p, identifier, ply, replay)",
        },
        known_limitations=known_limitations,
        root=REPOSITORY_ROOT,
        environment={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        deviations=[
            "Stage 1 positions use one observer (P24, the selected system's move "
            "model) instead of Phase 15's two-observer rotation; conservative "
            "match to the single pairing under study",
            "Stage 1 move-temperature arms at the same tau_r share the sixteen "
            "underlying searches (rollout stream keyed by the rollout "
            "configuration only), so tau differences are pure move-sampling "
            "effects on identical score vectors",
            "the Stage 1 survival filter is applied at both budgets (the "
            "conservative reading of the brief's single-margin rule), declared "
            "before any number was seen",
        ]
        + (
            [
                f"the repeat-encounter probe ran at {probe.get('preset_id')} "
                "rather than MEDIUM, a compute-budget deviation recorded here"
            ]
            if probe and probe.get("preset_id") != "MEDIUM"
            else []
        ),
    )
    path = REPOSITORY_ROOT / DEFAULT_CANDIDATE_PATH_16
    write_candidate_16(record, path)
    from stratego.search.phase16.candidate import load_candidate_16

    load_candidate_16(path)
    print(f"candidate frozen and re-read: {path}")
    return record


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def role_report(args) -> dict:
    from stratego.search.phase16 import report_text

    payload = report_text.render(
        boundary=_read(BOUNDARY_PATH),
        positions=_read(POSITION_MANIFEST_PATH),
        stage1=_read(STAGE1_GRID_PATH),
        stage2=_read(STAGE2_PACK_PATH) if STAGE2_PACK_PATH.is_file() else None,
        probe=json.loads(PROBE_PATH.read_text()) if PROBE_PATH.is_file() else None,
        latency=json.loads(LATENCY_PATH.read_text()) if LATENCY_PATH.is_file() else None,
        candidate=(
            json.loads(
                (
                    REPOSITORY_ROOT
                    / "checkpoints/phase16/phase16_stochastic_candidate_v1.json"
                ).read_text()
            )
            if (
                REPOSITORY_ROOT
                / "checkpoints/phase16/phase16_stochastic_candidate_v1.json"
            ).is_file()
            else None
        ),
        suite=json.loads(SUITE_PATH.read_text()) if SUITE_PATH.is_file() else None,
        benchscore=(
            json.loads(BENCHSCORE_PATH.read_text()) if BENCHSCORE_PATH.is_file() else None
        ),
        cli=(json.loads(CLI_PATH.read_text()) if CLI_PATH.is_file() else None),
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(payload)
    print(f"wrote {REPORT_PATH}")
    return {"path": str(REPORT_PATH), "bytes": len(payload)}


ROLES = {
    "boundary": role_boundary,
    "positions": role_positions,
    "stage1": role_stage1,
    "stage2": role_stage2,
    "probe": role_probe,
    "benchscore": role_benchscore,
    "latency": role_latency,
    "candidate": role_candidate,
    "report": role_report,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=sorted(ROLES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--budget", choices=("TINY", "MEDIUM"), default=None)
    parser.add_argument("--preset", choices=("TINY", "MEDIUM"), default=None)
    parser.add_argument("--positions", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--wait-lock", type=float, default=0.0, help="minutes to poll the compute lock")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    started = time.time()
    ROLES[args.role](args)
    print(f"role {args.role} done in {(time.time() - started)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
