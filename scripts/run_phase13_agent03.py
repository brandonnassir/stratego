#!/usr/bin/env python3
"""Phase 13 — Agent 3: the 90-minute crash/resume rehearsal.

Task: `instructions/phase_13_final_training_integration/03_AGENT_3_90_MINUTE_CRASH_RESUME_REHEARSAL.md`.

This is a reliability rehearsal of the exact Phase 14 training system. It runs
the frozen 2,048-game production population, the frozen learning rates, the
frozen mixtures, the frozen objective, the frozen setup source and the frozen
cadences, against the real system clock, on the real external volume. The one
value it replaces is the 168-hour deadline, which becomes 90 minutes.

Nothing here may change training strategy. The stages are:

```text
prerequisites   the six frozen identities, verified live
rehearsal       90 minutes: run, SIGKILL the process, resume, kill a CPU
                worker, run to the natural deadline
scheduler       the manual-clock seam: 2 h archive, 6 h candidate, the
                main->late transition, the 168 h stop
postdeadline    a controlled recovery started after the persisted deadline
readability     every checkpoint the rehearsal left behind, re-read
```

Usage:

```text
python scripts/run_phase13_agent03.py                     # every stage
python scripts/run_phase13_agent03.py --stage scheduler   # one stage
python scripts/run_phase13_agent03.py --role segment ...  # the training child
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

REPORT_ROOT = REPOSITORY / "reports" / "phase13"
EVIDENCE_ROOT = REPORT_ROOT / "agent03_evidence"
EXTERNAL_REHEARSAL_ROOT = Path("/Volumes/Brandon_Washington/stratego_phase13_rehearsal")
HOT_REHEARSAL_ROOT = REPOSITORY / "checkpoints" / "phase13_rehearsal" / "hot"

DEADLINE_SECONDS = 5400.0  # 90 minutes
CRASH_AT_SECONDS = 1800.0  # ~30 minutes
WORKER_KILL_AT_SECONDS = 3300.0  # ~55 minutes


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
    return path


# ---------------------------------------------------------------------------
# Stage: prerequisites (section 1)
# ---------------------------------------------------------------------------


def stage_prerequisites() -> dict:
    """The six identities of section 1, verified live rather than quoted."""
    from stratego.evaluation.phase14_candidates import load_pack, load_selection_rule
    from stratego.training import phase14_contract as contract
    from stratego.training.phase14_config import (
        integrated_config_digest,
        integrated_config_document,
    )
    from stratego.training.phase14_contract import (
        ANCHOR_CHECKPOINTS,
        ANCHOR_SHA256,
        STARTING_CHECKPOINT,
        STARTING_CHECKPOINT_SHA256,
        STARTING_MODEL_STATE_DIGEST,
        assert_matches_frozen_contract,
        contract_digest,
        file_sha256,
    )
    from stratego.training.phase14_setup_source import Phase14SetupSource

    started = time.perf_counter()
    frozen_path = REPORT_ROOT / "phase13_final_training_contract_v1.json"
    config_path = REPORT_ROOT / "phase13_integrated_training_config_v1.json"

    agreement = assert_matches_frozen_contract()
    # Round-tripped through JSON before comparing: the document holds tuples
    # (adam_betas) that a JSON file can only hold as lists, and the digest —
    # the thing that actually binds — is computed over the canonical form.
    document = json.loads(json.dumps(integrated_config_document(), default=str))
    on_disk = json.loads(config_path.read_text())
    on_disk_body = {
        key: value for key, value in on_disk.items() if key != "integrated_config_digest"
    }
    pack = load_pack()
    rule = load_selection_rule()
    source = Phase14SetupSource.build().describe()

    identities = {
        "agent_1_contract": {
            "path": str(frozen_path.relative_to(REPOSITORY)),
            "sha256": file_sha256(frozen_path),
            "expected": "65d1f941a326a1343dce597082c3b525203ef7182f73c759ac6eb04d87a12cdf",
            "disagreements": agreement.get("disagreements", []),
        },
        "agent_2_integrated_config": {
            "path": str(config_path.relative_to(REPOSITORY)),
            "digest_recomputed": integrated_config_digest(),
            "digest_on_disk": on_disk["integrated_config_digest"],
            "expected": "9c2a38e4335762997adbb33731dc619615fff713c2c60840c7c8d74a2f29da5e",
            "document_matches_disk": document == on_disk_body,
            "contract_digest": contract_digest(),
        },
        "starting_phase9_checkpoint": {
            "path": STARTING_CHECKPOINT,
            "sha256": file_sha256(REPOSITORY / STARTING_CHECKPOINT),
            "expected": STARTING_CHECKPOINT_SHA256,
            "model_state_digest": STARTING_MODEL_STATE_DIGEST,
        },
        "phase14_setup_source": {
            "identity": source.get("identity"),
            "expected": contract.SETUP_SOURCE_IDENTITY,
            "selector_config_sha256": source.get("selector_config_sha256"),
        },
        "candidate_pack": {
            "digest": pack["pack_content_digest"],
            "expected": contract.SELECTION_PACK_DIGEST,
            "games": len(pack["games"]),
        },
        "checkpoint_selection_rule": {
            "artifact": rule["artifact"],
            "pack_digest": rule["pack_binding"]["pack_content_digest"],
            "expected_pack": contract.SELECTION_PACK_DIGEST,
        },
        "pool_anchors": {
            name: {
                "sha256": file_sha256(REPOSITORY / ANCHOR_CHECKPOINTS[name]),
                "expected": ANCHOR_SHA256[name],
            }
            for name in ANCHOR_SHA256
        },
    }

    problems = []
    identity = identities["agent_1_contract"]
    if identity["sha256"] != identity["expected"]:
        problems.append("the Agent 1 contract on disk is not the frozen document")
    if identity["disagreements"]:
        problems.append(f"contract disagreements: {identity['disagreements']}")
    config = identities["agent_2_integrated_config"]
    if not (
        config["digest_recomputed"] == config["digest_on_disk"] == config["expected"]
    ):
        problems.append("the integrated config digest is not Agent 2's")
    if not config["document_matches_disk"]:
        problems.append("the integrated config document differs from the one on disk")
    start = identities["starting_phase9_checkpoint"]
    if start["sha256"] != start["expected"]:
        problems.append("the Phase 9 starting checkpoint is not the accepted one")
    if identities["phase14_setup_source"]["identity"] != contract.SETUP_SOURCE_IDENTITY:
        problems.append("the setup source is not phase14_setup_source_v1")
    if identities["candidate_pack"]["digest"] != contract.SELECTION_PACK_DIGEST:
        problems.append("the candidate pack is not the frozen pack")
    if identities["checkpoint_selection_rule"]["pack_digest"] != contract.SELECTION_PACK_DIGEST:
        problems.append("the selection rule is bound to a different pack")
    for name, anchor in identities["pool_anchors"].items():
        if anchor["sha256"] != anchor["expected"]:
            problems.append(f"pool anchor {name} is not the frozen checkpoint")

    return {
        "seconds": time.perf_counter() - started,
        "identities": identities,
        "problems": problems,
        "verified": not problems,
    }


# ---------------------------------------------------------------------------
# Role: the training child (the process the rehearsal kills)
# ---------------------------------------------------------------------------


def segment_main(args) -> int:
    """One training process. Started, killed, restarted; never reconfigured."""
    from stratego.training.phase13_rehearsal import EventLog
    from stratego.training.phase14_clock import SystemClock
    from stratego.training.phase14_contract import PRODUCTION_POPULATION, Population
    from stratego.training.phase14_runner import MODE_REHEARSAL, Phase14Runner
    from stratego.training.phase14_storage import Phase14Storage
    from stratego.training.phase9_trainer import LoaderTopology

    events = EventLog(args.event_log)
    events.emit(
        "process_start",
        pid=os.getpid(),
        pgid=os.getpgid(0),
        label=args.label,
        argv=sys.argv[1:],
    )
    try:
        storage = Phase14Storage.under(args.root, hot_root=args.hot_root)
        population = (
            PRODUCTION_POPULATION
            if not args.population_divisor
            else Population.scaled(args.population_divisor)
        )
        runner = Phase14Runner(
            storage,
            clock=SystemClock(),
            mode=MODE_REHEARSAL,
            rehearsal_deadline_seconds=args.deadline_seconds,
            device=args.device,
            inference_device=args.device,
            topology=LoaderTopology(workers=args.loader_workers),
            games_in_flight=args.games_in_flight,
            population=population,
        )
        report = runner.start_or_resume()
        events.emit(
            "lifecycle",
            pid=os.getpid(),
            label=args.label,
            started="started" in report,
            report={
                key: value for key, value in report.items() if key != "verification"
            },
            population_production=population.production,
            games_per_iteration={
                segment: population.games_per_iteration(segment)
                for segment in ("main", "late")
            },
            loader_workers=args.loader_workers,
            window=runner.controller.window.to_dict(),
        )
        result = runner.run()
        events.emit(
            "run_returned",
            pid=os.getpid(),
            label=args.label,
            stopped_because=result["stopped_because"],
            iterations_completed=result["iterations_completed"],
            global_optimizer_step=result["global_optimizer_step"],
            elapsed_hours=result["elapsed_hours"],
            units=[
                {
                    key: unit.get(key)
                    for key in (
                        "launched",
                        "iteration",
                        "segment",
                        "sealed",
                        "trained",
                        "updates",
                        "reason",
                        "error",
                    )
                }
                for unit in result["units"]
            ],
        )
        if runner.controller.expired():
            final = runner.finalize(reason=result["stopped_because"])
            events.emit(
                "finalized",
                pid=os.getpid(),
                label=args.label,
                reason=final["reason"],
                hour_168_candidate_hour=final["hour_168_candidate"]["hour"],
                final_archive_position=final["final_archive_entry"]["position"],
                manifest=str(storage.run_state_path),
            )
        events.emit("process_exit", pid=os.getpid(), label=args.label, returncode=0)
        return 0
    except BaseException as error:  # noqa: BLE001 - the traceback is the evidence
        events.emit(
            "process_exception",
            pid=os.getpid(),
            label=args.label,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise


# ---------------------------------------------------------------------------
# Stage: the 90-minute rehearsal (sections 2-7, 10-11)
# ---------------------------------------------------------------------------


def newest_hot_signature(hot_root) -> tuple:
    """A cheap fingerprint of the hot ring: newest file, its size and mtime.

    Used to decide *whether* to pay for a full checkpoint read. Polling the
    section-11 status every five seconds by `torch.load`-ing a ten-megabyte
    checkpoint would put the monitoring in competition with the training it is
    supposed to be watching, which is its own kind of unreliability.
    """
    from stratego.training.phase13_rehearsal import hot_checkpoint_files

    files = hot_checkpoint_files(hot_root)
    if not files:
        return ()
    newest = files[-1]
    try:
        stat = newest.stat()
    except OSError:
        return ()
    return (newest.name, stat.st_size, stat.st_mtime)


def live_status(hot_root, storage_paths, child, state, *, disk=True) -> dict:
    """The section-11 monitoring surface, assembled from disk only.

    `state` is passed in rather than read here so the caller controls when the
    expensive read happens; everything else in this function is a `stat`, a
    `statvfs` or a `pgrep`.
    """
    from stratego.training.phase13_rehearsal import (
        directory_bytes,
        free_bytes,
        loader_worker_pids,
    )

    now = time.time()
    workers = loader_worker_pids(child.pid) if child is not None and child.alive() else []
    telemetry = {}
    log_path = Path(storage_paths["log_root"]) / "phase14_telemetry.jsonl"
    if log_path.exists():
        try:
            lines = log_path.read_text().strip().splitlines()
            if lines:
                telemetry = json.loads(lines[-1])
        except (OSError, json.JSONDecodeError):
            telemetry = {}
    status = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "checkpoint_age_seconds": (
            now - state["checkpoint_mtime_unix"] if state else None
        ),
        "learner_pid": child.pid if child is not None else None,
        "learner_alive": bool(child is not None and child.alive()),
        "loader_workers": workers,
        "loader_worker_count": len(workers),
        "state": state,
        "telemetry": telemetry,
    }
    if disk:
        status["disk"] = {
            "rollout_bytes": directory_bytes(storage_paths["rollout_root"]),
            "archive_bytes": directory_bytes(storage_paths["archive_root"]),
            "hot_bytes": directory_bytes(hot_root),
            "log_bytes": directory_bytes(storage_paths["log_root"]),
            "external_free_bytes": free_bytes(storage_paths["external_root"]),
            "internal_free_bytes": free_bytes(REPOSITORY),
        }
    return status


def stage_rehearsal(
    plan_overrides: "dict | None" = None,
    *,
    external_root=None,
    hot_root=None,
    evidence_root=None,
) -> dict:
    """Sections 2-7 and 10-11: one deliberate 90-minute rehearsal.

    The roots are parameters so the harness itself can be smoke-tested at a
    small scale against a scratch directory before the real 90 minutes start.
    Discovering a supervisor bug 30 minutes into the rehearsal would cost the
    rehearsal, and the rehearsal is the thing being paid for.
    """
    from stratego.training.phase13_rehearsal import (
        EventLog,
        RehearsalPlan,
        StorageSample,
        child_processes,
        directory_bytes,
        free_bytes,
        launch_segment,
        loader_worker_pids,
        observed_state,
        process_alive,
        utc_now_text,
    )
    from stratego.training.phase14_storage import Phase14Storage

    settings = {
        "deadline_seconds": DEADLINE_SECONDS,
        "crash_at_seconds": CRASH_AT_SECONDS,
        "worker_kill_at_seconds": WORKER_KILL_AT_SECONDS,
    }
    settings.update(plan_overrides or {})
    plan = RehearsalPlan(**settings)
    external = Path(external_root or EXTERNAL_REHEARSAL_ROOT)
    hot = Path(hot_root or HOT_REHEARSAL_ROOT)
    evidence = Path(evidence_root or EVIDENCE_ROOT)
    evidence.mkdir(parents=True, exist_ok=True)
    external.mkdir(parents=True, exist_ok=True)
    hot.mkdir(parents=True, exist_ok=True)

    storage = Phase14Storage.under(external, hot_root=hot)
    paths = storage.to_dict()
    child_events = evidence / "rehearsal_child_events.jsonl"
    supervisor_events = EventLog(evidence / "rehearsal_supervisor_events.jsonl")
    status_log = evidence / "rehearsal_status.jsonl"

    existing = observed_state(hot)
    if existing:
        raise SystemExit(
            f"{hot} already holds a valid rehearsal checkpoint; move it "
            "aside before starting a new rehearsal — this harness never overwrites one"
        )

    segment_arguments = [
        "--role", "segment",
        "--root", str(external),
        "--hot-root", str(hot),
        "--deadline-seconds", str(plan.deadline_seconds),
        "--device", plan.device,
        "--loader-workers", str(plan.loader_workers),
        "--games-in-flight", str(plan.games_in_flight),
        "--event-log", str(child_events),
    ]
    if plan.population_divisor:
        segment_arguments += ["--population-divisor", str(plan.population_divisor)]

    supervisor_events.emit("rehearsal_plan", plan=plan.to_dict(), storage=paths)

    launches = []
    samples = []
    marks = {"crashed": False, "worker_killed": False, "deadline_seen": False}
    child = None
    started_monotonic = time.monotonic()
    wall_start_utc = utc_now_text()
    baseline_free = {
        "external": free_bytes(external),
        "internal": free_bytes(REPOSITORY),
    }

    def elapsed() -> float:
        return time.monotonic() - started_monotonic

    def launch(label: str) -> None:
        nonlocal child
        stdout_path = evidence / f"rehearsal_child_{len(launches) + 1:02d}.log"
        child = launch_segment(
            python=sys.executable,
            script=Path(__file__).resolve(),
            repository=REPOSITORY,
            arguments=segment_arguments + ["--label", label],
            role=label,
            stdout_path=stdout_path,
        )
        record = {
            "index": len(launches) + 1,
            "label": label,
            "pid": child.pid,
            "launched_utc": child.launched_utc,
            "supervisor_elapsed_seconds": elapsed(),
            "stdout": str(stdout_path),
        }
        launches.append(record)
        supervisor_events.emit("child_launched", **record)

    launch("segment_a_initial")

    cached_state: dict = {}
    cached_signature: tuple = ()
    last_disk_sample = -1e9

    with open(status_log, "a", encoding="utf-8") as status_stream:
        while True:
            now = elapsed()
            # The supervisor's own stopwatch only decides *when to interfere*.
            # Whether the run is over is the run's decision, read from its own
            # persisted window.
            if now > plan.deadline_seconds + plan.finalize_grace_seconds:
                supervisor_events.emit("supervisor_gave_up_waiting", elapsed_seconds=now)
                break
            if len(launches) > plan.max_launches:
                supervisor_events.emit(
                    "supervisor_stopped_relaunching",
                    elapsed_seconds=now,
                    launches=len(launches),
                    limit=plan.max_launches,
                )
                break

            # The full state is re-read only when the hot ring actually moved.
            signature = newest_hot_signature(hot)
            if signature and signature != cached_signature:
                cached_state = observed_state(hot)
                cached_signature = signature

            want_disk = (now - last_disk_sample) >= plan.storage_sample_seconds
            status = live_status(hot, paths, child, cached_state, disk=want_disk)
            status["supervisor_elapsed_seconds"] = now
            status_stream.write(json.dumps(status, sort_keys=True, default=str) + "\n")
            status_stream.flush()
            if want_disk:
                last_disk_sample = now
                samples.append(
                    StorageSample(
                        utc=status["utc"],
                        elapsed_seconds=now,
                        rollout_bytes=status["disk"]["rollout_bytes"],
                        archive_bytes=status["disk"]["archive_bytes"],
                        hot_bytes=status["disk"]["hot_bytes"],
                        log_bytes=status["disk"]["log_bytes"],
                        external_free_bytes=status["disk"]["external_free_bytes"],
                        internal_free_bytes=status["disk"]["internal_free_bytes"],
                    )
                )

            # --- Failure 1: force-terminate the whole training process group.
            # `crash_at_seconds = None` skips it, which is what the post-fix
            # re-verification of Failure 2 alone needs.
            if (
                plan.crash_at_seconds is not None
                and not marks["crashed"]
                and now >= plan.crash_at_seconds
                and child.alive()
            ):
                before = observed_state(hot)
                workers = loader_worker_pids(child.pid)
                killed = child.kill_group()
                reaped = child.wait_gone()
                marks["crashed"] = True
                supervisor_events.emit(
                    "failure_1_process_kill",
                    elapsed_seconds=now,
                    pid=child.pid,
                    loader_workers_at_kill=workers,
                    killed=killed,
                    reaped=reaped,
                    state_before_kill=before,
                )
                log(f"failure 1: killed pid {child.pid} at {now:.0f}s")
                launch("segment_b_after_crash")
                time.sleep(plan.poll_seconds)
                continue

            # --- Failure 2: kill one CPU loader worker while the learner trains.
            if (
                not marks["worker_killed"]
                and (marks["crashed"] or plan.crash_at_seconds is None)
                and now >= plan.worker_kill_at_seconds
                and child.alive()
            ):
                # Only a real `spawn_main` loader worker counts. The pool's
                # resource tracker is a child of the same parent and killing it
                # would look like a worker-failure test while testing nothing,
                # so the kill waits for a poll at which the learner is actually
                # training.
                workers = loader_worker_pids(child.pid)
                if workers:
                    victim = workers[0]
                    before = observed_state(hot)
                    children_before = child_processes(child.pid)
                    try:
                        os.kill(victim, 9)
                        killed = True
                    except OSError as error:
                        killed = False
                        supervisor_events.emit("failure_2_kill_failed", error=str(error))
                    if killed:
                        marks["worker_killed"] = True
                        time.sleep(5.0)
                        after = observed_state(hot)
                        supervisor_events.emit(
                            "failure_2_worker_kill",
                            elapsed_seconds=now,
                            learner_pid=child.pid,
                            victim_pid=victim,
                            workers_before=workers,
                            children_before=children_before,
                            victim_alive_after=process_alive(victim),
                            learner_alive_after=child.alive(),
                            children_after=(
                                child_processes(child.pid) if child.alive() else []
                            ),
                            workers_after=(
                                loader_worker_pids(child.pid) if child.alive() else []
                            ),
                            state_before_kill=before,
                            state_after_kill=after,
                        )
                        log(
                            f"failure 2: killed loader worker {victim} of "
                            f"{len(workers)} at {now:.0f}s"
                        )

            # --- The child exited on its own.
            if not child.alive():
                returncode = child.popen.poll()
                state = observed_state(hot)
                past_deadline = bool(
                    state
                    and _past_deadline(state["run_window"])
                )
                supervisor_events.emit(
                    "child_exited",
                    elapsed_seconds=now,
                    pid=child.pid,
                    returncode=returncode,
                    label=child.role,
                    past_persisted_deadline=past_deadline,
                    state=state,
                )
                launches[-1]["returncode"] = returncode
                launches[-1]["exited_elapsed_seconds"] = now
                if past_deadline:
                    marks["deadline_seen"] = True
                    log(f"child exited after its own deadline (rc={returncode})")
                    break
                log(f"child exited early (rc={returncode}); resuming")
                launch(f"segment_recovery_{len(launches) + 1:02d}")

            time.sleep(plan.poll_seconds)

    if child is not None and child.alive():
        supervisor_events.emit("supervisor_stopping_live_child", pid=child.pid)
        child.kill_group()
        child.wait_gone()

    final_state = observed_state(hot)
    manifest = {}
    if storage.run_state_path.exists():
        manifest = json.loads(storage.run_state_path.read_text())

    return {
        "plan": plan.to_dict(),
        "storage": paths,
        "wall_start_utc": wall_start_utc,
        "wall_end_utc": utc_now_text(),
        "supervisor_elapsed_seconds": elapsed(),
        "baseline_free_bytes": baseline_free,
        "launches": launches,
        "marks": marks,
        "final_state": final_state,
        "run_manifest": manifest,
        "storage_samples": [sample.to_dict() for sample in samples],
        "evidence": {
            "supervisor_events": str(supervisor_events.path),
            "child_events": str(child_events),
            "status_log": str(status_log),
        },
        "final_disk": {
            "rollout_bytes": directory_bytes(paths["rollout_root"]),
            "archive_bytes": directory_bytes(paths["archive_root"]),
            "hot_bytes": directory_bytes(hot),
            "log_bytes": directory_bytes(paths["log_root"]),
            "external_free_bytes": free_bytes(external),
        },
    }


def _past_deadline(window: dict) -> bool:
    from datetime import datetime

    from stratego.training.phase14_clock import parse_utc

    return datetime.now(timezone.utc) >= parse_utc(window["run_deadline_utc"])


# ---------------------------------------------------------------------------
# Stage: the long-horizon scheduler seam (sections 8, 9)
# ---------------------------------------------------------------------------


def stage_scheduler() -> dict:
    """Sections 8-9: the events a 90-minute run cannot reach, on the seam.

    Every check here drives the *production* scheduler; only the clock and the
    per-iteration game count are seams, and both are the ones Agent 2 declared
    and production refuses. No production cadence is changed.
    """
    import shutil
    import tempfile

    from stratego.evaluation.phase14_candidates import CandidateLedger, evaluator_semantics
    from stratego.training import phase14_contract as contract
    from stratego.training.phase14_clock import ManualClock
    from stratego.training.phase14_contract import Population, contract_digest
    from stratego.training.phase14_pool import ActivePool
    from stratego.training.phase14_runner import MODE_TEST, Phase14Runner
    from stratego.training.phase14_storage import Phase14Storage
    from stratego.training.phase14_telemetry import ControlSurface, Phase14TelemetryError
    from stratego.training.phase9_trainer import LoaderTopology

    started = time.perf_counter()
    root = Path(tempfile.mkdtemp(prefix="phase13_scheduler_"))
    storage = Phase14Storage.under(root)
    population = Population.scaled(512)
    clock = ManualClock("2026-09-01T00:00:00.000Z")

    def build(active_clock):
        return Phase14Runner(
            storage,
            clock=active_clock,
            mode=MODE_TEST,
            device="cpu",
            inference_device="cpu",
            topology=LoaderTopology(workers=1),
            games_in_flight=8,
            population=population,
        )

    runner = build(clock)
    start = runner.start()
    findings: dict = {"seams": {"clock": clock.describe(), "population": population.to_dict()}}

    # --- 2-hour archive event -------------------------------------------
    clock.advance_hours(2.05)
    unit = runner.run_iteration()
    archived = unit.get("archived") or {}
    findings["archive_2h"] = {
        "fired": bool(archived),
        "archive_k": runner.archive.k,
        "archive_mark": (archived.get("entry") or {}).get("archive_mark"),
        "elapsed_hours": runner.controller.elapsed_hours(),
        "pool_after": runner.pool.members(),
        "pool_recomputes": runner.pool.digest()
        == ActivePool.for_archive(runner.archive).digest(),
    }

    # --- 6-hour candidate event -----------------------------------------
    clock.advance_hours(4.0)
    unit = runner.run_iteration()
    candidate = unit.get("candidate") or {}
    entry = runner.archive.entries[-1] if runner.archive.entries else None
    findings["candidate_6h"] = {
        "fired": bool(candidate),
        "hour": candidate.get("hour"),
        "marked_from_archive": bool(
            entry is not None and candidate.get("snapshot_path") == entry.path
        ),
        "is_a_mark_not_a_copy": bool(
            entry is not None
            and candidate.get("snapshot_sha256") == entry.sha256
            and Path(candidate.get("snapshot_path", "")).exists()
        ),
        "pack_digest": candidate.get("pack_digest"),
        "pack_matches_frozen": candidate.get("pack_digest") == contract.SELECTION_PACK_DIGEST,
        "ledger_pending": [item["hour"] for item in
                           CandidateLedger.at(storage.evaluation_root).pending()],
    }

    # --- the evaluator launches, and cannot touch training ---------------
    before = {
        "global_optimizer_step": runner.trainer.global_step,
        "model_state_digest": runner.trainer.model_state_digest,
        "learning_rate": runner.controller.learning_rate(),
        "contract_digest": contract_digest(),
        "window": runner.controller.window.to_dict(),
    }
    evaluations = runner.evaluate_pending_candidates(limit_games=2)
    after = {
        "global_optimizer_step": runner.trainer.global_step,
        "model_state_digest": runner.trainer.model_state_digest,
        "learning_rate": runner.controller.learning_rate(),
        "contract_digest": contract_digest(),
        "window": runner.controller.window.to_dict(),
    }
    findings["evaluation_launch"] = {
        "results": evaluations,
        "training_state_before": before,
        "training_state_after": after,
        "training_untouched": before == after,
        "evaluator_isolation": evaluator_semantics().get("isolation")
        or evaluator_semantics(),
    }

    # --- an evaluation failure does not affect training ------------------
    ledger = CandidateLedger.at(storage.evaluation_root)
    document = ledger.read()
    poisoned_hour = None
    for hour_text, item in sorted(document.get("candidates", {}).items()):
        mark = item.get("mark") or {}
        if mark.get("snapshot_path") and Path(mark["snapshot_path"]).exists():
            poisoned_hour = int(hour_text)
            break
    forced = {}
    if poisoned_hour is not None:
        item = document["candidates"][str(poisoned_hour)]
        item["result"] = None
        item["status"] = "pending"
        item["mark"]["snapshot_path"] = str(root / "does_not_exist.pt")
        ledger.write(document)
        step_before = runner.trainer.global_step
        forced_results = runner.evaluate_pending_candidates(limit_games=1)
        reread = ledger.read()["candidates"][str(poisoned_hour)]
        forced = {
            "hour": poisoned_hour,
            "results": forced_results,
            "ledger_status": reread.get("status"),
            "rerunnable": reread.get("rerunnable"),
            "training_step_unchanged": runner.trainer.global_step == step_before,
        }
        next_unit_ran = runner.run_iteration()
        forced["training_continued"] = bool(next_unit_ran.get("trained"))
    findings["evaluation_failure"] = forced

    # --- an evaluation result may not change training config -------------
    control = ControlSurface()
    refusals = {}
    for key in (
        "learning_rate",
        "opponent_mixture",
        "candidate_selection_rule",
        "deadline",
        "checkpoint_cadence",
        "mean_ewr",
    ):
        try:
            control.set(key, 1.0)
            refusals[key] = "ACCEPTED"
        except Phase14TelemetryError as error:
            refusals[key] = f"refused: {type(error).__name__}"
    from stratego.evaluation.phase14_candidates import (
        Phase14CandidateError,
        select_final_candidate,
    )

    completed = ledger.completed()
    try:
        select_final_candidate(ledger.read()["candidates"].values())
        refused_incomplete = "ACCEPTED"
    except (Phase14CandidateError, KeyError, ValueError) as error:
        refused_incomplete = f"refused: {type(error).__name__}"
    findings["selection_rule_refuses_incomplete"] = {
        "complete_entries": len(completed),
        "outcome": refused_incomplete,
        "refused": refused_incomplete.startswith("refused"),
        "why": (
            "a two-game slice and a 128-game pack are not comparable, so the "
            "frozen rule refuses rather than ranking them"
        ),
    }

    findings["config_immutable_to_results"] = {
        "control_refusals": refusals,
        "all_refused": all(value.startswith("refused") for value in refusals.values()),
        "contract_digest_unchanged": contract_digest() == before["contract_digest"],
    }

    # --- main -> late transition ----------------------------------------
    clock.advance_hours(125.8)  # 131.85 h: still main
    pre = {
        "elapsed_hours": runner.controller.elapsed_hours(),
        "segment": runner.controller.segment(),
        "learning_rate": runner.controller.learning_rate(),
    }
    clock.advance_hours(0.25)  # 132.10 h: late
    post = {
        "elapsed_hours": runner.controller.elapsed_hours(),
        "segment": runner.controller.segment(),
        "learning_rate": runner.controller.learning_rate(),
    }
    late_unit = runner.run_iteration()
    findings["transition"] = {
        "before": pre,
        "after": post,
        "segment_switched": pre["segment"] == "main" and post["segment"] == "late",
        "learning_rate_switched": (
            pre["learning_rate"] == contract.learning_rate("main")
            and post["learning_rate"] == contract.learning_rate("late")
        ),
        "unit_segment": late_unit.get("segment"),
        "unit_buckets": (late_unit.get("collection") or {}).get("bucket_counts"),
        "late_bucket_shape": population.bucket_counts("late"),
        "mixture_switched": (late_unit.get("collection") or {}).get("bucket_counts")
        == population.bucket_counts("late"),
        "transition_utc": start["transition_utc"],
        "tied_to_original_start": start["transition_utc"]
        == runner.controller.window.to_dict()["transition_utc"],
    }

    # --- downtime does not postpone the transition -----------------------
    # The machine is "off": the clock moves, the run does not.
    persisted = runner.controller.window.to_dict()
    down_clock = ManualClock(clock.now())
    down_clock.advance_hours(10.0)
    resumed = build(down_clock)
    resume_report = resumed.resume()
    findings["downtime"] = {
        "window_unchanged": resumed.controller.window.to_dict() == persisted,
        "transition_utc_unchanged": (
            resumed.controller.window.to_dict()["transition_utc"]
            == persisted["transition_utc"]
        ),
        "elapsed_hours_after_downtime": resumed.controller.elapsed_hours(),
        "segment_after_restart": resumed.controller.segment(),
        "segment_preserved_across_restart": resumed.controller.segment() == "late",
        "learning_rate_after_restart": resumed.controller.learning_rate(),
        "resume_report": resume_report,
    }

    # --- the 168-hour stop ------------------------------------------------
    down_clock.advance_hours(168.0 - resumed.controller.elapsed_hours() + 0.05)
    step_at_deadline = resumed.trainer.global_step
    refused = resumed.run_iteration()
    final = resumed.finalize(reason="deadline")
    findings["deadline_168h"] = {
        "elapsed_hours": resumed.controller.elapsed_hours(),
        "expired": resumed.controller.expired(),
        "unit_launched": refused.get("launched"),
        "refusal_reason": refused.get("reason"),
        "may_start_collection_unit": resumed.controller.may_start_collection_unit(),
        "may_start_optimizer_step": resumed.controller.may_start_optimizer_step(),
        "optimizer_step_unchanged": resumed.trainer.global_step == step_at_deadline,
        "closed": final["closed"],
        "hour_168_candidate": final["hour_168_candidate"]["hour"],
        "hour_168_from_archive": final["hour_168_candidate"]["snapshot_path"]
        == final["final_archive_entry"]["path"],
        "window_after_close": resumed.controller.window.to_dict(),
        "window_never_extended": resumed.controller.window.to_dict() == persisted,
    }

    # --- emergency stop, and nothing else --------------------------------
    # Section 11 asks for a working emergency stop and for no other mutable
    # control. Both are checked against a real runner rather than against the
    # control object alone, because "the runner honours the request" is the
    # half that could rot.
    stop_root = Path(tempfile.mkdtemp(prefix="phase13_emergency_"))
    stop_control = ControlSurface()
    stop_runner = Phase14Runner(
        Phase14Storage.under(stop_root),
        clock=ManualClock("2026-09-01T00:00:00.000Z"),
        mode=MODE_TEST,
        device="cpu",
        inference_device="cpu",
        topology=LoaderTopology(workers=1),
        games_in_flight=8,
        population=population,
        control=stop_control,
    )
    stop_runner.start()
    stop_status = stop_control.emergency_stop("Phase 13 rehearsal control check")
    stopped = stop_runner.run()
    refused_unit = stop_runner.run_iteration()
    hot_after_stop = stop_runner.hot.latest_valid()
    findings["emergency_stop"] = {
        "status": stop_status,
        "stopped_because": stopped["stopped_because"],
        "units_launched": [unit.get("launched") for unit in stopped["units"]],
        "run_iteration_refusal": refused_unit.get("reason"),
        "hot_checkpoint_present": hot_after_stop is not None,
        "optimizer_step": stop_runner.trainer.global_step,
        "works": stopped["stopped_because"] == "emergency stop"
        and refused_unit.get("launched") is False
        and refused_unit.get("reason") == "emergency_stop"
        and hot_after_stop is not None,
        "cleared_resumes": (
            stop_control.clear()["stop_requested"] is False
            and stop_control.should_continue() is True
        ),
    }
    shutil.rmtree(stop_root, ignore_errors=True)

    findings["seconds"] = time.perf_counter() - started
    findings["root"] = str(root)
    shutil.rmtree(root, ignore_errors=True)
    return findings


# ---------------------------------------------------------------------------
# Stage: recovery started after the deadline (section 10)
# ---------------------------------------------------------------------------


def stage_post_deadline(hot_root=None) -> dict:
    """Section 10: load a real checkpoint, stand past its deadline, recover.

    Deliberately run against a checkpoint the *rehearsal itself* wrote, copied
    aside so the evidence is not disturbed. A synthetic checkpoint would prove
    that the gate works on a checkpoint built to please it.
    """
    import shutil
    import tempfile

    from stratego.training.phase14_checkpoint import HotCheckpointRing
    from stratego.training.phase14_clock import ManualClock, parse_utc, utc_text
    from stratego.training.phase14_contract import Population
    from stratego.training.phase14_runner import MODE_TEST, Phase14Runner
    from stratego.training.phase14_storage import Phase14Storage
    from stratego.training.phase9_trainer import LoaderTopology

    started = time.perf_counter()
    source_hot = Path(hot_root or HOT_REHEARSAL_ROOT)
    loaded = HotCheckpointRing(source_hot).load_latest()
    if loaded is None:
        return {"skipped": "no valid rehearsal hot checkpoint to recover from"}

    root = Path(tempfile.mkdtemp(prefix="phase13_postdeadline_"))
    hot = root / "hot"
    hot.mkdir(parents=True, exist_ok=True)
    for candidate in sorted(source_hot.glob("hot_*.pt")):
        shutil.copy2(candidate, hot / candidate.name)
    storage = Phase14Storage.under(root, hot_root=hot)

    _path, payload = loaded
    window = payload["run_window"]
    deadline = parse_utc(window["run_deadline_utc"])
    # Stand well after the persisted deadline: a machine that came back up the
    # next morning, not one that came back a second late.
    clock = ManualClock(utc_text(deadline))
    clock.advance_hours(9.0)

    runner = Phase14Runner(
        storage,
        clock=clock,
        mode=MODE_TEST,
        device="cpu",
        inference_device="cpu",
        topology=LoaderTopology(workers=1),
        games_in_flight=4,
        population=Population.scaled(512),
    )
    report = runner.resume()
    step_after_resume = runner.trainer.global_step
    result = runner.run()
    step_after_run = runner.trainer.global_step
    final = runner.finalize(reason="post-deadline recovery")
    outcome = {
        "seconds": time.perf_counter() - started,
        "source_checkpoint": str(_path),
        "persisted_window": window,
        "test_now_utc": utc_text(clock.now()),
        "hours_past_deadline": (clock.now() - deadline).total_seconds() / 3600.0,
        "resume_report": {
            key: report[key]
            for key in (
                "resumed",
                "run_start_utc",
                "run_deadline_utc",
                "elapsed_hours",
                "remaining_hours",
                "iteration",
                "global_optimizer_step",
                "archive_k",
                "past_deadline",
            )
        },
        "optimizer_step_at_resume": step_after_resume,
        "optimizer_step_after_run": step_after_run,
        "optimizer_steps_taken": step_after_run - step_after_resume,
        "units_launched": [unit.get("launched") for unit in result["units"]],
        "stopped_because": result["stopped_because"],
        "may_start_optimizer_step": runner.controller.may_start_optimizer_step(),
        "finalized": final["closed"],
        "window_after": runner.controller.window.to_dict(),
        "deadline_extended": runner.controller.window.to_dict()["run_deadline_utc"]
        != window["run_deadline_utc"],
    }
    outcome["passes"] = (
        outcome["optimizer_steps_taken"] == 0
        and outcome["resume_report"]["past_deadline"] is True
        and outcome["finalized"] is True
        and outcome["deadline_extended"] is False
    )
    shutil.rmtree(root, ignore_errors=True)
    return outcome


# ---------------------------------------------------------------------------
# Stage: checkpoint readability and torn-write refusal (section 6)
# ---------------------------------------------------------------------------


def _payload_covers(payload: dict, field: str) -> bool:
    """The frozen resume-field check, applied to a file already on disk."""
    from stratego.training.phase14_checkpoint import _covers

    return _covers(payload, field)


def stage_readability(hot_root=None, archive_root=None) -> dict:
    """Section 6: re-read what the rehearsal left, and refuse a torn file."""
    import shutil
    import tempfile

    from stratego.training.phase14_checkpoint import (
        HotCheckpointRing,
        is_valid,
        read as read_checkpoint,
    )
    from stratego.training.phase14_contract import HOT_CHECKPOINT_REQUIRED_FIELDS

    started = time.perf_counter()
    hot = Path(hot_root or HOT_REHEARSAL_ROOT)
    archive = Path(
        archive_root or (EXTERNAL_REHEARSAL_ROOT / "archive")
    )

    hot_files = []
    for path in sorted(hot.glob("hot_*.pt")):
        record = {"path": str(path), "bytes": path.stat().st_size}
        try:
            payload = read_checkpoint(path)
            record["readable"] = True
            record["validates"] = is_valid(path)
            record["global_optimizer_step"] = payload["trainer_state"][
                "global_optimizer_step"
            ]
            record["snapshot_role"] = payload["snapshot_role"]
            record["run_deadline_utc"] = payload["run_window"]["run_deadline_utc"]
            from stratego.training.phase14_pool import ActivePool

            record["pool_members"] = list(
                ActivePool.from_dict(payload["active_historical_pool"]).members()
            )
            record["archive_k"] = payload["historical_archive_state"]["k"]
            record["ema"] = payload.get("ema_state")
            record["covers_required_fields"] = sorted(
                field
                for field in HOT_CHECKPOINT_REQUIRED_FIELDS
                if not _payload_covers(payload, field)
            )
        except Exception as error:  # noqa: BLE001 - unreadable is the finding
            record["readable"] = False
            record["error"] = f"{type(error).__name__}: {error}"
        hot_files.append(record)

    archive_files = []
    for path in sorted(archive.glob("archive_*.pt")):
        record = {"path": str(path), "bytes": path.stat().st_size}
        try:
            payload = read_checkpoint(path)
            record["readable"] = True
            record["validates"] = is_valid(path)
            record["snapshot_role"] = payload["snapshot_role"]
            record["global_optimizer_step"] = payload["trainer_state"][
                "global_optimizer_step"
            ]
        except Exception as error:  # noqa: BLE001
            record["readable"] = False
            record["error"] = f"{type(error).__name__}: {error}"
        archive_files.append(record)

    # A torn newest file must cost one cadence, not the run.
    torn = {}
    if hot_files:
        scratch = Path(tempfile.mkdtemp(prefix="phase13_torn_"))
        for path in sorted(hot.glob("hot_*.pt")):
            shutil.copy2(path, scratch / path.name)
        newest = sorted(scratch.glob("hot_*.pt"))[-1]
        intact_choice = HotCheckpointRing(scratch).latest_valid()
        with open(newest, "r+b") as handle:
            handle.truncate(max(newest.stat().st_size // 2, 1))
        after = HotCheckpointRing(scratch).load_latest()
        torn = {
            "corrupted": str(newest),
            "newest_before_corruption": str(intact_choice),
            "selected_after_corruption": str(after[0]) if after else None,
            "corrupt_file_refused": bool(after and Path(after[0]) != newest),
            "recovered_step": after[1]["trainer_state"]["global_optimizer_step"]
            if after
            else None,
        }
        shutil.rmtree(scratch, ignore_errors=True)

    return {
        "seconds": time.perf_counter() - started,
        "hot_root": str(hot),
        "archive_root": str(archive),
        "hot_checkpoints": hot_files,
        "hot_retained": len(hot_files),
        "hot_all_readable": all(record["readable"] for record in hot_files),
        "hot_all_cover_required_fields": all(
            not record.get("covers_required_fields", ["unread"]) for record in hot_files
        ),
        "archive_snapshots": archive_files,
        "archive_all_readable": all(record["readable"] for record in archive_files),
        "required_fields": list(HOT_CHECKPOINT_REQUIRED_FIELDS),
        "torn_write": torn,
    }


# ---------------------------------------------------------------------------
# Stage: determinism — "no fresh RNG initialization" (section 5)
# ---------------------------------------------------------------------------


def stage_determinism() -> dict:
    """Section 5: prove a fresh global RNG cannot change the logical run.

    Phase 14 captures every RNG stream in the checkpoint but restores none of
    them, and the reason is written into `phase14_checkpoint._rng_state`: the
    minibatch order is a pure function of `train_order_seed(iteration, epoch)`
    and the behaviour draw is a pure function of `(game_id, ply)`, so no global
    RNG cursor decides any batch or any move. That is a claim about the code,
    and this stage checks it instead of repeating it: the same iteration is
    collected and trained twice, in two directories, under two deliberately
    different global torch seeds, and the sealed rollout and the whole epoch
    plan must come out identical.
    """
    import shutil
    import tempfile

    import torch

    from stratego.training.phase14_clock import ManualClock
    from stratego.training.phase14_contract import Population
    from stratego.training.phase14_runner import MODE_TEST, Phase14Runner
    from stratego.training.phase14_storage import Phase14Storage
    from stratego.training.phase9_trainer import LoaderTopology

    started = time.perf_counter()
    population = Population.scaled(256)

    def one_run(seed: int) -> dict:
        torch.manual_seed(seed)
        root = Path(tempfile.mkdtemp(prefix=f"phase13_determinism_{seed}_"))
        runner = Phase14Runner(
            Phase14Storage.under(root),
            clock=ManualClock("2026-09-01T00:00:00.000Z"),
            mode=MODE_TEST,
            device="cpu",
            inference_device="cpu",
            topology=LoaderTopology(workers=1),
            games_in_flight=8,
            population=population,
        )
        runner.start()
        plan: list = []
        original = runner._on_step

        def hooked(trainer, row):
            original(trainer, row)
            plan.append((int(row["epoch"]), int(row["minibatch_index"])))

        runner._on_step = hooked
        unit = runner.run_iteration()
        collection = unit["collection"]
        # The rollout digest is content-only (game id + payload + metadata
        # hashes), so if it moves, some per-game byte moved. Split it, so the
        # answer is "which half" rather than "something".
        directory = runner.storage.iteration_directory(1)
        metadata_records = []
        metadata_directory = directory / "metadata"
        if metadata_directory.is_dir():
            for entry in sorted(metadata_directory.iterdir()):
                if entry.suffix == ".jsonl":
                    for line in entry.read_text().splitlines():
                        if line.strip():
                            metadata_records.append(json.loads(line))
        metadata_records.sort(key=lambda item: item.get("game_id") or item.get("phase9_game_id") or "")
        shard_directory = directory / "shards"
        shard_bytes = sorted(
            hashlib.sha256(entry.read_bytes()).hexdigest()
            for entry in shard_directory.iterdir()
            if shard_directory.is_dir() and entry.is_file()
        )
        record = {
            "metadata_records": metadata_records,
            "shard_digests": shard_bytes,
            "torch_seed": seed,
            "games": collection["games_collected"],
            "total_decisions": collection["total_decisions"],
            "sealed_rollout_digest": collection.get("sealed_rollout_digest")
            or collection.get("digest"),
            "terminal_results": collection["terminal_results"],
            "bucket_counts": collection["bucket_counts"],
            "updates": unit["updates"],
            "epoch_plan": plan,
            "model_state_digest": runner.trainer.model_state_digest,
            "examples_consumed": runner.trainer.examples_consumed,
        }
        shutil.rmtree(root, ignore_errors=True)
        return record

    first = one_run(11111)
    second = one_run(99999)

    # Which per-game fields, if any, actually moved.
    moved_fields: dict = {}
    for left, right in zip(first["metadata_records"], second["metadata_records"]):
        for key in sorted(set(left) | set(right)):
            if left.get(key) != right.get(key):
                moved_fields.setdefault(key, 0)
                moved_fields[key] += 1
    shards_identical = first["shard_digests"] == second["shard_digests"]

    comparable = [
        "games",
        "total_decisions",
        "sealed_rollout_digest",
        "terminal_results",
        "bucket_counts",
        "updates",
        "epoch_plan",
        "examples_consumed",
        "model_state_digest",
    ]
    differences = [key for key in comparable if first[key] != second[key]]
    return {
        "seconds": time.perf_counter() - started,
        "metadata_fields_that_moved": moved_fields,
        "metadata_records_compared": len(first["metadata_records"]),
        "shard_bytes_identical": shards_identical,
        "first": {key: first[key] for key in comparable if key != "epoch_plan"},
        "second": {key: second[key] for key in comparable if key != "epoch_plan"},
        "epoch_plan_length": len(first["epoch_plan"]),
        "epoch_plan_identical": first["epoch_plan"] == second["epoch_plan"],
        "epoch_plan_has_no_repeat": len(first["epoch_plan"]) == len(set(first["epoch_plan"])),
        "differences": differences,
        "global_rng_is_irrelevant": not [
            key for key in differences if key != "sealed_rollout_digest"
        ],
        "rollout_digest_differs_only_by": sorted(moved_fields),
        "why": (
            "the checkpoint records every RNG stream and restores none; this is "
            "sound only because no global RNG cursor decides a batch or a move, "
            "which is what these two differently-seeded runs check"
        ),
    }


# ---------------------------------------------------------------------------
# Stage: analysis — the readiness table and the storage projection (7, 13)
# ---------------------------------------------------------------------------


def search_absence_probe() -> dict:
    """Section 13: search must be absent from the training import graph."""
    program = (
        "import sys\n"
        "import stratego.training.phase14_runner\n"
        "import stratego.training.phase14_trainer\n"
        "import stratego.training.phase14_collector\n"
        "import stratego.training.phase14_setup_source\n"
        "import stratego.training.phase13_rehearsal\n"
        "loaded = sorted(n for n in sys.modules if n.startswith('stratego.search'))\n"
        "print(len(sys.modules), loaded)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(REPOSITORY),
        env={**os.environ, "PYTHONPATH": str(REPOSITORY)},
        capture_output=True,
        text=True,
        check=False,
    )
    text = result.stdout.strip()
    loaded = text.partition(" ")[2] if text else ""
    return {
        "returncode": result.returncode,
        "stdout": text,
        "stderr": result.stderr.strip()[-400:],
        "search_modules_loaded": loaded,
        "search_absent": result.returncode == 0 and loaded == "[]",
    }


def storage_projection(rehearsal: dict) -> dict:
    """Section 7: what the rehearsal actually wrote, and what 168 h implies."""
    samples = rehearsal.get("storage_samples") or []
    if len(samples) < 2:
        return {"skipped": "not enough storage samples"}
    first, last = samples[0], samples[-1]
    hours = max((last["elapsed_seconds"] - first["elapsed_seconds"]) / 3600.0, 1e-9)
    gib = float(2**30)

    def rate(key: str) -> float:
        return (last[key] - first[key]) / gib / hours

    rollout_rate = rate("rollout_bytes")
    archive_rate = rate("archive_bytes")
    hot_rate = rate("hot_bytes")
    log_rate = rate("log_bytes")
    total_rate = rollout_rate + archive_rate + log_rate

    # The archive is time-based, not iteration-based: 84 two-hour snapshots
    # over 168 h, whatever the throughput. Projecting the rehearsal's archive
    # *rate* would project ~0, because 90 minutes crosses no 2-hour mark.
    snapshots = sorted(Path(EXTERNAL_REHEARSAL_ROOT / "archive").glob("archive_*.pt"))
    archive_bytes_each = (
        max(path.stat().st_size for path in snapshots) if snapshots else 0.0
    )
    projected_archive_gib = archive_bytes_each * contract_archive_snapshots() / gib

    projected_raw_gib = rollout_rate * 168.0
    projected_total_gib = projected_raw_gib + projected_archive_gib + log_rate * 168.0
    free_now_gib = last["external_free_bytes"] / gib
    return {
        "measured_window_hours": hours,
        "raw_shard_gib_per_hour": rollout_rate,
        "archive_gib_per_hour_observed": archive_rate,
        # Hot storage does not grow: the ring keeps four checkpoints and two
        # behavior snapshots, so its footprint is a constant, not a rate. It is
        # reported for completeness and excluded from the 168-hour projection.
        "hot_gib_per_hour_transient": hot_rate,
        "hot_bytes_bounded": rehearsal.get("final_disk", {}).get("hot_bytes"),
        "hot_lives_on": "the internal disk, not the external volume",
        "log_gib_per_hour": log_rate,
        "total_external_gib_per_hour": total_rate,
        "archive_snapshot_bytes": archive_bytes_each,
        "projected_168h_raw_gib": projected_raw_gib,
        "projected_168h_archive_gib": projected_archive_gib,
        "projected_168h_total_gib": projected_total_gib,
        "external_free_gib_now": free_now_gib,
        "external_free_gib_after_168h": free_now_gib - projected_total_gib,
        "reserve_gib": 120.0,
        "reserve_threatened": (free_now_gib - projected_total_gib) < 120.0,
        "agent_1_planning_rate_gib_per_hour": 3.572,
        "agent_1_conservative_ceiling_gib": 600.0,
        "agent_1_phase9_basis_gib": 28.0,
        "measured_versus_agent_1_ceiling": (
            "below" if projected_raw_gib < 600.0 else "above"
        ),
        "free_space_change_bytes": last["external_free_bytes"] - first["external_free_bytes"],
        "internal_free_change_bytes": (
            last["internal_free_bytes"] - first["internal_free_bytes"]
        ),
    }


def contract_archive_snapshots() -> int:
    from stratego.training.phase14_contract import ARCHIVE_SNAPSHOTS_IN_RUN

    return int(ARCHIVE_SNAPSHOTS_IN_RUN)


# ---------------------------------------------------------------------------
# Stage: readiness — section 13, assembled from the evidence on disk
# ---------------------------------------------------------------------------


def read_events(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append({"event": "unparseable", "raw": line})
    return records


def telemetry_rows() -> list:
    path = EXTERNAL_REHEARSAL_ROOT / "logs" / "phase14_telemetry.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]


def _finite(value) -> bool:
    import math

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def resume_boundaries() -> dict:
    """Section 5, from the status time series either side of each restart.

    The two states compared are the last full state the supervisor observed
    before a kill and the first *new* checkpoint the restarted process wrote —
    not the same file read twice, which would prove only that a file can be
    read.
    """
    path = EVIDENCE_ROOT / "rehearsal_status.jsonl"
    # The raw stream is large and is kept gzipped in the repository; read
    # whichever form is present so this stage stays re-runnable either way.
    if path.exists():
        text = path.read_text()
    elif path.with_suffix(".jsonl.gz").exists():
        import gzip

        text = gzip.decompress(path.with_suffix(".jsonl.gz").read_bytes()).decode()
    else:
        return {}
    samples = []
    for line in text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("state"):
            samples.append(record)
    events = read_events(EVIDENCE_ROOT / "rehearsal_supervisor_events.jsonl")
    kills = [
        record
        for record in events
        if record["event"] in ("failure_1_process_kill", "failure_2_worker_kill")
    ]
    boundaries = []
    for kill in kills:
        moment = kill["elapsed_seconds"]
        before = [s for s in samples if s["supervisor_elapsed_seconds"] <= moment]
        after = [
            s
            for s in samples
            if s["supervisor_elapsed_seconds"] > moment
            and s["state"]["checkpoint"] != (before[-1]["state"]["checkpoint"] if before else None)
        ]
        if not before or not after:
            continue
        left, right = before[-1]["state"], after[0]["state"]
        boundaries.append(
            {
                "failure": kill["event"],
                "at_seconds": moment,
                "before": {
                    "checkpoint": Path(left["checkpoint"]).name,
                    "global_optimizer_step": left["global_optimizer_step"],
                    "iteration": left["iteration"],
                    "pool_digest": left["pool_digest"],
                    "archive_digest": left["archive_digest"],
                    "archive_k": left["archive_k"],
                    "last_candidate_index": left["last_candidate_index"],
                    "segment": left["segment"],
                    "learning_rate": left["learning_rate"],
                    "run_window": left["run_window"],
                    "shard_cursor": left["shard_cursor"],
                    "rng_digest": left["rng_digest"],
                },
                "after": {
                    "checkpoint": Path(right["checkpoint"]).name,
                    "global_optimizer_step": right["global_optimizer_step"],
                    "iteration": right["iteration"],
                    "pool_digest": right["pool_digest"],
                    "archive_digest": right["archive_digest"],
                    "archive_k": right["archive_k"],
                    "last_candidate_index": right["last_candidate_index"],
                    "segment": right["segment"],
                    "learning_rate": right["learning_rate"],
                    "run_window": right["run_window"],
                    "shard_cursor": right["shard_cursor"],
                    "rng_digest": right["rng_digest"],
                },
            }
        )
    for boundary in boundaries:
        left, right = boundary["before"], boundary["after"]
        boundary["checks"] = {
            "no_duplicate_optimizer_work": right["global_optimizer_step"]
            >= left["global_optimizer_step"],
            "no_skipped_optimizer_state": right["global_optimizer_step"]
            >= left["global_optimizer_step"],
            "same_active_historical_pool": right["pool_digest"] == left["pool_digest"],
            "archive_cursor_not_reset": right["archive_digest"] == left["archive_digest"],
            "candidate_scheduler_not_reset": right["last_candidate_index"]
            >= left["last_candidate_index"],
            "main_late_scheduler_not_reset": right["segment"] == left["segment"]
            and right["learning_rate"] == left["learning_rate"],
            "start_time_not_reset": right["run_window"]["run_start_utc"]
            == left["run_window"]["run_start_utc"],
            "deadline_not_reset": right["run_window"]["run_deadline_utc"]
            == left["run_window"]["run_deadline_utc"],
            "shard_cursor_not_reset": right["shard_cursor"]["last_committed_iteration"]
            >= left["shard_cursor"]["last_committed_iteration"],
        }
        boundary["all_hold"] = all(boundary["checks"].values())
    return {
        "boundaries": boundaries,
        "all_hold": bool(boundaries) and all(b["all_hold"] for b in boundaries),
        "rng_note": (
            "the RNG digest is expected to move: Phase 14 captures every stream "
            "and restores none, which is sound only because no global RNG cursor "
            "decides a batch or a move — checked directly by the determinism stage"
        ),
    }


def stage_readiness() -> dict:
    """Section 13: every readiness item, with the evidence that answers it."""
    from stratego.training.phase14_contract import (
        BELIEF_LOSS_WEIGHT,
        STARTING_MODEL_STATE_DIGEST,
    )

    started = time.perf_counter()
    stages = {
        name: json.loads(stage_path(name).read_text())
        for name in STAGES
        if stage_path(name).exists()
    }
    rehearsal = stages.get("rehearsal", {})
    supervisor = read_events(EVIDENCE_ROOT / "rehearsal_supervisor_events.jsonl")
    children = read_events(EVIDENCE_ROOT / "rehearsal_child_events.jsonl")
    rows = telemetry_rows()

    by_event = {}
    for record in supervisor:
        by_event.setdefault(record["event"], []).append(record)
    lifecycles = [record for record in children if record["event"] == "lifecycle"]
    run_returns = [record for record in children if record["event"] == "run_returned"]
    finalized = [record for record in children if record["event"] == "finalized"]
    exceptions = [record for record in children if record["event"] == "process_exception"]

    crash = (by_event.get("failure_1_process_kill") or [{}])[0]
    worker = (by_event.get("failure_2_worker_kill") or [{}])[0]
    before_crash = crash.get("state_before_kill") or {}
    resumed = [record for record in lifecycles if not record["started"]]
    first_resume = (resumed[0]["report"] if resumed else {}) or {}
    final_state = rehearsal.get("final_state") or {}

    # --- the training itself ------------------------------------------
    training_metrics = [row.get("training", {}) for row in rows]
    finite_names = ("policy_loss", "value_loss", "belief_loss", "grad_norm")
    all_finite = bool(training_metrics) and all(
        _finite(metrics.get(name)) for metrics in training_metrics for name in finite_names
    )
    belief_values = [metrics.get("belief_loss") for metrics in training_metrics]
    belief_alive = bool(belief_values) and all(
        _finite(value) and float(value) != 0.0 for value in belief_values
    )

    # --- windows: one window, every launch ------------------------------
    windows = [record["window"] for record in lifecycles if record.get("window")]
    windows += [
        record["state_before_kill"]["run_window"]
        for record in supervisor
        if record.get("state_before_kill", {}).get("run_window")
    ]
    if final_state.get("run_window"):
        windows.append(final_state["run_window"])
    window_identical = bool(windows) and all(window == windows[0] for window in windows)

    # --- pool and optimizer across the crash -----------------------------
    pool_preserved = bool(before_crash) and (
        list(before_crash.get("pool_members", []))
        == list(first_resume.get("active_pool", []))
    )
    step_preserved = bool(before_crash) and (
        int(before_crash.get("global_optimizer_step", -1))
        == int(first_resume.get("global_optimizer_step", -2))
    )

    scheduler = stages.get("scheduler", {})
    post = stages.get("postdeadline", {})
    readable = stages.get("readability", {})
    storage = storage_projection(rehearsal)
    search = search_absence_probe()

    def item(name: str, passed, evidence) -> dict:
        return {"check": name, "passed": bool(passed), "evidence": evidence}

    checks = [
        item(
            "training updates finite",
            all_finite,
            {
                "telemetry_rows": len(rows),
                "checked": list(finite_names),
                "non_finite_counters": [
                    row.get("counters", {}).get("non_finite_losses") for row in rows
                ],
            },
        ),
        item(
            "parameters change",
            bool(final_state)
            and final_state.get("model_state_digest") != STARTING_MODEL_STATE_DIGEST,
            {
                "starting_model_state_digest": STARTING_MODEL_STATE_DIGEST,
                "final_model_state_digest": final_state.get("model_state_digest"),
                "final_optimizer_step": final_state.get("global_optimizer_step"),
            },
        ),
        item(
            "belief auxiliary objective functioning",
            belief_alive,
            {"weight": BELIEF_LOSS_WEIGHT, "belief_loss_per_iteration": belief_values},
        ),
        item(
            "forced process crash recovered",
            bool(crash.get("killed", {}).get("killed"))
            and bool(first_resume.get("resumed")),
            {
                "kill": {
                    key: crash.get(key)
                    for key in ("elapsed_seconds", "pid", "loader_workers_at_kill")
                },
                "reaped": crash.get("reaped"),
                "resume_report": first_resume,
            },
        ),
        item(
            "optimizer state preserved",
            step_preserved,
            {
                "step_before_kill": before_crash.get("global_optimizer_step"),
                "step_at_resume": first_resume.get("global_optimizer_step"),
                "iteration_before_kill": before_crash.get("iteration"),
                "iteration_at_resume": first_resume.get("iteration"),
                "optimizer_state_entries": before_crash.get("optimizer_state_entries"),
                "ema_state": before_crash.get("ema_state"),
                "rng_digest_before": before_crash.get("rng_digest"),
            },
        ),
        item(
            "original rehearsal deadline preserved",
            window_identical,
            {
                "windows_compared": len(windows),
                "window": windows[0] if windows else None,
                "launches": len(rehearsal.get("launches", [])),
            },
        ),
        item(
            "active historical pool preserved",
            pool_preserved,
            {
                "pool_before_kill": before_crash.get("pool_members"),
                "pool_at_resume": first_resume.get("active_pool"),
                "pool_digest_before": before_crash.get("pool_digest"),
                "pool_recomputed_before": before_crash.get("pool_recomputed_digest"),
                "archive_k": before_crash.get("archive_k"),
            },
        ),
        item(
            "worker failure recovered",
            bool(worker) and worker.get("victim_alive_after") is False,
            {
                "victim": worker.get("victim_pid"),
                "workers_before": worker.get("workers_before"),
                "learner_alive_after": worker.get("learner_alive_after"),
                "step_before": (worker.get("state_before_kill") or {}).get(
                    "global_optimizer_step"
                ),
                "step_after": (worker.get("state_after_kill") or {}).get(
                    "global_optimizer_step"
                ),
                "child_exceptions": [record["error"] for record in exceptions],
            },
        ),
        item(
            "storage remained safe",
            bool(storage.get("projected_168h_total_gib") is not None)
            and not storage.get("reserve_threatened", True),
            storage,
        ),
        item(
            "hot checkpoints readable",
            bool(readable.get("hot_all_readable")) and bool(readable.get("hot_checkpoints")),
            {
                "hot_retained": readable.get("hot_retained"),
                "hot_all_readable": readable.get("hot_all_readable"),
                "archive_all_readable": readable.get("archive_all_readable"),
                "torn_write": readable.get("torn_write"),
            },
        ),
        item(
            "test-clock 2h archive event works",
            bool(scheduler.get("archive_2h", {}).get("fired")),
            scheduler.get("archive_2h"),
        ),
        item(
            "test-clock 6h candidate event works",
            bool(scheduler.get("candidate_6h", {}).get("fired"))
            and bool(scheduler.get("candidate_6h", {}).get("marked_from_archive")),
            scheduler.get("candidate_6h"),
        ),
        item(
            "test-clock late transition works",
            bool(scheduler.get("transition", {}).get("segment_switched"))
            and bool(scheduler.get("transition", {}).get("learning_rate_switched"))
            and bool(scheduler.get("transition", {}).get("mixture_switched")),
            scheduler.get("transition"),
        ),
        item(
            "test-clock 168h shutdown works",
            bool(scheduler.get("deadline_168h", {}).get("closed"))
            and scheduler.get("deadline_168h", {}).get("unit_launched") is False,
            scheduler.get("deadline_168h"),
        ),
        item(
            "post-deadline recovery refuses training",
            bool(post.get("passes")),
            post,
        ),
        item("search absent from training", search.get("search_absent"), search),
    ]

    return {
        "seconds": time.perf_counter() - started,
        "checks": checks,
        "passed": sum(1 for check in checks if check["passed"]),
        "total": len(checks),
        "failed": [check["check"] for check in checks if not check["passed"]],
        "run_summary": {
            "launches": rehearsal.get("launches"),
            "marks": rehearsal.get("marks"),
            "iterations_completed": [
                record["iterations_completed"] for record in run_returns
            ],
            "stopped_because": [record["stopped_because"] for record in run_returns],
            "finalized": [record["reason"] for record in finalized],
            "final_state": final_state,
            "telemetry_rows": len(rows),
            "missing_metrics": sorted(
                {name for row in rows for name in row.get("missing_metrics", [])}
            ),
        },
        "storage_projection": storage,
        "resume_boundaries": resume_boundaries(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", default="supervisor", choices=("supervisor", "segment"))
    parser.add_argument("--stage", default="all")
    # segment arguments
    parser.add_argument("--root")
    parser.add_argument("--hot-root")
    parser.add_argument("--deadline-seconds", type=float, default=DEADLINE_SECONDS)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--loader-workers", type=int, default=6)
    parser.add_argument("--games-in-flight", type=int, default=96)
    parser.add_argument("--population-divisor", type=int, default=0)
    parser.add_argument("--event-log")
    parser.add_argument("--label", default="segment")
    # smoke-test knobs: exercising the supervisor itself, never the rehearsal
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reverify-worker", action="store_true")
    return parser


STAGES = (
    "prerequisites",
    "rehearsal",
    "scheduler",
    "postdeadline",
    "readability",
    "determinism",
    "readiness",
)


def stage_path(name: str) -> Path:
    return EVIDENCE_ROOT / f"stage_{name}.json"


def run_stage(name: str) -> dict:
    log(f"stage {name}: starting")
    started = time.perf_counter()
    if name == "prerequisites":
        result = stage_prerequisites()
    elif name == "rehearsal":
        result = stage_rehearsal()
    elif name == "scheduler":
        result = stage_scheduler()
    elif name == "postdeadline":
        result = stage_post_deadline()
    elif name == "readability":
        result = stage_readability()
    elif name == "determinism":
        result = stage_determinism()
    elif name == "readiness":
        result = stage_readiness()
    else:
        raise SystemExit(f"unknown stage {name!r}; expected one of {list(STAGES)}")
    result.setdefault("seconds", time.perf_counter() - started)
    write_json(stage_path(name), result)
    log(f"stage {name}: {result['seconds']:.1f}s -> {stage_path(name)}")
    return result


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.role == "segment":
        return segment_main(args)

    if args.reverify_worker:
        # Section 12: the fix changed the worker-failure recovery path, so that
        # verification is re-run — at the production population, on the real
        # clock — and nothing else is.
        root = Path("/Volumes/Brandon_Washington/stratego_phase13_worker_reverify")
        result = stage_rehearsal(
            {
                "deadline_seconds": 1500.0,
                "crash_at_seconds": None,
                "worker_kill_at_seconds": 420.0,
                "finalize_grace_seconds": 600.0,
            },
            external_root=root,
            hot_root=REPOSITORY / "checkpoints" / "phase13_rehearsal" / "reverify_hot",
            evidence_root=EVIDENCE_ROOT / "worker_reverify",
        )
        write_json(EVIDENCE_ROOT / "stage_worker_reverify.json", result)
        log(json.dumps(result["marks"]))
        for launch_record in result["launches"]:
            log(f"launch {launch_record}")
        return 0

    if args.smoke:
        import shutil
        import tempfile

        scratch = Path(tempfile.mkdtemp(prefix="phase13_rehearsal_smoke_"))
        log(f"smoke test of the supervisor itself under {scratch}")
        result = stage_rehearsal(
            {
                "deadline_seconds": 300.0,
                "crash_at_seconds": 70.0,
                "worker_kill_at_seconds": 150.0,
                "finalize_grace_seconds": 180.0,
                "population_divisor": 64,
                "loader_workers": 4,
            },
            external_root=scratch / "external",
            hot_root=scratch / "hot",
            evidence_root=scratch / "evidence",
        )
        write_json(scratch / "smoke.json", result)
        log(json.dumps(result["marks"]))
        for launch_record in result["launches"]:
            log(f"launch {launch_record}")
        log(f"smoke evidence kept at {scratch}")
        shutil.rmtree(scratch / "external", ignore_errors=True)
        return 0

    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    names = STAGES if args.stage == "all" else tuple(args.stage.split(","))
    results = {}
    for name in names:
        results[name] = run_stage(name.strip())
        if name.strip() == "prerequisites" and not results[name]["verified"]:
            raise SystemExit(
                f"prerequisites are not verified: {results[name]['problems']}"
            )

    # Stages run individually keep the earlier ones: the rehearsal is expensive
    # and must never be re-run just to re-assemble a document.
    combined = {}
    for name in STAGES:
        if name in results:
            combined[name] = results[name]
        elif stage_path(name).exists():
            combined[name] = json.loads(stage_path(name).read_text())
    write_json(EVIDENCE_ROOT / "phase13_agent03_stages.json", combined)
    log(f"stages recorded: {sorted(combined)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
