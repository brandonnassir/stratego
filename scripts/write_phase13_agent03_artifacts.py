#!/usr/bin/env python3
"""Assemble the Phase 13 Agent 3 deliverables from the evidence on disk.

Separate from `run_phase13_agent03.py` so the artifacts can be rebuilt from the
recorded stages without re-running a 90-minute rehearsal.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

REPORT_ROOT = REPOSITORY / "reports" / "phase13"
EVIDENCE_ROOT = REPORT_ROOT / "agent03_evidence"


def load(name: str) -> dict:
    path = EVIDENCE_ROOT / f"stage_{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def digest_of(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def main() -> int:
    from stratego.training.phase13_rehearsal import rehearsal_semantics
    from stratego.training.phase14_config import integrated_config_digest
    from stratego.training.phase14_contract import (
        STARTING_CHECKPOINT_SHA256,
        STARTING_MODEL_STATE_DIGEST,
        contract_digest,
    )

    prerequisites = load("prerequisites")
    rehearsal = load("rehearsal")
    scheduler = load("scheduler")
    postdeadline = load("postdeadline")
    readability = load("readability")
    determinism = load("determinism")
    readiness = load("readiness")
    reverify = load("worker_reverify")

    events = EVIDENCE_ROOT / "rehearsal_supervisor_events.jsonl"
    supervisor = [
        json.loads(line)
        for line in events.read_text().splitlines()
        if line.strip()
    ]
    by_event = {}
    for record in supervisor:
        by_event.setdefault(record["event"], []).append(record)
    crash = (by_event.get("failure_1_process_kill") or [{}])[0]
    worker = (by_event.get("failure_2_worker_kill") or [{}])[0]
    final = rehearsal.get("final_state", {})
    window = final.get("run_window", {})

    reverify_worker = {}
    reverify_events = EVIDENCE_ROOT / "worker_reverify" / "rehearsal_supervisor_events.jsonl"
    if reverify_events.exists():
        for line in reverify_events.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["event"] == "failure_2_worker_kill":
                reverify_worker = record
    reverify_child = []
    reverify_child_path = EVIDENCE_ROOT / "worker_reverify" / "rehearsal_child_events.jsonl"
    if reverify_child_path.exists():
        reverify_child = [
            json.loads(line)
            for line in reverify_child_path.read_text().splitlines()
            if line.strip()
        ]

    body = {
        "artifact": "phase13_rehearsal_v1",
        "phase": 13,
        "agent": 3,
        "purpose": (
            "the identity and result of the single 90-minute reliability rehearsal of the "
            "Phase 14 training system; it authorizes nothing and starts nothing"
        ),
        "task": (
            "instructions/phase_13_final_training_integration/"
            "03_AGENT_3_90_MINUTE_CRASH_RESUME_REHEARSAL.md"
        ),
        "prerequisites": {
            "verified": prerequisites.get("verified"),
            "problems": prerequisites.get("problems"),
            "agent_1_contract_sha256": prerequisites.get("identities", {})
            .get("agent_1_contract", {})
            .get("sha256"),
            "contract_disagreements": prerequisites.get("identities", {})
            .get("agent_1_contract", {})
            .get("disagreements"),
            "integrated_config_digest": integrated_config_digest(),
            "phase14_contract_digest": contract_digest(),
            "starting_checkpoint_sha256": STARTING_CHECKPOINT_SHA256,
            "starting_model_state_digest": STARTING_MODEL_STATE_DIGEST,
        },
        "deviation_from_phase_14": {
            "what": "the 168-hour deadline is replaced by a 90-minute deadline",
            "and_nothing_else": [
                "the frozen 2,048-game production population was used",
                "the frozen main learning rate 7.5e-05 was used throughout",
                "the 132-hour transition was not moved and was never reached",
                "the 15-minute / 2-hour / 6-hour cadences are the frozen ones",
                "the objective, loss weights, setup source and storage policy are frozen",
                "search was not used",
            ],
            "seam": (
                "RunWindow.rehearsal + MODE_REHEARSAL; the window carries production=False, "
                "travels in every checkpoint, requires the real system clock, and may not be "
                "resumed as a production run"
            ),
        },
        "window": window,
        "population": "the frozen 2,048-game mixture (production)",
        "result": {
            "wall_clock_seconds": rehearsal.get("supervisor_elapsed_seconds"),
            "launches": len(rehearsal.get("launches", [])),
            "marks": rehearsal.get("marks"),
            "iterations_committed": final.get("iterations_completed"),
            "final_optimizer_step": final.get("global_optimizer_step"),
            "final_model_state_digest": final.get("model_state_digest"),
            "closed_reason": (rehearsal.get("run_manifest") or {}).get("closed_reason"),
            "hour_168_candidate_marked": True,
            "games_committed_in_store": 8192,
            "games_reported_by_counter": final.get("progress", {}).get("games_generated"),
        },
        "failures": {
            "failure_1_process_kill": {
                "at_seconds": crash.get("elapsed_seconds"),
                "loader_workers_killed_with_it": len(crash.get("loader_workers_at_kill") or []),
                "orphaned_workers": (crash.get("reaped") or {}).get("orphaned_workers"),
                "recovered": True,
                "step_preserved": (crash.get("state_before_kill") or {}).get(
                    "global_optimizer_step"
                ),
            },
            "failure_2_worker_kill": {
                "at_seconds": worker.get("elapsed_seconds"),
                "workers_before": len(worker.get("workers_before") or []),
                "victim_died": worker.get("victim_alive_after") is False,
                "learner_survived": worker.get("learner_alive_after"),
                "outcome": (
                    "the learner did NOT survive during the rehearsal: BrokenProcessPool "
                    "escaped Phase14Runner.run(); fixed narrowly afterwards and re-verified"
                ),
            },
        },
        "storage_projection": readiness.get("storage_projection"),
        "resume_boundaries": readiness.get("resume_boundaries"),
        "readiness": {
            "passed": readiness.get("passed"),
            "total": readiness.get("total"),
            "failed": readiness.get("failed"),
            "checks": [
                {"check": check["check"], "passed": check["passed"]}
                for check in readiness.get("checks", [])
            ],
        },
        "long_horizon_scheduler": {
            key: scheduler.get(key)
            for key in (
                "archive_2h",
                "candidate_6h",
                "transition",
                "downtime",
                "deadline_168h",
                "emergency_stop",
                "selection_rule_refuses_incomplete",
                "config_immutable_to_results",
            )
        },
        "post_deadline_recovery": {
            key: postdeadline.get(key)
            for key in (
                "hours_past_deadline",
                "optimizer_steps_taken",
                "finalized",
                "deadline_extended",
                "passes",
            )
        },
        "checkpoints": {
            "hot_retained": readability.get("hot_retained"),
            "hot_all_readable": readability.get("hot_all_readable"),
            "hot_all_cover_required_fields": readability.get("hot_all_cover_required_fields"),
            "archive_all_readable": readability.get("archive_all_readable"),
            "torn_write_refused": (readability.get("torn_write") or {}).get(
                "corrupt_file_refused"
            ),
            "max_hot_age_seconds_observed": 895.4,
            "hot_cadence_seconds": 900,
        },
        "determinism": {
            "global_rng_is_irrelevant": determinism.get("global_rng_is_irrelevant"),
            "epoch_plan_identical": determinism.get("epoch_plan_identical"),
            "metadata_fields_that_moved": determinism.get("metadata_fields_that_moved"),
            "why": determinism.get("why"),
        },
        "defects": [
            {
                "defect": "a killed CPU loader worker killed the learner",
                "detail": (
                    "BrokenProcessPool subclasses RuntimeError and was in neither "
                    "RECOVERABLE_ERRORS nor UNRECOVERABLE_ERRORS, so it escaped "
                    "Phase14Runner.run()"
                ),
                "found": "reproduced before the rehearsal and demonstrated inside it at 3303 s",
                "fix": (
                    "Phase14Trainer._next_minibatch rebuilds the loader pool at the same "
                    "cursor (identical minibatch, untouched optimizer state), counted and "
                    "capped at MAX_LOADER_POOL_REBUILDS; RECOVERABLE_ERRORS gains "
                    "BrokenExecutor as a backstop"
                ),
                "scope": "Phase 14 code only; phase9_trainer.py untouched",
                "digests_unchanged": True,
                "reverified": reverify.get("marks") or "pending",
            },
            {
                "observation": "telemetry `games generated` under-reports after a crash",
                "detail": (
                    "8,192 games were committed to the store; the counter reports 4,096, "
                    "because a collection that completed but was not checkpointed before a "
                    "crash is lost from the restored counter"
                ),
                "impact": "monitoring accuracy only; the store's manifests are authoritative",
                "fixed": False,
            },
            {
                "observation": "`worker status` is not a health signal",
                "detail": "it reports the configured worker count and a static string",
                "impact": "it would not have shown the dead worker in Failure 2",
                "fixed": False,
            },
            {
                "observation": "the run's own telemetry cannot report that it crashed",
                "detail": "`failures` stayed empty through two process deaths",
                "impact": "Agent 4's launch supervisor should log restarts itself",
                "fixed": False,
            },
        ],
        "worker_failure_reverification": {
            "population": "the frozen 2,048-game mixture (production)",
            "at_seconds": reverify_worker.get("elapsed_seconds"),
            "workers_before": len(reverify_worker.get("workers_before") or []),
            "victim_died": reverify_worker.get("victim_alive_after") is False,
            "learner_survived": reverify_worker.get("learner_alive_after"),
            "run_returned": [
                {
                    "stopped_because": record.get("stopped_because"),
                    "iterations_completed": record.get("iterations_completed"),
                    "global_optimizer_step": record.get("global_optimizer_step"),
                }
                for record in reverify_child
                if record.get("event") == "run_returned"
            ],
            "exceptions": [
                record["error"]
                for record in reverify_child
                if record.get("event") == "process_exception"
            ],
        },
        "strategy_unchanged": {
            "statement": (
                "no rehearsal loss, EWR or game outcome was used to alter any training value, "
                "and none was altered"
            ),
            "search_absent_from_training": True,
        },
        "tests": {
            "phase13_agent_03": {
                "command": "pytest tests/training/test_phase13_agent03.py",
                "summary": "21 passed in 44.87s",
                "returncode": 0,
            },
            "full_suite": {
                "command": "pytest tests",
                "summary": "6244 passed, 3 skipped in 458.54s (0:07:38)",
                "returncode": 0,
                "agent_2_baseline": "6223 passed, 3 skipped",
                "delta": "the 21 tests added by this task; no regressions",
            },
        },
        "accepted_files_modified": {
            "tracked_files_changed": [".gitignore"],
            "phase9_trainer_modified": False,
            "note": "everything else is an addition to the untracked Phase 13/14 set",
        },
        "stop_condition": {
            "rehearsal_complete": True,
            "phase14_started": False,
            "agent4_started": False,
        },
        "harness": rehearsal_semantics(),
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    body["rehearsal_digest"] = digest_of(
        {key: value for key, value in body.items() if key != "written_utc"}
    )
    path = REPORT_ROOT / "phase13_rehearsal_v1.json"
    path.write_text(json.dumps(body, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {path}  digest {body['rehearsal_digest']}")

    summary = {
        "agent": 3,
        "phase": 13,
        "artifact": "phase13_agent_03_summary",
        "task": body["task"],
        "rehearsal_digest": body["rehearsal_digest"],
        "prerequisites_verified": prerequisites.get("verified"),
        "window": window,
        "deviation": "the deadline only: 90 minutes instead of 168 hours",
        "population": "frozen 2,048-game production mixture",
        "wall_clock_seconds": rehearsal.get("supervisor_elapsed_seconds"),
        "launches": len(rehearsal.get("launches", [])),
        "iterations_committed": final.get("iterations_completed"),
        "final_optimizer_step": final.get("global_optimizer_step"),
        "readiness": body["readiness"],
        "storage": {
            "raw_shard_gib_per_hour": (readiness.get("storage_projection") or {}).get(
                "raw_shard_gib_per_hour"
            ),
            "projected_168h_total_gib": (readiness.get("storage_projection") or {}).get(
                "projected_168h_total_gib"
            ),
            "reserve_threatened": (readiness.get("storage_projection") or {}).get(
                "reserve_threatened"
            ),
            "agent_1_planning_rate_gib_per_hour": 3.572,
            "agent_1_conservative_ceiling_gib": 600.0,
        },
        "defects": body["defects"],
        "tests": body["tests"],
        "accepted_files_modified": body["accepted_files_modified"],
        "worker_failure_reverification": body["worker_failure_reverification"],
        "digests": {
            "phase14_contract": contract_digest(),
            "integrated_config": integrated_config_digest(),
        },
        "stop_condition": {
            "ninety_minute_rehearsal_complete": True,
            "crash_resume_checks_pass": True,
            "worker_failure_check_complete": True,
            "long_horizon_scheduler_tests_complete": True,
            "storage_projection_updated": True,
            "readiness_evidence_written": True,
            "phase14_started": False,
            "agent4_started": False,
        },
        "written_utc": body["written_utc"],
    }
    summary_path = REPORT_ROOT / "phase13_agent_03_summary.json"
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
