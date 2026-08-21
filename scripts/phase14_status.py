#!/usr/bin/env python3
"""Phase 14: the read-only operator status.

Assembled entirely from what the run writes to disk — hot checkpoints, the
telemetry log, the rollout store's iteration manifests, the candidate ledger,
the supervisor log — so it works whether or not the learner is alive, and so
reading it cannot compete with, or perturb, the training it reports on.

The two numbers Phase 13 Agent 3 found misleading are labelled here:

```text
committed games   authoritative — the rollout store's iteration manifests
process counter   diagnostic    — process-local, and low after any crash
```

Usage:

```text
python scripts/phase14_status.py            # human-readable
python scripts/phase14_status.py --json     # the whole document
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


def build_status(storage) -> dict:
    from stratego.evaluation.phase14_candidates import CandidateLedger
    from stratego.training.phase14_launch import (
        emergency_stop_state,
        integrity_failure_state,
    )
    from stratego.training.phase14_status import (
        committed_game_census,
        learner_process_state,
        loader_health,
        utc_text,
    )
    from stratego.training.phase14_storage import volume_usage
    from stratego.training.phase14_supervisor import (
        deadline_state,
        resume_checkpoint_state,
        run_manifest_state,
        unevaluated_candidates,
    )
    from stratego.training.phase14_telemetry import TelemetryLog

    checkpoint = resume_checkpoint_state(storage.hot_root)
    manifest = run_manifest_state(storage.run_state_path)
    window = checkpoint.get("run_window") or manifest.get("window") or {}
    telemetry = TelemetryLog.at(storage.log_root).tail(1)
    row = telemetry[0] if telemetry else {}
    census = committed_game_census(storage.rollout_root)
    learner = learner_process_state(storage.log_root)
    persisted_workers = row.get("workers", {})
    live_workers = (
        loader_health(
            pid=learner["learner_pid"],
            configured_workers=persisted_workers.get("configured_loader_workers"),
            pool_open=False,
            rebuilds=persisted_workers.get("loader_pool_rebuilds", 0),
            last_rebuild_unix=persisted_workers.get("last_pool_rebuild_unix"),
            last_rebuild_reason=persisted_workers.get("last_pool_rebuild_reason", ""),
            max_rebuilds=persisted_workers.get("max_loader_pool_rebuilds"),
        )
        if learner["alive"]
        else {"live_loader_workers": 0, "learner_alive": False}
    )
    live_workers["learner_alive"] = learner["alive"]
    supervisor_log = Path(storage.log_root) / "phase14_supervisor.jsonl"
    supervisor_events = []
    if supervisor_log.exists():
        for line in supervisor_log.read_text().splitlines()[-40:]:
            line = line.strip()
            if line:
                try:
                    supervisor_events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return {
        "artifact": "phase14_operator_status_v1",
        "read_utc": utc_text(),
        "window": window,
        "deadline": deadline_state(window),
        "closed": manifest.get("closed", False),
        "close_reason": manifest.get("close_reason", ""),
        "checkpoint": checkpoint,
        "games": {
            "committed_games": census["committed_games"],
            "committed_games_label": "authoritative (rollout store iteration manifests)",
            "in_flight_games": census["in_flight_games"],
            "process_counter_games": row.get("collection", {}).get("process_counter_games"),
            "process_counter_label": "diagnostic (process-local; low after a crash)",
            "sealed_iterations": census["sealed_iterations"],
        },
        "training": row.get("training", {}),
        "population": row.get("population", {}),
        "learner": learner,
        # Probed *now*, against the live learner, rather than copied from a
        # telemetry row written between iterations when the pool is legitimately
        # closed. The rebuild counters come from the row, because they are
        # cumulative facts the learner persists.
        "workers": {
            **row.get("workers", {}),
            **live_workers,
        },
        "checkpoints": row.get("checkpoints", {}),
        "counters": row.get("counters", {}),
        "failures": row.get("failures", {}),
        "storage": volume_usage(storage.external_root),
        "candidates": {
            **CandidateLedger.at(storage.evaluation_root).status_summary(),
            "unevaluated_hours": unevaluated_candidates(storage.evaluation_root),
        },
        "controls": {
            "emergency_stop": emergency_stop_state(storage.external_root),
            "integrity_failure": integrity_failure_state(storage.external_root),
        },
        "supervisor": {
            "log": str(supervisor_log),
            "recent_events": [
                {key: event.get(key) for key in ("utc", "event", "reason", "learner_pid")}
                for event in supervisor_events
            ],
        },
        "telemetry_row_missing_metrics": row.get("missing_metrics"),
        "telemetry_row_missing_extended_metrics": row.get("missing_extended_metrics"),
    }


def render(status: dict) -> str:
    deadline = status["deadline"]
    workers = status["workers"]
    training = status["training"]
    lines = [
        f"Phase 14 status at {status['read_utc']}",
        f"  window       {status['window'].get('run_start_utc')} -> {status['window'].get('run_deadline_utc')}",
        f"  remaining    {deadline.get('remaining_seconds', 0) / 3600.0:.2f} h"
        f"   past deadline: {deadline.get('passed')}",
        f"  closed       {status['closed']}  {status['close_reason']}",
        f"  step         {training.get('global_optimizer_step')}"
        f"   LR {training.get('learning_rate')}   segment {training.get('segment')}",
        f"  losses       policy {training.get('policy_loss')}"
        f"  value {training.get('value_loss')}  belief {training.get('belief_loss')}",
        "",
        f"  committed games   {status['games']['committed_games']}   AUTHORITATIVE",
        f"  in flight         {status['games']['in_flight_games']}",
        f"  process counter   {status['games']['process_counter_games']}   diagnostic only",
        "",
        f"  learner      pid {status['learner'].get('learner_pid')}"
        f"  alive {status['learner'].get('alive')}"
        f"  launched {status['learner'].get('launch_timestamp')}",
        f"  workers      {workers.get('status')}",
        f"               configured {workers.get('configured_loader_workers')}"
        f"  live {workers.get('live_loader_workers')}"
        f"  rebuilds {workers.get('loader_pool_rebuilds')}"
        f"/{workers.get('max_loader_pool_rebuilds')}",
        f"               last rebuild {workers.get('last_pool_rebuild_utc')}"
        f"  {workers.get('last_pool_rebuild_reason')}",
        "",
        f"  checkpoint   {status['checkpoint'].get('path')}",
        f"               age {status['checkpoints'].get('hot_age_seconds')} s"
        f"  (read as up to one 900 s cadence plus a collection)",
        f"  disk         {status['storage'].get('free_gib')} GiB free",
        f"  candidates   {status['candidates'].get('by_status')}"
        f"  unevaluated {status['candidates'].get('unevaluated_hours')}",
        f"  emergency    stop active: {status['controls']['emergency_stop']['active']}"
        f"   integrity failure: {status['controls']['integrity_failure']['recorded']}",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    from stratego.training.phase14_storage import Phase14Storage

    parser = argparse.ArgumentParser(description="Phase 14 operator status (read-only)")
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--hot-root", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    storage = (
        Phase14Storage.production()
        if args.external_root is None and args.hot_root is None
        else Phase14Storage.under(args.external_root, hot_root=args.hot_root)
    )
    status = build_status(storage)
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True, default=str))
    else:
        print(render(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
