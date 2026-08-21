"""Phase 14: telemetry and the deliberately narrow control surface.

Specification source: `02_AGENT_2_FINAL_TRAINING_INTEGRATION.md` section 14,
over the frozen `monitoring_without_tuning` block.

Monitoring without tuning
-------------------------
The run is watched closely and steered not at all. Everything on the frozen
metric list is exposed; nothing on the frozen immutable list can be written.
:class:`ControlSurface` is where that asymmetry lives: it offers exactly one
mutation — emergency stop — and :meth:`ControlSurface.set` refuses the frozen
keys by name. A convenient live LR edit is not a feature this run is missing;
it is a thing the control surface exists to make impossible.

Why the metric list is checked
------------------------------
:func:`missing_metrics` compares a snapshot against the frozen list, and the
runner records the result. A metric that quietly stopped being emitted after a
refactor is exactly the kind of gap that only shows up when somebody needs it
at hour 140.

The Agent 4 additions
---------------------
The frozen metric list is part of the contract document and therefore of the
contract digest; it is not touched here. Phase 13 Agent 4's monitoring repairs
appear instead as :data:`EXTENDED_METRIC_PATHS`, checked by
:func:`missing_extended_metrics`, and as two changes in what existing fields
*mean*: ``collection.games_generated`` is now read from the rollout store's
iteration manifests rather than from a process-local counter, and
``workers.status`` is a live health sentence rather than a constant string.
Both were gaps Agent 3 found by running the system for 90 minutes.

The durable emergency stop
--------------------------
:class:`ControlSurface` gained one thing: a stop *file*. An in-process flag
cannot be set from a second terminal and cannot be honoured by a process that
has already been killed. It remains a stop and not a setting — every frozen
training key is still refused by name.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .phase14_contract import (
    FROZEN_METRIC_LIST,
    IMMUTABLE_CONTROL_KEYS,
    PHASE14_NAMESPACE,
)

PHASE14_TELEMETRY_VERSION = "phase14_telemetry_v1"

#: Frozen metric name -> the snapshot path that answers it. Keeping the map
#: explicit is what lets :func:`missing_metrics` be a real check instead of a
#: hopeful one.
METRIC_PATHS = {
    "elapsed wall-clock": ("clock", "elapsed_seconds"),
    "remaining wall-clock": ("clock", "remaining_seconds"),
    "optimizer step": ("training", "global_optimizer_step"),
    "games generated": ("collection", "games_generated"),
    "positions generated": ("collection", "positions_generated"),
    "collection throughput": ("collection", "games_per_second"),
    "learner throughput": ("training", "examples_per_second"),
    "policy loss": ("training", "policy_loss"),
    "value loss": ("training", "value_loss"),
    "belief auxiliary loss": ("training", "belief_loss"),
    "gradient norm": ("training", "grad_norm"),
    "learning rate": ("training", "learning_rate"),
    "advantage-filter acceptance fraction": ("training", "advantage_retention"),
    "draw rate": ("collection", "draw_rate"),
    "game length": ("collection", "mean_game_length"),
    "current/historical opponent mix": ("population", "percentages"),
    "active historical pool": ("population", "active_pool"),
    "archive size": ("population", "archive_k"),
    "checkpoint age": ("checkpoints", "hot_age_seconds"),
    "disk usage": ("storage", "free_gib"),
    "worker health": ("workers", "status"),
    "non-finite counters": ("counters", "non_finite_losses"),
    "candidate evaluation status": ("candidates", "by_status"),
}


#: The Phase 13 Agent 4 monitoring additions. Deliberately a *separate* map:
#: `FROZEN_METRIC_LIST` lives in the contract document and moving it would move
#: the contract digest, which Agent 4 may not do.
EXTENDED_METRIC_PATHS = {
    "committed games (authoritative)": ("collection", "committed_games"),
    "process game counter (diagnostic)": ("collection", "process_counter_games"),
    "configured loader workers": ("workers", "configured_loader_workers"),
    "live loader workers": ("workers", "live_loader_workers"),
    "loader pool rebuilds": ("workers", "loader_pool_rebuilds"),
    "last pool rebuild timestamp": ("workers", "last_pool_rebuild_utc"),
    "last pool rebuild reason": ("workers", "last_pool_rebuild_reason"),
}


class Phase14TelemetryError(RuntimeError):
    """Raised when the control surface is asked for something it may not do."""


def missing_metrics(snapshot: dict) -> list:
    """Every frozen metric the snapshot does not actually carry."""
    missing = []
    for metric, (section, key) in METRIC_PATHS.items():
        if key not in snapshot.get(section, {}):
            missing.append(metric)
    return missing


def missing_extended_metrics(snapshot: dict) -> list:
    """Every Agent 4 monitoring field the snapshot does not actually carry."""
    return [
        metric
        for metric, (section, key) in EXTENDED_METRIC_PATHS.items()
        if key not in snapshot.get(section, {})
    ]


def build_snapshot(
    *,
    clock: dict,
    training: dict,
    collection: dict,
    population: dict,
    checkpoints: dict,
    candidates: dict,
    storage: dict,
    workers: dict,
    counters: dict,
    failures: dict,
) -> dict:
    """One telemetry row covering the whole frozen metric list."""
    snapshot = {
        "telemetry_version": PHASE14_TELEMETRY_VERSION,
        "namespace": PHASE14_NAMESPACE,
        "unix": time.time(),
        "clock": dict(clock),
        "training": dict(training),
        "collection": dict(collection),
        "population": dict(population),
        "checkpoints": dict(checkpoints),
        "candidates": dict(candidates),
        "storage": dict(storage),
        "workers": dict(workers),
        "counters": dict(counters),
        "failures": dict(failures),
    }
    snapshot["missing_metrics"] = missing_metrics(snapshot)
    snapshot["missing_extended_metrics"] = missing_extended_metrics(snapshot)
    return snapshot


@dataclass
class TelemetryLog:
    """An append-only JSONL log of snapshots, on the durable volume."""

    path: Path
    written: int = 0

    @staticmethod
    def at(directory, name: str = "phase14_telemetry.jsonl") -> "TelemetryLog":
        return TelemetryLog(path=Path(directory) / name)

    def write(self, snapshot: dict) -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(snapshot, sort_keys=True, default=str) + "\n")
        self.written += 1
        return snapshot

    def tail(self, count: int = 1) -> list:
        if not self.path.exists():
            return []
        lines = self.path.read_text().strip().splitlines()
        return [json.loads(line) for line in lines[-int(count) :]]


@dataclass
class ControlSurface:
    """The only things an operator may change while Phase 14 runs.

    Emergency stop, and nothing else. Every frozen training value is refused by
    name with an explanation, so an attempt to "just nudge the LR" produces an
    error and a log line rather than a run that no longer matches its manifest.
    """

    stop_requested: bool = False
    stop_reason: str = ""
    stop_unix: "float | None" = None
    refusals: list = field(default_factory=list)
    #: A durable stop request, written by another process. An in-process flag
    #: cannot be set by an operator at 3 a.m. from a second terminal, and a
    #: process that has already been killed cannot be asked to stop politely;
    #: the file is how the request survives both. Frozen keys stay refused —
    #: this adds a stop, not a setting.
    stop_file: "Path | None" = None

    def emergency_stop(self, reason: str = "operator request") -> dict:
        """Request a clean stop at the next safe boundary.

        A *request*, not a kill: the runner finishes the collection unit or the
        optimizer step in flight, writes a hot checkpoint and exits, because a
        torn iteration is a thing the store then has to reconcile.
        """
        self.stop_requested = True
        self.stop_reason = str(reason)
        self.stop_unix = time.time()
        return self.status()

    def clear(self) -> dict:
        self.stop_requested = False
        self.stop_reason = ""
        self.stop_unix = None
        return self.status()

    def file_stop_requested(self) -> bool:
        """Whether an operator has written the durable emergency-stop file."""
        return self.stop_file is not None and Path(self.stop_file).exists()

    def should_continue(self) -> bool:
        if self.file_stop_requested():
            if not self.stop_requested:
                self.emergency_stop(f"emergency-stop file at {self.stop_file}")
            return False
        return not self.stop_requested

    def set(self, key: str, value) -> None:
        """Refuse every frozen training parameter, by name."""
        if key in IMMUTABLE_CONTROL_KEYS:
            self.refusals.append({"key": key, "unix": time.time()})
            raise Phase14TelemetryError(
                f"{key!r} is frozen for the whole Phase 14 run and is not writable "
                "through the control surface; changing it would make the run stop "
                "matching its launch manifest"
            )
        raise Phase14TelemetryError(
            f"the Phase 14 control surface exposes no writable setting {key!r}; "
            "emergency stop is the only control"
        )

    def status(self) -> dict:
        return {
            "stop_requested": bool(self.stop_requested),
            "stop_reason": self.stop_reason,
            "stop_unix": self.stop_unix,
            "immutable_keys": list(IMMUTABLE_CONTROL_KEYS),
            "refusals": len(self.refusals),
            "stop_file": None if self.stop_file is None else str(self.stop_file),
            "stop_file_present": self.file_stop_requested(),
        }


def telemetry_semantics() -> dict:
    return {
        "telemetry_version": PHASE14_TELEMETRY_VERSION,
        "metrics": list(FROZEN_METRIC_LIST),
        "metric_paths": {name: list(path) for name, path in METRIC_PATHS.items()},
        "extended_metrics": list(EXTENDED_METRIC_PATHS),
        "extended_metric_paths": {
            name: list(path) for name, path in EXTENDED_METRIC_PATHS.items()
        },
        "control": "emergency stop only",
        "immutable_keys": list(IMMUTABLE_CONTROL_KEYS),
        "games_generated_source": "rollout store iteration manifests (authoritative)",
        "worker_health_source": "live OS children of the learner",
    }
