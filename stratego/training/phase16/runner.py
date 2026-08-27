"""Phase 16 Agent 3: the arm runner.

Specification source: `03_AGENT_3_TRAINING_LOOP_V2.md` sections 3 and 4.

One arm is one process
----------------------
Load P24, build the arm's setup source and opponent mixture, then alternate
collect-window / train-window until the arm's clock runs out. At h = 0, 2, 4
and 6 the current weights are exported in the accepted evaluation format --
EMA where the arm has one, raw where it does not -- and recorded. **Scoring is
not done here.** A pack run inside the training process would take cores away
from training and make the 6 hours mean something different for each arm, so
the runner exports and the `evaluate` role scores.

Why the behavior identity is stable
-----------------------------------
Phase 9 and 14 rebuilt the behavior snapshot every iteration because an
iteration's games were played start-to-finish by one snapshot. A window
collector continues the *same* games after an update, so a game's decisions
legitimately come from several sets of weights. PPO does not mind -- the ratio's
denominator is the per-decision stored probability, and every row carries its
own -- but the acting-token check does, so the logical identity of the
learner's snapshot is the constant `CURRENT` and the weights behind it rotate.
Each window's telemetry records the state-dict digest that actually played it,
so the provenance a stable token would otherwise lose is written down instead.

The clock
---------
Elapsed wall-time against the arm's own start, checked between windows. A
window is never interrupted mid-update: the last window may push a few minutes
past the mark, and the telemetry records where it actually stopped rather than
where it was aimed.
"""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .checkpoint import (
    build_payload,
    export_evaluation_weights,
    load_starting_model,
    read as read_checkpoint,
    restore,
    save as save_checkpoint_payload,
)
from .collector import WindowCollector, collector_semantics
from .contract import (
    ARM_HOURS,
    EVALUATION_HOURS,
    PHASE16_RUNNER_VERSION,
    ArmConfig,
    Phase16TrainingError,
    contract_digest,
    game_id as phase16_game_id,
)
from .contract import OPPONENTS_PHASE14_MIXTURE
from .population import HistoricalPool, population_semantics, realized_shares
from .schedules import schedule_semantics
from .setups import assert_orientation_path, build_setup_source, setup_semantics
from .snapshots import bind_anchor, participants_for, snapshot_from_model
from .targets import targets_semantics
from .trainer import Phase16TrainerError, WindowTrainer, trainer_semantics

#: The stable logical identity of the learner's collection snapshot.
CURRENT_IDENTITY = "CURRENT"
#: The pool anchor: the weights the arm started from.
ANCHOR_IDENTITY = "P24"

DEFAULT_STORAGE_ROOT = "checkpoints/phase16/arms"
DEFAULT_TELEMETRY_ROOT = "reports/phase16/agent03"

#: How often the resumable hot checkpoint is rewritten, in seconds.
HOT_CHECKPOINT_SECONDS = 600.0

#: How often the arm's own weights are added to its historical pool.
#:
#: Only `phase14_mixture` arms draw historical opponents at all, and without a
#: cadence their "history" would be the frozen start for all six hours -- which
#: is not the Phase 14 recipe the control arm exists to reproduce. Phase 14
#: archived every five iterations over a 168-hour run; a wall-clock cadence is
#: the closer analogue here, because a Phase 16 iteration is roughly twenty
#: times cheaper and five of them is a couple of minutes.
ARCHIVE_CADENCE_SECONDS = 1800.0


class Phase16RunnerError(Phase16TrainingError):
    """A Phase 16 arm could not be run as specified."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


@dataclass
class ArmClock:
    """Elapsed wall-time against the arm's own start, across restarts."""

    hours: float = ARM_HOURS
    started_utc: str = field(default_factory=utc_now)
    accumulated_seconds: float = 0.0
    _origin: float = field(default_factory=time.monotonic, repr=False)

    @property
    def elapsed_seconds(self) -> float:
        return self.accumulated_seconds + (time.monotonic() - self._origin)

    @property
    def elapsed_hours(self) -> float:
        return self.elapsed_seconds / 3600.0

    @property
    def expired(self) -> bool:
        return self.elapsed_hours >= float(self.hours)

    def to_dict(self) -> dict:
        return {
            "hours": float(self.hours),
            "started_utc": self.started_utc,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "elapsed_hours": round(self.elapsed_hours, 5),
            "expired": bool(self.expired),
        }

    @classmethod
    def resume(cls, payload: dict, *, hours: float) -> "ArmClock":
        return cls(
            hours=float(hours),
            started_utc=str(payload.get("started_utc", utc_now())),
            accumulated_seconds=float(payload.get("elapsed_seconds", 0.0)),
        )


class ArmRunner:
    """One shootout arm, start to deadline."""

    def __init__(
        self,
        config: ArmConfig,
        *,
        root: "str | Path" = ".",
        storage_root: "str | Path | None" = None,
        telemetry_root: "str | Path | None" = None,
        hours: float = ARM_HOURS,
        device: "str | None" = None,
        collection_device: "str | None" = None,
    ) -> None:
        self.config = config
        self.root = Path(root)
        self.storage = Path(storage_root or (self.root / DEFAULT_STORAGE_ROOT)) / config.arm_id
        self.telemetry_root = Path(telemetry_root or (self.root / DEFAULT_TELEMETRY_ROOT))
        self.hours = float(hours)
        self.device = str(device or config.device)
        self.collection_device = str(collection_device or config.collection_device)
        self.storage.mkdir(parents=True, exist_ok=True)
        self.telemetry_root.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.telemetry_root / f"{config.arm_id}_windows.jsonl"
        self.checkpoint_path = self.storage / "hot.pt"
        self.clock = ArmClock(hours=self.hours)
        self.model = None
        self.trainer = None
        self.collector = None
        self.setup_source = None
        self.pool = None
        self.evaluations: list = []
        self.stopped: "dict | None" = None
        self._historical: dict = {}
        self._last_hot = 0.0
        self._last_archive = 0.0
        self.archived: list = []

    # -- construction ------------------------------------------------------

    def build(self, *, resume: bool = True) -> dict:
        """Load the start, assemble the arm, and resume if a checkpoint exists."""
        self.model = load_starting_model(device=self.device, root=self.root)
        self.trainer = WindowTrainer(self.config, self.model, device=self.device)
        self.setup_source = build_setup_source(self.config.setups, root=self.root)
        orientation = assert_orientation_path(
            self.setup_source, phase16_game_id(self.config.arm_id, 0, 0)
        )
        self.pool = HistoricalPool(anchor_identity=ANCHOR_IDENTITY)
        historical = bind_anchor(
            self.model,
            identity=ANCHOR_IDENTITY,
            device=self.collection_device,
            inference_batch_shape=self.config.inference_batch_shape,
        )
        self._historical = historical
        participants = participants_for(
            self.model,
            identity=CURRENT_IDENTITY,
            device=self.collection_device,
            historical=historical,
            inference_batch_shape=self.config.inference_batch_shape,
        )
        self.collector = WindowCollector(
            self.config,
            participants,
            setup_source=self.setup_source,
            pool=self.pool,
        )
        resumed = None
        if resume and self.checkpoint_path.is_file():
            resumed = self._resume()
        return {
            "arm": self.config.arm_id,
            "arm_digest": self.config.digest(),
            "device": self.device,
            "collection_device": self.collection_device,
            "orientation": orientation,
            "resumed": resumed,
            "storage": str(self.storage),
            "telemetry": str(self.telemetry_path),
        }

    def _resume(self) -> dict:
        payload = read_checkpoint(self.checkpoint_path)
        report = restore(
            payload,
            config=self.config,
            model=self.model,
            optimizer=self.trainer.optimizer,
            ema=self.trainer.ema,
        )
        self.trainer.restore_state(payload["trainer_state"])
        collector_state = payload.get("collector_state") or {}
        self.collector.iteration = int(collector_state.get("iteration", 0))
        self.collector.draw_counts = list(
            collector_state.get("draw_counts", self.collector.draw_counts)
        )
        self.collector.games_completed = int(collector_state.get("games_completed", 0))
        self.collector.decisions_collected = int(
            collector_state.get("decisions_collected", 0)
        )
        self.clock = ArmClock.resume(payload.get("clock") or {}, hours=self.hours)
        self.evaluations = list((payload.get("diagnostics") or {}).get("evaluations", []))
        self._rotate_behavior()
        report["resumed_at_iteration"] = self.collector.iteration
        report["note"] = (
            "in-flight games are not checkpointed; a resumed arm restarts its "
            "population from fresh draws and its window numbering continues"
        )
        return report

    # -- the loop ----------------------------------------------------------

    def _maybe_archive(self) -> "dict | None":
        """Add the current weights to the arm's own historical pool, on cadence.

        A no-op for `pure_current` arms: they never draw a historical opponent,
        so an archive would be weights nothing can play against.
        """
        if self.config.opponents != OPPONENTS_PHASE14_MIXTURE:
            return None
        now = time.monotonic()
        if self._last_archive and now - self._last_archive < ARCHIVE_CADENCE_SECONDS:
            return None
        self._last_archive = now
        identity = f"W{self.collector.iteration:04d}"
        if identity in self.pool.members():
            return None
        snapshot = snapshot_from_model(
            self.model,
            identity=identity,
            device=self.collection_device,
            inference_batch_shape=self.config.inference_batch_shape,
            provenance=f"the arm's own weights at window {self.collector.iteration}",
        )
        self._historical[identity] = snapshot
        self.pool.add(identity)
        entry = {
            "identity": identity,
            "iteration": int(self.collector.iteration),
            "elapsed_hours": round(self.clock.elapsed_hours, 5),
            "state_digest": snapshot.checkpoint_sha256,
            "pool_size": len(self.pool.members()),
        }
        self.archived.append(entry)
        return entry

    def _rotate_behavior(self) -> dict:
        """Point the collector at the current weights under the stable identity."""
        participants = participants_for(
            self.model,
            identity=CURRENT_IDENTITY,
            device=self.collection_device,
            historical=self._historical,
            inference_batch_shape=self.config.inference_batch_shape,
        )
        return self.collector.rebind(participants)

    def run(self, *, max_windows: "int | None" = None, progress=None) -> dict:
        """Alternate collect / train until the arm's clock expires."""
        if self.collector is None:
            raise Phase16RunnerError("call build() before run()")
        started = utc_now()
        windows: list = []
        self._maybe_evaluate(force_hour=0)

        while not self.clock.expired:
            if max_windows is not None and len(windows) >= int(max_windows):
                break
            window = self.collector.collect_window(
                should_continue=lambda: not self.clock.expired
            )
            if not window.rows:
                if window.stopped_early:
                    break
                raise Phase16RunnerError(
                    f"window {window.iteration} finished no games; the population "
                    "cannot make progress"
                )
            try:
                update = self.trainer.train_window(window.rows, iteration=window.iteration)
            except Phase16TrainerError as error:
                # An accepted hard veto (KL or clip fraction). The limit and its
                # semantics are unchanged -- it fires, training stops, the
                # counter is already incremented. What changes here is only how
                # the *process* ends: an unhandled raise would kill the arm and
                # take the rest of the shootout with it, leaving no h-curve at
                # all. Stopping cleanly keeps every export up to this point and
                # records the breach as a result rather than a crash.
                stopped = {
                    "reason": "hard_veto",
                    "error": str(error),
                    "iteration": int(window.iteration),
                    "elapsed_hours": round(self.clock.elapsed_hours, 5),
                    "kl_beta": float(self.trainer.controller.beta),
                    "counters": dict(self.trainer.counters),
                }
                self.stopped = stopped
                window.rows.clear()
                self._append_telemetry(
                    {
                        "runner_version": PHASE16_RUNNER_VERSION,
                        "arm": self.config.arm_id,
                        "iteration": int(window.iteration),
                        "utc": utc_now(),
                        "elapsed_hours": round(self.clock.elapsed_hours, 5),
                        "stopped": stopped,
                    }
                )
                break
            row = self._telemetry_row(window, update)
            windows.append(row)
            self._append_telemetry(row)
            if progress is not None:
                progress(row)
            # The rows are the window's only large allocation; drop them before
            # the next collection so two windows are never resident at once.
            window.rows.clear()
            self._maybe_archive()
            self._rotate_behavior()
            self._maybe_evaluate()
            self._maybe_hot_checkpoint()

        # A vetoed arm still exports where it got to, so its partial h-curve is
        # scoreable and the reader can see how far it ran before it stopped.
        self._maybe_evaluate(force_hour=int(round(self.hours)))
        self._write_hot_checkpoint()
        return {
            "stopped": self.stopped,
            "runner_version": PHASE16_RUNNER_VERSION,
            "arm": self.config.arm_id,
            "arm_digest": self.config.digest(),
            "started_utc": started,
            "finished_utc": utc_now(),
            "clock": self.clock.to_dict(),
            "windows": len(windows),
            "optimizer_steps": int(self.trainer.global_step),
            "examples_consumed": int(self.trainer.examples_consumed),
            "games_completed": int(self.collector.games_completed),
            "decisions_collected": int(self.collector.decisions_collected),
            "archived_snapshots": list(self.archived),
            "evaluations": list(self.evaluations),
            "telemetry": str(self.telemetry_path),
            "counters": dict(self.trainer.counters),
        }

    # -- telemetry ---------------------------------------------------------

    def _telemetry_row(self, window, update) -> dict:
        mixture = realized_shares(window.draws) if window.draws else {}
        collection = window.summary()
        optimization = update.summary()
        return {
            "runner_version": PHASE16_RUNNER_VERSION,
            "arm": self.config.arm_id,
            "iteration": int(window.iteration),
            "utc": utc_now(),
            "elapsed_hours": round(self.clock.elapsed_hours, 5),
            "behavior_state_digest": self.collector.participants.behavior.checkpoint_sha256,
            "collection": collection,
            "optimization": optimization,
            "mixture": mixture,
            "move_entropy": optimization.get("mean_policy_entropy"),
            "entropy_normalized": optimization.get("mean_entropy_normalized"),
            "behavior_kl": optimization.get("mean_behavior_kl"),
            "clip_fraction": optimization.get("mean_clip_fraction"),
            "iteration_wall_seconds": round(
                collection["seconds"] + optimization["seconds"], 3
            ),
            "collection_seconds": collection["seconds"],
            "training_seconds": optimization["seconds"],
        }

    def _append_telemetry(self, row: dict) -> None:
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.telemetry_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    # -- checkpoints and exports -------------------------------------------

    def _payload(self) -> dict:
        return build_payload(
            config=self.config,
            model=self.model,
            optimizer=self.trainer.optimizer,
            ema=self.trainer.ema,
            trainer_state=self.trainer.trainer_state(),
            collector_state=self.collector.state(),
            clock=self.clock.to_dict(),
            diagnostics={
                "evaluations": list(self.evaluations),
                "archived_snapshots": list(self.archived),
                "contract_digest": contract_digest(),
                "host": platform.node(),
                "pid": os.getpid(),
            },
        )

    def _write_hot_checkpoint(self) -> dict:
        written = save_checkpoint_payload(self._payload(), self.checkpoint_path)
        self._last_hot = time.monotonic()
        return written

    def _maybe_hot_checkpoint(self) -> None:
        if time.monotonic() - self._last_hot >= HOT_CHECKPOINT_SECONDS:
            self._write_hot_checkpoint()

    def _maybe_evaluate(self, *, force_hour: "int | None" = None) -> "dict | None":
        """Export the arm's weights at the section-4 hours. Never scores here."""
        done = {int(entry["hour"]) for entry in self.evaluations}
        if force_hour is not None:
            hour = int(force_hour)
            if hour in done:
                return None
        else:
            due = [
                mark
                for mark in EVALUATION_HOURS
                if mark not in done and self.clock.elapsed_hours >= mark
            ]
            if not due:
                return None
            hour = max(due)
            # More than one mark due means a window spanned an evaluation
            # boundary. Exporting the latest and leaving the earlier ones open
            # would export them next window from *later* weights and label them
            # with the earlier hour, which is the one way an h-curve can lie.
            # They are closed as skipped instead, with no weights attached.
            for passed in due:
                if passed == hour:
                    continue
                self.evaluations.append(
                    {
                        "arm": self.config.arm_id,
                        "hour": int(passed),
                        "utc": utc_now(),
                        "elapsed_hours": round(self.clock.elapsed_hours, 5),
                        "skipped": True,
                        "reason": (
                            f"one window spanned the h={passed} mark; the export "
                            f"at h={hour} is the first after it"
                        ),
                    }
                )
        return self._export(hour)

    def _export(self, hour: int) -> dict:
        use_ema = bool(self.config.ema)
        weights = self.storage / f"hour_{hour:02d}.pt"
        payload = self._payload()
        exported = export_evaluation_weights(payload, weights, use_ema=use_ema)
        entry = {
            "arm": self.config.arm_id,
            "hour": int(hour),
            "utc": utc_now(),
            "elapsed_hours": round(self.clock.elapsed_hours, 5),
            "iteration": int(self.collector.iteration),
            "optimizer_step": int(self.trainer.global_step),
            "weights_path": str(weights),
            "weights_sha256": exported["export_sha256"],
            "model_state_digest": exported["model_state_digest"],
            "source": exported["source"],
            "ema_updates": int(self.trainer.ema.updates) if self.trainer.ema else 0,
        }
        self.evaluations.append(entry)
        identity = self.storage / f"hour_{hour:02d}_identity.json"
        identity.write_text(json.dumps(entry, indent=1, sort_keys=True) + "\n")
        return entry

    # -- documents ---------------------------------------------------------

    def run_config(self) -> dict:
        """Every input that decides what this arm *is*, for the report."""
        return {
            "artifact": "phase16_arm_run_config_v1",
            "runner_version": PHASE16_RUNNER_VERSION,
            "contract_digest": contract_digest(),
            "arm": self.config.to_dict(),
            "arm_digest": self.config.digest(),
            "hours": self.hours,
            "device": self.device,
            "collection_device": self.collection_device,
            "schedules": schedule_semantics(self.config),
            "population": population_semantics(self.config),
            "setups": setup_semantics(self.setup_source) if self.setup_source else {},
            "collector": collector_semantics(),
            "targets": targets_semantics(),
            "trainer": trainer_semantics(),
            "evaluation_hours": list(EVALUATION_HOURS),
            "evaluation_note": (
                "the runner exports weights at each hour; scoring runs outside "
                "the training process so every arm's six hours are six hours of "
                "training"
            ),
        }


# ---------------------------------------------------------------------------
# Telemetry reading
# ---------------------------------------------------------------------------


def read_telemetry(path: "str | Path") -> list:
    rows = []
    handle = Path(path)
    if not handle.is_file():
        return rows
    for line in handle.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def telemetry_summary(rows) -> dict:
    """Per-iteration diagnostics condensed into the numbers section 4 asks for."""
    if not rows:
        return {}

    def series(key, path):
        values = []
        for row in rows:
            value = row
            for part in path:
                value = (value or {}).get(part) if isinstance(value, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        return values

    wall = [float(row.get("iteration_wall_seconds", 0.0)) for row in rows]
    collect = [float(row.get("collection_seconds", 0.0)) for row in rows]
    train = [float(row.get("training_seconds", 0.0)) for row in rows]
    # Sum of (learning rate x optimizer steps) over the run. Three arms given
    # equal wall-clock do not receive equal total step size -- the schedules and
    # the epoch count both move it -- and a conclusion about "damping" that did
    # not report this could not be told apart from a conclusion about "less
    # training".
    step_size = 0.0
    steps = 0
    row_counts = [
        float((row.get("collection") or {}).get("rows", 0))
        for row in rows
        if (row.get("collection") or {}).get("rows")
    ]
    for row in rows:
        optimization = row.get("optimization") or {}
        rate = optimization.get("learning_rate")
        count = optimization.get("optimizer_steps")
        if isinstance(rate, (int, float)) and isinstance(count, (int, float)):
            step_size += float(rate) * float(count)
            steps += int(count)
    entropy = series("entropy", ("optimization", "mean_policy_entropy"))
    kl = series("kl", ("optimization", "mean_behavior_kl"))
    clip = series("clip", ("optimization", "mean_clip_fraction"))
    retention = series("retention", ("optimization", "advantage_statistics", "retention_fraction"))
    lengths = series("length", ("collection", "game_length", "mean"))
    plies = series("plies", ("collection", "plies_per_second"))
    return {
        "iterations": len(rows),
        "iteration_wall_seconds": {
            "mean": float(np.mean(wall)) if wall else 0.0,
            "p50": float(np.percentile(wall, 50)) if wall else 0.0,
            "p90": float(np.percentile(wall, 90)) if wall else 0.0,
            "max": float(np.max(wall)) if wall else 0.0,
            "coefficient_of_variation": (
                float(np.std(wall) / np.mean(wall)) if wall and np.mean(wall) else 0.0
            ),
        },
        "move_entropy": {"first": entropy[0], "last": entropy[-1]} if entropy else {},
        "behavior_kl": {"mean": float(np.mean(kl)), "max": float(np.max(kl))} if kl else {},
        "clip_fraction": {"mean": float(np.mean(clip)), "max": float(np.max(clip))}
        if clip
        else {},
        "advantage_retention": {"mean": float(np.mean(retention))} if retention else {},
        "optimizer_steps": steps,
        "summed_step_size": step_size,
        # The window budget pins *data*, and therefore training time. Whether it
        # also pins collection time depends on the opponent mixture, so the two
        # halves are reported separately rather than only as their sum.
        "phase_seconds": {
            name: {
                "first10": float(np.mean(values[:10])) if values else 0.0,
                "last10": float(np.mean(values[-10:])) if values else 0.0,
                "coefficient_of_variation": (
                    float(np.std(values) / np.mean(values))
                    if values and np.mean(values)
                    else 0.0
                ),
            }
            for name, values in (("collection", collect), ("training", train))
        },
        "rows_per_iteration": {
            "coefficient_of_variation": (
                float(np.std(row_counts) / np.mean(row_counts))
                if row_counts and np.mean(row_counts)
                else 0.0
            )
        },
        "game_length_mean": {"first": lengths[0], "last": lengths[-1]} if lengths else {},
        "plies_per_second": {"mean": float(np.mean(plies))} if plies else {},
        "elapsed_hours": rows[-1].get("elapsed_hours"),
    }


__all__ = [
    "ANCHOR_IDENTITY",
    "ArmClock",
    "ArmRunner",
    "CURRENT_IDENTITY",
    "DEFAULT_STORAGE_ROOT",
    "DEFAULT_TELEMETRY_ROOT",
    "HOT_CHECKPOINT_SECONDS",
    "Phase16RunnerError",
    "read_telemetry",
    "telemetry_summary",
    "utc_now",
]
