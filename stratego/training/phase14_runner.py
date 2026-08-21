"""Phase 14: the run loop that ties the frozen contract together.

Specification source: `02_AGENT_2_FINAL_TRAINING_INTEGRATION.md` sections 3,
11, 12, 15 and 16.

The whole graph, in one object
------------------------------
Starting checkpoint -> learner -> Phase 14 setup source -> population self-play
(current | bounded historical pool | five handcrafted behaviours) -> accepted
trajectory reconstruction -> policy/value/belief updates -> hot checkpoints,
durable archive, candidates, evaluator, telemetry, deadline. Search appears
nowhere in that list and is imported nowhere on this path.

One bulk-synchronous unit
-------------------------
`collect 2,048 games -> bind targets -> 2 epochs -> committed boundary`. The
unit is the granularity of every long-horizon decision: the segment is fixed
when a unit is *launched*, the archive and candidate marks are taken at
committed boundaries, and the deadline stops new units rather than interrupting
one.

Restart is the normal case, not the exception
---------------------------------------------
A 168-hour run on one machine will be interrupted. :meth:`Phase14Runner.resume`
loads the newest *valid* hot checkpoint, restores the optimizer, controller,
counters, cursor, pool and shard cursor, reuses the ORIGINAL start and deadline,
re-derives the pool from the archive and refuses to continue if it differs from
what the checkpoint recorded. What it never does is start a fresh logical run:
every path that would produce a new 168-hour window is a refusal.
"""

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import BrokenExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .phase14_checkpoint import (
    SNAPSHOT_ROLE_ARCHIVE,
    SNAPSHOT_ROLE_BEHAVIOR,
    SNAPSHOT_ROLE_HOT,
    HotCheckpointRing,
    Phase14CheckpointError,
    Phase14SnapshotResolver,
    behavior_snapshot_path,
    build_payload,
    mark_candidate,
    read_candidate_marks,
    write_archive_snapshot,
    write_behavior_snapshot,
)
from .phase14_clock import (
    DeadlineController,
    Phase14ClockError,
    SystemClock,
    require_production_clock,
    utc_text,
)
from .phase14_collector import collect_iteration, resolve_pool_participants
from .phase14_contract import (
    ANCHOR_SHA256,
    CANDIDATE_HOURS,
    GAMES_PER_ITERATION,
    PHASE14_NAMESPACE,
    PRODUCTION_POPULATION,
    Population,
    STARTING_CHECKPOINT,
    STARTING_MODEL_STATE_DIGEST,
    Phase14ContractError,
    assert_matches_frozen_contract,
    contract_digest,
    file_sha256,
    repository_root,
)
from .phase14_pool import (
    ActivePool,
    HistoricalArchive,
    Phase14PoolError,
    assert_pool_matches,
)
from .phase14_schedule import (
    behavior_policy_token,
    behavior_snapshot_identity,
    iteration_mixture,
    population_digest,
)
from .phase14_seed import seed_contract_digest
from .phase14_setup_source import Phase14SetupSource, assert_orientation_path
from .phase14_storage import Phase14Storage, Phase14StorageError
from .phase14_telemetry import ControlSurface, TelemetryLog, build_snapshot
from .phase14_trainer import (
    Phase14Trainer,
    Phase14TrainerError,
    bind_sealed_rollout,
    load_starting_model,
)

PHASE14_RUNNER_VERSION = "phase14_runner_v1"

MODE_PRODUCTION = "production"
MODE_TEST = "test"
#: The Phase 13 Agent 3 reliability rehearsal: the *real* wall clock and the
#: real recovery path, over a deliberately shortened window. It is a declared
#: seam, not a production mode — production refuses a shortened window exactly
#: as it refuses a manual clock and a scaled population.
MODE_REHEARSAL = "rehearsal"
MODES = (MODE_PRODUCTION, MODE_TEST, MODE_REHEARSAL)

#: Failures the run is expected to survive: a worker dying, a transient MPS
#: fault, a torn shard write, the machine rebooting. They cost the affected
#: unit's un-committed games and nothing else.
#: `BrokenExecutor` is here because a killed CPU loader worker raises
#: `BrokenProcessPool`, which is a `RuntimeError` and was caught by neither
#: list — so a single dead worker ended the run. The trainer now rebuilds its
#: pool in place; this is the backstop for a pool that breaks somewhere the
#: rebuild does not cover, and it costs the unit rather than the run.
RECOVERABLE_ERRORS = (OSError, TimeoutError, BrokenExecutor)

#: Failures that mean the run is no longer the run it claims to be. These stop
#: it: continuing would produce artifacts nobody could interpret afterwards.
UNRECOVERABLE_ERRORS = (
    Phase14CheckpointError,
    Phase14ContractError,
    Phase14PoolError,
    Phase14TrainerError,
    Phase14ClockError,
)


class Phase14RunnerError(RuntimeError):
    """Raised when the Phase 14 run may not start, continue or resume."""


class Phase14IntegrityError(Phase14RunnerError):
    """An unrecoverable integrity failure. The run stops; it never restarts fresh."""


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------


@dataclass
class RunProgress:
    """The bookkeeping a checkpoint carries besides model and optimizer state."""

    iteration: int = 0
    last_hot_index: int = -1
    last_archive_mark: int = 0
    last_candidate_index: int = -1
    games_generated: int = 0
    positions_generated: int = 0
    decisions_generated: int = 0
    draws: int = 0
    collection_seconds: float = 0.0
    iterations_completed: int = 0
    failures: dict = field(default_factory=dict)
    closed: bool = False
    close_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "iteration": int(self.iteration),
            "last_hot_index": int(self.last_hot_index),
            "last_archive_mark": int(self.last_archive_mark),
            "last_candidate_index": int(self.last_candidate_index),
            "games_generated": int(self.games_generated),
            "positions_generated": int(self.positions_generated),
            "decisions_generated": int(self.decisions_generated),
            "draws": int(self.draws),
            "collection_seconds": float(self.collection_seconds),
            "iterations_completed": int(self.iterations_completed),
            "failures": dict(self.failures),
            "closed": bool(self.closed),
            "close_reason": self.close_reason,
        }

    @staticmethod
    def from_dict(payload: dict) -> "RunProgress":
        progress = RunProgress()
        for key, value in (payload or {}).items():
            if hasattr(progress, key):
                setattr(progress, key, value)
        return progress

    def record_failure(self, kind: str) -> int:
        self.failures[kind] = int(self.failures.get(kind, 0)) + 1
        return self.failures[kind]


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


class Phase14Runner:
    """One Phase 14 final-training run, from start (or resume) to deadline."""

    def __init__(
        self,
        storage: Phase14Storage,
        *,
        clock=None,
        mode: str = MODE_PRODUCTION,
        device: str = "mps",
        inference_device: "str | None" = None,
        topology=None,
        games_in_flight: int = 96,
        inference_batch_shape: int = 64,
        control: "ControlSurface | None" = None,
        evaluate_inline: bool = False,
        evaluation_limit: "int | None" = None,
        setup_source=None,
        population: Population = PRODUCTION_POPULATION,
        rehearsal_deadline_seconds: "float | None" = None,
    ) -> None:
        if mode not in MODES:
            raise Phase14RunnerError(f"unknown run mode {mode!r}; expected one of {list(MODES)}")
        self.mode = mode
        if mode == MODE_REHEARSAL and rehearsal_deadline_seconds is None:
            raise Phase14RunnerError(
                "a rehearsal run must declare its own deadline; "
                "pass rehearsal_deadline_seconds"
            )
        if mode != MODE_REHEARSAL and rehearsal_deadline_seconds is not None:
            raise Phase14RunnerError(
                f"a {mode!r} run may not carry a rehearsal deadline; the shortened "
                "window is available only in rehearsal mode"
            )
        self.rehearsal_deadline_seconds = (
            None if rehearsal_deadline_seconds is None else float(rehearsal_deadline_seconds)
        )
        if mode == MODE_PRODUCTION and not population.production:
            # Same rule as the clock: a scaled population is a test seam, and
            # production refuses it rather than being talked into it.
            raise Phase14RunnerError(
                "a scaled population is a test-only seam; production Phase 14 runs "
                "the frozen 2,048-game mixture"
            )
        self.population = population
        self.storage = storage
        self.clock = clock or SystemClock()
        if mode in (MODE_PRODUCTION, MODE_REHEARSAL):
            # The test scheduler seam is not something production can be
            # talked into using; it is something production rejects. A
            # rehearsal rejects it too: the whole point of a 90-minute
            # rehearsal is that the 90 minutes are real, and downtime inside
            # them is real downtime charged against a real deadline.
            require_production_clock(self.clock)
        self.device = device
        self.inference_device = inference_device or device
        self.inference_batch_shape = int(inference_batch_shape)
        self.topology = topology
        self.games_in_flight = int(games_in_flight)
        self.control = control or ControlSurface()
        # Candidate evaluation is out-of-band by default: 128 games inside the
        # training loop would spend deadline time on monitoring. The flag exists
        # so an integration test can prove the in-loop path runs at all.
        self.evaluate_inline = bool(evaluate_inline)
        self.evaluation_limit = evaluation_limit
        self.setup_source = setup_source

        self.controller: "DeadlineController | None" = None
        self.trainer: "Phase14Trainer | None" = None
        self.archive = HistoricalArchive()
        self.pool: "ActivePool | None" = None
        self.progress = RunProgress()
        self.hot = HotCheckpointRing(self.storage.hot_root)
        self.behavior_root = Path(self.storage.hot_root) / "behavior"
        self.telemetry = TelemetryLog.at(self.storage.log_root)
        self.last_hot_unix: "float | None" = None
        self.last_metrics: dict = {}
        self.started_from: str = ""

    # -- inputs ------------------------------------------------------------

    def verify_inputs(self) -> dict:
        """Prove every frozen input is present and is what it claims to be.

        Run before the window is stamped, because "the contract on disk is not
        the contract in the code" and "the starting checkpoint is not the
        accepted one" are things to discover before a 168-hour clock starts,
        not after.
        """
        from ..evaluation.phase14_candidates import load_pack, load_selection_rule

        report = {"contract": assert_matches_frozen_contract()}
        starting = repository_root() / STARTING_CHECKPOINT
        report["starting_checkpoint"] = {
            "path": str(starting),
            "sha256": file_sha256(starting),
        }
        for name, relative in ANCHOR_SHA256.items():
            from .phase14_contract import ANCHOR_CHECKPOINTS

            path = repository_root() / ANCHOR_CHECKPOINTS[name]
            observed = file_sha256(path)
            if observed != relative:
                raise Phase14IntegrityError(
                    f"pool anchor {name} at {path} has SHA-256 {observed}, not the "
                    f"frozen {relative}"
                )
        report["anchors_verified"] = list(ANCHOR_SHA256)
        pack = load_pack()
        report["pack"] = {
            "digest": pack["pack_content_digest"],
            "games": len(pack["games"]),
        }
        report["selection_rule"] = load_selection_rule()["artifact"]
        report["storage"] = self.storage.prepare()
        if report["storage"]["problems"]:
            raise Phase14StorageError(
                f"storage is not ready: {report['storage']['problems']}"
            )
        source = self._source()
        report["setup_source"] = source.describe()
        from .phase14_seed import game_id as phase14_game_id

        report["orientation_probe"] = assert_orientation_path(
            source, phase14_game_id(1, "current", 0)
        )
        report["population_digest"] = population_digest()
        report["seed_contract_digest"] = seed_contract_digest()
        report["contract_digest"] = contract_digest()
        return report

    def _source(self):
        if self.setup_source is None:
            self.setup_source = Phase14SetupSource.build()
        return self.setup_source

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> dict:
        """Stamp the window and begin a new logical run.

        `run_start_utc` is taken here — immediately before the loop — rather
        than at process start, so time spent verifying inputs and loading the
        model does not come out of the 168 hours.
        """
        if self.hot.latest_valid() is not None:
            raise Phase14RunnerError(
                f"{self.storage.hot_root} already holds a valid hot checkpoint; a new "
                "logical run may not be started over an existing one — resume it"
            )
        verification = self.verify_inputs()
        model = load_starting_model(device=self.device)
        self.trainer = Phase14Trainer(model, device=self.device, topology=self.topology)
        if self.trainer.model_state_digest != STARTING_MODEL_STATE_DIGEST:
            raise Phase14IntegrityError(
                "the loaded model is not the accepted Phase 9 starting checkpoint"
            )
        self.controller = (
            DeadlineController.rehearsal(self.clock, self.rehearsal_deadline_seconds)
            if self.mode == MODE_REHEARSAL
            else DeadlineController.start(self.clock)
        )
        self.archive = HistoricalArchive()
        self.pool = ActivePool.for_archive(self.archive)
        self.progress = RunProgress()
        self.started_from = "start"
        self._mark_hour_zero_candidate()
        written = self.hot_checkpoint(force=True)
        return {
            "started": True,
            "run_start_utc": utc_text(self.controller.window.run_start_utc),
            "run_deadline_utc": utc_text(self.controller.window.run_deadline_utc),
            "transition_utc": utc_text(self.controller.window.transition_utc),
            "hot_checkpoint": written,
            "verification": verification,
        }

    def _assert_window_kind(self, path) -> None:
        """Refuse a resume whose persisted window is the wrong kind of window.

        Without this, a rehearsal checkpoint left in the production hot
        directory would be resumed as "the" Phase 14 run — 90 minutes in and
        88 minutes from a deadline nobody intended — and a production
        checkpoint could be picked up by a rehearsal and quietly finalized.
        Neither is recoverable after the fact, so both are refused here.
        """
        assert self.controller is not None
        window = self.controller.window
        if self.mode == MODE_PRODUCTION and not window.production:
            raise Phase14IntegrityError(
                f"{path}: the persisted window is a rehearsal window "
                f"({window.deadline_seconds:.0f}s) and this is a production run; "
                "Phase 14 does not resume across the rehearsal seam"
            )
        if self.mode == MODE_REHEARSAL:
            if window.production:
                raise Phase14IntegrityError(
                    f"{path}: the persisted window is the full production window and "
                    "this is a rehearsal; a rehearsal may not adopt the real run"
                )
            declared = float(self.rehearsal_deadline_seconds)
            if abs(window.deadline_seconds - declared) > 1e-6:
                raise Phase14IntegrityError(
                    f"{path}: the persisted rehearsal window spans "
                    f"{window.deadline_seconds:.0f}s but this process declared "
                    f"{declared:.0f}s; a restart reuses the original deadline and "
                    "never negotiates a new one"
                )
        # MODE_TEST reads either kind: the post-deadline recovery check has to
        # be able to load a real rehearsal checkpoint under a manual clock.

    def resume(self) -> dict:
        """Continue the existing run from the newest valid hot checkpoint."""
        loaded = self.hot.load_latest()
        if loaded is None:
            raise Phase14IntegrityError(
                f"no valid hot checkpoint under {self.storage.hot_root}; Phase 14 does "
                "not silently start a fresh logical run"
            )
        path, payload = loaded
        if payload["upstream"]["phase14_contract_digest"] != contract_digest():
            raise Phase14IntegrityError(
                f"{path}: configuration digest mismatch — the checkpoint was written "
                f"under {payload['upstream']['phase14_contract_digest']}, this build is "
                f"{contract_digest()}"
            )
        from .phase14_checkpoint import model_from_payload

        model = model_from_payload(payload, device=self.device)
        self.trainer = Phase14Trainer(model, device=self.device, topology=self.topology)
        try:
            self.trainer.restore_state(payload)
        except (KeyError, ValueError, RuntimeError) as error:
            raise Phase14IntegrityError(
                f"{path}: the optimizer state could not be restored: {error}"
            ) from error
        self.controller = DeadlineController.resume(payload["run_window"], self.clock)
        self._assert_window_kind(path)
        self.archive = HistoricalArchive.from_dict(payload["historical_archive_state"])
        # f(k) is recomputed and compared rather than trusted: a pool that
        # silently differed would train against a different opponent
        # distribution than the checkpoint claims.
        self.pool = assert_pool_matches(self.archive, payload["active_historical_pool"])
        self.progress = RunProgress.from_dict(payload["schedule_state"].get("progress", {}))
        self.started_from = str(path)
        expired = self.controller.expired()
        return {
            "resumed": True,
            "checkpoint": str(path),
            "run_start_utc": payload["run_window"]["run_start_utc"],
            "run_deadline_utc": payload["run_window"]["run_deadline_utc"],
            "elapsed_hours": self.controller.elapsed_hours(),
            "remaining_hours": self.controller.remaining() / 3600.0,
            "iteration": self.progress.iteration,
            "global_optimizer_step": self.trainer.global_step,
            "archive_k": self.archive.k,
            "active_pool": self.pool.members(),
            "past_deadline": expired,
        }

    def start_or_resume(self) -> dict:
        if self.hot.latest_valid() is not None:
            return self.resume()
        return self.start()

    # -- checkpoints -------------------------------------------------------

    def _payload(self, role: str) -> dict:
        assert self.trainer is not None and self.controller is not None
        elapsed = self.controller.elapsed()
        return build_payload(
            model=self.trainer.model,
            optimizer=self.trainer.optimizer,
            snapshot_role=role,
            trainer_state=self.trainer.trainer_state(),
            run_window=self.controller.window.to_dict(),
            schedule_state={
                "segment": self.controller.segment(),
                "elapsed_seconds": elapsed,
                "elapsed_hours": elapsed / 3600.0,
                "remaining_seconds": self.controller.remaining(),
                "transition_utc": utc_text(self.controller.window.transition_utc),
                "learning_rate": self.controller.learning_rate(),
                "progress": self.progress.to_dict(),
                "runner_version": PHASE14_RUNNER_VERSION,
                "mode": self.mode,
                "clock": self.clock.describe(),
            },
            population_schedule_state={
                "population_digest": population_digest(),
                "games_per_iteration": GAMES_PER_ITERATION,
                "population": self.population.to_dict(),
                "iteration": self.progress.iteration,
                "segment": self.controller.segment(),
            },
            active_historical_pool=self.pool.to_dict(),
            historical_archive_state=self.archive.to_dict(),
            shard_cursor={
                "namespace": PHASE14_NAMESPACE,
                "rollout_root": str(self.storage.rollout_root),
                "last_committed_iteration": self.progress.iteration,
                "iteration_directory": str(
                    self.storage.iteration_directory(max(self.progress.iteration, 1))
                ),
            },
            storage_state=self.storage.storage_state(),
            candidate_evaluation_state=self._candidate_state(),
            device=self.device,
            diagnostics={"last_metrics": self.last_metrics},
        )

    def hot_checkpoint(self, *, force: bool = False) -> "dict | None":
        """Write a hot checkpoint if the 15-minute mark has passed."""
        assert self.controller is not None
        index = self.controller.hot_index_due()
        if not force and index <= self.progress.last_hot_index:
            return None
        written = self.hot.write(self._payload(SNAPSHOT_ROLE_HOT))
        self.progress.last_hot_index = index
        self.last_hot_unix = time.time()
        return written

    def _candidate_state(self) -> dict:
        from ..evaluation.phase14_candidates import CandidateLedger

        ledger = CandidateLedger.at(self.storage.evaluation_root)
        return {
            "ledger": ledger.status_summary(),
            "marks": len(read_candidate_marks(self.storage.archive_root)),
            "last_candidate_index": self.progress.last_candidate_index,
            "candidate_hours": list(CANDIDATE_HOURS),
        }

    def _mark_hour_zero_candidate(self) -> dict:
        """Hour 0 is the accepted starting checkpoint, marked in place."""
        from ..evaluation.phase14_candidates import CandidateLedger

        path = repository_root() / STARTING_CHECKPOINT
        record = mark_candidate(
            self.storage.archive_root,
            hour=0,
            snapshot_path=path,
            snapshot_sha256=file_sha256(path),
            model_state_digest=STARTING_MODEL_STATE_DIGEST,
            elapsed_seconds=0.0,
            written_utc=utc_text(self.controller.now()),
            iteration=0,
            global_optimizer_step=0,
        )
        CandidateLedger.at(self.storage.evaluation_root).record_candidate(0, record)
        self.progress.last_candidate_index = 0
        return record

    def maybe_archive(self) -> "dict | None":
        """Take a durable snapshot if a 2-hour mark has been crossed."""
        assert self.controller is not None and self.trainer is not None
        mark = self.controller.archive_index_due()
        if mark <= self.progress.last_archive_mark:
            return None
        position = self.archive.k + 1
        elapsed = self.controller.elapsed()
        # Several 2-hour marks can pass while the machine is down or while one
        # long unit is in flight. Those marks name no distinct weights, so they
        # coalesce into this snapshot and the fact is recorded rather than
        # inferred from a gap in the numbering.
        coalesced = list(range(self.progress.last_archive_mark + 1, mark))
        written = write_archive_snapshot(
            self.storage.archive_root, self._payload(SNAPSHOT_ROLE_ARCHIVE), position=position
        )
        entry = self.archive.append(
            archive_mark=mark,
            path=written["path"],
            sha256=written["sha256"],
            model_state_digest=written["model_state_digest"],
            elapsed_seconds=elapsed,
            written_utc=utc_text(self.controller.now()),
            iteration=self.progress.iteration,
            global_optimizer_step=self.trainer.global_step,
        )
        self.progress.last_archive_mark = mark
        # The pool is a pure function of the archive, so it is recomputed the
        # moment the archive changes and never edited in place.
        self.pool = ActivePool.for_archive(self.archive)
        return {
            "entry": entry.to_dict(),
            "written": written,
            "pool": self.pool.members(),
            "coalesced_marks": coalesced,
        }

    def maybe_candidate(self, archived: "dict | None") -> "dict | None":
        """Mark the newest archive snapshot as a candidate at a 6-hour mark."""
        from ..evaluation.phase14_candidates import CandidateLedger

        assert self.controller is not None
        index = self.controller.candidate_index_due()
        if index <= self.progress.last_candidate_index:
            return None
        if archived is None:
            if not self.archive.entries:
                return None
            entry = self.archive.entries[-1]
        else:
            entry = self.archive.entries[-1]
        hour = CANDIDATE_HOURS[index]
        skipped = [
            CANDIDATE_HOURS[position]
            for position in range(self.progress.last_candidate_index + 1, index)
        ]
        record = mark_candidate(
            self.storage.archive_root,
            hour=hour,
            snapshot_path=entry.path,
            snapshot_sha256=entry.sha256,
            model_state_digest=entry.model_state_digest,
            elapsed_seconds=self.controller.elapsed(),
            written_utc=utc_text(self.controller.now()),
            iteration=self.progress.iteration,
            global_optimizer_step=self.trainer.global_step,
            archive_position=entry.position,
        )
        record["coalesced_hours"] = skipped
        CandidateLedger.at(self.storage.evaluation_root).record_candidate(hour, record)
        self.progress.last_candidate_index = index
        return record

    def after_committed_iteration(self, iteration: int) -> dict:
        """Mark the committed iteration's shards disposable and watch the reserve.

        Marking is not deleting: full raw retention is the plan, and the mark
        only records that these shards have been consumed by every epoch that
        will ever read them. Deletion happens solely under the pre-authorized
        contingency, and only for shards carrying this mark.
        """
        from .phase14_storage import (
            execute_rolling_deletion,
            mark_shards_disposable,
            plan_rolling_deletion,
        )

        record: dict = {"iteration": int(iteration)}
        directory = self.storage.iteration_directory(iteration)
        if directory.exists():
            record["disposable_mark"] = mark_shards_disposable(
                directory, iteration=iteration, reason="iteration COMMITTED"
            )
        status = self.storage.reserve_status()
        record["reserve"] = {
            "free_gib": status["free_gib"],
            "breached": status["reserve_breached"],
        }
        if status["reserve_breached"]:
            plan = plan_rolling_deletion(
                self.storage, keep_iterations_after=max(iteration - 1, 0)
            )
            record["rolling_deletion"] = execute_rolling_deletion(self.storage, plan)
            self.progress.record_failure("storage_reserve_breach")
        return record

    # -- candidate evaluation ---------------------------------------------

    def evaluate_pending_candidates(self, *, limit_games: "int | None" = None) -> list:
        """Evaluate marked candidates that have no complete result yet.

        Deliberately decoupled: every failure is caught, recorded and dropped
        here, because an evaluation problem is not a training problem and must
        never stop the run.
        """
        from ..evaluation.phase14_candidates import CandidateLedger, evaluate_candidate

        ledger = CandidateLedger.at(self.storage.evaluation_root)
        anchor = repository_root() / STARTING_CHECKPOINT
        results: list = []
        for entry in ledger.pending():
            mark = entry.get("mark") or {}
            snapshot = mark.get("snapshot_path")
            if not snapshot or not Path(snapshot).exists():
                ledger.record_failure(entry["hour"], f"candidate bytes missing: {snapshot}")
                results.append({"hour": entry["hour"], "status": "failed"})
                continue
            try:
                weights = self._evaluation_weights(entry["hour"], snapshot)
                result = evaluate_candidate(
                    weights,
                    anchor_weights=self._evaluation_weights("anchor", anchor),
                    device=self.inference_device,
                    limit=limit_games if limit_games is not None else self.evaluation_limit,
                )
                ledger.record_result(entry["hour"], result)
                results.append(
                    {
                        "hour": entry["hour"],
                        "status": "complete",
                        "mean_ewr": result["mean_ewr"],
                        "games": result["games_played"],
                    }
                )
            except Exception as error:  # noqa: BLE001 - evaluation never stops training
                ledger.record_failure(entry["hour"], f"{type(error).__name__}: {error}")
                self.progress.record_failure("candidate_evaluation")
                results.append({"hour": entry["hour"], "status": "failed"})
        return results

    def _evaluation_weights(self, tag, source) -> Path:
        """The accepted-format export of one candidate's weights, cached."""
        from .phase14_checkpoint import export_evaluation_weights

        directory = self.storage.evaluation_root / "weights"
        directory.mkdir(parents=True, exist_ok=True)
        export = directory / f"{tag}.pt"
        if not export.exists():
            export_evaluation_weights(source, export)
        return export

    # -- the loop ----------------------------------------------------------

    def _behavior_snapshot(self, iteration: int):
        """Freeze the learner's weights for one iteration and bind them."""
        identity = behavior_snapshot_identity(iteration)
        self.behavior_root.mkdir(parents=True, exist_ok=True)
        path = behavior_snapshot_path(self.behavior_root, identity)
        if not path.exists():
            write_behavior_snapshot(
                self.behavior_root, self._payload(SNAPSHOT_ROLE_BEHAVIOR), identity=identity
            )
        resolver = Phase14SnapshotResolver(
            device=self.inference_device, inference_batch_shape=self.inference_batch_shape
        )
        snapshot = resolver.bind(
            path,
            logical_identity=identity,
            policy_token=behavior_policy_token(iteration),
            expected_sha256=file_sha256(path),
        )
        return snapshot, resolver

    def _next_iteration(self) -> tuple:
        """`(iteration, state)` of the next unit, skipping committed ones.

        A crash between "iteration N is COMMITTED" and the checkpoint that
        records it leaves the store ahead of the checkpoint by one unit. Those
        optimizer updates are gone with the weights that made them, and PPO may
        not re-consume N's rollout with drifted weights, so the honest recovery
        is to accept the loss and collect N+1 — not to replay N against a policy
        that no longer collected it.
        """
        from .phase9_rollout_store import read_iteration_state

        iteration = self.progress.iteration + 1
        while True:
            state = read_iteration_state(
                self.storage.rollout_root, PHASE14_NAMESPACE, iteration
            )
            if state is not None and state["state"] == "COMMITTED":
                self.progress.iteration = iteration
                iteration += 1
                continue
            return iteration, state

    def _resume_training_only(self, iteration: int, state: dict, *, updates) -> dict:
        """Finish a unit whose games are already sealed on disk.

        The crash-during-training path. The rollout is bound with `resuming`,
        the *checkpointed cursor* decides where the epochs continue, and the
        on-policy digest check is deliberately not re-applied: the weights have
        moved since collection precisely because some of this iteration's
        updates already landed.
        """
        assert self.controller is not None and self.trainer is not None
        segment = state.get("segment") or self.controller.segment()
        rollout = bind_sealed_rollout(
            self.storage.rollout_root,
            iteration,
            segment=segment,
            population=self.population,
            resuming=True,
        )
        cursor = self.trainer.cursor
        reusable = (
            cursor is not None
            and cursor.iteration == iteration
            and cursor.sealed_rollout_digest == rollout.sealed_rollout_digest
            and not cursor.finished
        )
        if reusable:
            self.trainer.resume_iteration(rollout, cursor)
        else:
            self.trainer.bind_iteration(rollout)
        rows = self.trainer.train_iteration(
            updates=updates,
            may_start_step=self.controller.may_start_optimizer_step,
            on_step=self._on_step,
        )
        completed = self.trainer.cursor.finished
        if completed:
            self.trainer.mark_iteration_trained()
            self.progress.iteration = iteration
            self.progress.iterations_completed += 1
        self.trainer.close()
        archived = self.maybe_archive()
        candidate = self.maybe_candidate(archived)
        retention = self.after_committed_iteration(iteration) if completed else None
        written = self.hot_checkpoint(force=True)
        telemetry = self.emit_telemetry(rows=rows)
        return {
            "launched": True,
            "resumed_training_only": True,
            "resumed_from_cursor": bool(reusable),
            "iteration": iteration,
            "segment": segment,
            "sealed": True,
            "trained": completed,
            "updates": len(rows),
            "collection": {"games_collected": 0, "already_sealed": True},
            "archived": archived,
            "candidate": candidate,
            "evaluations": [],
            "retention": retention,
            "hot_checkpoint": written,
            "telemetry": telemetry,
            "seconds": 0.0,
        }

    def run_iteration(self, *, collection_limit: "int | None" = None, updates: "int | None" = None) -> dict:
        """One complete bulk-synchronous unit, from launch to committed."""
        assert self.controller is not None and self.trainer is not None
        if not self.controller.may_start_collection_unit():
            return {"launched": False, "reason": "deadline"}
        if not self.control.should_continue():
            return {"launched": False, "reason": "emergency_stop"}

        iteration, state = self._next_iteration()
        if state is not None and state["state"] in ("SEALED", "TRAINING"):
            return self._resume_training_only(iteration, state, updates=updates)

        segment = state.get("segment") if state else None
        segment = segment or self.controller.segment()
        pool = self.pool
        started = self.clock.monotonic()

        snapshot, resolver = self._behavior_snapshot(iteration)
        if snapshot.loaded_state_dict_digest != self.trainer.model_state_digest:
            raise Phase14IntegrityError(
                f"iteration {iteration}: the behavior snapshot is not the learner's "
                "current weights; PPO would consume another policy's rollout"
            )
        participants = resolve_pool_participants(
            pool,
            behavior=snapshot,
            device=self.inference_device,
            inference_batch_shape=self.inference_batch_shape,
            resolver=resolver,
        )

        collection = collect_iteration(
            self.storage.rollout_root,
            iteration,
            participants,
            setup_source=self._source(),
            segment=segment,
            pool=pool,
            population=self.population,
            games_in_flight=self.games_in_flight,
            limit=collection_limit,
            should_continue=self.control.should_continue,
        )
        self.progress.collection_seconds += float(collection["seconds"])
        self.progress.games_generated += int(collection["games_collected"])
        self.progress.decisions_generated += int(collection["total_decisions"])
        self.progress.positions_generated += int(collection["total_decisions"])
        self.progress.draws += int(collection["terminal_results"].get("draw", 0))
        self.hot_checkpoint()

        if not collection.get("sealed"):
            return {
                "launched": True,
                "iteration": iteration,
                "segment": segment,
                "sealed": False,
                "collection": collection,
                "reason": "collection incomplete",
            }

        rollout = bind_sealed_rollout(
            self.storage.rollout_root,
            iteration,
            segment=segment,
            population=self.population,
            behavior_snapshot=snapshot,
            expected_model_state_digest=self.trainer.model_state_digest,
        )
        self.trainer.bind_iteration(rollout)
        rows = self.trainer.train_iteration(
            updates=updates,
            may_start_step=self.controller.may_start_optimizer_step,
            on_step=self._on_step,
        )
        completed = self.trainer.cursor.finished
        if completed:
            self.trainer.mark_iteration_trained()
            self.progress.iteration = iteration
            self.progress.iterations_completed += 1
        self.trainer.close()

        archived = self.maybe_archive()
        candidate = self.maybe_candidate(archived)
        retention = self.after_committed_iteration(iteration) if completed else None
        evaluations = (
            self.evaluate_pending_candidates()
            if (candidate is not None and self.evaluate_inline)
            else []
        )
        written = self.hot_checkpoint(force=True)
        telemetry = self.emit_telemetry(collection=collection, rows=rows)
        return {
            "launched": True,
            "iteration": iteration,
            "segment": segment,
            "sealed": True,
            "trained": completed,
            "updates": len(rows),
            "collection": collection,
            "archived": archived,
            "candidate": candidate,
            "evaluations": evaluations,
            "retention": retention,
            "hot_checkpoint": written,
            "telemetry": telemetry,
            "seconds": self.clock.monotonic() - started,
        }

    def _on_step(self, trainer, row) -> None:
        self.last_metrics = {
            key: row.get(key)
            for key in (
                "loss_total",
                "loss_ppo",
                "loss_value",
                "loss_belief",
                "behavior_kl",
                "policy_entropy",
                "clip_fraction",
                "grad_norm_post_clip",
                "advantage_retention",
                "learning_rate",
                "kl_beta",
                "global_optimizer_step",
            )
        }
        self.hot_checkpoint()

    def run(
        self,
        *,
        max_iterations: "int | None" = None,
        collection_limit: "int | None" = None,
        updates: "int | None" = None,
        max_consecutive_failures: int = 3,
    ) -> dict:
        """Run units until the deadline, an emergency stop or a limit.

        Recoverable failures cost the affected unit and are retried; the run
        stops on the frozen unrecoverable list rather than trying to carry on
        with a run whose identity is in doubt.
        """
        if self.controller is None or self.trainer is None:
            raise Phase14RunnerError("call start() or resume() before run()")
        units: list = []
        consecutive = 0
        while True:
            if max_iterations is not None and len(units) >= int(max_iterations):
                reason = "iteration limit"
                break
            if self.controller.expired():
                reason = "deadline"
                break
            if not self.control.should_continue():
                reason = "emergency stop"
                break
            try:
                unit = self.run_iteration(collection_limit=collection_limit, updates=updates)
                consecutive = 0
            except UNRECOVERABLE_ERRORS as error:
                self.progress.record_failure("unrecoverable")
                self.progress.close_reason = f"{type(error).__name__}: {error}"
                raise Phase14IntegrityError(
                    f"unrecoverable integrity failure: {type(error).__name__}: {error}"
                ) from error
            except RECOVERABLE_ERRORS as error:
                consecutive += 1
                self.progress.record_failure(type(error).__name__)
                units.append(
                    {
                        "launched": False,
                        "reason": "recoverable failure",
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(limit=3),
                    }
                )
                self.hot_checkpoint(force=True)
                if consecutive >= int(max_consecutive_failures):
                    reason = "consecutive recoverable failures"
                    break
                continue
            units.append(unit)
            if not unit.get("launched"):
                reason = unit.get("reason", "not launched")
                break
        return {
            "units": units,
            "stopped_because": reason,
            "iterations_completed": self.progress.iterations_completed,
            "global_optimizer_step": self.trainer.global_step,
            "elapsed_hours": self.controller.elapsed_hours(),
            "remaining_hours": self.controller.remaining() / 3600.0,
        }

    # -- finalization ------------------------------------------------------

    def finalize(self, reason: str = "deadline") -> dict:
        """Close the run: final state, hour-168 candidate, manifest, closed.

        Called at or after the deadline, and also by a recovery that starts
        after it — in which case no optimizer step runs at all and the run goes
        straight to this.
        """
        assert self.controller is not None and self.trainer is not None
        from ..evaluation.phase14_candidates import CandidateLedger

        elapsed = self.controller.elapsed()
        final_position = self.archive.k + 1
        written = write_archive_snapshot(
            self.storage.archive_root,
            self._payload(SNAPSHOT_ROLE_ARCHIVE),
            position=final_position,
        )
        entry = self.archive.append(
            archive_mark=self.controller.archive_index_due(),
            path=written["path"],
            sha256=written["sha256"],
            model_state_digest=written["model_state_digest"],
            elapsed_seconds=elapsed,
            written_utc=utc_text(self.controller.now()),
            iteration=self.progress.iteration,
            global_optimizer_step=self.trainer.global_step,
        )
        self.pool = ActivePool.for_archive(self.archive)
        final_hour = CANDIDATE_HOURS[-1]
        candidate = mark_candidate(
            self.storage.archive_root,
            hour=final_hour,
            snapshot_path=entry.path,
            snapshot_sha256=entry.sha256,
            model_state_digest=entry.model_state_digest,
            elapsed_seconds=elapsed,
            written_utc=utc_text(self.controller.now()),
            iteration=self.progress.iteration,
            global_optimizer_step=self.trainer.global_step,
            archive_position=entry.position,
        )
        CandidateLedger.at(self.storage.evaluation_root).record_candidate(final_hour, candidate)
        self.progress.last_candidate_index = len(CANDIDATE_HOURS) - 1
        self.progress.closed = True
        self.progress.close_reason = str(reason)
        hot = self.hot_checkpoint(force=True)
        manifest = self.run_manifest(reason=reason)
        return {
            "closed": True,
            "reason": reason,
            "final_archive_entry": entry.to_dict(),
            "hour_168_candidate": candidate,
            "hot_checkpoint": hot,
            "manifest": manifest,
        }

    def run_manifest(self, *, reason: str = "") -> dict:
        assert self.controller is not None and self.trainer is not None
        manifest = {
            "artifact": "phase14_run_manifest_v1",
            "runner_version": PHASE14_RUNNER_VERSION,
            "namespace": PHASE14_NAMESPACE,
            "mode": self.mode,
            "clock": self.clock.describe(),
            "population": self.population.to_dict(),
            "contract_digest": contract_digest(),
            "population_digest": population_digest(),
            "seed_contract_digest": seed_contract_digest(),
            "window": self.controller.window.to_dict(),
            "elapsed_hours": self.controller.elapsed_hours(),
            "progress": self.progress.to_dict(),
            "trainer": self.trainer.trainer_state(),
            "archive": self.archive.to_dict(),
            "active_pool": self.pool.to_dict(),
            "candidates": read_candidate_marks(self.storage.archive_root),
            "storage": self.storage.storage_state(),
            "started_from": self.started_from,
            "closed_reason": reason or self.progress.close_reason,
        }
        path = self.storage.run_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
        return manifest

    # -- telemetry ---------------------------------------------------------

    def emit_telemetry(self, *, collection: "dict | None" = None, rows=None) -> dict:
        assert self.controller is not None and self.trainer is not None
        rows = list(rows or [])
        last = rows[-1] if rows else {}
        state = self.trainer.trainer_state()
        collection = collection or {}
        games = int(collection.get("games_collected", 0)) or 1
        seconds = float(collection.get("seconds", 0.0)) or 1e-9
        mixture = iteration_mixture(
            max(self.progress.iteration, 1),
            segment=self.controller.segment(),
            pool=self.pool,
            population=self.population,
        )
        snapshot = build_snapshot(
            clock=self.controller.status(),
            training={
                "global_optimizer_step": state["global_optimizer_step"],
                "examples_consumed": state["examples_consumed"],
                "examples_per_second": (
                    state["examples_consumed"] / max(state["wall_clock"]["train_seconds"], 1e-9)
                ),
                # The accepted loss module names these `loss_ppo` / `loss_value`
                # / `loss_belief` / `behavior_kl`; the frozen metric list names
                # them policy/value/belief/KL. Mapping them here keeps both
                # spellings honest instead of renaming an accepted field.
                "policy_loss": last.get("loss_ppo"),
                "value_loss": last.get("loss_value"),
                "belief_loss": last.get("loss_belief"),
                "total_loss": last.get("loss_total"),
                "kl": last.get("behavior_kl"),
                "policy_entropy": last.get("policy_entropy"),
                "clip_fraction": last.get("clip_fraction"),
                "grad_norm": last.get("grad_norm_post_clip"),
                "learning_rate": state["learning_rate"],
                "entropy_coefficient": state["entropy_coefficient"],
                "advantage_retention": last.get("advantage_retention"),
                "kl_beta": state["kl_beta"],
                "segment": state["segment"],
                "cursor": state["cursor"],
            },
            collection={
                "games_generated": self.progress.games_generated,
                "positions_generated": self.progress.positions_generated,
                "games_per_second": int(collection.get("games_collected", 0)) / seconds,
                "draw_rate": (
                    int(collection.get("terminal_results", {}).get("draw", 0)) / games
                ),
                "mean_game_length": int(collection.get("total_plies", 0)) / games,
                "iteration": collection.get("iteration", self.progress.iteration),
            },
            population={
                "percentages": mixture["percentages"],
                "active_pool": self.pool.members(),
                "archive_k": self.archive.k,
                "historical_categories": {
                    name: entry["games"]
                    for name, entry in mixture["historical_categories"].items()
                },
            },
            checkpoints={
                "hot": self.hot.status(),
                "hot_age_seconds": (
                    None if self.last_hot_unix is None else time.time() - self.last_hot_unix
                ),
                "archive_snapshots": self.archive.k,
                "latest_archive": (
                    self.archive.entries[-1].to_dict() if self.archive.entries else None
                ),
            },
            candidates=self._candidate_state()["ledger"],
            storage=self.storage.reserve_status(),
            workers={
                "status": "single-process bulk-synchronous loop",
                "loader_workers": (
                    None if self.topology is None else getattr(self.topology, "workers", None)
                ),
            },
            counters=state["counters"],
            failures=dict(self.progress.failures),
        )
        self.telemetry.write(snapshot)
        return snapshot


def runner_semantics() -> dict:
    return {
        "runner_version": PHASE14_RUNNER_VERSION,
        "unit": "collect 2,048 games -> bind -> 2 epochs -> committed boundary",
        "segment_fixed_at": "collection-unit launch",
        "deadline": "stops new units and new optimizer steps; finalizes in place",
        "resume": "newest valid hot checkpoint; original window; pool re-derived and compared",
        "recoverable": [error.__name__ for error in RECOVERABLE_ERRORS],
        "unrecoverable": [error.__name__ for error in UNRECOVERABLE_ERRORS],
        "search": "absent from the training graph",
    }
