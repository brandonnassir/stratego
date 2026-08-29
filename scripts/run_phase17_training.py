#!/usr/bin/env python
"""Phase 17: the tandem training session and its production entry point.

This module is the production loop for operator decision D10's simplified
paper-shaped recipe (`phase17_simple_paper_tandem_v1`, run `RUN-2026-B`).
`--start` refuses to run without an explicit `--i-am-agent-7` token so the
12-hour job cannot be launched by accident from a smoke.

What the session owns, and why it is separate from `TandemRunner`
-----------------------------------------------------------------
`TandemRunner` is the ten contractual steps of one iteration. Everything with a
*cadence* -- checkpoints, 30-minute exports, telemetry rows, warning readings --
lives here, so the smoke and the production run drive the identical iteration
code and differ only in how long they are asked to run.

Hour 0 is before the first optimizer update
--------------------------------------------
Not "at the first thirty-minute mark". :meth:`TrainingSession.export_hour_zero`
must be called before :meth:`step`, and it is the Phase 9 move policy plus the
freshly initialised random setup policy -- the baseline every later candidate is
measured against.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.training.phase17.checkpoint import (  # noqa: E402
    CheckpointIdentity,
    file_sha256,
    json_digest,
    read_joint_checkpoint,
    write_joint_checkpoint,
)
from stratego.training.phase17.export import (  # noqa: E402
    EXPORT_INTERVAL_SECONDS,
    build_manifest,
    due_boundaries,
    write_paired_export,
)
from stratego.training.phase17.runner import (  # noqa: E402
    TandemConfig,
    TandemRunner,
    move_means,
)
from stratego.training.phase17.setup_contract import (  # noqa: E402
    PRODUCTION_RUN_ID,
    SETUP_RECIPE_VERSION,
)
from stratego.training.phase17.source_identity import (  # noqa: E402
    Phase17SourceError,
    production_source_closure,
    require_source_digest,
    verify_source_digest,
)
from stratego.training.phase17.supervisor import MODE_PRODUCTION  # noqa: E402
from stratego.training.phase17.telemetry import (  # noqa: E402
    TelemetryWriter,
    telemetry_schema,
)

MOVE_PARAMETER_COUNT = 863959
SETUP_PARAMETER_COUNT = 802320


class TrainingSession:
    """One tandem run: iterations, telemetry, checkpoints and paired exports."""

    def __init__(
        self,
        config: TandemConfig,
        *,
        directory: "str | Path",
        supervisor_mode: str = MODE_PRODUCTION,
        source_digest: str,
        schedule_digest: str = "",
        runner: "TandemRunner | None" = None,
        reading_every: int = 25,
        reading_samples: int = 160,
    ) -> None:
        self.config = config
        self.directory = Path(directory)
        self.checkpoints = self.directory / "checkpoints"
        self.exports = self.directory / "exports"
        for path in (self.checkpoints, self.exports):
            path.mkdir(parents=True, exist_ok=True)

        self.runner = runner or TandemRunner(
            config, supervisor_mode=supervisor_mode
        )
        self.telemetry = TelemetryWriter(
            path=self.directory / "telemetry.jsonl", run_id=config.run_id
        ).open()
        self.config_digest = json_digest(config.document())
        # Agent 4C correction 1. Refused here rather than at the first
        # checkpoint: an empty digest is written into every checkpoint, export
        # and resume check, where it compares equal to every other empty
        # digest, so a run that never bound its source passes every identity
        # check it has and nobody finds out until the candidates are compared.
        self.source_digest = require_source_digest(
            source_digest, context=f"run {config.run_id!r}"
        )
        self.schedule_digest = schedule_digest
        self.run_digest = json_digest(
            {
                "run_id": config.run_id,
                "config_digest": self.config_digest,
                "schedule_digest": schedule_digest,
                # The source closure is part of the run's identity, not a note
                # about it: the same config under different code is a different
                # run, and two candidate sets that disagree here are not
                # comparable.
                "source_digest": self.source_digest,
            }
        )
        self.checkpoint_generation = 0
        self.parent_checkpoint: dict = {"path": None, "generation": 0}
        self.candidates: list = []
        self.last_export_seconds = 0.0
        self.stopped: "dict | None" = None

        # The predicates that are not per-iteration quantities.
        #
        # P4 (setup prefix entropy) and P5 (flag effective support) need a
        # *sample* from the setup policy, which costs a forward pass over a few
        # hundred sequences, so they are read on their own cadence rather than
        # every iteration. Without this wiring they would simply never fire in
        # production: `TandemRunner._supervise` only sees quantities an
        # iteration already produced.
        self.reading_every = int(reading_every)
        self.reading_samples = int(reading_samples)
        self.readings: list = []
        # P6 needs a first-hour median before it can mean anything. The samples
        # and the median live in the supervisor, whose state document the
        # checkpoint carries: held here they would be rebuilt from a
        # post-resume sample and silently move P6's floor (Agent 4C
        # correction 4).
        # P7 needs to know when a whole export interval passed with no update.
        self.last_setup_update_seconds = 0.0

    @property
    def first_hour_move_entropies(self) -> list:
        return self.runner.supervisor.move_entropy_first_hour_samples

    @property
    def first_hour_median_set(self) -> bool:
        return self.runner.supervisor.move_entropy_first_hour_closed

    # -- exports -----------------------------------------------------------

    def _export(self, index: int) -> dict:
        runner = self.runner
        manifest = build_manifest(
            run_id=self.config.run_id,
            index=index,
            move_ema_digest=runner.move_ema_digest(),
            setup_ema_digest=runner.setup_ema_digest(),
            move_parameter_count=MOVE_PARAMETER_COUNT,
            setup_parameter_count=SETUP_PARAMETER_COUNT,
            start_identity={
                key: runner.start.identity[key]
                for key in ("path", "file_sha256", "model_state_digest")
            },
            parent_checkpoint=dict(self.parent_checkpoint),
            config_digest=self.config_digest,
            source_digest=self.source_digest,
            elapsed_active_training_seconds=runner.elapsed_active_training_seconds,
            iteration=runner.iteration,
        )
        candidate = write_paired_export(
            directory=self.exports,
            manifest=manifest,
            move_ema_state=runner.start.ema.state_dict(),
            setup_ema_state=runner.setup_trainer.ema.state_dict(),
        )
        self.candidates.append(candidate.document())
        return candidate.document()

    def export_hour_zero(self) -> dict:
        """The h0 candidate, before the first optimizer update touches anything."""
        if self.runner.iteration:
            raise RuntimeError(
                "hour 0 must be exported before the first optimizer update; this "
                f"session is already at iteration {self.runner.iteration}"
            )
        return self._export(0)

    def _due_exports(self, before: float) -> list:
        return [
            self._export(index)
            for index in due_boundaries(before, self.runner.elapsed_active_training_seconds)
        ]

    # -- checkpoints -------------------------------------------------------

    def checkpoint(self) -> CheckpointIdentity:
        self.checkpoint_generation += 1
        payload = self.runner.capture(
            checkpoint_generation=self.checkpoint_generation,
            parent_checkpoint_identity=dict(self.parent_checkpoint),
            config_digest=self.config_digest,
            source_digest=self.source_digest,
            run_digest=self.run_digest,
            # Agent 4C correction 2: the row for THIS iteration is appended
            # after the checkpoint lands, because it carries the checkpoint's
            # verified identity. Naming it here is what stops a resume from
            # truncating it as excess.
            telemetry_position=self.telemetry.position(
                pending_row_iteration=self.runner.iteration
            ),
            next_export_boundary_seconds=(
                (
                    int(
                        self.runner.elapsed_active_training_seconds
                        // EXPORT_INTERVAL_SECONDS
                    )
                    + 1
                )
                * EXPORT_INTERVAL_SECONDS
            ),
        )
        identity = write_joint_checkpoint(
            payload,
            self.checkpoints / f"joint_{self.checkpoint_generation:05d}.pt",
        )
        self.parent_checkpoint = identity.document()
        return identity

    def resume(self, path: "str | Path") -> dict:
        payload = read_joint_checkpoint(
            path,
            run_id=self.config.run_id,
            config_digest=self.config_digest,
            source_digest=self.source_digest,
        )
        report = self.runner.restore(payload)
        self.checkpoint_generation = int(payload["checkpoint_generation"])
        # Agent 4C correction 3. The parent of the NEXT checkpoint is the one
        # this resume loaded, not that checkpoint's own parent. Copying the
        # loaded checkpoint's parent linked generation g+1 straight back to
        # g-1, so the chain skipped a generation at every resume and a reader
        # walking the parent links could not tell a resumed run from one that
        # never wrote generation g at all.
        self.parent_checkpoint = CheckpointIdentity(
            path=str(path),
            generation=int(payload["checkpoint_generation"]),
            iteration=int(payload["iteration"]),
            file_sha256=file_sha256(path),
            payload_digest=payload["payload_digest"],
        ).document()
        self.telemetry.close()
        self.telemetry = TelemetryWriter.resume(
            payload["telemetry_position"], run_id=self.config.run_id
        ).open()
        return report

    # -- one step ----------------------------------------------------------

    # -- the predicates that are not per-iteration quantities ---------------

    def _cadence_guards(self, result) -> dict:
        """Feed P4, P5, P6 and P7, each on the cadence its quantity lives on."""
        supervisor = self.runner.supervisor
        elapsed = self.runner.elapsed_active_training_seconds

        # Every predicate fed here is a WARNING under D10 section 7. They are
        # still read on their proper cadence, because a warning nobody measures
        # is not telemetry.
        #
        # P6: collect the first hour, then fix the median once and for all.
        # The predicate is observed on EVERY iteration regardless, so the
        # telemetry carries a P6 verdict from iteration 1 rather than starting
        # to mention it an hour in. Before the median exists the supervisor
        # returns a non-tripped verdict by construction -- there is nothing yet
        # to be 25% of.
        verdicts = []
        entropy = move_means(result.move_update.means or {}, "policy_entropy")
        median = supervisor.note_first_hour_move_entropy(
            entropy, first_hour_complete=elapsed >= 3600.0
        )
        verdicts.append(
            supervisor.observe_move_entropy(entropy, first_hour_median=median)
        )

        # P7: silence for a whole export interval while work was available.
        updated = result.setup_update is not None and not result.setup_update.skipped
        if updated:
            self.last_setup_update_seconds = elapsed
        verdicts.append(
            supervisor.observe_setup_update_activity(
                updated=updated,
                interval_complete=(
                    elapsed - self.last_setup_update_seconds >= EXPORT_INTERVAL_SECONDS
                ),
                episodes_available=bool(result.buffer_telemetry.get("depth", 0)),
            )
        )

        # P4 and P5: on the reading cadence, from a fresh sample.
        reading = None
        if (
            self.reading_every
            and self.runner.iteration % self.reading_every == 0
        ):
            reading = self.runner.concentration_reading(
                samples=self.reading_samples,
                label=f"iteration_{self.runner.iteration}",
            )
            self.readings.append(reading)
            verdicts.append(
                supervisor.observe_setup_entropy(reading["mean_prefix_entropy_nats"])
            )
            verdicts.append(
                supervisor.observe_flag_support(reading["flag_effective_support"])
            )
        # Agent 4C correction 4. These verdicts are produced AFTER
        # `run_iteration` returned, so they are not in `result.verdicts` and
        # used to reach no telemetry row at all: P4-P7 could trip on every
        # iteration of the run and the log would never say so. They are carried
        # out here and merged into the row's warning list.
        return {
            "reading": reading,
            "first_hour_median_set": self.first_hour_median_set,
            "verdicts": verdicts,
        }

    def step(self, *, checkpoint: bool = True) -> dict:
        """One iteration, its telemetry row, its exports and its checkpoint."""
        before = self.runner.elapsed_active_training_seconds
        result = self.runner.run_iteration()
        cadence = self._cadence_guards(result)
        exports = self._due_exports(before)

        checkpoint_seconds = 0.0
        identity = None
        if checkpoint:
            started = time.perf_counter()
            identity = self.checkpoint()
            checkpoint_seconds = time.perf_counter() - started

        row = self.telemetry_row(
            result,
            exports=exports,
            checkpoint=identity.document() if identity else None,
            checkpoint_seconds=checkpoint_seconds,
            cadence=cadence,
        )
        receipt = self.telemetry.append(row)
        if self.runner.supervisor.should_stop and self.stopped is None:
            self.stopped = self.runner.supervisor.stop_record()
        return {
            "result": result,
            "row": row,
            "receipt": receipt,
            "exports": exports,
            "checkpoint": identity.document() if identity else None,
            "reading": cadence["reading"],
        }

    # -- the telemetry row -------------------------------------------------

    def telemetry_row(
        self, result, *, exports, checkpoint, checkpoint_seconds, cadence=None
    ) -> dict:
        runner = self.runner
        window = result.window
        move = result.move_update
        setup = result.setup_update
        provider = result.provider_telemetry
        means = move.means or {}
        divergence = window.divergence_summary()
        verdicts = list(result.verdicts) + list((cadence or {}).get("verdicts") or [])
        setup_means = {}
        if setup is not None and not setup.skipped:
            setup_means = (setup.epochs[-1] if setup.epochs else {}) or {}

        return {
            "move": {
                "transitions_harvested": int(window.transitions_harvested),
                "transitions_trained": int(move.trained_rows),
                "boundary_rows": int(window.boundary_rows),
                "terminal_rows": int(window.terminal_rows),
                "active_games": int(window.active_games),
                "games_completed": int(window.games_finished),
                "game_lengths": list(window.game_lengths),
                "terminal_reasons": dict(window.terminal_reasons),
                "terminal_results": dict(window.terminal_results),
                "plies_advanced": int(window.plies_advanced),
                "loss_components": {
                    name: float(value)
                    for name, value in means.items()
                    if isinstance(value, (int, float))
                },
                "entropy": move_means(means, "policy_entropy"),
                "entropy_normalized": move_means(means, "policy_entropy_normalized"),
                "mean_kl": move_means(means, "behavior_kl"),
                "kl_beta": float(move.kl_beta),
                "clip_fraction": move_means(means, "clip_fraction"),
                # NOT surfaced by the accepted Agent 2 trainer. It computes the
                # pre-clip norm inside `_step` and uses it only to refuse a
                # non-finite gradient; nothing carries it out. Reported as null
                # with the reason rather than as 0.0, because a zero here would
                # read as a measured collapse. Agent 2 would need a versioned
                # amendment to expose it; Agent 4 does not alter its behavior.
                "grad_norm": None,
                "grad_norm_unavailable_reason": (
                    "MoveWindowTrainer does not surface the pre-clip gradient "
                    "norm; the clip is fixed and non-finite gradients are counted"
                ),
                "gradient_clip_norm": 1.0,
                "non_finite_gradients": int(
                    move.counters.get("non_finite_gradients", 0)
                ),
                "learning_rate": float(move.learning_rate),
                "entropy_coefficient": float(move.entropy_coefficient),
                "raw_model_state_digest": move.raw_digest_after,
                "ema_model_state_digest": runner.move_ema_digest(),
                "optimizer_steps": int(move.steps),
                "participant_ledger": runner.collector.participant_ledger(),
                "boundary_target_divergence": divergence,
                "bootstrap_age_windows": int(divergence["max_windows_spanned"]),
                "carried_traces": int(window.carried_traces),
                "dropped_pending": int(window.dropped_pending),
                "sealed_at_boundary": int(window.sealed_at_boundary),
                "collection_seconds": float(result.seconds["collection"]),
                "target_seconds": float(result.seconds["collection"]),
                "optimization_seconds": float(result.seconds["move_optimization"]),
                "rebind": dict(result.rebind),
            },
            "setup": {
                "generated": int(provider.get("generated", 0)),
                "refills": int(provider.get("refills", 0)),
                "unused": int(provider.get("unused", 0)),
                "discarded_on_rebind": int(provider.get("discarded_on_rebind", 0)),
                "snapshot_iteration": int(provider.get("snapshot_iteration", 0)),
                "raw_model_state_digest": runner._setup_digest(),
                "ema_model_state_digest": runner.setup_ema_digest(),
                "legality_failures": int(runner.provider.legality_failures),
                "orientation_failures": int(runner.provider.orientation_failures),
                "fallback_attempts": int(runner.provider.fallback_attempts),
                "completed_episode_buffer": dict(result.buffer_telemetry),
                "activity": {
                    "skips": int(runner.setup_skips),
                    "updates": int(runner.setup_updates),
                },
                "updated": bool(setup is not None and not setup.skipped),
                "skip_reason": result.setup_skip_reason,
                "episodes_consumed": int(setup.episodes_consumed) if setup else 0,
                "loss_components": {
                    name: float(value)
                    for name, value in setup_means.items()
                    if isinstance(value, (int, float))
                },
                # D10 section 4: record the component magnitudes. The printed
                # advantage's entropy term is `alpha * (I - h)` in nats against
                # an outcome term bounded by 2, so their ratio is the number a
                # reader of the 12-hour curve needs and it is carried here every
                # iteration rather than derived afterwards.
                "advantage_components": dict(setup.advantage_telemetry or {})
                if setup and not setup.skipped
                else {},
                "empirical_entropy": float(
                    setup_means.get("mean_prefix_entropy_nats", 0.0)
                ),
                "predicted_entropy": float(
                    setup_means.get("conditional_entropy_loss", 0.0)
                ),
                "predicted_entropy_is": (
                    "L_h, the conditional-entropy head's squared error against "
                    "its target I/10. The prediction h itself IS read by the "
                    "D10 advantage, which subtracts it from I in nats."
                ),
                "mean_kl": float(setup.mean_iteration_kl) if setup and not setup.skipped else 0.0,
                "final_epoch_kl": float(setup.final_epoch_kl) if setup and not setup.skipped else 0.0,
                # A FIXED coefficient. Never named `kl_beta`, never carrying a
                # controller's target or bounds: D10 section 1.
                "kl_coefficient": float(runner.setup_config.behavior_kl_coefficient),
                "kl_direction": runner.setup_config.kl_direction,
                "grad_norm": float(setup.gradient_norm_mean) if setup and not setup.skipped else 0.0,
                "learning_rate": float(runner.setup_config.learning_rate),
                "alpha": float(setup.alpha) if setup else float(
                    runner.setup_config.alpha(max(1, result.iteration))
                ),
                "optimizer_steps": int(setup.optimizer_steps) if setup else 0,
                "concentration": {
                    "cadence_iterations": int(self.reading_every),
                    "samples_per_color": int(self.reading_samples),
                    "measured_this_iteration": bool(
                        cadence and cadence.get("reading")
                    ),
                    "last": (
                        {
                            key: (cadence["reading"])[key]
                            for key in (
                                "setup_iteration",
                                "mean_prefix_entropy_nats",
                                "percent_of_baseline",
                                "crosses_relative_floor",
                                "flag_effective_support",
                                "bomb_effective_support",
                                "flag_square_support",
                                "reflection_class_unique_fraction",
                                "min_class_distance",
                                "mean_top_token_concentration",
                                "setup_model_state_digest",
                            )
                        }
                        if cadence and cadence.get("reading")
                        else (
                            {
                                key: self.readings[-1][key]
                                for key in (
                                    "setup_iteration",
                                    "mean_prefix_entropy_nats",
                                    "percent_of_baseline",
                                    "crosses_relative_floor",
                                    "flag_effective_support",
                                )
                            }
                            if self.readings
                            else None
                        )
                    ),
                },
                "generation_seconds": float(result.seconds["setup_generation"]),
                "optimization_seconds": float(result.seconds["setup_optimization"]),
            },
            "system": {
                "recipe": SETUP_RECIPE_VERSION,
                "run_id": self.config.run_id,
                "work_package": self.config.work_package,
                "iteration": int(result.iteration),
                "elapsed_active_training_seconds": float(
                    runner.elapsed_active_training_seconds
                ),
                "cadence_index": int(
                    runner.elapsed_active_training_seconds // EXPORT_INTERVAL_SECONDS
                ),
                "memory_high_water_mib": _peak_memory_mib(),
                "checkpoint": checkpoint,
                "export": exports,
                # Partitioned by CONSEQUENCE, not by `fired`. P5 and P7 need
                # one observation to fire, so a tripped warning is a fired
                # verdict and the old `fired` test filed it under
                # `stop_predicates` -- a row claiming the run was stopping when
                # D10 section 7 says these can never stop it. `stops_the_run`
                # is already computed per verdict as "fired AND severity is
                # stop", which is exactly the question each list is asking.
                "warnings": [
                    verdict
                    for verdict in verdicts
                    if verdict["tripped"] and not verdict["stops_the_run"]
                ],
                "stop_predicates": [
                    verdict for verdict in verdicts if verdict["stops_the_run"]
                ],
                "external_result_status": "not_connected_in_this_session",
                "checkpoint_seconds": float(checkpoint_seconds),
                "total_seconds": float(result.seconds["total"]),
            },
        }

    def close(self) -> None:
        self.telemetry.close()


def _peak_memory_mib() -> float:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def load_frozen(schedule_path: Path, throughput_path: Path) -> dict:
    """The frozen horizon and the measured host configuration.

    The setup budget policy that used to be read from `throughput` is gone:
    D10 consumes every completed episode, so there is no quota to freeze. What
    survives is the move schedule's horizon `N`, which is still frozen before
    h0 and never recomputed from production speed.
    """
    schedule = json.loads(schedule_path.read_text())
    throughput = json.loads(throughput_path.read_text())
    return {"schedule": schedule, "throughput": throughput}


def build_production_config(frozen: dict, *, run_id: str) -> TandemConfig:
    schedule = frozen["schedule"]
    configuration = frozen["throughput"]["configuration"]
    return TandemConfig(
        run_id=run_id,
        total_iterations=int(schedule["N"]),
        move_budget=int(configuration["budget_transitions"]),
        population=int(configuration["population"]),
        pool_size_per_side=int(configuration["pool_size_per_side"]),
        move_device=frozen["throughput"]["host"]["move_device"],
        setup_device=frozen["throughput"]["host"]["setup_device"],
    )


def production_command(*, run_id: str, source_digest: str) -> str:
    """The exact shape Agent 6 binds and Agent 7 runs.

    Emitted from one place so the launch manifest, the handoff and the dry run
    cannot drift into three slightly different commands.
    """
    return (
        "nohup caffeinate -dimsu .venv/bin/python scripts/run_phase17_training.py "
        f"--run-id {run_id} --start --i-am-agent-7 "
        f"--source-digest {source_digest} > <log> 2>&1 &"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=PRODUCTION_RUN_ID)
    parser.add_argument(
        "--directory", default=f"checkpoints/phase17/{PRODUCTION_RUN_ID}"
    )
    parser.add_argument("--schedule", default="reports/phase17/agent_04_schedule.json")
    parser.add_argument("--throughput", default="reports/phase17/agent_04_throughput.json")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--i-am-agent-7", action="store_true")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument(
        "--source-digest",
        default=None,
        help=(
            "the production source-closure digest Agent 6 froze in the launch "
            "manifest. Required by --start: the working tree's closure is "
            "recomputed and must match it exactly."
        ),
    )
    arguments = parser.parse_args()

    frozen = load_frozen(
        REPOSITORY_ROOT / arguments.schedule, REPOSITORY_ROOT / arguments.throughput
    )
    config = build_production_config(frozen, run_id=arguments.run_id)
    # Recomputed on every invocation, describe included: --describe is how
    # Agent 6 reads the value it is about to freeze, so it must be the value
    # this program would bind, never a constant compiled in beside it.
    closure = production_source_closure(root=REPOSITORY_ROOT)

    if arguments.describe or not arguments.start:
        print(
            json.dumps(
                {
                    "recipe": SETUP_RECIPE_VERSION,
                    "would_run": config.document(),
                    "schedule_digest": frozen["schedule"]["schedule_digest"],
                    "telemetry_schema": telemetry_schema()["schema_version"],
                    "expected_candidates": 25,
                    "started": False,
                    "source_closure": closure,
                    "source_digest": closure["closure_digest"],
                    "production_command": production_command(
                        run_id=arguments.run_id,
                        source_digest=closure["closure_digest"],
                    ),
                    "note": (
                        "Agent 4C does not start the 12-hour run. Agent 6 binds "
                        "this command in the launch manifest, freezing the "
                        "source_digest printed above, and Agent 7 runs it with "
                        "--start --i-am-agent-7 --source-digest after operator "
                        "approval."
                    ),
                },
                indent=1,
                default=str,
            )
        )
        return 0

    if not arguments.i_am_agent_7:
        print(
            "refusing to start: the 12-hour production run needs --i-am-agent-7 "
            "and operator launch approval.",
            file=sys.stderr,
        )
        return 2

    # The authorized closure, recomputed from the bytes on disk. A mismatch is
    # a working tree that is not what the operator approved, and it is refused
    # before anything is written under the run's name.
    try:
        closure = verify_source_digest(
            arguments.source_digest,
            root=REPOSITORY_ROOT,
            context=f"run {arguments.run_id!r}",
        )
    except Phase17SourceError as error:
        print(f"refusing to start: {error}", file=sys.stderr)
        return 3

    session = TrainingSession(
        config,
        directory=REPOSITORY_ROOT / arguments.directory,
        supervisor_mode=MODE_PRODUCTION,
        source_digest=closure["closure_digest"],
        schedule_digest=frozen["schedule"]["schedule_digest"],
    )
    if arguments.resume:
        session.resume(arguments.resume)
    else:
        session.export_hour_zero()
    while session.runner.iteration < config.total_iterations and not session.stopped:
        session.step()
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
