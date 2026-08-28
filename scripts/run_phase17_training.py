#!/usr/bin/env python
"""Phase 17 Agent 4: the tandem training session and its production entry point.

This module is the production loop. Agent 4 does **not** start the 12-hour job;
`--start` exists so Agent 6 has one exact command to bind in the launch
manifest, and it refuses to run without an explicit `--i-am-agent-7` token so
it cannot be launched by accident from this agent's own rehearsals.

What the session owns, and why it is separate from `TandemRunner`
-----------------------------------------------------------------
`TandemRunner` is the ten contractual steps of one iteration. Everything with a
*cadence* -- checkpoints, 30-minute exports, telemetry rows, stop checks --
lives here, so the rehearsal and the production run drive the identical
iteration code and differ only in how long they are asked to run and in what
the supervisor is allowed to treat as fatal.

Hour 0 is before the first optimizer update
--------------------------------------------
Not "at the first thirty-minute mark". :meth:`TrainingSession.export_hour_zero`
must be called before :meth:`step`, and it is the Phase 9 policy plus the
freshly initialised setup policy -- the baseline every later candidate is
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
from stratego.training.phase17.queue import SetupBudgetPolicy  # noqa: E402
from stratego.training.phase17.runner import (  # noqa: E402
    TandemConfig,
    TandemRunner,
    move_means,
)
from stratego.training.phase17.supervisor import (  # noqa: E402
    MODE_INTEGRATION,
    MODE_PRODUCTION,
)
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
        budget_policy: "SetupBudgetPolicy | None" = None,
        supervisor_mode: str = MODE_PRODUCTION,
        source_digest: str = "",
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
            config, budget_policy=budget_policy, supervisor_mode=supervisor_mode
        )
        self.telemetry = TelemetryWriter(
            path=self.directory / "telemetry.jsonl", run_id=config.run_id
        ).open()
        self.config_digest = json_digest(config.document())
        self.source_digest = source_digest
        self.schedule_digest = schedule_digest
        self.run_digest = json_digest(
            {
                "run_id": config.run_id,
                "config_digest": self.config_digest,
                "schedule_digest": schedule_digest,
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
        # P6 needs a first-hour median before it can mean anything.
        self.first_hour_move_entropies: list = []
        self.first_hour_median_set = False
        # P7 needs to know when a whole export interval passed with no update.
        self.last_setup_update_seconds = 0.0

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
            telemetry_position=self.telemetry.position(),
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
        self.parent_checkpoint = dict(payload["parent_checkpoint_identity"])
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

        # P6: collect the first hour, then fix the median once and for all.
        # The predicate is observed on EVERY iteration regardless, so the
        # telemetry carries a P6 verdict from iteration 1 rather than starting
        # to mention it an hour in. Before the median exists the supervisor
        # returns a non-tripped verdict by construction -- there is nothing yet
        # to be 25% of.
        entropy = move_means(result.move_update.means or {}, "policy_entropy")
        median = None
        if not self.first_hour_median_set:
            self.first_hour_move_entropies.append(entropy)
            if elapsed >= 3600.0:
                ordered = sorted(self.first_hour_move_entropies)
                middle = len(ordered) // 2
                median = (
                    ordered[middle]
                    if len(ordered) % 2
                    else 0.5 * (ordered[middle - 1] + ordered[middle])
                )
                self.first_hour_median_set = True
        supervisor.observe_move_entropy(entropy, first_hour_median=median)

        # P7: silence for a whole export interval while work was available.
        updated = result.setup_update is not None and not result.setup_update.skipped
        if updated:
            self.last_setup_update_seconds = elapsed
        supervisor.observe_setup_update_activity(
            updated=updated,
            interval_complete=(
                elapsed - self.last_setup_update_seconds >= EXPORT_INTERVAL_SECONDS
            ),
            warmed_up=bool(self.runner.warmed_up),
            episodes_available=bool(result.queue_telemetry.get("depth", 0)),
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
            supervisor.observe_setup_entropy(reading["mean_prefix_entropy_nats"])
            supervisor.observe_flag_support(reading["flag_effective_support"])
        return {
            "reading": reading,
            "first_hour_median_set": self.first_hour_median_set,
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
                "queue": dict(result.queue_telemetry),
                "starvation": {
                    "warmed_up": bool(runner.warmed_up),
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
                "empirical_entropy": float(
                    setup_means.get("mean_prefix_entropy_nats", 0.0)
                ),
                "predicted_entropy": float(
                    setup_means.get("conditional_entropy_loss", 0.0)
                ),
                "predicted_entropy_is": (
                    "L_h, the conditional-entropy head's loss. The head's "
                    "prediction h is retained for telemetry and paper alignment "
                    "and is NOT read by the D7-B advantage."
                ),
                "mean_kl": float(setup.mean_iteration_kl) if setup and not setup.skipped else 0.0,
                "control_kl": float(setup.control_kl) if setup and not setup.skipped else 0.0,
                "kl_beta": float(setup.beta_after) if setup else float(runner.setup_trainer.controller.beta),
                "grad_norm": float(setup.gradient_norm_mean) if setup and not setup.skipped else 0.0,
                "learning_rate": float(runner.setup_config.learning_rate),
                "alpha": float(setup.alpha) if setup else float(
                    runner.setup_config.alpha(max(1, runner.setup_trainer.setup_iteration))
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
                "warnings": [
                    verdict
                    for verdict in result.verdicts
                    if verdict["tripped"] and not verdict["fired"]
                ],
                "stop_predicates": [
                    verdict for verdict in result.verdicts if verdict["fired"]
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
    schedule = json.loads(schedule_path.read_text())
    throughput = json.loads(throughput_path.read_text())
    policy = throughput["setup_budget_policy"]
    return {"schedule": schedule, "throughput": throughput, "policy": policy}


def build_production_config(frozen: dict, *, run_id: str) -> TandemConfig:
    schedule = frozen["schedule"]
    policy = frozen["policy"]
    configuration = frozen["throughput"]["configuration"]
    return TandemConfig(
        run_id=run_id,
        total_iterations=int(schedule["N"]),
        move_budget=int(configuration["budget_transitions"]),
        population=int(configuration["population"]),
        pool_size_per_side=int(configuration["pool_size_per_side"]),
        setup_budget=int(policy["budget"]),
        setup_queue_capacity=int(policy["capacity"]),
        setup_warm_up_minimum=int(policy["warm_up_minimum"]),
        setup_max_age_iterations=int(policy["max_age_iterations"]),
        move_device=frozen["throughput"]["host"]["move_device"],
        setup_device=frozen["throughput"]["host"]["setup_device"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="RUN-2026-A")
    parser.add_argument("--directory", default="checkpoints/phase17/RUN-2026-A")
    parser.add_argument("--schedule", default="reports/phase17/agent_04_schedule.json")
    parser.add_argument("--throughput", default="reports/phase17/agent_04_throughput.json")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--i-am-agent-7", action="store_true")
    parser.add_argument("--describe", action="store_true")
    arguments = parser.parse_args()

    frozen = load_frozen(
        REPOSITORY_ROOT / arguments.schedule, REPOSITORY_ROOT / arguments.throughput
    )
    config = build_production_config(frozen, run_id=arguments.run_id)

    if arguments.describe or not arguments.start:
        print(
            json.dumps(
                {
                    "would_run": config.document(),
                    "schedule_digest": frozen["schedule"]["schedule_digest"],
                    "setup_budget_policy": frozen["policy"],
                    "telemetry_schema": telemetry_schema()["schema_version"],
                    "expected_candidates": 25,
                    "started": False,
                    "note": (
                        "Agent 4 does not start the 12-hour run. Agent 6 binds "
                        "this command in the launch manifest and Agent 7 runs it "
                        "with --start --i-am-agent-7 after operator approval."
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
            "and operator launch approval. Agent 4 rehearses; it does not launch.",
            file=sys.stderr,
        )
        return 2

    session = TrainingSession(
        config,
        directory=REPOSITORY_ROOT / arguments.directory,
        budget_policy=None,
        supervisor_mode=MODE_PRODUCTION,
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
