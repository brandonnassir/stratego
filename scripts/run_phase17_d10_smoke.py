#!/usr/bin/env python
"""Phase 17 Agent 4B: the D10 end-to-end tandem smoke and the handoff it writes.

Operator decision D10 section 6 replaced the gate-heavy launch workflow with
one short end-to-end tandem smoke on the production code path, capped at 30
minutes, checking nine things. This script is that smoke and nothing more.

```text
smoke     run the nine D10 checks and write the preflight JSON
handoff   write phase17_simple_tandem_handoff_v1.json from a completed smoke
```

What it deliberately does not do
--------------------------------
It does not run a standalone setup gate, a diversity soak, an entropy-floor
experiment, a controller calibration, a population sweep, a queue-arrival
study, a strength test or a failure-injection campaign. D10 section 6 names
every one of those as retired, and Agent 2/4's existing evidence covers the
machinery that survived. It also does not start the 12-hour run.

Why the smoke runs at production shape
--------------------------------------
The move budget, population and pool size are the frozen production values, so
the thing being smoked is the thing that will run. That costs about 67 seconds
per iteration on this host, measured, which is why the default is three
iterations plus one resumed one: roughly five minutes inside a thirty-minute
cap. A tiny-shaped smoke would exercise the same code and prove less --
"exactly the configured transition count is emitted" means little at 400.

Its weights are discarded
-------------------------
The run ID is a smoke ID, never `RUN-2026-B`, and the output directory is
excluded from version control. D10 section 3: production reinitializes from
Phase 9 plus a newly random setup model, and no smoke state may enter it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(REPOSITORY_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

REPORT_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase17"

#: The source closure this smoke's verdict is bound to. A later edit to any of
#: these invalidates the smoke, which is what the digest is for.
AGENT_4B_SOURCES = (
    "stratego/training/phase17/setup_contract.py",
    "stratego/training/phase17/setup_episode.py",
    "stratego/training/phase17/setup_learning.py",
    "stratego/training/phase17/setup_metrics.py",
    "stratego/training/phase17/runner.py",
    "stratego/training/phase17/checkpoint.py",
    "stratego/training/phase17/supervisor.py",
    "stratego/training/phase17/telemetry.py",
    "stratego/training/phase17/export.py",
    "scripts/run_phase17_training.py",
    "scripts/run_phase17_d10_smoke.py",
)

AGENT_4B_TESTS = (
    "tests/training/phase17/test_simple_paper_recipe.py",
    "tests/training/phase17/test_setup_learning.py",
    "tests/training/phase17/test_setup_episode.py",
    "tests/training/phase17/test_runner_tandem.py",
    "tests/training/phase17/test_supervisor_predicates.py",
    "tests/training/phase17/test_checkpoint_persistence.py",
)

#: The accepted Phase 9 move start, restated so the check asserts against a
#: constant rather than against whatever the loader happened to produce.
PHASE9_MOVE_STATE_DIGEST = (
    "f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd"
)
PHASE9_MOVE_FILE_SHA256 = (
    "dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea"
)

#: D10 section 6's cap. Exceeding it aborts rather than quietly running long.
SMOKE_CAP_SECONDS = 30 * 60


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_closure(paths) -> dict:
    entries = [
        {"path": name, "sha256": file_sha256(REPOSITORY_ROOT / name)} for name in paths
    ]
    return {"files": entries, "closure_digest": json_digest(entries)}


def peak_memory_mib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def git(*arguments) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def write_json(name: str, payload: dict) -> Path:
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIRECTORY / name
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
    return path


# ---------------------------------------------------------------------------
# The nine checks
# ---------------------------------------------------------------------------


class Checks:
    """A recorder, so a failed check is data rather than a traceback."""

    def __init__(self) -> None:
        self.entries: list = []

    def record(self, number: int, name: str, ok: bool, evidence) -> bool:
        self.entries.append(
            {
                "check": number,
                "name": name,
                "ok": bool(ok),
                "evidence": evidence,
            }
        )
        print(f"  [{'PASS' if ok else 'FAIL'}] {number}. {name}", flush=True)
        return bool(ok)

    @property
    def all_passed(self) -> bool:
        return all(entry["ok"] for entry in self.entries)

    @property
    def failed(self) -> list:
        return [entry for entry in self.entries if not entry["ok"]]


def smoke(arguments) -> dict:
    from stratego.engine.constants import BLUE, PIECE_COUNTS, RED
    from stratego.training.phase9_behavior import state_dict_digest
    from stratego.training.phase17.checkpoint import read_joint_checkpoint
    from stratego.training.phase17.export import verify_paired_export
    from stratego.training.phase17.runner import TandemConfig
    from stratego.training.phase17.setup_contract import (
        PRODUCTION_RUN_ID,
        SETUP_BEHAVIOR_KL_COEFFICIENT,
        SETUP_RECIPE_VERSION,
        setup_alpha,
    )
    from stratego.training.phase17.setup_model import build_setup_model
    from stratego.training.phase17.setup_sampling import to_engine_setup
    from stratego.training.phase17.supervisor import MODE_PRODUCTION
    from stratego.training.phase17.telemetry import read_rows, telemetry_schema

    from run_phase17_training import (
        TrainingSession,
        build_production_config,
        load_frozen,
    )

    if arguments.run_id == PRODUCTION_RUN_ID:
        raise SystemExit(
            f"refusing to smoke under the production run ID {PRODUCTION_RUN_ID}: "
            "D10 section 3 requires production to reinitialize from Phase 9 plus "
            "a newly random setup model, and a smoke's weights are discarded"
        )

    frozen = load_frozen(
        REPOSITORY_ROOT / arguments.schedule, REPOSITORY_ROOT / arguments.throughput
    )
    production = build_production_config(frozen, run_id=PRODUCTION_RUN_ID)
    config = TandemConfig(
        run_id=arguments.run_id,
        total_iterations=production.total_iterations,
        move_budget=production.move_budget,
        population=production.population,
        pool_size_per_side=production.pool_size_per_side,
        setup_minibatch_episodes=production.setup_minibatch_episodes,
        move_device=arguments.device or production.move_device,
        setup_device=arguments.setup_device or production.setup_device,
        # A different seed from production's, so the smoke and the run cannot
        # be confused by their setup identity either.
        setup_model_seed=production.setup_model_seed + 1000,
    )

    directory = REPOSITORY_ROOT / arguments.directory
    if directory.exists():
        import shutil

        shutil.rmtree(directory)

    sources = source_closure(AGENT_4B_SOURCES)
    session = TrainingSession(
        config,
        directory=directory,
        supervisor_mode=MODE_PRODUCTION,
        source_digest=sources["closure_digest"],
        schedule_digest=frozen["schedule"]["schedule_digest"],
        # The concentration reading is descriptive under D10 and costs a
        # forward pass over 320 sequences; taken once so the row shape is
        # exercised, not on a production cadence.
        reading_every=arguments.iterations,
        reading_samples=64,
    )
    checks = Checks()
    started = time.perf_counter()
    runner = session.runner

    print(f"D10 smoke: {config.run_id}, {arguments.iterations} iterations", flush=True)
    print(f"  move {config.move_device} / setup {config.setup_device}", flush=True)

    # -- 1. exact Phase 9 move identity and fresh setup identity ------------
    fresh_setup = build_setup_model(device="cpu", seed=config.setup_model_seed)
    identity = runner.identity_document()
    checks.record(
        1,
        "exact Phase 9 move identity and a from-scratch setup model",
        identity["start_identity"]["model_state_digest"] == PHASE9_MOVE_STATE_DIGEST
        and identity["start_identity"]["file_sha256"] == PHASE9_MOVE_FILE_SHA256
        and identity["setup_start_model_state_digest"] == state_dict_digest(fresh_setup)
        and runner.setup_trainer.updates == 0
        and identity["recipe"] == SETUP_RECIPE_VERSION,
        {
            "move_start": identity["start_identity"],
            "setup_start_model_state_digest": identity["setup_start_model_state_digest"],
            "setup_model_seed": config.setup_model_seed,
            "setup_initialization": "from scratch under the recorded seed",
            "recipe": identity["recipe"],
            "production_refuses_an_injected_setup_model": True,
        },
    )

    h0 = session.export_hour_zero()
    verified = verify_paired_export(h0["path"], expected_file_sha256=h0["file_sha256"])

    # -- the iterations -----------------------------------------------------
    steps = []
    for index in range(arguments.iterations):
        elapsed = time.perf_counter() - started
        if elapsed > SMOKE_CAP_SECONDS:
            raise SystemExit(
                f"the D10 smoke exceeded its {SMOKE_CAP_SECONDS}s cap after "
                f"{index} iterations; report the measured rate rather than "
                "extending the cap"
            )
        step = session.step()
        steps.append(step)
        result = step["result"]
        print(
            f"  iteration {result.iteration}: "
            f"{result.window.transitions_harvested} transitions, "
            f"{result.window.games_finished} games, "
            f"setup {'skipped' if result.setup_skipped else result.setup_update.episodes_consumed},"
            f" {result.seconds['total']:.1f}s",
            flush=True,
        )

    results = [step["result"] for step in steps]
    windows = [result.window for result in results]

    # -- 2. both seats, current raw policy, sampled legal actions -----------
    known_digests = {
        entry["model_state_digest"] for entry in runner.cell.digest_history()
    }
    illegal = 0
    unknown = 0
    by_color = {int(RED): 0, int(BLUE): 0}
    for window in windows:
        for row in window.rows:
            by_color[int(row.color)] += 1
            if row.sampled_action not in row.legal_actions or not bool(
                row.legal_mask[row.sampled_action_model]
            ):
                illegal += 1
            if row.behavior_model_state_digest not in known_digests:
                unknown += 1
    # Sampling, not argmax: the sampled action must not always be the mode.
    argmax_matches = 0
    sampled_rows = 0
    for window in windows:
        for row in window.rows[:2000]:
            probabilities = list(row.behavior_probabilities)
            if len(probabilities) < 2:
                continue
            sampled_rows += 1
            best = max(range(len(probabilities)), key=probabilities.__getitem__)
            argmax_matches += int(best == row.sampled_action_index)
    checks.record(
        2,
        "both move seats use the current raw policy and sample legal actions",
        illegal == 0
        and unknown == 0
        and min(by_color.values()) > 0
        and sampled_rows > 0
        and argmax_matches < sampled_rows,
        {
            "illegal_sampled_actions": illegal,
            "rows_under_an_unknown_policy_digest": unknown,
            "rows_by_color": {"red": by_color[int(RED)], "blue": by_color[int(BLUE)]},
            "rebinds": [step["result"].rebind for step in steps],
            "argmax_agreement_fraction": argmax_matches / max(sampled_rows, 1),
            "argmax_rows_examined": sampled_rows,
        },
    )

    # -- 3. fresh legal, inventory-correct, oriented setups, no fallback ----
    inventory = {piece: count for piece, count in PIECE_COUNTS.items()}
    bad_inventory = 0
    bad_orientation = 0
    live = [game for game in runner.collector.slots if game is not None]
    for game in live:
        for color, setup in (
            (int(RED), tuple(game.builder.red_setup)),
            (int(BLUE), tuple(game.builder.blue_setup)),
        ):
            counts: dict = {}
            for piece in setup:
                counts[piece] = counts.get(piece, 0) + 1
            if counts != inventory or len(setup) != 40:
                bad_inventory += 1
            episodes = runner.provider.open_episodes.get(game.game_id)
            if episodes is None:
                continue
            episode = episodes.red if color == int(RED) else episodes.blue
            if tuple(episode.engine_setup) != to_engine_setup(
                tuple(episode.canonical_setup), color
            ):
                bad_orientation += 1
    provider = runner.provider.telemetry()
    checks.record(
        3,
        "fresh legal, inventory-correct, oriented setups with no library fallback",
        bad_inventory == 0
        and bad_orientation == 0
        and runner.provider.legality_failures == 0
        and runner.provider.orientation_failures == 0
        and runner.provider.fallback_attempts == 0
        and provider["generated"] > 0
        and all(result.pool_discarded > 0 for result in results[1:]),
        {
            "games_checked": len(live),
            "inventory_violations": bad_inventory,
            "orientation_violations": bad_orientation,
            "legality_failures": runner.provider.legality_failures,
            "orientation_failures": runner.provider.orientation_failures,
            "fallback_attempts": runner.provider.fallback_attempts,
            "setup_family": runner.provider.setup_family,
            "pool_regenerated_every_iteration": [
                {
                    "iteration": result.iteration,
                    "discarded_on_rebind": result.pool_discarded,
                    "snapshot_iteration": result.provider_telemetry["snapshot_iteration"],
                }
                for result in results
            ],
            "pool_telemetry": provider,
        },
    )

    # -- 4. exactly the configured transition count -------------------------
    #
    # "Emitted", not "PPO-trained". The accepted move objective keeps Phase 9's
    # top-quartile advantage filter, so `transitions_trained` is about a
    # quarter of the window by design and D10 leaves the move learner alone.
    # What has to be exact is the number of rows the window produces.
    checks.record(
        4,
        "exactly the configured move-transition count is emitted",
        all(window.transitions_harvested == config.move_budget for window in windows)
        and all(len(window.rows) == config.move_budget for window in windows)
        and all(
            window.boundary_rows + window.terminal_rows == config.move_budget
            for window in windows
        )
        and any(window.boundary_rows for window in windows),
        {
            "budget": config.move_budget,
            "harvested": [window.transitions_harvested for window in windows],
            "rows_emitted": [len(window.rows) for window in windows],
            "boundary_rows": [window.boundary_rows for window in windows],
            "terminal_rows": [window.terminal_rows for window in windows],
            "ppo_eligible_rows": [
                step["row"]["move"]["transitions_trained"] for step in steps
            ],
            "ppo_eligible_note": (
                "the accepted Phase 9 top-quartile advantage filter, unchanged "
                "by D10; the value head sees every emitted row"
            ),
            "active_games": [window.active_games for window in windows],
        },
    )

    # -- 5. a real completed game drives a five-epoch setup update ----------
    real = [result for result in results if not result.setup_update.skipped]
    checks.record(
        5,
        "at least one real completed game updates the setup model for five epochs",
        bool(real)
        and all(len(result.setup_update.epochs) == 5 for result in real)
        and all(result.setup_update.optimizer_steps > 0 for result in real)
        and all(
            result.setup_update.digest_before != result.setup_update.digest_after
            for result in real
        )
        and all(
            result.setup_update.episodes_consumed
            == 2 * result.window.games_finished
            for result in real
        )
        and all(result.buffer_telemetry["depth"] == 0 for result in real)
        and runner.setup_trainer.queue.enqueued_count
        == runner.setup_trainer.queue.consumed_count
        and not runner.enqueue_rejections,
        {
            "real_updates": len(real),
            "games_finished": [result.window.games_finished for result in results],
            "episodes_consumed": [
                result.setup_update.episodes_consumed for result in results
            ],
            "epochs": [len(result.setup_update.epochs) for result in real],
            "optimizer_steps": [result.setup_update.optimizer_steps for result in real],
            "buffer": runner.setup_trainer.queue.telemetry(
                runner.setup_trainer.setup_iteration
            ).__dict__,
            "enqueue_rejections": runner.enqueue_rejections,
            "skips": [
                result.setup_skip_reason for result in results if result.setup_skipped
            ],
            "seconds": [result.seconds["setup_optimization"] for result in results],
        },
    )

    # -- 6. the fixed reverse-KL coefficient --------------------------------
    coefficients = {result.setup_update.behavior_kl_coefficient for result in results}
    epoch_coefficients = {
        epoch["behavior_kl_coefficient"]
        for result in real
        for epoch in result.setup_update.epochs
    }
    checks.record(
        6,
        "setup reverse KL has fixed coefficient 0.1 with no adaptive controller",
        coefficients == {SETUP_BEHAVIOR_KL_COEFFICIENT}
        and epoch_coefficients in ({SETUP_BEHAVIOR_KL_COEFFICIENT}, set())
        and not hasattr(runner.setup_trainer, "controller")
        and runner.setup_config.kl_direction == "reverse_current_given_behavior"
        and all(
            step["row"]["setup"]["kl_coefficient"] == SETUP_BEHAVIOR_KL_COEFFICIENT
            for step in steps
        )
        and all("kl_beta" not in step["row"]["setup"] for step in steps),
        {
            "coefficient": SETUP_BEHAVIOR_KL_COEFFICIENT,
            "observed_per_iteration": sorted(coefficients),
            "observed_per_epoch": sorted(epoch_coefficients),
            "direction": runner.setup_config.kl_direction,
            "adaptive_controller_present": hasattr(runner.setup_trainer, "controller"),
            "telemetry_names_it": "setup.kl_coefficient (never setup.kl_beta)",
            "measured_final_epoch_kl": [
                result.setup_update.final_epoch_kl for result in real
            ],
            "measured_mean_iteration_kl": [
                result.setup_update.mean_iteration_kl for result in real
            ],
        },
    )

    # -- 7. alpha on the shared global iteration ----------------------------
    checks.record(
        7,
        "setup alpha equals 0.1 * n**-0.3 at the shared global iteration",
        all(
            abs(result.setup_update.alpha - setup_alpha(result.iteration)) < 1e-12
            for result in results
        )
        and all(
            result.setup_update.setup_iteration == result.iteration
            for result in results
        )
        and all(
            abs(step["row"]["setup"]["alpha"] - setup_alpha(step["result"].iteration))
            < 1e-12
            for step in steps
        ),
        {
            "formula": "0.1 * n**-0.3",
            "n_is": "the shared one-based global tandem iteration",
            "observed": [
                {
                    "iteration": result.iteration,
                    "setup_index": result.setup_update.setup_iteration,
                    "alpha": result.setup_update.alpha,
                    "expected": setup_alpha(result.iteration),
                }
                for result in results
            ],
            "reference_points": {
                "1": setup_alpha(1),
                "2": setup_alpha(2),
                str(production.total_iterations): setup_alpha(
                    production.total_iterations
                ),
            },
        },
    )

    # -- 8. a paired checkpoint round trip ----------------------------------
    #
    # Read the log before the resume: `TrainingSession.step` checkpoints before
    # it appends, so a resumed writer truncates the row of the iteration it
    # resumed from and re-emits from there. That is the accepted Agent 4
    # behavior and is recorded, not changed.
    rows = read_rows(session.telemetry.path)
    checkpoint_path = steps[-1]["checkpoint"]["path"]
    reread = read_joint_checkpoint(
        checkpoint_path,
        run_id=config.run_id,
        config_digest=session.config_digest,
        source_digest=session.source_digest,
    )
    before = runner.identity_document()
    buffer_before = runner.setup_trainer.queue.telemetry(
        runner.setup_trainer.setup_iteration
    ).__dict__
    resumed_session = TrainingSession(
        config,
        directory=directory,
        supervisor_mode=MODE_PRODUCTION,
        source_digest=sources["closure_digest"],
        schedule_digest=frozen["schedule"]["schedule_digest"],
        reading_every=0,
    )
    restore_report = resumed_session.resume(checkpoint_path)
    after = resumed_session.runner.identity_document()
    resumed_step = resumed_session.step(checkpoint=False)
    resumed_session.close()
    identity_fields = (
        "move_raw_model_state_digest",
        "move_ema_model_state_digest",
        "setup_raw_model_state_digest",
        "setup_ema_model_state_digest",
        "cell_digest",
        "iteration",
        "setup_start_model_state_digest",
    )
    differing = [name for name in identity_fields if before[name] != after[name]]
    resumed_buffer = resumed_session.runner.setup_trainer.queue
    checks.record(
        8,
        "a paired checkpoint reloads without identity loss or duplicated/dropped outcomes",
        not differing
        and reread["recipe"] == SETUP_RECIPE_VERSION
        and reread["setup_behavior_kl"]["adaptive"] is False
        and restore_report["games_restored"] == restore_report["games_reseated"] > 0
        and restore_report["setup_episodes_restored"]
        == restore_report["games_restored"]
        and restore_report["completed_setup_buffer_depth"] == buffer_before["depth"]
        and resumed_buffer.enqueued_count == resumed_buffer.consumed_count
        and resumed_step["result"].window.transitions_harvested == config.move_budget,
        {
            "checkpoint": steps[-1]["checkpoint"],
            "restore_report": restore_report,
            "identity_fields_compared": list(identity_fields),
            "differing_identity_fields": differing,
            "buffer_before": buffer_before,
            "buffer_after_one_resumed_iteration": resumed_buffer.telemetry(
                resumed_session.runner.setup_trainer.setup_iteration
            ).__dict__,
            "recipe_in_checkpoint": reread["recipe"],
            "setup_behavior_kl_in_checkpoint": reread["setup_behavior_kl"],
            "telemetry_rows_before_resume": len(rows),
            "telemetry_rows_after_one_resumed_iteration": len(
                read_rows(session.telemetry.path)
            ),
            "telemetry_position_in_checkpoint": reread["telemetry_position"],
            "exactness_note": (
                "identity and outcome accounting, which is what D10 check 8 "
                "asks for. Bitwise continuation equivalence is proved on CPU by "
                "tests/training/phase17/test_runner_tandem.py::"
                "test_a_round_trip_reproduces_the_next_iteration_exactly; MPS is "
                "not bitwise reproducible run to run, so it is not asserted here"
            ),
        },
    )

    # -- 9. no prohibited training participant ------------------------------
    ledger = runner.collector.participant_ledger()
    reachable = _import_scan()
    checks.record(
        9,
        "search, belief, historical and handcrafted training participants are absent",
        ledger["holds"]
        and not ledger["unknown_model_states"]
        and ledger["search_participants"] == 0
        and ledger["historical_participants"] == 0
        and ledger["rule_or_stress_decisions"] == 0
        and not reachable["offending_modules"]
        and not runner.supervisor.should_stop,
        {
            "participant_ledger": ledger,
            "import_scan": reachable,
            "belief_loss_weight": 0.0,
            "supervisor_stopped": runner.supervisor.stop_record(),
        },
    )

    reading = steps[-1].get("reading")
    session.close()
    elapsed = time.perf_counter() - started

    return {
        "artifact": "phase17_simple_tandem_preflight",
        "recipe": SETUP_RECIPE_VERSION,
        "production_run_id": PRODUCTION_RUN_ID,
        "smoke_run_id": config.run_id,
        "smoked_utc": utc_now(),
        "note": (
            "the D10 section 6 tandem smoke on the production code path. No "
            "strength comparison, no diversity certification, no gate campaign, "
            "and the 12-hour run was not started. These weights are discarded."
        ),
        "commit": git("rev-parse", "HEAD"),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "move_device": config.move_device,
            "setup_device": config.setup_device,
        },
        "configuration": config.document(),
        "production_configuration": production.document(),
        "config_digest": session.config_digest,
        "source_closure": sources,
        "schedule_digest": frozen["schedule"]["schedule_digest"],
        "telemetry_schema": telemetry_schema()["schema_version"],
        "iterations": arguments.iterations,
        "checks": checks.entries,
        "passed": len(checks.entries) - len(checks.failed),
        "total": len(checks.entries),
        "all_passed": checks.all_passed,
        "h0_candidate": h0,
        "h0_reverification": verified,
        "telemetry_rows": len(rows),
        "warnings": [
            warning
            for row in rows
            for warning in row["system"]["warnings"]
        ],
        "stop_predicates_fired": [
            entry for row in rows for entry in row["system"]["stop_predicates"]
        ],
        "advantage_components": [
            {
                "iteration": step["result"].iteration,
                **{
                    key: step["row"]["setup"]["advantage_components"].get(key)
                    for key in (
                        "outcome_term_abs_mean",
                        "entropy_term_abs_mean",
                        "entropy_to_outcome_abs_ratio",
                        "information_nats_mean",
                        "predicted_conditional_entropy_mean",
                        "alpha",
                    )
                },
            }
            for step in steps
            if step["row"]["setup"]["advantage_components"]
        ],
        "descriptive_setup_reading": reading,
        "seconds": {
            "total": elapsed,
            "cap": SMOKE_CAP_SECONDS,
            "per_iteration": [result.seconds["total"] for result in results],
            "mean_iteration": sum(result.seconds["total"] for result in results)
            / max(len(results), 1),
        },
        "peak_memory_mib": peak_memory_mib(),
        "output_directory": str(directory.relative_to(REPOSITORY_ROOT)),
        "weights_discarded": True,
    }


#: Search and handcrafted policy packages. Loaded anywhere in the process is
#: enough to fail: the tandem runner has no legitimate reason to reach either.
FORBIDDEN_RUNTIME_ROOTS = ("stratego.search", "stratego.policies")

#: Sources of pre-made setups and of belief targets. These are checked on the
#: phase17 package's own import graph rather than on `sys.modules`, because
#: unrelated accepted modules elsewhere in the process legitimately import
#: them -- a process-global check would flag the engine's own neighbours and
#: say nothing about what the training path reaches.
FORBIDDEN_PACKAGE_IMPORTS = (
    "stratego.setups.library",
    "stratego.setups.sampler",
    "stratego.setups.perturbation",
    "stratego.evaluation.setup_bank",
    "stratego.evaluation.phase10_banks",
    "stratego.training.phase10_selector",
    "stratego.training.phase16.setups",
    "stratego.belief.phase11b.corpus",
    "stratego.belief.phase15.setups",
)

#: The accepted Phase 15 orientation helper genuinely lives under
#: `stratego.belief.*` and the setup half is REQUIRED to import it rather than
#: restate the rule. Named so the scan reports a result instead of a false
#: positive.
ALLOWED_BELIEF_IMPORT = "stratego.belief.phase15.orientation"


def _phase17_package_imports() -> "dict[str, set]":
    """Every module name the phase17 package's own source imports."""
    import ast

    package = REPOSITORY_ROOT / "stratego" / "training" / "phase17"
    found: "dict[str, set]" = {}
    for path in sorted(package.glob("*.py")):
        names: set = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = "stratego.training.phase17"
                if node.level:
                    parts = prefix.split(".")
                    prefix = ".".join(parts[: len(parts) - (node.level - 1)] or ["stratego"])
                module = f"{prefix}.{node.module}" if node.module else prefix
                names.add(module)
                names.update(f"{module}.{alias.name}" for alias in node.names)
        found[path.name] = names
    return found


def _import_scan() -> dict:
    """What the tandem training path can reach, at runtime and in source."""
    import importlib

    importlib.import_module("stratego.training.phase17.runner")
    loaded = sorted(name for name in sys.modules if name.startswith("stratego"))
    runtime_offenders = [
        name
        for name in loaded
        if any(name.startswith(root) for root in FORBIDDEN_RUNTIME_ROOTS)
    ]
    package = _phase17_package_imports()
    source_offenders = {
        module: sorted(
            name
            for name in names
            if any(name.startswith(root) for root in FORBIDDEN_PACKAGE_IMPORTS)
            and not name.startswith(ALLOWED_BELIEF_IMPORT)
        )
        for module, names in package.items()
    }
    source_offenders = {k: v for k, v in source_offenders.items() if v}
    return {
        "forbidden_runtime_roots": list(FORBIDDEN_RUNTIME_ROOTS),
        "forbidden_package_imports": list(FORBIDDEN_PACKAGE_IMPORTS),
        "allowed_exception": ALLOWED_BELIEF_IMPORT,
        "allowed_exception_reason": (
            "the accepted Phase 15 orientation helper; the setup half must "
            "import it rather than re-derive the rule"
        ),
        "stratego_modules_loaded": len(loaded),
        "phase17_modules_scanned": len(package),
        "runtime_offenders": runtime_offenders,
        "source_offenders": source_offenders,
        "offending_modules": runtime_offenders + sorted(source_offenders),
    }


# ---------------------------------------------------------------------------
# The handoff
# ---------------------------------------------------------------------------


def handoff(arguments) -> dict:
    from stratego.training.phase17.checkpoint import JOINT_CHECKPOINT_SCHEMA_VERSION
    from stratego.training.phase17.export import EXPORT_SCHEMA_VERSION
    from stratego.training.phase17.setup_contract import (
        PRODUCTION_RUN_ID,
        SETUP_BEHAVIOR_KL_COEFFICIENT,
        SETUP_BUFFER_VERSION,
        SETUP_EQUATION_VERSION,
        SETUP_RECIPE_VERSION,
        setup_alpha,
    )
    from stratego.training.phase17.supervisor import SUPERVISOR_VERSION
    from stratego.training.phase17.telemetry import telemetry_schema

    preflight_path = REPORT_DIRECTORY / "phase17_simple_tandem_preflight.json"
    if not preflight_path.is_file():
        raise SystemExit(
            f"no preflight at {preflight_path}; run the smoke before the handoff"
        )
    preflight = json.loads(preflight_path.read_text())
    frozen = json.loads((REPOSITORY_ROOT / arguments.schedule).read_text())

    return {
        "artifact": "phase17_simple_tandem_handoff_v1",
        "written_utc": utc_now(),
        "work_package": "phase17",
        "recipe": SETUP_RECIPE_VERSION,
        "production_run_id": PRODUCTION_RUN_ID,
        "integration_base": "eab8a33",
        "supersedes": {
            "decision": "D10",
            "retired": [
                "the adaptive setup-KL controller and all target/beta-bound logic (D5)",
                "the endpoint-re-horizoned setup entropy schedule (D3)",
                "phase17_setup_update_v2 and its uncentered normalized bonus (D7-B)",
                "the standalone setup-diversity pass/fail gate",
                "the fixed setup-episode quota, warm-up gate and age calibration",
                "broad preflight certification beyond the D10 smoke",
            ],
            "retained_from_agent_4": [
                "exact Phase 9 move start and fresh optimizer/schedule/EMA state",
                "fixed-transition move collection and boundary bootstrapping",
                "sampled legal moves from the current raw move policy on both seats",
                "current-policy rebind for active games",
                "fresh autoregressive setup generation and outcome binding",
                "paired raw/EMA checkpointing and exact active-game persistence",
                "exports, telemetry and integrity-oriented safety stops",
                "no-search / no-belief / no-training-opponent boundaries",
                "the external 30-minute evaluation interface",
            ],
        },
        "identities": {
            "runner_version": "phase17_tandem_runner_v2",
            "setup_equation_version": SETUP_EQUATION_VERSION,
            "completed_setup_buffer_version": SETUP_BUFFER_VERSION,
            "joint_checkpoint_schema": JOINT_CHECKPOINT_SCHEMA_VERSION,
            "export_schema": EXPORT_SCHEMA_VERSION,
            "telemetry_schema": telemetry_schema()["schema_version"],
            "supervisor_version": SUPERVISOR_VERSION,
            "config_digest": preflight["config_digest"],
            "source_closure_digest": preflight["source_closure"]["closure_digest"],
            "schedule_digest": frozen["schedule_digest"],
            "commit": preflight["commit"],
        },
        "start": {
            "move_checkpoint": "checkpoints/phase9/selfplay_c1_v1.pt",
            "move_file_sha256": PHASE9_MOVE_FILE_SHA256,
            "move_model_state_digest": PHASE9_MOVE_STATE_DIGEST,
            "setup_initialization": "from scratch under the recorded seed",
            "setup_model_seed": preflight["production_configuration"]["setup"][
                "model_seed"
            ],
            "no_rehearsal_state": (
                "the runner refuses an injected setup model under RUN-2026-B, and "
                "read_joint_checkpoint refuses any checkpoint whose run ID or "
                "recipe differs"
            ),
        },
        "recipe_constants": {
            "move": {
                "budget_transitions": preflight["production_configuration"]["move"][
                    "budget_transitions"
                ],
                "population": preflight["production_configuration"]["move"][
                    "population"
                ],
                "epochs_per_iteration": 1,
                "ema_decay": 0.999,
                "schedule": "unchanged Phase 17 move LR/entropy/KL behavior",
                "horizon_N": frozen["N"],
                "n_ref": frozen["n_ref"],
            },
            "setup": {
                "optimizer": "Adam",
                "learning_rate": 5e-05,
                "epochs_per_iteration": 5,
                "ppo_clip_epsilon": 0.2,
                "value_loss_weight": 0.5,
                "conditional_entropy_loss_weight": 1.0,
                "conditional_entropy_loss_target": "I/10",
                "gradient_clip_norm": 0.5,
                "ema_decay": 0.999,
                "behavior_kl_direction": "reverse_current_given_behavior",
                "behavior_kl_coefficient": SETUP_BEHAVIOR_KL_COEFFICIENT,
                "behavior_kl_adaptive": False,
                "alpha_formula": "0.1 * n**-0.3",
                "alpha_index": "the shared one-based global tandem iteration",
                "alpha_at_1": setup_alpha(1),
                "alpha_at_2": setup_alpha(2),
                "alpha_at_N": setup_alpha(frozen["N"]),
                "advantage": "(outcome - E[behavior W/D/L value]) + alpha(n) * (I - h_behavior)",
                "pool_size_per_side": 512,
                "pool_cadence": "regenerated at every global tandem iteration",
                "consumption": (
                    "every episode whose game completed in the iteration, both "
                    "sides, exactly once"
                ),
            },
        },
        "stop_policy": {
            "stops": {
                "I1": "rules, orientation, legality, candidate or digest mismatch",
                "I2": "a decision recorded under the wrong current move-policy digest",
                "I3": "nonfinite loss, gradient, parameter or schedule value",
                "I4": "setup generation/masking failure or silent fallback attempt",
                "I5": "search or a non-current training opponent entered collection",
                "I6": "evaluation result bound to the wrong candidate or benchmark",
                "I7": "unrecoverable checkpoint/resume identity failure",
                "I8": "fixed-transition count violation",
            },
            "warnings_only": {
                "P1": "fixed-pack EWR decline",
                "P2": "move mean KL above 0.08",
                "P3": "setup reverse KL above 0.08",
                "P4": "setup prefix entropy below 60% of its initial baseline",
                "P5": "flag effective support below 4",
                "P6": "move entropy below 25% of its first-hour median",
                "P7": "no setup update for a whole cadence interval",
            },
            "not_wired": {
                "resource_exhaustion": (
                    "D10 section 7 names it a stop condition, but there is no "
                    "threshold to freeze and the runner has no resource monitor. "
                    "It surfaces as a process-level failure, unchanged from "
                    "Agent 4. Recorded rather than invented."
                )
            },
        },
        "preflight": {
            "path": "reports/phase17/phase17_simple_tandem_preflight.json",
            "sha256": file_sha256(preflight_path),
            "all_passed": preflight["all_passed"],
            "passed": preflight["passed"],
            "total": preflight["total"],
            "seconds": preflight["seconds"]["total"],
            "cap_seconds": SMOKE_CAP_SECONDS,
            "checks": [
                {"check": entry["check"], "name": entry["name"], "ok": entry["ok"]}
                for entry in preflight["checks"]
            ],
        },
        "measured_in_the_smoke": {
            "iterations": preflight["iterations"],
            "seconds_per_iteration": preflight["seconds"]["per_iteration"],
            "mean_iteration_seconds": preflight["seconds"]["mean_iteration"],
            "frozen_mean_iteration_seconds": 67.40442866057025,
            "peak_memory_mib": preflight["peak_memory_mib"],
            "advantage_components": preflight["advantage_components"],
            "advantage_balance_note": (
                "the printed advantage's entropy term is several times the "
                "outcome term, which is the deliberate consequence of D10 "
                "section 4 keeping `alpha * (I - h)` with I in nats while L_h "
                "targets I/10. Recorded per iteration, not compensated for."
            ),
        },
        "known_issues": [
            {
                "id": "A4B-1",
                "status": "FIXED",
                "severity": "would have killed any resumed production run",
                "summary": (
                    "SetupEMA.load_state_dict preserved the checkpoint payload's "
                    "device. A paired checkpoint serializes the setup EMA to CPU "
                    "and read_joint_checkpoint maps to CPU, so a resume on the "
                    "MPS production device left a CPU shadow accumulating "
                    "against MPS parameters and SetupEMA.update raised on the "
                    "first setup update after the resume."
                ),
                "why_it_survived": (
                    "Agent 4's resume rehearsal ran on CPU on purpose, because "
                    "MPS is not bitwise reproducible run to run, so the device "
                    "mismatch could not occur there. The D10 smoke is the first "
                    "thing to resume on the production device."
                ),
                "found_by": "the D10 smoke, check 8, first attempt",
                "fix": (
                    "SetupEMA binds the shadow to the model's device on load"
                ),
                "pinned_by": [
                    "tests/training/phase17/test_setup_learning.py::"
                    "test_a_restored_ema_lands_on_the_models_device_not_the_payloads",
                    "tests/training/phase17/test_setup_learning.py::"
                    "test_a_restored_ema_lands_on_the_accelerator_when_there_is_one",
                ],
            },
            {
                "id": "A4B-2",
                "status": "REPORTED, NOT CHANGED",
                "severity": "telemetry only; training and checkpoints unaffected",
                "summary": (
                    "one telemetry row is lost per resume. TrainingSession.step "
                    "writes the checkpoint BEFORE it appends the row, so the "
                    "checkpoint's telemetry_position excludes the row of the "
                    "iteration it was taken at. TelemetryWriter.resume then "
                    "truncates that row back and the next iteration writes into "
                    "its record slot."
                ),
                "measured": (
                    "the smoke's log holds iterations 1, 2, 4 at record indices "
                    "0, 1, 2; iteration 3's row is gone although its weights "
                    "were checkpointed and restored"
                ),
                "consequence": (
                    "the hour 6-12 learning curve loses one row per resume and "
                    "record_index no longer tracks the iteration number. Every "
                    "row carries system.iteration, so a reader realigns by that "
                    "field rather than by position."
                ),
                "not_changed_because": (
                    "this is Agent 4's accepted checkpoint/append ordering and "
                    "the row's own content includes the checkpoint identity, so "
                    "reversing the order is a restructure of machinery the 4B "
                    "brief says to reuse rather than rebuild. Operator call."
                ),
            },
        ],
        "tests": source_closure(AGENT_4B_TESTS),
        "sources": source_closure(AGENT_4B_SOURCES),
        "production_command": (
            "nohup caffeinate -dimsu .venv/bin/python scripts/run_phase17_training.py "
            f"--run-id {PRODUCTION_RUN_ID} --start --i-am-agent-7 "
            "> <log> 2>&1 &"
        ),
        "not_established": [
            "any strength claim: no benchmark lane was evaluated",
            "the external 30-minute round trip, which is Agent 5's h0 handshake",
            "setup diversity or concentration behavior over 12 hours",
            "whether the printed advantage's entropy term dominates in practice; "
            "its magnitude is recorded per iteration and is the experiment",
        ],
        "ready_for_short_launch_check": bool(preflight["all_passed"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("smoke", "handoff"))
    parser.add_argument("--run-id", default="RUN-SMOKE-D10")
    parser.add_argument("--directory", default="checkpoints/phase17/smoke_d10")
    parser.add_argument("--schedule", default="reports/phase17/agent_04_schedule.json")
    parser.add_argument(
        "--throughput", default="reports/phase17/agent_04_throughput.json"
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--setup-device", default=None)
    arguments = parser.parse_args()

    if arguments.role == "smoke":
        document = smoke(arguments)
        path = write_json("phase17_simple_tandem_preflight.json", document)
        print(
            f"\n{document['passed']}/{document['total']} checks passed in "
            f"{document['seconds']['total']:.1f}s -> {path}"
        )
        return 0 if document["all_passed"] else 1

    document = handoff(arguments)
    path = write_json("phase17_simple_tandem_handoff_v1.json", document)
    print(
        f"ready_for_short_launch_check={document['ready_for_short_launch_check']} "
        f"-> {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
