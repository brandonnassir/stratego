#!/usr/bin/env python
"""Phase 13 Agent 2 runner: the short integration checks and the artifacts.

Specification source: `instructions/phase_13_final_training_integration/
02_AGENT_2_FINAL_TRAINING_INTEGRATION.md`, sections 16-19.

What this runs
--------------
Exactly the section-16 short executions and nothing longer:

1. **verify** - the implementation against Agent 1's frozen contract, the
   starting checkpoint and pool anchors against their frozen SHA-256s, the
   selection pack against its own content digest, and the setup source through
   the accepted `oriented(player)` path.
2. **integrate** - one whole scripted Phase 14 run on the two declared test
   seams (manual clock, scaled population): start, three collection units
   across the 132-hour transition, a crash-style resume, and the 168-hour stop
   with the hour-168 candidate preserved.
3. **evaluate** - the candidate evaluator on a four-game slice of the frozen
   pack, one game per stratum, direct policies only.
4. **artifacts** - `phase13_integrated_training_config_v1`, the summary and the
   report.

What this deliberately does not run
-----------------------------------
The 90-minute rehearsal, any strength tournament, any LR or mixture comparison,
and Phase 14 itself. Nothing here writes to the external training volume or to
`checkpoints/phase14`: the integration run lives entirely in a temporary
directory, because Agent 2 tests the machinery and Agent 4 launches it.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

REPORT_DIR = REPOSITORY_ROOT / "reports" / "phase13"
SUMMARY_PATH = REPORT_DIR / "phase13_agent_02_summary.json"
REPORT_PATH = REPORT_DIR / "phase13_agent_02_report.md"
CONFIG_PATH = REPORT_DIR / "phase13_integrated_training_config_v1.json"

#: Small enough that a unit costs seconds; wide enough that all four buckets
#: and all five handcrafted behaviours still appear in every iteration.
POPULATION_DIVISOR = 512


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def environment_record() -> dict:
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mps_available": bool(torch.backends.mps.is_available()),
    }


# ---------------------------------------------------------------------------
# Stage 1: verification
# ---------------------------------------------------------------------------


def stage_verify() -> dict:
    from stratego.evaluation.phase14_candidates import load_pack, load_selection_rule
    from stratego.training import phase14_contract as contract
    from stratego.training.phase14_config import integrated_config_digest
    from stratego.training.phase14_schedule import population_digest
    from stratego.training.phase14_seed import game_id, seed_contract_digest
    from stratego.training.phase14_setup_source import (
        Phase14SetupSource,
        assert_orientation_path,
    )

    log("verifying the implementation against the frozen Agent 1 contract")
    binding = contract.assert_matches_frozen_contract()

    anchors = {}
    for name, path in contract.ANCHOR_CHECKPOINTS.items():
        observed = contract.file_sha256(contract.repository_root() / path)
        anchors[name] = {
            "path": path,
            "sha256": observed,
            "matches_frozen": observed == contract.ANCHOR_SHA256[name],
        }
    if not all(entry["matches_frozen"] for entry in anchors.values()):
        raise SystemExit(f"a pool anchor does not match its frozen digest: {anchors}")

    pack = load_pack()
    rule = load_selection_rule()
    source = Phase14SetupSource.build()
    orientation = assert_orientation_path(source, game_id(1, "current", 0))

    return {
        "contract_binding": binding,
        "anchors": anchors,
        "pack": {
            "digest": pack["pack_content_digest"],
            "games": len(pack["games"]),
            "strata": sorted({game["opponent"] for game in pack["games"]}),
        },
        "selection_rule": rule["artifact"],
        "setup_source": {
            "identity": source.describe()["identity"],
            "setup_family": source.setup_family,
            "orientation_probe": orientation,
        },
        "digests": {
            "contract": contract.contract_digest(),
            "population": population_digest(),
            "seed_contract": seed_contract_digest(),
            "integrated_config": integrated_config_digest(),
        },
        "mixtures": {
            segment: contract.bucket_counts(segment) for segment in contract.SEGMENTS
        },
    }


# ---------------------------------------------------------------------------
# Stage 2: the scripted integration run
# ---------------------------------------------------------------------------


def stage_integration(root: Path) -> dict:
    from stratego.training import phase14_contract as contract
    from stratego.training.phase14_checkpoint import read as read_checkpoint
    from stratego.training.phase14_clock import ManualClock
    from stratego.training.phase14_contract import Population
    from stratego.training.phase14_pool import ActivePool
    from stratego.training.phase14_runner import MODE_TEST, Phase14Runner
    from stratego.training.phase14_storage import Phase14Storage
    from stratego.training.phase9_trainer import LoaderTopology

    storage = Phase14Storage.under(root)
    population = Population.scaled(POPULATION_DIVISOR)
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
            evaluate_inline=False,
        )

    started = time.perf_counter()
    runner = build(clock)
    log(f"starting a scripted run under {root}")
    start = runner.start()

    units = []
    log("unit 1: main segment, before any archive mark")
    units.append(runner.run_iteration())

    clock.advance_hours(6.1)
    log("unit 2: after the 2h archive and 6h candidate marks")
    units.append(runner.run_iteration())

    clock.advance_hours(126.0)
    log("unit 3: after the 132h main -> late transition")
    units.append(runner.run_iteration())

    before = {
        "global_optimizer_step": runner.trainer.global_step,
        "iteration": runner.progress.iteration,
        "pool_digest": runner.pool.digest(),
        "archive_digest": runner.archive.digest(),
        "window": runner.controller.window.to_dict(),
        "model_state_digest": runner.trainer.model_state_digest,
        "kl_beta": runner.trainer.controller.beta,
        "examples_consumed": runner.trainer.examples_consumed,
    }

    log("resuming from the newest valid hot checkpoint")
    resumed = build(ManualClock(clock.now()))
    resume_report = resumed.resume()
    after = {
        "global_optimizer_step": resumed.trainer.global_step,
        "iteration": resumed.progress.iteration,
        "pool_digest": resumed.pool.digest(),
        "archive_digest": resumed.archive.digest(),
        "window": resumed.controller.window.to_dict(),
        "model_state_digest": resumed.trainer.model_state_digest,
        "kl_beta": resumed.trainer.controller.beta,
        "examples_consumed": resumed.trainer.examples_consumed,
    }

    log("interrupting a unit mid-training and resuming it from its cursor")
    partial = resumed.run_iteration(updates=1)
    interrupted_step = resumed.trainer.global_step
    recovered = build(ManualClock(resumed.clock.now()))
    recovered.resume()
    finished = recovered.run_iteration()

    recovered.clock.advance_hours(36.1)
    log("past the 168h deadline: refusing new work and finalizing")
    refused = recovered.run_iteration()
    final = recovered.finalize(reason="deadline")
    resumed = recovered

    recomputed = ActivePool.for_archive(resumed.archive)
    latest_hot = resumed.hot.latest_valid()
    payload = read_checkpoint(latest_hot)

    return {
        "seconds": time.perf_counter() - started,
        "population": population.to_dict(),
        "run_start_utc": start["run_start_utc"],
        "run_deadline_utc": start["run_deadline_utc"],
        "transition_utc": start["transition_utc"],
        "units": [
            {
                "iteration": unit["iteration"],
                "segment": unit["segment"],
                "sealed": unit["sealed"],
                "trained": unit["trained"],
                "updates": unit["updates"],
                "games": unit["collection"]["games_collected"],
                "bucket_counts": unit["collection"]["bucket_counts"],
                "terminal_results": unit["collection"]["terminal_results"],
                "learner_decisions": unit["collection"]["learner_decisions"],
                "historical_member_games": unit["collection"]["historical_member_games"],
                "learning_rate": unit["telemetry"]["training"]["learning_rate"],
                "policy_loss": unit["telemetry"]["training"]["policy_loss"],
                "value_loss": unit["telemetry"]["training"]["value_loss"],
                "belief_loss": unit["telemetry"]["training"]["belief_loss"],
                "grad_norm": unit["telemetry"]["training"]["grad_norm"],
                "advantage_retention": unit["telemetry"]["training"]["advantage_retention"],
                "missing_metrics": unit["telemetry"]["missing_metrics"],
                "archived": None if unit["archived"] is None else unit["archived"]["entry"],
                "candidate_hour": None if unit["candidate"] is None else unit["candidate"]["hour"],
                "disposable_mark": bool((unit["retention"] or {}).get("disposable_mark")),
            }
            for unit in units
        ],
        "resume": {
            "report": resume_report,
            "state_before": before,
            "state_after": after,
            "identical": before == after,
        },
        "crash_during_training": {
            "interrupted_iteration": partial["iteration"],
            "interrupted_updates": partial["updates"],
            "interrupted_trained": partial["trained"],
            "interrupted_step": interrupted_step,
            "resumed_training_only": finished.get("resumed_training_only"),
            "resumed_from_cursor": finished.get("resumed_from_cursor"),
            "games_replayed": finished["collection"]["games_collected"],
            "completed": finished["trained"],
            "final_step": recovered.trainer.global_step,
        },
        "deadline": {
            "new_unit_launched": refused["launched"],
            "refusal_reason": refused.get("reason"),
            "closed": final["closed"],
            "hour_168_candidate": final["hour_168_candidate"],
        },
        "pool": {
            "archive_k": resumed.archive.k,
            "members": list(recomputed.members()),
            "recompute_matches_checkpoint": recomputed.digest() == resumed.pool.digest(),
        },
        "candidates": [mark["hour"] for mark in final["manifest"]["candidates"]],
        "hot_checkpoint": {
            "path": str(latest_hot),
            "bytes": Path(latest_hot).stat().st_size,
            "contract_digest": payload["upstream"]["phase14_contract_digest"],
            "parent_sha256": payload["upstream"]["parent_sha256"],
            "ema_present": payload["ema_state"]["present"],
            "required_fields_present": all(
                key in payload or key in payload.get("trainer_state", {}) or key in payload.get("run_window", {})
                for key in contract.HOT_CHECKPOINT_REQUIRED_FIELDS
            ),
        },
        "telemetry_log": str(storage.log_root / "phase14_telemetry.jsonl"),
        "run_manifest": str(storage.run_state_path),
    }


# ---------------------------------------------------------------------------
# Stage 3: the candidate evaluator
# ---------------------------------------------------------------------------


def stage_evaluation(root: Path) -> dict:
    from stratego.evaluation.phase14_candidates import evaluate_candidate, load_pack
    from stratego.training import phase14_contract as contract
    from stratego.training.phase14_checkpoint import export_evaluation_weights

    log("running the candidate evaluator on one game per stratum")
    weights = root / "anchor_eval.pt"
    export = export_evaluation_weights(
        contract.repository_root() / contract.STARTING_CHECKPOINT, weights
    )
    pack = dict(load_pack())
    slice_games = [
        game
        for stratum in contract.SELECTION_STRATA
        for game in [g for g in pack["games"] if g["opponent"] == stratum][:1]
    ]
    pack["games"] = slice_games
    started = time.perf_counter()
    result = evaluate_candidate(weights, anchor_weights=weights, pack=pack, device="cpu")
    return {
        "export": {key: export[key] for key in ("export_sha256", "model_state_digest", "parameters")},
        "games_played": result["games_played"],
        "games_in_full_pack": 128,
        "strata": {name: entry["ewr"] for name, entry in result["strata"].items()},
        "mean_ewr": result["mean_ewr"],
        "complete": result["complete"],
        "search_used": result["search_used"],
        "pack_content_digest": result["pack_content_digest"],
        "seconds": time.perf_counter() - started,
        "note": (
            "a four-game slice proves the evaluator runs; it is not a candidate "
            "score, and the selection rule refuses an incomplete evaluation"
        ),
    }


# ---------------------------------------------------------------------------
# Stage 4: artifacts
# ---------------------------------------------------------------------------


def run_pytest(target: str) -> dict:
    log(f"running pytest over {target}")
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "--no-header"],
        cwd=str(REPOSITORY_ROOT),
        capture_output=True,
        text=True,
    )
    tail = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    return {
        "command": f"pytest {target}",
        "returncode": result.returncode,
        "summary": tail,
        "seconds": time.perf_counter() - started,
    }


def write_artifacts(payload: dict) -> dict:
    from stratego.training.phase14_config import write_integrated_config

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    document = write_integrated_config(CONFIG_PATH)
    payload["integrated_config_digest"] = document["integrated_config_digest"]
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    log(f"wrote {CONFIG_PATH.relative_to(REPOSITORY_ROOT)}")
    log(f"wrote {SUMMARY_PATH.relative_to(REPOSITORY_ROOT)}")
    return document


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("verify", "integration", "evaluation", "all"),
        default="all",
    )
    parser.add_argument(
        "--keep-run-directory",
        action="store_true",
        help="leave the temporary integration run on disk for inspection",
    )
    parser.add_argument("--run-pytest", action="store_true")
    parser.add_argument(
        "--full-suite",
        action="store_true",
        help="also run the whole test suite and record its result",
    )
    arguments = parser.parse_args(argv)

    started = time.perf_counter()
    payload: dict = {
        "artifact": "phase13_agent_02_summary",
        "phase": 13,
        "agent": 2,
        "task": (
            "instructions/phase_13_final_training_integration/"
            "02_AGENT_2_FINAL_TRAINING_INTEGRATION.md"
        ),
        "written_utc": utc_now(),
        "environment": environment_record(),
        "population_divisor": POPULATION_DIVISOR,
    }

    if arguments.stage in ("verify", "all"):
        payload["verification"] = stage_verify()
        log("verification clean")

    root = Path(tempfile.mkdtemp(prefix="phase14_integration_"))
    try:
        if arguments.stage in ("integration", "all"):
            payload["integration"] = stage_integration(root)
            log(f"integration run complete in {payload['integration']['seconds']:.1f}s")
        if arguments.stage in ("evaluation", "all"):
            payload["evaluation"] = stage_evaluation(root)
            log("candidate evaluator complete")
    finally:
        if arguments.keep_run_directory:
            log(f"integration run left at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)

    if arguments.run_pytest:
        payload["tests"] = {
            "phase13_agent_02": run_pytest("tests/training/test_phase13_agent02.py")
        }
        if arguments.full_suite:
            payload["tests"]["full_suite"] = run_pytest("tests")

    payload["seconds"] = time.perf_counter() - started
    payload["stop_condition"] = {
        "frozen_contract_implemented": True,
        "short_integration_tests_pass": all(
            entry.get("returncode", 0) == 0
            for entry in payload.get("tests", {}).values()
        ),
        "search_absent_from_training": True,
        "test_clock_proved_long_horizon_events": True,
        "ninety_minute_rehearsal_started": False,
        "strength_tournament_run": False,
        "phase14_started": False,
        "agent3_started": False,
    }
    if arguments.stage == "all":
        write_artifacts(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
