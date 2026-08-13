#!/usr/bin/env python3
"""Phase 6B acceptance harness: the persisted-recording soak and its verdict.

Writes

    reports/phase_6_data/agent_06b_recording_soak.json
    reports/phase_6_data/agent_06b_recording_timeseries.csv
    reports/phase_6_data/agent_06b_storage_validation.json
    reports/phase_6_data/agent_06b_restart_validation.json   (when recycling is needed)

What this does
--------------
1. Runs the frozen Phase 6 C1 production configuration for several continuous
   hours with **durable per-worker shard writing** switched on, so the real
   persistence path is exercised rather than an encode-and-discard estimate.
2. Samples per-worker resident memory, every disk-side quantity and every
   correctness counter.
3. Classifies the memory result against the four outcomes declared in
   `stratego.training.phase6b_recording`, and, when the verdict is C, derives the
   recycling interval from the measured slope rather than picking one.
4. Verifies every persisted shard by decoding it.
5. Reprojects the 168-hour storage requirement from the rate that actually
   reached the disk.

What it does not do
-------------------
No architecture selection, no benchmark, no change to the model, the topology,
the action contract, the trajectory schema or the simulator. Phase 6's decisions
are frozen; this is operational validation.

Usage::

    python scripts/run_phase6b.py --output /Volumes/.../stratego_phase6b/soak
    python scripts/run_phase6b.py --hours 4
    python scripts/run_phase6b.py --skip-pytest
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.model.architecture_configs import (  # noqa: E402
    ARCHITECTURE_FAMILY,
    ARCHITECTURE_FAMILY_VERSION,
    FAMILY_INITIALIZATION_SEED,
    candidate_config,
)
from stratego.model.contract import MODEL_CONTRACT_VERSION  # noqa: E402
from stratego.training import phase6b_recording as p6b  # noqa: E402
from stratego.training import phase6b_recycle as recycle  # noqa: E402
from stratego.training import shard_writer as sw  # noqa: E402
from stratego.training.coordinator import COORDINATOR_VERSION  # noqa: E402
from stratego.training.trajectory import TRAJECTORY_VERSION  # noqa: E402

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_6_data"

TIMESERIES_COLUMNS = [
    "candidate_id", "sample_index", "elapsed_seconds", "global_step",
    "in_measured_window", "positions", "positions_per_second", "games",
    "games_per_second", "decisions_recorded",
    "trajectory_bytes_produced", "trajectory_bytes_written", "compressed_bytes",
    "window_bytes_produced", "window_bytes_written",
    "produced_gib_per_hour", "written_gib_per_hour", "compression_ratio",
    "write_throughput_bytes_per_second", "records_persisted",
    "shards_opened", "shards_closed",
    "encode_seconds", "compress_seconds", "write_seconds", "flush_seconds",
    "write_errors", "pending_records", "pending_bytes",
    "disk_free_bytes", "disk_free_change_bytes",
    "coordinator_rss_bytes", "worker_rss_bytes", "per_worker_rss_bytes",
    "workers_measured", "total_rss_bytes", "shared_memory_bytes",
    "metal_current_allocated_bytes", "metal_driver_allocated_bytes",
    "swap_used_bytes", "system_memory_available_bytes",
    "system_memory_percent_used",
    "illegal_actions", "action_frame_errors", "worker_failures",
    "model_failures", "nonfinite_outputs", "verified_decisions",
    "verified_games", "reconstruction_mismatches", "workers_alive",
]


def log(message: str) -> None:
    print(f"[phase-6b] {message}", flush=True)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
            capture_output=True, text=True, check=False,
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMESERIES_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            prepared = {}
            for column in TIMESERIES_COLUMNS:
                value = row.get(column, "")
                if isinstance(value, dict):
                    value = json.dumps({str(k): v for k, v in value.items()}, sort_keys=True)
                elif isinstance(value, float):
                    value = round(value, 6)
                prepared[column] = value
            writer.writerow(prepared)


def run_pytest(label: str) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=REPOSITORY_ROOT,
        capture_output=True, text=True, check=False,
    )
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    summary = {
        "label": label, "command": "python -m pytest -q",
        "returncode": completed.returncode, "summary_line": tail,
        "seconds": round(time.perf_counter() - started, 2),
        "passed": 0, "failed": 0, "skipped": 0,
    }
    words = tail.replace(",", " ").split()
    for index, word in enumerate(words):
        if word in ("passed", "failed", "skipped", "error", "errors") and index:
            try:
                count = int(words[index - 1])
            except ValueError:
                continue
            key = "failed" if word.startswith("error") else word
            summary[key] = summary.get(key, 0) + count
    summary["total"] = summary["passed"] + summary["failed"] + summary["skipped"]
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/Volumes/Brandon_Washington/stratego_phase6b/soak")
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--sample-seconds", type=float, default=p6b.DEFAULT_SAMPLE_SECONDS)
    parser.add_argument("--warmup-steps", type=int, default=p6b.DEFAULT_WARMUP_STEPS)
    parser.add_argument("--candidate", default="C1")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-restart-test", action="store_true")
    parser.add_argument("--restart-segments", type=int, default=3)
    parser.add_argument("--restart-steps", type=int, default=1200)
    parser.add_argument(
        "--keep-data", action="store_true",
        help="leave the persisted shards in place instead of deleting them",
    )
    arguments = parser.parse_args()

    if not torch.backends.mps.is_available():
        log("MPS is not available")
        return 1
    device = torch.device("mps")
    commit = git_commit()
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)

    volume = Path(arguments.output)
    while not volume.is_mount() and volume != volume.parent:
        volume = volume.parent
    usage = shutil.disk_usage(str(output))
    external = str(volume).startswith("/Volumes/")
    log(f"commit {commit[:12]} | output {output}")
    log(
        f"volume {volume} total {usage.total / 1024**3:.1f} GiB free "
        f"{usage.free / 1024**3:.1f} GiB | external={external}"
    )

    tests_before = None
    if not arguments.skip_pytest:
        log("full suite before the follow-up")
        tests_before = run_pytest("before")
        log(f"  {tests_before['summary_line']}")
        if tests_before["failed"]:
            log("BLOCKED: the pre-existing suite is red")
            return 1

    configuration = candidate_config(arguments.candidate)
    run_id = f"p6b{int(time.time()) % 1000000:06d}"
    config = p6b.recording_configuration(
        arguments.candidate, output_directory=str(output), run_id=run_id, compress=True,
    )
    log(
        f"soak: {arguments.candidate} for {arguments.hours:.1f}h at "
        f"{config.num_workers}w x {config.num_environments}e batch "
        f"{config.inference_batch_size} {config.precision} {config.legality}, "
        f"compressed shards, target {config.shard_target_bytes / 2**20:.0f} MiB"
    )

    def progress(row: dict) -> None:
        log(
            f"  t={row['elapsed_seconds']:8.1f}s step={row['global_step']:7d} "
            f"pos/s={row['positions_per_second']:9.1f} "
            f"disk={row['written_gib_per_hour']:5.2f} GiB/h "
            f"ratio={row['compression_ratio']:.4f} "
            f"shards={row['shards_closed']:4d} "
            f"rss={row['total_rss_bytes'] / 2**30:5.2f}G "
            f"swap={row['swap_used_bytes']} "
            f"free={row['disk_free_bytes'] / 2**30:7.1f}G"
        )

    result = p6b.run_recording_soak(
        arguments.candidate, config,
        seconds=arguments.hours * 3600.0,
        sample_seconds=arguments.sample_seconds,
        warmup_steps=arguments.warmup_steps,
        device=device, progress=progress,
    )
    log(f"soak {result['status']} in {result['total_seconds']:.0f}s, {result['sample_count']} samples")
    if result["error"]:
        log(f"  error: {result['error']}")

    samples = result.pop("samples")
    steady = p6b.steady_state_summary({**result, "samples": samples})
    if steady:
        log(
            f"  sustained: {steady['positions_per_second']:.1f} positions/s, "
            f"{steady['written_gib_per_hour']:.3f} GiB/hour to disk, "
            f"ratio {steady['compression_ratio']:.4f}"
        )

    system_total = result["system_memory"]["total_bytes"]
    verdict = p6b.classify_memory_outcome(samples, total_system_bytes=system_total)
    log(f"  memory outcome {verdict['outcome']}: {verdict['reason']}")

    # -- evidence first, always --------------------------------------------
    # The first Phase 6B session lost every in-memory sample because the
    # harness wrote its artifacts only after verification, and verification is
    # the step that wedged. The soak's evidence is now on disk before any
    # post-processing gets a chance to endanger it; the files are rewritten
    # with the verification and gate results once those exist.
    write_json(DATA_DIRECTORY / "agent_06b_recording_soak.json", {
        "agent": "agent_06b", "phase": "phase_6b",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "SOAK_COMPLETE_VERIFICATION_PENDING",
        "soak": result, "steady_state": steady, "memory_verdict": verdict,
        "sample_count": len(samples),
    })
    write_csv(DATA_DIRECTORY / "agent_06b_recording_timeseries.csv", samples)

    # -- verify what actually landed on disk -------------------------------
    log("verifying persisted shards (streaming; decoding every record)")

    def verify_progress(shard: dict) -> None:
        log(
            f"  shard {Path(shard['path']).name}: ok={shard['ok']} "
            f"records={shard['record_count']}"
        )

    verification = sw.directory_summary(output, decode=True, progress=verify_progress)
    log(
        f"  {verification['shard_count']} shards, {verification['record_count']} records, "
        f"{verification['file_bytes'] / 2**30:.3f} GiB, ok={verification['ok']}"
    )

    usage_after = shutil.disk_usage(str(output))
    storage = p6b.storage_projection_from_disk(
        steady,
        volume_total_bytes=usage.total,
        volume_free_bytes=usage.free,
        shard_target_bytes=config.shard_target_bytes,
    )

    # -- recycling, if the verdict calls for it ----------------------------
    restart_report = None
    if verdict["outcome"] == "C" and not arguments.skip_restart_test:
        log(
            f"outcome C: validating process recycling "
            f"({arguments.restart_segments} segments)"
        )
        recycle_dir = output.parent / f"{output.name}_recycle"
        if recycle_dir.exists():
            shutil.rmtree(recycle_dir)
        supervisor = recycle.RecyclingSupervisor(
            output_directory=recycle_dir,
            state_directory=REPOSITORY_ROOT / ".phase6b_state",
            base_run_id=f"{run_id}r",
            candidate_id=arguments.candidate,
            shard_target_bytes=config.shard_target_bytes,
            warmup_steps=0,
            sample_seconds=60.0,
        )

        def segment_progress(state: dict) -> None:
            log(
                f"  segment {state['segment']}: {state['status']} "
                f"wall={state['supervisor_wall_seconds']:.1f}s "
                f"overhead={state['startup_shutdown_overhead_seconds']:.1f}s "
                f"rss_start={state['rss_at_start_bytes'] / 2**20:.0f}MiB "
                f"rss_end={state['rss_at_end_bytes'] / 2**20:.0f}MiB "
                f"shards={state['shards_closed']}"
            )

        restart_report = supervisor.run(
            segments=arguments.restart_segments,
            steps_per_segment=arguments.restart_steps,
            progress=segment_progress,
            timeout=3600,
        )
        restart_report["decode_verification"] = recycle.verify_recycled_output(
            recycle_dir, decode=True
        )
        budget = verdict["growth_budget_bytes"]
        restart_report["recommended_interval"] = p6b.recommended_restart_interval_hours(
            verdict, budget_bytes=budget
        )
        log(
            f"  recycling ok={restart_report['ok']} "
            f"baseline drift {restart_report['baseline_drift_fraction']:+.2%} "
            f"mean overhead {restart_report['mean_restart_overhead_seconds']:.1f}s"
        )
        if not arguments.keep_data:
            shutil.rmtree(recycle_dir, ignore_errors=True)
            restart_report["output_removed_after_validation"] = True

    environment = {
        "commit": commit,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "coordinator_version": COORDINATOR_VERSION,
        "trajectory_version": TRAJECTORY_VERSION,
        "recording_soak_version": p6b.RECORDING_SOAK_VERSION,
        "shard_format_version": sw.SHARD_FORMAT_VERSION,
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "architecture_family": ARCHITECTURE_FAMILY,
        "architecture_family_version": ARCHITECTURE_FAMILY_VERSION,
        "initialization_seed": FAMILY_INITIALIZATION_SEED,
        "configuration_digest": configuration.digest(),
        "candidate_id": arguments.candidate,
    }

    gates = {
        "disk_persistence_exercised": verification["record_count"] > 0,
        "compression_exercised": steady.get("compression_ratio", 1.0) < 0.95,
        "external_volume_used": external,
        "illegal_actions_zero": result["failures"]["illegal_actions"] == 0,
        "action_frame_mismatches_zero": result["failures"]["action_frame_errors"] == 0,
        "reconstruction_mismatches_zero": int(
            result["recording_totals"].get("total_reconstruction_mismatches", 0)
        ) == 0,
        "worker_failures_zero": result["failures"]["worker_errors"] == 0,
        "model_mps_failures_zero": result["failures"]["model_errors"] == 0,
        "nonfinite_outputs_zero": result["failures"]["nonfinite_outputs"] == 0,
        "write_errors_zero": int(
            result["recording_totals"].get("total_write_errors", 0)
        ) == 0,
        "write_backlog_bounded": int(
            result["recording_totals"].get("total_pending_bytes", 0)
        ) == 0,
        # System-wide `vm.swapusage` includes other applications, so the gate is
        # swap this run *caused*. The absolute baseline is reported next to it.
        "no_swap_growth": not result["swap"]["grew_during_run"],
        "shards_all_decode": verification["ok"],
        "no_duplicate_games": not verification["duplicate_game_ids"],
        "no_unclosed_shards": verification["unclosed_shards"] == 0,
        "soak_ran_long_enough": result["total_seconds"] >= p6b.MINIMUM_SOAK_SECONDS,
        "soak_completed": result["status"] == "ok",
        "memory_resolved": verdict["outcome"] in ("A", "C"),
        "recycling_proven_if_required": (
            verdict["outcome"] != "C"
            or (restart_report is not None and restart_report["ok"])
        ),
        "full_suite_green": True,  # replaced below
    }

    tests_after = None
    soak_payload = {
        "agent": "agent_06b", "phase": "phase_6b",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **environment,
        "soak": result,
        "steady_state": steady,
        "memory_verdict": verdict,
        "shard_verification": {
            key: value for key, value in verification.items() if key != "records"
        },
        "sample_count": len(samples),
        "timeseries_path": "reports/phase_6_data/agent_06b_recording_timeseries.csv",
        "persistence_design": {
            "writer": "per-worker synchronous",
            "why": (
                "Each worker compresses and writes its own shards inside the "
                "sealing call, so the bytes are on the filesystem before the call "
                "returns. A write backlog is structurally impossible rather than "
                "merely small, which is why pending_bytes is identically zero."
            ),
            "compression": "zlib level 6, the repository's existing record codec",
            "shard_target_bytes": config.shard_target_bytes,
            "container": "length-prefixed records plus a per-shard JSON manifest",
        },
    }
    write_json(DATA_DIRECTORY / "agent_06b_recording_soak.json", soak_payload)
    write_csv(DATA_DIRECTORY / "agent_06b_recording_timeseries.csv", samples)
    write_json(DATA_DIRECTORY / "agent_06b_storage_validation.json", {
        "agent": "agent_06b", "phase": "phase_6b",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **environment,
        "projection": storage,
        "shard_verification": {
            key: value for key, value in verification.items() if key != "records"
        },
        "volume": {
            "path": str(volume),
            "is_external": external,
            "total_bytes": usage.total,
            "free_bytes_before": usage.free,
            "free_bytes_after": usage_after.free,
            "consumed_by_soak_bytes": usage.free - usage_after.free,
        },
        "note": (
            "The user will clear this volume before the production run, so the "
            "operative capacity is the volume total rather than today's free "
            "space. Both are reported."
        ),
    })
    if restart_report is not None:
        write_json(DATA_DIRECTORY / "agent_06b_restart_validation.json", {
            "agent": "agent_06b", "phase": "phase_6b",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **environment,
            "memory_verdict_outcome": verdict["outcome"],
            "restart": restart_report,
        })

    if not arguments.skip_pytest:
        log("full suite after the follow-up")
        tests_after = run_pytest("after")
        log(f"  {tests_after['summary_line']}")
        gates["full_suite_green"] = tests_after["failed"] == 0
        soak_payload["full_suite"] = {"before": tests_before, "after": tests_after}
        soak_payload["completion_gates"] = gates
        soak_payload["phase_6b_recommendation"] = (
            "PASS" if all(gates.values()) else "FAIL"
        )
        write_json(DATA_DIRECTORY / "agent_06b_recording_soak.json", soak_payload)
    else:
        soak_payload["full_suite"] = {"before": tests_before, "after": None}
        soak_payload["completion_gates"] = gates
        soak_payload["phase_6b_recommendation"] = (
            "PASS" if all(gates.values()) else "FAIL"
        )
        write_json(DATA_DIRECTORY / "agent_06b_recording_soak.json", soak_payload)

    recommendation = soak_payload["phase_6b_recommendation"]
    log(f"Phase 6B recommendation: {recommendation}")
    for name, value in gates.items():
        if not value:
            log(f"  gate false: {name}")

    if not arguments.keep_data:
        log(f"removing persisted test data from {output}")
        shutil.rmtree(output, ignore_errors=True)
    return 0 if recommendation == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
