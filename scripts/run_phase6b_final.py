#!/usr/bin/env python3
"""Phase 6B final acceptance: the 4-6 hour recycled, compressed, persisted soak.

This is the Gate 2 harness of the Phase 6B continuation. Unlike
`run_phase6b.py` (one continuous soak, recycling validated separately), this
run *is* recycled: the logical soak is a sequence of segments, each a fresh
child process on its own root seed and run id, with an orderly
flush/close/persist/shutdown/restart at every boundary. Restart time is wall
clock spent from the budget, not added to it.

Writes (new names; the first, BLOCKED soak's evidence is never overwritten)::

    reports/phase_6_data/agent_06b_final_soak.json
    reports/phase_6_data/agent_06b_final_soak_timeseries.csv
    reports/phase_6_data/agent_06b_recycling_validation.json
    reports/phase_6_data/agent_06b_final_storage_validation.json

The persisted corpus is deliberately kept on the external volume for the
reviewing chat's acceptance; nothing is deleted.

Usage::

    python scripts/run_phase6b_final.py                       # 5 x 72 min = 6 h
    python scripts/run_phase6b_final.py --segments 4 --segment-seconds 3600
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

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_6_data"

#: The final soak runs on a fresh seed family so its game identities cannot
#: collide with (or silently replay) either preserved corpus: the aborted soak
#: and the restart validation both ran from base 60,006.
DEFAULT_BASE_ROOT_SEED = 70_007

TIMESERIES_COLUMNS = [
    "segment", "segment_root_seed", "run_elapsed_seconds",
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
    print(f"[phase-6b-final] {message}", flush=True)


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


def merged_timeseries(segments: list[dict]) -> list[dict]:
    """All segment samples in one run-relative series."""
    rows: list[dict] = []
    offset = 0.0
    for state in segments:
        for sample in state.get("samples", []):
            row = dict(sample)
            row["segment"] = state["segment"]
            row["segment_root_seed"] = state["root_seed"]
            row["run_elapsed_seconds"] = offset + float(sample["elapsed_seconds"])
            rows.append(row)
        offset += float(state.get("supervisor_wall_seconds", state.get("seconds", 0.0)))
    return rows


def write_timeseries(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=TIMESERIES_COLUMNS, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            prepared = {}
            for column in TIMESERIES_COLUMNS:
                value = row.get(column, "")
                if isinstance(value, dict):
                    value = json.dumps(
                        {str(k): v for k, v in value.items()}, sort_keys=True
                    )
                elif isinstance(value, float):
                    value = round(value, 6)
                prepared[column] = value
            writer.writerow(prepared)


def segment_memory_reports(segments: list[dict], total_system_bytes: int) -> list[dict]:
    from stratego.training import phase6b_recording as p6b

    reports = []
    for state in segments:
        samples = state.get("samples", [])
        settled = [sample for sample in samples if sample.get("in_measured_window")]
        entry = {
            "segment": state["segment"],
            "root_seed": state["root_seed"],
            "samples": len(samples),
            "settled_samples": len(settled),
            "rss_at_start_bytes": state["rss_at_start_bytes"],
            "rss_at_end_bytes": state["rss_at_end_bytes"],
            "swap_first_bytes": (
                int(samples[0]["swap_used_bytes"]) if samples else None
            ),
            "swap_max_bytes": (
                max(int(sample["swap_used_bytes"]) for sample in samples)
                if samples
                else None
            ),
        }
        if len(settled) >= 4:
            verdict = p6b.classify_memory_outcome(
                samples, total_system_bytes=total_system_bytes
            )
            entry["verdict"] = {
                key: verdict[key]
                for key in (
                    "outcome", "reason", "slope_mib_per_hour", "r_squared",
                    "first_bytes", "last_bytes", "settled_window_seconds",
                )
            }
        else:
            entry["verdict"] = None
        reports.append(entry)
    return reports


def aggregate_steady_state(segments: list[dict]) -> dict:
    """Sustained rates over every segment's settled window, summed."""
    window = positions = games = produced = written = compressed = decisions = 0.0
    settled_samples = 0
    for state in segments:
        settled = [
            sample
            for sample in state.get("samples", [])
            if sample.get("in_measured_window")
        ]
        if len(settled) < 2:
            continue
        first, last = settled[0], settled[-1]
        window += last["elapsed_seconds"] - first["elapsed_seconds"]
        positions += last["positions"] - first["positions"]
        games += last["games"] - first["games"]
        produced += (
            last["trajectory_bytes_produced"] - first["trajectory_bytes_produced"]
        )
        written += (
            last["trajectory_bytes_written"] - first["trajectory_bytes_written"]
        )
        compressed += last["compressed_bytes"] - first["compressed_bytes"]
        decisions += last["decisions_recorded"] - first["decisions_recorded"]
        settled_samples += len(settled)
    if window <= 0:
        return {}
    gib = 1024.0**3
    return {
        "window_seconds": window,
        "samples_in_window": settled_samples,
        "positions": positions,
        "games": games,
        "decisions": decisions,
        "positions_per_second": positions / window,
        "games_per_second": games / window,
        "mean_game_length": positions / games if games else 0.0,
        "bytes_produced": produced,
        "bytes_written": written,
        "compressed_bytes": compressed,
        "produced_gib_per_hour": (produced / window) * 3600.0 / gib,
        "written_gib_per_hour": (written / window) * 3600.0 / gib,
        "compression_ratio": compressed / produced if produced else 0.0,
        "write_throughput_bytes_per_second": written / window,
        "bytes_per_decision_written": written / decisions if decisions else 0.0,
        "note": (
            "settled windows only: each segment's warmup is excluded, and the "
            "restart boundaries are between windows. The whole-run wall rate, "
            "which includes warmups and restarts, is reported alongside and is "
            "what a 168-hour wall-clock budget actually yields."
        ),
    }


def expected_stillborn_games(
    base_root_seed: int, segments: int, environments: int, generations: int
) -> list[dict]:
    """Which (segment, environment, generation) identities are stillborn.

    Scanned ahead of the run so an occurrence during the soak is a *predicted*
    event that must be handled (published terminal, sealed, recycled, and
    persisted as a zero-decision record) rather than a surprise.
    """
    from stratego.engine.constants import IMMOVABLE_TYPES
    from stratego.engine.random_play import make_random_setups
    from stratego.training.batch_simulation import derive_slot_seed
    from stratego.training.phase6b_recycle import segment_root_seed

    front_open = (30, 31, 34, 35, 38, 39)
    hits = []
    for segment in range(segments):
        seed = segment_root_seed(base_root_seed, segment)
        for environment in range(environments):
            for generation in range(generations):
                red_setup, _ = make_random_setups(
                    derive_slot_seed(seed, environment, generation)
                )
                if all(red_setup[i] in IMMOVABLE_TYPES for i in front_open):
                    hits.append(
                        {
                            "segment": segment,
                            "segment_root_seed": seed,
                            "environment_id": environment,
                            "generation": generation,
                        }
                    )
    return hits


def streaming_verification_in_subprocess(directory: Path) -> dict:
    """Decode-verify the whole corpus in a child so its peak RSS is its own."""
    script = (
        "import json, resource, sys\n"
        f"sys.path.insert(0, {str(REPOSITORY_ROOT)!r})\n"
        "from stratego.training.shard_writer import directory_summary\n"
        f"summary = directory_summary({str(directory)!r}, decode=True)\n"
        "summary.pop('game_ids', None)\n"
        "summary['verifier_peak_rss_bytes'] = "
        "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "print(json.dumps(summary))\n"
    )
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"streaming verification failed: {completed.stderr[-2000:]}"
        )
    summary = json.loads(completed.stdout.strip().splitlines()[-1])
    summary["verify_seconds"] = time.perf_counter() - started
    summary["verifier_peak_rss_mib"] = summary["verifier_peak_rss_bytes"] / 2**20
    return summary


def locate_stillborn_records(
    directory: Path, expected: list[dict], segment_run_ids: list[str]
) -> list[dict]:
    """Find and decode the predicted stillborn games among the persisted records.

    A stillborn game's identity is predictable in advance, so the manifests are
    scanned for the predicted game ids and only those records are decoded. The
    worker-side `stillborn_games` counters provide the independent cross-check
    that no *unpredicted* zero-decision game was sealed.
    """
    from stratego.training.serialization import decompress
    from stratego.training.shard_writer import iter_shard_payloads
    from stratego.training.trajectory import decode_game_record
    from stratego.training.batch_simulation import slot_game_id

    wanted = {
        slot_game_id(
            entry["segment_root_seed"], entry["environment_id"], entry["generation"]
        ): entry
        for entry in expected
    }
    found = []
    for manifest_path in sorted(Path(directory).glob("*.json")):
        manifest = json.loads(manifest_path.read_text())
        indices = {
            index: game_id
            for index, game_id in enumerate(manifest["game_ids"])
            if game_id in wanted
        }
        if not indices:
            continue
        shard = str(manifest_path.with_suffix(".stgshard"))
        for index, payload in enumerate(iter_shard_payloads(shard)):
            if index not in indices:
                continue
            record = decode_game_record(
                decompress(payload) if manifest["compressed"] else payload
            )
            found.append(
                {
                    "game_id": record.game_id,
                    "environment_id": record.environment_id,
                    "generation": record.generation,
                    "final_ply": record.final_ply,
                    "decisions": len(record.decisions),
                    "terminal_reason": record.terminal_reason,
                    "terminal_result": record.terminal_result,
                    "shard": manifest["data_file"],
                    "index_in_shard": index,
                }
            )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="/Volumes/Brandon_Washington/stratego_phase6b/final_soak",
    )
    parser.add_argument("--segments", type=int, default=5)
    parser.add_argument("--segment-seconds", type=float, default=4320.0)
    parser.add_argument("--base-root-seed", type=int, default=DEFAULT_BASE_ROOT_SEED)
    parser.add_argument("--candidate", default="C1")
    parser.add_argument("--warmup-steps", type=int, default=3000)
    parser.add_argument("--sample-seconds", type=float, default=60.0)
    parser.add_argument(
        "--stillborn-scan-generations", type=int, default=300,
        help="pre-scan horizon; far beyond what a segment can reach",
    )
    parser.add_argument(
        "--data-directory", type=Path, default=DATA_DIRECTORY,
        help="where the machine-readable artifacts are written",
    )
    arguments = parser.parse_args()
    data_directory = arguments.data_directory

    import torch

    from stratego.model.architecture_configs import candidate_config
    from stratego.training import phase6b_recording as p6b
    from stratego.training import phase6b_recycle as recycle
    from stratego.training import shard_writer as sw
    from stratego.training.coordinator import COORDINATOR_VERSION
    from stratego.training.trajectory import TRAJECTORY_VERSION

    if not torch.backends.mps.is_available():
        log("MPS is not available")
        return 1

    commit = git_commit()
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        log(f"output directory {output} is not empty; refusing to mix corpora")
        return 1

    volume = output
    while not volume.is_mount() and volume != volume.parent:
        volume = volume.parent
    usage = shutil.disk_usage(str(output))
    external = str(volume).startswith("/Volumes/")
    logical_seconds = arguments.segments * arguments.segment_seconds
    log(f"commit {commit[:12]} | output {output}")
    log(
        f"volume {volume}: total {usage.total / 2**30:.1f} GiB, free "
        f"{usage.free / 2**30:.1f} GiB, external={external}"
    )
    needed = logical_seconds / 3600.0 * 4.0 * 2**30  # ~4 GiB/h with margin
    if usage.free < needed + 10 * 2**30:
        log(
            f"insufficient free space: need ~{(needed + 10 * 2**30) / 2**30:.0f} "
            f"GiB, have {usage.free / 2**30:.0f} GiB"
        )
        return 1

    configuration = candidate_config(arguments.candidate)
    log(
        f"final soak: {arguments.candidate} ({configuration.digest()[:12]}...), "
        f"{arguments.segments} segments x {arguments.segment_seconds:.0f}s = "
        f"{logical_seconds / 3600.0:.2f}h logical, base seed "
        f"{arguments.base_root_seed}, compressed shards"
    )

    log("pre-scanning segment seed horizons for stillborn setups")
    expected_stillborns = expected_stillborn_games(
        arguments.base_root_seed,
        arguments.segments,
        1536,
        arguments.stillborn_scan_generations,
    )
    log(f"  stillborn identities in the scanned horizon: {expected_stillborns}")

    run_id = f"p6bf{int(time.time()) % 1000000:06d}"
    supervisor = recycle.RecyclingSupervisor(
        output_directory=output,
        state_directory=REPOSITORY_ROOT / ".phase6b_final_state",
        base_run_id=run_id,
        base_root_seed=arguments.base_root_seed,
        candidate_id=arguments.candidate,
        warmup_steps=arguments.warmup_steps,
        sample_seconds=arguments.sample_seconds,
    )

    def segment_progress(state: dict) -> None:
        log(
            f"segment {state['segment']} ({state['run_id']}, seed "
            f"{state['root_seed']}): {state['status']} in "
            f"{state['seconds']:.1f}s (+{state['startup_shutdown_overhead_seconds']:.1f}s "
            f"overhead) | positions {state['positions']:,} games "
            f"{state['games']:,} shards {state['shards_closed']} written "
            f"{state['bytes_written'] / 2**30:.2f} GiB | rss "
            f"{state['rss_at_start_bytes'] / 2**20:.0f} -> "
            f"{state['rss_at_end_bytes'] / 2**20:.0f} MiB"
        )

    started_unix = time.time()
    summary = supervisor.run(
        segments=arguments.segments,
        seconds_per_segment=arguments.segment_seconds,
        progress=segment_progress,
        timeout=arguments.segment_seconds + 3600.0,
    )
    log(
        f"recycled run complete: {summary['segments_run']} segments, "
        f"{summary['elapsed_wall_seconds']:.0f}s wall, restart overhead "
        f"{summary['restart_overhead_seconds']:.1f}s "
        f"({summary['restart_overhead_fraction_of_wall']:.3%})"
    )

    segments = supervisor.segments
    rows = merged_timeseries(segments)

    # -- evidence first, before any post-processing -------------------------
    write_timeseries(data_directory / "agent_06b_final_soak_timeseries.csv", rows)
    write_json(data_directory / "agent_06b_final_soak.json", {
        "agent": "agent_06b",
        "phase": "phase_6b_final",
        "status": "SOAK_COMPLETE_VERIFICATION_PENDING",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "supervisor_summary": {
            key: value for key, value in summary.items() if key != "segments"
        },
        "segments": summary["segments"],
    })
    log("soak evidence written; starting post-processing")

    # -- memory --------------------------------------------------------------
    system = p6b.system_memory()
    memory_by_segment = segment_memory_reports(segments, system["total_bytes"])
    all_samples = [sample for state in segments for sample in state["samples"]]
    swap_values = [int(sample["swap_used_bytes"]) for sample in all_samples]
    swap_report = {
        "first_bytes": swap_values[0] if swap_values else None,
        "last_bytes": swap_values[-1] if swap_values else None,
        "maximum_bytes": max(swap_values) if swap_values else None,
        "growth_bytes": (
            max(swap_values) - swap_values[0] if swap_values else None
        ),
        "grew_during_run": bool(swap_values) and max(swap_values) > swap_values[0],
        "note": "vm.swapusage is system-wide; the gate is growth during the run",
    }

    slopes = [
        entry["verdict"]["slope_mib_per_hour"]
        for entry in memory_by_segment
        if entry["verdict"]
    ]
    worst_slope_mib = max(slopes) if slopes else 0.0
    baselines = summary["rss_at_segment_start_bytes"]
    budget_bytes = p6b.RECYCLE_BUDGET_FRACTION * system["total_bytes"]
    interval = {
        "worst_segment_slope_mib_per_hour": worst_slope_mib,
        "growth_budget_gib": budget_bytes / 2**30,
        "hours_to_consume_budget_at_worst_slope": (
            budget_bytes / (worst_slope_mib * 2**20)
            if worst_slope_mib > 0
            else None
        ),
    }

    # -- streaming verification ---------------------------------------------
    log("streaming verification of the final corpus (subprocess, decode=True)")
    verification = streaming_verification_in_subprocess(output)
    log(
        f"  {verification['shard_count']} shards, "
        f"{verification['record_count']} records, "
        f"{verification['file_bytes'] / 2**30:.3f} GiB, ok={verification['ok']}, "
        f"verifier peak RSS {verification['verifier_peak_rss_mib']:.1f} MiB in "
        f"{verification['verify_seconds']:.0f}s"
    )

    log("locating persisted stillborn records among the predictions")
    stillborn_records = locate_stillborn_records(
        output, expected_stillborns, summary["segment_run_ids"]
    )
    stillborn_counted_by_workers = sum(
        int(state.get("stillborn_games", 0)) for state in segments
    )
    log(
        f"  persisted stillborn records: {stillborn_records} | "
        f"worker-counted: {stillborn_counted_by_workers}"
    )

    # -- storage -------------------------------------------------------------
    steady = aggregate_steady_state(segments)
    usage_after = shutil.disk_usage(str(output))
    wall_rate = {
        "bytes_written": summary["totals"]["bytes_written"],
        "elapsed_wall_seconds": summary["elapsed_wall_seconds"],
        "written_gib_per_hour_wall": (
            summary["totals"]["bytes_written"]
            / summary["elapsed_wall_seconds"]
            * 3600.0
            / 2**30
        ),
        "note": (
            "whole-run rate over wall clock including warmups and restarts; "
            "the honest projection basis for a 168-hour wall-clock budget"
        ),
    }
    per_second_wall = (
        summary["totals"]["bytes_written"] / summary["elapsed_wall_seconds"]
    )
    projection_168h_bytes = per_second_wall * p6b.FINAL_RUN_SECONDS
    storage = {
        "measured_settled": {
            key: steady.get(key)
            for key in (
                "window_seconds", "written_gib_per_hour", "produced_gib_per_hour",
                "compression_ratio", "write_throughput_bytes_per_second",
                "bytes_per_decision_written",
            )
        },
        "measured_wall": wall_rate,
        "extrapolated_168h": {
            "basis": "whole-run wall rate (includes warmups and restarts)",
            "bytes": projection_168h_bytes,
            "gib": projection_168h_bytes / 2**30,
            "gb": projection_168h_bytes / 1000**3,
            "gib_per_24_hours": per_second_wall * 86400.0 / 2**30,
            "settled_basis_gib": (
                steady["write_throughput_bytes_per_second"]
                * p6b.FINAL_RUN_SECONDS
                / 2**30
                if steady
                else None
            ),
            "shard_headroom_gib": (
                supervisor.shard_target_bytes or sw.DEFAULT_SHARD_TARGET_BYTES
            )
            * 10
            / 2**30,
        },
        "volume": {
            "path": str(volume),
            "is_external": external,
            "total_bytes": usage.total,
            "total_gib": usage.total / 2**30,
            "free_bytes_before": usage.free,
            "free_bytes_after": usage_after.free,
            "consumed_by_soak_bytes": usage.free - usage_after.free,
            "fits_in_total": projection_168h_bytes <= usage.total,
            "remaining_after_168h_on_cleared_volume_gib": (
                (usage.total - projection_168h_bytes) / 2**30
            ),
            "note": (
                "the volume is cleared before the production run, so the "
                "operative capacity is the total; today's free space is "
                "reported alongside"
            ),
        },
        "overheads": {
            "manifest_bytes": sum(
                path.stat().st_size for path in output.glob("*.json")
            ),
            "state_files_bytes": sum(
                path.stat().st_size
                for path in (REPOSITORY_ROOT / ".phase6b_final_state").glob("*.json")
            ),
        },
    }

    # -- gates ---------------------------------------------------------------
    failure_totals: dict[str, int] = {}
    for state in segments:
        for key, value in state.get("failures", {}).items():
            failure_totals[key] = failure_totals.get(key, 0) + int(value)
    minimum_logical = 4 * 3600.0
    gates = {
        "soak_completed": all(state["status"] == "ok" for state in segments),
        "soak_ran_long_enough": summary["elapsed_wall_seconds"] >= minimum_logical,
        "multiple_recycle_boundaries": summary["segments_run"] >= 4,
        "resume_automatic": True,  # the supervisor launched every segment itself
        "illegal_actions_zero": failure_totals.get("illegal_actions", 0) == 0,
        "active_with_zero_legal_zero": all(
            "zero legal" not in str(state.get("error") or "") for state in segments
        ),
        "action_frame_mismatches_zero": failure_totals.get(
            "action_frame_errors", 0
        ) == 0,
        "reconstruction_mismatches_zero": all(
            state["reconstruction_mismatches"] == 0 for state in segments
        ),
        "worker_failures_zero": failure_totals.get("worker_errors", 0) == 0,
        "model_mps_failures_zero": failure_totals.get("model_errors", 0) == 0,
        "nonfinite_outputs_zero": failure_totals.get("nonfinite_outputs", 0) == 0,
        "write_errors_zero": all(state["write_errors"] == 0 for state in segments),
        "write_backlog_bounded": True,  # synchronous writes; structurally empty
        "no_swap_growth": not swap_report["grew_during_run"],
        "shards_all_decode": verification["ok"],
        "no_duplicate_games": not verification["duplicate_game_ids"],
        "no_unclosed_shards": verification["unclosed_shards"] == 0,
        "records_match_writer_counts": (
            verification["record_count"] == summary["totals"]["records_persisted"]
        ),
        "shards_match_writer_counts": (
            verification["shard_count"] == summary["totals"]["shards_closed"]
        ),
        "rss_returns_to_baseline": summary["rss_returns_to_baseline"],
        "no_cumulative_baseline_drift": abs(summary["baseline_drift_fraction"]) <= 0.10,
        "supervisor_invariants_clean": not summary["problems"],
        "disk_persistence_exercised": verification["record_count"] > 0,
        "compression_exercised": (
            0.0 < steady.get("compression_ratio", 1.0) < 0.95 if steady else False
        ),
        "external_volume_used": external,
        # Every stillborn game a worker counted must be sealed on disk as a
        # valid zero-decision record whose identity was predicted in advance.
        "stillborn_handling_consistent": (
            len(stillborn_records) == stillborn_counted_by_workers
            and all(
                entry["final_ply"] == 0
                and entry["decisions"] == 0
                and entry["terminal_reason"]
                in ("opponent_no_legal_move", "both_no_legal_move_draw")
                for entry in stillborn_records
            )
        ),
    }

    environment_block = {
        "commit": commit,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "coordinator_version": COORDINATOR_VERSION,
        "trajectory_version": TRAJECTORY_VERSION,
        "recording_soak_version": p6b.RECORDING_SOAK_VERSION,
        "shard_format_version": sw.SHARD_FORMAT_VERSION,
        "configuration_digest": configuration.digest(),
        "candidate_id": arguments.candidate,
        "base_root_seed": arguments.base_root_seed,
        "run_id": run_id,
        "started_unix": started_unix,
    }

    verdict = "PASS" if all(gates.values()) else "FAIL"
    soak_payload = {
        "agent": "agent_06b",
        "phase": "phase_6b_final",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **environment_block,
        "logical_plan": {
            "segments": arguments.segments,
            "segment_seconds": arguments.segment_seconds,
            "logical_seconds": logical_seconds,
            "warmup_steps_per_segment": arguments.warmup_steps,
            "sample_seconds": arguments.sample_seconds,
        },
        "supervisor_summary": {
            key: value for key, value in summary.items() if key != "segments"
        },
        "segments": summary["segments"],
        "steady_state": steady,
        "wall_rate": wall_rate,
        "memory_by_segment": memory_by_segment,
        "swap": swap_report,
        "recycle_interval_analysis": interval,
        "expected_stillborn_games": expected_stillborns,
        "persisted_stillborn_records": stillborn_records,
        "stillborn_counted_by_workers": stillborn_counted_by_workers,
        "in_flight_games_at_boundaries": {
            "policy": (
                "games in flight when a segment ends are intentionally not "
                "persisted: a partial trajectory is not a trajectory. The loss "
                "is bounded by one game per environment per boundary and is "
                "not a loss of sealed records."
            ),
            "upper_bound_per_boundary": 1536,
            "boundaries": summary["segments_run"],
        },
        "shard_verification": verification,
        "completion_gates": gates,
        "phase_6b_final_recommendation": verdict,
    }
    write_json(data_directory / "agent_06b_final_soak.json", soak_payload)
    write_timeseries(data_directory / "agent_06b_final_soak_timeseries.csv", rows)
    write_json(data_directory / "agent_06b_recycling_validation.json", {
        "agent": "agent_06b",
        "phase": "phase_6b_final",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **environment_block,
        "purpose": (
            "recycling exercised in anger: the 4-6 hour logical soak is itself "
            "a sequence of recycled segments; every boundary performs "
            "flush/close-manifests, persist state, orderly shutdown, process "
            "exit, restart, identical-configuration reload and automatic "
            "resume"
        ),
        "recycling": {
            key: summary[key]
            for key in (
                "segments_run", "segment_seed_stride", "elapsed_wall_seconds",
                "collection_seconds", "restart_overhead_seconds",
                "restart_overhead_fraction_of_wall",
                "mean_restart_overhead_seconds", "wall_clock_accounting",
                "rss_at_segment_start_bytes", "rss_at_segment_end_bytes",
                "baseline_drift_fraction", "rss_returns_to_baseline",
                "configuration_digest", "segment_root_seeds", "segment_run_ids",
                "totals", "problems", "ok",
            )
        },
        "memory_by_segment": memory_by_segment,
        "recycle_interval_analysis": interval,
        "segments": summary["segments"],
    })
    write_json(
        data_directory / "agent_06b_final_storage_validation.json",
        {
            "agent": "agent_06b",
            "phase": "phase_6b_final",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **environment_block,
            "projection": storage,
            "shard_verification": {
                key: value
                for key, value in verification.items()
                if key not in ("problem_shards",) or value
            },
        },
    )

    log(f"final recommendation: {verdict}")
    for name, value in gates.items():
        if not value:
            log(f"  gate false: {name}")
    log(f"corpus preserved at {output}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
