#!/usr/bin/env python3
"""Phase 3 Agent 5 acceptance harness: end-to-end pipeline, soak, and decision.

Integrates Agents 1-4 into the first bulk-synchronous self-play pipeline and
produces the evidence for the Phase 3 backend decision. Writes
`reports/phase_3_data/agent_05_end_to_end.json` plus
`reports/phase_3_data/agent_05_end_to_end_raw.csv` and
`reports/phase_3_data/agent_05_soak_timeseries.csv`:

- the integrated differential gate, in which an independent set of engine games
  is advanced in lockstep with the live pipeline using the actions the *model*
  sampled, covering observation, legality, action legality, resulting state,
  terminal result, generation/reset and trajectory identity;
- the reconstruction gate, in which stored decisions are round-tripped through
  the trajectory codec and rebuilt through Agent 3's path;
- a screened sweep over the required worker, environment and inference-batch
  dimensions, then sustained runs on the finalists;
- the simulation-pipeline rate with the model removed, which is the numerator of
  the decision ratio `R`;
- a continuous soak at the chosen configuration, sampling memory, swap,
  throughput, terminal reasons and worker liveness.

Metal is required. If it is unavailable the run stops and records `BLOCKED`
rather than substituting central-processing-unit numbers.

Trajectory records are built, encoded, counted, sampled for verification and
then **discarded**. A full-rate two-hour collection would be roughly 17 GB; the
storage path is exercised at its real cost without persisting the corpus. Only
a handful of retained records come back for inspection.

Usage:

    python scripts/run_phase3_agent05.py                 # full acceptance run
    python scripts/run_phase3_agent05.py --quick         # fast smoke run
    python scripts/run_phase3_agent05.py --skip-pytest   # measurements only
    python scripts/run_phase3_agent05.py --soak-seconds 14400
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_3_data"
DEFAULT_OUTPUT = DATA_DIRECTORY / "agent_05_end_to_end.json"
DEFAULT_RAW_OUTPUT = DATA_DIRECTORY / "agent_05_end_to_end_raw.csv"
DEFAULT_SOAK_OUTPUT = DATA_DIRECTORY / "agent_05_soak_timeseries.csv"

AGENT_01_DATA = DATA_DIRECTORY / "agent_01_batch_equivalence.json"
AGENT_02_DATA = DATA_DIRECTORY / "agent_02_shared_memory_scaling.json"
AGENT_03_DATA = DATA_DIRECTORY / "agent_03_trajectory_reconstruction.json"
AGENT_04_DATA = DATA_DIRECTORY / "agent_04_mps_inference.json"

AGENT = "agent_05"

TEST_TARGETS = (
    "tests/training/test_coordinator.py",
    "tests/training/test_end_to_end_pipeline.py",
)

#: Acceptance thresholds from `05_AGENT_5_END_TO_END_DECISION.md`.
REQUIRED_INTEGRATED_STEPS = 10_000
REQUIRED_RECONSTRUCTED_DECISIONS = 10_000
REQUIRED_SOAK_SECONDS = 2 * 60 * 60

FILES_CREATED = (
    "stratego/training/coordinator.py",
    "stratego/training/end_to_end_benchmark.py",
    "tests/training/test_coordinator.py",
    "tests/training/test_end_to_end_pipeline.py",
    "scripts/run_phase3_agent05.py",
    "reports/phase_3_data/agent_05_end_to_end.json",
    "reports/phase_3_data/agent_05_end_to_end_raw.csv",
    "reports/phase_3_data/agent_05_soak_timeseries.csv",
)

FILES_MODIFIED = (
    # Backward-compatible additions: the coordinator-written decision fields.
    "stratego/training/shared_buffers.py",
    # Backward-compatible addition: in-worker trajectory recording.
    "stratego/training/worker_pool.py",
    # Correctness fix: `_gumbel_noise` could return `+inf` when `torch.rand`
    # drew exactly 0, which made an illegal entry `NaN` and let `argmax` select
    # it. See the deviations section of the report.
    "stratego/training/representative_model.py",
    # Updated for the widened coordinator-written field set.
    "tests/training/test_shared_buffers.py",
    "reports/phase_3_implementation_report.md",
)


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------


def check_prerequisites() -> dict:
    """Agents 1-4 must all be `PASS`."""
    record: dict = {"satisfied": True, "agents": {}}
    for name, path in (
        ("agent_01", AGENT_01_DATA),
        ("agent_02", AGENT_02_DATA),
        ("agent_03", AGENT_03_DATA),
        ("agent_04", AGENT_04_DATA),
    ):
        if not path.exists():
            record["agents"][name] = {"present": False, "status": None}
            record["satisfied"] = False
            continue
        payload = json.loads(path.read_text())
        record["agents"][name] = {
            "present": True,
            "status": payload.get("status"),
            "implementation_version": payload.get("implementation_version"),
            "observation_version": payload.get("observation_version"),
            "rules_version": payload.get("rules_version"),
        }
        if payload.get("status") != "PASS":
            record["satisfied"] = False
    return record


def agent_04_reference() -> dict:
    """Agent 4's sustainable inference rates: the denominator of `R`."""
    payload = json.loads(AGENT_04_DATA.read_text())
    return {
        "sustainable_positions_per_second": payload.get(
            "sustainable_representative_model_positions_per_second"
        ),
        "sustainable_positions_per_second_float32_dense": payload.get(
            "sustainable_representative_model_positions_per_second_float32_dense"
        ),
        "sustained_throughput_recommended": payload.get(
            "sustained_throughput_recommended"
        ),
        "recommended_precision": payload.get("recommended_precision"),
        "recommended_legality_representation": payload.get(
            "recommended_legality_representation"
        ),
    }


def agent_03_reference() -> dict:
    payload = json.loads(AGENT_03_DATA.read_text())
    return {
        "recommended_snapshot_interval": payload.get("recommended_snapshot_interval"),
        "mean_decision_bytes": payload.get("mean_decision_bytes"),
        "reconstruction_positions_per_second": payload.get(
            "reconstruction_positions_per_second"
        ),
    }


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


def run_pytest(targets=TEST_TARGETS) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *targets],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    output = completed.stdout + completed.stderr
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "xfailed": 0}
    for key, pattern in (
        ("passed", r"(\d+) passed"),
        ("failed", r"(\d+) failed"),
        ("errors", r"(\d+) error"),
        ("skipped", r"(\d+) skipped"),
        ("xfailed", r"(\d+) xfailed"),
    ):
        match = re.search(pattern, output)
        if match:
            counts[key] = int(match.group(1))
    failures = [line for line in output.splitlines() if line.startswith("FAILED")]
    return {
        "targets": list(targets),
        "test_exit_code": completed.returncode,
        "test_passed": counts["passed"],
        "test_failed": counts["failed"],
        "test_errors": counts["errors"],
        "test_skipped": counts["skipped"],
        "test_expected_failures": counts["xfailed"],
        "test_total": sum(counts.values()),
        "test_seconds": round(elapsed, 3),
        "test_failure_lines": failures[:20],
    }


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------


RAW_COLUMNS = (
    "phase",
    "group",
    "label",
    "num_workers",
    "num_environments",
    "inference_batch_size",
    "precision",
    "legality",
    "record_trajectories",
    "status",
    "measured_seconds",
    "global_steps",
    "positions",
    "games",
    "chunks_per_step",
    "positions_per_second",
    "transitions_per_second",
    "games_per_second",
    "resets_per_second",
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "max_ms",
    "observation_seconds",
    "legality_seconds",
    "transfer_seconds",
    "inference_seconds",
    "sampling_seconds",
    "writeback_seconds",
    "worker_seconds",
    "barrier_seconds",
    "straggler_seconds",
    "coordinator_active_fraction",
    "coordinator_wait_fraction",
    "mps_active_fraction",
    "worker_active_fraction",
    "worker_barrier_wait_fraction",
    "process_memory_bytes",
    "shared_memory_bytes",
    "worker_max_rss_bytes",
    "error",
)


def write_raw_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        for row in rows:
            configuration = row.get("configuration", {})
            record = {
                "phase": row.get("phase", ""),
                "group": row.get("group", ""),
                "label": configuration.get("label", ""),
                "num_workers": configuration.get("num_workers", ""),
                "num_environments": configuration.get("num_environments", ""),
                "inference_batch_size": configuration.get("inference_batch_size", ""),
                "precision": configuration.get("precision", ""),
                "legality": configuration.get("legality", ""),
                "record_trajectories": configuration.get("record_trajectories", ""),
            }
            for column in RAW_COLUMNS:
                if column in record:
                    continue
                value = row.get(column, "")
                record[column] = round(value, 6) if isinstance(value, float) else value
            writer.writerow(record)


SOAK_COLUMNS = (
    "elapsed_seconds",
    "global_steps",
    "positions",
    "games",
    "resets",
    "window_seconds",
    "window_positions_per_second",
    "window_games_per_second",
    "coordinator_rss_bytes",
    "shared_memory_bytes",
    "workers_alive",
    "workers_expected",
    "mps_current_allocated_bytes",
    "mps_driver_allocated_bytes",
    "swap_total_bytes",
    "swap_used_bytes",
    "swap_free_bytes",
)


def write_soak_csv(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOAK_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for sample in samples:
            row = {
                key: (round(value, 6) if isinstance(value, float) else value)
                for key, value in sample.items()
                if key in SOAK_COLUMNS
            }
            writer.writerow(row)


def blocked_report(reason: str, prerequisites: dict, extra: dict | None = None) -> dict:
    report = {
        "agent": AGENT,
        "status": "BLOCKED",
        "blocked_reason": reason,
        "prerequisites": prerequisites,
        "files_created": list(FILES_CREATED),
        "files_modified": list(FILES_MODIFIED),
    }
    if extra:
        report.update(extra)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--soak-output", type=Path, default=DEFAULT_SOAK_OUTPUT)
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--screen-seconds", type=float, default=12.0)
    parser.add_argument("--finalist-seconds", type=float, default=60.0)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--simulation-seconds", type=float, default=60.0)
    parser.add_argument("--soak-seconds", type=float, default=REQUIRED_SOAK_SECONDS)
    parser.add_argument("--soak-sample-seconds", type=float, default=60.0)
    parser.add_argument("--gate-environments", type=int, default=64)
    parser.add_argument("--gate-workers", type=int, default=4)
    parser.add_argument("--integrated-steps", type=int, default=REQUIRED_INTEGRATED_STEPS)
    parser.add_argument(
        "--reconstructed-decisions", type=int, default=REQUIRED_RECONSTRUCTED_DECISIONS
    )
    parser.add_argument("--seed", type=int, default=50_005)
    options = parser.parse_args()

    started_wall = time.time()
    started = time.perf_counter()

    def progress(message: str) -> None:
        stamp = time.perf_counter() - started
        print(f"[{stamp / 60:6.1f} min] {message}", flush=True)

    prerequisites = check_prerequisites()
    if not prerequisites["satisfied"]:
        report = blocked_report(
            "Agents 1-4 must all be PASS before Agent 5 may run.", prerequisites
        )
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("BLOCKED: prerequisite agent is not PASS", file=sys.stderr)
        return 1

    try:
        import torch
    except ImportError as error:
        report = blocked_report(f"PyTorch is not importable: {error}", prerequisites)
        options.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("BLOCKED: PyTorch is unavailable", file=sys.stderr)
        return 1

    if not torch.backends.mps.is_available():
        report = blocked_report(
            "Metal is not available; the backend decision must not be taken on "
            "central-processing-unit inference numbers.",
            prerequisites,
        )
        options.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("BLOCKED: Metal is unavailable", file=sys.stderr)
        return 1

    from stratego.training.coordinator import (
        COORDINATOR_VERSION,
        CoordinatorConfig,
        mps_memory_bytes,
    )
    from stratego.training.end_to_end_benchmark import (
        BENCHMARK_VERSION,
        build_screening_plan,
        compute_ratio,
        measure_configuration,
        measure_simulation_pipeline,
        platform_report,
        run_integrated_gate,
        run_reconstruction_gate,
        run_soak,
        swap_bytes,
    )
    from stratego.training.mps_benchmark import detect_device_report, peak_memory_bytes
    from stratego.training.shared_buffers import POLICY_CAPACITY, buffer_nbytes
    from stratego.training.trajectory import TRAJECTORY_VERSION
    from stratego.training.worker_pool import DEFAULT_COLLECTION_POLICY_VERSION

    agent_04 = agent_04_reference()
    agent_03 = agent_03_reference()
    snapshot_interval = int(agent_03["recommended_snapshot_interval"] or 32)
    precision = agent_04["recommended_precision"] or "float16"
    legality = agent_04["recommended_legality_representation"] or "dense"

    if options.quick:
        options.screen_seconds = 4.0
        options.finalist_seconds = 8.0
        options.simulation_seconds = 6.0
        options.soak_seconds = min(options.soak_seconds, 90.0)
        options.soak_sample_seconds = 15.0
        options.integrated_steps = min(options.integrated_steps, 800)
        options.reconstructed_decisions = min(options.reconstructed_decisions, 600)
        options.finalists = 2

    swap_start = swap_bytes()
    report: dict = {
        "agent": AGENT,
        "benchmark_version": BENCHMARK_VERSION,
        "coordinator_version": COORDINATOR_VERSION,
        "trajectory_version": TRAJECTORY_VERSION,
        "quick_mode": bool(options.quick),
        "prerequisites": prerequisites,
        "agent_04_reference": agent_04,
        "agent_03_reference": agent_03,
        "device": detect_device_report(),
        "policy_capacity": POLICY_CAPACITY,
        "collection_policy_version": DEFAULT_COLLECTION_POLICY_VERSION,
        "started_epoch_seconds": started_wall,
        **platform_report(),
    }

    # -- tests --------------------------------------------------------------
    if options.skip_pytest:
        report["tests"] = {"skipped": True}
    else:
        progress("running the Agent 5 test suite")
        tests = run_pytest()
        report["tests"] = tests
        report.update({key: value for key, value in tests.items() if key.startswith("test_")})
        if tests["test_exit_code"] != 0:
            report.update(
                {
                    "status": "FAIL",
                    "failure_reason": "the Agent 5 test suite did not pass",
                    "files_created": list(FILES_CREATED),
                    "files_modified": list(FILES_MODIFIED),
                }
            )
            options.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print("FAIL: tests did not pass", file=sys.stderr)
            return 1
        progress(f"tests passed: {tests['test_passed']}")

    # -- integrated correctness gate ----------------------------------------
    progress(f"integrated differential gate: {options.integrated_steps:,} environment steps")
    gate = run_integrated_gate(
        num_environments=options.gate_environments,
        num_workers=options.gate_workers,
        inference_batch_size=min(256, options.gate_environments),
        target_environment_steps=options.integrated_steps,
        precision=precision,
        legality=legality,
        root_seed=options.seed,
    )
    report["integrated_gate"] = gate.as_dict()
    report["integrated_steps_checked"] = gate.environment_steps
    report["integrated_mismatches"] = gate.mismatches
    progress(
        f"integrated gate: {gate.environment_steps:,} steps, "
        f"{gate.mismatches} mismatches, {gate.games_completed} games"
    )

    progress(f"reconstruction gate: {options.reconstructed_decisions:,} stored decisions")
    reconstruction = run_reconstruction_gate(
        num_environments=256 if not options.quick else 64,
        num_workers=6 if not options.quick else 4,
        inference_batch_size=256 if not options.quick else 64,
        target_decisions=options.reconstructed_decisions,
        precision=precision,
        legality=legality,
        snapshot_interval=snapshot_interval,
        root_seed=options.seed + 2,
    )
    report["reconstruction_gate"] = reconstruction
    report["stored_decisions_reconstructed"] = reconstruction["decisions_reconstructed"]
    report["reconstruction_mismatches"] = reconstruction["reconstruction_mismatches"]
    progress(
        f"reconstruction gate: {reconstruction['decisions_reconstructed']:,} decisions, "
        f"{reconstruction['reconstruction_mismatches']} mismatches"
    )

    correctness_ok = (
        gate.mismatches == 0
        and reconstruction["reconstruction_mismatches"] == 0
        and gate.environment_steps >= options.integrated_steps
        and reconstruction["decisions_reconstructed"] >= options.reconstructed_decisions
    )
    report["correctness_gate_passed"] = correctness_ok
    if not correctness_ok:
        report.update(
            {
                "status": "FAIL",
                "failure_reason": (
                    "integrated correctness did not hold; performance conclusions "
                    "are blocked"
                ),
                "files_created": list(FILES_CREATED),
                "files_modified": list(FILES_MODIFIED),
            }
        )
        options.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("FAIL: integrated correctness gate did not pass", file=sys.stderr)
        return 1

    # -- screening ----------------------------------------------------------
    plan = build_screening_plan(precision=precision, legality=legality)
    progress(f"screening {len(plan)} configurations at {options.screen_seconds:.0f}s each")
    raw_rows: list[dict] = []
    screened: list[dict] = []
    for index, point in enumerate(plan, start=1):
        group = point.pop("group")
        config = CoordinatorConfig(root_seed=options.seed + 10, **point)
        result = measure_configuration(config, seconds=options.screen_seconds)
        result["phase"] = "screen"
        result["group"] = group
        screened.append(result)
        raw_rows.append(result)
        if result["status"] == "ok":
            progress(
                f"  [{index:2d}/{len(plan)}] {config.label:<42} "
                f"{result['positions_per_second']:9,.0f} pos/s"
            )
        else:
            progress(f"  [{index:2d}/{len(plan)}] {config.label:<42} {result['error']}")

    successful = [row for row in screened if row["status"] == "ok"]
    if not successful:
        report.update(
            {
                "status": "FAIL",
                "failure_reason": "every screened configuration failed",
                "screened_configurations": screened,
                "files_created": list(FILES_CREATED),
                "files_modified": list(FILES_MODIFIED),
            }
        )
        options.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 1

    # Finalists are chosen among the configurations that use the recommended
    # precision and legality; the baselines are measured for comparison, not as
    # candidates for the production configuration.
    candidates = [
        row
        for row in successful
        if row["configuration"]["precision"] == precision
        and row["configuration"]["legality"] == legality
    ]
    candidates.sort(key=lambda row: row["positions_per_second"], reverse=True)
    finalists_plan = candidates[: options.finalists]

    progress(
        f"sustained runs on {len(finalists_plan)} finalists "
        f"at {options.finalist_seconds:.0f}s each"
    )
    finalists: list[dict] = []
    for index, candidate in enumerate(finalists_plan, start=1):
        configuration = candidate["configuration"]
        config = CoordinatorConfig(
            num_environments=configuration["num_environments"],
            num_workers=configuration["num_workers"],
            inference_batch_size=configuration["inference_batch_size"],
            precision=configuration["precision"],
            legality=configuration["legality"],
            root_seed=options.seed + 20,
        )
        result = measure_configuration(config, seconds=options.finalist_seconds)
        result["phase"] = "finalist"
        result["group"] = "finalist"
        finalists.append(result)
        raw_rows.append(result)
        if result["status"] == "ok":
            progress(
                f"  [{index}/{len(finalists_plan)}] {config.label:<42} "
                f"{result['positions_per_second']:9,.0f} pos/s"
            )
        else:
            progress(
                f"  [{index}/{len(finalists_plan)}] {config.label:<42} "
                f"{result['error']}"
            )

    successful_finalists = [row for row in finalists if row["status"] == "ok"]
    if not successful_finalists:
        report.update(
            {
                "status": "FAIL",
                "failure_reason": "every finalist configuration failed",
                "screened_configurations": screened,
                "finalist_configurations": finalists,
                "files_created": list(FILES_CREATED),
                "files_modified": list(FILES_MODIFIED),
            }
        )
        write_raw_csv(options.raw_output, raw_rows)
        options.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("FAIL: no finalist configuration completed", file=sys.stderr)
        return 1
    best = max(successful_finalists, key=lambda row: row["positions_per_second"])
    best_configuration = best["configuration"]
    progress(
        f"best configuration: {best_configuration['label']} "
        f"at {best['positions_per_second']:,.0f} pos/s"
    )

    # -- the decision ratio --------------------------------------------------
    progress("measuring the simulation pipeline with the model removed")
    simulation = measure_simulation_pipeline(
        num_environments=best_configuration["num_environments"],
        num_workers=best_configuration["num_workers"],
        seconds=options.simulation_seconds,
        root_seed=options.seed + 30,
    )
    simulation_recording = measure_simulation_pipeline(
        num_environments=best_configuration["num_environments"],
        num_workers=best_configuration["num_workers"],
        seconds=options.simulation_seconds,
        root_seed=options.seed + 31,
        record_trajectories=True,
        snapshot_interval=snapshot_interval,
    )
    report["simulation_pipeline"] = simulation
    report["simulation_pipeline_with_recording"] = simulation_recording
    progress(
        f"simulation pipeline: {simulation['positions_per_second']:,.0f} pos/s "
        f"({simulation_recording['positions_per_second']:,.0f} with recording)"
    )

    denominator = float(agent_04["sustainable_positions_per_second"])
    ratio = compute_ratio(simulation["positions_per_second"], denominator)
    report.update(ratio)
    # Reported alongside, not as the decision: the same ratio taken against the
    # conservative float32 dense baseline and against the contended in-pipeline
    # inference rate the coordinator actually achieved.
    report["R_against_float32_dense_baseline"] = (
        simulation["positions_per_second"]
        / float(agent_04["sustainable_positions_per_second_float32_dense"])
    )
    report["R_against_measured_end_to_end_rate"] = (
        simulation["positions_per_second"] / best["positions_per_second"]
    )
    report["R_with_recording_numerator"] = (
        simulation_recording["positions_per_second"] / denominator
    )
    progress(
        f"R = {ratio['R']:.2f} -> {ratio['backend_decision']} "
        f"(Agent 6 required: {'yes' if ratio['optimized_backend_required'] else 'no'})"
    )

    # -- soak ----------------------------------------------------------------
    soak_config = CoordinatorConfig(
        num_environments=best_configuration["num_environments"],
        num_workers=best_configuration["num_workers"],
        inference_batch_size=best_configuration["inference_batch_size"],
        precision=best_configuration["precision"],
        legality=best_configuration["legality"],
        root_seed=options.seed + 40,
        record_trajectories=True,
        snapshot_interval=snapshot_interval,
        # Continuous verification at one game per worker at a time. The budget
        # is set high enough that it never runs out, so reconstruction is being
        # proved for the whole soak rather than only at the start; the
        # concurrency cap of one is what keeps the cost near 0.3 percent of
        # worker time, because digesting a decision costs roughly 35x recording
        # it.
        verify_target_decisions=1_000_000,
        max_concurrent_verifications=1,
        retain_games=1,
    )
    progress(
        f"soak: {soak_config.label} for {options.soak_seconds / 3600:.2f} h, "
        f"sampling every {options.soak_sample_seconds:.0f}s"
    )

    def on_sample(sample: dict) -> None:
        progress(
            f"  soak {sample['elapsed_seconds'] / 60:6.1f} min: "
            f"{sample['window_positions_per_second']:9,.0f} pos/s  "
            f"games={sample['games']:,}  "
            f"rss={sample['coordinator_rss_bytes'] / 2**20:7,.0f} MB  "
            f"workers={sample['workers_alive']}/{sample['workers_expected']}"
        )

    soak_failed = None
    try:
        soak = run_soak(
            soak_config,
            duration_seconds=options.soak_seconds,
            sample_interval_seconds=options.soak_sample_seconds,
            on_sample=on_sample,
        )
    except Exception as error:  # noqa: BLE001 - a soak failure is a result
        soak_failed = f"{type(error).__name__}: {error}"
        soak = {
            "configuration": soak_config.as_dict(),
            "duration_seconds": 0.0,
            "positions": 0,
            "games": 0,
            "samples": [],
            "errors": [{"error": soak_failed}],
            "memory_growth_bytes": 0,
            "throughput_change_fraction": 0.0,
            "terminal_reason_counts": {},
            "recording": {},
        }
    report["soak"] = {key: value for key, value in soak.items() if key != "samples"}
    report["soak_sample_count"] = len(soak.get("samples", []))
    write_soak_csv(options.soak_output, soak.get("samples", []))
    write_raw_csv(options.raw_output, raw_rows)

    soak_recording = soak.get("recording", {})
    soak_reconstruction_mismatches = int(
        soak_recording.get("total_reconstruction_mismatches", 0)
    )
    soak_ok = (
        soak_failed is None
        and not soak.get("errors")
        and soak_reconstruction_mismatches == 0
        and soak["duration_seconds"] >= min(options.soak_seconds, REQUIRED_SOAK_SECONDS)
        - 5.0
    )

    # -- assemble the report -------------------------------------------------
    swap_end = swap_bytes()
    mps_memory = mps_memory_bytes()
    report.update(
        {
            "screened_configurations": screened,
            "screened_configuration_count": len(screened),
            "finalist_configurations": finalists,
            "best_configuration": best_configuration,
            "best_end_to_end_positions_per_second": best["positions_per_second"],
            "best_games_per_second": best["games_per_second"],
            "best_transitions_per_second": best["transitions_per_second"],
            "worker_wait_fraction": best["worker_barrier_wait_fraction"],
            "worker_active_fraction": best["worker_active_fraction"],
            "coordinator_wait_fraction": best["coordinator_wait_fraction"],
            "coordinator_active_fraction": best["coordinator_active_fraction"],
            "mps_active_fraction_if_measurable": best["mps_active_fraction"],
            "mean_step_latency_ms": best["mean_ms"],
            "p50_step_latency_ms": best["p50_ms"],
            "p95_step_latency_ms": best["p95_ms"],
            "observation_build_seconds": best["observation_seconds"],
            "shared_memory_barrier_seconds": best["barrier_seconds"],
            "mps_inference_seconds": best["inference_seconds"],
            "legality_sampling_seconds": best["legality_seconds"] + best["sampling_seconds"],
            "trajectory_recording_seconds": soak_recording.get("recording_seconds", 0.0),
            "independent_resets_per_second": best["resets_per_second"],
            "memory_peak_bytes": max(
                peak_memory_bytes(),
                int(soak.get("peak_process_memory_bytes", 0)),
            ),
            "worker_max_rss_bytes": int(soak.get("worker_max_rss_bytes", 0)),
            "shared_memory_bytes": buffer_nbytes(
                best_configuration["num_environments"]
            ),
            "mps_peak_memory_bytes_if_available": mps_memory.get(
                "driver_allocated_bytes"
            ),
            "mps_memory": mps_memory,
            "swap_start_bytes": swap_start.get("swap_used_bytes"),
            "swap_end_bytes": swap_end.get("swap_used_bytes"),
            "swap_start": swap_start,
            "swap_end": swap_end,
            "soak_duration_seconds": soak["duration_seconds"],
            "soak_positions": soak["positions"],
            "soak_games": soak["games"],
            "soak_memory_growth_bytes": soak["memory_growth_bytes"],
            "soak_throughput_change_fraction": soak["throughput_change_fraction"],
            "soak_errors": len(soak.get("errors", [])),
            "soak_error_details": soak.get("errors", []),
            "soak_reconstruction_mismatches": soak_reconstruction_mismatches,
            "soak_passed": soak_ok,
            "terminal_reason_counts": soak.get("terminal_reason_counts")
            or best.get("terminal_reason_counts", {}),
            "recommended_worker_count": best_configuration["num_workers"],
            "recommended_environment_count": best_configuration["num_environments"],
            "recommended_inference_batch_size": best_configuration[
                "inference_batch_size"
            ],
            "recommended_precision": best_configuration["precision"],
            "recommended_legality_representation": best_configuration["legality"],
            "recommended_snapshot_interval": snapshot_interval,
            "elapsed_seconds": time.perf_counter() - started,
            "files_created": list(FILES_CREATED),
            "files_modified": list(FILES_MODIFIED),
        }
    )

    soak_long_enough = soak["duration_seconds"] >= REQUIRED_SOAK_SECONDS - 5.0
    passed = (
        correctness_ok
        and soak_ok
        and bool(successful_finalists)
        and (soak_long_enough or options.quick)
    )
    report["status"] = "PASS" if passed else "FAIL"
    if not passed:
        reasons = []
        if not soak_ok:
            reasons.append("the soak did not complete cleanly")
        if not soak_long_enough and not options.quick:
            reasons.append(
                f"the soak ran {soak['duration_seconds'] / 3600:.2f} h, below the "
                f"{REQUIRED_SOAK_SECONDS / 3600:.0f} h minimum"
            )
        report["failure_reason"] = "; ".join(reasons) or "acceptance gate not met"
    if options.quick:
        # A quick run exercises every path at reduced scale and deliberately
        # does not meet the acceptance thresholds.
        report["status"] = "QUICK"

    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    progress(f"status: {report['status']}")
    progress(f"backend decision: {report['backend_decision']} (R = {report['R']:.2f})")
    progress(f"wrote {options.output}")
    progress(f"wrote {options.raw_output}")
    progress(f"wrote {options.soak_output}")
    return 0 if report["status"] in ("PASS", "QUICK") else 1


if __name__ == "__main__":
    raise SystemExit(main())
