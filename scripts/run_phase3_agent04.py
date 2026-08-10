#!/usr/bin/env python3
"""Phase 3 Agent 4 acceptance harness.

Benchmarks the *representative* compact Transformer probe on the Apple Metal
Performance Shaders backend and writes
`reports/phase_3_data/agent_04_mps_inference.json` plus
`reports/phase_3_data/agent_04_mps_inference_raw.csv`:

- the batch-size sweep over 64, 128, 256, 512, 1,024, 1,536 and 2,048;
- float32 as the baseline precision, with float16 and bfloat16 probed and kept
  only where the whole path is actually supported and stable;
- dense `(B, 10000)` legality against a compact padded legal-identifier path;
- warm-up, synchronised latency, positions/second, model-only positions/second,
  legality+sampling positions/second, peak process memory, Metal allocator
  memory, and any out-of-memory or unsupported-operation failures;
- sustained-throughput runs for the fastest measured configuration, for the
  recommended configuration, and for the conservative float32 + dense baseline.

.. warning::

   The network benchmarked here is a throw-away probe of the *planned shape*.
   It is not the frozen model design and is never trained.

Metal is required. If it is unavailable the run stops and records `BLOCKED`
rather than substituting central-processing-unit numbers.

Usage:

    python scripts/run_phase3_agent04.py                 # full acceptance run
    python scripts/run_phase3_agent04.py --quick         # fast smoke run
    python scripts/run_phase3_agent04.py --skip-pytest   # measurements only
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_3_data"
DEFAULT_OUTPUT = DATA_DIRECTORY / "agent_04_mps_inference.json"
DEFAULT_RAW_OUTPUT = DATA_DIRECTORY / "agent_04_mps_inference_raw.csv"

AGENT_01_DATA = DATA_DIRECTORY / "agent_01_batch_equivalence.json"
AGENT_02_DATA = DATA_DIRECTORY / "agent_02_shared_memory_scaling.json"
AGENT_03_DATA = DATA_DIRECTORY / "agent_03_trajectory_reconstruction.json"

AGENT = "agent_04"

TEST_TARGET = "tests/training/test_representative_model.py"

FILES_CREATED = (
    "stratego/training/representative_model.py",
    "stratego/training/mps_benchmark.py",
    "tests/training/test_representative_model.py",
    "scripts/run_phase3_agent04.py",
    "requirements-training.txt",
    "reports/phase_3_data/agent_04_mps_inference.json",
    "reports/phase_3_data/agent_04_mps_inference_raw.csv",
)

FILES_MODIFIED = (
    # Docstring note only: the two PyTorch-dependent modules are deliberately
    # not re-exported, so `import stratego.training` still works without torch.
    "stratego/training/__init__.py",
    "reports/phase_3_implementation_report.md",
)


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------


def check_prerequisites() -> dict:
    """Agent 3 must be `PASS`; Agents 1 and 2 are recorded for the chain."""
    record: dict = {"satisfied": True, "agents": {}}
    for name, path, required in (
        ("agent_01", AGENT_01_DATA, False),
        ("agent_02", AGENT_02_DATA, False),
        ("agent_03", AGENT_03_DATA, True),
    ):
        if not path.exists():
            record["agents"][name] = {"present": False, "status": None}
            if required:
                record["satisfied"] = False
            continue
        payload = json.loads(path.read_text())
        status = payload.get("status")
        record["agents"][name] = {"present": True, "status": status}
        if required and status != "PASS":
            record["satisfied"] = False
    return record


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


def run_pytest(target: str = TEST_TARGET) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", target],
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
        "target": target,
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
# Raw CSV
# ---------------------------------------------------------------------------


RAW_COLUMNS = (
    "batch_size",
    "precision",
    "legality",
    "status",
    "iterations_per_pass",
    "trials",
    "warmup_iterations",
    "warmup_seconds",
    "end_to_end_mean_ms",
    "end_to_end_median_ms",
    "end_to_end_p95_ms",
    "end_to_end_stdev_ms",
    "transfer_mean_ms",
    "model_only_mean_ms",
    "legality_sampling_mean_ms",
    "readback_mean_ms",
    "positions_per_second",
    "model_only_positions_per_second",
    "legality_sampling_positions_per_second",
    "transfer_positions_per_second",
    "readback_positions_per_second",
    "model_share_of_end_to_end",
    "legality_share_of_end_to_end",
    "legality_host_bytes",
    "legality_bytes_per_position",
    "peak_process_memory_bytes",
    "mps_current_allocated_bytes",
    "mps_driver_allocated_bytes",
    "illegal_samples",
    "outputs_finite",
    "error",
)


def write_raw_csv(path: Path, results: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        for entry in results:
            latency = entry.get("latency", {})
            memory = entry.get("mps_memory_bytes", {})

            def stage(name: str, key: str = "mean_ms"):
                return latency.get(name, {}).get(key)

            writer.writerow(
                {
                    "batch_size": entry["batch_size"],
                    "precision": entry["precision"],
                    "legality": entry["legality"],
                    "status": entry["status"],
                    "iterations_per_pass": entry.get("iterations_per_pass"),
                    "trials": entry.get("trials"),
                    "warmup_iterations": entry.get("warmup_iterations"),
                    "warmup_seconds": entry.get("warmup_seconds"),
                    "end_to_end_mean_ms": stage("end_to_end"),
                    "end_to_end_median_ms": stage("end_to_end", "median_ms"),
                    "end_to_end_p95_ms": stage("end_to_end", "p95_ms"),
                    "end_to_end_stdev_ms": stage("end_to_end", "stdev_ms"),
                    "transfer_mean_ms": stage("transfer"),
                    "model_only_mean_ms": stage("model_only"),
                    "legality_sampling_mean_ms": stage("legality_and_sampling"),
                    "readback_mean_ms": stage("readback"),
                    "positions_per_second": entry.get("positions_per_second"),
                    "model_only_positions_per_second": entry.get(
                        "model_only_positions_per_second"
                    ),
                    "legality_sampling_positions_per_second": entry.get(
                        "legality_sampling_positions_per_second"
                    ),
                    "transfer_positions_per_second": entry.get(
                        "transfer_positions_per_second"
                    ),
                    "readback_positions_per_second": entry.get(
                        "readback_positions_per_second"
                    ),
                    "model_share_of_end_to_end": entry.get("model_share_of_end_to_end"),
                    "legality_share_of_end_to_end": entry.get(
                        "legality_share_of_end_to_end"
                    ),
                    "legality_host_bytes": entry.get("legality_host_bytes"),
                    "legality_bytes_per_position": entry.get(
                        "legality_bytes_per_position"
                    ),
                    "peak_process_memory_bytes": entry.get("peak_process_memory_bytes"),
                    "mps_current_allocated_bytes": memory.get("current_allocated_bytes"),
                    "mps_driver_allocated_bytes": memory.get("driver_allocated_bytes"),
                    "illegal_samples": entry.get("illegal_samples"),
                    "outputs_finite": entry.get("outputs_finite"),
                    "error": entry.get("error"),
                }
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def blocked_report(reason: str, prerequisites: dict, extra: dict | None = None) -> dict:
    report = {
        "agent": AGENT,
        "status": "BLOCKED",
        "blocked_reason": reason,
        "prerequisites": prerequisites,
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python_version": platform.python_version(),
        "files_created": list(FILES_CREATED),
        "files_modified": list(FILES_MODIFIED),
    }
    if extra:
        report.update(extra)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--pool-positions", type=int, default=2048)
    parser.add_argument("--pool-environments", type=int, default=128)
    parser.add_argument("--pool-stride", type=int, default=32)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--target-pass-seconds", type=float, default=2.0)
    parser.add_argument("--sustained-seconds", type=float, default=10.0)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--precisions", type=str, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=4)
    options = parser.parse_args()

    started = time.perf_counter()
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    prerequisites = check_prerequisites()
    if not prerequisites["satisfied"]:
        report = blocked_report(
            "Agent 3 is not PASS; Phase 3 agents run sequentially.", prerequisites
        )
        options.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("BLOCKED: prerequisite agent did not pass", file=sys.stderr)
        return 1

    try:
        import torch  # noqa: F401
    except ImportError as error:
        report = blocked_report(
            f"PyTorch is not installed: {error}. Agent 4 requires the Metal backend.",
            prerequisites,
            {"mps_available": False, "torch_version": None},
        )
        options.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("BLOCKED: PyTorch is unavailable", file=sys.stderr)
        return 1

    from stratego.training.mps_benchmark import (  # noqa: E402
        DEFAULT_BATCH_SIZES,
        DEFAULT_PRECISIONS,
        build_position_pool,
        detect_device_report,
        run_full_benchmark,
    )

    device_report = detect_device_report()
    if not device_report["mps_available"]:
        report = blocked_report(
            "Metal Performance Shaders are unavailable on this host. Agent 4 must "
            "not substitute central-processing-unit results for the required "
            "benchmark.",
            prerequisites,
            {
                "mps_available": False,
                "torch_version": device_report["torch_version"],
                "device": device_report,
            },
        )
        options.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("BLOCKED: MPS unavailable", file=sys.stderr)
        return 1

    batch_sizes = tuple(options.batch_sizes or DEFAULT_BATCH_SIZES)
    precisions = tuple(options.precisions or DEFAULT_PRECISIONS)
    trials = options.trials
    target_pass_seconds = options.target_pass_seconds
    sustained_seconds = options.sustained_seconds
    pool_positions = options.pool_positions
    if options.quick:
        batch_sizes = (64, 256, 2048)
        trials = 1
        target_pass_seconds = 0.3
        sustained_seconds = 2.0
        pool_positions = min(pool_positions, 512)

    print(f"[agent 4] building a real-position pool ({pool_positions} positions)...")
    pool = build_position_pool(
        target_positions=pool_positions,
        num_environments=options.pool_environments,
        collection_stride=options.pool_stride,
        root_seed=20260809,
    )
    print(f"[agent 4] pool ready in {pool.build_seconds:.1f}s: {pool.stats()}")

    def progress(message: str) -> None:
        print(f"[agent 4] {message}", flush=True)

    benchmark = run_full_benchmark(
        pool=pool,
        batch_sizes=batch_sizes,
        precisions=precisions,
        trials=trials,
        target_pass_seconds=target_pass_seconds,
        sustained_seconds=sustained_seconds,
        seed=options.seed,
        progress=progress,
    )

    tests = (
        {"skipped": True}
        if options.skip_pytest
        else run_pytest()
    )

    legality_choice = benchmark["recommended_legality_representation"]
    legality_rationale = benchmark["recommended_legality_rationale"]
    precision_choice = benchmark["recommended_precision"]
    precision_rationale = benchmark["recommended_precision_rationale"]

    successes = [entry for entry in benchmark["results"] if entry["status"] == "OK"]
    best_overall = (
        max(successes, key=lambda entry: entry["positions_per_second"])
        if successes
        else None
    )
    best_model_only = (
        max(successes, key=lambda entry: entry["model_only_positions_per_second"])
        if successes
        else None
    )
    baseline_dense_float32 = [
        entry
        for entry in successes
        if entry["precision"] == "float32" and entry["legality"] == "dense"
    ]
    best_baseline = (
        max(baseline_dense_float32, key=lambda entry: entry["positions_per_second"])
        if baseline_dense_float32
        else None
    )

    illegal_total = sum(entry.get("illegal_samples", 0) for entry in successes)
    non_finite = [entry for entry in successes if not entry.get("outputs_finite", False)]
    equivalence_ok = all(
        check["equivalent"] for check in benchmark["legality_equivalence"]
    )
    determinism_ok = all(
        record["max_absolute_logit_difference"] == 0.0
        for record in benchmark["determinism"].values()
    )
    completed_batches = set(benchmark["batch_sizes_completed"])
    all_batches_done = set(batch_sizes) <= completed_batches

    tests_ok = options.skip_pytest or (
        tests.get("test_exit_code") == 0 and tests.get("test_failed", 0) == 0
    )

    gate = {
        "mps_available": True,
        "float32_baseline_measured": best_baseline is not None,
        "all_requested_batch_sizes_measured": all_batches_done,
        "no_illegal_samples": illegal_total == 0,
        "all_outputs_finite": not non_finite,
        "dense_compact_equivalent": equivalence_ok,
        "repeat_stability": determinism_ok,
        "dense_and_compact_compared": (
            benchmark["best_dense_configuration"] is not None
            and benchmark["best_compact_configuration"] is not None
        ),
        "sustainable_rate_recorded": (
            benchmark.get("sustained_throughput") is not None
            and benchmark.get("sustained_throughput_recommended") is not None
        ),
        "tests_passed": tests_ok,
    }
    status = "PASS" if all(gate.values()) else "FAIL"

    report = {
        "agent": AGENT,
        "status": status,
        "completion_gate": gate,
        "prerequisites": prerequisites,
        "quick_mode": options.quick,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        # -- platform / device ------------------------------------------------
        "platform": device_report["platform"],
        "platform_full": device_report["platform_full"],
        "python_version": device_report["python_version"],
        "torch_version": device_report["torch_version"],
        "numpy_version": device_report["numpy_version"],
        "mps_available": device_report["mps_available"],
        "mps_built": device_report["mps_built"],
        "mps_device_info_if_available": device_report["mps_device_info"],
        # -- frozen contract identifiers --------------------------------------
        "implementation_version": benchmark["implementation_version"],
        "observation_version": benchmark["observation_version"],
        "benchmark_version": benchmark["benchmark_version"],
        "representative_model_version": benchmark["representative_model_version"],
        # -- model -------------------------------------------------------------
        "representative_model_parameter_count": benchmark[
            "representative_model_parameter_count"
        ],
        "architecture_summary": benchmark["architecture_summary"],
        "is_benchmark_probe": True,
        "model_disclaimer": (
            "Representative benchmark probe of the planned shape. NOT the frozen "
            "model architecture and never trained for playing strength."
        ),
        # -- sweep ---------------------------------------------------------------
        "position_pool": benchmark["position_pool"],
        "batch_sizes": list(batch_sizes),
        "batch_sizes_completed": benchmark["batch_sizes_completed"],
        "precision_modes": list(precisions),
        "precision_support": benchmark["precision_support"],
        "precision_stability": benchmark["precision_stability"],
        "legality_equivalence": benchmark["legality_equivalence"],
        "determinism": benchmark["determinism"],
        "dense_legality_results": benchmark["dense_legality_results"],
        "compact_legality_results": benchmark["compact_legality_results"],
        "best_dense_configuration": benchmark["best_dense_configuration"],
        "best_compact_configuration": benchmark["best_compact_configuration"],
        "compact_capacity_sensitivity": benchmark["compact_capacity_sensitivity"],
        "compact_capacity_sensitivity_batch_size": benchmark[
            "compact_capacity_sensitivity_batch_size"
        ],
        "legality_ab_repeatability": benchmark["legality_ab_repeatability"],
        "best_float32_dense_configuration": best_baseline,
        # -- recommendations -------------------------------------------------------
        "recommended_legality_representation": legality_choice,
        "recommended_legality_rationale": legality_rationale,
        "recommended_precision": precision_choice,
        "recommended_precision_rationale": precision_rationale,
        # -- headline rates ---------------------------------------------------------
        "best_inference_positions_per_second": (
            best_model_only["model_only_positions_per_second"] if best_model_only else None
        ),
        "best_inference_configuration": best_model_only,
        "best_end_to_end_model_step_positions_per_second": (
            best_overall["positions_per_second"] if best_overall else None
        ),
        "best_end_to_end_configuration": best_overall,
        "sustained_throughput": benchmark.get("sustained_throughput"),
        "sustained_throughput_recommended": benchmark.get(
            "sustained_throughput_recommended"
        ),
        "sustained_throughput_float32_dense": benchmark.get(
            "sustained_throughput_float32_dense"
        ),
        # The headline sustainable rate is the *recommended* configuration, not
        # the fastest one measured: Phase 4 will run what Agent 4 recommends.
        "sustainable_representative_model_positions_per_second": (
            benchmark.get("sustained_throughput_recommended")
            or benchmark.get("sustained_throughput")
            or {}
        ).get("positions_per_second"),
        "sustainable_representative_model_positions_per_second_float32_dense": (
            benchmark.get("sustained_throughput_float32_dense") or {}
        ).get("positions_per_second"),
        # -- memory / failures -----------------------------------------------------
        "peak_memory_bytes": benchmark["peak_memory_bytes"],
        "mps_peak_memory_bytes_if_available": benchmark["mps_memory_bytes"],
        "failures": benchmark["failures"],
        "illegal_samples_total": illegal_total,
        # -- tests -------------------------------------------------------------------
        "tests": tests,
        # -- files -------------------------------------------------------------------
        "files_created": list(FILES_CREATED),
        "files_modified": list(FILES_MODIFIED),
    }

    options.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_raw_csv(options.raw_output, benchmark["results"])

    print()
    print(f"status: {status}")
    print(f"parameters: {report['representative_model_parameter_count']:,}")
    if best_overall:
        print(
            f"best end-to-end model step: {best_overall['positions_per_second']:,.0f} "
            f"positions/s (batch {best_overall['batch_size']}, "
            f"{best_overall['precision']}, {best_overall['legality']} legality)"
        )
    if best_model_only:
        print(
            "best model-only inference: "
            f"{best_model_only['model_only_positions_per_second']:,.0f} positions/s"
        )
    if report["sustained_throughput"]:
        print(
            "sustained (fastest measured): "
            f"{report['sustained_throughput']['positions_per_second']:,.0f} positions/s"
        )
    if report["sustained_throughput_recommended"]:
        recommended = report["sustained_throughput_recommended"]
        print(
            "sustained (recommended configuration): "
            f"{recommended['positions_per_second']:,.0f} positions/s "
            f"(batch {recommended['batch_size']}, {recommended['precision']}, "
            f"{recommended['legality']} legality)"
        )
    if report["sustained_throughput_float32_dense"]:
        print(
            "sustained (float32 + dense baseline): "
            f"{report['sustained_throughput_float32_dense']['positions_per_second']:,.0f}"
            " positions/s"
        )
    print(f"recommended legality: {legality_choice}")
    print(f"recommended precision: {precision_choice}")
    print(f"failures: {len(report['failures'])}")
    print(f"wrote {options.output}")
    print(f"wrote {options.raw_output}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
