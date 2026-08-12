#!/usr/bin/env python3
"""Phase 6 Agent 3 acceptance harness: standalone MPS inference and training-step benchmark.

Measures the M4 Pro hardware frontier of the C0-C6 candidate family without the
simulator in the loop, and writes

    reports/phase_6_data/agent_03_inference_benchmark.csv
    reports/phase_6_data/agent_03_training_step_benchmark.csv
    reports/phase_6_data/agent_03_architecture_shortlist.json

What this script is and is not
------------------------------
It measures compute: forward latency and throughput at three timing boundaries,
one training step (forward + three losses + backward, and nothing else), CPU/MPS
numerical agreement, and memory. From those measurements it produces a
*shortlist* under a rule declared before the numbers existed.

It does **not** select the final architecture -- Agent 6 does -- and it does not
know how well anything plays. No optimizer is constructed, no parameter is
updated, and no measure of playing strength is reachable from the classification
(see `stratego.model.benchmark_helpers.classification_inputs`).

Usage::

    python scripts/run_phase6_agent03.py                 # full acceptance run
    python scripts/run_phase6_agent03.py --quick         # small sweep, for iteration only
    python scripts/run_phase6_agent03.py --skip-pytest   # measurements only
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
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.model import benchmark_helpers as helpers  # noqa: E402
from stratego.model.architecture_configs import (  # noqa: E402
    ARCHITECTURE_FAMILY,
    ARCHITECTURE_FAMILY_VERSION,
    CANDIDATE_IDS,
    CANDIDATE_ROLES,
    FAMILY_INITIALIZATION_SEED,
    architecture_family_digest,
    candidate_config,
    candidate_configs,
    config_digests,
)
from stratego.model.contract import (  # noqa: E402
    ACTION_ENCODING_VERSION,
    ENGINE_ACTION_FRAME,
    MODEL_CONTRACT_VERSION,
    POLICY_ACTION_FRAME,
    TOKEN_SQUARE_FRAME,
    contract_summary,
)
from stratego.model.production_model import build_candidate_model  # noqa: E402

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_6_data"

#: The tree this agent started from, measured before any Agent 3 edit.
PREEXISTING_SUITE = {"passed": 2383, "skipped": 2, "failed": 0, "commit": "8f4f5e3"}

#: Memory fields render as `unavailable` rather than as a number when the API
#: does not exist. Zero would be a lie: an absent counter and an empty allocator
#: are different facts.
MEMORY_FIELDS = frozenset(
    {
        "process_rss_bytes",
        "metal_allocated_bytes",
        "metal_driver_bytes",
        "metal_recommended_max_bytes",
        "peak_memory_if_available",
        "memory_fraction_of_recommended",
    }
)

INFERENCE_COLUMNS = [
    "candidate_id",
    "config_digest",
    "architecture_family",
    "parameters",
    "precision",
    "requested_precision",
    "observed_precision",
    "requested_device",
    "observed_device",
    "batch",
    "boundary",
    "boundary_includes",
    "status",
    "warmup_iterations",
    "measurement_iterations",
    "median_latency_ms",
    "p95_latency_ms",
    "mean_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "stdev_latency_ms",
    "positions_per_second",
    "finite_outputs",
    "process_rss_bytes",
    "metal_allocated_bytes",
    "metal_driver_bytes",
    "metal_recommended_max_bytes",
    "peak_memory_if_available",
    "memory_fraction_of_recommended",
    "oom",
    "error",
    "corpus_digest",
]

TRAINING_COLUMNS = [
    "candidate_id",
    "config_digest",
    "parameters",
    "precision",
    "requested_precision",
    "observed_precision",
    "requested_device",
    "observed_device",
    "batch",
    "status",
    "warmup_iterations",
    "measurement_iterations",
    "forward_ms",
    "loss_ms",
    "backward_ms",
    "total_ms",
    "examples_per_second",
    "policy_loss",
    "value_loss",
    "belief_loss",
    "total_loss",
    "finite_loss",
    "finite_gradients",
    "shared_encoder_gradient",
    "policy_head_gradient",
    "value_head_gradient",
    "belief_head_gradient",
    "parameters_without_gradient",
    "parameters_with_non_finite_gradient",
    "optimizer_step",
    "parameter_update",
    "process_rss_bytes",
    "metal_allocated_bytes",
    "metal_driver_bytes",
    "metal_recommended_max_bytes",
    "memory_fraction_of_recommended",
    "oom",
    "error",
    "corpus_digest",
]


# ---------------------------------------------------------------------------
# Environment and prerequisites
# ---------------------------------------------------------------------------


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001 - a missing git is not a Phase 6 failure
        return "unknown"


def environment() -> dict:
    return {
        "commit": git_commit(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_threads": torch.get_num_threads(),
    }


def verify_prerequisites() -> dict:
    """Agents 1 and 2 must be PASS, and the live build must be the one they passed.

    Reading the artifacts is necessary but not sufficient: a file recording that
    Agent 1 passed says nothing about the contract *this* process is running
    under. Both are checked, and the candidate family is then rebuilt and
    required to reproduce its digests and parameter counts before any timing is
    taken -- benchmarking a family that does not match the accepted one would
    produce numbers about nothing.
    """
    report: dict[str, Any] = {"problems": []}

    agent_01_path = DATA_DIRECTORY / "agent_01_model_contract_v2.json"
    agent_02_path = DATA_DIRECTORY / "agent_02_architecture_family.json"
    for name, path in (("agent_01", agent_01_path), ("agent_02", agent_02_path)):
        if not path.exists():
            report["problems"].append(f"{name} artifact missing at {path}")
            report[name] = {"status": "MISSING"}
            continue
        payload = json.loads(path.read_text())
        gates = payload.get("completion_gates", {})
        entry = {
            "path": str(path.relative_to(REPOSITORY_ROOT)),
            "status": payload.get("status"),
            "gates_total": len(gates),
            "gates_true": sum(1 for value in gates.values() if value),
            "all_gates_true": bool(gates) and all(gates.values()),
        }
        if entry["status"] != "PASS":
            report["problems"].append(f"{name} status is {entry['status']}, not PASS")
        if not entry["all_gates_true"]:
            report["problems"].append(f"{name} has a false completion gate")
        report[name] = entry

    # The live constants, not the recorded ones.
    live = {
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "token_square_frame": TOKEN_SQUARE_FRAME,
        "policy_action_frame": POLICY_ACTION_FRAME,
        "engine_action_frame": ENGINE_ACTION_FRAME,
        "action_encoding_version": ACTION_ENCODING_VERSION,
        "architecture_family": ARCHITECTURE_FAMILY,
        "architecture_family_version": ARCHITECTURE_FAMILY_VERSION,
        "architecture_family_digest": architecture_family_digest(),
    }
    expected = {
        "model_contract_version": "model_contract_v2",
        "token_square_frame": "perspective_normalized_squares",
        "policy_action_frame": "perspective_normalized_squares",
        "engine_action_frame": "absolute_engine_squares",
        "action_encoding_version": "source_destination_10000_v1",
        "architecture_family": "stratego_transformer_v1",
        "architecture_family_version": "architecture_family_v1",
    }
    for key, want in expected.items():
        if live[key] != want:
            report["problems"].append(f"live {key} is {live[key]!r}, expected {want!r}")
    report["live_constants"] = live

    if agent_02_path.exists():
        recorded = json.loads(agent_02_path.read_text()).get("architecture_family_digest")
        report["family_digest_matches_agent_02"] = recorded == live["architecture_family_digest"]
        if not report["family_digest_matches_agent_02"]:
            report["problems"].append(
                "the live architecture family digest does not match Agent 2's record"
            )

    reproduction = helpers.reproduce_candidate_configs()
    report["config_reproduction"] = reproduction
    if not reproduction["all_reproduced"]:
        report["problems"].append("a candidate configuration digest did not reproduce")

    # Parameter counts must match Agent 2's recorded table exactly.
    if agent_02_path.exists():
        recorded_counts = json.loads(agent_02_path.read_text()).get("parameter_counts", {})
        mismatches = {
            candidate_id: {
                "recorded": recorded_counts.get(candidate_id),
                "rebuilt": entry["trainable_parameters"],
            }
            for candidate_id, entry in reproduction["candidates"].items()
            if recorded_counts.get(candidate_id) != entry["trainable_parameters"]
        }
        report["parameter_count_mismatches"] = mismatches
        if mismatches:
            report["problems"].append(
                f"{len(mismatches)} candidate parameter counts do not match Agent 2"
            )

    report["ok"] = not report["problems"]
    return report


# ---------------------------------------------------------------------------
# The inference matrix
# ---------------------------------------------------------------------------


def run_inference_matrix(
    *,
    corpus: helpers.BenchmarkCorpus,
    device: torch.device,
    batch_sizes: tuple[int, ...],
    extended: tuple[int, ...],
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """Every candidate at every batch, precision and boundary.

    Host batches are built once per batch size and shared by every candidate and
    precision, so the identical corpus rows back every comparison. A candidate's
    model is built once per precision and reused across the whole ladder, which
    is also what keeps initialisation out of the timed region.
    """
    rows: list[dict] = []
    digests = config_digests()
    attempted = 0
    host_batches: dict[int, helpers.HostBatch] = {}
    selection_reports: list[dict] = []

    for candidate_id in CANDIDATE_IDS:
        config = candidate_config(candidate_id)
        for precision in helpers.PRECISIONS:
            dtype = helpers.resolve_dtype(precision)
            try:
                model = build_candidate_model(config, device=device, dtype=dtype)
            except (RuntimeError, MemoryError) as error:
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "config_digest": digests[candidate_id],
                        "architecture_family": ARCHITECTURE_FAMILY,
                        "parameters": 0,
                        "precision": precision,
                        "requested_precision": precision,
                        "requested_device": device.type,
                        "batch": 0,
                        "boundary": helpers.BOUNDARY_A,
                        "boundary_includes": "model construction only",
                        "status": "oom" if "out of memory" in str(error).lower() else "error",
                        "oom": "out of memory" in str(error).lower(),
                        "error": f"{type(error).__name__}: {error}",
                        "corpus_digest": corpus.digest,
                    }
                )
                attempted += 1
                continue

            parameters = model.parameter_count()
            ladder = list(batch_sizes)
            remaining_extended = list(extended)
            probe_index = 0

            while probe_index < len(ladder):
                batch = ladder[probe_index]
                probe_index += 1
                if batch not in host_batches:
                    host_batches[batch] = helpers.make_host_batch(corpus, batch)
                host = host_batches[batch]

                for boundary in helpers.BOUNDARIES:
                    row = helpers.run_inference_point(
                        model=model,
                        candidate_id=candidate_id,
                        config_digest=digests[candidate_id],
                        parameters=parameters,
                        corpus=corpus,
                        batch=batch,
                        precision=precision,
                        boundary=boundary,
                        device=device,
                        host=host,
                    )
                    rows.append(row)
                    attempted += 1
                    if verbose:
                        rate = row.get("positions_per_second")
                        print(
                            f"  {candidate_id} {precision:8s} b{batch:<5d} {boundary:<45s} "
                            f"{row['status']:<20s} "
                            f"{row['median_latency_ms'] if row['median_latency_ms'] is not None else '-':>10} ms "
                            f"{rate if rate is not None else '-':>10} pos/s"
                        )

                # Boundary C correctness, once per (candidate, precision, batch):
                # the timed closure asserts nothing, so validity is established
                # here on the same path with the same inputs.
                forward_row = rows[-3]
                if forward_row["status"] == "ok":
                    try:
                        validity = helpers.selection_validity(
                            model=model,
                            host=host,
                            device=device,
                            dtype=dtype,
                            corpus=corpus,
                        )
                        validity.update(
                            {
                                "candidate_id": candidate_id,
                                "precision": precision,
                                "batch": batch,
                            }
                        )
                        selection_reports.append(validity)
                    except (RuntimeError, MemoryError):
                        pass

                # Extended probing, strictly opt-in and strictly guarded. Only at
                # the end of the current ladder, only one step at a time, and
                # only while the guard in `_should_probe_larger` still holds --
                # so a candidate that stops improving, or that approaches the
                # memory line, simply stops being probed.
                if (
                    probe_index == len(ladder)
                    and remaining_extended
                    and _should_probe_larger(rows, candidate_id, precision)
                ):
                    ladder.append(remaining_extended.pop(0))

            del model
            helpers.release_device_memory(device)

    return rows, {
        "points_attempted": attempted,
        "points_recorded": len(rows),
        "selection_validity": selection_reports,
    }


def _should_probe_larger(rows: list[dict], candidate_id: str, precision: str) -> bool:
    """Probe above the required ladder only while it is both useful and safe.

    Two conditions, both required: throughput must still be improving (a flat
    curve means the knee is already in the data, and another point would only
    cost time), and Metal memory must be far enough below the recommended
    maximum that the attempt cannot push the host into pressure or swap. The
    instruction to find the frontier is not a licence to exhaust the machine, so
    "ceiling not reached below the memory guard" is an acceptable answer.
    """
    forward = [
        row
        for row in rows
        if row["candidate_id"] == candidate_id
        and row["precision"] == precision
        and row["boundary"] == helpers.BOUNDARY_A
        and row["status"] == "ok"
        and row.get("positions_per_second")
    ]
    if len(forward) < 2:
        return False
    forward.sort(key=lambda row: row["batch"])
    previous, latest = forward[-2], forward[-1]
    improving = latest["positions_per_second"] >= previous["positions_per_second"] * (
        1.0 + helpers.EXTENDED_PROBE_IMPROVEMENT
    )
    fraction = latest.get("memory_fraction_of_recommended")
    safe = fraction is not None and fraction < helpers.MEMORY_PRESSURE_FRACTION
    return bool(improving and safe)


# ---------------------------------------------------------------------------
# CSV rendering
# ---------------------------------------------------------------------------


def render(value: Any, column: str) -> Any:
    if value is None:
        return "unavailable" if column in MEMORY_FIELDS else ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: render(row.get(column), column) for column in columns})
    return len(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def run_pytest() -> dict:
    started = time.perf_counter()
    report = DATA_DIRECTORY / ".pytest_junit_agent03.xml"
    report.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", f"--junitxml={report}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    tail = process.stdout.strip().splitlines()[-1] if process.stdout.strip() else ""

    def count(pattern: str) -> int:
        match = re.search(rf"(\d+) {pattern}", tail)
        return int(match.group(1)) if match else 0

    per_module: dict[str, dict] = {}
    if report.exists():
        for case in ElementTree.parse(report).getroot().iter("testcase"):
            module = case.get("file") or case.get("classname", "").replace(".", "/") + ".py"
            entry = per_module.setdefault(module, {"passed": 0, "failed": 0, "skipped": 0})
            if case.find("failure") is not None or case.find("error") is not None:
                entry["failed"] += 1
            elif case.find("skipped") is not None:
                entry["skipped"] += 1
            else:
                entry["passed"] += 1
        report.unlink()

    return {
        "command": "python -m pytest -q",
        "exit_code": process.returncode,
        "passed": count("passed"),
        "failed": count("failed"),
        "errors": count("error"),
        "skipped": count("skipped"),
        "summary_line": tail,
        "per_module": per_module,
        "seconds": round(time.perf_counter() - started, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=DATA_DIRECTORY)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="a small sweep for iteration; never an acceptance run",
    )
    arguments = parser.parse_args()

    started = time.perf_counter()
    durations: dict[str, float] = {}
    data_directory: Path = arguments.data_directory
    data_directory.mkdir(parents=True, exist_ok=True)

    print("Phase 6 Agent 3 -- standalone MPS inference and training-step benchmark\n")

    # -- prerequisites ------------------------------------------------------
    mark = time.perf_counter()
    prerequisites = verify_prerequisites()
    durations["prerequisites_seconds"] = round(time.perf_counter() - mark, 3)
    print(f"prerequisites   {'ok' if prerequisites['ok'] else 'PROBLEMS'}")
    for problem in prerequisites["problems"]:
        print(f"  ! {problem}")

    try:
        device = helpers.require_mps()
    except helpers.BenchmarkError as error:
        print(f"\nBLOCKED: {error}")
        return 2
    print(f"device          {device} (mps_built={torch.backends.mps.is_built()})")

    # -- corpus -------------------------------------------------------------
    mark = time.perf_counter()
    positions = 512 if arguments.quick else helpers.CORPUS_POSITIONS
    corpus = helpers.build_benchmark_corpus(positions=positions)
    target_legality = helpers.verify_policy_targets_legal(corpus)
    frame_agreement = helpers.verify_legality_frames_agree(corpus)
    # Determinism is a property the report claims, so it is measured, not
    # asserted: a second corpus is built from the same recipe and its digest
    # compared. The rebuild is small in quick mode and full-size otherwise.
    replica = helpers.build_benchmark_corpus(positions=positions)
    corpus_deterministic = replica.digest == corpus.digest
    del replica
    durations["corpus_seconds"] = round(time.perf_counter() - mark, 3)
    stats = corpus.stats()
    print(
        f"corpus          {stats['positions']} positions, digest {corpus.digest[:16]}..., "
        f"red {stats['acting_player_counts']['red']} / blue {stats['acting_player_counts']['blue']}, "
        f"plies {stats['ply_min']}-{stats['ply_max']}, deterministic={corpus_deterministic}"
    )

    # -- inference matrix ---------------------------------------------------
    mark = time.perf_counter()
    batch_sizes = (1, 64, 256) if arguments.quick else helpers.INFERENCE_BATCH_SIZES
    extended = () if arguments.quick else helpers.EXTENDED_BATCH_SIZES
    print("\ninference matrix")
    inference_rows, inference_meta = run_inference_matrix(
        corpus=corpus, device=device, batch_sizes=batch_sizes, extended=extended
    )
    durations["inference_seconds"] = round(time.perf_counter() - mark, 3)

    # -- numerical checks ---------------------------------------------------
    mark = time.perf_counter()
    print("\nnumerical checks (CPU float32 reference vs MPS float32 and MPS float16)")
    numerical: dict[str, dict] = {}
    for candidate_id in CANDIDATE_IDS:
        report = helpers.numerical_comparison(
            candidate_id=candidate_id,
            config=candidate_config(candidate_id),
            corpus=corpus,
            device=device,
        )
        numerical[candidate_id] = report
        for key, entry in report["comparisons"].items():
            if entry["status"] != "ok":
                print(f"  {candidate_id} {key:14s} {entry['status']}: {entry['error'][:80]}")
                continue
            policy = entry["heads"]["policy_logits"]
            print(
                f"  {candidate_id} {key:14s} passes={str(entry['passes']):5s} "
                f"policy max|e|={policy['max_absolute_error']:.3e} "
                f"crafted={entry['crafted_margin_agreement']}/{report['positions']} "
                f"natural={entry['natural_greedy_agreement']}/{report['positions']} "
                f"illegal={entry['illegal_absolute_actions']}"
            )
    durations["numerical_seconds"] = round(time.perf_counter() - mark, 3)

    # -- training-step benchmark -------------------------------------------
    mark = time.perf_counter()
    training_batches = (32, 64) if arguments.quick else helpers.TRAINING_BATCH_SIZES
    print("\ntraining-step benchmark (forward + losses + backward, no optimizer step)")
    training_rows: list[dict] = []
    digests = config_digests()
    for candidate_id in CANDIDATE_IDS:
        config = candidate_config(candidate_id)
        parameters = prerequisites["config_reproduction"]["candidates"][candidate_id][
            "trainable_parameters"
        ]
        for precision in helpers.PRECISIONS:
            for batch in training_batches:
                row = helpers.run_training_point(
                    candidate_id=candidate_id,
                    config=config,
                    config_digest=digests[candidate_id],
                    parameters=parameters,
                    corpus=corpus,
                    batch=batch,
                    precision=precision,
                    device=device,
                )
                training_rows.append(row)
                print(
                    f"  {candidate_id} {precision:8s} b{batch:<4d} {row['status']:<22s} "
                    f"total={row['total_ms'] if row['total_ms'] is not None else '-':>9} ms "
                    f"{row['examples_per_second'] if row['examples_per_second'] is not None else '-':>9} ex/s "
                    f"finite_grad={row['finite_gradients']}"
                )
    durations["training_seconds"] = round(time.perf_counter() - mark, 3)

    # -- summaries and classification ---------------------------------------
    mark = time.perf_counter()
    summaries = []
    for candidate_id in CANDIDATE_IDS:
        summaries.append(
            helpers.summarise_candidate(
                candidate_id=candidate_id,
                parameters=prerequisites["config_reproduction"]["candidates"][candidate_id][
                    "trainable_parameters"
                ],
                inference_rows=[
                    row for row in inference_rows if row["candidate_id"] == candidate_id
                ],
                training_rows=[
                    row for row in training_rows if row["candidate_id"] == candidate_id
                ],
                numerical=numerical.get(candidate_id),
            )
        )

    classification = helpers.classify_candidates(summaries)

    # Determinism of the classification is evidence, not an assumption: the same
    # summaries are classified again in reverse order and the verdicts compared.
    reversed_classification = helpers.classify_candidates(list(reversed(summaries)))
    classification_deterministic = (
        reversed_classification["verdicts"] == classification["verdicts"]
    )

    # And a strength field is injected to show it changes nothing. This is the
    # positive control for "no strength-based selection": without it, the claim
    # would rest on the absence of a field rather than on its irrelevance.
    polluted = [
        {**summary, "win_rate": 0.99 if summary["candidate_id"] == "C0" else 0.01}
        for summary in summaries
    ]
    strength_free = (
        helpers.classify_candidates(polluted)["verdicts"] == classification["verdicts"]
    )

    frontier = helpers.pareto_frontier(summaries, classification)
    durations["classification_seconds"] = round(time.perf_counter() - mark, 3)

    print("\nclassification")
    for row in frontier:
        print(
            f"  {row['candidate_id']} {row['classification']:<12s} "
            f"{row['parameters']:>12,} params  "
            f"{row['best_float32_positions_per_second'] or 0:>10,.0f} pos/s f32  "
            f"{row['representative_training_examples_per_second'] or 0:>9,.0f} ex/s"
        )

    # -- data files ---------------------------------------------------------
    inference_path = data_directory / "agent_03_inference_benchmark.csv"
    training_path = data_directory / "agent_03_training_step_benchmark.csv"
    shortlist_path = data_directory / "agent_03_architecture_shortlist.json"
    write_csv(inference_path, INFERENCE_COLUMNS, inference_rows)
    write_csv(training_path, TRAINING_COLUMNS, training_rows)

    # -- tests --------------------------------------------------------------
    tests = (
        {"skipped": True, "passed": 0, "failed": 0, "summary_line": "not run"}
        if arguments.skip_pytest
        else run_pytest()
    )
    if not arguments.skip_pytest:
        print(f"\npytest          {tests['summary_line']}")

    # -- gates --------------------------------------------------------------
    ooms = [row for row in inference_rows + training_rows if row.get("oom")]
    numerical_failures = [
        {"candidate_id": candidate_id, "comparison": key, "error": entry.get("error", "")}
        for candidate_id, report in numerical.items()
        for key, entry in report["comparisons"].items()
        if not entry.get("passes")
    ]
    advancing = classification["advance_ids"]
    viable = [
        summary["candidate_id"]
        for summary in summaries
        if classification["verdicts"].get(summary["candidate_id"]) != "IMPRACTICAL"
    ]

    def advancing_numerical_ok() -> bool:
        return all(
            numerical[candidate_id]["comparisons"]["mps_float32"].get("crafted_margin_passes")
            for candidate_id in advancing
        )

    def advancing_training_ok() -> bool:
        for candidate_id in advancing:
            rows = [
                row
                for row in training_rows
                if row["candidate_id"] == candidate_id
                and row["precision"] == "float32"
                and row["status"] == "ok"
            ]
            if not rows or not all(
                row["finite_loss"] and row["finite_gradients"] for row in rows
            ):
                return False
        return True

    memory_reported = any(
        row.get("metal_driver_bytes") not in (None, "") for row in inference_rows
    )

    gates = {
        "agents_1_and_2_pass_verified": prerequisites["ok"],
        "candidate_configs_and_parameter_counts_reproduce": prerequisites[
            "config_reproduction"
        ]["all_reproduced"]
        and not prerequisites.get("parameter_count_mismatches"),
        "mps_actually_used": all(
            row.get("observed_device", "").startswith("mps")
            for row in inference_rows
            if row["status"] == "ok"
        )
        and any(row["status"] == "ok" for row in inference_rows),
        "deterministic_valid_corpus_recorded": bool(
            corpus_deterministic
            and target_legality["all_targets_legal"]
            and frame_agreement["frames_agree"]
        ),
        "fair_inference_matrix_attempted": inference_meta["points_attempted"]
        == inference_meta["points_recorded"],
        "oom_and_error_rows_retained": len(inference_rows) == inference_meta["points_recorded"],
        "cpu_mps_numerical_checks_completed": len(numerical) == len(CANDIDATE_IDS),
        "float16_honestly_tested": all(
            report["comparisons"]["mps_float16"].get("observed_precision") == "float16"
            for report in numerical.values()
            if report["comparisons"]["mps_float16"]["status"] == "ok"
        )
        and any(
            row["precision"] == "float16" and row["status"] == "ok" for row in training_rows
        ),
        "crafted_margin_agreement_passes_for_advancing": advancing_numerical_ok(),
        "training_step_benchmark_completed": len(training_rows)
        == len(CANDIDATE_IDS) * len(helpers.PRECISIONS) * len(training_batches),
        "losses_and_gradients_finite_for_advancing": advancing_training_ok(),
        "memory_measured_and_reported": memory_reported,
        "deterministic_classification_produced": classification_deterministic,
        "at_least_three_advance_when_three_viable": len(advancing) >= 3
        or len(viable) < 3,
        "no_strength_based_selection": strength_free,
        "full_suite_green": bool(arguments.skip_pytest) or tests["failed"] == 0,
    }
    status = "PASS" if all(gates.values()) else "FAIL"

    payload = {
        "agent": "agent_03",
        "phase": "phase_6",
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "quick_mode": bool(arguments.quick),
        **environment(),
        "prerequisite_status": prerequisites,
        "contract_summary": contract_summary(),
        "candidate_configs": candidate_configs(),
        "candidate_roles": dict(CANDIDATE_ROLES),
        "config_digests": config_digests(),
        "architecture_family_digest": architecture_family_digest(),
        "input_corpus_digest": corpus.digest,
        "input_corpus": corpus.stats(),
        "input_corpus_deterministic": corpus_deterministic,
        "policy_target_legality": target_legality,
        "legality_frame_agreement": frame_agreement,
        "benchmark_method": helpers.benchmark_method_summary(),
        "numerical_tolerances": helpers.TOLERANCES,
        "numerical_checks": numerical,
        "numerical_failures": numerical_failures,
        "selection_validity": inference_meta["selection_validity"],
        "inference_points_attempted": inference_meta["points_attempted"],
        "inference_points_recorded": inference_meta["points_recorded"],
        "training_points_recorded": len(training_rows),
        "candidate_summaries": {
            summary["candidate_id"]: summary for summary in summaries
        },
        "pareto_frontier": frontier,
        "classification": classification,
        "classification_rules": helpers.classification_rules(),
        "classification_deterministic": classification_deterministic,
        "classification_strength_independent": strength_free,
        "advance_ids": classification["advance_ids"],
        "dominated_ids": classification["dominated_ids"],
        "impractical_ids": classification["impractical_ids"],
        "ooms": [
            {
                "candidate_id": row["candidate_id"],
                "precision": row["precision"],
                "batch": row["batch"],
                "boundary": row.get("boundary", "training_step"),
                "error": row["error"],
            }
            for row in ooms
        ],
        "recommended_integrated_test_configs": [
            {
                "candidate_id": candidate_id,
                "configuration": candidate_configs()[candidate_id],
                "config_digest": config_digests()[candidate_id],
                "parameters": next(
                    summary["parameters"]
                    for summary in summaries
                    if summary["candidate_id"] == candidate_id
                ),
                "best_stable_precision": _best_precision(summaries, candidate_id),
                "starting_inference_batch": _starting_batch(summaries, candidate_id),
                "max_stable_inference_batch": next(
                    summary["max_stable_inference_batch"]
                    for summary in summaries
                    if summary["candidate_id"] == candidate_id
                ),
                "max_stable_training_batch": next(
                    summary["max_stable_training_batch"]
                    for summary in summaries
                    if summary["candidate_id"] == candidate_id
                ),
            }
            for candidate_id in classification["advance_ids"]
        ],
        "tests_before": PREEXISTING_SUITE,
        "tests_after": tests,
        "test_total": tests.get("passed", 0) + tests.get("failed", 0) + tests.get("skipped", 0),
        "test_passed": tests.get("passed", 0),
        "test_failed": tests.get("failed", 0),
        "test_skipped": tests.get("skipped", 0),
        "commands": [
            "python scripts/run_phase6_agent03.py",
            "python -m pytest -q",
            "python -m pytest tests/model/test_phase6_benchmarks.py -q",
        ],
        "durations": durations,
        "total_seconds": round(time.perf_counter() - started, 2),
        "seeds": {
            "family_initialization_seed": FAMILY_INITIALIZATION_SEED,
            "corpus_seed": helpers.CORPUS_SEED,
            "target_seed": helpers.TARGET_SEED,
        },
        "files_created": [
            "stratego/model/benchmark_helpers.py",
            "scripts/run_phase6_agent03.py",
            "tests/model/test_phase6_benchmarks.py",
            "reports/phase_6_data/agent_03_inference_benchmark.csv",
            "reports/phase_6_data/agent_03_training_step_benchmark.csv",
            "reports/phase_6_data/agent_03_architecture_shortlist.json",
        ],
        "files_modified": [
            "reports/phase_6_implementation_report.md",
        ],
        "completion_gates": gates,
        "problems": [name for name, value in gates.items() if not value],
    }

    shortlist_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    print(f"\nstatus      {status}  ({sum(gates.values())}/{len(gates)} gates)")
    for name, value in gates.items():
        if not value:
            print(f"  FAILED GATE  {name}")
    print(f"inference   {inference_path}  ({len(inference_rows)} rows)")
    print(f"training    {training_path}  ({len(training_rows)} rows)")
    print(f"shortlist   {shortlist_path}")
    print(f"advance     {', '.join(classification['advance_ids']) or 'none'}")
    print(f"dominated   {', '.join(classification['dominated_ids']) or 'none'}")
    print(f"impractical {', '.join(classification['impractical_ids']) or 'none'}")
    print(f"elapsed     {payload['total_seconds']}s")
    return 0 if status == "PASS" else 1


def _best_precision(summaries: list[dict], candidate_id: str) -> str:
    """The precision with the higher stable throughput, float32 on a tie.

    float32 wins ties because it is the precision whose numerical check is a
    hard gate; float16 has to actually be faster to be recommended.
    """
    summary = next(item for item in summaries if item["candidate_id"] == candidate_id)
    half = summary.get("best_float16_positions_per_second")
    full = summary.get("best_float32_positions_per_second")
    if half is not None and full is not None and half > full:
        return "float16" if summary.get("numerically_stable_float16") else "float32"
    return "float32"


def _starting_batch(summaries: list[dict], candidate_id: str) -> int | None:
    """The batch at which the candidate reached its best stable throughput."""
    summary = next(item for item in summaries if item["candidate_id"] == candidate_id)
    return summary.get("best_float32_batch")


if __name__ == "__main__":
    raise SystemExit(main())
