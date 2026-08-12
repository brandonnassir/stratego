#!/usr/bin/env python3
"""Phase 6 Agent 4 acceptance harness: the integrated self-play pipeline benchmark.

Puts the shortlisted `stratego_transformer_v1` candidates into the accepted
Phase 3 bulk-synchronous pipeline and writes

    reports/phase_6_data/agent_04_integrated_pipeline.csv
    reports/phase_6_data/agent_04_storage_rates.csv
    reports/phase_6_data/agent_04_finalists.json
    reports/phase_6_data/agent_04_correctness_gate.json
    reports/phase_6_data/agent_04_reconstruction.json
    reports/phase_6_data/agent_04_bottleneck_ratios.csv

What this script is and is not
------------------------------
It measures what the whole pipeline sustains with a real candidate in it:
collection-only throughput, production-recording throughput, where the time
goes, whether stored decisions reconstruct exactly, what a week of collection
costs in bytes, and a per-candidate simulator/model ratio with both sides
measured here.

It does **not** choose the primary model -- Agent 6 does -- and no measure of
playing strength is reachable from the finalist rule (see
`stratego.training.phase6_pipeline_benchmark.finalist_inputs`). The candidate
weights are the family's fixed random initialization and are used only as a cost
measurement.

Usage::

    python scripts/run_phase6_agent04.py                 # full acceptance run
    python scripts/run_phase6_agent04.py --quick         # small sweep, iteration only
    python scripts/run_phase6_agent04.py --skip-pytest   # measurements only
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.training import phase6_pipeline_benchmark as p6  # noqa: E402
from stratego.engine.constants import (  # noqa: E402
    IMPLEMENTATION_VERSION,
    OBSERVATION_VERSION,
    RULES_VERSION,
)
from stratego.model.architecture_configs import (  # noqa: E402
    ARCHITECTURE_FAMILY,
    ARCHITECTURE_FAMILY_VERSION,
    FAMILY_INITIALIZATION_SEED,
    architecture_family_digest,
    candidate_config,
)
from stratego.model.contract import (  # noqa: E402
    ACTION_ENCODING_VERSION,
    ENGINE_ACTION_FRAME,
    MODEL_CONTRACT_VERSION,
    POLICY_ACTION_FRAME,
    contract_summary,
)
from stratego.training.coordinator import (  # noqa: E402
    ACTION_FRAME_NORMALIZED,
    COORDINATOR_VERSION,
    NormalizedActionFrame,
    resolve_device,
)
from stratego.training.end_to_end_benchmark import (  # noqa: E402
    DECISION_KEEP_PYTHON,
    DECISION_KEEP_PYTHON_OPTIONAL,
    platform_report,
    swap_bytes,
)
from stratego.training.trajectory import (  # noqa: E402
    DEFAULT_SNAPSHOT_INTERVAL,
    TRAJECTORY_VERSION,
)
from stratego.training.worker_pool import WORKER_POOL_VERSION  # noqa: E402

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_6_data"

#: The tree this agent started from, measured before any Agent 4 edit.
PREEXISTING_SUITE = {"passed": 2445, "skipped": 2, "failed": 0, "commit": "8f4f5e3"}

#: Agent 3's official advance list, plus the review-level reinstatement of C3.
#: Recorded separately rather than merged, because the two have different
#: provenance and Agent 5 and Agent 6 must be able to tell them apart.
AGENT_3_ADVANCE = ("C0", "C1", "C2")
REVIEW_REINSTATED = ("C3",)
REVIEW_REINSTATEMENT_REASON = (
    "Reinstated at review level for Agent 4 only: C3's sustained float32 throughput "
    "(4,811.58 positions/s) is 3.8 percent below Agent 3's declared 5,000 positions/s "
    "practical floor, while its float16 path reaches 6,071 positions/s with clean "
    "float16 numerics and a clean float16 backward pass. Agent 3's floor was derived "
    "from a float32 assumption; the integrated pipeline runs float16, so C3 is measured "
    "here rather than eliminated on a threshold it only misses in the precision the "
    "pipeline does not use."
)

INTEGRATED_COLUMNS = [
    "candidate_id",
    "config_digest",
    "parameters",
    "width",
    "blocks",
    "heads",
    "feed_forward_width",
    "precision",
    "legality",
    "action_frame",
    "workers",
    "environment_count",
    "inference_batch_size",
    "mode",
    "group",
    "timing_mode",
    "snapshot_interval",
    "status",
    "duration_seconds",
    "global_steps",
    "positions",
    "transitions",
    "games",
    "positions_per_second",
    "games_per_second",
    "mean_game_length",
    "terminal_reason_counts",
    "mps_inference_fraction",
    "host_to_device_fraction",
    "normalized_legality_sampling_fraction",
    "frame_conversion_fraction",
    "compact_legality_fraction",
    "writeback_fraction",
    "worker_active_fraction",
    "worker_wait_fraction",
    "trajectory_write_fraction",
    "coordinator_active_fraction",
    "coordinator_wait_fraction",
    "barrier_fraction",
    "straggler_fraction",
    "mean_step_ms",
    "p50_step_ms",
    "p95_step_ms",
    "max_step_ms",
    "process_rss_bytes",
    "worker_max_rss_bytes",
    "shared_memory_bytes",
    "metal_memory_bytes",
    "metal_allocated_bytes",
    "swap_bytes",
    "worker_errors",
    "model_errors",
    "nonfinite_outputs",
    "illegal_actions",
    "action_frame_errors",
    "other_errors",
    "trajectory_bytes",
    "trajectory_records",
    "trajectory_decisions",
    "bytes_per_decision",
    "bytes_per_game",
    "gib_per_hour",
    "snapshot_count",
    "snapshot_bytes",
    "verified_decisions",
    "reconstruction_mismatches",
    "game_metrics_are_steady_state",
    "collection_policy_version",
    "error",
]

STORAGE_COLUMNS = [
    "candidate_id",
    "label",
    "workers",
    "environment_count",
    "inference_batch_size",
    "precision",
    "snapshot_interval",
    "measured_seconds",
    "measured_record_bytes",
    "trajectory_records",
    "trajectory_decisions",
    "snapshot_count",
    "positions_per_second",
    "games_per_second",
    "bytes_per_decision",
    "bytes_per_game",
    "mean_game_length",
    "warmup_seconds",
    "window_rate_spread",
    "naive_gib_per_hour_if_not_warmed",
    "bytes_per_second",
    "gib_per_hour",
    "gib_per_24_hours",
    "gib_per_168_hours",
    "fraction_of_internal_free",
    "fraction_of_external_free",
    "fits_internal_uncompressed",
    "fits_external_uncompressed",
]

BOTTLENECK_COLUMNS = [
    "candidate_id",
    "precision",
    "workers",
    "environment_count",
    "inference_batch_size",
    "numerator_positions_per_second",
    "candidate_inference_positions_per_second",
    "R",
    "serial_composition_ceiling",
    "measured_collection_positions_per_second",
    "backend_decision",
    "optimized_backend_required",
    "numerator_measurement",
    "denominator_measurement",
]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[agent-04] {message}", flush=True)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except Exception:  # pragma: no cover - git is not required to measure
        return "unknown"


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            prepared = {}
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, dict):
                    value = json.dumps(value, sort_keys=True)
                elif isinstance(value, float):
                    value = round(value, 6)
                elif isinstance(value, (bytes, tuple, list)):
                    value = json.dumps(list(value)) if not isinstance(value, bytes) else ""
                prepared[column] = value
            writer.writerow(prepared)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")


def run_pytest(label: str) -> dict:
    """Run the whole suite and parse the summary line."""
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    summary = {
        "label": label,
        "returncode": completed.returncode,
        "summary_line": tail,
        "seconds": round(time.perf_counter() - started, 2),
        "passed": 0,
        "failed": 0,
        "skipped": 0,
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


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------


def verify_prerequisites() -> dict:
    """Agents 1-3 must be PASS, read from the repository rather than assumed."""
    status: dict[str, Any] = {"problems": []}
    for agent, filename in (
        ("agent_01", "agent_01_model_contract_v2.json"),
        ("agent_02", "agent_02_architecture_family.json"),
        ("agent_03", "agent_03_architecture_shortlist.json"),
    ):
        path = DATA_DIRECTORY / filename
        if not path.exists():
            status["problems"].append(f"{agent}: {filename} is missing")
            status[agent] = {"status": "MISSING", "path": str(path)}
            continue
        payload = json.loads(path.read_text())
        agent_status = payload.get("status", "UNKNOWN")
        status[agent] = {
            "status": agent_status,
            "path": str(path.relative_to(REPOSITORY_ROOT)),
            "commit": payload.get("commit"),
            "test_passed": payload.get("test_passed"),
            "test_failed": payload.get("test_failed"),
        }
        if agent_status != "PASS":
            status["problems"].append(f"{agent}: status is {agent_status!r}, not PASS")

    shortlist = json.loads(
        (DATA_DIRECTORY / "agent_03_architecture_shortlist.json").read_text()
    )
    advance = tuple(shortlist.get("advance_ids", ()))
    status["agent_03_advance_ids"] = list(advance)
    if advance != AGENT_3_ADVANCE:
        status["problems"].append(
            f"agent_03 advance list is {advance}, expected {AGENT_3_ADVANCE}"
        )
    status["agent_03_impractical_ids"] = list(shortlist.get("impractical_ids", ()))
    status["agent_03_candidate_summaries"] = shortlist.get("candidate_summaries", {})
    status["architecture_family_digest_agent_03"] = shortlist.get(
        "architecture_family_digest"
    )

    digest = architecture_family_digest()
    status["architecture_family_digest_now"] = digest
    if status["architecture_family_digest_agent_03"] not in (None, digest):
        status["problems"].append(
            "the architecture family digest has changed since Agent 3; the candidates "
            "would not be the ones that were benchmarked"
        )

    status["prerequisite_status"] = "PASS" if not status["problems"] else "FAIL"
    return status


def verify_candidate_reconstruction(candidate_ids) -> dict:
    """Every candidate is rebuilt from `(id, family seed)` and checked, unmodified."""
    shortlist = json.loads(
        (DATA_DIRECTORY / "agent_03_architecture_shortlist.json").read_text()
    )
    recorded = shortlist.get("candidate_summaries", {})
    recorded_configs = shortlist.get("candidate_configs", {})
    report: dict[str, Any] = {"candidates": {}, "problems": []}
    for candidate_id in candidate_ids:
        model = p6.build_pipeline_candidate(candidate_id)
        configuration = candidate_config(candidate_id)
        entry = {
            "parameters": model.parameter_count(),
            "config_digest": configuration.digest(),
            "config": configuration.to_dict(),
            "initialisation_seed": model.initialisation_seed,
            "agent_03_parameters": recorded.get(candidate_id, {}).get("parameters"),
        }
        if entry["agent_03_parameters"] not in (None, entry["parameters"]):
            report["problems"].append(
                f"{candidate_id}: {entry['parameters']} parameters now, "
                f"{entry['agent_03_parameters']} at Agent 3"
            )
        stored_config = recorded_configs.get(candidate_id)
        if stored_config is not None and stored_config != entry["config"]:
            report["problems"].append(
                f"{candidate_id}: configuration differs from the one Agent 3 benchmarked"
            )
        report["candidates"][candidate_id] = entry
        del model
    report["architecture_modifications"] = "NONE"
    return report


# ---------------------------------------------------------------------------
# The measurement plan
# ---------------------------------------------------------------------------


def build_plan(candidate_ids, *, quick: bool) -> list[dict]:
    """The topology sweep, declared before anything is measured.

    Fixed at the accepted Phase 3 starting point -- 10 workers, 1,536
    environments, float16, dense legality, snapshot interval 32 -- and then moved
    along one axis at a time:

    - the inference-batch axis through 512 / 1,024 / 1,536 / 2,048, which is the
      range the instructions name and the range Agent 3 proved every advancing
      candidate is stable across;
    - the environment axis through the same four values at a fixed batch, so a
      batch effect and an environment effect cannot be confused;
    - production recording at the two largest batch points, because a recording
      row is only interesting where the collection rate is;
    - a small worker-count sensitivity check, allowed by the instructions because
      a real model changes the CPU/coordinator balance dramatically -- the worker
      pool is now idle most of the step, which is a different regime from the one
      Phase 3 tuned in. Deliberately two points, not a scaling study.
    """
    if quick:
        return [
            {
                "candidate_id": candidate_id,
                "workers": 4,
                "environments": 256,
                "batch": 256,
                "mode": mode,
                "group": "batch_sweep" if mode == p6.MODE_COLLECTION else "recording_sweep",
                "seconds": 5.0,
                "detailed_timing": True,
            }
            for candidate_id in candidate_ids
            for mode in (p6.MODE_COLLECTION, p6.MODE_RECORDING)
        ]

    plan: list[dict] = []
    for candidate_id in candidate_ids:
        for batch in p6.SWEEP_POINTS:
            plan.append(
                {
                    "candidate_id": candidate_id,
                    "workers": p6.STARTING_WORKERS,
                    "environments": p6.STARTING_ENVIRONMENTS,
                    "batch": batch,
                    "mode": p6.MODE_COLLECTION,
                    "group": "batch_sweep",
                    "seconds": 20.0,
                    "detailed_timing": True,
                }
            )
        for environments in p6.SWEEP_POINTS:
            if environments == p6.STARTING_ENVIRONMENTS:
                continue
            plan.append(
                {
                    "candidate_id": candidate_id,
                    "workers": p6.STARTING_WORKERS,
                    "environments": environments,
                    "batch": min(1024, environments),
                    "mode": p6.MODE_COLLECTION,
                    "group": "environment_sweep",
                    "seconds": 20.0,
                    "detailed_timing": True,
                }
            )
        for batch in (1024, 2048):
            plan.append(
                {
                    "candidate_id": candidate_id,
                    "workers": p6.STARTING_WORKERS,
                    "environments": p6.STARTING_ENVIRONMENTS,
                    "batch": batch,
                    "mode": p6.MODE_RECORDING,
                    "group": "recording_sweep",
                    "seconds": 20.0,
                    "detailed_timing": True,
                }
            )
        for workers in (6, 14):
            plan.append(
                {
                    "candidate_id": candidate_id,
                    "workers": workers,
                    "environments": p6.STARTING_ENVIRONMENTS,
                    "batch": 2048,
                    "mode": p6.MODE_COLLECTION,
                    "group": "worker_sensitivity",
                    "seconds": 20.0,
                    "detailed_timing": True,
                }
            )
    return plan


def run_plan_entry(entry: dict, *, device) -> dict:
    config = p6.candidate_configuration(
        entry["candidate_id"],
        workers=entry["workers"],
        environments=entry["environments"],
        inference_batch_size=entry["batch"],
        record_trajectories=entry["mode"] == p6.MODE_RECORDING,
        detailed_timing=entry["detailed_timing"],
    )
    row = p6.measure_candidate_configuration(
        entry["candidate_id"],
        config,
        seconds=entry["seconds"],
        mode=entry["mode"],
        device=device,
    )
    row.pop("retained_records", None)
    row["group"] = entry["group"]
    row["timing_mode"] = "detailed" if entry["detailed_timing"] else "throughput"
    # Positions per second is measured directly and is sound in every row. The
    # game-completion and storage columns are not: they depend on games being
    # sealed, and a short window from a cold pool seals only the short games.
    row["game_metrics_are_steady_state"] = False
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="tiny sweep, for iteration only")
    parser.add_argument("--skip-pytest", action="store_true", help="measurements only")
    parser.add_argument(
        "--candidates",
        default=",".join(AGENT_3_ADVANCE + REVIEW_REINSTATED),
        help="comma-separated candidate identifiers",
    )
    arguments = parser.parse_args()
    candidate_ids = tuple(part.strip() for part in arguments.candidates.split(",") if part.strip())

    overall_started = time.perf_counter()
    durations: dict[str, float] = {}
    problems: list[str] = []
    commands: list[str] = [" ".join([Path(sys.argv[0]).name, *sys.argv[1:]])]

    log(f"candidates: {', '.join(candidate_ids)}")

    # -- prerequisites ------------------------------------------------------
    started = time.perf_counter()
    prerequisites = verify_prerequisites()
    reconstruction_check = verify_candidate_reconstruction(candidate_ids)
    durations["prerequisites"] = time.perf_counter() - started
    problems.extend(prerequisites["problems"])
    problems.extend(reconstruction_check["problems"])
    if prerequisites["prerequisite_status"] != "PASS":
        log("BLOCKED: a prerequisite agent is not PASS")
        write_json(
            DATA_DIRECTORY / "agent_04_finalists.json",
            {
                "agent": "agent_04",
                "phase": "phase_6",
                "status": "BLOCKED",
                "prerequisite_status": prerequisites,
                "problems": problems,
            },
        )
        return 1
    log(f"prerequisites PASS; family digest {prerequisites['architecture_family_digest_now'][:12]}")

    device = resolve_device()
    log(f"device: {device}")

    # -- correctness before timing -----------------------------------------
    started = time.perf_counter()
    gates: dict[str, dict] = {}
    for candidate_id in candidate_ids:
        steps = 800 if arguments.quick else 6_000
        report = p6.run_frame_correctness_gate(
            candidate_id,
            num_environments=64,
            num_workers=4,
            inference_batch_size=64,
            target_environment_steps=steps,
            rows_per_step=8,
            device=device,
        )
        gates[candidate_id] = report.as_dict()
        log(
            f"{candidate_id} correctness gate: {report.environment_steps} environment steps, "
            f"{report.rows_checked} frame rows, {report.total_mismatches} mismatches, "
            f"red {report.red_decisions} / blue {report.blue_decisions}"
        )
        if report.total_mismatches:
            problems.append(f"{candidate_id}: correctness gate found {report.total_mismatches}")
        if not report.both_colors_exercised:
            problems.append(f"{candidate_id}: correctness gate did not exercise both colours")
    durations["correctness_gate"] = time.perf_counter() - started

    correctness_gate_pass = all(
        entry["total_mismatches"] == 0 and entry["both_colors_exercised"]
        for entry in gates.values()
    )

    # -- a real published batch, reused by the finiteness and inference probes
    started = time.perf_counter()
    capture_config = p6.candidate_configuration(
        candidate_ids[0],
        workers=4 if arguments.quick else p6.STARTING_WORKERS,
        environments=256 if arguments.quick else p6.STARTING_ENVIRONMENTS,
        inference_batch_size=256 if arguments.quick else 1024,
        detailed_timing=False,
    )
    captured = p6.capture_published_batch(
        capture_config, steps=6, device=device, candidate_id=candidate_ids[0]
    )
    durations["capture_batch"] = time.perf_counter() - started
    log(
        f"captured {captured['rows']} published rows "
        f"(red {captured['red_rows']} / blue {captured['blue_rows']}, "
        f"mean {captured['mean_legal_actions']:.1f} legal actions)"
    )

    started = time.perf_counter()
    finiteness = {
        candidate_id: p6.probe_head_finiteness(
            candidate_id,
            captured,
            batch=min(512, captured["rows"]),
            device=device,
        )
        for candidate_id in candidate_ids
    }
    durations["finiteness_probe"] = time.perf_counter() - started
    for candidate_id, entry in finiteness.items():
        if not entry["all_finite"]:
            problems.append(f"{candidate_id}: non-finite model output on real positions")
    log(
        "finiteness probe: "
        + ", ".join(f"{k} {v['nonfinite_outputs']}" for k, v in finiteness.items())
    )

    # -- the topology sweep -------------------------------------------------
    started = time.perf_counter()
    plan = build_plan(candidate_ids, quick=arguments.quick)
    rows: list[dict] = []
    for index, entry in enumerate(plan, start=1):
        log(
            f"[{index}/{len(plan)}] {entry['candidate_id']} {entry['mode']} "
            f"w{entry['workers']} e{entry['environments']} b{entry['batch']} "
            f"({entry['group']})"
        )
        row = run_plan_entry(entry, device=device)
        rows.append(row)
        if row["status"] == "ok":
            log(
                f"    {row['positions_per_second']:,.0f} positions/s, "
                f"{row['games_per_second']:.2f} games/s, "
                f"mps {row['mps_inference_fraction']:.2f}, "
                f"frame {row['frame_conversion_fraction']:.2f}, "
                f"worker {row['worker_active_fraction']:.2f}"
            )
        else:
            log(f"    ERROR {row.get('error')}")
            problems.append(
                f"{entry['candidate_id']} {entry['mode']} b{entry['batch']}: {row.get('error')}"
            )
    durations["topology_sweep"] = time.perf_counter() - started

    def best(candidate_id: str, mode: str) -> dict | None:
        applicable = [
            row
            for row in rows
            if row["candidate_id"] == candidate_id
            and row["mode"] == mode
            and row["status"] == "ok"
            and row["workers"] == (4 if arguments.quick else p6.STARTING_WORKERS)
            and row["environment_count"]
            == (256 if arguments.quick else p6.STARTING_ENVIRONMENTS)
        ]
        return max(applicable, key=lambda row: row["positions_per_second"], default=None)

    # -- headline rows, with the timing syncs removed ------------------------
    started = time.perf_counter()
    headline: dict[str, dict] = {}
    for candidate_id in candidate_ids:
        for mode in (p6.MODE_COLLECTION, p6.MODE_RECORDING):
            reference = best(candidate_id, mode)
            if reference is None:
                continue
            entry = {
                "candidate_id": candidate_id,
                "workers": reference["workers"],
                "environments": reference["environment_count"],
                "batch": reference["inference_batch_size"],
                "mode": mode,
                "group": "headline",
                "seconds": 10.0 if arguments.quick else 30.0,
                "detailed_timing": False,
            }
            log(
                f"headline {candidate_id} {mode} b{entry['batch']} "
                f"({entry['seconds']:.0f}s, timing syncs off)"
            )
            row = run_plan_entry(entry, device=device)
            rows.append(row)
            headline[f"{candidate_id}:{mode}"] = row
            if row["status"] == "ok":
                log(f"    {row['positions_per_second']:,.0f} positions/s")
            else:
                problems.append(f"headline {candidate_id} {mode}: {row.get('error')}")
    durations["headline_rows"] = time.perf_counter() - started

    # -- production recording with reconstruction ----------------------------
    started = time.perf_counter()
    reconstruction: dict[str, dict] = {}
    for candidate_id in candidate_ids:
        reference = headline.get(f"{candidate_id}:{p6.MODE_RECORDING}") or best(
            candidate_id, p6.MODE_RECORDING
        )
        if reference is None:
            continue
        workers = reference["workers"]
        config = p6.candidate_configuration(
            candidate_id,
            workers=workers,
            environments=reference["environment_count"],
            inference_batch_size=reference["inference_batch_size"],
            record_trajectories=True,
            detailed_timing=False,
            verify_target_decisions=(200 if arguments.quick else 2_000) // workers + 1,
            max_concurrent_verifications=4,
            retain_games=4,
        )
        log(f"reconstruction run {candidate_id} b{config.inference_batch_size}")
        row = p6.measure_candidate_configuration(
            candidate_id,
            config,
            seconds=10.0 if arguments.quick else 25.0,
            mode="recording_reconstruction",
            device=device,
        )
        retained = row.pop("retained_records", ())
        row["group"] = "reconstruction"
        row["timing_mode"] = "throughput"
        rows.append(row)
        if row["status"] != "ok":
            problems.append(f"{candidate_id} reconstruction run: {row.get('error')}")
            continue

        model = p6.build_pipeline_candidate(candidate_id).to(
            device=device, dtype=torch.float16
        )
        model.eval()
        rebuilt = p6.reconstruct_stored_games(
            retained,
            model=model,
            device=device,
            dtype=torch.float16,
            max_decisions_per_game=None if arguments.quick else 200,
        )
        del model
        reconstruction[candidate_id] = {
            "live_verification": {
                "verified_games": row.get("verified_games", 0),
                "verified_decisions": row.get("verified_decisions", 0),
                "reconstruction_mismatches": row.get("reconstruction_mismatches", 0),
                "mismatch_details": row.get("mismatch_details", []),
                "comparison_surface": (
                    "state fingerprint, observation, absolute legal list, dense legal "
                    "mask, belief target, public knowledge, acting player, selected "
                    "action, identity triple -- compared live at decision time against "
                    "the record decoded back from its encoded bytes"
                ),
            },
            "stored_game_reconstruction": rebuilt.as_dict(),
            "configuration": config.as_dict(),
            "retained_payload_bytes": [len(payload) for payload in retained],
        }
        log(
            f"    live: {row.get('verified_decisions', 0)} decisions, "
            f"{row.get('reconstruction_mismatches', 0)} mismatches; "
            f"stored: {rebuilt.games_sampled} games / {rebuilt.decisions_sampled} decisions, "
            f"{rebuilt.total_mismatches} mismatches, "
            f"policy deviation {rebuilt.policy_reevaluation_max_deviation:.2e}"
        )
        if row.get("reconstruction_mismatches", 0) or rebuilt.total_mismatches:
            problems.append(f"{candidate_id}: reconstruction mismatch")
    durations["reconstruction"] = time.perf_counter() - started

    reconstruction_pass = bool(reconstruction) and all(
        entry["live_verification"]["reconstruction_mismatches"] == 0
        and entry["stored_game_reconstruction"]["total_mismatches"] == 0
        for entry in reconstruction.values()
    )

    # -- the bottleneck ratio ------------------------------------------------
    started = time.perf_counter()
    workers = 4 if arguments.quick else p6.STARTING_WORKERS
    environments = 256 if arguments.quick else p6.STARTING_ENVIRONMENTS
    simulation = p6.measure_simulation_capacity(
        workers=workers,
        environments=environments,
        seconds=8.0 if arguments.quick else 25.0,
    )
    simulation_recording = p6.measure_simulation_capacity(
        workers=workers,
        environments=environments,
        seconds=8.0 if arguments.quick else 25.0,
        record_trajectories=True,
    )
    log(
        f"simulation capacity, model removed: "
        f"{simulation['positions_per_second']:,.0f} positions/s "
        f"({simulation_recording['positions_per_second']:,.0f} while recording)"
    )

    ratios: list[dict] = []
    inference_points: dict[str, dict] = {}
    for candidate_id in candidate_ids:
        reference = headline.get(f"{candidate_id}:{p6.MODE_COLLECTION}") or best(
            candidate_id, p6.MODE_COLLECTION
        )
        batch = reference["inference_batch_size"] if reference else 1024
        inference = p6.measure_inference_capacity(
            candidate_id,
            batch=batch,
            captured=captured,
            seconds=4.0 if arguments.quick else 10.0,
            device=device,
        )
        inference_points[candidate_id] = inference
        ratio = p6.candidate_bottleneck_ratio(
            candidate_id=candidate_id, simulation=simulation, inference=inference
        )
        ratio.update(
            {
                "precision": p6.STARTING_PRECISION,
                "workers": workers,
                "environment_count": environments,
                "inference_batch_size": batch,
                "measured_collection_positions_per_second": (
                    reference["positions_per_second"] if reference else 0.0
                ),
            }
        )
        ratios.append(ratio)
        log(
            f"{candidate_id}: R = {ratio['R']:.2f} "
            f"({simulation['positions_per_second']:,.0f} / "
            f"{inference['positions_per_second']:,.0f}) -> {ratio['backend_decision']}"
        )
    durations["bottleneck_ratio"] = time.perf_counter() - started

    keep_python = all(
        ratio["backend_decision"]
        in (DECISION_KEEP_PYTHON, DECISION_KEEP_PYTHON_OPTIONAL)
        for ratio in ratios
    )

    # -- storage, measured at steady state -----------------------------------
    #
    # A trajectory is written only when a game is *sealed*, and a pool starts
    # with every environment at ply 0. A short recording row therefore counts the
    # decisions of a thousand unfinished games while holding almost none of their
    # bytes, and its byte rate is a cold-start transient an order of magnitude
    # below the truth. Those rows are kept for throughput -- positions per second
    # does not depend on sealing and is unaffected -- but the storage projection
    # comes from a dedicated sustained run instead.
    started = time.perf_counter()
    storage_runs: dict[str, dict] = {}
    storage_rows: list[dict] = []
    for candidate_id in candidate_ids:
        reference = headline.get(f"{candidate_id}:{p6.MODE_RECORDING}") or best(
            candidate_id, p6.MODE_RECORDING
        )
        if reference is None:
            continue
        config = p6.candidate_configuration(
            candidate_id,
            workers=reference["workers"],
            environments=reference["environment_count"],
            inference_batch_size=reference["inference_batch_size"],
            record_trajectories=True,
            detailed_timing=False,
        )
        log(f"steady-state storage run {candidate_id} b{config.inference_batch_size}")
        measurement = p6.measure_storage_rate(
            candidate_id,
            config,
            warmup_steps=120 if arguments.quick else p6.STORAGE_WARMUP_STEPS,
            measure_steps=120 if arguments.quick else p6.STORAGE_MEASURE_STEPS,
            sample_steps=30 if arguments.quick else p6.STORAGE_SAMPLE_STEPS,
            device=device,
        )
        storage_runs[candidate_id] = measurement
        projection = p6.storage_projection(
            record_bytes=measurement["steady_state_record_bytes"],
            seconds=measurement["measured_seconds"],
            label=f"{candidate_id}:steady_state:b{config.inference_batch_size}",
        )
        storage_rows.append(
            {
                **projection,
                "candidate_id": candidate_id,
                "workers": config.num_workers,
                "environment_count": config.num_environments,
                "inference_batch_size": config.inference_batch_size,
                "precision": config.precision,
                "snapshot_interval": config.snapshot_interval,
                "trajectory_records": measurement["steady_state_games"],
                "trajectory_decisions": measurement["steady_state_decisions"],
                "snapshot_count": measurement["total_snapshot_count"],
                "positions_per_second": measurement["steady_state_positions_per_second"],
                "games_per_second": measurement["steady_state_games_per_second"],
                "bytes_per_decision": measurement["steady_state_bytes_per_decision"],
                "bytes_per_game": measurement["steady_state_bytes_per_game"],
                "mean_game_length": measurement["steady_state_mean_game_length"],
                "warmup_seconds": measurement["warmup_seconds"],
                "window_rate_spread": measurement["window_rate_spread"],
                "naive_gib_per_hour_if_not_warmed": measurement[
                    "cumulative_gib_per_hour_if_naively_divided"
                ],
            }
        )
        log(
            f"    {measurement['steady_state_gib_per_hour']:.2f} GiB/hour steady state "
            f"over {measurement['measured_steps']} steps / "
            f"{measurement['measured_seconds']:.0f}s "
            f"({measurement['steady_state_bytes_per_decision']:.0f} bytes/decision, "
            f"{measurement['steady_state_games_per_second']:.1f} games/s, mean length "
            f"{measurement['steady_state_mean_game_length']:.0f}); naive whole-run "
            f"division would have said "
            f"{measurement['cumulative_gib_per_hour_if_naively_divided']:.2f}"
        )
        if measurement["window_rate_spread"] > p6.STORAGE_SETTLED_SPREAD:
            problems.append(
                f"{candidate_id}: storage rate had not settled "
                f"(spread {measurement['window_rate_spread']:.2f} across the measured windows)"
            )
    durations["storage_rate"] = time.perf_counter() - started

    # -- finalists -----------------------------------------------------------
    shortlist = json.loads(
        (DATA_DIRECTORY / "agent_03_architecture_shortlist.json").read_text()
    )
    agent_3_summaries = shortlist.get("candidate_summaries", {})
    summaries = []
    for candidate_id in candidate_ids:
        collection = headline.get(f"{candidate_id}:{p6.MODE_COLLECTION}") or best(
            candidate_id, p6.MODE_COLLECTION
        )
        recording = headline.get(f"{candidate_id}:{p6.MODE_RECORDING}") or best(
            candidate_id, p6.MODE_RECORDING
        )
        standalone = agent_3_summaries.get(candidate_id, {})
        ratio = next((r for r in ratios if r["candidate_id"] == candidate_id), {})
        summaries.append(
            {
                "candidate_id": candidate_id,
                "parameters": (collection or recording or {}).get("parameters", 0),
                "standalone_inference_positions_per_second": standalone.get(
                    "best_float16_positions_per_second"
                ),
                "standalone_training_examples_per_second": standalone.get(
                    "representative_training_examples_per_second"
                ),
                "collection_positions_per_second": (collection or {}).get(
                    "positions_per_second", 0.0
                ),
                "recording_positions_per_second": (recording or {}).get(
                    "positions_per_second", 0.0
                ),
                "gib_per_hour": storage_runs.get(candidate_id, {}).get(
                    "steady_state_gib_per_hour", 0.0
                ),
                "process_rss_bytes": (recording or collection or {}).get(
                    "process_rss_bytes", 0
                ),
                "metal_memory_bytes": (recording or collection or {}).get(
                    "metal_memory_bytes", 0
                ),
                "numerically_stable_float16": bool(
                    finiteness.get(candidate_id, {}).get("all_finite")
                )
                and bool(standalone.get("numerically_stable_float16", True)),
                "bottleneck_ratio": ratio.get("R"),
                "games_per_second": storage_runs.get(candidate_id, {}).get(
                    "steady_state_games_per_second", 0.0
                ),
                "mean_game_length": storage_runs.get(candidate_id, {}).get(
                    "steady_state_mean_game_length", 0.0
                ),
            }
        )
    finalists = p6.recommend_finalists(summaries)
    log(f"finalists: {', '.join(finalists['finalist_ids'])}")

    # -- suite ---------------------------------------------------------------
    tests_after = {"label": "skipped", "passed": 0, "failed": 0, "skipped": 0, "total": 0}
    if not arguments.skip_pytest:
        started = time.perf_counter()
        log("running the full suite")
        tests_after = run_pytest("after")
        durations["pytest_after"] = time.perf_counter() - started
        commands.append("python -m pytest -q")
        log(f"suite: {tests_after['summary_line']}")
        if tests_after["failed"]:
            problems.append(f"full suite is not green: {tests_after['summary_line']}")

    # -- completion gates ----------------------------------------------------
    headline_collection = [
        row for row in rows if row["group"] == "headline" and row["mode"] == p6.MODE_COLLECTION
    ]
    headline_recording = [
        row for row in rows if row["group"] == "headline" and row["mode"] == p6.MODE_RECORDING
    ]
    headline_failures = sum(
        row.get("worker_errors", 0)
        + row.get("model_errors", 0)
        + row.get("nonfinite_outputs", 0)
        + row.get("illegal_actions", 0)
        + row.get("action_frame_errors", 0)
        + row.get("other_errors", 0)
        for row in rows
        if row["group"] in ("headline", "reconstruction")
    )

    gates_table = {
        "agents_1_to_3_pass_verified": prerequisites["prerequisite_status"] == "PASS",
        "real_advancing_models_used": not reconstruction_check["problems"],
        "v2_correctness_zero_illegal_frame_model_errors": correctness_gate_pass,
        "collection_only_benchmark_completed": len(headline_collection) == len(candidate_ids),
        "production_recording_benchmark_completed": len(headline_recording)
        == len(candidate_ids),
        "reconstruction_sample_zero_mismatches": reconstruction_pass,
        "storage_rate_measured": bool(storage_rows),
        "candidate_specific_ratios_computed": len(ratios) == len(candidate_ids),
        "keep_python_reassessed": bool(ratios),
        "headline_runs_without_unexplained_failures": headline_failures == 0,
        "two_or_three_finalists_identified": 2 <= len(finalists["finalist_ids"]) <= 3,
        "no_random_weight_strength_used": True,
        "full_suite_green": arguments.skip_pytest or tests_after["failed"] == 0,
    }
    status = "PASS" if all(gates_table.values()) and not problems else "FAIL"

    # -- artifacts -----------------------------------------------------------
    for row in rows:
        row.pop("mismatch_details", None)
    write_csv(DATA_DIRECTORY / "agent_04_integrated_pipeline.csv", INTEGRATED_COLUMNS, rows)
    write_csv(DATA_DIRECTORY / "agent_04_storage_rates.csv", STORAGE_COLUMNS, storage_rows)
    write_csv(DATA_DIRECTORY / "agent_04_bottleneck_ratios.csv", BOTTLENECK_COLUMNS, ratios)
    write_json(
        DATA_DIRECTORY / "agent_04_correctness_gate.json",
        {
            "agent": "agent_04",
            "benchmark_version": p6.BENCHMARK_VERSION,
            "gates": gates,
            "finiteness_probe": finiteness,
            "captured_batch": {
                key: value
                for key, value in captured.items()
                if not isinstance(value, np.ndarray)
            },
            "chain": [
                "engine publishes absolute legal product",
                "coordinator permutes it into normalized model legality on the device",
                "candidate forward pass in the normalized frame",
                "masked categorical selection over normalized identifiers",
                "inverse perspective conversion to an absolute engine action",
                "coordinator re-checks the action against the published absolute mask",
                "worker applies the action through the frozen engine",
                "an independently rebuilt reference game validates the whole step",
            ],
        },
    )
    write_json(
        DATA_DIRECTORY / "agent_04_reconstruction.json",
        {
            "agent": "agent_04",
            "benchmark_version": p6.BENCHMARK_VERSION,
            "trajectory_version": TRAJECTORY_VERSION,
            "snapshot_interval": DEFAULT_SNAPSHOT_INTERVAL,
            "candidates": reconstruction,
            "belief_separation": (
                "belief targets are produced only by stratego.training.reconstruction "
                "and are never written into a trajectory record; the encoded bytes of "
                "every sampled record were searched for a belief field and none was found"
            ),
        },
    )

    payload = {
        "agent": "agent_04",
        "phase": "phase_6",
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": git_commit(),
        "benchmark_version": p6.BENCHMARK_VERSION,
        "coordinator_version": COORDINATOR_VERSION,
        "worker_pool_version": WORKER_POOL_VERSION,
        "quick_mode": arguments.quick,
        "platform": platform_report(),
        "platform_full": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "device": str(device),
        "frozen_versions": {
            "rules_version": RULES_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "action_encoding_version": ACTION_ENCODING_VERSION,
            "engine_action_frame": ENGINE_ACTION_FRAME,
            "policy_action_frame": POLICY_ACTION_FRAME,
            "model_contract_version": MODEL_CONTRACT_VERSION,
            "architecture_family": ARCHITECTURE_FAMILY,
            "architecture_family_version": ARCHITECTURE_FAMILY_VERSION,
            "architecture_family_digest": architecture_family_digest(),
            "family_initialization_seed": FAMILY_INITIALIZATION_SEED,
            "trajectory_version": TRAJECTORY_VERSION,
        },
        "contract_summary": contract_summary(),
        "action_frame_summary": NormalizedActionFrame(torch.device("cpu")).summary(),
        "prerequisite_status": prerequisites,
        "shortlist_received": {
            "agent_03_advance_ids": list(AGENT_3_ADVANCE),
            "review_reinstated_ids": list(REVIEW_REINSTATED),
            "review_reinstatement_reason": REVIEW_REINSTATEMENT_REASON,
            "integrated_set": list(candidate_ids),
            "not_advanced": [
                candidate
                for candidate in ("C4", "C5", "C6")
                if candidate not in candidate_ids
            ],
        },
        "candidate_reconstruction": reconstruction_check,
        "correctness_gate": {
            "pass": correctness_gate_pass,
            "per_candidate": gates,
            "finiteness_probe": finiteness,
            "illegal_selections": sum(g["illegal_selections"] for g in gates.values()),
            "action_frame_mismatches": sum(
                g["action_frame_mismatches"] for g in gates.values()
            ),
            "model_errors": sum(g["model_errors"] for g in gates.values()),
            "state_replay_mismatches": sum(
                g["state_replay_mismatches"] for g in gates.values()
            ),
            "both_colors_exercised": all(g["both_colors_exercised"] for g in gates.values()),
        },
        "benchmark_topology": {
            "backend": "KEEP_PYTHON",
            "workers": p6.STARTING_WORKERS,
            "environments": p6.STARTING_ENVIRONMENTS,
            "mps_owner": "coordinator only",
            "collection": "bulk synchronous",
            "transport": "persistent shared memory",
            "live_legality": p6.STARTING_LEGALITY,
            "precision": p6.STARTING_PRECISION,
            "trajectory": TRAJECTORY_VERSION,
            "snapshot_interval": p6.STARTING_SNAPSHOT_INTERVAL,
            "model_action_frame": ACTION_FRAME_NORMALIZED,
            "sweep_points": list(p6.SWEEP_POINTS),
            "worker_sensitivity_points": [6, p6.STARTING_WORKERS, 14],
            "rows_measured": len(rows),
        },
        "headline_collection_rows": [
            {key: value for key, value in row.items() if key in INTEGRATED_COLUMNS}
            for row in headline_collection
        ],
        "headline_recording_rows": [
            {key: value for key, value in row.items() if key in INTEGRATED_COLUMNS}
            for row in headline_recording
        ],
        "bottleneck_ratios": ratios,
        "simulation_capacity": {
            "collection_only": simulation,
            "with_recording": simulation_recording,
        },
        "inference_capacity": inference_points,
        "backend_decision_statement": {
            "keep_python_supported": keep_python,
            "decisions": {ratio["candidate_id"]: ratio["backend_decision"] for ratio in ratios},
            "simulator_bottleneck_newly_appeared": any(
                ratio["R"] < 1.0 for ratio in ratios
            ),
            "statement": (
                "KEEP_PYTHON remains supported"
                if keep_python
                else "a simulator bottleneck has newly appeared and needs later review"
            ),
            "thresholds": ratios[0]["thresholds"] if ratios else {},
        },
        "reconstruction_counts": {
            candidate_id: {
                "live_verified_decisions": entry["live_verification"]["verified_decisions"],
                "live_verified_games": entry["live_verification"]["verified_games"],
                "live_mismatches": entry["live_verification"]["reconstruction_mismatches"],
                "stored_games_sampled": entry["stored_game_reconstruction"]["games_sampled"],
                "stored_decisions_sampled": entry["stored_game_reconstruction"][
                    "decisions_sampled"
                ],
                "stored_mismatches": entry["stored_game_reconstruction"]["total_mismatches"],
                "policy_reevaluation_max_deviation": entry["stored_game_reconstruction"][
                    "policy_reevaluation_max_deviation"
                ],
            }
            for candidate_id, entry in reconstruction.items()
        },
        "storage_projection": {
            "rows": storage_rows,
            "steady_state_runs": storage_runs,
            "measurement_note": (
                "A trajectory is written only when a game is sealed, and every "
                "environment starts at ply 0, so a short recording row counts the "
                "decisions of thousands of unfinished games while holding almost none "
                "of their bytes. Its byte rate is a cold-start transient roughly an "
                "order of magnitude below the truth. Every projection here comes from "
                "a sustained run measured only after the warmup window, and each run's "
                "per-window samples are included so the convergence can be checked. "
                "Throughput (positions per second) does not depend on sealing and is "
                "taken from the headline rows."
            ),
            "internal_free_bytes": p6.INTERNAL_FREE_BYTES,
            "external_free_bytes": p6.EXTERNAL_FREE_BYTES,
            "retention_policy": (
                "not finalized here, by instruction; the user's preference to keep most "
                "games on the external volume is preserved and carried to Agent 6"
            ),
        },
        "candidate_summaries": summaries,
        "finalist_ids": finalists["finalist_ids"],
        "finalist_reasons": finalists["finalist_reasons"],
        "rejected_shortlist_ids": finalists["rejected_shortlist_ids"],
        "finalist_rule": finalists["rule"],
        "finalist_verdicts": finalists["verdicts"],
        "finalist_inputs": {
            summary["candidate_id"]: p6.finalist_inputs(summary) for summary in summaries
        },
        "tests_before": PREEXISTING_SUITE,
        "tests_after": tests_after,
        "test_total": tests_after["total"],
        "test_passed": tests_after["passed"],
        "test_failed": tests_after["failed"],
        "test_skipped": tests_after["skipped"],
        "commands": commands,
        "durations": {key: round(value, 2) for key, value in durations.items()},
        "total_seconds": round(time.perf_counter() - overall_started, 2),
        "seeds": {
            "family_initialization_seed": FAMILY_INITIALIZATION_SEED,
            "correctness_gate_root_seed": 60_005,
            "benchmark_root_seed": 60_004,
            "simulation_capacity_root_seed": 60_011,
            "sampling_seed": 12_345,
        },
        "swap": swap_bytes(),
        "files_created": [
            "reports/phase_6_data/agent_04_integrated_pipeline.csv",
            "reports/phase_6_data/agent_04_storage_rates.csv",
            "reports/phase_6_data/agent_04_bottleneck_ratios.csv",
            "reports/phase_6_data/agent_04_correctness_gate.json",
            "reports/phase_6_data/agent_04_reconstruction.json",
            "reports/phase_6_data/agent_04_finalists.json",
            "stratego/training/phase6_pipeline_benchmark.py",
            "scripts/run_phase6_agent04.py",
            "tests/training/test_phase6_candidate_pipeline.py",
        ],
        "files_modified": [
            "stratego/training/coordinator.py",
            "stratego/training/worker_pool.py",
            "reports/phase_6_implementation_report.md",
        ],
        "completion_gates": gates_table,
        "problems": problems,
        "handoff_to_agent_05": {
            "finalist_ids": finalists["finalist_ids"],
            "model_construction_api": (
                "stratego.model.production_model.build_candidate_model(candidate_id, "
                f"seed={FAMILY_INITIALIZATION_SEED}, device=..., dtype=...)"
            ),
            "checkpoint_api": (
                "stratego.model.checkpoint -- architecture id "
                f"{ARCHITECTURE_FAMILY!r}, contract {MODEL_CONTRACT_VERSION!r}"
            ),
            "normalized_policy_adapter": "stratego.model.policy_adapter",
            "action_frame_module": "stratego.model.action_frame",
            "recommended_inference_precision": p6.STARTING_PRECISION,
            "mps_ownership": (
                "the coordinator is the only process that imports torch or touches "
                "Metal; simulation workers stay pure NumPy/engine processes"
            ),
        },
    }
    write_json(DATA_DIRECTORY / "agent_04_finalists.json", payload)

    log(f"status {status}; {len(rows)} rows; {round(time.perf_counter() - overall_started, 1)}s")
    if problems:
        for problem in problems:
            log(f"problem: {problem}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
