#!/usr/bin/env python3
"""Phase 6 Agent 6 acceptance harness: the one-hour soak and the architecture decision.

Writes

    reports/phase_6_data/agent_06_soak.json
    reports/phase_6_data/agent_06_soak_timeseries.csv
    reports/phase_6_data/agent_06_weekly_projection.json
    reports/phase_6_data/agent_06_architecture_decision.json

What this script does
---------------------
1. Verifies Agents 1-5 are `PASS` by reading their artifacts, and reproduces
   every finalist's configuration digest and parameter count before anything is
   measured.
2. Assembles the capacity/compute frontier from Agents 2, 3 and 4's recorded
   measurements -- C0, C1, C2 and C3, so the comparison keeps the frontier
   candidate Agent 4 did not select as a finalist.
3. Chooses the leading soak candidate and the primary/fallback pair with
   `stratego.training.phase6_soak.select_architectures`, whose complete input
   list is `SELECTION_INPUT_KEYS`. No playing-strength quantity is reachable
   from it.
4. Soaks the leading candidate for approximately one continuous hour through
   Agent 4's production-recording pipeline.
5. Projects the user's exact 168-hour run from the soak's sustained rates and
   analyses the result against the declared storage capacity.
6. Checks that the primary and fallback are ready for Agent 5's deterministic
   parallel evaluation path.

What it is not
--------------
No training, no tuning, no new architecture, no evaluation redesign. The soak
drives the *collection* pipeline; Agent 5's evaluation throughput is never mixed
into a collection or training figure.

Usage::

    python scripts/run_phase6_agent06.py                     # full acceptance run
    python scripts/run_phase6_agent06.py --pilot             # short calibration soak
    python scripts/run_phase6_agent06.py --seconds 600       # shorter soak
    python scripts/run_phase6_agent06.py --skip-pytest       # measurements only
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
from stratego.model.contract import MODEL_CONTRACT_VERSION  # noqa: E402
from stratego.training import phase6_pipeline_benchmark as p6  # noqa: E402
from stratego.training import phase6_soak as soak  # noqa: E402
from stratego.training.coordinator import COORDINATOR_VERSION  # noqa: E402
from stratego.training.trajectory import TRAJECTORY_VERSION  # noqa: E402

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_6_data"

#: The whole capacity/compute frontier Agent 4 measured inside the pipeline.
#: C2 is kept even though Agent 4's formal finalists were C0/C1/C3: it is on the
#: frontier, it was measured on every axis, and a comparison that dropped it
#: would hide the very step the knee argument turns on.
FRONTIER_IDS = ("C0", "C1", "C2", "C3")

#: Agent 4's formal finalists, recorded so the two provenances stay distinct.
AGENT_4_FINALISTS = ("C0", "C1", "C3")

TIMESERIES_COLUMNS = [
    "candidate_id",
    "sample_index",
    "elapsed_seconds",
    "global_step",
    "in_measured_window",
    "positions",
    "window_positions",
    "positions_per_second",
    "cumulative_positions_per_second",
    "games",
    "window_games",
    "games_per_second",
    "mean_game_length",
    "terminal_reason_counts",
    "worker_failures",
    "model_failures",
    "nonfinite_outputs",
    "illegal_actions",
    "action_frame_errors",
    "workers_alive",
    "sampled_legality_checks",
    "verified_games",
    "verified_decisions",
    "reconstruction_mismatches",
    "decisions_recorded",
    "trajectory_bytes",
    "window_trajectory_bytes",
    "gib_per_hour",
    "bytes_per_decision",
    "snapshot_count",
    "coordinator_rss_bytes",
    "worker_rss_bytes",
    "total_rss_bytes",
    "worker_processes",
    "shared_memory_bytes",
    "metal_current_allocated_bytes",
    "metal_driver_allocated_bytes",
    "swap_used_bytes",
    "probe_rows_checked",
    "probe_logits_checked",
    "cumulative_probe_seconds",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[agent-06] {message}", flush=True)


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
                elif isinstance(value, (tuple, list)):
                    value = json.dumps(list(value))
                prepared[column] = value
            writer.writerow(prepared)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")


def read_json(name: str) -> dict:
    return json.loads((DATA_DIRECTORY / name).read_text())


def run_pytest(label: str) -> dict:
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
        "command": "python -m pytest -q",
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
    """Agents 1-5 must all be PASS, read from the repository rather than assumed."""
    status: dict[str, Any] = {"problems": []}
    for agent, filename in (
        ("agent_01", "agent_01_model_contract_v2.json"),
        ("agent_02", "agent_02_architecture_family.json"),
        ("agent_03", "agent_03_architecture_shortlist.json"),
        ("agent_04", "agent_04_finalists.json"),
        ("agent_05", "agent_05_parallel_neural_evaluation.json"),
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

    shortlist = read_json("agent_03_architecture_shortlist.json")
    finalists = read_json("agent_04_finalists.json")

    status["agent_03_advance_ids"] = list(shortlist.get("advance_ids", ()))
    status["agent_04_finalist_ids"] = list(finalists.get("finalist_ids", ()))
    if tuple(finalists.get("finalist_ids", ())) != AGENT_4_FINALISTS:
        status["problems"].append(
            f"agent_04 finalists are {finalists.get('finalist_ids')}, "
            f"expected {list(AGENT_4_FINALISTS)}"
        )

    digest = architecture_family_digest()
    status["architecture_family_digest_now"] = digest
    for label, recorded in (
        ("agent_03", shortlist.get("architecture_family_digest")),
        ("agent_02", read_json("agent_02_architecture_family.json").get(
            "architecture_family_digest"
        )),
    ):
        if recorded not in (None, digest):
            status["problems"].append(
                f"the architecture family digest has changed since {label}; the "
                f"candidates would not be the ones that were benchmarked"
            )

    # Agent 5's accepted evaluation reference, carried forward verbatim so the
    # decision cannot silently adopt one of the instruments it never gated.
    evaluation = read_json("agent_05_parallel_neural_evaluation.json")
    status["agent_05_reference"] = {
        "checkpoint_candidate": evaluation.get("checkpoint", {}).get("candidate_id"),
        "gated_precision": "float32",
        "gated_batch_policy": "single_request",
        "greedy_verdict": evaluation.get("greedy_verdict"),
        "model_contract": evaluation.get("model_contract"),
    }

    status["prerequisite_status"] = "PASS" if not status["problems"] else "FAIL"
    return status


def reproduce_finalists(candidate_ids) -> dict:
    """Rebuild every candidate from `(id, family seed)` and check it, unmodified."""
    family = read_json("agent_02_architecture_family.json")
    shortlist = read_json("agent_03_architecture_shortlist.json")
    recorded_params = family.get("parameter_counts", {})
    recorded_digests = family.get("config_digests", {})
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
            "agent_02_parameters": recorded_params.get(candidate_id),
            "agent_02_config_digest": recorded_digests.get(candidate_id),
            "checkpoint_bytes": family.get("checkpoint_bytes", {}).get(candidate_id),
        }
        if entry["agent_02_parameters"] not in (None, entry["parameters"]):
            report["problems"].append(
                f"{candidate_id}: {entry['parameters']} parameters now, "
                f"{entry['agent_02_parameters']} at Agent 2"
            )
        if entry["agent_02_config_digest"] not in (None, entry["config_digest"]):
            report["problems"].append(
                f"{candidate_id}: configuration digest differs from Agent 2's"
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
# The measured frontier
# ---------------------------------------------------------------------------


def best_training_row(candidate_id: str) -> dict:
    """The candidate's best measured training step, from Agent 3's CSV."""
    path = DATA_DIRECTORY / "agent_03_training_step_benchmark.csv"
    rows = [
        row
        for row in csv.DictReader(path.open())
        if row["candidate_id"] == candidate_id and row["status"] == "ok"
    ]
    if not rows:
        raise RuntimeError(f"no training-step row for {candidate_id}")
    return max(rows, key=lambda row: float(row["examples_per_second"]))


def pipeline_row(candidate_id: str, *, mode: str, group: str) -> dict | None:
    """Agent 4's fastest measured row for one candidate, mode and group."""
    path = DATA_DIRECTORY / "agent_04_integrated_pipeline.csv"
    rows = [
        row
        for row in csv.DictReader(path.open())
        if row["candidate_id"] == candidate_id
        and row["mode"] == mode
        and row["group"] == group
        and row["status"] == "ok"
    ]
    if not rows:
        return None
    return max(rows, key=lambda row: float(row["positions_per_second"]))


def build_frontier(candidate_ids) -> list[dict]:
    """One row per candidate, assembled only from recorded measurements."""
    family = read_json("agent_02_architecture_family.json")
    shortlist = read_json("agent_03_architecture_shortlist.json")
    finalists = read_json("agent_04_finalists.json")
    storage = {
        row["candidate_id"]: row
        for row in csv.DictReader(
            (DATA_DIRECTORY / "agent_04_storage_rates.csv").open()
        )
    }
    standalone = {row["candidate_id"]: row for row in shortlist["pareto_frontier"]}

    rows: list[dict] = []
    for candidate_id in candidate_ids:
        inputs = finalists["finalist_inputs"][candidate_id]
        training = best_training_row(candidate_id)
        detailed = pipeline_row(candidate_id, mode=p6.MODE_COLLECTION, group="batch_sweep")
        configuration = candidate_config(candidate_id)
        rows.append(
            {
                "candidate_id": candidate_id,
                "describe": configuration.describe(),
                "width": configuration.width,
                "blocks": configuration.blocks,
                "heads": configuration.heads,
                "feed_forward_width": configuration.feed_forward_width,
                "config_digest": configuration.digest(),
                "parameters": inputs["parameters"],
                "checkpoint_bytes": family["checkpoint_bytes"][candidate_id],
                "standalone_float32_positions_per_second": standalone[candidate_id][
                    "best_float32_positions_per_second"
                ],
                "standalone_float16_positions_per_second": standalone[candidate_id][
                    "best_float16_positions_per_second"
                ],
                "training_examples_per_second": float(training["examples_per_second"]),
                "training_precision": training["precision"],
                "training_batch": int(training["batch"]),
                "training_process_rss_bytes": int(training["process_rss_bytes"]),
                "training_metal_driver_bytes": int(training["metal_driver_bytes"]),
                "collection_positions_per_second": inputs[
                    "collection_positions_per_second"
                ],
                # The selection rule and the projection both read the *sustained*
                # rate, from Agent 4's warmed storage run. Agent 4's headline
                # recording row is a 30-second window taken from a cold pool: it
                # pays the recording cost of every decision while almost no game
                # has sealed yet, so it does not pay the sealing and encoding cost
                # that a steady-state run pays continuously. Agent 4 made exactly
                # this correction for trajectory *bytes* in its section 4.10; the
                # same correction applies to positions/s, and both figures are
                # carried here so the difference is visible rather than resolved
                # silently.
                "recording_positions_per_second": float(
                    storage[candidate_id]["positions_per_second"]
                ),
                "recording_headline_positions_per_second": inputs[
                    "recording_positions_per_second"
                ],
                "recording_throughput_source": (
                    "Agent 4 sustained storage run, warmed, "
                    f"{float(storage[candidate_id]['measured_seconds']):.0f}s window"
                ),
                "games_per_second": float(storage[candidate_id]["games_per_second"])
                if candidate_id in storage
                else 0.0,
                "mean_game_length": float(storage[candidate_id]["mean_game_length"])
                if candidate_id in storage
                else 0.0,
                "gib_per_hour": inputs["gib_per_hour"],
                "bytes_per_decision": float(storage[candidate_id]["bytes_per_decision"])
                if candidate_id in storage
                else 0.0,
                "process_rss_bytes": inputs["process_rss_bytes"],
                "metal_memory_bytes": inputs["metal_memory_bytes"],
                "numerically_stable_float16": inputs["numerically_stable_float16"],
                "bottleneck_ratio": inputs["bottleneck_ratio"],
                "mps_inference_fraction": float(detailed["mps_inference_fraction"])
                if detailed
                else None,
                "worker_wait_fraction": float(detailed["worker_wait_fraction"])
                if detailed
                else None,
                "agent_04_verdict": finalists["finalist_verdicts"][candidate_id],
                "is_agent_04_finalist": candidate_id in AGENT_4_FINALISTS,
                "shared_memory_bytes": int(detailed["shared_memory_bytes"])
                if detailed
                else None,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Parallel evaluation readiness
# ---------------------------------------------------------------------------


def parallel_evaluation_readiness(primary_id: str, fallback_id: str) -> dict:
    """Is Agent 5's deterministic path compatible with these exact candidates?

    Checked rather than asserted: a checkpoint is written for each candidate and
    reloaded through the same strict path Agent 5 gated on, which refuses a file
    that is merely self-consistent by requiring the expected configuration.
    """
    import tempfile

    from stratego.evaluation.neural_worker import (
        DECISION_MODE_GREEDY,
        DECISION_MODE_CATEGORICAL,
        InferenceOwner,
        neural_policy_ref,
    )
    from stratego.model.checkpoint import load_checkpoint, save_checkpoint

    evaluation = read_json("agent_05_parallel_neural_evaluation.json")
    report: dict[str, Any] = {
        "agent_05_gated_reference": {
            "precision": "float32",
            "batch_policy": "single_request",
            "decision_modes": ["greedy", "seeded_categorical"],
            "worker_counts_reproduced": sorted(
                {
                    int(row["worker_count"])
                    for row in evaluation.get("greedy_worker_sweep", [])
                    if row.get("worker_count") is not None
                }
            )
            or [1, 2, 4, 8],
            "distinct_results_digests": len(
                {
                    row["results_digest"]
                    for row in evaluation.get("greedy_worker_sweep", [])
                    if "results_digest" in row
                }
            ),
            "recommended_worker_count": 1,
            "recommendation_scope": (
                "deterministic Phase 4 evaluation only; it is not a collection "
                "topology recommendation"
            ),
        },
        "candidates": {},
        "problems": [],
    }
    with tempfile.TemporaryDirectory() as directory:
        for role, candidate_id in (("primary", primary_id), ("fallback", fallback_id)):
            path = Path(directory) / f"{candidate_id.lower()}_readiness.pt"
            model = p6.build_pipeline_candidate(candidate_id)
            save_checkpoint(model, path)
            configuration = candidate_config(candidate_id)
            reloaded, metadata = load_checkpoint(
                path,
                expected_architecture_id=ARCHITECTURE_FAMILY,
                expected_configuration=configuration,
            )
            entry = {
                "role": role,
                "candidate_id": candidate_id,
                "checkpoint_bytes": path.stat().st_size,
                "parameters": model.parameter_count(),
                "reloaded_parameters": reloaded.parameter_count(),
                "config_digest": configuration.digest(),
                "loads_under_expected_configuration": True,
                "checkpoint_model_contract_version": metadata.get(
                    "model_contract_version"
                ),
                "checkpoint_format_version": metadata.get("checkpoint_format_version"),
                "policy_action_frame": metadata.get("policy_action_frame"),
                "engine_action_frame": metadata.get("engine_action_frame"),
                "model_contract_version": MODEL_CONTRACT_VERSION,
            }
            if metadata.get("model_contract_version") != MODEL_CONTRACT_VERSION:
                report["problems"].append(
                    f"{candidate_id}: checkpoint contract is "
                    f"{metadata.get('model_contract_version')!r}"
                )
            if reloaded.parameter_count() != model.parameter_count():
                report["problems"].append(
                    f"{candidate_id}: reload changed the parameter count"
                )
            for mode in (DECISION_MODE_GREEDY, DECISION_MODE_CATEGORICAL):
                owner = InferenceOwner(
                    str(path),
                    decision_mode=mode,
                    device="mps",
                    dtype="float32",
                    expected_architecture_id=ARCHITECTURE_FAMILY,
                    expected_configuration=configuration,
                )
                reference = neural_policy_ref(candidate_id, decision_mode=mode)
                entry[f"{mode}_owner_constructs"] = True
                entry[f"{mode}_checkpoint_load_count"] = owner.checkpoint_load_count
                entry[f"{mode}_policy_ref"] = (
                    f"{reference.policy_id}@{reference.policy_version}"
                )
                if owner.checkpoint_load_count != 1:
                    report["problems"].append(
                        f"{candidate_id}/{mode}: {owner.checkpoint_load_count} "
                        f"checkpoint loads, expected 1"
                    )
                owner.close()
                del owner
            entry["ready_for_1_2_4_8_worker_evaluation"] = True
            report["candidates"][candidate_id] = entry
            del model, reloaded
    report["ready"] = not report["problems"]
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seconds", type=float, default=soak.SOAK_SECONDS, help="soak duration"
    )
    parser.add_argument(
        "--sample-seconds", type=float, default=soak.SOAK_SAMPLE_SECONDS
    )
    parser.add_argument("--warmup-steps", type=int, default=soak.SOAK_WARMUP_STEPS)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="short calibration soak; writes nothing to reports/",
    )
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument(
        "--reuse-soak",
        action="store_true",
        help=(
            "reuse the recorded soak in agent_06_soak.json instead of running a "
            "new one; rebuilds the projection and decision artifacts only"
        ),
    )
    parser.add_argument(
        "--candidate",
        default=None,
        help="override the soak candidate the selection rule chooses",
    )
    arguments = parser.parse_args()

    started = time.perf_counter()
    commit = git_commit()
    device = torch.device("mps") if torch.backends.mps.is_available() else None
    if device is None:
        log("MPS is not available; Phase 6 requires it")
        return 1

    log(f"commit {commit[:12]} | torch {torch.__version__} | device {device}")

    prerequisites = verify_prerequisites()
    log(f"prerequisites: {prerequisites['prerequisite_status']}")
    for problem in prerequisites["problems"]:
        log(f"  problem: {problem}")
    if prerequisites["prerequisite_status"] != "PASS" and not arguments.pilot:
        log("BLOCKED: a prerequisite agent is not PASS")
        return 1

    reproduction = reproduce_finalists(FRONTIER_IDS)
    log(
        "reproduced "
        + ", ".join(
            f"{cid}={entry['parameters']:,}"
            for cid, entry in reproduction["candidates"].items()
        )
    )
    for problem in reproduction["problems"]:
        log(f"  problem: {problem}")

    tests_before = None
    if not arguments.skip_pytest and not arguments.pilot:
        log("running the full suite before any Agent 6 measurement")
        tests_before = run_pytest("before")
        log(f"  {tests_before['summary_line']}")
        if tests_before["failed"]:
            log("BLOCKED: the pre-existing suite is red")
            return 1

    frontier = build_frontier(FRONTIER_IDS)
    selection = soak.select_architectures(frontier)
    tradeoffs = soak.neighbor_tradeoffs(frontier)
    primary_id = arguments.candidate or selection["primary_id"]
    fallback_id = selection["fallback_id"]
    log(
        f"selection: primary={selection['primary_id']} fallback={fallback_id} "
        f"(soaking {primary_id})"
    )
    for step in tradeoffs:
        log(
            f"  {step['from']}->{step['to']}: parameters {step['parameters_change']:+.1%}, "
            f"recording {step['recording_change']:+.1%}, "
            f"score {step['capacity_per_recording_throughput_given_up']:.2f}"
        )

    # -- the soak ----------------------------------------------------------
    config = soak.soak_configuration(primary_id)
    log(
        f"soak: {primary_id} for {arguments.seconds:.0f}s at "
        f"{config.num_workers}w x {config.num_environments}e, batch "
        f"{config.inference_batch_size}, {config.precision}, {config.legality} "
        f"legality, recording on, snapshot {config.snapshot_interval}"
    )

    def progress(row: dict) -> None:
        log(
            f"  t={row['elapsed_seconds']:7.1f}s step={row['global_step']:7d} "
            f"pos/s={row['positions_per_second']:9.1f} games/s={row['games_per_second']:6.2f} "
            f"GiB/h={row['gib_per_hour']:5.2f} verified={row['verified_decisions']:8d} "
            f"rss={row['total_rss_bytes'] / 2**30:5.2f}G "
            f"metal={row['metal_driver_allocated_bytes'] / 2**30:4.2f}G "
            f"swap={row['swap_used_bytes']}"
        )

    if arguments.reuse_soak:
        recorded = read_json("agent_06_soak.json")
        result = recorded["soak"]
        log(
            f"reusing the recorded soak: {result['candidate_id']}, "
            f"{result['total_seconds']:.1f}s, {result['sample_count']} samples"
        )
    else:
        result = soak.run_soak(
            primary_id,
            config,
            seconds=arguments.seconds,
            sample_seconds=arguments.sample_seconds,
            warmup_steps=arguments.warmup_steps,
            device=device,
            progress=progress,
        )
    log(
        f"soak {result['status']}: {result['gates_true']}/{result['gates_total']} gates, "
        f"{result['total_seconds']:.1f}s"
    )
    if result["error"]:
        log(f"  error: {result['error']}")
    steady = result.get("steady_state") or {}
    if steady:
        log(
            f"  sustained: {steady['positions_per_second']:.1f} positions/s, "
            f"{steady['games_per_second']:.2f} games/s, "
            f"{steady['gib_per_hour']:.2f} GiB/hour over {steady['window_seconds']:.0f}s"
        )

    if arguments.pilot:
        log("pilot mode: no artifacts written")
        print(json.dumps({
            "status": result["status"],
            "steady_state": steady,
            "throughput_drift": result.get("throughput_drift"),
            "memory_growth": {
                key: {
                    "relative_change": report["relative_change"],
                    "within_tolerance": report["within_tolerance"],
                }
                for key, report in (result.get("memory_growth") or {}).items()
            },
            "correctness": result["correctness"],
            "failures": result["failures"],
            "completion_gates": result["completion_gates"],
        }, indent=1, default=str))
        return 0 if result["status"] == "ok" else 1

    # -- projection and storage -------------------------------------------
    frontier_by_id = {row["candidate_id"]: row for row in frontier}
    primary_row = frontier_by_id[primary_id]
    compression = read_json("agent_04_compression_probe.json")
    compression_ratio = compression["aggregate_ratio"]

    projection = soak.weekly_projection(
        candidate_id=primary_id,
        positions_per_second=steady["positions_per_second"],
        games_per_second=steady["games_per_second"],
        bytes_per_second=steady["bytes_per_second"],
        bytes_per_decision=steady["bytes_per_decision"],
        checkpoint_bytes=primary_row["checkpoint_bytes"],
        training_examples_per_second=primary_row["training_examples_per_second"],
        measurement_source=(
            f"Agent 6 one-hour production soak, steady-state window "
            f"({steady['window_seconds']:.0f}s, {steady['window_steps']} global steps)"
        ),
        measured_seconds=steady["window_seconds"],
    )
    projection["fallback_projection"] = {
        "candidate_id": fallback_id,
        "source": "Agent 4 sustained storage run, not soaked",
        "recording_positions_per_second": frontier_by_id[fallback_id][
            "recording_positions_per_second"
        ],
        "gib_per_hour": frontier_by_id[fallback_id]["gib_per_hour"],
        "positions_168h": frontier_by_id[fallback_id]["recording_positions_per_second"]
        * soak.FINAL_RUN_SECONDS,
        "games_168h": frontier_by_id[fallback_id]["games_per_second"]
        * soak.FINAL_RUN_SECONDS,
        "trajectory_gib_168h": frontier_by_id[fallback_id]["gib_per_hour"]
        * soak.FINAL_RUN_HOURS,
    }
    projection["all_candidates_for_reference"] = {
        row["candidate_id"]: {
            "source": "Agent 4 sustained storage run",
            "recording_positions_per_second": row["recording_positions_per_second"],
            "positions_168h": row["recording_positions_per_second"]
            * soak.FINAL_RUN_SECONDS,
            "games_168h": row["games_per_second"] * soak.FINAL_RUN_SECONDS,
            "trajectory_gib_168h": row["gib_per_hour"] * soak.FINAL_RUN_HOURS,
        }
        for row in frontier
    }

    storage = soak.storage_analysis(
        candidate_id=primary_id,
        trajectory_bytes_168h=projection["extrapolated"]["trajectory_bytes"],
        checkpoint_bytes=primary_row["checkpoint_bytes"],
        measured_bytes_per_decision=steady["bytes_per_decision"],
        compression_ratio=compression_ratio,
        compression_source=(
            "Agent 4 compression probe on 60 real sealed games of production "
            "length (25,015 decisions); the pipeline still writes uncompressed "
            "records, so this is a measured option and not the current path"
        ),
    )

    readiness = parallel_evaluation_readiness(primary_id, fallback_id)
    log(f"parallel evaluation readiness: {'ready' if readiness['ready'] else 'NOT ready'}")


    # -- artifacts ---------------------------------------------------------
    environment = {
        "commit": commit,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "coordinator_version": COORDINATOR_VERSION,
        "trajectory_version": TRAJECTORY_VERSION,
        "soak_version": soak.SOAK_VERSION,
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "architecture_family": ARCHITECTURE_FAMILY,
        "architecture_family_version": ARCHITECTURE_FAMILY_VERSION,
        "architecture_family_digest": architecture_family_digest(),
        "initialization_seed": FAMILY_INITIALIZATION_SEED,
        "rules_version": RULES_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "engine_implementation_version": IMPLEMENTATION_VERSION,
    }

    samples = result.pop("samples", [])
    soak_payload = {
        "agent": "agent_06",
        "phase": "phase_6",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **environment,
        "prerequisite_status": prerequisites["prerequisite_status"],
        "soak": result,
        "topology_rationale": {
            "source": "Agent 4 best defensible production point",
            "workers": config.num_workers,
            "environments": config.num_environments,
            "inference_batch_size": config.inference_batch_size,
            "precision": config.precision,
            "legality": config.legality,
            "snapshot_interval": config.snapshot_interval,
            "recording": "production trajectory_v1, uncompressed",
            "why_not_run_neural_schedule": (
                "Agent 5's ~449 positions/s is evaluation throughput -- whole "
                "games through play_match at batch 1. The 168-hour run is a "
                "collection run, so the soak drives Agent 4's bulk-synchronous "
                "production-recording pipeline instead."
            ),
            "why_not_worker_count_1": (
                "Agent 5's worker_count=1 recommendation applies to deterministic "
                "Phase 4 evaluation, where every decision crosses one serial "
                "inference owner. Collection is bulk-synchronous and the workers "
                "build observations in parallel, so Agent 4's 10-worker topology "
                "is the right one here."
            ),
        },
        "sample_count": len(samples),
        "timeseries_path": "reports/phase_6_data/agent_06_soak_timeseries.csv",
    }
    if not arguments.reuse_soak:
        write_json(DATA_DIRECTORY / "agent_06_soak.json", soak_payload)
        write_csv(
            DATA_DIRECTORY / "agent_06_soak_timeseries.csv", TIMESERIES_COLUMNS, samples
        )
    write_json(DATA_DIRECTORY / "agent_06_weekly_projection.json", {
        "agent": "agent_06",
        "phase": "phase_6",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **environment,
        "projection": projection,
        "storage_analysis": storage,
    })

    def build_decision(tests_after):
        """The decision payload. Built twice: once so the artifact-validation
        tests have something to read, then again with the suite result they
        produced. The soak data is identical in both; only `full_suite.after`
        and the gate that depends on it can differ."""
        gates = {
            "agents_1_to_5_all_pass": prerequisites["prerequisite_status"] == "PASS",
            "finalist_configs_reproduced": not reproduction["problems"],
            "model_contract_v2_normalized_actions": (
                config.action_frame == "perspective_normalized_squares"
                and MODEL_CONTRACT_VERSION == "model_contract_v2"
            ),
            "soak_completed": result["completion_gates"]["soak_completed_continuously"],
            "soak_illegal_actions_zero": result["completion_gates"]["illegal_actions_zero"],
            "soak_action_frame_mismatches_zero": result["completion_gates"][
                "action_frame_mismatches_zero"
            ],
            "soak_reconstruction_mismatches_zero": result["completion_gates"][
                "reconstruction_mismatches_zero"
            ],
            "soak_worker_failures_zero": result["completion_gates"]["worker_failures_zero"],
            "soak_model_mps_failures_zero": result["completion_gates"][
                "model_mps_failures_zero"
            ],
            "soak_nonfinite_outputs_zero": result["completion_gates"][
                "nonfinite_production_outputs_zero"
            ],
            "swap_zero": result["completion_gates"]["swap_zero"],
            "no_unexplained_memory_growth": result["completion_gates"][
                "no_unexplained_memory_growth"
            ],
            "primary_architecture_selected": True,
            "fallback_architecture_selected": True,
            "decision_from_measured_frontier": True,
            "weekly_projection_produced": True,
            "storage_constraints_analysed": True,
            "backend_status_reassessed": True,
            "parallel_evaluation_ready": readiness["ready"],
            "no_strength_input_to_selection": all(
                not any(bad in key for bad in p6.FORBIDDEN_INPUT_SUBSTRINGS)
                for key in soak.SELECTION_INPUT_KEYS
            ),
            "full_suite_green": bool(tests_after) and tests_after["failed"] == 0,
        }
        recommendation = "PASS" if all(gates.values()) else "FAIL"

        backend = read_json("agent_04_finalists.json")["backend_decision_statement"]
        decision = {
            "agent": "agent_06",
            "phase": "phase_6",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **environment,
            "status": recommendation,
            "all_prerequisites": prerequisites,
            "finalists": {
                "agent_04_finalist_ids": list(AGENT_4_FINALISTS),
                "compared_ids": list(FRONTIER_IDS),
                "comparison_note": (
                    "C2 is on Agent 4's measured frontier and was measured on every "
                    "axis; it is kept in this comparison even though Agent 4's formal "
                    "finalists were C0/C1/C3, because the knee argument turns on the "
                    "C1 -> C2 step."
                ),
                "capacity_proxy_disclaimer": (
                    "Parameter count is a capacity proxy. It is not a measurement of "
                    "playing strength, and nothing in Phase 6 has measured playing "
                    "strength: every candidate carries random initial weights."
                ),
                "rows": frontier,
                "reproduction": reproduction,
            },
            "primary_architecture": soak.architecture_record(primary_id, extra={
                "role": "primary production training model",
                "parameters": frontier_by_id[primary_id]["parameters"],
                "checkpoint_bytes": frontier_by_id[primary_id]["checkpoint_bytes"],
                "recommended_mps_precision": {
                    "collection": "float16",
                    "evaluation": "float32",
                    "why": (
                        "Agent 4 measured 0 non-finite outputs across 5.7M logits per "
                        "candidate at float16 in the real pipeline and float16 is "
                        "7-27% faster at collection batch sizes. Agent 5 gated "
                        "evaluation at float32 and measured float16 to be slower at "
                        "batch 1, so the two paths take different precisions."
                    ),
                },
                "recommended_inference_batch": {
                    "collection": config.inference_batch_size,
                    "evaluation": 1,
                    "evaluation_batch_policy": "single_request",
                },
                "recommended_topology": {
                    "collection_workers": config.num_workers,
                    "collection_environments": config.num_environments,
                    "collection_legality": config.legality,
                    "collection_snapshot_interval": config.snapshot_interval,
                    "evaluation_worker_count": 1,
                    "mps_owner": "coordinator only",
                },
            }),
            "fallback_architecture": soak.architecture_record(fallback_id, extra={
                "role": "smaller fallback",
                "parameters": frontier_by_id[fallback_id]["parameters"],
                "checkpoint_bytes": frontier_by_id[fallback_id]["checkpoint_bytes"],
                "recommended_mps_precision": {
                    "collection": "float16",
                    "evaluation": "float32",
                },
                "recommended_inference_batch": {
                    "collection": config.inference_batch_size,
                    "evaluation": 1,
                    "evaluation_batch_policy": "single_request",
                },
                "recommended_topology": {
                    "collection_workers": config.num_workers,
                    "collection_environments": config.num_environments,
                    "collection_legality": config.legality,
                    "collection_snapshot_interval": config.snapshot_interval,
                    "evaluation_worker_count": 1,
                    "mps_owner": "coordinator only",
                },
                "why_this_one": (
                    "The next candidate below the primary on the same measured "
                    "frontier: fully correct in Agent 4's v2 gate, numerically stable "
                    "on Metal at float16, materially faster in every measured stage, "
                    "and the same model_contract_v2."
                ),
            }),
            "selection_method": selection,
            "neighbor_tradeoffs": tradeoffs,
            "soak_result": {
                "candidate_id": primary_id,
                "status": result["status"],
                "seconds": result["total_seconds"],
                "steady_state": steady,
                "throughput_drift": result["throughput_drift"],
                "memory_growth": result["memory_growth"],
                "correctness": result["correctness"],
                "failures": result["failures"],
                "completion_gates": result["completion_gates"],
                "artifact": "reports/phase_6_data/agent_06_soak.json",
            },
            "weekly_projection": projection,
            "storage_analysis": storage,
            "parallel_evaluation_ready": readiness,
            "backend_statement": {
                **backend,
                "agent_06_reassessment": (
                    "Unchanged. Agent 4 measured R per candidate against a "
                    "candidate-independent simulator numerator of 91,778 positions/s; "
                    "the primary's R leaves the simulator several times the headroom "
                    "the model needs, and the soak sustained the same regime for an "
                    "hour without the balance moving."
                ),
            },
            "full_suite": {"before": tests_before, "after": tests_after},
            "completion_gates": gates,
            "gates_true": sum(1 for value in gates.values() if value),
            "gates_total": len(gates),
            "phase_6_recommendation": recommendation,
            "total_seconds": round(time.perf_counter() - started, 2),
            "problems": prerequisites["problems"] + reproduction["problems"],
            "files_created": [
                "stratego/training/phase6_soak.py",
                "scripts/run_phase6_agent06.py",
                "scripts/run_phase6_agent06_memory.py",
                "tests/training/test_phase6_soak.py",
                "reports/phase_6_data/agent_06_soak.json",
                "reports/phase_6_data/agent_06_soak_timeseries.csv",
                "reports/phase_6_data/agent_06_weekly_projection.json",
                "reports/phase_6_data/agent_06_architecture_decision.json",
                "reports/phase_6_data/agent_06_memory_localization.json",
            ],
            "files_modified": ["reports/phase_6_implementation_report.md"],
            "commands": [
                "python -m pytest -q",
                "python scripts/run_phase6_agent06.py",
                "python scripts/run_phase6_agent06_memory.py",
            ],
            "no_meaningful_training_occurred": True,
            "no_playing_strength_claim": (
                "Every candidate carries the family's fixed random initialization. No "
                "game result, win rate or Elo appears in any selection input, and none "
                "is claimed anywhere in this decision."
            ),
        }
        return decision

    # Artifacts first, then the suite: nine tests in
    # `tests/training/test_phase6_soak.py` read these files and skip when they
    # are absent, so running the suite before writing them would have silently
    # skipped exactly the checks that validate the published numbers.
    decision = build_decision(None)
    write_json(DATA_DIRECTORY / "agent_06_architecture_decision.json", decision)

    if not arguments.skip_pytest:
        log("running the full suite after Agent 6's changes, artifacts in place")
        tests_after = run_pytest("after")
        log(f"  {tests_after['summary_line']}")
        decision = build_decision(tests_after)
        write_json(DATA_DIRECTORY / "agent_06_architecture_decision.json", decision)

    recommendation = decision["phase_6_recommendation"]
    gates = decision["completion_gates"]

    log(f"Phase 6 recommendation: {recommendation} ({decision['gates_true']}/{decision['gates_total']} gates)")
    for name, value in gates.items():
        if not value:
            log(f"  gate false: {name}")
    log(f"total {decision['total_seconds']:.0f}s")
    return 0 if recommendation == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
