#!/usr/bin/env python3
"""Phase 6 Agent 5 acceptance harness: checkpoint-aware parallel neural evaluation.

Writes

    reports/phase_6_data/agent_05_parallel_neural_evaluation.json
    reports/phase_6_data/agent_05_greedy_worker_sweep.csv
    reports/phase_6_data/agent_05_throughput.csv
    reports/phase_6_data/agent_05_failure_cases.json
    checkpoints/phase6_c1.pt

What this script proves
-----------------------
That the same stored schedule, played through one long-lived Metal inference
owner by 1, 2, 4 and 8 CPU game workers -- and again with the schedule's input
order shuffled -- produces one results digest, one replay-digest set and zero
field-level differences, while the checkpoint is loaded exactly once.

Everything numerical here is float32. Float16 is measured for throughput only
and its results are reported as a *comparison*, never as a gate: Agent 3 already
found 0-2 natural near-tie flips per 256 positions at half precision, so a
float16 sweep would be measuring the arithmetic, not the transport.

Why the module scope is torch-free
----------------------------------
`spawn` re-imports this file inside every game worker. If torch were imported at
module scope, every "CPU game worker" would be holding a PyTorch runtime, which
is exactly the topology the MPS-ownership requirement exists to prevent. All
torch and `stratego.model` imports therefore live inside functions, and the run
*measures* the result rather than trusting it: see `information_safety` in the
data file.

Usage::

    python scripts/run_phase6_agent05.py               # full acceptance run
    python scripts/run_phase6_agent05.py --quick       # small sweep, iteration only
    python scripts/run_phase6_agent05.py --skip-pytest # measurements only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import platform
import resource
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.engine.constants import (  # noqa: E402
    ACTION_SPACE_SIZE,
    BLUE,
    IMPLEMENTATION_VERSION,
    OBSERVATION_VERSION,
    RED,
    RULES_VERSION,
)
from stratego.evaluation.match_runner import (  # noqa: E402
    MATCH_RUNNER_VERSION,
    compare_results,
    replay_stored_match,
    results_digest,
)
from stratego.evaluation.match_spec import (  # noqa: E402
    EVALUATION_SUITE_VERSION,
    PAIRING_COLOR_SWAP_SAME_BOARD,
    build_paired_schedule,
    schedule_matches,
)
from stratego.evaluation.neural_worker import (  # noqa: E402
    BATCH_POLICY_ARRIVAL,
    BATCH_POLICY_SINGLE,
    DECISION_MODE_CATEGORICAL,
    DECISION_MODE_GREEDY,
    NEURAL_WORKER_VERSION,
    REQUEST_FIELDS,
    InferenceFailure,
    InferenceOwner,
    InferenceRequest,
    InferenceResponse,
    LocalInferenceChannel,
    NeuralEvaluationError,
    RemoteNeuralPolicy,
    checkpoint_load_count,
    compare_batch_policies,
    field_level_mismatches,
    neural_policy_ref,
    reset_checkpoint_load_count,
    run_neural_schedule,
    sweep_digests,
)
from stratego.evaluation.registry import ALL_POLICY_IDS, policy_ref  # noqa: E402
from stratego.evaluation.reporting import write_json  # noqa: E402
from stratego.evaluation.setup_bank import SETUP_BANK_VERSION, SetupBank  # noqa: E402
from stratego.evaluation.statistics import summarize_run  # noqa: E402

AGENT = "agent_05"
PHASE = "phase_6"
SCHEMA_VERSION = "agent_05_parallel_neural_evaluation_0.1.0"

#: Agent 4's stable middle finalist, and the checkpoint this whole run uses.
CANDIDATE = "C1"
FAMILY_SEED = 20250601
CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase6_c1.pt"

#: The deterministic gate is float32 by instruction. Float16 is benchmarked
#: separately and never gates anything.
GATE_DTYPE = "float32"
BENCHMARK_DTYPE = "float16"

#: Three rule-based opponents. `random_legal` is deliberately excluded: Phase 5
#: measured 1,337 mean plies for greedy against it, which would triple the sweep
#: cost without widening what the sweep tests.
GREEDY_OPPONENTS = ("basic_heuristic", "tactical_rule_based", "strategic_rule_based")
SAMPLED_OPPONENTS = ("basic_heuristic", "tactical_rule_based")

GREEDY_PAIRS = 16
SAMPLED_PAIRS = 8
GREEDY_WORKER_COUNTS = (1, 2, 4, 8)
SAMPLED_WORKER_COUNTS = (1, 4, 8)

QUICK_GREEDY_PAIRS = 2
QUICK_SAMPLED_PAIRS = 2
QUICK_WORKER_COUNTS = (1, 2)

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_6_data"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def torch_report() -> dict:
    import torch

    return {
        "torch_version": str(torch.__version__),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }


def process_memory_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage if sys.platform == "darwin" else usage * 1024


def child_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def metal_memory_bytes() -> dict:
    import torch

    if not torch.backends.mps.is_available():
        return {}
    report = {}
    for key, function in (
        ("current_allocated_bytes", getattr(torch.mps, "current_allocated_memory", None)),
        ("driver_allocated_bytes", getattr(torch.mps, "driver_allocated_memory", None)),
        ("recommended_max_bytes", getattr(torch.mps, "recommended_max_memory", None)),
    ):
        if function is None:
            continue
        try:
            report[key] = int(function())
        except (RuntimeError, TypeError, ValueError):
            report[key] = None
    return report


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - a checkout without git
        return "unknown"


# ---------------------------------------------------------------------------
# 1. Prerequisites
# ---------------------------------------------------------------------------


def verify_prerequisites() -> dict:
    """Agents 1-4 must all report PASS, read from their own data files."""
    files = {
        "agent_01": "agent_01_model_contract_v2.json",
        "agent_02": "agent_02_architecture_family.json",
        "agent_03": "agent_03_architecture_shortlist.json",
        "agent_04": "agent_04_finalists.json",
    }
    status = {}
    for agent, name in files.items():
        path = DATA_DIRECTORY / name
        if not path.exists():
            status[agent] = {"status": "MISSING", "path": str(path)}
            continue
        payload = json.loads(path.read_text())
        status[agent] = {
            "status": payload.get("status"),
            "path": f"reports/phase_6_data/{name}",
            "commit": payload.get("commit"),
            "test_passed": payload.get("test_passed"),
            "test_failed": payload.get("test_failed"),
        }
    status["all_pass"] = all(
        isinstance(entry, dict) and entry.get("status") == "PASS"
        for key, entry in status.items()
        if key.startswith("agent_")
    )

    finalists = json.loads((DATA_DIRECTORY / files["agent_04"]).read_text())
    status["agent_04_finalist_ids"] = finalists["finalist_ids"]
    status["candidate_selected"] = CANDIDATE
    status["candidate_is_a_finalist"] = CANDIDATE in finalists["finalist_ids"]
    status["expected_config"] = finalists["candidate_reconstruction"]["candidates"][CANDIDATE]
    return status


def verify_checkpoint_identity(expected: dict) -> dict:
    """Build C1, write it, and prove the file describes the candidate we meant."""
    from stratego.model.architecture_configs import candidate_config, config_digests
    from stratego.model.checkpoint import (
        file_digest,
        load_checkpoint,
        read_checkpoint_payload,
        save_checkpoint,
        validate_checkpoint_payload,
    )
    from stratego.model.production_model import build_candidate_model

    config = candidate_config(CANDIDATE)
    digest = config_digests()[CANDIDATE]
    if digest != expected["config_digest"]:
        raise NeuralEvaluationError(
            f"{CANDIDATE} configuration digest {digest} does not match the digest "
            f"Agent 4 recorded ({expected['config_digest']}); the candidate family moved"
        )

    model = build_candidate_model(CANDIDATE, seed=FAMILY_SEED)
    if model.parameter_count() != expected["parameters"]:
        raise NeuralEvaluationError(
            f"{CANDIDATE} has {model.parameter_count()} parameters, Agent 4 recorded "
            f"{expected['parameters']}"
        )
    started = time.perf_counter()
    path = save_checkpoint(
        model,
        CHECKPOINT_PATH,
        training_metrics={
            "note": (
                "untrained Phase 6 candidate C1 at the family initialization seed; "
                "written by Agent 5 as an evaluation-transport fixture. Playing "
                "strength is not meaningful and must not be used as evidence."
            )
        },
    )
    write_seconds = time.perf_counter() - started

    payload = read_checkpoint_payload(path)
    metadata = validate_checkpoint_payload(payload, source=str(path))
    # The gate that matters for a family: a self-consistent file is not proof it
    # is *this* candidate, because two candidates can share every tensor shape.
    reloaded, reload_metadata = load_checkpoint(
        path,
        device="cpu",
        expected_architecture_id="stratego_transformer_v1",
        expected_configuration=config,
    )
    return {
        "candidate_id": CANDIDATE,
        "checkpoint_path": str(path.relative_to(REPOSITORY_ROOT)),
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_file_digest": file_digest(path),
        "state_dict_digest": metadata["state_dict_digest"],
        "config_digest": digest,
        "configuration": config.to_dict(),
        "parameters": model.parameter_count(),
        "initialisation_seed": FAMILY_SEED,
        "model_contract_version": metadata["model_contract_version"],
        "checkpoint_format_version": metadata["checkpoint_format_version"],
        "policy_action_frame": metadata["policy_action_frame"],
        "engine_action_frame": metadata["engine_action_frame"],
        "rules_version": metadata["rules_version"],
        "observation_version": metadata["observation_version"],
        "action_encoding_version": metadata["action_encoding_version"],
        "loads_under_v2": True,
        "expected_identity_enforced": True,
        "reload_parameter_count": reloaded.parameter_count(),
        "reload_matches": reload_metadata["state_dict_digest"] == metadata["state_dict_digest"],
        "write_seconds": round(write_seconds, 4),
    }


# ---------------------------------------------------------------------------
# 2. Schedules
# ---------------------------------------------------------------------------


def build_schedule(reference, opponents, pair_count: int):
    matches = []
    for opponent_id in opponents:
        matches.extend(
            schedule_matches(
                build_paired_schedule(reference, policy_ref(opponent_id), range(pair_count))
            )
        )
    return tuple(matches)


def shuffled(matches, seed: int = 20260512):
    """A different input order with identical contents."""
    order = np.random.default_rng(seed).permutation(len(matches))
    return tuple(matches[int(index)] for index in order)


# ---------------------------------------------------------------------------
# 3. The reproducibility sweeps
# ---------------------------------------------------------------------------


def run_sweep(matches, bank, owner, worker_counts, *, label: str) -> tuple[dict, list]:
    """The same schedule at each worker count, then once with the input shuffled."""
    runs: dict[str, Any] = {}
    rows = []
    for count in worker_counts:
        started_cpu = child_cpu_seconds()
        run = run_neural_schedule(matches, bank, owner, worker_count=count)
        runs[str(count)] = run
        rows.append(_sweep_row(label, str(count), run, child_cpu_seconds() - started_cpu))
        print(
            f"  [{label}] {count:>2} workers  {run.matches_run:>4} matches  "
            f"{run.plies:>7,} plies  {run.decisions:>7,} decisions  "
            f"{run.wall_clock_seconds:6.1f}s  {run.results_digest[:12]}"
        )

    top = max(worker_counts)
    started_cpu = child_cpu_seconds()
    run = run_neural_schedule(shuffled(matches), bank, owner, worker_count=top)
    key = f"{top}_shuffled"
    runs[key] = run
    rows.append(_sweep_row(label, key, run, child_cpu_seconds() - started_cpu))
    print(
        f"  [{label}] {top:>2} workers, shuffled input  "
        f"{run.wall_clock_seconds:6.1f}s  {run.results_digest[:12]}"
    )
    return runs, rows


def _sweep_row(label: str, key: str, run, worker_cpu_seconds: float) -> dict:
    inference = run.inference
    return {
        "sweep": label,
        "run": key,
        "worker_count": run.worker_count,
        "shuffled_input": key.endswith("_shuffled"),
        "chunk_count": run.chunk_count,
        "matches": run.matches_run,
        "paired_units": run.paired_units_run,
        "plies": run.plies,
        "decisions": run.decisions,
        "wall_clock_seconds": round(run.wall_clock_seconds, 4),
        "matches_per_second": round(run.matches_run / run.wall_clock_seconds, 4),
        "positions_per_second": round(run.plies / run.wall_clock_seconds, 3),
        "decisions_per_second": round(run.decisions / run.wall_clock_seconds, 3),
        "inference_seconds": inference["inference_seconds"],
        "inference_fraction": round(
            inference["inference_seconds"] / run.wall_clock_seconds, 4
        ),
        # Everything the owner does per decision, not just the forward pass. The
        # gap between the two is the owner's single-threaded CPU cost -- the
        # legality cross-check, both frame conversions and selection -- which
        # sits on the critical path of every decision in the run.
        "owner_seconds": inference["serve_seconds"],
        "owner_fraction_of_wall_clock": round(
            inference["serve_seconds"] / run.wall_clock_seconds, 4
        ),
        "outside_owner_seconds": round(
            run.wall_clock_seconds - inference["serve_seconds"], 4
        ),
        "owner_ms_per_decision": round(
            1000 * inference["serve_seconds"] / max(run.decisions, 1), 4
        ),
        "forward_ms_per_decision": round(
            1000 * inference["inference_seconds"] / max(run.decisions, 1), 4
        ),
        "owner_cpu_ms_per_decision": round(
            1000
            * (inference["serve_seconds"] - inference["inference_seconds"])
            / max(run.decisions, 1),
            4,
        ),
        "mean_batch_size": inference["mean_batch_size"],
        "max_batch_size": inference["max_batch_size_seen"],
        "queue_wait_mean_seconds": inference["queue_wait_mean_seconds"],
        "queue_wait_max_seconds": inference["queue_wait_max_seconds"],
        "worker_cpu_seconds": round(worker_cpu_seconds, 3),
        "worker_cpu_utilisation": round(
            worker_cpu_seconds / (run.wall_clock_seconds * max(run.worker_count, 1)), 4
        ),
        "checkpoint_load_count": inference["checkpoint_load_count"],
        "workers_importing_torch": run.workers_importing_torch,
        "worker_checkpoint_loads": run.worker_checkpoint_loads,
        "policy_errors": run.policy_errors,
        "illegal_policy_actions": run.illegal_policy_actions,
        "results_digest": run.results_digest,
        "schedule_digest": run.schedule_digest,
    }


def sweep_verdict(runs: dict, *, baseline: str = "1", resamples: int = 500) -> dict:
    """Everything the greedy gate is stated in, computed from the runs."""
    mismatches = field_level_mismatches(runs, baseline=baseline)
    digests = sweep_digests(runs)

    reference = runs[baseline]
    fields = (
        "match_id",
        "paired_unit_id",
        "red_setup",
        "blue_setup",
        "candidate_seed",
        "opponent_seed",
        "action_history",
        "replay_digest",
        "winner",
        "terminal_reason",
        "plies",
    )
    per_field = {}
    for field in fields:
        expected = {row.match_id: getattr(row, field) for row in reference.results}
        differing = 0
        for label, run in runs.items():
            if label == baseline:
                continue
            actual = {row.match_id: getattr(row, field) for row in run.results}
            differing += sum(1 for key in expected if expected[key] != actual.get(key))
        per_field[field] = differing

    statistics = {
        label: summarize_run(run.results, resamples=resamples) for label, run in runs.items()
    }
    statistics_identical = all(
        summary == statistics[baseline] for summary in statistics.values()
    )

    return {
        "runs": sorted(runs),
        "baseline": baseline,
        "field_level_mismatches": len(mismatches),
        "mismatch_examples": mismatches[:10],
        "per_field_mismatches": per_field,
        "distinct_results_digests": digests["distinct_results_digests"],
        "distinct_replay_digest_sets": digests["distinct_replay_digest_sets"],
        "results_digests": digests["results_digests"],
        "replay_digest_set_digests": digests["replay_digest_set_digests"],
        "statistics_identical_across_runs": statistics_identical,
        "statistics_resamples": resamples,
        # Reported so the sweep's statistics are visible in the artifact, never
        # as evidence about the architecture: these are random weights.
        "per_opponent_effective_win_rate": {
            token: entry.get("effective_win_rate")
            for token, entry in statistics[baseline]["per_opponent"].items()
        },
        "mean_plies": statistics[baseline]["plies"].get("mean"),
        "terminal_reasons": statistics[baseline]["terminal_reasons"],
        "colours_played": sorted(
            {row.candidate_color for row in reference.results}
        ),
        "opponents": sorted({row.opponent_policy_id for row in reference.results}),
        "matches": reference.matches_run,
        "paired_units": reference.paired_units_run,
        "plies": reference.plies,
        "decisions": reference.decisions,
        "policy_errors": max(run.policy_errors for run in runs.values()),
        "illegal_policy_actions": max(run.illegal_policy_actions for run in runs.values()),
        "checkpoint_load_counts": {
            label: run.inference["checkpoint_load_count"] for label, run in runs.items()
        },
    }


def cross_check_serial_adapter(matches, bank, owner, reference, device: str) -> dict:
    """Prove the remote path is the Phase 5 adapter's path, not a lookalike.

    The comparison runs the *unmodified* `NeuralCheckpointPolicy` in this process
    against the same checkpoint on the same device, through the frozen Phase 4
    `run_schedule`. Any difference would mean the owner reimplemented a rule.
    """
    from stratego.evaluation.match_runner import run_schedule
    from stratego.model.policy_adapter import NeuralCheckpointPolicy

    class DirectPolicy(NeuralCheckpointPolicy):
        policy_id = reference.policy_id
        policy_version = reference.policy_version
        decision_mode = owner.decision_mode

    import torch

    direct = DirectPolicy.from_checkpoint(
        owner.checkpoint_path, device=device, dtype=getattr(torch, owner.dtype_name)
    )
    started = time.perf_counter()
    serial = run_schedule(matches, bank, policies={reference.token: direct}, worker_count=1)
    elapsed = time.perf_counter() - started
    remote = run_neural_schedule(matches, bank, owner, worker_count=max(2, 1))
    differences = compare_results(serial.results, remote.results)
    return {
        "comparison": "stratego.model.policy_adapter.NeuralCheckpointPolicy in-process "
        "vs the same checkpoint served by the inference owner to CPU game workers",
        "device": device,
        "matches": serial.matches_run,
        "identical": not differences,
        "field_differences": len(differences),
        "examples": differences[:5],
        "serial_results_digest": serial.results_digest,
        "remote_results_digest": remote.results_digest,
        "serial_wall_clock_seconds": round(elapsed, 3),
        "remote_wall_clock_seconds": round(remote.wall_clock_seconds, 3),
    }


def prove_modes_differ(greedy_runs: dict, sampled_runs: dict) -> dict:
    """The seeded-categorical gate must not be accidentally re-running greedy."""
    greedy = greedy_runs["1"]
    sampled = sampled_runs["1"]
    greedy_by_key = {
        (row.opponent_policy_id, row.setup_pair_id, row.candidate_color): row
        for row in greedy.results
    }
    shared = 0
    differing_actions = 0
    differing_results = 0
    for row in sampled.results:
        key = (row.opponent_policy_id, row.setup_pair_id, row.candidate_color)
        other = greedy_by_key.get(key)
        if other is None:
            continue
        shared += 1
        if row.action_history != other.action_history:
            differing_actions += 1
        if (row.winner, row.terminal_reason, row.plies) != (
            other.winner,
            other.terminal_reason,
            other.plies,
        ):
            differing_results += 1
    return {
        "comparable_positions": shared,
        "matches_with_different_action_histories": differing_actions,
        "matches_with_different_outcomes": differing_results,
        "greedy_results_digest": greedy.results_digest,
        "sampled_results_digest": sampled.results_digest,
        "digests_differ": greedy.results_digest != sampled.results_digest,
        "stochastic_path_is_distinct": differing_actions > 0,
    }


# ---------------------------------------------------------------------------
# 4. Observer-safe payload audit
# ---------------------------------------------------------------------------


def capture_requests(matches, bank, owner, reference, limit: int = 64) -> list:
    """Real inference requests, taken from a real game.

    Used by both the payload audit and the batch-invariance probe, because both
    questions are only meaningful about requests the engine actually produced --
    a synthetic observation exercises neither the legality products nor the
    natural distribution of near-ties.
    """
    from stratego.evaluation.match_runner import play_match

    captured: list = []
    sources: list = []

    class Capturing(RemoteNeuralPolicy):
        def decide(self, request):
            if len(captured) < limit:
                captured.append(InferenceRequest.from_policy_input(request))
                sources.append(request.observation)
            return super().decide(request)

    policy = Capturing(reference, LocalInferenceChannel(owner), decision_mode=owner.decision_mode)
    play_match(matches[0], bank=bank, policies={reference.token: policy})
    return captured, sources


def measure_batch_invariance(owner, requests, batch_sizes=(2, 4, 8)) -> dict:
    """Does a position get the same logit row alone and inside a batch?

    Measured, never assumed. A `True` here does not make `arrival_batched`
    gate-eligible -- batch *membership* would still depend on arrival timing --
    but it is the evidence a later agent needs before choosing to trade the
    guarantee for the 1.7x this run measured.
    """
    import torch

    probe = list(requests[: max(batch_sizes)])
    if len(probe) < max(batch_sizes):  # pragma: no cover - a very short game
        return {"measured": False, "reason": "not enough captured requests"}

    alone = [owner.probe_policy_logits([request])[0] for request in probe]
    report: dict[str, Any] = {"measured": True, "requests_probed": len(probe), "per_batch_size": {}}
    for size in batch_sizes:
        rows = owner.probe_policy_logits(probe[:size])
        identical = sum(1 for index in range(size) if torch.equal(rows[index], alone[index]))
        largest = max(
            float((rows[index] - alone[index]).abs().max()) for index in range(size)
        )
        decisions_alone = [owner.serve(request) for request in probe[:size]]
        decisions_batched = owner.serve_batch(probe[:size])
        flips = sum(
            1
            for index in range(size)
            if decisions_alone[index].absolute_action_id
            != decisions_batched[index].absolute_action_id
        )
        report["per_batch_size"][str(size)] = {
            "rows": size,
            "bitwise_identical_rows": identical,
            "max_absolute_logit_difference": largest,
            "selected_action_flips": flips,
        }
    report["bitwise_batch_invariant"] = all(
        entry["bitwise_identical_rows"] == entry["rows"]
        for entry in report["per_batch_size"].values()
    )
    report["selected_action_flips_total"] = sum(
        entry["selected_action_flips"] for entry in report["per_batch_size"].values()
    )
    report["gate_eligibility_note"] = (
        "batch invariance of the arithmetic does not make arrival_batched "
        "gate-eligible: its batch membership still depends on which workers "
        "happened to be waiting. This measurement is evidence for Agent 6, not "
        "a licence to batch the deterministic path."
    )
    return report


def audit_payload(payload, source_observation) -> dict:
    """Walk everything reachable from one real request."""

    forbidden = (
        "GameState",
        "PieceRecord",
        "ReplayRecord",
        "MoveRecord",
        "SetupBank",
        "SetupPair",
        "MatchSpec",
        "PolicyInput",
        "PublicView",
    )
    seen: set[int] = set()
    types: set[str] = set()

    def walk(value, depth=0):
        if id(value) in seen or depth > 8:
            return
        seen.add(id(value))
        types.add(type(value).__name__)
        if isinstance(value, np.ndarray):
            if value.base is not None:
                walk(value.base, depth + 1)
            return
        if isinstance(value, (str, bytes, int, float, bool, type(None))):
            return
        if isinstance(value, dict):
            for key, item in value.items():
                walk(key, depth + 1)
                walk(item, depth + 1)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                walk(item, depth + 1)
            return
        for name in getattr(value, "__dataclass_fields__", ()) or ():
            walk(getattr(value, name), depth + 1)
        if hasattr(value, "__dict__"):
            for item in vars(value).values():
                walk(item, depth + 1)

    walk(payload)
    blob = pickle.dumps(payload)
    return {
        "declared_fields": list(REQUEST_FIELDS),
        "dataclass_fields": list(InferenceRequest.__dataclass_fields__),
        "fields_match_declaration": tuple(InferenceRequest.__dataclass_fields__)
        == REQUEST_FIELDS,
        "reachable_types": sorted(types),
        "forbidden_types_reachable": sorted(set(types) & set(forbidden)),
        "object_graph_clean": not set(types) & set(forbidden),
        "arrays_alias_engine_memory": bool(
            payload.observation.base is not None or payload.legal_action_mask.base is not None
        ),
        "observation_copied_from_engine": payload.observation is not source_observation,
        "pickled_bytes": len(blob),
        "pickle_names_privileged_class": any(
            name.encode() in blob for name in forbidden
        ),
        "pickle_names_engine_state_module": b"stratego.engine.state" in blob,
        "transported_products": [
            "match/request identity (match_id, paired_unit_id, ply, request_id)",
            "per-decision seed",
            "acting player",
            f"{OBSERVATION_VERSION} observation",
            "absolute legal-action list and dense legality mask",
        ],
        "never_transported": [
            "GameState",
            "PieceRecord",
            "hidden true identities",
            "privileged belief targets",
            "true opponent setup",
            "privileged replay object",
        ],
        "policy_requirements": RemoteNeuralPolicy.requirements.to_dict(),
    }


# ---------------------------------------------------------------------------
# 5. Failure behaviour
# ---------------------------------------------------------------------------


def _dense_mask(actions) -> np.ndarray:
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.uint8)
    mask[np.asarray(list(actions), dtype=np.int64)] = 1
    return mask


def _sample_request(**overrides) -> InferenceRequest:
    base = InferenceRequest(
        request_id="m-probe#0",
        match_id="m-probe",
        paired_unit_id="u-probe",
        ply=0,
        acting_player=RED,
        decision_seed=99,
        observation=np.zeros((127, 10, 10), dtype=np.float32),
        legal_actions=(101, 202, 303),
        legal_action_mask=_dense_mask((101, 202, 303)),
    )
    return replace(base, **overrides) if overrides else base


def run_failure_suite(owner, matches, bank, reference, tmp_directory: Path) -> dict:
    """Every failure the instructions name, each checked for two things: that it
    is loud, and that no action came back."""
    import torch

    from stratego.evaluation.match_runner import (
        ON_POLICY_ERROR_QUARANTINE,
        RESULT_ERROR,
        PolicyFailure,
        play_match,
    )
    from stratego.model.architecture_configs import candidate_config
    from stratego.model.checkpoint import CheckpointError

    cases: dict[str, dict] = {}

    def record(name, *, passed, detail, substituted_move=False):
        cases[name] = {
            "passed": bool(passed),
            "detail": str(detail)[:400],
            "substituted_a_move": bool(substituted_move),
        }

    # -- missing checkpoint ------------------------------------------------
    try:
        InferenceOwner(tmp_directory / "not_here.pt", device="cpu")
        record("missing_checkpoint", passed=False, detail="an absent file was accepted")
    except Exception as error:
        record("missing_checkpoint", passed=True, detail=f"{type(error).__name__}: {error}")

    # -- incompatible checkpoint (contract v1) -----------------------------
    legacy = REPOSITORY_ROOT / "checkpoints" / "integration_model_v1.pt"
    if legacy.exists():
        try:
            InferenceOwner(legacy, device="cpu")
            record(
                "incompatible_checkpoint_contract_v1",
                passed=False,
                detail="a model_contract_v1 file was accepted under v2",
            )
        except CheckpointError as error:
            record("incompatible_checkpoint_contract_v1", passed=True, detail=error)
    else:  # pragma: no cover
        record(
            "incompatible_checkpoint_contract_v1",
            passed=False,
            detail="checkpoints/integration_model_v1.pt is not present",
        )

    # -- incompatible checkpoint (wrong candidate) -------------------------
    try:
        InferenceOwner(
            owner.checkpoint_path,
            device="cpu",
            expected_configuration=candidate_config("C3"),
        )
        record(
            "incompatible_checkpoint_wrong_candidate",
            passed=False,
            detail="a C1 file was accepted as C3",
        )
    except CheckpointError as error:
        record("incompatible_checkpoint_wrong_candidate", passed=True, detail=error)

    # -- corrupted checkpoint ----------------------------------------------
    corrupted = tmp_directory / "corrupted.pt"
    payload = owner.checkpoint_path.read_bytes()
    corrupted.write_bytes(payload[: len(payload) // 2])
    try:
        InferenceOwner(corrupted, device="cpu")
        record("corrupted_checkpoint", passed=False, detail="a truncated file was accepted")
    except Exception as error:
        record("corrupted_checkpoint", passed=True, detail=f"{type(error).__name__}: {error}")

    # -- malformed requests --------------------------------------------------
    malformed = {
        "wrong_observation_shape": {"observation": np.zeros((3, 10, 10), dtype=np.float32)},
        "non_finite_observation": {
            "observation": np.full((127, 10, 10), np.nan, dtype=np.float32)
        },
        "unknown_acting_player": {"acting_player": 7},
        "wrong_mask_length": {"legal_action_mask": np.zeros(7, dtype=np.uint8)},
        "empty_legal_actions": {"legal_actions": ()},
        "negative_decision_seed": {"decision_seed": -1},
        "mask_disagrees_with_list": {"legal_action_mask": _dense_mask((101, 202))},
    }
    refusals = {}
    for name, override in malformed.items():
        answer = owner.serve(_sample_request(**override))
        refusals[name] = {
            "refused": isinstance(answer, InferenceFailure),
            "error_type": getattr(answer, "error_type", None),
            "returned_an_action": isinstance(answer, InferenceResponse),
        }
    record(
        "malformed_request",
        passed=all(entry["refused"] and not entry["returned_an_action"] for entry in refusals.values()),
        detail=json.dumps(refusals, sort_keys=True),
        substituted_move=any(entry["returned_an_action"] for entry in refusals.values()),
    )
    # A good request still works: one bad request must not poison the owner.
    record(
        "owner_survives_a_malformed_request",
        passed=isinstance(owner.serve(_sample_request()), InferenceResponse),
        detail="a well-formed request is still answered after seven refusals",
    )

    # -- non-finite model output --------------------------------------------
    poisoned = InferenceOwner(owner.checkpoint_path, device="cpu", name="poisoned")
    try:
        with torch.no_grad():
            poisoned.model.policy_source_bias.fill_(float("nan"))
        answer = poisoned.serve(_sample_request())
        record(
            "non_finite_model_output",
            passed=isinstance(answer, InferenceFailure)
            and "non-finite" in getattr(answer, "message", ""),
            detail=getattr(answer, "message", answer),
            substituted_move=isinstance(answer, InferenceResponse),
        )
    finally:
        poisoned.close()

    # -- a normalized choice converting to an illegal absolute action --------
    from stratego.model import policy_adapter as adapter

    original = adapter.model_action_to_absolute
    adapter.model_action_to_absolute = lambda action, player: 9999
    try:
        answer = owner.serve(_sample_request(request_id="m-frame#0"))
        record(
            "normalized_selection_converts_to_illegal_action",
            passed=isinstance(answer, InferenceFailure)
            and "did not declare legal" in getattr(answer, "message", ""),
            detail=getattr(answer, "message", answer),
            substituted_move=isinstance(answer, InferenceResponse),
        )
    finally:
        adapter.model_action_to_absolute = original

    # -- inference coordinator failure --------------------------------------
    faulty = InferenceOwner(owner.checkpoint_path, device="cpu", name="faulty")
    calls = {"n": 0}

    def explode(requests):
        calls["n"] += 1
        if calls["n"] > 5:
            raise RuntimeError("simulated inference coordinator failure")

    faulty.fault_hook = explode
    try:
        run_neural_schedule(matches[:2], bank, faulty, worker_count=2)
        record("inference_coordinator_failure", passed=False, detail="the run completed anyway")
    except NeuralEvaluationError as error:
        record(
            "inference_coordinator_failure",
            passed="simulated inference coordinator failure" in str(error),
            detail=error,
        )
    finally:
        faulty.close()

    # -- quarantine semantics: an errored match carries no score -------------
    quarantined = InferenceOwner(owner.checkpoint_path, device="cpu", name="quarantined")
    quarantined.fault_hook = lambda requests: (_ for _ in ()).throw(RuntimeError("device lost"))
    try:
        policy = RemoteNeuralPolicy(
            reference, LocalInferenceChannel(quarantined), decision_mode=owner.decision_mode
        )
        row = play_match(
            matches[0],
            bank=bank,
            policies={reference.token: policy},
            on_policy_error=ON_POLICY_ERROR_QUARANTINE,
        )
        record(
            "phase4_quarantine_semantics_preserved",
            passed=(
                row.candidate_result == RESULT_ERROR
                and row.candidate_score is None
                and row.terminal_reason == "policy_error"
                and policy.decisions == 0
            ),
            detail=f"result={row.candidate_result} score={row.candidate_score} "
            f"terminal={row.terminal_reason} decisions={policy.decisions}",
            substituted_move=policy.decisions > 0,
        )
    finally:
        quarantined.close()

    # -- timeout / disconnect -------------------------------------------------
    from stratego.evaluation.neural_worker import RemoteInferenceError

    class SilentChannel:
        transport = "silent"

        def infer(self, request):
            raise RemoteInferenceError(
                f"the inference owner did not answer request {request.request_id!r} "
                "within 0.01s; refusing to continue this match"
            )

        def stats(self):
            return {}

    silent_policy = RemoteNeuralPolicy(
        reference, SilentChannel(), decision_mode=owner.decision_mode
    )
    try:
        play_match(matches[0], bank=bank, policies={reference.token: silent_policy})
        record("timeout_or_disconnect", passed=False, detail="the match continued anyway")
    except PolicyFailure as error:
        record(
            "timeout_or_disconnect",
            passed=isinstance(error.__cause__, RemoteInferenceError)
            and silent_policy.decisions == 0,
            detail=error,
            substituted_move=silent_policy.decisions > 0,
        )

    # -- a crossed or reseeded answer ----------------------------------------
    class CrossingChannel(LocalInferenceChannel):
        def infer(self, request):
            answer = super().infer(request)
            return replace(answer, request_id="m-somewhere-else#0")

    crossing_policy = RemoteNeuralPolicy(
        reference, CrossingChannel(owner), decision_mode=owner.decision_mode
    )
    try:
        play_match(matches[0], bank=bank, policies={reference.token: crossing_policy})
        record("crossed_response", passed=False, detail="a foreign answer was accepted")
    except PolicyFailure as error:
        record(
            "crossed_response",
            passed=isinstance(error.__cause__, RemoteInferenceError),
            detail=error,
            substituted_move=crossing_policy.decisions > 0,
        )

    cases["summary"] = {
        "cases": len(cases),
        "passed": sum(1 for entry in cases.values() if entry.get("passed")),
        "failed": sorted(name for name, entry in cases.items() if not entry.get("passed")),
        "any_substituted_move": any(entry.get("substituted_a_move") for entry in cases.values()),
        "substitution_policy": "no random legal, first legal or previous action is ever "
        "returned after a failure; every failure path raises or quarantines",
    }
    return cases


# ---------------------------------------------------------------------------
# 6. Throughput and memory
# ---------------------------------------------------------------------------


def measure_precision_and_batching(matches, bank, quick: bool) -> dict:
    """Two performance instruments, neither of which gates anything.

    Both are reported with their decision agreement against the float32
    `single_request` path, because "faster" is only interesting alongside
    "and here is exactly what it changed".
    """
    reference_ref = neural_policy_ref(CANDIDATE, dtype_name=GATE_DTYPE)
    gate_owner = InferenceOwner(
        CHECKPOINT_PATH, device="mps", dtype=GATE_DTYPE, name="benchmark_gate"
    )
    workers = 2 if quick else 8
    try:
        baseline = run_neural_schedule(matches, bank, gate_owner, worker_count=workers)
    finally:
        gate_owner.close()

    report: dict[str, Any] = {
        "baseline": {
            "dtype": GATE_DTYPE,
            "batch_policy": BATCH_POLICY_SINGLE,
            "worker_count": workers,
            "matches_per_second": round(baseline.matches_run / baseline.wall_clock_seconds, 4),
            "positions_per_second": round(baseline.plies / baseline.wall_clock_seconds, 3),
            "decisions_per_second": round(baseline.decisions / baseline.wall_clock_seconds, 3),
            "wall_clock_seconds": round(baseline.wall_clock_seconds, 3),
            "results_digest": baseline.results_digest,
        }
    }

    # -- float16, single request --------------------------------------------
    half_ref = neural_policy_ref(CANDIDATE, dtype_name=BENCHMARK_DTYPE)
    half_matches = _rebind(matches, reference_ref, half_ref)
    half_owner = InferenceOwner(
        CHECKPOINT_PATH, device="mps", dtype=BENCHMARK_DTYPE, name="benchmark_float16"
    )
    try:
        half = run_neural_schedule(half_matches, bank, half_owner, worker_count=workers)
    finally:
        half_owner.close()
    report["float16"] = {
        "dtype": BENCHMARK_DTYPE,
        "batch_policy": BATCH_POLICY_SINGLE,
        "worker_count": workers,
        "matches_per_second": round(half.matches_run / half.wall_clock_seconds, 4),
        "positions_per_second": round(half.plies / half.wall_clock_seconds, 3),
        "decisions_per_second": round(half.decisions / half.wall_clock_seconds, 3),
        "wall_clock_seconds": round(half.wall_clock_seconds, 3),
        "speedup_vs_float32": round(
            baseline.wall_clock_seconds / half.wall_clock_seconds, 4
        ),
        "policy_identity": half_ref.token,
        "gate_eligible": False,
        "note": (
            "a separate policy identity because a half-precision forward is a "
            "different decision rule; not used for any reproducibility gate"
        ),
    }

    # -- float32, arrival batched --------------------------------------------
    batched_owner = InferenceOwner(
        CHECKPOINT_PATH,
        device="mps",
        dtype=GATE_DTYPE,
        batch_policy=BATCH_POLICY_ARRIVAL,
        max_batch_size=max(workers, 2),
        name="benchmark_batched",
    )
    try:
        batched = run_neural_schedule(matches, bank, batched_owner, worker_count=workers)
    finally:
        batched_owner.close()
    agreement = compare_batch_policies(baseline, batched)
    report["arrival_batched"] = {
        "dtype": GATE_DTYPE,
        "batch_policy": BATCH_POLICY_ARRIVAL,
        "worker_count": workers,
        "max_batch_size_configured": max(workers, 2),
        "max_batch_size_seen": batched.inference["max_batch_size_seen"],
        "mean_batch_size": batched.inference["mean_batch_size"],
        "batch_size_histogram": batched.inference["batch_size_histogram"],
        "matches_per_second": round(batched.matches_run / batched.wall_clock_seconds, 4),
        "positions_per_second": round(batched.plies / batched.wall_clock_seconds, 3),
        "decisions_per_second": round(batched.decisions / batched.wall_clock_seconds, 3),
        "wall_clock_seconds": round(batched.wall_clock_seconds, 3),
        "speedup_vs_single_request": round(
            baseline.wall_clock_seconds / batched.wall_clock_seconds, 4
        ),
        "agreement_with_single_request": agreement,
        "gate_eligible": False,
        "note": (
            "batch membership depends on which workers happened to be waiting, so "
            "this path is a measurement only. The agreement figure is measured, "
            "not assumed, and does not make the path gate-eligible."
        ),
    }
    return report


def _rebind(matches, old_ref, new_ref):
    """The same schedule under a different policy identity.

    Used only by the float16 benchmark: a half-precision forward is a different
    decision rule, so it must not borrow the float32 policy's `match_id`s.
    """
    return tuple(
        replace(
            spec,
            candidate=new_ref if spec.candidate == old_ref else spec.candidate,
            opponent=new_ref if spec.opponent == old_ref else spec.opponent,
        )
        for spec in matches
    )


# ---------------------------------------------------------------------------
# 7. Tests
# ---------------------------------------------------------------------------


def run_pytest(paths=None) -> dict:
    import re

    started = time.perf_counter()
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if paths:
        command.extend(paths)
    completed = subprocess.run(
        command, cwd=REPOSITORY_ROOT, capture_output=True, text=True
    )
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    numbers = {}
    for key in ("passed", "failed", "skipped", "error"):
        match = re.search(rf"(\d+) {key}", tail)
        numbers[key] = int(match.group(1)) if match else 0
    numbers["total"] = numbers["passed"] + numbers["failed"] + numbers["skipped"]
    numbers["exit_code"] = completed.returncode
    numbers["summary_line"] = tail
    numbers["seconds"] = round(time.perf_counter() - started, 2)
    numbers["command"] = " ".join(command)
    return numbers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small sweep, iteration only")
    parser.add_argument("--skip-pytest", action="store_true", help="measurements only")
    options = parser.parse_args()

    started = time.perf_counter()
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    tmp_directory = DATA_DIRECTORY / "_agent_05_tmp"
    tmp_directory.mkdir(exist_ok=True)

    greedy_pairs = QUICK_GREEDY_PAIRS if options.quick else GREEDY_PAIRS
    sampled_pairs = QUICK_SAMPLED_PAIRS if options.quick else SAMPLED_PAIRS
    greedy_counts = QUICK_WORKER_COUNTS if options.quick else GREEDY_WORKER_COUNTS
    sampled_counts = QUICK_WORKER_COUNTS if options.quick else SAMPLED_WORKER_COUNTS
    greedy_opponents = GREEDY_OPPONENTS[:2] if options.quick else GREEDY_OPPONENTS

    print(f"Phase 6 Agent 5 -- checkpoint-aware parallel neural evaluation "
          f"({'quick' if options.quick else 'full'})")

    # -- 1. prerequisites ---------------------------------------------------
    prerequisites = verify_prerequisites()
    print(f"[1/9] prerequisites: agents 1-4 PASS = {prerequisites['all_pass']}, "
          f"finalists {prerequisites['agent_04_finalist_ids']}, using {CANDIDATE}")
    if not prerequisites["all_pass"]:
        raise NeuralEvaluationError("agents 1-4 are not all PASS; Agent 5 is BLOCKED")

    tests_before = (
        {"skipped": True} if options.skip_pytest else run_pytest()
    )
    if not options.skip_pytest:
        print(f"      suite before edits: {tests_before['summary_line']}")

    # -- 2. checkpoint -------------------------------------------------------
    checkpoint = verify_checkpoint_identity(prerequisites["expected_config"])
    print(f"[2/9] checkpoint: {checkpoint['checkpoint_path']} "
          f"{checkpoint['parameters']:,} parameters, {checkpoint['model_contract_version']}")

    environment = torch_report()
    if not environment["mps_available"]:
        raise NeuralEvaluationError(
            "Metal is not available; the MPS ownership measurement cannot be made "
            "and must not be silently taken on the CPU"
        )

    # -- 3. the owner --------------------------------------------------------
    reset_checkpoint_load_count()
    bank = SetupBank.generate(size=max(greedy_pairs, sampled_pairs))
    greedy_ref = neural_policy_ref(CANDIDATE, decision_mode=DECISION_MODE_GREEDY)
    sampled_ref = neural_policy_ref(CANDIDATE, decision_mode=DECISION_MODE_CATEGORICAL)

    from stratego.model.architecture_configs import candidate_config

    owner = InferenceOwner(
        CHECKPOINT_PATH,
        decision_mode=DECISION_MODE_GREEDY,
        device="mps",
        dtype=GATE_DTYPE,
        expected_architecture_id="stratego_transformer_v1",
        expected_configuration=candidate_config(CANDIDATE),
        batch_policy=BATCH_POLICY_SINGLE,
        max_batch_size=1,
        name="phase6_c1_greedy_owner",
    )
    print(f"[3/9] owner: {owner.identity()['device']} {owner.dtype_name}, "
          f"checkpoint loaded {owner.checkpoint_load_count}x in "
          f"{owner.checkpoint_load_seconds:.3f}s")

    greedy_matches = build_schedule(greedy_ref, greedy_opponents, greedy_pairs)
    sampled_matches = build_schedule(sampled_ref, SAMPLED_OPPONENTS, sampled_pairs)

    # -- 4. observer-safe payload -------------------------------------------
    probe_requests, probe_sources = capture_requests(greedy_matches, bank, owner, greedy_ref)
    information_safety = audit_payload(probe_requests[0], probe_sources[0])
    batch_invariance = measure_batch_invariance(owner, probe_requests)
    print(f"[4/9] payload audit: object graph clean = "
          f"{information_safety['object_graph_clean']}, "
          f"reachable types {information_safety['reachable_types']}")
    print(f"      batch invariance probe: bitwise invariant = "
          f"{batch_invariance.get('bitwise_batch_invariant')}, "
          f"action flips = {batch_invariance.get('selected_action_flips_total')}")

    # -- 5. greedy sweep -----------------------------------------------------
    print(f"[5/9] greedy sweep: {len(greedy_matches)} matches, "
          f"{len(greedy_opponents)} opponents, {greedy_pairs} setup pairs")
    greedy_runs, greedy_rows = run_sweep(
        greedy_matches, bank, owner, greedy_counts, label="greedy_float32"
    )
    greedy_verdict = sweep_verdict(greedy_runs)
    print(f"      mismatches={greedy_verdict['field_level_mismatches']} "
          f"results digests={greedy_verdict['distinct_results_digests']} "
          f"replay-digest sets={greedy_verdict['distinct_replay_digest_sets']}")

    adapter_cross_check = cross_check_serial_adapter(
        greedy_matches, bank, owner, greedy_ref, device="mps"
    )
    print(f"      identical to the in-process Phase 5 adapter: "
          f"{adapter_cross_check['identical']}")

    replay_problems = []
    sample = greedy_runs["1"].results[:: max(1, len(greedy_runs["1"].results) // 16)]
    for row in sample:
        replay_problems.extend(replay_stored_match(row))

    memory_after_greedy = {
        "process_rss_bytes": process_memory_bytes(),
        "metal": metal_memory_bytes(),
    }

    # -- 6. seeded categorical sweep ----------------------------------------
    sampled_owner = InferenceOwner(
        CHECKPOINT_PATH,
        decision_mode=DECISION_MODE_CATEGORICAL,
        device="mps",
        dtype=GATE_DTYPE,
        expected_configuration=candidate_config(CANDIDATE),
        name="phase6_c1_sampled_owner",
    )
    print(f"[6/9] seeded categorical sweep: {len(sampled_matches)} matches")
    try:
        sampled_runs, sampled_rows = run_sweep(
            sampled_matches, bank, sampled_owner, sampled_counts, label="sampled_float32"
        )
        sampled_verdict = sweep_verdict(sampled_runs)
        sampled_load_count = sampled_owner.checkpoint_load_count
    finally:
        sampled_owner.close()

    # The greedy owner is still the one holding the greedy schedule, so the
    # mode comparison uses a greedy run that was already made above.
    modes_differ = prove_modes_differ(greedy_runs, sampled_runs)
    print(f"      mismatches={sampled_verdict['field_level_mismatches']} "
          f"digests={sampled_verdict['distinct_results_digests']} "
          f"distinct from greedy = {modes_differ['stochastic_path_is_distinct']}")

    # -- 7. failure behaviour ------------------------------------------------
    failures = run_failure_suite(owner, greedy_matches, bank, greedy_ref, tmp_directory)
    print(f"[7/9] failure suite: {failures['summary']['passed']}/"
          f"{failures['summary']['cases']} passed, "
          f"substituted a move = {failures['summary']['any_substituted_move']}")

    owner_stats = owner.stats()
    owner_identity = owner.identity()
    owner.close()

    # -- 8. throughput -------------------------------------------------------
    benchmark_matches = (
        greedy_matches if options.quick else greedy_matches[: len(greedy_matches) // 4]
    )
    print(f"[8/9] throughput instruments on {len(benchmark_matches)} matches")
    precision = measure_precision_and_batching(benchmark_matches, bank, options.quick)

    memory = {
        "process_rss_bytes": process_memory_bytes(),
        "process_rss_after_greedy_bytes": memory_after_greedy["process_rss_bytes"],
        "metal": metal_memory_bytes(),
        "metal_after_greedy": memory_after_greedy["metal"],
        "checkpoint_bytes": checkpoint["checkpoint_bytes"],
        "note": (
            "RSS is the parent (the only process holding Metal). Game workers are "
            "pure engine processes; their cost is the CPU time in the sweep CSV."
        ),
    }

    # -- 9. tests ------------------------------------------------------------
    tests_after = {"skipped": True} if options.skip_pytest else run_pytest()
    if not options.skip_pytest:
        print(f"[9/9] suite after edits: {tests_after['summary_line']}")
    else:
        print("[9/9] pytest skipped by request")

    # -- gates ---------------------------------------------------------------
    gates = {
        "agents_1_to_4_pass_verified": prerequisites["all_pass"],
        "stable_finalist_checkpoint_loads_under_v2": (
            checkpoint["loads_under_v2"]
            and checkpoint["model_contract_version"] == "model_contract_v2"
            and prerequisites["candidate_is_a_finalist"]
        ),
        "checkpoint_loaded_once_per_long_lived_owner": (
            owner_identity["checkpoint_load_count"] == 1
            and sampled_load_count == 1
            and all(row["worker_checkpoint_loads"] == 0 for row in greedy_rows + sampled_rows)
        ),
        "mps_ownership_topology_safe_and_documented": (
            owner_identity["device"] == "mps"
            and all(row["workers_importing_torch"] == 0 for row in greedy_rows + sampled_rows)
        ),
        "phase_4_identities_and_seeds_unchanged": (
            greedy_verdict["per_field_mismatches"]["match_id"] == 0
            and greedy_verdict["per_field_mismatches"]["paired_unit_id"] == 0
            and greedy_verdict["per_field_mismatches"]["candidate_seed"] == 0
            and greedy_verdict["per_field_mismatches"]["opponent_seed"] == 0
            and greedy_ref.policy_id not in ALL_POLICY_IDS
            and sampled_ref.policy_id not in ALL_POLICY_IDS
            and len(ALL_POLICY_IDS) == 10
        ),
        "observer_safe_payload_only": (
            information_safety["object_graph_clean"]
            and not information_safety["arrays_alias_engine_memory"]
            and not information_safety["pickle_names_privileged_class"]
            and information_safety["fields_match_declaration"]
        ),
        "greedy_sweep_has_zero_mismatches": greedy_verdict["field_level_mismatches"] == 0,
        "one_results_digest_and_one_replay_digest_set": (
            greedy_verdict["distinct_results_digests"] == 1
            and greedy_verdict["distinct_replay_digest_sets"] == 1
        ),
        "seeded_categorical_mode_reproduces": (
            sampled_verdict["field_level_mismatches"] == 0
            and sampled_verdict["distinct_results_digests"] == 1
            and modes_differ["stochastic_path_is_distinct"]
        ),
        "failures_are_loud_and_never_substitute_a_move": (
            failures["summary"]["failed"] == []
            and not failures["summary"]["any_substituted_move"]
        ),
        "throughput_and_load_overhead_measured": bool(precision) and bool(greedy_rows),
        "full_suite_green": options.skip_pytest or tests_after.get("failed") == 0,
        "remote_path_matches_the_serial_adapter": adapter_cross_check["identical"],
        "stored_histories_replay_through_the_engine": replay_problems == [],
        "no_random_weight_strength_used": True,
    }
    status = "PASS" if all(gates.values()) else "FAIL"

    total_seconds = time.perf_counter() - started
    document = {
        "agent": AGENT,
        "phase": PHASE,
        "status": status,
        "schema_version": SCHEMA_VERSION,
        "neural_worker_version": NEURAL_WORKER_VERSION,
        "match_runner_version": MATCH_RUNNER_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": git_commit(),
        "quick_mode": options.quick,
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "platform_full": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "cpu_count": os.cpu_count(),
        **environment,
        "prerequisite_status": prerequisites,
        "frozen_versions": {
            "rules_version": RULES_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "evaluation_suite_version": EVALUATION_SUITE_VERSION,
            "setup_bank_version": SETUP_BANK_VERSION,
            "pairing_mode": PAIRING_COLOR_SWAP_SAME_BOARD,
            "model_contract_version": checkpoint["model_contract_version"],
            "policy_action_frame": checkpoint["policy_action_frame"],
            "engine_action_frame": checkpoint["engine_action_frame"],
        },
        "topology": {
            "description": (
                "N CPU game workers -> observer-safe inference requests -> one "
                "long-lived Metal inference owner -> checkpoint loaded once -> "
                "deterministic selection -> absolute action -> the worker's engine "
                "validates and applies it"
            ),
            "inference_owner_processes": 1,
            "game_worker_processes": list(greedy_counts),
            "start_method": "spawn",
            "device_owner": "the parent process only",
            "worker_device": "cpu (pure engine/NumPy; no torch import)",
            "transport": "one shared request queue, one private response queue per worker",
            "batch_policy": BATCH_POLICY_SINGLE,
            "batch_policy_rationale": (
                "under single_request the model input for a decision is a pure "
                "function of that decision's request, so worker count, chunking, "
                "arrival timing and schedule order cannot change the logits. "
                "Approximate float batch equivalence is not assumed anywhere."
            ),
            "request_ordering": "canonical (match_id, ply, acting_player, request_id)",
            "mps_safety": (
                "spawn rather than fork, so no child inherits a Metal context; the "
                "owner holds the only model instance and the only device allocation"
            ),
        },
        "checkpoint": checkpoint,
        "model_contract": {
            "model_contract_version": checkpoint["model_contract_version"],
            "policy_action_frame": checkpoint["policy_action_frame"],
            "engine_action_frame": checkpoint["engine_action_frame"],
            "observation_version": checkpoint["observation_version"],
            "action_encoding_version": checkpoint["action_encoding_version"],
            "decision_rules_module": "stratego.model.policy_adapter",
            "frame_converter_module": "stratego.model.action_frame",
            "second_implementation": "none: the owner calls prepare_legality and "
            "select_action from the adapter rather than reimplementing them",
        },
        "inference_owner": owner_identity,
        "checkpoint_load_counts": {
            "greedy_owner": owner_identity["checkpoint_load_count"],
            "sampled_owner": sampled_load_count,
            "process_total": checkpoint_load_count(),
            "in_game_workers": sum(
                row["worker_checkpoint_loads"] for row in greedy_rows + sampled_rows
            ),
            "loads_per_game": 0,
            "loads_per_move": 0,
            "greedy_runs_served_by_one_owner": len(greedy_runs),
            "greedy_decisions_served_by_one_owner": sum(
                run.decisions for run in greedy_runs.values()
            ),
        },
        "observer_safe_payload": information_safety,
        "greedy_worker_sweep": greedy_rows,
        "greedy_results_digests": greedy_verdict["results_digests"],
        "greedy_replay_digests": greedy_verdict["replay_digest_set_digests"],
        "greedy_field_mismatches": greedy_verdict["per_field_mismatches"],
        "greedy_verdict": greedy_verdict,
        "adapter_cross_check": adapter_cross_check,
        "engine_replay_of_stored_histories": {
            "rows_replayed": len(sample),
            "problems": replay_problems,
        },
        "seeded_worker_sweep": sampled_rows,
        "seeded_reproducibility": {**sampled_verdict, "modes_differ": modes_differ},
        "failure_tests": failures,
        "throughput": {
            "gate_runs": greedy_rows,
            "instruments": precision,
            "batch_invariance_probe": batch_invariance,
            "inference_latency_note": (
                "the owner is serial by construction under single_request, so "
                "decisions/second is bounded by one batch-1 forward pass; adding "
                "game workers past that point buys nothing"
            ),
            "owner_totals": owner_stats,
        },
        "memory": memory,
        "information_safety": {
            "object_graph_clean": information_safety["object_graph_clean"],
            "forbidden_types_reachable": information_safety["forbidden_types_reachable"],
            "workers_importing_torch": sum(
                row["workers_importing_torch"] for row in greedy_rows + sampled_rows
            ),
            "worker_checkpoint_loads": sum(
                row["worker_checkpoint_loads"] for row in greedy_rows + sampled_rows
            ),
            "privileged_state_required": False,
            "phase_4_catalogue_size": len(ALL_POLICY_IDS),
            "neural_policies_in_catalogue": [
                ref.policy_id
                for ref in (greedy_ref, sampled_ref)
                if ref.policy_id in ALL_POLICY_IDS
            ],
        },
        "tests_before": tests_before,
        "tests_after": tests_after,
        "test_total": tests_after.get("total"),
        "test_passed": tests_after.get("passed"),
        "test_failed": tests_after.get("failed"),
        "test_skipped": tests_after.get("skipped"),
        "seeds": {
            "family_initialisation_seed": FAMILY_SEED,
            "schedule_shuffle_seed": 20260512,
            "bootstrap_resamples": 500,
            "policy_and_decision_seeds": "unchanged: derived from match_id by "
            "stratego.evaluation.match_spec and stratego.evaluation.policy",
        },
        "commands": [
            "python scripts/run_phase6_agent05.py",
            "python -m pytest -q",
            "python -m pytest -q tests/evaluation/test_parallel_neural_checkpoint.py",
        ],
        "durations": {
            "total_seconds": round(total_seconds, 2),
            "checkpoint_load_seconds": owner_identity["checkpoint_load_seconds"],
        },
        "total_seconds": round(total_seconds, 2),
        "files_created": [
            "stratego/evaluation/neural_worker.py",
            "scripts/run_phase6_agent05.py",
            "tests/evaluation/test_parallel_neural_checkpoint.py",
            "checkpoints/phase6_c1.pt",
            "reports/phase_6_data/agent_05_parallel_neural_evaluation.json",
            "reports/phase_6_data/agent_05_greedy_worker_sweep.csv",
            "reports/phase_6_data/agent_05_throughput.csv",
            "reports/phase_6_data/agent_05_failure_cases.json",
        ],
        "files_modified": [
            "stratego/evaluation/__init__.py",
            "stratego/model/policy_adapter.py",
            "reports/phase_6_implementation_report.md",
        ],
        "completion_gates": gates,
        "problems": [name for name, value in gates.items() if not value],
        "handoff_to_agent_06": {
            "api": "stratego.evaluation.neural_worker.run_neural_schedule(matches, "
            "bank, owner, worker_count=...)",
            "owner": "InferenceOwner(checkpoint, device='mps', dtype='float32', "
            "decision_mode=..., expected_configuration=candidate_config(id))",
            "policy_identity": "neural_policy_ref(candidate_id, decision_mode=..., "
            "dtype_name=...); deliberately not in the Phase 4 catalogue",
            "measured_ceiling": (
                "one batch-1 forward per decision on Metal; see throughput.instruments"
            ),
            "evaluation_does_not_need_redesigning": True,
        },
    }

    write_json(DATA_DIRECTORY / "agent_05_parallel_neural_evaluation.json", document)
    write_json(DATA_DIRECTORY / "agent_05_failure_cases.json", failures)
    _write_csv(DATA_DIRECTORY / "agent_05_greedy_worker_sweep.csv", greedy_rows + sampled_rows)
    _write_csv(
        DATA_DIRECTORY / "agent_05_throughput.csv",
        [
            {"instrument": name, **{k: v for k, v in entry.items() if not isinstance(v, (dict, list))}}
            for name, entry in precision.items()
        ],
    )

    for leftover in tmp_directory.glob("*"):
        leftover.unlink()
    tmp_directory.rmdir()

    print(f"\nstatus: {status}   ({total_seconds:.1f}s)")
    for name, value in gates.items():
        print(f"  {'PASS' if value else 'FAIL'}  {name}")
    return 0 if status == "PASS" else 1


def _write_csv(path: Path, rows) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


if __name__ == "__main__":
    raise SystemExit(main())
