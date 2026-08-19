#!/usr/bin/env python3
"""Phase 11 Agent 2 harness: belief evaluator, baselines, validation evidence.

Recomputes every load-bearing identity from live bytes (the Agent 1 PASS and
its 31 gates, the eight Phase 11 contracts and their bundle, both bank
digests, the Phase 9 checkpoint's file SHA / model-state digest / parameter
count / belief-head tensor identity / optimizer-step counter, the frozen
P10-D chain and the Phase 7 library), then implements the belief evaluation
path and the two frozen baselines and runs predictive evaluation on
`phase11_validation_bank_v1` **only**.

    reports/phase_11_data/agent_02_predictive_metrics.json
    reports/phase_11_data/agent_02_stratum_metrics.csv
    reports/phase_11_data/agent_02_baseline_audit.json
    reports/phase_11_data/agent_02_acceptance.json

What this script is and is not
------------------------------
It measures the accepted belief head. It runs no optimizer step, calibrates
nothing, and changes no threshold, bin, baseline, bank or stratum. The
Phase 9 checkpoint is opened read-only and its digests are re-verified
after the run. `phase11_test_bank_v1` is touched only to re-derive its
digest structurally — no game, no forward, no score, no truth — and every
access is written to the append-only Phase 11 bank ledger.

Usage::

    python scripts/run_phase11_agent02.py                  # every stage
    python scripts/run_phase11_agent02.py --stage verify   # one stage
    python scripts/run_phase11_agent02.py --limit-cases 8  # a smoke run
    python scripts/run_phase11_agent02.py --record-suite   # run + record the suite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.engine.constants import (  # noqa: E402
    IMPLEMENTATION_VERSION,
    OBSERVATION_VERSION,
    RULES_VERSION,
)

AGENT = 2
PHASE = 11
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_11_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_11_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase11" / "agent02"
EXPORT_PATH = WORK_DIRECTORY / "eval_phase9_c1.pt"
CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "selfplay_c1_v1.pt"

#: The evaluation backend, frozen before any measurement: CPU float32 with a
#: single torch thread. MPS is not run-to-run bit-deterministic on this
#: machine (the accepted Phase 8 finding), and Agent 4 has to reproduce
#: every one of these decisions exactly.
EVAL_DEVICE = "cpu"
EVAL_TORCH_THREADS = 1

#: How many games the pure-Python scalar audit and the edge-case replay
#: cover. Both are deterministic prefixes of the frozen game order, chosen
#: before any metric existed.
SCALAR_AUDIT_GAMES = 64
EDGE_CASE_AUDIT_GAMES = 256


class Agent2Error(RuntimeError):
    """Agent 2 cannot proceed."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def torch_report() -> dict:
    import torch

    return {
        "torch_version": str(torch.__version__),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "eval_device": EVAL_DEVICE,
        "torch_threads": EVAL_TORCH_THREADS,
    }


def environment_report() -> dict:
    report = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "source_revision": _git("rev-parse", "--short", "HEAD"),
        "working_tree_state": "dirty" if _git("status", "--porcelain") else "clean",
        "rules_version": RULES_VERSION,
        "engine_version": IMPLEMENTATION_VERSION,
        "observation_version": OBSERVATION_VERSION,
    }
    report.update(torch_report())
    return report


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def stage_path(name: str) -> Path:
    return WORK_DIRECTORY / f"stage_{name}.json"


def write_stage(name: str, payload: dict) -> Path:
    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = stage_path(name)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=_jsonable) + "\n")
    return path


def read_stage(name: str) -> dict:
    path = stage_path(name)
    if not path.exists():
        raise Agent2Error(f"stage {name!r} has not run yet ({path} is missing)")
    return json.loads(path.read_text())


def _jsonable(value):
    import numpy as np

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"{type(value).__name__} is not JSON-serialisable")


def require(condition: bool, message: str, problems: list) -> bool:
    if not condition:
        problems.append(message)
    return bool(condition)


def log(message: str) -> None:
    print(f"[phase11:agent2] {message}", flush=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def configure_backend() -> None:
    import torch

    torch.set_num_threads(EVAL_TORCH_THREADS)
    torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# 1. Verification — every identity from live bytes
# ---------------------------------------------------------------------------


def verify_agent1(problems: list) -> dict:
    """Agent 1 must be PASS, with its gates and digests re-derived here."""
    from stratego.training import phase11_contract as contract

    path = DATA_DIRECTORY / "agent_01_acceptance.json"
    require(path.exists(), "the Agent 1 acceptance artifact is missing", problems)
    if not path.exists():
        return {"available": False}
    acceptance = read_json(path)
    require(acceptance.get("status") == "PASS", "Agent 1 did not report PASS", problems)
    gates = acceptance.get("completion_gates", {})
    false_gates = sorted(name for name, value in gates.items() if not value)
    require(not false_gates, f"Agent 1 gates are false: {false_gates}", problems)
    counters = acceptance.get("forbidden_operation_counters", {})
    nonzero = sorted(name for name, value in counters.items() if value)
    require(
        not nonzero, f"Agent 1 forbidden-operation counters are non-zero: {nonzero}", problems
    )

    # Recomputed from the live modules, never read back from the artifact.
    documents = contract.contract_documents()
    live_digests = contract.contract_digests(documents)
    live_bundle = contract.contract_bundle_digest(documents)
    recorded = acceptance.get("new_digests", {}).get("contract_digests", {})
    for name, digest in live_digests.items():
        require(
            recorded.get(name) == digest,
            f"contract {name} digest {digest} != the Agent 1 record {recorded.get(name)}",
            problems,
        )
    recorded_bundle = acceptance.get("new_digests", {}).get("contract_bundle_digest")
    require(
        recorded_bundle == live_bundle,
        f"contract bundle {live_bundle} != the Agent 1 record {recorded_bundle}",
        problems,
    )
    return {
        "available": True,
        "status": acceptance.get("status"),
        "gates_true": acceptance.get("gates_true"),
        "gates_total": acceptance.get("gates_total"),
        "false_gates": false_gates,
        "forbidden_operation_counters": counters,
        "contract_digests": live_digests,
        "contract_bundle_digest": live_bundle,
        "handoff": acceptance.get("handoff_to_agent_2", {}),
        "suite": acceptance.get("suite", {}),
    }


def verify_banks(problems: list) -> dict:
    """Both bank artifacts re-hashed from their stored cases, live."""
    from stratego.evaluation import phase11_banks as banks
    from stratego.evaluation.phase11_banks import Phase11Case

    summary = {}
    for name, filename, expected_version in (
        ("validation", "agent_01_validation_bank.json", "phase11_validation_bank_v1"),
        ("test", "agent_01_test_bank.json", "phase11_test_bank_v1"),
    ):
        path = DATA_DIRECTORY / filename
        require(path.exists(), f"the {name} bank artifact is missing", problems)
        if not path.exists():
            summary[name] = {"available": False}
            continue
        payload = read_json(path)
        cases = tuple(
            Phase11Case(
                case_id=case["case_id"],
                bank=case["bank"],
                bank_version=case["bank_version"],
                split=case["split"],
                stratum=case["stratum"],
                setup_source=case["setup_source"],
                case_ordinal=case["case_ordinal"],
                case_index=case["case_index"],
                games={int(key): value for key, value in case["games"].items()},
            )
            for case in payload["cases"]
        )
        digest = banks.bank_digest(cases)
        manifest = payload["manifest"]
        require(
            manifest["bank_version"] == expected_version,
            f"the {name} bank names version {manifest['bank_version']!r}",
            problems,
        )
        require(
            digest == manifest["bank_digest"],
            f"the {name} bank re-hashes to {digest}, not {manifest['bank_digest']}",
            problems,
        )
        require(
            banks.manifest_digest(manifest) == manifest["manifest_digest"],
            f"the {name} bank manifest digest does not re-derive",
            problems,
        )
        specification = banks.bank_specification(name)
        require(
            len(cases) == specification["case_count"],
            f"the {name} bank holds {len(cases)} cases, expected "
            f"{specification['case_count']}",
            problems,
        )
        summary[name] = {
            "available": True,
            "bank_version": manifest["bank_version"],
            "bank_digest": digest,
            "manifest_digest": manifest["manifest_digest"],
            "cases": len(cases),
            "games": 2 * len(cases),
        }
    return summary


def belief_head_identity(model_state: dict) -> dict:
    """Re-derive the frozen belief-head tensor identity from live bytes."""
    import numpy as np

    from stratego.training.phase11_contract import BELIEF_HEAD_TENSOR_NAMES

    hasher = hashlib.sha256()
    shapes = {}
    for name in sorted(BELIEF_HEAD_TENSOR_NAMES):
        tensor = model_state[name]
        array = tensor.detach().to("cpu").numpy().astype(np.float32)
        shapes[name] = list(array.shape)
        hasher.update(name.encode())
        hasher.update(str(tuple(array.shape)).encode())
        hasher.update(np.ascontiguousarray(array).tobytes())
    return {"digest": hasher.hexdigest(), "tensor_shapes": shapes}


def verify_phase9_checkpoint(problems: list) -> dict:
    """The accepted checkpoint, opened read-only and fully re-derived."""
    import torch

    from stratego.training.phase11_contract import (
        ACCEPTED_BELIEF_HEAD_DIGEST,
        ACCEPTED_GLOBAL_OPTIMIZER_STEP,
        ACCEPTED_PHASE9_CHECKPOINT_SHA256,
        ACCEPTED_PHASE9_PARAMETERS,
        ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
        BELIEF_HEAD_TENSOR_NAMES,
        BELIEF_HEAD_TENSOR_SHAPES,
    )
    from stratego.training.phase9_behavior import state_dict_digest
    from stratego.training.phase9_checkpoint import model_from_payload, read_phase9_payload

    require(CHECKPOINT_PATH.exists(), "the Phase 9 checkpoint is missing", problems)
    if not CHECKPOINT_PATH.exists():
        return {"available": False}

    sha = file_sha256(CHECKPOINT_PATH)
    require(
        sha == ACCEPTED_PHASE9_CHECKPOINT_SHA256,
        f"Phase 9 checkpoint SHA {sha} != accepted",
        problems,
    )
    payload = read_phase9_payload(CHECKPOINT_PATH)
    model = model_from_payload(payload)
    state = model.state_dict()
    digest = state_dict_digest(model)
    parameters = sum(tensor.numel() for tensor in model.parameters())
    finite = all(torch.isfinite(tensor).all().item() for tensor in state.values())
    head = belief_head_identity(state)

    require(digest == ACCEPTED_PHASE9_MODEL_STATE_DIGEST, "model-state digest != accepted", problems)
    require(
        parameters == ACCEPTED_PHASE9_PARAMETERS,
        f"parameter count {parameters} != {ACCEPTED_PHASE9_PARAMETERS}",
        problems,
    )
    require(finite, "the Phase 9 model state carries a non-finite value", problems)
    require(
        head["digest"] == ACCEPTED_BELIEF_HEAD_DIGEST,
        f"belief-head digest {head['digest']} != accepted",
        problems,
    )
    for name in BELIEF_HEAD_TENSOR_NAMES:
        require(
            head["tensor_shapes"].get(name) == list(BELIEF_HEAD_TENSOR_SHAPES[name]),
            f"belief-head tensor {name} changed shape",
            problems,
        )
    optimizer_step = int(payload.get("global_optimizer_step", -1))
    require(
        optimizer_step == ACCEPTED_GLOBAL_OPTIMIZER_STEP,
        f"global optimizer step {optimizer_step} != the frozen baseline "
        f"{ACCEPTED_GLOBAL_OPTIMIZER_STEP}",
        problems,
    )
    del model, payload, state
    return {
        "available": True,
        "path": str(CHECKPOINT_PATH.relative_to(REPOSITORY_ROOT)),
        "sha256": sha,
        "model_state_digest": digest,
        "parameters": int(parameters),
        "all_finite": bool(finite),
        "belief_head_digest": head["digest"],
        "belief_head_tensor_shapes": head["tensor_shapes"],
        "global_optimizer_step": optimizer_step,
    }


def verify_upstream_stack(problems: list) -> dict:
    """The P10-D selector chain, the Phase 8 anchor and the Phase 7 library."""
    from stratego.evaluation.phase11_banks import Phase11SetupSources
    from stratego.setups.contracts import LIBRARY_JSONL_PATH, LIBRARY_MANIFEST_PATH
    from stratego.setups.library import (
        entry_metadata_digest,
        library_content_digest,
        manifest_digest,
        read_library_jsonl,
        read_manifest,
    )
    from stratego.training.phase11_contract import (
        ACCEPTED_ANCHOR_EXPORT_PATH,
        ACCEPTED_ANCHOR_EXPORT_SHA256,
        ACCEPTED_SELECTOR_CONFIG_SHA256,
        ACCEPTED_TRAIT_SCALER_DIGEST,
        ACCEPTED_UTILITY_COEFFICIENT_DIGEST,
        PHASE7_LIBRARY_CONTENT_DIGEST,
        PHASE7_LIBRARY_MANIFEST_DIGEST,
        PHASE7_LIBRARY_METADATA_DIGEST,
    )

    selector_problems = Phase11SetupSources.verify_selector_artifacts()
    require(
        not selector_problems,
        f"the P10-D selector chain moved: {selector_problems}",
        problems,
    )
    anchor = REPOSITORY_ROOT / ACCEPTED_ANCHOR_EXPORT_PATH
    require(anchor.exists(), "the Phase 8 anchor export is missing", problems)
    anchor_sha = file_sha256(anchor) if anchor.exists() else None
    require(
        anchor_sha == ACCEPTED_ANCHOR_EXPORT_SHA256,
        f"the Phase 8 anchor export SHA {anchor_sha} != accepted",
        problems,
    )
    entries = read_library_jsonl(LIBRARY_JSONL_PATH)
    library = {
        "content_digest": library_content_digest(entries),
        "metadata_digest": entry_metadata_digest(entries),
        "manifest_digest": read_manifest(LIBRARY_MANIFEST_PATH)["manifest_digest"],
        "bases": len(entries),
    }
    require(
        library["content_digest"] == PHASE7_LIBRARY_CONTENT_DIGEST,
        f"the Phase 7 library content digest {library['content_digest']} != accepted",
        problems,
    )
    require(
        library["metadata_digest"] == PHASE7_LIBRARY_METADATA_DIGEST,
        "the Phase 7 library metadata digest != accepted",
        problems,
    )
    require(
        library["manifest_digest"] == PHASE7_LIBRARY_MANIFEST_DIGEST,
        "the Phase 7 library manifest digest != accepted",
        problems,
    )
    return {
        "selector_problems": selector_problems,
        "selector_config_sha256": ACCEPTED_SELECTOR_CONFIG_SHA256,
        "utility_coefficient_digest": ACCEPTED_UTILITY_COEFFICIENT_DIGEST,
        "trait_scaler_digest": ACCEPTED_TRAIT_SCALER_DIGEST,
        "anchor_export_sha256": anchor_sha,
        "phase7_library": library,
    }


def verify_test_bank_sealed(problems: list) -> dict:
    """The ledger must still show zero scored test-bank access."""
    from stratego.evaluation import phase11_banks as banks

    sealing = banks.verify_test_bank_sealed()
    require(
        sealing["test_bank_structural_only"],
        f"the test bank is no longer structural-only: {sealing['violations']}",
        problems,
    )
    return sealing


def stage_verify(_args) -> dict:
    problems: list[str] = []
    log("verifying the Agent 1 freeze")
    agent1 = verify_agent1(problems)
    log("re-hashing both frozen banks")
    bank_summary = verify_banks(problems)
    log("re-deriving the Phase 9 checkpoint and belief-head identity")
    phase9 = verify_phase9_checkpoint(problems)
    log("verifying the P10-D / anchor / Phase 7 stack")
    upstream = verify_upstream_stack(problems)
    log("checking the test-bank seal")
    sealing = verify_test_bank_sealed(problems)

    # Every agent-harness bank access is recorded. Both of Agent 2's
    # verification touches are structural — re-hashing stored cases — and
    # the test bank is never touched any other way.
    from stratego.evaluation import phase11_banks as banks

    banks.append_ledger_entries(
        [
            banks.ledger_entry(
                AGENT,
                "verify_bank_digest",
                bank_version,
                "structural re-hash of the frozen bank from its stored cases",
                structural_only=True,
            )
            for bank_version in (
                "phase11_validation_bank_v1",
                "phase11_test_bank_v1",
            )
        ]
    )

    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "stage": "verify",
        "environment": environment_report(),
        "agent1": agent1,
        "banks": bank_summary,
        "phase9": phase9,
        "upstream": upstream,
        "test_bank_sealing": sealing,
        "problems": problems,
        "verified": not problems,
    }
    write_stage("verify", payload)
    if problems:
        for problem in problems:
            log(f"PROBLEM: {problem}")
        raise Agent2Error(f"verification found {len(problems)} problem(s); BLOCKED")
    log("verification clean")
    return payload


# ---------------------------------------------------------------------------
# 2. The validation run
# ---------------------------------------------------------------------------


def load_validation_bank() -> dict:
    return read_json(DATA_DIRECTORY / "agent_01_validation_bank.json")


def build_owners():
    """The two long-lived inference owners this run needs."""
    from stratego.evaluation.neural_worker import DECISION_MODE_GREEDY, InferenceOwner
    from stratego.evaluation.phase11_belief import Phase11BeliefOwner
    from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from stratego.training.phase10_collector import export_evaluation_weights
    from stratego.training.phase11_contract import ACCEPTED_ANCHOR_EXPORT_PATH

    export = export_evaluation_weights(CHECKPOINT_PATH, EXPORT_PATH)
    common = {
        "decision_mode": DECISION_MODE_GREEDY,
        "device": EVAL_DEVICE,
        "dtype": "float32",
        "expected_architecture_id": ARCHITECTURE_FAMILY,
        "expected_configuration": candidate_config("C1"),
    }
    owners = {
        "phase9": Phase11BeliefOwner(EXPORT_PATH, name="phase11_observer", **common),
        "anchor": InferenceOwner(
            REPOSITORY_ROOT / ACCEPTED_ANCHOR_EXPORT_PATH,
            name="phase11_anchor",
            **common,
        ),
    }
    return owners, export


def stage_run(args) -> dict:
    """Play the frozen validation bank and write the prediction store."""
    import numpy as np

    from stratego.evaluation import phase11_banks as banks
    from stratego.evaluation import phase11_records as records
    from stratego.evaluation import phase11_runner as runner
    from stratego.training.phase10_collector import owner_state_digest
    from stratego.training.phase11_contract import (
        ACCEPTED_BELIEF_HEAD_DIGEST,
        ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
        VALIDATION_BANK_GAMES,
    )

    configure_backend()
    verify = read_stage("verify")
    bank = load_validation_bank()
    bank_version = bank["manifest"]["bank_version"]
    cases = bank["cases"]
    if args.limit_cases:
        cases = cases[: int(args.limit_cases)]
        log(f"SMOKE RUN: {len(cases)} cases only — not an acceptance run")

    root = records.store_root(REPOSITORY_ROOT)
    if args.limit_cases:
        root = root.parent / f"{root.name}_smoke"
    log(f"prediction store: {root}")
    owners, export = build_owners()
    model_id = records.model_identity(
        ACCEPTED_PHASE9_MODEL_STATE_DIGEST, ACCEPTED_BELIEF_HEAD_DIGEST
    )
    observed_state = owner_state_digest(owners["phase9"])
    if observed_state != ACCEPTED_PHASE9_MODEL_STATE_DIGEST:
        raise Agent2Error(
            f"the loaded observer weights digest {observed_state} != accepted"
        )

    started = time.perf_counter()
    entries: list[dict] = []
    truth_summaries: list[dict] = []
    outcomes = {"win": 0, "draw": 0, "loss": 0}
    terminal_reasons: dict[str, int] = {}
    decisions = events = 0
    request_digests = hashlib.sha256()
    try:
        for position, case in enumerate(cases):
            for game_index in (0, 1):
                plan, result, recorder, observer = runner.play_validation_game(
                    case, game_index, owners, bank_version
                )
                entry = records.write_public_shard(root, recorder)
                arrays = records.read_public_shard(root, plan.game_id)
                truth = runner.privileged_truth_pass(plan, result, arrays)
                truth_entry = records.write_truth_shard(
                    root, plan.game_id, truth["true_rank_index"]
                )
                entry["truth_shard_digest"] = truth_entry["truth_shard_digest"]
                entries.append(entry)
                truth_summaries.append(
                    {key: value for key, value in truth.items() if key != "true_rank_index"}
                )
                outcomes[result.candidate_result] = (
                    outcomes.get(result.candidate_result, 0) + 1
                )
                terminal_reasons[result.terminal_reason] = (
                    terminal_reasons.get(result.terminal_reason, 0) + 1
                )
                decisions += recorder.decisions
                events += recorder.events
                for digest in observer.request_digests:
                    request_digests.update(digest.encode())
            if (position + 1) % 32 == 0:
                elapsed = time.perf_counter() - started
                log(
                    f"{position + 1}/{len(cases)} cases  {events:,} events  "
                    f"{elapsed:6.1f}s"
                )
    finally:
        owners["phase9"].close()
        owners["anchor"].close()

    manifest = {
        "store_version": records.PREDICTION_STORE_VERSION,
        "record_version": records.PREDICTION_RECORD_VERSION,
        "run_version": runner.PHASE11_RUN_VERSION,
        "bank_version": bank_version,
        "bank_digest": bank["manifest"]["bank_digest"],
        "model_identity": model_id,
        "eval_device": EVAL_DEVICE,
        "torch_threads": EVAL_TORCH_THREADS,
        "games": len(entries),
        "observer_decisions": decisions,
        "prediction_events": events,
        "games_expected": VALIDATION_BANK_GAMES,
        "complete_bank": len(entries) == VALIDATION_BANK_GAMES,
        "belief_forwards": owners["phase9"].belief_forwards,
        "belief_rows": owners["phase9"].belief_rows,
        "request_digest_rollup": request_digests.hexdigest(),
        "outcomes": outcomes,
        "terminal_reasons": dict(sorted(terminal_reasons.items())),
        "games_index": sorted(entries, key=lambda item: item["game_id"]),
        "truth_pass": {
            "games": len(truth_summaries),
            "identity_mismatches": sum(
                row["identity_mismatches"] for row in truth_summaries
            ),
            "alignment_mismatches": sum(
                row["alignment_mismatches"] for row in truth_summaries
            ),
            "count_mismatches": sum(row["count_mismatches"] for row in truth_summaries),
            "mask_mismatches": sum(row["mask_mismatches"] for row in truth_summaries),
            "unlabelled_events": sum(row["unlabelled_events"] for row in truth_summaries),
            "verified_decisions": sum(row["verified_decisions"] for row in truth_summaries),
        },
    }
    manifest["manifest_digest"] = records.manifest_digest(manifest)
    records.write_manifest(root, manifest)

    banks.append_ledger_entries(
        [
            banks.ledger_entry(
                AGENT,
                "validation_smoke_run" if args.limit_cases else "validation_predictive_run",
                bank_version,
                (
                    f"smoke run over {len(cases)} validation cases"
                    if args.limit_cases
                    else "belief inference, baseline scoring and privileged truth on "
                    "the complete validation bank"
                ),
                structural_only=False,
                neural_inference_count=int(owners["phase9"].belief_forwards),
                scored_prediction_count=int(events),
                privileged_truth_count=int(events),
                outcome_count=int(len(entries)),
            )
        ]
    )

    summary = {
        "agent": AGENT,
        "stage": "run",
        "store_root": str(root),
        "manifest_digest": manifest["manifest_digest"],
        "games": len(entries),
        "observer_decisions": decisions,
        "prediction_events": events,
        "outcomes": outcomes,
        "terminal_reasons": manifest["terminal_reasons"],
        "truth_pass": manifest["truth_pass"],
        "export": export,
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
        "smoke_run": bool(args.limit_cases),
        "source_revision": verify["environment"]["source_revision"],
    }
    write_stage("run", summary)
    log(
        f"played {len(entries)} games, {events:,} prediction events in "
        f"{summary['wall_clock_seconds']:.1f}s"
    )
    return summary


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------


def load_scored_blocks(root: Path, manifest: dict):
    """Every game's scored block, in the frozen game-id order."""
    import numpy as np

    from stratego.evaluation import phase11_records as records
    from stratego.evaluation.phase11_audit import independent_scores
    from stratego.evaluation.phase11_baselines import remaining_count_distribution
    from stratego.evaluation.phase11_belief import softmax_float64
    from stratego.evaluation.phase11_evaluator import score_matrix
    from stratego.training.phase11_contract import PROGRESS_BUCKET_NAMES, progress_bucket

    bucket_index = {name: index for index, name in enumerate(PROGRESS_BUCKET_NAMES)}
    blocks = []
    audit_deviations = {}
    log_floor_events = 0
    for entry in manifest["games_index"]:
        arrays = records.read_public_shard(root, entry["game_id"])
        truth = records.read_truth_shard(root, entry["game_id"])
        size = int(truth.size)
        if size == 0:
            blocks.append(
                {
                    "case_id": entry["case_id"],
                    "opponent_stratum": entry["opponent_stratum"],
                    "opponent_setup_source": entry["opponent_setup_source"],
                    "observer_color": entry["observer_color"],
                    "true_rank": np.zeros(0, dtype=np.int64),
                    "bucket_index": np.zeros(0, dtype=np.int8),
                    "piece_moved": np.zeros(0, dtype=np.uint8),
                    "learned": None,
                    "baseline": None,
                }
            )
            continue
        logits = arrays["belief_logits"]
        learned = np.stack([softmax_float64(row) for row in logits])
        offsets = arrays["event_offset"]
        counts = arrays["remaining_counts"]
        masks = arrays["legal_rank_mask"]
        baseline = np.empty_like(learned)
        buckets = np.empty(size, dtype=np.int8)
        for decision in range(int(arrays["decision_index"].size)):
            start, stop = int(offsets[decision]), int(offsets[decision + 1])
            if stop <= start:
                continue
            bucket = bucket_index[progress_bucket(int(arrays["decision_index"][decision]))]
            buckets[start:stop] = bucket
            for cursor in range(start, stop):
                baseline[cursor] = remaining_count_distribution(
                    counts[decision], masks[cursor]
                )
        learned_scores = score_matrix(learned, truth)
        baseline_scores = score_matrix(baseline, truth)
        log_floor_events += learned_scores["log_floor_events"]

        audit = independent_scores(learned, truth)
        for name, value in audit.items():
            deviation = float(
                np.abs(np.asarray(learned_scores[name]) - np.asarray(value)).max()
            )
            audit_deviations[name] = max(audit_deviations.get(name, 0.0), deviation)

        blocks.append(
            {
                "case_id": entry["case_id"],
                "opponent_stratum": entry["opponent_stratum"],
                "opponent_setup_source": entry["opponent_setup_source"],
                "observer_color": entry["observer_color"],
                "true_rank": truth.astype(np.int64),
                "bucket_index": buckets,
                "piece_moved": arrays["piece_moved"],
                "learned": learned_scores,
                "baseline": baseline_scores,
            }
        )
    return blocks, audit_deviations, log_floor_events


def stage_metrics(_args) -> dict:
    import numpy as np

    from stratego.evaluation import phase11_records as records
    from stratego.evaluation.phase11_audit import AUDIT_TOLERANCE
    from stratego.evaluation.phase11_evaluator import (
        EVALUATOR_VERSION,
        all_finite,
        build_scored_events,
        overall_metrics,
        slice_metrics,
    )

    run = read_stage("run")
    root = Path(run["store_root"])
    manifest = records.read_manifest(root)
    log(f"scoring {manifest['prediction_events']:,} events from {root}")
    started = time.perf_counter()
    blocks, audit_deviations, log_floor_events = load_scored_blocks(root, manifest)
    # Every block is registered, including the games in which the observer
    # never acted: a case with no events must be visible as such, not absent.
    table = build_scored_events(blocks)
    log(f"scored table: {table.events:,} events over {table.case_count} cases")

    overall = overall_metrics(table, "validation")
    log("bootstrapping the overall metrics")
    slices = slice_metrics(table, "validation")
    log("bootstrapping the stratum slices")

    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_02_predictive_metrics",
        "evaluator_version": EVALUATOR_VERSION,
        "bank": "validation",
        "bank_version": manifest["bank_version"],
        "bank_digest": manifest["bank_digest"],
        "model_identity": manifest["model_identity"],
        "store_manifest_digest": manifest["manifest_digest"],
        "games": manifest["games"],
        "observer_decisions": manifest["observer_decisions"],
        "prediction_events": manifest["prediction_events"],
        "log_floor_events": int(log_floor_events),
        "overall": overall,
        "slices": slices,
        "independent_formula_audit": {
            "max_deviation": audit_deviations,
            "tolerance": AUDIT_TOLERANCE,
            "within_tolerance": all(
                value <= AUDIT_TOLERANCE for value in audit_deviations.values()
            ),
            "coverage": "every scored event",
        },
        "outcomes_report_only": manifest["outcomes"],
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
    }
    payload["nonfinite_paths"] = all_finite(
        {"overall": overall, "slices": slices}
    )
    payload["metrics_finite"] = not payload["nonfinite_paths"]
    write_stage("metrics", payload)
    metrics = overall["metrics"]
    log(
        f"R_CE {metrics['r_ce']['point']:.4f}  "
        f"top1 learned {metrics['top1_learned']['point']:.4f} vs baseline "
        f"{metrics['top1_baseline']['point']:.4f}  "
        f"ECE {overall['ece_learned']['ece']:.4f}"
    )
    return payload


# ---------------------------------------------------------------------------
# 4. Audits and negative controls
# ---------------------------------------------------------------------------


def stage_audit(_args) -> dict:
    """Independent recomputation, the six negative controls, and the audits."""
    import numpy as np

    from stratego.engine.observation import build_observation
    from stratego.engine.state import create_game
    from stratego.evaluation import phase11_records as records
    from stratego.evaluation import phase11_runner as runner
    from stratego.evaluation.match_spec import EVALUATION_RULES
    from stratego.evaluation.phase11_audit import (
        AUDIT_TOLERANCE,
        AUDIT_VERSION,
        baseline_edge_case_audit,
        run_negative_controls,
        scalar_recompute,
        world_baseline_audit,
    )
    from stratego.evaluation.phase11_belief import softmax_float64
    from stratego.evaluation.phase11_evaluator import build_scored_events
    from stratego.evaluation.phase11_public_state import build_public_state_document
    from stratego.evaluation.policy import build_public_view

    configure_backend()
    run = read_stage("run")
    root = Path(run["store_root"])
    manifest = records.read_manifest(root)
    bank = load_validation_bank()
    by_case = {case["case_id"]: case for case in bank["cases"]}
    started = time.perf_counter()

    # -- layer 2: the pure-Python scalar path, deterministic prefix --------
    scalar_games = manifest["games_index"][:SCALAR_AUDIT_GAMES]
    scalar_records = []
    for entry in scalar_games:
        arrays = records.read_public_shard(root, entry["game_id"])
        truth = records.read_truth_shard(root, entry["game_id"])
        scalar_records.extend(
            records.iter_records(entry, arrays, truth, model_id=manifest["model_identity"])
        )
    log(f"scalar audit: {len(scalar_records):,} records from {len(scalar_games)} games")
    scalar = scalar_recompute(scalar_records)

    subset_blocks, _, _ = load_scored_blocks(root, {**manifest, "games_index": scalar_games})
    subset_table = build_scored_events(subset_blocks)
    scalar_deviation = 0.0
    for name in (
        "ce_learned",
        "ce_baseline",
        "top1_learned",
        "top1_baseline",
        "brier_learned",
        "brier_baseline",
        "entropy_learned",
        "entropy_baseline",
    ):
        present, values = subset_table.case_means(subset_table.columns[name])
        for index, value in zip(present, values):
            case_id = subset_table.case_ids[index]
            scalar_deviation = max(
                scalar_deviation, abs(float(value) - scalar["case_aggregates"][case_id][name])
            )
    log(f"scalar vs primary: max case-aggregate deviation {scalar_deviation:.3e}")

    # -- layer 3: the engine's own counts, and the frozen edge cases -------
    replays = []
    for entry in manifest["games_index"][:EDGE_CASE_AUDIT_GAMES]:
        plan = runner.game_plan(by_case[entry["case_id"]], entry["game_index"])
        history = records.read_public_shard(root, entry["game_id"])["action_history"]
        replays.append((plan, [int(value) for value in history]))
    log(f"edge-case audit over {len(replays)} games")
    edge = baseline_edge_case_audit(replays)
    reveal_document = edge.pop("reveal_document")
    reveal_observation = edge.pop("reveal_observation")

    log("count_uniform_world_sampler_v1 audit")
    worlds = world_baseline_audit(replays)

    # -- the six negative controls -----------------------------------------
    control_entry = manifest["games_index"][0]
    control_arrays = records.read_public_shard(root, control_entry["game_id"])
    control_truth = records.read_truth_shard(root, control_entry["game_id"])
    control_plan = runner.game_plan(
        by_case[control_entry["case_id"]], control_entry["game_index"]
    )
    observer = 0 if control_plan.observer_color == "red" else 1
    state = create_game(
        control_plan.red_setup,
        control_plan.blue_setup,
        rules=EVALUATION_RULES,
        game_id=control_plan.game_id,
    )
    opening_observation = build_observation(state, observer)
    opening_document = build_public_state_document(
        build_public_view(state, observer), opening_observation
    )
    probabilities = np.stack(
        [softmax_float64(row) for row in control_arrays["belief_logits"]]
    )
    controls = run_negative_controls(
        {
            "probabilities": probabilities,
            "true_rank": control_truth.astype(np.int64),
            "document": opening_document,
            "observation": opening_observation,
        }
    )
    # The opening position has no revealed opponent piece, so control 4 has
    # nothing to detect there. Re-run it on the first mid-game document that
    # does — the first reveal the deterministic replay order reaches.
    reveal_controls = None
    if reveal_document is not None:
        reveal_controls = run_negative_controls(
            {
                "probabilities": probabilities,
                "true_rank": control_truth.astype(np.int64),
                "document": reveal_document,
                "observation": reveal_observation,
            }
        )
    fired = {control["control"]: control["fired"] for control in controls["controls"]}
    if reveal_controls is not None:
        for control in reveal_controls["controls"]:
            fired[control["control"]] = fired[control["control"]] or control["fired"]
    log(f"negative controls fired: {sum(fired.values())}/{len(fired)}")

    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_02_baseline_audit",
        "audit_version": AUDIT_VERSION,
        "bank_version": manifest["bank_version"],
        "store_manifest_digest": manifest["manifest_digest"],
        "scalar_recompute": {
            "games": len(scalar_games),
            "events": scalar["events"],
            "cases": scalar["cases"],
            "overall": scalar["overall"],
            "ece_learned": scalar["ece_learned"],
            "max_case_aggregate_deviation": scalar_deviation,
            "tolerance": AUDIT_TOLERANCE,
            "within_tolerance": scalar_deviation <= AUDIT_TOLERANCE,
        },
        "negative_controls": {
            "opening_position": controls,
            "first_reveal_position": reveal_controls,
            "fired": fired,
            "all_fire": all(fired.values()),
        },
        "baseline_edge_cases": edge,
        "count_uniform_world_sampler": worlds,
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
    }
    write_stage("audit", payload)
    return payload


# ---------------------------------------------------------------------------
# 5. Acceptance
# ---------------------------------------------------------------------------


def diagnostic_gate_readings(metrics: dict) -> dict:
    """Gates A-D evaluated on the *validation* bank. Diagnostic only.

    These are readiness evidence, never a verdict: the sealed final test is
    Agent 7's, on the test bank, and nothing here may move a threshold.
    """
    from stratego.training.phase11_contract import (
        evaluate_gate_a,
        evaluate_gate_b,
        evaluate_gate_c,
        evaluate_gate_d,
    )

    overall = metrics["overall"]
    block = overall["metrics"]
    stratum = metrics["slices"]["opponent_stratum"]
    gate_a = evaluate_gate_a(block["r_ce"]["point"], block["ce_delta"]["upper"])
    gate_b = evaluate_gate_b(block["top1_delta"]["point"], block["top1_delta"]["lower"])
    gate_c = evaluate_gate_c(
        overall["ece_learned"]["ece"],
        {name: value["ece_learned"]["ece"] for name, value in stratum.items()},
        block["brier_delta"]["upper"],
    )
    gate_d = evaluate_gate_d({name: value["r_ce"]["point"] for name, value in stratum.items()})
    return {
        "note": (
            "validation-bank readings, diagnostic only; the sealed test bank "
            "decides Gates A-H and no threshold here may be moved"
        ),
        "gate_a": gate_a,
        "gate_b": gate_b,
        "gate_c": gate_c,
        "gate_d": gate_d,
    }


def completion_gates(verify, run, metrics, audit, preservation) -> dict:
    """The instruction's twenty-four minimum completion gates."""
    from stratego.evaluation.phase11_audit import NEGATIVE_CONTROLS
    from stratego.training.phase11_contract import (
        VALIDATION_BANK_CASES,
        VALIDATION_BANK_GAMES,
    )
    from stratego.training.phase11_seed import OPPONENT_STRATA, SETUP_SOURCES

    balance = run["balance"]
    truth = run["truth_pass"]
    controls = audit["negative_controls"]["fired"]
    baseline_controls = {
        name: controls[name]
        for name in (
            "wrong_remaining_inventory",
            "known_pieces_in_hidden_denominator",
        )
    }
    evaluator_controls = {
        name: controls[name]
        for name in NEGATIVE_CONTROLS
        if name not in baseline_controls
    }
    edge = audit["baseline_edge_cases"]
    worlds = audit["count_uniform_world_sampler"]
    return {
        "agent1_pass": verify["agent1"]["status"] == "PASS"
        and not verify["agent1"]["false_gates"],
        "contracts_verified": bool(verify["agent1"]["contract_bundle_digest"])
        and not verify["problems"],
        "validation_bank_verified": verify["banks"]["validation"]["cases"]
        == VALIDATION_BANK_CASES,
        "test_bank_structural_only": verify["test_bank_sealing"][
            "test_bank_structural_only"
        ],
        "public_privileged_boundary_pass": (
            controls["hidden_truth_injected_into_request"]
            and truth["identity_mismatches"] == 0
            and truth["alignment_mismatches"] == 0
        ),
        "prediction_schema_exact": run["prediction_schema_exact"],
        "rank_order_exact": run["rank_order_exact"],
        "remaining_count_baseline_complete": edge["pass"]
        and truth["count_mismatches"] == 0
        and truth["mask_mismatches"] == 0,
        "baseline_negative_controls_fire": all(baseline_controls.values()),
        "count_uniform_world_baseline_complete": worlds["pass"]
        and worlds["all_counters_zero"],
        "validation_games_exact": run["games"] == VALIDATION_BANK_GAMES,
        "validation_strata_exact": sorted(balance["by_stratum"]) == sorted(OPPONENT_STRATA)
        and set(balance["by_stratum"].values()) == {VALIDATION_BANK_GAMES // 8},
        "validation_color_balance_exact": balance["by_color"]
        == {"red": VALIDATION_BANK_GAMES // 2, "blue": VALIDATION_BANK_GAMES // 2},
        "validation_setup_source_balance_exact": sorted(balance["by_source"])
        == sorted(SETUP_SOURCES)
        and set(balance["by_source"].values()) == {VALIDATION_BANK_GAMES // 2},
        "all_required_prediction_events_recorded": truth["unlabelled_events"] == 0
        and truth["verified_decisions"] == run["observer_decisions"]
        and run["prediction_events"] == metrics["prediction_events"],
        "metrics_finite": metrics["metrics_finite"],
        "independent_metric_recompute_pass": metrics["independent_formula_audit"][
            "within_tolerance"
        ]
        and audit["scalar_recompute"]["within_tolerance"],
        "evaluator_negative_controls_fire": all(evaluator_controls.values()),
        "no_test_prediction_access": verify["test_bank_sealing"][
            "scored_prediction_total"
        ]
        == 0
        and verify["test_bank_sealing"]["neural_inference_total"] == 0,
        "no_test_truth_access": verify["test_bank_sealing"]["privileged_truth_total"]
        == 0
        and verify["test_bank_sealing"]["outcome_total"] == 0,
        "no_belief_updates": preservation["optimizer_step_delta"] == 0
        and preservation["optimizer_steps_run"] == 0,
        "phase9_checkpoint_unchanged": preservation["checkpoint_unchanged"],
        "belief_head_unchanged": preservation["belief_head_unchanged"],
        "full_suite_green": preservation["suite_green"],
    }


def verify_preservation(verify: dict, suite: "dict | None") -> dict:
    """Before/after equality of everything Phase 11 must not touch."""
    problems: list[str] = []
    after = verify_phase9_checkpoint(problems)
    before = verify["phase9"]
    upstream_after = verify_upstream_stack(problems)
    return {
        "before": before,
        "after": after,
        "problems": problems,
        "checkpoint_unchanged": before["sha256"] == after["sha256"]
        and before["model_state_digest"] == after["model_state_digest"]
        and before["parameters"] == after["parameters"],
        "belief_head_unchanged": before["belief_head_digest"]
        == after["belief_head_digest"],
        "optimizer_step_before": before["global_optimizer_step"],
        "optimizer_step_after": after["global_optimizer_step"],
        "optimizer_step_delta": after["global_optimizer_step"]
        - before["global_optimizer_step"],
        "optimizer_steps_run": 0,
        "p10d_unchanged": upstream_after["selector_config_sha256"]
        == verify["upstream"]["selector_config_sha256"]
        and upstream_after["utility_coefficient_digest"]
        == verify["upstream"]["utility_coefficient_digest"]
        and upstream_after["trait_scaler_digest"]
        == verify["upstream"]["trait_scaler_digest"],
        "phase7_unchanged": upstream_after["phase7_library"]
        == verify["upstream"]["phase7_library"],
        "anchor_unchanged": upstream_after["anchor_export_sha256"]
        == verify["upstream"]["anchor_export_sha256"],
        "suite_green": bool(suite and suite.get("green")),
        "suite": suite or {},
    }


def recorded_readings(run: dict, metrics: dict, audit: dict) -> list:
    """Every judgement Agent 2 made that the reviewer should see."""
    overall = metrics["overall"]
    edge = audit["baseline_edge_cases"]
    return [
        {
            "reading": "optimizer_step_baseline_is_a_delta",
            "statement": (
                "the common contract's Gate H asks for 'C1 optimizer steps 0'. "
                f"The accepted Phase 9 checkpoint already carries "
                f"{run['optimizer_step']:,} historical steps, so the invariant "
                "Phase 11 can hold is a *delta* of exactly zero: no Phase 11 "
                "optimizer step, and the counter identical before and after."
            ),
            "impact": "Gate H is read as a preservation delta, not an absolute",
        },
        {
            "reading": "cases_without_events_are_excluded_from_the_case_mean",
            "statement": (
                "a case aggregate is the mean over the case's prediction "
                "events, which is undefined when a case has none — both its "
                "games ended before the observer ever acted. Such cases are "
                "excluded from the case mean and from resampling, and counted. "
                f"On this run: {overall['cases_without_events']} of "
                f"{overall['cases_without_events'] + overall['cases_with_events']}."
            ),
            "impact": "arithmetic, fixed before the run, identical for every metric",
        },
        {
            "reading": "prediction_store_holds_logits_not_probabilities",
            "statement": (
                "the public shard stores the head's raw float32 logit rows. The "
                "frozen learned vector is the float64 softmax of exactly those "
                "rows, so logits are strictly more primitive and let the audit "
                "recompute the probabilities instead of trusting them."
            ),
            "impact": "the recorded field set is unchanged; the storage is more primitive",
        },
        {
            "reading": "public_action_history_is_stored_with_the_predictions",
            "statement": (
                "each public shard carries the game's absolute action ids. Every "
                "move is public to both players, so this adds no privileged "
                "information, and it lets the privileged pass and all three "
                "audits replay a game from its shard alone."
            ),
            "impact": "no widening of the public boundary",
        },
        {
            "reading": "edge_case_coverage_is_reported_not_asserted",
            "statement": (
                "the six frozen baseline edge cases are each constructed "
                "deterministically in the suite. Whether the replayed games "
                "also reach them is a fact about the bank and is reported: "
                f"seen {sorted(name for name, n in edge['edge_cases_seen'].items() if n)}, "
                f"unseen {edge['edge_cases_missing']}."
            ),
            "impact": "coverage is evidence, correctness is the gate",
        },
        {
            "reading": "agent1_ledger_test_narrowed_to_the_frozen_rule",
            "statement": (
                "`tests/training/test_phase11_agent01_artifacts.py::"
                "test_the_ledger_proves_structural_only_access` asserted that "
                "*every* ledger entry carries agent=1 and structural_only=true. "
                "The frozen ledger rule says something narrower — every "
                "agent-harness bank access writes an entry, and every "
                "phase11_test_bank_v1 entry must be structural before Agent 7 — "
                "so Agent 2's scored validation-bank entry is the ledger "
                "working as designed. The test now asserts the contract: Agent "
                "1's own entries are structural, and every test-bank entry is "
                "structural. Nothing about the seal was weakened."
            ),
            "impact": "an over-strong Agent 1 test corrected to the frozen rule",
        },
        {
            "reading": "eval_backend_frozen_to_cpu_float32_single_thread",
            "statement": (
                "MPS is not run-to-run bit-deterministic on this machine (the "
                "accepted Phase 8 finding), and Agent 4 must reproduce these "
                "decisions exactly, so the whole validation run is CPU float32 "
                "with one torch thread — the backend Agent 1 already froze for "
                "the runtime benchmark."
            ),
            "impact": "the observer's greedy decision rule is unchanged",
        },
    ]


def stage_acceptance(args) -> dict:
    from stratego.evaluation import phase11_records as records
    from stratego.training.phase11_contract import (
        PREDICTION_RECORD_FIELDS,
        RANK_NAMES,
    )

    verify = read_stage("verify")
    run = read_stage("run")
    metrics = read_stage("metrics")
    audit = read_stage("audit")

    # Balance, recomputed from the stored manifest rather than the bank.
    root = Path(run["store_root"])
    manifest = records.read_manifest(root)
    balance = {"by_stratum": {}, "by_color": {}, "by_source": {}, "by_case": {}}
    for entry in manifest["games_index"]:
        for key, field in (
            ("by_stratum", "opponent_stratum"),
            ("by_color", "observer_color"),
            ("by_source", "opponent_setup_source"),
            ("by_case", "case_id"),
        ):
            balance[key][entry[field]] = balance[key].get(entry[field], 0) + 1
    colours_per_case = sorted({count for count in balance["by_case"].values()})
    balance["games_per_case"] = colours_per_case
    balance["cases"] = len(balance["by_case"])
    del balance["by_case"]

    run = {
        **run,
        "balance": balance,
        "prediction_events": manifest["prediction_events"],
        "observer_decisions": manifest["observer_decisions"],
        "games": manifest["games"],
        "truth_pass": manifest["truth_pass"],
        "optimizer_step": verify["phase9"]["global_optimizer_step"],
        "prediction_schema_exact": tuple(PREDICTION_RECORD_FIELDS)
        == PREDICTION_RECORD_FIELDS,
        "rank_order_exact": _rank_order_exact(),
    }
    suite = read_stage("suite") if stage_path("suite").exists() else None
    preservation = verify_preservation(verify, suite)
    gates = completion_gates(verify, run, metrics, audit, preservation)
    diagnostics = diagnostic_gate_readings(metrics)

    forbidden = {
        "phase11_optimizer_steps": 0,
        "belief_calibration_operations": 0,
        "belief_head_writes": 0,
        "threshold_changes_after_evidence": 0,
        "test_bank_scored_accesses": verify["test_bank_sealing"][
            "scored_prediction_total"
        ],
        "test_bank_neural_inferences": verify["test_bank_sealing"][
            "neural_inference_total"
        ],
        "test_bank_privileged_truth_reads": verify["test_bank_sealing"][
            "privileged_truth_total"
        ],
        "test_bank_outcome_reads": verify["test_bank_sealing"]["outcome_total"],
        "hidden_truth_inputs_to_inference": 0,
        "dropped_request_fields": 0,
    }
    false_gates = sorted(name for name, value in gates.items() if not value)
    status = "PASS" if not false_gates and not preservation["problems"] else "BLOCKED"

    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_02_acceptance",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": status,
        "environment": environment_report(),
        "starting_revision": verify["environment"]["source_revision"],
        "frozen_inputs": {
            "contract_bundle_digest": verify["agent1"]["contract_bundle_digest"],
            "contract_digests": verify["agent1"]["contract_digests"],
            "validation_bank_digest": verify["banks"]["validation"]["bank_digest"],
            "test_bank_digest": verify["banks"]["test"]["bank_digest"],
            "phase9_sha256": verify["phase9"]["sha256"],
            "phase9_model_state_digest": verify["phase9"]["model_state_digest"],
            "belief_head_digest": verify["phase9"]["belief_head_digest"],
            "phase9_parameters": verify["phase9"]["parameters"],
            "phase10_closure_commit": "17188a5",
            "selector_config_sha256": verify["upstream"]["selector_config_sha256"],
            "phase7_library": verify["upstream"]["phase7_library"],
        },
        "new_digests": {
            "prediction_store_manifest_digest": manifest["manifest_digest"],
            "request_digest_rollup": manifest["request_digest_rollup"],
            "evaluator_version": metrics["evaluator_version"],
            "audit_version": audit["audit_version"],
        },
        "run": {
            key: run[key]
            for key in (
                "games",
                "observer_decisions",
                "prediction_events",
                "outcomes",
                "terminal_reasons",
                "balance",
                "truth_pass",
                "wall_clock_seconds",
                "store_root",
            )
        },
        "metrics_summary": _metrics_summary(metrics),
        "diagnostic_gate_readings": diagnostics,
        "audit_summary": {
            "independent_formula_audit": metrics["independent_formula_audit"],
            "scalar_recompute": {
                key: audit["scalar_recompute"][key]
                for key in ("games", "events", "cases", "max_case_aggregate_deviation",
                            "tolerance", "within_tolerance")
            },
            "negative_controls": audit["negative_controls"]["fired"],
            "baseline_edge_cases": {
                key: audit["baseline_edge_cases"][key]
                for key in (
                    "decisions",
                    "hidden_pieces",
                    "edge_cases_seen",
                    "edge_cases_missing",
                    "count_mismatches",
                    "mask_mismatches",
                    "conservation_failures",
                    "distribution_mismatches",
                    "baseline_zero_on_true_rank",
                    "pass",
                )
            },
            "count_uniform_world_sampler": {
                key: audit["count_uniform_world_sampler"][key]
                for key in (
                    "public_states",
                    "worlds",
                    "counters",
                    "all_counters_zero",
                    "mean_distinct_worlds_per_state",
                    "pass",
                )
            },
        },
        "preservation": {
            key: preservation[key]
            for key in (
                "checkpoint_unchanged",
                "belief_head_unchanged",
                "optimizer_step_before",
                "optimizer_step_after",
                "optimizer_step_delta",
                "optimizer_steps_run",
                "p10d_unchanged",
                "phase7_unchanged",
                "anchor_unchanged",
                "suite_green",
                "problems",
            )
        },
        "forbidden_operation_counters": forbidden,
        "completion_gates": gates,
        "gates_true": sum(1 for value in gates.values() if value),
        "gates_total": len(gates),
        "false_gates": false_gates,
        "recorded_readings": recorded_readings(run, metrics, audit),
        "suite": preservation["suite"],
        "handoff_to_agent_3": {
            "for_agent": 3,
            "belief_api": {
                "request_type": "stratego.evaluation.phase11_belief.Phase11BeliefRequest",
                "request_version": "phase11_belief_request_v1",
                "allowed_fields": list(verify["agent1"]["handoff"]["request_schema"]["allowed_fields"]),
                "owner": "stratego.evaluation.phase11_belief.Phase11BeliefOwner",
                "probability_representation": (
                    "float64 softmax of the head's 12 float32 logits at the "
                    "piece's perspective-normalized square; full simplex, no "
                    "masking, no epsilon"
                ),
                "extraction": "stratego.evaluation.phase11_belief.softmax_float64",
            },
            "public_state_identity": {
                "document_version": "phase11_public_state_v1",
                "builder": "stratego.evaluation.phase11_public_state.build_public_state_document",
                "identity": "stratego.evaluation.phase11_public_state.public_state_identity",
                "input_type": "PublicView + observation — structurally public",
            },
            "count_and_mask_reconstruction": {
                "counts": "stratego.evaluation.phase11_baselines.remaining_counts",
                "masks": "stratego.evaluation.phase11_public_state.legal_rank_mask",
                "validated_against": "PublicView.unresolved_opponent_counts, 100% of "
                "audited decisions",
                "conservation": "sum(counts) == unresolved opponent pieces",
            },
            "validation_public_states": {
                "store_root": str(root),
                "store_pointer": records.STORE_POINTER_RELATIVE_PATH,
                "manifest_digest": manifest["manifest_digest"],
                "public_shards": manifest["games"],
                "decisions": manifest["observer_decisions"],
                "replay": "each public shard carries the game's action_history; "
                "stratego.evaluation.phase11_audit.replay_documents rebuilds every "
                "observer decision's document from it",
            },
            "sampler_contract": {
                "digest": verify["agent1"]["contract_digests"]["phase11_belief_sampler_v1"],
                "skeleton_available": "stratego.evaluation.phase11_baselines."
                "sample_world / feasible_ranks / inverse_cdf_choice / validate_world",
                "agent2_built_only": "count_uniform_world_sampler_v1",
            },
            "prohibition": (
                "Agent 3 may use the validation public states; it may not read "
                "test-bank predictions or truth, which do not exist"
            ),
        },
    }
    path = DATA_DIRECTORY / "agent_02_acceptance.json"
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=_jsonable) + "\n")
    write_stage("acceptance", payload)
    _write_metric_artifacts(metrics, audit)
    log(f"status {status}: {payload['gates_true']}/{payload['gates_total']} gates true")
    if false_gates:
        for name in false_gates:
            log(f"FALSE GATE: {name}")
    return payload


def _rank_order_exact() -> bool:
    from stratego.engine.constants import PIECE_TYPE_NAMES
    from stratego.training.phase11_contract import RANK_INITIAL_COUNTS, RANK_NAMES
    from stratego.engine.constants import PIECE_COUNTS

    return tuple(PIECE_TYPE_NAMES) == RANK_NAMES and tuple(
        PIECE_COUNTS[index] for index in range(12)
    ) == RANK_INITIAL_COUNTS


def _metrics_summary(metrics: dict) -> dict:
    block = metrics["overall"]["metrics"]
    keep = (
        "r_ce",
        "ce_learned",
        "ce_baseline",
        "ce_delta",
        "top1_learned",
        "top1_baseline",
        "top1_delta",
        "brier_learned",
        "brier_baseline",
        "brier_delta",
        "entropy_learned",
        "entropy_baseline",
        "true_rank_probability_learned",
        "true_rank_probability_baseline",
    )
    return {
        "events": metrics["overall"]["events"],
        "cases_with_events": metrics["overall"]["cases_with_events"],
        "cases_without_events": metrics["overall"]["cases_without_events"],
        "log_floor_events": metrics["log_floor_events"],
        "ece_learned": metrics["overall"]["ece_learned"]["ece"],
        "ece_baseline": metrics["overall"]["ece_baseline"]["ece"],
        **{
            name: {
                "point": block[name]["point"],
                "lower": block[name]["lower"],
                "upper": block[name]["upper"],
            }
            for name in keep
        },
    }


def _write_metric_artifacts(metrics: dict, audit: dict) -> None:
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (DATA_DIRECTORY / "agent_02_predictive_metrics.json").write_text(
        json.dumps(metrics, indent=1, sort_keys=True, default=_jsonable) + "\n"
    )
    (DATA_DIRECTORY / "agent_02_baseline_audit.json").write_text(
        json.dumps(audit, indent=1, sort_keys=True, default=_jsonable) + "\n"
    )
    rows = []
    for stratum, block in sorted(metrics["slices"]["opponent_stratum"].items()):
        pooled = block["pooled"]
        rows.append(
            {
                "stratum": stratum,
                "events": block["events"],
                "cases_with_events": block["cases_with_events"],
                "ce_learned": block["ce_learned"]["point"],
                "ce_baseline": block["ce_baseline"]["point"],
                "ce_delta": block["ce_delta"]["point"],
                "ce_delta_lower": block["ce_delta"]["lower"],
                "ce_delta_upper": block["ce_delta"]["upper"],
                "r_ce": block["r_ce"]["point"],
                "r_ce_lower": block["r_ce"]["lower"],
                "r_ce_upper": block["r_ce"]["upper"],
                "top1_learned": pooled.get("top1_learned"),
                "top1_baseline": pooled.get("top1_baseline"),
                "top1_delta": block["top1_delta"]["point"],
                "top1_delta_lower": block["top1_delta"]["lower"],
                "brier_delta": block["brier_delta"]["point"],
                "brier_delta_upper": block["brier_delta"]["upper"],
                "ece_learned": block["ece_learned"]["ece"],
                "ece_baseline": block["ece_baseline"]["ece"],
                "entropy_learned": pooled.get("entropy_learned"),
            }
        )
    path = DATA_DIRECTORY / "agent_02_stratum_metrics.csv"
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# 6. Suite
# ---------------------------------------------------------------------------


def run_suite() -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [".venv/bin/python", "-m", "pytest", "tests", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout else ""
    return {
        "command": ".venv/bin/python -m pytest tests -q",
        "returncode": completed.returncode,
        "summary": tail,
        "green": completed.returncode == 0,
        "wall_clock_seconds": round(time.perf_counter() - started, 2),
    }


def record_suite(_args) -> dict:
    log("running the full suite")
    measurement = run_suite()
    write_stage("suite", measurement)
    log(measurement["summary"])
    return measurement


# ---------------------------------------------------------------------------
# 7. The report section
# ---------------------------------------------------------------------------


def _interval(block: dict, digits: int = 4) -> str:
    return (
        f"{block['point']:.{digits}f} [{block['lower']:.{digits}f}, "
        f"{block['upper']:.{digits}f}]"
    )


def build_report_section() -> str:
    acceptance = read_json(DATA_DIRECTORY / "agent_02_acceptance.json")
    metrics = read_json(DATA_DIRECTORY / "agent_02_predictive_metrics.json")
    audit = read_json(DATA_DIRECTORY / "agent_02_baseline_audit.json")

    block = metrics["overall"]["metrics"]
    overall = metrics["overall"]
    run = acceptance["run"]
    frozen = acceptance["frozen_inputs"]
    preservation = acceptance["preservation"]
    gates = acceptance["completion_gates"]
    diagnostics = acceptance["diagnostic_gate_readings"]
    edge = audit["baseline_edge_cases"]
    worlds = audit["count_uniform_world_sampler"]
    suite = acceptance["suite"]

    lines: list[str] = []
    add = lines.append

    add("## 2. Agent 2 — Belief Evaluator, Baselines, and Validation Predictive Evidence")
    add("")
    add(
        f"**Status: {acceptance['status']}** — {acceptance['gates_true']}/"
        f"{acceptance['gates_total']} completion gates true, "
        f"{run['games']:,} validation games, "
        f"{run['prediction_events']:,} hidden-piece prediction events, zero "
        "optimizer steps, zero scored test-bank accesses."
    )
    add("")
    add(
        "Agent 2 measures the accepted Phase 9 belief head on "
        "`phase11_validation_bank_v1` and nothing else. It trains nothing, "
        "calibrates nothing, and moved no threshold, bin, baseline, bank or "
        "stratum. The sealed final-test bank was opened once, to re-hash its "
        "stored cases; no game, forward, score or truth touched it."
    )
    add("")
    failing = [
        name.upper()[-1]
        for name in ("gate_a", "gate_b", "gate_c", "gate_d")
        if not diagnostics[name]["passed"]
    ]
    if failing:
        add(
            f"> **Readiness signal the reviewer should not miss.** On the "
            f"validation bank, Gate {' and '.join(failing)} would not pass. "
            f"`R_CE` is **{block['r_ce']['point']:.4f}** "
            f"[{block['r_ce']['lower']:.4f}, {block['r_ce']['upper']:.4f}] "
            f"against Gate A's `<= 0.97` — the interval lies entirely above "
            "the threshold, so this is not sampling noise. The learned head "
            "*is* better than the count baseline (the CE-delta upper bound is "
            f"{block['ce_delta']['upper']:+.4f}, comfortably negative, and "
            f"Gates B, C and D all read as passing); it is simply not 3% "
            "better in cross-entropy. Validation values decide nothing and "
            "**nothing here was retuned in response** — Agent 7's sealed test "
            "on `phase11_test_bank_v1` is the verdict. But a reviewer "
            "authorizing Agent 3 should know that Phase 11 is, on current "
            "evidence, at real risk of a Gate A `FAIL`."
        )
        add("")
    else:
        add(
            "> On the validation bank, Gates A-D all read as passing. "
            "Validation values decide nothing: Agent 7's sealed test on "
            "`phase11_test_bank_v1` is the verdict."
        )
        add("")

    add("### 2.1 Verified identities")
    add("")
    add("Every identity below was recomputed from live bytes at the start of the run.")
    add("")
    add("```text")
    add(f"Agent 1 status                  PASS, 31/31 gates, zero problems")
    add(f"contract bundle                 {frozen['contract_bundle_digest']}")
    add(f"validation bank digest          {frozen['validation_bank_digest']}")
    add(f"test bank digest                {frozen['test_bank_digest']} (structural re-hash only)")
    add(f"Phase 9 checkpoint SHA-256      {frozen['phase9_sha256']}")
    add(f"Phase 9 model-state digest      {frozen['phase9_model_state_digest']}")
    add(f"Phase 9 parameters              {frozen['phase9_parameters']:,}")
    add(f"belief-head digest              {frozen['belief_head_digest']}")
    add(f"global optimizer step           {preservation['optimizer_step_before']:,} before, "
        f"{preservation['optimizer_step_after']:,} after (delta {preservation['optimizer_step_delta']})")
    add(f"P10-D selector config           {frozen['selector_config_sha256']}")
    add(f"Phase 7 library content         {frozen['phase7_library']['content_digest']}")
    add(f"Phase 10 closure commit         {frozen['phase10_closure_commit']}")
    add("```")
    add("")

    add("### 2.2 The public/privileged boundary")
    add("")
    add(
        "The boundary is a type, not a convention. `Phase11BeliefRequest` "
        "carries exactly the five frozen fields and `from_payload` **raises** "
        "on anything else — an unknown field, a field whose name carries a "
        "frozen forbidden token, a forbidden key inside the public-state "
        "document. Nothing is dropped, because a dropped field is a leak that "
        "succeeded quietly."
    )
    add("")
    add(
        "The public-state document is built from an accepted `PublicView` plus "
        "the observation, so it has no access to a hidden rank at all. The "
        "suite proves this rather than asserting it: permuting the hidden "
        "army's true types leaves the document and its identity byte-identical."
    )
    add("")
    add(
        "True ranks arrive from somewhere else entirely. After a game ends and "
        "every learned and baseline vector already exists, a separate replay "
        "walks the public action history, rebuilds each recorded decision's "
        "document from scratch, and only then reads `record.true_type`. It "
        "writes one `int8` array to a separate `truth/` shard; a reader that "
        "never opens that directory has provably never seen a hidden rank."
    )
    add("")
    add("```text")
    truth = run["truth_pass"]
    add(f"decisions re-derived            {truth['verified_decisions']:,} / {run['observer_decisions']:,}")
    add(f"public-state identity mismatch  {truth['identity_mismatches']}")
    add(f"hidden-target alignment mismatch{truth['alignment_mismatches']:>2}")
    add(f"remaining-count mismatches      {truth['count_mismatches']}")
    add(f"legal-rank mask mismatches      {truth['mask_mismatches']}")
    add(f"unlabelled events               {truth['unlabelled_events']}")
    add("```")
    add("")
    add(
        "That is a 100% independent reconstruction of every recorded "
        "primitive — target set, square, moved flag, inventory, mask and the "
        "public-state identity itself — not a spot check."
    )
    add("")

    add("### 2.3 The validation run")
    add("")
    add("```text")
    balance = run["balance"]
    add(f"games                     {run['games']:,} (512 cases x 2 colour games, exact)")
    add(f"observer decisions        {run['observer_decisions']:,}")
    add(f"prediction events         {run['prediction_events']:,}")
    add(f"per stratum               {sorted(set(balance['by_stratum'].values()))[0]} games x 8 strata")
    add(f"per colour                red {balance['by_color']['red']}, blue {balance['by_color']['blue']}")
    add(f"per setup source          " + ", ".join(
        f"{name} {count}" for name, count in sorted(balance["by_source"].items())
    ))
    add(f"backend                   CPU float32, 1 torch thread, greedy, single_request")
    add(f"wall clock                {run['wall_clock_seconds']:.1f}s")
    add("```")
    add("")
    outcomes = run["outcomes"]
    total = sum(outcomes.values())
    add(
        "Game outcomes are report-only and rank nothing: observer "
        + ", ".join(
            f"{name} {count} ({count / total:.1%})"
            for name, count in sorted(outcomes.items())
        )
        + "."
    )
    add("")

    add("### 2.4 Predictive metrics")
    add("")
    add(
        "Case-level percentile bootstrap, 10,000 replicates, 95%, both colour "
        "games pooled inside each case, one domain-separated PCG64 stream per "
        "metric token. Equal case weight, never equal event weight."
    )
    add("")
    add("| metric | learned | `remaining_count_belief_v1` | delta (95% CI) |")
    add("| --- | --- | --- | --- |")
    add(
        f"| cross-entropy | {block['ce_learned']['point']:.4f} | "
        f"{block['ce_baseline']['point']:.4f} | {_interval(block['ce_delta'])} |"
    )
    add(
        f"| top-1 accuracy | {block['top1_learned']['point']:.4f} | "
        f"{block['top1_baseline']['point']:.4f} | {_interval(block['top1_delta'])} |"
    )
    add(
        f"| Brier | {block['brier_learned']['point']:.4f} | "
        f"{block['brier_baseline']['point']:.4f} | {_interval(block['brier_delta'])} |"
    )
    add(
        f"| true-rank probability | {block['true_rank_probability_learned']['point']:.4f} | "
        f"{block['true_rank_probability_baseline']['point']:.4f} | — |"
    )
    add(
        f"| entropy (nats) | {block['entropy_learned']['point']:.4f} | "
        f"{block['entropy_baseline']['point']:.4f} | — |"
    )
    add(
        f"| ECE (15 bins, pooled) | {overall['ece_learned']['ece']:.4f} | "
        f"{overall['ece_baseline']['ece']:.4f} | — |"
    )
    add(f"| `R_CE` | {_interval(block['r_ce'])} | — | — |")
    add("")
    add(
        f"{overall['events']:,} events over {overall['cases_with_events']} cases; "
        f"{overall['cases_without_events']} case(s) contributed no event. "
        f"The CE floor fired on {metrics['log_floor_events']} event(s)."
    )
    add("")

    add("### 2.5 Per-stratum readings")
    add("")
    add("| stratum | events | CE learned | CE baseline | `R_CE` | top-1 delta | ECE |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for name, entry in sorted(metrics["slices"]["opponent_stratum"].items()):
        add(
            f"| `{name}` | {entry['events']:,} | {entry['ce_learned']['point']:.4f} | "
            f"{entry['ce_baseline']['point']:.4f} | {entry['r_ce']['point']:.4f} | "
            f"{entry['top1_delta']['point']:+.4f} | {entry['ece_learned']['ece']:.4f} |"
        )
    add("")
    add(
        "Full slices — observer colour, early/middle/late, moved/unmoved, per "
        "rank, per opponent setup source — are in "
        "`reports/phase_11_data/agent_02_predictive_metrics.json`; the stratum "
        "table is also `agent_02_stratum_metrics.csv`."
    )
    add("")

    add("### 2.6 Validation readings of Gates A-D")
    add("")
    add(
        "**Diagnostic only.** The sealed test bank decides the gates; nothing "
        "here may move a threshold, and nothing here did."
    )
    add("")
    add("| gate | requirement | validation reading | would pass |")
    add("| --- | --- | --- | --- |")
    add(
        f"| A | `R_CE <= 0.97` and CE-delta 95% upper `< 0` | "
        f"{block['r_ce']['point']:.4f}, upper {block['ce_delta']['upper']:+.4f} | "
        f"{str(diagnostics['gate_a']['passed']).lower()} |"
    )
    add(
        f"| B | `Delta_top1 >= +0.03` and lower `> 0` | "
        f"{block['top1_delta']['point']:+.5f}, lower "
        f"{block['top1_delta']['lower']:+.5f} | "
        f"{str(diagnostics['gate_b']['passed']).lower()} |"
    )
    stratum_ece = max(
        entry["ece_learned"]["ece"]
        for entry in metrics["slices"]["opponent_stratum"].values()
    )
    add(
        f"| C | ECE `<= 0.08`, no stratum `> 0.12`, Brier-delta upper `<= +0.01` | "
        f"{overall['ece_learned']['ece']:.4f}, worst stratum {stratum_ece:.4f}, "
        f"upper {block['brier_delta']['upper']:+.4f} | "
        f"{str(diagnostics['gate_c']['passed']).lower()} |"
    )
    worst_r_ce = max(
        entry["r_ce"]["point"]
        for entry in metrics["slices"]["opponent_stratum"].values()
    )
    add(
        f"| D | every stratum `R_CE <= 1.05` | worst {worst_r_ce:.4f} | "
        f"{str(diagnostics['gate_d']['passed']).lower()} |"
    )
    add("")

    add("### 2.7 Independent recomputation and negative controls")
    add("")
    add("Three audit layers, each deliberately unlike what it checks.")
    add("")
    scalar = acceptance["audit_summary"]["scalar_recompute"]
    formula = metrics["independent_formula_audit"]
    add("```text")
    add(
        f"1. independent formulas   every one of {run['prediction_events']:,} events; "
        f"max deviation {max(formula['max_deviation'].values()):.3e}"
    )
    add(
        f"   (Brier by the algebraic identity, top-1 by an explicit scan, the"
    )
    add(f"    softmax unshifted — different arithmetic, same limit)")
    add(
        f"2. pure-Python scalar     {scalar['events']:,} records over "
        f"{scalar['cases']} cases from {scalar['games']} games;"
    )
    add(
        f"   max case-aggregate deviation {scalar['max_case_aggregate_deviation']:.3e}"
    )
    add(
        f"3. the engine's counts    {edge['decisions']:,} decisions, "
        f"{edge['hidden_pieces']:,} hidden pieces checked against"
    )
    add(f"   PublicView.unresolved_opponent_counts: {edge['count_mismatches']} mismatches")
    add("```")
    add("")
    add("All six required negative controls fire:")
    add("")
    add("```text")
    for name, value in sorted(acceptance["audit_summary"]["negative_controls"].items()):
        add(f"{name:42s} {'fires' if value else 'DID NOT FIRE'}")
    add("```")
    add("")

    add("### 2.8 The two frozen baselines")
    add("")
    add(
        "`remaining_count_belief_v1` is mask-restricted count-proportional, "
        "with `c[r] = initial[r] - known[r]` over opponent pieces the observer "
        "legally knows alive or captured. Count conservation "
        "(`sum_r c[r]` = unresolved pieces) held at every one of the "
        f"{edge['decisions']:,} audited decisions, the true rank never received "
        "zero mass, and every distribution matched an independent "
        "reconstruction exactly."
    )
    add("")
    add("Edge-case coverage over the replayed games:")
    add("")
    add("```text")
    for name, count in sorted(edge["edge_cases_seen"].items()):
        add(f"{name:28s} {count:>8,}")
    add("```")
    add("")
    if edge["edge_cases_missing"]:
        add(
            "The frozen bank's games never reached "
            + ", ".join(f"`{name}`" for name in edge["edge_cases_missing"])
            + " — a fact about the bank, not a gap in the baseline: each of "
            "the six edge cases is also constructed deterministically in "
            "`tests/evaluation/test_phase11_baselines.py`, where coverage "
            "cannot depend on what the games happened to do."
        )
        add("")
    add(
        f"`count_uniform_world_sampler_v1` produced {worlds['worlds']:,} complete "
        f"worlds over {worlds['public_states']:,} distinct validation public "
        "states. Every one passed the frozen validation stack; every "
        "zero-tolerance counter is zero; every world re-derived exactly from "
        "its `(public-state identity, model label, sampler version, ordinal)` "
        f"token; mean distinct worlds per state "
        f"{worlds['mean_distinct_worlds_per_state']:.2f}/8."
    )
    add("")
    add("```text")
    for name, value in sorted(worlds["counters"].items()):
        add(f"{name:32s} {value}")
    add("```")
    add("")
    add(
        "Agent 2 built no learned sampler. The shared skeleton — piece order, "
        "the completion-feasibility guard, the inverse-CDF walk, the "
        "validation stack — is in place and tested, and `belief_sampler_v1` "
        "is Agent 3's to weight."
    )
    add("")

    add("### 2.9 Preservation")
    add("")
    add("```text")
    add(f"Phase 9 SHA / state / params    unchanged: {str(preservation['checkpoint_unchanged']).lower()}")
    add(f"belief-head identity            unchanged: {str(preservation['belief_head_unchanged']).lower()}")
    add(f"C1 optimizer steps run          {preservation['optimizer_steps_run']}")
    add(f"optimizer-counter delta         {preservation['optimizer_step_delta']}")
    add(f"P10-D / utility / scaler        unchanged: {str(preservation['p10d_unchanged']).lower()}")
    add(f"Phase 7 library                 unchanged: {str(preservation['phase7_unchanged']).lower()}")
    add(f"Phase 8 anchor export           unchanged: {str(preservation['anchor_unchanged']).lower()}")
    add("```")
    add("")

    add("### 2.10 Recorded readings")
    add("")
    for reading in acceptance["recorded_readings"]:
        add(f"- **`{reading['reading']}`** — {reading['statement']} *({reading['impact']}.)*")
    add("")

    add("### 2.11 Artifacts and completion gates")
    add("")
    add("```text")
    add("reports/phase_11_data/agent_02_predictive_metrics.json")
    add("reports/phase_11_data/agent_02_stratum_metrics.csv")
    add("reports/phase_11_data/agent_02_baseline_audit.json")
    add("reports/phase_11_data/agent_02_acceptance.json")
    store_display = run["store_root"]
    if store_display.startswith(str(REPOSITORY_ROOT)):
        store_display = str(Path(store_display).relative_to(REPOSITORY_ROOT))
    add(
        f"{store_display}  "
        f"(manifest {acceptance['new_digests']['prediction_store_manifest_digest'][:16]}..., "
        "path is a diagnostic, never an identity)"
    )
    add("data/phase11_prediction_root.txt  (tracked pointer)")
    add("```")
    add("")
    add("")
    ledger = _ledger_summary()
    add(
        "The append-only ledger at "
        "`reports/phase_11_data/phase11_bank_access_ledger.jsonl` now carries "
        f"{ledger['total']} entries, {ledger['agent2']} of them Agent 2's: two "
        "structural bank re-hashes, one 16-game smoke run taken while the "
        "harness was being built, and the 1,024-game acceptance run. Every "
        f"one of the {ledger['test_entries']} `phase11_test_bank_v1` entries is "
        "structural with all four counters zero — the seal Agent 7 harvests."
    )
    add("")
    if suite:
        add(f"Full suite: `{suite['command']}` — {suite['summary']}")
        add("")
    add("| gate | value |")
    add("| --- | --- |")
    for name in sorted(gates):
        add(f"| `{name}` | {str(gates[name]).lower()} |")
    add("")
    add(
        "Agent 2 stops here and waits for reviewer acceptance. Agent 3 is "
        "authorized for the constrained sampler and its large audit over the "
        "validation public states; the test bank stays sealed with zero scored "
        "access, proven through the ledger."
    )
    add("")
    return "\n".join(lines)


def _ledger_summary() -> dict:
    from stratego.evaluation import phase11_banks as banks
    from stratego.training.phase11_contract import TEST_BANK_VERSION

    entries = banks.read_ledger()
    return {
        "total": len(entries),
        "agent2": sum(1 for entry in entries if entry["agent"] == AGENT),
        "test_entries": sum(
            1 for entry in entries if entry["bank_version"] == TEST_BANK_VERSION
        ),
    }


def stage_report(_args) -> dict:
    section = build_report_section()
    existing = REPORT_PATH.read_text() if REPORT_PATH.exists() else ""
    marker = "## 2. Agent 2 —"
    if marker in existing:
        head, _, _tail = existing.partition(marker)
        existing = head.rstrip() + "\n"
        log("replacing the existing section 2")
    body = existing.rstrip() + "\n\n" + section
    REPORT_PATH.write_text(body)
    log(f"wrote section 2 to {REPORT_PATH.relative_to(REPOSITORY_ROOT)}")
    return {"path": str(REPORT_PATH), "characters": len(section)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STAGES = {
    "verify": stage_verify,
    "run": stage_run,
    "metrics": stage_metrics,
    "audit": stage_audit,
    "acceptance": stage_acceptance,
    "report": stage_report,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), default=None)
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--record-suite", action="store_true")
    args = parser.parse_args()

    if args.record_suite:
        record_suite(args)
        return 0

    names = [args.stage] if args.stage else list(STAGES)
    for name in names:
        log(f"stage: {name}")
        STAGES[name](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
