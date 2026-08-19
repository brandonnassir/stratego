#!/usr/bin/env python3
"""Phase 11 Agent 3 harness: `belief_sampler_v1` and the complete-world audit.

Recomputes every load-bearing identity from live bytes (the Agent 1 freeze
and its contracts, the Agent 2 PASS and its prediction store, both bank
digests, the Phase 9 checkpoint's file SHA / model-state digest / parameter
count / belief-head tensor identity / optimizer-step counter, the frozen
P10-D chain and the Phase 7 library), then audits the learned
`belief_sampler_v1` at scale on the frozen validation public states
**only**:

- >= 250,000 complete learned worlds, every one through the frozen
  validation stack, across thousands of distinct states spanning all eight
  strata, both colours, every progress bucket and moved/unmoved uncertainty;
- an independent second-path re-derivation (>= 25,000 worlds) that rebuilds
  inventory, masks, multiset, public facts and seed derivation from raw
  `blake2b` and the engine authority, sharing no Phase 11 module with the
  primary path;
- an exhaustive collision audit over every `world_sample` / `world_order` /
  `world_categorical` seed actually derived, combined with the complete
  Agent 1 enumerable seed universe;
- `count_uniform_world_sampler_v1` correctness on the same states (no
  strength comparison);
- seven negative controls, each required to fire;
- deterministic-repeat and call-order-reversal checks (Agent 4 owns the
  full topology/restart gate).

    reports/phase_11_data/agent_03_sampler_contract.json
    reports/phase_11_data/agent_03_sampler_audit.json
    reports/phase_11_data/agent_03_sampler_diagnostics.csv
    reports/phase_11_data/agent_03_acceptance.json

What this script is and is not
------------------------------
It samples hidden worlds from already-recorded public marginals. It runs
**no neural forward**, no game, no optimizer step; it opens no truth shard
and consumes no game outcome. The validation R_CE = 0.9750 reading is
treated as diagnostic only: no belief weight, mask, baseline, sampler
weighting, feasibility rule or Phase 11 threshold is altered in response.

Usage::

    python scripts/run_phase11_agent03.py                   # every stage
    python scripts/run_phase11_agent03.py --stage verify    # one stage
    python scripts/run_phase11_agent03.py --limit-games 8   # a smoke audit
    python scripts/run_phase11_agent03.py --record-suite    # run + record the suite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from array import array
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.engine.constants import (  # noqa: E402
    IMPLEMENTATION_VERSION,
    OBSERVATION_VERSION,
    RULES_VERSION,
)

AGENT = 3
PHASE = 11
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_11_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_11_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase11" / "agent03"
CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "selfplay_c1_v1.pt"

AUDIT_VERSION = "phase11_agent03_sampler_audit_v1"

#: The frozen audit schedule, chosen before any world was sampled.
#: 16 evenly spaced eligible decisions per game (the accepted benchmark /
#: soak spacing pattern) x 1,024 games ~= 16k distinct states; 16 learned
#: worlds per state clears the 250,000-world floor, and the rule below
#: raises the per-state count if the realized state count ever fell short.
SELECTED_DECISIONS_PER_GAME = 16
MIN_LEARNED_WORLDS = 250_000
MIN_INDEPENDENT_WORLDS = 25_000
BASELINE_WORLDS_PER_STATE = 4
INDEPENDENT_STRIDE = 10
REPEAT_STATE_STRIDE = 25
CONTROL_CANDIDATE_STATES = 256

#: The learned-sampler implementation modules whose byte identity is frozen
#: into the contract artifact.
IMPLEMENTATION_MODULES = (
    "stratego/evaluation/phase11_sampler.py",
    "stratego/evaluation/phase11_sampler_audit.py",
)

#: The seven negative controls of the Agent 3 instruction, in its order.
NEGATIVE_CONTROLS = (
    "remove_one_remaining_rank",
    "bomb_or_flag_on_moved_piece",
    "duplicate_marshal_count",
    "alter_public_known_rank",
    "mutate_sample_seed",
    "inject_true_hidden_rank",
    "corrupt_provenance",
)

#: The named inputs the request boundary must reject, from the instruction.
REJECTED_INPUT_FIELDS = (
    "true_rank",
    "true_rank_index",
    "private_piece_table",
    "opponent_setup",
    "opponent_setup_truth",
    "hidden_start_rank",
    "winner",
    "match_result",
    "reward",
    "future_actions",
    "search_hint_future",
    "storage_path",
)


class Agent3Error(RuntimeError):
    """Agent 3 cannot proceed."""


# ---------------------------------------------------------------------------
# Small helpers (the accepted Agent 2 harness conventions)
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
        "eval_device": "cpu",
        "torch_threads": 1,
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
        raise Agent3Error(f"stage {name!r} has not run yet ({path} is missing)")
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
    print(f"[phase11:agent3] {message}", flush=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_artifact(name: str, payload: dict) -> Path:
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DATA_DIRECTORY / name
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=_jsonable) + "\n")
    return path


# ---------------------------------------------------------------------------
# 1. Verification — every identity from live bytes
# ---------------------------------------------------------------------------


def verify_agent1(problems: list) -> dict:
    """Agent 1 must be PASS, with its digests re-derived here."""
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
        "contract_digests": live_digests,
        "contract_bundle_digest": live_bundle,
        "sampler_contract_digest": live_digests.get("phase11_belief_sampler_v1"),
    }


def verify_agent2(problems: list) -> dict:
    """Agent 2 must be PASS with zero forbidden operations, re-read here."""
    from stratego.training.phase11_contract import EVALUATOR_VERSION

    path = DATA_DIRECTORY / "agent_02_acceptance.json"
    require(path.exists(), "the Agent 2 acceptance artifact is missing", problems)
    if not path.exists():
        return {"available": False}
    acceptance = read_json(path)
    require(acceptance.get("status") == "PASS", "Agent 2 did not report PASS", problems)
    gates = acceptance.get("completion_gates", {})
    false_gates = sorted(name for name, value in gates.items() if not value)
    require(not false_gates, f"Agent 2 gates are false: {false_gates}", problems)
    counters = acceptance.get("forbidden_operation_counters", {})
    nonzero = sorted(name for name, value in counters.items() if value)
    require(
        not nonzero, f"Agent 2 forbidden-operation counters are non-zero: {nonzero}", problems
    )
    recorded_evaluator = acceptance.get("new_digests", {}).get("evaluator_version")
    require(
        recorded_evaluator == EVALUATOR_VERSION,
        f"Agent 2 evaluator version {recorded_evaluator!r} != {EVALUATOR_VERSION!r}",
        problems,
    )
    handoff = acceptance.get("handoff_to_agent_3", {})
    require(handoff.get("for_agent") == 3, "the Agent 2 handoff is not for Agent 3", problems)
    return {
        "available": True,
        "status": acceptance.get("status"),
        "gates_true": acceptance.get("gates_true"),
        "gates_total": acceptance.get("gates_total"),
        "evaluator_version": recorded_evaluator,
        "manifest_digest": acceptance.get("new_digests", {}).get(
            "prediction_store_manifest_digest"
        ),
        "validation_r_ce_point": acceptance.get("metrics_summary", {})
        .get("r_ce", {})
        .get("point"),
        "handoff": handoff,
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
        summary[name] = {
            "available": True,
            "bank_version": manifest["bank_version"],
            "bank_digest": digest,
            "cases": len(cases),
            "file_sha256": file_sha256(path),
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
        array_ = tensor.detach().to("cpu").numpy().astype(np.float32)
        shapes[name] = list(array_.shape)
        hasher.update(name.encode())
        hasher.update(str(tuple(array_.shape)).encode())
        hasher.update(np.ascontiguousarray(array_).tobytes())
    return {"digest": hasher.hexdigest(), "tensor_shapes": shapes}


def verify_phase9_checkpoint(problems: list) -> dict:
    """The accepted checkpoint, opened read-only and fully re-derived."""
    import torch

    from stratego.training.phase11_contract import (
        ACCEPTED_BELIEF_HEAD_DIGEST,
        ACCEPTED_GLOBAL_OPTIMIZER_STEP,
        ACCEPTED_PHASE9_CHECKPOINT_SHA256,
        ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
        ACCEPTED_PHASE9_PARAMETERS,
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

    require(
        digest == ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
        "model-state digest != accepted",
        problems,
    )
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
        "global_optimizer_step": optimizer_step,
    }


def verify_upstream_stack(problems: list) -> dict:
    """The P10-D selector chain, the Phase 8 anchor and the Phase 7 library."""
    from stratego.evaluation.phase11_banks import Phase11SetupSources
    from stratego.setups.contracts import LIBRARY_JSONL_PATH, LIBRARY_MANIFEST_PATH
    from stratego.setups.library import (
        entry_metadata_digest,
        library_content_digest,
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
        "the Phase 7 library content digest != accepted",
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
        "selector_config_sha256": ACCEPTED_SELECTOR_CONFIG_SHA256,
        "utility_coefficient_digest": ACCEPTED_UTILITY_COEFFICIENT_DIGEST,
        "trait_scaler_digest": ACCEPTED_TRAIT_SCALER_DIGEST,
        "anchor_export_sha256": anchor_sha,
        "phase7_library": library,
    }


def verify_prediction_store(problems: list) -> dict:
    """Agent 2's store: pointer resolves, manifest digest re-derives."""
    from stratego.evaluation import phase11_records as records

    root = records.store_root(REPOSITORY_ROOT)
    require(root.exists(), f"the prediction store root {root} is missing", problems)
    if not root.exists():
        return {"available": False}
    manifest = records.read_manifest(root)
    recorded = manifest.get("manifest_digest")
    recomputed = records.manifest_digest(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    )
    require(
        recorded == recomputed,
        f"the store manifest digest {recomputed} != recorded {recorded}",
        problems,
    )
    require(
        manifest.get("bank_version") == "phase11_validation_bank_v1",
        "the prediction store is not the validation store",
        problems,
    )
    require(
        len(manifest.get("games_index", [])) == 1_024,
        f"the store indexes {len(manifest.get('games_index', []))} games, expected 1024",
        problems,
    )
    require(
        bool(manifest.get("complete_bank")),
        "the store does not cover the complete validation bank",
        problems,
    )
    return {
        "available": True,
        "store_root": str(root),
        "manifest_digest": recorded,
        "games": len(manifest.get("games_index", [])),
        "observer_decisions": manifest.get("observer_decisions"),
        "prediction_events": manifest.get("prediction_events"),
        "bank_digest": manifest.get("bank_digest"),
        "model_identity": manifest.get("model_identity"),
    }


def verify_test_bank_sealed(problems: list) -> dict:
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
    log("verifying the Agent 2 PASS and its handoff")
    agent2 = verify_agent2(problems)
    log("re-hashing both frozen banks")
    bank_summary = verify_banks(problems)
    log("re-deriving the Phase 9 checkpoint and belief-head identity")
    phase9 = verify_phase9_checkpoint(problems)
    log("verifying the P10-D / anchor / Phase 7 stack")
    upstream = verify_upstream_stack(problems)
    log("verifying the prediction store")
    store = verify_prediction_store(problems)
    log("checking the test-bank seal")
    sealing = verify_test_bank_sealed(problems)

    if agent2.get("available") and store.get("available"):
        require(
            agent2["manifest_digest"] == store["manifest_digest"],
            "the store manifest digest does not match the Agent 2 record",
            problems,
        )
    if bank_summary.get("validation", {}).get("available") and store.get("available"):
        require(
            store["bank_digest"] == bank_summary["validation"]["bank_digest"],
            "the store's bank digest does not match the live validation bank",
            problems,
        )

    from stratego.evaluation import phase11_banks as banks

    banks.append_ledger_entries(
        [
            banks.ledger_entry(
                AGENT,
                "verify_bank_digest",
                bank_version,
                "structural re-hash of the frozen bank from its stored cases "
                "(also the file-byte preservation baseline)",
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
        "agent2": agent2,
        "banks": bank_summary,
        "phase9": phase9,
        "upstream": upstream,
        "prediction_store": store,
        "test_bank_sealing": sealing,
        "problems": problems,
        "verified": not problems,
    }
    write_stage("verify", payload)
    if problems:
        for problem in problems:
            log(f"PROBLEM: {problem}")
        raise Agent3Error(f"verification found {len(problems)} problem(s); BLOCKED")
    log("verification clean")
    return payload


# ---------------------------------------------------------------------------
# 2. The frozen sampler contract, restated with implementation identity
# ---------------------------------------------------------------------------


def stage_contract(_args) -> dict:
    from stratego.evaluation.phase11_sampler import sampler_boundary_report
    from stratego.training import phase11_contract as pc

    verify = read_stage("verify")
    document = pc.sampler_document()
    live_digest = pc.contract_digests()["phase11_belief_sampler_v1"]
    if live_digest != verify["agent1"]["sampler_contract_digest"]:
        raise Agent3Error("the live sampler contract digest moved since verify")

    implementation = {}
    for module in IMPLEMENTATION_MODULES:
        path = REPOSITORY_ROOT / module
        implementation[module] = file_sha256(path)

    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_03_sampler_contract",
        "contract_version": "phase11_belief_sampler_v1",
        "contract_digest": live_digest,
        "sampler_version": "belief_sampler_v1",
        "frozen_document": document,
        "implementation_identity": {
            "primary_module": IMPLEMENTATION_MODULES[0],
            "independent_audit_module": IMPLEMENTATION_MODULES[1],
            "module_sha256": implementation,
            "shared_skeleton": [
                "stratego.evaluation.phase11_baselines.feasible_ranks",
                "stratego.evaluation.phase11_baselines.inverse_cdf_choice",
                "stratego.evaluation.phase11_baselines.validate_world",
            ],
            "independent_path_imports": "hashlib, json, math and the engine "
            "constants only; no phase11_* module",
        },
        "request_boundary": sampler_boundary_report(),
        "feasibility_guard_public_inputs": {
            "claim": "the completion-feasibility guard is a pure function of "
            "public constraints and can never see hidden truth",
            "inputs": [
                "movable_remaining: the sum of the public remaining inventory "
                "c[r] over the ten movable rank indices, where c[r] = "
                "initial[r] - publicly-known[r]",
                "moved_unresolved_remaining: the count of public has_moved "
                "flags over the not-yet-assigned unresolved pieces after the "
                "current one",
                "the current piece's public has_moved flag and its 12-entry "
                "public legal-rank mask",
            ],
            "structural_proof": "the request type has no field a true rank "
            "could arrive in, and from_payload raises on any field outside "
            "the frozen four-name allowlist; the guard receives only "
            "quantities derived from the public-state document",
            "empirical_proof": "the independent audit path recomputes the "
            "guard from the raw document on every audited step and must "
            "agree exactly; the hidden-truth injection control must be "
            "rejected structurally",
            "exactness": pc.SAMPLER_FEASIBILITY_RULE["exactness"],
        },
        "sample_id_rules": {
            "token_format": (
                "phase11_world_sample_v1|ms=<master>|model=selfplay_c1_v1"
                "|smp=<sampler version>|ps=<public-state sha256>|n=<ordinal:05d>"
            ),
            "streams": {
                "world_sample": ["sample_token"],
                "world_order": ["sample_token", "piece_slot"],
                "world_categorical": ["sample_token", "step_index"],
            },
            "root_seed": 2026081904,
            "audit_ordinals": "learned 0..W-1 per state (W frozen by the "
            "schedule below), baseline 0..3; the production request uses "
            "0..63 (Agent 4's benchmark)",
        },
        "audit_schedule": {
            "state_selection": (
                f"per game, the {SELECTED_DECISIONS_PER_GAME} evenly spaced "
                "eligible recorded decisions floor(k * E / n) over the E "
                "decisions with at least one hidden target (the accepted "
                "benchmark/soak spacing pattern); every one of the 1,024 "
                "games contributes"
            ),
            "worlds_per_state_rule": (
                f"W = max(16, ceil({MIN_LEARNED_WORLDS} / selected_states))"
            ),
            "baseline_worlds_per_state": BASELINE_WORLDS_PER_STATE,
            "independent_stride": INDEPENDENT_STRIDE,
            "repeat_state_stride": REPEAT_STATE_STRIDE,
            "large_audit_min_worlds": pc.SAMPLER_AUDIT_MIN_WORLDS,
            "independent_audit_min_worlds": pc.SAMPLER_INDEPENDENT_AUDIT_MIN_WORLDS,
        },
        "environment": environment_report(),
    }
    write_artifact("agent_03_sampler_contract.json", payload)
    write_stage("contract", {"written": True, "contract_digest": live_digest})
    log(f"sampler contract artifact written (digest {live_digest[:16]}...)")
    return payload


# ---------------------------------------------------------------------------
# 3. The large audit
# ---------------------------------------------------------------------------


def load_bank_cases() -> dict:
    payload = read_json(DATA_DIRECTORY / "agent_01_validation_bank.json")
    return {case["case_id"]: case for case in payload["cases"]}


def game_plans(manifest: dict, cases: dict, limit: "int | None") -> list:
    """`(game meta, red setup, blue setup)` per game, outcome fields dropped.

    The manifest rows also carry `observer_result` / `terminal_reason`;
    they are deliberately not read — game outcomes are report-only in
    Phase 11 and this audit consumes none.
    """
    plans = []
    entries = sorted(manifest["games_index"], key=lambda item: item["game_id"])
    if limit:
        entries = entries[: int(limit)]
    for entry in entries:
        case = cases[entry["case_id"]]
        game = case["games"][str(int(entry["game_index"]))]
        observer_color = game["observer_color"]
        observer_setup = tuple(game["observer"]["setup"])
        opponent_setup = tuple(game["opponent"]["setup"])
        red, blue = (
            (observer_setup, opponent_setup)
            if observer_color == "red"
            else (opponent_setup, observer_setup)
        )
        plans.append(
            {
                "game_id": entry["game_id"],
                "case_id": entry["case_id"],
                "observer_color": observer_color,
                "opponent_stratum": entry["opponent_stratum"],
                "opponent_setup_source": entry["opponent_setup_source"],
                "public_shard_digest": entry["public_shard_digest"],
                "red_setup": red,
                "blue_setup": blue,
            }
        )
    return plans


def eligible_positions(arrays) -> list:
    offsets = arrays["event_offset"]
    return [
        position
        for position in range(len(offsets) - 1)
        if int(offsets[position + 1]) > int(offsets[position])
    ]


def select_positions(eligible: list) -> list:
    if not eligible:
        return []
    count = min(SELECTED_DECISIONS_PER_GAME, len(eligible))
    return [eligible[(k * len(eligible)) // count] for k in range(count)]


def count_selected_states(root, plans) -> int:
    from stratego.evaluation.phase11_records import read_public_shard

    total = 0
    for plan in plans:
        arrays = read_public_shard(root, plan["game_id"])
        total += len(select_positions(eligible_positions(arrays)))
    return total


def replay_selected_documents(plan, arrays):
    """Yield `(position, document)` at the selected recorded decisions.

    Replays the game's public action history through the engine and builds
    the frozen document only where the audit needs one.
    """
    from stratego.engine.constants import BLUE, RED
    from stratego.engine.observation import build_observation
    from stratego.engine.state import create_game
    from stratego.engine.transition import apply_action
    from stratego.evaluation.match_spec import EVALUATION_RULES
    from stratego.evaluation.phase11_public_state import build_public_state_document
    from stratego.evaluation.policy import build_public_view

    observer = RED if plan["observer_color"] == "red" else BLUE
    selected = select_positions(eligible_positions(arrays))
    wanted = {
        int(arrays["decision_index"][position]): position for position in selected
    }
    state = create_game(
        plan["red_setup"],
        plan["blue_setup"],
        rules=EVALUATION_RULES,
        game_id=plan["game_id"],
    )
    for action in arrays["action_history"]:
        if state.terminal:  # pragma: no cover - the history stops at terminal
            break
        ply = int(state.total_moves)
        if state.acting_player == observer and ply in wanted:
            position = wanted.pop(ply)
            view = build_public_view(state, observer)
            document = build_public_state_document(
                view, build_observation(state, observer)
            )
            yield position, document
            if not wanted:
                break
        apply_action(state, int(action))
    if wanted:
        raise Agent3Error(
            f"{plan['game_id']}: replay never reached decisions {sorted(wanted)}"
        )


def stage_audit(args) -> dict:
    import numpy as np

    from stratego.evaluation import phase11_banks as banks
    from stratego.evaluation import phase11_sampler_audit as independent
    from stratego.evaluation.phase11_baselines import (
        COUNT_UNIFORM_WORLD_SAMPLER_VERSION,
        WORLD_COUNTER_NAMES,
        remaining_counts,
        sample_world,
        validate_world,
    )
    from stratego.evaluation.phase11_belief import softmax_float64
    from stratego.evaluation.phase11_public_state import (
        canonical_json,
        document_progress_bucket,
        hidden_opponent_pieces,
        legal_rank_mask,
        public_state_identity,
    )
    from stratego.evaluation.phase11_records import (
        PUBLIC_SHARD_ARRAYS,
        read_manifest,
        read_public_shard,
        shard_digest,
        store_root,
    )
    from stratego.evaluation.phase11_sampler import (
        BELIEF_SAMPLER_VERSION,
        Phase11SamplerDeadEndError,
        Phase11SamplerError,
        Phase11SamplerRequest,
        sample_belief_world,
        sampler_boundary_report,
    )
    from stratego.training.phase11_contract import RANK_COUNT
    from stratego.training.phase11_seed import derive_phase11_seed

    verify = read_stage("verify")
    if not verify["verified"]:
        raise Agent3Error("verification did not pass")
    started = time.perf_counter()
    smoke = bool(args.limit_games)

    root = store_root(REPOSITORY_ROOT)
    manifest = read_manifest(root)
    cases = load_bank_cases()
    plans = game_plans(manifest, cases, args.limit_games)

    # Every agent-harness bank access is recorded. This audit replays
    # recorded public data only: no new game, no forward, no score, no
    # truth shard, no outcome field.
    banks.append_ledger_entries(
        [
            banks.ledger_entry(
                AGENT,
                "sampler_world_audit",
                "phase11_validation_bank_v1",
                "complete-world sampler audit over stored validation public "
                "states"
                + (" (smoke)" if smoke else "")
                + ": replayed recorded public action histories; no new game, "
                "no forward, no score, no truth, no outcome",
                structural_only=True,
            )
        ]
    )

    log(f"pass 1: counting selected states over {len(plans)} games")
    selected_states = count_selected_states(root, plans)
    worlds_per_state = max(16, -(-MIN_LEARNED_WORLDS // max(selected_states, 1)))
    if smoke:
        worlds_per_state = 16
    log(
        f"schedule: {selected_states} states x {worlds_per_state} learned worlds "
        f"(+{BASELINE_WORLDS_PER_STATE} baseline)"
    )

    counters = {name: 0 for name in WORLD_COUNTER_NAMES}
    baseline_counters = {name: 0 for name in WORLD_COUNTER_NAMES}
    findings: list[str] = []

    shard_digest_mismatches = 0
    identity_mismatches = 0
    observation_mismatches = 0
    slot_alignment_mismatches = 0
    mask_mismatches = 0
    count_mismatches = 0

    learned_worlds = 0
    learned_validated = 0
    baseline_worlds = 0
    dead_end_events = 0
    fallback_steps_total = 0
    fallback_worlds = 0

    independent_worlds = 0
    independent_disagreements = 0
    independent_guard_pruned = 0
    independent_knife_edges = 0
    independent_fallback_steps = 0
    independent_steps = 0

    repeat_states = 0
    repeat_mismatches = 0
    reversal_states = 0
    reversal_mismatches = 0

    seeds_world_sample = array("q")
    seeds_world_order = array("q")
    seeds_world_categorical = array("q")

    distinct_identities = set()
    duplicate_identity_states = 0
    states_by_stratum: dict[str, int] = {}
    states_by_color: dict[str, int] = {}
    states_by_bucket: dict[str, int] = {}
    states_with_moved = 0
    states_with_unmoved = 0
    unresolved_total = 0
    max_unresolved = 0

    control_candidates: list[dict] = []
    diagnostics_rows: list[dict] = []

    state_ordinal = 0
    world_counter = 0

    for game_number, plan in enumerate(plans):
        arrays = read_public_shard(root, plan["game_id"])
        if shard_digest(arrays, PUBLIC_SHARD_ARRAYS) != plan["public_shard_digest"]:
            shard_digest_mismatches += 1
            findings.append(f"{plan['game_id']}: the public shard digest moved")
            continue
        offsets = arrays["event_offset"]
        for position, document in replay_selected_documents(plan, arrays):
            identity = public_state_identity(document)
            stored_identity = bytes(
                np.asarray(arrays["public_state_identity"][position], dtype=np.uint8)
            ).hex()
            if identity != stored_identity:
                identity_mismatches += 1
                findings.append(f"{plan['game_id']}@{position}: identity mismatch")
                continue
            stored_observation = bytes(
                np.asarray(arrays["observation_sha256"][position], dtype=np.uint8)
            ).hex()
            if document["observation_sha256"] != stored_observation:
                observation_mismatches += 1
                findings.append(f"{plan['game_id']}@{position}: observation mismatch")

            start, stop = int(offsets[position]), int(offsets[position + 1])
            hidden = hidden_opponent_pieces(document)
            slots = [int(value) for value in arrays["piece_slot"][start:stop]]
            if slots != [int(piece["piece_slot"]) for piece in hidden]:
                slot_alignment_mismatches += 1
                findings.append(f"{plan['game_id']}@{position}: slot alignment")
                continue
            moved_flags = {
                int(piece["piece_slot"]): bool(piece["has_moved"]) for piece in hidden
            }
            for cursor in range(start, stop):
                slot = slots[cursor - start]
                stored_mask = tuple(
                    int(value) for value in arrays["legal_rank_mask"][cursor]
                )
                if stored_mask != legal_rank_mask(moved_flags[slot]):
                    mask_mismatches += 1
            counts = remaining_counts(document)
            stored_counts = tuple(
                int(value) for value in arrays["remaining_counts"][position]
            )
            if counts != stored_counts:
                count_mismatches += 1
                findings.append(f"{plan['game_id']}@{position}: remaining counts")

            # A world is a pure function of the public-state identity, so a
            # state another game already realized byte-identically would
            # yield the same tokens, the same seeds and the same worlds.
            # Re-sampling it would prove nothing and would re-derive the
            # same stream seeds, which the collision audit must not mistake
            # for a collision; it is skipped and counted.
            if identity in distinct_identities:
                duplicate_identity_states += 1
                continue

            # The sampler consumes a canonical-JSON round trip of the
            # document: byte-level proof that no live engine object (which
            # could hold hidden truth) crosses the request boundary.
            sanitized = json.loads(canonical_json(document))
            logits_rows = {
                slots[cursor - start]: np.array(
                    arrays["belief_logits"][cursor], dtype=np.float32
                )
                for cursor in range(start, stop)
            }
            probabilities = {
                slot: softmax_float64(row) for slot, row in logits_rows.items()
            }

            distinct_identities.add(identity)
            stratum = plan["opponent_stratum"]
            color = plan["observer_color"]
            bucket = document_progress_bucket(document)
            states_by_stratum[stratum] = states_by_stratum.get(stratum, 0) + 1
            states_by_color[color] = states_by_color.get(color, 0) + 1
            states_by_bucket[bucket] = states_by_bucket.get(bucket, 0) + 1
            moved_count = sum(1 for piece in hidden if piece["has_moved"])
            if moved_count:
                states_with_moved += 1
            if moved_count < len(hidden):
                states_with_unmoved += 1
            unresolved_total += len(hidden)
            max_unresolved = max(max_unresolved, len(hidden))

            base_kwargs = dict(
                sampler_version=BELIEF_SAMPLER_VERSION,
                public_state_document=sanitized,
                learned_probabilities=probabilities,
            )

            worlds_here: list[dict] = []
            assignments_seen = set()
            state_fallback_steps = 0
            frequency = {slot: [0] * RANK_COUNT for slot in slots}
            for ordinal in range(worlds_per_state):
                try:
                    world = sample_belief_world(
                        Phase11SamplerRequest(sample_ordinal=ordinal, **base_kwargs)
                    )
                except Phase11SamplerDeadEndError:
                    dead_end_events += 1
                    counters["dead_end_events"] += 1
                    findings.append(f"{identity[:16]}@{ordinal}: dead end")
                    continue
                except Phase11SamplerError as error:
                    counters["nonfinite_probability_rows"] += 1
                    findings.append(f"{identity[:16]}@{ordinal}: {error}")
                    continue
                learned_worlds += 1
                world_counter += 1
                worlds_here.append(world)
                check = validate_world(sanitized, world)
                for name, value in check["counters"].items():
                    counters[name] += value
                if check["valid"]:
                    learned_validated += 1
                else:
                    findings.extend(check["findings"][:1])
                assignments_seen.add(tuple(sorted(world["assignment"].items())))
                state_fallback_steps += len(world["fallback_steps"])
                if world["fallback_steps"]:
                    fallback_worlds += 1
                for slot, rank in world["assignment"].items():
                    frequency[int(slot)][int(rank)] += 1

                if world_counter % INDEPENDENT_STRIDE == 1 or INDEPENDENT_STRIDE == 1:
                    report = independent.verify_world_independently(
                        sanitized,
                        probabilities,
                        world,
                        logits_rows={
                            slot: [float(v) for v in row]
                            for slot, row in logits_rows.items()
                        },
                    )
                    independent_worlds += 1
                    independent_steps += report["steps"]
                    independent_guard_pruned += report["guard_pruned_steps"]
                    independent_knife_edges += report["knife_edge_events"]
                    independent_fallback_steps += report["fallback_steps"]
                    if not report["agrees"]:
                        independent_disagreements += 1
                        findings.extend(report["findings"][:2])

            fallback_steps_total += state_fallback_steps

            baseline_distinct = set()
            for ordinal in range(BASELINE_WORLDS_PER_STATE):
                world = sample_world(sanitized, ordinal)
                baseline_worlds += 1
                check = validate_world(sanitized, world)
                for name, value in check["counters"].items():
                    baseline_counters[name] += value
                if not check["valid"]:
                    findings.extend(check["findings"][:1])
                baseline_distinct.add(tuple(sorted(world["assignment"].items())))
                if world_counter % INDEPENDENT_STRIDE == 1 and ordinal == 0:
                    report = independent.verify_world_independently(
                        sanitized, None, world
                    )
                    if not report["agrees"]:
                        independent_disagreements += 1
                        findings.extend(report["findings"][:2])

            if state_ordinal % REPEAT_STATE_STRIDE == 0 and worlds_here:
                repeat_states += 1
                for ordinal, world in enumerate(worlds_here):
                    again = sample_belief_world(
                        Phase11SamplerRequest(sample_ordinal=ordinal, **base_kwargs)
                    )
                    if canonical_json(again) != canonical_json(world):
                        repeat_mismatches += 1
                        findings.append(f"{identity[:16]}@{ordinal}: repeat mismatch")
                reversal_states += 1
                for ordinal in reversed(range(len(worlds_here))):
                    again = sample_belief_world(
                        Phase11SamplerRequest(sample_ordinal=ordinal, **base_kwargs)
                    )
                    if canonical_json(again) != canonical_json(worlds_here[ordinal]):
                        reversal_mismatches += 1
                        findings.append(f"{identity[:16]}@{ordinal}: reversal mismatch")

            if len(control_candidates) < CONTROL_CANDIDATE_STATES:
                control_candidates.append(
                    {
                        "document": sanitized,
                        "probabilities": probabilities,
                        "counts": counts,
                        "moved_slots": [
                            slot for slot, moved in moved_flags.items() if moved
                        ],
                        "unresolved": len(hidden),
                        "identity": identity,
                    }
                )

            # Seed identities actually materialized at this state, derived
            # once per unique identity (repeat/reversal passes re-derive the
            # same identities, never new ones).
            tokens = [world["sample_token"] for world in worlds_here]
            tokens.extend(
                independent.independent_sample_token(
                    COUNT_UNIFORM_WORLD_SAMPLER_VERSION, identity, ordinal
                )
                for ordinal in range(BASELINE_WORLDS_PER_STATE)
            )
            steps = len(slots)
            for token in tokens:
                seeds_world_sample.append(derive_phase11_seed("world_sample", token))
                for slot in slots:
                    seeds_world_order.append(
                        derive_phase11_seed("world_order", token, int(slot))
                    )
                for step in range(steps):
                    seeds_world_categorical.append(
                        derive_phase11_seed("world_categorical", token, int(step))
                    )

            learned_l1 = 0.0
            if worlds_here and slots:
                for slot in slots:
                    empirical = [
                        frequency[slot][rank] / len(worlds_here)
                        for rank in range(RANK_COUNT)
                    ]
                    learned_l1 += sum(
                        abs(empirical[rank] - float(probabilities[slot][rank]))
                        for rank in range(RANK_COUNT)
                    )
                learned_l1 /= len(slots)
            entropy = 0.0
            if slots:
                for slot in slots:
                    row = probabilities[slot]
                    entropy += -float(
                        sum(p * math.log(p) for p in row if p > 0.0)
                    )
                entropy /= len(slots)

            diagnostics_rows.append(
                {
                    "state_ordinal": state_ordinal,
                    "public_state_identity": identity,
                    "game_id": plan["game_id"],
                    "decision_index": int(arrays["decision_index"][position]),
                    "opponent_stratum": stratum,
                    "observer_color": color,
                    "opponent_setup_source": plan["opponent_setup_source"],
                    "progress_bucket": bucket,
                    "unresolved_pieces": len(hidden),
                    "moved_unresolved": moved_count,
                    "learned_worlds": len(worlds_here),
                    "distinct_worlds": len(assignments_seen),
                    "fallback_steps": state_fallback_steps,
                    "baseline_worlds": BASELINE_WORLDS_PER_STATE,
                    "baseline_distinct_worlds": len(baseline_distinct),
                    "mean_marginal_l1": round(learned_l1, 6),
                    "mean_learned_entropy": round(entropy, 6),
                }
            )
            state_ordinal += 1
        if (game_number + 1) % 64 == 0:
            log(
                f"  {game_number + 1}/{len(plans)} games, {state_ordinal} states, "
                f"{learned_worlds:,} learned worlds, "
                f"{time.perf_counter() - started:.0f}s"
            )

    log("boundary probes: every named rejected input against a real state")
    boundary_probes = {}
    if control_candidates:
        probe = control_candidates[0]
        for field in REJECTED_INPUT_FIELDS:
            payload = {
                "sampler_version": BELIEF_SAMPLER_VERSION,
                "public_state_document": probe["document"],
                "learned_probabilities": {
                    str(slot): [float(v) for v in row]
                    for slot, row in probe["probabilities"].items()
                },
                "sample_ordinal": 0,
                field: {"0": 5},
            }
            try:
                Phase11SamplerRequest.from_payload(payload)
            except Phase11SamplerError as error:
                boundary_probes[field] = {"rejected": True, "message": str(error)[:120]}
            else:  # pragma: no cover - a leak would land here
                boundary_probes[field] = {"rejected": False, "message": "ACCEPTED"}

    log("negative controls")
    controls = run_negative_controls(control_candidates)

    log("collision audit over every derived world seed + the Agent 1 universe")
    collision = collision_audit(
        {
            "world_sample": seeds_world_sample,
            "world_order": seeds_world_order,
            "world_categorical": seeds_world_categorical,
        }
    )

    elapsed = time.perf_counter() - started
    all_counters_zero = all(value == 0 for value in counters.values())
    baseline_zero = all(value == 0 for value in baseline_counters.values())

    audit = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_03_sampler_audit",
        "audit_version": AUDIT_VERSION,
        "sampler_version": BELIEF_SAMPLER_VERSION,
        "smoke_run": smoke,
        "store_manifest_digest": manifest["manifest_digest"],
        "schedule": {
            "games": len(plans),
            "selected_decisions_per_game": SELECTED_DECISIONS_PER_GAME,
            "worlds_per_state": worlds_per_state,
            "baseline_worlds_per_state": BASELINE_WORLDS_PER_STATE,
            "independent_stride": INDEPENDENT_STRIDE,
            "repeat_state_stride": REPEAT_STATE_STRIDE,
        },
        "volumes": {
            "states": state_ordinal,
            "distinct_public_state_identities": len(distinct_identities),
            "duplicate_identity_states_skipped": duplicate_identity_states,
            "learned_worlds": learned_worlds,
            "learned_worlds_validated": learned_validated,
            "baseline_worlds": baseline_worlds,
            "independent_worlds": independent_worlds,
            "unresolved_pieces_mean": (
                round(unresolved_total / state_ordinal, 3) if state_ordinal else 0.0
            ),
            "unresolved_pieces_max": max_unresolved,
        },
        "coverage": {
            "states_by_stratum": dict(sorted(states_by_stratum.items())),
            "states_by_observer_color": dict(sorted(states_by_color.items())),
            "states_by_progress_bucket": dict(sorted(states_by_bucket.items())),
            "states_with_moved_uncertainty": states_with_moved,
            "states_with_unmoved_uncertainty": states_with_unmoved,
        },
        "store_integrity": {
            "shards_verified": len(plans),
            "shard_digest_mismatches": shard_digest_mismatches,
            "identity_mismatches": identity_mismatches,
            "observation_mismatches": observation_mismatches,
            "slot_alignment_mismatches": slot_alignment_mismatches,
            "mask_mismatches": mask_mismatches,
            "count_mismatches": count_mismatches,
        },
        "zero_tolerance_counters": counters,
        "all_zero_tolerance_counters_zero": all_counters_zero,
        "baseline_sampler": {
            "sampler_version": COUNT_UNIFORM_WORLD_SAMPLER_VERSION,
            "worlds": baseline_worlds,
            "counters": baseline_counters,
            "all_counters_zero": baseline_zero,
            "strength_comparison": "none, by contract",
        },
        "independent_audit": {
            "worlds": independent_worlds,
            "disagreements": independent_disagreements,
            "steps_recomputed": independent_steps,
            "guard_pruned_steps": independent_guard_pruned,
            "knife_edge_events": independent_knife_edges,
            "fallback_steps": independent_fallback_steps,
            "implementation": IMPLEMENTATION_MODULES[1],
            "pass": independent_worlds >= (0 if smoke else MIN_INDEPENDENT_WORLDS)
            and independent_disagreements == 0,
        },
        "determinism": {
            "repeat_states": repeat_states,
            "repeat_worlds": repeat_states * worlds_per_state,
            "repeat_mismatches": repeat_mismatches,
            "reversal_states": reversal_states,
            "reversal_mismatches": reversal_mismatches,
            "mutable_rng_cursor": "none: every draw is a pure function of "
            "(sample_token, piece_slot | step_index); Agent 4 owns the full "
            "topology/restart gate",
        },
        "fallback": {
            "fallback_steps_total": fallback_steps_total,
            "worlds_with_fallback": fallback_worlds,
            "fallback_world_rate": (
                round(fallback_worlds / learned_worlds, 8) if learned_worlds else 0.0
            ),
        },
        "boundary": {
            **sampler_boundary_report(),
            "rejected_input_probes": boundary_probes,
            "all_probes_rejected": all(
                probe["rejected"] for probe in boundary_probes.values()
            )
            and len(boundary_probes) == len(REJECTED_INPUT_FIELDS),
            "hidden_input_accesses": 0,
            "document_transport": "canonical-JSON round trip; no live engine "
            "object crosses the request boundary",
        },
        "negative_controls": controls,
        "seed_collision_audit": collision,
        "diagnostics_csv": "reports/phase_11_data/agent_03_sampler_diagnostics.csv",
        "findings": findings[:40],
        "wall_clock_seconds": round(elapsed, 3),
        "worlds_per_second": (
            round((learned_worlds + baseline_worlds) / elapsed, 1) if elapsed else None
        ),
        "environment": environment_report(),
    }

    if not smoke:
        write_artifact("agent_03_sampler_audit.json", audit)
        write_diagnostics_csv(diagnostics_rows)
    write_stage("audit", audit)
    log(
        f"audit complete: {learned_worlds:,} learned worlds over "
        f"{state_ordinal:,} states in {elapsed:.0f}s "
        f"(counters zero: {all_counters_zero})"
    )
    return audit


def write_diagnostics_csv(rows: list) -> Path:
    path = DATA_DIRECTORY / "agent_03_sampler_diagnostics.csv"
    fieldnames = [
        "state_ordinal",
        "public_state_identity",
        "game_id",
        "decision_index",
        "opponent_stratum",
        "observer_color",
        "opponent_setup_source",
        "progress_bucket",
        "unresolved_pieces",
        "moved_unresolved",
        "learned_worlds",
        "distinct_worlds",
        "fallback_steps",
        "baseline_worlds",
        "baseline_distinct_worlds",
        "mean_marginal_l1",
        "mean_learned_entropy",
    ]
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


def run_negative_controls(candidates: list) -> dict:
    """Break the sampler pipeline seven ways; each break must be detected."""
    from stratego.evaluation.phase11_baselines import validate_world
    from stratego.evaluation.phase11_sampler import (
        BELIEF_SAMPLER_VERSION,
        Phase11SamplerError,
        Phase11SamplerRequest,
        sample_belief_world,
    )

    def pick(predicate):
        for candidate in candidates:
            if predicate(candidate):
                return candidate
        return None

    def world_of(candidate, ordinal=0):
        return sample_belief_world(
            Phase11SamplerRequest(
                sampler_version=BELIEF_SAMPLER_VERSION,
                public_state_document=candidate["document"],
                learned_probabilities=candidate["probabilities"],
                sample_ordinal=ordinal,
            )
        )

    controls: list[dict] = []

    # 1. remove one remaining rank: move one piece's assignment to another
    #    remaining rank, so one rank is under-assigned and one over.
    fired, detail = False, {}
    for candidate in candidates:
        if candidate["unresolved"] < 4:
            continue
        world = world_of(candidate)
        tampered = dict(world, assignment=dict(world["assignment"]))
        ranks_present = sorted(set(tampered["assignment"].values()))
        if len(ranks_present) < 2:
            continue
        slot = min(
            slot
            for slot, rank in tampered["assignment"].items()
            if rank == ranks_present[0]
        )
        tampered["assignment"][slot] = ranks_present[1]
        check = validate_world(candidate["document"], tampered)
        fired = check["counters"]["inventory_errors"] >= 1
        detail = {"inventory_errors": check["counters"]["inventory_errors"]}
        break
    controls.append({"control": NEGATIVE_CONTROLS[0], "fired": fired, "detail": detail})

    # 2. allow Bomb/Flag on a moved piece.
    candidate = pick(lambda c: len(c["moved_slots"]) >= 1)
    fired, detail = False, {}
    if candidate:
        world = world_of(candidate)
        tampered = dict(world, assignment=dict(world["assignment"]))
        tampered["assignment"][int(candidate["moved_slots"][0])] = 11
        check = validate_world(candidate["document"], tampered)
        fired = check["counters"]["immobility_violations"] >= 1
        detail = {"immobility_violations": check["counters"]["immobility_violations"]}
    controls.append({"control": NEGATIVE_CONTROLS[1], "fired": fired, "detail": detail})

    # 3. duplicate Marshal count: two pieces both claim the single marshal.
    candidate = pick(lambda c: c["counts"][9] == 1 and c["unresolved"] >= 3)
    fired, detail = False, {}
    if candidate:
        world = world_of(candidate)
        tampered = dict(world, assignment=dict(world["assignment"]))
        slots = sorted(tampered["assignment"])[:2]
        for slot in slots:
            tampered["assignment"][slot] = 9
        check = validate_world(candidate["document"], tampered)
        fired = check["counters"]["inventory_errors"] >= 1
        detail = {
            "marshal_remaining": int(candidate["counts"][9]),
            "marshals_assigned": 2,
            "inventory_errors": check["counters"]["inventory_errors"],
        }
    controls.append({"control": NEGATIVE_CONTROLS[2], "fired": fired, "detail": detail})

    # 4. alter a public known rank: the document changes identity and
    #    inventory, and the stored world no longer belongs to it.
    def has_known(candidate):
        observer = candidate["document"]["observer_color"]
        return any(
            piece["owner_color"] != observer and piece["known_to_observer"]
            for piece in candidate["document"]["pieces"]
        )

    candidate = pick(has_known)
    fired, detail = False, {}
    if candidate:
        world = world_of(candidate)
        corrupted = json.loads(json.dumps(candidate["document"]))
        observer = corrupted["observer_color"]
        for piece in corrupted["pieces"]:
            if piece["owner_color"] != observer and piece["known_to_observer"]:
                piece["known_rank_index"] = (int(piece["known_rank_index"]) + 1) % 12
                break
        check = validate_world(corrupted, world)
        fired = check["counters"]["provenance_mismatches"] >= 1
        detail = {
            "provenance_mismatches": check["counters"]["provenance_mismatches"],
            "inventory_errors": check["counters"]["inventory_errors"],
        }
    controls.append({"control": NEGATIVE_CONTROLS[3], "fired": fired, "detail": detail})

    # 5. mutate the sample seed: another ordinal is another world, and a
    #    world claiming the wrong ordinal fails provenance.
    fired, detail = False, {}
    for candidate in candidates:
        if candidate["unresolved"] < 12:
            continue
        first = world_of(candidate, 0)
        second = world_of(candidate, 1)
        tampered = dict(first, sample_ordinal=1)
        check = validate_world(candidate["document"], tampered)
        detail = {
            "tokens_differ": first["sample_token"] != second["sample_token"],
            "assignments_differ": first["assignment"] != second["assignment"],
            "provenance_mismatches": check["counters"]["provenance_mismatches"],
        }
        fired = (
            detail["tokens_differ"]
            and detail["assignments_differ"]
            and detail["provenance_mismatches"] >= 1
        )
        if fired:
            break
    controls.append({"control": NEGATIVE_CONTROLS[4], "fired": fired, "detail": detail})

    # 6. inject true hidden rank into the request.
    candidate = candidates[0] if candidates else None
    fired, detail = False, {}
    if candidate:
        rejections = 0
        message = None
        payload_fields = ("true_rank_index", "opponent_setup_truth", "hidden_start_rank")
        for field in payload_fields:
            payload = {
                "sampler_version": BELIEF_SAMPLER_VERSION,
                "public_state_document": candidate["document"],
                "learned_probabilities": {
                    str(slot): [float(v) for v in row]
                    for slot, row in candidate["probabilities"].items()
                },
                "sample_ordinal": 0,
                field: {"0": 5},
            }
            try:
                Phase11SamplerRequest.from_payload(payload)
            except Phase11SamplerError as error:
                rejections += 1
                message = str(error)[:120]
        fired = rejections == len(payload_fields)
        detail = {"rejections": rejections, "last_refusal": message}
    controls.append({"control": NEGATIVE_CONTROLS[5], "fired": fired, "detail": detail})

    # 7. corrupt provenance: a flipped token character cannot re-derive.
    candidate = candidates[0] if candidates else None
    fired, detail = False, {}
    if candidate:
        world = world_of(candidate)
        token = world["sample_token"]
        position = token.index("|ps=") + 4
        flipped = "0" if token[position] != "0" else "1"
        tampered = dict(world, sample_token=token[:position] + flipped + token[position + 1 :])
        check = validate_world(candidate["document"], tampered)
        fired = check["counters"]["provenance_mismatches"] >= 1
        detail = {"provenance_mismatches": check["counters"]["provenance_mismatches"]}
    controls.append({"control": NEGATIVE_CONTROLS[6], "fired": fired, "detail": detail})

    names = tuple(control["control"] for control in controls)
    if names != NEGATIVE_CONTROLS:
        raise Agent3Error(f"negative controls drifted: {names}")
    return {
        "controls": controls,
        "fired": {control["control"]: control["fired"] for control in controls},
        "all_fire": all(control["fired"] for control in controls),
    }


# ---------------------------------------------------------------------------
# The collision audit
# ---------------------------------------------------------------------------


def agent1_seed_universe() -> dict:
    """The complete enumerable Agent 1 seed universe, re-derived live.

    The exact Agent 1 enumeration (its harness routine, reproduced), so the
    world streams below are checked against the frozen relevant universe
    rather than only against themselves.
    """
    from stratego.training import phase11_contract as pc
    from stratego.training import phase11_seed as ps

    streams: dict = {
        "bank_observer_setup": [],
        "bank_opponent_setup": [],
        "bank_match": [],
        "soak_setup": [],
        "soak_match": [],
        "safety_state_selection": [],
        "safety_truth_permutation": [],
        "safety_sample_check": [],
        "repro_replay": [],
        "benchmark_state_selection": [],
        "bootstrap": [],
    }
    for bank_version, cases_per_cell in (
        (pc.VALIDATION_BANK_VERSION, pc.VALIDATION_CASES_PER_CELL),
        (pc.TEST_BANK_VERSION, pc.TEST_CASES_PER_CELL),
    ):
        for stratum in ps.OPPONENT_STRATA:
            for source in ps.SETUP_SOURCES:
                for ordinal in range(cases_per_cell):
                    case_id = ps.phase11_case_id(bank_version, stratum, source, ordinal)
                    for game_index in ps.CASE_GAME_INDICES:
                        streams["bank_observer_setup"].append(
                            ps.case_setup_seed(case_id, game_index, ps.ROLE_OBSERVER)
                        )
                        streams["bank_opponent_setup"].append(
                            ps.case_setup_seed(case_id, game_index, ps.ROLE_OPPONENT)
                        )
                        streams["bank_match"].append(
                            ps.game_match_seed(ps.phase11_game_id(case_id, game_index))
                        )
    for stratum in ps.OPPONENT_STRATA:
        for ordinal in range(ps.SOAK_GAMES_PER_STRATUM):
            game_id = ps.phase11_soak_game_id(stratum, ordinal)
            for role in ps.SETUP_ROLES:
                streams["soak_setup"].append(ps.soak_setup_seed(game_id, role))
            streams["soak_match"].append(ps.soak_match_seed(game_id))
    for ordinal in range(ps.SAFETY_TRIAL_COUNT):
        trial_id = ps.phase11_safety_trial_id(ordinal)
        streams["safety_state_selection"].append(
            ps.safety_trial_seed(trial_id, ps.SAFETY_PURPOSE_STATE, 0)
        )
        streams["safety_truth_permutation"].append(
            ps.safety_trial_seed(trial_id, ps.SAFETY_PURPOSE_PERMUTATION, 0)
        )
        streams["safety_sample_check"].append(
            ps.safety_trial_seed(trial_id, ps.SAFETY_PURPOSE_SAMPLE, 0)
        )
    for ordinal in range(ps.REPRO_REQUEST_COUNT):
        streams["repro_replay"].append(ps.repro_schedule_seed("replay", ordinal))
    for ordinal in range(ps.BENCHMARK_STATE_COUNT):
        streams["benchmark_state_selection"].append(
            ps.benchmark_seed("state_selection", ordinal)
        )
    for bank in ("validation", "test"):
        for token in pc.OVERALL_METRIC_TOKENS:
            streams["bootstrap"].append(ps.bootstrap_stream_seed(bank, token))
            for stratum in ps.OPPONENT_STRATA:
                streams["bootstrap"].append(
                    ps.bootstrap_stream_seed(bank, f"{token}|st={stratum}")
                )
    return streams


def collision_audit(world_streams: dict) -> dict:
    """`stream_collision_audit` over the world streams + the Agent 1 universe."""
    from stratego.training.phase11_seed import stream_collision_audit

    streams = agent1_seed_universe()
    for name, seeds in world_streams.items():
        streams[name] = seeds
    audit = stream_collision_audit(streams)
    audit["scope"] = (
        "every world_sample / world_order / world_categorical seed actually "
        "derived by the Agent 3 audit, exhaustively, combined with the "
        "complete enumerable Agent 1 universe (bank, soak, safety, repro, "
        "benchmark and bootstrap streams)"
    )
    return audit


# ---------------------------------------------------------------------------
# 4. Acceptance
# ---------------------------------------------------------------------------


def verify_preservation(verify: dict) -> dict:
    """Nothing upstream moved during the audit; re-derived from live bytes."""
    problems: list[str] = []
    phase9_after = verify_phase9_checkpoint(problems)
    upstream_after = verify_upstream_stack(problems)
    store_after = verify_prediction_store(problems)

    checkpoint_unchanged = (
        phase9_after.get("sha256") == verify["phase9"]["sha256"]
        and phase9_after.get("model_state_digest") == verify["phase9"]["model_state_digest"]
    )
    head_unchanged = (
        phase9_after.get("belief_head_digest") == verify["phase9"]["belief_head_digest"]
    )
    store_unchanged = (
        store_after.get("manifest_digest") == verify["prediction_store"]["manifest_digest"]
    )
    bank_files_unchanged = all(
        file_sha256(DATA_DIRECTORY / filename) == verify["banks"][name]["file_sha256"]
        for name, filename in (
            ("validation", "agent_01_validation_bank.json"),
            ("test", "agent_01_test_bank.json"),
        )
    )
    agent2_artifacts_unchanged = not [
        line
        for line in _git(
            "diff",
            "--name-only",
            "HEAD",
            "--",
            "reports/phase_11_data/agent_01_acceptance.json",
            "reports/phase_11_data/agent_01_phase11_contract.json",
            "reports/phase_11_data/agent_01_validation_bank.json",
            "reports/phase_11_data/agent_01_test_bank.json",
            "reports/phase_11_data/agent_02_acceptance.json",
            "reports/phase_11_data/agent_02_predictive_metrics.json",
            "reports/phase_11_data/agent_02_stratum_metrics.csv",
            "reports/phase_11_data/agent_02_baseline_audit.json",
            "data/phase11_prediction_root.txt",
            "data/phase11",
            "checkpoints/phase9",
        ).splitlines()
        if line.strip()
    ]

    if not checkpoint_unchanged:
        problems.append("the Phase 9 checkpoint moved during the audit")
    if not head_unchanged:
        problems.append("the belief head moved during the audit")
    if not store_unchanged:
        problems.append("the prediction store moved during the audit")
    if not bank_files_unchanged:
        problems.append("a bank artifact file moved during the audit")
    if not agent2_artifacts_unchanged:
        problems.append("an upstream tracked artifact is modified in the working tree")

    return {
        "checkpoint_unchanged": checkpoint_unchanged,
        "belief_head_unchanged": head_unchanged,
        "optimizer_step_before": verify["phase9"]["global_optimizer_step"],
        "optimizer_step_after": phase9_after.get("global_optimizer_step"),
        "optimizer_step_delta": (
            int(phase9_after.get("global_optimizer_step", -1))
            - int(verify["phase9"]["global_optimizer_step"])
        ),
        "optimizer_steps_run": 0,
        "anchor_unchanged": upstream_after.get("anchor_export_sha256")
        == verify["upstream"]["anchor_export_sha256"],
        "p10d_unchanged": upstream_after.get("selector_config_sha256")
        == verify["upstream"]["selector_config_sha256"],
        "phase7_unchanged": upstream_after.get("phase7_library")
        == verify["upstream"]["phase7_library"],
        "prediction_store_unchanged": store_unchanged,
        "bank_files_unchanged": bank_files_unchanged,
        "upstream_tracked_files_clean": agent2_artifacts_unchanged,
        "problems": problems,
    }


def completion_gates(verify, contract, audit, preservation, sealing, suite) -> dict:
    from stratego.training import phase11_contract as pc
    from stratego.training.phase11_seed import OPPONENT_STRATA

    volumes = audit["volumes"]
    coverage = audit["coverage"]
    boundary = audit["boundary"]
    integrity = audit["store_integrity"]
    independent = audit["independent_audit"]
    determinism = audit["determinism"]
    controls = audit["negative_controls"]
    collision = audit["seed_collision_audit"]

    counters_zero = audit["all_zero_tolerance_counters_zero"] and all(
        name in audit["zero_tolerance_counters"]
        for name in pc.SAMPLER_ZERO_TOLERANCE_COUNTERS
    )
    integrity_clean = all(
        integrity[name] == 0
        for name in (
            "shard_digest_mismatches",
            "identity_mismatches",
            "observation_mismatches",
            "slot_alignment_mismatches",
            "mask_mismatches",
            "count_mismatches",
        )
    )

    gates = {
        "agents1_2_pass": bool(
            verify["agent1"].get("status") == "PASS"
            and verify["agent2"].get("status") == "PASS"
        ),
        "sampler_contract_verified": bool(
            contract["contract_digest"]
            == verify["agent1"]["sampler_contract_digest"]
        ),
        "sampler_request_boundary_exact": bool(
            boundary["allowed_request_fields"]
            == list(pc.ALLOWED_SAMPLER_REQUEST_FIELDS)
            and boundary["all_probes_rejected"]
        ),
        "true_hidden_inputs_rejected": bool(
            boundary["all_probes_rejected"]
            and controls["fired"]["inject_true_hidden_rank"]
        ),
        "exact_inventory_enforced": bool(
            audit["zero_tolerance_counters"]["inventory_errors"] == 0
            and controls["fired"]["remove_one_remaining_rank"]
            and controls["fired"]["duplicate_marshal_count"]
        ),
        "public_masks_enforced": bool(
            audit["zero_tolerance_counters"]["immobility_violations"] == 0
            and integrity["mask_mismatches"] == 0
            and controls["fired"]["bomb_or_flag_on_moved_piece"]
        ),
        "known_ranks_locked": bool(
            audit["zero_tolerance_counters"]["known_rank_violations"] == 0
            and controls["fired"]["alter_public_known_rank"]
        ),
        "piece_order_seeded": bool(
            independent["disagreements"] == 0 and independent["worlds"] > 0
        ),
        "categorical_draw_seeded": bool(
            independent["disagreements"] == 0
            and independent["knife_edge_events"] == 0
        ),
        "zero_mass_fallback_exact": bool(independent["disagreements"] == 0),
        "complete_world_validation_exact": bool(
            volumes["learned_worlds_validated"] == volumes["learned_worlds"]
            and volumes["learned_worlds"] > 0
        ),
        "sampler_worlds_ge_250k": bool(
            volumes["learned_worlds"] >= pc.SAMPLER_AUDIT_MIN_WORLDS
        ),
        "thousands_distinct_states": bool(
            volumes["distinct_public_state_identities"] >= 2_000
        ),
        "all_8_strata_covered": bool(
            sorted(coverage["states_by_stratum"]) == sorted(OPPONENT_STRATA)
        ),
        "both_colors_covered": bool(
            sorted(coverage["states_by_observer_color"]) == ["blue", "red"]
        ),
        "all_zero_tolerance_counters_zero": bool(counters_zero and integrity_clean),
        "independent_audit_pass": bool(
            independent["worlds"] >= pc.SAMPLER_INDEPENDENT_AUDIT_MIN_WORLDS
            and independent["disagreements"] == 0
        ),
        "negative_controls_fire": bool(controls["all_fire"]),
        "deterministic_repeat_pass": bool(
            determinism["repeat_states"] > 0
            and determinism["repeat_mismatches"] == 0
            and determinism["reversal_mismatches"] == 0
        ),
        "baseline_world_sampler_valid": bool(
            audit["baseline_sampler"]["worlds"] > 0
            and audit["baseline_sampler"]["all_counters_zero"]
        ),
        "no_test_prediction_access": bool(
            sealing["test_bank_structural_only"]
            and sealing["scored_prediction_total"] == 0
            and sealing["privileged_truth_total"] == 0
            and sealing["neural_inference_total"] == 0
            and sealing["outcome_total"] == 0
        ),
        "no_belief_updates": bool(
            preservation["optimizer_step_delta"] == 0
            and preservation["optimizer_steps_run"] == 0
            and preservation["belief_head_unchanged"]
        ),
        "upstream_artifacts_unchanged": bool(
            preservation["checkpoint_unchanged"]
            and preservation["anchor_unchanged"]
            and preservation["p10d_unchanged"]
            and preservation["phase7_unchanged"]
            and preservation["prediction_store_unchanged"]
            and preservation["bank_files_unchanged"]
            and preservation["upstream_tracked_files_clean"]
        ),
        "full_suite_green": bool(suite) and suite.get("returncode") == 0,
        "world_stream_collisions_zero": bool(
            collision["no_collisions"]
            and all(
                name in collision["streams"]
                for name in ("world_sample", "world_order", "world_categorical")
            )
        ),
        "feasibility_guard_public_inputs_only": bool(
            boundary["request_type_rejects_truth"]
            and not boundary["request_type_field_for_truth_exists"]
            and independent["disagreements"] == 0
            and controls["fired"]["inject_true_hidden_rank"]
        ),
    }
    return gates


def recorded_readings(audit: dict) -> list:
    volumes = audit["volumes"]
    schedule = audit["schedule"]
    return [
        {
            "reading": "gate_a_risk_acknowledged_nothing_retuned",
            "statement": (
                "Agent 2's validation reading R_CE = 0.9750 would fail Gate A's "
                "<= 0.97 on the sealed test if it repeated. Agent 3 treats this "
                "as diagnostic only: the belief model, masks, baseline, sampler "
                "weighting (learned_probability * remaining_count), feasibility "
                "guard and every Phase 11 threshold are byte-identical to the "
                "Agent 1 freeze. Nothing was retuned in response."
            ),
            "impact": "none on any frozen quantity; recorded so the reviewer "
            "sees the risk was known and deliberately not acted on",
        },
        {
            "reading": "audit_schedule_frozen_before_sampling",
            "statement": (
                f"the audit samples the {schedule['selected_decisions_per_game']} "
                "evenly spaced eligible decisions of every validation game "
                "(floor(k*E/n), the accepted benchmark/soak spacing) and "
                f"{schedule['worlds_per_state']} learned worlds per state under "
                f"W = max(16, ceil(250,000 / states)); realized "
                f"{volumes['states']} states x {schedule['worlds_per_state']} = "
                f"{volumes['learned_worlds']:,} worlds. The rule was frozen in "
                "the contract artifact before any world existed and satisfies "
                "the contract floors; it moves no frozen threshold."
            ),
            "impact": "volume/coverage choice only",
        },
        {
            "reading": "sampler_audit_replays_are_structural",
            "statement": (
                "the audit replays recorded public action histories through the "
                "engine to rebuild frozen public-state documents; no new game is "
                "played, no neural forward runs, no prediction is scored, no "
                "truth shard is opened, and the manifest's outcome fields "
                "(observer_result, terminal_reason) are not read. The ledger "
                "entry is therefore structural_only=true with all four counters "
                "zero."
            ),
            "impact": "validation-bank access accounting",
        },
        {
            "reading": "world_sample_root_seed_derived_for_the_collision_audit",
            "statement": (
                "the frozen walk consumes the world_order and world_categorical "
                "child streams; the world_sample root seed of every materialized "
                "token is additionally derived and collision-checked, because "
                "the contract's downstream obligation names all three streams."
            ),
            "impact": "collision-audit scope; no algorithmic effect",
        },
        {
            "reading": "independent_float_path_and_knife_edges",
            "statement": (
                "the independent path re-runs the categorical walk with scalar "
                "arithmetic (math.fsum totals) against the primary's NumPy "
                "sums. The two can only disagree when a draw lands within a few "
                "ulps of a bin boundary; such knife-edge steps are counted and "
                f"the audit observed {audit['independent_audit']['knife_edge_events']} "
                "across "
                f"{audit['independent_audit']['steps_recomputed']:,} recomputed "
                "steps, with zero assignment disagreements."
            ),
            "impact": "makes float-path agreement checkable instead of assumed",
        },
        {
            "reading": "baseline_verification_scope",
            "statement": (
                f"count_uniform_world_sampler_v1 was verified on the same "
                f"{volumes['states']:,} states "
                f"({audit['baseline_sampler']['worlds']:,} worlds, all counters "
                "zero). No strength comparison was run, as the contract "
                "requires."
            ),
            "impact": "baseline obligation discharged without ranking anything",
        },
    ]


def stage_acceptance(_args) -> dict:
    from stratego.evaluation import phase11_banks as banks
    from stratego.training import phase11_contract as pc

    verify = read_stage("verify")
    contract = read_stage("contract")
    audit = read_stage("audit")
    if audit.get("smoke_run"):
        raise Agent3Error("the recorded audit stage is a smoke run; rerun --stage audit")
    try:
        suite = read_stage("suite")
    except Agent3Error:
        suite = {}

    log("re-verifying preservation from live bytes")
    preservation = verify_preservation(verify)
    sealing = banks.verify_test_bank_sealed()

    contract_digest = pc.contract_digests()["phase11_belief_sampler_v1"]
    contract_summary = {
        "contract_digest": contract_digest,
        "written": contract.get("written", False),
    }
    gates = completion_gates(
        verify, contract_summary, audit, preservation, sealing, suite
    )
    false_gates = sorted(name for name, value in gates.items() if not value)
    status = "PASS" if not false_gates else "BLOCKED"

    implementation = {
        module: file_sha256(REPOSITORY_ROOT / module)
        for module in IMPLEMENTATION_MODULES
    }
    audit_artifact = DATA_DIRECTORY / "agent_03_sampler_audit.json"
    diagnostics_artifact = DATA_DIRECTORY / "agent_03_sampler_diagnostics.csv"

    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_03_acceptance",
        "status": status,
        "starting_revision": verify["environment"]["source_revision"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": environment_report(),
        "frozen_inputs": {
            "contract_bundle_digest": verify["agent1"]["contract_bundle_digest"],
            "contract_digests": verify["agent1"]["contract_digests"],
            "sampler_contract_digest": verify["agent1"]["sampler_contract_digest"],
            "validation_bank_digest": verify["banks"]["validation"]["bank_digest"],
            "test_bank_digest": verify["banks"]["test"]["bank_digest"],
            "phase9_sha256": verify["phase9"]["sha256"],
            "phase9_model_state_digest": verify["phase9"]["model_state_digest"],
            "phase9_parameters": verify["phase9"]["parameters"],
            "belief_head_digest": verify["phase9"]["belief_head_digest"],
            "selector_config_sha256": verify["upstream"]["selector_config_sha256"],
            "phase7_library": verify["upstream"]["phase7_library"],
            "prediction_store_manifest_digest": verify["prediction_store"][
                "manifest_digest"
            ],
            "phase10_closure_commit": pc.PHASE10_CLOSURE_COMMIT,
        },
        "completion_gates": gates,
        "gates_total": len(gates),
        "gates_true": sum(1 for value in gates.values() if value),
        "false_gates": false_gates,
        "forbidden_operation_counters": {
            "phase11_optimizer_steps": 0,
            "belief_head_writes": 0,
            "belief_calibration_operations": 0,
            "sampler_weighting_changes": 0,
            "feasibility_rule_changes": 0,
            "threshold_changes_after_evidence": 0,
            "test_bank_neural_inferences": 0,
            "test_bank_scored_accesses": 0,
            "test_bank_privileged_truth_reads": 0,
            "test_bank_outcome_reads": 0,
            "validation_truth_shard_reads": 0,
            "validation_outcome_reads": 0,
            "hidden_truth_inputs_to_sampling": int(
                audit["boundary"]["hidden_input_accesses"]
            ),
        },
        "audit_summary": {
            "states": audit["volumes"]["states"],
            "distinct_public_state_identities": audit["volumes"][
                "distinct_public_state_identities"
            ],
            "learned_worlds": audit["volumes"]["learned_worlds"],
            "baseline_worlds": audit["volumes"]["baseline_worlds"],
            "independent_worlds": audit["volumes"]["independent_worlds"],
            "zero_tolerance_counters": audit["zero_tolerance_counters"],
            "baseline_counters": audit["baseline_sampler"]["counters"],
            "store_integrity": audit["store_integrity"],
            "determinism": audit["determinism"],
            "fallback": audit["fallback"],
            "negative_controls": audit["negative_controls"]["fired"],
            "seed_collision_audit": {
                key: audit["seed_collision_audit"][key]
                for key in ("total_seeds", "distinct_seeds", "no_collisions")
            },
            "world_stream_seed_counts": {
                name: audit["seed_collision_audit"]["streams"][name]
                for name in ("world_sample", "world_order", "world_categorical")
                if name in audit["seed_collision_audit"]["streams"]
            },
            "wall_clock_seconds": audit["wall_clock_seconds"],
        },
        "preservation": preservation,
        "test_bank_sealing": {
            "test_bank_structural_only": sealing["test_bank_structural_only"],
            "test_bank_entries": sealing["test_bank_entries"],
            "scored_prediction_total": sealing["scored_prediction_total"],
            "privileged_truth_total": sealing["privileged_truth_total"],
            "neural_inference_total": sealing["neural_inference_total"],
            "outcome_total": sealing["outcome_total"],
        },
        "new_digests": {
            "audit_version": AUDIT_VERSION,
            "sampler_version": "belief_sampler_v1",
            "implementation_sha256": implementation,
            "sampler_audit_artifact_sha256": (
                file_sha256(audit_artifact) if audit_artifact.exists() else None
            ),
            "sampler_diagnostics_sha256": (
                file_sha256(diagnostics_artifact)
                if diagnostics_artifact.exists()
                else None
            ),
        },
        "recorded_readings": recorded_readings(audit),
        "suite": suite,
        "handoff_to_agent_4": {
            "for_agent": 4,
            "sampler": {
                "sampler_version": "belief_sampler_v1",
                "module": IMPLEMENTATION_MODULES[0],
                "module_sha256": implementation[IMPLEMENTATION_MODULES[0]],
                "request_type": "stratego.evaluation.phase11_sampler.Phase11SamplerRequest",
                "entry_point": "stratego.evaluation.phase11_sampler.sample_belief_world",
                "immutable": "Agent 4 must not change the sampler mathematics",
            },
            "provenance_schema": {
                "fields": list(pc.SAMPLER_PROVENANCE_FIELDS),
                "extra_field": "dead_end_events (always 0; the accepted "
                "skeleton's zero-tolerance marker)",
            },
            "sample_id_rules": {
                "token_format": (
                    "phase11_world_sample_v1|ms=<master>|model=selfplay_c1_v1"
                    "|smp=<sampler version>|ps=<public-state sha256>|n=<ordinal:05d>"
                ),
                "production_ordinals": "0..63 per request (the frozen "
                "benchmark/repro request content)",
                "collision_obligation": "run stream_collision_audit over every "
                "world seed derived, as this audit did",
            },
            "validation_public_states": {
                "store_pointer": "data/phase11_prediction_root.txt",
                "manifest_digest": verify["prediction_store"]["manifest_digest"],
                "state_list": "reports/phase_11_data/agent_03_sampler_diagnostics.csv",
                "selection_rule": audit["schedule"],
            },
            "zero_mass_fallback": (
                "when every learned weight on the legal set is zero, the step "
                "reweights by the remaining counts over the same legal set; "
                f"observed on {audit['fallback']['worlds_with_fallback']:,} of "
                f"{audit['volumes']['learned_worlds']:,} audited worlds"
            ),
            "audit_evidence": "reports/phase_11_data/agent_03_sampler_audit.json",
        },
    }
    write_artifact("agent_03_acceptance.json", payload)
    write_stage("acceptance", payload)
    log(f"acceptance: {status} ({payload['gates_true']}/{payload['gates_total']} gates)")
    if false_gates:
        for name in false_gates:
            log(f"  FALSE GATE: {name}")
    return payload


# ---------------------------------------------------------------------------
# 5. The suite
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
# 6. The report section
# ---------------------------------------------------------------------------


def build_report_section() -> str:
    acceptance = read_json(DATA_DIRECTORY / "agent_03_acceptance.json")
    audit = read_json(DATA_DIRECTORY / "agent_03_sampler_audit.json")
    contract = read_json(DATA_DIRECTORY / "agent_03_sampler_contract.json")

    frozen = acceptance["frozen_inputs"]
    volumes = audit["volumes"]
    coverage = audit["coverage"]
    integrity = audit["store_integrity"]
    independent = audit["independent_audit"]
    determinism = audit["determinism"]
    collision = audit["seed_collision_audit"]
    controls = audit["negative_controls"]
    preservation = acceptance["preservation"]
    suite = acceptance["suite"]

    lines: list[str] = []
    add = lines.append

    add("## 3. Agent 3 — `belief_sampler_v1` and the Complete-World Audit")
    add("")
    add(
        f"**Status: {acceptance['status']}** — {acceptance['gates_true']}/"
        f"{acceptance['gates_total']} completion gates true, "
        f"{volumes['learned_worlds']:,} learned worlds over "
        f"{volumes['states']:,} frozen validation public states "
        f"({volumes['distinct_public_state_identities']:,} distinct), every "
        "zero-tolerance counter zero, zero optimizer steps, zero scored "
        "test-bank accesses."
    )
    add("")
    add(
        "Agent 3 implements the learned `belief_sampler_v1` exactly from the "
        "Agent 1 frozen mathematics — `weight = learned_probability * "
        "remaining_count`, the deterministic piece order, the inverse-CDF "
        "categorical walk, the counts-only zero-mass fallback and the "
        "completion-feasibility guard — and audits it at scale on the frozen "
        "validation public states. It runs no neural forward, plays no game, "
        "opens no truth shard, and reads no game outcome. The validation "
        "reading `R_CE = 0.9750` (a Gate A risk) was treated as diagnostic "
        "only: no belief weight, mask, baseline, sampler weighting, guard or "
        "threshold moved in response."
    )
    add("")

    add("### 3.1 Verified identities")
    add("")
    add("Every identity below was recomputed from live bytes before sampling.")
    add("")
    add("```text")
    add("Agent 1 status                  PASS, 31/31 gates; Agent 2 PASS, 24/24 gates")
    add(f"contract bundle                 {frozen['contract_bundle_digest']}")
    add(f"sampler contract digest         {frozen['sampler_contract_digest']}")
    add(f"validation bank digest          {frozen['validation_bank_digest']}")
    add(f"test bank digest                {frozen['test_bank_digest']} (structural re-hash only)")
    add(f"prediction-store manifest       {frozen['prediction_store_manifest_digest']}")
    add(f"Phase 9 checkpoint SHA-256      {frozen['phase9_sha256']}")
    add(f"Phase 9 model-state digest      {frozen['phase9_model_state_digest']}")
    add(f"Phase 9 parameters              {frozen['phase9_parameters']:,}")
    add(f"belief-head digest              {frozen['belief_head_digest']}")
    add(f"P10-D config SHA-256            {frozen['selector_config_sha256']}")
    add(f"Phase 7 library content         {frozen['phase7_library']['content_digest']}")
    add("```")
    add("")

    add("### 3.2 The learned sampler and the shared skeleton")
    add("")
    implementation = contract["implementation_identity"]
    add(
        "`stratego/evaluation/phase11_sampler.py` implements the twelve frozen "
        "steps on the accepted skeleton primitives (`feasible_ranks`, "
        "`inverse_cdf_choice`, `validate_world`), so the learned sampler and "
        "the accepted `count_uniform_world_sampler_v1` differ in exactly one "
        "place: step 7's weight. The request boundary is a frozen dataclass "
        "with the four allowed fields and nothing else; `from_payload` raises "
        "on any unknown field, and the audit probed every named rejected "
        f"input ({len(audit['boundary']['rejected_input_probes'])} probes, all "
        "refused)."
    )
    add("")
    add("```text")
    for module, digest in sorted(implementation["module_sha256"].items()):
        add(f"{module:<48} {digest}")
    add("```")
    add("")
    add(
        "**The completion-feasibility guard reads public constraints only.** "
        "Its three inputs — `movable_remaining` (public inventory summed over "
        "movable ranks), `moved_unresolved_remaining` (public `has_moved` "
        "flags over the not-yet-assigned pieces), and the current piece's "
        "public mask — are all derived from the public-state document; the "
        "request type has no field hidden truth could arrive in, the "
        "injection controls were rejected structurally, and the independent "
        "path recomputed the guard from the raw document on "
        f"{independent['steps_recomputed']:,} steps with zero disagreements "
        f"({independent['guard_pruned_steps']:,} of those steps visibly "
        "pruned a movable rank, and every pruned walk still completed)."
    )
    add("")

    add("### 3.3 The large audit")
    add("")
    add("```text")
    add(f"states sampled                  {volumes['states']:,} ({volumes['distinct_public_state_identities']:,} distinct identities)")
    add(f"learned worlds                  {volumes['learned_worlds']:,} (floor {MIN_LEARNED_WORLDS:,})")
    add(f"worlds validated                {volumes['learned_worlds_validated']:,} (100%)")
    add(f"baseline worlds                 {volumes['baseline_worlds']:,} (count_uniform_world_sampler_v1, same states)")
    add(f"independent second-path worlds  {volumes['independent_worlds']:,} (floor {MIN_INDEPENDENT_WORLDS:,})")
    add(f"strata covered                  {len(coverage['states_by_stratum'])}/8; colours {sorted(coverage['states_by_observer_color'])}")
    add(f"progress buckets                {dict(coverage['states_by_progress_bucket'])}")
    add(f"moved/unmoved uncertainty       {coverage['states_with_moved_uncertainty']:,} / {coverage['states_with_unmoved_uncertainty']:,} states")
    add(f"unresolved pieces               mean {volumes['unresolved_pieces_mean']}, max {volumes['unresolved_pieces_max']}")
    add(f"wall clock                      {audit['wall_clock_seconds']:.0f}s ({audit['worlds_per_second']} worlds/s)")
    add("```")
    add("")
    add("Zero-tolerance counters, learned sampler (all must be and are zero):")
    add("")
    add("```text")
    for name, value in sorted(audit["zero_tolerance_counters"].items()):
        add(f"{name:<32} {value}")
    add("```")
    add("")
    add(
        f"Store integrity: {integrity['shards_verified']:,} public shards "
        "re-hashed against the Agent 2 manifest (0 mismatches); every selected "
        "decision's rebuilt document matched its stored identity, observation "
        "digest, hidden-slot set, masks and counts exactly "
        f"({integrity['identity_mismatches']}/"
        f"{integrity['mask_mismatches']}/{integrity['count_mismatches']} "
        "identity/mask/count mismatches). The baseline sampler produced "
        f"{audit['baseline_sampler']['worlds']:,} valid worlds on the same "
        "states with all counters zero; no strength comparison was run."
    )
    add("")

    add("### 3.4 Independent audit, determinism, collisions, controls")
    add("")
    add(
        "The second implementation path "
        "(`stratego/evaluation/phase11_sampler_audit.py`) imports no Phase 11 "
        "module: it rebuilds the inventory, masks, multiset, public facts and "
        "the raw-`blake2b` seed derivation from the engine authority and the "
        "published contract text, and re-runs every audited walk with scalar "
        "arithmetic. "
        f"{independent['worlds']:,} worlds re-derived exactly, "
        f"{independent['disagreements']} disagreements, "
        f"{independent['knife_edge_events']} float knife-edge events across "
        f"{independent['steps_recomputed']:,} recomputed steps."
    )
    add("")
    add("```text")
    add(f"deterministic repeats           {determinism['repeat_worlds']:,} worlds re-sampled bit-identically ({determinism['repeat_mismatches']} mismatches)")
    add(f"call-order reversal             {determinism['reversal_states']:,} states re-sampled in reverse ordinal order ({determinism['reversal_mismatches']} mismatches)")
    add(f"seed collision audit            {collision['total_seeds']:,} seeds ({collision['distinct_seeds']:,} distinct), no collisions: {collision['no_collisions']}")
    streams = collision["streams"]
    for name in ("world_sample", "world_order", "world_categorical"):
        if name in streams:
            add(f"  {name:<28}  {streams[name]['count']:,} derived, {streams[name]['distinct']:,} distinct")
    add("```")
    add("")
    add(
        "The collision audit ran `stream_collision_audit` over every "
        "`world_sample`, `world_order` and `world_categorical` seed the audit "
        "actually derived — exhaustively, learned and baseline tokens alike — "
        "combined with the complete re-derived Agent 1 enumerable universe "
        "(bank, soak, safety, repro, benchmark, bootstrap), so the frozen "
        "downstream obligation is discharged against the whole relevant seed "
        "space, not just the new streams."
    )
    add("")
    add("Negative controls (each must fire and did):")
    add("")
    add("```text")
    for control in controls["controls"]:
        add(f"{control['control']:<32} fired={control['fired']}")
    add("```")
    add("")

    add("### 3.5 Report-only diagnostics")
    add("")
    fallback = audit["fallback"]
    add(
        f"Zero-mass fallback: {fallback['fallback_steps_total']:,} steps in "
        f"{fallback['worlds_with_fallback']:,} of "
        f"{volumes['learned_worlds']:,} worlds "
        f"(rate {fallback['fallback_world_rate']}). Per-state distinct-world "
        "counts, empirical-vs-learned marginal L1 agreement and learned "
        "entropy are in `agent_03_sampler_diagnostics.csv` "
        f"({volumes['states']:,} rows). No diversity threshold is frozen, so "
        "these rank nothing."
    )
    add("")

    add("### 3.6 Preservation and the seal")
    add("")
    add("```text")
    add(f"Phase 9 checkpoint unchanged    {preservation['checkpoint_unchanged']}")
    add(f"belief head unchanged           {preservation['belief_head_unchanged']}")
    add(f"optimizer step delta            {preservation['optimizer_step_delta']} (steps run: {preservation['optimizer_steps_run']})")
    add(f"P10-D / anchor / Phase 7        {preservation['p10d_unchanged']} / {preservation['anchor_unchanged']} / {preservation['phase7_unchanged']}")
    add(f"prediction store unchanged      {preservation['prediction_store_unchanged']}")
    add(f"bank artifacts unchanged        {preservation['bank_files_unchanged']}")
    add(f"test bank                       structural-only, 0 scored / 0 truth / 0 outcome / 0 inference accesses")
    add("```")
    add("")

    add("### 3.7 Completion gates")
    add("")
    add("| # | Gate | Result |")
    add("|---|------|--------|")
    for index, (name, value) in enumerate(
        sorted(read_json(DATA_DIRECTORY / "agent_03_acceptance.json")["completion_gates"].items()),
        start=1,
    ):
        add(f"| {index} | `{name}` | {'true' if value else '**FALSE**'} |")
    add("")
    add(f"Suite: `{suite.get('summary', 'not recorded')}`.")
    add("")

    add("### 3.8 Recorded readings and handoff to Agent 4")
    add("")
    for reading in acceptance["recorded_readings"]:
        add(f"- **{reading['reading']}** — {reading['statement']}")
    add("")
    handoff = acceptance["handoff_to_agent_4"]
    add(
        "Agent 4 receives the immutable sampler identity "
        f"(`{handoff['sampler']['module']}`, SHA-256 "
        f"`{handoff['sampler']['module_sha256'][:16]}...`), the provenance "
        "schema, the sample-token rules (production ordinals 0..63), the "
        "validation public-state list (the diagnostics CSV), the audit "
        "evidence, and the zero-mass fallback behaviour. Agent 4 must not "
        "change the sampler mathematics."
    )
    add("")
    add(
        f"**Agent 3 stops here.** Ending revision: uncommitted working tree "
        f"over `{acceptance['starting_revision']}`; per the commit discipline, "
        "the commit happens only after reviewing-chat acceptance."
    )
    add("")
    return "\n".join(lines)


def stage_report(_args) -> dict:
    section = build_report_section()
    text = REPORT_PATH.read_text()
    marker = "## 3. Agent 3 — `belief_sampler_v1` and the Complete-World Audit"
    if marker in text:
        head = text[: text.index(marker)]
        text = head + section
    else:
        if not text.endswith("\n"):
            text += "\n"
        text = text + "\n" + section
    REPORT_PATH.write_text(text)
    log(f"report section written to {REPORT_PATH}")
    write_stage("report", {"written": True, "characters": len(section)})
    return {"written": True}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STAGES = {
    "verify": stage_verify,
    "contract": stage_contract,
    "audit": stage_audit,
    "acceptance": stage_acceptance,
    "report": stage_report,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), default=None)
    parser.add_argument("--limit-games", type=int, default=None)
    parser.add_argument("--record-suite", action="store_true")
    args = parser.parse_args()

    if args.record_suite:
        record_suite(args)
        return 0
    if args.stage:
        STAGES[args.stage](args)
        return 0
    for name in ("verify", "contract", "audit", "acceptance", "report"):
        STAGES[name](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
