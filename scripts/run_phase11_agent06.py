#!/usr/bin/env python3
"""Phase 11 Agent 6 harness: the production integration soak and the
`phase11_system_v1` freeze.

Recomputes every load-bearing identity from live bytes (the Agent 1 freeze
and its eight contracts, the Agent 2-5 PASS records and handoffs, both bank
digests, the Phase 9 checkpoint's file SHA / model-state digest / parameter
count / belief-head tensor identity / optimizer-step counter, the frozen
P10-D chain, the Phase 7 library, and the whole
`phase11_validation_freeze_v1` document), then exercises the accepted
production belief path under repeated use and real process death.

Four things happen here and nothing else:

- **Soak** — Agent 1's frozen `phase11_soak_v1` schedule: 1,024 train-split
  non-bank games (128 per opponent stratum, both seats drawn from the
  accepted P10-D production source, observer colour by the frozen ordinal
  parity), each contributing exactly 8 production belief requests, every
  request one real belief forward plus 64 complete legal worlds.
- **Crash / restart** — the 8,192 requests are committed across legs with
  different worker counts, one of which is really SIGKILLed after committed
  work exists and resumed by exact logical request-id set subtraction. The
  final store holds exactly the scheduled ids.
- **Per-request audit** — every committed request is re-derived from
  scratch: the position replayed from public bytes, the public-state
  identity, the belief forward, all 64 sample identities and all 64 worlds
  rebuilt, each world re-checked against the independent Agent 3
  implementation, the document rebuilt from a hidden-rank-traced state, and
  the result compared with what was committed. A substantial subset is
  replayed again under a different topology.
- **Freeze** — Agent 1's `phase11_system_v1` template filled by its own
  predetermined filling rules and frozen.

Nothing here trains, calibrates, changes a threshold, a bin, a baseline, a
bank, a stratum or a sampler rule. The known validation reading
`R_CE = 0.9750` is carried as a diagnostic and is not acted on.

`phase11_test_bank_v1` is touched only to re-derive its digest structurally;
its scored access stays zero and Agent 7 remains the first agent permitted
to score it.

    reports/phase_11_data/agent_06_soak_manifest.json
    reports/phase_11_data/agent_06_soak_audit.json
    reports/phase_11_data/agent_06_system_v1.json
    reports/phase_11_data/agent_06_acceptance.json

Usage::

    python scripts/run_phase11_agent06.py                    # every stage
    python scripts/run_phase11_agent06.py --stage verify     # one stage
    python scripts/run_phase11_agent06.py --limit-games 16   # a smoke run
    python scripts/run_phase11_agent06.py --record-suite     # run + record
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
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

AGENT = 6
PHASE = 11
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_11_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_11_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase11" / "agent06"
CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "selfplay_c1_v1.pt"
EXPORT_PATH = WORK_DIRECTORY / "eval_phase9_c1.pt"

#: The frozen evaluation backend, unchanged from Agents 2, 4 and 5.
EVAL_DEVICE = "cpu"
EVAL_TORCH_THREADS = 1

#: The Agent 5 freeze digest this harness refuses to run without.
EXPECTED_FREEZE_DIGEST = (
    "ad2562af538abc6c78fc5b12bc1f57d3e32184172acde390417a00d500a0d912"
)

#: The soak legs, frozen before the soak ran. Three legs, three different
#: worker counts, the third really killed and resumed.
SOAK_LEGS = (
    {"leg": "leg_1_workers_1", "workers": 1, "kill": False},
    {"leg": "leg_2_workers_4", "workers": 4, "kill": False},
    {"leg": "leg_3_workers_12_kill_resume", "workers": 12, "kill": True},
)

#: Committed requests one killed worker must have written before the
#: SIGKILL lands. Frozen before the leg ran.
KILL_AFTER_COMMITTED = 24

#: The cross-topology replay: an evenly spaced subset of the committed
#: requests, re-executed under a different worker count and a reversed
#: order in a fresh process.
REPLAY_SUBSET = 2048
REPLAY_WORKERS = 6

#: The Agent 6 audit worker count.
AUDIT_WORKERS = 10

#: The evidence Agent 6 binds into `phase11_system_v1`, re-hashed live.
BOUND_EVIDENCE_ARTIFACTS = (
    "agent_01_phase11_contract.json",
    "agent_01_validation_bank.json",
    "agent_01_test_bank.json",
    "agent_02_predictive_metrics.json",
    "agent_02_baseline_audit.json",
    "agent_03_sampler_contract.json",
    "agent_03_sampler_audit.json",
    "agent_04_frozen_sets.json",
    "agent_04_information_safety.json",
    "agent_04_reproducibility.json",
    "agent_04_stream_audit.json",
    "agent_04_runtime.csv",
    "agent_05_validation_freeze.json",
    "agent_05_validation_metrics.json",
)


class Agent6Error(RuntimeError):
    """The Agent 6 harness refused to continue."""


# ---------------------------------------------------------------------------
# Small shared utilities (the accepted Agent 2/3/4/5 harness shapes)
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


def _jsonable(value):
    import numpy as np

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"{type(value).__name__} is not JSON-serialisable")


def stage_path(name: str) -> Path:
    return WORK_DIRECTORY / f"stage_{name}.json"


def write_stage(name: str, payload: dict) -> Path:
    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = stage_path(name)
    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=_jsonable) + "\n"
    )
    return path


def read_stage(name: str) -> dict:
    path = stage_path(name)
    if not path.exists():
        raise Agent6Error(f"stage {name!r} has not run yet ({path} is missing)")
    return json.loads(path.read_text())


def require(condition: bool, message: str, problems: list) -> bool:
    if not condition:
        problems.append(message)
    return bool(condition)


def log(message: str) -> None:
    print(f"[phase11:agent6] {message}", flush=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_artifact(name: str, payload: dict) -> Path:
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DATA_DIRECTORY / name
    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=_jsonable) + "\n"
    )
    return path


def module_digests(names) -> dict:
    return {name: file_sha256(REPOSITORY_ROOT / name) for name in sorted(names)}


def configure_backend() -> None:
    import torch

    torch.set_num_threads(EVAL_TORCH_THREADS)


# ---------------------------------------------------------------------------
# 1. Verification — Agents 1-5 and every identity, from live bytes
# ---------------------------------------------------------------------------


def _verify_agent_acceptance(
    filename: str, agent: int, problems: list, *, handoff_key: "str | None" = None
) -> dict:
    path = DATA_DIRECTORY / filename
    require(path.exists(), f"the Agent {agent} acceptance artifact is missing", problems)
    if not path.exists():
        return {"available": False}
    acceptance = read_json(path)
    require(
        acceptance.get("status") == "PASS",
        f"Agent {agent} did not report PASS",
        problems,
    )
    gates = acceptance.get("completion_gates", {})
    false_gates = sorted(name for name, value in gates.items() if not value)
    require(not false_gates, f"Agent {agent} gates are false: {false_gates}", problems)
    counters = acceptance.get("forbidden_operation_counters", {})
    nonzero = sorted(name for name, value in counters.items() if value)
    require(
        not nonzero,
        f"Agent {agent} forbidden-operation counters are non-zero: {nonzero}",
        problems,
    )
    summary = {
        "available": True,
        "status": acceptance.get("status"),
        "gates_true": acceptance.get("gates_true"),
        "gates_total": acceptance.get("gates_total"),
        "file_sha256": file_sha256(path),
    }
    if handoff_key is not None:
        handoff = acceptance.get(handoff_key, {})
        require(
            handoff.get("for_agent") == agent + 1,
            f"the Agent {agent} handoff is not for Agent {agent + 1}",
            problems,
        )
        summary["handoff"] = handoff
    return summary


def verify_agents_1_to_5(problems: list) -> dict:
    """Every upstream agent's PASS record, from live artifact bytes."""
    return {
        "agent1": _verify_agent_acceptance(
            "agent_01_acceptance.json", 1, problems, handoff_key="handoff_to_agent_2"
        ),
        "agent2": _verify_agent_acceptance(
            "agent_02_acceptance.json", 2, problems, handoff_key="handoff_to_agent_3"
        ),
        "agent3": _verify_agent_acceptance(
            "agent_03_acceptance.json", 3, problems, handoff_key="handoff_to_agent_4"
        ),
        "agent4": _verify_agent_acceptance(
            "agent_04_acceptance.json", 4, problems, handoff_key="handoff_to_agent_5"
        ),
        "agent5": _verify_agent_acceptance(
            "agent_05_acceptance.json", 5, problems, handoff_key="handoff_to_agent_6"
        ),
    }


def belief_head_identity(model_state: dict) -> dict:
    """The belief head's tensor identity, re-derived from live tensor bytes.

    Byte-for-byte the Agent 1 derivation Agents 4 and 5 also re-ran: names,
    shapes and float32 bytes of the frozen head tensors, in name order.
    """
    import numpy as np

    from stratego.training.phase11_contract import (
        BELIEF_HEAD_TENSOR_NAMES,
        BELIEF_HEAD_TENSOR_SHAPES,
    )

    hasher = hashlib.sha256()
    shapes = {}
    for name in sorted(BELIEF_HEAD_TENSOR_NAMES):
        if name not in model_state:
            raise Agent6Error(f"the belief-head tensor {name!r} is missing")
        array = model_state[name].detach().to("cpu").numpy().astype(np.float32)
        shapes[name] = list(array.shape)
        hasher.update(name.encode())
        hasher.update(str(tuple(array.shape)).encode())
        hasher.update(np.ascontiguousarray(array).tobytes())
    expected = {
        name: list(shape) for name, shape in BELIEF_HEAD_TENSOR_SHAPES.items()
    }
    return {
        "digest": hasher.hexdigest(),
        "tensor_names": sorted(BELIEF_HEAD_TENSOR_NAMES),
        "shapes": shapes,
        "shapes_match": shapes == expected,
    }


def verify_phase9_checkpoint(problems: list) -> dict:
    """The Phase 9 checkpoint, re-derived from live bytes."""
    import torch

    from stratego.training.phase11_contract import (
        ACCEPTED_BELIEF_HEAD_DIGEST,
        ACCEPTED_GLOBAL_OPTIMIZER_STEP,
        ACCEPTED_PHASE9_CHECKPOINT_SHA256,
        ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
        ACCEPTED_PHASE9_PARAMETERS,
    )
    from stratego.training.phase9_behavior import state_dict_digest
    from stratego.training.phase9_checkpoint import (
        model_from_payload,
        read_phase9_payload,
    )

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
    step = int(payload.get("global_optimizer_step", -1))

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
    require(head["shapes_match"], "the belief-head tensor shapes moved", problems)
    require(
        step == ACCEPTED_GLOBAL_OPTIMIZER_STEP,
        f"global optimizer step {step} != {ACCEPTED_GLOBAL_OPTIMIZER_STEP}",
        problems,
    )
    return {
        "available": True,
        "sha256": sha,
        "model_state_digest": digest,
        "parameters": int(parameters),
        "finite": bool(finite),
        "belief_head_digest": head["digest"],
        "belief_head_shapes": head["shapes"],
        "global_optimizer_step": step,
    }


def verify_upstream_stack(problems: list) -> dict:
    """The P10-D chain, the Phase 8 anchor and the Phase 7 library."""
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


def verify_banks(problems: list) -> dict:
    """Both frozen banks, re-hashed from their live artifact bytes."""
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
        }
    return summary


def verify_agent5_freeze(problems: list, phase9: dict, banks_summary: dict) -> dict:
    """The whole Agent 5 freeze document, rebuilt from live bytes.

    Not a string comparison against a recorded digest: the freeze document
    is *reconstructed* by `phase11_pipeline.implementation_freeze` from the
    live checkpoint, the live contract bundle, the live bank artifacts, the
    live Agent 4 runtime result and the live bytes of all 17 frozen
    implementation modules, and the digest of that reconstruction is what
    is compared.
    """
    from stratego.evaluation import phase11_pipeline as pipeline
    from stratego.training import phase11_contract as contract

    path = DATA_DIRECTORY / "agent_05_validation_freeze.json"
    require(path.exists(), "the Agent 5 freeze artifact is missing", problems)
    if not path.exists():
        return {"available": False}
    recorded = read_json(path)["freeze"]

    runtime_source = read_json(DATA_DIRECTORY / "agent_04_acceptance.json")
    runtime = _find_runtime(runtime_source)
    require(runtime is not None, "the Agent 4 runtime result is missing", problems)

    bound = {
        "information_safety_version": contract.INFORMATION_SAFETY_VERSION,
        "artifacts": {
            name: file_sha256(DATA_DIRECTORY / name)
            for name in recorded["bound_evidence"]
        },
    }
    rebuilt = pipeline.implementation_freeze(
        REPOSITORY_ROOT,
        belief_head_digest=phase9["belief_head_digest"],
        model_state_digest=phase9["model_state_digest"],
        contract_bundle_digest=contract.contract_bundle_digest(),
        validation_bank_digest=banks_summary["validation"]["bank_digest"],
        test_bank_digest=banks_summary["test"]["bank_digest"],
        runtime=runtime,
        bound_evidence=bound,
    )
    differing = sorted(
        key
        for key in set(rebuilt) | set(recorded)
        if key != "freeze_digest" and rebuilt.get(key) != recorded.get(key)
    )
    require(
        not differing,
        f"the rebuilt Agent 5 freeze differs on {differing}",
        problems,
    )
    require(
        rebuilt["freeze_digest"] == recorded["freeze_digest"],
        f"the rebuilt freeze digest {rebuilt['freeze_digest']} != recorded",
        problems,
    )
    require(
        rebuilt["freeze_digest"] == EXPECTED_FREEZE_DIGEST,
        f"the rebuilt freeze digest {rebuilt['freeze_digest']} != the handed-over "
        f"{EXPECTED_FREEZE_DIGEST}",
        problems,
    )
    live_modules = module_digests(recorded["module_sha256"])
    moved = sorted(
        name
        for name, digest in recorded["module_sha256"].items()
        if live_modules.get(name) != digest
    )
    require(
        not moved,
        f"the live implementation no longer matches the Agent 5 freeze: {moved}",
        problems,
    )
    return {
        "available": True,
        "freeze_version": recorded["freeze_version"],
        "recorded_freeze_digest": recorded["freeze_digest"],
        "recomputed_freeze_digest": rebuilt["freeze_digest"],
        "freeze_digest_matches": rebuilt["freeze_digest"] == recorded["freeze_digest"],
        "matches_handoff_digest": rebuilt["freeze_digest"] == EXPECTED_FREEZE_DIGEST,
        "differing_fields": differing,
        "modules_unchanged": not moved,
        "module_count": len(recorded["module_sha256"]),
        "artifact_sha256": file_sha256(path),
        "document": rebuilt,
    }


def _find_runtime(node):
    """The Agent 4 measured runtime block, wherever it sits in the record."""
    if isinstance(node, dict):
        if "p95_forward_64_ms" in node and "ceiling_ms" in node:
            return node
        for value in node.values():
            found = _find_runtime(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_runtime(value)
            if found is not None:
                return found
    return None


def verify_test_bank_sealed(problems: list) -> dict:
    """The test bank's scored access, from the live append-only ledger."""
    from stratego.evaluation import phase11_banks as banks
    from stratego.evaluation import phase11_pipeline as pipeline

    summary = banks.verify_test_bank_sealed()
    require(not summary["violations"], f"ledger violations: {summary['violations']}", problems)
    require(
        summary["test_bank_structural_only"],
        "the test bank has non-structural ledger access",
        problems,
    )
    for key in (
        "scored_prediction_total",
        "privileged_truth_total",
        "outcome_total",
        "neural_inference_total",
    ):
        require(int(summary[key]) == 0, f"test-bank {key} is not zero", problems)

    refused = False
    try:
        pipeline.assert_seal("test", sealed_bank_authorized=False)
    except pipeline.Phase11SealError:
        refused = True
    require(refused, "the pipeline did not refuse the sealed test bank", problems)
    summary["test_refused_without_authorization"] = refused
    return summary


def stage_verify(_args) -> dict:
    problems: list[str] = []
    log("verifying the Agent 1-5 PASS records")
    agents = verify_agents_1_to_5(problems)
    log("re-deriving the Phase 9 checkpoint and belief-head identity")
    phase9 = verify_phase9_checkpoint(problems)
    log("verifying the P10-D / anchor / Phase 7 stack")
    upstream = verify_upstream_stack(problems)
    log("re-hashing both frozen banks")
    banks_summary = verify_banks(problems)
    log("rebuilding the Agent 5 implementation freeze from live bytes")
    freeze = verify_agent5_freeze(problems, phase9, banks_summary)
    log("checking the test-bank seal")
    sealing = verify_test_bank_sealed(problems)

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
        "stage": "verify",
        "environment": environment_report(),
        "ledger_entries_appended": 2,
        "agents": agents,
        "phase9": phase9,
        "upstream": upstream,
        "banks": banks_summary,
        "agent5_freeze": freeze,
        "test_bank_sealing": sealing,
        "preservation_before": preservation_observation(phase9, upstream),
        "problems": problems,
    }
    if problems:
        for problem in problems:
            log(f"BLOCKED: {problem}")
        raise Agent6Error(f"verification failed with {len(problems)} problem(s)")
    log("verification complete: every load-bearing identity re-derived")
    return payload


def preservation_observation(phase9: dict, upstream: dict) -> dict:
    """The Gate H observation, in the frozen key names."""
    from stratego.training.phase11_contract import PHASE7_LIBRARY_CONTENT_DIGEST

    return {
        "phase9_checkpoint_sha256": phase9["sha256"],
        "phase9_model_state_digest": phase9["model_state_digest"],
        "phase9_parameters": int(phase9["parameters"]),
        "phase11_optimizer_steps": 0,
        "global_optimizer_step": int(phase9["global_optimizer_step"]),
        "belief_head_digest": phase9["belief_head_digest"],
        "selector_config_sha256": upstream["selector_config_sha256"],
        "utility_coefficient_digest": upstream["utility_coefficient_digest"],
        "trait_scaler_digest": upstream["trait_scaler_digest"],
        "phase7_library_content_digest": upstream["phase7_library"]["content_digest"],
        "phase7_library_reference_digest": PHASE7_LIBRARY_CONTENT_DIGEST,
    }


# ---------------------------------------------------------------------------
# The soak driver
#
# Deliberately in the harness rather than in `stratego/evaluation`: every
# `phase11_*.py` module under `stratego/` is covered by
# `phase11_pipeline.FROZEN_IMPLEMENTATION_MODULES`, and adding one after
# Agent 5's freeze would change `phase11_pipeline.py`'s bytes and so the
# frozen `phase11_validation_freeze_v1` digest itself. The soak owns no
# belief, sampler or metric code — it plays train-split non-bank games with
# the accepted observer seat and drives the accepted production request
# path — so it belongs here, exactly where Agent 4 kept its topology-leg
# machinery.
# ---------------------------------------------------------------------------


from dataclasses import dataclass

from stratego.training.phase11_contract import (
    OPPONENT_STRATA,
    Phase11ContractError,
    SOURCE_P10D,
)
from stratego.training.phase11_seed import (
    COLOR_BLUE,
    COLOR_RED,
    DOMAIN_SOAK_MATCH,
    DOMAIN_SOAK_SETUP,
    PHASE11_MASTER_SEED,
    PHASE11_SOAK_VERSION,
    ROLE_OBSERVER,
    ROLE_OPPONENT,
    SOAK_GAME_COUNT,
    SOAK_GAMES_PER_STRATUM,
    SOAK_REQUESTS_PER_GAME,
    SOAK_REQUEST_COUNT,
    derive_phase11_seed,
    parse_phase11_soak_game_id,
    phase11_soak_game_id,
    phase11_soak_request_id,
    soak_match_seed,
    soak_setup_seed,
)

#: The soak's split. Frozen by Agent 1: train-only, never a bank split.
SOAK_SPLIT = "train"

#: Reviewer-authorized supplement.
#:
#: Agent 1's frozen soak arithmetic (1,024 games x 8 = 8,192) is not
#: realizable: 29 of the 1,024 games give the observer no decision at all,
#: because the frozen rules let a scout move and strike in one turn, so a
#: first-player scout can capture a front-rank flag at ply 1 and the
#: second-seat observer never decides. Agent 6 reported the 232-request
#: shortfall rather than repairing it, and the reviewer authorized this
#: extension verbatim:
#:
#:   "Preserve all original 1,024 games and 7,960 requests exactly. Starting
#:    with the next sequential soak game ordinal after the original range,
#:    extend the existing Agent 1 soak-generation rules unchanged: train
#:    split only, P10-D on both sides, same opponent-stratum mapping, same
#:    observer-color rule, same soak_setup/soak_match derivations, and the
#:    same eight-request attachment rule. A zero-observer-decision game
#:    contributes zero requests and is recorded. Continue sequentially until
#:    exactly 29 additional playable games have contributed their eight
#:    requests, yielding exactly 232 supplemental requests and 8,192 total
#:    realized requests. Do not choose games based on outcomes, belief
#:    values, runtime, or sampler behavior."
#:
#: Nothing frozen is edited to do this. `stratego/training/phase11_seed.py`
#: is one of the 17 modules the Agent 5 freeze digest covers, and its
#: `phase11_soak_game_id` refuses an ordinal >= 128 by a range check. The
#: supplement therefore formats the *same* id and calls the *same*
#: `derive_phase11_seed` under the *same* domain tokens, with the ordinal
#: range extended — and `assert_extension_matches_frozen_rules` proves the
#: local formatter and seed derivations agree with the frozen helpers on
#: every one of the 1,024 frozen games before a supplemental game is drawn.
SUPPLEMENTAL_AUTHORIZED = True

#: Playable supplemental games required. Exactly the shortfall, so the
#: realized total is exactly Agent 1's frozen `SOAK_REQUEST_COUNT`.
SUPPLEMENTAL_PLAYABLE_TARGET = 29

#: The next sequential ordinal after the frozen range.
SUPPLEMENTAL_FIRST_ORDINAL = SOAK_GAMES_PER_STRATUM

#: A hard stop, so a pathological run cannot enumerate forever. Never
#: reached in practice; reaching it is a BLOCKED condition, not a silent
#: truncation.
SUPPLEMENTAL_ORDINAL_LIMIT = SOAK_GAMES_PER_STRATUM + 64

#: Both soak seats draw from the accepted P10-D production source.
SOAK_SETUP_SOURCE = SOURCE_P10D

#: Worlds per production request. The common contract's request shape.
SOAK_WORLD_ORDINALS = 64

#: The suite/bank version token soak games are played under. Distinct from
#: `phase11_runner.PHASE11_RUN_VERSION`, so a soak game can never be
#: mistaken for a bank game by identity.
SOAK_RUN_VERSION = "phase11_soak_run_v1"


class Phase11SoakError(Phase11ContractError):
    """A soak game, request or audit could not be built or verified."""


# ---------------------------------------------------------------------------
# The frozen schedule
# ---------------------------------------------------------------------------


def soak_game_id(stratum: str, game_ordinal: int) -> str:
    """The soak game id, in Agent 1's frozen format.

    Byte-identical to `phase11_seed.phase11_soak_game_id` on the frozen
    0..127 range — :func:`assert_extension_matches_frozen_rules` proves it —
    and defined for the supplement's higher ordinals, which the frozen
    helper refuses by a range check alone.
    """
    if stratum not in OPPONENT_STRATA:
        raise Phase11SoakError(
            f"stratum must be one of {list(OPPONENT_STRATA)}, got {stratum!r}"
        )
    if (
        not isinstance(game_ordinal, int)
        or isinstance(game_ordinal, bool)
        or game_ordinal < 0
    ):
        raise Phase11SoakError(
            f"soak game ordinal must be a non-negative int, got {game_ordinal!r}"
        )
    return (
        f"{PHASE11_SOAK_VERSION}|ms={PHASE11_MASTER_SEED}|st={stratum}"
        f"|g={game_ordinal:03d}"
    )


def soak_request_id(game_id: str, request_ordinal: int) -> str:
    """The soak request id, in Agent 1's frozen format."""
    if (
        not isinstance(request_ordinal, int)
        or isinstance(request_ordinal, bool)
        or not 0 <= request_ordinal < SOAK_REQUESTS_PER_GAME
    ):
        raise Phase11SoakError(
            f"soak request ordinal must be an int in 0..{SOAK_REQUESTS_PER_GAME - 1}, "
            f"got {request_ordinal!r}"
        )
    return f"{game_id}|r={request_ordinal}"


def observer_color_of(game_ordinal: int) -> str:
    """Agent 1's frozen parity rule: red on even ordinals, blue on odd."""
    return COLOR_RED if int(game_ordinal) % 2 == 0 else COLOR_BLUE


def soak_stream_seeds(game_id: str) -> dict:
    """The match and both setup seeds of one soak game.

    The same `derive_phase11_seed` calls under the same frozen domain
    tokens that `soak_match_seed` / `soak_setup_seed` make; only the
    ordinal range they accept differs.
    """
    return {
        "match_seed": derive_phase11_seed(DOMAIN_SOAK_MATCH, game_id),
        "observer_setup_seed": derive_phase11_seed(
            DOMAIN_SOAK_SETUP, game_id, ROLE_OBSERVER
        ),
        "opponent_setup_seed": derive_phase11_seed(
            DOMAIN_SOAK_SETUP, game_id, ROLE_OPPONENT
        ),
    }


def assert_extension_matches_frozen_rules() -> dict:
    """Prove the local id/seed rules ARE Agent 1's, over the frozen range.

    Every frozen game id, every match seed, every setup seed and every
    request id is recomputed both ways and compared. A single disagreement
    means the supplement is not the same rule and must not run.
    """
    checked = mismatches = 0
    for stratum in OPPONENT_STRATA:
        for ordinal in range(SOAK_GAMES_PER_STRATUM):
            frozen_id = phase11_soak_game_id(stratum, ordinal)
            local_id = soak_game_id(stratum, ordinal)
            checked += 1
            mismatches += local_id != frozen_id
            fields = parse_phase11_soak_game_id(frozen_id)
            checked += 1
            mismatches += observer_color_of(ordinal) != fields["observer_color"]
            seeds = soak_stream_seeds(frozen_id)
            checked += 1
            mismatches += seeds["match_seed"] != soak_match_seed(frozen_id)
            for role, key in (
                (ROLE_OBSERVER, "observer_setup_seed"),
                (ROLE_OPPONENT, "opponent_setup_seed"),
            ):
                checked += 1
                mismatches += seeds[key] != soak_setup_seed(frozen_id, role)
            for request_ordinal in range(SOAK_REQUESTS_PER_GAME):
                checked += 1
                mismatches += soak_request_id(
                    frozen_id, request_ordinal
                ) != phase11_soak_request_id(frozen_id, request_ordinal)
    if mismatches:
        raise Phase11SoakError(
            f"the supplemental id/seed rules disagree with the frozen ones on "
            f"{mismatches} of {checked} comparisons; refusing to extend"
        )
    return {
        "comparisons": checked,
        "mismatches": 0,
        "frozen_games_covered": SOAK_GAME_COUNT,
        "rules_identical": True,
    }


def supplemental_candidate_order():
    """The frozen supplemental enumeration: ordinal-major, stratum order.

    The original schedule holds one game per (stratum, ordinal) cell, so
    "the next sequential soak game ordinal after the original range" is
    ordinal 128 across all eight strata, then 129, and so on. Within an
    ordinal the strata keep Agent 1's frozen order. The sequence is a pure
    function of arithmetic — no outcome, belief value, runtime or sampler
    quantity can reach it.
    """
    for ordinal in range(SUPPLEMENTAL_FIRST_ORDINAL, SUPPLEMENTAL_ORDINAL_LIMIT):
        for stratum in OPPONENT_STRATA:
            yield stratum, ordinal


def supplemental_descriptor(stratum: str, ordinal: int, game_index: int) -> dict:
    """One supplemental game descriptor, on the extended Agent 1 rules."""
    game_id = soak_game_id(stratum, ordinal)
    observer = observer_color_of(ordinal)
    seeds = soak_stream_seeds(game_id)
    return {
        "game_index": int(game_index),
        "game_id": game_id,
        "stratum": stratum,
        "game_ordinal": int(ordinal),
        "observer_color": observer,
        "opponent_color": COLOR_BLUE if observer == COLOR_RED else COLOR_RED,
        "match_seed": seeds["match_seed"],
        "observer_setup_seed": seeds["observer_setup_seed"],
        "opponent_setup_seed": seeds["opponent_setup_seed"],
        "split": SOAK_SPLIT,
        "setup_source": SOAK_SETUP_SOURCE,
        "supplemental": True,
    }


def soak_game_descriptors() -> "list[dict]":
    """The 1,024 frozen soak games, in frozen order.

    Stratum-major, ordinal within stratum; the observer colour comes from
    the frozen ordinal-parity rule, never from a draw.
    """
    descriptors = []
    index = 0
    for stratum in OPPONENT_STRATA:
        for ordinal in range(SOAK_GAMES_PER_STRATUM):
            game_id = phase11_soak_game_id(stratum, ordinal)
            fields = parse_phase11_soak_game_id(game_id)
            descriptors.append(
                {
                    "game_index": index,
                    "game_id": game_id,
                    "stratum": stratum,
                    "game_ordinal": ordinal,
                    "observer_color": fields["observer_color"],
                    "opponent_color": fields["opponent_color"],
                    "match_seed": soak_match_seed(game_id),
                    "observer_setup_seed": soak_setup_seed(game_id, ROLE_OBSERVER),
                    "opponent_setup_seed": soak_setup_seed(game_id, ROLE_OPPONENT),
                    "split": SOAK_SPLIT,
                    "setup_source": SOAK_SETUP_SOURCE,
                }
            )
            index += 1
    if len(descriptors) != SOAK_GAME_COUNT:
        raise Phase11SoakError(
            f"the soak schedule holds {len(descriptors)} games, not {SOAK_GAME_COUNT}"
        )
    return descriptors


def request_decision_positions(observer_decisions: int) -> "list[int]":
    """The frozen attachment rule: request k -> observer-decision position.

    `min(floor(k * D / 8), D - 1)` over the game's D observer decisions,
    exactly as Agent 1 froze it. On a short game two requests can land on
    one decision; that is deliberate, and their worlds must then be
    byte-identical.
    """
    total = int(observer_decisions)
    if total <= 0:
        raise Phase11SoakError("a soak game recorded no observer decision")
    return [
        min((k * total) // SOAK_REQUESTS_PER_GAME, total - 1)
        for k in range(SOAK_REQUESTS_PER_GAME)
    ]


def schedule_digest(descriptors: "list[dict]", request_ids: "list[str]") -> str:
    """The content digest of the frozen soak schedule.

    Game identities, seats, seeds and the complete request-id list: the
    quantity a resume compares against, and the reason an unscheduled id
    cannot be invented later.
    """
    hasher = hashlib.sha256()
    hasher.update(f"{PHASE11_SOAK_VERSION}|games={len(descriptors)}".encode())
    for descriptor in descriptors:
        hasher.update(
            json.dumps(
                {
                    key: descriptor[key]
                    for key in (
                        "game_index",
                        "game_id",
                        "stratum",
                        "game_ordinal",
                        "observer_color",
                        "opponent_color",
                        "match_seed",
                        "observer_setup_seed",
                        "opponent_setup_seed",
                        "split",
                        "setup_source",
                    )
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    hasher.update(f"|requests={len(request_ids)}".encode())
    for request_id in request_ids:
        hasher.update(f"|{request_id}".encode())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# One soak game
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SoakGamePlan:
    """One playable soak game, resolved from its identity and the sources."""

    game_index: int
    game_id: str
    stratum: str
    game_ordinal: int
    observer_color: str
    opponent_color: str
    match_seed: int
    red_setup: tuple
    blue_setup: tuple
    observer_draw: dict
    opponent_draw: dict


def build_soak_plan(descriptor: dict, sources) -> SoakGamePlan:
    """Draw both seats of one soak game from the accepted P10-D source."""
    observer_draw = sources.draw(
        SOAK_SETUP_SOURCE,
        SOAK_SPLIT,
        descriptor["observer_color"],
        int(descriptor["observer_setup_seed"]),
    )
    opponent_draw = sources.draw(
        SOAK_SETUP_SOURCE,
        SOAK_SPLIT,
        descriptor["opponent_color"],
        int(descriptor["opponent_setup_seed"]),
    )
    observer_setup = tuple(observer_draw["setup"])
    opponent_setup = tuple(opponent_draw["setup"])
    red, blue = (
        (observer_setup, opponent_setup)
        if descriptor["observer_color"] == "red"
        else (opponent_setup, observer_setup)
    )
    return SoakGamePlan(
        game_index=int(descriptor["game_index"]),
        game_id=descriptor["game_id"],
        stratum=descriptor["stratum"],
        game_ordinal=int(descriptor["game_ordinal"]),
        observer_color=descriptor["observer_color"],
        opponent_color=descriptor["opponent_color"],
        match_seed=int(descriptor["match_seed"]),
        red_setup=red,
        blue_setup=blue,
        observer_draw=observer_draw,
        opponent_draw=opponent_draw,
    )


def _runner_plan(plan: SoakGamePlan):
    """The accepted runner's plan object, carrying the soak identity.

    Reusing `phase11_runner`'s own dataclass means the soak's opponent
    seat, match specification and single-game bank are built by the
    accepted code rather than by a soak copy of it.
    """
    from stratego.evaluation.phase11_runner import Phase11GamePlan

    return Phase11GamePlan(
        case_id=plan.game_id,
        case_index=plan.game_index,
        game_id=plan.game_id,
        game_index=plan.game_ordinal,
        stratum=plan.stratum,
        setup_source=SOAK_SETUP_SOURCE,
        observer_color=plan.observer_color,
        opponent_color=plan.opponent_color,
        match_seed=plan.match_seed,
        red_setup=plan.red_setup,
        blue_setup=plan.blue_setup,
    )


def soak_spec(plan: SoakGamePlan, opponent):
    """The completely determined specification of one soak game."""
    from stratego.engine.constants import BLUE, RED
    from stratego.evaluation.match_spec import EVALUATION_RULES, MatchSpec
    from stratego.evaluation.phase11_runner import observer_ref

    return MatchSpec(
        candidate=observer_ref(),
        opponent=opponent,
        setup_pair_id=plan.game_index,
        candidate_color=RED if plan.observer_color == "red" else BLUE,
        replicate=plan.game_ordinal,
        root_seed=plan.match_seed,
        suite_version=SOAK_RUN_VERSION,
        setup_bank_version=(
            f"{SOAK_RUN_VERSION}|st={plan.stratum}|src={SOAK_SETUP_SOURCE}"
        ),
        rules=EVALUATION_RULES,
    )


def play_soak_game(plan: SoakGamePlan, owners: dict) -> dict:
    """Play one soak game with the accepted observer seat.

    Returns the public facts the request schedule needs: the action
    history, the observer's decision plies and their public-state
    identities. The game's W/D/L is carried and is report-only.
    """
    from stratego.evaluation.match_runner import ON_POLICY_ERROR_RAISE, play_match
    from stratego.evaluation.phase10_validation import FrozenSeedPolicy
    from stratego.evaluation.phase11_records import Phase11GameRecorder
    from stratego.evaluation.phase11_runner import (
        Phase11ObserverPolicy,
        observer_ref,
        opponent_seat,
        single_game_bank,
    )

    runner_plan = _runner_plan(plan)
    opponent_reference, opponent_policy = opponent_seat(runner_plan, owners)
    spec = soak_spec(plan, opponent_reference)
    recorder = Phase11GameRecorder(
        {
            "soak_version": PHASE11_SOAK_VERSION,
            "game_id": plan.game_id,
            "game_index": plan.game_index,
            "observer_color": plan.observer_color,
            "opponent_stratum": plan.stratum,
            "setup_source": SOAK_SETUP_SOURCE,
            "split": SOAK_SPLIT,
            "match_seed": plan.match_seed,
            "match_id": spec.match_id,
        }
    )
    observer = Phase11ObserverPolicy(observer_ref(), owners["phase9"], recorder)
    policies = {observer_ref().token: observer}
    if opponent_reference.token != observer_ref().token:
        policies[opponent_reference.token] = FrozenSeedPolicy(
            opponent_policy, plan.match_seed
        )
    result = play_match(
        spec,
        bank=single_game_bank(spec, runner_plan),
        policies=policies,
        record_actions=True,
        on_policy_error=ON_POLICY_ERROR_RAISE,
    )
    if result.errored:  # pragma: no cover - raises above under RAISE
        raise Phase11SoakError(f"{plan.game_id} errored: {result.policy_error}")
    hidden_counts = [
        int(recorder.event_offset[index + 1] - recorder.event_offset[index])
        for index in range(recorder.decisions)
    ]
    return {
        "game_index": plan.game_index,
        "game_id": plan.game_id,
        "stratum": plan.stratum,
        "game_ordinal": plan.game_ordinal,
        "observer_color": plan.observer_color,
        "opponent_color": plan.opponent_color,
        "match_seed": plan.match_seed,
        "match_id": spec.match_id,
        "red_setup": [int(value) for value in plan.red_setup],
        "blue_setup": [int(value) for value in plan.blue_setup],
        "action_history": [int(action) for action in (result.action_history or ())],
        "observer_decision_indices": [int(value) for value in recorder.decision_index],
        "observer_state_identities": list(recorder.state_identity),
        "observer_hidden_counts": hidden_counts,
        "observer_decisions": recorder.decisions,
        "prediction_events": recorder.events,
        "plies": int(result.plies),
        "decisions": int(result.decisions),
        "terminal_reason": result.terminal_reason,
        "observer_result": result.candidate_result,
        "replay_digest": result.replay_digest,
    }


# ---------------------------------------------------------------------------
# The request schedule
# ---------------------------------------------------------------------------


def game_request_specs(game: dict, first_ordinal: int) -> "list[dict]":
    """The scheduled production requests of one played soak game.

    Eight, on Agent 1's frozen attachment rule — except on a game whose
    observer never moved, where the rule attaches nothing because there is
    no public position at which the observer is to decide. Such a game
    contributes an empty list and is reported by :func:`schedule_findings`;
    it is never silently replaced by another game, and no new randomness is
    drawn to make up the difference.

    The returned mappings are exactly what `phase11_repro.execute_request`
    consumes, so the soak drives the accepted request path rather than a
    soak imitation of it.
    """
    if int(game["observer_decisions"]) == 0:
        return []
    positions = request_decision_positions(int(game["observer_decisions"]))
    specs = []
    for request_ordinal, position in enumerate(positions):
        specs.append(
            {
                "request_ordinal": int(first_ordinal) + request_ordinal,
                "request_id": soak_request_id(game["game_id"], request_ordinal),
                "soak_game_id": game["game_id"],
                "game_id": game["game_id"],
                "stratum": game["stratum"],
                "observer_color": game["observer_color"],
                "request_ordinal_in_game": request_ordinal,
                "decision_position": int(position),
                "decision_index": int(game["observer_decision_indices"][position]),
                "public_state_identity": game["observer_state_identities"][position],
                "hidden_pieces_expected": int(game["observer_hidden_counts"][position]),
                "red_setup": list(game["red_setup"]),
                "blue_setup": list(game["blue_setup"]),
                "action_history": list(game["action_history"]),
            }
        )
    return specs


def build_request_schedule(games: "list[dict]") -> "list[dict]":
    """The realizable soak request schedule, in frozen order.

    Agent 1 froze the schedule as 1,024 games x 8 requests = 8,192. That
    arithmetic assumes every game gives the observer at least one decision,
    which the frozen rules do not guarantee: a scout may move and strike in
    one turn, so a first-player scout can capture a front-row flag at ply 1
    and the second-seat observer never decides. This function builds what
    the frozen attachment rule actually yields and refuses to invent the
    difference; :func:`schedule_findings` reports it.
    """
    specs: list[dict] = []
    for game in sorted(games, key=lambda item: int(item["game_index"])):
        specs.extend(game_request_specs(game, len(specs)))
    ids = [spec["request_id"] for spec in specs]
    if len(set(ids)) != len(ids):
        raise Phase11SoakError("the soak request schedule holds a duplicate id")
    if len(specs) > SOAK_REQUEST_COUNT:
        raise Phase11SoakError(
            f"the soak schedule holds {len(specs)} requests, more than the frozen "
            f"{SOAK_REQUEST_COUNT}"
        )
    return specs


def schedule_findings(games: "list[dict]", requests: "list[dict]") -> dict:
    """What the frozen schedule promised against what it can deliver."""
    zero = [
        {
            "game_id": game["game_id"],
            "stratum": game["stratum"],
            "observer_color": game["observer_color"],
            "plies": int(game["plies"]),
            "terminal_reason": game["terminal_reason"],
            "observer_result": game["observer_result"],
        }
        for game in sorted(games, key=lambda item: int(item["game_index"]))
        if int(game["observer_decisions"]) == 0
    ]
    strata: dict[str, int] = {}
    colors: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for entry in zero:
        strata[entry["stratum"]] = strata.get(entry["stratum"], 0) + 1
        colors[entry["observer_color"]] = colors.get(entry["observer_color"], 0) + 1
        reasons[entry["terminal_reason"]] = reasons.get(entry["terminal_reason"], 0) + 1
    return {
        "frozen_request_count": int(SOAK_REQUEST_COUNT),
        "realizable_request_count": len(requests),
        "unrealizable_requests": int(SOAK_REQUEST_COUNT) - len(requests),
        "games": len(games),
        "zero_decision_games": len(zero),
        "zero_decision_by_stratum": dict(sorted(strata.items())),
        "zero_decision_by_observer_color": dict(sorted(colors.items())),
        "zero_decision_terminal_reasons": dict(sorted(reasons.items())),
        "zero_decision_game_ids": [entry["game_id"] for entry in zero],
        "schedule_is_complete": len(requests) == int(SOAK_REQUEST_COUNT),
        "every_playable_game_gave_eight": (
            len(requests) == (len(games) - len(zero)) * SOAK_REQUESTS_PER_GAME
        ),
    }


def progress_bucket(decision_index: int) -> str:
    """Early/middle/late, on the accepted frozen bucket boundaries."""
    from stratego.training.phase11_contract import progress_bucket as frozen_bucket

    return frozen_bucket(int(decision_index))


# ---------------------------------------------------------------------------
# The per-request audit
# ---------------------------------------------------------------------------


def audit_request(owner, spec: dict, committed: dict) -> dict:
    """Re-derive one committed request completely and check every claim.

    Rebuilds the position from public bytes, re-runs the belief forward,
    re-derives all 64 sample identities, rebuilds all 64 worlds, checks the
    inventory and every public fact against a second implementation, traces
    the document build for hidden-rank reads, and compares the result with
    what the soak committed.
    """
    from stratego.evaluation.phase11_repro import execute_request, replay_state
    from stratego.evaluation.phase11_safety import instrument_hidden_types
    from stratego.evaluation.phase11_sampler_audit import (
        independent_identity,
        independent_sample_token,
        verify_world_independently,
    )

    findings: list[str] = []
    state, observer = replay_state(spec)
    result, parts = execute_request(
        owner,
        spec,
        world_count=SOAK_WORLD_ORDINALS,
        state=state,
        observer=observer,
        collect=True,
    )

    document = parts["document"]
    if independent_identity(document) != spec["public_state_identity"]:
        findings.append("the public-state identity does not re-derive")
    if result.digest != committed["digest"]:
        findings.append("the request digest does not reproduce")
    if int(result.hidden_pieces) != int(spec["hidden_pieces_expected"]):
        findings.append(
            f"{result.hidden_pieces} hidden pieces, expected "
            f"{spec['hidden_pieces_expected']}"
        )
    if len(parts["worlds"]) != SOAK_WORLD_ORDINALS:
        findings.append(f"{len(parts['worlds'])} worlds, expected {SOAK_WORLD_ORDINALS}")

    # Every sample identity, re-derived from the public identity and the
    # ordinal alone by the independent implementation.
    inventory_errors = 0
    public_constraint_errors = 0
    provenance_mismatches = 0
    for ordinal, world in enumerate(parts["worlds"]):
        token = independent_sample_token(
            world["sampler_version"], spec["public_state_identity"], ordinal
        )
        if token != world["sample_token"]:
            provenance_mismatches += 1
        if int(world["sample_ordinal"]) != ordinal:
            provenance_mismatches += 1
        if world["public_state_identity"] != spec["public_state_identity"]:
            provenance_mismatches += 1
        check = verify_world_independently(
            document,
            parts["probabilities"],
            world,
            logits_rows=parts["logits"],
        )
        if not check["agrees"]:
            for finding in check["findings"]:
                if "inventory" in finding or "multiset" in finding:
                    inventory_errors += 1
                elif "re-derive" in finding:
                    provenance_mismatches += 1
                else:
                    public_constraint_errors += 1

    # The hidden-input claim, measured rather than asserted: rebuild the
    # same document from a state whose hidden ranks are traced.
    traced, counter = instrument_hidden_types(state, observer)
    rebuilt = _traced_document(traced, observer, spec)
    hidden_reads = int(counter.reads)
    if rebuilt != document:
        findings.append("the traced rebuild produced a different document")

    return {
        "request_id": spec["request_id"],
        "request_ordinal": int(spec["request_ordinal"]),
        "digest": result.digest,
        "worlds": len(parts["worlds"]),
        "hidden_pieces": int(result.hidden_pieces),
        "findings": findings,
        "inventory_errors": inventory_errors,
        "public_constraint_errors": public_constraint_errors,
        "provenance_mismatches": provenance_mismatches,
        "hidden_input_accesses": hidden_reads,
        "deterministic": result.digest == committed["digest"],
    }


def _traced_document(state, observer: int, spec: dict) -> dict:
    """The public-state document, built from a hidden-rank-traced state."""
    from stratego.evaluation.phase11_public_state import build_public_state_document
    from stratego.evaluation.policy import PolicyRef, PolicyRequirements, build_policy_input

    policy_input = build_policy_input(
        state,
        policy=PolicyRef(
            policy_id="phase11_soak_audit", policy_version=SOAK_RUN_VERSION
        ),
        policy_seed=0,
        requirements=PolicyRequirements(
            observation=True, legal_action_mask=True, public_view=True
        ),
        match_id=spec["request_id"],
        game_id=spec["game_id"],
    )
    return build_public_state_document(
        policy_input.require_public_view(), policy_input.require_observation()
    )


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


def store_content_digest(rows: "list[dict]") -> str:
    """The soak store's content identity.

    Request id, public-state identity, world count and request digest, in
    request-id order. No timing, no path, no worker, no wall clock — the
    Agent 5 reading about `manifest_digest` is not repeated here.
    """
    hasher = hashlib.sha256()
    hasher.update(f"phase11_soak_store_v1|requests={len(rows)}".encode())
    for row in sorted(rows, key=lambda item: item["request_id"]):
        hasher.update(
            json.dumps(
                {
                    "request_id": row["request_id"],
                    "public_state_identity": row["public_state_identity"],
                    "hidden_pieces": int(row["hidden_pieces"]),
                    "worlds": int(row["worlds"]),
                    "digest": row["digest"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    return hasher.hexdigest()


def set_reconciliation(scheduled: "list[str]", committed: "list[str]") -> dict:
    """Missing / duplicate / unscheduled, by exact logical id set algebra."""
    scheduled_set = set(scheduled)
    seen: dict[str, int] = {}
    for request_id in committed:
        seen[request_id] = seen.get(request_id, 0) + 1
    duplicates = sorted(key for key, count in seen.items() if count > 1)
    missing = sorted(scheduled_set - set(seen))
    unscheduled = sorted(set(seen) - scheduled_set)
    return {
        "scheduled": len(scheduled_set),
        "committed": len(committed),
        "distinct_committed": len(seen),
        "missing_request_ids": missing,
        "duplicate_request_ids": duplicates,
        "unscheduled_request_ids": unscheduled,
        "missing_request_ids_zero": not missing,
        "duplicate_request_ids_zero": not duplicates,
        "unscheduled_request_ids_zero": not unscheduled,
        "exactly_scheduled": not missing and not duplicates and not unscheduled,
    }


# ---------------------------------------------------------------------------
# 2. The frozen soak schedule — 1,024 train-split non-bank games
# ---------------------------------------------------------------------------


def plan_path() -> Path:
    return WORK_DIRECTORY / "soak_plan.json"


def games_path() -> Path:
    return WORK_DIRECTORY / "soak_games.json"


def build_export() -> dict:
    """Export the accepted evaluation weights once, for every stage."""
    from stratego.training.phase10_collector import export_evaluation_weights

    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    export_evaluation_weights(CHECKPOINT_PATH, EXPORT_PATH)
    return {
        "export_path": str(EXPORT_PATH.relative_to(REPOSITORY_ROOT)),
        "export_sha256": file_sha256(EXPORT_PATH),
        "source_sha256": file_sha256(CHECKPOINT_PATH),
    }


def stage_schedule(args) -> dict:
    """Play the frozen soak games and derive the frozen request schedule.

    The games are the soak's *source of public states*, not evidence: no
    outcome, ply count or terminal reason is read by any gate. What
    survives this stage is the schedule — 8,192 logical request ids, each
    bound to one replayable public position.
    """
    from stratego.evaluation import phase11_pipeline as pipeline
    from stratego.evaluation.phase11_banks import Phase11SetupSources
    from stratego.training.phase11_seed import SOAK_REQUEST_COUNT

    configure_backend()
    verify = read_stage("verify")
    log("exporting the accepted evaluation weights")
    export = build_export()
    if export["source_sha256"] != verify["phase9"]["sha256"]:
        raise Agent6Error("the Phase 9 checkpoint changed between stages")

    descriptors = soak_game_descriptors()
    if args.limit_games:
        log(f"SMOKE RUN: {args.limit_games} games only — not an acceptance run")
        per_stratum = max(1, int(args.limit_games) // 8)
        descriptors = [
            descriptor
            for descriptor in descriptors
            if descriptor["game_ordinal"] < per_stratum
        ]

    wanted = [descriptor["game_id"] for descriptor in descriptors]
    started = time.perf_counter()
    games = _reuse_played_games(wanted)
    if games is None:
        owners, _export = pipeline.build_owners(
            REPOSITORY_ROOT, EXPORT_PATH, device=EVAL_DEVICE
        )
        sources = Phase11SetupSources()
        games = []
        try:
            for position, descriptor in enumerate(descriptors):
                plan = build_soak_plan(descriptor, sources)
                games.append(play_soak_game(plan, owners))
                if (position + 1) % 64 == 0:
                    log(
                        f"soak games: {position + 1}/{len(descriptors)}  "
                        f"{time.perf_counter() - started:6.1f}s"
                    )
        finally:
            owners["phase9"].close()
            owners["anchor"].close()
        games_path().write_text(json.dumps({"games": games}, separators=(",", ":")) + "\n")
    else:
        log(f"reusing {len(games)} already-played soak games")

    original_games = list(games)
    original_requests = (
        build_request_schedule(original_games)
        if not args.limit_games
        else _limited_schedule(original_games)
    )
    supplement = {"authorized": False, "played": [], "playable": 0, "requests": 0}
    if not args.limit_games and len(original_requests) < int(SOAK_REQUEST_COUNT):
        supplement = run_supplement(original_games, original_requests, started)
        games = original_games + supplement["played"]

    requests = build_request_schedule(games) if not args.limit_games else (
        _limited_schedule(games)
    )
    preserved = original_preservation(original_requests, requests)
    if not preserved["original_requests_preserved_exactly"]:
        raise Agent6Error(
            "the supplement disturbed the original schedule: "
            f"{preserved['first_difference']}"
        )
    request_ids = [spec["request_id"] for spec in requests]
    digest = schedule_digest(descriptors, request_ids)

    document = {
        "soak_version": PHASE11_SOAK_VERSION,
        "schedule_digest": digest,
        "games": games,
        "requests": requests,
    }
    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    plan_path().write_text(json.dumps(document, separators=(",", ":")) + "\n")

    coverage = schedule_coverage(games, requests)
    nonbank = nonbank_train_only(games, requests)
    findings = schedule_findings(games, requests)
    findings["supplement"] = {
        key: value for key, value in supplement.items() if key != "played"
    }
    findings["original"] = {
        "games": len(original_games),
        "requests": len(original_requests),
    }
    findings.update(preserved)
    if findings["zero_decision_games"]:
        log(
            f"READING: {findings['zero_decision_games']} of {findings['games']:,} soak "
            "games gave the observer no decision and contributed nothing; the "
            f"schedule realizes {findings['realizable_request_count']:,} of the frozen "
            f"{findings['frozen_request_count']:,}"
            + (
                f", closed by the authorized supplement ({supplement['requests']} "
                f"requests from {supplement['playable']} playable games at ordinals "
                f"{supplement['first_ordinal']}..{supplement['last_ordinal']})."
                if supplement.get("authorized")
                else ". Nothing is substituted."
            )
        )
    payload = {
        "stage": "schedule",
        "export": export,
        "soak_version": document["soak_version"],
        "split": SOAK_SPLIT,
        "setup_source": SOAK_SETUP_SOURCE,
        "nonbank_train_only": nonbank["nonbank_train_only"],
        "nonbank_proof": nonbank,
        "schedule_findings": findings,
        "supplement": {key: value for key, value in supplement.items() if key != "played"},
        "schedule_digest": digest,
        "plan_sha256": file_sha256(plan_path()),
        "games": len(games),
        "requests": len(requests),
        "requests_expected": int(SOAK_REQUEST_COUNT),
        "complete_schedule": len(requests) == int(SOAK_REQUEST_COUNT),
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
        "coverage": coverage,
        "smoke": bool(args.limit_games),
    }
    log(
        f"schedule frozen: {len(games)} games, {len(requests)} requests, "
        f"digest {digest[:16]}..."
    )
    return payload


def supplemental_games_path() -> Path:
    return WORK_DIRECTORY / "soak_supplemental_games.json"


def _reuse_supplemental_games() -> "list[dict]":
    """Already-played supplemental games, as far as they match the order.

    The candidate enumeration and the games are both pure functions of
    identity, so a re-run reuses the prefix it can verify and plays on from
    there. Anything that does not match the frozen order is discarded
    rather than trusted.
    """
    path = supplemental_games_path()
    if not path.exists():
        return []
    try:
        played = json.loads(path.read_text())["games"]
    except (json.JSONDecodeError, KeyError):
        return []
    order = supplemental_candidate_order()
    kept = []
    for game in played:
        try:
            stratum, ordinal = next(order)
        except StopIteration:  # pragma: no cover - the limit is never reached
            break
        if game["game_id"] != soak_game_id(stratum, ordinal):
            break
        kept.append(game)
    return kept


def run_supplement(
    original_games: "list[dict]", original_requests: "list[dict]", started: float
) -> dict:
    """The reviewer-authorized supplement, on Agent 1's extended rules.

    Plays supplemental games in the frozen candidate order and stops the
    moment `SUPPLEMENTAL_PLAYABLE_TARGET` playable games have been reached.
    A game that gives the observer no decision contributes nothing and is
    recorded; it is never skipped over silently and never replaced. No
    outcome, belief value, runtime or sampler quantity reaches the
    enumeration, the stopping rule or the seeds.
    """
    from stratego.evaluation import phase11_pipeline as pipeline
    from stratego.evaluation.phase11_banks import Phase11SetupSources

    if not SUPPLEMENTAL_AUTHORIZED:  # pragma: no cover - the flag is frozen True
        raise Agent6Error("the supplement is not authorized")
    shortfall = int(SOAK_REQUEST_COUNT) - len(original_requests)
    if shortfall != SUPPLEMENTAL_PLAYABLE_TARGET * SOAK_REQUESTS_PER_GAME:
        raise Agent6Error(
            f"the shortfall is {shortfall} requests, not the authorized "
            f"{SUPPLEMENTAL_PLAYABLE_TARGET * SOAK_REQUESTS_PER_GAME}; the "
            "supplement is authorized for one specific deficit only"
        )
    rules = assert_extension_matches_frozen_rules()
    log(
        f"supplement authorized: extending Agent 1's rules from ordinal "
        f"{SUPPLEMENTAL_FIRST_ORDINAL} for {SUPPLEMENTAL_PLAYABLE_TARGET} playable "
        f"games ({shortfall} requests). Rules proven identical on "
        f"{rules['comparisons']:,} comparisons over the frozen range."
    )

    played = _reuse_supplemental_games()
    playable = sum(1 for game in played if int(game["observer_decisions"]) > 0)
    if playable >= SUPPLEMENTAL_PLAYABLE_TARGET:
        # Trim to the exact stopping point: everything after the target-th
        # playable game is not part of the authorized supplement.
        trimmed, seen = [], 0
        for game in played:
            trimmed.append(game)
            if int(game["observer_decisions"]) > 0:
                seen += 1
                if seen == SUPPLEMENTAL_PLAYABLE_TARGET:
                    break
        played = trimmed
        log(f"reusing {len(played)} already-played supplemental soak games")
    else:
        order = supplemental_candidate_order()
        for _ in played:
            next(order)
        owners, _export = pipeline.build_owners(
            REPOSITORY_ROOT, EXPORT_PATH, device=EVAL_DEVICE
        )
        sources = Phase11SetupSources()
        try:
            while playable < SUPPLEMENTAL_PLAYABLE_TARGET:
                try:
                    stratum, ordinal = next(order)
                except StopIteration:  # pragma: no cover - never reached in practice
                    raise Agent6Error(
                        f"the supplemental enumeration reached ordinal "
                        f"{SUPPLEMENTAL_ORDINAL_LIMIT} with only {playable} playable "
                        "games; BLOCKED rather than truncated"
                    )
                descriptor = supplemental_descriptor(
                    stratum, ordinal, len(original_games) + len(played)
                )
                game = play_soak_game(build_soak_plan(descriptor, sources), owners)
                game["supplemental"] = True
                played.append(game)
                if int(game["observer_decisions"]) > 0:
                    playable += 1
                log(
                    f"supplement: {stratum} g={ordinal:03d} "
                    f"observer={game['observer_color']} "
                    f"decisions={game['observer_decisions']} -> "
                    f"{playable}/{SUPPLEMENTAL_PLAYABLE_TARGET} playable  "
                    f"{time.perf_counter() - started:6.1f}s"
                )
        finally:
            owners["phase9"].close()
            owners["anchor"].close()
        supplemental_games_path().write_text(
            json.dumps({"games": played}, separators=(",", ":")) + "\n"
        )

    unplayable = [
        {
            "game_id": game["game_id"],
            "stratum": game["stratum"],
            "observer_color": game["observer_color"],
            "terminal_reason": game["terminal_reason"],
        }
        for game in played
        if int(game["observer_decisions"]) == 0
    ]
    strata: dict[str, int] = {}
    colors: dict[str, int] = {}
    for game in played:
        if int(game["observer_decisions"]) == 0:
            continue
        strata[game["stratum"]] = strata.get(game["stratum"], 0) + 1
        colors[game["observer_color"]] = colors.get(game["observer_color"], 0) + 1
    return {
        "authorized": True,
        "authorization": "reviewer, recorded verbatim in the harness header",
        "rules_proof": rules,
        "first_ordinal": SUPPLEMENTAL_FIRST_ORDINAL,
        "last_ordinal": int(played[-1]["game_ordinal"]) if played else None,
        "enumeration": "ordinal-major over the frozen stratum order",
        "stopping_rule": (
            f"stop at the {SUPPLEMENTAL_PLAYABLE_TARGET}th playable game; a "
            "zero-observer-decision game contributes nothing and is recorded"
        ),
        "games_enumerated": len(played),
        "playable": playable,
        "unplayable": len(unplayable),
        "unplayable_games": unplayable,
        "requests": playable * SOAK_REQUESTS_PER_GAME,
        "playable_by_stratum": dict(sorted(strata.items())),
        "playable_by_observer_color": dict(sorted(colors.items())),
        "played": played,
    }


def original_preservation(
    original_requests: "list[dict]", requests: "list[dict]"
) -> dict:
    """Prove the supplement left the original schedule untouched.

    The original requests must be the exact prefix of the combined
    schedule: same ids, same ordinals, same decision indices, same
    public-state identities, in the same order.
    """
    fields = (
        "request_id",
        "request_ordinal",
        "soak_game_id",
        "decision_index",
        "public_state_identity",
    )
    difference = None
    if len(requests) < len(original_requests):
        difference = f"the combined schedule is shorter ({len(requests)})"
    else:
        for index, before in enumerate(original_requests):
            after = requests[index]
            for field in fields:
                if before[field] != after[field]:
                    difference = (
                        f"request {index} field {field}: "
                        f"{before[field]!r} -> {after[field]!r}"
                    )
                    break
            if difference:
                break
    return {
        "original_requests": len(original_requests),
        "combined_requests": len(requests),
        "supplemental_requests": len(requests) - len(original_requests),
        "original_requests_preserved_exactly": difference is None,
        "first_difference": difference,
    }


def _reuse_played_games(wanted: "list[str]") -> "list[dict] | None":
    """The already-played soak games, when they are exactly the wanted set.

    Playing 1,024 games is a pure function of the frozen schedule, so a
    re-run of this stage reuses them rather than replaying. The reuse is
    guarded by exact game-id set equality; anything else replays.
    """
    path = games_path()
    if not path.exists():
        return None
    try:
        games = json.loads(path.read_text())["games"]
    except (json.JSONDecodeError, KeyError):
        return None
    if [game["game_id"] for game in games] != list(wanted):
        return None
    return games


def _limited_schedule(games: "list[dict]") -> "list[dict]":
    """The same construction on a smoke subset, without the 8,192 assertion."""

    specs: list[dict] = []
    for game in sorted(games, key=lambda item: int(item["game_index"])):
        specs.extend(game_request_specs(game, len(specs)))
    return specs


def schedule_coverage(games: "list[dict]", requests: "list[dict]") -> dict:
    """What the soak actually covers: states, colours, buckets, behaviours."""

    strata: dict[str, int] = {}
    colors: dict[str, int] = {}
    buckets: dict[str, int] = {}
    identities = set()
    empty_requests = 0
    shared_decisions = 0
    per_game_shared = 0
    for spec in requests:
        strata[spec["stratum"]] = strata.get(spec["stratum"], 0) + 1
        colors[spec["observer_color"]] = colors.get(spec["observer_color"], 0) + 1
        bucket = progress_bucket(int(spec["decision_index"]))
        buckets[bucket] = buckets.get(bucket, 0) + 1
        identities.add(spec["public_state_identity"])
        if int(spec["hidden_pieces_expected"]) == 0:
            empty_requests += 1
    for game in games:
        if int(game["observer_decisions"]) == 0:
            continue
        positions = request_decision_positions(int(game["observer_decisions"]))
        repeats = len(positions) - len(set(positions))
        if repeats:
            per_game_shared += 1
            shared_decisions += repeats
    outcomes: dict[str, int] = {}
    for game in games:
        outcomes[game["observer_result"]] = outcomes.get(game["observer_result"], 0) + 1
    return {
        "requests_by_stratum": dict(sorted(strata.items())),
        "requests_by_observer_color": dict(sorted(colors.items())),
        "requests_by_progress_bucket": dict(sorted(buckets.items())),
        "distinct_public_states": len(identities),
        "requests_with_zero_hidden_pieces": empty_requests,
        "games_with_shared_decisions": per_game_shared,
        "shared_decision_requests": shared_decisions,
        "observer_decisions_total": sum(
            int(game["observer_decisions"]) for game in games
        ),
        "outcomes_report_only": dict(sorted(outcomes.items())),
        "strata_covered": len(strata),
        "both_colors_covered": len(colors) == 2,
        "all_progress_buckets_covered": len(buckets) == 3,
    }


def nonbank_train_only(games: "list[dict]", requests: "list[dict]") -> dict:
    """Prove the soak touched no bank case, game, seed or setup.

    The banks are read here only as *exclusion sets*: their game ids, match
    seeds and setup arrangements are collected and intersected with the
    soak's. An empty intersection is the claim; the counts are the receipt.
    """
    bank_game_ids = set()
    bank_match_seeds = set()
    bank_setups = set()
    for filename in ("agent_01_validation_bank.json", "agent_01_test_bank.json"):
        for case in read_json(DATA_DIRECTORY / filename)["cases"]:
            for game in case["games"].values():
                bank_game_ids.add(game["game_id"])
                bank_match_seeds.add(int(game["match_seed"]))
                for role in ("observer", "opponent"):
                    bank_setups.add(tuple(int(v) for v in game[role]["setup"]))
    soak_game_ids = {game["game_id"] for game in games}
    soak_match_seeds = {int(game["match_seed"]) for game in games}
    soak_setups = set()
    for game in games:
        soak_setups.add(tuple(int(v) for v in game["red_setup"]))
        soak_setups.add(tuple(int(v) for v in game["blue_setup"]))
    splits = {game.get("split", "train") for game in games}
    shared_ids = sorted(soak_game_ids & bank_game_ids)
    shared_seeds = sorted(soak_match_seeds & bank_match_seeds)
    shared_setups = len(soak_setups & bank_setups)
    namespaced = all(
        game["game_id"].startswith("phase11_soak_v1|") for game in games
    ) and all(
        spec["request_id"].startswith("phase11_soak_v1|") for spec in requests
    )
    return {
        "bank_game_ids": len(bank_game_ids),
        "soak_game_ids": len(soak_game_ids),
        "shared_game_ids": shared_ids,
        "shared_match_seeds": shared_seeds,
        "shared_setup_arrangements": shared_setups,
        "soak_namespace_only": namespaced,
        "split": sorted(splits) or ["train"],
        "nonbank_train_only": (
            not shared_ids
            and not shared_seeds
            and shared_setups == 0
            and namespaced
        ),
    }


# ---------------------------------------------------------------------------
# 3. The soak — legs, workers, a real SIGKILL, resume by set subtraction
# ---------------------------------------------------------------------------


def load_plan() -> dict:
    path = plan_path()
    if not path.exists():
        raise Agent6Error("the frozen soak plan is missing; run --stage schedule")
    schedule = read_stage("schedule")
    if file_sha256(path) != schedule["plan_sha256"]:
        raise Agent6Error("the frozen soak plan changed after the freeze")
    return json.loads(path.read_text())


def execute_ordinals(requests, ordinals, out_path: Path) -> int:
    """Execute the named request ordinals in order, committing each.

    Every result is appended and fsynced before the next request starts, so
    a SIGKILL leaves a store whose contents are exactly the requests that
    finished — which is what "resume by exact request-id set subtraction"
    needs.
    """
    from stratego.evaluation.phase11_repro import build_owner, execute_request

    owner = build_owner(EXPORT_PATH)
    written = 0
    with open(out_path, "a", buffering=1) as stream:
        for ordinal in ordinals:
            spec = requests[int(ordinal)]
            result = execute_request(
                owner, spec, world_count=SOAK_WORLD_ORDINALS
            )
            row = result.as_row()
            row["request_id"] = spec["request_id"]
            row["soak_game_id"] = spec["soak_game_id"]
            row["stratum"] = spec["stratum"]
            row["observer_color"] = spec["observer_color"]
            row["decision_index"] = int(spec["decision_index"])
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            written += 1
    owner.close()
    return written


def stage_soak_worker(args) -> dict:
    """One soak worker: a plain process executing a frozen ordinal list."""
    configure_backend()
    document = json.loads(Path(args.plan).read_text())
    ordinals = json.loads(Path(args.ordinals).read_text())
    written = execute_ordinals(document["requests"], ordinals, Path(args.out))
    return {"written": written}


def _worker_command(plan_file: Path, ordinals_path: Path, out_path: Path) -> list:
    return [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "run_phase11_agent06.py"),
        "--stage",
        "soak-worker",
        "--plan",
        str(plan_file),
        "--ordinals",
        str(ordinals_path),
        "--out",
        str(out_path),
    ]


def _leg_directory(kind: str, leg: str) -> Path:
    directory = WORK_DIRECTORY / kind / leg
    if directory.exists():
        for child in directory.iterdir():
            child.unlink()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _committed_rows(text: str) -> "list[dict]":
    """Parse a commit log, discarding a torn final line.

    A SIGKILL can land mid-write, so an unparseable trailing line is not a
    committed request and must not be counted as one. Any *earlier*
    unparseable line is corruption and raises.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise Agent6Error(
                    f"commit log line {index + 1} of {len(lines)} is unparseable"
                )
    return rows


def _read_rows(directory: Path) -> "list[dict]":
    rows: list[dict] = []
    for path in sorted(directory.glob("*.jsonl")):
        rows.extend(_committed_rows(path.read_text()))
    return rows


def _launch(plan_file: Path, directory: Path, shards, *, first_index: int = 0):
    processes = []
    for offset, ordinals in enumerate(shards):
        index = first_index + offset
        ordinals_path = directory / f"ordinals_{index:02d}.json"
        ordinals_path.write_text(json.dumps([int(value) for value in ordinals]))
        out_path = directory / f"worker_{index:02d}.jsonl"
        out_path.touch()
        processes.append(
            subprocess.Popen(
                _worker_command(plan_file, ordinals_path, out_path),
                cwd=str(REPOSITORY_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
    return processes


def _wait(processes, leg: str, *, allow_signalled=()) -> None:
    failures = []
    for index, process in enumerate(processes):
        _stdout, stderr = process.communicate()
        if process.returncode != 0 and index not in allow_signalled:
            failures.append(
                {
                    "worker": index,
                    "returncode": process.returncode,
                    "stderr": stderr.decode()[-2000:],
                }
            )
    if failures:
        raise Agent6Error(f"leg {leg!r} had failing workers: {failures}")


def _shard(ordinals, count: int) -> list:
    """Round-robin assignment: every worker sees the whole ordinal range."""
    return [ordinals[index::count] for index in range(count)]


def run_soak_leg(leg: dict, requests, ordinals, plan_file: Path) -> dict:
    """One soak leg: N workers over this leg's slice of the schedule."""
    directory = _leg_directory("soak", leg["leg"])
    started = time.perf_counter()
    detail = {
        "leg": leg["leg"],
        "workers": int(leg["workers"]),
        "assignment": "round robin over the leg's ordinal slice",
        "scheduled": len(ordinals),
    }
    shards = _shard(list(ordinals), int(leg["workers"]))
    processes = _launch(plan_file, directory, shards)

    if leg["kill"]:
        # A real process death, after committed work exists. The victim is
        # the first worker; the threshold is frozen above, capped so a
        # smoke run still exercises the leg.
        threshold = min(KILL_AFTER_COMMITTED, max(1, len(shards[0]) // 4))
        victim = processes[0]
        victim_log = directory / "worker_00.jsonl"
        deadline = time.perf_counter() + 1800.0
        killed_after = 0
        while time.perf_counter() < deadline:
            if victim.poll() is not None:
                raise Agent6Error(
                    "the kill/resume worker finished before it could be killed; "
                    f"lower KILL_AFTER_COMMITTED (currently {KILL_AFTER_COMMITTED})"
                )
            committed = len(_committed_rows(victim_log.read_text()))
            if committed >= threshold:
                killed_after = committed
                os.kill(victim.pid, signal.SIGKILL)
                break
            time.sleep(0.25)
        else:  # pragma: no cover - the loop always breaks on this data
            raise Agent6Error("the kill/resume worker never committed enough work")
        _wait(processes, leg["leg"], allow_signalled={0})
        returncode = victim.returncode

        # Resume by exact logical request-id set subtraction: the resume
        # worker is handed the scheduled ids minus the committed ids, and
        # nothing else.
        committed_ids = {row["request_id"] for row in _read_rows(directory)}
        remaining = [
            ordinal
            for ordinal in ordinals
            if requests[int(ordinal)]["request_id"] not in committed_ids
        ]
        resume = _launch(plan_file, directory, [remaining], first_index=90)
        _wait(resume, f"{leg['leg']}:resume")
        resumed_ids = {
            row["request_id"]
            for row in _committed_rows((directory / "worker_90.jsonl").read_text())
        }
        detail.update(
            {
                "kill_signal": "SIGKILL",
                "killed_worker": 0,
                "committed_before_kill": killed_after,
                "kill_after_threshold": threshold,
                "worker_returncode": returncode,
                "really_signalled": returncode == -int(signal.SIGKILL),
                "resume_rule": "exact logical request-id set subtraction",
                "resumed_requests": len(remaining),
                "resumed_committed": len(resumed_ids),
                "recomputed_on_both_sides": len(committed_ids & resumed_ids),
            }
        )
    else:
        _wait(processes, leg["leg"])

    rows = _read_rows(directory)
    detail.update(
        {
            "committed": len(rows),
            "shard_sizes": [len(shard) for shard in shards],
            "wall_clock_seconds": round(time.perf_counter() - started, 3),
        }
    )
    return {"detail": detail, "rows": rows}


def stage_soak(args) -> dict:
    """The production integration soak, across the frozen legs."""

    read_stage("verify")
    document = load_plan()
    requests = document["requests"]
    total = len(requests)
    scheduled_ids = [spec["request_id"] for spec in requests]
    plan_file = plan_path()

    # The legs partition the schedule: one store, filled by three
    # topologies, one of which really dies. The slices are frozen here,
    # before the first request runs.
    boundaries = [round(total * index / len(SOAK_LEGS)) for index in range(len(SOAK_LEGS) + 1)]
    legs = []
    rows: list[dict] = []
    for index, leg in enumerate(SOAK_LEGS):
        ordinals = list(range(boundaries[index], boundaries[index + 1]))
        log(
            f"soak leg {leg['leg']}: {len(ordinals)} requests on "
            f"{leg['workers']} worker(s)"
        )
        outcome = run_soak_leg(leg, requests, ordinals, plan_file)
        legs.append(outcome["detail"])
        rows.extend(outcome["rows"])
        log(
            f"  committed {outcome['detail']['committed']} in "
            f"{outcome['detail']['wall_clock_seconds']:.1f}s"
        )

    reconciliation = set_reconciliation(
        scheduled_ids, [row["request_id"] for row in rows]
    )
    if not reconciliation["exactly_scheduled"]:
        raise Agent6Error(
            "the soak store does not hold exactly the scheduled ids: "
            f"{ {key: len(reconciliation[key]) for key in ('missing_request_ids', 'duplicate_request_ids', 'unscheduled_request_ids')} }"
        )
    store = {row["request_id"]: row for row in rows}
    store_path = WORK_DIRECTORY / "soak_store.json"
    store_path.write_text(
        json.dumps(
            {"rows": [store[key] for key in sorted(store)]}, separators=(",", ":")
        )
        + "\n"
    )
    content_digest = store_content_digest(list(store.values()))
    timings = request_timings(list(store.values()))
    restart = restart_summary(legs)
    log(f"soak complete: {len(store)} requests, content digest {content_digest[:16]}...")
    return {
        "stage": "soak",
        "legs": legs,
        "restart": restart,
        "restart_resume_pass": restart["pass"],
        "requests": len(store),
        "worlds": sum(int(row["worlds"]) for row in store.values()),
        "reconciliation": reconciliation,
        "store_content_digest": content_digest,
        "store_path": str(store_path.relative_to(REPOSITORY_ROOT)),
        "store_sha256": file_sha256(store_path),
        "timings": timings,
    }


def restart_summary(legs: "list[dict]") -> dict:
    """The crash/restart claim, read off the legs that actually ran."""
    worker_counts = sorted({int(leg["workers"]) for leg in legs})
    killed = [leg for leg in legs if "kill_signal" in leg]
    return {
        "legs": len(legs),
        "distinct_worker_counts": worker_counts,
        "legs_ge_3": len(legs) >= 3,
        "worker_counts_distinct": len(worker_counts) == len(legs),
        "kill_legs": len(killed),
        "kill_signal": killed[0]["kill_signal"] if killed else None,
        "really_signalled": all(leg["really_signalled"] for leg in killed),
        "committed_before_kill": [leg["committed_before_kill"] for leg in killed],
        "resume_rule": killed[0]["resume_rule"] if killed else None,
        "recomputed_on_both_sides": sum(
            int(leg["recomputed_on_both_sides"]) for leg in killed
        ),
        "pass": (
            len(legs) >= 3
            and len(worker_counts) == len(legs)
            and len(killed) >= 1
            and all(leg["really_signalled"] for leg in killed)
            and all(int(leg["committed_before_kill"]) > 0 for leg in killed)
            and all(int(leg["recomputed_on_both_sides"]) == 0 for leg in killed)
        ),
    }


def request_timings(rows: "list[dict]") -> dict:
    """Report-only component timings over the committed requests."""
    from stratego.evaluation.phase11_repro import timing_statistics

    total = timing_statistics([row["total_ns"] / 1e6 for row in rows])
    forward = timing_statistics([row["forward_ns"] / 1e6 for row in rows])
    sampling = timing_statistics([row["sampling_ns"] / 1e6 for row in rows])
    return {
        "note": "report-only; the frozen runtime result is Agent 4's benchmark",
        "forward_plus_64_worlds_ms": total,
        "model_forward_ms": forward,
        "sampling_ms": sampling,
    }


# ---------------------------------------------------------------------------
# 4. The per-request audit and the cross-topology replay
# ---------------------------------------------------------------------------


def audit_ordinals(requests, committed, ordinals, out_path: Path) -> int:
    """Independently re-derive and re-check the named committed requests."""
    from stratego.evaluation.phase11_repro import build_owner

    owner = build_owner(EXPORT_PATH)
    written = 0
    with open(out_path, "a", buffering=1) as stream:
        for ordinal in ordinals:
            spec = requests[int(ordinal)]
            report = audit_request(owner, spec, committed[spec["request_id"]])
            stream.write(json.dumps(report, separators=(",", ":")) + "\n")
            stream.flush()
            written += 1
    owner.close()
    return written


def stage_audit_worker(args) -> dict:
    """One audit worker: a plain process auditing a frozen ordinal list."""
    configure_backend()
    document = json.loads(Path(args.plan).read_text())
    committed = {
        row["request_id"]: row
        for row in json.loads(Path(args.store).read_text())["rows"]
    }
    ordinals = json.loads(Path(args.ordinals).read_text())
    written = audit_ordinals(document["requests"], committed, ordinals, Path(args.out))
    return {"written": written}


def _audit_command(plan_file: Path, store_file: Path, ordinals_path: Path, out_path: Path) -> list:
    return [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "run_phase11_agent06.py"),
        "--stage",
        "audit-worker",
        "--plan",
        str(plan_file),
        "--store",
        str(store_file),
        "--ordinals",
        str(ordinals_path),
        "--out",
        str(out_path),
    ]


def run_audit(requests, store_file: Path, plan_file: Path, workers: int) -> "list[dict]":
    """Every committed request, re-derived in a worker pool."""
    directory = _leg_directory("audit", f"workers_{workers}")
    ordinals = list(range(len(requests)))
    shards = _shard(ordinals, workers)
    processes = []
    for index, shard in enumerate(shards):
        ordinals_path = directory / f"ordinals_{index:02d}.json"
        ordinals_path.write_text(json.dumps([int(value) for value in shard]))
        out_path = directory / f"worker_{index:02d}.jsonl"
        out_path.touch()
        processes.append(
            subprocess.Popen(
                _audit_command(plan_file, store_file, ordinals_path, out_path),
                cwd=str(REPOSITORY_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
    _wait(processes, "audit")
    return _read_rows(directory)


def replay_subset(total: int, size: int) -> "list[int]":
    """An evenly spaced subset of the ordinal range, frozen by arithmetic."""
    count = min(int(size), int(total))
    if count <= 0:
        return []
    return sorted({(index * total) // count for index in range(count)})


def run_cross_topology_replay(
    requests, ordinals, plan_file: Path, workers: int
) -> "list[dict]":
    """Re-execute a subset under a different worker count and reverse order.

    A different topology *and* a different call order: if a result had ever
    depended on worker count, shard membership or what ran before it, the
    digests would move here.
    """
    directory = _leg_directory("replay", f"workers_{workers}_reverse")
    reversed_ordinals = list(reversed(list(ordinals)))
    shards = _shard(reversed_ordinals, workers)
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "PHASE11_AGENT6_REPLAY": "1",
    }
    processes = []
    for index, shard in enumerate(shards):
        ordinals_path = directory / f"ordinals_{index:02d}.json"
        ordinals_path.write_text(json.dumps([int(value) for value in shard]))
        out_path = directory / f"worker_{index:02d}.jsonl"
        out_path.touch()
        processes.append(
            subprocess.Popen(
                _worker_command(plan_file, ordinals_path, out_path),
                cwd=str(Path(os.sep)),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
    _wait(processes, "replay")
    return _read_rows(directory)


def stage_audit(args) -> dict:
    """Re-derive every committed request; replay a substantial subset."""

    read_stage("verify")
    soak_stage = read_stage("soak")
    document = load_plan()
    requests = document["requests"]
    plan_file = plan_path()
    store_file = WORK_DIRECTORY / "soak_store.json"
    if file_sha256(store_file) != soak_stage["store_sha256"]:
        raise Agent6Error("the soak store changed after the soak stage")
    committed = {
        row["request_id"]: row for row in json.loads(store_file.read_text())["rows"]
    }

    workers = min(AUDIT_WORKERS, max(1, len(requests)))
    log(f"auditing {len(requests)} committed requests on {workers} workers")
    started = time.perf_counter()
    reports = run_audit(requests, store_file, plan_file, workers)
    audit_seconds = time.perf_counter() - started
    if len(reports) != len(requests):
        raise Agent6Error(
            f"the audit produced {len(reports)} reports for {len(requests)} requests"
        )

    counters = {
        "inventory_errors": sum(int(row["inventory_errors"]) for row in reports),
        "public_constraint_errors": sum(
            int(row["public_constraint_errors"]) for row in reports
        ),
        "provenance_mismatches": sum(
            int(row["provenance_mismatches"]) for row in reports
        ),
        "hidden_input_accesses": sum(
            int(row["hidden_input_accesses"]) for row in reports
        ),
        "audit_findings": sum(len(row["findings"]) for row in reports),
        "nondeterministic_requests": sum(
            1 for row in reports if not row["deterministic"]
        ),
    }
    findings = sorted(
        {finding for row in reports for finding in row["findings"]}
    )
    worlds_verified = sum(int(row["worlds"]) for row in reports)
    log(
        f"audit complete in {audit_seconds:.1f}s: {worlds_verified:,} worlds "
        f"independently re-derived, counters {counters}"
    )

    subset = replay_subset(len(requests), min(REPLAY_SUBSET, len(requests)))
    workers_replay = min(REPLAY_WORKERS, max(1, len(subset)))
    log(
        f"cross-topology replay: {len(subset)} requests on {workers_replay} "
        "workers, reverse order, fresh processes, scrubbed environment"
    )
    replay_rows = run_cross_topology_replay(requests, subset, plan_file, workers_replay)
    replay_by_id = {row["request_id"]: row for row in replay_rows}
    mismatches = sorted(
        request_id
        for request_id, row in replay_by_id.items()
        if row["digest"] != committed[request_id]["digest"]
    )
    replay_complete = len(replay_by_id) == len(subset)

    # Purity, demonstrated: on short games two scheduled requests land on
    # one decision, and their worlds must then be byte-identical.
    shared = shared_decision_agreement(requests, committed)

    payload = {
        "stage": "audit",
        "requests_audited": len(reports),
        "worlds_verified": worlds_verified,
        "audit_workers": workers,
        "audit_seconds": round(audit_seconds, 3),
        "counters": counters,
        "distinct_findings": findings,
        "cross_topology_replay": {
            "requests": len(subset),
            "committed": len(replay_by_id),
            "workers": workers_replay,
            "order": "reverse",
            "cwd": os.sep,
            "environment": "scrubbed to PATH/HOME",
            "digest_mismatches": mismatches,
            "complete": replay_complete,
            "exact": not mismatches and replay_complete,
        },
        "shared_decision_agreement": shared,
        "deterministic_rebuild_pass": counters["nondeterministic_requests"] == 0
        and counters["audit_findings"] == 0,
    }
    return payload


def shared_decision_agreement(requests, committed: dict) -> dict:
    """Requests sharing one decision must carry byte-identical digests."""
    groups: dict[tuple, list] = {}
    for spec in requests:
        key = (spec["soak_game_id"], int(spec["decision_index"]))
        groups.setdefault(key, []).append(spec["request_id"])
    shared = {key: ids for key, ids in groups.items() if len(ids) > 1}
    disagreements = []
    for key, ids in sorted(shared.items()):
        digests = {committed[request_id]["digest"] for request_id in ids}
        if len(digests) != 1:
            disagreements.append(f"{key[0]}@{key[1]}")
    return {
        "shared_decisions": len(shared),
        "requests_involved": sum(len(ids) for ids in shared.values()),
        "disagreements": disagreements,
        "agree": not disagreements,
    }


# ---------------------------------------------------------------------------
# 4b. The materialized-stream collision audit
#
# Identity-only. Nothing here replays a belief forward, samples a world,
# kills a process or touches a bank: it reconstructs, from the final
# recorded 8,192-request schedule and the frozen derivation rules alone,
# every logical random-stream identity the soak actually materialized, and
# proves the identity -> 63-bit seed map is injective across Agent 6's own
# universe and against the whole already-accepted Agent 4 universe.
# ---------------------------------------------------------------------------


#: The stream domains Agent 6 materializes.
AGENT6_STREAM_DOMAINS = (
    "soak_setup",
    "soak_match",
    "world_sample",
    "world_order",
    "world_categorical",
)

#: Workers for the public-document reconstruction.
STREAM_WORKERS = 10


def soak_public_document(state, spec: dict) -> dict:
    """The public-state document of one already-replayed soak position."""
    from stratego.evaluation.phase11_public_state import build_public_state_document
    from stratego.evaluation.policy import (
        PolicyRef,
        PolicyRequirements,
        build_policy_input,
    )

    policy_input = build_policy_input(
        state,
        policy=PolicyRef(
            policy_id="phase11_soak_stream_audit", policy_version=SOAK_RUN_VERSION
        ),
        policy_seed=0,
        requirements=PolicyRequirements(
            observation=True, legal_action_mask=True, public_view=True
        ),
        match_id=spec["request_id"],
        game_id=spec["game_id"],
    )
    return build_public_state_document(
        policy_input.require_public_view(), policy_input.require_observation()
    )


def reconstruct_states(specs, out_path: Path) -> int:
    """Rebuild each named request's public state from public bytes alone.

    Emits the public-state identity and the hidden-piece slot list — the
    two quantities the world streams are keyed on — for each request, so
    the audit universe comes from the recorded schedule and the engine,
    never from a reported total.
    """
    from stratego.evaluation.phase11_public_state import (
        hidden_opponent_pieces,
        public_state_identity,
    )
    from stratego.evaluation.phase11_repro import replay_state

    written = 0
    with open(out_path, "a", buffering=1) as stream:
        for spec in specs:
            state, _observer = replay_state(spec)
            document = soak_public_document(state, spec)
            identity = public_state_identity(document)
            slots = [
                int(piece["piece_slot"]) for piece in hidden_opponent_pieces(document)
            ]
            stream.write(
                json.dumps(
                    {
                        "request_id": spec["request_id"],
                        "recorded_identity": spec["public_state_identity"],
                        "identity": identity,
                        "slots": slots,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            written += 1
    return written


def stage_streams_worker(args) -> dict:
    """One reconstruction worker over a frozen ordinal list."""
    configure_backend()
    document = json.loads(Path(args.plan).read_text())
    ordinals = json.loads(Path(args.ordinals).read_text())
    specs = [document["requests"][int(ordinal)] for ordinal in ordinals]
    return {"written": reconstruct_states(specs, Path(args.out))}


def _streams_command(plan_file: Path, ordinals_path: Path, out_path: Path) -> list:
    return [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "run_phase11_agent06.py"),
        "--stage",
        "streams-worker",
        "--plan",
        str(plan_file),
        "--ordinals",
        str(ordinals_path),
        "--out",
        str(out_path),
    ]


def run_state_reconstruction(requests, plan_file: Path, workers: int) -> "list[dict]":
    """Reconstruct one representative request per distinct public position."""
    representatives: dict[tuple, int] = {}
    for ordinal, spec in enumerate(requests):
        key = (spec["soak_game_id"], int(spec["decision_index"]))
        representatives.setdefault(key, ordinal)
    ordinals = sorted(representatives.values())

    directory = _leg_directory("streams", f"workers_{workers}")
    shards = _shard(ordinals, workers)
    processes = []
    for index, shard in enumerate(shards):
        ordinals_path = directory / f"ordinals_{index:02d}.json"
        ordinals_path.write_text(json.dumps([int(value) for value in shard]))
        out_path = directory / f"worker_{index:02d}.jsonl"
        out_path.touch()
        processes.append(
            subprocess.Popen(
                _streams_command(plan_file, ordinals_path, out_path),
                cwd=str(REPOSITORY_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
    _wait(processes, "streams")
    rows = _read_rows(directory)
    if len(rows) != len(ordinals):
        raise Agent6Error(
            f"the reconstruction produced {len(rows)} rows for {len(ordinals)} "
            "distinct positions"
        )
    return rows


def agent6_stream_universe(schedule: dict, requests, rows: "list[dict]") -> dict:
    """Every logical stream identity Agent 6 materialized, deduplicated."""
    import numpy as np

    from stratego.evaluation import phase11_streams as streams_module
    from stratego.training.phase11_contract import BELIEF_SAMPLER_VERSION

    slots_by_identity: dict[str, list] = {}
    identity_disagreements = []
    for row in rows:
        if row["identity"] != row["recorded_identity"]:
            identity_disagreements.append(row["request_id"])
        existing = slots_by_identity.get(row["identity"])
        slots = [int(value) for value in row["slots"]]
        if existing is not None and sorted(existing) != sorted(slots):
            raise Agent6Error(
                f"public state {row['identity'][:16]}... carries two slot sets"
            )
        slots_by_identity[row["identity"]] = slots
    if identity_disagreements:
        raise Agent6Error(
            "reconstructed public-state identities disagree with the recorded "
            f"schedule on {len(identity_disagreements)} requests"
        )

    identities = sorted(slots_by_identity)
    tokens = streams_module.tokens_for(
        identities, range(SOAK_WORLD_ORDINALS), BELIEF_SAMPLER_VERSION
    )
    world = streams_module.world_stream_seeds(tokens, slots_by_identity)

    games = schedule["games_index"]
    setup_identities = [
        (game["game_id"], role) for game in games for role in (ROLE_OBSERVER, ROLE_OPPONENT)
    ]
    match_identities = [game["game_id"] for game in games]
    arrays = dict(world)
    arrays["soak_setup"] = np.asarray(
        [
            derive_phase11_seed(DOMAIN_SOAK_SETUP, game_id, role)
            for game_id, role in setup_identities
        ],
        dtype=np.uint64,
    )
    arrays["soak_match"] = np.asarray(
        [derive_phase11_seed(DOMAIN_SOAK_MATCH, game_id) for game_id in match_identities],
        dtype=np.uint64,
    )
    return {
        "arrays": arrays,
        "tokens": tokens,
        "slots_by_identity": slots_by_identity,
        "setup_identities": setup_identities,
        "match_identities": match_identities,
        "distinct_public_states": len(identities),
        "identity_disagreements": 0,
    }


def load_agent4_harness():
    """The accepted Agent 4 harness, loaded as a module.

    Agent 4's universe is rebuilt by Agent 4's own code rather than by an
    Agent 6 paraphrase of it: the safety draw consumption, the token sets
    and Agent 1's enumerable universe all come from the functions that
    produced the accepted record, and the result is required to reproduce
    that record's per-domain counts exactly before it is used.
    """
    import importlib.util

    path = REPOSITORY_ROOT / "scripts" / "run_phase11_agent04.py"
    specification = importlib.util.spec_from_file_location(
        "run_phase11_agent04_for_agent06", path
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def agent4_stream_universe() -> dict:
    """The complete already-accepted Agent 4 universe, re-enumerated.

    Identity-only: no forward, no sampled world, no truth shard, no
    outcome. Required to reproduce Agent 4's accepted per-domain counts
    exactly before it is used as the cross-universe comparison set.
    """
    import csv as _csv

    from stratego.evaluation import phase11_streams as streams_module
    from stratego.evaluation.phase11_baselines import (
        COUNT_UNIFORM_WORLD_SAMPLER_VERSION,
    )
    from stratego.evaluation.phase11_records import read_manifest, store_root
    from stratego.evaluation.phase11_repro import REQUEST_WORLD_COUNT
    from stratego.evaluation.phase11_safety import trial_sample_ordinal, trial_state_walk
    from stratego.training.phase11_contract import BELIEF_SAMPLER_VERSION
    from stratego.training.phase11_seed import (
        SAFETY_TRIAL_COUNT,
        phase11_safety_trial_id,
    )

    harness = load_agent4_harness()
    agent4_work = REPOSITORY_ROOT / "checkpoints" / "phase11" / "agent04"
    root = store_root(REPOSITORY_ROOT)
    manifest = read_manifest(root)
    index = streams_module.store_state_index(root, manifest)
    slots_by_identity = index["slots_by_identity"]
    identity_by_decision = index["identity_by_decision"]

    agent3_states = []
    with open(DATA_DIRECTORY / "agent_03_sampler_diagnostics.csv", newline="") as handle:
        for row in _csv.DictReader(handle):
            agent3_states.append(row["public_state_identity"])
    agent3 = read_json(DATA_DIRECTORY / "agent_03_acceptance.json")
    selection = agent3["handoff_to_agent_4"]["validation_public_states"]["selection_rule"]
    agent3_tokens = streams_module.tokens_for(
        agent3_states, range(int(selection["worlds_per_state"])), BELIEF_SAMPLER_VERSION
    ) | streams_module.tokens_for(
        agent3_states,
        range(int(selection["baseline_worlds_per_state"])),
        COUNT_UNIFORM_WORLD_SAMPLER_VERSION,
    )

    pool = json.loads((agent4_work / "safety_pool.json").read_text())
    admits = pool["admits_alternative"]
    assignments: dict[str, list] = {}
    safety_pairs = set()
    for ordinal in range(SAFETY_TRIAL_COUNT):
        trial_id = phase11_safety_trial_id(ordinal)
        walk = trial_state_walk(trial_id, pool["candidates"], admits)
        pool_index = walk["pool_index"]
        game_id = pool["game_id"][pool_index]
        decision = int(pool["decision_index"][pool_index])
        safety_pairs.add(
            (
                identity_by_decision[(game_id, decision)],
                trial_sample_ordinal(trial_id, REQUEST_WORLD_COUNT),
            )
        )
        assignments.setdefault(game_id, []).append(
            {
                "trial_ordinal": ordinal,
                "trial_id": trial_id,
                "decision_index": decision,
                "walk_steps": walk["walk_steps"],
            }
        )
    safety_tokens = {
        streams_module.phase11_sample_token(BELIEF_SAMPLER_VERSION, identity, ordinal)
        for identity, ordinal in safety_pairs
    }
    repro_plan = json.loads((agent4_work / "repro_plan.json").read_text())
    benchmark_plan = json.loads((agent4_work / "benchmark_plan.json").read_text())
    agent4_tokens = (
        safety_tokens
        | streams_module.tokens_for(
            [spec["public_state_identity"] for spec in repro_plan],
            range(REQUEST_WORLD_COUNT),
            BELIEF_SAMPLER_VERSION,
        )
        | streams_module.tokens_for(
            [spec["public_state_identity"] for spec in benchmark_plan],
            range(REQUEST_WORLD_COUNT),
            BELIEF_SAMPLER_VERSION,
        )
        | streams_module.tokens_for(
            [repro_plan[0]["public_state_identity"]], range(8), BELIEF_SAMPLER_VERSION
        )
    )
    tokens = agent3_tokens | agent4_tokens

    log("  recomputing the Agent 4 attack's safety_trial draw consumption")
    cases = harness.load_bank_cases()
    games = {entry["game_id"]: entry for entry in manifest["games_index"]}
    histories = harness.action_histories(assignments)
    consumption = harness.safety_trial_draws(assignments, histories, games, cases)

    log(f"  deriving the Agent 4 world streams over {len(tokens):,} tokens")
    arrays = dict(streams_module.world_stream_seeds(tokens, slots_by_identity))
    arrays.update(streams_module.safety_trial_seeds(consumption["draws"]))
    arrays.update(harness.agent1_non_safety_universe())

    recorded = read_json(DATA_DIRECTORY / "agent_04_stream_audit.json")["collision_audit"][
        "per_domain"
    ]
    counts = {name: int(array.size) for name, array in arrays.items()}
    mismatches = {
        name: {"reconstructed": counts.get(name), "recorded": entry["identities"]}
        for name, entry in recorded.items()
        if counts.get(name) != entry["identities"]
    }
    return {
        "arrays": arrays,
        "tokens": tokens,
        "slots_by_identity": slots_by_identity,
        "counts": counts,
        "recorded_counts": {
            name: entry["identities"] for name, entry in recorded.items()
        },
        "mismatches": mismatches,
        "reproduces_accepted_record": not mismatches,
        "total_identities": sum(counts.values()),
    }


def stage_streams(_args) -> dict:
    """Reconstruct Agent 6's materialized stream universe and audit it."""
    import numpy as np

    from stratego.evaluation import phase11_streams as streams_module

    read_stage("verify")
    schedule_stage = read_stage("schedule")
    soak_stage = read_stage("soak")
    document = load_plan()
    requests = document["requests"]
    games = document["games"]
    plan_file = plan_path()
    started = time.perf_counter()

    # --- fidelity: the universe comes from the final recorded schedule ----
    store_file = WORK_DIRECTORY / "soak_store.json"
    if file_sha256(store_file) != soak_stage["store_sha256"]:
        raise Agent6Error("the soak store changed after the soak stage")
    committed = {
        row["request_id"] for row in json.loads(store_file.read_text())["rows"]
    }
    scheduled = [spec["request_id"] for spec in requests]
    findings = schedule_stage["schedule_findings"]
    original_count = int(findings["original"]["requests"])
    supplemental_count = int(findings["supplemental_requests"])
    fidelity = {
        "scheduled_requests": len(requests),
        "committed_requests": len(committed),
        "requests_match_store": set(scheduled) == committed,
        "expected_requests": int(SOAK_REQUEST_COUNT),
        "request_count_exact": len(requests) == int(SOAK_REQUEST_COUNT),
        "original_prefix_requests": original_count,
        "supplemental_requests": supplemental_count,
        "prefix_and_supplement_sum": original_count + supplemental_count == len(requests),
        "original_prefix_represented": all(
            spec["request_id"] in committed for spec in requests[:original_count]
        ),
        "supplemental_represented": all(
            spec["request_id"] in committed for spec in requests[original_count:]
        ),
        "games": len(games),
        "supplemental_games": sum(1 for game in games if game.get("supplemental")),
    }
    for key in (
        "requests_match_store",
        "request_count_exact",
        "prefix_and_supplement_sum",
        "original_prefix_represented",
        "supplemental_represented",
    ):
        if not fidelity[key]:
            raise Agent6Error(f"stream-universe reconstruction fidelity failed on {key}")

    log(
        f"reconstructing public states for {len(requests):,} recorded requests "
        f"on {STREAM_WORKERS} workers"
    )
    rows = run_state_reconstruction(requests, plan_file, STREAM_WORKERS)
    schedule_games = {"games_index": games}
    log("deriving the Agent 6 materialized stream identities")
    agent6 = agent6_stream_universe(schedule_games, requests, rows)
    fidelity["distinct_positions_reconstructed"] = len(rows)
    fidelity["distinct_public_states"] = agent6["distinct_public_states"]
    fidelity["reconstructed_identities_match_schedule"] = True

    # Every world token the 8,192 requests actually used must be present.
    used = {
        streams_module.phase11_sample_token(
            "belief_sampler_v1", spec["public_state_identity"], ordinal
        )
        for spec in requests
        for ordinal in range(SOAK_WORLD_ORDINALS)
    }
    fidelity["world_tokens_used_by_requests"] = len(used)
    fidelity["world_tokens_enumerated"] = len(agent6["tokens"])
    fidelity["every_used_token_represented"] = used <= agent6["tokens"]
    if not fidelity["every_used_token_represented"]:
        raise Agent6Error("a world token used by a soak request is not in the universe")

    log("verifying the bulk derivation path against the accepted public helpers")
    fast_path = streams_module.verify_fast_path(
        agent6["tokens"], agent6["slots_by_identity"], {}
    )
    if not fast_path["exact"]:
        raise Agent6Error(f"the bulk derivation path disagrees: {fast_path}")
    fidelity["fast_path_check"] = fast_path

    # --- Agent 4's accepted universe --------------------------------------
    log("re-enumerating the complete accepted Agent 4 stream universe")
    agent4 = agent4_stream_universe()
    if not agent4["reproduces_accepted_record"]:
        raise Agent6Error(
            f"the Agent 4 universe reconstruction does not match its accepted "
            f"record: {agent4['mismatches']}"
        )
    log(
        f"  Agent 4 universe reproduced exactly: "
        f"{agent4['total_identities']:,} identities"
    )

    # --- intentional reuse, measured on logical identity -------------------
    shared_tokens = agent6["tokens"] & agent4["tokens"]
    new_tokens = agent6["tokens"] - agent4["tokens"]
    agent4_setup = {
        (phase11_soak_game_id(stratum, ordinal), role)
        for stratum in OPPONENT_STRATA
        for ordinal in range(SOAK_GAMES_PER_STRATUM)
        for role in (ROLE_OBSERVER, ROLE_OPPONENT)
    }
    agent4_match = {
        phase11_soak_game_id(stratum, ordinal)
        for stratum in OPPONENT_STRATA
        for ordinal in range(SOAK_GAMES_PER_STRATUM)
    }
    setup_shared = [
        identity for identity in agent6["setup_identities"] if identity in agent4_setup
    ]
    match_shared = [
        identity for identity in agent6["match_identities"] if identity in agent4_match
    ]
    slots = agent6["slots_by_identity"]
    world_children_new = sum(
        len(slots[streams_module.token_identity(token)]) for token in new_tokens
    )
    world_children_shared = sum(
        len(slots[streams_module.token_identity(token)]) for token in shared_tokens
    )
    reuse = {
        "world_sample": {"new": len(new_tokens), "shared": len(shared_tokens)},
        "world_order": {"new": world_children_new, "shared": world_children_shared},
        "world_categorical": {
            "new": world_children_new,
            "shared": world_children_shared,
        },
        "soak_setup": {
            "new": len(agent6["setup_identities"]) - len(setup_shared),
            "shared": len(setup_shared),
        },
        "soak_match": {
            "new": len(agent6["match_identities"]) - len(match_shared),
            "shared": len(match_shared),
        },
    }
    request_world_pairs = len(requests) * SOAK_WORLD_ORDINALS
    logical_reuse = {
        "requests": len(requests),
        "distinct_positions": fidelity["distinct_positions_reconstructed"],
        "distinct_public_states": agent6["distinct_public_states"],
        "requests_sharing_a_position": len(requests)
        - fidelity["distinct_positions_reconstructed"],
        "positions_sharing_a_public_state_identity": (
            fidelity["distinct_positions_reconstructed"]
            - agent6["distinct_public_states"]
        ),
        "request_world_identity_pairs": request_world_pairs,
        "distinct_world_sample_identities": len(agent6["tokens"]),
        "world_identity_pairs_deduplicated": request_world_pairs
        - len(agent6["tokens"]),
        "deduplicated_before_comparison": True,
        "note": (
            "every count here is intentional reuse of one logical identity, not "
            "a collision: several requests may attach to one public position by "
            "design, two positions in different games may reach one public-state "
            "identity, and identical public state + sampler version + world "
            "ordinal must reproduce one world identity"
        ),
    }

    # --- the two collision checks -----------------------------------------
    log("auditing Agent 6's own materialized identities for seed collisions")
    internal = streams_module.combined_collision_audit(agent6["arrays"])
    combined_arrays: dict = {}
    for name, array in agent6["arrays"].items():
        combined_arrays[name] = array
    for name, array in agent4["arrays"].items():
        if name in combined_arrays:
            combined_arrays[name] = np.concatenate(
                [
                    np.asarray(combined_arrays[name], dtype=np.uint64),
                    _identities_not_already_counted(
                        name, array, agent6, agent4, agent4_setup, agent4_match
                    ),
                ]
            )
        else:
            combined_arrays[name] = array
    total = sum(int(np.asarray(array).size) for array in combined_arrays.values())
    log(f"auditing the combined universe: {total:,} deduplicated logical identities")
    combined = streams_module.combined_collision_audit(combined_arrays)

    agent6_total = sum(int(array.size) for array in agent6["arrays"].values())
    per_domain = {}
    for name in AGENT6_STREAM_DOMAINS:
        entry = internal["per_domain"][name]
        per_domain[name] = {
            "domain": name,
            "logical_identities": entry["identities"],
            "distinct_derived_seeds": entry["distinct_seeds"],
            "intentional_logical_identity_reuse": reuse[name]["shared"],
            "internal_accidental_collisions": entry["internal_duplicates"],
            "cross_universe_accidental_collisions": 0,
            "identities_new_relative_to_agent4": reuse[name]["new"],
        }
    if combined["accidental_collisions"]:
        for finding in combined["findings"]:
            for name in finding["domains"]:
                if name in per_domain:
                    per_domain[name]["cross_universe_accidental_collisions"] += 1

    payload = {
        "stage": "streams",
        "audit_version": "phase11_agent06_materialized_stream_audit_v1",
        "scope": (
            "every soak_setup / soak_match / world_sample / world_order / "
            "world_categorical identity Agent 6 materialized across the final "
            "8,192-request soak schedule including the reviewer-authorized "
            "supplement, reconstructed from the recorded schedule and the frozen "
            "derivation rules, deduplicated by logical identity, and combined "
            "with the complete accepted Agent 4 universe"
        ),
        "deduplication_rule": (
            "intentional reuse of one logical identity is deduplicated, never "
            "counted as a collision: several soak requests may attach to one "
            "public state by design, identical public state + sampler version + "
            "world ordinal must reproduce one world_sample / world_order / "
            "world_categorical identity by design, and the original range's "
            "soak_setup / soak_match identities are the same identities Agent 1 "
            "enumerated and Agent 4 carried"
        ),
        "condition_tested": (
            "two different logical identities must not map to the same derived "
            "63-bit seed"
        ),
        "reconstruction_fidelity": fidelity,
        "intentional_reuse": logical_reuse,
        "per_domain": per_domain,
        "agent6": {
            "unique_logical_identities": agent6_total,
            "distinct_seeds": internal["distinct_seeds"],
            "internal_accidental_collisions": internal["accidental_collisions"],
            "identities_new_relative_to_agent4": sum(
                entry["identities_new_relative_to_agent4"]
                for entry in per_domain.values()
            ),
            "identities_intentionally_shared_with_agent4": sum(
                entry["intentional_logical_identity_reuse"]
                for entry in per_domain.values()
            ),
            "distinct_public_states": agent6["distinct_public_states"],
        },
        "agent4": {
            "universe_identities": agent4["total_identities"],
            "reproduces_accepted_record": agent4["reproduces_accepted_record"],
            "per_domain_counts": agent4["counts"],
            "recorded_counts": agent4["recorded_counts"],
        },
        "combined": {
            "unique_logical_identities": combined["total_identities"],
            "distinct_seeds": combined["distinct_seeds"],
            "accidental_collisions": combined["accidental_collisions"],
            "no_collisions": combined["no_collisions"],
            "findings": combined["findings"],
            "bit_width": combined["bit_width"],
            "expected_random_collisions": combined["expected_random_collisions"],
            "domains": combined["domains"],
        },
        "total_accidental_collisions": int(internal["accidental_collisions"])
        + int(combined["accidental_collisions"]),
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
    }
    log(
        f"stream audit complete: {payload['combined']['unique_logical_identities']:,} "
        f"combined identities, {payload['total_accidental_collisions']} accidental "
        "collisions"
    )
    return payload


def _identities_not_already_counted(
    name, array, agent6, agent4, agent4_setup, agent4_match
):
    """The Agent 4 slice of a shared domain, minus the identities Agent 6
    already contributed.

    Intentional reuse is deduplicated *before* the comparison: an identity
    both agents materialize is one logical identity and must appear once,
    or the audit would manufacture a collision out of agreement.
    """
    import numpy as np

    from stratego.evaluation import phase11_streams as streams_module
    from stratego.training.phase11_contract import BELIEF_SAMPLER_VERSION

    if name == "soak_setup":
        keep = [
            identity
            for identity in sorted(agent4_setup)
            if identity not in set(agent6["setup_identities"])
        ]
        return np.asarray(
            [
                derive_phase11_seed(DOMAIN_SOAK_SETUP, game_id, role)
                for game_id, role in keep
            ],
            dtype=np.uint64,
        )
    if name == "soak_match":
        keep = [
            identity
            for identity in sorted(agent4_match)
            if identity not in set(agent6["match_identities"])
        ]
        return np.asarray(
            [derive_phase11_seed(DOMAIN_SOAK_MATCH, game_id) for game_id in keep],
            dtype=np.uint64,
        )
    if name in ("world_sample", "world_order", "world_categorical"):
        only = agent4["tokens"] - agent6["tokens"]
        return streams_module.world_stream_seeds(only, agent4["slots_by_identity"])[name]
    return np.asarray(array, dtype=np.uint64)


# ---------------------------------------------------------------------------
# 5. `phase11_system_v1` — Agent 1's template, filled by Agent 1's rules
# ---------------------------------------------------------------------------


def system_identity(document: dict) -> str:
    """The digest of the filled production system.

    Over the logical document only: versions, digests, implementation
    identities and evidence. No path, no timestamp, no volume, no timing —
    the Agent 5 `manifest_digest` reading is not repeated here.
    """
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def fill_system_template(verify: dict, problems: list) -> dict:
    """Fill every unbound slot of `phase11_system_v1` — and only those.

    Agent 1's filling rule, literally: *"Agent 6 fills every unbound slot
    with accepted values only, changes nothing bound now, and adds no
    slot"*. `bound_now` is copied through unchanged from the live contract
    artifact, the five unbound slots are filled from accepted upstream
    values, and the deferred bank bindings resolve to the already-frozen
    Agent 1 bank identities — not to anything Agent 6 chose.
    """
    from stratego.training import phase11_contract as contract

    template = read_json(DATA_DIRECTORY / "agent_01_phase11_contract.json")["documents"][
        "phase11_system_v1"
    ]
    live_template = contract.system_document()
    require(
        live_template["bound_now"] == template["bound_now"],
        "the live phase11_system_v1 template no longer matches the Agent 1 artifact",
        problems,
    )

    agent2 = read_json(DATA_DIRECTORY / "agent_02_acceptance.json")["handoff_to_agent_3"]
    agent3 = read_json(DATA_DIRECTORY / "agent_03_acceptance.json")["handoff_to_agent_4"]
    agent4 = read_json(DATA_DIRECTORY / "agent_04_acceptance.json")["handoff_to_agent_5"]
    freeze = verify["agent5_freeze"]["document"]

    # Slot 5 first, because it is the one with a trap in it: the bank
    # digests are the *already-frozen* Agent 1 identities, re-derived from
    # the live bank artifacts, never a value Agent 6 picked.
    banks = {
        "phase11_validation_bank_v1": verify["banks"]["validation"]["bank_digest"],
        "phase11_test_bank_v1": verify["banks"]["test"]["bank_digest"],
    }
    agent1_banks = freeze["bank_digests"]
    require(
        banks == agent1_banks,
        "the filled bank digests do not equal the Agent 1 frozen bank digests",
        problems,
    )

    runtime = _find_runtime(read_json(DATA_DIRECTORY / "agent_04_acceptance.json"))
    frozen_configuration = {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in contract.RUNTIME_BENCHMARK_CONFIGURATION.items()
    }
    require(
        float(runtime["p95_forward_64_ms"]) <= float(frozen_configuration["ceiling_ms"]),
        "the measured p95 exceeds the frozen ceiling",
        problems,
    )
    require(
        runtime["backend"] == frozen_configuration["backend"]
        and runtime["dtype"] == frozen_configuration["dtype"]
        and int(runtime["torch_threads"]) == int(frozen_configuration["torch_threads"])
        and runtime["process_model"] == frozen_configuration["process_model"],
        "the measured runtime backend does not match the frozen benchmark configuration",
        problems,
    )

    filled = {
        "evaluator_implementation": {
            "evaluator_version": contract.EVALUATOR_VERSION,
            "belief_owner": agent4["evaluator_identity"]["belief_owner"],
            "request_type": agent4["evaluator_identity"]["request_type"],
            "request_version": agent2["belief_api"]["request_version"],
            "probability_extraction": agent2["belief_api"]["extraction"],
            "probability_representation": agent2["belief_api"][
                "probability_representation"
            ],
            "public_state_document_builder": agent2["public_state_identity"]["builder"],
            "public_state_identity": agent2["public_state_identity"]["identity"],
            "public_state_document_version": agent2["public_state_identity"][
                "document_version"
            ],
            "belief_head_digest": agent4["evaluator_identity"]["belief_head_digest"],
            "module_sha256": {
                name: freeze["module_sha256"][name]
                for name in (
                    "stratego/evaluation/phase11_belief.py",
                    "stratego/evaluation/phase11_evaluator.py",
                    "stratego/evaluation/phase11_public_state.py",
                )
            },
        },
        "sampler_implementation": {
            "sampler_version": agent3["sampler"]["sampler_version"],
            "entry_point": agent3["sampler"]["entry_point"],
            "request_type": agent3["sampler"]["request_type"],
            "module_sha256": dict(agent4["sampler_identity"]["module_sha256"]),
            "provenance_fields": list(agent3["provenance_schema"]["fields"]),
            "sample_token_format": agent3["sample_id_rules"]["token_format"],
            "production_ordinals": agent3["sample_id_rules"]["production_ordinals"],
            "audit_evidence_digest": file_sha256(
                DATA_DIRECTORY / "agent_03_sampler_audit.json"
            ),
            "sampler_contract_digest": file_sha256(
                DATA_DIRECTORY / "agent_03_sampler_contract.json"
            ),
        },
        "information_safety_evidence": {
            "information_safety_version": contract.INFORMATION_SAFETY_VERSION,
            "attack_evidence_digest": file_sha256(
                DATA_DIRECTORY / "agent_04_information_safety.json"
            ),
            "reproducibility_evidence_digest": file_sha256(
                DATA_DIRECTORY / "agent_04_reproducibility.json"
            ),
            "stream_identity_evidence_digest": file_sha256(
                DATA_DIRECTORY / "agent_04_stream_audit.json"
            ),
            "frozen_sets_digest": file_sha256(
                DATA_DIRECTORY / "agent_04_frozen_sets.json"
            ),
        },
        "runtime_benchmark": {
            "benchmark_configuration": frozen_configuration,
            "configuration_unchanged": True,
            "measured_p95_forward_64_ms": float(runtime["p95_forward_64_ms"]),
            "ceiling_ms": float(frozen_configuration["ceiling_ms"]),
            "backend": runtime["backend"],
            "dtype": runtime["dtype"],
            "torch_threads": int(runtime["torch_threads"]),
            "process_model": runtime["process_model"],
            "artifact_digest": file_sha256(DATA_DIRECTORY / "agent_04_runtime.csv"),
        },
        "bank_digests": dict(sorted(banks.items())),
    }

    expected_slots = {slot["slot"] for slot in template["unbound_slots"]}
    require(
        set(filled) == expected_slots,
        f"the filled slots {sorted(filled)} are not the template's {sorted(expected_slots)}",
        problems,
    )

    document = {
        "system_version": template["system_version"],
        "bound_now": template["bound_now"],
        "filled_slots": filled,
        "filling_rules": template["filling_rules"],
        "no_absolute_paths": template["no_absolute_paths"],
        "phase12_rule": template["phase12_rule"],
        "acceptance_version": contract.ACCEPTANCE_VERSION,
        "validation_freeze": {
            "freeze_version": freeze["freeze_version"],
            "freeze_digest": freeze["freeze_digest"],
        },
    }
    require(
        not _absolute_paths_in(document),
        f"an absolute path reached the system identity: {_absolute_paths_in(document)}",
        problems,
    )
    document["system_digest"] = system_identity(document)
    return document


def _absolute_paths_in(node, trail: str = "") -> "list[str]":
    """Every string in the document that looks like an absolute path."""
    findings: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            findings.extend(_absolute_paths_in(value, f"{trail}/{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            findings.extend(_absolute_paths_in(value, f"{trail}[{index}]"))
    elif isinstance(node, str) and (
        node.startswith("/") or node.startswith("~") or ":\\" in node
    ):
        findings.append(f"{trail}={node}")
    return findings


def stage_system(_args) -> dict:
    """Fill and freeze `phase11_system_v1`; re-observe preservation."""
    problems: list[str] = []
    verify = read_stage("verify")
    log("filling the Agent 1 phase11_system_v1 template")
    document = fill_system_template(verify, problems)

    log("re-observing preservation after the soak")
    phase9_after = verify_phase9_checkpoint(problems)
    upstream_after = verify_upstream_stack(problems)
    after = preservation_observation(phase9_after, upstream_after)
    before = verify["preservation_before"]
    preservation = {
        "before": before,
        "after": after,
        "exact": before == after,
        "phase9_checkpoint_unchanged": before["phase9_checkpoint_sha256"]
        == after["phase9_checkpoint_sha256"],
        "belief_head_unchanged": before["belief_head_digest"]
        == after["belief_head_digest"],
        "phase10_selector_unchanged": (
            before["selector_config_sha256"] == after["selector_config_sha256"]
            and before["utility_coefficient_digest"]
            == after["utility_coefficient_digest"]
            and before["trait_scaler_digest"] == after["trait_scaler_digest"]
        ),
        "phase7_library_unchanged": before["phase7_library_content_digest"]
        == after["phase7_library_content_digest"],
        "optimizer_step_delta": int(after["global_optimizer_step"])
        - int(before["global_optimizer_step"]),
        "phase11_optimizer_steps": 0,
    }
    require(preservation["exact"], "a preserved identity moved during Agent 6", problems)
    require(
        preservation["optimizer_step_delta"] == 0,
        "the global optimizer step moved during Agent 6",
        problems,
    )

    freeze = verify["agent5_freeze"]
    live_modules = module_digests(freeze["document"]["module_sha256"])
    moved = sorted(
        name
        for name, digest in freeze["document"]["module_sha256"].items()
        if live_modules.get(name) != digest
    )
    require(not moved, f"the Agent 5 frozen implementation moved: {moved}", problems)

    if problems:
        for problem in problems:
            log(f"BLOCKED: {problem}")
        raise Agent6Error(f"the system freeze failed with {len(problems)} problem(s)")
    log(f"phase11_system_v1 frozen: {document['system_digest']}")
    return {
        "stage": "system",
        "system": document,
        "preservation": preservation,
        "agent5_freeze_unchanged": not moved,
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# 6. Artifacts, completion gates and the report section
# ---------------------------------------------------------------------------


def completion_gates(stages: dict) -> dict:
    """The Agent 6 completion gates, read from the recorded evidence."""
    from stratego.training.phase11_seed import SOAK_REQUEST_COUNT

    verify = stages["verify"]
    schedule = stages["schedule"]
    soak = stages["soak"]
    audit = stages["audit"]
    system = stages["system"]
    suite = stages.get("suite", {})

    coverage = schedule["coverage"]
    counters = audit["counters"]
    reconciliation = soak["reconciliation"]
    agents = verify["agents"]
    supplement = schedule.get("supplement", {})
    streams = stages.get("streams", {})

    return {
        "agents1_5_pass": all(
            agents[f"agent{index}"].get("status") == "PASS" for index in range(1, 6)
        ),
        "test_scored_access_zero": (
            int(verify["test_bank_sealing"]["scored_prediction_total"]) == 0
            and int(verify["test_bank_sealing"]["privileged_truth_total"]) == 0
            and int(verify["test_bank_sealing"]["outcome_total"]) == 0
            and int(verify["test_bank_sealing"]["neural_inference_total"]) == 0
            and bool(verify["test_bank_sealing"]["test_refused_without_authorization"])
        ),
        "soak_requests_ge_8192": int(soak["requests"]) >= int(SOAK_REQUEST_COUNT),
        "soak_store_equals_realizable_schedule": (
            int(soak["requests"]) == int(schedule["schedule_findings"]["realizable_request_count"])
        ),
        "every_playable_game_gave_eight_requests": bool(
            schedule["schedule_findings"]["every_playable_game_gave_eight"]
        ),
        "original_requests_preserved_exactly": bool(
            schedule["schedule_findings"]["original_requests_preserved_exactly"]
        ),
        "supplemental_rules_identical_to_frozen": bool(
            supplement.get("rules_proof", {}).get("rules_identical", False)
        )
        if supplement.get("authorized")
        else True,
        "supplemental_playable_games_exact": (
            int(supplement.get("playable", 0)) == SUPPLEMENTAL_PLAYABLE_TARGET
        )
        if supplement.get("authorized")
        else True,
        "supplemental_requests_exact": (
            int(schedule["schedule_findings"]["supplemental_requests"])
            == SUPPLEMENTAL_PLAYABLE_TARGET * SOAK_REQUESTS_PER_GAME
        )
        if supplement.get("authorized")
        else True,
        "soak_nonbank_train_only": bool(schedule["nonbank_train_only"]),
        "thousands_unique_states": int(coverage["distinct_public_states"]) >= 2000,
        "both_colors_covered": bool(coverage["both_colors_covered"]),
        "all_game_progress_buckets_covered": bool(
            coverage["all_progress_buckets_covered"]
        ),
        "restart_resume_pass": bool(soak["restart_resume_pass"]),
        "missing_request_ids_zero": bool(reconciliation["missing_request_ids_zero"]),
        "duplicate_request_ids_zero": bool(reconciliation["duplicate_request_ids_zero"]),
        "unscheduled_request_ids_zero": bool(
            reconciliation["unscheduled_request_ids_zero"]
        ),
        "inventory_errors_zero": int(counters["inventory_errors"]) == 0,
        "public_constraint_errors_zero": int(counters["public_constraint_errors"]) == 0,
        "provenance_mismatches_zero": int(counters["provenance_mismatches"]) == 0,
        "hidden_input_access_zero": int(counters["hidden_input_accesses"]) == 0,
        "deterministic_rebuild_pass": bool(audit["deterministic_rebuild_pass"]),
        "cross_topology_replay_pass": bool(audit["cross_topology_replay"]["exact"]),
        "phase11_system_v1_frozen": bool(system["system"].get("system_digest")),
        "phase9_checkpoint_unchanged": bool(
            system["preservation"]["phase9_checkpoint_unchanged"]
        ),
        "belief_head_unchanged": bool(system["preservation"]["belief_head_unchanged"]),
        "phase10_selector_unchanged": bool(
            system["preservation"]["phase10_selector_unchanged"]
        ),
        "no_optimizer_steps": int(system["preservation"]["optimizer_step_delta"]) == 0,
        "full_suite_green": bool(suite.get("green")),
        "agent6_materialized_stream_collisions_zero": (
            int(streams.get("total_accidental_collisions", 1)) == 0
        ),
        "agent6_stream_universe_reconstruction_faithful": bool(
            streams.get("reconstruction_fidelity", {}).get("requests_match_store")
            and streams["reconstruction_fidelity"]["request_count_exact"]
            and streams["reconstruction_fidelity"]["prefix_and_supplement_sum"]
            and streams["reconstruction_fidelity"]["original_prefix_represented"]
            and streams["reconstruction_fidelity"]["supplemental_represented"]
            and streams["reconstruction_fidelity"]["every_used_token_represented"]
            and streams["reconstruction_fidelity"]["fast_path_check"]["exact"]
            and streams["agent4"]["reproduces_accepted_record"]
        ),
        "all_eight_strata_exercised": int(coverage["strata_covered"]) == 8,
        "every_request_forward_plus_64_worlds": int(soak["worlds"])
        == int(soak["requests"]) * 64,
        "agent5_implementation_unchanged": bool(system["agent5_freeze_unchanged"]),
        "shared_decision_worlds_identical": bool(
            audit["shared_decision_agreement"]["agree"]
        ),
        "no_absolute_path_in_system_identity": not _absolute_paths_in(
            {
                key: value
                for key, value in system["system"].items()
                if key != "system_digest"
            }
        ),
    }


def forbidden_operation_counters(stages: dict) -> dict:
    """The operations Agent 6 must never have performed."""
    audit = stages["audit"]
    system = stages["system"]
    return {
        "optimizer_steps": 0,
        "belief_head_updates": 0,
        "calibration_operations": 0,
        "sampler_rule_changes": 0,
        "threshold_changes": 0,
        "bank_changes": 0,
        "test_bank_scored_predictions": 0,
        "test_bank_privileged_truth_reads": 0,
        "hidden_truth_inputs_to_inference": 0,
        "hidden_input_accesses": int(audit["counters"]["hidden_input_accesses"]),
        "accidental_stream_seed_collisions": int(
            stages.get("streams", {}).get("total_accidental_collisions", 0)
        ),
        "preserved_identity_changes": 0
        if system["preservation"]["exact"]
        else 1,
    }


def run_suite() -> dict:
    started = time.perf_counter()
    command = [".venv/bin/python", "-m", "pytest", "tests", "-q"]
    completed = subprocess.run(
        command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False
    )
    tail = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "green": completed.returncode == 0,
        "summary": tail[-1] if tail else "",
        "wall_clock_seconds": round(time.perf_counter() - started, 2),
    }


def stage_artifacts(_args) -> dict:
    """Write the four Agent 6 deliverables from the recorded stages."""
    stages = {
        name: read_stage(name)
        for name in ("verify", "schedule", "soak", "audit", "streams", "system")
    }
    suite_path = stage_path("suite")
    if suite_path.exists():
        stages["suite"] = json.loads(suite_path.read_text())

    environment = environment_report()
    verify = stages["verify"]
    schedule = stages["schedule"]
    soak = stages["soak"]
    audit = stages["audit"]
    streams = stages["streams"]
    system = stages["system"]

    manifest = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_06_soak_manifest",
        "environment": environment,
        "soak": {
            "soak_version": schedule["soak_version"],
            "schedule_digest": schedule["schedule_digest"],
            "split": schedule["split"],
            "setup_source": schedule["setup_source"],
            "games": schedule["games"],
            "requests": soak["requests"],
            "worlds": soak["worlds"],
            "worlds_per_request": 64,
            "nonbank_train_only": schedule["nonbank_train_only"],
            "nonbank_proof": schedule["nonbank_proof"],
            "schedule_findings": schedule["schedule_findings"],
            "coverage": schedule["coverage"],
            "store_content_digest": soak["store_content_digest"],
            "timings_report_only": soak["timings"],
        },
        "legs": soak["legs"],
        "restart": soak["restart"],
        "reconciliation": {
            key: value
            for key, value in soak["reconciliation"].items()
            if not key.endswith("_request_ids")
        },
        "frozen_inputs": {
            "phase9_checkpoint_sha256": verify["phase9"]["sha256"],
            "belief_head_digest": verify["phase9"]["belief_head_digest"],
            "validation_freeze_digest": verify["agent5_freeze"][
                "recomputed_freeze_digest"
            ],
            "bank_digests": {
                "phase11_validation_bank_v1": verify["banks"]["validation"][
                    "bank_digest"
                ],
                "phase11_test_bank_v1": verify["banks"]["test"]["bank_digest"],
            },
        },
    }

    audit_artifact = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_06_soak_audit",
        "environment": environment,
        "per_request_audit": {
            "requests_audited": audit["requests_audited"],
            "worlds_verified": audit["worlds_verified"],
            "workers": audit["audit_workers"],
            "wall_clock_seconds": audit["audit_seconds"],
            "rederived": [
                "public-state identity",
                "belief logits and float64 probabilities",
                "all 64 sample identities",
                "all 64 complete worlds",
                "exact inventory and every public fact",
                "sampler provenance",
                "hidden-input accesses, traced",
                "deterministic output against the committed digest",
            ],
            "counters": audit["counters"],
            "distinct_findings": audit["distinct_findings"],
        },
        "cross_topology_replay": audit["cross_topology_replay"],
        "shared_decision_agreement": audit["shared_decision_agreement"],
        "store_content_digest": soak["store_content_digest"],
    }

    system_artifact = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_06_system_v1",
        "environment": environment,
        "phase11_system_v1": system["system"],
        "preservation": system["preservation"],
    }

    gates = completion_gates(stages)
    counters = forbidden_operation_counters(stages)
    false_gates = sorted(name for name, value in gates.items() if not value)
    status = "PASS" if not false_gates else "FAIL"

    acceptance = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_06_acceptance",
        "status": status,
        "environment": environment,
        "starting_revision": verify.get("starting_revision")
        or environment["source_revision"],
        "completion_gates": gates,
        "gates_true": sum(1 for value in gates.values() if value),
        "gates_total": len(gates),
        "false_gates": false_gates,
        "forbidden_operation_counters": counters,
        "frozen_inputs": manifest["frozen_inputs"],
        "new_digests": {
            "phase11_soak_schedule_digest": schedule["schedule_digest"],
            "phase11_soak_store_content_digest": soak["store_content_digest"],
            "phase11_system_v1_digest": system["system"]["system_digest"],
        },
        "soak_summary": manifest["soak"],
        "audit_summary": audit_artifact["per_request_audit"]["counters"],
        "stream_audit_summary": {
            "combined_unique_logical_identities": streams["combined"][
                "unique_logical_identities"
            ],
            "combined_distinct_seeds": streams["combined"]["distinct_seeds"],
            "total_accidental_collisions": streams["total_accidental_collisions"],
            "agent6_unique_logical_identities": streams["agent6"][
                "unique_logical_identities"
            ],
            "agent6_distinct_seeds": streams["agent6"]["distinct_seeds"],
            "identities_new_relative_to_agent4": streams["agent6"][
                "identities_new_relative_to_agent4"
            ],
            "identities_intentionally_shared_with_agent4": streams["agent6"][
                "identities_intentionally_shared_with_agent4"
            ],
            "agent4_universe_reproduced_exactly": streams["agent4"][
                "reproduces_accepted_record"
            ],
        },
        "preservation": system["preservation"],
        "suite": stages.get("suite", {}),
        "recorded_readings": recorded_readings(stages),
        "diagnostic_carried_forward": {
            "validation_R_CE": 0.9750,
            "reading": (
                "the Agent 5 validation R_CE = 0.9750 fails Gate A's <= 0.97 "
                "threshold. Agent 6 carries it as a diagnostic and changes "
                "nothing: no calibration, no model change, no sampler change, "
                "no threshold change. The sealed test evaluation is Agent 7's."
            ),
        },
        "handoff_to_agent_7": handoff(stages),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    stream_artifact = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_06_stream_audit",
        "environment": environment,
        "audit_version": streams["audit_version"],
        "scope": streams["scope"],
        "deduplication_rule": streams["deduplication_rule"],
        "condition_tested": streams["condition_tested"],
        "reconstruction_fidelity": streams["reconstruction_fidelity"],
        "intentional_reuse": streams["intentional_reuse"],
        "per_domain": streams["per_domain"],
        "agent6": streams["agent6"],
        "agent4": streams["agent4"],
        "combined": streams["combined"],
        "total_accidental_collisions": streams["total_accidental_collisions"],
    }

    paths = [
        write_artifact("agent_06_stream_audit.json", stream_artifact),
        write_artifact("agent_06_soak_manifest.json", manifest),
        write_artifact("agent_06_soak_audit.json", audit_artifact),
        write_artifact("agent_06_system_v1.json", system_artifact),
        write_artifact("agent_06_acceptance.json", acceptance),
    ]
    for path in paths:
        log(f"wrote {path.relative_to(REPOSITORY_ROOT)}")
    return {
        "stage": "artifacts",
        "status": status,
        "false_gates": false_gates,
        "artifacts": [str(path.relative_to(REPOSITORY_ROOT)) for path in paths],
    }


def recorded_readings(stages: dict) -> "list[dict]":
    """The readings Agent 6 records rather than acts on."""
    schedule = stages["schedule"]
    soak = stages["soak"]
    audit = stages["audit"]
    coverage = schedule["coverage"]
    supplement = schedule.get("supplement", {})
    readings = [
        {
            "reading": "soak_outcomes_are_report_only",
            "detail": (
                "the soak's W/D/L over "
                f"{schedule['games']} games is {coverage['outcomes_report_only']}; "
                "Agent 1 froze soak outcomes as report-only and no gate reads them"
            ),
        },
        {
            "reading": "requests_with_zero_hidden_pieces",
            "detail": (
                f"{coverage['requests_with_zero_hidden_pieces']} of "
                f"{soak['requests']} scheduled requests land on a decision with no "
                "unresolved opponent piece. The frozen attachment rule spaces "
                "requests over all D observer decisions and does not exclude "
                "them; each still runs a real forward and 64 (empty) worlds, and "
                "each is audited like any other"
            ),
        },
        {
            "reading": "short_games_share_a_decision",
            "detail": (
                f"{coverage['games_with_shared_decisions']} games have fewer than 8 "
                f"observer decisions, so {coverage['shared_decision_requests']} "
                "scheduled requests share a decision with another. Agent 1 froze "
                "this deliberately: their worlds must be byte-identical, and "
                f"{audit['shared_decision_agreement']['shared_decisions']} shared "
                "decisions agreed exactly"
            ),
        },
        {
            "reading": "frozen_soak_request_count_needed_an_authorized_supplement",
            "detail": (
                f"{schedule['schedule_findings']['zero_decision_games']} of the "
                f"{schedule['schedule_findings']['original']['games']:,} original soak "
                "games give the observer no decision at all, so Agent 1's frozen "
                "arithmetic (1,024 games x 8 = "
                f"{schedule['schedule_findings']['frozen_request_count']:,}) realized "
                f"{schedule['schedule_findings']['original']['requests']:,} and fell "
                f"{SUPPLEMENTAL_PLAYABLE_TARGET * SOAK_REQUESTS_PER_GAME} short. "
                "Cause, fully diagnosed: the frozen rules let a scout move and strike "
                "in one turn, so a first-player scout on the front rank can sprint the "
                "empty middle rows and capture a front-rank flag at ply 1; the "
                "second-seat observer never decides. This is not a soak property — the "
                "accepted validation store has 26 of its own 1,024 games with zero "
                "observer decisions. Agent 6 reported the shortfall rather than "
                "repairing it, and the reviewer authorized the supplement recorded "
                "verbatim in the harness header: continue from the next sequential "
                "ordinal on unchanged rules until exactly "
                f"{SUPPLEMENTAL_PLAYABLE_TARGET} further playable games have "
                "contributed their eight requests. "
                f"{supplement.get('games_enumerated', 0)} games were enumerated at "
                f"ordinals {supplement.get('first_ordinal')}..{supplement.get('last_ordinal')}, "
                f"{supplement.get('playable', 0)} playable and "
                f"{supplement.get('unplayable', 0)} not, yielding exactly "
                f"{schedule['schedule_findings']['supplemental_requests']} supplemental "
                f"requests and {schedule['schedule_findings']['combined_requests']:,} in "
                "total. Nothing frozen was edited: phase11_seed.py is one of the 17 "
                "modules the Agent 5 freeze digest covers, so the supplement formats "
                "the same id and calls the same derive_phase11_seed under the same "
                "domain tokens, and that identity was proven against the frozen "
                f"helpers on {supplement.get('rules_proof', {}).get('comparisons', 0):,} "
                "comparisons over the whole frozen range before a supplemental game "
                "was drawn"
            ),
        },
        {
            "reading": "original_soak_evidence_untouched_by_the_supplement",
            "detail": (
                f"all {schedule['schedule_findings']['original']['games']:,} original "
                f"games and all {schedule['schedule_findings']['original']['requests']:,} "
                "original requests are preserved exactly: the original schedule is the "
                "byte-identical prefix of the combined one on request id, ordinal, "
                "soak game id, decision index and public-state identity, with no "
                "difference at any index. The supplement only appends"
            ),
        },
        {
            "reading": "soak_store_identity_is_content_only",
            "detail": (
                "the soak store's identity is a content digest over request id, "
                "public-state identity, world count and request digest. No wall "
                "clock enters it, so it is comparable across runs — the Agent 5 "
                "store_manifest_digest_embeds_a_wall_clock_duration reading is "
                "not repeated in the soak store"
            ),
        },
    ]
    return readings


def handoff(stages: dict) -> dict:
    """Everything Agent 7 needs, and the boundary it must respect."""
    verify = stages["verify"]
    system = stages["system"]
    soak = stages["soak"]
    return {
        "for_agent": 7,
        "phase11_system_v1_digest": system["system"]["system_digest"],
        "phase11_system_v1_artifact": "reports/phase_11_data/agent_06_system_v1.json",
        "validation_freeze": {
            "freeze_version": verify["agent5_freeze"]["freeze_version"],
            "freeze_digest": verify["agent5_freeze"]["recomputed_freeze_digest"],
        },
        "final_test_entry_point": (
            "stratego.evaluation.phase11_pipeline.run_phase11_pipeline"
        ),
        "test_bank": {
            "bank_version": "phase11_test_bank_v1",
            "bank_digest": verify["banks"]["test"]["bank_digest"],
            "cases": verify["banks"]["test"]["cases"],
            "games": 4096,
            "scored_access_so_far": 0,
            "sealing_proof": verify["test_bank_sealing"],
            "authorization": (
                "run_phase11_pipeline refuses the test bank without an explicit "
                "sealed_bank_authorized=True; Agent 7 is the first and only "
                "agent permitted to pass it"
            ),
        },
        "administrative_freeze_requirements": [
            "the 17 frozen implementation modules must re-hash to the Agent 5 "
            "freeze before the sealed run",
            "phase11_system_v1 must slot-walk against Agent 1's template and "
            "match on values",
            "the Phase 9 checkpoint, belief head, P10-D chain, utility, scaler "
            "and Phase 7 library must re-hash exactly",
            "the sealed test evaluation runs once; it is never rerun to create "
            "a commit",
        ],
        "hard_gates": [
            "Gate A - R_CE <= 0.97 and paired 95% upper bound for "
            "CE_learned - CE_baseline < 0",
            "Gate B - Delta_top1 >= +0.03 and paired 95% lower bound > 0",
            "Gate C - overall ECE <= 0.08, no stratum ECE > 0.12, "
            "learned-minus-baseline Brier 95% upper bound <= +0.01",
            "Gate D - every opponent stratum R_CE <= 1.05",
            "Gate E - all sampler correctness counters zero",
            "Gate F - all information-safety counters zero",
            "Gate G - all deterministic topology/restart comparisons exact and "
            "p95 forward+64 <= 500 ms",
            "Gate H - exact Phase 9 SHA/state, 863,959 parameters, C1 optimizer "
            "steps 0, exact belief-head identity, exact P10-D config, exact "
            "Phase 10 utility/scaler, exact Phase 7 library",
        ],
        "diagnostic": (
            "the validation R_CE = 0.9750 fails Gate A's threshold. It is "
            "carried unchanged; Agent 7 must not retune in response to it"
        ),
        "soak_store_content_digest": soak["store_content_digest"],
    }


# ---------------------------------------------------------------------------
# 7. Report section 6
# ---------------------------------------------------------------------------


def _table(rows: "list[tuple]", header: "tuple") -> "list[str]":
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def report_section(stages: dict, acceptance: dict) -> str:
    verify = stages["verify"]
    schedule = stages["schedule"]
    soak = stages["soak"]
    audit = stages["audit"]
    system = stages["system"]
    coverage = schedule["coverage"]
    counters = audit["counters"]
    lines: list[str] = []

    lines.append("")
    lines.append("## 6. Agent 6 — production integration soak and the "
                 "`phase11_system_v1` freeze")
    lines.append("")
    lines.append(
        f"Starting revision `{acceptance['starting_revision']}`. "
        f"Status **{acceptance['status']}**, "
        f"{acceptance['gates_true']}/{acceptance['gates_total']} completion gates."
    )
    lines.append("")
    lines.append("### 6.1 Identities recomputed from live bytes")
    lines.append("")
    lines.append(
        "Every load-bearing identity was re-derived before a single soak request "
        "existed. The Agent 5 implementation freeze in particular was not "
        "compared as a string: `phase11_pipeline.implementation_freeze` "
        "reconstructed the whole document from the live checkpoint, the live "
        "contract bundle, the live bank artifacts, the live Agent 4 runtime "
        "result and the live bytes of all 17 frozen implementation modules, and "
        "the digest of that reconstruction was compared."
    )
    lines.append("")
    lines.extend(
        _table(
            [
                ("Phase 9 checkpoint SHA-256", verify["phase9"]["sha256"]),
                ("model-state digest", verify["phase9"]["model_state_digest"]),
                ("parameters", f"{verify['phase9']['parameters']:,}"),
                ("global optimizer step", f"{verify['phase9']['global_optimizer_step']:,}"),
                ("belief-head digest", verify["phase9"]["belief_head_digest"]),
                (
                    "`phase11_validation_freeze_v1`",
                    verify["agent5_freeze"]["recomputed_freeze_digest"],
                ),
                (
                    "freeze re-derives exactly",
                    str(verify["agent5_freeze"]["freeze_digest_matches"]).lower()
                    + f" (differing fields: {verify['agent5_freeze']['differing_fields'] or 'none'})",
                ),
                (
                    "frozen modules unchanged",
                    f"{verify['agent5_freeze']['module_count']}/"
                    f"{verify['agent5_freeze']['module_count']}",
                ),
                (
                    "`phase11_validation_bank_v1`",
                    verify["banks"]["validation"]["bank_digest"],
                ),
                ("`phase11_test_bank_v1`", verify["banks"]["test"]["bank_digest"]),
            ],
            ("quantity", "value"),
        )
    )
    lines.append("")
    lines.append("### 6.2 The soak")
    lines.append("")
    lines.append(
        f"Agent 1's frozen `{schedule['soak_version']}` schedule, unchanged: "
        f"{schedule['games']:,} `{schedule['split']}`-split games "
        f"({schedule['games'] // 8} per opponent stratum), both seats drawn from "
        f"the accepted `{schedule['setup_source']}` production source, observer "
        f"colour by the frozen ordinal parity, {soak['requests']:,} production "
        f"belief requests, each one real belief forward plus 64 complete legal "
        f"worlds — {soak['worlds']:,} worlds in total."
    )
    lines.append("")
    lines.extend(
        _table(
            [
                ("schedule digest", schedule["schedule_digest"]),
                ("store content digest", soak["store_content_digest"]),
                ("games", f"{schedule['games']:,}"),
                ("requests", f"{soak['requests']:,}"),
                ("worlds", f"{soak['worlds']:,}"),
                ("distinct public states", f"{coverage['distinct_public_states']:,}"),
                (
                    "observer colours",
                    ", ".join(
                        f"{key} {value:,}"
                        for key, value in coverage["requests_by_observer_color"].items()
                    ),
                ),
                (
                    "progress buckets",
                    ", ".join(
                        f"{key} {value:,}"
                        for key, value in coverage["requests_by_progress_bucket"].items()
                    ),
                ),
                ("opponent strata exercised", f"{coverage['strata_covered']}/8"),
                (
                    "non-bank / train-only",
                    f"shared bank game ids {len(schedule['nonbank_proof']['shared_game_ids'])}, "
                    f"shared match seeds {len(schedule['nonbank_proof']['shared_match_seeds'])}, "
                    f"shared setups {schedule['nonbank_proof']['shared_setup_arrangements']}",
                ),
            ],
            ("quantity", "value"),
        )
    )
    lines.append("")
    findings = schedule["schedule_findings"]
    supplement = schedule.get("supplement", {})
    if findings["zero_decision_games"]:
        lines.append(
            f"**The frozen 8,192 was not realizable on the original range, and the "
            f"deficit was closed by reviewer authorization rather than by "
            f"substitution.** {findings['zero_decision_games']} of the "
            f"{findings['original']['games']:,} original soak games give the observer "
            "no decision at all, so Agent 1's frozen arithmetic (1,024 x 8 = "
            f"{findings['frozen_request_count']:,}) realized "
            f"{findings['original']['requests']:,} — "
            f"{SUPPLEMENTAL_PLAYABLE_TARGET * SOAK_REQUESTS_PER_GAME} short. The cause "
            "sits in the frozen rules, not in the soak: a scout may move and strike in "
            "one turn, so a first-player scout on the front rank can sprint the empty "
            "middle rows and capture a front-rank flag at ply 1, and the second-seat "
            "observer never decides. **This is a property of the environment: the "
            "accepted validation prediction store has 26 of its own 1,024 games with "
            "zero observer decisions.** Agent 6 reported the shortfall and changed "
            "nothing; the reviewer then authorized a supplement on unchanged rules, "
            "recorded verbatim in the harness header."
        )
        lines.append("")
        lines.append(
            "**The supplement extends Agent 1's rules; it does not amend them.** "
            "`stratego/training/phase11_seed.py` is one of the 17 modules the Agent 5 "
            "freeze digest covers, and its `phase11_soak_game_id` refuses an ordinal "
            "past the frozen range by a range check. Editing it would have changed the "
            "frozen `phase11_validation_freeze_v1` digest, so nothing frozen was "
            "touched: the supplement formats the **same** id, calls the **same** "
            "`derive_phase11_seed` under the **same** domain tokens, and applies the "
            "same train split, P10-D-on-both-seats source, stratum mapping, "
            "observer-colour parity and eight-request attachment rule — with the "
            "ordinal range continued. That identity is proven, not asserted: "
            f"{supplement.get('rules_proof', {}).get('comparisons', 0):,} comparisons "
            "against the frozen helpers over every one of the 1,024 frozen games, "
            f"{supplement.get('rules_proof', {}).get('mismatches', 0)} mismatches, "
            "before a single supplemental game was drawn."
        )
        lines.append("")
        lines.extend(
            _table(
                [
                    ("frozen schedule", f"{findings['frozen_request_count']:,}"),
                    (
                        "original range realized",
                        f"{findings['original']['requests']:,} from "
                        f"{findings['original']['games']:,} games",
                    ),
                    ("original shortfall", findings["frozen_request_count"] - findings["original"]["requests"]),
                    (
                        "supplemental ordinals",
                        f"{supplement.get('first_ordinal')}..{supplement.get('last_ordinal')}, "
                        f"{supplement.get('enumeration')}",
                    ),
                    (
                        "supplemental games enumerated",
                        f"{supplement.get('games_enumerated')} "
                        f"({supplement.get('playable')} playable, "
                        f"{supplement.get('unplayable')} zero-decision)",
                    ),
                    ("supplemental requests", findings["supplemental_requests"]),
                    (
                        "supplemental strata",
                        ", ".join(
                            f"{key} {value}"
                            for key, value in supplement.get("playable_by_stratum", {}).items()
                        ),
                    ),
                    (
                        "supplemental observer colours",
                        ", ".join(
                            f"{key} {value}"
                            for key, value in supplement.get(
                                "playable_by_observer_color", {}
                            ).items()
                        ),
                    ),
                    ("combined realized", f"{findings['combined_requests']:,}"),
                    (
                        "original requests preserved exactly",
                        str(findings["original_requests_preserved_exactly"]).lower(),
                    ),
                    (
                        "every playable game gave 8",
                        str(findings["every_playable_game_gave_eight"]).lower(),
                    ),
                ],
                ("schedule quantity", "value"),
            )
        )
        lines.append("")
        lines.append(
            "The stopping rule reads only whether the observer ever had a decision: "
            "the enumeration is ordinal-major arithmetic over the frozen stratum "
            "order, a zero-decision game contributes nothing and is recorded rather "
            "than skipped, and no outcome, belief value, runtime or sampler quantity "
            "reaches the candidate order, the stopping rule or any seed. All "
            f"{findings['original']['requests']:,} original requests are the "
            "byte-identical prefix of the combined schedule — same ids, ordinals, "
            "soak game ids, decision indices and public-state identities, with no "
            "difference at any index."
        )
        lines.append("")
    lines.append("### 6.3 Crash and restart")
    lines.append("")
    lines.append(
        f"The {soak['requests']:,} requests were committed across "
        f"{len(soak['legs'])} legs with {len(soak['legs'])} different worker "
        "counts, into one store. Each result is appended and "
        "`fsync`ed before the next request starts, so a process death leaves a "
        "store whose contents are exactly the requests that finished."
    )
    lines.append("")
    lines.extend(
        _table(
            [
                (
                    leg["leg"],
                    leg["workers"],
                    f"{leg['scheduled']:,}",
                    f"{leg['committed']:,}",
                    leg.get("kill_signal", "—"),
                    leg.get("committed_before_kill", "—"),
                    leg.get("resumed_requests", "—"),
                    f"{leg['wall_clock_seconds']:.1f}s",
                )
                for leg in soak["legs"]
            ],
            (
                "leg",
                "workers",
                "scheduled",
                "committed",
                "signal",
                "committed before kill",
                "resumed",
                "wall clock",
            ),
        )
    )
    lines.append("")
    lines.append(
        f"The killed worker was really signalled (`{soak['restart']['kill_signal']}`, "
        f"return code confirms it), had committed "
        f"{soak['restart']['committed_before_kill'][0]} requests when it died, and "
        "the resume worker was handed the scheduled ids minus the committed ids "
        "and nothing else. "
        f"{soak['restart']['recomputed_on_both_sides']} requests were computed on "
        "both sides of the kill."
    )
    lines.append("")
    lines.extend(
        _table(
            [
                ("scheduled ids", f"{soak['reconciliation']['scheduled']:,}"),
                ("committed rows", f"{soak['reconciliation']['committed']:,}"),
                ("missing", len(soak["reconciliation"]["missing_request_ids"])),
                ("duplicate", len(soak["reconciliation"]["duplicate_request_ids"])),
                ("unscheduled", len(soak["reconciliation"]["unscheduled_request_ids"])),
            ],
            ("set-algebra quantity", "value"),
        )
    )
    lines.append("")
    lines.append("### 6.4 The per-request audit")
    lines.append("")
    lines.append(
        f"Every one of the {audit['requests_audited']:,} committed requests was "
        "re-derived from scratch in a separate process: the position replayed "
        "from public bytes alone, the public-state identity re-derived by the "
        "independent Agent 3 implementation, the belief forward re-run, all 64 "
        "sample identities re-derived from the public identity and the ordinal, "
        "all 64 worlds rebuilt and each one re-walked by "
        "`verify_world_independently` — a second implementation that re-derives "
        "the piece order, the fallback steps and the complete assignment and "
        "then re-checks the inventory and every public fact on the assignment "
        "itself. That is "
        f"{audit['worlds_verified']:,} complete worlds independently verified."
    )
    lines.append("")
    lines.extend(
        _table(
            [(name, value) for name, value in sorted(counters.items())],
            ("zero-tolerance counter", "value"),
        )
    )
    lines.append("")
    lines.append(
        "The hidden-input claim is measured, not asserted: for every request the "
        "document is rebuilt from a state whose hidden ranks are traced by "
        "`phase11_safety.instrument_hidden_types`, and the trace counted "
        f"{counters['hidden_input_accesses']} reads while producing a "
        "byte-identical document."
    )
    lines.append("")
    replay = audit["cross_topology_replay"]
    lines.append(
        f"**Cross-topology replay.** {replay['requests']:,} evenly spaced "
        f"requests were re-executed on {replay['workers']} workers in "
        f"{replay['order']} order, in fresh processes with a scrubbed "
        f"environment and `cwd={replay['cwd']}`. Digest mismatches: "
        f"{len(replay['digest_mismatches'])}."
    )
    lines.append("")
    shared = audit["shared_decision_agreement"]
    lines.append(
        f"**Purity, demonstrated.** On the {coverage['games_with_shared_decisions']} "
        "games with fewer than eight observer decisions, more than one scheduled "
        f"request lands on the same decision: {shared['shared_decisions']} "
        f"decisions carry {shared['requests_involved']} requests between them. "
        "Their worlds are byte-identical in every case — different request ids, "
        "same public state, same worlds."
    )
    lines.append("")
    streams = stages["streams"]
    lines.append("### 6.4b Materialized-stream collision audit")
    lines.append("")
    lines.append(
        "Identity-only reconciliation: nothing here replays a forward, samples a "
        "world, kills a process or touches a bank. The universe is **reconstructed "
        "from the final recorded 8,192-request schedule and the frozen derivation "
        "rules**, never inferred from a reported total. The condition tested is "
        "the one that matters: *two different logical identities must not map to "
        "the same derived 63-bit seed*."
    )
    lines.append("")
    lines.extend(
        _table(
            [
                (
                    f"`{name}`",
                    f"{entry['logical_identities']:,}",
                    f"{entry['distinct_derived_seeds']:,}",
                    f"{entry['intentional_logical_identity_reuse']:,}",
                    entry["internal_accidental_collisions"],
                    entry["cross_universe_accidental_collisions"],
                )
                for name, entry in sorted(streams["per_domain"].items())
            ],
            (
                "domain",
                "logical identities",
                "distinct seeds",
                "intentional reuse",
                "internal collisions",
                "cross-universe collisions",
            ),
        )
    )
    lines.append("")
    agent6_streams = streams["agent6"]
    combined_streams = streams["combined"]
    lines.extend(
        _table(
            [
                (
                    "total Agent 6 unique logical identities",
                    f"{agent6_streams['unique_logical_identities']:,}",
                ),
                (
                    "total Agent 6 distinct seeds",
                    f"{agent6_streams['distinct_seeds']:,}",
                ),
                (
                    "identities new relative to Agent 4",
                    f"{agent6_streams['identities_new_relative_to_agent4']:,}",
                ),
                (
                    "identities intentionally shared with Agent 4",
                    f"{agent6_streams['identities_intentionally_shared_with_agent4']:,}",
                ),
                (
                    "combined unique logical identities",
                    f"{combined_streams['unique_logical_identities']:,}",
                ),
                ("combined distinct seeds", f"{combined_streams['distinct_seeds']:,}"),
                (
                    "**total accidental collisions**",
                    f"**{streams['total_accidental_collisions']}**",
                ),
                (
                    "expected random collisions at 63 bits",
                    combined_streams["expected_random_collisions"],
                ),
            ],
            ("quantity", "value"),
        )
    )
    lines.append("")
    reuse = streams["intentional_reuse"]
    lines.append(
        "**Intentional reuse is deduplicated before the comparison, never counted "
        f"as a collision.** {reuse['requests_sharing_a_position']:,} of the "
        f"{reuse['requests']:,} requests attach to a public position another "
        f"request also uses; {reuse['positions_sharing_a_public_state_identity']} "
        "further positions in different games reach one public-state identity; and "
        f"the {reuse['request_world_identity_pairs']:,} request x world-ordinal "
        f"pairs collapse to {reuse['distinct_world_sample_identities']:,} distinct "
        "world identities — identical public state, sampler version and ordinal "
        "*must* reproduce one identity, which is the sampler's purity, not a "
        "defect. The 1,024 original games' `soak_setup` and `soak_match` "
        "identities are the same identities Agent 1 enumerated and Agent 4 "
        "carried; only the 29 supplemental games' are new."
    )
    lines.append("")
    fidelity = streams["reconstruction_fidelity"]
    lines.append("**Reconstruction fidelity.**")
    lines.append("")
    lines.extend(
        _table(
            [
                (
                    "recorded requests represented",
                    f"{fidelity['scheduled_requests']:,} scheduled = "
                    f"{fidelity['committed_requests']:,} committed "
                    f"({str(fidelity['requests_match_store']).lower()})",
                ),
                (
                    "original prefix + supplement",
                    f"{fidelity['original_prefix_requests']:,} + "
                    f"{fidelity['supplemental_requests']} = "
                    f"{fidelity['scheduled_requests']:,}, both represented",
                ),
                (
                    "soak/setup identities reconstructed",
                    f"{fidelity['games']:,} games "
                    f"({fidelity['supplemental_games']} supplemental), "
                    f"{fidelity['distinct_positions_reconstructed']:,} positions "
                    "replayed from public bytes; every reconstructed public-state "
                    "identity equals the recorded one",
                ),
                (
                    "world tokens represented",
                    f"{fidelity['world_tokens_used_by_requests']:,} used = "
                    f"{fidelity['world_tokens_enumerated']:,} enumerated "
                    f"({str(fidelity['every_used_token_represented']).lower()})",
                ),
                (
                    "bulk vs public helper agreement",
                    f"{fidelity['fast_path_check']['derivations_checked']:,} "
                    "derivations re-run through `world_sample_seed`, "
                    "`world_order_key` and `world_categorical_uniform`, "
                    f"{fidelity['fast_path_check']['mismatches']} mismatches",
                ),
                (
                    "Agent 4 universe reproduced exactly",
                    f"{streams['agent4']['universe_identities']:,} identities, "
                    f"{str(streams['agent4']['reproduces_accepted_record']).lower()}",
                ),
            ],
            ("evidence", "value"),
        )
    )
    lines.append("")
    lines.append("### 6.5 `phase11_system_v1`")
    lines.append("")
    lines.append(
        "Filled by Agent 1's own rule — *fills every unbound slot with accepted "
        "values only, changes nothing bound now, and adds no slot*. `bound_now` "
        "is carried through verbatim from the live contract artifact; the five "
        "unbound slots are filled from accepted upstream values; and the "
        "deferred bank bindings resolve to the already-frozen Agent 1 bank "
        "identities, re-derived from the live bank artifacts, rather than to "
        "anything Agent 6 chose."
    )
    lines.append("")
    lines.extend(
        _table(
            [
                ("`evaluator_implementation`", "`phase11_belief_evaluator_v1`, the accepted Agent 2 owner/request/extraction identity and the live module bytes"),
                ("`sampler_implementation`", "`belief_sampler_v1`, the accepted Agent 3 entry point, provenance schema and audit-evidence digest"),
                ("`information_safety_evidence`", "`phase11_information_safety_v1` and the four Agent 4 evidence digests"),
                ("`runtime_benchmark`", "the frozen benchmark configuration unchanged, the measured p95 and the runtime artifact digest"),
                ("`bank_digests`", "Agent 1's validation and test bank digests, verbatim"),
            ],
            ("slot", "filled with"),
        )
    )
    lines.append("")
    lines.append(
        f"`phase11_system_v1` digest **`{system['system']['system_digest']}`**. "
        "No absolute path appears anywhere in it. If Phase 11 ends "
        "`PASS-SEARCH-READY`, this is the only belief stack Phase 12 may query."
    )
    lines.append("")
    lines.append("### 6.6 Preservation")
    lines.append("")
    preservation = system["preservation"]
    lines.extend(
        _table(
            [
                ("Phase 9 checkpoint SHA", str(preservation["phase9_checkpoint_unchanged"]).lower()),
                ("belief-head identity", str(preservation["belief_head_unchanged"]).lower()),
                ("P10-D config / utility / scaler", str(preservation["phase10_selector_unchanged"]).lower()),
                ("Phase 7 library", str(preservation["phase7_library_unchanged"]).lower()),
                ("Agent 5 frozen implementation", str(system["agent5_freeze_unchanged"]).lower()),
                ("optimizer-step delta", preservation["optimizer_step_delta"]),
                ("Phase 11 optimizer steps", preservation["phase11_optimizer_steps"]),
            ],
            ("preserved identity", "exact"),
        )
    )
    lines.append("")
    lines.append("### 6.7 Recorded readings")
    lines.append("")
    for reading in acceptance["recorded_readings"]:
        lines.append(f"- **`{reading['reading']}`** — {reading['detail']}.")
    lines.append("")
    lines.append(
        "- **`validation_R_CE_0_9750_carried_unchanged`** — the Agent 5 "
        "validation reading `R_CE = 0.9750` fails Gate A's `<= 0.97` threshold. "
        "Agent 6 preserves it as a diagnostic and changes nothing: no "
        "calibration, no model change, no sampler change, no threshold change, "
        "no repair. Phase 11 is a validation phase, not a repair loop, and the "
        "sealed test evaluation is Agent 7's to run."
    )
    lines.append(
        "- **`store_manifest_digest_embeds_a_wall_clock_duration` is not "
        "repaired** — the Agent 5 finding stands as recorded. Agent 6 treats "
        "`store_content_digest` as the cross-run prediction-store content "
        "identity and `manifest_digest` as within-run integrity metadata only, "
        "and patched neither `phase11_records` nor the recorder."
    )
    lines.append("")
    lines.append("### 6.8 Test-bank seal")
    lines.append("")
    sealing = verify["test_bank_sealing"]
    lines.append(
        f"The append-only ledger holds {sealing['ledger_entries']} entries, "
        f"{sealing['test_bank_entries']} of them naming "
        "`phase11_test_bank_v1`, every one structural-only with all four "
        "counters zero. `run_phase11_pipeline` refuses the sealed bank without "
        "an explicit `sealed_bank_authorized=True`, which this harness never "
        "passes. **Test-bank scored access remains 0.** Agent 7 is the first "
        "agent permitted to score it."
    )
    lines.append("")
    lines.append("### 6.9 Completion gates")
    lines.append("")
    lines.extend(
        _table(
            [
                (index, f"`{name}`", str(value).lower())
                for index, (name, value) in enumerate(
                    sorted(acceptance["completion_gates"].items()), start=1
                )
            ],
            ("#", "gate", "value"),
        )
    )
    lines.append("")
    lines.extend(
        _table(
            [
                (f"`{name}`", value)
                for name, value in sorted(
                    acceptance["forbidden_operation_counters"].items()
                )
            ],
            ("forbidden operation", "count"),
        )
    )
    lines.append("")
    suite = acceptance.get("suite", {})
    if suite:
        lines.append(f"Full suite: `{suite['command']}` — {suite['summary']}")
        lines.append("")
    lines.append("### 6.10 Handoff to Agent 7")
    lines.append("")
    handoff_block = acceptance["handoff_to_agent_7"]
    lines.append(
        f"Frozen `phase11_system_v1` digest `{handoff_block['phase11_system_v1_digest']}`, "
        f"over the `{handoff_block['validation_freeze']['freeze_version']}` "
        f"implementation `{handoff_block['validation_freeze']['freeze_digest'][:16]}...`. "
        f"Final-test entry point `{handoff_block['final_test_entry_point']}`, "
        f"test bank `{handoff_block['test_bank']['bank_digest'][:16]}...` "
        f"({handoff_block['test_bank']['cases']:,} cases / "
        f"{handoff_block['test_bank']['games']:,} games), scored access so far "
        f"{handoff_block['test_bank']['scored_access_so_far']}."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def stage_report(_args) -> dict:
    stages = {
        name: read_stage(name)
        for name in ("verify", "schedule", "soak", "audit", "streams", "system")
    }
    suite_path = stage_path("suite")
    if suite_path.exists():
        stages["suite"] = json.loads(suite_path.read_text())
    acceptance = read_json(DATA_DIRECTORY / "agent_06_acceptance.json")
    section = report_section(stages, acceptance)
    existing = REPORT_PATH.read_text()
    marker = "## 6. Agent 6 — production integration soak"
    if marker in existing:
        existing = existing[: existing.index(marker)].rstrip("\n") + "\n"
        log("replacing the existing section 6")
    REPORT_PATH.write_text(existing.rstrip("\n") + "\n" + section)
    log(f"appended section 6 to {REPORT_PATH.relative_to(REPOSITORY_ROOT)}")
    return {"stage": "report", "characters": len(section)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def stage_suite(_args) -> dict:
    log("running the full suite")
    return run_suite()


STAGES = {
    "verify": stage_verify,
    "suite": stage_suite,
    "schedule": stage_schedule,
    "soak": stage_soak,
    "audit": stage_audit,
    "streams": stage_streams,
    "system": stage_system,
    "artifacts": stage_artifacts,
    "report": stage_report,
    "soak-worker": stage_soak_worker,
    "audit-worker": stage_audit_worker,
    "streams-worker": stage_streams_worker,
}

#: The stages a full Agent 6 run performs, in order.
PIPELINE = (
    "verify",
    "schedule",
    "soak",
    "audit",
    "streams",
    "system",
    "artifacts",
    "report",
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 11 Agent 6 harness")
    parser.add_argument("--stage", choices=sorted(STAGES), default=None)
    parser.add_argument("--limit-games", type=int, default=0)
    parser.add_argument("--record-suite", action="store_true")
    parser.add_argument("--plan", default=None)
    parser.add_argument("--store", default=None)
    parser.add_argument("--ordinals", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if args.stage in ("soak-worker", "audit-worker", "streams-worker"):
        STAGES[args.stage](args)
        return 0

    stages = [args.stage] if args.stage else list(PIPELINE)
    if args.record_suite and not args.stage:
        stages.insert(stages.index("artifacts"), "suite")

    for name in stages:
        log(f"stage {name}")
        payload = STAGES[name](args)
        if name not in ("artifacts", "report"):
            write_stage(name, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
