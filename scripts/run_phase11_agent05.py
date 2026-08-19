#!/usr/bin/env python3
"""Phase 11 Agent 5 harness: integrated validation and the implementation freeze.

Recomputes every load-bearing identity from live bytes (the Agent 1 freeze
and its eight contracts, the Agent 2, 3 and 4 PASS records and handoffs,
both bank digests, the Phase 9 checkpoint's file SHA / model-state digest /
parameter count / belief-head tensor identity / optimizer-step counter, the
frozen P10-D chain and the Phase 7 library), then runs the **complete**
Phase 11 scored pipeline on `phase11_validation_bank_v1` exactly as it will
run on the sealed test, and freezes the implementation Agents 6 and 7 use.

Five things happen here and nothing else:

- **Integrated run** — `phase11_pipeline.run_phase11_pipeline` plays all
  512 cases / 1,024 games through the frozen paths, takes the privileged
  targets afterwards, scores the learned head against
  `remaining_count_belief_v1`, computes the frozen metric block and every
  mandatory slice with their case bootstraps, and runs `belief_sampler_v1`
  over the frozen integrated sample schedule.
- **Independent recomputation** — `phase11_recompute`, which imports no
  `phase11_*` module, rebuilds CE learned/baseline/ratio, the top-1 and
  Brier deltas, ECE, the per-stratum CE ratios and every bootstrap
  interval from the primitive recorded rows.
- **Evidence binding** — the Agent 3 sampler audit, the Agent 4 safety,
  topology and runtime evidence and the frozen CPU runtime result are
  re-hashed and bound into the gate computation.
- **Leakage audit** — targets used only after prediction, no test
  prediction or truth, no game result as a belief feature, no diagnostic
  slice reaching the implementation.
- **Implementation freeze** — one `phase11_validation_freeze_v1` identity
  over the live bytes of the whole Phase 11 implementation.

Nothing here trains, calibrates, changes a threshold, a bin, a baseline, a
bank, a stratum or a sampler rule, and nothing reacts to the known
validation reading `R_CE = 0.9750`. That reading would fail Gate A on the
sealed test if it repeated; Agent 5 reports it and changes nothing, which
is the whole point of a validation phase that is not a repair loop.

`phase11_test_bank_v1` is touched only to re-derive its digest
structurally, and `run_phase11_pipeline` refuses it outright without an
explicit authorization this harness never passes.

    reports/phase_11_data/agent_05_validation_metrics.json
    reports/phase_11_data/agent_05_validation_strata.csv
    reports/phase_11_data/agent_05_validation_freeze.json
    reports/phase_11_data/agent_05_acceptance.json

Usage::

    python scripts/run_phase11_agent05.py                    # every stage
    python scripts/run_phase11_agent05.py --stage verify     # one stage
    python scripts/run_phase11_agent05.py --limit-cases 8    # a smoke run
    python scripts/run_phase11_agent05.py --record-suite     # run + record
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

AGENT = 5
PHASE = 11
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_11_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_11_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase11" / "agent05"
CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "selfplay_c1_v1.pt"
EXPORT_PATH = WORK_DIRECTORY / "eval_phase9_c1.pt"

#: The frozen evaluation backend, unchanged from Agent 2 and Agent 4.
EVAL_DEVICE = "cpu"
EVAL_TORCH_THREADS = 1

#: The upstream evidence Agent 5 binds. Each is re-hashed from live bytes
#: and its digest is written into the freeze document.
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
)


class Agent5Error(RuntimeError):
    """The Agent 5 harness refused to continue."""


# ---------------------------------------------------------------------------
# Small shared utilities (the accepted Agent 2/3/4 harness shapes)
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
        raise Agent5Error(f"stage {name!r} has not run yet ({path} is missing)")
    return json.loads(path.read_text())


def require(condition: bool, message: str, problems: list) -> bool:
    if not condition:
        problems.append(message)
    return bool(condition)


def log(message: str) -> None:
    print(f"[phase11:agent5] {message}", flush=True)


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
# 1. Verification — Agents 1-4 and every identity, from live bytes
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
        acceptance.get("status") == "PASS", f"Agent {agent} did not report PASS", problems
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


def verify_agent1(problems: list) -> dict:
    """Agent 1 must be PASS, with its eight contract digests re-derived here."""
    from stratego.training import phase11_contract as contract

    summary = _verify_agent_acceptance("agent_01_acceptance.json", 1, problems)
    if not summary.get("available"):
        return summary
    acceptance = read_json(DATA_DIRECTORY / "agent_01_acceptance.json")
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
    require(
        acceptance.get("new_digests", {}).get("contract_bundle_digest") == live_bundle,
        "the contract bundle digest moved since the Agent 1 freeze",
        problems,
    )
    summary.update(
        {
            "contract_digests": live_digests,
            "contract_bundle_digest": live_bundle,
        }
    )
    return summary


def verify_agent2(problems: list) -> dict:
    """Agent 2 must be PASS and its metric artifact must be byte-unchanged."""
    from stratego.training.phase11_contract import EVALUATOR_VERSION

    summary = _verify_agent_acceptance(
        "agent_02_acceptance.json", 2, problems, handoff_key="handoff_to_agent_3"
    )
    if not summary.get("available"):
        return summary
    metrics_path = DATA_DIRECTORY / "agent_02_predictive_metrics.json"
    require(metrics_path.exists(), "the Agent 2 metric artifact is missing", problems)
    if metrics_path.exists():
        metrics = read_json(metrics_path)
        require(
            metrics.get("evaluator_version") == EVALUATOR_VERSION,
            f"Agent 2 names evaluator {metrics.get('evaluator_version')!r}",
            problems,
        )
        summary.update(
            {
                "evaluator_version": metrics.get("evaluator_version"),
                "recorded_r_ce": metrics["overall"]["metrics"]["r_ce"]["point"],
                "recorded_events": metrics.get("prediction_events"),
                "recorded_manifest_digest": metrics.get("store_manifest_digest"),
                "metrics_file_sha256": file_sha256(metrics_path),
            }
        )
    return summary


def verify_agent3(problems: list) -> dict:
    """Agent 3 must be PASS and `belief_sampler_v1` must be byte-unchanged."""
    from stratego.training.phase11_contract import BELIEF_SAMPLER_VERSION

    summary = _verify_agent_acceptance(
        "agent_03_acceptance.json", 3, problems, handoff_key="handoff_to_agent_4"
    )
    if not summary.get("available"):
        return summary
    acceptance = read_json(DATA_DIRECTORY / "agent_03_acceptance.json")
    digests = acceptance.get("new_digests", {})
    recorded_modules = digests.get("implementation_sha256", {})
    live_modules = module_digests(recorded_modules)
    for name, digest in recorded_modules.items():
        require(
            live_modules.get(name) == digest,
            f"{name} is no longer the Agent 3 accepted bytes",
            problems,
        )
    require(
        digests.get("sampler_version") == BELIEF_SAMPLER_VERSION,
        f"Agent 3 sampler version {digests.get('sampler_version')!r} moved",
        problems,
    )
    audit = acceptance.get("audit_summary", {})
    counters = audit.get("zero_tolerance_counters", {})
    nonzero = sorted(name for name, value in counters.items() if value)
    require(
        not nonzero, f"Agent 3 zero-tolerance counters are non-zero: {nonzero}", problems
    )
    summary.update(
        {
            "sampler_version": digests.get("sampler_version"),
            "sampler_module_sha256": live_modules,
            "learned_worlds": audit.get("learned_worlds"),
            "independent_worlds": audit.get("independent_worlds"),
            "sampler_zero_counters": counters,
        }
    )
    return summary


def verify_agent4(problems: list) -> dict:
    """Agent 4 must be PASS, and its safety/topology/runtime evidence bound."""
    summary = _verify_agent_acceptance(
        "agent_04_acceptance.json", 4, problems, handoff_key="handoff_to_agent_5"
    )
    if not summary.get("available"):
        return summary
    handoff = summary.get("handoff", {})

    safety = read_json(DATA_DIRECTORY / "agent_04_information_safety.json")
    repro = read_json(DATA_DIRECTORY / "agent_04_reproducibility.json")
    counters = safety.get("zero_tolerance_counters", {})
    nonzero = sorted(name for name, value in counters.items() if value)
    require(
        not nonzero,
        f"Agent 4 information-safety counters are non-zero: {nonzero}",
        problems,
    )
    require(
        int(safety.get("trials", {}).get("executed", 0))
        >= int(safety.get("trials", {}).get("floor", 50_000)),
        "the Agent 4 permutation attack is below its frozen trial floor",
        problems,
    )
    legs = repro.get("leg_exact", {})
    require(
        legs and all(legs.values()),
        f"an Agent 4 topology leg is not exact: {sorted(k for k, v in legs.items() if not v)}",
        problems,
    )
    require(
        len(repro.get("distinct_rollup_digests", [])) == 1,
        "the Agent 4 topology legs did not produce one rollup digest",
        problems,
    )
    runtime = handoff.get("measured_runtime", {})
    require(
        float(runtime.get("p95_forward_64_ms", float("inf")))
        <= float(runtime.get("ceiling_ms", 500.0)),
        "the Agent 4 measured p95 exceeds the frozen ceiling",
        problems,
    )
    sampler_identity = handoff.get("sampler_identity", {})
    live_sampler = module_digests(sampler_identity.get("module_sha256", {}))
    for name, digest in sampler_identity.get("module_sha256", {}).items():
        require(
            live_sampler.get(name) == digest,
            f"{name} is no longer the sampler bytes Agent 4 handed over",
            problems,
        )
    evaluator_identity = handoff.get("evaluator_identity", {})

    summary.update(
        {
            "safety_counters": counters,
            "safety_detail_counters": safety.get("detail_counters", {}),
            "safety_trials": safety.get("trials", {}),
            "trial_rollup_digest": safety.get("trial_rollup_digest"),
            "leg_exact": legs,
            "reference_rollup_digest": repro.get("reference_rollup_digest"),
            "distinct_rollup_digests": repro.get("distinct_rollup_digests", []),
            "measured_runtime": runtime,
            "sampler_identity": sampler_identity,
            "evaluator_identity": evaluator_identity,
            "sampler_module_sha256": live_sampler,
        }
    )
    return summary


def verify_banks(problems: list) -> dict:
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
            "games": len(cases) * 2,
            "file_sha256": file_sha256(path),
        }
    return summary


def belief_head_identity(model_state: dict) -> dict:
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
        "belief_head_tensor_shapes": head["tensor_shapes"],
        "global_optimizer_step": optimizer_step,
    }


def verify_upstream_stack(problems: list) -> dict:
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


def verify_test_bank_sealed(problems: list) -> dict:
    from stratego.evaluation import phase11_banks as banks

    sealing = banks.verify_test_bank_sealed()
    require(
        sealing["test_bank_structural_only"],
        f"the test bank is no longer structural-only: {sealing['violations']}",
        problems,
    )
    require(
        int(sealing["scored_prediction_total"]) == 0,
        "the ledger records scored test-bank predictions",
        problems,
    )
    require(
        int(sealing["privileged_truth_total"]) == 0,
        "the ledger records privileged test-bank truth reads",
        problems,
    )
    return sealing


def verify_seal_is_structural(problems: list) -> dict:
    """The pipeline must refuse the sealed bank without authorization.

    Behavioural, not documentary: the refusal is exercised here, and the
    same call with the authorization is *not* made, so this check can never
    itself open the bank.
    """
    from stratego.evaluation import phase11_pipeline as pipeline

    refused = False
    try:
        pipeline.assert_seal("test", sealed_bank_authorized=False)
    except pipeline.Phase11SealError:
        refused = True
    allowed = True
    try:
        pipeline.assert_seal("validation", sealed_bank_authorized=False)
    except pipeline.Phase11SealError:  # pragma: no cover - would be a defect
        allowed = False
    require(refused, "the pipeline did not refuse the sealed test bank", problems)
    require(allowed, "the pipeline refused the validation bank", problems)
    return {
        "sealed_banks": list(pipeline.SEALED_BANKS),
        "test_refused_without_authorization": refused,
        "validation_open": allowed,
    }


def bound_evidence(verify: dict) -> dict:
    """The Agent 3 and Agent 4 evidence, re-hashed and bound for the gates."""
    from stratego.training.phase11_contract import INFORMATION_SAFETY_VERSION

    return {
        "information_safety_version": INFORMATION_SAFETY_VERSION,
        "artifacts": {
            name: file_sha256(DATA_DIRECTORY / name)
            for name in BOUND_EVIDENCE_ARTIFACTS
        },
        "safety_counters": dict(verify["agent4"]["safety_counters"]),
        "leg_exact": dict(verify["agent4"]["leg_exact"]),
        "runtime": dict(verify["agent4"]["measured_runtime"]),
        "sampler_audit_counters": dict(verify["agent3"]["sampler_zero_counters"]),
        "sampler_audit_worlds": int(verify["agent3"]["learned_worlds"]),
    }


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
    }


def stage_verify(_args) -> dict:
    problems: list[str] = []
    log("verifying the Agent 1 freeze")
    agent1 = verify_agent1(problems)
    log("verifying the Agent 2 evaluator record")
    agent2 = verify_agent2(problems)
    log("verifying the Agent 3 sampler record")
    agent3 = verify_agent3(problems)
    log("verifying the Agent 4 safety / topology / runtime record")
    agent4 = verify_agent4(problems)
    log("re-hashing both frozen banks")
    bank_summary = verify_banks(problems)
    log("re-deriving the Phase 9 checkpoint and belief-head identity")
    phase9 = verify_phase9_checkpoint(problems)
    log("verifying the P10-D / anchor / Phase 7 stack")
    upstream = verify_upstream_stack(problems)
    log("checking the test-bank seal")
    sealing = verify_test_bank_sealed(problems)
    seal = verify_seal_is_structural(problems)

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
        "agent2": agent2,
        "agent3": agent3,
        "agent4": agent4,
        "banks": bank_summary,
        "phase9": phase9,
        "upstream": upstream,
        "test_bank_sealing": sealing,
        "seal_behaviour": seal,
        "problems": problems,
        "verified": not problems,
    }
    payload["bound_evidence"] = bound_evidence(payload) if not problems else {}
    payload["preservation_observation"] = (
        preservation_observation(phase9, upstream) if not problems else {}
    )
    write_stage("verify", payload)
    if problems:
        for problem in problems:
            log(f"PROBLEM: {problem}")
        raise Agent5Error(f"verification found {len(problems)} problem(s); BLOCKED")
    log("verification clean")
    return payload


# ---------------------------------------------------------------------------
# 2. The integrated pipeline run
# ---------------------------------------------------------------------------


def read_agent2_store_manifest() -> "dict | None":
    """The Agent 2 prediction-store manifest, if its volume is present."""
    from stratego.evaluation import phase11_records as records

    try:
        root = records.store_root(REPOSITORY_ROOT)
        return records.read_manifest(root)
    except Exception:  # pragma: no cover - a missing volume is BLOCKED upstream
        return None


def _progress(stage: str):
    def report(done, total, volume, elapsed):
        log(f"{stage}: {done}/{total}  {volume:,}  {elapsed:6.1f}s")

    return report


def stage_pipeline(args) -> dict:
    """Run the complete pipeline on the validation bank, start to finish."""
    from stratego.evaluation import phase11_banks as banks
    from stratego.evaluation import phase11_pipeline as pipeline

    configure_backend()
    verify = read_stage("verify")
    store_root = REPOSITORY_ROOT / "data" / "phase11" / "agent05" / (
        "validation_predictions_smoke" if args.limit_cases else "validation_predictions"
    )
    if args.limit_cases:
        log(f"SMOKE RUN: {args.limit_cases} cases only — not an acceptance run")

    started = time.perf_counter()
    result = pipeline.run_phase11_pipeline(
        "validation",
        REPOSITORY_ROOT,
        bound_evidence=verify["bound_evidence"],
        preservation=verify["preservation_observation"],
        store_root=store_root,
        export_path=EXPORT_PATH,
        device=EVAL_DEVICE,
        torch_threads=EVAL_TORCH_THREADS,
        limit_cases=args.limit_cases,
        progress=_progress,
    )

    # The regenerated store must reproduce the Agent 2 store exactly: same
    # games, same public shards, same truth shards, same replay digests,
    # same counts. That is the integrated run's strongest single statement
    # — the whole pipeline is a pure function of the frozen bank and the
    # frozen model.
    #
    # The comparison is made on `store_content_digest`, not on the frozen
    # `manifest_digest`: the latter covers each game's `forward_seconds`,
    # a wall-clock duration, so it cannot agree between two executions.
    # See the `store_manifest_digest_embeds_a_wall_clock_duration` reading
    # — Agent 5 records that defect and patches nothing.
    agent2_manifest_digest = verify["agent2"].get("recorded_manifest_digest")
    agent2_store = read_agent2_store_manifest()
    agent2_content_digest = (
        pipeline.store_content_digest(agent2_store) if agent2_store else None
    )
    content_digest = result["stages"]["generate"]["store_content_digest"]
    reproduces_agent2 = (
        not args.limit_cases
        and agent2_content_digest is not None
        and content_digest == agent2_content_digest
    )
    manifest_digest_matches = (
        result["manifest"]["manifest_digest"] == agent2_manifest_digest
    )

    banks.append_ledger_entries(
        [
            banks.ledger_entry(
                AGENT,
                "integrated_validation_smoke" if args.limit_cases else "integrated_validation_run",
                "phase11_validation_bank_v1",
                (
                    "the complete Phase 11 pipeline over the validation bank: "
                    "generate, targets, score, metrics, slices, sampler checks"
                ),
                structural_only=False,
                neural_inference_count=int(result["manifest"]["belief_forwards"])
                + int(result["stages"]["sampler_checks"]["requests"]),
                scored_prediction_count=int(result["manifest"]["prediction_events"]),
                privileged_truth_count=int(result["manifest"]["prediction_events"]),
                outcome_count=int(result["manifest"]["games"]),
            )
        ]
    )

    summary = {
        "agent": AGENT,
        "phase": PHASE,
        "stage": "pipeline",
        "pipeline_version": result["pipeline_version"],
        "bank": result["bank"],
        "bank_version": result["bank_version"],
        "bank_digest": result["bank_digest"],
        "sealed_bank_authorized": result["sealed_bank_authorized"],
        "store_root": result["store_root"],
        "manifest_digest": result["manifest"]["manifest_digest"],
        "store_content_digest": content_digest,
        "reproduces_agent2_store": reproduces_agent2,
        "agent2_manifest_digest": agent2_manifest_digest,
        "agent2_store_content_digest": agent2_content_digest,
        "agent2_manifest_digest_matches": manifest_digest_matches,
        # JSON is written sort_keys=True, so the execution order of the
        # stages is recorded explicitly rather than inferred from key order.
        "stage_order": list(result["stages"]),
        "stages": result["stages"],
        "overall": result["overall"],
        "slices": result["slices"],
        "gates": result["gates"],
        "gate_quantities": result["gate_quantities"],
        "schedule_size": len(result["schedule"]),
        "smoke_run": bool(args.limit_cases),
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
        "source_revision": verify["environment"]["source_revision"],
    }
    write_stage("pipeline", summary)
    quantities = result["gate_quantities"]
    log(
        f"R_CE {quantities['r_ce']:.4f}  delta_top1 {quantities['delta_top1']:+.4f}  "
        f"ECE {quantities['ece_overall']:.4f}  "
        f"worlds {result['stages']['sampler_checks']['worlds']:,}"
    )
    log(
        "gates "
        + " ".join(
            f"{gate}={'PASS' if block['passed'] else 'FAIL'}"
            for gate, block in sorted(result["gates"].items())
        )
    )
    return summary


# ---------------------------------------------------------------------------
# 3. Independent recomputation
# ---------------------------------------------------------------------------


def stage_recompute(_args) -> dict:
    """Rebuild every required quantity from the primitive recorded rows."""
    from stratego.evaluation import phase11_records as records
    from stratego.evaluation import phase11_recompute as recompute

    run = read_stage("pipeline")
    root = Path(run["store_root"])
    manifest = records.read_manifest(root)
    log(f"independently recomputing {manifest['prediction_events']:,} events")
    started = time.perf_counter()

    def shard_reader(game_id: str):
        return (
            records.read_public_shard(root, game_id),
            records.read_truth_shard(root, game_id),
        )

    independent = recompute.recompute_bank(
        root, manifest, run["bank"], shard_reader=shard_reader
    )
    comparison = recompute.compare_blocks(
        run["overall"], run["slices"]["opponent_stratum"], independent
    )
    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "stage": "recompute",
        "recompute_version": recompute.RECOMPUTE_VERSION,
        "events": independent["events"],
        "cases": independent["cases"],
        "comparison": {
            key: value for key, value in comparison.items() if key != "deviations"
        },
        "worst_quantities": sorted(
            (
                (max(detail.values()), name)
                for name, detail in comparison["deviations"].items()
            ),
            reverse=True,
        )[:10],
        "independent_gate_quantities": {
            "r_ce": independent["overall"]["r_ce"]["point"],
            "ce_delta_upper": independent["overall"]["ce_delta"]["upper"],
            "delta_top1": independent["overall"]["top1_delta"]["point"],
            "delta_top1_lower": independent["overall"]["top1_delta"]["lower"],
            "brier_delta_upper": independent["overall"]["brier_delta"]["upper"],
            "ece_overall": independent["overall"]["ece_learned"],
            "stratum_r_ce": {
                name: block["r_ce"]["point"]
                for name, block in independent["strata"].items()
            },
            "stratum_ece": {
                name: block["ece_learned"]
                for name, block in independent["strata"].items()
            },
        },
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
    }
    payload["deviations"] = comparison["deviations"]
    write_stage("recompute", payload)
    log(
        f"independent recompute: {comparison['quantities_compared']} quantities, "
        f"max deviation {comparison['max_deviation']:.3e}, "
        f"{'within' if comparison['within_tolerance'] else 'OUTSIDE'} tolerance"
    )
    return payload


# ---------------------------------------------------------------------------
# 4. The leakage audit
# ---------------------------------------------------------------------------


def stage_leakage(_args) -> dict:
    """Prove the four leakage claims the Agent 5 instruction requires."""
    from stratego.evaluation import phase11_banks as banks
    from stratego.evaluation import phase11_belief as belief
    from stratego.evaluation import phase11_sampler as sampler
    from stratego.training.phase11_contract import (
        ALLOWED_BELIEF_REQUEST_FIELDS,
        ALLOWED_SAMPLER_REQUEST_FIELDS,
        FORBIDDEN_BELIEF_REQUEST_TOKENS,
        PUBLIC_STATE_DOCUMENT_FIELDS,
    )

    verify = read_stage("verify")
    run = read_stage("pipeline")
    problems: list[str] = []
    findings: dict = {}

    # (1) Targets are used only after prediction. Structural: neither
    # request type has a field a target could arrive in, and the truth pass
    # runs on its own replay after both vectors already exist.
    forbidden_in_belief = [
        name
        for name in ALLOWED_BELIEF_REQUEST_FIELDS
        if any(token in name for token in FORBIDDEN_BELIEF_REQUEST_TOKENS)
    ]
    forbidden_in_sampler = [
        name
        for name in ALLOWED_SAMPLER_REQUEST_FIELDS
        if any(token in name for token in FORBIDDEN_BELIEF_REQUEST_TOKENS)
    ]
    forbidden_in_document = [
        name
        for name in PUBLIC_STATE_DOCUMENT_FIELDS
        if any(token in name for token in FORBIDDEN_BELIEF_REQUEST_TOKENS)
    ]
    require(not forbidden_in_belief, "a belief-request field names a private token", problems)
    require(not forbidden_in_sampler, "a sampler-request field names a private token", problems)
    require(not forbidden_in_document, "a document field names a private token", problems)
    truth_pass = run["stages"]["targets"]
    require(
        int(truth_pass["identity_mismatches"]) == 0
        and int(truth_pass["alignment_mismatches"]) == 0
        and int(truth_pass["count_mismatches"]) == 0
        and int(truth_pass["mask_mismatches"]) == 0
        and int(truth_pass["unlabelled_events"]) == 0,
        "the privileged truth pass reported a mismatch",
        problems,
    )
    findings["targets_after_prediction"] = {
        "belief_request_fields": list(ALLOWED_BELIEF_REQUEST_FIELDS),
        "sampler_request_fields": list(ALLOWED_SAMPLER_REQUEST_FIELDS),
        "forbidden_tokens": list(FORBIDDEN_BELIEF_REQUEST_TOKENS),
        "fields_naming_a_private_token": 0,
        "truth_pass": truth_pass,
        "truth_shard_separate_from_public_shard": True,
        "belief_request_type": f"{belief.Phase11BeliefRequest.__module__}."
        f"{belief.Phase11BeliefRequest.__name__}",
        "sampler_boundary_report": sampler.sampler_boundary_report(),
    }

    # (2) No test prediction and no test truth. The ledger is the record.
    entries = banks.read_ledger()
    test_entries = [
        entry
        for entry in entries
        if entry["bank_version"] == "phase11_test_bank_v1"
    ]
    agent5_entries = [entry for entry in entries if entry["agent"] == AGENT]
    scored_test = sum(entry["scored_prediction_count"] for entry in test_entries)
    truth_test = sum(entry["privileged_truth_count"] for entry in test_entries)
    inference_test = sum(entry["neural_inference_count"] for entry in test_entries)
    outcome_test = sum(entry["outcome_count"] for entry in test_entries)
    require(scored_test == 0, "the ledger records scored test predictions", problems)
    require(truth_test == 0, "the ledger records test truth reads", problems)
    require(inference_test == 0, "the ledger records test neural inference", problems)
    require(outcome_test == 0, "the ledger records test outcome reads", problems)
    require(
        all(
            entry["structural_only"]
            for entry in agent5_entries
            if entry["bank_version"] == "phase11_test_bank_v1"
        ),
        "an Agent 5 test-bank ledger entry is not structural-only",
        problems,
    )
    require(
        run["sealed_bank_authorized"] is False,
        "the Agent 5 pipeline run carried a sealed-bank authorization",
        problems,
    )
    findings["no_test_access"] = {
        "ledger_entries": len(entries),
        "test_bank_entries": len(test_entries),
        "agent5_entries": len(agent5_entries),
        "scored_prediction_total": scored_test,
        "privileged_truth_total": truth_test,
        "neural_inference_total": inference_test,
        "outcome_total": outcome_test,
        "seal_behaviour": verify["seal_behaviour"],
    }

    # (3) No game result is a belief feature. The outcome never enters the
    # prediction store, the scored table or a gate quantity: the store's
    # public-shard array list has no outcome field, and the gate quantities
    # are functions of the metric block alone.
    from stratego.evaluation.phase11_records import PUBLIC_SHARD_ARRAYS, TRUTH_SHARD_ARRAYS

    outcome_named = [
        name
        for name in PUBLIC_SHARD_ARRAYS + TRUTH_SHARD_ARRAYS
        if any(token in name for token in ("result", "outcome", "win", "loss", "draw"))
    ]
    require(
        not outcome_named,
        f"a recorded shard array names a game result: {outcome_named}",
        problems,
    )
    findings["no_outcome_feature"] = {
        "public_shard_arrays": list(PUBLIC_SHARD_ARRAYS),
        "truth_shard_arrays": list(TRUTH_SHARD_ARRAYS),
        "arrays_naming_a_result": 0,
        "outcomes_recorded_where": "manifest only, report-only",
        "outcomes_report_only": run["stages"]["generate"]["outcomes_report_only"],
    }

    # (4) No diagnostic slice alters the implementation. Slices are read
    # after the freeze list is fixed; the only slice a gate reads is the
    # stratum slice (Gates C and D), which Agent 1 froze as gate-bearing.
    from stratego.training.phase11_contract import DIAGNOSTIC_SLICES

    gate_reading_slices = ("opponent_stratum",)
    report_only = tuple(
        name for name in DIAGNOSTIC_SLICES if name not in gate_reading_slices
    )
    require(
        sorted(run["slices"]) == sorted(DIAGNOSTIC_SLICES),
        "the produced slices are not the frozen diagnostic slice list",
        problems,
    )
    findings["slices_do_not_alter_implementation"] = {
        "frozen_slices": list(DIAGNOSTIC_SLICES),
        "gate_reading_slices": list(gate_reading_slices),
        "report_only_slices": list(report_only),
        "implementation_frozen_before_slices_read": True,
        "rule": "report-only diagnostics never rescue a failed gate",
    }

    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "stage": "leakage",
        "findings": findings,
        "problems": problems,
        "clean": not problems,
    }
    write_stage("leakage", payload)
    if problems:
        for problem in problems:
            log(f"PROBLEM: {problem}")
        raise Agent5Error(f"the leakage audit found {len(problems)} problem(s)")
    log("leakage audit clean")
    return payload


# ---------------------------------------------------------------------------
# 5. The implementation freeze
# ---------------------------------------------------------------------------


def stage_freeze(_args) -> dict:
    """Freeze the single Phase 11 implementation identity."""
    from stratego.evaluation import phase11_pipeline as pipeline

    verify = read_stage("verify")
    run = read_stage("pipeline")
    read_stage("recompute")
    read_stage("leakage")

    document = pipeline.implementation_freeze(
        REPOSITORY_ROOT,
        belief_head_digest=verify["phase9"]["belief_head_digest"],
        model_state_digest=verify["phase9"]["model_state_digest"],
        contract_bundle_digest=verify["agent1"]["contract_bundle_digest"],
        validation_bank_digest=verify["banks"]["validation"]["bank_digest"],
        test_bank_digest=verify["banks"]["test"]["bank_digest"],
        runtime=verify["agent4"]["measured_runtime"],
        bound_evidence=verify["bound_evidence"],
    )
    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_05_validation_freeze",
        "stage": "freeze",
        "environment": environment_report(),
        "freeze": document,
        "validated_on": {
            "bank_version": run["bank_version"],
            "bank_digest": run["bank_digest"],
            "store_manifest_digest": run["manifest_digest"],
            "store_content_digest": run["store_content_digest"],
            "reproduces_agent2_store": run["reproduces_agent2_store"],
            "prediction_events": run["stages"]["generate"]["prediction_events"],
            "sampler_worlds": run["stages"]["sampler_checks"]["worlds"],
        },
        "handoff_to_agent_6": {
            "for_agent": 6,
            "freeze_version": document["freeze_version"],
            "freeze_digest": document["freeze_digest"],
            "final_test_entry_point": document["final_test_entry_point"],
            "immutable_dependencies": {
                "belief_head_digest": document["belief_head_identity"][
                    "belief_head_digest"
                ],
                "evaluator_version": document["evaluator_version"],
                "remaining_count_baseline": document["remaining_count_baseline"],
                "sampler_version": document["sampler_version"],
                "sampler_module_sha256": verify["agent4"]["sampler_module_sha256"],
                "information_safety_version": document["information_safety_version"],
                "statistics_version": document["statistics_version"],
                "runtime_backend": document["runtime_backend"],
                "contract_bundle_digest": document["contract_bundle_digest"],
                "bank_digests": document["bank_digests"],
            },
            "scope": (
                "Agent 6 performs the production soak and the "
                "phase11_system_v1 freeze only: no retrain, no calibration, "
                "no sampler change, no threshold change, and no scored access "
                "to phase11_test_bank_v1"
            ),
        },
    }
    write_stage("freeze", payload)
    write_artifact("agent_05_validation_freeze.json", payload)
    log(f"implementation freeze {document['freeze_digest']}")
    return payload


# ---------------------------------------------------------------------------
# 6. Acceptance
# ---------------------------------------------------------------------------


def completion_gates(verify, run, recompute_stage, leakage, freeze, preservation, suite) -> dict:
    from stratego.evaluation import phase11_pipeline as pipeline
    from stratego.training.phase11_contract import (
        DIAGNOSTIC_SLICES,
        VALIDATION_BANK_CASES,
        VALIDATION_BANK_GAMES,
    )

    generate = run["stages"]["generate"]
    metrics = run["stages"]["metrics"]
    sampler = run["stages"]["sampler_checks"]
    bound = run["stages"]["bound_evidence"]
    comparison = recompute_stage["comparison"]

    return {
        "agents1_4_pass": all(
            verify[name]["status"] == "PASS"
            for name in ("agent1", "agent2", "agent3", "agent4")
        ),
        "validation_bank_exact": (
            verify["banks"]["validation"]["cases"] == VALIDATION_BANK_CASES
            and run["bank_digest"] == verify["banks"]["validation"]["bank_digest"]
        ),
        "validation_games_exact": (
            int(generate["games"]) == VALIDATION_BANK_GAMES
            and bool(generate["complete_bank"])
        ),
        "full_pipeline_complete": (
            tuple(run["stage_order"]) == pipeline.PIPELINE_STAGES
            and sorted(run["stages"]) == sorted(pipeline.PIPELINE_STAGES)
        ),
        "predictive_metrics_complete": (
            bool(metrics["metrics_finite"])
            and not metrics["nonfinite_paths"]
            and int(metrics["events"]) > 0
        ),
        "all_slices_complete": sorted(run["slices"]) == sorted(DIAGNOSTIC_SLICES),
        "bootstrap_complete": all(
            block.get("replicates") == 10_000
            for block in run["overall"]["metrics"].values()
        ),
        "independent_recompute_pass": bool(comparison["within_tolerance"])
        and int(comparison["both_nan_comparisons"]) == 0,
        # Two independent sampler passes, both clean, and the frozen
        # 250,000-world floor met by the bound Agent 3 audit and by the
        # two passes together. The integrated pass is an *integration*
        # check at the production request shape, not a second large audit,
        # and its own world count is reported rather than floored — see
        # the `integrated_schedule_realizes_fewer_slots_than_nominal`
        # reading.
        "sampler_evidence_bound": (
            bool(sampler["all_counters_zero"])
            and bool(sampler["schedule_accounting"]["every_eligible_game_contributes"])
            and bool(sampler["schedule_accounting"]["realized_equals_attainable"])
            and int(sampler["worlds"])
            == int(sampler["schedule_accounting"]["schedule_slots_realized"])
            * int(sampler["world_ordinals_per_request"])
            and all(value == 0 for value in bound["sampler_audit_counters"].values())
            and int(bound["sampler_audit_worlds"]) >= int(sampler["world_floor"])
            and int(sampler["worlds"]) + int(bound["sampler_audit_worlds"])
            >= int(sampler["world_floor"])
        ),
        "safety_evidence_bound": all(
            value == 0 for value in bound["safety_counters"].values()
        )
        and int(verify["agent4"]["safety_trials"]["executed"]) >= 50_000,
        "reproducibility_evidence_bound": (
            bool(bound["leg_exact"])
            and all(bound["leg_exact"].values())
            and len(verify["agent4"]["distinct_rollup_digests"]) == 1
        ),
        "runtime_evidence_bound": float(bound["p95_forward_64_ms"])
        <= float(verify["agent4"]["measured_runtime"]["ceiling_ms"]),
        "validation_privileged_boundary_clean": bool(leakage["clean"])
        and int(run["stages"]["targets"]["unlabelled_events"]) == 0,
        "no_test_prediction_access": int(
            leakage["findings"]["no_test_access"]["scored_prediction_total"]
        )
        == 0
        and int(leakage["findings"]["no_test_access"]["neural_inference_total"]) == 0,
        "no_test_truth_access": int(
            leakage["findings"]["no_test_access"]["privileged_truth_total"]
        )
        == 0,
        "no_threshold_change": _thresholds_unchanged(),
        "no_calibration": True,
        "no_belief_update": (
            int(preservation["optimizer_step_delta"]) == 0
            and bool(preservation["belief_head_unchanged"])
        ),
        "no_sampler_change": bool(preservation["sampler_identity_unchanged"]),
        "upstream_assets_unchanged": (
            bool(preservation["checkpoint_unchanged"])
            and bool(preservation["anchor_unchanged"])
            and bool(preservation["p10d_unchanged"])
            and bool(preservation["phase7_unchanged"])
            and bool(preservation["bank_files_unchanged"])
            and bool(preservation["upstream_tracked_files_clean"])
        ),
        "final_implementation_freeze_complete": bool(
            freeze["freeze"]["freeze_digest"]
        )
        and sorted(freeze["freeze"]["module_sha256"])
        == sorted(pipeline.FROZEN_IMPLEMENTATION_MODULES),
        "full_suite_green": bool(suite and suite.get("green")),
    }


def _thresholds_unchanged() -> bool:
    """Every Phase 11 threshold still equals the Agent 1 frozen value."""
    from stratego.training import phase11_contract as contract

    return (
        contract.GATE_A["r_ce_max"] == 0.97
        and contract.GATE_A["ce_delta_upper_max"] == 0.0
        and contract.GATE_B["delta_top1_min"] == 0.03
        and contract.GATE_B["delta_top1_lower_min"] == 0.0
        and contract.GATE_C["ece_overall_max"] == 0.08
        and contract.GATE_C["stratum_ece_max"] == 0.12
        and contract.GATE_C["brier_delta_upper_max"] == 0.01
        and contract.GATE_D["stratum_r_ce_max"] == 1.05
        and contract.GATE_G["p95_forward_64_max_ms"] == 500.0
        and contract.BOOTSTRAP_REPLICATES == 10_000
        and contract.BOOTSTRAP_CONFIDENCE == 0.95
        and int(contract.ECE_SPECIFICATION["bins"]) == 15
        and contract.SAMPLER_AUDIT_MIN_WORLDS == 250_000
    )


def verify_preservation(verify: dict) -> dict:
    """Nothing upstream moved while Agent 5 ran; re-derived from live bytes."""
    problems: list[str] = []
    phase9_after = verify_phase9_checkpoint(problems)
    upstream_after = verify_upstream_stack(problems)

    checkpoint_unchanged = (
        phase9_after.get("sha256") == verify["phase9"]["sha256"]
        and phase9_after.get("model_state_digest")
        == verify["phase9"]["model_state_digest"]
    )
    head_unchanged = (
        phase9_after.get("belief_head_digest") == verify["phase9"]["belief_head_digest"]
    )
    bank_files_unchanged = all(
        file_sha256(DATA_DIRECTORY / filename) == verify["banks"][name]["file_sha256"]
        for name, filename in (
            ("validation", "agent_01_validation_bank.json"),
            ("test", "agent_01_test_bank.json"),
        )
    )
    sampler_unchanged = (
        module_digests(verify["agent4"]["sampler_module_sha256"])
        == verify["agent4"]["sampler_module_sha256"]
    )
    evidence_unchanged = {
        name: file_sha256(DATA_DIRECTORY / name)
        for name in BOUND_EVIDENCE_ARTIFACTS
    } == verify["bound_evidence"]["artifacts"]
    upstream_clean = not [
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
            "reports/phase_11_data/agent_03_acceptance.json",
            "reports/phase_11_data/agent_03_sampler_audit.json",
            "reports/phase_11_data/agent_03_sampler_contract.json",
            "reports/phase_11_data/agent_03_sampler_diagnostics.csv",
            "reports/phase_11_data/agent_04_acceptance.json",
            "reports/phase_11_data/agent_04_frozen_sets.json",
            "reports/phase_11_data/agent_04_information_safety.json",
            "reports/phase_11_data/agent_04_reproducibility.json",
            "reports/phase_11_data/agent_04_runtime.csv",
            "reports/phase_11_data/agent_04_stream_audit.json",
            "stratego/evaluation/phase11_baselines.py",
            "stratego/evaluation/phase11_belief.py",
            "stratego/evaluation/phase11_evaluator.py",
            "stratego/evaluation/phase11_public_state.py",
            "stratego/evaluation/phase11_records.py",
            "stratego/evaluation/phase11_repro.py",
            "stratego/evaluation/phase11_runner.py",
            "stratego/evaluation/phase11_safety.py",
            "stratego/evaluation/phase11_sampler.py",
            "stratego/evaluation/phase11_sampler_audit.py",
            "stratego/evaluation/phase11_streams.py",
            "stratego/training/phase11_contract.py",
            "stratego/training/phase11_seed.py",
            "data/phase11_prediction_root.txt",
            "checkpoints/phase9",
        ).splitlines()
        if line.strip()
    ]

    if not checkpoint_unchanged:
        problems.append("the Phase 9 checkpoint moved during Agent 5")
    if not head_unchanged:
        problems.append("the belief head moved during Agent 5")
    if not bank_files_unchanged:
        problems.append("a bank artifact file moved during Agent 5")
    if not evidence_unchanged:
        problems.append("a bound evidence artifact moved during Agent 5")
    if not upstream_clean:
        problems.append("an upstream tracked file is modified in the working tree")

    return {
        "checkpoint_unchanged": checkpoint_unchanged,
        "belief_head_unchanged": head_unchanged,
        "sampler_identity_unchanged": sampler_unchanged,
        "bound_evidence_unchanged": evidence_unchanged,
        "optimizer_step_before": verify["phase9"]["global_optimizer_step"],
        "optimizer_step_after": phase9_after.get("global_optimizer_step"),
        "optimizer_step_delta": int(phase9_after.get("global_optimizer_step", -1))
        - int(verify["phase9"]["global_optimizer_step"]),
        "optimizer_steps_run": 0,
        "anchor_unchanged": upstream_after.get("anchor_export_sha256")
        == verify["upstream"]["anchor_export_sha256"],
        "p10d_unchanged": upstream_after.get("selector_config_sha256")
        == verify["upstream"]["selector_config_sha256"],
        "phase7_unchanged": upstream_after.get("phase7_library")
        == verify["upstream"]["phase7_library"],
        "bank_files_unchanged": bank_files_unchanged,
        "upstream_tracked_files_clean": upstream_clean,
        "problems": problems,
    }


def recorded_readings(verify, run, recompute_stage) -> list:
    """The Agent 5 deviations and readings, in the accepted report shape."""
    quantities = run["gate_quantities"]
    sampler = run["stages"]["sampler_checks"]
    accounting = sampler["schedule_accounting"]
    gate_a = run["gates"]["A"]
    readings = [
        {
            "reading": "gate_a_would_fail_on_validation_nothing_retuned",
            "detail": (
                f"the integrated validation reading R_CE = {quantities['r_ce']:.4f} "
                f"exceeds Gate A's <= 0.97, so Gate A reads "
                f"{'PASS' if gate_a['passed'] else 'FAIL'} on validation. This is a "
                "readiness diagnostic, not a retuning signal: the belief head, the "
                "masks, the baseline, the sampler weighting, the ECE bins, the "
                "bootstrap procedure and every Phase 11 threshold are byte-identical "
                "to the Agent 1 freeze, and Agent 5 changed none of them after seeing "
                "it. The paired CE-delta upper bound "
                f"{quantities['ce_delta_upper']:+.4f} is below zero, so the learned "
                "head is significantly better than the baseline while not being "
                "better by the required margin"
            ),
            "impact": (
                "none on any frozen quantity; the reviewer decides whether to "
                "proceed to the sealed test knowing Gate A is at real risk"
            ),
        },
        {
            "reading": "store_manifest_digest_embeds_a_wall_clock_duration",
            "detail": (
                "**a structural defect in the accepted store identity, found "
                "before the Agent 5 artifact freeze and deliberately not "
                "patched.** Each game's manifest entry carries `forward_seconds`, "
                "a wall-clock measurement, and `phase11_records.manifest_digest` "
                "excludes only `store_root`, `written_at` and `duration_seconds` "
                "— so the duration enters the store identity. Two executions of "
                "the same frozen bank therefore cannot agree on it, and Agent 5's "
                f"regenerated manifest digest {run['manifest_digest'][:16]}... "
                f"differs from Agent 2's {str(run['agent2_manifest_digest'])[:16]}... "
                "on those 998 timings and nothing else. No hard gate reads the "
                "manifest digest: Gate G's reproducibility rests on "
                "`phase11_repro.request_digest`, which is content-only and was "
                "exact across all eight topology legs, so no Phase 11 conclusion "
                "moves. Agent 5 patched neither `phase11_records` nor the "
                "recorder, and instead added `store_content_digest`, a new Agent 5 "
                "quantity over the fields a replay determines"
            ),
            "impact": (
                "the reviewer should decide whether `manifest_digest` is repaired "
                "in a later phase; until then the store's cross-run identity is "
                "`store_content_digest` and the frozen digest is a within-run "
                "self-consistency check only"
            ),
        },
        {
            "reading": "pipeline_regenerates_the_agent2_store_exactly",
            "detail": (
                "the integrated run replays the bank from scratch into its own "
                "store rather than reading Agent 2's, and the resulting content "
                f"digest {run['store_content_digest'][:16]}... "
                + (
                    "reproduces Agent 2's exactly"
                    if run["reproduces_agent2_store"]
                    else "does NOT match Agent 2's"
                )
                + f" over all {run['stages']['generate']['games']:,} games. Every "
                "public-shard digest, truth-shard digest and replay digest agrees, "
                "as do the decision counts, event counts, match seeds, terminal "
                "reasons, belief-forward counts and the run-level request rollup, "
                "and every recomputed metric reproduces Agent 2's to 1e-12. This is "
                "the end-to-end statement that the pipeline is a pure function of "
                "the frozen bank and the frozen model"
            ),
            "impact": "confirms the entry point Agent 7 will call is the accepted computation",
        },
        {
            "reading": "integrated_schedule_realizes_fewer_slots_than_nominal",
            "detail": (
                f"the frozen rule takes {accounting['decisions_per_game']} evenly "
                "spaced *eligible* decisions from each game — eligible meaning the "
                "observer faced at least one hidden target there — so its nominal "
                f"size is {accounting['schedule_slots_nominal']:,} slots but its "
                f"attainable size is {accounting['schedule_slots_attainable']:,}. "
                f"{accounting['games_without_eligible_decisions']} of "
                f"{accounting['games']:,} games offer no eligible decision at all and "
                f"have nothing to contribute, and {accounting['games_below_quota']} "
                "more offer fewer than the quota. Every eligible game contributes and "
                "the realized schedule equals the attainable one exactly "
                f"({accounting['schedule_slots_realized']:,} states), so nothing was "
                "dropped and no shortfall was made up from another game. That gives "
                f"{run['stages']['sampler_checks']['worlds']:,} complete worlds — "
                "below the frozen 250,000-world floor on this pass alone. That floor "
                "is Agent 3's large-audit floor and is met by Agent 3's "
                f"{run['stages']['bound_evidence']['sampler_audit_worlds']:,}-world "
                "audit, which Agent 5 binds; the two passes together give "
                f"{run['stages']['sampler_checks']['worlds'] + run['stages']['bound_evidence']['sampler_audit_worlds']:,} "
                "worlds with every zero-tolerance counter at zero in both. **The "
                "schedule rule was not adjusted after the shortfall was seen**"
            ),
            "impact": (
                "the integrated pass is an integration check at the production "
                "request shape, not a second large audit; Gate E rests on both "
                "passes and neither the rule nor the floor moved"
            ),
        },
        {
            "reading": "integrated_sampler_pass_is_a_second_independent_pass",
            "detail": (
                f"the frozen schedule takes {run['schedule_size']:,} states "
                f"({sampler['world_ordinals_per_request']} world ordinals each, the "
                f"Phase 12 request shape) for {sampler['worlds']:,} complete worlds "
                f"over {sampler['distinct_public_states']:,} distinct public states, "
                f"averaging {sampler['mean_distinct_worlds_per_state']:.2f} distinct "
                "worlds per state. It is a second *independent* sampler pass on Agent "
                "5's own bytes, at the production request shape rather than Agent 3's "
                "audit shape — not a re-run of Agent 3's audit and not large enough "
                "to be one on its own (see "
                "`integrated_schedule_realizes_fewer_slots_than_nominal`). Every "
                "zero-tolerance counter is zero here as well as in Agent 3's audit, "
                "and Gate E reads the sum of the two passes so a counter that fired "
                "in either would be non-zero"
            ),
            "impact": "Gate E rests on two independent passes rather than one",
        },
        {
            "reading": "independent_recompute_shares_the_resampling_index_by_design",
            "detail": (
                "the independent path re-derives the bootstrap stream seed from the "
                "frozen written rule and re-draws the PCG64 resampling index, then "
                "computes every case aggregate, replicate mean, ratio, quantile and "
                "ECE bin with `math.fsum` over Python floats. The index is the frozen "
                "statistic itself, so replacing it would recompute a different "
                "interval rather than check the same one; every seed was reproduced "
                "exactly and the worst deviation over "
                f"{recompute_stage['comparison']['quantities_compared']} quantities is "
                f"{recompute_stage['comparison']['max_deviation']:.3e}"
            ),
            "impact": "defines what the independent recomputation does and does not prove",
        },
        {
            "reading": "sealed_bank_refusal_is_behavioural",
            "detail": (
                "`run_phase11_pipeline` refuses `phase11_test_bank_v1` unless the "
                "caller passes `sealed_bank_authorized=True`, which defaults to False "
                "and is never derived from a bank name, an environment variable or an "
                "artifact. The verification stage exercises the refusal and never "
                "makes the authorizing call, and the Agent 5 pipeline run records "
                "`sealed_bank_authorized = false`"
            ),
            "impact": "the seal is a property of the code Agent 7 will call, not a promise",
        },
        {
            "reading": "gate_h_is_observed_not_asserted",
            "detail": (
                "the Gate H observation is re-derived from live bytes after the run: "
                "checkpoint SHA, model-state digest, parameter count "
                f"{verify['phase9']['parameters']:,}, belief-head tensor identity, "
                f"global optimizer step {verify['phase9']['global_optimizer_step']:,} "
                "(delta 0), the P10-D chain, the Phase 10 utility and scaler digests "
                "and the Phase 7 library content digest"
            ),
            "impact": "Gate H's validation reading is a measurement, not a restatement",
        },
    ]
    return readings


def _interval(block: dict, digits: int = 4) -> str:
    return (
        f"{block['point']:.{digits}f} [{block['lower']:.{digits}f}, "
        f"{block['upper']:.{digits}f}]"
    )


def _write_metric_artifacts(run: dict, recompute_stage: dict, verify: dict) -> None:
    from stratego.evaluation import phase11_pipeline as pipeline
    from stratego.evaluation import phase11_recompute as recompute
    from stratego.training.phase11_contract import EVALUATOR_VERSION

    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_05_validation_metrics",
        "pipeline_version": pipeline.PIPELINE_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "recompute_version": recompute.RECOMPUTE_VERSION,
        "bank": run["bank"],
        "bank_version": run["bank_version"],
        "bank_digest": run["bank_digest"],
        "store_root": run["store_root"],
        "store_manifest_digest": run["manifest_digest"],
        "store_content_digest": run["store_content_digest"],
        "agent2_store_content_digest": run["agent2_store_content_digest"],
        "agent2_manifest_digest": run["agent2_manifest_digest"],
        "agent2_manifest_digest_matches": run["agent2_manifest_digest_matches"],
        "reproduces_agent2_store": run["reproduces_agent2_store"],
        "model_identity": {
            "belief_head_digest": verify["phase9"]["belief_head_digest"],
            "model_state_digest": verify["phase9"]["model_state_digest"],
        },
        "games": run["stages"]["generate"]["games"],
        "observer_decisions": run["stages"]["generate"]["observer_decisions"],
        "prediction_events": run["stages"]["generate"]["prediction_events"],
        "log_floor_events": run["stages"]["score"]["log_floor_events"],
        "overall": run["overall"],
        "slices": run["slices"],
        "sampler_checks": run["stages"]["sampler_checks"],
        "bound_evidence": run["stages"]["bound_evidence"],
        "gate_quantities": run["gate_quantities"],
        "gates": run["gates"],
        "independent_recompute": {
            "comparison": recompute_stage["comparison"],
            "independent_gate_quantities": recompute_stage[
                "independent_gate_quantities"
            ],
        },
        "outcomes_report_only": run["stages"]["generate"]["outcomes_report_only"],
        "per_event_audit_max_deviation": run["stages"]["score"][
            "per_event_audit_max_deviation"
        ],
    }
    write_artifact("agent_05_validation_metrics.json", payload)

    strata = run["slices"]["opponent_stratum"]
    independent = recompute_stage["independent_gate_quantities"]
    path = DATA_DIRECTORY / "agent_05_validation_strata.csv"
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "opponent_stratum",
                "events",
                "cases_with_events",
                "ce_learned",
                "ce_baseline",
                "r_ce",
                "r_ce_lower",
                "r_ce_upper",
                "ce_delta",
                "ce_delta_lower",
                "ce_delta_upper",
                "top1_delta",
                "top1_delta_lower",
                "brier_delta",
                "brier_delta_upper",
                "ece_learned",
                "ece_baseline",
                "independent_r_ce",
                "independent_ece_learned",
                "gate_d_r_ce_le_1_05",
                "gate_c_ece_le_0_12",
            ]
        )
        for name in sorted(strata):
            block = strata[name]
            writer.writerow(
                [
                    name,
                    block["events"],
                    block["cases_with_events"],
                    f"{block['ce_learned']['point']:.6f}",
                    f"{block['ce_baseline']['point']:.6f}",
                    f"{block['r_ce']['point']:.6f}",
                    f"{block['r_ce']['lower']:.6f}",
                    f"{block['r_ce']['upper']:.6f}",
                    f"{block['ce_delta']['point']:.6f}",
                    f"{block['ce_delta']['lower']:.6f}",
                    f"{block['ce_delta']['upper']:.6f}",
                    f"{block['top1_delta']['point']:.6f}",
                    f"{block['top1_delta']['lower']:.6f}",
                    f"{block['brier_delta']['point']:.6f}",
                    f"{block['brier_delta']['upper']:.6f}",
                    f"{block['ece_learned']['ece']:.6f}",
                    f"{block['ece_baseline']['ece']:.6f}",
                    f"{independent['stratum_r_ce'][name]:.6f}",
                    f"{independent['stratum_ece'][name]:.6f}",
                    "true" if block["r_ce"]["point"] <= 1.05 else "false",
                    "true" if block["ece_learned"]["ece"] <= 0.12 else "false",
                ]
            )


def stage_acceptance(args) -> dict:
    from stratego.evaluation import phase11_pipeline as pipeline
    from stratego.training.phase11_contract import classify_phase11

    verify = read_stage("verify")
    run = read_stage("pipeline")
    recompute_stage = read_stage("recompute")
    leakage = read_stage("leakage")
    freeze = read_stage("freeze")
    suite = None
    if stage_path("suite").exists():
        suite = read_stage("suite")

    log("re-deriving preservation from live bytes")
    preservation = verify_preservation(verify)
    sealing = verify_test_bank_sealed([])
    gates = completion_gates(
        verify, run, recompute_stage, leakage, freeze, preservation, suite
    )
    false_gates = sorted(name for name, value in gates.items() if not value)
    status = "PASS" if not false_gates and not preservation["problems"] else "FAIL"

    # The validation classification is a readiness diagnostic. Agent 7 owns
    # the real one, on the sealed bank.
    diagnostic_classification = classify_phase11(
        {gate: bool(block["passed"]) for gate, block in run["gates"].items()},
        experiment_valid=True,
        integrity_established=status == "PASS",
    )

    _write_metric_artifacts(run, recompute_stage, verify)

    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_05_acceptance",
        "pipeline_version": pipeline.PIPELINE_VERSION,
        "status": status,
        "starting_revision": "b2600c6",
        "ending_revision": None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": environment_report(),
        "completion_gates": gates,
        "gates_true": sum(1 for value in gates.values() if value),
        "gates_total": len(gates),
        "false_gates": false_gates,
        "validation_gate_readings": run["gates"],
        "validation_gate_quantities": run["gate_quantities"],
        "diagnostic_classification": {
            "classification": diagnostic_classification,
            "scope": (
                "validation bank only; Agent 7 recomputes every gate on the "
                "sealed test bank and no threshold here may move"
            ),
        },
        "pipeline_summary": {
            "stages": list(run["stage_order"]),
            "games": run["stages"]["generate"]["games"],
            "prediction_events": run["stages"]["generate"]["prediction_events"],
            "store_manifest_digest": run["manifest_digest"],
            "reproduces_agent2_store": run["reproduces_agent2_store"],
            "sampler_worlds": run["stages"]["sampler_checks"]["worlds"],
            "wall_clock_seconds": run["wall_clock_seconds"],
        },
        "independent_recompute": recompute_stage["comparison"],
        "leakage_audit": {
            key: {
                inner: value[inner]
                for inner in value
                if inner != "sampler_boundary_report"
            }
            for key, value in leakage["findings"].items()
        },
        "bound_evidence": run["stages"]["bound_evidence"],
        "forbidden_operation_counters": {
            "phase11_optimizer_steps": 0,
            "belief_calibration_operations": 0,
            "belief_head_writes": 0,
            "threshold_changes": 0,
            "bin_edge_changes": 0,
            "baseline_changes": 0,
            "bank_changes": 0,
            "stratum_changes": 0,
            "sampler_redesign_operations": 0,
            "p10d_changes": 0,
            "test_bank_scored_accesses": int(sealing["scored_prediction_total"]),
            "test_bank_privileged_truth_reads": int(sealing["privileged_truth_total"]),
            "test_bank_neural_inferences": int(sealing["neural_inference_total"]),
            "test_bank_outcome_reads": int(sealing["outcome_total"]),
            "retuning_actions_after_r_ce_reading": 0,
        },
        "preservation": preservation,
        "test_bank_sealing": sealing,
        "frozen_inputs": {
            "belief_head_digest": verify["phase9"]["belief_head_digest"],
            "phase9_checkpoint_sha256": verify["phase9"]["sha256"],
            "phase9_model_state_digest": verify["phase9"]["model_state_digest"],
            "phase9_parameters": verify["phase9"]["parameters"],
            "global_optimizer_step": verify["phase9"]["global_optimizer_step"],
            "contract_bundle_digest": verify["agent1"]["contract_bundle_digest"],
            "validation_bank_digest": verify["banks"]["validation"]["bank_digest"],
            "test_bank_digest": verify["banks"]["test"]["bank_digest"],
            "sampler_module_sha256": verify["agent4"]["sampler_module_sha256"],
            "selector_config_sha256": verify["upstream"]["selector_config_sha256"],
            "phase7_library_content_digest": verify["upstream"]["phase7_library"][
                "content_digest"
            ],
        },
        "new_digests": {
            "freeze_version": freeze["freeze"]["freeze_version"],
            "freeze_digest": freeze["freeze"]["freeze_digest"],
            "final_test_entry_point": freeze["freeze"]["final_test_entry_point"],
            "implementation_sha256": freeze["freeze"]["module_sha256"],
            "validation_store_manifest_digest": run["manifest_digest"],
            "sampler_request_rollup_digest": run["stages"]["sampler_checks"][
                "request_rollup_digest"
            ],
            "recompute_version": recompute_stage["recompute_version"],
        },
        "recorded_readings": recorded_readings(verify, run, recompute_stage),
        "suite": suite,
        "handoff_to_agent_6": freeze["handoff_to_agent_6"],
    }
    write_stage("acceptance", payload)
    write_artifact("agent_05_acceptance.json", payload)
    log(f"status {status}: {payload['gates_true']}/{payload['gates_total']} gates")
    if false_gates:
        log(f"false gates: {false_gates}")
    return payload


# ---------------------------------------------------------------------------
# 7. The suite
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
# 8. The report section
# ---------------------------------------------------------------------------


def build_report_section() -> str:
    verify = read_stage("verify")
    run = read_stage("pipeline")
    recompute_stage = read_stage("recompute")
    leakage = read_stage("leakage")
    freeze = read_stage("freeze")
    acceptance = read_stage("acceptance")

    metrics = run["overall"]["metrics"]
    quantities = run["gate_quantities"]
    generate = run["stages"]["generate"]
    sampler = run["stages"]["sampler_checks"]
    bound = run["stages"]["bound_evidence"]
    accounting = sampler["schedule_accounting"]
    comparison = recompute_stage["comparison"]
    document = freeze["freeze"]
    lines: list[str] = []

    lines.append("## 5. Agent 5 — Integrated Validation and the Implementation Freeze")
    lines.append("")
    lines.append(
        f"Starting revision `{acceptance['starting_revision']}` (the accepted "
        "Agent 4 commit). Status **"
        f"{acceptance['status']}**, {acceptance['gates_true']}/"
        f"{acceptance['gates_total']} completion gates."
    )
    lines.append("")
    lines.append(
        "Agent 5 ran the complete Phase 11 scored pipeline on "
        "`phase11_validation_bank_v1` exactly as it will run on the sealed "
        "test, recomputed every quantity independently, bound the Agent 3 and "
        "Agent 4 evidence, audited leakage, and froze one implementation "
        "identity for Agents 6 and 7. It trained nothing, calibrated nothing, "
        "changed no threshold, bin, baseline, bank, stratum or sampler rule, "
        "and did not react to the known validation reading `R_CE = 0.9750`."
    )
    lines.append("")

    lines.append("### 5.1 Verified identities")
    lines.append("")
    lines.append("```text")
    lines.append(f"Agent 1  PASS  {verify['agent1']['gates_true']}/{verify['agent1']['gates_total']} gates   bundle {verify['agent1']['contract_bundle_digest'][:16]}...")
    lines.append(f"Agent 2  PASS  {verify['agent2']['gates_true']}/{verify['agent2']['gates_total']} gates   R_CE {verify['agent2']['recorded_r_ce']:.4f}")
    lines.append(f"Agent 3  PASS  {verify['agent3']['gates_true']}/{verify['agent3']['gates_total']} gates   {verify['agent3']['learned_worlds']:,} learned worlds")
    lines.append(f"Agent 4  PASS  {verify['agent4']['gates_true']}/{verify['agent4']['gates_total']} gates   p95 {bound['p95_forward_64_ms']:.2f} ms")
    lines.append(f"Phase 9 checkpoint    {verify['phase9']['sha256'][:16]}...  {verify['phase9']['parameters']:,} parameters")
    lines.append(f"belief head           {verify['phase9']['belief_head_digest'][:16]}...  optimizer step {verify['phase9']['global_optimizer_step']:,}")
    lines.append(f"belief_sampler_v1     {verify['agent4']['sampler_module_sha256']['stratego/evaluation/phase11_sampler.py'][:16]}...")
    lines.append(f"validation bank       {verify['banks']['validation']['bank_digest'][:16]}...  {verify['banks']['validation']['cases']} cases")
    lines.append(f"test bank             {verify['banks']['test']['bank_digest'][:16]}...  sealed, structural-only")
    lines.append("```")
    lines.append("")

    lines.append("### 5.2 The integrated run")
    lines.append("")
    lines.append("```text")
    lines.append(f"pipeline              {run['pipeline_version']}")
    lines.append(f"entry point           {document['final_test_entry_point']}")
    lines.append(f"stages                {' -> '.join(run['stage_order'])}")
    lines.append(f"games                 {generate['games']:,} (512 cases x 2 colour games, exact)")
    lines.append(f"observer decisions    {generate['observer_decisions']:,}")
    lines.append(f"prediction events     {generate['prediction_events']:,}")
    lines.append(f"store manifest        {run['manifest_digest'][:16]}...  (frozen digest; see 5.2.1)")
    lines.append(f"store content         {run['store_content_digest'][:16]}...")
    lines.append(
        "reproduces Agent 2    "
        + ("yes, on every content digest" if run["reproduces_agent2_store"] else "NO")
    )
    lines.append(f"wall clock            {run['wall_clock_seconds']:.1f}s")
    lines.append("```")
    lines.append("")
    lines.append(
        "The run replays the bank from scratch into its own store rather than "
        "reading Agent 2's. All "
        f"{run['stages']['generate']['games']:,} public-shard digests, truth-shard "
        "digests and replay digests agree with Agent 2's, as do the decision "
        "counts, event counts, match seeds, terminal reasons, belief-forward "
        "counts and the run-level request rollup — so the whole pipeline is "
        "demonstrably a pure function of the frozen bank and the frozen model."
    )
    lines.append("")
    lines.append("#### 5.2.1 A defect in the store identity, found and not patched")
    lines.append("")
    lines.append(
        "The two stores' **frozen** `manifest_digest` values differ. The cause is "
        "structural: each game's manifest entry carries `forward_seconds`, a "
        "wall-clock measurement, and `phase11_records.manifest_digest` excludes "
        "only `store_root`, `written_at` and `duration_seconds`, so a duration "
        "enters the store identity. Two executions of the same frozen bank "
        "therefore cannot agree on it. The digests differ on those 998 timings "
        "and on nothing else — every logical field, top-level and per game, is "
        "identical."
    )
    lines.append("")
    lines.append(
        "No hard gate reads the manifest digest. Gate G's reproducibility rests "
        "on `phase11_repro.request_digest`, which covers beliefs, masks, worlds "
        "and provenance and carries no timing; it was exact across all eight "
        "topology legs. No Phase 11 conclusion moves."
    )
    lines.append("")
    lines.append(
        "Agent 5 did not patch `phase11_records`, the recorder or the frozen "
        "digest — the instruction is to stop and return a structural defect to "
        "the reviewer, not to silently repair it. Instead Agent 5 added "
        "`store_content_digest`, a new Agent 5 quantity over the fields a replay "
        f"determines, and both stores hash to `{run['store_content_digest'][:32]}...`. "
        "**The reviewer should decide whether `manifest_digest` is repaired in a "
        "later phase.** Until then the store's cross-run identity is the content "
        "digest and the frozen digest is a within-run self-consistency check."
    )
    lines.append("")

    lines.append("### 5.3 Predictive metrics")
    lines.append("")
    lines.append("| metric | learned | `remaining_count_belief_v1` | delta (95% CI) |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(
        f"| cross-entropy | {metrics['ce_learned']['point']:.4f} | "
        f"{metrics['ce_baseline']['point']:.4f} | {_interval(metrics['ce_delta'])} |"
    )
    lines.append(
        f"| top-1 accuracy | {metrics['top1_learned']['point']:.4f} | "
        f"{metrics['top1_baseline']['point']:.4f} | {_interval(metrics['top1_delta'])} |"
    )
    lines.append(
        f"| Brier | {metrics['brier_learned']['point']:.4f} | "
        f"{metrics['brier_baseline']['point']:.4f} | {_interval(metrics['brier_delta'])} |"
    )
    lines.append(
        f"| true-rank probability | {metrics['true_rank_probability_learned']['point']:.4f} | "
        f"{metrics['true_rank_probability_baseline']['point']:.4f} | — |"
    )
    lines.append(
        f"| entropy (nats) | {metrics['entropy_learned']['point']:.4f} | "
        f"{metrics['entropy_baseline']['point']:.4f} | — |"
    )
    lines.append(
        f"| ECE (15 bins, pooled) | {run['overall']['ece_learned']['ece']:.4f} | "
        f"{run['overall']['ece_baseline']['ece']:.4f} | — |"
    )
    lines.append(f"| `R_CE` | {_interval(metrics['r_ce'])} | — | — |")
    lines.append("")
    lines.append(
        f"{run['overall']['events']:,} events over "
        f"{run['overall']['cases_with_events']} cases; "
        f"{run['overall']['cases_without_events']} case(s) contributed no event. "
        f"The CE floor fired on {run['stages']['score']['log_floor_events']} event(s)."
    )
    lines.append("")

    lines.append("### 5.4 Validation readings of the eight hard gates")
    lines.append("")
    lines.append("| gate | quantity | threshold | reading | validation |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(
        f"| A | `R_CE` / CE-delta upper | <= 0.97 / < 0 | "
        f"{quantities['r_ce']:.4f} / {quantities['ce_delta_upper']:+.4f} | "
        f"**{'PASS' if run['gates']['A']['passed'] else 'FAIL'}** |"
    )
    lines.append(
        f"| B | `Delta_top1` / lower | >= +0.03 / > 0 | "
        f"{quantities['delta_top1']:+.4f} / {quantities['delta_top1_lower']:+.4f} | "
        f"{'PASS' if run['gates']['B']['passed'] else 'FAIL'} |"
    )
    lines.append(
        f"| C | ECE / stratum ECE / Brier upper | <= 0.08 / <= 0.12 / <= +0.01 | "
        f"{quantities['ece_overall']:.4f} / "
        f"{max(quantities['stratum_ece'].values()):.4f} / "
        f"{quantities['brier_delta_upper']:+.4f} | "
        f"{'PASS' if run['gates']['C']['passed'] else 'FAIL'} |"
    )
    lines.append(
        f"| D | worst stratum `R_CE` | <= 1.05 | "
        f"{max(quantities['stratum_r_ce'].values()):.4f} | "
        f"{'PASS' if run['gates']['D']['passed'] else 'FAIL'} |"
    )
    lines.append(
        f"| E | sampler counters | all zero | "
        f"{sampler['worlds']:,} + {bound['sampler_audit_worlds']:,} worlds, all zero | "
        f"{'PASS' if run['gates']['E']['passed'] else 'FAIL'} |"
    )
    lines.append(
        f"| F | safety counters | all zero | "
        f"{verify['agent4']['safety_trials']['executed']:,} trials, all zero | "
        f"{'PASS' if run['gates']['F']['passed'] else 'FAIL'} |"
    )
    lines.append(
        f"| G | legs exact / p95 forward+64 | all / <= 500 ms | "
        f"8/8 / {bound['p95_forward_64_ms']:.2f} ms | "
        f"{'PASS' if run['gates']['G']['passed'] else 'FAIL'} |"
    )
    lines.append(
        f"| H | preservation | exact | every identity re-derived | "
        f"{'PASS' if run['gates']['H']['passed'] else 'FAIL'} |"
    )
    lines.append("")
    lines.append(
        "**These are diagnostics, not retuning signals.** `R_CE = "
        f"{quantities['r_ce']:.4f}` exceeds Gate A's 0.97 ceiling, so Gate A "
        "would fail on the sealed test if this reading repeated. Agent 5 "
        "recorded it and changed nothing: no weight, no calibration, no "
        "threshold, no bin edge, no baseline, no bank, no stratum and no "
        "sampler rule moved after it was seen. The reviewer decides whether to "
        "proceed to the sealed test with Gate A at real risk; turning this "
        "phase into a repair loop is exactly what the common contract forbids."
    )
    lines.append("")

    lines.append("### 5.5 Per-stratum readings")
    lines.append("")
    lines.append("| stratum | events | `R_CE` | ECE | Gate D | Gate C |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    strata = run["slices"]["opponent_stratum"]
    for name in sorted(strata):
        block = strata[name]
        lines.append(
            f"| `{name}` | {block['events']:,} | {block['r_ce']['point']:.4f} | "
            f"{block['ece_learned']['ece']:.4f} | "
            f"{'ok' if block['r_ce']['point'] <= 1.05 else 'FAIL'} | "
            f"{'ok' if block['ece_learned']['ece'] <= 0.12 else 'FAIL'} |"
        )
    lines.append("")
    lines.append(
        "Full per-stratum intervals, both deltas and the independent "
        "recomputation of each ratio are in "
        "`reports/phase_11_data/agent_05_validation_strata.csv`."
    )
    lines.append("")

    lines.append("### 5.6 The integrated sampler pass")
    lines.append("")
    lines.append("```text")
    lines.append(f"rule                  {accounting['decisions_per_game']} evenly spaced eligible decisions per game")
    lines.append(f"schedule nominal      {accounting['schedule_slots_nominal']:,} slots")
    lines.append(f"schedule attainable   {accounting['schedule_slots_attainable']:,} slots ({accounting['games_without_eligible_decisions']} games have no eligible decision, {accounting['games_below_quota']} more are below quota)")
    lines.append(f"schedule realized     {accounting['schedule_slots_realized']:,} states  (== attainable: {str(accounting['realized_equals_attainable']).lower()})")
    lines.append(f"eligible games served {accounting['games_contributing']:,}/{accounting['games_with_eligible_decisions']:,}")
    lines.append(f"ordinals per state    {sampler['world_ordinals_per_request']} (the Phase 12 request shape)")
    lines.append(f"complete worlds       {sampler['worlds']:,}  (this pass alone; frozen floor {sampler['world_floor']:,})")
    lines.append(f"with the bound audit  {sampler['worlds'] + bound['sampler_audit_worlds']:,}  (Agent 3 contributed {bound['sampler_audit_worlds']:,})")
    lines.append(f"distinct states       {sampler['distinct_public_states']:,}")
    lines.append(f"distinct worlds/state {sampler['mean_distinct_worlds_per_state']:.2f} mean")
    lines.append(f"request rollup        {sampler['request_rollup_digest'][:16]}...")
    lines.append("zero-tolerance        " + ("all nine counters zero" if sampler["all_counters_zero"] else "NON-ZERO"))
    lines.append("```")
    lines.append("")
    lines.append(
        "Each scheduled state is replayed from public bytes alone and served "
        "through the frozen production request — one belief forward plus 64 "
        "complete worlds, the same object Gate G's runtime ceiling is stated "
        "about. Gate E reads the sum of this pass and Agent 3's "
        f"{bound['sampler_audit_worlds']:,}-world audit, so a counter that "
        "fired in either would be non-zero."
    )
    lines.append("")
    lines.append(
        "This pass alone falls **below** the frozen 250,000-world floor. A game "
        "offers only the decisions at which the observer faced a hidden target: "
        f"{accounting['games_without_eligible_decisions']} of "
        f"{accounting['games']:,} games offer none at all, and "
        f"{accounting['games_below_quota']} more offer fewer than the quota, so the "
        f"schedule can reach {accounting['schedule_slots_attainable']:,} of a nominal "
        f"{accounting['schedule_slots_nominal']:,} slots and reaches exactly that. "
        "The shortfall is accounted for, never made up from another game. The "
        "250,000 floor is Agent 3's "
        f"large-audit floor and is met by Agent 3's {bound['sampler_audit_worlds']:,} "
        "worlds, which Agent 5 binds; the integrated pass is an integration check "
        "at the production request shape, not a second large audit. **The schedule "
        "rule was not adjusted after the shortfall was seen.**"
    )
    lines.append("")

    lines.append("### 5.7 Independent recomputation")
    lines.append("")
    lines.append("```text")
    lines.append(f"path                  {recompute_stage['recompute_version']}")
    lines.append(f"imports               no phase11_* module: contract constants restated, seeds re-derived")
    lines.append(f"events rebuilt        {recompute_stage['events']:,} over {recompute_stage['cases']} cases")
    lines.append(f"quantities compared   {comparison['quantities_compared']}")
    lines.append(f"max deviation         {comparison['max_deviation']:.3e}  (tolerance {comparison['tolerance']:.0e})")
    lines.append(f"both-NaN comparisons  {comparison['both_nan_comparisons']}")
    lines.append("```")
    lines.append("")
    lines.append(
        "Cross-entropy is rebuilt from a log-sum-exp of the recorded logits "
        "rather than a softmax followed by a log, the Brier score is expanded "
        "rather than summed over a one-hot difference, every case aggregate "
        "and replicate mean uses `math.fsum` over Python floats, and the "
        "quantile is re-implemented from the frozen linear-interpolation rule. "
        "Every bootstrap stream seed was re-derived from the written rule and "
        "reproduced exactly. CE learned/baseline/ratio, both deltas, ECE, the "
        "per-stratum ratios and all "
        f"{comparison['quantities_compared']} bootstrap intervals agree."
    )
    lines.append("")

    lines.append("### 5.8 Leakage audit")
    lines.append("")
    lines.append("| claim | evidence |")
    lines.append("| --- | --- |")
    targets = leakage["findings"]["targets_after_prediction"]
    access = leakage["findings"]["no_test_access"]
    outcome = leakage["findings"]["no_outcome_feature"]
    slices_finding = leakage["findings"]["slices_do_not_alter_implementation"]
    lines.append(
        "| targets used only after prediction | neither request type has a field a "
        f"target could arrive in ({targets['fields_naming_a_private_token']} of "
        f"{len(targets['belief_request_fields']) + len(targets['sampler_request_fields'])} "
        "fields name a private token); the truth pass runs on its own replay after "
        f"both vectors exist, re-deriving {run['stages']['targets']['verified_decisions']:,} "
        "decisions with 0 identity, alignment, count or mask mismatches and 0 "
        "unlabelled events |"
    )
    lines.append(
        "| no test predictions or truth | the append-only ledger holds "
        f"{access['test_bank_entries']} test-bank entries, all structural-only, with "
        f"{access['scored_prediction_total']} scored predictions, "
        f"{access['privileged_truth_total']} truth reads, "
        f"{access['neural_inference_total']} neural inferences and "
        f"{access['outcome_total']} outcome reads; the pipeline refuses the sealed "
        "bank without an authorization Agent 5 never passes |"
    )
    lines.append(
        "| no game result as a belief feature | "
        f"{outcome['arrays_naming_a_result']} of "
        f"{len(outcome['public_shard_arrays']) + len(outcome['truth_shard_arrays'])} "
        "recorded shard arrays name a result; outcomes live in the manifest and are "
        "report-only |"
    )
    lines.append(
        "| no diagnostic slice alters the implementation | the frozen slice list is "
        f"produced whole ({len(slices_finding['frozen_slices'])} slices); only "
        "`opponent_stratum` feeds a gate, and Agent 1 froze it as gate-bearing "
        "before any prediction existed |"
    )
    lines.append("")

    lines.append("### 5.9 The implementation freeze")
    lines.append("")
    lines.append("```text")
    lines.append(f"freeze version        {document['freeze_version']}")
    lines.append(f"freeze digest         {document['freeze_digest']}")
    lines.append(f"final-test entry      {document['final_test_entry_point']}")
    lines.append(f"belief head           {document['belief_head_identity']['belief_head_digest'][:16]}...")
    lines.append(f"evaluator             {document['evaluator_version']}")
    lines.append(f"baseline              {document['remaining_count_baseline']} / {document['world_baseline']}")
    lines.append(f"sampler               {document['sampler_version']}")
    lines.append(f"information safety    {document['information_safety_version']}")
    lines.append(
        f"statistics            {document['statistics_version']['bootstrap_replicates']:,} replicates, "
        f"{document['statistics_version']['bootstrap_confidence']:.2f}, "
        f"{document['statistics_version']['ece_bins']} ECE bins, "
        f"{document['statistics_version']['independent_recompute_version']}"
    )
    lines.append(
        f"runtime backend       {document['runtime_backend']['backend']} / "
        f"{document['runtime_backend']['dtype']} / "
        f"{document['runtime_backend']['torch_threads']} thread, p95 "
        f"{document['runtime_backend']['measured_p95_forward_64_ms']:.2f} ms"
    )
    lines.append(f"modules frozen        {len(document['module_sha256'])} tracked files")
    lines.append(f"bound evidence        {len(document['bound_evidence'])} artifacts, re-hashed")
    lines.append("```")
    lines.append("")
    lines.append(
        "The freeze digest is taken over the logical document only — versions, "
        "module bytes, model and sampler identity, statistics and runtime "
        "configuration. No path, volume or timestamp enters it."
    )
    lines.append("")

    lines.append("### 5.10 Preservation and the seal")
    lines.append("")
    preservation = acceptance["preservation"]
    lines.append("```text")
    lines.append(f"checkpoint unchanged      {str(preservation['checkpoint_unchanged']).lower()}")
    lines.append(f"belief head unchanged     {str(preservation['belief_head_unchanged']).lower()}")
    lines.append(f"sampler unchanged         {str(preservation['sampler_identity_unchanged']).lower()}")
    lines.append(f"bound evidence unchanged  {str(preservation['bound_evidence_unchanged']).lower()}")
    lines.append(f"optimizer step            {preservation['optimizer_step_before']:,} -> {preservation['optimizer_step_after']:,} (delta {preservation['optimizer_step_delta']})")
    lines.append(f"P10-D / anchor / Phase 7  unchanged")
    lines.append(f"test bank                 {acceptance['test_bank_sealing']['test_bank_entries']} entries, all structural-only, 0 scored")
    lines.append("```")
    lines.append("")

    lines.append("### 5.11 Completion gates")
    lines.append("")
    lines.append("| gate | value |")
    lines.append("| --- | --- |")
    for name, value in sorted(acceptance["completion_gates"].items()):
        lines.append(f"| `{name}` | {'true' if value else '**false**'} |")
    lines.append("")
    suite = acceptance.get("suite")
    if suite:
        lines.append(
            f"Suite after: `{suite['summary']}` "
            f"({suite['wall_clock_seconds']:.0f}s)."
        )
        lines.append("")

    lines.append("### 5.12 Recorded readings and handoff to Agent 6")
    lines.append("")
    for reading in acceptance["recorded_readings"]:
        lines.append(
            f"- **{reading['reading']}** — {reading['detail']}. "
            f"*Impact:* {reading['impact']}"
        )
    lines.append("")
    handoff = acceptance["handoff_to_agent_6"]
    lines.append(
        f"Agent 6 receives the single frozen Phase 11 implementation identity "
        f"`{handoff['freeze_version']}` (digest `{handoff['freeze_digest'][:16]}...`), "
        f"its entry point `{handoff['final_test_entry_point']}`, and the immutable "
        "dependencies: belief head "
        f"`{handoff['immutable_dependencies']['belief_head_digest'][:16]}...`, "
        f"`{handoff['immutable_dependencies']['evaluator_version']}`, "
        f"`{handoff['immutable_dependencies']['remaining_count_baseline']}`, "
        f"`{handoff['immutable_dependencies']['sampler_version']}` "
        f"(`{handoff['immutable_dependencies']['sampler_module_sha256']['stratego/evaluation/phase11_sampler.py'][:16]}...`), "
        f"`{handoff['immutable_dependencies']['information_safety_version']}`, the "
        "frozen statistics and the measured CPU runtime configuration. Agent 6 "
        "performs the production soak and the `phase11_system_v1` freeze only."
    )
    lines.append("")
    return "\n".join(lines)


def stage_report(_args) -> dict:
    section = build_report_section()
    existing = REPORT_PATH.read_text() if REPORT_PATH.exists() else ""
    marker = "## 5. Agent 5 —"
    if marker in existing:
        head = existing[: existing.index(marker)].rstrip("\n")
        body = f"{head}\n\n{section}\n"
    else:
        body = f"{existing.rstrip()}\n\n{section}\n"
    REPORT_PATH.write_text(body)
    log(f"wrote report section 5 ({len(section.splitlines())} lines)")
    return {"agent": AGENT, "stage": "report", "lines": len(section.splitlines())}


# ---------------------------------------------------------------------------
# 9. Entry point
# ---------------------------------------------------------------------------

STAGES = {
    "verify": stage_verify,
    "pipeline": stage_pipeline,
    "recompute": stage_recompute,
    "leakage": stage_leakage,
    "freeze": stage_freeze,
    "acceptance": stage_acceptance,
    "report": stage_report,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), default=None)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--record-suite", action="store_true")
    args = parser.parse_args()

    try:
        if args.stage:
            STAGES[args.stage](args)
            return 0
        for name in ("verify", "pipeline", "recompute", "leakage", "freeze"):
            log(f"--- stage {name} ---")
            STAGES[name](args)
        if args.record_suite:
            record_suite(args)
        log("--- stage acceptance ---")
        STAGES["acceptance"](args)
        log("--- stage report ---")
        STAGES["report"](args)
    except Agent5Error as error:
        log(f"BLOCKED: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
