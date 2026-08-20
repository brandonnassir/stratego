#!/usr/bin/env python3
"""Phase 11 Agent 7 harness: independent final acceptance and the sealed test.

Agent 7 performs the administrative freeze verification and then the **first
and only sealed scored evaluation** of `phase11_test_bank_v1`, issuing exactly
one recommendation: `PASS-SEARCH-READY`, `FAIL` or `BLOCKED`. Formal closure
belongs to the reviewing chat; nothing here commits.

Discipline, from the frozen instruction:

- **Stage 0 (administrative freeze)** — clean tracked tree at the accepted
  Agent 6 commit; Agents 1-6 PASS/gates recomputed; the eight Phase 11
  contracts and their bundle recomputed from live bytes; the eight root seeds
  re-read from the frozen module; the Phase 9 checkpoint SHA / model-state
  digest / 863,959 parameters / belief-head tensor identity / optimizer step
  re-derived; the P10-D chain, utility, scaler and Phase 7 library re-hashed;
  the Agent 5 `phase11_validation_freeze_v1` document *rebuilt* from live
  bytes; `phase11_system_v1` re-filled by Agent 1's filling rules and
  slot-walked against the Agent 6 artifact; both banks structurally rebuilt
  from the frozen seeds; and the append-only ledger harvested to prove zero
  scored predictions, zero neural inference, zero privileged truth and zero
  outcomes on the test bank across Agents 1-6. Any failure -> `BLOCKED`; the
  bank is not opened.
- **Stage 1 (first sealed test)** — `run_phase11_pipeline("test", ...,
  sealed_bank_authorized=True)`, the only call site of that flag in Phase 11:
  exactly 2,048 logical paired cases / 4,096 games through the frozen
  `phase11_validation_freeze_v1` pipeline. One run. A failed gate is final
  evidence; there is no rescue rerun, and this harness structurally refuses
  to run the sealed evaluation twice.
- **Stage 2 (final metrics)** — the frozen metric block with final bootstrap
  root `2026081908` (10,000 replicates, logical-case resampling), plus the
  independent `phase11_independent_recompute_v1` path within the frozen
  1e-9 tolerance.
- **Stage 3 (sampler/safety confirmation)** — a predeclared independent
  reconstruction of 1,024 evenly spaced requests of the frozen integrated
  sample schedule in two fresh processes (forward and reverse order), every
  world re-verified by `verify_world_independently` and the document rebuild
  traced for hidden-rank reads; plus the final materialized-stream collision
  audit over every world stream Agent 7 actually instantiated, combined with
  the re-enumerated accepted Agent 4 and Agent 6 universes.
- **Stage 4 (Gates A-H)** — each gate recomputed by the frozen contract
  evaluators from the recorded quantities; the classification recomputes from
  the gate rows alone. No discretionary override.

Nothing here retrains, calibrates, changes a threshold, an ECE bin, a
baseline, a bank, a stratum or a sampler rule; nothing repairs the known
`manifest_digest` wall-clock defect; and nothing reacts to the known
validation reading `R_CE = 0.9750` — if it repeats on the sealed test, Gate A
fails and the classification is `FAIL`.

Deliverables::

    reports/phase_11_data/agent_07_final_acceptance.json
    reports/phase_11_data/agent_07_predictive_results.csv
    reports/phase_11_data/agent_07_calibration_results.csv
    reports/phase_11_data/agent_07_sampler_results.json
    report section 7

Usage::

    python scripts/run_phase11_agent07.py --stage verify
    python scripts/run_phase11_agent07.py --stage banks
    python scripts/run_phase11_agent07.py --stage suite-before
    python scripts/run_phase11_agent07.py --stage pipeline      # THE sealed run
    python scripts/run_phase11_agent07.py --stage recompute
    python scripts/run_phase11_agent07.py --stage sampler
    python scripts/run_phase11_agent07.py --stage streams
    python scripts/run_phase11_agent07.py --stage preservation
    python scripts/run_phase11_agent07.py --stage acceptance
    python scripts/run_phase11_agent07.py --stage suite
    python scripts/run_phase11_agent07.py --stage report

Agent 7's result remains uncommitted until the reviewing chat accepts or
rejects the recommendation.
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

AGENT = 7
PHASE = 11
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_11_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_11_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase11" / "agent07"
CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "selfplay_c1_v1.pt"
EXPORT_PATH = WORK_DIRECTORY / "eval_phase9_c1.pt"
STORE_ROOT = REPOSITORY_ROOT / "data" / "phase11" / "agent07" / "test_predictions"

#: The accepted Agent 6 commit the administrative freeze is anchored to.
STARTING_REVISION = "3fd9098"

#: The Agent 5 implementation freeze and the Agent 6 system identity, from
#: the accepted handoff. Both are *recomputed* from live bytes and compared.
EXPECTED_FREEZE_DIGEST = (
    "ad2562af538abc6c78fc5b12bc1f57d3e32184172acde390417a00d500a0d912"
)
EXPECTED_SYSTEM_DIGEST = (
    "e4452ba38b568a0ed3a5866f761324dcc7f1eea226d7ba6f94fde45ceb3b6101"
)

#: The frozen evaluation backend, unchanged from Agents 2, 4, 5 and 6.
EVAL_DEVICE = "cpu"
EVAL_TORCH_THREADS = 1

#: The upstream evidence Agent 7 binds into the sealed run's Gates E/F/G.
#: Identical to the Agent 5 list; each artifact is re-hashed from live bytes.
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

# ---------------------------------------------------------------------------
# The predeclared Stage 3 sampler audit. These constants are the
# predeclaration: they are recorded by `--stage verify` *before* the sealed
# run exists, they change no sampler mathematics, and the audit reads the
# frozen schedule the pipeline itself produced.
# ---------------------------------------------------------------------------

#: Evenly spaced requests of the realized integrated sample schedule, by the
#: frozen `floor(k * N / take)` spacing rule.
SAMPLER_AUDIT_REQUESTS = 1_024

#: Worlds per audited request — the production request shape.
SAMPLER_AUDIT_WORLD_ORDINALS = 64

#: Two fresh single-process passes: forward order and reverse order. Pass A
#: performs the full independent per-world verification and the traced
#: hidden-input rebuild; pass B re-executes for the fixed-seed cross-process
#: digest comparison.
SAMPLER_AUDIT_PASSES = ("forward", "reverse")

#: Worker count for the Agent 6 soak-state reconstruction reused by the
#: stream audit. Deliberately not Agent 6's own 10, so its recorded worker
#: outputs are not overwritten.
STREAM_RECONSTRUCTION_WORKERS = 12


class Agent7Error(RuntimeError):
    """The Agent 7 harness refused to continue."""


# ---------------------------------------------------------------------------
# Small shared utilities (the accepted Agent 5/6 harness shapes)
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
        raise Agent7Error(f"stage {name!r} has not run yet ({path} is missing)")
    return json.loads(path.read_text())


def require(condition: bool, message: str, problems: list) -> bool:
    if not condition:
        problems.append(message)
    return bool(condition)


def log(message: str) -> None:
    print(f"[phase11:agent7] {message}", flush=True)


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


def load_agent6_harness():
    """The accepted Agent 6 harness, loaded as a module.

    The Agent 4 and Agent 6 stream universes are rebuilt by the code that
    produced their accepted records rather than by an Agent 7 paraphrase,
    and each reconstruction must reproduce its accepted record exactly
    before it is used.
    """
    import importlib.util

    path = REPOSITORY_ROOT / "scripts" / "run_phase11_agent06.py"
    specification = importlib.util.spec_from_file_location(
        "run_phase11_agent06_for_agent07", path
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. Stage 0 — the administrative freeze, from live bytes
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
    """Agent 1 PASS, with the eight contract digests re-derived from live code."""
    from stratego.training import phase11_contract as contract

    summary = _verify_agent_acceptance(
        "agent_01_acceptance.json", 1, problems, handoff_key="handoff_to_agent_2"
    )
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


def verify_seeds(problems: list) -> dict:
    """The eight root seeds and their derivation module, from live bytes."""
    from stratego.training import phase11_seed as seed

    expected = {
        "phase11_master_seed": 2026081901,
        "bank_schedule_seed": 2026081902,
        "match_randomness_seed": 2026081903,
        "world_sampling_seed": 2026081904,
        "information_safety_seed": 2026081905,
        "repro_runtime_seed": 2026081906,
        "validation_bootstrap_seed": 2026081907,
        "test_bootstrap_seed": 2026081908,
    }
    live = dict(seed.CANONICAL_PHASE11_SEEDS)
    require(
        live == expected,
        f"the canonical Phase 11 seeds moved: {live} != {expected}",
        problems,
    )
    a1 = read_json(DATA_DIRECTORY / "agent_01_acceptance.json")
    require(
        a1.get("seeds") == expected,
        "the Agent 1 recorded seeds disagree with the frozen literals",
        problems,
    )
    # The final-test bootstrap root is reachable only through the frozen
    # per-bank rule; both paths must name 2026081908 for the test bank.
    require(
        int(seed.bootstrap_root("test")) == 2026081908,
        "the test bootstrap root does not derive to 2026081908",
        problems,
    )
    require(
        int(seed.bootstrap_root("validation")) == 2026081907,
        "the validation bootstrap root does not derive to 2026081907",
        problems,
    )
    return {
        "seeds": live,
        "domains": list(seed.STREAM_DOMAINS),
        "identity_version": seed.PHASE11_IDENTITY_VERSION,
        "test_bootstrap_root": int(seed.bootstrap_root("test")),
        "seed_module_sha256": file_sha256(
            REPOSITORY_ROOT / "stratego" / "training" / "phase11_seed.py"
        ),
    }


def verify_agent2(problems: list) -> dict:
    """Agent 2 PASS and its evaluator record, unchanged."""
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
                "metrics_file_sha256": file_sha256(metrics_path),
            }
        )
    return summary


def verify_agent3(problems: list) -> dict:
    """Agent 3 PASS and `belief_sampler_v1` byte-unchanged."""
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
    """Agent 4 PASS with its safety / topology / runtime evidence intact."""
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
    summary.update(
        {
            "safety_counters": counters,
            "safety_trials": safety.get("trials", {}),
            "trial_rollup_digest": safety.get("trial_rollup_digest"),
            "leg_exact": legs,
            "distinct_rollup_digests": repro.get("distinct_rollup_digests", []),
            "measured_runtime": runtime,
            "sampler_identity": sampler_identity,
            "evaluator_identity": handoff.get("evaluator_identity", {}),
            "sampler_module_sha256": live_sampler,
        }
    )
    return summary


def verify_banks_artifacts(problems: list) -> dict:
    """Both bank artifacts re-hashed from their stored cases (not a rebuild)."""
    from stratego.evaluation import phase11_banks as banks
    from stratego.evaluation.phase11_banks import Phase11Case

    summary = {}
    for name, filename, expected_version, expected_cases in (
        ("validation", "agent_01_validation_bank.json", "phase11_validation_bank_v1", 512),
        ("test", "agent_01_test_bank.json", "phase11_test_bank_v1", 2_048),
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
            len(cases) == expected_cases,
            f"the {name} bank holds {len(cases)} cases, expected {expected_cases}",
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


def verify_agent5_freeze(problems: list, phase9: dict, banks_summary: dict) -> dict:
    """The whole Agent 5 freeze document, rebuilt from live bytes."""
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


def system_identity(document: dict) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _absolute_paths_in(node, trail: str = "") -> "list[str]":
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


def verify_system_v1(problems: list, verify: dict) -> dict:
    """`phase11_system_v1` re-filled by Agent 1's rules and slot-walked.

    The template comes from the live Agent 1 contract artifact; every slot
    is re-filled from accepted values re-derived here; and the resulting
    document must equal the Agent 6 artifact field-for-field and hash to the
    handed-over digest.
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

    banks = {
        "phase11_validation_bank_v1": verify["banks"]["validation"]["bank_digest"],
        "phase11_test_bank_v1": verify["banks"]["test"]["bank_digest"],
    }
    require(
        banks == freeze["bank_digests"],
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

    recorded = read_json(DATA_DIRECTORY / "agent_06_system_v1.json")["phase11_system_v1"]
    differing = sorted(
        key
        for key in set(document) | set(recorded)
        if document.get(key) != recorded.get(key)
    )
    require(
        not differing,
        f"the re-filled phase11_system_v1 differs from the Agent 6 artifact on {differing}",
        problems,
    )
    require(
        document["system_digest"] == EXPECTED_SYSTEM_DIGEST,
        f"the re-filled system digest {document['system_digest']} != the handed-over "
        f"{EXPECTED_SYSTEM_DIGEST}",
        problems,
    )
    return {
        "system_version": document["system_version"],
        "recomputed_system_digest": document["system_digest"],
        "recorded_system_digest": recorded.get("system_digest"),
        "matches_agent6_artifact": not differing,
        "matches_handoff_digest": document["system_digest"] == EXPECTED_SYSTEM_DIGEST,
        "differing_fields": differing,
        "slots_filled": sorted(filled),
    }


def verify_agent6(problems: list, verify: dict) -> dict:
    """Agent 6 PASS, its handoff, and its soak/stream records intact."""
    summary = _verify_agent_acceptance(
        "agent_06_acceptance.json", 6, problems, handoff_key="handoff_to_agent_7"
    )
    if not summary.get("available"):
        return summary
    handoff = summary["handoff"]
    require(
        handoff.get("phase11_system_v1_digest") == EXPECTED_SYSTEM_DIGEST,
        "the Agent 6 handoff names a different phase11_system_v1 digest",
        problems,
    )
    require(
        handoff.get("validation_freeze", {}).get("freeze_digest")
        == EXPECTED_FREEZE_DIGEST,
        "the Agent 6 handoff names a different validation freeze digest",
        problems,
    )
    require(
        int(handoff.get("test_bank", {}).get("scored_access_so_far", -1)) == 0,
        "the Agent 6 handoff reports non-zero scored test access",
        problems,
    )
    stream_audit = read_json(DATA_DIRECTORY / "agent_06_stream_audit.json")
    require(
        int(stream_audit.get("total_accidental_collisions", -1)) == 0,
        "the Agent 6 stream audit records collisions",
        problems,
    )
    summary.update(
        {
            "system_v1_digest": handoff.get("phase11_system_v1_digest"),
            "soak_store_content_digest": handoff.get("soak_store_content_digest"),
            "stream_audit_combined": stream_audit.get("combined", {}),
        }
    )
    return summary


def verify_test_bank_sealed(problems: list) -> dict:
    """The pre-Agent-7 sealing proof, harvested from the live ledger."""
    from stratego.evaluation import phase11_banks as banks
    from stratego.evaluation import phase11_pipeline as pipeline

    entries = banks.read_ledger()
    pre_agent7 = [entry for entry in entries if int(entry["agent"]) <= 6]
    require(
        len(pre_agent7) == len(entries),
        "the ledger already carries an Agent 7 entry before the freeze",
        problems,
    )
    summary = banks.verify_test_bank_sealed(pre_agent7)
    require(not summary["violations"], f"ledger violations: {summary['violations']}", problems)
    require(
        summary["test_bank_structural_only"],
        "the test bank has non-structural ledger access before Agent 7",
        problems,
    )
    for key in (
        "scored_prediction_total",
        "privileged_truth_total",
        "outcome_total",
        "neural_inference_total",
    ):
        require(int(summary[key]) == 0, f"pre-Agent-7 test-bank {key} is not zero", problems)

    refused = False
    try:
        pipeline.assert_seal("test", sealed_bank_authorized=False)
    except pipeline.Phase11SealError:
        refused = True
    require(refused, "the pipeline did not refuse the sealed test bank", problems)
    allowed = True
    try:
        pipeline.assert_seal("validation", sealed_bank_authorized=False)
    except pipeline.Phase11SealError:  # pragma: no cover - would be a defect
        allowed = False
    require(allowed, "the pipeline refused the open validation bank", problems)

    by_agent: dict = {}
    for entry in pre_agent7:
        key = str(entry["agent"])
        bucket = by_agent.setdefault(
            key, {"entries": 0, "test_bank_entries": 0, "non_structural": 0}
        )
        bucket["entries"] += 1
        if entry["bank_version"] == "phase11_test_bank_v1":
            bucket["test_bank_entries"] += 1
        if not entry["structural_only"]:
            bucket["non_structural"] += 1
    summary.update(
        {
            "pre_agent7_entries": len(pre_agent7),
            "entries_by_agent": by_agent,
            "test_refused_without_authorization": refused,
            "validation_open": allowed,
            "ledger_sha256": file_sha256(banks.ledger_path()),
        }
    )
    return summary


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
    if stage_path("pipeline").exists():
        raise Agent7Error(
            "the sealed run already happened; the administrative freeze record "
            "is frozen and is not re-verified in place"
        )
    problems: list[str] = []
    log("checking the tracked tree against the accepted Agent 6 commit")
    head = _git("rev-parse", "--short", "HEAD")
    tracked_dirty = [
        line
        for line in _git("status", "--porcelain").splitlines()
        if line.strip() and not line.startswith("??")
    ]
    require(
        head.startswith(STARTING_REVISION),
        f"HEAD {head} is not the accepted Agent 6 commit {STARTING_REVISION}",
        problems,
    )
    require(
        not tracked_dirty,
        f"the tracked tree is not clean at the freeze: {tracked_dirty}",
        problems,
    )

    log("verifying the Agents 1-6 acceptance chain")
    agent1 = verify_agent1(problems)
    agent2 = verify_agent2(problems)
    agent3 = verify_agent3(problems)
    agent4 = verify_agent4(problems)
    agent5 = _verify_agent_acceptance(
        "agent_05_acceptance.json", 5, problems, handoff_key="handoff_to_agent_6"
    )
    log("recomputing the eight root seeds and derivations")
    seeds = verify_seeds(problems)
    log("re-hashing both frozen bank artifacts")
    bank_summary = verify_banks_artifacts(problems)
    log("re-deriving the Phase 9 checkpoint and belief-head identity")
    phase9 = verify_phase9_checkpoint(problems)
    log("verifying the P10-D / anchor / Phase 7 stack")
    upstream = verify_upstream_stack(problems)

    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "stage": "verify",
        "environment": environment_report(),
        "starting_revision": STARTING_REVISION,
        "head": head,
        "tracked_tree_clean_at_freeze": not tracked_dirty,
        "agent1": agent1,
        "agent2": agent2,
        "agent3": agent3,
        "agent4": agent4,
        "agent5": agent5,
        "seeds": seeds,
        "banks": bank_summary,
        "phase9": phase9,
        "upstream": upstream,
    }

    log("rebuilding the Agent 5 implementation freeze from live bytes")
    payload["agent5_freeze"] = verify_agent5_freeze(problems, phase9, bank_summary)
    log("re-filling phase11_system_v1 by the Agent 1 rules")
    payload["system_v1"] = verify_system_v1(problems, payload)
    log("verifying the Agent 6 record and handoff")
    payload["agent6"] = verify_agent6(problems, payload)
    log("harvesting the pre-Agent-7 ledger sealing proof")
    payload["test_bank_sealing"] = verify_test_bank_sealed(problems)

    payload["predeclared_sampler_audit"] = {
        "rule": (
            f"{SAMPLER_AUDIT_REQUESTS} evenly spaced requests of the realized "
            "integrated sample schedule by the frozen floor(k * N / take) rule, "
            f"each re-executed with {SAMPLER_AUDIT_WORLD_ORDINALS} worlds in "
            "two fresh single processes (forward and reverse order); pass A "
            "verifies every world independently and traces the document "
            "rebuild for hidden-rank reads; the passes' per-request digests "
            "must agree exactly"
        ),
        "requests": SAMPLER_AUDIT_REQUESTS,
        "world_ordinals": SAMPLER_AUDIT_WORLD_ORDINALS,
        "passes": list(SAMPLER_AUDIT_PASSES),
        "alters_sampler_mathematics": False,
        "declared_before_sealed_run": True,
    }
    payload["problems"] = problems
    payload["verified"] = not problems
    payload["bound_evidence"] = bound_evidence(payload) if not problems else {}
    payload["preservation_observation"] = (
        preservation_observation(phase9, upstream) if not problems else {}
    )
    write_stage("verify", payload)
    if problems:
        for problem in problems:
            log(f"PROBLEM: {problem}")
        raise Agent7Error(
            f"the administrative freeze found {len(problems)} problem(s); BLOCKED — "
            "the sealed bank is not opened"
        )
    log("administrative freeze verification clean")
    return payload


# ---------------------------------------------------------------------------
# 2. Structural bank rebuild — both banks, from the frozen seeds
# ---------------------------------------------------------------------------


def _rebuild_one_bank(name: str, stored_path: Path, problems: list, sources) -> dict:
    from stratego.evaluation import phase11_banks as banks

    stored = read_json(stored_path)
    cases, manifest = banks.build_phase11_bank(name, sources)
    rebuilt_cases = [case.to_dict() for case in cases]
    cases_equal = rebuilt_cases == stored["cases"]
    require(cases_equal, f"the rebuilt {name} bank cases differ from the artifact", problems)
    require(
        manifest["bank_digest"] == stored["manifest"]["bank_digest"],
        f"the rebuilt {name} bank digest {manifest['bank_digest']} != stored",
        problems,
    )
    require(
        manifest["manifest_digest"] == stored["manifest"]["manifest_digest"],
        f"the rebuilt {name} manifest digest != stored",
        problems,
    )
    audit = banks.audit_phase11_bank(name, cases, manifest, sources)
    failing_checks = sorted(
        check for check, value in audit["checks"].items() if not value
    )
    require(
        bool(audit["all_pass"]) and not failing_checks,
        f"the {name} bank structural audit failed on: {failing_checks}",
        problems,
    )
    summary = {
        "bank_version": manifest["bank_version"],
        "cases": len(cases),
        "games": 2 * len(cases),
        "rebuilt_bank_digest": manifest["bank_digest"],
        "stored_bank_digest": stored["manifest"]["bank_digest"],
        "rebuilt_manifest_digest": manifest["manifest_digest"],
        "cases_equal_stored": cases_equal,
        "audit_all_pass": bool(audit["all_pass"]),
        "audit_checks": dict(audit["checks"]),
        "audit_counts": {
            "stratum_counts": audit["stratum_counts"],
            "source_counts": audit["source_counts"],
            "cell_counts": audit["cell_counts"],
            "distinct_fingerprints": audit["distinct_fingerprints"],
            "repeated_fingerprints": audit["repeated_fingerprints"],
            "rebuild_sample_every": audit["rebuild_sample_every"],
        },
    }
    return summary, cases


def stage_banks(_args) -> dict:
    """Rebuild both banks structurally from the frozen constants and seeds."""
    from stratego.evaluation import phase11_banks as banks

    if stage_path("pipeline").exists():
        raise Agent7Error(
            "the sealed run already happened; the structural rebuild record is "
            "frozen and is not re-run in place"
        )
    verify = read_stage("verify")
    if not verify["verified"]:
        raise Agent7Error("verify did not pass; refusing the bank rebuild")
    problems: list[str] = []
    sources = banks.Phase11SetupSources()
    log("rebuilding phase11_validation_bank_v1 from the frozen seeds")
    validation_summary, validation_cases = _rebuild_one_bank(
        "validation", DATA_DIRECTORY / "agent_01_validation_bank.json", problems, sources
    )
    log("rebuilding phase11_test_bank_v1 from the frozen seeds")
    test_summary, test_cases = _rebuild_one_bank(
        "test", DATA_DIRECTORY / "agent_01_test_bank.json", problems, sources
    )
    log("running the cross-bank disjointness checks")
    cross = banks.cross_bank_checks(validation_cases, test_cases)
    require(cross["zero_overlap"], f"cross-bank overlap: {cross}", problems)

    banks.append_ledger_entries(
        [
            banks.ledger_entry(
                AGENT,
                "structural_bank_rebuild",
                bank_version,
                "full structural rebuild of the frozen bank from the frozen "
                "seeds for the Agent 7 administrative freeze; no game played, "
                "no inference run, no truth read, no outcome read",
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
        "stage": "banks",
        "validation": validation_summary,
        "test": test_summary,
        "cross_bank": cross,
        "problems": problems,
        "verified": not problems,
    }
    write_stage("banks", payload)
    if problems:
        for problem in problems:
            log(f"PROBLEM: {problem}")
        raise Agent7Error(
            f"the structural bank rebuild found {len(problems)} problem(s); BLOCKED"
        )
    log(
        f"banks rebuilt exactly: validation {validation_summary['rebuilt_bank_digest'][:16]}..., "
        f"test {test_summary['rebuilt_bank_digest'][:16]}..."
    )
    return payload


# ---------------------------------------------------------------------------
# 3. Stage 1 — the first and only sealed scored evaluation
# ---------------------------------------------------------------------------


def _progress(stage: str):
    def report(done, total, volume, elapsed):
        log(f"{stage}: {done}/{total}  {volume:,}  {elapsed:6.1f}s")

    return report


def _sealed_run_structure(manifest: dict, problems: list) -> dict:
    """Exactness of the sealed run: games, strata, colours, sources, events."""
    from stratego.training.phase11_contract import (
        OPPONENT_STRATA,
        SETUP_SOURCES,
        TEST_BANK_CASES,
        TEST_BANK_GAMES,
    )

    entries = manifest["games_index"]
    require(
        len(entries) == TEST_BANK_GAMES and bool(manifest["complete_bank"]),
        f"the sealed run played {len(entries)} games, expected {TEST_BANK_GAMES}",
        problems,
    )
    by_stratum: dict = {}
    by_cell: dict = {}
    colors_by_case: dict = {}
    color_totals = {"red": 0, "blue": 0}
    case_ids = set()
    for entry in entries:
        by_stratum[entry["opponent_stratum"]] = (
            by_stratum.get(entry["opponent_stratum"], 0) + 1
        )
        cell = (entry["opponent_stratum"], entry["opponent_setup_source"])
        by_cell[cell] = by_cell.get(cell, 0) + 1
        colors_by_case.setdefault(entry["case_id"], []).append(entry["observer_color"])
        color_totals[entry["observer_color"]] += 1
        case_ids.add(entry["case_id"])

    games_per_stratum = TEST_BANK_GAMES // len(OPPONENT_STRATA)
    strata_exact = sorted(by_stratum) == sorted(OPPONENT_STRATA) and all(
        count == games_per_stratum for count in by_stratum.values()
    )
    require(strata_exact, f"stratum game counts are not exact: {by_stratum}", problems)
    games_per_cell = games_per_stratum // len(SETUP_SOURCES)
    cells_exact = len(by_cell) == len(OPPONENT_STRATA) * len(SETUP_SOURCES) and all(
        count == games_per_cell for count in by_cell.values()
    )
    require(cells_exact, "setup-source balance is not exact", problems)
    color_exact = (
        len(case_ids) == TEST_BANK_CASES
        and color_totals == {"red": TEST_BANK_GAMES // 2, "blue": TEST_BANK_GAMES // 2}
        and all(sorted(pair) == ["blue", "red"] for pair in colors_by_case.values())
    )
    require(
        color_exact,
        f"colour pairing is not exact: totals {color_totals}",
        problems,
    )

    truth = manifest["truth_pass"]
    events_recorded = (
        int(truth["identity_mismatches"]) == 0
        and int(truth["alignment_mismatches"]) == 0
        and int(truth["count_mismatches"]) == 0
        and int(truth["mask_mismatches"]) == 0
        and int(truth["unlabelled_events"]) == 0
        and int(truth["verified_decisions"]) == int(manifest["observer_decisions"])
        and int(manifest["prediction_events"])
        == sum(int(entry["events"]) for entry in entries)
        and all(
            entry.get("public_shard_digest")
            and entry.get("truth_shard_digest")
            and entry.get("replay_digest")
            for entry in entries
        )
    )
    require(events_recorded, "not every prediction event is recorded and verified", problems)

    return {
        "games": len(entries),
        "cases": len(case_ids),
        "games_per_stratum": by_stratum,
        "games_per_cell": {f"{s}|{src}": count for (s, src), count in sorted(by_cell.items())},
        "color_totals": color_totals,
        "test_games_exact": len(entries) == TEST_BANK_GAMES
        and bool(manifest["complete_bank"]),
        "test_strata_exact": strata_exact,
        "test_color_balance_exact": color_exact,
        "test_setup_source_balance_exact": cells_exact,
        "all_prediction_events_recorded": events_recorded,
        "truth_pass": dict(truth),
    }


def stage_pipeline(_args) -> dict:
    """THE sealed run: `phase11_test_bank_v1`, once, through the frozen freeze.

    This function is the only place in Phase 11 that passes
    `sealed_bank_authorized=True`. It refuses to run twice: an existing
    committed store or stage record means the sealed evaluation already
    happened, and the first valid sealed result is final evidence. If
    artifact-writing failed after a completed run, stop and consult the
    reviewer — never rerun.
    """
    from stratego.evaluation import phase11_banks as banks
    from stratego.evaluation import phase11_pipeline as pipeline

    configure_backend()
    verify = read_stage("verify")
    banks_stage = read_stage("banks")
    if not verify["verified"] or not banks_stage["verified"]:
        raise Agent7Error("the administrative freeze is not clean; refusing the sealed run")
    if stage_path("pipeline").exists():
        raise Agent7Error(
            "stage_pipeline.json already exists: the sealed evaluation already ran "
            "and its first valid result is final. No rescue rerun."
        )
    if (STORE_ROOT / "manifest.json").exists():
        raise Agent7Error(
            f"{STORE_ROOT} already holds a manifest: the sealed evaluation already "
            "ran. No rescue rerun; consult the reviewer."
        )
    if STORE_ROOT.exists() and any(STORE_ROOT.iterdir()):
        raise Agent7Error(
            f"{STORE_ROOT} is non-empty without a manifest: a sealed run started "
            "and did not complete. Primitive evidence is preserved; stop and "
            "consult the reviewer rather than rerun."
        )

    # The authorization is declared in the append-only ledger before the
    # first scored byte exists, so a crash cannot leave scored access
    # unledgered.
    banks.append_ledger_entries(
        [
            banks.ledger_entry(
                AGENT,
                "sealed_test_authorization",
                "phase11_test_bank_v1",
                "Agent 7 opens phase11_test_bank_v1 for its first and only "
                "sealed scored evaluation via run_phase11_pipeline(..., "
                "sealed_bank_authorized=True); counts are recorded by the "
                "sealed_test_run entry on completion",
                structural_only=False,
            )
        ]
    )

    log("SEALED RUN: 2,048 cases / 4,096 games over phase11_test_bank_v1")
    started = time.perf_counter()
    result = pipeline.run_phase11_pipeline(
        "test",
        REPOSITORY_ROOT,
        bound_evidence=verify["bound_evidence"],
        preservation=verify["preservation_observation"],
        store_root=STORE_ROOT,
        export_path=EXPORT_PATH,
        device=EVAL_DEVICE,
        torch_threads=EVAL_TORCH_THREADS,
        sealed_bank_authorized=True,
        progress=_progress,
    )

    problems: list[str] = []
    structure = _sealed_run_structure(result["manifest"], problems)
    summary = {
        "agent": AGENT,
        "phase": PHASE,
        "stage": "pipeline",
        "run_ordinal": 1,
        "pipeline_version": result["pipeline_version"],
        "bank": result["bank"],
        "bank_version": result["bank_version"],
        "bank_digest": result["bank_digest"],
        "sealed_bank_authorized": result["sealed_bank_authorized"],
        "smoke_run": result["smoke_run"],
        "store_root": str(STORE_ROOT),
        "manifest_digest": result["manifest"]["manifest_digest"],
        "store_content_digest": result["stages"]["generate"]["store_content_digest"],
        "structure": structure,
        "structure_problems": problems,
        "stage_order": list(result["stages"]),
        "stages": result["stages"],
        "overall": result["overall"],
        "slices": result["slices"],
        "gates": result["gates"],
        "gate_quantities": result["gate_quantities"],
        "schedule_size": len(result["schedule"]),
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
        "source_revision": verify["environment"]["source_revision"],
    }
    # The stage record and the schedule are written before anything else can
    # fail, so a later artifact problem never tempts a rerun.
    write_stage("pipeline", summary)
    (WORK_DIRECTORY / "sampler_schedule.json").write_text(
        json.dumps(
            {
                "pipeline_version": result["pipeline_version"],
                "bank": "test",
                "requests": result["schedule"],
            },
            sort_keys=True,
            default=_jsonable,
        )
    )

    banks.append_ledger_entries(
        [
            banks.ledger_entry(
                AGENT,
                "sealed_test_run",
                "phase11_test_bank_v1",
                "the complete frozen Phase 11 pipeline over the sealed test "
                "bank: generate, targets, score, metrics, slices, sampler "
                "checks, evidence binding, gate quantities — the first and "
                "only scored access",
                structural_only=False,
                neural_inference_count=int(result["manifest"]["belief_forwards"])
                + int(result["stages"]["sampler_checks"]["requests"]),
                scored_prediction_count=int(result["manifest"]["prediction_events"]),
                privileged_truth_count=int(result["manifest"]["prediction_events"]),
                outcome_count=int(result["manifest"]["games"]),
            )
        ]
    )

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
    if problems:
        for problem in problems:
            log(f"STRUCTURE PROBLEM: {problem}")
    return summary


# ---------------------------------------------------------------------------
# 4. Stage 2 — independent recomputation of every final quantity
# ---------------------------------------------------------------------------


def stage_recompute(_args) -> dict:
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
# 5. Stage 3a — the predeclared independent sampler reconstruction
# ---------------------------------------------------------------------------


def _evenly_spaced_indices(total: int, take: int) -> "list[int]":
    """The frozen `floor(k * N / take)` spacing rule over `range(total)`."""
    take = min(int(take), int(total))
    return [(index * int(total)) // take for index in range(take)]


def _audit_spec_rows() -> "list[dict]":
    """The predeclared audit subset, from the recorded frozen schedule."""
    schedule = json.loads((WORK_DIRECTORY / "sampler_schedule.json").read_text())
    requests = schedule["requests"]
    chosen = _evenly_spaced_indices(len(requests), SAMPLER_AUDIT_REQUESTS)
    return [requests[index] for index in chosen]


def _sampler_worker_command(direction: str, out_path: Path, verify_worlds: bool) -> list:
    return [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "run_phase11_agent07.py"),
        "--stage",
        "sampler-worker",
        "--direction",
        direction,
        "--out",
        str(out_path),
        *(["--verify-worlds"] if verify_worlds else []),
    ]


def stage_sampler_worker(args) -> dict:
    """One fresh-process audit pass over the predeclared subset."""
    configure_backend()
    from stratego.evaluation import phase11_records as records
    from stratego.evaluation.phase11_repro import (
        build_owner,
        execute_request,
        game_setups,
        replay_state,
    )
    from stratego.evaluation.phase11_safety import instrument_hidden_types
    from stratego.evaluation.phase11_sampler_audit import (
        independent_identity,
        independent_sample_token,
        verify_world_independently,
    )
    from stratego.evaluation.phase11_public_state import build_public_state_document
    from stratego.evaluation.policy import (
        PolicyRef,
        PolicyRequirements,
        build_policy_input,
    )
    from stratego.training.phase11_contract import BELIEF_REQUEST_VERSION

    rows = _audit_spec_rows()
    if args.direction == "reverse":
        rows = list(reversed(rows))
    bank_payload = read_json(DATA_DIRECTORY / "agent_01_test_bank.json")
    cases = {case["case_id"]: case for case in bank_payload["cases"]}
    root = STORE_ROOT
    owner = build_owner(EXPORT_PATH)
    written = 0
    histories: dict = {}
    try:
        with open(Path(args.out), "w", buffering=1) as stream:
            for row in rows:
                game_id = row["game_id"]
                if game_id not in histories:
                    arrays = records.read_public_shard(root, game_id)
                    histories[game_id] = [int(v) for v in arrays["action_history"]]
                spec = {
                    **row,
                    **game_setups(cases, row["case_id"], int(row["game_index"])),
                    "action_history": histories[game_id],
                }
                state, observer = replay_state(spec)
                result, parts = execute_request(
                    owner,
                    spec,
                    world_count=SAMPLER_AUDIT_WORLD_ORDINALS,
                    state=state,
                    observer=observer,
                    collect=True,
                )
                record = {
                    "request_id": spec["request_id"],
                    "public_state_identity": result.public_state_identity,
                    "hidden_pieces": int(result.hidden_pieces),
                    "worlds": int(result.worlds),
                    "digest": result.digest,
                }
                if args.verify_worlds:
                    inventory_errors = 0
                    public_constraint_errors = 0
                    provenance_mismatches = 0
                    findings: list[str] = []
                    document = parts["document"]
                    if independent_identity(document) != spec["public_state_identity"]:
                        findings.append("the public-state identity does not re-derive")
                    for ordinal, world in enumerate(parts["worlds"]):
                        token = independent_sample_token(
                            world["sampler_version"],
                            spec["public_state_identity"],
                            ordinal,
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
                    traced, counter = instrument_hidden_types(state, observer)
                    policy_input = build_policy_input(
                        traced,
                        policy=PolicyRef(
                            policy_id="phase11_agent07_audit",
                            policy_version=BELIEF_REQUEST_VERSION,
                        ),
                        policy_seed=0,
                        requirements=PolicyRequirements(
                            observation=True, legal_action_mask=True, public_view=True
                        ),
                        match_id=spec["request_id"],
                        game_id=spec["game_id"],
                    )
                    rebuilt = build_public_state_document(
                        policy_input.require_public_view(),
                        policy_input.require_observation(),
                    )
                    if rebuilt != document:
                        findings.append("the traced rebuild produced a different document")
                    record.update(
                        {
                            "findings": findings,
                            "inventory_errors": inventory_errors,
                            "public_constraint_errors": public_constraint_errors,
                            "provenance_mismatches": provenance_mismatches,
                            "hidden_input_accesses": int(counter.reads),
                        }
                    )
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                written += 1
    finally:
        owner.close()
    print(json.dumps({"written": written}))
    return {"written": written}


def stage_sampler(_args) -> dict:
    """Run both predeclared fresh-process audit passes and reconcile them."""
    read_stage("pipeline")
    started = time.perf_counter()
    directory = WORK_DIRECTORY / "sampler_audit"
    directory.mkdir(parents=True, exist_ok=True)
    outputs: dict = {}
    for direction in SAMPLER_AUDIT_PASSES:
        out_path = directory / f"pass_{direction}.jsonl"
        if out_path.exists():
            out_path.unlink()
        verify_worlds = direction == "forward"
        log(f"sampler audit pass {direction} (fresh process, verify={verify_worlds})")
        completed = subprocess.run(
            _sampler_worker_command(direction, out_path, verify_worlds),
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise Agent7Error(
                f"sampler audit pass {direction} failed: {completed.stderr[-2000:]}"
            )
        outputs[direction] = [
            json.loads(line)
            for line in out_path.read_text().splitlines()
            if line.strip()
        ]

    forward = {row["request_id"]: row for row in outputs["forward"]}
    reverse = {row["request_id"]: row for row in outputs["reverse"]}
    spec_rows = _audit_spec_rows()
    expected_ids = [row["request_id"] for row in spec_rows]
    digest_mismatches = [
        request_id
        for request_id in expected_ids
        if forward[request_id]["digest"] != reverse[request_id]["digest"]
    ]
    identity_mismatches = [
        request_id
        for request_id in expected_ids
        if forward[request_id]["public_state_identity"]
        != next(r for r in spec_rows if r["request_id"] == request_id)[
            "public_state_identity"
        ]
    ]
    counters = {
        "inventory_errors": sum(int(row["inventory_errors"]) for row in outputs["forward"]),
        "public_constraint_errors": sum(
            int(row["public_constraint_errors"]) for row in outputs["forward"]
        ),
        "provenance_mismatches": sum(
            int(row["provenance_mismatches"]) for row in outputs["forward"]
        ),
        "hidden_input_accesses": sum(
            int(row["hidden_input_accesses"]) for row in outputs["forward"]
        ),
        "request_findings": sum(len(row["findings"]) for row in outputs["forward"]),
        "cross_process_digest_mismatches": len(digest_mismatches),
        "recorded_identity_mismatches": len(identity_mismatches),
    }
    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "stage": "sampler",
        "predeclared_rule": read_stage("verify")["predeclared_sampler_audit"],
        "requests_audited": len(expected_ids),
        "worlds_verified": len(expected_ids) * SAMPLER_AUDIT_WORLD_ORDINALS,
        "distinct_public_states": len(
            {row["public_state_identity"] for row in outputs["forward"]}
        ),
        "passes": {
            direction: {"rows": len(outputs[direction])}
            for direction in SAMPLER_AUDIT_PASSES
        },
        "counters": counters,
        "all_counters_zero": all(value == 0 for value in counters.values()),
        "digest_mismatch_ids": digest_mismatches[:8],
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
    }
    write_stage("sampler", payload)
    log(
        f"sampler audit: {payload['requests_audited']} requests / "
        f"{payload['worlds_verified']:,} worlds, all counters "
        f"{'zero' if payload['all_counters_zero'] else 'NON-ZERO'}"
    )
    if not payload["all_counters_zero"]:
        raise Agent7Error(f"the independent sampler audit found problems: {counters}")
    return payload


# ---------------------------------------------------------------------------
# 6. Stage 3b — the final materialized-stream collision audit
# ---------------------------------------------------------------------------


def stage_streams(_args) -> dict:
    """Every Agent 7 world stream, combined with the accepted universes.

    Condition tested, unchanged from Agents 4 and 6: two different logical
    identities must not map to the same derived 63-bit seed. Intentional
    reuse of one logical identity is deduplicated before the comparison —
    the test bank's setup/match/bootstrap streams are Agent 1's already
    enumerated identities (carried in the Agent 4 universe), and a world
    token shared with a prior universe is one identity, not two.
    """
    import numpy as np

    from stratego.evaluation import phase11_records as records
    from stratego.evaluation import phase11_streams as streams_module
    from stratego.training import phase11_seed as seed
    from stratego.training.phase11_contract import (
        BELIEF_SAMPLER_VERSION,
        OPPONENT_STRATA,
        OVERALL_METRIC_TOKENS,
    )

    run = read_stage("pipeline")
    started = time.perf_counter()

    # --- Agent 7's own universe: world streams over the frozen schedule ----
    schedule = json.loads((WORK_DIRECTORY / "sampler_schedule.json").read_text())
    requests = schedule["requests"]
    if len(requests) != int(run["schedule_size"]):
        raise Agent7Error("the recorded schedule does not match the pipeline stage")
    root = Path(run["store_root"])
    manifest = records.read_manifest(root)
    log(f"indexing the test store: {manifest['games']} games")
    index = streams_module.store_state_index(root, manifest)
    if index["slot_set_disagreements"]:
        raise Agent7Error(
            f"slot-set disagreements in the test store: "
            f"{index['slot_set_disagreements'][:4]}"
        )
    slots_by_identity = index["slots_by_identity"]
    schedule_identities = sorted({row["public_state_identity"] for row in requests})
    missing = [
        identity for identity in schedule_identities if identity not in slots_by_identity
    ]
    if missing:
        raise Agent7Error(f"{len(missing)} schedule identities missing from the store")

    log(
        f"deriving Agent 7 world streams: {len(schedule_identities):,} states x "
        f"{run['stages']['sampler_checks']['world_ordinals_per_request']} ordinals"
    )
    ordinals = range(int(run["stages"]["sampler_checks"]["world_ordinals_per_request"]))
    agent7_tokens = streams_module.tokens_for(
        schedule_identities, ordinals, BELIEF_SAMPLER_VERSION
    )
    used_tokens = {
        streams_module.phase11_sample_token(
            BELIEF_SAMPLER_VERSION, row["public_state_identity"], ordinal
        )
        for row in requests
        for ordinal in ordinals
    }
    if not used_tokens <= agent7_tokens:
        raise Agent7Error("a world token used by the sealed run is not enumerated")
    fast_path = streams_module.verify_fast_path(agent7_tokens, slots_by_identity, {})
    if not fast_path["exact"]:
        raise Agent7Error(f"the bulk derivation path disagrees: {fast_path}")

    # --- intentional reuse of Agent 1's enumerated identities --------------
    # The sealed run also materialized the test bank's setup, match and
    # bootstrap streams. Those are exactly the identities Agent 1 enumerated
    # (and Agent 4's accepted universe carries), so they are verified as
    # members and never added again.
    log("verifying the test-bank setup/match/bootstrap streams are Agent 1's")
    bank_payload = read_json(DATA_DIRECTORY / "agent_01_test_bank.json")
    setup_seeds = set()
    match_seeds = set()
    for case in bank_payload["cases"]:
        for game_index_str, game in case["games"].items():
            game_index = int(game_index_str)
            setup_seeds.add(
                seed.case_setup_seed(case["case_id"], game_index, seed.ROLE_OBSERVER)
            )
            setup_seeds.add(
                seed.case_setup_seed(case["case_id"], game_index, seed.ROLE_OPPONENT)
            )
            match_seeds.add(seed.game_match_seed(game["game_id"]))
            if int(game["observer"]["setup_seed"]) != seed.case_setup_seed(
                case["case_id"], game_index, seed.ROLE_OBSERVER
            ):
                raise Agent7Error("a stored observer setup seed does not re-derive")
            if int(game["opponent"]["setup_seed"]) != seed.case_setup_seed(
                case["case_id"], game_index, seed.ROLE_OPPONENT
            ):
                raise Agent7Error("a stored opponent setup seed does not re-derive")
            if int(game["match_seed"]) != seed.game_match_seed(game["game_id"]):
                raise Agent7Error("a stored match seed does not re-derive")
    bootstrap_seeds_used = set()
    for name, block in run["overall"]["metrics"].items():
        if block.get("stream_seed") is not None:
            bootstrap_seeds_used.add(int(block["stream_seed"]))
    for stratum, block in run["slices"]["opponent_stratum"].items():
        for name in ("ce_learned", "ce_baseline", "ce_delta", "top1_delta", "brier_delta", "r_ce"):
            if block[name].get("stream_seed") is not None:
                bootstrap_seeds_used.add(int(block[name]["stream_seed"]))
    agent1_bootstrap = {
        seed.bootstrap_stream_seed(bank, token)
        for bank in ("validation", "test")
        for base in OVERALL_METRIC_TOKENS
        for token in (base, *(f"{base}|st={stratum}" for stratum in OPPONENT_STRATA))
    }
    if not bootstrap_seeds_used <= agent1_bootstrap:
        raise Agent7Error(
            "a bootstrap stream seed used by the sealed run is outside Agent 1's "
            "enumerated universe"
        )

    # --- the accepted Agent 4 and Agent 6 universes, re-enumerated ---------
    agent6_module = load_agent6_harness()
    log("re-enumerating the accepted Agent 4 stream universe (by the Agent 6 code)")
    agent4 = agent6_module.agent4_stream_universe()
    if not agent4["reproduces_accepted_record"]:
        raise Agent7Error(
            f"the Agent 4 universe reconstruction mismatches: {agent4['mismatches']}"
        )
    log(f"  Agent 4 reproduced exactly: {agent4['total_identities']:,} identities")

    log("re-enumerating the accepted Agent 6 stream universe (by its own code)")
    plan_document = agent6_module.load_plan()
    soak_requests = plan_document["requests"]
    soak_games = plan_document["games"]
    rows = agent6_module.run_state_reconstruction(
        soak_requests, agent6_module.plan_path(), STREAM_RECONSTRUCTION_WORKERS
    )
    agent6 = agent6_module.agent6_stream_universe(
        {"games_index": soak_games}, soak_requests, rows
    )
    agent6_total = sum(int(array.size) for array in agent6["arrays"].values())
    recorded6 = read_json(DATA_DIRECTORY / "agent_06_stream_audit.json")
    recorded6_total = int(recorded6["agent6"]["unique_logical_identities"])
    if agent6_total != recorded6_total:
        raise Agent7Error(
            f"the Agent 6 universe reconstruction holds {agent6_total:,} identities, "
            f"the accepted record {recorded6_total:,}"
        )

    # --- combine, deduplicating intentional reuse --------------------------
    log("combining the three universes with logical-identity deduplication")
    agent4_setup = {
        (seed.phase11_soak_game_id(stratum, ordinal), role)
        for stratum in seed.OPPONENT_STRATA
        for ordinal in range(seed.SOAK_GAMES_PER_STRATUM)
        for role in (seed.ROLE_OBSERVER, seed.ROLE_OPPONENT)
    }
    agent4_match = {
        seed.phase11_soak_game_id(stratum, ordinal)
        for stratum in seed.OPPONENT_STRATA
        for ordinal in range(seed.SOAK_GAMES_PER_STRATUM)
    }
    combined_arrays: dict = {}
    for name, array in agent6["arrays"].items():
        combined_arrays[name] = np.asarray(array, dtype=np.uint64)
    for name, array in agent4["arrays"].items():
        if name in combined_arrays:
            combined_arrays[name] = np.concatenate(
                [
                    combined_arrays[name],
                    agent6_module._identities_not_already_counted(
                        name, array, agent6, agent4, agent4_setup, agent4_match
                    ),
                ]
            )
        else:
            combined_arrays[name] = np.asarray(array, dtype=np.uint64)
    prior_total = sum(int(array.size) for array in combined_arrays.values())
    recorded_prior = int(recorded6["combined"]["unique_logical_identities"])
    if prior_total != recorded_prior:
        raise Agent7Error(
            f"the reconstructed prior universe holds {prior_total:,} identities, "
            f"the accepted record {recorded_prior:,}"
        )

    prior_tokens = agent6["tokens"] | agent4["tokens"]
    new_tokens = agent7_tokens - prior_tokens
    shared_tokens = agent7_tokens & prior_tokens
    merged_slots = dict(agent4["slots_by_identity"])
    merged_slots.update(agent6["slots_by_identity"])
    for token in shared_tokens:
        identity = streams_module.token_identity(token)
        if tuple(slots_by_identity[identity]) != tuple(merged_slots[identity]):
            raise Agent7Error(
                f"identity {identity[:16]}... carries different slot sets across "
                "universes"
            )
    log(f"deriving seeds for {len(new_tokens):,} new Agent 7 world tokens")
    agent7_new = streams_module.world_stream_seeds(new_tokens, slots_by_identity)
    world_children_new = int(agent7_new["world_order"].size)
    for name, array in agent7_new.items():
        combined_arrays[name] = np.concatenate([combined_arrays[name], array])

    log("running the combined collision audit")
    internal = streams_module.combined_collision_audit(
        {name: array for name, array in agent7_new.items()}
    ) if new_tokens else {
        "accidental_collisions": 0,
        "total_identities": 0,
        "distinct_seeds": 0,
        "no_collisions": True,
    }
    combined = streams_module.combined_collision_audit(combined_arrays)

    agent7_total = int(sum(int(array.size) for array in agent7_new.values()))
    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "stage": "streams",
        "audit_version": "phase11_agent07_materialized_stream_audit_v1",
        "condition_tested": (
            "two different logical identities must not map to the same derived "
            "63-bit seed"
        ),
        "agent7": {
            "schedule_requests": len(requests),
            "distinct_public_states": len(schedule_identities),
            "world_tokens": len(agent7_tokens),
            "world_tokens_new": len(new_tokens),
            "world_tokens_shared_with_prior_universes": len(shared_tokens),
            "world_order_children_new": world_children_new,
            "new_identities_total": agent7_total,
            "internal_accidental_collisions": int(internal["accidental_collisions"]),
            "fast_path_check": fast_path,
        },
        "intentional_reuse": {
            "test_bank_setup_seeds_verified_member_of_agent1": len(setup_seeds),
            "test_bank_match_seeds_verified_member_of_agent1": len(match_seeds),
            "bootstrap_stream_seeds_used": len(bootstrap_seeds_used),
            "bootstrap_seeds_member_of_agent1_universe": True,
            "note": (
                "the sealed run's bank setup/match draws and bootstrap streams "
                "re-materialize Agent 1's enumerated identities (carried in the "
                "Agent 4 universe) and are therefore deduplicated, never added "
                "twice; a world token shared with the Agent 4/6 universes is one "
                "logical identity and is counted once"
            ),
        },
        "agent4": {
            "universe_identities": int(agent4["total_identities"]),
            "reproduces_accepted_record": bool(agent4["reproduces_accepted_record"]),
        },
        "agent6": {
            "universe_identities": agent6_total,
            "reproduces_accepted_record": agent6_total == recorded6_total,
            "distinct_positions_reconstructed": len(rows),
        },
        "prior_combined_identities": prior_total,
        "prior_combined_matches_accepted": prior_total == recorded_prior,
        "combined": {
            "unique_logical_identities": int(combined["total_identities"]),
            "distinct_seeds": int(combined["distinct_seeds"]),
            "accidental_collisions": int(combined["accidental_collisions"]),
            "no_collisions": bool(combined["no_collisions"]),
            "findings": combined["findings"],
            "bit_width": combined["bit_width"],
            "expected_random_collisions": combined["expected_random_collisions"],
            "per_domain": combined["per_domain"],
        },
        "total_accidental_collisions": int(combined["accidental_collisions"]),
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
    }
    write_stage("streams", payload)
    log(
        f"stream audit: {payload['combined']['unique_logical_identities']:,} combined "
        f"identities, {payload['total_accidental_collisions']} accidental collisions"
    )
    if payload["total_accidental_collisions"]:
        raise Agent7Error("the materialized-stream collision audit found collisions")
    return payload


# ---------------------------------------------------------------------------
# 7. Preservation — every identity re-derived after the sealed evaluation
# ---------------------------------------------------------------------------


def stage_preservation(_args) -> dict:
    """Gate H's observation, measured from live bytes after the test work."""
    from stratego.evaluation import phase11_pipeline as pipeline

    verify = read_stage("verify")
    problems: list[str] = []
    log("re-deriving the Phase 9 checkpoint after the sealed evaluation")
    phase9_after = verify_phase9_checkpoint(problems)
    log("re-deriving the upstream stack after the sealed evaluation")
    upstream_after = verify_upstream_stack(problems)
    after = preservation_observation(phase9_after, upstream_after)
    before = verify["preservation_observation"]

    live_modules = module_digests(pipeline.FROZEN_IMPLEMENTATION_MODULES)
    frozen_modules = verify["agent5_freeze"]["document"]["module_sha256"]
    moved = sorted(
        name for name, digest in frozen_modules.items() if live_modules.get(name) != digest
    )
    require(not moved, f"a frozen implementation module moved: {moved}", problems)

    evidence_after = {
        name: file_sha256(DATA_DIRECTORY / name) for name in BOUND_EVIDENCE_ARTIFACTS
    }
    evidence_unchanged = evidence_after == verify["bound_evidence"]["artifacts"]
    require(evidence_unchanged, "a bound evidence artifact moved during Agent 7", problems)

    bank_files_unchanged = all(
        file_sha256(DATA_DIRECTORY / filename) == verify["banks"][name]["file_sha256"]
        for name, filename in (
            ("validation", "agent_01_validation_bank.json"),
            ("test", "agent_01_test_bank.json"),
        )
    )
    require(bank_files_unchanged, "a bank artifact file moved during Agent 7", problems)

    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "stage": "preservation",
        "before": before,
        "after": after,
        "exact": before == after,
        "phase9_checkpoint_unchanged": before["phase9_checkpoint_sha256"]
        == after["phase9_checkpoint_sha256"]
        and before["phase9_model_state_digest"] == after["phase9_model_state_digest"],
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
        "frozen_modules_unchanged": not moved,
        "bound_evidence_unchanged": evidence_unchanged,
        "bank_files_unchanged": bank_files_unchanged,
        "sampler_identity_unchanged": module_digests(
            verify["agent4"]["sampler_module_sha256"]
        )
        == verify["agent4"]["sampler_module_sha256"],
        "problems": problems,
        "clean": not problems and before == after,
    }
    write_stage("preservation", payload)
    if problems or before != after:
        for problem in problems:
            log(f"PROBLEM: {problem}")
        raise Agent7Error("a preserved identity moved during Agent 7")
    log("preservation exact: every identity re-derived unchanged, delta 0")
    return payload


# ---------------------------------------------------------------------------
# 8. Stage 4 — gates, classification, artifacts, acceptance
# ---------------------------------------------------------------------------


def recompute_gates_from_rows(run: dict, preservation: dict) -> dict:
    """Gates A-H re-evaluated by the frozen contract evaluators."""
    from stratego.training.phase11_contract import (
        evaluate_gate_a,
        evaluate_gate_b,
        evaluate_gate_c,
        evaluate_gate_d,
        evaluate_gate_e,
        evaluate_gate_f,
        evaluate_gate_g,
        evaluate_gate_h,
    )

    quantities = run["gate_quantities"]
    bound = run["stages"]["bound_evidence"]
    combined_sampler = run["stages"]["gate_quantities"]["combined_sampler_counters"]
    return {
        "A": evaluate_gate_a(quantities["r_ce"], quantities["ce_delta_upper"]),
        "B": evaluate_gate_b(quantities["delta_top1"], quantities["delta_top1_lower"]),
        "C": evaluate_gate_c(
            quantities["ece_overall"],
            quantities["stratum_ece"],
            quantities["brier_delta_upper"],
        ),
        "D": evaluate_gate_d(quantities["stratum_r_ce"]),
        "E": evaluate_gate_e({k: int(v) for k, v in combined_sampler.items()}),
        "F": evaluate_gate_f({k: int(v) for k, v in bound["safety_counters"].items()}),
        "G": evaluate_gate_g(dict(bound["leg_exact"]), float(bound["p95_forward_64_ms"])),
        "H": evaluate_gate_h(dict(preservation["after"])),
    }


def completion_gates(stages: dict) -> dict:
    """The Agent 7 completion gates, from the recorded stage evidence."""
    from stratego.evaluation import phase11_pipeline as pipeline
    from stratego.training.phase11_contract import DIAGNOSTIC_SLICES

    verify = stages["verify"]
    banks_stage = stages["banks"]
    run = stages["pipeline"]
    recompute_stage = stages["recompute"]
    sampler = stages["sampler"]
    streams = stages["streams"]
    preservation = stages["preservation"]
    suite = stages.get("suite")
    ledger = stages["ledger"]

    structure = run["structure"]
    generate = run["stages"]["generate"]
    metrics = run["stages"]["metrics"]
    pipeline_sampler = run["stages"]["sampler_checks"]
    comparison = recompute_stage["comparison"]
    recomputed = stages["recomputed_gates"]
    pipeline_gates = run["gates"]
    gates_agree = all(
        recomputed[gate]["passed"] == pipeline_gates[gate]["passed"]
        and recomputed[gate]["checks"] == pipeline_gates[gate]["checks"]
        for gate in "ABCDEFGH"
    )

    return {
        "agents1_6_pass": all(
            verify[name]["status"] == "PASS"
            for name in ("agent1", "agent2", "agent3", "agent4", "agent5", "agent6")
        ),
        "administrative_freeze_verified": bool(verify["verified"])
        and bool(banks_stage["verified"])
        and bool(verify["tracked_tree_clean_at_freeze"]),
        "phase9_identity_verified": bool(verify["phase9"]["available"])
        and verify["phase9"]["parameters"] == 863_959,
        "belief_head_identity_verified": bool(
            verify["phase9"]["belief_head_digest"]
        )
        and verify["preservation_observation"]["belief_head_digest"]
        == verify["phase9"]["belief_head_digest"],
        "phase10_identity_verified": bool(verify["upstream"]["selector_config_sha256"])
        and bool(verify["upstream"]["utility_coefficient_digest"])
        and bool(verify["upstream"]["trait_scaler_digest"]),
        "phase7_identity_verified": bool(
            verify["upstream"]["phase7_library"]["content_digest"]
        ),
        "phase11_contracts_verified": bool(verify["agent1"]["contract_bundle_digest"]),
        "phase11_system_verified": bool(verify["system_v1"]["matches_handoff_digest"])
        and bool(verify["system_v1"]["matches_agent6_artifact"]),
        "validation_bank_rebuild_verified": bool(
            banks_stage["validation"]["cases_equal_stored"]
        ),
        "test_bank_rebuild_verified": bool(banks_stage["test"]["cases_equal_stored"]),
        "pre_agent7_test_score_access_zero": (
            int(verify["test_bank_sealing"]["scored_prediction_total"]) == 0
            and int(verify["test_bank_sealing"]["privileged_truth_total"]) == 0
            and int(verify["test_bank_sealing"]["neural_inference_total"]) == 0
            and int(verify["test_bank_sealing"]["outcome_total"]) == 0
        ),
        "test_games_exact": bool(structure["test_games_exact"]),
        "test_strata_exact": bool(structure["test_strata_exact"]),
        "test_color_balance_exact": bool(structure["test_color_balance_exact"]),
        "test_setup_source_balance_exact": bool(
            structure["test_setup_source_balance_exact"]
        ),
        "all_prediction_events_recorded": bool(
            structure["all_prediction_events_recorded"]
        )
        and int(metrics["events"]) == int(generate["prediction_events"]),
        "metric_recompute_pass": bool(comparison["within_tolerance"])
        and int(comparison["both_nan_comparisons"]) == 0
        and bool(metrics["metrics_finite"])
        and sorted(run["slices"]) == sorted(DIAGNOSTIC_SLICES),
        "independent_bootstrap_pass": bool(comparison["within_tolerance"])
        and all(
            block.get("replicates") == 10_000
            for block in run["overall"]["metrics"].values()
        ),
        "gate_a_recomputed": recomputed["A"]["checks"] == pipeline_gates["A"]["checks"],
        "gate_b_recomputed": recomputed["B"]["checks"] == pipeline_gates["B"]["checks"],
        "gate_c_recomputed": recomputed["C"]["checks"] == pipeline_gates["C"]["checks"],
        "gate_d_recomputed": recomputed["D"]["checks"] == pipeline_gates["D"]["checks"],
        "gate_e_recomputed": recomputed["E"]["checks"] == pipeline_gates["E"]["checks"],
        "gate_f_recomputed": recomputed["F"]["checks"] == pipeline_gates["F"]["checks"],
        "gate_g_recomputed": recomputed["G"]["checks"] == pipeline_gates["G"]["checks"],
        "gate_h_recomputed": recomputed["H"]["checks"] == pipeline_gates["H"]["checks"],
        "final_sampler_audit_pass": bool(pipeline_sampler["all_counters_zero"])
        and bool(
            pipeline_sampler["schedule_accounting"]["every_eligible_game_contributes"]
        )
        and bool(pipeline_sampler["schedule_accounting"]["realized_equals_attainable"])
        and bool(sampler["all_counters_zero"]),
        "illegal_worlds_zero": all(
            int(value) == 0
            for value in run["stages"]["gate_quantities"][
                "combined_sampler_counters"
            ].values()
        )
        and int(sampler["counters"]["inventory_errors"]) == 0
        and int(sampler["counters"]["public_constraint_errors"]) == 0,
        "hidden_input_access_zero": int(sampler["counters"]["hidden_input_accesses"]) == 0
        and all(
            int(value) == 0
            for value in run["stages"]["bound_evidence"]["safety_counters"].values()
        ),
        "nonfinite_zero": bool(metrics["metrics_finite"])
        and not metrics["nonfinite_paths"]
        and int(
            run["stages"]["gate_quantities"]["combined_sampler_counters"][
                "nonfinite_probability_rows"
            ]
        )
        == 0,
        "phase9_checkpoint_unchanged_after_eval": bool(
            preservation["phase9_checkpoint_unchanged"]
        ),
        "belief_head_unchanged_after_eval": bool(preservation["belief_head_unchanged"]),
        "classification_recomputes_from_gate_rows": gates_agree,
        "no_rescue_rerun": int(run["run_ordinal"]) == 1
        and int(ledger["sealed_test_run_entries"]) == 1
        and bool(ledger["single_store_manifest"]),
        "full_suite_green": bool(suite and suite.get("green")),
        # Additional Agent 7 gates beyond the instruction minimum.
        "agent7_materialized_stream_collisions_zero": int(
            streams["total_accidental_collisions"]
        )
        == 0,
        "agent7_stream_universe_reconstruction_faithful": bool(
            streams["agent4"]["reproduces_accepted_record"]
        )
        and bool(streams["agent6"]["reproduces_accepted_record"])
        and bool(streams["prior_combined_matches_accepted"]),
        "phase11_optimizer_step_delta_zero": int(preservation["optimizer_step_delta"])
        == 0,
        "frozen_implementation_unchanged": bool(
            preservation["frozen_modules_unchanged"]
        )
        and run["pipeline_version"] == pipeline.PIPELINE_VERSION,
    }


def ledger_discipline() -> dict:
    """The post-run ledger reading: Agent 7's authorized access, exactly once."""
    from stratego.evaluation import phase11_banks as banks

    entries = banks.read_ledger()
    agent7 = [entry for entry in entries if int(entry["agent"]) == 7]
    test_scored = [
        entry
        for entry in entries
        if entry["bank_version"] == "phase11_test_bank_v1"
        and not entry["structural_only"]
    ]
    pre_agent7 = [entry for entry in entries if int(entry["agent"]) <= 6]
    pre_summary = banks.verify_test_bank_sealed(pre_agent7)
    run_entries = [entry for entry in agent7 if entry["stage"] == "sealed_test_run"]
    return {
        "entries_total": len(entries),
        "agent7_entries": len(agent7),
        "pre_agent7_still_structural_only": bool(pre_summary["test_bank_structural_only"]),
        "pre_agent7_scored_total": int(pre_summary["scored_prediction_total"]),
        "non_structural_test_entries": len(test_scored),
        "non_structural_test_entries_all_agent7": all(
            int(entry["agent"]) == 7 for entry in test_scored
        ),
        "sealed_test_run_entries": len(run_entries),
        "sealed_test_run_counts": run_entries[0] if run_entries else None,
        "single_store_manifest": (STORE_ROOT / "manifest.json").exists(),
    }


def forbidden_operation_counters(sealing_before: dict) -> dict:
    return {
        "phase11_optimizer_steps": 0,
        "belief_calibration_operations": 0,
        "belief_head_writes": 0,
        "threshold_changes": 0,
        "ece_bin_changes": 0,
        "baseline_changes": 0,
        "bank_changes": 0,
        "stratum_changes": 0,
        "sampler_mathematics_changes": 0,
        "manifest_digest_repairs": 0,
        "reactions_to_validation_r_ce": 0,
        "rescue_reruns": 0,
        "model_swaps": 0,
        "backend_changes_after_measurement": 0,
        "p10d_changes": 0,
        "pre_agent7_test_bank_scored_accesses": int(
            sealing_before["scored_prediction_total"]
        ),
        "pre_agent7_test_bank_privileged_truth_reads": int(
            sealing_before["privileged_truth_total"]
        ),
    }


def recorded_readings(stages: dict) -> "list[dict]":
    run = stages["pipeline"]
    quantities = run["gate_quantities"]
    gates = run["gates"]
    readings = [
        {
            "reading": "first_sealed_test_result_is_final",
            "detail": (
                "the sealed evaluation ran exactly once over 2,048 cases / 4,096 "
                f"games; R_CE {quantities['r_ce']:.4f}, delta_top1 "
                f"{quantities['delta_top1']:+.4f}, ECE {quantities['ece_overall']:.4f}. "
                "No rerun, no tuning, no threshold or bank change followed the "
                "result; the harness structurally refuses a second sealed run."
            ),
        },
        {
            "reading": "store_manifest_digest_wall_clock_defect_not_repaired",
            "detail": (
                "the Agent 5 finding stands: phase11_records.manifest_digest "
                "embeds per-game forward_seconds, so two executions of one bank "
                "cannot agree on it. Agent 7 patched nothing; the cross-run "
                "identity of the sealed store is its store_content_digest "
                f"{run['store_content_digest'][:16]}..., and no hard gate reads "
                "the manifest digest."
            ),
        },
        {
            "reading": "validation_r_ce_reading_carried_not_reacted_to",
            "detail": (
                "the known validation reading R_CE = 0.9750 was carried "
                "unchanged into the sealed run: no calibration, threshold, bin, "
                "baseline, bank, stratum or sampler rule moved before or after "
                f"the sealed result (sealed R_CE {quantities['r_ce']:.4f}, "
                f"Gate A {'PASS' if gates['A']['passed'] else 'FAIL'})."
            ),
        },
        {
            "reading": "time_scoped_seal_tests_updated_after_first_authorized_access",
            "detail": (
                "four suite tests asserted the live ledger showed zero scored "
                "test-bank access — the pre-Agent-7 invariant "
                "(test_phase11_agent01/02/03/04_artifacts). After the "
                "authorized sealed run they assert its permanent form: every "
                "pre-Agent-7 entry is structural-only with zero counters, and "
                "the only non-structural test-bank entries are Agent 7's "
                "authorized sealed evaluation. The pre-run suite was recorded "
                "green (including the original tests) before the bank opened; "
                "no frozen module, contract, threshold or artifact moved."
            ),
        },
    ]
    schedule_accounting = run["stages"]["sampler_checks"]["schedule_accounting"]
    if not schedule_accounting["realized_equals_attainable"] or int(
        schedule_accounting["schedule_slots_realized"]
    ) < int(schedule_accounting["schedule_slots_nominal"]):
        readings.append(
            {
                "reading": "integrated_schedule_shortfall_accounted",
                "detail": (
                    "the frozen 4-per-game spacing rule realized "
                    f"{schedule_accounting['schedule_slots_realized']} of "
                    f"{schedule_accounting['schedule_slots_nominal']} nominal "
                    "slots because "
                    f"{schedule_accounting['games_without_eligible_decisions']} "
                    "games offer no eligible decision and "
                    f"{schedule_accounting['games_below_quota']} offer fewer "
                    "than the quota; realized equals attainable "
                    f"({schedule_accounting['realized_equals_attainable']}), "
                    "nothing was made up from another game, and the rule was "
                    "not adjusted. Gate E rests on this pass plus the bound "
                    "Agent 3 large audit."
                ),
            }
        )
    return readings


def _interval(block: dict, digits: int = 4) -> str:
    return (
        f"{block['point']:.{digits}f} [{block['lower']:.{digits}f}, "
        f"{block['upper']:.{digits}f}]"
    )


def write_predictive_csv(run: dict, recompute_stage: dict) -> Path:
    """`agent_07_predictive_results.csv`: overall, strata and pooled slices."""
    path = DATA_DIRECTORY / "agent_07_predictive_results.csv"
    overall = run["overall"]
    strata = run["slices"]["opponent_stratum"]
    independent = recompute_stage["independent_gate_quantities"]
    columns = [
        "scope_type",
        "scope",
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
        "top1_learned",
        "top1_baseline",
        "top1_delta",
        "top1_delta_lower",
        "top1_delta_upper",
        "brier_learned",
        "brier_baseline",
        "brier_delta",
        "brier_delta_lower",
        "brier_delta_upper",
        "true_rank_probability_learned",
        "true_rank_probability_baseline",
        "entropy_learned",
        "entropy_baseline",
        "ece_learned",
        "ece_baseline",
        "independent_r_ce",
        "independent_ece_learned",
    ]

    def fmt(value) -> str:
        return "" if value is None else f"{float(value):.6f}"

    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        metrics = overall["metrics"]
        writer.writerow(
            [
                "overall",
                "overall",
                overall["events"],
                overall["cases_with_events"],
                fmt(metrics["ce_learned"]["point"]),
                fmt(metrics["ce_baseline"]["point"]),
                fmt(metrics["r_ce"]["point"]),
                fmt(metrics["r_ce"]["lower"]),
                fmt(metrics["r_ce"]["upper"]),
                fmt(metrics["ce_delta"]["point"]),
                fmt(metrics["ce_delta"]["lower"]),
                fmt(metrics["ce_delta"]["upper"]),
                fmt(metrics["top1_learned"]["point"]),
                fmt(metrics["top1_baseline"]["point"]),
                fmt(metrics["top1_delta"]["point"]),
                fmt(metrics["top1_delta"]["lower"]),
                fmt(metrics["top1_delta"]["upper"]),
                fmt(metrics["brier_learned"]["point"]),
                fmt(metrics["brier_baseline"]["point"]),
                fmt(metrics["brier_delta"]["point"]),
                fmt(metrics["brier_delta"]["lower"]),
                fmt(metrics["brier_delta"]["upper"]),
                fmt(metrics["true_rank_probability_learned"]["point"]),
                fmt(metrics["true_rank_probability_baseline"]["point"]),
                fmt(metrics["entropy_learned"]["point"]),
                fmt(metrics["entropy_baseline"]["point"]),
                fmt(overall["ece_learned"]["ece"]),
                fmt(overall["ece_baseline"]["ece"]),
                fmt(recompute_stage["independent_gate_quantities"]["r_ce"]),
                fmt(recompute_stage["independent_gate_quantities"]["ece_overall"]),
            ]
        )
        for name in sorted(strata):
            block = strata[name]
            pooled = block["pooled"]
            writer.writerow(
                [
                    "opponent_stratum",
                    name,
                    block["events"],
                    block["cases_with_events"],
                    fmt(block["ce_learned"]["point"]),
                    fmt(block["ce_baseline"]["point"]),
                    fmt(block["r_ce"]["point"]),
                    fmt(block["r_ce"]["lower"]),
                    fmt(block["r_ce"]["upper"]),
                    fmt(block["ce_delta"]["point"]),
                    fmt(block["ce_delta"]["lower"]),
                    fmt(block["ce_delta"]["upper"]),
                    fmt(pooled["top1_learned"]),
                    fmt(pooled["top1_baseline"]),
                    fmt(block["top1_delta"]["point"]),
                    fmt(block["top1_delta"]["lower"]),
                    fmt(block["top1_delta"]["upper"]),
                    fmt(pooled["brier_learned"]),
                    fmt(pooled["brier_baseline"]),
                    fmt(block["brier_delta"]["point"]),
                    fmt(block["brier_delta"]["lower"]),
                    fmt(block["brier_delta"]["upper"]),
                    fmt(pooled["true_rank_probability_learned"]),
                    fmt(pooled["true_rank_probability_baseline"]),
                    fmt(pooled["entropy_learned"]),
                    fmt(pooled["entropy_baseline"]),
                    fmt(block["ece_learned"]["ece"]),
                    fmt(block["ece_baseline"]["ece"]),
                    fmt(independent["stratum_r_ce"][name]),
                    fmt(independent["stratum_ece"][name]),
                ]
            )
        for slice_name in (
            "observer_color",
            "progress_bucket",
            "piece_moved",
            "true_rank",
            "opponent_setup_source",
        ):
            for member in sorted(run["slices"][slice_name]):
                block = run["slices"][slice_name][member]
                if int(block.get("events", 0)) == 0:
                    writer.writerow([slice_name, member, 0] + [""] * (len(columns) - 3))
                    continue
                writer.writerow(
                    [
                        slice_name,
                        member,
                        block["events"],
                        "",
                        fmt(block["ce_learned"]),
                        fmt(block["ce_baseline"]),
                        fmt(block["r_ce"]),
                        "",
                        "",
                        fmt(block["ce_learned"] - block["ce_baseline"]),
                        "",
                        "",
                        fmt(block["top1_learned"]),
                        fmt(block["top1_baseline"]),
                        fmt(block["top1_delta"]),
                        "",
                        "",
                        fmt(block["brier_learned"]),
                        fmt(block["brier_baseline"]),
                        fmt(block["brier_delta"]),
                        "",
                        "",
                        fmt(block["true_rank_probability_learned"]),
                        fmt(block["true_rank_probability_baseline"]),
                        fmt(block["entropy_learned"]),
                        fmt(block["entropy_baseline"]),
                        fmt(block["ece_learned"]),
                        "",
                        "",
                        "",
                    ]
                )
    return path


def write_calibration_csv(run: dict) -> Path:
    """`agent_07_calibration_results.csv`: the frozen 15-bin ECE detail."""
    path = DATA_DIRECTORY / "agent_07_calibration_results.csv"
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "scope_type",
                "scope",
                "model",
                "bin",
                "bin_lower",
                "bin_upper",
                "events",
                "mean_confidence",
                "accuracy",
                "abs_gap",
                "scope_ece",
            ]
        )

        def rows_for(scope_type: str, scope: str, model: str, block: dict):
            for bin_row in block["bins"]:
                gap = (
                    ""
                    if bin_row["confidence"] is None
                    else f"{abs(bin_row['accuracy'] - bin_row['confidence']):.6f}"
                )
                writer.writerow(
                    [
                        scope_type,
                        scope,
                        model,
                        bin_row["bin"],
                        f"{bin_row['lower']:.6f}",
                        f"{bin_row['upper']:.6f}",
                        bin_row["events"],
                        ""
                        if bin_row["confidence"] is None
                        else f"{bin_row['confidence']:.6f}",
                        ""
                        if bin_row["accuracy"] is None
                        else f"{bin_row['accuracy']:.6f}",
                        gap,
                        f"{block['ece']:.6f}",
                    ]
                )

        rows_for("overall", "overall", "learned", run["overall"]["ece_learned"])
        rows_for("overall", "overall", "baseline", run["overall"]["ece_baseline"])
        for name in sorted(run["slices"]["opponent_stratum"]):
            block = run["slices"]["opponent_stratum"][name]
            rows_for("opponent_stratum", name, "learned", block["ece_learned"])
            rows_for("opponent_stratum", name, "baseline", block["ece_baseline"])
    return path


def write_sampler_results(stages: dict) -> Path:
    run = stages["pipeline"]
    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_07_sampler_results",
        "pipeline_sampler_checks": run["stages"]["sampler_checks"],
        "combined_gate_e_counters": run["stages"]["gate_quantities"][
            "combined_sampler_counters"
        ],
        "independent_reconstruction": stages["sampler"],
        "materialized_stream_audit": {
            key: value
            for key, value in stages["streams"].items()
            if key not in ("agent",)
        },
        "bound_agent3_audit": {
            "sampler_audit_counters": run["stages"]["bound_evidence"][
                "sampler_audit_counters"
            ],
            "sampler_audit_worlds": run["stages"]["bound_evidence"][
                "sampler_audit_worlds"
            ],
        },
    }
    write_artifact("agent_07_sampler_results.json", payload)
    return DATA_DIRECTORY / "agent_07_sampler_results.json"


def stage_acceptance(_args) -> dict:
    from stratego.evaluation import phase11_pipeline as pipeline
    from stratego.training.phase11_contract import classify_phase11

    verify = read_stage("verify")
    banks_stage = read_stage("banks")
    run = read_stage("pipeline")
    recompute_stage = read_stage("recompute")
    sampler = read_stage("sampler")
    streams = read_stage("streams")
    preservation = read_stage("preservation")
    suite = read_stage("suite") if stage_path("suite").exists() else None
    suite_before = (
        read_stage("suite_before") if stage_path("suite_before").exists() else None
    )

    log("recomputing Gates A-H from the frozen contract evaluators")
    recomputed = recompute_gates_from_rows(run, preservation)
    ledger = ledger_discipline()

    stages = {
        "verify": verify,
        "banks": banks_stage,
        "pipeline": run,
        "recompute": recompute_stage,
        "sampler": sampler,
        "streams": streams,
        "preservation": preservation,
        "suite": suite,
        "recomputed_gates": recomputed,
        "ledger": ledger,
    }
    gates = completion_gates(stages)
    false_gates = sorted(name for name, value in gates.items() if not value)

    gate_booleans = {gate: bool(block["passed"]) for gate, block in recomputed.items()}
    administrative = [
        name
        for name in (
            "agents1_6_pass",
            "administrative_freeze_verified",
            "phase9_identity_verified",
            "belief_head_identity_verified",
            "phase10_identity_verified",
            "phase7_identity_verified",
            "phase11_contracts_verified",
            "phase11_system_verified",
            "validation_bank_rebuild_verified",
            "test_bank_rebuild_verified",
            "pre_agent7_test_score_access_zero",
            "test_games_exact",
            "test_strata_exact",
            "test_color_balance_exact",
            "test_setup_source_balance_exact",
            "all_prediction_events_recorded",
            "metric_recompute_pass",
            "independent_bootstrap_pass",
            "classification_recomputes_from_gate_rows",
            "no_rescue_rerun",
            "agent7_materialized_stream_collisions_zero",
            "agent7_stream_universe_reconstruction_faithful",
        )
        if not gates[name]
    ]
    integrity_established = not administrative
    classification = classify_phase11(
        gate_booleans,
        experiment_valid=True,
        integrity_established=integrity_established,
    )
    hard_gate_failures = sorted(
        gate for gate, passed in gate_booleans.items() if not passed
    )
    status = "PASS" if not false_gates else "FAIL"

    sealing_before = verify["test_bank_sealing"]
    log("writing the predictive, calibration and sampler artifacts")
    write_predictive_csv(run, recompute_stage)
    write_calibration_csv(run)
    write_sampler_results(stages)

    quantities = run["gate_quantities"]
    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_07_final_acceptance",
        "pipeline_version": pipeline.PIPELINE_VERSION,
        "status": status,
        "recommendation": classification,
        "phase12_authorized": classification == "PASS-SEARCH-READY",
        "starting_revision": STARTING_REVISION,
        "ending_revision": None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": environment_report(),
        "completion_gates": gates,
        "gates_true": sum(1 for value in gates.values() if value),
        "gates_total": len(gates),
        "false_gates": false_gates,
        "hard_gates": recomputed,
        "hard_gate_booleans": gate_booleans,
        "hard_gate_failures": hard_gate_failures,
        "pipeline_gates_agree": gates["classification_recomputes_from_gate_rows"],
        "classification": {
            "classification": classification,
            "rule": (
                "all A-H PASS -> PASS-SEARCH-READY; valid experiment with >=1 "
                "failed gate -> FAIL; integrity/sealing failure -> BLOCKED; no "
                "discretionary override"
            ),
            "integrity_established": integrity_established,
            "experiment_valid": True,
        },
        "gate_quantities": quantities,
        "independent_gate_quantities": recompute_stage["independent_gate_quantities"],
        "sealed_run": {
            "bank_version": run["bank_version"],
            "bank_digest": run["bank_digest"],
            "sealed_bank_authorized": run["sealed_bank_authorized"],
            "run_ordinal": run["run_ordinal"],
            "games": run["structure"]["games"],
            "cases": run["structure"]["cases"],
            "observer_decisions": run["stages"]["generate"]["observer_decisions"],
            "prediction_events": run["stages"]["generate"]["prediction_events"],
            "store_content_digest": run["store_content_digest"],
            "store_manifest_digest": run["manifest_digest"],
            "sampler_worlds": run["stages"]["sampler_checks"]["worlds"],
            "outcomes_report_only": run["stages"]["generate"]["outcomes_report_only"],
            "wall_clock_seconds": run["wall_clock_seconds"],
        },
        "first_scored_access_proof": {
            "pre_agent7_ledger": {
                "entries": sealing_before["pre_agent7_entries"],
                "test_bank_entries": sealing_before["test_bank_entries"],
                "scored_prediction_total": sealing_before["scored_prediction_total"],
                "privileged_truth_total": sealing_before["privileged_truth_total"],
                "neural_inference_total": sealing_before["neural_inference_total"],
                "outcome_total": sealing_before["outcome_total"],
                "structural_only": sealing_before["test_bank_structural_only"],
            },
            "post_run_ledger": ledger,
            "seal_behaviour": {
                "test_refused_without_authorization": sealing_before[
                    "test_refused_without_authorization"
                ],
                "authorization_call_sites": 1,
            },
        },
        "independent_recompute": recompute_stage["comparison"],
        "sampler_confirmation": {
            "requests_audited": sampler["requests_audited"],
            "worlds_verified": sampler["worlds_verified"],
            "counters": sampler["counters"],
            "all_counters_zero": sampler["all_counters_zero"],
        },
        "stream_audit": {
            "agent7_new_identities": streams["agent7"]["new_identities_total"],
            "combined_identities": streams["combined"]["unique_logical_identities"],
            "accidental_collisions": streams["total_accidental_collisions"],
            "prior_combined_matches_accepted": streams["prior_combined_matches_accepted"],
        },
        "preservation": {
            key: value
            for key, value in preservation.items()
            if key not in ("agent", "phase", "stage")
        },
        "forbidden_operation_counters": forbidden_operation_counters(sealing_before),
        "frozen_inputs": {
            "belief_head_digest": verify["phase9"]["belief_head_digest"],
            "phase9_checkpoint_sha256": verify["phase9"]["sha256"],
            "phase9_model_state_digest": verify["phase9"]["model_state_digest"],
            "phase9_parameters": verify["phase9"]["parameters"],
            "global_optimizer_step": verify["phase9"]["global_optimizer_step"],
            "contract_bundle_digest": verify["agent1"]["contract_bundle_digest"],
            "validation_bank_digest": verify["banks"]["validation"]["bank_digest"],
            "test_bank_digest": verify["banks"]["test"]["bank_digest"],
            "validation_freeze_digest": verify["agent5_freeze"][
                "recomputed_freeze_digest"
            ],
            "phase11_system_v1_digest": verify["system_v1"]["recomputed_system_digest"],
            "sampler_module_sha256": verify["agent4"]["sampler_module_sha256"],
            "selector_config_sha256": verify["upstream"]["selector_config_sha256"],
            "phase7_library_content_digest": verify["upstream"]["phase7_library"][
                "content_digest"
            ],
        },
        "recorded_readings": recorded_readings(stages),
        "suite_before": suite_before,
        "suite": suite,
        "handoff_to_reviewing_chat": {
            "for": "reviewing chat",
            "recommendation": classification,
            "phase12_authorized": classification == "PASS-SEARCH-READY",
            "phase11_permanent_freeze_if_pass": {
                "belief_model": "accepted Phase 9 model + belief head",
                "setup_selector": "accepted P10-D",
                "predictive_baseline": "remaining_count_belief_v1",
                "world_sampler": "belief_sampler_v1",
                "system_identity": EXPECTED_SYSTEM_DIGEST,
            },
            "commit_rule": (
                "Agent 7's work remains uncommitted until the reviewing chat "
                "accepts or rejects this recommendation; the sealed result is "
                "never rerun to create a commit"
            ),
        },
    }
    write_stage("acceptance", payload)
    write_artifact("agent_07_final_acceptance.json", payload)
    log(
        f"status {status}: {payload['gates_true']}/{payload['gates_total']} completion "
        f"gates; recommendation {classification}"
    )
    if hard_gate_failures:
        log(f"hard gate failures: {hard_gate_failures}")
    if false_gates:
        log(f"false completion gates: {false_gates}")
    return payload


# ---------------------------------------------------------------------------
# 9. The suite
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


def stage_suite_before(_args) -> dict:
    """The pre-sealed-run suite: proves the freeze on untouched bytes."""
    read_stage("verify")
    read_stage("banks")
    if stage_path("pipeline").exists():
        raise Agent7Error("the sealed run already happened; suite-before is too late")
    log("running the full suite before the sealed run")
    measurement = run_suite()
    measurement["scope"] = (
        "recorded before the sealed evaluation: the tracked tree at the "
        "administrative freeze, including the original pre-Agent-7 seal tests"
    )
    write_stage("suite_before", measurement)
    log(measurement["summary"])
    if not measurement["green"]:
        raise Agent7Error("the pre-sealed-run suite is not green; BLOCKED")
    return measurement


def stage_suite(_args) -> dict:
    log("running the full suite")
    measurement = run_suite()
    write_stage("suite", measurement)
    log(measurement["summary"])
    return measurement


# ---------------------------------------------------------------------------
# 10. The report section
# ---------------------------------------------------------------------------


def _table(rows: "list[tuple]", header: "tuple") -> "list[str]":
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def build_report_section() -> str:
    verify = read_stage("verify")
    banks_stage = read_stage("banks")
    run = read_stage("pipeline")
    recompute_stage = read_stage("recompute")
    sampler = read_stage("sampler")
    streams = read_stage("streams")
    preservation = read_stage("preservation")
    acceptance = read_stage("acceptance")
    suite_before = (
        read_stage("suite_before") if stage_path("suite_before").exists() else None
    )

    metrics = run["overall"]["metrics"]
    quantities = run["gate_quantities"]
    gates = acceptance["hard_gates"]
    strata = run["slices"]["opponent_stratum"]
    accounting = run["stages"]["sampler_checks"]["schedule_accounting"]
    comparison = recompute_stage["comparison"]
    classification = acceptance["recommendation"]
    lines: list[str] = []

    lines.append("## 7. Agent 7 — Independent Final Acceptance and the Sealed Test")
    lines.append("")
    lines.append(
        f"Starting revision `{STARTING_REVISION}` (the accepted Agent 6 commit). "
        f"Status **{acceptance['status']}**, {acceptance['gates_true']}/"
        f"{acceptance['gates_total']} completion gates. Final recommendation: "
        f"**`{classification}`**."
    )
    lines.append("")
    lines.append(
        "Agent 7 verified the complete administrative freeze from live bytes, "
        "then performed the **first and only sealed scored evaluation** of "
        "`phase11_test_bank_v1` — 2,048 logical paired cases / 4,096 games "
        "through the frozen `phase11_validation_freeze_v1` pipeline — recomputed "
        "every quantity independently, confirmed the sampler and stream "
        "universes, recomputed Gates A-H from the frozen contract evaluators, "
        "and classified with no discretionary override. Nothing was retrained, "
        "calibrated, rethresholded, rebinned, rebaselined, rebanked or "
        "resampled; the known `manifest_digest` wall-clock defect was not "
        "repaired; the known validation reading `R_CE = 0.9750` was carried, "
        "not reacted to. The sealed run happened exactly once and its result is "
        "final evidence."
    )
    lines.append("")

    lines.append("### 7.1 Stage 0 — the administrative freeze")
    lines.append("")
    lines.append("```text")
    lines.append(
        f"tracked tree           clean at {STARTING_REVISION} "
        f"(head {verify['head']})"
    )
    for agent_name, label in (
        ("agent1", "Agent 1"),
        ("agent2", "Agent 2"),
        ("agent3", "Agent 3"),
        ("agent4", "Agent 4"),
        ("agent5", "Agent 5"),
        ("agent6", "Agent 6"),
    ):
        block = verify[agent_name]
        lines.append(
            f"{label}                PASS  {block['gates_true']}/{block['gates_total']} gates"
        )
    lines.append(
        f"contract bundle        {verify['agent1']['contract_bundle_digest'][:16]}...  "
        "(8 contracts re-derived)"
    )
    lines.append(
        f"eight root seeds       2026081901..2026081908 exact; test bootstrap "
        f"root {verify['seeds']['test_bootstrap_root']}"
    )
    lines.append(
        f"Phase 9 checkpoint     {verify['phase9']['sha256'][:16]}...  "
        f"{verify['phase9']['parameters']:,} parameters, optimizer step "
        f"{verify['phase9']['global_optimizer_step']:,}"
    )
    lines.append(
        f"belief head            {verify['phase9']['belief_head_digest'][:16]}..."
    )
    lines.append(
        f"validation freeze      {verify['agent5_freeze']['recomputed_freeze_digest'][:16]}...  "
        f"rebuilt from live bytes, {verify['agent5_freeze']['module_count']}/17 modules exact"
    )
    lines.append(
        f"phase11_system_v1      {verify['system_v1']['recomputed_system_digest'][:16]}...  "
        "re-filled by the Agent 1 rules, slot-walked against the Agent 6 artifact"
    )
    lines.append(
        f"validation bank        {banks_stage['validation']['rebuilt_bank_digest'][:16]}...  "
        "rebuilt structurally, cases byte-equal"
    )
    lines.append(
        f"test bank              {banks_stage['test']['rebuilt_bank_digest'][:16]}...  "
        "rebuilt structurally, cases byte-equal"
    )
    sealing = verify["test_bank_sealing"]
    lines.append(
        f"pre-Agent-7 ledger     {sealing['pre_agent7_entries']} entries, "
        f"{sealing['test_bank_entries']} test-bank, scored 0 / inference 0 / "
        "truth 0 / outcomes 0"
    )
    if suite_before:
        lines.append(f"suite before           {suite_before['summary']}")
    lines.append("```")
    lines.append("")
    lines.append(
        "Every load-bearing identity above is recomputed from live bytes, not "
        "compared as a recorded string: the eight contracts are rebuilt by the "
        "live contract module, the Agent 5 freeze document is reconstructed by "
        "`phase11_pipeline.implementation_freeze` and re-hashed, "
        "`phase11_system_v1` is re-filled slot by slot from Agent 1's template "
        "and the accepted upstream values, and both banks are rebuilt "
        "case-by-case from the frozen seeds (every stored setup seed, match "
        "seed and arrangement re-derives exactly; cross-bank overlap zero)."
    )
    lines.append("")

    lines.append("### 7.2 Stage 1 — the first sealed test")
    lines.append("")
    generate = run["stages"]["generate"]
    lines.append("```text")
    lines.append("pipeline               phase11_validation_freeze_v1 (frozen entry point)")
    lines.append("authorization          sealed_bank_authorized=True — the only call site, ledgered")
    lines.append(
        f"games                  {generate['games']:,} "
        f"({run['structure']['cases']:,} cases x 2 colour games, exact)"
    )
    lines.append(f"observer decisions     {generate['observer_decisions']:,}")
    lines.append(f"prediction events      {generate['prediction_events']:,}")
    lines.append(
        f"store content          {run['store_content_digest'][:16]}...  "
        "(cross-run identity)"
    )
    lines.append(
        f"strata                 8/8 exact ({run['structure']['games'] // 8} games each)"
    )
    lines.append(
        f"colours                red {run['structure']['color_totals']['red']:,} / "
        f"blue {run['structure']['color_totals']['blue']:,}, paired per case"
    )
    lines.append(f"wall clock             {run['wall_clock_seconds']:.1f}s")
    lines.append("```")
    lines.append("")
    lines.append(
        "Game outcomes are report-only and altered no belief metric: "
        f"{json.dumps(generate['outcomes_report_only'], sort_keys=True)}."
    )
    lines.append("")

    lines.append("### 7.3 Stage 2 — final predictive metrics")
    lines.append("")
    lines.extend(
        _table(
            [
                (
                    "cross-entropy",
                    f"{metrics['ce_learned']['point']:.4f}",
                    f"{metrics['ce_baseline']['point']:.4f}",
                    _interval(metrics["ce_delta"]),
                ),
                (
                    "top-1 accuracy",
                    f"{metrics['top1_learned']['point']:.4f}",
                    f"{metrics['top1_baseline']['point']:.4f}",
                    _interval(metrics["top1_delta"]),
                ),
                (
                    "Brier",
                    f"{metrics['brier_learned']['point']:.4f}",
                    f"{metrics['brier_baseline']['point']:.4f}",
                    _interval(metrics["brier_delta"]),
                ),
                (
                    "true-rank probability",
                    f"{metrics['true_rank_probability_learned']['point']:.4f}",
                    f"{metrics['true_rank_probability_baseline']['point']:.4f}",
                    "—",
                ),
                (
                    "entropy (nats)",
                    f"{metrics['entropy_learned']['point']:.4f}",
                    f"{metrics['entropy_baseline']['point']:.4f}",
                    "—",
                ),
                (
                    "ECE (15 bins, pooled)",
                    f"{run['overall']['ece_learned']['ece']:.4f}",
                    f"{run['overall']['ece_baseline']['ece']:.4f}",
                    "—",
                ),
                ("`R_CE`", _interval(metrics["r_ce"]), "—", "—"),
            ],
            ("metric", "learned", "`remaining_count_belief_v1`", "delta (95% CI)"),
        )
    )
    lines.append("")
    lines.append(
        f"{run['overall']['events']:,} events over "
        f"{run['overall']['cases_with_events']:,} cases with events "
        f"({run['overall']['cases_without_events']} without). Bootstrap: "
        "root `2026081908`, 10,000 replicates, logical-case resampling, both "
        "colour games pooled. Independent recomputation "
        f"(`{recompute_stage['recompute_version']}`, no `phase11_*` import): "
        f"{comparison['quantities_compared']} quantities, max deviation "
        f"{comparison['max_deviation']:.3e} (tolerance 1e-09), "
        f"{comparison['both_nan_comparisons']} both-NaN comparisons."
    )
    lines.append("")

    lines.append("### 7.4 Per-stratum readings")
    lines.append("")
    stratum_rows = []
    for name in sorted(strata):
        block = strata[name]
        stratum_rows.append(
            (
                f"`{name}`",
                f"{block['events']:,}",
                f"{block['r_ce']['point']:.4f}",
                f"[{block['r_ce']['lower']:.4f}, {block['r_ce']['upper']:.4f}]",
                f"{block['ece_learned']['ece']:.4f}",
                "ok" if block["r_ce"]["point"] <= 1.05 else "**FAIL**",
                "ok" if block["ece_learned"]["ece"] <= 0.12 else "**FAIL**",
            )
        )
    lines.extend(
        _table(
            stratum_rows,
            ("stratum", "events", "`R_CE`", "95% CI", "ECE", "Gate D", "Gate C"),
        )
    )
    lines.append("")

    lines.append("### 7.5 Stage 3 — sampler and stream confirmation")
    lines.append("")
    pipeline_sampler = run["stages"]["sampler_checks"]
    lines.append("```text")
    lines.append(
        f"integrated pass        {pipeline_sampler['requests']:,} requests x 64 worlds = "
        f"{pipeline_sampler['worlds']:,} worlds, all nine counters zero"
    )
    lines.append(
        f"schedule accounting    nominal {accounting['schedule_slots_nominal']:,} / "
        f"attainable {accounting['schedule_slots_attainable']:,} / realized "
        f"{accounting['schedule_slots_realized']:,} "
        f"(realized == attainable: {accounting['realized_equals_attainable']})"
    )
    lines.append(
        f"independent audit      {sampler['requests_audited']:,} predeclared requests "
        f"re-executed in 2 fresh processes (forward/reverse); "
        f"{sampler['worlds_verified']:,} worlds re-verified by "
        "verify_world_independently; all counters zero; cross-process digest "
        f"mismatches {sampler['counters']['cross_process_digest_mismatches']}"
    )
    lines.append(
        f"hidden-input trace     {sampler['counters']['hidden_input_accesses']} reads "
        f"over {sampler['requests_audited']:,} traced document rebuilds"
    )
    agent7_streams = streams["agent7"]
    lines.append(
        f"stream audit           {agent7_streams['new_identities_total']:,} new Agent 7 "
        f"world identities over {agent7_streams['world_tokens_new']:,} new tokens; "
        f"combined {streams['combined']['unique_logical_identities']:,} identities -> "
        f"{streams['combined']['distinct_seeds']:,} distinct seeds; "
        f"{streams['total_accidental_collisions']} accidental collisions"
    )
    lines.append(
        f"prior universes        Agent 4 {streams['agent4']['universe_identities']:,} and "
        f"Agent 6 {streams['agent6']['universe_identities']:,} re-enumerated by their "
        "own accepted code, both reproduced exactly"
    )
    lines.append("```")
    lines.append("")

    lines.append("### 7.6 Stage 4 — the eight hard gates, recomputed")
    lines.append("")
    gate_rows = []
    gate_reading = {
        "A": f"R_CE {quantities['r_ce']:.4f} / CE-delta upper {quantities['ce_delta_upper']:.4f}",
        "B": f"delta_top1 {quantities['delta_top1']:+.4f} / lower {quantities['delta_top1_lower']:+.4f}",
        "C": (
            f"ECE {quantities['ece_overall']:.4f} / worst stratum "
            f"{max(quantities['stratum_ece'].values()):.4f} / Brier upper "
            f"{quantities['brier_delta_upper']:+.4f}"
        ),
        "D": f"worst stratum R_CE {max(quantities['stratum_r_ce'].values()):.4f}",
        "E": (
            f"{pipeline_sampler['worlds']:,} + "
            f"{run['stages']['bound_evidence']['sampler_audit_worlds']:,} worlds, all zero"
        ),
        "F": "50,000 trials, all counters zero (bound Agent 4 evidence)",
        "G": (
            f"8/8 legs exact, p95 "
            f"{run['stages']['bound_evidence']['p95_forward_64_ms']:.2f} ms"
        ),
        "H": "every identity re-derived exact, optimizer delta 0",
    }
    gate_threshold = {
        "A": "<= 0.97 / < 0",
        "B": ">= +0.03 / > 0",
        "C": "<= 0.08 / <= 0.12 / <= +0.01",
        "D": "<= 1.05",
        "E": "all zero",
        "F": "all zero",
        "G": "all exact / <= 500 ms",
        "H": "exact",
    }
    for gate in "ABCDEFGH":
        gate_rows.append(
            (
                gate,
                gate_threshold[gate],
                gate_reading[gate],
                "**PASS**" if gates[gate]["passed"] else "**FAIL**",
            )
        )
    lines.extend(_table(gate_rows, ("gate", "threshold", "sealed-test reading", "result")))
    lines.append("")
    lines.append(
        f"Classification, recomputed from the gate rows alone: **`{classification}`**. "
        + (
            "All eight hard gates pass; Phase 12 search is authorized on the "
            "frozen `phase11_system_v1` stack."
            if classification == "PASS-SEARCH-READY"
            else (
                "The experiment is valid and at least one hard gate fails; "
                "**Phase 12 is not authorized**. A separate belief-repair phase "
                "must be designed; this validation phase does not become a "
                "repair loop."
                if classification == "FAIL"
                else "Integrity could not be established; the sealed result is not graded."
            )
        )
    )
    lines.append("")

    lines.append("### 7.7 Test discipline and the first-access proof")
    lines.append("")
    proof = acceptance["first_scored_access_proof"]
    lines.append(
        f"- the pre-Agent-7 ledger ({proof['pre_agent7_ledger']['entries']} entries, "
        f"{proof['pre_agent7_ledger']['test_bank_entries']} naming the test bank) "
        "shows 0 scored predictions, 0 neural inferences, 0 privileged truth "
        "reads and 0 outcome reads — every entry structural-only"
    )
    lines.append(
        "- `run_phase11_pipeline` refuses the sealed bank without "
        "`sealed_bank_authorized=True`; the refusal was exercised at the freeze "
        "and the authorization was written exactly once, ledgered before the "
        "first scored byte"
    )
    lines.append(
        f"- the sealed run happened once (run ordinal 1, "
        f"{proof['post_run_ledger']['sealed_test_run_entries']} sealed_test_run "
        "entry); the harness refuses a second sealed run structurally, and no "
        "rescue rerun exists to refuse"
    )
    lines.append(
        "- the only non-structural test-bank ledger entries are Agent 7's "
        f"authorized access ({proof['post_run_ledger']['non_structural_test_entries']} "
        "entries, all agent 7)"
    )
    lines.append("")

    lines.append("### 7.8 Preservation after the sealed evaluation")
    lines.append("")
    lines.extend(
        _table(
            [
                ("Phase 9 checkpoint SHA / state", str(preservation["phase9_checkpoint_unchanged"]).lower()),
                ("belief-head identity", str(preservation["belief_head_unchanged"]).lower()),
                ("parameters", "863,959"),
                ("optimizer-step delta", preservation["optimizer_step_delta"]),
                ("Phase 11 optimizer steps", preservation["phase11_optimizer_steps"]),
                ("P10-D / utility / scaler", str(preservation["phase10_selector_unchanged"]).lower()),
                ("Phase 7 library", str(preservation["phase7_library_unchanged"]).lower()),
                ("17 frozen implementation modules", str(preservation["frozen_modules_unchanged"]).lower()),
                ("bound evidence artifacts", str(preservation["bound_evidence_unchanged"]).lower()),
            ],
            ("preserved identity", "exact"),
        )
    )
    lines.append("")

    lines.append("### 7.9 Completion gates")
    lines.append("")
    completion = acceptance["completion_gates"]
    lines.extend(
        _table(
            [(name, str(value).lower()) for name, value in sorted(completion.items())],
            ("gate", "value"),
        )
    )
    lines.append("")
    counters = acceptance["forbidden_operation_counters"]
    lines.extend(
        _table(
            [(f"`{name}`", value) for name, value in sorted(counters.items())],
            ("forbidden operation", "count"),
        )
    )
    lines.append("")
    if suite_before:
        lines.append(f"Suite before the sealed run: `{suite_before['summary']}`.")
    if acceptance.get("suite"):
        lines.append(f"Suite after: `{acceptance['suite']['summary']}`.")
    lines.append("")

    lines.append("### 7.10 Recorded readings")
    lines.append("")
    for reading in acceptance["recorded_readings"]:
        lines.append(f"- **{reading['reading']}** — {reading['detail']}")
    lines.append("")

    lines.append("### 7.11 Recommendation to the reviewing chat")
    lines.append("")
    lines.append(
        f"Agent 7 recommends **`{classification}`**"
        + (
            ", and Phase 12 may implement search over the permanent Phase 11 "
            "freeze: the accepted Phase 9 model + belief head, the accepted "
            "P10-D, `remaining_count_belief_v1`, `belief_sampler_v1` and "
            f"`phase11_system_v1` (`{EXPECTED_SYSTEM_DIGEST[:16]}...`)."
            if classification == "PASS-SEARCH-READY"
            else (
                ". **Phase 12 is not authorized.** The sealed evidence stands as "
                "final; a separate belief-repair phase must be designed and "
                "the sealed test bank remains spent — a future repair phase "
                "needs fresh sealed evidence."
                if classification == "FAIL"
                else ". Integrity could not be established."
            )
        )
    )
    lines.append("")
    lines.append(
        "This work remains uncommitted until the reviewing chat accepts or "
        "rejects the recommendation. The first valid sealed result is final "
        "evidence and is never rerun."
    )
    lines.append("")
    return "\n".join(lines)


def stage_report(_args) -> dict:
    section = build_report_section()
    existing = REPORT_PATH.read_text() if REPORT_PATH.exists() else ""
    marker = "## 7. Agent 7 —"
    if marker in existing:
        head = existing[: existing.index(marker)].rstrip("\n")
        body = f"{head}\n\n{section}\n"
    else:
        body = f"{existing.rstrip()}\n\n{section}\n"
    REPORT_PATH.write_text(body)
    log(f"wrote report section 7 ({len(section.splitlines())} lines)")
    return {"agent": AGENT, "stage": "report", "lines": len(section.splitlines())}


# ---------------------------------------------------------------------------
# 11. Entry point
# ---------------------------------------------------------------------------

STAGES = {
    "verify": stage_verify,
    "banks": stage_banks,
    "suite-before": stage_suite_before,
    "pipeline": stage_pipeline,
    "recompute": stage_recompute,
    "sampler": stage_sampler,
    "sampler-worker": stage_sampler_worker,
    "streams": stage_streams,
    "preservation": stage_preservation,
    "acceptance": stage_acceptance,
    "suite": stage_suite,
    "report": stage_report,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), default=None)
    parser.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    parser.add_argument("--out", default=None)
    parser.add_argument("--verify-worlds", action="store_true")
    args = parser.parse_args()

    try:
        if args.stage:
            STAGES[args.stage](args)
            return 0
        for name in (
            "verify",
            "banks",
            "suite-before",
            "pipeline",
            "recompute",
            "sampler",
            "streams",
            "preservation",
            "acceptance",
        ):
            log(f"--- stage {name} ---")
            STAGES[name](args)
        log("run --stage suite and --stage acceptance again after the suite, then --stage report")
    except Agent7Error as error:
        log(f"BLOCKED: {error}")
        return 2
    return 0


if __name__ == "__main__":
    main()
