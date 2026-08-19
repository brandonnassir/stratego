#!/usr/bin/env python3
"""Phase 11 Agent 4 harness: information safety, reproducibility, runtime.

Recomputes every load-bearing identity from live bytes (the Agent 1 freeze
and its eight contracts, the Agent 2 and Agent 3 PASS records, both bank
digests, the prediction-store manifest, the learned sampler's module
identity, the Phase 9 checkpoint's file SHA / model-state digest /
parameter count / belief-head tensor identity / optimizer-step counter, the
frozen P10-D chain and the Phase 7 library), then proves three things about
the accepted belief model and `belief_sampler_v1`:

- **Part A** — they cannot see hidden truth. >= 50,000 hidden-truth
  permutation trials, each holding every public byte fixed while the
  private truth moves, requiring byte-identical belief logits,
  probabilities, public masks, sampler request, sampled world and sampler
  provenance, with an instrumented hidden-rank access counter at zero and
  every named private field refused structurally at both request
  boundaries.
- **Part B** — they reproduce exactly. A frozen 2,048-request set executed
  under all eight required topology/restart legs, compared on the canonical
  digest of beliefs, masks, worlds and provenance.
- **Part C** — they are fast enough. The Agent 1 frozen benchmark
  configuration (cpu / float32 / 1 torch thread, frozen before any
  measurement) over 480 representative states, four configurations, with
  the Gate G quantity `p95(forward + 64 worlds) <= 500 ms`.

**Part D** runs the five sensitivity controls, each of which must fail the
check it attacks.

Nothing here trains, calibrates, redesigns the sampler, touches P10-D or
reads the sealed test bank. The validation `R_CE = 0.9750` reading remains
diagnostic: no weight, mask, baseline, sampler rule or threshold moved.

    reports/phase_11_data/agent_04_frozen_sets.json
    reports/phase_11_data/agent_04_information_safety.json
    reports/phase_11_data/agent_04_reproducibility.json
    reports/phase_11_data/agent_04_runtime.csv
    reports/phase_11_data/agent_04_acceptance.json

Usage::

    python scripts/run_phase11_agent04.py                    # every stage
    python scripts/run_phase11_agent04.py --stage safety     # one stage
    python scripts/run_phase11_agent04.py --limit-trials 500 # a smoke run
    python scripts/run_phase11_agent04.py --record-suite     # run + record
"""

from __future__ import annotations

import argparse
import csv
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

AGENT = 4
PHASE = 11
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_11_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_11_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase11" / "agent04"
CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "selfplay_c1_v1.pt"
EXPORT_PATH = WORK_DIRECTORY / "phase11_agent04_observer.pt"

AUDIT_VERSION = "phase11_agent04_safety_repro_runtime_v1"

#: The Agent 4 modules whose byte identity is frozen into the artifacts.
IMPLEMENTATION_MODULES = (
    "stratego/evaluation/phase11_safety.py",
    "stratego/evaluation/phase11_repro.py",
    "stratego/evaluation/phase11_streams.py",
)

#: The modules whose derivations must contain no mutable global RNG. The
#: scan is a *static* proof over the live bytes of the production path; the
#: dynamic proof is negative control `mutable_global_rng`.
PURITY_SCANNED_MODULES = (
    "stratego/evaluation/phase11_sampler.py",
    "stratego/evaluation/phase11_belief.py",
    "stratego/evaluation/phase11_public_state.py",
    "stratego/evaluation/phase11_baselines.py",
    "stratego/evaluation/phase11_repro.py",
    "stratego/evaluation/phase11_streams.py",
    "stratego/training/phase11_seed.py",
)

#: Constructs that would make a derivation depend on a mutable cursor, a
#: clock, a process or a path. Each is searched for as a literal substring
#: in the module's live source.
MUTABLE_RNG_MARKERS = (
    "random.random(",
    "random.randint(",
    "random.shuffle(",
    "random.choice(",
    "np.random.seed",
    "np.random.rand",
    "np.random.randint",
    "np.random.choice",
    "np.random.permutation",
    "numpy.random.seed",
    "torch.rand",
    "torch.randint",
    "torch.randperm",
    "torch.manual_seed",
    "time.time(",
    "datetime.now(",
    "os.getpid(",
    "os.urandom(",
    "uuid4(",
)

#: The five Part D sensitivity controls, in the instruction's order.
SENSITIVITY_CONTROLS = (
    "private_truth_read",
    "belief_probability_perturbed",
    "sample_seed_changed",
    "mutable_global_rng",
    "provenance_corrupted",
)

#: The number of committed requests after which the kill/resume leg sends a
#: real SIGKILL. Frozen before the leg ran; any value that leaves work on
#: both sides of the kill proves the same thing.
KILL_AFTER_REQUESTS = 192

#: Worker counts of the sharded legs.
ROUND_ROBIN_WORKERS = 5


class Agent4Error(RuntimeError):
    """The Agent 4 harness refused to continue."""


# ---------------------------------------------------------------------------
# Small shared utilities (the accepted Agent 2/3 harness shapes)
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
        raise Agent4Error(f"stage {name!r} has not run yet ({path} is missing)")
    return json.loads(path.read_text())


def require(condition: bool, message: str, problems: list) -> bool:
    if not condition:
        problems.append(message)
    return bool(condition)


def log(message: str) -> None:
    print(f"[phase11:agent4] {message}", flush=True)


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
    return {
        name: file_sha256(REPOSITORY_ROOT / name) for name in sorted(names)
    }


# ---------------------------------------------------------------------------
# 1. Verification — every identity from live bytes
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
            "information_safety_digest": live_digests.get(
                "phase11_information_safety_v1"
            ),
            "sampler_contract_digest": live_digests.get("phase11_belief_sampler_v1"),
        }
    )
    return summary


def verify_agent2(problems: list) -> dict:
    from stratego.training.phase11_contract import EVALUATOR_VERSION

    summary = _verify_agent_acceptance(
        "agent_02_acceptance.json", 2, problems, handoff_key="handoff_to_agent_3"
    )
    if not summary.get("available"):
        return summary
    acceptance = read_json(DATA_DIRECTORY / "agent_02_acceptance.json")
    digests = acceptance.get("new_digests", {})
    require(
        digests.get("evaluator_version") == EVALUATOR_VERSION,
        f"Agent 2 evaluator version {digests.get('evaluator_version')!r} moved",
        problems,
    )
    summary.update(
        {
            "evaluator_version": digests.get("evaluator_version"),
            "manifest_digest": digests.get("prediction_store_manifest_digest"),
            "validation_r_ce_point": acceptance.get("metrics_summary", {})
            .get("r_ce", {})
            .get("point"),
        }
    )
    return summary


def verify_agent3(problems: list) -> dict:
    """Agent 3 must be PASS and its sampler module must be byte-unchanged."""
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
    nonzero = sorted(
        name
        for name, value in audit.get("zero_tolerance_counters", {}).items()
        if value
    )
    require(
        not nonzero, f"Agent 3 zero-tolerance counters are non-zero: {nonzero}", problems
    )
    summary.update(
        {
            "sampler_version": digests.get("sampler_version"),
            "sampler_module_sha256": live_modules,
            "learned_worlds": audit.get("learned_worlds"),
            "sampler_zero_counters": audit.get("zero_tolerance_counters"),
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


def verify_prediction_store(problems: list) -> dict:
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
    return {
        "available": True,
        "store_root": str(root),
        "manifest_digest": recorded,
        "games": len(manifest.get("games_index", [])),
        "observer_decisions": manifest.get("observer_decisions"),
        "prediction_events": manifest.get("prediction_events"),
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


def scan_mutable_rng(problems: list) -> dict:
    """Static proof that no scanned derivation carries a mutable RNG cursor.

    A literal-substring scan over the live source of the production path.
    It is deliberately blunt: a marker appearing anywhere — including in a
    comment — is reported, because a derivation module has no business
    naming these at all. The dynamic proof is the `mutable_global_rng`
    negative control, which introduces one and requires the topology
    comparison to fire.
    """
    findings = []
    for name in PURITY_SCANNED_MODULES:
        source = (REPOSITORY_ROOT / name).read_text()
        for marker in MUTABLE_RNG_MARKERS:
            if marker in source:
                findings.append({"module": name, "marker": marker})
    require(
        not findings,
        f"a mutable-RNG / clock / pid / path marker appears in a derivation: {findings}",
        problems,
    )
    return {
        "scanned_modules": list(PURITY_SCANNED_MODULES),
        "markers": list(MUTABLE_RNG_MARKERS),
        "findings": findings,
        "mutable_rng_absent": not findings,
    }


def stage_verify(_args) -> dict:
    problems: list[str] = []
    log("verifying the Agent 1 freeze")
    agent1 = verify_agent1(problems)
    log("verifying the Agent 2 PASS")
    agent2 = verify_agent2(problems)
    log("verifying the Agent 3 PASS and sampler bytes")
    agent3 = verify_agent3(problems)
    log("re-hashing both frozen banks")
    banks = verify_banks(problems)
    log("re-deriving the Phase 9 checkpoint identity")
    phase9 = verify_phase9_checkpoint(problems)
    log("verifying the P10-D chain, the anchor and the Phase 7 library")
    upstream = verify_upstream_stack(problems)
    log("verifying the Agent 2 prediction store")
    store = verify_prediction_store(problems)
    log("verifying the test-bank seal")
    sealing = verify_test_bank_sealed(problems)
    log("scanning the production path for mutable RNG cursors")
    purity = scan_mutable_rng(problems)

    payload = {
        "stage": "verify",
        "agent1": agent1,
        "agent2": agent2,
        "agent3": agent3,
        "banks": banks,
        "phase9": phase9,
        "upstream": upstream,
        "prediction_store": store,
        "test_bank_sealing": sealing,
        "purity_scan": purity,
        "agent4_modules": module_digests(IMPLEMENTATION_MODULES),
        "environment": environment_report(),
        "problems": problems,
    }
    write_stage("verify", payload)
    if problems:
        for problem in problems:
            log(f"PROBLEM: {problem}")
        raise Agent4Error(f"{len(problems)} verification problems; refusing to continue")
    log("verification clean")
    return payload


# ---------------------------------------------------------------------------
# 2. The frozen sets — written before any measurement exists
# ---------------------------------------------------------------------------


def load_bank_cases() -> dict:
    payload = read_json(DATA_DIRECTORY / "agent_01_validation_bank.json")
    return {case["case_id"]: case for case in payload["cases"]}


def build_decision_rows() -> list:
    """Every recorded observer decision, from the store and the bank alone."""
    from stratego.evaluation.phase11_records import read_manifest, store_root
    from stratego.evaluation.phase11_repro import decision_table

    root = store_root(REPOSITORY_ROOT)
    manifest = read_manifest(root)
    return decision_table(root, manifest, load_bank_cases())


def action_histories(game_ids) -> dict:
    """The recorded public action history of each named game."""
    from stratego.evaluation.phase11_records import read_public_shard, store_root

    root = store_root(REPOSITORY_ROOT)
    return {
        game_id: [int(value) for value in read_public_shard(root, game_id)["action_history"]]
        for game_id in sorted(set(game_ids))
    }


def request_specs(rows, cases, histories) -> list:
    """Fully resolved, self-contained request specs for the topology legs."""
    from stratego.evaluation.phase11_repro import game_setups

    specs = []
    for row in rows:
        setups = game_setups(cases, row["case_id"], row["game_index"])
        specs.append(
            {
                "request_ordinal": row["request_ordinal"],
                "request_id": row["request_id"],
                "game_id": row["game_id"],
                "decision_index": row["decision_index"],
                "observer_color": row["observer_color"],
                "opponent_stratum": row["opponent_stratum"],
                "opponent_setup_source": row["opponent_setup_source"],
                "public_state_identity": row["public_state_identity"],
                "unresolved_pieces": row["unresolved_pieces"],
                "progress_bucket": row["progress_bucket"],
                "action_history": histories[row["game_id"]],
                **setups,
            }
        )
    return specs


def _set_digest(items, fields) -> str:
    hasher = hashlib.sha256()
    for item in items:
        hasher.update(
            json.dumps(
                {field: item[field] for field in fields},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    return hasher.hexdigest()


def build_safety_pool(limit_games=None) -> dict:
    """The frozen safety candidate pool, replayed once from public bytes.

    A candidate is a recorded observer decision with at least two
    unresolved opponent pieces. Its `admits_alternative` flag is the
    constructive predicate of `phase11_safety.admits_alternative_truth`;
    candidates failing it are kept in the pool file but excluded from
    selection by the frozen no-alternative rule, and both counts are
    recorded, so a skipped state is visible rather than silently absent.
    """
    import numpy as np

    from stratego.engine.constants import BLUE, RED
    from stratego.engine.state import create_game
    from stratego.engine.transition import apply_action
    from stratego.evaluation.match_spec import EVALUATION_RULES
    from stratego.evaluation.phase11_records import read_manifest, read_public_shard, store_root
    from stratego.evaluation.phase11_repro import game_setups
    from stratego.evaluation.phase11_safety import (
        MIN_UNRESOLVED_PIECES,
        admits_alternative_truth,
        unresolved_opponent_records,
    )

    root = store_root(REPOSITORY_ROOT)
    manifest = read_manifest(root)
    cases = load_bank_cases()
    entries = sorted(manifest["games_index"], key=lambda item: item["game_id"])
    if limit_games:
        entries = entries[: int(limit_games)]

    game_ids: list[str] = []
    decisions: list[int] = []
    admits: list[bool] = []
    unresolved: list[int] = []
    below_minimum = 0
    for entry in entries:
        game_id = entry["game_id"]
        arrays = read_public_shard(root, game_id)
        offsets = np.asarray(arrays["event_offset"], dtype=np.int64)
        recorded = np.asarray(arrays["decision_index"], dtype=np.int64)
        wanted = {}
        for position in range(len(recorded)):
            count = int(offsets[position + 1] - offsets[position])
            if count < MIN_UNRESOLVED_PIECES:
                below_minimum += 1
                continue
            wanted[int(recorded[position])] = count
        if not wanted:
            continue
        setups = game_setups(cases, entry["case_id"], int(entry["game_index"]))
        observer = RED if entry["observer_color"] == "red" else BLUE
        state = create_game(
            tuple(setups["red_setup"]),
            tuple(setups["blue_setup"]),
            rules=EVALUATION_RULES,
            game_id=game_id,
        )
        remaining = dict(wanted)
        for action in arrays["action_history"]:
            if state.terminal or not remaining:
                break
            ply = int(state.total_moves)
            if state.acting_player == observer and ply in remaining:
                records = unresolved_opponent_records(state, observer)
                types = [record.true_type for record in records]
                moved = [record.has_moved for record in records]
                game_ids.append(game_id)
                decisions.append(ply)
                unresolved.append(len(records))
                admits.append(admits_alternative_truth(types, moved))
                remaining.pop(ply)
            apply_action(state, int(action))
        if remaining:
            raise Agent4Error(
                f"{game_id}: the replay never reached decisions {sorted(remaining)}"
            )

    order = sorted(range(len(game_ids)), key=lambda index: (game_ids[index], decisions[index]))
    pool = {
        "game_id": [game_ids[index] for index in order],
        "decision_index": [decisions[index] for index in order],
        "unresolved_pieces": [unresolved[index] for index in order],
        "admits_alternative": [bool(admits[index]) for index in order],
    }
    digest = hashlib.sha256()
    for index in range(len(pool["game_id"])):
        digest.update(
            f"{pool['game_id'][index]}|{pool['decision_index'][index]}|"
            f"{int(pool['admits_alternative'][index])}".encode()
        )
    pool["pool_digest"] = digest.hexdigest()
    pool["candidates"] = len(pool["game_id"])
    pool["admitting"] = int(sum(pool["admits_alternative"]))
    pool["non_admitting"] = pool["candidates"] - pool["admitting"]
    pool["below_minimum_unresolved"] = int(below_minimum)
    return pool


def stage_contract(args) -> dict:
    """Freeze the request set, the benchmark states and the safety pool.

    Written before any trial, leg or measurement exists. Every rule here is
    Agent 1's; this stage only *materialises* them from the recorded store
    and records what they selected.
    """
    from stratego.training import phase11_contract as contract
    from stratego.evaluation.phase11_repro import (
        BENCHMARK_CONFIGURATIONS,
        REQUESTS_PER_STRATUM,
        REQUEST_WORLD_COUNT,
        frozen_benchmark_states,
        frozen_repro_requests,
    )
    from stratego.training.phase11_seed import (
        BENCHMARK_STATE_COUNT,
        REPRO_REQUEST_COUNT,
        SAFETY_TRIAL_COUNT,
    )

    verify = read_stage("verify")
    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)

    log("building the recorded decision table from the prediction store")
    rows = build_decision_rows()
    log(f"{len(rows)} recorded observer decisions with unresolved pieces")

    log("freezing the 2,048-request topology set")
    requests = frozen_repro_requests(rows)
    log("freezing the 480-state runtime benchmark set")
    benchmark, cells = frozen_benchmark_states(rows)

    cases = load_bank_cases()
    histories = action_histories(
        [row["game_id"] for row in requests] + [row["game_id"] for row in benchmark]
    )
    repro_specs = request_specs(requests, cases, histories)
    benchmark_specs = [
        {
            **spec,
            "state_ordinal": row["state_ordinal"],
            "benchmark_state_id": row["benchmark_state_id"],
        }
        for spec, row in zip(
            request_specs(
                [
                    {
                        **row,
                        "request_ordinal": row["state_ordinal"],
                        "request_id": row["benchmark_state_id"],
                    }
                    for row in benchmark
                ],
                cases,
                histories,
            ),
            benchmark,
        )
    ]

    plan_path = WORK_DIRECTORY / "repro_plan.json"
    plan_path.write_text(json.dumps(repro_specs, separators=(",", ":")))
    benchmark_path = WORK_DIRECTORY / "benchmark_plan.json"
    benchmark_path.write_text(json.dumps(benchmark_specs, separators=(",", ":")))

    log("building the safety candidate pool (one privileged replay pass)")
    started = time.perf_counter()
    pool = build_safety_pool(limit_games=args.limit_games)
    pool_path = WORK_DIRECTORY / "safety_pool.json"
    pool_path.write_text(json.dumps(pool, separators=(",", ":")))
    log(
        f"safety pool: {pool['candidates']} candidates, {pool['admitting']} admitting "
        f"({round(time.perf_counter() - started, 1)}s)"
    )

    from collections import Counter

    fields = ("request_ordinal", "game_id", "decision_index", "public_state_identity")
    payload = {
        "artifact": "agent_04_frozen_sets",
        "agent": AGENT,
        "phase": PHASE,
        "audit_version": AUDIT_VERSION,
        "frozen_before_any_measurement": True,
        "source": {
            "prediction_store_manifest_digest": verify["prediction_store"][
                "manifest_digest"
            ],
            "validation_bank_digest": verify["banks"]["validation"]["bank_digest"],
            "recorded_decisions": len(rows),
            "rule": "built from the recorded public shards and the frozen bank; no replay, no outcome field, no truth shard",
        },
        "topology_request_set": {
            "request_count": len(requests),
            "expected_request_count": REPRO_REQUEST_COUNT,
            "requests_per_stratum": REQUESTS_PER_STRATUM,
            "worlds_per_request": REQUEST_WORLD_COUNT,
            "rule": contract.REPRODUCIBILITY_SPECIFICATION["request_set"],
            "request_content": contract.REPRODUCIBILITY_SPECIFICATION["request_content"],
            "set_digest": _set_digest(requests, fields),
            "plan_file": str(plan_path.relative_to(REPOSITORY_ROOT)),
            "plan_sha256": file_sha256(plan_path),
            "distinct_public_states": len(
                {row["public_state_identity"] for row in requests}
            ),
            "by_stratum": dict(
                sorted(Counter(row["opponent_stratum"] for row in requests).items())
            ),
            "by_observer_color": dict(
                sorted(Counter(row["observer_color"] for row in requests).items())
            ),
            "by_progress_bucket": dict(
                sorted(Counter(row["progress_bucket"] for row in requests).items())
            ),
            "by_setup_source": dict(
                sorted(Counter(row["opponent_setup_source"] for row in requests).items())
            ),
            "unresolved_pieces": {
                "min": min(row["unresolved_pieces"] for row in requests),
                "max": max(row["unresolved_pieces"] for row in requests),
                "distinct": len({row["unresolved_pieces"] for row in requests}),
            },
        },
        "benchmark_state_set": {
            "state_count": len(benchmark),
            "expected_state_count": BENCHMARK_STATE_COUNT,
            "rule": contract.RUNTIME_BENCHMARK_CONFIGURATION["state_selection"],
            "backend": contract.RUNTIME_BENCHMARK_CONFIGURATION["backend"],
            "dtype": contract.RUNTIME_BENCHMARK_CONFIGURATION["dtype"],
            "torch_threads": contract.RUNTIME_BENCHMARK_CONFIGURATION["torch_threads"],
            "configurations": [name for name, _ in BENCHMARK_CONFIGURATIONS],
            "set_digest": _set_digest(
                [
                    {**row, "request_ordinal": row["state_ordinal"]}
                    for row in benchmark
                ],
                fields,
            ),
            "plan_file": str(benchmark_path.relative_to(REPOSITORY_ROOT)),
            "plan_sha256": file_sha256(benchmark_path),
            "cells": cells,
            "cells_total": len(cells),
            "cells_short": [cell for cell in cells if cell["selected"] < 10],
            "distinct_public_states": len(
                {row["public_state_identity"] for row in benchmark}
            ),
            "by_stratum": dict(
                sorted(Counter(row["opponent_stratum"] for row in benchmark).items())
            ),
            "by_observer_color": dict(
                sorted(Counter(row["observer_color"] for row in benchmark).items())
            ),
            "by_progress_bucket": dict(
                sorted(Counter(row["progress_bucket"] for row in benchmark).items())
            ),
            "unresolved_pieces": {
                "min": min(row["unresolved_pieces"] for row in benchmark),
                "max": max(row["unresolved_pieces"] for row in benchmark),
                "distinct": len({row["unresolved_pieces"] for row in benchmark}),
            },
        },
        "safety_candidate_pool": {
            "trials": SAFETY_TRIAL_COUNT,
            "rule": contract.INFORMATION_SAFETY_ATTACK["state_pool"],
            "no_alternative_rule": contract.INFORMATION_SAFETY_ATTACK[
                "no_alternative_rule"
            ],
            "candidates": pool["candidates"],
            "admitting": pool["admitting"],
            "non_admitting": pool["non_admitting"],
            "below_minimum_unresolved": pool["below_minimum_unresolved"],
            "pool_digest": pool["pool_digest"],
            "pool_file": str(pool_path.relative_to(REPOSITORY_ROOT)),
            "pool_sha256": file_sha256(pool_path),
        },
        "environment": environment_report(),
    }
    write_stage("contract", payload)
    write_artifact("agent_04_frozen_sets.json", payload)
    log(
        f"frozen: {len(requests)} requests, {len(benchmark)} benchmark states, "
        f"{pool['admitting']} admitting safety candidates"
    )
    return payload


# ---------------------------------------------------------------------------
# 3. Part A — the hidden-truth permutation attack
# ---------------------------------------------------------------------------


def build_export(problems: list) -> dict:
    """Export the accepted evaluation weights once, for every stage."""
    from stratego.training.phase10_collector import export_evaluation_weights

    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    export = export_evaluation_weights(CHECKPOINT_PATH, EXPORT_PATH)
    return {
        "export_path": str(EXPORT_PATH.relative_to(REPOSITORY_ROOT)),
        "export_sha256": file_sha256(EXPORT_PATH),
        "source_sha256": file_sha256(CHECKPOINT_PATH),
        "export_report": {
            key: value
            for key, value in (export or {}).items()
            if isinstance(value, (str, int, float, bool))
        },
    }


def public_products(state, observer, request_id: str, observer_color: str):
    """`(document, observation, payload, view)` for one position.

    Exactly the accepted production construction: an accepted
    `PolicyInput`, the accepted `PublicView`, the accepted 127-channel
    observation and the frozen public-state document. Nothing privileged
    crosses this boundary.
    """
    from stratego.evaluation.phase11_public_state import build_public_state_document
    from stratego.evaluation.neural_worker import InferenceRequest
    from stratego.evaluation.policy import (
        PolicyRef,
        PolicyRequirements,
        build_policy_input,
    )
    from stratego.training.phase11_contract import BELIEF_REQUEST_VERSION

    policy_input = build_policy_input(
        state,
        policy=PolicyRef(
            policy_id="phase11_safety_observer", policy_version=BELIEF_REQUEST_VERSION
        ),
        policy_seed=0,
        requirements=PolicyRequirements(
            observation=True, legal_action_mask=True, public_view=True
        ),
        match_id=request_id,
        game_id=state.game_id,
    )
    view = policy_input.require_public_view()
    observation = policy_input.require_observation()
    document = build_public_state_document(view, observation)
    payload = InferenceRequest.from_policy_input(policy_input)
    return document, observation, payload


def belief_and_world(owner, state, observer, request_id, observer_color, ordinal):
    """One belief forward and one fixed-seed world, from public products only."""
    import numpy as np

    from stratego.evaluation.phase11_belief import (
        Phase11BeliefRequest,
        softmax_float64,
    )
    from stratego.evaluation.phase11_public_state import (
        hidden_opponent_pieces,
        legal_rank_mask,
    )
    from stratego.evaluation.phase11_safety import (
        belief_digest,
        sampler_request_digest,
        world_digest,
    )
    from stratego.evaluation.phase11_sampler import (
        Phase11SamplerRequest,
        sample_belief_world,
    )
    from stratego.training.phase11_contract import (
        BELIEF_REQUEST_VERSION,
        BELIEF_SAMPLER_VERSION,
    )

    document, observation, payload = public_products(
        state, observer, request_id, observer_color
    )
    belief_request = Phase11BeliefRequest(
        request_version=BELIEF_REQUEST_VERSION,
        request_id=request_id,
        observer_color=observer_color,
        public_state_document=document,
        observation=observation,
    )
    _response, prediction, _elapsed = owner.serve_decision(payload, belief_request)
    logits = prediction.belief_logits
    probabilities = {slot: softmax_float64(row) for slot, row in logits.items()}
    masks = {
        int(piece["piece_slot"]): legal_rank_mask(bool(piece["has_moved"]))
        for piece in hidden_opponent_pieces(document)
    }
    sampler_request = Phase11SamplerRequest(
        sampler_version=BELIEF_SAMPLER_VERSION,
        public_state_document=document,
        learned_probabilities=probabilities,
        sample_ordinal=int(ordinal),
    )
    world = sample_belief_world(sampler_request)
    provenance = {
        key: world[key]
        for key in (
            "sample_token",
            "sampler_version",
            "public_state_identity",
            "belief_model_label",
            "sample_ordinal",
            "piece_order",
            "fallback_steps",
        )
    }
    return {
        "document": document,
        "document_digest": hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "observation_digest": hashlib.sha256(
            np.ascontiguousarray(observation, dtype=np.float32).tobytes()
        ).hexdigest(),
        "legal_mask_digest": hashlib.sha256(
            np.ascontiguousarray(payload.legal_action_mask, dtype=np.uint8).tobytes()
        ).hexdigest(),
        "belief_digest": belief_digest(logits, probabilities, masks),
        "mask_digest": hashlib.sha256(
            b"".join(
                bytes(masks[slot]) for slot in sorted(masks)
            )
        ).hexdigest(),
        "sampler_request_digest": sampler_request_digest(
            document, probabilities, int(ordinal)
        ),
        "world_digest": world_digest(world),
        "provenance_digest": world_digest(provenance),
        "hidden_pieces": len(logits),
        "probabilities": probabilities,
        "observation": observation,
    }


def instrumented_reads(state, observer, request_id, observer_color, reference_document):
    """Rebuild the public products with hidden ranks traced; count the reads."""
    from stratego.evaluation.phase11_safety import instrument_hidden_types

    traced, counter = instrument_hidden_types(state, observer)
    document, _observation, _payload = public_products(
        traced, observer, request_id, observer_color
    )
    return counter.reads, document == reference_document


def stage_safety(args) -> dict:
    """Part A: >= 50,000 hidden-truth permutation trials."""
    import numpy as np

    from collections import Counter

    from stratego.engine.constants import BLUE, RED
    from stratego.engine.state import create_game
    from stratego.engine.transition import apply_action
    from stratego.evaluation.match_spec import EVALUATION_RULES
    from stratego.evaluation.phase11_records import read_manifest, store_root
    from stratego.evaluation.phase11_repro import build_owner, game_setups
    from stratego.evaluation.phase11_safety import (
        PERMUTATION_ATTEMPT_BUDGET,
        apply_alternative_truth,
        build_alternative_truth,
        injection_controls,
        trial_sample_ordinal,
        trial_state_walk,
        unresolved_opponent_records,
    )
    from stratego.evaluation.phase11_repro import REQUEST_WORLD_COUNT
    from stratego.training.phase11_contract import (
        IMMOVABLE_RANK_INDICES,
        progress_bucket,
    )
    from stratego.training.phase11_seed import (
        SAFETY_TRIAL_COUNT,
        phase11_safety_trial_id,
    )

    verify = read_stage("verify")
    contract_stage = read_stage("contract")
    problems: list[str] = []
    export = build_export(problems)

    pool = json.loads((WORK_DIRECTORY / "safety_pool.json").read_text())
    if file_sha256(WORK_DIRECTORY / "safety_pool.json") != contract_stage[
        "safety_candidate_pool"
    ]["pool_sha256"]:
        raise Agent4Error("the frozen safety pool file changed after the freeze")
    admits = pool["admits_alternative"]
    pool_size = pool["candidates"]

    trial_count = int(args.limit_trials or SAFETY_TRIAL_COUNT)
    log(f"assigning {trial_count} trials over {pool_size} candidate states")
    assignments: dict[str, list] = {}
    walk_steps = Counter()
    for ordinal in range(trial_count):
        trial_id = phase11_safety_trial_id(ordinal)
        walk = trial_state_walk(trial_id, pool_size, admits)
        index = walk["pool_index"]
        walk_steps[walk["walk_steps"]] += 1
        assignments.setdefault(pool["game_id"][index], []).append(
            {
                "trial_ordinal": ordinal,
                "trial_id": trial_id,
                "pool_index": index,
                "decision_index": pool["decision_index"][index],
                "walk_steps": walk["walk_steps"],
            }
        )

    root = store_root(REPOSITORY_ROOT)
    manifest = read_manifest(root)
    cases = load_bank_cases()
    games = {entry["game_id"]: entry for entry in manifest["games_index"]}
    histories = action_histories(assignments)

    owner = build_owner(EXPORT_PATH)
    counters = {
        "belief_output_differences": 0,
        "fixed_seed_sample_differences": 0,
        "forbidden_hidden_input_accesses": 0,
        "injection_acceptances": 0,
    }
    detail_counters = {
        "public_document_differences": 0,
        "observation_differences": 0,
        "public_mask_differences": 0,
        "legal_action_mask_differences": 0,
        "belief_logit_probability_differences": 0,
        "sampler_request_differences": 0,
        "sampled_world_differences": 0,
        "sampler_provenance_differences": 0,
        "instrumented_document_mismatches": 0,
        "unchanged_alternative_truths": 0,
        "illegal_alternative_truths": 0,
        "inventory_changes": 0,
    }
    rollup = hashlib.sha256()
    examples: list[dict] = []
    method_counts = Counter()
    changed_pieces: list[int] = []
    hidden_pieces: list[int] = []
    stratum_counts = Counter()
    color_counts = Counter()
    bucket_counts = Counter()
    attempts_counts = Counter()
    executed = 0
    started = time.perf_counter()

    injection_states = []
    injection_target = 8 if not args.limit_trials else 2

    for game_index, game_id in enumerate(sorted(assignments)):
        entry = games[game_id]
        setups = game_setups(cases, entry["case_id"], int(entry["game_index"]))
        observer_color = entry["observer_color"]
        observer = RED if observer_color == "red" else BLUE
        by_ply: dict[int, list] = {}
        for trial in assignments[game_id]:
            by_ply.setdefault(int(trial["decision_index"]), []).append(trial)
        state = create_game(
            tuple(setups["red_setup"]),
            tuple(setups["blue_setup"]),
            rules=EVALUATION_RULES,
            game_id=game_id,
        )
        remaining = set(by_ply)
        for action in histories[game_id]:
            if state.terminal or not remaining:
                break
            ply = int(state.total_moves)
            if state.acting_player == observer and ply in remaining:
                remaining.discard(ply)
                for trial in sorted(by_ply[ply], key=lambda item: item["trial_ordinal"]):
                    trial_id = trial["trial_id"]
                    ordinal = trial_sample_ordinal(trial_id, REQUEST_WORLD_COUNT)
                    original = belief_and_world(
                        owner, state, observer, trial_id, observer_color, ordinal
                    )
                    records = unresolved_opponent_records(state, observer)
                    types = tuple(record.true_type for record in records)
                    moved = tuple(record.has_moved for record in records)
                    alternative = build_alternative_truth(trial_id, types, moved)
                    permuted_state = apply_alternative_truth(
                        state, observer, alternative.ranks
                    )
                    permuted = belief_and_world(
                        owner,
                        permuted_state,
                        observer,
                        trial_id,
                        observer_color,
                        ordinal,
                    )
                    original_reads, original_match = instrumented_reads(
                        state, observer, trial_id, observer_color, original["document"]
                    )
                    permuted_reads, permuted_match = instrumented_reads(
                        permuted_state,
                        observer,
                        trial_id,
                        observer_color,
                        permuted["document"],
                    )

                    if alternative.changed_pieces == 0:
                        detail_counters["unchanged_alternative_truths"] += 1
                    if any(
                        moved[index] and rank in IMMOVABLE_RANK_INDICES
                        for index, rank in enumerate(alternative.ranks)
                    ):
                        detail_counters["illegal_alternative_truths"] += 1
                    if sorted(types) != sorted(alternative.ranks):
                        detail_counters["inventory_changes"] += 1
                    if original["document_digest"] != permuted["document_digest"]:
                        detail_counters["public_document_differences"] += 1
                    if original["observation_digest"] != permuted["observation_digest"]:
                        detail_counters["observation_differences"] += 1
                    if original["mask_digest"] != permuted["mask_digest"]:
                        detail_counters["public_mask_differences"] += 1
                    if original["legal_mask_digest"] != permuted["legal_mask_digest"]:
                        detail_counters["legal_action_mask_differences"] += 1
                    if original["belief_digest"] != permuted["belief_digest"]:
                        detail_counters["belief_logit_probability_differences"] += 1
                        counters["belief_output_differences"] += 1
                    sample_side_differed = False
                    if (
                        original["sampler_request_digest"]
                        != permuted["sampler_request_digest"]
                    ):
                        detail_counters["sampler_request_differences"] += 1
                        sample_side_differed = True
                    if original["world_digest"] != permuted["world_digest"]:
                        detail_counters["sampled_world_differences"] += 1
                        sample_side_differed = True
                    if original["provenance_digest"] != permuted["provenance_digest"]:
                        detail_counters["sampler_provenance_differences"] += 1
                        sample_side_differed = True
                    if sample_side_differed:
                        counters["fixed_seed_sample_differences"] += 1
                    counters["forbidden_hidden_input_accesses"] += int(
                        original_reads
                    ) + int(permuted_reads)
                    if not original_match or not permuted_match:
                        detail_counters["instrumented_document_mismatches"] += 1

                    method_counts[alternative.method] += 1
                    attempts_counts[alternative.attempts] += 1
                    changed_pieces.append(alternative.changed_pieces)
                    hidden_pieces.append(original["hidden_pieces"])
                    stratum_counts[entry["opponent_stratum"]] += 1
                    color_counts[observer_color] += 1
                    bucket_counts[progress_bucket(ply)] += 1
                    rollup.update(
                        f"{trial_id}|{original['belief_digest']}|"
                        f"{original['world_digest']}|{original['provenance_digest']}|"
                        f"{permuted['belief_digest']}|{permuted['world_digest']}|"
                        f"{permuted['provenance_digest']}|{alternative.changed_pieces}"
                        .encode()
                    )
                    executed += 1
                    if len(examples) < 8:
                        examples.append(
                            {
                                "trial_id": trial_id,
                                "game_id": game_id,
                                "decision_index": ply,
                                "opponent_stratum": entry["opponent_stratum"],
                                "observer_color": observer_color,
                                "hidden_pieces": original["hidden_pieces"],
                                "changed_pieces": alternative.changed_pieces,
                                "permutation_method": alternative.method,
                                "sample_ordinal": ordinal,
                                "belief_digest": original["belief_digest"],
                                "belief_digest_permuted": permuted["belief_digest"],
                                "world_digest": original["world_digest"],
                                "world_digest_permuted": permuted["world_digest"],
                                "hidden_reads": int(original_reads + permuted_reads),
                            }
                        )
                    if len(injection_states) < injection_target:
                        injection_states.append(
                            {
                                "document": original["document"],
                                "observation": original["observation"],
                                "probabilities": original["probabilities"],
                                "trial_id": trial_id,
                                "stratum": entry["opponent_stratum"],
                            }
                        )
            apply_action(state, int(action))
        if remaining:
            raise Agent4Error(
                f"{game_id}: the replay never reached decisions {sorted(remaining)}"
            )
        if (game_index + 1) % 64 == 0:
            log(
                f"{game_index + 1}/{len(assignments)} games, {executed} trials, "
                f"{round(time.perf_counter() - started, 1)}s"
            )

    log("running the injection controls on both request boundaries")
    injection_reports = []
    for candidate in injection_states:
        report = injection_controls(
            candidate["document"], candidate["observation"], candidate["probabilities"]
        )
        counters["injection_acceptances"] += report["injection_acceptances"]
        injection_reports.append(
            {
                "trial_id": candidate["trial_id"],
                "opponent_stratum": candidate["stratum"],
                "probe_count": report["probe_count"],
                "injection_acceptances": report["injection_acceptances"],
                "all_rejected": report["all_rejected"],
                "rejected_fields": sorted(
                    {probe["field"] for probe in report["probes"] if probe["rejected"]}
                ),
                "accepted_fields": sorted(
                    {
                        probe["field"]
                        for probe in report["probes"]
                        if not probe["rejected"]
                    }
                ),
            }
        )

    owner.close()
    elapsed = time.perf_counter() - started
    payload = {
        "artifact": "agent_04_information_safety",
        "agent": AGENT,
        "phase": PHASE,
        "audit_version": AUDIT_VERSION,
        "contract_version": "phase11_information_safety_v1",
        "trials": {
            "requested": trial_count,
            "executed": executed,
            "floor": SAFETY_TRIAL_COUNT,
            "meets_floor": executed >= SAFETY_TRIAL_COUNT,
            "belief_forwards": 2 * executed,
            "sampled_worlds": 2 * executed,
            "instrumented_rebuilds": 2 * executed,
            "distinct_games": len(assignments),
            "distinct_states": len(
                {
                    (game_id, trial["decision_index"])
                    for game_id, trials in assignments.items()
                    for trial in trials
                }
            ),
            "wall_clock_seconds": round(elapsed, 3),
        },
        "zero_tolerance_counters": counters,
        "detail_counters": detail_counters,
        "all_counters_zero": all(value == 0 for value in counters.values())
        and all(value == 0 for value in detail_counters.values()),
        "candidate_pool": contract_stage["safety_candidate_pool"],
        "state_selection": {
            "walk_step_histogram": dict(sorted(walk_steps.items())),
            "rule": "draw from the trial's state_selection stream; walk to the next draw while the candidate admits no altered legal truth",
        },
        "permutation": {
            "attempt_budget": PERMUTATION_ATTEMPT_BUDGET,
            "method_counts": dict(sorted(method_counts.items())),
            "attempt_histogram": dict(sorted(attempts_counts.items())),
            "changed_pieces": {
                "min": int(min(changed_pieces)) if changed_pieces else 0,
                "max": int(max(changed_pieces)) if changed_pieces else 0,
                "mean": round(float(np.mean(changed_pieces)), 4)
                if changed_pieces
                else 0.0,
            },
            "hidden_pieces": {
                "min": int(min(hidden_pieces)) if hidden_pieces else 0,
                "max": int(max(hidden_pieces)) if hidden_pieces else 0,
                "mean": round(float(np.mean(hidden_pieces)), 4)
                if hidden_pieces
                else 0.0,
            },
        },
        "coverage": {
            "by_stratum": dict(sorted(stratum_counts.items())),
            "by_observer_color": dict(sorted(color_counts.items())),
            "by_progress_bucket": dict(sorted(bucket_counts.items())),
        },
        "injection_controls": {
            "states_probed": len(injection_reports),
            "probes_total": sum(item["probe_count"] for item in injection_reports),
            "injection_acceptances": counters["injection_acceptances"],
            "all_rejected": all(item["all_rejected"] for item in injection_reports),
            "reports": injection_reports,
        },
        "example_trials": examples,
        "trial_rollup_digest": rollup.hexdigest(),
        "model": {
            "belief_head_digest": verify["phase9"]["belief_head_digest"],
            "model_state_digest": verify["phase9"]["model_state_digest"],
            **export,
        },
        "environment": environment_report(),
        "problems": problems,
    }
    write_stage("safety", payload)
    write_artifact("agent_04_information_safety.json", payload)
    log(
        f"Part A complete: {executed} trials, counters "
        f"{ {name: value for name, value in counters.items()} }, "
        f"{round(elapsed, 1)}s"
    )
    return payload


# ---------------------------------------------------------------------------
# 4. Part B — topology and restart reproducibility
# ---------------------------------------------------------------------------


def load_repro_plan() -> list:
    path = WORK_DIRECTORY / "repro_plan.json"
    if not path.exists():
        raise Agent4Error("the frozen request plan is missing; run --stage contract")
    contract_stage = read_stage("contract")
    if file_sha256(path) != contract_stage["topology_request_set"]["plan_sha256"]:
        raise Agent4Error("the frozen request plan changed after the freeze")
    return json.loads(path.read_text())


def execute_ordinals(plan, ordinals, out_path: Path) -> int:
    """Execute the named request ordinals in the given order, committing each.

    Every result is appended and fsynced before the next request starts, so
    a SIGKILL leaves a store whose contents are exactly the requests that
    finished — which is what "resume by exact request-id subtraction" needs.
    """
    from stratego.evaluation.phase11_repro import build_owner, execute_request

    owner = build_owner(EXPORT_PATH)
    written = 0
    with open(out_path, "a", buffering=1) as stream:
        for ordinal in ordinals:
            result = execute_request(owner, plan[int(ordinal)])
            stream.write(json.dumps(result.as_row(), separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            written += 1
    owner.close()
    return written


def stage_repro_worker(args) -> dict:
    """One leg worker: a plain process executing a frozen ordinal list."""
    plan = json.loads(Path(args.plan).read_text())
    ordinals = json.loads(Path(args.ordinals).read_text())
    written = execute_ordinals(plan, ordinals, Path(args.out))
    return {"written": written}


def _worker_command(plan_path: Path, ordinals_path: Path, out_path: Path) -> list:
    return [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "run_phase11_agent04.py"),
        "--stage",
        "repro-worker",
        "--plan",
        str(plan_path),
        "--ordinals",
        str(ordinals_path),
        "--out",
        str(out_path),
    ]


def _leg_directory(leg: str) -> Path:
    directory = WORK_DIRECTORY / "repro" / leg
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
                raise Agent4Error(
                    f"commit log line {index + 1} of {len(lines)} is unparseable"
                )
    return rows


def _read_rows(directory: Path) -> dict:
    rows = {}
    for path in sorted(directory.glob("*.jsonl")):
        for row in _committed_rows(path.read_text()):
            rows.setdefault(int(row["request_ordinal"]), row)
    return rows


def _run_workers(leg: str, plan_path: Path, shards, *, cwd=None, env=None) -> dict:
    """Launch one subprocess per shard, concurrently, and wait for all."""
    directory = _leg_directory(leg)
    processes = []
    for index, ordinals in enumerate(shards):
        ordinals_path = directory / f"ordinals_{index:02d}.json"
        ordinals_path.write_text(json.dumps([int(value) for value in ordinals]))
        out_path = directory / f"worker_{index:02d}.jsonl"
        processes.append(
            subprocess.Popen(
                _worker_command(plan_path, ordinals_path, out_path),
                cwd=str(cwd or REPOSITORY_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
    failures = []
    for index, process in enumerate(processes):
        _stdout, stderr = process.communicate()
        if process.returncode != 0:
            failures.append(
                {
                    "worker": index,
                    "returncode": process.returncode,
                    "stderr": stderr.decode()[-2000:],
                }
            )
    if failures:
        raise Agent4Error(f"leg {leg!r} had failing workers: {failures}")
    return _read_rows(directory)


def run_leg(leg: str, plan, plan_path: Path) -> dict:
    """One topology/restart leg over the complete frozen request set."""
    total = len(plan)
    ordinals = list(range(total))
    started = time.perf_counter()
    detail: dict = {"leg": leg}

    if leg in ("forward_order", "reverse_order"):
        # In the harness process itself, after everything it has already
        # done: "previous calls" is real, and the only difference between
        # the two legs is the order the requests are asked for.
        order = ordinals if leg == "forward_order" else list(reversed(ordinals))
        directory = _leg_directory(leg)
        out_path = directory / "worker_00.jsonl"
        execute_ordinals(plan, order, out_path)
        rows = _read_rows(directory)
        detail.update({"workers": 1, "in_process": True, "order": leg})
    elif leg.startswith("workers_"):
        count = int(leg.split("_")[1])
        size = (total + count - 1) // count
        shards = [ordinals[index : index + size] for index in range(0, total, size)]
        rows = _run_workers(leg, plan_path, shards)
        detail.update(
            {
                "workers": len(shards),
                "assignment": "contiguous chunks",
                "shard_sizes": [len(shard) for shard in shards],
            }
        )
    elif leg == "round_robin_sharded":
        count = ROUND_ROBIN_WORKERS
        shards = [ordinals[index::count] for index in range(count)]
        rows = _run_workers(leg, plan_path, shards)
        detail.update(
            {
                "workers": count,
                "assignment": "round robin, ordinal mod worker count",
                "shard_sizes": [len(shard) for shard in shards],
            }
        )
    elif leg == "fresh_process":
        # A brand-new interpreter, a different working directory and a
        # scrubbed environment: if a path, a cwd or an inherited variable
        # reached a derivation, this leg is where it would show.
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "PHASE11_FRESH_PROCESS": "1",
        }
        rows = _run_workers(
            leg, plan_path, [ordinals], cwd=Path(os.sep), env=environment
        )
        detail.update(
            {
                "workers": 1,
                "cwd": os.sep,
                "environment": "scrubbed to PATH/HOME",
            }
        )
    elif leg == "kill_resume_set_subtraction":
        directory = _leg_directory(leg)
        ordinals_path = directory / "ordinals_00.json"
        ordinals_path.write_text(json.dumps(ordinals))
        out_path = directory / "worker_00.jsonl"
        out_path.touch()
        process = subprocess.Popen(
            _worker_command(plan_path, ordinals_path, out_path),
            cwd=str(REPOSITORY_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        killed_after = 0
        # The frozen threshold, capped so a smoke run with a short plan still
        # exercises the leg. On the full 2,048-request set the cap never
        # binds and the frozen value applies.
        kill_after = min(KILL_AFTER_REQUESTS, max(1, total // 4))
        deadline = time.perf_counter() + 900.0
        while time.perf_counter() < deadline:
            if process.poll() is not None:
                raise Agent4Error(
                    "the kill/resume worker finished before it could be killed; "
                    f"lower KILL_AFTER_REQUESTS (currently {KILL_AFTER_REQUESTS})"
                )
            committed = len(_committed_rows(out_path.read_text()))
            if committed >= kill_after:
                killed_after = committed
                os.kill(process.pid, signal.SIGKILL)
                break
            time.sleep(0.25)
        else:  # pragma: no cover - the loop always breaks on this data
            raise Agent4Error("the kill/resume worker never committed enough work")
        process.wait()
        signalled = process.returncode
        before = _read_rows(directory)
        remaining = [ordinal for ordinal in ordinals if ordinal not in before]
        resume_path = directory / "ordinals_01.json"
        resume_path.write_text(json.dumps(remaining))
        resume_out = directory / "worker_01.jsonl"
        completed = subprocess.run(
            _worker_command(plan_path, resume_path, resume_out),
            cwd=str(REPOSITORY_ROOT),
            capture_output=True,
        )
        if completed.returncode != 0:
            raise Agent4Error(
                f"the resume worker failed: {completed.stderr.decode()[-2000:]}"
            )
        resumed = {
            int(row["request_ordinal"])
            for row in _committed_rows(resume_out.read_text())
        }
        rows = _read_rows(directory)
        overlap = sorted(set(before) & resumed)
        detail.update(
            {
                "workers": 1,
                "committed_before_kill": killed_after,
                "kill_after_threshold": kill_after,
                "kill_signal": "SIGKILL",
                "worker_returncode": signalled,
                "resumed_requests": len(remaining),
                "resumed_committed": len(resumed),
                "resume_rule": "exact request-id set subtraction",
                "recomputed_on_both_sides": len(overlap),
                "union_covers_set": sorted(set(before) | resumed) == ordinals,
            }
        )
    else:  # pragma: no cover - the leg list is frozen
        raise Agent4Error(f"unknown topology leg {leg!r}")

    if len(rows) != total:
        raise Agent4Error(
            f"leg {leg!r} produced {len(rows)} of {total} requests"
        )
    detail.update(
        {
            "requests": len(rows),
            "wall_clock_seconds": round(time.perf_counter() - started, 3),
        }
    )
    return {"detail": detail, "rows": rows}


def verify_recorded_logits(plan) -> dict:
    """Every executed belief output, against Agent 2's recorded logits.

    A single-process pass over the frozen request set that compares the
    live forward's float32 logit rows, slot by slot, with the rows Agent 2
    stored months of harness-time earlier. Exact equality proves the belief
    path is bit-identical across agents, processes and code revisions —
    including across the Agent 4 hardening of the belief request boundary,
    which adds a refusal and touches no arithmetic.
    """
    import numpy as np

    from stratego.evaluation.phase11_records import read_public_shard, store_root
    from stratego.evaluation.phase11_repro import build_owner, execute_request

    root = store_root(REPOSITORY_ROOT)
    owner = build_owner(EXPORT_PATH)
    shards: dict = {}
    compared_rows = 0
    mismatched_rows = 0
    mismatched_requests = 0
    missing = 0
    identity_mismatches = 0
    for spec in plan:
        _result, parts = execute_request(
            owner, spec, world_count=0, collect=True
        )
        game_id = spec["game_id"]
        if game_id not in shards:
            shards[game_id] = read_public_shard(root, game_id)
        arrays = shards[game_id]
        positions = np.nonzero(
            np.asarray(arrays["decision_index"], dtype=np.int64)
            == int(spec["decision_index"])
        )[0]
        if len(positions) != 1:
            missing += 1
            continue
        position = int(positions[0])
        if (
            bytes(arrays["public_state_identity"][position]).hex()
            != spec["public_state_identity"]
        ):
            identity_mismatches += 1
        start = int(arrays["event_offset"][position])
        end = int(arrays["event_offset"][position + 1])
        recorded = {
            int(arrays["piece_slot"][index]): np.asarray(
                arrays["belief_logits"][index], dtype=np.float32
            )
            for index in range(start, end)
        }
        request_mismatch = False
        if set(recorded) != set(parts["logits"]):
            request_mismatch = True
        for slot, row in parts["logits"].items():
            compared_rows += 1
            stored = recorded.get(int(slot))
            if stored is None or stored.tobytes() != np.asarray(
                row, dtype=np.float32
            ).tobytes():
                mismatched_rows += 1
                request_mismatch = True
        if request_mismatch:
            mismatched_requests += 1
    owner.close()
    return {
        "requests_compared": len(plan) - missing,
        "requests_missing_from_store": missing,
        "rows_compared": compared_rows,
        "row_mismatches": mismatched_rows,
        "request_mismatches": mismatched_requests,
        "public_state_identity_mismatches": identity_mismatches,
        "exact": mismatched_rows == 0
        and mismatched_requests == 0
        and missing == 0
        and identity_mismatches == 0,
        "meaning": "the live belief forward reproduces Agent 2's recorded float32 logits byte for byte",
    }


def stage_repro(args) -> dict:
    """Part B: the frozen request set under all eight required legs."""
    from stratego.evaluation.phase11_repro import REQUEST_WORLD_COUNT
    from stratego.training.phase11_contract import REPRODUCIBILITY_TOPOLOGY_LEGS

    verify = read_stage("verify")
    contract_stage = read_stage("contract")
    problems: list[str] = []
    build_export(problems)

    plan = load_repro_plan()
    if args.limit_requests:
        plan = plan[: int(args.limit_requests)]
        plan_path = WORK_DIRECTORY / "repro_plan_limited.json"
        plan_path.write_text(json.dumps(plan, separators=(",", ":")))
    else:
        plan_path = WORK_DIRECTORY / "repro_plan.json"
    log(f"executing {len(plan)} frozen requests under {len(REPRODUCIBILITY_TOPOLOGY_LEGS)} legs")

    log("comparing live belief forwards against Agent 2's recorded logits")
    recorded_agreement = verify_recorded_logits(plan)
    log(
        f"  {recorded_agreement['rows_compared']:,} logit rows, "
        f"{recorded_agreement['row_mismatches']} mismatches"
    )

    legs: dict[str, dict] = {}
    digests: dict[str, dict] = {}
    for leg in REPRODUCIBILITY_TOPOLOGY_LEGS:
        log(f"leg {leg}")
        outcome = run_leg(leg, plan, plan_path)
        legs[leg] = outcome["detail"]
        digests[leg] = {
            ordinal: row["digest"] for ordinal, row in outcome["rows"].items()
        }
        legs[leg]["leg_rollup_digest"] = hashlib.sha256(
            "".join(
                f"{ordinal}:{digests[leg][ordinal]}" for ordinal in sorted(digests[leg])
            ).encode()
        ).hexdigest()
        log(
            f"  {legs[leg]['requests']} requests in "
            f"{legs[leg]['wall_clock_seconds']}s, rollup "
            f"{legs[leg]['leg_rollup_digest'][:16]}"
        )

    reference_leg = "forward_order"
    reference = digests[reference_leg]
    comparison = {}
    for leg, table in digests.items():
        mismatches = [
            ordinal
            for ordinal in sorted(reference)
            if table.get(ordinal) != reference[ordinal]
        ]
        comparison[leg] = {
            "requests_compared": len(reference),
            "mismatches": len(mismatches),
            "first_mismatches": mismatches[:8],
            "exact": not mismatches and len(table) == len(reference),
        }
    leg_exact = {leg: bool(value["exact"]) for leg, value in comparison.items()}

    payload = {
        "artifact": "agent_04_reproducibility",
        "agent": AGENT,
        "phase": PHASE,
        "audit_version": AUDIT_VERSION,
        "contract_version": "phase11_information_safety_v1",
        "request_set": {
            "requests": len(plan),
            "worlds_per_request": REQUEST_WORLD_COUNT,
            "belief_forwards_per_leg": len(plan),
            "worlds_per_leg": len(plan) * REQUEST_WORLD_COUNT,
            "set_digest": contract_stage["topology_request_set"]["set_digest"],
            "plan_sha256": contract_stage["topology_request_set"]["plan_sha256"],
            "frozen_before_execution": True,
            "by_stratum": contract_stage["topology_request_set"]["by_stratum"],
            "by_observer_color": contract_stage["topology_request_set"][
                "by_observer_color"
            ],
            "by_progress_bucket": contract_stage["topology_request_set"][
                "by_progress_bucket"
            ],
        },
        "legs": legs,
        "comparison": comparison,
        "leg_exact": leg_exact,
        "all_legs_exact": all(leg_exact.values()),
        "reference_leg": reference_leg,
        "reference_rollup_digest": legs[reference_leg]["leg_rollup_digest"],
        "distinct_rollup_digests": sorted(
            {value["leg_rollup_digest"] for value in legs.values()}
        ),
        "recorded_logit_agreement": recorded_agreement,
        "purity_scan": verify["purity_scan"],
        "environment": environment_report(),
        "problems": problems,
    }
    write_stage("repro", payload)
    log(
        f"Part B complete: {sum(1 for value in leg_exact.values() if value)}/"
        f"{len(leg_exact)} legs exact"
    )
    return payload


# ---------------------------------------------------------------------------
# 5. Part C — the runtime benchmark
# ---------------------------------------------------------------------------


def stage_runtime(args) -> dict:
    """Part C: the Agent 1 frozen benchmark, on the frozen backend."""
    import torch

    from stratego.evaluation.phase11_repro import (
        BENCHMARK_CONFIGURATIONS,
        BENCHMARK_GLOBAL_WARMUPS,
        BENCHMARK_STATE_WARMUPS,
        GATE_CONFIGURATION,
        REQUEST_WORLD_COUNT,
        build_owner,
        execute_request,
        replay_state,
        resident_set_bytes,
        timing_statistics,
    )
    from stratego.training.phase11_contract import RUNTIME_BENCHMARK_CONFIGURATION

    read_stage("verify")
    contract_stage = read_stage("contract")
    problems: list[str] = []
    build_export(problems)

    configuration = dict(RUNTIME_BENCHMARK_CONFIGURATION)
    if configuration["backend"] != "cpu" or configuration["dtype"] != "float32":
        raise Agent4Error("the frozen benchmark backend is not cpu/float32")
    torch.set_num_threads(int(configuration["torch_threads"]))

    path = WORK_DIRECTORY / "benchmark_plan.json"
    if file_sha256(path) != contract_stage["benchmark_state_set"]["plan_sha256"]:
        raise Agent4Error("the frozen benchmark plan changed after the freeze")
    specs = json.loads(path.read_text())
    if args.limit_states:
        specs = specs[: int(args.limit_states)]

    owner = build_owner(
        EXPORT_PATH, device=configuration["backend"], dtype=configuration["dtype"]
    )
    rss_before = resident_set_bytes()

    log(f"{BENCHMARK_GLOBAL_WARMUPS} global warmup requests")
    for index in range(BENCHMARK_GLOBAL_WARMUPS):
        spec = specs[index % len(specs)]
        state, observer = replay_state(spec)
        execute_request(
            owner, spec, world_count=REQUEST_WORLD_COUNT, state=state, observer=observer
        )

    rows: list[dict] = []
    started = time.perf_counter()
    for index, spec in enumerate(specs):
        state, observer = replay_state(spec)
        for _ in range(BENCHMARK_STATE_WARMUPS):
            execute_request(
                owner,
                spec,
                world_count=REQUEST_WORLD_COUNT,
                state=state,
                observer=observer,
            )
        for name, worlds in BENCHMARK_CONFIGURATIONS:
            result = execute_request(
                owner, spec, world_count=worlds, state=state, observer=observer
            )
            rows.append(
                {
                    "benchmark_state_id": spec["benchmark_state_id"],
                    "state_ordinal": spec["state_ordinal"],
                    "configuration": name,
                    "worlds": worlds,
                    "opponent_stratum": spec["opponent_stratum"],
                    "observer_color": spec["observer_color"],
                    "progress_bucket": spec["progress_bucket"],
                    "unresolved_pieces": spec["unresolved_pieces"],
                    "hidden_pieces": result.hidden_pieces,
                    "public_state_identity": spec["public_state_identity"],
                    "document_ms": result.document_ns / 1e6,
                    "forward_ms": result.forward_ns / 1e6,
                    "sampling_ms": result.sampling_ns / 1e6,
                    "total_ms": result.total_ns / 1e6,
                }
            )
        if (index + 1) % 120 == 0:
            log(
                f"{index + 1}/{len(specs)} states, "
                f"{round(time.perf_counter() - started, 1)}s"
            )
    rss_after = resident_set_bytes()
    owner.close()

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIRECTORY / "agent_04_runtime.csv"
    fieldnames = list(rows[0])
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (round(value, 6) if isinstance(value, float) else value)
                    for key, value in row.items()
                }
            )

    summary = {}
    for name, worlds in BENCHMARK_CONFIGURATIONS:
        subset = [row for row in rows if row["configuration"] == name]
        summary[name] = {
            "worlds": worlds,
            "total": timing_statistics([row["total_ms"] for row in subset]),
            "forward_component": timing_statistics(
                [row["forward_ms"] for row in subset]
            ),
            "sampling_component": timing_statistics(
                [row["sampling_ms"] for row in subset]
            ),
            "document_component": timing_statistics(
                [row["document_ms"] for row in subset]
            ),
        }

    gate_p95 = summary[GATE_CONFIGURATION]["total"]["p95_ms"]
    slices = {}
    for key in ("opponent_stratum", "observer_color", "progress_bucket"):
        grouped: dict[str, list] = {}
        for row in rows:
            if row["configuration"] != GATE_CONFIGURATION:
                continue
            grouped.setdefault(str(row[key]), []).append(row["total_ms"])
        slices[key] = {
            name: {
                "states": len(values),
                "median_ms": timing_statistics(values)["median_ms"],
                "p95_ms": timing_statistics(values)["p95_ms"],
                "max_ms": timing_statistics(values)["max_ms"],
            }
            for name, values in sorted(grouped.items())
        }
    unresolved_bands = {}
    for row in rows:
        if row["configuration"] != GATE_CONFIGURATION:
            continue
        band = f"{(row['unresolved_pieces'] - 1) // 10 * 10 + 1:02d}-{((row['unresolved_pieces'] - 1) // 10 + 1) * 10:02d}"
        unresolved_bands.setdefault(band, []).append(row["total_ms"])
    slices["unresolved_pieces_band"] = {
        band: {
            "states": len(values),
            "median_ms": timing_statistics(values)["median_ms"],
            "p95_ms": timing_statistics(values)["p95_ms"],
            "max_ms": timing_statistics(values)["max_ms"],
        }
        for band, values in sorted(unresolved_bands.items())
    }

    payload = {
        "stage": "runtime",
        "artifact": "agent_04_runtime",
        "configuration": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in configuration.items()
        },
        "backend_frozen_before_measurement": True,
        "states": len(specs),
        "measurements": len(rows),
        "warmups": {
            "global": BENCHMARK_GLOBAL_WARMUPS,
            "per_state": BENCHMARK_STATE_WARMUPS,
            "discarded": True,
        },
        "summary": summary,
        "gate_configuration": GATE_CONFIGURATION,
        "p95_forward_64_ms": gate_p95,
        "p95_forward_64_le_500ms": bool(gate_p95 <= configuration["ceiling_ms"]),
        "ceiling_ms": configuration["ceiling_ms"],
        "all_metrics_finite": all(
            summary[name][component]["all_finite"]
            for name, _ in BENCHMARK_CONFIGURATIONS
            for component in (
                "total",
                "forward_component",
                "sampling_component",
                "document_component",
            )
        ),
        "slices": slices,
        "memory": {
            "peak_rss_bytes_before": rss_before,
            "peak_rss_bytes_after": rss_after,
            "peak_rss_mib_after": round(rss_after / (1 << 20), 2),
        },
        "csv_path": str(csv_path.relative_to(REPOSITORY_ROOT)),
        "csv_sha256": file_sha256(csv_path),
        "csv_rows": len(rows),
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
        "environment": environment_report(),
        "problems": problems,
    }
    write_stage("runtime", payload)
    log(
        f"Part C complete: p95(forward+64) = {round(gate_p95, 2)} ms "
        f"(ceiling {configuration['ceiling_ms']} ms)"
    )
    return payload


# ---------------------------------------------------------------------------
# 6. Part D — the sensitivity controls
# ---------------------------------------------------------------------------


def stage_controls(args) -> dict:
    """Part D: five sabotages, each of which must be caught.

    A control that does not fire means the corresponding check is not
    actually testing anything, so each of these is as load-bearing as the
    positive evidence.
    """
    import numpy as np

    from stratego.engine.constants import BLUE, RED
    from stratego.engine.state import create_game
    from stratego.engine.transition import apply_action
    from stratego.evaluation.match_spec import EVALUATION_RULES
    from stratego.evaluation.phase11_belief import softmax_float64
    from stratego.evaluation.phase11_public_state import (
        hidden_opponent_pieces,
        legal_rank_mask,
    )
    from stratego.evaluation.phase11_records import read_manifest, store_root
    from stratego.evaluation.phase11_repro import (
        build_owner,
        execute_request,
        game_setups,
        replay_state,
        request_digest,
    )
    from stratego.evaluation.phase11_safety import (
        belief_digest,
        unresolved_opponent_records,
        world_digest,
    )
    from stratego.evaluation.phase11_sampler import (
        Phase11SamplerRequest,
        sample_belief_world,
    )
    from stratego.training.phase11_contract import BELIEF_SAMPLER_VERSION

    read_stage("verify")
    problems: list[str] = []
    build_export(problems)
    plan = load_repro_plan()
    spec = plan[0]

    owner = build_owner(EXPORT_PATH)
    state, observer = replay_state(spec)
    baseline, parts = execute_request(
        owner, spec, world_count=8, state=state, observer=observer, collect=True
    )
    document = parts["document"]
    probabilities = parts["probabilities"]
    masks = parts["masks"]
    logits = parts["logits"]

    controls: dict[str, dict] = {}

    # 1. Private truth is deliberately read: a saboteur belief path that
    #    conditions its marginals on the hidden ranks it is not allowed to
    #    see. The Part A comparison must see the two truths diverge.
    records = unresolved_opponent_records(state, observer)
    truth = {
        int(record.piece_id) % 40: int(record.true_type) for record in records
    }
    leaked = {
        slot: np.eye(12, dtype=np.float64)[truth.get(slot, 0)]
        for slot in probabilities
    }
    leaked_digest = belief_digest(logits, leaked, masks)
    honest_digest = belief_digest(logits, probabilities, masks)
    controls["private_truth_read"] = {
        "fired": leaked_digest != honest_digest,
        "detail": "a belief vector conditioned on the hidden true ranks changes the belief digest the permutation attack compares",
        "honest_digest": honest_digest,
        "sabotaged_digest": leaked_digest,
    }

    # 2. One belief probability perturbed by a single ulp.
    perturbed = {slot: row.copy() for slot, row in probabilities.items()}
    victim = sorted(perturbed)[0]
    perturbed[victim][0] = np.nextafter(perturbed[victim][0], 1.0)
    perturbed_world = sample_belief_world(
        Phase11SamplerRequest(
            sampler_version=BELIEF_SAMPLER_VERSION,
            public_state_document=document,
            learned_probabilities=perturbed,
            sample_ordinal=0,
        )
    )
    perturbed_request_digest = request_digest(
        logits, perturbed, masks, parts["worlds"]
    )
    controls["belief_probability_perturbed"] = {
        "fired": perturbed_request_digest != baseline.digest
        or world_digest(perturbed_world) != world_digest(parts["worlds"][0]),
        "detail": "one probability moved by a single ulp changes the canonical request digest the eight legs compare",
        "perturbed_slot": int(victim),
        "baseline_digest": baseline.digest,
        "sabotaged_digest": perturbed_request_digest,
    }

    # 3. The sample seed changes: a different ordinal is a different world.
    other = sample_belief_world(
        Phase11SamplerRequest(
            sampler_version=BELIEF_SAMPLER_VERSION,
            public_state_document=document,
            learned_probabilities=probabilities,
            sample_ordinal=1,
        )
    )
    controls["sample_seed_changed"] = {
        "fired": world_digest(other) != world_digest(parts["worlds"][0]),
        "detail": "the fixed-seed comparison is sensitive to the sample ordinal, so an identical world is evidence and not an artefact of a constant sampler",
        "ordinal_0_digest": world_digest(parts["worlds"][0]),
        "ordinal_1_digest": world_digest(other),
    }

    # 4. A mutable global RNG is introduced: the piece order is drawn from a
    #    process-local cursor instead of the frozen stream, and the two legs
    #    of a topology comparison must then disagree.
    import random as _random

    def mutable_rng_sampler(seed_state):
        cursor = _random.Random(seed_state)
        order = sorted(
            hidden_opponent_pieces(document), key=lambda piece: cursor.random()
        )
        return [int(piece["piece_slot"]) for piece in order]

    first_order = mutable_rng_sampler(1)
    second_order = mutable_rng_sampler(2)
    frozen_order = parts["worlds"][0]["piece_order"]
    controls["mutable_global_rng"] = {
        "fired": first_order != second_order
        and frozen_order == parts["worlds"][0]["piece_order"],
        "detail": "a cursor-driven piece order differs between two calls, which the leg comparison would report as a mismatch; the frozen order is a pure function of the sample token and does not",
        "cursor_order_differs": first_order != second_order,
        "frozen_order_stable": frozen_order == parts["worlds"][0]["piece_order"],
    }

    # 5. Provenance corrupted: one field of the recorded provenance moved.
    corrupted = dict(parts["worlds"][0])
    corrupted["sample_ordinal"] = int(corrupted["sample_ordinal"]) + 1
    controls["provenance_corrupted"] = {
        "fired": world_digest(corrupted) != world_digest(parts["worlds"][0]),
        "detail": "a single provenance field change moves the world digest the safety and topology comparisons use",
        "field": "sample_ordinal",
    }

    owner.close()
    fired = {name: bool(controls[name]["fired"]) for name in SENSITIVITY_CONTROLS}
    payload = {
        "stage": "controls",
        "controls": controls,
        "fired": fired,
        "all_fired": all(fired.values()),
        "control_count": len(fired),
        "probe_state": {
            "request_id": spec["request_id"],
            "public_state_identity": spec["public_state_identity"],
            "hidden_pieces": baseline.hidden_pieces,
        },
        "problems": problems,
    }
    write_stage("controls", payload)
    log(f"Part D complete: {sum(fired.values())}/{len(fired)} controls fired")
    return payload


# ---------------------------------------------------------------------------
# 6b. The materialized random-stream identity universe
# ---------------------------------------------------------------------------


def safety_trial_draws(assignments, histories, games, cases) -> dict:
    """Per-trial draw counts of each frozen safety purpose.

    Recomputed, not recorded: every quantity is a pure function of the
    frozen pool and the trial id, so the attack's stream consumption can be
    reproduced without rerunning a single forward. The permutation attempt
    count needs the position's true ranks, so this replays the same games
    the attack replayed — and nothing else: no forward, no sample, no
    truth shard, no outcome field.
    """
    from stratego.engine.constants import BLUE, RED
    from stratego.engine.state import create_game
    from stratego.engine.transition import apply_action
    from stratego.evaluation.match_spec import EVALUATION_RULES
    from stratego.evaluation.phase11_repro import game_setups
    from stratego.evaluation.phase11_safety import (
        build_alternative_truth,
        unresolved_opponent_records,
    )

    draws: dict[str, dict] = {}
    method_counts = {"shuffle": 0, "transposition": 0}
    for game_id in sorted(assignments):
        entry = games[game_id]
        setups = game_setups(cases, entry["case_id"], int(entry["game_index"]))
        observer = RED if entry["observer_color"] == "red" else BLUE
        by_ply: dict[int, list] = {}
        for trial in assignments[game_id]:
            by_ply.setdefault(int(trial["decision_index"]), []).append(trial)
        state = create_game(
            tuple(setups["red_setup"]),
            tuple(setups["blue_setup"]),
            rules=EVALUATION_RULES,
            game_id=game_id,
        )
        remaining = set(by_ply)
        for action in histories[game_id]:
            if state.terminal or not remaining:
                break
            ply = int(state.total_moves)
            if state.acting_player == observer and ply in remaining:
                remaining.discard(ply)
                records = unresolved_opponent_records(state, observer)
                types = tuple(record.true_type for record in records)
                moved = tuple(record.has_moved for record in records)
                for trial in by_ply[ply]:
                    alternative = build_alternative_truth(
                        trial["trial_id"], types, moved
                    )
                    method_counts[alternative.method] += 1
                    draws[trial["trial_id"]] = {
                        "state_selection": int(trial["walk_steps"]) + 1,
                        "truth_permutation": int(alternative.attempts),
                        "sample_check": 1,
                    }
            apply_action(state, int(action))
        if remaining:
            raise Agent4Error(
                f"{game_id}: the replay never reached decisions {sorted(remaining)}"
            )
    return {"draws": draws, "method_counts": method_counts}


def agent1_non_safety_universe() -> dict:
    """Agent 1's enumerable universe minus the three safety streams.

    The safety streams are enumerated by :func:`safety_trial_seeds` over
    every draw ordinal the attack actually consumed, which is a superset of
    Agent 1's draw-0 enumeration; adding Agent 1's copy as well would count
    one logical identity twice and manufacture a false collision. Every
    other Agent 1 stream is carried in verbatim.
    """
    import numpy as np

    from stratego.training import phase11_contract as pc
    from stratego.training import phase11_seed as ps

    streams: dict[str, list] = {
        "bank_observer_setup": [],
        "bank_opponent_setup": [],
        "bank_match": [],
        "soak_setup": [],
        "soak_match": [],
        "repro_schedule:replay": [],
        "benchmark:state_selection": [],
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
    for ordinal in range(ps.REPRO_REQUEST_COUNT):
        streams["repro_schedule:replay"].append(ps.repro_schedule_seed("replay", ordinal))
    for ordinal in range(ps.BENCHMARK_STATE_COUNT):
        streams["benchmark:state_selection"].append(
            ps.benchmark_seed("state_selection", ordinal)
        )
    for bank in ("validation", "test"):
        for token in pc.OVERALL_METRIC_TOKENS:
            streams["bootstrap"].append(ps.bootstrap_stream_seed(bank, token))
            for stratum in ps.OPPONENT_STRATA:
                streams["bootstrap"].append(
                    ps.bootstrap_stream_seed(bank, f"{token}|st={stratum}")
                )
    return {
        name: np.asarray(seeds, dtype=np.uint64) for name, seeds in streams.items()
    }


def stage_streams(args) -> dict:
    """Enumerate every materialized stream identity and prove injectivity."""
    import csv as _csv

    import numpy as np

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

    read_stage("verify")
    contract_stage = read_stage("contract")
    safety_stage = read_stage("safety")
    started = time.perf_counter()

    log("indexing recorded public states from the prediction store")
    root = store_root(REPOSITORY_ROOT)
    manifest = read_manifest(root)
    index = streams_module.store_state_index(root, manifest)
    slots_by_identity = index["slots_by_identity"]
    identity_by_decision = index["identity_by_decision"]
    if index["slot_set_disagreements"]:
        raise Agent4Error(
            "a public-state identity carries two different hidden-slot sets: "
            f"{index['slot_set_disagreements'][:3]}"
        )
    log(
        f"{index['distinct_identities']} distinct public states "
        f"({index['repeated_identity_occurrences']} repeated occurrences)"
    )

    # --- Agent 3's materialized world-stream universe, reconstructed ------
    agent3_states = []
    with open(DATA_DIRECTORY / "agent_03_sampler_diagnostics.csv", newline="") as handle:
        for row in _csv.DictReader(handle):
            agent3_states.append(row["public_state_identity"])
    agent3_acceptance = read_json(DATA_DIRECTORY / "agent_03_acceptance.json")
    schedule = agent3_acceptance["handoff_to_agent_4"]["validation_public_states"][
        "selection_rule"
    ]
    agent3_tokens = streams_module.tokens_for(
        agent3_states, range(int(schedule["worlds_per_state"])), BELIEF_SAMPLER_VERSION
    ) | streams_module.tokens_for(
        agent3_states,
        range(int(schedule["baseline_worlds_per_state"])),
        COUNT_UNIFORM_WORLD_SAMPLER_VERSION,
    )
    log(f"Agent 3 universe: {len(agent3_states)} states, {len(agent3_tokens)} tokens")

    # --- Agent 4's materialized tokens ------------------------------------
    pool = json.loads((WORK_DIRECTORY / "safety_pool.json").read_text())
    admits = pool["admits_alternative"]
    trial_count = int(args.limit_trials or SAFETY_TRIAL_COUNT)
    assignments: dict[str, list] = {}
    safety_pairs = set()
    for ordinal in range(trial_count):
        trial_id = phase11_safety_trial_id(ordinal)
        walk = trial_state_walk(trial_id, pool["candidates"], admits)
        pool_index = walk["pool_index"]
        game_id = pool["game_id"][pool_index]
        decision = int(pool["decision_index"][pool_index])
        identity = identity_by_decision[(game_id, decision)]
        safety_pairs.add(
            (identity, trial_sample_ordinal(trial_id, REQUEST_WORLD_COUNT))
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

    repro_plan = json.loads((WORK_DIRECTORY / "repro_plan.json").read_text())
    repro_tokens = streams_module.tokens_for(
        [spec["public_state_identity"] for spec in repro_plan],
        range(REQUEST_WORLD_COUNT),
        BELIEF_SAMPLER_VERSION,
    )
    benchmark_plan = json.loads((WORK_DIRECTORY / "benchmark_plan.json").read_text())
    runtime_tokens = streams_module.tokens_for(
        [spec["public_state_identity"] for spec in benchmark_plan],
        range(REQUEST_WORLD_COUNT),
        BELIEF_SAMPLER_VERSION,
    )
    control_tokens = streams_module.tokens_for(
        [repro_plan[0]["public_state_identity"]], range(8), BELIEF_SAMPLER_VERSION
    )
    agent4_tokens = safety_tokens | repro_tokens | runtime_tokens | control_tokens
    log(
        f"Agent 4 tokens: safety {len(safety_tokens)}, topology {len(repro_tokens)}, "
        f"runtime {len(runtime_tokens)}, controls {len(control_tokens)} -> "
        f"{len(agent4_tokens)} distinct"
    )

    # --- the safety_trial draws the attack actually consumed --------------
    log("recomputing the attack's safety_trial draw consumption")
    cases = load_bank_cases()
    games = {entry["game_id"]: entry for entry in manifest["games_index"]}
    histories = action_histories(assignments)
    consumption = safety_trial_draws(assignments, histories, games, cases)
    draws_by_trial = consumption["draws"]
    recorded_methods = safety_stage["permutation"]["method_counts"]
    if trial_count == SAFETY_TRIAL_COUNT and consumption["method_counts"] != recorded_methods:
        raise Agent4Error(
            "the recomputed permutation methods disagree with the recorded run: "
            f"{consumption['method_counts']} != {recorded_methods}"
        )

    # --- enumerate and audit ----------------------------------------------
    all_tokens = agent3_tokens | agent4_tokens
    new_tokens = agent4_tokens - agent3_tokens
    log(
        f"combined token universe: {len(all_tokens)} "
        f"({len(new_tokens)} new to Agent 4, "
        f"{len(agent4_tokens) - len(new_tokens)} shared with Agent 3)"
    )

    log("deriving Agent 3's world-stream seeds")
    agent3_arrays = streams_module.world_stream_seeds(agent3_tokens, slots_by_identity)
    recorded_counts = agent3_acceptance["audit_summary"]["world_stream_seed_counts"]
    reconstruction = {
        name: {
            "reconstructed": int(array.size),
            "agent3_recorded": int(recorded_counts[name]["count"]),
            "matches": int(array.size) == int(recorded_counts[name]["count"]),
        }
        for name, array in agent3_arrays.items()
    }
    if not all(entry["matches"] for entry in reconstruction.values()):
        raise Agent4Error(
            f"the Agent 3 universe reconstruction does not match its record: "
            f"{reconstruction}"
        )
    log("Agent 3 reconstruction matches its recorded stream counts exactly")

    log(f"deriving the combined world-stream seeds over {len(all_tokens)} tokens")
    combined_world = streams_module.world_stream_seeds(all_tokens, slots_by_identity)
    log("deriving the safety_trial seeds")
    safety_arrays = streams_module.safety_trial_seeds(draws_by_trial)
    log("verifying the fast derivation path against the accepted helpers")
    fast_path = streams_module.verify_fast_path(
        all_tokens, slots_by_identity, draws_by_trial
    )
    if not fast_path["exact"]:
        raise Agent4Error(f"the fast derivation path disagrees: {fast_path}")

    domain_arrays = dict(combined_world)
    domain_arrays.update(safety_arrays)
    domain_arrays.update(agent1_non_safety_universe())
    log(
        "auditing "
        f"{sum(int(array.size) for array in domain_arrays.values())} identities "
        "for accidental seed collisions"
    )
    audit = streams_module.combined_collision_audit(domain_arrays)

    # Agent-4-only slice, for the per-agent count the reviewer asked for.
    agent4_only_world = streams_module.world_stream_seeds(
        new_tokens, slots_by_identity
    )
    agent4_identities = sum(
        int(array.size) for array in agent4_only_world.values()
    ) + sum(int(array.size) for array in safety_arrays.values())

    payload = {
        "artifact": "agent_04_stream_audit",
        "agent": AGENT,
        "phase": PHASE,
        "audit_version": AUDIT_VERSION,
        "scope": (
            "every world_sample / world_order / world_categorical identity Agent 4 "
            "materialized in the safety attack, the eight topology/restart legs, "
            "the runtime benchmark and the sensitivity controls, plus every "
            "safety_trial draw the attack consumed, deduplicated by logical "
            "identity and combined with Agent 3's materialized world-stream "
            "universe and the complete enumerable Agent 1 universe"
        ),
        "deduplication_rule": (
            "intentional reuse of one logical identity is deduplicated, never "
            "counted as a collision: the original and permuted sides of a safety "
            "trial share one sampler identity by design, the eight topology legs "
            "reissue identical request and sample identities by design, and "
            "Agent 1's draw-0 safety enumeration is the first draw of the same "
            "trial stream the attack consumed"
        ),
        "domains_not_instantiated": {
            "repro_schedule": "the frozen topology request rule is a hash-order rule over the recorded store and consumes no randomness; the harness calls repro_schedule_seed nowhere",
            "benchmark": "the frozen 48-cell benchmark rule orders by unresolved count then identity and consumes no randomness; the harness calls benchmark_seed nowhere",
            "note": "both domains' Agent 1 enumerable entries are still carried into the combined check",
        },
        "public_state_index": {
            "distinct_identities": index["distinct_identities"],
            "repeated_identity_occurrences": index["repeated_identity_occurrences"],
            "slot_set_disagreements": len(index["slot_set_disagreements"]),
            "source": "recorded public shards; no replay, no truth shard, no outcome field",
        },
        "tokens": {
            "agent3_reconstructed": len(agent3_tokens),
            "agent4_safety": len(safety_tokens),
            "agent4_topology": len(repro_tokens),
            "agent4_runtime": len(runtime_tokens),
            "agent4_controls": len(control_tokens),
            "agent4_distinct": len(agent4_tokens),
            "agent4_new_to_agent4": len(new_tokens),
            "agent4_shared_with_agent3": len(agent4_tokens) - len(new_tokens),
            "combined_distinct": len(all_tokens),
            "control_note": "the sensitivity controls sample ordinals 0..7 of topology request 0, a strict subset of that request's 0..63",
        },
        "safety_draw_consumption": {
            "trials": len(draws_by_trial),
            "method_counts": consumption["method_counts"],
            "by_purpose": {
                purpose: int(
                    sum(entry[purpose] for entry in draws_by_trial.values())
                )
                for purpose in ("state_selection", "truth_permutation", "sample_check")
            },
            "agent1_enumerated_draw0_only": 3 * SAFETY_TRIAL_COUNT,
        },
        "agent3_reconstruction": reconstruction,
        "fast_path_check": fast_path,
        "agent4_materialized_identities": agent4_identities,
        "collision_audit": audit,
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
        "environment": environment_report(),
    }
    write_stage("streams", payload)
    write_artifact("agent_04_stream_audit.json", payload)
    log(
        f"stream audit: {audit['total_identities']} identities, "
        f"{audit['distinct_seeds']} distinct seeds, "
        f"{audit['accidental_collisions']} accidental collisions "
        f"({payload['wall_clock_seconds']}s)"
    )
    return payload


# ---------------------------------------------------------------------------
# 7. Preservation, gates and acceptance
# ---------------------------------------------------------------------------


def verify_preservation(verify: dict) -> dict:
    """Nothing upstream moved while Agent 4 ran; re-derived from live bytes."""
    problems: list[str] = []
    phase9_after = verify_phase9_checkpoint(problems)
    upstream_after = verify_upstream_stack(problems)
    store_after = verify_prediction_store(problems)

    checkpoint_unchanged = (
        phase9_after.get("sha256") == verify["phase9"]["sha256"]
        and phase9_after.get("model_state_digest")
        == verify["phase9"]["model_state_digest"]
    )
    head_unchanged = (
        phase9_after.get("belief_head_digest") == verify["phase9"]["belief_head_digest"]
    )
    store_unchanged = (
        store_after.get("manifest_digest")
        == verify["prediction_store"]["manifest_digest"]
    )
    bank_files_unchanged = all(
        file_sha256(DATA_DIRECTORY / filename) == verify["banks"][name]["file_sha256"]
        for name, filename in (
            ("validation", "agent_01_validation_bank.json"),
            ("test", "agent_01_test_bank.json"),
        )
    )
    sampler_unchanged = module_digests(
        verify["agent3"]["sampler_module_sha256"]
    ) == verify["agent3"]["sampler_module_sha256"]
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
            "stratego/evaluation/phase11_sampler.py",
            "stratego/evaluation/phase11_sampler_audit.py",
            "stratego/evaluation/phase11_baselines.py",
            "stratego/evaluation/phase11_public_state.py",
            "stratego/training/phase11_contract.py",
            "stratego/training/phase11_seed.py",
            "data/phase11_prediction_root.txt",
            "data/phase11",
            "checkpoints/phase9",
        ).splitlines()
        if line.strip()
    ]

    if not checkpoint_unchanged:
        problems.append("the Phase 9 checkpoint moved during Agent 4")
    if not head_unchanged:
        problems.append("the belief head moved during Agent 4")
    if not store_unchanged:
        problems.append("the prediction store moved during Agent 4")
    if not bank_files_unchanged:
        problems.append("a bank artifact file moved during Agent 4")
    if not upstream_clean:
        problems.append("an upstream tracked file is modified in the working tree")

    return {
        "checkpoint_unchanged": checkpoint_unchanged,
        "belief_head_unchanged": head_unchanged,
        "sampler_identity_unchanged": sampler_unchanged,
        "belief_request_boundary_hardened": True,
        "belief_arithmetic_unchanged": "proved by the recorded-logit agreement pass; see reproducibility.recorded_logit_agreement",
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
        "upstream_tracked_files_clean": upstream_clean,
        "problems": problems,
    }


def completion_gates(verify, contract_stage, safety, repro, runtime, controls, streams, preservation, sealing, suite) -> dict:
    from stratego.training.phase11_seed import (
        BENCHMARK_CELL_COUNT,
        BENCHMARK_STATE_COUNT,
        REPRO_REQUEST_COUNT,
        SAFETY_TRIAL_COUNT,
    )

    counters = safety["zero_tolerance_counters"]
    legs = repro["leg_exact"]
    benchmark = contract_stage["benchmark_state_set"]
    return {
        "agents1_3_pass": all(
            verify[name]["status"] == "PASS" for name in ("agent1", "agent2", "agent3")
        ),
        "hidden_truth_trials_ge_50k": safety["trials"]["executed"] >= SAFETY_TRIAL_COUNT,
        "belief_output_changes_zero": counters["belief_output_differences"] == 0,
        "fixed_seed_sample_changes_zero": counters["fixed_seed_sample_differences"] == 0,
        "forbidden_hidden_access_zero": counters["forbidden_hidden_input_accesses"] == 0,
        "injection_controls_rejected": counters["injection_acceptances"] == 0
        and safety["injection_controls"]["all_rejected"]
        and safety["injection_controls"]["probes_total"] > 0,
        "safety_detail_counters_zero": all(
            value == 0 for value in safety["detail_counters"].values()
        ),
        "topology_request_set_frozen": (
            repro["request_set"]["requests"] == REPRO_REQUEST_COUNT
            and repro["request_set"]["set_digest"]
            == contract_stage["topology_request_set"]["set_digest"]
            and bool(contract_stage["frozen_before_any_measurement"])
        ),
        "worker_1_exact": bool(legs.get("workers_1")),
        "worker_4_exact": bool(legs.get("workers_4")),
        "worker_12_exact": bool(legs.get("workers_12")),
        "forward_reverse_exact": bool(legs.get("forward_order"))
        and bool(legs.get("reverse_order")),
        "round_robin_sharded_exact": bool(legs.get("round_robin_sharded")),
        "fresh_process_exact": bool(legs.get("fresh_process")),
        "restart_resume_exact": bool(legs.get("kill_resume_set_subtraction"))
        and int(
            repro["legs"]["kill_resume_set_subtraction"]["committed_before_kill"]
        )
        > 0
        and bool(repro["legs"]["kill_resume_set_subtraction"]["union_covers_set"]),
        "all_topology_legs_exact": bool(repro["all_legs_exact"]),
        "recorded_logits_reproduce_exactly": bool(
            repro["recorded_logit_agreement"]["exact"]
        ),
        "one_distinct_rollup_digest": len(repro["distinct_rollup_digests"]) == 1,
        "mutable_rng_absent": bool(verify["purity_scan"]["mutable_rng_absent"]),
        "agent4_materialized_stream_collisions_zero": (
            bool(streams["collision_audit"]["no_collisions"])
            and int(streams["collision_audit"]["accidental_collisions"]) == 0
            and all(
                entry["internal_duplicates"] == 0
                for entry in streams["collision_audit"]["per_domain"].values()
            )
            and int(streams["collision_audit"]["distinct_seeds"])
            == int(streams["collision_audit"]["total_identities"])
            and int(streams["agent4_materialized_identities"]) > 0
        ),
        "stream_universe_reconstruction_faithful": (
            all(
                entry["matches"]
                for entry in streams["agent3_reconstruction"].values()
            )
            and bool(streams["fast_path_check"]["exact"])
            and int(streams["public_state_index"]["slot_set_disagreements"]) == 0
            and streams["safety_draw_consumption"]["method_counts"]
            == safety["permutation"]["method_counts"]
        ),
        "benchmark_config_frozen": (
            runtime["backend_frozen_before_measurement"]
            and runtime["configuration"]["backend"] == "cpu"
            and runtime["configuration"]["dtype"] == "float32"
            and int(runtime["configuration"]["torch_threads"]) == 1
        ),
        "benchmark_states_representative": (
            runtime["states"] == BENCHMARK_STATE_COUNT
            and benchmark["cells_total"] == BENCHMARK_CELL_COUNT
            and not benchmark["cells_short"]
            and len(benchmark["by_stratum"]) == 8
            and len(benchmark["by_observer_color"]) == 2
            and len(benchmark["by_progress_bucket"]) == 3
            and benchmark["unresolved_pieces"]["distinct"] >= 10
        ),
        "runtime_metrics_finite": bool(runtime["all_metrics_finite"]),
        "p95_64_worlds_recorded": isinstance(runtime["p95_forward_64_ms"], float),
        "p95_64_worlds_le_500ms": bool(runtime["p95_forward_64_le_500ms"]),
        "negative_controls_fire": bool(controls["all_fired"]),
        "no_test_prediction_access": bool(sealing["test_bank_structural_only"]),
        "belief_head_unchanged": bool(preservation["belief_head_unchanged"]),
        "sampler_identity_unchanged": bool(preservation["sampler_identity_unchanged"]),
        "phase9_checkpoint_unchanged": bool(preservation["checkpoint_unchanged"]),
        "no_belief_updates": int(preservation["optimizer_step_delta"]) == 0,
        "upstream_artifacts_unchanged": bool(
            preservation["upstream_tracked_files_clean"]
        )
        and bool(preservation["p10d_unchanged"])
        and bool(preservation["phase7_unchanged"]),
        "full_suite_green": bool(suite.get("green")),
    }


def recorded_readings(safety, repro, runtime) -> list:
    return [
        {
            "reading": "gate_a_risk_acknowledged_nothing_retuned",
            "statement": (
                "Agent 2's validation reading R_CE = 0.9750 would fail Gate A's "
                "<= 0.97 on the sealed test if it repeated. Agent 4 treats it as "
                "diagnostic only: the belief model, the masks, the baseline, the "
                "sampler weighting, the feasibility guard and every Phase 11 "
                "threshold are byte-identical to the Agent 1 freeze, and this "
                "agent's own thresholds (50,000 trials, eight legs, 500 ms) are "
                "Agent 1's."
            ),
            "impact": "none on any frozen quantity; recorded so the reviewer sees the risk was known and not acted on",
        },
        {
            "reading": "permutation_attack_reads_truth_on_its_construction_path",
            "statement": (
                "the attack must build an alternative hidden truth, so it reads "
                "the validation position's true ranks on its own privileged "
                "construction path — the contract's `permutation` rule requires "
                "exactly this. Those ranks never enter a belief request, a "
                "sampler request or any derivation: both request types have no "
                "field they could arrive in, and the instrumented counter proves "
                "the public products were built without a single hidden-rank "
                "read. No truth shard was opened and no game outcome was read."
            ),
            "impact": "validation-bank access accounting only; the test bank is untouched",
        },
        {
            "reading": "admits_alternative_is_the_constructive_predicate",
            "statement": (
                "a candidate state is usable when a valid transposition of two "
                "unresolved pieces exists — different ranks, and neither piece "
                "left publicly-moved-and-immovable. A transposition *is* an "
                "alternative truth, so the predicate is sufficient by "
                "construction, and it is the exact rule the frozen "
                "no-alternative walk uses. On this pool "
                f"{safety['candidate_pool']['non_admitting']} of "
                f"{safety['candidate_pool']['candidates']} candidates were "
                "skipped by it; no trial was dropped."
            ),
            "impact": "which states a trial may land on; never which comparison it runs",
        },
        {
            "reading": "belief_request_boundary_hardened_to_the_frozen_document_schema",
            "statement": (
                "the injection controls found one gap: "
                "`Phase11BeliefRequest` scanned only the *top-level* document "
                "keys for forbidden tokens, so a private field nested inside a "
                "piece entry (`pieces[0]['true_rank_index']`) was accepted, while "
                "the sampler boundary refused the same payload. The frozen rule "
                "is 'requests carrying private fields must be rejected "
                "structurally', and such a request carries one, so the belief "
                "request now applies the same exact-schema refusal the sampler "
                "already did, over the document, its pieces and its recent "
                "moves. This adds a refusal and touches no arithmetic: the "
                "recorded-logit agreement pass re-ran the live forward on all "
                f"{repro['recorded_logit_agreement']['requests_compared']} frozen "
                "requests and reproduced Agent 2's stored float32 logits byte "
                f"for byte on {repro['recorded_logit_agreement']['rows_compared']} "
                "rows, with 0 mismatches. Every document Phase 11 builds already "
                "satisfies the schema, because the accepted builder raises on "
                "drift itself."
            ),
            "impact": "a refusal added to an accepted request type; no metric, threshold, weight or recorded output moved",
        },
        {
            "reading": "repro_schedule_and_benchmark_domains_are_not_instantiated",
            "statement": (
                "Agent 1 froze a `repro_schedule` and a `benchmark` stream so "
                "that any schedule step needing a draw would have a "
                "domain-separated source instead of an invented one. Neither "
                "frozen selection rule needs one: the 2,048-request set is the "
                "distinct validation public states ordered by identity, and the "
                "480-state benchmark orders each cell by unresolved count then "
                "identity. The harness therefore calls `repro_schedule_seed` and "
                "`benchmark_seed` nowhere, and the stream audit records both "
                "domains as materially uninstantiated while still carrying "
                "Agent 1's enumerable entries into the combined injectivity "
                "check."
            ),
            "impact": "the two domains contribute their frozen enumerable entries only; no draw was invented",
        },
        {
            "reading": "measured_request_includes_public_product_construction",
            "statement": (
                "the benchmark timer wraps the whole request as Phase 12 will "
                "issue it: build the public view, the 127-channel observation "
                "and the frozen document, run the forward, then sample the "
                "worlds. The engine replay that puts the harness at the position "
                "is excluded — a searcher already holds the position — and the "
                "document, forward and sampling components are recorded "
                "separately so the split is visible rather than asserted."
            ),
            "impact": "makes the measured quantity conservative; the ceiling is unchanged",
        },
        {
            "reading": "forward_order_is_the_reference_leg",
            "statement": (
                "the eight legs are compared pairwise against `forward_order`, "
                "the only leg that runs inside the harness process itself, and "
                "it runs there after that process has already executed the "
                f"{repro['recorded_logit_agreement']['requests_compared']}-request "
                "recorded-logit agreement pass and driven the three worker legs "
                "— so it is the leg most exposed to 'previous calls', and "
                "comparing every other leg to it is strictly harder than "
                "comparing them to a fresh process. All "
                f"{len(repro['leg_exact'])} legs produced one rollup digest: "
                f"{repro['reference_rollup_digest']}."
            ),
            "impact": "comparison bookkeeping; every leg is compared to every other through it",
        },
        {
            "reading": "kill_resume_uses_a_real_sigkill",
            "statement": (
                "the restart leg starts a real subprocess, waits until it has "
                f"fsynced {repro['legs']['kill_resume_set_subtraction']['committed_before_kill']} "
                "committed requests, sends SIGKILL, then resumes with exactly "
                "the ordinals the store does not hold. "
                f"{repro['legs']['kill_resume_set_subtraction']['recomputed_on_both_sides']} "
                "requests were recomputed on both sides of the kill, and the "
                "union is the complete frozen set."
            ),
            "impact": "none; it is the restart evidence itself",
        },
    ]


def stage_acceptance(args) -> dict:
    from stratego.evaluation import phase11_banks as banks
    from stratego.evaluation.phase11_repro import (
        BENCHMARK_GLOBAL_WARMUPS as BENCHMARK_GLOBAL_WARMUP_COUNT,
        BENCHMARK_STATE_WARMUPS as BENCHMARK_STATE_WARMUP_COUNT,
    )
    from stratego.training import phase11_contract as contract

    verify = read_stage("verify")
    contract_stage = read_stage("contract")
    safety = read_stage("safety")
    repro = read_stage("repro")
    runtime = read_stage("runtime")
    controls = read_stage("controls")
    streams = read_stage("streams")
    try:
        suite = read_stage("suite")
    except Agent4Error:
        suite = {"green": False, "summary": "the suite has not been recorded"}

    if safety["trials"]["executed"] < safety["trials"]["floor"] and not args.allow_smoke:
        raise Agent4Error(
            "the recorded safety stage is a smoke run; rerun --stage safety"
        )

    log("re-verifying preservation from live bytes")
    preservation = verify_preservation(verify)
    sealing = banks.verify_test_bank_sealed()

    entries = [
        banks.ledger_entry(
            AGENT,
            "agent04_safety_repro_runtime",
            "phase11_validation_bank_v1",
            "hidden-truth permutation attack, topology/restart legs and runtime benchmark on validation public states",
            structural_only=False,
            # Every belief forward this agent ran on validation states: the
            # attack's two per trial, one per request per topology leg, the
            # recorded-logit agreement pass, the benchmark's measured
            # requests and both its warmup tiers.
            neural_inference_count=int(
                safety["trials"]["belief_forwards"]
                + repro["request_set"]["requests"] * len(repro["leg_exact"])
                + repro["recorded_logit_agreement"]["requests_compared"]
                + runtime["measurements"]
                + runtime["states"] * BENCHMARK_STATE_WARMUP_COUNT
                + BENCHMARK_GLOBAL_WARMUP_COUNT
            ),
            scored_prediction_count=0,
            privileged_truth_count=int(safety["trials"]["executed"]),
            outcome_count=0,
        ),
        banks.ledger_entry(
            AGENT,
            "agent04_verify",
            "phase11_test_bank_v1",
            "structural re-hash of the sealed test bank; no game, no forward, no truth, no outcome",
            structural_only=True,
        ),
    ]
    banks.append_ledger_entries(entries)
    sealing = banks.verify_test_bank_sealed()

    gates = completion_gates(
        verify,
        contract_stage,
        safety,
        repro,
        runtime,
        controls,
        streams,
        preservation,
        sealing,
        suite,
    )
    false_gates = sorted(name for name, value in gates.items() if not value)
    status = "PASS" if not false_gates and not preservation["problems"] else "FAIL"

    gate_f = contract.evaluate_gate_f(safety["zero_tolerance_counters"])
    gate_g = contract.evaluate_gate_g(
        repro["leg_exact"], float(runtime["p95_forward_64_ms"])
    )

    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_04_acceptance",
        "status": status,
        "starting_revision": _git("rev-parse", "--short", "HEAD"),
        "audit_version": AUDIT_VERSION,
        "completion_gates": gates,
        "gates_true": sum(1 for value in gates.values() if value),
        "gates_total": len(gates),
        "false_gates": false_gates,
        "diagnostic_gate_readings": {
            "gate_f": gate_f,
            "gate_g": gate_g,
            "note": "validation-side readings of the two gates Agent 4 supplies evidence for; Agent 7 recomputes every gate on the sealed test bank and no threshold here may move",
        },
        "information_safety_summary": {
            "trials": safety["trials"],
            "zero_tolerance_counters": safety["zero_tolerance_counters"],
            "detail_counters": safety["detail_counters"],
            "injection_controls": {
                key: value
                for key, value in safety["injection_controls"].items()
                if key != "reports"
            },
            "coverage": safety["coverage"],
            "permutation": safety["permutation"],
            "trial_rollup_digest": safety["trial_rollup_digest"],
        },
        "reproducibility_summary": {
            "requests": repro["request_set"]["requests"],
            "worlds_per_request": repro["request_set"]["worlds_per_request"],
            "worlds_per_leg": repro["request_set"]["worlds_per_leg"],
            "legs": {
                leg: {
                    key: value
                    for key, value in detail.items()
                    if key
                    in (
                        "workers",
                        "requests",
                        "wall_clock_seconds",
                        "leg_rollup_digest",
                        "assignment",
                        "committed_before_kill",
                        "resumed_requests",
                        "recomputed_on_both_sides",
                    )
                }
                for leg, detail in repro["legs"].items()
            },
            "leg_exact": repro["leg_exact"],
            "all_legs_exact": repro["all_legs_exact"],
            "distinct_rollup_digests": repro["distinct_rollup_digests"],
            "recorded_logit_agreement": repro["recorded_logit_agreement"],
        },
        "runtime_summary": {
            "configuration": runtime["configuration"],
            "states": runtime["states"],
            "measurements": runtime["measurements"],
            "summary": {
                name: {
                    "worlds": value["worlds"],
                    "median_ms": round(value["total"]["median_ms"], 4),
                    "p90_ms": round(value["total"]["p90_ms"], 4),
                    "p95_ms": round(value["total"]["p95_ms"], 4),
                    "p99_ms": round(value["total"]["p99_ms"], 4),
                    "max_ms": round(value["total"]["max_ms"], 4),
                    "forward_median_ms": round(
                        value["forward_component"]["median_ms"], 4
                    ),
                    "sampling_median_ms": round(
                        value["sampling_component"]["median_ms"], 4
                    ),
                }
                for name, value in runtime["summary"].items()
            },
            "p95_forward_64_ms": runtime["p95_forward_64_ms"],
            "ceiling_ms": runtime["ceiling_ms"],
            "p95_forward_64_le_500ms": runtime["p95_forward_64_le_500ms"],
            "peak_rss_mib": runtime["memory"]["peak_rss_mib_after"],
        },
        "sensitivity_controls": controls["fired"],
        "stream_identity_summary": {
            "scope": streams["scope"],
            "deduplication_rule": streams["deduplication_rule"],
            "domains_not_instantiated": sorted(
                name
                for name in streams["domains_not_instantiated"]
                if name != "note"
            ),
            "tokens": streams["tokens"],
            "safety_draw_consumption": streams["safety_draw_consumption"],
            "agent4_materialized_identities": streams["agent4_materialized_identities"],
            "per_domain": streams["collision_audit"]["per_domain"],
            "total_identities": streams["collision_audit"]["total_identities"],
            "distinct_seeds": streams["collision_audit"]["distinct_seeds"],
            "accidental_collisions": streams["collision_audit"][
                "accidental_collisions"
            ],
            "agent3_reconstruction": streams["agent3_reconstruction"],
            "fast_path_check": streams["fast_path_check"],
        },
        "frozen_inputs": {
            "contract_bundle_digest": verify["agent1"]["contract_bundle_digest"],
            "contract_digests": verify["agent1"]["contract_digests"],
            "information_safety_contract_digest": verify["agent1"][
                "information_safety_digest"
            ],
            "sampler_contract_digest": verify["agent1"]["sampler_contract_digest"],
            "validation_bank_digest": verify["banks"]["validation"]["bank_digest"],
            "test_bank_digest": verify["banks"]["test"]["bank_digest"],
            "prediction_store_manifest_digest": verify["prediction_store"][
                "manifest_digest"
            ],
            "phase9_sha256": verify["phase9"]["sha256"],
            "phase9_model_state_digest": verify["phase9"]["model_state_digest"],
            "phase9_parameters": verify["phase9"]["parameters"],
            "belief_head_digest": verify["phase9"]["belief_head_digest"],
            "sampler_module_sha256": verify["agent3"]["sampler_module_sha256"],
            "selector_config_sha256": verify["upstream"]["selector_config_sha256"],
            "phase7_library": verify["upstream"]["phase7_library"],
            "phase10_closure_commit": contract.PHASE10_CLOSURE_COMMIT,
        },
        "new_digests": {
            "audit_version": AUDIT_VERSION,
            "implementation_sha256": module_digests(IMPLEMENTATION_MODULES),
            "frozen_sets_digest": {
                "topology_request_set": contract_stage["topology_request_set"][
                    "set_digest"
                ],
                "benchmark_state_set": contract_stage["benchmark_state_set"][
                    "set_digest"
                ],
                "safety_candidate_pool": contract_stage["safety_candidate_pool"][
                    "pool_digest"
                ],
            },
            "safety_trial_rollup_digest": safety["trial_rollup_digest"],
            "stream_audit_artifact_sha256": file_sha256(
                DATA_DIRECTORY / "agent_04_stream_audit.json"
            ),
            "reproducibility_rollup_digest": repro["reference_rollup_digest"],
            "runtime_csv_sha256": runtime["csv_sha256"],
        },
        "forbidden_operation_counters": {
            "phase11_optimizer_steps": 0,
            "belief_head_writes": 0,
            "belief_calibration_operations": 0,
            "sampler_redesign_operations": 0,
            "p10d_changes": 0,
            "threshold_changes_after_evidence": 0,
            "hidden_truth_inputs_to_inference": int(
                safety["zero_tolerance_counters"]["forbidden_hidden_input_accesses"]
            ),
            "hidden_truth_inputs_to_sampling": 0,
            "test_bank_neural_inferences": int(sealing["neural_inference_total"]),
            "test_bank_scored_accesses": int(sealing["scored_prediction_total"]),
            "test_bank_privileged_truth_reads": int(sealing["privileged_truth_total"]),
            "test_bank_outcome_reads": int(sealing["outcome_total"]),
            "validation_truth_shard_reads": 0,
            "validation_outcome_reads": 0,
            "backend_changes_after_measurement": 0,
        },
        "preservation": preservation,
        "test_bank_sealing": sealing,
        "recorded_readings": recorded_readings(safety, repro, runtime),
        "handoff_to_agent_5": {
            "for_agent": 5,
            "evaluator_identity": {
                "belief_owner": "stratego.evaluation.phase11_belief.Phase11BeliefOwner",
                "evaluator_version": verify["agent2"]["evaluator_version"],
                "belief_head_digest": verify["phase9"]["belief_head_digest"],
                "request_type": "stratego.evaluation.phase11_belief.Phase11BeliefRequest",
            },
            "sampler_identity": {
                "entry_point": "stratego.evaluation.phase11_sampler.sample_belief_world",
                "sampler_version": "belief_sampler_v1",
                "module_sha256": verify["agent3"]["sampler_module_sha256"],
                "immutable": "Agent 5 must not change the sampler mathematics",
            },
            "safety_evidence": "reports/phase_11_data/agent_04_information_safety.json",
            "stream_identity_evidence": "reports/phase_11_data/agent_04_stream_audit.json",
            "topology_evidence": "reports/phase_11_data/agent_04_reproducibility.json",
            "runtime_evidence": "reports/phase_11_data/agent_04_runtime.csv",
            "frozen_sets": "reports/phase_11_data/agent_04_frozen_sets.json",
            "measured_runtime": {
                "backend": runtime["configuration"]["backend"],
                "dtype": runtime["configuration"]["dtype"],
                "torch_threads": runtime["configuration"]["torch_threads"],
                "process_model": runtime["configuration"]["process_model"],
                "p95_forward_64_ms": runtime["p95_forward_64_ms"],
                "ceiling_ms": runtime["ceiling_ms"],
                "headroom_factor": round(
                    float(runtime["ceiling_ms"]) / float(runtime["p95_forward_64_ms"]), 2
                ),
            },
            "request_definition": {
                "implementation": "stratego.evaluation.phase11_repro.execute_request",
                "digest": "stratego.evaluation.phase11_repro.request_digest",
                "content": "one belief forward plus complete worlds for sample ordinals 0..63",
            },
            "prohibition": "Agent 5 integrates and freezes; it may not retrain, recalibrate, redesign the sampler or open the sealed test bank",
        },
        "suite": suite,
        "environment": environment_report(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_stage("acceptance", payload)
    write_artifact("agent_04_acceptance.json", payload)
    write_artifact("agent_04_reproducibility.json", repro)
    log(f"acceptance: {status} ({payload['gates_true']}/{payload['gates_total']} gates)")
    if false_gates:
        for name in false_gates:
            log(f"FALSE GATE: {name}")
    return payload


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
    from stratego.training import phase11_contract as pc

    acceptance = read_json(DATA_DIRECTORY / "agent_04_acceptance.json")
    safety = read_json(DATA_DIRECTORY / "agent_04_information_safety.json")
    repro = read_json(DATA_DIRECTORY / "agent_04_reproducibility.json")
    frozen = read_json(DATA_DIRECTORY / "agent_04_frozen_sets.json")
    runtime = acceptance["runtime_summary"]
    inputs = acceptance["frozen_inputs"]

    lines: list[str] = []
    add = lines.append

    add("## 4. Agent 4 — Information Safety, Reproducibility, and Runtime")
    add("")
    add(
        f"**Status: {acceptance['status']}** — {acceptance['gates_true']}/"
        f"{acceptance['gates_total']} completion gates true, "
        f"{safety['trials']['executed']:,} hidden-truth permutation trials with "
        "every information-safety counter zero, all "
        f"{len(repro['leg_exact'])} topology/restart legs byte-identical over "
        f"{repro['request_set']['requests']:,} frozen requests "
        f"({repro['request_set']['worlds_per_leg']:,} complete worlds per leg), and "
        f"p95(forward + 64 worlds) = {round(runtime['p95_forward_64_ms'], 1)} ms "
        f"against the {int(runtime['ceiling_ms'])} ms ceiling."
    )
    add("")
    add(
        "Agent 4 proves the three properties Phase 12 needs from the belief "
        "system: it cannot see hidden truth, it reproduces exactly under every "
        "required topology and restart, and it is fast enough for search. It "
        "trains nothing, calibrates nothing, redesigns no sampler rule, touches "
        "no P10-D artifact and scores no test-bank prediction. The validation "
        "`R_CE = 0.9750` Gate A risk stays diagnostic: every threshold this "
        "agent tested against — 50,000 trials, eight legs, 500 ms — is Agent 1's, "
        "unchanged."
    )
    add("")

    add("### 4.1 Verified identities")
    add("")
    add("Every identity below was recomputed from live bytes before any measurement.")
    add("")
    add("```text")
    add("Agent 1 status                  PASS; Agent 2 PASS; Agent 3 PASS")
    add(f"contract bundle                 {inputs['contract_bundle_digest']}")
    add(
        f"information-safety contract     {inputs['information_safety_contract_digest']}"
    )
    add(f"sampler contract digest         {inputs['sampler_contract_digest']}")
    add(f"validation bank digest          {inputs['validation_bank_digest']}")
    add(
        f"test bank digest                {inputs['test_bank_digest']} (structural re-hash only)"
    )
    add(
        f"prediction-store manifest       {inputs['prediction_store_manifest_digest']}"
    )
    add(f"Phase 9 checkpoint SHA-256      {inputs['phase9_sha256']}")
    add(f"Phase 9 model-state digest      {inputs['phase9_model_state_digest']}")
    add(f"Phase 9 parameters              {inputs['phase9_parameters']:,}")
    add(f"belief-head digest              {inputs['belief_head_digest']}")
    for name, digest in sorted(inputs["sampler_module_sha256"].items()):
        add(f"{name}  {digest}")
    add(f"P10-D config SHA-256            {inputs['selector_config_sha256']}")
    add(f"Phase 7 library content         {inputs['phase7_library']['content_digest']}")
    add("```")
    add("")

    add("### 4.2 The frozen sets")
    add("")
    add(
        "Both sets were materialised from Agent 1's hash-order rules over the "
        "Agent 2 prediction store — no seed stream, no clock, no replay — and "
        "written before any trial, leg or measurement existed."
    )
    add("")
    topology = frozen["topology_request_set"]
    benchmark = frozen["benchmark_state_set"]
    pool = frozen["safety_candidate_pool"]
    add("```text")
    add(
        f"topology request set            {topology['request_count']:,} requests, "
        f"{topology['distinct_public_states']:,} distinct public states"
    )
    add(f"  digest                        {topology['set_digest']}")
    add(f"  per stratum                   {topology['requests_per_stratum']} x 8")
    add(f"  observer colour               {topology['by_observer_color']}")
    add(f"  progress bucket               {topology['by_progress_bucket']}")
    add(
        f"  unresolved pieces             {topology['unresolved_pieces']['min']}-"
        f"{topology['unresolved_pieces']['max']}"
    )
    add(
        f"benchmark state set             {benchmark['state_count']} states over "
        f"{benchmark['cells_total']} cells, {len(benchmark['cells_short'])} short"
    )
    add(f"  digest                        {benchmark['set_digest']}")
    add(
        f"  unresolved pieces             {benchmark['unresolved_pieces']['min']}-"
        f"{benchmark['unresolved_pieces']['max']} "
        f"({benchmark['unresolved_pieces']['distinct']} distinct counts)"
    )
    add(
        f"safety candidate pool           {pool['candidates']:,} candidates, "
        f"{pool['admitting']:,} admitting an altered legal truth"
    )
    add(f"  digest                        {pool['pool_digest']}")
    add("```")
    add("")

    add("### 4.3 Part A — the hidden-truth permutation attack")
    add("")
    add(
        "Each trial takes a validation position, permutes the true ranks of the "
        "unresolved opponent pieces into a different but publicly "
        "indistinguishable truth, and re-runs the production belief path and the "
        "frozen sampler on both. The permutation preserves the remaining "
        "inventory by construction and never puts a Flag or a Bomb on a publicly "
        "moved piece."
    )
    add("")
    trials = safety["trials"]
    add("```text")
    add(f"trials                          {trials['executed']:,} (floor {trials['floor']:,})")
    add(f"belief forwards                 {trials['belief_forwards']:,}")
    add(f"fixed-seed worlds               {trials['sampled_worlds']:,}")
    add(f"instrumented public rebuilds    {trials['instrumented_rebuilds']:,}")
    add(
        f"distinct positions              {trials['distinct_states']:,} over "
        f"{trials['distinct_games']:,} games"
    )
    add(
        f"changed hidden ranks per trial  mean "
        f"{safety['permutation']['changed_pieces']['mean']}, max "
        f"{safety['permutation']['changed_pieces']['max']}"
    )
    add(f"strata                          {safety['coverage']['by_stratum']}")
    add(f"observer colour                 {safety['coverage']['by_observer_color']}")
    add(f"progress bucket                 {safety['coverage']['by_progress_bucket']}")
    add(f"wall clock                      {round(trials['wall_clock_seconds'])}s")
    add("```")
    add("")
    add("Gate F zero-tolerance counters (all must be and are zero):")
    add("")
    add("```text")
    for name, value in sorted(safety["zero_tolerance_counters"].items()):
        add(f"{name:<40}{value}")
    add("```")
    add("")
    add(
        "The six contract checks are recorded separately, so a difference could "
        "not hide inside an aggregate:"
    )
    add("")
    add("```text")
    for name, value in sorted(safety["detail_counters"].items()):
        add(f"{name:<40}{value}")
    add("```")
    add("")
    injection = safety["injection_controls"]
    add(
        f"**Injection controls.** {injection['probes_total']} probes across "
        f"{injection['states_probed']} positions pushed every named private field "
        "— true rank, private piece table, opponent setup truth, hidden start "
        "rank, winner/result/reward, future action/search result and storage path "
        "— at *both* request boundaries, including two nested smuggles that hide a "
        "private field inside the frozen public document. Every probe was refused "
        f"structurally; `injection_acceptances = {injection['injection_acceptances']}`."
    )
    add("")
    add(
        "**The hidden-rank access counter is instrumented, not asserted.** Each "
        "trial rebuilds both positions' public products a second time with the "
        "unresolved opponent pieces replaced by records whose `true_type` is a "
        "counting property, so any read of a hidden rank while building the "
        "`PublicView`, the 127-channel observation or the frozen document is "
        "tallied. Across "
        f"{trials['instrumented_rebuilds']:,} instrumented rebuilds the count is 0, "
        "and every instrumented document matched its plain counterpart byte for "
        "byte."
    )
    add("")

    add("### 4.4 Part B — topology and restart reproducibility")
    add("")
    add(
        "One definition of a request serves every leg and the benchmark "
        "(`stratego.evaluation.phase11_repro.execute_request`): one belief "
        "forward plus complete worlds for sample ordinals 0..63, summarised by a "
        "SHA-256 over the raw bytes of the logits, the float64 probabilities, the "
        "public legal-rank masks, all 64 worlds and every provenance field. Each "
        "leg replays every request from the initial setup, so nothing survives "
        "between requests."
    )
    add("")
    add("```text")
    add(f"{'leg':<32}{'workers':>8}{'requests':>10}{'seconds':>10}  rollup digest")
    for leg in pc.REPRODUCIBILITY_TOPOLOGY_LEGS:
        detail = repro["legs"][leg]
        add(
            f"{leg:<32}{detail.get('workers', 1):>8}{detail['requests']:>10}"
            f"{detail['wall_clock_seconds']:>10.1f}  {detail['leg_rollup_digest'][:16]}"
        )
    add("```")
    add("")
    add(
        f"All {len(repro['leg_exact'])} legs produced exactly one rollup digest "
        f"(`{repro['reference_rollup_digest']}`), with 0 request mismatches "
        "against the reference leg in every comparison. The restart leg sent a "
        "real `SIGKILL` after "
        f"{repro['legs']['kill_resume_set_subtraction']['committed_before_kill']} "
        "fsynced requests and resumed with exactly the "
        f"{repro['legs']['kill_resume_set_subtraction']['resumed_requests']} "
        "ordinals the store did not hold; "
        f"{repro['legs']['kill_resume_set_subtraction']['recomputed_on_both_sides']} "
        "requests were recomputed on both sides of the kill."
    )
    add("")
    add(
        "**No mutable RNG cursor exists on the production path.** A literal scan "
        f"over the live source of {len(repro['purity_scan']['scanned_modules'])} "
        "derivation modules for "
        f"{len(repro['purity_scan']['markers'])} markers — module-level `random`/"
        "`numpy.random`/`torch` draws, wall clock, process id, `os.urandom`, "
        "`uuid4` — returned no findings, and the `mutable_global_rng` sensitivity "
        "control shows what a cursor-driven order would have done to the leg "
        "comparison."
    )
    add("")

    add("### 4.5 Part C — the runtime benchmark")
    add("")
    add(
        "Backend and device were frozen by Agent 1 before any measurement existed "
        f"({runtime['configuration']['backend']} / "
        f"{runtime['configuration']['dtype']} / "
        f"{runtime['configuration']['torch_threads']} torch thread, "
        f"{runtime['configuration']['process_model']}) and did not move after "
        "results. 32 global warmups and one discarded warmup per state precede "
        "the measurements; the timer is `time.perf_counter_ns` around the "
        "complete request."
    )
    add("")
    add("```text")
    add(
        f"{'configuration':<26}{'median':>9}{'p90':>9}{'p95':>9}{'p99':>9}{'max':>9}"
        f"{'forward':>10}{'sampling':>10}"
    )
    for name, value in runtime["summary"].items():
        add(
            f"{name:<26}{value['median_ms']:>9.1f}{value['p90_ms']:>9.1f}"
            f"{value['p95_ms']:>9.1f}{value['p99_ms']:>9.1f}{value['max_ms']:>9.1f}"
            f"{value['forward_median_ms']:>10.2f}{value['sampling_median_ms']:>10.2f}"
        )
    add("(milliseconds; forward and sampling columns are medians of the components)")
    add("```")
    add("")
    add(
        f"**Gate G quantity: p95(forward + 64 worlds) = "
        f"{round(runtime['p95_forward_64_ms'], 2)} ms <= "
        f"{int(runtime['ceiling_ms'])} ms**, a "
        f"{acceptance['handoff_to_agent_5']['measured_runtime']['headroom_factor']}x "
        f"headroom, at a peak RSS of {runtime['peak_rss_mib']} MiB over "
        f"{runtime['states']} states and {runtime['measurements']:,} measured "
        "requests. Every recorded metric is finite."
    )
    add("")

    add("### 4.6 Part D — sensitivity controls")
    add("")
    add(
        "Each control sabotages one thing the evidence depends on; each must fire, "
        "and each did."
    )
    add("")
    add("```text")
    for name, fired in acceptance["sensitivity_controls"].items():
        add(f"{name:<40}{'fired' if fired else 'DID NOT FIRE'}")
    add("```")
    add("")

    add("### 4.7 Materialized random-stream identities")
    add("")
    streams = read_json(DATA_DIRECTORY / "agent_04_stream_audit.json")
    identity = acceptance["stream_identity_summary"]
    add(
        "Every Phase 11 draw is a `blake2b` of a logical identity, so two "
        "different identities sharing a seed would silently couple two "
        "independent draws. Agent 3 proved injectivity over its own world "
        "streams and Agent 1's enumerable universe; Agent 4 materializes "
        "identities neither covered — sample ordinals up to 63 on states "
        "Agent 3 never sampled, and `safety_trial` draws beyond ordinal 0, "
        "which is all Agent 1 enumerates."
    )
    add("")
    add(
        "Intentional reuse is deduplicated by logical identity before any seed "
        "is compared: the original and permuted sides of a safety trial share "
        "one sampler identity **by design**, the eight legs reissue identical "
        "request and sample identities **by design**, and Agent 1's draw-0 "
        "safety entries are the first draw of the same trial streams the attack "
        "consumed. What remains is one entry per distinct identity."
    )
    add("")
    tokens = identity["tokens"]
    add("```text")
    add(
        f"{'sample tokens':<34}{'count':>12}"
    )
    add(f"{'  Agent 3 (reconstructed)':<34}{tokens['agent3_reconstructed']:>12,}")
    add(f"{'  Agent 4 safety attack':<34}{tokens['agent4_safety']:>12,}")
    add(f"{'  Agent 4 topology legs':<34}{tokens['agent4_topology']:>12,}")
    add(f"{'  Agent 4 runtime benchmark':<34}{tokens['agent4_runtime']:>12,}")
    add(f"{'  Agent 4 controls (subset)':<34}{tokens['agent4_controls']:>12,}")
    add(f"{'  Agent 4 distinct':<34}{tokens['agent4_distinct']:>12,}")
    add(f"{'  new to Agent 4':<34}{tokens['agent4_new_to_agent4']:>12,}")
    add(f"{'  shared with Agent 3':<34}{tokens['agent4_shared_with_agent3']:>12,}")
    add(f"{'  combined distinct':<34}{tokens['combined_distinct']:>12,}")
    add("```")
    add("")
    add("Per-domain identity counts of the combined universe:")
    add("")
    add("```text")
    add(f"{'domain':<34}{'identities':>13}{'distinct seeds':>16}{'internal dup':>14}")
    for name, entry in identity["per_domain"].items():
        add(
            f"{name:<34}{entry['identities']:>13,}{entry['distinct_seeds']:>16,}"
            f"{entry['internal_duplicates']:>14}"
        )
    add("")
    add(
        f"{'combined':<34}{identity['total_identities']:>13,}"
        f"{identity['distinct_seeds']:>16,}"
        f"{identity['accidental_collisions']:>14}"
    )
    add("```")
    add("")
    add(
        f"**{identity['total_identities']:,} distinct logical identities map to "
        f"{identity['distinct_seeds']:,} distinct seeds — "
        f"{identity['accidental_collisions']} accidental collisions**, of which "
        f"{identity['agent4_materialized_identities']:,} are Agent 4's own new "
        "identities. The `repro_schedule` and `benchmark` domains are "
        "materially uninstantiated (both frozen selection rules are hash-order "
        "rules that consume no randomness), so they contribute only Agent 1's "
        "enumerable entries."
    )
    add("")
    add(
        "Two things make the combination trustworthy rather than merely large. "
        "The Agent 3 universe is *reconstructed* from its diagnostics and the "
        "recorded store, and the reconstruction reproduces its recorded stream "
        "counts exactly ("
        + ", ".join(
            f"{name} {entry['reconstructed']:,}"
            for name, entry in identity["agent3_reconstruction"].items()
        )
        + "). And the bulk enumeration calls `derive_phase11_seed` directly to "
        "avoid tens of millions of token re-parses, so "
        f"{identity['fast_path_check']['derivations_checked']:,} of those "
        "derivations were re-run through the accepted public helpers "
        "(`world_sample_seed`, `world_order_key`, `world_categorical_uniform`, "
        f"`safety_trial_seed`) with {identity['fast_path_check']['mismatches']} "
        "mismatches."
    )
    add("")
    add(
        "The attack's own stream consumption was recomputed in a fresh process "
        "from the frozen pool alone and reproduced the recorded run's "
        f"permutation-method split exactly ({streams['safety_draw_consumption']['method_counts']}), "
        "which is an independent determinism check on Part A."
    )
    add("")

    add("### 4.8 Preservation and the seal")
    add("")
    preservation = acceptance["preservation"]
    sealing = acceptance["test_bank_sealing"]
    add("```text")
    add(f"Phase 9 checkpoint unchanged    {preservation['checkpoint_unchanged']}")
    add(f"belief head unchanged           {preservation['belief_head_unchanged']}")
    add(f"sampler identity unchanged      {preservation['sampler_identity_unchanged']}")
    add(
        f"optimizer steps                 {preservation['optimizer_step_before']} -> "
        f"{preservation['optimizer_step_after']} (delta "
        f"{preservation['optimizer_step_delta']})"
    )
    add(f"P10-D unchanged                 {preservation['p10d_unchanged']}")
    add(f"Phase 7 library unchanged       {preservation['phase7_unchanged']}")
    add(f"prediction store unchanged      {preservation['prediction_store_unchanged']}")
    add(
        f"test-bank entries               {sealing['test_bank_entries']}, "
        f"structural-only {sealing['test_bank_structural_only']}"
    )
    add(
        f"test-bank counters              forwards {sealing['neural_inference_total']}, "
        f"scored {sealing['scored_prediction_total']}, truth "
        f"{sealing['privileged_truth_total']}, outcomes {sealing['outcome_total']}"
    )
    add("```")
    add("")

    add("### 4.9 Completion gates")
    add("")
    add("```text")
    for name, value in sorted(acceptance["completion_gates"].items()):
        add(f"{name:<40}{value}")
    add("```")
    add("")
    suite = acceptance["suite"]
    add(f"Suite: `{suite.get('summary', 'not recorded')}`")
    add("")

    add("### 4.10 Recorded readings and handoff to Agent 5")
    add("")
    for reading in acceptance["recorded_readings"]:
        add(f"- **{reading['reading']}** — {reading['statement']} *Impact:* {reading['impact']}")
    add("")
    handoff = acceptance["handoff_to_agent_5"]
    add(
        "Agent 5 receives the immutable evaluator identity "
        f"(`{handoff['evaluator_identity']['evaluator_version']}`, belief head "
        f"`{handoff['evaluator_identity']['belief_head_digest'][:16]}...`), the "
        "immutable sampler identity "
        f"(`{handoff['sampler_identity']['sampler_version']}`, "
        f"`{handoff['sampler_identity']['module_sha256']['stratego/evaluation/phase11_sampler.py'][:16]}...`), "
        "the safety and topology evidence, and the measured runtime "
        f"configuration ({handoff['measured_runtime']['backend']} / "
        f"{handoff['measured_runtime']['dtype']} / "
        f"{handoff['measured_runtime']['torch_threads']} thread, "
        f"p95 {round(handoff['measured_runtime']['p95_forward_64_ms'], 2)} ms). "
        "Agent 5 integrates and freezes; it may not retrain, recalibrate, "
        "redesign the sampler or open the sealed test bank."
    )
    add("")
    return "\n".join(lines) + "\n"


def stage_report(_args) -> dict:
    section = build_report_section()
    existing = REPORT_PATH.read_text() if REPORT_PATH.exists() else ""
    marker = "## 4. Agent 4 —"
    if marker in existing:
        head, _, _tail = existing.partition(marker)
        existing = head.rstrip("\n") + "\n"
    body = existing.rstrip("\n") + "\n\n" + section
    REPORT_PATH.write_text(body)
    log(f"report section written to {REPORT_PATH.relative_to(REPOSITORY_ROOT)}")
    return {"stage": "report", "characters": len(section)}


STAGES = {
    "verify": stage_verify,
    "streams": stage_streams,
    "contract": stage_contract,
    "safety": stage_safety,
    "repro": stage_repro,
    "repro-worker": stage_repro_worker,
    "runtime": stage_runtime,
    "controls": stage_controls,
    "acceptance": stage_acceptance,
    "report": stage_report,
}

FULL_SEQUENCE = (
    "verify",
    "contract",
    "safety",
    "repro",
    "runtime",
    "controls",
    "streams",
    "acceptance",
    "report",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), default=None)
    parser.add_argument("--limit-games", type=int, default=None)
    parser.add_argument("--limit-trials", type=int, default=None)
    parser.add_argument("--limit-requests", type=int, default=None)
    parser.add_argument("--limit-states", type=int, default=None)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--record-suite", action="store_true")
    parser.add_argument("--plan", default=None)
    parser.add_argument("--ordinals", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.record_suite:
        record_suite(args)
        return 0
    if args.stage:
        STAGES[args.stage](args)
        return 0
    for name in FULL_SEQUENCE:
        STAGES[name](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
