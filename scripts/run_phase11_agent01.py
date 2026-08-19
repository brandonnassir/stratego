#!/usr/bin/env python3
"""Phase 11 Agent 1 harness: contracts, seeds, banks, metrics, gates.

Verifies every accepted upstream identity from live bytes (the Phase 10
Agents 1-7 acceptance chain and formal closure commit, the accepted Phase 9
checkpoint's file SHA / model-state digest / parameter count / finiteness /
belief-head tensor identity, the Phase 9 contract+amendment chain, the
frozen P10-D selector config / utility / scaler / phase10_system_v1, the
Phase 8 anchor export, the Phase 7 library digests and splits, and the
observation/model contracts), then freezes the complete Phase 11 experiment
and writes the four Agent 1 artifacts:

    reports/phase_11_data/agent_01_phase11_contract.json
    reports/phase_11_data/agent_01_validation_bank.json
    reports/phase_11_data/agent_01_test_bank.json
    reports/phase_11_data/agent_01_acceptance.json

What this script is and is not
------------------------------
It freezes the *pre-evidence contract*. It plays no Phase 11 game, runs no
belief inference, scores no prediction, samples no world and reads no
outcome. The only thing it does to the sealed final-test bank is build,
hash and structurally audit it — the `structural_build`/`structural_audit`
purposes the sealing rules explicitly allow — and every access is written
to the append-only Phase 11 bank ledger.

Usage::

    python scripts/run_phase11_agent01.py                 # every stage
    python scripts/run_phase11_agent01.py --stage verify  # one stage
    python scripts/run_phase11_agent01.py --record-suite  # run + record the suite
"""

from __future__ import annotations

import argparse
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

AGENT = 1
PHASE = 11
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_11_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_11_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase11" / "agent01"

CONTRACT_ARTIFACT = DATA_DIRECTORY / "agent_01_phase11_contract.json"
VALIDATION_BANK_ARTIFACT = DATA_DIRECTORY / "agent_01_validation_bank.json"
TEST_BANK_ARTIFACT = DATA_DIRECTORY / "agent_01_test_bank.json"
ACCEPTANCE_ARTIFACT = DATA_DIRECTORY / "agent_01_acceptance.json"

CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "selfplay_c1_v1.pt"

SECTION_MARKER = "## 1. Agent 1 — Contracts, Seeds, Banks, Metrics, and Acceptance Freeze"

#: The full suite as measured immediately before any Phase 11 Agent 1
#: change, at the Phase 10 formal closure commit.
TESTS_BEFORE = {
    "command": ".venv/bin/python -m pytest tests -q",
    "summary": "5199 passed, 3 skipped in 313.94s (0:05:13)",
    "passed": 5199,
    "failed": 0,
    "skipped": 3,
    "seconds": 313.94,
    "measured_at_commit": "17188a5",
}

#: The Phase 10 acceptance chain this phase builds on.
PHASE10_ACCEPTANCE_SOURCES = {
    1: "agent_01_acceptance.json",
    2: "agent_02_acceptance.json",
    3: "agent_03_acceptance.json",
    4: "agent_04_acceptance.json",
    5: "agent_05_acceptance.json",
    6: "agent_06_acceptance.json",
    7: "agent_07_final_acceptance.json",
}


class Agent1Error(RuntimeError):
    """A precondition or frozen identity failed. Always raised, never patched."""


# ---------------------------------------------------------------------------
# Environment and helpers
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
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return path


def read_stage(name: str) -> dict:
    path = stage_path(name)
    if not path.exists():
        raise Agent1Error(f"stage {name!r} has not run yet ({path} is missing)")
    return json.loads(path.read_text())


def require(condition: bool, message: str, problems: list) -> bool:
    if not condition:
        problems.append(message)
    return bool(condition)


def log(message: str) -> None:
    print(f"[phase11:agent1] {message}", flush=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# 1. Verification
# ---------------------------------------------------------------------------


def verify_phase10_closure(problems: list) -> dict:
    """Phase 10 must be formally closed: agents 1-7 PASS at a real commit."""
    from stratego.training.phase11_contract import PHASE10_CLOSURE_COMMIT

    directory = REPOSITORY_ROOT / "reports" / "phase_10_data"
    records = {}
    for agent, name in PHASE10_ACCEPTANCE_SOURCES.items():
        payload = read_json(directory / name)
        gates = payload.get("completion_gates", {})
        false_gates = sorted(key for key, value in gates.items() if not value)
        expected_status = "PASS-NONINFERIOR" if agent == 7 else "PASS"
        records[str(agent)] = {
            "artifact": name,
            "status": payload.get("status"),
            "gates_total": len(gates),
            "gates_true": sum(bool(value) for value in gates.values()),
            "false_gates": false_gates,
        }
        require(
            payload.get("status") == expected_status,
            f"Phase 10 agent {agent} status is {payload.get('status')!r}, "
            f"expected {expected_status!r}",
            problems,
        )
        require(
            not false_gates,
            f"Phase 10 agent {agent} has false completion gates: {false_gates}",
            problems,
        )
    final = read_json(directory / PHASE10_ACCEPTANCE_SOURCES[7])
    require(
        final.get("recommendation") == "PASS-NONINFERIOR",
        "Phase 10 Agent 7 recommendation is not PASS-NONINFERIOR",
        problems,
    )

    closure_commit = _git("rev-parse", "--short", PHASE10_CLOSURE_COMMIT)
    closure_subject = _git("log", "-1", "--format=%s", PHASE10_CLOSURE_COMMIT)
    require(
        closure_commit == PHASE10_CLOSURE_COMMIT,
        f"the recorded Phase 10 closure commit {PHASE10_CLOSURE_COMMIT} does not "
        "resolve in live Git state",
        problems,
    )
    require(
        "closure" in closure_subject.lower(),
        f"commit {PHASE10_CLOSURE_COMMIT} does not read as a closure commit: "
        f"{closure_subject!r}",
        problems,
    )
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PHASE10_CLOSURE_COMMIT, "HEAD"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
    ).returncode
    require(
        ancestry == 0,
        f"the closure commit {PHASE10_CLOSURE_COMMIT} is not an ancestor of HEAD",
        problems,
    )
    tracked_dirty = [
        line
        for line in _git("status", "--porcelain").splitlines()
        if line and not line.startswith("??")
    ]
    return {
        "agents": records,
        "closure_commit": closure_commit,
        "closure_subject": closure_subject,
        "closure_is_ancestor_of_head": ancestry == 0,
        "tracked_tree_dirty_entries": tracked_dirty,
        "untracked_entries": [
            line
            for line in _git("status", "--porcelain").splitlines()
            if line.startswith("??")
        ],
    }


def verify_phase9_checkpoint(problems: list) -> dict:
    """File SHA, state digest, parameters, finiteness — and the belief head."""
    import torch

    from stratego.model.architecture_configs import config_digests
    from stratego.training import phase9_behavior, phase9_checkpoint
    from stratego.training import phase10_contract as p10
    from stratego.training import phase11_contract as pc

    observed_sha = file_sha256(CHECKPOINT_PATH)
    payload = phase9_checkpoint.read_phase9_payload(CHECKPOINT_PATH)
    model = phase9_checkpoint.model_from_payload(payload)
    state_dict = model.state_dict()
    state_digest = phase9_behavior.state_dict_digest(model)
    parameters = sum(tensor.numel() for tensor in model.parameters())
    finite = all(bool(torch.isfinite(tensor).all()) for tensor in state_dict.values())
    c1_digest = config_digests()["C1"]
    global_step = payload.get("global_optimizer_step")

    observed_names = tuple(sorted(name for name in state_dict if name.startswith("belief_output.")))
    hasher = hashlib.sha256()
    shapes = {}
    head_finite = True
    for name in observed_names:
        tensor = state_dict[name]
        shapes[name] = tuple(tensor.shape)
        hasher.update(name.encode())
        array = tensor.detach().to("cpu", torch.float32).contiguous().numpy()
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
        head_finite = head_finite and bool(torch.isfinite(tensor).all())
    head_digest = hasher.hexdigest()

    require(
        observed_sha == p10.ACCEPTED_PHASE9_CHECKPOINT_SHA256,
        f"Phase 9 checkpoint SHA {observed_sha} != accepted",
        problems,
    )
    require(
        state_digest == p10.ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
        f"Phase 9 model-state digest {state_digest} != accepted",
        problems,
    )
    require(
        parameters == p10.ACCEPTED_PHASE9_PARAMETERS,
        f"Phase 9 parameter count {parameters} != {p10.ACCEPTED_PHASE9_PARAMETERS}",
        problems,
    )
    require(finite, "Phase 9 model carries a non-finite parameter", problems)
    require(
        c1_digest == p10.ACCEPTED_C1_CONFIG_DIGEST,
        f"C1 config digest {c1_digest} != accepted",
        problems,
    )
    require(
        global_step == pc.ACCEPTED_GLOBAL_OPTIMIZER_STEP,
        f"global optimizer step {global_step} != recorded "
        f"{pc.ACCEPTED_GLOBAL_OPTIMIZER_STEP}",
        problems,
    )
    require(
        observed_names == pc.BELIEF_HEAD_TENSOR_NAMES,
        f"belief-head tensors {observed_names} != frozen {pc.BELIEF_HEAD_TENSOR_NAMES}",
        problems,
    )
    require(
        shapes == pc.BELIEF_HEAD_TENSOR_SHAPES,
        f"belief-head shapes {shapes} != frozen",
        problems,
    )
    require(
        head_digest == pc.ACCEPTED_BELIEF_HEAD_DIGEST,
        f"belief-head digest {head_digest} != frozen {pc.ACCEPTED_BELIEF_HEAD_DIGEST}",
        problems,
    )
    require(head_finite, "belief-head tensors carry a non-finite value", problems)
    del model
    return {
        "path": str(CHECKPOINT_PATH.relative_to(REPOSITORY_ROOT)),
        "sha256": observed_sha,
        "model_state_digest": state_digest,
        "parameters": parameters,
        "all_parameters_finite": finite,
        "c1_config_digest": c1_digest,
        "global_optimizer_step": global_step,
        "belief_head_tensor_names": list(observed_names),
        "belief_head_tensor_shapes": {name: list(shape) for name, shape in shapes.items()},
        "belief_head_digest": head_digest,
        "belief_head_finite": head_finite,
        "rl_iteration": payload.get("rl_iteration"),
        "snapshot_role": payload.get("snapshot_role"),
    }


def verify_phase9_chain(problems: list) -> dict:
    """The accepted Phase 9 contract and both amendments, recomputed live."""
    from stratego.training import phase9_amendment, phase9_amendment_v2, phase9_contract
    from stratego.training import phase10_contract as p10

    observed = {
        "contract_digest": phase9_contract.contract_digest(),
        "amendment_v1_digest": phase9_amendment.amendment_digest(),
        "amendment_v2_digest": phase9_amendment_v2.amendment_digest(),
    }
    expected = {
        "contract_digest": p10.ACCEPTED_PHASE9_CONTRACT_DIGEST,
        "amendment_v1_digest": p10.ACCEPTED_PHASE9_AMENDMENT_V1_DIGEST,
        "amendment_v2_digest": p10.ACCEPTED_PHASE9_AMENDMENT_V2_DIGEST,
    }
    for key, value in expected.items():
        require(
            observed[key] == value,
            f"Phase 9 chain {key} {observed[key]} != accepted {value}",
            problems,
        )
    return {"observed": observed, "expected": expected, "chain_intact": observed == expected}


def verify_phase10_selector(problems: list) -> dict:
    """P10-D config, utility, scaler and phase10_system_v1, from live bytes."""
    from stratego.evaluation.phase11_banks import Phase11SetupSources
    from stratego.training import phase11_contract as pc
    from stratego.training.phase10_selector import candidate

    artifact_problems = Phase11SetupSources.verify_selector_artifacts()
    for finding in artifact_problems:
        problems.append(f"selector artifacts: {finding}")

    config_path = DATA_DIRECTORY.parent / "phase_10_data" / "agent_05_frozen_selector_config.json"
    config_sha = file_sha256(config_path)
    config = read_json(config_path)
    winner = config.get("winner", {})
    require(
        winner.get("candidate_id") == pc.ACCEPTED_SELECTOR_CANDIDATE_ID
        and winner.get("utility_model") == pc.ACCEPTED_SELECTOR_UTILITY_MODEL
        and float(winner.get("temperature", 0.0)) == pc.ACCEPTED_SELECTOR_TEMPERATURE,
        f"frozen winner row {winner!r} disagrees with the accepted P10-D",
        problems,
    )
    frozen_candidate = candidate(pc.ACCEPTED_SELECTOR_CANDIDATE_ID)
    require(
        frozen_candidate.selector_identity == pc.ACCEPTED_SELECTOR_IDENTITY,
        f"selector identity {frozen_candidate.selector_identity!r} != accepted",
        problems,
    )

    utility_path = REPOSITORY_ROOT / "checkpoints" / "phase10" / "setup_utility_v1.json"
    utility_sha = file_sha256(utility_path)
    utility = read_json(utility_path)

    manifest = read_json(
        DATA_DIRECTORY.parent / "phase_10_data" / "agent_06_production_selector_manifest.json"
    )
    from stratego.training.phase10_contract import document_digest as p10_document_digest

    system_digest = p10_document_digest(manifest["phase10_system_v1"])
    require(
        system_digest == pc.ACCEPTED_PHASE10_SYSTEM_DIGEST,
        f"phase10_system_v1 digest {system_digest} != accepted",
        problems,
    )
    require(
        manifest["phase10_system_v1_digest"] == system_digest,
        "the stored phase10_system_v1 digest disagrees with its own document",
        problems,
    )
    return {
        "config_sha256": config_sha,
        "winner": {key: winner.get(key) for key in ("candidate_id", "utility_model", "temperature")},
        "selector_identity": frozen_candidate.selector_identity,
        "utility_file_sha256": utility_sha,
        "model_T_coefficient_digest": utility["models"]["model_T"]["coefficient_digest"],
        "trait_scaler_digest": utility["scaler_digest"],
        "phase10_system_digest": system_digest,
        "artifact_problems": artifact_problems,
    }


def verify_phase8_anchor(problems: list) -> dict:
    """The Phase 8 anchor export the phase8_anchor stratum plays."""
    from stratego.training import phase11_contract as pc

    path = REPOSITORY_ROOT / pc.ACCEPTED_ANCHOR_EXPORT_PATH
    require(path.exists(), f"anchor export {path} is missing", problems)
    observed = file_sha256(path) if path.exists() else None
    require(
        observed == pc.ACCEPTED_ANCHOR_EXPORT_SHA256,
        f"anchor export SHA {observed} != accepted {pc.ACCEPTED_ANCHOR_EXPORT_SHA256}",
        problems,
    )
    recorded = read_json(
        REPOSITORY_ROOT / "reports" / "phase_9_data" / "agent_01_acceptance.json"
    )
    accepted_record = None
    stack = [recorded]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "export_sha256" and isinstance(value, str):
                    accepted_record = value
                stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    require(
        accepted_record == observed,
        "the anchor export SHA disagrees with the accepted Phase 9 record",
        problems,
    )
    return {
        "path": pc.ACCEPTED_ANCHOR_EXPORT_PATH,
        "sha256": observed,
        "accepted_phase9_record": accepted_record,
    }


def verify_phase7_library(problems: list) -> dict:
    """Library digests, split arithmetic and the accepted neutral profile."""
    from stratego.setups.contracts import (
        LIBRARY_JSONL_PATH,
        LIBRARY_MANIFEST_PATH,
        parse_base_setup_id,
        split_for_base_index,
    )
    from stratego.setups.library import (
        entry_metadata_digest,
        library_content_digest,
        manifest_digest,
        read_library_jsonl,
        read_manifest,
    )
    from stratego.setups.sampler import NEUTRAL_PROFILE
    from stratego.training import phase10_contract as p10

    entries = read_library_jsonl(LIBRARY_JSONL_PATH)
    manifest = read_manifest(LIBRARY_MANIFEST_PATH)
    observed = {
        "content_digest": library_content_digest(entries),
        "metadata_digest": entry_metadata_digest(entries),
        "manifest_digest": manifest["manifest_digest"],
        "bases": len(entries),
    }
    require(
        observed["manifest_digest"] == p10.PHASE7_LIBRARY_MANIFEST_DIGEST,
        "Phase 7 library manifest digest != accepted",
        problems,
    )
    require(
        manifest_digest(manifest) == manifest["manifest_digest"],
        "Phase 7 library manifest does not re-hash to its own recorded digest",
        problems,
    )
    require(
        observed["content_digest"] == p10.PHASE7_LIBRARY_CONTENT_DIGEST,
        "Phase 7 library content digest != accepted",
        problems,
    )
    require(
        observed["metadata_digest"] == p10.PHASE7_LIBRARY_METADATA_DIGEST,
        "Phase 7 library metadata digest != accepted",
        problems,
    )
    splits = {"train": 0, "validation": 0, "test": 0}
    for entry in entries:
        _, _, base_index = parse_base_setup_id(entry.base_setup_id)
        splits[split_for_base_index(base_index)] += 1
    observed["splits"] = dict(splits)
    require(
        splits == {"train": 6400, "validation": 800, "test": 800},
        f"Phase 7 splits {splits} != accepted 6400/800/800",
        problems,
    )
    observed["neutral_profile"] = {
        "name": NEUTRAL_PROFILE.name,
        "reflection_probability": NEUTRAL_PROFILE.reflection_probability,
        "perturbation_probability": NEUTRAL_PROFILE.perturbation_probability,
    }
    require(
        NEUTRAL_PROFILE.name == "neutral_v1"
        and (NEUTRAL_PROFILE.reflection_probability, NEUTRAL_PROFILE.perturbation_probability)
        == (0.5, 0.5),
        "the accepted neutral_v1 profile moved",
        problems,
    )
    return observed


def verify_observation_and_model_contracts(problems: list) -> dict:
    """The 127-channel observation and model contract, from live code."""
    from stratego.engine.constants import PIECE_TYPE_NAMES
    from stratego.engine.observation import (
        observation_channel_metadata,
        observation_metadata_document,
    )
    from stratego.model.contract import (
        BELIEF_IGNORE_INDEX,
        BELIEF_TYPE_COUNT,
        MODEL_CONTRACT_VERSION,
    )
    from stratego.training import phase11_contract as pc
    from stratego.training.belief_targets import BELIEF_TARGET_VERSION, PIECE_TYPE_INDEX

    channels = observation_channel_metadata()
    document = observation_metadata_document()
    observation_digest = pc.document_digest(document)
    require(len(channels) == 127, f"observation has {len(channels)} channels", problems)
    require(
        OBSERVATION_VERSION == "observation_v2_1_127ch",
        f"observation version {OBSERVATION_VERSION!r}",
        problems,
    )
    require(
        MODEL_CONTRACT_VERSION == "model_contract_v2",
        f"model contract {MODEL_CONTRACT_VERSION!r}",
        problems,
    )
    require(BELIEF_TYPE_COUNT == 12, f"belief type count {BELIEF_TYPE_COUNT}", problems)
    require(
        pc.RANK_NAMES == PIECE_TYPE_NAMES,
        "the frozen rank order disagrees with the engine enumeration",
        problems,
    )
    require(
        all(PIECE_TYPE_INDEX[name] == index for index, name in enumerate(pc.RANK_NAMES)),
        "the belief-target index map disagrees with the frozen rank order",
        problems,
    )
    return {
        "observation_version": OBSERVATION_VERSION,
        "observation_channels": len(channels),
        "observation_metadata_digest": observation_digest,
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "belief_type_count": BELIEF_TYPE_COUNT,
        "belief_ignore_index": BELIEF_IGNORE_INDEX,
        "belief_target_version": BELIEF_TARGET_VERSION,
        "rank_order": list(pc.RANK_NAMES),
    }


def verify_no_preexisting_phase11_evidence(problems: list) -> dict:
    """No Phase 11 prediction, score, sampler output or outcome may exist."""
    existing = sorted(
        str(path.relative_to(REPOSITORY_ROOT))
        for path in DATA_DIRECTORY.glob("*")
        if path.name not in {
            "agent_01_phase11_contract.json",
            "agent_01_validation_bank.json",
            "agent_01_test_bank.json",
            "agent_01_acceptance.json",
            "phase11_bank_access_ledger.jsonl",
        }
    )
    require(
        not existing,
        f"unexpected pre-existing Phase 11 evidence: {existing}",
        problems,
    )
    return {"unexpected_entries": existing}


def stage_verify(_args) -> dict:
    problems: list = []
    log("verifying the Phase 10 formal closure and acceptance chain")
    closure = verify_phase10_closure(problems)
    log("verifying the Phase 9 checkpoint and deriving the belief-head identity")
    checkpoint = verify_phase9_checkpoint(problems)
    log("verifying the Phase 9 contract/amendment chain")
    chain = verify_phase9_chain(problems)
    log("verifying the frozen P10-D selector, utility, scaler and system")
    selector = verify_phase10_selector(problems)
    log("verifying the Phase 8 anchor export")
    anchor = verify_phase8_anchor(problems)
    log("verifying the Phase 7 library and neutral_v1")
    library = verify_phase7_library(problems)
    log("verifying the observation and model contracts")
    contracts = verify_observation_and_model_contracts(problems)
    log("verifying that no Phase 11 evidence pre-exists")
    preexisting = verify_no_preexisting_phase11_evidence(problems)

    payload = {
        "environment": environment_report(),
        "phase10_closure": closure,
        "phase9_checkpoint": checkpoint,
        "phase9_chain": chain,
        "phase10_selector": selector,
        "phase8_anchor": anchor,
        "phase7_library": library,
        "observation_model_contracts": contracts,
        "preexisting_phase11_evidence": preexisting,
        "problems": problems,
    }
    if problems:
        write_stage("verify", payload)
        raise Agent1Error(f"verification failed with {len(problems)} problem(s): {problems}")
    write_stage("verify", payload)
    log("verification clean")
    return payload


# ---------------------------------------------------------------------------
# 2. Contracts and the seed collision audit
# ---------------------------------------------------------------------------


def seed_collision_audit() -> dict:
    """Exhaustive audit over every currently enumerable Phase 11 id space."""
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

    audit = ps.stream_collision_audit(streams)
    audit["million_scale_obligation"] = (
        "the world-sample space is keyed by public-state identities that do "
        "not exist yet; Agents 3, 4, 6 and 7 must run this audit over every "
        "world_sample/world_order/world_categorical seed they actually derive"
    )
    return audit


def stage_contracts(_args) -> dict:
    from stratego.training import phase11_contract as pc

    log("building the eight frozen contract documents")
    documents = pc.contract_documents()
    digests = pc.contract_digests(documents)
    bundle = pc.contract_bundle_digest(documents)
    for name in pc.CONTRACT_VERSIONS:
        log(f"  {name:<34} {digests[name][:16]}")
    log(f"  {'bundle':<34} {bundle[:16]}")

    log("running the seed collision audit over every enumerable id space")
    audit = seed_collision_audit()
    if not audit["no_collisions"]:
        raise Agent1Error(f"seed collision audit failed: {audit['findings'][:8]}")
    log(
        f"  {audit['total_seeds']:,} seeds enumerated, "
        f"{audit['distinct_seeds']:,} distinct, zero collisions"
    )

    payload = {
        "contract_digests": digests,
        "contract_bundle_digest": bundle,
        "seed_collision_audit": {
            "streams": audit["streams"],
            "total_seeds": audit["total_seeds"],
            "distinct_seeds": audit["distinct_seeds"],
            "no_collisions": audit["no_collisions"],
            "findings": audit["findings"],
            "million_scale_obligation": audit["million_scale_obligation"],
        },
        "root_seeds": dict(pc.CANONICAL_PHASE11_SEEDS),
    }
    write_stage("contracts", payload)

    log("writing the contract artifact")
    artifact = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": CONTRACT_ARTIFACT.stem,
        "contract_versions": list(pc.CONTRACT_VERSIONS),
        "documents": documents,
        "contract_digests": digests,
        "contract_bundle_digest": bundle,
        "root_seeds": dict(pc.CANONICAL_PHASE11_SEEDS),
        "seed_collision_audit": payload["seed_collision_audit"],
    }
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CONTRACT_ARTIFACT.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n")
    log(f"  {CONTRACT_ARTIFACT.relative_to(REPOSITORY_ROOT)}")
    return payload


# ---------------------------------------------------------------------------
# 3. Banks
# ---------------------------------------------------------------------------


def stage_banks(_args) -> dict:
    from stratego.evaluation import phase11_banks as pb

    sources = pb.Phase11SetupSources()
    results: dict = {}
    ledger_entries: list = []
    all_cases: dict = {}

    for bank in ("validation", "test"):
        version = pb.BANK_SPECIFICATIONS[bank]["bank_version"]
        log(f"building {version}")
        cases, manifest = pb.build_phase11_bank(bank, sources)
        ledger_entries.append(
            pb.ledger_entry(AGENT, "banks", version, "structural_build", structural_only=True)
        )
        ledger_entries.append(
            pb.ledger_entry(AGENT, "banks", version, "digest_computation", structural_only=True)
        )
        log(f"  {len(cases)} cases, digest {manifest['bank_digest'][:16]}")
        log(f"auditing {version} structurally")
        audit = pb.audit_phase11_bank(bank, cases, manifest, sources, rebuild_sample_every=16)
        ledger_entries.append(
            pb.ledger_entry(AGENT, "banks", version, "structural_audit", structural_only=True)
        )
        if not audit["all_pass"]:
            failed = [name for name, value in audit["checks"].items() if not value]
            raise Agent1Error(f"{version} structural audit failed: {failed}")
        results[bank] = {"manifest": manifest, "audit": audit}
        all_cases[bank] = cases

    log("running the cross-bank disjointness checks")
    cross = pb.cross_bank_checks(all_cases["validation"], all_cases["test"])
    if not cross["zero_overlap"]:
        raise Agent1Error(f"cross-bank overlap detected: {cross}")

    log("appending the bank access ledger")
    pb.append_ledger_entries(ledger_entries)

    payload = {
        "validation": {
            "manifest": results["validation"]["manifest"],
            "audit": results["validation"]["audit"],
        },
        "test": {
            "manifest": results["test"]["manifest"],
            "audit": results["test"]["audit"],
        },
        "cross_bank": cross,
        "ledger_appended": len(ledger_entries),
    }
    write_stage("banks", payload)

    log("writing the two bank artifacts")
    for bank, artifact_path in (
        ("validation", VALIDATION_BANK_ARTIFACT),
        ("test", TEST_BANK_ARTIFACT),
    ):
        version = pb.BANK_SPECIFICATIONS[bank]["bank_version"]
        artifact = {
            "agent": AGENT,
            "phase": PHASE,
            "artifact": artifact_path.stem,
            "manifest": results[bank]["manifest"],
            "audit_checks": results[bank]["audit"]["checks"],
            "audit_all_pass": results[bank]["audit"]["all_pass"],
            "cases": [case.to_dict() for case in all_cases[bank]],
        }
        DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n")
        pb.append_ledger_entries(
            [
                pb.ledger_entry(
                    AGENT, "artifacts", version, "structural_artifact_write",
                    structural_only=True,
                )
            ]
        )
        log(f"  {artifact_path.relative_to(REPOSITORY_ROOT)}")
    return payload


# ---------------------------------------------------------------------------
# 4. Acceptance
# ---------------------------------------------------------------------------


def completion_gates(verify: dict, contracts: dict, banks: dict) -> dict:
    from stratego.evaluation import phase11_banks as pb
    from stratego.training import phase11_contract as pc
    from stratego.training import phase11_seed as ps
    from tests.training import phase11_frozen_digests as pins

    ledger = pb.read_ledger()
    sealed = pb.verify_test_bank_sealed(ledger)

    validation_audit = banks["validation"]["audit"]
    test_audit = banks["test"]["audit"]

    gates = {
        "upstream_phase10_closed": verify["phase10_closure"]["closure_is_ancestor_of_head"]
        and all(
            not record["false_gates"]
            for record in verify["phase10_closure"]["agents"].values()
        )
        and not verify["phase10_closure"]["tracked_tree_dirty_entries"],
        "phase9_identity_verified": verify["phase9_checkpoint"]["sha256"]
        == pc.GATE_H["phase9_checkpoint_sha256"]
        and verify["phase9_checkpoint"]["all_parameters_finite"],
        "belief_head_identity_frozen": verify["phase9_checkpoint"]["belief_head_digest"]
        == pc.ACCEPTED_BELIEF_HEAD_DIGEST
        and tuple(verify["phase9_checkpoint"]["belief_head_tensor_names"])
        == pc.BELIEF_HEAD_TENSOR_NAMES,
        "phase10_selector_identity_verified": verify["phase10_selector"]["config_sha256"]
        == pc.ACCEPTED_SELECTOR_CONFIG_SHA256
        and verify["phase10_selector"]["phase10_system_digest"]
        == pc.ACCEPTED_PHASE10_SYSTEM_DIGEST,
        "phase7_identity_verified": verify["phase7_library"]["splits"]
        == {"train": 6400, "validation": 800, "test": 800},
        "phase8_anchor_identity_verified": verify["phase8_anchor"]["sha256"]
        == pc.ACCEPTED_ANCHOR_EXPORT_SHA256,
        "observation_contract_verified": verify["observation_model_contracts"][
            "observation_channels"
        ]
        == 127,
        "eight_contracts_frozen": contracts["contract_digests"] == pins.CONTRACT_DIGESTS,
        "contract_bundle_frozen": contracts["contract_bundle_digest"]
        == pins.CONTRACT_BUNDLE_DIGEST,
        "eight_root_seeds_frozen": contracts["root_seeds"]
        == dict(ps.CANONICAL_PHASE11_SEEDS)
        and len(ps.CANONICAL_PHASE11_SEEDS) == 8,
        "randomness_domains_frozen": len(ps.STREAM_DOMAINS) == 12,
        "seed_collision_audit_clean": contracts["seed_collision_audit"]["no_collisions"],
        "validation_bank_exact": banks["validation"]["manifest"]["bank_digest"]
        == pins.BANK_DIGESTS["validation"],
        "test_bank_exact": banks["test"]["manifest"]["bank_digest"]
        == pins.BANK_DIGESTS["test"],
        "validation_balance_exact": validation_audit["checks"]["cell_balance_exact"]
        and validation_audit["checks"]["stratum_balance_exact"]
        and validation_audit["checks"]["source_balance_exact"]
        and validation_audit["checks"]["colour_pairing_exact"],
        "test_balance_exact": test_audit["checks"]["cell_balance_exact"]
        and test_audit["checks"]["stratum_balance_exact"]
        and test_audit["checks"]["source_balance_exact"]
        and test_audit["checks"]["colour_pairing_exact"],
        "isolated_case_rebuild_pass": validation_audit["checks"]["isolated_rebuild_exact"]
        and test_audit["checks"]["isolated_rebuild_exact"],
        "bank_overlap_zero": banks["cross_bank"]["zero_overlap"],
        "prediction_target_contract_frozen": "belief_target"
        in pc.belief_contract_document(),
        "baselines_frozen": pc.baseline_document()["baseline_count"] == 2,
        "sampler_math_frozen": len(pc.SAMPLER_ALGORITHM_STEPS) == 12,
        "metrics_frozen": len(pc.OVERALL_METRIC_TOKENS) == 14,
        "bootstrap_frozen": pc.BOOTSTRAP_REPLICATES == 10_000
        and pc.BOOTSTRAP_CONFIDENCE == 0.95,
        "acceptance_gates_frozen": pc.HARD_GATE_IDS == tuple("ABCDEFGH"),
        "classification_frozen": set(pc.CLASSIFICATIONS)
        == {"PASS-SEARCH-READY", "FAIL", "BLOCKED"},
        "ledger_initialized": sealed["ledger_entries"] > 0,
        "test_outcome_access_zero": sealed["test_bank_structural_only"]
        and sealed["outcome_total"] == 0,
        "no_phase11_predictions_scored": sealed["scored_prediction_total"] == 0
        and not verify["preexisting_phase11_evidence"]["unexpected_entries"],
        "no_neural_updates": verify["phase9_checkpoint"]["global_optimizer_step"]
        == pc.ACCEPTED_GLOBAL_OPTIMIZER_STEP,
        "phase9_checkpoint_unchanged": file_sha256(CHECKPOINT_PATH)
        == pc.GATE_H["phase9_checkpoint_sha256"],
        "full_suite_green": False,  # set by --record-suite from the measurement
    }
    return gates


def recorded_deviations() -> list:
    return [
        {
            "reading": "bank_split_binding",
            "detail": (
                "the common contract names no Phase 7 split for the banks; the "
                "frozen reading follows the accepted Phase 9/10 precedent — "
                "validation bank on the validation split, test bank on the test "
                "split, both seats alike"
            ),
        },
        {
            "reading": "colour_pairing_order",
            "detail": (
                "the common contract requires the observer Red once and Blue "
                "once per case without fixing the order; frozen as observer Red "
                "in game 0, Blue in game 1 (the accepted Phase 10 pairing)"
            ),
        },
        {
            "reading": "no_rejection_draws",
            "detail": (
                "bank setup draws are pure first-attempt draws from the frozen "
                "production sources — no fingerprint-isolation or distinctness "
                "rejection — because Phase 11 selects nothing and rejection "
                "would distort the production distributions the belief system "
                "must be measured under"
            ),
        },
        {
            "reading": "sampler_completion_feasibility_rule",
            "detail": (
                "step 6's legal-rank set carries a frozen completion-"
                "feasibility guard (an unmoved piece may take a movable rank "
                "only when the movable surplus covers the remaining moved "
                "pieces); the common contract requires completion and tolerates "
                "zero invalid worlds, but the unguarded set can dead-end on a "
                "feasible instance; the guard is exact — every valid world "
                "stays reachable — and the step-7 weighting is unchanged; "
                "recorded for reviewer acceptance at this handoff"
            ),
        },
        {
            "reading": "soak_namespace_frozen_now",
            "detail": (
                "the common contract froze no dedicated Agent 6 soak root; the "
                "soak's setup/match streams hang off the bank-schedule and "
                "match-randomness roots under distinct domain tokens, with id "
                "formats and volumes frozen now — closing the Phase 10 "
                "soak-namespace deviation in advance"
            ),
        },
        {
            "reading": "shared_bank_schedule_root",
            "detail": (
                "both banks' streams hang off the single bank/case-schedule "
                "root 2026081902, domain-separated by the bank-version token "
                "inside every case id — the accepted Phase 10 case-schedule "
                "reading, reused"
            ),
        },
        {
            "reading": "progress_bucket_thresholds",
            "detail": (
                "early/middle/late are frozen at pre-action plies 0-39 / "
                "40-119 / 120+, fixed from accepted Phase 10 soak evidence "
                "(mean 116.8 plies per canonical self-play game) before any "
                "Phase 11 result existed; diagnostic slices only"
            ),
        },
        {
            "reading": "unit_uniform_tail_edge",
            "detail": (
                "the accepted seed-to-uniform convention rounds the extreme "
                "top of the 63-bit range to exactly 1.0 under float64; every "
                "frozen inverse-CDF walk therefore carries a last-element tail "
                "guard, and the convention is kept bit-identical to Phase 10's"
            ),
        },
        {
            "reading": "hash_order_schedule_selection",
            "detail": (
                "the reproducibility request set and benchmark states are "
                "selected by deterministic hash-order rules that consume no "
                "randomness; the repro_schedule and benchmark domains stay "
                "frozen and available for any predeclared draw a downstream "
                "agent needs"
            ),
        },
        {
            "reading": "untracked_phase10b_drafts",
            "detail": (
                "two untracked draft modules for the optional Phase 10B "
                "experiment (stratego/training/phase10b_contract.py, "
                "phase10b_seed.py) predate this session and are not Phase 11 "
                "material; left untracked and untouched"
            ),
        },
        {
            "reading": "phase10b_draft_drift_restored",
            "detail": (
                "the same pre-session Phase 10B drafting also left an "
                "uncommitted modification to the frozen Phase 9 module "
                "stratego/training/phase9_rollout_store.py (an optional "
                "id_parser hook, Phase 9 behaviour unchanged, mtime "
                "2026-08-19 00:23). The upstream_phase10_closed gate refused "
                "to freeze over the dirty tracked tree; the draft was "
                "preserved losslessly to the untracked "
                "phase10b_rollout_store_draft.patch (verified to re-apply "
                "cleanly) and the module restored to the closure-commit "
                "bytes before the freeze completed. No Phase 11 computation "
                "reads the modified function, and the accepted digests are "
                "unaffected"
            ),
        },
    ]


def stage_acceptance(_args) -> dict:
    from stratego.evaluation import phase11_banks as pb
    from stratego.training import phase11_contract as pc

    verify = read_stage("verify")
    contracts = read_stage("contracts")
    banks = read_stage("banks")

    gates = completion_gates(verify, contracts, banks)
    non_suite = {name: value for name, value in gates.items() if name != "full_suite_green"}
    false_gates = sorted(name for name, value in non_suite.items() if not value)
    if false_gates:
        raise Agent1Error(f"completion gates failed: {false_gates}")

    sealed = pb.verify_test_bank_sealed()

    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_01_acceptance",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "PENDING-SUITE",
        "completion_gates": gates,
        "gates_total": len(gates),
        "gates_true": sum(bool(value) for value in gates.values()),
        "false_gates": sorted(name for name, value in gates.items() if not value),
        "problems": [],
        "environment": verify["environment"],
        "frozen_inputs": {
            "phase10_closure_commit": verify["phase10_closure"]["closure_commit"],
            "phase9_checkpoint_sha256": verify["phase9_checkpoint"]["sha256"],
            "phase9_model_state_digest": verify["phase9_checkpoint"]["model_state_digest"],
            "phase9_parameters": verify["phase9_checkpoint"]["parameters"],
            "phase9_global_optimizer_step": verify["phase9_checkpoint"][
                "global_optimizer_step"
            ],
            "c1_config_digest": verify["phase9_checkpoint"]["c1_config_digest"],
            "belief_head_tensor_names": verify["phase9_checkpoint"][
                "belief_head_tensor_names"
            ],
            "belief_head_digest": verify["phase9_checkpoint"]["belief_head_digest"],
            "phase9_contract_digest": verify["phase9_chain"]["observed"]["contract_digest"],
            "phase9_amendment_v1_digest": verify["phase9_chain"]["observed"][
                "amendment_v1_digest"
            ],
            "phase9_amendment_v2_digest": verify["phase9_chain"]["observed"][
                "amendment_v2_digest"
            ],
            "selector_config_sha256": verify["phase10_selector"]["config_sha256"],
            "utility_file_sha256": verify["phase10_selector"]["utility_file_sha256"],
            "model_T_coefficient_digest": verify["phase10_selector"][
                "model_T_coefficient_digest"
            ],
            "trait_scaler_digest": verify["phase10_selector"]["trait_scaler_digest"],
            "phase10_system_digest": verify["phase10_selector"]["phase10_system_digest"],
            "phase8_anchor_sha256": verify["phase8_anchor"]["sha256"],
            "phase7_library_content_digest": verify["phase7_library"]["content_digest"],
            "phase7_library_metadata_digest": verify["phase7_library"]["metadata_digest"],
            "observation_metadata_digest": verify["observation_model_contracts"][
                "observation_metadata_digest"
            ],
        },
        "new_digests": {
            "contract_digests": contracts["contract_digests"],
            "contract_bundle_digest": contracts["contract_bundle_digest"],
            "validation_bank_digest": banks["validation"]["manifest"]["bank_digest"],
            "validation_manifest_digest": banks["validation"]["manifest"][
                "manifest_digest"
            ],
            "test_bank_digest": banks["test"]["manifest"]["bank_digest"],
            "test_manifest_digest": banks["test"]["manifest"]["manifest_digest"],
        },
        "seeds": contracts["root_seeds"],
        "seed_collision_audit": contracts["seed_collision_audit"],
        "bank_summaries": {
            bank: {
                "bank_version": banks[bank]["manifest"]["bank_version"],
                "case_count": banks[bank]["manifest"]["case_count"],
                "game_count": banks[bank]["manifest"]["game_count"],
                "p10d_branch_histogram": banks[bank]["manifest"][
                    "p10d_branch_histogram"
                ],
                "audit_all_pass": banks[bank]["audit"]["all_pass"],
            }
            for bank in ("validation", "test")
        },
        "cross_bank": banks["cross_bank"],
        "ledger": sealed,
        "recorded_deviations": recorded_deviations(),
        "forbidden_operation_counters": {
            "phase11_optimizer_steps": 0,
            "belief_calibration_operations": 0,
            "phase11_games_played": 0,
            "phase11_predictions_scored": 0,
            "phase11_worlds_sampled": 0,
            "test_bank_scored_accesses": 0,
            "privileged_truth_reads": 0,
            "outcome_reads": 0,
        },
        "handoff_to_agent_2": {
            "for_agent": 2,
            "mission": (
                "implement the belief evaluation path and the two frozen "
                "baselines, then run predictive evaluation on "
                "phase11_validation_bank_v1 only"
            ),
            "contract_bundle_digest": contracts["contract_bundle_digest"],
            "contract_digests": contracts["contract_digests"],
            "bank_identities": {
                "validation": {
                    "bank_version": pc.VALIDATION_BANK_VERSION,
                    "bank_digest": banks["validation"]["manifest"]["bank_digest"],
                    "manifest_digest": banks["validation"]["manifest"]["manifest_digest"],
                },
                "test": {
                    "bank_version": pc.TEST_BANK_VERSION,
                    "bank_digest": banks["test"]["manifest"]["bank_digest"],
                    "manifest_digest": banks["test"]["manifest"]["manifest_digest"],
                },
            },
            "belief_head": {
                "tensor_names": list(pc.BELIEF_HEAD_TENSOR_NAMES),
                "digest": pc.ACCEPTED_BELIEF_HEAD_DIGEST,
            },
            "rank_indexing": list(pc.RANK_NAMES),
            "prediction_record_fields": list(pc.PREDICTION_RECORD_FIELDS),
            "prediction_record_version": pc.PREDICTION_RECORD_VERSION,
            "public_state_document_version": pc.PUBLIC_STATE_DOCUMENT_VERSION,
            "request_schema": {
                "request_version": pc.BELIEF_REQUEST_VERSION,
                "allowed_fields": list(pc.ALLOWED_BELIEF_REQUEST_FIELDS),
                "forbidden_tokens": list(pc.FORBIDDEN_BELIEF_REQUEST_TOKENS),
            },
            "baseline_versions": [
                pc.REMAINING_COUNT_BASELINE_VERSION,
                pc.WORLD_BASELINE_VERSION,
            ],
            "evaluator_version": pc.EVALUATOR_VERSION,
            "metric_formulas": dict(pc.METRIC_FORMULAS),
            "statistics": dict(pc.STATISTICS),
            "authorization": (
                "validation-bank predictive evaluation only; the test bank "
                "stays structural with zero scored access, proven through the "
                "ledger"
            ),
            "prohibition": (
                "Agent 2 must not implement or tune belief_sampler_v1 beyond "
                "the simple frozen count_uniform_world_sampler_v1 baseline"
            ),
        },
        "tests_before": TESTS_BEFORE,
        "tests_after": None,
        "suite": None,
        "suite_confirmation": None,
    }
    ACCEPTANCE_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ACCEPTANCE_ARTIFACT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    write_stage("acceptance", {"gates": gates, "false_gates": false_gates})
    log(
        f"acceptance written: {payload['gates_true']}/{payload['gates_total']} gates "
        "true (full_suite_green pending --record-suite)"
    )
    return payload


# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------


def build_report_section() -> str:
    from stratego.training import phase11_contract as pc

    acceptance = read_json(ACCEPTANCE_ARTIFACT)
    contracts = read_stage("contracts")
    banks = read_stage("banks")
    frozen = acceptance["frozen_inputs"]
    fresh = acceptance["new_digests"]
    audit = acceptance["seed_collision_audit"]

    suite = acceptance.get("suite")
    if suite:
        suite_line = suite["summary"]
    else:
        suite_line = "pending --record-suite"

    lines = [
        SECTION_MARKER,
        "",
        f"**Status: {acceptance['status']}** — "
        f"{acceptance['gates_true']}/{acceptance['gates_total']} completion gates true, "
        "zero problems, zero Phase 11 predictions, zero sampled worlds, zero games,",
        "zero optimizer steps.",
        "",
        "Agent 1 freezes the entire Phase 11 experiment before any prediction",
        "score, sampler output or test outcome exists. Nothing below was chosen",
        "after seeing a Phase 11 result, because no Phase 11 result exists yet.",
        "",
        "### 1.1 Verified upstream identities",
        "",
        "Every identity was recomputed from live bytes, not read from a record.",
        "",
        "```text",
        f"Phase 10 closure commit         {frozen['phase10_closure_commit']} "
        "(Agents 1-7 all PASS, Agent 7 PASS-NONINFERIOR)",
        f"Phase 9 checkpoint SHA-256      {frozen['phase9_checkpoint_sha256']}",
        f"Phase 9 model-state digest      {frozen['phase9_model_state_digest']}",
        f"Phase 9 parameters              {frozen['phase9_parameters']:,}, all finite; "
        f"global optimizer step {frozen['phase9_global_optimizer_step']:,}",
        f"belief head (live tensors)      {', '.join(frozen['belief_head_tensor_names'])}",
        f"belief-head digest              {frozen['belief_head_digest']}",
        f"C1 config digest                {frozen['c1_config_digest']}",
        f"P10-D config SHA-256            {frozen['selector_config_sha256']}",
        f"utility model_T digest          {frozen['model_T_coefficient_digest']}",
        f"trait scaler digest             {frozen['trait_scaler_digest']}",
        f"phase10_system_v1 digest        {frozen['phase10_system_digest']}",
        f"Phase 8 anchor export           {frozen['phase8_anchor_sha256']}",
        f"Phase 7 library content         {frozen['phase7_library_content_digest']}",
        "Phase 7 splits                  6,400 / 800 / 800; neutral_v1 0.5/0.5",
        f"observation metadata digest     {frozen['observation_metadata_digest']} "
        "(127 channels)",
        "pre-existing Phase 11 work      none: no predictions, no worlds, no outcomes",
        "```",
        "",
        "The belief-head tensor identity is derived from the live checkpoint —",
        "`belief_output.bias` and `belief_output.weight` hashed under the accepted",
        "state-digest recipe — and frozen for every later agent to re-derive.",
        "",
        "### 1.2 Frozen contracts",
        "",
        "Eight documents, canonical JSON, SHA-256.",
        "",
        "```text",
    ]
    for name in pc.CONTRACT_VERSIONS:
        lines.append(f"{name:<34}{contracts['contract_digests'][name]}")
    lines.extend(
        [
            f"{'bundle':<34}{contracts['contract_bundle_digest']}",
            "```",
            "",
            "`phase11_system_v1` binds what exists now — the accepted belief model",
            "and its head identity, P10-D with utility/scaler, the Phase 7 library,",
            "the baselines and the bank versions — and leaves five slots unbound with",
            "their filling rules (evaluator, sampler implementation, safety evidence,",
            "runtime benchmark result, bank digests). Agent 6 fills them at the",
            "production freeze.",
            "",
            "### 1.3 Seeds and derivations",
            "",
            "```text",
            "master                    2026081901     bank/case schedule        2026081902",
            "game/match randomness     2026081903     belief/world sampling     2026081904",
            "information safety        2026081905     repro/runtime audit       2026081906",
            "validation bootstrap      2026081907     final-test bootstrap      2026081908",
            "```",
            "",
            "Beneath the eight roots sit **twelve derived domains** under the new",
            "`strat-b11` personalization — disjoint from every accepted upstream tag:",
            "",
            "```text",
            "bank_observer_setup  bank_opponent_setup  bank_match       world_sample",
            "world_order          world_categorical    safety_trial     repro_schedule",
            "benchmark            bootstrap            soak_setup       soak_match",
            "```",
            "",
            "The Agent 6 soak namespace is frozen *now* — id formats, volumes",
            "(1,024 games x 8 requests = 8,192), colour parity, request attachment",
            "rule — closing the Phase 10 soak-namespace deviation in advance.",
            "",
            f"The collision audit enumerated {audit['total_seeds']:,} seeds across "
            f"every currently enumerable id space and found "
            f"{audit['distinct_seeds']:,}",
            "distinct values — zero duplicates inside a stream and zero collisions",
            "across streams. The million-scale world-sample space is keyed by",
            "public-state identities that do not exist yet; Agents 3, 4, 6 and 7",
            "carry the frozen obligation to re-run this audit over every world",
            "stream they actually derive.",
            "",
            "### 1.4 Frozen banks",
            "",
            "```text",
            "phase11_validation_bank_v1   512 cases  1,024 games  validation split",
            "phase11_test_bank_v1       2,048 cases  4,096 games  test split",
            "8 opponent strata x 2 setup sources x 32/128 cases per cell",
            "observer: accepted Phase 9 policy+belief head, P10-D setups, both banks",
            "```",
            "",
            "```text",
            f"validation bank digest    {fresh['validation_bank_digest']}",
            f"validation manifest       {fresh['validation_manifest_digest']}",
            f"test bank digest          {fresh['test_bank_digest']}",
            f"test manifest             {fresh['test_manifest_digest']}",
            "```",
            "",
            "A case fixes both seats' setups (each drawn from its frozen source",
            "conditioned on its own colour — never mirrored) and one match seed per",
            "game. There is **no rejection of any kind**: Phase 11 selects nothing,",
            "so every arrangement is exactly what production would produce. The",
            "structural audits rebuild provenance, re-derive every draw",
            "independently, validate every arrangement through the engine, and",
            "rebuild sampled cases in isolation — all exact, for both banks; the",
            "cross-bank check proves zero id, seed and fingerprint overlap.",
            "",
            "P10-D branch mixture over the materialized draws:",
            "",
            "```text",
        ]
    )
    for bank in ("validation", "test"):
        histogram = banks[bank]["manifest"]["p10d_branch_histogram"]
        total = sum(histogram.values())
        lines.append(
            f"{bank:<12}learned {histogram.get('learned', 0):>5} / neutral "
            f"{histogram.get('neutral', 0):>5}  ({histogram.get('learned', 0) / total:.3f} "
            f"/ {histogram.get('neutral', 0) / total:.3f} of {total})"
        )
    lines.extend(
        [
            "```",
            "",
            "### 1.5 Frozen target, metrics, sampler, safety and gates",
            "",
            "- **Targets**: every live opponent piece not legally known to the",
            "  observer, at every observer-acting decision — the engine's accepted",
            "  `belief_target` semantics; publicly known ranks are never events.",
            "  Rank order is the engine enumeration (spy..marshal, flag, bomb).",
            "- **Learned vector**: raw float64 softmax of the head's 12 logits at",
            "  the piece's square — no masking, no epsilon; CE floors only inside",
            "  the log at 1e-12 (report-only counter).",
            "- **Baseline**: `remaining_count_belief_v1`, mask-restricted",
            "  count-proportional; provably positive on the true rank.",
            "- **Sampler**: the common-contract twelve steps with",
            "  `weight = learned_probability x remaining_count`, plus a frozen",
            "  completion-feasibility guard on step 6's legal set (recorded reading:",
            "  the unguarded walk can dead-end on feasible instances; the guard is",
            "  exact and keeps every valid world reachable).",
            "- **Statistics**: case-level percentile bootstrap, 10,000 replicates,",
            "  95%, both colour games kept together, domain-separated streams per",
            "  metric; ECE 15 equal-width bins, pooled events.",
            "- **Gates A-H** exactly as the common contract, with explicit",
            "  strict/non-strict operators and boundary tests in the suite;",
            "  classification PASS-SEARCH-READY / FAIL / BLOCKED recomputes from",
            "  gate booleans with no discretionary override.",
            "- **Runtime**: backend frozen before measurement — CPU float32, one",
            "  torch thread, 480 benchmark states over 48 cells; hard ceiling",
            "  p95(forward + 64 worlds) <= 500 ms.",
            "",
            "### 1.6 Access ledger and readings",
            "",
            "The append-only ledger at `reports/phase_11_data/",
            "phase11_bank_access_ledger.jsonl` records every Agent 1 bank access:",
            "structural build, digest, audit and artifact write for each bank — all",
            "structural-only with zero neural/scored/privileged/outcome counters.",
            "",
            f"{len(acceptance['recorded_deviations'])} recorded readings (bank split "
            "binding, colour-pairing order, no-rejection",
            "draws, the sampler completion-feasibility rule, the soak namespace,",
            "the shared bank-schedule root, progress-bucket thresholds, the",
            "unit-uniform tail edge, hash-order schedule selection, and the",
            "untracked Phase 10B drafts) are itemized in the acceptance artifact",
            "for reviewer acceptance at this handoff.",
            "",
            "### 1.7 Artifacts and completion gates",
            "",
            "```text",
            "reports/phase_11_data/agent_01_phase11_contract.json",
            "reports/phase_11_data/agent_01_validation_bank.json",
            "reports/phase_11_data/agent_01_test_bank.json",
            "reports/phase_11_data/agent_01_acceptance.json",
            "reports/phase_11_data/phase11_bank_access_ledger.jsonl",
            "```",
            "",
            f"Full suite: `{TESTS_BEFORE['command']}` — {suite_line}",
            "",
            "| gate | value |",
            "| --- | --- |",
        ]
    )
    for name, value in acceptance["completion_gates"].items():
        lines.append(f"| `{name}` | {str(bool(value)).lower()} |")
    lines.extend(
        [
            "",
            "Agent 1 stops here and waits for reviewer acceptance. Agent 2 is",
            "authorized for validation-bank predictive evaluation only; the test",
            "bank stays sealed with zero scored access, proven through the ledger.",
            "",
        ]
    )
    return "\n".join(lines)


REPORT_PREAMBLE = [
    "# Phase 11 Implementation Report",
    "",
    "Phase 11 is a **belief-system validation and search-readiness** phase. It",
    "asks whether the accepted Phase 9 belief head produces accurate,",
    "calibrated, information-safe, reproducible beliefs about hidden opponent",
    "ranks, and whether those marginals convert into complete legal hidden",
    "worlds fast enough for Phase 12 search. Nothing is trained, calibrated or",
    "repaired: `checkpoints/phase9/selfplay_c1_v1.pt` must be byte-identical",
    "before and after the phase, and a failing system ends the phase as FAIL",
    "rather than becoming a repair loop.",
    "",
]


def stage_report(_args) -> dict:
    section = build_report_section()
    if REPORT_PATH.exists():
        text = REPORT_PATH.read_text()
        if SECTION_MARKER in text:
            head, _, tail = text.partition(SECTION_MARKER)
            remainder = tail.split("\n## ", 1)
            trailing = ("\n## " + remainder[1]) if len(remainder) == 2 else "\n"
            new_text = head + section + trailing.rstrip("\n") + "\n"
        else:
            new_text = text.rstrip("\n") + "\n\n" + section
    else:
        new_text = "\n".join(REPORT_PREAMBLE) + "\n" + section
    REPORT_PATH.write_text(new_text)
    log(f"report section written to {REPORT_PATH.relative_to(REPOSITORY_ROOT)}")
    return {"report": str(REPORT_PATH.relative_to(REPOSITORY_ROOT))}


# ---------------------------------------------------------------------------
# Suite recording
# ---------------------------------------------------------------------------


def run_suite() -> dict:
    log("running the full suite (this takes ~5 minutes)")
    started = time.time()
    completed = subprocess.run(
        [str(REPOSITORY_ROOT / ".venv" / "bin" / "python"), "-m", "pytest", "tests", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    seconds = round(time.time() - started, 2)
    tail = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    summary = tail[-1] if tail else "no output"
    # parse "5199 passed, 3 skipped in 313.94s (0:05:13)"
    numbers = {"passed": 0, "failed": 0, "skipped": 0}
    words = summary.replace(",", "").split()
    for index, word in enumerate(words):
        if word in ("passed", "failed", "skipped") and index > 0:
            try:
                numbers[word] = int(words[index - 1])
            except ValueError:  # pragma: no cover - defensive
                pass
    return {
        "command": ".venv/bin/python -m pytest tests -q",
        "summary": summary,
        "returncode": completed.returncode,
        "seconds": seconds,
        **numbers,
    }


def record_suite(_args) -> dict:
    acceptance = read_json(ACCEPTANCE_ARTIFACT)
    measurement = run_suite()
    log(f"suite: {measurement['summary']}")
    green = (
        measurement["returncode"] == 0
        and measurement["failed"] == 0
        and measurement["passed"] > 0
    )
    if acceptance.get("suite") is None:
        acceptance["suite"] = measurement
        acceptance["tests_after"] = measurement
    else:
        acceptance["suite_confirmation"] = measurement
        green = green and bool(acceptance["completion_gates"]["full_suite_green"])
    acceptance["completion_gates"]["full_suite_green"] = bool(
        acceptance["suite"]["returncode"] == 0
        and acceptance["suite"]["failed"] == 0
        and acceptance["suite"]["passed"] > 0
    )
    acceptance["gates_true"] = sum(
        bool(value) for value in acceptance["completion_gates"].values()
    )
    acceptance["false_gates"] = sorted(
        name for name, value in acceptance["completion_gates"].items() if not value
    )
    acceptance["status"] = (
        "PASS" if not acceptance["false_gates"] else "PENDING-SUITE"
    )
    ACCEPTANCE_ARTIFACT.write_text(json.dumps(acceptance, indent=1, sort_keys=True) + "\n")
    stage_report(_args)
    log(
        f"recorded: status {acceptance['status']}, "
        f"{acceptance['gates_true']}/{len(acceptance['completion_gates'])} gates true"
    )
    return measurement


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STAGES = {
    "verify": stage_verify,
    "contracts": stage_contracts,
    "banks": stage_banks,
    "acceptance": stage_acceptance,
    "report": stage_report,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), default=None)
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
