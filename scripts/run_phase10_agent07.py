#!/usr/bin/env python3
"""Phase 10 Agent 7 harness: independent final acceptance and Phase 10 freeze.

Independently recomputes every Agent 1-6 identity and discipline claim from
live bytes, rebuilds and audits the sealed test bank, proves the bank has
had zero prior outcome evaluation, and only then performs the **first**
final-test game-outcome evaluation on `phase10_test_bank_v1`: the single
permanently selected P10-D configuration against the fixed `neutral_v1`
baseline, on identical logical cases, under the frozen Gates A-H.

What this script is and is not
------------------------------
It verifies and it evaluates. It trains nothing, refits nothing, replaces
no candidate, changes no temperature, mixture or threshold, takes zero
optimizer steps on the Phase 9 checkpoint, and never uses a final-test
outcome to repair the system. Report-only diagnostics (Phase 9 fingerprint
landings, stress) never rescue a gate. Formal closure belongs to the
reviewing chat.

Usage::

    python scripts/run_phase10_agent07.py                 # every stage
    python scripts/run_phase10_agent07.py --stage verify  # one stage
    python scripts/run_phase10_agent07.py --record-suite  # record the suite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

AGENT = 7
PHASE = 10
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_10_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_10_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase10" / "agent07"
STAGE_DIRECTORY = WORK_DIRECTORY / "stages"
GAMES_DIRECTORY = WORK_DIRECTORY / "games"
EXPORT_PATH = WORK_DIRECTORY / "eval_phase9_c1.pt"

ACCEPTANCE_ARTIFACT = DATA_DIRECTORY / "agent_07_final_acceptance.json"
STRENGTH_ARTIFACT = DATA_DIRECTORY / "agent_07_strength_results.csv"
DIVERSITY_ARTIFACT = DATA_DIRECTORY / "agent_07_diversity_results.csv"

SECTION_MARKER = "## 7. Agent 7 — Independent Final Acceptance and Phase 10 Freeze"

CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "selfplay_c1_v1.pt"
UTILITY_PATH = REPOSITORY_ROOT / "checkpoints" / "phase10" / "setup_utility_v1.json"
PHASE8_CHECKPOINT = REPOSITORY_ROOT / "checkpoints" / "phase8" / "warmstart_c1_v1.pt"
ANCHOR_EXPORT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "agent01" / "anchor_eval.pt"
CONFIG_ARTIFACT = DATA_DIRECTORY / "agent_05_frozen_selector_config.json"
PRODUCTION_MANIFEST = DATA_DIRECTORY / "agent_06_production_selector_manifest.json"

#: The administrative freeze this run begins from. The relevant tracked tree
#: must be byte-identical to this commit between prerequisite verification
#: and the opening of the sealed bank.
FREEZE_COMMIT = "97751ef0bfd60b46fb1c17a688e4fd8bf1711ad0"

#: The upstream identities Agent 7 refuses to proceed without. Every one is
#: recomputed from live bytes; a mismatch is BLOCKED, never an accommodation.
ACCEPTED_PHASE9_SHA256 = "dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea"
ACCEPTED_PHASE9_STATE_DIGEST = "f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd"
ACCEPTED_PHASE9_PARAMETERS = 863_959
ACCEPTED_UTILITY_FILE_SHA256 = "50cb947dae633417858dc3352ee1e68e41c1c54845c5d3a261f735571983c25d"
ACCEPTED_COEFFICIENT_DIGESTS = {
    "model_F": "7bc2539af6045e478cd3dbbf78e16c6123616d285a3f32dd1b1a5c1da96ad935",
    "model_T": "d898782a2ae7cf4ed1cb2833fad6e53d8407ec2048dafbd34a6a20c1c9766edc",
}
ACCEPTED_SCALER_DIGEST = "fa6eb1c112defc4c1034831b84db8848181e1f674f8439c9c265916d89e8b7f9"
ACCEPTED_CORPUS_CONTENT_DIGEST = (
    "1977bb6f5e2611b0498c7976f6129718fdfe7f6f44216f3b3f1932c8192b3c50"
)
ACCEPTED_CONTRACT_BUNDLE_DIGEST = (
    "257f140dadddc00e4f75217ecedfe726390167de8769db0b5c40021e4388612f"
)
ACCEPTED_VALIDATION_BANK_DIGEST = (
    "a37ff113d03a0f67e760e447a462cc0d0d8de83f063d395715aeb77be355657f"
)
ACCEPTED_VALIDATION_MANIFEST_DIGEST = (
    "459cef36d7032beb8fc9665efa7692dac3c40c68109e9f0bcdefa6141bd0906e"
)
ACCEPTED_TEST_BANK_DIGEST = (
    "be04b891ba5ab142aacbd937ab24f79054843310e8d28f6b8cbee65daef931ad"
)
ACCEPTED_TEST_MANIFEST_DIGEST = (
    "c6f21bcdb829fe77b208e49d9960b05a1b65bcf1dc7944d3f10420bea132a755"
)
ACCEPTED_SELECTOR_CONTRACT_DIGEST = (
    "ed1198f3a4bfc8f73264cf22602f6d8ba89d9458e9ae5c8a8ddf7f0543e35e59"
)
ACCEPTED_CONFIG_SHA256 = (
    "6e227815bc3cb44f19cdeee55d00ec0ae75726fb411ee9131660aa712bb86668"
)
ACCEPTED_SYSTEM_DIGEST = (
    "615cc3c3a4fab6e4400e20a5a93b13a08c43ab6c3ca63828c6a64742e98175d2"
)
ACCEPTED_TRAIN_DIGESTS = {
    "red": "9ac5b52edbbf0ff92fbebe5c61eefe5a13f0092a3b685eae8857f66b261e491f",
    "blue": "abef229983e2f4b6caf5323171618b5c82d6a67f59463256098343639f6e957f",
}
ACCEPTED_SOAK_CONTENT_DIGEST = (
    "f2922d6b5bf339f642aaf33864b510b3a1099683bf3270c7e5e10b5796ef670e"
)

SELECTED_WINNER = {
    "candidate_id": "P10-D",
    "utility_model": "model_T",
    "temperature": 0.75,
    "selector_identity": "learned_setup_source_v1|k=P10-D|m=model_T|T=0.75",
}

#: The full suite as measured immediately before any Agent 7 change, at the
#: administrative freeze commit.
TESTS_BEFORE = {
    "command": ".venv/bin/python -m pytest tests -q",
    "summary": "5168 passed, 3 skipped in 320.05s (0:05:20)",
    "passed": 5168,
    "failed": 0,
    "skipped": 3,
    "seconds": 320.05,
    "measured_at_commit": "97751ef",
}

#: Every access this script makes to either sealed bank, with its purpose.
BANK_ACCESS_LOG: list = []


class Agent7Error(RuntimeError):
    """Raised when a prerequisite or an invariant of Agent 7's mission fails."""


def log(message: str) -> None:
    print(f"[agent07] {message}", flush=True)


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 -- environment record, never a gate
        return "unknown"


def environment_record() -> dict:
    import torch

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "cpu_count": os.cpu_count(),
        "source_revision": git_output("rev-parse", "--short", "HEAD"),
        "working_tree_state": "clean" if not git_output("status", "--porcelain") else "dirty",
    }


def stage_file_path(name: str) -> Path:
    return STAGE_DIRECTORY / f"stage_{name}.json"


def write_stage(name: str, payload: dict) -> None:
    write_json(stage_file_path(name), payload)


def read_stage(name: str) -> dict:
    path = stage_file_path(name)
    if not path.exists():
        raise Agent7Error(
            f"stage {name!r} has not run; run it before the stage that depends on it"
        )
    return read_json(path)


def record_bank_access(stage: str, bank: str, purpose: str, *, neural: bool, outcomes: bool) -> None:
    BANK_ACCESS_LOG.append(
        {
            "stage": stage,
            "bank": bank,
            "purpose": purpose,
            "neural": neural,
            "outcomes": outcomes,
        }
    )


def checkpoint_identity(path: Path) -> dict:
    """File SHA, model-state digest and parameter count of a Phase 9 payload.

    Recomputed through the accepted Phase 9 helpers, so a moved checkpoint is
    caught by the same functions that accepted it.
    """
    import torch

    from stratego.training.phase9_behavior import state_dict_digest
    from stratego.training.phase9_checkpoint import model_from_payload, read_phase9_payload

    payload = read_phase9_payload(path)
    model = model_from_payload(payload)
    parameters = sum(int(tensor.numel()) for tensor in model.parameters())
    digest = state_dict_digest(model)
    finite = all(bool(torch.isfinite(tensor).all()) for tensor in model.state_dict().values())
    del model, payload
    return {
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "sha256": file_sha256(path),
        "model_state_digest": digest,
        "parameters": parameters,
        "all_parameters_finite": finite,
    }


# ---------------------------------------------------------------------------
# stage: verify — every prerequisite and every discipline claim, from live bytes
# ---------------------------------------------------------------------------

#: The completion-gate names each prior agent's acceptance artifact carries,
#: and the artifact that carries them.
PRIOR_AGENT_ARTIFACTS = {
    1: "agent_01_acceptance.json",
    2: "agent_02_acceptance.json",
    3: "agent_03_acceptance.json",
    4: "agent_04_acceptance.json",
    5: "agent_05_acceptance.json",
    6: "agent_06_acceptance.json",
}


def stage_verify(args) -> dict:
    from stratego.evaluation.phase10_banks import phase9_isolation_set
    from stratego.setups.families import FAMILY_IDS
    from stratego.setups.library import entry_metadata_digest, library_content_digest
    from stratego.setups.sampler import load_library_index
    from stratego.training.phase10_contract import (
        CANDIDATE_IDS,
        CANDIDATE_MATRIX,
        LEARNED_MIXTURE_WEIGHT,
        NEUTRAL_MIXTURE_WEIGHT,
        PHASE7_LIBRARY_CONTENT_DIGEST,
        PHASE7_LIBRARY_MANIFEST_DIGEST,
        PHASE7_LIBRARY_METADATA_DIGEST,
        contract_bundle_digest,
        contract_digests,
        document_digest,
        system_document,
    )
    from stratego.training.phase10_seed import CANONICAL_PHASE10_SEEDS
    from stratego.training.phase10_selector import NEUTRAL_PROFILE, load_scorer
    from stratego.training.phase10_soak import SELECTED_CANDIDATE_ID, SELECTED_CONFIG_SHA256
    from tests.training.phase10_frozen_digests import CONTRACT_DIGESTS

    problems: list = []

    log("verifying the administrative freeze")
    head = git_output("rev-parse", "HEAD")
    porcelain = git_output("status", "--porcelain")
    tracked_dirty = [
        line for line in porcelain.splitlines() if line and not line.startswith("??")
    ]
    untracked = [
        line[3:] for line in porcelain.splitlines() if line.startswith("??")
    ]
    freeze = {
        "commit": head,
        "expected_commit": FREEZE_COMMIT,
        "commit_matches": head == FREEZE_COMMIT,
        "tracked_modifications": tracked_dirty,
        "untracked_files": untracked,
        "untracked_note": (
            "untracked files are Agent 7's own new code, work directories and "
            "artifacts; no tracked byte differs from the freeze commit"
        ),
    }
    if not freeze["commit_matches"]:
        problems.append(f"HEAD {head} is not the administrative freeze commit {FREEZE_COMMIT}")
    if tracked_dirty:
        problems.append(f"tracked files modified since the freeze: {tracked_dirty}")

    log("verifying Agents 1-6 acceptance records")
    prior = {}
    for agent, artifact in PRIOR_AGENT_ARTIFACTS.items():
        record = read_json(DATA_DIRECTORY / artifact)
        gates = record.get("completion_gates", {})
        false_gates = sorted(name for name, value in gates.items() if value is not True)
        prior[f"agent_{agent}"] = {
            "artifact": artifact,
            "status": record.get("status"),
            "gates_true": record.get("gates_true"),
            "gates_total": record.get("gates_total"),
            "false_gates_recomputed": false_gates,
        }
        if record.get("status") != "PASS":
            problems.append(f"Agent {agent} status is {record.get('status')!r}, expected PASS")
        if false_gates:
            problems.append(f"Agent {agent} completion gates recompute false: {false_gates}")
        if record.get("gates_true") != record.get("gates_total") or record.get(
            "gates_true"
        ) != len(gates):
            problems.append(f"Agent {agent} gate counts disagree with its own gate map")

    log("recomputing the eight contract digests and the bundle")
    digests = contract_digests()
    bundle = contract_bundle_digest()
    for name, expected in CONTRACT_DIGESTS.items():
        if digests.get(name) != expected:
            problems.append(f"contract {name} digest {digests.get(name)} != pinned freeze")
    if bundle != ACCEPTED_CONTRACT_BUNDLE_DIGEST:
        problems.append(f"contract bundle digest {bundle} != accepted")

    log("verifying the eight root seeds")
    expected_seeds = {
        "phase10_master_seed": 2026081801,
        "outcome_schedule_seed": 2026081802,
        "setup_draw_seed": 2026081803,
        "utility_fit_seed": 2026081804,
        "selector_draw_seed": 2026081805,
        "case_schedule_seed": 2026081806,
        "validation_bootstrap_seed": 2026081807,
        "test_bootstrap_seed": 2026081808,
    }
    if dict(CANONICAL_PHASE10_SEEDS) != expected_seeds:
        problems.append(
            f"root seeds moved: {CANONICAL_PHASE10_SEEDS} != contract {expected_seeds}"
        )

    log("verifying the Phase 9 checkpoint and the Phase 8 anchor")
    phase9 = checkpoint_identity(CHECKPOINT_PATH)
    if phase9["sha256"] != ACCEPTED_PHASE9_SHA256:
        problems.append(f"Phase 9 checkpoint SHA {phase9['sha256']} != accepted")
    if phase9["model_state_digest"] != ACCEPTED_PHASE9_STATE_DIGEST:
        problems.append("Phase 9 model-state digest != accepted")
    if phase9["parameters"] != ACCEPTED_PHASE9_PARAMETERS:
        problems.append(f"Phase 9 parameter count {phase9['parameters']} != accepted")
    if not phase9["all_parameters_finite"]:
        problems.append("a Phase 9 parameter is non-finite")

    from stratego.training.phase9_contract import EXPECTED_PHASE8_CHECKPOINT_SHA256

    anchor = {
        "phase8_checkpoint_sha256": file_sha256(PHASE8_CHECKPOINT),
        "anchor_export_sha256": file_sha256(ANCHOR_EXPORT_PATH),
        "anchor_export_path": str(ANCHOR_EXPORT_PATH.relative_to(REPOSITORY_ROOT)),
        "accepted_phase8_checkpoint_sha256": EXPECTED_PHASE8_CHECKPOINT_SHA256,
    }
    accepted_anchor = _find_accepted_anchor_sha(
        read_json(REPOSITORY_ROOT / "reports" / "phase_9_data" / "agent_08_final_acceptance.json")
    )
    anchor["accepted_export_sha256"] = accepted_anchor
    if accepted_anchor is None:
        problems.append("the accepted Phase 9 anchor export SHA could not be found")
    elif anchor["anchor_export_sha256"] != accepted_anchor:
        problems.append("Phase 8 anchor export SHA != the accepted Phase 9 value")
    if anchor["phase8_checkpoint_sha256"] != EXPECTED_PHASE8_CHECKPOINT_SHA256:
        problems.append("the Phase 8 anchor checkpoint SHA != the accepted Phase 9 value")

    log("verifying the Phase 7 library from live bytes")
    index = load_library_index()
    entries = index.entries
    library = {
        "content_digest": library_content_digest(entries),
        "metadata_digest": entry_metadata_digest(entries),
        "manifest_digest_pinned": PHASE7_LIBRARY_MANIFEST_DIGEST,
        "families": len(FAMILY_IDS),
        "bases": len(entries),
    }
    if library["content_digest"] != PHASE7_LIBRARY_CONTENT_DIGEST:
        problems.append("Phase 7 library content digest != accepted")
    if library["metadata_digest"] != PHASE7_LIBRARY_METADATA_DIGEST:
        problems.append("Phase 7 library metadata digest != accepted")
    splits = {"train": 0, "validation": 0, "test": 0}
    from stratego.setups.contracts import parse_base_setup_id, split_for_base_index

    for entry in entries:
        _, _, base_index = parse_base_setup_id(entry.base_setup_id)
        splits[split_for_base_index(base_index)] += 1
    library["splits"] = dict(splits)
    if splits != {"train": 6400, "validation": 800, "test": 800}:
        problems.append(f"Phase 7 splits {splits} != accepted 6400/800/800")

    log("verifying the fitted utility, the scaler and neutral_v1")
    utility_sha = file_sha256(UTILITY_PATH)
    if utility_sha != ACCEPTED_UTILITY_FILE_SHA256:
        problems.append(f"setup_utility_v1 file SHA {utility_sha} != accepted")
    coefficient_digests, scaler_digest, utility_problems = _verify_utility_artifact()
    problems.extend(utility_problems)
    scorer = load_scorer()
    neutral = {
        "name": NEUTRAL_PROFILE.name,
        "reflection_probability": NEUTRAL_PROFILE.reflection_probability,
        "perturbation_probability": NEUTRAL_PROFILE.perturbation_probability,
    }
    if neutral["name"] != "neutral_v1":
        problems.append(f"the baseline profile is {neutral['name']!r}, expected neutral_v1")
    if (neutral["reflection_probability"], neutral["perturbation_probability"]) != (0.5, 0.5):
        problems.append("neutral_v1 reflection/perturbation probabilities moved")

    log("verifying the frozen selector configuration and the six candidates")
    config = read_json(CONFIG_ARTIFACT)
    config_sha = file_sha256(CONFIG_ARTIFACT)
    if config_sha != ACCEPTED_CONFIG_SHA256:
        problems.append(f"selector config SHA {config_sha} != accepted")
    if config_sha != SELECTED_CONFIG_SHA256:
        problems.append("the soak module pins a different selector config SHA")
    winner = config.get("winner", {})
    for key, expected in SELECTED_WINNER.items():
        observed = winner.get(key)
        if key == "temperature":
            matches = float(observed) == float(expected)
        else:
            matches = observed == expected
        if not matches:
            problems.append(f"frozen winner {key} is {observed!r}, expected {expected!r}")
    if SELECTED_CANDIDATE_ID != SELECTED_WINNER["candidate_id"]:
        problems.append("the soak module pins a different winner candidate id")
    matrix = {entry["candidate_id"]: entry for entry in CANDIDATE_MATRIX}
    if tuple(matrix) != CANDIDATE_IDS or len(CANDIDATE_IDS) != 6:
        problems.append("the frozen candidate matrix is not exactly the six candidates")
    frozen_winner_row = matrix[SELECTED_WINNER["candidate_id"]]
    if (
        frozen_winner_row["utility_model"] != SELECTED_WINNER["utility_model"]
        or float(frozen_winner_row["temperature"]) != SELECTED_WINNER["temperature"]
    ):
        problems.append("the winner's model/temperature disagree with the frozen matrix")

    log("recomputing every published selector distribution digest (36 cells)")
    agent4 = read_json(DATA_DIRECTORY / "agent_04_acceptance.json")
    handoff4 = agent4["handoff_to_agent_5"]
    distribution_digests, distribution_problems = _recompute_distribution_digests(
        handoff4["distribution_digests"], scorer, index
    )
    problems.extend(distribution_problems)
    winner_id = SELECTED_WINNER["candidate_id"]
    production = {
        color: distribution_digests[winner_id][color]["train"] for color in ("red", "blue")
    }
    for color, expected in ACCEPTED_TRAIN_DIGESTS.items():
        if production[color] != expected:
            problems.append(f"production {color} train digest {production[color]} != accepted")

    from stratego.training.phase10_selector import selector_contract_digest

    recomputed_contract = selector_contract_digest(distribution_digests)
    if recomputed_contract != ACCEPTED_SELECTOR_CONTRACT_DIGEST:
        problems.append("selector contract digest recomputes to an unaccepted value")

    log("verifying phase10_system_v1: frozen template vs filled instance")
    system_report, system_problems = _verify_system_document(document_digest, system_document)
    problems.extend(system_problems)

    log("verifying the sealed Agent 2 corpus from live bytes")
    corpus, corpus_problems = _verify_corpus()
    problems.extend(corpus_problems)

    log("verifying the sealed Agent 6 soak from live bytes")
    soak_record, soak_problems = _verify_soak()
    problems.extend(soak_problems)

    log("harvesting every recorded test-bank access across Agents 1-6")
    access_history, access_problems = _pre_agent7_access_history()
    problems.extend(access_problems)

    log("verifying both evaluation bank identities (structural only)")
    from stratego.evaluation.phase10_banks import (
        bank_digest,
        build_phase10_bank,
        manifest_digest,
    )

    validation_cases, validation_manifest = build_phase10_bank("validation")
    record_bank_access("verify", "phase10_validation_bank_v1", "digest_computation",
                       neural=False, outcomes=False)
    validation = {
        "bank_version": "phase10_validation_bank_v1",
        "cases": len(validation_cases),
        "bank_digest": bank_digest(validation_cases),
        "manifest_digest": manifest_digest(validation_manifest),
    }
    if validation["bank_digest"] != ACCEPTED_VALIDATION_BANK_DIGEST:
        problems.append(f"validation bank digest {validation['bank_digest']} != accepted")
    if validation["manifest_digest"] != ACCEPTED_VALIDATION_MANIFEST_DIGEST:
        problems.append("validation bank manifest digest != accepted")
    del validation_cases

    test_cases, test_manifest = build_phase10_bank("test")
    record_bank_access("verify", "phase10_test_bank_v1", "structural_digest_only",
                       neural=False, outcomes=False)
    test_bank = {
        "bank_version": "phase10_test_bank_v1",
        "cases": len(test_cases),
        "bank_digest": bank_digest(test_cases),
        "manifest_digest": manifest_digest(test_manifest),
        "access": "structural_digest_only",
        "outcomes_read": 0,
    }
    if test_bank["bank_digest"] != ACCEPTED_TEST_BANK_DIGEST:
        problems.append(f"test bank digest {test_bank['bank_digest']} != accepted")
    if test_bank["manifest_digest"] != ACCEPTED_TEST_MANIFEST_DIGEST:
        problems.append("test bank manifest digest != accepted")
    if test_bank["cases"] != 512:
        problems.append(f"test bank has {test_bank['cases']} cases, expected 512")
    del test_cases

    isolation, isolation_meta = phase9_isolation_set()

    payload = {
        "stage": "verify",
        "problems": problems,
        "administrative_freeze": freeze,
        "prior_agents": prior,
        "contract_digests": digests,
        "contract_bundle_digest": bundle,
        "root_seeds": dict(CANONICAL_PHASE10_SEEDS),
        "phase9_checkpoint": phase9,
        "phase8_anchor": anchor,
        "library": library,
        "utility": {
            "file_sha256": utility_sha,
            "coefficient_digests": coefficient_digests,
            "scaler_digest": scaler_digest,
            "refit_by_agent_7": False,
        },
        "neutral_v1": neutral,
        "selector_config": {
            "artifact_sha256": config_sha,
            "winner": {key: winner.get(key) for key in SELECTED_WINNER},
        },
        "mixture": {
            "neutral_weight": NEUTRAL_MIXTURE_WEIGHT,
            "learned_weight": LEARNED_MIXTURE_WEIGHT,
        },
        "distribution_digests_recomputed": distribution_digests,
        "production_train_digests": production,
        "selector_contract_digest": recomputed_contract,
        "phase10_system_v1": system_report,
        "corpus": corpus,
        "soak": soak_record,
        "pre_agent7_test_bank_access": access_history,
        "validation_bank": validation,
        "test_bank": test_bank,
        "phase9_isolation_set_size": len(isolation),
        "phase9_isolation_meta": isolation_meta,
        "bank_access_log": list(BANK_ACCESS_LOG),
        "environment": environment_record(),
    }
    write_stage("verify", payload)
    if problems:
        for problem in problems:
            log(f"  PROBLEM: {problem}")
        raise Agent7Error(f"{len(problems)} prerequisite problem(s); Agent 7 is BLOCKED")
    log(
        f"  verified: freeze {head[:12]}, bundle {bundle[:12]}, system "
        f"{system_report['filled_instance_digest'][:12]}, test bank "
        f"{test_bank['bank_digest'][:12]}, prior test outcomes 0"
    )
    return payload


def _verify_utility_artifact() -> tuple:
    """Recompute both coefficient digests and the scaler from live bytes."""
    from stratego.training.phase10_utility_fit import document_digest

    problems: list = []
    artifact = read_json(UTILITY_PATH)
    tracked = read_json(DATA_DIRECTORY / "agent_03_utility_models.json")
    recomputed = {}
    for model_id, entry in sorted(artifact["models"].items()):
        digest = document_digest(
            {
                "utility_version": entry["utility_version"],
                "model_id": entry["model_id"],
                "colour_order": entry["colour_order"],
                "family_order": entry["family_order"],
                "feature_order": entry["feature_order"],
                "red_first_intercept": entry["red_first_intercept"],
                "family_offsets_raw": entry["family_offsets_raw"],
                "trait_weights": entry["trait_weights"],
            }
        )
        recomputed[model_id] = digest
        if digest != ACCEPTED_COEFFICIENT_DIGESTS.get(model_id):
            problems.append(f"{model_id} coefficient digest {digest} != accepted")
        if digest != entry["coefficient_digest"]:
            problems.append(f"{model_id} stored digest disagrees with its own coefficients")
    if sorted(artifact["models"]) != ["model_F", "model_T"]:
        problems.append(
            f"the utility artifact holds models {sorted(artifact['models'])}, "
            "expected exactly model_F and model_T"
        )
    for model_id in artifact["models"]:
        for field in (
            "utility_version", "model_id", "colour_order", "family_order",
            "feature_order", "red_first_intercept", "family_offsets_raw",
            "trait_weights", "coefficient_digest",
        ):
            if tracked["models"][model_id][field] != artifact["models"][model_id][field]:
                problems.append(f"{model_id}.{field} differs from the tracked Agent 3 record")
    scaler_digest = artifact["scaler_digest"]
    if scaler_digest != ACCEPTED_SCALER_DIGEST:
        problems.append(f"trait scaler digest {scaler_digest} != accepted")
    return recomputed, scaler_digest, problems


def _find_accepted_anchor_sha(record) -> "str | None":
    """The accepted Phase 8 anchor export SHA, wherever Phase 9 recorded it."""
    stack = [record]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "anchor_export_sha256" and isinstance(value, str):
                    return value
                stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return None


def _recompute_distribution_digests(published: dict, scorer, index) -> tuple:
    """Rebuild every candidate x colour x split digest and compare."""
    from stratego.training.phase10_selector import build_distribution, candidate

    problems: list = []
    recomputed: dict = {}
    for candidate_id, by_color in sorted(published.items()):
        recomputed[candidate_id] = {}
        selector = candidate(candidate_id)
        for color, by_split in sorted(by_color.items()):
            recomputed[candidate_id][color] = {}
            for split, digest in sorted(by_split.items()):
                distribution = build_distribution(selector, color, split, scorer, index)
                observed = distribution.probability_vector_digest()
                recomputed[candidate_id][color][split] = observed
                if observed != digest:
                    problems.append(
                        f"{candidate_id}/{color}/{split} probability digest {observed} "
                        f"!= Agent 4's published {digest}"
                    )
                if not distribution.mixture_is_exact():
                    problems.append(f"{candidate_id}/{color}/{split} mixture is not 0.35/0.65")
    return recomputed, problems


def _verify_system_document(document_digest, system_document) -> tuple:
    """The frozen template and the filled instance, distinguished and checked.

    Two different identities share the name `phase10_system_v1` and must not
    be conflated: the Agent 1 **contract** is the binding schema — slots and
    filling rules, digest pinned in the bundle — while the Agent 6
    **instance** is that schema filled with the accepted values. A changed
    instance digest is not a changed upstream contract; what makes the
    instance acceptable is that the template digest still recomputes and the
    instance satisfies the template's filling rules.
    """
    from stratego.training.phase10_contract import (
        LEARNED_MIXTURE_WEIGHT,
        NEUTRAL_MIXTURE_WEIGHT,
        CANDIDATE_IDS,
    )
    from tests.training.phase10_frozen_digests import CONTRACT_DIGESTS

    problems: list = []
    template = system_document()
    template_digest = document_digest(template)
    if template_digest != CONTRACT_DIGESTS["phase10_system_v1"]:
        problems.append(
            f"the frozen phase10_system_v1 template digest {template_digest} != "
            "the Agent 1 freeze; the upstream contract itself moved"
        )

    manifest = read_json(PRODUCTION_MANIFEST)
    instance = manifest["phase10_system_v1"]
    instance_digest = document_digest(instance)
    if instance_digest != ACCEPTED_SYSTEM_DIGEST:
        problems.append(
            f"the filled phase10_system_v1 instance digest {instance_digest} != "
            "the accepted production identity"
        )
    if manifest.get("phase10_system_v1_digest") != instance_digest:
        problems.append("the manifest's stored system digest disagrees with its document")

    filling: dict = {}

    filling["binding_schema_digest_recomputed"] = (
        instance.get("binding_schema_digest") == template_digest
        and instance.get("binding_schema_pinned_digest") == template_digest
    )

    utility = instance.get("accepted_utility_model", {})
    filling["accepted_utility_model"] = (
        utility.get("utility_version") == "phase10_setup_utility_v1"
        and utility.get("model_id") in ("model_F", "model_T")
        and utility.get("model_id") == SELECTED_WINNER["utility_model"]
        and utility.get("coefficient_digest")
        == ACCEPTED_COEFFICIENT_DIGESTS[SELECTED_WINNER["utility_model"]]
        and utility.get("fit_corpus_content_digest") == ACCEPTED_CORPUS_CONTENT_DIGEST
        and utility.get("single_fit") is True
    )

    scaler = instance.get("accepted_trait_scaler", {})
    filling["accepted_trait_scaler"] = (
        scaler.get("scaler_version") == "phase10_trait_scaler_v1"
        and scaler.get("scaler_digest") == ACCEPTED_SCALER_DIGEST
    )

    config = instance.get("selected_selector_config", {})
    mixture = config.get("mixture", {})
    filling["selected_selector_config"] = (
        config.get("candidate_id") in CANDIDATE_IDS
        and config.get("candidate_id") == SELECTED_WINNER["candidate_id"]
        and float(config.get("temperature", -1.0)) == SELECTED_WINNER["temperature"]
        and float(mixture.get("neutral_weight", -1.0)) == NEUTRAL_MIXTURE_WEIGHT
        and float(mixture.get("learned_weight", -1.0)) == LEARNED_MIXTURE_WEIGHT
        and config.get("production_source_version") == "learned_setup_source_v1"
        and config.get("config_artifact_sha256") == ACCEPTED_CONFIG_SHA256
    )

    template_bound = template["bound_now"]
    move_model = instance.get("move_model", {})
    # The template's bound_now names the checkpoint identifier under `path`;
    # the filled instance names the identical repository-relative identifier
    # under `artifact`. The binding is judged on values, never key spelling.
    filling["bound_move_model_unchanged"] = move_model.get(
        "artifact", move_model.get("path")
    ) == template_bound["move_model"]["path"] and all(
        move_model.get(key) == template_bound["move_model"][key]
        for key in ("sha256", "model_state_digest", "parameters", "mutability")
    )
    filling["bound_library_unchanged"] = instance.get("library") == template_bound["library"]
    filling["bound_post_selection_path_unchanged"] = (
        instance.get("reflection_perturbation") == template_bound["post_selection_path"]
    )
    filling["neutral_v1_not_redefined"] = (
        instance.get("neutral_v1", {}).get("profile") == template_bound["baseline_profile"]
        and instance.get("neutral_v1", {}).get("redefined") is False
    )
    filling["separation_rule_carried"] = (
        instance.get("separation_rule") == template["separation_rule"]
    )
    serialized = json.dumps(instance)
    filling["no_filesystem_path_in_identity"] = (
        "/Volumes/" not in serialized and "/Users/" not in serialized
    )
    filling["production_train_digests_bound"] = (
        instance.get("production_distributions", {}).get("red_digest")
        == ACCEPTED_TRAIN_DIGESTS["red"]
        and instance.get("production_distributions", {}).get("blue_digest")
        == ACCEPTED_TRAIN_DIGESTS["blue"]
    )
    filling["evaluation_banks_bound"] = (
        instance.get("evaluation_banks", {}).get("test", {}).get("bank_digest")
        == ACCEPTED_TEST_BANK_DIGEST
        and instance.get("evaluation_banks", {}).get("validation", {}).get("bank_digest")
        == ACCEPTED_VALIDATION_BANK_DIGEST
    )

    for name, value in filling.items():
        if not value:
            problems.append(f"phase10_system_v1 filling rule failed: {name}")

    report = {
        "frozen_template_digest": template_digest,
        "frozen_template_pinned": CONTRACT_DIGESTS["phase10_system_v1"],
        "filled_instance_digest": instance_digest,
        "filled_instance_accepted": ACCEPTED_SYSTEM_DIGEST,
        "distinction": (
            "the template digest is the Agent 1 contract identity (slots and "
            "filling rules, part of the frozen bundle); the instance digest is "
            "Agent 6's filled production document. They are different objects "
            "with different digests by design, and the instance is verified "
            "against the template's filling rules rather than against the "
            "template's digest"
        ),
        "filling_rules": filling,
        "all_filling_rules_pass": all(filling.values()),
    }
    return report, problems


def _verify_corpus() -> tuple:
    """The sealed outcome corpus, re-verified from live bytes."""
    from stratego.training.phase10_outcome_store import verify_seal
    from stratego.training.phase10_storage import check_corpus_root, default_corpus_root

    problems: list = []
    root = default_corpus_root()
    findings = check_corpus_root(root)
    if not findings["usable"]:
        problems.append(f"the corpus root is not usable: {findings['blocked']}; BLOCKED")
        return {"root_diagnostic": str(root), "usable": False}, problems
    seal = verify_seal(root)
    corpus = {
        "root_diagnostic": str(root),
        "usable": True,
        "seal_all_pass": seal["all_pass"],
        "content_digest": seal["observed_content_digest"],
        "committed_games": seal["observed_committed_games"],
        "train_only": "phase10_outcome_v1",
    }
    if not seal["all_pass"]:
        problems.append("the sealed corpus fails its own seal verification")
    if corpus["content_digest"] != ACCEPTED_CORPUS_CONTENT_DIGEST:
        problems.append(f"corpus content digest {corpus['content_digest']} != accepted")
    if corpus["committed_games"] != 16_384:
        problems.append(f"corpus holds {corpus['committed_games']} games, expected 16,384")
    return corpus, problems


def _verify_soak() -> tuple:
    """The sealed Agent 6 soak store, re-verified from live bytes."""
    from stratego.training.phase10_soak import default_soak_root, verify_soak_seal

    problems: list = []
    root = default_soak_root()
    seal = verify_soak_seal(root)
    record = {
        "root_diagnostic": str(root),
        "seal_all_pass": seal["all_pass"],
        "content_digest": seal["observed_content_digest"],
        "committed_games": seal["observed_committed_games"],
    }
    if not seal["all_pass"]:
        problems.append("the sealed soak store fails its own seal verification")
    if record["content_digest"] != ACCEPTED_SOAK_CONTENT_DIGEST:
        problems.append(f"soak content digest {record['content_digest']} != accepted")
    if record["committed_games"] != 8_192:
        problems.append(f"soak holds {record['committed_games']} games, expected 8,192")
    return record, problems


def _pre_agent7_access_history() -> tuple:
    """Every recorded test-bank access across Agents 1-6, harvested and checked.

    The proof that `phase10_test_bank_v1` has had zero outcome evaluation
    before this run: every ledger entry any prior agent recorded for the
    test bank must be structural (`neural=false`, `outcomes=false`), the
    Agent 5/6 discipline counters must record zero test outcomes read, and
    no stored evaluation cell anywhere in the work tree may name the test
    bank version.
    """
    problems: list = []
    entries: list = []
    for agent, artifact in PRIOR_AGENT_ARTIFACTS.items():
        record = read_json(DATA_DIRECTORY / artifact)
        # Agent 1 records its test-bank accesses under `test_bank_access_log`
        # (entries carry no bank field: the log itself names the bank); later
        # agents record both banks under `bank_access_log`.
        recorded = [
            entry | {"bank": entry.get("bank", "phase10_test_bank_v1")}
            for entry in record.get("test_bank_access_log", [])
        ] + list(record.get("bank_access_log", []))
        for entry in recorded:
            if entry.get("bank") == "phase10_test_bank_v1":
                entries.append({"agent": agent, "source": artifact, **entry})
                if entry.get("neural") or entry.get("outcomes"):
                    problems.append(
                        f"Agent {agent} recorded a non-structural test-bank access: {entry}"
                    )
    for agent in (1, 2, 3, 4, 5, 6):
        stage_dir = REPOSITORY_ROOT / "checkpoints" / "phase10" / f"agent0{agent}"
        for path in sorted(stage_dir.rglob("stage_*.json")):
            try:
                record = read_json(path)
            except Exception:  # noqa: BLE001 -- unreadable stage files are not ledgers
                continue
            for entry in record.get("bank_access_log", []):
                if entry.get("bank") == "phase10_test_bank_v1":
                    entries.append(
                        {"agent": agent, "source": str(path.relative_to(REPOSITORY_ROOT)), **entry}
                    )
                    if entry.get("neural") or entry.get("outcomes"):
                        problems.append(
                            f"stage record {path.name} shows a non-structural "
                            f"test-bank access: {entry}"
                        )

    agent5 = read_json(DATA_DIRECTORY / "agent_05_acceptance.json")
    discipline5 = agent5.get("discipline", {})
    if int(discipline5.get("test_bank_outcome_access", -1)) != 0:
        problems.append("Agent 5 discipline does not record zero test-bank outcome reads")
    agent6 = read_json(DATA_DIRECTORY / "agent_06_acceptance.json")
    unopened = agent6.get("test_bank_unopened", {})
    for key in ("games", "neural_inference", "outcomes_read"):
        if int(unopened.get(key, -1)) != 0:
            problems.append(f"Agent 6 records test-bank {key} != 0")

    from stratego.training.phase10_contract import TEST_BANK_VERSION

    stored_cells: list = []
    for agent in (1, 2, 3, 4, 5, 6):
        games_dir = REPOSITORY_ROOT / "checkpoints" / "phase10" / f"agent0{agent}" / "games"
        if not games_dir.exists():
            continue
        for cell in sorted(path.name for path in games_dir.iterdir()):
            stored_cells.append({"agent": agent, "cell": cell})
            if TEST_BANK_VERSION in cell:
                problems.append(
                    f"Agent {agent} stored an evaluation cell naming the test bank: {cell}"
                )

    history = {
        "test_bank_ledger_entries": entries,
        "all_entries_structural": all(
            not entry.get("neural") and not entry.get("outcomes") for entry in entries
        ),
        "agent5_test_bank_outcome_access": int(discipline5.get("test_bank_outcome_access", -1)),
        "agent6_test_bank_unopened": {
            key: int(unopened.get(key, -1))
            for key in ("games", "neural_inference", "outcomes_read")
        },
        "stored_evaluation_cells_checked": len(stored_cells),
        "cells_naming_test_bank": [
            cell for cell in stored_cells if TEST_BANK_VERSION in cell["cell"]
        ],
        "prior_outcome_evaluations": 0 if not problems else None,
        "conclusion": (
            "every recorded pre-Agent-7 test-bank access is structural digest "
            "recomputation with neural=false and outcomes=false; no stored "
            "evaluation cell names the test bank; Agent 7 performs the first "
            "outcome evaluation"
        ),
    }
    return history, problems


# ---------------------------------------------------------------------------
# stage: banks — independent rebuild and exhaustive structural audit
# ---------------------------------------------------------------------------


def stage_banks(args) -> dict:
    from stratego.evaluation.phase10_banks import (
        audit_phase10_bank,
        build_phase10_bank,
        cross_bank_isolation,
        manifest_digest,
        phase9_isolation_set,
        phase9_raw_board_coverage,
    )

    read_stage("verify")
    problems: list = []
    isolation, isolation_meta = phase9_isolation_set()
    if len(isolation) != 1184:
        problems.append(f"isolation set holds {len(isolation)} identities, expected 1,184")

    coverage = phase9_raw_board_coverage()
    if not coverage["all_pass"]:
        problems.append("the Phase 9 raw-board coverage reconciliation failed")

    log("rebuilding the validation bank and auditing every case")
    validation_cases, validation_manifest = build_phase10_bank(
        "validation", isolation, isolation_meta
    )
    record_bank_access("banks", "phase10_validation_bank_v1", "structural_rebuild_audit",
                       neural=False, outcomes=False)
    validation_audit = audit_phase10_bank(
        "validation", validation_cases, validation_manifest, isolation,
        rebuild_sample_every=1,
    )
    if not validation_audit["all_pass"]:
        problems.append("the validation bank structural audit failed")
    if validation_audit["bank_digest"] != ACCEPTED_VALIDATION_BANK_DIGEST:
        problems.append("the rebuilt validation bank digest != accepted")

    log("rebuilding the sealed test bank and auditing every case")
    test_cases, test_manifest = build_phase10_bank("test", isolation, isolation_meta)
    record_bank_access("banks", "phase10_test_bank_v1", "structural_rebuild_audit",
                       neural=False, outcomes=False)
    test_audit = audit_phase10_bank(
        "test", test_cases, test_manifest, isolation, rebuild_sample_every=1
    )
    if not test_audit["all_pass"]:
        problems.append("the test bank structural audit failed")
    if test_audit["bank_digest"] != ACCEPTED_TEST_BANK_DIGEST:
        problems.append("the rebuilt test bank digest != accepted")
    if manifest_digest(test_manifest) != ACCEPTED_TEST_MANIFEST_DIGEST:
        problems.append("the rebuilt test bank manifest digest != accepted")

    cross = cross_bank_isolation(validation_cases, test_cases)
    if not cross["zero_overlap"]:
        problems.append("the two Phase 10 banks share a frozen fingerprint")

    payload = {
        "stage": "banks",
        "problems": problems,
        "phase9_isolation": {
            "set_size": len(isolation),
            "set_digest": isolation_meta["set_digest"],
            "raw_board_coverage": coverage,
        },
        "validation_audit": validation_audit,
        "test_audit": test_audit,
        "cross_bank_isolation": cross,
        "rebuild_sample_every": 1,
        "rebuild_note": (
            "rebuild_sample_every=1: every case of both banks was additionally "
            "rebuilt in isolation from its identity and required to equal its "
            "stored construction"
        ),
        "bank_access_log": list(BANK_ACCESS_LOG),
        "environment": environment_record(),
    }
    write_stage("banks", payload)
    if problems:
        for problem in problems:
            log(f"  PROBLEM: {problem}")
        raise Agent7Error(f"{len(problems)} bank problem(s); Agent 7 is BLOCKED")
    log(
        f"  both banks rebuilt exactly: validation {validation_audit['bank_digest'][:12]}, "
        f"test {test_audit['bank_digest'][:12]}, cross-overlap 0"
    )
    return payload


# ---------------------------------------------------------------------------
# stage: games — the first and only final-test outcome evaluation
# ---------------------------------------------------------------------------


def work_units(case_count: int, chunk: int) -> list:
    """Every `(arm, candidate, matchup, case slice)` final-test unit, frozen order."""
    from stratego.evaluation.phase10_final import (
        FINAL_NEUTRAL_ARM_MATCHUPS,
        SELECTED_CANDIDATE_ID,
    )
    from stratego.evaluation.phase10_validation import ARM_LEARNED, ARM_NEUTRAL
    from stratego.training.phase10_contract import MATCHUP_TOKENS

    units = []
    cells = [(ARM_NEUTRAL, None, token) for token in FINAL_NEUTRAL_ARM_MATCHUPS]
    cells += [(ARM_LEARNED, SELECTED_CANDIDATE_ID, token) for token in MATCHUP_TOKENS]
    for arm, candidate_id, matchup in cells:
        for start in range(0, case_count, chunk):
            units.append(
                {
                    "arm": arm,
                    "candidate_id": candidate_id,
                    "matchup": matchup,
                    "start": start,
                    "stop": min(start + chunk, case_count),
                }
            )
    return units


def unit_path(unit: dict) -> Path:
    selector = unit["candidate_id"] or "neutral_v1"
    directory = GAMES_DIRECTORY / f"{unit['arm']}__{selector}__{unit['matchup']}"
    return directory / f"cases_{unit['start']:04d}_{unit['stop']:04d}.pkl"


def run_unit(unit: dict, *, cases, source, own_ref, own_policy, opponents, isolation) -> dict:
    """Play one final-test work unit's games and return its rows."""
    from stratego.evaluation.match_runner import ON_POLICY_ERROR_QUARANTINE
    from stratego.evaluation.phase10_final import play_final_game
    from stratego.evaluation.phase10_validation import (
        ARM_LEARNED,
        EXTERNAL_OPPONENT_POLICY_IDS,
        NEURAL_OPPONENT_MATCHUP,
        game_setups,
        learned_own_side,
        neutral_own_side,
    )
    from stratego.evaluation.registry import policy_ref
    from stratego.training.phase10_contract import MATCHUP_LEARNED_VS_NEUTRAL

    matchup = unit["matchup"]
    arm = unit["arm"]
    candidate_id = unit["candidate_id"]
    if matchup == MATCHUP_LEARNED_VS_NEUTRAL:
        opponent_ref, opponent_policy = own_ref, None
    elif matchup == NEURAL_OPPONENT_MATCHUP:
        opponent_ref, opponent_policy = opponents["anchor"]
    else:
        opponent_ref = policy_ref(EXTERNAL_OPPONENT_POLICY_IDS[matchup])
        opponent_policy = opponents[matchup]

    rows = []
    started = time.perf_counter()
    for case in cases[unit["start"] : unit["stop"]]:
        if arm == ARM_LEARNED:
            own = {
                color: learned_own_side(source, case, color)
                for color in ("red", "blue")
            }
        else:
            own = {color: neutral_own_side(case, color) for color in ("red", "blue")}
        for setup_row in game_setups(case, matchup, own):
            spec, result = play_final_game(
                case,
                setup_row,
                matchup,
                arm=arm,
                candidate_id=candidate_id,
                own_ref=own_ref,
                opponent_ref=opponent_ref,
                own_policy=own_policy,
                opponent_policy=opponent_policy,
                record_actions=True,
                on_policy_error=ON_POLICY_ERROR_QUARANTINE,
            )
            own_draw = own[setup_row["own_color"]]
            rows.append(
                {
                    "match_id": result.match_id,
                    "case_id": case.case_id,
                    "case_family": case.family_id,
                    "case_index": case.case_index,
                    "game_index": setup_row["game_index"],
                    "own_color": setup_row["own_color"],
                    "arm": arm,
                    "candidate_id": candidate_id,
                    "matchup": matchup,
                    "candidate_result": result.candidate_result,
                    "score": None if result.errored else float(result.candidate_score),
                    "terminal_reason": result.terminal_reason,
                    "plies": int(result.plies),
                    "decisions": int(result.decisions),
                    "replay_digest": result.replay_digest,
                    "own_fingerprint": own_draw.final_setup_fingerprint,
                    "own_base_setup_id": own_draw.base_setup_id,
                    "own_family_id": own_draw.family_id,
                    "own_branch": own_draw.branch,
                    "own_landed_in_phase9_set": own_draw.final_setup_fingerprint in isolation,
                    "policy_error": result.policy_error,
                    "policy_error_category": result.policy_error_category,
                    "setup_bank_version": result.setup_bank_version,
                    "root_seed": int(spec.root_seed),
                    "result_row": result.to_dict(),
                }
            )
    return {
        "unit": dict(unit),
        "rows": rows,
        "seconds": time.perf_counter() - started,
    }


def cell_worker(args) -> None:
    """One worker process: every final-test unit whose position matches this slice."""
    import torch

    torch.set_num_threads(args.torch_threads)
    from stratego.evaluation.neural_worker import (
        DECISION_MODE_GREEDY,
        InferenceOwner,
        LocalInferenceChannel,
        RemoteNeuralPolicy,
        neural_policy_ref,
    )
    from stratego.evaluation.phase10_banks import phase9_isolation_set
    from stratego.evaluation.phase10_final import SELECTED_CANDIDATE_ID, final_cases
    from stratego.evaluation.phase10_validation import (
        EVAL_DTYPE,
        EXTERNAL_OPPONENT_POLICY_IDS,
        PHASE10_EVAL_MOVE_POLICY_ID,
        PHASE8_ANCHOR_CANDIDATE_ID,
    )
    from stratego.evaluation.registry import build_policy
    from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from stratego.training.phase10_selector import (
        LearnedSetupSource,
        candidate,
        load_library_index,
        load_scorer,
    )

    cases, _manifest = final_cases()
    cases = cases[: args.cases]
    units = work_units(len(cases), args.chunk)
    mine = [unit for position, unit in enumerate(units) if position % args.workers == args.worker]
    pending = [unit for unit in mine if not unit_path(unit).exists()]
    if not pending:
        return

    isolation, _meta = phase9_isolation_set()
    scorer = load_scorer()
    index = load_library_index()
    source = LearnedSetupSource(candidate(SELECTED_CANDIDATE_ID), scorer, index)
    own_ref = neural_policy_ref(
        PHASE10_EVAL_MOVE_POLICY_ID, decision_mode=DECISION_MODE_GREEDY, dtype_name=EVAL_DTYPE
    )
    anchor_ref = neural_policy_ref(
        PHASE8_ANCHOR_CANDIDATE_ID, decision_mode=DECISION_MODE_GREEDY, dtype_name=EVAL_DTYPE
    )
    owners = {
        "own": InferenceOwner(
            EXPORT_PATH,
            decision_mode=DECISION_MODE_GREEDY,
            device=args.device,
            dtype=EVAL_DTYPE,
            expected_architecture_id=ARCHITECTURE_FAMILY,
            expected_configuration=candidate_config("C1"),
            name=f"a7_phase9_w{args.worker:02d}",
        ),
        "anchor": InferenceOwner(
            ANCHOR_EXPORT_PATH,
            decision_mode=DECISION_MODE_GREEDY,
            device=args.device,
            dtype=EVAL_DTYPE,
            expected_architecture_id=ARCHITECTURE_FAMILY,
            expected_configuration=candidate_config("C1"),
            name=f"a7_anchor_w{args.worker:02d}",
        ),
    }
    try:
        own_policy = RemoteNeuralPolicy(
            own_ref, LocalInferenceChannel(owners["own"]), decision_mode=DECISION_MODE_GREEDY
        )
        anchor_policy = RemoteNeuralPolicy(
            anchor_ref, LocalInferenceChannel(owners["anchor"]),
            decision_mode=DECISION_MODE_GREEDY,
        )
        opponents = {"anchor": (anchor_ref, anchor_policy)}
        for token, policy_id in EXTERNAL_OPPONENT_POLICY_IDS.items():
            opponents[token] = build_policy(policy_id)
        for unit in pending:
            path = unit_path(unit)
            if path.exists():
                continue
            produced = run_unit(
                unit,
                cases=cases,
                source=source,
                own_ref=own_ref,
                own_policy=own_policy,
                opponents=opponents,
                isolation=isolation,
            )
            produced["inference"] = {
                name: {
                    "requests_served": int(owner.stats().get("requests_served", 0)),
                    "failures_returned": int(owner.stats().get("failures_returned", 0)),
                }
                for name, owner in owners.items()
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as stream:
                pickle.dump(produced, stream)
            log(
                f"  w{args.worker:02d} {unit['arm']}/{unit['candidate_id'] or 'neutral_v1'}/"
                f"{unit['matchup']} [{unit['start']}:{unit['stop']}] "
                f"{len(produced['rows'])} games in {produced['seconds']:.1f}s"
            )
    finally:
        for owner in owners.values():
            owner.close()


def stage_games(args) -> dict:
    """Open the sealed bank once and run every final-test game, resumably."""
    from stratego.evaluation.phase10_final import final_cases
    from stratego.training import phase10_collector as collector

    verify = read_stage("verify")
    banks = read_stage("banks")
    if verify.get("problems") or banks.get("problems"):
        raise Agent7Error("prerequisite problems recorded; the sealed bank stays closed")

    log("exporting the accepted Phase 9 weights to the evaluation format")
    export = collector.export_evaluation_weights(CHECKPOINT_PATH, EXPORT_PATH)
    if export["source_sha256"] != ACCEPTED_PHASE9_SHA256:
        raise Agent7Error("the exported source is not the accepted Phase 9 checkpoint")
    if export["model_state_digest"] != ACCEPTED_PHASE9_STATE_DIGEST:
        raise Agent7Error("the evaluation export changed the model state")

    cases, manifest = final_cases()
    record_bank_access(
        "games", "phase10_test_bank_v1", "final_evaluation", neural=True, outcomes=True
    )
    authorization = {
        "bank": "phase10_test_bank_v1",
        "purpose": "final_evaluation",
        "authorized_agent": 7,
        "first_outcome_evaluation": True,
        "prior_outcome_evaluations": 0,
        "basis": (
            "stage verify harvested every recorded Agent 1-6 test-bank access "
            "and found all structural; this stage is the bank's first game, "
            "first neural inference and first outcome read"
        ),
    }
    cases = cases[: args.cases]
    units = work_units(len(cases), args.chunk)
    log(f"{len(units)} work units over {len(cases)} cases")

    pending = [unit for unit in units if not unit_path(unit).exists()]
    started = time.perf_counter()
    if pending:
        log(f"{len(pending)} unit(s) pending; launching {args.workers} worker(s)")
        processes = []
        for worker in range(args.workers):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--cell-worker",
                "--worker", str(worker),
                "--workers", str(args.workers),
                "--chunk", str(args.chunk),
                "--cases", str(args.cases),
                "--device", args.device,
                "--torch-threads", str(args.torch_threads),
            ]
            processes.append((worker, subprocess.Popen(command, cwd=REPOSITORY_ROOT)))
        failed = [worker for worker, process in processes if process.wait() != 0]
        if failed:
            raise Agent7Error(f"evaluation worker(s) {failed} failed")
    elapsed = time.perf_counter() - started

    missing = [unit for unit in units if not unit_path(unit).exists()]
    if missing:
        raise Agent7Error(f"{len(missing)} work unit(s) produced no result file")

    rows = []
    inference = {"requests_served": 0, "failures_returned": 0}
    for unit in units:
        with open(unit_path(unit), "rb") as stream:
            stored = pickle.load(stream)
        rows.extend(stored["rows"])
        for stats in stored.get("inference", {}).values():
            inference["requests_served"] += int(stats["requests_served"])
            inference["failures_returned"] += int(stats["failures_returned"])

    expected = sum(2 * (unit["stop"] - unit["start"]) for unit in units)
    if len(rows) != expected:
        raise Agent7Error(f"{len(rows)} game rows, expected {expected}")

    payload = {
        "stage": "games",
        "cases": len(cases),
        "bank_digest": manifest["bank_digest"],
        "authorization": authorization,
        "units": len(units),
        "games": len(rows),
        "wall_clock_seconds": elapsed,
        "workers": args.workers,
        "device": args.device,
        "export": export,
        "inference": inference,
        "bank_access_log": list(BANK_ACCESS_LOG),
        "environment": environment_record(),
    }
    write_stage("games", payload)
    slim = [{key: value for key, value in row.items() if key != "result_row"} for row in rows]
    with open(WORK_DIRECTORY / "rows.pkl", "wb") as stream:
        pickle.dump(slim, stream)
    log(f"  {len(rows)} games recorded in {elapsed:.1f}s")
    return payload


# ---------------------------------------------------------------------------
# stage: audit — every game replayed, every seat reconciled, every setup rebuilt
# ---------------------------------------------------------------------------


def expected_seat_counts(case_count: int) -> dict:
    """The seat totals the frozen matchup mapping implies, derived not observed."""
    direct_games = 2 * case_count
    external_games_per_matchup = 2 * case_count
    return {
        "phase9": 2 * direct_games + 5 * external_games_per_matchup + 5 * external_games_per_matchup,
        "each_external_opponent": 2 * external_games_per_matchup,
    }


def stage_audit_shard(args) -> None:
    """One audit worker: replay, reconcile and rebuild its slice of units."""
    from stratego.engine.setup import deserialize_setup, validate_setup
    from stratego.evaluation.match_runner import MatchResult, replay_stored_match
    from stratego.evaluation.neural_worker import DECISION_MODE_GREEDY, neural_policy_ref
    from stratego.evaluation.phase10_banks import phase9_isolation_set
    from stratego.evaluation.phase10_final import build_final_spec, final_cases
    from stratego.evaluation.phase10_validation import (
        EVAL_DTYPE,
        EXTERNAL_OPPONENT_POLICY_IDS,
        NEURAL_OPPONENT_MATCHUP,
        PHASE10_EVAL_MOVE_POLICY_ID,
        PHASE8_ANCHOR_CANDIDATE_ID,
        game_setups,
        learned_own_side,
        neutral_own_side,
    )
    from stratego.evaluation.registry import policy_ref
    from stratego.setups.contracts import parse_base_setup_id, split_for_base_index
    from stratego.setups.mobility import setup_has_initial_mobility
    from stratego.training.phase10_contract import MATCHUP_LEARNED_VS_NEUTRAL
    from stratego.training.phase10_seed import CASE_GAME_COLOR, case_match_seed
    from stratego.training.phase10_selector import (
        LearnedSetupSource,
        candidate,
        load_library_index,
        load_scorer,
    )
    from stratego.training.phase10_soak import SELECTED_CANDIDATE_ID

    cases, _manifest = final_cases()
    cases = cases[: args.cases]
    by_case = {case.case_id: case for case in cases}
    units = work_units(len(cases), args.chunk)
    mine = [unit for position, unit in enumerate(units) if position % args.workers == args.worker]

    isolation, _meta = phase9_isolation_set()
    scorer = load_scorer()
    index = load_library_index()
    source = LearnedSetupSource(candidate(SELECTED_CANDIDATE_ID), scorer, index)

    phase9_ref = neural_policy_ref(
        PHASE10_EVAL_MOVE_POLICY_ID, decision_mode=DECISION_MODE_GREEDY, dtype_name=EVAL_DTYPE
    )
    anchor_ref = neural_policy_ref(
        PHASE8_ANCHOR_CANDIDATE_ID, decision_mode=DECISION_MODE_GREEDY, dtype_name=EVAL_DTYPE
    )

    def opposing_ref(matchup: str):
        if matchup == MATCHUP_LEARNED_VS_NEUTRAL:
            return phase9_ref
        if matchup == NEURAL_OPPONENT_MATCHUP:
            return anchor_ref
        return policy_ref(EXTERNAL_OPPONENT_POLICY_IDS[matchup])

    problems: list = []
    counters = {
        "games": 0,
        "replayed_games": 0,
        "replayed_actions": 0,
        "seat_mismatches": 0,
        "setup_reconstruction_mismatches": 0,
        "illegal_setups": 0,
        "inventory_errors": 0,
        "stranded_sampled_setups": 0,
        "split_violations": 0,
        "landings_learned": 0,
        "landings_neutral": 0,
    }
    seats: dict = {}
    own_cache: dict = {}

    for unit in mine:
        with open(unit_path(unit), "rb") as stream:
            stored = pickle.load(stream)
        for row in stored["rows"]:
            counters["games"] += 1
            case = by_case[row["case_id"]]
            matchup = row["matchup"]
            game_index = int(row["game_index"])
            label = (
                f"{row['arm']}/{row['candidate_id'] or 'neutral_v1'}/{matchup}/"
                f"{row['case_id']}/g{game_index}"
            )

            # Seat reconciliation: rebuild the full intended specification and
            # require the recorded identifier to match it cryptographically.
            other_ref = opposing_ref(matchup)
            spec = build_final_spec(
                case,
                game_index,
                matchup,
                arm=row["arm"],
                candidate_id=row["candidate_id"],
                own_ref=phase9_ref,
                opponent_ref=other_ref,
            )
            if spec.match_id != row["match_id"]:
                counters["seat_mismatches"] += 1
                problems.append(f"{label}: match_id != rebuilt specification")
                continue
            if spec.setup_bank_version != row["setup_bank_version"]:
                counters["seat_mismatches"] += 1
                problems.append(f"{label}: cell token != rebuilt")
            frozen_seed = case_match_seed(row["case_id"], game_index, matchup)
            if spec.root_seed != row["root_seed"] or spec.root_seed != frozen_seed:
                counters["seat_mismatches"] += 1
                problems.append(f"{label}: match seed != frozen")
            if CASE_GAME_COLOR[game_index] != row["own_color"]:
                counters["seat_mismatches"] += 1
                problems.append(f"{label}: selector colour != frozen pairing")
            other_color = "blue" if row["own_color"] == "red" else "red"
            seats[(matchup, phase9_ref.token, "selector", row["own_color"])] = (
                seats.get((matchup, phase9_ref.token, "selector", row["own_color"]), 0) + 1
            )
            seats[(matchup, other_ref.token, "opposing", other_color)] = (
                seats.get((matchup, other_ref.token, "opposing", other_color), 0) + 1
            )

            # Own-side reconstruction from selector identity and seed alone.
            cache_key = (row["arm"], row["case_id"], row["own_color"])
            if cache_key not in own_cache:
                if row["arm"] == "learned":
                    own_cache[cache_key] = learned_own_side(source, case, row["own_color"])
                else:
                    own_cache[cache_key] = neutral_own_side(case, row["own_color"])
            draw = own_cache[cache_key]
            if draw.final_setup_fingerprint != row["own_fingerprint"]:
                counters["setup_reconstruction_mismatches"] += 1
                problems.append(f"{label}: reconstructed fingerprint != recorded")
            if draw.base_setup_id != row["own_base_setup_id"]:
                counters["setup_reconstruction_mismatches"] += 1
                problems.append(f"{label}: reconstructed base != recorded")
            if draw.branch != row["own_branch"]:
                counters["setup_reconstruction_mismatches"] += 1
                problems.append(f"{label}: reconstructed branch != recorded")
            landed = draw.final_setup_fingerprint in isolation
            if landed != bool(row["own_landed_in_phase9_set"]):
                counters["setup_reconstruction_mismatches"] += 1
                problems.append(f"{label}: landing flag != recomputed")
            if landed:
                counters["landings_" + row["arm"]] += 1

            # The engine-facing setups the game actually used must equal the
            # arrangements the case identity and the reconstructed draw imply.
            own = {row["own_color"]: draw}
            if matchup == MATCHUP_LEARNED_VS_NEUTRAL:
                own[other_color] = own_cache.setdefault(
                    (row["arm"], row["case_id"], other_color),
                    learned_own_side(source, case, other_color)
                    if row["arm"] == "learned"
                    else neutral_own_side(case, other_color),
                )
            else:
                # game_setups produces both colour rows; only this row's own
                # colour reaches the comparison below, so the other colour's
                # entry is a structural filler for the unused row.
                own[other_color] = draw
            setup_row = game_setups(case, matchup, own)[game_index]
            result = MatchResult.from_dict(row["result_row"])
            recorded_red = tuple(deserialize_setup(result.red_setup))
            recorded_blue = tuple(deserialize_setup(result.blue_setup))
            if recorded_red != tuple(setup_row["red_setup"]) or recorded_blue != tuple(
                setup_row["blue_setup"]
            ):
                counters["setup_reconstruction_mismatches"] += 1
                problems.append(f"{label}: played setups != case reconstruction")

            # Engine legality, inventory and mobility of the produced draw.
            try:
                validate_setup(tuple(draw.canonical), 0)
            except Exception as error:  # noqa: BLE001 -- an invalid setup is a finding
                counters["illegal_setups"] += 1
                counters["inventory_errors"] += 1
                problems.append(f"{label}: produced setup invalid: {error}")
            if not setup_has_initial_mobility(tuple(draw.canonical)):
                counters["stranded_sampled_setups"] += 1
                problems.append(f"{label}: produced setup is stranded")
            _, _, base_index = parse_base_setup_id(draw.base_setup_id)
            if split_for_base_index(base_index) != case.split:
                counters["split_violations"] += 1
                problems.append(f"{label}: base outside the {case.split} split")

            # Exhaustive move-legality replay of the stored action history.
            replay_problems = replay_stored_match(result)
            if replay_problems:
                problems.extend(f"{label}: {entry}" for entry in replay_problems)
            else:
                counters["replayed_games"] += 1
                counters["replayed_actions"] += len(result.action_history or ())

    shard = {
        "worker": args.worker,
        "workers": args.workers,
        "units": len(mine),
        "counters": counters,
        "seats": {"|".join(map(str, key)): value for key, value in sorted(seats.items())},
        "problems": problems[:200],
        "problem_count": len(problems),
    }
    write_stage(f"audit_shard_{args.worker}", shard)


def stage_audit(args) -> dict:
    """Fan the exhaustive audit across processes and merge the shards."""
    read_stage("games")
    log(f"launching {args.audit_workers} audit shard worker(s)")
    processes = []
    for worker in range(args.audit_workers):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--audit-shard",
            "--worker", str(worker),
            "--workers", str(args.audit_workers),
            "--chunk", str(args.chunk),
            "--cases", str(args.cases),
            "--device", args.device,
            "--torch-threads", str(args.torch_threads),
        ]
        processes.append((worker, subprocess.Popen(command, cwd=REPOSITORY_ROOT)))
    failed = [worker for worker, process in processes if process.wait() != 0]
    if failed:
        raise Agent7Error(f"audit shard worker(s) {failed} failed")

    merged = {
        "games": 0,
        "replayed_games": 0,
        "replayed_actions": 0,
        "seat_mismatches": 0,
        "setup_reconstruction_mismatches": 0,
        "illegal_setups": 0,
        "inventory_errors": 0,
        "stranded_sampled_setups": 0,
        "split_violations": 0,
        "landings_learned": 0,
        "landings_neutral": 0,
    }
    seats: dict = {}
    problems: list = []
    for worker in range(args.audit_workers):
        shard = read_stage(f"audit_shard_{worker}")
        for key, value in shard["counters"].items():
            merged[key] += int(value)
        for key, value in shard["seats"].items():
            seats[key] = seats.get(key, 0) + int(value)
        problems.extend(shard["problems"])
        if shard["problem_count"] != len(shard["problems"]):
            problems.append(
                f"shard {worker} truncated {shard['problem_count'] - len(shard['problems'])} "
                "further problems"
            )

    with open(WORK_DIRECTORY / "rows.pkl", "rb") as stream:
        rows = pickle.load(stream)
    if merged["games"] != len(rows):
        problems.append(f"audit covered {merged['games']} games, expected {len(rows)}")
    if merged["replayed_games"] != len(rows):
        problems.append(
            f"only {merged['replayed_games']} of {len(rows)} games replayed cleanly"
        )

    from stratego.evaluation.neural_worker import DECISION_MODE_GREEDY, neural_policy_ref
    from stratego.evaluation.phase10_validation import (
        EVAL_DTYPE,
        EXTERNAL_OPPONENT_POLICY_IDS,
        PHASE10_EVAL_MOVE_POLICY_ID,
        PHASE8_ANCHOR_CANDIDATE_ID,
    )
    from stratego.evaluation.registry import policy_ref

    phase9_token = neural_policy_ref(
        PHASE10_EVAL_MOVE_POLICY_ID, decision_mode=DECISION_MODE_GREEDY, dtype_name=EVAL_DTYPE
    ).token
    anchor_token = neural_policy_ref(
        PHASE8_ANCHOR_CANDIDATE_ID, decision_mode=DECISION_MODE_GREEDY, dtype_name=EVAL_DTYPE
    ).token
    aggregate: dict = {}
    for key, value in seats.items():
        token = key.split("|")[1]
        aggregate[token] = aggregate.get(token, 0) + value
    expected = expected_seat_counts(args.cases)
    expected_totals = {phase9_token: expected["phase9"], anchor_token: expected["each_external_opponent"]}
    for policy_id in EXTERNAL_OPPONENT_POLICY_IDS.values():
        expected_totals[policy_ref(policy_id).token] = expected["each_external_opponent"]
    for token, count in sorted(expected_totals.items()):
        if aggregate.get(token, 0) != count:
            problems.append(f"seat count for {token}: {aggregate.get(token, 0)} != {count}")
    if set(aggregate) != set(expected_totals):
        problems.append(f"unexpected seat tokens {sorted(set(aggregate) - set(expected_totals))}")
    if sum(aggregate.values()) != 2 * len(rows):
        problems.append("seat total does not cover both seats of every game")

    binding = _weights_binding_control(rows, args)
    for entry in binding:
        if entry["correct_owner_reproduces"] != entry["sampled_games"]:
            problems.append(
                f"{entry['matchup']}: recorded games do not replay under the bound owner"
            )
        if entry["swapped_owner_changes_the_game"] != entry["sampled_games"]:
            problems.append(
                f"{entry['matchup']}: swapping the checkpoint did not change the game"
            )

    payload = {
        "stage": "audit",
        "problems": problems,
        "audit_workers": args.audit_workers,
        "counters": merged,
        "aggregate_seat_counts": dict(sorted(aggregate.items())),
        "expected_seat_counts": dict(sorted(expected_totals.items())),
        "seats_audited": sum(aggregate.values()),
        "replay_note": (
            "every recorded game's stored action history was re-applied through a "
            "fresh reference engine: every action legal, terminal reason, winner, "
            "ply count and rebuilt replay digest all required to match the row"
        ),
        "weights_binding": binding,
        "scheduled_games_rerun": 0,
        "selection_changed": False,
        "environment": environment_record(),
    }
    write_stage("audit", payload)
    if problems:
        for problem in problems[:20]:
            log(f"  PROBLEM: {problem}")
        raise Agent7Error(f"{len(problems)} audit problem(s); STOP")
    log(
        f"  {merged['games']:,} games replayed action-by-action "
        f"({merged['replayed_actions']:,} actions), {payload['seats_audited']:,} seats "
        "reconciled, zero mismatches"
    )
    return payload


def _weights_binding_control(rows, args) -> list:
    """Replay recorded neural games with the bound owner, and the wrong one."""
    from stratego.evaluation.match_runner import ON_POLICY_ERROR_QUARANTINE, play_match
    from stratego.evaluation.neural_worker import (
        DECISION_MODE_GREEDY,
        InferenceOwner,
        LocalInferenceChannel,
        RemoteNeuralPolicy,
        neural_policy_ref,
    )
    from stratego.evaluation.phase10_final import build_final_spec, final_cases
    from stratego.evaluation.phase10_validation import (
        EVAL_DTYPE,
        FrozenSeedPolicy,
        PHASE10_EVAL_MOVE_POLICY_ID,
        PHASE8_ANCHOR_CANDIDATE_ID,
        game_setups,
        learned_own_side,
        neutral_own_side,
        single_game_bank,
    )
    from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from stratego.training.phase10_contract import MATCHUP_LEARNED_VS_NEUTRAL
    from stratego.training.phase10_seed import case_match_seed
    from stratego.training.phase10_selector import (
        LearnedSetupSource,
        candidate,
        load_library_index,
        load_scorer,
    )
    from stratego.training.phase10_soak import SELECTED_CANDIDATE_ID
    from stratego.evaluation.phase10_validation import NEURAL_OPPONENT_MATCHUP

    cases, _manifest = final_cases()
    by_case = {case.case_id: case for case in cases}
    phase9_ref = neural_policy_ref(
        PHASE10_EVAL_MOVE_POLICY_ID, decision_mode=DECISION_MODE_GREEDY, dtype_name=EVAL_DTYPE
    )
    anchor_ref = neural_policy_ref(
        PHASE8_ANCHOR_CANDIDATE_ID, decision_mode=DECISION_MODE_GREEDY, dtype_name=EVAL_DTYPE
    )

    def build_owner(path, name):
        return InferenceOwner(
            Path(path),
            decision_mode=DECISION_MODE_GREEDY,
            device=args.device,
            dtype=EVAL_DTYPE,
            expected_architecture_id=ARCHITECTURE_FAMILY,
            expected_configuration=candidate_config("C1"),
            name=name,
        )

    scorer, index = load_scorer(), load_library_index()
    source = LearnedSetupSource(candidate(SELECTED_CANDIDATE_ID), scorer, index)
    phase9_owner = build_owner(EXPORT_PATH, "a7audit_phase9")
    anchor_owner = build_owner(ANCHOR_EXPORT_PATH, "a7audit_anchor")
    results = []
    try:
        for matchup in (MATCHUP_LEARNED_VS_NEUTRAL, NEURAL_OPPONENT_MATCHUP):
            direct = matchup == MATCHUP_LEARNED_VS_NEUTRAL
            other_ref = phase9_ref if direct else anchor_ref
            bound, swapped = (
                (phase9_owner, anchor_owner) if direct else (anchor_owner, phase9_owner)
            )
            sample = [
                row for row in rows if row["matchup"] == matchup and row["arm"] == "learned"
            ][: args.audit_sample]
            reproduced = changed = 0
            for row in sample:
                case = by_case[row["case_id"]]
                own = {
                    color: learned_own_side(source, case, color) for color in ("red", "blue")
                }
                setup_row = game_setups(case, matchup, own)[int(row["game_index"])]
                spec = build_final_spec(
                    case,
                    int(row["game_index"]),
                    matchup,
                    arm="learned",
                    candidate_id=SELECTED_CANDIDATE_ID,
                    own_ref=phase9_ref,
                    opponent_ref=other_ref,
                )
                bank = single_game_bank(spec, setup_row["red_setup"], setup_row["blue_setup"])
                seed = case_match_seed(case.case_id, int(row["game_index"]), matchup)
                for owner, counter in ((bound, "reproduced"), (swapped, "changed")):
                    under_test = RemoteNeuralPolicy(
                        other_ref, LocalInferenceChannel(owner), decision_mode=DECISION_MODE_GREEDY
                    )
                    policies = (
                        {phase9_ref.token: under_test}
                        if direct
                        else {
                            phase9_ref.token: RemoteNeuralPolicy(
                                phase9_ref,
                                LocalInferenceChannel(phase9_owner),
                                decision_mode=DECISION_MODE_GREEDY,
                            ),
                            other_ref.token: FrozenSeedPolicy(under_test, seed),
                        }
                    )
                    replayed = play_match(
                        spec,
                        bank=bank,
                        policies=policies,
                        record_actions=False,
                        on_policy_error=ON_POLICY_ERROR_QUARANTINE,
                    )
                    matches = replayed.replay_digest == row["replay_digest"]
                    if counter == "reproduced":
                        reproduced += int(matches)
                    else:
                        changed += int(not matches)
            results.append(
                {
                    "matchup": matchup,
                    "seat_under_test": other_ref.token,
                    "bound_checkpoint": "phase9" if direct else "phase8_anchor",
                    "swapped_checkpoint": "phase8_anchor" if direct else "phase9",
                    "sampled_games": len(sample),
                    "correct_owner_reproduces": reproduced,
                    "swapped_owner_changes_the_game": changed,
                }
            )
    finally:
        phase9_owner.close()
        anchor_owner.close()
    return results


# ---------------------------------------------------------------------------
# stage: reproduce — Gate G evidence across worker orders and processes
# ---------------------------------------------------------------------------


def stage_reproduce_shard(args) -> None:
    """One reconstruction worker: re-derive every selector draw in its slice.

    The slice walk depends on the worker topology (`--workers/--worker`) and
    the direction flag, so two topologies enumerate the same draws in
    different orders from different processes; the derived values must not
    care.
    """
    from stratego.evaluation.phase10_final import final_cases
    from stratego.training.phase10_selector import (
        LearnedSetupSource,
        SelectorRequest,
        candidate,
        load_library_index,
        load_scorer,
        neutral_baseline_draw,
    )
    from stratego.training.phase10_soak import SELECTED_CANDIDATE_ID

    cases, _manifest = final_cases()
    cases = cases[: args.cases]
    source = LearnedSetupSource(
        candidate(SELECTED_CANDIDATE_ID), load_scorer(), load_library_index()
    )
    jobs = [
        (case, color, arm)
        for case in cases
        for color in ("red", "blue")
        for arm in ("learned", "neutral")
    ]
    if args.reverse_order:
        jobs = list(reversed(jobs))
    mine = [job for position, job in enumerate(jobs) if position % args.workers == args.worker]

    derived = {}
    for case, color, arm in mine:
        seed = int(case.selector_seeds[color])
        if arm == "learned":
            draw = source.draw(
                SelectorRequest(split=case.split, color=color, selector_seed=seed)
            )
            provenance = {
                "base_setup_id": draw.base_setup_id,
                "family_id": draw.family_id,
                "branch": draw.branch,
                "final_setup_fingerprint": draw.final_setup_fingerprint,
                "canonical": list(draw.setup.canonical),
                "reflection_applied": bool(draw.setup.provenance["reflection_applied"]),
                "perturbation_applied": bool(draw.setup.provenance["perturbation_applied"]),
                "perturbation_swap_count": draw.setup.provenance["perturbation_swap_count"],
                "selector_provenance": draw.selector_provenance(),
                "setup_provenance": dict(draw.setup.provenance),
            }
        else:
            sampled = neutral_baseline_draw(case.split, seed)
            provenance = {
                "base_setup_id": sampled.base_setup_id,
                "family_id": sampled.family_id,
                "branch": None,
                "final_setup_fingerprint": sampled.provenance["final_setup_fingerprint"],
                "canonical": list(sampled.canonical),
                "reflection_applied": bool(sampled.provenance["reflection_applied"]),
                "perturbation_applied": bool(sampled.provenance["perturbation_applied"]),
                "perturbation_swap_count": sampled.provenance["perturbation_swap_count"],
                "setup_provenance": dict(sampled.provenance),
                "frozen_provenance": dict(case.neutral_provenance[color]),
            }
        derived[f"{arm}|{case.case_id}|{color}"] = provenance

    label = "r" if args.reverse_order else "f"
    write_stage(f"reproduce_shard_{label}{args.worker}", {"draws": derived})


def stage_reproduce(args) -> dict:
    """Deterministic reconstruction of every final selector draw, twice.

    Two independent reconstructions of all 2,048 `(arm, case, colour)` draws
    run in fresh processes under different worker topologies and opposite
    enumeration orders; both must agree with each other bit for bit, with the
    frozen bank provenance, and with what the recorded games actually played.
    One recorded work unit is additionally deleted and rebuilt game-for-game
    by a fresh single worker.
    """
    from stratego.evaluation.phase10_validation import rows_digest
    from stratego.training.phase10_selector import Phase10SelectorError, SelectorRequest

    read_stage("games")
    problems: list = []

    topologies = {"forward": (5, False), "reverse": (3, True)}
    for label, (workers, reverse) in topologies.items():
        log(f"reconstructing all draws: {label} topology, {workers} workers")
        processes = []
        for worker in range(workers):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--reproduce-shard",
                "--worker", str(worker),
                "--workers", str(workers),
                "--cases", str(args.cases),
                "--chunk", str(args.chunk),
            ]
            if reverse:
                command.append("--reverse-order")
            processes.append((worker, subprocess.Popen(command, cwd=REPOSITORY_ROOT)))
        failed = [worker for worker, process in processes if process.wait() != 0]
        if failed:
            raise Agent7Error(f"reconstruction worker(s) {failed} failed ({label})")

    reconstructions = {}
    for label, (workers, reverse) in topologies.items():
        merged = {}
        prefix = "r" if reverse else "f"
        for worker in range(workers):
            merged.update(read_stage(f"reproduce_shard_{prefix}{worker}")["draws"])
        reconstructions[label] = merged

    forward = reconstructions["forward"]
    reverse = reconstructions["reverse"]
    if set(forward) != set(reverse):
        problems.append("the two topologies enumerated different draw sets")
    disagreements = [key for key in forward if forward[key] != reverse.get(key)]
    if disagreements:
        problems.append(
            f"{len(disagreements)} draw(s) differ across topologies: {disagreements[:5]}"
        )

    with open(WORK_DIRECTORY / "rows.pkl", "rb") as stream:
        rows = pickle.load(stream)
    played_mismatches = 0
    for row in rows:
        key = f"{row['arm']}|{row['case_id']}|{row['own_color']}"
        derived = forward.get(key)
        if derived is None:
            played_mismatches += 1
            continue
        if (
            derived["final_setup_fingerprint"] != row["own_fingerprint"]
            or derived["base_setup_id"] != row["own_base_setup_id"]
            or derived["branch"] != row["own_branch"]
        ):
            played_mismatches += 1
    if played_mismatches:
        problems.append(
            f"{played_mismatches} recorded game(s) disagree with the reconstruction"
        )

    neutral_frozen_mismatches = sum(
        1
        for key, value in forward.items()
        if key.startswith("neutral|")
        and value["setup_provenance"] != value["frozen_provenance"]
    )
    if neutral_frozen_mismatches:
        problems.append(
            f"{neutral_frozen_mismatches} neutral draw(s) disagree with the frozen bank"
        )

    log("replaying one recorded work unit in a fresh single-worker process")
    units = work_units(args.cases, args.chunk)
    chosen = next(
        unit
        for unit in units
        if unit["arm"] == "learned"
        and unit["matchup"] == args.reproduce_matchup
        and unit["start"] == args.reproduce_start
    )
    path = unit_path(chosen)
    with open(path, "rb") as stream:
        original = pickle.load(stream)
    strip = lambda rows_: [
        {key: value for key, value in row.items() if key != "result_row"} for row in rows_
    ]
    before = rows_digest(strip(original["rows"]))
    backup = path.with_suffix(".replay-backup")
    path.replace(backup)
    try:
        subprocess.run(
            [
                sys.executable, str(Path(__file__).resolve()),
                "--cell-worker",
                "--worker", "0", "--workers", "1",
                "--chunk", str(args.chunk), "--cases", str(args.cases),
                "--device", args.device, "--torch-threads", str(args.torch_threads),
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        )
        with open(path, "rb") as stream:
            replayed = pickle.load(stream)
    finally:
        if not path.exists():
            backup.replace(path)
        elif backup.exists():
            backup.unlink()
    after = rows_digest(strip(replayed["rows"]))

    # Every determined field must come back identical. `result_row` embeds
    # `wall_clock_seconds`, a timing measurement rather than a determination,
    # so the stored result is compared through the accepted `comparable()`
    # form — which keeps both setups, seeds, outcome, plies, replay digest
    # and the complete action history, and drops only timing/file references.
    from stratego.evaluation.match_runner import MatchResult

    def determined(rows_):
        return [
            {key: value for key, value in row.items() if key != "result_row"}
            | {"result": MatchResult.from_dict(row["result_row"]).comparable()}
            for row in rows_
        ]

    unit_identical = determined(original["rows"]) == determined(replayed["rows"])
    if before != after or not unit_identical:
        problems.append("a replayed work unit did not reproduce its recorded games")

    log("running the hidden-input positive control")
    injection = {"attempts": [], "all_raised": True}
    for field, value in (
        ("opponent_family", "F07"),
        ("opponent_base_id", "setup_base_v1|F07|03333"),
        ("outcome", 1.0),
        ("path", "/Volumes/x"),
        ("final_setup_fingerprint", "deadbeef"),
    ):
        payload = {"split": "test", "color": "red", "selector_seed": 7, field: value}
        try:
            SelectorRequest.from_payload(payload)
        except Phase10SelectorError:
            injection["attempts"].append({"field": field, "raised": True})
        else:
            injection["attempts"].append({"field": field, "raised": False})
            injection["all_raised"] = False
            problems.append(f"injected selector field {field!r} was not rejected")

    reproducibility_report = {
        "same_base": not disagreements and not played_mismatches,
        "same_reflection": not disagreements and not neutral_frozen_mismatches,
        "same_perturbation": not disagreements and not neutral_frozen_mismatches,
        "same_final_fingerprint": not disagreements
        and not played_mismatches
        and not neutral_frozen_mismatches,
        "worker_order_independent": not disagreements,
        "process_restart_independent": before == after and bool(unit_identical),
    }

    payload = {
        "stage": "reproduce",
        "problems": problems,
        "draws_reconstructed": len(forward),
        "topologies": {
            "forward": {"workers": 5, "order": "ascending"},
            "reverse": {"workers": 3, "order": "descending"},
        },
        "cross_topology_disagreements": len(disagreements),
        "played_game_mismatches": played_mismatches,
        "neutral_frozen_mismatches": neutral_frozen_mismatches,
        "unit_replay": {
            "unit": dict(chosen),
            "games": len(original["rows"]),
            "recorded_workers": 12,
            "replay_workers": 1,
            "fresh_process": True,
            "digest_before": before,
            "digest_after": after,
            "every_field_identical": bool(unit_identical),
        },
        "hidden_input_control": injection,
        "reproducibility_report": reproducibility_report,
        "environment": environment_record(),
    }
    write_stage("reproduce", payload)
    if problems:
        for problem in problems[:20]:
            log(f"  PROBLEM: {problem}")
        raise Agent7Error(f"{len(problems)} reproducibility problem(s); STOP")
    log(
        f"  {len(forward)} draws reconstructed identically across topologies; unit "
        f"replay identical; injection control raised {len(injection['attempts'])}/"
        f"{len(injection['attempts'])}"
    )
    return payload


# ---------------------------------------------------------------------------
# stage: gates — the eight hard gates, recomputed from primitives
# ---------------------------------------------------------------------------


def stage_gates(args) -> dict:
    from stratego.evaluation.phase10_banks import phase9_isolation_set
    from stratego.evaluation.phase10_final import final_cases
    from stratego.evaluation.phase10_validation import (
        case_game_pairs,
        color_split,
        counts_from_rows,
        family_split,
        landing_counts,
        length_summary,
        rows_digest,
        safety_counters,
        terminal_reasons,
    )
    from stratego.training.phase10_acceptance import (
        MatchupOutcomes,
        classify,
        evaluate_acceptance,
    )
    from stratego.training.phase10_contract import (
        CANDIDATE_IDS,
        HARD_GATE_IDS,
        MATCHUP_LEARNED_VS_NEUTRAL,
        MATCHUP_TOKENS,
    )
    from stratego.training.phase10_selector import (
        build_distribution,
        candidate,
        evaluate_diversity,
        load_library_index,
        load_scorer,
    )
    from stratego.training.phase10_soak import SELECTED_CANDIDATE_ID

    games = read_stage("games")
    audit = read_stage("audit")
    reproduce = read_stage("reproduce")
    banks = read_stage("banks")
    with open(WORK_DIRECTORY / "rows.pkl", "rb") as stream:
        rows = pickle.load(stream)

    cases, _manifest = final_cases()
    cases = cases[: args.cases]
    case_ids = tuple(case.case_id for case in cases)
    isolation, _meta = phase9_isolation_set()

    def cell_rows(arm, candidate_id, matchup):
        return [
            row
            for row in rows
            if row["arm"] == arm
            and row["candidate_id"] == candidate_id
            and row["matchup"] == matchup
        ]

    def by_case(entries):
        indexed: dict = {}
        for row in entries:
            indexed.setdefault(row["case_id"], {})[row["game_index"]] = row
        return indexed

    log("assembling the primitive outcomes of every matchup")
    unscored = [row for row in rows if row["score"] is None]
    if unscored:
        raise Agent7Error(f"{len(unscored)} final-test game(s) recorded no score")

    cells: dict = {}
    outcomes: dict = {}
    neutral_pairs: dict = {}
    for matchup in MATCHUP_TOKENS:
        if matchup == MATCHUP_LEARNED_VS_NEUTRAL:
            continue
        entries = cell_rows("neutral", None, matchup)
        neutral_pairs[matchup] = case_game_pairs(by_case(entries), case_ids)
        cells[("neutral", matchup)] = {
            "rows": entries,
            "counts": counts_from_rows(entries),
            "color_split": color_split(entries),
            "family_split": family_split(entries),
            "terminal_reasons": terminal_reasons(entries),
            "lengths": length_summary(entries),
            "safety": safety_counters(entries),
            "landings": landing_counts(entries, isolation),
            "digest": rows_digest(entries),
        }
    for matchup in MATCHUP_TOKENS:
        entries = cell_rows("learned", SELECTED_CANDIDATE_ID, matchup)
        learned_pairs = case_game_pairs(by_case(entries), case_ids)
        cells[("learned", matchup)] = {
            "rows": entries,
            "counts": counts_from_rows(entries),
            "color_split": color_split(entries),
            "family_split": family_split(entries),
            "terminal_reasons": terminal_reasons(entries),
            "lengths": length_summary(entries),
            "safety": safety_counters(entries),
            "landings": landing_counts(entries, isolation),
            "digest": rows_digest(entries),
        }
        outcomes[matchup] = MatchupOutcomes(
            token=matchup,
            case_ids=case_ids,
            learned_games=learned_pairs,
            neutral_games=(
                None if matchup == MATCHUP_LEARNED_VS_NEUTRAL else neutral_pairs[matchup]
            ),
        )

    log("recomputing the exact selector distributions (36 cells) for Gate E")
    scorer = load_scorer()
    index = load_library_index()
    diversity_rows = []
    non_finite_distribution_cells = 0
    worst = {
        "min_normalized_family_entropy": float("inf"),
        "min_effective_families": float("inf"),
        "min_family_probability": float("inf"),
        "max_family_probability": float("-inf"),
        "min_within_family_normalized_base_entropy": float("inf"),
        "max_conditional_base_probability": float("-inf"),
    }
    production_cells = {}
    for candidate_id in CANDIDATE_IDS:
        selector = candidate(candidate_id)
        for color in ("red", "blue"):
            for split in ("train", "validation", "test"):
                distribution = build_distribution(selector, color, split, scorer, index)
                finiteness = distribution.finiteness()
                if not finiteness.get("all_finite", False):
                    non_finite_distribution_cells += 1
                metrics = distribution.diversity()
                verdict = evaluate_diversity(metrics)
                diversity_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "color": color,
                        "split": split,
                        "normalized_family_entropy": metrics["normalized_family_entropy"],
                        "effective_families": metrics["effective_families"],
                        "min_family_probability": metrics["min_family_probability"],
                        "max_family_probability": metrics["max_family_probability"],
                        "min_within_family_normalized_base_entropy": metrics[
                            "min_within_family_normalized_base_entropy"
                        ],
                        "max_conditional_base_probability": metrics[
                            "max_conditional_base_probability"
                        ],
                        "effective_base_diversity": metrics["effective_base_diversity"],
                        "all_thresholds_pass": verdict["all_pass"],
                        "probability_vector_digest": distribution.probability_vector_digest(),
                        "mixture_exact": distribution.mixture_is_exact(),
                        "all_finite": bool(finiteness.get("all_finite", False)),
                    }
                )
                worst["min_normalized_family_entropy"] = min(
                    worst["min_normalized_family_entropy"],
                    metrics["normalized_family_entropy"],
                )
                worst["min_effective_families"] = min(
                    worst["min_effective_families"], metrics["effective_families"]
                )
                worst["min_family_probability"] = min(
                    worst["min_family_probability"], metrics["min_family_probability"]
                )
                worst["max_family_probability"] = max(
                    worst["max_family_probability"], metrics["max_family_probability"]
                )
                worst["min_within_family_normalized_base_entropy"] = min(
                    worst["min_within_family_normalized_base_entropy"],
                    metrics["min_within_family_normalized_base_entropy"],
                )
                worst["max_conditional_base_probability"] = max(
                    worst["max_conditional_base_probability"],
                    metrics["max_conditional_base_probability"],
                )
                if candidate_id == SELECTED_CANDIDATE_ID:
                    production_cells[f"{color}/{split}"] = {
                        "metrics": {
                            key: metrics[key]
                            for key in (
                                "normalized_family_entropy",
                                "effective_families",
                                "min_family_probability",
                                "max_family_probability",
                                "min_within_family_normalized_base_entropy",
                                "max_conditional_base_probability",
                                "effective_base_diversity",
                            )
                        },
                        "all_thresholds_pass": verdict["all_pass"],
                        "digest": distribution.probability_vector_digest(),
                    }
    diversity_report = dict(worst)
    diversity_report["scope"] = "worst case over every candidate, colour and split"
    diversity_report["cells"] = len(diversity_rows)
    diversity_report["production_candidate_cells"] = production_cells

    log("assembling the Gate F correctness counters")
    total_safety = {
        "policy_errors": 0,
        "illegal_actions": 0,
        "engine_rejections": 0,
        "policy_exceptions": 0,
        "contract_violations": 0,
        "non_finite_scores": 0,
        "illegal_setups": 0,
        "unscored_games": 0,
    }
    for cell in cells.values():
        for key in total_safety:
            total_safety[key] += int(cell["safety"].get(key, 0))
    audit_counters = audit["counters"]
    test_audit = banks["test_audit"]
    correctness_report = {
        "illegal_setups": total_safety["illegal_setups"]
        + audit_counters["illegal_setups"]
        + len(test_audit["engine_failures"]),
        "inventory_errors": audit_counters["inventory_errors"],
        "stranded_sampled_setups": audit_counters["stranded_sampled_setups"],
        "split_leakage": audit_counters["split_violations"]
        + len(test_audit["split_violations"]),
        "provenance_mismatch": audit_counters["setup_reconstruction_mismatches"]
        + reproduce["played_game_mismatches"]
        + reproduce["neutral_frozen_mismatches"]
        + len(test_audit["provenance_mismatches"]),
        "hidden_opponent_selector_inputs": sum(
            1 for attempt in reproduce["hidden_input_control"]["attempts"]
            if not attempt["raised"]
        ),
        "illegal_neural_moves": total_safety["illegal_actions"]
        + total_safety["engine_rejections"]
        + (audit_counters["games"] - audit_counters["replayed_games"]),
        "non_finite_selector_outputs": non_finite_distribution_cells
        + total_safety["non_finite_scores"],
        "inference_failures": int(games["inference"]["failures_returned"]),
        "supporting": {
            "total_safety": total_safety,
            "audit_counters": {
                key: value for key, value in audit_counters.items()
                if not key.startswith("landings")
            },
            "policy_errors": total_safety["policy_errors"],
            "policy_exceptions": total_safety["policy_exceptions"],
            "contract_violations": total_safety["contract_violations"],
            "unscored_games": total_safety["unscored_games"],
        },
    }

    log("re-hashing the Phase 9 checkpoint after all final games")
    phase9_after = checkpoint_identity(CHECKPOINT_PATH)
    preservation_report = {
        "checkpoint_sha256": phase9_after["sha256"],
        "model_state_digest": phase9_after["model_state_digest"],
        "parameters": phase9_after["parameters"],
        "all_parameters_finite": phase9_after["all_parameters_finite"],
        "c1_optimizer_steps": 0,
        "optimizer_note": (
            "no optimizer object, no backward pass and no parameter write exists "
            "anywhere in the Agent 7 harness or the final evaluation path"
        ),
    }

    reproducibility_report = dict(reproduce["reproducibility_report"])

    log("recomputing every hard gate from primitives")
    acceptance = evaluate_acceptance(
        outcomes,
        bank="test",
        diversity_report=diversity_report,
        correctness_report=correctness_report,
        reproducibility_report=reproducibility_report,
        preservation_report=preservation_report,
    )

    log("cross-checking the gate arithmetic independently")
    independent = _independent_gate_check(rows, case_ids, acceptance)

    recomputed_classification = classify(acceptance["gates"])
    if recomputed_classification != acceptance["classification"]:
        raise Agent7Error("the classification does not recompute from its own gate rows")

    landing_diagnostic = {
        "granularity": "candidate x arm x matchup x bank",
        "use": "report_only",
        "gate": False,
        "isolation_set_size": len(isolation),
        "rows": [
            {
                "candidate_id": SELECTED_CANDIDATE_ID if arm == "learned" else "neutral_v1",
                "arm": arm,
                "matchup": matchup,
                "bank": "phase10_test_bank_v1",
                "games": cells[(arm, matchup)]["landings"]["games"],
                "landings": cells[(arm, matchup)]["landings"]["landings"],
                "landing_rate": cells[(arm, matchup)]["landings"]["rate"],
            }
            for arm, matchup in sorted(cells)
        ],
        "statement": (
            "recorded and never read: no retry, no gate, no score and no "
            "rejection sampling consulted these counts; rejecting a learned "
            "draw at evaluation time would distort the frozen mixed distribution"
        ),
    }

    payload = {
        "stage": "gates",
        "bank": "test",
        "case_count": len(cases),
        "matchups": acceptance["matchups"],
        "gates": acceptance["gates"],
        "hard_gates_all_pass": acceptance["hard_gates_all_pass"],
        "gates_true": acceptance["gates_true"],
        "gates_total": acceptance["gates_total"],
        "classification": acceptance["classification"],
        "classification_recomputed_from_gate_rows": recomputed_classification,
        "hard_gate_ids": list(HARD_GATE_IDS),
        "diversity_report": {
            key: value for key, value in diversity_report.items() if key != "cells"
        },
        "diversity_rows": diversity_rows,
        "correctness_report": correctness_report,
        "preservation_report": preservation_report,
        "reproducibility_report": reproducibility_report,
        "independent_check": independent,
        "landing_diagnostic": landing_diagnostic,
        "cells": {
            f"{arm}/{matchup}": {
                key: value for key, value in cell.items() if key != "rows"
            }
            for (arm, matchup), cell in sorted(cells.items())
        },
        "environment": environment_record(),
    }
    write_stage("gates", payload)
    log(
        f"  gates {payload['gates_true']}/{payload['gates_total']} pass; "
        f"classification {payload['classification']}"
    )
    return payload


def _independent_gate_check(rows, case_ids, acceptance) -> dict:
    """Recompute the headline gate quantities without the acceptance helpers.

    Every EWR and delta is rebuilt from the raw stored rows by direct
    arithmetic; every interval is rebuilt by calling the project bootstrap
    directly on independently assembled per-case values under the same frozen
    stream seeds. Agreement is required to the last bit for the point
    estimates and for the interval bounds.
    """
    from stratego.evaluation.statistics import bootstrap_interval
    from stratego.training.phase10_contract import (
        BOOTSTRAP_CONFIDENCE,
        BOOTSTRAP_REPLICATES,
        GATE_B,
        MATCHUP_LEARNED_VS_NEUTRAL,
        MATCHUP_PHASE8_ANCHOR,
        MATCHUP_STRATEGIC,
        MATCHUP_TACTICAL,
        MATCHUP_TOKENS,
    )
    from stratego.training.phase10_seed import bootstrap_stream_seed

    def case_means(arm, matchup):
        scores: dict = {}
        for row in rows:
            if row["arm"] != arm or row["matchup"] != matchup:
                continue
            scores.setdefault(row["case_id"], {})[row["game_index"]] = float(row["score"])
        return [
            (scores[case_id][0] + scores[case_id][1]) / 2.0 for case_id in case_ids
        ]

    checks = {}
    matchups = acceptance["matchups"]
    mismatches: list = []
    for matchup in MATCHUP_TOKENS:
        learned = case_means("learned", matchup)
        summary = matchups[matchup]
        ewr = sum(learned) / len(learned)
        entry = {"learned_ewr_recomputed": ewr, "matches": ewr == summary["learned_ewr"]}
        interval = bootstrap_interval(
            learned,
            resamples=BOOTSTRAP_REPLICATES,
            seed=bootstrap_stream_seed("test", f"{matchup}:learned"),
            confidence=BOOTSTRAP_CONFIDENCE,
            resampling_unit="phase10_logical_case",
        )
        entry["learned_interval_matches"] = (
            interval.lower == summary["learned_interval"]["lower"]
            and interval.upper == summary["learned_interval"]["upper"]
        )
        if matchup != MATCHUP_LEARNED_VS_NEUTRAL:
            neutral = case_means("neutral", matchup)
            differences = [a - b for a, b in zip(learned, neutral)]
            delta = sum(differences) / len(differences)
            entry["delta_recomputed"] = delta
            entry["delta_matches"] = delta == summary["delta"]
            delta_interval = bootstrap_interval(
                differences,
                resamples=BOOTSTRAP_REPLICATES,
                seed=bootstrap_stream_seed("test", f"{matchup}:delta"),
                confidence=BOOTSTRAP_CONFIDENCE,
                resampling_unit="phase10_logical_case",
            )
            entry["delta_interval_matches"] = (
                delta_interval.lower == summary["delta_interval"]["lower"]
                and delta_interval.upper == summary["delta_interval"]["upper"]
            )
        checks[matchup] = entry
        for key, value in entry.items():
            if key.endswith("matches") and not value:
                mismatches.append(f"{matchup}:{key}")

    weights = GATE_B["league_weights"]
    tokens = {
        "delta_strategic": MATCHUP_STRATEGIC,
        "delta_tactical": MATCHUP_TACTICAL,
        "delta_phase8_anchor": MATCHUP_PHASE8_ANCHOR,
    }
    per_matchup = {
        name: [
            a - b
            for a, b in zip(case_means("learned", token), case_means("neutral", token))
        ]
        for name, token in tokens.items()
    }
    league = [
        sum(weights[name] * per_matchup[name][index] for name in tokens)
        for index in range(len(case_ids))
    ]
    delta_l = sum(league) / len(league)
    league_interval = bootstrap_interval(
        league,
        resamples=BOOTSTRAP_REPLICATES,
        seed=bootstrap_stream_seed("test", "league:delta_l"),
        confidence=BOOTSTRAP_CONFIDENCE,
        resampling_unit="phase10_logical_case",
    )
    gate_b = acceptance["gates"]["B"]
    checks["league"] = {
        "delta_l_recomputed": delta_l,
        "delta_l_matches": abs(delta_l - gate_b["delta_l"]) < 1e-15,
        "interval_matches": (
            league_interval.lower == gate_b["interval"]["lower"]
            and league_interval.upper == gate_b["interval"]["upper"]
        ),
    }
    if not checks["league"]["delta_l_matches"]:
        mismatches.append("league:delta_l")
    if not checks["league"]["interval_matches"]:
        mismatches.append("league:interval")

    if mismatches:
        raise Agent7Error(f"independent gate check disagrees: {mismatches}")
    return {"checks": checks, "mismatches": mismatches, "all_agree": not mismatches}


# ---------------------------------------------------------------------------
# stage: artifacts — acceptance JSON, strength CSV, diversity CSV
# ---------------------------------------------------------------------------

#: The 28 completion gates of the Agent 7 instruction, in its order.
COMPLETION_GATES = (
    "agents1_6_pass",
    "administrative_freeze_verified",
    "phase9_identity_verified",
    "phase7_identity_verified",
    "phase10_contracts_verified",
    "utility_and_selector_digests_verified",
    "phase10_system_identity_verified",
    "validation_bank_rebuild_verified",
    "test_bank_rebuild_verified",
    "test_bank_structural_audit_pass",
    "pre_agent7_test_outcome_access_zero",
    "outcome_corpus_train_only_verified",
    "candidate_count_6_verified",
    "selection_validation_only_verified",
    "phase9_checkpoint_unchanged_before_eval",
    "gate_a_recomputed",
    "gate_b_recomputed",
    "gate_c_recomputed",
    "gate_d_recomputed",
    "gate_e_recomputed",
    "gate_f_recomputed",
    "gate_g_recomputed",
    "gate_h_recomputed",
    "final_setup_replay_audit_pass",
    "illegal_actions_zero",
    "nonfinite_zero",
    "opponent_hidden_selector_inputs_zero",
    "phase9_checkpoint_unchanged_after_eval",
    "classification_recomputes_from_gate_rows",
    "full_suite_green",
)


def _completion_gates(verify, banks, games, audit, reproduce, gates, suite) -> dict:
    correctness = gates["correctness_report"]
    return {
        "agents1_6_pass": all(
            entry["status"] == "PASS" and not entry["false_gates_recomputed"]
            for entry in verify["prior_agents"].values()
        ),
        "administrative_freeze_verified": bool(
            verify["administrative_freeze"]["commit_matches"]
        )
        and not verify["administrative_freeze"]["tracked_modifications"],
        "phase9_identity_verified": verify["phase9_checkpoint"]["sha256"]
        == ACCEPTED_PHASE9_SHA256
        and verify["phase9_checkpoint"]["model_state_digest"] == ACCEPTED_PHASE9_STATE_DIGEST
        and verify["phase9_checkpoint"]["parameters"] == ACCEPTED_PHASE9_PARAMETERS
        and bool(verify["phase9_checkpoint"]["all_parameters_finite"]),
        "phase7_identity_verified": verify["library"]["splits"]
        == {"train": 6400, "validation": 800, "test": 800},
        "phase10_contracts_verified": verify["contract_bundle_digest"]
        == ACCEPTED_CONTRACT_BUNDLE_DIGEST,
        "utility_and_selector_digests_verified": verify["utility"]["file_sha256"]
        == ACCEPTED_UTILITY_FILE_SHA256
        and verify["selector_config"]["artifact_sha256"] == ACCEPTED_CONFIG_SHA256
        and verify["production_train_digests"] == ACCEPTED_TRAIN_DIGESTS,
        "phase10_system_identity_verified": verify["phase10_system_v1"][
            "all_filling_rules_pass"
        ]
        and verify["phase10_system_v1"]["filled_instance_digest"] == ACCEPTED_SYSTEM_DIGEST
        and verify["phase10_system_v1"]["frozen_template_digest"]
        == verify["phase10_system_v1"]["frozen_template_pinned"],
        "validation_bank_rebuild_verified": banks["validation_audit"]["bank_digest"]
        == ACCEPTED_VALIDATION_BANK_DIGEST
        and banks["validation_audit"]["all_pass"],
        "test_bank_rebuild_verified": banks["test_audit"]["bank_digest"]
        == ACCEPTED_TEST_BANK_DIGEST,
        "test_bank_structural_audit_pass": banks["test_audit"]["all_pass"]
        and banks["cross_bank_isolation"]["zero_overlap"],
        "pre_agent7_test_outcome_access_zero": verify["pre_agent7_test_bank_access"][
            "all_entries_structural"
        ]
        and verify["pre_agent7_test_bank_access"]["prior_outcome_evaluations"] == 0,
        "outcome_corpus_train_only_verified": verify["corpus"]["seal_all_pass"]
        and verify["corpus"]["content_digest"] == ACCEPTED_CORPUS_CONTENT_DIGEST
        and verify["corpus"]["committed_games"] == 16_384,
        "candidate_count_6_verified": len(verify["distribution_digests_recomputed"]) == 6,
        "selection_validation_only_verified": verify["pre_agent7_test_bank_access"][
            "agent5_test_bank_outcome_access"
        ]
        == 0,
        "phase9_checkpoint_unchanged_before_eval": games["export"]["source_sha256"]
        == ACCEPTED_PHASE9_SHA256,
        "gate_a_recomputed": "A" in gates["gates"],
        "gate_b_recomputed": "B" in gates["gates"],
        "gate_c_recomputed": "C" in gates["gates"],
        "gate_d_recomputed": "D" in gates["gates"],
        "gate_e_recomputed": "E" in gates["gates"],
        "gate_f_recomputed": "F" in gates["gates"],
        "gate_g_recomputed": "G" in gates["gates"],
        "gate_h_recomputed": "H" in gates["gates"],
        "final_setup_replay_audit_pass": not audit["problems"]
        and audit["counters"]["replayed_games"] == audit["counters"]["games"]
        and audit["counters"]["seat_mismatches"] == 0
        and audit["counters"]["setup_reconstruction_mismatches"] == 0,
        "illegal_actions_zero": correctness["illegal_neural_moves"] == 0,
        "nonfinite_zero": correctness["non_finite_selector_outputs"] == 0,
        "opponent_hidden_selector_inputs_zero": correctness[
            "hidden_opponent_selector_inputs"
        ]
        == 0,
        "phase9_checkpoint_unchanged_after_eval": gates["preservation_report"][
            "checkpoint_sha256"
        ]
        == ACCEPTED_PHASE9_SHA256
        and gates["preservation_report"]["model_state_digest"]
        == ACCEPTED_PHASE9_STATE_DIGEST
        and gates["preservation_report"]["parameters"] == ACCEPTED_PHASE9_PARAMETERS,
        "classification_recomputes_from_gate_rows": gates["classification"]
        == gates["classification_recomputed_from_gate_rows"],
        "full_suite_green": suite is not None
        and suite.get("returncode") == 0
        and suite.get("failed") == 0,
    }


def _merged_access_log(*stages) -> list:
    merged: list = []
    for stage in stages:
        for entry in stage.get("bank_access_log", []):
            if entry not in merged:
                merged.append(entry)
    return merged


def stage_artifacts(args) -> dict:
    from stratego.training.phase10_contract import HARD_GATE_IDS

    verify = read_stage("verify")
    banks = read_stage("banks")
    games = read_stage("games")
    audit = read_stage("audit")
    reproduce = read_stage("reproduce")
    gates = read_stage("gates")
    suite = getattr(args, "suite", None)
    if suite is None and stage_file_path("suite").exists():
        suite = read_stage("suite")

    completion = _completion_gates(verify, banks, games, audit, reproduce, gates, suite)
    false_gates = sorted(name for name, value in completion.items() if not value)
    hard_gates_pass = gates["hard_gates_all_pass"]
    evidence_complete = not [name for name in false_gates if name != "full_suite_green"]

    if not evidence_complete:
        recommendation = "BLOCKED"
    elif not hard_gates_pass:
        recommendation = "FAIL"
    else:
        recommendation = gates["classification"]
    status = recommendation

    access_log = _merged_access_log(verify, banks, games)
    final_access = [
        entry
        for entry in access_log
        if entry["bank"] == "phase10_test_bank_v1" and entry["outcomes"]
    ]

    acceptance = {
        "phase": PHASE,
        "agent": AGENT,
        "artifact": "agent_07_final_acceptance",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": status,
        "recommendation": recommendation,
        "classification": gates["classification"],
        "classification_logic": {
            "rule": (
                "BLOCKED if identity/sealing/discipline evidence is incomplete; "
                "FAIL if any hard gate fails after correct execution; "
                "PASS-IMPROVED if all eight gates pass, Gate A meets the improved "
                "criterion and Gate B is significantly positive; "
                "PASS-NONINFERIOR otherwise"
            ),
            "hard_gates_all_pass": hard_gates_pass,
            "gate_a_improved": gates["gates"]["A"]["improved"],
            "gate_b_significantly_positive": gates["gates"]["B"]["significantly_positive"],
            "evidence_complete": evidence_complete,
            "recomputed_from_gate_rows": gates["classification_recomputed_from_gate_rows"],
        },
        "bank": {
            "bank_version": "phase10_test_bank_v1",
            "bank_digest": banks["test_audit"]["bank_digest"],
            "cases": gates["case_count"],
            "bootstrap_root": 2026081808,
            "bootstrap_replicates": 10_000,
            "bootstrap_unit": "phase10_logical_case",
        },
        "evaluated_system": {
            "candidate_id": SELECTED_WINNER["candidate_id"],
            "utility_model": SELECTED_WINNER["utility_model"],
            "temperature": SELECTED_WINNER["temperature"],
            "selector_identity": SELECTED_WINNER["selector_identity"],
            "selector_config_sha256": ACCEPTED_CONFIG_SHA256,
            "phase10_system_v1_digest": ACCEPTED_SYSTEM_DIGEST,
            "baseline": "neutral_v1",
        },
        "games": {
            "total": games["games"],
            "learned_arm": 6 * gates["case_count"] * 2,
            "neutral_arm": 5 * gates["case_count"] * 2,
            "wall_clock_seconds": games["wall_clock_seconds"],
            "workers": games["workers"],
            "device": games["device"],
            "inference": games["inference"],
        },
        "matchups": gates["matchups"],
        "gates": gates["gates"],
        "gates_true": gates["gates_true"],
        "gates_total": gates["gates_total"],
        "hard_gate_ids": list(HARD_GATE_IDS),
        "diversity_report": gates["diversity_report"],
        "correctness_report": gates["correctness_report"],
        "reproducibility_report": gates["reproducibility_report"],
        "preservation_report": gates["preservation_report"],
        "independent_check": gates["independent_check"],
        "landing_diagnostic": gates["landing_diagnostic"],
        "critical_identities": {
            "contract_bundle_digest": verify["contract_bundle_digest"],
            "contract_digests": verify["contract_digests"],
            "root_seeds": verify["root_seeds"],
            "phase9_checkpoint_sha256": ACCEPTED_PHASE9_SHA256,
            "phase9_model_state_digest": ACCEPTED_PHASE9_STATE_DIGEST,
            "phase9_parameters": ACCEPTED_PHASE9_PARAMETERS,
            "phase7_library_content_digest": verify["library"]["content_digest"],
            "utility_file_sha256": verify["utility"]["file_sha256"],
            "utility_coefficient_digests": verify["utility"]["coefficient_digests"],
            "trait_scaler_digest": verify["utility"]["scaler_digest"],
            "selector_config_sha256": verify["selector_config"]["artifact_sha256"],
            "selector_contract_digest": verify["selector_contract_digest"],
            "production_train_digests": verify["production_train_digests"],
            "phase10_system_v1_template_digest": verify["phase10_system_v1"][
                "frozen_template_digest"
            ],
            "phase10_system_v1_instance_digest": verify["phase10_system_v1"][
                "filled_instance_digest"
            ],
            "validation_bank_digest": verify["validation_bank"]["bank_digest"],
            "test_bank_digest": verify["test_bank"]["bank_digest"],
            "corpus_content_digest": verify["corpus"]["content_digest"],
            "soak_content_digest": verify["soak"]["content_digest"],
        },
        "phase10_system_v1": verify["phase10_system_v1"],
        "administrative_freeze": verify["administrative_freeze"],
        "pre_agent7_test_bank_access": {
            key: value
            for key, value in verify["pre_agent7_test_bank_access"].items()
            if key != "test_bank_ledger_entries"
        },
        "pre_agent7_ledger_entries": verify["pre_agent7_test_bank_access"][
            "test_bank_ledger_entries"
        ],
        "bank_access_log": access_log,
        "final_evaluation_access": {
            "entries": final_access,
            "count": len(final_access),
            "statement": (
                "exactly one outcome-bearing access exists: the Agent 7 games "
                "stage, purpose final_evaluation"
            ),
        },
        "phase9_preservation": {
            "before": {
                "sha256": verify["phase9_checkpoint"]["sha256"],
                "model_state_digest": verify["phase9_checkpoint"]["model_state_digest"],
            },
            "after": {
                "sha256": gates["preservation_report"]["checkpoint_sha256"],
                "model_state_digest": gates["preservation_report"]["model_state_digest"],
            },
            "parameters": gates["preservation_report"]["parameters"],
            "all_parameters_finite": gates["preservation_report"]["all_parameters_finite"],
            "c1_optimizer_steps": 0,
            "unchanged": verify["phase9_checkpoint"]["sha256"]
            == gates["preservation_report"]["checkpoint_sha256"],
        },
        "discipline": {
            "utility_models_fit": 0,
            "candidates_added": 0,
            "candidates_evaluated_on_test_bank": 1,
            "temperature_changes": 0,
            "mixture_changes": 0,
            "threshold_changes": 0,
            "rescue_reruns": 0,
            "winner_switches_after_test": 0,
            "report_only_metrics_used_in_gates": 0,
            "c1_optimizer_steps": 0,
            "human_games_used": 0,
            "test_bank_outcome_access_by_agent_7": games["games"],
        },
        "reproduce": {
            key: value
            for key, value in reproduce.items()
            if key in ("draws_reconstructed", "topologies", "cross_topology_disagreements",
                        "unit_replay", "hidden_input_control")
        },
        "audit_summary": {
            "games_audited": audit["counters"]["games"],
            "actions_replayed": audit["counters"]["replayed_actions"],
            "seats_audited": audit["seats_audited"],
            "seat_mismatches": audit["counters"]["seat_mismatches"],
            "weights_binding": audit["weights_binding"],
        },
        "completion_gates": completion,
        "false_gates": false_gates,
        "suite": suite,
        "suite_before": TESTS_BEFORE,
        "closure": {
            "on_pass_freeze_permanently": [
                "neutral_v1",
                "learned_setup_source_v1 (accepted P10-D configuration)",
                "phase10_selector_config_v1 6e227815",
                "setup_utility_v1 model_T d898782a + trait scaler fa6eb1c1",
                "accepted Phase 9 selfplay_c1_v1.pt dfd698e5",
            ],
            "phase11": (
                "belief validation; does not retune setup selection"
            ),
            "formal_closure": "belongs to the reviewing chat",
        },
        "environment": environment_record(),
    }
    write_json(ACCEPTANCE_ARTIFACT, acceptance)

    log("writing the strength CSV")
    with open(STRENGTH_ARTIFACT, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "bank", "arm", "candidate_id", "matchup", "cases", "games",
                "wins", "draws", "losses", "ewr", "red_ewr", "blue_ewr",
                "ci_lower", "ci_upper", "delta", "delta_ci_lower", "delta_ci_upper",
                "landings", "landing_rate",
            ]
        )
        for key, cell in sorted(gates["cells"].items()):
            arm, matchup = key.split("/")
            summary = gates["matchups"].get(matchup, {})
            is_learned = arm == "learned"
            counts = cell["counts"]
            colors = cell["color_split"]
            writer.writerow(
                [
                    "phase10_test_bank_v1",
                    arm,
                    SELECTED_WINNER["candidate_id"] if is_learned else "neutral_v1",
                    matchup,
                    gates["case_count"],
                    counts["games"],
                    counts["wins"],
                    counts["draws"],
                    counts["losses"],
                    f"{counts['ewr']:.6f}",
                    f"{colors['red']['ewr']:.6f}",
                    f"{colors['blue']['ewr']:.6f}",
                    (
                        f"{summary['learned_interval']['lower']:.6f}"
                        if is_learned
                        else f"{summary['neutral_interval']['lower']:.6f}"
                    ),
                    (
                        f"{summary['learned_interval']['upper']:.6f}"
                        if is_learned
                        else f"{summary['neutral_interval']['upper']:.6f}"
                    ),
                    (
                        f"{summary['delta']:.6f}"
                        if is_learned and "delta" in summary
                        else ""
                    ),
                    (
                        f"{summary['delta_interval']['lower']:.6f}"
                        if is_learned and "delta_interval" in summary
                        else ""
                    ),
                    (
                        f"{summary['delta_interval']['upper']:.6f}"
                        if is_learned and "delta_interval" in summary
                        else ""
                    ),
                    cell["landings"]["landings"],
                    f"{cell['landings']['rate']:.6f}",
                ]
            )

    log("writing the diversity CSV")
    with open(DIVERSITY_ARTIFACT, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "candidate_id", "color", "split",
                "normalized_family_entropy", "effective_families",
                "min_family_probability", "max_family_probability",
                "min_within_family_normalized_base_entropy",
                "max_conditional_base_probability", "effective_base_diversity",
                "all_thresholds_pass", "mixture_exact", "all_finite",
                "probability_vector_digest",
            ]
        )
        for row in gates["diversity_rows"]:
            writer.writerow(
                [
                    row["candidate_id"],
                    row["color"],
                    row["split"],
                    f"{row['normalized_family_entropy']:.6f}",
                    f"{row['effective_families']:.6f}",
                    f"{row['min_family_probability']:.6f}",
                    f"{row['max_family_probability']:.6f}",
                    f"{row['min_within_family_normalized_base_entropy']:.6f}",
                    f"{row['max_conditional_base_probability']:.6f}",
                    f"{row['effective_base_diversity']:.6f}",
                    row["all_thresholds_pass"],
                    row["mixture_exact"],
                    row["all_finite"],
                    row["probability_vector_digest"],
                ]
            )

    payload = {
        "stage": "artifacts",
        "status": status,
        "recommendation": recommendation,
        "completion_gates": completion,
        "false_gates": false_gates,
        "artifacts": [
            str(ACCEPTANCE_ARTIFACT.relative_to(REPOSITORY_ROOT)),
            str(STRENGTH_ARTIFACT.relative_to(REPOSITORY_ROOT)),
            str(DIVERSITY_ARTIFACT.relative_to(REPOSITORY_ROOT)),
        ],
        "environment": environment_record(),
    }
    write_stage("artifacts", payload)
    log(f"  status {status}; {sum(completion.values())}/{len(completion)} completion gates")
    return payload


# ---------------------------------------------------------------------------
# stage: report — §7 of the implementation report
# ---------------------------------------------------------------------------


def stage_report(args) -> dict:
    acceptance = read_json(ACCEPTANCE_ARTIFACT)
    verify = read_stage("verify")
    banks = read_stage("banks")
    games = read_stage("games")
    audit = read_stage("audit")
    reproduce = read_stage("reproduce")
    gates = read_stage("gates")

    section = _render_section(acceptance, verify, banks, games, audit, reproduce, gates)
    text = REPORT_PATH.read_text(encoding="utf-8")
    if SECTION_MARKER in text:
        text = text[: text.index(SECTION_MARKER)].rstrip() + "\n\n" + section
    else:
        text = text.rstrip() + "\n\n" + section
    REPORT_PATH.write_text(text, encoding="utf-8")
    payload = {"stage": "report", "section_marker": SECTION_MARKER, "bytes": len(section)}
    write_stage("report", payload)
    log(f"  report section written ({len(section):,} bytes)")
    return payload


def _render_section(acceptance, verify, banks, games, audit, reproduce, gates) -> str:
    lines: list = []
    add = lines.append
    matchups = gates["matchups"]
    gate_rows = gates["gates"]

    add(SECTION_MARKER)
    add("")
    add(f"Status: **{acceptance['status']}** — recommendation "
        f"`{acceptance['recommendation']}`, "
        f"{gates['gates_true']}/{gates['gates_total']} hard gates pass, "
        f"{sum(acceptance['completion_gates'].values())}/"
        f"{len(acceptance['completion_gates'])} completion gates true.")
    add("")
    add("Agent 7 independently recomputed every Agent 1-6 identity and discipline")
    add("claim from live bytes, rebuilt and exhaustively audited both frozen banks,")
    add("proved from the recorded access ledgers that `phase10_test_bank_v1` had")
    add("zero prior outcome evaluation, and then performed the first and only")
    add("final-test evaluation: the permanently selected P10-D configuration against")
    add("the fixed `neutral_v1` baseline on the 512 sealed cases. No training, no")
    add("candidate replacement, no threshold change, no rescue rerun; report-only")
    add("diagnostics rescued nothing. Formal closure belongs to the reviewing chat.")
    add("")
    add("### 7.1 Prerequisites, from live bytes")
    add("")
    add("Agents 1-6 all report `PASS` with every completion gate recomputing true.")
    add(f"The working tree stands at the administrative freeze commit "
        f"`{verify['administrative_freeze']['commit'][:7]}` with zero tracked")
    add("modifications; untracked files are Agent 7's own code and artifacts. The")
    add("eight contract digests and the bundle recompute exactly; the eight root")
    add("seeds match the contract; the Phase 9 checkpoint hashes to the accepted")
    add(f"`{ACCEPTED_PHASE9_SHA256[:16]}…` with the accepted model-state digest and")
    add("863,959 finite parameters; the Phase 7 library re-digests exactly with")
    add("6,400/800/800 splits; the utility artifact, both coefficient digests, the")
    add("train-only scaler, the frozen P10-D selector config and all 36 published")
    add("distribution digests recompute exactly; the production train vectors match")
    add(f"red `{ACCEPTED_TRAIN_DIGESTS['red'][:16]}…` / blue "
        f"`{ACCEPTED_TRAIN_DIGESTS['blue'][:16]}…`; the sealed corpus")
    add(f"(`{verify['corpus']['content_digest'][:16]}…`, 16,384 train-only games) and")
    add(f"the sealed soak (`{verify['soak']['content_digest'][:16]}…`, 8,192 games)")
    add("re-verify from live bytes.")
    add("")
    add("### 7.2 The two `phase10_system_v1` identities, distinguished")
    add("")
    add("```text")
    add(f"frozen template  (Agent 1 contract)   "
        f"{verify['phase10_system_v1']['frozen_template_digest']}")
    add(f"filled instance  (Agent 6 production) "
        f"{verify['phase10_system_v1']['filled_instance_digest']}")
    add("```")
    add("")
    add("The template digest is part of the frozen Agent 1 bundle and recomputes")
    add("unchanged, so no upstream contract moved. The instance is a different")
    add("document by design — the same binding schema with its three slots filled —")
    add("and it was verified against the template's *filling rules* rather than its")
    add("digest: utility slot names `phase10_setup_utility_v1`/`model_T` with the")
    add("accepted coefficient digest and fit-corpus digest and `single_fit: true`;")
    add("scaler slot names `phase10_trait_scaler_v1` with the accepted train-only")
    add("digest; selector slot names P10-D from the frozen six at T=0.75 under the")
    add("unchanged 0.35/0.65 mixture through `learned_setup_source_v1`; every")
    add("bound-now field (move model, library, reflection/perturbation path,")
    add("`neutral_v1` untouched) is byte-equal to the template's; and no filesystem")
    add("path appears in the identity. All "
        f"{len(verify['phase10_system_v1']['filling_rules'])} filling rules pass.")
    add("")
    add("### 7.3 Bank rebuild and sealing")
    add("")
    add("```text")
    add(f"validation bank   rebuilt {banks['validation_audit']['bank_digest'][:16]}… "
        "== accepted, every case audited")
    add(f"test bank         rebuilt {banks['test_audit']['bank_digest'][:16]}… "
        "== accepted, every case audited")
    add("isolated rebuild  every case of both banks rebuilt alone (sample_every=1)")
    add(f"cross-bank overlap {banks['cross_bank_isolation']['overlap_count']}")
    add(f"Phase 9 isolation  {banks['phase9_isolation']['set_size']} identities, "
        "coverage reconciliation exact")
    add("```")
    add("")
    add("Every recorded pre-Agent-7 test-bank access across all six prior agents is")
    add("structural (`neural=false`, `outcomes=false`); Agent 5 records zero")
    add("test-bank outcome reads; Agent 6 records the bank unopened; no stored")
    add("evaluation cell anywhere names the test bank version. The games stage was")
    add("therefore the bank's first game, first neural inference and first outcome")
    add("read.")
    add("")
    add("### 7.4 What was played")
    add("")
    add("```text")
    add("bank              phase10_test_bank_v1, 512 logical paired cases")
    add("learned arm       P10-D (model_T, T=0.75), 6 matchups x 512 cases x 2 games")
    add("neutral arm       neutral_v1, 5 matchups x 512 cases x 2 games")
    add(f"games             {games['games']:,} in {games['wall_clock_seconds']:.0f}s "
        f"on {games['workers']} workers ({games['device']})")
    add("move behaviour    accepted Phase 9 checkpoint, greedy float32 single_request,")
    add("                  no search, both arms on identical logical cases")
    add(f"inference         {games['inference']['requests_served']:,} requests, "
        f"{games['inference']['failures_returned']} failures")
    add("```")
    add("")
    add("### 7.5 Final results and the eight hard gates")
    add("")
    direct = matchups["learned_vs_neutral"]
    add("```text")
    add("matchup              learned EWR   neutral EWR   delta      delta 95% CI")
    for token in ("learned_vs_neutral", "vs_strategic", "vs_tactical",
                  "vs_phase8_anchor", "vs_random", "vs_basic"):
        entry = matchups[token]
        if "delta" in entry:
            add(
                f"{token:<20} {entry['learned_ewr']:.4f}        "
                f"{entry['neutral_ewr']:.4f}        {entry['delta']:+.4f}    "
                f"[{entry['delta_interval']['lower']:+.4f}, "
                f"{entry['delta_interval']['upper']:+.4f}]"
            )
        else:
            add(
                f"{token:<20} {entry['learned_ewr']:.4f}        —             —          "
                f"[{entry['learned_interval']['lower']:.4f}, "
                f"{entry['learned_interval']['upper']:.4f}] (EWR CI)"
            )
    add("```")
    add("")
    add("```text")
    a = gate_rows["A"]
    add(f"A direct         EWR {a['ewr']:.4f}, paired 95% LB {a['lower_bound']:.4f}  "
        f"-> {'PASS' if a['pass'] else 'FAIL'}"
        f"{' + improved' if a['improved'] else ''}")
    b = gate_rows["B"]
    add(f"B league         Delta_L {b['delta_l']:+.4f}, LB {b['interval']['lower']:+.4f}  "
        f"-> {'PASS' if b['pass'] else 'FAIL'}"
        f"{' + significantly positive' if b['significantly_positive'] else ''}")
    c = gate_rows["C"]
    bounds = ", ".join(f"{k.split('_',1)[1]} {v:+.4f}" for k, v in c["lower_bounds"].items())
    add(f"C individual     LBs: {bounds}  -> {'PASS' if c['pass'] else 'FAIL'}")
    d = gate_rows["D"]
    add(f"D easy           Random {d['random_overall']:.4f} "
        f"(R {d['random_red']:.4f}/B {d['random_blue']:.4f}), "
        f"Basic {d['basic']:.4f}  -> {'PASS' if d['pass'] else 'FAIL'}")
    e = gate_rows["E"]
    add(f"E diversity      worst H/log16 {e['observed']['normalized_family_entropy']:.4f}, "
        f"eff fams {e['observed']['effective_families']:.2f}  "
        f"-> {'PASS' if e['pass'] else 'FAIL'}")
    f = gate_rows["F"]
    add(f"F correctness    nine counters all zero: {f['pass']}  "
        f"-> {'PASS' if f['pass'] else 'FAIL'}")
    g = gate_rows["G"]
    add(f"G reproducible   all six checks true: {g['pass']}  "
        f"-> {'PASS' if g['pass'] else 'FAIL'}")
    h = gate_rows["H"]
    add(f"H preservation   exact SHA/state/params, zero steps: {h['pass']}  "
        f"-> {'PASS' if h['pass'] else 'FAIL'}")
    add("```")
    add("")
    add(f"Classification: **{gates['classification']}** — recomputed independently "
        "from the gate rows, and every")
    add("point estimate and interval bound was re-derived from the raw stored rows")
    add("by direct arithmetic and an independently assembled bootstrap under the")
    add("frozen final-test root 2026081808 (10,000 paired replicates, logical-case")
    add("resampling), agreeing to the last bit.")
    add("")
    add("### 7.6 Final replay/safety audit")
    add("")
    add("```text")
    add(f"games audited               {audit['counters']['games']:,}")
    add(f"actions replayed            {audit['counters']['replayed_actions']:,} "
        "(every stored history re-applied through a fresh engine)")
    add(f"seats reconciled            {audit['seats_audited']:,} (cryptographic match_id check)")
    add(f"setup reconstructions       every game's own-side draw re-derived from "
        "identity/seed alone")
    add(f"draws re-derived twice      {reproduce['draws_reconstructed']:,} across two "
        "worker topologies (5 fwd / 3 rev)")
    add(f"cross-topology mismatches   {reproduce['cross_topology_disagreements']}")
    add(f"unit replay                 {reproduce['unit_replay']['games']} games under "
        "1 worker, every field identical")
    add(f"hidden-input control        {len(reproduce['hidden_input_control']['attempts'])}"
        "/5 forbidden fields rejected")
    add("```")
    add("")
    add("### 7.7 Phase 9 fingerprint landings — report-only")
    add("")
    add("Agent 1's standing obligation at the required granularity — candidate x")
    add("arm x matchup x bank, count and rate. Recorded and never read: no gate, no")
    add("retry, no rejection sampling consulted these values.")
    add("")
    add("```text")
    add("arm      selector     matchup               landings / games     rate")
    for row in gates["landing_diagnostic"]["rows"]:
        add(
            f"{row['arm']:<8} {row['candidate_id']:<12} {row['matchup']:<20}  "
            f"{row['landings']:>3} / {row['games']:>5}       {row['landing_rate']:.4f}"
        )
    add("```")
    add("")
    add("The per-matchup count is constant within an arm because an own-side draw")
    add("depends on the case, the colour and the selector — never on the opponent.")
    add("The learned rate sits well below Agent 4's unconditioned test-split")
    add("measurement (~0.136 for P10-D over 200k free draws) for a structural")
    add("reason: the bank's accepted selector seeds were chosen so their")
    add("`neutral_v1` draws are isolation-clean, so the 35% neutral branch cannot")
    add("land by construction and only learned-branch draws can. The baseline arm")
    add("records zero by the same walk. Higher than the validation-bank rate")
    add("(0.0273) because the Phase 9 test bank drew 1,024 of the isolation set's")
    add("sides from this same test split.")
    add("")
    add("### 7.8 Phase 9 preservation and discipline")
    add("")
    p = acceptance["phase9_preservation"]
    add("```text")
    add(f"before   {p['before']['sha256']}")
    add(f"after    {p['after']['sha256']}")
    add(f"state    {p['after']['model_state_digest']} (unchanged: {p['unchanged']})")
    add(f"parameters {p['parameters']:,} all finite; C1 optimizer steps 0")
    add("```")
    add("")
    add("Utility models fit 0; candidates added 0; candidates evaluated on the test")
    add("bank 1 (the permanently selected P10-D); temperature/mixture/threshold")
    add("changes 0/0/0; rescue reruns 0; winner switches after test 0; report-only")
    add("metrics used in gates 0; human games 0.")
    add("")
    add("### 7.9 Artifacts and completion gates")
    add("")
    add("```text")
    add("reports/phase_10_data/agent_07_final_acceptance.json")
    add("reports/phase_10_data/agent_07_strength_results.csv")
    add("reports/phase_10_data/agent_07_diversity_results.csv")
    add("```")
    add("")
    suite = acceptance.get("suite") or {}
    add(f"Full suite: `{suite.get('command', '')}` — {suite.get('summary', '')}")
    add("(recorded via `--record-suite`; the artifact test checks the gate against")
    add("the recorded measurement, and a confirming re-record was taken with the")
    add("artifacts in their final state).")
    add("")
    add("| gate | value |")
    add("| --- | --- |")
    for name in COMPLETION_GATES:
        value = acceptance["completion_gates"].get(name)
        add(f"| `{name}` | {str(value).lower()} |")
    add("")
    add("### 7.10 Recommendation and closure")
    add("")
    add(f"Recommendation: **`{acceptance['recommendation']}`**")
    add("")
    add("On acceptance by the reviewing chat, the following are frozen permanently:")
    add("`neutral_v1`; the accepted `learned_setup_source_v1` (P10-D, model_T,")
    add("T=0.75, 0.35/0.65 mixture, config `6e227815…`); the accepted utility")
    add("(`model_T` `d898782a…`) and trait scaler (`fa6eb1c1…`); and the accepted")
    add("Phase 9 `selfplay_c1_v1.pt` (`dfd698e5…`). Phase 11 validates the belief")
    add("system and does not retune setup selection.")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage registry, suite runner, main
# ---------------------------------------------------------------------------

STAGES = {
    "verify": stage_verify,
    "banks": stage_banks,
    "games": stage_games,
    "audit": stage_audit,
    "reproduce": stage_reproduce,
    "gates": stage_gates,
    "artifacts": stage_artifacts,
    "report": stage_report,
}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), action="append")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--cases", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--run-pytest", action="store_true")
    parser.add_argument("--record-suite", action="store_true")
    parser.add_argument("--audit-workers", type=int, default=8)
    parser.add_argument("--audit-sample", type=int, default=12)
    parser.add_argument("--reproduce-matchup", default="vs_strategic")
    parser.add_argument("--reproduce-start", type=int, default=128)
    parser.add_argument("--cell-worker", action="store_true")
    parser.add_argument("--audit-shard", action="store_true")
    parser.add_argument("--reproduce-shard", action="store_true")
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--worker", type=int, default=0)
    return parser.parse_args(argv)


def run_suite(args) -> dict:
    log("running the full suite")
    started = time.perf_counter()
    completed = subprocess.run(
        [".venv/bin/python", "-m", "pytest", "tests", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    tail = completed.stdout.strip().splitlines()
    summary = tail[-1] if tail else ""
    passed = failed = skipped = 0
    for count, label in re.findall(r"(\d+)\s+(passed|failed|skipped|error)", summary):
        if label == "passed":
            passed = int(count)
        elif label == "failed":
            failed = int(count)
        elif label == "skipped":
            skipped = int(count)
    return {
        "command": ".venv/bin/python -m pytest tests -q",
        "returncode": completed.returncode,
        "summary": summary,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "seconds": elapsed,
    }


def main(argv=None) -> int:
    args = parse_arguments(argv)
    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if args.cell_worker:
        cell_worker(args)
        return 0
    if args.audit_shard:
        stage_audit_shard(args)
        return 0
    if args.reproduce_shard:
        stage_reproduce_shard(args)
        return 0

    if args.record_suite:
        # `full_suite_green` is a claim about a suite that contains the test
        # asserting it, so it cannot be evidenced by a single run: the first
        # run sees the previous artifact. Recording the measurement in its
        # own stage makes the ordering explicit and auditable — write the
        # artifact from a recorded run, then re-record to confirm the suite
        # is green with the artifact in its final state.
        measured = run_suite(args)
        write_stage("suite", measured)
        log(f"  recorded suite: {measured['summary']}")
        return 0 if measured["returncode"] == 0 else 1

    if args.run_pytest:
        args.suite = run_suite(args)
        if args.suite["returncode"] != 0:
            log(f"  suite FAILED: {args.suite['summary']}")
            return 1
        log(f"  suite: {args.suite['summary']}")
    elif stage_file_path("suite").exists():
        args.suite = read_stage("suite")
    else:
        args.suite = None

    requested = args.stage or list(STAGES)
    for name in STAGES:
        if name not in requested:
            continue
        log(f"stage {name}")
        started = time.perf_counter()
        STAGES[name](args)
        log(f"  stage {name} finished in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
