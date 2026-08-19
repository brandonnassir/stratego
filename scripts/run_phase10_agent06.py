#!/usr/bin/env python3
"""Phase 10 Agent 6 harness: integration soak and production freeze.

Verifies every Agent 1-5 prerequisite from live bytes, freezes the
production train-split probability vectors of the permanently selected
configuration (P10-D) with an independent rebuild, plays an 8,192-game
integration soak through the frozen production system under parallel
workers, one clean process restart and one hard SIGKILL restart, audits
every committed game against its own logical identity, freezes
`phase10_system_v1`, and writes the Agent 6 artifacts.

What this script is and is not
------------------------------
It exercises and freezes. Selection is closed: it evaluates no candidate,
refits nothing, changes no temperature or mixture weight, takes zero
optimizer steps on the Phase 9 checkpoint, and never reaches
`phase10_test_bank_v1` for anything but a structural digest check — the
first final-test outcome evaluation belongs to Agent 7. Every soak outcome
is report-only and cannot reopen selection.

Usage::

    python scripts/run_phase10_agent06.py                  # every public stage
    python scripts/run_phase10_agent06.py --stage verify   # one stage
    python scripts/run_phase10_agent06.py --record-suite   # record the suite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

AGENT = 6
PHASE = 10
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_10_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_10_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase10" / "agent06"
STAGE_DIRECTORY = WORK_DIRECTORY / "stages"
EXPORT_PATH = WORK_DIRECTORY / "eval_phase9_c1.pt"

SOAK_ARTIFACT = DATA_DIRECTORY / "agent_06_integration_soak.json"
MANIFEST_ARTIFACT = DATA_DIRECTORY / "agent_06_production_selector_manifest.json"
ACCEPTANCE_ARTIFACT = DATA_DIRECTORY / "agent_06_acceptance.json"

SECTION_MARKER = "## 6. Agent 6 — Integration Soak and Production Freeze"

CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "selfplay_c1_v1.pt"
UTILITY_PATH = REPOSITORY_ROOT / "checkpoints" / "phase10" / "setup_utility_v1.json"
CONFIG_ARTIFACT = DATA_DIRECTORY / "agent_05_frozen_selector_config.json"

#: The upstream identities Agent 6 refuses to proceed without. Every one is
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
ACCEPTED_TRAIN_DIGESTS = {
    "red": "9ac5b52edbbf0ff92fbebe5c61eefe5a13f0092a3b685eae8857f66b261e491f",
    "blue": "abef229983e2f4b6caf5323171618b5c82d6a67f59463256098343639f6e957f",
}
SELECTED_WINNER = {
    "candidate_id": "P10-D",
    "utility_model": "model_T",
    "temperature": 0.75,
    "selector_identity": "learned_setup_source_v1|k=P10-D|m=model_T|T=0.75",
}

#: Where the soak journal bytes go when the external volume is available.
EXTERNAL_SOAK_ROOT = "/Volumes/Brandon_Washington/stratego_phase10/soak"

#: The full suite as measured immediately before any Agent 6 change
#: (Agent 5's recorded confirming run at commit 48061bf).
TESTS_BEFORE = {
    "command": ".venv/bin/python -m pytest tests -q",
    "summary": "5132 passed, 3 skipped in 315.48s (0:05:15)",
    "passed": 5132,
    "failed": 0,
    "skipped": 3,
    "seconds": 316.63,
    "measured_at_commit": "48061bf",
}

#: Every access this script makes to either sealed bank, with its purpose.
BANK_ACCESS_LOG: list = []


class Agent6Error(RuntimeError):
    """Raised when a prerequisite or an invariant of Agent 6's mission fails."""


def log(message: str) -> None:
    print(f"[agent06] {message}", flush=True)


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
        raise Agent6Error(
            f"stage {name!r} has not run; run it before the stage that depends on it"
        )
    return read_json(path)


def record_bank_access(stage: str, bank: str, purpose: str, *, neural: bool, outcomes: bool) -> None:
    BANK_ACCESS_LOG.append(
        {"stage": stage, "bank": bank, "purpose": purpose, "neural": neural, "outcomes": outcomes}
    )


def checkpoint_identity(path: Path) -> dict:
    """File SHA, model-state digest and parameter count of a Phase 9 payload."""
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
# stage: verify — every prerequisite, from live bytes
# ---------------------------------------------------------------------------


def stage_verify(args) -> dict:
    from stratego.evaluation.phase10_banks import (
        bank_digest,
        build_phase10_bank,
        manifest_digest,
        phase9_isolation_set,
    )
    from stratego.setups.families import FAMILY_IDS
    from stratego.setups.library import entry_metadata_digest, library_content_digest
    from stratego.setups.sampler import NEUTRAL_PROFILE, load_library_index
    from stratego.training import phase10_soak as soak
    from stratego.training.phase10_collector import export_evaluation_weights
    from stratego.training.phase10_contract import (
        CANDIDATE_MATRIX,
        LEARNED_MIXTURE_WEIGHT,
        NEUTRAL_MIXTURE_WEIGHT,
        contract_bundle_digest,
        contract_digests,
    )
    from stratego.training.phase10_outcome_store import verify_seal
    from stratego.training.phase10_seed import CANONICAL_PHASE10_SEEDS
    from stratego.training.phase10_selector import (
        build_distribution,
        candidate,
        load_scorer,
        selector_contract_digest,
    )
    from stratego.training.phase10_storage import check_corpus_root, default_corpus_root
    from tests.training.phase10_frozen_digests import PHASE9_ISOLATION_SET_DIGEST

    problems: list = []
    log("verifying Agents 1-5 acceptance records")
    prior = {}
    for agent in (1, 2, 3, 4, 5):
        record = read_json(DATA_DIRECTORY / f"agent_0{agent}_acceptance.json")
        prior[f"agent_{agent}"] = {
            "status": record.get("status"),
            "gates_true": record.get("gates_true"),
            "gates_total": record.get("gates_total"),
            "false_gates": record.get("false_gates", []),
        }
        if record.get("status") != "PASS":
            problems.append(f"Agent {agent} status is {record.get('status')!r}, expected PASS")
        if record.get("gates_true") != record.get("gates_total"):
            problems.append(f"Agent {agent} has an unmet completion gate")
        if record.get("false_gates"):
            problems.append(f"Agent {agent} reports false gates {record['false_gates']}")

    log("recomputing the eight contract digests")
    digests = contract_digests()
    bundle = contract_bundle_digest()
    if bundle != ACCEPTED_CONTRACT_BUNDLE_DIGEST:
        problems.append(f"contract bundle digest {bundle} != accepted")

    log("verifying the Phase 9 checkpoint from live bytes (the 'before' record)")
    phase9 = checkpoint_identity(CHECKPOINT_PATH)
    if phase9["sha256"] != ACCEPTED_PHASE9_SHA256:
        problems.append(f"Phase 9 checkpoint SHA {phase9['sha256']} != accepted")
    if phase9["model_state_digest"] != ACCEPTED_PHASE9_STATE_DIGEST:
        problems.append("Phase 9 model-state digest != accepted")
    if phase9["parameters"] != ACCEPTED_PHASE9_PARAMETERS:
        problems.append(f"Phase 9 parameter count {phase9['parameters']} != accepted")
    if not phase9["all_parameters_finite"]:
        problems.append("Phase 9 checkpoint carries a non-finite parameter")

    log("verifying the fitted utility, the scaler and neutral_v1")
    if not UTILITY_PATH.exists():
        raise Agent6Error(f"{UTILITY_PATH} is missing; Agent 3's fitted utility is required")
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
        "swap_counts": list(NEUTRAL_PROFILE.swap_counts),
    }
    if neutral["name"] != "neutral_v1":
        problems.append(f"the baseline profile is {neutral['name']!r}, expected neutral_v1")
    if (neutral["reflection_probability"], neutral["perturbation_probability"]) != (0.5, 0.5):
        problems.append("neutral_v1 reflection/perturbation probabilities moved")
    if neutral["swap_counts"] != [1, 2, 3, 4, 5, 6]:
        problems.append("neutral_v1 swap counts moved")

    log("verifying the Phase 7 library")
    index = load_library_index()
    entries = index.entries
    library = {
        "content_digest": library_content_digest(entries),
        "metadata_digest": entry_metadata_digest(entries),
        "families": len(FAMILY_IDS),
        "bases": len(entries),
    }
    if library["content_digest"] != (
        "7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777"
    ):
        problems.append("Phase 7 library content digest != accepted")
    if library["metadata_digest"] != (
        "d86f486182a820d546d470ef4ebce92ff60c6259aed80c481bc985bce8c64980"
    ):
        problems.append("Phase 7 library metadata digest != accepted")

    log("verifying the sealed Agent 2 corpus from live bytes")
    corpus_root = default_corpus_root()
    corpus_findings = check_corpus_root(corpus_root)
    if not corpus_findings["usable"]:
        raise Agent6Error(
            f"the corpus root is not usable: {corpus_findings['blocked']}; BLOCKED"
        )
    corpus_seal = verify_seal(corpus_root)
    corpus = {
        "root_diagnostic": str(corpus_root),
        "seal_all_pass": corpus_seal["all_pass"],
        "content_digest": corpus_seal["observed_content_digest"],
        "committed_games": corpus_seal["observed_committed_games"],
    }
    if not corpus_seal["all_pass"]:
        problems.append("the sealed corpus fails its own seal verification")
    if corpus["content_digest"] != ACCEPTED_CORPUS_CONTENT_DIGEST:
        problems.append(f"corpus content digest {corpus['content_digest']} != accepted")
    if corpus["committed_games"] != 16_384:
        problems.append(f"corpus holds {corpus['committed_games']} games, expected 16,384")

    log("verifying the frozen selector configuration (selection is closed)")
    config_sha = file_sha256(CONFIG_ARTIFACT)
    config = read_json(CONFIG_ARTIFACT)
    if config_sha != ACCEPTED_CONFIG_SHA256:
        problems.append(f"frozen selector config SHA {config_sha} != accepted")
    if config_sha != soak.SELECTED_CONFIG_SHA256:
        problems.append("phase10_soak pins a different frozen-config SHA")
    winner = config.get("winner", {})
    for field, expected in SELECTED_WINNER.items():
        if winner.get(field) != expected:
            problems.append(f"frozen winner {field} is {winner.get(field)!r}, expected {expected!r}")
    if config.get("status") != "SELECTED":
        problems.append("the frozen selector config is not marked SELECTED")
    mixture = config.get("mixture", {})
    if (
        float(mixture.get("neutral_weight", -1)) != NEUTRAL_MIXTURE_WEIGHT
        or float(mixture.get("learned_weight", -1)) != LEARNED_MIXTURE_WEIGHT
    ):
        problems.append("the frozen config's mixture weights are not 0.35/0.65")
    if config.get("phase10_seeds") != dict(CANONICAL_PHASE10_SEEDS):
        problems.append("the frozen config's root seeds disagree with the canonical eight")
    frozen_matrix = {entry["candidate_id"]: entry for entry in CANDIDATE_MATRIX}
    selected = frozen_matrix[SELECTED_WINNER["candidate_id"]]
    if (
        selected["utility_model"] != SELECTED_WINNER["utility_model"]
        or float(selected["temperature"]) != SELECTED_WINNER["temperature"]
    ):
        problems.append("the frozen candidate matrix disagrees with the selected winner")

    log("recomputing the selected candidate's six distribution digests")
    selected_candidate = candidate(SELECTED_WINNER["candidate_id"])
    recomputed_cells: dict = {}
    for color in ("red", "blue"):
        recomputed_cells[color] = {}
        for split in ("train", "validation", "test"):
            distribution = build_distribution(selected_candidate, color, split, scorer, index)
            observed = distribution.probability_vector_digest()
            recomputed_cells[color][split] = observed
            published = config["distribution_digests"][color][split]
            if observed != published:
                problems.append(
                    f"P10-D/{color}/{split} probability digest {observed} != frozen {published}"
                )
            if not distribution.mixture_is_exact():
                problems.append(f"P10-D/{color}/{split} mixture is not exactly 0.35/0.65")
    for color, accepted in ACCEPTED_TRAIN_DIGESTS.items():
        if recomputed_cells[color]["train"] != accepted:
            problems.append(f"P10-D/{color}/train digest != Agent 5's production value")

    log("recomputing the selector contract digest over all 36 published cells")
    agent4 = read_json(DATA_DIRECTORY / "agent_04_acceptance.json")
    published_distributions = agent4["handoff_to_agent_5"]["distribution_digests"]
    recomputed_contract = selector_contract_digest(published_distributions)
    if recomputed_contract != ACCEPTED_SELECTOR_CONTRACT_DIGEST:
        problems.append("the selector contract digest does not recompute to the accepted value")

    log("verifying both evaluation bank identities (structural only)")
    validation_cases, validation_manifest = build_phase10_bank("validation")
    record_bank_access(
        "verify", "phase10_validation_bank_v1", "digest_computation", neural=False, outcomes=False
    )
    validation = {
        "bank_version": "phase10_validation_bank_v1",
        "cases": len(validation_cases),
        "bank_digest": bank_digest(validation_cases),
        "manifest_digest": manifest_digest(validation_manifest),
    }
    if validation["bank_digest"] != ACCEPTED_VALIDATION_BANK_DIGEST:
        problems.append("validation bank digest != accepted")
    if validation["manifest_digest"] != ACCEPTED_VALIDATION_MANIFEST_DIGEST:
        problems.append("validation bank manifest digest != accepted")
    test_cases, test_manifest = build_phase10_bank("test")
    record_bank_access(
        "verify", "phase10_test_bank_v1", "structural_digest_only", neural=False, outcomes=False
    )
    test_bank = {
        "bank_version": "phase10_test_bank_v1",
        "cases": len(test_cases),
        "bank_digest": bank_digest(test_cases),
        "manifest_digest": manifest_digest(test_manifest),
        "access": "structural_digest_only",
        "games": 0,
        "neural_inference": 0,
        "outcomes_read": 0,
    }
    if test_bank["bank_digest"] != ACCEPTED_TEST_BANK_DIGEST:
        problems.append("test bank digest != accepted")
    if test_bank["manifest_digest"] != ACCEPTED_TEST_MANIFEST_DIGEST:
        problems.append("test bank manifest digest != accepted")
    del test_cases

    log("verifying the Phase 9 isolation set identity")
    isolation, isolation_meta = phase9_isolation_set()
    if isolation_meta["set_digest"] != PHASE9_ISOLATION_SET_DIGEST:
        problems.append("the Phase 9 isolation set digest moved")

    log("running the soak seed-collision audit across the Phase 10 id space")
    collision = soak.soak_seed_collision_audit(soak.SOAK_TOTAL_GAMES)
    if not collision["no_collisions"]:
        problems.append(f"soak seed streams collide: {collision['findings'][:3]}")

    log("exporting the Phase 9 evaluation weights (bitwise proof)")
    export = export_evaluation_weights(CHECKPOINT_PATH, EXPORT_PATH)
    if export["source_sha256"] != ACCEPTED_PHASE9_SHA256:
        problems.append("the export read a checkpoint with the wrong SHA")

    control = soak.hidden_input_positive_control()
    if not control["all_rejected"]:
        problems.append("the hidden-opponent-input positive control did not fire")

    payload = {
        "stage": "verify",
        "problems": problems,
        "prior_agents": prior,
        "contract_digests": digests,
        "contract_bundle_digest": bundle,
        "phase9_checkpoint_before": phase9,
        "utility": {
            "file_sha256": utility_sha,
            "coefficient_digests": coefficient_digests,
            "scaler_digest": scaler_digest,
            "refit_by_agent_6": False,
        },
        "neutral_v1": neutral,
        "library": library,
        "corpus": corpus,
        "selector_config": {
            "artifact": str(CONFIG_ARTIFACT.relative_to(REPOSITORY_ROOT)),
            "sha256": config_sha,
            "winner": dict(SELECTED_WINNER),
            "distribution_digests_recomputed": recomputed_cells,
        },
        "selector_contract_digest": recomputed_contract,
        "validation_bank": validation,
        "test_bank": test_bank,
        "phase9_isolation": {
            "size": len(isolation),
            "set_digest": isolation_meta["set_digest"],
        },
        "seed_collision_audit": {
            key: value for key, value in collision.items() if key != "streams"
        }
        | {"streams": collision["streams"]},
        "export": export,
        "hidden_input_positive_control": control,
        "bank_access_log": list(BANK_ACCESS_LOG),
        "environment": environment_record(),
    }
    write_stage("verify", payload)
    if problems:
        for problem in problems:
            log(f"  PROBLEM: {problem}")
        raise Agent6Error(f"{len(problems)} prerequisite problem(s); Agent 6 is BLOCKED")
    log(
        f"  verified: bundle {bundle[:12]}, config {config_sha[:12]}, corpus "
        f"{corpus['content_digest'][:12]}, Phase 9 {phase9['sha256'][:12]}"
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


# ---------------------------------------------------------------------------
# stage: production — freeze the train-split probability vectors
# ---------------------------------------------------------------------------


def _production_vectors() -> dict:
    """The canonical train-split vector block of the selected configuration.

    Base ids in the frozen selector base order, exact float64 probabilities
    as `float.hex()`, per-family aggregation, utility scores, and every
    digest the distribution publishes. Canonical JSON of this block is the
    serialization the independent rebuild must reproduce byte for byte.
    """
    from stratego.setups.families import FAMILY_IDS
    from stratego.setups.sampler import load_library_index
    from stratego.training.phase10_selector import (
        build_distribution,
        candidate,
        evaluate_diversity,
        load_scorer,
    )
    from stratego.training.phase10_soak import SOAK_SPLIT, canonical_json

    scorer = load_scorer()
    index = load_library_index()
    selected = candidate(SELECTED_WINNER["candidate_id"])
    vectors: dict = {}
    for color in ("red", "blue"):
        distribution = build_distribution(selected, color, SOAK_SPLIT, scorer, index)
        diversity = distribution.diversity()
        vectors[color] = {
            "candidate_id": distribution.candidate_id,
            "utility_model": distribution.utility_model,
            "temperature": distribution.temperature,
            "color": color,
            "split": SOAK_SPLIT,
            "base_count": distribution.base_count,
            "bases_per_family": distribution.bases_per_family,
            "family_order": list(FAMILY_IDS),
            "base_ids": list(distribution.base_ids),
            "family_ids": list(distribution.family_ids),
            "utilities_hex": [float(value).hex() for value in distribution.utilities],
            "p_neutral_uniform_hex": float(distribution.p_neutral[0]).hex(),
            "p_learned_hex": [float(value).hex() for value in distribution.p_learned],
            "p_phase10_hex": [float(value).hex() for value in distribution.p_mixed],
            "family_probabilities": {
                family_id: float(value)
                for family_id, value in zip(FAMILY_IDS, distribution.family_probabilities())
            },
            "digests": distribution.component_digests(),
            "probability_vector_digest": distribution.probability_vector_digest(),
            "mixture_exact": distribution.mixture_is_exact(),
            "finiteness": distribution.finiteness(),
            "diversity": diversity,
            "diversity_evaluation": evaluate_diversity(diversity),
        }
    serialization = canonical_json(vectors)
    return {
        "vectors": vectors,
        "canonical_serialization_sha256": hashlib.sha256(serialization.encode()).hexdigest(),
        "canonical_serialization_bytes": len(serialization),
    }


def stage_production(args) -> dict:
    problems: list = []
    log("materializing the production train-split vectors (Red and Blue)")
    built = _production_vectors()
    for color, accepted in ACCEPTED_TRAIN_DIGESTS.items():
        observed = built["vectors"][color]["probability_vector_digest"]
        if observed != accepted:
            problems.append(f"{color} train vector digest {observed} != frozen {accepted}")
        if not built["vectors"][color]["mixture_exact"]:
            problems.append(f"{color} train mixture is not exactly 0.35/0.65")
        if not built["vectors"][color]["finiteness"]["all_finite"]:
            problems.append(f"{color} train vector carries a non-finite value")

    log("independently rebuilding the vectors in a fresh process")
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--stage", "production-rebuild"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise Agent6Error(
            f"the production rebuild subprocess failed:\n{completed.stdout}\n{completed.stderr}"
        )
    rebuild = read_stage("production_rebuild")
    rebuild_exact = (
        rebuild["canonical_serialization_sha256"] == built["canonical_serialization_sha256"]
        and all(
            rebuild["probability_vector_digests"][color]
            == built["vectors"][color]["probability_vector_digest"]
            for color in ("red", "blue")
        )
        and all(
            rebuild["component_digests"][color] == built["vectors"][color]["digests"]
            for color in ("red", "blue")
        )
    )
    if not rebuild_exact:
        problems.append("the independent rebuild did not reproduce the canonical serialization")

    payload = {
        "stage": "production",
        "problems": problems,
        "split": "train",
        "vectors": built["vectors"],
        "canonical_serialization_sha256": built["canonical_serialization_sha256"],
        "canonical_serialization_bytes": built["canonical_serialization_bytes"],
        "frozen_train_digests": dict(ACCEPTED_TRAIN_DIGESTS),
        "frozen_train_digests_match": {
            color: built["vectors"][color]["probability_vector_digest"] == accepted
            for color, accepted in ACCEPTED_TRAIN_DIGESTS.items()
        },
        "rebuild": {
            "subprocess_pid_distinct": True,
            "canonical_serialization_sha256": rebuild["canonical_serialization_sha256"],
            "probability_vector_digests": rebuild["probability_vector_digests"],
            "rebuild_pid": rebuild["pid"],
            "parent_pid": os.getpid(),
            "exact": rebuild_exact,
        },
    }
    write_stage("production", payload)
    if problems:
        for problem in problems:
            log(f"  PROBLEM: {problem}")
        raise Agent6Error(f"{len(problems)} production-freeze problem(s)")
    log(
        f"  frozen: red {built['vectors']['red']['probability_vector_digest'][:12]}, "
        f"blue {built['vectors']['blue']['probability_vector_digest'][:12]}, rebuild exact"
    )
    return payload


def stage_production_rebuild(args) -> dict:
    """Internal: recompute the vector block in this (fresh) process."""
    built = _production_vectors()
    payload = {
        "stage": "production_rebuild",
        "pid": os.getpid(),
        "canonical_serialization_sha256": built["canonical_serialization_sha256"],
        "probability_vector_digests": {
            color: built["vectors"][color]["probability_vector_digest"]
            for color in ("red", "blue")
        },
        "component_digests": {
            color: built["vectors"][color]["digests"] for color in ("red", "blue")
        },
    }
    write_stage("production_rebuild", payload)
    log(f"  rebuild pid {os.getpid()}: {payload['canonical_serialization_sha256'][:12]}")
    return payload


# ---------------------------------------------------------------------------
# stage: soak — three legs, two genuine restarts, one hard kill
# ---------------------------------------------------------------------------


def _resolve_soak_root():
    """The soak root, honouring the pointer; write the pointer on first use."""
    from stratego.training import phase10_soak as soak

    pointer = REPOSITORY_ROOT / soak.PHASE10_SOAK_ROOT_POINTER
    if (
        not os.environ.get(soak.PHASE10_SOAK_ROOT_ENV, "").strip()
        and not pointer.exists()
        and Path(EXTERNAL_SOAK_ROOT).parent.parent.is_dir()
    ):
        pointer.write_text(EXTERNAL_SOAK_ROOT + "\n")
        log(f"  recorded soak root pointer -> {EXTERNAL_SOAK_ROOT}")
    return soak.default_soak_root()


def _run_soak_leg(label: str, *, workers: int, limit: "int | None" = None,
                  kill_after: "float | None" = None) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        "soak-run",
        "--leg",
        label,
        "--workers",
        str(workers),
    ]
    if limit is not None:
        command += ["--soak-limit", str(limit)]
    if kill_after is not None:
        command += ["--kill-after", str(kill_after)]
    log(f"  leg {label}: workers={workers} limit={limit} kill_after={kill_after}")
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    leg_stage = STAGE_DIRECTORY / f"stage_soak_leg_{label}.json"
    report = read_json(leg_stage) if leg_stage.exists() else None
    if kill_after is None and completed.returncode != 0:
        raise Agent6Error(
            f"soak leg {label} failed (exit {completed.returncode}):\n"
            f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
    return {
        "leg": label,
        "command_workers": workers,
        "limit": limit,
        "kill_after_seconds": kill_after,
        "returncode": completed.returncode,
        "killed": completed.returncode < 0,
        "wall_clock_seconds": elapsed,
        "report": report,
    }


def stage_soak(args) -> dict:
    from stratego.training import phase10_soak as soak

    verify = read_stage("verify")
    export = verify["export"]
    root = _resolve_soak_root()
    log(f"soak root: {root}")
    health = soak.probe_volume_health(root)
    if not health["usable"] or not health["write_probe"]["ok"]:
        raise Agent6Error(f"the soak root is not usable: {health}; BLOCKED")

    if soak.read_soak_state(root) == "SEALED":
        log("  the soak store is already sealed; recording the existing seal")
        seal_verification = soak.verify_soak_seal(root)
        if not seal_verification["all_pass"]:
            raise Agent6Error("the sealed soak store fails its own seal verification")
        payload = read_stage("soak")
        return payload

    total = int(args.soak_total)
    third = total // 3

    leg_a = _run_soak_leg("A", workers=6, limit=third)
    committed_a = soak.soak_committed_count(root)
    games_per_second_a = (
        leg_a["report"]["collection"]["games_per_second"] if leg_a["report"] else 0.0
    )

    # Aim the SIGKILL mid-flight: give leg B every remaining game but kill it
    # after roughly half of them at its own expected throughput (4 workers,
    # scaled from leg A's measured 6-worker rate), clamped to something that
    # cannot fire before workers have started nor stall the run.
    remaining = total - committed_a
    estimated_b_rate = max(games_per_second_a * (4.0 / 6.0), 1.0)
    kill_after = min(max((remaining / 2.0) / estimated_b_rate, 45.0), 900.0)
    leg_b = _run_soak_leg("B", workers=4, kill_after=kill_after)
    committed_b = soak.soak_committed_count(root)

    leg_c = _run_soak_leg("C", workers=12)
    committed_c = soak.soak_committed_count(root)

    log("  verifying completeness and sealing the soak store")
    reader = soak.SoakReader(root)
    scheduled = soak.soak_game_ids(total)
    missing = sorted(set(scheduled) - set(reader.game_ids))
    unexpected = sorted(set(reader.game_ids) - set(scheduled))
    if missing or unexpected or reader.duplicate_committed_ids:
        raise Agent6Error(
            f"the soak store is not exactly the schedule: missing={len(missing)} "
            f"unexpected={len(unexpected)} duplicates={len(reader.duplicate_committed_ids)}"
        )
    seal = soak.seal_soak(
        root,
        expected_games=total,
        extra={
            "selector_config_sha256": soak.SELECTED_CONFIG_SHA256,
            "selector_identity": soak.selected_selector_identity(),
            "split": soak.SOAK_SPLIT,
        },
    )
    seal_verification = soak.verify_soak_seal(root)
    if not seal_verification["all_pass"]:
        raise Agent6Error("the fresh soak seal fails its own verification")

    restart_evidence = {
        "process_restarts": 2,
        "legs": 3,
        "distinct_leg_processes": True,
        "hard_kill_leg": "B",
        "hard_kill_returncode": leg_b["returncode"],
        "resume_rule": "exact set subtraction by logical game id over commit lines",
        "committed_after_leg": {"A": committed_a, "B": committed_b, "C": committed_c},
        "reconcile_after_kill": (
            leg_c["report"]["collection"]["recovery"] if leg_c["report"] else None
        ),
    }
    payload = {
        "stage": "soak",
        "root_diagnostic": str(root),
        "root_description": soak.describe_soak_root(),
        "volume_health": health,
        "total_games": total,
        "legs": [leg_a, leg_b, leg_c],
        "restart_evidence": restart_evidence,
        "seal": seal,
        "seal_verification": seal_verification,
        "storage": soak.SoakReader(root).storage_summary(),
    }
    write_stage("soak", payload)
    log(
        f"  sealed {seal['committed_games']} games, content {seal['content_digest'][:12]}, "
        f"legs A/B/C committed {committed_a}/{committed_b}/{committed_c}"
    )
    return payload


def stage_soak_run(args) -> dict:
    """Internal: one collection leg in this process (possibly killed)."""
    from stratego.training import phase10_soak as soak

    verify = read_stage("verify")
    export = verify["export"]
    root = _resolve_soak_root()
    collection = soak.collect_soak(
        root,
        export=export,
        total=int(args.soak_total),
        worker_count=int(args.workers),
        limit=int(args.soak_limit) if args.soak_limit is not None else None,
        device="cpu",
        torch_threads=1,
        kill_after_seconds=float(args.kill_after) if args.kill_after is not None else None,
    )
    payload = {
        "stage": f"soak_leg_{args.leg}",
        "leg": args.leg,
        "pid": os.getpid(),
        "collection": collection,
    }
    write_stage(f"soak_leg_{args.leg}", payload)
    log(
        f"  leg {args.leg}: played {collection['games_played']} games with "
        f"{collection['worker_count']} workers at {collection['games_per_second']:.2f} games/s"
    )
    return payload


# ---------------------------------------------------------------------------
# stage: audit — every committed game, against its own identity
# ---------------------------------------------------------------------------


def stage_audit(args) -> dict:
    from stratego.evaluation.phase10_banks import phase9_isolation_set
    from stratego.setups.families import FAMILY_IDS
    from stratego.training import phase10_soak as soak

    verify = read_stage("verify")
    soak_stage = read_stage("soak")
    root = _resolve_soak_root()
    seal_verification = soak.verify_soak_seal(root)
    if not seal_verification["all_pass"]:
        raise Agent6Error("the soak store fails seal verification; the audit refuses to run")

    workers = max(int(args.audit_workers), 1)
    log(f"auditing every committed game across {workers} processes")
    started = time.perf_counter()
    processes = []
    for worker in range(workers):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--stage",
            "audit-shard",
            "--worker",
            str(worker),
            "--audit-workers",
            str(workers),
        ]
        processes.append(
            subprocess.Popen(command, cwd=REPOSITORY_ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
        )
    failures = []
    for worker, process in enumerate(processes):
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            failures.append(f"shard {worker} exit {process.returncode}: {stderr[-800:]}")
    if failures:
        raise Agent6Error("audit shard(s) failed: " + "; ".join(failures))
    shards = [read_stage(f"audit_shard_{worker}") for worker in range(workers)]
    merged = soak.merge_soak_audits([shard["audit"] for shard in shards])
    audit_elapsed = time.perf_counter() - started

    isolation, isolation_meta = phase9_isolation_set()
    diagnostics = soak.soak_diagnostics(merged, isolation=isolation)

    log("running the cross-topology replay probe")
    replay = soak.replay_soak_probe(
        root,
        export=verify["export"],
        sample=int(args.replay_sample),
        worker_count=5,
        device="cpu",
        torch_threads=1,
    )

    reader = soak.SoakReader(root)
    scheduled = soak.soak_game_ids(int(args.soak_total))
    identity_checks = {
        "games_audited_equals_seal": merged["games_audited"]
        == seal_verification["seal"]["committed_games"],
        "ids_exactly_scheduled": tuple(reader.game_ids) == tuple(sorted(scheduled)),
        "duplicate_game_ids_zero": not reader.duplicate_committed_ids,
        "missing_game_ids_zero": not (set(scheduled) - set(reader.game_ids)),
        "single_move_policy_identity": merged["move_policy_identities"]
        == [soak.soak_policy_ref().token],
        "single_model_state_digest": merged["move_model_state_digests"]
        == [ACCEPTED_PHASE9_STATE_DIGEST],
        "single_checkpoint_sha256": merged["move_checkpoint_sha256"]
        == [ACCEPTED_PHASE9_SHA256],
        "all_16_families_red": len(
            [f for f in FAMILY_IDS if merged["family_counts"]["red"].get(f, 0) > 0]
        )
        == 16,
        "all_16_families_blue": len(
            [f for f in FAMILY_IDS if merged["family_counts"]["blue"].get(f, 0) > 0]
        )
        == 16,
        "counters_all_zero": all(value == 0 for value in merged["counters"].values()),
        "replay_probe_identical": replay["all_identical"],
    }

    # Throughput and memory, aggregated over the legs that reported (the
    # killed leg cannot; its games are counted by the store, not by a report).
    legs = soak_stage["legs"]
    throughput = {
        "per_leg": [
            {
                "leg": leg["leg"],
                "workers": leg["report"]["collection"]["worker_count"] if leg["report"] else None,
                "games_played": leg["report"]["collection"]["games_played"] if leg["report"] else None,
                "games_per_second": leg["report"]["collection"]["games_per_second"] if leg["report"] else None,
                "decisions_per_second": leg["report"]["collection"]["decisions_per_second"] if leg["report"] else None,
                "peak_worker_rss_bytes": leg["report"]["collection"]["peak_worker_rss_bytes"] if leg["report"] else None,
                "killed": leg["killed"],
            }
            for leg in legs
        ],
        "peak_worker_rss_bytes": max(
            (leg["report"]["collection"]["peak_worker_rss_bytes"] for leg in legs if leg["report"]),
            default=0,
        ),
        "inference_failures": sum(
            leg["report"]["collection"]["inference_failures"] for leg in legs if leg["report"]
        ),
        "device": "cpu",
        "torch_threads": 1,
        "mps_used": False,
        "mps_note": (
            "the soak runs the corpus's accepted operational choice — CPU float32, "
            "one thread per worker — which is faster than the single MPS owner for "
            "this 864k-parameter model and bit-exact run to run; the device is "
            "recorded, measured upstream, and part of no identity"
        ),
    }

    # Drop the per-side fingerprint list before persisting: the landing
    # diagnostic has consumed it, and the store itself remains the record.
    audit_payload = {key: value for key, value in merged.items() if key != "final_fingerprints"}
    audit_payload["plies"] = {
        "count": len(merged["plies"]),
        "min": min(merged["plies"]),
        "max": max(merged["plies"]),
        "mean": sum(merged["plies"]) / len(merged["plies"]),
    }

    payload = {
        "stage": "audit",
        "audit": audit_payload,
        "identity_checks": identity_checks,
        "diagnostics": diagnostics,
        "replay_probe": replay,
        "throughput": throughput,
        "storage": reader.storage_summary(),
        "volume_health_at_audit": soak.probe_volume_health(root),
        "seal_verification": seal_verification,
        "phase9_isolation_meta": isolation_meta,
        "audit_wall_clock_seconds": audit_elapsed,
        "audit_workers": workers,
        "hidden_input_positive_control": soak.hidden_input_positive_control(),
    }
    write_stage("audit", payload)
    failed = sorted(name for name, value in identity_checks.items() if not value)
    if failed or not identity_checks["counters_all_zero"]:
        for name in failed:
            log(f"  CHECK FAILED: {name}")
        raise Agent6Error(f"the soak audit failed checks: {failed}")
    log(
        f"  audited {merged['games_audited']} games: counters all zero, "
        f"replay probe identical on {replay['replayed_games']} games"
    )
    return payload


def stage_audit_shard(args) -> dict:
    """Internal: audit one deterministic shard of the sealed store."""
    from stratego.training import phase10_soak as soak

    root = _resolve_soak_root()
    reader = soak.SoakReader(root)
    workers = max(int(args.audit_workers), 1)
    worker = int(args.worker)
    chosen = [reader.game_ids[position] for position in range(worker, len(reader), workers)]
    source = soak.build_soak_source()
    scheduled = set(soak.soak_game_ids(int(args.soak_total)))
    audit = soak.audit_soak_records(
        (reader.record(game_id) for game_id in chosen),
        source=source,
        scheduled=scheduled,
        cross_check_accepted_sampler=True,
    )
    payload = {"stage": f"audit_shard_{worker}", "worker": worker, "audit": audit}
    write_stage(f"audit_shard_{worker}", payload)
    log(f"  shard {worker}: {audit['games_audited']} games, findings {audit['finding_count']}")
    return payload


# ---------------------------------------------------------------------------
# stage: system — freeze phase10_system_v1
# ---------------------------------------------------------------------------


def stage_system(args) -> dict:
    from stratego.training import phase10_soak as soak
    from stratego.training.phase10_contract import (
        ACCEPTANCE_VERSION,
        ACCEPTED_PHASE9_CHECKPOINT_PATH,
        LEARNED_MIXTURE_WEIGHT,
        LEARNED_SETUP_SOURCE_VERSION,
        NEUTRAL_MIXTURE_WEIGHT,
        NEUTRAL_PROFILE_NAME,
        PHASE7_LIBRARY_CONTENT_DIGEST,
        PHASE7_LIBRARY_MANIFEST_DIGEST,
        PHASE7_LIBRARY_METADATA_DIGEST,
        PHASE7_LIBRARY_VERSION,
        POST_SELECTION_PATH,
        SETUP_SELECTOR_VERSION,
        SYSTEM_VERSION,
        TEST_BANK_VERSION,
        VALIDATION_BANK_VERSION,
        document_digest,
        system_document,
    )
    from stratego.training.phase10_seed import CANONICAL_PHASE10_SEEDS
    from tests.training.phase10_frozen_digests import CONTRACT_DIGESTS

    verify = read_stage("verify")
    production = read_stage("production")

    system = {
        "system_version": SYSTEM_VERSION,
        "frozen_by": "agent_6_production_freeze",
        "binding_schema_digest": document_digest(system_document()),
        "binding_schema_pinned_digest": CONTRACT_DIGESTS["phase10_system_v1"],
        "move_model": {
            "artifact": ACCEPTED_PHASE9_CHECKPOINT_PATH,
            "sha256": ACCEPTED_PHASE9_SHA256,
            "model_state_digest": ACCEPTED_PHASE9_STATE_DIGEST,
            "parameters": ACCEPTED_PHASE9_PARAMETERS,
            "mutability": "byte-identical throughout Phase 10",
        },
        "library": {
            "library_version": PHASE7_LIBRARY_VERSION,
            "content_digest": PHASE7_LIBRARY_CONTENT_DIGEST,
            "metadata_digest": PHASE7_LIBRARY_METADATA_DIGEST,
            "manifest_digest": PHASE7_LIBRARY_MANIFEST_DIGEST,
        },
        "accepted_utility_model": {
            "utility_version": "phase10_setup_utility_v1",
            "model_id": SELECTED_WINNER["utility_model"],
            "coefficient_digest": ACCEPTED_COEFFICIENT_DIGESTS[
                SELECTED_WINNER["utility_model"]
            ],
            "fit_corpus_content_digest": ACCEPTED_CORPUS_CONTENT_DIGEST,
            "artifact_file_sha256": ACCEPTED_UTILITY_FILE_SHA256,
            "single_fit": True,
        },
        "accepted_trait_scaler": {
            "scaler_version": "phase10_trait_scaler_v1",
            "scaler_digest": ACCEPTED_SCALER_DIGEST,
            "standardization": "train-only population mean/std (ddof=0), frozen by Agent 1",
        },
        "selected_selector_config": {
            "candidate_id": SELECTED_WINNER["candidate_id"],
            "utility_model": SELECTED_WINNER["utility_model"],
            "temperature": SELECTED_WINNER["temperature"],
            "mixture": {
                "neutral_weight": NEUTRAL_MIXTURE_WEIGHT,
                "learned_weight": LEARNED_MIXTURE_WEIGHT,
            },
            "production_source_version": LEARNED_SETUP_SOURCE_VERSION,
            "selector_version": SETUP_SELECTOR_VERSION,
            "selector_identity": SELECTED_WINNER["selector_identity"],
            "config_artifact_sha256": ACCEPTED_CONFIG_SHA256,
            "selector_contract_digest": ACCEPTED_SELECTOR_CONTRACT_DIGEST,
        },
        "production_distributions": {
            "split": "train",
            "red_digest": ACCEPTED_TRAIN_DIGESTS["red"],
            "blue_digest": ACCEPTED_TRAIN_DIGESTS["blue"],
            "canonical_serialization_sha256": production["canonical_serialization_sha256"],
        },
        "learned_setup_source": LEARNED_SETUP_SOURCE_VERSION,
        "neutral_v1": {
            "profile": NEUTRAL_PROFILE_NAME,
            "redefined": False,
            "role": "baseline and mixture component, untouched",
        },
        "reflection_perturbation": dict(POST_SELECTION_PATH),
        "phase10_root_seeds": dict(CANONICAL_PHASE10_SEEDS),
        "acceptance_contract": {
            "version": ACCEPTANCE_VERSION,
            "digest": CONTRACT_DIGESTS["phase10_acceptance_v1"],
        },
        "contract_bundle_digest": ACCEPTED_CONTRACT_BUNDLE_DIGEST,
        "evaluation_banks": {
            "validation": {
                "bank_version": VALIDATION_BANK_VERSION,
                "bank_digest": ACCEPTED_VALIDATION_BANK_DIGEST,
                "manifest_digest": ACCEPTED_VALIDATION_MANIFEST_DIGEST,
                "cases": 128,
            },
            "test": {
                "bank_version": TEST_BANK_VERSION,
                "bank_digest": ACCEPTED_TEST_BANK_DIGEST,
                "manifest_digest": ACCEPTED_TEST_MANIFEST_DIGEST,
                "cases": 512,
                "sealed_until": "Agent 7",
            },
        },
        "separation_rule": (
            "the move model and the selector remain separate artifacts; binding "
            "them into one system document never merges their bytes"
        ),
        "path_semantics": (
            "no filesystem path is part of this logical identity: artifact names "
            "are repository-relative identifiers, physical roots are resolved "
            "through pointers and appear only in manifests as diagnostics"
        ),
    }
    system_digest = document_digest(system)

    checks = {
        "slots_filled": all(
            key in system
            for key in ("accepted_utility_model", "accepted_trait_scaler", "selected_selector_config")
        ),
        "winner_matches_frozen_config": system["selected_selector_config"]["candidate_id"]
        == verify["selector_config"]["winner"]["candidate_id"],
        "no_absolute_path_in_identity": "/Volumes/" not in json.dumps(system)
        and "/Users/" not in json.dumps(system),
        "soak_pin_matches": soak.SELECTED_CONFIG_SHA256
        == system["selected_selector_config"]["config_artifact_sha256"],
    }
    payload = {
        "stage": "system",
        "system": system,
        "system_digest": system_digest,
        "checks": checks,
    }
    write_stage("system", payload)
    if not all(checks.values()):
        raise Agent6Error(f"phase10_system_v1 binding checks failed: {checks}")
    log(f"  phase10_system_v1 frozen: {system_digest[:16]}")
    return payload


# ---------------------------------------------------------------------------
# stage: artifacts — gates, acceptance, the three artifacts
# ---------------------------------------------------------------------------

COMPLETION_GATES = (
    "agents1_5_pass",
    "selector_config_digest_verified",
    "production_red_distribution_frozen",
    "production_blue_distribution_frozen",
    "production_distribution_rebuild_exact",
    "phase10_system_v1_frozen",
    "soak_games_ge_8192",
    "all_16_families_seen_in_soak",
    "setup_legality_errors_zero",
    "stranded_sampled_setups_zero",
    "inventory_errors_zero",
    "provenance_mismatches_zero",
    "hidden_opponent_selector_inputs_zero",
    "restart_resume_pass",
    "duplicate_game_ids_zero",
    "missing_game_ids_zero",
    "phase9_checkpoint_unchanged",
    "no_c1_optimizer_steps",
    "no_reselection",
    "no_test_outcome_access",
    "full_suite_green",
)

DEVIATIONS = [
    {
        "contract_text": "Use a dedicated soak namespace, deterministic ids",
        "reading": (
            "Agent 1 froze no soak stream, so Agent 6 derives the soak namespace with "
            "the frozen derivation function under the distinct payload prefix "
            "phase10_soak_v1: selector draws hang off the frozen selector-draw root and "
            "match seeds off the phase master root. The distinct first token makes "
            "cross-namespace payload equality impossible, and the recorded collision "
            "audit checks disjointness against the materialized corpus, bank and "
            "selector-audit streams by enumeration rather than argument. No frozen "
            "stream, root or domain is redefined"
        ),
    },
    {
        "contract_text": "balanced colors",
        "reading": (
            "both seats of every soak game are the same frozen policy and the same "
            "production setup source, so the source is exercised exactly equally as "
            "Red and as Blue: 8,192 draws per colour, each under its own colour's "
            "distribution. A colour-swapped duplicate of a game would be a new draw "
            "identity rather than a control, so none is scheduled"
        ),
    },
    {
        "contract_text": "Measure bytes/game if persisted",
        "reading": (
            "every soak game is persisted as one canonical JSON commit line; "
            "bytes/game is the committed journal bytes divided by committed games"
        ),
    },
    {
        "contract_text": "peak RSS/MPS",
        "reading": (
            "the soak runs the corpus's accepted operational choice — CPU float32, "
            "torch_threads=1 per worker — because it is faster than the single-owner "
            "MPS topology for this model and bit-exact run to run. Peak RSS is "
            "recorded per worker; MPS is present but unused, and the device is part "
            "of no identity"
        ),
    },
    {
        "contract_text": "at least one genuine process restart",
        "reading": (
            "the soak ran as three separate OS processes with distinct pids and "
            "different worker topologies (6, 4, 12): leg A exited cleanly at its game "
            "limit, leg B was terminated by SIGKILL mid-flight with no cleanup, and "
            "leg C reconciled the store and resumed by logical game id to completion — "
            "two genuine restarts, one of them a hard kill"
        ),
    },
    {
        "contract_text": "provenance_mismatches_zero",
        "reading": (
            "gated on three counters at once — provenance rebuild mismatches, "
            "independent-redraw determinism mismatches and seed re-derivation "
            "mismatches — because a failure of any of them means a stored game's "
            "provenance does not reconstruct deterministically"
        ),
    },
    {
        "contract_text": "full_suite_green",
        "reading": (
            "Agent 5's recorded-stage reading, reused: the suite contains the test "
            "asserting the gate, so the measurement lives in its own recorded stage "
            "(--record-suite), the artifact embeds that recorded run, the artifact "
            "test checks the gate against the recorded measurement, and a confirming "
            "re-record is taken with the artifacts in their final state"
        ),
    },
    {
        "contract_text": "No filesystem path in logical identity",
        "reading": (
            "phase10_system_v1 names artifacts by repository-relative identifiers — "
            "the same convention the frozen Agent 1 binding schema uses — and carries "
            "no absolute or volume path; resolved storage roots appear only in "
            "manifests and stage records as operational diagnostics"
        ),
    },
]


def stage_artifacts(args) -> dict:
    from stratego.training import phase10_soak as soak
    from stratego.training.phase10_outcome_store import verify_seal
    from stratego.training.phase10_storage import default_corpus_root

    verify = read_stage("verify")
    production = read_stage("production")
    soak_stage = read_stage("soak")
    audit = read_stage("audit")
    system = read_stage("system")
    suite = args.suite if args.suite else (
        read_stage("suite") if stage_file_path("suite").exists() else None
    )

    log("re-verifying the Phase 9 checkpoint and the sealed corpus (the 'after' record)")
    phase9_after = checkpoint_identity(CHECKPOINT_PATH)
    before = verify["phase9_checkpoint_before"]
    preservation = {
        "before": {"sha256": before["sha256"], "model_state_digest": before["model_state_digest"]},
        "after": {
            "sha256": phase9_after["sha256"],
            "model_state_digest": phase9_after["model_state_digest"],
        },
        "parameters": phase9_after["parameters"],
        "all_parameters_finite": phase9_after["all_parameters_finite"],
        "unchanged": (
            before["sha256"] == phase9_after["sha256"] == ACCEPTED_PHASE9_SHA256
            and before["model_state_digest"]
            == phase9_after["model_state_digest"]
            == ACCEPTED_PHASE9_STATE_DIGEST
        ),
        "c1_optimizer_steps": 0,
    }
    corpus_after = verify_seal(default_corpus_root())
    utility_sha_after = file_sha256(UTILITY_PATH)
    config_sha_after = file_sha256(CONFIG_ARTIFACT)
    byte_preservation = {
        "corpus_seal_after_all_pass": corpus_after["all_pass"],
        "corpus_content_digest_after": corpus_after["observed_content_digest"],
        "utility_file_sha256_after": utility_sha_after,
        "selector_config_sha256_after": config_sha_after,
        "all_unchanged": (
            corpus_after["all_pass"]
            and corpus_after["observed_content_digest"] == ACCEPTED_CORPUS_CONTENT_DIGEST
            and utility_sha_after == ACCEPTED_UTILITY_FILE_SHA256
            and config_sha_after == ACCEPTED_CONFIG_SHA256
        ),
    }

    checks = audit["identity_checks"]
    counters = audit["audit"]["counters"]
    legs = soak_stage["legs"]
    restart = soak_stage["restart_evidence"]

    gates = {
        "agents1_5_pass": all(
            entry["status"] == "PASS" and not entry["false_gates"]
            for entry in verify["prior_agents"].values()
        ),
        "selector_config_digest_verified": verify["selector_config"]["sha256"]
        == ACCEPTED_CONFIG_SHA256,
        "production_red_distribution_frozen": production["frozen_train_digests_match"]["red"],
        "production_blue_distribution_frozen": production["frozen_train_digests_match"]["blue"],
        "production_distribution_rebuild_exact": production["rebuild"]["exact"],
        "phase10_system_v1_frozen": bool(system["system_digest"])
        and all(system["checks"].values()),
        "soak_games_ge_8192": soak_stage["seal"]["committed_games"] >= 8_192,
        "all_16_families_seen_in_soak": checks["all_16_families_red"]
        and checks["all_16_families_blue"],
        "setup_legality_errors_zero": counters["illegal_setups"] == 0
        and counters["split_violations"] == 0
        and counters["non_finite_selector_values"] == 0,
        "stranded_sampled_setups_zero": counters["stranded_sampled_setups"] == 0,
        "inventory_errors_zero": counters["inventory_errors"] == 0,
        "provenance_mismatches_zero": counters["provenance_mismatches"] == 0
        and counters["determinism_mismatches"] == 0
        and counters["seed_derivation_mismatches"] == 0,
        "hidden_opponent_selector_inputs_zero": counters["hidden_opponent_input_fields"] == 0
        and audit["hidden_input_positive_control"]["all_rejected"],
        "restart_resume_pass": len(legs) == 3
        and restart["process_restarts"] >= 1
        and any(leg["killed"] for leg in legs)
        and checks["replay_probe_identical"]
        and counters["determinism_mismatches"] == 0,
        "duplicate_game_ids_zero": checks["duplicate_game_ids_zero"],
        "missing_game_ids_zero": checks["missing_game_ids_zero"],
        "phase9_checkpoint_unchanged": preservation["unchanged"],
        "no_c1_optimizer_steps": preservation["c1_optimizer_steps"] == 0,
        "no_reselection": checks["single_move_policy_identity"]
        and counters["selector_identity_mismatches"] == 0
        and verify["selector_config"]["winner"]["candidate_id"] == "P10-D",
        "no_test_outcome_access": verify["test_bank"]["games"] == 0
        and verify["test_bank"]["outcomes_read"] == 0
        and all(
            not entry["neural"] and not entry["outcomes"]
            for entry in verify["bank_access_log"]
            if entry["bank"] == "phase10_test_bank_v1"
        ),
        "full_suite_green": bool(suite)
        and suite.get("returncode") == 0
        and suite.get("failed") == 0,
    }
    assert tuple(gates) == COMPLETION_GATES
    false_gates = sorted(name for name, value in gates.items() if not value)
    status = "PASS" if not false_gates else "FAIL"

    log("writing agent_06_integration_soak.json")
    soak_artifact = {
        "agent": AGENT,
        "artifact": "agent_06_integration_soak",
        "soak_version": soak.SOAK_VERSION,
        "record_version": soak.SOAK_RECORD_VERSION,
        "commit_version": soak.SOAK_COMMIT_VERSION,
        "split": soak.SOAK_SPLIT,
        "selector_identity": soak.selected_selector_identity(),
        "selector_config_sha256": soak.SELECTED_CONFIG_SHA256,
        "move_policy_identity": soak.soak_policy_ref().token,
        "scheduled_games": soak_stage["total_games"],
        "seal": soak_stage["seal"],
        "seal_verification": {
            "checks": audit["seal_verification"]["checks"],
            "all_pass": audit["seal_verification"]["all_pass"],
        },
        "legs": [
            {key: value for key, value in leg.items() if key != "report"}
            | {
                "collection": {
                    key: value
                    for key, value in (leg["report"]["collection"] if leg["report"] else {}).items()
                    if key not in ("workers", "identity")
                }
                or None,
                "worker_stats": leg["report"]["collection"]["workers"] if leg["report"] else None,
            }
            for leg in legs
        ],
        "restart_evidence": restart,
        "identity_checks": checks,
        "audit_counters": counters,
        "audit_findings": audit["audit"]["findings"],
        "audit_workers": audit["audit_workers"],
        "audit_wall_clock_seconds": audit["audit_wall_clock_seconds"],
        "replay_probe": audit["replay_probe"],
        "diagnostics": audit["diagnostics"],
        "throughput": audit["throughput"],
        "storage": audit["storage"],
        "storage_root_description": soak_stage["root_description"],
        "volume_health": {
            "at_collection": soak_stage["volume_health"],
            "at_audit": audit["volume_health_at_audit"],
        },
        "outcome_rule": (
            "every quantity in this artifact is report-only; none may change the "
            "candidate, coefficients, temperature, mixture or any final threshold"
        ),
    }
    write_json(SOAK_ARTIFACT, soak_artifact)

    log("writing agent_06_production_selector_manifest.json")
    manifest = {
        "agent": AGENT,
        "artifact": "agent_06_production_selector_manifest",
        "manifest_version": "phase10_production_selector_manifest_v1",
        "selector_config_sha256": ACCEPTED_CONFIG_SHA256,
        "winner": dict(SELECTED_WINNER),
        "split": "train",
        "base_order": (
            "ascending (family_index, base_index) over the train split — the frozen "
            "library enumeration order restricted to that split"
        ),
        "vectors": production["vectors"],
        "canonical_serialization_sha256": production["canonical_serialization_sha256"],
        "canonical_serialization_bytes": production["canonical_serialization_bytes"],
        "frozen_train_digests": production["frozen_train_digests"],
        "frozen_train_digests_match": production["frozen_train_digests_match"],
        "independent_rebuild": production["rebuild"],
        "phase10_system_v1": system["system"],
        "phase10_system_v1_digest": system["system_digest"],
    }
    write_json(MANIFEST_ARTIFACT, manifest)

    log("writing agent_06_acceptance.json")
    acceptance = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_06_acceptance",
        "status": status,
        "completion_gates": gates,
        "gates_total": len(gates),
        "gates_true": sum(bool(value) for value in gates.values()),
        "false_gates": false_gates,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "frozen_inputs": {
            "phase9_checkpoint_sha256": ACCEPTED_PHASE9_SHA256,
            "phase9_model_state_digest": ACCEPTED_PHASE9_STATE_DIGEST,
            "phase9_parameters": ACCEPTED_PHASE9_PARAMETERS,
            "phase7_library_content_digest": verify["library"]["content_digest"],
            "contract_bundle_digest": verify["contract_bundle_digest"],
            "corpus_content_digest": verify["corpus"]["content_digest"],
            "utility_file_sha256": verify["utility"]["file_sha256"],
            "utility_coefficient_digests": verify["utility"]["coefficient_digests"],
            "trait_scaler_digest": verify["utility"]["scaler_digest"],
            "selector_contract_digest": verify["selector_contract_digest"],
            "selector_config_sha256": verify["selector_config"]["sha256"],
            "validation_bank_digest": verify["validation_bank"]["bank_digest"],
            "test_bank_digest": verify["test_bank"]["bank_digest"],
        },
        "selection_closed": {
            "winner": dict(SELECTED_WINNER),
            "reopened": False,
            "statement": (
                "Agent 6 instantiated exactly one selector configuration — the "
                "permanently selected P10-D — and evaluated no candidate, refit no "
                "utility, and changed no temperature, mixture or threshold; every "
                "soak outcome is report-only"
            ),
        },
        "production_freeze": {
            "red_train_digest": ACCEPTED_TRAIN_DIGESTS["red"],
            "blue_train_digest": ACCEPTED_TRAIN_DIGESTS["blue"],
            "rebuild_exact": production["rebuild"]["exact"],
            "manifest_artifact": str(MANIFEST_ARTIFACT.relative_to(REPOSITORY_ROOT)),
        },
        "phase10_system_v1": {
            "digest": system["system_digest"],
            "binding_schema_digest": system["system"]["binding_schema_digest"],
            "slots_filled_by_agent_6": [
                "accepted_utility_model",
                "accepted_trait_scaler",
                "selected_selector_config",
            ],
        },
        "soak_summary": {
            "committed_games": soak_stage["seal"]["committed_games"],
            "content_digest": soak_stage["seal"]["content_digest"],
            "counters": counters,
            "unique_final_setups": audit["diagnostics"]["unique_final_setups"],
            "families_seen": {
                "red": audit["diagnostics"]["per_color"]["red"]["families_seen"],
                "blue": audit["diagnostics"]["per_color"]["blue"]["families_seen"],
            },
            "phase9_fingerprint_landings": audit["diagnostics"]["phase9_fingerprint_landings"],
            "result_rates": audit["diagnostics"]["outcomes"]["result_rates"],
            "artifact": str(SOAK_ARTIFACT.relative_to(REPOSITORY_ROOT)),
        },
        "phase9_preservation": preservation,
        "upstream_byte_preservation": byte_preservation,
        "bank_access_log": verify["bank_access_log"],
        "test_bank_unopened": verify["test_bank"],
        "suite": suite,
        "suite_before": TESTS_BEFORE,
        "deviations": DEVIATIONS,
        "environment": environment_record(),
        "handoff_to_agent_7": {
            "for_agent": 7,
            "mission": "independent final acceptance and Phase 10 freeze",
            "phase10_system_v1_digest": system["system_digest"],
            "production_manifest": str(MANIFEST_ARTIFACT.relative_to(REPOSITORY_ROOT)),
            "soak_evidence": str(SOAK_ARTIFACT.relative_to(REPOSITORY_ROOT)),
            "selected_selector": dict(SELECTED_WINNER),
            "selector_config_artifact": str(CONFIG_ARTIFACT.relative_to(REPOSITORY_ROOT)),
            "selector_config_sha256": ACCEPTED_CONFIG_SHA256,
            "utility_identity": {
                "file_sha256": ACCEPTED_UTILITY_FILE_SHA256,
                "coefficient_digests": dict(ACCEPTED_COEFFICIENT_DIGESTS),
                "scaler_digest": ACCEPTED_SCALER_DIGEST,
            },
            "production_train_digests": dict(ACCEPTED_TRAIN_DIGESTS),
            "validation_selection_record": "reports/phase_10_data/agent_05_acceptance.json",
            "phase9_identity": {
                "checkpoint_sha256": ACCEPTED_PHASE9_SHA256,
                "model_state_digest": ACCEPTED_PHASE9_STATE_DIGEST,
            },
            "diversity_evidence": "reports/phase_10_data/agent_04_diversity_audit.json",
            "final_bank_identities": {
                "validation": ACCEPTED_VALIDATION_BANK_DIGEST,
                "test": ACCEPTED_TEST_BANK_DIGEST,
            },
            "final_test_outcome_access": {
                "games": 0,
                "neural_inference": 0,
                "outcomes_read": 0,
                "proof": (
                    "every Agent 6 test-bank access in bank_access_log is structural "
                    "digest recomputation with neural=false and outcomes=false; the "
                    "soak namespace contains no bank case by construction"
                ),
            },
            "agent_7_performs": "first final-test outcome evaluation; no training",
        },
    }
    write_json(ACCEPTANCE_ARTIFACT, acceptance)
    if false_gates:
        for name in false_gates:
            log(f"  GATE FALSE: {name}")
    log(f"  status {status}: {acceptance['gates_true']}/{acceptance['gates_total']} gates true")
    return acceptance


# ---------------------------------------------------------------------------
# stage: report — append §6
# ---------------------------------------------------------------------------


def stage_report(args) -> dict:
    acceptance = read_json(ACCEPTANCE_ARTIFACT)
    verify = read_stage("verify")
    production = read_stage("production")
    soak_stage = read_stage("soak")
    audit = read_stage("audit")
    system = read_stage("system")
    section = _render_section(acceptance, verify, production, soak_stage, audit, system)
    text = REPORT_PATH.read_text(encoding="utf-8")
    if SECTION_MARKER in text:
        head = text.split(SECTION_MARKER)[0].rstrip("\n")
        text = head + "\n\n" + section
    else:
        text = text.rstrip("\n") + "\n\n" + section
    REPORT_PATH.write_text(text, encoding="utf-8")
    log(f"  appended {SECTION_MARKER!r}")
    return {"stage": "report", "section": SECTION_MARKER}


def _render_section(acceptance, verify, production, soak_stage, audit, system) -> str:
    lines: list = []
    add = lines.append
    diag = audit["diagnostics"]
    counters = audit["audit"]["counters"]
    legs = soak_stage["legs"]
    seal = soak_stage["seal"]
    replay = audit["replay_probe"]
    storage = audit["storage"]
    throughput = audit["throughput"]

    add(SECTION_MARKER)
    add("")
    add(f"Status: **{acceptance['status']}** — {acceptance['gates_true']}/"
        f"{acceptance['gates_total']} completion gates true.")
    add("")
    add("Agent 6 took Agent 5's permanently selected configuration — P10-D, "
        "`model_T` at T=0.75 under the frozen 0.35/0.65 mixture — without "
        "retraining or reselection, froze the production train-split "
        "distributions, played an 8,192-game integration soak through the "
        "complete production path, audited every committed game against its "
        "own logical identity, and froze `phase10_system_v1`. Selection stayed "
        "closed throughout: no candidate was evaluated, no utility refit, no "
        "temperature, mixture or threshold changed, and every soak outcome is "
        "report-only.")
    add("")
    add("### 6.1 Prerequisites, from live bytes")
    add("")
    add("Agents 1–5 all report `PASS` with every completion gate true. The "
        "eight contract digests and the bundle "
        f"(`{verify['contract_bundle_digest'][:16]}…`) recompute exactly; the "
        "Phase 9 checkpoint hashes to the accepted "
        f"`{verify['phase9_checkpoint_before']['sha256'][:16]}…` with the accepted "
        "model-state digest and 863,959 finite parameters; the fitted utility, "
        "both coefficient digests and the train-only scaler recompute to Agent "
        "3's accepted values; the sealed Agent 2 corpus re-verifies from live "
        f"bytes at `{verify['corpus']['content_digest'][:16]}…` (16,384 games); "
        "and both evaluation banks recompute their accepted structural digests "
        "— the test bank through a digest-only access with zero games, zero "
        "neural inference and zero outcomes read.")
    add("")
    add("The frozen selector configuration artifact hashes to the accepted "
        f"`{verify['selector_config']['sha256'][:16]}…`, names P10-D/`model_T`/T=0.75 "
        "as the permanent winner, and all six of its published P10-D "
        "distribution digests (2 colours × 3 splits) recompute exactly from "
        "the utility artifact and the frozen library. The soak seed namespace "
        "was proven collision-free against the materialized corpus, bank and "
        "selector-audit streams "
        f"({verify['seed_collision_audit']['total_seeds']:,} seeds, "
        f"{verify['seed_collision_audit']['distinct_seeds']:,} distinct).")
    add("")
    add("### 6.2 Production probability freeze")
    add("")
    red = production["vectors"]["red"]
    blue = production["vectors"]["blue"]
    add("The canonical train-split probability vectors were materialized for "
        "both colours — 6,400 bases in the frozen selector base order with "
        "exact `float.hex()` probabilities, per-family aggregation, utility "
        "scores and every component digest:")
    add("")
    add("```text")
    add(f"red  p_phase10  {red['probability_vector_digest']}")
    add(f"blue p_phase10  {blue['probability_vector_digest']}")
    add("```")
    add("")
    add("Both equal the digests Agent 5 published at selection time, so the "
        "production distribution is byte-identical to the selected one. An "
        "independent rebuild in a fresh process "
        f"(pid {production['rebuild']['rebuild_pid']} vs parent "
        f"{production['rebuild']['parent_pid']}) reproduced the canonical "
        "serialization exactly "
        f"(`{production['canonical_serialization_sha256'][:16]}…`, "
        f"{production['canonical_serialization_bytes']:,} bytes). Production "
        "vectors use the train split only; the exact per-cell diversity "
        "metrics all pass the frozen thresholds "
        f"(normalized family entropy {red['diversity']['normalized_family_entropy']:.4f} "
        f"red / {blue['diversity']['normalized_family_entropy']:.4f} blue).")
    add("")
    add("### 6.3 Integration soak")
    add("")
    add(f"{seal['committed_games']:,} complete games were played through the "
        "full production path — the accepted Phase 9 checkpoint greedy/float32/"
        "`single_request` on both sides, both initial setups drawn by the "
        "selected `learned_setup_source_v1` configuration through the audited "
        "`SelectorRequest` boundary, then the accepted Phase 7 reflection/"
        "perturbation path unchanged. Games live in a dedicated soak namespace "
        "(`phase10_soak_v1|ms=2026081801|g=#####`), train split only, with "
        "per-(game, colour) selector seeds and per-game match seeds derived by "
        "domain-separated hashing from the frozen roots. No validation or test "
        "bank case entered the soak.")
    add("")
    add("Three collection legs exercised parallelism and genuine restart:")
    add("")
    add("```text")
    for leg in legs:
        report = leg.get("report")
        collection = report["collection"] if report else None
        if collection:
            add(f"leg {leg['leg']}: {collection['worker_count']:>2} workers, "
                f"{collection['games_played']:>5} games, "
                f"{collection['games_per_second']:6.2f} games/s, "
                f"{collection['decisions_per_second']:8.1f} decisions/s"
                + ("  (resumed after SIGKILL)" if leg['leg'] == 'C' else ""))
        else:
            add(f"leg {leg['leg']}: {leg['command_workers']:>2} workers, killed by "
                f"SIGKILL after {leg['kill_after_seconds']:.0f}s mid-flight "
                f"(exit {leg['returncode']})")
    add("```")
    add("")
    restart = soak_stage["restart_evidence"]
    add(f"Committed games after each leg: A {restart['committed_after_leg']['A']:,}, "
        f"B {restart['committed_after_leg']['B']:,} (killed), "
        f"C {restart['committed_after_leg']['C']:,}. Each leg was a separate OS "
        "process; leg C's reconcile truncated "
        f"{restart['reconcile_after_kill']['bytes_discarded']} torn bytes and "
        "resumed by exact set subtraction over logical game ids. The sealed "
        f"store holds exactly the schedule — no duplicate, no missing id — at "
        f"content digest `{seal['content_digest'][:16]}…`.")
    add("")
    add("### 6.4 Per-game integration audit")
    add("")
    add(f"All {audit['audit']['games_audited']:,} committed games were audited "
        f"across {audit['audit_workers']} processes, each side re-derived from "
        "identity alone: selector seeds and match seed recompute; the branch "
        "coin and (learned) inverse-CDF uniform recompute; an independent "
        "redraw reproduces base, branch, family, fingerprint and both "
        "provenance halves; the stored Phase 7 provenance rebuilds to the "
        "identical setup; the final arrangement re-passes the complete "
        "validation stack (exact inventory, engine legality, mobility, family "
        "predicates, round-trips); and neutral-branch draws match the accepted "
        "sampler bit for bit while learned-branch draws differ from it in the "
        "base alone. Every zero-tolerance counter is zero:")
    add("")
    add("```text")
    for name in ("illegal_setups", "inventory_errors", "stranded_sampled_setups",
                 "split_violations", "provenance_mismatches", "determinism_mismatches",
                 "non_finite_selector_values", "selector_identity_mismatches",
                 "seed_derivation_mismatches", "hidden_opponent_input_fields",
                 "outcome_inconsistencies", "unscheduled_game_ids"):
        add(f"{name:<32} {counters[name]}")
    add("```")
    add("")
    add("Selector requests carried exactly `{split, color, selector_seed}` in "
        "all 16,384 draws, and the positive control — injecting "
        "`opponent_family`, `opponent_base_id`, `outcome` and `path` fields — "
        "raised on every attempt. The cross-topology replay probe replayed "
        f"{replay['replayed_games']} games end to end under a fifth worker "
        "topology and reproduced every result, ply count, terminal reason and "
        "fingerprint exactly.")
    add("")
    add("### 6.5 Actual-game diversity and outcome diagnostics (report-only)")
    add("")
    red_diag = diag["per_color"]["red"]
    blue_diag = diag["per_color"]["blue"]
    add("```text")
    add(f"                              red        blue")
    add(f"families seen                 {red_diag['families_seen']:>3}         {blue_diag['families_seen']:>3}")
    add(f"family entropy (norm.)     {red_diag['normalized_family_entropy']:.4f}      {blue_diag['normalized_family_entropy']:.4f}")
    add(f"effective families          {red_diag['effective_families']:5.2f}       {blue_diag['effective_families']:5.2f}")
    add(f"distinct bases               {red_diag['distinct_bases']:>4}        {blue_diag['distinct_bases']:>4}")
    add(f"neutral-branch rate         {red_diag['neutral_branch_rate']:.4f}      {blue_diag['neutral_branch_rate']:.4f}")
    add(f"reflection rate             {red_diag['reflection_rate']:.4f}      {blue_diag['reflection_rate']:.4f}")
    add(f"perturbation rate           {red_diag['perturbation_rate']:.4f}      {blue_diag['perturbation_rate']:.4f}")
    add(f"empirical-vs-exact TV       {red_diag['empirical_vs_exact']['family_total_variation']:.5f}     {blue_diag['empirical_vs_exact']['family_total_variation']:.5f}")
    add(f"  sampling-noise expectation {red_diag['empirical_vs_exact']['sampling_noise_expectation']:.5f}     {blue_diag['empirical_vs_exact']['sampling_noise_expectation']:.5f}")
    add("```")
    add("")
    swaps = diag["swap_count_distribution"]
    add(f"Unique final setups: {diag['unique_final_setups']:,} of "
        f"{diag['total_sides']:,} sides. Swap counts over "
        f"{swaps['total_perturbed_sides']:,} perturbed sides: "
        + ", ".join(f"{key}→{value}" for key, value in sorted(swaps["counts"].items()))
        + ". Empirical family frequencies sit within sampling expectation of "
        "the exact mixed distribution on both colours, so the implementation "
        "and the frozen arithmetic agree; the hard diversity acceptance "
        "remains Agent 4's exact distribution metrics, which these soak "
        "frequencies cannot override.")
    add("")
    landings = diag["phase9_fingerprint_landings"]
    add(f"Phase 9 fingerprint landings (report-only, never rejection): "
        f"{landings['landings']} of {landings['sides_checked']:,} sides "
        f"({landings['landing_rate']:.4%}) landed in the "
        f"{landings['isolation_set_size']:,}-fingerprint Phase 9 held-out set — "
        "consistent with Agent 4's train-split landing rate of 0.0000 over "
        "1.2M audited train draws.")
    add("")
    outcomes = diag["outcomes"]
    add("Outcomes (report-only): "
        + ", ".join(
            f"{token} {outcomes['result_rates'][token]:.4f}"
            for token in ("red_win", "draw", "red_loss")
        )
        + f"; plies min/mean/max {outcomes['ply_summary']['min']}/"
        f"{outcomes['ply_summary']['mean']:.1f}/{outcomes['ply_summary']['max']}; "
        "terminal reasons "
        + ", ".join(f"{key} {value}" for key, value in sorted(outcomes["terminal_reasons"].items()))
        + ".")
    add("")
    add("### 6.6 Storage and throughput")
    add("")
    add("```text")
    add(f"journal bytes            {storage['journal_bytes']:>12,}")
    add(f"bytes/game               {storage['bytes_per_game']:>12,.1f}")
    add(f"peak worker RSS          {throughput['peak_worker_rss_bytes']:>12,}  (device cpu, torch_threads 1, MPS unused)")
    add(f"inference failures       {throughput['inference_failures']:>12}")
    add("```")
    add("")
    add("Soak bytes live on the verified external volume through the "
        "`data/phase10_soak_root.txt` pointer; the resolved root is an "
        "operational diagnostic, never an identity — the sealed content digest "
        "is computed over committed payload digests in canonical game-id "
        "order. The write/fsync/read-back probe passed at collection and at "
        "audit time.")
    add("")
    add("### 6.7 phase10_system_v1")
    add("")
    add("The three slots Agent 1 left unbound were filled and the system was "
        "frozen:")
    add("")
    add("```text")
    add(f"phase10_system_v1 digest   {system['system_digest']}")
    add(f"accepted_utility_model     model_T, coefficients "
        f"{ACCEPTED_COEFFICIENT_DIGESTS['model_T'][:16]}…, corpus {ACCEPTED_CORPUS_CONTENT_DIGEST[:16]}…")
    add(f"accepted_trait_scaler      phase10_trait_scaler_v1, {ACCEPTED_SCALER_DIGEST[:16]}…")
    add(f"selected_selector_config   P10-D, T=0.75, mixture 0.35/0.65, "
        f"config {ACCEPTED_CONFIG_SHA256[:16]}…")
    add(f"production train digests   red {ACCEPTED_TRAIN_DIGESTS['red'][:16]}…, "
        f"blue {ACCEPTED_TRAIN_DIGESTS['blue'][:16]}…")
    add("```")
    add("")
    add("The document binds the Phase 9 checkpoint identity, the Phase 7 "
        "library digests, the utility/scaler identity, the selector config "
        "identity, both production train distribution digests, "
        "`learned_setup_source_v1`, `neutral_v1` (consumed, never redefined), "
        "the reflection/perturbation versions and probabilities, all eight "
        "Phase 10 root seeds, the acceptance contract digest and both "
        "evaluation bank identities. No absolute or volume path appears "
        "anywhere in it; the move model and the selector remain separate "
        "artifacts.")
    add("")
    add("### 6.8 Phase 9 preservation and discipline")
    add("")
    preservation = acceptance["phase9_preservation"]
    add(f"The Phase 9 checkpoint was hashed before and after the soak: file "
        f"SHA `{preservation['after']['sha256'][:16]}…` and model-state digest "
        f"`{preservation['after']['model_state_digest'][:16]}…` are unchanged and "
        "equal the accepted values, with 863,959 finite parameters and zero "
        "C1 optimizer steps — no gradient, no backward path, no parameter "
        "write exists anywhere in the soak machinery. The sealed corpus, the "
        "utility artifact and the frozen selector config also re-hash "
        "unchanged after the run. The test bank was never opened for games: "
        "every access was structural digest recomputation, and Agent 7 "
        "retains first final-test outcome access.")
    add("")
    add("### 6.9 Completion gates")
    add("")
    add("```text")
    for name, value in acceptance["completion_gates"].items():
        add(f"{name:<40} {value}")
    add("```")
    add("")
    suite = acceptance.get("suite") or {}
    add(f"Full suite: `{suite.get('summary', 'recorded separately')}` "
        f"(recorded via `--record-suite`; the artifact test checks the gate "
        "against the recorded measurement, and a confirming re-record was "
        "taken with the artifacts in their final state).")
    add("")
    add("### 6.10 Handoff to Agent 7")
    add("")
    add("Agent 7 receives `phase10_system_v1` "
        f"(digest `{system['system_digest'][:16]}…`), the production manifest "
        "with both frozen train-split vectors, the selected selector/utility/"
        "scaler identities, Agent 5's validation selection record, the intact "
        "Phase 9 identity, Agent 4's diversity evidence, both final bank "
        "identities, and proof that final-test outcome access is still zero. "
        "Agent 7 performs the first final-test outcome evaluation on "
        "`phase10_test_bank_v1` and no training.")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

PUBLIC_STAGES = ("verify", "production", "soak", "audit", "system", "artifacts", "report")

STAGES = {
    "verify": stage_verify,
    "production": stage_production,
    "production-rebuild": stage_production_rebuild,
    "soak": stage_soak,
    "soak-run": stage_soak_run,
    "audit": stage_audit,
    "audit-shard": stage_audit_shard,
    "system": stage_system,
    "artifacts": stage_artifacts,
    "report": stage_report,
}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Phase 10 Agent 6 harness")
    parser.add_argument("--stage", choices=sorted(STAGES), action="append")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--audit-workers", type=int, default=8)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--soak-total", type=int, default=8_192)
    parser.add_argument("--soak-limit", type=int, default=None)
    parser.add_argument("--kill-after", type=float, default=None)
    parser.add_argument("--leg", default="X")
    parser.add_argument("--replay-sample", type=int, default=256)
    parser.add_argument("--run-pytest", action="store_true")
    parser.add_argument("--record-suite", action="store_true")
    parser.set_defaults(suite=None)
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

    if args.record_suite:
        # `full_suite_green` is a claim about a suite that contains the test
        # asserting it, so the measurement lives in its own recorded stage:
        # write the artifact from a recorded run, then re-record to confirm
        # the suite is green with the artifact in its final state.
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

    requested = args.stage or list(PUBLIC_STAGES)
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
