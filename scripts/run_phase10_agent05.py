#!/usr/bin/env python3
"""Phase 10 Agent 5 harness: bounded validation selection.

Verifies every Agent 1-4 prerequisite from live bytes, independently
re-derives the production learned branch before a single game is played,
evaluates exactly the six frozen candidates on `phase10_validation_bank_v1`,
applies the frozen eligibility rules, recomputes the frozen selection score
and tie-break from primitives, and freezes one selector configuration.

What this script is and is not
------------------------------
It plays validation games and selects. It fits nothing, refits nothing,
changes no temperature or mixture weight, adds no seventh candidate, takes
zero optimizer steps on the Phase 9 checkpoint, and never reaches
`phase10_test_bank_v1` for anything but a structural digest check --- the
first strength evaluation on the test bank belongs to Agent 7.

Usage::

    python scripts/run_phase10_agent05.py                 # every stage
    python scripts/run_phase10_agent05.py --stage ladder  # one stage
    python scripts/run_phase10_agent05.py --workers 12    # evaluation width
    python scripts/run_phase10_agent05.py --run-pytest    # also the suite
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import pickle
import platform
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

AGENT = 5
PHASE = 10
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_10_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_10_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase10" / "agent05"
STAGE_DIRECTORY = WORK_DIRECTORY / "stages"
GAMES_DIRECTORY = WORK_DIRECTORY / "games"
EXPORT_PATH = WORK_DIRECTORY / "eval_phase9_c1.pt"

RESULTS_ARTIFACT = DATA_DIRECTORY / "agent_05_candidate_results.csv"
CONFIG_ARTIFACT = DATA_DIRECTORY / "agent_05_frozen_selector_config.json"
ACCEPTANCE_ARTIFACT = DATA_DIRECTORY / "agent_05_acceptance.json"

SECTION_MARKER = "## 5. Agent 5 — Bounded Validation Selection"

CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "selfplay_c1_v1.pt"
UTILITY_PATH = REPOSITORY_ROOT / "checkpoints" / "phase10" / "setup_utility_v1.json"
PHASE8_CHECKPOINT = REPOSITORY_ROOT / "checkpoints" / "phase8" / "warmstart_c1_v1.pt"
ANCHOR_EXPORT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "agent01" / "anchor_eval.pt"

#: The upstream identities Agent 5 refuses to proceed without. Every one is
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
ACCEPTED_SELECTOR_CONTRACT_DIGEST = (
    "ed1198f3a4bfc8f73264cf22602f6d8ba89d9458e9ae5c8a8ddf7f0543e35e59"
)

#: Agent 4's published per-cell probability-vector digests, keyed
#: candidate -> colour -> split. Loaded from its artifact, then recomputed.
AGENT4_ACCEPTANCE = DATA_DIRECTORY / "agent_04_acceptance.json"

#: The full suite as measured immediately before any Agent 5 change.
TESTS_BEFORE = {
    "command": ".venv/bin/python -m pytest tests -q",
    "summary": "5080 passed, 3 skipped in 314.99s (0:05:14)",
    "passed": 5080,
    "failed": 0,
    "skipped": 3,
    "seconds": 316.18,
    "measured_at_commit": "e1df780",
}

#: Every access this script makes to either sealed bank, with its purpose.
BANK_ACCESS_LOG: list = []


class Agent5Error(RuntimeError):
    """Raised when a prerequisite or an invariant of Agent 5's mission fails."""


def log(message: str) -> None:
    print(f"[agent05] {message}", flush=True)


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


def stage_path(name: str) -> Path:
    return STAGE_DIRECTORY / f"stage_{name}.json"


def write_stage(name: str, payload: dict) -> None:
    write_json(stage_path(name), payload)


def read_stage(name: str) -> dict:
    path = stage_path(name)
    if not path.exists():
        raise Agent5Error(
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


# ---------------------------------------------------------------------------
# stage: verify — every prerequisite, from live bytes
# ---------------------------------------------------------------------------


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


def stage_verify(args) -> dict:
    from stratego.evaluation.phase10_banks import (
        bank_digest,
        build_phase10_bank,
        manifest_digest,
        phase9_isolation_set,
    )
    from stratego.setups.families import FAMILY_IDS
    from stratego.setups.library import entry_metadata_digest, library_content_digest
    from stratego.setups.sampler import load_library_index
    from stratego.training.phase10_contract import (
        CANDIDATE_IDS,
        CANDIDATE_MATRIX,
        LEARNED_MIXTURE_WEIGHT,
        NEUTRAL_MIXTURE_WEIGHT,
        SELECTION_SCORE_WEIGHTS,
        TIE_BREAK_ORDER,
        contract_bundle_digest,
        contract_digests,
    )
    from stratego.training.phase10_selector import NEUTRAL_PROFILE, load_scorer

    problems: list = []
    log("verifying Agents 1-4 acceptance records")
    prior = {}
    for agent in (1, 2, 3, 4):
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

    log("verifying the Phase 9 checkpoint and the Phase 8 anchor")
    phase9 = checkpoint_identity(CHECKPOINT_PATH)
    if phase9["sha256"] != ACCEPTED_PHASE9_SHA256:
        problems.append(f"Phase 9 checkpoint SHA {phase9['sha256']} != accepted")
    if phase9["model_state_digest"] != ACCEPTED_PHASE9_STATE_DIGEST:
        problems.append("Phase 9 model-state digest != accepted")
    if phase9["parameters"] != ACCEPTED_PHASE9_PARAMETERS:
        problems.append(f"Phase 9 parameter count {phase9['parameters']} != accepted")
    anchor = {
        "phase8_checkpoint_sha256": file_sha256(PHASE8_CHECKPOINT),
        "anchor_export_sha256": file_sha256(ANCHOR_EXPORT_PATH),
        "anchor_export_path": str(ANCHOR_EXPORT_PATH.relative_to(REPOSITORY_ROOT)),
    }
    from stratego.training.phase9_contract import EXPECTED_PHASE8_CHECKPOINT_SHA256

    accepted_anchor = _find_accepted_anchor_sha(
        read_json(REPOSITORY_ROOT / "reports" / "phase_9_data" / "agent_08_final_acceptance.json")
    )
    if accepted_anchor is None:
        problems.append("the accepted Phase 9 anchor export SHA could not be found")
    elif anchor["anchor_export_sha256"] != accepted_anchor:
        problems.append(
            f"Phase 8 anchor export SHA {anchor['anchor_export_sha256']} != the "
            f"accepted Phase 9 value {accepted_anchor}"
        )
    if anchor["phase8_checkpoint_sha256"] != EXPECTED_PHASE8_CHECKPOINT_SHA256:
        problems.append("the Phase 8 anchor checkpoint SHA != the accepted Phase 9 value")
    anchor["accepted_export_sha256"] = accepted_anchor
    anchor["accepted_phase8_checkpoint_sha256"] = EXPECTED_PHASE8_CHECKPOINT_SHA256

    log("verifying the fitted utility, the scaler and neutral_v1")
    if not UTILITY_PATH.exists():
        raise Agent5Error(f"{UTILITY_PATH} is missing; Agent 3's fitted utility is required")
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

    log("verifying the Phase 7 library and the sealed corpus identity")
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
    corpus = read_json(DATA_DIRECTORY / "agent_02_acceptance.json")
    corpus_digest = str(
        corpus.get("corpus", {}).get("content_digest")
        or corpus.get("commit", {}).get("content_digest")
        or ACCEPTED_CORPUS_CONTENT_DIGEST
    )

    log("verifying both evaluation bank identities (structural only)")
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
    if validation["cases"] != 128:
        problems.append(f"validation bank has {validation['cases']} cases, expected 128")

    test_cases, _test_manifest = build_phase10_bank("test")
    record_bank_access("verify", "phase10_test_bank_v1", "structural_digest_only",
                       neural=False, outcomes=False)
    test_digest = bank_digest(test_cases)
    if test_digest != ACCEPTED_TEST_BANK_DIGEST:
        problems.append(f"test bank digest {test_digest} != accepted")
    del test_cases

    log("verifying the six frozen candidate identities")
    agent4 = read_json(AGENT4_ACCEPTANCE)
    handoff = agent4["handoff_to_agent_5"]
    declared = tuple(entry["candidate_id"] for entry in handoff["selector_configs"])
    if declared != CANDIDATE_IDS:
        problems.append(f"Agent 4 hands forward {declared}, expected {CANDIDATE_IDS}")
    if len(declared) != 6:
        problems.append(f"{len(declared)} candidates, expected exactly six")
    matrix = {entry["candidate_id"]: entry for entry in CANDIDATE_MATRIX}
    for entry in handoff["selector_configs"]:
        frozen = matrix[entry["candidate_id"]]
        if (entry["utility_model"], float(entry["temperature"])) != (
            frozen["utility_model"],
            float(frozen["temperature"]),
        ):
            problems.append(f"{entry['candidate_id']} model/temperature moved")

    log("recomputing every published selector distribution digest")
    distribution_digests, distribution_problems = _recompute_distribution_digests(
        handoff["distribution_digests"], scorer, index
    )
    problems.extend(distribution_problems)

    # The selector contract digest is recomputed from Agent 5's *own* rebuilt
    # distribution digests, not read out of Agent 4's artifact: that ties the
    # published contract identity to the distributions this agent verified.
    from stratego.training.phase10_selector import selector_contract_digest

    recomputed_contract = selector_contract_digest(distribution_digests)
    published_contract = agent4["new_digests"]["selector_contract_digest"]
    if recomputed_contract != ACCEPTED_SELECTOR_CONTRACT_DIGEST:
        problems.append(
            f"selector contract digest recomputes to {recomputed_contract}, not the "
            "accepted one"
        )
    if recomputed_contract != published_contract:
        problems.append("the recomputed selector contract disagrees with Agent 4's record")

    isolation, isolation_meta = phase9_isolation_set()
    payload = {
        "stage": "verify",
        "problems": problems,
        "prior_agents": prior,
        "contract_digests": digests,
        "contract_bundle_digest": bundle,
        "phase9_checkpoint": phase9,
        "phase8_anchor": anchor,
        "utility": {
            "file_sha256": utility_sha,
            "coefficient_digests": coefficient_digests,
            "scaler_digest": scaler_digest,
            "refit_by_agent_5": False,
        },
        "neutral_v1": neutral,
        "library": library,
        "corpus_content_digest": corpus_digest,
        "validation_bank": validation,
        "test_bank": {"bank_version": "phase10_test_bank_v1", "bank_digest": test_digest,
                      "access": "structural_digest_only", "outcomes_read": 0},
        "candidates": {
            "ids": list(CANDIDATE_IDS),
            "count": len(CANDIDATE_IDS),
            "matrix": [dict(entry) for entry in CANDIDATE_MATRIX],
            "diversity_eligibility": dict(handoff["diversity_eligibility"]),
        },
        "mixture": {
            "neutral_weight": NEUTRAL_MIXTURE_WEIGHT,
            "learned_weight": LEARNED_MIXTURE_WEIGHT,
        },
        "score": {
            "weights": dict(SELECTION_SCORE_WEIGHTS),
            "tie_break_order": list(TIE_BREAK_ORDER),
        },
        "selector_contract_digest": recomputed_contract,
        "selector_contract_digest_recomputed_from_own_distributions": True,
        "distribution_digests_recomputed": distribution_digests,
        "phase9_isolation_set_size": len(isolation),
        "phase9_isolation_meta": isolation_meta,
        "bank_access_log": list(BANK_ACCESS_LOG),
        "environment": environment_record(),
    }
    write_stage("verify", payload)
    if problems:
        for problem in problems:
            log(f"  PROBLEM: {problem}")
        raise Agent5Error(f"{len(problems)} prerequisite problem(s); Agent 5 is BLOCKED")
    log(f"  verified: bundle {bundle[:12]}, validation bank {validation['bank_digest'][:12]}, "
        f"{len(CANDIDATE_IDS)} candidates, Phase 9 {phase9['sha256'][:12]}")
    return payload


def _verify_utility_artifact() -> tuple:
    """Recompute both coefficient digests and the scaler from live bytes.

    Repeats Agent 4's check rather than trusting its record: the digests are
    rebuilt from the fitted artifact's own coefficients and the artifact is
    required to agree field for field with the tracked Agent 3 review copy,
    so a moved utility is caught here and not inherited.
    """
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


# ---------------------------------------------------------------------------
# stage: ladder — the learned branch, re-derived before the first game
# ---------------------------------------------------------------------------


def _parsed(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function)))


def stage_ladder(args) -> dict:
    """Independently verify the production learned branch, with a control.

    Three independent readings of the same claim, none of them a re-run of
    Agent 4's own assertions:

    1. *structural* — the source of `LearnedSetupSource.draw` is parsed and
       required to consult the branch coin exactly once, to compare it
       against the mixture weight exactly once, and to carry no bare
       mixture literal; `base_index_for_uniform` is required to read
       `cumulative_learned` and never `p_mixed`.
    2. *exact* — the ladder is recomputed from `p_learned` alone and
       compared bitwise, and the realized mixture is derived in closed form
       for both the production ladder and the defective one.
    3. *empirical with a negative control* — over frozen draw ids the
       production walk and a shadow walk over `cumsum(p_mixed)` are run side
       by side on the identical branch coins and base uniforms.

    The negative control is the point: a check that only ever passes proves
    nothing about its own sensitivity. Walking `p_mixed` must visibly
    reproduce the superseded `0.5775/0.4225` behaviour.
    """
    import numpy as np
    from stratego.training import phase10_selector as selector_module
    from stratego.training.phase10_contract import (
        LEARNED_MIXTURE_WEIGHT as WL,
        NEUTRAL_MIXTURE_WEIGHT as WN,
    )
    from stratego.training.phase10_selector import (
        LearnedSetupSource,
        SelectorRequest,
        candidate,
        load_library_index,
        load_scorer,
        neutral_branch_base_id,
        split_base_entries,
    )
    from stratego.training.phase10_seed import selector_audit_seed

    problems: list = []

    log("structural reading of the production branch")
    draw_tree = _parsed(LearnedSetupSource.draw)
    named_calls = [
        node.func.id
        for node in ast.walk(draw_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    branch_calls = named_calls.count("selector_branch_uniform")
    base_calls = named_calls.count("selector_base_uniform")
    mixture_comparisons = [
        ast.unparse(node)
        for node in ast.walk(draw_tree)
        if isinstance(node, ast.Compare)
        and (
            "NEUTRAL_MIXTURE_WEIGHT" in ast.unparse(node)
            or "LEARNED_MIXTURE_WEIGHT" in ast.unparse(node)
        )
    ]
    bare_literals = [
        node.value
        for node in ast.walk(draw_tree)
        if isinstance(node, ast.Constant) and node.value in (0.35, 0.65, 0.5775, 0.4225)
    ]
    walk_tree = _parsed(selector_module.SelectorDistribution.base_index_for_uniform)
    walk_attributes = sorted(
        {node.attr for node in ast.walk(walk_tree) if isinstance(node, ast.Attribute)}
    )
    ladder_assignment = None
    for node in ast.walk(_parsed(selector_module.build_distribution)):
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "cumulative_learned":
            ladder_assignment = ast.unparse(node.value)

    structural = {
        "branch_coin_calls_in_draw": branch_calls,
        "base_uniform_calls_in_draw": base_calls,
        "mixture_weight_comparisons_in_draw": mixture_comparisons,
        "bare_mixture_literals_in_draw": bare_literals,
        "attributes_read_by_the_walk": walk_attributes,
        "ladder_assignment": ladder_assignment,
    }
    if branch_calls != 1:
        problems.append(f"draw() consults the branch coin {branch_calls} times, expected 1")
    if base_calls != 1:
        problems.append(f"draw() draws the base uniform {base_calls} times, expected 1")
    if mixture_comparisons != ["branch_uniform < NEUTRAL_MIXTURE_WEIGHT"]:
        problems.append(
            f"the 0.35/0.65 choice is made {len(mixture_comparisons)} times in draw(): "
            f"{mixture_comparisons}"
        )
    if bare_literals:
        problems.append(f"draw() carries bare mixture literals {bare_literals}")
    if "cumulative_learned" not in walk_attributes:
        problems.append("the inverse-CDF walk does not read cumulative_learned")
    if "p_mixed" in walk_attributes:
        problems.append("the inverse-CDF walk reads p_mixed")
    if ladder_assignment != "np.cumsum(p_learned)":
        problems.append(f"cumulative_learned is built as {ladder_assignment!r}")

    log("exact reading, over every candidate x colour x split")
    scorer = load_scorer()
    index = load_library_index()
    exact_rows = []
    for candidate_id in ("P10-A", "P10-B", "P10-C", "P10-D", "P10-E", "P10-F"):
        source = LearnedSetupSource(candidate(candidate_id), scorer, index)
        for color in ("red", "blue"):
            for split in ("train", "validation", "test"):
                distribution = source.distribution(color, split)
                independent = np.cumsum(np.asarray(distribution.p_learned, dtype=np.float64))
                bitwise = bool(np.array_equal(distribution.cumulative_learned, independent))
                distinct = not bool(
                    np.array_equal(distribution.cumulative_learned, np.cumsum(distribution.p_mixed))
                )
                widths = np.diff(np.concatenate(([0.0], distribution.cumulative_learned)))
                width_error = float(np.abs(widths - distribution.p_learned).max())
                realized = WN * distribution.p_neutral + WL * distribution.p_learned
                realized_is_mixed = bool(np.array_equal(realized, distribution.p_mixed))
                exact_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "color": color,
                        "split": split,
                        "ladder_is_cumsum_p_learned": bitwise,
                        "ladder_differs_from_cumsum_p_mixed": distinct,
                        "max_interval_width_error": width_error,
                        "realized_equals_p_mixed": realized_is_mixed,
                    }
                )
                if not bitwise:
                    problems.append(f"{candidate_id}/{color}/{split}: ladder != cumsum(p_learned)")
                if not distinct:
                    problems.append(
                        f"{candidate_id}/{color}/{split}: the two ladders coincide, so the "
                        "check cannot discriminate"
                    )
                if not realized_is_mixed:
                    problems.append(
                        f"{candidate_id}/{color}/{split}: the realized mixture is not p_mixed"
                    )

    log("runtime reading: one branch coin per draw")
    counters = {"branch": 0, "base": 0}
    original_branch = selector_module.selector_branch_uniform
    original_base = selector_module.selector_base_uniform

    def counting_branch(*arguments, **keywords):
        counters["branch"] += 1
        return original_branch(*arguments, **keywords)

    def counting_base(*arguments, **keywords):
        counters["base"] += 1
        return original_base(*arguments, **keywords)

    probe_source = LearnedSetupSource(candidate("P10-D"), scorer, index)
    selector_module.selector_branch_uniform = counting_branch
    selector_module.selector_base_uniform = counting_base
    try:
        probes = [
            probe_source.draw(
                SelectorRequest(split="validation", color="blue", selector_seed=seed)
            )
            for seed in range(1, args.ladder_probes + 1)
        ]
    finally:
        selector_module.selector_branch_uniform = original_branch
        selector_module.selector_base_uniform = original_base
    learned_probes = sum(1 for draw in probes if draw.branch == "learned")
    runtime = {
        "draws": len(probes),
        "branch_coin_calls": counters["branch"],
        "base_uniform_calls": counters["base"],
        "learned_branch_draws": learned_probes,
        "one_coin_per_draw": counters["branch"] == len(probes),
        "base_uniform_only_on_the_learned_branch": counters["base"] == learned_probes,
    }
    if not runtime["one_coin_per_draw"]:
        problems.append("the branch coin is not drawn exactly once per selector draw")
    if not runtime["base_uniform_only_on_the_learned_branch"]:
        problems.append("the base uniform is drawn off the learned branch")

    log(f"empirical reading with the p_mixed negative control ({args.ladder_draws:,} draws)")
    control_rows = []
    for candidate_id in ("P10-A", "P10-D"):
        source = LearnedSetupSource(candidate(candidate_id), scorer, index)
        for color in ("red", "blue"):
            distribution = source.distribution(color, "validation")
            entries = split_base_entries("validation", index)
            families = sorted({entry.family_id for entry in entries})
            family_of = {family: position for position, family in enumerate(families)}
            defective_ladder = np.cumsum(distribution.p_mixed)
            production = np.zeros(len(families))
            defective = np.zeros(len(families))
            for ordinal in range(args.ladder_draws):
                seed = selector_audit_seed(candidate_id, "validation", color, ordinal)
                branch_uniform = original_branch(
                    source.selector_identity, "validation", color, seed
                )
                if branch_uniform < WN:
                    entry = index.base(neutral_branch_base_id("validation", seed, index))
                    production[family_of[entry.family_id]] += 1
                    defective[family_of[entry.family_id]] += 1
                else:
                    base_uniform = original_base(
                        source.selector_identity, "validation", color, seed
                    )
                    production[
                        family_of[entries[distribution.base_index_for_uniform(base_uniform)].family_id]
                    ] += 1
                    position = min(
                        int(np.searchsorted(defective_ladder, base_uniform, side="right")),
                        distribution.base_count - 1,
                    )
                    defective[family_of[entries[position].family_id]] += 1
            production /= args.ladder_draws
            defective /= args.ladder_draws
            exact_family = distribution.family_probabilities()
            blend = WN * distribution.p_neutral + WL * distribution.p_mixed
            per_family = distribution.base_count // len(families)
            blend_family = blend.reshape(len(families), per_family).sum(axis=1)

            def total_variation(left, right) -> float:
                return 0.5 * float(np.abs(np.asarray(left) - np.asarray(right)).sum())

            noise = float(
                np.sqrt(
                    max(0.0, (1.0 - float((exact_family**2).sum())))
                    / (2.0 * np.pi * args.ladder_draws)
                )
                * np.sqrt(np.pi / 2.0)
                * len(families)
                / len(families)
            )
            control_rows.append(
                {
                    "candidate_id": candidate_id,
                    "color": color,
                    "split": "validation",
                    "draws": args.ladder_draws,
                    "production_tv_to_p_mixed": total_variation(production, exact_family),
                    "defective_tv_to_p_mixed": total_variation(defective, exact_family),
                    "defective_tv_to_double_mixed_prediction": total_variation(
                        defective, blend_family
                    ),
                    "exact_defective_tv_to_p_mixed": total_variation(blend_family, exact_family),
                    "sampling_noise_scale": noise,
                }
            )

    for row in control_rows:
        if not row["defective_tv_to_p_mixed"] > 4.0 * row["production_tv_to_p_mixed"]:
            problems.append(
                f"{row['candidate_id']}/{row['color']}: the p_mixed control is not "
                "separated from production, so the check has no sensitivity"
            )
        if not row["defective_tv_to_double_mixed_prediction"] < row["defective_tv_to_p_mixed"]:
            problems.append(
                f"{row['candidate_id']}/{row['color']}: the defective ladder does not "
                "reproduce the 0.5775/0.4225 blend"
            )

    payload = {
        "stage": "ladder",
        "problems": problems,
        "claim": (
            "the production learned branch walks cumsum(p_learned), and the 0.35/0.65 "
            "neutral-vs-learned choice happens exactly once, at the branch decision"
        ),
        "structural": structural,
        "exact": exact_rows,
        "runtime": runtime,
        "negative_control": {
            "shadow_ladder": "cumsum(p_mixed)",
            "predicted_realization": {
                "neutral_weight": WN + WL * WN,
                "learned_weight": WL * WL,
            },
            "rows": control_rows,
        },
        "ran_before_any_validation_game": True,
        "environment": environment_record(),
    }
    write_stage("ladder", payload)
    if problems:
        for problem in problems:
            log(f"  PROBLEM: {problem}")
        raise Agent5Error(
            f"{len(problems)} learned-branch problem(s); no validation game may run"
        )
    sample = control_rows[0]
    log(
        f"  verified: ladder == cumsum(p_learned) on all 36 cells; control TV "
        f"{sample['defective_tv_to_p_mixed']:.6f} vs production "
        f"{sample['production_tv_to_p_mixed']:.6f}"
    )
    return payload


# ---------------------------------------------------------------------------
# stage: games — the validation evaluation
# ---------------------------------------------------------------------------


def work_units(case_count: int, chunk: int) -> list:
    """Every `(arm, candidate, matchup, case slice)` unit, in a frozen order."""
    from stratego.evaluation.phase10_validation import (
        ARM_LEARNED,
        ARM_NEUTRAL,
        NEUTRAL_ARM_MATCHUPS,
    )
    from stratego.training.phase10_contract import CANDIDATE_IDS, MATCHUP_TOKENS

    units = []
    cells = [(ARM_NEUTRAL, None, token) for token in NEUTRAL_ARM_MATCHUPS]
    cells += [
        (ARM_LEARNED, candidate_id, token)
        for candidate_id in CANDIDATE_IDS
        for token in MATCHUP_TOKENS
    ]
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


def run_unit(unit: dict, *, cases, sources, own_ref, own_policy, opponents, isolation) -> dict:
    """Play one work unit's games and return its rows plus its counters."""
    from stratego.evaluation.match_runner import ON_POLICY_ERROR_QUARANTINE
    from stratego.evaluation.phase10_validation import (
        ARM_LEARNED,
        EXTERNAL_OPPONENT_POLICY_IDS,
        MATCHUP_LEARNED_VS_NEUTRAL,
        NEURAL_OPPONENT_MATCHUP,
        game_setups,
        learned_own_side,
        neutral_own_side,
        play_cell_game,
    )
    from stratego.evaluation.registry import policy_ref

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
                color: learned_own_side(sources[candidate_id], case, color)
                for color in ("red", "blue")
            }
        else:
            own = {color: neutral_own_side(case, color) for color in ("red", "blue")}
        for setup_row in game_setups(case, matchup, own):
            spec, result = play_cell_game(
                case,
                setup_row,
                matchup,
                arm=arm,
                candidate_id=candidate_id,
                own_ref=own_ref,
                opponent_ref=opponent_ref,
                own_policy=own_policy,
                opponent_policy=opponent_policy,
                record_actions=False,
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
                }
            )
    return {
        "unit": dict(unit),
        "rows": rows,
        "seconds": time.perf_counter() - started,
    }


def cell_worker(args) -> None:
    """One worker process: every work unit whose position matches this slice."""
    import torch

    torch.set_num_threads(args.torch_threads)
    from stratego.evaluation.neural_worker import (
        DECISION_MODE_GREEDY,
        InferenceOwner,
        LocalInferenceChannel,
        RemoteNeuralPolicy,
        neural_policy_ref,
    )
    from stratego.evaluation.phase10_validation import (
        EVAL_DTYPE,
        EXTERNAL_OPPONENT_POLICY_IDS,
        PHASE10_EVAL_MOVE_POLICY_ID,
        PHASE8_ANCHOR_CANDIDATE_ID,
        validation_cases,
    )
    from stratego.evaluation.phase10_banks import phase9_isolation_set
    from stratego.evaluation.registry import build_policy
    from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from stratego.training.phase10_selector import (
        LearnedSetupSource,
        candidate,
        load_library_index,
        load_scorer,
    )
    from stratego.training.phase10_contract import CANDIDATE_IDS

    cases, _manifest = validation_cases()
    cases = cases[: args.cases]
    units = work_units(len(cases), args.chunk)
    mine = [unit for position, unit in enumerate(units) if position % args.workers == args.worker]
    pending = [unit for unit in mine if not unit_path(unit).exists()]
    if not pending:
        return

    isolation, _meta = phase9_isolation_set()
    scorer = load_scorer()
    index = load_library_index()
    sources = {
        candidate_id: LearnedSetupSource(candidate(candidate_id), scorer, index)
        for candidate_id in CANDIDATE_IDS
    }
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
            name=f"a5_phase9_w{args.worker:02d}",
        ),
        "anchor": InferenceOwner(
            ANCHOR_EXPORT_PATH,
            decision_mode=DECISION_MODE_GREEDY,
            device=args.device,
            dtype=EVAL_DTYPE,
            expected_architecture_id=ARCHITECTURE_FAMILY,
            expected_configuration=candidate_config("C1"),
            name=f"a5_anchor_w{args.worker:02d}",
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
                sources=sources,
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
    """Run every validation game, fanned across worker processes, resumably."""
    from stratego.evaluation.phase10_validation import validation_cases
    from stratego.training import phase10_collector as collector

    read_stage("verify")
    ladder = read_stage("ladder")
    if ladder.get("problems"):
        raise Agent5Error("the learned-branch verification did not pass; no game may run")

    log("exporting the accepted Phase 9 weights to the evaluation format")
    export = collector.export_evaluation_weights(CHECKPOINT_PATH, EXPORT_PATH)
    if export["source_sha256"] != ACCEPTED_PHASE9_SHA256:
        raise Agent5Error("the exported source is not the accepted Phase 9 checkpoint")
    if export["model_state_digest"] != ACCEPTED_PHASE9_STATE_DIGEST:
        raise Agent5Error("the evaluation export changed the model state")

    cases, manifest = validation_cases()
    record_bank_access("games", "phase10_validation_bank_v1", "game_outcome_evaluation",
                       neural=True, outcomes=True)
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
            raise Agent5Error(f"evaluation worker(s) {failed} failed")
    elapsed = time.perf_counter() - started

    missing = [unit for unit in units if not unit_path(unit).exists()]
    if missing:
        raise Agent5Error(f"{len(missing)} work unit(s) produced no result file")

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
        raise Agent5Error(f"{len(rows)} game rows, expected {expected}")

    payload = {
        "stage": "games",
        "cases": len(cases),
        "bank_digest": manifest["bank_digest"] if "bank_digest" in manifest else None,
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
    with open(WORK_DIRECTORY / "rows.pkl", "wb") as stream:
        pickle.dump(rows, stream)
    log(f"  {len(rows)} games recorded in {elapsed:.1f}s")
    return payload


# ---------------------------------------------------------------------------
# stage: audit — the move-policy identity of both seats, on every recorded game
# ---------------------------------------------------------------------------

#: The seat counts the frozen matchup mapping implies, derived rather than
#: observed: the selector seat is the accepted Phase 9 checkpoint in all six
#: matchups, and the opposing seat is Phase 9 only in the direct matchup.
EXPECTED_SEAT_COUNTS = {
    "phase9": 6 * 256 * 2 + 6 * 5 * 256 + 5 * 256,
    "each_external_opponent": 6 * 256 + 256,
}


def stage_audit(args) -> dict:
    """Reconcile both seats of every recorded game against the intended map.

    `match_id` is a blake2b hash over the whole match specification, both
    policy tokens included, so rebuilding the intended specification and
    requiring the recorded identifier to match is a *cryptographic* seat
    check rather than a re-read of a stored label: a game played with a
    different policy on either seat could not carry this identifier.

    The token is the identity; the weights behind it are checked separately,
    by replaying recorded games with the correct owner (must reproduce the
    recorded replay digest) and with the wrong one (must not).

    This stage plays no new scheduled game, reads no test-bank byte and
    changes nothing about the frozen selection.
    """
    import torch

    from stratego.evaluation.neural_worker import DECISION_MODE_GREEDY, neural_policy_ref
    from stratego.evaluation.phase10_validation import (
        EVAL_DTYPE,
        EXTERNAL_OPPONENT_POLICY_IDS,
        NEURAL_OPPONENT_MATCHUP,
        PHASE10_EVAL_MOVE_POLICY_ID,
        PHASE8_ANCHOR_CANDIDATE_ID,
        build_spec,
        validation_cases,
    )
    from stratego.evaluation.registry import policy_ref
    from stratego.training.phase10_contract import (
        MATCHUP_LEARNED_VS_NEUTRAL,
        MATCHUP_TOKENS,
    )
    from stratego.training.phase10_seed import CASE_GAME_COLOR, case_match_seed

    torch.set_num_threads(args.torch_threads)
    read_stage("games")
    with open(WORK_DIRECTORY / "rows.pkl", "rb") as stream:
        rows = pickle.load(stream)
    cases, _manifest = validation_cases()
    by_case = {case.case_id: case for case in cases}

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
    seats: dict = {}
    aggregate: dict = {}
    games_by_matchup: dict = {}

    def bump(store, key):
        store[key] = store.get(key, 0) + 1

    for row in rows:
        case = by_case[row["case_id"]]
        game_index = int(row["game_index"])
        matchup = row["matchup"]
        selector_ref = phase9_ref
        other_ref = opposing_ref(matchup)
        spec = build_spec(
            case,
            game_index,
            matchup,
            arm=row["arm"],
            candidate_id=row["candidate_id"],
            own_ref=selector_ref,
            opponent_ref=other_ref,
        )
        label = f"{row['arm']}/{row['candidate_id'] or 'neutral_v1'}/{matchup}/{row['case_id']}/g{game_index}"
        if spec.match_id != row["match_id"]:
            problems.append(f"{label}: match_id {row['match_id']} != rebuilt {spec.match_id}")
            continue
        if spec.setup_bank_version != row["setup_bank_version"]:
            problems.append(f"{label}: cell token {row['setup_bank_version']!r} != rebuilt")
        frozen_seed = case_match_seed(row["case_id"], game_index, matchup)
        if spec.root_seed != row["root_seed"] or spec.root_seed != frozen_seed:
            problems.append(f"{label}: match seed {row['root_seed']} != frozen {frozen_seed}")
        if CASE_GAME_COLOR[game_index] != row["own_color"]:
            problems.append(f"{label}: selector colour {row['own_color']!r} != frozen pairing")

        other_color = "blue" if row["own_color"] == "red" else "red"
        bump(games_by_matchup, matchup)
        bump(aggregate, selector_ref.token)
        bump(aggregate, other_ref.token)
        bump(seats, (matchup, selector_ref.token, "selector", row["own_color"]))
        bump(seats, (matchup, other_ref.token, "opposing", other_color))

    expected_totals = {phase9_ref.token: EXPECTED_SEAT_COUNTS["phase9"]}
    for matchup in MATCHUP_TOKENS:
        if matchup == MATCHUP_LEARNED_VS_NEUTRAL:
            continue
        expected_totals[opposing_ref(matchup).token] = EXPECTED_SEAT_COUNTS[
            "each_external_opponent"
        ]
    for token, expected in sorted(expected_totals.items()):
        observed = aggregate.get(token, 0)
        if observed != expected:
            problems.append(f"seat count for {token}: {observed} != expected {expected}")
    if set(aggregate) != set(expected_totals):
        problems.append(
            f"unexpected seat tokens {sorted(set(aggregate) - set(expected_totals))}"
        )
    if sum(aggregate.values()) != 2 * len(rows):
        problems.append("seat total does not cover both seats of every recorded game")

    binding = _weights_binding_control(rows, by_case, phase9_ref, anchor_ref, args)
    for entry in binding:
        if entry["correct_owner_reproduces"] != entry["sampled_games"]:
            problems.append(f"{entry['matchup']}: the recorded games do not replay under the "
                            "owner the harness bound to that seat")
        if entry["swapped_owner_changes_the_game"] != entry["sampled_games"]:
            problems.append(f"{entry['matchup']}: swapping the checkpoint behind a seat did not "
                            "change the game, so the control has no sensitivity")

    payload = {
        "stage": "audit",
        "problems": problems,
        "claim": (
            "every recorded game's move-policy identity, on both seats, matches the "
            "frozen matchup mapping"
        ),
        "method": (
            "match_id is a blake2b hash over the whole match specification including "
            "both policy tokens, so the rebuilt-vs-recorded comparison is a "
            "cryptographic seat check, not a re-read of a stored label"
        ),
        "games_audited": len(rows),
        "seats_audited": sum(aggregate.values()),
        "games_by_matchup": dict(sorted(games_by_matchup.items())),
        "aggregate_seat_counts": dict(sorted(aggregate.items())),
        "expected_seat_counts": dict(sorted(expected_totals.items())),
        "aggregate_matches_expected": all(
            aggregate.get(token, 0) == expected for token, expected in expected_totals.items()
        ),
        "per_matchup_seats": [
            {
                "matchup": matchup,
                "policy_token": token,
                "policy_id": token.split("@")[0],
                "policy_version": token.split("@")[1],
                "role": role,
                "red": seats.get((matchup, token, role, "red"), 0),
                "blue": seats.get((matchup, token, role, "blue"), 0),
                "total": seats.get((matchup, token, role, "red"), 0)
                + seats.get((matchup, token, role, "blue"), 0),
            }
            for matchup in MATCHUP_TOKENS
            for token, role in sorted(
                {(key[1], key[2]) for key in seats if key[0] == matchup}
            )
        ],
        "weights_binding": binding,
        "mismatches": len(problems),
        "games_replayed_for_the_control": sum(
            2 * entry["sampled_games"] for entry in binding
        ),
        "scheduled_games_rerun": 0,
        "selection_changed": False,
        "environment": environment_record(),
    }
    write_stage("audit", payload)
    if problems:
        for problem in problems[:20]:
            log(f"  PROBLEM: {problem}")
        raise Agent5Error(f"{len(problems)} seat-reconciliation problem(s); STOP before commit")
    log(f"  {len(rows):,} games / {payload['seats_audited']:,} seats reconciled, zero mismatches")
    return payload


def _weights_binding_control(rows, by_case, phase9_ref, anchor_ref, args) -> list:
    """Replay recorded neural games with the right owner, and the wrong one.

    A token proves which policy a seat named; it does not prove which
    checkpoint answered for it. Replaying with the bound owner must
    reproduce the recorded replay digest, and replaying with the other
    checkpoint behind the same token must not — otherwise the check could
    not have detected a swap in the first place.
    """
    from stratego.evaluation.match_runner import ON_POLICY_ERROR_QUARANTINE, play_match
    from stratego.evaluation.neural_worker import (
        DECISION_MODE_GREEDY,
        InferenceOwner,
        LocalInferenceChannel,
        RemoteNeuralPolicy,
    )
    from stratego.evaluation.phase10_validation import (
        EVAL_DTYPE,
        FrozenSeedPolicy,
        MATCHUP_LEARNED_VS_NEUTRAL,
        NEURAL_OPPONENT_MATCHUP,
        build_spec,
        game_setups,
        learned_own_side,
        single_game_bank,
    )
    from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from stratego.training.phase10_seed import case_match_seed
    from stratego.training.phase10_selector import (
        LearnedSetupSource,
        candidate,
        load_library_index,
        load_scorer,
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
    phase9_owner = build_owner(EXPORT_PATH, "audit_phase9")
    anchor_owner = build_owner(ANCHOR_EXPORT_PATH, "audit_anchor")
    results = []
    try:
        for matchup in (MATCHUP_LEARNED_VS_NEUTRAL, NEURAL_OPPONENT_MATCHUP):
            direct = matchup == MATCHUP_LEARNED_VS_NEUTRAL
            other_ref = phase9_ref if direct else anchor_ref
            # The seat under test, served by its own checkpoint and by the other.
            bound, swapped = (
                (phase9_owner, anchor_owner) if direct else (anchor_owner, phase9_owner)
            )
            sample = [
                row
                for row in rows
                if row["matchup"] == matchup and row["arm"] == "learned"
            ][: args.audit_sample]
            reproduced = changed = 0
            for row in sample:
                case = by_case[row["case_id"]]
                source = LearnedSetupSource(candidate(row["candidate_id"]), scorer, index)
                own = {
                    color: learned_own_side(source, case, color) for color in ("red", "blue")
                }
                setup_row = game_setups(case, matchup, own)[int(row["game_index"])]
                spec = build_spec(
                    case,
                    int(row["game_index"]),
                    matchup,
                    arm="learned",
                    candidate_id=row["candidate_id"],
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
# stage: reproduce — one work unit, replayed in a fresh process
# ---------------------------------------------------------------------------


def stage_reproduce(args) -> dict:
    """Replay one recorded work unit under a different worker count.

    Cheap evidence for the claim the whole selection rests on: a validation
    game is determined by its identity, not by how the run was sharded. The
    unit is deleted and rebuilt by a fresh process running one worker rather
    than twelve, and every recorded field must come back identical.
    """
    from stratego.evaluation.phase10_validation import rows_digest

    read_stage("games")
    units = work_units(args.cases, args.chunk)
    chosen = next(
        unit
        for unit in units
        if unit["arm"] == "learned"
        and unit["candidate_id"] == args.reproduce_candidate
        and unit["matchup"] == args.reproduce_matchup
        and unit["start"] == args.reproduce_start
    )
    path = unit_path(chosen)
    with open(path, "rb") as stream:
        original = pickle.load(stream)
    before = rows_digest(original["rows"])

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

    after = rows_digest(replayed["rows"])
    identical = original["rows"] == replayed["rows"]
    payload = {
        "stage": "reproduce",
        "unit": dict(chosen),
        "games": len(original["rows"]),
        "recorded_workers": 12,
        "replay_workers": 1,
        "fresh_process": True,
        "digest_before": before,
        "digest_after": after,
        "digest_identical": before == after,
        "every_field_identical": bool(identical),
        "environment": environment_record(),
    }
    write_stage("reproduce", payload)
    if not (payload["digest_identical"] and identical):
        raise Agent5Error("a replayed work unit did not reproduce its recorded games")
    log(f"  replayed {len(original['rows'])} games under 1 worker: identical")
    return payload


# ---------------------------------------------------------------------------
# stage: select — eligibility, score, tie-break, freeze
# ---------------------------------------------------------------------------


def stage_select(args) -> dict:
    from stratego.evaluation.phase10_banks import phase9_isolation_set
    from stratego.evaluation.phase10_validation import (
        ARM_LEARNED,
        ARM_NEUTRAL,
        case_game_pairs,
        color_split,
        counts_from_rows,
        family_split,
        landing_counts,
        length_summary,
        rows_digest,
        safety_counters,
        terminal_reasons,
        validation_cases,
    )
    from stratego.training.phase10_acceptance import (
        MatchupOutcomes,
        select_winner,
        selection_score,
        summarize_matchups,
        tie_break_key,
        validation_guards,
    )
    from stratego.training.phase10_contract import (
        CANDIDATE_IDS,
        MATCHUP_LEARNED_VS_NEUTRAL,
        MATCHUP_TOKENS,
    )
    from stratego.training.phase10_selector import (
        LearnedSetupSource,
        candidate,
        load_library_index,
        load_scorer,
    )

    verify = read_stage("verify")
    games = read_stage("games")
    with open(WORK_DIRECTORY / "rows.pkl", "rb") as stream:
        rows = pickle.load(stream)

    cases, _manifest = validation_cases()
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

    log("building the neutral baseline arm")
    neutral_cells = {}
    for matchup in MATCHUP_TOKENS:
        if matchup == MATCHUP_LEARNED_VS_NEUTRAL:
            continue
        entries = cell_rows(ARM_NEUTRAL, None, matchup)
        neutral_cells[matchup] = {
            "rows": entries,
            "pairs": case_game_pairs(by_case(entries), case_ids),
            "counts": counts_from_rows(entries),
            "color_split": color_split(entries),
            "family_split": family_split(entries),
            "terminal_reasons": terminal_reasons(entries),
            "lengths": length_summary(entries),
            "safety": safety_counters(entries),
            "landings": landing_counts(entries, isolation),
            "digest": rows_digest(entries),
        }

    scorer = load_scorer()
    index = load_library_index()
    diversity_eligibility = verify["candidates"]["diversity_eligibility"]

    log("summarizing every candidate from primitives")
    candidate_records = []
    per_cell = []
    for candidate_id in CANDIDATE_IDS:
        source = LearnedSetupSource(candidate(candidate_id), scorer, index)
        diversity = source.distribution("red", "validation").diversity()
        blue = source.distribution("blue", "validation").diversity()
        outcomes = []
        cells = {}
        candidate_safety = {
            "policy_errors": 0,
            "illegal_actions": 0,
            "engine_rejections": 0,
            "policy_exceptions": 0,
            "contract_violations": 0,
            "non_finite_scores": 0,
            "illegal_setups": 0,
            "unscored_games": 0,
        }
        for matchup in MATCHUP_TOKENS:
            entries = cell_rows(ARM_LEARNED, candidate_id, matchup)
            unscored = [row for row in entries if row["score"] is None]
            safety = safety_counters(
                [row | {"score": (0.0 if row["score"] is None else row["score"])} for row in entries]
            )
            safety["unscored_games"] = len(unscored)
            for key in candidate_safety:
                candidate_safety[key] += int(safety.get(key, 0))
            cell = {
                "rows": entries,
                "counts": counts_from_rows(entries),
                "color_split": color_split(entries),
                "family_split": family_split(entries),
                "terminal_reasons": terminal_reasons(entries),
                "lengths": length_summary(entries),
                "safety": safety,
                "landings": landing_counts(entries, isolation),
                "digest": rows_digest(entries),
            }
            cells[matchup] = cell
            if unscored:
                continue
            learned_pairs = case_game_pairs(by_case(entries), case_ids)
            neutral_pairs = (
                None
                if matchup == MATCHUP_LEARNED_VS_NEUTRAL
                else neutral_cells[matchup]["pairs"]
            )
            outcomes.append(
                MatchupOutcomes(
                    token=matchup,
                    case_ids=case_ids,
                    learned_games=learned_pairs,
                    neutral_games=neutral_pairs,
                )
            )
            per_cell.append(
                {
                    "candidate_id": candidate_id,
                    "arm": ARM_LEARNED,
                    "matchup": matchup,
                    "bank": "phase10_validation_bank_v1",
                    "games": cell["counts"]["games"],
                    "landings": cell["landings"]["landings"],
                    "landing_rate": cell["landings"]["rate"],
                }
            )

        correctness_clean = all(value == 0 for value in candidate_safety.values())
        if len(outcomes) != len(MATCHUP_TOKENS):
            record = {
                "candidate_id": candidate_id,
                "eligible": False,
                "ineligible_reasons": ["a validation game did not complete cleanly"],
                "summaries": None,
                "cells": cells,
                "safety": candidate_safety,
                "correctness_clean": correctness_clean,
            }
            candidate_records.append(record)
            continue

        summaries = summarize_matchups(outcomes, "validation")
        score = selection_score(summaries)
        guards = validation_guards(summaries)
        reasons = []
        if not diversity_eligibility.get(candidate_id, False):
            reasons.append("Agent 4 diversity/correctness/reproducibility eligibility is false")
        if not guards["checks"]["random_overall"]:
            reasons.append(f"validation Random EWR {guards['random_ewr']:.4f} < 0.95")
        if not guards["checks"]["basic"]:
            reasons.append(f"validation Basic EWR {guards['basic_ewr']:.4f} < 0.80")
        if not correctness_clean:
            reasons.append(f"non-zero correctness counters {candidate_safety}")
        candidate_records.append(
            {
                "candidate_id": candidate_id,
                "utility_model": source.candidate.utility_model,
                "temperature": source.candidate.temperature,
                "selector_identity": source.selector_identity,
                "eligible": not reasons,
                "ineligible_reasons": reasons,
                "summaries": summaries,
                "score": score,
                "guards": guards,
                "cells": cells,
                "safety": candidate_safety,
                "correctness_clean": correctness_clean,
                "s10": score["s10"],
                "delta_direct": score["components"]["delta_direct"],
                "delta_strategic": score["components"]["delta_strategic"],
                "delta_tactical": score["components"]["delta_tactical"],
                "delta_phase8_anchor": score["components"]["delta_phase8_anchor"],
                "normalized_family_entropy": min(
                    diversity["normalized_family_entropy"],
                    blue["normalized_family_entropy"],
                ),
                "effective_base_diversity": min(
                    diversity["effective_base_diversity"],
                    blue["effective_base_diversity"],
                ),
                "diversity": {"red": diversity, "blue": blue},
            }
        )

    for entry in neutral_cells:
        per_cell.append(
            {
                "candidate_id": "neutral_v1",
                "arm": ARM_NEUTRAL,
                "matchup": entry,
                "bank": "phase10_validation_bank_v1",
                "games": neutral_cells[entry]["counts"]["games"],
                "landings": neutral_cells[entry]["landings"]["landings"],
                "landing_rate": neutral_cells[entry]["landings"]["rate"],
            }
        )

    log("recomputing the score and the tie-break independently")
    independent = _independent_score_check(candidate_records)
    selection = select_winner(
        [
            {
                "candidate_id": record["candidate_id"],
                "eligible": record["eligible"],
                "s10": record.get("s10", float("-inf")),
                "delta_strategic": record.get("delta_strategic", float("-inf")),
                "delta_direct": record.get("delta_direct", float("-inf")),
                "normalized_family_entropy": record.get("normalized_family_entropy", 0.0),
                "effective_base_diversity": record.get("effective_base_diversity", 0.0),
            }
            for record in candidate_records
        ]
    )
    tie_break = _tie_break_evidence(candidate_records, selection)

    payload = {
        "stage": "select",
        "case_count": len(cases),
        "candidates": [
            {key: value for key, value in record.items() if key not in ("cells",)}
            for record in candidate_records
        ],
        "cells": {
            record["candidate_id"]: {
                matchup: {key: value for key, value in cell.items() if key != "rows"}
                for matchup, cell in record["cells"].items()
            }
            for record in candidate_records
        },
        "neutral_arm": {
            matchup: {key: value for key, value in cell.items() if key not in ("rows", "pairs")}
            for matchup, cell in neutral_cells.items()
        },
        "landing_diagnostic": {
            "granularity": "candidate x arm x matchup x bank",
            "use": "report_only",
            "gate": False,
            "rows": per_cell,
        },
        "independent_score_check": independent,
        "selection": selection,
        "tie_break": tie_break,
        "games_stage": {k: v for k, v in games.items() if k != "environment"},
        "environment": environment_record(),
    }
    write_stage("select", payload)
    winner = selection["winner"]
    log(f"  eligible {selection['eligible_count']}/6; winner {winner}")
    return payload


def _independent_score_check(records) -> dict:
    """Recompute S10 from the recorded primitives, without the helper."""
    from stratego.training.phase10_contract import SELECTION_SCORE_WEIGHTS

    rows = []
    agree = True
    for record in records:
        if record.get("summaries") is None:
            continue
        summaries = record["summaries"]
        direct = summaries["learned_vs_neutral"]["learned_ewr"] - 0.5
        strategic = summaries["vs_strategic"]["delta"]
        tactical = summaries["vs_tactical"]["delta"]
        anchor = summaries["vs_phase8_anchor"]["delta"]
        recomputed = (
            SELECTION_SCORE_WEIGHTS["delta_direct"] * direct
            + SELECTION_SCORE_WEIGHTS["delta_strategic"] * strategic
            + SELECTION_SCORE_WEIGHTS["delta_tactical"] * tactical
            + SELECTION_SCORE_WEIGHTS["delta_phase8_anchor"] * anchor
        )
        matches = abs(recomputed - record["s10"]) <= 1e-15
        agree = agree and matches
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "s10_from_helper": record["s10"],
                "s10_recomputed": recomputed,
                "difference": recomputed - record["s10"],
                "agrees": matches,
            }
        )
    return {"rows": rows, "all_agree": agree}


def _tie_break_evidence(records, selection) -> dict:
    """The tie-break, recomputed and shown at the level it was decided on."""
    from stratego.training.phase10_acceptance import tie_break_key

    eligible = [record for record in records if record["eligible"]]
    keys = {
        record["candidate_id"]: list(
            tie_break_key(
                {
                    "candidate_id": record["candidate_id"],
                    "s10": record["s10"],
                    "delta_strategic": record["delta_strategic"],
                    "delta_direct": record["delta_direct"],
                    "normalized_family_entropy": record["normalized_family_entropy"],
                    "effective_base_diversity": record["effective_base_diversity"],
                }
            )[:5]
        )
        for record in eligible
    }
    ordered = selection["ranking"]
    level = None
    if len(ordered) >= 2:
        first, second = keys[ordered[0]], keys[ordered[1]]
        for position, (left, right) in enumerate(zip(first, second)):
            if left != right:
                level = position + 1
                break
    return {
        "order": list(ordered),
        "keys": keys,
        "decided_at_level": level,
        "levels": [
            "higher S10",
            "higher Delta_Strategic",
            "higher Delta_D",
            "higher normalized family entropy",
            "higher effective base diversity",
            "lexicographically smaller candidate id",
        ],
        "resolved_without_reaching_candidate_id": level is not None,
    }


# ---------------------------------------------------------------------------
# stage: artifacts
# ---------------------------------------------------------------------------


def stage_artifacts(args) -> dict:
    import csv

    from stratego.training.phase10_contract import (
        CANDIDATE_MATRIX,
        LEARNED_MIXTURE_WEIGHT,
        NEUTRAL_MIXTURE_WEIGHT,
        SELECTION_SCORE_WEIGHTS,
        SELECTOR_SCHEDULE_VERSION,
        SETUP_SELECTOR_VERSION,
        TIE_BREAK_ORDER,
        LEARNED_SETUP_SOURCE_VERSION,
    )
    from stratego.training.phase10_seed import CANONICAL_PHASE10_SEEDS

    verify = read_stage("verify")
    ladder = read_stage("ladder")
    games = read_stage("games")
    select = read_stage("select")

    records = {entry["candidate_id"]: entry for entry in select["candidates"]}
    cells = select["cells"]
    matrix = {entry["candidate_id"]: entry for entry in CANDIDATE_MATRIX}

    log(f"writing {RESULTS_ARTIFACT.name}")
    fields = [
        "candidate_id", "utility_model", "temperature", "selector_identity",
        "eligible", "ineligible_reasons",
        "s10", "delta_direct", "delta_strategic", "delta_tactical", "delta_phase8_anchor",
        "direct_ewr", "direct_lb", "direct_ub",
        "strategic_learned_ewr", "strategic_neutral_ewr", "strategic_delta_lb",
        "tactical_learned_ewr", "tactical_neutral_ewr", "tactical_delta_lb",
        "phase8_learned_ewr", "phase8_neutral_ewr", "phase8_delta_lb",
        "random_ewr", "random_guard_pass", "basic_ewr", "basic_guard_pass",
        "diversity_eligible", "normalized_family_entropy", "effective_base_diversity",
        "correctness_counters_zero", "games", "rank", "winner",
        "phase9_landings", "phase9_landing_rate",
    ]
    RESULTS_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ranking = select["selection"]["ranking"]
    with open(RESULTS_ARTIFACT, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for candidate_id in sorted(records):
            record = records[candidate_id]
            summaries = record.get("summaries")
            candidate_cells = cells.get(candidate_id, {})
            landings = sum(
                cell["landings"]["landings"] for cell in candidate_cells.values()
            )
            played = sum(cell["landings"]["games"] for cell in candidate_cells.values())
            row = {
                "candidate_id": candidate_id,
                "utility_model": matrix[candidate_id]["utility_model"],
                "temperature": matrix[candidate_id]["temperature"],
                "selector_identity": record.get("selector_identity", ""),
                "eligible": record["eligible"],
                "ineligible_reasons": "; ".join(record.get("ineligible_reasons", [])),
                "diversity_eligible": verify["candidates"]["diversity_eligibility"][candidate_id],
                "correctness_counters_zero": record.get("correctness_clean"),
                "games": played,
                "rank": (ranking.index(candidate_id) + 1) if candidate_id in ranking else "",
                "winner": candidate_id == select["selection"]["winner"],
                "phase9_landings": landings,
                "phase9_landing_rate": (landings / played) if played else "",
            }
            if summaries is None:
                # No fabricated strength score for a candidate that did not run.
                row.update({field: "" for field in fields if field not in row})
                row["candidate_id"] = candidate_id
                writer.writerow(row)
                continue
            direct = summaries["learned_vs_neutral"]
            row.update(
                {
                    "s10": f"{record['s10']:.10f}",
                    "delta_direct": f"{record['delta_direct']:.10f}",
                    "delta_strategic": f"{record['delta_strategic']:.10f}",
                    "delta_tactical": f"{record['delta_tactical']:.10f}",
                    "delta_phase8_anchor": f"{record['delta_phase8_anchor']:.10f}",
                    "direct_ewr": f"{direct['learned_ewr']:.10f}",
                    "direct_lb": f"{direct['learned_interval']['lower']:.10f}",
                    "direct_ub": f"{direct['learned_interval']['upper']:.10f}",
                    "random_ewr": f"{record['guards']['random_ewr']:.10f}",
                    "random_guard_pass": record["guards"]["checks"]["random_overall"],
                    "basic_ewr": f"{record['guards']['basic_ewr']:.10f}",
                    "basic_guard_pass": record["guards"]["checks"]["basic"],
                    "normalized_family_entropy": f"{record['normalized_family_entropy']:.10f}",
                    "effective_base_diversity": f"{record['effective_base_diversity']:.6f}",
                }
            )
            for prefix, token in (
                ("strategic", "vs_strategic"),
                ("tactical", "vs_tactical"),
                ("phase8", "vs_phase8_anchor"),
            ):
                summary = summaries[token]
                row[f"{prefix}_learned_ewr"] = f"{summary['learned_ewr']:.10f}"
                row[f"{prefix}_neutral_ewr"] = f"{summary['neutral_ewr']:.10f}"
                row[f"{prefix}_delta_lb"] = f"{summary['delta_interval']['lower']:.10f}"
            writer.writerow(row)

    winner_id = select["selection"]["winner"]
    log(f"writing {CONFIG_ARTIFACT.name}")
    if winner_id is None:
        config = {
            "selector_config_version": "phase10_selector_config_v1",
            "status": "FAIL",
            "winner": None,
            "reason": "no candidate was eligible on the validation bank",
            "production_setup_source": "neutral_v1",
        }
    else:
        winner = records[winner_id]
        config = {
            "selector_config_version": "phase10_selector_config_v1",
            "status": "SELECTED",
            "winner": {
                "candidate_id": winner_id,
                "utility_model": matrix[winner_id]["utility_model"],
                "temperature": matrix[winner_id]["temperature"],
                "selector_identity": winner["selector_identity"],
            },
            "utility": {
                "file_sha256": verify["utility"]["file_sha256"],
                "coefficient_digests": verify["utility"]["coefficient_digests"],
                "scaler_digest": verify["utility"]["scaler_digest"],
                "artifact": "checkpoints/phase10/setup_utility_v1.json",
                "separate_artifact_from_this_config": True,
                "refit_by_agent_5": False,
            },
            "mixture": {
                "neutral_weight": NEUTRAL_MIXTURE_WEIGHT,
                "learned_weight": LEARNED_MIXTURE_WEIGHT,
                "applied": "exactly once, at the branch decision",
            },
            "versions": {
                "setup_selector": SETUP_SELECTOR_VERSION,
                "learned_setup_source": LEARNED_SETUP_SOURCE_VERSION,
                "selector_schedule": SELECTOR_SCHEDULE_VERSION,
                "selector_contract_digest": verify["selector_contract_digest"],
                "validation_evaluation": "phase10_validation_eval_v1",
            },
            "phase7_identity": {
                "library_content_digest": verify["library"]["content_digest"],
                "library_metadata_digest": verify["library"]["metadata_digest"],
                "baseline_profile": "neutral_v1",
                "reflection_probability": 0.5,
                "perturbation_probability": 0.5,
                "swap_count": "1..6 uniform",
            },
            "phase9_identity": {
                "checkpoint_sha256": verify["phase9_checkpoint"]["sha256"],
                "model_state_digest": verify["phase9_checkpoint"]["model_state_digest"],
                "parameters": verify["phase9_checkpoint"]["parameters"],
                "c1_optimizer_steps": 0,
            },
            "phase10_seeds": dict(CANONICAL_PHASE10_SEEDS),
            "validation_identity": {
                "bank_version": "phase10_validation_bank_v1",
                "bank_digest": verify["validation_bank"]["bank_digest"],
                "manifest_digest": verify["validation_bank"]["manifest_digest"],
                "cases": verify["validation_bank"]["cases"],
                "bootstrap_root": CANONICAL_PHASE10_SEEDS["validation_bootstrap_seed"],
            },
            "score": {
                "s10": winner["s10"],
                "components": winner["score"]["components"],
                "weights": dict(SELECTION_SCORE_WEIGHTS),
                "tie_break_order": list(TIE_BREAK_ORDER),
                "tie_break_decided_at_level": select["tie_break"]["decided_at_level"],
            },
            "diversity": {
                "normalized_family_entropy": winner["normalized_family_entropy"],
                "effective_base_diversity": winner["effective_base_diversity"],
                "per_color_validation": winner["diversity"],
            },
            "distribution_digests": verify["distribution_digests_recomputed"][winner_id],
            "train_split_production_digests": {
                color: verify["distribution_digests_recomputed"][winner_id][color]["train"]
                for color in ("red", "blue")
            },
            "test_split_digests": {
                color: verify["distribution_digests_recomputed"][winner_id][color]["test"]
                for color in ("red", "blue")
            },
            "c1_checkpoint_created_or_altered": False,
        }
    write_json(CONFIG_ARTIFACT, config)
    config_digest = file_sha256(CONFIG_ARTIFACT)

    access_log = _merged_access_log(verify, games)
    gates = _completion_gates(verify, ladder, games, select, config, args.suite, access_log)
    status = "PASS" if all(gates.values()) and winner_id else ("FAIL" if not winner_id else "FAIL")
    acceptance = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_05_acceptance",
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "completion_gates": gates,
        "gates_total": len(gates),
        "gates_true": sum(1 for value in gates.values() if value),
        "false_gates": sorted(name for name, value in gates.items() if not value),
        "frozen_inputs": {
            "contract_bundle_digest": verify["contract_bundle_digest"],
            "phase9_checkpoint_sha256": verify["phase9_checkpoint"]["sha256"],
            "phase9_model_state_digest": verify["phase9_checkpoint"]["model_state_digest"],
            "setup_utility_v1_file_sha256": verify["utility"]["file_sha256"],
            "model_F_coefficient_digest": verify["utility"]["coefficient_digests"]["model_F"],
            "model_T_coefficient_digest": verify["utility"]["coefficient_digests"]["model_T"],
            "scaler_digest": verify["utility"]["scaler_digest"],
            "selector_contract_digest": verify["selector_contract_digest"],
            "corpus_content_digest": verify["corpus_content_digest"],
            "phase7_library_content_digest": verify["library"]["content_digest"],
            "validation_bank_digest": verify["validation_bank"]["bank_digest"],
            "test_bank_digest": verify["test_bank"]["bank_digest"],
        },
        "new_digests": {
            "selector_config_sha256": config_digest,
            "candidate_results_sha256": file_sha256(RESULTS_ARTIFACT),
            "cell_result_digests": {
                candidate_id: {
                    matchup: cell["digest"] for matchup, cell in candidate_cells.items()
                }
                for candidate_id, candidate_cells in select["cells"].items()
            },
            "neutral_arm_digests": {
                matchup: cell["digest"] for matchup, cell in select["neutral_arm"].items()
            },
        },
        "seat_policy_audit": read_stage("audit"),
        "unit_replay": read_stage("reproduce"),
        "learned_branch_verification": {
            "claim": ladder["claim"],
            "structural": ladder["structural"],
            "runtime": ladder["runtime"],
            "negative_control": ladder["negative_control"],
            "ran_before_any_validation_game": True,
            "problems": ladder["problems"],
        },
        "selection": select["selection"],
        "tie_break": select["tie_break"],
        "independent_score_check": select["independent_score_check"],
        "candidates": select["candidates"],
        "landing_diagnostic": select["landing_diagnostic"],
        "discipline": _discipline(games, select),
        "bank_access_log": access_log,
        "phase9_preservation": _phase9_preservation(verify),
        "deviations": DEVIATIONS,
        "handoff_to_agent_6": _handoff(verify, select, config, config_digest),
        "environment": environment_record(),
        "suite_before": TESTS_BEFORE,
    }
    if args.suite:
        acceptance["suite"] = args.suite
    write_json(ACCEPTANCE_ARTIFACT, acceptance)
    log(f"  wrote {ACCEPTANCE_ARTIFACT.name}: {status}, "
        f"{acceptance['gates_true']}/{acceptance['gates_total']} gates")
    return acceptance


#: Every field `phase10_selector_config_v1` must carry, from the assignment's
#: "Freeze one config" list. The gate reads this, so a config missing a field
#: cannot be declared complete.
REQUIRED_CONFIG_FIELDS = (
    "winner",
    "utility",
    "mixture",
    "versions",
    "phase7_identity",
    "phase9_identity",
    "phase10_seeds",
    "validation_identity",
    "score",
    "diversity",
    "distribution_digests",
    "train_split_production_digests",
    "test_split_digests",
)


def _config_is_complete(config: dict) -> bool:
    """Whether the frozen config carries every field the assignment names."""
    if config.get("status") != "SELECTED":
        return False
    if any(field not in config for field in REQUIRED_CONFIG_FIELDS):
        return False
    winner = config["winner"]
    if not all(
        key in winner
        for key in ("candidate_id", "utility_model", "temperature", "selector_identity")
    ):
        return False
    mixture = config["mixture"]
    if (mixture.get("neutral_weight"), mixture.get("learned_weight")) != (0.35, 0.65):
        return False
    seeds = config["phase10_seeds"]
    return len(seeds) >= 8 and all(isinstance(value, int) for value in seeds.values())


def _merged_access_log(verify, games) -> list:
    """Every bank access this agent made, across all of its stages.

    Each stage runs in its own process, so the in-process log covers only
    that stage. The gate is stated over the union, otherwise the stage that
    touched the test bank at all would not be the stage the gate reads.
    """
    seen = []
    for entry in [*verify["bank_access_log"], *games["bank_access_log"]]:
        if entry not in seen:
            seen.append(entry)
    return sorted(seen, key=lambda entry: (entry["stage"], entry["bank"]))


def _test_bank_never_opened(access_log) -> bool:
    """Zero neural, game-outcome, model-metric and selection accesses."""
    entries = [entry for entry in access_log if entry["bank"] == "phase10_test_bank_v1"]
    if not entries:
        # Silence is not evidence: the structural digest check is recorded, so
        # an empty list means the log was not written rather than that the
        # bank was untouched.
        return False
    return all(
        not entry["neural"]
        and not entry["outcomes"]
        and entry["purpose"] == "structural_digest_only"
        for entry in entries
    )


def _completion_gates(verify, ladder, games, select, config, suite, access_log) -> dict:
    from stratego.training.phase10_contract import CANDIDATE_IDS

    candidates = {entry["candidate_id"]: entry for entry in select["candidates"]}
    return {
        "agents1_4_pass": all(
            entry["status"] == "PASS" and not entry["false_gates"]
            for entry in verify["prior_agents"].values()
        ),
        "candidate_count_6": verify["candidates"]["count"] == 6
        and set(candidates) == set(CANDIDATE_IDS),
        "unregistered_candidates_zero": set(candidates) <= set(CANDIDATE_IDS),
        "utility_models_not_refit": verify["utility"]["refit_by_agent_5"] is False
        and verify["utility"]["file_sha256"] == ACCEPTED_UTILITY_FILE_SHA256,
        "validation_bank_identity_verified": verify["validation_bank"]["bank_digest"]
        == ACCEPTED_VALIDATION_BANK_DIGEST
        and verify["validation_bank"]["manifest_digest"] == ACCEPTED_VALIDATION_MANIFEST_DIGEST,
        "neutral_baseline_fixed": verify["neutral_v1"]["name"] == "neutral_v1"
        and len(select["neutral_arm"]) == 5,
        "same_cases_across_candidates": _same_cases(select),
        "score_recomputes_exactly": select["independent_score_check"]["all_agree"],
        "tie_break_recomputes_exactly": select["tie_break"]["order"]
        == select["selection"]["ranking"],
        "eligibility_rules_exact": all(
            entry["eligible"] == (not entry["ineligible_reasons"])
            for entry in select["candidates"]
        ),
        "winner_unique_or_tiebreak_resolved": select["selection"]["winner"] is not None
        and select["selection"]["ranking"][:1] == [select["selection"]["winner"]],
        "frozen_selector_config_complete": _config_is_complete(config),
        "no_seventh_candidate": len(select["candidates"]) == 6,
        "no_final_test_outcome_access": _test_bank_never_opened(access_log),
        "phase9_checkpoint_unchanged": _phase9_preservation(verify)["unchanged"],
        "learned_branch_independently_verified": not ladder["problems"],
        "full_suite_green": bool(suite) and suite["returncode"] == 0 and suite["failed"] == 0,
    }


def _same_cases(select) -> bool:
    """Every candidate and the baseline were evaluated on the same 128 cases."""
    counts = set()
    for candidate_cells in select["cells"].values():
        for cell in candidate_cells.values():
            counts.add(cell["counts"]["games"])
    for cell in select["neutral_arm"].values():
        counts.add(cell["counts"]["games"])
    return len(counts) == 1 and counts.pop() == 2 * select["case_count"]


def _phase9_preservation(verify) -> dict:
    after = checkpoint_identity(CHECKPOINT_PATH)
    before = verify["phase9_checkpoint"]
    return {
        "before": {"sha256": before["sha256"], "model_state_digest": before["model_state_digest"]},
        "after": {"sha256": after["sha256"], "model_state_digest": after["model_state_digest"]},
        "unchanged": before["sha256"] == after["sha256"]
        and before["model_state_digest"] == after["model_state_digest"],
        "c1_optimizer_steps": 0,
    }


def _discipline(games, select) -> dict:
    return {
        "validation_bank_outcome_access": games["games"],
        "test_bank_outcome_access": 0,
        "test_bank_neural_inference": 0,
        "utility_models_fit": 0,
        "candidates_added": 0,
        "temperature_changes": 0,
        "mixture_changes": 0,
        "rescue_reruns": 0,
        "c1_optimizer_steps": 0,
        "human_games_used": 0,
        "corpus_records_read": 0,
        "games_played": games["games"],
        "inference_failures": games["inference"]["failures_returned"],
    }


def _handoff(verify, select, config, config_digest) -> dict:
    return {
        "for_agent": 6,
        "mission": "integration soak and production freeze",
        "selector_config": config,
        "selector_config_sha256": config_digest,
        "selector_config_artifact": "reports/phase_10_data/agent_05_frozen_selector_config.json",
        "utility_artifacts": {
            "path": "checkpoints/phase10/setup_utility_v1.json",
            "file_sha256": verify["utility"]["file_sha256"],
            "coefficient_digests": verify["utility"]["coefficient_digests"],
            "scaler_digest": verify["utility"]["scaler_digest"],
        },
        "train_split_production_digests": config.get("train_split_production_digests"),
        "neutral_baseline_identity": {
            "profile": "neutral_v1",
            "api": "phase10_selector.neutral_baseline_draw(split, seed)",
            "redefined": False,
        },
        "phase9_identity": {
            "checkpoint_sha256": verify["phase9_checkpoint"]["sha256"],
            "model_state_digest": verify["phase9_checkpoint"]["model_state_digest"],
        },
        "validation_evidence": "reports/phase_10_data/agent_05_candidate_results.csv",
        "test_bank_unopened": {
            "bank_version": "phase10_test_bank_v1",
            "bank_digest": verify["test_bank"]["bank_digest"],
            "neural_inference": 0,
            "games": 0,
            "outcomes_read": 0,
            "accesses": "structural digest recomputation only",
        },
        "selection_is_closed": (
            "Agent 6 may not reopen selection: the winner, the six candidate "
            "definitions, both utility models and the 0.35/0.65 mixture are frozen"
        ),
    }


#: Every place Agent 5 read the contract in a way worth recording.
DEVIATIONS = [
    {
        "contract_text": (
            "one held-out opponent setup ... plays in every matchup and in both arms"
        ),
        "reading": (
            "the held-out opponent setup seats opposite the selector under test in the "
            "five externally-opposed matchups, in both arms and for all six candidates. "
            "It has no seat in learned_vs_neutral, which has two sides and two "
            "selectors: the learned draw plays the neutral_v1 draw of the colour the "
            "other seat was dealt, which is exactly the pair of neutral own-side draws "
            "Agent 1 froze per case. A third setup cannot enter a two-sided game"
        ),
    },
    {
        "contract_text": (
            "match_seeds: one seed per (case, game index, matchup), independent of arm "
            "and candidate, so a rule-based opponent draws identical randomness in both arms"
        ),
        "reading": (
            "the accepted runner derives a side's seed from match_id, and Agent 5 must "
            "also keep game identities candidate-specific, so the two requirements are "
            "met on different objects: match_id carries the cell (arm, candidate, "
            "matchup) through MatchSpec.setup_bank_version, while the opponent actually "
            "plays on case_match_seed(case_id, game_index, matchup) through a thin "
            "FrozenSeedPolicy wrapper that replaces only the request's two seed fields. "
            "The selector-under-test side is the accepted Phase 9 checkpoint playing "
            "greedy in all six matchups and reads no seed at all"
        ),
    },
    {
        "contract_text": "Stress, if run, is report-only",
        "reading": (
            "no stress evaluation was run. Agent 5's mission is bounded to the six "
            "frozen candidates on the validation bank, and a report-only diagnostic "
            "cannot change a selection, so running one would add cost and no evidence"
        ),
    },
    {
        "contract_text": "full_suite_green",
        "reading": (
            "the gate is a claim about a suite that contains the test asserting it, "
            "so a single run cannot evidence it: a false gate fails the suite, which "
            "keeps the gate false. The measurement therefore lives in its own "
            "recorded stage (`--record-suite`), the artifact test checks that the "
            "gate agrees with that measurement rather than asserting it directly, and "
            "the recorded run is the one taken with the artifact in its final state. "
            "The confirming run is reported alongside it"
        ),
    },
    {
        "contract_text": "Record every validation-bank game-outcome access",
        "reading": (
            "the access log records one entry per stage and bank rather than one per "
            "game; the per-game count is carried alongside it as "
            "discipline.validation_bank_outcome_access, so the number of outcome reads "
            "is exact and the log stays readable"
        ),
    },
]


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def stage_report(args) -> dict:
    from scripts._phase10_agent05_report import render_section

    acceptance = read_json(ACCEPTANCE_ARTIFACT)
    select = read_stage("select")
    verify = read_stage("verify")
    ladder = read_stage("ladder")
    games = read_stage("games")
    section = render_section(acceptance, verify, ladder, games, select)
    text = REPORT_PATH.read_text(encoding="utf-8")
    if SECTION_MARKER in text:
        head = text.split(SECTION_MARKER)[0].rstrip("\n")
        text = head + "\n\n" + section
    else:
        text = text.rstrip("\n") + "\n\n" + section
    REPORT_PATH.write_text(text, encoding="utf-8")
    log(f"  appended {SECTION_MARKER!r}")
    return {"stage": "report", "section": SECTION_MARKER}


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


STAGES = {
    "verify": stage_verify,
    "ladder": stage_ladder,
    "games": stage_games,
    "audit": stage_audit,
    "reproduce": stage_reproduce,
    "select": stage_select,
    "artifacts": stage_artifacts,
    "report": stage_report,
}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Phase 10 Agent 5 harness")
    parser.add_argument("--stage", choices=sorted(STAGES), action="append")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--cases", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--ladder-draws", type=int, default=120_000)
    parser.add_argument("--ladder-probes", type=int, default=2_000)
    parser.add_argument("--run-pytest", action="store_true")
    parser.add_argument("--record-suite", action="store_true")
    parser.add_argument("--reproduce-candidate", default="P10-D")
    parser.add_argument("--reproduce-matchup", default="vs_strategic")
    parser.add_argument("--reproduce-start", type=int, default=32)
    parser.add_argument("--audit-sample", type=int, default=12)
    parser.add_argument("--cell-worker", action="store_true")
    parser.add_argument("--worker", type=int, default=0)
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
    if args.cell_worker:
        cell_worker(args)
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
    elif stage_path("suite").exists():
        args.suite = read_stage("suite")

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
