#!/usr/bin/env python3
"""Phase 7 Agent 1 acceptance harness: setup contract, taxonomy, diversity standard.

Verifies the Agent 1 prerequisites and completion gates, exercises the frozen
contracts end to end (fixtures, canonicalization, mobility, thresholds), and
writes the two Agent 1 artifacts:

    reports/phase_7_data/agent_01_setup_contract.json
    reports/phase_7_data/agent_01_diversity_thresholds.json

What this script is and is not
------------------------------
It freezes the *pre-generation contract*: the canonical representation,
reflection/canonicalization, stable identity, the 16 family contracts, the
trait schema, the split and seeding rules, the perturbation invariants, and
every numeric diversity threshold. It generates no production setups — the
8,000-base library is Agent 2's deliverable, and the thresholds recorded here
are frozen before that library exists.

Usage::

    python scripts/run_phase7_agent01.py                 # verify + write artifacts
    python scripts/run_phase7_agent01.py --run-pytest    # also run the full suite
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "tests"))

import torch  # noqa: E402

from stratego.engine.constants import (  # noqa: E402
    IMPLEMENTATION_VERSION,
    RULES_VERSION,
)
from stratego.engine.setup import random_setup, validate_setup  # noqa: E402
from stratego.setups import (  # noqa: E402
    BASE_SETUP_COUNT,
    BASES_PER_FAMILY,
    DEFAULT_LIBRARY_MASTER_SEED,
    FAMILY_CONTRACTS,
    FAMILY_IDS,
    SETUP_FAMILY_VERSION,
    SETUP_GENERATOR_CONTRACT_VERSION,
    SETUP_LIBRARY_VERSION,
    SETUP_TRAIT_VECTOR_VERSION,
    TEST_TOTAL,
    TRAIN_TOTAL,
    VALIDATION_TOTAL,
    canonical_class_representative,
    class_fingerprint,
    contract_document,
    derive_base_seed,
    evaluate_family,
    isolated_rebuild_sample_indices,
    reflect_canonical,
    setup_has_initial_mobility,
    split_for_base_index,
)
from stratego.setups.diversity import (  # noqa: E402
    DIVERSITY_STANDARD_VERSION,
    DIVERSITY_THRESHOLDS_V1,
    LibraryEntry,
    evaluate_against_thresholds,
)

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_7_data"
CONTRACT_ARTIFACT = DATA_DIRECTORY / "agent_01_setup_contract.json"
THRESHOLDS_ARTIFACT = DATA_DIRECTORY / "agent_01_diversity_thresholds.json"


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def _environment() -> dict:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
    }


def _verify_prerequisites() -> dict:
    """Phase 6 formal acceptance and the frozen 1.2.0 engine."""
    decision_path = REPOSITORY_ROOT / "reports" / "phase_6_data" / "agent_06b_final_decision.json"
    decision = json.loads(decision_path.read_text())
    report_text = (REPOSITORY_ROOT / "reports" / "phase_6_implementation_report.md").read_text()
    return {
        "phase_6_final_decision_status": decision["status"],
        "phase_6_final_decision_gates": decision.get("gates", None),
        "phase_6_report_recommends_close": "SAFE TO FORMALLY CLOSE" in report_text,
        "phase_6_accepted": decision["status"] == "PASS",
        "reference_engine": IMPLEMENTATION_VERSION,
        "reference_engine_is_1_2_0": IMPLEMENTATION_VERSION == "phase2_1_reference_1.2.0",
        "rules_version": RULES_VERSION,
        "production_library_absent": not (
            REPOSITORY_ROOT / "data" / "setups" / "setup_library_v1.jsonl"
        ).exists(),
    }


def _exercise_contracts() -> dict:
    """Prove every contract is executable before freezing the artifacts."""
    from setups.family_fixtures import build_fixture, build_negative_fixture

    started = time.time()
    fixture_results = {}
    for family_id in FAMILY_IDS:
        fixture = build_fixture(family_id)
        validate_setup(fixture, 0)
        satisfied, violations = evaluate_family(family_id, fixture)
        negative_satisfied, _ = evaluate_family(family_id, build_negative_fixture(family_id))
        mirrored_satisfied, _ = evaluate_family(family_id, reflect_canonical(fixture))
        fixture_results[family_id] = {
            "positive_satisfies": satisfied,
            "positive_violations": violations,
            "positive_mobile": setup_has_initial_mobility(fixture),
            "reflection_satisfies": mirrored_satisfied,
            "negative_rejected": not negative_satisfied,
        }

    # Canonicalization and identity on a seeded sample.
    rng = random.Random(20260813)
    samples = [random_setup(rng) for _ in range(200)]
    involution_failures = sum(
        1 for setup in samples if reflect_canonical(reflect_canonical(setup)) != tuple(setup)
    )
    class_failures = sum(
        1
        for setup in samples
        if canonical_class_representative(setup)
        != canonical_class_representative(reflect_canonical(setup))
    )
    fingerprint_failures = sum(
        1 for setup in samples if class_fingerprint(setup) != class_fingerprint(reflect_canonical(setup))
    )

    # The diversity standard executes end to end on a synthetic collection.
    entries = [
        LibraryEntry(
            family_id=FAMILY_IDS[index % 3],
            split=("train", "train", "validation", "test")[index % 4],
            canonical=canonical_class_representative(random_setup(rng)),
        )
        for index in range(120)
    ]
    threshold_run = evaluate_against_thresholds(entries)
    non_family_failures = [
        check["check"]
        for check in threshold_run["checks"]
        if not check["pass"] and not check["check"].endswith("self_satisfaction")
    ]

    return {
        "fixtures": fixture_results,
        "all_fixtures_pass": all(
            result["positive_satisfies"]
            and result["positive_mobile"]
            and result["reflection_satisfies"]
            and result["negative_rejected"]
            for result in fixture_results.values()
        ),
        "reflection_involution_failures": involution_failures,
        "canonical_class_failures": class_failures,
        "class_fingerprint_failures": fingerprint_failures,
        "identity_sample_size": len(samples),
        "diversity_standard_executed_checks": len(threshold_run["checks"]),
        "diversity_synthetic_non_family_failures": non_family_failures,
        "duration_seconds": round(time.time() - started, 3),
    }


def _run_pytest() -> dict:
    started = time.time()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    counts = {
        key: int(value)
        for value, key in re.findall(r"(\d+) (passed|failed|skipped|error)", tail)
    }
    return {
        "command": "python -m pytest -q",
        "exit_code": completed.returncode,
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0),
        "skipped": counts.get("skipped", 0),
        "errors": counts.get("error", 0),
        "summary_line": tail,
        "duration_seconds": round(time.time() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-pytest", action="store_true", help="run the full suite too")
    arguments = parser.parse_args()

    started = time.time()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    commit = _git("rev-parse", "HEAD")
    working_tree_state = "dirty" if _git("status", "--porcelain") else "clean"

    prerequisites = _verify_prerequisites()
    exercise = _exercise_contracts()
    pytest_result = _run_pytest() if arguments.run_pytest else None

    gates = {
        "phase_6_accepted": prerequisites["phase_6_accepted"],
        "reference_engine_is_1_2_0": prerequisites["reference_engine_is_1_2_0"],
        "sixteen_family_contracts_explicit": len(FAMILY_CONTRACTS) == 16,
        "library_counts_exact": BASE_SETUP_COUNT == 8000
        and BASES_PER_FAMILY == 500
        and (TRAIN_TOTAL, VALIDATION_TOTAL, TEST_TOTAL) == (6400, 800, 800),
        "split_rule_exact": [split_for_base_index(i) for i in (0, 399, 400, 449, 450, 499)]
        == ["train", "train", "validation", "validation", "test", "test"],
        "all_family_fixtures_executable": exercise["all_fixtures_pass"],
        "reflection_involution_clean": exercise["reflection_involution_failures"] == 0,
        "canonicalization_stable": exercise["canonical_class_failures"] == 0,
        "class_fingerprint_reflection_invariant": exercise["class_fingerprint_failures"] == 0,
        "diversity_standard_executable": exercise["diversity_synthetic_non_family_failures"] == [],
        "thresholds_frozen_before_generation": prerequisites["production_library_absent"],
    }

    contract_payload = {
        "agent": "agent_01",
        "phase": "phase_7",
        "assignment": "Setup Contract, Taxonomy, and Diversity Standard",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "schema_version": "phase_7_agent_01_v1",
        "timestamp": timestamp,
        "commit": commit,
        "working_tree_state": working_tree_state,
        **_environment(),
        "prerequisite_status": prerequisites,
        "frozen_versions": {
            "rules": RULES_VERSION,
            "reference_engine": IMPLEMENTATION_VERSION,
            "contract_version": SETUP_GENERATOR_CONTRACT_VERSION,
            "library_version": SETUP_LIBRARY_VERSION,
            "family_contract_version": SETUP_FAMILY_VERSION,
            "trait_schema_version": SETUP_TRAIT_VECTOR_VERSION,
            "diversity_standard_version": DIVERSITY_STANDARD_VERSION,
        },
        "library_version": SETUP_LIBRARY_VERSION,
        "contract_version": SETUP_GENERATOR_CONTRACT_VERSION,
        "family_contract_version": SETUP_FAMILY_VERSION,
        "master_seed": DEFAULT_LIBRARY_MASTER_SEED,
        "setup_count": BASE_SETUP_COUNT,
        "split_counts": {"train": TRAIN_TOTAL, "validation": VALIDATION_TOTAL, "test": TEST_TOTAL},
        "family_counts": {family_id: BASES_PER_FAMILY for family_id in FAMILY_IDS},
        "seeds": {
            "master_seed": DEFAULT_LIBRARY_MASTER_SEED,
            "example_base_seed_F00_0": derive_base_seed(
                SETUP_GENERATOR_CONTRACT_VERSION,
                SETUP_LIBRARY_VERSION,
                DEFAULT_LIBRARY_MASTER_SEED,
                "F00",
                0,
            ),
            "fixture_exercise_seed": 20260813,
        },
        "isolated_rebuild_indices_per_family": list(isolated_rebuild_sample_indices()),
        "contract_exercise": exercise,
        "tests_before": {
            "command": "python -m pytest -q",
            "passed": 2732,
            "failed": 0,
            "skipped": 3,
            "errors": 0,
            "note": "recorded before any Phase 7 edit; identical to the Phase 6 accepted totals",
        },
        "tests_after": pytest_result,
        "commands": [
            "python scripts/run_phase7_agent01.py" + (" --run-pytest" if arguments.run_pytest else ""),
        ],
        "durations": {"total_seconds": None},  # patched below
        "files_created": [
            "stratego/setups/__init__.py",
            "stratego/setups/identity.py",
            "stratego/setups/mobility.py",
            "stratego/setups/traits.py",
            "stratego/setups/families.py",
            "stratego/setups/contracts.py",
            "stratego/setups/diversity.py",
            "tests/setups/__init__.py",
            "tests/setups/family_fixtures.py",
            "tests/setups/test_identity.py",
            "tests/setups/test_mobility.py",
            "tests/setups/test_traits.py",
            "tests/setups/test_families.py",
            "tests/setups/test_contracts.py",
            "tests/setups/test_diversity.py",
            "scripts/run_phase7_agent01.py",
            "reports/phase_7_data/agent_01_setup_contract.json",
            "reports/phase_7_data/agent_01_diversity_thresholds.json",
            "reports/phase_7_implementation_report.md",
        ],
        "files_modified": [],
        "completion_gates": gates,
        "gates_total": len(gates),
        "gates_true": sum(1 for value in gates.values() if value),
        "problems": [],
        "deviations": [
            "canonical-frame helpers are reimplemented in stratego/setups/identity.py "
            "rather than imported from the frozen Phase 4 evaluation fixture; "
            "tests/setups/test_identity.py pins both implementations together "
            "exhaustively so one convention exists",
            "the F15 irregular family deliberately permits any Flag rank, including "
            "the front rank; every other family constrains the Flag to the back "
            "rank(s) as part of its contract",
        ],
        "contract": contract_document(),
    }

    thresholds_payload = {
        "agent": "agent_01",
        "phase": "phase_7",
        "artifact": "diversity_thresholds",
        "status": contract_payload["status"],
        "timestamp": timestamp,
        "commit": commit,
        "diversity_standard_version": DIVERSITY_STANDARD_VERSION,
        "frozen_before_generation": prerequisites["production_library_absent"],
        "contract_version": SETUP_GENERATOR_CONTRACT_VERSION,
        "library_version": SETUP_LIBRARY_VERSION,
        "metric_definitions": {
            "class_distance": (
                "min(Hamming(a, b), Hamming(a, reflect(b))) over the 40 canonical "
                "squares; symmetric and well-defined on reflection classes"
            ),
            "within_family_nn_distance": (
                "per base setup, the minimum class distance to any other base in "
                "the same family (all 500, any split)"
            ),
            "near_duplicate_pair_fraction": (
                "fraction of unordered within-family pairs with class distance "
                "strictly below the declared near-duplicate distance"
            ),
            "cross_split_nn_distance": (
                "minimum class distance over every pair of bases in different "
                "splits, across the whole library"
            ),
            "global_min_pairwise_distance": (
                "minimum class distance over every unordered pair of bases in "
                "the library"
            ),
            "per_square_entropy_bits": (
                "Shannon entropy (base 2) of the piece-type distribution at each "
                "of the 40 canonical cells across a family's 500 bases; the "
                "family metric is the mean over the 40 cells"
            ),
            "folded_support": (
                "count of distinct reflection-invariant folded cells "
                "(rank, min(file, 9-file)) — 20 possible — occupied by the "
                "named piece group anywhere in the family"
            ),
            "distinct_trait_vectors": (
                "count of distinct full setup_trait_vector_v1 values among a "
                "family's bases"
            ),
            "family_overlap_matrix": (
                "matrix[i][j] = fraction of family i bases satisfying family j's "
                "contract; diagonal must be exactly 1.0; off-diagonal is "
                "report-only"
            ),
            "procedural_stress_targets": (
                "the same identity/quality/family/split rules apply to every "
                "procedural output: >= 100,000 samples with zero legality, "
                "inventory, stranded, family, or split-leak failures, and "
                "perturbation class distance within [2, 12] of the base"
            ),
        },
        "statistical_rationale": (
            "independent structured draws over the residual 30+ free squares "
            "concentrate near class distance 30/40; the floors (6/8/4) are far "
            "below that concentration, so a correct generator passes with "
            "overwhelming margin while template repetition with cosmetic swaps "
            "fails immediately; entropy/support floors are likewise set well "
            "below honest-generation values but far above degenerate repetition"
        ),
        "thresholds": DIVERSITY_THRESHOLDS_V1.to_dict(),
    }

    contract_payload["durations"]["total_seconds"] = round(time.time() - started, 1)

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CONTRACT_ARTIFACT.write_text(json.dumps(contract_payload, indent=1, sort_keys=True) + "\n")
    THRESHOLDS_ARTIFACT.write_text(json.dumps(thresholds_payload, indent=1, sort_keys=True) + "\n")

    print(f"status                 {contract_payload['status']}")
    print(f"gates                  {contract_payload['gates_true']}/{contract_payload['gates_total']}")
    print(f"contract artifact      {CONTRACT_ARTIFACT.relative_to(REPOSITORY_ROOT)}")
    print(f"thresholds artifact    {THRESHOLDS_ARTIFACT.relative_to(REPOSITORY_ROOT)}")
    if pytest_result is not None:
        print(f"pytest                 {pytest_result['summary_line']}")
    return 0 if contract_payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
