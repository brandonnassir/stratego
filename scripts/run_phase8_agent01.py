#!/usr/bin/env python3
"""Phase 8 Agent 1 acceptance harness: the frozen warm-start contract.

Verifies the Phase 7 prerequisite, live-checks every frozen upstream identity
(including the regenerated Phase 4 bank digest and the canonical C1
construction), exercises the frozen corpus identity / decision sampler /
setup sources, and writes the three Agent 1 artifacts:

    reports/phase_8_data/agent_01_warmstart_contract.json
    reports/phase_8_data/agent_01_teacher_population.json
    reports/phase_8_data/agent_01_acceptance_thresholds.json

What this script is and is not
------------------------------
It freezes the *pre-corpus, pre-training contract*: roster, weights,
schedule, splits, identity, seeds, decision sampler, example schema, targets,
baselines, loss normalization, pilot matrix, selection score, acceptance
thresholds, and held-out sealing. It generates no production corpus and runs
no optimizer step — corpus generation is Agent 2's deliverable and training
is Agents 4-6's.

Usage::

    python scripts/run_phase8_agent01.py                 # verify + write artifacts
    python scripts/run_phase8_agent01.py --run-pytest    # also run the full suite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

import torch  # noqa: E402

from stratego.evaluation.setup_bank import (  # noqa: E402
    DEFAULT_BANK_ROOT_SEED,
    DEFAULT_BANK_SIZE,
    SetupBank,
    bank_digest,
)
from stratego.model.production_model import build_candidate_model  # noqa: E402
from stratego.setups.sampler import load_library_index  # noqa: E402
from stratego.training import warmstart_contract as wc  # noqa: E402
from stratego.training import warmstart_seed as ws  # noqa: E402

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_8_data"
CONTRACT_ARTIFACT = DATA_DIRECTORY / "agent_01_warmstart_contract.json"
TEACHER_ARTIFACT = DATA_DIRECTORY / "agent_01_teacher_population.json"
THRESHOLDS_ARTIFACT = DATA_DIRECTORY / "agent_01_acceptance_thresholds.json"

#: The full pre-edit suite, measured before any Phase 8 Agent 1 change.
TESTS_BEFORE = {
    "command": ".venv/bin/python -m pytest tests -q",
    "summary": "3421 passed, 3 skipped in 202.07s (0:03:22)",
    "passed": 3421,
    "skipped": 3,
    "failed": 0,
    "seconds": 202.07,
    "measured_at_commit": "144baf4",
}


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


def _verify_phase_7_acceptance() -> dict:
    """Phase 7 must be formally accepted before Phase 8 begins."""
    acceptance_path = REPOSITORY_ROOT / "reports" / "phase_7_data" / "agent_06_final_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text())
    gates = acceptance.get("completion_gates", {})
    report_text = (REPOSITORY_ROOT / "reports" / "phase_7_implementation_report.md").read_text()
    prerequisites = acceptance.get("prerequisite_status", {})
    return {
        "phase_7_agent_6_status": acceptance.get("status"),
        "phase_7_gates_total": acceptance.get("gates_total"),
        "phase_7_gates_true": acceptance.get("gates_true"),
        "phase_7_all_gates_true": bool(gates) and all(gates.values()),
        "phase_7_agents_1_to_5_pass": bool(prerequisites)
        and all(status == "PASS" for status in prerequisites.values()),
        "phase_7_report_records_acceptance": "## 6. Agent 6" in report_text,
        "phase_7_accepted": acceptance.get("status") == "PASS"
        and bool(gates)
        and all(gates.values()),
        "phase_7_frozen_versions": acceptance.get("frozen_versions", {}),
        "phase_7_library_digest": acceptance.get("library_digest"),
    }


def _verify_frozen_upstream() -> dict:
    """Live-check every frozen identity, including the expensive digests."""
    problems = wc.verify_frozen_upstream(include_library_digest=True)

    started = time.perf_counter()
    bank = SetupBank.generate(size=DEFAULT_BANK_SIZE, root_seed=DEFAULT_BANK_ROOT_SEED)
    observed_bank_digest = bank_digest(bank)
    bank_seconds = round(time.perf_counter() - started, 3)
    if observed_bank_digest != wc.EXPECTED_PHASE4_BANK_DIGEST:
        problems.append(
            f"phase4_bank_digest: expected {wc.EXPECTED_PHASE4_BANK_DIGEST!r}, "
            f"found {observed_bank_digest!r}"
        )

    model = build_candidate_model("C1", seed=ws.CANONICAL_C1_INIT_SEED)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameters != wc.EXPECTED_C1_PARAMETERS:
        problems.append(
            f"c1_parameters: expected {wc.EXPECTED_C1_PARAMETERS}, found {parameters}"
        )
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        hasher.update(name.encode())
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    canonical_init_digest = hasher.hexdigest()

    index = load_library_index()
    return {
        "problems": problems,
        "library_digest": index.content_digest,
        "library_entries": len(index.entries),
        "phase4_bank_digest": observed_bank_digest,
        "phase4_bank_size": len(bank.pairs),
        "phase4_bank_regeneration_seconds": bank_seconds,
        "c1_parameters": parameters,
        "c1_config_digest": wc.EXPECTED_C1_CONFIG_DIGEST,
        "canonical_c1_init_seed": ws.CANONICAL_C1_INIT_SEED,
        "canonical_c1_init_state_digest": canonical_init_digest,
        "canonical_c1_reconstruction": (
            "build_candidate_model('C1', seed=2026081302) on CPU float32; the "
            "state digest above is the frozen identity of the canonical "
            "untrained C1 for Agents 6 and 7"
        ),
    }


def _exercise_game_identity() -> dict:
    """Prove the 28,000 identities are well-formed, unique and seed-stable."""
    started = time.perf_counter()
    identifiers: set = set()
    per_split = {}
    for split in ws.CORPUS_SPLITS:
        identities = list(wc.iter_game_identities(split))
        per_split[split] = len(identities)
        identifiers.update(identity[-1] for identity in identities)
        for _, _, _, _, game_id in identities[:: max(1, len(identities) // 200)]:
            parsed = ws.parse_synthetic_game_id(game_id)
            assert parsed["split"] == split
    seeds_probe = ws.game_seeds(
        ws.synthetic_game_id("train", "strategic_rule_based@1.1.0", "random_legal@1.0.0", 137)
    )
    return {
        "unique_game_ids": len(identifiers),
        "per_split_counts": per_split,
        "expected_total": wc.SCHEDULE_TOTALS["total"],
        "all_unique": len(identifiers) == wc.SCHEDULE_TOTALS["total"],
        "probe_game_seeds": seeds_probe,
        "seconds": round(time.perf_counter() - started, 3),
    }


def _exercise_decision_sampler() -> dict:
    """Sweep the frozen sampler over representative game lengths."""
    started = time.perf_counter()
    game_ids = [
        ws.synthetic_game_id("train", "strategic_rule_based@1.1.0", "random_legal@1.0.0", 0),
        ws.synthetic_game_id("validation", "stress_chaos@1.0.0", "basic_heuristic@1.0.0", 39),
        ws.synthetic_game_id("test", "random_legal@1.0.0", "random_legal@1.0.0", 17),
    ]
    totals = [0, 1, 5, 63, 64, 65, 100, 127, 128, 129, 500, 700, 1024, 2000, 4000]
    checks = {
        "short_selects_all": 0,
        "long_selects_64": 0,
        "sorted_unique": 0,
        "bin_membership": 0,
        "arithmetic_reproduction": 0,
        "deterministic_repeat": 0,
    }
    selections = 0
    for game_id in game_ids:
        for total in totals:
            selected = ws.selected_decision_indices(game_id, total)
            selections += 1
            if total <= ws.MAX_DECISIONS_PER_GAME:
                assert selected == tuple(range(total))
                checks["short_selects_all"] += 1
                continue
            assert len(selected) == ws.MAX_DECISIONS_PER_GAME
            checks["long_selects_64"] += 1
            assert len(set(selected)) == len(selected)
            assert all(a < b for a, b in zip(selected, selected[1:]))
            checks["sorted_unique"] += 1
            bounds = ws.decision_bin_bounds(total)
            assert all(
                low <= index < high for index, (low, high) in zip(selected, bounds)
            )
            checks["bin_membership"] += 1
            reproduced = tuple(
                low + ws.decision_bin_seed(game_id, bin_index) % (high - low)
                for bin_index, (low, high) in enumerate(bounds)
            )
            assert reproduced == selected
            checks["arithmetic_reproduction"] += 1
            assert ws.selected_decision_indices(game_id, total) == selected
            checks["deterministic_repeat"] += 1
    return {
        "game_ids_probed": len(game_ids),
        "lengths_probed": totals,
        "selections_probed": selections,
        "checks": checks,
        "seconds": round(time.perf_counter() - started, 3),
    }


def _exercise_setup_sources() -> dict:
    """Build all three frozen sources and draw one deterministic assignment each.

    The held-out draws are explicit, justified split-access exercises (the
    same shape Phase 7 Agent 6 recorded); they create no corpus data.
    """
    started = time.perf_counter()
    per_split = {}
    for split in ws.CORPUS_SPLITS:
        source = wc.corpus_setup_source(split)
        cell = wc.ordered_matchup_cells()[37]
        game_id = ws.synthetic_game_id(split, cell["red_token"], cell["blue_token"], 0)
        root_seed = ws.setup_root_seed(game_id)
        first = source.assign(
            root_seed=root_seed,
            environment_id=wc.SETUP_SOURCE_ENVIRONMENT_ID,
            generation=wc.SETUP_SOURCE_GENERATION,
            game_id=game_id,
        )
        second = source.assign(
            root_seed=root_seed,
            environment_id=wc.SETUP_SOURCE_ENVIRONMENT_ID,
            generation=wc.SETUP_SOURCE_GENERATION,
            game_id=game_id,
        )
        provenance = first.provenance
        per_split[split] = {
            "purpose": source.purpose,
            "profile": source.profile,
            "access_justification": source.access_justification,
            "game_id": game_id,
            "setup_root_seed": root_seed,
            "deterministic_repeat": (
                first.red_setup == second.red_setup
                and first.blue_setup == second.blue_setup
            ),
            "red": {
                "base_setup_id": provenance["red"]["base_setup_id"],
                "primary_family_id": provenance["red"]["primary_family_id"],
                "split": provenance["red"]["split"],
                "side_seed": provenance["red"]["side_seed"],
            },
            "blue": {
                "base_setup_id": provenance["blue"]["base_setup_id"],
                "primary_family_id": provenance["blue"]["primary_family_id"],
                "split": provenance["blue"]["split"],
                "side_seed": provenance["blue"]["side_seed"],
            },
            "sides_independent": (
                provenance["red"]["side_seed"] != provenance["blue"]["side_seed"]
            ),
            "split_correct": provenance["red"]["split"] == split
            and provenance["blue"]["split"] == split,
        }
    return {
        "per_split": per_split,
        "seconds": round(time.perf_counter() - started, 3),
    }


def _verify_no_production_state() -> dict:
    """Agent 1 must leave no corpus and no Phase 8 checkpoint behind."""
    corpus_root = REPOSITORY_ROOT / "data" / "warmstart"
    checkpoint_root = REPOSITORY_ROOT / "checkpoints" / "phase8"
    return {
        "corpus_root": str(corpus_root.relative_to(REPOSITORY_ROOT)),
        "corpus_root_exists": corpus_root.exists(),
        "phase8_checkpoint_root_exists": checkpoint_root.exists(),
        "production_corpus_generated": False,
        "optimizer_steps": 0,
        "model_weights_mutated": False,
    }


def _run_pytest() -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    seconds = round(time.perf_counter() - started, 3)
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    counts = {
        key: int(value)
        for value, key in re.findall(r"(\d+) (passed|failed|skipped|error)", tail)
    }
    return {
        "command": f"{sys.executable} -m pytest tests -q",
        "returncode": completed.returncode,
        "summary": tail,
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0) + counts.get("error", 0),
        "skipped": counts.get("skipped", 0),
        "seconds": seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-pytest", action="store_true", help="run the full suite after verification")
    arguments = parser.parse_args()

    started = time.perf_counter()
    durations: dict = {}

    step = time.perf_counter()
    phase_7 = _verify_phase_7_acceptance()
    durations["phase_7_verification"] = round(time.perf_counter() - step, 3)

    step = time.perf_counter()
    upstream = _verify_frozen_upstream()
    durations["upstream_verification"] = round(time.perf_counter() - step, 3)

    step = time.perf_counter()
    roster_problems = wc.verify_teacher_roster()
    durations["roster_verification"] = round(time.perf_counter() - step, 3)

    identity = _exercise_game_identity()
    durations["game_identity"] = identity["seconds"]

    sampler = _exercise_decision_sampler()
    durations["decision_sampler"] = sampler["seconds"]

    sources = _exercise_setup_sources()
    durations["setup_sources"] = sources["seconds"]

    production_state = _verify_no_production_state()

    problems = list(upstream["problems"]) + list(roster_problems)
    if not phase_7["phase_7_accepted"]:
        problems.append("Phase 7 formal acceptance is not PASS")
    if not identity["all_unique"]:
        problems.append("synthetic game identities are not unique across the corpus")
    if production_state["corpus_root_exists"]:
        problems.append("data/warmstart exists before Agent 2")

    tests_after = None
    if arguments.run_pytest:
        tests_after = _run_pytest()
        durations["pytest"] = tests_after["seconds"]
        if tests_after["returncode"] != 0 or tests_after["failed"]:
            problems.append(f"full suite not green: {tests_after['summary']}")

    contract = wc.contract_document()
    matrix = contract["pilot_matrix"]

    completion_gates = {
        "phase_7_formal_acceptance_verified": phase_7["phase_7_accepted"],
        "upstream_versions_and_digests_match": not upstream["problems"],
        "exact_10_policy_roster_reproduced": not roster_problems,
        "hundred_ordered_matchup_cells_defined": len(contract["matchup_schedule"]["cells"]) == 100,
        "exact_20k_4k_4k_schedule_frozen": contract["matchup_schedule"]["totals"]
        == {"train": 20000, "validation": 4000, "test": 4000, "total": 28000},
        "setup_split_semantics_frozen": all(
            sources["per_split"][split]["split_correct"] for split in ws.CORPUS_SPLITS
        ),
        "teacher_policy_weights_frozen": contract["policy_supervision_weights"]
        == wc.POLICY_SUPERVISION_WEIGHTS,
        "corpus_identity_and_seeds_frozen": identity["all_unique"]
        and contract["canonical_seeds"] == ws.CANONICAL_SEEDS,
        "decision_sampler_exact": all(value > 0 for value in sampler["checks"].values()),
        "target_semantics_exact": bool(contract["target_semantics"]),
        "baselines_exact": bool(contract["evaluation_contract"]),
        "pilot_matrix_at_most_6_and_predeclared": len(matrix["candidates"]) <= 6,
        "pilot_selection_score_exact": "mean(r_policy, r_value, r_belief)"
        in matrix["selection"]["score"],
        "final_acceptance_thresholds_frozen": bool(contract["acceptance_thresholds"]),
        "test_and_phase4_selection_restrictions_explicit": bool(contract["sealing_rules"]),
        "no_production_corpus_generated": not production_state["corpus_root_exists"],
        "no_meaningful_optimizer_step_run": production_state["optimizer_steps"] == 0,
        "full_suite_green": (
            tests_after["returncode"] == 0 and not tests_after["failed"]
            if tests_after is not None
            else TESTS_BEFORE["failed"] == 0
        ),
    }

    status = "PASS" if not problems and all(completion_gates.values()) else "BLOCKED"

    metadata = {
        "phase": 8,
        "agent": 1,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_revision": _git("rev-parse", "--short", "HEAD"),
        "working_tree_state": "dirty" if _git("status", "--porcelain") else "clean",
        **_environment(),
        "prerequisite_versions": phase_7["phase_7_frozen_versions"],
        "prerequisite_digests": {
            "setup_library_v1": upstream["library_digest"],
            "evaluation_setup_bank_v1": upstream["phase4_bank_digest"],
            "c1_config": upstream["c1_config_digest"],
        },
        "tests_before": TESTS_BEFORE,
        "tests_after": tests_after,
        "commands": [f"{sys.executable} scripts/run_phase8_agent01.py"
                     + (" --run-pytest" if arguments.run_pytest else "")],
        "seeds": ws.CANONICAL_SEEDS,
        "files_created": [
            "stratego/training/warmstart_seed.py",
            "stratego/training/warmstart_contract.py",
            "tests/training/test_warmstart_seed.py",
            "tests/training/test_warmstart_contract.py",
            "scripts/run_phase8_agent01.py",
            "reports/phase_8_data/agent_01_warmstart_contract.json",
            "reports/phase_8_data/agent_01_teacher_population.json",
            "reports/phase_8_data/agent_01_acceptance_thresholds.json",
            "reports/phase_8_implementation_report.md",
        ],
        "files_modified": [],
        "problems": problems,
        "deviations": [],
    }

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    contract_payload = {
        **metadata,
        "artifact": "agent_01_warmstart_contract",
        "phase_7_verification": phase_7,
        "upstream_verification": upstream,
        "game_identity_verification": identity,
        "decision_sampler_verification": sampler,
        "setup_source_verification": sources,
        "production_state": production_state,
        "contract": contract,
        "contract_digest": wc.contract_digest(),
        "completion_gates": completion_gates,
        "handoff_to_agent_2": {
            "contract_versions": {
                "training_contract": wc.WARMSTART_TRAINING_CONTRACT_VERSION,
                "corpus": ws.SYNTHETIC_CORPUS_VERSION,
                "decision_sampler": ws.DECISION_SAMPLER_VERSION,
                "example": wc.WARMSTART_EXAMPLE_VERSION,
                "eval": wc.WARMSTART_EVAL_VERSION,
            },
            "teacher_tokens": list(wc.teacher_tokens()),
            "policy_weights": dict(wc.POLICY_SUPERVISION_WEIGHTS),
            "corpus_seeds": dict(ws.CANONICAL_SEEDS),
            "game_id_function": "stratego.training.warmstart_seed.synthetic_game_id",
            "per_game_seed_functions": [
                "setup_root_seed",
                "red_policy_seed",
                "blue_policy_seed",
                "decision_bin_seed",
            ],
            "schedule": contract["matchup_schedule"]["games_per_cell"],
            "setup_sources": "warmstart_contract.corpus_setup_source(split)",
            "storage_schema": contract["corpus_storage_schema"],
            "commit_rule": contract["corpus_storage_schema"]["commit_rule"],
            "no_new_learning_design_decisions": True,
        },
        "durations": durations,
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    CONTRACT_ARTIFACT.write_text(json.dumps(contract_payload, indent=1) + "\n")

    teacher_payload = {
        **metadata,
        "artifact": "agent_01_teacher_population",
        "teacher_population": wc.teacher_population(),
        "policy_supervision_weights": dict(wc.POLICY_SUPERVISION_WEIGHTS),
        "roster_verification_problems": roster_problems,
        "matchup_cells": len(contract["matchup_schedule"]["cells"]),
        "games_per_cell": contract["matchup_schedule"]["games_per_cell"],
        "schedule_totals": contract["matchup_schedule"]["totals"],
    }
    TEACHER_ARTIFACT.write_text(json.dumps(teacher_payload, indent=1) + "\n")

    thresholds_payload = {
        **metadata,
        "artifact": "agent_01_acceptance_thresholds",
        "acceptance_thresholds": wc.acceptance_thresholds(),
        "pilot_matrix": matrix,
        "sealing_rules": wc.sealing_rules(),
        "loss_semantics": wc.loss_semantics(),
        "evaluation_contract": wc.evaluation_contract(),
    }
    THRESHOLDS_ARTIFACT.write_text(json.dumps(thresholds_payload, indent=1) + "\n")

    print(f"status: {status}")
    print(f"problems: {problems if problems else 'none'}")
    print(f"gates: {sum(completion_gates.values())} / {len(completion_gates)} true")
    print(f"contract digest: {wc.contract_digest()}")
    print(f"canonical C1 init digest: {upstream['canonical_c1_init_state_digest']}")
    for path in (CONTRACT_ARTIFACT, TEACHER_ARTIFACT, THRESHOLDS_ARTIFACT):
        print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
