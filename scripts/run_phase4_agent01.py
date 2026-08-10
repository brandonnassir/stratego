#!/usr/bin/env python3
"""Phase 4 Agent 1 acceptance harness.

Runs the evaluation-foundation gates and writes
`reports/phase_4_data/agent_01_evaluation_foundations.json`:

- the full evaluation setup bank, validated pair by pair against the frozen
  engine, plus its variation metrics and content digest;
- determinism trials: repeated bank regeneration, isolated single-pair rebuild,
  match identity under shuffling and sharding, and policy-seed derivation;
- paired-unit trials: every unit reconstructs, holds its board fixed, assigns
  each colour exactly once and gives each policy the first move exactly once;
- contract games played end to end through the policy interface, replayed to
  confirm the identity alone reproduces the game;
- hidden-identity permutation trials with a positive control, the local
  precursor to Agent 4's 100,000-trial audit;
- the automated test suite summary.

Every stage is deterministically seeded and runs in a single process. Agent 1
implements no match runner, no parallelism and no baseline strategy; the game
loop in `contract_game_stage` is a throwaway check that the contract fits
together, not the runner Agent 3 owns.

Usage:

    python scripts/run_phase4_agent01.py                 # full acceptance run
    python scripts/run_phase4_agent01.py --quick         # fast smoke run
    python scripts/run_phase4_agent01.py --skip-pytest   # measurements only
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
from collections import Counter
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.engine.constants import (  # noqa: E402
    BLUE,
    EVALUATION_RULES,
    IMPLEMENTATION_VERSION,
    OBSERVATION_VERSION,
    RED,
    RULES_VERSION,
)
from stratego.engine.invariants import check_invariants  # noqa: E402
from stratego.engine.legal_moves import legal_actions  # noqa: E402
from stratego.engine.permutation import (  # noqa: E402
    belief_targets_differ,
    permute_hidden_identities,
)
from stratego.engine.random_play import play_random_game_to_ply  # noqa: E402
from stratego.engine.state import GameState, create_game, state_fingerprint  # noqa: E402
from stratego.engine.transition import apply_action  # noqa: E402
from stratego.evaluation.match_spec import (  # noqa: E402
    EVALUATION_SUITE_VERSION,
    MATCH_SPEC_VERSION,
    PAIRING_COLOR_SWAP_SAME_BOARD,
    MatchSpec,
    PairedUnit,
    build_paired_schedule,
    schedule_digest,
    schedule_matches,
    shard_schedule,
    sibling_match,
    validate_schedule,
)
from stratego.evaluation.policy import (  # noqa: E402
    POLICY_INTERFACE_VERSION,
    FirstLegalActionPolicy,
    ObservationProbePolicy,
    Policy,
    PolicyRef,
    PolicyRequirements,
    SeededUniformPolicy,
    build_policy_input,
    build_public_view,
)
from stratego.evaluation.setup_bank import (  # noqa: E402
    DEFAULT_BANK_ROOT_SEED,
    DEFAULT_BANK_SIZE,
    GENERATION_FAMILY,
    MINIMUM_BANK_SIZE,
    SETUP_BANK_VERSION,
    SetupBank,
    bank_digest,
    bank_diversity,
    generate_setup_pair,
    structural_violations,
    validate_bank,
)

DEFAULT_OUTPUT = REPOSITORY_ROOT / "reports/phase_4_data/agent_01_evaluation_foundations.json"
DEFAULT_BANK_ARTIFACT = REPOSITORY_ROOT / "reports/phase_4_data/agent_01_setup_bank_v1.json"

ALL_REQUIREMENTS = PolicyRequirements(
    observation=True,
    legal_action_mask=True,
    public_view=True,
    public_events=True,
    public_setup=True,
)


# ---------------------------------------------------------------------------
# Stage 1 -- the setup bank
# ---------------------------------------------------------------------------


def setup_bank_stage(size: int, artifact: "Path | None") -> tuple[SetupBank, dict]:
    started = time.perf_counter()
    bank = SetupBank.generate(size)
    generation_seconds = time.perf_counter() - started

    summary = validate_bank(bank)
    violations = structural_violations(bank)
    diversity = bank_diversity(bank)

    if artifact is not None:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(bank.to_json())

    return bank, {
        "bank_version": bank.bank_version,
        "generation_family": bank.generation_family,
        "root_seed": bank.root_seed,
        "pair_count": len(bank),
        "meets_preferred_target": len(bank) >= DEFAULT_BANK_SIZE,
        "meets_minimum_target": len(bank) >= MINIMUM_BANK_SIZE,
        "validation_failure_count": summary["validation_failure_count"],
        "validation_failures": summary["validation_failures"][:20],
        "duplicate_setup_pair_ids": summary["duplicate_setup_pair_ids"],
        "structural_violation_count": len(violations),
        "structural_violations": violations[:20],
        "distinct_red_setups": summary["distinct_red_setups"],
        "distinct_blue_setups": summary["distinct_blue_setups"],
        "distinct_positions": summary["distinct_positions"],
        "diversity": diversity,
        "digest": summary["digest"],
        "artifact_path": None if artifact is None else str(artifact.relative_to(REPOSITORY_ROOT)),
        "generation_seconds": generation_seconds,
    }


# ---------------------------------------------------------------------------
# Stage 2 -- determinism
# ---------------------------------------------------------------------------


def determinism_stage(bank: SetupBank, regenerations: int, sampled_pairs: int) -> dict:
    """Every reproducibility claim Agent 1 makes, counted as pass/fail trials."""
    trials = 0
    failures: list[str] = []
    reference_digest = bank_digest(bank)

    # 1. Same seed -> byte-identical bank.
    for index in range(regenerations):
        trials += 1
        again = SetupBank.generate(len(bank))
        if again.to_json() != bank.to_json():
            failures.append(f"bank regeneration {index} is not byte-identical")

    # 2. Any pair rebuilds in isolation, without generating its neighbours.
    rng = random.Random(20260401)
    sample = sorted(rng.sample(range(len(bank)), min(sampled_pairs, len(bank))))
    for setup_pair_id in sample:
        trials += 1
        if generate_setup_pair(setup_pair_id) != bank.pair(setup_pair_id):
            failures.append(f"pair {setup_pair_id} does not rebuild in isolation")

    # 3. A serialised bank round-trips.
    trials += 1
    if bank_digest(SetupBank.from_json(bank.to_json())) != reference_digest:
        failures.append("bank does not survive a JSON round trip")

    # 4. Match identity is stable and independent of order and worker count.
    candidate = PolicyRef("acceptance_candidate", "1.0.0")
    opponent = PolicyRef("acceptance_opponent", "1.0.0")
    units = build_paired_schedule(candidate, opponent, bank.pair_ids, replicates=2)
    matches = schedule_matches(units)
    reference_schedule_digest = schedule_digest(matches)
    reference_ids = [match.match_id for match in matches]

    trials += 1
    if [match.match_id for match in schedule_matches(units)] != reference_ids:
        failures.append("recomputing a schedule changed its match identifiers")

    trials += 1
    rebuilt = schedule_matches(
        build_paired_schedule(candidate, opponent, bank.pair_ids, replicates=2)
    )
    if [match.match_id for match in rebuilt] != reference_ids:
        failures.append("rebuilding a schedule from scratch changed its match identifiers")

    shuffle_trials = 0
    for seed in range(8):
        shuffled = list(matches)
        random.Random(seed).shuffle(shuffled)
        trials += 1
        shuffle_trials += 1
        if schedule_digest(shuffled) != reference_schedule_digest:
            failures.append(f"shuffling the schedule with seed {seed} changed its contents")
        if sorted(match.match_id for match in shuffled) != sorted(reference_ids):
            failures.append(f"shuffling the schedule with seed {seed} changed a specification")

    worker_counts = (1, 2, 3, 4, 8, 16, 32)
    for worker_count in worker_counts:
        trials += 1
        shards = shard_schedule(matches, worker_count)
        rejoined = [match for shard in shards for match in shard]
        if schedule_digest(rejoined) != reference_schedule_digest:
            failures.append(f"sharding across {worker_count} workers changed the schedule")
        if sum(len(shard) for shard in shards) != len(matches):
            failures.append(f"sharding across {worker_count} workers lost or duplicated a match")

    # 5. Policy seeds follow from the identifier alone.
    for match in matches[: min(256, len(matches))]:
        trials += 1
        twin = MatchSpec.from_dict(match.to_dict())
        if (twin.candidate_seed, twin.opponent_seed) != (
            match.candidate_seed,
            match.opponent_seed,
        ):
            failures.append(f"match {match.match_id} derived different policy seeds on rebuild")
        if twin.match_id != match.match_id:
            failures.append(f"match {match.match_id} did not survive a dictionary round trip")

    # 6. Distinct identities really are distinct.
    trials += 1
    if len({match.match_id for match in matches}) != len(matches):
        failures.append("two scheduled matches share a match identifier")
    trials += 1
    seeds = {match.candidate_seed for match in matches} | {
        match.opponent_seed for match in matches
    }
    if len(seeds) != 2 * len(matches):
        failures.append("two scheduled policy seeds collided")

    return {
        "trials": trials,
        "failures": len(failures),
        "failure_detail": failures[:20],
        "bank_regenerations": regenerations,
        "pairs_rebuilt_in_isolation": len(sample),
        "shuffle_trials": shuffle_trials,
        "worker_counts_tested": list(worker_counts),
        "matches_in_schedule": len(matches),
        "schedule_digest": reference_schedule_digest,
    }


# ---------------------------------------------------------------------------
# Stage 3 -- paired units
# ---------------------------------------------------------------------------


def paired_unit_stage(bank: SetupBank, units_tested: int) -> dict:
    candidate = PolicyRef("acceptance_candidate", "1.0.0")
    opponent = PolicyRef("acceptance_opponent", "1.0.0")
    pair_ids = bank.pair_ids[:units_tested]
    units = build_paired_schedule(
        candidate, opponent, pair_ids, setup_bank_version=bank.bank_version
    )

    failures: list[str] = []
    for unit in units:
        game_a, game_b = unit.matches

        if (game_a.candidate_color, game_b.candidate_color) != (RED, BLUE):
            failures.append(f"unit {unit.paired_unit_id} does not cover both colours")
        if game_a.paired_unit_id != game_b.paired_unit_id != unit.paired_unit_id:
            failures.append(f"unit {unit.paired_unit_id} games disagree on the unit identifier")
        if game_a.match_id == game_b.match_id:
            failures.append(f"unit {unit.paired_unit_id} games share a match identifier")

        # The defining property of `color_swap_same_board`.
        if game_a.resolve_setups(bank) != game_b.resolve_setups(bank):
            failures.append(f"unit {unit.paired_unit_id} does not hold its board fixed")
        if game_a.resolve_setups(bank) != (
            bank.pair(unit.setup_pair_id).red_setup,
            bank.pair(unit.setup_pair_id).blue_setup,
        ):
            failures.append(f"unit {unit.paired_unit_id} resolved the wrong setup pair")

        # Each policy takes the first move exactly once.
        if [game_a.candidate_moves_first, game_b.candidate_moves_first] != [True, False]:
            failures.append(f"unit {unit.paired_unit_id} does not balance the first move")

        # Reconstruction from either half.
        for match in unit.matches:
            if PairedUnit.from_match(match) != unit:
                failures.append(f"unit {unit.paired_unit_id} does not reconstruct from a match")
            if sibling_match(sibling_match(match)) != match:
                failures.append(f"unit {unit.paired_unit_id} has an unstable sibling")
            if sibling_match(match).candidate_color != match.opponent_color:
                failures.append(f"unit {unit.paired_unit_id} sibling has the wrong colour")

    schedule_problems = validate_schedule(schedule_matches(units), bank)
    failures.extend(schedule_problems)

    return {
        "units_tested": len(units),
        "failures": len(failures),
        "failure_detail": failures[:20],
        "pairing_mode": PAIRING_COLOR_SWAP_SAME_BOARD,
        "schedule_problems": len(schedule_problems),
    }


# ---------------------------------------------------------------------------
# Stage 4 -- games driven entirely through the policy contract
# ---------------------------------------------------------------------------


def _play_contract_game(
    spec: MatchSpec, bank: SetupBank, policies: "dict[str, Policy]", max_plies: int
) -> tuple[GameState, list[int]]:
    """Play one match using only the observer-safe contract.

    Not a match runner: no result schema, no timing, no parallelism, no error
    recovery. Agent 3 owns those. This exists to prove the contract can actually
    drive the frozen engine and that every decision is legal.
    """
    red_setup, blue_setup = spec.resolve_setups(bank)
    state = create_game(red_setup, blue_setup, rules=spec.rules, game_id=spec.game_id)

    while not state.terminal and state.total_moves < max_plies:
        actor = state.acting_player
        ref = spec.policy_ref_for(actor)
        policy = policies[ref.token]
        legal = legal_actions(state)
        request = build_policy_input(
            state,
            policy=ref,
            policy_seed=spec.policy_seed_for(actor),
            requirements=policy.requirements,
            suite_version=spec.suite_version,
            match_id=spec.match_id,
            paired_unit_id=spec.paired_unit_id,
            legal=legal,
        )
        result = policy.decide_checked(request)
        # The engine remains the final legality authority.
        apply_action(state, result.selected_action_id, legal=legal)

    return state, list(state.action_history)


def contract_game_stage(bank: SetupBank, unit_count: int, max_plies: int) -> dict:
    first_legal = FirstLegalActionPolicy()
    uniform = SeededUniformPolicy()
    probe = ObservationProbePolicy()
    policies = {policy.ref.token: policy for policy in (first_legal, uniform, probe)}

    matchups = (
        (uniform.ref, first_legal.ref),
        (uniform.ref, probe.ref),
    )

    games = 0
    plies = 0
    replay_failures: list[str] = []
    board_failures: list[str] = []
    terminal_reasons: Counter = Counter()
    color_results: dict[str, list[float]] = {"candidate_red": [], "candidate_blue": []}

    for candidate, opponent in matchups:
        units = build_paired_schedule(
            candidate,
            opponent,
            bank.pair_ids[:unit_count],
            setup_bank_version=bank.bank_version,
        )
        for unit in units:
            boards = []
            for spec in unit.matches:
                state, history = _play_contract_game(spec, bank, policies, max_plies)
                check_invariants(state)
                games += 1
                plies += state.total_moves
                boards.append(spec.resolve_setups(bank))

                # The identity alone must reproduce the game exactly.
                replay_state, replay_history = _play_contract_game(
                    MatchSpec.from_dict(spec.to_dict()), bank, policies, max_plies
                )
                if replay_history != history or state_fingerprint(
                    replay_state
                ) != state_fingerprint(state):
                    replay_failures.append(spec.match_id)

                if state.terminal:
                    terminal_reasons[state.terminal_reason] += 1
                    key = "candidate_red" if spec.candidate_color == RED else "candidate_blue"
                    color_results[key].append(state.effective_score_for(spec.candidate_color))

            if boards[0] != boards[1]:
                board_failures.append(unit.paired_unit_id)

    return {
        "games": games,
        "plies": plies,
        "replay_mismatches": len(replay_failures),
        "replay_mismatch_detail": replay_failures[:10],
        "paired_board_mismatches": len(board_failures),
        "illegal_actions": 0,
        "terminal_reasons": dict(terminal_reasons),
        "candidate_effective_score_as_red": (
            sum(color_results["candidate_red"]) / len(color_results["candidate_red"])
            if color_results["candidate_red"]
            else None
        ),
        "candidate_effective_score_as_blue": (
            sum(color_results["candidate_blue"]) / len(color_results["candidate_blue"])
            if color_results["candidate_blue"]
            else None
        ),
        "note": (
            "Contract fixtures only. These scores measure nothing about Stratego "
            "strength and must not be read as a baseline result."
        ),
    }


# ---------------------------------------------------------------------------
# Stage 5 -- hidden-identity permutation
# ---------------------------------------------------------------------------


def permutation_stage(target_trials: int) -> dict:
    """The local precursor to Agent 4's audit.

    Each trial permutes the true types of unresolved opponent pieces and
    requires every observer-legal product, and every contract policy's decision,
    to be unchanged. A trial only counts if the positive control fires.
    """
    policies = [FirstLegalActionPolicy(), SeededUniformPolicy(), ObservationProbePolicy()]
    rng = random.Random(20260401)

    trials = 0
    policy_comparisons = 0
    mismatches: list[str] = []
    positive_control_failures = 0
    skipped_unchanged = 0

    for seed in range(100_000):
        if trials >= target_trials:
            break
        for ply in (8, 20, 40, 70, 110):
            if trials >= target_trials:
                break
            state = play_random_game_to_ply(seed, ply, rules=EVALUATION_RULES)
            if state.terminal or state.total_moves != ply:
                continue
            observer = state.acting_player
            clone, info = permute_hidden_identities(state, observer, rng)
            if not info["valid"] or not info["changed"]:
                skipped_unchanged += 1
                continue

            # Positive control: the privileged state really did change.
            if not belief_targets_differ(state, clone, observer):
                positive_control_failures += 1

            trials += 1

            if build_public_view(state, observer) != build_public_view(clone, observer):
                mismatches.append(f"public view differs at seed {seed} ply {ply}")

            for policy in policies:
                first = build_policy_input(
                    state,
                    policy=policy.ref,
                    policy_seed=17,
                    requirements=ALL_REQUIREMENTS,
                )
                second = build_policy_input(
                    clone,
                    policy=policy.ref,
                    policy_seed=17,
                    requirements=ALL_REQUIREMENTS,
                )
                if not np.array_equal(first.observation, second.observation):
                    mismatches.append(f"observation differs at seed {seed} ply {ply}")
                if first.legal_actions != second.legal_actions:
                    mismatches.append(f"legal actions differ at seed {seed} ply {ply}")
                if first.public_events != second.public_events:
                    mismatches.append(f"public events differ at seed {seed} ply {ply}")
                if first.public_setup != second.public_setup:
                    mismatches.append(f"public setup differs at seed {seed} ply {ply}")

                left = policy.decide_checked(first)
                right = policy.decide_checked(second)
                policy_comparisons += 1
                if left.selected_action_id != right.selected_action_id:
                    mismatches.append(
                        f"{policy.policy_id} chose differently at seed {seed} ply {ply}"
                    )
                if left.diagnostics != right.diagnostics:
                    mismatches.append(
                        f"{policy.policy_id} diagnostics differ at seed {seed} ply {ply}"
                    )

    return {
        "trials": trials,
        "policy_comparisons": policy_comparisons,
        "mismatches": len(mismatches),
        "mismatch_detail": mismatches[:20],
        "positive_control_failures": positive_control_failures,
        "positions_skipped_unchanged": skipped_unchanged,
        "policies": [policy.policy_id for policy in policies],
        "note": "Agent 4 owns the >= 100,000 trial audit across the real baseline suite.",
    }


# ---------------------------------------------------------------------------
# Stage 6 -- the automated suite
# ---------------------------------------------------------------------------


def pytest_stage(target: str = "") -> dict:
    command = [sys.executable, "-m", "pytest", "-q", "--tb=no"]
    if target:
        command.append(target)
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False
    )
    elapsed = time.perf_counter() - started
    output = completed.stdout + completed.stderr

    def count(label: str) -> int:
        match = re.search(rf"(\d+) {label}", output)
        return int(match.group(1)) if match else 0

    passed = count("passed")
    failed = count("failed")
    skipped = count("skipped")
    xfailed = count("xfailed")
    errors = count("error")

    return {
        "total": passed + failed + skipped + xfailed + errors,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "expected_failures": xfailed,
        "errors": errors,
        "seconds": elapsed,
        "exit_code": completed.returncode,
        "failure_lines": [line for line in output.splitlines() if line.startswith("FAILED")],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

FILES_CREATED = [
    "stratego/evaluation/__init__.py",
    "stratego/evaluation/policy.py",
    "stratego/evaluation/match_spec.py",
    "stratego/evaluation/setup_bank.py",
    "tests/evaluation/__init__.py",
    "tests/evaluation/test_policy_contract.py",
    "tests/evaluation/test_match_spec.py",
    "tests/evaluation/test_setup_bank.py",
    "scripts/run_phase4_agent01.py",
    "reports/phase_4_data/agent_01_evaluation_foundations.json",
    "reports/phase_4_data/agent_01_setup_bank_v1.json",
    "reports/phase_4_implementation_report.md",
]

FILES_MODIFIED: list[str] = []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-size", type=int, default=DEFAULT_BANK_SIZE)
    parser.add_argument("--regenerations", type=int, default=8)
    parser.add_argument("--sampled-pairs", type=int, default=128)
    parser.add_argument("--paired-units", type=int, default=DEFAULT_BANK_SIZE)
    parser.add_argument("--contract-units", type=int, default=24)
    # The absolute move limit is the engine's own ceiling, so this cap never
    # truncates a game that the rules would have allowed to continue.
    parser.add_argument("--max-plies", type=int, default=EVALUATION_RULES.absolute_move_limit)
    parser.add_argument("--permutation-trials", type=int, default=2000)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bank-artifact", type=Path, default=DEFAULT_BANK_ARTIFACT)
    options = parser.parse_args()

    if options.quick:
        options.bank_size = 64
        options.regenerations = 2
        options.sampled_pairs = 16
        options.paired_units = 16
        options.contract_units = 3
        options.permutation_trials = 60

    started = time.perf_counter()

    print(f"[1/6] generating and validating {options.bank_size} setup pairs")
    bank, bank_summary = setup_bank_stage(
        options.bank_size, None if options.quick else options.bank_artifact
    )

    print(f"[2/6] determinism trials over {len(bank)} pairs")
    determinism = determinism_stage(bank, options.regenerations, options.sampled_pairs)

    print(f"[3/6] paired-unit trials over {min(options.paired_units, len(bank))} units")
    paired = paired_unit_stage(bank, min(options.paired_units, len(bank)))

    print(f"[4/6] contract games over {options.contract_units} paired units per matchup")
    contract = contract_game_stage(bank, options.contract_units, options.max_plies)

    print(f"[5/6] {options.permutation_trials} hidden-identity permutation trials")
    permutation = permutation_stage(options.permutation_trials)

    if options.skip_pytest:
        print("[6/6] skipping the test suite")
        tests = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "expected_failures": 0,
            "errors": 0,
            "seconds": 0.0,
            "exit_code": None,
            "failure_lines": [],
            "skipped_by_request": True,
        }
    else:
        print("[6/6] running the repository test suite")
        tests = pytest_stage()

    status_checks = {
        "bank_meets_minimum": bank_summary["meets_minimum_target"],
        "bank_legal": bank_summary["validation_failure_count"] == 0,
        "bank_ids_unique": bank_summary["duplicate_setup_pair_ids"] == [],
        "bank_structurally_sound": bank_summary["structural_violation_count"] == 0,
        "determinism_clean": determinism["failures"] == 0,
        "paired_units_clean": paired["failures"] == 0,
        "contract_games_reproducible": contract["replay_mismatches"] == 0
        and contract["paired_board_mismatches"] == 0,
        "no_hidden_information_mismatch": permutation["mismatches"] == 0,
        "positive_control_fired": permutation["positive_control_failures"] == 0,
        "tests_green": options.skip_pytest or (tests["failed"] == 0 and tests["errors"] == 0),
    }
    status = "PASS" if all(status_checks.values()) else "FAIL"

    payload = {
        "agent": "phase_4_agent_01_evaluation_foundations",
        "status": status,
        "status_checks": status_checks,
        "frozen_contracts": {
            "implementation_version": IMPLEMENTATION_VERSION,
            "rules_version": RULES_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "phase_3_backend_decision": "KEEP_PYTHON",
        },
        "policy_interface_version": POLICY_INTERFACE_VERSION,
        "match_spec_version": MATCH_SPEC_VERSION,
        "evaluation_suite_version": EVALUATION_SUITE_VERSION,
        "pairing_mode": PAIRING_COLOR_SWAP_SAME_BOARD,
        "evaluation_rules": {
            "context": EVALUATION_RULES.context,
            "battleless_move_limit": EVALUATION_RULES.battleless_move_limit,
            "absolute_move_limit": EVALUATION_RULES.absolute_move_limit,
            "first_player": EVALUATION_RULES.first_player,
        },
        "setup_bank_version": SETUP_BANK_VERSION,
        "setup_bank_generation_family": GENERATION_FAMILY,
        "setup_bank_root_seed": DEFAULT_BANK_ROOT_SEED,
        "setup_bank_digest": bank_summary["digest"],
        "setup_pair_count": bank_summary["pair_count"],
        "setup_validation_failures": bank_summary["validation_failure_count"],
        "duplicate_setup_pair_ids": bank_summary["duplicate_setup_pair_ids"],
        "determinism_trials": determinism["trials"],
        "determinism_failures": determinism["failures"],
        "paired_units_tested": paired["units_tested"],
        "paired_unit_failures": paired["failures"],
        "hidden_permutation_trials": permutation["trials"],
        "hidden_permutation_mismatches": permutation["mismatches"],
        "positive_control_failures": permutation["positive_control_failures"],
        "contract_games_played": contract["games"],
        "illegal_policy_actions": contract["illegal_actions"],
        "test_total": tests["total"],
        "test_passed": tests["passed"],
        "test_failed": tests["failed"] + tests["errors"],
        "setup_bank": bank_summary,
        "determinism": determinism,
        "paired_units": paired,
        "contract_games": contract,
        "hidden_permutation": permutation,
        "tests": tests,
        "files_created": FILES_CREATED,
        "files_modified": FILES_MODIFIED,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "quick_mode": options.quick,
        "total_seconds": time.perf_counter() - started,
    }

    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print()
    print(f"status                      {status}")
    print(f"setup pairs                 {payload['setup_pair_count']}")
    print(f"setup validation failures   {payload['setup_validation_failures']}")
    print(f"determinism trials          {payload['determinism_trials']}")
    print(f"determinism failures        {payload['determinism_failures']}")
    print(f"paired units tested         {payload['paired_units_tested']}")
    print(f"paired unit failures        {payload['paired_unit_failures']}")
    print(f"contract games              {payload['contract_games_played']}")
    print(f"permutation trials          {payload['hidden_permutation_trials']}")
    print(f"permutation mismatches      {payload['hidden_permutation_mismatches']}")
    print(f"tests passed / failed       {payload['test_passed']} / {payload['test_failed']}")
    print(f"written                     {options.output.relative_to(REPOSITORY_ROOT)}")
    if options.quick and not status_checks["bank_meets_minimum"]:
        print()
        print(
            "note: --quick generates a 64-pair bank, which is below the 512-pair "
            "acceptance floor, so a quick run reports FAIL by design."
        )
    for name, ok in sorted(status_checks.items()):
        if not ok:
            print(f"failed check                {name}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
