#!/usr/bin/env python3
"""Phase 3 Agent 1 acceptance harness.

Runs the batch-wrapper gates that are too slow for the ordinary pytest run and
writes `reports/phase_3_data/agent_01_batch_equivalence.json`:

- the automated test suite summary;
- >= 100,000 differential state/action comparisons between the batch wrapper and
  independently stepped frozen reference games, across several batch sizes;
- independent-reset trials with full-history isolation checks and generation
  accounting;
- illegal-action inertness trials, including finished and unknown slots.

Every stage is deterministically seeded and runs in a single process: Agent 1
deliberately implements no multiprocessing, shared memory or MPS.

Usage:

    python scripts/run_phase3_agent01.py                 # full acceptance run
    python scripts/run_phase3_agent01.py --quick         # fast smoke run
    python scripts/run_phase3_agent01.py --skip-pytest   # measurements only
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.engine.constants import (  # noqa: E402
    ACTION_SPACE_SIZE,
    IMMOVABLE_TYPES,
    IMPLEMENTATION_VERSION,
    LAKE_SQUARES,
    NOT_TERMINAL,
    OBSERVATION_VERSION,
    RULES_VERSION,
    SCOUT,
    TRAINING_RULES,
    RulesConfig,
    opponent_of,
)
from stratego.engine.coordinates import NEIGHBOURS  # noqa: E402
from stratego.engine.state import state_fingerprint  # noqa: E402
from stratego.training.batch_simulation import (  # noqa: E402
    BATCH_INTERFACE_VERSION,
    BatchIllegalActionError,
    BatchSimulator,
    BatchTerminalStateError,
    UnknownEnvironmentError,
)
from tests.training.differential import (  # noqa: E402
    choose_action,
    reference_game,
    run_differential,
)

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_3_data"
DEFAULT_OUTPUT = DATA_DIRECTORY / "agent_01_batch_equivalence.json"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def step_once(simulator: BatchSimulator, slot: int) -> bool:
    """Apply the shared deterministic policy to one slot. False if finished."""
    if simulator.is_terminal(slot):
        return False
    action = choose_action(
        simulator.root_seed,
        simulator.environment_id(slot),
        simulator.generation(slot),
        simulator.game_state(slot).total_moves,
        simulator.legal_actions(slot),
    )
    simulator.step({slot: action})
    return True


def advance(simulator: BatchSimulator, slot: int, plies: int) -> None:
    for _ in range(plies):
        if not step_once(simulator, slot):
            return


def drive_to_terminal(simulator: BatchSimulator, slot: int, limit: int = 8_000) -> bool:
    for _ in range(limit):
        if not step_once(simulator, slot):
            return True
    return simulator.is_terminal(slot)


def stagger(simulator: BatchSimulator, spread: int = 211) -> None:
    """Put the batch at substantially different plies.

    Uses a coprime stride so consecutive slots land far apart rather than in a
    tight ramp.
    """
    for slot in range(len(simulator)):
        advance(simulator, slot, (slot * 37) % spread)


# ---------------------------------------------------------------------------
# Stage 1: differential equivalence
# ---------------------------------------------------------------------------


def differential_stage(plan: "list[tuple[int, int, int]]") -> dict:
    """Run `run_differential` once per `(batch size, root seed, comparisons)`."""
    runs: list[dict] = []
    behaviors: Counter = Counter()
    terminal_reasons: Counter = Counter()
    totals = {
        "state_action_comparisons": 0,
        "equivalence_mismatches": 0,
        "batch_steps": 0,
        "games_completed": 0,
        "resets": 0,
        "generation_errors": 0,
        "ordinary_moves": 0,
        "scout_multisquare_moves": 0,
        "combats": 0,
        "identity_reveals": 0,
    }
    details: list[dict] = []

    for num_environments, root_seed, comparisons in plan:
        started = time.perf_counter()
        report = run_differential(
            num_environments=num_environments,
            root_seed=root_seed,
            target_comparisons=comparisons,
        )
        elapsed = time.perf_counter() - started
        payload = report.as_dict()
        behaviors.update(report.behavior_counts)
        terminal_reasons.update(report.terminal_reason_counts)
        for key in totals:
            totals[key] += payload[key]
        details.extend(payload["mismatch_details"])
        runs.append(
            {
                "batch_size": num_environments,
                "root_seed": root_seed,
                "state_action_comparisons": payload["state_action_comparisons"],
                "equivalence_mismatches": payload["equivalence_mismatches"],
                "batch_steps": payload["batch_steps"],
                "games_completed": payload["games_completed"],
                "resets": payload["resets"],
                "max_ply_reached": payload["max_ply_reached"],
                "elapsed_seconds": elapsed,
                "comparisons_per_second": payload["state_action_comparisons"] / elapsed,
            }
        )
        print(
            f"  batch {num_environments:>5}: "
            f"{payload['state_action_comparisons']:>7} comparisons, "
            f"{payload['equivalence_mismatches']} mismatches, "
            f"{payload['games_completed']} games, {elapsed:6.1f}s",
            flush=True,
        )

    return {
        **totals,
        "batch_sizes_tested": [size for size, _, _ in plan],
        "behavior_types_observed": dict(sorted(behaviors.items())),
        "terminal_reason_counts": dict(sorted(terminal_reasons.items())),
        "runs": runs,
        "mismatch_details": details,
    }


# ---------------------------------------------------------------------------
# Stage 2: independent reset and generation semantics
# ---------------------------------------------------------------------------


def fresh_game_problems(
    simulator: BatchSimulator, slot: int, rules: RulesConfig
) -> list[str]:
    """Check that a reset slot holds a brand-new legal game and nothing older."""
    problems: list[str] = []
    state = simulator.game_state(slot)

    if state.total_moves != 0:
        problems.append("reset slot did not restart the ply counter")
    if state.battleless_moves != 0:
        problems.append("reset slot kept a battleless counter")
    if state.terminal or state.terminal_reason != NOT_TERMINAL:
        problems.append("reset slot is still terminal")
    if state.winner is not None or state.is_draw:
        problems.append("reset slot kept a result")
    if state.acting_player != rules.first_player:
        problems.append("reset slot does not start with the first player")
    if list(state.recent_moves):
        problems.append("reset slot kept recent moves")
    if state.behavior_memory:
        problems.append("reset slot kept behavioural events")
    if state.active_threat_relations:
        problems.append("reset slot kept threat relations")
    if state.events:
        problems.append("reset slot kept a derived event log")
    if state.action_history:
        problems.append("reset slot kept an action history")
    if not simulator.legal_actions(slot):
        problems.append("reset slot has no legal action")

    for record in state.pieces:
        if not record.alive or record.has_moved or record.capture_ply is not None:
            problems.append("reset slot kept piece history")
            break
        if record.current_square != record.starting_square:
            problems.append("reset slot kept a displaced piece")
            break
        if record.known_to(opponent_of(record.owner)):
            problems.append("reset slot kept opponent knowledge")
            break

    expected = reference_game(
        simulator.root_seed, simulator.environment_id(slot), simulator.generation(slot), rules
    )
    if simulator.slot_fingerprint(slot)[-1] != state_fingerprint(expected):
        problems.append("reset slot is not the game its new generation seeds")
    return problems


def reset_stage(
    *,
    num_environments: int,
    root_seed: int,
    rounds: int,
    per_round: int,
    rules: RulesConfig = TRAINING_RULES,
) -> dict:
    """Terminate and reset selected slots while the rest of the batch stands still."""
    simulator = BatchSimulator(num_environments, root_seed=root_seed, rules=rules)
    stagger(simulator)

    trials = 0
    isolation_checks = 0
    mismatches = 0
    generation_errors = 0
    problems: list[dict] = []
    trajectory_keys: set[tuple[int, int]] = set()
    forced_terminations = 0
    unfinished = 0
    ply_spreads: list[int] = []

    for slot in range(num_environments):
        trajectory_keys.add(simulator.trajectory_key(slot))

    def note(slot: int, found: list[str], stage: str) -> None:
        nonlocal mismatches
        if not found:
            return
        mismatches += len(found)
        if len(problems) < 10:
            problems.append(
                {
                    "stage": stage,
                    "slot": slot,
                    "environment_id": simulator.environment_id(slot),
                    "generation": simulator.generation(slot),
                    "problems": found,
                }
            )

    for round_index in range(rounds):
        selected = sorted(
            (round_index * per_round + offset) % num_environments
            for offset in range(per_round)
        )
        selected = sorted(set(selected))
        untouched = [slot for slot in range(num_environments) if slot not in set(selected)]

        # Keep the batch at mixed plies rather than letting it converge.
        for slot in untouched[:: max(1, len(untouched) // 4)]:
            advance(simulator, slot, 1 + (slot + round_index) % 23)

        plies = [simulator.game_state(slot).total_moves for slot in range(num_environments)]
        ply_spreads.append(max(plies) - min(plies))

        # Force only the selected slots to terminate.
        before_forcing = {
            slot: simulator.slot_fingerprint(slot) for slot in untouched
        }
        for slot in selected:
            if drive_to_terminal(simulator, slot):
                forced_terminations += 1
            else:  # pragma: no cover - the absolute move limit always terminates
                unfinished += 1
        for slot in untouched:
            isolation_checks += 1
            if simulator.slot_fingerprint(slot) != before_forcing[slot]:
                note(slot, ["slot advanced while another slot was driven to terminal"], "drive")

        before_reset = {slot: simulator.slot_fingerprint(slot) for slot in untouched}
        previous_generations = {slot: simulator.generation(slot) for slot in selected}
        previous_fingerprints = {slot: simulator.slot_fingerprint(slot) for slot in selected}
        previous_environment_ids = {slot: simulator.environment_id(slot) for slot in selected}

        new_generations = simulator.reset_slots(selected)

        for slot in untouched:
            isolation_checks += 1
            if simulator.slot_fingerprint(slot) != before_reset[slot]:
                note(slot, ["slot changed during another slot's reset"], "reset")

        for position, slot in enumerate(selected):
            trials += 1
            found: list[str] = []
            if simulator.generation(slot) != previous_generations[slot] + 1:
                generation_errors += 1
                found.append("generation did not increment exactly once")
            if new_generations[position] != simulator.generation(slot):
                generation_errors += 1
                found.append("reset_slots reported the wrong generation")
            if simulator.environment_id(slot) != previous_environment_ids[slot]:
                generation_errors += 1
                found.append("environment_id changed across a reset")
            key = simulator.trajectory_key(slot)
            if key in trajectory_keys:
                generation_errors += 1
                found.append(f"trajectory key {key} was reused")
            trajectory_keys.add(key)
            if simulator.slot_fingerprint(slot) == previous_fingerprints[slot]:
                found.append("reset produced an identical game")
            found.extend(fresh_game_problems(simulator, slot, rules))
            note(slot, found, "fresh")

    return {
        "independent_reset_trials": trials,
        "reset_mismatches": mismatches,
        "generation_errors": generation_errors,
        "reset_isolation_checks": isolation_checks,
        "forced_terminations": forced_terminations,
        "unfinished_forced_games": unfinished,
        "distinct_trajectory_keys": len(trajectory_keys),
        "mean_ply_spread": float(np.mean(ply_spreads)) if ply_spreads else 0.0,
        "max_ply_spread": max(ply_spreads) if ply_spreads else 0,
        "reset_problem_details": problems,
    }


# ---------------------------------------------------------------------------
# Stage 3: illegal-action inertness
# ---------------------------------------------------------------------------


def illegal_candidates(simulator: BatchSimulator, slot: int) -> list[tuple[str, int]]:
    """Labelled illegal action identifiers for one slot's current position.

    Every candidate is checked against the legal-action list before it is
    returned, so a category that happens to be legal in this position is dropped
    rather than counted.
    """
    state = simulator.game_state(slot)
    player = state.acting_player
    opponent = opponent_of(player)
    legal = set(simulator.legal_actions(slot))
    candidates: list[tuple[str, int]] = []

    own = [record for record in state.pieces_of(player) if record.alive]
    enemy = [record for record in state.pieces_of(opponent) if record.alive]
    mover = next((record for record in own if record.true_type not in IMMOVABLE_TYPES), None)
    immovable = next((record for record in own if record.true_type in IMMOVABLE_TYPES), None)
    non_scout = next(
        (
            record
            for record in own
            if record.true_type not in IMMOVABLE_TYPES and record.true_type != SCOUT
        ),
        None,
    )

    if mover is not None:
        source = mover.current_square
        row, column = divmod(source, 10)
        candidates.append(("into_lake", 100 * source + LAKE_SQUARES[0]))
        candidates.append(("stand_still", 100 * source + source))
        # Diagonal step: never legal for any piece type.
        if row < 9 and column < 9:
            candidates.append(("diagonal", 100 * source + source + 11))
        friend = next(
            (record for record in own if record.piece_id != mover.piece_id), None
        )
        if friend is not None:
            candidates.append(("onto_own_piece", 100 * source + friend.current_square))
    if non_scout is not None:
        source = non_scout.current_square
        row, column = divmod(source, 10)
        # Two squares in a straight cardinal line: legal only for a Scout.
        for row_step, column_step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            far_row, far_column = row + 2 * row_step, column + 2 * column_step
            if not (0 <= far_row < 10 and 0 <= far_column < 10):
                continue
            far = far_row * 10 + far_column
            if far in LAKE_SQUARES:
                continue
            candidates.append(("non_scout_two_squares", 100 * source + far))
            break
    if immovable is not None:
        source = immovable.current_square
        for neighbour in NEIGHBOURS[source]:
            if state.board[neighbour] is None:
                candidates.append(("immovable_piece", 100 * source + neighbour))
                break
    if enemy:
        source = enemy[0].current_square
        for neighbour in NEIGHBOURS[source]:
            if state.board[neighbour] is None:
                candidates.append(("wrong_player", 100 * source + neighbour))
                break
    empty = next(
        (
            square
            for square in range(100)
            if state.board[square] is None and square not in LAKE_SQUARES
        ),
        None,
    )
    if empty is not None:
        for neighbour in NEIGHBOURS[empty]:
            candidates.append(("empty_source", 100 * empty + neighbour))
            break
    candidates.append(("above_action_space", ACTION_SPACE_SIZE + 7))
    candidates.append(("negative", -5))

    return [(label, action) for label, action in candidates if action not in legal]


def illegal_action_stage(
    *,
    num_environments: int,
    root_seed: int,
    rounds: int,
    rules: RulesConfig = TRAINING_RULES,
) -> dict:
    """Submit illegal actions through the batch API and prove nothing mutates."""
    simulator = BatchSimulator(num_environments, root_seed=root_seed, rules=rules)
    stagger(simulator, spread=97)

    trials = 0
    failures = 0
    full_history_checks = 0
    categories: Counter = Counter()
    failure_details: list[dict] = []

    def signature() -> tuple:
        """Cheap complete signature of the whole batch.

        The history-free fingerprint covers board, pieces, counters, recent
        moves, threat relations and behavioural memory; the second element covers
        the derived event log and the action history, which `apply_action`
        appends to last. Comparing the action history and the event count is
        enough to detect a partial mutation there without re-hashing every event
        on every one of thousands of trials -- the full-history fingerprint is
        compared once per round instead.
        """
        return (
            simulator.batch_fingerprint(include_history=False),
            tuple(
                (
                    len(simulator.game_state(slot).events),
                    tuple(simulator.game_state(slot).action_history),
                )
                for slot in range(len(simulator))
            ),
        )

    def legal_companions(exclude: int) -> dict[int, int]:
        """A legal action for every active slot except `exclude`."""
        return {
            slot: simulator.legal_actions(slot)[0]
            for slot in simulator.active_slots()
            if slot != exclude
        }

    def expect_rejection(actions: dict[int, int], error: type, label: str) -> None:
        nonlocal trials, failures
        before = signature()
        trials += 1
        categories[label] += 1
        try:
            simulator.step(actions)
        except error:
            pass
        except Exception as unexpected:  # pragma: no cover - defensive
            failures += 1
            failure_details.append({"category": label, "error": repr(unexpected)})
            return
        else:
            failures += 1
            failure_details.append({"category": label, "error": "no error raised"})
            return
        if signature() != before:
            failures += 1
            failure_details.append({"category": label, "error": "batch was mutated"})

    for round_index in range(rounds):
        round_start = simulator.batch_fingerprint()
        for slot in simulator.active_slots():
            for label, action in illegal_candidates(simulator, slot):
                actions = legal_companions(slot)
                actions[slot] = action
                expect_rejection(actions, BatchIllegalActionError, label)

        # An unknown slot index must not step the slots that were valid.
        actions = legal_companions(-1)
        actions[num_environments + round_index] = 0
        expect_rejection(actions, UnknownEnvironmentError, "unknown_slot")

        full_history_checks += 1
        if simulator.batch_fingerprint() != round_start:
            failures += 1
            failure_details.append(
                {"category": "round_full_history", "error": "batch was mutated"}
            )

        # Advance the whole batch so the next round tests different positions.
        simulator.step(legal_companions(-1))

    # A finished slot must refuse to step, alongside otherwise legal actions.
    finished = 0
    for slot in range(num_environments):
        if drive_to_terminal(simulator, slot):
            finished += 1
        if finished >= 2:
            break
    for slot in simulator.finished_slots():
        actions = legal_companions(slot)
        actions[slot] = 0
        expect_rejection(actions, BatchTerminalStateError, "finished_slot")

    return {
        "illegal_action_inert_trials": trials,
        "illegal_action_inert_failures": failures,
        "illegal_action_categories": dict(sorted(categories.items())),
        "illegal_action_full_history_checks": full_history_checks,
        "illegal_action_failure_details": failure_details,
    }


# ---------------------------------------------------------------------------
# Stage 4: automated test suite summary
# ---------------------------------------------------------------------------


def pytest_stage() -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
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
    "stratego/training/__init__.py",
    "stratego/training/batch_simulation.py",
    "tests/training/__init__.py",
    "tests/training/differential.py",
    "tests/training/test_batch_simulation.py",
    "scripts/run_phase3_agent01.py",
    "reports/phase_3_data/agent_01_batch_equivalence.json",
    "reports/phase_3_implementation_report.md",
]

FILES_MODIFIED: list[str] = []


def _max_rss() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    return usage if sys.platform == "darwin" else usage * 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset-rounds", type=int, default=40)
    parser.add_argument("--reset-environments", type=int, default=32)
    parser.add_argument("--reset-per-round", type=int, default=8)
    parser.add_argument("--illegal-rounds", type=int, default=48)
    parser.add_argument("--illegal-environments", type=int, default=12)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    options = parser.parse_args()

    # (batch size, root seed, comparisons). Batch size 1 is the degenerate case
    # and 256 is the Phase 3 engineering scale; the total clears the 100,000
    # acceptance floor with margin.
    plan = [
        (1, 301, 6_000),
        (8, 302, 18_000),
        (64, 303, 46_000),
        (256, 304, 60_000),
    ]
    if options.quick:
        plan = [(1, 301, 300), (8, 302, 600), (64, 303, 900)]
        options.reset_rounds = 3
        options.reset_environments = 8
        options.reset_per_round = 2
        options.illegal_rounds = 2
        options.illegal_environments = 4

    started = time.perf_counter()

    print("stage: differential equivalence", flush=True)
    differential = differential_stage(plan)

    print("stage: independent reset", flush=True)
    reset = reset_stage(
        num_environments=options.reset_environments,
        root_seed=401,
        rounds=options.reset_rounds,
        per_round=options.reset_per_round,
    )
    print(
        f"  {reset['independent_reset_trials']} reset trials, "
        f"{reset['reset_mismatches']} mismatches, "
        f"{reset['generation_errors']} generation errors",
        flush=True,
    )

    print("stage: illegal-action inertness", flush=True)
    illegal = illegal_action_stage(
        num_environments=options.illegal_environments,
        root_seed=501,
        rounds=options.illegal_rounds,
    )
    print(
        f"  {illegal['illegal_action_inert_trials']} trials, "
        f"{illegal['illegal_action_inert_failures']} failures",
        flush=True,
    )

    if options.skip_pytest:
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
        }
    else:
        print("stage: pytest", flush=True)
        tests = pytest_stage()
        print(f"  {tests['passed']} passed, {tests['failed']} failed", flush=True)

    elapsed = time.perf_counter() - started

    behaviors = differential["behavior_types_observed"]
    passed = (
        differential["state_action_comparisons"] >= 100_000
        and differential["equivalence_mismatches"] == 0
        and reset["reset_mismatches"] == 0
        and reset["generation_errors"] == 0
        and reset["independent_reset_trials"] > 0
        and illegal["illegal_action_inert_failures"] == 0
        and illegal["illegal_action_inert_trials"] > 0
        and differential["scout_multisquare_moves"] > 0
        and differential["combats"] > 0
        and differential["identity_reveals"] > 0
        and len(behaviors) == 5
        and bool(differential["terminal_reason_counts"])
        and (options.skip_pytest or (tests["failed"] == 0 and tests["exit_code"] == 0))
    )

    metrics = {
        "agent": "agent_01_batch_wrapper",
        "status": "PASS" if passed else "FAIL",
        "implementation_version": IMPLEMENTATION_VERSION,
        "batch_interface_version": BATCH_INTERFACE_VERSION,
        "rules_version": RULES_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "multiprocessing_used": False,
        "quick_mode": options.quick,
        "test_total": tests["total"],
        "test_passed": tests["passed"],
        "test_failed": tests["failed"],
        "test_skipped": tests["skipped"],
        "test_expected_failures": tests["expected_failures"],
        "test_errors": tests["errors"],
        "test_exit_code": tests["exit_code"],
        "test_seconds": tests["seconds"],
        "test_failure_lines": tests["failure_lines"],
        "batch_sizes_tested": differential["batch_sizes_tested"],
        "state_action_comparisons": differential["state_action_comparisons"],
        "equivalence_mismatches": differential["equivalence_mismatches"],
        "equivalence_mismatch_details": differential["mismatch_details"],
        "batch_steps": differential["batch_steps"],
        "games_completed": differential["games_completed"],
        "differential_resets": differential["resets"],
        "independent_reset_trials": reset["independent_reset_trials"],
        "reset_mismatches": reset["reset_mismatches"],
        "reset_isolation_checks": reset["reset_isolation_checks"],
        "forced_terminations": reset["forced_terminations"],
        "mean_ply_spread": reset["mean_ply_spread"],
        "max_ply_spread": reset["max_ply_spread"],
        "distinct_trajectory_keys": reset["distinct_trajectory_keys"],
        "reset_problem_details": reset["reset_problem_details"],
        "generation_errors": reset["generation_errors"] + differential["generation_errors"],
        "illegal_action_inert_trials": illegal["illegal_action_inert_trials"],
        "illegal_action_inert_failures": illegal["illegal_action_inert_failures"],
        "illegal_action_categories": illegal["illegal_action_categories"],
        "illegal_action_full_history_checks": illegal["illegal_action_full_history_checks"],
        "illegal_action_failure_details": illegal["illegal_action_failure_details"],
        "behavior_types_observed": behaviors,
        "terminal_reason_counts": differential["terminal_reason_counts"],
        "ordinary_moves": differential["ordinary_moves"],
        "scout_multisquare_moves": differential["scout_multisquare_moves"],
        "combats": differential["combats"],
        "identity_reveals": differential["identity_reveals"],
        "differential_runs": differential["runs"],
        "peak_memory_bytes": _max_rss(),
        "elapsed_seconds": elapsed,
        "files_created": FILES_CREATED,
        "files_modified": FILES_MODIFIED,
    }

    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    print(f"\nstatus: {metrics['status']}")
    print(f"comparisons: {metrics['state_action_comparisons']}")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"wrote {options.output.relative_to(REPOSITORY_ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
