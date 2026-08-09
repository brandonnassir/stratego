#!/usr/bin/env python3
"""Phase Two acceptance harness.

Runs the large-scale gates that are too slow for the ordinary pytest run and
writes `reports/phase_2_metrics.json`:

- deterministic replay over >= 10,000 complete games;
- >= 100,000 valid hidden-identity permutation trials;
- state-invariant stress over complete games;
- snapshot/restore across every required game phase;
- legal-action list versus mask agreement;
- the exhaustive combat matrix;
- random-game statistics;
- performance and storage baselines.

Every stage is deterministically seeded. Work is split across processes only to
shorten wall-clock time; each worker owns a fixed seed range, so results do not
depend on the worker count.

Usage:

    python scripts/run_phase2_validation.py                 # full acceptance run
    python scripts/run_phase2_validation.py --quick         # fast smoke run
    python scripts/run_phase2_validation.py --skip-pytest   # metrics only
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import re
import resource
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import psutil

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.engine.combat import resolve_combat  # noqa: E402
from stratego.engine.constants import (  # noqa: E402
    BLUE,
    IMPLEMENTATION_VERSION,
    OBSERVATION_VERSION,
    PIECE_TYPE_BY_NAME,
    RED,
    RULES_VERSION,
    TRAINING_RULES,
)
from stratego.engine.events import filter_events_for_observer, public_board_view  # noqa: E402
from stratego.engine.invariants import (  # noqa: E402
    capture_baseline,
    capture_knowledge,
    check_invariants,
)
from stratego.engine.legal_moves import legal_action_mask, legal_actions  # noqa: E402
from stratego.engine.observation import build_observation  # noqa: E402
from stratego.engine.permutation import (  # noqa: E402
    belief_targets_differ,
    compare_public_surfaces,
    permute_hidden_identities,
    public_surface,
)
from stratego.engine.random_play import (  # noqa: E402
    make_random_setups,
    play_random_game,
    select_random_action,
)
from stratego.engine.replay import (  # noqa: E402
    build_replay_record,
    initial_state_from_record,
    rebuild_final_state,
    replay_plies,
)
from stratego.engine.snapshot import (  # noqa: E402
    clone_state,
    create_snapshot,
    restore_snapshot,
    snapshot_to_json,
)
from stratego.engine.state import create_game, state_fingerprint  # noqa: E402
from stratego.engine.transition import apply_action  # noqa: E402

# Import the literal expected combat table from the test suite so the harness
# and the unit tests cannot drift apart.
from tests.engine.test_combat import COMBAT_CASES  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def ply_digest(state, events) -> bytes:
    """Digest of everything replay is required to reproduce at one ply."""
    hasher = hashlib.blake2b(digest_size=16)

    hasher.update(repr(tuple(state.board)).encode())
    for record in state.pieces:
        hasher.update(
            repr(
                (
                    record.current_square,
                    record.alive,
                    record.has_moved,
                    record.known_to_red,
                    record.known_to_blue,
                    record.reveal_reason_red,
                    record.reveal_reason_blue,
                    record.capture_ply,
                )
            ).encode()
        )
    hasher.update(
        repr(
            (
                state.acting_player,
                state.total_moves,
                state.battleless_moves,
                state.terminal,
                state.terminal_reason,
                state.winner,
                state.is_draw,
                tuple(move.as_tuple() for move in state.recent_moves),
                tuple(sorted(state.active_threat_relations)),
                tuple(sorted(event.as_tuple() for event in state.behavior_memory.values())),
            )
        ).encode()
    )

    actions = legal_actions(state)
    hasher.update(repr(actions).encode())
    hasher.update(legal_action_mask(state, actions).tobytes())
    hasher.update(build_observation(state, RED).tobytes())
    hasher.update(build_observation(state, BLUE).tobytes())
    hasher.update(json.dumps(events, sort_keys=True).encode())
    return hasher.digest()


def chunk_ranges(total: int, workers: int) -> list[tuple[int, int]]:
    """Split `range(total)` into `workers` contiguous, deterministic blocks."""
    if total <= 0:
        return []
    workers = max(1, min(workers, total))
    size, remainder = divmod(total, workers)
    ranges = []
    start = 0
    for index in range(workers):
        length = size + (1 if index < remainder else 0)
        ranges.append((start, start + length))
        start += length
    return ranges


def run_parallel(function, arguments, workers):
    """Map `function` over `arguments`, in-process when `workers <= 1`."""
    if workers <= 1:
        return [function(argument) for argument in arguments]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, arguments))


def merge_counters(results, key):
    total = Counter()
    for result in results:
        total.update(result[key])
    return total


# ---------------------------------------------------------------------------
# Stage 1: random-game statistics and replay reconstruction
# ---------------------------------------------------------------------------

REPLAY_SEED_BASE = 1_000_000


def play_with_digests(game_seed: int):
    """Play one seeded random game, recording a digest after every ply.

    This mirrors `play_random_game` exactly, including its seed derivation, but
    captures the per-ply digest while the game is *first* played. The replay
    pass is then compared against these live digests rather than against a
    second replay, which is what makes the comparison meaningful.
    """
    red_setup, blue_setup = make_random_setups(game_seed)
    state = create_game(
        red_setup, blue_setup, rules=TRAINING_RULES, game_id=f"random-{game_seed}"
    )
    digests = [ply_digest(state, [])]

    rng = random.Random(game_seed + 1_000_003)
    while not state.terminal:
        actions = legal_actions(state)
        events = apply_action(state, select_random_action(state, rng, actions), legal=actions)
        digests.append(ply_digest(state, events))

    record = build_replay_record(
        state,
        red_setup,
        blue_setup,
        seeds={"game_seed": game_seed, "agent": "uniform_random_legal"},
    )
    return state, record, digests


def replay_worker(bounds: tuple[int, int]) -> dict:
    """Play, record and replay every game in a seed range."""
    start, end = bounds
    lengths: list[int] = []
    reasons: Counter = Counter()
    results: Counter = Counter()
    plies = 0
    state_mismatches = 0
    observation_mismatches = 0
    event_mismatches = 0
    result_mismatches = 0
    replay_bytes = 0

    for seed in range(start, end):
        game_seed = REPLAY_SEED_BASE + seed
        original, record, live_digests = play_with_digests(game_seed)

        lengths.append(original.total_moves)
        reasons[original.terminal_reason] += 1
        results[record.terminal_result] += 1
        replay_bytes += len(record.to_json())
        plies += original.total_moves

        replay_digests = [
            ply_digest(state, events) for _, state, events in replay_plies(record)
        ]
        if replay_digests != live_digests:
            # The digest covers board, piece records, knowledge, behaviour
            # memory, counters, legal actions, both observations and the ply's
            # events, so any of them disagreeing lands here.
            state_mismatches += 1

        replayed = rebuild_final_state(record)
        if state_fingerprint(replayed) != state_fingerprint(original):
            state_mismatches += 1
        if replayed.events != original.events:
            event_mismatches += 1
        if (
            replayed.terminal_reason != original.terminal_reason
            or replayed.winner != original.winner
            or replayed.total_moves != original.total_moves
        ):
            result_mismatches += 1
        if not np.array_equal(
            build_observation(replayed, RED), build_observation(original, RED)
        ) or not np.array_equal(
            build_observation(replayed, BLUE), build_observation(original, BLUE)
        ):
            observation_mismatches += 1

    return {
        "games": end - start,
        "plies": plies,
        "lengths": lengths,
        "reasons": reasons,
        "results": results,
        "state_mismatches": state_mismatches,
        "observation_mismatches": observation_mismatches,
        "event_mismatches": event_mismatches,
        "result_mismatches": result_mismatches,
        "replay_bytes": replay_bytes,
    }


def replay_stage(games: int, workers: int) -> dict:
    started = time.perf_counter()
    results = run_parallel(replay_worker, chunk_ranges(games, workers), workers)
    elapsed = time.perf_counter() - started

    lengths = [length for result in results for length in result["lengths"]]
    reasons = merge_counters(results, "reasons")
    outcomes = merge_counters(results, "results")

    return {
        "games": sum(result["games"] for result in results),
        "plies": sum(result["plies"] for result in results),
        "state_mismatches": sum(result["state_mismatches"] for result in results),
        "observation_mismatches": sum(result["observation_mismatches"] for result in results),
        "event_mismatches": sum(result["event_mismatches"] for result in results),
        "result_mismatches": sum(result["result_mismatches"] for result in results),
        "mean_replay_bytes": (
            sum(result["replay_bytes"] for result in results) / max(1, len(lengths))
        ),
        "lengths": lengths,
        "terminal_reason_counts": dict(sorted(reasons.items())),
        "outcome_counts": dict(sorted(outcomes.items())),
        "seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Stage 2: hidden-identity anti-leak trials
# ---------------------------------------------------------------------------

ANTILEAK_SEED_BASE = 2_000_000


def sample_position(seed: int):
    """A reproducible mid-game position drawn from a seeded random game."""
    rng = random.Random(seed)
    target = rng.choice([8, 20, 35, 55, 80, 110, 150, 200, 260])
    state, _ = play_random_game(ANTILEAK_SEED_BASE + seed, max_plies=target)
    return state


def antileak_worker(task: tuple[int, int, int]) -> dict:
    """Run permutation trials for a block of sampled positions."""
    first_position, last_position, trials_per_position = task
    attempted = 0
    valid = 0
    changed = 0
    positions = 0
    skipped_positions = 0
    mismatches = Counter()
    belief_control_checks = 0
    belief_control_failures = 0

    for index in range(first_position, last_position):
        state = sample_position(index)
        if state.terminal:
            skipped_positions += 1
            continue
        observer = state.acting_player
        baseline = public_surface(state, observer)
        rng = random.Random(index * 7919 + 13)
        positions += 1

        for _ in range(trials_per_position):
            permuted, info = permute_hidden_identities(state, observer, rng)
            attempted += info["attempts"] if info["attempts"] else 1
            if not info["valid"]:
                continue
            valid += 1
            if info["changed"]:
                changed += 1

            candidate = public_surface(permuted, observer)
            for category, count in compare_public_surfaces(baseline, candidate).items():
                if count:
                    mismatches[category] += count

            if info["changed"]:
                belief_control_checks += 1
                if not belief_targets_differ(state, permuted, observer):
                    belief_control_failures += 1

    return {
        "attempted": attempted,
        "valid": valid,
        "changed": changed,
        "positions": positions,
        "skipped_positions": skipped_positions,
        "mismatches": mismatches,
        "belief_control_checks": belief_control_checks,
        "belief_control_failures": belief_control_failures,
    }


def antileak_stage(trials: int, trials_per_position: int, workers: int) -> dict:
    positions_needed = max(1, trials // trials_per_position)
    started = time.perf_counter()
    tasks = [
        (start, end, trials_per_position)
        for start, end in chunk_ranges(positions_needed, workers)
    ]
    results = run_parallel(antileak_worker, tasks, workers)
    elapsed = time.perf_counter() - started

    mismatches = merge_counters(results, "mismatches")
    return {
        "attempted": sum(result["attempted"] for result in results),
        "valid": sum(result["valid"] for result in results),
        "changed": sum(result["changed"] for result in results),
        "positions": sum(result["positions"] for result in results),
        "skipped_positions": sum(result["skipped_positions"] for result in results),
        "observation_mismatches": mismatches.get("observation", 0),
        "action_mismatches": mismatches.get("legal_actions", 0),
        "event_mismatches": mismatches.get("events", 0),
        "public_view_mismatches": mismatches.get("board_view", 0)
        + mismatches.get("setup_view", 0),
        "belief_control_checks": sum(result["belief_control_checks"] for result in results),
        "belief_control_failures": sum(
            result["belief_control_failures"] for result in results
        ),
        "seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Stage 3: state-invariant stress
# ---------------------------------------------------------------------------

INVARIANT_SEED_BASE = 3_000_000


def invariant_worker(bounds: tuple[int, int]) -> dict:
    start, end = bounds
    transitions = 0
    violations: list[str] = []

    for seed in range(start, end):
        game_seed = INVARIANT_SEED_BASE + seed
        red_setup, blue_setup = make_random_setups(game_seed)
        state = create_game(
            red_setup, blue_setup, rules=TRAINING_RULES, game_id=f"invariant-{game_seed}"
        )
        baseline = capture_baseline(state)
        knowledge = capture_knowledge(state)
        rng = random.Random(game_seed + 1_000_003)
        try:
            check_invariants(state, baseline=baseline)
            while not state.terminal:
                actions = legal_actions(state)
                apply_action(state, select_random_action(state, rng, actions), legal=actions)
                check_invariants(state, baseline=baseline, previous_knowledge=knowledge)
                knowledge = capture_knowledge(state)
                transitions += 1
        except AssertionError as error:
            violations.append(f"seed {game_seed}: {error}")

    return {"games": end - start, "transitions": transitions, "violations": violations}


def invariant_stage(games: int, workers: int) -> dict:
    started = time.perf_counter()
    results = run_parallel(invariant_worker, chunk_ranges(games, workers), workers)
    elapsed = time.perf_counter() - started
    violations = [item for result in results for item in result["violations"]]
    return {
        "games": sum(result["games"] for result in results),
        "transitions": sum(result["transitions"] for result in results),
        "violations": len(violations),
        "violation_details": violations[:20],
        "seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Stage 4: snapshot / restore
# ---------------------------------------------------------------------------

SNAPSHOT_SEED_BASE = 4_000_000

SNAPSHOT_PHASES = (
    "early_game",
    "middle_game",
    "late_game",
    "before_combat",
    "after_combat",
    "near_draw_limit",
)


def snapshot_positions(seed: int):
    """One representative position per required snapshot phase."""
    rng = random.Random(seed)
    positions = {}

    for label, target in (("early_game", 6), ("middle_game", 60), ("late_game", 200)):
        state, _ = play_random_game(SNAPSHOT_SEED_BASE + seed, max_plies=target)
        if not state.terminal:
            positions[label] = state

    # Walk a game until a combat is about to happen, and capture both sides of it.
    state, _ = play_random_game(SNAPSHOT_SEED_BASE + seed, max_plies=0)
    walker = random.Random(seed + 77)
    for _ in range(400):
        if state.terminal:
            break
        actions = legal_actions(state)
        attacks = [
            action for action in actions if state.board[action % 100] is not None
        ]
        if attacks and "before_combat" not in positions:
            positions["before_combat"] = clone_state(state)
            apply_action(state, attacks[0])
            if not state.terminal:
                positions["after_combat"] = clone_state(state)
            continue
        apply_action(state, select_random_action(state, walker, actions))

    # A position sitting close to the battleless draw threshold.
    state, _ = play_random_game(SNAPSHOT_SEED_BASE + seed, max_plies=0)
    walker = random.Random(seed + 101)
    for _ in range(1500):
        if state.terminal:
            break
        if state.battleless_moves >= state.rules.battleless_move_limit - 3:
            positions["near_draw_limit"] = clone_state(state)
            break
        apply_action(state, select_random_action(state, walker))

    return positions


def snapshot_worker(bounds: tuple[int, int]) -> dict:
    start, end = bounds
    tested = Counter()
    mismatches = Counter()

    for seed in range(start, end):
        for label, state in snapshot_positions(seed).items():
            tested[label] += 1
            snapshot = create_snapshot(state, include_history=True)
            restored = restore_snapshot(snapshot)

            if state_fingerprint(restored) != state_fingerprint(state):
                mismatches["state"] += 1
            if legal_actions(restored) != legal_actions(state):
                mismatches["legal_actions"] += 1
            for observer in (RED, BLUE):
                if not np.array_equal(
                    build_observation(restored, observer), build_observation(state, observer)
                ):
                    mismatches["observation"] += 1
                if public_board_view(restored, observer) != public_board_view(state, observer):
                    mismatches["public_view"] += 1
                if filter_events_for_observer(
                    restored.events, observer
                ) != filter_events_for_observer(state.events, observer):
                    mismatches["public_events"] += 1

            if not state.terminal:
                action = legal_actions(state)[0]
                original_events = apply_action(clone_state(state), action)
                restored_events = apply_action(restored, action)
                if original_events != restored_events:
                    mismatches["next_transition"] += 1

    return {"tested": tested, "mismatches": mismatches}


def snapshot_stage(seeds: int, workers: int) -> dict:
    started = time.perf_counter()
    results = run_parallel(snapshot_worker, chunk_ranges(seeds, workers), workers)
    elapsed = time.perf_counter() - started
    tested = merge_counters(results, "tested")
    mismatches = merge_counters(results, "mismatches")
    return {
        "snapshots": sum(tested.values()),
        "by_phase": {phase: tested.get(phase, 0) for phase in SNAPSHOT_PHASES},
        "legal_action_mismatches": mismatches.get("legal_actions", 0),
        "observation_mismatches": mismatches.get("observation", 0),
        "public_event_mismatches": mismatches.get("public_events", 0),
        "public_view_mismatches": mismatches.get("public_view", 0),
        "next_transition_mismatches": mismatches.get("next_transition", 0),
        "state_mismatches": mismatches.get("state", 0),
        "total_mismatches": sum(mismatches.values()),
        "seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Stage 5: legal-action list versus mask
# ---------------------------------------------------------------------------

LEGAL_SEED_BASE = 5_000_000


def legal_worker(bounds: tuple[int, int]) -> dict:
    start, end = bounds
    positions = 0
    discrepancies = 0
    empty_positions = 0
    largest = 0

    for seed in range(start, end):
        state, _ = play_random_game(LEGAL_SEED_BASE + seed, max_plies=0)
        walker = random.Random(seed)
        for _ in range(40):
            if state.terminal:
                break
            actions = legal_actions(state)
            mask = legal_action_mask(state)
            positions += 1
            largest = max(largest, len(actions))
            if len(actions) == 0:
                empty_positions += 1
            if set(np.flatnonzero(mask).tolist()) != set(actions):
                discrepancies += 1
            if int(mask.sum()) != len(actions):
                discrepancies += 1
            if len(actions) != len(set(actions)) or actions != sorted(actions):
                discrepancies += 1
            apply_action(state, select_random_action(state, walker, actions), legal=actions)

    return {
        "positions": positions,
        "discrepancies": discrepancies,
        "empty_positions": empty_positions,
        "largest": largest,
    }


def legal_stage(seeds: int, workers: int) -> dict:
    started = time.perf_counter()
    results = run_parallel(legal_worker, chunk_ranges(seeds, workers), workers)
    return {
        "positions": sum(result["positions"] for result in results),
        "discrepancies": sum(result["discrepancies"] for result in results),
        "empty_positions": sum(result["empty_positions"] for result in results),
        "largest_action_count": max(result["largest"] for result in results),
        "seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# Stage 6: combat matrix
# ---------------------------------------------------------------------------


def combat_stage() -> dict:
    failures = []
    for attacker, defender, expected in COMBAT_CASES:
        outcome = resolve_combat(PIECE_TYPE_BY_NAME[attacker], PIECE_TYPE_BY_NAME[defender])
        if outcome != expected:
            failures.append(f"{attacker} vs {defender}: {outcome} != {expected}")
    return {
        "cases_tested": len(COMBAT_CASES),
        "cases_failed": len(failures),
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Stage 7: performance and storage baselines
# ---------------------------------------------------------------------------


def performance_stage(sample_states, repeats: int) -> dict:
    """Single-threaded component throughput on representative positions."""
    from stratego.engine.behavior import (
        build_behavior_events,
        capture_pre_move_context,
        compute_threat_relations,
    )

    def timed(function, iterations):
        gc.collect()
        started = time.perf_counter()
        for index in range(iterations):
            function(index)
        return (time.perf_counter() - started) / iterations

    states = sample_states
    count = len(states)

    legal_seconds = timed(lambda index: legal_actions(states[index % count]), repeats)
    mask_seconds = timed(lambda index: legal_action_mask(states[index % count]), repeats)
    observation_seconds = timed(
        lambda index: build_observation(states[index % count]), repeats
    )
    snapshot_seconds = timed(lambda index: create_snapshot(states[index % count]), repeats)

    # Transitions run on pre-built throwaway clones with their legal-action
    # lists already generated, so the measurement covers the transition alone.
    transition_samples = max(200, repeats // 8)
    clones = [clone_state(states[index % count]) for index in range(transition_samples)]
    clone_actions = [legal_actions(clone) for clone in clones]
    gc.collect()
    started = time.perf_counter()
    for clone, actions in zip(clones, clone_actions):
        apply_action(clone, actions[0], legal=actions)
    transition_seconds = (time.perf_counter() - started) / transition_samples
    del clones, clone_actions

    # Behavioural processing is the part of a transition contributed by
    # `behavior.py`; measured directly to estimate its share.
    def behaviour_once(index):
        state = states[index % count]
        actions = legal_actions(state)
        source, destination = divmod(actions[index % len(actions)], 100)
        context = capture_pre_move_context(state, source, destination)
        relations = compute_threat_relations(state, context)
        build_behavior_events(state, context, relations)

    behaviour_seconds = timed(behaviour_once, repeats // 2)

    gc.collect()
    started = time.perf_counter()
    game_plies = 0
    game_count = 60
    for seed in range(game_count):
        state, _ = play_random_game(9_000_000 + seed)
        game_plies += state.total_moves
    games_seconds = time.perf_counter() - started

    return {
        "legal_actions_per_second": 1.0 / legal_seconds,
        "legal_action_masks_per_second": 1.0 / mask_seconds,
        "observations_per_second": 1.0 / observation_seconds,
        "snapshots_per_second": 1.0 / snapshot_seconds,
        "state_transitions_per_second": 1.0 / transition_seconds,
        "random_games_per_second": game_count / games_seconds,
        "random_game_plies_per_second": game_plies / games_seconds,
        "component_seconds": {
            "legal_action_generation": legal_seconds,
            "observation_construction": observation_seconds,
            "state_transition_total": transition_seconds,
            "behavioural_processing": behaviour_seconds,
        },
    }


def storage_stage(sample_states) -> dict:
    snapshot_sizes = [
        len(snapshot_to_json(create_snapshot(state))) for state in sample_states[:20]
    ]
    replay_sizes = []
    for seed in range(200):
        _, record = play_random_game(8_000_000 + seed)
        replay_sizes.append(len(record.to_json()))

    mean_replay = statistics.mean(replay_sizes)
    return {
        "snapshot_serialized_bytes": int(statistics.mean(snapshot_sizes)),
        "mean_replay_serialized_bytes": int(mean_replay),
        "median_replay_serialized_bytes": int(statistics.median(replay_sizes)),
        "thousand_replays_bytes": int(mean_replay * 1000),
        "estimated_million_game_storage_bytes": int(mean_replay * 1_000_000),
    }


# ---------------------------------------------------------------------------
# Stage 8: automated test suite summary
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

    failure_lines = [line for line in output.splitlines() if line.startswith("FAILED")]
    skip_lines = [line for line in output.splitlines() if line.startswith("SKIPPED")]

    return {
        "total": passed + failed + skipped + xfailed + errors,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "expected_failures": xfailed,
        "errors": errors,
        "seconds": elapsed,
        "exit_code": completed.returncode,
        "failure_lines": failure_lines,
        "skipped_lines": skip_lines,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _max_rss(who: int) -> int:
    usage = resource.getrusage(who).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    return usage if sys.platform == "darwin" else usage * 1024


def peak_memory_bytes() -> tuple[int, int]:
    """Peak resident memory of the coordinator, and the children figure.

    The coordinator value is the meaningful single-process peak: the
    performance and storage stages run in-process, so it reflects what one
    engine instance costs. The children value comes from `RUSAGE_CHILDREN`,
    whose interpretation is platform dependent (macOS appears to aggregate
    across reaped children rather than reporting the largest one), so it is
    kept as a separate diagnostic and never used as *the* peak.
    """
    return _max_rss(resource.RUSAGE_SELF), _max_rss(resource.RUSAGE_CHILDREN)


def build_sample_states(count: int) -> list:
    states = []
    for index in range(count):
        target = (7, 25, 60, 110, 180)[index % 5]
        state, _ = play_random_game(6_000_000 + index, max_plies=target)
        if not state.terminal and legal_actions(state):
            states.append(state)
    return states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-games", type=int, default=10_000)
    parser.add_argument("--antileak-trials", type=int, default=100_000)
    parser.add_argument("--antileak-trials-per-position", type=int, default=50)
    parser.add_argument("--invariant-games", type=int, default=2_000)
    parser.add_argument("--snapshot-seeds", type=int, default=120)
    parser.add_argument("--legal-seeds", type=int, default=250)
    parser.add_argument("--perf-repeats", type=int, default=4_000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument(
        "--output", type=Path, default=REPOSITORY_ROOT / "reports" / "phase_2_metrics.json"
    )
    options = parser.parse_args()

    if options.quick:
        options.replay_games = 60
        options.antileak_trials = 2_000
        options.invariant_games = 20
        options.snapshot_seeds = 6
        options.legal_seeds = 10
        options.perf_repeats = 400

    process = psutil.Process()
    memory_samples = [process.memory_info().rss]

    def note(message: str) -> None:
        memory_samples.append(process.memory_info().rss)
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    note(f"workers={options.workers} python={platform.python_version()}")

    note("stage 1/8 combat matrix")
    combat = combat_stage()

    note(f"stage 2/8 legal-action consistency ({options.legal_seeds} game walks)")
    legal = legal_stage(options.legal_seeds, options.workers)

    note(f"stage 3/8 replay reconstruction ({options.replay_games} games)")
    replay = replay_stage(options.replay_games, options.workers)

    note(f"stage 4/8 anti-leak permutations ({options.antileak_trials} trials)")
    antileak = antileak_stage(
        options.antileak_trials, options.antileak_trials_per_position, options.workers
    )

    note(f"stage 5/8 invariant stress ({options.invariant_games} games)")
    invariant = invariant_stage(options.invariant_games, options.workers)

    note(f"stage 6/8 snapshot / restore ({options.snapshot_seeds} seeds)")
    snapshot = snapshot_stage(options.snapshot_seeds, options.workers)

    note("stage 7/8 performance and storage baselines")
    sample_states = build_sample_states(24)
    performance = performance_stage(sample_states, options.perf_repeats)
    storage = storage_stage(sample_states)

    tests = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "expected_failures": 0}
    if not options.skip_pytest:
        note("stage 8/8 automated test suite")
        tests = pytest_stage()
    else:
        note("stage 8/8 skipped")

    lengths = replay["lengths"]
    outcomes = replay["outcome_counts"]
    components = performance["component_seconds"]
    component_total = sum(components.values())

    metrics = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "rules_version": RULES_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "numpy_version": np.__version__,
        "cpu_count": os.cpu_count(),
        "harness_workers": options.workers,
        "tests_total": tests["total"],
        "tests_passed": tests["passed"],
        "tests_failed": tests["failed"],
        "tests_skipped": tests["skipped"],
        "tests_expected_failures": tests["expected_failures"],
        "tests_seconds": tests.get("seconds", 0.0),
        "tests_failure_lines": tests.get("failure_lines", []),
        "tests_skipped_lines": tests.get("skipped_lines", []),
        "combat_cases_tested": combat["cases_tested"],
        "combat_cases_failed": combat["cases_failed"],
        "combat_failures": combat["failures"],
        "legal_action_positions_tested": legal["positions"],
        "legal_action_mismatches": legal["discrepancies"],
        "legal_action_largest_count": legal["largest_action_count"],
        "legal_action_empty_positions": legal["empty_positions"],
        "anti_leak_trials_attempted": antileak["attempted"],
        "anti_leak_trials_valid": antileak["valid"],
        "anti_leak_trials_changed": antileak["changed"],
        "anti_leak_positions": antileak["positions"],
        "anti_leak_observation_mismatches": antileak["observation_mismatches"],
        "anti_leak_action_mismatches": antileak["action_mismatches"],
        "anti_leak_event_mismatches": antileak["event_mismatches"],
        "anti_leak_public_view_mismatches": antileak["public_view_mismatches"],
        "anti_leak_belief_control_checks": antileak["belief_control_checks"],
        "anti_leak_belief_control_failures": antileak["belief_control_failures"],
        "replay_games": replay["games"],
        "replay_plies": replay["plies"],
        "replay_state_mismatches": replay["state_mismatches"],
        "replay_observation_mismatches": replay["observation_mismatches"],
        "replay_event_mismatches": replay["event_mismatches"],
        "replay_result_mismatches": replay["result_mismatches"],
        "snapshot_tests": snapshot["snapshots"],
        "snapshot_tests_by_phase": snapshot["by_phase"],
        "snapshot_mismatches": snapshot["total_mismatches"],
        "snapshot_legal_action_mismatches": snapshot["legal_action_mismatches"],
        "snapshot_observation_mismatches": snapshot["observation_mismatches"],
        "snapshot_public_event_mismatches": snapshot["public_event_mismatches"],
        "snapshot_next_transition_mismatches": snapshot["next_transition_mismatches"],
        "invariant_games": invariant["games"],
        "invariant_transitions": invariant["transitions"],
        "invariant_violations": invariant["violations"],
        "invariant_violation_details": invariant["violation_details"],
        "random_games": len(lengths),
        "random_game_mean_moves": statistics.mean(lengths) if lengths else 0,
        "random_game_median_moves": statistics.median(lengths) if lengths else 0,
        "random_game_min_moves": min(lengths) if lengths else 0,
        "random_game_max_moves": max(lengths) if lengths else 0,
        "random_red_wins": outcomes.get("red_win", 0),
        "random_blue_wins": outcomes.get("blue_win", 0),
        "random_draws": outcomes.get("draw", 0),
        "terminal_reason_counts": replay["terminal_reason_counts"],
        "legal_actions_per_second": performance["legal_actions_per_second"],
        "legal_action_masks_per_second": performance["legal_action_masks_per_second"],
        "state_transitions_per_second": performance["state_transitions_per_second"],
        "observations_per_second": performance["observations_per_second"],
        "snapshots_per_second": performance["snapshots_per_second"],
        "random_games_per_second": performance["random_games_per_second"],
        "random_game_plies_per_second": performance["random_game_plies_per_second"],
        "engine_time_fractions": {
            name: value / component_total for name, value in components.items()
        },
        "mean_memory_bytes": int(statistics.mean(memory_samples)),
        "peak_memory_bytes": peak_memory_bytes()[0],
        "children_rusage_maxrss_bytes": peak_memory_bytes()[1],
        "snapshot_serialized_bytes": storage["snapshot_serialized_bytes"],
        "mean_replay_serialized_bytes": storage["mean_replay_serialized_bytes"],
        "median_replay_serialized_bytes": storage["median_replay_serialized_bytes"],
        "thousand_replays_bytes": storage["thousand_replays_bytes"],
        "estimated_million_game_storage_bytes": storage[
            "estimated_million_game_storage_bytes"
        ],
        "stage_seconds": {
            "legal_actions": legal["seconds"],
            "replay": replay["seconds"],
            "anti_leak": antileak["seconds"],
            "invariants": invariant["seconds"],
            "snapshots": snapshot["seconds"],
            "tests": tests.get("seconds", 0.0),
        },
    }

    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    note(f"wrote {options.output}")

    unexplained = (
        metrics["combat_cases_failed"]
        + metrics["legal_action_mismatches"]
        + metrics["anti_leak_observation_mismatches"]
        + metrics["anti_leak_action_mismatches"]
        + metrics["anti_leak_event_mismatches"]
        + metrics["anti_leak_public_view_mismatches"]
        + metrics["replay_state_mismatches"]
        + metrics["replay_observation_mismatches"]
        + metrics["replay_event_mismatches"]
        + metrics["replay_result_mismatches"]
        + metrics["snapshot_mismatches"]
        + metrics["invariant_violations"]
        + metrics["tests_failed"]
    )
    print(f"\nunexplained mismatches across every gate: {unexplained}")
    return 0 if unexplained == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
