#!/usr/bin/env python3
"""Phase 4 Agent 2 acceptance harness.

Runs the baseline-suite gates and writes
`reports/phase_4_data/agent_02_baseline_agents.json`:

- the policy catalogue, with identities, versions and declared requirements;
- a legality sweep: every policy decides in thousands of seeded positions
  spanning every game phase, and every selection must be in the legal list;
- determinism trials: repeated decisions, fresh instances, and seed sensitivity;
- a local hidden-identity permutation sweep with a positive control -- the
  intermediate step between the per-commit test suite and Agent 4's audit;
- behavioural profiling: attack rate, Scout and Miner usage, game length, draw
  and Flag-capture rates, reveal rate and movement entropy, per policy;
- an informational ladder screen over paired colour-swapped units.

Two things this script deliberately is not.

It is **not a match runner**. The game loop below has no parallelism, no result
schema and no error recovery; Agent 3 owns those. It exists so behavioural
claims can be measured over whole games rather than single positions.

Its ladder screen is **not calibration**. Agent 4 owns the strength tiers, the
sample sizes that make them significant, and any weight tuning. The screen here
is a smoke test whose only job is to catch a policy that is misimplemented
rather than merely weak, and its numbers are recorded as informational.

Usage:

    python scripts/run_phase4_agent02.py                 # full acceptance run
    python scripts/run_phase4_agent02.py --quick         # fast smoke run
    python scripts/run_phase4_agent02.py --skip-pytest   # measurements only
"""

from __future__ import annotations

import argparse
import json
import math
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

from stratego.engine.actions import decode_action  # noqa: E402
from stratego.engine.constants import (  # noqa: E402
    DRAW_TERMINAL_REASONS,
    EVALUATION_RULES,
    IMPLEMENTATION_VERSION,
    MINER,
    OBSERVATION_VERSION,
    PIECE_TYPE_NAMES,
    PIECES_PER_PLAYER,
    RULES_VERSION,
    SCOUT,
    TERMINAL_BATTLELESS_MOVE_LIMIT_DRAW,
    TERMINAL_FLAG_CAPTURE,
)
from stratego.engine.coordinates import square_column, square_row  # noqa: E402
from stratego.engine.legal_moves import legal_actions  # noqa: E402
from stratego.engine.permutation import (  # noqa: E402
    belief_targets_differ,
    hidden_opponent_piece_ids,
    permute_hidden_identities,
)
from stratego.engine.random_play import make_random_setups, select_random_action  # noqa: E402
from stratego.engine.snapshot import clone_state  # noqa: E402
from stratego.engine.state import GameState, create_game  # noqa: E402
from stratego.engine.transition import apply_action  # noqa: E402
from stratego.evaluation.baselines import BASELINE_SUITE_VERSION, ScoringPolicy  # noqa: E402
from stratego.evaluation.heuristics import HEURISTICS_VERSION, build_context, rank_moves  # noqa: E402
from stratego.evaluation.match_spec import (  # noqa: E402
    EVALUATION_SUITE_VERSION,
    MatchSpec,
    build_paired_schedule,
)
from stratego.evaluation.policy import (  # noqa: E402
    POLICY_INTERFACE_VERSION,
    Policy,
    build_policy_input,
)
from stratego.evaluation.registry import (  # noqa: E402
    ALL_POLICY_IDS,
    LADDER_POLICY_IDS,
    STRESS_POLICY_IDS,
    build_policies,
    build_policy,
    policy_catalog,
    policy_ref,
)
from stratego.evaluation.setup_bank import SetupBank  # noqa: E402

DEFAULT_OUTPUT = REPOSITORY_ROOT / "reports/phase_4_data/agent_02_baseline_agents.json"
DEFAULT_BEHAVIOR_CSV = REPOSITORY_ROOT / "reports/phase_4_data/agent_02_behavior_profile.csv"

#: Plies at which a generated game is snapshotted. Chosen to cover the opening,
#: both middle-game phases and the late game, because how much is hidden -- and
#: therefore what a leak could look like -- changes completely across a game.
SNAPSHOT_PLIES = (6, 15, 30, 55, 85, 125, 175)

#: Behavioural metrics are not opponent-independent -- Chaos, for instance,
#: attacks twice as often against a policy that walks into it as against one that
#: does not. Profiling against two references at opposite ends of the ladder
#: makes that dependence visible instead of hiding it behind a single number,
#: and means every policy shares at least one reference with every other. A
#: policy cannot be scheduled against itself, so a reference has one row rather
#: than two; the data file records which references each row used.
BEHAVIOR_REFERENCES = ("random_legal", "strategic_rule_based")


# ---------------------------------------------------------------------------
# Position generation
# ---------------------------------------------------------------------------


def generate_positions(count: int, plies=SNAPSHOT_PLIES) -> list[GameState]:
    """Seeded nonterminal positions spanning every phase of a game.

    One random game yields a snapshot at each checkpoint ply, which is far
    cheaper than replaying a game from scratch per position and gives the same
    coverage. Every position is a `clone_state`, so a consumer that mutates one
    cannot disturb the generator.
    """
    positions: list[GameState] = []
    seed = 0
    while len(positions) < count:
        red_setup, blue_setup = make_random_setups(seed)
        state = create_game(
            red_setup, blue_setup, rules=EVALUATION_RULES, game_id=f"probe-{seed}"
        )
        rng = random.Random(seed + 7_654_321)
        targets = set(plies)
        limit = max(plies)
        while not state.terminal and state.total_moves < limit:
            actions = legal_actions(state)
            apply_action(state, select_random_action(state, rng, actions), legal=actions)
            if state.total_moves in targets and not state.terminal:
                positions.append(clone_state(state))
                if len(positions) >= count:
                    break
        seed += 1
    return positions[:count]


def make_request(state: GameState, policy: Policy, seed: int = 90210):
    return build_policy_input(
        state,
        policy=policy.ref,
        policy_seed=seed,
        requirements=policy.requirements,
        suite_version=EVALUATION_SUITE_VERSION,
        match_id="m-agent02",
        paired_unit_id="u-agent02",
    )


# ---------------------------------------------------------------------------
# Stage 1 -- the catalogue
# ---------------------------------------------------------------------------


def catalogue_stage() -> dict:
    catalog = policy_catalog()
    problems: list[str] = []

    if len(LADDER_POLICY_IDS) != 4:
        problems.append(f"expected a four-tier ladder, found {len(LADDER_POLICY_IDS)}")
    if len(STRESS_POLICY_IDS) < 4:
        problems.append(f"expected >= 4 stress policies, found {len(STRESS_POLICY_IDS)}")
    if len(set(ALL_POLICY_IDS)) != len(ALL_POLICY_IDS):
        problems.append("the catalogue contains a duplicate policy identifier")
    for policy_id in ALL_POLICY_IDS:
        if policy_id.startswith("contract_"):
            problems.append(f"{policy_id} uses the reserved contract-fixture prefix")
    for policy in build_policies():
        if policy.requirements.observation or policy.requirements.legal_action_mask:
            problems.append(f"{policy.policy_id} requests a product no baseline needs")

    return {
        "policies": catalog,
        "ladder_policy_ids": list(LADDER_POLICY_IDS),
        "stress_policy_ids": list(STRESS_POLICY_IDS),
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Stage 2 -- legality
# ---------------------------------------------------------------------------


def legality_stage(positions: list[GameState]) -> dict:
    """Every policy must select an action from the engine's own legal list."""
    per_policy: dict[str, int] = {}
    illegal = 0
    failures: list[str] = []
    empty_legal_sets = 0

    for policy in build_policies():
        tested = 0
        for state in positions:
            request = make_request(state, policy)
            if not request.legal_actions:  # pragma: no cover - defended by the contract
                empty_legal_sets += 1
                continue
            result = policy.decide(request)
            if result.selected_action_id not in request.legal_actions:
                illegal += 1
                if len(failures) < 20:
                    failures.append(
                        f"{policy.policy_id} chose {result.selected_action_id} at ply "
                        f"{state.total_moves} of {state.game_id}"
                    )
            tested += 1
        per_policy[policy.policy_id] = tested

    return {
        "positions_tested_per_policy": per_policy,
        "positions_in_pool": len(positions),
        "illegal_actions": illegal,
        "illegal_action_detail": failures,
        "empty_legal_sets": empty_legal_sets,
        "plies_covered": sorted({state.total_moves for state in positions}),
    }


# ---------------------------------------------------------------------------
# Stage 3 -- determinism
# ---------------------------------------------------------------------------


def determinism_stage(positions: list[GameState], repeats: int = 3) -> dict:
    """Reproducibility from `(public input, policy seed, ply)` alone."""
    trials = 0
    failures: list[str] = []
    seed_sensitive: dict[str, int] = {}
    near_ties: dict[str, int] = {}

    for policy in build_policies():
        varied = 0
        ties = 0
        for state in positions:
            request = make_request(state, policy)
            first = policy.decide_checked(request)

            # 1. The same input and seed, repeated.
            for _ in range(repeats):
                trials += 1
                again = policy.decide_checked(make_request(state, policy))
                if again.selected_action_id != first.selected_action_id:
                    failures.append(f"{policy.policy_id} varied across identical requests")
                elif again.diagnostics != first.diagnostics:
                    failures.append(f"{policy.policy_id} diagnostics varied across requests")

            # 2. A fresh instance must agree; no policy may carry state.
            trials += 1
            fresh = build_policy(policy.policy_id)
            if (
                fresh.decide_checked(make_request(state, fresh)).selected_action_id
                != first.selected_action_id
            ):
                failures.append(f"{policy.policy_id} depends on instance state")

            # 3. Seed sensitivity, measured rather than assumed.
            trials += 1
            actions = {
                policy.decide_checked(make_request(state, policy, seed=seed)).selected_action_id
                for seed in (1, 2, 3, 4, 5, 6, 7, 8)
            }
            if len(actions) > 1:
                varied += 1

            if isinstance(policy, ScoringPolicy) and policy.selection_margin > 0.0:
                context = build_context(make_request(state, policy))
                ranked = rank_moves(policy.score(context, move) for move in context.moves)
                best = ranked[0].score
                pool = [
                    move for move in ranked if move.score >= best - policy.selection_margin
                ]
                if len(pool) > 1:
                    ties += 1
                    if len(actions) == 1:
                        failures.append(
                            f"{policy.policy_id} ignored its seed at a genuine near-tie"
                        )

        seed_sensitive[policy.policy_id] = varied
        near_ties[policy.policy_id] = ties

    stochastic_without_variation = [
        policy.policy_id
        for policy in build_policies()
        if policy.stochastic and seed_sensitive[policy.policy_id] == 0
    ]
    for policy_id in stochastic_without_variation:
        failures.append(f"{policy_id} declares itself stochastic but never used its seed")

    return {
        "trials": trials,
        "failures": len(failures),
        "failure_detail": failures[:20],
        "positions_per_policy": len(positions),
        "repeats_per_position": repeats,
        "positions_where_the_seed_changed_the_action": seed_sensitive,
        "positions_with_a_near_tie": near_ties,
    }


# ---------------------------------------------------------------------------
# Stage 4 -- hidden-identity permutation
# ---------------------------------------------------------------------------


def permutation_stage(positions: list[GameState], target_trials: int) -> dict:
    """The local sweep between the per-commit suite and Agent 4's full audit.

    A trial counts only if the permutation actually reassigned a type *and* the
    positive control fires, so an inert fixture cannot inflate the number.
    """
    policies = build_policies()
    rng = random.Random(20260402)

    trials = 0
    comparisons = 0
    mismatches: list[str] = []
    positive_control_failures = 0
    skipped_unchanged = 0
    leak_detector_failures = 0
    plies_covered: Counter = Counter()

    index = 0
    while trials < target_trials and index < len(positions):
        state = positions[index]
        index += 1
        observer = state.acting_player

        clone, info = permute_hidden_identities(state, observer, rng)
        if not info["valid"] or not info["changed"]:
            skipped_unchanged += 1
            continue

        if not belief_targets_differ(state, clone, observer):
            positive_control_failures += 1

        # Leak detector: confirm these two states really do differ in exactly the
        # way a leaking policy would notice.
        hidden = hidden_opponent_piece_ids(state, observer)
        if [state.pieces[piece_id].true_type for piece_id in hidden] == [
            clone.pieces[piece_id].true_type for piece_id in hidden
        ]:
            leak_detector_failures += 1

        trials += 1
        plies_covered[state.total_moves] += 1

        for policy in policies:
            left = policy.decide_checked(make_request(state, policy))
            right = policy.decide_checked(make_request(clone, policy))
            comparisons += 1
            if left.selected_action_id != right.selected_action_id:
                mismatches.append(
                    f"{policy.policy_id} chose differently at ply {state.total_moves}"
                )
            if left.diagnostics != right.diagnostics:
                mismatches.append(
                    f"{policy.policy_id} diagnostics differ at ply {state.total_moves}"
                )

            if isinstance(policy, ScoringPolicy):
                # Stronger than the argmax: the whole score vector must match.
                left_context = build_context(make_request(state, policy))
                right_context = build_context(make_request(clone, policy))
                if rank_moves(
                    policy.score(left_context, move) for move in left_context.moves
                ) != rank_moves(
                    policy.score(right_context, move) for move in right_context.moves
                ):
                    mismatches.append(
                        f"{policy.policy_id} score vector differs at ply {state.total_moves}"
                    )

    return {
        "trials": trials,
        "policy_comparisons": comparisons,
        "mismatches": len(mismatches),
        "mismatch_detail": mismatches[:20],
        "positive_control_failures": positive_control_failures,
        "leak_detector_failures": leak_detector_failures,
        "positions_skipped_unchanged": skipped_unchanged,
        "plies_covered": dict(sorted(plies_covered.items())),
        "policies": list(ALL_POLICY_IDS),
        "note": (
            "Local sweep. Agent 4 owns the >= 100,000 trial audit across the whole "
            "suite."
        ),
    }


# ---------------------------------------------------------------------------
# Game loop and behavioural profiling
# ---------------------------------------------------------------------------


def play_match(spec: MatchSpec, policies: dict, bank: SetupBank, profile: Counter | None = None):
    """Play one match through the observer-safe contract.

    Not a runner: no parallelism, no result schema, no recovery. Agent 3 owns
    those. `profile` accumulates behavioural counters for the candidate side.
    """
    red_setup, blue_setup = spec.resolve_setups(bank)
    state = create_game(red_setup, blue_setup, rules=spec.rules, game_id=spec.game_id)
    candidate_ref = spec.candidate

    while not state.terminal:
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

        if profile is not None and ref == candidate_ref:
            source, destination = decode_action(result.selected_action_id)
            mover = state.piece_at(source)
            target = state.piece_at(destination)
            # A legal move runs along one rank or one file, so the two deltas are
            # never both nonzero and the sum is the true step count.
            distance = abs(square_row(destination) - square_row(source)) + abs(
                square_column(destination) - square_column(source)
            )
            profile["moves"] += 1
            profile[f"piece_{PIECE_TYPE_NAMES[mover.true_type]}"] += 1
            profile["distance"] += distance
            if target is not None:
                profile["attacks"] += 1
                if mover.true_type == MINER:
                    profile["miner_attacks"] += 1
            if mover.true_type == SCOUT and distance > 1:
                profile["scout_runs"] += 1

        # The engine stays the final legality authority.
        apply_action(state, result.selected_action_id, legal=legal)

    if profile is not None:
        profile["games"] += 1
        profile["plies"] += state.total_moves
        profile[f"terminal_{state.terminal_reason}"] += 1
        if state.terminal_reason in DRAW_TERMINAL_REASONS:
            profile["draws"] += 1
        elif state.winner == spec.candidate_color:
            profile["wins"] += 1
        else:
            profile["losses"] += 1
        if state.terminal_reason == TERMINAL_FLAG_CAPTURE and state.winner == (
            spec.candidate_color
        ):
            profile["flag_captures"] += 1
        # Reveals: own pieces whose type the opponent legally learned.
        revealed = sum(
            1
            for record in state.pieces_of(spec.candidate_color)
            if record.known_to(spec.opponent_color)
        )
        profile["own_pieces_revealed"] += revealed
    return state


def entropy(counts: list[int]) -> float:
    """Shannon entropy in bits of a count distribution."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts:
        if count <= 0:
            continue
        share = count / total
        value -= share * math.log2(share)
    return value


def summarise_profile(profile: Counter) -> dict:
    """Turn raw behavioural counters into the rates Agent 4 characterises with."""
    moves = max(profile["moves"], 1)
    games = max(profile["games"], 1)
    piece_counts = [profile[f"piece_{name}"] for name in PIECE_TYPE_NAMES]
    return {
        "games": profile["games"],
        "moves": profile["moves"],
        "mean_game_plies": profile["plies"] / games,
        "attack_rate": profile["attacks"] / moves,
        "scout_move_rate": profile["piece_scout"] / moves,
        "scout_run_rate": profile["scout_runs"] / moves,
        "miner_move_rate": profile["piece_miner"] / moves,
        "miner_attack_rate": profile["miner_attacks"] / moves,
        "mean_move_distance": profile["distance"] / moves,
        "piece_type_entropy_bits": entropy(piece_counts),
        "own_reveal_rate": profile["own_pieces_revealed"] / (games * PIECES_PER_PLAYER),
        "draw_rate": profile["draws"] / games,
        "battleless_draw_rate": profile[f"terminal_{TERMINAL_BATTLELESS_MOVE_LIMIT_DRAW}"]
        / games,
        "flag_capture_win_rate": profile["flag_captures"] / games,
        "effective_win_rate": (profile["wins"] + 0.5 * profile["draws"]) / games,
        "wins": profile["wins"],
        "draws": profile["draws"],
        "losses": profile["losses"],
        "terminal_reasons": {
            key.removeprefix("terminal_"): value
            for key, value in sorted(profile.items())
            if key.startswith("terminal_")
        },
    }


def profile_against(candidate_id: str, reference_id: str, bank: SetupBank, units: int) -> Counter:
    candidate = build_policy(candidate_id)
    opponent = build_policy(reference_id)
    policies = {candidate.ref.token: candidate, opponent.ref.token: opponent}
    profile: Counter = Counter()
    for unit in build_paired_schedule(
        candidate.ref, opponent.ref, range(units), setup_bank_version=bank.bank_version
    ):
        for spec in unit.matches:
            play_match(spec, policies, bank, profile)
    return profile


def behavior_stage(bank: SetupBank, units: int) -> dict:
    """Per-policy behavioural fingerprints over whole games.

    These are the metrics Agent 4 uses to show that the stress policies produce
    materially different games. Each policy is profiled against every reference
    it can legally be scheduled against, and both the per-reference rows and the
    pooled row are reported, so a metric that depends on the opponent is visible
    as a spread rather than averaged into a single misleading number.

    Agent 4 recomputes all of this over the full calibration league; these
    numbers exist so a stress policy that silently stopped being unusual is
    caught here rather than three agents later.
    """
    rows: dict[str, dict] = {}
    for policy_id in ALL_POLICY_IDS:
        references = [
            reference for reference in BEHAVIOR_REFERENCES if reference != policy_id
        ]
        pooled: Counter = Counter()
        by_reference: dict[str, dict] = {}
        for reference_id in references:
            profile = profile_against(policy_id, reference_id, bank, units)
            by_reference[reference_id] = summarise_profile(profile)
            pooled.update(profile)

        rows[policy_id] = {
            "role": "ladder" if policy_id in LADDER_POLICY_IDS else "stress",
            "references": references,
            "combined": summarise_profile(pooled),
            "by_reference": by_reference,
        }
    return rows


# ---------------------------------------------------------------------------
# Stage 6 -- the informational ladder screen
# ---------------------------------------------------------------------------


def screen_stage(bank: SetupBank, units: int) -> dict:
    """A smoke screen over the ladder. Explicitly not calibration.

    Its only purpose is to catch a policy that is broken rather than merely
    weak. Agent 4 owns the strength tiers, the sample sizes that make them
    significant, and any weight revision.
    """
    results: list[dict] = []
    ladder = list(LADDER_POLICY_IDS)
    for index, candidate_id in enumerate(ladder):
        for opponent_id in ladder[index + 1 :]:
            candidate = build_policy(candidate_id)
            opponent = build_policy(opponent_id)
            policies = {candidate.ref.token: candidate, opponent.ref.token: opponent}

            unit_scores: list[float] = []
            wins = draws = losses = 0
            plies = 0
            for unit in build_paired_schedule(
                candidate.ref, opponent.ref, range(units), setup_bank_version=bank.bank_version
            ):
                unit_total = 0.0
                for spec in unit.matches:
                    state = play_match(spec, policies, bank)
                    score = state.effective_score_for(spec.candidate_color)
                    unit_total += score
                    plies += state.total_moves
                    if score == 1.0:
                        wins += 1
                    elif score == 0.5:
                        draws += 1
                    else:
                        losses += 1
                # The paired unit is the independent observation, not the game.
                unit_scores.append(unit_total / 2.0)

            mean = sum(unit_scores) / len(unit_scores)
            variance = (
                sum((score - mean) ** 2 for score in unit_scores) / (len(unit_scores) - 1)
                if len(unit_scores) > 1
                else 0.0
            )
            standard_error = math.sqrt(variance / len(unit_scores))
            results.append(
                {
                    "candidate": candidate_id,
                    "opponent": opponent_id,
                    "paired_units": len(unit_scores),
                    "games": wins + draws + losses,
                    "candidate_effective_win_rate": mean,
                    "paired_unit_standard_error": standard_error,
                    "approximate_95_interval": [
                        mean - 1.96 * standard_error,
                        mean + 1.96 * standard_error,
                    ],
                    "separated_from_even": abs(mean - 0.5) > 1.96 * standard_error,
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "mean_plies": plies / max(wins + draws + losses, 1),
                }
            )

    ordered = [row for row in results if row["candidate_effective_win_rate"] < 0.5]
    return {
        "note": (
            "Informational only. Agent 4 owns baseline calibration, the strength-tier "
            "gate and any weight revision. Intervals are normal approximations over "
            "paired units and are not the interval method Agent 3 will implement."
        ),
        "resampling_unit": "paired_unit",
        "units_per_matchup": units,
        "matchups": results,
        "matchups_where_the_lower_tier_scored_below_even": len(ordered),
    }


# ---------------------------------------------------------------------------
# Stage 7 -- the automated suite
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
    "stratego/evaluation/heuristics.py",
    "stratego/evaluation/baselines.py",
    "stratego/evaluation/stress.py",
    "stratego/evaluation/registry.py",
    "tests/evaluation/test_baselines.py",
    "tests/evaluation/test_baseline_information_safety.py",
    "scripts/run_phase4_agent02.py",
    "reports/phase_4_data/agent_02_baseline_agents.json",
    "reports/phase_4_data/agent_02_behavior_profile.csv",
]

FILES_MODIFIED = [
    "stratego/evaluation/__init__.py",
    "reports/phase_4_implementation_report.md",
]

BEHAVIOR_CSV_COLUMNS = (
    "policy_id",
    "role",
    "reference_opponent",
    "games",
    "moves",
    "mean_game_plies",
    "attack_rate",
    "scout_move_rate",
    "scout_run_rate",
    "miner_move_rate",
    "miner_attack_rate",
    "mean_move_distance",
    "piece_type_entropy_bits",
    "own_reveal_rate",
    "draw_rate",
    "battleless_draw_rate",
    "flag_capture_win_rate",
    "effective_win_rate",
)


def write_behavior_csv(path: Path, rows: dict) -> None:
    """One row per (policy, reference), plus a pooled `all` row per policy."""
    lines = [",".join(BEHAVIOR_CSV_COLUMNS)]
    for policy_id, row in rows.items():
        entries = list(row["by_reference"].items()) + [("all", row["combined"])]
        for reference_id, metrics in entries:
            values = [policy_id, row["role"], reference_id]
            for column in BEHAVIOR_CSV_COLUMNS[3:]:
                value = metrics[column]
                values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
            lines.append(",".join(values))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=5000)
    parser.add_argument("--permutation-trials", type=int, default=1500)
    parser.add_argument("--behavior-units", type=int, default=64)
    parser.add_argument("--screen-units", type=int, default=192)
    parser.add_argument("--bank-size", type=int, default=256)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--behavior-csv", type=Path, default=DEFAULT_BEHAVIOR_CSV)
    options = parser.parse_args()

    if options.quick:
        options.positions = 200
        options.permutation_trials = 40
        options.behavior_units = 4
        options.screen_units = 8
        options.bank_size = 16

    started = time.perf_counter()

    print("[1/7] policy catalogue")
    catalogue = catalogue_stage()

    print(f"[2/7] generating {options.positions} probe positions")
    positions = generate_positions(options.positions)

    print(f"[3/7] legality sweep over {len(positions)} positions x {len(ALL_POLICY_IDS)} policies")
    legality = legality_stage(positions)

    determinism_positions = positions[: min(400, len(positions))]
    print(f"[4/7] determinism trials over {len(determinism_positions)} positions")
    determinism = determinism_stage(determinism_positions)

    print(f"[5/7] {options.permutation_trials} hidden-identity permutation trials")
    permutation = permutation_stage(positions, options.permutation_trials)

    bank = SetupBank.generate(options.bank_size)

    print(f"[6/7] behavioural profiling over {options.behavior_units} paired units per policy")
    behavior = behavior_stage(bank, options.behavior_units)
    write_behavior_csv(options.behavior_csv, behavior)

    print(f"[7/7] informational ladder screen over {options.screen_units} paired units")
    screen = screen_stage(bank, options.screen_units)

    if options.skip_pytest:
        print("      skipping the test suite")
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
        print("      running the repository test suite")
        tests = pytest_stage()

    status_checks = {
        "catalogue_complete": not catalogue["problems"],
        "ladder_has_four_tiers": len(LADDER_POLICY_IDS) == 4,
        "at_least_four_stress_policies": len(STRESS_POLICY_IDS) >= 4,
        "no_illegal_actions": legality["illegal_actions"] == 0,
        "no_empty_legal_sets": legality["empty_legal_sets"] == 0,
        "determinism_clean": determinism["failures"] == 0,
        "no_hidden_information_mismatch": permutation["mismatches"] == 0,
        "positive_control_fired": permutation["positive_control_failures"] == 0,
        "leak_detector_fired": permutation["leak_detector_failures"] == 0,
        "permutation_trials_met_target": permutation["trials"] >= options.permutation_trials,
        "tests_green": options.skip_pytest or (tests["failed"] == 0 and tests["errors"] == 0),
    }
    status = "PASS" if all(status_checks.values()) else "FAIL"

    payload = {
        "agent": "phase_4_agent_02_baseline_opponents",
        "status": status,
        "status_checks": status_checks,
        "frozen_contracts": {
            "implementation_version": IMPLEMENTATION_VERSION,
            "rules_version": RULES_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "phase_3_backend_decision": "KEEP_PYTHON",
        },
        "policy_interface_version": POLICY_INTERFACE_VERSION,
        "evaluation_suite_version": EVALUATION_SUITE_VERSION,
        "baseline_suite_version": BASELINE_SUITE_VERSION,
        "heuristics_version": HEURISTICS_VERSION,
        "policy_ids": list(ALL_POLICY_IDS),
        "policy_versions": {
            policy_id: policy_ref(policy_id).policy_version for policy_id in ALL_POLICY_IDS
        },
        "ladder_policy_ids": list(LADDER_POLICY_IDS),
        "stress_policy_ids": list(STRESS_POLICY_IDS),
        "positions_tested_per_policy": legality["positions_tested_per_policy"],
        "illegal_actions": legality["illegal_actions"],
        "determinism_trials": determinism["trials"],
        "determinism_failures": determinism["failures"],
        "local_hidden_permutation_trials": permutation["trials"],
        "local_hidden_permutation_failures": permutation["mismatches"],
        "local_hidden_permutation_comparisons": permutation["policy_comparisons"],
        "behavior_summary": behavior,
        "test_total": tests["total"],
        "test_passed": tests["passed"],
        "test_failed": tests["failed"] + tests["errors"],
        "catalogue": catalogue,
        "legality": legality,
        "determinism": determinism,
        "hidden_permutation": permutation,
        "ladder_screen": screen,
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
    print(f"status                          {status}")
    print(f"policies                        {len(ALL_POLICY_IDS)} "
          f"({len(LADDER_POLICY_IDS)} ladder, {len(STRESS_POLICY_IDS)} stress)")
    print(f"positions per policy            {min(legality['positions_tested_per_policy'].values())}")
    print(f"illegal actions                 {payload['illegal_actions']}")
    print(f"determinism trials / failures   {determinism['trials']} / {determinism['failures']}")
    print(f"permutation trials              {permutation['trials']}")
    print(f"permutation comparisons         {permutation['policy_comparisons']}")
    print(f"permutation mismatches          {permutation['mismatches']}")
    print(f"tests passed / failed           {payload['test_passed']} / {payload['test_failed']}")
    print(f"written                         {options.output.relative_to(REPOSITORY_ROOT)}")
    print(f"                                {options.behavior_csv.relative_to(REPOSITORY_ROOT)}")
    for name, ok in sorted(status_checks.items()):
        if not ok:
            print(f"failed check                    {name}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
