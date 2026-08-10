#!/usr/bin/env python3
"""Phase 4 Agent 3 acceptance harness.

Runs the match-runner and statistics gates and writes
`reports/phase_4_data/agent_03_match_runner_statistics.json`:

- prerequisite check: Agents 1 and 2 must both report `PASS`, and Agent 1's
  frozen setup-bank digest must still regenerate;
- runner correctness: exact reproduction from identity, reproduction from a
  stored row without the bank, engine replay of every stored history, and the
  raw-result schema;
- policy-failure handling: an illegal action, a raising policy and two contract
  violations must each be raised loudly, classified, and never replaced by a
  substitute legal move;
- the parallel reproducibility sweep: one substantial schedule run at several
  worker counts, requiring identical match identities, setups, replay digests,
  results, terminal reasons and ply counts;
- statistical validation against synthetic tables with known answers, including
  the test that proves the resampling unit is the paired unit and not the game;
- acceptance statistics over the swept schedule at the full bootstrap resample
  count, with colour, setup, terminal and per-opponent breakdowns;
- the secondary Bradley-Terry league rating.

What this script deliberately is not
------------------------------------
It is **not baseline calibration**. Agent 4 owns the strength tiers, the sample
sizes that make them significant, and any weight revision. The four matchups
below were chosen to exercise the statistics -- a mismatch, a close pair, and a
draw-heavy stress matchup that produces long games -- not to measure the ladder.
Every effective win rate it prints is a by-product of the reproducibility sweep.

Usage:

    python scripts/run_phase4_agent03.py                 # full acceptance run
    python scripts/run_phase4_agent03.py --quick         # fast smoke run
    python scripts/run_phase4_agent03.py --skip-pytest   # measurements only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
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
from stratego.evaluation.match_spec import (  # noqa: E402
    EVALUATION_SUITE_VERSION,
    MatchSpec,
)
from stratego.engine.replay import rebuild_final_state  # noqa: E402
from stratego.evaluation.match_runner import (  # noqa: E402
    ERROR_CONTRACT_VIOLATION,
    ERROR_ILLEGAL_ACTION,
    ERROR_POLICY_EXCEPTION,
    MATCH_RESULT_SCHEMA_VERSION,
    MATCH_RUNNER_VERSION,
    ON_POLICY_ERROR_QUARANTINE,
    MatchResult,
    PolicyFailure,
    compare_results,
    play_match,
    replay_digest,
    replay_stored_match,
    reproduce_match,
    results_digest,
    run_schedule,
)
from stratego.evaluation.policy import (  # noqa: E402
    POLICY_INTERFACE_VERSION,
    Policy,
    PolicyRequirements,
    PolicyResult,
)
from stratego.evaluation.registry import build_policy, policy_ref  # noqa: E402
from stratego.evaluation.reporting import (  # noqa: E402
    REPORTING_VERSION,
    attach_replay_reference,
    read_replays_jsonl,
    render_league_table,
    render_matchup_table,
    render_run_report,
    render_worker_table,
    run_manifest,
    write_json,
    write_replays_jsonl,
    write_results_csv,
)
from stratego.evaluation.scheduler import (  # noqa: E402
    SCHEDULER_VERSION,
    EvaluationSchedule,
    build_matchup_schedule,
    merge_schedules,
    require_valid_schedule,
    schedule_fingerprint,
)
from stratego.evaluation.setup_bank import (  # noqa: E402
    DEFAULT_BANK_SIZE,
    SETUP_BANK_VERSION,
    SetupBank,
)
from stratego.evaluation.statistics import (  # noqa: E402
    BOOTSTRAP_METHOD,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    LEAGUE_METHOD,
    STATISTICS_VERSION,
    OutcomeCounts,
    bootstrap_interval,
    bradley_terry_ratings,
    build_paired_units,
    color_split,
    detect_result_problems,
    effective_win_rate,
    paired_bootstrap_interval,
    summarize_matchup,
    summarize_run,
    synthetic_results,
    unit_score_histogram,
)

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_4_data"
DEFAULT_OUTPUT = DATA_DIRECTORY / "agent_03_match_runner_statistics.json"
DEFAULT_RAW_CSV = DATA_DIRECTORY / "agent_03_reproducibility_raw.csv"
DEFAULT_RESULTS_CSV = DATA_DIRECTORY / "agent_03_acceptance_results.csv"
DEFAULT_REPLAYS = DATA_DIRECTORY / "agent_03_acceptance_replays.jsonl"
AGENT_01_DATA = DATA_DIRECTORY / "agent_01_evaluation_foundations.json"
AGENT_02_DATA = DATA_DIRECTORY / "agent_02_baseline_agents.json"
AGENT_01_BANK = DATA_DIRECTORY / "agent_01_setup_bank_v1.json"

#: The sweep's matchups. Chosen for statistical variety, not for calibration:
#: an extreme mismatch, two adjacent ladder tiers, the pair Agent 2 could not
#: separate, and a stress matchup whose games are long and often drawn.
SWEEP_MATCHUPS = (
    ("random_legal", "strategic_rule_based"),
    ("basic_heuristic", "tactical_rule_based"),
    ("tactical_rule_based", "strategic_rule_based"),
    ("strategic_rule_based", "stress_draw_seeker"),
)


# ---------------------------------------------------------------------------
# Stage 1 -- prerequisites
# ---------------------------------------------------------------------------


def prerequisite_stage() -> dict:
    """Agents 1 and 2 must both be `PASS`, and Agent 1's bank must regenerate."""
    problems: list[str] = []
    payload: dict = {"agent_01_status": None, "agent_02_status": None}

    for label, path in (("agent_01", AGENT_01_DATA), ("agent_02", AGENT_02_DATA)):
        if not path.exists():
            problems.append(f"{label} data file is missing: {path.name}")
            continue
        data = json.loads(path.read_text())
        payload[f"{label}_status"] = data.get("status")
        if data.get("status") != "PASS":
            problems.append(f"{label} reports {data.get('status')!r}, not PASS")

    # Agent 1 recommended this as a standing reproducibility check: the frozen
    # bank artefact must still be what generation produces.
    if AGENT_01_BANK.exists():
        stored = SetupBank.from_json(AGENT_01_BANK.read_text())
        regenerated = SetupBank.generate(
            size=len(stored), root_seed=stored.root_seed, bank_version=stored.bank_version
        )
        payload["setup_bank_pairs"] = len(stored)
        payload["setup_bank_digest_stored"] = stored.digest()
        payload["setup_bank_digest_regenerated"] = regenerated.digest()
        payload["setup_bank_digest_matches"] = stored.digest() == regenerated.digest()
        if not payload["setup_bank_digest_matches"]:
            problems.append("Agent 1's frozen setup-bank digest no longer regenerates")
    else:
        problems.append(f"Agent 1's setup bank artefact is missing: {AGENT_01_BANK.name}")

    payload["problems"] = problems
    return payload


# ---------------------------------------------------------------------------
# Broken policies, for the failure-handling stage
# ---------------------------------------------------------------------------


class IllegalActionPolicy(Policy):
    """Returns an action outside the legal list."""

    policy_id = "acceptance_broken_illegal"
    policy_version = "1.0.0"
    requirements = PolicyRequirements()

    def decide(self, request):
        return self.result(request, max(request.legal_actions) + 1)


class ExplodingPolicy(Policy):
    """Raises instead of deciding."""

    policy_id = "acceptance_broken_exploding"
    policy_version = "1.0.0"

    def decide(self, request):
        raise RuntimeError("deliberate acceptance failure")


class WrongSeedPolicy(Policy):
    """Returns a result whose decision seed does not match the request."""

    policy_id = "acceptance_broken_seed"
    policy_version = "1.0.0"

    def decide(self, request):
        return PolicyResult(
            selected_action_id=request.legal_actions[0],
            policy=request.policy,
            decision_seed=request.decision_seed ^ 1,
        )


class NonResultPolicy(Policy):
    """Returns a bare integer instead of a PolicyResult."""

    policy_id = "acceptance_broken_type"
    policy_version = "1.0.0"

    def decide(self, request):
        return request.legal_actions[0]


BROKEN_CASES = (
    (IllegalActionPolicy, ERROR_ILLEGAL_ACTION),
    (ExplodingPolicy, ERROR_POLICY_EXCEPTION),
    (WrongSeedPolicy, ERROR_CONTRACT_VIOLATION),
    (NonResultPolicy, ERROR_CONTRACT_VIOLATION),
)


# ---------------------------------------------------------------------------
# Stage 2 -- runner correctness
# ---------------------------------------------------------------------------

#: The minimum raw-result fields the Agent 3 instructions enumerate.
REQUIRED_RESULT_FIELDS = (
    "match_id",
    "paired_unit_id",
    "candidate_policy_id",
    "candidate_policy_version",
    "opponent_policy_id",
    "opponent_policy_version",
    "candidate_color",
    "setup_pair_id",
    "replicate",
    "root_seed",
    "candidate_seed",
    "opponent_seed",
    "winner",
    "draw",
    "candidate_result",
    "terminal_reason",
    "plies",
    "replay_digest",
    "wall_clock_seconds",
    "policy_error",
)


def runner_stage(bank: SetupBank, matches: int) -> dict:
    """Exact reproduction, row-only reproduction, and engine replay."""
    schedule = merge_schedules(
        [
            build_matchup_schedule(candidate, opponent, max(matches // (2 * 4), 1))
            for candidate, opponent in SWEEP_MATCHUPS
        ],
        name="runner_correctness",
    ).limited(matches)

    checks = {
        "reproduced_identically": 0,
        "reproduction_mismatches": 0,
        "row_only_reproductions": 0,
        "row_only_mismatches": 0,
        "engine_replays": 0,
        "engine_replay_problems": 0,
        "final_state_rebuilds": 0,
        "final_state_mismatches": 0,
        "schema_incomplete": 0,
        "serialisation_mismatches": 0,
        "paired_board_mismatches": 0,
    }
    problems: list[str] = []
    plies = 0
    by_unit: dict[str, list[MatchResult]] = {}

    for spec in schedule.matches:
        first = play_match(spec, bank=bank)
        plies += first.plies
        by_unit.setdefault(first.paired_unit_id, []).append(first)

        # 1. Same identity, played again in this process.
        second = play_match(spec, bank=bank)
        if first.comparable_digest() == second.comparable_digest():
            checks["reproduced_identically"] += 1
        else:
            checks["reproduction_mismatches"] += 1
            problems.append(f"match {spec.match_id} did not reproduce in-process")

        # 2. Rebuilt from the stored row alone, with no setup bank.
        rebuilt = MatchResult.from_dict(json.loads(json.dumps(first.to_dict())))
        if rebuilt.comparable() != first.comparable():
            checks["serialisation_mismatches"] += 1
            problems.append(f"match {spec.match_id} did not survive serialisation")
        third = reproduce_match(rebuilt)
        if third.comparable_digest() == first.comparable_digest():
            checks["row_only_reproductions"] += 1
        else:
            checks["row_only_mismatches"] += 1
            problems.append(f"match {spec.match_id} did not reproduce from its row")

        # 3. Pure engine replay of the stored history.
        replay_problems = replay_stored_match(first)
        checks["engine_replays"] += 1
        if replay_problems:
            checks["engine_replay_problems"] += 1
            problems.extend(replay_problems)

        # 4. The replay record rebuilds the same terminal state.
        state = rebuild_final_state(first.replay_record())
        checks["final_state_rebuilds"] += 1
        if (
            state.terminal_reason != first.terminal_reason
            or state.winner != first.winner
            or state.total_moves != first.plies
            or replay_digest(first.replay_record()) != first.replay_digest
        ):
            checks["final_state_mismatches"] += 1
            problems.append(f"match {spec.match_id}: rebuilt final state disagrees with the row")

        # 5. The row carries every field the instructions require.
        payload = first.to_dict()
        missing = [field for field in REQUIRED_RESULT_FIELDS if field not in payload]
        if missing:
            checks["schema_incomplete"] += 1
            problems.append(f"match {spec.match_id}: missing raw fields {missing}")

    # 6. Both games of a unit share one board; only the colours flip.
    for unit_id, members in by_unit.items():
        if len(members) != 2:
            continue
        first, second = members
        if first.red_setup != second.red_setup or first.blue_setup != second.blue_setup:
            checks["paired_board_mismatches"] += 1
            problems.append(f"paired unit {unit_id}: the two games used different boards")
        if sorted(row.candidate_color for row in members) != [RED, BLUE]:
            checks["paired_board_mismatches"] += 1
            problems.append(f"paired unit {unit_id}: colour assignments are not one of each")

    return {
        "matches": len(schedule.matches),
        "paired_units": len(by_unit),
        "plies": plies,
        "checks": checks,
        "problems": problems[:20],
        "problem_count": len(problems),
        "required_result_fields": list(REQUIRED_RESULT_FIELDS),
    }


def failure_stage(bank: SetupBank) -> dict:
    """Every policy failure must be raised, classified, and never substituted."""
    opponent_id = "basic_heuristic"
    opponent = policy_ref(opponent_id)
    cases: list[dict] = []
    raised = classified = quarantined = substituted = 0

    for policy_class, expected_category in BROKEN_CASES:
        policy = policy_class()
        policies = {policy.ref.token: policy, opponent.token: build_policy(opponent_id)}
        spec = MatchSpec(
            candidate=policy.ref,
            opponent=opponent,
            setup_pair_id=0,
            candidate_color=RED,
            setup_bank_version=bank.bank_version,
        )

        entry: dict = {"policy": policy.policy_id, "expected_category": expected_category}

        # Default mode: the failure must propagate.
        try:
            play_match(spec, bank=bank, policies=policies)
            entry["raised"] = False
            substituted += 1
        except PolicyFailure as error:
            raised += 1
            entry["raised"] = True
            entry["category"] = error.category
            entry["ply"] = error.ply
            entry["role"] = error.role
            if error.category == expected_category:
                classified += 1

        # Quarantine mode: recorded, unscored, and still not substituted.
        row = play_match(
            spec, bank=bank, policies=policies, on_policy_error=ON_POLICY_ERROR_QUARANTINE
        )
        entry["quarantined"] = row.errored
        entry["quarantine_score"] = row.candidate_score
        entry["quarantine_terminal_reason"] = row.terminal_reason
        entry["quarantine_category"] = row.policy_error_category
        if row.errored and row.candidate_score is None:
            quarantined += 1
        else:
            substituted += 1

        # A quarantined match must be refused by the statistics unless it is
        # explicitly acknowledged.
        try:
            build_paired_units([row])
            entry["statistics_refused_silent_error"] = False
        except Exception:  # noqa: BLE001 -- any refusal is the correct behaviour
            entry["statistics_refused_silent_error"] = True

        cases.append(entry)

    return {
        "cases": cases,
        "raised": raised,
        "correctly_classified": classified,
        "quarantined_without_score": quarantined,
        "substituted_legal_moves": substituted,
        "statistics_refusals": sum(
            1 for case in cases if case.get("statistics_refused_silent_error")
        ),
    }


# ---------------------------------------------------------------------------
# Stage 3 -- parallel reproducibility sweep
# ---------------------------------------------------------------------------


def build_sweep_schedule(matches: int) -> EvaluationSchedule:
    """One schedule of `matches` games, split evenly over the sweep matchups."""
    units_per_matchup = max(matches // (2 * len(SWEEP_MATCHUPS)), 1)
    return merge_schedules(
        [
            build_matchup_schedule(candidate, opponent, units_per_matchup)
            for candidate, opponent in SWEEP_MATCHUPS
        ],
        name="agent_03_reproducibility_sweep",
    )


def _evidence_rows(worker_count: int, results) -> list[dict]:
    """Per-match evidence from one run, for the raw reproducibility CSV.

    Built from *that run's own* results. Writing the baseline's rows once per
    worker count would produce a file that agrees with itself by construction and
    proves nothing.
    """
    return [
        {
            "worker_count": worker_count,
            "match_id": row.match_id,
            "paired_unit_id": row.paired_unit_id,
            "candidate_policy_id": row.candidate_policy_id,
            "opponent_policy_id": row.opponent_policy_id,
            "candidate_color": row.candidate_color,
            "setup_pair_id": row.setup_pair_id,
            "replay_digest": row.replay_digest,
            "winner": row.winner,
            "terminal_reason": row.terminal_reason,
            "plies": row.plies,
            "candidate_result": row.candidate_result,
        }
        for row in results
    ]


def parallel_stage(
    bank: SetupBank, schedule: EvaluationSchedule, worker_counts: "tuple[int, ...]"
) -> dict:
    """Run one schedule at several worker counts and require identical content."""
    rows: list[dict] = []
    evidence: list[dict] = []
    mismatches = 0
    problems: list[str] = []
    baseline = None

    for count in worker_counts:
        started = time.perf_counter()
        run = run_schedule(schedule.matches, bank, worker_count=count)
        elapsed = time.perf_counter() - started
        evidence.extend(_evidence_rows(count, run.results))

        if baseline is None:
            baseline = run
            differences: list[str] = []
        else:
            differences = compare_results(baseline.results, run.results)
            mismatches += len(differences)
            problems.extend(differences[:10])

        rows.append(
            {
                "worker_count": count,
                "chunk_count": run.chunk_count,
                "matches_run": run.matches_run,
                "paired_units_run": run.paired_units_run,
                "total_plies": run.plies,
                "wall_clock_seconds": elapsed,
                "matches_per_second": run.matches_run / elapsed if elapsed else 0.0,
                "speedup": rows[0]["wall_clock_seconds"] / elapsed if rows and elapsed else 1.0,
                "results_digest": run.results_digest,
                "schedule_digest": run.schedule_digest,
                "policy_errors": run.policy_errors,
                "illegal_policy_actions": run.illegal_policy_actions,
                "mismatches": len(differences),
            }
        )

    # A shuffled schedule at a mid worker count: order must not matter either.
    shuffled = schedule.shuffled(seed=4242)
    shuffle_workers = max(worker_counts[-1] // 2, 1)
    shuffled_run = run_schedule(shuffled.matches, bank, worker_count=shuffle_workers)
    shuffle_differences = compare_results(baseline.results, shuffled_run.results)
    mismatches += len(shuffle_differences)
    problems.extend(shuffle_differences[:10])
    evidence.extend(_evidence_rows(-shuffle_workers, shuffled_run.results))

    digests = {row["results_digest"] for row in rows} | {shuffled_run.results_digest}

    return {
        "worker_counts": list(worker_counts),
        "runs": rows,
        "shuffled_run": {
            "worker_count": shuffle_workers,
            "seed": 4242,
            "note": "recorded in the raw CSV with a negative worker_count",
            "results_digest": shuffled_run.results_digest,
            "mismatches": len(shuffle_differences),
        },
        "distinct_results_digests": len(digests),
        "mismatches": mismatches,
        "problems": problems[:20],
        "schedule_digest": schedule.digest,
        "schedule_fingerprint": schedule_fingerprint(schedule),
        # Private: popped before the payload is serialised.
        "_baseline_run": baseline,
        "_evidence_rows": evidence,
    }


def write_reproducibility_csv(path: Path, rows: "list[dict]") -> Path:
    """One row per (worker count, match): the per-match evidence behind the gate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "worker_count",
        "match_id",
        "paired_unit_id",
        "candidate_policy_id",
        "opponent_policy_id",
        "candidate_color",
        "setup_pair_id",
        "replay_digest",
        "winner",
        "terminal_reason",
        "plies",
        "candidate_result",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for entry in rows:
            writer.writerow(entry)
    return path


# ---------------------------------------------------------------------------
# Stage 4 -- statistical validation
# ---------------------------------------------------------------------------

WIN, DRAW, LOSS = 1.0, 0.5, 0.0


def statistics_stage(resamples: int) -> dict:
    """Synthetic tables with known answers, one check per instruction bullet."""
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: object = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    # -- effective win rate on known tables -------------------------------
    for name, outcomes, expected in (
        ("all_wins_give_1.0", [(WIN, WIN)] * 32, 1.0),
        ("all_losses_give_0.0", [(LOSS, LOSS)] * 32, 0.0),
        ("all_draws_give_0.5", [(DRAW, DRAW)] * 32, 0.5),
        ("colour_split_gives_0.5", [(WIN, LOSS)] * 32, 0.5),
        ("mixed_outcomes", [(WIN, WIN)] * 10 + [(WIN, DRAW)] * 5
         + [(LOSS, DRAW)] * 5 + [(LOSS, LOSS)] * 12, 0.46875),
    ):
        summary = summarize_matchup(synthetic_results(outcomes), resamples=resamples)
        record(name, abs(summary.effective_win_rate - expected) < 1e-12,
               summary.effective_win_rate)
        record(
            f"{name}_reports_wdl",
            summary.counts.games == 2 * len(outcomes),
            f"{summary.counts.wins}/{summary.counts.draws}/{summary.counts.losses}",
        )

    # -- the resampling unit is the paired unit ---------------------------
    # Every unit here scores exactly 0.5 (one win, one loss), so a bootstrap over
    # units can only produce 0.5 and its interval must be a point. A bootstrap
    # over the individual games sees N/2 wins and N/2 losses and must land near
    # the normal-theory width for independent games at p = 0.5.
    rows = synthetic_results([(WIN, LOSS)] * 32)
    units = build_paired_units(rows)
    paired = paired_bootstrap_interval(units, resamples=resamples, seed=5)
    scores = [float(row.candidate_score) for row in rows]
    game_level = bootstrap_interval(scores, resamples=resamples, seed=5)
    expected_game_width = 2 * 1.959964 * 0.5 / math.sqrt(len(scores))
    record(
        "paired_bootstrap_is_not_a_game_bootstrap",
        paired.width == 0.0 and game_level.width >= 0.8 * expected_game_width,
        f"paired width {paired.width:.4f}; game-level width {game_level.width:.4f} "
        f"against a normal-theory expectation of {expected_game_width:.4f}",
    )

    # -- bootstrap reproducibility ----------------------------------------
    values = [unit.score for unit in build_paired_units(
        synthetic_results([(WIN, WIN)] * 10 + [(WIN, DRAW)] * 6 + [(LOSS, LOSS)] * 16)
    )]
    first = bootstrap_interval(values, resamples=resamples, seed=777)
    second = bootstrap_interval(values, resamples=resamples, seed=777)
    record(
        "same_seed_gives_an_identical_interval",
        (first.lower, first.upper) == (second.lower, second.upper),
        f"[{first.lower:.6f}, {first.upper:.6f}]",
    )
    spread = {
        (bootstrap_interval(values, resamples=resamples, seed=seed).lower,
         bootstrap_interval(values, resamples=resamples, seed=seed).upper)
        for seed in range(12)
    }
    record("different_seeds_move_the_interval", len(spread) > 1, f"{len(spread)} distinct")
    record(
        "the_interval_contains_the_point_estimate",
        first.lower <= sum(values) / len(values) <= first.upper,
        f"{sum(values) / len(values):.4f}",
    )
    small = paired_bootstrap_interval(units[:8], resamples=resamples, seed=6)
    record(
        "more_units_narrow_the_interval",
        paired_bootstrap_interval(
            build_paired_units(synthetic_results([(WIN, DRAW)] * 8 + [(LOSS, LOSS)] * 8)),
            resamples=resamples, seed=6,
        ).width
        > paired_bootstrap_interval(
            build_paired_units(
                synthetic_results(([(WIN, DRAW)] * 8 + [(LOSS, LOSS)] * 8) * 8)
            ),
            resamples=resamples, seed=6,
        ).width,
        f"reference width {small.width:.4f}",
    )

    # -- colour split ------------------------------------------------------
    split = color_split(synthetic_results([(WIN, LOSS)] * 16))
    record(
        "colour_split_separates_red_from_blue",
        split["red"]["effective_win_rate"] == 1.0
        and split["blue"]["effective_win_rate"] == 0.0,
        f"red {split['red']['effective_win_rate']}, blue {split['blue']['effective_win_rate']}",
    )
    record(
        "colour_split_records_first_player",
        split["red"]["moves_first"] and not split["blue"]["moves_first"],
    )

    # -- missing and duplicate pair detection ------------------------------
    clean = list(synthetic_results([(WIN, LOSS)] * 8))
    record("a_clean_table_reports_no_problems", detect_result_problems(clean) == [])
    record(
        "a_duplicate_row_is_detected",
        any("duplicate" in problem for problem in detect_result_problems(clean + [clean[0]])),
    )
    record(
        "a_half_unit_is_detected",
        any("game(s)" in problem for problem in detect_result_problems(clean[:-1])),
    )
    twin = MatchResult.from_dict({**clean[0].to_dict(), "match_id": "m-forced-twin"})
    record(
        "two_games_of_the_same_colour_are_detected",
        any("colour" in problem for problem in detect_result_problems([clean[0], twin])),
    )
    refused = False
    try:
        build_paired_units(clean[:-1])
    except Exception:  # noqa: BLE001 -- a refusal is the expected behaviour
        refused = True
    record("a_broken_table_cannot_be_summarised", refused)

    # -- ordering invariance ----------------------------------------------
    mixed = list(synthetic_results(
        [(WIN, WIN)] * 8 + [(WIN, DRAW)] * 4 + [(LOSS, LOSS)] * 12
    ))
    baseline = summarize_matchup(mixed, resamples=resamples, seed=31).to_dict()
    invariant = True
    for seed in (1, 2, 3, 4):
        shuffled = list(mixed)
        random.Random(seed).shuffle(shuffled)
        if summarize_matchup(shuffled, resamples=resamples, seed=31).to_dict() != baseline:
            invariant = False
    record("aggregation_is_invariant_to_row_order", invariant)

    multi = mixed + list(
        synthetic_results([(WIN, WIN)] * 6, candidate="candidate@1.0.0", opponent="other@1.0.0")
    )
    shuffled_multi = list(multi)
    random.Random(8).shuffle(shuffled_multi)
    record(
        "run_aggregation_is_invariant_to_row_order",
        summarize_run(multi, resamples=resamples) == summarize_run(shuffled_multi, resamples=resamples),
    )

    # -- unit histogram ----------------------------------------------------
    histogram = unit_score_histogram(
        build_paired_units(synthetic_results([(WIN, WIN)] * 4 + [(LOSS, LOSS)] * 4))
    )
    record(
        "the_unit_histogram_separates_sweeps_from_splits",
        histogram == {"0.0": 4, "0.25": 0, "0.5": 0, "0.75": 0, "1.0": 4},
        histogram,
    )

    # -- league rating -----------------------------------------------------
    league_rows = list(synthetic_results([(WIN, WIN)] * 8, candidate="a@1", opponent="b@1"))
    league_rows += list(synthetic_results([(WIN, WIN)] * 8, candidate="a@1", opponent="c@1"))
    league_rows += list(synthetic_results([(WIN, DRAW)] * 8, candidate="b@1", opponent="c@1"))
    ratings = bradley_terry_ratings(league_rows)
    record(
        "league_ratings_recover_the_true_order",
        ratings.ranking == ("a@1", "b@1", "c@1"),
        ratings.ranking,
    )
    shuffled_league = list(league_rows)
    random.Random(2).shuffle(shuffled_league)
    record(
        "league_ratings_are_deterministic",
        bradley_terry_ratings(shuffled_league).to_dict() == ratings.to_dict(),
    )
    equal = bradley_terry_ratings(
        synthetic_results([(WIN, LOSS)] * 16, candidate="a@1", opponent="b@1")
    )
    record(
        "equal_policies_receive_equal_ratings",
        abs(equal.ratings["a@1"] - equal.ratings["b@1"]) < 1e-9,
        f"{equal.ratings['a@1']:.3f} vs {equal.ratings['b@1']:.3f}",
    )
    swept = bradley_terry_ratings(
        synthetic_results([(WIN, WIN)] * 8, candidate="a@1", opponent="b@1")
    )
    record(
        "the_prior_keeps_an_undefeated_policy_finite",
        swept.converged and all(abs(value) < 1e6 for value in swept.ratings.values()),
        f"{swept.ratings}",
    )

    # -- guard rails -------------------------------------------------------
    guarded = 0
    for probe in (
        lambda: effective_win_rate(0, 0, 0),
        lambda: effective_win_rate(-1, 0, 1),
        lambda: bootstrap_interval([]),
        lambda: bootstrap_interval([0.5], resamples=0),
        lambda: summarize_matchup([]),
        lambda: summarize_run([]),
    ):
        try:
            probe()
        except Exception:  # noqa: BLE001 -- every one of these must raise
            guarded += 1
    record("invalid_inputs_are_rejected", guarded == 6, f"{guarded}/6")

    passed = sum(1 for check in checks if check["passed"])
    return {
        "total": len(checks),
        "passed": passed,
        "failed": len(checks) - passed,
        "resamples": resamples,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Stage 5 -- acceptance statistics over the swept schedule
# ---------------------------------------------------------------------------


def acceptance_statistics_stage(results, resamples: int, seed: int) -> dict:
    """Full-resample statistics over the sweep. Informational, not calibration."""
    summary = summarize_run(results, resamples=resamples, seed=seed, include_setup_table=False)
    counts = OutcomeCounts.from_results(results)
    return {
        "note": (
            "Informational. The sweep exists to prove reproducibility, and its "
            "matchups were chosen for statistical variety rather than to measure "
            "the ladder. Agent 4 owns baseline calibration and the strength-tier gate."
        ),
        "summary": summary,
        "pooled": counts.to_dict(),
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
    "stratego/evaluation/match_runner.py",
    "stratego/evaluation/scheduler.py",
    "stratego/evaluation/statistics.py",
    "stratego/evaluation/reporting.py",
    "tests/evaluation/test_match_runner.py",
    "tests/evaluation/test_statistics.py",
    "tests/evaluation/test_parallel_reproducibility.py",
    "scripts/run_phase4_agent03.py",
    "reports/phase_4_data/agent_03_match_runner_statistics.json",
    "reports/phase_4_data/agent_03_reproducibility_raw.csv",
    "reports/phase_4_data/agent_03_acceptance_results.csv",
    "reports/phase_4_data/agent_03_acceptance_replays.jsonl",
]

FILES_MODIFIED = [
    "stratego/evaluation/__init__.py",
    "reports/phase_4_implementation_report.md",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=int, default=512,
                        help="matches in the reproducibility sweep")
    parser.add_argument("--runner-matches", type=int, default=64,
                        help="matches in the exact-reproduction stage")
    parser.add_argument("--workers", type=str, default="1,2,4,8")
    parser.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--test-resamples", type=int, default=2000,
                        help="resamples for the statistical validation stage")
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bank-size", type=int, default=DEFAULT_BANK_SIZE)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--replays", type=Path, default=DEFAULT_REPLAYS)
    options = parser.parse_args()

    if options.quick:
        options.matches = 32
        options.runner_matches = 8
        options.workers = "1,2"
        options.resamples = 500
        options.test_resamples = 200
        options.bank_size = 32

    worker_counts = tuple(int(value) for value in options.workers.split(",") if value.strip())
    started = time.perf_counter()

    print("[1/7] prerequisites")
    prerequisites = prerequisite_stage()
    for problem in prerequisites["problems"]:
        print(f"      PROBLEM: {problem}")

    bank = SetupBank.generate(size=options.bank_size)
    print(f"      setup bank: {len(bank)} pairs, digest {bank.digest()[:12]}")

    print(f"[2/7] runner correctness over {options.runner_matches} matches")
    runner = runner_stage(bank, options.runner_matches)

    print(f"[3/7] policy-failure handling ({len(BROKEN_CASES)} broken policies)")
    failures = failure_stage(bank)

    schedule = require_valid_schedule(build_sweep_schedule(options.matches), bank)
    print(
        f"[4/7] parallel sweep: {len(schedule)} matches x worker counts "
        f"{list(worker_counts)}"
    )
    sweep = parallel_stage(bank, schedule, worker_counts)
    baseline_run = sweep.pop("_baseline_run")
    evidence_rows = sweep.pop("_evidence_rows")
    baseline_results = baseline_run.results
    raw_csv_path = write_reproducibility_csv(options.raw_csv, evidence_rows)

    print(f"[5/7] statistical validation ({options.test_resamples} resamples)")
    statistics = statistics_stage(options.test_resamples)
    print(f"      {statistics['passed']}/{statistics['total']} checks passed")

    print(f"[6/7] acceptance statistics ({options.resamples:,} resamples)")
    acceptance = acceptance_statistics_stage(
        baseline_results, options.resamples, options.bootstrap_seed
    )

    # Artefacts: the CSV keeps the digests, the sidecar keeps the histories.
    results_csv = write_results_csv(options.results_csv, baseline_results)
    replays_path, replay_count = write_replays_jsonl(options.replays, baseline_results)
    reloaded = read_replays_jsonl(replays_path)
    sidecar_ok = all(
        replay_digest(reloaded[row.match_id]) == row.replay_digest
        for row in baseline_results
        if not row.errored
    )
    stripped = attach_replay_reference(baseline_results, replays_path.name)
    sidecar_digest_stable = results_digest(stripped) == results_digest(baseline_results)
    print(
        f"      {results_csv.name} ({results_csv.stat().st_size:,} bytes), "
        f"{replays_path.name} ({replay_count} replays, "
        f"{replays_path.stat().st_size:,} bytes)"
    )

    if options.skip_pytest:
        print("[7/7] skipping the test suite")
        tests = {
            "total": 0, "passed": 0, "failed": 0, "skipped": 0, "expected_failures": 0,
            "errors": 0, "seconds": 0.0, "exit_code": None, "failure_lines": [],
            "skipped_by_request": True,
        }
    else:
        print("[7/7] running the repository test suite")
        tests = pytest_stage()

    checks = runner["checks"]
    status_checks = {
        "prerequisites_pass": not prerequisites["problems"],
        "exact_reproduction_works": (
            checks["reproduction_mismatches"] == 0
            and checks["reproduced_identically"] == runner["matches"]
        ),
        "row_only_reproduction_works": (
            checks["row_only_mismatches"] == 0
            and checks["row_only_reproductions"] == runner["matches"]
        ),
        "stored_rows_replay_through_the_engine": checks["engine_replay_problems"] == 0,
        "replay_records_rebuild_the_final_state": checks["final_state_mismatches"] == 0,
        "raw_result_schema_complete": checks["schema_incomplete"] == 0,
        "paired_units_share_a_board": checks["paired_board_mismatches"] == 0,
        "policy_failures_are_loud": failures["raised"] == len(BROKEN_CASES),
        "policy_failures_are_classified": (
            failures["correctly_classified"] == len(BROKEN_CASES)
        ),
        "no_substituted_legal_moves": failures["substituted_legal_moves"] == 0,
        "statistics_refuse_silent_errors": failures["statistics_refusals"] == len(BROKEN_CASES),
        "parallel_results_identical": sweep["mismatches"] == 0,
        "one_results_digest_across_worker_counts": sweep["distinct_results_digests"] == 1,
        "worker_counts_tested": len(worker_counts) >= 2,
        "statistics_checks_pass": statistics["failed"] == 0,
        "replay_sidecar_round_trips": sidecar_ok,
        "digest_survives_dropping_histories": sidecar_digest_stable,
        "no_policy_errors_in_the_sweep": all(
            entry["policy_errors"] == 0 for entry in sweep["runs"]
        ),
        "tests_green": options.skip_pytest or (tests["failed"] == 0 and tests["errors"] == 0),
    }
    status = "PASS" if all(status_checks.values()) else "FAIL"

    league = acceptance["summary"]["league"]
    payload = {
        "agent": "phase_4_agent_03_match_runner_statistics",
        "status": status,
        "status_checks": status_checks,
        "frozen_contracts": {
            "implementation_version": IMPLEMENTATION_VERSION,
            "rules_version": RULES_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "phase_3_backend_decision": "KEEP_PYTHON",
        },
        "match_result_schema_version": MATCH_RESULT_SCHEMA_VERSION,
        "runner_version": MATCH_RUNNER_VERSION,
        "statistics_version": STATISTICS_VERSION,
        "scheduler_version": SCHEDULER_VERSION,
        "reporting_version": REPORTING_VERSION,
        "policy_interface_version": POLICY_INTERFACE_VERSION,
        "evaluation_suite_version": EVALUATION_SUITE_VERSION,
        "setup_bank_version": SETUP_BANK_VERSION,
        "rules_context": EVALUATION_RULES.context,
        "matches_run": sweep["runs"][0]["matches_run"],
        "paired_units_run": sweep["runs"][0]["paired_units_run"],
        "total_matches_played": (
            sum(entry["matches_run"] for entry in sweep["runs"])
            + len(schedule)  # the shuffled rerun
            + runner["matches"] * 3
            + len(BROKEN_CASES) * 2
        ),
        "worker_counts_tested": list(worker_counts),
        "parallel_reproducibility_mismatches": sweep["mismatches"],
        "illegal_policy_actions": sum(
            entry["illegal_policy_actions"] for entry in sweep["runs"]
        ),
        "policy_errors": sum(entry["policy_errors"] for entry in sweep["runs"]),
        "statistics_unit_tests": statistics["total"],
        "statistics_unit_tests_passed": statistics["passed"],
        "statistics_unit_tests_failed": statistics["failed"],
        "bootstrap_resamples_acceptance": options.resamples,
        "bootstrap_seed": options.bootstrap_seed,
        "bootstrap_method": BOOTSTRAP_METHOD,
        "league_method": LEAGUE_METHOD,
        "league_ranking": league["ranking"],
        "league_ratings": league["ratings"],
        "test_total": tests["total"],
        "test_passed": tests["passed"],
        "test_failed": tests["failed"] + tests["errors"],
        "prerequisites": prerequisites,
        "runner_correctness": runner,
        "policy_failure_handling": failures,
        "parallel_sweep": sweep,
        "statistical_validation": statistics,
        "acceptance_statistics": acceptance,
        "artefacts": {
            "results_csv": str(results_csv.relative_to(REPOSITORY_ROOT)),
            "results_csv_bytes": results_csv.stat().st_size,
            "replays_jsonl": str(replays_path.relative_to(REPOSITORY_ROOT)),
            "replays_jsonl_bytes": replays_path.stat().st_size,
            "replays_written": replay_count,
            "reproducibility_csv": str(raw_csv_path.relative_to(REPOSITORY_ROOT)),
            "reproducibility_csv_rows": len(evidence_rows),
            "sidecar_round_trip_ok": sidecar_ok,
            "digest_stable_without_histories": sidecar_digest_stable,
        },
        "run_manifest": run_manifest(
            baseline_run, schedule_manifest=schedule.manifest(bank)
        ),
        "tests": tests,
        "files_created": FILES_CREATED,
        "files_modified": FILES_MODIFIED,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
        },
        "quick_mode": options.quick,
        "total_seconds": time.perf_counter() - started,
    }

    write_json(options.output, payload)

    print()
    print(f"status                          {status}")
    print(f"matches in the sweep            {payload['matches_run']} "
          f"({payload['paired_units_run']} paired units)")
    print(f"worker counts tested            {list(worker_counts)}")
    print(f"parallel mismatches             {sweep['mismatches']}")
    print(f"distinct results digests        {sweep['distinct_results_digests']} (want 1)")
    print(f"exact reproductions             {checks['reproduced_identically']} "
          f"+ {checks['row_only_reproductions']} from stored rows")
    print(f"engine replay problems          {checks['engine_replay_problems']}")
    print(f"policy failures raised          {failures['raised']}/{len(BROKEN_CASES)}")
    print(f"substituted legal moves         {failures['substituted_legal_moves']} (want 0)")
    print(f"statistics checks               {statistics['passed']}/{statistics['total']}")
    print(f"bootstrap resamples             {options.resamples:,} (seed {options.bootstrap_seed})")
    print(f"tests passed / failed           {payload['test_passed']} / {payload['test_failed']}")
    print()
    print(render_worker_table(sweep["runs"]))
    print()
    print(render_matchup_table(acceptance["summary"]["per_matchup"]))
    print()
    print(render_league_table(league))
    print()
    print(f"written                         {options.output.relative_to(REPOSITORY_ROOT)}")
    for path in (raw_csv_path, results_csv, replays_path):
        print(f"                                {path.relative_to(REPOSITORY_ROOT)}")
    for name, ok in sorted(status_checks.items()):
        if not ok:
            print(f"failed check                    {name}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
