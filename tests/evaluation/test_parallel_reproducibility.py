"""Parallel execution must not change a single game.

Kept in its own module because every test here starts worker processes, which
costs roughly a second per pool under macOS `spawn`. The schedule is therefore
small: the mechanism is what is under test, and the large sweep (512 matches at
1, 2, 4 and 8 workers) lives in `scripts/run_phase4_agent03.py`.

What is actually being proved
-----------------------------
Every input to a game is fixed by `MatchSpec` before dispatch, and no identifier
reads a worker index, shard boundary or arrival order. These tests check that
claim from the outside: run the same schedule several ways and require the
results to be identical in content, whatever order they come back in.
"""

import pytest

from stratego.evaluation.match_runner import (
    MatchRunnerError,
    compare_results,
    results_digest,
    run_schedule,
    suggested_worker_count,
)
from stratego.evaluation.reporting import attach_replay_reference
from stratego.evaluation.scheduler import build_league_schedule, build_matchup_schedule
from stratego.evaluation.setup_bank import SetupBank
from stratego.evaluation.statistics import summarize_run

#: 8 paired units = 16 matches. Enough to spread over 4 workers with several
#: chunks each; small enough that four pools cost a few seconds in total.
UNITS = 8

#: Serial plus three parallel counts. 3 is deliberately not a divisor of the
#: match count, so an off-by-one in chunking shows up as a lost or duplicated
#: match rather than hiding behind even division.
WORKER_COUNTS = (1, 2, 3, 4)


@pytest.fixture(scope="module")
def bank() -> SetupBank:
    return SetupBank.generate(size=UNITS)


@pytest.fixture(scope="module")
def schedule():
    return build_matchup_schedule("basic_heuristic", "tactical_rule_based", UNITS)


@pytest.fixture(scope="module")
def runs(bank, schedule):
    """The same schedule executed at each worker count."""
    return {count: run_schedule(schedule.matches, bank, worker_count=count) for count in WORKER_COUNTS}


# ---------------------------------------------------------------------------
# The core gate
# ---------------------------------------------------------------------------


def test_every_worker_count_produces_the_same_results(runs):
    serial = runs[1]
    for count, run in runs.items():
        assert compare_results(serial.results, run.results) == [], f"worker_count={count}"
        assert run.results_digest == serial.results_digest, f"worker_count={count}"


def test_every_worker_count_runs_every_match(runs, schedule):
    for count, run in runs.items():
        assert run.matches_run == len(schedule.matches), f"worker_count={count}"
        assert run.paired_units_run == UNITS, f"worker_count={count}"
        identifiers = sorted(row.match_id for row in run.results)
        assert identifiers == sorted(m.match_id for m in schedule.matches)
        assert len(set(identifiers)) == len(identifiers), f"worker_count={count}"


@pytest.mark.parametrize(
    "field",
    ["match_id", "red_setup", "blue_setup", "replay_digest", "winner", "terminal_reason", "plies"],
)
def test_the_gate_fields_are_identical_across_worker_counts(runs, field):
    """The specific fields the instructions name: identity, setups, replay
    digests, results, terminal reasons and ply counts."""
    serial = {row.match_id: getattr(row, field) for row in runs[1].results}
    for count, run in runs.items():
        parallel = {row.match_id: getattr(row, field) for row in run.results}
        assert parallel == serial, f"{field} differs at worker_count={count}"


def test_action_histories_are_identical_across_worker_counts(runs):
    serial = {row.match_id: row.action_history for row in runs[1].results}
    for count, run in runs.items():
        assert {row.match_id: row.action_history for row in run.results} == serial, (
            f"worker_count={count}"
        )
        assert all(row.action_history for row in run.results)


def test_policy_seeds_do_not_depend_on_the_worker_count(runs):
    serial = {
        row.match_id: (row.candidate_seed, row.opponent_seed) for row in runs[1].results
    }
    for count, run in runs.items():
        seeds = {row.match_id: (row.candidate_seed, row.opponent_seed) for row in run.results}
        assert seeds == serial, f"worker_count={count}"


def test_statistics_are_identical_across_worker_counts(runs):
    """The reproducibility guarantee has to survive as far as the reported numbers."""
    baseline = summarize_run(runs[1].results, resamples=500)
    for count, run in runs.items():
        assert summarize_run(run.results, resamples=500) == baseline, f"worker_count={count}"


# ---------------------------------------------------------------------------
# Order and sharding
# ---------------------------------------------------------------------------


def test_shuffling_the_schedule_changes_nothing(bank, schedule):
    shuffled = schedule.shuffled(seed=99)
    assert [m.match_id for m in shuffled.matches] != [m.match_id for m in schedule.matches]
    baseline = run_schedule(schedule.matches, bank, worker_count=2)
    reordered = run_schedule(shuffled.matches, bank, worker_count=2)
    assert compare_results(baseline.results, reordered.results) == []
    assert reordered.results_digest == baseline.results_digest
    assert reordered.schedule_digest == baseline.schedule_digest


def test_the_chunk_count_does_not_change_results(bank, schedule):
    baseline = run_schedule(schedule.matches, bank, worker_count=2, chunks_per_worker=1)
    for chunks_per_worker in (2, 8):
        run = run_schedule(
            schedule.matches, bank, worker_count=2, chunks_per_worker=chunks_per_worker
        )
        assert compare_results(baseline.results, run.results) == []
        assert run.chunk_count >= baseline.chunk_count


def test_more_workers_than_matches_is_harmless(bank, schedule):
    baseline = run_schedule(schedule.matches, bank, worker_count=1)
    crowded = run_schedule(schedule.matches, bank, worker_count=len(schedule.matches) + 4)
    assert compare_results(baseline.results, crowded.results) == []
    assert crowded.matches_run == len(schedule.matches)


def test_results_are_sorted_regardless_of_completion_order(runs):
    for count, run in runs.items():
        identifiers = [row.match_id for row in run.results]
        assert identifiers == sorted(identifiers), f"worker_count={count}"


# ---------------------------------------------------------------------------
# Storage choices must not affect the gate
# ---------------------------------------------------------------------------


def test_the_digest_survives_dropping_the_action_histories(runs):
    """A digest-only artefact must stay comparable with a full one."""
    serial = runs[1].results
    stripped = attach_replay_reference(serial, "sidecar.jsonl")
    assert all(row.action_history is None for row in stripped)
    assert results_digest(stripped) == results_digest(serial)
    assert compare_results(serial, stripped) == []


def test_parallel_runs_agree_without_storing_histories(bank, schedule):
    lean_serial = run_schedule(schedule.matches, bank, worker_count=1, record_actions=False)
    lean_parallel = run_schedule(schedule.matches, bank, worker_count=3, record_actions=False)
    assert compare_results(lean_serial.results, lean_parallel.results) == []
    assert lean_parallel.results_digest == lean_serial.results_digest


# ---------------------------------------------------------------------------
# Several matchups at once
# ---------------------------------------------------------------------------


def test_a_multi_matchup_league_is_reproducible_in_parallel(bank):
    """Chunks mix matchups, so a worker-local cache must not leak between them."""
    league = build_league_schedule(
        ["random_legal", "basic_heuristic", "tactical_rule_based"], 3, name="parallel_league"
    )
    serial = run_schedule(league.matches, bank, worker_count=1)
    parallel = run_schedule(league.matches, bank, worker_count=4)
    assert compare_results(serial.results, parallel.results) == []
    assert parallel.results_digest == serial.results_digest
    assert len({row.matchup for row in parallel.results}) == 3


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def test_an_invalid_worker_count_is_rejected(bank, schedule):
    with pytest.raises(MatchRunnerError, match="worker_count must be"):
        run_schedule(schedule.matches, bank, worker_count=0)
    with pytest.raises(MatchRunnerError, match="chunks_per_worker must be"):
        run_schedule(schedule.matches, bank, worker_count=2, chunks_per_worker=0)


def test_the_suggested_worker_count_is_sane():
    suggested = suggested_worker_count()
    assert 1 <= suggested <= 8
    assert suggested_worker_count(maximum=2) <= 2
