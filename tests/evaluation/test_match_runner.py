"""Match runner, raw result schema, and scheduling tests.

Covers the Agent 3 gates that do not need worker processes:

- a match reproduces exactly from its identity;
- a raw row is sufficient to rebuild and replay the game;
- a policy failure is loud and never becomes a substitute legal move;
- quarantine records the failure without scoring the match;
- schedules are storable, digest-verified and order-independent.

Parallel execution has its own file, so this one stays fast.
"""

import json

import pytest

from stratego.engine.constants import BLUE, EVALUATION_RULES, RED, TRAINING_RULES
from stratego.engine.replay import rebuild_final_state
from stratego.engine.setup import deserialize_setup
from stratego.evaluation.match_runner import (
    ERROR_CONTRACT_VIOLATION,
    ERROR_ILLEGAL_ACTION,
    ERROR_POLICY_EXCEPTION,
    MATCH_RESULT_SCHEMA_VERSION,
    ON_POLICY_ERROR_QUARANTINE,
    RESULT_DRAW,
    RESULT_ERROR,
    RESULT_LABELS,
    RESULT_LOSS,
    RESULT_WIN,
    TERMINAL_POLICY_ERROR,
    MatchResult,
    MatchRunnerError,
    PolicyFailure,
    compare_results,
    play_match,
    replay_digest,
    replay_stored_match,
    reproduce_match,
    resolve_policies,
    results_digest,
    run_schedule,
)
from stratego.evaluation.match_spec import (
    MatchSpec,
    build_paired_schedule,
    schedule_matches,
    sibling_match,
)
from stratego.evaluation.policy import Policy, PolicyRef, PolicyRequirements, PolicyResult
from stratego.evaluation.registry import ALL_POLICY_IDS, policy_ref
from stratego.evaluation.scheduler import (
    EvaluationSchedule,
    ScheduleError,
    build_gauntlet_schedule,
    build_ladder_schedule,
    build_league_schedule,
    build_matchup_schedule,
    merge_schedules,
    require_valid_schedule,
    schedule_fingerprint,
)
from stratego.evaluation.setup_bank import SetupBank

# A small bank keeps the module fast; nothing here depends on bank size.
BANK_SIZE = 8


@pytest.fixture(scope="module")
def bank() -> SetupBank:
    return SetupBank.generate(size=BANK_SIZE)


@pytest.fixture(scope="module")
def schedule():
    return build_matchup_schedule("basic_heuristic", "tactical_rule_based", 3)


@pytest.fixture(scope="module")
def played(bank, schedule):
    """One run of a small schedule, reused by the read-only assertions."""
    return run_schedule(schedule.matches, bank, worker_count=1)


# ---------------------------------------------------------------------------
# Broken policies, used only here
# ---------------------------------------------------------------------------


class IllegalActionPolicy(Policy):
    """Returns an action that is never legal. Must be caught, never substituted."""

    policy_id = "broken_illegal"
    policy_version = "1.0.0"
    requirements = PolicyRequirements()

    def decide(self, request):
        illegal = max(request.legal_actions) + 1
        while illegal in request.legal_actions:  # pragma: no cover -- max+1 suffices
            illegal += 1
        return self.result(request, illegal)


class ExplodingPolicy(Policy):
    """Raises. The runner must classify it, not swallow it."""

    policy_id = "broken_exploding"
    policy_version = "1.0.0"

    def decide(self, request):
        raise ZeroDivisionError("deliberate failure")


class WrongSeedPolicy(Policy):
    """Returns a result whose decision seed does not match the request."""

    policy_id = "broken_seed"
    policy_version = "1.0.0"

    def decide(self, request):
        return PolicyResult(
            selected_action_id=request.legal_actions[0],
            policy=request.policy,
            decision_seed=request.decision_seed + 1,
        )


class NonResultPolicy(Policy):
    """Returns a bare action identifier instead of a PolicyResult."""

    policy_id = "broken_type"
    policy_version = "1.0.0"

    def decide(self, request):
        return request.legal_actions[0]


def _broken_spec(policy: Policy, bank: SetupBank) -> tuple[MatchSpec, dict]:
    """A match pitting a broken policy against a real one."""
    opponent = policy_ref("basic_heuristic")
    spec = MatchSpec(
        candidate=policy.ref,
        opponent=opponent,
        setup_pair_id=0,
        candidate_color=RED,
        setup_bank_version=bank.bank_version,
    )
    from stratego.evaluation.registry import build_policy

    return spec, {policy.ref.token: policy, opponent.token: build_policy("basic_heuristic")}


# ---------------------------------------------------------------------------
# Playing a match
# ---------------------------------------------------------------------------


def test_a_match_reaches_a_terminal_state(played):
    assert played.matches_run == 6
    assert played.paired_units_run == 3
    for row in played.results:
        assert row.candidate_result in RESULT_LABELS
        assert not row.errored
        assert row.plies > 0
        assert row.decisions == row.plies


def test_the_result_schema_carries_every_required_field(played):
    """The instructions list a minimum field set; none of it may be missing."""
    required = {
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
    }
    payload = played.results[0].to_dict()
    assert required <= set(payload)
    assert payload["schema_version"] == MATCH_RESULT_SCHEMA_VERSION


def test_result_serialisation_round_trips(played):
    for row in played.results:
        rebuilt = MatchResult.from_dict(json.loads(json.dumps(row.to_dict())))
        assert rebuilt.comparable() == row.comparable()
        assert rebuilt.action_history == row.action_history
        assert rebuilt.replay_digest == row.replay_digest


def test_winner_and_score_agree(played):
    for row in played.results:
        if row.candidate_result == RESULT_WIN:
            assert row.winner == row.candidate_color and row.candidate_score == 1.0
        elif row.candidate_result == RESULT_LOSS:
            assert row.winner == row.opponent_color and row.candidate_score == 0.0
        else:
            assert row.candidate_result == RESULT_DRAW
            assert row.winner is None and row.draw and row.candidate_score == 0.5


def test_both_colors_are_played_in_each_unit(played):
    by_unit: dict[str, list[int]] = {}
    for row in played.results:
        by_unit.setdefault(row.paired_unit_id, []).append(row.candidate_color)
    assert all(sorted(colors) == [RED, BLUE] for colors in by_unit.values())


def test_the_two_games_of_a_unit_share_a_board(played):
    """`color_swap_same_board`: only the colour assignment flips."""
    by_unit: dict[str, list[MatchResult]] = {}
    for row in played.results:
        by_unit.setdefault(row.paired_unit_id, []).append(row)
    for members in by_unit.values():
        first, second = members
        assert first.red_setup == second.red_setup
        assert first.blue_setup == second.blue_setup
        assert first.match_id != second.match_id


def test_setups_match_the_bank(played, bank):
    for row in played.results:
        red, blue = row.spec().resolve_setups(bank)
        assert deserialize_setup(row.red_setup) == red
        assert deserialize_setup(row.blue_setup) == blue


def test_wall_clock_is_recorded_but_not_part_of_identity(played):
    for row in played.results:
        assert row.wall_clock_seconds > 0.0
        assert "wall_clock_seconds" not in row.comparable()


# ---------------------------------------------------------------------------
# Exact reproduction
# ---------------------------------------------------------------------------


def test_replaying_the_same_spec_gives_an_identical_result(bank, schedule):
    for spec in schedule.matches[:2]:
        first = play_match(spec, bank=bank)
        second = play_match(spec, bank=bank)
        assert first.comparable_digest() == second.comparable_digest()
        assert first.action_history == second.action_history


def test_a_stored_row_reproduces_without_the_bank(played):
    """The whole point of storing both setups on the row."""
    for row in played.results[:2]:
        again = reproduce_match(row)
        assert again.comparable_digest() == row.comparable_digest()


def test_a_row_replays_through_the_engine(played):
    for row in played.results:
        assert replay_stored_match(row) == []


def test_the_replay_record_rebuilds_the_final_state(played):
    for row in played.results[:2]:
        state = rebuild_final_state(row.replay_record())
        assert state.terminal
        assert state.terminal_reason == row.terminal_reason
        assert state.winner == row.winner
        assert state.total_moves == row.plies
        assert replay_digest(row.replay_record()) == row.replay_digest


def test_a_tampered_action_history_is_detected(played):
    row = played.results[0]
    tampered = MatchResult.from_dict({**row.to_dict(), "plies": row.plies + 1})
    problems = replay_stored_match(tampered)
    assert problems and any("plies" in problem for problem in problems)


def test_a_tampered_identity_field_is_detected(played):
    row = played.results[0]
    broken = MatchResult.from_dict({**row.to_dict(), "setup_pair_id": row.setup_pair_id + 1})
    with pytest.raises(MatchRunnerError, match="does not match the specification"):
        broken.spec()


def test_a_tampered_rules_payload_is_detected(played):
    row = played.results[0]
    payload = row.to_dict()
    payload["rules_payload"] = {**payload["rules_payload"], "battleless_move_limit": 7}
    with pytest.raises(MatchRunnerError, match="rules_payload rebuilds"):
        MatchResult.from_dict(payload).rules_config()


def test_the_rules_configuration_survives_a_round_trip(played):
    for row in played.results[:2]:
        assert row.rules_config() == EVALUATION_RULES


def test_results_digest_ignores_row_order(played):
    forward = results_digest(played.results)
    backward = results_digest(list(reversed(played.results)))
    assert forward == backward


def test_results_digest_ignores_whether_histories_were_stored(bank, schedule):
    """A digest-only run and a full run must remain comparable."""
    full = run_schedule(schedule.matches, bank, worker_count=1, record_actions=True)
    lean = run_schedule(schedule.matches, bank, worker_count=1, record_actions=False)
    assert all(row.action_history is None for row in lean.results)
    assert full.results_digest == lean.results_digest
    assert compare_results(full.results, lean.results) == []


def test_compare_results_reports_a_real_difference(played):
    row = played.results[0]
    altered = MatchResult.from_dict({**row.to_dict(), "plies": row.plies + 5})
    problems = compare_results(played.results, [altered, *played.results[1:]])
    assert any("plies differs" in problem for problem in problems)


def test_compare_results_reports_a_missing_match(played):
    problems = compare_results(played.results, played.results[1:])
    assert any("first run only" in problem for problem in problems)


# ---------------------------------------------------------------------------
# Policy failures are loud
# ---------------------------------------------------------------------------


def test_an_illegal_action_raises_rather_than_being_replaced(bank):
    spec, policies = _broken_spec(IllegalActionPolicy(), bank)
    with pytest.raises(PolicyFailure) as caught:
        play_match(spec, bank=bank, policies=policies)
    assert caught.value.category == ERROR_ILLEGAL_ACTION
    assert "illegal action" in str(caught.value)


def test_a_raising_policy_is_classified_and_chained(bank):
    spec, policies = _broken_spec(ExplodingPolicy(), bank)
    with pytest.raises(PolicyFailure) as caught:
        play_match(spec, bank=bank, policies=policies)
    assert caught.value.category == ERROR_POLICY_EXCEPTION
    assert isinstance(caught.value.__cause__, ZeroDivisionError)


@pytest.mark.parametrize("policy_class", [WrongSeedPolicy, NonResultPolicy])
def test_a_contract_violation_is_classified(bank, policy_class):
    spec, policies = _broken_spec(policy_class(), bank)
    with pytest.raises(PolicyFailure) as caught:
        play_match(spec, bank=bank, policies=policies)
    assert caught.value.category == ERROR_CONTRACT_VIOLATION


def test_quarantine_records_the_failure_without_scoring_it(bank):
    spec, policies = _broken_spec(IllegalActionPolicy(), bank)
    row = play_match(
        spec, bank=bank, policies=policies, on_policy_error=ON_POLICY_ERROR_QUARANTINE
    )
    assert row.candidate_result == RESULT_ERROR
    assert row.candidate_score is None
    assert row.winner is None and row.draw is False
    assert row.terminal_reason == TERMINAL_POLICY_ERROR
    assert row.policy_error_category == ERROR_ILLEGAL_ACTION
    assert row.policy_error_policy == "broken_illegal@1.0.0"
    assert row.policy_error_ply == 0
    assert not row.scored


def test_a_quarantined_row_still_serialises(bank):
    spec, policies = _broken_spec(ExplodingPolicy(), bank)
    row = play_match(
        spec, bank=bank, policies=policies, on_policy_error=ON_POLICY_ERROR_QUARANTINE
    )
    rebuilt = MatchResult.from_dict(json.loads(json.dumps(row.to_dict())))
    assert rebuilt.errored and rebuilt.policy_error == row.policy_error


def test_a_quarantined_row_is_not_replayable(bank):
    spec, policies = _broken_spec(ExplodingPolicy(), bank)
    row = play_match(
        spec, bank=bank, policies=policies, on_policy_error=ON_POLICY_ERROR_QUARANTINE
    )
    assert any("unfinished" in problem for problem in replay_stored_match(row))


def test_a_scored_row_cannot_be_built_without_a_score(played):
    payload = played.results[0].to_dict()
    payload["candidate_score"] = None
    with pytest.raises(MatchRunnerError, match="no score and no error"):
        MatchResult.from_dict(payload)


def test_an_errored_row_cannot_carry_a_score(bank):
    spec, policies = _broken_spec(ExplodingPolicy(), bank)
    row = play_match(
        spec, bank=bank, policies=policies, on_policy_error=ON_POLICY_ERROR_QUARANTINE
    )
    payload = {**row.to_dict(), "candidate_score": 1.0}
    with pytest.raises(MatchRunnerError, match="must not carry a score"):
        MatchResult.from_dict(payload)


def test_an_unknown_error_mode_is_rejected(bank, schedule):
    with pytest.raises(MatchRunnerError, match="unknown on_policy_error"):
        play_match(schedule.matches[0], bank=bank, on_policy_error="ignore")


def test_a_run_counts_illegal_actions_separately(bank):
    spec, policies = _broken_spec(IllegalActionPolicy(), bank)
    run = run_schedule(
        [spec, sibling_match(spec)],
        bank,
        policies=policies,
        worker_count=1,
        on_policy_error=ON_POLICY_ERROR_QUARANTINE,
    )
    assert run.policy_errors == 2
    assert run.illegal_policy_actions == 2


# ---------------------------------------------------------------------------
# Runner plumbing
# ---------------------------------------------------------------------------


def test_the_runner_needs_exactly_one_position_source(bank, schedule):
    spec = schedule.matches[0]
    with pytest.raises(MatchRunnerError, match="exactly one"):
        play_match(spec, bank=bank, setups=(range(40), range(40)))
    with pytest.raises(MatchRunnerError, match="exactly one"):
        play_match(spec)


def test_a_mixed_rules_schedule_is_rejected(bank):
    candidate, opponent = policy_ref("random_legal"), policy_ref("basic_heuristic")
    evaluation = build_paired_schedule(
        candidate, opponent, [0], setup_bank_version=bank.bank_version, rules=EVALUATION_RULES
    )
    training = build_paired_schedule(
        candidate, opponent, [1], setup_bank_version=bank.bank_version, rules=TRAINING_RULES
    )
    mixed = schedule_matches(evaluation) + schedule_matches(training)
    with pytest.raises(MatchRunnerError, match="one rules configuration"):
        run_schedule(mixed, bank, worker_count=1)


def test_an_empty_schedule_is_rejected(bank):
    with pytest.raises(MatchRunnerError, match="empty schedule"):
        run_schedule([], bank, worker_count=1)


def test_resolve_policies_rejects_a_version_mismatch(bank):
    """A stored schedule must not silently replay against a re-versioned policy."""
    catalogued = policy_ref("basic_heuristic")
    stale = PolicyRef(catalogued.policy_id, "0.0.1-stale")
    spec = MatchSpec(
        candidate=stale,
        opponent=policy_ref("random_legal"),
        setup_pair_id=0,
        candidate_color=RED,
        setup_bank_version=bank.bank_version,
    )
    with pytest.raises(MatchRunnerError, match="cannot be replayed"):
        resolve_policies(spec)


def test_run_summary_reports_its_own_totals(played, schedule):
    payload = played.summary_dict()
    assert payload["matches_run"] == len(schedule.matches)
    assert payload["paired_units_run"] == len(schedule.paired_unit_ids)
    assert payload["policy_errors"] == 0
    assert payload["total_plies"] == sum(row.plies for row in played.results)
    assert payload["schedule_digest"] == schedule.digest


def test_results_are_returned_sorted_by_match_id(played):
    identifiers = [row.match_id for row in played.results]
    assert identifiers == sorted(identifiers)


def test_invariants_can_be_checked_every_ply(bank, schedule):
    """Expensive, so it is off by default -- but it must work when asked."""
    row = play_match(schedule.matches[0], bank=bank, verify_invariants=True)
    assert row.plies > 0


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def test_a_matchup_schedule_pairs_every_setup(bank):
    schedule = build_matchup_schedule("random_legal", "basic_heuristic", 5)
    assert len(schedule) == 10
    assert len(schedule.paired_unit_ids) == 5
    assert schedule.validate(bank) == []


def test_a_schedule_round_trips_and_verifies_its_digest():
    schedule = build_matchup_schedule("random_legal", "basic_heuristic", 4)
    rebuilt = EvaluationSchedule.from_json(schedule.to_json())
    assert rebuilt.digest == schedule.digest
    assert [m.match_id for m in rebuilt.matches] == [m.match_id for m in schedule.matches]


def test_a_removed_match_breaks_the_stored_digest():
    schedule = build_matchup_schedule("random_legal", "basic_heuristic", 4)
    payload = schedule.to_dict()
    payload["matches"] = payload["matches"][:-1]
    with pytest.raises(ScheduleError, match="does not match the rebuilt schedule"):
        EvaluationSchedule.from_dict(payload)


def test_shuffling_a_schedule_changes_nothing_that_matters():
    schedule = build_matchup_schedule("random_legal", "basic_heuristic", 6)
    shuffled = schedule.shuffled(seed=17)
    assert shuffled.digest == schedule.digest
    assert [m.match_id for m in shuffled.matches] != [m.match_id for m in schedule.matches]
    assert sorted(m.match_id for m in shuffled.matches) == sorted(
        m.match_id for m in schedule.matches
    )


def test_chunking_preserves_every_match():
    schedule = build_matchup_schedule("random_legal", "basic_heuristic", 6)
    for chunk_count in (1, 2, 3, 5, 7, 12, 40):
        chunks = schedule.chunks(chunk_count)
        flattened = [match.match_id for chunk in chunks for match in chunk]
        assert sorted(flattened) == sorted(m.match_id for m in schedule.matches)


def test_limiting_a_schedule_keeps_whole_paired_units():
    schedule = build_matchup_schedule("random_legal", "basic_heuristic", 6)
    limited = schedule.limited(7)
    assert len(limited) == 6
    assert len(limited.paired_unit_ids) == 3
    counts = {unit_id: 0 for unit_id in limited.paired_unit_ids}
    for match in limited.matches:
        counts[match.paired_unit_id] += 1
    assert set(counts.values()) == {2}


def test_a_league_schedules_each_unordered_pair_once():
    schedule = build_league_schedule(["random_legal", "basic_heuristic", "tactical_rule_based"], 2)
    assert len(schedule.matchups) == 3
    assert len(schedule) == 3 * 2 * 2


def test_a_gauntlet_covers_every_opponent():
    schedule = build_gauntlet_schedule(
        "strategic_rule_based", ["random_legal", "basic_heuristic"], 2
    )
    assert len(schedule.matchups) == 2
    # Pinned deliberately: a schedule must name the version it was built
    # against, so a policy re-versioned without recalibration breaks here rather
    # than silently playing games recorded under the old identifier. Bumped to
    # 1.1.0 by the Phase 4 Agent 4 exposure recalibration.
    assert all(
        candidate == "strategic_rule_based@1.1.0" for candidate, _ in schedule.matchups
    )


def test_a_gauntlet_rejects_the_candidate_as_its_own_opponent():
    with pytest.raises(ScheduleError, match="also appears"):
        build_gauntlet_schedule("random_legal", ["random_legal"], 2)


def test_the_ladder_schedule_covers_the_four_tiers():
    schedule = build_ladder_schedule(2)
    assert len(schedule.policy_tokens) == 4
    assert len(schedule.matchups) == 6


def test_an_unknown_policy_is_rejected():
    with pytest.raises(ScheduleError, match="unknown policy_id"):
        build_matchup_schedule("no_such_policy", "random_legal", 2)


def test_a_duplicate_league_entry_is_rejected():
    with pytest.raises(ScheduleError, match="duplicate"):
        build_league_schedule(["random_legal", "random_legal"], 2)


def test_validation_catches_a_bank_version_mismatch(bank):
    schedule = build_matchup_schedule(
        "random_legal", "basic_heuristic", 2, setup_bank_version="other_bank_v9"
    )
    problems = schedule.validate(bank)
    assert any("does not match the schedule" in problem for problem in problems)
    with pytest.raises(ScheduleError):
        require_valid_schedule(schedule, bank)


def test_validation_catches_a_setup_pair_outside_the_bank(bank):
    schedule = build_matchup_schedule(
        "random_legal", "basic_heuristic", [0, BANK_SIZE + 50]
    )
    problems = schedule.validate(bank)
    assert any("missing from the bank" in problem for problem in problems)


def test_merging_schedules_rejects_a_duplicate_match():
    first = build_matchup_schedule("random_legal", "basic_heuristic", 2)
    with pytest.raises(ScheduleError, match="more than one schedule"):
        merge_schedules([first, first], name="doubled")


def test_merging_schedules_rejects_mixed_rules():
    first = build_matchup_schedule("random_legal", "basic_heuristic", 2)
    second = build_matchup_schedule(
        "random_legal", "tactical_rule_based", 2, rules=TRAINING_RULES
    )
    with pytest.raises(ScheduleError, match="different rules"):
        merge_schedules([first, second], name="mixed")


def test_the_fingerprint_separates_differently_bound_schedules():
    first = build_matchup_schedule("random_legal", "basic_heuristic", 2)
    second = build_matchup_schedule(
        "random_legal", "basic_heuristic", 2, setup_bank_version="other_bank_v9"
    )
    # Different bank versions change every match_id, so both digests move; the
    # fingerprint additionally records the binding itself.
    assert schedule_fingerprint(first) != schedule_fingerprint(second)
    assert first.digest != second.digest


def test_every_catalogued_policy_can_be_scheduled_and_played(bank):
    """A policy absent from this loop would be absent from any league."""
    opponent = "basic_heuristic"
    for policy_id in ALL_POLICY_IDS:
        if policy_id == opponent:
            continue
        schedule = build_matchup_schedule(policy_id, opponent, [0])
        assert schedule.validate(bank) == []
        row = play_match(schedule.matches[0], bank=bank)
        assert not row.errored
