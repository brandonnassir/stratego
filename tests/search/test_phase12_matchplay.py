"""Phase 12 match play: identity, balance, the driver loop and the boundary probe.

These tests do not play a search match — that needs the accepted Phase 9
inference owner and the real checkpoint, and belongs to the runner. What is
tested here is everything that decides *which* games get played and whether
the driver can be trusted to play them: the board-id grammar, the seed
streams, the cell balance, the arm definitions, the play loop's legality and
bookkeeping, and — most importantly — whether the match-time probe can
actually catch a seat that reads a hidden rank.
"""

import pytest

from stratego.engine.constants import BLUE, RED
from stratego.engine.legal_moves import legal_actions
from stratego.engine.pieces import piece_setup_slot
from stratego.engine.permutation import hidden_opponent_piece_ids
from stratego.evaluation.match_spec import EVALUATION_RULES
from stratego.search.phase12 import PROVIDER_REMAINING_COUNT, build_belief_provider
from stratego.search.phase12 import matchplay as mp
from stratego.search.phase12.contract import search_preset
from stratego.search.phase12.engine import Phase12SearchEngine

from tests.helpers import full_inventory_setup


# ---------------------------------------------------------------------------
# Fixtures and stubs
# ---------------------------------------------------------------------------


class StubSources:
    """Two deterministic legal setups, so plan tests need no setup library."""

    def __init__(self):
        self.calls = []

    def draw(self, source, library_split, color, seed):
        self.calls.append((source, library_split, color, seed))
        setup = full_inventory_setup()
        return setup if color == "red" else tuple(reversed(setup))


def stub_plan(stratum="strategic_rule", player_color="red", ordinal=0):
    setup = full_inventory_setup()
    reversed_setup = tuple(reversed(setup))
    red, blue = (
        (setup, reversed_setup) if player_color == "red" else (reversed_setup, setup)
    )
    return mp.MatchGamePlan(
        board_id=mp.board_id(stratum, "p10d", player_color, ordinal),
        stratum=stratum,
        setup_source="p10d",
        player_color=player_color,
        opponent_color="blue" if player_color == "red" else "red",
        ordinal=ordinal,
        cell_index=0,
        match_seed=12345,
        red_setup=red,
        blue_setup=blue,
    )


class FirstLegalSeat:
    """A public-only stub seat: always the lowest legal action id."""

    arm = mp.ARM_DIRECT
    kind = "direct"

    def decide(self, state, legal, spec, plan):
        return int(min(legal)), {
            "ply": int(state.total_moves),
            "seconds": 0.0,
            "legal_actions": len(legal),
            "move_changed": None,
            "c1_forwards": 1,
            "unique_worlds": None,
            "candidates": None,
        }


class IllegalSeat(FirstLegalSeat):
    def decide(self, state, legal, spec, plan):
        action, record = super().decide(state, legal, spec, plan)
        return max(legal) + 10_000, record


class LeakySeat(FirstLegalSeat):
    """A seat that reads the opponent's hidden ranks. The probe must catch it."""

    def decide(self, state, legal, spec, plan):
        hidden = sorted(hidden_opponent_piece_ids(state, state.acting_player))
        fingerprint = sum(
            (index + 1) * state.pieces[piece_id].true_type
            for index, piece_id in enumerate(hidden)
        )
        ordered = sorted(legal)
        return int(ordered[fingerprint % len(ordered)]), {
            "ply": int(state.total_moves),
            "seconds": 0.0,
            "legal_actions": len(legal),
            "move_changed": None,
            "c1_forwards": 1,
            "unique_worlds": None,
            "candidates": None,
        }


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_board_id_round_trips_through_its_grammar():
    identifier = mp.board_id("tactical_rule", "neutral", "blue", 5)
    assert identifier.startswith("phase12_match_v1|")
    assert mp.parse_board_id(identifier) == {
        "master_seed": mp.MATCH_MASTER_SEED,
        "stratum": "tactical_rule",
        "setup_source": "neutral",
        "player_color": "blue",
        "ordinal": 5,
    }


@pytest.mark.parametrize(
    "arguments",
    [
        ("no_such_stratum", "p10d", "red", 0),
        ("tactical_rule", "no_such_source", "red", 0),
        ("tactical_rule", "p10d", "green", 0),
        ("tactical_rule", "p10d", "red", -1),
        ("tactical_rule", "p10d", "red", 100),
    ],
)
def test_board_id_refuses_an_identity_outside_the_match_set(arguments):
    with pytest.raises(mp.Phase12MatchError):
        mp.board_id(*arguments)


def test_parse_refuses_a_foreign_identifier():
    with pytest.raises(mp.Phase12MatchError):
        mp.parse_board_id("phase12_diag_v1|ms=2026082002|st=phase9_selfplay|src=p10d|obs=red|g=0000")


def test_the_match_master_seed_is_not_the_diagnostic_master_seed():
    from stratego.search.phase12 import positions as diag

    assert mp.MATCH_MASTER_SEED != diag.DIAGNOSTIC_MASTER_SEED


def test_search_seed_depends_on_board_and_ply_and_nothing_else():
    board = mp.board_id("scout_rush", "p10d", "red", 1)
    other = mp.board_id("scout_rush", "p10d", "blue", 1)
    assert mp.search_seed_for(board, 40) == mp.search_seed_for(board, 40)
    assert mp.search_seed_for(board, 40) != mp.search_seed_for(board, 41)
    assert mp.search_seed_for(board, 40) != mp.search_seed_for(other, 40)
    assert 0 <= mp.search_seed_for(board, 40) < 2**63


def test_the_four_seed_streams_are_independent():
    board = mp.board_id("strategic_rule", "neutral", "red", 0)
    values = {
        mp.match_seed_value(domain, board)
        for domain in (
            mp.DOMAIN_PLAYER_SETUP,
            mp.DOMAIN_OPPONENT_SETUP,
            mp.DOMAIN_MATCH,
            mp.DOMAIN_SEARCH,
            mp.DOMAIN_PROBE,
        )
    }
    assert len(values) == 5


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


def test_the_match_set_is_balanced_over_opponents_sources_and_colours():
    plans = mp.match_plans(StubSources(), games_per_opponent=8)
    assert len(plans) == 8 * len(mp.MATCH_STRATA)
    for stratum in mp.MATCH_STRATA:
        cell = [plan for plan in plans if plan.stratum == stratum]
        assert len(cell) == 8
        assert sum(1 for plan in cell if plan.player_color == "red") == 4
        assert sum(1 for plan in cell if plan.setup_source == "p10d") == 4
    assert len({plan.board_id for plan in plans}) == len(plans)


def test_the_match_set_is_ordinal_major_so_a_truncated_run_stays_balanced():
    plans = mp.match_plans(StubSources(), games_per_opponent=8)
    first_half = plans[: len(plans) // 2]
    for stratum in mp.MATCH_STRATA:
        cell = [plan for plan in first_half if plan.stratum == stratum]
        assert len(cell) == 4
        assert sum(1 for plan in cell if plan.player_color == "red") == 2


def test_an_unbalanced_game_count_is_refused():
    with pytest.raises(mp.Phase12MatchError):
        mp.match_plans(StubSources(), games_per_opponent=6)


def test_a_plan_is_a_pure_function_of_its_identity():
    first = mp.match_plan("tactical_rule", "neutral", "blue", 1, StubSources())
    second = mp.match_plan("tactical_rule", "neutral", "blue", 1, StubSources())
    assert first == second


def test_the_player_always_draws_from_the_accepted_production_source():
    sources = StubSources()
    mp.match_plan("scout_rush", "neutral", "red", 0, sources)
    drawn = {(source, color) for source, _, color, _ in sources.calls}
    assert ("p10d", "red") in drawn
    assert ("neutral", "blue") in drawn


def test_the_match_identity_names_an_arm_independent_player():
    plan = stub_plan()
    reference, _ = mp.opponent_seat(plan, {})
    spec = mp.build_spec(plan, reference)
    assert spec.candidate == mp.player_ref()
    assert mp.MATCH_VERSION in spec.candidate.token
    assert all(
        arm.arm_id not in spec.match_id and arm.arm_id not in spec.candidate.token
        for arm in mp.ALL_ARMS
    )
    assert spec.rules == EVALUATION_RULES


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def test_the_instructed_arms_are_present_and_typed():
    assert [arm.arm_id for arm in mp.PRODUCTION_ARMS] == [
        "direct_c1",
        "search_remaining_count",
        "search_original_phase11",
        "search_agent1c",
    ]
    assert mp.ARM_ORACLE.diagnostic_only is True
    assert all(arm.diagnostic_only is False for arm in mp.PRODUCTION_ARMS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "search", "provider_id": None},
        {"kind": "direct", "provider_id": "agent1c"},
        {"kind": "nonsense", "provider_id": None},
    ],
)
def test_an_incoherent_arm_is_refused(kwargs):
    with pytest.raises(mp.Phase12MatchError):
        mp.MatchArm("x", kwargs["kind"], "x", kwargs["provider_id"])


def test_a_search_seat_refuses_an_engine_carrying_another_provider(random_c1):
    provider = build_belief_provider(PROVIDER_REMAINING_COUNT)
    engine = Phase12SearchEngine(random_c1, provider, search_preset("TINY"))
    with pytest.raises(mp.Phase12MatchError):
        mp.SearchSeat(mp.ARM_AGENT1C, engine)


# ---------------------------------------------------------------------------
# Opponents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stratum,policy_id",
    [
        ("strategic_rule", "strategic_rule_based"),
        ("tactical_rule", "tactical_rule_based"),
        ("scout_rush", "stress_scout_rush"),
    ],
)
def test_the_opponent_seat_is_the_accepted_stratum_policy(stratum, policy_id):
    reference, policy = mp.opponent_seat(stub_plan(stratum=stratum), {})
    assert reference.policy_id == policy_id
    assert policy.frozen_policy_seed == 12345


def test_the_phase9_opponent_seat_needs_the_accepted_owner():
    reference, _ = mp.opponent_seat(
        stub_plan(stratum="phase9_selfplay"), {"phase9": object()}
    )
    assert reference.policy_id == mp.PHASE9_OPPONENT_POLICY_ID


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


def test_the_driver_plays_a_whole_legal_game_and_records_it():
    plan = stub_plan()
    record = mp.play_arm_game(plan, FirstLegalSeat(), {})
    assert record.board_id == plan.board_id
    assert record.arm_id == mp.ARM_DIRECT.arm_id
    assert record.outcome in ("win", "draw", "loss")
    assert record.effective_score in (0.0, 0.5, 1.0)
    assert record.outcome == mp.outcome_of(record.effective_score)
    assert record.plies > 0
    # The player moves on its own plies only, so it decides at most half of them.
    assert 0 < record.player_decisions <= record.plies
    assert len(record.moves) == record.player_decisions
    assert record.terminal_reason
    row = record.row()
    assert row["move_change_rate"] is None  # a direct seat reports no comparison
    assert row["match_id"].startswith("m-")


def test_the_driver_refuses_an_illegal_action():
    with pytest.raises(mp.Phase12MatchError, match="illegal action"):
        mp.play_arm_game(stub_plan(), IllegalSeat(), {})


def test_the_same_board_replays_identically_for_the_same_seat():
    plan = stub_plan(stratum="tactical_rule")
    first = mp.play_arm_game(plan, FirstLegalSeat(), {})
    second = mp.play_arm_game(plan, FirstLegalSeat(), {})
    assert (first.outcome, first.plies, first.terminal_reason) == (
        second.outcome,
        second.plies,
        second.terminal_reason,
    )


# ---------------------------------------------------------------------------
# The match-time boundary probe
# ---------------------------------------------------------------------------


def probe_once(seat, plan, state, **kwargs):
    probe = mp.SeatProbe(interval=1, budget=4, **kwargs)
    reference, _ = mp.opponent_seat(plan, {})
    spec = mp.build_spec(plan, reference)
    legal = legal_actions(state)
    action, record = seat.decide(state, legal, spec, plan)
    probe.run(seat, state, legal, spec, plan, action, record)
    return probe


def test_the_probe_catches_a_seat_that_reads_a_hidden_rank(midgame_state):
    plan = stub_plan(player_color="red" if midgame_state.acting_player == RED else "blue")
    probe = probe_once(LeakySeat(), plan, midgame_state)
    assert probe.permutation_changed == 1
    assert not probe.summary()["passed"]
    assert probe.failures[0]["check"] == "permutation_invariance"


def test_a_seat_that_is_meant_to_read_truth_counts_as_sensitivity_not_failure(
    midgame_state,
):
    """The oracle arm's reading: changing the answer is the expected outcome."""
    plan = stub_plan(player_color="red" if midgame_state.acting_player == RED else "blue")
    probe = probe_once(
        LeakySeat(), plan, midgame_state, expects_hidden_truth=True
    )
    assert probe.permutation_sensitive == 1
    assert probe.summary()["passed"]
    assert probe.summary()["expects_hidden_truth"] is True


def test_the_probe_passes_a_public_only_seat(midgame_state):
    plan = stub_plan(player_color="red" if midgame_state.acting_player == RED else "blue")
    probe = probe_once(FirstLegalSeat(), plan, midgame_state)
    assert probe.permutation_checks == 1
    assert probe.summary()["passed"]


def test_the_probe_never_spends_its_first_decision_on_the_opening():
    probe = mp.SeatProbe(interval=4, budget=2)
    assert probe.due(0) is False
    assert probe.due(4) is True
    probe.budget = 0
    assert probe.due(4) is False


# ---------------------------------------------------------------------------
# A search seat decides
# ---------------------------------------------------------------------------


def test_a_search_seat_returns_a_legal_action_and_a_search_record(
    random_c1, midgame_state
):
    plan = stub_plan(player_color="red" if midgame_state.acting_player == RED else "blue")
    provider = build_belief_provider(PROVIDER_REMAINING_COUNT)
    engine = Phase12SearchEngine(random_c1, provider, search_preset("TINY"))
    seat = mp.SearchSeat(mp.ARM_COUNT, engine)
    reference, _ = mp.opponent_seat(plan, {})
    spec = mp.build_spec(plan, reference)
    legal = legal_actions(midgame_state)

    action, record = seat.decide(midgame_state, legal, spec, plan)
    assert action in legal
    assert record["move_changed"] in (True, False)
    assert record["direct_action_id"] in legal
    assert record["unique_worlds"] >= 1
    assert record["c1_forwards"] > 1
    assert record["seconds"] > 0.0

    # The seat's seed is the board-and-ply seed, so the same decision repeats.
    again, _ = seat.decide(midgame_state, legal, spec, plan)
    assert again == action
