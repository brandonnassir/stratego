"""Phase 12 working player: modes, time cap, fallback, seat, candidate record.

Hermetic like the rest of the Phase 12 tests: a randomly initialized C1 and
a randomly initialized Agent 1C-shaped head stand in for the accepted
checkpoints, because every property tested here — mode selection, the
structural oracle refusal, the timeout and error fallbacks, legality, seat
bookkeeping, the artifact schema — must hold for any weights. The runner
script exercises the real digests and the real latency profile.
"""

import math
import time
from types import SimpleNamespace

import pytest

from stratego.belief.phase11b.heads import CANDIDATE_1C, build_candidate
from stratego.belief.phase11b.interface import Phase11BBeliefModel
from stratego.engine.legal_moves import legal_action_mask, legal_actions
from stratego.engine.observation import build_observation
from stratego.model.policy_adapter import (
    DECISION_MODE_GREEDY,
    prepare_legality,
    select_action,
)
from stratego.model.tokenization import observation_batch_from_numpy, observation_to_tokens
from stratego.search.phase12 import matchplay as mp
from stratego.search.phase12 import player as pl
from stratego.search.phase12.contract import (
    PROVIDER_AGENT1C,
    SEARCH_PRESETS,
    Phase12SearchError,
    Phase12SearchTimeout,
)
from stratego.search.phase12.engine import Phase12SearchEngine
from stratego.search.phase12.providers import (
    AdapterNeuralBeliefProvider,
    OracleBeliefProvider,
    RemainingCountBeliefProvider,
)

from tests.helpers import full_inventory_setup


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def agent1c_provider(random_c1):
    head = build_candidate(CANDIDATE_1C, random_c1)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    model = Phase11BBeliefModel(
        random_c1, head, candidate_id=CANDIDATE_1C, device="cpu"
    )
    return AdapterNeuralBeliefProvider(
        model, provider_id=PROVIDER_AGENT1C, identity={"weights": "random"}
    )


@pytest.fixture(scope="module")
def player(random_c1, agent1c_provider):
    return pl.Phase12SearchPlayer(
        random_c1, agent1c_provider, model_identity={"weights": "random"}
    )


def accepted_direct_action(model, state):
    """The accepted greedy-adapter action, computed independently."""
    import torch

    legal = legal_actions(state)
    legality = prepare_legality(legal, legal_action_mask(state, legal), state.acting_player)
    batch = observation_batch_from_numpy(
        [build_observation(state, state.acting_player)],
        dtype=torch.float32,
        device="cpu",
    )
    with torch.no_grad():
        outputs = model(observation_to_tokens(batch))
    row = outputs.policy_logits.detach().to("cpu", torch.float32)[0]
    return int(
        select_action(row, legality, decision_mode=DECISION_MODE_GREEDY).absolute_action_id
    )


class StubEngine:
    """Stands in for one mode's engine to force a failure path."""

    def __init__(self, outcome):
        self.config = SEARCH_PRESETS["TINY"]
        self.outcome = outcome

    def choose_action(self, state, *, seed, deadline=None):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def stub_decision(selected, direct, score=0.5):
    return SimpleNamespace(
        selected_action_id=selected,
        direct_action_id=direct,
        move_changed=selected != direct,
        candidates=(SimpleNamespace(score=score),),
    )


# ---------------------------------------------------------------------------
# Construction: the production stack is structural
# ---------------------------------------------------------------------------


def test_player_refuses_every_non_production_provider(random_c1):
    with pytest.raises(Phase12SearchError):
        pl.Phase12SearchPlayer(random_c1, RemainingCountBeliefProvider())
    with pytest.raises(Phase12SearchError):
        pl.Phase12SearchPlayer(random_c1, OracleBeliefProvider(offline_diagnostic=True))
    assert pl.ORACLE_AVAILABLE_IN_PRODUCTION is False


def test_oracle_is_not_a_mode_and_unknown_modes_are_refused(player):
    with pytest.raises(Phase12SearchError):
        pl.check_mode("oracle")
    with pytest.raises(Phase12SearchError):
        player.set_mode("LARGE")
    with pytest.raises(Phase12SearchError):
        player.decide(None, mode="oracle")
    assert player.mode == pl.DEFAULT_MODE == pl.MODE_TINY


def test_mode_switching_is_visible_in_status(player):
    previous = player.set_mode("small")
    assert previous == "tiny"
    status = player.status()
    assert status["mode"] == "small"
    assert "SMALL" in status["budget"]
    assert status["time_cap_seconds"] == pl.MODE_TIME_CAP_SECONDS["small"]
    assert status["oracle_available_in_production"] is False
    player.set_mode("tiny")
    described = player.describe()
    assert described["default_mode"] == "tiny"
    assert set(described["presets"]) == set(pl.SEARCH_MODES)
    assert described["fallback_policy"] == pl.FALLBACK_POLICY


def test_medium_is_the_designated_maximum_strength_mode(player):
    assert pl.MAX_STRENGTH_MODE == pl.MODE_MEDIUM == "medium"
    assert pl.MAX_STRENGTH_MODE in pl.SEARCH_MODES
    assert player.describe()["max_strength_mode"] == "medium"
    assert player.status()["max_strength_mode"] == "medium"
    # The designation does not disturb the frozen production default.
    assert pl.DEFAULT_MODE == pl.MODE_TINY


def test_time_caps_validate(random_c1, agent1c_provider):
    with pytest.raises(Phase12SearchError):
        pl.Phase12SearchPlayer(random_c1, agent1c_provider, time_caps={"direct": 1.0})
    with pytest.raises(Phase12SearchError):
        pl.Phase12SearchPlayer(random_c1, agent1c_provider, time_caps={"tiny": 0.0})


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def test_direct_mode_is_the_accepted_adapter_decision(player, random_c1, midgame_state):
    decision = player.decide(midgame_state, mode="direct")
    assert decision.action_id == accepted_direct_action(random_c1, midgame_state)
    assert decision.action_id in set(legal_actions(midgame_state))
    assert decision.used_search is False
    assert decision.fallback_reason is None
    assert decision.preset_id is None and decision.time_cap_seconds is None
    again = player.decide(midgame_state, mode="direct")
    assert again.action_id == decision.action_id


def test_tiny_search_returns_the_engines_own_decision(
    player, random_c1, agent1c_provider, midgame_state
):
    engine = Phase12SearchEngine(random_c1, agent1c_provider, SEARCH_PRESETS["TINY"])
    reference = engine.choose_action(midgame_state, seed=77)
    decision = player.decide(midgame_state, seed=77)
    assert decision.used_search is True
    assert decision.fallback_reason is None
    assert decision.action_id == int(reference.selected_action_id)
    assert decision.direct_action_id == int(reference.direct_action_id)
    assert decision.move_changed == bool(reference.move_changed)
    assert decision.search.c1_forwards == reference.c1_forwards
    assert decision.action_id in set(legal_actions(midgame_state))
    assert decision.preset_id == "TINY"
    assert decision.time_cap_seconds == pl.MODE_TIME_CAP_SECONDS["tiny"]


def test_default_seed_is_deterministic_in_game_and_ply(player, midgame_state):
    first = player.decide(midgame_state)
    second = player.decide(midgame_state)
    assert first.seed == second.seed == pl.player_seed_for(
        midgame_state.game_id, midgame_state.total_moves
    )
    assert first.action_id == second.action_id
    assert pl.player_seed_for("g", 1) != pl.player_seed_for("g", 2)


# ---------------------------------------------------------------------------
# The engine deadline
# ---------------------------------------------------------------------------


def test_deadline_none_is_bit_identical_and_past_deadline_raises(
    random_c1, agent1c_provider, midgame_state
):
    engine = Phase12SearchEngine(random_c1, agent1c_provider, SEARCH_PRESETS["TINY"])
    bare = engine.choose_action(midgame_state, seed=11)
    roomy = engine.choose_action(
        midgame_state, seed=11, deadline=time.perf_counter() + 60.0
    )
    assert bare.selected_action_id == roomy.selected_action_id
    assert bare.c1_forwards == roomy.c1_forwards
    assert [c.score for c in bare.candidates] == [c.score for c in roomy.candidates]
    with pytest.raises(Phase12SearchTimeout):
        engine.choose_action(midgame_state, seed=11, deadline=time.perf_counter() - 1.0)


# ---------------------------------------------------------------------------
# Fallbacks: every failure ends in a legal direct move
# ---------------------------------------------------------------------------


def fallback_case(player, midgame_state, monkeypatch, outcome, expected_reason):
    monkeypatch.setitem(player.engines, "tiny", StubEngine(outcome))
    before = dict(player.fallback_counts)
    decision = player.decide(midgame_state, mode="tiny")
    assert decision.used_search is False
    assert decision.fallback_reason == expected_reason
    assert decision.action_id in set(legal_actions(midgame_state))
    assert player.fallback_counts[expected_reason] == before[expected_reason] + 1
    return decision


def test_timeout_falls_back_to_direct(
    random_c1, agent1c_provider, midgame_state
):
    tight = pl.Phase12SearchPlayer(
        random_c1, agent1c_provider, time_caps={"tiny": 1e-4}
    )
    decision = tight.decide(midgame_state)
    assert decision.fallback_reason == pl.FALLBACK_TIMEOUT
    assert decision.used_search is False
    assert decision.action_id == accepted_direct_action(random_c1, midgame_state)
    assert tight.fallback_counts[pl.FALLBACK_TIMEOUT] == 1
    assert tight.status()["fallback_total"] == 1
    assert tight.fallback_events[-1]["reason"] == pl.FALLBACK_TIMEOUT


def test_search_error_falls_back(player, random_c1, midgame_state, monkeypatch):
    decision = fallback_case(
        player, midgame_state, monkeypatch,
        Phase12SearchError("forced"), pl.FALLBACK_SEARCH_ERROR,
    )
    assert decision.action_id == accepted_direct_action(random_c1, midgame_state)


def test_unexpected_error_falls_back(player, midgame_state, monkeypatch):
    fallback_case(
        player, midgame_state, monkeypatch,
        RuntimeError("forced"), pl.FALLBACK_UNEXPECTED_ERROR,
    )


def test_non_finite_score_falls_back(player, midgame_state, monkeypatch):
    legal = legal_actions(midgame_state)
    fallback_case(
        player, midgame_state, monkeypatch,
        stub_decision(int(min(legal)), int(min(legal)), score=math.nan),
        pl.FALLBACK_NON_FINITE,
    )


def test_illegal_search_action_falls_back(player, midgame_state, monkeypatch):
    fallback_case(
        player, midgame_state, monkeypatch,
        stub_decision(-1, -1, score=0.25),
        pl.FALLBACK_ILLEGAL_ACTION,
    )


def test_last_resort_never_forfeits(player, midgame_state, monkeypatch):
    monkeypatch.setitem(
        player.engines, "tiny", StubEngine(Phase12SearchError("forced"))
    )

    def broken_direct(state, legal):
        raise RuntimeError("forced direct failure")

    monkeypatch.setattr(player, "_direct_action", broken_direct)
    decision = player.decide(midgame_state, mode="tiny")
    assert decision.action_id == int(min(legal_actions(midgame_state)))
    assert decision.fallback_reason == pl.FALLBACK_DIRECT_ERROR
    assert decision.used_search is False


# ---------------------------------------------------------------------------
# The match seat
# ---------------------------------------------------------------------------


def seat_plan(player_color="red", stratum="strategic_rule"):
    setup = full_inventory_setup()
    reversed_setup = tuple(reversed(setup))
    red, blue = (
        (setup, reversed_setup) if player_color == "red" else (reversed_setup, setup)
    )
    return mp.MatchGamePlan(
        board_id=mp.board_id(stratum, "p10d", player_color, 0),
        stratum=stratum,
        setup_source="p10d",
        player_color=player_color,
        opponent_color="blue" if player_color == "red" else "red",
        ordinal=0,
        cell_index=0,
        match_seed=12345,
        red_setup=red,
        blue_setup=blue,
    )


def test_search_seat_uses_the_match_seed_stream(player, midgame_state):
    seat = pl.Phase12PlayerSeat(player, "tiny")
    plan = seat_plan()
    legal = legal_actions(midgame_state)
    action, record = seat.decide(midgame_state, legal, None, plan)
    assert action in set(legal)
    assert record["used_search"] is True
    assert record["fallback_reason"] is None
    assert record["c1_forwards"] > 1
    assert record["unique_worlds"] is not None
    expected_seed = mp.search_seed_for(plan.board_id, midgame_state.total_moves)
    reference = player.decide(midgame_state, seed=expected_seed, mode="tiny")
    assert action == reference.action_id
    assert seat.arm.arm_id == "player_search_tiny"
    assert seat.describe()["mode"] == "tiny"


def test_direct_seat_records_match_the_accepted_shape(player, midgame_state):
    seat = pl.Phase12PlayerSeat(player, "direct")
    legal = legal_actions(midgame_state)
    action, record = seat.decide(midgame_state, legal, None, seat_plan())
    assert action in set(legal)
    assert record["c1_forwards"] == 1
    assert record["unique_worlds"] is None
    assert record["move_changed"] is None
    assert record["used_search"] is False
    assert seat.arm.kind == "direct"


def test_direct_player_seat_completes_a_real_game(player):
    seat = pl.Phase12PlayerSeat(player, "direct")
    record = mp.play_arm_game(seat_plan(), seat, owners={}, keep_moves=False)
    assert record.outcome in ("win", "draw", "loss")
    assert record.player_decisions > 0
    assert record.plies > 0
    assert record.arm_id == "player_direct"


# ---------------------------------------------------------------------------
# The engineering candidate record
# ---------------------------------------------------------------------------


AGENT4_POINT = {
    "source": "reports/phase12/agent_04_summary.json",
    "preset_id": "TINY",
    "games": 64,
    "ewr": 0.640625,
    "direct_ewr": 0.5234375,
    "move_seconds_median": 0.12614,
    "move_seconds_p95": 0.13816,
    "move_seconds_max": 0.1929,
    "search_seconds_per_game": 10.247821875,
}

AGENT4_MEDIUM_POINT = {
    "source": "reports/phase12/agent_04_summary.json",
    "preset_id": "MEDIUM",
    "games": 64,
    "ewr": 0.6875,
    "move_seconds_median": 0.84625,
    "move_seconds_p95": 0.91635,
    "move_seconds_max": 0.98072,
    "search_seconds_per_game": 66.5451859375,
}


def test_candidate_record_carries_the_contract_fields():
    record = pl.build_candidate_record(
        move_model_identity={"model_state_digest": "m" * 64},
        belief_model_identity={"state_dict_digest": "b" * 64},
        agent4=AGENT4_POINT,
        agent4_medium=AGENT4_MEDIUM_POINT,
        generated_utc="2026-08-20T00:00:00Z",
        environment={"device": "cpu"},
        quick_checks={"all_passed": True},
        known_limitations=["engineering sample"],
    )
    assert record["artifact"] == pl.CANDIDATE_ARTIFACT == "phase12_search_candidate_v1"
    assert record["move_model"] == "accepted Phase 9 C1"
    assert record["belief_model"] == "Agent1C"
    assert record["search_version"] == "phase12_root_world_search_v1"
    assert record["selected_preset"] == "TINY"
    assert record["worlds"] == 8
    assert record["root_candidates"] == "<= 8"
    assert record["depth"] == 4
    assert record["beta"] == 0.1
    assert record["epsilon"] == 1e-6
    assert record["expected_latency_median"] == "0.126 s/move"
    assert record["expected_latency_p95"] == "0.138 s/move"
    assert record["Agent4_quick_EWR"] == 0.6406
    assert record["Agent4_direct_EWR"] == 0.5234
    assert record["fallback_policy"] == "direct accepted Phase 9 C1"
    assert record["time_cap_seconds"] == 0.5
    assert record["oracle_available_in_production"] is False
    assert record["phase11_final_classification"] == "FAIL"
    assert record["phase11b_selection"] == "Agent1C"
    assert record["scientific_validation_status"] == "not performed"
    assert record["move_model_identity"]["model_state_digest"] == "m" * 64
    assert record["belief_model_identity"]["state_dict_digest"] == "b" * 64
    assert len(record["candidate_config_digest"]) == 64
    assert int(record["candidate_config_digest"], 16) >= 0
    assert record["max_strength_mode"] == "medium"
    maximum = record["maximum_strength_candidate"]
    assert maximum["preset"] == "MEDIUM"
    assert maximum["worlds"] == 32 and maximum["depth"] == 8
    assert maximum["Agent4_quick_EWR"] == 0.6875
    assert maximum["ewr_lead_over_selected"] == 0.0469
    assert maximum["expected_latency_median"] == "0.846 s/move"
    assert maximum["expected_latency_p95"] == "0.916 s/move"
    assert maximum["time_cap_seconds"] == 3.5
    assert "not a validated ordering" in maximum["caveat"]
