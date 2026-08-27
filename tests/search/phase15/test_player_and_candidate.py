"""The working player, its refusals and fallbacks, and the frozen candidate."""

import json
import time

import pytest

from stratego.search.phase15.player import (
    DIAGNOSTIC_MODES,
    MODE_MAX_STRENGTH,
    MODE_P18_DIRECT,
    MODE_P24_DIRECT,
    MODE_SELECTED,
    PLAYER_MODES,
    REQUIRED_MODES,
    Phase15PlayerError,
    Phase15SearchPlayer,
)
from stratego.search.phase15.systems import build_engine


@pytest.fixture()
def player(fake_models):
    systems = {
        MODE_SELECTED: build_engine("p24_b18", fake_models, "TINY"),
        MODE_MAX_STRENGTH: build_engine("p24_b18", fake_models, "SMALL"),
    }
    return Phase15SearchPlayer(
        systems,
        fake_models,
        mode=MODE_SELECTED,
        time_caps={MODE_SELECTED: 5.0, MODE_MAX_STRENGTH: 5.0},
    )


# -- modes ------------------------------------------------------------------


def test_the_four_required_modes_exist():
    assert REQUIRED_MODES == (
        "p18_direct",
        "p24_direct",
        "selected_search",
        "maximum_strength",
    )
    for mode in REQUIRED_MODES:
        assert mode in PLAYER_MODES


def test_diagnostic_names_are_available_but_never_the_oracle():
    assert "p24_b18" in DIAGNOSTIC_MODES
    assert not [mode for mode in PLAYER_MODES if "oracle" in mode]


@pytest.mark.parametrize("mode", ["oracle", "p18_oracle", "p24_oracle"])
def test_the_oracle_is_refused_as_a_mode(mode):
    with pytest.raises(Phase15PlayerError, match="offline diagnostic"):
        Phase15SearchPlayer.check_mode(mode)


def test_an_unknown_mode_is_refused():
    with pytest.raises(Phase15PlayerError, match="unknown mode"):
        Phase15SearchPlayer.check_mode("turbo")


def test_a_player_will_not_hold_an_oracle_system(fake_models):
    bundle = build_engine("p18_oracle", fake_models, "TINY", production=False)
    with pytest.raises(Phase15PlayerError, match="carries the oracle"):
        Phase15SearchPlayer({MODE_SELECTED: bundle}, fake_models)


def test_setting_a_mode_with_no_system_is_refused(player):
    with pytest.raises(Phase15PlayerError, match="has no system"):
        player.set_mode("p18_b18")


def test_direct_modes_need_no_system(player):
    assert player.set_mode(MODE_P18_DIRECT) == MODE_P18_DIRECT
    assert player.mode_move_model(MODE_P18_DIRECT) == "p18"
    assert player.set_mode(MODE_P24_DIRECT) == MODE_P24_DIRECT
    assert player.mode_move_model(MODE_P24_DIRECT) == "p24"


# -- decisions --------------------------------------------------------------


def test_a_search_decision_is_legal_and_recorded(player, midgame_state):
    from stratego.engine.legal_moves import legal_actions

    legal = legal_actions(midgame_state)
    decision = player.decide(midgame_state, legal=legal)
    assert decision.action_id in legal
    assert decision.searched is True
    assert decision.fallback_reason is None
    assert decision.preset_id == "TINY"
    assert decision.unique_worlds is not None
    assert "phase15" in decision.log_line()


def test_a_direct_decision_takes_one_forward(player, midgame_state):
    from stratego.engine.legal_moves import legal_actions

    legal = legal_actions(midgame_state)
    decision = player.decide(midgame_state, legal=legal, mode=MODE_P18_DIRECT)
    assert decision.searched is False
    assert decision.c1_forwards == 1
    assert decision.action_id == player.direct_action(
        midgame_state, legal, move_model="p18"
    )


def test_the_same_seed_reproduces_the_decision(player, midgame_state):
    first = player.decide(midgame_state, seed=17)
    second = player.decide(midgame_state, seed=17)
    assert first.action_id == second.action_id


def test_each_mode_falls_back_to_its_own_move_model(fake_models, midgame_state):
    from stratego.engine.legal_moves import legal_actions

    legal = legal_actions(midgame_state)
    for pairing_id, expected in (("p18_b24", "p18"), ("p24_b18", "p24")):
        systems = {MODE_SELECTED: build_engine(pairing_id, fake_models, "TINY")}
        subject = Phase15SearchPlayer(systems, fake_models, mode=MODE_SELECTED)
        reference = subject.direct_action(midgame_state, legal, move_model=expected)
        broken = subject.decide(midgame_state, legal=legal, force_error=True)
        assert broken.fallback_reason == "search_error"
        assert broken.action_id == reference
        assert broken.move_model == expected


def test_an_expired_deadline_falls_back_to_the_direct_move(player, midgame_state):
    from stratego.engine.legal_moves import legal_actions

    legal = legal_actions(midgame_state)
    decision = player.decide(
        midgame_state, legal=legal, deadline_override=time.perf_counter() - 1.0
    )
    assert decision.fallback_reason == "timeout"
    assert decision.action_id == player.direct_action(midgame_state, legal, move_model="p24")
    assert decision.searched is False


def test_a_fallback_is_counted_and_reported(player, midgame_state):
    player.decide(midgame_state, force_error=True)
    status = player.status()
    assert status["fallbacks"]["search_error"] == 1
    assert status["fallback_rate"] > 0


def test_a_decision_with_no_legal_action_is_refused(player, midgame_state):
    with pytest.raises(Phase15PlayerError, match="no legal action"):
        player.decide(midgame_state, legal=[])


def test_describe_names_every_mode_and_the_oracle_fact(player):
    report = player.describe()
    assert set(report["modes"]) == set(PLAYER_MODES)
    assert report["oracle_available_in_production"] is False
    assert report["modes"][MODE_SELECTED]["available"] is True
    assert report["modes"]["p18_b18"]["available"] is False


def test_the_player_seat_decides_through_the_player(fake_models, midgame_state):
    from stratego.search.phase15.player import Phase15PlayerSeat

    systems = {MODE_SELECTED: build_engine("p18_b18", fake_models, "TINY")}
    subject = Phase15SearchPlayer(systems, fake_models, mode=MODE_SELECTED)
    seat = Phase15PlayerSeat(subject, MODE_SELECTED)
    assert seat.arm_id == "player|selected_search"
    assert seat.pairing.pairing_id == "p18_b18"


# -- the frozen candidate ---------------------------------------------------


def _record(fake_models, tmp_path):
    from stratego.search.phase15.candidate import build_candidate_record

    return build_candidate_record(
        selected_pairing="p24_b18",
        selected_preset="TINY",
        maximum_strength_preset="MEDIUM",
        models=fake_models,
        time_caps={"selected_search": 0.5, "maximum_strength": 3.5},
        latency={"selected_preset": {"p95_seconds_per_move": 0.14}},
        match_manifest_digest="a" * 64,
        position_manifest_digest="b" * 64,
        gate={"passed": True},
        stage_a={},
        stage_b={},
        stage_c={},
        system_matrix={},
        known_limitations=["a compact engineering pack"],
    )


def test_the_candidate_records_both_identities_and_the_budget(fake_models, tmp_path):
    record = _record(fake_models, tmp_path)
    assert record["selected_system"]["pairing_id"] == "p24_b18"
    assert record["move_model"]["logical_identity"] == "P24"
    assert record["belief_model"]["provider_id"] == "b18"
    assert record["belief_model"]["prefix_backbone"] == "p18"
    assert record["search"]["worlds"] == 8
    assert record["search"]["rollout_depth"] == 4
    assert record["search"]["beta"] == 0.1
    assert record["direct_fallback"]["identity"] == "P24"
    assert record["oracle_available_in_production"] is False
    assert record["scientific_validation_status"] == "not performed"
    assert record["known_limitations"]


def test_the_candidate_refuses_a_direct_or_oracle_selection(fake_models):
    from stratego.search.phase15.candidate import (
        Phase15CandidateError,
        build_candidate_record,
    )

    for pairing_id in ("p18_direct", "p24_oracle"):
        with pytest.raises(Phase15CandidateError, match="not a deployable"):
            build_candidate_record(
                selected_pairing=pairing_id,
                selected_preset="TINY",
                maximum_strength_preset="TINY",
                models=fake_models,
                time_caps={},
                latency={},
                match_manifest_digest="a" * 64,
                position_manifest_digest="b" * 64,
                gate={},
                stage_a={},
                stage_b={},
                stage_c={},
                system_matrix={},
            )


def test_the_candidate_round_trips_through_disk(fake_models, tmp_path):
    from stratego.search.phase15.candidate import load_candidate, write_candidate

    path = write_candidate(_record(fake_models, tmp_path), tmp_path / "candidate.json")
    assert load_candidate(path)["selected_system"]["pairing_id"] == "p24_b18"


def test_reading_back_refuses_a_misstated_status(fake_models, tmp_path):
    from stratego.search.phase15.candidate import (
        Phase15CandidateError,
        load_candidate,
        write_candidate,
    )

    record = _record(fake_models, tmp_path)
    record["oracle_available_in_production"] = True
    path = write_candidate(record, tmp_path / "bad.json")
    with pytest.raises(Phase15CandidateError, match="oracle_available_in_production"):
        load_candidate(path)

    record["oracle_available_in_production"] = False
    record["scientific_validation_status"] = "validated"
    path = write_candidate(record, tmp_path / "bad2.json")
    with pytest.raises(Phase15CandidateError, match="misstates its validation status"):
        load_candidate(path)


def test_reading_back_refuses_a_foreign_document(tmp_path):
    from stratego.search.phase15.candidate import Phase15CandidateError, load_candidate

    path = tmp_path / "other.json"
    path.write_text(json.dumps({"artifact": "phase12_search_candidate_v1"}))
    with pytest.raises(Phase15CandidateError, match="is not a phase15_search_candidate_v1"):
        load_candidate(path)
