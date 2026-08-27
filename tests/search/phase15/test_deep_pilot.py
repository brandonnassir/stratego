"""The deeper-search pilot: its control, its checks and its decision rule."""

import pytest

from stratego.search.phase15.contract import (
    DEEP_MEANINGFUL_GAIN_HIGH,
    DEEP_MEANINGFUL_GAIN_LOW,
    DEEP_PILOT_PAIRING,
    DEEP_PILOT_PRESET_NAMES,
    SEARCH_PRESETS,
    naive_compute_units,
    preset,
)
from stratego.search.phase15.deep import (
    BASELINE_PRESET,
    Phase15DeepError,
    check_configuration_invariants,
    check_frozen_identity,
    check_medium_reproduces,
    decide,
    first_divergence,
)


# -- the rungs --------------------------------------------------------------


def test_the_ladder_is_medium_large_xlarge():
    assert DEEP_PILOT_PRESET_NAMES == ("MEDIUM", "LARGE", "XLARGE")
    assert BASELINE_PRESET == "MEDIUM"
    assert DEEP_PILOT_PAIRING == "p24_b24"


def test_compute_grows_roughly_two_and_four_fold():
    medium = naive_compute_units(preset("MEDIUM"))
    assert 1.9 <= naive_compute_units(preset("LARGE")) / medium <= 2.6
    assert 3.6 <= naive_compute_units(preset("XLARGE")) / medium <= 4.6


def test_compute_grows_primarily_through_worlds():
    medium, large, xlarge = (preset(name) for name in DEEP_PILOT_PRESET_NAMES)
    assert large.worlds == 2 * medium.worlds
    assert xlarge.worlds == 3 * medium.worlds
    # Depth grows, but far less than worlds do.
    assert medium.rollout_depth < large.rollout_depth < xlarge.rollout_depth
    assert xlarge.rollout_depth / medium.rollout_depth < 2.0


def test_candidate_handling_and_regularization_are_untouched():
    """The pilot's own control. Held by inheritance, not by promise."""
    medium = preset("MEDIUM")
    for name in DEEP_PILOT_PRESET_NAMES:
        config = preset(name)
        assert config.max_root_candidates == medium.max_root_candidates == 8
        assert config.beta == medium.beta == 0.1
        assert config.epsilon == medium.epsilon
        assert config.deduplicate_worlds == medium.deduplicate_worlds
        assert config.production == medium.production


def test_the_pilot_rungs_are_not_the_section_7_strong_preset():
    """STRONG raises candidates to 12, which this pilot forbids."""
    assert SEARCH_PRESETS["STRONG"].max_root_candidates == 12
    for name in ("LARGE", "XLARGE"):
        assert preset(name).max_root_candidates == 8
        assert preset(name) is not SEARCH_PRESETS["STRONG"]


def test_the_configuration_control_passes_and_can_fail():
    report = check_configuration_invariants()
    assert report["passed"] is True
    assert set(report["rungs"]) == set(DEEP_PILOT_PRESET_NAMES)
    assert report["rungs"]["XLARGE"]["naive_ratio_vs_medium"] > 3.5


def test_the_configuration_control_catches_a_changed_candidate_rule(monkeypatch):
    from dataclasses import replace

    import stratego.search.phase15.deep as deep

    tampered = replace(preset("LARGE"), max_root_candidates=12)
    monkeypatch.setattr(
        deep, "preset_of", lambda name: tampered if name == "LARGE" else preset(name)
    )
    report = deep.check_configuration_invariants()
    assert report["passed"] is False
    assert any("max_root_candidates" in finding for finding in report["findings"])


# -- identity ---------------------------------------------------------------


def _candidate(models):
    move = models.move_models["p24"].identity
    belief = models.specialists["b24"].identity
    return {
        "selected_system": {"pairing_id": "p24_b24"},
        "move_model": {
            "checkpoint_sha256": move["checkpoint_sha256"],
            "model_state_digest": move["model_state_digest"],
        },
        "belief_model": {
            "checkpoint_sha256": belief["checkpoint_sha256"],
            "state_digest": belief["state_digest"],
        },
        "belief_calibration": {
            "applied_temperature": belief["applied_temperature"]
        },
    }


def test_identity_matches_the_frozen_candidate(fake_models):
    report = check_frozen_identity(fake_models, _candidate(fake_models))
    assert report["passed"] is True


def test_a_changed_digest_is_caught(fake_models):
    candidate = _candidate(fake_models)
    candidate["belief_model"]["state_digest"] = "f" * 64
    report = check_frozen_identity(fake_models, candidate)
    assert report["passed"] is False
    assert any("belief_state_digest" in finding for finding in report["findings"])


def test_a_pilot_on_the_wrong_system_is_refused(fake_models):
    candidate = _candidate(fake_models)
    candidate["selected_system"]["pairing_id"] = "p18_b18"
    with pytest.raises(Phase15DeepError, match="the pilot is defined for"):
        check_frozen_identity(fake_models, candidate)


# -- determinism and legality on real engines -------------------------------


def test_stronger_search_is_deterministic_and_legal(fake_models, midgame_state):
    from stratego.search.phase15.deep import check_determinism

    states = [({"position_id": "x"}, midgame_state, None)]
    report = check_determinism(fake_models, states, presets=("MEDIUM", "LARGE"))
    assert report["passed"] is True, report["findings"]
    assert report["decisions"] == 2
    assert report["rungs"]["LARGE"]["legal_decisions"] == 1


def test_decision_divergence_is_zero_against_itself(fake_models, midgame_state):
    from stratego.search.phase15.deep import decision_divergence

    states = [({"position_id": "x"}, midgame_state, None)]
    report = decision_divergence(fake_models, states, presets=("MEDIUM", "LARGE"))
    assert report["MEDIUM"]["fraction_differing_from_medium"] == 0.0
    assert report["LARGE"]["fraction_differing_from_medium"] in (0.0, 1.0)


# -- reuse verification -----------------------------------------------------


def test_medium_reproduction_check_compares_shared_boards():
    fresh = [
        {"board_id": "b0", "effective_score": 1.0, "outcome": "win", "plies": 10, "player_decisions": 5},
        {"board_id": "b1", "effective_score": 0.0, "outcome": "loss", "plies": 8, "player_decisions": 4},
    ]
    report = check_medium_reproduces(fresh, list(fresh))
    assert report["passed"] is True
    assert report["boards_compared"] == 2


def test_medium_reproduction_check_catches_a_difference():
    fresh = [{"board_id": "b0", "effective_score": 1.0, "outcome": "win", "plies": 10, "player_decisions": 5}]
    stale = [{"board_id": "b0", "effective_score": 0.0, "outcome": "loss", "plies": 10, "player_decisions": 5}]
    report = check_medium_reproduces(fresh, stale)
    assert report["passed"] is False
    assert report["findings"]


def test_medium_reproduction_check_refuses_an_empty_overlap():
    report = check_medium_reproduces([{"board_id": "a", "effective_score": 1.0, "outcome": "win", "plies": 1, "player_decisions": 1}], [])
    assert report["passed"] is False


def test_first_divergence_finds_the_parting_ply():
    rows = {
        "MEDIUM": [{"board_id": "b0", "actions": [1, 2, 3, 4]}],
        "LARGE": [{"board_id": "b0", "actions": [1, 2, 9, 4]}],
        "XLARGE": [{"board_id": "b0", "actions": [1, 2, 3, 4]}],
    }
    report = first_divergence(rows)
    assert report["LARGE"]["median_first_divergence_ply"] == 2
    assert report["LARGE"]["games_that_diverged"] == 1
    assert report["XLARGE"]["games_identical_to_medium"] == 1


# -- the decision rule ------------------------------------------------------


def _rungs(large_gain, xlarge_gain, large_p95=3.9, xlarge_p95=7.0):
    return {
        "LARGE": {
            "paired_vs_medium": {"delta": large_gain, "standard_error": 0.04},
            "idle_latency": {"p95_seconds_per_move": large_p95},
        },
        "XLARGE": {
            "paired_vs_medium": {"delta": xlarge_gain, "standard_error": 0.04},
            "idle_latency": {"p95_seconds_per_move": xlarge_p95},
        },
    }


def test_a_tiny_gain_keeps_medium():
    verdict = decide(_rungs(0.005, 0.010))
    assert verdict.recommendation == "MEDIUM"
    assert "not worth spending" in verdict.reason
    assert verdict.detail["both_rungs_regressed"] is False


def test_a_regression_is_reported_as_a_regression_not_as_no_gain():
    """"Found no gain" and "measurably worse" are different answers."""
    verdict = decide(_rungs(-0.075, -0.075))
    assert verdict.recommendation == "MEDIUM"
    assert verdict.detail["both_rungs_regressed"] is True
    assert "worse" in verdict.reason
    assert "regression" in verdict.reason


def test_one_regression_and_one_flat_rung_is_not_called_a_regression():
    verdict = decide(_rungs(-0.05, 0.01))
    assert verdict.recommendation == "MEDIUM"
    assert verdict.detail["both_rungs_regressed"] is False


def test_large_improves_and_xlarge_does_not_chooses_large():
    verdict = decide(_rungs(0.06, 0.005, xlarge_p95=4.0))
    assert verdict.recommendation == "LARGE"


def test_xlarge_is_refused_when_it_cannot_fit_the_ceiling():
    """A rung whose p95 exceeds the cap is not shippable, whatever it scores."""
    verdict = decide(_rungs(0.06, 0.20, xlarge_p95=7.0))
    assert verdict.recommendation == "LARGE"
    assert verdict.detail["xlarge_fits_latency_ceiling"] is False


def test_both_clear_and_xlarge_adds_more_chooses_xlarge():
    verdict = decide(_rungs(0.05, 0.12, xlarge_p95=4.5))
    assert verdict.recommendation == "XLARGE"


def test_both_clear_but_xlarge_adds_nothing_chooses_large():
    verdict = decide(_rungs(0.06, 0.07, xlarge_p95=4.5))
    assert verdict.recommendation == "LARGE"
    assert "adds nothing meaningful" in verdict.reason


def test_the_band_is_the_instructed_one():
    assert DEEP_MEANINGFUL_GAIN_LOW == 0.03
    assert DEEP_MEANINGFUL_GAIN_HIGH == 0.05
    assert decide(_rungs(0.029, 0.0)).recommendation == "MEDIUM"
    assert decide(_rungs(0.031, 0.0, large_p95=3.9)).recommendation == "LARGE"


def test_a_missing_rung_does_not_crash_the_rule():
    assert decide({}).recommendation == "MEDIUM"
