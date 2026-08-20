"""The Phase 12 budget ladder: which rungs get played, and how they are read.

No game is played here. The ladder's arm construction is checked against
Agent 3's match apparatus, and the stopping rule — the part of Agent 4 that
decides whether to spend another hour of compute — is exercised directly on
constructed numbers, because a rule that can only be reached through a
two-hour match run is a rule nobody can check.
"""

import pytest

from stratego.search.phase12 import matchplay as mp
from stratego.search.phase12 import budget as bd
from stratego.search.phase12.contract import (
    PROVIDER_AGENT1C,
    SEARCH_PRESETS,
)


def point(
    preset_id="SMALL",
    *,
    ewr=0.6,
    seconds=28.0,
    median=0.32,
    p95=0.35,
    unstable=False,
    instability=(),
    games=64,
):
    """One measured rung, carrying the real budget of the preset it names."""
    config = bd.ladder_config(preset_id)
    return bd.BudgetPoint(
        preset_id=preset_id,
        worlds=config.worlds,
        rollout_depth=config.rollout_depth,
        max_root_candidates=config.max_root_candidates,
        games=games,
        ewr=ewr,
        move_seconds_median=median,
        move_seconds_p95=p95,
        search_seconds_per_game=seconds,
        forwards_per_move=880.0,
        unstable=unstable,
        instability=tuple(instability),
    )


# ---------------------------------------------------------------------------
# The ladder itself
# ---------------------------------------------------------------------------


def test_the_ladder_is_the_three_instructed_presets_cheapest_first():
    assert bd.LADDER_PRESET_NAMES == ("TINY", "SMALL", "MEDIUM")
    configs = [bd.ladder_config(name) for name in bd.LADDER_PRESET_NAMES]
    assert [(c.worlds, c.rollout_depth) for c in configs] == [(8, 4), (16, 6), (32, 8)]
    costs = [bd.relative_cost(c, configs[0]) for c in configs]
    assert costs == sorted(costs)


def test_every_search_arm_carries_agent1c_and_nothing_else():
    arms = bd.ladder_arms()
    assert arms[0] is mp.ARM_DIRECT
    search = arms[1:]
    assert len(search) == len(bd.LADDER_PRESET_NAMES)
    assert {arm.provider_id for arm in search} == {PROVIDER_AGENT1C}
    assert len({arm.arm_id for arm in arms}) == len(arms)


@pytest.mark.parametrize("name", ["TINY", "SMALL", "MEDIUM", "LARGE"])
def test_an_arm_id_round_trips_back_to_the_rung_it_names(name):
    assert bd.preset_of_arm(bd.ladder_arm(name).arm_id) == name


def test_an_arm_that_is_not_on_the_ladder_has_no_rung():
    with pytest.raises(bd.Phase12BudgetError):
        bd.preset_of_arm(mp.ARM_DIRECT.arm_id)


def test_the_reference_arm_is_the_accepted_direct_seat():
    # Every section 4 metric is a delta, so the zero-search anchor has to be
    # on the same boards rather than quoted from another match set.
    assert bd.ladder_arms()[0].kind == "direct"


@pytest.mark.parametrize("names", [(), ("SMALL", "SMALL")])
def test_an_incoherent_ladder_is_refused(names):
    with pytest.raises(bd.Phase12BudgetError):
        bd.ladder_arms(names)


def test_the_gated_larger_preset_is_not_added_to_agent_1s_preset_table():
    assert bd.PRESET_LARGE.preset_id == "LARGE"
    assert (bd.PRESET_LARGE.worlds, bd.PRESET_LARGE.rollout_depth) == (64, 10)
    assert "LARGE" not in SEARCH_PRESETS
    assert bd.ladder_config("LARGE") is bd.PRESET_LARGE


def test_an_unknown_rung_is_refused():
    with pytest.raises(bd.Phase12BudgetError):
        bd.ladder_config("HUGE")


def test_relative_cost_is_worlds_times_candidates_times_plies_to_leaf():
    small = bd.ladder_config("SMALL")
    medium = bd.ladder_config("MEDIUM")
    expected = (32 * 8 * 9) / (16 * 8 * 7)
    assert bd.relative_cost(medium, small) == pytest.approx(expected)
    assert bd.relative_cost(small, small) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# One rung
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs", [{"ewr": 1.5}, {"ewr": -0.1}, {"games": 0}, {"seconds": -1.0}]
)
def test_an_impossible_rung_is_refused(kwargs):
    with pytest.raises(bd.Phase12BudgetError):
        point(**kwargs)


def test_a_rung_whose_label_disagrees_with_its_budget_is_refused():
    with pytest.raises(bd.Phase12BudgetError):
        bd.BudgetPoint(
            preset_id="MEDIUM",
            worlds=16,
            rollout_depth=6,
            max_root_candidates=8,
            games=64,
            ewr=0.6,
            move_seconds_median=0.32,
            move_seconds_p95=0.35,
            search_seconds_per_game=28.0,
            forwards_per_move=880.0,
        )


# ---------------------------------------------------------------------------
# Reading the ladder
# ---------------------------------------------------------------------------


def test_analysis_reports_the_step_from_the_rung_below_and_from_the_reference():
    rows = bd.ladder_analysis(
        [
            point("TINY", ewr=0.55, seconds=10.0),
            point("SMALL", ewr=0.65, seconds=30.0),
        ],
        reference_ewr=0.50,
        reference_seconds_per_game=0.0,
    )
    assert rows[0]["delta_ewr_from_previous"] is None
    assert rows[0]["ewr_gain_vs_reference"] == pytest.approx(0.05)
    assert rows[1]["previous_preset_id"] == "TINY"
    assert rows[1]["delta_ewr_from_previous"] == pytest.approx(0.10)
    assert rows[1]["extra_search_seconds_per_game"] == pytest.approx(20.0)
    assert rows[1]["ewr_gain_per_extra_search_second"] == pytest.approx(0.005)
    assert rows[1]["ewr_gain_per_search_second_vs_reference"] == pytest.approx(0.15 / 30)
    assert rows[1]["search_seconds_multiple_of_previous"] == pytest.approx(3.0)


def test_the_operating_point_is_the_cheapest_rung_inside_the_margin():
    chosen = bd.select_operating_point(
        [
            point("TINY", ewr=0.60, seconds=10.0),
            point("SMALL", ewr=0.64, seconds=30.0),
            point("MEDIUM", ewr=0.66, seconds=75.0),
        ]
    )
    assert chosen["selected_preset_id"] == "TINY"
    assert chosen["strongest_preset_id"] == "MEDIUM"
    assert chosen["presets_within_margin"] == ["TINY", "SMALL", "MEDIUM"]


def test_a_rung_that_clearly_leads_is_selected_even_though_it_costs_more():
    chosen = bd.select_operating_point(
        [
            point("TINY", ewr=0.45, seconds=10.0),
            point("SMALL", ewr=0.70, seconds=30.0),
        ]
    )
    assert chosen["selected_preset_id"] == "SMALL"
    assert chosen["presets_within_margin"] == ["SMALL"]


def test_an_unstable_rung_cannot_be_the_operating_point():
    chosen = bd.select_operating_point(
        [
            point("TINY", ewr=0.70, seconds=10.0, unstable=True, instability=("probe",)),
            point("SMALL", ewr=0.65, seconds=30.0),
        ]
    )
    assert chosen["selected_preset_id"] == "SMALL"
    assert chosen["excluded_unstable"] == ["TINY"]


def test_an_entirely_unstable_ladder_is_refused():
    with pytest.raises(bd.Phase12BudgetError):
        bd.select_operating_point([point("TINY", unstable=True)])


# ---------------------------------------------------------------------------
# The stopping rule
# ---------------------------------------------------------------------------


def test_every_condition_is_reported_with_its_evidence_whether_it_fired_or_not():
    verdict = bd.stopping_rule([point("TINY", ewr=0.55, seconds=10.0)])
    assert set(verdict["conditions"]) == {
        "strength_clearly_stopped_improving",
        "latency_rises_much_faster_than_strength",
        "human_play_latency_impractical",
        "useful_operating_point_already_obvious",
        "larger_search_creates_instability",
        "next_preset_consumes_disproportionate_compute",
    }
    for block in verdict["conditions"].values():
        assert isinstance(block["fired"], bool)
        assert block["reading"]


def test_a_flat_top_rung_stops_the_ladder():
    verdict = bd.stopping_rule(
        [
            point("SMALL", ewr=0.64, seconds=30.0),
            point("MEDIUM", ewr=0.66, seconds=75.0, median=0.81, p95=0.90),
        ]
    )
    assert verdict["stop_scaling"] is True
    assert "strength_clearly_stopped_improving" in verdict["conditions_fired"]
    assert verdict["conditions"]["strength_clearly_stopped_improving"][
        "delta_ewr_from_previous"
    ] == pytest.approx(0.02)


def test_a_rung_that_pays_for_itself_does_not_trip_the_strength_or_latency_rules():
    verdict = bd.stopping_rule(
        [
            point("SMALL", ewr=0.50, seconds=30.0),
            point("MEDIUM", ewr=0.75, seconds=45.0, median=0.81, p95=0.90),
        ],
        next_config=None,
    )
    assert "strength_clearly_stopped_improving" not in verdict["conditions_fired"]
    assert "latency_rises_much_faster_than_strength" not in verdict["conditions_fired"]
    assert verdict["stop_scaling"] is False


def test_cost_climbing_faster_than_strength_stops_the_ladder():
    verdict = bd.stopping_rule(
        [
            point("SMALL", ewr=0.50, seconds=30.0),
            # +0.10 EWR is exactly one margin, but it cost 4x the seconds.
            point("MEDIUM", ewr=0.60, seconds=150.0, median=0.81, p95=0.90),
        ],
        next_config=None,
    )
    assert "latency_rises_much_faster_than_strength" in verdict["conditions_fired"]


def test_a_seat_too_slow_for_a_human_stops_the_ladder():
    verdict = bd.stopping_rule([point("MEDIUM", median=6.0, p95=7.5)], next_config=None)
    assert "human_play_latency_impractical" in verdict["conditions_fired"]
    block = verdict["conditions"]["human_play_latency_impractical"]
    assert block["past_comfort"] is True


def test_latency_past_comfort_but_inside_practicality_does_not_stop_the_ladder():
    block = bd.stopping_rule([point("MEDIUM", median=2.0, p95=2.4)], next_config=None)[
        "conditions"
    ]["human_play_latency_impractical"]
    assert block["fired"] is False
    assert block["past_comfort"] is True


def test_instability_anywhere_on_the_ladder_stops_it():
    verdict = bd.stopping_rule(
        [
            point("TINY", unstable=True, instability=("probe failure",)),
            point("SMALL"),
        ],
        next_config=None,
    )
    assert "larger_search_creates_instability" in verdict["conditions_fired"]
    assert verdict["conditions"]["larger_search_creates_instability"]["instability"] == {
        "TINY": ["probe failure"]
    }


def test_a_disproportionately_expensive_next_rung_stops_the_ladder():
    verdict = bd.stopping_rule(
        [point("MEDIUM", ewr=0.66, seconds=75.0)], next_config=bd.PRESET_LARGE
    )
    block = verdict["conditions"]["next_preset_consumes_disproportionate_compute"]
    assert block["fired"] is True
    assert block["next_cost_multiple_of_top"] == pytest.approx((64 * 8 * 11) / (32 * 8 * 9))


def test_an_obvious_cheap_operating_point_stops_the_ladder():
    verdict = bd.stopping_rule(
        [
            point("TINY", ewr=0.64, seconds=10.0),
            point("SMALL", ewr=0.66, seconds=30.0),
        ],
        next_config=None,
    )
    assert "useful_operating_point_already_obvious" in verdict["conditions_fired"]
    assert verdict["operating_point"]["selected_preset_id"] == "TINY"


def test_the_rule_refuses_an_empty_ladder():
    with pytest.raises(bd.Phase12BudgetError):
        bd.stopping_rule([])


def test_the_thresholds_are_echoed_so_a_reader_can_see_which_number_to_move():
    verdict = bd.stopping_rule([point()], next_config=None)
    assert verdict["thresholds"]["meaningful_ewr_gain"] == bd.MEANINGFUL_EWR_GAIN
    assert verdict["budget_version"] == bd.BUDGET_VERSION
