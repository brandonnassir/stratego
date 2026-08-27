"""Reading the pack: EWR slices, paired deltas, selection and the ladder."""

import pytest

from stratego.search.phase15.analysis import (
    MIN_SLICE_GAMES,
    Phase15AnalysisError,
    analyse_pack,
    arm_summary,
    paired_delta,
    select_system,
    system_matrix,
)
from stratego.search.phase15.budget import (
    Phase15BudgetError,
    human_play_verdict,
    ladder_analysis,
    ladder_points,
    maximum_strength_mode,
    select_budget,
    strong_depth_pilot,
    strong_gate,
)


def _row(board, arm, score, **extra):
    base = {
        "board_id": board,
        "arm_id": arm,
        "move_model": arm.split("_")[0],
        "provider": None if arm.endswith("direct") else arm.split("_", 1)[1],
        "preset_id": "direct" if arm.endswith("direct") else "TINY",
        "opponent": extra.get("opponent", "p18"),
        "opponent_class": "neural",
        "setup_source": "neutral_v1",
        "requested_family": "any",
        "player_family_key": extra.get("family", "balanced_conventional"),
        "player_color": extra.get("color", "red"),
        "ordinal": 0,
        "match_id": board,
        "outcome": "win" if score > 0.5 else ("draw" if score == 0.5 else "loss"),
        "effective_score": score,
        "winner": "red",
        "terminal_reason": "flag_capture",
        "plies": 100,
        "player_decisions": 50,
        "seconds": 10.0,
        "player_seconds": 5.0,
        "seconds_per_player_move": extra.get("per_move", 0.1),
        "move_changes": 5,
        "move_change_rate": 0.1,
        "fallbacks": 0,
        "c1_forwards": 100,
    }
    base.update({key: value for key, value in extra.items() if key in base})
    return base


# -- summaries --------------------------------------------------------------


def test_arm_summary_computes_ewr_and_every_slice():
    rows = [
        _row("b0", "p18_b18", 1.0, opponent="p18"),
        _row("b1", "p18_b18", 0.0, opponent="stress_chaos"),
        _row("b2", "p18_b18", 0.5, opponent="p18", color="blue"),
        _row("b3", "p18_b18", 1.0, opponent="stress_chaos", color="blue"),
    ]
    report = arm_summary(rows)
    assert report["games"] == 4
    assert report["ewr"] == 0.625
    assert report["wins"] == 2 and report["draws"] == 1 and report["losses"] == 1
    assert report["ewr_by_opponent"]["p18"]["ewr"] == 0.75
    assert report["ewr_by_color"]["blue"]["ewr"] == 0.75
    assert report["fallback_rate"] == 0.0


def test_a_thin_slice_is_reported_but_never_the_worst_stratum():
    rows = [_row(f"b{index}", "p18_b18", 1.0, opponent="p18") for index in range(6)]
    rows.append(_row("bx", "p18_b18", 0.0, opponent="stress_chaos"))
    report = arm_summary(rows)
    assert report["ewr_by_opponent"]["stress_chaos"]["ewr"] == 0.0
    assert report["min_opponent"]["name"] == "p18"
    assert report["min_opponent"]["games"] >= MIN_SLICE_GAMES


def test_an_empty_arm_is_refused():
    with pytest.raises(Phase15AnalysisError, match="no games"):
        arm_summary([])


# -- paired deltas ----------------------------------------------------------


def test_paired_delta_uses_only_shared_boards():
    left = [_row("b0", "p18_b18", 1.0), _row("b1", "p18_b18", 0.0), _row("bz", "p18_b18", 1.0)]
    right = [_row("b0", "p18_direct", 0.0), _row("b1", "p18_direct", 0.0)]
    report = paired_delta(left, right)
    assert report["boards"] == 2
    assert report["delta"] == 0.5
    assert report["wins"] == 1 and report["ties"] == 1


def test_paired_delta_with_no_overlap_says_so():
    report = paired_delta([_row("a", "x_b18", 1.0)], [_row("b", "x_direct", 1.0)])
    assert report == {"boards": 0, "delta": None, "standard_error": None}


def test_analyse_pack_attaches_each_arm_to_its_own_direct_reference():
    entries = []
    for board in ("b0", "b1", "b2", "b3"):
        entries.append({"row": _row(board, "p18_direct", 0.0), "move_seconds": []})
        entries.append({"row": _row(board, "p24_direct", 1.0), "move_seconds": []})
        entries.append({"row": _row(board, "p18_b18", 1.0), "move_seconds": []})
        entries.append({"row": _row(board, "p24_b24", 1.0), "move_seconds": []})
    for entry in entries:
        entry["row"]["preset_id"] = (
            "direct" if entry["row"]["arm_id"].endswith("direct") else "TINY"
        )
    summaries = analyse_pack(entries)
    assert summaries["p18_b18|TINY"]["paired_vs_direct"]["delta"] == 1.0
    assert summaries["p24_b24|TINY"]["paired_vs_direct"]["delta"] == 0.0


# -- the matrix and the selection ------------------------------------------


def _matrix(**overrides):
    base = {
        pairing_id: {
            "move_model": pairing_id.split("_")[0],
            "belief_model": pairing_id.split("_")[1],
            "direct_ewr": 0.5,
            "search_ewr": 0.60,
            "paired_delta_vs_direct": 0.1,
            "paired_standard_error": 0.05,
            "worst_opponent": {"name": "p24", "ewr": 0.4, "games": 12},
            "worst_family": {"name": "miner_forward", "ewr": 0.45, "games": 12},
            "weakness_pack_family_ewr": 0.5,
            "weakness_pack_opponent_ewr": 0.5,
            "median_seconds_per_move": 0.12,
            "p95_seconds_per_move": 0.15,
            "fallback_rate": 0.0,
            "move_change_rate": 0.2,
            "games": 120,
        }
        for pairing_id in ("p18_b18", "p18_b24", "p24_b18", "p24_b24")
    }
    for pairing_id, changes in overrides.items():
        base[pairing_id].update(changes)
    return base


def test_selection_prefers_the_stronger_system():
    selection = select_system(_matrix(p24_b18={"search_ewr": 0.95, "worst_opponent": {"name": "p24", "ewr": 0.9, "games": 12}}))
    assert selection["selected"] == "p24_b18"


def test_an_effective_tie_breaks_on_latency_then_simplicity():
    matrix = _matrix(
        p18_b24={"median_seconds_per_move": 0.30},
        p24_b18={"median_seconds_per_move": 0.30},
    )
    selection = select_system(matrix)
    assert len(selection["contenders_within_margin"]) == 4
    assert selection["selected"] in ("p18_b18", "p24_b24")


def test_worst_stratum_moves_the_ranking():
    matrix = _matrix(p18_b18={"worst_opponent": {"name": "x", "ewr": 0.0, "games": 12}})
    ranked = select_system(matrix)["ranked"]
    assert ranked[-1]["pairing_id"] == "p18_b18"


def test_an_empty_matrix_is_refused():
    with pytest.raises(Phase15AnalysisError, match="no complete system"):
        select_system({})


def test_system_matrix_pairs_each_search_arm_with_its_own_direct_arm():
    summaries = {
        "p18_direct|direct": {"ewr": 0.4},
        "p24_direct|direct": {"ewr": 0.6},
        "p18_b18|TINY": {
            "ewr": 0.7,
            "min_opponent": {"name": "a", "ewr": 0.5},
            "min_family": {"name": "b", "ewr": 0.5},
            "weakness_pack_family_ewr": 0.5,
            "weakness_pack_opponent_ewr": 0.5,
            "median_seconds_per_move": 0.1,
            "p95_seconds_per_move": 0.2,
            "fallback_rate": 0.0,
            "move_change_rate": 0.1,
            "games": 10,
            "paired_vs_direct": {"delta": 0.3, "standard_error": 0.1},
        },
    }
    matrix = system_matrix(summaries, preset_id="TINY")
    assert matrix["p18_b18"]["direct_ewr"] == 0.4
    assert matrix["p18_b18"]["paired_delta_vs_direct"] == 0.3
    assert "p24_b24" not in matrix


# -- the budget ladder ------------------------------------------------------


def _summaries(tiny=0.60, small=0.62, medium=0.63, medium_p95=0.95):
    def entry(ewr, per_move, seconds):
        return {
            "ewr": ewr,
            "games": 60,
            "min_opponent": {"name": "p24", "ewr": ewr - 0.1},
            "min_family": {"name": "miner_forward", "ewr": ewr - 0.05},
            "weakness_pack_family_ewr": ewr - 0.02,
            "weakness_pack_opponent_ewr": ewr - 0.02,
            "search_seconds_per_game": seconds,
            "median_seconds_per_move": per_move,
            "p95_seconds_per_move": per_move * 1.2,
            "max_seconds_per_move": per_move * 2,
            "move_change_rate": 0.2,
            "fallback_rate": 0.0,
        }

    return {
        "p24_b18|TINY": entry(tiny, 0.12, 6.0),
        "p24_b18|SMALL": entry(small, 0.33, 16.0),
        "p24_b18|MEDIUM": entry(medium, medium_p95 / 1.2, 42.0),
    }


def test_the_ladder_reports_cost_and_gain_per_added_second():
    points = ladder_points(_summaries(), "p24_b18")
    report = ladder_analysis(points)
    assert report["order"] == ["TINY", "SMALL", "MEDIUM"]
    assert report["rungs"][0]["ewr_gain_per_added_search_second"] is None
    assert report["rungs"][1]["added_search_seconds_per_game"] == 10.0
    assert report["rungs"][1]["ewr_gain_per_added_search_second"] == pytest.approx(0.002)
    assert report["rungs"][2]["cost_multiple_over_previous"] == pytest.approx(2.625)


def test_a_missing_ladder_is_refused():
    with pytest.raises(Phase15BudgetError, match="no ladder rung"):
        ladder_points({}, "p24_b18")


def test_the_cheapest_adequate_rung_is_selected_not_the_largest():
    selection = select_budget(ladder_points(_summaries(), "p24_b18"))
    assert selection["selected_preset"] == "TINY"
    assert selection["strongest_observed_preset"] == "MEDIUM"


def test_a_genuinely_better_rung_is_selected():
    selection = select_budget(ladder_points(_summaries(tiny=0.40), "p24_b18"))
    assert selection["selected_preset"] == "SMALL"


def test_a_rung_that_regresses_the_weakness_pack_is_skipped():
    summaries = _summaries()
    summaries["p24_b18|TINY"]["weakness_pack_family_ewr"] = 0.90
    selection = select_budget(ladder_points(summaries, "p24_b18"))
    assert selection["selected_preset"] == "TINY"
    assert selection["rungs"][1]["regresses_weakness_pack"] is True


def test_human_play_verdicts_follow_the_two_instructed_lines():
    assert human_play_verdict({"median_seconds_per_move": 0.1, "p95_seconds_per_move": 0.2, "max_seconds_per_move": 0.4})["verdict"] == "comfortable"
    assert human_play_verdict({"median_seconds_per_move": 2.5, "p95_seconds_per_move": 3.0, "max_seconds_per_move": 4.5})["verdict"] == "acceptable"
    assert human_play_verdict({"median_seconds_per_move": 4.0, "p95_seconds_per_move": 6.0, "max_seconds_per_move": 9.0})["verdict"] == "impractical"
    assert human_play_verdict({})["verdict"] == "unmeasured"


def test_the_strong_gate_refuses_a_medium_that_bought_nothing():
    gate = strong_gate(ladder_points(_summaries(), "p24_b18"))
    assert gate["allowed"] is False
    assert "useful improvement" in gate["reason"]


def test_the_strong_gate_allows_a_useful_and_practical_medium():
    gate = strong_gate(ladder_points(_summaries(tiny=0.40, small=0.42, medium=0.70), "p24_b18"))
    assert gate["allowed"] is True
    assert gate["shows_useful_improvement"] is True


def test_the_strong_gate_refuses_an_impractical_medium():
    points = ladder_points(
        _summaries(tiny=0.40, small=0.42, medium=0.70, medium_p95=9.0), "p24_b18"
    )
    gate = strong_gate(points)
    assert gate["allowed"] is False
    assert "latency budget" in gate["reason"]


def test_maximum_strength_names_the_strongest_practical_rung():
    points = ladder_points(_summaries(), "p24_b18")
    report = maximum_strength_mode(points, "TINY")
    assert report["mode"] == "MEDIUM"
    assert report["buys_a_useful_gain"] is True
    assert report["inside_engineering_margin"] is True


def test_the_strong_depth_pilot_stays_inside_the_instructed_range():
    report = strong_depth_pilot(
        {10: {"p95_seconds": 2.0}, 11: {"p95_seconds": 3.0}, 12: {"p95_seconds": 8.0}}
    )
    assert report["chosen_depth"] == 11
    with pytest.raises(Phase15BudgetError, match="instructed range"):
        strong_depth_pilot({8: {"p95_seconds": 1.0}})
