"""Phase 16 Agent 3: section 5's predeclared rules and the report renderer."""

import pytest

from stratego.training.phase16 import analysis as A
from stratego.training.phase16 import contract as C
from stratego.training.phase16.report_text import render_report


def hour_entry(benchmark: float, adversarial: float, *, hour: int, games: int = 60):
    return {
        "arm": "x",
        "hour": hour,
        "iteration": hour * 100,
        "optimizer_step": hour * 1000,
        "source": "ema",
        "benchmark": {
            "pack": "phase16_benchmark_v1",
            "subset": "quick60",
            "games": games,
            "ewr": benchmark,
        },
        "adversarial": {
            "pack": "phase16_adversarial_baseline_v1",
            "stratum": "adversarial_both",
            "games": 96,
            "ewr": adversarial,
        },
    }


def curves_for(a_final, b_final, c_final, *, a_h4=None, b_h4=None, c_h4=None,
               a_adv=0.70, b_adv=0.70, c_adv=0.70):
    def arm(final, h4, adv):
        return {
            "0": hour_entry(0.60, 0.60, hour=0),
            "2": hour_entry(0.65, 0.65, hour=2),
            "4": hour_entry(final if h4 is None else h4, adv, hour=4),
            "6": hour_entry(final, adv, hour=6),
        }

    return {
        A.CONTROL_ARM: arm(a_final, a_h4, a_adv),
        A.DAMPED_ARM: arm(b_final, b_h4, b_adv),
        A.DAMPED_PLUS_ARM: arm(c_final, c_h4, c_adv),
    }


# ---------------------------------------------------------------------------
# Standard errors
# ---------------------------------------------------------------------------


def test_standard_error_is_the_binomial_one_and_never_a_test():
    assert A.standard_error(0.5, 100) == pytest.approx(0.05)
    assert A.standard_error(0.8, 60) == pytest.approx((0.8 * 0.2 / 60) ** 0.5)
    assert A.standard_error(1.0, 60) == 0.0
    assert A.standard_error(0.5, 0) == 0.0


def test_a_60_board_subset_has_the_noise_the_limitations_claim():
    """`known_limitations` promises roughly +-0.05 at an EWR near 0.8."""
    assert 0.04 < A.standard_error(0.8, 60) < 0.06


# ---------------------------------------------------------------------------
# adopt_recipe
# ---------------------------------------------------------------------------


def test_adopt_fires_when_a_damped_arm_clears_the_margin():
    decision = A.decide_recipe(curves_for(0.70, 0.75, 0.72))
    assert decision["verdict"] == A.VERDICT_ADOPT
    assert decision["winner"] == A.DAMPED_ARM
    assert decision["adopt_recipe"]["threshold"] == pytest.approx(0.73)
    assert decision["adopt_recipe"]["clearing"] == [A.DAMPED_ARM]


def test_the_higher_of_two_clearing_arms_wins():
    decision = A.decide_recipe(curves_for(0.70, 0.74, 0.80))
    assert decision["winner"] == A.DAMPED_PLUS_ARM
    assert sorted(decision["adopt_recipe"]["clearing"]) == [A.DAMPED_ARM, A.DAMPED_PLUS_ARM]


def test_the_margin_is_a_threshold_not_a_tendency():
    """Exactly at the margin adopts; a hair under does not."""
    assert A.decide_recipe(curves_for(0.70, 0.73, 0.60))["verdict"] == A.VERDICT_ADOPT
    assert A.decide_recipe(curves_for(0.70, 0.7299, 0.60))["verdict"] == A.VERDICT_STOP


def test_stop_fires_when_neither_damped_arm_clears():
    decision = A.decide_recipe(curves_for(0.80, 0.79, 0.78))
    assert decision["verdict"] == A.VERDICT_STOP
    assert decision["winner"] is None
    assert "stop_rule" in decision["statement"]
    assert "No long run is authorized" in decision["statement"]
    assert "recipe" not in decision


def test_an_unfinished_shootout_is_incomplete_not_a_stop():
    curves = curves_for(0.70, 0.75, 0.72)
    del curves[A.DAMPED_PLUS_ARM]
    decision = A.decide_recipe(curves)
    assert decision["verdict"] == A.VERDICT_INCOMPLETE
    assert decision["missing_arms"] == [A.DAMPED_PLUS_ARM]


def test_the_winning_arms_config_rides_with_the_verdict():
    configs = {A.DAMPED_ARM: {"arm": {"arm_id": "b_damped"}, "hours": 6.0}}
    decision = A.decide_recipe(curves_for(0.70, 0.80, 0.72), configs)
    assert decision["recipe"] == configs[A.DAMPED_ARM]


# ---------------------------------------------------------------------------
# setups_causal and plateau_check
# ---------------------------------------------------------------------------


def test_setups_causal_compares_c_to_b_on_the_adversarial_stratum():
    decision = A.decide_recipe(curves_for(0.70, 0.75, 0.75, b_adv=0.60, c_adv=0.66))
    causal = decision["setups_causal"]
    assert causal["b_ewr"] == 0.60 and causal["c_ewr"] == 0.66
    assert causal["delta"] == pytest.approx(0.06)
    assert causal["pass"] is True
    tighter = A.decide_recipe(curves_for(0.70, 0.75, 0.75, b_adv=0.60, c_adv=0.62))
    assert tighter["setups_causal"]["pass"] is False


def test_the_plateau_slope_is_reported_for_every_arm():
    decision = A.decide_recipe(
        curves_for(0.70, 0.78, 0.72, a_h4=0.68, b_h4=0.70, c_h4=0.72)
    )
    slopes = decision["plateau_check"]["slopes"]
    assert slopes[A.DAMPED_ARM]["delta"] == pytest.approx(0.08)
    assert slopes[A.DAMPED_ARM]["per_hour"] == pytest.approx(0.04)
    assert slopes[A.DAMPED_ARM]["flat"] is False
    assert slopes[A.DAMPED_PLUS_ARM]["flat"] is True


def test_a_flat_but_passing_arm_still_adopts():
    """Section 5: `a flat B/C with a passing h6 still adopts`."""
    decision = A.decide_recipe(curves_for(0.70, 0.75, 0.72, b_h4=0.75))
    assert decision["verdict"] == A.VERDICT_ADOPT
    assert decision["plateau_check"]["slopes"][A.DAMPED_ARM]["flat"] is True
    assert "plateau moved" in decision["plateau_check"]["rule"]


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def test_a_curve_carries_every_evaluation_hour_with_its_pack_and_se():
    rows = A.arm_curve(curves_for(0.70, 0.75, 0.72), A.DAMPED_ARM)
    assert [row["hour"] for row in rows] == list(C.EVALUATION_HOURS)
    for row in rows:
        assert row["benchmark_pack"] == "phase16_benchmark_v1"
        assert row["benchmark_subset"] == "quick60"
        assert row["adversarial_stratum"] == "adversarial_both"
        assert row["benchmark_se"] > 0
    assert A.arm_curve({}, A.DAMPED_ARM) == []
    assert A.final_hour(curves_for(0.7, 0.75, 0.72), A.DAMPED_ARM)["hour"] == 6


def test_a_missing_hour_is_skipped_rather_than_invented():
    curves = curves_for(0.70, 0.75, 0.72)
    del curves[A.DAMPED_ARM]["4"]
    rows = A.arm_curve(curves, A.DAMPED_ARM)
    assert [row["hour"] for row in rows] == [0, 2, 6]
    assert A.plateau_slope(curves, A.DAMPED_ARM) is None


def test_throughput_verdict_reads_the_measured_pair():
    payload = {
        "phase16": {"plies_per_second": 2000.0},
        "phase14": {"plies_per_second": 1600.0},
        "gate": {"pass": True, "ratio_phase16_over_phase14": 1.25},
    }
    verdict = A.throughput_verdict(payload)
    assert verdict["pass"] is True
    assert "2000.0" in verdict["statement"] and "1600.0" in verdict["statement"]
    assert A.throughput_verdict({"gate": {"pass": None, "note": "n/a"}})["pass"] is None


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_the_report_renders_from_artifacts_alone(tmp_path):
    curves = curves_for(0.70, 0.78, 0.75, b_adv=0.60, c_adv=0.68)
    decision = A.decide_recipe(curves)
    text = render_report(
        gates={
            "gates": {
                "window_edge_invariant": {
                    "pass": True,
                    "trials": 24,
                    "minimum_windows": 4,
                    "max_advantage_difference": 0.0,
                    "max_wdl_difference": 0.0,
                    "tolerance": 1e-6,
                },
                "smoke_run": {
                    "pass": True,
                    "windows": 18,
                    "minutes": 20.1,
                    "cpu_rerun_bit_identical": True,
                },
                "collection_throughput": {"pass": True, "note": "1.20x"},
                "full_pytest": {"pass": True, "note": ""},
            }
        },
        throughput={
            "phase16": {"plies_per_second": 2001.7},
            "phase14": {"plies_per_second": 1672.6},
            "gate": {"pass": True, "ratio_phase16_over_phase14": 1.1967},
        },
        curves=curves,
        configs={},
        candidate=decision,
        telemetry_root=tmp_path,
        report_root=tmp_path,
    )
    assert "# Phase 16 — Agent 3 report" in text
    assert "## 2. Correctness gates" in text
    assert "## 3. The three h-curves" in text
    assert "## 5. The decision the predeclared rules produced" in text
    assert "## 6. `known_limitations`" in text
    # every EWR table names its pack (overview section 6)
    assert "phase16_benchmark_v1" in text and "phase16_adversarial_baseline_v1" in text
    assert "Verdict: ADOPT" in text
    assert "bit-identical" in text
    assert "2001.7" in text


def test_a_stop_verdict_says_so_in_the_report(tmp_path):
    curves = curves_for(0.80, 0.79, 0.78)
    text = render_report(
        gates={},
        throughput={},
        curves=curves,
        configs={},
        candidate=A.decide_recipe(curves),
        telemetry_root=tmp_path,
        report_root=tmp_path,
    )
    assert "Verdict: STOP" in text
    assert "No long run is authorized by this file" in text


def test_the_report_renders_with_nothing_recorded(tmp_path):
    text = render_report(
        gates={}, throughput={}, curves={}, configs={}, candidate={},
        telemetry_root=tmp_path, report_root=tmp_path,
    )
    assert "_No hour curves recorded._" in text
    assert "_No candidate decision recorded._" in text
    assert "## 6. `known_limitations`" in text


def test_the_limitations_section_states_the_real_ones(tmp_path):
    text = render_report(
        gates={}, throughput={}, curves={}, configs={}, candidate={},
        telemetry_root=tmp_path, report_root=tmp_path,
    )
    limitations = text.split("## 6. `known_limitations`")[1]
    for claim in (
        "6-hour horizon",
        "One seed per arm",
        # the ones the run itself taught, not only the ones the brief predicted
        "smaller than the instrument's noise",
        "No arm learned",
        "pins data, not collection time",
        "boundary bootstrap exists",
        "In-flight games are not checkpointed",
    ):
        assert claim in limitations, claim


def test_the_report_carries_the_schedule_amendment(tmp_path):
    text = render_report(
        gates={}, throughput={}, curves={}, configs={}, candidate={},
        telemetry_root=tmp_path, report_root=tmp_path,
    )
    section = text.split("## 1b. Deviations from the brief")[1].split("## 2.")[0]
    assert "§2.3" in section
    assert "n_ref" in section
    assert str(C.PLANNED_ITERATIONS) in section
    assert "Entropy deliberately not re-horizoned" in section
    # the control's row must show it untouched at Phase 14's constant
    assert "7.50e-05 | 7.50e-05 | 7.50e-05 | 7.50e-05" in section
    # and the amendment must not be sold as equalising the arms
    assert "does **not** equalise the arms" in section


# ---------------------------------------------------------------------------
# What the collector change is worth per hour
# ---------------------------------------------------------------------------


def telemetry_rows(count=5, plies=64000, rows_=63000, collect=41.0, train=28.0):
    return [
        {
            "collection": {"plies_advanced": plies, "rows": rows_, "seconds": collect},
            "optimization": {"seconds": train, "optimizer_steps": 123},
        }
        for _ in range(count)
    ]


DECOMPOSITION = {
    "whole_run": {"collection_share": 0.1733, "training_share": 0.8267,
                  "collection_hours": 10.39, "training_hours": 49.57},
    "collection_plies_per_second": {"min": 1325.5, "median": 1784.2, "max": 1900.6},
    "growth": {
        "mean_game_length": {"first5": 265.0, "last5": 733.6},
        "iteration_minutes": {"first5": 20.6, "last5": 170.1},
        "collection_minutes": {"first5": 4.8, "last5": 16.3},
        "training_minutes": {"first5": 15.7, "last5": 153.8},
        "training_share": {"first5": 0.765, "last5": 0.8984},
        "optimizer_steps_per_iteration": {"first5": 1712, "last5": 5247},
        "seconds_per_optimizer_step": {"first5": 0.5514, "last5": 1.7595},
    },
    "source": {"note": "derived from the run's own per-iteration telemetry"},
}


def test_the_comparison_reports_where_the_time_goes_not_a_collector_speedup():
    throughput = {
        "phase16": {"plies_per_second": 2001.7},
        "phase14": {"plies_per_second": 1672.6},
    }
    economics = A.end_to_end_comparison(throughput, telemetry_rows(), DECOMPOSITION)
    ours, theirs = economics["phase16"], economics["phase14"]
    # collection is a wash and the report must not sell it as a win
    assert 0.8 < economics["ratios"]["collection_plies_per_second"] < 1.2
    assert "not faster" in economics["gate3_caveat"]
    assert "is a wash" in economics["gate3_caveat"]
    # the split is the finding
    assert theirs["training_share_of_wall"] > 0.8
    assert ours["training_share_of_wall"] < theirs["training_share_of_wall"]
    assert "iteration *sizing*" in economics["finding"]


def test_without_the_decomposition_no_collection_claim_is_made():
    """The gate-3 caveat and the finding both require the recorded split."""
    throughput = {"phase16": {"plies_per_second": 2000.0}, "phase14": {"plies_per_second": 1600.0}}
    economics = A.end_to_end_comparison(throughput, telemetry_rows(), None)
    assert "gate3_caveat" not in economics
    assert "finding" not in economics
    assert "collection_plies_per_second" not in economics["ratios"]


def test_the_first_window_is_excluded_as_a_cold_start():
    throughput = {"phase16": {"plies_per_second": 2000.0}, "phase14": {"plies_per_second": 1600.0}}
    rows = telemetry_rows(count=4)
    rows[0]["collection"] = {"plies_advanced": 999999, "rows": 1, "seconds": 1.0}
    economics = A.end_to_end_comparison(throughput, rows, DECOMPOSITION)
    assert economics["phase16"]["windows"] == 3
    # the absurd cold-start window must not reach the steady-state rate
    assert economics["phase16"]["collection_plies_per_second"] < 2000.0


def test_the_comparison_states_its_caveats_and_its_phase14_source():
    throughput = {"phase16": {"plies_per_second": 2000.0}, "phase14": {"plies_per_second": 1600.0}}
    economics = A.end_to_end_comparison(throughput, telemetry_rows(), DECOMPOSITION)
    assert "candidate evaluations" in " ".join(economics["caveats"])
    assert "not the same unit" in " ".join(economics["caveats"])
    assert "smoke run" in " ".join(economics["caveats"])
    assert str(A.PHASE14_RUN_STEPS) in economics["phase14"]["source"]


def test_the_comparison_is_empty_without_evidence():
    assert A.end_to_end_comparison({"phase16": {}}, []) == {}
    assert A.end_to_end_comparison({}, [{"collection": {}}]) == {}


def test_the_report_prints_where_the_time_goes(tmp_path):
    text = render_report(
        gates={"gates": {"smoke_run": {"pass": True, "telemetry": telemetry_rows()}}},
        throughput={
            "phase16": {"plies_per_second": 2001.7},
            "phase14": {"plies_per_second": 1672.6},
            "gate": {"pass": True, "ratio_phase16_over_phase14": 1.1967},
        },
        curves={}, configs={}, candidate={},
        telemetry_root=tmp_path, report_root=tmp_path,
    )
    section = text.split("### Where the time actually goes")[1]
    assert "trained decisions / hour" in section
    assert "training share of wall" in section


# ---------------------------------------------------------------------------
# The horizon handoff to Agent 5
# ---------------------------------------------------------------------------


def test_horizon_evidence_carries_inputs_not_just_the_constant(repository_root):
    evidence = A.horizon_evidence(repository_root)
    assert evidence["measured"]["seconds_per_iteration"] == C.MEASURED_ITERATION_SECONDS
    assert evidence["measured"]["planned_iterations_for_six_hours"] == C.PLANNED_ITERATIONS
    assert evidence["derivation"]["n_ref"] == C.LR_REFERENCE_ITERATION
    assert evidence["derivation"]["lr_horizon_fraction"] == C.LR_HORIZON_FRACTION
    # the whole point: a later run must recompute, not inherit
    assert "Do not carry" in evidence["derivation"]["recompute_for_a_longer_run"]
    assert "section 4" in evidence["for_agent_5"]
    assert evidence["measured"]["conditions"]


def test_horizon_evidence_binds_the_decomposition_by_digest(repository_root):
    evidence = A.horizon_evidence(repository_root)
    block = evidence["phase14_decomposition"]
    assert block["path"] == A.PHASE14_DECOMPOSITION_PATH
    assert block["regenerate"].endswith("--role decompose")
    if block["present"]:
        assert block["sha256"] and len(block["sha256"]) == 64
        assert block["whole_run"]["training_share"] > 0.8
        assert "A fixed decision budget pins that" in block["finding"]


def test_horizon_evidence_survives_a_missing_decomposition(tmp_path):
    evidence = A.horizon_evidence(tmp_path)
    assert evidence["phase14_decomposition"]["present"] is False
    assert evidence["phase14_decomposition"]["sha256"] is None
    # the derivation is still complete without it
    assert evidence["derivation"]["n_ref"] == C.LR_REFERENCE_ITERATION


# ---------------------------------------------------------------------------
# Measurement power
# ---------------------------------------------------------------------------


def test_the_decision_records_whether_its_instrument_can_resolve_the_margin():
    decision = A.decide_recipe(curves_for(0.70, 0.75, 0.72))
    power = decision["power"]
    assert power["games"] == 60
    assert power["decision_margin"] == C.ADOPT_RECIPE_MARGIN
    # 0.03 against an SE near 0.055 is well under one standard error
    assert power["margin_in_standard_errors"] < 1.0
    assert power["resolvable"] is False
    assert "not as evidence that one recipe is better" in power["statement"]


def test_a_bigger_instrument_would_be_recorded_as_resolvable():
    """The flag tracks the instrument, not a fixed answer."""
    assert A.standard_error(0.76, 60) > C.ADOPT_RECIPE_MARGIN
    assert A.standard_error(0.76, 1000) < C.ADOPT_RECIPE_MARGIN


def test_the_full_pack_reading_is_carried_but_never_decides():
    curves = curves_for(0.70, 0.75, 0.72)
    for arm in curves.values():
        for hour in arm.values():
            hour["benchmark_full"] = {
                "pack": "phase16_benchmark_v1", "subset": "full",
                "games": 120, "ewr": 0.99,
            }
    decision = A.decide_recipe(curves)
    # the absurd full-pack value must not move any rule
    assert decision["adopt_recipe"]["candidates"][A.DAMPED_ARM] == 0.75
    assert decision["power"]["higher_powered_secondary"]["games"] == 120
    rows = A.arm_curve(curves, A.DAMPED_ARM)
    assert rows[0]["benchmark_full_ewr"] == 0.99
    assert rows[0]["benchmark_full_se"] < rows[0]["benchmark_se"]


def test_the_report_leads_with_the_power_warning(tmp_path):
    curves = curves_for(0.70, 0.78, 0.75)
    text = render_report(
        gates={}, throughput={}, curves=curves, configs={},
        candidate=A.decide_recipe(curves), telemetry_root=tmp_path, report_root=tmp_path,
    )
    section = text.split("## 5. The decision")[1]
    assert "Read this first" in section
    assert section.index("Read this first") < section.index("Verdict:")


def test_the_instrument_check_catches_a_nondeterministic_evaluator():
    curves = curves_for(0.70, 0.75, 0.72)
    for arm in curves.values():
        arm["0"]["model_state_digest"] = "622d9e" * 10
    good = A.instrument_check(curves)
    assert good["scores_agree"] is True
    assert len(set(good["starting_model_state_digest"].values())) == 1
    # a single arm scoring its own start differently is the fault it exists to find
    curves[A.DAMPED_ARM]["0"]["benchmark"]["ewr"] = 0.61
    assert A.instrument_check(curves)["scores_agree"] is False
    assert A.instrument_check({}) == {}


def test_a_stop_report_refuses_to_let_setups_causal_be_misread(tmp_path):
    """The failing `setups_causal` must not read as "expanded setups fail"."""
    curves = curves_for(0.80, 0.79, 0.78, b_adv=0.78, c_adv=0.70)
    text = render_report(
        gates={}, throughput={}, curves=curves, configs={},
        candidate=A.decide_recipe(curves), telemetry_root=tmp_path, report_root=tmp_path,
    )
    section = text.split("### What the shootout established")[1]
    assert "does not show that expanded setups fail" in section
    assert "need a longer horizon" in section
    assert "nothing here should be cited for the first" in section
