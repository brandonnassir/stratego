"""Phase 11 Agent 2: the frozen metric formulas, ECE and the case bootstrap."""

import math

import numpy as np
import pytest

from stratego.evaluation.phase11_evaluator import (
    Phase11EvaluatorError,
    all_finite,
    bootstrap_mean,
    bootstrap_ratio,
    build_scored_events,
    expected_calibration_error,
    overall_metrics,
    score_matrix,
    slice_metrics,
)
from stratego.training.phase11_contract import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_REPLICATES,
    LOG_PROBABILITY_FLOOR,
    OVERALL_METRIC_TOKENS,
)
from stratego.training.phase11_seed import OPPONENT_STRATA, bootstrap_stream_seed


def uniform(n=1):
    return np.full((n, 12), 1.0 / 12.0)


# ---------------------------------------------------------------------------
# Per-event formulas, against values computed by hand
# ---------------------------------------------------------------------------


def test_uniform_row_has_the_hand_computed_metrics():
    scores = score_matrix(uniform(), np.array([3]))
    assert scores["ce"][0] == pytest.approx(math.log(12.0))
    assert scores["entropy"][0] == pytest.approx(math.log(12.0))
    assert scores["true_rank_probability"][0] == pytest.approx(1 / 12)
    assert scores["confidence"][0] == pytest.approx(1 / 12)
    # 11 * (1/12)^2 + (1/12 - 1)^2
    expected = 11 * (1 / 12) ** 2 + (1 / 12 - 1) ** 2
    assert scores["brier"][0] == pytest.approx(expected)
    assert scores["top1"][0] == 0.0  # argmax is index 0, the true rank is 3


def test_a_point_mass_on_the_truth_is_perfect():
    row = np.zeros((1, 12))
    row[0, 5] = 1.0
    scores = score_matrix(row, np.array([5]))
    assert scores["ce"][0] == pytest.approx(0.0, abs=1e-15)
    assert scores["top1"][0] == 1.0
    assert scores["brier"][0] == pytest.approx(0.0, abs=1e-15)
    assert scores["entropy"][0] == pytest.approx(0.0, abs=1e-15)


def test_zero_true_probability_is_floored_inside_the_log_only():
    row = np.zeros((1, 12))
    row[0, 0] = 1.0
    scores = score_matrix(row, np.array([7]))
    assert scores["ce"][0] == pytest.approx(-math.log(LOG_PROBABILITY_FLOOR))
    assert scores["true_rank_probability"][0] == 0.0  # stored unfloored
    assert scores["log_floor_events"] == 1


def test_argmax_ties_go_to_the_first_occurrence():
    row = np.zeros((1, 12))
    row[0, 2] = row[0, 9] = 0.5
    assert score_matrix(row, np.array([2]))["top1"][0] == 1.0
    assert score_matrix(row, np.array([9]))["top1"][0] == 0.0


def test_entropy_treats_zero_log_zero_as_zero():
    row = np.zeros((1, 12))
    row[0, 0] = row[0, 1] = 0.5
    scores = score_matrix(row, np.array([0]))
    assert scores["entropy"][0] == pytest.approx(math.log(2.0))


def test_score_matrix_refuses_a_bad_shape_or_rank():
    with pytest.raises(Phase11EvaluatorError):
        score_matrix(np.zeros((2, 11)), np.array([0, 0]))
    with pytest.raises(Phase11EvaluatorError):
        score_matrix(uniform(2), np.array([0]))
    with pytest.raises(Phase11EvaluatorError):
        score_matrix(uniform(1), np.array([12]))


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------


def test_ece_uses_fifteen_equal_width_bins_with_a_closed_top():
    result = expected_calibration_error(np.array([1.0]), np.array([1.0]))
    assert len(result["bins"]) == 15
    assert result["bins"][14]["events"] == 1
    assert result["ece"] == pytest.approx(0.0)


def test_ece_is_the_pooled_weighted_gap():
    confidence = np.array([0.1, 0.1, 0.9, 0.9])
    correct = np.array([0.0, 0.0, 1.0, 0.0])
    result = expected_calibration_error(confidence, correct)
    # bin 1 = [1/15, 2/15): two events, confidence .1, accuracy 0 -> gap .1
    # bin 13 = [13/15, 14/15): two events, confidence .9, accuracy .5 -> gap .4
    assert result["ece"] == pytest.approx(0.5 * 0.1 + 0.5 * 0.4)
    assert result["events"] == 4


def test_a_perfectly_calibrated_sample_has_zero_ece():
    confidence = np.full(100, 0.5)
    correct = np.array([1.0] * 50 + [0.0] * 50)
    assert expected_calibration_error(confidence, correct)["ece"] == pytest.approx(0.0)


def test_ece_of_an_empty_sample_is_not_a_number():
    result = expected_calibration_error(np.zeros(0), np.zeros(0))
    assert math.isnan(result["ece"])
    assert result["events"] == 0


# ---------------------------------------------------------------------------
# The case bootstrap
# ---------------------------------------------------------------------------


def test_the_bootstrap_is_a_pure_function_of_its_frozen_stream():
    values = np.linspace(0.1, 0.9, 64)
    first = bootstrap_mean(values, "validation", "ce_learned", replicates=500)
    again = bootstrap_mean(values, "validation", "ce_learned", replicates=500)
    assert first == again
    assert first["stream_seed"] == bootstrap_stream_seed("validation", "ce_learned")


def test_two_metric_tokens_draw_from_different_streams():
    values = np.linspace(0.1, 0.9, 64)
    left = bootstrap_mean(values, "validation", "ce_learned", replicates=500)
    right = bootstrap_mean(values, "validation", "ce_baseline", replicates=500)
    assert left["stream_seed"] != right["stream_seed"]
    assert (left["lower"], left["upper"]) != (right["lower"], right["upper"])


def test_a_stratum_slice_has_its_own_stream():
    token = f"r_ce|st={OPPONENT_STRATA[0]}"
    assert bootstrap_stream_seed("validation", token) != bootstrap_stream_seed(
        "validation", "r_ce"
    )


def test_the_two_banks_have_different_streams():
    assert bootstrap_stream_seed("validation", "r_ce") != bootstrap_stream_seed(
        "test", "r_ce"
    )


def test_the_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(11)
    values = rng.normal(0.5, 0.1, size=256)
    interval = bootstrap_mean(values, "validation", "ce_learned", replicates=2000)
    assert interval["lower"] <= interval["point"] <= interval["upper"]
    assert interval["confidence"] == BOOTSTRAP_CONFIDENCE


def test_a_constant_sample_has_a_degenerate_interval():
    values = np.full(32, 0.25)
    interval = bootstrap_mean(values, "validation", "ce_learned", replicates=200)
    assert interval["lower"] == pytest.approx(0.25)
    assert interval["upper"] == pytest.approx(0.25)


def test_the_ratio_resamples_both_aggregates_together():
    """A ratio built from two independent draws is not the frozen ratio."""
    rng = np.random.default_rng(5)
    numerator = rng.normal(1.0, 0.2, size=128)
    denominator = numerator * 2.0  # perfectly correlated: the ratio is exact
    interval = bootstrap_ratio(numerator, denominator, "validation", "r_ce", replicates=500)
    assert interval["point"] == pytest.approx(0.5)
    assert interval["lower"] == pytest.approx(0.5)
    assert interval["upper"] == pytest.approx(0.5)


def test_an_empty_sample_gives_a_non_finite_interval():
    interval = bootstrap_mean(np.zeros(0), "validation", "ce_learned")
    assert math.isnan(interval["point"])
    assert interval["cases"] == 0


def test_the_frozen_replicate_count_is_ten_thousand():
    assert BOOTSTRAP_REPLICATES == 10_000
    assert BOOTSTRAP_CONFIDENCE == 0.95


# ---------------------------------------------------------------------------
# The table and the metric block
# ---------------------------------------------------------------------------


def synthetic_block(case_id, stratum, source, color, size, rng):
    probabilities = rng.dirichlet(np.ones(12), size=size)
    truth = rng.integers(0, 12, size=size)
    baseline = np.tile(np.ones(12) / 12.0, (size, 1))
    return {
        "case_id": case_id,
        "opponent_stratum": stratum,
        "opponent_setup_source": source,
        "observer_color": color,
        "true_rank": truth,
        "bucket_index": rng.integers(0, 3, size=size).astype(np.int8),
        "piece_moved": rng.integers(0, 2, size=size).astype(np.uint8),
        "learned": score_matrix(probabilities, truth),
        "baseline": score_matrix(baseline, truth),
    }


@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(2026)
    blocks = []
    for index, stratum in enumerate(OPPONENT_STRATA):
        for source in ("p10d", "neutral"):
            for ordinal in range(2):
                case_id = f"case|{stratum}|{source}|{ordinal}"
                for color in ("red", "blue"):
                    blocks.append(
                        synthetic_block(case_id, stratum, source, color, 40, rng)
                    )
    return build_scored_events(blocks)


def test_the_table_pools_both_colour_games_into_one_case(table):
    assert table.case_count == len(OPPONENT_STRATA) * 2 * 2
    counts = table.case_event_counts()
    assert set(counts.tolist()) == {80}


def test_the_overall_block_carries_every_frozen_token(table):
    overall = overall_metrics(table, "validation")
    assert set(OVERALL_METRIC_TOKENS) <= set(overall["metrics"])
    assert overall["cases_without_events"] == 0
    assert not all_finite(overall)


def test_the_case_mean_weights_cases_equally_not_events():
    rng = np.random.default_rng(7)
    small = synthetic_block("small", OPPONENT_STRATA[0], "p10d", "red", 4, rng)
    large = synthetic_block("large", OPPONENT_STRATA[0], "p10d", "red", 400, rng)
    table = build_scored_events([small, large])
    present, means = table.case_means(table.columns["ce_learned"])
    assert present.size == 2
    overall = overall_metrics(table, "validation")
    assert overall["metrics"]["ce_learned"]["point"] == pytest.approx(float(means.mean()))
    pooled = float(table.columns["ce_learned"].mean())
    assert overall["metrics"]["ce_learned"]["point"] != pytest.approx(pooled)


def test_the_delta_is_the_mean_of_per_case_deltas(table):
    overall = overall_metrics(table, "validation")
    metrics = overall["metrics"]
    assert metrics["top1_delta"]["point"] == pytest.approx(
        metrics["top1_learned"]["point"] - metrics["top1_baseline"]["point"]
    )


def test_r_ce_is_the_ratio_of_case_mean_ces(table):
    overall = overall_metrics(table, "validation")
    metrics = overall["metrics"]
    assert metrics["r_ce"]["point"] == pytest.approx(
        metrics["ce_learned"]["point"] / metrics["ce_baseline"]["point"]
    )


def test_every_required_slice_is_present(table):
    slices = slice_metrics(table, "validation")
    assert set(slices) == {
        "opponent_stratum",
        "observer_color",
        "progress_bucket",
        "piece_moved",
        "true_rank",
        "opponent_setup_source",
    }
    assert set(slices["opponent_stratum"]) == set(OPPONENT_STRATA)
    assert set(slices["observer_color"]) == {"red", "blue"}
    assert set(slices["piece_moved"]) == {"moved", "unmoved"}
    for stratum in OPPONENT_STRATA:
        assert "r_ce" in slices["opponent_stratum"][stratum]
        assert "ece_learned" in slices["opponent_stratum"][stratum]


def test_all_finite_reports_the_path_of_a_bad_value():
    assert all_finite({"a": {"b": float("nan")}}) == ["a.b"]
    assert all_finite({"a": [1.0, float("inf")]}) == ["a[1]"]
    assert all_finite({"a": 1.0, "b": True, "c": "text"}) == []
