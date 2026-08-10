"""Statistics tests against synthetic tables with known answers.

No games are played here. `synthetic_results` builds structurally valid rows --
real paired-unit identifiers, one red and one blue per unit -- with outcomes
chosen by the test, so every claim about the statistics is checked against an
answer computed by hand rather than against a measurement.
"""

import random

import pytest

from stratego.evaluation.match_runner import (
    ERROR_ILLEGAL_ACTION,
    RESULT_ERROR,
    MatchResult,
)
from stratego.evaluation.statistics import (
    BOOTSTRAP_METHOD,
    DEFAULT_CONFIDENCE,
    LEAGUE_METHOD,
    OutcomeCounts,
    StatisticsError,
    bootstrap_interval,
    bradley_terry_ratings,
    build_paired_units,
    color_split,
    detect_result_problems,
    effective_win_rate,
    group_by_matchup,
    matchup_seed,
    normal_interval,
    paired_bootstrap_interval,
    pairwise_table,
    ply_summary,
    quantile,
    setup_pair_stratification,
    summarize_matchup,
    summarize_per_opponent,
    summarize_run,
    synthetic_results,
    terminal_reason_frequencies,
    unit_score_histogram,
)

# Small resample counts keep the suite fast. The acceptance run uses 10,000.
TEST_RESAMPLES = 400

WIN, DRAW, LOSS = 1.0, 0.5, 0.0

#: 32 units: 10 swept, 5 won-and-drawn, 5 lost-and-drawn, 12 swept against.
MIXED = [(WIN, WIN)] * 10 + [(WIN, DRAW)] * 5 + [(LOSS, DRAW)] * 5 + [(LOSS, LOSS)] * 12


# ---------------------------------------------------------------------------
# Effective win rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wins,draws,losses,expected",
    [
        (10, 0, 0, 1.0),
        (0, 0, 10, 0.0),
        (0, 10, 0, 0.5),
        (5, 0, 5, 0.5),
        (3, 4, 3, 0.5),
        (6, 2, 2, 0.7),
    ],
)
def test_effective_win_rate_matches_the_definition(wins, draws, losses, expected):
    assert effective_win_rate(wins, draws, losses) == pytest.approx(expected)


def test_effective_win_rate_rejects_an_empty_sample():
    with pytest.raises(StatisticsError, match="undefined for zero games"):
        effective_win_rate(0, 0, 0)


def test_effective_win_rate_rejects_negative_counts():
    with pytest.raises(StatisticsError, match="non-negative"):
        effective_win_rate(-1, 0, 1)


def test_outcome_counts_exclude_errored_games_from_the_denominator():
    counts = OutcomeCounts(wins=3, draws=2, losses=5, errors=4)
    assert counts.games == 10
    assert counts.effective_win_rate == pytest.approx(0.4)
    assert counts.to_dict()["errors"] == 4


# ---------------------------------------------------------------------------
# Known-result tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "outcomes,expected_rate,expected_counts",
    [
        ([(WIN, WIN)] * 16, 1.0, (32, 0, 0)),
        ([(LOSS, LOSS)] * 16, 0.0, (0, 0, 32)),
        ([(DRAW, DRAW)] * 16, 0.5, (0, 32, 0)),
        ([(WIN, LOSS)] * 16, 0.5, (16, 0, 16)),
        (MIXED, 0.46875, (25, 10, 29)),
    ],
)
def test_a_known_table_produces_the_expected_summary(outcomes, expected_rate, expected_counts):
    summary = summarize_matchup(synthetic_results(outcomes), resamples=TEST_RESAMPLES)
    wins, draws, losses = expected_counts
    assert summary.effective_win_rate == pytest.approx(expected_rate)
    assert (summary.counts.wins, summary.counts.draws, summary.counts.losses) == expected_counts
    assert summary.counts.games == wins + draws + losses
    assert summary.paired_units == len(outcomes)


def test_the_interval_contains_the_point_estimate():
    summary = summarize_matchup(synthetic_results(MIXED), resamples=2000)
    assert summary.interval.lower <= summary.effective_win_rate <= summary.interval.upper


@pytest.mark.parametrize(
    "outcomes,separated",
    [
        ([(WIN, WIN)] * 16, True),
        ([(LOSS, LOSS)] * 16, True),
        ([(DRAW, DRAW)] * 16, False),
        ([(WIN, LOSS)] * 16, False),
    ],
)
def test_separation_from_even_reflects_the_interval(outcomes, separated):
    summary = summarize_matchup(synthetic_results(outcomes), resamples=TEST_RESAMPLES)
    assert summary.separated_from_even is separated


def test_the_effective_win_rate_equals_the_mean_paired_unit_score():
    """The two must agree, or the interval would be centred on the wrong statistic."""
    rows = synthetic_results(MIXED)
    units = build_paired_units(rows)
    mean_unit_score = sum(unit.score for unit in units) / len(units)
    counts = OutcomeCounts.from_results(rows)
    assert mean_unit_score == pytest.approx(counts.effective_win_rate)


def test_the_unit_score_histogram_distinguishes_equal_win_rates():
    """All-draws and win-one-lose-one both score 0.5 and are not the same result."""
    all_draws = build_paired_units(synthetic_results([(DRAW, DRAW)] * 8))
    split = build_paired_units(synthetic_results([(WIN, LOSS)] * 8))
    assert unit_score_histogram(all_draws) == unit_score_histogram(split)

    sweeps = build_paired_units(synthetic_results([(WIN, WIN)] * 4 + [(LOSS, LOSS)] * 4))
    assert unit_score_histogram(sweeps) == {"0.0": 4, "0.25": 0, "0.5": 0, "0.75": 0, "1.0": 4}
    assert unit_score_histogram(all_draws)["0.5"] == 8


# ---------------------------------------------------------------------------
# The resampling unit is the paired unit
# ---------------------------------------------------------------------------


def test_the_bootstrap_resamples_paired_units_not_games():
    """The decisive test for the paired interval.

    In a table where the candidate wins every red game and loses every blue one,
    every paired unit scores exactly 0.5, so a bootstrap over units can only ever
    produce 0.5 and its interval is a point. A bootstrap over the individual games
    sees an even split of wins and losses, and must land near the normal-theory
    width for independent observations at p = 0.5. The two differ by the entire
    width of the interval, so this cannot pass by accident.
    """
    import math

    rows = synthetic_results([(WIN, LOSS)] * 16)
    units = build_paired_units(rows)
    paired = paired_bootstrap_interval(units, resamples=2000, seed=5)
    assert paired.lower == paired.upper == pytest.approx(0.5)
    assert paired.width == pytest.approx(0.0)
    assert paired.resampling_unit == "paired_unit"

    scores = [float(row.candidate_score) for row in rows]
    game_level = bootstrap_interval(scores, resamples=2000, seed=5)
    expected = 2 * 1.959964 * 0.5 / math.sqrt(len(scores))
    assert game_level.width >= 0.8 * expected
    assert game_level.width == pytest.approx(expected, rel=0.25)


def test_a_unit_contributes_the_mean_of_its_two_games():
    units = build_paired_units(synthetic_results([(WIN, DRAW)]))
    assert units[0].red_score == 1.0
    assert units[0].blue_score == 0.5
    assert units[0].score == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Bootstrap reproducibility
# ---------------------------------------------------------------------------


def test_the_same_seed_gives_a_bit_identical_interval():
    values = [unit.score for unit in build_paired_units(synthetic_results(MIXED))]
    first = bootstrap_interval(values, resamples=1500, seed=4242)
    second = bootstrap_interval(values, resamples=1500, seed=4242)
    assert (first.lower, first.upper) == (second.lower, second.upper)
    assert first.to_dict() == second.to_dict()


def test_different_seeds_give_different_intervals():
    """Asserted over a set of seeds, not a single pair.

    A resample mean over 32 units lands on a multiple of 1/128, so two seeds
    coinciding on both endpoints is common; requiring a specific pair to differ
    would be a flaky test of nothing.
    """
    values = [unit.score for unit in build_paired_units(synthetic_results(MIXED))]
    intervals = {
        (
            bootstrap_interval(values, resamples=1500, seed=seed).lower,
            bootstrap_interval(values, resamples=1500, seed=seed).upper,
        )
        for seed in range(12)
    }
    assert len(intervals) > 1


def test_the_interval_does_not_depend_on_the_internal_block_size():
    """The generator is one stream, so blocking cannot move an endpoint."""
    from stratego.evaluation import statistics as stats

    values = [unit.score for unit in build_paired_units(synthetic_results(MIXED))]
    original = stats._BOOTSTRAP_BLOCK
    try:
        stats._BOOTSTRAP_BLOCK = 2000
        whole = bootstrap_interval(values, resamples=1500, seed=808)
        stats._BOOTSTRAP_BLOCK = 97
        blocked = bootstrap_interval(values, resamples=1500, seed=808)
    finally:
        stats._BOOTSTRAP_BLOCK = original
    assert (whole.lower, whole.upper) == (blocked.lower, blocked.upper)


def test_the_interval_records_its_own_parameters():
    values = [unit.score for unit in build_paired_units(synthetic_results(MIXED))]
    interval = bootstrap_interval(values, resamples=750, seed=11, confidence=0.9)
    payload = interval.to_dict()
    assert payload["method"] == BOOTSTRAP_METHOD
    assert payload["resamples"] == 750
    assert payload["seed"] == 11
    assert payload["confidence"] == 0.9
    assert payload["sample_size"] == len(values)


def test_a_wider_confidence_level_gives_a_wider_interval():
    values = [unit.score for unit in build_paired_units(synthetic_results(MIXED))]
    narrow = bootstrap_interval(values, resamples=3000, seed=3, confidence=0.5)
    wide = bootstrap_interval(values, resamples=3000, seed=3, confidence=0.99)
    assert wide.width > narrow.width


def test_a_single_observation_gives_a_point_interval():
    units = build_paired_units(synthetic_results([(WIN, DRAW)]))
    interval = paired_bootstrap_interval(units, resamples=50, seed=1)
    assert interval.lower == interval.upper == pytest.approx(0.75)
    assert interval.sample_size == 1


def test_the_bootstrap_rejects_bad_parameters():
    with pytest.raises(StatisticsError, match="empty sample"):
        bootstrap_interval([])
    with pytest.raises(StatisticsError, match="resamples must be"):
        bootstrap_interval([0.5, 1.0], resamples=0)
    with pytest.raises(StatisticsError, match="confidence must be"):
        bootstrap_interval([0.5, 1.0], confidence=1.0)


def test_more_units_narrow_the_interval():
    """The interval must actually respond to sample size."""
    small = paired_bootstrap_interval(
        build_paired_units(synthetic_results(MIXED)), resamples=3000, seed=6
    )
    large = paired_bootstrap_interval(
        build_paired_units(synthetic_results(MIXED * 8)), resamples=3000, seed=6
    )
    assert large.width < small.width / 2


# ---------------------------------------------------------------------------
# Quantiles and the normal cross-check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "probability,expected",
    [(0.0, 1.0), (0.5, 3.0), (1.0, 5.0), (0.25, 2.0), (0.125, 1.5)],
)
def test_quantile_interpolates_linearly(probability, expected):
    assert quantile([1.0, 2.0, 3.0, 4.0, 5.0], probability) == pytest.approx(expected)


def test_quantile_rejects_a_bad_probability():
    with pytest.raises(StatisticsError, match="probability must be"):
        quantile([1.0, 2.0], 1.5)


def test_the_normal_interval_is_reported_alongside_the_bootstrap():
    summary = summarize_matchup(synthetic_results(MIXED), resamples=2000)
    assert summary.normal is not None
    assert summary.normal.method == "normal_approximation"
    # Two estimators of the same quantity: close, but not required to be equal.
    assert abs(summary.normal.lower - summary.interval.lower) < 0.1


def test_the_normal_interval_needs_two_observations():
    with pytest.raises(StatisticsError, match="at least two"):
        normal_interval([0.5])


def test_the_normal_interval_uses_the_expected_z_for_95_percent():
    """A sanity check on the bisection inverse-normal, not on the interval."""
    values = [0.0, 1.0]
    interval = normal_interval(values, confidence=DEFAULT_CONFIDENCE)
    mean, standard_error = 0.5, (0.5 ** 0.5) / (2 ** 0.5)
    assert (mean - interval.lower) / standard_error == pytest.approx(1.959964, abs=1e-5)


# ---------------------------------------------------------------------------
# Colour, setup, terminal and length diagnostics
# ---------------------------------------------------------------------------


def test_the_colour_split_separates_red_from_blue():
    rows = synthetic_results([(WIN, LOSS)] * 8)
    split = color_split(rows)
    assert split["red"]["effective_win_rate"] == 1.0
    assert split["blue"]["effective_win_rate"] == 0.0
    assert split["difference_red_minus_blue"] == pytest.approx(1.0)
    assert split["red"]["games"] == split["blue"]["games"] == 8


def test_the_colour_split_records_which_side_moves_first():
    split = color_split(synthetic_results([(WIN, LOSS)] * 4))
    assert split["red"]["moves_first"] is True
    assert split["blue"]["moves_first"] is False


def test_terminal_reasons_are_counted_and_shared():
    rows = list(synthetic_results([(WIN, WIN)] * 3, terminal_reason="flag_capture"))
    rows += list(
        synthetic_results(
            [(DRAW, DRAW)] * 1,
            terminal_reason="battleless_move_limit_draw",
            candidate="candidate@1.0.0",
            opponent="other@1.0.0",
        )
    )
    frequencies = terminal_reason_frequencies(rows)
    assert frequencies["counts"]["flag_capture"] == 6
    assert frequencies["counts"]["battleless_move_limit_draw"] == 2
    assert sum(frequencies["shares"].values()) == pytest.approx(1.0)


def test_setup_pair_stratification_reports_the_extremes():
    outcomes = [(WIN, WIN)] * 4 + [(LOSS, LOSS)] * 4
    stratification = setup_pair_stratification(synthetic_results(outcomes))
    assert stratification["setup_pairs"] == 8
    assert stratification["pairs_candidate_won_outright"] == 4
    assert stratification["pairs_candidate_lost_outright"] == 4
    assert stratification["pair_effective_win_rate_minimum"] == 0.0
    assert stratification["pair_effective_win_rate_maximum"] == 1.0
    assert "table" not in stratification


def test_the_setup_table_is_available_on_request():
    stratification = setup_pair_stratification(
        synthetic_results([(WIN, LOSS)] * 3), include_table=True
    )
    assert set(stratification["table"]) == {0, 1, 2}
    assert stratification["table"][0]["games"] == 2


def test_ply_summary_reports_mean_median_and_range():
    rows = list(synthetic_results([(WIN, WIN)], plies=100))
    rows += list(
        synthetic_results([(WIN, WIN)], plies=200, candidate="candidate@1.0.0",
                          opponent="other@1.0.0")
    )
    summary = ply_summary(rows)
    assert summary["games"] == 4
    assert summary["mean"] == pytest.approx(150.0)
    assert summary["median"] == pytest.approx(150.0)
    assert summary["minimum"] == 100
    assert summary["maximum"] == 200
    assert summary["total"] == 600


# ---------------------------------------------------------------------------
# Structural problems
# ---------------------------------------------------------------------------


def test_a_duplicate_row_is_detected():
    rows = list(synthetic_results([(WIN, LOSS)] * 3))
    problems = detect_result_problems(rows + [rows[0]])
    assert any("duplicate match_id" in problem for problem in problems)


def test_a_half_unit_is_detected():
    rows = list(synthetic_results([(WIN, LOSS)] * 3))
    problems = detect_result_problems(rows[:-1])
    assert any("has 1 game(s)" in problem for problem in problems)


def test_a_unit_with_two_of_the_same_colour_is_detected():
    rows = list(synthetic_results([(WIN, LOSS)]))
    # Same paired unit, same colour twice, distinct match ids.
    twin = MatchResult.from_dict({**rows[0].to_dict(), "match_id": "m-forced-duplicate"})
    problems = detect_result_problems([rows[0], twin])
    assert any("colour assignments" in problem for problem in problems)


def test_building_units_raises_on_a_structural_problem():
    rows = list(synthetic_results([(WIN, LOSS)] * 3))
    with pytest.raises(StatisticsError, match="not cleanly paired"):
        build_paired_units(rows[:-1])


def test_a_clean_table_reports_no_problems():
    assert detect_result_problems(synthetic_results(MIXED)) == []


# ---------------------------------------------------------------------------
# Errored matches
# ---------------------------------------------------------------------------


def _errored_rows():
    rows = list(synthetic_results([(WIN, LOSS)] * 4))
    broken = MatchResult.from_dict(
        {
            **rows[0].to_dict(),
            "candidate_result": RESULT_ERROR,
            "candidate_score": None,
            "winner": None,
            "winner_name": None,
            "draw": False,
            "terminal_reason": "policy_error",
            "policy_error": "deliberate",
            "policy_error_category": ERROR_ILLEGAL_ACTION,
            "policy_error_role": "candidate",
            "policy_error_policy": "candidate@1.0.0",
            "policy_error_ply": 3,
        }
    )
    return [broken, *rows[1:]]


def test_an_errored_row_blocks_summarisation_by_default():
    with pytest.raises(StatisticsError, match="policy error"):
        build_paired_units(_errored_rows())


def test_an_acknowledged_errored_row_drops_its_whole_unit():
    """Half a unit is not an observation -- it would reintroduce the colour bias."""
    rows = _errored_rows()
    units = build_paired_units(rows, allow_policy_errors=True)
    assert len(units) == 3
    assert all(unit.paired_unit_id != rows[0].paired_unit_id for unit in units)


def test_an_errored_row_is_counted_but_never_scored():
    counts = OutcomeCounts.from_results(_errored_rows())
    assert counts.errors == 1
    # 4 units x 2 games, with one of the 8 rows replaced by an errored row.
    assert counts.games == 7
    summary = summarize_matchup(
        _errored_rows(), resamples=TEST_RESAMPLES, allow_policy_errors=True
    )
    assert summary.policy_errors == 1
    assert summary.paired_units == 3


def test_the_run_summary_counts_illegal_actions():
    summary = summarize_run(
        _errored_rows(), resamples=TEST_RESAMPLES, allow_policy_errors=True, league=False
    )
    assert summary["policy_errors"] == 1
    assert summary["illegal_policy_actions"] == 1


# ---------------------------------------------------------------------------
# Order invariance
# ---------------------------------------------------------------------------


def test_a_matchup_summary_is_invariant_to_row_order():
    rows = list(synthetic_results(MIXED))
    baseline = summarize_matchup(rows, resamples=1000, seed=31).to_dict()
    for seed in (1, 2, 3):
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        assert summarize_matchup(shuffled, resamples=1000, seed=31).to_dict() == baseline


def test_a_run_summary_is_invariant_to_row_order():
    rows = list(synthetic_results([(WIN, LOSS)] * 6, candidate="a@1", opponent="b@1"))
    rows += list(synthetic_results([(WIN, WIN)] * 6, candidate="a@1", opponent="c@1"))
    baseline = summarize_run(rows, resamples=500)
    shuffled = list(rows)
    random.Random(9).shuffle(shuffled)
    assert summarize_run(shuffled, resamples=500) == baseline


def test_the_matchup_seed_depends_on_the_matchup_not_its_position():
    assert matchup_seed(7, "a@1 vs b@1") == matchup_seed(7, "a@1 vs b@1")
    assert matchup_seed(7, "a@1 vs b@1") != matchup_seed(7, "a@1 vs c@1")
    assert matchup_seed(7, "a@1 vs b@1") != matchup_seed(8, "a@1 vs b@1")


def test_adding_a_matchup_leaves_the_others_untouched():
    """A run-level consequence of seeding per matchup rather than per position."""
    first = list(synthetic_results([(WIN, LOSS)] * 6, candidate="a@1", opponent="b@1"))
    second = list(synthetic_results([(WIN, WIN)] * 6, candidate="a@1", opponent="c@1"))
    alone = summarize_run(first, resamples=500, league=False)
    together = summarize_run(first + second, resamples=500, league=False)
    key = "a@1 vs b@1"
    assert together["per_matchup"][key] == alone["per_matchup"][key]


# ---------------------------------------------------------------------------
# Grouping and pooling
# ---------------------------------------------------------------------------


def test_matchups_are_grouped_and_summarised_separately():
    rows = list(synthetic_results([(WIN, WIN)] * 4, candidate="a@1", opponent="b@1"))
    rows += list(synthetic_results([(LOSS, LOSS)] * 4, candidate="a@1", opponent="c@1"))
    grouped = group_by_matchup(rows)
    assert set(grouped) == {"a@1 vs b@1", "a@1 vs c@1"}
    summary = summarize_run(rows, resamples=TEST_RESAMPLES, league=False)
    assert summary["per_matchup"]["a@1 vs b@1"]["effective_win_rate"] == 1.0
    assert summary["per_matchup"]["a@1 vs c@1"]["effective_win_rate"] == 0.0


def test_a_mixed_matchup_table_is_rejected():
    rows = list(synthetic_results([(WIN, WIN)] * 2, candidate="a@1", opponent="b@1"))
    rows += list(synthetic_results([(WIN, WIN)] * 2, candidate="a@1", opponent="c@1"))
    with pytest.raises(StatisticsError, match="received 2 matchups"):
        summarize_matchup(rows)


def test_per_opponent_pooling_counts_both_sides_of_every_game():
    rows = list(synthetic_results([(WIN, WIN)] * 5, candidate="a@1", opponent="b@1"))
    pooled = summarize_per_opponent(rows)
    assert pooled["a@1"]["wins"] == 10 and pooled["a@1"]["losses"] == 0
    assert pooled["b@1"]["losses"] == 10 and pooled["b@1"]["wins"] == 0
    assert pooled["a@1"]["effective_win_rate"] == 1.0
    assert pooled["b@1"]["effective_win_rate"] == 0.0


def test_the_pairwise_table_is_symmetric_in_its_key():
    rows = synthetic_results([(WIN, DRAW)] * 4, candidate="z@1", opponent="a@1")
    table = pairwise_table(rows)
    assert list(table) == ["a@1|z@1"]
    entry = table["a@1|z@1"]
    assert entry["games"] == 8
    assert entry["score_z@1"] + entry["score_a@1"] == pytest.approx(8.0)
    assert entry["score_z@1"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# League ratings
# ---------------------------------------------------------------------------


def _three_policy_league():
    rows = list(synthetic_results([(WIN, WIN)] * 8, candidate="a@1", opponent="b@1"))
    rows += list(synthetic_results([(WIN, WIN)] * 8, candidate="a@1", opponent="c@1"))
    rows += list(synthetic_results([(WIN, DRAW)] * 8, candidate="b@1", opponent="c@1"))
    return rows


def test_league_ratings_recover_the_true_order():
    ratings = bradley_terry_ratings(_three_policy_league())
    assert ratings.ranking == ("a@1", "b@1", "c@1")
    assert ratings.method == LEAGUE_METHOD
    assert ratings.converged


def test_league_ratings_are_deterministic():
    rows = _three_policy_league()
    first = bradley_terry_ratings(rows)
    shuffled = list(rows)
    random.Random(2).shuffle(shuffled)
    second = bradley_terry_ratings(shuffled)
    assert first.to_dict() == second.to_dict()


def test_league_strengths_are_normalised_to_a_unit_geometric_mean():
    import math

    ratings = bradley_terry_ratings(_three_policy_league())
    logs = [math.log(value) for value in ratings.strengths.values()]
    assert sum(logs) / len(logs) == pytest.approx(0.0, abs=1e-9)


def test_equal_policies_receive_equal_ratings():
    rows = synthetic_results([(WIN, LOSS)] * 16, candidate="a@1", opponent="b@1")
    ratings = bradley_terry_ratings(rows)
    assert ratings.ratings["a@1"] == pytest.approx(ratings.ratings["b@1"])
    assert ratings.ratings["a@1"] == pytest.approx(1500.0)


def test_the_prior_keeps_an_undefeated_policy_finite():
    """Without a prior the likelihood has no finite maximum for a clean sweep."""
    rows = synthetic_results([(WIN, WIN)] * 8, candidate="a@1", opponent="b@1")
    ratings = bradley_terry_ratings(rows, prior_draws=1.0)
    assert ratings.converged
    assert ratings.ratings["a@1"] > ratings.ratings["b@1"]
    assert all(abs(value) < 1e6 for value in ratings.ratings.values())


def test_without_a_prior_a_swept_policy_is_rejected_loudly():
    rows = synthetic_results([(WIN, WIN)] * 8, candidate="a@1", opponent="b@1")
    with pytest.raises(StatisticsError, match="no comparable games"):
        bradley_terry_ratings(rows, prior_draws=0.0)


def test_a_larger_prior_shrinks_the_rating_spread():
    rows = _three_policy_league()
    weak = bradley_terry_ratings(rows, prior_draws=0.5)
    strong = bradley_terry_ratings(rows, prior_draws=20.0)

    def spread(ratings):
        values = list(ratings.ratings.values())
        return max(values) - min(values)

    assert spread(strong) < spread(weak)


def test_league_ratings_record_their_own_parameters():
    payload = bradley_terry_ratings(_three_policy_league()).to_dict()
    assert payload["method"] == LEAGUE_METHOD
    assert payload["prior_draws"] == 1.0
    assert payload["elo_anchor"] == 1500.0
    assert payload["elo_scale"] == 400.0
    assert payload["iterations"] >= 1
    assert set(payload["ratings"]) == {"a@1", "b@1", "c@1"}


def test_league_ratings_need_two_policies():
    rows = synthetic_results([(WIN, LOSS)] * 2)
    with pytest.raises(StatisticsError, match="at least one scored game"):
        bradley_terry_ratings([])
    assert bradley_terry_ratings(rows).ranking  # two policies is enough


def test_the_run_summary_includes_the_league_by_default():
    summary = summarize_run(_three_policy_league(), resamples=TEST_RESAMPLES)
    assert summary["league"]["ranking"] == ["a@1", "b@1", "c@1"]
    assert summary["matchups"] == 3
    assert summary["paired_units"] == 24


def test_the_run_summary_reports_its_bootstrap_configuration():
    summary = summarize_run(_three_policy_league(), resamples=TEST_RESAMPLES, league=False)
    bootstrap = summary["bootstrap"]
    assert bootstrap["resampling_unit"] == "paired_unit"
    assert bootstrap["method"] == BOOTSTRAP_METHOD
    assert bootstrap["resamples"] == TEST_RESAMPLES


def test_summarising_nothing_is_an_error():
    with pytest.raises(StatisticsError, match="no results"):
        summarize_run([])
    with pytest.raises(StatisticsError, match="no results"):
        summarize_matchup([])
