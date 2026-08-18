"""Regression: the frozen Phase 10 gates, their inequalities and boundaries.

The contract mixes strict and non-strict thresholds, and the difference
decides real cases. Every gate below therefore gets a boundary pair: one
input exactly at the threshold and one a single representable step on the
failing side. If a `>` ever becomes a `>=`, or a threshold moves, exactly
one of these fails.

Two shapes of input appear, deliberately:

- `TestPrimitives` and the end-to-end classification tests drive the gates
  from real per-game scores, because the contract requires the gates to be
  computable from primitives alone;
- the boundary tests hand-build the summary a matchup produces, because a
  game score is one of exactly three values and an effective win rate of
  `0.49 - 1e-9` is therefore not something a real game list can express.
  Constant per-case inputs are used wherever an interval endpoint is under
  test, so a bootstrap bound is the quantity itself rather than a sampling
  artifact.

Nothing here plays a game or loads a model.
"""

import math

import pytest

from stratego.training import phase10_acceptance as pa
from stratego.training import phase10_contract as pc

STEP = 1e-9


def case_ids(count):
    return tuple(f"case_{index:04d}" for index in range(count))


def games(count, red_score, blue_score):
    return tuple((red_score, blue_score) for _ in range(count))


def summary(token, ewr, *, lower=None, red=None, blue=None, delta=None, delta_lower=None):
    """One matchup summary, built directly at the value under test."""
    payload = {
        "token": token,
        "bank": "test",
        "case_count": 64,
        "learned_ewr": ewr,
        "learned_red_ewr": ewr if red is None else red,
        "learned_blue_ewr": ewr if blue is None else blue,
        "learned_interval": {"lower": ewr if lower is None else lower, "upper": 1.0},
    }
    if delta is not None:
        payload["delta"] = delta
        payload["neutral_ewr"] = ewr - delta
        payload["delta_interval"] = {
            "lower": delta if delta_lower is None else delta_lower,
            "upper": 1.0,
        }
    return payload


class TestPrimitives:
    def test_case_score_is_the_mean_of_the_two_colour_games(self):
        entry = pa.MatchupOutcomes("t", case_ids(2), ((1.0, 0.0), (0.5, 1.0)))
        assert entry.learned_case_scores() == (0.5, 0.75)

    def test_colour_scores_follow_the_frozen_pairing(self):
        entry = pa.MatchupOutcomes("t", case_ids(2), ((1.0, 0.0), (0.5, 1.0)))
        assert entry.learned_color_scores(0) == (1.0, 0.5)
        assert entry.learned_color_scores(1) == (0.0, 1.0)

    def test_a_summary_is_derived_from_primitives_only(self):
        entry = pa.MatchupOutcomes(
            "t", case_ids(2), ((1.0, 1.0), (0.0, 1.0)), ((0.5, 0.5), (0.5, 0.5))
        )
        derived = pa.matchup_summary(entry, "test")
        assert derived["learned_ewr"] == pytest.approx(0.75)
        assert derived["learned_red_ewr"] == pytest.approx(0.5)
        assert derived["learned_blue_ewr"] == pytest.approx(1.0)
        assert derived["neutral_ewr"] == pytest.approx(0.5)
        assert derived["delta"] == pytest.approx(0.25)

    def test_paired_differences_need_a_neutral_arm(self):
        entry = pa.MatchupOutcomes("t", case_ids(1), ((1.0, 0.0),))
        with pytest.raises(pa.Phase10AcceptanceError):
            entry.paired_differences()

    def test_an_impossible_game_score_is_refused(self):
        with pytest.raises(pa.Phase10AcceptanceError):
            pa.MatchupOutcomes("t", case_ids(1), ((0.75, 1.0),))

    def test_mismatched_arm_lengths_are_refused(self):
        with pytest.raises(pa.Phase10AcceptanceError):
            pa.MatchupOutcomes("t", case_ids(2), games(2, 1.0, 1.0), games(1, 1.0, 1.0))

    def test_a_case_needs_exactly_two_colour_games(self):
        with pytest.raises(pa.Phase10AcceptanceError):
            pa.MatchupOutcomes("t", case_ids(1), ((1.0, 1.0, 1.0),))

    def test_effective_win_rate_of_an_empty_sample_is_refused(self):
        with pytest.raises(pa.Phase10AcceptanceError):
            pa.effective_win_rate([])

    def test_duplicate_matchup_tokens_are_refused(self):
        entry = pa.MatchupOutcomes("t", case_ids(1), ((1.0, 1.0),))
        with pytest.raises(pa.Phase10AcceptanceError):
            pa.summarize_matchups([entry, entry], "test")


class TestInequalityHelpers:
    def test_at_least_is_non_strict_and_above_is_strict(self):
        assert pa.at_least(0.49, 0.49)
        assert not pa.above(0.49, 0.49)
        assert pa.above(0.49 + STEP, 0.49)
        assert not pa.at_least(0.49 - STEP, 0.49)


class TestGateA:
    def _summaries(self, ewr, lower=None):
        return {
            pc.MATCHUP_LEARNED_VS_NEUTRAL: summary(
                pc.MATCHUP_LEARNED_VS_NEUTRAL, ewr, lower=lower
            )
        }

    def test_exactly_at_the_ordinary_ewr_threshold_passes(self):
        result = pa.gate_a(self._summaries(0.49, lower=0.48))
        assert result["ordinary_checks"]["ewr_ok"]
        assert result["pass"]

    def test_below_the_ordinary_ewr_threshold_fails(self):
        result = pa.gate_a(self._summaries(0.49 - STEP, lower=0.48))
        assert not result["ordinary_checks"]["ewr_ok"]
        assert not result["pass"]

    def test_a_lower_bound_exactly_at_0_47_fails_because_the_bound_is_strict(self):
        result = pa.gate_a(self._summaries(0.60, lower=0.47))
        assert not result["ordinary_checks"]["lb_ok"]
        assert not result["pass"]

    def test_a_lower_bound_one_step_above_0_47_passes(self):
        assert pa.gate_a(self._summaries(0.60, lower=0.47 + STEP))["pass"]

    def test_the_improved_criterion_needs_0_52_and_a_bound_above_0_50(self):
        assert pa.gate_a(self._summaries(0.52, lower=0.50 + STEP))["improved"]
        assert not pa.gate_a(self._summaries(0.52 - STEP, lower=0.60))["improved"]
        assert not pa.gate_a(self._summaries(0.60, lower=0.50))["improved"]


class TestGateB:
    def _summaries(self, strategic, tactical, anchor):
        return {
            pc.MATCHUP_STRATEGIC: summary(pc.MATCHUP_STRATEGIC, 0.5, delta=strategic),
            pc.MATCHUP_TACTICAL: summary(pc.MATCHUP_TACTICAL, 0.5, delta=tactical),
            pc.MATCHUP_PHASE8_ANCHOR: summary(
                pc.MATCHUP_PHASE8_ANCHOR, 0.5, delta=anchor
            ),
        }

    def test_league_weights_combine_the_three_deltas(self):
        assert pa.league_delta(self._summaries(0.10, 0.0, 0.0))["delta_l"] == pytest.approx(
            0.045
        )
        assert pa.league_delta(self._summaries(0.0, 0.10, 0.0))["delta_l"] == pytest.approx(
            0.035
        )
        assert pa.league_delta(self._summaries(0.0, 0.0, 0.10))["delta_l"] == pytest.approx(
            0.020
        )

    def test_exactly_at_minus_0_01_with_a_bound_above_minus_0_03_passes(self):
        result = pa.gate_b(self._summaries(-0.01, -0.01, -0.01), [-0.02] * 64, "test")
        assert result["delta_l"] == pytest.approx(-0.01)
        assert result["checks"]["delta_l_ok"]
        assert result["checks"]["lb_ok"]
        assert result["pass"]

    def test_just_below_minus_0_01_fails(self):
        summaries = self._summaries(-0.01 - STEP, -0.01 - STEP, -0.01 - STEP)
        assert not pa.gate_b(summaries, [-0.02] * 64, "test")["checks"]["delta_l_ok"]

    def test_a_bound_exactly_at_minus_0_03_fails_because_it_is_strict(self):
        result = pa.gate_b(self._summaries(0.0, 0.0, 0.0), [-0.03] * 64, "test")
        assert result["interval"]["lower"] == pytest.approx(-0.03)
        assert not result["checks"]["lb_ok"]
        assert not result["pass"]

    def test_a_bound_one_step_above_minus_0_03_passes(self):
        result = pa.gate_b(self._summaries(0.0, 0.0, 0.0), [-0.03 + STEP] * 64, "test")
        assert result["checks"]["lb_ok"]

    def test_zero_league_delta_is_not_significantly_positive(self):
        result = pa.gate_b(self._summaries(0.0, 0.0, 0.0), [0.0] * 64, "test")
        assert result["pass"]
        assert not result["significantly_positive"]

    def test_a_positive_league_delta_with_a_positive_bound_is_significant(self):
        result = pa.gate_b(self._summaries(0.04, 0.04, 0.04), [0.04] * 64, "test")
        assert result["significantly_positive"]

    def test_no_cases_is_refused(self):
        with pytest.raises(pa.Phase10AcceptanceError):
            pa.gate_b(self._summaries(0.0, 0.0, 0.0), [], "test")

    def test_league_differences_require_one_shared_case_list(self):
        entries = {
            token: pa.MatchupOutcomes(
                token, case_ids(count), games(count, 1.0, 1.0), games(count, 0.5, 0.5)
            )
            for token, count in (
                (pc.MATCHUP_STRATEGIC, 4),
                (pc.MATCHUP_TACTICAL, 4),
                (pc.MATCHUP_PHASE8_ANCHOR, 2),
            )
        }
        with pytest.raises(pa.Phase10AcceptanceError):
            pa.league_case_differences(entries)

    def test_league_differences_are_the_weighted_per_case_combination(self):
        entries = {
            token: pa.MatchupOutcomes(
                token, case_ids(2), games(2, 1.0, 1.0), games(2, 0.5, 0.5)
            )
            for token in (
                pc.MATCHUP_STRATEGIC,
                pc.MATCHUP_TACTICAL,
                pc.MATCHUP_PHASE8_ANCHOR,
            )
        }
        assert pa.league_case_differences(entries) == pytest.approx((0.5, 0.5))


class TestGateC:
    def _summaries(self, lower):
        return {
            token: summary(token, 0.5, delta=0.0, delta_lower=lower)
            for token in pc.GATE_C["opponents"]
        }

    def test_a_bound_exactly_at_minus_0_03_fails(self):
        assert not pa.gate_c(self._summaries(-0.03))["pass"]

    def test_a_bound_just_above_minus_0_03_passes(self):
        assert pa.gate_c(self._summaries(-0.03 + STEP))["pass"]

    def test_one_failing_opponent_fails_the_gate(self):
        summaries = self._summaries(0.0)
        summaries[pc.MATCHUP_TACTICAL] = summary(
            pc.MATCHUP_TACTICAL, 0.5, delta=0.0, delta_lower=-0.04
        )
        result = pa.gate_c(summaries)
        assert not result["pass"]
        assert not result["checks"][pc.MATCHUP_TACTICAL]


class TestGateD:
    def _summaries(self, overall, red, blue, basic, paired_lower=0.0):
        return {
            pc.MATCHUP_RANDOM: summary(
                pc.MATCHUP_RANDOM,
                overall,
                red=red,
                blue=blue,
                delta=0.0,
                delta_lower=paired_lower,
            ),
            pc.MATCHUP_BASIC: summary(
                pc.MATCHUP_BASIC, basic, delta=0.0, delta_lower=paired_lower
            ),
        }

    def test_exactly_at_every_threshold_passes(self):
        assert pa.gate_d(self._summaries(0.95, 1.0, 0.90, 0.80))["pass"]

    def test_a_blue_rate_just_below_0_90_fails_even_when_overall_passes(self):
        result = pa.gate_d(self._summaries(0.99, 1.0, 0.90 - STEP, 0.95))
        assert not result["checks"]["random_blue"]
        assert not result["pass"]

    def test_a_red_rate_just_below_0_90_fails(self):
        result = pa.gate_d(self._summaries(0.99, 0.90 - STEP, 1.0, 0.95))
        assert not result["checks"]["random_red"]

    def test_overall_just_below_0_95_fails(self):
        result = pa.gate_d(self._summaries(0.95 - STEP, 1.0, 1.0, 0.95))
        assert not result["checks"]["random_overall"]

    def test_basic_just_below_0_80_fails(self):
        assert not pa.gate_d(self._summaries(1.0, 1.0, 1.0, 0.80 - STEP))["checks"]["basic"]

    def test_a_paired_bound_exactly_at_minus_0_03_fails(self):
        result = pa.gate_d(self._summaries(1.0, 1.0, 1.0, 1.0, paired_lower=-0.03))
        assert not result["pass"]
        assert not result["checks"][f"{pc.MATCHUP_RANDOM}_paired_lb"]

    def test_a_paired_bound_one_step_above_passes(self):
        assert pa.gate_d(
            self._summaries(1.0, 1.0, 1.0, 1.0, paired_lower=-0.03 + STEP)
        )["pass"]


class TestGateE:
    def _report(self, **overrides):
        report = {
            "min_normalized_family_entropy": 0.85,
            "min_effective_families": 10.0,
            "min_family_probability": 0.015,
            "max_family_probability": 0.18,
            "min_within_family_normalized_base_entropy": 0.70,
            "max_conditional_base_probability": 0.10,
        }
        report.update(overrides)
        return report

    def test_exactly_at_every_threshold_passes(self):
        assert pa.gate_e(self._report())["pass"]

    @pytest.mark.parametrize(
        "key,value",
        [
            ("min_normalized_family_entropy", 0.85 - STEP),
            ("min_effective_families", 10.0 - STEP),
            ("min_family_probability", 0.015 - STEP),
            ("max_family_probability", 0.18 + STEP),
            ("min_within_family_normalized_base_entropy", 0.70 - STEP),
            ("max_conditional_base_probability", 0.10 + STEP),
        ],
    )
    def test_one_step_over_any_threshold_fails(self, key, value):
        assert not pa.gate_e(self._report(**{key: value}))["pass"]


class TestGateF:
    def test_all_zero_counters_pass(self):
        assert pa.gate_f({name: 0 for name in pa.GATE_F_COUNTERS})["pass"]

    @pytest.mark.parametrize("name", pa.GATE_F_COUNTERS)
    def test_a_single_nonzero_counter_fails(self, name):
        report = {counter: 0 for counter in pa.GATE_F_COUNTERS}
        report[name] = 1
        assert not pa.gate_f(report)["pass"]

    def test_a_missing_counter_fails_rather_than_passing_silently(self):
        report = {counter: 0 for counter in pa.GATE_F_COUNTERS}
        report.pop("illegal_setups")
        assert not pa.gate_f(report)["pass"]

    def test_the_nine_frozen_counters(self):
        assert pa.GATE_F_COUNTERS == (
            "illegal_setups",
            "inventory_errors",
            "stranded_sampled_setups",
            "split_leakage",
            "provenance_mismatch",
            "hidden_opponent_selector_inputs",
            "illegal_neural_moves",
            "non_finite_selector_outputs",
            "inference_failures",
        )


class TestGateGAndH:
    def _reproducibility(self, **overrides):
        report = {
            "same_base": True,
            "same_reflection": True,
            "same_perturbation": True,
            "same_final_fingerprint": True,
            "worker_order_independent": True,
            "process_restart_independent": True,
        }
        report.update(overrides)
        return report

    def test_reproducibility_needs_every_link(self):
        assert pa.gate_g(self._reproducibility())["pass"]
        for link in self._reproducibility():
            assert not pa.gate_g(self._reproducibility(**{link: False}))["pass"]

    def _preservation(self, **overrides):
        report = {
            "checkpoint_sha256": pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256,
            "model_state_digest": pc.ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
            "parameters": pc.ACCEPTED_PHASE9_PARAMETERS,
            "c1_optimizer_steps": 0,
        }
        report.update(overrides)
        return report

    def test_phase9_preservation_is_exact(self):
        assert pa.gate_h(self._preservation())["pass"]
        assert not pa.gate_h(self._preservation(c1_optimizer_steps=1))["pass"]
        assert not pa.gate_h(self._preservation(model_state_digest="0" * 64))["pass"]
        assert not pa.gate_h(self._preservation(checkpoint_sha256="0" * 64))["pass"]
        assert not pa.gate_h(self._preservation(parameters=863_958))["pass"]


class TestSelectionScore:
    def _summaries(self, direct_ewr, strategic, tactical, anchor, random_ewr, basic_ewr):
        return {
            pc.MATCHUP_LEARNED_VS_NEUTRAL: summary(
                pc.MATCHUP_LEARNED_VS_NEUTRAL, direct_ewr
            ),
            pc.MATCHUP_STRATEGIC: summary(pc.MATCHUP_STRATEGIC, 0.5, delta=strategic),
            pc.MATCHUP_TACTICAL: summary(pc.MATCHUP_TACTICAL, 0.5, delta=tactical),
            pc.MATCHUP_PHASE8_ANCHOR: summary(pc.MATCHUP_PHASE8_ANCHOR, 0.5, delta=anchor),
            pc.MATCHUP_RANDOM: summary(pc.MATCHUP_RANDOM, random_ewr, delta=0.0),
            pc.MATCHUP_BASIC: summary(pc.MATCHUP_BASIC, basic_ewr, delta=0.0),
        }

    def test_selection_score_uses_the_frozen_weights(self):
        summaries = self._summaries(0.60, 0.10, 0.10, 0.10, 1.0, 1.0)
        score = pa.selection_score(summaries)
        assert score["components"]["delta_direct"] == pytest.approx(0.10)
        assert score["s10"] == pytest.approx(0.10)

    def test_random_and_basic_are_not_score_components(self):
        strong = self._summaries(0.60, 0.10, 0.10, 0.10, 1.0, 1.0)
        weak = self._summaries(0.60, 0.10, 0.10, 0.10, 0.96, 0.81)
        assert pa.selection_score(strong)["s10"] == pa.selection_score(weak)["s10"]

    def test_validation_guards_are_guards(self):
        assert pa.validation_guards(
            self._summaries(0.60, 0.0, 0.0, 0.0, 0.95, 0.80)
        )["all_pass"]
        assert not pa.validation_guards(
            self._summaries(0.60, 0.0, 0.0, 0.0, 0.95 - STEP, 0.80)
        )["all_pass"]
        assert not pa.validation_guards(
            self._summaries(0.60, 0.0, 0.0, 0.0, 0.95, 0.80 - STEP)
        )["all_pass"]

    def test_a_missing_matchup_is_refused(self):
        summaries = self._summaries(0.60, 0.10, 0.10, 0.10, 1.0, 1.0)
        del summaries[pc.MATCHUP_STRATEGIC]
        with pytest.raises(pa.Phase10AcceptanceError):
            pa.selection_score(summaries)


class TestWinnerSelection:
    def _candidate(self, candidate_id, **overrides):
        entry = {
            "candidate_id": candidate_id,
            "eligible": True,
            "s10": 0.01,
            "delta_strategic": 0.01,
            "delta_direct": 0.01,
            "normalized_family_entropy": 0.9,
            "effective_base_diversity": 100.0,
        }
        entry.update(overrides)
        return entry

    def test_the_highest_score_wins(self):
        result = pa.select_winner(
            [self._candidate("P10-A"), self._candidate("P10-D", s10=0.02)]
        )
        assert result["winner"] == "P10-D"

    def test_a_full_tie_breaks_to_the_lexicographically_smaller_id(self):
        assert pa.select_winner(
            [self._candidate("P10-F"), self._candidate("P10-A")]
        )["winner"] == "P10-A"

    def test_the_tie_break_order_is_the_frozen_one(self):
        assert pa.select_winner(
            [
                self._candidate("P10-A", delta_strategic=0.005),
                self._candidate("P10-B", delta_strategic=0.006),
            ]
        )["winner"] == "P10-B"
        assert pa.select_winner(
            [
                self._candidate("P10-A", delta_direct=0.005),
                self._candidate("P10-B", delta_direct=0.006),
            ]
        )["winner"] == "P10-B"
        assert pa.select_winner(
            [
                self._candidate("P10-A", normalized_family_entropy=0.90),
                self._candidate("P10-B", normalized_family_entropy=0.91),
            ]
        )["winner"] == "P10-B"
        assert pa.select_winner(
            [
                self._candidate("P10-A", effective_base_diversity=100.0),
                self._candidate("P10-B", effective_base_diversity=101.0),
            ]
        )["winner"] == "P10-B"

    def test_an_ineligible_candidate_never_wins(self):
        assert pa.select_winner(
            [
                self._candidate("P10-A", s10=0.9, eligible=False),
                self._candidate("P10-B", s10=0.001),
            ]
        )["winner"] == "P10-B"

    def test_no_eligible_candidate_is_a_fail(self):
        result = pa.select_winner([self._candidate("P10-A", eligible=False)])
        assert result["winner"] is None
        assert result["outcome"] == "FAIL"

    def test_an_unknown_candidate_id_is_refused(self):
        with pytest.raises(pa.Phase10AcceptanceError):
            pa.select_winner([self._candidate("P10-G")])


class TestEndToEndClassification:
    """The whole acceptance path, driven only by real per-game scores."""

    def _outcomes(self, direct_games, league_delta_games):
        count = 64
        learned, neutral = league_delta_games
        entries = {
            pc.MATCHUP_LEARNED_VS_NEUTRAL: pa.MatchupOutcomes(
                pc.MATCHUP_LEARNED_VS_NEUTRAL, case_ids(count), games(count, *direct_games)
            )
        }
        for token in (
            pc.MATCHUP_STRATEGIC,
            pc.MATCHUP_TACTICAL,
            pc.MATCHUP_PHASE8_ANCHOR,
        ):
            entries[token] = pa.MatchupOutcomes(
                token,
                case_ids(count),
                games(count, *learned),
                games(count, *neutral),
            )
        for token in (pc.MATCHUP_RANDOM, pc.MATCHUP_BASIC):
            entries[token] = pa.MatchupOutcomes(
                token,
                case_ids(count),
                games(count, 1.0, 1.0),
                games(count, 1.0, 1.0),
            )
        return entries

    def _evaluate(self, direct_games, league_delta_games):
        return pa.evaluate_acceptance(
            self._outcomes(direct_games, league_delta_games),
            bank="test",
            diversity_report={
                "min_normalized_family_entropy": 0.9,
                "min_effective_families": 12.0,
                "min_family_probability": 0.03,
                "max_family_probability": 0.10,
                "min_within_family_normalized_base_entropy": 0.9,
                "max_conditional_base_probability": 0.02,
            },
            correctness_report={name: 0 for name in pa.GATE_F_COUNTERS},
            reproducibility_report={
                "same_base": True,
                "same_reflection": True,
                "same_perturbation": True,
                "same_final_fingerprint": True,
                "worker_order_independent": True,
                "process_restart_independent": True,
            },
            preservation_report={
                "checkpoint_sha256": pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256,
                "model_state_digest": pc.ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
                "parameters": pc.ACCEPTED_PHASE9_PARAMETERS,
                "c1_optimizer_steps": 0,
            },
        )

    def test_a_neutral_result_classifies_non_inferior(self):
        result = self._evaluate((1.0, 0.0), ((0.5, 0.5), (0.5, 0.5)))
        assert result["gates"]["A"]["ewr"] == pytest.approx(0.5)
        assert result["gates"]["B"]["delta_l"] == pytest.approx(0.0)
        assert result["hard_gates_all_pass"]
        assert result["gates_true"] == result["gates_total"] == 8
        assert result["classification"] == "PASS-NONINFERIOR"

    def test_a_strong_result_classifies_improved(self):
        result = self._evaluate((1.0, 0.5), ((1.0, 1.0), (0.5, 0.5)))
        assert result["gates"]["A"]["ewr"] == pytest.approx(0.75)
        assert result["gates"]["A"]["improved"]
        assert result["gates"]["B"]["significantly_positive"]
        assert result["classification"] == "PASS-IMPROVED"

    def test_a_failed_gate_classifies_fail(self):
        result = self._evaluate((0.0, 0.5), ((0.5, 0.5), (0.5, 0.5)))
        assert not result["gates"]["A"]["pass"]
        assert result["classification"] == "FAIL"

    def test_classify_refuses_a_missing_gate(self):
        with pytest.raises(pa.Phase10AcceptanceError):
            pa.classify({"A": {"pass": True}})


class TestBootstrap:
    def test_intervals_are_reproducible_from_the_frozen_stream(self):
        values = [0.5 + 0.01 * math.sin(index) for index in range(64)]
        first = pa.interval(values, "test", "vs_strategic:delta")
        assert first == pa.interval(values, "test", "vs_strategic:delta")
        assert first["resamples"] == 10_000
        assert first["confidence"] == 0.95
        assert first["engine"] == "numpy_pcg64"
        assert first["resampling_unit"] == "phase10_logical_case"

    def test_different_tokens_use_different_streams(self):
        values = [0.5 + 0.01 * math.sin(index) for index in range(64)]
        assert (
            pa.interval(values, "test", "vs_strategic:delta")["seed"]
            != pa.interval(values, "test", "vs_tactical:delta")["seed"]
        )

    def test_the_validation_and_test_banks_never_share_a_stream(self):
        values = [0.5] * 8
        assert (
            pa.interval(values, "validation", "x")["seed"]
            != pa.interval(values, "test", "x")["seed"]
        )

    def test_a_constant_sample_gives_a_point_interval(self):
        result = pa.interval([0.25] * 32, "test", "constant")
        assert result["lower"] == pytest.approx(0.25)
        assert result["upper"] == pytest.approx(0.25)
