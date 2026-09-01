"""Tests for the Phase 18 Gate G1 paired non-inferiority statistics."""

from __future__ import annotations

import numpy as np
import pytest

from stratego.evaluation.phase18.noninferiority import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_REPLICATES,
    DIRECTION_DELTA_MAX,
    DIRECTION_DELTA_MIN,
    NonInferiorityError,
    assess_margin,
    paired_ratio_delta,
    paired_unit_delta,
)
from stratego.evaluation.statistics import bootstrap_interval
from stratego.training.warmstart_baselines import bootstrap_ratio_interval


def _rows(seed: int, games: int = 64):
    generator = np.random.default_rng(seed)
    denominators = generator.integers(4, 40, size=games).astype(float)
    return denominators, generator.random(games) * denominators


class TestPairing:
    def test_identical_arms_give_a_degenerate_interval(self):
        """The whole point of pairing: no difference means no uncertainty.

        Resampling the two arms independently would leave a wide interval
        around zero here, so this is the test that would fail if the shared
        index draw were ever split into two.
        """
        denominators, numerators = _rows(11)
        interval = paired_ratio_delta(
            numerators, denominators, numerators, denominators,
            seed=2026081307, replicates=500,
        )
        assert interval.delta == 0.0
        assert interval.lower == 0.0
        assert interval.upper == 0.0
        assert interval.sample_size == denominators.size

    def test_identical_unit_scores_give_a_degenerate_interval(self):
        scores = [1.0, 0.5, 0.0, 0.5, 1.0, 0.0]
        interval = paired_unit_delta(scores, scores, seed=20260403, replicates=500)
        assert (interval.delta, interval.lower, interval.upper) == (0.0, 0.0, 0.0)

    def test_pairing_is_tighter_than_independent_resampling(self):
        """A correlated pair of arms must produce the narrower interval."""
        denominators, reference = _rows(3, games=128)
        generator = np.random.default_rng(4)
        candidate = reference + generator.normal(0.0, 0.05, size=reference.size)

        paired = paired_ratio_delta(
            candidate, denominators, reference, denominators,
            seed=2026081307, replicates=2_000,
        )
        independent_candidate = bootstrap_ratio_interval(
            candidate, denominators, seed=2026081307, replicates=2_000
        )
        independent_reference = bootstrap_ratio_interval(
            reference, denominators, seed=2026081306, replicates=2_000
        )
        naive_width = (
            independent_candidate["upper"] - independent_candidate["lower"]
        ) + (independent_reference["upper"] - independent_reference["lower"])
        assert (paired.upper - paired.lower) < naive_width


class TestAgreementWithTheAcceptedPrimitives:
    def test_ratio_delta_against_a_constant_arm_matches_the_accepted_draw(self):
        """A zero-variance reference reduces the paired delta to the accepted
        single-arm ratio bootstrap, shifted by the reference's constant."""
        denominators, numerators = _rows(5, games=96)
        reference_numerators = denominators * 0.25  # ratio is exactly 0.25 in every resample

        paired = paired_ratio_delta(
            numerators, denominators, reference_numerators, denominators,
            seed=2026081307, replicates=1_000,
        )
        accepted = bootstrap_ratio_interval(
            numerators, denominators, seed=2026081307, replicates=1_000
        )
        assert paired.lower == pytest.approx(accepted["lower"] - 0.25, abs=1e-12)
        assert paired.upper == pytest.approx(accepted["upper"] - 0.25, abs=1e-12)
        assert paired.delta == pytest.approx(accepted["point"] - 0.25, abs=1e-12)

    def test_unit_delta_against_a_constant_arm_matches_the_accepted_draw(self):
        generator = np.random.default_rng(9)
        candidate = generator.choice([0.0, 0.5, 1.0], size=256).astype(float)
        reference = np.full(candidate.size, 0.5)

        paired = paired_unit_delta(
            candidate, reference, seed=20260403, replicates=1_000
        )
        accepted = bootstrap_interval(
            candidate.tolist(), resamples=1_000, seed=20260403
        )
        assert paired.lower == pytest.approx(accepted.lower - 0.5, abs=1e-12)
        assert paired.upper == pytest.approx(accepted.upper - 0.5, abs=1e-12)

    def test_the_chunk_size_cannot_move_a_result(self):
        denominators, numerators = _rows(13, games=32)
        generator = np.random.default_rng(14)
        candidate = numerators + generator.normal(0.0, 0.1, size=numerators.size)
        one = paired_ratio_delta(
            candidate, denominators, numerators, denominators,
            seed=2026081306, replicates=1_000, chunk=97,
        )
        two = paired_ratio_delta(
            candidate, denominators, numerators, denominators,
            seed=2026081306, replicates=1_000, chunk=500,
        )
        assert one.to_dict() | {"method": ""} == two.to_dict() | {"method": ""}


class TestMarginSemantics:
    def test_delta_max_reads_the_upper_bound(self):
        interval = paired_unit_delta(
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], seed=1, replicates=200
        )
        verdict = assess_margin("policy_ce_ratio", interval, margin=0.02,
                                direction=DIRECTION_DELTA_MAX)
        assert verdict.deciding_bound_name == "upper"
        assert verdict.non_inferior is True

    def test_delta_min_reads_the_lower_bound(self):
        interval = paired_unit_delta(
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], seed=1, replicates=200
        )
        verdict = assess_margin("vs_random_ewr", interval, margin=-0.01,
                                direction=DIRECTION_DELTA_MIN)
        assert verdict.deciding_bound_name == "lower"
        assert verdict.non_inferior is True

    def test_a_good_point_estimate_with_a_crossing_interval_fails(self):
        """The rule the contract states in one line: a point estimate never
        decides non-inferiority."""
        generator = np.random.default_rng(21)
        reference = generator.choice([0.0, 1.0], size=64).astype(float)
        candidate = generator.choice([0.0, 1.0], size=64).astype(float)
        interval = paired_unit_delta(
            candidate, reference, seed=20260403, replicates=2_000
        )
        verdict = assess_margin("vs_init_ewr", interval, margin=-0.03,
                                direction=DIRECTION_DELTA_MIN)
        assert interval.lower < -0.03 < interval.upper
        assert verdict.non_inferior is False

    def test_unknown_direction_is_refused(self):
        interval = paired_unit_delta([0.5], [0.5], seed=1, replicates=10)
        with pytest.raises(NonInferiorityError, match="unknown direction"):
            assess_margin("x", interval, margin=0.0, direction="whatever")


class TestRefusals:
    def test_misaligned_arms_are_refused(self):
        with pytest.raises(NonInferiorityError, match="misaligned"):
            paired_unit_delta([0.5, 0.5], [0.5], seed=1, replicates=10)

    def test_an_empty_sample_is_refused(self):
        with pytest.raises(NonInferiorityError, match="no observations"):
            paired_unit_delta([], [], seed=1, replicates=10)

    def test_a_zero_denominator_arm_is_refused(self):
        with pytest.raises(NonInferiorityError, match="zero denominator"):
            paired_ratio_delta([1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0],
                               seed=1, replicates=10)


def test_the_frozen_contract_constants_are_what_the_contract_says():
    assert BOOTSTRAP_REPLICATES == 10_000
    assert BOOTSTRAP_CONFIDENCE == 0.95
