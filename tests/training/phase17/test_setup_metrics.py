"""Phase 17 Agent 3 section 7: the diversity measurements and their calibration."""

import numpy as np
import pytest

from stratego.engine.constants import BOMB, FLAG
from stratego.setups.diversity import LibraryEntry, per_square_entropy_bits
from stratego.setups.identity import reflect_canonical
from stratego.training.phase17.setup_contract import Phase17SetupError
from stratego.training.phase17.setup_metrics import (
    DiversityAlarms,
    distance_metrics,
    diversity_profile,
    effective_support,
    empirical_entropy_metrics,
    information_metrics,
    placement_metrics,
    prefix_entropy_metrics,
    shannon_entropy_nats,
    uniqueness_metrics,
)


def _setups(samples):
    return [sample.canonical_setup for sample in samples]


# -- primitives -------------------------------------------------------------


def test_entropy_and_effective_support_agree_on_a_known_case():
    uniform = np.full(4, 0.25)
    assert shannon_entropy_nats(uniform) == pytest.approx(np.log(4))
    assert effective_support(uniform) == pytest.approx(4.0)
    degenerate = np.array([1.0, 0.0, 0.0, 0.0])
    assert effective_support(degenerate) == pytest.approx(1.0)


def test_effective_support_is_not_a_raw_count():
    """99 boards on one square and 1 on another is support 2, effective ~1.06."""
    lopsided = np.array([0.99, 0.01])
    assert int((lopsided > 0).sum()) == 2
    assert effective_support(lopsided) < 1.1


# -- uniqueness -------------------------------------------------------------


def test_mirrored_copies_inflate_exact_uniqueness_but_not_class_uniqueness(red_samples):
    """Section 7: raw uniqueness alone is not evidence."""
    base = red_samples[0].canonical_setup
    mirrored = [base, reflect_canonical(base)]
    metrics = uniqueness_metrics(mirrored)
    assert metrics["exact_unique"] == 2
    assert metrics["reflection_class_unique"] == 1
    assert metrics["mirrored_pairs_present"] == 1


def test_a_diverse_sample_is_unique_by_both_measures(red_samples):
    metrics = uniqueness_metrics(_setups(red_samples))
    assert metrics["exact_unique"] == len(red_samples)
    assert metrics["reflection_class_unique"] == len(red_samples)
    assert metrics["reflection_class_collision_rate"] == 0.0


# -- entropy ----------------------------------------------------------------


def test_per_square_entropy_matches_the_accepted_helper(red_samples):
    """The `LibraryEntry` wrapper carries an inert label; pin it to be sure."""
    setups = _setups(red_samples)
    entries = [
        LibraryEntry(family_id="something-else", split="other", canonical=setup)
        for setup in setups
    ]
    assert empirical_entropy_metrics(setups)["per_square_entropy_bits"] == pytest.approx(
        per_square_entropy_bits(entries)
    )


def test_prefix_entropy_is_in_nats_and_bounded_by_the_legal_count(red_samples):
    probabilities = np.stack([s.behavior_probabilities for s in red_samples])
    metrics = prefix_entropy_metrics(probabilities)
    assert 0.0 < metrics["mean_prefix_entropy_nats"] < np.log(12)
    assert metrics["first_prefix_entropy_nats"] <= np.log(12) + 1e-6
    # The last prefix has exactly one legal type left, so zero entropy.
    assert metrics["per_prefix_entropy_nats"][-1] == pytest.approx(0.0, abs=1e-6)
    assert metrics["mean_sequence_entropy_nats"] == pytest.approx(
        sum(metrics["per_prefix_entropy_nats"]), abs=1e-4
    )


def test_a_wrongly_shaped_probability_block_is_refused():
    with pytest.raises(Phase17SetupError, match="expected"):
        prefix_entropy_metrics(np.zeros((3, 39, 12)))


def test_sequence_information_reports_its_spread(red_samples):
    information = np.stack([s.suffix_information_content for s in red_samples])
    metrics = information_metrics(information)
    assert metrics["sequence_information_mean_nats"] > 0.0
    assert (
        metrics["sequence_information_min_nats"]
        <= metrics["sequence_information_quantiles_nats"]["0.5"]
        <= metrics["sequence_information_max_nats"]
    )


# -- distances --------------------------------------------------------------


def test_class_distance_is_reflection_invariant(red_samples):
    from stratego.training.phase17.setup_metrics import class_distance, hamming_distance

    first, second = red_samples[0].canonical_setup, red_samples[1].canonical_setup
    assert class_distance(first, second) == class_distance(first, reflect_canonical(second))
    assert class_distance(first, second) <= hamming_distance(first, second)


def test_distance_metrics_match_a_direct_recomputation(red_samples):
    from stratego.training.phase17.setup_metrics import class_distance, hamming_distance

    setups = _setups(red_samples)[:8]
    metrics = distance_metrics(setups, sample_cap=8)
    plain = [
        hamming_distance(setups[i], setups[j])
        for i in range(len(setups))
        for j in range(i + 1, len(setups))
    ]
    folded = [
        class_distance(setups[i], setups[j])
        for i in range(len(setups))
        for j in range(i + 1, len(setups))
    ]
    assert metrics["mean_hamming"] == pytest.approx(float(np.mean(plain)))
    assert metrics["mean_class_distance"] == pytest.approx(float(np.mean(folded)))
    assert metrics["min_class_distance"] == min(folded)
    assert metrics["distance_pairs"] == len(plain)


def test_an_identical_pair_has_distance_zero(red_samples):
    base = red_samples[0].canonical_setup
    metrics = distance_metrics([base, base], sample_cap=8)
    assert metrics["min_class_distance"] == 0
    assert metrics["near_duplicate_pair_fraction"] == 1.0


def test_distances_need_two_setups(red_samples):
    with pytest.raises(Phase17SetupError, match="at least two"):
        distance_metrics([red_samples[0].canonical_setup])


# -- placement --------------------------------------------------------------


def test_flag_and_bomb_support_are_measured_and_effective(red_samples):
    metrics = placement_metrics(_setups(red_samples))
    assert metrics["flag_square_support"] >= 1
    assert metrics["flag_effective_support"] <= metrics["flag_square_support"]
    assert metrics["bomb_pattern_unique"] >= 1
    assert 0.0 < metrics["mean_top_token_concentration"] <= 1.0


def test_a_collapsed_flag_distribution_shows_effective_support_one(red_samples):
    """A collapse the raw support count would also catch -- but this is the
    quantity the stop condition is written in."""
    collapsed = []
    for sample in red_samples:
        setup = list(sample.canonical_setup)
        flag_at = setup.index(FLAG)
        setup[flag_at], setup[0] = setup[0], setup[flag_at]
        collapsed.append(tuple(setup))
    metrics = placement_metrics(collapsed)
    assert metrics["flag_square_support"] == 1
    assert metrics["flag_effective_support"] == pytest.approx(1.0)


def test_a_piece_absent_from_the_sample_is_refused_not_zeroed():
    with pytest.raises(Phase17SetupError, match="no placements"):
        placement_metrics([tuple([BOMB] * 40)])


# -- the profile and its alarms ---------------------------------------------


def test_the_profile_carries_every_section_7_measurement(red_samples):
    profile = diversity_profile(
        _setups(red_samples),
        behavior_probabilities=np.stack([s.behavior_probabilities for s in red_samples]),
        suffix_information=np.stack([s.suffix_information_content for s in red_samples]),
        label="initial",
        distance_sample_cap=24,
    )
    for name in (
        "exact_unique",
        "reflection_class_unique",
        "mean_per_square_entropy_bits",
        "mean_prefix_entropy_nats",
        "sequence_information_mean_nats",
        "mean_hamming",
        "mean_class_distance",
        "flag_effective_support",
        "bomb_pattern_unique",
        "mean_top_token_concentration",
        "reflection_class_collision_rate",
        "most_frequent_reflection_class",
    ):
        assert name in profile, name


def test_alarms_are_calibrated_against_the_baseline_not_a_library_standard(red_samples):
    profile = diversity_profile(
        _setups(red_samples),
        behavior_probabilities=np.stack([s.behavior_probabilities for s in red_samples]),
        label="initial",
        distance_sample_cap=24,
    )
    alarms = DiversityAlarms.from_baseline(profile)
    assert alarms.hard_mean_prefix_entropy_nats == pytest.approx(
        0.60 * profile["mean_prefix_entropy_nats"]
    )
    assert alarms.hard_flag_effective_support == 4.0
    assert alarms.evaluate(profile)["status"] == "ok"


def test_a_collapsed_profile_trips_the_hard_alarm(red_samples):
    profile = diversity_profile(
        _setups(red_samples),
        behavior_probabilities=np.stack([s.behavior_probabilities for s in red_samples]),
        label="initial",
        distance_sample_cap=24,
    )
    alarms = DiversityAlarms.from_baseline(profile)
    collapsed = dict(profile)
    collapsed["mean_prefix_entropy_nats"] = profile["mean_prefix_entropy_nats"] * 0.3
    collapsed["flag_effective_support"] = 1.5
    verdict = alarms.evaluate(collapsed)
    assert verdict["status"] == "hard"
    assert verdict["checks"]["mean_prefix_entropy_nats"]["status"] == "hard"
    assert verdict["checks"]["flag_effective_support"]["status"] == "hard"


def test_a_mild_drift_warns_before_it_stops(red_samples):
    profile = diversity_profile(
        _setups(red_samples),
        behavior_probabilities=np.stack([s.behavior_probabilities for s in red_samples]),
        label="initial",
        distance_sample_cap=24,
    )
    alarms = DiversityAlarms.from_baseline(profile)
    drifting = dict(profile)
    drifting["mean_prefix_entropy_nats"] = profile["mean_prefix_entropy_nats"] * 0.7
    assert alarms.evaluate(drifting)["status"] == "warning"
