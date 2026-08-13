"""Diversity metrics and thresholds: executable now, frozen before generation."""

import json
import random

from stratego.engine.setup import random_setup
from stratego.setups.diversity import (
    DIVERSITY_STANDARD_VERSION,
    DIVERSITY_THRESHOLDS_V1,
    FLAG_FOLDED_SUPPORT_MINIMUM,
    LibraryEntry,
    class_distance,
    distance_metrics,
    entropy_metrics,
    evaluate_against_thresholds,
    family_overlap_matrix,
    folded_support,
    hamming_distance,
    identity_metrics,
    per_square_entropy_bits,
    trait_diversity_metrics,
)
from stratego.setups.families import FAMILY_IDS
from stratego.setups.identity import (
    canonical_class_representative,
    reflect_canonical,
)

from .family_fixtures import build_fixture


def _synthetic_collection(
    families: "tuple[str, ...]" = ("F00", "F01", "F02"),
    per_family: int = 40,
    seed: int = 20260813,
) -> list:
    """Uniform random arrangements labelled with synthetic family/split tags.

    Content is honest random data (stored as class representatives); the
    family labels are arbitrary, which the self-satisfaction checks are
    expected to flag — that expectation is itself under test below.
    """
    rng = random.Random(seed)
    entries = []
    splits = ("train", "train", "validation", "test")  # round-robin quotas
    for family_id in families:
        for index in range(per_family):
            entries.append(
                LibraryEntry(
                    family_id=family_id,
                    split=splits[index % len(splits)],
                    canonical=canonical_class_representative(random_setup(rng)),
                )
            )
    return entries


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------


def test_hamming_and_class_distance_hand_cases():
    base = _synthetic_collection(per_family=1)[0].canonical
    assert hamming_distance(base, base) == 0
    assert class_distance(base, reflect_canonical(base)) == 0
    swapped = list(base)
    first_other = next(
        index for index, piece in enumerate(swapped) if piece != swapped[0]
    )
    swapped[0], swapped[first_other] = swapped[first_other], swapped[0]
    assert hamming_distance(base, tuple(swapped)) == 2
    assert class_distance(base, tuple(swapped)) == 2


def test_class_distance_is_symmetric_and_reflection_stable():
    entries = _synthetic_collection(per_family=3)
    for a in entries[:3]:
        for b in entries[3:6]:
            direct = class_distance(a.canonical, b.canonical)
            assert direct == class_distance(b.canonical, a.canonical)
            assert direct == class_distance(reflect_canonical(a.canonical), b.canonical)
            assert direct == class_distance(a.canonical, reflect_canonical(b.canonical))


# ---------------------------------------------------------------------------
# Identity metrics
# ---------------------------------------------------------------------------


def test_identity_metrics_are_clean_on_distinct_representatives():
    entries = _synthetic_collection()
    metrics = identity_metrics(entries)
    assert metrics["entry_count"] == len(entries)
    assert metrics["exact_duplicate_groups"] == 0
    assert metrics["reflection_class_duplicate_groups"] == 0
    assert metrics["cross_split_class_duplicate_groups"] == 0
    assert metrics["non_canonical_entries"] == 0
    assert metrics["distinct_class_fingerprints"] == len(entries)


def test_identity_metrics_detect_planted_exact_duplicate():
    entries = _synthetic_collection(per_family=5)
    entries.append(entries[0])
    metrics = identity_metrics(entries)
    assert metrics["exact_duplicate_groups"] == 1
    assert metrics["reflection_class_duplicate_groups"] == 1


def test_identity_metrics_detect_planted_reflection_duplicate():
    entries = _synthetic_collection(per_family=5)
    mirrored = reflect_canonical(entries[0].canonical)
    entries.append(LibraryEntry("F00", "train", mirrored))
    metrics = identity_metrics(entries)
    assert metrics["exact_duplicate_groups"] == 0
    assert metrics["reflection_class_duplicate_groups"] == 1
    assert metrics["non_canonical_entries"] == 1  # the mirror is not the representative


def test_identity_metrics_detect_cross_split_leakage():
    entries = _synthetic_collection(per_family=5)
    leaked = LibraryEntry("F01", "test", entries[0].canonical)
    assert entries[0].split != "test"
    entries.append(leaked)
    metrics = identity_metrics(entries)
    assert metrics["cross_split_class_duplicate_groups"] == 1


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------


def test_distance_metrics_on_random_collection_clear_the_frozen_floors():
    entries = _synthetic_collection()
    metrics = distance_metrics(entries)
    thresholds = DIVERSITY_THRESHOLDS_V1
    for family_metrics in metrics["within_family"].values():
        assert family_metrics["min_nn_distance"] >= thresholds.min_within_family_nn_distance
        assert (
            family_metrics["near_duplicate_pair_fraction"]
            <= thresholds.max_within_family_near_duplicate_fraction
        )
    assert metrics["cross_split_min_nn_distance"] >= thresholds.min_cross_split_nn_distance
    assert metrics["global_min_pairwise_distance"] >= thresholds.min_global_pairwise_distance


def test_distance_metrics_detect_a_planted_near_duplicate():
    entries = _synthetic_collection(per_family=10)
    near = list(entries[0].canonical)
    other = next(index for index, piece in enumerate(near) if piece != near[0])
    near[0], near[other] = near[other], near[0]
    entries.append(LibraryEntry(entries[0].family_id, "test", tuple(near)))
    metrics = distance_metrics(entries)
    assert metrics["cross_split_min_nn_distance"] == 2
    assert metrics["global_min_pairwise_distance"] == 2
    assert metrics["within_family"][entries[0].family_id]["min_nn_distance"] == 2


def test_distance_metrics_treat_mirrored_near_copies_as_near():
    entries = _synthetic_collection(per_family=10)
    near = list(reflect_canonical(entries[0].canonical))
    other = next(index for index, piece in enumerate(near) if piece != near[0])
    near[0], near[other] = near[other], near[0]
    entries.append(LibraryEntry(entries[0].family_id, "test", tuple(near)))
    metrics = distance_metrics(entries)
    assert metrics["cross_split_min_nn_distance"] == 2


# ---------------------------------------------------------------------------
# Entropy and support metrics
# ---------------------------------------------------------------------------


def test_per_square_entropy_is_zero_for_a_constant_collection():
    entry = _synthetic_collection(per_family=1)[0]
    constant = [entry, entry, entry]
    assert per_square_entropy_bits(constant) == [0.0] * 40


def test_per_square_entropy_counts_variation():
    entries = _synthetic_collection(per_family=30, families=("F00",))
    values = per_square_entropy_bits(entries)
    assert len(values) == 40
    assert all(0.0 <= value <= 3.585 for value in values)  # log2(12) ceiling
    assert sum(values) / len(values) > 2.0  # uniform data is highly varied


def test_folded_support_hand_case():
    from stratego.engine.constants import FLAG

    entries = _synthetic_collection(per_family=25, families=("F00",))
    support = folded_support(entries, (FLAG,))
    assert 1 <= support <= 20
    single = folded_support([entries[0]], (FLAG,))
    assert single == 1


def test_entropy_metrics_shape_and_floors_on_random_data():
    entries = _synthetic_collection()
    metrics = entropy_metrics(entries)
    thresholds = DIVERSITY_THRESHOLDS_V1
    assert set(metrics["per_family"]) == {"F00", "F01", "F02"}
    for family_id, family_metrics in metrics["per_family"].items():
        assert (
            family_metrics["mean_per_square_entropy_bits"]
            >= thresholds.min_family_mean_per_square_entropy_bits
        )
        assert (
            family_metrics["flag_folded_support"]
            >= thresholds.min_flag_folded_support[family_id]
        )
        assert family_metrics["bomb_folded_support"] >= thresholds.min_bomb_folded_support
        assert family_metrics["scout_folded_support"] >= thresholds.min_scout_folded_support
        assert family_metrics["miner_folded_support"] >= thresholds.min_miner_folded_support
        assert (
            family_metrics["high_rank_folded_support"]
            >= thresholds.min_high_rank_folded_support
        )
    assert (
        metrics["global_mean_per_square_entropy_bits"]
        >= thresholds.min_global_mean_per_square_entropy_bits
    )


# ---------------------------------------------------------------------------
# Trait diversity and overlap
# ---------------------------------------------------------------------------


def test_trait_diversity_metrics_on_random_data():
    entries = _synthetic_collection(per_family=30)
    metrics = trait_diversity_metrics(entries)
    for family_metrics in metrics["per_family"].values():
        assert family_metrics["member_count"] == 30
        assert family_metrics["distinct_trait_vectors"] >= 25
        assert family_metrics["distinct_bomb_rank_histograms"] >= 5
        assert family_metrics["distinct_scout_rank_histograms"] >= 5


def test_overlap_matrix_diagonal_is_one_for_true_fixture_members():
    entries = [
        LibraryEntry(family_id, "train", build_fixture(family_id))
        for family_id in FAMILY_IDS
    ]
    matrix = family_overlap_matrix(entries)["matrix"]
    for family_id in FAMILY_IDS:
        assert matrix[family_id][family_id] == 1.0
    # Overlap is allowed but the conventional and irregular poles exclude
    # each other by construction.
    assert matrix["F14"]["F15"] == 0.0


# ---------------------------------------------------------------------------
# Thresholds: frozen, serializable, executable
# ---------------------------------------------------------------------------


def test_thresholds_serialize_with_every_numeric_value_present():
    payload = DIVERSITY_THRESHOLDS_V1.to_dict()
    assert payload["diversity_standard_version"] == DIVERSITY_STANDARD_VERSION
    text = json.dumps(payload, sort_keys=True)
    assert json.loads(text) == json.loads(json.dumps(payload, sort_keys=True))

    def numbers(value):
        if isinstance(value, dict):
            for child in value.values():
                yield from numbers(child)
        elif isinstance(value, (int, float)):
            yield value

    assert all(value is not None for value in numbers(payload))
    identity = payload["identity"]
    assert all(value == 0 for value in identity.values())  # hard zeros
    assert set(payload["positional_support"]["min_flag_folded_support"]) == set(FAMILY_IDS)
    assert payload["distance"]["min_within_family_nn_distance"] == 6
    assert payload["distance"]["min_cross_split_nn_distance"] == 8
    assert payload["distance"]["min_global_pairwise_distance"] == 4


def test_flag_folded_floor_never_exceeds_the_family_possible_region():
    # F00 pins one folded cell; F01/F02 allow exactly two; F03-F14 allow at
    # most ten; F15 allows all twenty.
    possible = {"F00": 1, "F01": 2, "F02": 2, "F14": 5, "F15": 20}
    for family_id, floor in FLAG_FOLDED_SUPPORT_MINIMUM.items():
        assert floor <= possible.get(family_id, 10)


def test_evaluate_against_thresholds_is_executable_end_to_end():
    entries = _synthetic_collection()
    result = evaluate_against_thresholds(entries)
    assert result["diversity_standard_version"] == DIVERSITY_STANDARD_VERSION
    assert result["checks"]
    json.dumps(result, sort_keys=True)  # fully serializable
    # Honest random content clears every statistical floor; the only failing
    # checks are the self-satisfaction ones, because the synthetic labels are
    # arbitrary — exactly what the standard must detect.
    failing = [check["check"] for check in result["checks"] if not check["pass"]]
    assert failing
    assert all(check.endswith("self_satisfaction") for check in failing)
    assert result["all_pass"] is False


def test_evaluate_against_thresholds_flags_planted_duplicates():
    entries = _synthetic_collection(per_family=10)
    entries.append(entries[0])
    result = evaluate_against_thresholds(entries)
    failing = {check["check"] for check in result["checks"] if not check["pass"]}
    assert "exact_duplicate_groups" in failing
    assert "reflection_class_duplicate_groups" in failing
    assert "global_min_pairwise_distance" in failing
