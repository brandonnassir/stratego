"""Trait-vector schema, determinism, correctness and reflection invariance."""

import random

import pytest

from stratego.engine.constants import (
    COLONEL,
    GENERAL,
    MAJOR,
    MARSHAL,
    PIECE_COUNTS,
    PIECE_RANKS,
)
from stratego.engine.setup import random_setup
from stratego.setups.identity import SetupLibraryError, reflect_canonical
from stratego.setups.traits import (
    HIGH_RANK_TYPES,
    TRAIT_NAMES,
    TRAIT_SCHEMA,
    TRAIT_SCHEMA_VERSION,
    UNCONVENTIONAL_FEATURES,
    compute_trait_vector,
    trait_schema_document,
)

from .family_fixtures import build_fixture


def _sample_setups(count: int, seed: int = 1) -> list:
    rng = random.Random(seed)
    return [random_setup(rng) for _ in range(count)]


def test_schema_version_and_field_count():
    assert TRAIT_SCHEMA_VERSION == "setup_trait_vector_v1"
    assert len(TRAIT_SCHEMA) == len(TRAIT_NAMES) == 35
    assert len(set(TRAIT_NAMES)) == 35


def test_vector_keys_match_schema_exactly_and_in_order():
    vector = compute_trait_vector(build_fixture("F00"))
    assert tuple(vector.keys()) == TRAIT_NAMES


def test_high_rank_types_come_from_the_combat_table():
    assert set(HIGH_RANK_TYPES) == {MAJOR, COLONEL, GENERAL, MARSHAL}
    assert all(PIECE_RANKS[t] >= 7 for t in HIGH_RANK_TYPES)
    assert sum(PIECE_COUNTS[t] for t in HIGH_RANK_TYPES) == 7


def test_unconventional_feature_list_is_fixed():
    assert len(UNCONVENTIONAL_FEATURES) == 8
    assert len({name for name, _ in UNCONVENTIONAL_FEATURES}) == 8


def test_vector_is_deterministic():
    for setup in _sample_setups(20):
        first = compute_trait_vector(setup)
        second = compute_trait_vector(tuple(setup))
        assert first == second  # exact equality, floats included


def test_histograms_sum_to_inventory():
    for setup in _sample_setups(30, seed=2):
        vector = compute_trait_vector(setup)
        assert sum(vector["bomb_rank_histogram"]) == 6
        assert sum(vector["scout_rank_histogram"]) == 8
        assert sum(vector["miner_rank_histogram"]) == 5
        assert sum(vector["high_rank_histogram"]) == 7
        assert vector["bomb_front2_count"] + vector["bomb_back2_count"] == 6
        assert vector["scout_front2_count"] + vector["scout_back2_count"] == 8
        assert vector["miner_front2_count"] + vector["miner_back2_count"] == 5
        assert vector["high_front2_count"] + vector["high_back2_count"] == 7
        assert (
            vector["movable_front_rank_count"] + vector["front_rank_immovable_count"]
            == 10
        )


def test_reflection_invariance_of_every_invariant_field():
    invariant_names = [f.name for f in TRAIT_SCHEMA if f.reflection_invariant]
    assert "flag_file" not in invariant_names
    for setup in _sample_setups(30, seed=3):
        original = compute_trait_vector(setup)
        mirrored = compute_trait_vector(reflect_canonical(setup))
        for name in invariant_names:
            assert original[name] == mirrored[name], name


def test_flag_file_mirrors_under_reflection():
    for setup in _sample_setups(10, seed=4):
        original = compute_trait_vector(setup)
        mirrored = compute_trait_vector(reflect_canonical(setup))
        assert mirrored["flag_file"] == 9 - original["flag_file"]


def test_known_fixture_values_by_hand():
    # F00 fixture: Flag (0,0); Bombs (0,1),(1,0),(1,1),(0,5),(1,7),(2,3);
    # Marshal (1,4); General (0,4); Spy (0,6); Scouts on ranks 2-3.
    vector = compute_trait_vector(build_fixture("F00"))
    assert vector["flag_rank"] == 0
    assert vector["flag_file"] == 0
    assert vector["flag_edge_distance"] == 0
    assert vector["flag_orth_bomb_guards"] == 2  # (0,1) and (1,0)
    assert vector["flag_diag_bomb_guards"] == 1  # (1,1)
    assert vector["flag_zone_bomb_count_r2"] == 3  # (0,1),(1,0),(1,1)
    assert vector["marshal_rank"] == 1
    assert vector["general_rank"] == 0
    assert vector["spy_rank"] == 0
    assert vector["scout_front_rank_count"] == 6
    assert vector["scout_front2_count"] == 8
    assert vector["miner_back2_count"] == 4
    assert vector["front_rank_immovable_count"] == 0
    assert vector["movable_front_rank_count"] == 10
    assert vector["open_file_movable_front_count"] == 6


def test_bomb_geometry_fields_by_hand():
    vector = compute_trait_vector(build_fixture("F00"))
    # Bombs: (0,1),(1,0),(1,1),(0,5),(1,7),(2,3)
    assert vector["bomb_rank_histogram"] == [2, 3, 1, 0]
    assert vector["bomb_front_rank_count"] == 0
    assert vector["bomb_distinct_files"] == 5  # files {1,0,1,5,7,3} -> {0,1,3,5,7}
    assert vector["bomb_adjacent_pairs"] == 2  # (0,1)-(1,1) and (1,0)-(1,1)
    # Mean pairwise Manhattan distance over the 15 bomb pairs:
    # computed independently below.
    cells = [(0, 1), (1, 0), (1, 1), (0, 5), (1, 7), (2, 3)]
    distances = [
        abs(a[0] - b[0]) + abs(a[1] - b[1])
        for i, a in enumerate(cells)
        for b in cells[i + 1 :]
    ]
    assert vector["bomb_mean_pairwise_manhattan"] == round(
        sum(distances) / len(distances), 6
    )


def test_decoy_pocket_measures_the_f05_fixture():
    vector = compute_trait_vector(build_fixture("F05"))
    # The Colonel at (0,8) sits at Manhattan 7 from the Flag with Bombs at
    # (0,7), (0,9) and (1,8).
    assert vector["decoy_pocket_bombs"] == 3


def test_entropy_fields_are_rounded_rationals():
    vector = compute_trait_vector(build_fixture("F00"))
    for name in (
        "bomb_rank_entropy_bits",
        "scout_rank_entropy_bits",
        "miner_rank_entropy_bits",
        "bomb_mean_pairwise_manhattan",
    ):
        assert vector[name] == round(vector[name], 6)
    # Scouts sit half on rank 2 and half on rank 3 in the F00 fixture... they
    # actually sit 6 front-rank / 2 rank-2, entropy of (0,0,2,6) over 8:
    import math

    expected = -(0.25 * math.log2(0.25) + 0.75 * math.log2(0.75))
    assert vector["scout_rank_entropy_bits"] == round(expected, 6)


def test_unconventional_feature_count_on_fixtures():
    conventional = compute_trait_vector(build_fixture("F14"))
    irregular = compute_trait_vector(build_fixture("F15"))
    assert conventional["unconventional_feature_count"] <= 1
    assert irregular["unconventional_feature_count"] >= 2


def test_schema_document_is_complete_and_serializable():
    import json

    document = trait_schema_document()
    assert document["trait_schema_version"] == TRAIT_SCHEMA_VERSION
    assert [field["name"] for field in document["fields"]] == list(TRAIT_NAMES)
    for field in document["fields"]:
        assert field["units"]
        assert field["description"]
        assert isinstance(field["reflection_invariant"], bool)
    json.loads(json.dumps(document))  # round-trips


def test_malformed_input_is_rejected():
    with pytest.raises(SetupLibraryError):
        compute_trait_vector([0] * 39)
    with pytest.raises(SetupLibraryError):
        compute_trait_vector([0] * 40)  # no flag present
