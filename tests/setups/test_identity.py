"""Canonical frame, reflection, canonicalization, fingerprint and seed tests."""

import random

import pytest

from stratego.engine.constants import BLUE, FLAG, PIECE_COUNTS, PLAYERS, RED
from stratego.engine.setup import random_setup, validate_setup
from stratego.evaluation import setup_bank as phase4_bank
from stratego.setups.identity import (
    CANONICAL_CELLS,
    CANONICAL_FILES,
    CANONICAL_RANKS,
    SetupLibraryError,
    canonical_class_representative,
    canonical_index,
    canonical_neighbours,
    canonical_rank_file,
    class_fingerprint,
    content_fingerprint,
    deorient_setup,
    derive_attempt_seed,
    derive_base_seed,
    derive_stream_seed,
    edge_file_distance,
    is_canonical_representative,
    orient_setup,
    reflect_canonical,
)


def _sample_setups(count: int, seed: int = 20260813) -> list:
    rng = random.Random(seed)
    return [random_setup(rng) for _ in range(count)]


# ---------------------------------------------------------------------------
# The canonical frame equals the accepted Phase 4 convention
# ---------------------------------------------------------------------------


def test_canonical_frame_constants_match_phase4():
    assert CANONICAL_RANKS == phase4_bank.CANONICAL_RANKS
    assert CANONICAL_FILES == phase4_bank.CANONICAL_FILES
    assert CANONICAL_CELLS == phase4_bank.CANONICAL_CELLS


def test_canonical_index_and_inverse_match_phase4_exhaustively():
    for rank in range(CANONICAL_RANKS):
        for file in range(CANONICAL_FILES):
            index = canonical_index(rank, file)
            assert index == phase4_bank.canonical_index(rank, file)
            assert canonical_rank_file(index) == (rank, file)
            assert canonical_rank_file(index) == phase4_bank.canonical_rank_file(index)


def test_canonical_neighbours_match_phase4_exhaustively():
    for index in range(CANONICAL_CELLS):
        assert canonical_neighbours(index) == phase4_bank.canonical_neighbours(index)


def test_orientation_matches_phase4_for_both_players():
    for setup in _sample_setups(20):
        for player in PLAYERS:
            assert orient_setup(setup, player) == phase4_bank.orient_setup(setup, player)
            assert deorient_setup(orient_setup(setup, player), player) == tuple(setup)


def test_orientation_is_identity_for_red_and_self_inverse_for_blue():
    setup = tuple(range(4)) * 10
    assert orient_setup(setup, RED) == setup
    oriented = orient_setup(setup, BLUE)
    assert oriented != setup
    assert orient_setup(oriented, BLUE) == setup


@pytest.mark.parametrize("bad", [(-1, 0), (4, 0), (0, -1), (0, 10)])
def test_canonical_index_rejects_out_of_range(bad):
    with pytest.raises(SetupLibraryError):
        canonical_index(*bad)


def test_orient_rejects_unknown_player_and_bad_length():
    with pytest.raises(SetupLibraryError):
        orient_setup([0] * CANONICAL_CELLS, 5)
    with pytest.raises(SetupLibraryError):
        orient_setup([0] * 39, RED)


# ---------------------------------------------------------------------------
# Edge distance
# ---------------------------------------------------------------------------


def test_edge_file_distance_is_reflection_invariant_and_bounded():
    for file in range(CANONICAL_FILES):
        assert edge_file_distance(file) == edge_file_distance(9 - file)
        assert 0 <= edge_file_distance(file) <= 4
    assert edge_file_distance(0) == 0
    assert edge_file_distance(9) == 0
    assert edge_file_distance(4) == 4
    assert edge_file_distance(5) == 4
    with pytest.raises(SetupLibraryError):
        edge_file_distance(10)


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------


def test_reflection_is_an_involution():
    for setup in _sample_setups(50):
        assert reflect_canonical(reflect_canonical(setup)) == tuple(setup)


def test_reflection_mirrors_files_within_each_rank():
    for setup in _sample_setups(10):
        reflected = reflect_canonical(setup)
        for rank in range(CANONICAL_RANKS):
            for file in range(CANONICAL_FILES):
                assert (
                    reflected[canonical_index(rank, file)]
                    == setup[canonical_index(rank, 9 - file)]
                )


def test_reflection_preserves_inventory():
    for setup in _sample_setups(10):
        assert sorted(reflect_canonical(setup)) == sorted(setup)
        validate_setup(reflect_canonical(setup), RED)


def test_no_legal_setup_equals_its_own_reflection():
    # The single Flag cannot occupy both file f and file 9-f, so the pinned
    # representative examples and a broad random sample must all differ.
    for setup in _sample_setups(200):
        assert reflect_canonical(setup) != tuple(setup)


def test_reflection_pins_corner_and_centre_files():
    # Representative examples required by the Phase 7 common contract:
    # file 0 <-> file 9 (left corner <-> right corner), file 4 <-> file 5.
    setup = list(_sample_setups(1)[0])
    flag_index = setup.index(FLAG)
    setup[flag_index], setup[0] = setup[0], setup[flag_index]
    reflected = reflect_canonical(tuple(setup))
    assert reflected[canonical_index(0, 9)] == FLAG
    centre = _sample_setups(1, seed=99)[0]
    reflected_centre = reflect_canonical(centre)
    assert reflected_centre[canonical_index(2, 5)] == centre[canonical_index(2, 4)]
    assert reflected_centre[canonical_index(2, 4)] == centre[canonical_index(2, 5)]


def test_reflection_preserves_flag_bomb_adjacency_structure():
    from stratego.setups.traits import compute_trait_vector

    for setup in _sample_setups(20, seed=7):
        original = compute_trait_vector(setup)
        mirrored = compute_trait_vector(reflect_canonical(setup))
        assert original["flag_orth_bomb_guards"] == mirrored["flag_orth_bomb_guards"]
        assert original["flag_rank"] == mirrored["flag_rank"]
        assert original["bomb_rank_histogram"] == mirrored["bomb_rank_histogram"]


# ---------------------------------------------------------------------------
# Canonical class representative
# ---------------------------------------------------------------------------


def test_representative_is_shared_by_both_class_members():
    for setup in _sample_setups(100):
        representative = canonical_class_representative(setup)
        assert representative == canonical_class_representative(reflect_canonical(setup))
        assert representative in (tuple(setup), reflect_canonical(setup))


def test_representative_is_idempotent_and_lexicographically_minimal():
    for setup in _sample_setups(100, seed=3):
        representative = canonical_class_representative(setup)
        assert canonical_class_representative(representative) == representative
        assert representative <= reflect_canonical(representative)
        assert is_canonical_representative(representative)
        # Exactly one member of every class is the representative.
        other = reflect_canonical(representative)
        assert not is_canonical_representative(other)


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def test_class_fingerprint_is_reflection_invariant_and_deterministic():
    for setup in _sample_setups(50, seed=11):
        fingerprint = class_fingerprint(setup)
        assert fingerprint == class_fingerprint(reflect_canonical(setup))
        assert fingerprint == class_fingerprint(setup)  # recompute, same value
        assert len(fingerprint) == 64
        int(fingerprint, 16)  # valid hex


def test_content_fingerprint_distinguishes_orientations():
    for setup in _sample_setups(20, seed=13):
        assert content_fingerprint(setup) != content_fingerprint(reflect_canonical(setup))


def test_fingerprints_differ_between_different_setups():
    setups = _sample_setups(50, seed=17)
    assert len({class_fingerprint(setup) for setup in setups}) == len(setups)


def test_fingerprint_rejects_illegal_inventory():
    with pytest.raises(Exception):
        class_fingerprint([FLAG] * CANONICAL_CELLS)


def test_fingerprint_matches_pinned_golden_value():
    # The all-sorted inventory arrangement is a stable cross-process anchor:
    # any change to serialization, domain prefix or canonicalization moves it.
    pieces: list[int] = []
    for piece_type, count in sorted(PIECE_COUNTS.items()):
        pieces.extend([piece_type] * count)
    setup = tuple(pieces)
    assert class_fingerprint(setup) == class_fingerprint(reflect_canonical(setup))
    assert (
        content_fingerprint(setup)
        == "97d88a98ac06937345a65a9e0f58f14b325289175280bbd807c6e94495e037e8"
    )
    assert (
        class_fingerprint(setup)
        == "fb826cfbddb90b03fbaa7b62e71c09bb0ba2b9616f10ad896448736cd01f7329"
    )


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def test_base_seed_is_deterministic_and_isolated():
    seed_a = derive_base_seed("c1", "l1", 123, "F00", 0)
    assert seed_a == derive_base_seed("c1", "l1", 123, "F00", 0)
    # every identity component changes the stream
    assert seed_a != derive_base_seed("c2", "l1", 123, "F00", 0)
    assert seed_a != derive_base_seed("c1", "l2", 123, "F00", 0)
    assert seed_a != derive_base_seed("c1", "l1", 124, "F00", 0)
    assert seed_a != derive_base_seed("c1", "l1", 123, "F01", 0)
    assert seed_a != derive_base_seed("c1", "l1", 123, "F00", 1)


def test_base_seeds_are_hashed_not_sequential():
    seeds = [
        derive_base_seed("c1", "l1", 123, "F00", index) for index in range(4)
    ]
    deltas = {seeds[i + 1] - seeds[i] for i in range(3)}
    assert len(deltas) == 3  # arithmetic progressions would collapse the set


def test_attempt_seed_streams_are_distinct():
    base = derive_base_seed("c1", "l1", 123, "F00", 0)
    attempts = [derive_attempt_seed(base, attempt) for attempt in range(8)]
    assert len(set(attempts)) == 8
    assert derive_attempt_seed(base, 0) == attempts[0]


def test_stream_seed_separates_purposes():
    assert derive_stream_seed("perturbation", 1) != derive_stream_seed("reflection", 1)
    with pytest.raises(SetupLibraryError):
        derive_stream_seed("")


def test_seed_derivation_rejects_negative_indices():
    with pytest.raises(SetupLibraryError):
        derive_base_seed("c1", "l1", 123, "F00", -1)
    with pytest.raises(SetupLibraryError):
        derive_attempt_seed(1, -1)


def test_seed_streams_do_not_collide_with_phase4_bank_streams():
    # Same numeric inputs, different personalization: the library can never
    # replay the frozen evaluation bank's randomness.
    assert derive_stream_seed("pair", 20260101, 0) != phase4_bank.derive_pair_seed(
        20260101, 0
    )
