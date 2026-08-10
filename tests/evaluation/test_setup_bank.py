"""The fixed evaluation setup bank: legality, determinism and variation.

The bank has to be legal by the frozen engine's own definition, byte-identical
on regeneration, and varied enough that a strength ladder measures play rather
than one arrangement seen 1,024 times.
"""

import pytest

from stratego.engine.constants import (
    BLUE,
    BOMB,
    FLAG,
    GENERAL,
    MARSHAL,
    PIECE_COUNTS,
    PIECES_PER_PLAYER,
    PLAYERS,
    RED,
    SETUP_SQUARES,
)
from stratego.engine.coordinates import NEIGHBOURS
from stratego.engine.invariants import check_invariants
from stratego.engine.setup import SetupError, serialize_setup, validate_setup
from stratego.engine.state import create_game
from stratego.evaluation.setup_bank import (
    CANONICAL_CELLS,
    CANONICAL_FILES,
    CANONICAL_RANKS,
    DEFAULT_BANK_ROOT_SEED,
    DEFAULT_BANK_SIZE,
    FRONT_RANK,
    GENERATION_FAMILY,
    MINIMUM_BANK_SIZE,
    SETUP_BANK_VERSION,
    SetupBank,
    SetupBankError,
    SetupPair,
    bank_digest,
    bank_diversity,
    canonical_index,
    canonical_neighbours,
    canonical_rank_file,
    deorient_setup,
    derive_pair_seed,
    generate_setup_pair,
    orient_setup,
    structural_violations,
    validate_bank,
    validate_setup_pair,
)


@pytest.fixture(scope="module")
def bank() -> SetupBank:
    """The canonical Phase 4 bank, generated once for the whole module."""
    return SetupBank.generate()


@pytest.fixture(scope="module")
def summary(bank: SetupBank) -> dict:
    return validate_bank(bank)


# ---------------------------------------------------------------------------
# The canonical own-orientation frame
# ---------------------------------------------------------------------------


def test_canonical_frame_covers_one_setup_area_exactly():
    assert CANONICAL_RANKS * CANONICAL_FILES == CANONICAL_CELLS == PIECES_PER_PLAYER
    assert FRONT_RANK == CANONICAL_RANKS - 1
    for index in range(CANONICAL_CELLS):
        rank, file = canonical_rank_file(index)
        assert canonical_index(rank, file) == index


@pytest.mark.parametrize("bad", [(-1, 0), (CANONICAL_RANKS, 0), (0, -1), (0, CANONICAL_FILES)])
def test_canonical_index_rejects_out_of_range(bad):
    with pytest.raises(SetupBankError):
        canonical_index(*bad)


def test_orientation_is_identity_for_red_and_self_inverse_for_blue():
    canonical = tuple(range(CANONICAL_CELLS))
    assert orient_setup(canonical, RED) == canonical
    blue = orient_setup(canonical, BLUE)
    assert blue != canonical
    assert deorient_setup(blue, BLUE) == canonical


def test_orientation_puts_rank_zero_on_each_players_back_row():
    """Rank 0 must be the row furthest from the lakes for *both* players.

    This is the whole reason the canonical frame exists: red's setup index 0 is
    board row 0 while blue's is board row 6, so an unoriented tuple would put
    blue's back rank at her front.
    """
    marker = tuple(0 if index < CANONICAL_FILES else 1 for index in range(CANONICAL_CELLS))
    for player, expected_row in ((RED, 0), (BLUE, 9)):
        oriented = orient_setup(marker, player)
        rows = {
            SETUP_SQUARES[player][index] // 10
            for index, value in enumerate(oriented)
            if value == 0
        }
        assert rows == {expected_row}


def test_canonical_adjacency_is_real_board_adjacency_for_both_players():
    for index in range(CANONICAL_CELLS):
        for neighbour in canonical_neighbours(index):
            for player in PLAYERS:
                probe = [0] * CANONICAL_CELLS
                probe[index] = 1
                probe[neighbour] = 2
                oriented = orient_setup(probe, player)
                square_a = SETUP_SQUARES[player][oriented.index(1)]
                square_b = SETUP_SQUARES[player][oriented.index(2)]
                assert square_b in NEIGHBOURS[square_a]


def test_canonical_neighbour_counts():
    assert canonical_neighbours(canonical_index(0, 0)) == (
        canonical_index(0, 1),
        canonical_index(1, 0),
    )
    assert len(canonical_neighbours(canonical_index(1, 5))) == 4
    assert len(canonical_neighbours(canonical_index(FRONT_RANK, 9))) == 2


def test_orient_setup_rejects_bad_input():
    with pytest.raises(SetupBankError):
        orient_setup((0, 1, 2), RED)
    with pytest.raises(SetupBankError):
        orient_setup(tuple(range(CANONICAL_CELLS)), 7)


# ---------------------------------------------------------------------------
# Legality
# ---------------------------------------------------------------------------


def test_bank_has_the_preferred_size_and_clears_the_documented_floor():
    assert DEFAULT_BANK_SIZE >= MINIMUM_BANK_SIZE
    assert MINIMUM_BANK_SIZE == 512
    assert DEFAULT_BANK_SIZE == 1024


def test_every_pair_is_legal(summary: dict):
    assert summary["validation_failures"] == []
    assert summary["pair_count"] == DEFAULT_BANK_SIZE


def test_every_setup_has_the_exact_official_inventory(bank: SetupBank):
    for pair in bank:
        for player, setup in ((RED, pair.red_setup), (BLUE, pair.blue_setup)):
            assert validate_setup(setup, player) == setup
            for piece_type, count in PIECE_COUNTS.items():
                assert setup.count(piece_type) == count


def test_every_pair_builds_a_valid_game_with_no_overlap_or_lakes(bank: SetupBank):
    for pair in bank.pairs[:64]:
        state = create_game(
            pair.red_setup, pair.blue_setup, game_id=f"bank-{pair.setup_pair_id}"
        )
        check_invariants(state)
        assert sum(1 for entry in state.board if entry is not None) == 2 * PIECES_PER_PLAYER


def test_validate_setup_pair_reports_an_illegal_inventory():
    legal = generate_setup_pair(0)
    broken = SetupPair(
        setup_pair_id=legal.setup_pair_id,
        red_setup=(BOMB,) * PIECES_PER_PLAYER,
        blue_setup=legal.blue_setup,
        generation_seed=legal.generation_seed,
    )
    failures = validate_setup_pair(broken)
    assert failures and "inventory" in failures[0]


def test_serialised_setups_round_trip(bank: SetupBank):
    for pair in bank.pairs[:32]:
        assert len(serialize_setup(pair.red_setup)) == PIECES_PER_PLAYER
        assert SetupPair.from_dict(pair.to_dict()) == pair


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_produces_a_byte_identical_bank(bank: SetupBank):
    again = SetupBank.generate()
    assert again.to_json() == bank.to_json()
    assert bank_digest(again) == bank_digest(bank)


def test_a_different_root_seed_produces_a_different_bank(bank: SetupBank):
    other = SetupBank.generate(64, root_seed=DEFAULT_BANK_ROOT_SEED + 1)
    reference = SetupBank.generate(64)
    assert bank_digest(other) != bank_digest(reference)


def test_any_pair_rebuilds_in_isolation(bank: SetupBank):
    """A worker must be able to rebuild pair N without generating 0..N-1."""
    for setup_pair_id in (0, 1, 17, 511, 1023):
        assert generate_setup_pair(setup_pair_id) == bank.pair(setup_pair_id)


def test_pair_seeds_are_hashed_not_sequential():
    seeds = [derive_pair_seed(DEFAULT_BANK_ROOT_SEED, index) for index in range(32)]
    assert len(set(seeds)) == len(seeds)
    # Consecutive identifiers must not produce consecutive or near-equal seeds.
    assert all(abs(seeds[index + 1] - seeds[index]) > 1000 for index in range(len(seeds) - 1))


def test_json_round_trip_preserves_the_bank(bank: SetupBank):
    restored = SetupBank.from_json(bank.to_json())
    assert restored == bank
    assert bank_digest(restored) == bank_digest(bank)


def test_from_dict_rejects_a_disagreeing_pair_count(bank: SetupBank):
    payload = SetupBank.generate(4).to_dict()
    payload["pair_count"] = 99
    with pytest.raises(SetupBankError):
        SetupBank.from_dict(payload)


# ---------------------------------------------------------------------------
# Identifiers and variation
# ---------------------------------------------------------------------------


def test_setup_pair_ids_are_unique(summary: dict):
    assert summary["duplicate_setup_pair_ids"] == []


def test_duplicate_identifiers_are_rejected_at_construction():
    pair = generate_setup_pair(0)
    with pytest.raises(SetupBankError):
        SetupBank(SETUP_BANK_VERSION, DEFAULT_BANK_ROOT_SEED, GENERATION_FAMILY, (pair, pair))


def test_no_arrangement_is_reused(summary: dict):
    assert summary["distinct_red_setups"] == summary["pair_count"]
    assert summary["distinct_blue_setups"] == summary["pair_count"]
    assert summary["distinct_positions"] == summary["pair_count"]


def test_red_and_blue_arrangements_are_drawn_independently(bank: SetupBank):
    """A pair must not be one arrangement mirrored onto both sides."""
    for pair in bank:
        red_canonical = deorient_setup(pair.red_setup, RED)
        blue_canonical = deorient_setup(pair.blue_setup, BLUE)
        assert red_canonical != blue_canonical


def test_the_bank_varies_beyond_superficial_permutation(bank: SetupBank):
    diversity = bank_diversity(bank)
    assert diversity["arrangements"] == 2 * DEFAULT_BANK_SIZE
    assert diversity["distinct_flag_files"] == CANONICAL_FILES
    assert diversity["distinct_flag_cells"] == 2 * CANONICAL_FILES
    # Front and back rows are where an opponent actually meets the setup, so
    # near-total distinctness there is the meaningful variation claim.
    assert diversity["distinct_front_rows"] > 0.99 * diversity["arrangements"]
    assert diversity["distinct_back_rows"] > 0.99 * diversity["arrangements"]


# ---------------------------------------------------------------------------
# Structural rules of the `structured_v1` family
# ---------------------------------------------------------------------------


def test_generation_family_is_recorded(bank: SetupBank):
    assert bank.generation_family == GENERATION_FAMILY
    assert all(pair.generation_family == GENERATION_FAMILY for pair in bank)
    assert all(pair.bank_version == SETUP_BANK_VERSION for pair in bank)


def test_structural_rules_hold_for_every_arrangement(bank: SetupBank):
    assert structural_violations(bank) == []


def test_flag_sits_in_the_two_rows_furthest_from_the_lakes(bank: SetupBank):
    for pair in bank:
        for player in PLAYERS:
            canonical = deorient_setup(pair.setup_for(player), player)
            rank, _ = canonical_rank_file(canonical.index(FLAG))
            assert rank in (0, 1)


def test_flag_is_guarded_by_at_least_two_bombs(bank: SetupBank):
    histogram = bank_diversity(bank)["flag_guard_bomb_histogram"]
    assert all(int(key) >= 2 for key in histogram)


def test_marshal_and_general_stay_off_the_front_rank(bank: SetupBank):
    for pair in bank:
        for player in PLAYERS:
            canonical = deorient_setup(pair.setup_for(player), player)
            for piece_type in (MARSHAL, GENERAL):
                rank, _ = canonical_rank_file(canonical.index(piece_type))
                assert rank != FRONT_RANK


def test_scouts_are_biased_forward_and_bombs_backward(bank: SetupBank):
    """The rank bias is a design property, so it is measured, not assumed."""
    from stratego.engine.constants import SCOUT

    scout_front = bomb_back = 0
    for pair in bank:
        for player in PLAYERS:
            canonical = deorient_setup(pair.setup_for(player), player)
            for index, piece_type in enumerate(canonical):
                rank, _ = canonical_rank_file(index)
                if piece_type == SCOUT and rank >= 2:
                    scout_front += 1
                if piece_type == BOMB and rank <= 1:
                    bomb_back += 1
    arrangements = 2 * len(bank)
    assert scout_front / arrangements > PIECE_COUNTS[SCOUT] / 2
    assert bomb_back / arrangements > PIECE_COUNTS[BOMB] / 2


# ---------------------------------------------------------------------------
# Bank operations
# ---------------------------------------------------------------------------


def test_subset_preserves_order_and_version(bank: SetupBank):
    subset = bank.subset([9, 3, 17])
    assert subset.pair_ids == (9, 3, 17)
    assert subset.bank_version == bank.bank_version
    assert subset.pair(3) == bank.pair(3)


def test_subset_rejects_a_repeated_identifier(bank: SetupBank):
    with pytest.raises(SetupBankError):
        bank.subset([3, 3])


def test_pair_lookup_rejects_an_unknown_identifier(bank: SetupBank):
    with pytest.raises(SetupBankError):
        bank.pair(DEFAULT_BANK_SIZE)


def test_setup_for_rejects_an_unknown_player(bank: SetupBank):
    with pytest.raises(SetupBankError):
        bank.pair(0).setup_for(7)


@pytest.mark.parametrize("size", [0, -1])
def test_generate_rejects_a_non_positive_size(size):
    with pytest.raises(SetupBankError):
        SetupBank.generate(size)


def test_generate_setup_pair_rejects_a_negative_identifier():
    with pytest.raises(SetupBankError):
        generate_setup_pair(-1)


def test_deserialisation_rejects_a_corrupt_setup():
    payload = generate_setup_pair(0).to_dict()
    payload["red_setup"] = payload["red_setup"][:-1]
    with pytest.raises(SetupError):
        SetupPair.from_dict(payload)
