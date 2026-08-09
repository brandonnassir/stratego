"""Setup inventory and legality validation.

Covers `04_engine_validation_plan.md` section 3, `02_project_ruleset.md`
section 6 and Step 3 of the Phase Two instructions.
"""

import random
from collections import Counter

import pytest

from stratego.engine.constants import (
    BLUE,
    LAKE_SQUARES,
    PIECE_COUNTS,
    PIECE_TYPE_BY_NAME,
    PIECE_TYPE_NAMES,
    PIECES_PER_PLAYER,
    PLAYERS,
    RED,
    SETUP_SQUARES,
)
from stratego.engine.coordinates import square_from_name
from stratego.engine.setup import (
    SetupError,
    deserialize_setup,
    random_setup,
    reflect_setup,
    serialize_setup,
    setup_squares,
    setup_to_placements,
    validate_setup,
    validate_setup_placement,
)
from stratego.engine.state import create_game
from tests.helpers import full_inventory_setup

# The official inventory transcribed independently from `01_official_rules.md`
# section 2, so the test does not simply re-read the implementation's table.
OFFICIAL_INVENTORY = {
    "flag": 1,
    "spy": 1,
    "scout": 8,
    "miner": 5,
    "sergeant": 4,
    "lieutenant": 4,
    "captain": 4,
    "major": 3,
    "colonel": 2,
    "general": 1,
    "marshal": 1,
    "bomb": 6,
}


def test_official_inventory_totals_forty():
    assert sum(OFFICIAL_INVENTORY.values()) == 40
    assert PIECES_PER_PLAYER == 40


@pytest.mark.parametrize("type_name,expected", sorted(OFFICIAL_INVENTORY.items()))
def test_engine_inventory_matches_official_counts(type_name, expected):
    assert PIECE_COUNTS[PIECE_TYPE_BY_NAME[type_name]] == expected


@pytest.mark.parametrize("player", PLAYERS)
def test_setup_squares_are_the_players_four_nearest_rows(player):
    squares = setup_squares(player)
    assert len(squares) == 40
    assert len(set(squares)) == 40
    rows = sorted({index // 10 for index in squares})
    assert rows == ([0, 1, 2, 3] if player == RED else [6, 7, 8, 9])
    assert not set(squares) & set(LAKE_SQUARES)


@pytest.mark.parametrize("player", PLAYERS)
def test_valid_setup_is_accepted(player):
    setup = full_inventory_setup()
    assert validate_setup(setup, player) == setup


def test_setup_with_wrong_length_is_rejected():
    setup = full_inventory_setup()
    with pytest.raises(SetupError, match="exactly 40 pieces"):
        validate_setup(setup[:-1], RED)
    with pytest.raises(SetupError, match="exactly 40 pieces"):
        validate_setup(setup + (PIECE_TYPE_BY_NAME["scout"],), RED)


@pytest.mark.parametrize("type_name", sorted(OFFICIAL_INVENTORY))
def test_every_incorrect_count_class_is_rejected(type_name):
    """One rejection test per piece type, as required by the validation plan."""
    piece_type = PIECE_TYPE_BY_NAME[type_name]
    substitute = PIECE_TYPE_BY_NAME["scout" if type_name != "scout" else "miner"]

    setup = list(full_inventory_setup())
    index = setup.index(piece_type)
    setup[index] = substitute  # one too few of `type_name`, one too many of the other

    with pytest.raises(SetupError, match="inventory mismatch"):
        validate_setup(tuple(setup), RED)


def test_unknown_piece_type_is_rejected():
    setup = list(full_inventory_setup())
    setup[0] = 99
    with pytest.raises(SetupError, match="unknown piece type"):
        validate_setup(tuple(setup), RED)
    setup[0] = "scout"
    with pytest.raises(SetupError, match="unknown piece type"):
        validate_setup(tuple(setup), RED)


def test_valid_placement_map_is_accepted():
    placements = setup_to_placements(full_inventory_setup(), RED)
    assert validate_setup_placement(placements, RED) == full_inventory_setup()


def test_placement_on_a_lake_is_rejected():
    placements = setup_to_placements(full_inventory_setup(), RED)
    victim = sorted(placements)[0]
    placements[LAKE_SQUARES[0]] = placements.pop(victim)
    with pytest.raises(SetupError, match="lake square"):
        validate_setup_placement(placements, RED)


def test_placement_outside_the_setup_area_is_rejected():
    placements = setup_to_placements(full_inventory_setup(), RED)
    victim = sorted(placements)[0]
    # e5 is empty middle ground, legal to stand on but not a red setup square.
    placements[square_from_name("e5")] = placements.pop(victim)
    with pytest.raises(SetupError, match="outside the red setup area"):
        validate_setup_placement(placements, RED)


def test_placement_in_the_opponents_area_is_rejected():
    placements = setup_to_placements(full_inventory_setup(), RED)
    victim = sorted(placements)[0]
    placements[SETUP_SQUARES[BLUE][0]] = placements.pop(victim)
    with pytest.raises(SetupError, match="outside the red setup area"):
        validate_setup_placement(placements, RED)


def test_placement_leaving_a_square_empty_is_rejected():
    placements = setup_to_placements(full_inventory_setup(), RED)
    placements.pop(sorted(placements)[0])
    with pytest.raises(SetupError, match="exactly 40 pieces"):
        validate_setup_placement(placements, RED)


def test_placement_with_wrong_inventory_is_rejected():
    placements = setup_to_placements(full_inventory_setup(), RED)
    flag_square = next(
        square
        for square, piece_type in placements.items()
        if piece_type == PIECE_TYPE_BY_NAME["flag"]
    )
    placements[flag_square] = PIECE_TYPE_BY_NAME["bomb"]
    with pytest.raises(SetupError, match="inventory mismatch"):
        validate_setup_placement(placements, RED)


def test_invalid_setup_is_never_silently_repaired():
    setup = list(full_inventory_setup())
    setup[0] = PIECE_TYPE_BY_NAME["bomb"]
    with pytest.raises(SetupError):
        create_game(tuple(setup), full_inventory_setup())


@pytest.mark.parametrize("seed", range(25))
def test_random_setup_is_always_legal(seed):
    setup = random_setup(random.Random(seed))
    validate_setup(setup, RED)
    assert Counter(setup) == {
        PIECE_TYPE_BY_NAME[name]: count for name, count in OFFICIAL_INVENTORY.items()
    }


def test_random_setup_is_deterministic_under_a_seed():
    assert random_setup(random.Random(4)) == random_setup(random.Random(4))
    assert random_setup(random.Random(4)) != random_setup(random.Random(5))


def test_reflection_mirrors_each_row_and_is_an_involution():
    setup = random_setup(random.Random(11))
    reflected = reflect_setup(setup)
    assert reflect_setup(reflected) == setup
    for row_start in range(0, 40, 10):
        assert reflected[row_start : row_start + 10] == tuple(
            reversed(setup[row_start : row_start + 10])
        )
    validate_setup(reflected, RED)


def test_serialisation_round_trip():
    setup = random_setup(random.Random(3))
    text = serialize_setup(setup)
    assert len(text) == 40
    assert deserialize_setup(text) == setup


def test_deserialisation_rejects_malformed_text():
    with pytest.raises(SetupError):
        deserialize_setup("too short")
    with pytest.raises(SetupError):
        deserialize_setup("Z" * 40)


def test_new_game_places_forty_pieces_per_player_on_their_setup_squares():
    state = create_game(full_inventory_setup(), full_inventory_setup())
    for player in PLAYERS:
        records = state.pieces_of(player)
        assert len(records) == 40
        assert all(record.alive for record in records)
        assert {record.current_square for record in records} == set(SETUP_SQUARES[player])
        counts = Counter(PIECE_TYPE_NAMES[record.true_type] for record in records)
        assert counts == OFFICIAL_INVENTORY


def test_piece_identifiers_are_stable_and_type_independent():
    state_a = create_game(full_inventory_setup(), full_inventory_setup())
    shuffled = random_setup(random.Random(99))
    state_b = create_game(shuffled, full_inventory_setup())

    # Same physical setup slots receive the same identifiers regardless of the
    # types placed there (`04_engine_validation_plan.md` section 21.2).
    for record_a, record_b in zip(state_a.pieces, state_b.pieces):
        assert record_a.piece_id == record_b.piece_id
        assert record_a.owner == record_b.owner
        assert record_a.starting_square == record_b.starting_square
