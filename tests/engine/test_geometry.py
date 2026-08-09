"""Static board geometry and coordinate conversion.

Covers `04_engine_validation_plan.md` section 2 and Step 1 of the Phase Two
instructions.
"""

import pytest

from stratego.engine.constants import (
    BLUE,
    BOARD_COLUMNS,
    BOARD_ROWS,
    LAKE_SQUARES,
    NUM_SQUARES,
    OCCUPIABLE_SQUARES,
    PLAYERS,
    RED,
    SETUP_SQUARES,
)
from stratego.engine.coordinates import (
    NEIGHBOURS,
    RAYS,
    is_lake,
    normalized_coordinate,
    square_from_name,
    square_index,
    square_name,
    square_row_column,
    to_perspective,
)

# The eight lake squares written out independently of the implementation, using
# the human notation defined in `01_official_rules.md` section 1.
EXPECTED_LAKE_NAMES = ("c5", "d5", "g5", "h5", "c6", "d6", "g6", "h6")


def test_board_is_ten_by_ten():
    assert BOARD_ROWS == 10
    assert BOARD_COLUMNS == 10
    assert NUM_SQUARES == 100


def test_occupiable_and_lake_counts():
    assert len(OCCUPIABLE_SQUARES) == 92
    assert len(LAKE_SQUARES) == 8
    assert len(set(OCCUPIABLE_SQUARES) | set(LAKE_SQUARES)) == 100


def test_lake_geometry_matches_expected_squares():
    expected = {square_from_name(name) for name in EXPECTED_LAKE_NAMES}
    assert set(LAKE_SQUARES) == expected


def test_lake_geometry_is_symmetric():
    # The lake mask is symmetric under a 180 degree rotation, which is what makes
    # the static lake plane identical from both perspectives.
    assert {99 - square for square in LAKE_SQUARES} == set(LAKE_SQUARES)
    # It is also left-right symmetric within each lake row.
    assert {
        square_index(row, 9 - column)
        for row, column in (square_row_column(square) for square in LAKE_SQUARES)
    } == set(LAKE_SQUARES)


@pytest.mark.parametrize("index", range(NUM_SQUARES))
def test_row_column_round_trip(index):
    row, column = square_row_column(index)
    assert square_index(row, column) == index


@pytest.mark.parametrize("index", range(NUM_SQUARES))
def test_human_name_round_trip(index):
    assert square_from_name(square_name(index)) == index


def test_human_names_are_unique_and_well_formed():
    names = [square_name(index) for index in range(NUM_SQUARES)]
    assert len(set(names)) == NUM_SQUARES
    assert names[0] == "a1"
    assert names[99] == "j10"
    assert square_name(square_index(0, 9)) == "j1"
    assert square_name(square_index(9, 0)) == "a10"


def test_name_parsing_rejects_malformed_input():
    for bad in ("", "a", "k1", "a0", "a11", "aa", "1a"):
        with pytest.raises(ValueError):
            square_from_name(bad)


@pytest.mark.parametrize("player", PLAYERS)
@pytest.mark.parametrize("index", range(NUM_SQUARES))
def test_perspective_transform_is_an_involution(player, index):
    assert to_perspective(to_perspective(index, player), player) == index


def test_red_perspective_is_the_identity():
    assert all(to_perspective(index, RED) == index for index in range(NUM_SQUARES))


def test_blue_perspective_is_a_180_degree_rotation():
    assert all(to_perspective(index, BLUE) == 99 - index for index in range(NUM_SQUARES))


@pytest.mark.parametrize("player", PLAYERS)
def test_perspective_preserves_lake_geometry(player):
    assert {to_perspective(square, player) for square in LAKE_SQUARES} == set(LAKE_SQUARES)


@pytest.mark.parametrize("player", PLAYERS)
def test_perspective_places_own_setup_rows_at_the_bottom(player):
    normalized_rows = {
        to_perspective(square, player) // BOARD_COLUMNS for square in SETUP_SQUARES[player]
    }
    assert normalized_rows == {0, 1, 2, 3}


def test_normalized_coordinate_endpoints():
    assert normalized_coordinate(0) == pytest.approx(-1.0)
    assert normalized_coordinate(9) == pytest.approx(1.0)
    assert normalized_coordinate(4) == pytest.approx(2 * 4 / 9 - 1)


def test_setup_areas_are_disjoint_and_avoid_lakes():
    red_squares = set(SETUP_SQUARES[RED])
    blue_squares = set(SETUP_SQUARES[BLUE])
    assert len(red_squares) == len(blue_squares) == 40
    assert red_squares.isdisjoint(blue_squares)
    assert red_squares.isdisjoint(set(LAKE_SQUARES))
    assert blue_squares.isdisjoint(set(LAKE_SQUARES))


def test_neighbours_never_include_lakes_or_diagonals():
    for index in range(NUM_SQUARES):
        row, column = square_row_column(index)
        for neighbour in NEIGHBOURS[index]:
            neighbour_row, neighbour_column = square_row_column(neighbour)
            distance = abs(neighbour_row - row) + abs(neighbour_column - column)
            assert distance == 1
            assert not is_lake(neighbour)


def test_corner_and_edge_neighbour_counts():
    assert len(NEIGHBOURS[square_from_name("a1")]) == 2
    assert len(NEIGHBOURS[square_from_name("j10")]) == 2
    assert len(NEIGHBOURS[square_from_name("a5")]) == 3
    assert len(NEIGHBOURS[square_from_name("e3")]) == 4
    # b5 sits directly left of the western lake, so its eastern neighbour is gone.
    assert len(NEIGHBOURS[square_from_name("b5")]) == 3


def test_rays_stop_at_lakes_and_edges():
    # Moving east from b5 immediately runs into the c5 lake.
    east_ray = RAYS[square_from_name("b5")][3]
    assert east_ray == ()
    # Moving north from a1 walks the whole a-file.
    north_ray = RAYS[square_from_name("a1")][1]
    assert [square_name(index) for index in north_ray] == [
        f"a{row}" for row in range(2, 11)
    ]
    # No ray ever contains a lake square.
    for index in range(NUM_SQUARES):
        for ray in RAYS[index]:
            assert not any(is_lake(step) for step in ray)
