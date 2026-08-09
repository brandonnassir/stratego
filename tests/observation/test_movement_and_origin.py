"""Moved-status and starting-coordinate planes (channels 26-31).

Covers `07_observation_validation_matrix.md` section 5.
"""

import pytest

from stratego.engine.constants import BLUE, RED
from stratego.engine.coordinates import normalized_coordinate, square_from_name, to_perspective
from stratego.engine.observation import (
    CH_OPPONENT_MOVED,
    CH_OPPONENT_START_COLUMN,
    CH_OPPONENT_START_ROW,
    CH_OWN_MOVED,
    CH_OWN_START_COLUMN,
    CH_OWN_START_ROW,
    build_observation,
)
from tests.helpers import cell, make_position, plane_sum, play


def test_moved_plane_is_zero_before_the_first_move():
    state = make_position(
        red={"e3": "captain", "a1": "flag"}, blue={"e7": "captain", "j10": "flag"}
    )
    observation = build_observation(state, RED)
    assert plane_sum(observation, CH_OWN_MOVED) == 0.0
    assert plane_sum(observation, CH_OPPONENT_MOVED) == 0.0


def test_moved_plane_follows_the_piece_and_persists():
    state = make_position(
        red={"e3": "captain", "a1": "flag"}, blue={"e7": "captain", "j10": "flag"}
    )
    play(state, "e3 e4")
    observation = build_observation(state, RED)
    assert cell(observation, CH_OWN_MOVED, "e4") == 1.0
    assert plane_sum(observation, CH_OWN_MOVED) == 1.0

    play(state, "e7 e6", "e4 d4")
    observation = build_observation(state, RED)
    assert cell(observation, CH_OWN_MOVED, "d4") == 1.0
    assert cell(observation, CH_OWN_MOVED, "e4") == 0.0
    assert cell(observation, CH_OPPONENT_MOVED, "e6") == 1.0


def test_moved_plane_disappears_when_the_piece_is_captured():
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e5": "marshal", "j10": "flag"},
        acting_player=RED,
    )
    play(state, "e3 e4", "e5 e4")  # blue marshal takes the moved red captain
    observation = build_observation(state, RED)
    assert plane_sum(observation, CH_OWN_MOVED) == 0.0


def test_flag_and_bomb_are_never_marked_moved():
    state = make_position(
        red={"a1": "flag", "b1": "bomb", "e3": "captain"},
        blue={"e7": "captain", "j10": "flag"},
    )
    play(state, "e3 e4", "e7 e6")
    observation = build_observation(state, RED)
    assert cell(observation, CH_OWN_MOVED, "a1") == 0.0
    assert cell(observation, CH_OWN_MOVED, "b1") == 0.0


@pytest.mark.parametrize(
    "square_name", ["a1", "e3", "j4", "a4", "j1", "c2", "h3"]
)
def test_origin_coordinates_use_the_documented_normalization(square_name):
    state = make_position(red={square_name: "captain"}, blue={"j10": "flag"})
    observation = build_observation(state, RED)
    origin = square_from_name(square_name)
    row, column = divmod(origin, 10)

    assert cell(observation, CH_OWN_START_ROW, square_name) == pytest.approx(
        normalized_coordinate(row)
    )
    assert cell(observation, CH_OWN_START_COLUMN, square_name) == pytest.approx(
        normalized_coordinate(column)
    )


def test_origin_coordinates_span_minus_one_to_plus_one():
    state = make_position(red={"a1": "captain"}, blue={"j10": "flag"})
    observation = build_observation(state, RED)
    assert cell(observation, CH_OWN_START_ROW, "a1") == pytest.approx(-1.0)
    assert cell(observation, CH_OWN_START_COLUMN, "a1") == pytest.approx(-1.0)

    # From blue's own perspective its j10 flag normalizes to the a1 corner.
    assert cell(observation, CH_OPPONENT_START_ROW, "j10") == pytest.approx(1.0)
    assert cell(observation, CH_OPPONENT_START_COLUMN, "j10") == pytest.approx(1.0)


def test_origin_travels_with_the_piece_and_never_changes():
    state = make_position(
        red={"c2": "scout", "a1": "flag"}, blue={"e7": "captain", "j10": "flag"}
    )
    origin_row, origin_column = divmod(square_from_name("c2"), 10)
    expected_row = normalized_coordinate(origin_row)
    expected_column = normalized_coordinate(origin_column)

    for move_pair in (("c2 c4", "e7 e6"), ("c4 d4", "e6 e7"), ("d4 d3", "e7 e6")):
        play(state, *move_pair)
        observation = build_observation(state, RED)
        destination = move_pair[0].split()[1]
        assert cell(observation, CH_OWN_START_ROW, destination) == pytest.approx(expected_row)
        assert cell(observation, CH_OWN_START_COLUMN, destination) == pytest.approx(
            expected_column
        )
        # Exactly one live own piece carries a nonzero origin per plane cell.
        assert cell(observation, CH_OWN_START_ROW, move_pair[0].split()[0]) == 0.0


def test_origin_disappears_when_the_piece_is_captured():
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e4": "marshal", "j10": "flag"},
        acting_player=RED,
    )
    play(state, "e3 e4")
    observation = build_observation(state, RED)
    assert cell(observation, CH_OWN_START_ROW, "e4") == 0.0
    assert cell(observation, CH_OWN_START_COLUMN, "e4") == 0.0


def test_empty_squares_and_the_other_side_hold_zero():
    state = make_position(
        red={"e3": "captain", "a1": "flag"}, blue={"e7": "captain", "j10": "flag"}
    )
    observation = build_observation(state, RED)
    assert cell(observation, CH_OWN_START_ROW, "e5") == 0.0
    assert cell(observation, CH_OWN_START_ROW, "e7") == 0.0
    assert cell(observation, CH_OPPONENT_START_ROW, "e3") == 0.0


def test_opponent_origins_are_public_even_for_hidden_pieces():
    """Origin tracking is public: it says which physical piece, not which type."""
    state = make_position(red={"e3": "captain"}, blue={"e7": "marshal", "j10": "flag"})
    observation = build_observation(state, RED)
    normalized = to_perspective(square_from_name("e7"), RED)
    row, column = divmod(normalized, 10)
    assert cell(observation, CH_OPPONENT_START_ROW, "e7") == pytest.approx(
        normalized_coordinate(row)
    )
    assert cell(observation, CH_OPPONENT_START_COLUMN, "e7") == pytest.approx(
        normalized_coordinate(column)
    )
