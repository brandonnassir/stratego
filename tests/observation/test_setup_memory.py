"""Persistent setup-memory planes (channels 32-55).

Covers `07_observation_validation_matrix.md` section 6.
"""

import random

import numpy as np

from stratego.engine.constants import BLUE, PIECE_TYPE_BY_NAME, RED
from stratego.engine.coordinates import square_name
from stratego.engine.observation import (
    CH_KNOWN_OPPONENT_SETUP,
    CH_OWN_SETUP,
    build_observation,
)
from stratego.engine.random_play import play_random_game_to_ply
from stratego.engine.setup import random_setup, setup_squares
from stratego.engine.state import create_game
from tests.helpers import cell, full_inventory_setup, make_position, piece_at, play

OWN_SETUP_CHANNELS = list(range(CH_OWN_SETUP, CH_OWN_SETUP + 12))
KNOWN_OPPONENT_SETUP_CHANNELS = list(
    range(CH_KNOWN_OPPONENT_SETUP, CH_KNOWN_OPPONENT_SETUP + 12)
)


def test_own_setup_block_holds_exactly_forty_ones():
    state = create_game(random_setup(random.Random(1)), random_setup(random.Random(2)))
    for observer in (RED, BLUE):
        observation = build_observation(state, observer)
        assert observation[OWN_SETUP_CHANNELS].sum() == 40.0


def test_every_own_setup_square_is_represented_exactly_once():
    state = create_game(random_setup(random.Random(5)), random_setup(random.Random(6)))
    observation = build_observation(state, RED)
    block = observation[OWN_SETUP_CHANNELS]
    occupancy = block.sum(axis=0)
    for square in range(100):
        row, column = divmod(square, 10)
        expected = 1.0 if square in setup_squares(RED) else 0.0
        assert occupancy[row, column] == expected


def test_own_setup_records_the_true_type_at_the_original_square():
    state = create_game(random_setup(random.Random(9)), random_setup(random.Random(10)))
    observation = build_observation(state, RED)
    for record in state.pieces_of(RED):
        assert (
            cell(observation, CH_OWN_SETUP + record.true_type, square_name(record.starting_square))
            == 1.0
        )


def test_own_setup_planes_never_change_during_a_game():
    state = play_random_game_to_ply(4, 0)
    baseline = build_observation(state, RED)[OWN_SETUP_CHANNELS].copy()

    for ply in (10, 45, 120):
        later = play_random_game_to_ply(4, ply)
        assert np.array_equal(build_observation(later, RED)[OWN_SETUP_CHANNELS], baseline)


def test_own_setup_survives_movement_and_capture():
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e4": "marshal", "j10": "flag"},
        acting_player=RED,
    )
    captain = PIECE_TYPE_BY_NAME["captain"]
    assert cell(build_observation(state, RED), CH_OWN_SETUP + captain, "e3") == 1.0

    play(state, "e3 e4")  # captain dies attacking the marshal
    assert cell(build_observation(state, RED), CH_OWN_SETUP + captain, "e3") == 1.0


def test_opponent_setup_plane_is_empty_before_any_revelation():
    state = create_game(random_setup(random.Random(3)), random_setup(random.Random(4)))
    for observer in (RED, BLUE):
        observation = build_observation(state, observer)
        assert observation[KNOWN_OPPONENT_SETUP_CHANNELS].sum() == 0.0


def test_opponent_setup_entry_appears_at_the_original_square_after_revelation():
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e4": "marshal", "j10": "flag"},
        acting_player=RED,
    )
    marshal_piece = piece_at(state, "e4")
    marshal = PIECE_TYPE_BY_NAME["marshal"]

    assert cell(build_observation(state, RED), CH_KNOWN_OPPONENT_SETUP + marshal, "e4") == 0.0

    play(state, "e3 e4")  # combat reveals the marshal

    observation = build_observation(state, RED)
    assert cell(observation, CH_KNOWN_OPPONENT_SETUP + marshal, "e4") == 1.0
    assert marshal_piece.starting_square == 34  # e4


def test_opponent_setup_entry_stays_at_the_original_square_after_movement():
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e4": "marshal", "j10": "flag"},
        acting_player=RED,
    )
    marshal = PIECE_TYPE_BY_NAME["marshal"]
    play(state, "e3 e4", "e4 e5")

    observation = build_observation(state, RED)
    assert cell(observation, CH_KNOWN_OPPONENT_SETUP + marshal, "e4") == 1.0
    assert cell(observation, CH_KNOWN_OPPONENT_SETUP + marshal, "e5") == 0.0


def test_opponent_setup_entry_persists_after_the_piece_is_captured():
    state = make_position(
        red={"e3": "marshal", "a1": "flag", "d3": "captain"},
        blue={"e4": "captain", "j10": "flag"},
        acting_player=RED,
    )
    captain = PIECE_TYPE_BY_NAME["captain"]
    play(state, "e3 e4")  # blue captain revealed and captured in one event

    observation = build_observation(state, RED)
    assert cell(observation, CH_KNOWN_OPPONENT_SETUP + captain, "e4") == 1.0


def test_scout_revelation_also_writes_the_opponent_setup_entry():
    state = make_position(
        red={"a1": "captain", "j1": "flag"},
        blue={"a10": "scout", "j10": "flag"},
        acting_player=BLUE,
    )
    play(state, "a10 a7")
    observation = build_observation(state, RED)
    scout = PIECE_TYPE_BY_NAME["scout"]
    assert cell(observation, CH_KNOWN_OPPONENT_SETUP + scout, "a10") == 1.0


def test_opponent_setup_planes_are_identical_under_hidden_type_permutation():
    """Anti-leak case from validation-matrix section 6.2."""
    first = create_game(full_inventory_setup(), random_setup(random.Random(21)))
    second = create_game(full_inventory_setup(), random_setup(random.Random(22)))

    assert np.array_equal(
        build_observation(first, RED)[KNOWN_OPPONENT_SETUP_CHANNELS],
        build_observation(second, RED)[KNOWN_OPPONENT_SETUP_CHANNELS],
    )
