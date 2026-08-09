"""Identity, disclosure and hidden-occupancy planes (channels 0-25).

Covers `07_observation_validation_matrix.md` section 4.
"""

import pytest

from stratego.engine.constants import BLUE, PIECE_TYPE_BY_NAME, PIECE_TYPE_NAMES, RED
from stratego.engine.observation import (
    CH_HIDDEN_OPPONENT_OCCUPANCY,
    CH_KNOWN_OPPONENT_IDENTITY,
    CH_KNOWN_OPPONENT_SETUP,
    CH_OWN_IDENTITY,
    CH_OWN_KNOWN_TO_OPPONENT,
    build_observation,
)
from tests.helpers import cell, make_position, piece_at, plane_sum, play

ALL_TYPES = sorted(PIECE_TYPE_BY_NAME)
MOVABLE_TYPES = [name for name in ALL_TYPES if name not in ("flag", "bomb")]


@pytest.mark.parametrize("type_name", ALL_TYPES)
def test_own_piece_appears_in_exactly_one_identity_plane(type_name):
    state = make_position(red={"e3": type_name}, blue={"j10": "flag"})
    observation = build_observation(state, RED)
    piece_type = PIECE_TYPE_BY_NAME[type_name]

    assert cell(observation, CH_OWN_IDENTITY + piece_type, "e3") == 1.0
    for other in range(12):
        expected = 1.0 if other == piece_type else 0.0
        assert cell(observation, CH_OWN_IDENTITY + other, "e3") == expected
    # Exactly one living own piece, so exactly one `1` across the whole block.
    assert sum(plane_sum(observation, CH_OWN_IDENTITY + other) for other in range(12)) == 1.0


@pytest.mark.parametrize("type_name", MOVABLE_TYPES)
def test_identity_plane_follows_the_piece_when_it_moves(type_name):
    state = make_position(red={"e3": type_name}, blue={"j10": "flag"}, acting_player=RED)
    piece_type = PIECE_TYPE_BY_NAME[type_name]
    play(state, "e3 e4")

    observation = build_observation(state, RED)
    assert cell(observation, CH_OWN_IDENTITY + piece_type, "e3") == 0.0
    assert cell(observation, CH_OWN_IDENTITY + piece_type, "e4") == 1.0


@pytest.mark.parametrize("type_name", [name for name in ALL_TYPES if name != "bomb"])
def test_captured_own_piece_leaves_the_identity_planes(type_name):
    # A blue Marshal attacking removes every red piece except a Bomb: it
    # outranks everything, and the Marshal-versus-Marshal case removes both.
    state = make_position(
        red={"e3": type_name, "a1": "scout"},
        blue={"e4": "marshal", "j10": "flag"},
        acting_player=BLUE,
    )
    piece_type = PIECE_TYPE_BY_NAME[type_name]
    play(state, "e4 e3")

    observation = build_observation(state, RED)
    assert cell(observation, CH_OWN_IDENTITY + piece_type, "e3") == 0.0
    assert cell(observation, CH_OWN_IDENTITY + piece_type, "e4") == 0.0


def test_captured_own_bomb_leaves_the_identity_planes():
    state = make_position(
        red={"e3": "bomb", "a1": "scout"},
        blue={"e4": "miner", "j10": "flag"},
        acting_player=BLUE,
    )
    play(state, "e4 e3")

    observation = build_observation(state, RED)
    bomb = PIECE_TYPE_BY_NAME["bomb"]
    assert cell(observation, CH_OWN_IDENTITY + bomb, "e3") == 0.0


def test_hidden_opponent_occupies_channel_24_and_no_identity_plane():
    state = make_position(red={"e3": "captain"}, blue={"e6": "marshal", "j10": "flag"})
    observation = build_observation(state, RED)

    assert cell(observation, CH_HIDDEN_OPPONENT_OCCUPANCY, "e6") == 1.0
    for piece_type in range(12):
        assert cell(observation, CH_KNOWN_OPPONENT_IDENTITY + piece_type, "e6") == 0.0


def test_channel_24_and_the_known_planes_never_overlap():
    state = make_position(
        red={"e3": "captain"},
        blue={"e6": "marshal", "f6": "scout", "j10": "flag"},
        revealed={"f6"},
    )
    observation = build_observation(state, RED)
    for square_name in ("e6", "f6", "j10"):
        hidden = cell(observation, CH_HIDDEN_OPPONENT_OCCUPANCY, square_name)
        known = sum(
            cell(observation, CH_KNOWN_OPPONENT_IDENTITY + piece_type, square_name)
            for piece_type in range(12)
        )
        assert hidden + known == 1.0


def test_revealing_an_opponent_clears_channel_24_and_sets_the_type_plane():
    state = make_position(
        red={"e3": "marshal", "a1": "flag"},
        blue={"e4": "captain", "j10": "flag"},
        acting_player=RED,
    )
    defender = piece_at(state, "e4")
    before = build_observation(state, RED)
    assert cell(before, CH_HIDDEN_OPPONENT_OCCUPANCY, "e4") == 1.0

    # A losing defender is captured, so use a defender that survives instead.
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e4": "marshal", "j10": "flag"},
        acting_player=RED,
    )
    defender = piece_at(state, "e4")
    play(state, "e3 e4")

    after = build_observation(state, RED)
    marshal = PIECE_TYPE_BY_NAME["marshal"]
    assert cell(after, CH_HIDDEN_OPPONENT_OCCUPANCY, "e4") == 0.0
    assert cell(after, CH_KNOWN_OPPONENT_IDENTITY + marshal, "e4") == 1.0
    # And its original square is recorded in the setup-memory block.
    assert cell(after, CH_KNOWN_OPPONENT_SETUP + marshal, "e4") == 1.0
    assert defender.known_to(RED)


def test_known_opponent_identity_is_never_forgotten():
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e4": "marshal", "j10": "flag"},
        acting_player=RED,
    )
    play(state, "e3 e4")  # red captain dies, blue marshal revealed
    play(state, "e4 e5")  # marshal moves on

    observation = build_observation(state, RED)
    marshal = PIECE_TYPE_BY_NAME["marshal"]
    assert cell(observation, CH_KNOWN_OPPONENT_IDENTITY + marshal, "e5") == 1.0
    assert cell(observation, CH_HIDDEN_OPPONENT_OCCUPANCY, "e5") == 0.0


def test_captured_opponent_leaves_the_known_identity_planes():
    state = make_position(
        red={"e3": "marshal", "a1": "flag"},
        blue={"e4": "captain", "j10": "flag"},
        acting_player=RED,
    )
    play(state, "e3 e4")
    observation = build_observation(state, RED)
    captain = PIECE_TYPE_BY_NAME["captain"]
    assert cell(observation, CH_KNOWN_OPPONENT_IDENTITY + captain, "e4") == 0.0


def test_own_known_to_opponent_plane_states():
    """The three states required by validation-matrix section 4.3."""
    # 1. never revealed
    state = make_position(red={"e3": "captain"}, blue={"j10": "flag"})
    assert cell(build_observation(state, RED), CH_OWN_KNOWN_TO_OPPONENT, "e3") == 0.0

    # 2. revealed by combat and still alive
    state = make_position(
        red={"e3": "marshal", "a1": "flag"},
        blue={"e4": "captain", "j10": "flag"},
        acting_player=RED,
    )
    play(state, "e3 e4")
    assert cell(build_observation(state, RED), CH_OWN_KNOWN_TO_OPPONENT, "e4") == 1.0

    # 3. logically identified as a Scout by a multi-square move
    state = make_position(red={"a1": "scout"}, blue={"j10": "flag"}, acting_player=RED)
    play(state, "a1 a4")
    assert cell(build_observation(state, RED), CH_OWN_KNOWN_TO_OPPONENT, "a4") == 1.0


def test_own_identities_are_always_visible_to_their_owner():
    state = make_position(
        red={"e3": "spy", "d3": "bomb", "c3": "flag"},
        blue={"e7": "marshal", "j10": "flag"},
    )
    observation = build_observation(state, RED)
    for square_name, type_name in (("e3", "spy"), ("d3", "bomb"), ("c3", "flag")):
        piece_type = PIECE_TYPE_BY_NAME[type_name]
        assert cell(observation, CH_OWN_IDENTITY + piece_type, square_name) == 1.0


def test_each_side_sees_the_other_as_hidden():
    state = make_position(
        red={"e3": "spy"}, blue={"e7": "marshal", "j10": "flag"}, acting_player=RED
    )
    red_view = build_observation(state, RED)
    blue_view = build_observation(state, BLUE)

    assert cell(red_view, CH_HIDDEN_OPPONENT_OCCUPANCY, "e7", RED) == 1.0
    assert cell(blue_view, CH_HIDDEN_OPPONENT_OCCUPANCY, "e3", BLUE) == 1.0
    spy = PIECE_TYPE_BY_NAME["spy"]
    assert cell(blue_view, CH_KNOWN_OPPONENT_IDENTITY + spy, "e3", BLUE) == 0.0
