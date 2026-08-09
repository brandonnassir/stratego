"""Ordinary movement and immovable pieces.

Covers `04_engine_validation_plan.md` sections 4 and 5.
"""

import pytest

from stratego.engine.actions import decode_action, encode_action
from stratego.engine.constants import BLUE, OCCUPIABLE_SQUARES, RED
from stratego.engine.coordinates import square_from_name, square_name
from stratego.engine.legal_moves import legal_actions
from tests.helpers import make_position, square

# One representative movable type; ordinary movement is identical for all of
# them, and Scout ray movement has its own module.
ORDINARY_TYPES = [
    "spy",
    "miner",
    "sergeant",
    "lieutenant",
    "captain",
    "major",
    "colonel",
    "general",
    "marshal",
]


def destinations_from(state, origin_name: str) -> set[str]:
    origin = square_from_name(origin_name)
    return {
        square_name(decode_action(action)[1])
        for action in legal_actions(state)
        if decode_action(action)[0] == origin
    }


@pytest.mark.parametrize("type_name", ORDINARY_TYPES)
def test_interior_piece_moves_one_square_in_four_directions(type_name):
    state = make_position(red={"e3": type_name}, blue={"a10": "flag"})
    assert destinations_from(state, "e3") == {"e2", "e4", "d3", "f3"}


@pytest.mark.parametrize("type_name", ORDINARY_TYPES)
def test_ordinary_piece_never_moves_diagonally_or_two_squares(type_name):
    state = make_position(red={"e3": type_name}, blue={"a10": "flag"})
    reachable = destinations_from(state, "e3")
    for illegal in ("d2", "f2", "d4", "f4", "e1", "e5", "c3", "g3"):
        assert illegal not in reachable


def test_corner_piece_has_two_moves():
    state = make_position(red={"a1": "captain"}, blue={"a10": "flag"})
    assert destinations_from(state, "a1") == {"a2", "b1"}
    state = make_position(red={"j1": "captain"}, blue={"a10": "flag"})
    assert destinations_from(state, "j1") == {"j2", "i1"}


def test_edge_piece_has_three_moves():
    state = make_position(red={"a3": "captain"}, blue={"j10": "flag"})
    assert destinations_from(state, "a3") == {"a2", "a4", "b3"}


def test_no_move_onto_a_lake():
    # b5 sits immediately west of the c5 lake.
    state = make_position(red={"b5": "captain"}, blue={"a10": "flag"})
    reachable = destinations_from(state, "b5")
    assert "c5" not in reachable
    assert reachable == {"a5", "b4", "b6"}


def test_no_move_onto_a_friendly_piece():
    state = make_position(red={"e3": "captain", "e4": "miner"}, blue={"a10": "flag"})
    assert "e4" not in destinations_from(state, "e3")


def test_attack_onto_an_adjacent_enemy_is_legal():
    state = make_position(red={"e3": "captain"}, blue={"e4": "sergeant", "a10": "flag"})
    assert "e4" in destinations_from(state, "e3")


def test_moves_are_generated_only_for_the_acting_players_pieces():
    state = make_position(
        red={"e3": "captain"}, blue={"e7": "captain", "a10": "flag"}, acting_player=RED
    )
    sources = {decode_action(action)[0] for action in legal_actions(state)}
    assert sources == {square("e3")}

    state.acting_player = BLUE
    sources = {decode_action(action)[0] for action in legal_actions(state)}
    assert sources == {square("e7")}


@pytest.mark.parametrize("type_name", ["flag", "bomb"])
@pytest.mark.parametrize(
    "location", ["a1", "e3", "b5", "j10", "e6", "a5", "j1", "d4", "f7"]
)
def test_flag_and_bomb_never_have_a_legal_move(type_name, location):
    state = make_position(red={location: type_name}, blue={"j10" if location != "j10" else "a10": "flag"})
    assert destinations_from(state, location) == set()


def test_flag_and_bomb_have_no_moves_anywhere_on_the_board():
    for target in OCCUPIABLE_SQUARES:
        name = square_name(target)
        other = "a10" if name != "a10" else "b10"
        state = make_position(red={name: "bomb"}, blue={other: "flag"})
        assert legal_actions(state) == []


def test_flag_and_bomb_can_still_be_attacked():
    state = make_position(red={"e3": "miner"}, blue={"e4": "bomb", "a10": "flag"})
    assert "e4" in destinations_from(state, "e3")
    state = make_position(red={"e3": "miner"}, blue={"e4": "flag"})
    assert "e4" in destinations_from(state, "e3")


def test_pieces_may_move_between_the_lakes():
    # e5/f5 form the gap between the two lakes.
    state = make_position(red={"e5": "captain"}, blue={"a10": "flag"})
    assert destinations_from(state, "e5") == {"e4", "e6", "f5"}


def test_all_generated_actions_are_well_formed():
    state = make_position(
        red={"e3": "captain", "b5": "scout", "a1": "bomb"},
        blue={"e4": "sergeant", "a10": "flag"},
    )
    actions = legal_actions(state)
    assert actions == sorted(set(actions))
    for action in actions:
        source, destination = decode_action(action)
        assert encode_action(source, destination) == action
        assert source != destination
