"""Scout ray movement and the Scout identity revelation.

Covers `04_engine_validation_plan.md` section 6 and `01_official_rules.md`
section 4.
"""

from stratego.engine.actions import decode_action
from stratego.engine.constants import BLUE, RED
from stratego.engine.coordinates import square_from_name, square_name
from stratego.engine.events import EVENT_IDENTITY_REVEAL
from stratego.engine.legal_moves import legal_actions
from stratego.engine.observation import (
    CH_KNOWN_OPPONENT_IDENTITY,
    CH_OWN_KNOWN_TO_OPPONENT,
    build_observation,
)
from tests.helpers import make_position, piece_at, play, square


def destinations_from(state, origin_name: str) -> set[str]:
    origin = square_from_name(origin_name)
    return {
        square_name(decode_action(action)[1])
        for action in legal_actions(state)
        if decode_action(action)[0] == origin
    }


def test_scout_moves_any_distance_along_an_open_file():
    state = make_position(red={"a1": "scout"}, blue={"j10": "flag"})
    assert destinations_from(state, "a1") == {
        f"a{row}" for row in range(2, 11)
    } | {chr(ord("a") + column) + "1" for column in range(1, 10)}


def test_scout_moves_a_single_square_too():
    state = make_position(red={"e3": "scout"}, blue={"a10": "flag"})
    assert "e4" in destinations_from(state, "e3")


def test_scout_stops_before_a_friendly_piece():
    state = make_position(red={"a1": "scout", "a5": "miner"}, blue={"j10": "flag"})
    reachable = destinations_from(state, "a1")
    assert {"a2", "a3", "a4"} <= reachable
    assert "a5" not in reachable
    assert "a6" not in reachable


def test_scout_attacks_the_first_opponent_and_cannot_pass_it():
    state = make_position(
        red={"a1": "scout"}, blue={"a5": "sergeant", "a7": "captain", "j10": "flag"}
    )
    reachable = destinations_from(state, "a1")
    assert {"a2", "a3", "a4", "a5"} <= reachable
    assert "a6" not in reachable
    assert "a7" not in reachable


def test_scout_cannot_cross_a_lake():
    # Travelling east along row 5 from a5 hits the c5 lake.
    state = make_position(red={"a5": "scout"}, blue={"j10": "flag"})
    reachable = destinations_from(state, "a5")
    assert "b5" in reachable
    assert "c5" not in reachable
    assert "d5" not in reachable
    assert "e5" not in reachable


def test_scout_rays_terminate_at_the_board_edge():
    state = make_position(red={"e3": "scout"}, blue={"a10": "flag"})
    reachable = destinations_from(state, "e3")
    assert "e1" in reachable
    assert "a3" in reachable
    assert "j3" in reachable
    assert "e10" in reachable
    assert len(reachable) == 9 + 9  # whole rank plus whole file, minus own square


def test_scout_never_moves_diagonally():
    state = make_position(red={"e3": "scout"}, blue={"a10": "flag"})
    reachable = destinations_from(state, "e3")
    for illegal in ("d2", "f4", "a1", "j10", "c5"):
        assert illegal not in reachable


def test_scout_in_all_four_directions():
    state = make_position(red={"e3": "scout"}, blue={"a10": "flag"})
    reachable = destinations_from(state, "e3")
    assert {"e2", "e1"} <= reachable  # towards row 1
    assert {"e4", "e5"} <= reachable  # towards row 10
    assert {"d3", "c3"} <= reachable  # west
    assert {"f3", "g3"} <= reachable  # east


def test_multi_square_scout_move_reveals_the_scout_to_the_opponent():
    state = make_position(red={"a1": "scout"}, blue={"j10": "flag"}, acting_player=RED)
    scout = piece_at(state, "a1")
    assert not scout.known_to(BLUE)

    events = play(state, "a1 a4")[0]

    assert scout.known_to(BLUE)
    assert scout.reveal_reason_blue == "scout_multisquare"
    reveals = [event for event in events if event["event_type"] == EVENT_IDENTITY_REVEAL]
    assert len(reveals) == 1
    assert reveals[0]["reason"] == "scout_multisquare"
    assert reveals[0]["piece_type"] == "scout"
    assert reveals[0]["newly_known_to"] == ["blue"]


def test_single_square_scout_move_does_not_reveal_the_scout():
    state = make_position(red={"a1": "scout"}, blue={"j10": "flag"}, acting_player=RED)
    scout = piece_at(state, "a1")
    events = play(state, "a1 a2")[0]
    assert not scout.known_to(BLUE)
    assert not any(event["event_type"] == EVENT_IDENTITY_REVEAL for event in events)


def test_scout_revelation_appears_in_both_observations():
    state = make_position(
        red={"a1": "scout"}, blue={"j10": "flag", "j9": "captain"}, acting_player=RED
    )
    play(state, "a1 a4")

    # Blue now legally knows the piece on a4 is a Scout.
    blue_view = build_observation(state, BLUE)
    from stratego.engine.constants import SCOUT
    from stratego.engine.coordinates import to_perspective

    normalized = to_perspective(square("a4"), BLUE)
    row, column = divmod(normalized, 10)
    assert blue_view[CH_KNOWN_OPPONENT_IDENTITY + SCOUT, row, column] == 1.0

    # Red sees that this piece is now known to the opponent.
    red_view = build_observation(state, RED)
    row, column = divmod(square("a4"), 10)
    assert red_view[CH_OWN_KNOWN_TO_OPPONENT, row, column] == 1.0


def test_scout_revelation_persists_after_further_moves():
    state = make_position(
        red={"a1": "scout"}, blue={"j10": "flag", "j9": "captain"}, acting_player=RED
    )
    play(state, "a1 a4", "j9 i9", "a4 a5")
    scout = piece_at(state, "a5")
    assert scout.known_to(BLUE)


def test_scout_attack_at_range_resolves_combat_and_reveals_by_combat():
    state = make_position(
        red={"a1": "scout"}, blue={"a5": "sergeant", "j10": "flag"}, acting_player=RED
    )
    scout = piece_at(state, "a1")
    defender = piece_at(state, "a5")
    play(state, "a1 a5")

    assert not scout.alive  # sergeant outranks scout
    assert defender.alive
    assert scout.known_to(BLUE)
    assert defender.known_to(RED)
    assert scout.reveal_reason_blue == "combat"
