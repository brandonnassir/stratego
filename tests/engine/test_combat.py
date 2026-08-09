"""Exhaustive combat matrix and post-combat occupancy.

Covers `04_engine_validation_plan.md` section 7 and section 8 of the Phase Two
instructions.

The expected outcomes are a literal table transcribed from
`01_official_rules.md` sections 6 and 7 rather than a reimplementation of the
resolver, so a logic error in `combat.py` cannot be mirrored by the test.
"""

import pytest

from stratego.engine.combat import (
    ATTACKER_WINS,
    BOTH_REMOVED,
    DEFENDER_WINS,
    CombatError,
    resolve_combat,
)
from stratego.engine.constants import BLUE, PIECE_TYPE_BY_NAME, RED, TERMINAL_FLAG_CAPTURE
from tests.helpers import make_position, piece_at, play, square

# Column order of the table below.
DEFENDERS = (
    "spy",
    "scout",
    "miner",
    "sergeant",
    "lieutenant",
    "captain",
    "major",
    "colonel",
    "general",
    "marshal",
    "flag",
    "bomb",
)

# Rows are attackers, columns are defenders in DEFENDERS order.
# A = attacker survives, D = defender survives, B = both removed.
COMBAT_TABLE = {
    "spy": "B D D D D D D D D A A D",
    "scout": "A B D D D D D D D D A D",
    "miner": "A A B D D D D D D D A A",
    "sergeant": "A A A B D D D D D D A D",
    "lieutenant": "A A A A B D D D D D A D",
    "captain": "A A A A A B D D D D A D",
    "major": "A A A A A A B D D D A D",
    "colonel": "A A A A A A A B D D A D",
    "general": "A A A A A A A A B D A D",
    "marshal": "A A A A A A A A A B A D",
}

OUTCOME_BY_LETTER = {"A": ATTACKER_WINS, "D": DEFENDER_WINS, "B": BOTH_REMOVED}

COMBAT_CASES = [
    (attacker, defender, OUTCOME_BY_LETTER[letter])
    for attacker, row in COMBAT_TABLE.items()
    for defender, letter in zip(DEFENDERS, row.split())
]


def test_combat_matrix_covers_every_meaningful_pair():
    # 10 movable attackers against all 12 defender types.
    assert len(COMBAT_CASES) == 120
    assert len({(attacker, defender) for attacker, defender, _ in COMBAT_CASES}) == 120


@pytest.mark.parametrize("attacker,defender,expected", COMBAT_CASES)
def test_combat_matrix(attacker, defender, expected):
    outcome = resolve_combat(PIECE_TYPE_BY_NAME[attacker], PIECE_TYPE_BY_NAME[defender])
    assert outcome == expected, f"{attacker} attacking {defender}"


def test_spy_attacking_marshal_wins():
    assert resolve_combat(PIECE_TYPE_BY_NAME["spy"], PIECE_TYPE_BY_NAME["marshal"]) == (
        ATTACKER_WINS
    )


def test_marshal_attacking_spy_wins():
    assert resolve_combat(PIECE_TYPE_BY_NAME["marshal"], PIECE_TYPE_BY_NAME["spy"]) == (
        ATTACKER_WINS
    )


def test_miner_attacking_bomb_wins():
    assert resolve_combat(PIECE_TYPE_BY_NAME["miner"], PIECE_TYPE_BY_NAME["bomb"]) == (
        ATTACKER_WINS
    )


@pytest.mark.parametrize(
    "attacker", [name for name in COMBAT_TABLE if name not in ("miner",)]
)
def test_every_other_piece_attacking_a_bomb_loses(attacker):
    assert resolve_combat(PIECE_TYPE_BY_NAME[attacker], PIECE_TYPE_BY_NAME["bomb"]) == (
        DEFENDER_WINS
    )


@pytest.mark.parametrize("attacker", sorted(COMBAT_TABLE))
def test_every_movable_piece_captures_the_flag(attacker):
    assert resolve_combat(PIECE_TYPE_BY_NAME[attacker], PIECE_TYPE_BY_NAME["flag"]) == (
        ATTACKER_WINS
    )


@pytest.mark.parametrize("rank_name", sorted(COMBAT_TABLE))
def test_equal_ranks_remove_both(rank_name):
    assert resolve_combat(
        PIECE_TYPE_BY_NAME[rank_name], PIECE_TYPE_BY_NAME[rank_name]
    ) == BOTH_REMOVED


@pytest.mark.parametrize("immovable", ["flag", "bomb"])
def test_immovable_pieces_can_never_attack(immovable):
    with pytest.raises(CombatError):
        resolve_combat(PIECE_TYPE_BY_NAME[immovable], PIECE_TYPE_BY_NAME["scout"])


# ---------------------------------------------------------------------------
# Post-combat occupancy through real transitions
# ---------------------------------------------------------------------------


def test_attacker_wins_occupies_the_destination():
    state = make_position(
        red={"e3": "marshal"}, blue={"e4": "captain", "a10": "flag"}, acting_player=RED
    )
    attacker = piece_at(state, "e3")
    defender = piece_at(state, "e4")
    play(state, "e3 e4")

    assert attacker.alive and attacker.current_square == defender.starting_square
    assert not defender.alive and defender.current_square is None
    assert state.board[attacker.current_square] == attacker.piece_id
    assert state.board[square("e3")] is None


def test_defender_wins_keeps_the_destination_and_empties_the_source():
    state = make_position(
        red={"e3": "captain"}, blue={"e4": "marshal", "a10": "flag"}, acting_player=RED
    )
    attacker = piece_at(state, "e3")
    defender = piece_at(state, "e4")
    source = attacker.current_square
    destination = defender.current_square
    play(state, "e3 e4")

    assert not attacker.alive
    assert defender.alive and defender.current_square == destination
    assert state.board[source] is None
    assert state.board[destination] == defender.piece_id


def test_tie_removes_both_and_empties_the_destination():
    state = make_position(
        red={"e3": "captain"}, blue={"e4": "captain", "a10": "flag"}, acting_player=RED
    )
    attacker = piece_at(state, "e3")
    defender = piece_at(state, "e4")
    source, destination = attacker.current_square, defender.current_square
    play(state, "e3 e4")

    assert not attacker.alive and not defender.alive
    assert state.board[source] is None
    assert state.board[destination] is None


def test_flag_capture_ends_the_game_immediately():
    state = make_position(
        red={"e3": "scout"}, blue={"e4": "flag", "j10": "captain"}, acting_player=RED
    )
    play(state, "e3 e4")
    assert state.terminal
    assert state.terminal_reason == TERMINAL_FLAG_CAPTURE
    assert state.winner == RED
    assert state.result_for(RED) == 1.0
    assert state.result_for(BLUE) == -1.0


def test_blue_can_also_capture_the_flag():
    state = make_position(
        red={"e3": "flag", "a1": "captain"}, blue={"e4": "scout"}, acting_player=BLUE
    )
    play(state, "e4 e3")
    assert state.terminal
    assert state.terminal_reason == TERMINAL_FLAG_CAPTURE
    assert state.winner == BLUE


def test_combat_reveals_both_identities_to_both_players():
    state = make_position(
        red={"e3": "marshal"}, blue={"e4": "captain", "a10": "flag"}, acting_player=RED
    )
    attacker = piece_at(state, "e3")
    defender = piece_at(state, "e4")
    assert not attacker.known_to(BLUE)
    assert not defender.known_to(RED)

    play(state, "e3 e4")

    assert attacker.known_to(RED) and attacker.known_to(BLUE)
    assert defender.known_to(RED) and defender.known_to(BLUE)


def test_combat_resets_the_battleless_counter():
    state = make_position(
        red={"e3": "marshal"},
        blue={"e4": "captain", "a10": "flag"},
        acting_player=RED,
        battleless_moves=37,
    )
    play(state, "e3 e4")
    assert state.battleless_moves == 0


def test_miner_survives_a_bomb_and_takes_the_square():
    state = make_position(
        red={"e3": "miner"}, blue={"e4": "bomb", "a10": "flag"}, acting_player=RED
    )
    miner = piece_at(state, "e3")
    bomb = piece_at(state, "e4")
    play(state, "e3 e4")
    assert miner.alive and miner.current_square == bomb.starting_square
    assert not bomb.alive


def test_non_miner_dies_to_a_bomb_which_survives():
    state = make_position(
        red={"e3": "marshal"}, blue={"e4": "bomb", "a10": "flag"}, acting_player=RED
    )
    marshal = piece_at(state, "e3")
    bomb = piece_at(state, "e4")
    play(state, "e3 e4")
    assert not marshal.alive
    assert bomb.alive and bomb.current_square == bomb.starting_square
    assert not bomb.has_moved


def test_spy_defeats_an_attacking_marshal_only_when_the_spy_attacks():
    state = make_position(
        red={"e3": "spy"}, blue={"e4": "marshal", "a10": "flag"}, acting_player=RED
    )
    spy = piece_at(state, "e3")
    marshal = piece_at(state, "e4")
    play(state, "e3 e4")
    assert spy.alive and not marshal.alive

    state = make_position(
        red={"e3": "spy"}, blue={"e4": "marshal", "a10": "flag"}, acting_player=BLUE
    )
    spy = piece_at(state, "e3")
    marshal = piece_at(state, "e4")
    play(state, "e4 e3")
    assert marshal.alive and not spy.alive
