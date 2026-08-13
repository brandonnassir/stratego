"""Initial-position mobility termination (`phase2_1_reference_1.2.0`).

A legal random setup can place Flag and Bombs on all six front-row squares
whose forward square is not a lake. At ply 0 every other own piece is boxed in,
so the player is stranded before any action is applied. Until 1.2.0
`create_game` returned such a position labelled active with zero legal moves --
the state that aborted the first Phase 6B production soak. The rule
(`01_official_rules.md` section 8, `02_project_ruleset.md` section 1) decides
the game at creation: the mobile opponent wins, or neither side can move and it
is a draw.

These are the permanent regressions mandated by the Phase 6B engine-correction
review. The exact production case `(root_seed 60006, environment 112,
generation 98)` lives in `tests/training/test_stillborn_games.py`, because
rebuilding it needs the training layer's slot-seed derivation.
"""

import pytest

from stratego.engine.actions import encode_action
from stratego.engine.constants import (
    BLUE,
    BOMB,
    FLAG,
    IMMOVABLE_TYPES,
    MINER,
    PIECE_COUNTS,
    RED,
    TERMINAL_BOTH_NO_LEGAL_MOVE_DRAW,
    TERMINAL_OPPONENT_NO_LEGAL_MOVE,
)
from stratego.engine.events import EVENT_GAME_END
from stratego.engine.legal_moves import has_legal_action, legal_action_mask, legal_actions
from stratego.engine.random_play import make_random_setups
from stratego.engine.replay import build_replay_record, rebuild_final_state
from stratego.engine.setup import validate_setup
from stratego.engine.snapshot import create_snapshot, restore_snapshot
from stratego.engine.state import create_game, state_fingerprint
from stratego.engine.transition import apply_action

#: Setup indices whose square has a non-lake forward square, per player. Red's
#: front row is row 3 (indices 30..39, squares 30..39); blue's is row 6
#: (indices 0..9, squares 60..69). Lake columns 2, 3, 6, 7 block the rest.
RED_FRONT_OPEN = (30, 31, 34, 35, 38, 39)
BLUE_FRONT_OPEN = (0, 1, 4, 5, 8, 9)


def build_setup(placements: dict[int, int]) -> tuple[int, ...]:
    """A legal 40-entry setup with `placements` pinned and the rest filled.

    The fill walks the remaining official inventory in ascending type order, so
    the result is deterministic and always inventory-exact.
    """
    inventory: list[int] = []
    for piece_type, count in sorted(PIECE_COUNTS.items()):
        inventory.extend([piece_type] * count)
    for piece_type in placements.values():
        inventory.remove(piece_type)
    setup: list[int | None] = [None] * 40
    for index, piece_type in placements.items():
        setup[index] = piece_type
    for index in range(40):
        if setup[index] is None:
            setup[index] = inventory.pop(0)
    assert not inventory
    return validate_setup(tuple(setup), RED)


def stranded_placements(front_open: tuple[int, ...], back_index: int) -> dict[int, int]:
    """Flag plus five Bombs across the open front squares, sixth Bomb behind."""
    placements = {front_open[0]: FLAG, back_index: BOMB}
    for index in front_open[1:]:
        placements[index] = BOMB
    return placements


def stranded_red_setup() -> tuple[int, ...]:
    return build_setup(stranded_placements(RED_FRONT_OPEN, back_index=0))


def stranded_blue_setup() -> tuple[int, ...]:
    return build_setup(stranded_placements(BLUE_FRONT_OPEN, back_index=39))


def mobile_red_setup() -> tuple[int, ...]:
    """A mobile setup with a Miner on the open front column `a`.

    Pinning the Miner makes `encode_action(30, 40)` a guaranteed legal
    single-step forward move that cannot reach, reveal or capture anything, so
    a test can advance the game one ply with a completely predictable result.
    """
    return build_setup({30: MINER})


class TestStrandedFirstPlayer:
    def test_red_stranded_blue_mobile_is_a_blue_win_at_ply_zero(self):
        state = create_game(stranded_red_setup(), make_random_setups(0)[1])
        assert state.terminal is True
        assert state.terminal_reason == TERMINAL_OPPONENT_NO_LEGAL_MOVE
        assert state.winner == BLUE
        assert state.is_draw is False
        assert state.total_moves == 0
        assert state.action_history == []
        assert legal_actions(state) == []
        assert not legal_action_mask(state).any()
        assert has_legal_action(state, RED) is False
        assert has_legal_action(state, BLUE) is True
        assert state.result_for(BLUE) == 1.0
        assert state.result_for(RED) == -1.0

    def test_the_stranding_pattern_is_what_this_module_says_it_is(self):
        setup = stranded_red_setup()
        assert all(setup[index] in IMMOVABLE_TYPES for index in RED_FRONT_OPEN)
        # Exactly the official inventory, just arranged adversarially.
        assert sorted(setup) == sorted(
            piece for piece, count in PIECE_COUNTS.items() for _ in range(count)
        )

    def test_both_players_stranded_is_a_draw_at_ply_zero(self):
        state = create_game(stranded_red_setup(), stranded_blue_setup())
        assert state.terminal is True
        assert state.terminal_reason == TERMINAL_BOTH_NO_LEGAL_MOVE_DRAW
        assert state.winner is None
        assert state.is_draw is True
        assert state.total_moves == 0
        assert state.result_for(RED) == 0.0
        assert state.result_for(BLUE) == 0.0

    def test_a_game_decided_at_creation_emits_exactly_one_game_end_event(self):
        state = create_game(stranded_red_setup(), make_random_setups(0)[1])
        ends = [event for event in state.events if event["event_type"] == EVENT_GAME_END]
        assert len(ends) == 1
        assert ends[0] is state.events[-1]
        assert ends[0]["ply"] == 0
        assert ends[0]["winner"] == "blue"
        assert ends[0]["result"] == "blue_win"
        assert ends[0]["terminal_reason"] == TERMINAL_OPPONENT_NO_LEGAL_MOVE


class TestStrandedSecondPlayer:
    def test_red_mobile_blue_stranded_is_not_terminal_at_creation(self):
        state = create_game(mobile_red_setup(), stranded_blue_setup())
        assert state.terminal is False
        assert state.acting_player == RED
        assert legal_actions(state)
        assert has_legal_action(state, BLUE) is False
        assert state.events == []

    def test_reds_first_move_then_ends_the_game_as_before(self):
        # The pre-1.2.0 behaviour for a stranded *second* player is preserved:
        # the transition-time evaluation ends the game at ply 1.
        state = create_game(mobile_red_setup(), stranded_blue_setup())
        apply_action(state, encode_action(30, 40))
        assert state.terminal is True
        assert state.terminal_reason == TERMINAL_OPPONENT_NO_LEGAL_MOVE
        assert state.winner == RED
        assert state.total_moves == 1


class TestOrdinaryCreationUnchanged:
    @pytest.mark.parametrize("seed", [0, 1, 7, 991])
    def test_a_mobile_starting_position_is_created_exactly_as_before(self, seed):
        red_setup, blue_setup = make_random_setups(seed)
        state = create_game(red_setup, blue_setup)
        assert state.terminal is False
        assert state.terminal_reason == "not_terminal"
        assert state.winner is None
        assert state.acting_player == RED
        assert state.total_moves == 0
        assert state.events == []
        assert len(legal_actions(state)) > 0


class TestBornTerminalStateIsAFullCitizen:
    def test_snapshot_round_trips_a_born_terminal_state(self):
        state = create_game(stranded_red_setup(), make_random_setups(0)[1])
        restored = restore_snapshot(create_snapshot(state, include_history=True))
        assert state_fingerprint(restored) == state_fingerprint(state)
        assert restored.terminal is True
        assert restored.terminal_reason == TERMINAL_OPPONENT_NO_LEGAL_MOVE

    def test_replay_record_round_trips_a_born_terminal_game(self):
        red_setup = stranded_red_setup()
        blue_setup = make_random_setups(0)[1]
        state = create_game(red_setup, blue_setup, game_id="stillborn-replay")
        record = build_replay_record(state, red_setup, blue_setup)
        assert record.actions == []
        assert record.total_moves == 0
        assert record.terminal_result == "blue_win"
        assert record.terminal_reason == TERMINAL_OPPONENT_NO_LEGAL_MOVE
        rebuilt = rebuild_final_state(record)
        assert state_fingerprint(rebuilt) == state_fingerprint(state)
