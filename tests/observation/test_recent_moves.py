"""Recent-move planes (channels 108-123).

Covers `07_observation_validation_matrix.md` section 9.
"""

import numpy as np

from stratego.engine.constants import BLUE, RED, RulesConfig
from stratego.engine.coordinates import to_perspective
from stratego.engine.observation import CH_RECENT_MOVES, build_observation
from tests.helpers import cell, make_position, nonterminal_state, play, square

RECENT_CHANNELS = list(range(CH_RECENT_MOVES, CH_RECENT_MOVES + 16))
LOOSE_RULES = RulesConfig(battleless_move_limit=10_000, absolute_move_limit=10_000)


def shuffle_state():
    return make_position(
        red={"a1": "captain", "j1": "flag"},
        blue={"a10": "captain", "j10": "flag"},
        acting_player=RED,
        rules=LOOSE_RULES,
    )


def test_history_is_empty_at_the_start_of_a_game():
    observation = build_observation(shuffle_state(), RED)
    assert observation[RECENT_CHANNELS].sum() == 0.0


def test_newest_move_occupies_channel_108():
    state = shuffle_state()
    play(state, "a1 a2")
    observation = build_observation(state, RED)
    assert cell(observation, CH_RECENT_MOVES, "a1") == -1.0
    assert cell(observation, CH_RECENT_MOVES, "a2") == 1.0
    assert observation[CH_RECENT_MOVES].sum() == 0.0  # one -1 and one +1


def test_each_plane_holds_exactly_one_source_and_one_destination():
    state = nonterminal_state(40)
    observation = build_observation(state, RED)
    for channel in RECENT_CHANNELS:
        plane = observation[channel]
        assert int((plane == -1.0).sum()) == 1
        assert int((plane == 1.0).sum()) == 1
        assert int((plane == 0.0).sum()) == 98


def test_planes_shift_by_exactly_one_channel_after_each_ply():
    state = shuffle_state()
    play(state, "a1 a2")
    first = build_observation(state, RED)[CH_RECENT_MOVES].copy()

    play(state, "a10 a9")
    second = build_observation(state, RED)
    assert np.array_equal(second[CH_RECENT_MOVES + 1], first)
    assert not np.array_equal(second[CH_RECENT_MOVES], first)

    play(state, "a2 a1")
    third = build_observation(state, RED)
    assert np.array_equal(third[CH_RECENT_MOVES + 2], first)


def distinct_move_state():
    """Two Scouts walking along opposite edges, so every move is unique."""
    return make_position(
        red={"a1": "scout", "j1": "flag"},
        blue={"j10": "scout", "a10": "flag"},
        acting_player=RED,
        rules=LOOSE_RULES,
    )


# Two non-intersecting paths whose every step is a distinct source/destination
# pair, so each recent-move plane is individually identifiable.
RED_PATH = ["a1", "a2", "a3", "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4"]
BLUE_PATH = ["j10", "j9", "j8", "j7", "i7", "h7", "g7", "f7", "e7", "d7", "c7"]


def walk(state, count):
    """Play `count` plies along the two fixed paths and return the moves made."""
    played = []
    red_step = blue_step = 0
    for _ in range(count):
        if state.acting_player == RED:
            move = f"{RED_PATH[red_step]} {RED_PATH[red_step + 1]}"
            red_step += 1
        else:
            move = f"{BLUE_PATH[blue_step]} {BLUE_PATH[blue_step + 1]}"
            blue_step += 1
        play(state, move)
        played.append(move)
    return played


def test_planes_match_the_last_sixteen_moves_newest_first():
    state = distinct_move_state()
    played = walk(state, 20)
    observation = build_observation(state, RED)

    for offset, move in enumerate(reversed(played[-16:])):
        source_name, destination_name = move.split()
        channel = CH_RECENT_MOVES + offset
        assert cell(observation, channel, source_name) == -1.0, move
        assert cell(observation, channel, destination_name) == 1.0, move


def test_the_window_drops_moves_older_than_sixteen_plies():
    state = distinct_move_state()
    played = walk(state, 20)
    observation = build_observation(state, RED)

    assert len(state.recent_moves) == 16
    assert state.recent_moves[0].ply == state.total_moves - 15

    # The first four moves have fallen out of the window entirely.
    for move in played[:4]:
        source_name, destination_name = move.split()
        for channel in RECENT_CHANNELS:
            same_source = cell(observation, channel, source_name) == -1.0
            same_destination = cell(observation, channel, destination_name) == 1.0
            assert not (same_source and same_destination), move


def test_opponent_moves_are_recorded_from_both_perspectives():
    state = shuffle_state()
    play(state, "a1 a2")

    red_view = build_observation(state, RED)
    blue_view = build_observation(state, BLUE)

    assert cell(red_view, CH_RECENT_MOVES, "a1", RED) == -1.0
    assert cell(blue_view, CH_RECENT_MOVES, "a1", BLUE) == -1.0
    assert cell(blue_view, CH_RECENT_MOVES, "a2", BLUE) == 1.0


def test_recent_move_planes_use_the_normalized_frame():
    state = shuffle_state()
    play(state, "a1 a2")
    blue_view = build_observation(state, BLUE)

    normalized_source = to_perspective(square("a1"), BLUE)
    row, column = divmod(normalized_source, 10)
    assert blue_view[CH_RECENT_MOVES, row, column] == -1.0
    # In blue's frame the red a1 corner appears at j10.
    assert normalized_source == square("j10")


def test_scout_ray_move_marks_only_its_endpoints():
    state = make_position(
        red={"a1": "scout", "j1": "flag"}, blue={"a10": "captain", "j10": "flag"},
        acting_player=RED, rules=LOOSE_RULES,
    )
    play(state, "a1 a5")
    observation = build_observation(state, RED)
    assert cell(observation, CH_RECENT_MOVES, "a1") == -1.0
    assert cell(observation, CH_RECENT_MOVES, "a5") == 1.0
    for intermediate in ("a2", "a3", "a4"):
        assert cell(observation, CH_RECENT_MOVES, intermediate) == 0.0
