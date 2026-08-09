"""Perspective normalization.

Covers `06_observation_v2_127ch.md` section 3, `04_engine_validation_plan.md`
section 12 and section 15 of the Phase Two instructions.

The strongest available statement is a *mirror equivalence*: build a game and its
colour-swapped, 180-degree-rotated twin, then require that the two players who
occupy equivalent roles receive byte-identical observations. If any channel used
the wrong coordinate frame, that equality would break.
"""

import random

import numpy as np
import pytest

from stratego.engine.actions import action_to_perspective, decode_action, encode_action
from stratego.engine.constants import BLUE, RED, RulesConfig
from stratego.engine.coordinates import to_perspective
from stratego.engine.legal_moves import legal_actions
from stratego.engine.observation import (
    CH_KNOWN_OPPONENT_SETUP,
    CH_LAKE_MASK,
    CH_OWN_SETUP,
    CH_OWN_START_ROW,
    CH_RECENT_MOVES,
    build_observation,
)
from stratego.engine.random_play import select_random_action
from stratego.engine.setup import random_setup
from stratego.engine.state import create_game
from stratego.engine.transition import apply_action

MIRROR_RULES = RulesConfig(first_player=BLUE)


def mirror_square(square: int) -> int:
    return 99 - square


def mirror_action(action_id: int) -> int:
    source, destination = decode_action(action_id)
    return encode_action(mirror_square(source), mirror_square(destination))


def mirrored_games(seed: int):
    """A game and its colour-swapped, rotated twin.

    In the twin, red plays the role blue plays in the original and vice versa.
    Reversing a row-major setup tuple is exactly the 180-degree rotation of that
    player's four setup rows onto the other player's four rows.
    """
    red_setup = random_setup(random.Random(seed))
    blue_setup = random_setup(random.Random(seed + 500))

    original = create_game(red_setup, blue_setup, game_id="original")
    twin = create_game(
        tuple(reversed(blue_setup)),
        tuple(reversed(red_setup)),
        rules=MIRROR_RULES,
        game_id="twin",
    )
    return original, twin


def play_mirrored(original, twin, plies, seed):
    """Advance both games through the same sequence of mirrored actions."""
    rng = random.Random(seed)
    for _ in range(plies):
        if original.terminal or twin.terminal:
            break
        action = select_random_action(original, rng)
        apply_action(original, action)
        apply_action(twin, mirror_action(action))


@pytest.mark.parametrize("seed", range(5))
def test_mirrored_games_start_with_identical_normalized_observations(seed):
    original, twin = mirrored_games(seed)
    assert np.array_equal(
        build_observation(original, RED), build_observation(twin, BLUE)
    )
    assert np.array_equal(
        build_observation(original, BLUE), build_observation(twin, RED)
    )


# Channels outside the behavioural block. These must be mirror-exact in every
# position; see `test_behaviour_counterpart_selection_is_orientation_dependent`
# for why the behavioural block is treated separately.
NON_BEHAVIOUR_CHANNELS = list(range(0, 68)) + list(range(108, 127))


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("plies", [1, 6, 25, 60, 150])
def test_mirrored_games_keep_identical_non_behaviour_channels(seed, plies):
    original, twin = mirrored_games(seed)
    play_mirrored(original, twin, plies, seed)

    for left, right in ((RED, BLUE), (BLUE, RED)):
        first = build_observation(original, left)[NON_BEHAVIOUR_CHANNELS]
        second = build_observation(twin, right)[NON_BEHAVIOUR_CHANNELS]
        assert np.array_equal(first, second), f"observer pair {left}/{right}"


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("plies", [1, 6, 25])
def test_mirrored_games_agree_on_behaviour_actor_placement(seed, plies):
    """Which piece owns a behaviour record is orientation independent.

    Only the *counterpart* selection depends on absolute square order, so the
    recency planes of the four actor-selected behaviours must still mirror
    exactly. `was_protected` is excluded because its actor is the selected
    counterpart of the corresponding `protect` event.
    """
    original, twin = mirrored_games(seed)
    play_mirrored(original, twin, plies, seed)

    recency_channels = [
        block + 4 * behavior
        for block in (68, 88)
        for behavior in range(4)  # threat, evade, declined attack, protect
    ]
    for left, right in ((RED, BLUE), (BLUE, RED)):
        assert np.array_equal(
            build_observation(original, left)[recency_channels],
            build_observation(twin, right)[recency_channels],
        )


def test_behaviour_counterpart_selection_is_orientation_dependent():
    """Documented consequence of the absolute-index tie-break rule.

    `06_observation_v2_127ch.md` section 10 breaks counterpart ties by lowest
    *absolute* board-square index. Absolute indices are not preserved by the
    180-degree perspective rotation, so a colour-swapped twin can legitimately
    select the other candidate. This test pins that behaviour rather than
    asserting a symmetry the specification does not promise; the Phase Two
    report records it as an open question.
    """
    from tests.helpers import make_position, piece_at, play

    # Red's captain ends adjacent to two blue pieces at once.
    red_view = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"d4": "sergeant", "e5": "marshal", "j10": "flag"},
        acting_player=RED,
    )
    low_square_piece = piece_at(red_view, "d4")
    play(red_view, "e3 e4")
    chosen = red_view.behavior_event(
        piece_at(red_view, "e4").piece_id, "threat"
    ).counterpart_piece_id
    assert chosen == low_square_piece.piece_id

    # The same geometry rotated 180 degrees, with the colours swapped, selects
    # the mirror image of the *other* candidate.
    blue_view = make_position(
        red={"g7": "sergeant", "f6": "marshal", "a1": "flag"},
        blue={"f8": "captain", "j10": "flag"},
        acting_player=BLUE,
    )
    play(blue_view, "f8 f7")
    mirrored_choice = blue_view.behavior_event(
        piece_at(blue_view, "f7").piece_id, "threat"
    ).counterpart_piece_id
    assert mirrored_choice == piece_at(blue_view, "f6").piece_id


def test_setup_memory_planes_transform_consistently():
    original, twin = mirrored_games(3)
    play_mirrored(original, twin, 30, 3)
    channels = list(range(CH_OWN_SETUP, CH_KNOWN_OPPONENT_SETUP + 12))
    assert np.array_equal(
        build_observation(original, BLUE)[channels],
        build_observation(twin, RED)[channels],
    )


def test_starting_coordinate_planes_transform_consistently():
    original, twin = mirrored_games(4)
    play_mirrored(original, twin, 24, 4)
    channels = list(range(CH_OWN_START_ROW, CH_OWN_START_ROW + 4))
    assert np.array_equal(
        build_observation(original, BLUE)[channels],
        build_observation(twin, RED)[channels],
    )


def test_recent_move_planes_transform_consistently():
    original, twin = mirrored_games(6)
    play_mirrored(original, twin, 40, 6)
    channels = list(range(CH_RECENT_MOVES, CH_RECENT_MOVES + 16))
    assert np.array_equal(
        build_observation(original, RED)[channels],
        build_observation(twin, BLUE)[channels],
    )


def test_lake_plane_is_identical_from_both_perspectives():
    original, twin = mirrored_games(2)
    reference = build_observation(original, RED)[CH_LAKE_MASK]
    for state, observer in ((original, BLUE), (twin, RED), (twin, BLUE)):
        assert np.array_equal(build_observation(state, observer)[CH_LAKE_MASK], reference)


@pytest.mark.parametrize("plies", [0, 3, 17])
def test_legal_actions_map_correctly_between_mirrored_games(plies):
    original, twin = mirrored_games(8)
    play_mirrored(original, twin, plies, 8)

    original_actions = legal_actions(original)
    twin_actions = legal_actions(twin)
    assert {mirror_action(action) for action in original_actions} == set(twin_actions)

    # And both sides see the same action set once normalized to their own frame.
    assert {
        action_to_perspective(action, original.acting_player) for action in original_actions
    } == {action_to_perspective(action, twin.acting_player) for action in twin_actions}


def test_own_setup_rows_always_normalize_to_the_bottom_of_the_board():
    original, _ = mirrored_games(1)
    for observer in (RED, BLUE):
        observation = build_observation(original, observer)
        own_setup = observation[list(range(CH_OWN_SETUP, CH_OWN_SETUP + 12))].sum(axis=0)
        assert own_setup[:4].sum() == 40.0
        assert own_setup[4:].sum() == 0.0


def test_square_transform_matches_the_action_transform():
    for square in range(0, 100, 9):
        assert to_perspective(square, BLUE) == mirror_square(square)
        assert to_perspective(square, RED) == square
    action = encode_action(3, 47)
    assert action_to_perspective(action, BLUE) == mirror_action(action)
    assert action_to_perspective(action, RED) == action


def test_terminal_results_agree_between_mirrored_games():
    original, twin = mirrored_games(11)
    play_mirrored(original, twin, 4000, 11)
    assert original.terminal == twin.terminal
    if original.terminal:
        assert original.terminal_reason == twin.terminal_reason
        if original.winner is None:
            assert twin.winner is None
        else:
            assert twin.winner == (BLUE if original.winner == RED else RED)
