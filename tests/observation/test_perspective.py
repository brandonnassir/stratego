"""Perspective normalization.

Covers `06_observation_v2_127ch.md` sections 3 and 10.6,
`07_observation_validation_matrix.md` section 9A, `04_engine_validation_plan.md`
section 12 and section 15 of the Phase Two instructions.

The contract is a *mirror equivalence*: build a game and its colour-swapped,
180-degree-rotated twin, then require that the two players who occupy equivalent
roles receive byte-identical observations across all 127 channels. If any
channel used the wrong coordinate frame, that equality would break.

Under the superseded `observation_v2_127ch` this held only for channels 0-67 and
108-126, because behavioural counterpart ties were broken by absolute square
index. `observation_v2_1_127ch` breaks them by normalized index, so the
equivalence now covers the whole tensor.
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


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("plies", [1, 6, 25, 60, 150])
def test_mirrored_games_stay_identical_across_all_127_channels(seed, plies):
    original, twin = mirrored_games(seed)
    play_mirrored(original, twin, plies, seed)

    for left, right in ((RED, BLUE), (BLUE, RED)):
        first = build_observation(original, left)
        second = build_observation(twin, right)
        differing = [
            channel
            for channel in range(127)
            if not np.array_equal(first[channel], second[channel])
        ]
        assert differing == [], f"observer pair {left}/{right}, channels {differing}"


@pytest.mark.parametrize("seed", range(6))
def test_mirrored_games_agree_at_every_ply(seed):
    """Equality must hold continuously, not only at sampled checkpoints."""
    original, twin = mirrored_games(seed)
    for _ in range(40):
        play_mirrored(original, twin, 1, seed)
        if original.terminal:
            break
        for left, right in ((RED, BLUE), (BLUE, RED)):
            assert np.array_equal(
                build_observation(original, left), build_observation(twin, right)
            )


def test_behaviour_counterpart_selection_is_orientation_independent():
    """`06_observation_v2_127ch.md` section 10.6.

    Ties are broken by the lowest square index *after normalization into the
    acting player's perspective*, so a colour-swapped, rotated position selects
    the mirror image of the same counterpart. Under the superseded
    `observation_v2_127ch` absolute-index rule this test would fail.
    """
    from tests.helpers import make_position, mirror_name, piece_at, play

    # Red's captain ends adjacent to two blue pieces at once. In red's frame the
    # normalization is the identity, so d4 (index 33) beats e5 (index 44).
    red_to_move = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"d4": "sergeant", "e5": "marshal", "j10": "flag"},
        acting_player=RED,
    )
    expected = piece_at(red_to_move, "d4")
    play(red_to_move, "e3 e4")
    chosen = red_to_move.behavior_event(
        piece_at(red_to_move, "e4").piece_id, "threat"
    ).counterpart_piece_id
    assert chosen == expected.piece_id

    # Exactly the same geometry, colours swapped and rotated 180 degrees. The
    # mirror of d4 is g7, whose absolute index (66) is now the *higher* of the
    # two, so only a normalized tie-break picks it.
    blue_to_move = make_position(
        red=mirror_placements_for({"d4": "sergeant", "e5": "marshal", "j10": "flag"}),
        blue=mirror_placements_for({"e3": "captain", "a1": "flag"}),
        acting_player=BLUE,
    )
    mirrored_source = mirror_name("e3")
    mirrored_destination = mirror_name("e4")
    play(blue_to_move, f"{mirrored_source} {mirrored_destination}")

    mirrored_choice = blue_to_move.behavior_event(
        piece_at(blue_to_move, mirrored_destination).piece_id, "threat"
    ).counterpart_piece_id
    assert mirrored_choice == piece_at(blue_to_move, mirror_name("d4")).piece_id


def mirror_placements_for(placements):
    from tests.helpers import mirror_placements

    return mirror_placements(placements)


# Scripted positions that guarantee a live event of each behaviour type, plus a
# genuine counterpart tie. Random mirrored games reach all of these too, but
# these cases pin the coverage required by `07_...` section 9A.
LOOSE_RULES = RulesConfig(battleless_move_limit=10_000, absolute_move_limit=10_000)

SCRIPTED_MIRROR_CASES = {
    "threat": (
        {"e3": "captain", "a1": "flag"},
        {"e5": "sergeant", "j10": "flag"},
        RED,
        ["e3 e4"],
    ),
    "evade": (
        {"e3": "captain", "c3": "miner", "a1": "flag"},
        {"e5": "sergeant", "j9": "scout", "j10": "flag"},
        BLUE,
        ["e5 e4", "e3 e2"],
    ),
    "declined_attack": (
        {"e3": "captain", "c3": "miner", "a1": "flag"},
        {"e4": "sergeant", "j10": "flag"},
        RED,
        ["c3 c4"],
    ),
    "protect": (
        {"e3": "captain", "c3": "miner", "a1": "flag"},
        {"e5": "sergeant", "j9": "scout", "j10": "flag"},
        BLUE,
        ["e5 e4", "c3 d3"],
    ),
    "was_protected": (
        {"e3": "captain", "c3": "miner", "a1": "flag"},
        {"e5": "sergeant", "j9": "scout", "j10": "flag"},
        BLUE,
        ["e5 e4", "c3 d3"],
    ),
    "multiple_counterparts": (
        {"e3": "captain", "a1": "flag"},
        {"d4": "sergeant", "e5": "marshal", "j10": "flag"},
        RED,
        ["e3 e4"],
    ),
}


def build_scripted_mirror_pair(case):
    """Build a scripted position and its colour-swapped, rotated twin."""
    from tests.helpers import make_position, mirror_move, mirror_placements, play

    red, blue, acting, moves = case
    original = make_position(
        red=red, blue=blue, acting_player=acting, rules=LOOSE_RULES
    )
    twin = make_position(
        red=mirror_placements(blue),
        blue=mirror_placements(red),
        acting_player=BLUE if acting == RED else RED,
        rules=LOOSE_RULES,
    )
    for move in moves:
        play(original, move)
        play(twin, mirror_move(move))
    return original, twin


@pytest.mark.parametrize("label", sorted(SCRIPTED_MIRROR_CASES))
def test_scripted_mirrored_positions_match_on_all_channels(label):
    original, twin = build_scripted_mirror_pair(SCRIPTED_MIRROR_CASES[label])

    if label in ("threat", "evade", "declined_attack", "protect", "was_protected"):
        assert any(
            key[1] == label for key in original.behavior_memory
        ), f"scripted case did not record a {label} event"

    for left, right in ((RED, BLUE), (BLUE, RED)):
        first = build_observation(original, left)
        second = build_observation(twin, right)
        differing = [
            channel
            for channel in range(127)
            if not np.array_equal(first[channel], second[channel])
        ]
        assert differing == [], f"{label}: observer pair {left}/{right}, {differing}"


def test_scripted_tie_case_really_has_multiple_eligible_counterparts():
    original, _ = build_scripted_mirror_pair(
        SCRIPTED_MIRROR_CASES["multiple_counterparts"]
    )
    assert len(original.active_threat_relations) >= 2


def test_behaviour_planes_transform_consistently():
    """The whole behavioural block, including counterpart-derived features."""
    original, twin = mirrored_games(5)
    play_mirrored(original, twin, 120, 5)
    channels = list(range(68, 108))
    for left, right in ((RED, BLUE), (BLUE, RED)):
        assert np.array_equal(
            build_observation(original, left)[channels],
            build_observation(twin, right)[channels],
        )


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
