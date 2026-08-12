"""The absolute/normalized action frame: bijection, geometry and legality.

Covers the Phase 6 Agent 1 gates `action_round_trip`, `reverse_round_trip` and
`legal_action_and_mask_equivalence`.

The exhaustive audits here are the whole reason the conversion is allowed to be
a table lookup on the hot path: correctness is established once, over the entire
10,000-identifier space and both players, rather than sampled.
"""

from __future__ import annotations

import numpy as np
import pytest

from stratego.engine.actions import decode_action, encode_action
from stratego.engine.constants import ACTION_SPACE_SIZE, BLUE, PLAYERS, RED, TRAINING_RULES
from stratego.engine.coordinates import square_name, to_perspective
from stratego.engine.legal_moves import legal_action_mask, legal_actions
from stratego.engine.random_play import play_random_game_to_ply
from stratego.model.action_frame import (
    ActionFrameError,
    absolute_action_to_model,
    absolute_legal_actions_to_model,
    absolute_legal_mask_to_model,
    action_frame_summary,
    model_action_to_absolute,
    model_legal_actions_to_absolute,
    model_legal_mask_to_absolute,
)


def position_corpus(plies=(0, 7, 18, 40, 75, 120, 190, 260), seed: int = 0) -> list:
    """Real non-terminal positions spread across both colours and the game.

    Built by playing real random games rather than by assembling boards, so the
    legality products under test are the ones the engine actually produces. The
    ply list crosses the acting player back and forth, which matters here: the
    conversion is a per-player transform and a corpus of one colour would pass
    with the player argument ignored entirely.
    """
    corpus = []
    for ply in plies:
        for attempt in range(seed, seed + 200):
            state = play_random_game_to_ply(attempt, ply, rules=TRAINING_RULES)
            if not state.terminal and state.total_moves == ply:
                corpus.append(state)
                break
    return corpus


CORPUS = position_corpus()


# ---------------------------------------------------------------------------
# Exhaustive bijection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("player", sorted(PLAYERS))
def test_every_absolute_action_round_trips(player):
    """10,000 actions x 2 players: absolute -> model -> absolute is the identity."""
    mismatches = [
        action
        for action in range(ACTION_SPACE_SIZE)
        if model_action_to_absolute(absolute_action_to_model(action, player), player) != action
    ]
    assert mismatches == []


@pytest.mark.parametrize("player", sorted(PLAYERS))
def test_every_model_action_round_trips(player):
    """The reverse direction, over the same 10,000 identifiers."""
    mismatches = [
        action
        for action in range(ACTION_SPACE_SIZE)
        if absolute_action_to_model(model_action_to_absolute(action, player), player) != action
    ]
    assert mismatches == []


@pytest.mark.parametrize("player", sorted(PLAYERS))
def test_the_frame_change_is_a_bijection_with_no_collisions(player):
    """Distinctness, not just round-tripping.

    A many-to-one map can still round-trip on the values a test samples while
    two different moves share a policy logit. Comparing the sorted image with
    `0..9999` rules that out for the whole space at once.
    """
    image = [absolute_action_to_model(action, player) for action in range(ACTION_SPACE_SIZE)]
    assert len(set(image)) == ACTION_SPACE_SIZE
    assert sorted(image) == list(range(ACTION_SPACE_SIZE))


def test_red_is_the_identity_and_blue_is_not():
    assert all(absolute_action_to_model(a, RED) == a for a in range(ACTION_SPACE_SIZE))
    moved = [a for a in range(ACTION_SPACE_SIZE) if absolute_action_to_model(a, BLUE) != a]
    # No fixed points at all: `99 - s == s` has no integer solution because the
    # board has an even number of squares, so blue's transform moves every one of
    # the 10,000 identifiers. An adapter that forgot to convert for blue -- the
    # most likely way to get this wrong -- cannot hide anywhere in the space.
    assert len(moved) == ACTION_SPACE_SIZE


def test_the_transform_agrees_with_the_engine_helper_on_both_endpoints():
    """The conversion must *be* the observation's normalization, not resemble it."""
    for player in sorted(PLAYERS):
        for action in range(0, ACTION_SPACE_SIZE, 7):
            source, destination = decode_action(action)
            expected = encode_action(
                to_perspective(source, player), to_perspective(destination, player)
            )
            assert absolute_action_to_model(action, player) == expected


# ---------------------------------------------------------------------------
# Pinned geometry
# ---------------------------------------------------------------------------
#
# Representative moves written out by hand, so a future refactor that changes the
# convention has to change a human-readable expectation rather than a formula
# that agrees with itself.


@pytest.mark.parametrize(
    "source_name, destination_name, blue_source_name, blue_destination_name, note",
    [
        ("a1", "a2", "j10", "j9", "first square, single step forward"),
        ("j10", "j9", "a1", "a2", "last square, single step"),
        ("a1", "b1", "j10", "i10", "first row, lateral"),
        ("a10", "a9", "j1", "j2", "first column, top row"),
        ("j1", "j2", "a10", "a9", "last column, bottom row"),
        ("a1", "a10", "j10", "j1", "full-column scout run"),
        ("a5", "j5", "j6", "a6", "full-row scout run"),
        ("e5", "e6", "f6", "f5", "centre, adjacent to the lakes"),
        ("d4", "d7", "g7", "g4", "long scout past a lake column"),
    ],
)
def test_representative_geometry_is_pinned(
    source_name, destination_name, blue_source_name, blue_destination_name, note
):
    source = _square(source_name)
    destination = _square(destination_name)
    action = encode_action(source, destination)

    assert absolute_action_to_model(action, RED) == action, note

    blue_model = absolute_action_to_model(action, BLUE)
    blue_source, blue_destination = decode_action(blue_model)
    assert square_name(blue_source) == blue_source_name, note
    assert square_name(blue_destination) == blue_destination_name, note
    assert model_action_to_absolute(blue_model, BLUE) == action, note


def test_move_length_and_orientation_survive_the_frame_change():
    """A rotation preserves distance and axis; only direction flips.

    This is the property that makes the frame change *learnable* rather than
    merely reversible, so it is asserted directly over the whole space.
    """
    for action in range(0, ACTION_SPACE_SIZE, 13):
        source, destination = decode_action(action)
        model_source, model_destination = decode_action(
            absolute_action_to_model(action, BLUE)
        )
        assert (destination - source) == -(model_destination - model_source)
        assert (source // 10 == destination // 10) == (
            model_source // 10 == model_destination // 10
        )


def _square(name: str) -> int:
    from stratego.engine.coordinates import square_from_name

    return square_from_name(name)


# ---------------------------------------------------------------------------
# Legality products over real positions
# ---------------------------------------------------------------------------


def test_the_corpus_covers_both_colours_and_a_range_of_positions():
    """Guards the corpus itself: a silently red-only corpus proves much less."""
    assert len({state.acting_player for state in CORPUS}) == 2
    assert len(CORPUS) >= 8
    assert max(len(legal_actions(state)) for state in CORPUS) > 1


def test_the_converted_list_and_mask_agree_on_every_position():
    """Two independently converted legality products must describe one set."""
    for state in CORPUS:
        player = state.acting_player
        absolute = legal_actions(state)
        mask = legal_action_mask(state, absolute)

        model_actions = absolute_legal_actions_to_model(absolute, player)
        model_mask = absolute_legal_mask_to_model(mask, player)

        assert list(model_actions) == sorted(np.flatnonzero(model_mask).tolist())
        assert len(model_actions) == len(absolute)
        assert int(model_mask.sum()) == int(mask.sum())
        assert model_mask.dtype == mask.dtype


def test_the_normalized_legal_set_maps_back_to_the_engine_set_exactly():
    for state in CORPUS:
        player = state.acting_player
        absolute = legal_actions(state)
        model_actions = absolute_legal_actions_to_model(absolute, player)
        restored = model_legal_actions_to_absolute(model_actions, player)
        assert set(restored) == set(absolute)
        assert len(restored) == len(absolute)


def test_the_dense_mask_round_trips_bit_for_bit():
    for state in CORPUS:
        player = state.acting_player
        mask = legal_action_mask(state)
        restored = model_legal_mask_to_absolute(
            absolute_legal_mask_to_model(mask, player), player
        )
        assert np.array_equal(restored, mask)


def test_a_blue_position_actually_moves_its_legal_set():
    """Otherwise every conversion assertion above could pass on a no-op."""
    blue_states = [state for state in CORPUS if state.acting_player == BLUE]
    assert blue_states, "the corpus must contain a blue-to-move position"
    for state in blue_states:
        absolute = legal_actions(state)
        model_actions = absolute_legal_actions_to_model(absolute, BLUE)
        assert set(model_actions) != set(absolute)


def test_converting_with_the_wrong_player_produces_a_different_set():
    """Pins that the player argument is used, not accepted and ignored."""
    for state in CORPUS:
        absolute = legal_actions(state)
        as_red = absolute_legal_actions_to_model(absolute, RED)
        as_blue = absolute_legal_actions_to_model(absolute, BLUE)
        assert set(as_red) != set(as_blue)


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("player", [2, -1, "red", None, True])
def test_an_unknown_player_is_refused(player):
    with pytest.raises(ActionFrameError):
        absolute_action_to_model(0, player)


@pytest.mark.parametrize("action", [-1, ACTION_SPACE_SIZE, 10_001, 1.5, "0", None])
def test_an_out_of_range_action_is_refused(action):
    with pytest.raises(ActionFrameError):
        absolute_action_to_model(action, RED)
    with pytest.raises(ActionFrameError):
        model_action_to_absolute(action, RED)


@pytest.mark.parametrize("actions", [[], [0, 0], [0, ACTION_SPACE_SIZE], [[0, 1], [2, 3]]])
def test_a_malformed_legal_action_product_is_refused(actions):
    with pytest.raises(ActionFrameError):
        absolute_legal_actions_to_model(actions, RED)


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros(ACTION_SPACE_SIZE - 1, dtype=np.uint8),
        np.zeros((2, ACTION_SPACE_SIZE), dtype=np.uint8),
        np.full(ACTION_SPACE_SIZE, 2, dtype=np.uint8),
    ],
)
def test_a_malformed_mask_is_refused(mask):
    with pytest.raises(ActionFrameError):
        absolute_legal_mask_to_model(mask, RED)


def test_the_summary_states_both_frames():
    summary = action_frame_summary()
    assert summary["engine_action_frame"] == "absolute_engine_squares"
    assert summary["model_action_frame"] == "perspective_normalized_squares"
    assert summary["action_space_size"] == ACTION_SPACE_SIZE
