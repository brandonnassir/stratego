"""Unresolved opponent inventory planes (channels 56-67).

Covers `07_observation_validation_matrix.md` section 7.

These channels answer "how many copies of each type remain unassigned among the
opponent's unresolved hidden identities", which is deliberately *not* a count of
pieces still alive.
"""

import random

import numpy as np
import pytest

from stratego.engine.constants import (
    BLUE,
    PIECE_COUNTS,
    PIECE_TYPE_BY_NAME,
    PIECE_TYPE_NAMES,
    RED,
)
from stratego.engine.observation import CH_UNRESOLVED_INVENTORY, build_observation
from stratego.engine.setup import random_setup
from stratego.engine.state import create_game
from tests.helpers import make_position, piece_at, play

INVENTORY_CHANNELS = list(range(CH_UNRESOLVED_INVENTORY, CH_UNRESOLVED_INVENTORY + 12))


def inventory_value(state, piece_type, observer=RED):
    observation = build_observation(state, observer)
    plane = observation[CH_UNRESOLVED_INVENTORY + piece_type]
    assert plane.min() == plane.max(), "inventory planes must be spatially constant"
    return float(plane[0, 0])


def fresh_game():
    return create_game(random_setup(random.Random(7)), random_setup(random.Random(8)))


def test_every_inventory_channel_starts_at_one():
    state = fresh_game()
    for observer in (RED, BLUE):
        observation = build_observation(state, observer)
        assert np.allclose(observation[INVENTORY_CHANNELS], 1.0)


@pytest.mark.parametrize("type_name", sorted(PIECE_TYPE_BY_NAME))
def test_revealing_one_identity_reduces_the_channel_by_one_unit(type_name):
    piece_type = PIECE_TYPE_BY_NAME[type_name]
    count = PIECE_COUNTS[piece_type]
    state = fresh_game()

    assert inventory_value(state, piece_type) == pytest.approx(1.0)

    targets = [record for record in state.pieces_of(BLUE) if record.true_type == piece_type]
    targets[0].set_known_to(RED, "combat")
    assert inventory_value(state, piece_type) == pytest.approx((count - 1) / count)


@pytest.mark.parametrize("type_name", sorted(PIECE_TYPE_BY_NAME))
def test_revealing_every_identity_of_a_type_drives_the_channel_to_zero(type_name):
    piece_type = PIECE_TYPE_BY_NAME[type_name]
    state = fresh_game()
    for record in state.pieces_of(BLUE):
        if record.true_type == piece_type:
            record.set_known_to(RED, "combat")
    assert inventory_value(state, piece_type) == pytest.approx(0.0)


def test_inventory_uses_the_observers_knowledge_not_the_hidden_state():
    """Blue's own knowledge of blue pieces must not change red's channels."""
    state = fresh_game()
    before = build_observation(state, RED)[INVENTORY_CHANNELS].copy()
    # Blue already knows all of its own pieces; nothing about red's view changes.
    assert np.array_equal(build_observation(state, RED)[INVENTORY_CHANNELS], before)


def test_capturing_an_already_known_piece_does_not_change_the_channel():
    """Validation-matrix section 7 critical distinction, first half."""
    state = make_position(
        red={"e3": "marshal", "a1": "flag", "d3": "captain"},
        blue={"e4": "sergeant", "j10": "flag", "j9": "scout"},
        acting_player=RED,
        revealed={"e4"},
    )
    sergeant = PIECE_TYPE_BY_NAME["sergeant"]
    before = inventory_value(state, sergeant)

    play(state, "e3 e4")  # the already-known sergeant is captured

    assert inventory_value(state, sergeant) == pytest.approx(before)


def test_capturing_a_hidden_piece_reduces_the_channel_exactly_once():
    """Validation-matrix section 7 critical distinction, second half."""
    state = make_position(
        red={"e3": "marshal", "a1": "flag", "d3": "captain"},
        blue={"e4": "sergeant", "j10": "flag", "j9": "scout"},
        acting_player=RED,
    )
    sergeant = PIECE_TYPE_BY_NAME["sergeant"]
    count = PIECE_COUNTS[sergeant]
    before = inventory_value(state, sergeant)

    play(state, "e3 e4")
    after_capture = inventory_value(state, sergeant)
    assert after_capture == pytest.approx(before - 1 / count)

    # Later plies must not reduce it a second time for the same piece.
    play(state, "j9 i9", "d3 d4")
    assert inventory_value(state, sergeant) == pytest.approx(after_capture)


def test_inventory_is_broadcast_to_every_cell():
    state = fresh_game()
    for record in state.pieces_of(BLUE)[:3]:
        record.set_known_to(RED, "combat")
    observation = build_observation(state, RED)
    for channel in INVENTORY_CHANNELS:
        plane = observation[channel]
        assert np.allclose(plane, plane[0, 0])


def test_inventory_reflects_the_official_counts():
    state = fresh_game()
    blue_pieces = state.pieces_of(BLUE)
    # Reveal one piece of every type and check each denominator individually.
    for piece_type, count in sorted(PIECE_COUNTS.items()):
        record = next(item for item in blue_pieces if item.true_type == piece_type)
        record.set_known_to(RED, "combat")
        assert inventory_value(state, piece_type) == pytest.approx(
            (count - 1) / count
        ), PIECE_TYPE_NAMES[piece_type]
