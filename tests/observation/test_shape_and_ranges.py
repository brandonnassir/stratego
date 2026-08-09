"""Global observation acceptance rules and channel metadata.

Covers `07_observation_validation_matrix.md` section 2 and the machine-readable
metadata requirement of section 14 and instruction section 14.
"""

import numpy as np
import pytest

from stratego.engine.constants import (
    BLUE,
    OBSERVATION_CHANNELS,
    OBSERVATION_VERSION,
    RED,
)
from stratego.engine.observation import (
    CH_BATTLELESS_PROGRESS,
    CH_GAME_PROGRESS,
    CH_HIDDEN_OPPONENT_OCCUPANCY,
    CH_KNOWN_OPPONENT_IDENTITY,
    CH_KNOWN_OPPONENT_SETUP,
    CH_LAKE_MASK,
    CH_OPPONENT_BEHAVIOR,
    CH_OPPONENT_MOVED,
    CH_OPPONENT_START_ROW,
    CH_OWN_BEHAVIOR,
    CH_OWN_IDENTITY,
    CH_OWN_KNOWN_TO_OPPONENT,
    CH_OWN_MOVED,
    CH_OWN_SETUP,
    CH_OWN_START_ROW,
    CH_RECENT_MOVES,
    CH_UNRESOLVED_INVENTORY,
    build_observation,
    observation_channel_metadata,
    observation_metadata_document,
)
from tests.helpers import nonterminal_state

BINARY_CHANNELS = (
    list(range(CH_OWN_IDENTITY, CH_OWN_IDENTITY + 12))
    + list(range(CH_KNOWN_OPPONENT_IDENTITY, CH_KNOWN_OPPONENT_IDENTITY + 12))
    + [CH_HIDDEN_OPPONENT_OCCUPANCY, CH_OWN_KNOWN_TO_OPPONENT, CH_OWN_MOVED, CH_OPPONENT_MOVED]
    + list(range(CH_OWN_SETUP, CH_OWN_SETUP + 12))
    + list(range(CH_KNOWN_OPPONENT_SETUP, CH_KNOWN_OPPONENT_SETUP + 12))
    + [CH_LAKE_MASK]
)

COORDINATE_CHANNELS = [CH_OWN_START_ROW, CH_OWN_START_ROW + 1, CH_OPPONENT_START_ROW, CH_OPPONENT_START_ROW + 1]

BEHAVIOR_SPECIAL_CHANNELS = [
    block + 4 * behavior + 3
    for block in (CH_OWN_BEHAVIOR, CH_OPPONENT_BEHAVIOR)
    for behavior in range(5)
]
BEHAVIOR_UNIT_CHANNELS = [
    block + 4 * behavior + offset
    for block in (CH_OWN_BEHAVIOR, CH_OPPONENT_BEHAVIOR)
    for behavior in range(5)
    for offset in (0, 1, 2)
]

SAMPLE_PLIES = (0, 1, 7, 33, 90, 180)


def sample_observations():
    for ply in SAMPLE_PLIES:
        state = nonterminal_state(ply)
        for observer in (RED, BLUE):
            yield build_observation(state, observer)


@pytest.mark.parametrize("ply", SAMPLE_PLIES)
@pytest.mark.parametrize("observer", [RED, BLUE])
def test_shape_and_dtype(ply, observer):
    observation = build_observation(nonterminal_state(ply), observer)
    assert observation.shape == (127, 10, 10)
    assert observation.dtype == np.float32


def test_all_values_are_finite():
    for observation in sample_observations():
        assert np.isfinite(observation).all()


def test_binary_planes_contain_only_zero_and_one():
    for observation in sample_observations():
        values = np.unique(observation[BINARY_CHANNELS])
        assert set(values.tolist()) <= {0.0, 1.0}


def test_coordinate_planes_stay_within_minus_one_and_one():
    for observation in sample_observations():
        block = observation[COORDINATE_CHANNELS]
        assert block.min() >= -1.0
        assert block.max() <= 1.0


def test_recent_move_planes_only_hold_minus_one_zero_and_one():
    channels = list(range(CH_RECENT_MOVES, CH_RECENT_MOVES + 16))
    for observation in sample_observations():
        values = np.unique(observation[channels])
        assert set(values.tolist()) <= {-1.0, 0.0, 1.0}


def test_behaviour_unit_planes_stay_in_zero_to_one():
    for observation in sample_observations():
        block = observation[BEHAVIOR_UNIT_CHANNELS]
        assert block.min() >= 0.0
        assert block.max() <= 1.0


def test_behaviour_special_planes_only_hold_minus_one_zero_and_one():
    for observation in sample_observations():
        values = np.unique(observation[BEHAVIOR_SPECIAL_CHANNELS])
        assert set(values.tolist()) <= {-1.0, 0.0, 1.0}


def test_progress_and_inventory_planes_stay_in_zero_to_one():
    channels = [CH_GAME_PROGRESS, CH_BATTLELESS_PROGRESS] + list(
        range(CH_UNRESOLVED_INVENTORY, CH_UNRESOLVED_INVENTORY + 12)
    )
    for observation in sample_observations():
        block = observation[channels]
        assert block.min() >= 0.0
        assert block.max() <= 1.0


def test_observation_is_deterministic():
    state = nonterminal_state(40)
    assert build_observation(state, RED).tobytes() == build_observation(state, RED).tobytes()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_describes_every_channel_exactly_once():
    metadata = observation_channel_metadata()
    assert len(metadata) == OBSERVATION_CHANNELS
    assert [entry["channel"] for entry in metadata] == list(range(OBSERVATION_CHANNELS))
    assert len({entry["name"] for entry in metadata}) == OBSERVATION_CHANNELS


def test_metadata_entries_carry_the_required_fields():
    for entry in observation_channel_metadata():
        assert entry["observation_version"] == OBSERVATION_VERSION
        assert isinstance(entry["name"], str) and entry["name"]
        assert isinstance(entry["description"], str) and entry["description"]
        low, high = entry["valid_range"]
        assert low < high


def test_metadata_ranges_bound_real_observations():
    metadata = observation_channel_metadata()
    for observation in sample_observations():
        for entry in metadata:
            low, high = entry["valid_range"]
            plane = observation[entry["channel"]]
            assert plane.min() >= low, entry["name"]
            assert plane.max() <= high, entry["name"]


def test_metadata_document_records_shape_and_action_encoding():
    document = observation_metadata_document()
    assert document["observation_version"] == OBSERVATION_VERSION
    assert document["shape"] == [127, 10, 10]
    assert document["dtype"] == "float32"
    assert document["legal_action_mask"]["separate_input"] is True
    assert document["legal_action_mask"]["size"] == 10_000
    assert len(document["channels"]) == 127


def test_channel_group_boundaries_match_the_specification_table():
    """The channel map in `06_observation_v2_127ch.md` section 4."""
    assert (CH_OWN_IDENTITY, CH_KNOWN_OPPONENT_IDENTITY) == (0, 12)
    assert (CH_HIDDEN_OPPONENT_OCCUPANCY, CH_OWN_KNOWN_TO_OPPONENT) == (24, 25)
    assert (CH_OWN_MOVED, CH_OPPONENT_MOVED) == (26, 27)
    assert CH_OWN_START_ROW == 28
    assert CH_OWN_SETUP == 32
    assert CH_KNOWN_OPPONENT_SETUP == 44
    assert CH_UNRESOLVED_INVENTORY == 56
    assert CH_OWN_BEHAVIOR == 68
    assert CH_OPPONENT_BEHAVIOR == 88
    assert CH_RECENT_MOVES == 108
    assert (CH_LAKE_MASK, CH_GAME_PROGRESS, CH_BATTLELESS_PROGRESS) == (124, 125, 126)
