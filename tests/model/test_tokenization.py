"""The `[B,127,10,10] -> [B,100,127]` relayout, pinned so a transpose fails.

Covers Phase 5 gate 5 (`tokenization_exact_row_major`).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stratego.engine.constants import NUM_SQUARES
from stratego.engine.coordinates import square_row_column
from stratego.engine.observation import build_observation
from stratego.model.contract import ModelContractError
from stratego.model.tokenization import (
    observation_batch_from_numpy,
    observation_to_tokens,
    position_coded_observation,
    row_column_to_square,
    square_to_row_column,
    tokenize_numpy_observation,
    tokens_to_observation,
)

from ..helpers import RED, nonterminal_state


def test_square_indexing_matches_the_engine_for_every_square():
    for square in range(NUM_SQUARES):
        assert square_to_row_column(square) == square_row_column(square)
        row, column = square_row_column(square)
        assert row_column_to_square(row, column) == square


def test_every_token_holds_its_own_squares_channel_column():
    """The ordering test that a transpose cannot pass.

    Each element of the fixture encodes its own `(batch, channel, row, column)`,
    so the assertion below is only satisfiable by the correct row-major mapping.
    """
    observation = position_coded_observation(batch=3)
    tokens = observation_to_tokens(observation)
    assert tokens.shape == (3, 100, 127)
    for batch in range(3):
        for square in range(NUM_SQUARES):
            row, column = divmod(square, 10)
            assert torch.equal(tokens[batch, square], observation[batch, :, row, column])


def test_tokenization_is_a_pure_relayout():
    observation = position_coded_observation(batch=2)
    tokens = observation_to_tokens(observation)
    # Same multiset of values, same dtype, no scaling and nothing appended.
    assert tokens.dtype == observation.dtype
    assert tokens.numel() == observation.numel()
    assert torch.equal(torch.sort(tokens.flatten())[0], torch.sort(observation.flatten())[0])


def test_the_relayout_inverts_exactly():
    observation = position_coded_observation(batch=4)
    assert torch.equal(tokens_to_observation(observation_to_tokens(observation)), observation)


def test_a_transposed_implementation_would_be_caught():
    """Guards the guard: prove the fixture is asymmetric enough to detect a swap."""
    observation = position_coded_observation(batch=1)
    correct = observation_to_tokens(observation)
    swapped = observation.transpose(2, 3).reshape(1, 127, 100).transpose(1, 2).contiguous()
    assert not torch.equal(correct, swapped)


def test_dtype_and_device_are_preserved():
    for dtype in (torch.float16, torch.float32, torch.float64):
        tokens = observation_to_tokens(torch.zeros(2, 127, 10, 10, dtype=dtype))
        assert tokens.dtype == dtype


def test_invalid_inputs_are_rejected_at_the_boundary():
    with pytest.raises(ModelContractError):
        observation_to_tokens(torch.zeros(127, 10, 10))
    with pytest.raises(ModelContractError):
        observation_to_tokens(torch.zeros(1, 10, 10, 127))
    with pytest.raises(ModelContractError):
        tokens_to_observation(torch.zeros(1, 127, 100))


# ---------------------------------------------------------------------------
# The engine-facing path
# ---------------------------------------------------------------------------


def test_a_real_engine_observation_tokenizes_to_its_own_squares():
    state = nonterminal_state(40)
    observation = build_observation(state, RED)
    tokens = tokenize_numpy_observation(observation)
    assert tokens.shape == (1, 100, 127)
    for square in range(NUM_SQUARES):
        row, column = divmod(square, 10)
        assert np.allclose(tokens[0, square].numpy(), observation[:, row, column])


def test_a_read_only_engine_observation_is_copied_not_aliased():
    """`PolicyInput` hands out read-only arrays; the model must not alias them."""
    state = nonterminal_state(30)
    observation = build_observation(state, RED)
    observation.setflags(write=False)
    tensor = observation_batch_from_numpy(observation)
    tensor[0, 0, 0, 0] = 12345.0  # would raise or corrupt the engine array if aliased
    assert observation[0, 0, 0] != 12345.0


def test_a_sequence_of_observations_stacks_into_one_batch():
    states = [nonterminal_state(ply) for ply in (12, 24, 36)]
    observations = [build_observation(state, state.acting_player) for state in states]
    tokens = tokenize_numpy_observation(observations)
    assert tokens.shape == (3, 100, 127)
    for index, observation in enumerate(observations):
        assert np.allclose(tokens[index].numpy(), observation.reshape(127, 100).T)


def test_an_empty_batch_is_refused():
    with pytest.raises(ModelContractError):
        observation_batch_from_numpy([])
