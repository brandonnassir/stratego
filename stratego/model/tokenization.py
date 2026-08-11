"""The one layout operation between the observation tensor and the model.

Specification sources:

- Phase 5 single-agent instructions, section 3 ("Input") and 4.1
- `06_observation_v2_127ch.md` section 3 (the `(127, 10, 10)` tensor)

The whole module implements exactly one idea:

.. code-block:: text

    [B, 127, 10, 10]  ->  [B, 100, 127]

Token `i` is normalized row-major board square `i`, so token `i` holds the 127
channel values at row ``i // 10``, column ``i % 10``. That is the *same* square
indexing the engine uses (`observation.py` writes through a `(127, 100)` view of
the tensor with `planes[channel, square]`), which is what makes this a pure
relayout rather than an interpretation.

Nothing here infers, reorders, appends, normalizes, scales or otherwise alters a
value. The transformation is a reshape plus a transpose, and
:func:`tokens_to_observation` inverts it exactly. The tests pin the ordering with
position-coded tensors, where a value encodes its own `(channel, row, column)`,
so a transposed implementation cannot accidentally pass.
"""

from __future__ import annotations

import numpy as np
import torch

from ..engine.constants import BOARD_COLUMNS, BOARD_ROWS, NUM_SQUARES, OBSERVATION_CHANNELS
from .contract import ModelContractError, validate_observation_batch, validate_token_batch


def square_to_row_column(square: int) -> tuple[int, int]:
    """Normalized row-major square index -> `(row, column)`.

    Duplicated from the engine's coordinate helper on purpose: this module must
    keep working from the frozen convention itself, and a test asserts the two
    agree for all 100 squares.
    """
    if not 0 <= square < NUM_SQUARES:
        raise ModelContractError(f"square index out of range: {square}")
    return divmod(square, BOARD_COLUMNS)


def row_column_to_square(row: int, column: int) -> int:
    """`(row, column)` -> normalized row-major square index."""
    if not (0 <= row < BOARD_ROWS and 0 <= column < BOARD_COLUMNS):
        raise ModelContractError(f"coordinates out of range: row={row}, column={column}")
    return row * BOARD_COLUMNS + column


def observation_to_tokens(observation: torch.Tensor) -> torch.Tensor:
    """`[B, 127, 10, 10]` -> `[B, 100, 127]`, one token per board square.

    Validates the canonical input boundary first, so a wrong channel count or a
    rank-3 single observation is rejected here rather than surfacing as a
    confusing matrix-multiply error inside the encoder.

    `reshape` flattens `(10, 10)` row-major into the 100 square indices, and the
    transpose moves the channel axis last. Dtype and device are preserved: the
    caller decides precision, not the layout.
    """
    batch = validate_observation_batch(observation)
    flattened = observation.reshape(batch, OBSERVATION_CHANNELS, NUM_SQUARES)
    tokens = flattened.transpose(1, 2)
    # `transpose` returns a non-contiguous view. Making it contiguous costs one
    # copy and keeps every downstream kernel on the fast path, and `contiguous()`
    # never changes a value.
    return tokens.contiguous()


def tokens_to_observation(tokens: torch.Tensor) -> torch.Tensor:
    """`[B, 100, 127]` -> `[B, 127, 10, 10]`. Exact inverse of the above."""
    batch = validate_token_batch(tokens)
    planes = tokens.transpose(1, 2)
    return planes.reshape(batch, OBSERVATION_CHANNELS, BOARD_ROWS, BOARD_COLUMNS).contiguous()


def observation_batch_from_numpy(
    observations: "np.ndarray | list[np.ndarray]",
    *,
    dtype: torch.dtype = torch.float32,
    device: "torch.device | str" = "cpu",
) -> torch.Tensor:
    """Stack engine observations into the canonical `[B, 127, 10, 10]` tensor.

    Accepts a single `(127, 10, 10)` array, a stacked `(B, 127, 10, 10)` array or
    a sequence of single observations. Engine observations are handed out
    read-only (`PolicyInput` sets `writeable=False`), and `torch.from_numpy`
    refuses those, so this copies -- which is also what keeps a model from ever
    writing back into engine-owned memory.
    """
    if isinstance(observations, np.ndarray):
        array = observations
    else:
        stacked = list(observations)
        if not stacked:
            raise ModelContractError("cannot build an observation batch from an empty sequence")
        array = np.stack([np.asarray(entry) for entry in stacked], axis=0)

    array = np.asarray(array)
    if array.ndim == 3:
        array = array[None, ...]
    elif array.ndim != 4:
        raise ModelContractError(
            f"expected a (127, 10, 10) or (B, 127, 10, 10) observation array, got "
            f"shape {array.shape}"
        )

    # `copy=True` is required, not just tidy: engine observations arrive
    # read-only and `torch.as_tensor` would otherwise alias engine-owned memory.
    tensor = torch.as_tensor(np.array(array, copy=True), dtype=dtype, device=device)
    validate_observation_batch(tensor)
    return tensor


def tokenize_numpy_observation(
    observations: "np.ndarray | list[np.ndarray]",
    *,
    dtype: torch.dtype = torch.float32,
    device: "torch.device | str" = "cpu",
) -> torch.Tensor:
    """The full engine-to-model input path: NumPy observation(s) -> `[B, 100, 127]`."""
    return observation_to_tokens(
        observation_batch_from_numpy(observations, dtype=dtype, device=device)
    )


def position_coded_observation(batch: int = 1) -> torch.Tensor:
    """A `[B, 127, 10, 10]` tensor whose every entry encodes its own coordinates.

    ``value = ((b * 127 + channel) * 10 + row) * 10 + column``

    Every element is unique across the whole tensor, so any transposition,
    reversal or axis swap in a layout change produces a different tensor. The
    tokenization tests use this to freeze the row-major ordering; a symmetric
    fixture (zeros, or a channel-constant tensor) would let a transpose pass.
    """
    if batch < 1:
        raise ModelContractError(f"batch must be at least 1, got {batch}")
    indices = torch.arange(
        batch * OBSERVATION_CHANNELS * BOARD_ROWS * BOARD_COLUMNS, dtype=torch.float64
    )
    return indices.reshape(batch, OBSERVATION_CHANNELS, BOARD_ROWS, BOARD_COLUMNS).to(
        torch.float32
    )


__all__ = [
    "observation_batch_from_numpy",
    "observation_to_tokens",
    "position_coded_observation",
    "row_column_to_square",
    "square_to_row_column",
    "tokenize_numpy_observation",
    "tokens_to_observation",
]
