"""Privileged dense belief-learning targets. **Training only.**

.. warning::

   This module reads true hidden piece types. It is a *training label* builder
   and must never be reachable from model inference. It deliberately lives in
   :mod:`stratego.training` rather than :mod:`stratego.model`, and nothing in
   `stratego.model` imports it -- that separation is what the Phase 5
   object-graph audit checks, so please keep it.

Specification sources:

- `08_internal_state_spec.md` section 16 (privileged belief-learning target)
- `06_observation_v2_127ch.md` section 15 (labels, not observation channels)
- Phase 5 single-agent instructions, section 3 ("Belief output") and 5.4

From the sparse engine target to a `[100]` label vector
-------------------------------------------------------
:func:`stratego.engine.observation.belief_target` is the frozen authority. It
returns one entry per **live opponent piece whose type the observer may not
legally know**, as `{piece_id, square, true_type}` with human-readable names and
an *absolute* square. The model's belief head is per token, and token `i` is
*normalized* square `i`, so this module does two things and only two things:

1. maps each absolute square through the observer's perspective table;
2. maps each type name back to its index.

Loss mask semantics (the Phase 5 decision)
------------------------------------------
``mask[square]`` is true **exactly** on squares holding a live opponent piece
whose type is still unresolved for the observer. Everything else is excluded:

- own pieces -- the observer already knows those types, so predicting them is
  free information rather than belief;
- legally revealed opponent pieces -- likewise already known;
- empty squares and the two lakes -- no piece to have a type;
- captured pieces -- not on the board.

Excluded squares carry :data:`~stratego.model.contract.BELIEF_IGNORE_INDEX` in
the label vector, which is the value `cross_entropy(ignore_index=...)` skips, so
a caller that forgets to apply the mask still cannot accidentally train on them.
"""

from __future__ import annotations

import numpy as np

from ..engine.constants import NUM_SQUARES, PIECE_TYPE_NAMES
from ..engine.coordinates import to_perspective
from ..engine.observation import belief_target
from ..engine.state import GameState
from ..model.contract import BELIEF_IGNORE_INDEX, BELIEF_TYPE_COUNT

#: Reverse of `PIECE_TYPE_NAMES`, so the sparse engine target's names can be
#: turned back into the indices the belief head predicts.
PIECE_TYPE_INDEX: dict[str, int] = {name: index for index, name in enumerate(PIECE_TYPE_NAMES)}

#: Recorded in reports so a later phase can tell which convention the targets used.
BELIEF_TARGET_VERSION = "dense_belief_target_v1"
BELIEF_TARGET_SQUARE_FRAME = "perspective_normalized_squares"


def dense_belief_target(
    state: GameState, observer: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """`(labels, mask)` for one position, both indexed by normalized square.

    `labels` is `int64[100]`, holding a piece-type index on supervised squares
    and `BELIEF_IGNORE_INDEX` everywhere else. `mask` is `bool[100]`, true on
    exactly the supervised squares.

    Built from the frozen sparse target so the two can never drift apart.
    """
    if observer is None:
        observer = state.acting_player

    labels = np.full(NUM_SQUARES, BELIEF_IGNORE_INDEX, dtype=np.int64)
    mask = np.zeros(NUM_SQUARES, dtype=bool)

    for entry in belief_target(state, observer):
        square = entry["square"]
        if square is None:  # pragma: no cover - a captured piece has no square
            continue
        normalized = to_perspective(int(square), observer)
        type_index = PIECE_TYPE_INDEX[entry["true_type"]]
        if not 0 <= type_index < BELIEF_TYPE_COUNT:  # pragma: no cover - defensive
            raise ValueError(f"piece type index out of range: {type_index}")
        if mask[normalized]:  # pragma: no cover - two pieces cannot share a square
            raise ValueError(f"two belief targets landed on normalized square {normalized}")
        labels[normalized] = type_index
        mask[normalized] = True

    return labels, mask


def dense_belief_target_batch(
    states: "list[GameState]", observers: "list[int] | None" = None
) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-position targets into `int64[B, 100]` and `bool[B, 100]`."""
    if observers is None:
        observers = [state.acting_player for state in states]
    if len(observers) != len(states):
        raise ValueError("states and observers must have the same length")
    pairs = [dense_belief_target(state, observer) for state, observer in zip(states, observers)]
    labels = np.stack([pair[0] for pair in pairs], axis=0)
    mask = np.stack([pair[1] for pair in pairs], axis=0)
    return labels, mask


def belief_target_summary() -> dict:
    """Serialisable statement of the target convention, for reports."""
    return {
        "belief_target_version": BELIEF_TARGET_VERSION,
        "square_frame": BELIEF_TARGET_SQUARE_FRAME,
        "labels_shape": [NUM_SQUARES],
        "type_count": BELIEF_TYPE_COUNT,
        "ignore_index": BELIEF_IGNORE_INDEX,
        "supervised_squares": "live opponent pieces whose type the observer cannot legally know",
        "excluded_squares": [
            "own pieces",
            "legally revealed opponent pieces",
            "empty squares",
            "lakes",
            "captured pieces",
        ],
        "privileged": True,
        "reachable_from_model_input": False,
    }


__all__ = [
    "BELIEF_TARGET_SQUARE_FRAME",
    "BELIEF_TARGET_VERSION",
    "PIECE_TYPE_INDEX",
    "belief_target_summary",
    "dense_belief_target",
    "dense_belief_target_batch",
]
