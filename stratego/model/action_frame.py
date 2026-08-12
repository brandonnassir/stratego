"""The one conversion between absolute engine actions and model actions.

Specification sources:

- Phase 6 Agent 1 instructions, "Action-frame transformations"
- `00_PHASE_6_SEQUENCE_AND_COMMON_CONTRACT.md`, "Approved Phase 6 action-frame
  decision"
- `03_game_engine_spec.md` section 8 (`action_id = 100 * source + destination`)
- `06_observation_v2_127ch.md` section 3 (perspective normalization)

Why this module exists
----------------------
`model_contract_v1` let the network read a perspective-normalized board and then
emit an action in *absolute* engine squares. A single network playing both
colours therefore had to learn "advance" twice: once as decreasing row indices
for blue and once as increasing row indices for red, even though the observation
had already been rotated so that both players' own setup sits at the bottom.
`model_contract_v2` closes that gap by putting the policy head in the same frame
as the tokens, so one weight means one strategic move regardless of colour.

The engine does not move
------------------------
Absolute square identifiers and the 10,000 absolute action identifiers are
frozen (`source_destination_10000_v1`). Nothing here re-encodes, renumbers or
reinterprets an engine action: the conversion is applied at the model boundary
and inverted before the action is ever shown to `apply_action`.

.. code-block:: text

    engine legal actions (absolute)
        -> absolute_legal_actions_to_model  -> model frame
    model selects a model-frame action
        -> model_action_to_absolute          -> absolute
    engine validates and applies the absolute action

The transformation itself
-------------------------
Both endpoints pass through the observation's own coordinate normalization, so
the frames agree by construction rather than by coincidence:

.. code-block:: text

    red   : identity
    blue  : square -> 99 - square, applied to source and destination alike

That is not re-derived here. The tables below are built by calling the frozen
engine helpers :func:`stratego.engine.actions.action_to_perspective` and
:func:`~stratego.engine.actions.action_from_perspective` for every one of the
10,000 identifiers, which is what keeps this module from becoming a second,
competing coordinate convention. Import-time checks then prove that the two
directions compose to the identity in both orders and that each table is a
permutation of `0..9999`, so a bijection failure is impossible to ship.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from ..engine.actions import action_from_perspective, action_to_perspective
from ..engine.constants import ACTION_SPACE_SIZE, PLAYERS
from .contract import ENGINE_ACTION_FRAME, MODEL_ACTION_FRAME, ModelContractError

__all__ = [
    "ENGINE_ACTION_FRAME",
    "MODEL_ACTION_FRAME",
    "ActionFrameError",
    "absolute_action_to_model",
    "absolute_legal_actions_to_model",
    "absolute_legal_mask_to_model",
    "action_frame_summary",
    "model_action_to_absolute",
    "model_legal_actions_to_absolute",
    "model_legal_mask_to_absolute",
    "validate_acting_player",
]


class ActionFrameError(ModelContractError):
    """Raised when an action, a player or a legality product is unusable.

    Subclasses :class:`~stratego.model.contract.ModelContractError` so a frame
    failure is caught by everything that already treats a contract violation as
    fatal. Like every other validator in this package it raises rather than
    repairing: a silently "fixed" action identifier is a move nobody chose.
    """


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------


def _build_tables() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Precompute both directions for both players from the engine helpers.

    A table lookup rather than per-call arithmetic, because the mask conversion
    touches all 10,000 entries at once and NumPy fancy-indexing a precomputed
    permutation is the difference between a microsecond and a millisecond on
    every single decision.
    """
    forward: dict[int, np.ndarray] = {}
    inverse: dict[int, np.ndarray] = {}
    for player in PLAYERS:
        forward[player] = np.fromiter(
            (action_to_perspective(action, player) for action in range(ACTION_SPACE_SIZE)),
            dtype=np.int64,
            count=ACTION_SPACE_SIZE,
        )
        inverse[player] = np.fromiter(
            (action_from_perspective(action, player) for action in range(ACTION_SPACE_SIZE)),
            dtype=np.int64,
            count=ACTION_SPACE_SIZE,
        )

    identity = np.arange(ACTION_SPACE_SIZE, dtype=np.int64)
    for player in PLAYERS:
        for name, table in (("forward", forward[player]), ("inverse", inverse[player])):
            # A permutation, not merely a mapping: without this a many-to-one
            # table would still round-trip for the actions a test happened to
            # sample, while two distinct moves quietly shared one policy logit.
            if not np.array_equal(np.sort(table), identity):
                raise ActionFrameError(
                    f"the {name} action table for player {player} is not a permutation "
                    "of 0..9999; the model action space would not be a bijection"
                )
        if not np.array_equal(inverse[player][forward[player]], identity):
            raise ActionFrameError(
                f"absolute -> model -> absolute is not the identity for player {player}"
            )
        if not np.array_equal(forward[player][inverse[player]], identity):
            raise ActionFrameError(
                f"model -> absolute -> model is not the identity for player {player}"
            )
    return forward, inverse


#: `_TO_MODEL[player][absolute_action] -> model_action`, and its inverse. Kept
#: private: callers go through the functions below so every entry point
#: validates its arguments.
_TO_MODEL, _TO_ABSOLUTE = _build_tables()

# The tables are shared by every caller, so a write through a returned view
# would corrupt the frame for the whole process.
for _table in (*_TO_MODEL.values(), *_TO_ABSOLUTE.values()):
    _table.setflags(write=False)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_acting_player(acting_player: int) -> int:
    """Check that `acting_player` names a real player and return it as an int.

    The conversion is a *per-player* transform, so a wrong or missing player is
    not a cosmetic error: for blue it silently produces the 180-degree-wrong
    move, which the engine would then reject as illegal at a random later ply.
    """
    if isinstance(acting_player, bool) or not isinstance(acting_player, (int, np.integer)):
        raise ActionFrameError(
            f"acting player must be an integer player identifier, got {acting_player!r}"
        )
    player = int(acting_player)
    if player not in PLAYERS:
        raise ActionFrameError(
            f"unknown acting player {player!r}; expected one of {tuple(PLAYERS)}"
        )
    return player


def _validate_action(action_id: int, frame: str) -> int:
    if isinstance(action_id, bool) or not isinstance(action_id, (int, np.integer)):
        raise ActionFrameError(
            f"{frame} action identifier must be an integer, got {action_id!r}"
        )
    action = int(action_id)
    if not 0 <= action < ACTION_SPACE_SIZE:
        raise ActionFrameError(
            f"{frame} action identifier {action} is outside 0..{ACTION_SPACE_SIZE - 1}"
        )
    return action


def _validate_action_array(actions: "Iterable[int]", frame: str) -> np.ndarray:
    """A one-dimensional int64 array of in-range, duplicate-free identifiers."""
    array = np.asarray(list(actions) if not isinstance(actions, np.ndarray) else actions)
    if array.size == 0:
        # Not merely unusual: a non-terminal position always has at least one
        # legal action, so an empty product means the caller lost the list.
        raise ActionFrameError(f"the {frame} legal-action product is empty")
    if array.ndim != 1:
        raise ActionFrameError(
            f"the {frame} legal-action product must be one-dimensional, got shape {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise ActionFrameError(
            f"the {frame} legal-action product must hold integers, got dtype {array.dtype}"
        )
    array = array.astype(np.int64, copy=False)
    if array.min() < 0 or array.max() >= ACTION_SPACE_SIZE:
        raise ActionFrameError(
            f"a {frame} legal action identifier is outside 0..{ACTION_SPACE_SIZE - 1}"
        )
    if np.unique(array).size != array.size:
        raise ActionFrameError(f"the {frame} legal-action product contains a duplicate")
    return array


def _validate_mask(mask: np.ndarray, frame: str) -> np.ndarray:
    """A dense length-10,000 legality mask holding only 0/1 (or booleans)."""
    array = np.asarray(mask)
    if array.ndim != 1 or array.shape[0] != ACTION_SPACE_SIZE:
        raise ActionFrameError(
            f"the {frame} legality mask must be a flat length-{ACTION_SPACE_SIZE} array, "
            f"got shape {array.shape}"
        )
    if array.dtype != bool and not np.isin(array, (0, 1)).all():
        raise ActionFrameError(f"the {frame} legality mask holds a value other than 0 or 1")
    return array


# ---------------------------------------------------------------------------
# Single actions
# ---------------------------------------------------------------------------


def absolute_action_to_model(action_id: int, acting_player: int) -> int:
    """Absolute engine action -> the acting player's normalized model action."""
    player = validate_acting_player(acting_player)
    return int(_TO_MODEL[player][_validate_action(action_id, ENGINE_ACTION_FRAME)])


def model_action_to_absolute(action_id: int, acting_player: int) -> int:
    """Normalized model action -> the absolute engine action to apply.

    This is the last step before a `PolicyResult` is built, and the engine
    validates the result independently afterwards.
    """
    player = validate_acting_player(acting_player)
    return int(_TO_ABSOLUTE[player][_validate_action(action_id, MODEL_ACTION_FRAME)])


# ---------------------------------------------------------------------------
# Legal-action lists
# ---------------------------------------------------------------------------


def absolute_legal_actions_to_model(
    legal_actions: "Sequence[int]", acting_player: int
) -> tuple[int, ...]:
    """The engine's legal actions, in the model frame, ascending.

    Sorted because the adapter's tie-break is defined as "the lowest identifier
    among the maximal logits" and the identifiers being ordered is what makes
    that reproducible. Sorting happens *after* conversion, so the order is the
    model frame's order -- for blue that is a different order than the engine's,
    which is exactly the point: the tie-break must not depend on colour.
    """
    player = validate_acting_player(acting_player)
    array = _validate_action_array(legal_actions, ENGINE_ACTION_FRAME)
    return tuple(int(action) for action in np.sort(_TO_MODEL[player][array]))


def model_legal_actions_to_absolute(
    legal_actions: "Sequence[int]", acting_player: int
) -> tuple[int, ...]:
    """The inverse of :func:`absolute_legal_actions_to_model`, ascending."""
    player = validate_acting_player(acting_player)
    array = _validate_action_array(legal_actions, MODEL_ACTION_FRAME)
    return tuple(int(action) for action in np.sort(_TO_ABSOLUTE[player][array]))


# ---------------------------------------------------------------------------
# Dense masks
# ---------------------------------------------------------------------------


def absolute_legal_mask_to_model(mask: np.ndarray, acting_player: int) -> np.ndarray:
    """The engine's dense legality mask, permuted into the model frame.

    A scatter rather than a gather: `model[to_model[a]] = absolute[a]` is the
    definition of the frame change, and writing it that way means the result
    does not depend on the transform happening to be an involution. The dtype is
    preserved so a `uint8` engine mask stays a `uint8` mask.
    """
    player = validate_acting_player(acting_player)
    array = _validate_mask(mask, ENGINE_ACTION_FRAME)
    converted = np.zeros_like(array)
    converted[_TO_MODEL[player]] = array
    return converted


def model_legal_mask_to_absolute(mask: np.ndarray, acting_player: int) -> np.ndarray:
    """The inverse of :func:`absolute_legal_mask_to_model`."""
    player = validate_acting_player(acting_player)
    array = _validate_mask(mask, MODEL_ACTION_FRAME)
    converted = np.zeros_like(array)
    converted[_TO_ABSOLUTE[player]] = array
    return converted


# ---------------------------------------------------------------------------
# Machine-readable description
# ---------------------------------------------------------------------------


def action_frame_summary() -> dict:
    """Serialisable statement of the frames, for reports and handoff notes."""
    return {
        "engine_action_frame": ENGINE_ACTION_FRAME,
        "model_action_frame": MODEL_ACTION_FRAME,
        "action_space_size": ACTION_SPACE_SIZE,
        "red_transform": "identity",
        "blue_transform": "square -> 99 - square, applied to source and destination",
        "action_index_rule": "action_id = 100 * source + destination",
        "implementation": "stratego.engine.actions.action_to_perspective",
    }
