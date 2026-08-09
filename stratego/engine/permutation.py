"""Hidden-identity permutation for the anti-leak acceptance gate.

Specification sources:

- `07_observation_validation_matrix.md` section 11
- `09_public_event_and_replay_schema.md` section 16
- `04_engine_validation_plan.md` section 21.6

The gate compares two privileged states that share an identical public history
but differ in the true types of opponent pieces the observer cannot legally
know. Everything the observer may see must be byte-identical; only privileged
records and belief targets may differ.

A permutation is *valid* only if it preserves every publicly deducible
constraint. Exactly one such constraint exists in this ruleset: a piece that has
moved cannot be a Flag or a Bomb, because both are immovable and the movement is
public. Piece counts are preserved automatically because the permutation is a
rearrangement of the hidden pieces' own types.
"""

import random

import numpy as np

from .constants import IMMOVABLE_TYPES, opponent_of
from .events import filter_events_for_observer, public_board_view, public_setup_view
from .legal_moves import generate_actions_for_player
from .observation import belief_target, build_observation
from .snapshot import clone_state
from .state import GameState


def hidden_opponent_piece_ids(state: GameState, observer: int) -> list[int]:
    """Living opponent pieces whose exact type the observer cannot know.

    Captured pieces are excluded because every capture in this ruleset results
    from combat, and combat reveals both identities to both players.
    """
    opponent = opponent_of(observer)
    return [
        record.piece_id
        for record in state.pieces_of(opponent)
        if record.alive and not record.known_to(observer)
    ]


def permutation_is_valid(state: GameState, piece_ids: list[int], types: list[int]) -> bool:
    """Whether assigning `types` to `piece_ids` preserves public constraints."""
    for piece_id, piece_type in zip(piece_ids, types):
        if piece_type in IMMOVABLE_TYPES and state.pieces[piece_id].has_moved:
            return False
    return True


def _constructive_assignment(
    state: GameState, piece_ids: list[int], types: list[int], rng: random.Random
) -> list[int]:
    """Sample a valid assignment directly instead of by rejection.

    Immovable types are dealt only to pieces that have never moved; the rest are
    shuffled freely. Used as a fallback when uniform shuffling keeps producing
    assignments that violate the moved/immovable constraint, which happens in
    late-game positions where most hidden pieces have moved.
    """
    immovable = [piece_type for piece_type in types if piece_type in IMMOVABLE_TYPES]
    movable = [piece_type for piece_type in types if piece_type not in IMMOVABLE_TYPES]
    rng.shuffle(immovable)
    rng.shuffle(movable)

    unmoved_positions = [
        index for index, piece_id in enumerate(piece_ids) if not state.pieces[piece_id].has_moved
    ]
    rng.shuffle(unmoved_positions)

    assignment: list[int | None] = [None] * len(piece_ids)
    for piece_type in immovable:
        assignment[unmoved_positions.pop()] = piece_type
    for index in range(len(assignment)):
        if assignment[index] is None:
            assignment[index] = movable.pop()
    return assignment  # type: ignore[return-value]


def permute_hidden_identities(
    state: GameState,
    observer: int,
    rng: random.Random,
    shuffle_attempts: int = 4,
) -> tuple[GameState, dict]:
    """Return a clone of `state` with hidden opponent types permuted.

    The second element reports trial bookkeeping used by the acceptance metrics:

    - `attempts`: uniform shuffles tried, including rejected ones;
    - `valid`: whether a valid assignment was produced;
    - `changed`: whether the assignment differs from the original;
    - `hidden_pieces`: how many pieces were eligible for permutation.
    """
    piece_ids = hidden_opponent_piece_ids(state, observer)
    original_types = [state.pieces[piece_id].true_type for piece_id in piece_ids]

    info = {
        "attempts": 0,
        "valid": False,
        "changed": False,
        "hidden_pieces": len(piece_ids),
        "used_fallback": False,
    }

    if len(piece_ids) < 2:
        clone = clone_state(state)
        info["valid"] = True
        return clone, info

    assignment: list[int] | None = None
    for _ in range(shuffle_attempts):
        info["attempts"] += 1
        candidate = list(original_types)
        rng.shuffle(candidate)
        if permutation_is_valid(state, piece_ids, candidate):
            assignment = candidate
            break

    if assignment is None:
        assignment = _constructive_assignment(state, piece_ids, original_types, rng)
        info["used_fallback"] = True
        info["attempts"] += 1
        if not permutation_is_valid(state, piece_ids, assignment):  # pragma: no cover
            return clone_state(state), info

    info["valid"] = True
    info["changed"] = assignment != original_types

    clone = clone_state(state)
    for piece_id, piece_type in zip(piece_ids, assignment):
        clone.pieces[piece_id].true_type = piece_type
    return clone, info


# ---------------------------------------------------------------------------
# The public surface that a permutation must leave untouched
# ---------------------------------------------------------------------------


def public_surface(state: GameState, observer: int) -> dict:
    """Everything `observer` is legally allowed to see.

    `09_public_event_and_replay_schema.md` section 16 lists exactly these
    products as the ones that must remain identical under a hidden-identity
    permutation.
    """
    return {
        "observation": build_observation(state, observer),
        "legal_actions": generate_actions_for_player(state, observer),
        "board_view": public_board_view(state, observer),
        "events": filter_events_for_observer(state.events, observer),
        "setup_view": public_setup_view(state, observer),
    }


def compare_public_surfaces(first: dict, second: dict) -> dict:
    """Count mismatches between two public surfaces, by category."""
    return {
        "observation": 0 if np.array_equal(first["observation"], second["observation"]) else 1,
        "legal_actions": 0 if first["legal_actions"] == second["legal_actions"] else 1,
        "board_view": 0 if first["board_view"] == second["board_view"] else 1,
        "events": 0 if first["events"] == second["events"] else 1,
        "setup_view": 0 if first["setup_view"] == second["setup_view"] else 1,
    }


def belief_targets_differ(first: GameState, second: GameState, observer: int) -> bool:
    """Positive control: privileged belief targets are expected to change."""
    return belief_target(first, observer) != belief_target(second, observer)
