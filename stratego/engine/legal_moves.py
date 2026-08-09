"""Legal-action generation and the 10,000-entry legality mask.

Specification sources:

- `01_official_rules.md` sections 3, 4, 5 (movement, Scout, immobility)
- `03_game_engine_spec.md` section 9
- `04_engine_validation_plan.md` sections 4, 5, 6, 13

The no-battle and absolute-move limits are *termination* rules. They never
restrict which moves are legal, and the two-square and continuous-chasing rules
are not implemented at all, so nothing in this module inspects move history.
"""

import numpy as np

from .actions import encode_action
from .constants import ACTION_SPACE_SIZE, IMMOVABLE_TYPES, SCOUT
from .coordinates import NEIGHBOURS, RAYS
from .state import GameState


def generate_actions_for_player(state: GameState, player: int) -> list[int]:
    """All legal action identifiers for `player`, ignoring whose turn it is.

    Returned in ascending order, which makes the list deterministic and lets it
    be compared directly against the mask.
    """
    board = state.board
    pieces = state.pieces
    actions: list[int] = []

    for record in state.pieces_of(player):
        if not record.alive:
            continue
        if record.true_type in IMMOVABLE_TYPES:
            # Flag and Bomb never generate an action in any position.
            continue

        source = record.current_square

        if record.true_type == SCOUT:
            # Cardinal ray movement: any positive number of unobstructed
            # squares. The precomputed rays already stop at board edges and at
            # lakes, so only occupancy has to be handled here.
            for ray in RAYS[source]:
                for destination in ray:
                    occupant = board[destination]
                    if occupant is None:
                        actions.append(encode_action(source, destination))
                        continue
                    # The first occupied square terminates the ray. It is a
                    # legal attack only if the occupant belongs to the opponent.
                    if pieces[occupant].owner != player:
                        actions.append(encode_action(source, destination))
                    break
        else:
            for destination in NEIGHBOURS[source]:
                occupant = board[destination]
                if occupant is None or pieces[occupant].owner != player:
                    actions.append(encode_action(source, destination))

    actions.sort()
    return actions


def has_legal_action(state: GameState, player: int) -> bool:
    """Whether `player` has at least one legal action, with early exit.

    Used by the terminal-condition check, which only needs existence.
    """
    board = state.board
    pieces = state.pieces

    for record in state.pieces_of(player):
        if not record.alive or record.true_type in IMMOVABLE_TYPES:
            continue
        source = record.current_square
        if record.true_type == SCOUT:
            for ray in RAYS[source]:
                for destination in ray:
                    occupant = board[destination]
                    if occupant is None or pieces[occupant].owner != player:
                        return True
                    break
        else:
            for destination in NEIGHBOURS[source]:
                occupant = board[destination]
                if occupant is None or pieces[occupant].owner != player:
                    return True
    return False


def legal_actions(state: GameState) -> list[int]:
    """Legal actions for the acting player. A terminal state has none."""
    if state.terminal:
        return []
    return generate_actions_for_player(state, state.acting_player)


def legal_action_mask(state: GameState, actions: "list[int] | None" = None) -> np.ndarray:
    """Dense `uint8` legality mask of length 10,000.

    Built from the same action list that :func:`legal_actions` returns, so the
    two agree by construction. The consistency tests still compare them
    independently, since agreement is an explicit acceptance gate.
    """
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.uint8)
    if actions is None:
        actions = legal_actions(state)
    if actions:
        mask[np.asarray(actions, dtype=np.int64)] = 1
    return mask


def adjacent_attack_opportunities(state: GameState, player: int) -> dict[int, list[int]]:
    """Legal adjacent attacks available to each of `player`'s pieces.

    Returns `piece_id -> [target_piece_id, ...]` ordered by the target's absolute
    board-square index, the deterministic ordering required by
    `08_internal_state_spec.md` section 11. Long Scout attacks are excluded
    because `06_observation_v2_127ch.md` section 10.3 defines the declined-attack
    opportunity in terms of orthogonal adjacency.

    No hidden piece type is inspected: adjacency and ownership are public facts,
    and an attack is legal regardless of what the defender turns out to be.
    """
    board = state.board
    pieces = state.pieces
    opportunities: dict[int, list[int]] = {}

    for record in state.pieces_of(player):
        if not record.alive or record.true_type in IMMOVABLE_TYPES:
            continue
        targets: list[int] = []
        for destination in NEIGHBOURS[record.current_square]:
            occupant = board[destination]
            if occupant is not None and pieces[occupant].owner != player:
                targets.append(occupant)
        if targets:
            # NEIGHBOURS is already ascending by absolute square index, so the
            # target list inherits that ordering.
            opportunities[record.piece_id] = targets
    return opportunities
