"""Detection of the five approved behavioural events.

Specification sources:

- `06_observation_v2_127ch.md` section 10 (formal definitions)
- `08_internal_state_spec.md` sections 9, 10, 11 (threat relations, memory, ordering)
- `07_observation_validation_matrix.md` section 8 (positive and negative cases)

Every function here works from public geometry only: squares, ownership,
adjacency and stable piece identifiers. No detection decision reads a hidden
`true_type`, which is invariant 11 of `08_internal_state_spec.md` section 18.

Counterpart selection
---------------------
When several counterparts qualify for the same event, the one with the lowest
square index **after normalization into the acting player's perspective** is
selected, evaluated at the moment the specification names for that event.

`observation_v2_127ch` used the raw absolute index here. Absolute indices are
not preserved by the 180 degree perspective rotation, so a position and its
colour-swapped mirror could select non-equivalent counterparts and the
behavioural channels of the two were not mirror images. `observation_v2_1_127ch`
orders by the normalized index instead, which makes the selection depend only on
the geometry the acting player actually sees. For red the normalization is the
identity, so red-to-move selections are unchanged.
"""

from dataclasses import dataclass
from typing import Callable

from .constants import (
    BEHAVIOR_DECLINED_ATTACK,
    BEHAVIOR_EVADE,
    BEHAVIOR_PROTECT,
    BEHAVIOR_THREAT,
    BEHAVIOR_WAS_PROTECTED,
)
from .coordinates import NEIGHBOURS, PERSPECTIVE_TABLES, are_adjacent
from .legal_moves import adjacent_attack_opportunities
from .state import BehaviorEvent, GameState


def normalized_square_key(state: GameState, player: int) -> Callable[[int], int]:
    """Sort key ordering piece identifiers by normalized square index.

    The returned key reads each piece's *current* square, so callers must apply
    it while the board still holds the squares the specification refers to.
    """
    table = PERSPECTIVE_TABLES[player]

    def key(piece_id: int) -> int:
        return table[state.pieces[piece_id].current_square]

    return key


@dataclass
class PreMoveContext:
    """Public facts captured before the selected action mutates the board.

    `08_internal_state_spec.md` section 14 requires turn-start attack
    opportunities and the previous move's threat relations to be captured before
    the transition, because both are destroyed by the move itself.
    """

    mover_piece_id: int
    source: int
    destination: int
    is_attack: bool
    defender_piece_id: int | None
    # actor piece id -> counterpart piece id, already reduced to the single
    # deterministic counterpart for that actor.
    declined_attacks: dict[int, int]
    # Pieces of the acting player that the previous opponent move threatened,
    # mapped to the lowest-index threatener: threatened -> threatener.
    threatened_own_pieces: dict[int, int]
    # Squares of those pieces at turn start, so post-move adjacency comparisons
    # are made against a stable reference.
    threatened_squares: dict[int, int]


def capture_pre_move_context(
    state: GameState, source: int, destination: int
) -> PreMoveContext:
    """Collect the pre-move public context the behavioural rules depend on."""
    player = state.acting_player
    mover_piece_id = state.board[source]
    defender_piece_id = state.board[destination]

    # -- declined attacks (06 section 10.3) --------------------------------
    # Every legal adjacent attack that exists at turn start and is not the
    # action actually chosen counts as declined for the piece that could have
    # made it, whether or not that piece is the one that moves.
    declined_attacks: dict[int, int] = {}
    square_key = normalized_square_key(state, player)
    for actor_id, target_ids in adjacent_attack_opportunities(state, player).items():
        actor_square = state.pieces[actor_id].current_square
        declined = [
            target_id
            for target_id in target_ids
            if not (
                actor_square == source
                and state.pieces[target_id].current_square == destination
            )
        ]
        if declined:
            declined_attacks[actor_id] = min(declined, key=square_key)

    # -- threats created by the previous opponent move (08 section 9) ------
    threatened_own_pieces: dict[int, int] = {}
    threatened_squares: dict[int, int] = {}
    for threatener_id, threatened_id, _ in state.active_threat_relations:
        threatened = state.pieces[threatened_id]
        if threatened.owner != player or not threatened.alive:
            continue
        current = threatened_own_pieces.get(threatened_id)
        # Every relation in the set was created by the same move and therefore
        # shares one threatener, so this comparison never actually has to break
        # a tie. It is kept normalized for consistency with the other rules.
        if current is None or square_key(threatener_id) < square_key(current):
            threatened_own_pieces[threatened_id] = threatener_id
        threatened_squares[threatened_id] = threatened.current_square

    return PreMoveContext(
        mover_piece_id=mover_piece_id,
        source=source,
        destination=destination,
        is_attack=defender_piece_id is not None,
        defender_piece_id=defender_piece_id,
        declined_attacks=declined_attacks,
        threatened_own_pieces=threatened_own_pieces,
        threatened_squares=threatened_squares,
    )


def compute_threat_relations(
    state: GameState, context: PreMoveContext
) -> list[tuple[int, int, int]]:
    """All threat relations created by the resolved move.

    A relation exists for every opponent piece orthogonally adjacent to the
    moved piece once the action has fully resolved, excluding the piece the
    action attacked. `08_internal_state_spec.md` section 9 requires the complete
    set to be retained even though the behavioural `threat` event records only
    one counterpart.
    """
    mover = state.pieces[context.mover_piece_id]
    if not mover.alive:
        # Definition 10.1 condition 3: the actor must survive the move.
        return []

    relations: list[tuple[int, int, int]] = []
    for square in NEIGHBOURS[mover.current_square]:
        occupant_id = state.board[square]
        if occupant_id is None:
            continue
        if occupant_id == context.defender_piece_id:
            # Combat is not additionally recorded as a threat against the
            # piece that was attacked.
            continue
        if state.pieces[occupant_id].owner == mover.owner:
            continue
        relations.append((mover.piece_id, occupant_id, state.total_moves))
    return relations


def detect_threat_counterpart(
    state: GameState, relations: list[tuple[int, int, int]], player: int
) -> int | None:
    """Counterpart for the `threat` event: lowest normalized square index."""
    if not relations:
        return None
    return min(
        (threatened for _, threatened, _ in relations),
        key=normalized_square_key(state, player),
    )


def detect_evade_counterpart(state: GameState, context: PreMoveContext) -> int | None:
    """Counterpart of an `evade` event for the moved piece, if one occurred.

    Definition 10.2: the piece must have been threatened by the immediately
    preceding opponent move, must move to an empty square, and must end
    non-adjacent to the threatener. An attack never counts as an evade.
    """
    if context.is_attack:
        return None
    mover_id = context.mover_piece_id
    if mover_id not in context.threatened_own_pieces:
        return None

    mover = state.pieces[mover_id]
    escaped: list[int] = []
    for threatener_id, threatened_id, _ in state.active_threat_relations:
        if threatened_id != mover_id:
            continue
        threatener = state.pieces[threatener_id]
        if not threatener.alive:
            continue
        if not are_adjacent(mover.current_square, threatener.current_square):
            escaped.append(threatener_id)
    if not escaped:
        return None
    return min(escaped, key=normalized_square_key(state, mover.owner))


def detect_protection(
    state: GameState, context: PreMoveContext
) -> tuple[int, int] | None:
    """Detect a `protect` event; returns `(protected_piece_id, threatener_id)`.

    Definition 10.4: a distinct friendly piece moves to an empty square and
    becomes newly adjacent to a piece that the previous opponent move threatened.
    Version 1 deliberately excludes protection of empty squares, so a protected
    piece must actually exist.
    """
    if context.is_attack:
        return None

    mover_id = context.mover_piece_id
    mover = state.pieces[mover_id]

    candidates: list[int] = []
    for protected_id, threatener_id in context.threatened_own_pieces.items():
        if protected_id == mover_id:
            continue
        protected = state.pieces[protected_id]
        if not protected.alive:
            continue
        protected_square = protected.current_square
        if are_adjacent(context.source, protected_square):
            # Already adjacent before the move, so no new protection.
            continue
        if not are_adjacent(mover.current_square, protected_square):
            continue
        candidates.append(protected_id)

    if not candidates:
        return None
    protected_id = min(candidates, key=normalized_square_key(state, mover.owner))
    return protected_id, context.threatened_own_pieces[protected_id]


def build_behavior_events(
    state: GameState, context: PreMoveContext, threat_relations: list[tuple[int, int, int]]
) -> list[BehaviorEvent]:
    """Produce every behavioural event generated by the resolved move.

    Called after identity knowledge has been updated, which is why the
    `actor_knew_counterpart_type` flags below read the current knowledge state:
    `08_internal_state_spec.md` section 14 places knowledge updates (step 6)
    before behavioural event generation (steps 9 and 11).

    Events are returned in the deterministic emission order required by
    `09_public_event_and_replay_schema.md` section 17: behaviour type first,
    then actor piece identifier.
    """
    ply = state.total_moves
    events: list[BehaviorEvent] = []

    def actor_knows(actor_id: int, counterpart_id: int) -> bool:
        owner = state.pieces[actor_id].owner
        return state.pieces[counterpart_id].known_to(owner)

    # threat
    threat_counterpart = detect_threat_counterpart(
        state, threat_relations, state.pieces[context.mover_piece_id].owner
    )
    if threat_counterpart is not None:
        actor_id = context.mover_piece_id
        events.append(
            BehaviorEvent(
                event_type=BEHAVIOR_THREAT,
                actor_piece_id=actor_id,
                counterpart_piece_id=threat_counterpart,
                event_ply=ply,
                actor_knew_counterpart_type=actor_knows(actor_id, threat_counterpart),
            )
        )

    # evade
    evade_counterpart = detect_evade_counterpart(state, context)
    if evade_counterpart is not None:
        actor_id = context.mover_piece_id
        events.append(
            BehaviorEvent(
                event_type=BEHAVIOR_EVADE,
                actor_piece_id=actor_id,
                counterpart_piece_id=evade_counterpart,
                event_ply=ply,
                actor_knew_counterpart_type=actor_knows(actor_id, evade_counterpart),
            )
        )

    # declined attack
    for actor_id in sorted(context.declined_attacks):
        counterpart_id = context.declined_attacks[actor_id]
        events.append(
            BehaviorEvent(
                event_type=BEHAVIOR_DECLINED_ATTACK,
                actor_piece_id=actor_id,
                counterpart_piece_id=counterpart_id,
                event_ply=ply,
                actor_knew_counterpart_type=actor_knows(actor_id, counterpart_id),
            )
        )

    # protect / was protected
    protection = detect_protection(state, context)
    if protection is not None:
        protected_id, threatener_id = protection
        protector_id = context.mover_piece_id
        events.append(
            BehaviorEvent(
                event_type=BEHAVIOR_PROTECT,
                actor_piece_id=protector_id,
                counterpart_piece_id=protected_id,
                event_ply=ply,
                actor_knew_counterpart_type=actor_knows(protector_id, protected_id),
                context_piece_id=threatener_id,
            )
        )
        events.append(
            BehaviorEvent(
                event_type=BEHAVIOR_WAS_PROTECTED,
                actor_piece_id=protected_id,
                counterpart_piece_id=protector_id,
                event_ply=ply,
                actor_knew_counterpart_type=actor_knows(protected_id, protector_id),
                context_piece_id=threatener_id,
            )
        )

    return events


BEHAVIOR_EMISSION_ORDER = {
    BEHAVIOR_THREAT: 0,
    BEHAVIOR_EVADE: 1,
    BEHAVIOR_DECLINED_ATTACK: 2,
    BEHAVIOR_PROTECT: 3,
    BEHAVIOR_WAS_PROTECTED: 4,
}


def sort_behavior_events(events: list[BehaviorEvent]) -> list[BehaviorEvent]:
    """Order behavioural events by behaviour type, then actor piece identifier."""
    return sorted(
        events,
        key=lambda event: (
            BEHAVIOR_EMISSION_ORDER[event.event_type],
            event.actor_piece_id,
        ),
    )
