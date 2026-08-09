"""State invariant checking.

Specification sources:

- `08_internal_state_spec.md` section 18
- `04_engine_validation_plan.md` sections 16, 21.1
- `PHASE_2_IMPLEMENTATION_INSTRUCTIONS.md` section 20

Every violation raises :class:`InvariantViolation` naming the exact invariant, so
a stress run reports *which* rule broke rather than only that something did.
"""

from .constants import (
    BOMB,
    FLAG,
    IMMOVABLE_TYPES,
    LAKE_SQUARE_SET,
    NUM_SQUARES,
    PIECE_TYPE_NAMES,
    PIECES_PER_PLAYER,
    PLAYERS,
    TERMINAL_REASONS,
    TOTAL_PHYSICAL_PIECES,
)
from .pieces import piece_id_name, piece_owner, piece_setup_slot
from .setup import setup_squares
from .state import GameState

# Reveal reasons the engine is allowed to record. A reveal with any other reason
# means knowledge was granted through an illegal cause.
LEGAL_REVEAL_REASONS = frozenset({"own_piece", "combat", "scout_multisquare"})


class InvariantViolation(AssertionError):
    """Raised when the privileged state violates a documented invariant."""


def capture_baseline(state: GameState) -> tuple:
    """Immutable per-piece facts that must never change during a game."""
    return tuple(
        (record.piece_id, record.owner, record.true_type, record.starting_square)
        for record in state.pieces
    )


def capture_knowledge(state: GameState) -> tuple:
    """Per-piece knowledge flags, for the monotonicity check."""
    return tuple((record.known_to_red, record.known_to_blue) for record in state.pieces)


def check_invariants(
    state: GameState,
    baseline: tuple | None = None,
    previous_knowledge: tuple | None = None,
    check_setup_slots: bool = True,
) -> None:
    """Verify every documented invariant, raising on the first violation.

    `baseline` enables the immutability checks (owner, identifier, starting
    square, true type). `previous_knowledge` enables the monotonicity check.
    Both are optional so the function is usable on a state in isolation.

    `check_setup_slots` verifies that each piece's starting square is the one its
    setup slot implies. That always holds for a real game, but synthetic test
    positions place pieces directly on arbitrary squares, so those tests disable
    the extra check while keeping every other invariant active.
    """
    _check_piece_table(state, check_setup_slots)
    _check_board_agreement(state)
    _check_own_knowledge(state)
    _check_reveal_reasons(state)
    _check_immobile_pieces(state)
    _check_counters(state)
    _check_threat_relations(state)
    _check_behavior_memory(state)

    if baseline is not None:
        _check_immutability(state, baseline)
    if previous_knowledge is not None:
        _check_knowledge_monotonic(state, previous_knowledge)


def _fail(invariant: str, detail: str) -> None:
    raise InvariantViolation(f"{invariant}: {detail}")


def _check_piece_table(state: GameState, check_setup_slots: bool = True) -> None:
    if len(state.pieces) != TOTAL_PHYSICAL_PIECES:
        _fail("piece_record_count", f"expected 80 records, found {len(state.pieces)}")
    for player in PLAYERS:
        records = state.pieces_of(player)
        if len(records) != PIECES_PER_PLAYER:
            _fail(
                "piece_records_per_player",
                f"player {player} has {len(records)} records, expected {PIECES_PER_PLAYER}",
            )
        if any(record.owner != player for record in records):
            _fail("piece_owner_block", f"player {player} block contains a foreign piece")

    for index, record in enumerate(state.pieces):
        if record.piece_id != index:
            _fail(
                "stable_piece_identifier",
                f"record at index {index} carries piece_id {record.piece_id}",
            )
        if record.owner != piece_owner(record.piece_id):
            _fail("piece_owner", f"{piece_id_name(record.piece_id)} owner disagrees with its id")
        if check_setup_slots:
            expected_start = setup_squares(record.owner)[piece_setup_slot(record.piece_id)]
            if record.starting_square != expected_start:
                _fail(
                    "starting_square_matches_slot",
                    f"{piece_id_name(record.piece_id)} starts at {record.starting_square}, "
                    f"slot implies {expected_start}",
                )
        if record.alive:
            if record.current_square is None:
                _fail(
                    "live_piece_has_square",
                    f"{piece_id_name(record.piece_id)} is alive with no square",
                )
            if record.current_square in LAKE_SQUARE_SET:
                _fail(
                    "live_piece_not_on_lake",
                    f"{piece_id_name(record.piece_id)} stands on lake {record.current_square}",
                )
            if record.capture_ply is not None:
                _fail(
                    "live_piece_not_captured",
                    f"{piece_id_name(record.piece_id)} is alive with capture_ply set",
                )
        else:
            if record.current_square is not None:
                _fail(
                    "captured_piece_has_no_square",
                    f"{piece_id_name(record.piece_id)} is captured but holds a square",
                )
            if record.capture_ply is None:
                _fail(
                    "captured_piece_has_capture_ply",
                    f"{piece_id_name(record.piece_id)} is captured without a capture ply",
                )


def _check_board_agreement(state: GameState) -> None:
    if len(state.board) != NUM_SQUARES:
        _fail("board_size", f"board has {len(state.board)} cells")

    seen: dict[int, int] = {}
    for square, piece_id in enumerate(state.board):
        if piece_id is None:
            continue
        if square in LAKE_SQUARE_SET:
            _fail("lake_squares_empty", f"lake square {square} holds a piece")
        if piece_id in seen:
            _fail(
                "single_square_per_piece",
                f"{piece_id_name(piece_id)} appears on squares {seen[piece_id]} and {square}",
            )
        seen[piece_id] = square
        record = state.pieces[piece_id]
        if not record.alive:
            _fail(
                "board_holds_only_live_pieces",
                f"captured {piece_id_name(piece_id)} occupies square {square}",
            )
        if record.current_square != square:
            _fail(
                "board_matches_piece_records",
                f"{piece_id_name(piece_id)} records square {record.current_square} "
                f"but stands on {square}",
            )

    for record in state.pieces:
        if record.alive and state.board[record.current_square] != record.piece_id:
            _fail(
                "board_matches_piece_records",
                f"square {record.current_square} does not point back to "
                f"{piece_id_name(record.piece_id)}",
            )


def _check_own_knowledge(state: GameState) -> None:
    for record in state.pieces:
        if not record.known_to(record.owner):
            _fail(
                "player_knows_own_identities",
                f"{piece_id_name(record.piece_id)} is unknown to its own owner",
            )


def _check_reveal_reasons(state: GameState) -> None:
    for record in state.pieces:
        for reason in (record.reveal_reason_red, record.reveal_reason_blue):
            if reason is not None and reason not in LEGAL_REVEAL_REASONS:
                _fail(
                    "legal_reveal_cause",
                    f"{piece_id_name(record.piece_id)} revealed via {reason!r}",
                )
        # A reveal reason without the matching knowledge flag would mean the
        # flag was cleared after being set.
        if record.reveal_reason_red is not None and not record.known_to_red:
            _fail("knowledge_matches_reveal_reason", f"{piece_id_name(record.piece_id)} (red)")
        if record.reveal_reason_blue is not None and not record.known_to_blue:
            _fail("knowledge_matches_reveal_reason", f"{piece_id_name(record.piece_id)} (blue)")


def _check_immobile_pieces(state: GameState) -> None:
    for record in state.pieces:
        if record.true_type not in IMMOVABLE_TYPES:
            continue
        label = "flag_never_moves" if record.true_type == FLAG else "bomb_never_moves"
        if record.has_moved:
            _fail(label, f"{piece_id_name(record.piece_id)} has has_moved set")
        if record.alive and record.current_square != record.starting_square:
            _fail(
                label,
                f"{PIECE_TYPE_NAMES[record.true_type]} {piece_id_name(record.piece_id)} "
                f"moved from {record.starting_square} to {record.current_square}",
            )


def _check_counters(state: GameState) -> None:
    if state.total_moves < 0:
        _fail("counters_non_negative", f"total_moves={state.total_moves}")
    if state.battleless_moves < 0:
        _fail("counters_non_negative", f"battleless_moves={state.battleless_moves}")
    if state.battleless_moves > state.total_moves:
        _fail(
            "battleless_counter_bounded",
            f"battleless_moves={state.battleless_moves} exceeds total_moves={state.total_moves}",
        )
    if state.acting_player not in PLAYERS:
        _fail("valid_acting_player", f"acting_player={state.acting_player}")
    if state.terminal_reason not in TERMINAL_REASONS:
        _fail("valid_terminal_reason", f"terminal_reason={state.terminal_reason!r}")
    if state.terminal:
        if state.terminal_reason == "not_terminal":
            _fail("terminal_reason_set", "terminal state reports not_terminal")
        if state.is_draw != (state.winner is None):
            _fail("draw_flag_consistent", "is_draw disagrees with winner")
    else:
        if state.terminal_reason != "not_terminal":
            _fail(
                "terminal_reason_clear",
                f"non-terminal state reports {state.terminal_reason!r}",
            )
        if state.winner is not None:
            _fail("winner_only_when_terminal", "non-terminal state has a winner")


def _check_threat_relations(state: GameState) -> None:
    for threatener_id, threatened_id, creation_ply in state.active_threat_relations:
        for piece_id in (threatener_id, threatened_id):
            if not 0 <= piece_id < TOTAL_PHYSICAL_PIECES:
                _fail("threat_relation_piece_ids", f"unknown piece id {piece_id}")
        if state.pieces[threatener_id].owner == state.pieces[threatened_id].owner:
            _fail(
                "threat_relation_is_cross_player",
                f"{piece_id_name(threatener_id)} threatens a friendly piece",
            )
        if creation_ply != state.total_moves:
            _fail(
                "threat_relations_are_current",
                f"relation created at ply {creation_ply}, current ply {state.total_moves}",
            )


def _check_behavior_memory(state: GameState) -> None:
    for (piece_id, behavior_type), event in state.behavior_memory.items():
        if event.actor_piece_id != piece_id or event.event_type != behavior_type:
            _fail(
                "behavior_memory_key_matches_event",
                f"key ({piece_id}, {behavior_type!r}) holds {event.as_tuple()}",
            )
        if event.event_ply > state.total_moves:
            _fail(
                "behavior_event_in_past",
                f"event at ply {event.event_ply} exceeds current ply {state.total_moves}",
            )


def _check_immutability(state: GameState, baseline: tuple) -> None:
    current = capture_baseline(state)
    if current == baseline:
        return
    for index, (now, before) in enumerate(zip(current, baseline)):
        if now == before:
            continue
        names = ("stable_piece_identifier", "piece_owner", "true_type", "starting_square")
        for field_index, field_name in enumerate(names):
            if now[field_index] != before[field_index]:
                _fail(
                    f"{field_name}_never_changes",
                    f"piece index {index}: {before[field_index]} -> {now[field_index]}",
                )


def _check_knowledge_monotonic(state: GameState, previous_knowledge: tuple) -> None:
    for record, (was_known_to_red, was_known_to_blue) in zip(state.pieces, previous_knowledge):
        if was_known_to_red and not record.known_to_red:
            _fail(
                "knowledge_is_monotonic",
                f"red forgot {piece_id_name(record.piece_id)}",
            )
        if was_known_to_blue and not record.known_to_blue:
            _fail(
                "knowledge_is_monotonic",
                f"blue forgot {piece_id_name(record.piece_id)}",
            )


def check_bomb_and_flag_counts(state: GameState) -> None:
    """Sanity check used by the long-run stress harness."""
    for player in PLAYERS:
        records = state.pieces_of(player)
        flags = sum(1 for record in records if record.true_type == FLAG)
        bombs = sum(1 for record in records if record.true_type == BOMB)
        if flags != 1 or bombs != 6:
            _fail(
                "inventory_stable",
                f"player {player} has {flags} flag(s) and {bombs} bomb(s)",
            )
