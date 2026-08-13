"""Atomic state transitions.

Specification sources:

- `03_game_engine_spec.md` section 10 (atomicity)
- `08_internal_state_spec.md` section 14 (conceptual transition order)
- `09_public_event_and_replay_schema.md` section 17 (event ordering)
- `02_project_ruleset.md` sections 3, 4 (draw limits)

`apply_action` mutates the state in place, but only after the action has been
fully validated against the legal-action list. No field is touched on the
illegal path, which is what makes "an illegal action leaves the state
completely unchanged" true by construction rather than by careful bookkeeping.
"""

from bisect import bisect_left

from .actions import decode_action
from .behavior import (
    build_behavior_events,
    capture_pre_move_context,
    compute_threat_relations,
    sort_behavior_events,
)
from .combat import ATTACKER_WINS, BOTH_REMOVED, DEFENDER_WINS, resolve_combat
from .constants import (
    BOARD_COLUMNS,
    FLAG,
    NOT_TERMINAL,
    PHASE_TERMINAL,
    PLAYERS,
    TERMINAL_ABSOLUTE_MOVE_LIMIT_DRAW,
    TERMINAL_BATTLELESS_MOVE_LIMIT_DRAW,
    TERMINAL_BOTH_NO_LEGAL_MOVE_DRAW,
    TERMINAL_FLAG_CAPTURE,
    TERMINAL_OPPONENT_NO_LEGAL_MOVE,
    opponent_of,
)
from .events import (
    make_behavior_event,
    make_combat_event,
    make_game_end_event,
    make_identity_reveal_event,
    make_move_event,
)
from .legal_moves import has_legal_action, legal_actions
from .pieces import PieceRecord
from .state import GameState, RecentMove


class IllegalActionError(ValueError):
    """Raised when an action is not in the current legal-action list."""


class TerminalStateError(ValueError):
    """Raised when a transition is attempted on a finished game."""


def _capture(record: PieceRecord, ply: int) -> None:
    """Remove a piece from play. Board bookkeeping is done by the caller."""
    record.alive = False
    record.current_square = None
    record.capture_ply = ply


def apply_action(
    state: GameState, action_id: int, legal: "list[int] | None" = None
) -> list[dict]:
    """Apply one legal action and return the events it generated.

    `legal` may be supplied when the caller already generated the legal-action
    list (a random agent does), which avoids regenerating it purely to validate.
    """
    if state.terminal:
        raise TerminalStateError(
            f"game {state.game_id} is already terminal ({state.terminal_reason}); "
            "reset before applying further actions"
        )

    if legal is None:
        legal = legal_actions(state)
    # `legal` is ascending, so membership is a binary search.
    position = bisect_left(legal, action_id)
    if position >= len(legal) or legal[position] != action_id:
        raise IllegalActionError(
            f"action {action_id} is not legal for "
            f"{'red' if state.acting_player == 0 else 'blue'} in game {state.game_id}"
        )

    source, destination = decode_action(action_id)
    player = state.acting_player
    opponent = opponent_of(player)
    ply = state.total_moves + 1

    mover_id = state.board[source]
    mover = state.pieces[mover_id]
    defender_id = state.board[destination]

    # Step 2/3: capture turn-start context before anything moves.
    context = capture_pre_move_context(state, source, destination)

    source_row, source_column = divmod(source, BOARD_COLUMNS)
    destination_row, destination_column = divmod(destination, BOARD_COLUMNS)
    distance = abs(destination_row - source_row) + abs(destination_column - source_column)

    # ---- Step 5: move / resolve combat atomically ------------------------
    combat_outcome = None
    flag_captured = False
    defender: PieceRecord | None = None

    state.board[source] = None
    mover.has_moved = True

    if defender_id is None:
        state.board[destination] = mover_id
        mover.current_square = destination
    else:
        defender = state.pieces[defender_id]
        combat_outcome = resolve_combat(mover.true_type, defender.true_type)
        flag_captured = defender.true_type == FLAG
        if combat_outcome == ATTACKER_WINS:
            _capture(defender, ply)
            state.board[destination] = mover_id
            mover.current_square = destination
        elif combat_outcome == DEFENDER_WINS:
            _capture(mover, ply)
            state.board[destination] = defender_id
        else:  # BOTH_REMOVED
            _capture(mover, ply)
            _capture(defender, ply)
            state.board[destination] = None

    # ---- Step 6: identity knowledge --------------------------------------
    reveal_events: list[dict] = []
    if defender is not None:
        # Combat makes both identities public to both players.
        for record in sorted((mover, defender), key=lambda item: item.piece_id):
            newly_known = [
                observer for observer in PLAYERS if record.set_known_to(observer, "combat")
            ]
            if newly_known:
                reveal_events.append(
                    make_identity_reveal_event(
                        ply, record.piece_id, record.owner, record.true_type, "combat", newly_known
                    )
                )
    elif distance >= 2:
        # Only a Scout can legally move more than one square, so the move itself
        # identifies the piece to the opponent.
        if mover.set_known_to(opponent, "scout_multisquare"):
            reveal_events.append(
                make_identity_reveal_event(
                    ply,
                    mover.piece_id,
                    mover.owner,
                    mover.true_type,
                    "scout_multisquare",
                    [opponent],
                )
            )

    # ---- Step 8: counters ------------------------------------------------
    state.total_moves = ply
    if combat_outcome is None:
        state.battleless_moves += 1
    else:
        state.battleless_moves = 0

    # ---- Steps 9-11: behavioural events and new threat relations ---------
    threat_relations = compute_threat_relations(state, context)
    behavior_events = sort_behavior_events(
        build_behavior_events(state, context, threat_relations)
    )
    for event in behavior_events:
        state.behavior_memory[(event.actor_piece_id, event.event_type)] = event
    state.active_threat_relations = threat_relations

    # ---- Step 12: recent-move history ------------------------------------
    state.recent_moves.append(
        RecentMove(
            ply=ply,
            player=player,
            piece_id=mover_id,
            source=source,
            destination=destination,
            destination_had_opponent=defender_id is not None,
            target_piece_id=defender_id,
        )
    )

    # ---- Step 13: terminal conditions ------------------------------------
    _evaluate_terminal(state, player, opponent, flag_captured)

    # ---- Step 14: acting player ------------------------------------------
    if not state.terminal:
        state.acting_player = opponent

    # ---- Event emission in the documented order --------------------------
    generated: list[dict] = [
        make_move_event(
            ply, player, mover_id, source, destination, distance, defender_id is not None, defender_id
        )
    ]
    generated.extend(reveal_events)
    if defender is not None:
        generated.append(
            make_combat_event(
                ply,
                mover_id,
                defender_id,
                mover.true_type,
                defender.true_type,
                combat_outcome,
                flag_captured,
            )
        )
    generated.extend(make_behavior_event(event) for event in behavior_events)
    if state.terminal:
        generated.append(make_game_end_event(state))

    state.events.extend(generated)
    state.action_history.append(action_id)
    return generated


def _evaluate_mobility_terminal(state: GameState, next_mover: int, other: int) -> bool:
    """The mobility-termination rule, in one place for every call site.

    A game ends when the player about to move has no legal action
    (`01_official_rules.md` section 8): if the other player possesses a legal
    move they win (`opponent_no_legal_move`), and if neither does the game is a
    draw (`both_no_legal_move_draw`). Returns whether the game was finished.

    After a transition, `next_mover` is the opponent of the player who just
    moved; at game creation it is the configured first player. Both call sites
    share this implementation so there is exactly one interpretation of the
    rule.
    """
    if has_legal_action(state, next_mover):
        return False
    if has_legal_action(state, other):
        _finish(state, TERMINAL_OPPONENT_NO_LEGAL_MOVE, winner=other)
    else:
        _finish(state, TERMINAL_BOTH_NO_LEGAL_MOVE_DRAW, winner=None)
    return True


def evaluate_initial_terminal(state: GameState) -> None:
    """Evaluate the mobility rule for a freshly created ply-0 state.

    `phase2_1_reference_1.2.0`: a uniformly shuffled legal setup can place
    Flag and Bombs on all six front-row squares whose forward square is not a
    lake, leaving the first player with no legal move before any action is
    applied. Until 1.2.0 that rules-terminal position entered play labelled
    active, because `_evaluate_terminal` runs only inside `apply_action`; the
    Phase 6B production soak aborted on exactly such a state
    (`batch60006-env000112-gen000098`). Flag capture cannot apply at ply 0 and
    the draw counters are zero there, so mobility is the only condition that
    can end a game at creation and the frozen precedence order is unchanged.

    A game decided here still emits its `game_end` event, preserving the event
    contract that every finished game carries exactly one, as its final event.
    """
    if _evaluate_mobility_terminal(
        state, state.acting_player, opponent_of(state.acting_player)
    ):
        state.events.append(make_game_end_event(state))


def _evaluate_terminal(
    state: GameState, player: int, opponent: int, flag_captured: bool
) -> None:
    """Set terminal fields, following the project precedence order.

    Precedence, highest first (`02_project_ruleset.md` section 9A):

    1. `flag_capture`
    2. `opponent_no_legal_move`
    3. `both_no_legal_move_draw`
    4. `battleless_move_limit_draw`
    5. `absolute_move_limit_draw`

    Genuine Stratego game-ending conditions outrank the project's own training
    termination limits. The ordering is only observable when a single move both
    reaches a draw threshold and settles the mobility question; in every other
    position at most one condition applies.
    """
    rules = state.rules

    if flag_captured:
        _finish(state, TERMINAL_FLAG_CAPTURE, winner=player)
        return

    if _evaluate_mobility_terminal(state, opponent, player):
        # The player about to move (the opponent of the mover) is stranded, or
        # both players are (`01_official_rules.md` section 8).
        return

    if state.battleless_moves >= rules.battleless_move_limit:
        _finish(state, TERMINAL_BATTLELESS_MOVE_LIMIT_DRAW, winner=None)
        return
    if state.total_moves >= rules.absolute_move_limit:
        _finish(state, TERMINAL_ABSOLUTE_MOVE_LIMIT_DRAW, winner=None)


def _finish(state: GameState, reason: str, winner: int | None) -> None:
    state.terminal = True
    state.phase = PHASE_TERMINAL
    state.terminal_reason = reason
    state.winner = winner
    state.is_draw = winner is None


def apply_action_by_squares(state: GameState, source: int, destination: int) -> list[dict]:
    """Convenience wrapper used by tests and manual inspection examples."""
    from .actions import encode_action

    return apply_action(state, encode_action(source, destination))


def is_terminal(state: GameState) -> bool:
    return state.terminal


def terminal_reason(state: GameState) -> str:
    return state.terminal_reason if state.terminal else NOT_TERMINAL
