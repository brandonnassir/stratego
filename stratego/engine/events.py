"""Derived engine events, observer filtering and browser-safe board views.

Specification source: `09_public_event_and_replay_schema.md` (all sections).

Three products are kept distinct:

1. *derived engine events* -- the full public fact stream produced by a move;
2. *observer-filtered events* -- what a specific player or browser may receive;
3. *observer-filtered board view* -- a serialisable current position.

None of the three ever contains a hidden `true_type`. Types appear only after
the corresponding legal revelation, which is exactly why the reveal event
carries the type and the move event does not.
"""

from .constants import (
    DRAW_TERMINAL_REASONS,
    LAKE_SQUARE_SET,
    NUM_SQUARES,
    PIECE_TYPE_NAMES,
    PLAYER_NAMES,
)
from .coordinates import square_name
from .pieces import piece_id_name
from .state import BehaviorEvent, GameState

EVENT_MOVE = "move"
EVENT_IDENTITY_REVEAL = "identity_reveal"
EVENT_COMBAT = "combat"
EVENT_BEHAVIOR = "behavior"
EVENT_GAME_END = "game_end"


def make_move_event(
    ply: int,
    player: int,
    piece_id: int,
    source: int,
    destination: int,
    distance: int,
    is_attack: bool,
    target_piece_id: int | None,
) -> dict:
    """Publicly observable movement (`09_...` section 5)."""
    return {
        "event_type": EVENT_MOVE,
        "ply": ply,
        "player": PLAYER_NAMES[player],
        "piece_id": piece_id_name(piece_id),
        "source": source,
        "destination": destination,
        "distance": distance,
        "is_attack": is_attack,
        "target_piece_id": None if target_piece_id is None else piece_id_name(target_piece_id),
    }


def make_identity_reveal_event(
    ply: int, piece_id: int, owner: int, piece_type: int, reason: str, newly_known_to: list[int]
) -> dict:
    """A new legal disclosure of an exact piece type (`09_...` section 6)."""
    return {
        "event_type": EVENT_IDENTITY_REVEAL,
        "ply": ply,
        "piece_id": piece_id_name(piece_id),
        "owner": PLAYER_NAMES[owner],
        "piece_type": PIECE_TYPE_NAMES[piece_type],
        "reason": reason,
        "newly_known_to": [PLAYER_NAMES[player] for player in sorted(newly_known_to)],
    }


def make_combat_event(
    ply: int,
    attacker_piece_id: int,
    defender_piece_id: int,
    attacker_type: int,
    defender_type: int,
    outcome: str,
    flag_captured: bool,
) -> dict:
    """Combat resolution (`09_...` section 7).

    Both types are present because combat makes both identities public.
    """
    return {
        "event_type": EVENT_COMBAT,
        "ply": ply,
        "attacker_piece_id": piece_id_name(attacker_piece_id),
        "defender_piece_id": piece_id_name(defender_piece_id),
        "attacker_type": PIECE_TYPE_NAMES[attacker_type],
        "defender_type": PIECE_TYPE_NAMES[defender_type],
        "outcome": outcome,
        "flag_captured": flag_captured,
    }


def make_behavior_event(event: BehaviorEvent) -> dict:
    """Derived behavioural fact (`09_...` section 8). Carries no type fields."""
    return {
        "event_type": EVENT_BEHAVIOR,
        "behavior_type": event.event_type,
        "ply": event.event_ply,
        "actor_piece_id": piece_id_name(event.actor_piece_id),
        "counterpart_piece_id": piece_id_name(event.counterpart_piece_id),
        "actor_knew_counterpart_type": event.actor_knew_counterpart_type,
        "context_piece_id": (
            None if event.context_piece_id is None else piece_id_name(event.context_piece_id)
        ),
    }


def make_game_end_event(state: GameState) -> dict:
    """Terminal event (`09_...` section 10)."""
    if state.is_draw or state.winner is None:
        result = "draw"
        winner = None
    else:
        winner = PLAYER_NAMES[state.winner]
        result = f"{winner}_win"
    return {
        "event_type": EVENT_GAME_END,
        "ply": state.total_moves,
        "winner": winner,
        "result": result,
        "terminal_reason": state.terminal_reason,
        "total_moves": state.total_moves,
        "moves_since_last_combat": state.battleless_moves,
    }


# ---------------------------------------------------------------------------
# Observer filtering
# ---------------------------------------------------------------------------

# Field whitelists per event type. Building filtered events from an explicit
# whitelist means a future field addition cannot silently leak into the browser
# stream: it has to be added here deliberately.
_PUBLIC_FIELDS = {
    EVENT_MOVE: (
        "event_type",
        "ply",
        "player",
        "piece_id",
        "source",
        "destination",
        "distance",
        "is_attack",
        "target_piece_id",
    ),
    EVENT_IDENTITY_REVEAL: (
        "event_type",
        "ply",
        "piece_id",
        "owner",
        "piece_type",
        "reason",
        "newly_known_to",
    ),
    EVENT_COMBAT: (
        "event_type",
        "ply",
        "attacker_piece_id",
        "defender_piece_id",
        "attacker_type",
        "defender_type",
        "outcome",
        "flag_captured",
    ),
    EVENT_BEHAVIOR: (
        "event_type",
        "behavior_type",
        "ply",
        "actor_piece_id",
        "counterpart_piece_id",
        "actor_knew_counterpart_type",
        "context_piece_id",
    ),
    EVENT_GAME_END: (
        "event_type",
        "ply",
        "winner",
        "result",
        "terminal_reason",
        "total_moves",
        "moves_since_last_combat",
    ),
}


def filter_event_for_observer(event: dict, observer: int) -> dict:
    """Project one derived event onto the fields an observer may receive.

    Every event category listed in `09_...` section 17 is public to both
    players: a move is visible, combat reveals both identities to both players,
    reveal events only ever disclose a type that has just become legal to
    disclose, and behavioural events contain no type information at all. The
    filter therefore whitelists fields rather than dropping whole events.
    """
    del observer  # no event category is currently observer-specific
    fields = _PUBLIC_FIELDS[event["event_type"]]
    return {name: event[name] for name in fields}


def filter_events_for_observer(events: list[dict], observer: int) -> list[dict]:
    """Observer-filtered public event stream (`09_...` section 15)."""
    return [filter_event_for_observer(event, observer) for event in events]


# ---------------------------------------------------------------------------
# Observer-filtered board view
# ---------------------------------------------------------------------------


def public_board_view(state: GameState, observer: int) -> dict:
    """Serialisable browser-safe board view (`09_...` section 12).

    Own pieces show their exact type. Opponent pieces show their type only once
    the observer legally knows it; otherwise the entry is marked hidden. The
    stable piece identifier is always present because a human can follow a
    concealed physical piece as it moves.
    """
    squares: list[dict | None] = []
    for square in range(NUM_SQUARES):
        if square in LAKE_SQUARE_SET:
            squares.append({"square": square, "name": square_name(square), "lake": True})
            continue
        piece_id = state.board[square]
        if piece_id is None:
            squares.append(
                {"square": square, "name": square_name(square), "lake": False, "piece": None}
            )
            continue
        record = state.pieces[piece_id]
        known = record.known_to(observer)
        squares.append(
            {
                "square": square,
                "name": square_name(square),
                "lake": False,
                "piece": {
                    "piece_id": piece_id_name(record.piece_id),
                    "owner": PLAYER_NAMES[record.owner],
                    "piece_type": PIECE_TYPE_NAMES[record.true_type] if known else None,
                    "hidden": not known,
                    "has_moved": record.has_moved,
                },
            }
        )

    captured = []
    for record in state.pieces:
        if record.alive:
            continue
        known = record.known_to(observer)
        captured.append(
            {
                "piece_id": piece_id_name(record.piece_id),
                "owner": PLAYER_NAMES[record.owner],
                # Every capture results from combat, which reveals both
                # identities, so in a reachable state this is never hidden.
                "piece_type": PIECE_TYPE_NAMES[record.true_type] if known else None,
                "capture_ply": record.capture_ply,
            }
        )

    return {
        "observer": PLAYER_NAMES[observer],
        "acting_player": PLAYER_NAMES[state.acting_player],
        "phase": state.phase,
        "total_moves": state.total_moves,
        "moves_since_last_combat": state.battleless_moves,
        "battleless_move_limit": state.rules.battleless_move_limit,
        "terminal": state.terminal,
        "terminal_reason": state.terminal_reason,
        "winner": None if state.winner is None else PLAYER_NAMES[state.winner],
        "is_draw": state.is_draw
        or (state.terminal and state.terminal_reason in DRAW_TERMINAL_REASONS),
        "squares": squares,
        "captured": captured,
    }


def public_setup_view(state: GameState, observer: int) -> dict:
    """Observer-specific game-start setup view (`09_...` section 11).

    The observer receives exact identities for all 40 of their own setup
    squares and occupancy only for the opponent's.
    """
    own: dict[str, str] = {}
    opponent: dict[str, str] = {}
    for record in state.pieces:
        key = square_name(record.starting_square)
        if record.owner == observer:
            own[key] = PIECE_TYPE_NAMES[record.true_type]
        else:
            opponent[key] = piece_id_name(record.piece_id)
    return {
        "observer": PLAYER_NAMES[observer],
        "own_setup": own,
        "opponent_setup_occupancy": opponent,
    }
