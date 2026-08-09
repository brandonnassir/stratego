"""Compact state snapshot and restore, for later decision-time search.

Specification sources:

- `03_game_engine_spec.md` section 13
- `08_internal_state_spec.md` section 15

A snapshot contains everything that can change future legality, observations,
values or events. `08_internal_state_spec.md` section 15 explicitly excludes the
long-form replay record, so the derived event log and the action history are
optional; `include_history=True` produces a snapshot that restores to a fully
byte-identical state, which the round-trip tests use.
"""

import json
from collections import deque

from .constants import RECENT_MOVE_WINDOW, RulesConfig
from .pieces import PieceRecord
from .state import BehaviorEvent, GameState, RecentMove

SNAPSHOT_VERSION = "snapshot_v1"


def create_snapshot(state: GameState, include_history: bool = False) -> dict:
    """Create an independent compact snapshot of `state`.

    Every value stored is an immutable scalar or a tuple of scalars, so the
    snapshot shares no mutable structure with the live state. The rules
    configuration is a frozen dataclass and is shared by reference on purpose:
    it must not change during a game.
    """
    snapshot = {
        "snapshot_version": SNAPSHOT_VERSION,
        "rules": state.rules,
        "game_id": state.game_id,
        "board": tuple(state.board),
        "pieces": tuple(
            (
                record.piece_id,
                record.owner,
                record.true_type,
                record.starting_square,
                record.current_square,
                record.alive,
                record.has_moved,
                record.known_to_red,
                record.known_to_blue,
                record.reveal_reason_red,
                record.reveal_reason_blue,
                record.capture_ply,
            )
            for record in state.pieces
        ),
        "acting_player": state.acting_player,
        "phase": state.phase,
        "total_moves": state.total_moves,
        "battleless_moves": state.battleless_moves,
        "terminal": state.terminal,
        "terminal_reason": state.terminal_reason,
        "winner": state.winner,
        "is_draw": state.is_draw,
        "recent_moves": tuple(move.as_tuple() for move in state.recent_moves),
        "active_threat_relations": tuple(state.active_threat_relations),
        "behavior_memory": tuple(
            (key[0], key[1]) + event.as_tuple() for key, event in state.behavior_memory.items()
        ),
    }
    if include_history:
        snapshot["events"] = tuple(json.dumps(event, sort_keys=True) for event in state.events)
        snapshot["action_history"] = tuple(state.action_history)
    return snapshot


def restore_snapshot(snapshot: dict) -> GameState:
    """Rebuild a live `GameState` from a snapshot."""
    if snapshot.get("snapshot_version") != SNAPSHOT_VERSION:
        raise ValueError(f"unsupported snapshot version: {snapshot.get('snapshot_version')!r}")

    rules = snapshot["rules"]
    if not isinstance(rules, RulesConfig):  # pragma: no cover - defensive
        raise ValueError("snapshot is missing its rules configuration")

    pieces = [
        PieceRecord(
            piece_id=fields[0],
            owner=fields[1],
            true_type=fields[2],
            starting_square=fields[3],
            current_square=fields[4],
            alive=fields[5],
            has_moved=fields[6],
            known_to_red=fields[7],
            known_to_blue=fields[8],
            reveal_reason_red=fields[9],
            reveal_reason_blue=fields[10],
            capture_ply=fields[11],
        )
        for fields in snapshot["pieces"]
    ]

    recent_moves = deque(
        (RecentMove(*fields) for fields in snapshot["recent_moves"]),
        maxlen=RECENT_MOVE_WINDOW,
    )

    behavior_memory: dict[tuple[int, str], BehaviorEvent] = {}
    for entry in snapshot["behavior_memory"]:
        piece_id, behavior_type = entry[0], entry[1]
        behavior_memory[(piece_id, behavior_type)] = BehaviorEvent(
            event_type=entry[2],
            actor_piece_id=entry[3],
            counterpart_piece_id=entry[4],
            event_ply=entry[5],
            actor_knew_counterpart_type=entry[6],
            context_piece_id=entry[7],
        )

    state = GameState(
        rules=rules,
        game_id=snapshot["game_id"],
        board=list(snapshot["board"]),
        pieces=pieces,
        acting_player=snapshot["acting_player"],
        phase=snapshot["phase"],
        total_moves=snapshot["total_moves"],
        battleless_moves=snapshot["battleless_moves"],
        terminal=snapshot["terminal"],
        terminal_reason=snapshot["terminal_reason"],
        winner=snapshot["winner"],
        is_draw=snapshot["is_draw"],
        recent_moves=recent_moves,
        active_threat_relations=list(snapshot["active_threat_relations"]),
        behavior_memory=behavior_memory,
    )
    if "events" in snapshot:
        state.events = [json.loads(text) for text in snapshot["events"]]
    if "action_history" in snapshot:
        state.action_history = list(snapshot["action_history"])
    return state


def clone_state(state: GameState) -> GameState:
    """Full independent copy of a state, including its derived event log."""
    return restore_snapshot(create_snapshot(state, include_history=True))


def snapshot_to_json(snapshot: dict) -> str:
    """Serialise a snapshot to JSON, used for the storage baseline measurement."""
    payload = dict(snapshot)
    rules = payload.pop("rules")
    payload["rules"] = {
        "rules_version": rules.rules_version,
        "board_geometry_version": rules.board_geometry_version,
        "first_player": rules.first_player,
        "battleless_move_limit": rules.battleless_move_limit,
        "absolute_move_limit": rules.absolute_move_limit,
        "two_square_rule_enabled": rules.two_square_rule_enabled,
        "continuous_chasing_rule_enabled": rules.continuous_chasing_rule_enabled,
        "context": rules.context,
    }
    return json.dumps(payload, separators=(",", ":"), default=list)
