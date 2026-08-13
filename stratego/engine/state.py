"""The authoritative privileged game state.

Specification source: `08_internal_state_spec.md` (all sections).

The state stores compact *facts*. Nothing here is a model-ready observation
channel; `observation.py` derives all 127 planes from these facts on demand.

Ply numbering convention
------------------------
`total_moves` counts completed plies and doubles as the current ply index. A
move that completes is stamped with the ply number it produced, so the first
move of the game is ply 1. Observation recency then evaluates to
`delta = total_moves - event_ply`, which is `0` for an event recorded by the
move that just completed. That reproduces the `recency = 1.0 immediately after
the event` expectation stated in `07_observation_validation_matrix.md` section
8.1 exactly.
"""

from collections import deque
from dataclasses import dataclass, field

from .constants import (
    BLUE,
    LAKE_SQUARE_SET,
    NOT_TERMINAL,
    NUM_SQUARES,
    PHASE_PLAY,
    PIECES_PER_PLAYER,
    RECENT_MOVE_WINDOW,
    RED,
    RulesConfig,
    TRAINING_RULES,
)
from .pieces import PieceRecord, create_piece_records
from .setup import setup_squares, validate_setup


@dataclass
class RecentMove:
    """One entry of the 16-ply observation history window.

    Fields are exactly those required by `08_internal_state_spec.md` section 8.
    """

    ply: int
    player: int
    piece_id: int
    source: int
    destination: int
    destination_had_opponent: bool
    target_piece_id: int | None

    def as_tuple(self) -> tuple:
        return (
            self.ply,
            self.player,
            self.piece_id,
            self.source,
            self.destination,
            self.destination_had_opponent,
            self.target_piece_id,
        )


@dataclass
class BehaviorEvent:
    """Stored metadata for one behavioural event.

    `08_internal_state_spec.md` section 10 requires metadata rather than
    precomputed channel values, so no rank or special encoding is stored here.
    """

    event_type: str
    actor_piece_id: int
    counterpart_piece_id: int
    event_ply: int
    actor_knew_counterpart_type: bool
    context_piece_id: int | None = None

    def as_tuple(self) -> tuple:
        return (
            self.event_type,
            self.actor_piece_id,
            self.counterpart_piece_id,
            self.event_ply,
            self.actor_knew_counterpart_type,
            self.context_piece_id,
        )


@dataclass
class GameState:
    """Privileged full game state.

    The board is a 100-entry list holding either `None` or a `piece_id`. Lake
    squares are permanently `None`; `invariants.py` enforces that.
    """

    rules: RulesConfig
    game_id: str
    board: list[int | None]
    pieces: list[PieceRecord]
    acting_player: int
    phase: str = PHASE_PLAY
    total_moves: int = 0
    battleless_moves: int = 0
    terminal: bool = False
    terminal_reason: str = NOT_TERMINAL
    winner: int | None = None
    is_draw: bool = False
    recent_moves: deque = field(default_factory=lambda: deque(maxlen=RECENT_MOVE_WINDOW))
    # (threatener_piece_id, threatened_piece_id, creation_ply) produced by the
    # most recent completed move; consumed by evade/protect detection.
    active_threat_relations: list[tuple[int, int, int]] = field(default_factory=list)
    # (piece_id, behavior_type) -> latest BehaviorEvent of that type.
    behavior_memory: dict[tuple[int, str], BehaviorEvent] = field(default_factory=dict)
    # Derived engine events in emission order (`09_public_event_and_replay_schema.md`).
    events: list[dict] = field(default_factory=list)
    # Ordered action identifiers actually applied, sufficient for replay.
    action_history: list[int] = field(default_factory=list)

    # -- accessors ---------------------------------------------------------

    @property
    def ply(self) -> int:
        """Current ply index; identical to `total_moves` by construction."""
        return self.total_moves

    def piece(self, piece_id: int) -> PieceRecord:
        return self.pieces[piece_id]

    def piece_at(self, square: int) -> PieceRecord | None:
        piece_id = self.board[square]
        return None if piece_id is None else self.pieces[piece_id]

    def pieces_of(self, player: int) -> list[PieceRecord]:
        start = player * PIECES_PER_PLAYER
        return self.pieces[start : start + PIECES_PER_PLAYER]

    def live_pieces_of(self, player: int) -> list[PieceRecord]:
        return [record for record in self.pieces_of(player) if record.alive]

    def behavior_event(self, piece_id: int, behavior_type: str) -> BehaviorEvent | None:
        return self.behavior_memory.get((piece_id, behavior_type))

    def result_for(self, player: int) -> float:
        """Reinforcement-learning result from `player`'s perspective.

        `02_project_ruleset.md` section 9: win `+1`, loss `-1`, draw `0`.
        """
        if not self.terminal:
            raise ValueError("result requested for a non-terminal state")
        if self.is_draw or self.winner is None:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def effective_score_for(self, player: int) -> float:
        """Headline evaluation score: win `1.0`, draw `0.5`, loss `0.0`."""
        return (self.result_for(player) + 1.0) / 2.0


def create_game(
    red_setup: "list[int] | tuple[int, ...]",
    blue_setup: "list[int] | tuple[int, ...]",
    rules: RulesConfig = TRAINING_RULES,
    game_id: str = "game",
) -> GameState:
    """Build a fresh playable state from two validated setups.

    Both setups are validated before anything is constructed, so an illegal
    setup can never produce a partially built state.
    """
    validated_red = validate_setup(red_setup, RED)
    validated_blue = validate_setup(blue_setup, BLUE)

    records = create_piece_records(RED, validated_red, setup_squares(RED))
    records += create_piece_records(BLUE, validated_blue, setup_squares(BLUE))

    board: list[int | None] = [None] * NUM_SQUARES
    for record in records:
        square = record.starting_square
        if square in LAKE_SQUARE_SET:  # pragma: no cover - setup squares exclude lakes
            raise ValueError(f"setup square {square} is a lake")
        if board[square] is not None:  # pragma: no cover - slots are unique
            raise ValueError(f"duplicate occupancy at square {square}")
        board[square] = record.piece_id

    state = GameState(
        rules=rules,
        game_id=game_id,
        board=board,
        pieces=records,
        acting_player=rules.first_player,
    )

    # `phase2_1_reference_1.2.0`: the mobility-termination rule applies to the
    # initial position too -- a legal random setup can strand the first player
    # at ply 0, and such a game is already decided (`01_official_rules.md`
    # section 8). The evaluation lives in `transition.py` next to the per-move
    # terminal logic so there is exactly one interpretation of the rule; the
    # import is local because `transition` imports this module.
    from .transition import evaluate_initial_terminal

    evaluate_initial_terminal(state)
    return state


# ---------------------------------------------------------------------------
# Canonical comparison
# ---------------------------------------------------------------------------


def state_fingerprint(state: GameState, include_history: bool = True) -> tuple:
    """A canonical, fully detailed value describing the entire state.

    Two states with equal fingerprints are indistinguishable to every other part
    of the engine. Used by the illegal-action immutability test, snapshot
    round-trip tests and replay comparison.

    `include_history=False` omits the derived event log and the action history,
    matching the reduced contents that `08_internal_state_spec.md` section 15
    permits a search snapshot to carry.
    """
    pieces = tuple(
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
    )
    behavior = tuple(
        sorted(
            (key[0], key[1]) + event.as_tuple()
            for key, event in state.behavior_memory.items()
        )
    )
    core = (
        state.rules,
        state.game_id,
        tuple(state.board),
        pieces,
        state.acting_player,
        state.phase,
        state.total_moves,
        state.battleless_moves,
        state.terminal,
        state.terminal_reason,
        state.winner,
        state.is_draw,
        tuple(move.as_tuple() for move in state.recent_moves),
        tuple(sorted(state.active_threat_relations)),
        behavior,
    )
    if not include_history:
        return core
    events = tuple(_canonical_event(event) for event in state.events)
    return core + (events, tuple(state.action_history))


def _canonical_event(event: dict) -> tuple:
    """Order-independent canonical form of one derived event dictionary."""
    return tuple(sorted((key, _canonical_value(value)) for key, value in event.items()))


def _canonical_value(value):
    if isinstance(value, dict):
        return _canonical_event(value)
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_value(item) for item in value)
    return value


def render_board(state: GameState, observer: int | None = None) -> str:
    """Render the board as text for manual inspection and report examples.

    With `observer=None` every true type is shown (privileged view). With an
    observer, that player's own pieces and legally known opponent pieces show
    their type code while unresolved opponent pieces show `?`.
    """
    from .constants import PIECE_TYPE_CODES
    from .coordinates import COLUMN_LETTERS

    lines = ["    " + " ".join(f"{letter:>2}" for letter in COLUMN_LETTERS)]
    for row in range(9, -1, -1):
        cells = []
        for column in range(10):
            square = row * 10 + column
            if square in LAKE_SQUARE_SET:
                cells.append("~~")
                continue
            piece_id = state.board[square]
            if piece_id is None:
                cells.append(" .")
                continue
            record = state.pieces[piece_id]
            side = "r" if record.owner == RED else "b"
            visible = observer is None or record.owner == observer or record.known_to(observer)
            code = PIECE_TYPE_CODES[record.true_type] if visible else "?"
            cells.append(f"{side}{code}")
        lines.append(f"{row + 1:>3} " + " ".join(cells))
    return "\n".join(lines)
