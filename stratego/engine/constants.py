"""Board geometry, piece enumeration, rules configuration and terminal labels.

Specification sources:

- `01_official_rules.md` sections 1, 2 (board, inventory)
- `02_project_ruleset.md` sections 1, 3, 4, 5, 10 (included rules, draw limits, versioning)
- `03_game_engine_spec.md` sections 4, 6, 11 (coordinates, type order, terminal reasons)
- `08_internal_state_spec.md` section 3 (rules configuration)

Everything in this module is immutable data. No game logic lives here.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Board geometry
# ---------------------------------------------------------------------------

BOARD_ROWS = 10
BOARD_COLUMNS = 10
NUM_SQUARES = BOARD_ROWS * BOARD_COLUMNS  # 100 absolute square indices, 0..99

# The two central lake regions. Using zero-based (row, column) indices the lakes
# occupy rows 4 and 5, columns 2-3 and 6-7. In the human notation defined by
# `01_official_rules.md` those are squares c5-d5, g5-h5, c6-d6 and g6-h6.
LAKE_ROWS = (4, 5)
LAKE_COLUMNS = (2, 3, 6, 7)

LAKE_SQUARES = tuple(
    sorted(row * BOARD_COLUMNS + column for row in LAKE_ROWS for column in LAKE_COLUMNS)
)
assert len(LAKE_SQUARES) == 8

LAKE_SQUARE_SET = frozenset(LAKE_SQUARES)

OCCUPIABLE_SQUARES = tuple(
    square for square in range(NUM_SQUARES) if square not in LAKE_SQUARE_SET
)
assert len(OCCUPIABLE_SQUARES) == 92

NUM_OCCUPIABLE_SQUARES = len(OCCUPIABLE_SQUARES)
NUM_LAKE_SQUARES = len(LAKE_SQUARES)

# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

RED = 0
BLUE = 1
PLAYERS = (RED, BLUE)
PLAYER_NAMES = {RED: "red", BLUE: "blue"}
PLAYER_BY_NAME = {"red": RED, "blue": BLUE}


def opponent_of(player: int) -> int:
    """Return the other player."""
    return BLUE if player == RED else RED


# Red owns the four rows nearest red (rows 0-3, human rows 1-4).
# Blue owns rows 6-9 (human rows 7-10).
SETUP_ROWS = {RED: (0, 1, 2, 3), BLUE: (6, 7, 8, 9)}

SETUP_SQUARES = {
    player: tuple(
        row * BOARD_COLUMNS + column
        for row in SETUP_ROWS[player]
        for column in range(BOARD_COLUMNS)
    )
    for player in PLAYERS
}
assert all(len(squares) == 40 for squares in SETUP_SQUARES.values())
# No setup square may be a lake square; the lakes sit in rows 4-5, between the
# two setup areas, so this holds by construction. The assertion documents it.
assert all(
    square not in LAKE_SQUARE_SET
    for squares in SETUP_SQUARES.values()
    for square in squares
)

SETUP_SQUARE_SETS = {player: frozenset(SETUP_SQUARES[player]) for player in PLAYERS}

# ---------------------------------------------------------------------------
# Piece types
# ---------------------------------------------------------------------------

# Stable enumeration from `03_game_engine_spec.md` section 6. The same order is
# used by the identity planes of `observation_v2_127ch` (see `06_...` section 5).
SPY = 0
SCOUT = 1
MINER = 2
SERGEANT = 3
LIEUTENANT = 4
CAPTAIN = 5
MAJOR = 6
COLONEL = 7
GENERAL = 8
MARSHAL = 9
FLAG = 10
BOMB = 11

NUM_PIECE_TYPES = 12
PIECE_TYPES = tuple(range(NUM_PIECE_TYPES))

PIECE_TYPE_NAMES = (
    "spy",
    "scout",
    "miner",
    "sergeant",
    "lieutenant",
    "captain",
    "major",
    "colonel",
    "general",
    "marshal",
    "flag",
    "bomb",
)
PIECE_TYPE_BY_NAME = {name: index for index, name in enumerate(PIECE_TYPE_NAMES)}

# Single-character codes used for compact setup serialisation and for the
# human-readable board renderings in the report.
PIECE_TYPE_CODES = ("S", "9", "8", "7", "6", "5", "4", "3", "2", "1", "F", "B")
PIECE_TYPE_BY_CODE = {code: index for index, code in enumerate(PIECE_TYPE_CODES)}

# Combat ranks. Flag and Bomb have no numeric rank; `None` forces every rank
# comparison involving them through an explicit special case in `combat.py`.
PIECE_RANKS = {
    SPY: 1,
    SCOUT: 2,
    MINER: 3,
    SERGEANT: 4,
    LIEUTENANT: 5,
    CAPTAIN: 6,
    MAJOR: 7,
    COLONEL: 8,
    GENERAL: 9,
    MARSHAL: 10,
    FLAG: None,
    BOMB: None,
}

MAX_RANK = 10

# Immovable pieces (`01_official_rules.md` section 5).
IMMOVABLE_TYPES = frozenset({FLAG, BOMB})
MOVABLE_TYPES = frozenset(PIECE_TYPES) - IMMOVABLE_TYPES

# Official per-player inventory (`01_official_rules.md` section 2).
PIECE_COUNTS = {
    SPY: 1,
    SCOUT: 8,
    MINER: 5,
    SERGEANT: 4,
    LIEUTENANT: 4,
    CAPTAIN: 4,
    MAJOR: 3,
    COLONEL: 2,
    GENERAL: 1,
    MARSHAL: 1,
    FLAG: 1,
    BOMB: 6,
}
PIECES_PER_PLAYER = sum(PIECE_COUNTS.values())
assert PIECES_PER_PLAYER == 40

TOTAL_PHYSICAL_PIECES = PIECES_PER_PLAYER * len(PLAYERS)  # 80

# ---------------------------------------------------------------------------
# Action space (`03_game_engine_spec.md` section 8)
# ---------------------------------------------------------------------------

ACTION_SPACE_SIZE = NUM_SQUARES * NUM_SQUARES  # 10,000

# ---------------------------------------------------------------------------
# Terminal reasons (`03_game_engine_spec.md` section 11)
# ---------------------------------------------------------------------------

TERMINAL_FLAG_CAPTURE = "flag_capture"
TERMINAL_OPPONENT_NO_LEGAL_MOVE = "opponent_no_legal_move"
TERMINAL_BOTH_NO_LEGAL_MOVE_DRAW = "both_no_legal_move_draw"
TERMINAL_BATTLELESS_MOVE_LIMIT_DRAW = "battleless_move_limit_draw"
TERMINAL_ABSOLUTE_MOVE_LIMIT_DRAW = "absolute_move_limit_draw"
NOT_TERMINAL = "not_terminal"

TERMINAL_REASONS = (
    TERMINAL_FLAG_CAPTURE,
    TERMINAL_OPPONENT_NO_LEGAL_MOVE,
    TERMINAL_BOTH_NO_LEGAL_MOVE_DRAW,
    TERMINAL_BATTLELESS_MOVE_LIMIT_DRAW,
    TERMINAL_ABSOLUTE_MOVE_LIMIT_DRAW,
    NOT_TERMINAL,
)

DRAW_TERMINAL_REASONS = frozenset(
    {
        TERMINAL_BOTH_NO_LEGAL_MOVE_DRAW,
        TERMINAL_BATTLELESS_MOVE_LIMIT_DRAW,
        TERMINAL_ABSOLUTE_MOVE_LIMIT_DRAW,
    }
)

# Game phases (`08_internal_state_spec.md` section 7).
PHASE_SETUP = "setup"
PHASE_PLAY = "play"
PHASE_TERMINAL = "terminal"

# ---------------------------------------------------------------------------
# Behavioural event types (`06_observation_v2_127ch.md` section 9)
# ---------------------------------------------------------------------------

BEHAVIOR_THREAT = "threat"
BEHAVIOR_EVADE = "evade"
BEHAVIOR_DECLINED_ATTACK = "declined_attack"
BEHAVIOR_PROTECT = "protect"
BEHAVIOR_WAS_PROTECTED = "was_protected"

# The order below is also the channel order inside the behavioural plane blocks.
BEHAVIOR_TYPES = (
    BEHAVIOR_THREAT,
    BEHAVIOR_EVADE,
    BEHAVIOR_DECLINED_ATTACK,
    BEHAVIOR_PROTECT,
    BEHAVIOR_WAS_PROTECTED,
)
NUM_BEHAVIOR_TYPES = len(BEHAVIOR_TYPES)
BEHAVIOR_FEATURES_PER_TYPE = 4

# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

RULES_VERSION = "stratego_project_v1"
OBSERVATION_VERSION = "observation_v2_1_127ch"
# `observation_v2_127ch` is superseded. The channel count, order and meaning are
# unchanged; only the behavioural counterpart tie-break differs, which is enough
# to require a new identifier under `06_observation_v2_1_127ch` section 1.
SUPERSEDED_OBSERVATION_VERSIONS = ("observation_v1_68ch", "observation_v2_127ch")
REPLAY_VERSION = "replay_v1"
EVENT_SCHEMA_VERSION = "event_schema_v1"
IMPLEMENTATION_VERSION = "phase2_1_reference_1.1.0"

# ---------------------------------------------------------------------------
# Observation shape (`06_observation_v2_127ch.md` section 3)
# ---------------------------------------------------------------------------

OBSERVATION_CHANNELS = 127
OBSERVATION_SHAPE = (OBSERVATION_CHANNELS, BOARD_ROWS, BOARD_COLUMNS)
RECENT_MOVE_WINDOW = 16

# Denominator of the recency decay in `06_observation_v2_127ch.md` section 9.2.
BEHAVIOR_RECENCY_SCALE = 32.0

# ---------------------------------------------------------------------------
# Rules configuration (`08_internal_state_spec.md` section 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RulesConfig:
    """Immutable per-game rules configuration.

    A rules configuration must not change during an active game, so this object
    is frozen and shared by reference.
    """

    rules_version: str = RULES_VERSION
    board_geometry_version: str = "board_10x10_v1"
    first_player: int = RED
    battleless_move_limit: int = 100
    absolute_move_limit: int = 4000
    two_square_rule_enabled: bool = False
    continuous_chasing_rule_enabled: bool = False
    context: str = "training"

    def __post_init__(self) -> None:
        # Both excluded rules are project decisions (`02_project_ruleset.md`
        # section 2). The engine has no implementation for them at all, so a
        # configuration requesting them must fail loudly rather than silently
        # play without them.
        if self.two_square_rule_enabled or self.continuous_chasing_rule_enabled:
            raise ValueError(
                "The two-square and continuous-chasing rules are deliberately "
                "excluded from stratego_project_v1 and are not implemented."
            )
        if self.battleless_move_limit <= 0:
            raise ValueError("battleless_move_limit must be positive")
        if self.absolute_move_limit <= 0:
            raise ValueError("absolute_move_limit must be positive")
        if self.first_player not in PLAYERS:
            raise ValueError("first_player must be RED or BLUE")


# Project defaults from `08_internal_state_spec.md` section 3.
TRAINING_RULES = RulesConfig(
    battleless_move_limit=100, absolute_move_limit=4000, context="training"
)
EVALUATION_RULES = RulesConfig(
    battleless_move_limit=200, absolute_move_limit=4000, context="evaluation"
)
