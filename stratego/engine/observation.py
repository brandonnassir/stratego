"""Construction of `observation_v2_127ch` and its machine-readable metadata.

Specification sources:

- `06_observation_v2_127ch.md` (authoritative channel contract)
- `07_observation_validation_matrix.md` (per-channel acceptance behaviour)
- `08_internal_state_spec.md` sections 12, 13, 16 (derivation, not storage)

Hidden-information discipline
-----------------------------
`build_observation` reads `record.true_type` only through one of three guarded
paths: the piece belongs to the observer, the observer legally knows the piece,
or a behavioural counterpart passes both the historical actor-knew test and the
observer's current knowledge test. There is no other `true_type` access in this
module. Privileged belief targets live in :func:`belief_target`, which is never
called by the observation builder.
"""

import numpy as np

from .constants import (
    BEHAVIOR_RECENCY_SCALE,
    BEHAVIOR_TYPES,
    BOARD_COLUMNS,
    BOARD_ROWS,
    BOMB,
    FLAG,
    LAKE_SQUARES,
    MAX_RANK,
    NUM_PIECE_TYPES,
    NUM_SQUARES,
    OBSERVATION_CHANNELS,
    OBSERVATION_SHAPE,
    OBSERVATION_VERSION,
    PIECE_COUNTS,
    PIECE_RANKS,
    PIECE_TYPE_NAMES,
    PLAYER_NAMES,
    RECENT_MOVE_WINDOW,
    opponent_of,
)
from .coordinates import PERSPECTIVE_TABLES, normalized_coordinate
from .pieces import piece_id_name
from .state import GameState

# ---------------------------------------------------------------------------
# Channel layout (`06_observation_v2_127ch.md` section 4)
# ---------------------------------------------------------------------------

CH_OWN_IDENTITY = 0  # 0-11
CH_KNOWN_OPPONENT_IDENTITY = 12  # 12-23
CH_HIDDEN_OPPONENT_OCCUPANCY = 24
CH_OWN_KNOWN_TO_OPPONENT = 25
CH_OWN_MOVED = 26
CH_OPPONENT_MOVED = 27
CH_OWN_START_ROW = 28
CH_OWN_START_COLUMN = 29
CH_OPPONENT_START_ROW = 30
CH_OPPONENT_START_COLUMN = 31
CH_OWN_SETUP = 32  # 32-43
CH_KNOWN_OPPONENT_SETUP = 44  # 44-55
CH_UNRESOLVED_INVENTORY = 56  # 56-67
CH_OWN_BEHAVIOR = 68  # 68-87
CH_OPPONENT_BEHAVIOR = 88  # 88-107
CH_RECENT_MOVES = 108  # 108-123
CH_LAKE_MASK = 124
CH_GAME_PROGRESS = 125
CH_BATTLELESS_PROGRESS = 126

BEHAVIOR_INDEX = {name: index for index, name in enumerate(BEHAVIOR_TYPES)}

# Offsets within one behaviour's four-channel block.
BEHAVIOR_FEATURE_STRIDE = 4
BEHAVIOR_RECENCY_OFFSET = 0
BEHAVIOR_RANK_OFFSET = 1
BEHAVIOR_ACTOR_KNEW_OFFSET = 2
BEHAVIOR_SPECIAL_OFFSET = 3

OBSERVATION_DTYPE = np.float32

# The lake plane is static, and the 180 degree perspective rotation maps the
# lake mask onto itself, so one precomputed row serves both observers.
_LAKE_PLANE = np.zeros(NUM_SQUARES, dtype=OBSERVATION_DTYPE)
_LAKE_PLANE[list(LAKE_SQUARES)] = 1.0

# Normalized coordinate per row/column index, from section 6 of the observation
# specification: coord(i) = 2 * i / 9 - 1.
_COORDINATE = [normalized_coordinate(index) for index in range(BOARD_ROWS)]


def build_observation(state: GameState, observer: int | None = None) -> np.ndarray:
    """Build the `(127, 10, 10)` float32 observation for `observer`.

    `observer` defaults to the acting player. Building an observation for the
    non-acting player is required by the snapshot and anti-leak tests, which
    compare "the observation for either player".
    """
    if observer is None:
        observer = state.acting_player
    opponent = opponent_of(observer)

    tensor = np.zeros(OBSERVATION_SHAPE, dtype=OBSERVATION_DTYPE)
    # A flat (127, 100) view lets every write use an absolute square index.
    planes = tensor.reshape(OBSERVATION_CHANNELS, NUM_SQUARES)
    perspective = PERSPECTIVE_TABLES[observer]

    known_opponent_type_counts = [0] * NUM_PIECE_TYPES

    for record in state.pieces:
        is_own = record.owner == observer
        observer_knows = record.known_to(observer)
        normalized_start = perspective[record.starting_square]

        # -- persistent setup memory (channels 32-55) ----------------------
        # These planes cover captured pieces too, so they are written outside
        # the alive check.
        if is_own:
            planes[CH_OWN_SETUP + record.true_type, normalized_start] = 1.0
        elif observer_knows:
            planes[CH_KNOWN_OPPONENT_SETUP + record.true_type, normalized_start] = 1.0
            known_opponent_type_counts[record.true_type] += 1

        if not record.alive:
            continue

        normalized_square = perspective[record.current_square]
        start_row, start_column = divmod(normalized_start, BOARD_COLUMNS)

        if is_own:
            planes[CH_OWN_IDENTITY + record.true_type, normalized_square] = 1.0
            if record.known_to(opponent):
                planes[CH_OWN_KNOWN_TO_OPPONENT, normalized_square] = 1.0
            if record.has_moved:
                planes[CH_OWN_MOVED, normalized_square] = 1.0
            planes[CH_OWN_START_ROW, normalized_square] = _COORDINATE[start_row]
            planes[CH_OWN_START_COLUMN, normalized_square] = _COORDINATE[start_column]
        else:
            if observer_knows:
                planes[CH_KNOWN_OPPONENT_IDENTITY + record.true_type, normalized_square] = 1.0
            else:
                planes[CH_HIDDEN_OPPONENT_OCCUPANCY, normalized_square] = 1.0
            if record.has_moved:
                planes[CH_OPPONENT_MOVED, normalized_square] = 1.0
            planes[CH_OPPONENT_START_ROW, normalized_square] = _COORDINATE[start_row]
            planes[CH_OPPONENT_START_COLUMN, normalized_square] = _COORDINATE[start_column]

    # -- unresolved opponent inventory (channels 56-67) --------------------
    # U_T = N_T - K_T, normalized by N_T, where K_T counts opponent pieces of
    # type T whose identity the observer legally knows, alive or captured.
    for piece_type in range(NUM_PIECE_TYPES):
        initial_count = PIECE_COUNTS[piece_type]
        unresolved = initial_count - known_opponent_type_counts[piece_type]
        planes[CH_UNRESOLVED_INVENTORY + piece_type, :] = unresolved / initial_count

    # -- behavioural history (channels 68-107) -----------------------------
    total_moves = state.total_moves
    for (actor_id, behavior_type), event in state.behavior_memory.items():
        actor = state.pieces[actor_id]
        if not actor.alive:
            # A captured piece occupies no square, so its behavioural features
            # simply disappear from the board tokens.
            continue
        block = CH_OWN_BEHAVIOR if actor.owner == observer else CH_OPPONENT_BEHAVIOR
        base = block + BEHAVIOR_FEATURE_STRIDE * BEHAVIOR_INDEX[behavior_type]
        normalized_square = perspective[actor.current_square]

        delta = total_moves - event.event_ply
        planes[base + BEHAVIOR_RECENCY_OFFSET, normalized_square] = 1.0 / (
            1.0 + delta / BEHAVIOR_RECENCY_SCALE
        )
        if event.actor_knew_counterpart_type:
            planes[base + BEHAVIOR_ACTOR_KNEW_OFFSET, normalized_square] = 1.0

        # Counterpart rank and special encoding are exposed only when the actor
        # knew the counterpart at event time *and* the current observer is
        # legally allowed to know it now (`06_...` section 9.3).
        counterpart = state.pieces[event.counterpart_piece_id]
        if event.actor_knew_counterpart_type and counterpart.known_to(observer):
            counterpart_type = counterpart.true_type
            rank = PIECE_RANKS[counterpart_type]
            if rank is not None:
                planes[base + BEHAVIOR_RANK_OFFSET, normalized_square] = rank / MAX_RANK
            elif counterpart_type == BOMB:
                planes[base + BEHAVIOR_SPECIAL_OFFSET, normalized_square] = 1.0
            elif counterpart_type == FLAG:
                planes[base + BEHAVIOR_SPECIAL_OFFSET, normalized_square] = -1.0

    # -- recent moves (channels 108-123) -----------------------------------
    # Channel 108 is the immediately preceding ply and older plies follow.
    for offset, move in enumerate(reversed(state.recent_moves)):
        if offset >= RECENT_MOVE_WINDOW:  # pragma: no cover - deque is bounded
            break
        channel = CH_RECENT_MOVES + offset
        planes[channel, perspective[move.source]] = -1.0
        planes[channel, perspective[move.destination]] = 1.0

    # -- global planes (channels 124-126) ----------------------------------
    planes[CH_LAKE_MASK, :] = _LAKE_PLANE
    planes[CH_GAME_PROGRESS, :] = min(
        state.total_moves / state.rules.absolute_move_limit, 1.0
    )
    planes[CH_BATTLELESS_PROGRESS, :] = min(
        state.battleless_moves / state.rules.battleless_move_limit, 1.0
    )

    return tensor


def observation_and_mask(state: GameState) -> tuple[np.ndarray, np.ndarray]:
    """Acting player's observation plus the separate legal-action mask."""
    from .legal_moves import legal_action_mask

    return build_observation(state, state.acting_player), legal_action_mask(state)


# ---------------------------------------------------------------------------
# Privileged belief targets (`06_...` section 15, `08_...` section 16)
# ---------------------------------------------------------------------------


def belief_target(state: GameState, observer: int | None = None) -> list[dict]:
    """Ground-truth labels for opponent pieces still hidden from `observer`.

    This function deliberately reads privileged hidden types. It is a *training
    target* and must never be fed to the policy/value/belief encoder input. No
    code path inside :func:`build_observation` calls it.
    """
    if observer is None:
        observer = state.acting_player
    targets = []
    for record in state.pieces:
        if record.owner == observer or not record.alive:
            continue
        if record.known_to(observer):
            continue
        targets.append(
            {
                "piece_id": piece_id_name(record.piece_id),
                "square": record.current_square,
                "true_type": PIECE_TYPE_NAMES[record.true_type],
            }
        )
    targets.sort(key=lambda item: item["piece_id"])
    return targets


# ---------------------------------------------------------------------------
# Machine-readable channel metadata (`07_...` section 14 phase gate)
# ---------------------------------------------------------------------------

_BINARY_RANGE = (0.0, 1.0)
_COORDINATE_RANGE = (-1.0, 1.0)
_SPECIAL_RANGE = (-1.0, 1.0)


def observation_channel_metadata() -> list[dict]:
    """One descriptor per channel: index, name, valid range and description."""
    metadata: list[dict] = []

    def add(index: int, name: str, valid_range: tuple[float, float], description: str) -> None:
        metadata.append(
            {
                "observation_version": OBSERVATION_VERSION,
                "channel": index,
                "name": name,
                "valid_range": [float(valid_range[0]), float(valid_range[1])],
                "description": description,
            }
        )

    for piece_type in range(NUM_PIECE_TYPES):
        add(
            CH_OWN_IDENTITY + piece_type,
            f"own_identity_{PIECE_TYPE_NAMES[piece_type]}",
            _BINARY_RANGE,
            f"1 at the current square of each living own {PIECE_TYPE_NAMES[piece_type]}",
        )
    for piece_type in range(NUM_PIECE_TYPES):
        add(
            CH_KNOWN_OPPONENT_IDENTITY + piece_type,
            f"known_opponent_identity_{PIECE_TYPE_NAMES[piece_type]}",
            _BINARY_RANGE,
            f"1 at the current square of each living opponent {PIECE_TYPE_NAMES[piece_type]} "
            "whose identity is legally known to the observer",
        )
    add(
        CH_HIDDEN_OPPONENT_OCCUPANCY,
        "hidden_opponent_occupancy",
        _BINARY_RANGE,
        "1 at every square holding an opponent piece of unknown exact identity",
    )
    add(
        CH_OWN_KNOWN_TO_OPPONENT,
        "own_identity_known_to_opponent",
        _BINARY_RANGE,
        "1 at the current square of each living own piece the opponent legally knows",
    )
    add(
        CH_OWN_MOVED,
        "own_has_moved",
        _BINARY_RANGE,
        "1 at the current square of each living own piece that has moved at least once",
    )
    add(
        CH_OPPONENT_MOVED,
        "opponent_has_moved",
        _BINARY_RANGE,
        "1 at the current square of each living opponent piece that has moved at least once",
    )
    add(
        CH_OWN_START_ROW,
        "own_start_row",
        _COORDINATE_RANGE,
        "normalized starting row of the living own piece standing on the square",
    )
    add(
        CH_OWN_START_COLUMN,
        "own_start_column",
        _COORDINATE_RANGE,
        "normalized starting column of the living own piece standing on the square",
    )
    add(
        CH_OPPONENT_START_ROW,
        "opponent_start_row",
        _COORDINATE_RANGE,
        "normalized starting row of the living opponent piece standing on the square",
    )
    add(
        CH_OPPONENT_START_COLUMN,
        "opponent_start_column",
        _COORDINATE_RANGE,
        "normalized starting column of the living opponent piece standing on the square",
    )
    for piece_type in range(NUM_PIECE_TYPES):
        add(
            CH_OWN_SETUP + piece_type,
            f"own_setup_{PIECE_TYPE_NAMES[piece_type]}",
            _BINARY_RANGE,
            f"1 at the original setup square of every own {PIECE_TYPE_NAMES[piece_type]}, "
            "alive or captured; static for the whole game",
        )
    for piece_type in range(NUM_PIECE_TYPES):
        add(
            CH_KNOWN_OPPONENT_SETUP + piece_type,
            f"known_opponent_setup_{PIECE_TYPE_NAMES[piece_type]}",
            _BINARY_RANGE,
            f"1 at the original setup square of every opponent {PIECE_TYPE_NAMES[piece_type]} "
            "whose identity has become legally known",
        )
    for piece_type in range(NUM_PIECE_TYPES):
        add(
            CH_UNRESOLVED_INVENTORY + piece_type,
            f"unresolved_opponent_{PIECE_TYPE_NAMES[piece_type]}",
            _BINARY_RANGE,
            "broadcast (N_T - K_T) / N_T for this opponent piece type, where K_T counts "
            "identities legally known to the observer",
        )

    feature_names = ("recency", "rank", "actor_knew", "special")
    feature_ranges = (_BINARY_RANGE, _BINARY_RANGE, _BINARY_RANGE, _SPECIAL_RANGE)
    feature_descriptions = (
        "1 / (1 + plies_since_event / 32) for the latest {behavior} event of the piece "
        "standing on the square",
        "counterpart rank / 10 when the counterpart identity may legally be shown",
        "whether the actor knew the counterpart identity when the {behavior} event occurred",
        "+1 Bomb, -1 Flag, 0 otherwise, when the counterpart identity may legally be shown",
    )
    for side, block in (("own", CH_OWN_BEHAVIOR), ("opponent", CH_OPPONENT_BEHAVIOR)):
        for behavior_type in BEHAVIOR_TYPES:
            base = block + BEHAVIOR_FEATURE_STRIDE * BEHAVIOR_INDEX[behavior_type]
            for offset, feature_name in enumerate(feature_names):
                add(
                    base + offset,
                    f"{side}_{behavior_type}_{feature_name}",
                    feature_ranges[offset],
                    feature_descriptions[offset].format(behavior=behavior_type),
                )

    for offset in range(RECENT_MOVE_WINDOW):
        add(
            CH_RECENT_MOVES + offset,
            f"recent_move_minus_{offset + 1}",
            _COORDINATE_RANGE,
            f"-1 at the source and +1 at the destination of the move played "
            f"{offset + 1} ply/plies ago",
        )
    add(CH_LAKE_MASK, "lake_mask", _BINARY_RANGE, "1 on the eight lake squares")
    add(
        CH_GAME_PROGRESS,
        "game_progress",
        _BINARY_RANGE,
        "broadcast min(total_moves / absolute_move_limit, 1)",
    )
    add(
        CH_BATTLELESS_PROGRESS,
        "battleless_progress",
        _BINARY_RANGE,
        "broadcast min(moves_since_last_combat / battleless_move_limit, 1)",
    )

    assert len(metadata) == OBSERVATION_CHANNELS
    assert [entry["channel"] for entry in metadata] == list(range(OBSERVATION_CHANNELS))
    return metadata


def observation_metadata_document() -> dict:
    """Full metadata document, including shape and dtype, for export."""
    return {
        "observation_version": OBSERVATION_VERSION,
        "shape": list(OBSERVATION_SHAPE),
        "dtype": np.dtype(OBSERVATION_DTYPE).name,
        "board_rows": BOARD_ROWS,
        "board_columns": BOARD_COLUMNS,
        "perspective": {
            PLAYER_NAMES[0]: "identity",
            PLAYER_NAMES[1]: "180 degree rotation (square -> 99 - square)",
        },
        "legal_action_mask": {
            "separate_input": True,
            "size": NUM_SQUARES * NUM_SQUARES,
            "encoding": "action_id = 100 * source + destination",
        },
        "channels": observation_channel_metadata(),
    }
