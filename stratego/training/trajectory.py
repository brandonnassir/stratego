"""Compact self-play trajectory records.

Specification sources:

- `08_internal_state_spec.md` sections 15, 16 (snapshot contents, belief targets)
- `09_public_event_and_replay_schema.md` sections 3, 13, 18 (replay record)
- `05_project_plan.md` phase three storage requirement

Storage principle
-----------------
A decision never stores its `127 x 10 x 10` observation. It stores the facts the
frozen reference engine needs to rebuild that observation:

- one game header carrying both true setups, the rules configuration and the
  identity of the collecting run;
- the ordered action list;
- a compact engine snapshot every `snapshot_interval` plies;
- one sparse decision record per ply, holding only the legal action identifiers
  and one probability per legal action.

:mod:`stratego.training.reconstruction` turns any of those back into a full
state, an `observation_v2_1_127ch` tensor, a legal-action list and a privileged
belief target. Every derivation runs through `stratego.engine`; nothing in this
module reimplements a rule.

Derived-fact rule
-----------------
Anything the header already determines is not stored again in a snapshot: piece
identifiers, owners, true types and starting squares come from the two setups,
and the board array is rebuilt from the living pieces. The encoder verifies each
derivation against the snapshot it was handed, so a future engine change that
broke an assumption fails loudly at collection time instead of producing records
that decode into the wrong position.

Belief targets are deliberately absent from every structure here. They are a
training target and are produced only by :mod:`reconstruction`, in a field that
is separate from the observation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..engine.actions import ACTION_SPACE_SIZE
from ..engine.constants import (
    IMPLEMENTATION_VERSION,
    NUM_SQUARES,
    OBSERVATION_VERSION,
    PIECES_PER_PLAYER,
    PLAYER_BY_NAME,
    PLAYER_NAMES,
    RULES_VERSION,
    RulesConfig,
)
from ..engine.replay import ReplayRecord, terminal_result_label
from ..engine.setup import serialize_setup, setup_squares
from ..engine.snapshot import SNAPSHOT_VERSION, create_snapshot
from ..engine.state import GameState
from .serialization import (
    SERIALIZATION_VERSION,
    ByteReader,
    ByteWriter,
    CodecError,
    StringTable,
    compress,
    decompress,
    read_string_table,
    to_float32,
    write_string_table,
)

TRAJECTORY_VERSION = "trajectory_v1"

# Bumped only for a change that makes previously written bytes unreadable.
TRAJECTORY_FORMAT_VERSION = 1

_MAGIC = b"STJ1"

# `05_project_plan.md` asks for a configurable snapshot cadence; 32 is the
# initial default and the acceptance harness measures all three.
DEFAULT_SNAPSHOT_INTERVAL = 32
SUPPORTED_SNAPSHOT_INTERVALS = (16, 32, 64)

# The placeholder collection policy used until Agent 4 supplies a real network.
# It is a version string like any other, so a real checkpoint identifier drops
# into the same field without a schema change.
SYNTHETIC_POLICY_VERSION = "synthetic_hash_policy_v1"

# The setup generator behind `BatchSimulator`, recorded as the setup family so a
# later curriculum can be told apart from uniform random placement.
BATCH_RANDOM_SETUP_FAMILY = "batch_random_uniform_v1"

# Sum-to-one tolerance for a stored `float32` distribution.
PROBABILITY_SUM_TOLERANCE = 1e-4

# Exactly what `create_snapshot(state, include_history=False)` produces. The
# encoder refuses anything else rather than silently dropping a field.
_EXPECTED_SNAPSHOT_FIELDS = frozenset(
    {
        "snapshot_version",
        "rules",
        "game_id",
        "board",
        "pieces",
        "acting_player",
        "phase",
        "total_moves",
        "battleless_moves",
        "terminal",
        "terminal_reason",
        "winner",
        "is_draw",
        "recent_moves",
        "active_threat_relations",
        "behavior_memory",
    }
)

_NUM_PIECES = 2 * PIECES_PER_PLAYER


class TrajectoryError(ValueError):
    """A trajectory record is inconsistent with the frozen engine contracts."""


# ---------------------------------------------------------------------------
# Setup identity
# ---------------------------------------------------------------------------


def setup_id(red_setup: "tuple[int, ...]", blue_setup: "tuple[int, ...]") -> str:
    """Stable short identifier for a pair of true setups.

    The engine has no setup-family concept of its own, so the identifier is
    derived here: a digest over both serialised setups. It is reproducible from
    the record and is what a later setup curriculum can group on.
    """
    payload = f"{serialize_setup(red_setup)}|{serialize_setup(blue_setup)}".encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# Deterministic synthetic model outputs
# ---------------------------------------------------------------------------
#
# There is no network yet. These stand in for one so that storage fidelity can
# be tested end to end: a sparse distribution over the real legal set and a
# three-class value. Agent 4 replaces the two calls, not the schema.


def _uniform_from_digest(*parts: object) -> float:
    payload = "|".join(str(part) for part in parts).encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return (int.from_bytes(digest, "big") + 1) / (2**64 + 1)


def synthetic_policy(
    game_id: str, ply: int, legal_action_ids: "tuple[int, ...] | list[int]"
) -> tuple[float, ...]:
    """Deterministic normalised distribution over exactly the legal actions."""
    if not legal_action_ids:
        raise TrajectoryError("a decision needs at least one legal action")
    weights = [
        _uniform_from_digest("policy", game_id, ply, action) for action in legal_action_ids
    ]
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def synthetic_value(game_id: str, ply: int) -> tuple[float, float, float]:
    """Deterministic normalised win/draw/loss prediction."""
    weights = [_uniform_from_digest("value", game_id, ply, label) for label in ("w", "d", "l")]
    total = sum(weights)
    return (weights[0] / total, weights[1] / total, weights[2] / total)


def select_action_from_policy(
    game_id: str, ply: int, legal_action_ids: "tuple[int, ...] | list[int]",
    probabilities: "tuple[float, ...] | list[float]",
) -> int:
    """Deterministically sample one action from a decision's own distribution.

    Sampling rather than taking the argmax keeps collected games diverse while
    staying a pure function of `(game_id, ply)`, so a run reproduces exactly.
    """
    threshold = _uniform_from_digest("select", game_id, ply)
    cumulative = 0.0
    for action, probability in zip(legal_action_ids, probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return int(action)
    return int(legal_action_ids[-1])  # pragma: no cover - float32 tail guard


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionRecord:
    """One stored decision.

    `legal_action_ids` is the ascending legal set from the reference engine and
    `old_probabilities` holds one entry per legal action in the same order. The
    dense 10,000-entry vector is never stored; :mod:`reconstruction` rebuilds a
    dense mask on demand when a consumer needs one.
    """

    game_id: str
    ply: int
    acting_player: int
    selected_action_id: int
    legal_action_ids: tuple[int, ...]
    old_probabilities: tuple[float, ...]
    win_draw_loss_prediction: tuple[float, float, float]
    collection_policy_version: str
    snapshot_reference: int

    @property
    def selected_action_index(self) -> int:
        """Position of the selected action inside the legal list."""
        return self.legal_action_ids.index(self.selected_action_id)


@dataclass(frozen=True)
class SnapshotEntry:
    """One compact engine snapshot, addressed by the ply it was taken at."""

    ply: int
    payload: bytes

    @property
    def nbytes(self) -> int:
        return len(self.payload)


@dataclass(frozen=True)
class GameRecord:
    """One complete collected game.

    Everything needed to rebuild any historical position is here: the identity
    of the run, both true setups, the rules configuration, the ordered actions,
    the periodic snapshots and the sparse decisions.
    """

    game_id: str
    environment_id: int
    generation: int
    trajectory_version: str
    rules_version: str
    observation_version: str
    implementation_version: str
    red_setup: tuple[int, ...]
    blue_setup: tuple[int, ...]
    first_player: str
    setup_family: str | None
    setup_id: str | None
    board_geometry_version: str
    battleless_move_limit: int
    absolute_move_limit: int
    rules_context: str
    terminal_result: str
    terminal_reason: str
    final_ply: int
    collection_policy_version: str
    collection_checkpoint_id: str | None
    root_seed: int
    slot_seed: int
    snapshot_interval: int
    actions: tuple[int, ...]
    snapshots: tuple[SnapshotEntry, ...]
    decisions: tuple[DecisionRecord, ...]

    # -- derived views -----------------------------------------------------

    def rules(self) -> RulesConfig:
        """The exact rules configuration this game was played under."""
        return RulesConfig(
            rules_version=self.rules_version,
            board_geometry_version=self.board_geometry_version,
            first_player=PLAYER_BY_NAME[self.first_player],
            battleless_move_limit=self.battleless_move_limit,
            absolute_move_limit=self.absolute_move_limit,
            context=self.rules_context,
        )

    def context(self) -> "GameContext":
        return GameContext(
            rules=self.rules(),
            game_id=self.game_id,
            red_setup=self.red_setup,
            blue_setup=self.blue_setup,
        )

    def snapshot_index_for_ply(self, ply: int) -> int:
        """Index of the nearest snapshot at or before `ply`."""
        if not 0 <= ply <= len(self.actions):
            raise TrajectoryError(
                f"ply {ply} is outside game {self.game_id} (0..{len(self.actions)})"
            )
        chosen = -1
        for index, entry in enumerate(self.snapshots):
            if entry.ply <= ply:
                chosen = index
            else:
                break
        if chosen < 0:  # pragma: no cover - a ply-0 snapshot is always written
            raise TrajectoryError(f"game {self.game_id} has no snapshot at or before ply {ply}")
        return chosen

    def decision_at(self, ply: int) -> DecisionRecord:
        for decision in self.decisions:
            if decision.ply == ply:
                return decision
        raise TrajectoryError(f"game {self.game_id} stored no decision at ply {ply}")

    def to_replay_record(self) -> ReplayRecord:
        """Interoperability with the frozen replay schema."""
        from ..engine.constants import EVENT_SCHEMA_VERSION, REPLAY_VERSION

        return ReplayRecord(
            replay_version=REPLAY_VERSION,
            rules_version=self.rules_version,
            observation_version=self.observation_version,
            event_schema_version=EVENT_SCHEMA_VERSION,
            game_id=self.game_id,
            red_setup=serialize_setup(self.red_setup),
            blue_setup=serialize_setup(self.blue_setup),
            first_player=self.first_player,
            battleless_move_limit=self.battleless_move_limit,
            absolute_move_limit=self.absolute_move_limit,
            rules_context=self.rules_context,
            actions=list(self.actions),
            terminal_result=self.terminal_result,
            terminal_reason=self.terminal_reason,
            total_moves=self.final_ply,
            seeds={
                "trajectory_version": self.trajectory_version,
                "root_seed": self.root_seed,
                "environment_id": self.environment_id,
                "generation": self.generation,
                "slot_seed": self.slot_seed,
            },
        )

    @property
    def snapshot_bytes(self) -> int:
        return sum(entry.nbytes for entry in self.snapshots)


@dataclass(frozen=True)
class GameContext:
    """The per-game facts a snapshot decoder needs but does not store."""

    rules: RulesConfig
    game_id: str
    red_setup: tuple[int, ...]
    blue_setup: tuple[int, ...]

    def true_type(self, piece_id: int) -> int:
        setup = self.red_setup if piece_id < PIECES_PER_PLAYER else self.blue_setup
        return setup[piece_id % PIECES_PER_PLAYER]

    def starting_square(self, piece_id: int) -> int:
        owner = piece_id // PIECES_PER_PLAYER
        return setup_squares(owner)[piece_id % PIECES_PER_PLAYER]


# ---------------------------------------------------------------------------
# Snapshot codec
# ---------------------------------------------------------------------------


def encode_snapshot(snapshot: dict, context: GameContext) -> bytes:
    """Encode one `create_snapshot(..., include_history=False)` dictionary.

    Self-contained apart from `context`: given the game header, the returned
    bytes decode back to a dictionary that `restore_snapshot` accepts.
    """
    fields = set(snapshot)
    if fields != _EXPECTED_SNAPSHOT_FIELDS:
        missing = sorted(_EXPECTED_SNAPSHOT_FIELDS - fields)
        extra = sorted(fields - _EXPECTED_SNAPSHOT_FIELDS)
        raise TrajectoryError(
            "the engine snapshot no longer matches the trajectory codec "
            f"(missing={missing}, unexpected={extra}); revise the codec rather "
            "than dropping fields"
        )
    if snapshot["snapshot_version"] != SNAPSHOT_VERSION:
        raise TrajectoryError(
            f"unsupported engine snapshot version: {snapshot['snapshot_version']!r}"
        )
    if snapshot["game_id"] != context.game_id:
        raise TrajectoryError(
            f"snapshot game {snapshot['game_id']!r} does not belong to context "
            f"game {context.game_id!r}"
        )

    pieces = snapshot["pieces"]
    if len(pieces) != _NUM_PIECES:
        raise TrajectoryError(f"expected {_NUM_PIECES} piece records, got {len(pieces)}")

    table = StringTable()
    body = ByteWriter()

    body.uvarint(snapshot["acting_player"])
    body.uvarint(table.intern(snapshot["phase"]))
    body.uvarint(snapshot["total_moves"])
    body.uvarint(snapshot["battleless_moves"])
    body.flags((bool(snapshot["terminal"]), bool(snapshot["is_draw"])))
    body.uvarint(table.intern(snapshot["terminal_reason"]))
    body.optional_uvarint(snapshot["winner"])

    rebuilt_board: list[int | None] = [None] * NUM_SQUARES
    for piece_id, record in enumerate(pieces):
        (
            stored_id,
            owner,
            true_type,
            starting_square,
            current_square,
            alive,
            has_moved,
            known_to_red,
            known_to_blue,
            reveal_reason_red,
            reveal_reason_blue,
            capture_ply,
        ) = record
        # Derived-fact checks: these four are already in the game header.
        if stored_id != piece_id or owner != piece_id // PIECES_PER_PLAYER:
            raise TrajectoryError(
                f"piece record {piece_id} is not in canonical identifier order"
            )
        if true_type != context.true_type(piece_id):
            raise TrajectoryError(
                f"piece {piece_id} true type {true_type} disagrees with the stored setup"
            )
        if starting_square != context.starting_square(piece_id):
            raise TrajectoryError(
                f"piece {piece_id} starting square {starting_square} disagrees with "
                "the canonical setup squares"
            )
        if alive and current_square is not None:
            rebuilt_board[current_square] = piece_id

        body.flags(
            (
                bool(alive),
                bool(has_moved),
                bool(known_to_red),
                bool(known_to_blue),
                current_square is not None,
                reveal_reason_red is not None,
                reveal_reason_blue is not None,
                capture_ply is not None,
            )
        )
        if current_square is not None:
            body.uvarint(current_square)
        if reveal_reason_red is not None:
            body.uvarint(table.intern(reveal_reason_red))
        if reveal_reason_blue is not None:
            body.uvarint(table.intern(reveal_reason_blue))
        if capture_ply is not None:
            body.uvarint(capture_ply)

    if rebuilt_board != list(snapshot["board"]):
        raise TrajectoryError(
            "the board could not be derived from the piece records; the codec's "
            "occupancy assumption no longer holds"
        )

    recent_moves = snapshot["recent_moves"]
    body.uvarint(len(recent_moves))
    for move in recent_moves:
        ply, player, piece_id, source, destination, had_opponent, target = move
        body.uvarint(ply)
        body.flags((bool(player), bool(had_opponent)))
        body.uvarint(piece_id)
        body.uvarint(source)
        body.uvarint(destination)
        body.optional_uvarint(target)

    relations = snapshot["active_threat_relations"]
    body.uvarint(len(relations))
    for threatener, threatened, creation_ply in relations:
        body.uvarint(threatener)
        body.uvarint(threatened)
        body.uvarint(creation_ply)

    memory = snapshot["behavior_memory"]
    body.uvarint(len(memory))
    for entry in memory:
        (
            piece_id,
            behavior_type,
            event_type,
            actor_piece_id,
            counterpart_piece_id,
            event_ply,
            actor_knew,
            context_piece_id,
        ) = entry
        body.uvarint(piece_id)
        body.uvarint(table.intern(behavior_type))
        body.uvarint(table.intern(event_type))
        body.uvarint(actor_piece_id)
        body.uvarint(counterpart_piece_id)
        body.uvarint(event_ply)
        body.flags((bool(actor_knew),))
        body.optional_uvarint(context_piece_id)

    return write_string_table(table) + body.to_bytes()


def decode_snapshot(payload: bytes, context: GameContext) -> dict:
    """Inverse of :func:`encode_snapshot`, ready for `restore_snapshot`."""
    reader = ByteReader(payload)
    table = read_string_table(reader)

    acting_player = reader.uvarint()
    phase = table.resolve(reader.uvarint())
    total_moves = reader.uvarint()
    battleless_moves = reader.uvarint()
    terminal, is_draw = reader.flags(2)
    terminal_reason = table.resolve(reader.uvarint())
    winner = reader.optional_uvarint()

    board: list[int | None] = [None] * NUM_SQUARES
    pieces = []
    for piece_id in range(_NUM_PIECES):
        (
            alive,
            has_moved,
            known_to_red,
            known_to_blue,
            has_square,
            has_reveal_red,
            has_reveal_blue,
            has_capture_ply,
        ) = reader.flags(8)
        current_square = reader.uvarint() if has_square else None
        reveal_reason_red = table.resolve(reader.uvarint()) if has_reveal_red else None
        reveal_reason_blue = table.resolve(reader.uvarint()) if has_reveal_blue else None
        capture_ply = reader.uvarint() if has_capture_ply else None
        if alive and current_square is not None:
            board[current_square] = piece_id
        pieces.append(
            (
                piece_id,
                piece_id // PIECES_PER_PLAYER,
                context.true_type(piece_id),
                context.starting_square(piece_id),
                current_square,
                alive,
                has_moved,
                known_to_red,
                known_to_blue,
                reveal_reason_red,
                reveal_reason_blue,
                capture_ply,
            )
        )

    recent_moves = []
    for _ in range(reader.uvarint()):
        ply = reader.uvarint()
        player, had_opponent = reader.flags(2)
        piece_id = reader.uvarint()
        source = reader.uvarint()
        destination = reader.uvarint()
        target = reader.optional_uvarint()
        recent_moves.append(
            (ply, int(player), piece_id, source, destination, had_opponent, target)
        )

    relations = []
    for _ in range(reader.uvarint()):
        threatener = reader.uvarint()
        threatened = reader.uvarint()
        creation_ply = reader.uvarint()
        relations.append((threatener, threatened, creation_ply))

    memory = []
    for _ in range(reader.uvarint()):
        piece_id = reader.uvarint()
        behavior_type = table.resolve(reader.uvarint())
        event_type = table.resolve(reader.uvarint())
        actor_piece_id = reader.uvarint()
        counterpart_piece_id = reader.uvarint()
        event_ply = reader.uvarint()
        (actor_knew,) = reader.flags(1)
        context_piece_id = reader.optional_uvarint()
        memory.append(
            (
                piece_id,
                behavior_type,
                event_type,
                actor_piece_id,
                counterpart_piece_id,
                event_ply,
                actor_knew,
                context_piece_id,
            )
        )

    reader.expect_exhausted()

    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "rules": context.rules,
        "game_id": context.game_id,
        "board": tuple(board),
        "pieces": tuple(pieces),
        "acting_player": acting_player,
        "phase": phase,
        "total_moves": total_moves,
        "battleless_moves": battleless_moves,
        "terminal": terminal,
        "terminal_reason": terminal_reason,
        "winner": winner,
        "is_draw": is_draw,
        "recent_moves": tuple(recent_moves),
        "active_threat_relations": tuple(relations),
        "behavior_memory": tuple(memory),
    }


# ---------------------------------------------------------------------------
# Game record codec
# ---------------------------------------------------------------------------


def encode_game_record(record: GameRecord) -> bytes:
    """Serialise one game to the compact binary format."""
    table = StringTable()
    body = ByteWriter()

    for text in (
        record.trajectory_version,
        record.rules_version,
        record.observation_version,
        record.implementation_version,
        record.game_id,
        record.first_player,
        record.board_geometry_version,
        record.rules_context,
        record.terminal_result,
        record.terminal_reason,
    ):
        body.uvarint(table.intern(text))
    body.uvarint(table.intern(record.setup_family))
    body.uvarint(table.intern(record.setup_id))
    body.uvarint(table.intern(record.collection_policy_version))
    body.uvarint(table.intern(record.collection_checkpoint_id))

    body.uvarint(record.environment_id)
    body.uvarint(record.generation)
    body.uvarint(record.root_seed)
    body.uvarint(record.slot_seed)
    body.uvarint(record.battleless_move_limit)
    body.uvarint(record.absolute_move_limit)
    body.uvarint(record.final_ply)
    body.uvarint(record.snapshot_interval)

    if len(record.red_setup) != PIECES_PER_PLAYER or len(record.blue_setup) != PIECES_PER_PLAYER:
        raise TrajectoryError("a setup must hold exactly 40 piece types")
    for piece_type in record.red_setup:
        body.uvarint(piece_type)
    for piece_type in record.blue_setup:
        body.uvarint(piece_type)

    body.uvarint(len(record.actions))
    for action_id in record.actions:
        body.uvarint(action_id)

    body.uvarint(len(record.snapshots))
    for entry in record.snapshots:
        body.uvarint(entry.ply)
        body.blob(entry.payload)

    body.uvarint(len(record.decisions))
    for decision in record.decisions:
        body.uvarint(decision.ply)
        body.uvarint(decision.acting_player)
        body.uvarint(decision.selected_action_id)
        body.uvarint(decision.snapshot_reference)
        body.uvarint(table.intern(decision.collection_policy_version))
        body.ascending_uvarints(decision.legal_action_ids)
        if len(decision.old_probabilities) != len(decision.legal_action_ids):
            raise TrajectoryError(
                f"decision at ply {decision.ply} has {len(decision.old_probabilities)} "
                f"probabilities for {len(decision.legal_action_ids)} legal actions"
            )
        for probability in decision.old_probabilities:
            body.float32(probability)
        for component in decision.win_draw_loss_prediction:
            body.float32(component)

    header = ByteWriter()
    header.raw(_MAGIC)
    header.uvarint(TRAJECTORY_FORMAT_VERSION)
    return header.to_bytes() + write_string_table(table) + body.to_bytes()


def decode_game_record(payload: bytes) -> GameRecord:
    """Inverse of :func:`encode_game_record`."""
    if payload[:4] != _MAGIC:
        raise CodecError("not a trajectory record: bad magic")
    reader = ByteReader(payload[4:])
    format_version = reader.uvarint()
    if format_version != TRAJECTORY_FORMAT_VERSION:
        raise CodecError(
            f"unsupported trajectory format version {format_version}; this build "
            f"writes version {TRAJECTORY_FORMAT_VERSION}"
        )
    table = read_string_table(reader)

    # A list comprehension, not a generator: the reads must happen in written
    # order, which only the eager form guarantees.
    (
        trajectory_version,
        rules_version,
        observation_version,
        implementation_version,
        game_id,
        first_player,
        board_geometry_version,
        rules_context,
        terminal_result,
        terminal_reason,
    ) = [table.resolve(reader.uvarint()) for _ in range(10)]
    setup_family = table.resolve(reader.uvarint())
    record_setup_id = table.resolve(reader.uvarint())
    collection_policy_version = table.resolve(reader.uvarint())
    collection_checkpoint_id = table.resolve(reader.uvarint())

    environment_id = reader.uvarint()
    generation = reader.uvarint()
    root_seed = reader.uvarint()
    slot_seed = reader.uvarint()
    battleless_move_limit = reader.uvarint()
    absolute_move_limit = reader.uvarint()
    final_ply = reader.uvarint()
    snapshot_interval = reader.uvarint()

    red_setup = tuple(reader.uvarint() for _ in range(PIECES_PER_PLAYER))
    blue_setup = tuple(reader.uvarint() for _ in range(PIECES_PER_PLAYER))

    actions = tuple(reader.uvarint() for _ in range(reader.uvarint()))

    snapshots = []
    for _ in range(reader.uvarint()):
        snapshot_ply = reader.uvarint()
        snapshots.append(SnapshotEntry(ply=snapshot_ply, payload=reader.blob()))
    snapshots = tuple(snapshots)

    decisions = []
    for _ in range(reader.uvarint()):
        ply = reader.uvarint()
        acting_player = reader.uvarint()
        selected_action_id = reader.uvarint()
        snapshot_reference = reader.uvarint()
        decision_policy_version = table.resolve(reader.uvarint())
        legal_action_ids = reader.ascending_uvarints()
        probabilities = tuple(reader.float32() for _ in legal_action_ids)
        value_prediction = (reader.float32(), reader.float32(), reader.float32())
        decisions.append(
            DecisionRecord(
                game_id=game_id,
                ply=ply,
                acting_player=acting_player,
                selected_action_id=selected_action_id,
                legal_action_ids=legal_action_ids,
                old_probabilities=probabilities,
                win_draw_loss_prediction=value_prediction,
                collection_policy_version=decision_policy_version,
                snapshot_reference=snapshot_reference,
            )
        )

    reader.expect_exhausted()

    return GameRecord(
        game_id=game_id,
        environment_id=environment_id,
        generation=generation,
        trajectory_version=trajectory_version,
        rules_version=rules_version,
        observation_version=observation_version,
        implementation_version=implementation_version,
        red_setup=red_setup,
        blue_setup=blue_setup,
        first_player=first_player,
        setup_family=setup_family,
        setup_id=record_setup_id,
        board_geometry_version=board_geometry_version,
        battleless_move_limit=battleless_move_limit,
        absolute_move_limit=absolute_move_limit,
        rules_context=rules_context,
        terminal_result=terminal_result,
        terminal_reason=terminal_reason,
        final_ply=final_ply,
        collection_policy_version=collection_policy_version,
        collection_checkpoint_id=collection_checkpoint_id,
        root_seed=root_seed,
        slot_seed=slot_seed,
        snapshot_interval=snapshot_interval,
        actions=actions,
        snapshots=snapshots,
        decisions=tuple(decisions),
    )


def encode_game_record_compressed(record: GameRecord, level: int | None = None) -> bytes:
    from .serialization import DEFAULT_COMPRESSION_LEVEL

    return compress(
        encode_game_record(record),
        DEFAULT_COMPRESSION_LEVEL if level is None else level,
    )


def decode_game_record_compressed(payload: bytes) -> GameRecord:
    return decode_game_record(decompress(payload))


# ---------------------------------------------------------------------------
# Sparse decision-storage validation
# ---------------------------------------------------------------------------


def validate_decision_record(decision: DecisionRecord) -> list[str]:
    """Every sparse-storage rule, as a list of human-readable problems."""
    problems: list[str] = []
    legal = decision.legal_action_ids

    if not legal:
        problems.append(f"ply {decision.ply}: empty legal action list")
    if len(set(legal)) != len(legal):
        problems.append(f"ply {decision.ply}: duplicate legal action identifiers")
    if list(legal) != sorted(legal):
        problems.append(f"ply {decision.ply}: legal action identifiers are not ascending")
    if any(not 0 <= action < ACTION_SPACE_SIZE for action in legal):
        problems.append(f"ply {decision.ply}: legal action identifier outside the action space")
    if len(decision.old_probabilities) != len(legal):
        problems.append(
            f"ply {decision.ply}: {len(decision.old_probabilities)} probabilities for "
            f"{len(legal)} legal actions"
        )
    else:
        total = 0.0
        for probability in decision.old_probabilities:
            if probability != probability or probability in (float("inf"), float("-inf")):
                problems.append(f"ply {decision.ply}: non-finite probability")
                break
            if probability < 0.0:
                problems.append(f"ply {decision.ply}: negative probability")
                break
            total += probability
        else:
            if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
                problems.append(f"ply {decision.ply}: probabilities sum to {total!r}")
    if decision.selected_action_id not in legal:
        problems.append(
            f"ply {decision.ply}: selected action {decision.selected_action_id} is not legal"
        )
    if not decision.collection_policy_version:
        problems.append(f"ply {decision.ply}: missing collection policy version")
    if decision.acting_player not in (0, 1):
        problems.append(f"ply {decision.ply}: acting player {decision.acting_player}")

    value = decision.win_draw_loss_prediction
    if len(value) != 3:
        problems.append(f"ply {decision.ply}: value prediction has {len(value)} entries")
    else:
        if any(component != component or component in (float("inf"), float("-inf")) for component in value):
            problems.append(f"ply {decision.ply}: non-finite value component")
        elif all(component >= 0.0 for component in value):
            # Stored as probabilities, so it must be normalised.
            if abs(sum(value) - 1.0) > PROBABILITY_SUM_TOLERANCE:
                problems.append(f"ply {decision.ply}: value probabilities sum to {sum(value)!r}")
    return problems


def validate_game_record(record: GameRecord) -> list[str]:
    """Structural validation of a whole game record."""
    problems: list[str] = []

    if record.trajectory_version != TRAJECTORY_VERSION:
        problems.append(f"unexpected trajectory version {record.trajectory_version!r}")
    if record.rules_version != RULES_VERSION:
        problems.append(f"unexpected rules version {record.rules_version!r}")
    if record.observation_version != OBSERVATION_VERSION:
        problems.append(f"unexpected observation version {record.observation_version!r}")
    if record.implementation_version != IMPLEMENTATION_VERSION:
        problems.append(f"unexpected implementation version {record.implementation_version!r}")
    if record.snapshot_interval <= 0:
        problems.append(f"non-positive snapshot interval {record.snapshot_interval}")
    if record.final_ply != len(record.actions):
        problems.append(
            f"final ply {record.final_ply} disagrees with {len(record.actions)} actions"
        )
    if len(record.decisions) != len(record.actions):
        problems.append(
            f"{len(record.decisions)} decisions for {len(record.actions)} actions"
        )
    if not record.snapshots or record.snapshots[0].ply != 0:
        problems.append("the first snapshot must be at ply 0")

    snapshot_plies = [entry.ply for entry in record.snapshots]
    if snapshot_plies != sorted(set(snapshot_plies)):
        problems.append("snapshot plies are not strictly ascending")
    for entry in record.snapshots:
        if entry.ply % record.snapshot_interval != 0:
            problems.append(f"snapshot at ply {entry.ply} is off the configured cadence")

    for index, decision in enumerate(record.decisions):
        if decision.game_id != record.game_id:
            problems.append(f"decision {index} carries game {decision.game_id!r}")
        if decision.ply != index:
            problems.append(f"decision {index} is stamped ply {decision.ply}")
        elif decision.selected_action_id != record.actions[index]:
            problems.append(
                f"ply {index}: stored action {decision.selected_action_id} disagrees with "
                f"the action list entry {record.actions[index]}"
            )
        if not 0 <= decision.snapshot_reference < len(record.snapshots):
            problems.append(f"ply {decision.ply}: snapshot reference out of range")
        elif record.snapshots[decision.snapshot_reference].ply > decision.ply:
            problems.append(
                f"ply {decision.ply}: snapshot reference points past the decision"
            )
        problems.extend(validate_decision_record(decision))

    return problems


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


@dataclass
class GameTrajectoryBuilder:
    """Accumulates one game's decisions and snapshots as it is played.

    A caller records each decision *before* the action is applied, then calls
    :meth:`finish` on the terminal state. Snapshots are taken automatically on
    the configured cadence, which is why the builder needs the live state rather
    than just the chosen action.
    """

    game_id: str
    environment_id: int
    generation: int
    red_setup: tuple[int, ...]
    blue_setup: tuple[int, ...]
    rules: RulesConfig
    root_seed: int = 0
    slot_seed: int = 0
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL
    collection_policy_version: str = SYNTHETIC_POLICY_VERSION
    collection_checkpoint_id: str | None = None
    setup_family: str | None = BATCH_RANDOM_SETUP_FAMILY

    _actions: list[int] = field(default_factory=list, init=False, repr=False)
    _snapshots: list[SnapshotEntry] = field(default_factory=list, init=False, repr=False)
    _decisions: list[DecisionRecord] = field(default_factory=list, init=False, repr=False)
    _context: GameContext | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.snapshot_interval <= 0:
            raise TrajectoryError("snapshot_interval must be positive")
        self._context = GameContext(
            rules=self.rules,
            game_id=self.game_id,
            red_setup=tuple(self.red_setup),
            blue_setup=tuple(self.blue_setup),
        )

    @property
    def context(self) -> GameContext:
        assert self._context is not None
        return self._context

    def record_decision(
        self,
        state: GameState,
        *,
        legal_action_ids: "tuple[int, ...] | list[int]",
        probabilities: "tuple[float, ...] | list[float]",
        win_draw_loss_prediction: "tuple[float, float, float]",
        selected_action_id: int,
        collection_policy_version: str | None = None,
    ) -> DecisionRecord:
        """Store one decision, taking a snapshot first when the cadence says so."""
        ply = state.total_moves
        if ply != len(self._decisions):
            raise TrajectoryError(
                f"game {self.game_id}: decision for ply {ply} arrived after "
                f"{len(self._decisions)} decisions; record every ply in order"
            )
        if state.game_id != self.game_id:
            raise TrajectoryError(
                f"state game {state.game_id!r} does not match builder game {self.game_id!r}"
            )
        if ply % self.snapshot_interval == 0:
            self._snapshots.append(
                SnapshotEntry(
                    ply=ply,
                    payload=encode_snapshot(create_snapshot(state), self.context),
                )
            )

        decision = DecisionRecord(
            game_id=self.game_id,
            ply=ply,
            acting_player=state.acting_player,
            selected_action_id=int(selected_action_id),
            legal_action_ids=tuple(int(action) for action in legal_action_ids),
            # Rounded to `float32` at the point of storage, so the in-memory
            # record is byte-for-byte what its encoded form decodes back to.
            old_probabilities=tuple(to_float32(value) for value in probabilities),
            win_draw_loss_prediction=(
                to_float32(win_draw_loss_prediction[0]),
                to_float32(win_draw_loss_prediction[1]),
                to_float32(win_draw_loss_prediction[2]),
            ),
            collection_policy_version=(
                collection_policy_version or self.collection_policy_version
            ),
            snapshot_reference=len(self._snapshots) - 1,
        )
        self._decisions.append(decision)
        self._actions.append(int(selected_action_id))
        return decision

    def finish(self, state: GameState) -> GameRecord:
        """Seal the record on a terminal state."""
        if not state.terminal:
            raise TrajectoryError(
                f"game {self.game_id} is not terminal ({state.terminal_reason})"
            )
        if list(state.action_history) != self._actions:
            raise TrajectoryError(
                f"game {self.game_id}: recorded {len(self._actions)} actions but the "
                f"engine applied {len(state.action_history)}"
            )
        return GameRecord(
            game_id=self.game_id,
            environment_id=self.environment_id,
            generation=self.generation,
            trajectory_version=TRAJECTORY_VERSION,
            rules_version=self.rules.rules_version,
            observation_version=OBSERVATION_VERSION,
            implementation_version=IMPLEMENTATION_VERSION,
            red_setup=tuple(self.red_setup),
            blue_setup=tuple(self.blue_setup),
            first_player=PLAYER_NAMES[self.rules.first_player],
            setup_family=self.setup_family,
            setup_id=setup_id(self.red_setup, self.blue_setup),
            board_geometry_version=self.rules.board_geometry_version,
            battleless_move_limit=self.rules.battleless_move_limit,
            absolute_move_limit=self.rules.absolute_move_limit,
            rules_context=self.rules.context,
            terminal_result=terminal_result_label(state),
            terminal_reason=state.terminal_reason,
            final_ply=state.total_moves,
            collection_policy_version=self.collection_policy_version,
            collection_checkpoint_id=self.collection_checkpoint_id,
            root_seed=self.root_seed,
            slot_seed=self.slot_seed,
            snapshot_interval=self.snapshot_interval,
            actions=tuple(self._actions),
            snapshots=tuple(self._snapshots),
            decisions=tuple(self._decisions),
        )


def builder_for_slot(
    simulator,
    slot: int,
    *,
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
    collection_policy_version: str = SYNTHETIC_POLICY_VERSION,
    collection_checkpoint_id: str | None = None,
) -> GameTrajectoryBuilder:
    """Builder for the game currently sitting in a `BatchSimulator` slot.

    Everything the header needs is already on the slot, including the identity
    triple `(root_seed, environment_id, generation)` that Agent 1 made
    sufficient to regenerate the game from scratch.
    """
    red_setup, blue_setup = simulator.setups(slot)
    return GameTrajectoryBuilder(
        game_id=simulator.game_id(slot),
        environment_id=simulator.environment_id(slot),
        generation=simulator.generation(slot),
        red_setup=red_setup,
        blue_setup=blue_setup,
        rules=simulator.rules,
        root_seed=simulator.root_seed,
        slot_seed=simulator.slot_seed(slot),
        snapshot_interval=snapshot_interval,
        collection_policy_version=collection_policy_version,
        collection_checkpoint_id=collection_checkpoint_id,
    )


def collect_games(
    simulator,
    *,
    games: int,
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
    collection_policy_version: str = SYNTHETIC_POLICY_VERSION,
    collection_checkpoint_id: str | None = None,
    on_decision=None,
    on_game_finished=None,
):
    """Play `games` complete games on `simulator`, yielding `GameRecord`s.

    The synthetic policy stands in for Agent 4's network: it produces a sparse
    distribution over the true legal set and the action is sampled from it, so
    the collected games are varied but fully determined by the simulator's
    seeds.

    `on_decision(state, decision, builder)` is called with the live state
    immediately before the action is applied; `builder` carries the slot
    identity, so a consumer does not have to parse it back out of the game
    identifier. The acceptance harness uses the callback to capture the
    live-time digests that reconstruction is later compared against; nothing it
    returns is stored in the record.

    `on_game_finished(record, state)` is called with the sealed record and the
    live terminal state, before the slot is reset. It is the only chance to read
    the finished game's derived event log, which a compact record does not
    carry.
    """
    builders: dict[int, GameTrajectoryBuilder] = {
        slot: builder_for_slot(
            simulator,
            slot,
            snapshot_interval=snapshot_interval,
            collection_policy_version=collection_policy_version,
            collection_checkpoint_id=collection_checkpoint_id,
        )
        for slot in range(len(simulator))
    }
    produced = 0

    while produced < games:
        active = simulator.active_slots()
        if not active:  # pragma: no cover - a reset always refills the batch
            break
        actions: dict[int, int] = {}
        for slot in active:
            state = simulator.game_state(slot)
            legal = simulator.legal_actions(slot)
            probabilities = synthetic_policy(state.game_id, state.total_moves, legal)
            value = synthetic_value(state.game_id, state.total_moves)
            selected = select_action_from_policy(
                state.game_id, state.total_moves, legal, probabilities
            )
            decision = builders[slot].record_decision(
                state,
                legal_action_ids=legal,
                probabilities=probabilities,
                win_draw_loss_prediction=value,
                selected_action_id=selected,
            )
            if on_decision is not None:
                on_decision(state, decision, builders[slot])
            actions[slot] = selected

        result = simulator.step(actions)

        for slot in result.newly_terminal:
            terminal_state = simulator.game_state(slot)
            record = builders[slot].finish(terminal_state)
            if on_game_finished is not None:
                on_game_finished(record, terminal_state)
            produced += 1
            yield record
            if produced >= games:
                return

        if result.newly_terminal:
            simulator.reset_slots(result.newly_terminal)
            for slot in result.newly_terminal:
                builders[slot] = builder_for_slot(
                    simulator,
                    slot,
                    snapshot_interval=snapshot_interval,
                    collection_policy_version=collection_policy_version,
                    collection_checkpoint_id=collection_checkpoint_id,
                )


__all__ = [
    "BATCH_RANDOM_SETUP_FAMILY",
    "DEFAULT_SNAPSHOT_INTERVAL",
    "PROBABILITY_SUM_TOLERANCE",
    "SERIALIZATION_VERSION",
    "SUPPORTED_SNAPSHOT_INTERVALS",
    "SYNTHETIC_POLICY_VERSION",
    "TRAJECTORY_FORMAT_VERSION",
    "TRAJECTORY_VERSION",
    "DecisionRecord",
    "GameContext",
    "GameRecord",
    "GameTrajectoryBuilder",
    "SnapshotEntry",
    "TrajectoryError",
    "builder_for_slot",
    "collect_games",
    "decode_game_record",
    "decode_game_record_compressed",
    "decode_snapshot",
    "encode_game_record",
    "encode_game_record_compressed",
    "encode_snapshot",
    "select_action_from_policy",
    "setup_id",
    "synthetic_policy",
    "synthetic_value",
    "validate_decision_record",
    "validate_game_record",
]
