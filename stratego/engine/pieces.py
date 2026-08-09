"""Physical piece records, stable identifiers and the piece record container.

Specification sources:

- `03_game_engine_spec.md` section 5 (minimum piece fields)
- `08_internal_state_spec.md` sections 4, 5 (canonical identifier, piece record)

A *physical piece* is one of the 80 pieces that exist for the whole game. Its
identifier never changes, does not encode its type, and remains valid after the
piece is captured.
"""

from dataclasses import dataclass, replace

from .constants import (
    BLUE,
    FLAG,
    IMMOVABLE_TYPES,
    PIECE_RANKS,
    PIECE_TYPE_NAMES,
    PIECES_PER_PLAYER,
    PLAYER_NAMES,
    RED,
)

# ---------------------------------------------------------------------------
# Stable identifiers
# ---------------------------------------------------------------------------
#
# `08_internal_state_spec.md` section 4 recommends the conceptual form
# `(owner, setup_slot_index)`. The engine stores that pair packed into a single
# integer so that identifiers are cheap to compare, hash and serialise:
#
#     piece_id = owner * 40 + setup_slot_index
#
# Red therefore owns identifiers 0..39 and blue owns 40..79. The packing is
# purely positional and carries no information about the piece's type, which is
# the property the specification actually requires.


def make_piece_id(owner: int, setup_slot: int) -> int:
    """Pack `(owner, setup_slot)` into a stable identifier."""
    if not 0 <= setup_slot < PIECES_PER_PLAYER:
        raise ValueError(f"setup slot out of range: {setup_slot}")
    return owner * PIECES_PER_PLAYER + setup_slot


def piece_owner(piece_id: int) -> int:
    """Owner encoded in a piece identifier."""
    return piece_id // PIECES_PER_PLAYER


def piece_setup_slot(piece_id: int) -> int:
    """Row-major setup slot `0..39` encoded in a piece identifier."""
    return piece_id % PIECES_PER_PLAYER


def piece_id_name(piece_id: int) -> str:
    """Human/serialisation form of an identifier, for example `red:07`."""
    return f"{PLAYER_NAMES[piece_owner(piece_id)]}:{piece_setup_slot(piece_id):02d}"


def piece_id_from_name(name: str) -> int:
    """Inverse of :func:`piece_id_name`."""
    owner_text, _, slot_text = name.partition(":")
    owner = RED if owner_text == "red" else BLUE
    return make_piece_id(owner, int(slot_text))


# ---------------------------------------------------------------------------
# Piece record
# ---------------------------------------------------------------------------


@dataclass
class PieceRecord:
    """One of the 80 physical pieces.

    `known_to_red` / `known_to_blue` record whether that player may legally know
    the exact type. A player always knows their own pieces, so the owner's flag
    is `True` from creation and never changes.
    """

    piece_id: int
    owner: int
    true_type: int
    starting_square: int
    current_square: int | None
    alive: bool = True
    has_moved: bool = False
    known_to_red: bool = False
    known_to_blue: bool = False
    reveal_reason_red: str | None = None
    reveal_reason_blue: str | None = None
    capture_ply: int | None = None

    def copy(self) -> "PieceRecord":
        """Independent copy; every field is an immutable scalar."""
        return replace(self)

    @property
    def rank(self) -> int | None:
        """Combat rank, or `None` for Flag and Bomb."""
        return PIECE_RANKS[self.true_type]

    @property
    def is_movable_type(self) -> bool:
        """Whether the true type is able to move at all."""
        return self.true_type not in IMMOVABLE_TYPES

    @property
    def type_name(self) -> str:
        return PIECE_TYPE_NAMES[self.true_type]

    def known_to(self, observer: int) -> bool:
        """Whether `observer` may legally know this piece's exact type."""
        return self.known_to_red if observer == RED else self.known_to_blue

    def set_known_to(self, observer: int, reason: str) -> bool:
        """Grant knowledge to `observer`; return whether knowledge changed.

        Knowledge is monotonic: this method only ever turns a flag on.
        """
        if observer == RED:
            if self.known_to_red:
                return False
            self.known_to_red = True
            self.reveal_reason_red = reason
            return True
        if self.known_to_blue:
            return False
        self.known_to_blue = True
        self.reveal_reason_blue = reason
        return True


def create_piece_records(
    owner: int, setup_types: "list[int] | tuple[int, ...]", setup_squares: tuple[int, ...]
) -> list[PieceRecord]:
    """Build the 40 immutable-identity piece records for one player.

    `setup_types[i]` is the type placed on `setup_squares[i]`, where the setup
    squares are listed in row-major order. The index `i` becomes the piece's
    permanent setup slot and therefore its stable identifier.
    """
    if len(setup_types) != PIECES_PER_PLAYER:
        raise ValueError(f"expected {PIECES_PER_PLAYER} piece types")
    if len(setup_squares) != PIECES_PER_PLAYER:
        raise ValueError(f"expected {PIECES_PER_PLAYER} setup squares")

    records = []
    for slot, (piece_type, square) in enumerate(zip(setup_types, setup_squares)):
        record = PieceRecord(
            piece_id=make_piece_id(owner, slot),
            owner=owner,
            true_type=piece_type,
            starting_square=square,
            current_square=square,
        )
        # A player always knows their own identities.
        record.set_known_to(owner, "own_piece")
        records.append(record)
    return records


def flag_piece_id(records: "list[PieceRecord]", owner: int) -> int:
    """Identifier of a player's Flag, used by terminal-condition checks."""
    for record in records:
        if record.owner == owner and record.true_type == FLAG:
            return record.piece_id
    raise ValueError(f"no flag found for player {owner}")
