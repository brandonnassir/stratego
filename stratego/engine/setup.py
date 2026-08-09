"""Setup representation, validation, generation and serialisation.

Specification sources:

- `02_project_ruleset.md` section 6 (setup legality)
- `03_game_engine_spec.md` section 7 (required setup operations)

A setup is the tuple of 40 piece types assigned to a player's 40 setup squares
in row-major order. Row-major order means index 0 is the leftmost square of the
row furthest from the centre for red, and the same physical enumeration
(`SETUP_SQUARES[player]`, ascending absolute index) for blue.

Setups are validated, never repaired. Every rejection raises `SetupError` with a
message naming the specific violated condition.
"""

import random
from collections import Counter

from .constants import (
    BOARD_COLUMNS,
    LAKE_SQUARE_SET,
    PIECE_COUNTS,
    PIECE_TYPE_BY_CODE,
    PIECE_TYPE_CODES,
    PIECE_TYPE_NAMES,
    PIECES_PER_PLAYER,
    PLAYER_NAMES,
    PLAYERS,
    SETUP_SQUARE_SETS,
    SETUP_SQUARES,
)


class SetupError(ValueError):
    """Raised when a supplied setup violates a documented legality condition."""


def setup_squares(player: int) -> tuple[int, ...]:
    """The player's 40 legal setup squares in row-major order."""
    return SETUP_SQUARES[player]


def validate_setup(setup: "list[int] | tuple[int, ...]", player: int) -> tuple[int, ...]:
    """Validate a row-major setup tuple and return it normalized to a tuple.

    Checks, in the order listed by `02_project_ruleset.md` section 6:

    - exactly 40 entries;
    - every entry is a known piece type;
    - piece counts exactly match the official inventory.

    Placement legality is implicit in this representation because the 40 entries
    are bound to the 40 legal setup squares. :func:`validate_setup_placement`
    covers the square-oriented form where placement can go wrong.
    """
    if player not in PLAYERS:
        raise SetupError(f"unknown player: {player!r}")

    entries = tuple(setup)
    if len(entries) != PIECES_PER_PLAYER:
        raise SetupError(
            f"setup must contain exactly {PIECES_PER_PLAYER} pieces, got {len(entries)}"
        )

    for index, piece_type in enumerate(entries):
        if not isinstance(piece_type, int) or not 0 <= piece_type < len(PIECE_TYPE_NAMES):
            raise SetupError(f"setup slot {index} holds an unknown piece type: {piece_type!r}")

    counts = Counter(entries)
    for piece_type, required in PIECE_COUNTS.items():
        actual = counts.get(piece_type, 0)
        if actual != required:
            raise SetupError(
                f"inventory mismatch for {PIECE_TYPE_NAMES[piece_type]}: "
                f"expected {required}, got {actual}"
            )
    return entries


def validate_setup_placement(placements: "dict[int, int]", player: int) -> tuple[int, ...]:
    """Validate a `{square: piece_type}` placement map and convert it to a setup.

    This is the form a human interface or setup generator would submit, so it can
    violate placement rules that the row-major tuple form cannot express:

    - wrong number of placements;
    - a piece on a lake square;
    - a piece outside the player's setup area;
    - a setup square left empty;
    - duplicate occupancy (impossible in a dict, but checked for square validity).
    """
    if player not in PLAYERS:
        raise SetupError(f"unknown player: {player!r}")

    if len(placements) != PIECES_PER_PLAYER:
        raise SetupError(
            f"setup must place exactly {PIECES_PER_PLAYER} pieces, got {len(placements)}"
        )

    legal_squares = SETUP_SQUARE_SETS[player]
    for square in placements:
        if not isinstance(square, int) or not 0 <= square < BOARD_COLUMNS * 10:
            raise SetupError(f"invalid square index in setup: {square!r}")
        if square in LAKE_SQUARE_SET:
            raise SetupError(f"setup places a piece on lake square {square}")
        if square not in legal_squares:
            raise SetupError(
                f"square {square} is outside the {PLAYER_NAMES[player]} setup area"
            )

    missing = sorted(legal_squares - set(placements))
    if missing:
        raise SetupError(
            f"setup leaves {len(missing)} legal setup square(s) empty, "
            f"first missing square index {missing[0]}"
        )

    ordered = tuple(placements[square] for square in SETUP_SQUARES[player])
    return validate_setup(ordered, player)


def setup_to_placements(setup: "tuple[int, ...]", player: int) -> dict[int, int]:
    """Convert a row-major setup tuple to a `{square: piece_type}` map."""
    return dict(zip(SETUP_SQUARES[player], setup))


def random_setup(rng: random.Random, player: int = 0) -> tuple[int, ...]:
    """Generate a uniformly shuffled legal setup.

    The generator only guarantees legality; strategic quality is a Phase Seven
    concern. `player` is accepted for interface symmetry and does not change the
    distribution, because both players use the same inventory.
    """
    del player  # inventory is identical for both players
    pieces: list[int] = []
    for piece_type, count in sorted(PIECE_COUNTS.items()):
        pieces.extend([piece_type] * count)
    rng.shuffle(pieces)
    return tuple(pieces)


def reflect_setup(setup: "tuple[int, ...]") -> tuple[int, ...]:
    """Mirror a setup left-to-right within its four rows.

    Required by `03_game_engine_spec.md` section 7 so the later setup library can
    double its effective size for free.
    """
    entries = tuple(setup)
    if len(entries) != PIECES_PER_PLAYER:
        raise SetupError(f"expected {PIECES_PER_PLAYER} entries to reflect")
    reflected: list[int] = []
    for row_start in range(0, PIECES_PER_PLAYER, BOARD_COLUMNS):
        row = entries[row_start : row_start + BOARD_COLUMNS]
        reflected.extend(reversed(row))
    return tuple(reflected)


def serialize_setup(setup: "tuple[int, ...]") -> str:
    """Encode a setup as a 40-character string of single-character type codes."""
    return "".join(PIECE_TYPE_CODES[piece_type] for piece_type in setup)


def deserialize_setup(text: str) -> tuple[int, ...]:
    """Inverse of :func:`serialize_setup`."""
    if len(text) != PIECES_PER_PLAYER:
        raise SetupError(
            f"serialized setup must be {PIECES_PER_PLAYER} characters, got {len(text)}"
        )
    try:
        return tuple(PIECE_TYPE_BY_CODE[character] for character in text)
    except KeyError as error:  # pragma: no cover - defensive
        raise SetupError(f"unknown piece code {error.args[0]!r}") from error
