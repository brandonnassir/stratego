"""Coordinate conversion, adjacency and perspective normalization.

Specification sources:

- `01_official_rules.md` section 1 (columns a-j, rows 1-10)
- `03_game_engine_spec.md` section 4 (internal row/column indices)
- `06_observation_v2_127ch.md` section 3 (perspective normalization)

Three coordinate spaces exist in the engine:

1. absolute square index `0..99` -- the authoritative internal space;
2. `(row, column)` with both in `0..9`;
3. human notation such as `a1` or `j10`.

On top of those, observations use a *normalized* view in which the observing
player's own setup rows always sit at the bottom of the board. Red is already
oriented that way, so red normalization is the identity and blue normalization
is a 180 degree rotation of the board.
"""

from .constants import (
    BLUE,
    BOARD_COLUMNS,
    BOARD_ROWS,
    LAKE_SQUARE_SET,
    NUM_SQUARES,
    PLAYERS,
    RED,
)

COLUMN_LETTERS = "abcdefghij"


def square_index(row: int, column: int) -> int:
    """Convert `(row, column)` to an absolute square index."""
    if not (0 <= row < BOARD_ROWS and 0 <= column < BOARD_COLUMNS):
        raise ValueError(f"coordinates out of range: row={row}, column={column}")
    return row * BOARD_COLUMNS + column


def square_row(square: int) -> int:
    """Row index of an absolute square index."""
    return square // BOARD_COLUMNS


def square_column(square: int) -> int:
    """Column index of an absolute square index."""
    return square % BOARD_COLUMNS


def square_row_column(square: int) -> tuple[int, int]:
    """`(row, column)` of an absolute square index."""
    if not 0 <= square < NUM_SQUARES:
        raise ValueError(f"square index out of range: {square}")
    return divmod(square, BOARD_COLUMNS)


def square_name(square: int) -> str:
    """Human-readable name such as `a1` (square 0) or `j10` (square 99)."""
    row, column = square_row_column(square)
    return f"{COLUMN_LETTERS[column]}{row + 1}"


def square_from_name(name: str) -> int:
    """Inverse of :func:`square_name`."""
    text = name.strip().lower()
    if len(text) < 2:
        raise ValueError(f"malformed square name: {name!r}")
    column_letter, row_text = text[0], text[1:]
    if column_letter not in COLUMN_LETTERS:
        raise ValueError(f"unknown column letter in {name!r}")
    if not row_text.isdigit():
        raise ValueError(f"malformed row number in {name!r}")
    row = int(row_text) - 1
    column = COLUMN_LETTERS.index(column_letter)
    return square_index(row, column)


def is_lake(square: int) -> bool:
    """Whether the square is one of the eight non-occupiable lake squares."""
    return square in LAKE_SQUARE_SET


def is_occupiable(square: int) -> bool:
    """Whether the square exists on the board and is not a lake."""
    return 0 <= square < NUM_SQUARES and square not in LAKE_SQUARE_SET


# Cardinal steps as (row delta, column delta): north, south, west, east.
# The order is fixed so that generated move lists are deterministic.
CARDINAL_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _build_neighbour_table() -> tuple[tuple[int, ...], ...]:
    """Precompute the occupiable cardinal neighbours of every square."""
    table: list[tuple[int, ...]] = []
    for square in range(NUM_SQUARES):
        row, column = divmod(square, BOARD_COLUMNS)
        neighbours = []
        for row_delta, column_delta in CARDINAL_STEPS:
            neighbour_row = row + row_delta
            neighbour_column = column + column_delta
            if not (
                0 <= neighbour_row < BOARD_ROWS and 0 <= neighbour_column < BOARD_COLUMNS
            ):
                continue
            neighbour = neighbour_row * BOARD_COLUMNS + neighbour_column
            if neighbour in LAKE_SQUARE_SET:
                continue
            neighbours.append(neighbour)
        table.append(tuple(sorted(neighbours)))
    return tuple(table)


# `NEIGHBOURS[square]` lists the occupiable orthogonal neighbours in ascending
# absolute index order. Lake squares have entries too, but nothing ever stands
# on a lake so those entries are never consulted.
NEIGHBOURS = _build_neighbour_table()


def are_adjacent(square_a: int, square_b: int) -> bool:
    """Whether two occupiable squares are orthogonally adjacent."""
    return square_b in NEIGHBOURS[square_a]


def _build_ray_table() -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Precompute, for each square, the four cardinal rays of occupiable squares.

    A ray stops at the board edge and at the first lake square, because a Scout
    may neither leave the board nor cross a lake. Occupancy blocking is applied
    at move-generation time.
    """
    table: list[tuple[tuple[int, ...], ...]] = []
    for square in range(NUM_SQUARES):
        row, column = divmod(square, BOARD_COLUMNS)
        rays: list[tuple[int, ...]] = []
        for row_delta, column_delta in CARDINAL_STEPS:
            ray: list[int] = []
            step_row, step_column = row + row_delta, column + column_delta
            while 0 <= step_row < BOARD_ROWS and 0 <= step_column < BOARD_COLUMNS:
                candidate = step_row * BOARD_COLUMNS + step_column
                if candidate in LAKE_SQUARE_SET:
                    break
                ray.append(candidate)
                step_row += row_delta
                step_column += column_delta
            rays.append(tuple(ray))
        table.append(tuple(rays))
    return tuple(table)


RAYS = _build_ray_table()


# ---------------------------------------------------------------------------
# Perspective normalization
# ---------------------------------------------------------------------------


def _build_perspective_table() -> dict[int, tuple[int, ...]]:
    """Map absolute square -> normalized square for each observing player.

    Red already occupies the low rows, so red's transform is the identity. Blue
    occupies the high rows, so the board is rotated 180 degrees for blue:
    `(row, column) -> (9 - row, 9 - column)`, i.e. `square -> 99 - square`.

    The 180 degree rotation is an involution and maps the lake mask onto itself,
    so the static lake plane is identical from both perspectives.
    """
    tables = {
        RED: tuple(range(NUM_SQUARES)),
        BLUE: tuple(NUM_SQUARES - 1 - square for square in range(NUM_SQUARES)),
    }
    for player in PLAYERS:
        table = tables[player]
        # Involution and lake preservation are properties the observation code
        # relies on, so verify them once at import time.
        assert all(table[table[square]] == square for square in range(NUM_SQUARES))
        assert all(
            (square in LAKE_SQUARE_SET) == (table[square] in LAKE_SQUARE_SET)
            for square in range(NUM_SQUARES)
        )
    return tables


PERSPECTIVE_TABLES = _build_perspective_table()


def to_perspective(square: int, observer: int) -> int:
    """Absolute square index -> observer-normalized square index."""
    return PERSPECTIVE_TABLES[observer][square]


def from_perspective(square: int, observer: int) -> int:
    """Observer-normalized square index -> absolute square index.

    Both transforms are involutions, so this is the same table lookup; it exists
    as a separate name because call sites read much more clearly with it.
    """
    return PERSPECTIVE_TABLES[observer][square]


def perspective_row_column(square: int, observer: int) -> tuple[int, int]:
    """`(row, column)` of a square after normalizing for `observer`."""
    return divmod(PERSPECTIVE_TABLES[observer][square], BOARD_COLUMNS)


def normalized_coordinate(index: int) -> float:
    """Normalize a row or column index to `[-1, +1]`.

    `06_observation_v2_127ch.md` section 6 defines `coord(i) = 2 * i / 9 - 1`.
    """
    return 2.0 * index / (BOARD_ROWS - 1) - 1.0
