"""Hand-constructed positive and negative fixtures for the 16 family contracts.

These exist to prove the Agent 1 contracts are executable: one legal, mobile,
predicate-satisfying arrangement per family, plus a minimal mutation per
family that breaks its contract. They are test fixtures only — the production
generator is Agent 2's deliverable and shares nothing with this module.

Construction
------------
Each fixture explicitly places every piece of every type any family clause
can reference — Flag, all 6 Bombs, all 8 Scouts, all 5 Miners, Marshal,
General, Spy, all 3 Majors, both Colonels (28 pieces) — and fills the 12
remaining cells with Sergeants, Lieutenants and Captains in a fixed order.
The filler types are movable and are referenced by no clause, so the fill can
never flip a family predicate.

Cells are `(rank, file)` in the canonical own-orientation frame (rank 0 =
back row, rank 3 = front row).
"""

from stratego.engine.constants import (
    BOMB,
    CAPTAIN,
    COLONEL,
    FLAG,
    GENERAL,
    LIEUTENANT,
    MARSHAL,
    MINER,
    MAJOR,
    PIECE_COUNTS,
    SCOUT,
    SERGEANT,
    SPY,
)
from stratego.setups.identity import CANONICAL_CELLS, CANONICAL_FILES

#: Piece multiset every fixture must place explicitly.
_EXPLICIT_COUNTS = {
    FLAG: 1,
    BOMB: 6,
    SCOUT: 8,
    MINER: 5,
    MARSHAL: 1,
    GENERAL: 1,
    SPY: 1,
    MAJOR: 3,
    COLONEL: 2,
}
assert all(PIECE_COUNTS[piece] == count for piece, count in _EXPLICIT_COUNTS.items())

#: Filler for the 12 unplaced cells, in fixed deterministic order.
_FILL_TYPES = [SERGEANT] * 4 + [LIEUTENANT] * 4 + [CAPTAIN] * 4
assert len(_FILL_TYPES) == CANONICAL_CELLS - sum(_EXPLICIT_COUNTS.values())


def _placements(**groups: "list[tuple[int, int]]") -> dict:
    """Expand `{type_name: [cells]}` keyword groups into `{cell: type}`."""
    type_by_name = {
        "flag": FLAG,
        "bombs": BOMB,
        "scouts": SCOUT,
        "miners": MINER,
        "marshal": MARSHAL,
        "general": GENERAL,
        "spy": SPY,
        "majors": MAJOR,
        "colonels": COLONEL,
    }
    placed: dict[tuple[int, int], int] = {}
    for name, cells in groups.items():
        piece_type = type_by_name[name]
        for cell in cells:
            if cell in placed:
                raise AssertionError(f"duplicate fixture cell: {cell}")
            placed[cell] = piece_type
    counts = {piece: 0 for piece in _EXPLICIT_COUNTS}
    for piece_type in placed.values():
        counts[piece_type] += 1
    if counts != _EXPLICIT_COUNTS:
        raise AssertionError(f"fixture piece counts wrong: {counts}")
    return placed


FIXTURE_PLACEMENTS: dict[str, dict] = {
    "F00": _placements(
        flag=[(0, 0)],
        bombs=[(0, 1), (1, 0), (1, 1), (0, 5), (1, 7), (2, 3)],
        scouts=[(3, 0), (3, 1), (3, 4), (3, 5), (3, 8), (3, 9), (2, 0), (2, 1)],
        miners=[(0, 2), (0, 3), (1, 2), (1, 3), (2, 2)],
        marshal=[(1, 4)],
        general=[(0, 4)],
        spy=[(0, 6)],
        majors=[(2, 4), (2, 5), (2, 6)],
        colonels=[(1, 5), (1, 6)],
    ),
    "F01": _placements(
        flag=[(0, 1)],
        bombs=[(0, 0), (0, 2), (1, 1), (1, 4), (2, 6), (0, 7)],
        scouts=[(3, 0), (3, 1), (3, 4), (3, 5), (3, 8), (2, 0), (2, 1), (2, 4)],
        miners=[(0, 3), (0, 4), (1, 2), (1, 3), (2, 2)],
        marshal=[(1, 0)],
        general=[(0, 5)],
        spy=[(0, 6)],
        majors=[(2, 3), (2, 5), (1, 5)],
        colonels=[(1, 6), (1, 7)],
    ),
    "F02": _placements(
        flag=[(0, 4)],
        bombs=[(0, 3), (0, 5), (1, 4), (1, 7), (2, 2), (0, 8)],
        scouts=[(3, 0), (3, 1), (3, 4), (3, 5), (3, 8), (3, 9), (2, 0), (2, 4)],
        miners=[(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)],
        marshal=[(1, 5)],
        general=[(1, 3)],
        spy=[(0, 6)],
        majors=[(2, 5), (2, 6), (1, 6)],
        colonels=[(1, 2), (0, 2)],
    ),
    "F03": _placements(
        flag=[(0, 2)],
        bombs=[(0, 1), (1, 6), (2, 7), (0, 8), (1, 8), (2, 5)],
        scouts=[(3, 0), (3, 1), (3, 4), (3, 5), (2, 0), (2, 1), (3, 8), (3, 9)],
        miners=[(0, 3), (1, 2), (1, 3), (0, 4), (2, 2)],
        marshal=[(1, 4)],
        general=[(1, 5)],
        spy=[(0, 0)],
        majors=[(2, 3), (2, 4), (1, 7)],
        colonels=[(0, 5), (0, 6)],
    ),
    "F04": _placements(
        flag=[(0, 3)],
        bombs=[(2, 0), (2, 7), (1, 7), (0, 8), (3, 2), (1, 0)],
        scouts=[(3, 0), (3, 1), (3, 4), (3, 5), (2, 1), (2, 4), (3, 8), (3, 9)],
        miners=[(0, 2), (0, 4), (1, 2), (1, 3), (1, 4)],
        marshal=[(0, 0)],
        general=[(0, 1)],
        spy=[(0, 5)],
        majors=[(2, 2), (2, 3), (2, 5)],
        colonels=[(1, 5), (1, 6)],
    ),
    "F05": _placements(
        flag=[(0, 1)],
        bombs=[(0, 7), (0, 9), (1, 8), (2, 3), (2, 6), (1, 4)],
        scouts=[(3, 0), (3, 1), (3, 4), (3, 5), (2, 0), (2, 1), (3, 8), (3, 9)],
        miners=[(0, 2), (0, 3), (1, 2), (1, 3), (2, 2)],
        marshal=[(1, 0)],
        general=[(1, 1)],
        spy=[(0, 0)],
        majors=[(2, 4), (2, 5), (1, 5)],
        colonels=[(0, 8), (1, 6)],
    ),
    "F06": _placements(
        flag=[(0, 0)],
        bombs=[(0, 2), (1, 4), (2, 6), (0, 6), (3, 8), (1, 0)],
        scouts=[(3, 0), (3, 1), (3, 4), (3, 5), (2, 0), (2, 1), (3, 9), (2, 4)],
        miners=[(0, 3), (0, 4), (1, 2), (1, 3), (2, 2)],
        marshal=[(1, 1)],
        general=[(0, 1)],
        spy=[(0, 5)],
        majors=[(2, 3), (2, 5), (1, 5)],
        colonels=[(1, 6), (1, 7)],
    ),
    "F07": _placements(
        flag=[(0, 4)],
        bombs=[(2, 2), (2, 6), (3, 3), (3, 6), (2, 7), (0, 3)],
        scouts=[(3, 0), (3, 1), (3, 4), (3, 5), (2, 0), (2, 1), (3, 8), (3, 9)],
        miners=[(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)],
        marshal=[(1, 4)],
        general=[(1, 5)],
        spy=[(0, 5)],
        majors=[(2, 3), (2, 5), (1, 3)],
        colonels=[(1, 6), (1, 7)],
    ),
    "F08": _placements(
        flag=[(0, 0)],
        bombs=[(0, 1), (1, 0), (1, 1), (0, 7), (1, 7), (2, 7)],
        scouts=[(3, 0), (3, 1), (3, 5), (3, 8), (3, 9), (2, 0), (2, 1), (2, 4)],
        miners=[(0, 2), (0, 3), (1, 2), (1, 3), (1, 4)],
        marshal=[(3, 4)],
        general=[(2, 5)],
        spy=[(0, 4)],
        majors=[(2, 3), (3, 7), (2, 6)],
        colonels=[(2, 2), (0, 5)],
    ),
    "F09": _placements(
        flag=[(0, 5)],
        bombs=[(0, 6), (1, 5), (2, 2), (0, 8), (1, 8), (2, 7)],
        scouts=[(3, 0), (3, 1), (3, 4), (3, 5), (3, 8), (3, 9), (2, 0), (2, 1)],
        miners=[(1, 0), (0, 0), (0, 1), (1, 6), (1, 7)],
        marshal=[(0, 4)],
        general=[(1, 4)],
        spy=[(0, 7)],
        majors=[(0, 3), (1, 3), (1, 2)],
        colonels=[(0, 2), (1, 1)],
    ),
    "F10": _placements(
        flag=[(0, 0)],
        bombs=[(0, 1), (1, 0), (1, 1), (2, 6), (0, 6), (1, 6)],
        scouts=[(3, 0), (3, 1), (3, 4), (2, 0), (2, 1), (2, 4), (0, 8), (1, 8)],
        miners=[(0, 2), (0, 3), (1, 2), (1, 3), (2, 2)],
        marshal=[(1, 4)],
        general=[(1, 5)],
        spy=[(0, 4)],
        majors=[(2, 3), (2, 5), (0, 5)],
        colonels=[(1, 7), (0, 7)],
    ),
    "F11": _placements(
        flag=[(0, 0)],
        bombs=[(0, 1), (1, 0), (1, 1), (2, 7), (1, 7), (0, 8)],
        scouts=[(0, 2), (0, 3), (0, 6), (1, 2), (1, 3), (2, 0), (2, 1), (1, 6)],
        miners=[(0, 4), (0, 5), (1, 4), (1, 5), (2, 2)],
        marshal=[(2, 4)],
        general=[(2, 5)],
        spy=[(0, 7)],
        majors=[(2, 3), (2, 6), (1, 8)],
        colonels=[(0, 9), (1, 9)],
    ),
    "F12": _placements(
        flag=[(0, 3)],
        bombs=[(0, 2), (0, 4), (1, 2), (1, 6), (2, 6), (0, 8)],
        scouts=[(3, 0), (3, 1), (3, 5), (3, 8), (3, 9), (2, 0), (2, 1), (2, 4)],
        miners=[(2, 2), (2, 5), (3, 4), (1, 3), (2, 7)],
        marshal=[(1, 4)],
        general=[(1, 5)],
        spy=[(0, 5)],
        majors=[(2, 3), (1, 7), (0, 6)],
        colonels=[(1, 0), (1, 1)],
    ),
    "F13": _placements(
        flag=[(0, 0)],
        bombs=[(0, 1), (1, 0), (1, 1), (2, 6), (1, 6), (0, 6)],
        scouts=[(3, 0), (3, 1), (3, 4), (3, 5), (3, 8), (3, 9), (2, 0), (2, 1)],
        miners=[(0, 2), (0, 3), (1, 2), (1, 3), (2, 2)],
        marshal=[(1, 4)],
        general=[(1, 5)],
        spy=[(0, 4)],
        majors=[(2, 3), (2, 4), (2, 5)],
        colonels=[(0, 5), (1, 7)],
    ),
    "F14": _placements(
        flag=[(0, 2)],
        bombs=[(0, 1), (0, 3), (1, 2), (2, 7), (1, 7), (3, 6)],
        scouts=[(2, 0), (2, 1), (3, 0), (3, 1), (3, 4), (0, 8), (1, 8), (0, 6)],
        miners=[(0, 0), (1, 0), (1, 1), (2, 2), (2, 5)],
        marshal=[(1, 5)],
        general=[(2, 4)],
        spy=[(0, 5)],
        majors=[(2, 3), (2, 6), (1, 4)],
        colonels=[(1, 6), (0, 7)],
    ),
    "F15": _placements(
        flag=[(2, 4)],
        bombs=[(3, 3), (3, 6), (3, 7), (0, 0), (0, 9), (1, 5)],
        scouts=[(0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 8), (1, 8), (0, 7)],
        miners=[(0, 3), (0, 4), (1, 2), (1, 3), (2, 2)],
        marshal=[(0, 5)],
        general=[(0, 6)],
        spy=[(1, 6)],
        majors=[(2, 6), (2, 7), (1, 7)],
        colonels=[(0, 8), (2, 1)],
    ),
}

#: Cell swaps applied to the completed positive fixture to produce a setup
#: that violates the named family's contract. Each entry is a list of
#: `((rank, file), (rank, file))` swaps applied in order.
NEGATIVE_MUTATIONS: dict[str, list] = {
    "F00": [((0, 0), (0, 4))],  # Flag leaves the corner
    "F01": [((0, 1), (0, 4))],  # Flag moves to the centre
    "F02": [((0, 4), (0, 0))],  # Flag moves to the corner
    "F03": [((0, 1), (0, 5))],  # the single guard Bomb walks away
    "F04": [((0, 2), (2, 0))],  # a Bomb arrives beside the Flag
    "F05": [((0, 0), (2, 3)), ((0, 2), (2, 6))],  # Flag gains a two-Bomb wall
    "F06": [((0, 1), (0, 2))],  # defense concentrates on the Flag
    "F07": [((2, 2), (0, 0)), ((2, 6), (0, 1))],  # Bombs retreat to the back
    "F08": [((3, 4), (0, 2))],  # the Marshal retreats
    "F09": [((0, 4), (3, 4))],  # the Marshal advances
    "F10": [((3, 0), (1, 2)), ((3, 1), (1, 3))],  # Scouts leave the front
    "F11": [((0, 2), (3, 0))],  # a Scout appears on the front rank
    "F12": [((2, 2), (0, 6)), ((2, 5), (0, 5))],  # Miners retreat
    "F13": [((0, 2), (3, 0))],  # a Miner appears on the front rank
    "F14": [((0, 2), (2, 2))],  # the Flag leaves the back rank
    "F15": [((2, 4), (0, 0)), ((0, 1), (0, 9)), ((1, 0), (3, 3))],  # conventional fortress forms
}


def build_fixture(family_id: str) -> tuple[int, ...]:
    """The complete 40-cell positive fixture for `family_id`."""
    placements = FIXTURE_PLACEMENTS[family_id]
    cells: list[int | None] = [None] * CANONICAL_CELLS
    for (rank, file), piece_type in placements.items():
        cells[rank * CANONICAL_FILES + file] = piece_type
    fill = iter(_FILL_TYPES)
    for index in range(CANONICAL_CELLS):
        if cells[index] is None:
            cells[index] = next(fill)
    return tuple(cells)  # type: ignore[arg-type]


def build_negative_fixture(family_id: str) -> tuple[int, ...]:
    """The mutated fixture that must violate `family_id`'s contract."""
    cells = list(build_fixture(family_id))
    for (rank_a, file_a), (rank_b, file_b) in NEGATIVE_MUTATIONS[family_id]:
        index_a = rank_a * CANONICAL_FILES + file_a
        index_b = rank_b * CANONICAL_FILES + file_b
        cells[index_a], cells[index_b] = cells[index_b], cells[index_a]
    return tuple(cells)
