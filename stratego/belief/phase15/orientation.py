"""Phase 15 Agent 1 section 4: the orientation gate.

Specification source: `01_AGENT_1_BELIEF_HEAD_TRAINING.md` section 4
("Critical orientation correction").

The defect this module exists to make impossible
-------------------------------------------------
`stratego/belief/phase11b/corpus.py`'s `Phase11BSetupSources.draw` returns a
*canonical own-orientation* 40-tuple, and the old glue passed that tuple
straight to `create_game()` for BLUE. Canonical rank 0 is a player's own
back rank; Blue's engine setup order runs front-to-back, so an unoriented
Blue army is placed reversed — which is why Phase 12 saw a flag on the
front row of 47 of 64 boards. Every downstream artifact built on that glue
(the Phase 11B corpus, the Phase 12 match packs) is contaminated, and none
of it may be reused here.

The rule, stated once
---------------------
```text
Red engine row  = canonical rank
Blue engine row = 9 - canonical rank
```

Red's map is the identity because `SETUP_SQUARES[RED]` is rows 0-3 in
row-major order and canonical rank 0 is Red's back rank, which is engine
row 0. Blue's setup squares are rows 6-9, so canonical rank `r` — `r` rows
forward of Blue's back rank, which is engine row 9 — lands on engine row
`9 - r`. That is exactly what the accepted `orient_setup` computes, and
:func:`assert_engine_orientation` re-derives it from the engine's own
`SETUP_SQUARES` rather than from `orient_setup`, so the check is
independent of the helper it is checking.

Not documentary — mechanical
----------------------------
:func:`assert_engine_orientation` is a hard runtime assertion on the
production path: every Phase 15 board passes through it before
`create_game`. :func:`orientation_gate` is the bulk proof section 4
requires, and it includes the negative canary — a Blue canonical tuple
handed over directly, which the assertion must reject.
"""

from __future__ import annotations

import time

from ...engine.constants import (
    BLUE,
    BOARD_COLUMNS,
    FLAG,
    PIECE_COUNTS,
    PIECES_PER_PLAYER,
    PLAYERS,
    RED,
    SETUP_SQUARES,
)
from ...setups.identity import (
    CANONICAL_CELLS,
    CANONICAL_RANKS,
    canonical_rank_file,
    orient_setup,
)
from .contract import Phase15Error

#: The gate identity a manifest records.
ORIENTATION_RULE_VERSION = "phase15_orientation_rule_v1"

#: The rule, as the string a report and a manifest carry verbatim.
ORIENTATION_RULE = "red engine row == canonical rank; blue engine row == 9 - canonical rank"

#: The accepted helpers a Phase 15 setup is allowed to have come through.
ACCEPTED_ORIENTATION_PATHS = (
    "SelectorDraw.oriented(player)",
    "SampledSetup.oriented(player)",
    "stratego.setups.identity.orient_setup(canonical, player)",
)


class Phase15OrientationError(Phase15Error):
    """A setup reached the engine in the wrong orientation."""


def expected_engine_row(canonical_rank: int, player: int) -> int:
    """The engine board row canonical rank `canonical_rank` must land on."""
    if player not in PLAYERS:
        raise Phase15OrientationError(f"unknown player: {player!r}")
    if not 0 <= int(canonical_rank) < CANONICAL_RANKS:
        raise Phase15OrientationError(f"canonical rank out of range: {canonical_rank!r}")
    rank = int(canonical_rank)
    return rank if player == RED else (BOARD_COLUMNS - 1) - rank


def assert_engine_orientation(
    canonical: "tuple[int, ...]", engine_setup: "tuple[int, ...]", player: int
) -> None:
    """Refuse an engine setup that is not `canonical` oriented for `player`.

    Re-derives the placement from the engine's own `SETUP_SQUARES` rather
    than from `orient_setup`, so a defect in the helper cannot hide behind
    a check written in terms of it. Every piece of the canonical tuple must
    appear on the board row and column this player's orientation sends it
    to.
    """
    canonical = tuple(canonical)
    engine_setup = tuple(engine_setup)
    if player not in PLAYERS:
        raise Phase15OrientationError(f"unknown player: {player!r}")
    if len(canonical) != CANONICAL_CELLS or len(engine_setup) != PIECES_PER_PLAYER:
        raise Phase15OrientationError(
            f"expected two {CANONICAL_CELLS}-entry setups, got "
            f"{len(canonical)} canonical and {len(engine_setup)} engine"
        )
    squares = SETUP_SQUARES[player]
    for index, piece_type in enumerate(canonical):
        rank, file = canonical_rank_file(index)
        row = expected_engine_row(rank, player)
        square = row * BOARD_COLUMNS + file
        try:
            slot = squares.index(square)
        except ValueError:  # pragma: no cover - setup rows are contiguous
            raise Phase15OrientationError(
                f"square {square} is not a setup square of player {player}"
            ) from None
        if engine_setup[slot] != piece_type:
            raise Phase15OrientationError(
                f"player {player}: canonical rank {rank} file {file} holds piece "
                f"{piece_type}, but engine row {row} column {file} holds "
                f"{engine_setup[slot]} — the setup did not pass through an "
                f"accepted orientation path ({', '.join(ACCEPTED_ORIENTATION_PATHS)})"
            )


def oriented_for(canonical: "tuple[int, ...]", player: int) -> "tuple[int, ...]":
    """The engine-ready setup for `player`, checked before it is returned.

    The single production entry point. A Phase 15 board is built from two
    calls to this function and nothing else, so no code path exists on
    which a canonical tuple can reach `create_game`.
    """
    engine_setup = orient_setup(tuple(canonical), player)
    assert_engine_orientation(canonical, engine_setup, player)
    return engine_setup


# ---------------------------------------------------------------------------
# The bulk proof
# ---------------------------------------------------------------------------


def _flag_report(engine_setup: "tuple[int, ...]", player: int) -> dict:
    squares = SETUP_SQUARES[player]
    slot = engine_setup.index(FLAG)
    square = squares[slot]
    row, column = divmod(square, BOARD_COLUMNS)
    return {"square": int(square), "row": int(row), "column": int(column)}


def check_board(
    red_canonical: "tuple[int, ...]", blue_canonical: "tuple[int, ...]"
) -> dict:
    """Every section 4 check on one paired board.

    Flag location, legal setup rows, complete inventory, and the Red/Blue
    paired orientation. Returns the observed facts; raises on any failure.
    """
    findings = {}
    for player, canonical in ((RED, red_canonical), (BLUE, blue_canonical)):
        engine_setup = oriented_for(canonical, player)

        counts: dict[int, int] = {}
        for piece_type in engine_setup:
            counts[piece_type] = counts.get(piece_type, 0) + 1
        if counts != PIECE_COUNTS:
            raise Phase15OrientationError(
                f"player {player}: inventory {counts} != the official {PIECE_COUNTS}"
            )

        squares = SETUP_SQUARES[player]
        rows = {square // BOARD_COLUMNS for square in squares}
        legal_rows = {0, 1, 2, 3} if player == RED else {6, 7, 8, 9}
        if rows != legal_rows:
            raise Phase15OrientationError(  # pragma: no cover - engine invariant
                f"player {player}: setup rows {sorted(rows)} != {sorted(legal_rows)}"
            )

        flag = _flag_report(engine_setup, player)
        canonical_flag_rank = canonical_rank_file(canonical.index(FLAG))[0]
        if flag["row"] != expected_engine_row(canonical_flag_rank, player):
            raise Phase15OrientationError(  # pragma: no cover - assert_ catches first
                f"player {player}: flag on engine row {flag['row']}, expected "
                f"{expected_engine_row(canonical_flag_rank, player)}"
            )
        findings[player] = {
            "flag": flag,
            "canonical_flag_rank": int(canonical_flag_rank),
            "inventory_exact": True,
        }

    # The paired statement: two armies built from the *same* canonical tuple
    # must be mirror images of each other, row for row.
    mirrored = all(
        oriented_for(red_canonical, RED)[index]
        == oriented_for(red_canonical, BLUE)[
            SETUP_SQUARES[BLUE].index(
                expected_engine_row(canonical_rank_file(index)[0], BLUE) * BOARD_COLUMNS
                + canonical_rank_file(index)[1]
            )
        ]
        for index in range(CANONICAL_CELLS)
    )
    if not mirrored:  # pragma: no cover - implied by assert_engine_orientation
        raise Phase15OrientationError("red and blue orientations are not paired mirrors")
    return {
        "red": findings[RED],
        "blue": findings[BLUE],
        "paired_mirror": True,
    }


def negative_canary() -> dict:
    """Prove the gate catches the exact Phase 11B mistake.

    Hands `assert_engine_orientation` a Blue canonical tuple *as if it were*
    the engine setup — the old glue's behaviour — and requires a refusal.
    A canonical tuple whose four rank rows happen to be palindromic would be
    its own orientation, so the canary uses a setup that is provably not.
    """
    from ...setups.sampler import load_library_index

    index = load_library_index()
    for entry in index.entries:
        canonical = tuple(entry.canonical_setup)
        if orient_setup(canonical, BLUE) == canonical:
            continue  # a self-oriented board proves nothing
        try:
            assert_engine_orientation(canonical, canonical, BLUE)
        except Phase15OrientationError as error:
            return {
                "canary": "blue_canonical_passed_directly",
                "detected": True,
                "base_setup_id": entry.base_setup_id,
                "message": str(error).split(" — ")[0],
            }
        raise Phase15OrientationError(  # pragma: no cover - the gate must catch this
            f"the gate accepted a raw Blue canonical tuple ({entry.base_setup_id}); "
            "the Phase 11B defect would pass undetected"
        )
    raise Phase15OrientationError(  # pragma: no cover - library is not all palindromes
        "no library setup differs from its own Blue orientation"
    )


def orientation_gate(boards: int = 4096, *, split: str = "train") -> dict:
    """The section 4 gate. No corpus generation may begin until this passes.

    Draws `boards` paired setups through the accepted sampler, runs every
    check on each, and finishes with the negative canary. Returns the
    evidence block a manifest and a report carry.
    """
    from ...setups.sampler import load_library_index, sample_setup

    from .seeds import orientation_seed

    if int(boards) < 1:
        raise Phase15OrientationError(f"boards must be positive, got {boards!r}")
    index = load_library_index()
    started = time.perf_counter()
    flag_rows = {RED: {}, BLUE: {}}
    front_row_flags = 0
    for ordinal in range(int(boards)):
        red = tuple(
            sample_setup(split, orientation_seed("red", ordinal), index=index).canonical
        )
        blue = tuple(
            sample_setup(split, orientation_seed("blue", ordinal), index=index).canonical
        )
        report = check_board(red, blue)
        for player, key in ((RED, "red"), (BLUE, "blue")):
            row = report[key]["flag"]["row"]
            flag_rows[player][row] = flag_rows[player].get(row, 0) + 1
            # "Front row" is the row nearest the opponent: 3 for Red, 6 for
            # Blue. Under the defect this was 47/64; a correct corpus shows
            # a flag there only when the *canonical* setup put it on the
            # front canonical rank, which the library rarely does.
            if row == (3 if player == RED else 6):
                front_row_flags += 1

    canary = negative_canary()
    # What the Phase 11B glue would have produced from the *same* draws: the
    # raw canonical tuple used as Blue's engine setup puts canonical rank r on
    # engine row 6 + r, so every board whose flag sits on Blue's own back rank
    # shows a front-row flag. Reporting it turns "we fixed the orientation"
    # into a number a reader can compare against Phase 12's 47/64.
    defect_front_row = sum(
        count for row, count in flag_rows[BLUE].items() if row == BOARD_COLUMNS - 1
    )
    return {
        "orientation_rule_version": ORIENTATION_RULE_VERSION,
        "orientation_rule": ORIENTATION_RULE,
        "accepted_paths": list(ACCEPTED_ORIENTATION_PATHS),
        "library_split": split,
        "library_content_digest": index.content_digest,
        "boards_checked": int(boards),
        "armies_checked": int(boards) * 2,
        "flag_row_histogram": {
            "red": {str(row): count for row, count in sorted(flag_rows[RED].items())},
            "blue": {str(row): count for row, count in sorted(flag_rows[BLUE].items())},
        },
        "front_row_flags": int(front_row_flags),
        "front_row_flag_rate": front_row_flags / (int(boards) * 2),
        "defect_counterfactual": {
            "description": (
                "blue front-row flags the Phase 11B glue would have produced from "
                "these same draws, by passing the canonical tuple to create_game"
            ),
            "blue_front_row_flags": int(defect_front_row),
            "blue_boards": int(boards),
            "rate": defect_front_row / int(boards),
            "observed_blue_front_row_flags": int(
                flag_rows[BLUE].get(BOARD_COLUMNS - 4, 0)
            ),
        },
        "inventory_exact": True,
        "legal_setup_rows": True,
        "paired_orientation": True,
        "negative_canary": canary,
        "passed": True,
        "seconds": round(time.perf_counter() - started, 3),
    }


__all__ = [
    "ACCEPTED_ORIENTATION_PATHS",
    "ORIENTATION_RULE",
    "ORIENTATION_RULE_VERSION",
    "Phase15OrientationError",
    "assert_engine_orientation",
    "check_board",
    "expected_engine_row",
    "negative_canary",
    "orientation_gate",
    "oriented_for",
]
