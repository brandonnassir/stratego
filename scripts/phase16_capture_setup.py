#!/usr/bin/env python
"""Capture one of the operator's setups into `operator_harvest`.

Accepts a 4x10 rank grid as text — four lines of ten tokens — validates it
through the imported Phase 15 orientation gate, and appends it to the
`operator_harvest` family of `data/phase16/phase16_adversarial_setups_v1.json`
(deduplicated by tuple; the authored families are never touched).

Grid format
-----------
One line per rank, ten tokens per line, in **your own orientation**:
by default line 1 is your BACK row (where flags usually live) and line 4 is
your FRONT row (nearest the lakes). Pass `--front-first` if you wrote the
front row first. Tokens are the repository's piece codes:

```text
1 = Marshal   2 = General   3 = Colonel   4 = Major     5 = Captain
6 = Lieutenant 7 = Sergeant 8 = Miner     9 = Scout     S = Spy
F = Flag      B = Bomb
```

(Lower case is accepted.) The inventory must be exact: 1xF, 6xB, 1x1, 1x2,
2x3, 3x4, 4x5, 4x6, 4x7, 5x8, 8x9, 1xS.

Examples:

    .venv/bin/python scripts/phase16_capture_setup.py --file my_setup.txt \
        --note "beat maximum_strength twice with a left-corner decoy"
    echo "F B 7 ... " | .venv/bin/python scripts/phase16_capture_setup.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.engine.constants import (  # noqa: E402
    PIECE_COUNTS,
    PIECE_TYPE_BY_CODE,
    PIECE_TYPE_CODES,
    PIECE_TYPE_NAMES,
)
from stratego.evaluation.phase16.adversarial import (  # noqa: E402
    DEFAULT_LIBRARY_PATH,
    append_harvest_setup,
    load_library,
    save_library,
    setup_properties,
    validate_setup,
)
from stratego.evaluation.phase16.contract import Phase16MeasurementError  # noqa: E402
from stratego.evaluation.phase16.operator_log import utc_now  # noqa: E402

RANKS, FILES = 4, 10


def parse_grid(text: str, *, front_first: bool = False) -> "tuple[int, ...]":
    """The canonical 40-tuple of a 4x10 rank grid."""
    lines = [line.split() for line in text.strip().splitlines() if line.split()]
    if len(lines) != RANKS:
        raise Phase16MeasurementError(
            f"expected {RANKS} non-empty lines of {FILES} tokens, got {len(lines)} lines"
        )
    for number, tokens in enumerate(lines, start=1):
        if len(tokens) != FILES:
            raise Phase16MeasurementError(
                f"line {number} has {len(tokens)} tokens, expected {FILES}"
            )
    if front_first:
        lines = lines[::-1]
    pieces = []
    for tokens in lines:
        for token in tokens:
            code = token.upper()
            if code not in PIECE_TYPE_BY_CODE:
                raise Phase16MeasurementError(
                    f"unknown piece code {token!r}; codes are {list(PIECE_TYPE_CODES)}"
                )
            pieces.append(PIECE_TYPE_BY_CODE[code])
    counts: dict[int, int] = {}
    for piece in pieces:
        counts[piece] = counts.get(piece, 0) + 1
    if counts != PIECE_COUNTS:
        problems = []
        for piece_type in sorted(set(counts) | set(PIECE_COUNTS)):
            have, want = counts.get(piece_type, 0), PIECE_COUNTS.get(piece_type, 0)
            if have != want:
                problems.append(
                    f"{PIECE_TYPE_NAMES[piece_type]} ({PIECE_TYPE_CODES[piece_type]}): "
                    f"{have} given, {want} required"
                )
        raise Phase16MeasurementError("inventory is wrong — " + "; ".join(problems))
    return tuple(pieces)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=None, help="grid file (default: stdin)")
    parser.add_argument(
        "--front-first", action="store_true", help="line 1 is the front row"
    )
    parser.add_argument("--note", default=None, help="context for the provenance record")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and report, do not write"
    )
    parser.add_argument("--library", default=None, help="library path override")
    arguments = parser.parse_args()

    if arguments.file:
        text = Path(arguments.file).read_text()
    else:
        if sys.stdin.isatty():
            print("paste the 4x10 grid (back row first), then EOF (ctrl-d):")
        text = sys.stdin.read()

    try:
        canonical = parse_grid(text, front_first=arguments.front_first)
        validate_setup(canonical)
    except Phase16MeasurementError as error:
        print(f"REFUSED: {error}")
        return 1

    facts = setup_properties(canonical)
    print(
        f"valid setup: flag at rank {facts['flag_rank']} file {facts['flag_file']}, "
        f"{facts['bombs_adjacent_to_flag']} adjacent bomb(s), marshal rank "
        f"{facts['marshal_rank']}"
    )
    if arguments.dry_run:
        print("dry run — nothing written")
        return 0

    library_path = arguments.library or DEFAULT_LIBRARY_PATH
    document = load_library(library_path, root=REPOSITORY_ROOT)
    entry = append_harvest_setup(
        document,
        canonical,
        provenance={
            "capture_tool": "scripts/phase16_capture_setup.py",
            "note": arguments.note,
            "front_first_input": bool(arguments.front_first),
        },
        captured_utc=utc_now(),
    )
    if entry is None:
        print("already harvested — identical setup is in operator_harvest; nothing written")
        return 0
    save_library(document, library_path, root=REPOSITORY_ROOT)
    print(
        f"appended {entry['setup_id']} (harvest revision "
        f"{document['harvest_revision']}, library now {document['setup_count']} setups)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
