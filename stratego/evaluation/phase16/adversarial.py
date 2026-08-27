"""Phase 16 Agent 1 section 5: `phase16_adversarial_setups_v1`.

The pack that models the operator: human-realistic and operator-exploit
setups, authored from documented human conventions and the Ataraxos paper's
setup analysis (bombed-in corner flags dominate human play; the operator
beats the system with setups *outside* the training/search distribution).

Format
------
The accepted setup-library representation: **canonical own-orientation
40-tuples** (rank 0 = own back row, index = rank*10 + file) plus family
metadata. Every entry passes the imported Phase 15 section-4 gate
(`check_board`): flag row, legal rows, exact inventory, paired-mirror on the
oriented output. Nothing in this module invents an orientation path — engine
tuples are produced only by the imported `oriented_for`.

Families
--------
```text
operator_harvest      the operator's own winning setups — present but EMPTY
                      until captured (scripts/phase16_capture_setup.py or the
                      operator-log harvester); never authored here
bombed_corner_flag    flag in a back corner, bombed in (the classic)
bombed_center_flag    flag mid-back-rank behind a bomb shell
scout_screen          front rank heavy with scouts, high pieces ranks 1-2
aggressive_marshal    marshal (and general) at or near the front
spy_shadow            spy shadowing the centre lanes; high-bomb traps
miner_wall            miners spread wide on rank 2, anti-bomb posture
decoy_flag_structure  bomb-ringed decoy corner opposite a lightly-kept flag
free_novelty          convention-breaking structures a bot would not expect
```

Each family is internally varied: 12 authored setups per family, every one
deterministic from `derive_measure_seed(DOMAIN_ADVERSARIAL, family, ordinal)`
and deduplicated across the whole library.

Two digests
-----------
`authored_digest` covers the eight authored families and never changes once
frozen. `library_digest` covers everything including `operator_harvest`, so
a harvest append is a visible revision (`harvest_revision` increments), not
a silent edit of a frozen instrument.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from ...engine.constants import (
    BLUE,
    BOMB,
    CAPTAIN,
    COLONEL,
    FLAG,
    GENERAL,
    LIEUTENANT,
    MAJOR,
    MARSHAL,
    MINER,
    PIECE_COUNTS,
    RED,
    SCOUT,
    SERGEANT,
    SPY,
)
from ...setups.identity import (
    CANONICAL_CELLS,
    CANONICAL_FILES,
    CANONICAL_RANKS,
    canonical_index,
    canonical_neighbours,
    canonical_rank_file,
)
from ...belief.phase15.orientation import (
    ORIENTATION_RULE,
    ORIENTATION_RULE_VERSION,
    check_board,
    oriented_for,
)
from .contract import (
    ADVERSARIAL_VERSION,
    AUTHORED_FAMILIES,
    ADVERSARIAL_FAMILIES,
    DOMAIN_ADVERSARIAL,
    FAMILY_OPERATOR_HARVEST,
    Phase16MeasurementError,
    SETUPS_PER_FAMILY,
    derive_measure_seed,
)

#: Where the library lives.
DEFAULT_LIBRARY_PATH = Path("data/phase16/phase16_adversarial_setups_v1.json")

#: The authoring identity: any change to a template is a new version.
AUTHORING_VERSION = "phase16_adversarial_authoring_v1"

FAMILY_DESCRIPTIONS = {
    FAMILY_OPERATOR_HARVEST: (
        "the operator's own winning setups, captured via "
        "scripts/phase16_capture_setup.py or harvested from the operator game "
        "log; never authored"
    ),
    "bombed_corner_flag": (
        "flag in a back corner (canonical rank 0, file 0 or 9) with every "
        "orthogonal approach bombed; the dominant human convention (~2/3 of "
        "Ataraxos-analysed setups)"
    ),
    "bombed_center_flag": (
        "flag mid-back-rank (files 3-6) behind a complete bomb shell"
    ),
    "scout_screen": (
        "front rank heavy with scouts for early information; marshal, general "
        "and colonels held on ranks 1-2"
    ),
    "aggressive_marshal": (
        "marshal at the front rank with the general close behind; immediate "
        "pressure"
    ),
    "spy_shadow": (
        "spy stationed on the central attack lanes at ranks 2-3, with a "
        "high-piece-plus-bomb trap to bait captures"
    ),
    "miner_wall": (
        "all five miners spread wide across rank 2 behind a scout screen; "
        "anti-bomb posture"
    ),
    "decoy_flag_structure": (
        "a bomb-ringed decoy corner holding a low piece, opposite a true flag "
        "kept deliberately light"
    ),
    "free_novelty": (
        "convention breakers: mid-board flags, front-rank bombs, wing-massed "
        "bombs, back-rank scouts — structures outside every accepted-library "
        "family"
    ),
}


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------


class _Builder:
    """A partially placed canonical 40-tuple with inventory accounting."""

    def __init__(self) -> None:
        self.cells: "list[int | None]" = [None] * CANONICAL_CELLS
        self.remaining = dict(PIECE_COUNTS)

    def empty(self, candidates=None) -> "list[int]":
        pool = range(CANONICAL_CELLS) if candidates is None else candidates
        return [cell for cell in pool if self.cells[cell] is None]

    def place(self, rank: int, file: int, piece: int) -> int:
        index = canonical_index(rank, file)
        if self.cells[index] is not None:
            raise Phase16MeasurementError(
                f"cell (rank {rank}, file {file}) is already occupied"
            )
        if self.remaining.get(piece, 0) <= 0:
            raise Phase16MeasurementError(f"no piece of type {piece} left to place")
        self.cells[index] = piece
        self.remaining[piece] -= 1
        return index

    def place_random(self, rng: random.Random, piece: int, candidates) -> int:
        pool = self.empty(candidates)
        if not pool:
            pool = self.empty()
        if not pool:
            raise Phase16MeasurementError("no empty cell left")
        if self.remaining.get(piece, 0) <= 0:
            raise Phase16MeasurementError(f"no piece of type {piece} left to place")
        cell = pool[rng.randrange(len(pool))]
        self.cells[cell] = piece
        self.remaining[piece] -= 1
        return cell

    def finish(self) -> "tuple[int, ...]":
        if any(count for count in self.remaining.values()):
            raise Phase16MeasurementError(
                f"inventory not exhausted: {self.remaining}"
            )
        if any(cell is None for cell in self.cells):
            raise Phase16MeasurementError("not every canonical cell was filled")
        return tuple(self.cells)  # type: ignore[arg-type]


def _rank_cells(*ranks: int) -> "list[int]":
    return [
        canonical_index(rank, file) for rank in ranks for file in range(CANONICAL_FILES)
    ]


#: Default placement preferences of the fill tail, piece-major. Ranks are
#: tried in order; a piece falls back to any empty cell. The flag is placed
#: FIRST: a template that leaves it to the fill still gets a back-rank flag,
#: never a front-rank accident after the bombs have filled rank 0.
_FILL_ORDER = (
    (FLAG, (0,)),
    (BOMB, (0, 1)),
    (MINER, (1, 2)),
    (MARSHAL, (2, 3)),
    (GENERAL, (2, 3)),
    (COLONEL, (2, 3)),
    (SPY, (1, 2)),
    (SCOUT, (3, 2)),
    (MAJOR, (1, 2, 3)),
    (CAPTAIN, (2, 3)),
    (LIEUTENANT, (1, 2, 3)),
    (SERGEANT, (1, 2, 3)),
)


def _fill_rest(builder: _Builder, rng: random.Random, overrides: "dict | None" = None) -> None:
    """Place every remaining piece with human-plausible rank preferences."""
    preferences = dict(_FILL_ORDER)
    if overrides:
        preferences.update(overrides)
    for piece, ranks in preferences.items():
        while builder.remaining.get(piece, 0) > 0:
            placed = False
            for rank in ranks:
                pool = builder.empty(_rank_cells(rank))
                if pool:
                    builder.place_random(rng, piece, pool)
                    placed = True
                    break
            if not placed:
                builder.place_random(rng, piece, None)


# ---------------------------------------------------------------------------
# Family templates
# ---------------------------------------------------------------------------


def _bombed_corner_flag(rng: random.Random, ordinal: int) -> _Builder:
    builder = _Builder()
    corner_file = 0 if ordinal % 2 == 0 else CANONICAL_FILES - 1
    inner = 1 if corner_file == 0 else CANONICAL_FILES - 2
    builder.place(0, corner_file, FLAG)
    builder.place(0, inner, BOMB)
    builder.place(1, corner_file, BOMB)
    if rng.random() < 0.5:
        builder.place(1, inner, BOMB)  # the diagonal third bomb of the full seal
    _fill_rest(builder, rng)
    return builder


def _bombed_center_flag(rng: random.Random, ordinal: int) -> _Builder:
    builder = _Builder()
    flag_file = 3 + (ordinal % 4)  # files 3-6
    builder.place(0, flag_file, FLAG)
    builder.place(0, flag_file - 1, BOMB)
    builder.place(0, flag_file + 1, BOMB)
    builder.place(1, flag_file, BOMB)
    if rng.random() < 0.5:
        builder.place(1, flag_file - 1, BOMB)
    _fill_rest(builder, rng)
    return builder


def _scout_screen(rng: random.Random, ordinal: int) -> _Builder:
    builder = _Builder()
    front_scouts = 6 + (ordinal % 3)  # 6, 7 or 8 scouts on the front rank
    files = list(range(CANONICAL_FILES))
    rng.shuffle(files)
    for file in files[:front_scouts]:
        builder.place(3, file, SCOUT)
    builder.place_random(rng, MARSHAL, _rank_cells(1, 2))
    builder.place_random(rng, GENERAL, _rank_cells(1, 2))
    builder.place_random(rng, COLONEL, _rank_cells(1, 2))
    builder.place_random(rng, COLONEL, _rank_cells(1, 2))
    _fill_rest(builder, rng)
    return builder


def _aggressive_marshal(rng: random.Random, ordinal: int) -> _Builder:
    builder = _Builder()
    marshal_file = 2 + (ordinal % 6)  # files 2-7, the attack lanes
    builder.place(3, marshal_file, MARSHAL)
    general_file = marshal_file + (1 if marshal_file < CANONICAL_FILES - 1 else -1)
    builder.place(2 if rng.random() < 0.5 else 3, general_file, GENERAL)
    builder.place_random(rng, COLONEL, _rank_cells(2, 3))
    builder.place_random(rng, COLONEL, _rank_cells(2, 3))
    _fill_rest(builder, rng)
    return builder


def _spy_shadow(rng: random.Random, ordinal: int) -> _Builder:
    builder = _Builder()
    lane = 3 + (ordinal % 4)  # central files 3-6
    spy_rank = 2 + (ordinal % 2)  # rank 2 or 3
    spy_cell = builder.place(spy_rank, lane, SPY)
    # The general stands next to the spy so the spy shadows the square the
    # enemy marshal must come through to take it.
    neighbours = [cell for cell in canonical_neighbours(spy_cell)]
    rng.shuffle(neighbours)
    for cell in neighbours:
        if builder.cells[cell] is None:
            rank, file = canonical_rank_file(cell)
            builder.place(rank, file, GENERAL)
            break
    # A high-bomb trap: a colonel with two bombs beside it, away from the lane.
    trap_file = 0 if lane >= 5 else CANONICAL_FILES - 1
    builder.place(2, trap_file, COLONEL)
    builder.place(1, trap_file, BOMB)
    inner = trap_file + (1 if trap_file == 0 else -1)
    builder.place(2, inner, BOMB)
    _fill_rest(builder, rng)
    return builder


def _miner_wall(rng: random.Random, ordinal: int) -> _Builder:
    builder = _Builder()
    spreads = (
        (0, 2, 4, 6, 9),
        (0, 2, 5, 7, 9),
        (0, 3, 5, 7, 9),
        (1, 3, 5, 7, 9),
        (0, 2, 4, 7, 9),
        (0, 3, 4, 6, 9),
    )
    for file in spreads[ordinal % len(spreads)]:
        builder.place(2, file, MINER)
    _fill_rest(builder, rng)
    return builder


def _decoy_flag_structure(rng: random.Random, ordinal: int) -> _Builder:
    builder = _Builder()
    true_file = 0 if ordinal % 2 == 0 else CANONICAL_FILES - 1
    decoy_file = CANONICAL_FILES - 1 - true_file
    # The true flag: back corner, deliberately light — no adjacent bomb, one
    # sergeant in front of it.
    builder.place(0, true_file, FLAG)
    builder.place(1, true_file, SERGEANT)
    # The decoy: the classic corner fortress, holding a scout instead.
    decoy_inner = 1 if decoy_file == 0 else CANONICAL_FILES - 2
    builder.place(0, decoy_file, SCOUT)
    builder.place(0, decoy_inner, BOMB)
    builder.place(1, decoy_file, BOMB)
    if rng.random() < 0.6:
        builder.place(1, decoy_inner, BOMB)
    # Keep the remaining bombs away from the true flag so the structure reads
    # as the decoy corner being the defended one.
    _fill_rest(builder, rng, overrides={BOMB: (1, 2)})
    return builder


def _free_novelty(rng: random.Random, ordinal: int) -> _Builder:
    builder = _Builder()
    pattern = ordinal % 4
    if pattern == 0:
        # Mid-board flag, unguarded; bombs pushed to the front rank.
        builder.place(2, rng.randrange(CANONICAL_FILES), FLAG)
        for _ in range(4):
            builder.place_random(rng, BOMB, _rank_cells(3))
        _fill_rest(builder, rng)
    elif pattern == 1:
        # Marshal sits in the flag corner; scouts hide on the back rank.
        corner = 0 if rng.random() < 0.5 else CANONICAL_FILES - 1
        builder.place(0, corner, MARSHAL)
        builder.place(1, 4 + rng.randrange(2), FLAG)
        for _ in range(5):
            builder.place_random(rng, SCOUT, _rank_cells(0))
        _fill_rest(builder, rng, overrides={SCOUT: (0, 1)})
    elif pattern == 2:
        # Every bomb massed on one wing; the flag on the other, rank 1.
        wing = rng.random() < 0.5
        wing_files = (0, 1, 2, 3) if wing else (6, 7, 8, 9)
        cells = [
            canonical_index(rank, file) for rank in (0, 1, 2) for file in wing_files
        ]
        for _ in range(6):
            builder.place_random(rng, BOMB, cells)
        flag_file = 8 if wing else 1
        builder.place(1, flag_file, FLAG)
        _fill_rest(builder, rng)
    else:
        # A diagonal bomb chain; the spy leads from the front rank and the
        # flag hides mid-board inside the chain's shadow.
        for step, file in enumerate(sorted(rng.sample(range(CANONICAL_FILES), 6))):
            builder.place(step % 3, file, BOMB)
        builder.place_random(rng, SPY, _rank_cells(3))
        builder.place_random(rng, FLAG, _rank_cells(1))
        _fill_rest(builder, rng)
    return builder


_TEMPLATES = {
    "bombed_corner_flag": _bombed_corner_flag,
    "bombed_center_flag": _bombed_center_flag,
    "scout_screen": _scout_screen,
    "aggressive_marshal": _aggressive_marshal,
    "spy_shadow": _spy_shadow,
    "miner_wall": _miner_wall,
    "decoy_flag_structure": _decoy_flag_structure,
    "free_novelty": _free_novelty,
}
assert set(_TEMPLATES) == set(AUTHORED_FAMILIES)


# ---------------------------------------------------------------------------
# Properties and family checks
# ---------------------------------------------------------------------------


def setup_properties(canonical: "tuple[int, ...]") -> dict:
    """Structural facts of one canonical setup, for metadata and checks."""
    canonical = tuple(canonical)
    flag_index = canonical.index(FLAG)
    flag_rank, flag_file = canonical_rank_file(flag_index)
    marshal_rank, marshal_file = canonical_rank_file(canonical.index(MARSHAL))
    spy_rank, spy_file = canonical_rank_file(canonical.index(SPY))
    general_rank, general_file = canonical_rank_file(canonical.index(GENERAL))
    bombs_adjacent_to_flag = sum(
        1 for cell in canonical_neighbours(flag_index) if canonical[cell] == BOMB
    )
    scouts_front = sum(
        1
        for index, piece in enumerate(canonical)
        if piece == SCOUT and canonical_rank_file(index)[0] == CANONICAL_RANKS - 1
    )
    bombs_front = sum(
        1
        for index, piece in enumerate(canonical)
        if piece == BOMB and canonical_rank_file(index)[0] == CANONICAL_RANKS - 1
    )
    miner_ranks = sorted(
        canonical_rank_file(index)[0]
        for index, piece in enumerate(canonical)
        if piece == MINER
    )
    miner_files = sorted(
        canonical_rank_file(index)[1]
        for index, piece in enumerate(canonical)
        if piece == MINER
    )
    return {
        "flag_rank": flag_rank,
        "flag_file": flag_file,
        "bombs_adjacent_to_flag": bombs_adjacent_to_flag,
        "flag_neighbours": len(canonical_neighbours(flag_index)),
        "marshal_rank": marshal_rank,
        "spy_general_distance": abs(spy_rank - general_rank) + abs(spy_file - general_file),
        "spy_rank": spy_rank,
        "scouts_on_front_rank": scouts_front,
        "bombs_on_front_rank": bombs_front,
        "miner_ranks": miner_ranks,
        "miner_file_spread": (miner_files[-1] - miner_files[0]) if miner_files else 0,
    }


def _check_family_shape(family: str, canonical: "tuple[int, ...]") -> None:
    """Refuse an authored setup that does not carry its family's signature."""
    facts = setup_properties(canonical)
    ok = True
    if family == "bombed_corner_flag":
        ok = (
            facts["flag_rank"] == 0
            and facts["flag_file"] in (0, CANONICAL_FILES - 1)
            and facts["bombs_adjacent_to_flag"] == facts["flag_neighbours"]
        )
    elif family == "bombed_center_flag":
        ok = (
            facts["flag_rank"] == 0
            and 3 <= facts["flag_file"] <= 6
            and facts["bombs_adjacent_to_flag"] == facts["flag_neighbours"]
        )
    elif family == "scout_screen":
        ok = facts["scouts_on_front_rank"] >= 6
    elif family == "aggressive_marshal":
        ok = facts["marshal_rank"] == CANONICAL_RANKS - 1
    elif family == "spy_shadow":
        ok = facts["spy_rank"] >= 2 and facts["spy_general_distance"] <= 2
    elif family == "miner_wall":
        ok = facts["miner_ranks"] == [2, 2, 2, 2, 2] and facts["miner_file_spread"] >= 6
    elif family == "decoy_flag_structure":
        corner = 0 if facts["flag_file"] == 0 else CANONICAL_FILES - 1
        decoy_index = canonical_index(0, CANONICAL_FILES - 1 - corner)
        decoy_sealed = all(
            canonical[cell] == BOMB for cell in canonical_neighbours(decoy_index)
        )
        ok = (
            facts["flag_rank"] == 0
            and facts["flag_file"] in (0, CANONICAL_FILES - 1)
            and facts["bombs_adjacent_to_flag"] == 0
            and decoy_sealed
            and canonical[decoy_index] != FLAG
        )
    elif family == "free_novelty":
        ok = (
            facts["flag_rank"] > 0
            or facts["bombs_on_front_rank"] >= 2
            or (facts["bombs_adjacent_to_flag"] == 0 and facts["marshal_rank"] == 0)
        )
    if not ok:
        raise Phase16MeasurementError(
            f"authored setup does not carry the {family!r} signature: {facts}"
        )


def validate_setup(canonical: "tuple[int, ...]") -> dict:
    """The full imported section-4 gate on one canonical setup.

    Runs `check_board` with the setup on both seats, which exercises
    `oriented_for` for RED and BLUE, the exact-inventory check and the
    paired-mirror check — and returns the gate's findings.
    """
    canonical = tuple(canonical)
    if len(canonical) != CANONICAL_CELLS:
        raise Phase16MeasurementError(
            f"a canonical setup has {CANONICAL_CELLS} cells, got {len(canonical)}"
        )
    report = check_board(canonical, canonical)
    # Explicit second checks on the tuples a game would actually consume.
    oriented_for(canonical, RED)
    oriented_for(canonical, BLUE)
    return report


# ---------------------------------------------------------------------------
# Authoring the library
# ---------------------------------------------------------------------------


def author_setup(family: str, ordinal: int) -> dict:
    """One authored entry, deterministic from `(family, ordinal)`."""
    if family not in AUTHORED_FAMILIES:
        raise Phase16MeasurementError(
            f"family must be one of {list(AUTHORED_FAMILIES)}, got {family!r}"
        )
    if not 0 <= int(ordinal) < SETUPS_PER_FAMILY:
        raise Phase16MeasurementError(
            f"ordinal must be in 0..{SETUPS_PER_FAMILY - 1}, got {ordinal!r}"
        )
    seed = derive_measure_seed(DOMAIN_ADVERSARIAL, family, int(ordinal))
    rng = random.Random(seed)
    canonical = _TEMPLATES[family](rng, int(ordinal)).finish()
    validate_setup(canonical)
    _check_family_shape(family, canonical)
    return {
        "setup_id": f"p16adv|{family}|{int(ordinal):02d}",
        "family": family,
        "ordinal": int(ordinal),
        "canonical_setup": list(canonical),
        "authoring_seed": seed,
        "authoring_version": AUTHORING_VERSION,
        "source": "authored",
        "properties": setup_properties(canonical),
    }


def author_library() -> "dict[str, list[dict]]":
    """Every authored family, deduplicated across the whole library."""
    families: dict[str, list[dict]] = {FAMILY_OPERATOR_HARVEST: []}
    seen: set = set()
    for family in AUTHORED_FAMILIES:
        entries = []
        for ordinal in range(SETUPS_PER_FAMILY):
            entry = author_setup(family, ordinal)
            key = tuple(entry["canonical_setup"])
            if key in seen:
                raise Phase16MeasurementError(
                    f"duplicate canonical setup: {entry['setup_id']}"
                )
            seen.add(key)
            entries.append(entry)
        families[family] = entries
    return families


def _entries_digest(families: "dict[str, list[dict]]", *, include) -> str:
    body = json.dumps(
        {
            family: [
                {
                    "setup_id": entry["setup_id"],
                    "canonical_setup": list(entry["canonical_setup"]),
                }
                for entry in families.get(family, [])
            ]
            for family in include
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(body.encode()).hexdigest()


def build_library_document(
    families: "dict[str, list[dict]]", *, generated_utc: str, harvest_revision: int = 0
) -> dict:
    for family in ADVERSARIAL_FAMILIES:
        families.setdefault(family, [])
    unknown = set(families) - set(ADVERSARIAL_FAMILIES)
    if unknown:
        raise Phase16MeasurementError(f"unknown families: {sorted(unknown)}")
    total = sum(len(entries) for entries in families.values())
    return {
        "artifact": ADVERSARIAL_VERSION,
        "format": (
            "accepted setup-library representation: canonical own-orientation "
            "40-tuples (rank 0 = own back row, index = rank*10 + file) plus "
            "family metadata"
        ),
        "authoring_version": AUTHORING_VERSION,
        "generated_utc": generated_utc,
        "orientation_rule_version": ORIENTATION_RULE_VERSION,
        "orientation_rule": ORIENTATION_RULE,
        "gate": (
            "every entry passed stratego.belief.phase15.orientation.check_board "
            "(imported): flag row, legal rows, exact inventory, paired mirror"
        ),
        "families": {
            family: {
                "description": FAMILY_DESCRIPTIONS[family],
                "authored": family != FAMILY_OPERATOR_HARVEST,
                "setups": families[family],
                "setup_count": len(families[family]),
            }
            for family in ADVERSARIAL_FAMILIES
        },
        "setup_count": total,
        "setups_per_authored_family": SETUPS_PER_FAMILY,
        "harvest_revision": int(harvest_revision),
        "authored_digest": _entries_digest(families, include=AUTHORED_FAMILIES),
        "library_digest": _entries_digest(families, include=ADVERSARIAL_FAMILIES),
    }


def save_library(document: dict, path: "Path | str" = DEFAULT_LIBRARY_PATH, *, root: "Path | str" = ".") -> Path:
    full = Path(root) / Path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    return full


def load_library(path: "Path | str" = DEFAULT_LIBRARY_PATH, *, root: "Path | str" = ".") -> dict:
    full = Path(root) / Path(path)
    if not full.is_file():
        raise Phase16MeasurementError(f"no adversarial library at {full}")
    document = json.loads(full.read_text())
    if document.get("artifact") != ADVERSARIAL_VERSION:
        raise Phase16MeasurementError(f"{full} is not a {ADVERSARIAL_VERSION} document")
    families = {
        family: list(block["setups"]) for family, block in document["families"].items()
    }
    for name, key in (("authored_digest", AUTHORED_FAMILIES), ("library_digest", ADVERSARIAL_FAMILIES)):
        observed = _entries_digest(families, include=key)
        if observed != document.get(name):
            raise Phase16MeasurementError(
                f"{full}: stored {name} {document.get(name)} != recomputed {observed}; "
                "refusing a tampered library"
            )
    return document


def library_entry(document: dict, family: str, ordinal: int) -> dict:
    try:
        return document["families"][family]["setups"][int(ordinal)]
    except (KeyError, IndexError):
        raise Phase16MeasurementError(
            f"no setup {ordinal!r} in family {family!r}"
        ) from None


def append_harvest_setup(
    document: dict,
    canonical: "tuple[int, ...]",
    *,
    provenance: dict,
    captured_utc: str,
) -> "dict | None":
    """Append one operator setup to `operator_harvest`, deduplicated by tuple.

    Returns the new entry, or `None` when the identical tuple is already in
    the harvest family. The authored digest is untouched; the library digest
    and `harvest_revision` change, which is the visible revision mark.
    """
    canonical = tuple(int(piece) for piece in canonical)
    validate_setup(canonical)
    harvest = document["families"][FAMILY_OPERATOR_HARVEST]["setups"]
    if any(tuple(entry["canonical_setup"]) == canonical for entry in harvest):
        return None
    entry = {
        "setup_id": f"p16adv|{FAMILY_OPERATOR_HARVEST}|{len(harvest):02d}",
        "family": FAMILY_OPERATOR_HARVEST,
        "ordinal": len(harvest),
        "canonical_setup": list(canonical),
        "source": "operator",
        "captured_utc": captured_utc,
        "provenance": dict(provenance),
        "properties": setup_properties(canonical),
    }
    harvest.append(entry)
    document["families"][FAMILY_OPERATOR_HARVEST]["setup_count"] = len(harvest)
    document["setup_count"] = sum(
        block["setup_count"] for block in document["families"].values()
    )
    document["harvest_revision"] = int(document.get("harvest_revision", 0)) + 1
    families = {
        family: list(block["setups"]) for family, block in document["families"].items()
    }
    document["library_digest"] = _entries_digest(families, include=ADVERSARIAL_FAMILIES)
    return entry


__all__ = [
    "AUTHORING_VERSION",
    "DEFAULT_LIBRARY_PATH",
    "FAMILY_DESCRIPTIONS",
    "append_harvest_setup",
    "author_library",
    "author_setup",
    "build_library_document",
    "library_entry",
    "load_library",
    "save_library",
    "setup_properties",
    "validate_setup",
]
