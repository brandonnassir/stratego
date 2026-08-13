"""Deterministic structural trait vector: `setup_trait_vector_v1`.

Specification sources:

- `01_AGENT_1_SETUP_CONTRACT_AND_TAXONOMY.md` (structural trait vector)
- `00_PHASE_7_SEQUENCE_AND_COMMON_CONTRACT.md` (diversity standard)

Every setup in the library carries one trait vector computed by
:func:`compute_trait_vector` from the canonical own-orientation 40-tuple
alone. Traits support family validation, diversity measurement, family
overlap analysis and future stratified evaluation. No trait involves playing
strength, game outcomes or neural outputs.

Frame and invariance
--------------------
Traits are computed in the canonical frame (rank 0 = own back row, rank 3 =
own front row). Every field except `flag_file` is reflection-invariant:
rank-based counts ignore files entirely, file-based fields use edge distance
or file multisets, distances are preserved by the file mirror, and
existential fields quantify over cells, which reflection permutes. Family
predicates may only reference reflection-invariant fields, which is what
makes family membership a property of the reflection class. `flag_file` is
recorded for inspection of the stored representative only.

Determinism
-----------
All fields are small-integer arithmetic; the four float fields are rounded to
six decimal places at computation time, so equality of trait vectors is exact
equality of values, reproducible across processes and platforms.

Combat ranks come from the engine's `PIECE_RANKS` table. Flag and Bomb are
identified by their piece-type constants, never inferred from enum ordinals.
"""

import math
from dataclasses import dataclass

from ..engine.constants import (
    BOMB,
    FLAG,
    GENERAL,
    IMMOVABLE_TYPES,
    MARSHAL,
    MINER,
    PIECE_COUNTS,
    PIECE_RANKS,
    PIECE_TYPES,
    SCOUT,
    SPY,
)
from .identity import (
    CANONICAL_CELLS,
    CANONICAL_FILES,
    CANONICAL_RANKS,
    FRONT_RANK,
    SetupLibraryError,
    canonical_rank_file,
    edge_file_distance,
)

TRAIT_SCHEMA_VERSION = "setup_trait_vector_v1"

#: Piece types counted as "high ranks": combat rank 7 or above (Major,
#: Colonel, General, Marshal — 3 + 2 + 1 + 1 = 7 pieces). Membership is
#: decided by the engine's combat-rank table, not by enum position.
HIGH_RANK_THRESHOLD = 7
HIGH_RANK_TYPES = tuple(
    sorted(
        piece_type
        for piece_type in PIECE_TYPES
        if PIECE_RANKS[piece_type] is not None
        and PIECE_RANKS[piece_type] >= HIGH_RANK_THRESHOLD
    )
)
assert sum(PIECE_COUNTS[t] for t in HIGH_RANK_TYPES) == 7

#: Canonical front-rank files whose forward square is not a lake. The lakes
#: occupy board columns 2, 3, 6 and 7 in both central rows, so a front-rank
#: piece on any other file faces an empty square at game start. The set is
#: symmetric under the file mirror `f -> 9 - f`.
OPEN_FRONT_FILES = (0, 1, 4, 5, 8, 9)

#: Manhattan distance from the Flag at or beyond which a guarded movable piece
#: counts as a decoy pocket rather than part of the Flag's own defense.
DECOY_MIN_FLAG_DISTANCE = 4

#: The fixed unconventional-structure feature list used by family F15. Each
#: entry is `(feature_name, description)`; the corresponding predicate is
#: implemented in :func:`_unconventional_features`. Frozen with the trait
#: schema: changing the list is a new trait-schema version.
UNCONVENTIONAL_FEATURES = (
    ("flag_forward", "Flag on rank 2 or 3 (front half of the setup zone)"),
    ("flag_unguarded", "no Bomb orthogonally adjacent to the Flag"),
    ("bombs_on_front_rank", "3 or more Bombs on the front rank"),
    ("marshal_on_front_rank", "Marshal on the front rank"),
    ("general_on_front_rank", "General on the front rank"),
    ("no_front_rank_scouts", "no Scout on the front rank"),
    ("miners_on_front_rank", "3 or more Miners on the front rank"),
    ("spy_on_front_rank", "Spy on the front rank"),
)


@dataclass(frozen=True)
class TraitField:
    """Schema entry for one trait-vector field."""

    name: str
    kind: str  # "int", "float6", or "int_list4"
    units: str
    description: str
    reflection_invariant: bool


TRAIT_SCHEMA: tuple[TraitField, ...] = (
    TraitField("flag_rank", "int", "canonical rank 0..3", "rank of the Flag (0 = back row)", True),
    TraitField("flag_file", "int", "canonical file 0..9", "file of the Flag in the stored representative orientation", False),
    TraitField("flag_edge_distance", "int", "files 0..4", "min(flag_file, 9 - flag_file)", True),
    TraitField("flag_orth_bomb_guards", "int", "bombs 0..4", "Bombs orthogonally adjacent to the Flag", True),
    TraitField("flag_diag_bomb_guards", "int", "bombs 0..4", "Bombs diagonally adjacent to the Flag", True),
    TraitField("flag_zone_bomb_count_r2", "int", "bombs 0..6", "Bombs within Chebyshev distance 2 of the Flag", True),
    TraitField("bomb_rank_histogram", "int_list4", "bombs per rank, sums to 6", "Bomb count on ranks 0..3", True),
    TraitField("bomb_front2_count", "int", "bombs 0..6", "Bombs on ranks 2..3", True),
    TraitField("bomb_back2_count", "int", "bombs 0..6", "Bombs on ranks 0..1", True),
    TraitField("bomb_front_rank_count", "int", "bombs 0..6", "Bombs on rank 3", True),
    TraitField("bomb_distinct_files", "int", "files 1..6", "distinct files occupied by Bombs", True),
    TraitField("bomb_adjacent_pairs", "int", "pairs 0..15", "unordered orthogonally adjacent Bomb pairs", True),
    TraitField("bomb_mean_pairwise_manhattan", "float6", "squares", "mean Manhattan distance over the 15 Bomb pairs", True),
    TraitField("scout_rank_histogram", "int_list4", "scouts per rank, sums to 8", "Scout count on ranks 0..3", True),
    TraitField("scout_front2_count", "int", "scouts 0..8", "Scouts on ranks 2..3", True),
    TraitField("scout_back2_count", "int", "scouts 0..8", "Scouts on ranks 0..1", True),
    TraitField("scout_front_rank_count", "int", "scouts 0..8", "Scouts on rank 3", True),
    TraitField("miner_rank_histogram", "int_list4", "miners per rank, sums to 5", "Miner count on ranks 0..3", True),
    TraitField("miner_front2_count", "int", "miners 0..5", "Miners on ranks 2..3", True),
    TraitField("miner_back2_count", "int", "miners 0..5", "Miners on ranks 0..1", True),
    TraitField("miner_front_rank_count", "int", "miners 0..5", "Miners on rank 3", True),
    TraitField("high_rank_histogram", "int_list4", "pieces per rank, sums to 7", "combat rank >= 7 pieces on ranks 0..3", True),
    TraitField("high_front2_count", "int", "pieces 0..7", "combat rank >= 7 pieces on ranks 2..3", True),
    TraitField("high_back2_count", "int", "pieces 0..7", "combat rank >= 7 pieces on ranks 0..1", True),
    TraitField("marshal_rank", "int", "canonical rank 0..3", "rank of the Marshal", True),
    TraitField("general_rank", "int", "canonical rank 0..3", "rank of the General", True),
    TraitField("spy_rank", "int", "canonical rank 0..3", "rank of the Spy", True),
    TraitField("movable_front_rank_count", "int", "pieces 0..10", "movable pieces on rank 3", True),
    TraitField("front_rank_immovable_count", "int", "pieces 0..10", "Flag/Bomb pieces on rank 3", True),
    TraitField("open_file_movable_front_count", "int", "pieces 0..6", "movable rank-3 pieces on files {0,1,4,5,8,9}", True),
    TraitField("decoy_pocket_bombs", "int", "bombs 0..4", "max orthogonal Bombs around a movable back-half piece at Manhattan >= 4 from the Flag", True),
    TraitField("bomb_rank_entropy_bits", "float6", "bits", "Shannon entropy of the Bomb rank histogram", True),
    TraitField("scout_rank_entropy_bits", "float6", "bits", "Shannon entropy of the Scout rank histogram", True),
    TraitField("miner_rank_entropy_bits", "float6", "bits", "Shannon entropy of the Miner rank histogram", True),
    TraitField("unconventional_feature_count", "int", "features 0..8", "count of true UNCONVENTIONAL_FEATURES predicates", True),
)

TRAIT_NAMES = tuple(field.name for field in TRAIT_SCHEMA)


def trait_schema_document() -> dict:
    """The machine-readable trait schema, for the Agent 1 contract artifact."""
    return {
        "trait_schema_version": TRAIT_SCHEMA_VERSION,
        "frame": "canonical own-orientation (rank 0 = back row, rank 3 = front row)",
        "float_rounding": "float6 fields are rounded to 6 decimal places at computation time",
        "high_rank_threshold": HIGH_RANK_THRESHOLD,
        "high_rank_types": list(HIGH_RANK_TYPES),
        "open_front_files": list(OPEN_FRONT_FILES),
        "decoy_min_flag_distance": DECOY_MIN_FLAG_DISTANCE,
        "unconventional_features": [
            {"name": name, "description": description}
            for name, description in UNCONVENTIONAL_FEATURES
        ],
        "fields": [
            {
                "name": field.name,
                "kind": field.kind,
                "units": field.units,
                "description": field.description,
                "reflection_invariant": field.reflection_invariant,
            }
            for field in TRAIT_SCHEMA
        ],
    }


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _cells_of_type(canonical: tuple[int, ...], piece_type: int) -> tuple[int, ...]:
    return tuple(
        index for index, entry in enumerate(canonical) if entry == piece_type
    )


def _rank_histogram(cells: "tuple[int, ...]") -> list[int]:
    histogram = [0] * CANONICAL_RANKS
    for cell in cells:
        histogram[canonical_rank_file(cell)[0]] += 1
    return histogram


def _entropy_bits(histogram: "list[int]") -> float:
    total = sum(histogram)
    entropy = 0.0
    for count in histogram:
        if count > 0:
            probability = count / total
            entropy -= probability * math.log2(probability)
    return round(entropy, 6)


def _manhattan(cell_a: int, cell_b: int) -> int:
    rank_a, file_a = canonical_rank_file(cell_a)
    rank_b, file_b = canonical_rank_file(cell_b)
    return abs(rank_a - rank_b) + abs(file_a - file_b)


def _chebyshev(cell_a: int, cell_b: int) -> int:
    rank_a, file_a = canonical_rank_file(cell_a)
    rank_b, file_b = canonical_rank_file(cell_b)
    return max(abs(rank_a - rank_b), abs(file_a - file_b))


def _orthogonal_bomb_count(canonical: tuple[int, ...], cell: int) -> int:
    rank, file = canonical_rank_file(cell)
    count = 0
    for delta_rank, delta_file in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbour_rank = rank + delta_rank
        neighbour_file = file + delta_file
        if 0 <= neighbour_rank < CANONICAL_RANKS and 0 <= neighbour_file < CANONICAL_FILES:
            if canonical[neighbour_rank * CANONICAL_FILES + neighbour_file] == BOMB:
                count += 1
    return count


def _diagonal_bomb_count(canonical: tuple[int, ...], cell: int) -> int:
    rank, file = canonical_rank_file(cell)
    count = 0
    for delta_rank, delta_file in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        neighbour_rank = rank + delta_rank
        neighbour_file = file + delta_file
        if 0 <= neighbour_rank < CANONICAL_RANKS and 0 <= neighbour_file < CANONICAL_FILES:
            if canonical[neighbour_rank * CANONICAL_FILES + neighbour_file] == BOMB:
                count += 1
    return count


def _unconventional_features(canonical: tuple[int, ...]) -> dict[str, bool]:
    """Evaluate the fixed F15 feature list. Order matches the schema tuple."""
    flag_cell = canonical.index(FLAG)
    flag_rank = canonical_rank_file(flag_cell)[0]
    bomb_cells = _cells_of_type(canonical, BOMB)
    bomb_front_rank = sum(
        1 for cell in bomb_cells if canonical_rank_file(cell)[0] == FRONT_RANK
    )
    scout_front_rank = sum(
        1
        for cell in _cells_of_type(canonical, SCOUT)
        if canonical_rank_file(cell)[0] == FRONT_RANK
    )
    miner_front_rank = sum(
        1
        for cell in _cells_of_type(canonical, MINER)
        if canonical_rank_file(cell)[0] == FRONT_RANK
    )
    return {
        "flag_forward": flag_rank >= 2,
        "flag_unguarded": _orthogonal_bomb_count(canonical, flag_cell) == 0,
        "bombs_on_front_rank": bomb_front_rank >= 3,
        "marshal_on_front_rank": canonical_rank_file(canonical.index(MARSHAL))[0] == FRONT_RANK,
        "general_on_front_rank": canonical_rank_file(canonical.index(GENERAL))[0] == FRONT_RANK,
        "no_front_rank_scouts": scout_front_rank == 0,
        "miners_on_front_rank": miner_front_rank >= 3,
        "spy_on_front_rank": canonical_rank_file(canonical.index(SPY))[0] == FRONT_RANK,
    }


def compute_trait_vector(canonical: "list[int] | tuple[int, ...]") -> dict:
    """The deterministic `setup_trait_vector_v1` of a canonical arrangement.

    The result is a plain dict whose keys appear in `TRAIT_SCHEMA` order.
    Input inventory is checked implicitly: the single Flag/Marshal/General/Spy
    lookups and the fixed-size histograms fail loudly on malformed input.
    """
    entries = tuple(canonical)
    if len(entries) != CANONICAL_CELLS:
        raise SetupLibraryError(
            f"expected {CANONICAL_CELLS} canonical entries, got {len(entries)}"
        )

    flag_cells = _cells_of_type(entries, FLAG)
    if len(flag_cells) != 1:
        raise SetupLibraryError(f"expected exactly one Flag, found {len(flag_cells)}")
    flag_cell = flag_cells[0]
    flag_rank, flag_file = canonical_rank_file(flag_cell)

    bomb_cells = _cells_of_type(entries, BOMB)
    if len(bomb_cells) != PIECE_COUNTS[BOMB]:
        raise SetupLibraryError(f"expected {PIECE_COUNTS[BOMB]} Bombs, found {len(bomb_cells)}")
    scout_cells = _cells_of_type(entries, SCOUT)
    miner_cells = _cells_of_type(entries, MINER)
    high_cells = tuple(
        index for index, entry in enumerate(entries) if entry in HIGH_RANK_TYPES
    )

    bomb_histogram = _rank_histogram(bomb_cells)
    scout_histogram = _rank_histogram(scout_cells)
    miner_histogram = _rank_histogram(miner_cells)
    high_histogram = _rank_histogram(high_cells)

    bomb_pair_distances = [
        _manhattan(bomb_cells[i], bomb_cells[j])
        for i in range(len(bomb_cells))
        for j in range(i + 1, len(bomb_cells))
    ]
    bomb_adjacent_pairs = sum(1 for distance in bomb_pair_distances if distance == 1)

    front_rank_cells = tuple(
        FRONT_RANK * CANONICAL_FILES + file for file in range(CANONICAL_FILES)
    )
    movable_front = sum(
        1 for cell in front_rank_cells if entries[cell] not in IMMOVABLE_TYPES
    )
    open_file_movable_front = sum(
        1
        for file in OPEN_FRONT_FILES
        if entries[FRONT_RANK * CANONICAL_FILES + file] not in IMMOVABLE_TYPES
    )

    decoy_pocket = 0
    for cell, entry in enumerate(entries):
        if entry in IMMOVABLE_TYPES:
            continue
        if canonical_rank_file(cell)[0] > 1:
            continue  # pockets are back-half structures (ranks 0..1)
        if _manhattan(cell, flag_cell) < DECOY_MIN_FLAG_DISTANCE:
            continue
        decoy_pocket = max(decoy_pocket, _orthogonal_bomb_count(entries, cell))

    features = _unconventional_features(entries)

    return {
        "flag_rank": flag_rank,
        "flag_file": flag_file,
        "flag_edge_distance": edge_file_distance(flag_file),
        "flag_orth_bomb_guards": _orthogonal_bomb_count(entries, flag_cell),
        "flag_diag_bomb_guards": _diagonal_bomb_count(entries, flag_cell),
        "flag_zone_bomb_count_r2": sum(
            1 for cell in bomb_cells if _chebyshev(cell, flag_cell) <= 2
        ),
        "bomb_rank_histogram": bomb_histogram,
        "bomb_front2_count": bomb_histogram[2] + bomb_histogram[3],
        "bomb_back2_count": bomb_histogram[0] + bomb_histogram[1],
        "bomb_front_rank_count": bomb_histogram[FRONT_RANK],
        "bomb_distinct_files": len(
            {canonical_rank_file(cell)[1] for cell in bomb_cells}
        ),
        "bomb_adjacent_pairs": bomb_adjacent_pairs,
        "bomb_mean_pairwise_manhattan": round(
            sum(bomb_pair_distances) / len(bomb_pair_distances), 6
        ),
        "scout_rank_histogram": scout_histogram,
        "scout_front2_count": scout_histogram[2] + scout_histogram[3],
        "scout_back2_count": scout_histogram[0] + scout_histogram[1],
        "scout_front_rank_count": scout_histogram[FRONT_RANK],
        "miner_rank_histogram": miner_histogram,
        "miner_front2_count": miner_histogram[2] + miner_histogram[3],
        "miner_back2_count": miner_histogram[0] + miner_histogram[1],
        "miner_front_rank_count": miner_histogram[FRONT_RANK],
        "high_rank_histogram": high_histogram,
        "high_front2_count": high_histogram[2] + high_histogram[3],
        "high_back2_count": high_histogram[0] + high_histogram[1],
        "marshal_rank": canonical_rank_file(entries.index(MARSHAL))[0],
        "general_rank": canonical_rank_file(entries.index(GENERAL))[0],
        "spy_rank": canonical_rank_file(entries.index(SPY))[0],
        "movable_front_rank_count": movable_front,
        "front_rank_immovable_count": CANONICAL_FILES - movable_front,
        "open_file_movable_front_count": open_file_movable_front,
        "decoy_pocket_bombs": decoy_pocket,
        "bomb_rank_entropy_bits": _entropy_bits(bomb_histogram),
        "scout_rank_entropy_bits": _entropy_bits(scout_histogram),
        "miner_rank_entropy_bits": _entropy_bits(miner_histogram),
        "unconventional_feature_count": sum(1 for value in features.values() if value),
    }
