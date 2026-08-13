"""Diversity metrics and pre-generation numeric thresholds.

Specification sources:

- `00_PHASE_7_SEQUENCE_AND_COMMON_CONTRACT.md` (diversity standard)
- `01_AGENT_1_SETUP_CONTRACT_AND_TAXONOMY.md` (diversity metrics)

Everything numeric in this module is frozen **before** the 8,000 production
setups exist. Agent 2 generates against these thresholds, Agent 3 audits
against them, and no later agent may move them after seeing results.

Design position
---------------
The standard must distinguish a strategically constrained family from a
library that repeats a few arrangements. It deliberately does **not** demand
uniform randomness: families legitimately pin strategically defining cells
(F00 pins three), so entropy and support floors are set well below the
unconstrained ceiling and, where family identity constrains a region, the
floors are family-specific.

Distance thresholds are set far below the distances independent structured
draws actually produce (expected class distance is roughly 30 of 40 squares),
so a correct generator passes with enormous margin while template repetition
with cosmetic swaps — the failure the standard exists to catch — fails
immediately. Because generation is collision-oblivious per base identity
(see `contracts.py`), a threshold violation under the frozen contract is a
finding to report, never a licence to silently regenerate.

Folded cells
------------
Positional support metrics count "folded" cells `(rank, edge_file_distance)`
— 4 ranks x 5 edge distances = 20 cells — which are reflection-invariant, so
support is a property of reflection classes, not of which representative the
canonicalization happened to store.

Entries
-------
Library-level functions take `LibraryEntry` records (family, split, canonical
arrangement). Agent 3 builds them from the materialized JSONL; the functions
recompute everything from content and never trust stored metadata.
"""

import math
from dataclasses import dataclass

import numpy as np

from ..engine.constants import BOMB, FLAG, MINER, NUM_PIECE_TYPES, SCOUT
from .families import FAMILY_CONTRACTS, FAMILY_IDS
from .identity import (
    CANONICAL_CELLS,
    SetupLibraryError,
    canonical_rank_file,
    class_fingerprint,
    edge_file_distance,
    is_canonical_representative,
    reflect_canonical,
)
from .traits import HIGH_RANK_TYPES, compute_trait_vector

DIVERSITY_STANDARD_VERSION = "setup_diversity_standard_v1"

FOLDED_RANKS = 4
FOLDED_EDGE_DISTANCES = 5
FOLDED_CELL_COUNT = FOLDED_RANKS * FOLDED_EDGE_DISTANCES


@dataclass(frozen=True)
class LibraryEntry:
    """The minimum content needed to audit one base setup."""

    family_id: str
    split: str
    canonical: tuple[int, ...]


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------


def hamming_distance(
    a: "list[int] | tuple[int, ...]", b: "list[int] | tuple[int, ...]"
) -> int:
    """Piece-type Hamming distance over the 40 canonical squares."""
    left, right = tuple(a), tuple(b)
    if len(left) != CANONICAL_CELLS or len(right) != CANONICAL_CELLS:
        raise SetupLibraryError("hamming distance requires two 40-entry setups")
    return sum(1 for x, y in zip(left, right) if x != y)


def class_distance(
    a: "list[int] | tuple[int, ...]", b: "list[int] | tuple[int, ...]"
) -> int:
    """Reflection-class distance: `min(H(a, b), H(a, reflect(b)))`.

    Symmetric and well-defined on reflection classes: replacing either
    argument by its reflection leaves the value unchanged, so leakage checks
    catch mirrored near-copies as effectively as direct ones.
    """
    return min(
        hamming_distance(a, b),
        hamming_distance(a, reflect_canonical(tuple(b))),
    )


def _setups_matrix(entries: "list[LibraryEntry]") -> np.ndarray:
    matrix = np.array([entry.canonical for entry in entries], dtype=np.uint8)
    if matrix.ndim != 2 or (matrix.size and matrix.shape[1] != CANONICAL_CELLS):
        raise SetupLibraryError("entries must hold 40-entry canonical setups")
    return matrix


def _pairwise_class_distances(entries: "list[LibraryEntry]") -> np.ndarray:
    """Dense class-distance matrix (uint8) with an infinite-like diagonal.

    Vectorized over numpy so Agent 3 can afford the full 8,000 x 8,000 audit.
    The diagonal is set to CANONICAL_CELLS + 1 so nearest-neighbour reductions
    ignore self-pairs.
    """
    direct = _setups_matrix(entries)
    mirrored = np.array(
        [reflect_canonical(entry.canonical) for entry in entries], dtype=np.uint8
    )
    count = direct.shape[0]
    distances = np.empty((count, count), dtype=np.uint8)
    block = 256
    for start in range(0, count, block):
        stop = min(start + block, count)
        chunk = direct[start:stop, None, :]
        plain = (chunk != direct[None, :, :]).sum(axis=2)
        flipped = (chunk != mirrored[None, :, :]).sum(axis=2)
        distances[start:stop] = np.minimum(plain, flipped).astype(np.uint8)
    fill = CANONICAL_CELLS + 1
    np.fill_diagonal(distances, fill)
    return distances


# ---------------------------------------------------------------------------
# Identity / leakage metrics
# ---------------------------------------------------------------------------


def identity_metrics(entries: "list[LibraryEntry]") -> dict:
    """Exact, reflection-equivalent, and cross-split duplicate counts.

    All counts are computed from content via the reflection-class
    fingerprint, so a stored non-canonical orientation cannot hide a
    duplicate. `non_canonical_entries` separately counts entries that are not
    their own class representative, which the storage contract forbids.
    """
    fingerprints = [class_fingerprint(entry.canonical) for entry in entries]
    exact = {}
    for index, entry in enumerate(entries):
        exact.setdefault(entry.canonical, []).append(index)
    classes: dict[str, list[int]] = {}
    for index, fingerprint in enumerate(fingerprints):
        classes.setdefault(fingerprint, []).append(index)

    cross_split_duplicates = 0
    for members in classes.values():
        splits = {entries[index].split for index in members}
        if len(members) > 1 and len(splits) > 1:
            cross_split_duplicates += 1

    return {
        "entry_count": len(entries),
        "exact_duplicate_groups": sum(1 for group in exact.values() if len(group) > 1),
        "reflection_class_duplicate_groups": sum(
            1 for group in classes.values() if len(group) > 1
        ),
        "cross_split_class_duplicate_groups": cross_split_duplicates,
        "distinct_class_fingerprints": len(classes),
        "non_canonical_entries": sum(
            1 for entry in entries if not is_canonical_representative(entry.canonical)
        ),
    }


# ---------------------------------------------------------------------------
# Nearest-neighbour distance metrics
# ---------------------------------------------------------------------------


def distance_metrics(entries: "list[LibraryEntry]") -> dict:
    """Within-family, cross-split, and global nearest-neighbour statistics."""
    if len(entries) < 2:
        raise SetupLibraryError("distance metrics need at least two entries")
    distances = _pairwise_class_distances(entries)
    families = np.array([FAMILY_IDS.index(entry.family_id) for entry in entries])
    splits = np.array([entry.split for entry in entries])

    per_family: dict[str, dict] = {}
    for family_index, family_id in enumerate(FAMILY_IDS):
        member_indices = np.nonzero(families == family_index)[0]
        if member_indices.size < 2:
            continue
        submatrix = distances[np.ix_(member_indices, member_indices)]
        upper = submatrix[np.triu_indices(member_indices.size, k=1)]
        per_family[family_id] = {
            "min_nn_distance": int(upper.min()),
            "near_duplicate_pair_fraction": float(
                round((upper < 10).sum() / upper.size, 8)
            ),
            "pair_count": int(upper.size),
        }

    cross_mask = splits[:, None] != splits[None, :]
    cross_values = distances[cross_mask]
    global_upper = distances[np.triu_indices(len(entries), k=1)]

    return {
        "within_family": per_family,
        "cross_split_min_nn_distance": int(cross_values.min())
        if cross_values.size
        else None,
        "global_min_pairwise_distance": int(global_upper.min()),
    }


# ---------------------------------------------------------------------------
# Entropy and positional-coverage metrics
# ---------------------------------------------------------------------------


def per_square_entropy_bits(entries: "list[LibraryEntry]") -> list[float]:
    """Shannon entropy (bits) of the piece-type distribution at each cell."""
    if not entries:
        raise SetupLibraryError("entropy needs at least one entry")
    matrix = _setups_matrix(entries)
    result: list[float] = []
    for cell in range(CANONICAL_CELLS):
        counts = np.bincount(matrix[:, cell], minlength=NUM_PIECE_TYPES)
        probabilities = counts[counts > 0] / matrix.shape[0]
        result.append(round(float(-(probabilities * np.log2(probabilities)).sum()), 6))
    return result


def _folded_cell(cell: int) -> tuple[int, int]:
    rank, file = canonical_rank_file(cell)
    return rank, edge_file_distance(file)


def folded_support(entries: "list[LibraryEntry]", piece_types: "tuple[int, ...]") -> int:
    """Distinct folded `(rank, edge_distance)` cells the types occupy."""
    support: set[tuple[int, int]] = set()
    for entry in entries:
        for cell, piece_type in enumerate(entry.canonical):
            if piece_type in piece_types:
                support.add(_folded_cell(cell))
    return len(support)


def entropy_metrics(entries: "list[LibraryEntry]") -> dict:
    """Mean per-square entropy and folded positional support, per family."""
    by_family: dict[str, list[LibraryEntry]] = {family_id: [] for family_id in FAMILY_IDS}
    for entry in entries:
        by_family[entry.family_id].append(entry)

    per_family: dict[str, dict] = {}
    for family_id, members in by_family.items():
        if not members:
            continue
        per_family[family_id] = {
            "mean_per_square_entropy_bits": round(
                float(np.mean(per_square_entropy_bits(members))), 6
            ),
            "flag_folded_support": folded_support(members, (FLAG,)),
            "bomb_folded_support": folded_support(members, (BOMB,)),
            "scout_folded_support": folded_support(members, (SCOUT,)),
            "miner_folded_support": folded_support(members, (MINER,)),
            "high_rank_folded_support": folded_support(members, HIGH_RANK_TYPES),
        }

    return {
        "per_family": per_family,
        "global_mean_per_square_entropy_bits": round(
            float(np.mean(per_square_entropy_bits(entries))), 6
        )
        if entries
        else None,
    }


# ---------------------------------------------------------------------------
# Trait-diversity metrics
# ---------------------------------------------------------------------------


def trait_diversity_metrics(entries: "list[LibraryEntry]") -> dict:
    """Distinct trait vectors and key histograms per family.

    A family must not satisfy its count target by repeating one structural
    pattern with cosmetic swaps; distinct full trait vectors and distinct
    Bomb/Scout rank histograms are the coarse detectors for that.
    """
    by_family: dict[str, list[LibraryEntry]] = {family_id: [] for family_id in FAMILY_IDS}
    for entry in entries:
        by_family[entry.family_id].append(entry)

    per_family: dict[str, dict] = {}
    for family_id, members in by_family.items():
        if not members:
            continue
        vectors = [compute_trait_vector(entry.canonical) for entry in members]
        per_family[family_id] = {
            "member_count": len(members),
            "distinct_trait_vectors": len(
                {
                    tuple(
                        tuple(value) if isinstance(value, list) else value
                        for value in vector.values()
                    )
                    for vector in vectors
                }
            ),
            "distinct_bomb_rank_histograms": len(
                {tuple(vector["bomb_rank_histogram"]) for vector in vectors}
            ),
            "distinct_scout_rank_histograms": len(
                {tuple(vector["scout_rank_histogram"]) for vector in vectors}
            ),
        }
    return {"per_family": per_family}


# ---------------------------------------------------------------------------
# Family overlap / confusion matrix
# ---------------------------------------------------------------------------


def family_overlap_matrix(entries: "list[LibraryEntry]") -> dict:
    """`matrix[i][j]` = fraction of family i entries satisfying family j.

    The diagonal is a hard acceptance requirement (1.0 exactly); off-diagonal
    overlap is expected and report-only.
    """
    by_family: dict[str, list[LibraryEntry]] = {family_id: [] for family_id in FAMILY_IDS}
    for entry in entries:
        by_family[entry.family_id].append(entry)

    matrix: dict[str, dict[str, float]] = {}
    for family_id, members in by_family.items():
        if not members:
            continue
        row: dict[str, float] = {}
        vectors = [compute_trait_vector(entry.canonical) for entry in members]
        for other in FAMILY_CONTRACTS:
            satisfied = sum(
                1 for vector in vectors if other.evaluate(vector)[0]
            )
            row[other.family_id] = round(satisfied / len(members), 6)
        matrix[family_id] = row
    return {"matrix": matrix}


# ---------------------------------------------------------------------------
# Thresholds — frozen before generation
# ---------------------------------------------------------------------------

#: Family-specific floors for distinct folded Flag cells. The possible folded
#: Flag region differs by family (F00 pins one folded cell; F01/F02 allow
#: two; F03-F14 allow the back two ranks; F15 allows the whole zone), so the
#: floors are family-specific by design, not loosened after the fact.
FLAG_FOLDED_SUPPORT_MINIMUM = {
    "F00": 1,
    "F01": 2,
    "F02": 2,
    "F03": 4,
    "F04": 4,
    "F05": 4,
    "F06": 4,
    "F07": 4,
    "F08": 4,
    "F09": 4,
    "F10": 4,
    "F11": 4,
    "F12": 4,
    "F13": 4,
    "F14": 3,
    "F15": 8,
}
assert set(FLAG_FOLDED_SUPPORT_MINIMUM) == set(FAMILY_IDS)


@dataclass(frozen=True)
class DiversityThresholds:
    """Every numeric acceptance threshold of the Phase 7 diversity standard."""

    # Identity / leakage — hard zeros.
    max_exact_duplicate_groups: int = 0
    max_reflection_class_duplicate_groups: int = 0
    max_cross_split_class_duplicate_groups: int = 0
    max_non_canonical_entries: int = 0
    max_stable_id_collisions: int = 0
    max_fingerprint_collisions: int = 0

    # Quality — hard zeros.
    max_engine_invalid_bases: int = 0
    max_stranded_bases: int = 0
    max_family_contract_violations: int = 0

    # Distance floors (class distance over 40 canonical squares).
    min_within_family_nn_distance: int = 6
    max_within_family_near_duplicate_fraction: float = 0.001
    within_family_near_duplicate_distance: int = 10
    min_cross_split_nn_distance: int = 8
    min_global_pairwise_distance: int = 4

    # Entropy floors (bits; unconstrained ceiling is ~3.28).
    min_family_mean_per_square_entropy_bits: float = 1.0
    min_global_mean_per_square_entropy_bits: float = 1.5

    # Folded positional-support floors (of 20 folded cells).
    min_flag_folded_support: "dict[str, int]" = None  # type: ignore[assignment]
    min_bomb_folded_support: int = 10
    min_scout_folded_support: int = 8
    min_miner_folded_support: int = 6
    min_high_rank_folded_support: int = 6

    # Trait-diversity floors (of 500 members per family).
    min_distinct_trait_vectors_per_family: int = 250
    min_distinct_bomb_rank_histograms_per_family: int = 8
    min_distinct_scout_rank_histograms_per_family: int = 8

    # Family overlap.
    required_self_satisfaction: float = 1.0

    def __post_init__(self) -> None:
        if self.min_flag_folded_support is None:
            object.__setattr__(
                self, "min_flag_folded_support", dict(FLAG_FOLDED_SUPPORT_MINIMUM)
            )

    def to_dict(self) -> dict:
        return {
            "diversity_standard_version": DIVERSITY_STANDARD_VERSION,
            "identity": {
                "max_exact_duplicate_groups": self.max_exact_duplicate_groups,
                "max_reflection_class_duplicate_groups": self.max_reflection_class_duplicate_groups,
                "max_cross_split_class_duplicate_groups": self.max_cross_split_class_duplicate_groups,
                "max_non_canonical_entries": self.max_non_canonical_entries,
                "max_stable_id_collisions": self.max_stable_id_collisions,
                "max_fingerprint_collisions": self.max_fingerprint_collisions,
            },
            "quality": {
                "max_engine_invalid_bases": self.max_engine_invalid_bases,
                "max_stranded_bases": self.max_stranded_bases,
                "max_family_contract_violations": self.max_family_contract_violations,
            },
            "distance": {
                "min_within_family_nn_distance": self.min_within_family_nn_distance,
                "within_family_near_duplicate_distance": self.within_family_near_duplicate_distance,
                "max_within_family_near_duplicate_fraction": self.max_within_family_near_duplicate_fraction,
                "min_cross_split_nn_distance": self.min_cross_split_nn_distance,
                "min_global_pairwise_distance": self.min_global_pairwise_distance,
            },
            "entropy": {
                "min_family_mean_per_square_entropy_bits": self.min_family_mean_per_square_entropy_bits,
                "min_global_mean_per_square_entropy_bits": self.min_global_mean_per_square_entropy_bits,
            },
            "positional_support": {
                "min_flag_folded_support": dict(self.min_flag_folded_support),
                "min_bomb_folded_support": self.min_bomb_folded_support,
                "min_scout_folded_support": self.min_scout_folded_support,
                "min_miner_folded_support": self.min_miner_folded_support,
                "min_high_rank_folded_support": self.min_high_rank_folded_support,
            },
            "trait_diversity": {
                "min_distinct_trait_vectors_per_family": self.min_distinct_trait_vectors_per_family,
                "min_distinct_bomb_rank_histograms_per_family": self.min_distinct_bomb_rank_histograms_per_family,
                "min_distinct_scout_rank_histograms_per_family": self.min_distinct_scout_rank_histograms_per_family,
            },
            "family_overlap": {
                "required_self_satisfaction": self.required_self_satisfaction,
                "off_diagonal": "report-only",
            },
        }


DIVERSITY_THRESHOLDS_V1 = DiversityThresholds()


# ---------------------------------------------------------------------------
# Threshold evaluation
# ---------------------------------------------------------------------------


def _check(name: str, passed: bool, observed, required) -> dict:
    return {"check": name, "pass": bool(passed), "observed": observed, "required": required}


def evaluate_against_thresholds(
    entries: "list[LibraryEntry]",
    thresholds: DiversityThresholds = DIVERSITY_THRESHOLDS_V1,
) -> dict:
    """Evaluate every diversity metric of `entries` against `thresholds`.

    Returns `{"checks": [...], "all_pass": bool, "metrics": {...}}`. Families
    absent from `entries` are simply not checked, so the function is usable
    both on the full production library and on partial audit slices; Agent 3
    must run it on the complete library, where every family is present.
    """
    identity = identity_metrics(entries)
    distances = distance_metrics(entries) if len(entries) >= 2 else None
    entropy = entropy_metrics(entries)
    traits = trait_diversity_metrics(entries)
    overlap = family_overlap_matrix(entries)

    checks: list[dict] = []
    checks.append(
        _check(
            "exact_duplicate_groups",
            identity["exact_duplicate_groups"] <= thresholds.max_exact_duplicate_groups,
            identity["exact_duplicate_groups"],
            thresholds.max_exact_duplicate_groups,
        )
    )
    checks.append(
        _check(
            "reflection_class_duplicate_groups",
            identity["reflection_class_duplicate_groups"]
            <= thresholds.max_reflection_class_duplicate_groups,
            identity["reflection_class_duplicate_groups"],
            thresholds.max_reflection_class_duplicate_groups,
        )
    )
    checks.append(
        _check(
            "cross_split_class_duplicate_groups",
            identity["cross_split_class_duplicate_groups"]
            <= thresholds.max_cross_split_class_duplicate_groups,
            identity["cross_split_class_duplicate_groups"],
            thresholds.max_cross_split_class_duplicate_groups,
        )
    )
    checks.append(
        _check(
            "non_canonical_entries",
            identity["non_canonical_entries"] <= thresholds.max_non_canonical_entries,
            identity["non_canonical_entries"],
            thresholds.max_non_canonical_entries,
        )
    )

    if distances is not None:
        for family_id, family_metrics in distances["within_family"].items():
            checks.append(
                _check(
                    f"{family_id}:min_within_family_nn_distance",
                    family_metrics["min_nn_distance"]
                    >= thresholds.min_within_family_nn_distance,
                    family_metrics["min_nn_distance"],
                    thresholds.min_within_family_nn_distance,
                )
            )
            checks.append(
                _check(
                    f"{family_id}:within_family_near_duplicate_fraction",
                    family_metrics["near_duplicate_pair_fraction"]
                    <= thresholds.max_within_family_near_duplicate_fraction,
                    family_metrics["near_duplicate_pair_fraction"],
                    thresholds.max_within_family_near_duplicate_fraction,
                )
            )
        if distances["cross_split_min_nn_distance"] is not None:
            checks.append(
                _check(
                    "cross_split_min_nn_distance",
                    distances["cross_split_min_nn_distance"]
                    >= thresholds.min_cross_split_nn_distance,
                    distances["cross_split_min_nn_distance"],
                    thresholds.min_cross_split_nn_distance,
                )
            )
        checks.append(
            _check(
                "global_min_pairwise_distance",
                distances["global_min_pairwise_distance"]
                >= thresholds.min_global_pairwise_distance,
                distances["global_min_pairwise_distance"],
                thresholds.min_global_pairwise_distance,
            )
        )

    for family_id, family_metrics in entropy["per_family"].items():
        checks.append(
            _check(
                f"{family_id}:mean_per_square_entropy_bits",
                family_metrics["mean_per_square_entropy_bits"]
                >= thresholds.min_family_mean_per_square_entropy_bits,
                family_metrics["mean_per_square_entropy_bits"],
                thresholds.min_family_mean_per_square_entropy_bits,
            )
        )
        checks.append(
            _check(
                f"{family_id}:flag_folded_support",
                family_metrics["flag_folded_support"]
                >= thresholds.min_flag_folded_support[family_id],
                family_metrics["flag_folded_support"],
                thresholds.min_flag_folded_support[family_id],
            )
        )
        for metric_name, floor in (
            ("bomb_folded_support", thresholds.min_bomb_folded_support),
            ("scout_folded_support", thresholds.min_scout_folded_support),
            ("miner_folded_support", thresholds.min_miner_folded_support),
            ("high_rank_folded_support", thresholds.min_high_rank_folded_support),
        ):
            checks.append(
                _check(
                    f"{family_id}:{metric_name}",
                    family_metrics[metric_name] >= floor,
                    family_metrics[metric_name],
                    floor,
                )
            )
    if entropy["global_mean_per_square_entropy_bits"] is not None:
        checks.append(
            _check(
                "global_mean_per_square_entropy_bits",
                entropy["global_mean_per_square_entropy_bits"]
                >= thresholds.min_global_mean_per_square_entropy_bits,
                entropy["global_mean_per_square_entropy_bits"],
                thresholds.min_global_mean_per_square_entropy_bits,
            )
        )

    for family_id, family_metrics in traits["per_family"].items():
        member_count = family_metrics["member_count"]
        # Trait floors are declared for full 500-member families; partial
        # slices scale proportionally so the standard stays executable on
        # audit subsets without weakening the full-library requirement.
        scale = member_count / 500.0
        checks.append(
            _check(
                f"{family_id}:distinct_trait_vectors",
                family_metrics["distinct_trait_vectors"]
                >= math.ceil(thresholds.min_distinct_trait_vectors_per_family * scale),
                family_metrics["distinct_trait_vectors"],
                math.ceil(thresholds.min_distinct_trait_vectors_per_family * scale),
            )
        )
        for metric_name, floor in (
            (
                "distinct_bomb_rank_histograms",
                thresholds.min_distinct_bomb_rank_histograms_per_family,
            ),
            (
                "distinct_scout_rank_histograms",
                thresholds.min_distinct_scout_rank_histograms_per_family,
            ),
        ):
            scaled_floor = min(floor, max(1, math.ceil(floor * scale)))
            checks.append(
                _check(
                    f"{family_id}:{metric_name}",
                    family_metrics[metric_name] >= scaled_floor,
                    family_metrics[metric_name],
                    scaled_floor,
                )
            )

    for family_id, row in overlap["matrix"].items():
        checks.append(
            _check(
                f"{family_id}:self_satisfaction",
                row[family_id] >= thresholds.required_self_satisfaction,
                row[family_id],
                thresholds.required_self_satisfaction,
            )
        )

    return {
        "diversity_standard_version": DIVERSITY_STANDARD_VERSION,
        "checks": checks,
        "all_pass": all(check["pass"] for check in checks),
        "metrics": {
            "identity": identity,
            "distance": distances,
            "entropy": entropy,
            "trait_diversity": traits,
            "family_overlap": overlap,
        },
    }
