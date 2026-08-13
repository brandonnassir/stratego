"""Independent exhaustive audit of the materialized `setup_library_v1`.

Specification sources:

- `03_AGENT_3_LIBRARY_AUDIT.md` (exhaustive legality audit, count audit,
  duplicate audit, cross-split leakage audit, within-family diversity,
  between-family analysis, reflection audit, independent metric
  implementation, performance)
- `00_PHASE_7_SEQUENCE_AND_COMMON_CONTRACT.md` (diversity standard, global
  acceptance)
- Agent 1's frozen contracts in `identity/traits/families/contracts/diversity`

Audit position
--------------
Everything here is recomputed from the materialized JSONL content plus the
frozen engine and the frozen Agent 1 contracts. Agent 2's generation counters,
preflight results and manifest values are audit *subjects*, never audit
inputs: the manifest stage compares them against recomputation, and no stage
reads them to decide anything.

The auditor reuses Agent 1's authoritative family/trait/identity definitions
(explicitly permitted) but carries its own similarity implementation: a dense
blocked class-distance matrix built in this module, cross-checked pair-by-pair
against Agent 1's frozen scalar `class_distance` on a deterministic sample and
reconciled against the frozen `distance_metrics` reduction. Distances,
entropies, supports and overlaps are all recomputed from raw setups.

Failures are findings, not exceptions
-------------------------------------
A library defect must produce a recorded failure with the offending base ids,
never an auditor crash and never a repair. Per-entry checks are therefore
exception-guarded: a malformed entry is counted against every check it
prevents. Auditor exceptions are reserved for misuse of the API itself.

Thresholds are frozen inputs
----------------------------
`DIVERSITY_THRESHOLDS_V1` is evaluated exactly as Agent 1 froze it. A
threshold failure is reported as FAIL; nothing here weakens, reinterprets or
repairs anything.
"""

import json
import math
import time
from collections import Counter

import numpy as np

from ..engine.constants import BLUE, PIECE_COUNTS, RED
from ..engine.setup import (
    SetupError,
    deserialize_setup,
    serialize_setup,
    setup_to_placements,
    validate_setup,
    validate_setup_placement,
)
from .contracts import (
    BASE_ENTRY_REQUIRED_FIELDS,
    BASE_SETUP_COUNT,
    BASES_PER_FAMILY,
    DEFAULT_LIBRARY_MASTER_SEED,
    FAMILY_COUNT,
    SETUP_FAMILY_VERSION,
    SETUP_GENERATOR_CONTRACT_VERSION,
    SETUP_LIBRARY_VERSION,
    SETUP_TRAIT_VECTOR_VERSION,
    SPLITS,
    TEST_PER_FAMILY,
    TEST_TOTAL,
    TRAIN_PER_FAMILY,
    TRAIN_TOTAL,
    VALIDATION_PER_FAMILY,
    VALIDATION_TOTAL,
    base_entry_json_line,
    base_setup_id,
    parse_base_setup_id,
    split_for_base_index,
)
from .diversity import (
    DIVERSITY_THRESHOLDS_V1,
    DiversityThresholds,
    LibraryEntry,
    class_distance,
    evaluate_against_thresholds,
)
from .families import FAMILY_BY_ID, FAMILY_CONTRACTS, FAMILY_IDS, family_contract
from .identity import (
    CANONICAL_CELLS,
    SetupLibraryError,
    canonical_class_representative,
    class_fingerprint,
    content_fingerprint,
    derive_attempt_seed,
    derive_base_seed,
    is_canonical_representative,
    orient_setup,
    reflect_canonical,
)
from .library import (
    entry_metadata_digest,
    library_content_digest,
    manifest_digest,
)
from .mobility import setup_has_initial_mobility
from .traits import TRAIT_SCHEMA, UNCONVENTIONAL_FEATURES, compute_trait_vector

AUDIT_VERSION = "setup_library_audit_v1"

#: Distance strictly below which a within-family pair counts as a near
#: duplicate — Agent 1's frozen `within_family_near_duplicate_distance`.
NEAR_DUPLICATE_DISTANCE = DIVERSITY_THRESHOLDS_V1.within_family_near_duplicate_distance

#: Cross-split nearest-neighbour floor — Agent 1's frozen
#: `min_cross_split_nn_distance`.
CROSS_SPLIT_FLOOR = DIVERSITY_THRESHOLDS_V1.min_cross_split_nn_distance

#: Deterministic seed of the similarity cross-check pair sample. The sample
#: only chooses which pairs are re-verified with Agent 1's scalar metric; it
#: never influences any reported metric value.
CROSS_CHECK_SEED = 20260813
CROSS_CHECK_PAIRS = 2000

#: Descriptive percentiles reported for every distance distribution. Agent 1
#: thresholded only the minima; these are report-only context, computed by the
#: nearest-rank rule so every value is an actually observed distance.
REPORT_PERCENTILES = (1, 5, 25, 50)

_SPLIT_PAIRS = (("train", "validation"), ("train", "test"), ("validation", "test"))

#: Scalar trait fields (int/float6) used for the descriptive family centroids.
_SCALAR_TRAITS = tuple(
    field.name for field in TRAIT_SCHEMA if field.kind in ("int", "float6")
)


# ---------------------------------------------------------------------------
# Raw line / serialization audit
# ---------------------------------------------------------------------------


def line_format_audit(raw_text: str) -> dict:
    """Audit the JSONL bytes themselves against the frozen line contract.

    Every line must parse as JSON, carry every frozen required field, and be
    byte-identical to `base_entry_json_line` of its own payload — the frozen
    canonical serialization. A line that cannot even be parsed or whose setup
    cannot be deserialized is a serialization failure with its line number.
    """
    lines = [line for line in raw_text.splitlines() if line.strip()]
    unparseable: list[int] = []
    missing_fields: list[int] = []
    noncanonical: list[int] = []
    undeserializable: list[int] = []
    for number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            unparseable.append(number)
            continue
        if any(field not in payload for field in BASE_ENTRY_REQUIRED_FIELDS):
            missing_fields.append(number)
            continue
        if base_entry_json_line(payload) != line:
            noncanonical.append(number)
        try:
            setup = deserialize_setup(payload["canonical_setup"])
            validate_setup(setup, RED)
        except SetupError:
            undeserializable.append(number)
    failures = len(unparseable) + len(missing_fields) + len(noncanonical) + len(
        undeserializable
    )
    return {
        "line_count": len(lines),
        "unparseable_lines": unparseable,
        "missing_field_lines": missing_fields,
        "noncanonical_lines": noncanonical,
        "undeserializable_lines": undeserializable,
        "serialization_failures": failures,
    }


# ---------------------------------------------------------------------------
# Exhaustive per-base audit
# ---------------------------------------------------------------------------


def _engine_valid(setup: "tuple[int, ...]") -> "tuple[str | None, bool]":
    """Engine legality of one canonical arrangement, for both colours.

    Validates the row-major tuple form and the square-oriented placement form
    for red and blue, so inventory and placement are both the frozen engine's
    verdict. Returns `(error_message, placement_stage)`: the engine's message
    on the first violation (else None), and whether the violation was raised
    by the placement validator rather than the tuple validator.
    """
    for player in (RED, BLUE):
        try:
            oriented = orient_setup(setup, player)
            validate_setup(oriented, player)
        except (SetupError, SetupLibraryError) as error:
            return str(error), False
        try:
            validate_setup_placement(setup_to_placements(oriented, player), player)
        except (SetupError, SetupLibraryError) as error:
            return str(error), True
    return None, False


def per_base_audit(entries) -> dict:
    """The exhaustive per-entry audit over every supplied base.

    For every entry: exact inventory, engine legality of the base and of its
    reflection (both colours, tuple and placement forms), initial mobility of
    the base and of its reflection through the frozen engine, canonical form,
    independently recomputed fingerprints and trait vector, primary-family
    predicate on the base and on its reflection, serialization and reflection
    round trips, and identity/split/seed re-derivation from the base index.
    Returns failure id lists (all expected empty) plus recomputed trait
    vectors and family satisfaction for downstream stages.
    """
    inventory_failures: list[str] = []
    engine_failures: list[dict] = []
    reflected_engine_failures: list[dict] = []
    placement_failures: list[str] = []
    mobility_failures: list[str] = []
    reflected_mobility_failures: list[str] = []
    family_failures: list[dict] = []
    reflected_family_failures: list[dict] = []
    serialization_failures: list[str] = []
    reflection_roundtrip_failures: list[str] = []
    canonicalization_failures: list[str] = []
    fingerprint_failures: list[dict] = []
    trait_failures: list[str] = []
    identity_failures: list[dict] = []
    seed_failures: list[str] = []
    version_failures: list[str] = []
    reflection_symmetric: list[str] = []

    trait_vectors: dict[str, dict] = {}

    for entry in entries:
        identifier = entry.base_setup_id
        setup = tuple(entry.canonical_setup)

        # Exact inventory, recomputed directly rather than via the engine.
        if Counter(setup) != dict(PIECE_COUNTS):
            inventory_failures.append(identifier)

        # Engine legality: base and reflection, both colours, both forms.
        base_error, base_placement = _engine_valid(setup)
        if base_error is not None:
            engine_failures.append({"base_setup_id": identifier, "error": base_error})
            if base_placement:
                placement_failures.append(identifier)
        try:
            reflection = reflect_canonical(setup)
        except SetupLibraryError as error:
            reflection = None
            reflected_engine_failures.append(
                {"base_setup_id": identifier, "error": str(error)}
            )
        if reflection is not None:
            reflected_error, reflected_placement = _engine_valid(reflection)
            if reflected_error is not None:
                reflected_engine_failures.append(
                    {"base_setup_id": identifier, "error": reflected_error}
                )
                if reflected_placement:
                    placement_failures.append(identifier)

        # Initial mobility through the frozen engine, base and reflection.
        try:
            if not setup_has_initial_mobility(setup):
                mobility_failures.append(identifier)
        except (SetupError, SetupLibraryError):
            mobility_failures.append(identifier)
        if reflection is not None:
            try:
                if not setup_has_initial_mobility(reflection):
                    reflected_mobility_failures.append(identifier)
            except (SetupError, SetupLibraryError):
                reflected_mobility_failures.append(identifier)

        # Serialization round trip: tuple -> string -> tuple, exact.
        try:
            serialized = serialize_setup(setup)
            if deserialize_setup(serialized) != setup:
                serialization_failures.append(identifier)
        except (SetupError, SetupLibraryError):
            serialization_failures.append(identifier)

        # Reflection round trips: involution and canonicalization agreement.
        if reflection is None:
            reflection_roundtrip_failures.append(identifier)
        else:
            if reflect_canonical(reflection) != setup:
                reflection_roundtrip_failures.append(identifier)
            if setup == reflection:
                reflection_symmetric.append(identifier)

        # Canonical form: the stored arrangement is its class representative,
        # and the reflection canonicalizes back to the same stored base.
        try:
            if not is_canonical_representative(setup):
                canonicalization_failures.append(identifier)
            elif (
                canonical_class_representative(setup) != setup
                or reflection is None
                or canonical_class_representative(reflection) != setup
            ):
                canonicalization_failures.append(identifier)
        except (SetupError, SetupLibraryError):
            canonicalization_failures.append(identifier)

        # Fingerprints, recomputed from content.
        try:
            recomputed_class = class_fingerprint(setup)
            recomputed_content = content_fingerprint(setup)
            recomputed_reflected = (
                content_fingerprint(reflection) if reflection is not None else None
            )
            reflected_class = (
                class_fingerprint(reflection) if reflection is not None else None
            )
            mismatched = {}
            if recomputed_class != entry.fingerprint:
                mismatched["fingerprint"] = recomputed_class
            if reflected_class is not None and reflected_class != recomputed_class:
                mismatched["reflection_class_fingerprint"] = reflected_class
            if recomputed_content != entry.content_fingerprint:
                mismatched["content_fingerprint"] = recomputed_content
            if (
                recomputed_reflected is not None
                and recomputed_reflected != entry.reflected_content_fingerprint
            ):
                mismatched["reflected_content_fingerprint"] = recomputed_reflected
            if mismatched:
                fingerprint_failures.append(
                    {"base_setup_id": identifier, "mismatched": sorted(mismatched)}
                )
        except (SetupError, SetupLibraryError):
            fingerprint_failures.append(
                {"base_setup_id": identifier, "mismatched": ["uncomputable"]}
            )

        # Trait vector, recomputed independently and compared exactly.
        traits = None
        try:
            traits = compute_trait_vector(setup)
            if traits != entry.trait_vector:
                trait_failures.append(identifier)
        except (SetupError, SetupLibraryError):
            trait_failures.append(identifier)
        if traits is not None:
            trait_vectors[identifier] = traits

        # Primary-family predicate on the base and on its reflection.
        if traits is None:
            family_failures.append(
                {"base_setup_id": identifier, "violations": ["traits_uncomputable"]}
            )
        else:
            try:
                satisfied, violations = family_contract(entry.family_id).evaluate(traits)
            except (SetupError, SetupLibraryError) as error:
                satisfied, violations = False, [str(error)]
            if not satisfied:
                family_failures.append(
                    {"base_setup_id": identifier, "violations": violations}
                )
        if reflection is not None:
            try:
                reflected_ok, reflected_violations = family_contract(
                    entry.family_id
                ).evaluate(compute_trait_vector(reflection))
            except (SetupError, SetupLibraryError) as error:
                reflected_ok, reflected_violations = False, [str(error)]
            if not reflected_ok:
                reflected_family_failures.append(
                    {"base_setup_id": identifier, "violations": reflected_violations}
                )

        # Stable identity, split rule and seed derivation, all from identity.
        try:
            expected_id = base_setup_id(entry.family_id, entry.base_index)
            parsed = parse_base_setup_id(identifier)
            expected_split = split_for_base_index(entry.base_index)
            if (
                identifier != expected_id
                or parsed != (SETUP_LIBRARY_VERSION, entry.family_id, entry.base_index)
                or entry.split != expected_split
                or entry.family_key != FAMILY_BY_ID[entry.family_id].key
            ):
                identity_failures.append(
                    {
                        "base_setup_id": identifier,
                        "expected_id": expected_id,
                        "split": entry.split,
                        "expected_split": expected_split,
                    }
                )
        except (SetupError, SetupLibraryError) as error:
            identity_failures.append(
                {"base_setup_id": identifier, "error": str(error)}
            )

        try:
            expected_base_seed = derive_base_seed(
                SETUP_GENERATOR_CONTRACT_VERSION,
                SETUP_LIBRARY_VERSION,
                DEFAULT_LIBRARY_MASTER_SEED,
                entry.family_id,
                entry.base_index,
            )
            expected_attempt_seed = derive_attempt_seed(
                expected_base_seed, entry.accepted_attempt_index
            )
            if (
                entry.master_seed != DEFAULT_LIBRARY_MASTER_SEED
                or entry.generation_seed != expected_base_seed
                or entry.accepted_attempt_seed != expected_attempt_seed
                or entry.generation_attempts != entry.accepted_attempt_index + 1
            ):
                seed_failures.append(identifier)
        except (SetupError, SetupLibraryError):
            seed_failures.append(identifier)

        if (
            entry.library_version != SETUP_LIBRARY_VERSION
            or entry.contract_version != SETUP_GENERATOR_CONTRACT_VERSION
            or entry.family_contract_version != SETUP_FAMILY_VERSION
            or entry.trait_schema_version != SETUP_TRAIT_VECTOR_VERSION
        ):
            version_failures.append(identifier)

    entry_count = len(list(entries))
    return {
        "entry_count": entry_count,
        "inventory_failures": inventory_failures,
        "engine_failures": engine_failures,
        "reflected_engine_failures": reflected_engine_failures,
        "placement_failures": placement_failures,
        "mobility_failures": mobility_failures,
        "reflected_mobility_failures": reflected_mobility_failures,
        "family_failures": family_failures,
        "reflected_family_failures": reflected_family_failures,
        "serialization_failures": serialization_failures,
        "reflection_roundtrip_failures": reflection_roundtrip_failures,
        "canonicalization_failures": canonicalization_failures,
        "fingerprint_failures": fingerprint_failures,
        "trait_failures": trait_failures,
        "identity_failures": identity_failures,
        "seed_failures": seed_failures,
        "version_failures": version_failures,
        "reflection_symmetric_bases": reflection_symmetric,
        "trait_vectors": trait_vectors,
    }


# ---------------------------------------------------------------------------
# Count audit
# ---------------------------------------------------------------------------


def count_audit(entries) -> dict:
    """Recompute every count and compare against the frozen exact targets."""
    family_counts = {family_id: 0 for family_id in FAMILY_IDS}
    split_counts = {split: 0 for split in SPLITS}
    family_split_counts = {
        family_id: {split: 0 for split in SPLITS} for family_id in FAMILY_IDS
    }
    unknown_families: list[str] = []
    unknown_splits: list[str] = []
    for entry in entries:
        if entry.family_id in family_counts:
            family_counts[entry.family_id] += 1
        else:
            unknown_families.append(entry.base_setup_id)
        if entry.split in split_counts:
            split_counts[entry.split] += 1
        else:
            unknown_splits.append(entry.base_setup_id)
        if entry.family_id in family_split_counts and entry.split in SPLITS:
            family_split_counts[entry.family_id][entry.split] += 1

    expected_family_split = {
        "train": TRAIN_PER_FAMILY,
        "validation": VALIDATION_PER_FAMILY,
        "test": TEST_PER_FAMILY,
    }
    checks = {
        "total_exact": len(list(entries)) == BASE_SETUP_COUNT,
        "family_count_exact": len(
            [c for c in family_counts.values() if c == BASES_PER_FAMILY]
        )
        == FAMILY_COUNT
        and not unknown_families,
        "split_totals_exact": split_counts
        == {"train": TRAIN_TOTAL, "validation": VALIDATION_TOTAL, "test": TEST_TOTAL}
        and not unknown_splits,
        "family_split_exact": all(
            row == expected_family_split for row in family_split_counts.values()
        ),
    }
    return {
        "total": len(list(entries)),
        "family_counts": family_counts,
        "split_counts": split_counts,
        "family_split_counts": family_split_counts,
        "unknown_families": unknown_families,
        "unknown_splits": unknown_splits,
        "checks": checks,
        "all_exact": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# Global duplicate audit
# ---------------------------------------------------------------------------


def duplicate_audit(entries) -> dict:
    """Global duplicate detection over arrangements, classes and stable ids.

    Never limited to within-family comparisons: groups are formed over the
    whole supplied library. Reflection-equivalent duplicates are detected via
    the recomputed class fingerprint, and additionally by intersecting the
    stored arrangements with the set of all mirrored arrangements.
    """
    by_setup: dict[tuple, list[str]] = {}
    by_class: dict[str, list[str]] = {}
    by_identifier: dict[str, list[int]] = {}
    mirrored_of: dict[tuple, str] = {}

    entry_list = list(entries)
    for index, entry in enumerate(entry_list):
        setup = tuple(entry.canonical_setup)
        by_setup.setdefault(setup, []).append(entry.base_setup_id)
        try:
            fingerprint = class_fingerprint(setup)
        except (SetupError, SetupLibraryError):
            fingerprint = f"uncomputable:{entry.base_setup_id}"
        by_class.setdefault(fingerprint, []).append(entry.base_setup_id)
        by_identifier.setdefault(entry.base_setup_id, []).append(index)
        try:
            mirrored_of[reflect_canonical(setup)] = entry.base_setup_id
        except SetupLibraryError:
            pass

    exact_groups = {ids[0]: ids for ids in by_setup.values() if len(ids) > 1}
    class_groups = {ids[0]: ids for ids in by_class.values() if len(ids) > 1}

    # Same stable id -> different setup.
    same_id_different_setup: list[str] = []
    for identifier, indices in by_identifier.items():
        if len(indices) > 1:
            setups = {tuple(entry_list[i].canonical_setup) for i in indices}
            if len(setups) > 1:
                same_id_different_setup.append(identifier)

    # Different stable id -> same equivalence class.
    different_id_same_class = [
        ids for ids in by_class.values() if len(set(ids)) > 1
    ]

    # A stored arrangement that equals the mirror of another stored one.
    stored_mirror_overlap = sorted(
        {
            f"{mirrored_of[setup]}~{ids[0]}"
            for setup, ids in by_setup.items()
            if setup in mirrored_of and mirrored_of[setup] not in ids
        }
    )

    # Cross-split class duplicates: one equivalence class in two splits.
    split_of = {entry.base_setup_id: entry.split for entry in entry_list}
    cross_split_groups = [
        ids
        for ids in by_class.values()
        if len(ids) > 1 and len({split_of[i] for i in ids}) > 1
    ]

    return {
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_members": sorted(
            identifier for ids in exact_groups.values() for identifier in ids
        ),
        "reflection_class_duplicate_groups": len(class_groups),
        "reflection_class_duplicate_members": sorted(
            identifier for ids in class_groups.values() for identifier in ids
        ),
        "stable_id_collisions": sum(
            1 for indices in by_identifier.values() if len(indices) > 1
        ),
        "same_id_different_setup": sorted(same_id_different_setup),
        "different_id_same_class_groups": [sorted(g) for g in different_id_same_class],
        "stored_mirror_overlap": stored_mirror_overlap,
        "cross_split_class_duplicate_groups": len(cross_split_groups),
        "cross_split_class_duplicate_members": sorted(
            identifier for ids in cross_split_groups for identifier in ids
        ),
        "distinct_arrangements": len(by_setup),
        "distinct_class_fingerprints": len(
            {f for f in by_class if not f.startswith("uncomputable:")}
        ),
        "distinct_stable_ids": len(by_identifier),
    }


# ---------------------------------------------------------------------------
# Independent similarity / leakage audit
# ---------------------------------------------------------------------------


def _audit_distance_matrix(entry_list) -> np.ndarray:
    """This module's own dense class-distance matrix.

    Independent of `diversity._pairwise_class_distances`: distances are
    accumulated as matches-complement (`40 - matching cells`) in int16 blocks
    before the direct/mirrored minimum is taken. The diagonal is left at 0;
    reductions mask it explicitly rather than relying on a sentinel fill.
    """
    direct = np.array([entry.canonical_setup for entry in entry_list], dtype=np.uint8)
    if direct.ndim != 2 or direct.shape[1] != CANONICAL_CELLS:
        raise SetupLibraryError("similarity audit requires 40-cell canonical setups")
    mirrored = np.array(
        [reflect_canonical(tuple(entry.canonical_setup)) for entry in entry_list],
        dtype=np.uint8,
    )
    count = direct.shape[0]
    distances = np.empty((count, count), dtype=np.uint8)
    block = 512
    for start in range(0, count, block):
        stop = min(start + block, count)
        chunk = direct[start:stop, None, :]
        plain_matches = (chunk == direct[None, :, :]).sum(axis=2, dtype=np.int16)
        mirror_matches = (chunk == mirrored[None, :, :]).sum(axis=2, dtype=np.int16)
        best = np.maximum(plain_matches, mirror_matches)
        distances[start:stop] = (CANONICAL_CELLS - best).astype(np.uint8)
    return distances


def _histogram(values: np.ndarray) -> np.ndarray:
    return np.bincount(values.astype(np.int64), minlength=CANONICAL_CELLS + 1)


def _distribution_stats(histogram: np.ndarray) -> dict:
    """Exact min/percentiles/mean/max of a distance histogram (nearest rank)."""
    total = int(histogram.sum())
    if total == 0:
        return {"count": 0}
    cumulative = np.cumsum(histogram)
    nonzero = np.nonzero(histogram)[0]

    def nearest_rank(percent: float) -> int:
        rank = max(1, math.ceil(percent / 100.0 * total))
        return int(np.searchsorted(cumulative, rank, side="left"))

    return {
        "count": total,
        "min": int(nonzero[0]),
        "max": int(nonzero[-1]),
        "mean": round(
            float((np.arange(histogram.size) * histogram).sum() / total), 6
        ),
        **{f"p{percent}": nearest_rank(percent) for percent in REPORT_PERCENTILES},
        "median": nearest_rank(50),
    }


def _offending_pairs(
    distances: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    identifiers: "list[str]",
    floor: int,
) -> "list[dict]":
    """Every unordered pair below `floor` between two index sets, with ids."""
    sub = distances[np.ix_(rows, cols)]
    below = np.argwhere(sub < floor)
    seen: set[tuple[int, int]] = set()
    offenders: list[dict] = []
    for row_position, col_position in below:
        left = int(rows[row_position])
        right = int(cols[col_position])
        if left == right:
            continue
        key = (min(left, right), max(left, right))
        if key in seen:
            continue
        seen.add(key)
        offenders.append(
            {
                "a": identifiers[key[0]],
                "b": identifiers[key[1]],
                "class_distance": int(distances[key[0], key[1]]),
            }
        )
    return sorted(offenders, key=lambda item: (item["class_distance"], item["a"]))


def _pair_scope_stats(
    distances: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    identifiers: "list[str]",
) -> dict:
    """Pairwise and NN statistics between two disjoint index sets."""
    sub = distances[np.ix_(rows, cols)]
    pair_histogram = _histogram(sub.ravel())
    return {
        "pair_count": int(sub.size),
        "pairwise": _distribution_stats(pair_histogram),
        "nn_a_to_b": _distribution_stats(_histogram(sub.min(axis=1))),
        "nn_b_to_a": _distribution_stats(_histogram(sub.min(axis=0))),
        "pairs_below_cross_split_floor": int((sub < CROSS_SPLIT_FLOOR).sum()),
        "pairs_below_near_duplicate": int((sub < NEAR_DUPLICATE_DISTANCE).sum()),
        "offending_pairs": _offending_pairs(
            distances, rows, cols, identifiers, CROSS_SPLIT_FLOOR
        ),
    }


def similarity_audit(entries) -> dict:
    """Cross-split leakage and within-family distance audit, recomputed here.

    Builds this module's own class-distance matrix, cross-checks it against
    Agent 1's frozen scalar metric on a deterministic pair sample, and reports
    within-family distance/NN distributions, cross-split NN distributions per
    split pair (globally and within each family), near-duplicate counts and
    every offending pair below the frozen floors.
    """
    entry_list = list(entries)
    if len(entry_list) < 2:
        raise SetupLibraryError("similarity audit needs at least two entries")
    identifiers = [entry.base_setup_id for entry in entry_list]
    started = time.time()
    distances = _audit_distance_matrix(entry_list)
    matrix_seconds = round(time.time() - started, 3)

    count = len(entry_list)
    off_diagonal = ~np.eye(count, dtype=bool)

    # Cross-check the audit matrix against Agent 1's frozen scalar metric.
    rng = np.random.default_rng(CROSS_CHECK_SEED)
    checked = 0
    mismatches = 0
    while checked < min(CROSS_CHECK_PAIRS, count * (count - 1) // 2):
        left, right = (int(v) for v in rng.integers(0, count, size=2))
        if left == right:
            continue
        expected = class_distance(
            entry_list[left].canonical_setup, entry_list[right].canonical_setup
        )
        if int(distances[left, right]) != expected or int(
            distances[right, left]
        ) != expected:
            mismatches += 1
        checked += 1
    symmetric = bool((distances == distances.T).all())

    families = np.array(
        [
            FAMILY_IDS.index(entry.family_id) if entry.family_id in FAMILY_IDS else -1
            for entry in entry_list
        ]
    )
    splits = np.array([entry.split for entry in entry_list])

    # Global pairwise / nearest-neighbour picture.
    masked = distances.astype(np.int16)
    np.fill_diagonal(masked, CANONICAL_CELLS + 1)
    global_nn = masked.min(axis=1)
    upper = distances[np.triu_indices(count, k=1)]
    global_stats = {
        "pairwise": _distribution_stats(_histogram(upper)),
        "nn": _distribution_stats(_histogram(global_nn.astype(np.int64))),
        "min_pairwise_distance": int(upper.min()),
    }

    # Within-family distances.
    within_family: dict[str, dict] = {}
    for family_index, family_id in enumerate(FAMILY_IDS):
        member_indices = np.nonzero(families == family_index)[0]
        if member_indices.size < 2:
            continue
        sub = distances[np.ix_(member_indices, member_indices)]
        sub_masked = sub.astype(np.int16)
        np.fill_diagonal(sub_masked, CANONICAL_CELLS + 1)
        pair_values = sub[np.triu_indices(member_indices.size, k=1)]
        near_duplicates = int((pair_values < NEAR_DUPLICATE_DISTANCE).sum())
        within_family[family_id] = {
            "member_count": int(member_indices.size),
            "pairwise": _distribution_stats(_histogram(pair_values)),
            "nn": _distribution_stats(
                _histogram(sub_masked.min(axis=1).astype(np.int64))
            ),
            "min_nn_distance": int(sub_masked.min()),
            "near_duplicate_pairs": near_duplicates,
            "near_duplicate_pair_fraction": round(
                near_duplicates / pair_values.size, 8
            ),
            "offending_pairs": _offending_pairs(
                distances,
                member_indices,
                member_indices,
                identifiers,
                DIVERSITY_THRESHOLDS_V1.min_within_family_nn_distance,
            ),
        }

    # Cross-split leakage: global and per family, every split pair.
    cross_split: dict[str, dict] = {}
    for split_a, split_b in _SPLIT_PAIRS:
        rows = np.nonzero(splits == split_a)[0]
        cols = np.nonzero(splits == split_b)[0]
        if not rows.size or not cols.size:
            continue
        scope = {"global": _pair_scope_stats(distances, rows, cols, identifiers)}
        for family_index, family_id in enumerate(FAMILY_IDS):
            family_rows = rows[families[rows] == family_index]
            family_cols = cols[families[cols] == family_index]
            if not family_rows.size or not family_cols.size:
                continue
            scope[family_id] = _pair_scope_stats(
                distances, family_rows, family_cols, identifiers
            )
        cross_split[f"{split_a}__{split_b}"] = scope

    any_cross = splits[:, None] != splits[None, :]
    cross_values = distances[any_cross]
    cross_split_min = int(cross_values.min()) if cross_values.size else None

    ordered_pairs = count * count - count
    return {
        "method": (
            "dense blocked numpy class-distance matrix (uint8), matches-"
            "complement formulation, direct and mirrored orientations; "
            "independent of diversity._pairwise_class_distances"
        ),
        "matrix_shape": [count, count],
        "ordered_pair_comparisons": ordered_pairs,
        "unordered_pairs": ordered_pairs // 2,
        "cell_comparisons": ordered_pairs * CANONICAL_CELLS * 2,
        "matrix_seconds": matrix_seconds,
        "cross_check": {
            "sampled_pairs": checked,
            "mismatches_vs_frozen_metric": mismatches,
            "matrix_symmetric": symmetric,
            "seed": CROSS_CHECK_SEED,
        },
        "global": global_stats,
        "within_family": within_family,
        "cross_split": cross_split,
        "cross_split_min_nn_distance": cross_split_min,
        "global_min_pairwise_distance": int(upper.min()),
    }


# ---------------------------------------------------------------------------
# Frozen-threshold audit
# ---------------------------------------------------------------------------


def threshold_audit(entries, thresholds: DiversityThresholds = DIVERSITY_THRESHOLDS_V1) -> dict:
    """Evaluate Agent 1's frozen diversity standard, exactly as frozen.

    Delegates to the frozen `evaluate_against_thresholds` (the authoritative
    implementation Agent 1 shipped and tested), then reports every check as
    `metric / required / measured / pass`. Nothing is weakened or scaled here;
    a failure is reported, never repaired.
    """
    library_entries = [
        LibraryEntry(
            family_id=entry.family_id,
            split=entry.split,
            canonical=tuple(entry.canonical_setup),
        )
        for entry in entries
    ]
    evaluation = evaluate_against_thresholds(library_entries, thresholds)
    failed = [check for check in evaluation["checks"] if not check["pass"]]
    return {
        "diversity_standard_version": evaluation["diversity_standard_version"],
        "check_count": len(evaluation["checks"]),
        "checks": evaluation["checks"],
        "failed_checks": failed,
        "all_pass": evaluation["all_pass"],
        "metrics": evaluation["metrics"],
    }


def similarity_cross_check(similarity: dict, threshold_metrics: dict) -> dict:
    """Reconcile the audit-side distance values with the frozen reduction.

    The independent matrix and Agent 1's frozen `distance_metrics` must agree
    on every thresholded number: per-family minimum NN distance and near-
    duplicate fraction, the global cross-split NN minimum, and the global
    pairwise minimum. Disagreement would mean one implementation is wrong and
    the audit cannot stand; agreement is a recorded gate.
    """
    frozen = threshold_metrics["distance"]
    disagreements: list[str] = []
    for family_id, frozen_family in frozen["within_family"].items():
        audit_family = similarity["within_family"].get(family_id)
        if audit_family is None:
            disagreements.append(f"{family_id}: missing from audit-side matrix")
            continue
        if audit_family["min_nn_distance"] != frozen_family["min_nn_distance"]:
            disagreements.append(
                f"{family_id}: min NN {audit_family['min_nn_distance']} "
                f"!= frozen {frozen_family['min_nn_distance']}"
            )
        if round(audit_family["near_duplicate_pair_fraction"], 8) != round(
            frozen_family["near_duplicate_pair_fraction"], 8
        ):
            disagreements.append(f"{family_id}: near-duplicate fraction differs")
    if similarity["cross_split_min_nn_distance"] != frozen["cross_split_min_nn_distance"]:
        disagreements.append("cross-split minimum differs")
    if similarity["global_min_pairwise_distance"] != frozen["global_min_pairwise_distance"]:
        disagreements.append("global pairwise minimum differs")
    return {"agrees": not disagreements, "disagreements": disagreements}


# ---------------------------------------------------------------------------
# Between-family overlap / confusion audit
# ---------------------------------------------------------------------------


def overlap_audit(entries, trait_vectors: "dict[str, dict] | None" = None) -> dict:
    """The full descriptive family overlap/confusion matrix, with attribution.

    `matrix[i][j]` is the fraction of family `i` bases whose recomputed trait
    vector satisfies family `j`'s frozen contract. The diagonal is a hard gate
    (exactly 1.0); off-diagonal overlap is expected — families are not
    disjoint by design — and is reported descriptively. For every off-diagonal
    cell at or above 0.25, the audit attributes the overlap clause-by-clause:
    the satisfaction rate of each of family `j`'s required and forbidden
    clauses among family `i`'s members, and for F15 the unconventional-feature
    incidence, so a large overlap is explained rather than merely reported.
    """
    entry_list = list(entries)
    by_family: dict[str, list] = {family_id: [] for family_id in FAMILY_IDS}
    vectors: dict[str, dict] = {}
    for entry in entry_list:
        if entry.family_id not in by_family:
            continue
        by_family[entry.family_id].append(entry)
        if trait_vectors is not None and entry.base_setup_id in trait_vectors:
            vectors[entry.base_setup_id] = trait_vectors[entry.base_setup_id]
        else:
            vectors[entry.base_setup_id] = compute_trait_vector(
                tuple(entry.canonical_setup)
            )

    matrix_fraction: dict[str, dict[str, float]] = {}
    matrix_count: dict[str, dict[str, int]] = {}
    for family_id, members in by_family.items():
        if not members:
            continue
        fraction_row: dict[str, float] = {}
        count_row: dict[str, int] = {}
        for other in FAMILY_CONTRACTS:
            satisfied = sum(
                1
                for member in members
                if other.evaluate(vectors[member.base_setup_id])[0]
            )
            count_row[other.family_id] = satisfied
            fraction_row[other.family_id] = round(satisfied / len(members), 6)
        matrix_fraction[family_id] = fraction_row
        matrix_count[family_id] = count_row

    diagonal_failures = [
        family_id
        for family_id, row in matrix_fraction.items()
        if row.get(family_id) != 1.0
    ]

    off_diagonal = sorted(
        (
            {
                "declared_family": source,
                "also_satisfies": target,
                "fraction": fraction,
                "count": matrix_count[source][target],
            }
            for source, row in matrix_fraction.items()
            for target, fraction in row.items()
            if source != target and fraction > 0.0
        ),
        key=lambda item: (-item["fraction"], item["declared_family"], item["also_satisfies"]),
    )

    # Clause-level attribution for every substantial overlap.
    attributions: list[dict] = []
    for item in off_diagonal:
        if item["fraction"] < 0.25:
            continue
        source, target = item["declared_family"], item["also_satisfies"]
        members = by_family[source]
        contract = FAMILY_BY_ID[target]
        clause_rates: dict[str, float] = {}
        for clause in contract.required:
            passing = sum(
                1
                for member in members
                if clause.evaluate(vectors[member.base_setup_id])
            )
            clause_rates[f"required:{clause.name}"] = round(passing / len(members), 6)
        for clause in contract.forbidden:
            firing = sum(
                1
                for member in members
                if clause.evaluate(vectors[member.base_setup_id])
            )
            clause_rates[f"forbidden:{clause.name}"] = round(firing / len(members), 6)
        attribution = {
            **item,
            "target_clause_rates": clause_rates,
        }
        if target == "F15":
            satisfying = [
                member
                for member in members
                if FAMILY_BY_ID["F15"].evaluate(vectors[member.base_setup_id])[0]
            ]
            feature_counts = Counter()
            for member in satisfying:
                traits = vectors[member.base_setup_id]
                for name, active in _feature_flags(traits).items():
                    if active:
                        feature_counts[name] += 1
            attribution["f15_feature_incidence_among_satisfying"] = {
                name: round(feature_counts.get(name, 0) / len(satisfying), 6)
                if satisfying
                else 0.0
                for name, _ in UNCONVENTIONAL_FEATURES
            }
            attribution["satisfying_count"] = len(satisfying)
        attributions.append(attribution)

    centroids = _trait_centroids(by_family, vectors)

    return {
        "matrix": matrix_fraction,
        "matrix_counts": matrix_count,
        "diagonal_failures": diagonal_failures,
        "off_diagonal_overlaps": off_diagonal,
        "largest_off_diagonal": off_diagonal[0] if off_diagonal else None,
        "attributions": attributions,
        "trait_centroids": centroids,
    }


def _feature_flags(traits: dict) -> dict:
    """Reconstruct the eight F15 feature booleans from a trait vector.

    The trait vector stores only the feature count; the audit re-derives each
    individual flag from the same trait fields the schema defines, so feature
    incidence can be attributed without re-walking the raw setups.
    """
    return {
        "flag_forward": traits["flag_rank"] >= 2,
        "flag_unguarded": traits["flag_orth_bomb_guards"] == 0,
        "bombs_on_front_rank": traits["bomb_front_rank_count"] >= 3,
        "marshal_on_front_rank": traits["marshal_rank"] == 3,
        "general_on_front_rank": traits["general_rank"] == 3,
        "no_front_rank_scouts": traits["scout_front_rank_count"] == 0,
        "miners_on_front_rank": traits["miner_front_rank_count"] >= 3,
        "spy_on_front_rank": traits["spy_rank"] == 3,
    }


def _trait_centroids(by_family: dict, vectors: dict) -> dict:
    """Per-family mean and standard deviation of every scalar trait."""
    centroids: dict[str, dict] = {}
    for family_id, members in by_family.items():
        if not members:
            continue
        family_stats: dict[str, dict] = {}
        for trait in _SCALAR_TRAITS:
            values = np.array(
                [vectors[member.base_setup_id][trait] for member in members],
                dtype=np.float64,
            )
            family_stats[trait] = {
                "mean": round(float(values.mean()), 6),
                "std": round(float(values.std()), 6),
            }
        centroids[family_id] = family_stats
    return centroids


# ---------------------------------------------------------------------------
# Manifest audit
# ---------------------------------------------------------------------------


def manifest_audit(entries, manifest: dict, expected_digests: "dict | None" = None) -> dict:
    """Compare the production manifest against full recomputation.

    Digests are recomputed from the supplied entries; the manifest's own
    digest is recomputed over its identity fields; counts and versions are
    compared against the frozen contract; and, when Agent 2's handoff digests
    are supplied, the recomputed values must match them exactly.
    """
    recomputed_library = library_content_digest(entries)
    recomputed_metadata = entry_metadata_digest(entries)
    recomputed_manifest = manifest_digest(manifest)

    checks = {
        "library_content_digest_matches": manifest.get("library_content_digest")
        == recomputed_library,
        "entry_metadata_digest_matches": manifest.get("entry_metadata_digest")
        == recomputed_metadata,
        "manifest_digest_matches": manifest.get("manifest_digest")
        == recomputed_manifest,
        "entry_count_matches": manifest.get("entry_count") == len(list(entries)),
        "family_counts_match": manifest.get("family_counts")
        == count_audit(entries)["family_counts"],
        "split_counts_match": manifest.get("split_counts")
        == count_audit(entries)["split_counts"],
        "versions_match": (
            manifest.get("library_version") == SETUP_LIBRARY_VERSION
            and manifest.get("generator_contract_version")
            == SETUP_GENERATOR_CONTRACT_VERSION
            and manifest.get("family_version") == SETUP_FAMILY_VERSION
            and manifest.get("trait_schema_version") == SETUP_TRAIT_VECTOR_VERSION
            and manifest.get("master_seed") == DEFAULT_LIBRARY_MASTER_SEED
        ),
    }
    if expected_digests is not None:
        checks["handoff_library_digest_matches"] = (
            expected_digests.get("library_content_digest") == recomputed_library
        )
        checks["handoff_metadata_digest_matches"] = (
            expected_digests.get("entry_metadata_digest") == recomputed_metadata
        )
        checks["handoff_manifest_digest_matches"] = (
            expected_digests.get("manifest_digest") == recomputed_manifest
        )
    return {
        "recomputed": {
            "library_content_digest": recomputed_library,
            "entry_metadata_digest": recomputed_metadata,
            "manifest_digest": recomputed_manifest,
        },
        "checks": checks,
        "all_pass": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# Full audit
# ---------------------------------------------------------------------------


def _gate(metric: str, required: str, measured, passed: bool) -> dict:
    return {
        "metric": metric,
        "required": required,
        "measured": measured,
        "pass": bool(passed),
    }


def audit_library(
    entries,
    manifest: "dict | None" = None,
    raw_text: "str | None" = None,
    expected_digests: "dict | None" = None,
    thresholds: DiversityThresholds = DIVERSITY_THRESHOLDS_V1,
) -> dict:
    """The complete independent audit of a materialized base library.

    Runs every stage — line format, per-base legality/identity, counts,
    global duplicates, independent similarity/leakage, the frozen diversity
    thresholds, the family overlap matrix, and (when supplied) the manifest —
    and assembles every hard gate as `metric / required / measured / pass`.
    Returns a verdict of PASS only if every hard gate and every frozen
    threshold check passes. A failing library yields FAIL with the offending
    ids recorded; nothing is repaired.
    """
    entry_list = list(entries)
    durations: dict[str, float] = {}

    def timed(name, function, *arguments, **keywords):
        started = time.time()
        value = function(*arguments, **keywords)
        durations[name] = round(time.time() - started, 3)
        return value

    line_format = (
        timed("line_format", line_format_audit, raw_text) if raw_text is not None else None
    )
    per_base = timed("per_base", per_base_audit, entry_list)
    counts = timed("counts", count_audit, entry_list)
    duplicates = timed("duplicates", duplicate_audit, entry_list)
    similarity = timed("similarity", similarity_audit, entry_list)
    threshold_result = timed("thresholds", threshold_audit, entry_list, thresholds)
    reconciliation = similarity_cross_check(similarity, threshold_result["metrics"])
    overlap = timed(
        "overlap", overlap_audit, entry_list, per_base["trait_vectors"]
    )
    manifest_result = (
        timed("manifest", manifest_audit, entry_list, manifest, expected_digests)
        if manifest is not None
        else None
    )

    total = len(entry_list)
    gates = [
        _gate("total_bases", f"== {BASE_SETUP_COUNT}", total, total == BASE_SETUP_COUNT),
        _gate("family_counts", f"== {BASES_PER_FAMILY} x {FAMILY_COUNT}",
              counts["family_counts"], counts["checks"]["family_count_exact"]),
        _gate("split_totals",
              f"== {TRAIN_TOTAL}/{VALIDATION_TOTAL}/{TEST_TOTAL}",
              counts["split_counts"], counts["checks"]["split_totals_exact"]),
        _gate("family_split_counts",
              f"== {TRAIN_PER_FAMILY}/{VALIDATION_PER_FAMILY}/{TEST_PER_FAMILY} per family",
              "exact" if counts["checks"]["family_split_exact"] else counts["family_split_counts"],
              counts["checks"]["family_split_exact"]),
        _gate("base_engine_validation_failures", "== 0",
              len(per_base["engine_failures"]), not per_base["engine_failures"]),
        _gate("reflected_engine_validation_failures", "== 0",
              len(per_base["reflected_engine_failures"]),
              not per_base["reflected_engine_failures"]),
        _gate("inventory_failures", "== 0",
              len(per_base["inventory_failures"]), not per_base["inventory_failures"]),
        _gate("placement_failures", "== 0",
              len(per_base["placement_failures"]), not per_base["placement_failures"]),
        _gate("initial_mobility_failures", "== 0",
              len(per_base["mobility_failures"]), not per_base["mobility_failures"]),
        _gate("reflected_mobility_failures", "== 0",
              len(per_base["reflected_mobility_failures"]),
              not per_base["reflected_mobility_failures"]),
        _gate("family_predicate_failures", "== 0",
              len(per_base["family_failures"]), not per_base["family_failures"]),
        _gate("reflected_family_predicate_failures", "== 0",
              len(per_base["reflected_family_failures"]),
              not per_base["reflected_family_failures"]),
        _gate("serialization_failures", "== 0",
              len(per_base["serialization_failures"]),
              not per_base["serialization_failures"]),
        _gate("reflection_roundtrip_failures", "== 0",
              len(per_base["reflection_roundtrip_failures"]),
              not per_base["reflection_roundtrip_failures"]),
        _gate("canonicalization_failures", "== 0",
              len(per_base["canonicalization_failures"]),
              not per_base["canonicalization_failures"]),
        _gate("fingerprint_failures", "== 0",
              len(per_base["fingerprint_failures"]), not per_base["fingerprint_failures"]),
        _gate("trait_vector_failures", "== 0",
              len(per_base["trait_failures"]), not per_base["trait_failures"]),
        _gate("identity_or_split_failures", "== 0",
              len(per_base["identity_failures"]), not per_base["identity_failures"]),
        _gate("seed_derivation_failures", "== 0",
              len(per_base["seed_failures"]), not per_base["seed_failures"]),
        _gate("version_field_failures", "== 0",
              len(per_base["version_failures"]), not per_base["version_failures"]),
        _gate("exact_duplicate_groups", "== 0",
              duplicates["exact_duplicate_groups"],
              duplicates["exact_duplicate_groups"] == 0),
        _gate("reflection_class_duplicate_groups", "== 0",
              duplicates["reflection_class_duplicate_groups"],
              duplicates["reflection_class_duplicate_groups"] == 0),
        _gate("stable_id_collisions", "== 0",
              duplicates["stable_id_collisions"],
              duplicates["stable_id_collisions"] == 0),
        _gate("same_id_different_setup", "== 0",
              len(duplicates["same_id_different_setup"]),
              not duplicates["same_id_different_setup"]),
        _gate("different_id_same_class_groups", "== 0",
              len(duplicates["different_id_same_class_groups"]),
              not duplicates["different_id_same_class_groups"]),
        _gate("stored_mirror_overlap", "== 0",
              len(duplicates["stored_mirror_overlap"]),
              not duplicates["stored_mirror_overlap"]),
        _gate("cross_split_class_duplicate_groups", "== 0",
              duplicates["cross_split_class_duplicate_groups"],
              duplicates["cross_split_class_duplicate_groups"] == 0),
        _gate("cross_split_min_nn_distance", f">= {CROSS_SPLIT_FLOOR}",
              similarity["cross_split_min_nn_distance"],
              similarity["cross_split_min_nn_distance"] is not None
              and similarity["cross_split_min_nn_distance"] >= CROSS_SPLIT_FLOOR),
        _gate("global_min_pairwise_distance",
              f">= {thresholds.min_global_pairwise_distance}",
              similarity["global_min_pairwise_distance"],
              similarity["global_min_pairwise_distance"]
              >= thresholds.min_global_pairwise_distance),
        _gate("similarity_matrix_cross_check", "0 mismatches, symmetric",
              similarity["cross_check"],
              similarity["cross_check"]["mismatches_vs_frozen_metric"] == 0
              and similarity["cross_check"]["matrix_symmetric"]),
        _gate("similarity_reduction_reconciliation", "audit == frozen metric",
              reconciliation, reconciliation["agrees"]),
        _gate("frozen_diversity_thresholds",
              f"all {threshold_result['check_count']} checks pass",
              f"{threshold_result['check_count'] - len(threshold_result['failed_checks'])}"
              f" / {threshold_result['check_count']}",
              threshold_result["all_pass"]),
        _gate("family_self_satisfaction_diagonal", "== 1.0 for all 16",
              overlap["diagonal_failures"] or "1.0 x 16",
              not overlap["diagonal_failures"]),
        _gate("reflection_symmetric_bases", "reported (0 possible for legal setups)",
              len(per_base["reflection_symmetric_bases"]), True),
    ]
    if line_format is not None:
        gates.append(
            _gate("jsonl_line_format_failures", "== 0",
                  line_format["serialization_failures"],
                  line_format["serialization_failures"] == 0)
        )
    if manifest_result is not None:
        gates.append(
            _gate("manifest_and_handoff_digests", "all recomputed digests match",
                  manifest_result["checks"], manifest_result["all_pass"])
        )

    all_gates_pass = all(gate["pass"] for gate in gates)
    status = "PASS" if all_gates_pass and threshold_result["all_pass"] else "FAIL"

    per_base_report = {
        key: value for key, value in per_base.items() if key != "trait_vectors"
    }
    return {
        "audit_version": AUDIT_VERSION,
        "status": status,
        "gates": gates,
        "gates_true": sum(1 for gate in gates if gate["pass"]),
        "gates_total": len(gates),
        "line_format": line_format,
        "per_base": per_base_report,
        "counts": counts,
        "duplicates": duplicates,
        "similarity": similarity,
        "thresholds": threshold_result,
        "similarity_reconciliation": reconciliation,
        "overlap": overlap,
        "manifest": manifest_result,
        "durations": durations,
    }
