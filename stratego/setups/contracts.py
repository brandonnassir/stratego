"""Phase 7 setup-library contract: `setup_generator_contract_v1`.

Specification sources:

- `00_PHASE_7_SEQUENCE_AND_COMMON_CONTRACT.md` (library target, splits,
  reflection rule, determinism and identity, trajectory/persistence)
- `01_AGENT_1_SETUP_CONTRACT_AND_TAXONOMY.md` (fixed library size and split,
  versioned contracts, perturbation invariants, serialization)

This module is the single authoritative statement of the Phase 7 library
constants: version identifiers, exact counts, the deterministic split rule,
stable semantic identifiers, the master-seed input contract, the production
serialization format, the library digest, the isolated-rebuild sample rule,
and the procedural-perturbation invariants. Agents 2-6 read these values;
they do not restate them.

A future semantic change to any of these definitions requires a new version
identifier, never a silent reinterpretation of the `v1` names.
"""

import json
import hashlib

from ..engine.constants import FLAG, IMPLEMENTATION_VERSION, RULES_VERSION
from ..engine.setup import validate_setup
from .families import FAMILY_BY_ID, FAMILY_CONTRACT_VERSION, families_document, family_contract
from .identity import (
    CLASS_FINGERPRINT_DOMAIN,
    CONTENT_FINGERPRINT_DOMAIN,
    SetupLibraryError,
)
from .mobility import setup_has_initial_mobility
from .traits import TRAIT_SCHEMA_VERSION, compute_trait_vector, trait_schema_document

# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

SETUP_GENERATOR_CONTRACT_VERSION = "setup_generator_contract_v1"
SETUP_LIBRARY_VERSION = "setup_library_v1"
SETUP_TRAIT_VECTOR_VERSION = TRAIT_SCHEMA_VERSION  # re-export for one import site
SETUP_FAMILY_VERSION = FAMILY_CONTRACT_VERSION

# ---------------------------------------------------------------------------
# Fixed library size and split (exact, frozen)
# ---------------------------------------------------------------------------

FAMILY_COUNT = 16
BASES_PER_FAMILY = 500
BASE_SETUP_COUNT = FAMILY_COUNT * BASES_PER_FAMILY
assert BASE_SETUP_COUNT == 8000

TRAIN_PER_FAMILY = 400
VALIDATION_PER_FAMILY = 50
TEST_PER_FAMILY = 50
assert TRAIN_PER_FAMILY + VALIDATION_PER_FAMILY + TEST_PER_FAMILY == BASES_PER_FAMILY

TRAIN_TOTAL = TRAIN_PER_FAMILY * FAMILY_COUNT
VALIDATION_TOTAL = VALIDATION_PER_FAMILY * FAMILY_COUNT
TEST_TOTAL = TEST_PER_FAMILY * FAMILY_COUNT
assert (TRAIN_TOTAL, VALIDATION_TOTAL, TEST_TOTAL) == (6400, 800, 800)

SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_TEST = "test"
SPLITS = (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)


def split_for_base_index(base_index: int) -> str:
    """The permanent split of a base index within its family.

    Contiguous index ranges: `0..399` train, `400..449` validation,
    `450..499` test. The rule is a pure function of the index, decided before
    any setup exists, so split assignment can never react to content, game
    strength or model results, and isolated regeneration recovers the split
    without any lookup table. Reflections and perturbed descendants inherit
    the base split.
    """
    if not 0 <= base_index < BASES_PER_FAMILY:
        raise SetupLibraryError(
            f"base_index must be in 0..{BASES_PER_FAMILY - 1}, got {base_index}"
        )
    if base_index < TRAIN_PER_FAMILY:
        return SPLIT_TRAIN
    if base_index < TRAIN_PER_FAMILY + VALIDATION_PER_FAMILY:
        return SPLIT_VALIDATION
    return SPLIT_TEST


# ---------------------------------------------------------------------------
# Stable semantic identity
# ---------------------------------------------------------------------------


def base_setup_id(family_id: str, base_index: int) -> str:
    """The stable semantic identifier of one base setup.

    `setup_library_v1:F07:042` — library version, family id, zero-padded base
    index. Independent of process enumeration order and of content; content
    identity is the separate reflection-class fingerprint.
    """
    if family_id not in FAMILY_BY_ID:
        raise SetupLibraryError(f"unknown family id: {family_id!r}")
    if not 0 <= base_index < BASES_PER_FAMILY:
        raise SetupLibraryError(
            f"base_index must be in 0..{BASES_PER_FAMILY - 1}, got {base_index}"
        )
    return f"{SETUP_LIBRARY_VERSION}:{family_id}:{base_index:03d}"


def parse_base_setup_id(identifier: str) -> tuple[str, str, int]:
    """Inverse of :func:`base_setup_id`: `(library_version, family_id, index)`."""
    parts = identifier.split(":")
    if len(parts) != 3:
        raise SetupLibraryError(f"malformed base_setup_id: {identifier!r}")
    library_version, family_id, index_text = parts
    if library_version != SETUP_LIBRARY_VERSION:
        raise SetupLibraryError(
            f"unknown library version in base_setup_id: {identifier!r}"
        )
    if family_id not in FAMILY_BY_ID:
        raise SetupLibraryError(f"unknown family id in base_setup_id: {identifier!r}")
    if len(index_text) != 3 or not index_text.isdigit():
        raise SetupLibraryError(f"malformed base index in base_setup_id: {identifier!r}")
    base_index = int(index_text)
    if not 0 <= base_index < BASES_PER_FAMILY:
        raise SetupLibraryError(f"base index out of range in base_setup_id: {identifier!r}")
    return library_version, family_id, base_index


# ---------------------------------------------------------------------------
# Master seed input contract
# ---------------------------------------------------------------------------

#: The canonical master seed of `setup_library_v1`. Follows the Phase 4
#: date-seed precedent (`DEFAULT_BANK_ROOT_SEED = 20260101`). Frozen with the
#: contract: changing it changes every base setup, so a different value is a
#: different library.
DEFAULT_LIBRARY_MASTER_SEED = 20260813

# ---------------------------------------------------------------------------
# Production artifact locations and serialization
# ---------------------------------------------------------------------------

#: The materialized production library lives outside `reports/`, per the
#: Phase 7 common contract's preferred paths.
LIBRARY_JSONL_PATH = "data/setups/setup_library_v1.jsonl"
LIBRARY_MANIFEST_PATH = "data/setups/setup_library_v1_manifest.json"

#: Fields every base-library JSONL entry must carry, in the exact serialized
#: form produced by :func:`base_entry_json_line`.
BASE_ENTRY_REQUIRED_FIELDS = (
    "base_setup_id",
    "library_version",
    "contract_version",
    "family_contract_version",
    "trait_schema_version",
    "family_id",
    "family_key",
    "base_index",
    "split",
    "canonical_setup",
    "fingerprint",
    "generation_seed",
    "generation_attempts",
    "trait_vector",
)


def base_entry_json_line(entry: dict) -> str:
    """The canonical single-line JSON form of one base-library entry.

    Sorted keys, compact separators — the same canonical JSON convention the
    Phase 4 bank froze — so the JSONL file bytes are a deterministic function
    of the entries and can be digest-compared across regenerations.
    """
    missing = [field for field in BASE_ENTRY_REQUIRED_FIELDS if field not in entry]
    if missing:
        raise SetupLibraryError(f"base entry missing required fields: {missing}")
    return json.dumps(entry, sort_keys=True, separators=(",", ":"))


def library_digest(fingerprint_pairs: "list[tuple[str, str]]") -> str:
    """SHA-256 identity of a whole library.

    `fingerprint_pairs` is `[(base_setup_id, fingerprint), ...]` in canonical
    library order — families `F00..F15`, base index ascending within each
    family. The digest is over newline-joined `id:fingerprint` lines, so it is
    invariant to serialization cosmetics but pins every identity and every
    content class in order.
    """
    payload = "\n".join(
        f"{identifier}:{fingerprint}" for identifier, fingerprint in fingerprint_pairs
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Isolated-rebuild sample rule
# ---------------------------------------------------------------------------


def isolated_rebuild_sample_indices() -> tuple[int, ...]:
    """The fixed per-family base indices Agent 2 must rebuild in isolation.

    Forty indices per family: the head of train, both sides of the
    train/validation boundary, both sides of the validation/test boundary,
    and the tail of test — 640 isolated rebuilds across the library, every
    one required to match the materialized entry exactly.
    """
    return tuple(range(0, 10)) + tuple(range(395, 405)) + tuple(range(445, 455)) + tuple(
        range(490, 500)
    )


# ---------------------------------------------------------------------------
# Procedural-perturbation invariants (for Agent 4, frozen now)
# ---------------------------------------------------------------------------

#: A perturbed descendant must differ from its base by at least this many of
#: the 40 canonical squares (2 is the minimum any piece swap can produce).
PERTURBATION_MIN_HAMMING = 2
#: ... and by at most this many, so a descendant keeps its base's strategic
#: character instead of becoming an unrelated arrangement.
PERTURBATION_MAX_HAMMING = 12

#: The invariant list itself, machine-readable for the contract artifact.
PERTURBATION_INVARIANTS = (
    "exact piece inventory is preserved (engine validate_setup passes)",
    "occupancy stays inside the 40 canonical setup cells (representation guarantees this)",
    "canonical own-orientation semantics are unchanged",
    "the descendant inherits the base setup's split verbatim",
    "the descendant inherits the base setup's primary family verbatim",
    "every required family clause still holds and every forbidden clause still fails",
    "the Flag occupies the same canonical cell as in the base setup",
    "the descendant has initial mobility under the frozen engine",
    f"canonical Hamming distance from the base is in [{PERTURBATION_MIN_HAMMING}, {PERTURBATION_MAX_HAMMING}]",
    "the descendant is a pure function of (base_setup_id, sampler_version, perturbation_seed)",
    "provenance records base_setup_id, split, family, reflection flag and perturbation seed",
    "a descendant never becomes a new base-library identity",
)


def validate_perturbation(
    base_canonical: "list[int] | tuple[int, ...]",
    candidate_canonical: "list[int] | tuple[int, ...]",
    family_id: str,
) -> list[str]:
    """Every perturbation-invariant violation of `candidate`, as strings.

    The executable form of the invariant list: Agent 4's sampler must produce
    only candidates for which this returns `[]`, and Agent 3/6 can audit any
    descendant independently. An empty list is the only acceptance.
    """
    base = tuple(base_canonical)
    candidate = tuple(candidate_canonical)
    violations: list[str] = []

    try:
        validate_setup(candidate, 0)
    except Exception as error:  # engine names the violated legality condition
        violations.append(f"inventory/legality: {error}")
        return violations

    if len(base) != len(candidate):  # pragma: no cover - both validated above
        violations.append("length mismatch")
        return violations

    hamming = sum(1 for a, b in zip(base, candidate) if a != b)
    if hamming < PERTURBATION_MIN_HAMMING:
        violations.append(
            f"hamming distance {hamming} below minimum {PERTURBATION_MIN_HAMMING} "
            "(an unchanged output must be recorded as unperturbed, not perturbed)"
        )
    if hamming > PERTURBATION_MAX_HAMMING:
        violations.append(
            f"hamming distance {hamming} above maximum {PERTURBATION_MAX_HAMMING}"
        )

    if base.index(FLAG) != candidate.index(FLAG):
        violations.append("the Flag moved off its base cell")

    satisfied, clause_violations = family_contract(family_id).evaluate(
        compute_trait_vector(candidate)
    )
    if not satisfied:
        violations.extend(
            f"family {family_id} clause violated: {name}" for name in clause_violations
        )

    if not setup_has_initial_mobility(candidate):
        violations.append("stranded: no initial legal move for the owner")

    return violations


# ---------------------------------------------------------------------------
# The complete machine-readable contract document
# ---------------------------------------------------------------------------


def contract_document() -> dict:
    """The full `setup_generator_contract_v1` document.

    This is the content Agent 1 freezes into
    `reports/phase_7_data/agent_01_setup_contract.json` and the definition
    Agents 2-6 implement against. Everything in it is restated from the code
    in this package, so the artifact can be regenerated and diffed at any
    time.
    """
    return {
        "contract_version": SETUP_GENERATOR_CONTRACT_VERSION,
        "library_version": SETUP_LIBRARY_VERSION,
        "family_contract_version": SETUP_FAMILY_VERSION,
        "trait_schema_version": SETUP_TRAIT_VECTOR_VERSION,
        "frozen_stack": {
            "rules_version": RULES_VERSION,
            "reference_engine": IMPLEMENTATION_VERSION,
        },
        "library_target": {
            "base_setup_count": BASE_SETUP_COUNT,
            "family_count": FAMILY_COUNT,
            "bases_per_family": BASES_PER_FAMILY,
            "train_per_family": TRAIN_PER_FAMILY,
            "validation_per_family": VALIDATION_PER_FAMILY,
            "test_per_family": TEST_PER_FAMILY,
            "train_total": TRAIN_TOTAL,
            "validation_total": VALIDATION_TOTAL,
            "test_total": TEST_TOTAL,
        },
        "canonical_representation": {
            "frame": (
                "canonical own-orientation: rank 0 = own back row (furthest "
                "from the lakes), rank 3 = own front row (nearest the lakes), "
                "file 0..9 left to right, canonical index = rank * 10 + file"
            ),
            "storage": (
                "40 engine piece types in canonical row-major order; the "
                "frozen engine's orient map (identity for red, rank-row "
                "reversal for blue) produces per-player setup tuples"
            ),
            "serialization": (
                "the engine's 40-character piece-code string "
                "(stratego.engine.setup.serialize_setup / deserialize_setup)"
            ),
        },
        "reflection": {
            "definition": "file f -> 9 - f within every rank (engine reflect_setup)",
            "involution": "reflect(reflect(setup)) == setup",
            "class_rule": (
                "a base setup and its reflection are one library identity; "
                "the stored representative is the lexicographically smaller "
                "piece-type tuple; no legal setup equals its own reflection, "
                "so every class has exactly two members"
            ),
        },
        "identity": {
            "base_setup_id": "setup_library_v1:<family_id>:<base_index %03d>",
            "class_fingerprint": (
                f"SHA-256 over '{CLASS_FINGERPRINT_DOMAIN}:' + serialized "
                "class representative"
            ),
            "content_fingerprint": (
                f"SHA-256 over '{CONTENT_FINGERPRINT_DOMAIN}:' + serialized "
                "specific orientation"
            ),
            "process_hash_forbidden": (
                "Python's process-randomized built-in hash is never used"
            ),
        },
        "split_rule": {
            "assignment": "base_index 0..399 train, 400..449 validation, 450..499 test",
            "permanence": (
                "split is a pure function of base_index, fixed before "
                "generation; reflections and perturbed descendants inherit "
                "the base split; no outcome-based reassignment is permitted"
            ),
        },
        "seeding": {
            "master_seed_default": DEFAULT_LIBRARY_MASTER_SEED,
            "base_seed": (
                "blake2b(person='strat-lb7') over "
                "'contract:library:master_seed:family_id:base_index'"
            ),
            "attempt_seed": (
                "blake2b(person='strat-at7') over 'base_seed:attempt'; "
                "generation draws attempts 0,1,2,... and accepts the first "
                "candidate satisfying the contract, so rejection is local to "
                "the base identity"
            ),
            "isolated_regeneration": (
                "contract + library version + master seed + family id + "
                "base index -> exact same base setup and metadata, without "
                "generating any preceding setup"
            ),
            "cross_base_independence": (
                "generation of one base must never condition on any other "
                "base's outcome; global uniqueness/distance requirements are "
                "acceptance gates, and violating them under the frozen "
                "contract is a BLOCKED outcome, not a licence to reroll"
            ),
        },
        "quality": {
            "initial_mobility_rule": (
                "every accepted base setup and every generated descendant "
                "must have at least one legal move for its owner in initial "
                "board geometry, verified through the frozen engine "
                "(create_game + has_legal_action); stranded candidates are "
                "rejected and counted; this is a library acceptance "
                "criterion, not an engine or ruleset change"
            ),
            "engine_validation": (
                "every accepted setup passes engine validate_setup and "
                "constructs a game via create_game"
            ),
        },
        "serialization_contract": {
            "library_jsonl_path": LIBRARY_JSONL_PATH,
            "library_manifest_path": LIBRARY_MANIFEST_PATH,
            "entry_fields": list(BASE_ENTRY_REQUIRED_FIELDS),
            "entry_line_format": "json.dumps(entry, sort_keys=True, separators=(',', ':'))",
            "file_order": "families F00..F15, base index ascending within each family",
            "library_digest": (
                "SHA-256 over newline-joined 'base_setup_id:fingerprint' "
                "lines in file order"
            ),
        },
        "isolated_rebuild_sample": {
            "per_family_indices": list(isolated_rebuild_sample_indices()),
            "total_rebuilds": FAMILY_COUNT * len(isolated_rebuild_sample_indices()),
        },
        "perturbation_invariants": {
            "min_hamming": PERTURBATION_MIN_HAMMING,
            "max_hamming": PERTURBATION_MAX_HAMMING,
            "invariants": list(PERTURBATION_INVARIANTS),
            "mutable": (
                "any non-Flag placement may change, within the Hamming bound, "
                "provided every family clause, the inventory, and initial "
                "mobility are preserved"
            ),
            "immutable": "Flag cell, split, primary family, inventory, orientation semantics",
        },
        "prohibitions": {
            "no_outcome_based_selection": (
                "win rates, Elo, game outcomes, value/policy scores and human "
                "preference are forbidden inputs to setup acceptance, "
                "rejection, weighting or split assignment"
            ),
            "no_setup_neural_network": (
                "no learned generator, selector, value or entropy model is "
                "authorized in Phase 7"
            ),
            "observer_safety": (
                "setup provenance (family, base id, seeds, fingerprints) is "
                "training/debug metadata and must never cross the "
                "observer-safe model input boundary; observation_v2_1_127ch "
                "is unchanged"
            ),
        },
        "family_contracts": families_document(),
        "trait_schema": trait_schema_document(),
    }


# ---------------------------------------------------------------------------
# Convenience re-exports used by later agents
# ---------------------------------------------------------------------------

__all__ = [
    "SETUP_GENERATOR_CONTRACT_VERSION",
    "SETUP_LIBRARY_VERSION",
    "SETUP_FAMILY_VERSION",
    "SETUP_TRAIT_VECTOR_VERSION",
    "FAMILY_COUNT",
    "BASES_PER_FAMILY",
    "BASE_SETUP_COUNT",
    "TRAIN_PER_FAMILY",
    "VALIDATION_PER_FAMILY",
    "TEST_PER_FAMILY",
    "TRAIN_TOTAL",
    "VALIDATION_TOTAL",
    "TEST_TOTAL",
    "SPLITS",
    "SPLIT_TRAIN",
    "SPLIT_VALIDATION",
    "SPLIT_TEST",
    "split_for_base_index",
    "base_setup_id",
    "parse_base_setup_id",
    "DEFAULT_LIBRARY_MASTER_SEED",
    "LIBRARY_JSONL_PATH",
    "LIBRARY_MANIFEST_PATH",
    "BASE_ENTRY_REQUIRED_FIELDS",
    "base_entry_json_line",
    "library_digest",
    "isolated_rebuild_sample_indices",
    "PERTURBATION_MIN_HAMMING",
    "PERTURBATION_MAX_HAMMING",
    "PERTURBATION_INVARIANTS",
    "validate_perturbation",
    "contract_document",
]
