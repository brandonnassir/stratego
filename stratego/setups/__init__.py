"""Phase 7 setup-library contracts: taxonomy, identity, traits, diversity.

This package is the pre-generation contract for `setup_library_v1`, owned by
Phase 7 Agent 1. It defines — and freezes — the canonical setup
representation, reflection/canonicalization, stable identity, the 16 primary
family contracts, the structural trait vector, the initial-mobility quality
rule, the deterministic seeding/split rules, the perturbation invariants, and
the diversity standard with its numeric thresholds.

It deliberately contains no production generator: Agent 2 builds the 8,000
base setups against these contracts, and Agent 3 audits them independently
through the same public API.
"""

from .contracts import (
    BASE_ENTRY_REQUIRED_FIELDS,
    BASE_SETUP_COUNT,
    BASES_PER_FAMILY,
    DEFAULT_LIBRARY_MASTER_SEED,
    FAMILY_COUNT,
    LIBRARY_JSONL_PATH,
    LIBRARY_MANIFEST_PATH,
    PERTURBATION_INVARIANTS,
    PERTURBATION_MAX_HAMMING,
    PERTURBATION_MIN_HAMMING,
    SETUP_FAMILY_VERSION,
    SETUP_GENERATOR_CONTRACT_VERSION,
    SETUP_LIBRARY_VERSION,
    SETUP_TRAIT_VECTOR_VERSION,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    SPLITS,
    TEST_PER_FAMILY,
    TEST_TOTAL,
    TRAIN_PER_FAMILY,
    TRAIN_TOTAL,
    VALIDATION_PER_FAMILY,
    VALIDATION_TOTAL,
    base_entry_json_line,
    base_setup_id,
    contract_document,
    isolated_rebuild_sample_indices,
    library_digest,
    parse_base_setup_id,
    split_for_base_index,
    validate_perturbation,
)
from .diversity import (
    DIVERSITY_STANDARD_VERSION,
    DIVERSITY_THRESHOLDS_V1,
    DiversityThresholds,
    LibraryEntry,
    class_distance,
    evaluate_against_thresholds,
    hamming_distance,
)
from .families import (
    FAMILY_CONTRACT_VERSION,
    FAMILY_CONTRACTS,
    FAMILY_IDS,
    FAMILY_KEYS,
    FamilyContract,
    evaluate_family,
    families_document,
    family_contract,
)
from .identity import (
    CANONICAL_CELLS,
    CANONICAL_FILES,
    CANONICAL_RANKS,
    FRONT_RANK,
    SetupLibraryError,
    canonical_class_representative,
    canonical_index,
    canonical_neighbours,
    canonical_rank_file,
    class_fingerprint,
    content_fingerprint,
    deorient_setup,
    derive_attempt_seed,
    derive_base_seed,
    derive_stream_seed,
    edge_file_distance,
    is_canonical_representative,
    orient_setup,
    reflect_canonical,
)
from .mobility import setup_has_initial_mobility
from .traits import (
    TRAIT_NAMES,
    TRAIT_SCHEMA,
    TRAIT_SCHEMA_VERSION,
    UNCONVENTIONAL_FEATURES,
    compute_trait_vector,
    trait_schema_document,
)
