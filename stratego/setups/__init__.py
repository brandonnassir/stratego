"""Phase 7 setup library: contracts, taxonomy, identity, traits, generation.

`identity`, `mobility`, `traits`, `families`, `contracts` and `diversity` are
the pre-generation contract for `setup_library_v1`, owned by Phase 7 Agent 1.
They define — and freeze — the canonical setup representation,
reflection/canonicalization, stable identity, the 16 primary family
contracts, the structural trait vector, the initial-mobility quality rule, the
deterministic seeding/split rules, the perturbation invariants, and the
diversity standard with its numeric thresholds. Those definitions are frozen:
later agents implement against them and never weaken or reinterpret them.

`seed`, `generator` and `library` are Phase 7 Agent 2's deterministic
production generator, built strictly against those frozen contracts: one
construction framework driven by 16 declarative family plans, the frozen
acceptance stack (engine validation, family predicate, initial mobility), and
the materialized 8,000-base library with its manifest. Agent 3 audits the
result independently through the same public API.

`audit` is Phase 7 Agent 3's independent exhaustive auditor: it recomputes
legality, mobility, identity, splits, family contracts, duplicates,
cross-split leakage, the frozen diversity thresholds and the family overlap
matrix from the materialized JSONL plus the frozen engine and contracts,
never from Agent 2's counters. It reports findings; it repairs nothing.

`perturbation` and `sampler` are Phase 7 Agent 4's runtime setup sampler,
built strictly on top of the audited library: constrained family-preserving
perturbation (`setup_perturbation_v1`) whose every candidate is accepted only
by Agent 1's frozen `validate_perturbation`, and the deterministic sampler
(`setup_sampler_v1`) that chooses family, base, perturbation and left-right
orientation from seeded streams, validates every finished output from
scratch, and returns the setup with provenance that rebuilds it exactly. The
sampler reads the base library and never writes to it.
"""

from .audit import (
    AUDIT_VERSION,
    audit_library,
    count_audit,
    duplicate_audit,
    line_format_audit,
    manifest_audit,
    overlap_audit,
    per_base_audit,
    similarity_audit,
    similarity_cross_check,
    threshold_audit,
)
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
from .generator import (
    FAMILY_PLANS,
    GENERATOR_VERSION,
    MAX_ATTEMPTS_PER_BASE,
    REJECTION_REASONS,
    BaseSetupEntry,
    BombPlan,
    FamilyPlan,
    FlagPlan,
    GroupPlan,
    construct_candidate,
    family_plan,
    generate_base_setup,
    plans_document,
    rebuild_base_setup,
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
from .library import (
    LibraryGenerationResult,
    build_manifest,
    entry_metadata_digest,
    generate_library,
    library_content_digest,
    library_order,
    manifest_digest,
    read_library_jsonl,
    read_manifest,
    verify_library,
    write_library_jsonl,
    write_manifest,
)
from .mobility import setup_has_initial_mobility
from .perturbation import (
    DEFAULT_OPERATOR_WEIGHTS,
    MAX_PERTURBATION_ATTEMPTS,
    MAX_SWAP_COUNT,
    MIN_SWAP_COUNT,
    OPERATOR_NAMES,
    OPERATORS,
    PERTURBATION_SEED_ENCODING,
    PERTURBATION_VERSION,
    PerturbationResult,
    SwapOperator,
    apply_swaps,
    decode_perturbation_seed,
    encode_perturbation_seed,
    operator_mix_document,
    perturb_setup,
    perturbation_id,
)
from .sampler import (
    DEFAULT_PROFILE,
    NEUTRAL_PROFILE,
    PERTURBATION_ONLY_PROFILE,
    PROFILES,
    PROVENANCE_FIELDS,
    REFLECTION_ONLY_PROFILE,
    REQUIRED_PROVENANCE_FIELDS,
    SAMPLER_VERSION,
    SPLIT_BASE_RANGES,
    STRESS_CORPUS_VERSION,
    STRESS_OUTPUTS_PER_FAMILY,
    STRESS_SPLIT_OUTPUTS,
    STRESS_TOTAL_OUTPUTS,
    SampledSetup,
    SamplerProfile,
    SetupLibraryIndex,
    StressDraw,
    build_descendant,
    build_stress_output,
    load_library_index,
    provenance_is_observer_safe,
    provenance_round_trips,
    rebuild_from_provenance,
    sample_setup,
    sampler_contract_document,
    sampler_profile,
    stress_corpus_plan,
    validate_sampled_setup,
)
from .seed import (
    DEFAULT_SEED_CONTEXT,
    SEED_CONTEXT_VERSION,
    LibrarySeedContext,
)
from .traits import (
    TRAIT_NAMES,
    TRAIT_SCHEMA,
    TRAIT_SCHEMA_VERSION,
    UNCONVENTIONAL_FEATURES,
    compute_trait_vector,
    trait_schema_document,
)
