"""Phase 10 Agent 1: the frozen learned-setup-selection contract.

Specification sources:

- `00_PHASE_10_SEQUENCE_AND_COMMON_CONTRACT.md` (the whole document)
- `01_AGENT_1_CONTRACT_SEEDS_BANKS_ACCEPTANCE.md` ("Freeze contracts",
  "Freeze exactly six candidates", "Build/freeze both evaluation banks",
  "Freeze acceptance/statistics")

What this module is
-------------------
The single place every Phase 10 learning and evaluation decision is written
down, frozen **before any Phase 10 outcome game was played and before either
utility model was fit**. Agents 2-7 execute what is here; none of them
re-decides it. Constants live at module scope so a later agent imports the
decision rather than restating it, and the eight contract documents are
built from those same constants so a document can never drift from the code
that enforces it.

The module deliberately imports no torch, no library loader and no bank
builder at module scope: it is the layer *under* those, exactly as
`phase9_contract` sat under `phase9_banks`. The two documents that need the
schedule and the utility definition import them at function scope.

Digest convention
-----------------
Every document is hashed as SHA-256 over its canonical JSON
(`sort_keys=True`, `separators=(",", ":")`) — the repository's frozen
convention since Phase 7. Digests are pinned in
`tests/training/test_phase10_contract.py`, so an edit anywhere in this
module that changes a frozen decision fails the suite instead of quietly
redefining the experiment.
"""

from __future__ import annotations

import hashlib
import json

from ..setups.families import FAMILY_IDS
from .phase10_seed import (
    CANONICAL_PHASE10_SEEDS,
    CASE_GAME_COLOR,
    COLORS,
    PHASE10_IDENTITY_VERSION,
    PHASE10_MASTER_SEED,
    seed_derivation_document,
)

#: The umbrella version of the Phase 10 setup-selection contract.
SETUP_CONTRACT_VERSION = "phase10_setup_contract_v1"

#: The bound system version: Phase 9 move model + accepted utility/scaler +
#: selected selector config + frozen Phase 7 reflection/perturbation path.
SYSTEM_VERSION = "phase10_system_v1"

# ---------------------------------------------------------------------------
# Frozen upstream identities — verified from live bytes by Agent 1 before
# anything here was frozen, and re-verified by every later agent.
# ---------------------------------------------------------------------------

ACCEPTED_PHASE9_CHECKPOINT_PATH = "checkpoints/phase9/selfplay_c1_v1.pt"
ACCEPTED_PHASE9_CHECKPOINT_SHA256 = (
    "dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea"
)
ACCEPTED_PHASE9_MODEL_STATE_DIGEST = (
    "f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd"
)
ACCEPTED_PHASE9_PARAMETERS = 863_959
ACCEPTED_C1_CONFIG_DIGEST = (
    "31ca84ab140c523e65567787b0289fe0dbdf5ab0344667410a5fda7060cfe07d"
)
ACCEPTED_PHASE9_CONTRACT_DIGEST = (
    "ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34"
)
ACCEPTED_PHASE9_AMENDMENT_V1_DIGEST = (
    "ee4b05078c676128f78c8e5c31bd10ce4f0841e34a57c4c7c3fca6616e083ac4"
)
ACCEPTED_PHASE9_AMENDMENT_V2_DIGEST = (
    "92ad4f67fb07a14551ef555335b71000d6369cd817dad59c839d793888de9e71"
)

PHASE7_LIBRARY_VERSION = "setup_library_v1"
PHASE7_LIBRARY_CONTENT_DIGEST = (
    "7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777"
)
PHASE7_LIBRARY_METADATA_DIGEST = (
    "d86f486182a820d546d470ef4ebce92ff60c6259aed80c481bc985bce8c64980"
)
PHASE7_LIBRARY_MANIFEST_DIGEST = (
    "53139ab7e21c4e8a31987507d6fb1eabf93f36cdc1221fe85d08042963488f31"
)

#: The frozen Phase 7 stack Phase 10 consumes and may never change.
FROZEN_PHASE7_IDENTITIES = (
    "setup_generator_contract_v1",
    "setup_family_v1",
    "setup_trait_vector_v1",
    "setup_diversity_standard_v1",
    "setup_perturbation_v1",
    "setup_sampler_v1",
    "setup_source_v1",
)

#: The frozen engine/model identities every Phase 10 game runs under.
FROZEN_RUNTIME_IDENTITIES = {
    "rules": "stratego_project_v1",
    "reference_engine": "phase2_1_reference_1.2.0",
    "observation": "observation_v2_1_127ch",
    "engine_action_encoding": "source_destination_10000_v1",
    "model_action_frame": "perspective_normalized_squares",
    "model_contract": "model_contract_v2",
    "trajectory": "trajectory_v1",
    "backend": "KEEP_PYTHON",
}

#: The hard invariant of the whole phase.
PHASE9_PRESERVATION_INVARIANT = (
    "Phase 9 checkpoint before Phase 10 == Phase 9 checkpoint after Phase 10, "
    "in both file SHA-256 and model-state digest, with zero C1 optimizer steps"
)

#: What no Phase 10 agent may do. Restated as data so a later agent can
#: assert against the list rather than remember it.
NON_GOALS = (
    "update C1 policy/value/belief weights",
    "run PPO or continue Phase 9 RL",
    "change the Phase 7 library, splits, family definitions, reflection or "
    "perturbation semantics",
    "use opponent true setup/family/base/seed or other hidden opponent "
    "information as selector input",
    "change rules, observation channels, action encoding or engine semantics",
    "build a full setup Transformer",
    "perform Phase 11 belief redesign or Phase 12 search",
    "use human games",
    "begin the official 168-hour campaign",
    "use Phase 10 final-test outcomes for candidate selection or threshold changes",
)

# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

SETUP_SELECTOR_VERSION = "phase10_setup_selector_v1"
LEARNED_SETUP_SOURCE_VERSION = "learned_setup_source_v1"
SELECTOR_SCHEDULE_VERSION = "phase10_selector_schedule_v1"

#: The baseline profile Phase 10 consumes and never redefines.
NEUTRAL_PROFILE_NAME = "neutral_v1"

#: The frozen mixture. `p_phase10 = 0.35 * p_neutral_v1 + 0.65 * p_learned`.
NEUTRAL_MIXTURE_WEIGHT = 0.35
LEARNED_MIXTURE_WEIGHT = 0.65
assert abs(NEUTRAL_MIXTURE_WEIGHT + LEARNED_MIXTURE_WEIGHT - 1.0) < 1e-12

#: The only inputs a Phase 10 selector may read. Anything else — most of all
#: anything about the opponent — is an information-safety failure.
ALLOWED_SELECTOR_INPUTS = (
    "own color",
    "requested Phase 7 split",
    "candidate base's own family",
    "candidate base's own trait vector",
    "selector identity",
    "selector seed",
)

FORBIDDEN_SELECTOR_INPUTS = (
    "opponent true setup",
    "opponent family",
    "opponent base id",
    "opponent seed",
    "any other hidden opponent information",
    "any game outcome of the case being played",
    "any physical storage path",
)

#: The accepted Phase 7 path after base selection, unchanged.
POST_SELECTION_PATH = {
    "reflection_probability": 0.5,
    "perturbation_probability": 0.5,
    "swap_counts": [1, 2, 3, 4, 5, 6],
    "swap_count_distribution": "uniform over 1..6",
    "hamming_distance_window": [2, 12],
    "retry_semantics": "unchanged setup_perturbation_v1 semantics",
    "sampler_version": "setup_sampler_v1",
    "perturbation_version": "setup_perturbation_v1",
}

#: The frozen base ordering the learned softmax and its inverse-CDF walk use.
SELECTOR_BASE_ORDER = (
    "ascending (family_index, base_index) over the requested split, i.e. the "
    "frozen library enumeration order restricted to that split"
)

# ---------------------------------------------------------------------------
# Exactly six candidates. There is no seventh.
# ---------------------------------------------------------------------------

CANDIDATE_MATRIX = (
    {"candidate_id": "P10-A", "utility_model": "model_F", "temperature": 0.75},
    {"candidate_id": "P10-B", "utility_model": "model_F", "temperature": 1.25},
    {"candidate_id": "P10-C", "utility_model": "model_F", "temperature": 2.00},
    {"candidate_id": "P10-D", "utility_model": "model_T", "temperature": 0.75},
    {"candidate_id": "P10-E", "utility_model": "model_T", "temperature": 1.25},
    {"candidate_id": "P10-F", "utility_model": "model_T", "temperature": 2.00},
)
CANDIDATE_IDS = tuple(entry["candidate_id"] for entry in CANDIDATE_MATRIX)
CANDIDATE_COUNT = len(CANDIDATE_MATRIX)
assert CANDIDATE_COUNT == 6
assert len(set(CANDIDATE_IDS)) == 6

#: `neutral_v1` is the baseline every candidate is measured against. It is
#: not a seventh candidate and is never eligible to win.
BASELINE_SELECTOR_ID = "neutral_v1"

# ---------------------------------------------------------------------------
# Evaluation banks
# ---------------------------------------------------------------------------

EVAL_BANK_VERSION = "phase10_eval_bank_v1"

VALIDATION_BANK_VERSION = "phase10_validation_bank_v1"
VALIDATION_BANK_CASES = 128
VALIDATION_CASES_PER_FAMILY = 8

TEST_BANK_VERSION = "phase10_test_bank_v1"
TEST_BANK_CASES = 512
TEST_CASES_PER_FAMILY = 32

assert VALIDATION_BANK_CASES == VALIDATION_CASES_PER_FAMILY * len(FAMILY_IDS)
assert TEST_BANK_CASES == TEST_CASES_PER_FAMILY * len(FAMILY_IDS)

#: Frozen rejection ceiling for a case's family-conditioned, isolation-clean
#: opponent draw and for a case's isolation-clean selector seed.
BANK_MAX_ATTEMPTS = 2048

#: The frozen case structure. A case fixes the opponent's setup and the
#: selector's draw identity; the selector's own setup is produced at
#: evaluation time by whichever selector is under test.
CASE_STRUCTURE = {
    "opponent_setup": (
        "one held-out setup drawn by neutral_v1 from the bank's split, "
        "conditioned on the case family; the same physical arrangement plays "
        "in every matchup and in both arms"
    ),
    "selector_seeds": (
        "one seed per (case, colour), identical for the learned candidate and "
        "the neutral baseline, so a difference is measured on one draw identity"
    ),
    "colour_pairing": (
        "the selector under test plays Red in game 0 and Blue in game 1 "
        "against the same fixed opponent setup"
    ),
    "match_seeds": (
        "one seed per (case, game index, matchup), independent of arm and "
        "candidate, so a rule-based opponent draws identical randomness in "
        "both arms"
    ),
    "bootstrap_unit": "the logical case, scoring the mean of its two games",
    "family_identity": "the opponent setup's primary family",
}

#: The six frozen validation matchups. Token order is the report order.
MATCHUP_LEARNED_VS_NEUTRAL = "learned_vs_neutral"
MATCHUP_STRATEGIC = "vs_strategic"
MATCHUP_TACTICAL = "vs_tactical"
MATCHUP_PHASE8_ANCHOR = "vs_phase8_anchor"
MATCHUP_RANDOM = "vs_random"
MATCHUP_BASIC = "vs_basic"

MATCHUPS = (
    {
        "token": MATCHUP_LEARNED_VS_NEUTRAL,
        "opponent": "neutral_v1 selector, accepted Phase 9 policy both sides",
        "opponent_policy_id": None,
        "neutral_arm": False,
        "role": "direct setup-selector comparison (Delta_D)",
    },
    {
        "token": MATCHUP_STRATEGIC,
        "opponent": "strategic_rule_based",
        "opponent_policy_id": "strategic_rule_based",
        "neutral_arm": True,
        "role": "strong-opponent league component",
    },
    {
        "token": MATCHUP_TACTICAL,
        "opponent": "tactical_rule_based",
        "opponent_policy_id": "tactical_rule_based",
        "neutral_arm": True,
        "role": "strong-opponent league component",
    },
    {
        "token": MATCHUP_PHASE8_ANCHOR,
        "opponent": "Phase 8 anchor checkpoint",
        "opponent_policy_id": "phase8_anchor",
        "neutral_arm": True,
        "role": "strong-opponent league component",
    },
    {
        "token": MATCHUP_RANDOM,
        "opponent": "random_legal",
        "opponent_policy_id": "random_legal",
        "neutral_arm": True,
        "role": "easy-opponent guard",
    },
    {
        "token": MATCHUP_BASIC,
        "opponent": "basic_heuristic",
        "opponent_policy_id": "basic_heuristic",
        "neutral_arm": True,
        "role": "easy-opponent guard",
    },
)
MATCHUP_TOKENS = tuple(entry["token"] for entry in MATCHUPS)
assert len(set(MATCHUP_TOKENS)) == len(MATCHUPS) == 6

#: All move play, in every matchup, under one frozen behaviour.
EVAL_MOVE_BEHAVIOR = {
    "decision_mode": "greedy",
    "dtype": "float32",
    "batch_policy": "single_request",
    "search": "none",
}

# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

#: `S10 = 0.40*Delta_D + 0.30*Delta_Strategic + 0.20*Delta_Tactical + 0.10*Delta_Phase8`
SELECTION_SCORE_WEIGHTS = {
    "delta_direct": 0.40,
    "delta_strategic": 0.30,
    "delta_tactical": 0.20,
    "delta_phase8_anchor": 0.10,
}
assert abs(sum(SELECTION_SCORE_WEIGHTS.values()) - 1.0) < 1e-12

#: Random and Basic are guards, never score components.
SCORE_EXCLUDED_MATCHUPS = (MATCHUP_RANDOM, MATCHUP_BASIC)

TIE_BREAK_ORDER = (
    "higher S10",
    "higher Delta_Strategic",
    "higher Delta_D",
    "higher normalized family entropy",
    "higher effective base diversity",
    "lexicographically smaller candidate id",
)

#: Validation point guards. A candidate below either is ineligible.
VALIDATION_RANDOM_MIN_EWR = 0.95
VALIDATION_BASIC_MIN_EWR = 0.80

ELIGIBILITY_RULE = (
    "a candidate failing correctness, diversity or reproducibility is "
    "ineligible regardless of score; if no candidate is eligible, overall "
    "Phase 10 returns FAIL and later agents stop"
)

# ---------------------------------------------------------------------------
# Diversity contract — stated over the final mixed distribution
# ---------------------------------------------------------------------------

DIVERSITY_THRESHOLDS = {
    "normalized_family_entropy_min": 0.85,
    "effective_families_min": 10.0,
    "family_probability_min": 0.015,
    "family_probability_max": 0.18,
    "within_family_normalized_base_entropy_min": 0.70,
    "max_conditional_base_probability": 0.10,
}

DIVERSITY_SCOPE = (
    "the final mixed distribution p_phase10 = 0.35*p_neutral_v1 + 0.65*p_learned, "
    "for every candidate, colour and split"
)

#: Agent 4's frozen selector-audit volume, per candidate x colour x split.
SELECTOR_AUDIT_DRAWS = 100_000

SELECTOR_AUDIT_ZERO_TOLERANCE = (
    "illegal setups",
    "inventory errors",
    "stranded sampled setups",
    "split violations",
    "provenance mismatches",
    "determinism mismatches",
    "non-finite selector values",
)

SELECTOR_AUDIT_REQUIREMENTS = (
    "all 16 families represented across the audited draws",
)

# ---------------------------------------------------------------------------
# Final acceptance gates
# ---------------------------------------------------------------------------

ACCEPTANCE_VERSION = "phase10_acceptance_v1"

GATE_A = {
    "gate": "A",
    "name": "direct learned-v-neutral non-inferiority",
    "ordinary": {"ewr_min": 0.49, "ewr_min_strict": False, "lb_min": 0.47, "lb_min_strict": True},
    "improved": {"ewr_min": 0.52, "ewr_min_strict": False, "lb_min": 0.50, "lb_min_strict": True},
}

GATE_B = {
    "gate": "B",
    "name": "strong-opponent league non-inferiority",
    "league_weights": {
        "delta_strategic": 0.45,
        "delta_tactical": 0.35,
        "delta_phase8_anchor": 0.20,
    },
    "delta_l_min": -0.01,
    "delta_l_min_strict": False,
    "lb_min": -0.03,
    "lb_min_strict": True,
    "significant": {"delta_l_min": 0.0, "delta_l_min_strict": True, "lb_min": 0.0, "lb_min_strict": True},
}
assert abs(sum(GATE_B["league_weights"].values()) - 1.0) < 1e-12

GATE_C = {
    "gate": "C",
    "name": "individual strong-opponent guards",
    "opponents": [MATCHUP_STRATEGIC, MATCHUP_TACTICAL, MATCHUP_PHASE8_ANCHOR],
    "lb_min": -0.03,
    "lb_min_strict": True,
}

GATE_D = {
    "gate": "D",
    "name": "easy-opponent guards",
    "random_overall_min": 0.95,
    "random_red_min": 0.90,
    "random_blue_min": 0.90,
    "basic_min": 0.80,
    "min_strict": False,
    "paired_lb_min": -0.03,
    "paired_lb_min_strict": True,
    "paired_opponents": [MATCHUP_RANDOM, MATCHUP_BASIC],
}

GATE_E = {"gate": "E", "name": "diversity", "thresholds": dict(DIVERSITY_THRESHOLDS)}

GATE_F = {
    "gate": "F",
    "name": "correctness and information safety",
    "zero_tolerance": (
        "illegal setups",
        "inventory errors",
        "stranded sampled setups",
        "split leakage",
        "provenance mismatch",
        "hidden-opponent selector inputs",
        "illegal neural moves",
        "non-finite selector outputs",
        "inference failures",
    ),
}

GATE_G = {
    "gate": "G",
    "name": "reproducibility",
    "rule": (
        "logical game id + selector seed + selector identity + requested split "
        "+ colour -> same base -> same reflection -> same perturbation -> same "
        "final fingerprint, independent of worker order and process restart"
    ),
}

GATE_H = {
    "gate": "H",
    "name": "Phase 9 preservation",
    "checkpoint_sha256": ACCEPTED_PHASE9_CHECKPOINT_SHA256,
    "model_state_digest": ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
    "parameters": ACCEPTED_PHASE9_PARAMETERS,
    "c1_optimizer_steps": 0,
}

HARD_GATES = (GATE_A, GATE_B, GATE_C, GATE_D, GATE_E, GATE_F, GATE_G, GATE_H)
HARD_GATE_IDS = tuple(gate["gate"] for gate in HARD_GATES)
assert HARD_GATE_IDS == ("A", "B", "C", "D", "E", "F", "G", "H")

CLASSIFICATIONS = {
    "PASS-IMPROVED": (
        "all eight hard gates pass AND Gate A's improved criterion passes AND "
        "Gate B is significantly positive"
    ),
    "PASS-NONINFERIOR": (
        "all eight hard gates pass but the improved criteria are not both met"
    ),
    "FAIL": "the experiment runs correctly but a hard gate fails",
    "BLOCKED": "prerequisite identity/sealing/discipline evidence cannot be verified",
}

DIAGNOSTIC_RULE = "report-only diagnostics never rescue a failed gate"

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

STATISTICS = {
    "unit": "paired logical setup case",
    "method": "paired_unit_percentile_bootstrap",
    "rng": "numpy_pcg64",
    "replicates": 10_000,
    "confidence": 0.95,
    "validation_bootstrap_root": CANONICAL_PHASE10_SEEDS["validation_bootstrap_seed"],
    "final_bootstrap_root": CANONICAL_PHASE10_SEEDS["test_bootstrap_seed"],
    "token_rule": (
        "each matchup and each learned-minus-neutral difference receives a "
        "domain-separated token through bootstrap_stream_seed(bank, token)"
    ),
    "recompute_rule": (
        "gate booleans and selection scores must be recomputed from the "
        "primitive recorded outcomes, never read back from a summary"
    ),
}

BOOTSTRAP_REPLICATES = STATISTICS["replicates"]
BOOTSTRAP_CONFIDENCE = STATISTICS["confidence"]

# ---------------------------------------------------------------------------
# Sealing
# ---------------------------------------------------------------------------

TEST_BANK_SEALING = {
    "sealed_until": "Agent 7",
    "allowed_before_agent_7": (
        "structural_build",
        "structural_audit",
        "digest_computation",
        "fingerprint_isolation_check",
        "structural_artifact_write",
    ),
    "forbidden_before_agent_7": (
        "neural inference on a case",
        "playing any game on a case",
        "computing any model metric on a case",
        "candidate selection using a case",
        "hyperparameter or threshold selection using a case",
    ),
    "access_log_rule": "every test-bank access is recorded with its purpose",
}

STOP_CONDITIONS = {
    "BLOCKED": (
        "an accepted upstream identity mismatches",
        "held-out data enters fitting",
        "selector inputs require opponent-private information",
        "exact bank fingerprint isolation cannot be established",
        "final-test outcomes are accessed before Agent 7",
        "storage identity or mount safety fails",
    ),
    "FAIL": (
        "no candidate survives validation",
        "a final strength or diversity gate fails",
    ),
    "never": (
        "reopen Phase 9",
        "change Phase 7",
        "add candidates",
        "retune after results",
    ),
}


class Phase10ContractError(ValueError):
    """Raised when a Phase 10 contract condition is violated."""


def document_digest(document) -> str:
    """SHA-256 over a document's canonical JSON — the frozen convention."""
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# The eight frozen documents
# ---------------------------------------------------------------------------


def setup_contract_document() -> dict:
    """`phase10_setup_contract_v1` — scope, upstream identities, non-goals."""
    return {
        "contract_version": SETUP_CONTRACT_VERSION,
        "identity_version": PHASE10_IDENTITY_VERSION,
        "mission": (
            "learned setup selection over the frozen Phase 7 setup library, with "
            "the accepted Phase 9 move policy unchanged; not a move-policy "
            "training phase and not a setup Transformer"
        ),
        "pipeline": (
            "setup_library_v1 -> lightweight setup utility -> setup_selector_v1 "
            "-> frozen reflection/perturbation -> initial setup, plus the "
            "accepted Phase 9 C1 move policy, unchanged"
        ),
        "accepted_move_model": {
            "path": ACCEPTED_PHASE9_CHECKPOINT_PATH,
            "sha256": ACCEPTED_PHASE9_CHECKPOINT_SHA256,
            "model_state_digest": ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
            "parameters": ACCEPTED_PHASE9_PARAMETERS,
            "c1_config_digest": ACCEPTED_C1_CONFIG_DIGEST,
        },
        "accepted_phase9_chain": {
            "contract_digest": ACCEPTED_PHASE9_CONTRACT_DIGEST,
            "amendment_v1_digest": ACCEPTED_PHASE9_AMENDMENT_V1_DIGEST,
            "amendment_v2_digest": ACCEPTED_PHASE9_AMENDMENT_V2_DIGEST,
        },
        "frozen_phase7_stack": {
            "library_version": PHASE7_LIBRARY_VERSION,
            "content_digest": PHASE7_LIBRARY_CONTENT_DIGEST,
            "metadata_digest": PHASE7_LIBRARY_METADATA_DIGEST,
            "manifest_digest": PHASE7_LIBRARY_MANIFEST_DIGEST,
            "families": 16,
            "bases": 8000,
            "train": 6400,
            "validation": 800,
            "test": 800,
            "profile_baseline": NEUTRAL_PROFILE_NAME,
            "frozen_identities": list(FROZEN_PHASE7_IDENTITIES),
        },
        "frozen_runtime_identities": dict(FROZEN_RUNTIME_IDENTITIES),
        "phase9_preservation_invariant": PHASE9_PRESERVATION_INVARIANT,
        "non_goals": list(NON_GOALS),
        "seeds": seed_derivation_document(),
        "stop_conditions": {key: list(value) for key, value in STOP_CONDITIONS.items()},
        "storage_semantics": (
            "logical identities are path-independent; resolver/pointer semantics "
            "carry Phase 10 bytes; a pointer naming an absent external volume is "
            "BLOCKED, never a silent internal replacement"
        ),
    }


def outcome_corpus_document() -> dict:
    """`phase10_setup_outcome_corpus_v1` — the 16,384-game corpus contract."""
    from .phase10_schedule import corpus_contract_document

    return corpus_contract_document()


def utility_document() -> dict:
    """`phase10_setup_utility_v1` — features, standardizer, fit protocol."""
    from .phase10_utility import utility_contract_document

    return utility_contract_document()


def selector_document() -> dict:
    """`phase10_setup_selector_v1` — selector semantics and diversity contract."""
    return {
        "selector_version": SETUP_SELECTOR_VERSION,
        "production_source_version": LEARNED_SETUP_SOURCE_VERSION,
        "baseline_profile": NEUTRAL_PROFILE_NAME,
        "baseline_is_not_a_candidate": True,
        "allowed_inputs": list(ALLOWED_SELECTOR_INPUTS),
        "forbidden_inputs": list(FORBIDDEN_SELECTOR_INPUTS),
        "learned_distribution": "p_learned(s | c, split) = softmax(u(s, c) / T)",
        "learned_support": "every base of the requested split, all 16 families",
        "base_order": SELECTOR_BASE_ORDER,
        "mixture": {
            "formula": "p_phase10 = 0.35 * p_neutral_v1 + 0.65 * p_learned",
            "neutral_weight": NEUTRAL_MIXTURE_WEIGHT,
            "learned_weight": LEARNED_MIXTURE_WEIGHT,
        },
        "neutral_branch_definition": (
            "p_neutral_v1 is the accepted sampler's base choice: uniform over the "
            "16 families and uniform over that family's bases inside the split; "
            "because every family contributes equally many bases to a split, it is "
            "uniform over the split's bases"
        ),
        "draw_procedure": [
            "branch = neutral if selector_branch_uniform(...) < 0.35 else learned",
            "neutral branch: take the base the accepted setup_sampler_v1 would "
            "have taken for (split, selector_seed, profile='neutral_v1'), so a "
            "neutral-branch draw is bit-identical to the neutral baseline's draw",
            "learned branch: inverse-CDF walk of selector_base_uniform(...) over "
            "the split's bases in the frozen base order, using float64 cumulative "
            "softmax mass, with the last base as the float tail guard",
            "then the accepted Phase 7 path, unchanged: reflection coin, "
            "perturbation coin, uniform swap count, frozen perturbation and the "
            "complete final-output validation stack",
        ],
        "post_selection_path": dict(POST_SELECTION_PATH),
        "utility_domain": (
            "u is a function of a Phase 7 base setup's own family and own trait "
            "vector only, so every selector input stays legal at choice time"
        ),
        "candidates": [dict(entry) for entry in CANDIDATE_MATRIX],
        "candidate_count": CANDIDATE_COUNT,
        "no_seventh_candidate": (
            "exactly six candidates; the two utility models are fit once and "
            "candidate-specific refitting is forbidden; neutral_v1 is a baseline"
        ),
        "diversity": {
            "scope": DIVERSITY_SCOPE,
            "thresholds": dict(DIVERSITY_THRESHOLDS),
            "audit_draws_per_candidate_colour_split": SELECTOR_AUDIT_DRAWS,
            "audit_draw_identity": (
                "phase10_selector_audit_v1|ms=<master>|k=<candidate>|s=<split>"
                "|c=<colour>|n=<ordinal:05d>; the audit's selector seed is "
                "selector_audit_seed(candidate_id, split, colour, draw_ordinal) "
                "under the selector_audit domain, so a draw is addressable by id, "
                "resume is exact set subtraction by draw id, and re-sharding "
                "across workers cannot move a single draw"
            ),
            "zero_tolerance": list(SELECTOR_AUDIT_ZERO_TOLERANCE),
            "requirements": list(SELECTOR_AUDIT_REQUIREMENTS),
        },
        "reproducibility": GATE_G["rule"],
    }


def selector_schedule_document() -> dict:
    """`phase10_selector_schedule_v1` — matchups, arms, scoring, guards."""
    return {
        "schedule_version": SELECTOR_SCHEDULE_VERSION,
        "move_behavior": dict(EVAL_MOVE_BEHAVIOR),
        "move_policy": "accepted Phase 9 checkpoint on the evaluated side in every matchup",
        "matchups": [dict(entry) for entry in MATCHUPS],
        "matchup_count": len(MATCHUPS),
        "neutral_arm_rule": (
            "for matchups 2-6 the Phase9+neutral_v1 arm is evaluated on the exact "
            "same logical cases, so a setup-selector delta is a paired quantity"
        ),
        "deltas": {
            "delta_direct": "EWR(learned selector vs neutral selector) - 0.5",
            "delta_opponent": "EWR(Phase9+learned vs O) - EWR(Phase9+neutral vs O)",
        },
        "selection_score": {
            "formula": (
                "S10 = 0.40*Delta_D + 0.30*Delta_Strategic + 0.20*Delta_Tactical "
                "+ 0.10*Delta_Phase8"
            ),
            "weights": dict(SELECTION_SCORE_WEIGHTS),
            "excluded_matchups": list(SCORE_EXCLUDED_MATCHUPS),
        },
        "tie_break_order": list(TIE_BREAK_ORDER),
        "validation_guards": {
            "random_overall_min": VALIDATION_RANDOM_MIN_EWR,
            "basic_min": VALIDATION_BASIC_MIN_EWR,
            "role": "guards, not score components",
        },
        "eligibility": ELIGIBILITY_RULE,
        "selection_data": (
            "candidate selection uses the validation bank only; the test bank is "
            "sealed until Agent 7 and corpus outcomes may never select"
        ),
    }


def eval_bank_document() -> dict:
    """`phase10_eval_bank_v1` — bank structure, isolation and sealing."""
    return {
        "eval_bank_version": EVAL_BANK_VERSION,
        "master_seed": PHASE10_MASTER_SEED,
        "banks": [
            {
                "bank": "validation",
                "bank_version": VALIDATION_BANK_VERSION,
                "split": "validation",
                "case_count": VALIDATION_BANK_CASES,
                "cases_per_opponent_family": VALIDATION_CASES_PER_FAMILY,
                "bootstrap_root": CANONICAL_PHASE10_SEEDS["validation_bootstrap_seed"],
            },
            {
                "bank": "test",
                "bank_version": TEST_BANK_VERSION,
                "split": "test",
                "case_count": TEST_BANK_CASES,
                "cases_per_opponent_family": TEST_CASES_PER_FAMILY,
                "bootstrap_root": CANONICAL_PHASE10_SEEDS["test_bootstrap_seed"],
            },
        ],
        "family_order": list(FAMILY_IDS),
        "case_structure": dict(CASE_STRUCTURE),
        "colour_pairing": {str(k): v for k, v in sorted(CASE_GAME_COLOR.items())},
        "colours": list(COLORS),
        "max_attempts": BANK_MAX_ATTEMPTS,
        "matchup_tokens": list(MATCHUP_TOKENS),
        "isolation": {
            "claim": (
                "Phase 10 does not claim a wholly unseen base-template universe; "
                "it guarantees new logical case ids, new Phase 10 seeds, new "
                "procedural descendants, zero exact final-setup fingerprint "
                "overlap with the Phase 9 held-out evaluation banks over every "
                "setup a Phase 10 case fixes, and zero use of any Phase 9 "
                "per-case outcome in Phase 10 fitting or selection"
            ),
            "frozen_setups_per_case": (
                "the fixed opponent setup and the two neutral_v1 own-side draws "
                "(one per colour) — every arrangement a case determines before a "
                "selector exists"
            ),
            "rejection_rule": (
                "walk the frozen attempt streams and accept the first draw that is "
                "family-correct, outside the Phase 9 held-out fingerprint set, and "
                "distinct from the case's other frozen setups"
            ),
            "base_id_reuse": "allowed across phases; exact final fingerprint reuse is not",
            "runtime_monitor": (
                "a learned selector's own-side draw cannot be enumerated before the "
                "selector exists, so Agents 5-7 record — as a report-only "
                "diagnostic, never a gate — how many produced final setups land in "
                "the Phase 9 held-out fingerprint set; rejecting them at draw time "
                "would distort the very mixed distribution the diversity contract "
                "is stated over"
            ),
        },
        "sealing": dict(TEST_BANK_SEALING),
        "no_outcome_selection": (
            "construction plays no game and reads no strength signal; draws are "
            "rejected only for family identity, fingerprint isolation and "
            "within-case distinctness"
        ),
    }


def acceptance_document() -> dict:
    """`phase10_acceptance_v1` — the eight hard gates and the statistics."""
    return {
        "acceptance_version": ACCEPTANCE_VERSION,
        "hard_gates": [
            {key: (list(value) if isinstance(value, tuple) else value) for key, value in gate.items()}
            for gate in HARD_GATES
        ],
        "hard_gate_ids": list(HARD_GATE_IDS),
        "gate_count": len(HARD_GATES),
        "all_gates_are_hard": True,
        "inequality_semantics": (
            "thresholds marked strict use '>' and unmarked thresholds use '>='; "
            "every gate below states its own strictness explicitly"
        ),
        "classifications": dict(CLASSIFICATIONS),
        "diagnostic_rule": DIAGNOSTIC_RULE,
        "statistics": dict(STATISTICS),
        "evaluated_on": TEST_BANK_VERSION,
        "evaluated_by": "Agent 7, on the single frozen winner",
    }


def system_document() -> dict:
    """`phase10_system_v1` — the binding schema of the finished system.

    Agent 1 freezes the *binding rules and slots*, not their values: the
    accepted utility, scaler and selector configuration do not exist yet and
    inventing placeholders for them would be exactly the pre-commitment this
    phase forbids. Agent 6 fills the unbound slots at the production freeze,
    and Agent 7 verifies the filled document against these rules. The move
    model and the selector stay separate artifacts throughout.
    """
    return {
        "system_version": SYSTEM_VERSION,
        "bound_now": {
            "move_model": {
                "path": ACCEPTED_PHASE9_CHECKPOINT_PATH,
                "sha256": ACCEPTED_PHASE9_CHECKPOINT_SHA256,
                "model_state_digest": ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
                "parameters": ACCEPTED_PHASE9_PARAMETERS,
                "mutability": "byte-identical throughout Phase 10",
            },
            "post_selection_path": dict(POST_SELECTION_PATH),
            "baseline_profile": NEUTRAL_PROFILE_NAME,
            "library": {
                "library_version": PHASE7_LIBRARY_VERSION,
                "content_digest": PHASE7_LIBRARY_CONTENT_DIGEST,
                "metadata_digest": PHASE7_LIBRARY_METADATA_DIGEST,
                "manifest_digest": PHASE7_LIBRARY_MANIFEST_DIGEST,
            },
        },
        "unbound_slots": [
            {
                "slot": "accepted_utility_model",
                "filled_by": "Agent 6",
                "requires": [
                    "utility_version phase10_setup_utility_v1",
                    "model id in {model_F, model_T}",
                    "the parameter digest of the single fit of that model",
                    "the corpus digest the fit consumed",
                ],
            },
            {
                "slot": "accepted_trait_scaler",
                "filled_by": "Agent 6",
                "requires": [
                    "scaler_version phase10_trait_scaler_v1",
                    "the train-only scaler digest frozen by Agent 1",
                ],
            },
            {
                "slot": "selected_selector_config",
                "filled_by": "Agent 6",
                "requires": [
                    "one candidate id from the frozen six",
                    "its temperature",
                    "the 0.35/0.65 mixture, unchanged",
                    "learned_setup_source_v1",
                ],
            },
        ],
        "separation_rule": (
            "the move model and the selector remain separate artifacts; binding "
            "them into one system document never merges their bytes"
        ),
        "phase11_preservation": [
            NEUTRAL_PROFILE_NAME,
            LEARNED_SETUP_SOURCE_VERSION,
            "accepted Phase 10 selector config/utility",
            ACCEPTED_PHASE9_CHECKPOINT_PATH,
        ],
    }


#: The eight frozen contracts, in the order the common contract lists them.
CONTRACT_BUILDERS = (
    (SETUP_CONTRACT_VERSION, setup_contract_document),
    ("phase10_setup_outcome_corpus_v1", outcome_corpus_document),
    ("phase10_setup_utility_v1", utility_document),
    (SETUP_SELECTOR_VERSION, selector_document),
    (SELECTOR_SCHEDULE_VERSION, selector_schedule_document),
    (EVAL_BANK_VERSION, eval_bank_document),
    (ACCEPTANCE_VERSION, acceptance_document),
    (SYSTEM_VERSION, system_document),
)
CONTRACT_VERSIONS = tuple(name for name, _ in CONTRACT_BUILDERS)
assert len(CONTRACT_VERSIONS) == 8 == len(set(CONTRACT_VERSIONS))


def contract_documents() -> dict:
    """Every frozen Phase 10 contract document, keyed by its version."""
    return {name: builder() for name, builder in CONTRACT_BUILDERS}


def contract_digests(documents: "dict | None" = None) -> dict:
    """The stable SHA-256 of every frozen contract document."""
    documents = contract_documents() if documents is None else documents
    return {name: document_digest(documents[name]) for name in CONTRACT_VERSIONS}


def contract_bundle_digest(documents: "dict | None" = None) -> str:
    """One digest over the eight contracts — the Phase 10 freeze identity."""
    return document_digest(contract_digests(documents))


__all__ = [
    "ACCEPTANCE_VERSION",
    "ACCEPTED_C1_CONFIG_DIGEST",
    "ACCEPTED_PHASE9_AMENDMENT_V1_DIGEST",
    "ACCEPTED_PHASE9_AMENDMENT_V2_DIGEST",
    "ACCEPTED_PHASE9_CHECKPOINT_PATH",
    "ACCEPTED_PHASE9_CHECKPOINT_SHA256",
    "ACCEPTED_PHASE9_CONTRACT_DIGEST",
    "ACCEPTED_PHASE9_MODEL_STATE_DIGEST",
    "ACCEPTED_PHASE9_PARAMETERS",
    "ALLOWED_SELECTOR_INPUTS",
    "BANK_MAX_ATTEMPTS",
    "BASELINE_SELECTOR_ID",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_REPLICATES",
    "CANDIDATE_COUNT",
    "CANDIDATE_IDS",
    "CANDIDATE_MATRIX",
    "CASE_STRUCTURE",
    "CLASSIFICATIONS",
    "CONTRACT_BUILDERS",
    "CONTRACT_VERSIONS",
    "DIVERSITY_SCOPE",
    "DIVERSITY_THRESHOLDS",
    "ELIGIBILITY_RULE",
    "EVAL_BANK_VERSION",
    "EVAL_MOVE_BEHAVIOR",
    "FORBIDDEN_SELECTOR_INPUTS",
    "FROZEN_PHASE7_IDENTITIES",
    "FROZEN_RUNTIME_IDENTITIES",
    "GATE_A",
    "GATE_B",
    "GATE_C",
    "GATE_D",
    "GATE_E",
    "GATE_F",
    "GATE_G",
    "GATE_H",
    "HARD_GATES",
    "HARD_GATE_IDS",
    "LEARNED_MIXTURE_WEIGHT",
    "LEARNED_SETUP_SOURCE_VERSION",
    "MATCHUPS",
    "MATCHUP_BASIC",
    "MATCHUP_LEARNED_VS_NEUTRAL",
    "MATCHUP_PHASE8_ANCHOR",
    "MATCHUP_RANDOM",
    "MATCHUP_STRATEGIC",
    "MATCHUP_TACTICAL",
    "MATCHUP_TOKENS",
    "NEUTRAL_MIXTURE_WEIGHT",
    "NEUTRAL_PROFILE_NAME",
    "NON_GOALS",
    "PHASE7_LIBRARY_CONTENT_DIGEST",
    "PHASE7_LIBRARY_MANIFEST_DIGEST",
    "PHASE7_LIBRARY_METADATA_DIGEST",
    "PHASE7_LIBRARY_VERSION",
    "PHASE9_PRESERVATION_INVARIANT",
    "POST_SELECTION_PATH",
    "SCORE_EXCLUDED_MATCHUPS",
    "SELECTION_SCORE_WEIGHTS",
    "SELECTOR_AUDIT_DRAWS",
    "SELECTOR_AUDIT_REQUIREMENTS",
    "SELECTOR_AUDIT_ZERO_TOLERANCE",
    "SELECTOR_BASE_ORDER",
    "SELECTOR_SCHEDULE_VERSION",
    "SETUP_CONTRACT_VERSION",
    "SETUP_SELECTOR_VERSION",
    "STATISTICS",
    "STOP_CONDITIONS",
    "SYSTEM_VERSION",
    "TEST_BANK_CASES",
    "TEST_BANK_SEALING",
    "TEST_BANK_VERSION",
    "TEST_CASES_PER_FAMILY",
    "TIE_BREAK_ORDER",
    "VALIDATION_BANK_CASES",
    "VALIDATION_BANK_VERSION",
    "VALIDATION_BASIC_MIN_EWR",
    "VALIDATION_CASES_PER_FAMILY",
    "VALIDATION_RANDOM_MIN_EWR",
    "Phase10ContractError",
    "acceptance_document",
    "contract_bundle_digest",
    "contract_digests",
    "contract_documents",
    "document_digest",
    "eval_bank_document",
    "outcome_corpus_document",
    "selector_document",
    "selector_schedule_document",
    "setup_contract_document",
    "system_document",
    "utility_document",
]
