"""Phase 11 Agent 1: the frozen belief-validation and search-readiness contract.

Specification sources:

- `00_PHASE_11_SEQUENCE_AND_COMMON_CONTRACT.md` (the whole document)
- `01_AGENT_1_CONTRACTS_SEEDS_BANKS_ACCEPTANCE.md` ("Freeze contracts",
  "Freeze observer/opponent semantics", "Freeze target/event semantics",
  "Freeze baselines", "Freeze sampler mathematics", "Freeze
  metrics/statistics", "Freeze Gates A-H", "Freeze classification")

What this module is
-------------------
The single place every Phase 11 scientific decision is written down, frozen
**before any Phase 11 prediction score, validation result, sampler output or
test outcome exists**. Agents 2-7 execute what is here; none of them
re-decides it. Constants live at module scope so a later agent imports the
decision rather than restating it, and the eight contract documents are
built from those same constants so a document can never drift from the code
that enforces it.

Phase 11 is a **validation phase**: it measures the belief head the accepted
Phase 9 checkpoint already carries and the frozen count-constrained sampler
built on top of it. Nothing here trains, calibrates, or repairs anything;
if the system fails its gates, Phase 11 ends as FAIL and a separate repair
phase must be designed.

Digest convention
-----------------
Every document is hashed as SHA-256 over its canonical JSON
(`sort_keys=True`, `separators=(",", ":")`) — the repository's frozen
convention since Phase 7. Digests are pinned in
`tests/training/phase11_frozen_digests.py`, so an edit anywhere in this
module that changes a frozen decision fails the suite instead of quietly
redefining the experiment.
"""

from __future__ import annotations

import hashlib
import json
import math

from .phase10_contract import (
    ACCEPTED_C1_CONFIG_DIGEST,
    ACCEPTED_PHASE9_CHECKPOINT_PATH,
    ACCEPTED_PHASE9_CHECKPOINT_SHA256,
    ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
    ACCEPTED_PHASE9_PARAMETERS,
    FROZEN_PHASE7_IDENTITIES,
    FROZEN_RUNTIME_IDENTITIES,
    LEARNED_SETUP_SOURCE_VERSION,
    NEUTRAL_PROFILE_NAME,
    PHASE7_LIBRARY_CONTENT_DIGEST,
    PHASE7_LIBRARY_MANIFEST_DIGEST,
    PHASE7_LIBRARY_METADATA_DIGEST,
    PHASE7_LIBRARY_VERSION,
)
from .phase11_seed import (
    BELIEF_MODEL_LABEL,
    BENCHMARK_CELL_COUNT,
    BENCHMARK_STATES_PER_CELL,
    BENCHMARK_STATE_COUNT,
    CANONICAL_PHASE11_SEEDS,
    CASE_GAME_INDICES,
    CASE_GAME_OBSERVER_COLOR,
    CASE_GAME_OPPONENT_COLOR,
    COLORS,
    OPPONENT_STRATA,
    PHASE11_IDENTITY_VERSION,
    PHASE11_MASTER_SEED,
    REPRO_REQUEST_COUNT,
    SAFETY_TRIAL_COUNT,
    SETUP_SOURCES,
    SOAK_GAMES_PER_STRATUM,
    SOAK_GAME_COUNT,
    SOAK_REQUESTS_PER_GAME,
    SOAK_REQUEST_COUNT,
    SOURCE_NEUTRAL,
    SOURCE_P10D,
    STRATUM_BASIC,
    STRATUM_INFORMATION_MISER,
    STRATUM_MINER_RUSH,
    STRATUM_PHASE8_ANCHOR,
    STRATUM_PHASE9,
    STRATUM_SCOUT_RUSH,
    STRATUM_STRATEGIC,
    STRATUM_TACTICAL,
    seed_derivation_document,
)

#: The umbrella version of the Phase 11 belief-validation contract.
BELIEF_CONTRACT_VERSION = "phase11_belief_contract_v1"
BASELINE_CONTRACT_VERSION = "phase11_belief_baseline_v1"
BANK_CONTRACT_VERSION = "phase11_belief_bank_v1"
METRICS_CONTRACT_VERSION = "phase11_belief_metrics_v1"
SAMPLER_CONTRACT_VERSION = "phase11_belief_sampler_v1"
INFORMATION_SAFETY_VERSION = "phase11_information_safety_v1"
ACCEPTANCE_VERSION = "phase11_acceptance_v1"
SYSTEM_VERSION = "phase11_system_v1"

#: Component version tokens frozen now; implementations arrive downstream
#: under exactly these names.
EVALUATOR_VERSION = "phase11_belief_evaluator_v1"
REMAINING_COUNT_BASELINE_VERSION = "remaining_count_belief_v1"
WORLD_BASELINE_VERSION = "count_uniform_world_sampler_v1"
BELIEF_SAMPLER_VERSION = "belief_sampler_v1"
PREDICTION_RECORD_VERSION = "phase11_prediction_record_v1"
PUBLIC_STATE_DOCUMENT_VERSION = "phase11_public_state_v1"
BELIEF_REQUEST_VERSION = "phase11_belief_request_v1"

# ---------------------------------------------------------------------------
# Frozen upstream identities — verified from live bytes by Agent 1 before
# anything here was frozen, and re-verified by every later agent.
# ---------------------------------------------------------------------------

#: The formal Phase 10 closure commit, recorded from live Git state.
PHASE10_CLOSURE_COMMIT = "17188a5"

#: The belief head inside the accepted Phase 9 checkpoint: exact live tensor
#: names and the SHA-256 over their `(name, shape, float32 little-endian
#: bytes)` in sorted name order — the same recipe as the accepted
#: model-state digest, restricted to the head. Derived from live checkpoint
#: bytes at the Agent 1 freeze; every later agent re-derives it.
BELIEF_HEAD_TENSOR_NAMES = ("belief_output.bias", "belief_output.weight")
BELIEF_HEAD_TENSOR_SHAPES = {
    "belief_output.weight": (12, 128),
    "belief_output.bias": (12,),
}
ACCEPTED_BELIEF_HEAD_DIGEST = (
    "a9df48a1adcd29b1a46c42ff1e605ede485119a36c247f1ae74f249f6d6f1dc7"
)

#: The accepted Phase 9 checkpoint's training-history optimizer position.
#: Phase 11 requires this to be *unchanged* — the phase itself takes zero
#: optimizer steps.
ACCEPTED_GLOBAL_OPTIMIZER_STEP = 47_086

#: The accepted Phase 10 production selector: P10-D exactly as frozen.
ACCEPTED_SELECTOR_CANDIDATE_ID = "P10-D"
ACCEPTED_SELECTOR_UTILITY_MODEL = "model_T"
ACCEPTED_SELECTOR_TEMPERATURE = 0.75
ACCEPTED_SELECTOR_IDENTITY = "learned_setup_source_v1|k=P10-D|m=model_T|T=0.75"
ACCEPTED_SELECTOR_NEUTRAL_WEIGHT = 0.35
ACCEPTED_SELECTOR_LEARNED_WEIGHT = 0.65
ACCEPTED_SELECTOR_CONFIG_SHA256 = (
    "6e227815bc3cb44f19cdeee55d00ec0ae75726fb411ee9131660aa712bb86668"
)

#: The accepted Phase 10 utility (model_T) and trait scaler.
ACCEPTED_UTILITY_COEFFICIENT_DIGEST = (
    "d898782a2ae7cf4ed1cb2833fad6e53d8407ec2048dafbd34a6a20c1c9766edc"
)
ACCEPTED_TRAIT_SCALER_DIGEST = (
    "fa6eb1c112defc4c1034831b84db8848181e1f674f8439c9c265916d89e8b7f9"
)
ACCEPTED_UTILITY_FILE_SHA256 = (
    "50cb947dae633417858dc3352ee1e68e41c1c54845c5d3a261f735571983c25d"
)

#: The accepted filled `phase10_system_v1` instance digest (Agent 6 freeze,
#: Agent 7 verified).
ACCEPTED_PHASE10_SYSTEM_DIGEST = (
    "615cc3c3a4fab6e4400e20a5a93b13a08c43ab6c3ca63828c6a64742e98175d2"
)

#: The accepted Phase 8 anchor evaluation export (the `phase8_anchor`
#: stratum's weights), verified against the accepted Phase 9 record.
ACCEPTED_ANCHOR_EXPORT_PATH = "checkpoints/phase9/agent01/anchor_eval.pt"
ACCEPTED_ANCHOR_EXPORT_SHA256 = (
    "cd0b22d24d36dbe01f88897c3e2bde325b7e141d07d092edc74918e6b0cd6dda"
)

#: The hard invariant of the whole phase.
PHASE9_PRESERVATION_INVARIANT = (
    "Phase 9 checkpoint before Phase 11 == Phase 9 checkpoint after Phase 11, "
    "in file SHA-256, model-state digest and belief-head digest, with zero "
    "Phase 11 optimizer steps"
)

#: What no Phase 11 agent may do — the common contract's ten prohibitions,
#: restated as data so a later agent can assert against the list.
NON_GOALS = (
    "update any Phase 9 neural parameter",
    "run a belief optimizer step",
    "calibrate or temperature-scale the belief head",
    "change the 127-channel observation design",
    "change P10-D, utility, scaler, temperature, or mixture",
    "modify Phase 7 setup generation",
    "begin Phase 12 search",
    "use Phase 11 test evidence to repair or retune anything",
    "use hidden opponent truth as an input to belief inference or sampling",
    "silently change a frozen metric, threshold, bank, seed, or statistical "
    "procedure",
)

# ---------------------------------------------------------------------------
# Rank space
# ---------------------------------------------------------------------------

#: The frozen belief rank order: the accepted engine `PIECE_TYPE_NAMES`
#: enumeration, which is exactly the index order the accepted belief head
#: was trained under (`dense_belief_target_v1`). Index i of every 12-vector
#: in Phase 11 means rank `RANK_NAMES[i]`.
RANK_NAMES = (
    "spy",
    "scout",
    "miner",
    "sergeant",
    "lieutenant",
    "captain",
    "major",
    "colonel",
    "general",
    "marshal",
    "flag",
    "bomb",
)
RANK_COUNT = len(RANK_NAMES)
assert RANK_COUNT == 12

#: Initial per-player inventory, by rank index. Sums to 40.
RANK_INITIAL_COUNTS = (1, 8, 5, 4, 4, 4, 3, 2, 1, 1, 1, 6)
assert sum(RANK_INITIAL_COUNTS) == 40

#: The two publicly immovable ranks, by index (flag, bomb).
IMMOVABLE_RANK_INDICES = (10, 11)
MOVABLE_RANK_INDICES = tuple(
    index for index in range(RANK_COUNT) if index not in IMMOVABLE_RANK_INDICES
)

# ---------------------------------------------------------------------------
# Banks
# ---------------------------------------------------------------------------

VALIDATION_BANK_VERSION = "phase11_validation_bank_v1"
VALIDATION_CASES_PER_CELL = 32
VALIDATION_BANK_CASES = 512
VALIDATION_BANK_GAMES = 1_024

TEST_BANK_VERSION = "phase11_test_bank_v1"
TEST_CASES_PER_CELL = 128
TEST_BANK_CASES = 2_048
TEST_BANK_GAMES = 4_096

_CELLS = len(OPPONENT_STRATA) * len(SETUP_SOURCES)
assert VALIDATION_BANK_CASES == VALIDATION_CASES_PER_CELL * _CELLS
assert TEST_BANK_CASES == TEST_CASES_PER_CELL * _CELLS
assert VALIDATION_BANK_GAMES == 2 * VALIDATION_BANK_CASES
assert TEST_BANK_GAMES == 2 * TEST_BANK_CASES

#: The frozen Phase 7 split each bank draws every setup from — the accepted
#: held-out-split precedent of the Phase 9 and Phase 10 banks, recorded as
#: an Agent 1 reading because the common contract does not name a split.
BANK_SPLITS = {"validation": "validation", "test": "test"}

#: The eight opponent-behaviour strata, bound to their exact accepted
#: implementations. `opponent_policy_id` is the accepted evaluation-registry
#: identifier for rule/stress policies and None for neural seats, which are
#: bound by checkpoint identity instead.
STRATUM_BINDINGS = (
    {
        "stratum": STRATUM_PHASE9,
        "description": "accepted Phase 9 policy (self-play opponent)",
        "opponent_policy_id": None,
        "opponent_checkpoint_sha256": ACCEPTED_PHASE9_CHECKPOINT_SHA256,
    },
    {
        "stratum": STRATUM_PHASE8_ANCHOR,
        "description": "accepted Phase 8 anchor checkpoint",
        "opponent_policy_id": None,
        "opponent_checkpoint_sha256": ACCEPTED_ANCHOR_EXPORT_SHA256,
    },
    {
        "stratum": STRATUM_STRATEGIC,
        "description": "strategic rule opponent",
        "opponent_policy_id": "strategic_rule_based",
        "opponent_checkpoint_sha256": None,
    },
    {
        "stratum": STRATUM_TACTICAL,
        "description": "tactical rule opponent",
        "opponent_policy_id": "tactical_rule_based",
        "opponent_checkpoint_sha256": None,
    },
    {
        "stratum": STRATUM_BASIC,
        "description": "basic rule opponent",
        "opponent_policy_id": "basic_heuristic",
        "opponent_checkpoint_sha256": None,
    },
    {
        "stratum": STRATUM_INFORMATION_MISER,
        "description": "information-hoarding stress opponent",
        "opponent_policy_id": "stress_information_miser",
        "opponent_checkpoint_sha256": None,
    },
    {
        "stratum": STRATUM_SCOUT_RUSH,
        "description": "scout-rush stress opponent",
        "opponent_policy_id": "stress_scout_rush",
        "opponent_checkpoint_sha256": None,
    },
    {
        "stratum": STRATUM_MINER_RUSH,
        "description": "miner-rush stress opponent",
        "opponent_policy_id": "stress_miner_rush",
        "opponent_checkpoint_sha256": None,
    },
)
assert tuple(entry["stratum"] for entry in STRATUM_BINDINGS) == OPPONENT_STRATA

#: All neural seats — the observer in every game, and the opponent in the
#: two neural strata — play the accepted evaluation move behaviour.
EVAL_MOVE_BEHAVIOR = {
    "decision_mode": "greedy",
    "dtype": "float32",
    "batch_policy": "single_request",
    "search": "none",
}

#: The observer's own setup source: the accepted P10-D production source,
#: constant across both banks, as the common contract's default requires.
#: No implementation constraint prevents it — verified before this freeze.
OBSERVER_SETUP_SOURCE = {
    "source": LEARNED_SETUP_SOURCE_VERSION,
    "candidate_id": ACCEPTED_SELECTOR_CANDIDATE_ID,
    "selector_identity": ACCEPTED_SELECTOR_IDENTITY,
    "mixture": {
        "neutral_weight": ACCEPTED_SELECTOR_NEUTRAL_WEIGHT,
        "learned_weight": ACCEPTED_SELECTOR_LEARNED_WEIGHT,
    },
    "constant_across_banks": True,
}

#: The opponent's setup source, by setup-source stratum token.
OPPONENT_SETUP_SOURCES = {
    SOURCE_P10D: {
        "source": LEARNED_SETUP_SOURCE_VERSION,
        "candidate_id": ACCEPTED_SELECTOR_CANDIDATE_ID,
        "selector_identity": ACCEPTED_SELECTOR_IDENTITY,
    },
    SOURCE_NEUTRAL: {
        "source": "setup_sampler_v1",
        "profile": NEUTRAL_PROFILE_NAME,
    },
}

# ---------------------------------------------------------------------------
# Target and event semantics
# ---------------------------------------------------------------------------

#: Progress buckets over the pre-action `total_moves` of the decision, in
#: plies. Thresholds fixed from *accepted* Phase 10 evidence (soak mean
#: 116.8 plies per canonical self-play game), before any Phase 11 result
#: existed. Diagnostic slices only; no gate reads them.
PROGRESS_BUCKETS = (
    {"bucket": "early", "min_total_moves": 0, "max_total_moves": 39},
    {"bucket": "middle", "min_total_moves": 40, "max_total_moves": 119},
    {"bucket": "late", "min_total_moves": 120, "max_total_moves": None},
)
PROGRESS_BUCKET_NAMES = tuple(entry["bucket"] for entry in PROGRESS_BUCKETS)


def progress_bucket(total_moves: int) -> str:
    """The frozen progress bucket of one decision's pre-action ply count."""
    if not isinstance(total_moves, int) or isinstance(total_moves, bool) or total_moves < 0:
        raise Phase11ContractError(
            f"total_moves must be a non-negative int, got {total_moves!r}"
        )
    for entry in PROGRESS_BUCKETS:
        upper = entry["max_total_moves"]
        if total_moves >= entry["min_total_moves"] and (
            upper is None or total_moves <= upper
        ):
            return entry["bucket"]
    raise Phase11ContractError(f"no progress bucket covers {total_moves}")  # pragma: no cover


#: The floor applied inside `ln` when scoring cross-entropy, and only there.
#: Recorded probabilities are never floored; a floored event increments the
#: report-only `log_floor_events` diagnostic.
LOG_PROBABILITY_FLOOR = 1e-12

#: The exact prediction-record field list, in the frozen order.
PREDICTION_RECORD_FIELDS = (
    "record_version",
    "bank_version",
    "case_id",
    "game_id",
    "prediction_id",
    "decision_index",
    "observer_color",
    "opponent_stratum",
    "opponent_setup_source",
    "public_state_identity",
    "observation_sha256",
    "piece_slot",
    "piece_square",
    "piece_moved",
    "progress_bucket",
    "legal_rank_mask",
    "remaining_counts",
    "learned_probabilities",
    "baseline_probabilities",
    "true_rank_index",
    "model_identity",
    "prediction_identity",
)

#: The one privileged column. Everything else in the record is public.
PRIVILEGED_RECORD_FIELDS = ("true_rank_index",)

#: The frozen public-state document field list. The document is
#: observer-relative — it holds exactly what the observer may legally see —
#: and embeds the observation digest, so the public-state identity covers
#: the complete model input and the sampled-world purity claim is exact.
PUBLIC_STATE_DOCUMENT_FIELDS = (
    "document_version",
    "observer_color",
    "acting_player_color",
    "total_moves",
    "battleless_moves",
    "rules_version",
    "engine_version",
    "observation_version",
    "pieces",
    "recent_moves",
    "observation_sha256",
)

#: Per-piece public sub-record of the document's `pieces` list (all 80
#: pieces in stable piece-id order).
PUBLIC_PIECE_FIELDS = (
    "piece_slot",
    "owner_color",
    "alive",
    "current_square",
    "has_moved",
    "known_to_observer",
    "known_rank_index",
    "starting_square",
)

#: The production belief-inference request: these fields and nothing else.
#: The Agent 2 request type must reject any other field structurally, the
#: way the accepted `SelectorRequest.from_payload` does.
ALLOWED_BELIEF_REQUEST_FIELDS = (
    "request_version",
    "request_id",
    "observer_color",
    "public_state_document",
    "observation",
)

#: Name fragments that mark a request field as hidden-truth/outcome
#: information. A rejected injection must raise, never be dropped.
FORBIDDEN_BELIEF_REQUEST_TOKENS = (
    "true",
    "truth",
    "label",
    "target",
    "private",
    "winner",
    "result",
    "reward",
    "outcome",
    "future",
    "path",
)


class Phase11ContractError(ValueError):
    """Raised when a Phase 11 contract condition is violated."""


def document_digest(document) -> str:
    """SHA-256 over a document's canonical JSON — the frozen convention."""
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95

#: The frozen overall metric tokens; a stratum slice appends `|st=<stratum>`.
OVERALL_METRIC_TOKENS = (
    "ce_learned",
    "ce_baseline",
    "ce_delta",
    "r_ce",
    "top1_learned",
    "top1_baseline",
    "top1_delta",
    "brier_learned",
    "brier_baseline",
    "brier_delta",
    "true_rank_probability_learned",
    "true_rank_probability_baseline",
    "entropy_learned",
    "entropy_baseline",
)

STATISTICS = {
    "unit": "logical paired case (both colour games kept together)",
    "case_aggregate": (
        "the unweighted arithmetic mean of the metric's per-event values over "
        "every prediction event of the case's two games, pooled"
    ),
    "overall_point_estimate": (
        "the unweighted arithmetic mean of the case aggregates (equal case "
        "weight)"
    ),
    "delta_metrics": (
        "per-case delta = case-aggregate learned minus case-aggregate "
        "baseline; overall delta = mean of per-case deltas; the pairing is "
        "the case"
    ),
    "ratio_metrics": (
        "R_CE = mean-of-case-aggregate learned CE / mean-of-case-aggregate "
        "baseline CE, recomputed from the resampled case aggregates inside "
        "every bootstrap replicate"
    ),
    "method": "case_percentile_bootstrap",
    "rng": "numpy_pcg64_default_rng",
    "replicates": BOOTSTRAP_REPLICATES,
    "confidence": BOOTSTRAP_CONFIDENCE,
    "resampling": (
        "resample the bank's logical case ids with replacement (stratum CIs "
        "resample that stratum's cases only); never bootstrap pieces, events "
        "or games independently — predictions within a game are correlated"
    ),
    "interval": (
        "percentile interval via the accepted linear-interpolation quantile "
        "(stratego.evaluation.statistics.quantile) at (1-confidence)/2 and "
        "1-(1-confidence)/2 over the sorted replicate statistics"
    ),
    "stream_rule": (
        "every bootstrapped quantity draws from "
        "bootstrap_stream_seed(bank, metric_token); stratum slices use "
        "'<metric_token>|st=<stratum>'"
    ),
    "validation_bootstrap_root": CANONICAL_PHASE11_SEEDS["validation_bootstrap_seed"],
    "final_bootstrap_root": CANONICAL_PHASE11_SEEDS["test_bootstrap_seed"],
    "ece_aggregation": (
        "ECE pools prediction events (equal event weight) — a binned "
        "calibration summary, not a case mean; overall and per-stratum alike"
    ),
    "recompute_rule": (
        "gate booleans must be recomputed from the primitive recorded rows, "
        "never read back from a summary"
    ),
}

#: ECE: 15 equal-width confidence bins on [0, 1] (the last bin closed),
#: confidence = max learned probability, accuracy = top-1 correctness,
#: weighted absolute gap — the Agent 1 instruction's recommendation, frozen.
ECE_SPECIFICATION = {
    "bins": 15,
    "bin_rule": "bin k = [k/15, (k+1)/15) for k < 14; bin 14 = [14/15, 1]",
    "confidence": "max probability of the scored 12-vector",
    "accuracy": "top-1 correctness (argmax, first-occurrence tie rule)",
    "formula": "ECE = sum_k (n_k / N) * |accuracy_k - confidence_k|",
    "weighting": "equal event weight, pooled",
}

METRIC_FORMULAS = {
    "probability_source": (
        "learned 12-vector = softmax over the observer's belief-head logits "
        "at the piece's perspective-normalized square, computed in float64 "
        "(subtract max, exponentiate, normalize), full simplex, no masking, "
        "no epsilon — the head is measured exactly as trained and exactly as "
        "the sampler consumes it"
    ),
    "ce": "CE = -ln(max(p[true_rank_index], 1e-12)); natural log",
    "r_ce": "R_CE = CE_learned / CE_remaining_count (overall aggregates)",
    "top1": "top1 = 1[argmax(p) == true_rank_index]; first occurrence wins ties",
    "brier": "Brier = sum_r (p[r] - 1[r == true_rank_index])^2 over the 12 ranks",
    "true_rank_probability": "p[true_rank_index], unfloored",
    "entropy": "H = -sum_r p[r] * ln(p[r]) with 0*ln(0) = 0; natural log",
    "ece": dict(ECE_SPECIFICATION),
    "float_policy": (
        "all metric arithmetic in float64; a non-finite metric value fails "
        "its gate (comparisons with NaN are false) and is itself a finding"
    ),
}

#: Mandatory diagnostic slices. Slices are report-only; only the stratum
#: slice feeds gates (C and D).
DIAGNOSTIC_SLICES = (
    "opponent_stratum",
    "observer_color",
    "progress_bucket",
    "piece_moved",
    "true_rank",
    "opponent_setup_source",
)

# ---------------------------------------------------------------------------
# Sampler mathematics
# ---------------------------------------------------------------------------

#: Gate E's zero-tolerance counters, exactly.
SAMPLER_ZERO_TOLERANCE_COUNTERS = (
    "inventory_errors",
    "public_knowledge_violations",
    "known_rank_violations",
    "immobility_violations",
    "impossible_assignments",
    "nonfinite_probability_rows",
    "provenance_mismatches",
    "hidden_input_accesses",
    "dead_end_events",
)

#: Every check the complete-world validation stack must pass, per world.
WORLD_VALIDATION_STACK = (
    "exact remaining-inventory match: assigned hidden multiset equals the "
    "publicly inferable remaining inventory, rank by rank",
    "publicly known ranks locked: every known piece carries exactly its "
    "known rank",
    "captured/dead pieces excluded: no assignment to a dead piece",
    "immobility legality: no publicly moved piece is assigned flag or bomb "
    "(public Scout deductions are already locked knowns)",
    "ownership and alive/dead status preserved exactly",
    "public start information preserved: assignments are keyed by the "
    "public piece tracker (owner, setup slot), never by reindexing",
    "known rank-by-start information preserved through the locked knowns",
    "probability rows finite and non-negative at every step",
    "provenance rebuilds: the recorded (public-state identity, model label, "
    "sampler version, sample ordinal) re-derives the identical world",
)

#: Sampler-request boundary: these inputs and nothing else.
ALLOWED_SAMPLER_REQUEST_FIELDS = (
    "sampler_version",
    "public_state_document",
    "learned_probabilities",
    "sample_ordinal",
)
FORBIDDEN_SAMPLER_REQUEST_TOKENS = FORBIDDEN_BELIEF_REQUEST_TOKENS

SAMPLER_ALGORITHM_STEPS = (
    "1. read the public belief state and the learned per-piece marginals "
    "only; hidden truth is structurally absent from the request",
    "2. lock publicly known ranks (combat reveals and public Scout "
    "deductions are knowns, not masks)",
    "3. compute the exact remaining hidden inventory c[r] = initial[r] - "
    "known[r] from public information",
    "4. apply the public impossibility masks (a publicly moved hidden piece "
    "cannot be flag or bomb)",
    "5. derive the deterministic random unresolved-piece order: ascending "
    "(world_order_key(sample_token, piece_slot), piece_slot)",
    "6. for each unresolved piece in that order, legal ranks are the public "
    "legal ranks with remaining count > 0 that also satisfy the frozen "
    "completion-feasibility rule",
    "7. weight each legal rank by learned_probability * remaining_count",
    "8. renormalize in float64 and draw by inverse-CDF walk of "
    "world_categorical_uniform(sample_token, step_index) over the ranks in "
    "frozen rank-index order, last legal rank as the float tail guard",
    "9. decrement the chosen rank's remaining count",
    "10. if every legal weight is zero (zero learned mass on the legal "
    "set), fall back for that step to the normalized remaining counts over "
    "the same legal set",
    "11. continue until every unresolved piece is assigned",
    "12. verify the complete world against the frozen validation stack",
)

#: The completion-feasibility rule of step 6, with its exactness argument.
SAMPLER_FEASIBILITY_RULE = {
    "rule": (
        "an unmoved unresolved piece may take a movable rank only when "
        "movable_remaining - 1 >= moved_unresolved_remaining (counted over "
        "the not-yet-assigned pieces after the current one); moved pieces "
        "and immovable choices need no guard beyond the public mask and "
        "counts"
    ),
    "why": (
        "the common contract requires every walk to complete (steps 11-12) "
        "and tolerates zero invalid worlds, but the unguarded legal set can "
        "dead-end on a feasible instance: unmoved pieces drawn early can "
        "exhaust the movable inventory that later moved pieces need. The "
        "guard maintains the invariant movable_remaining >= "
        "moved_unresolved_remaining, which the true assignment guarantees "
        "at the start, so the walk provably never dead-ends"
    ),
    "exactness": (
        "every valid complete world remains reachable: if a valid world "
        "assigns rank r to the current piece, the remaining assignment in "
        "that world proves the guard inequality, so the guard only removes "
        "doomed prefixes"
    ),
    "fallback_totality": (
        "with the guard, the legal set of step 6 is never empty on a "
        "publicly consistent state, so step 10's counts-only fallback "
        "always has positive mass; a dead end is therefore impossible and "
        "`dead_end_events` is a zero-tolerance counter proving it"
    ),
    "status": (
        "an Agent 1 design reading amending step 6's legal-rank definition, "
        "recorded for reviewer acceptance at the Agent 1 handoff; the "
        "step-7 weighting is the common contract's, unchanged"
    ),
}

#: Report-only sampler diagnostics — never gates, unless a reviewer freezes
#: a threshold before any sampling exists (none is frozen).
SAMPLER_REPORT_ONLY_DIAGNOSTICS = (
    "zero-mass fallback count/rate",
    "distinct worlds per position",
    "marginal agreement between sampled worlds and the learned marginals",
    "sampler entropy/diversity",
)

#: The frozen provenance fields of one sampled world.
SAMPLER_PROVENANCE_FIELDS = (
    "sample_token",
    "sampler_version",
    "public_state_identity",
    "belief_model_label",
    "sample_ordinal",
    "piece_order",
    "fallback_steps",
    "assignment",
)

#: The large-audit floor volumes.
SAMPLER_AUDIT_MIN_WORLDS = 250_000
SAMPLER_INDEPENDENT_AUDIT_MIN_WORLDS = 25_000

# ---------------------------------------------------------------------------
# Information safety, reproducibility, runtime
# ---------------------------------------------------------------------------

#: Gate F's zero-tolerance counters, exactly.
INFORMATION_SAFETY_ZERO_COUNTERS = (
    "belief_output_differences",
    "fixed_seed_sample_differences",
    "forbidden_hidden_input_accesses",
    "injection_acceptances",
)

INFORMATION_SAFETY_ATTACK = {
    "trials": SAFETY_TRIAL_COUNT,
    "trial_id_format": "phase11_safety_trial_v1|ms=<master>|n=<ordinal:05d>",
    "state_pool": (
        "validation-bank public states with at least two unresolved "
        "opponent pieces, selected deterministically by the trial's "
        "state_selection stream"
    ),
    "permutation": (
        "construct an alternative private hidden truth by permuting the "
        "true ranks among the unresolved pieces, driven by the trial's "
        "truth_permutation stream, subject to: identical public state and "
        "history bytes, identical remaining inventory (automatic for a "
        "permutation), no publicly moved piece receiving flag or bomb, and "
        "at least one piece's true rank changed"
    ),
    "no_alternative_rule": (
        "a state admitting no altered legal truth (for example, all "
        "unresolved pieces share one rank) is skipped and the trial "
        "deterministically walks to the next candidate state in its "
        "state_selection stream — trials are never silently dropped"
    ),
    "checks": (
        "belief logits/probabilities byte-identical",
        "public legal-rank masks byte-identical",
        "sampler request byte-identical",
        "fixed-seed sampled world byte-identical",
        "sampler provenance byte-identical",
        "instrumented hidden-input access counters zero",
    ),
    "injection_controls": (
        "requests carrying private fields (true rank, private piece table, "
        "opponent setup truth, hidden start rank, winner/result/reward, "
        "future action/search result, storage path) must be rejected "
        "structurally"
    ),
    "zero_tolerance_counters": INFORMATION_SAFETY_ZERO_COUNTERS,
}

#: Gate G's required deterministic topology/restart legs, exactly.
REPRODUCIBILITY_TOPOLOGY_LEGS = (
    "workers_1",
    "workers_4",
    "workers_12",
    "forward_order",
    "reverse_order",
    "round_robin_sharded",
    "fresh_process",
    "kill_resume_set_subtraction",
)

REPRODUCIBILITY_SPECIFICATION = {
    "request_set": (
        f"{REPRO_REQUEST_COUNT} frozen requests: per stratum, the distinct "
        "validation public states ordered by public_state_identity, taking "
        f"the first {REPRO_REQUEST_COUNT // len(OPPONENT_STRATA)} — a "
        "hash-order deterministic rule consuming no randomness"
    ),
    "request_id_format": "phase11_repro_request_v1|ms=<master>|n=<ordinal:05d>",
    "request_content": (
        "one belief forward plus complete worlds for sample ordinals 0..63, "
        "digested canonically (beliefs, masks, worlds, provenance)"
    ),
    "legs": REPRODUCIBILITY_TOPOLOGY_LEGS,
    "comparison": (
        "the canonical digest of every request must be identical across "
        "every leg; one differing byte fails Gate G"
    ),
    "purity": (
        "a sampled world is a pure function of public-state identity, "
        "belief-model identity, sampler identity and sample ordinal; worker "
        "count, call order, process id, path, wall clock and previous calls "
        "must be absent from every derivation"
    ),
}

#: The frozen runtime benchmark configuration. Backend and device are fixed
#: *now*, before any measurement exists, and may not move after results.
RUNTIME_BENCHMARK_CONFIGURATION = {
    "backend": "cpu",
    "dtype": "float32",
    "torch_threads": 1,
    "process_model": "single process, single request at a time",
    "state_count": BENCHMARK_STATE_COUNT,
    "state_selection": (
        f"{BENCHMARK_STATES_PER_CELL} states per (stratum x colour x "
        f"progress bucket) cell over the {BENCHMARK_CELL_COUNT} cells: "
        "order the cell's distinct validation public states by unresolved-"
        "piece count then public_state_identity and take evenly spaced "
        "picks, so unresolved-count variation is covered deterministically; "
        "a cell with fewer states contributes what it has (recorded)"
    ),
    "measured_configurations": (
        "forward_only",
        "forward_plus_16_worlds",
        "forward_plus_32_worlds",
        "forward_plus_64_worlds",
    ),
    "warmup": "32 global warmup requests, then 1 discarded warmup per state",
    "timer": "time.perf_counter_ns around the complete request",
    "components": (
        "model-forward time and sampling time recorded separately per "
        "request, plus RSS and backend/device"
    ),
    "statistics": "median/p90/p95/p99/max over states, per configuration",
    "gate_quantity": "p95(forward_plus_64_worlds)",
    "ceiling_ms": 500.0,
}

# ---------------------------------------------------------------------------
# Acceptance gates
# ---------------------------------------------------------------------------

GATE_A = {
    "gate": "A",
    "name": "predictive superiority",
    "r_ce_max": 0.97,
    "r_ce_max_strict": False,
    "ce_delta_upper_max": 0.0,
    "ce_delta_upper_max_strict": True,
    "ce_delta": "paired 95% bootstrap upper bound of CE_learned - CE_baseline",
}

GATE_B = {
    "gate": "B",
    "name": "top-1 improvement",
    "delta_top1_min": 0.03,
    "delta_top1_min_strict": False,
    "delta_top1_lower_min": 0.0,
    "delta_top1_lower_min_strict": True,
    "delta_top1": "paired 95% bootstrap lower bound of top1_learned - top1_baseline",
}

GATE_C = {
    "gate": "C",
    "name": "calibration",
    "ece_overall_max": 0.08,
    "ece_overall_max_strict": False,
    "stratum_ece_max": 0.12,
    "stratum_ece_max_strict": False,
    "brier_delta_upper_max": 0.01,
    "brier_delta_upper_max_strict": False,
    "brier_delta": "paired 95% bootstrap upper bound of Brier_learned - Brier_baseline",
}

GATE_D = {
    "gate": "D",
    "name": "robustness",
    "stratum_r_ce_max": 1.05,
    "stratum_r_ce_max_strict": False,
    "scope": "every one of the eight opponent strata",
}

GATE_E = {
    "gate": "E",
    "name": "sampler correctness",
    "zero_tolerance": SAMPLER_ZERO_TOLERANCE_COUNTERS,
}

GATE_F = {
    "gate": "F",
    "name": "information safety",
    "zero_tolerance": INFORMATION_SAFETY_ZERO_COUNTERS,
}

GATE_G = {
    "gate": "G",
    "name": "reproducibility and runtime",
    "topology_legs": REPRODUCIBILITY_TOPOLOGY_LEGS,
    "requires_all_exact": True,
    "p95_forward_64_max_ms": RUNTIME_BENCHMARK_CONFIGURATION["ceiling_ms"],
    "p95_forward_64_max_ms_strict": False,
}

GATE_H = {
    "gate": "H",
    "name": "preservation",
    "phase9_checkpoint_sha256": ACCEPTED_PHASE9_CHECKPOINT_SHA256,
    "phase9_model_state_digest": ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
    "phase9_parameters": ACCEPTED_PHASE9_PARAMETERS,
    "phase11_optimizer_steps": 0,
    "global_optimizer_step": ACCEPTED_GLOBAL_OPTIMIZER_STEP,
    "belief_head_digest": ACCEPTED_BELIEF_HEAD_DIGEST,
    "selector_config_sha256": ACCEPTED_SELECTOR_CONFIG_SHA256,
    "utility_coefficient_digest": ACCEPTED_UTILITY_COEFFICIENT_DIGEST,
    "trait_scaler_digest": ACCEPTED_TRAIT_SCALER_DIGEST,
    "phase7_library_content_digest": PHASE7_LIBRARY_CONTENT_DIGEST,
}

HARD_GATES = (GATE_A, GATE_B, GATE_C, GATE_D, GATE_E, GATE_F, GATE_G, GATE_H)
HARD_GATE_IDS = tuple(gate["gate"] for gate in HARD_GATES)
assert HARD_GATE_IDS == ("A", "B", "C", "D", "E", "F", "G", "H")

INEQUALITY_SEMANTICS = (
    "thresholds marked strict use the strict operator ('<' or '>') and "
    "unmarked thresholds use the non-strict one ('<=' or '>='); every gate "
    "states its own strictness explicitly; a non-finite quantity fails its "
    "comparison"
)

CLASSIFICATIONS = {
    "PASS-SEARCH-READY": "all eight hard gates A-H pass",
    "FAIL": "the experiment runs correctly but at least one hard gate fails",
    "BLOCKED": (
        "integrity, sealing or prerequisite evidence cannot be established; "
        "the sealed test is not opened"
    ),
}

DIAGNOSTIC_RULE = "report-only diagnostics never rescue a failed gate"

STOP_CONDITIONS = {
    "BLOCKED": (
        "an accepted upstream identity mismatches",
        "the belief-head tensor identity cannot be re-derived",
        "test-bank scored access is observed before Agent 7",
        "exact bank rebuild cannot be established",
        "sealing or ledger integrity fails",
    ),
    "FAIL": (
        "a hard gate A-H fails on the sealed test",
    ),
    "never": (
        "retrain, refit or calibrate the belief head",
        "change P10-D, the utility, the scaler or Phase 7",
        "turn this validation phase into a repair loop",
        "rerun the sealed test to rescue a result",
    ),
}


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def evaluate_gate_a(r_ce: float, ce_delta_upper: float) -> dict:
    """Gate A: R_CE <= 0.97 and paired CE-delta 95% upper bound < 0."""
    checks = {
        "r_ce_le_0_97": _finite(r_ce) and r_ce <= GATE_A["r_ce_max"],
        "ce_delta_upper_lt_0": _finite(ce_delta_upper)
        and ce_delta_upper < GATE_A["ce_delta_upper_max"],
    }
    return {"gate": "A", "checks": checks, "passed": all(checks.values())}


def evaluate_gate_b(delta_top1: float, delta_top1_lower: float) -> dict:
    """Gate B: Delta_top1 >= +0.03 and paired 95% lower bound > 0."""
    checks = {
        "delta_top1_ge_0_03": _finite(delta_top1)
        and delta_top1 >= GATE_B["delta_top1_min"],
        "delta_top1_lower_gt_0": _finite(delta_top1_lower)
        and delta_top1_lower > GATE_B["delta_top1_lower_min"],
    }
    return {"gate": "B", "checks": checks, "passed": all(checks.values())}


def evaluate_gate_c(
    ece_overall: float, stratum_ece: "dict[str, float]", brier_delta_upper: float
) -> dict:
    """Gate C: overall ECE, every-stratum ECE, and the Brier-delta bound."""
    strata_complete = tuple(sorted(stratum_ece)) == tuple(sorted(OPPONENT_STRATA))
    checks = {
        "ece_overall_le_0_08": _finite(ece_overall)
        and ece_overall <= GATE_C["ece_overall_max"],
        "all_strata_present": strata_complete,
        "no_stratum_ece_gt_0_12": strata_complete
        and all(
            _finite(value) and value <= GATE_C["stratum_ece_max"]
            for value in stratum_ece.values()
        ),
        "brier_delta_upper_le_0_01": _finite(brier_delta_upper)
        and brier_delta_upper <= GATE_C["brier_delta_upper_max"],
    }
    return {"gate": "C", "checks": checks, "passed": all(checks.values())}


def evaluate_gate_d(stratum_r_ce: "dict[str, float]") -> dict:
    """Gate D: every opponent stratum's R_CE <= 1.05."""
    strata_complete = tuple(sorted(stratum_r_ce)) == tuple(sorted(OPPONENT_STRATA))
    checks = {
        "all_strata_present": strata_complete,
        "every_stratum_r_ce_le_1_05": strata_complete
        and all(
            _finite(value) and value <= GATE_D["stratum_r_ce_max"]
            for value in stratum_r_ce.values()
        ),
    }
    return {"gate": "D", "checks": checks, "passed": all(checks.values())}


def _zero_counter_gate(gate: str, expected: "tuple[str, ...]", counters: dict) -> dict:
    complete = tuple(sorted(counters)) == tuple(sorted(expected))
    checks = {
        "all_counters_present": complete,
        "all_counters_zero": complete
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value == 0
            for value in counters.values()
        ),
    }
    return {"gate": gate, "checks": checks, "passed": all(checks.values())}


def evaluate_gate_e(counters: dict) -> dict:
    """Gate E: every sampler-correctness counter present and zero."""
    return _zero_counter_gate("E", SAMPLER_ZERO_TOLERANCE_COUNTERS, counters)


def evaluate_gate_f(counters: dict) -> dict:
    """Gate F: every information-safety counter present and zero."""
    return _zero_counter_gate("F", INFORMATION_SAFETY_ZERO_COUNTERS, counters)


def evaluate_gate_g(leg_exact: "dict[str, bool]", p95_forward_64_ms: float) -> dict:
    """Gate G: all topology legs exact and p95(forward+64) <= 500 ms."""
    legs_complete = tuple(sorted(leg_exact)) == tuple(
        sorted(REPRODUCIBILITY_TOPOLOGY_LEGS)
    )
    checks = {
        "all_legs_present": legs_complete,
        "all_legs_exact": legs_complete
        and all(value is True for value in leg_exact.values()),
        "p95_forward_64_le_500ms": _finite(p95_forward_64_ms)
        and p95_forward_64_ms <= GATE_G["p95_forward_64_max_ms"],
    }
    return {"gate": "G", "checks": checks, "passed": all(checks.values())}


def evaluate_gate_h(observed: dict) -> dict:
    """Gate H: every preservation identity exact and zero optimizer steps."""
    expected = {key: value for key, value in GATE_H.items() if key not in ("gate", "name")}
    checks = {
        key: observed.get(key) == value for key, value in sorted(expected.items())
    }
    return {"gate": "H", "checks": checks, "passed": all(checks.values())}


def classify_phase11(
    gate_passed: "dict[str, bool]",
    *,
    experiment_valid: bool = True,
    integrity_established: bool = True,
) -> str:
    """The frozen three-way classification, recomputed from gate booleans.

    `BLOCKED` outranks everything: without established integrity there is no
    experiment to grade. A valid experiment classifies purely from the eight
    gate booleans; no discretionary override exists.
    """
    if not integrity_established:
        return "BLOCKED"
    if not experiment_valid:
        return "BLOCKED"
    if tuple(sorted(gate_passed)) != tuple(sorted(HARD_GATE_IDS)):
        raise Phase11ContractError(
            f"classification needs exactly the gates {list(HARD_GATE_IDS)}, got "
            f"{sorted(gate_passed)}"
        )
    if all(gate_passed[gate] is True for gate in HARD_GATE_IDS):
        return "PASS-SEARCH-READY"
    return "FAIL"


# ---------------------------------------------------------------------------
# Bank access ledger
# ---------------------------------------------------------------------------

LEDGER_VERSION = "phase11_bank_ledger_v1"
LEDGER_RELATIVE_PATH = "reports/phase_11_data/phase11_bank_access_ledger.jsonl"

#: The append-only ledger entry schema, exactly.
LEDGER_ENTRY_FIELDS = (
    "ledger_version",
    "agent",
    "stage",
    "bank_version",
    "purpose",
    "structural_only",
    "neural_inference_count",
    "scored_prediction_count",
    "privileged_truth_count",
    "outcome_count",
)

LEDGER_RULES = (
    "append-only: entries are never edited or removed",
    "every agent-harness bank access writes one entry before that agent's "
    "artifacts freeze; suite tests re-exercise the same structural code "
    "paths under the recorded suite measurement rather than writing "
    "per-invocation entries",
    "before Agent 7, every phase11_test_bank_v1 entry must carry "
    "structural_only=true and all four counters zero",
    "Agent 7 harvests the complete ledger as the first-scored-access proof",
)

# ---------------------------------------------------------------------------
# The eight frozen documents
# ---------------------------------------------------------------------------


def belief_contract_document() -> dict:
    """`phase11_belief_contract_v1` — scope, upstream identities, semantics."""
    return {
        "contract_version": BELIEF_CONTRACT_VERSION,
        "identity_version": PHASE11_IDENTITY_VERSION,
        "mission": (
            "validate that the accepted Phase 9 belief head produces accurate, "
            "calibrated, information-safe, reproducible beliefs about hidden "
            "opponent ranks, and that those marginals convert into complete "
            "legal hidden worlds fast enough for Phase 12 search; a validation "
            "phase, never a training or repair phase"
        ),
        "phase10_closure_commit": PHASE10_CLOSURE_COMMIT,
        "accepted_belief_model": {
            "path": ACCEPTED_PHASE9_CHECKPOINT_PATH,
            "sha256": ACCEPTED_PHASE9_CHECKPOINT_SHA256,
            "model_state_digest": ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
            "parameters": ACCEPTED_PHASE9_PARAMETERS,
            "c1_config_digest": ACCEPTED_C1_CONFIG_DIGEST,
            "global_optimizer_step": ACCEPTED_GLOBAL_OPTIMIZER_STEP,
            "model_label": BELIEF_MODEL_LABEL,
            "belief_head": {
                "tensor_names": list(BELIEF_HEAD_TENSOR_NAMES),
                "tensor_shapes": {
                    name: list(shape)
                    for name, shape in sorted(BELIEF_HEAD_TENSOR_SHAPES.items())
                },
                "digest": ACCEPTED_BELIEF_HEAD_DIGEST,
                "digest_recipe": (
                    "sha256 over (name, str(shape), float32 C-order bytes) of "
                    "the belief-head tensors in sorted name order — the "
                    "accepted model-state-digest recipe restricted to the head"
                ),
            },
        },
        "accepted_phase10_selector": {
            "candidate_id": ACCEPTED_SELECTOR_CANDIDATE_ID,
            "utility_model": ACCEPTED_SELECTOR_UTILITY_MODEL,
            "temperature": ACCEPTED_SELECTOR_TEMPERATURE,
            "selector_identity": ACCEPTED_SELECTOR_IDENTITY,
            "mixture": {
                "neutral_weight": ACCEPTED_SELECTOR_NEUTRAL_WEIGHT,
                "learned_weight": ACCEPTED_SELECTOR_LEARNED_WEIGHT,
            },
            "config_sha256": ACCEPTED_SELECTOR_CONFIG_SHA256,
            "utility_coefficient_digest": ACCEPTED_UTILITY_COEFFICIENT_DIGEST,
            "trait_scaler_digest": ACCEPTED_TRAIT_SCALER_DIGEST,
            "utility_file_sha256": ACCEPTED_UTILITY_FILE_SHA256,
            "phase10_system_digest": ACCEPTED_PHASE10_SYSTEM_DIGEST,
        },
        "accepted_phase8_anchor": {
            "path": ACCEPTED_ANCHOR_EXPORT_PATH,
            "sha256": ACCEPTED_ANCHOR_EXPORT_SHA256,
        },
        "frozen_phase7_stack": {
            "library_version": PHASE7_LIBRARY_VERSION,
            "content_digest": PHASE7_LIBRARY_CONTENT_DIGEST,
            "metadata_digest": PHASE7_LIBRARY_METADATA_DIGEST,
            "manifest_digest": PHASE7_LIBRARY_MANIFEST_DIGEST,
            "frozen_identities": list(FROZEN_PHASE7_IDENTITIES),
        },
        "frozen_runtime_identities": dict(FROZEN_RUNTIME_IDENTITIES),
        "rank_space": {
            "rank_names": list(RANK_NAMES),
            "rank_count": RANK_COUNT,
            "initial_counts": list(RANK_INITIAL_COUNTS),
            "immovable_rank_indices": list(IMMOVABLE_RANK_INDICES),
            "order_authority": (
                "the accepted engine PIECE_TYPE_NAMES enumeration — the exact "
                "index order the belief head was trained under "
                "(dense_belief_target_v1); index i of every Phase 11 12-vector "
                "means rank_names[i]"
            ),
        },
        "belief_target": {
            "hidden_targets": (
                "live opponent pieces whose exact rank the observer may not "
                "legally know (the accepted engine belief_target semantics: "
                "combat reveals and public Scout multi-square deductions make "
                "a piece known)"
            ),
            "recording_rule": (
                "prediction events are recorded at every decision where the "
                "observer is the acting player in a non-terminal state — the "
                "same forward that chooses the observer's move — one event "
                "per hidden target; a decision with an empty hidden-target "
                "set contributes no events (allowed, counted)"
            ),
            "known_exclusion": (
                "publicly known ranks (own pieces, revealed opponent pieces, "
                "captured pieces, empty squares, lakes) are never prediction "
                "events and can never inflate hidden-rank accuracy"
            ),
            "probability_extraction": METRIC_FORMULAS["probability_source"],
            "legal_rank_mask": (
                "12 binary entries; a publicly moved hidden piece excludes "
                "flag and bomb, an unmoved one excludes nothing — the mask "
                "records movement impossibility only and never consults "
                "counts; recorded, and consumed by the sampler and baseline, "
                "never applied to the learned scoring vector"
            ),
            "zero_true_rank_probability": (
                f"CE floors the scored probability at {LOG_PROBABILITY_FLOOR} "
                "inside the logarithm only; stored vectors are unfloored and "
                "floored events increment the report-only log_floor_events "
                "diagnostic"
            ),
            "progress_buckets": [dict(entry) for entry in PROGRESS_BUCKETS],
            "moved_unmoved": (
                "an event is 'moved' exactly when the target piece's public "
                "has_moved flag is true at the decision"
            ),
            "decision_index": (
                "the pre-action total_moves of the observer decision that "
                "produced the forward"
            ),
        },
        "prediction_record": {
            "record_version": PREDICTION_RECORD_VERSION,
            "fields": list(PREDICTION_RECORD_FIELDS),
            "privileged_fields": list(PRIVILEGED_RECORD_FIELDS),
            "privileged_rule": (
                "true_rank_index is written by the privileged evaluator only "
                "after the prediction vectors exist, on a scoring path "
                "isolated from production inference"
            ),
        },
        "public_state_document": {
            "document_version": PUBLIC_STATE_DOCUMENT_VERSION,
            "fields": list(PUBLIC_STATE_DOCUMENT_FIELDS),
            "piece_fields": list(PUBLIC_PIECE_FIELDS),
            "identity": (
                "public_state_identity = sha256 over the document's canonical "
                "JSON; the document embeds observation_sha256 (sha256 over the "
                "127x10x10 float32 C-order observation bytes), so the identity "
                "covers the complete model input and the sampled-world purity "
                "claim is exact"
            ),
            "observer_relative": (
                "the document holds exactly what the observer may legally "
                "see; ranks appear only through the known-to-observer gate"
            ),
        },
        "production_request": {
            "request_version": BELIEF_REQUEST_VERSION,
            "allowed_fields": list(ALLOWED_BELIEF_REQUEST_FIELDS),
            "forbidden_tokens": list(FORBIDDEN_BELIEF_REQUEST_TOKENS),
            "rule": (
                "the Agent 2 production request type must reject any field "
                "outside the allowlist structurally (raise, never drop), the "
                "accepted SelectorRequest.from_payload pattern"
            ),
        },
        "non_goals": list(NON_GOALS),
        "phase9_preservation_invariant": PHASE9_PRESERVATION_INVARIANT,
        "seeds": seed_derivation_document(),
        "stop_conditions": {key: list(value) for key, value in STOP_CONDITIONS.items()},
        "storage_semantics": (
            "logical identities are path-independent; prediction storage may "
            "live on the external volume behind a tracked pointer, and a "
            "pointer naming an absent volume is BLOCKED, never a silent "
            "internal replacement"
        ),
    }


def baseline_document() -> dict:
    """`phase11_belief_baseline_v1` — the two frozen baselines, exactly."""
    return {
        "contract_version": BASELINE_CONTRACT_VERSION,
        "baseline_count": 2,
        "remaining_count_belief_v1": {
            "version": REMAINING_COUNT_BASELINE_VERSION,
            "role": "primary predictive baseline",
            "inputs": "public information only; hidden truth is unreadable",
            "remaining_inventory": (
                "c[r] = initial_counts[r] - known[r], where known[r] counts "
                "opponent pieces of rank r whose exact rank the observer "
                "legally knows, alive or captured (every captured piece is "
                "known through its combat reveal)"
            ),
            "per_piece_distribution": (
                "q[r] = c[r] * mask[r] / sum_r' c[r'] * mask[r'], float64, "
                "where mask is the piece's public legal-rank mask"
            ),
            "well_definedness": (
                "the denominator is at least 1 in every legal public state: "
                "the piece's own true rank contributes a positive masked "
                "count, so the baseline's true-rank probability is always "
                "positive"
            ),
            "count_conservation": (
                "sum_r c[r] equals the number of unresolved opponent pieces; "
                "checked as a baseline correctness invariant"
            ),
        },
        "count_uniform_world_sampler_v1": {
            "version": WORLD_BASELINE_VERSION,
            "role": (
                "search fallback / joint-sampling baseline only; never a "
                "predictive-gate input"
            ),
            "algorithm": (
                "the frozen belief_sampler_v1 skeleton (same identity streams "
                "under its own sampler-version token, same piece order, same "
                "categorical walk, same feasibility rule, same validation "
                "stack) with the step-7 weight replaced by remaining_count "
                "alone — no learned factor anywhere"
            ),
            "stream_separation": (
                "its sample tokens carry smp=count_uniform_world_sampler_v1, "
                "so every stream is domain-separated from the learned "
                "sampler's"
            ),
        },
        "no_hidden_truth": (
            "neither baseline may read hidden truth; both are pure functions "
            "of the public-state document (plus, for the world sampler, the "
            "sample ordinal)"
        ),
    }


def bank_document() -> dict:
    """`phase11_belief_bank_v1` — bank structure, balance, sealing, ledger."""
    return {
        "contract_version": BANK_CONTRACT_VERSION,
        "master_seed": PHASE11_MASTER_SEED,
        "banks": [
            {
                "bank": "validation",
                "bank_version": VALIDATION_BANK_VERSION,
                "split": BANK_SPLITS["validation"],
                "case_count": VALIDATION_BANK_CASES,
                "game_count": VALIDATION_BANK_GAMES,
                "cases_per_stratum": VALIDATION_CASES_PER_CELL * len(SETUP_SOURCES),
                "cases_per_cell": VALIDATION_CASES_PER_CELL,
                "bootstrap_root": CANONICAL_PHASE11_SEEDS["validation_bootstrap_seed"],
            },
            {
                "bank": "test",
                "bank_version": TEST_BANK_VERSION,
                "split": BANK_SPLITS["test"],
                "case_count": TEST_BANK_CASES,
                "game_count": TEST_BANK_GAMES,
                "cases_per_stratum": TEST_CASES_PER_CELL * len(SETUP_SOURCES),
                "cases_per_cell": TEST_CASES_PER_CELL,
                "bootstrap_root": CANONICAL_PHASE11_SEEDS["test_bootstrap_seed"],
            },
        ],
        "split_reading": (
            "the common contract names no Phase 7 split; the frozen reading "
            "follows the accepted Phase 9/10 precedent — the validation bank "
            "draws every setup from the validation split, the test bank from "
            "the test split"
        ),
        "strata": [dict(entry) for entry in STRATUM_BINDINGS],
        "stratum_order": list(OPPONENT_STRATA),
        "setup_sources": list(SETUP_SOURCES),
        "observer": {
            "policy": (
                "the accepted Phase 9 policy + belief head, on the observer "
                "seat of every game of every stratum"
            ),
            "checkpoint_sha256": ACCEPTED_PHASE9_CHECKPOINT_SHA256,
            "setup_source": dict(OBSERVER_SETUP_SOURCE),
            "move_behavior": dict(EVAL_MOVE_BEHAVIOR),
        },
        "opponent_setup_sources": {
            token: dict(entry) for token, entry in sorted(OPPONENT_SETUP_SOURCES.items())
        },
        "case_structure": {
            "case_id_rule": (
                "case_index = ((stratum_index * 2) + source_index) * "
                "cases_per_cell + ordinal, strata in the frozen order, "
                "sources p10d then neutral — balance over strata, sources "
                "and colours is a property of the id space"
            ),
            "colour_pairing": (
                "the observer plays Red in game 0 and Blue in game 1 against "
                "the same opponent stratum"
            ),
            "setup_draws": (
                "each seat of each game draws its own setup from its frozen "
                "source conditioned on its colour, under "
                "case_setup_seed(case_id, game_index, role) — four draws per "
                "case, materialized and hashed at bank construction, so a "
                "P10-D seat's colour-conditional distribution is never "
                "distorted by mirroring"
            ),
            "match_seeds": (
                "game_match_seed(game_id), independent of everything but the "
                "frozen game identity; rule opponents draw their randomness "
                "from the accepted runner derivations rooted here"
            ),
            "no_rejection": (
                "draws are pure first-attempt draws from the frozen sources — "
                "no fingerprint-isolation or distinctness rejection, because "
                "Phase 11 selects nothing and rejection would distort the "
                "production distributions the belief system must be measured "
                "under (an Agent 1 reading, recorded)"
            ),
            "bootstrap_unit": "the logical case, both colour games together",
        },
        "outcomes": "game outcomes are report-only and rank nothing",
        "isolated_rebuild": (
            "every case rebuilds from its case id alone, independent of "
            "worker count, order and path"
        ),
        "sealing": {
            "sealed_until": "Agent 7",
            "allowed_before_agent_7": [
                "structural_build",
                "structural_audit",
                "digest_computation",
                "structural_artifact_write",
            ],
            "forbidden_before_agent_7": [
                "neural inference on a test case",
                "playing any game on a test case",
                "scoring any prediction on a test case",
                "reading privileged truth on a test case",
                "reading any outcome of a test case",
            ],
        },
        "ledger": {
            "ledger_version": LEDGER_VERSION,
            "path": LEDGER_RELATIVE_PATH,
            "entry_fields": list(LEDGER_ENTRY_FIELDS),
            "rules": list(LEDGER_RULES),
        },
    }


def metrics_document() -> dict:
    """`phase11_belief_metrics_v1` — metrics, aggregation, bootstrap."""
    return {
        "contract_version": METRICS_CONTRACT_VERSION,
        "formulas": {
            key: (dict(value) if isinstance(value, dict) else value)
            for key, value in METRIC_FORMULAS.items()
        },
        "log_probability_floor": LOG_PROBABILITY_FLOOR,
        "statistics": dict(STATISTICS),
        "overall_metric_tokens": list(OVERALL_METRIC_TOKENS),
        "stratum_token_rule": "'<metric_token>|st=<stratum>'",
        "diagnostic_slices": list(DIAGNOSTIC_SLICES),
        "slice_aggregation": (
            "gate-feeding stratum metrics aggregate that stratum's complete "
            "cases exactly as the overall metrics do; every other slice is a "
            "report-only pooled-event mean"
        ),
        "validation_role": (
            "validation-bank values are diagnostics and readiness evidence; "
            "they never retune weights, thresholds, bins, baselines, banks or "
            "strata"
        ),
    }


def sampler_document() -> dict:
    """`phase11_belief_sampler_v1` — the frozen sampler mathematics."""
    return {
        "contract_version": SAMPLER_CONTRACT_VERSION,
        "sampler_version": BELIEF_SAMPLER_VERSION,
        "weighting": "weight = learned_probability * remaining_count",
        "algorithm_steps": list(SAMPLER_ALGORITHM_STEPS),
        "feasibility_rule": dict(SAMPLER_FEASIBILITY_RULE),
        "piece_order": (
            "ascending (world_order_key(sample_token, piece_slot), "
            "piece_slot) over the unresolved pieces"
        ),
        "categorical_draw": (
            "inverse-CDF walk of world_categorical_uniform(sample_token, "
            "step_index) over the ranks in frozen rank-index order, float64 "
            "cumulative mass, last legal rank as the float tail guard"
        ),
        "zero_mass_fallback": (
            "when every legal weight is zero, that step reweights by the "
            "normalized remaining counts over the same legal set; with the "
            "feasibility rule this set is provably non-empty, and every "
            "fallback increments the report-only fallback diagnostic"
        ),
        "count_decrement": "the chosen rank's remaining count decrements by one",
        "validation_stack": list(WORLD_VALIDATION_STACK),
        "provenance_fields": list(SAMPLER_PROVENANCE_FIELDS),
        "identity_inputs": [
            "public_state_identity",
            "belief_model_identity (label selfplay_c1_v1, digest-bound)",
            "sampler_version",
            "sample_ordinal",
        ],
        "purity": (
            "a sampled world is a pure function of the identity inputs; "
            "worker count, call order, process id, path, wall clock and "
            "previous calls appear in no derivation and no mutable RNG "
            "cursor exists"
        ),
        "request_boundary": {
            "allowed_fields": list(ALLOWED_SAMPLER_REQUEST_FIELDS),
            "forbidden_tokens": list(FORBIDDEN_SAMPLER_REQUEST_TOKENS),
            "rejected_inputs": [
                "true rank",
                "private piece table",
                "opponent setup truth",
                "hidden start rank",
                "winner/result/reward",
                "future action/search result",
                "storage path",
            ],
        },
        "zero_tolerance_counters": list(SAMPLER_ZERO_TOLERANCE_COUNTERS),
        "report_only_diagnostics": list(SAMPLER_REPORT_ONLY_DIAGNOSTICS),
        "audit_volumes": {
            "large_audit_min_worlds": SAMPLER_AUDIT_MIN_WORLDS,
            "independent_audit_min_worlds": SAMPLER_INDEPENDENT_AUDIT_MIN_WORLDS,
            "coverage": (
                "thousands of distinct validation public states spanning all "
                "8 strata, both colours, early/middle/late and moved/unmoved "
                "uncertainty"
            ),
        },
        "diversity_thresholds": (
            "none frozen — world-diversity statistics are report-only, as the "
            "common contract provides when Agent 1 freezes no threshold"
        ),
    }


def information_safety_document() -> dict:
    """`phase11_information_safety_v1` — attack, reproducibility, runtime."""
    return {
        "contract_version": INFORMATION_SAFETY_VERSION,
        "hidden_truth_attack": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in INFORMATION_SAFETY_ATTACK.items()
        },
        "reproducibility": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in REPRODUCIBILITY_SPECIFICATION.items()
        },
        "runtime_benchmark": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in RUNTIME_BENCHMARK_CONFIGURATION.items()
        },
        "backend_discipline": (
            "the benchmark backend/device is frozen here, before any "
            "measurement exists; switching after seeing results is forbidden"
        ),
        "soak": {
            "soak_version": "phase11_soak_v1",
            "games_per_stratum": SOAK_GAMES_PER_STRATUM,
            "game_count": SOAK_GAME_COUNT,
            "requests_per_game": SOAK_REQUESTS_PER_GAME,
            "request_count": SOAK_REQUEST_COUNT,
            "split": "train",
            "setup_source": (
                "both seats draw from the accepted P10-D production source "
                "under soak_setup_seed — the production-shaped exercise"
            ),
            "observer_color_rule": "red on even game ordinals, blue on odd",
            "request_rule": (
                "request k of a game attaches to observer-decision index "
                "min(floor(k * D / 8), D - 1) of the game's D observer "
                "decisions (the game is deterministic, so the attachment is "
                "a pure function of the game identity); each request runs "
                "one real belief forward plus worlds for sample ordinals "
                "0..63; two requests landing on one decision of a short game "
                "must produce byte-identical worlds — purity, demonstrated"
            ),
            "restart_rule": (
                "at least three legs with different worker counts, one real "
                "SIGKILL after committed work exists, resume by exact "
                "request-id set subtraction; the final store holds exactly "
                "the 8,192 scheduled ids"
            ),
            "outcome_rule": "soak outcomes/results are report-only",
        },
    }


def acceptance_document() -> dict:
    """`phase11_acceptance_v1` — the eight hard gates and the classification."""
    return {
        "acceptance_version": ACCEPTANCE_VERSION,
        "hard_gates": [
            {
                key: (list(value) if isinstance(value, tuple) else value)
                for key, value in gate.items()
            }
            for gate in HARD_GATES
        ],
        "hard_gate_ids": list(HARD_GATE_IDS),
        "gate_count": len(HARD_GATES),
        "all_gates_are_hard": True,
        "inequality_semantics": INEQUALITY_SEMANTICS,
        "classifications": dict(CLASSIFICATIONS),
        "classification_rule": (
            "exactly one of PASS-SEARCH-READY / FAIL / BLOCKED, recomputed "
            "from the gate booleans by classify_phase11; no discretionary "
            "override"
        ),
        "diagnostic_rule": DIAGNOSTIC_RULE,
        "statistics": dict(STATISTICS),
        "evaluated_on": TEST_BANK_VERSION,
        "evaluated_by": "Agent 7, first and only sealed scored evaluation",
        "on_fail": (
            "Phase 12 is not authorized; a separate belief-repair phase must "
            "be designed"
        ),
    }


def system_document() -> dict:
    """`phase11_system_v1` — the binding template of the finished system.

    Agent 1 freezes the binding rules and the slots that already have
    accepted values. The evaluator, sampler implementation, safety evidence
    and runtime result do not exist yet; inventing placeholders for them
    would be pre-commitment. Agent 6 fills the unbound slots at the
    production freeze, and Agent 7 verifies the filled document against
    these rules.
    """
    return {
        "system_version": SYSTEM_VERSION,
        "bound_now": {
            "belief_model": {
                "path": ACCEPTED_PHASE9_CHECKPOINT_PATH,
                "sha256": ACCEPTED_PHASE9_CHECKPOINT_SHA256,
                "model_state_digest": ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
                "parameters": ACCEPTED_PHASE9_PARAMETERS,
                "belief_head_tensor_names": list(BELIEF_HEAD_TENSOR_NAMES),
                "belief_head_digest": ACCEPTED_BELIEF_HEAD_DIGEST,
                "model_label": BELIEF_MODEL_LABEL,
                "mutability": "byte-identical throughout Phase 11",
            },
            "setup_selector": {
                "selector_identity": ACCEPTED_SELECTOR_IDENTITY,
                "config_sha256": ACCEPTED_SELECTOR_CONFIG_SHA256,
                "utility_coefficient_digest": ACCEPTED_UTILITY_COEFFICIENT_DIGEST,
                "trait_scaler_digest": ACCEPTED_TRAIT_SCALER_DIGEST,
                "phase10_system_digest": ACCEPTED_PHASE10_SYSTEM_DIGEST,
            },
            "library": {
                "library_version": PHASE7_LIBRARY_VERSION,
                "content_digest": PHASE7_LIBRARY_CONTENT_DIGEST,
                "metadata_digest": PHASE7_LIBRARY_METADATA_DIGEST,
                "manifest_digest": PHASE7_LIBRARY_MANIFEST_DIGEST,
            },
            "baselines": {
                "predictive": REMAINING_COUNT_BASELINE_VERSION,
                "world": WORLD_BASELINE_VERSION,
            },
            "sampler_contract": BELIEF_SAMPLER_VERSION,
            "bank_versions": [VALIDATION_BANK_VERSION, TEST_BANK_VERSION],
            "acceptance_version": ACCEPTANCE_VERSION,
            "observation": FROZEN_RUNTIME_IDENTITIES["observation"],
            "model_contract": FROZEN_RUNTIME_IDENTITIES["model_contract"],
        },
        "unbound_slots": [
            {
                "slot": "evaluator_implementation",
                "filled_by": "Agent 6",
                "requires": [
                    f"evaluator version {EVALUATOR_VERSION}",
                    "the accepted Agent 2 implementation identity/digest",
                ],
            },
            {
                "slot": "sampler_implementation",
                "filled_by": "Agent 6",
                "requires": [
                    f"sampler version {BELIEF_SAMPLER_VERSION}",
                    "the accepted Agent 3 implementation digest",
                    "the Agent 3 audit evidence digest",
                ],
            },
            {
                "slot": "information_safety_evidence",
                "filled_by": "Agent 6",
                "requires": [
                    f"info-safety contract {INFORMATION_SAFETY_VERSION}",
                    "the Agent 4 attack/reproducibility evidence digests",
                ],
            },
            {
                "slot": "runtime_benchmark",
                "filled_by": "Agent 6",
                "requires": [
                    "the frozen benchmark configuration, unchanged",
                    "the measured p95 forward+64 result and its artifact digest",
                ],
            },
            {
                "slot": "bank_digests",
                "filled_by": "Agent 6",
                "requires": [
                    "the Agent 1 validation and test bank digests, verbatim",
                ],
            },
        ],
        "filling_rules": (
            "Agent 6 fills every unbound slot with accepted values only, "
            "changes nothing bound now, and adds no slot; Agent 7 verifies "
            "the filled instance against this template by slot walk, judging "
            "on values"
        ),
        "phase12_rule": (
            "if Phase 11 ends PASS-SEARCH-READY, the filled phase11_system_v1 "
            "is the only belief stack Phase 12 may query"
        ),
        "no_absolute_paths": "no absolute path appears in any logical identity",
    }


#: The eight frozen contracts, in the order the common contract lists them.
CONTRACT_BUILDERS = (
    (BELIEF_CONTRACT_VERSION, belief_contract_document),
    (BASELINE_CONTRACT_VERSION, baseline_document),
    (BANK_CONTRACT_VERSION, bank_document),
    (METRICS_CONTRACT_VERSION, metrics_document),
    (SAMPLER_CONTRACT_VERSION, sampler_document),
    (INFORMATION_SAFETY_VERSION, information_safety_document),
    (ACCEPTANCE_VERSION, acceptance_document),
    (SYSTEM_VERSION, system_document),
)
CONTRACT_VERSIONS = tuple(name for name, _ in CONTRACT_BUILDERS)
assert len(CONTRACT_VERSIONS) == 8 == len(set(CONTRACT_VERSIONS))


def contract_documents() -> dict:
    """Every frozen Phase 11 contract document, keyed by its version."""
    return {name: builder() for name, builder in CONTRACT_BUILDERS}


def contract_digests(documents: "dict | None" = None) -> dict:
    """The stable SHA-256 of every frozen contract document."""
    documents = contract_documents() if documents is None else documents
    return {name: document_digest(documents[name]) for name in CONTRACT_VERSIONS}


def contract_bundle_digest(documents: "dict | None" = None) -> str:
    """One digest over the eight contracts — the Phase 11 freeze identity."""
    return document_digest(contract_digests(documents))


__all__ = [
    "ACCEPTANCE_VERSION",
    "ACCEPTED_ANCHOR_EXPORT_PATH",
    "ACCEPTED_ANCHOR_EXPORT_SHA256",
    "ACCEPTED_BELIEF_HEAD_DIGEST",
    "ACCEPTED_GLOBAL_OPTIMIZER_STEP",
    "ACCEPTED_PHASE10_SYSTEM_DIGEST",
    "ACCEPTED_SELECTOR_CANDIDATE_ID",
    "ACCEPTED_SELECTOR_CONFIG_SHA256",
    "ACCEPTED_SELECTOR_IDENTITY",
    "ACCEPTED_SELECTOR_LEARNED_WEIGHT",
    "ACCEPTED_SELECTOR_NEUTRAL_WEIGHT",
    "ACCEPTED_SELECTOR_TEMPERATURE",
    "ACCEPTED_SELECTOR_UTILITY_MODEL",
    "ACCEPTED_TRAIT_SCALER_DIGEST",
    "ACCEPTED_UTILITY_COEFFICIENT_DIGEST",
    "ACCEPTED_UTILITY_FILE_SHA256",
    "ALLOWED_BELIEF_REQUEST_FIELDS",
    "ALLOWED_SAMPLER_REQUEST_FIELDS",
    "BANK_CONTRACT_VERSION",
    "BANK_SPLITS",
    "BASELINE_CONTRACT_VERSION",
    "BELIEF_CONTRACT_VERSION",
    "BELIEF_HEAD_TENSOR_NAMES",
    "BELIEF_HEAD_TENSOR_SHAPES",
    "BELIEF_REQUEST_VERSION",
    "BELIEF_SAMPLER_VERSION",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_REPLICATES",
    "CLASSIFICATIONS",
    "CONTRACT_BUILDERS",
    "CONTRACT_VERSIONS",
    "DIAGNOSTIC_RULE",
    "DIAGNOSTIC_SLICES",
    "ECE_SPECIFICATION",
    "EVALUATOR_VERSION",
    "EVAL_MOVE_BEHAVIOR",
    "FORBIDDEN_BELIEF_REQUEST_TOKENS",
    "FORBIDDEN_SAMPLER_REQUEST_TOKENS",
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
    "IMMOVABLE_RANK_INDICES",
    "INEQUALITY_SEMANTICS",
    "INFORMATION_SAFETY_ATTACK",
    "INFORMATION_SAFETY_VERSION",
    "INFORMATION_SAFETY_ZERO_COUNTERS",
    "LEDGER_ENTRY_FIELDS",
    "LEDGER_RELATIVE_PATH",
    "LEDGER_RULES",
    "LEDGER_VERSION",
    "LOG_PROBABILITY_FLOOR",
    "METRICS_CONTRACT_VERSION",
    "METRIC_FORMULAS",
    "MOVABLE_RANK_INDICES",
    "NON_GOALS",
    "OBSERVER_SETUP_SOURCE",
    "OPPONENT_SETUP_SOURCES",
    "OVERALL_METRIC_TOKENS",
    "PHASE10_CLOSURE_COMMIT",
    "PHASE9_PRESERVATION_INVARIANT",
    "PREDICTION_RECORD_FIELDS",
    "PREDICTION_RECORD_VERSION",
    "PRIVILEGED_RECORD_FIELDS",
    "PROGRESS_BUCKETS",
    "PROGRESS_BUCKET_NAMES",
    "PUBLIC_PIECE_FIELDS",
    "PUBLIC_STATE_DOCUMENT_FIELDS",
    "PUBLIC_STATE_DOCUMENT_VERSION",
    "RANK_COUNT",
    "RANK_INITIAL_COUNTS",
    "RANK_NAMES",
    "REMAINING_COUNT_BASELINE_VERSION",
    "REPRODUCIBILITY_SPECIFICATION",
    "REPRODUCIBILITY_TOPOLOGY_LEGS",
    "RUNTIME_BENCHMARK_CONFIGURATION",
    "SAMPLER_ALGORITHM_STEPS",
    "SAMPLER_AUDIT_MIN_WORLDS",
    "SAMPLER_CONTRACT_VERSION",
    "SAMPLER_FEASIBILITY_RULE",
    "SAMPLER_INDEPENDENT_AUDIT_MIN_WORLDS",
    "SAMPLER_PROVENANCE_FIELDS",
    "SAMPLER_REPORT_ONLY_DIAGNOSTICS",
    "SAMPLER_ZERO_TOLERANCE_COUNTERS",
    "STATISTICS",
    "STOP_CONDITIONS",
    "STRATUM_BINDINGS",
    "SYSTEM_VERSION",
    "TEST_BANK_CASES",
    "TEST_BANK_GAMES",
    "TEST_BANK_VERSION",
    "TEST_CASES_PER_CELL",
    "VALIDATION_BANK_CASES",
    "VALIDATION_BANK_GAMES",
    "VALIDATION_BANK_VERSION",
    "VALIDATION_CASES_PER_CELL",
    "WORLD_BASELINE_VERSION",
    "WORLD_VALIDATION_STACK",
    "Phase11ContractError",
    "acceptance_document",
    "bank_document",
    "baseline_document",
    "belief_contract_document",
    "classify_phase11",
    "contract_bundle_digest",
    "contract_digests",
    "contract_documents",
    "document_digest",
    "evaluate_gate_a",
    "evaluate_gate_b",
    "evaluate_gate_c",
    "evaluate_gate_d",
    "evaluate_gate_e",
    "evaluate_gate_f",
    "evaluate_gate_g",
    "evaluate_gate_h",
    "information_safety_document",
    "metrics_document",
    "progress_bucket",
    "sampler_document",
    "system_document",
]
