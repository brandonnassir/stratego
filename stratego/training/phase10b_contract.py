"""Optional Phase 10B: the frozen contract of the setup-conditioned experiment.

Specification source: `OPTIONAL_PHASE_10B_SETUP_CONDITIONED_FINE_TUNING_AGENT.md`.

What this module is
-------------------
Every learning-design constant Phase 10B is allowed to have, in one place,
frozen before the first rollout exists. It holds no state, touches no file
and reads no outcome: a reviewer can diff this module against the plan and
know the whole experiment's physics.

What Phase 10B inherits and never restates
------------------------------------------
The PPO objective, the KL controller, the advantage construction, the
optimizer constraints and the hard safety limits are the **accepted Phase 9**
ones, imported from :mod:`stratego.training.phase9_contract` rather than
copied. The plan freezes exactly two departures — a decayed learning rate and
a lower entropy schedule — and both are defined here as explicit multiples of
the accepted Phase 9 canonical values, so the relationship stays auditable.

What Phase 10B may never touch
------------------------------
The accepted Phase 9 checkpoint, the P10-D selector config, the Phase 10
utility and scaler, and the Phase 7 library. Those identities appear here as
*expectations to verify*, never as things to write.
"""

from __future__ import annotations

import hashlib
import json

from .phase9_contract import (
    ADVANTAGE_FILTER_FLOOR,
    ADVANTAGE_FILTER_QUANTILE,
    ADVANTAGE_STANDARDIZATION_EPSILON,
    BEHAVIOR_KL_TARGET,
    BELIEF_LOSS_WEIGHT,
    CLIP_FRACTION_HARD_LIMIT,
    EPOCHS_PER_ROLLOUT,
    GAMMA,
    KL_BETA_DECREASE_FACTOR,
    KL_BETA_DECREASE_THRESHOLD,
    KL_BETA_INCREASE_FACTOR,
    KL_BETA_INCREASE_THRESHOLD,
    KL_BETA_MAX,
    KL_BETA_MIN,
    KL_HARD_LIMIT,
    LAMBDA_ADVANTAGE,
    LAMBDA_VALUE,
    MINIBATCH_SIZE,
    OPTIMIZER_CONSTRAINTS,
    PPO_CLIP_EPSILON,
    VALUE_LOSS_WEIGHT,
    contract_digest as phase9_contract_digest,
)

PHASE10B_CONTRACT_VERSION = "phase10b_contract_v1"
PHASE10B_ROLLOUT_VERSION = "phase10b_rollout_v1"
PHASE10B_POPULATION_VERSION = "phase10b_population_v1"
PHASE10B_SCHEDULE_VERSION = "phase10b_rollout_schedule_v1"
PHASE10B_TRAINER_VERSION = "phase10b_trainer_v1"
PHASE10B_STORE_VERSION = "phase10b_rollout_store_v1"
PHASE10B_VALIDATION_BANK_VERSION = "phase10b_validation_bank_v1"
PHASE10B_TEST_BANK_VERSION = "phase10b_test_bank_v1"
PHASE10B_ACCEPTANCE_VERSION = "phase10b_acceptance_v1"

#: The single run namespace. Phase 10B is one experiment, not a matrix.
PHASE10B_NAMESPACE = "phase10b"


class Phase10BContractError(RuntimeError):
    """Raised when the frozen Phase 10B contract is violated or unusable."""


# ---------------------------------------------------------------------------
# Frozen upstream identities (verified from live bytes, never written)
# ---------------------------------------------------------------------------

ACCEPTED_PHASE9_CHECKPOINT = "checkpoints/phase9/selfplay_c1_v1.pt"
ACCEPTED_PHASE9_SHA256 = (
    "dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea"
)
ACCEPTED_PHASE9_STATE_DIGEST = (
    "f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd"
)
ACCEPTED_PHASE9_PARAMETERS = 863_959
ACCEPTED_C1_CONFIG_DIGEST = (
    "31ca84ab140c523e65567787b0289fe0dbdf5ab0344667410a5fda7060cfe07d"
)

SELECTED_CANDIDATE_ID = "P10-D"
SELECTOR_UTILITY_MODEL = "model_T"
SELECTOR_TEMPERATURE = 0.75
NEUTRAL_MIXTURE_WEIGHT = 0.35
LEARNED_MIXTURE_WEIGHT = 0.65
ACCEPTED_SELECTOR_CONFIG_SHA256 = (
    "6e227815bc3cb44f19cdeee55d00ec0ae75726fb411ee9131660aa712bb86668"
)
ACCEPTED_MODEL_T_COEFFICIENT_DIGEST = (
    "d898782a2ae7cf4ed1cb2833fad6e53d8407ec2048dafbd34a6a20c1c9766edc"
)
ACCEPTED_TRAIT_SCALER_DIGEST = (
    "fa6eb1c112defc4c1034831b84db8848181e1f674f8439c9c265916d89e8b7f9"
)
ACCEPTED_PHASE10_SYSTEM_DIGEST = (
    "615cc3c3a4fab6e4400e20a5a93b13a08c43ab6c3ca63828c6a64742e98175d2"
)
ACCEPTED_LIBRARY_CONTENT_DIGEST = (
    "7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777"
)
ACCEPTED_LIBRARY_METADATA_DIGEST = (
    "d86f486182a820d546d470ef4ebce92ff60c6259aed80c481bc985bce8c64980"
)

#: Every upstream artifact Gate H requires to be byte-identical afterwards.
UPSTREAM_PRESERVED_ARTIFACTS = (
    "checkpoints/phase9/selfplay_c1_v1.pt",
    "checkpoints/phase10/setup_utility_v1.json",
    "reports/phase_10_data/agent_05_frozen_selector_config.json",
    "reports/phase_10_data/agent_06_production_selector_manifest.json",
    "data/setups/setup_library_v1.jsonl",
)


# ---------------------------------------------------------------------------
# Population mixture — exact scheduled counts, never sampled
# ---------------------------------------------------------------------------

BUCKET_CURRENT = "current"
BUCKET_ANCHOR = "anchor"
BUCKET_ARCHIVE = "archive"
BUCKET_OPPONENT = "opponent"

#: Declared bucket order. Ties in the largest-remainder allocation below are
#: broken by this order, so the counts are a function of the plan alone.
POPULATION_BUCKETS = (BUCKET_CURRENT, BUCKET_ANCHOR, BUCKET_ARCHIVE, BUCKET_OPPONENT)

#: The plan's proportions of one iteration.
POPULATION_PROPORTIONS = {
    BUCKET_CURRENT: 0.60,
    BUCKET_ANCHOR: 0.20,
    BUCKET_ARCHIVE: 0.10,
    BUCKET_OPPONENT: 0.10,
}

GAMES_PER_ITERATION = 2048

#: Exact per-bucket counts, the largest-remainder allocation of the plan's
#: proportions over 2,048 games with ties broken by declared bucket order.
BUCKET_COUNTS = {
    BUCKET_CURRENT: 1229,
    BUCKET_ANCHOR: 409,
    BUCKET_ARCHIVE: 205,
    BUCKET_OPPONENT: 205,
}

#: The rule/stress roster inside the opponent bucket, in declared order with
#: the plan's within-bucket shares.
OPPONENT_ROSTER = (
    ("strategic_rule_based", 0.30),
    ("tactical_rule_based", 0.25),
    ("basic_heuristic", 0.10),
    ("stress_information_miser", 0.10),
    ("stress_scout_rush", 0.10),
    ("stress_miner_rush", 0.10),
    ("random_legal", 0.05),
)

#: Exact per-policy counts inside the 205-game opponent bucket, same rule.
OPPONENT_COUNTS = {
    "strategic_rule_based": 62,
    "tactical_rule_based": 51,
    "basic_heuristic": 21,
    "stress_information_miser": 21,
    "stress_scout_rush": 20,
    "stress_miner_rush": 20,
    "random_legal": 10,
}

#: Contiguous ordinal subranges inside the opponent bucket, roster order.
OPPONENT_ORDER = tuple(policy_id for policy_id, _share in OPPONENT_ROSTER)


def largest_remainder(total: int, shares, order) -> dict:
    """The frozen exact allocation of `total` seats over `shares`.

    Floor every exact share, then hand the remaining seats to the largest
    fractional parts, breaking ties by position in `order`. Deterministic,
    outcome-independent, and reproducible from the plan's percentages alone —
    which is why the counts above can be checked rather than trusted.
    """
    exact = {key: total * float(shares[key]) for key in order}
    counts = {key: int(exact[key]) for key in order}
    remaining = total - sum(counts.values())
    ranked = sorted(
        order,
        key=lambda key: (-(exact[key] - counts[key]), order.index(key)),
    )
    for key in ranked[:remaining]:
        counts[key] += 1
    return counts


def bucket_counts() -> dict:
    """The frozen per-bucket counts, re-derived and cross-checked."""
    derived = largest_remainder(
        GAMES_PER_ITERATION, POPULATION_PROPORTIONS, POPULATION_BUCKETS
    )
    if derived != BUCKET_COUNTS:
        raise Phase10BContractError(
            f"the declared bucket counts {BUCKET_COUNTS} are not the largest-"
            f"remainder allocation {derived} of the plan's proportions"
        )
    return dict(BUCKET_COUNTS)


def opponent_counts() -> dict:
    """The frozen per-policy counts of the opponent bucket, cross-checked."""
    shares = dict(OPPONENT_ROSTER)
    derived = largest_remainder(
        BUCKET_COUNTS[BUCKET_OPPONENT], shares, OPPONENT_ORDER
    )
    if derived != OPPONENT_COUNTS:
        raise Phase10BContractError(
            f"the declared opponent counts {OPPONENT_COUNTS} are not the "
            f"largest-remainder allocation {derived} of the plan's shares"
        )
    return dict(OPPONENT_COUNTS)


def opponent_policy_for_ordinal(ordinal: int) -> str:
    """The frozen Phase 4 policy of one opponent-bucket ordinal.

    Contiguous subranges in roster order — the same mechanism Phase 9 uses
    for its rule tiers, so per-iteration counts are exact by construction
    rather than by a draw that has to be audited afterwards.
    """
    counts = opponent_counts()
    total = sum(counts.values())
    if not 0 <= ordinal < total:
        raise Phase10BContractError(
            f"opponent ordinal {ordinal} is outside 0..{total - 1}"
        )
    cursor = 0
    for policy_id in OPPONENT_ORDER:
        cursor += counts[policy_id]
        if ordinal < cursor:
            return policy_id
    raise Phase10BContractError("unreachable: opponent ordinal beyond every policy")


# ---------------------------------------------------------------------------
# Learner control and colour balance
# ---------------------------------------------------------------------------

LEARNER_CONTROL_RED = "red"
LEARNER_CONTROL_BLUE = "blue"
LEARNER_CONTROL_BOTH = "both"

#: Which sides of each bucket receive Phase 10B policy/value/belief loss.
TRAINING_ELIGIBILITY = {
    BUCKET_CURRENT: "both colors",
    BUCKET_ANCHOR: "current-policy side only",
    BUCKET_ARCHIVE: "current-policy side only",
    BUCKET_OPPONENT: "current-policy side only",
}


def learner_color(bucket: str, iteration: int, ordinal: int):
    """The learner's colour in one asymmetric game; `None` for self-play.

    The accepted Phase 9 parity rule, unchanged: red when
    ``(ordinal + iteration) % 2 == 0``. Any even-sized ordinal range splits
    exactly in half and an odd range's single remainder alternates with
    iteration parity.
    """
    if bucket not in POPULATION_BUCKETS:
        raise Phase10BContractError(f"unknown bucket: {bucket!r}")
    if iteration < 1:
        raise Phase10BContractError(f"iteration must be >= 1, got {iteration}")
    if ordinal < 0:
        raise Phase10BContractError(f"ordinal must be >= 0, got {ordinal}")
    if bucket == BUCKET_CURRENT:
        return None
    return "red" if (ordinal + iteration) % 2 == 0 else "blue"


def learner_control_for(bucket: str, iteration: int, ordinal: int) -> str:
    color = learner_color(bucket, iteration, ordinal)
    return LEARNER_CONTROL_BOTH if color is None else color


# ---------------------------------------------------------------------------
# History policy
# ---------------------------------------------------------------------------

#: The accepted Phase 9 checkpoint's identity as Phase 10B history member zero.
#: It is never evicted and is the permanent rollback anchor.
ANCHOR_IDENTITY = "P9A"
ARCHIVE_CADENCE_ITERATIONS = 5
ACTIVE_ARCHIVE_WINDOW = 4


def archive_snapshot_id(iteration: int) -> str:
    if iteration < 1:
        raise Phase10BContractError(f"iteration must be >= 1, got {iteration}")
    if iteration % ARCHIVE_CADENCE_ITERATIONS != 0:
        raise Phase10BContractError(
            f"iteration {iteration} is off the frozen archive cadence of "
            f"{ARCHIVE_CADENCE_ITERATIONS}"
        )
    # `A005` is the archive slot created after iteration 5; `B006` is the
    # behavior snapshot that collects iteration 6. They hold the *same*
    # weights and are deliberately spelled differently, because they answer
    # different questions and appear side by side in the history manifest.
    return f"A{iteration:03d}"


def archived_iterations_before(iteration: int) -> tuple:
    if iteration < 1:
        raise Phase10BContractError(f"iteration must be >= 1, got {iteration}")
    return tuple(
        past
        for past in range(
            ARCHIVE_CADENCE_ITERATIONS, iteration, ARCHIVE_CADENCE_ITERATIONS
        )
    )


def active_archive_window(iteration: int) -> tuple:
    """The archive bucket's frozen sampling pool at one iteration, oldest first.

    The plan freezes "the accepted Phase 9 anchor + up to 4 most recent
    eligible Phase 10B archives", sampled uniformly, with the anchor never
    evicted. Iterations 1-5 have produced no Phase 10B archive yet, so the
    pool is exactly the anchor: the count stays exact, the draw stays
    outcome-independent, and no future checkpoint is ever fabricated to fill
    a schedule. The consequence — the anchor carries the whole 30% opponent-
    checkpoint share until iteration 5 archives — is declared, not discovered.
    """
    recent = archived_iterations_before(iteration)[-ACTIVE_ARCHIVE_WINDOW:]
    return (ANCHOR_IDENTITY,) + tuple(
        archive_snapshot_id(past) for past in recent
    )


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 30
MAX_TRAINING_GAMES = MAX_ITERATIONS * GAMES_PER_ITERATION
MAX_OPTIMIZER_EPOCHS = MAX_ITERATIONS * EPOCHS_PER_ROLLOUT
WALL_CLOCK_CEILING_HOURS = 12
WALL_CLOCK_CEILING_SECONDS = WALL_CLOCK_CEILING_HOURS * 3600

VALIDATION_CADENCE_ITERATIONS = 5
VALIDATION_ITERATIONS = tuple(
    range(VALIDATION_CADENCE_ITERATIONS, MAX_ITERATIONS + 1, VALIDATION_CADENCE_ITERATIONS)
)


# ---------------------------------------------------------------------------
# The two Phase 10B departures from the accepted Phase 9 schedule
# ---------------------------------------------------------------------------

#: The accepted Phase 9 canonical starting learning rate (candidate P9-C),
#: read from the accepted pilot matrix rather than retyped.
PHASE9_CANONICAL_LEARNING_RATE = 3e-4
PHASE9_CANONICAL_INITIAL_KL_BETA = 0.005

INITIAL_LEARNING_RATE_FACTOR = 0.25
FINAL_LEARNING_RATE_FACTOR = 0.10
INITIAL_LEARNING_RATE = INITIAL_LEARNING_RATE_FACTOR * PHASE9_CANONICAL_LEARNING_RATE
FINAL_LEARNING_RATE = FINAL_LEARNING_RATE_FACTOR * PHASE9_CANONICAL_LEARNING_RATE

ENTROPY_COEFFICIENT_START = 0.0010
ENTROPY_COEFFICIENT_END = 0.0005


def _linear(start: float, end: float, iteration: int) -> float:
    if not 1 <= iteration <= MAX_ITERATIONS:
        raise Phase10BContractError(
            f"iteration {iteration} is outside 1..{MAX_ITERATIONS}"
        )
    progress = (iteration - 1) / (MAX_ITERATIONS - 1)
    return start + progress * (end - start)


def learning_rate(iteration: int) -> float:
    """The frozen linear LR decay, constant within an iteration.

    Iteration 1 uses 0.25x and iteration 30 uses 0.10x of the accepted Phase 9
    canonical starting rate. The intent the plan states is to adapt rather
    than relearn; no post-hoc adjustment is permitted in either direction.
    """
    return _linear(INITIAL_LEARNING_RATE, FINAL_LEARNING_RATE, iteration)


def entropy_coefficient(iteration: int) -> float:
    """The frozen linear entropy schedule, constant within an iteration."""
    return _linear(ENTROPY_COEFFICIENT_START, ENTROPY_COEFFICIENT_END, iteration)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

MATCHUP_DIRECT = "direct_p10d"
MATCHUP_NEUTRAL = "neutral_rollback"
MATCHUP_STRATEGIC = "strategic"
MATCHUP_TACTICAL = "tactical"
MATCHUP_PHASE8 = "phase8_anchor"
MATCHUP_RANDOM = "random"
MATCHUP_BASIC = "basic"

MATCHUP_TOKENS = (
    MATCHUP_DIRECT,
    MATCHUP_NEUTRAL,
    MATCHUP_STRATEGIC,
    MATCHUP_TACTICAL,
    MATCHUP_PHASE8,
    MATCHUP_RANDOM,
    MATCHUP_BASIC,
)

#: Matchups whose delta is candidate-minus-Phase9 on identical logical cases.
PAIRED_DELTA_MATCHUPS = (
    MATCHUP_STRATEGIC,
    MATCHUP_TACTICAL,
    MATCHUP_PHASE8,
    MATCHUP_RANDOM,
    MATCHUP_BASIC,
)

#: The two head-to-head matchups: one arm per side, so there is no baseline
#: arm to evaluate separately.
HEAD_TO_HEAD_MATCHUPS = (MATCHUP_DIRECT, MATCHUP_NEUTRAL)

VALIDATION_CASES_PER_MATCHUP = 256
TEST_CASES_PER_MATCHUP = 512
GAMES_PER_CASE = 2

#: The frozen validation score. Random and Basic are guardrails, never terms.
VALIDATION_SCORE_WEIGHTS = {
    "delta_direct": 0.40,
    "delta_neutral": 0.20,
    "delta_strategic": 0.15,
    "delta_tactical": 0.15,
    "delta_phase8": 0.10,
}

VALIDATION_TIE_BREAK = (
    "higher S10B",
    "higher delta_direct",
    "higher delta_neutral",
    "higher delta_strategic",
    "lower behavior KL",
    "earlier iteration",
)

VALIDATION_ELIGIBILITY = {
    "random_ewr_min": 0.95,
    "basic_ewr_min": 0.80,
    "neutral_rollback_ewr_min": 0.48,
}


def validation_score(deltas: dict) -> float:
    """`S10B` from the five frozen delta terms."""
    missing = sorted(set(VALIDATION_SCORE_WEIGHTS) - set(deltas))
    if missing:
        raise Phase10BContractError(f"the validation score needs {missing}")
    return sum(
        weight * float(deltas[name])
        for name, weight in VALIDATION_SCORE_WEIGHTS.items()
    )


# ---------------------------------------------------------------------------
# Final hard gates
# ---------------------------------------------------------------------------

STRONG_OPPONENT_WEIGHTS = {
    MATCHUP_STRATEGIC: 0.45,
    MATCHUP_TACTICAL: 0.35,
    MATCHUP_PHASE8: 0.20,
}

FINAL_GATES = {
    "gate_a_direct_adaptation": {
        "ewr_min": 0.52,
        "paired_lower_bound_min": 0.50,
    },
    "gate_b_neutral_rollback": {
        "ewr_min": 0.49,
        "paired_lower_bound_min": 0.47,
    },
    "gate_c_strong_composite": {
        "point_min": 0.00,
        "lower_bound_min": -0.02,
    },
    "gate_d_individual_regression": {
        "lower_bound_min": -0.03,
        "opponents": list(STRONG_OPPONENT_WEIGHTS),
    },
    "gate_e_easy_opponents": {
        "random_overall_min": 0.95,
        "random_red_min": 0.90,
        "random_blue_min": 0.90,
        "basic_min": 0.80,
        "paired_lower_bound_min": -0.03,
    },
    "gate_f_training_safety": {
        "hard_kl_violations_max": 0,
        "hard_clip_fraction_violations_max": 0,
        "nonfinite_losses_max": 0,
        "nonfinite_gradients_max": 0,
        "optimizer_corruption_max": 0,
        "illegal_training_actions_max": 0,
    },
    "gate_g_belief_preservation": {
        "ce_ratio_max": 1.05,
        "top1_degradation_max": 0.02,
    },
    "gate_h_upstream_preservation": {
        "byte_identical": list(UPSTREAM_PRESERVED_ARTIFACTS),
    },
}

HARD_GATE_IDS = tuple(sorted(FINAL_GATES))

CLASSIFICATIONS = ("PASS-CANDIDATE", "FAIL", "BLOCKED")

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95


def strong_opponent_composite(deltas: dict) -> float:
    """`Delta_L` over the three strong opponents."""
    missing = sorted(set(STRONG_OPPONENT_WEIGHTS) - set(deltas))
    if missing:
        raise Phase10BContractError(f"the strong-opponent composite needs {missing}")
    return sum(
        weight * float(deltas[token])
        for token, weight in STRONG_OPPONENT_WEIGHTS.items()
    )


# ---------------------------------------------------------------------------
# The contract document
# ---------------------------------------------------------------------------


def contract_document() -> dict:
    """Everything Phase 10B freezes, as one reviewable JSON-able document."""
    return {
        "contract_version": PHASE10B_CONTRACT_VERSION,
        "status": "optional side experiment; advisory only; never auto-promoted",
        "namespace": PHASE10B_NAMESPACE,
        "versions": {
            "rollout": PHASE10B_ROLLOUT_VERSION,
            "population": PHASE10B_POPULATION_VERSION,
            "schedule": PHASE10B_SCHEDULE_VERSION,
            "trainer": PHASE10B_TRAINER_VERSION,
            "store": PHASE10B_STORE_VERSION,
            "validation_bank": PHASE10B_VALIDATION_BANK_VERSION,
            "test_bank": PHASE10B_TEST_BANK_VERSION,
            "acceptance": PHASE10B_ACCEPTANCE_VERSION,
        },
        "upstream": {
            "phase9_checkpoint": ACCEPTED_PHASE9_CHECKPOINT,
            "phase9_sha256": ACCEPTED_PHASE9_SHA256,
            "phase9_model_state_digest": ACCEPTED_PHASE9_STATE_DIGEST,
            "phase9_parameters": ACCEPTED_PHASE9_PARAMETERS,
            "c1_config_digest": ACCEPTED_C1_CONFIG_DIGEST,
            "selector_candidate": SELECTED_CANDIDATE_ID,
            "selector_utility_model": SELECTOR_UTILITY_MODEL,
            "selector_temperature": SELECTOR_TEMPERATURE,
            "mixture_neutral": NEUTRAL_MIXTURE_WEIGHT,
            "mixture_learned": LEARNED_MIXTURE_WEIGHT,
            "selector_config_sha256": ACCEPTED_SELECTOR_CONFIG_SHA256,
            "model_t_coefficient_digest": ACCEPTED_MODEL_T_COEFFICIENT_DIGEST,
            "trait_scaler_digest": ACCEPTED_TRAIT_SCALER_DIGEST,
            "phase10_system_digest": ACCEPTED_PHASE10_SYSTEM_DIGEST,
            "library_content_digest": ACCEPTED_LIBRARY_CONTENT_DIGEST,
            "library_metadata_digest": ACCEPTED_LIBRARY_METADATA_DIGEST,
            "read_only": True,
        },
        "setup_conditioning": {
            "red_setup_source": SELECTED_CANDIDATE_ID,
            "blue_setup_source": SELECTED_CANDIDATE_ID,
            "split": "train",
            "rule": (
                "both sides draw independently through their own colour-"
                "specific frozen distribution; no draw may read the opponent's "
                "setup, hidden ranks, outcome prediction, checkpoint strength "
                "or matchup identity"
            ),
        },
        "population": {
            "games_per_iteration": GAMES_PER_ITERATION,
            "proportions": dict(POPULATION_PROPORTIONS),
            "bucket_counts": bucket_counts(),
            "opponent_counts": opponent_counts(),
            "allocation_rule": (
                "largest remainder over the plan's percentages, ties broken by "
                "declared order"
            ),
            "training_eligibility": dict(TRAINING_ELIGIBILITY),
            "colour_balance": "learner is red when (ordinal + iteration) % 2 == 0",
        },
        "history": {
            "anchor": ANCHOR_IDENTITY,
            "anchor_is": "the accepted Phase 9 checkpoint; never evicted",
            "archive_cadence_iterations": ARCHIVE_CADENCE_ITERATIONS,
            "active_archive_window": ACTIVE_ARCHIVE_WINDOW,
            "sampling": "uniform over the active window; no performance weighting",
            "pre_archive_rule": (
                "before iteration 5 archives, the archive bucket's pool is "
                "exactly the anchor"
            ),
        },
        "budget": {
            "max_iterations": MAX_ITERATIONS,
            "games_per_iteration": GAMES_PER_ITERATION,
            "max_training_games": MAX_TRAINING_GAMES,
            "max_optimizer_epochs": MAX_OPTIMIZER_EPOCHS,
            "wall_clock_ceiling_hours": WALL_CLOCK_CEILING_HOURS,
            "stop_rule": (
                "the earliest of 30 completed iterations, the 12-hour ceiling, "
                "or a hard safety stop; never extended because results are weak"
            ),
        },
        "optimization": {
            "inherited_from": "accepted Phase 9 PPO/KL machinery",
            "phase9_contract_digest": phase9_contract_digest(),
            "ppo_clip_epsilon": PPO_CLIP_EPSILON,
            "behavior_kl_target": BEHAVIOR_KL_TARGET,
            "kl_hard_limit": KL_HARD_LIMIT,
            "clip_fraction_hard_limit": CLIP_FRACTION_HARD_LIMIT,
            "kl_beta_increase_threshold": KL_BETA_INCREASE_THRESHOLD,
            "kl_beta_decrease_threshold": KL_BETA_DECREASE_THRESHOLD,
            "kl_beta_increase_factor": KL_BETA_INCREASE_FACTOR,
            "kl_beta_decrease_factor": KL_BETA_DECREASE_FACTOR,
            "kl_beta_min": KL_BETA_MIN,
            "kl_beta_max": KL_BETA_MAX,
            "initial_kl_beta": PHASE9_CANONICAL_INITIAL_KL_BETA,
            "value_loss_weight": VALUE_LOSS_WEIGHT,
            "belief_loss_weight": BELIEF_LOSS_WEIGHT,
            "gamma": GAMMA,
            "lambda_advantage": LAMBDA_ADVANTAGE,
            "lambda_value": LAMBDA_VALUE,
            "advantage_filter_quantile": ADVANTAGE_FILTER_QUANTILE,
            "advantage_filter_floor": ADVANTAGE_FILTER_FLOOR,
            "advantage_standardization_epsilon": ADVANTAGE_STANDARDIZATION_EPSILON,
            "minibatch_size": MINIBATCH_SIZE,
            "epochs_per_rollout": EPOCHS_PER_ROLLOUT,
            "optimizer": dict(OPTIMIZER_CONSTRAINTS),
            "replay": "none",
            "search_in_training": "forbidden",
        },
        "phase10b_schedule_departures": {
            "rule": (
                "the plan explicitly freezes these two and only these two; "
                "every other optimization constant is the accepted Phase 9 one"
            ),
            "learning_rate": {
                "phase9_canonical_start": PHASE9_CANONICAL_LEARNING_RATE,
                "initial_factor": INITIAL_LEARNING_RATE_FACTOR,
                "final_factor": FINAL_LEARNING_RATE_FACTOR,
                "initial": INITIAL_LEARNING_RATE,
                "final": FINAL_LEARNING_RATE,
                "schedule": f"linear decay across {MAX_ITERATIONS} iterations",
                "phase9_schedule_replaced": OPTIMIZER_CONSTRAINTS[
                    "learning_rate_schedule"
                ],
                "post_hoc_change": "forbidden in either direction",
            },
            "entropy_coefficient": {
                "start": ENTROPY_COEFFICIENT_START,
                "end": ENTROPY_COEFFICIENT_END,
                "schedule": f"linear across {MAX_ITERATIONS} iterations",
            },
        },
        "validation": {
            "cadence_iterations": VALIDATION_CADENCE_ITERATIONS,
            "iterations": list(VALIDATION_ITERATIONS),
            "cases_per_matchup": VALIDATION_CASES_PER_MATCHUP,
            "games_per_case": GAMES_PER_CASE,
            "purpose": "checkpoint selection only",
            "may_not_change": [
                "learning rate",
                "population mix",
                "setup selector",
                "PPO thresholds",
                "number of iterations",
                "entropy schedule",
            ],
            "score_weights": dict(VALIDATION_SCORE_WEIGHTS),
            "tie_break": list(VALIDATION_TIE_BREAK),
            "eligibility": dict(VALIDATION_ELIGIBILITY),
            "guardrails_not_score_terms": ["random", "basic"],
        },
        "final_evaluation": {
            "cases_per_matchup": TEST_CASES_PER_MATCHUP,
            "games_per_case": GAMES_PER_CASE,
            "matchups": list(MATCHUP_TOKENS),
            "first_and_only": True,
            "bootstrap": {
                "method": "paired-unit percentile bootstrap over logical cases",
                "replicates": BOOTSTRAP_REPLICATES,
                "confidence": BOOTSTRAP_CONFIDENCE,
            },
            "gates": {key: dict(value) for key, value in FINAL_GATES.items()},
            "strong_opponent_weights": {
                key: value for key, value in STRONG_OPPONENT_WEIGHTS.items()
            },
            "classifications": list(CLASSIFICATIONS),
        },
        "prohibitions": [
            "alter the accepted Phase 9 checkpoint",
            "alter P10-D",
            "refit the Phase 10 utility model",
            "change the P10-D temperature",
            "change the 0.35/0.65 mixture",
            "change neutral_v1",
            "mutate the Phase 7 setup library",
            "change the 127-channel observation design",
            "change the belief-head architecture",
            "introduce search into training",
            "use Phase 11 or Phase 12 test evidence for training",
            "use Phase 10 final-test outcomes as training data",
            "change Phase 9 PPO/KL mechanics beyond the two frozen departures",
            "run a 168-hour final training budget",
            "block, pause or alter Phase 11 execution",
        ],
        "promotion": (
            "no automatic promotion; a PASS-CANDIDATE result returns to the "
            "reviewing chat and changes no Phase 9, Phase 10, Phase 11 or "
            "Phase 12 artifact"
        ),
    }


def canonical_json(document) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def contract_digest() -> str:
    """The digest stamped into every Phase 10B rollout sidecar and checkpoint."""
    return hashlib.sha256(canonical_json(contract_document()).encode()).hexdigest()


__all__ = [name for name in dir() if not name.startswith("_")]
