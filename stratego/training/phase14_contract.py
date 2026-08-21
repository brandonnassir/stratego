"""Phase 14: the frozen final-training contract, as executable constants.

Specification source: `reports/phase13/phase13_final_training_contract_v1.json`
(Phase 13 Agent 1, status FROZEN), via
`instructions/phase_13_final_training_integration/02_AGENT_2_FINAL_TRAINING_INTEGRATION.md`.

What this module is
-------------------
Every value the 168-hour run needs, resolved to a number, in one place. Agent 1
froze them; this module restates them so code can consume them and adds one
thing a JSON document cannot: :func:`verify_against_frozen_contract`, which
re-reads Agent 1's document from disk and reports every disagreement. The
constants below are therefore not a second source of truth — they are a
*checked copy*, and the check runs in the test suite and again at run start.

What Phase 14 inherits and never restates
-----------------------------------------
The PPO objective, the value and belief-auxiliary objectives and their weights,
the adaptive KL controller, the advantage construction and filter, the
optimizer family and the hard safety limits are the **accepted Phase 9** ones,
imported from :mod:`stratego.training.phase9_contract`. Agent 1 froze exactly
four departures, and each is defined here against its accepted Phase 9 value so
the relationship stays auditable:

1. a conservative continuation learning rate (0.25x LR9 main, 0.125x LR9 late);
2. a constant entropy coefficient at the accepted schedule's terminal value;
3. a new opponent mixture with a wall-clock main/late transition;
4. a new bounded historical pool, `phase14_active_pool_v1`.

Search
------
Absent. Nothing in this module, or in anything the Phase 14 training path
imports, reaches :mod:`stratego.search`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .phase9_contract import (
    ADVANTAGE_FILTER_FLOOR,
    ADVANTAGE_FILTER_QUANTILE,
    ADVANTAGE_STANDARDIZATION_EPSILON,
    BEHAVIOR_KL_TARGET,
    BEHAVIOR_PROBABILITY_ABS_TOLERANCE,
    BEHAVIOR_TEMPERATURE,
    BELIEF_LOSS_WEIGHT,
    CLIP_FRACTION_HARD_LIMIT,
    ENTROPY_COEFFICIENT_END,
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
    LEARNER_CONTROL_BLUE,
    LEARNER_CONTROL_BOTH,
    LEARNER_CONTROL_RED,
    MINIBATCH_SIZE,
    OPTIMIZER_CONSTRAINTS,
    PPO_CLIP_EPSILON,
    VALUE_LOSS_WEIGHT,
    contract_digest as phase9_contract_digest,
)

PHASE14_CONTRACT_VERSION = "phase14_contract_v1"
PHASE14_ROLLOUT_VERSION = "phase14_rollout_v1"
PHASE14_POPULATION_VERSION = "phase14_population_v1"
PHASE14_SCHEDULE_VERSION = "phase14_schedule_v1"
PHASE14_TRAINER_VERSION = "phase14_trainer_v1"
PHASE14_COLLECTOR_VERSION = "phase14_collector_v1"
PHASE14_POOL_VERSION = "phase14_active_pool_v1"

#: The single run namespace. One Phase 14 run exists; a second would be a
#: different experiment and would need its own namespace, not a reused one.
PHASE14_NAMESPACE = "phase14"


class Phase14ContractError(RuntimeError):
    """Raised when a Phase 14 contract value is requested outside its domain."""


# ---------------------------------------------------------------------------
# Where Agent 1's frozen documents live
# ---------------------------------------------------------------------------


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


FROZEN_CONTRACT_RELATIVE_PATH = "reports/phase13/phase13_final_training_contract_v1.json"
SETUP_SOURCE_RELATIVE_PATH = "reports/phase13/phase14_setup_source_v1.json"
SELECTION_PACK_RELATIVE_PATH = "reports/phase13/phase14_checkpoint_selection_pack_v1.json"
SELECTION_RULE_RELATIVE_PATH = "reports/phase13/phase14_checkpoint_selection_rule_v1.json"
SETUP_CENSUS_RELATIVE_PATH = "reports/phase13/phase13_setup_census_v1.json"


# ---------------------------------------------------------------------------
# The starting model
# ---------------------------------------------------------------------------

STARTING_CHECKPOINT = "checkpoints/phase9/selfplay_c1_v1.pt"
STARTING_CHECKPOINT_SHA256 = (
    "dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea"
)
STARTING_MODEL_STATE_DIGEST = (
    "f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd"
)

#: The accepted C1 parameter count and config digest, as bound by the accepted
#: Phase 10B contract. A model that is not these is not the accepted
#: architecture and may not be optimized under this contract.
ACCEPTED_C1_PARAMETERS = 863_959
ACCEPTED_C1_CONFIG_DIGEST = (
    "31ca84ab140c523e65567787b0289fe0dbdf5ab0344667410a5fda7060cfe07d"
)

#: Agent 1C is preserved and is *not* the policy/value start. Named here so the
#: refusal is a constant a test can bind, not a comment.
AGENT1C_CHECKPOINT = "checkpoints/phase11b/agent01_1c_final_block_plus_mlp.pt"
AGENT1C_SHA256 = "a125208605f5e68c897214016e1803718439755e6286e5a185447636ffcd9fad"
AGENT1C_ROLE = "belief/search pipeline only; never the Phase 14 policy/value start"

#: Fresh AdamW moments, the accepted initial beta, new RNG namespace. Phase 14
#: is a new run identity, not a resume of the Phase 9 run.
OPTIMIZER_STATE_AT_START = "fresh AdamW moments (zero); KL controller beta0 0.005"
INITIAL_KL_BETA = 0.005

#: The accepted Phase 9 trainer keeps no EMA. The absence is recorded rather
#: than omitted, so a hot checkpoint answers "no EMA" instead of "no field".
EMA_PRESENT = False
EMA_STATE_RECORD = "absent - the accepted Phase 9 system maintains no EMA"


# ---------------------------------------------------------------------------
# Learning rate
# ---------------------------------------------------------------------------

LR9 = 3e-4
MAIN_LEARNING_RATE = 7.5e-5
LATE_LEARNING_RATE = 3.75e-5
MAIN_LR_MULTIPLIER = 0.25
LATE_LR_MULTIPLIER = 0.125
LEARNING_RATE_SCHEDULE = "constant within each segment; no warmup, no decay"

#: Phase 14 runs at the accepted entropy schedule's terminal value for the whole
#: run: the run is wall-clock terminated, so an iteration-indexed ramp has no
#: defined endpoint.
ENTROPY_COEFFICIENT = 0.001


# ---------------------------------------------------------------------------
# Segments and the wall clock
# ---------------------------------------------------------------------------

SEGMENT_MAIN = "main"
SEGMENT_LATE = "late"
SEGMENTS = (SEGMENT_MAIN, SEGMENT_LATE)

MAIN_SEGMENT_HOURS = 132
LATE_SEGMENT_HOURS = 36
TOTAL_HOURS = 168

TRANSITION_SECONDS = MAIN_SEGMENT_HOURS * 3600  # 475,200
DEADLINE_SECONDS = TOTAL_HOURS * 3600  # 604,800
MAIN_FRACTION = MAIN_SEGMENT_HOURS / TOTAL_HOURS

TRANSITION_RULE = (
    "elapsed = current_utc - run_start_utc (the ORIGINAL start; downtime counts "
    "and never moves the transition). The first collection unit launched with "
    "elapsed >= 132h runs under the late segment; a unit already in flight "
    "finishes under main settings, including its optimizer epochs."
)

DEADLINE_RULE = (
    "run_deadline_utc = run_start_utc + 604800s, computed once and persisted; a "
    "restart never creates a fresh 168-hour duration"
)


# ---------------------------------------------------------------------------
# The population
# ---------------------------------------------------------------------------

GAMES_PER_ITERATION = 2048

BUCKET_CURRENT = "current"
BUCKET_HISTORICAL = "historical"
BUCKET_RULE = "rule"
BUCKET_STRESS = "stress"

#: Ordinal layout order, the Phase 9 `rule_tier_for_ordinal` idiom: contiguous
#: frozen subranges inside an iteration, never a sampled assignment.
POPULATION_BUCKETS = (BUCKET_CURRENT, BUCKET_HISTORICAL, BUCKET_RULE, BUCKET_STRESS)

RULE_FAMILY_ORDER = ("strategic_rule_based", "tactical_rule_based")
STRESS_FAMILY_ORDER = (
    "stress_scout_rush",
    "stress_miner_rush",
    "stress_information_miser",
)
HANDCRAFTED_FAMILY_ORDER = RULE_FAMILY_ORDER + STRESS_FAMILY_ORDER

#: Exactly the frozen 3/3/2/2/2-of-12% realization: 61/61/41/41/41 of 2,048.
HANDCRAFTED_COUNTS = {
    "strategic_rule_based": 61,
    "tactical_rule_based": 61,
    "stress_scout_rush": 41,
    "stress_miner_rush": 41,
    "stress_information_miser": 41,
}
RULE_BUCKET_COUNT = sum(HANDCRAFTED_COUNTS[name] for name in RULE_FAMILY_ORDER)  # 122
STRESS_BUCKET_COUNT = sum(HANDCRAFTED_COUNTS[name] for name in STRESS_FAMILY_ORDER)  # 123
HANDCRAFTED_TOTAL = RULE_BUCKET_COUNT + STRESS_BUCKET_COUNT  # 245

MAIN_CURRENT_GAMES = 1188
MAIN_HISTORICAL_GAMES = 615
LATE_CURRENT_GAMES = 819
LATE_HISTORICAL_GAMES = 984

SEGMENT_BUCKET_COUNTS = {
    SEGMENT_MAIN: {
        BUCKET_CURRENT: MAIN_CURRENT_GAMES,
        BUCKET_HISTORICAL: MAIN_HISTORICAL_GAMES,
        BUCKET_RULE: RULE_BUCKET_COUNT,
        BUCKET_STRESS: STRESS_BUCKET_COUNT,
    },
    SEGMENT_LATE: {
        BUCKET_CURRENT: LATE_CURRENT_GAMES,
        BUCKET_HISTORICAL: LATE_HISTORICAL_GAMES,
        BUCKET_RULE: RULE_BUCKET_COUNT,
        BUCKET_STRESS: STRESS_BUCKET_COUNT,
    },
}

MIXTURE_IS_SCHEDULED_NOT_SAMPLED = (
    "games are scheduled counts, never sampled; no adaptive reweighting from "
    "results is possible because no result reaches the mixer"
)


def require_segment(segment: str) -> str:
    if segment not in SEGMENTS:
        raise Phase14ContractError(
            f"unknown Phase 14 segment {segment!r}; expected one of {list(SEGMENTS)}"
        )
    return segment


def bucket_counts(segment: str) -> dict:
    """The frozen per-bucket game counts of one iteration in one segment."""
    return dict(SEGMENT_BUCKET_COUNTS[require_segment(segment)])


def games_per_iteration(segment: str) -> int:
    return sum(bucket_counts(segment).values())


def learning_rate(segment: str) -> float:
    """The frozen constant learning rate of one segment."""
    return MAIN_LEARNING_RATE if require_segment(segment) == SEGMENT_MAIN else LATE_LEARNING_RATE


def entropy_coefficient() -> float:
    """Constant for the whole run; takes no schedule position by design."""
    return ENTROPY_COEFFICIENT


@dataclass(frozen=True)
class Population:
    """How many games of each kind one iteration holds.

    Production is :data:`PRODUCTION_POPULATION` — the frozen 2,048-game mixture,
    and the only population a production run accepts. The scaled variant exists
    for the same reason the manual clock does: a short integration test cannot
    collect 2,048 games per unit, and the alternative to a declared, refused-in-
    production seam is untested production code. `production=False` travels with
    the object, and :class:`~stratego.training.phase14_runner.Phase14Runner`
    refuses it outside test mode.
    """

    family_counts: dict
    current: dict
    historical: dict
    production: bool = True
    divisor: int = 1

    def bucket_counts(self, segment: str) -> dict:
        require_segment(segment)
        rule = sum(self.family_counts[name] for name in RULE_FAMILY_ORDER)
        stress = sum(self.family_counts[name] for name in STRESS_FAMILY_ORDER)
        return {
            BUCKET_CURRENT: int(self.current[segment]),
            BUCKET_HISTORICAL: int(self.historical[segment]),
            BUCKET_RULE: int(rule),
            BUCKET_STRESS: int(stress),
        }

    def games_per_iteration(self, segment: str) -> int:
        return sum(self.bucket_counts(segment).values())

    def handcrafted_policy_for_ordinal(self, bucket: str, ordinal: int) -> str:
        """The frozen policy id of one rule/stress ordinal: contiguous subranges."""
        if bucket == BUCKET_RULE:
            order = RULE_FAMILY_ORDER
        elif bucket == BUCKET_STRESS:
            order = STRESS_FAMILY_ORDER
        else:
            raise Phase14ContractError(f"bucket {bucket!r} has no handcrafted family layout")
        total = sum(self.family_counts[name] for name in order)
        if not 0 <= ordinal < total:
            raise Phase14ContractError(f"{bucket} ordinal {ordinal} is outside 0..{total - 1}")
        cursor = 0
        for name in order:
            cursor += self.family_counts[name]
            if ordinal < cursor:
                return name
        raise Phase14ContractError("unreachable: handcrafted ordinal beyond every family")

    def to_dict(self) -> dict:
        return {
            "production": bool(self.production),
            "divisor": int(self.divisor),
            "family_counts": dict(self.family_counts),
            "segments": {
                segment: self.bucket_counts(segment) for segment in SEGMENTS
            },
        }

    @staticmethod
    def scaled(divisor: int) -> "Population":
        """A proportionally smaller, test-only population.

        Every family keeps at least one game so the mixer, the colour balance
        and all five handcrafted behaviours are still exercised; the shape is
        the frozen one, the size is not.
        """
        if not isinstance(divisor, int) or isinstance(divisor, bool) or divisor < 1:
            raise Phase14ContractError(f"divisor must be an int >= 1, got {divisor!r}")
        if divisor == 1:
            return PRODUCTION_POPULATION
        return Population(
            family_counts={
                name: max(1, count // divisor) for name, count in HANDCRAFTED_COUNTS.items()
            },
            current={
                SEGMENT_MAIN: max(1, MAIN_CURRENT_GAMES // divisor),
                SEGMENT_LATE: max(1, LATE_CURRENT_GAMES // divisor),
            },
            historical={
                SEGMENT_MAIN: max(1, MAIN_HISTORICAL_GAMES // divisor),
                SEGMENT_LATE: max(1, LATE_HISTORICAL_GAMES // divisor),
            },
            production=False,
            divisor=int(divisor),
        )


PRODUCTION_POPULATION = Population(
    family_counts=dict(HANDCRAFTED_COUNTS),
    current={SEGMENT_MAIN: MAIN_CURRENT_GAMES, SEGMENT_LATE: LATE_CURRENT_GAMES},
    historical={SEGMENT_MAIN: MAIN_HISTORICAL_GAMES, SEGMENT_LATE: LATE_HISTORICAL_GAMES},
    production=True,
    divisor=1,
)


def handcrafted_policy_for_ordinal(bucket: str, ordinal: int) -> str:
    """The frozen policy id of one rule/stress ordinal: contiguous subranges.

    Rule bucket: 0..60 strategic, 61..121 tactical. Stress bucket: 0..40
    scout-rush, 41..81 miner-rush, 82..122 information-miser. Exact counts are
    a property of the ranges, so no draw and no rotation can move them.
    """
    if bucket == BUCKET_RULE:
        order = RULE_FAMILY_ORDER
    elif bucket == BUCKET_STRESS:
        order = STRESS_FAMILY_ORDER
    else:
        raise Phase14ContractError(
            f"bucket {bucket!r} has no handcrafted family layout"
        )
    total = sum(HANDCRAFTED_COUNTS[name] for name in order)
    if not 0 <= ordinal < total:
        raise Phase14ContractError(
            f"{bucket} ordinal {ordinal} is outside 0..{total - 1}"
        )
    cursor = 0
    for name in order:
        cursor += HANDCRAFTED_COUNTS[name]
        if ordinal < cursor:
            return name
    raise Phase14ContractError("unreachable: handcrafted ordinal beyond every family")


def learner_color(bucket: str, iteration: int, ordinal: int) -> "str | None":
    """The learner's colour in one asymmetric game; `None` for self-play.

    The accepted Phase 9 parity rule, unchanged: red when
    ``(ordinal + iteration) % 2 == 0``. Any even-sized ordinal range splits
    exactly in half; an odd range's single remainder alternates with iteration
    parity.
    """
    if bucket not in POPULATION_BUCKETS:
        raise Phase14ContractError(f"unknown bucket: {bucket!r}")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 1:
        raise Phase14ContractError(f"iteration must be an int >= 1, got {iteration!r}")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
        raise Phase14ContractError(f"ordinal must be an int >= 0, got {ordinal!r}")
    if bucket == BUCKET_CURRENT:
        return None
    return "red" if (ordinal + iteration) % 2 == 0 else "blue"


def learner_control_for(bucket: str, iteration: int, ordinal: int) -> str:
    color = learner_color(bucket, iteration, ordinal)
    if color is None:
        return LEARNER_CONTROL_BOTH
    return LEARNER_CONTROL_RED if color == "red" else LEARNER_CONTROL_BLUE


# ---------------------------------------------------------------------------
# The historical archive and the bounded active pool
# ---------------------------------------------------------------------------

ARCHIVE_CADENCE_SECONDS = 2 * 3600
ARCHIVE_SNAPSHOTS_IN_RUN = TOTAL_HOURS // 2  # 84

ANCHOR_P8 = "P8"
ANCHOR_P9 = "P9"
POOL_ANCHORS = (ANCHOR_P8, ANCHOR_P9)
ANCHOR_CHECKPOINTS = {
    ANCHOR_P8: "checkpoints/phase8/warmstart_c1_v1.pt",
    ANCHOR_P9: "checkpoints/phase9/selfplay_c1_v1.pt",
}
ANCHOR_SHA256 = {
    ANCHOR_P8: "f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca",
    ANCHOR_P9: STARTING_CHECKPOINT_SHA256,
}

POOL_SIZE = 16
POOL_SNAPSHOT_SLOTS = POOL_SIZE - len(POOL_ANCHORS)  # 14
POOL_RECENT_SLOTS = 6
POOL_OLDER_SLOTS = 4
POOL_MIDDLE_SLOTS = 4

CATEGORY_ANCHOR = "anchor"
CATEGORY_OLDER = "older"
CATEGORY_MIDDLE = "middle"
CATEGORY_RECENT = "recent"
POOL_CATEGORIES = (CATEGORY_ANCHOR, CATEGORY_OLDER, CATEGORY_MIDDLE, CATEGORY_RECENT)
POOL_SNAPSHOT_CATEGORIES = (CATEGORY_OLDER, CATEGORY_MIDDLE, CATEGORY_RECENT)

POOL_CATEGORY_WEIGHTS = {
    CATEGORY_ANCHOR: 0.20,
    CATEGORY_OLDER: 0.25,
    CATEGORY_MIDDLE: 0.25,
    CATEGORY_RECENT: 0.30,
}

POOL_ADMISSION = "automatic on durable write; no tournament, no result ever read"


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

HOT_CHECKPOINT_SECONDS = 15 * 60
HOT_CHECKPOINT_RETAIN = 4
CANDIDATE_CADENCE_SECONDS = 6 * 3600
CANDIDATE_HOURS = tuple(range(0, TOTAL_HOURS + 1, 6))  # 0, 6, ..., 168
CANDIDATE_COUNT = len(CANDIDATE_HOURS)  # 29

HOT_CHECKPOINT_REQUIRED_FIELDS = (
    "model_state",
    "optimizer_state",
    "ema_state",
    "global_optimizer_step",
    "rng",
    "population_schedule_state",
    "active_historical_pool",
    "historical_archive_state",
    "shard_cursor",
    "storage_state",
    "run_start_utc",
    "run_deadline_utc",
    "schedule_state",
    "candidate_evaluation_state",
)


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------

SELECTION_PACK_DIGEST = (
    "896a753b3d568902e93e803f1a45de9e8834ff1cdf90bc08cfacf90bcf0c2bde"
)
SELECTION_PACK_GAMES = 128
SELECTION_STRATA = (
    "phase9_anchor",
    "strategic_rule_based",
    "tactical_rule_based",
    "stress_scout_rush",
)
SELECTION_GAMES_PER_STRATUM = 32
SELECTION_RULE = (
    "highest equal-weight mean EWR across the four strata; tie-break on highest "
    "minimum stratum EWR; then the later candidate hour"
)
CANDIDATE_EVALUATION_ISOLATION = (
    "monitoring only: an evaluation may never stop training early, change LR, "
    "mixture, setup source, pool logic or cadence, or extend the deadline; a "
    "failed evaluation preserves the candidate and reruns later on the same pack"
)


# ---------------------------------------------------------------------------
# Setup source
# ---------------------------------------------------------------------------

SETUP_SOURCE_IDENTITY = "phase14_setup_source_v1"
SETUP_SELECTOR_CANDIDATE = "P10-D"
SETUP_SPLIT = "train"
SETUP_NEUTRAL_WEIGHT = 0.35
SETUP_LEARNED_WEIGHT = 0.65
SETUP_SELECTOR_CONFIG_SHA256 = (
    "6e227815bc3cb44f19cdeee55d00ec0ae75726fb411ee9131660aa712bb86668"
)
SETUP_ORIENTATION_RULE = (
    "every engine placement goes through SelectorDraw.oriented(player); the "
    "Phase 11B glue (Phase11BSetupSources) returns canonical tuples and must "
    "not be reused"
)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

EXTERNAL_VOLUME = "/Volumes/Brandon_Washington"
EXTERNAL_RUN_DIRECTORY = "/Volumes/Brandon_Washington/stratego_phase14"
HOT_CHECKPOINT_DIRECTORY = "checkpoints/phase14/hot"
DURABLE_ARCHIVE_SUBDIRECTORY = "archive"
ROLLOUT_SUBDIRECTORY = "rollouts"
LOG_SUBDIRECTORY = "logs"
EVALUATION_SUBDIRECTORY = "evaluations"

STORAGE_RESERVE_GIB = 120
FULL_RAW_RETENTION = True
ROLLING_DELETION_TRIGGER_GIB = STORAGE_RESERVE_GIB
ROLLING_DELETION_RULE = (
    "pre-authorized only below 120 GiB free: delete already-consumed disposable "
    "Phase 14 raw shards oldest-first, retaining all checkpoints, metrics, "
    "historical snapshots and a 1-in-16 sample of deleted shard ranges"
)
NO_DELETION_RULE = (
    "earlier accepted project evidence is never deleted under any condition, "
    "including to create storage space"
)


# ---------------------------------------------------------------------------
# Search prohibition
# ---------------------------------------------------------------------------

SEARCH_PROHIBITION = (
    "search may not be used for self-play action selection, training targets, "
    "opponent policies, policy improvement or trajectory generation; Phase 14 "
    "trains the direct C1 policy/value system only"
)

#: Module prefixes the training path may never import. Enforced by a test that
#: walks the real import graph, not by convention.
FORBIDDEN_TRAINING_IMPORTS = ("stratego.search",)


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

FROZEN_METRIC_LIST = (
    "elapsed wall-clock",
    "remaining wall-clock",
    "optimizer step",
    "games generated",
    "positions generated",
    "collection throughput",
    "learner throughput",
    "policy loss",
    "value loss",
    "belief auxiliary loss",
    "gradient norm",
    "learning rate",
    "advantage-filter acceptance fraction",
    "draw rate",
    "game length",
    "current/historical opponent mix",
    "active historical pool",
    "archive size",
    "checkpoint age",
    "disk usage",
    "worker health",
    "non-finite counters",
    "candidate evaluation status",
)

#: Values the run's control surface may never write. The controller refuses
#: them by name rather than by hoping no caller asks.
IMMUTABLE_CONTROL_KEYS = (
    "learning_rate",
    "loss_weights",
    "opponent_mixture",
    "setup_source",
    "historical_pool_algorithm",
    "candidate_selection_rule",
    "deadline",
    "checkpoint_cadence",
)


# ---------------------------------------------------------------------------
# The contract document and its digest
# ---------------------------------------------------------------------------


def inherited_phase9_values() -> dict:
    """Every accepted Phase 9 value Phase 14 consumes without change."""
    return {
        "phase9_contract_digest": phase9_contract_digest(),
        "optimizer": dict(OPTIMIZER_CONSTRAINTS),
        "ppo_clip_epsilon": PPO_CLIP_EPSILON,
        "value_loss_weight": VALUE_LOSS_WEIGHT,
        "belief_loss_weight": BELIEF_LOSS_WEIGHT,
        "behavior_kl_target": BEHAVIOR_KL_TARGET,
        "kl_beta_increase_threshold": KL_BETA_INCREASE_THRESHOLD,
        "kl_beta_decrease_threshold": KL_BETA_DECREASE_THRESHOLD,
        "kl_beta_increase_factor": KL_BETA_INCREASE_FACTOR,
        "kl_beta_decrease_factor": KL_BETA_DECREASE_FACTOR,
        "kl_beta_min": KL_BETA_MIN,
        "kl_beta_max": KL_BETA_MAX,
        "kl_hard_limit": KL_HARD_LIMIT,
        "clip_fraction_hard_limit": CLIP_FRACTION_HARD_LIMIT,
        "gamma": GAMMA,
        "lambda_advantage": LAMBDA_ADVANTAGE,
        "lambda_value": LAMBDA_VALUE,
        "advantage_filter_quantile": ADVANTAGE_FILTER_QUANTILE,
        "advantage_filter_floor": ADVANTAGE_FILTER_FLOOR,
        "advantage_standardization_epsilon": ADVANTAGE_STANDARDIZATION_EPSILON,
        "minibatch_size": MINIBATCH_SIZE,
        "epochs_per_rollout": EPOCHS_PER_ROLLOUT,
        "behavior_temperature": BEHAVIOR_TEMPERATURE,
        "behavior_probability_abs_tolerance": BEHAVIOR_PROBABILITY_ABS_TOLERANCE,
        "entropy_schedule_terminal_value": ENTROPY_COEFFICIENT_END,
    }


def contract_document() -> dict:
    """The whole Phase 14 training contract as one canonical document."""
    return {
        "contract_version": PHASE14_CONTRACT_VERSION,
        "namespace": PHASE14_NAMESPACE,
        "frozen_source": FROZEN_CONTRACT_RELATIVE_PATH,
        "starting_model": {
            "checkpoint": STARTING_CHECKPOINT,
            "file_sha256": STARTING_CHECKPOINT_SHA256,
            "model_state_digest": STARTING_MODEL_STATE_DIGEST,
            "parameters": ACCEPTED_C1_PARAMETERS,
            "optimizer_state": OPTIMIZER_STATE_AT_START,
            "initial_kl_beta": INITIAL_KL_BETA,
            "ema": EMA_STATE_RECORD,
            "agent1c": {"checkpoint": AGENT1C_CHECKPOINT, "role": AGENT1C_ROLE},
        },
        "inherited_phase9": inherited_phase9_values(),
        "learning_rate": {
            "LR9": LR9,
            "main": MAIN_LEARNING_RATE,
            "late": LATE_LEARNING_RATE,
            "main_multiplier": MAIN_LR_MULTIPLIER,
            "late_multiplier": LATE_LR_MULTIPLIER,
            "schedule": LEARNING_RATE_SCHEDULE,
        },
        "entropy_coefficient": ENTROPY_COEFFICIENT,
        "wall_clock": {
            "main_segment_hours": MAIN_SEGMENT_HOURS,
            "late_segment_hours": LATE_SEGMENT_HOURS,
            "total_hours": TOTAL_HOURS,
            "transition_seconds": TRANSITION_SECONDS,
            "deadline_seconds": DEADLINE_SECONDS,
            "transition_rule": TRANSITION_RULE,
            "deadline_rule": DEADLINE_RULE,
        },
        "population": {
            "games_per_iteration": GAMES_PER_ITERATION,
            "buckets": list(POPULATION_BUCKETS),
            "segment_bucket_counts": {
                segment: dict(counts)
                for segment, counts in SEGMENT_BUCKET_COUNTS.items()
            },
            "handcrafted_counts": dict(HANDCRAFTED_COUNTS),
            "ordinal_layout": list(HANDCRAFTED_FAMILY_ORDER),
            "scheduling": MIXTURE_IS_SCHEDULED_NOT_SAMPLED,
        },
        "historical": {
            "pool_version": PHASE14_POOL_VERSION,
            "archive_cadence_seconds": ARCHIVE_CADENCE_SECONDS,
            "archive_snapshots_in_run": ARCHIVE_SNAPSHOTS_IN_RUN,
            "pool_size": POOL_SIZE,
            "anchors": {name: ANCHOR_SHA256[name] for name in POOL_ANCHORS},
            "slots": {
                CATEGORY_OLDER: POOL_OLDER_SLOTS,
                CATEGORY_MIDDLE: POOL_MIDDLE_SLOTS,
                CATEGORY_RECENT: POOL_RECENT_SLOTS,
            },
            "weights": dict(POOL_CATEGORY_WEIGHTS),
            "admission": POOL_ADMISSION,
        },
        "checkpoints": {
            "hot_seconds": HOT_CHECKPOINT_SECONDS,
            "hot_retain": HOT_CHECKPOINT_RETAIN,
            "archive_seconds": ARCHIVE_CADENCE_SECONDS,
            "candidate_seconds": CANDIDATE_CADENCE_SECONDS,
            "candidate_hours": list(CANDIDATE_HOURS),
            "candidate_count": CANDIDATE_COUNT,
            "hot_required_fields": list(HOT_CHECKPOINT_REQUIRED_FIELDS),
        },
        "candidate_evaluation": {
            "pack_digest": SELECTION_PACK_DIGEST,
            "games": SELECTION_PACK_GAMES,
            "strata": list(SELECTION_STRATA),
            "games_per_stratum": SELECTION_GAMES_PER_STRATUM,
            "selection_rule": SELECTION_RULE,
            "isolation": CANDIDATE_EVALUATION_ISOLATION,
        },
        "setup_source": {
            "identity": SETUP_SOURCE_IDENTITY,
            "candidate": SETUP_SELECTOR_CANDIDATE,
            "split": SETUP_SPLIT,
            "neutral_weight": SETUP_NEUTRAL_WEIGHT,
            "learned_weight": SETUP_LEARNED_WEIGHT,
            "selector_config_sha256": SETUP_SELECTOR_CONFIG_SHA256,
            "orientation": SETUP_ORIENTATION_RULE,
        },
        "storage": {
            "external_volume": EXTERNAL_VOLUME,
            "external_run_directory": EXTERNAL_RUN_DIRECTORY,
            "hot_directory": HOT_CHECKPOINT_DIRECTORY,
            "full_raw_retention": FULL_RAW_RETENTION,
            "reserve_gib": STORAGE_RESERVE_GIB,
            "rolling_deletion": ROLLING_DELETION_RULE,
            "no_deletion_rule": NO_DELETION_RULE,
        },
        "search": SEARCH_PROHIBITION,
        "monitoring": {
            "metrics": list(FROZEN_METRIC_LIST),
            "immutable_control_keys": list(IMMUTABLE_CONTROL_KEYS),
        },
    }


def canonical_json(document) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


@lru_cache(maxsize=1)
def contract_digest() -> str:
    """The identity of the Phase 14 training contract as implemented."""
    return hashlib.sha256(canonical_json(contract_document()).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Verification against Agent 1's frozen document
# ---------------------------------------------------------------------------


def frozen_contract_path() -> Path:
    return repository_root() / FROZEN_CONTRACT_RELATIVE_PATH


def frozen_contract_document() -> dict:
    """Agent 1's frozen contract, read from disk."""
    path = frozen_contract_path()
    if not path.exists():
        raise Phase14ContractError(
            f"the frozen Agent 1 contract is missing at {path}; Phase 14 may not "
            "be configured from anything else"
        )
    return json.loads(path.read_text())


def file_sha256(path: "str | Path", *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_against_frozen_contract(document: "dict | None" = None) -> list:
    """Every disagreement between this module and Agent 1's frozen document.

    An empty list is the only acceptable result before Phase 14 starts. The
    comparison is value by value against the document's own spelling, so a
    typo here or a later edit there both surface as findings rather than as a
    run that quietly trains something else.
    """
    frozen = frozen_contract_document() if document is None else document
    problems: list = []

    def check(label: str, observed, expected) -> None:
        if observed != expected:
            problems.append(f"{label}: implemented {observed!r} != frozen {expected!r}")

    if frozen.get("status") != "FROZEN":
        problems.append(f"the Agent 1 contract is {frozen.get('status')!r}, not FROZEN")

    starting = frozen["starting_model"]
    check("starting_checkpoint", STARTING_CHECKPOINT, starting["checkpoint"])
    check("starting_sha256", STARTING_CHECKPOINT_SHA256, starting["file_sha256"])
    check(
        "starting_model_state_digest",
        STARTING_MODEL_STATE_DIGEST,
        starting["model_state_digest"],
    )

    retrieved = frozen["phase9_retrieved_values"]
    check("LR9", LR9, retrieved["accepted_phase9_learning_rate"])
    check(
        "belief_loss_weight",
        BELIEF_LOSS_WEIGHT,
        retrieved["belief_auxiliary_objective"]["weight"],
    )
    check(
        "belief_present",
        True,
        retrieved["belief_auxiliary_objective"]["present_in_accepted_phase9"],
    )
    check("ema_present", EMA_PRESENT, retrieved["ema_behavior"]["present_in_accepted_phase9"])
    check(
        "value_loss_weight",
        VALUE_LOSS_WEIGHT,
        0.5 if "weight 0.5" in retrieved["value_objective"] else None,
    )
    check(
        "ppo_clip_epsilon",
        PPO_CLIP_EPSILON,
        retrieved["ratio_clipping"]["ppo_clip_epsilon"],
    )
    check(
        "clip_fraction_hard_limit",
        CLIP_FRACTION_HARD_LIMIT,
        retrieved["ratio_clipping"]["clip_fraction_hard_limit"],
    )
    check(
        "initial_kl_beta",
        INITIAL_KL_BETA,
        retrieved["behavior_policy_kl_regularization"]["initial_beta"],
    )
    check(
        "behavior_kl_target",
        BEHAVIOR_KL_TARGET,
        retrieved["behavior_policy_kl_regularization"]["target"],
    )
    batch = retrieved["batch_update_settings"]
    check("minibatch_size", MINIBATCH_SIZE, batch["minibatch_size"])
    check("epochs_per_rollout", EPOCHS_PER_ROLLOUT, batch["epochs_per_rollout"])
    check("games_per_iteration", GAMES_PER_ITERATION, batch["games_per_iteration"])
    check(
        "entropy_coefficient",
        f"constant {ENTROPY_COEFFICIENT}",
        " ".join(retrieved["entropy_schedule"]["phase14_resolution"].split()[:2]),
    )

    rates = frozen["continuation_learning_rate"]
    check("main_lr", MAIN_LEARNING_RATE, rates["main_continuation_LR"])
    check("late_lr", LATE_LEARNING_RATE, rates["late_continuation_LR"])
    check("main_multiplier", MAIN_LR_MULTIPLIER, rates["multipliers"]["main"])
    check("late_multiplier", LATE_LR_MULTIPLIER, rates["multipliers"]["late"])

    schedule = frozen["main_late_schedule"]
    check("main_segment_hours", MAIN_SEGMENT_HOURS, schedule["main_segment_hours"])
    check("late_segment_hours", LATE_SEGMENT_HOURS, schedule["late_segment_hours"])
    check("total_hours", TOTAL_HOURS, schedule["total_hours"])
    check(
        "transition_elapsed_hours",
        TRANSITION_SECONDS / 3600,
        schedule["transition_elapsed_hours"],
    )

    mixture = frozen["opponent_mixture"]
    main = mixture["main_segment_counts_per_2048"]
    late = mixture["late_segment_counts_per_2048"]
    check("main_current", MAIN_CURRENT_GAMES, main["current_neural"])
    check("main_historical", MAIN_HISTORICAL_GAMES, main["historical_neural"])
    check("late_current", LATE_CURRENT_GAMES, late["current_neural"])
    check("late_historical", LATE_HISTORICAL_GAMES, late["historical_neural"])
    for family, count in HANDCRAFTED_COUNTS.items():
        check(f"handcrafted[{family}] main", count, main["handcrafted"][family])
        check(f"handcrafted[{family}] late", count, late["handcrafted"][family])

    pool_block = frozen["historical_archive_and_active_pool"]
    pool = pool_block[f"active_pool_algorithm {PHASE14_POOL_VERSION}"]
    weights = pool["sampling_weights_within_historical_share"]
    check("pool_weight_anchor", POOL_CATEGORY_WEIGHTS[CATEGORY_ANCHOR], weights["permanent_anchors"])
    check("pool_weight_older", POOL_CATEGORY_WEIGHTS[CATEGORY_OLDER], weights["older"])
    check("pool_weight_middle", POOL_CATEGORY_WEIGHTS[CATEGORY_MIDDLE], weights["middle"])
    check("pool_weight_recent", POOL_CATEGORY_WEIGHTS[CATEGORY_RECENT], weights["recent"])
    for name in POOL_ANCHORS:
        if ANCHOR_SHA256[name] not in pool["permanent_anchors"][name]:
            problems.append(
                f"anchor {name} sha256 {ANCHOR_SHA256[name]} is not the frozen "
                f"{pool['permanent_anchors'][name]!r}"
            )

    hierarchy = frozen["checkpoint_hierarchy"]
    # The cadence and retention values live inside prose sentences in the
    # frozen document. Substring containment is the honest comparison: it
    # cannot be fooled by a different number and does not pretend the prose
    # has a grammar it does not have.
    for label, expected_text, frozen_text in (
        ("hot_cadence", "every 15 minutes", hierarchy["hot_resume"]["cadence"]),
        (
            "hot_retain",
            f"most recent {HOT_CHECKPOINT_RETAIN} valid hot checkpoints",
            hierarchy["hot_resume"]["retention"],
        ),
        (
            "archive_cadence",
            "every 2 hours",
            hierarchy["durable_archive"]["cadence"],
        ),
        (
            "candidate_cadence",
            f"every 6 hours: hours {', '.join(str(hour) for hour in CANDIDATE_HOURS[:3])}",
            hierarchy["final_policy_candidates"]["cadence"],
        ),
        (
            "candidate_count",
            f"({CANDIDATE_COUNT} candidates)",
            hierarchy["final_policy_candidates"]["cadence"],
        ),
    ):
        if expected_text not in frozen_text:
            problems.append(
                f"{label}: {expected_text!r} is not stated by the frozen "
                f"{frozen_text!r}"
            )

    evaluation = frozen["candidate_evaluation"]
    if SELECTION_PACK_DIGEST not in evaluation["pack"]:
        problems.append(
            f"selection pack digest {SELECTION_PACK_DIGEST} is not named by the "
            "frozen candidate_evaluation block"
        )

    deadline = frozen["wall_clock_contract"]
    if str(DEADLINE_SECONDS) not in deadline["run_deadline_utc"]:
        problems.append(
            f"deadline seconds {DEADLINE_SECONDS} is not the frozen "
            f"{deadline['run_deadline_utc']!r}"
        )

    setup = frozen["setup_source"]
    check("setup_source_identity", SETUP_SOURCE_IDENTITY, setup["identity"].split()[0])

    storage = frozen["storage_retention"]
    if EXTERNAL_VOLUME not in storage["volume"]:
        problems.append(f"external volume {EXTERNAL_VOLUME} is not {storage['volume']!r}")
    if str(STORAGE_RESERVE_GIB) not in storage["contingency_rolling_policy"]:
        problems.append(
            f"reserve {STORAGE_RESERVE_GIB} GiB is not the frozen contingency trigger"
        )

    for name in ("phase12_tiny", "phase12_small", "phase12_medium"):
        if frozen["search_prohibition"][name] != "NOT USED":
            problems.append(f"search_prohibition[{name}] is not 'NOT USED'")

    return problems


def assert_matches_frozen_contract() -> dict:
    """Refuse to proceed unless the implementation is the frozen contract."""
    problems = verify_against_frozen_contract()
    if problems:
        raise Phase14ContractError(
            "the Phase 14 implementation disagrees with the frozen Agent 1 "
            f"contract: {problems}"
        )
    return {
        "frozen_contract": FROZEN_CONTRACT_RELATIVE_PATH,
        "frozen_contract_sha256": file_sha256(frozen_contract_path()),
        "implementation_contract_digest": contract_digest(),
        "disagreements": 0,
    }
