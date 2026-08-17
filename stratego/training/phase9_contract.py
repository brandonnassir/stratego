"""Phase 9 Agent 1: the frozen population self-play RL contract.

Specification sources:

- `01_AGENT_1_RL_CONTRACT_AND_EVAL_BANKS.md` (everything this module freezes)
- `00_PHASE_9_SEQUENCE_AND_COMMON_CONTRACT.md` (frozen Phase 8 inputs,
  mission boundaries, mixture, league, learner control, behavior policy,
  targets, PPO, rollout state machine, checkpoint contents, banks, score,
  pilot matrix, canonical run, final gates)

What "frozen" means here
------------------------
Every learning-design decision of Phase 9 is stated in this module **before**
any Phase 9 optimizer step runs and before any trainable Phase 9 rollout is
generated: the population mixture and its exact per-bucket counts, the
opponent maps, the colour-balance rule, learner-control semantics, the
behavior-policy softmax and its storage/verification representation, the
same-player temporal targets, advantage filtering, the PPO objective and its
damping, the full loss, the rollout store and its lifecycle, the checkpoint
contract, the evaluation banks and their score, the pilot matrix, and the
final acceptance gates. Agents 2-8 read these values; they do not choose
them. A different value is a reviewed new version of this contract, never an
in-place edit.

Nothing in this module collects rollouts, builds training batches, or
touches model weights. The only executable behaviour is verification (does
the live repository still match the frozen expectation?), schedule
arithmetic (pure functions of frozen constants), and hand-computable target
mathematics used by the regression tests.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from ..engine.constants import BLUE, PLAYER_NAMES, RED
from ..evaluation.match_spec import PAIRING_COLOR_SWAP_SAME_BOARD
from ..evaluation.registry import LADDER_POLICY_IDS, POLICY_INDEX, STRESS_POLICY_IDS
from .phase9_seed import (
    BUCKET_CURRENT,
    BUCKET_HISTORICAL,
    BUCKET_RULE,
    BUCKET_STRESS,
    CANONICAL_NAMESPACE,
    CANONICAL_PHASE9_SEEDS,
    PHASE9_ROLLOUT_VERSION,
    PILOT_NAMESPACES,
    POPULATION_BUCKETS,
    TEST_BOOTSTRAP_SEED,
    VALIDATION_BOOTSTRAP_SEED,
    historical_opponent_seed,
    parse_phase9_game_id,
    phase9_game_id,
)
from .warmstart_contract import (
    EXPECTED_C1_CONFIG_DIGEST,
    EXPECTED_C1_PARAMETERS,
    EXPECTED_SETUP_PROFILE,
    METRIC_LOG_EPSILON,
    verify_frozen_upstream,
    verify_teacher_roster,
)
from .warmstart_seed import CANONICAL_C1_INIT_SEED

# ---------------------------------------------------------------------------
# Contract identities
# ---------------------------------------------------------------------------

PHASE9_RL_CONTRACT_VERSION = "phase9_rl_contract_v1"
PHASE9_POPULATION_VERSION = "phase9_population_v1"
PHASE9_ROLLOUT_SCHEDULE_VERSION = "phase9_rollout_schedule_v1"
PHASE9_ROLLOUT_STORE_VERSION = "phase9_rollout_store_v1"
PHASE9_ADVANTAGE_VERSION = "phase9_advantage_v1"
PHASE9_TRAIN_ORDER_VERSION = "phase9_train_order_v1"
PHASE9_CHECKPOINT_VERSION = "phase9_checkpoint_v1"
PHASE9_EVAL_BANK_VERSION = "phase9_eval_bank_v1"
PHASE9_ACCEPTANCE_VERSION = "phase9_acceptance_v1"

CONTRACT_IDENTITIES = (
    PHASE9_RL_CONTRACT_VERSION,
    PHASE9_POPULATION_VERSION,
    PHASE9_ROLLOUT_SCHEDULE_VERSION,
    PHASE9_ROLLOUT_STORE_VERSION,
    PHASE9_ADVANTAGE_VERSION,
    PHASE9_TRAIN_ORDER_VERSION,
    PHASE9_CHECKPOINT_VERSION,
    PHASE9_EVAL_BANK_VERSION,
    PHASE9_ACCEPTANCE_VERSION,
)

#: Frozen Phase 8 inputs Phase 9 starts from and must never mutate.
EXPECTED_PHASE8_CHECKPOINT_PATH = "checkpoints/phase8/warmstart_c1_v1.pt"
EXPECTED_PHASE8_CHECKPOINT_SHA256 = (
    "f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca"
)
EXPECTED_PHASE8_SELECTED_UPDATE = 24000
EXPECTED_PHASE8_INIT_PATH = "checkpoints/phase8/warmstart_c1_v1_initialisation.pt"
EXPECTED_PHASE8_INIT_SHA256 = (
    "01c907eeef86ec04121db55ccffb9365e8df27fdf05921b921947d4af365754c"
)
EXPECTED_PHASE8_INIT_STATE_CHECKSUM = (
    "cfe60bb0cb342b03e2506259b5c4d39d321148f7bc8c030bf722e5a234e042b8"
)
EXPECTED_TRAIN_CONFIG_DOCUMENT_DIGEST = (
    "3cab772bd8f74677efcdc1f90ec6f383490313f7652d82bd7fedf86153919ae7"
)
EXPECTED_TRAINER_RUNTIME_IDENTITY_DIGEST = (
    "64db92539a7d6c06ac4d01e4e904857da5b95c3d86d1287e108ede19e4f03879"
)
EXPECTED_CORPUS_VERSION = "synthetic_warmstart_corpus_v1"
EXPECTED_CORPUS_CONTENT_DIGEST = (
    "c95c3545b07f2341e7efbc83c79e6342510dd973038b0f72e7eae013cff87d0d"
)
EXPECTED_CORPUS_METADATA_DIGEST = (
    "1db0f02fe45b16f539f070b1e12d4fdd6f390fd0487180fe660af0f4d49c81bb"
)
EXPECTED_CORPUS_COMMIT_INDEX_DIGEST = (
    "32e8e18d1ca57ee555ed848851284f5938d4989ceb6c864f83ca4b9286c15db1"
)


class Phase9ContractError(RuntimeError):
    """Raised when the frozen Phase 9 contract is violated or unusable."""


# ---------------------------------------------------------------------------
# Population mixture — exact scheduled counts, never sampled
# ---------------------------------------------------------------------------

#: Canonical mixture proportions of one 2,048-game iteration.
POPULATION_PROPORTIONS = {
    BUCKET_CURRENT: 0.50,
    BUCKET_HISTORICAL: 0.25,
    BUCKET_RULE: 0.15,
    BUCKET_STRESS: 0.10,
}

#: Exact per-bucket game counts of one canonical iteration.
CANONICAL_GAMES_PER_ITERATION = 2048
CANONICAL_BUCKET_COUNTS = {
    BUCKET_CURRENT: 1024,
    BUCKET_HISTORICAL: 512,
    BUCKET_RULE: 307,
    BUCKET_STRESS: 205,
}

#: Exact rule-tier subdivision of the canonical rule bucket.
CANONICAL_RULE_TIER_COUNTS = {
    "strategic_rule_based": 154,
    "tactical_rule_based": 107,
    "basic_heuristic": 46,
}

#: Exact per-bucket game counts of one pilot iteration.
PILOT_GAMES_PER_ITERATION = 1024
PILOT_BUCKET_COUNTS = {
    BUCKET_CURRENT: 512,
    BUCKET_HISTORICAL: 256,
    BUCKET_RULE: 154,
    BUCKET_STRESS: 102,
}

#: Exact rule-tier subdivision of the pilot rule bucket.
PILOT_RULE_TIER_COUNTS = {
    "strategic_rule_based": 77,
    "tactical_rule_based": 54,
    "basic_heuristic": 23,
}

#: Rule-tier order inside the rule bucket's ordinal space: contiguous
#: subranges, strongest tier first. Exact counts by construction.
RULE_TIER_ORDER = ("strategic_rule_based", "tactical_rule_based", "basic_heuristic")

#: The frozen Phase 4 stress roster, in registry order. The stress bucket's
#: ordinal->policy map rotates through this tuple.
STRESS_POLICY_ROSTER = (
    "stress_scout_rush",
    "stress_miner_rush",
    "stress_draw_seeker",
    "stress_berserker",
    "stress_information_miser",
    "stress_chaos",
)


def bucket_counts(namespace: str) -> dict:
    """The frozen per-bucket counts of one iteration in one namespace."""
    if namespace == CANONICAL_NAMESPACE:
        return dict(CANONICAL_BUCKET_COUNTS)
    if namespace in PILOT_NAMESPACES:
        return dict(PILOT_BUCKET_COUNTS)
    raise Phase9ContractError(f"unknown Phase 9 namespace: {namespace!r}")


def rule_tier_counts(namespace: str) -> dict:
    """The frozen rule-tier subdivision of one iteration in one namespace."""
    if namespace == CANONICAL_NAMESPACE:
        return dict(CANONICAL_RULE_TIER_COUNTS)
    if namespace in PILOT_NAMESPACES:
        return dict(PILOT_RULE_TIER_COUNTS)
    raise Phase9ContractError(f"unknown Phase 9 namespace: {namespace!r}")


def games_per_iteration(namespace: str) -> int:
    return sum(bucket_counts(namespace).values())


def rule_tier_for_ordinal(namespace: str, ordinal: int) -> str:
    """The rule tier of one rule-bucket ordinal: contiguous frozen subranges.

    Canonical: 0..153 strategic, 154..260 tactical, 261..306 basic.
    Pilot: 0..76 strategic, 77..130 tactical, 131..153 basic.
    Exact counts are a property of the ranges, not of any draw.
    """
    counts = rule_tier_counts(namespace)
    total = sum(counts.values())
    if not 0 <= ordinal < total:
        raise Phase9ContractError(
            f"rule ordinal {ordinal} is outside 0..{total - 1} for {namespace!r}"
        )
    cursor = 0
    for tier in RULE_TIER_ORDER:
        cursor += counts[tier]
        if ordinal < cursor:
            return tier
    raise Phase9ContractError("unreachable: rule ordinal beyond every tier")


def stress_policy_for_ordinal(iteration: int, ordinal: int, *, namespace: str) -> str:
    """The stress policy of one stress-bucket ordinal.

    `(ordinal + iteration) % 6` walks the frozen roster: exact per-iteration
    counts (one policy receives the odd remainder), and the remainder rotates
    across iterations so no stress policy is systematically favoured.
    """
    total = bucket_counts(namespace)[BUCKET_STRESS]
    if not 0 <= ordinal < total:
        raise Phase9ContractError(
            f"stress ordinal {ordinal} is outside 0..{total - 1} for {namespace!r}"
        )
    if iteration < 1:
        raise Phase9ContractError(f"iteration must be >= 1, got {iteration}")
    return STRESS_POLICY_ROSTER[(ordinal + iteration) % len(STRESS_POLICY_ROSTER)]


# ---------------------------------------------------------------------------
# Learner control and colour balance
# ---------------------------------------------------------------------------

LEARNER_CONTROL_RED = "red"
LEARNER_CONTROL_BLUE = "blue"
LEARNER_CONTROL_BOTH = "both"
LEARNER_CONTROLS = (LEARNER_CONTROL_RED, LEARNER_CONTROL_BLUE, LEARNER_CONTROL_BOTH)

#: Which sides of each bucket are Phase 9 policy/value/belief trainable.
TRAINING_ELIGIBILITY = {
    BUCKET_CURRENT: "both colors",
    BUCKET_HISTORICAL: "current-policy side only",
    BUCKET_RULE: "current-policy side only",
    BUCKET_STRESS: "current-policy side only",
}


def learner_color(bucket: str, iteration: int, ordinal: int) -> "int | None":
    """The learner's colour in one asymmetric game; `None` for self-play.

    Frozen colour-balance rule: the learner is red when
    ``(ordinal + iteration) % 2 == 0``, blue otherwise. Within any
    even-sized ordinal range the colours split exactly in half; an
    odd-sized range's one-game remainder alternates sides with iteration
    parity, exactly as the common contract requires.
    """
    if bucket not in POPULATION_BUCKETS:
        raise Phase9ContractError(f"unknown bucket: {bucket!r}")
    if iteration < 1:
        raise Phase9ContractError(f"iteration must be >= 1, got {iteration}")
    if ordinal < 0:
        raise Phase9ContractError(f"ordinal must be >= 0, got {ordinal}")
    if bucket == BUCKET_CURRENT:
        return None
    return RED if (ordinal + iteration) % 2 == 0 else BLUE


def learner_control_for(bucket: str, iteration: int, ordinal: int) -> str:
    """The frozen `learner_control` label of one scheduled game."""
    color = learner_color(bucket, iteration, ordinal)
    if color is None:
        return LEARNER_CONTROL_BOTH
    return LEARNER_CONTROL_RED if color == RED else LEARNER_CONTROL_BLUE


# ---------------------------------------------------------------------------
# Historical league
# ---------------------------------------------------------------------------

#: The Phase 8 accepted checkpoint is archive member zero and never leaves
#: the active window.
HISTORICAL_ANCHOR_ID = "H000"
ARCHIVE_CADENCE_ITERATIONS = 5
ACTIVE_WINDOW_RECENT_SNAPSHOTS = 8


def archive_snapshot_id(iteration: int) -> str:
    """The immutable archive identity created after `iteration` commits."""
    if iteration < 1:
        raise Phase9ContractError(f"iteration must be >= 1, got {iteration}")
    if iteration % ARCHIVE_CADENCE_ITERATIONS != 0:
        raise Phase9ContractError(
            f"iteration {iteration} is off the frozen archive cadence of "
            f"{ARCHIVE_CADENCE_ITERATIONS}"
        )
    return f"H{iteration:03d}"


def archived_iterations_before(iteration: int) -> tuple:
    """Archive-creating iterations strictly before `iteration`, ascending."""
    if iteration < 1:
        raise Phase9ContractError(f"iteration must be >= 1, got {iteration}")
    return tuple(
        past
        for past in range(ARCHIVE_CADENCE_ITERATIONS, iteration, ARCHIVE_CADENCE_ITERATIONS)
    )


def active_historical_window(iteration: int) -> tuple:
    """The frozen active sampling window of one iteration, oldest first.

    Phase 8 anchor `H000` plus the (up to) 8 most recent archive snapshots
    created by iterations strictly before `iteration`. Older snapshots
    remain stored but inactive. Iteration 1's window is exactly `(H000,)`.
    """
    recent = archived_iterations_before(iteration)[-ACTIVE_WINDOW_RECENT_SNAPSHOTS:]
    return (HISTORICAL_ANCHOR_ID,) + tuple(
        archive_snapshot_id(past) for past in recent
    )


def historical_opponent_for(game_id: str) -> str:
    """The frozen uniform active-window draw of one historical-bucket game.

    `historical_opponent_seed(game_id) % len(window)` indexes the window
    (oldest first). A pure function of the game identity and the frozen
    opponent-schedule seed: no cursor, no arrival order.
    """
    fields = parse_phase9_game_id(game_id)
    window = active_historical_window(fields["iteration"])
    return window[historical_opponent_seed(game_id) % len(window)]


# ---------------------------------------------------------------------------
# The rollout schedule — one fully determined logical game
# ---------------------------------------------------------------------------


def scheduled_game(namespace: str, iteration: int, bucket: str, ordinal: int) -> dict:
    """Everything `phase9_rollout_schedule_v1` fixes about one logical game.

    Pure arithmetic over frozen constants plus the domain-separated
    opponent draw for the historical bucket. Setups are not resolved here —
    the collector resolves them through the frozen `setup_source_v1` train
    path with `setup_root_seed(game_id)` — but the identity that determines
    them is.
    """
    counts = bucket_counts(namespace)
    if bucket not in counts:
        raise Phase9ContractError(f"unknown bucket: {bucket!r}")
    if not 0 <= ordinal < counts[bucket]:
        raise Phase9ContractError(
            f"ordinal {ordinal} is outside 0..{counts[bucket] - 1} for bucket "
            f"{bucket!r} in {namespace!r}"
        )
    game_id = phase9_game_id(namespace, iteration, bucket, ordinal)
    control = learner_control_for(bucket, iteration, ordinal)
    color = learner_color(bucket, iteration, ordinal)

    if bucket == BUCKET_CURRENT:
        opponent = {"kind": "current_policy", "identity": "behavior_snapshot"}
    elif bucket == BUCKET_HISTORICAL:
        opponent = {
            "kind": "historical_snapshot",
            "identity": historical_opponent_for(game_id),
        }
    elif bucket == BUCKET_RULE:
        tier = rule_tier_for_ordinal(namespace, ordinal)
        opponent = {
            "kind": "rule_policy",
            "identity": f"{tier}@{POLICY_INDEX[tier].policy_version}",
        }
    else:
        policy_id = stress_policy_for_ordinal(iteration, ordinal, namespace=namespace)
        opponent = {
            "kind": "stress_policy",
            "identity": f"{policy_id}@{POLICY_INDEX[policy_id].policy_version}",
        }

    return {
        "game_id": game_id,
        "namespace": namespace,
        "iteration": iteration,
        "bucket": bucket,
        "ordinal": ordinal,
        "learner_control": control,
        "learner_color": None if color is None else PLAYER_NAMES[color],
        "opponent": opponent,
    }


def iter_scheduled_games(namespace: str, iteration: int):
    """Every logical game of one iteration, bucket-major then ordinal order.

    The order is a *schedule* order only: collection may proceed in any
    order because every game is a pure function of its own identity.
    """
    for bucket in POPULATION_BUCKETS:
        for ordinal in range(bucket_counts(namespace)[bucket]):
            yield scheduled_game(namespace, iteration, bucket, ordinal)


# ---------------------------------------------------------------------------
# Behavior policy and its storage representation
# ---------------------------------------------------------------------------

BEHAVIOR_TEMPERATURE = 1.0

#: Per-entry absolute tolerance when a stored behavior distribution is
#: re-derived under the exact behavior snapshot. Float32 storage rounding is
#: <= 6e-8 relative and the CPU-float32-reference vs MPS forward gap is
#: measured in 1e-6..1e-5 on logits, so 1e-4 passes every honest
#: reconstruction while a wrong checkpoint (probability shifts of order
#: 1e-1) fails by three orders of magnitude.
BEHAVIOR_PROBABILITY_ABS_TOLERANCE = 1e-4

#: Probability floor inside every log used by PPO ratios and KL terms.
BEHAVIOR_LOG_EPSILON = METRIC_LOG_EPSILON


def behavior_policy_semantics() -> dict:
    """The frozen behavior-policy definition and its storage decision.

    The storage decision reuses `trajectory_v1` faithfully — no field
    changes its meaning:

    - `DecisionRecord.old_probabilities` was defined in Phase 3 as one
      probability per legal action from the collecting policy; a Phase 9
      neural decision stores exactly that: the full legal-softmax behavior
      distribution, rounded to float32 at the point of storage.
    - `DecisionRecord.collection_policy_version` was documented as the slot
      a real checkpoint identifier drops into; a Phase 9 decision stores
      the acting side's policy token (behavior snapshot token, historical
      archive token, or rule/stress `id@version`).
    - `GameRecord.collection_checkpoint_id` stores the behavior snapshot's
      SHA-256, so the learner side of every game is anchored to the exact
      immutable checkpoint that produced it.

    Everything per-game that `trajectory_v1` has no field for (bucket,
    learner control, opponent identity, per-side snapshot digests) lives in
    the `phase9_rollout_store_v1` metadata sidecar, exactly as Phase 7/8
    provenance did.
    """
    return {
        "temperature": BEHAVIOR_TEMPERATURE,
        "definition": (
            "pi_b(a|s) = exp(z_a) / sum_{a' in A(s)} exp(z_{a'}) over exactly "
            "the legal actions A(s), logits z from the frozen behavior "
            "snapshot, temperature 1.0; evaluation banks use greedy argmax "
            "instead"
        ),
        "action_selection": (
            "walk legal actions in ascending action-id order accumulating "
            "behavior probabilities; select the first action whose cumulative "
            "probability >= behavior_sample_uniform(game_id, ply); float32 "
            "tail shortfall selects the last legal action"
        ),
        "applies_to": (
            "every neural participant of a rollout game samples its own "
            "legal softmax at temperature 1.0: the current policy and any "
            "historical snapshot alike"
        ),
        "storage": {
            "trajectory_version": "trajectory_v1 (reused, meaning unchanged)",
            "stored_quantity": (
                "the full legal-action behavior distribution, one entry per "
                "legal action in ascending legal order "
                "(DecisionRecord.old_probabilities)"
            ),
            "dtype": "float32 (rounded at storage time by the frozen builder)",
            "normalization": (
                "distribution sums to 1 within the frozen trajectory_v1 "
                "tolerance of 1e-4"
            ),
            "per_decision_identity": (
                "DecisionRecord.collection_policy_version = the acting side's "
                "policy token"
            ),
            "per_game_identity": (
                "GameRecord.collection_checkpoint_id = the behavior "
                "snapshot's SHA-256; per-side identities and digests repeat "
                "in the phase9_rollout_store_v1 sidecar"
            ),
            "opponent_rule_policy_representation": (
                "a rule/stress opponent decision stores the one-hot "
                "distribution on its realized action with its policy token, "
                "and the neutral (1/3, 1/3, 1/3) value prediction — the "
                "accepted Phase 8 corpus representation; such decisions are "
                "never PPO-trainable (learner-control semantics), so no "
                "behavior probability is claimed for them"
            ),
            "value_prediction": (
                "a neural decision stores the acting network's own W/D/L "
                "softmax from the acting player's perspective "
                "(DecisionRecord.win_draw_loss_prediction, float32); the "
                "same-player targets consume only learner-controlled "
                "decisions' stored predictions"
            ),
        },
        "training_time_probability": (
            "pi_b(a_t|s_t) in the PPO ratio is the STORED float32 probability "
            "of the realized action — the storage is the authority, so the "
            "ratio is exact and independent of any device recomputation; "
            "log pi_b = ln(max(p_stored, 1e-12))"
        ),
        "verification": {
            "rule": (
                "recompute logits under the exact frozen behavior snapshot, "
                "legal softmax at temperature 1.0, and require "
                "max |p_stored - p_recomputed| <= 1e-4 per legal entry; the "
                "realized action's stored probability must match within the "
                "same tolerance"
            ),
            "reference_device": (
                "CPU float32 is the bit-stable reference verifier; an MPS "
                "recomputation uses the same tolerance"
            ),
            "max_abs_mismatch": BEHAVIOR_PROBABILITY_ABS_TOLERANCE,
            "identity_check": (
                "the checkpoint whose SHA-256 matches the recorded behavior "
                "snapshot digest is the only admissible reconstruction "
                "checkpoint; a digest mismatch is a hard veto, not a "
                "tolerance question"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Same-player temporal targets (`phase9_advantage_v1`)
# ---------------------------------------------------------------------------

GAMMA = 1.0
LAMBDA_ADVANTAGE = 0.5
LAMBDA_VALUE = 0.8

#: Advantage filtering: per sealed iteration, tau = max(Q75(|A|), 0.01).
ADVANTAGE_FILTER_QUANTILE = 0.75
ADVANTAGE_FILTER_FLOOR = 0.01
ADVANTAGE_STANDARDIZATION_EPSILON = 1e-8


def behavior_value_scalar(wdl: "tuple[float, float, float]") -> float:
    """`v_t = P(W) - P(L)` from one stored W/D/L prediction."""
    return float(wdl[0]) - float(wdl[2])


def terminal_z(outcome: str) -> int:
    """`z in {-1, 0, +1}` from the acting player's final perspective."""
    if outcome == "win":
        return 1
    if outcome == "draw":
        return 0
    if outcome == "loss":
        return -1
    raise Phase9ContractError(f"unknown terminal outcome: {outcome!r}")


def temporal_deltas(values: "list[float]", z: int) -> list:
    """The frozen per-step deltas of one learner-colour sequence.

    `values` holds `v_t` for that player's own decisions in order;
    `delta_t = v_{t+1} - v_t` when another learner decision follows and
    `delta_t = z - v_t` when the game terminates before that player's next
    decision (always the final entry).
    """
    if not values:
        return []
    deltas = [values[t + 1] - values[t] for t in range(len(values) - 1)]
    deltas.append(float(z) - values[-1])
    return deltas


def advantages(values: "list[float]", z: int) -> list:
    """`A_t = delta_t + lambda_A * A_{t+1}` with `A` beyond the end = 0."""
    deltas = temporal_deltas(values, z)
    result = [0.0] * len(deltas)
    following = 0.0
    for t in range(len(deltas) - 1, -1, -1):
        result[t] = deltas[t] + LAMBDA_ADVANTAGE * following
        following = result[t]
    return result


def wdl_lambda_targets(
    predictions: "list[tuple[float, float, float]]", outcome: str
) -> list:
    """The frozen soft W/D/L value targets of one learner-colour sequence.

    The terminal target (the final decision's) is the one-hot outcome `Z`;
    every earlier target blends the next decision's behavior prediction
    with the next target: `Y_t = (1 - lambda_V) * P_{t+1} + lambda_V *
    Y_{t+1}`. Value loss is categorical cross-entropy against these soft
    targets.
    """
    if not predictions:
        return []
    one_hot = {
        "win": (1.0, 0.0, 0.0),
        "draw": (0.0, 1.0, 0.0),
        "loss": (0.0, 0.0, 1.0),
    }
    if outcome not in one_hot:
        raise Phase9ContractError(f"unknown terminal outcome: {outcome!r}")
    targets: "list[tuple[float, float, float]]" = [None] * len(predictions)  # type: ignore[list-item]
    targets[-1] = one_hot[outcome]
    for t in range(len(predictions) - 2, -1, -1):
        next_prediction = predictions[t + 1]
        next_target = targets[t + 1]
        targets[t] = tuple(
            (1.0 - LAMBDA_VALUE) * float(next_prediction[k])
            + LAMBDA_VALUE * float(next_target[k])
            for k in range(3)
        )
    return targets


def quantile_linear(sorted_values: "list[float]", probability: float) -> float:
    """The frozen quantile rule: linear interpolation (numpy default)."""
    if not sorted_values:
        raise Phase9ContractError("quantile of an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise Phase9ContractError(f"probability must be in [0, 1], got {probability}")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def advantage_filter_threshold(all_advantages: "list[float]") -> float:
    """`tau = max(Q_0.75(|A|), 0.01)` over one sealed iteration."""
    magnitudes = sorted(abs(float(value)) for value in all_advantages)
    if not magnitudes:
        raise Phase9ContractError("advantage filter over an empty iteration")
    return max(
        quantile_linear(magnitudes, ADVANTAGE_FILTER_QUANTILE), ADVANTAGE_FILTER_FLOOR
    )


def advantage_semantics() -> dict:
    """The serializable `phase9_advantage_v1` contract."""
    return {
        "advantage_version": PHASE9_ADVANTAGE_VERSION,
        "constants": {
            "gamma": GAMMA,
            "lambda_A": LAMBDA_ADVANTAGE,
            "lambda_V": LAMBDA_VALUE,
        },
        "sequence_definition": (
            "per game and per learner-controlled colour, the sequence holds "
            "only that player's own decisions in ply order; the opponent move "
            "between consecutive learner states is never a learner training "
            "step"
        ),
        "behavior_value_scalar": "v_t = P_t(W) - P_t(L) from the stored behavior W/D/L",
        "delta": (
            "delta_t = v_{t+1} - v_t when another learner decision follows; "
            "delta_t = z - v_t when the game terminates before that player's "
            "next decision, z in {-1, 0, +1} from that player's final "
            "perspective"
        ),
        "advantage": "A_t = delta_t + lambda_A * A_{t+1}; A beyond the final step = 0",
        "wdl_lambda_target": (
            "terminal decision: Y_t = Z (one-hot outcome); otherwise "
            "Y_t = (1 - lambda_V) * P_{t+1} + lambda_V * Y_{t+1}; value loss "
            "stays categorical W/D/L cross-entropy against the soft target"
        ),
        "filter": {
            "threshold": "tau = max(Q_0.75(|A|), 0.01) per sealed iteration",
            "quantile_rule": "linear interpolation over the sorted |A| sample",
            "eligibility": "policy-gradient eligible iff |A_t| >= tau",
            "scope": (
                "the filter applies only to the PPO policy loss; value and "
                "belief train on every learner-controlled decision"
            ),
        },
        "standardization": (
            "advantages are standardized over the PPO-selected subset only: "
            "A_hat = (A - mean_selected) / (std_selected + 1e-8), population "
            "std (ddof=0), computed once per sealed iteration"
        ),
    }


# ---------------------------------------------------------------------------
# PPO, damping, and the full loss
# ---------------------------------------------------------------------------

PPO_CLIP_EPSILON = 0.20
BEHAVIOR_KL_TARGET = 0.015
KL_BETA_INCREASE_THRESHOLD = 0.0300
KL_BETA_DECREASE_THRESHOLD = 0.0075
KL_BETA_INCREASE_FACTOR = 2.0
KL_BETA_DECREASE_FACTOR = 0.5
KL_BETA_MIN = 1e-4
KL_BETA_MAX = 0.2

KL_HARD_LIMIT = 0.08
CLIP_FRACTION_HARD_LIMIT = 0.75

VALUE_LOSS_WEIGHT = 0.5
BELIEF_LOSS_WEIGHT = 0.25

ENTROPY_COEFFICIENT_START = 0.005
ENTROPY_COEFFICIENT_END = 0.001

MINIBATCH_SIZE = 512
EPOCHS_PER_ROLLOUT = 2

#: Common optimizer constraints. Learning rate and the initial KL beta are
#: selected only through the frozen pilot matrix; the Adam moments carry the
#: accepted Phase 8 values forward unchanged.
OPTIMIZER_CONSTRAINTS = {
    "precision": "float32",
    "device": "mps",
    "optimizer": "AdamW",
    "adam_betas": (0.9, 0.999),
    "adam_epsilon": 1e-8,
    "weight_decay": 0.01,
    "gradient_clip_norm": 1.0,
    "learning_rate_schedule": "constant (no warmup, no decay)",
    "minibatch_size": MINIBATCH_SIZE,
    "epochs_per_rollout": EPOCHS_PER_ROLLOUT,
}


def adaptive_kl_beta(current_beta: float, mean_epoch_kl: float) -> float:
    """The frozen KL controller update, applied once after each epoch."""
    if current_beta <= 0.0:
        raise Phase9ContractError(f"beta must be positive, got {current_beta}")
    if mean_epoch_kl > KL_BETA_INCREASE_THRESHOLD:
        updated = current_beta * KL_BETA_INCREASE_FACTOR
    elif mean_epoch_kl < KL_BETA_DECREASE_THRESHOLD:
        updated = current_beta * KL_BETA_DECREASE_FACTOR
    else:
        updated = current_beta
    return min(max(updated, KL_BETA_MIN), KL_BETA_MAX)


def entropy_coefficient(iteration: int, total_iterations: int) -> float:
    """The frozen linear entropy schedule, constant within an iteration.

    Linear in the 1-based iteration index over the run's own scheduled
    budget: iteration 1 uses the start value and iteration
    `total_iterations` the end value. Both the pilot (8) and the canonical
    run (60) traverse the same endpoints over their own frozen budgets.
    """
    if total_iterations < 1:
        raise Phase9ContractError(f"total_iterations must be >= 1, got {total_iterations}")
    if not 1 <= iteration <= total_iterations:
        raise Phase9ContractError(
            f"iteration {iteration} is outside 1..{total_iterations}"
        )
    if total_iterations == 1:
        return ENTROPY_COEFFICIENT_START
    progress = (iteration - 1) / (total_iterations - 1)
    return ENTROPY_COEFFICIENT_START + progress * (
        ENTROPY_COEFFICIENT_END - ENTROPY_COEFFICIENT_START
    )


def ppo_semantics() -> dict:
    """The frozen PPO objective, damping, and full loss."""
    return {
        "ratio": (
            "r_t(theta) = pi_theta(a_t|s_t) / pi_b(a_t|s_t); the denominator "
            "is the stored float32 behavior probability of the realized "
            "action; both logs floor at 1e-12"
        ),
        "clip_epsilon": PPO_CLIP_EPSILON,
        "objective": (
            "L_PPO = -E[min(r_t * A_hat_t, clip(r_t, 0.8, 1.2) * A_hat_t)] "
            "over the advantage-filtered, standardized PPO subset"
        ),
        "clip_fraction": (
            "fraction of PPO-subset decisions in an epoch with "
            "|r_t - 1| > 0.20"
        ),
        "behavior_kl": {
            "direction": (
                "D_KL(pi_b || pi_theta) = sum_a pi_b(a|s) * "
                "ln(pi_b(a|s) / pi_theta(a|s)) over the legal set"
            ),
            "population": (
                "every learner-controlled decision of the minibatch — not "
                "just the PPO subset — so damping tracks global drift from "
                "the behavior snapshot"
            ),
            "target": BEHAVIOR_KL_TARGET,
            "adaptive_beta": {
                "cadence": "updated once after each optimizer epoch",
                "increase": f"mean epoch KL > {KL_BETA_INCREASE_THRESHOLD} -> beta *= 2",
                "decrease": f"mean epoch KL < {KL_BETA_DECREASE_THRESHOLD} -> beta *= 0.5",
                "otherwise": "unchanged",
                "clamp": [KL_BETA_MIN, KL_BETA_MAX],
            },
            "hard_limits": {
                "mean_iteration_or_epoch_kl_fail": KL_HARD_LIMIT,
                "ppo_clip_fraction_fail": CLIP_FRACTION_HARD_LIMIT,
            },
        },
        "full_loss": (
            "L = L_PPO + 0.5 * L_value + 0.25 * L_belief + beta_KL * D_KL "
            "- c_H * H(pi_theta)"
        ),
        "loss_populations": {
            "ppo": "advantage-filtered learner decisions of the minibatch",
            "value": (
                "every learner-controlled decision of the minibatch: "
                "categorical cross-entropy against the soft W/D/L lambda "
                "target"
            ),
            "belief": (
                "every learner-controlled decision of the minibatch: the "
                "frozen Phase 8 hidden-only masked belief cross-entropy "
                "(continued belief supervision)"
            ),
            "kl": "every learner-controlled decision of the minibatch",
            "entropy": (
                "every learner-controlled decision of the minibatch: mean "
                "legal-softmax entropy of pi_theta"
            ),
        },
        "entropy_schedule": {
            "start": ENTROPY_COEFFICIENT_START,
            "end": ENTROPY_COEFFICIENT_END,
            "rule": (
                "linear in the 1-based iteration index over the run's own "
                "scheduled iteration budget; constant within an iteration"
            ),
        },
        "loss_weights": {"value": VALUE_LOSS_WEIGHT, "belief": BELIEF_LOSS_WEIGHT},
        "optimizer_constraints": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in OPTIMIZER_CONSTRAINTS.items()
        },
        "train_order": {
            "version": PHASE9_TRAIN_ORDER_VERSION,
            "universe": (
                "the sealed iteration's learner-controlled decisions (the "
                "value/belief population)"
            ),
            "shuffle": (
                "each epoch shuffles the universe with "
                "random.Random(train_order_seed(namespace, iteration, "
                "epoch)).shuffle over indices sorted by (game_id, ply)"
            ),
            "minibatches": (
                "contiguous 512-decision slices of the shuffled order; the "
                "final partial minibatch is consumed, never dropped"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Behavior-snapshot lifecycle and the rollout store
# ---------------------------------------------------------------------------

ROLLOUT_STATES = ("COLLECTING", "SEALED", "TRAINING", "EVALUATED", "COMMITTED")


def rollout_store_schema() -> dict:
    """The serializable `phase9_rollout_store_v1` contract."""
    return {
        "store_version": PHASE9_ROLLOUT_STORE_VERSION,
        "root": "data/phase9/rollouts/",
        "relocation": (
            "the root may be redirected to an external volume by explicit "
            "operator configuration (the cleared external drive is available "
            "for Phase 9 rollout storage); the manifest records the actual "
            "location, and identity is version + digests, never a path — "
            "the accepted Phase 8 relocation precedent"
        ),
        "payload": {
            "trajectory_version": "trajectory_v1",
            "snapshot_interval": 32,
            "role": "replay authority: setups, actions, decisions, terminal state",
        },
        "behavior_snapshot_lifecycle": [
            "freeze the current learner as an immutable behavior snapshot",
            "hash and record it (SHA-256)",
            "collect the entire iteration from that snapshot",
            "seal the rollout",
            "only then optimize",
            "create the next behavior snapshot only after the iteration is COMMITTED",
        ],
        "states": list(ROLLOUT_STATES),
        "metadata_fields": [
            "game_id",
            "rollout_version",
            "namespace",
            "iteration",
            "bucket",
            "ordinal",
            "learner_control",
            "learner_color",
            "red_policy_token",
            "blue_policy_token",
            "behavior_snapshot_id",
            "behavior_checkpoint_sha256",
            "opponent_kind",
            "opponent_identity",
            "opponent_checkpoint_sha256 (neural opponents; null otherwise)",
            "setup_root_seed",
            "red_policy_seed (rule/stress sides; null otherwise)",
            "blue_policy_seed (rule/stress sides; null otherwise)",
            "setup_provenance (both sides, setup_provenance_v1)",
            "terminal_result",
            "terminal_reason",
            "total_decisions",
            "learner_decision_count",
        ],
        "commit_rule": (
            "a game is trainable only once a commit record exists, written "
            "after both the trajectory payload and the Phase 9 metadata "
            "exist and verify; the Phase 8 corpus-commit journal pattern is "
            "the accepted shape"
        ),
        "seal_rule": (
            "a rollout seals only when every scheduled game id of the "
            "iteration is committed; the sealed rollout digest is the "
            "SHA-256 over the sorted committed (game_id, payload_sha256, "
            "metadata_sha256) triples; sealed rollouts are immutable"
        ),
        "crash_rules": [
            "collection crash -> deterministically regenerate only missing/uncommitted game ids",
            "no game becomes trainable until payload + metadata + commit all verify",
            "sealed rollouts are immutable",
            (
                "training crash -> resume the exact logical minibatch/"
                "optimizer/scheduler/KL-controller state from the same "
                "sealed rollout"
            ),
            "no next-iteration game may be generated before the current iteration is COMMITTED",
            "one iteration must never mix two behavior snapshot identities",
        ],
        "git_rule": (
            "production rollout shards and checkpoint archives stay out of "
            "Git; compact manifests, digests, contracts, tests and reports "
            "are tracked"
        ),
    }


#: `phase9_checkpoint_v1` minimum contents, frozen verbatim from the common
#: contract. Absolute paths are diagnostic only and never define identity.
CHECKPOINT_REQUIRED_FIELDS = (
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "global_optimizer_step",
    "rl_iteration",
    "minibatch_cursor",
    "examples_consumed",
    "behavior_snapshot_identity",
    "behavior_checkpoint_sha256",
    "rollout_iteration_identity",
    "sealed_rollout_digest",
    "kl_beta",
    "kl_controller_state",
    "entropy_schedule_position",
    "population_version",
    "active_historical_identities",
    "historical_checkpoint_digests",
    "opponent_schedule_version",
    "setup_sampler_version",
    "best_validation_score",
    "best_checkpoint_identity",
    "validation_history",
    "phase9_seeds",
    "corpus_identities",
    "rules_model_observation_versions",
    "wall_clock_counters",
    "software_runtime_versions",
)


# ---------------------------------------------------------------------------
# Evaluation banks (`phase9_eval_bank_v1`)
# ---------------------------------------------------------------------------

VALIDATION_BANK_VERSION = "phase9_validation_bank_v1"
TEST_BANK_VERSION = "phase9_test_bank_v1"

BANK_GENERATION_FAMILY = "phase9_family_balanced_v1"

VALIDATION_BANK_SPLIT = "validation"
TEST_BANK_SPLIT = "test"

VALIDATION_CASES_PER_FAMILY = 8
TEST_CASES_PER_FAMILY = 32
SETUP_FAMILY_COUNT = 16

VALIDATION_BANK_CASES = VALIDATION_CASES_PER_FAMILY * SETUP_FAMILY_COUNT
TEST_BANK_CASES = TEST_CASES_PER_FAMILY * SETUP_FAMILY_COUNT

#: Deterministic rejection-sampling attempt ceiling per bank-case side. The
#: frozen sampler draws families uniformly, so the acceptance probability
#: per attempt is 1/16 and 2,048 attempts failing has probability
#: (15/16)^2048 < 1e-57: the cap is unreachable in an honest library.
BANK_MAX_ATTEMPTS_PER_SIDE = 2048

#: Core evaluation opponents of both banks.
CORE_OPPONENTS = (
    "phase8_anchor",
    "random_legal",
    "basic_heuristic",
    "tactical_rule_based",
    "strategic_rule_based",
)

#: Report-only stress schedules: deterministic bank-prefix subsets, never a
#: score component.
VALIDATION_STRESS_PAIRS = 32
TEST_STRESS_PAIRS = 64

#: Frozen bootstrap statistics for every bank interval.
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_METHOD = "paired_unit_percentile_bootstrap"
BOOTSTRAP_ENGINE = "numpy_pcg64"


def eval_bank_contract() -> dict:
    """The serializable `phase9_eval_bank_v1` construction contract."""
    return {
        "eval_bank_version": PHASE9_EVAL_BANK_VERSION,
        "banks": {
            "validation": {
                "bank_version": VALIDATION_BANK_VERSION,
                "split": VALIDATION_BANK_SPLIT,
                "paired_cases": VALIDATION_BANK_CASES,
                "cases_per_family": VALIDATION_CASES_PER_FAMILY,
                "bootstrap_seed": VALIDATION_BOOTSTRAP_SEED,
                "role": (
                    "model selection: anchor baseline, pilot selection, "
                    "canonical validation cadence, best-checkpoint choice"
                ),
            },
            "test": {
                "bank_version": TEST_BANK_VERSION,
                "split": TEST_BANK_SPLIT,
                "paired_cases": TEST_BANK_CASES,
                "cases_per_family": TEST_CASES_PER_FAMILY,
                "bootstrap_seed": TEST_BOOTSTRAP_SEED,
                "role": (
                    "sealed final evaluation: structural audit only before "
                    "Agent 8; no neural model inference before Agent 8"
                ),
            },
        },
        "construction": {
            "generation_family": BANK_GENERATION_FAMILY,
            "case_identity": (
                "setup_pair_id = family_index * cases_per_family + "
                "case_ordinal over families F00..F15 (family-major)"
            ),
            "family_purity": (
                "both sides of a case draw from the case's family: the case "
                "family is unambiguous and setup-family EWR attributes "
                "cleanly; the learner colour swaps inside the paired unit, "
                "so family purity is the only colour-symmetric choice"
            ),
            "side_draws": (
                "for side in (red, blue): attempt k = 0, 1, ... draws "
                "sample_setup(split, eval_bank_draw_seed(bank_version, "
                "family_id, case_ordinal, side, k), profile='neutral_v1') "
                "through the frozen setup_sampler_v1 and accepts the first "
                "draw whose primary_family_id equals the case family; the "
                "accepted draw's engine orientation for that side is the "
                "case setup"
            ),
            "max_attempts_per_side": BANK_MAX_ATTEMPTS_PER_SIDE,
            "no_outcome_selection": (
                "construction never plays a game, never reads a strength "
                "signal, and never rejects a draw for anything but family "
                "identity"
            ),
        },
        "pairing_mode": PAIRING_COLOR_SWAP_SAME_BOARD,
        "evaluation_protocol": {
            "decision_rule": "greedy argmax (evaluation is never sampled)",
            "dtype": "float32",
            "rules": "frozen Phase 4 EVALUATION_RULES via the untouched match machinery",
            "match_root_seed": 20260401,
            "match_root_seed_rule": (
                "the frozen Phase 4 evaluation root seed; Phase 9 match "
                "identity separates from earlier phases through the bank "
                "version inside every match id"
            ),
            "anchor_candidate_identity": (
                "the Phase 8 anchor plays as neural_policy_ref("
                "'c1_warmstart', dtype_name='float32') through a bitwise-"
                "verified evaluation export of the accepted checkpoint — "
                "the accepted Agent 7 identity"
            ),
            "harness": (
                "run_neural_schedule for neural-vs-rule matchups; "
                "neural-vs-neural (current vs Phase 8 anchor and league "
                "cross-play) drives play_match directly with two in-process "
                "inference owners — the accepted Agent 7 shape"
            ),
        },
        "core_opponents": list(CORE_OPPONENTS),
        "stress_schedule": {
            "policies": list(STRESS_POLICY_ROSTER),
            "validation_pairs_per_policy": VALIDATION_STRESS_PAIRS,
            "validation_rule": (
                "the first 32 setup_pair_ids (0..31) of the validation bank "
                "per stress policy, report-only"
            ),
            "test_pairs_per_policy": TEST_STRESS_PAIRS,
            "test_rule": (
                "the first 64 setup_pair_ids (0..63) of the test bank per "
                "stress policy, report-only, Agent 8 only"
            ),
        },
        "statistics": {
            "method": BOOTSTRAP_METHOD,
            "engine": BOOTSTRAP_ENGINE,
            "resampling_unit": "paired_unit",
            "replicates": BOOTSTRAP_REPLICATES,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "seed_rule": (
                "matchup interval seed = matchup_seed(bank bootstrap seed, "
                "'candidate_token|opponent_token') through the frozen Phase 4 "
                "statistics derivation; validation banks use 2026081607, the "
                "final-test bank 2026081608"
            ),
            "paired_difference_rule": (
                "paired improvement resamples per-unit score differences "
                "(candidate unit score - anchor unit score on the same "
                "setup_pair_id) with the same frozen method, replicates, "
                "confidence, and seed rule under matchup token "
                "'diff|candidate_token|anchor_token|opponent_token'"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Validation score and cadence
# ---------------------------------------------------------------------------

VALIDATION_SCORE_WEIGHTS = {
    "strategic_rule_based": 0.45,
    "tactical_rule_based": 0.35,
    "phase8_anchor": 0.20,
}

VALIDATION_CADENCE_ITERATIONS = 5


def validation_score(
    strategic_ewr: float, tactical_ewr: float, anchor_ewr: float
) -> float:
    """`S = 0.45 * E_Strategic + 0.35 * E_Tactical + 0.20 * E_Phase8-anchor`."""
    for name, value in (
        ("strategic", strategic_ewr),
        ("tactical", tactical_ewr),
        ("anchor", anchor_ewr),
    ):
        if not 0.0 <= value <= 1.0:
            raise Phase9ContractError(f"{name} EWR must be in [0, 1], got {value}")
    return (
        VALIDATION_SCORE_WEIGHTS["strategic_rule_based"] * strategic_ewr
        + VALIDATION_SCORE_WEIGHTS["tactical_rule_based"] * tactical_ewr
        + VALIDATION_SCORE_WEIGHTS["phase8_anchor"] * anchor_ewr
    )


VALIDATION_TIE_BREAK = (
    "higher validation score",
    "higher Strategic EWR",
    "lower mean behavior KL",
    "higher training examples/s",
)

#: Random and Basic are regression guards on every validation pass, not
#: score components.
VALIDATION_REGRESSION_GUARDS = {
    "random_legal_ewr_min": 0.90,
    "basic_heuristic_ewr_min": 0.60,
}


def validation_contract() -> dict:
    return {
        "score": (
            "S = 0.45 * E_Strategic + 0.35 * E_Tactical + "
            "0.20 * E_Phase8_anchor; higher is better"
        ),
        "weights": dict(VALIDATION_SCORE_WEIGHTS),
        "bank": VALIDATION_BANK_VERSION,
        "games_per_opponent": VALIDATION_BANK_CASES * 2,
        "regression_guards": dict(VALIDATION_REGRESSION_GUARDS),
        "tie_break": list(VALIDATION_TIE_BREAK),
        "canonical_cadence": (
            "a full validation pass after every "
            f"{VALIDATION_CADENCE_ITERATIONS}th committed iteration "
            "(iterations 5, 10, ..., 60)"
        ),
        "pilot_cadence": "full validation passes after iterations 4 and 8",
        "best_checkpoint": (
            "the strictly highest frozen validation score among cadence "
            "passes; the final iteration is not automatically selected; "
            "exact ties fall through the frozen tie-break chain"
        ),
        "weights_rule": "a validation pass may select checkpoints; it may never update weights",
    }


# ---------------------------------------------------------------------------
# Pilot matrix (`exactly six candidates`)
# ---------------------------------------------------------------------------

PILOT_LEARNING_RATES = (1e-4, 3e-4, 6e-4)
PILOT_INITIAL_KL_BETAS = (0.005, 0.020)

PILOT_CANDIDATES = (
    {"candidate_id": "P9-A", "namespace": "pilot_p9a", "learning_rate": 1e-4, "initial_kl_beta": 0.005},
    {"candidate_id": "P9-B", "namespace": "pilot_p9b", "learning_rate": 1e-4, "initial_kl_beta": 0.020},
    {"candidate_id": "P9-C", "namespace": "pilot_p9c", "learning_rate": 3e-4, "initial_kl_beta": 0.005},
    {"candidate_id": "P9-D", "namespace": "pilot_p9d", "learning_rate": 3e-4, "initial_kl_beta": 0.020},
    {"candidate_id": "P9-E", "namespace": "pilot_p9e", "learning_rate": 6e-4, "initial_kl_beta": 0.005},
    {"candidate_id": "P9-F", "namespace": "pilot_p9f", "learning_rate": 6e-4, "initial_kl_beta": 0.020},
)

PILOT_CANDIDATE_LIMIT = 6

PILOT_ITERATIONS = 8

#: Pilot hard vetoes — any one disqualifies the candidate.
PILOT_HARD_VETOES = {
    "illegal_neural_action_max": 0,
    "non_finite_loss_max": 0,
    "non_finite_gradient_max": 0,
    "non_finite_parameter_max": 0,
    "behavior_identity_mismatch_max": 0,
    "target_reconstruction_mismatch_max": 0,
    "observer_safety_failure_max": 0,
    "checkpoint_resume_failure_max": 0,
    "mean_iteration_or_epoch_kl_max": KL_HARD_LIMIT,
    "iteration_ppo_clip_fraction_max": CLIP_FRACTION_HARD_LIMIT,
    "validation_random_ewr_min": 0.90,
    "validation_basic_ewr_min": 0.60,
}


def pilot_matrix() -> dict:
    """The complete frozen pilot contract: exactly six candidates."""
    if len(PILOT_CANDIDATES) != PILOT_CANDIDATE_LIMIT:
        raise Phase9ContractError(
            f"{len(PILOT_CANDIDATES)} pilot candidates; the frozen matrix has "
            f"exactly {PILOT_CANDIDATE_LIMIT}"
        )
    return {
        "candidates": [dict(candidate) for candidate in PILOT_CANDIDATES],
        "candidate_limit": PILOT_CANDIDATE_LIMIT,
        "allowed_dimensions": ["learning_rate", "initial_kl_beta"],
        "per_candidate_budget": {
            "start": "fresh from the accepted Phase 8 checkpoint",
            "rl_iterations": PILOT_ITERATIONS,
            "games_per_iteration": PILOT_GAMES_PER_ITERATION,
            "optimizer_epochs_per_rollout": EPOCHS_PER_ROLLOUT,
            "mixture": dict(PILOT_BUCKET_COUNTS),
            "rule_subdivision": dict(PILOT_RULE_TIER_COUNTS),
        },
        "selection": {
            "score": (
                "the frozen validation score at the final pilot checkpoint "
                "(after iteration 8); higher is better"
            ),
            "validation_cadence": "after iterations 4 and 8",
            "tie_break": list(VALIDATION_TIE_BREAK),
            "hard_vetoes": dict(PILOT_HARD_VETOES),
            "veto_scope": (
                "count-based vetoes cover the whole pilot run; threshold "
                "vetoes apply to every measured iteration/epoch and every "
                "validation pass"
            ),
            "forbidden_evidence": [
                "final-test results",
                "architecture changes",
                "mixture changes",
                "seed changes",
            ],
        },
        "no_seventh_run": True,
        "no_opportunistic_early_stop": True,
        "output": (
            "phase9_train_config_v1: the winner's learning rate and initial "
            "KL beta joined with every constant this contract already "
            "freezes"
        ),
    }


# ---------------------------------------------------------------------------
# Canonical run budget
# ---------------------------------------------------------------------------

CANONICAL_ITERATIONS = 60
CANONICAL_MAX_SCHEDULED_GAMES = CANONICAL_ITERATIONS * CANONICAL_GAMES_PER_ITERATION
CANONICAL_WALL_CLOCK_CEILING_HOURS = 12


def canonical_run_contract() -> dict:
    return {
        "start": "fresh from the accepted Phase 8 checkpoint",
        "rl_iterations": CANONICAL_ITERATIONS,
        "games_per_iteration": CANONICAL_GAMES_PER_ITERATION,
        "max_scheduled_games": CANONICAL_MAX_SCHEDULED_GAMES,
        "optimizer_epochs_per_rollout": EPOCHS_PER_ROLLOUT,
        "validation_cadence_iterations": VALIDATION_CADENCE_ITERATIONS,
        "archive_cadence_iterations": ARCHIVE_CADENCE_ITERATIONS,
        "wall_clock_ceiling_hours": CANONICAL_WALL_CLOCK_CEILING_HOURS,
        "ceiling_rule": (
            "the 12-hour ceiling is an operational maximum, not permission "
            "to shorten the logical contract silently; an incomplete run "
            "reports incomplete/blocked rather than pretending completion"
        ),
        "best_checkpoint": (
            "strictly highest frozen validation score among cadence passes"
        ),
    }


# ---------------------------------------------------------------------------
# Final hard gates (`phase9_acceptance_v1`)
# ---------------------------------------------------------------------------


def final_gates() -> dict:
    """The machine-readable Phase 9 acceptance gates. Never relaxed later."""
    return {
        "acceptance_version": PHASE9_ACCEPTANCE_VERSION,
        "bank": TEST_BANK_VERSION,
        "gate_a_direct_improvement_over_anchor": {
            "opponent": "Phase 8 anchor (the accepted warm-start checkpoint)",
            "paired_cases": TEST_BANK_CASES,
            "games": TEST_BANK_CASES * 2,
            "effective_win_rate_min": 0.58,
            "paired_bootstrap_lower_bound_exclusive": 0.53,
        },
        "gate_b_strategic": {
            "opponent": "strategic_rule_based",
            "final_ewr_min": 0.52,
            "paired_improvement_over_anchor_min": 0.05,
            "improvement_ci_lower_bound_exclusive": 0.0,
        },
        "gate_c_tactical": {
            "opponent": "tactical_rule_based",
            "final_ewr_min": 0.52,
            "paired_improvement_over_anchor_min": 0.05,
            "improvement_ci_lower_bound_exclusive": 0.0,
        },
        "stretch_report_only": {
            "strategic_ewr": 0.55,
            "tactical_ewr": 0.55,
        },
        "gate_d_random_guard": {
            "opponent": "random_legal",
            "overall_ewr_min": 0.94,
            "red_ewr_min": 0.90,
            "blue_ewr_min": 0.90,
            "paired_bootstrap_lower_bound_exclusive": 0.92,
        },
        "gate_e_basic_guard": {
            "opponent": "basic_heuristic",
            "ewr_min": 0.65,
            "paired_bootstrap_lower_bound_exclusive": 0.60,
        },
        "gate_f_safety": {
            "illegal_actions_max": 0,
            "model_failures_max": 0,
            "non_finite_outputs_max": 0,
            "observer_safety_failures_max": 0,
        },
        "gate_g_policy_collapse": {
            "population": (
                "every final-candidate decision across the final-test games"
            ),
            "max_legal_probability_threshold": 0.999,
            "fraction_above_threshold_max_exclusive": 0.25,
        },
        "gate_h_belief_retention": {
            "benchmark": (
                "the accepted Phase 8 held-out synthetic belief benchmark "
                "(sealed test split, warmstart_eval_v1 semantics)"
            ),
            "belief_ce_ratio_vs_remaining_count_max": 0.98,
            "belief_top1_must_beat_remaining_count_top1": True,
            "report_only": (
                "Phase 8-style teacher policy imitation CE is report-only "
                "in Phase 9"
            ),
        },
        "anchor_procedure": {
            "rule": (
                "after Agent 8 legitimately opens the final-test bank, the "
                "Phase 8 anchor plays the same final cases against "
                "Tactical and Strategic (512 pairs each, greedy, float32, "
                "frozen seeds); paired improvement per unit = candidate "
                "unit score - anchor unit score on the same setup_pair_id; "
                "the improvement CI is the frozen paired-difference "
                "bootstrap under the final-test seed rule"
            ),
        },
        "statistics": {
            "method": BOOTSTRAP_METHOD,
            "resampling_unit": "paired_unit",
            "replicates": BOOTSTRAP_REPLICATES,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "bootstrap_seed": TEST_BOOTSTRAP_SEED,
        },
        "report_only_rule": "report-only diagnostics may not rescue a failed hard gate",
    }


# ---------------------------------------------------------------------------
# Sealing — testable access rules
# ---------------------------------------------------------------------------

FINAL_EVALUATION_AGENT = 8

TEST_BANK_ALLOWED_ALWAYS = ("structural_audit",)
TEST_BANK_AGENT8_ONLY = ("final_evaluation",)
TEST_BANK_PROHIBITED_BEFORE_8 = (
    "neural_model_inference",
    "model_metric",
    "checkpoint_selection",
    "hyperparameter_selection",
)

VALIDATION_BANK_ALLOWED_ALWAYS = (
    "structural_audit",
    "anchor_baseline",
    "validation_scoring",
    "pilot_selection",
    "checkpoint_selection",
)
VALIDATION_BANK_PROHIBITED_ALWAYS = ("weight_update",)


class Phase9SealingError(Phase9ContractError):
    """A sealed Phase 9 resource was requested for a prohibited purpose."""


@dataclass(frozen=True)
class Phase9Access:
    """One authorized access to a Phase 9 evaluation bank."""

    resource: str
    purpose: str
    phase9_agent: int


def check_test_bank_access(purpose: str, *, phase9_agent: int) -> Phase9Access:
    """Authorize (or refuse) one access to the sealed Phase 9 test bank.

    Pure and stateless: agents 1-7 may run structural audits; every purpose
    that runs a neural model or informs a selection raises before Agent 8;
    Agent 8 may run the final evaluation. No test metric may influence
    Agents 1-7.
    """
    agent = int(phase9_agent)
    if not 1 <= agent <= FINAL_EVALUATION_AGENT:
        raise Phase9SealingError(f"unknown Phase 9 agent: {phase9_agent!r}")
    if purpose in TEST_BANK_ALLOWED_ALWAYS:
        return Phase9Access("phase9_test_bank", purpose, agent)
    if purpose in TEST_BANK_AGENT8_ONLY:
        if agent == FINAL_EVALUATION_AGENT:
            return Phase9Access("phase9_test_bank", purpose, agent)
        raise Phase9SealingError(
            f"test-bank purpose {purpose!r} is sealed until Agent "
            f"{FINAL_EVALUATION_AGENT}; agent {agent} may not open it"
        )
    if purpose in TEST_BANK_PROHIBITED_BEFORE_8:
        raise Phase9SealingError(
            f"test-bank purpose {purpose!r} is prohibited before Agent "
            f"{FINAL_EVALUATION_AGENT}; the sealed gate for Agent "
            f"{FINAL_EVALUATION_AGENT} is 'final_evaluation'"
        )
    raise Phase9SealingError(f"unknown test-bank purpose: {purpose!r}")


def check_validation_bank_access(purpose: str, *, phase9_agent: int) -> Phase9Access:
    """Authorize (or refuse) one access to the Phase 9 validation bank.

    The validation bank exists for model selection, so selection purposes
    are allowed to every agent; updating weights from it never is.
    """
    agent = int(phase9_agent)
    if not 1 <= agent <= FINAL_EVALUATION_AGENT:
        raise Phase9SealingError(f"unknown Phase 9 agent: {phase9_agent!r}")
    if purpose in VALIDATION_BANK_ALLOWED_ALWAYS:
        return Phase9Access("phase9_validation_bank", purpose, agent)
    if purpose in VALIDATION_BANK_PROHIBITED_ALWAYS:
        raise Phase9SealingError(
            f"validation-bank purpose {purpose!r} is always prohibited: the "
            "validation bank selects checkpoints and may never update weights"
        )
    raise Phase9SealingError(f"unknown validation-bank purpose: {purpose!r}")


def sealing_rules() -> dict:
    """The serializable held-out access policy."""
    return {
        "phase9_test_bank": {
            "allowed_always": list(TEST_BANK_ALLOWED_ALWAYS),
            "agent8_only": list(TEST_BANK_AGENT8_ONLY),
            "prohibited_before_agent_8": list(TEST_BANK_PROHIBITED_BEFORE_8),
        },
        "phase9_validation_bank": {
            "allowed_always": list(VALIDATION_BANK_ALLOWED_ALWAYS),
            "prohibited_always": list(VALIDATION_BANK_PROHIBITED_ALWAYS),
        },
        "phase8_test_corpus": {
            "rule": (
                "the accepted Phase 8 belief benchmark (gate H) reuses the "
                "sealed Phase 8 test split; Agent 8 alone may run it, "
                "through the Phase 8 gate check_test_corpus_access("
                "'final_evaluation', phase8_agent=7) semantics extended by "
                "this contract to the Phase 9 final evaluator; no Phase 9 "
                "agent 1-7 may run any model over Phase 8 test examples"
            ),
        },
        "enforcement": (
            "stratego.training.phase9_contract.check_test_bank_access / "
            "check_validation_bank_access; pure, stateless, regression-tested"
        ),
        "no_test_metric_before_agent_8": True,
    }


# ---------------------------------------------------------------------------
# Report-only diagnostics
# ---------------------------------------------------------------------------

REPORT_ONLY_DIAGNOSTICS = (
    "W/D/L and EWR by color",
    "setup-family EWR",
    "terminal-reason distribution",
    "game-length distribution",
    "policy entropy",
    "PPO clip fraction",
    "behavior KL",
    "advantage distribution",
    "advantage-filter retention fraction",
    "value calibration",
    "belief accuracy by piece type",
    "belief accuracy by game progress",
    "historical-opponent performance",
    "stress-policy performance",
    "league cross-play matrix",
    "archive pairwise EWR",
    "rollout throughput",
    "training throughput",
    "storage volume",
    "MPS memory",
    "CPU memory",
)


# ---------------------------------------------------------------------------
# Frozen upstream verification
# ---------------------------------------------------------------------------


def verify_phase9_upstream(*, include_library_digest: bool = True) -> list:
    """Every disagreement between the frozen Phase 9 inputs and live source.

    Reuses the accepted Phase 8 upstream verification (rules, engine,
    observation, model contract, action encoding, trajectory, banks,
    sampler, profile, C1 config) and layers the Phase 9 roster expectations
    on top. Checkpoint SHA-256 digests are file facts, so the acceptance
    runner verifies them against the filesystem and records the observation.
    """
    problems = list(verify_frozen_upstream(include_library_digest=include_library_digest))
    problems.extend(verify_teacher_roster())
    if tuple(LADDER_POLICY_IDS) != (
        "random_legal",
        "basic_heuristic",
        "tactical_rule_based",
        "strategic_rule_based",
    ):
        problems.append(f"ladder roster drifted: {LADDER_POLICY_IDS!r}")
    if tuple(STRESS_POLICY_IDS) != STRESS_POLICY_ROSTER:
        problems.append(
            f"stress roster drifted: expected {STRESS_POLICY_ROSTER!r}, found "
            f"{STRESS_POLICY_IDS!r}"
        )
    for tier in RULE_TIER_ORDER:
        if tier not in POLICY_INDEX:
            problems.append(f"rule tier {tier!r} is missing from the Phase 4 registry")
    return problems


# ---------------------------------------------------------------------------
# The complete serialized contract
# ---------------------------------------------------------------------------


def population_contract() -> dict:
    """The serializable `phase9_population_v1` document."""
    return {
        "population_version": PHASE9_POPULATION_VERSION,
        "proportions": dict(POPULATION_PROPORTIONS),
        "canonical": {
            "games_per_iteration": CANONICAL_GAMES_PER_ITERATION,
            "bucket_counts": dict(CANONICAL_BUCKET_COUNTS),
            "rule_tier_counts": dict(CANONICAL_RULE_TIER_COUNTS),
        },
        "pilot": {
            "games_per_iteration": PILOT_GAMES_PER_ITERATION,
            "bucket_counts": dict(PILOT_BUCKET_COUNTS),
            "rule_tier_counts": dict(PILOT_RULE_TIER_COUNTS),
        },
        "rule_tier_order": list(RULE_TIER_ORDER),
        "stress_roster": list(STRESS_POLICY_ROSTER),
        "stress_rotation": "(ordinal + iteration) % 6 over the frozen roster",
        "learner_control": {
            "labels": list(LEARNER_CONTROLS),
            "training_eligibility": dict(TRAINING_ELIGIBILITY),
            "opponent_rule": (
                "opponent decisions remain in the trajectory for state "
                "reconstruction but receive no Phase 9 policy/value/belief "
                "loss in that iteration"
            ),
        },
        "color_balance": (
            "asymmetric games: learner is red iff (ordinal + iteration) % 2 "
            "== 0; odd remainders alternate by deterministic iteration parity"
        ),
        "historical_league": {
            "anchor": HISTORICAL_ANCHOR_ID,
            "anchor_source": EXPECTED_PHASE8_CHECKPOINT_PATH,
            "anchor_sha256": EXPECTED_PHASE8_CHECKPOINT_SHA256,
            "archive_cadence_iterations": ARCHIVE_CADENCE_ITERATIONS,
            "archive_id_rule": "H%03d at each archive-creating iteration",
            "active_window": (
                "Phase 8 anchor + the 8 most recent archive snapshots "
                "created strictly before the current iteration"
            ),
            "sampling": "uniform over the active window via the frozen opponent stream",
            "immutability": "no archive checkpoint may be overwritten",
            "meaning": "historical replay means historical opponents, not stale PPO examples",
        },
        "setup_assignment": {
            "source": (
                f"training_setup_source({EXPECTED_SETUP_PROFILE!r}) — the "
                "frozen Phase 7 train split; rollouts never touch a held-out "
                "base"
            ),
            "assign_call": {
                "root_seed": "phase9_seed.setup_root_seed(game_id)",
                "environment_id": 0,
                "generation": 0,
                "game_id": "the Phase 9 rollout game id",
            },
        },
    }


def rl_contract_document() -> dict:
    """The full frozen `phase9_rl_contract_v1` as one document."""
    return {
        "contract_version": PHASE9_RL_CONTRACT_VERSION,
        "contract_identities": list(CONTRACT_IDENTITIES),
        "rollout_version": PHASE9_ROLLOUT_VERSION,
        "frozen_phase8_inputs": {
            "checkpoint_path": EXPECTED_PHASE8_CHECKPOINT_PATH,
            "checkpoint_sha256": EXPECTED_PHASE8_CHECKPOINT_SHA256,
            "selected_update": EXPECTED_PHASE8_SELECTED_UPDATE,
            "canonical_untrained_path": EXPECTED_PHASE8_INIT_PATH,
            "canonical_untrained_sha256": EXPECTED_PHASE8_INIT_SHA256,
            "canonical_init_seed": CANONICAL_C1_INIT_SEED,
            "canonical_init_state_checksum": EXPECTED_PHASE8_INIT_STATE_CHECKSUM,
            "c1_parameters": EXPECTED_C1_PARAMETERS,
            "c1_config_digest": EXPECTED_C1_CONFIG_DIGEST,
            "train_config_document_digest": EXPECTED_TRAIN_CONFIG_DOCUMENT_DIGEST,
            "trainer_runtime_identity_digest": EXPECTED_TRAINER_RUNTIME_IDENTITY_DIGEST,
            "train_config_digest_note": (
                "two distinct namespaces: the 41-field frozen document vs "
                "the 31-field runtime identity; they are not required to be "
                "equal and must stay labeled distinctly"
            ),
            "corpus": {
                "version": EXPECTED_CORPUS_VERSION,
                "resolver": "stratego.training.synthetic_corpus.default_corpus_root()",
                "content_digest": EXPECTED_CORPUS_CONTENT_DIGEST,
                "metadata_digest": EXPECTED_CORPUS_METADATA_DIGEST,
                "commit_index_digest": EXPECTED_CORPUS_COMMIT_INDEX_DIGEST,
                "rules": (
                    "identity is version + digests, never a path; a digest "
                    "mismatch is BLOCKED; the corpus is never regenerated or "
                    "repaired in Phase 9"
                ),
            },
        },
        "canonical_seeds": dict(CANONICAL_PHASE9_SEEDS),
        "seed_derivation": (
            "stratego.training.phase9_seed: blake2b person 'strat-rl9', "
            "domain-separated streams, no global RNG cursor anywhere"
        ),
        "game_identity": {
            "id_function": "stratego.training.phase9_seed.phase9_game_id",
            "fields": ["rollout_version", "master_seed", "namespace", "iteration", "bucket", "ordinal"],
            "format_example": phase9_game_id(CANONICAL_NAMESPACE, 12, BUCKET_HISTORICAL, 137),
            "per_game_streams": {
                "setup_root": "setup-source root seed",
                "opponent:historical": "active-window archive draw (historical bucket only)",
                "policy:red": "red rule/stress policy match-level seed",
                "policy:blue": "blue rule/stress policy match-level seed",
                "behavior_sampler": "per-decision neural action sampling",
            },
            "per_ply_policy_randomness": (
                "derive_decision_seed(policy_seed, ply) — the frozen Phase 4 "
                "per-ply derivation, unchanged"
            ),
        },
        "population": population_contract(),
        "rollout_schedule": {
            "schedule_version": PHASE9_ROLLOUT_SCHEDULE_VERSION,
            "rule": (
                "pure arithmetic: exact per-bucket counts, contiguous rule "
                "subranges, stress rotation, parity colour balance; the only "
                "seeded draw is the historical active-window member"
            ),
            "entry_point": "phase9_contract.scheduled_game(namespace, iteration, bucket, ordinal)",
        },
        "behavior_policy": behavior_policy_semantics(),
        "advantage": advantage_semantics(),
        "ppo": ppo_semantics(),
        "rollout_store": rollout_store_schema(),
        "checkpoint": {
            "checkpoint_version": PHASE9_CHECKPOINT_VERSION,
            "required_fields": list(CHECKPOINT_REQUIRED_FIELDS),
            "path_rule": "absolute paths are diagnostic only and must not define identity",
            "resume_rule": (
                "exact logical resume: minibatch cursor, optimizer state, "
                "scheduler state, KL controller state and entropy position "
                "restore from the checkpoint against the same sealed rollout"
            ),
        },
        "eval_banks": eval_bank_contract(),
        "validation": validation_contract(),
        "pilot_matrix": pilot_matrix(),
        "canonical_run": canonical_run_contract(),
        "final_gates": final_gates(),
        "sealing_rules": sealing_rules(),
        "report_only_diagnostics": list(REPORT_ONLY_DIAGNOSTICS),
        "mission_boundaries": {
            "forbidden": [
                "decision-time search",
                "MCTS",
                "belief-search rollouts",
                "learned setup generation or selection",
                "setup-policy RL",
                "human training data",
                "Phase 8 corpus modification",
                "architecture search or C2/C3 replacement",
                "mixed-precision optimizer training",
                "rule changes, including two-square or continuous-chasing",
                "official 168-hour final run",
                "Phase 10+ work",
            ],
        },
        "artifact_namespaces": [
            "reports/phase_9_data/",
            "data/phase9/rollouts/",
            "checkpoints/phase9/",
            "checkpoints/phase9/archive/",
        ],
    }


def contract_digest() -> str:
    """SHA-256 over the canonical JSON of the frozen contract document."""
    canonical = json.dumps(rl_contract_document(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


__all__ = [
    "ACTIVE_WINDOW_RECENT_SNAPSHOTS",
    "ADVANTAGE_FILTER_FLOOR",
    "ADVANTAGE_FILTER_QUANTILE",
    "ADVANTAGE_STANDARDIZATION_EPSILON",
    "ARCHIVE_CADENCE_ITERATIONS",
    "BANK_GENERATION_FAMILY",
    "BANK_MAX_ATTEMPTS_PER_SIDE",
    "BEHAVIOR_KL_TARGET",
    "BEHAVIOR_LOG_EPSILON",
    "BEHAVIOR_PROBABILITY_ABS_TOLERANCE",
    "BEHAVIOR_TEMPERATURE",
    "BELIEF_LOSS_WEIGHT",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_REPLICATES",
    "CANONICAL_BUCKET_COUNTS",
    "CANONICAL_GAMES_PER_ITERATION",
    "CANONICAL_ITERATIONS",
    "CANONICAL_MAX_SCHEDULED_GAMES",
    "CANONICAL_RULE_TIER_COUNTS",
    "CHECKPOINT_REQUIRED_FIELDS",
    "CLIP_FRACTION_HARD_LIMIT",
    "CONTRACT_IDENTITIES",
    "CORE_OPPONENTS",
    "ENTROPY_COEFFICIENT_END",
    "ENTROPY_COEFFICIENT_START",
    "EPOCHS_PER_ROLLOUT",
    "EXPECTED_PHASE8_CHECKPOINT_PATH",
    "EXPECTED_PHASE8_CHECKPOINT_SHA256",
    "EXPECTED_PHASE8_INIT_PATH",
    "EXPECTED_PHASE8_INIT_SHA256",
    "EXPECTED_PHASE8_INIT_STATE_CHECKSUM",
    "FINAL_EVALUATION_AGENT",
    "GAMMA",
    "HISTORICAL_ANCHOR_ID",
    "KL_BETA_DECREASE_THRESHOLD",
    "KL_BETA_INCREASE_THRESHOLD",
    "KL_BETA_MAX",
    "KL_BETA_MIN",
    "KL_HARD_LIMIT",
    "LAMBDA_ADVANTAGE",
    "LAMBDA_VALUE",
    "LEARNER_CONTROLS",
    "LEARNER_CONTROL_BLUE",
    "LEARNER_CONTROL_BOTH",
    "LEARNER_CONTROL_RED",
    "MINIBATCH_SIZE",
    "OPTIMIZER_CONSTRAINTS",
    "PHASE9_ACCEPTANCE_VERSION",
    "PHASE9_ADVANTAGE_VERSION",
    "PHASE9_CHECKPOINT_VERSION",
    "PHASE9_EVAL_BANK_VERSION",
    "PHASE9_POPULATION_VERSION",
    "PHASE9_RL_CONTRACT_VERSION",
    "PHASE9_ROLLOUT_SCHEDULE_VERSION",
    "PHASE9_ROLLOUT_STORE_VERSION",
    "PHASE9_TRAIN_ORDER_VERSION",
    "PILOT_BUCKET_COUNTS",
    "PILOT_CANDIDATES",
    "PILOT_CANDIDATE_LIMIT",
    "PILOT_GAMES_PER_ITERATION",
    "PILOT_HARD_VETOES",
    "PILOT_INITIAL_KL_BETAS",
    "PILOT_ITERATIONS",
    "PILOT_LEARNING_RATES",
    "PILOT_RULE_TIER_COUNTS",
    "POPULATION_PROPORTIONS",
    "PPO_CLIP_EPSILON",
    "REPORT_ONLY_DIAGNOSTICS",
    "ROLLOUT_STATES",
    "RULE_TIER_ORDER",
    "SETUP_FAMILY_COUNT",
    "STRESS_POLICY_ROSTER",
    "TEST_BANK_CASES",
    "TEST_BANK_VERSION",
    "TEST_CASES_PER_FAMILY",
    "TRAINING_ELIGIBILITY",
    "VALIDATION_BANK_CASES",
    "VALIDATION_BANK_VERSION",
    "VALIDATION_CADENCE_ITERATIONS",
    "VALIDATION_CASES_PER_FAMILY",
    "VALIDATION_REGRESSION_GUARDS",
    "VALIDATION_SCORE_WEIGHTS",
    "VALIDATION_STRESS_PAIRS",
    "TEST_STRESS_PAIRS",
    "VALUE_LOSS_WEIGHT",
    "Phase9Access",
    "Phase9ContractError",
    "Phase9SealingError",
    "active_historical_window",
    "adaptive_kl_beta",
    "advantage_filter_threshold",
    "advantage_semantics",
    "advantages",
    "archive_snapshot_id",
    "archived_iterations_before",
    "behavior_policy_semantics",
    "behavior_value_scalar",
    "bucket_counts",
    "canonical_run_contract",
    "check_test_bank_access",
    "check_validation_bank_access",
    "contract_digest",
    "entropy_coefficient",
    "eval_bank_contract",
    "final_gates",
    "games_per_iteration",
    "historical_opponent_for",
    "iter_scheduled_games",
    "learner_color",
    "learner_control_for",
    "pilot_matrix",
    "population_contract",
    "ppo_semantics",
    "quantile_linear",
    "rl_contract_document",
    "rollout_store_schema",
    "rule_tier_counts",
    "rule_tier_for_ordinal",
    "scheduled_game",
    "sealing_rules",
    "stress_policy_for_ordinal",
    "temporal_deltas",
    "terminal_z",
    "validation_contract",
    "validation_score",
    "verify_phase9_upstream",
    "wdl_lambda_targets",
]
