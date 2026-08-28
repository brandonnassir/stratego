"""Phase 17 Agent 3: the setup half's frozen constants, identities and refusals.

Specification sources:

- `00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md` sections 7, 8, 10, 12, 13
- `03_AGENT_3_AUTOREGRESSIVE_SETUP_NETWORK.md` sections 2-8
- `reports/phase17/phase17_contract_handoff_v1.json`
  (`schedules_and_controllers.setup`, `schedules_and_controllers.setup_architecture`,
  `schemas.setup_episode`, `schemas.encoding_rules`)
- `reports/phase17/ataraxos_method_map_v1.md` rows S01-S17

Why a separate `setup_contract` and not `contract.py`
-----------------------------------------------------
Agent 1's recommended module split reserves `phase17/contract.py` for the
shared move-side constants that Agent 2 owns. Agents 2 and 3 work in
parallel, so this module takes a setup-scoped name: nothing here is a move
constant and nothing here may be imported as one. The two gradient clips are
the concrete reason -- the setup side clips at 0.5 and the move side at 1.0
(method map rows S13 and M12), and merging them into one symbol is exactly
the mistake that would silently retune the move learner.

Nothing numeric here was chosen by this agent. Every value is either
transcribed from the paper through Agent 1's frozen map, or carries a
`PROVISIONAL_` prefix because Agent 1 deferred it to this agent's soak
(operator decision D5). A provisional value is a starting point that the gate
must measure, never a result.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace

from ...engine.constants import (
    BLUE,
    NUM_PIECE_TYPES,
    PIECE_COUNTS,
    PIECES_PER_PLAYER,
    PLAYERS,
    RED,
)
from ...setups.identity import CANONICAL_CELLS

# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------

#: Work package. A concrete training execution carries a separate run ID.
WORK_PACKAGE = "phase17"

#: This module's own contract identity, recorded in every artifact it touches.
SETUP_CONTRACT_VERSION = "phase17_setup_contract_v1"

#: The episode wire schema frozen by Agent 1.
SETUP_EPISODE_SCHEMA_VERSION = "phase17_setup_episode_v1"

#: The setup network's architecture identity.
SETUP_MODEL_VERSION = "phase17_setup_model_v1"

#: The setup update's identity. Bumping this invalidates recorded behavior data.
#:
#: v2 (operator decision D7-B, 2026-08-27): the advantage's entropy term changed
#: from the centered residual `alpha*(I/10 - h)` to the uncentered bonus
#: `0.9*alpha*(I/10)`. Agent 3's soak measured the centered form converging to
#: zero by construction -- `L_h` trains `h` toward `I/10`, so the residual is the
#: error of a prediction another loss term is actively making accurate, and it
#: fell to ~1/100th of the outcome term within ten iterations. The uncentered
#: form is the paper's own `(H - h) ~ 0.9*H` shape expressed in the normalized
#: I/10 units, which keeps it commensurate with the outcome term.
SETUP_EQUATION_VERSION = "phase17_setup_update_v2"

#: The queue identity frozen by Agent 1.
SETUP_QUEUE_VERSION = "phase17_setup_episode_queue_v1"

#: The KL controller identity frozen by Agent 1 (constants provisional, D5).
SETUP_KL_CONTROLLER_VERSION = "phase17_setup_behavior_kl_controller_v1"

#: The accepted Phase 15 orientation rule. Never re-derived here.
ORIENTATION_RULE_VERSION = "phase15_orientation_rule_v1"


class Phase17SetupError(RuntimeError):
    """Any refusal raised by the Phase 17 setup half.

    Every failure mode in this half is fatal by contract: there is no setup
    library, no repair path and no fallback, so a subclass that a caller could
    catch and continue from would defeat the point.
    """


class Phase17SetupGenerationError(Phase17SetupError):
    """Sampling produced something that is not a legal setup."""


class Phase17SetupOrientationError(Phase17SetupError):
    """A setup reached the engine boundary in the wrong frame."""


# ---------------------------------------------------------------------------
# Architecture (method map S01; handoff `setup_architecture`, status FINAL)
# ---------------------------------------------------------------------------

SETUP_BLOCKS = 4
SETUP_WIDTH = 128
SETUP_HEADS = 4
SETUP_FEED_FORWARD_WIDTH = 512
SETUP_NORMALIZATION = "pre_layernorm"

#: 12 piece types plus one start token.
SETUP_VOCABULARY = NUM_PIECE_TYPES + 1
START_TOKEN = NUM_PIECE_TYPES

#: start token + 40 canonical row-major piece tokens.
SETUP_SEQUENCE_LENGTH = PIECES_PER_PLAYER + 1
SETUP_PREFIXES = PIECES_PER_PLAYER

#: The paper's learned-positional-embedding initialisation (Table 23).
POSITIONAL_INIT_STD = 0.1

#: Agent 1's arithmetic, restated so the gate can assert against a number that
#: was frozen before the model existed rather than against whatever it builds.
SETUP_PARAMETER_TARGET = 802_320

#: The architecture gate fails outside this band rather than adjusting widths.
#: Zero width: the count is fully determined by 4/128/4/512 and the three
#: heads, so any deviation at all means the shape changed.
SETUP_PARAMETER_TOLERANCE = 0

# ---------------------------------------------------------------------------
# Optimisation (method map S09-S16; handoff `schedules_and_controllers.setup`)
# ---------------------------------------------------------------------------

#: Adam, constant. The paper schedules only the move learning rate (S14).
SETUP_LEARNING_RATE = 5e-5
SETUP_OPTIMIZER = "Adam"
SETUP_ADAM_BETAS = (0.9, 0.999)
SETUP_ADAM_EPSILON = 1e-8

SETUP_PPO_CLIP_EPSILON = 0.2
SETUP_VALUE_LOSS_WEIGHT = 0.5
SETUP_CONDITIONAL_ENTROPY_LOSS_WEIGHT = 1.0

#: The paper's 1/10 normaliser (Eq. 1). Both sides of the advantage's entropy
#: term are expressed in these units -- operator decision D4.
SETUP_CONDITIONAL_ENTROPY_NORMALIZER = 0.1

#: The paper's `(H - h)` reduces to `0.9*H` once `h` has converged to `H/10`.
#: D7-B keeps that coefficient and expresses it in normalized units, so the
#: bonus is `0.9 * alpha * (I/10)`.
SETUP_ENTROPY_BONUS_COEFFICIENT = 0.9

#: Distinct from the move side's 1.0 (row M12). Do not merge.
SETUP_GRADIENT_CLIP_NORM = 0.5

SETUP_EMA_DECAY = 0.999
SETUP_EPOCHS_PER_ITERATION = 5

# ---------------------------------------------------------------------------
# Regularization temperature (method map S15; operator decision D3)
# ---------------------------------------------------------------------------

#: Agent 1's derived constant: the paper's own iteration count.
N_PAPER = 42_376
PAPER_ALPHA_START = 0.1
PAPER_ALPHA_EXPONENT = 0.3

#: alpha(N_paper) under the paper's raw schedule. Both endpoints are preserved
#: exactly by the re-horizoning, so this is also the floor.
ALPHA_FLOOR = PAPER_ALPHA_START * float(N_PAPER) ** -PAPER_ALPHA_EXPONENT


def setup_alpha_exponent(total_iterations: int) -> float:
    """`p = 0.3 * ln(N_paper) / ln(N)` -- the endpoint-preserving rescale.

    Operator decision D3. The raw transcription `0.1 * n**-0.3` ends 3.5x
    more heavily regularized than the paper on a 12-hour horizon, so the setup
    policy would never leave the high-entropy regime. Rescaling the exponent
    keeps the same power-law family and pins both endpoints; the move LR's
    `n_ref` shift preserves only the upper one.
    """
    if total_iterations < 2:
        raise Phase17SetupError(
            f"the alpha horizon needs at least 2 iterations, got {total_iterations}"
        )
    return PAPER_ALPHA_EXPONENT * math.log(N_PAPER) / math.log(float(total_iterations))


def setup_alpha(iteration: int, total_iterations: int) -> float:
    """`alpha(n) = max(0.1 * n**-p, alpha(N_paper))`, one-based `n`."""
    if iteration < 1:
        raise Phase17SetupError(f"iteration is one-based, got {iteration}")
    exponent = setup_alpha_exponent(total_iterations)
    return max(PAPER_ALPHA_START * float(iteration) ** -exponent, ALPHA_FLOOR)


# ---------------------------------------------------------------------------
# PROVISIONAL constants -- operator decision D5, calibrated by this agent's gate
# ---------------------------------------------------------------------------

#: Reverse KL, `D_KL(pi_current || pi_behavior)`, which is the paper's
#: direction (S11) and deliberately NOT the move controller's forward
#: direction. The direction is a required logged field precisely so the two
#: are never reported as one another.
#:
#: RESOLVED by operator decision D5, 2026-08-27, from Agent 3's measured soak.
#: The names keep their `PROVISIONAL_` prefix nowhere: these are frozen. The
#: 0.015 target was borrowed from the move controller and measurement showed it
#: sits roughly 4x above the observed p95, so the controller drove beta to its
#: lower bound and held it there for every iteration -- regulating nothing. The
#: target is now the measured scale and the lower bound is widened by a decade
#: so the controller has room to act in both directions.
SETUP_KL_DIRECTION = "reverse_current_given_behavior"

#: The observed MEDIAN of the per-iteration control KL, not its p95.
#:
#: The first D5 resolution took 0.0037 -- the p95 Agent 3's v1 report
#: highlighted. Measured against a controller that steps once per iteration on
#: the final epoch's KL, that target sits above the median (0.00176), so 55% of
#: iterations fall below the decrease threshold against 2.5% above it and beta
#: walks to its floor by iteration 30. Anchoring on the median centres the hold
#: band on the actual scale: roughly 8% decrease / 75% hold / 17% increase.
#: Operator decision, 2026-08-27.
SETUP_KL_TARGET = 0.0018
SETUP_KL_BETA_INITIAL = 0.1
SETUP_KL_BETA_BOUNDS = (0.001, 1.0)
SETUP_KL_HARD_LIMIT = 0.08

#: A controller that lives at a bound is not controlling anything. The soak
#: reports the fraction of iterations spent at each bound and the gate fails
#: above this share -- the failure mode D5 was opened to diagnose.
SETUP_KL_PINNED_FRACTION_LIMIT = 0.50

#: Agent 3's chosen response shape. D5 fixed the direction, target, initial
#: beta, bounds and hard limit; it left the response ratios and factors open,
#: and these are not the accepted Phase 9 controller's. Phase 9 increases above
#: 2.0x its target and steps by 2.0 / 0.5; these are gentler (1.5x, 1.5 / 1.5),
#: which suits a controller whose measured signal is small and rises steadily
#: within an iteration. Stated explicitly because an earlier version of this
#: comment claimed Phase 9 fidelity that the numbers do not have.
SETUP_KL_INCREASE_THRESHOLD_RATIO = 1.5
SETUP_KL_DECREASE_THRESHOLD_RATIO = 0.5
SETUP_KL_INCREASE_FACTOR = 1.5
SETUP_KL_DECREASE_FACTOR = 1.0 / 1.5

PROVISIONAL_SETUP_QUEUE_CAPACITY = 4096
PROVISIONAL_SETUP_QUEUE_MAX_AGE_ITERATIONS = 8

#: Common contract section 12's provisional production floors.
PROVISIONAL_PREFIX_ENTROPY_FLOOR_FRACTION = 0.60
PROVISIONAL_PREFIX_ENTROPY_FLOOR_CONSECUTIVE_CHECKS = 3
PROVISIONAL_FLAG_EFFECTIVE_SUPPORT_FLOOR = 4.0

#: Pool sizing band (S04). The gate picks the smallest size that keeps the
#: game creator supplied.
SETUP_POOL_SIZE_BAND = (512, 1000)

PROVISIONAL_FIELDS = (
    "setup_queue_capacity",
    "setup_queue_max_age_iterations",
)

#: Frozen by the operator after Agent 3's measured soak.
RESOLVED_FIELDS = {
    "D5 (setup KL controller)": (
        "direction, target 0.0037, beta0 0.1, bounds [0.001, 1.0], hard limit 0.08"
    ),
    "D7-B (setup advantage entropy term)": "0.9 * alpha * (I/10), uncentered",
}

# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

#: `PIECE_COUNTS` as a dense 12-vector, in piece-type order. The single source
#: of truth stays the engine's dict; this is only its array view.
INVENTORY_VECTOR = tuple(PIECE_COUNTS[piece_type] for piece_type in range(NUM_PIECE_TYPES))
assert sum(INVENTORY_VECTOR) == PIECES_PER_PLAYER == CANONICAL_CELLS


# ---------------------------------------------------------------------------
# Digests (handoff `schemas.encoding_rules`)
# ---------------------------------------------------------------------------


def json_document_digest(payload) -> str:
    """sha256 over the canonical JSON serialization Phase 16 already uses."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path) -> str:
    """sha256 over a file's bytes."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Seed domains (Agent 3 instruction section 4)
# ---------------------------------------------------------------------------

_ROOT_SEED_PERSON = b"p17setuprt"
_TOKEN_SEED_PERSON = b"p17setuptk"


def _derive(person: bytes, payload: str) -> int:
    """A 63-bit domain-separated blake2b seed, matching `setups.identity`."""
    digest = hashlib.blake2b(payload.encode(), digest_size=8, person=person).digest()
    return int.from_bytes(digest, "big") >> 1


def setup_root_seed(run_id: str, game_id: str, color: int) -> int:
    """The `(run, game, side)` seed domain.

    Hashed rather than arithmetic so that two adjacent games, or the two sides
    of one game, never receive correlated streams -- the Red and Blue setups of
    a single game must be independent draws or the diversity metrics measure
    the seed scheme instead of the model.
    """
    if color not in PLAYERS:
        raise Phase17SetupError(f"unknown colour: {color!r}")
    if not run_id or not game_id:
        raise Phase17SetupError("run_id and game_id must both be non-empty")
    return _derive(_ROOT_SEED_PERSON, f"{SETUP_CONTRACT_VERSION}:{run_id}:{game_id}:{int(color)}")


def setup_token_seed(root_seed: int, prefix: int) -> int:
    """The per-prefix seed domain.

    Derived from the root rather than from a sequential generator so that
    prefix `k`'s draw is independent of how the chain was batched. A pool of
    512 chains and a single chain therefore produce identical setups.
    """
    if not 0 <= prefix < SETUP_PREFIXES:
        raise Phase17SetupError(f"prefix out of range: {prefix!r}")
    return _derive(_TOKEN_SEED_PERSON, f"{int(root_seed)}:{int(prefix)}")


#: The exclusion value used on the TRAINING path. Generation excludes with
#: `-inf`, which is exact and has no backward pass; a differentiable
#: `log_softmax` over `-inf` produces NaN gradients, so the loss uses a finite
#: sentinel instead. In float32 `exp(-1e9)` underflows to exactly 0.0, so the
#: masked probability is still exactly zero -- the mask is not softened, only
#: its representation is.
MASKED_LOGIT = -1e9


_SHUFFLE_SEED_PERSON = b"p17setupsh"


def derive_shuffle_seed(run_id: str, setup_iteration: int, epoch: int, offset: int = 0) -> int:
    """The per-(iteration, epoch) minibatch-order seed.

    Domain-separated blake2b rather than Python's `hash`, which is randomized
    per process: a resume in a fresh process must reproduce the same epoch and
    minibatch order or the checkpoint round trip proves nothing.
    """
    payload = f"{run_id}:{int(setup_iteration)}:{int(epoch)}:{int(offset)}"
    return _derive(_SHUFFLE_SEED_PERSON, payload)


#: 53 bits is the exact mantissa of a float64, so `seed -> uniform` is
#: lossless and identical on every platform.
_UNIFORM_BITS = 53
_UNIFORM_MASK = (1 << _UNIFORM_BITS) - 1
_UNIFORM_SCALE = float(1 << _UNIFORM_BITS)


def seed_uniform(seed: int) -> float:
    """Map a seed to a uniform draw in `[0, 1)` without a PRNG object.

    Sampling by inverse CDF against this value keeps generation exactly
    reproducible under any batching, on any device, in any order -- which a
    per-chain `torch.Generator` would not.
    """
    return float(int(seed) & _UNIFORM_MASK) / _UNIFORM_SCALE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SetupTrainingConfig:
    """Everything the setup half needs, with its provenance attached."""

    run_id: str
    total_iterations: int
    device: str = "cpu"
    epochs_per_iteration: int = SETUP_EPOCHS_PER_ITERATION
    minibatch_episodes: int = 32
    pool_size_per_side: int = SETUP_POOL_SIZE_BAND[0]
    learning_rate: float = SETUP_LEARNING_RATE
    ppo_clip_epsilon: float = SETUP_PPO_CLIP_EPSILON
    value_loss_weight: float = SETUP_VALUE_LOSS_WEIGHT
    conditional_entropy_loss_weight: float = SETUP_CONDITIONAL_ENTROPY_LOSS_WEIGHT
    gradient_clip_norm: float = SETUP_GRADIENT_CLIP_NORM
    ema_decay: float = SETUP_EMA_DECAY
    kl_direction: str = SETUP_KL_DIRECTION
    kl_target: float = SETUP_KL_TARGET
    kl_beta_initial: float = SETUP_KL_BETA_INITIAL
    kl_beta_bounds: tuple = SETUP_KL_BETA_BOUNDS
    kl_hard_limit: float = SETUP_KL_HARD_LIMIT
    queue_capacity: int = PROVISIONAL_SETUP_QUEUE_CAPACITY
    queue_max_age_iterations: int = PROVISIONAL_SETUP_QUEUE_MAX_AGE_ITERATIONS
    seed_offset: int = 0

    def __post_init__(self) -> None:
        if self.total_iterations < 2:
            raise Phase17SetupError("total_iterations must be at least 2")
        if self.epochs_per_iteration < 1:
            raise Phase17SetupError("epochs_per_iteration must be at least 1")
        if self.minibatch_episodes < 1:
            raise Phase17SetupError("minibatch_episodes must be at least 1")
        if self.pool_size_per_side < 1:
            raise Phase17SetupError("pool_size_per_side must be at least 1")
        low, high = self.kl_beta_bounds
        if not 0.0 < low <= high:
            raise Phase17SetupError(f"invalid kl beta bounds: {self.kl_beta_bounds!r}")
        if not low <= self.kl_beta_initial <= high:
            raise Phase17SetupError("kl_beta_initial is outside its own bounds")
        if self.queue_capacity < 1:
            raise Phase17SetupError("queue_capacity must be at least 1")

    def replace(self, **changes) -> "SetupTrainingConfig":
        return replace(self, **changes)

    def alpha(self, iteration: int) -> float:
        return setup_alpha(iteration, self.total_iterations)

    def document(self) -> dict:
        """The digestible record of this configuration."""
        return {
            "setup_contract_version": SETUP_CONTRACT_VERSION,
            "setup_model_version": SETUP_MODEL_VERSION,
            "setup_equation_version": SETUP_EQUATION_VERSION,
            "setup_episode_schema_version": SETUP_EPISODE_SCHEMA_VERSION,
            "setup_queue_version": SETUP_QUEUE_VERSION,
            "setup_kl_controller_version": SETUP_KL_CONTROLLER_VERSION,
            "orientation_rule_version": ORIENTATION_RULE_VERSION,
            "work_package": WORK_PACKAGE,
            "run_id": self.run_id,
            "architecture": {
                "blocks": SETUP_BLOCKS,
                "width": SETUP_WIDTH,
                "heads": SETUP_HEADS,
                "feed_forward_width": SETUP_FEED_FORWARD_WIDTH,
                "normalization": SETUP_NORMALIZATION,
                "vocabulary": SETUP_VOCABULARY,
                "sequence_length": SETUP_SEQUENCE_LENGTH,
                "parameter_target": SETUP_PARAMETER_TARGET,
            },
            "optimisation": {
                "optimizer": SETUP_OPTIMIZER,
                "learning_rate": self.learning_rate,
                "adam_betas": list(SETUP_ADAM_BETAS),
                "adam_epsilon": SETUP_ADAM_EPSILON,
                "ppo_clip_epsilon": self.ppo_clip_epsilon,
                "value_loss_weight": self.value_loss_weight,
                "conditional_entropy_loss_weight": self.conditional_entropy_loss_weight,
                "conditional_entropy_normalizer": SETUP_CONDITIONAL_ENTROPY_NORMALIZER,
                "gradient_clip_norm": self.gradient_clip_norm,
                "ema_decay": self.ema_decay,
                "epochs_per_iteration": self.epochs_per_iteration,
                "minibatch_episodes": self.minibatch_episodes,
            },
            "alpha_schedule": {
                "formula": "max(0.1 * n**-p, 0.1 * N_paper**-0.3)",
                "n_paper": N_PAPER,
                "total_iterations": self.total_iterations,
                "p": setup_alpha_exponent(self.total_iterations),
                "alpha_first": setup_alpha(1, self.total_iterations),
                "alpha_last": setup_alpha(self.total_iterations, self.total_iterations),
                "floor": ALPHA_FLOOR,
                "operator_decision": "D3 ACCEPTED",
            },
            "kl_controller": {
                "version": SETUP_KL_CONTROLLER_VERSION,
                "direction": self.kl_direction,
                "target": self.kl_target,
                "beta_initial": self.kl_beta_initial,
                "beta_bounds": list(self.kl_beta_bounds),
                "hard_limit": self.kl_hard_limit,
                "status": "RESOLVED by operator decision D5, 2026-08-27",
            },
            "queue": {
                "version": SETUP_QUEUE_VERSION,
                "capacity": self.queue_capacity,
                "max_age_iterations": self.queue_max_age_iterations,
                "status": "PROVISIONAL pending operator decision D5",
            },
            "pool": {
                "size_per_side": self.pool_size_per_side,
                "band": list(SETUP_POOL_SIZE_BAND),
                "fallback": "none -- generation or orientation failure is fatal",
            },
            "provisional_fields": list(PROVISIONAL_FIELDS),
        }

    def config_digest(self) -> str:
        return json_document_digest(self.document())


__all__ = [
    "ALPHA_FLOOR",
    "BLUE",
    "INVENTORY_VECTOR",
    "MASKED_LOGIT",
    "N_PAPER",
    "ORIENTATION_RULE_VERSION",
    "PROVISIONAL_FIELDS",
    "PROVISIONAL_FLAG_EFFECTIVE_SUPPORT_FLOOR",
    "PROVISIONAL_PREFIX_ENTROPY_FLOOR_CONSECUTIVE_CHECKS",
    "PROVISIONAL_PREFIX_ENTROPY_FLOOR_FRACTION",
    "SETUP_KL_BETA_BOUNDS",
    "SETUP_KL_BETA_INITIAL",
    "SETUP_KL_DIRECTION",
    "SETUP_KL_HARD_LIMIT",
    "SETUP_KL_PINNED_FRACTION_LIMIT",
    "SETUP_KL_TARGET",
    "PROVISIONAL_SETUP_QUEUE_CAPACITY",
    "PROVISIONAL_SETUP_QUEUE_MAX_AGE_ITERATIONS",
    "RED",
    "SETUP_ADAM_BETAS",
    "SETUP_ADAM_EPSILON",
    "SETUP_BLOCKS",
    "SETUP_CONDITIONAL_ENTROPY_LOSS_WEIGHT",
    "SETUP_CONDITIONAL_ENTROPY_NORMALIZER",
    "SETUP_CONTRACT_VERSION",
    "SETUP_EMA_DECAY",
    "SETUP_ENTROPY_BONUS_COEFFICIENT",
    "SETUP_EPISODE_SCHEMA_VERSION",
    "SETUP_EPOCHS_PER_ITERATION",
    "RESOLVED_FIELDS",
    "SETUP_EQUATION_VERSION",
    "SETUP_FEED_FORWARD_WIDTH",
    "SETUP_GRADIENT_CLIP_NORM",
    "SETUP_HEADS",
    "SETUP_KL_CONTROLLER_VERSION",
    "SETUP_KL_DECREASE_FACTOR",
    "SETUP_KL_DECREASE_THRESHOLD_RATIO",
    "SETUP_KL_INCREASE_FACTOR",
    "SETUP_KL_INCREASE_THRESHOLD_RATIO",
    "SETUP_LEARNING_RATE",
    "SETUP_MODEL_VERSION",
    "SETUP_NORMALIZATION",
    "SETUP_OPTIMIZER",
    "SETUP_PARAMETER_TARGET",
    "SETUP_PARAMETER_TOLERANCE",
    "SETUP_POOL_SIZE_BAND",
    "SETUP_PPO_CLIP_EPSILON",
    "SETUP_PREFIXES",
    "SETUP_QUEUE_VERSION",
    "SETUP_SEQUENCE_LENGTH",
    "SETUP_VALUE_LOSS_WEIGHT",
    "SETUP_VOCABULARY",
    "SETUP_WIDTH",
    "START_TOKEN",
    "SetupTrainingConfig",
    "Phase17SetupError",
    "Phase17SetupGenerationError",
    "Phase17SetupOrientationError",
    "POSITIONAL_INIT_STD",
    "WORK_PACKAGE",
    "derive_shuffle_seed",
    "file_sha256",
    "json_document_digest",
    "seed_uniform",
    "setup_alpha",
    "setup_alpha_exponent",
    "setup_root_seed",
    "setup_token_seed",
]
