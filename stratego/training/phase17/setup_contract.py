"""Phase 17: the setup half's frozen constants, identities and refusals.

Specification sources:

- `09_OPERATOR_DECISION_D10_SIMPLIFIED_PAPER_TANDEM.md` sections 3, 4 and 7
- `00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md` sections 7, 8, 10, 12, 13
- `reports/phase17/ataraxos_method_map_v1.md` rows S01-S17

Why a separate `setup_contract` and not `contract.py`
-----------------------------------------------------
Agent 1's recommended module split reserves `phase17/contract.py` for the
shared move-side constants that Agent 2 owns. Nothing here is a move constant
and nothing here may be imported as one. The two gradient clips are the
concrete reason -- the setup side clips at 0.5 and the move side at 1.0
(method map rows S13 and M12), and merging them into one symbol is exactly
the mistake that would silently retune the move learner.

What operator decision D10 changed here (2026-08-28)
-----------------------------------------------------
D10 stopped treating the setup learner as an independently certified
subsystem and put the paper's printed recipe on the active path. Three things
in this module were replaced outright rather than parameterised, because a
switch between two recipes is a second recipe:

```text
adaptive reverse-KL controller  ->  fixed coefficient 0.1
alpha re-horizoned on N (D3)    ->  alpha(n) = 0.1 * n**-0.3, no floor
uncentered I/10 bonus (D7-B)    ->  the printed advantage, alpha * (I - h)
fixed quota / warm-up / max age ->  every completed episode, exactly once
```

The measurements that produced D3, D5 and D7-B remain valid history; they are
simply no longer the active recipe. Nothing numeric here was chosen by this
agent: every value is transcribed from the paper through Agent 1's frozen map
or restated verbatim from D10 section 4.
"""

from __future__ import annotations

import hashlib
import json
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

#: The active recipe under operator decision D10. Recorded in every config
#: document, checkpoint and handoff so a D7-B/D5 artifact and a D10 artifact can
#: never be read as the same experiment.
SETUP_RECIPE_VERSION = "phase17_simple_paper_tandem_v1"

#: The one production run of this recipe. Agent 3/4's rehearsal lineage
#: `RUN-2026-A` is a different run and its state may not enter production.
PRODUCTION_RUN_ID = "RUN-2026-B"

#: The setup update's identity. Bumping this invalidates recorded behavior data,
#: which is the point: a `phase17_setup_update_v2` checkpoint carries advantages
#: built from a different equation and must be refused, not migrated.
#:
#: D10 retires v2's locally invented uncentered `0.9*alpha*(I/10)` bonus and
#: uses the paper's printed advantage `(o - E[v]) + alpha*(I - h)` directly.
SETUP_EQUATION_VERSION = "phase17_setup_update_paper_v1"

#: The completed-episode buffer's identity. Under D10 this is a pending buffer
#: that is fully drained every global iteration, not Agent 3's bounded FIFO with
#: a fixed quota, so the version changes with the semantics.
SETUP_BUFFER_VERSION = "phase17_completed_setup_buffer_v1"

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

#: The paper's 1/10 normaliser (Eq. 1). Under D10 it applies to `L_h`'s TARGET
#: only: the conditional-entropy head is trained toward `I/10`, while the
#: printed advantage uses `I` in raw nats against the recorded `h`. That
#: asymmetry is the paper's own and D10 section 4 keeps it deliberately --
#: "do not add a compensating scale, floor, centering rule, horizon map, or
#: controller". It is measured, not corrected: `advantage_terms` reports the
#: outcome and entropy magnitudes separately every iteration.
SETUP_CONDITIONAL_ENTROPY_NORMALIZER = 0.1

#: Distinct from the move side's 1.0 (row M12). Do not merge.
SETUP_GRADIENT_CLIP_NORM = 0.5

SETUP_EMA_DECAY = 0.999
SETUP_EPOCHS_PER_ITERATION = 5

# ---------------------------------------------------------------------------
# Behavior KL (method map S11; operator decision D10 section 4)
# ---------------------------------------------------------------------------

#: Reverse KL, `D_KL(pi_current || pi_behavior)`, which is the paper's
#: direction and deliberately NOT the move controller's forward direction. The
#: direction is a required logged field precisely so the two are never reported
#: as one another.
SETUP_KL_DIRECTION = "reverse_current_given_behavior"

#: FIXED. Not a beta, not a controller state, not a target.
#:
#: D5's adaptive controller is retired. Agent 4's 200-iteration tandem soak sat
#: at the controller's UPPER bound for 97.5% of its iterations, which is a
#: controller that is not controlling; D10 reads that as evidence for removing
#: it rather than for calibrating it again. Every artifact this module feeds
#: names this a coefficient so no reader can mistake a constant for a regulated
#: quantity.
SETUP_BEHAVIOR_KL_COEFFICIENT = 0.1

# ---------------------------------------------------------------------------
# Regularization temperature (method map S15; operator decision D10 section 4)
# ---------------------------------------------------------------------------

PAPER_ALPHA_START = 0.1
PAPER_ALPHA_EXPONENT = 0.3


def setup_alpha(iteration: int) -> float:
    """`alpha(n) = 0.1 * n**-0.3`, with `n` the one-based GLOBAL tandem iteration.

    The paper's printed schedule, transcribed, with no floor and no dependence
    on the expected run length `N`.

    Two earlier choices are retired here and both mattered:

    - D3 rescaled the exponent to `0.3*ln(N_paper)/ln(N)` so that alpha's two
      endpoints landed where the paper's did on a 42,376-iteration run. That
      re-horizoning is gone. D10 accepts that a 640-iteration run ends less
      annealed than the paper's and treats the resulting curve as the
      experiment rather than as something to be matched.
    - Agent 4 left it unresolved (`A4-CF6`) whether `n` counted setup updates
      or global iterations, and the two diverge whenever a setup update is
      skipped. D10 settles it: `n` is the shared one-based global tandem
      iteration the runner is on, the same `n` the move schedule reads. A
      skipped setup update therefore still advances alpha, because alpha is a
      property of where the run is, not of how many times the setup optimizer
      has fired.
    """
    if iteration < 1:
        raise Phase17SetupError(f"iteration is one-based, got {iteration}")
    return PAPER_ALPHA_START * float(iteration) ** -PAPER_ALPHA_EXPONENT


# ---------------------------------------------------------------------------
# Descriptive floors -- telemetry only under operator decision D10 section 7
# ---------------------------------------------------------------------------

#: These were the standalone setup gate's pass/fail thresholds. D10 retired the
#: gate and demoted every statistical setup reading to a warning, so they are
#: kept only so the telemetry can still say "below the level Agent 3 measured
#: as collapse". Nothing reads them to stop a run.
DESCRIPTIVE_PREFIX_ENTROPY_FLOOR_FRACTION = 0.60
DESCRIPTIVE_FLAG_EFFECTIVE_SUPPORT_FLOOR = 4.0

#: Pool sizing (S04, D10 section 4): 512 fresh samples per side, regenerated at
#: every global tandem iteration.
SETUP_POOL_SIZE_PER_SIDE = 512

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
    """Everything the setup half needs, with its provenance attached.

    One recipe, one path. There is no adaptive-controller branch to fall into
    and no quota to satisfy: `behavior_kl_coefficient` is a constant, alpha is a
    function of the shared global iteration alone, and every completed episode
    is consumed.
    """

    run_id: str
    device: str = "cpu"
    epochs_per_iteration: int = SETUP_EPOCHS_PER_ITERATION
    minibatch_episodes: int = 32
    pool_size_per_side: int = SETUP_POOL_SIZE_PER_SIDE
    learning_rate: float = SETUP_LEARNING_RATE
    ppo_clip_epsilon: float = SETUP_PPO_CLIP_EPSILON
    value_loss_weight: float = SETUP_VALUE_LOSS_WEIGHT
    conditional_entropy_loss_weight: float = SETUP_CONDITIONAL_ENTROPY_LOSS_WEIGHT
    gradient_clip_norm: float = SETUP_GRADIENT_CLIP_NORM
    ema_decay: float = SETUP_EMA_DECAY
    kl_direction: str = SETUP_KL_DIRECTION
    behavior_kl_coefficient: float = SETUP_BEHAVIOR_KL_COEFFICIENT
    seed_offset: int = 0

    def __post_init__(self) -> None:
        if self.epochs_per_iteration < 1:
            raise Phase17SetupError("epochs_per_iteration must be at least 1")
        if self.minibatch_episodes < 1:
            raise Phase17SetupError("minibatch_episodes must be at least 1")
        if self.pool_size_per_side < 1:
            raise Phase17SetupError("pool_size_per_side must be at least 1")
        if self.kl_direction != SETUP_KL_DIRECTION:
            raise Phase17SetupError(
                f"the setup behavior KL is {SETUP_KL_DIRECTION!r}; "
                f"{self.kl_direction!r} would silently flip the paper's direction "
                "into the move controller's"
            )
        if not self.behavior_kl_coefficient >= 0.0:
            raise Phase17SetupError(
                f"the behavior KL coefficient must be >= 0, got "
                f"{self.behavior_kl_coefficient}"
            )

    def replace(self, **changes) -> "SetupTrainingConfig":
        return replace(self, **changes)

    def alpha(self, iteration: int) -> float:
        """`alpha(n)` at the one-based GLOBAL tandem iteration `n`."""
        return setup_alpha(iteration)

    def document(self) -> dict:
        """The digestible record of this configuration."""
        return {
            "recipe": SETUP_RECIPE_VERSION,
            "setup_contract_version": SETUP_CONTRACT_VERSION,
            "setup_model_version": SETUP_MODEL_VERSION,
            "setup_equation_version": SETUP_EQUATION_VERSION,
            "setup_episode_schema_version": SETUP_EPISODE_SCHEMA_VERSION,
            "setup_buffer_version": SETUP_BUFFER_VERSION,
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
                "initialization": "from scratch under a recorded seed",
            },
            "optimisation": {
                "optimizer": SETUP_OPTIMIZER,
                "learning_rate": self.learning_rate,
                "adam_betas": list(SETUP_ADAM_BETAS),
                "adam_epsilon": SETUP_ADAM_EPSILON,
                "ppo_clip_epsilon": self.ppo_clip_epsilon,
                "value_loss_weight": self.value_loss_weight,
                "conditional_entropy_loss_weight": self.conditional_entropy_loss_weight,
                "conditional_entropy_loss_target": "I/10",
                "gradient_clip_norm": self.gradient_clip_norm,
                "ema_decay": self.ema_decay,
                "epochs_per_iteration": self.epochs_per_iteration,
                "minibatch_episodes": self.minibatch_episodes,
            },
            "advantage": {
                "formula": "(outcome - E[behavior W/D/L value]) + alpha(n) * (I - h_behavior)",
                "information_units": "nats",
                "h_units": "the recorded behavior prediction, trained toward I/10",
                "source": "the paper's printed setup advantage, D10 section 4",
                "retired": "phase17_setup_update_v2 (D7-B) 0.9 * alpha * (I/10)",
            },
            "alpha_schedule": {
                "formula": "0.1 * n**-0.3",
                "n": "the shared one-based global tandem iteration",
                "floor": None,
                "depends_on_run_length": False,
                "operator_decision": "D10 section 4 (supersedes D3)",
            },
            "behavior_kl": {
                "direction": self.kl_direction,
                "coefficient": self.behavior_kl_coefficient,
                "adaptive": False,
                "controller": None,
                "operator_decision": "D10 section 4 (supersedes D5)",
            },
            "completed_episode_buffer": {
                "version": SETUP_BUFFER_VERSION,
                "policy": (
                    "every episode whose game completed in the current "
                    "fixed-transition iteration, both sides, exactly once"
                ),
                "quota": None,
                "warm_up_minimum": None,
                "max_age_iterations": None,
                "persisted": "only the current iteration's unconsumed buffer",
            },
            "pool": {
                "size_per_side": self.pool_size_per_side,
                "cadence": "regenerated at every global tandem iteration",
                "fallback": "none -- generation or orientation failure is fatal",
            },
        }

    def config_digest(self) -> str:
        return json_document_digest(self.document())


__all__ = [
    "BLUE",
    "DESCRIPTIVE_FLAG_EFFECTIVE_SUPPORT_FLOOR",
    "DESCRIPTIVE_PREFIX_ENTROPY_FLOOR_FRACTION",
    "INVENTORY_VECTOR",
    "MASKED_LOGIT",
    "ORIENTATION_RULE_VERSION",
    "PAPER_ALPHA_EXPONENT",
    "PAPER_ALPHA_START",
    "POSITIONAL_INIT_STD",
    "PRODUCTION_RUN_ID",
    "Phase17SetupError",
    "Phase17SetupGenerationError",
    "Phase17SetupOrientationError",
    "RED",
    "SETUP_ADAM_BETAS",
    "SETUP_ADAM_EPSILON",
    "SETUP_BEHAVIOR_KL_COEFFICIENT",
    "SETUP_BLOCKS",
    "SETUP_CONDITIONAL_ENTROPY_LOSS_WEIGHT",
    "SETUP_CONDITIONAL_ENTROPY_NORMALIZER",
    "SETUP_CONTRACT_VERSION",
    "SETUP_EMA_DECAY",
    "SETUP_EPISODE_SCHEMA_VERSION",
    "SETUP_EPOCHS_PER_ITERATION",
    "SETUP_EQUATION_VERSION",
    "SETUP_FEED_FORWARD_WIDTH",
    "SETUP_GRADIENT_CLIP_NORM",
    "SETUP_HEADS",
    "SETUP_KL_DIRECTION",
    "SETUP_LEARNING_RATE",
    "SETUP_MODEL_VERSION",
    "SETUP_NORMALIZATION",
    "SETUP_OPTIMIZER",
    "SETUP_PARAMETER_TARGET",
    "SETUP_PARAMETER_TOLERANCE",
    "SETUP_POOL_SIZE_PER_SIDE",
    "SETUP_PPO_CLIP_EPSILON",
    "SETUP_PREFIXES",
    "SETUP_BUFFER_VERSION",
    "SETUP_RECIPE_VERSION",
    "SETUP_SEQUENCE_LENGTH",
    "SETUP_VALUE_LOSS_WEIGHT",
    "SETUP_VOCABULARY",
    "SETUP_WIDTH",
    "START_TOKEN",
    "SetupTrainingConfig",
    "WORK_PACKAGE",
    "derive_shuffle_seed",
    "file_sha256",
    "json_document_digest",
    "seed_uniform",
    "setup_alpha",
    "setup_root_seed",
    "setup_token_seed",
]
