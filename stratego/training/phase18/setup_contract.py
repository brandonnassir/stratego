"""Phase 18: the setup learner's frozen constants, identities, seeds and refusals.

Specification sources:

- `00_PHASE_18_ADAPTIVE_SEQUENCE_AND_COMMON_CONTRACT.md` sections 3.1 and 8
- `06_AGENT_4_G2_SETUP_PARITY_AND_SYNTHETIC_ASSAY.md` Part E
- `reports/phase18/ataraxos_setup_method_map_v2.json` rows S01-S30
- the authors' published implementation, commit
  `92db29e8ffc323b1b8a2804b5c3f84695d036b05` (`pyengine/core/rl.py` RLConfig
  `arr_*` defaults; `networks/arrangement_transformer.py`; `arrangement/buffer.py`)

Every number here is transcribed from the paper or the published code through
the method map; nothing numeric was chosen by this agent. Where the paper and
the code disagree the map's resolution is followed and named (S20 pool size:
code; S13 entropy residual: code; S25 optimizer: AdamW at zero decay, which is
Adam).

What Phase 18 must not inherit from Phase 17 (common contract 3.1)
-------------------------------------------------------------------
```text
I - h            ->  I - 10h            (S13; the stored h predicts I/10)
no handedness    ->  Flag forced right, seeded 50% reflection after (S04, S05)
one outcome      ->  running mean of every outcome per exact setup (S09)
32-episode steps ->  1,024-setup optimizer minibatch, 5 epochs (S26)
```

The refusals live in `SetupTrainingConfig.__post_init__`: a non-zero weight
decay, a lambda other than 1.0, a foreign KL direction or a changed
normalizer is refused at construction rather than silently computing a
different recipe.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

from ...engine.constants import (
    BLUE,
    FLAG,
    NUM_PIECE_TYPES,
    PIECE_COUNTS,
    PIECES_PER_PLAYER,
    PLAYERS,
    RED,
)
from ...setups.identity import CANONICAL_CELLS, CANONICAL_FILES, derive_stream_seed

# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------

WORK_PACKAGE = "phase18_setup_integrated_warmstart"
SETUP_CONTRACT_VERSION = "phase18_setup_contract_v1"
SETUP_MODEL_VERSION = "phase18_setup_model_v1"
SETUP_RECIPE_VERSION = "phase18_published_setup_recipe_v1"
SETUP_EQUATION_VERSION = "phase18_setup_update_i_minus_10h_v1"
SETUP_BUFFER_VERSION = "phase18_setup_pool_buffer_v1"
SETUP_CHECKPOINT_VERSION = "phase18_setup_checkpoint_v1"

#: The accepted Phase 15 orientation rule. Never re-derived here.
ORIENTATION_RULE_VERSION = "phase15_orientation_rule_v1"

METHOD_MAP = "reports/phase18/ataraxos_setup_method_map_v2.json"
PAPER_ID = "arXiv:2511.07312v1"
PAPER_SHA256 = "f5d2d7c77dedd0b48c7278f8890aad784c44375cb996a7dbf728dbfc2e2afd04"
PUBLISHED_SOURCE = "https://github.com/AtaraxosAI/stratego"
PUBLISHED_SOURCE_COMMIT = "92db29e8ffc323b1b8a2804b5c3f84695d036b05"


class Phase18SetupError(RuntimeError):
    """Any refusal raised by the Phase 18 setup learner. Always fatal."""


class Phase18SetupConfigError(Phase18SetupError):
    """A configuration that would silently compute a different recipe."""


class Phase18SetupGenerationError(Phase18SetupError):
    """Sampling produced something that is not a legal, handed setup."""


class Phase18SetupOrientationError(Phase18SetupError):
    """A setup reached the engine boundary in the wrong frame."""


class Phase18SetupAttributionError(Phase18SetupError):
    """An outcome could not be attributed to a pooled setup. Never dropped."""


# ---------------------------------------------------------------------------
# Architecture (S01, S03, S30)
# ---------------------------------------------------------------------------

SETUP_BLOCKS = 4
SETUP_WIDTH = 128
SETUP_HEADS = 4
SETUP_FEED_FORWARD_WIDTH = 512
SETUP_NORMALIZATION = "pre_layernorm"

#: 12 live piece types plus one start token (S03: the published 14-way head
#: carries two structurally dead classes; the 12-way head is the same
#: distribution restricted to the live classes).
SETUP_VOCABULARY = NUM_PIECE_TYPES + 1
START_TOKEN = NUM_PIECE_TYPES
SETUP_SEQUENCE_LENGTH = PIECES_PER_PLAYER + 1
SETUP_PREFIXES = PIECES_PER_PLAYER
POSITIONAL_INIT_STD = 0.1

#: Frozen by the instruction: exactly 802,320 trainable parameters.
SETUP_PARAMETER_TARGET = 802_320
SETUP_PARAMETER_TOLERANCE = 0

#: S08: the published W/D/L category order is (loss, draw, win) = (0, 1, 2)
#: and the expected value is `probabilities @ CATEGORICAL_AGGREGATION`.
WDL_CLASS_ORDER = ("loss", "draw", "win")
WDL_LOSS, WDL_DRAW, WDL_WIN = 0, 1, 2
CATEGORICAL_AGGREGATION = (-1.0, 0.0, 1.0)

# ---------------------------------------------------------------------------
# Handedness and reflection (S04, S05)
# ---------------------------------------------------------------------------

#: The published `right_side` buffer: files 5..9 of every rank. The Flag is
#: illegal on the other five files during generation.
FLAG_PERMITTED_FILES = tuple(range(CANONICAL_FILES // 2, CANONICAL_FILES))
REFLECTION_PROBABILITY = 0.5

# ---------------------------------------------------------------------------
# Optimisation (S16-S18, S25-S28)
# ---------------------------------------------------------------------------

SETUP_LEARNING_RATE = 5e-5
SETUP_OPTIMIZER = "AdamW"
SETUP_WEIGHT_DECAY = 0.0
SETUP_ADAM_BETAS = (0.9, 0.999)
SETUP_ADAM_EPSILON = 1e-8

SETUP_PPO_CLIP_EPSILON = 0.2
SETUP_POLICY_LOSS_WEIGHT = 1.0
SETUP_VALUE_LOSS_WEIGHT = 0.5
SETUP_ENTROPY_PREDICTION_LOSS_WEIGHT = 1.0
SETUP_BEHAVIOR_KL_COEFFICIENT = 0.1
SETUP_KL_DIRECTION = "reverse_current_given_behavior"

#: S12/S13: the paper's normalizing constant. The entropy head predicts
#: `I / ENTROPY_NORMALIZER`; the residual restores nats with `I - 10h`.
ENTROPY_NORMALIZER = 10.0

SETUP_GRADIENT_CLIP_NORM = 0.5
SETUP_EMA_DECAY = 0.999
SETUP_EPOCHS_PER_UPDATE = 5

#: S26: the OPTIMIZER MINIBATCH is 1,024 setups; a step is taken per minibatch.
SETUP_BATCH_SIZE = 1024

#: S20: one pool of 1,024 per snapshot, 512 per player-use lane.
SETUP_POOL_SIZE = 1024
SETUP_POOL_PER_LANE = SETUP_POOL_SIZE // 2

#: S15: pinned. The flat advantage is the published recursion's lambda = 1
#: specialization and is refused for anything else.
TD_LAMBDA = 1.0
GAE_LAMBDA = 1.0

# ---------------------------------------------------------------------------
# Regularization temperature (S14)
# ---------------------------------------------------------------------------

ALPHA_COEFFICIENT = 0.1
ALPHA_DECAY = 0.3
ALPHA_CEIL = 1.0
ALPHA_FLOOR = 0.001


def setup_alpha(iteration: int) -> float:
    """`alpha(n) = clip(0.1 / n**0.3, 0.001, 1.0)`, `n` the one-based iteration.

    The published `power_schedule(coef, step, decay, ceil, floor)` evaluates
    `coef / (step + 1)**decay` on the zero-based global counter, which is this
    one-based form. Neither clamp binds inside any realistic horizon; both are
    recorded so a later re-horizoning cannot reintroduce them silently.
    """
    if int(iteration) < 1:
        raise Phase18SetupError(f"iteration is one-based, got {iteration}")
    value = ALPHA_COEFFICIENT / (float(iteration) ** ALPHA_DECAY)
    return min(max(value, ALPHA_FLOOR), ALPHA_CEIL)


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

INVENTORY_VECTOR = tuple(PIECE_COUNTS[piece_type] for piece_type in range(NUM_PIECE_TYPES))
assert sum(INVENTORY_VECTOR) == PIECES_PER_PLAYER == CANONICAL_CELLS

#: The exclusion value on the TRAINING path. Generation excludes with `-inf`;
#: a differentiable `log_softmax` over `-inf` gives NaN gradients, so the loss
#: uses a finite sentinel whose float32 `exp` underflows to exactly 0.0.
MASKED_LOGIT = -1e9

# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def json_document_digest(payload) -> str:
    """sha256 over the canonical JSON serialization the project already uses."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Seeds: every stream derives through the recorded `derive_stream_seed`
# ---------------------------------------------------------------------------

#: 53 bits is the exact mantissa of a float64, so `seed -> uniform` is lossless.
_UNIFORM_BITS = 53
_UNIFORM_MASK = (1 << _UNIFORM_BITS) - 1
_UNIFORM_SCALE = float(1 << _UNIFORM_BITS)


def seed_uniform(seed: int) -> float:
    """Map a seed to a uniform in `[0, 1)` without a generator object."""
    return float(int(seed) & _UNIFORM_MASK) / _UNIFORM_SCALE


def stream_seed(namespace: str, *parts) -> int:
    """`derive_stream_seed(namespace, *parts)`: one recorded seed function."""
    if not namespace:
        raise Phase18SetupError("seed namespace must be non-empty")
    return derive_stream_seed(namespace, *[str(part) for part in parts])


def model_seed(namespace: str, seed_index: int) -> int:
    return stream_seed(namespace, "model_init", int(seed_index))


def pool_root_seed(namespace: str, seed_index: int, snapshot_iteration: int, index: int) -> int:
    """The `(seed, snapshot, pool index)` root of one chain's 40 token draws."""
    return stream_seed(namespace, "pool", int(seed_index), int(snapshot_iteration), int(index))


def token_seed(root_seed: int, prefix: int) -> int:
    """Per-prefix seed, derived from the chain root so a draw at prefix `k` is
    independent of how the chain was batched."""
    if not 0 <= int(prefix) < SETUP_PREFIXES:
        raise Phase18SetupError(f"prefix out of range: {prefix!r}")
    return stream_seed("phase18_setup_token", int(root_seed), int(prefix))


def reflection_seed(namespace: str, seed_index: int, snapshot_iteration: int, index: int) -> int:
    """S05: an INDEPENDENT stream from the token draws."""
    return stream_seed(namespace, "reflection", int(seed_index), int(snapshot_iteration), int(index))


def shuffle_seed(namespace: str, seed_index: int, update: int, epoch: int) -> int:
    return stream_seed(namespace, "shuffle", int(seed_index), int(update), int(epoch))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SetupTrainingConfig:
    """Everything the setup learner needs, with its provenance attached.

    One recipe, one path. Every field defaults to the published value; the
    fields that would change the recipe if moved are refused in
    `__post_init__` rather than parameterised.
    """

    run_id: str
    device: str = "cpu"
    learning_rate: float = SETUP_LEARNING_RATE
    weight_decay: float = SETUP_WEIGHT_DECAY
    adam_betas: tuple = SETUP_ADAM_BETAS
    adam_epsilon: float = SETUP_ADAM_EPSILON
    ppo_clip_epsilon: float = SETUP_PPO_CLIP_EPSILON
    policy_loss_weight: float = SETUP_POLICY_LOSS_WEIGHT
    value_loss_weight: float = SETUP_VALUE_LOSS_WEIGHT
    entropy_prediction_loss_weight: float = SETUP_ENTROPY_PREDICTION_LOSS_WEIGHT
    behavior_kl_coefficient: float = SETUP_BEHAVIOR_KL_COEFFICIENT
    kl_direction: str = SETUP_KL_DIRECTION
    entropy_normalizer: float = ENTROPY_NORMALIZER
    gradient_clip_norm: float = SETUP_GRADIENT_CLIP_NORM
    ema_decay: float = SETUP_EMA_DECAY
    epochs_per_update: int = SETUP_EPOCHS_PER_UPDATE
    batch_size: int = SETUP_BATCH_SIZE
    pool_size: int = SETUP_POOL_SIZE
    td_lambda: float = TD_LAMBDA
    gae_lambda: float = GAE_LAMBDA
    force_handedness: bool = True
    reflection_probability: float = REFLECTION_PROBABILITY

    def __post_init__(self) -> None:
        if not self.run_id:
            raise Phase18SetupConfigError("run_id must be non-empty")
        if float(self.weight_decay) != 0.0:
            raise Phase18SetupConfigError(
                f"the setup weight decay is 0.0 (S25: AdamW at zero decay is the "
                f"paper's Adam); {self.weight_decay} would be a different optimizer"
            )
        if float(self.td_lambda) != 1.0 or float(self.gae_lambda) != 1.0:
            raise Phase18SetupConfigError(
                "the flat advantage is the published recursion's lambda = 1.0 "
                f"specialization; td_lambda={self.td_lambda}, "
                f"gae_lambda={self.gae_lambda} would silently compute a different "
                "target (S15)"
            )
        if self.kl_direction != SETUP_KL_DIRECTION:
            raise Phase18SetupConfigError(
                f"the setup behavior KL is {SETUP_KL_DIRECTION!r}; "
                f"{self.kl_direction!r} would flip the paper's direction (S17)"
            )
        if float(self.entropy_normalizer) != ENTROPY_NORMALIZER:
            raise Phase18SetupConfigError(
                f"the entropy normalizer is {ENTROPY_NORMALIZER} (S12/S13); "
                f"{self.entropy_normalizer} would change both the target and the residual"
            )
        if self.epochs_per_update < 1 or self.batch_size < 1 or self.pool_size < 1:
            raise Phase18SetupConfigError("epochs, batch size and pool size must be positive")
        if not 0.0 < float(self.ema_decay) < 1.0:
            raise Phase18SetupConfigError(f"EMA decay must be in (0, 1), got {self.ema_decay}")
        if not 0.0 <= float(self.reflection_probability) <= 1.0:
            raise Phase18SetupConfigError("reflection probability must be in [0, 1]")
        if self.behavior_kl_coefficient < 0.0:
            raise Phase18SetupConfigError("the behavior KL coefficient must be >= 0")

    def replace(self, **changes) -> "SetupTrainingConfig":
        return replace(self, **changes)

    def alpha(self, iteration: int) -> float:
        return setup_alpha(iteration)

    def document(self) -> dict:
        return {
            "recipe": SETUP_RECIPE_VERSION,
            "setup_contract_version": SETUP_CONTRACT_VERSION,
            "setup_model_version": SETUP_MODEL_VERSION,
            "setup_equation_version": SETUP_EQUATION_VERSION,
            "setup_buffer_version": SETUP_BUFFER_VERSION,
            "orientation_rule_version": ORIENTATION_RULE_VERSION,
            "method_map": METHOD_MAP,
            "paper": PAPER_ID,
            "published_source_commit": PUBLISHED_SOURCE_COMMIT,
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
                "prefixes": SETUP_PREFIXES,
                "parameter_target": SETUP_PARAMETER_TARGET,
                "positional_init_std": POSITIONAL_INIT_STD,
                "wdl_class_order": list(WDL_CLASS_ORDER),
                "categorical_aggregation": list(CATEGORICAL_AGGREGATION),
            },
            "generation": {
                "force_handedness": self.force_handedness,
                "flag_permitted_files": list(FLAG_PERMITTED_FILES),
                "reflection_probability": self.reflection_probability,
                "reflection_seed_stream": "independent of the token stream",
                "pool_size": self.pool_size,
                "pool_per_lane": self.pool_size // 2,
                "actor": "raw model only",
            },
            "advantage": {
                "outcome_term": "z_bar - E[v_behavior], E[v] = p_win - p_loss",
                "entropy_residual": "I - 10h (h predicts I/10; restored to nats)",
                "alpha": "clip(0.1 / n**0.3, 0.001, 1.0), n one-based global iteration",
                "form": "flat; equals the published TD/GAE recursion at lambda = 1",
                "td_lambda": self.td_lambda,
                "gae_lambda": self.gae_lambda,
                "advantage_filter": None,
            },
            "aggregation": {
                "outcome": "running mean of every completed outcome per exact setup and snapshot within one collection period",
                "value_target": "mean one-hot W/D/L (soft distribution)",
                "zero_outcomes": "excluded, never a draw",
                "identity": "content fingerprint of the played canonical board; duplicates collapse to the newest snapshot",
            },
            "optimisation": {
                "optimizer": SETUP_OPTIMIZER,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "adam_betas": list(self.adam_betas),
                "adam_epsilon": self.adam_epsilon,
                "ppo_clip_epsilon": self.ppo_clip_epsilon,
                "policy_loss_weight": self.policy_loss_weight,
                "value_loss_weight": self.value_loss_weight,
                "entropy_prediction_loss_weight": self.entropy_prediction_loss_weight,
                "entropy_prediction_target": "I/10",
                "entropy_normalizer": self.entropy_normalizer,
                "behavior_kl": {
                    "direction": self.kl_direction,
                    "coefficient": self.behavior_kl_coefficient,
                    "adaptive": False,
                },
                "gradient_clip_norm": self.gradient_clip_norm,
                "epochs_per_update": self.epochs_per_update,
                "batch_size": self.batch_size,
                "ema_decay": self.ema_decay,
                "ema_cadence": "once after the complete setup update",
                "evaluation_model": "EMA only",
            },
            "device": self.device,
        }

    def config_digest(self) -> str:
        return json_document_digest(self.document())


__all__ = [
    "ALPHA_CEIL",
    "ALPHA_COEFFICIENT",
    "ALPHA_DECAY",
    "ALPHA_FLOOR",
    "BLUE",
    "CATEGORICAL_AGGREGATION",
    "ENTROPY_NORMALIZER",
    "FLAG",
    "FLAG_PERMITTED_FILES",
    "GAE_LAMBDA",
    "INVENTORY_VECTOR",
    "MASKED_LOGIT",
    "METHOD_MAP",
    "ORIENTATION_RULE_VERSION",
    "PAPER_ID",
    "PAPER_SHA256",
    "PLAYERS",
    "POSITIONAL_INIT_STD",
    "PUBLISHED_SOURCE",
    "PUBLISHED_SOURCE_COMMIT",
    "Phase18SetupAttributionError",
    "Phase18SetupConfigError",
    "Phase18SetupError",
    "Phase18SetupGenerationError",
    "Phase18SetupOrientationError",
    "RED",
    "REFLECTION_PROBABILITY",
    "SETUP_ADAM_BETAS",
    "SETUP_ADAM_EPSILON",
    "SETUP_BATCH_SIZE",
    "SETUP_BEHAVIOR_KL_COEFFICIENT",
    "SETUP_BLOCKS",
    "SETUP_BUFFER_VERSION",
    "SETUP_CHECKPOINT_VERSION",
    "SETUP_CONTRACT_VERSION",
    "SETUP_EMA_DECAY",
    "SETUP_ENTROPY_PREDICTION_LOSS_WEIGHT",
    "SETUP_EPOCHS_PER_UPDATE",
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
    "SETUP_POLICY_LOSS_WEIGHT",
    "SETUP_POOL_PER_LANE",
    "SETUP_POOL_SIZE",
    "SETUP_PPO_CLIP_EPSILON",
    "SETUP_PREFIXES",
    "SETUP_RECIPE_VERSION",
    "SETUP_SEQUENCE_LENGTH",
    "SETUP_VALUE_LOSS_WEIGHT",
    "SETUP_VOCABULARY",
    "SETUP_WEIGHT_DECAY",
    "SETUP_WIDTH",
    "START_TOKEN",
    "SetupTrainingConfig",
    "TD_LAMBDA",
    "WDL_CLASS_ORDER",
    "WDL_DRAW",
    "WDL_LOSS",
    "WDL_WIN",
    "WORK_PACKAGE",
    "file_sha256",
    "json_document_digest",
    "model_seed",
    "pool_root_seed",
    "reflection_seed",
    "seed_uniform",
    "setup_alpha",
    "shuffle_seed",
    "stream_seed",
    "token_seed",
]
