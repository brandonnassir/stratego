"""Phase 16 Agent 3: the training-loop-v2 contract.

Specification source: `03_AGENT_3_TRAINING_LOOP_V2.md` sections 2-5.

What is Phase 16's own
----------------------
The *schedule* and the *data distribution*, and nothing else. Every number
that decides what the gradient is comes from the accepted Phase 9 contract and
is imported here rather than restated: PPO clip, both lambdas, the advantage
filter quantile/floor, the value and belief weights, the adaptive-beta KL
thresholds, the minibatch size. A tuned value would have to be tuned in the
frozen Phase 9 contract, where its own digest would catch it.

What this module adds is the set of things section 2 makes a *flag*, so the
three shootout arms differ by data rather than by code:

```text
collection      window (fixed learner-decision budget) -- the only collector
lr_schedule     power_law | constant
entropy         annealed | constant
epochs          1 | 2
ema             on | off              (evaluation-side only)
opponents       pure_current | phase14_mixture
setups          library | expanded
```

Seeds
-----
`phase16.agent3` domain roots through a blake2b derivation with this agent's
own personalization, matching the shape Agents 1 and 2 use. Nothing in this
phase draws from a Phase 14 seed domain: a Phase 16 game with a Phase 14-shaped
ordinal must not replay a Phase 14 board.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace

from ..phase9_contract import (
    ADVANTAGE_FILTER_FLOOR,
    ADVANTAGE_FILTER_QUANTILE,
    ADVANTAGE_STANDARDIZATION_EPSILON,
    BEHAVIOR_KL_TARGET,
    BELIEF_LOSS_WEIGHT,
    CLIP_FRACTION_HARD_LIMIT,
    ENTROPY_COEFFICIENT_END,
    ENTROPY_COEFFICIENT_START,
    GAMMA,
    KL_BETA_MAX,
    KL_BETA_MIN,
    KL_HARD_LIMIT,
    LAMBDA_ADVANTAGE,
    LAMBDA_VALUE,
    MINIBATCH_SIZE,
    PPO_CLIP_EPSILON,
    VALUE_LOSS_WEIGHT,
)

PHASE16_TRAINING_VERSION = "phase16_training_v2"
PHASE16_CONTRACT_VERSION = "phase16_training_contract_v1"
PHASE16_COLLECTOR_VERSION = "phase16_window_collector_v1"
PHASE16_TARGETS_VERSION = "phase16_window_targets_v1"
PHASE16_TRAINER_VERSION = "phase16_window_trainer_v1"
PHASE16_CHECKPOINT_VERSION = "phase16_checkpoint_v1"
PHASE16_RUNNER_VERSION = "phase16_arm_runner_v1"
PHASE16_NAMESPACE = "phase16a3"

#: The read-only P24 copy every arm starts from, and its two frozen digests.
STARTING_CHECKPOINT = "checkpoints/phase15/p24_source_readonly.pt"
STARTING_CHECKPOINT_SHA256 = (
    "9bf256a9b085176bf48c1eca424fa10cef109f09c90999b23be62e685e917fb1"
)
STARTING_MODEL_STATE_DIGEST = (
    "622d9e6caa723c932dedc5b77c257d532c1b0f8931f79851d863658f3cbbb87f"
)


class Phase16TrainingError(RuntimeError):
    """A Phase 16 training request is outside its contract."""


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

SEED_NAMESPACE = "phase16.agent3"
TRAINING_IDENTITY_VERSION = "phase16_training_identity_v1"

#: Master seed of this agent, folded into every stream below.
TRAINING_MASTER_SEED = 2026082530

PHASE16_TRAINING_PERSON = b"strat16t"

DOMAIN_GAME_DRAW = "game_draw"
DOMAIN_SETUP_SIDE = "setup_side"
DOMAIN_ACTION_SAMPLING = "action_sampling"
DOMAIN_TRAINING_ORDER = "training_order"
DOMAIN_OPPONENT_DRAW = "opponent_draw"
DOMAIN_POLICY = "policy"

STREAM_DOMAINS = (
    DOMAIN_GAME_DRAW,
    DOMAIN_SETUP_SIDE,
    DOMAIN_ACTION_SAMPLING,
    DOMAIN_TRAINING_ORDER,
    DOMAIN_OPPONENT_DRAW,
    DOMAIN_POLICY,
)

DOMAIN_ROOTS = {
    DOMAIN_GAME_DRAW: TRAINING_MASTER_SEED + 1,
    DOMAIN_SETUP_SIDE: TRAINING_MASTER_SEED + 2,
    DOMAIN_ACTION_SAMPLING: TRAINING_MASTER_SEED + 3,
    DOMAIN_TRAINING_ORDER: TRAINING_MASTER_SEED + 4,
    DOMAIN_OPPONENT_DRAW: TRAINING_MASTER_SEED + 5,
    DOMAIN_POLICY: TRAINING_MASTER_SEED + 6,
}


def derive_train_seed(domain: str, *parts: "int | str") -> int:
    """A 63-bit deterministic seed for one Phase 16 training stream."""
    if domain not in STREAM_DOMAINS:
        raise Phase16TrainingError(f"unknown Phase 16 training domain: {domain!r}")
    for part in parts:
        if not isinstance(part, (int, str)) or isinstance(part, bool):
            raise Phase16TrainingError(
                f"stream identity parts must be int or str, got {type(part).__name__}"
            )
        if isinstance(part, str) and ":" in part:
            raise Phase16TrainingError(
                f"string identity parts may not contain ':' (got {part!r})"
            )
    payload = ":".join(
        [
            TRAINING_IDENTITY_VERSION,
            SEED_NAMESPACE,
            domain,
            str(DOMAIN_ROOTS[domain]),
            *[str(part) for part in parts],
        ]
    )
    digest = hashlib.blake2b(
        payload.encode(), digest_size=8, person=PHASE16_TRAINING_PERSON
    ).digest()
    return int.from_bytes(digest, "big") >> 1


def uniform_from_seed(seed: int) -> float:
    """A float in [0, 1) from a derived seed, by the accepted 53-bit rule."""
    return ((int(seed) >> 10) & ((1 << 53) - 1)) / float(1 << 53)


# ---------------------------------------------------------------------------
# Logical game identity
# ---------------------------------------------------------------------------

#: `phase16_game_v1|ms=<master>|arm=<arm>|slot=<slot>|draw=<draw>`.
#:
#: A window collector has no iteration to key on: a game outlives the window
#: it started in, and two games in the same window may be at completely
#: different plies. The identity that *is* stable is "the n-th game played in
#: this population slot", which is exactly what a persistent population has
#: and what makes a replacement draw reproducible from the run state alone.
GAME_ID_VERSION = "phase16_game_v1"

_GAME_ID_PATTERN = re.compile(
    rf"^{re.escape(GAME_ID_VERSION)}\|ms={TRAINING_MASTER_SEED}"
    r"\|arm=(?P<arm>[a-z0-9_]+)\|slot=(?P<slot>[0-9]{4})\|draw=(?P<draw>[0-9]{6})$"
)

MAX_SLOT_FORMAT = 9999
MAX_DRAW_FORMAT = 999999


def game_id(arm: str, slot: int, draw: int) -> str:
    """The stable identifier of one logical Phase 16 training game."""
    if not re.fullmatch(r"[a-z0-9_]+", str(arm)):
        raise Phase16TrainingError(f"arm id must be lowercase alphanumeric: {arm!r}")
    for name, value, limit in (("slot", slot, MAX_SLOT_FORMAT), ("draw", draw, MAX_DRAW_FORMAT)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise Phase16TrainingError(f"{name} must be an int >= 0, got {value!r}")
        if value > limit:
            raise Phase16TrainingError(f"{name} {value} exceeds the id format limit {limit}")
    return (
        f"{GAME_ID_VERSION}|ms={TRAINING_MASTER_SEED}|arm={arm}"
        f"|slot={int(slot):04d}|draw={int(draw):06d}"
    )


def parse_game_id(identifier: str) -> dict:
    match = _GAME_ID_PATTERN.match(str(identifier))
    if match is None:
        raise Phase16TrainingError(f"not a Phase 16 training game id: {identifier!r}")
    return {
        "arm": match.group("arm"),
        "slot": int(match.group("slot")),
        "draw": int(match.group("draw")),
    }


# ---------------------------------------------------------------------------
# The flags of section 2
# ---------------------------------------------------------------------------

LR_POWER_LAW = "power_law"
LR_CONSTANT = "constant"
LR_SCHEDULES = (LR_POWER_LAW, LR_CONSTANT)

ENTROPY_ANNEALED = "annealed"
ENTROPY_CONSTANT = "constant"
ENTROPY_SCHEDULES = (ENTROPY_ANNEALED, ENTROPY_CONSTANT)

OPPONENTS_PURE_CURRENT = "pure_current"
OPPONENTS_PHASE14_MIXTURE = "phase14_mixture"
OPPONENT_MIXTURES = (OPPONENTS_PURE_CURRENT, OPPONENTS_PHASE14_MIXTURE)

SETUPS_LIBRARY = "library"
SETUPS_EXPANDED = "expanded"
SETUP_MIXTURES = (SETUPS_LIBRARY, SETUPS_EXPANDED)

#: Section 2.1 defaults.
DEFAULT_POPULATION = 96
DEFAULT_WINDOW_DECISIONS = 65_536

#: Measured iteration rate of a `pure_current` arm at the production window on
#: this machine: 16 steady-state smoke windows in 1,104 s = 69.0 s/iteration,
#: so a six-hour arm is ~313 iterations.
MEASURED_ITERATION_SECONDS = 69.0
PLANNED_ITERATIONS = 313

#: Amendment (2026-08-26), agreed with the brief's author: the section 2.3
#: exponents were transcribed from a run of ~43,000 iterations. At `n_ref = 1`
#: the power law floors at n = 9, so arms B/C would spend ~97% of six hours at
#: 1.5e-5 -- five times below the control -- and the shootout would measure a
#: starved learning rate rather than a damped schedule. `n_ref = ceil(0.125*N)`
#: maps the same shape onto this horizon.
LR_HORIZON_FRACTION = 0.125
LR_REFERENCE_ITERATION = 40  # ceil(0.125 * 313)

#: Section 2.3 defaults.
DEFAULT_LR_MAX = 1.5e-4
DEFAULT_LR_MIN = 1.5e-5
DEFAULT_LR_EXPONENT = 1.1
DEFAULT_ENTROPY_START = ENTROPY_COEFFICIENT_START  # 0.005, the accepted Phase 9 level
DEFAULT_ENTROPY_FLOOR = ENTROPY_COEFFICIENT_END  # 0.001, Phase 14's terminal floor
DEFAULT_ENTROPY_EXPONENT = 0.3

#: Phase 14's own two numbers, for the control arm.
PHASE14_CONSTANT_LR = 7.5e-5
PHASE14_CONSTANT_ENTROPY = 0.001

DEFAULT_EMA_DECAY = 0.999

#: Section 2.7: the expanded mixture is half library, half adversarial.
EXPANDED_ADVERSARIAL_WEIGHT = 0.5

#: Phase 14's opponent mixture, as proportions of the population (58/30/12).
PHASE14_MIXTURE_SHARES = {
    "current": 0.58,
    "historical": 0.30,
    "handcrafted": 0.12,
}

#: The frozen inference batch shape and device of section 2's "unchanged" list.
INFERENCE_BATCH_SHAPE = 64
DEFAULT_DEVICE = "mps"
DEFAULT_COLLECTION_DEVICE = "cpu"


@dataclass(frozen=True)
class ArmConfig:
    """Every flag that decides what one shootout arm *is*.

    Frozen and hashable: an arm is compared by digest, so a run that resumes
    under a different flag is caught by the resume identity check rather than
    by reading two logs side by side.
    """

    arm_id: str
    label: str
    lr_schedule: str = LR_POWER_LAW
    lr_max: float = DEFAULT_LR_MAX
    lr_min: float = DEFAULT_LR_MIN
    lr_exponent: float = DEFAULT_LR_EXPONENT
    lr_reference: int = 1
    planned_iterations: int = 0
    lr_constant: float = PHASE14_CONSTANT_LR
    entropy_schedule: str = ENTROPY_ANNEALED
    entropy_start: float = DEFAULT_ENTROPY_START
    entropy_floor: float = DEFAULT_ENTROPY_FLOOR
    entropy_exponent: float = DEFAULT_ENTROPY_EXPONENT
    entropy_constant: float = PHASE14_CONSTANT_ENTROPY
    epochs: int = 1
    ema: bool = True
    ema_decay: float = DEFAULT_EMA_DECAY
    opponents: str = OPPONENTS_PURE_CURRENT
    setups: str = SETUPS_LIBRARY
    population: int = DEFAULT_POPULATION
    window_decisions: int = DEFAULT_WINDOW_DECISIONS
    minibatch_size: int = MINIBATCH_SIZE
    device: str = DEFAULT_DEVICE
    collection_device: str = DEFAULT_COLLECTION_DEVICE
    inference_batch_shape: int = INFERENCE_BATCH_SHAPE
    adversarial_pack: str = "phase16_adversarial_setups_v1"

    def __post_init__(self) -> None:
        if self.lr_schedule not in LR_SCHEDULES:
            raise Phase16TrainingError(f"unknown lr schedule: {self.lr_schedule!r}")
        if self.entropy_schedule not in ENTROPY_SCHEDULES:
            raise Phase16TrainingError(
                f"unknown entropy schedule: {self.entropy_schedule!r}"
            )
        if self.opponents not in OPPONENT_MIXTURES:
            raise Phase16TrainingError(f"unknown opponent mixture: {self.opponents!r}")
        if self.setups not in SETUP_MIXTURES:
            raise Phase16TrainingError(f"unknown setup mixture: {self.setups!r}")
        if self.epochs not in (1, 2):
            raise Phase16TrainingError(
                f"epochs is 1 (paper) or 2 (Phase 14), got {self.epochs!r}"
            )
        if self.population < 1:
            raise Phase16TrainingError("the population must hold at least one game")
        if self.window_decisions < self.minibatch_size:
            raise Phase16TrainingError(
                f"a {self.window_decisions}-decision window is smaller than one "
                f"{self.minibatch_size}-example minibatch"
            )
        if not 0.0 < self.ema_decay < 1.0:
            raise Phase16TrainingError(f"ema decay must be in (0, 1): {self.ema_decay!r}")
        if (
            not isinstance(self.lr_reference, int)
            or isinstance(self.lr_reference, bool)
            or self.lr_reference < 1
        ):
            raise Phase16TrainingError(
                f"the lr reference iteration must be an int >= 1, got "
                f"{self.lr_reference!r}"
            )
        parse_game_id(game_id(self.arm_id, 0, 0))  # the arm id must be id-safe

    def to_dict(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def replace(self, **changes) -> "ArmConfig":
        return replace(self, **changes)


# ---------------------------------------------------------------------------
# The three shootout arms (section 4)
# ---------------------------------------------------------------------------

ARM_A = ArmConfig(
    arm_id="a_control",
    label="control: window collector + Phase 14 hyperparameters",
    lr_schedule=LR_CONSTANT,
    entropy_schedule=ENTROPY_CONSTANT,
    epochs=2,
    ema=False,
    opponents=OPPONENTS_PHASE14_MIXTURE,
    setups=SETUPS_LIBRARY,
)

ARM_B = ArmConfig(
    arm_id="b_damped",
    label="damped: power-law LR, annealed entropy, 1 epoch, EMA, pure self-play",
    lr_schedule=LR_POWER_LAW,
    lr_reference=LR_REFERENCE_ITERATION,
    planned_iterations=PLANNED_ITERATIONS,
    entropy_schedule=ENTROPY_ANNEALED,
    epochs=1,
    ema=True,
    opponents=OPPONENTS_PURE_CURRENT,
    setups=SETUPS_LIBRARY,
)

ARM_C = ARM_B.replace(
    arm_id="c_damped_plus",
    label="damped+: arm B plus the expanded (adversarial) setup mixture",
    setups=SETUPS_EXPANDED,
)

SHOOTOUT_ARMS = (ARM_A, ARM_B, ARM_C)
ARMS_BY_ID = {arm.arm_id: arm for arm in SHOOTOUT_ARMS}


def arm(arm_id: str) -> ArmConfig:
    if arm_id not in ARMS_BY_ID:
        raise Phase16TrainingError(
            f"unknown shootout arm {arm_id!r}; expected one of {sorted(ARMS_BY_ID)}"
        )
    return ARMS_BY_ID[arm_id]


# ---------------------------------------------------------------------------
# Predeclared decision rules (section 5)
# ---------------------------------------------------------------------------

#: `max(B, C)` final-hour benchmark EWR must clear A by this margin.
ADOPT_RECIPE_MARGIN = 0.03
#: C's adversarial-stratum EWR must clear B's by this margin.
SETUPS_CAUSAL_MARGIN = 0.03
#: Evaluation hours of section 4.
EVALUATION_HOURS = (0, 2, 4, 6)
#: No run in this instruction exceeds six hours.
ARM_HOURS = 6.0

DECISION_RULES = {
    "adopt_recipe": (
        f"max(B, C) final-hour benchmark EWR >= A + {ADOPT_RECIPE_MARGIN}"
    ),
    "setups_causal": (
        f"C adversarial-stratum EWR >= B + {SETUPS_CAUSAL_MARGIN}"
    ),
    "plateau_check": (
        "report each arm's h4->h6 slope; a flat B/C with a passing h6 still "
        "adopts, but the report must say the plateau moved, not vanished"
    ),
    "stop_rule": (
        "if neither B nor C clears adopt_recipe: STOP, write the report, hand "
        "back to the operator; no long run is authorized"
    ),
}

#: Correctness gates that must pass before any 6-hour run (section 3).
CORRECTNESS_GATES = (
    "smoke_run",
    "window_edge_invariant",
    "collection_throughput",
    "full_pytest",
)


def inherited_phase9_values() -> dict:
    """Every objective constant Phase 16 consumes unchanged, named in one place."""
    return {
        "gamma": GAMMA,
        "lambda_A": LAMBDA_ADVANTAGE,
        "lambda_V": LAMBDA_VALUE,
        "ppo_clip_epsilon": PPO_CLIP_EPSILON,
        "advantage_filter_quantile": ADVANTAGE_FILTER_QUANTILE,
        "advantage_filter_floor": ADVANTAGE_FILTER_FLOOR,
        "advantage_standardization_epsilon": ADVANTAGE_STANDARDIZATION_EPSILON,
        "value_loss_weight": VALUE_LOSS_WEIGHT,
        "belief_loss_weight": BELIEF_LOSS_WEIGHT,
        "behavior_kl_target": BEHAVIOR_KL_TARGET,
        "kl_beta_min": KL_BETA_MIN,
        "kl_beta_max": KL_BETA_MAX,
        "kl_hard_limit": KL_HARD_LIMIT,
        "clip_fraction_hard_limit": CLIP_FRACTION_HARD_LIMIT,
        "minibatch_size": MINIBATCH_SIZE,
        "source": "stratego.training.phase9_contract, imported unmodified",
    }


def contract_document() -> dict:
    """The deterministic identity of the Phase 16 training loop."""
    return {
        "artifact": PHASE16_CONTRACT_VERSION,
        "phase": 16,
        "agent": 3,
        "training_version": PHASE16_TRAINING_VERSION,
        "namespace": PHASE16_NAMESPACE,
        "modules": {
            "collector": PHASE16_COLLECTOR_VERSION,
            "targets": PHASE16_TARGETS_VERSION,
            "trainer": PHASE16_TRAINER_VERSION,
            "checkpoint": PHASE16_CHECKPOINT_VERSION,
            "runner": PHASE16_RUNNER_VERSION,
        },
        "starting_checkpoint": {
            "path": STARTING_CHECKPOINT,
            "sha256": STARTING_CHECKPOINT_SHA256,
            "model_state_digest": STARTING_MODEL_STATE_DIGEST,
            "optimizer": "fresh AdamW moments for every arm",
        },
        "objective": inherited_phase9_values(),
        "collection": {
            "unit": "one window = a fixed budget of learner decisions",
            "population": DEFAULT_POPULATION,
            "window_decisions": DEFAULT_WINDOW_DECISIONS,
            "replacement": "a game that ends mid-window is replaced by a fresh draw",
            "search": "absent; no module under stratego.search is imported",
        },
        "schedule_amendment": {
            "amended_utc": "2026-08-26",
            "section": "03_AGENT_3_TRAINING_LOOP_V2.md section 2.3",
            "origin": "raised by this agent, confirmed by the brief's author",
            "defect": (
                "the exponents were transcribed from a ~43,000-iteration run; at "
                "n_ref = 1 the power law floors at n = 9, so a six-hour arm at this "
                "machine's ~313 iterations would spend ~97% of itself at 1.5e-5 -- "
                "five times below the control -- and the shootout would measure a "
                "starved learning rate rather than a damped schedule"
            ),
            "change": (
                f"lr(n) = clamp(lr_max * (n/n_ref)**-1.1, lr_min, lr_max) with "
                f"n_ref = ceil({LR_HORIZON_FRACTION} * N) = {LR_REFERENCE_ITERATION} "
                f"for N = {PLANNED_ITERATIONS}"
            ),
            "unchanged": (
                "lr_max, lr_min, the exponent, the entropy anneal, and arm A in "
                "its entirety"
            ),
            "entropy_not_re_horizoned": (
                "0.005 * n**-0.3 reaches the 0.001 terminal floor at n = 213, ~68% "
                "of a 313-iteration run: already a smooth decay across most of the "
                "run followed by the terminal value, which is what section 2.3 "
                "asks for. Re-horizoning it to n_ref_H = 13 as first proposed would "
                "start it at 0.0108 -- above the accepted Phase 9 level the section "
                "restores -- and end at 0.0018, never reaching the floor."
            ),
            "measured_iteration_seconds": MEASURED_ITERATION_SECONDS,
            "planned_iterations": PLANNED_ITERATIONS,
        },
        "window_edge_targets": {
            "advantage": (
                "TD(lambda_A) over stored values within the window; the tail is "
                "bootstrapped from v at the boundary and A beyond the boundary is 0"
            ),
            "wdl": (
                "lambda_V blending toward the final outcome only once the game "
                "finishes; buffered per game until then"
            ),
            "invariant": (
                "for a finished game, windowed targets equal whole-game targets "
                "to float32 tolerance"
            ),
        },
        "flags": {
            "lr_schedule": list(LR_SCHEDULES),
            "entropy_schedule": list(ENTROPY_SCHEDULES),
            "epochs": [1, 2],
            "ema": [True, False],
            "opponents": list(OPPONENT_MIXTURES),
            "setups": list(SETUP_MIXTURES),
        },
        "arms": {arm.arm_id: arm.to_dict() for arm in SHOOTOUT_ARMS},
        "decision_rules": dict(DECISION_RULES),
        "correctness_gates": list(CORRECTNESS_GATES),
        "seed_namespace": SEED_NAMESPACE,
        "seed_master": TRAINING_MASTER_SEED,
        "seed_domains": list(STREAM_DOMAINS),
        "non_goals": [
            "no reward shaping; training reward stays pure W/D/L",
            "no search in the training loop",
            "no marginal belief-head scaling",
            "no deeper search rungs",
        ],
    }


def contract_digest() -> str:
    return hashlib.sha256(
        json.dumps(contract_document(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "ARMS_BY_ID",
    "ARM_A",
    "ARM_B",
    "ARM_C",
    "ARM_HOURS",
    "ADOPT_RECIPE_MARGIN",
    "ArmConfig",
    "CORRECTNESS_GATES",
    "DECISION_RULES",
    "DEFAULT_EMA_DECAY",
    "DEFAULT_ENTROPY_EXPONENT",
    "DEFAULT_ENTROPY_FLOOR",
    "DEFAULT_ENTROPY_START",
    "DEFAULT_LR_EXPONENT",
    "DEFAULT_LR_MAX",
    "DEFAULT_LR_MIN",
    "DEFAULT_POPULATION",
    "DEFAULT_WINDOW_DECISIONS",
    "DOMAIN_ACTION_SAMPLING",
    "DOMAIN_GAME_DRAW",
    "DOMAIN_OPPONENT_DRAW",
    "DOMAIN_POLICY",
    "DOMAIN_SETUP_SIDE",
    "DOMAIN_TRAINING_ORDER",
    "ENTROPY_ANNEALED",
    "ENTROPY_CONSTANT",
    "EVALUATION_HOURS",
    "EXPANDED_ADVERSARIAL_WEIGHT",
    "GAME_ID_VERSION",
    "INFERENCE_BATCH_SHAPE",
    "LR_CONSTANT",
    "LR_HORIZON_FRACTION",
    "LR_POWER_LAW",
    "LR_REFERENCE_ITERATION",
    "MEASURED_ITERATION_SECONDS",
    "PLANNED_ITERATIONS",
    "OPPONENTS_PHASE14_MIXTURE",
    "OPPONENTS_PURE_CURRENT",
    "PHASE14_CONSTANT_ENTROPY",
    "PHASE14_CONSTANT_LR",
    "PHASE14_MIXTURE_SHARES",
    "PHASE16_CHECKPOINT_VERSION",
    "PHASE16_COLLECTOR_VERSION",
    "PHASE16_CONTRACT_VERSION",
    "PHASE16_NAMESPACE",
    "PHASE16_RUNNER_VERSION",
    "PHASE16_TARGETS_VERSION",
    "PHASE16_TRAINER_VERSION",
    "PHASE16_TRAINING_VERSION",
    "Phase16TrainingError",
    "SEED_NAMESPACE",
    "SETUPS_CAUSAL_MARGIN",
    "SETUPS_EXPANDED",
    "SETUPS_LIBRARY",
    "SHOOTOUT_ARMS",
    "STARTING_CHECKPOINT",
    "STARTING_CHECKPOINT_SHA256",
    "STARTING_MODEL_STATE_DIGEST",
    "STREAM_DOMAINS",
    "TRAINING_MASTER_SEED",
    "arm",
    "contract_digest",
    "contract_document",
    "derive_train_seed",
    "game_id",
    "inherited_phase9_values",
    "parse_game_id",
    "uniform_from_seed",
]
