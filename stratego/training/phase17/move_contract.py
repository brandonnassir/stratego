"""Phase 17 Agent 2: the move half's frozen constants, identities and refusals.

Specification sources:

- `00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md` sections 4, 5, 6, 9, 10, 12, 13
- `02_AGENT_2_FIXED_TRANSITION_MOVE_TRAINING.md`
- Agent 1's verified `reports/phase17/phase17_contract_handoff_v1.json`, whose
  `schedules_and_controllers.move` block is FINAL

What lives here
---------------
Constants, versions, seed derivation, the schedule horizon, the transition
schema's enumerations, and the refusals that make "100% current policy" a
structural property rather than a claim. No model, no optimizer, no engine.

Where a constant already exists in an accepted phase it is *imported*, never
restated: `PPO_CLIP_EPSILON`, the KL controller's target/bounds/factors, the
advantage filter, `VALUE_LOSS_WEIGHT` and the minibatch size all come from
`phase9_contract`, so a tuned value would have to be tuned where the accepted
Phase 9 contract digest sees it. This module restates exactly three numbers
the accepted phases do not contain -- the Phase 17 belief weight, the move
LR band, and the entropy band -- and every one of them is quoted from the
Agent 1 handoff.

The horizon
-----------
`N` (the 12-hour iteration count) is **not** frozen here. Agent 4 measures it
in the preflight rehearsal and freezes `N`, `n_ref` and the whole curve before
launch. :class:`MoveScheduleHorizon` is the object that freezing produces; it
refuses to be built from a running process's changing speed by requiring `N`
explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

from ..phase9_contract import (
    ADVANTAGE_FILTER_FLOOR,
    ADVANTAGE_FILTER_QUANTILE,
    ADVANTAGE_STANDARDIZATION_EPSILON,
    BEHAVIOR_KL_TARGET,
    CLIP_FRACTION_HARD_LIMIT,
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
    PPO_CLIP_EPSILON,
    VALUE_LOSS_WEIGHT,
)
from ..phase16.schedules import annealed_entropy, power_law_learning_rate

# ---------------------------------------------------------------------------
# Versions and identities
# ---------------------------------------------------------------------------

PHASE17_MOVE_CONTRACT_VERSION = "phase17_move_contract_v1"
MOVE_TRANSITION_VERSION = "phase17_move_transition_v1"
MOVE_TARGETS_VERSION = "phase17_move_targets_v1"
MOVE_COLLECTOR_VERSION = "phase17_move_collector_v1"
MOVE_TRAINER_VERSION = "phase17_move_trainer_v1"
MOVE_LOSS_VERSION = "phase17_move_loss_v1"
MOVE_KL_CONTROLLER_NAME = "phase17_move_behavior_kl_controller_v1"

WORK_PACKAGE = "phase17"

#: The provisional production run id (common contract section 3). A concrete
#: execution binds its own; nothing in this module assumes this one.
DEFAULT_RUN_ID = "RUN-2026-A"

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-_]{0,63}$")

#: Common contract section 4: the *only* accepted move-policy start.
START_CHECKPOINT_PATH = "checkpoints/phase9/selfplay_c1_v1.pt"
START_FILE_SHA256 = "dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea"
START_MODEL_STATE_DIGEST = (
    "f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd"
)
START_PARAMETER_COUNT = 863959
START_CANDIDATE_ID = "C1"
START_LINEAGE = "behavior_B041.pt, post-iteration 40"

#: The *other* function named `state_dict_digest` yields this on the same
#: bytes. Recorded so a future reader cannot mistake a mismatch for corruption.
START_CONTAINER_STATE_DIGEST = (
    "f0994cf0eb985848016b79c7db376f7d9499a7f7352b430b5ac2d39aece9869e"
)

RULES_VERSION = "stratego_project_v1"
OBSERVATION_VERSION = "observation_v2_1_127ch"
ACTION_ENCODING_VERSION = "source_destination_10000_v1"
MODEL_CONTRACT_VERSION = "model_contract_v2"
ENGINE_ACTION_FRAME = "absolute_engine_squares"
POLICY_ACTION_FRAME = "perspective_normalized_squares"

#: Both seats are the same current raw policy, so both sides' decisions are
#: stored under one policy token. The token names the *seat role*; the weights
#: are named by `behavior_model_state_digest`, which moves every iteration.
#: They are separate fields precisely because a rebind changes one and not the
#: other -- see `move_snapshot.CurrentMovePolicy`.
CURRENT_POLICY_IDENTITY = "P17RAW"
CURRENT_POLICY_TOKEN = "phase17_current_raw_move_v1|P17RAW"

BEHAVIOR_TEMPERATURE = 1.0

#: Section 6 default: exactly this many learner transitions per iteration.
WINDOW_TRANSITIONS = 65536

#: float32 sum-to-one slack on a stored behavior distribution.
BEHAVIOR_PROBABILITY_ABS_TOLERANCE = 1e-4

#: float32 round-trip slack for the reduction invariant `G-M4a`.
TARGET_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Enumerations the transition schema uses
# ---------------------------------------------------------------------------

BOUNDARY_INTERIOR = "interior"
BOUNDARY_WINDOW = "window_boundary"
BOUNDARY_TERMINAL = "terminal"
BOUNDARY_STATUSES = (BOUNDARY_INTERIOR, BOUNDARY_WINDOW, BOUNDARY_TERMINAL)

PROVENANCE_TERMINAL = "terminal_z"
PROVENANCE_BOOTSTRAP = "boundary_bootstrap"
TARGET_PROVENANCES = (PROVENANCE_TERMINAL, PROVENANCE_BOOTSTRAP)


# ---------------------------------------------------------------------------
# The move objective and its schedules
# ---------------------------------------------------------------------------

#: Common contract section 4: the Phase 9 marginal belief auxiliary loss is
#: disabled. The head stays for checkpoint compatibility and receives no loss.
#: The accepted Phase 9 weight is 0.25 and is deliberately NOT imported here --
#: overriding it silently by importing it would hide the divergence.
BELIEF_LOSS_WEIGHT = 0.0
PHASE9_ACCEPTED_BELIEF_LOSS_WEIGHT = 0.25

MOVE_EPOCHS_PER_ITERATION = 1
MOVE_EMA_DECAY = 0.999
MOVE_GRADIENT_CLIP_NORM = 1.0
MOVE_INITIAL_KL_BETA = 0.005

MOVE_LR_MAX = 1.5e-4
MOVE_LR_MIN = 1.5e-5
MOVE_LR_EXPONENT = 1.1
MOVE_LR_REFERENCE_FRACTION = 0.125

MOVE_ENTROPY_START = 0.005
MOVE_ENTROPY_FLOOR = 0.001
MOVE_ENTROPY_EXPONENT = 0.3

MOVE_OPTIMIZER = {
    "name": "AdamW",
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.01,
    "gradient_clip_norm": MOVE_GRADIENT_CLIP_NORM,
    "precision": "float32",
}


class Phase17MoveError(RuntimeError):
    """A Phase 17 move-side object was configured or used outside its contract."""


def _require_positive_int(value, *, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise Phase17MoveError(f"{what} must be an int >= 1, got {value!r}")
    return int(value)


def reference_iteration(total_iterations: int) -> int:
    """`n_ref = ceil(0.125 * N)`, the frozen re-horizoning of the move LR.

    The constant exists because the paper's exponents were transcribed from a
    ~42,000-iteration run. Applied unshifted to a run of a few hundred
    iterations the power law reaches its floor at n = 9, and the run would
    measure a starved learning rate rather than a decayed one.
    """
    total = _require_positive_int(total_iterations, what="the iteration horizon N")
    return max(1, int(math.ceil(MOVE_LR_REFERENCE_FRACTION * total)))


@dataclass(frozen=True)
class MoveScheduleHorizon:
    """The frozen move LR / entropy curve of one run.

    Built once, from a horizon Agent 4 *measured*; every schedule value is a
    pure function of the 1-based completed iteration `n`, so a resumed run
    recomputes exactly what an uninterrupted one would from its restored
    iteration number alone.
    """

    total_iterations: int
    lr_max: float = MOVE_LR_MAX
    lr_min: float = MOVE_LR_MIN
    lr_exponent: float = MOVE_LR_EXPONENT
    entropy_start: float = MOVE_ENTROPY_START
    entropy_floor: float = MOVE_ENTROPY_FLOOR
    entropy_exponent: float = MOVE_ENTROPY_EXPONENT

    def __post_init__(self) -> None:
        _require_positive_int(self.total_iterations, what="the iteration horizon N")
        if not self.lr_min <= self.lr_max:
            raise Phase17MoveError(
                f"lr_min {self.lr_min} exceeds lr_max {self.lr_max}"
            )
        if not 0.0 <= self.entropy_floor <= self.entropy_start:
            raise Phase17MoveError(
                f"the entropy floor {self.entropy_floor} is not in "
                f"[0, {self.entropy_start}]"
            )

    @property
    def reference_iteration(self) -> int:
        return reference_iteration(self.total_iterations)

    def learning_rate(self, iteration: int) -> float:
        """`clamp(lr_max * (n/n_ref)**-1.1, lr_min, lr_max)`."""
        return power_law_learning_rate(
            _require_positive_int(iteration, what="the schedule index"),
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            exponent=self.lr_exponent,
            reference=self.reference_iteration,
        )

    def entropy_coefficient(self, iteration: int) -> float:
        """`max(0.001, 0.005 * n**-0.3)`. An entropy BONUS, not a KL."""
        return annealed_entropy(
            _require_positive_int(iteration, what="the schedule index"),
            start=self.entropy_start,
            floor=self.entropy_floor,
            exponent=self.entropy_exponent,
        )

    def row(self, iteration: int) -> dict:
        return {
            "iteration": int(iteration),
            "learning_rate": self.learning_rate(iteration),
            "entropy_coefficient": self.entropy_coefficient(iteration),
        }

    def curve(self) -> list:
        """Every scheduled row of the frozen horizon, for the run config."""
        return [self.row(n) for n in range(1, self.total_iterations + 1)]

    def to_dict(self) -> dict:
        return {
            "total_iterations": int(self.total_iterations),
            "reference_iteration": self.reference_iteration,
            "learning_rate_formula": (
                f"clamp({self.lr_max} * (n/{self.reference_iteration})"
                f"**-{self.lr_exponent}, {self.lr_min}, {self.lr_max})"
            ),
            "entropy_formula": (
                f"max({self.entropy_floor}, {self.entropy_start} * "
                f"n**-{self.entropy_exponent})"
            ),
            "entropy_is": "an entropy bonus; the paper has no move entropy bonus",
            "first": self.row(1),
            "last": self.row(self.total_iterations),
        }


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

MOVE_IDENTITY_VERSION = "phase17_move_identity_v1"

#: Phase 17's own master seed. A Phase 17 game must not replay the actions of
#: a Phase 16 game that happens to share a slot and draw number, so the roots
#: below are disjoint from `phase16.contract.DOMAIN_ROOTS` by construction.
MOVE_MASTER_SEED = 2026082710

PHASE17_MOVE_PERSON = b"strat17m"

DOMAIN_GAME_DRAW = "game_draw"
DOMAIN_SETUP_DRAW = "setup_draw"
DOMAIN_ACTION_SAMPLING = "action_sampling"
DOMAIN_TRAINING_ORDER = "training_order"

MOVE_STREAM_DOMAINS = (
    DOMAIN_GAME_DRAW,
    DOMAIN_SETUP_DRAW,
    DOMAIN_ACTION_SAMPLING,
    DOMAIN_TRAINING_ORDER,
)

MOVE_DOMAIN_ROOTS = {
    DOMAIN_GAME_DRAW: MOVE_MASTER_SEED + 1,
    DOMAIN_SETUP_DRAW: MOVE_MASTER_SEED + 2,
    DOMAIN_ACTION_SAMPLING: MOVE_MASTER_SEED + 3,
    DOMAIN_TRAINING_ORDER: MOVE_MASTER_SEED + 4,
}


def derive_move_seed(domain: str, *parts: "int | str") -> int:
    """A 63-bit deterministic seed for one Phase 17 move stream."""
    if domain not in MOVE_STREAM_DOMAINS:
        raise Phase17MoveError(f"unknown Phase 17 move domain: {domain!r}")
    for part in parts:
        if not isinstance(part, (int, str)) or isinstance(part, bool):
            raise Phase17MoveError(
                f"stream identity parts must be int or str, got {type(part).__name__}"
            )
        if isinstance(part, str) and ":" in part:
            raise Phase17MoveError(
                f"string identity parts may not contain ':' (got {part!r})"
            )
    payload = ":".join(
        [
            MOVE_IDENTITY_VERSION,
            domain,
            str(MOVE_DOMAIN_ROOTS[domain]),
            *[str(part) for part in parts],
        ]
    )
    digest = hashlib.blake2b(
        payload.encode(), digest_size=8, person=PHASE17_MOVE_PERSON
    ).digest()
    return int.from_bytes(digest, "big") >> 1


def uniform_from_seed(seed: int) -> float:
    """A float in [0, 1) from a derived seed, by the accepted 53-bit rule."""
    return ((int(seed) >> 10) & ((1 << 53) - 1)) / float(1 << 53)


# ---------------------------------------------------------------------------
# Logical game identity
# ---------------------------------------------------------------------------

GAME_ID_VERSION = "phase17_game_v1"

_GAME_ID_PATTERN = re.compile(
    rf"^{re.escape(GAME_ID_VERSION)}\|run=(?P<run>[A-Za-z0-9\-_]+)"
    r"\|slot=(?P<slot>[0-9]{4})\|draw=(?P<draw>[0-9]{6})$"
)

MAX_SLOT_FORMAT = 9999
MAX_DRAW_FORMAT = 999999


def require_run_id(run_id: str) -> str:
    """A run id is immutable and must not contain a separator character."""
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise Phase17MoveError(
            f"a Phase 17 run id must match {_RUN_ID_PATTERN.pattern!r}, got {run_id!r}"
        )
    return run_id


def game_id(run_id: str, slot: int, draw: int) -> str:
    """The stable identifier of one logical Phase 17 training game.

    A persistent population has no iteration to key on: a game outlives the
    window it started in. What *is* stable is "the n-th game played in this
    population slot", which is what makes a replacement draw reproducible from
    the run state alone.
    """
    run = require_run_id(run_id)
    if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot <= MAX_SLOT_FORMAT:
        raise Phase17MoveError(f"slot must be an int in 0..{MAX_SLOT_FORMAT}, got {slot!r}")
    if not isinstance(draw, int) or isinstance(draw, bool) or not 0 <= draw <= MAX_DRAW_FORMAT:
        raise Phase17MoveError(f"draw must be an int in 0..{MAX_DRAW_FORMAT}, got {draw!r}")
    return f"{GAME_ID_VERSION}|run={run}|slot={slot:04d}|draw={draw:06d}"


def parse_game_id(identifier: str) -> dict:
    """`(run_id, slot, draw)` back out of a Phase 17 game id."""
    match = _GAME_ID_PATTERN.fullmatch(str(identifier))
    if match is None:
        raise Phase17MoveError(f"not a Phase 17 game id: {identifier!r}")
    return {
        "run_id": match.group("run"),
        "slot": int(match.group("slot")),
        "draw": int(match.group("draw")),
    }


# ---------------------------------------------------------------------------
# The refusals that make "100% current policy" structural
# ---------------------------------------------------------------------------

#: Names a Phase 17 *training* configuration may never contain. Every one of
#: these is an evaluation instrument (common contract section 5); naming one in
#: collection is immediate stop condition I5.
FORBIDDEN_TRAINING_PARTICIPANTS = (
    "historical",
    "archive",
    "anchor",
    "phase9_anchor",
    "handcrafted",
    "rule",
    "rule_based",
    "strategic_rule_based",
    "tactical_rule_based",
    "stress",
    "search",
    "belief_search",
    "p18",
    "p24",
    "oracle",
    "opponent_pool",
    "mixture",
)


def assert_current_policy_only(participants) -> dict:
    """Refuse any training population that names a non-current participant.

    `participants` is anything iterable of names, or a mapping whose keys and
    string values are names. The check is on *names* deliberately: it runs
    before a single object is constructed, so a configuration error is refused
    at parse time rather than after an hour of collection.
    """
    if participants is None:
        names: list = []
    elif isinstance(participants, dict):
        names = [str(key) for key in participants]
        names += [str(value) for value in participants.values() if isinstance(value, str)]
    elif isinstance(participants, str):
        names = [participants]
    else:
        names = [str(entry) for entry in participants]

    offending = []
    for name in names:
        lowered = name.lower()
        for token in FORBIDDEN_TRAINING_PARTICIPANTS:
            if re.search(rf"(?:^|[^a-z0-9]){re.escape(token)}(?:[^a-z0-9]|$)", lowered):
                offending.append({"name": name, "matched": token})
                break
    if offending:
        raise Phase17MoveError(
            "Phase 17 training is 100% current-policy self-play; these named "
            f"participants are evaluation instruments only: {offending}"
        )
    return {"checked": len(names), "offending": 0, "names": list(names)}


# ---------------------------------------------------------------------------
# The serializable contract
# ---------------------------------------------------------------------------


def move_contract_document() -> dict:
    """Everything the move half is frozen to, in one serializable mapping."""
    return {
        "contract_version": PHASE17_MOVE_CONTRACT_VERSION,
        "work_package": WORK_PACKAGE,
        "versions": {
            "transition": MOVE_TRANSITION_VERSION,
            "targets": MOVE_TARGETS_VERSION,
            "collector": MOVE_COLLECTOR_VERSION,
            "trainer": MOVE_TRAINER_VERSION,
            "loss": MOVE_LOSS_VERSION,
            "kl_controller": MOVE_KL_CONTROLLER_NAME,
            "game_id": GAME_ID_VERSION,
            "identity": MOVE_IDENTITY_VERSION,
        },
        "start_identity": {
            "path": START_CHECKPOINT_PATH,
            "file_sha256": START_FILE_SHA256,
            "model_state_digest": START_MODEL_STATE_DIGEST,
            "digest_function": "stratego.training.phase9_behavior.state_dict_digest",
            "not_the_digest_function": (
                "stratego.model.checkpoint.state_dict_digest, which yields "
                f"{START_CONTAINER_STATE_DIGEST} on the same bytes"
            ),
            "parameter_count": START_PARAMETER_COUNT,
            "candidate_id": START_CANDIDATE_ID,
            "lineage": START_LINEAGE,
            "semantics": "weights-only warm start; a new lineage, not a resume",
        },
        "contract_versions": {
            "rules_version": RULES_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "action_encoding_version": ACTION_ENCODING_VERSION,
            "model_contract_version": MODEL_CONTRACT_VERSION,
            "engine_action_frame": ENGINE_ACTION_FRAME,
            "policy_action_frame": POLICY_ACTION_FRAME,
        },
        "population": {
            "red": "the current raw move snapshot",
            "blue": "the same current raw move snapshot",
            "policy_token": CURRENT_POLICY_TOKEN,
            "search": "prohibited from collection and training",
            "historical_and_rule_and_stress": "evaluation instruments only",
            "forbidden_names": list(FORBIDDEN_TRAINING_PARTICIPANTS),
            "action_selection": "categorical sample over the legal set; argmax prohibited",
            "behavior_temperature": BEHAVIOR_TEMPERATURE,
        },
        "window": {
            "transitions": WINDOW_TRANSITIONS,
            "meaning": (
                "a HARVEST budget of learner transitions; transitions_trained "
                "is the smaller post-advantage-filter count and both are logged"
            ),
            "partial_emission": True,
            "waits_for_terminal_outcomes": False,
        },
        "targets": {
            "gamma": GAMMA,
            "lambda_advantage": LAMBDA_ADVANTAGE,
            "lambda_value": LAMBDA_VALUE,
            "governing_invariant": "G-M4a (reduction); G-M4b retired by operator decision D2",
            "boundary_statuses": list(BOUNDARY_STATUSES),
            "target_provenances": list(TARGET_PROVENANCES),
            "tolerance": TARGET_TOLERANCE,
        },
        "objective": {
            "ppo_clip_epsilon": PPO_CLIP_EPSILON,
            "value_loss_weight": VALUE_LOSS_WEIGHT,
            "belief_loss_weight": BELIEF_LOSS_WEIGHT,
            "phase9_accepted_belief_loss_weight": PHASE9_ACCEPTED_BELIEF_LOSS_WEIGHT,
            "advantage_filter_quantile": ADVANTAGE_FILTER_QUANTILE,
            "advantage_filter_floor": ADVANTAGE_FILTER_FLOOR,
            "advantage_standardization_epsilon": ADVANTAGE_STANDARDIZATION_EPSILON,
            "minibatch_size": MINIBATCH_SIZE,
            "epochs_per_iteration": MOVE_EPOCHS_PER_ITERATION,
            "ema_decay": MOVE_EMA_DECAY,
            "optimizer": {
                key: (list(value) if isinstance(value, tuple) else value)
                for key, value in MOVE_OPTIMIZER.items()
            },
        },
        "kl_controller": {
            "name": MOVE_KL_CONTROLLER_NAME,
            "inherits": "the accepted Phase 9 controller, unchanged",
            "direction": "FORWARD: D_KL(pi_behavior || pi_current) over the legal set",
            "target": BEHAVIOR_KL_TARGET,
            "beta_initial": MOVE_INITIAL_KL_BETA,
            "beta_bounds": [KL_BETA_MIN, KL_BETA_MAX],
            "increase_threshold": KL_BETA_INCREASE_THRESHOLD,
            "increase_factor": KL_BETA_INCREASE_FACTOR,
            "decrease_threshold": KL_BETA_DECREASE_THRESHOLD,
            "decrease_factor": KL_BETA_DECREASE_FACTOR,
            "hard_mean_kl_limit": KL_HARD_LIMIT,
            "clip_fraction_hard_limit": CLIP_FRACTION_HARD_LIMIT,
            "not_the_papers": (
                "the paper uses a fixed 0.1 coefficient on the REVERSE KL; "
                "named, never conflated"
            ),
        },
        "schedules": {
            "index": "the 1-based completed iteration; never the optimizer step",
            "learning_rate": {
                "lr_max": MOVE_LR_MAX,
                "lr_min": MOVE_LR_MIN,
                "exponent": MOVE_LR_EXPONENT,
                "reference_fraction": MOVE_LR_REFERENCE_FRACTION,
            },
            "entropy": {
                "start": MOVE_ENTROPY_START,
                "floor": MOVE_ENTROPY_FLOOR,
                "exponent": MOVE_ENTROPY_EXPONENT,
                "term": "an entropy bonus subtracted from the loss",
            },
            "horizon_N": "MEASURED by Agent 4's preflight and frozen before hour 0",
        },
        "seeds": {
            "master": MOVE_MASTER_SEED,
            "domains": list(MOVE_STREAM_DOMAINS),
            "roots": dict(MOVE_DOMAIN_ROOTS),
        },
    }


def contract_digest() -> str:
    """`sha256` over the canonical JSON of :func:`move_contract_document`."""
    payload = json.dumps(
        move_contract_document(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ACTION_ENCODING_VERSION",
    "BEHAVIOR_PROBABILITY_ABS_TOLERANCE",
    "BEHAVIOR_TEMPERATURE",
    "BELIEF_LOSS_WEIGHT",
    "BOUNDARY_INTERIOR",
    "BOUNDARY_STATUSES",
    "BOUNDARY_TERMINAL",
    "BOUNDARY_WINDOW",
    "CURRENT_POLICY_IDENTITY",
    "CURRENT_POLICY_TOKEN",
    "DEFAULT_RUN_ID",
    "DOMAIN_ACTION_SAMPLING",
    "DOMAIN_GAME_DRAW",
    "DOMAIN_SETUP_DRAW",
    "DOMAIN_TRAINING_ORDER",
    "FORBIDDEN_TRAINING_PARTICIPANTS",
    "GAME_ID_VERSION",
    "MODEL_CONTRACT_VERSION",
    "MOVE_COLLECTOR_VERSION",
    "MOVE_EMA_DECAY",
    "MOVE_EPOCHS_PER_ITERATION",
    "MOVE_GRADIENT_CLIP_NORM",
    "MOVE_INITIAL_KL_BETA",
    "MOVE_KL_CONTROLLER_NAME",
    "MOVE_LOSS_VERSION",
    "MOVE_OPTIMIZER",
    "MOVE_TARGETS_VERSION",
    "MOVE_TRAINER_VERSION",
    "MOVE_TRANSITION_VERSION",
    "MoveScheduleHorizon",
    "OBSERVATION_VERSION",
    "PHASE17_MOVE_CONTRACT_VERSION",
    "PHASE9_ACCEPTED_BELIEF_LOSS_WEIGHT",
    "PROVENANCE_BOOTSTRAP",
    "PROVENANCE_TERMINAL",
    "Phase17MoveError",
    "RULES_VERSION",
    "START_CHECKPOINT_PATH",
    "START_FILE_SHA256",
    "START_MODEL_STATE_DIGEST",
    "START_PARAMETER_COUNT",
    "TARGET_PROVENANCES",
    "TARGET_TOLERANCE",
    "WINDOW_TRANSITIONS",
    "WORK_PACKAGE",
    "assert_current_policy_only",
    "contract_digest",
    "derive_move_seed",
    "game_id",
    "move_contract_document",
    "parse_game_id",
    "reference_iteration",
    "require_run_id",
    "uniform_from_seed",
]
