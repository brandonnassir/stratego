"""Phase 17 Agent 2: the `phase17_move_transition_v1` row.

Schema source: Agent 1's `phase17_contract_handoff_v1.schemas.move_transition`,
transcribed field for field. One row is **one learner decision by one seat at
one ply**, and because both seats are learners in Phase 17 every legal model
decision is a learner transition.

Two fields carry the whole current-policy claim
-----------------------------------------------
`behavior_model_state_digest` names the raw weights that produced
`behavior_probabilities`. `policy_age_iterations` says how old they were when
the row was trained. Together they make "was this decision taken under the
current policy?" a question the stored data answers, rather than one the
collector asserts about itself.

`behavior_probabilities` is stored, never recomputed
----------------------------------------------------
It is the PPO ratio's denominator. Recomputing it at training time from the
current weights would silence exactly the bug the in-flight rebind fix exists
to expose -- a stale-weights decision would produce a ratio of 1.0 and look
perfectly healthy.

Telemetry, not gates
--------------------
`boundary_target_divergence` and `bootstrap_age_windows` are recorded because
operator decision D2 retired `G-M4b`: a bootstrapped target *is* different from
the whole-game one, by construction, so the difference is measured and reported
rather than asserted away. Neither field may become a gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .move_contract import (
    ACTION_ENCODING_VERSION,
    BEHAVIOR_PROBABILITY_ABS_TOLERANCE,
    BEHAVIOR_TEMPERATURE,
    BOUNDARY_STATUSES,
    MOVE_TRANSITION_VERSION,
    OBSERVATION_VERSION,
    RULES_VERSION,
    TARGET_PROVENANCES,
    WORK_PACKAGE,
    Phase17MoveError,
)


class Phase17TransitionError(Phase17MoveError):
    """A move transition is outside `phase17_move_transition_v1`."""


@dataclass
class MoveTransition:
    """One learner decision, complete except for the targets the window fills.

    Field names match `Phase9RLExample` wherever the objective reads them, so
    the Phase 17 collation is the accepted collation with the fields this
    schema adds -- not a second example schema the objective has to learn.
    """

    # -- identity ----------------------------------------------------------
    run_id: str
    iteration: int
    window_index: int
    game_id: str
    ply: int
    color: int
    perspective_player: int

    # -- model input and the decision --------------------------------------
    observation: np.ndarray = field(repr=False)
    legal_mask: np.ndarray = field(repr=False)
    legal_actions: tuple = ()
    behavior_probabilities: tuple = ()
    sampled_action: int = -1
    sampled_action_index: int = -1
    sampled_action_model: int = -1
    behavior_action_probability: float = 0.0
    behavior_action_logprob: float = 0.0
    action_seed: int = 0
    behavior_model_state_digest: str = ""
    behavior_snapshot_iteration: int = 0
    behavior_temperature: float = BEHAVIOR_TEMPERATURE

    # -- what the acting snapshot predicted --------------------------------
    stored_value_scalar: float = 0.0
    stored_wdl: tuple = (0.0, 1.0, 0.0)

    # -- filled when the window closes -------------------------------------
    boundary_status: str = "interior"
    target_provenance: str = "boundary_bootstrap"
    advantage_target: float = 0.0
    wdl_target: tuple = (0.0, 1.0, 0.0)
    standardized_advantage: float = 0.0
    ppo_eligible: bool = False
    value_row_weight: float = 1.0
    policy_age_iterations: int = 0
    bootstrap_age_windows: int = 0

    # -- filled later, when the game ends. TELEMETRY, never a gate ---------
    boundary_target_divergence: "float | None" = None
    boundary_wdl_divergence: "float | None" = None

    # -- frozen contract versions -----------------------------------------
    work_package: str = WORK_PACKAGE
    schema_version: str = MOVE_TRANSITION_VERSION
    rules_version: str = RULES_VERSION
    observation_version: str = OBSERVATION_VERSION
    action_encoding_version: str = ACTION_ENCODING_VERSION

    @property
    def key(self) -> tuple:
        """`(game, colour, ply)` -- the identity a transition is unique under."""
        return (self.game_id, int(self.color), int(self.ply))

    # -- the names the accepted collation reads -----------------------------
    #
    # `stratego.training.phase9_loss.behavior_probability_matrix` is the
    # accepted dense builder over the stored float32 bytes -- never
    # recomputed, never renormalized. It reads four field names Phase 9 fixed.
    # Exposing them as aliases reuses that builder verbatim rather than
    # forking it to learn a second set of names.

    @property
    def learner_side(self) -> int:
        return int(self.color)

    @property
    def decision_index(self) -> int:
        return int(self.ply)

    @property
    def behavior_legal_actions(self) -> tuple:
        return self.legal_actions

    @property
    def behavior_legal_probabilities(self) -> tuple:
        return self.behavior_probabilities

    @property
    def advantage(self) -> float:
        """The name the accepted window statistics read."""
        return float(self.advantage_target)

    @advantage.setter
    def advantage(self, value: float) -> None:
        self.advantage_target = float(value)

    def nbytes(self) -> int:
        return int(self.observation.nbytes + self.legal_mask.nbytes)

    def identity_row(self) -> dict:
        """The row without its tensors: what telemetry and a ledger persist."""
        return {
            "schema_version": self.schema_version,
            "work_package": self.work_package,
            "run_id": self.run_id,
            "iteration": int(self.iteration),
            "window_index": int(self.window_index),
            "game_id": self.game_id,
            "ply": int(self.ply),
            "color": int(self.color),
            "perspective_player": int(self.perspective_player),
            "legal_actions": len(self.legal_actions),
            "sampled_action": int(self.sampled_action),
            "sampled_action_index": int(self.sampled_action_index),
            "action_seed": int(self.action_seed),
            "behavior_action_probability": float(self.behavior_action_probability),
            "behavior_model_state_digest": self.behavior_model_state_digest,
            "behavior_snapshot_iteration": int(self.behavior_snapshot_iteration),
            "behavior_temperature": float(self.behavior_temperature),
            "stored_value_scalar": float(self.stored_value_scalar),
            "stored_wdl": [float(v) for v in self.stored_wdl],
            "boundary_status": self.boundary_status,
            "target_provenance": self.target_provenance,
            "advantage_target": float(self.advantage_target),
            "wdl_target": [float(v) for v in self.wdl_target],
            "standardized_advantage": float(self.standardized_advantage),
            "trained": bool(self.ppo_eligible),
            "value_row_weight": float(self.value_row_weight),
            "policy_age_iterations": int(self.policy_age_iterations),
            "bootstrap_age_windows": int(self.bootstrap_age_windows),
            "boundary_target_divergence": (
                None
                if self.boundary_target_divergence is None
                else float(self.boundary_target_divergence)
            ),
            "boundary_wdl_divergence": (
                None
                if self.boundary_wdl_divergence is None
                else float(self.boundary_wdl_divergence)
            ),
            "rules_version": self.rules_version,
            "observation_version": self.observation_version,
            "action_encoding_version": self.action_encoding_version,
        }


def validate_transition(row: MoveTransition, *, where: str = "<row>") -> None:
    """Refuse a row that is outside the schema. Absence is never a default."""
    if not isinstance(row, MoveTransition):
        raise Phase17TransitionError(
            f"{where}: expected a MoveTransition, got {type(row).__name__}"
        )
    if row.schema_version != MOVE_TRANSITION_VERSION:
        raise Phase17TransitionError(
            f"{where}: schema {row.schema_version!r} is not {MOVE_TRANSITION_VERSION!r}"
        )
    if not row.behavior_model_state_digest:
        raise Phase17TransitionError(
            f"{where}: a transition with no behavior model-state digest cannot "
            "prove which policy produced it"
        )
    if row.boundary_status not in BOUNDARY_STATUSES:
        raise Phase17TransitionError(
            f"{where}: boundary_status {row.boundary_status!r} is not one of "
            f"{list(BOUNDARY_STATUSES)}"
        )
    if row.target_provenance not in TARGET_PROVENANCES:
        raise Phase17TransitionError(
            f"{where}: target_provenance {row.target_provenance!r} is not one of "
            f"{list(TARGET_PROVENANCES)}"
        )
    actions = tuple(int(a) for a in row.legal_actions)
    if not actions:
        raise Phase17TransitionError(f"{where}: a transition with no legal actions")
    if list(actions) != sorted(actions):
        raise Phase17TransitionError(
            f"{where}: legal_actions must be ASCENDING absolute engine ids; the "
            "accepted sampler walks them in order"
        )
    if len(row.behavior_probabilities) != len(actions):
        raise Phase17TransitionError(
            f"{where}: {len(row.behavior_probabilities)} probabilities for "
            f"{len(actions)} legal actions"
        )
    probabilities = np.asarray(row.behavior_probabilities, dtype=np.float64)
    if not np.isfinite(probabilities).all() or bool((probabilities < 0).any()):
        raise Phase17TransitionError(
            f"{where}: a stored behavior probability is negative or non-finite"
        )
    total = float(probabilities.sum())
    if abs(total - 1.0) > BEHAVIOR_PROBABILITY_ABS_TOLERANCE:
        raise Phase17TransitionError(
            f"{where}: the stored behavior distribution sums to {total!r}, "
            f"outside the {BEHAVIOR_PROBABILITY_ABS_TOLERANCE} tolerance"
        )
    if not 0 <= int(row.sampled_action_index) < len(actions):
        raise Phase17TransitionError(
            f"{where}: sampled_action_index {row.sampled_action_index} is outside "
            f"0..{len(actions) - 1}"
        )
    if int(row.sampled_action) != actions[int(row.sampled_action_index)]:
        raise Phase17TransitionError(
            f"{where}: sampled_action {row.sampled_action} is not "
            f"legal_actions[{row.sampled_action_index}]"
        )
    if int(row.perspective_player) != int(row.color):
        raise Phase17TransitionError(
            f"{where}: both seats are learners, so perspective_player must equal "
            f"color; got {row.perspective_player} and {row.color}"
        )
    wdl = np.asarray(row.wdl_target, dtype=np.float64)
    if wdl.shape != (3,) or not np.isfinite(wdl).all() or bool((wdl < 0).any()):
        raise Phase17TransitionError(f"{where}: the W/D/L target is not a simplex point")
    if abs(float(wdl.sum()) - 1.0) > BEHAVIOR_PROBABILITY_ABS_TOLERANCE:
        raise Phase17TransitionError(
            f"{where}: the W/D/L target sums to {float(wdl.sum())!r}"
        )
    if not np.isfinite([row.advantage_target, row.standardized_advantage]).all():
        raise Phase17TransitionError(f"{where}: a non-finite advantage")
    if int(row.policy_age_iterations) < 0:
        raise Phase17TransitionError(
            f"{where}: policy_age_iterations {row.policy_age_iterations} is negative"
        )
    if int(row.bootstrap_age_windows) < 0:
        raise Phase17TransitionError(
            f"{where}: bootstrap_age_windows {row.bootstrap_age_windows} is negative"
        )


def assert_unique(rows) -> dict:
    """Every emitted transition appears exactly once. The no-duplicate proof."""
    seen: dict = {}
    for row in rows:
        if row.key in seen:
            raise Phase17TransitionError(
                f"transition {row.key} was emitted twice: first in window "
                f"{seen[row.key]}, again in window {row.window_index}"
            )
        seen[row.key] = int(row.window_index)
    return {"rows": len(seen), "duplicates": 0}


def transition_schema_document() -> dict:
    return {
        "schema_version": MOVE_TRANSITION_VERSION,
        "one_row_is": "one learner decision by one seat at one ply",
        "both_seats_are_learners": True,
        "fields": sorted(MoveTransition.__dataclass_fields__),
        "unique_under": ["game_id", "color", "ply"],
        "behavior_probabilities": "stored, never recomputed; the PPO denominator",
        "legal_actions": "ASCENDING absolute engine action ids",
        "current_policy_proof": "behavior_model_state_digest",
        "telemetry_never_a_gate": [
            "boundary_target_divergence",
            "boundary_wdl_divergence",
            "bootstrap_age_windows",
        ],
    }


__all__ = [
    "MoveTransition",
    "Phase17TransitionError",
    "assert_unique",
    "transition_schema_document",
    "validate_transition",
]
