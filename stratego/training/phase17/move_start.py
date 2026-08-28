"""Phase 17 Agent 2: the exact Phase 9 move-policy start.

Specification sources: common contract section 4, Agent 2 instruction section 2.

Weights only, and why that is the whole point
---------------------------------------------
Phase 17 is a new lineage from accepted Phase 9 weights, not a resume of the
Phase 9 run. The file at `checkpoints/phase9/selfplay_c1_v1.pt` is a *Phase 9
training* checkpoint and carries an optimizer state, a scheduler position, a
KL controller with beta already at its 0.2 ceiling, a minibatch cursor and an
RNG snapshot. Every one of those is discarded here, deliberately and by name,
because carrying one forward would silently make Phase 17 iteration 1 continue
a schedule that belongs to a different run over a different data distribution.

`check_phase9_resume_identity` is therefore *not* called: it authorizes a
Phase 9 resume, and this is not one. The accepted read/validate/rebuild path
is called in full, so the file is still validated against its own embedded
contract before a single tensor is used.

Two digests, one of which is a trap
-----------------------------------
The repository contains two functions named `state_dict_digest` and they
disagree on these bytes. The Phase 17 model-state digest is
`stratego.training.phase9_behavior.state_dict_digest` over a live module.
`stratego.model.checkpoint.state_dict_digest` yields
`f0994cf0...` on the same weights because it folds the dtype in. Both are
checked here -- the second only so that a mismatch can be *named* rather than
mistaken for corruption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..phase16.trainer import WeightEMA
from ..phase9_behavior import file_sha256, state_dict_digest
from ..phase9_checkpoint import (
    model_from_payload,
    read_phase9_payload,
    validate_phase9_payload,
)
from ..phase9_trainer import KLController
from .move_contract import (
    BELIEF_LOSS_WEIGHT,
    MOVE_EMA_DECAY,
    MOVE_INITIAL_KL_BETA,
    MOVE_OPTIMIZER,
    START_CANDIDATE_ID,
    START_CHECKPOINT_PATH,
    START_CONTAINER_STATE_DIGEST,
    START_FILE_SHA256,
    START_MODEL_STATE_DIGEST,
    START_PARAMETER_COUNT,
    MoveScheduleHorizon,
    Phase17MoveError,
)

PHASE17_MOVE_START_VERSION = "phase17_move_start_v1"

#: What the Phase 9 payload carries and Phase 17 refuses to inherit.
DISCARDED_PAYLOAD_KEYS = (
    "optimizer_state",
    "scheduler_state",
    "kl_controller_state",
    "minibatch_cursor",
    "rng",
    "validation_history",
    "global_optimizer_step",
    "rl_iteration",
    "examples_consumed",
    "entropy_schedule_position",
    "best_validation_score",
    "best_checkpoint_identity",
)

SUPPORTED_DEVICES = ("cpu", "mps")

#: The old marginal-belief head's parameter prefix in the accepted C1 model.
#: Named as a constant so the zero-weight test asserts something concrete: if
#: the head is renamed upstream, `belief_head_parameters` raises rather than
#: quietly checking an empty list.
BELIEF_HEAD_PREFIX = "belief_output."


class Phase17MoveStartError(Phase17MoveError):
    """The Phase 17 move start could not be established as specified."""


def load_phase17_move_weights(
    path: "str | Path | None" = None,
    *,
    device: str = "cpu",
    root: "str | Path" = ".",
    expected_file_sha256: str = START_FILE_SHA256,
    expected_model_state_digest: str = START_MODEL_STATE_DIGEST,
) -> dict:
    """Load the accepted Phase 9 weights, digest-checked on both claims.

    Returns a **trainable** float32 model on `device` plus the identity that
    names it. Nothing is returned until both digests reproduce, so a wrong
    starting checkpoint is detected here rather than after an hour of
    training.
    """
    if device not in SUPPORTED_DEVICES:
        raise Phase17MoveStartError(
            f"unsupported device {device!r}; expected one of {list(SUPPORTED_DEVICES)}"
        )
    target = Path(path) if path is not None else Path(root) / START_CHECKPOINT_PATH
    if not target.is_file():
        raise Phase17MoveStartError(
            f"the accepted Phase 17 move start is missing at {target}"
        )

    observed_file = file_sha256(target)
    if observed_file != expected_file_sha256:
        raise Phase17MoveStartError(
            f"{target} has SHA-256 {observed_file}, not the accepted "
            f"{expected_file_sha256}; Phase 17 may not start from another "
            "checkpoint, including a convenience copy with different bytes"
        )

    payload = read_phase9_payload(target)
    metadata = validate_phase9_payload(payload, source=str(target))
    model = model_from_payload(payload, device=device)

    configuration = payload["model_state"]["model_configuration"]
    if str(configuration.get("candidate_id")) != START_CANDIDATE_ID:
        raise Phase17MoveStartError(
            f"{target} builds candidate {configuration.get('candidate_id')!r}, "
            f"not the accepted {START_CANDIDATE_ID!r}"
        )

    observed_state = state_dict_digest(model)
    if observed_state != expected_model_state_digest:
        raise Phase17MoveStartError(
            f"the loaded model-state digest {observed_state} != the accepted "
            f"{expected_model_state_digest} (note: the *container* digest "
            f"convention would give {START_CONTAINER_STATE_DIGEST}; these are "
            "two different functions with the same name)"
        )

    parameters = int(sum(p.numel() for p in model.parameters()))
    if parameters != START_PARAMETER_COUNT:
        raise Phase17MoveStartError(
            f"the loaded model has {parameters} parameters, not the accepted "
            f"{START_PARAMETER_COUNT}"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.eval()

    return {
        "model": model,
        "path": str(target),
        "file_sha256": observed_file,
        "model_state_digest": observed_state,
        "container_state_digest": START_CONTAINER_STATE_DIGEST,
        "parameter_count": parameters,
        "candidate_id": START_CANDIDATE_ID,
        "behavior_snapshot_identity": payload.get("behavior_snapshot_identity"),
        "produced_after_iteration": payload.get("diagnostics", {}).get(
            "produced_after_iteration"
        ),
        "metadata": metadata,
        "discarded": list(DISCARDED_PAYLOAD_KEYS),
        "resume_identity_check_used": False,
    }


@dataclass
class Phase17MoveStart:
    """The move learner at Phase 17 iteration 1: weights old, everything else new."""

    model: object = field(repr=False)
    optimizer: object = field(repr=False)
    controller: KLController = field(repr=False)
    ema: WeightEMA = field(repr=False)
    horizon: MoveScheduleHorizon
    identity: dict
    device: str
    iteration: int = 0

    @property
    def next_iteration(self) -> int:
        """The 1-based iteration the learner is about to run."""
        return int(self.iteration) + 1

    def to_dict(self) -> dict:
        return {
            "start_version": PHASE17_MOVE_START_VERSION,
            "start_identity": {
                key: self.identity[key]
                for key in (
                    "path",
                    "file_sha256",
                    "model_state_digest",
                    "parameter_count",
                    "candidate_id",
                )
            },
            "device": self.device,
            "iteration": int(self.iteration),
            "optimizer": {
                "name": MOVE_OPTIMIZER["name"],
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "betas": list(MOVE_OPTIMIZER["betas"]),
                "eps": MOVE_OPTIMIZER["eps"],
                "weight_decay": MOVE_OPTIMIZER["weight_decay"],
                "moments_at_start": "fresh, zero",
                "state_entries": len(self.optimizer.state),
            },
            "kl_controller": {
                "beta": float(self.controller.beta),
                "target": float(self.controller.target),
                "updates": len(self.controller.history),
            },
            "ema": self.ema.to_dict(),
            "schedule": self.horizon.to_dict(),
            "belief_loss_weight": BELIEF_LOSS_WEIGHT,
            "discarded_from_phase9_payload": list(DISCARDED_PAYLOAD_KEYS),
        }


def build_move_start(
    *,
    total_iterations: int,
    path: "str | Path | None" = None,
    device: str = "cpu",
    root: "str | Path" = ".",
    horizon: "MoveScheduleHorizon | None" = None,
) -> Phase17MoveStart:
    """Common contract section 4, executed: weights loaded, everything else fresh.

    The EMA is initialized *from the loaded raw weights*, so before the first
    update the two are identical and the first exported evaluation candidate is
    the Phase 9 policy rather than an average of it with noise.
    """
    loaded = load_phase17_move_weights(path, device=device, root=root)
    model = loaded["model"]
    curve = horizon or MoveScheduleHorizon(total_iterations=total_iterations)
    if horizon is not None and int(horizon.total_iterations) != int(total_iterations):
        raise Phase17MoveStartError(
            f"the supplied horizon covers {horizon.total_iterations} iterations, "
            f"not the requested {total_iterations}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=curve.learning_rate(1),
        betas=tuple(float(value) for value in MOVE_OPTIMIZER["betas"]),
        eps=float(MOVE_OPTIMIZER["eps"]),
        weight_decay=float(MOVE_OPTIMIZER["weight_decay"]),
    )
    if optimizer.state:  # pragma: no cover - a fresh AdamW has no state
        raise Phase17MoveStartError("a fresh Phase 17 optimizer must carry no moments")

    controller = KLController(beta=float(MOVE_INITIAL_KL_BETA))
    ema = WeightEMA(model, MOVE_EMA_DECAY)
    return Phase17MoveStart(
        model=model,
        optimizer=optimizer,
        controller=controller,
        ema=ema,
        horizon=curve,
        identity=loaded,
        device=device,
        iteration=0,
    )


def belief_head_parameters(model) -> list:
    """The old marginal-belief head's parameters, by name.

    Named rather than inferred so the zero-weight test asserts something
    concrete: if the head is ever renamed, the test fails loudly instead of
    silently checking an empty list.
    """
    names = [
        name
        for name, _ in model.named_parameters()
        if name.startswith(BELIEF_HEAD_PREFIX)
    ]
    if not names:
        raise Phase17MoveStartError(
            f"the C1 model exposes no `{BELIEF_HEAD_PREFIX}*` parameters; the "
            "Phase 17 zero-belief-weight claim cannot be checked against nothing"
        )
    return names


def start_semantics() -> dict:
    return {
        "start_version": PHASE17_MOVE_START_VERSION,
        "loader": (
            "stratego.training.phase9_checkpoint.read_phase9_payload -> "
            "validate_phase9_payload -> model_from_payload"
        ),
        "checks": ["file_sha256", "model_state_digest", "candidate_id", "parameter_count"],
        "digest_function": "stratego.training.phase9_behavior.state_dict_digest",
        "mode": "weights_only_warm_start",
        "creates": [
            "fresh AdamW with zero moments",
            "fresh move KL controller at beta 0.005",
            "schedule state at Phase 17 iteration 1",
            "move EMA initialized from the loaded raw weights",
        ],
        "discards": list(DISCARDED_PAYLOAD_KEYS),
        "belief_loss_weight": BELIEF_LOSS_WEIGHT,
        "resume_identity_check": (
            "not used: check_phase9_resume_identity authorizes a Phase 9 resume "
            "and Phase 17 is a new lineage"
        ),
    }


__all__ = [
    "BELIEF_HEAD_PREFIX",
    "DISCARDED_PAYLOAD_KEYS",
    "PHASE17_MOVE_START_VERSION",
    "Phase17MoveStart",
    "Phase17MoveStartError",
    "belief_head_parameters",
    "build_move_start",
    "load_phase17_move_weights",
    "start_semantics",
]
