"""Phase 16 Agent 3: the evaluation seat for a trained arm checkpoint.

Agent 1's runner scores "any object implementing the Phase 15 decision-seat
interface, supplied as a factory". An arm's checkpoint is not a Phase 15
pairing id -- it is weights that did not exist when the handoff was frozen --
so this module is that factory, and it is the *only* glue between training and
measurement.

The seat is the accepted `DirectSeat`
-------------------------------------
A direct arm is one greedy forward per decision, and the accepted class already
does exactly that against an `InferenceOwner`. What this module supplies is an
owner pointing at the arm's exported weights instead of at a frozen Phase 15
move model. Nothing about how the decision is made, checked or timed differs
from how `p24_direct` is scored, which is what makes an arm's number
comparable to the baseline Agent 1 already measured on the same pack.

Search is deliberately absent: sections 2 and 7 keep search out of this phase's
training loop and out of its recipe comparison. An arm is scored as a direct
policy, which is what the recipe changed.
"""

from __future__ import annotations

from pathlib import Path

from .contract import Phase16TrainingError

#: The pairing whose seat shape a Phase 16 arm borrows: P24 played directly.
BASE_PAIRING_ID = "p24_direct"


class Phase16SeatError(Phase16TrainingError):
    """A Phase 16 arm could not be seated for measurement."""


def build_seat(
    *,
    models=None,
    owners=None,
    preset: str = "direct",
    device: str = "cpu",
    weights_path: str,
    arm_id: str = "phase16_arm",
    expected_sha256: "str | None" = None,
):
    """The Agent 1 provider factory for one arm's exported weights.

    Called once per worker as
    `factory(models=..., owners=..., preset=..., device=..., **kwargs)`.
    """
    from ...evaluation.neural_worker import InferenceOwner
    from ...model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from ...model.policy_adapter import DECISION_MODE_GREEDY
    from ...search.phase15.contract import pairing as pairing_of
    from ...search.phase15.matchplay import DirectSeat

    path = Path(weights_path)
    if not path.is_file():
        raise Phase16SeatError(f"the arm's exported weights are missing at {path}")
    if expected_sha256 is not None:
        from .checkpoint import file_sha256

        observed = file_sha256(path)
        if observed != expected_sha256:
            raise Phase16SeatError(
                f"{path} has SHA-256 {observed}, but the arm is bound to "
                f"{expected_sha256}"
            )

    target = pairing_of(BASE_PAIRING_ID)
    owner = InferenceOwner(
        str(path),
        decision_mode=DECISION_MODE_GREEDY,
        device=device,
        dtype="float32",
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
        name=f"phase16_{arm_id}",
    )
    seat = DirectSeat(target, {target.move_model: owner})
    seat.arm_id = str(arm_id)
    return seat


def provider_spec(weights_path: str, *, arm_id: str, expected_sha256: "str | None" = None) -> dict:
    """The `{'factory': ..., 'kwargs': ..., 'arm_id': ...}` Agent 1's runner takes."""
    kwargs = {"weights_path": str(weights_path), "arm_id": str(arm_id)}
    if expected_sha256:
        kwargs["expected_sha256"] = str(expected_sha256)
    return {
        "factory": "stratego.training.phase16.seat:build_seat",
        "kwargs": kwargs,
        "arm_id": str(arm_id),
    }


__all__ = ["BASE_PAIRING_ID", "Phase16SeatError", "build_seat", "provider_spec"]
