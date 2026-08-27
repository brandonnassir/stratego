"""Phase 16 Agent 3: arm checkpoints, resume, and the evaluation export.

A 6-hour arm needs three things from a checkpoint and no more: it must resume
where it stopped, it must be scoreable at fixed hours, and it must be provably
the arm it claims to be. So a `phase16_checkpoint_v1` payload carries the raw
model state, the optimizer moments, the EMA state when the arm has one, the
trainer and collector state, and the arm config *and its digest*.

Resume identity
---------------
A resume is refused if the stored arm digest does not match the arm being
resumed. Two arms of this shootout differ only by flags, so "resumed arm B's
checkpoint into arm C" is exactly the mistake that would silently produce a
fourth, unnamed recipe and report it as one of the three.

The evaluation export
---------------------
Candidates are scored through the accepted `InferenceOwner`, which reads
`stratego.model.checkpoint`'s format. Producing that format here -- and
verifying every tensor round-trips bitwise before the file is used -- is what
lets the evaluator be the accepted machinery rather than a second greedy
decision path. The procedure is Phase 9 Agent 8's, unchanged in substance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from ...model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
from ...model.checkpoint import load_checkpoint, save_checkpoint
from ..phase9_behavior import state_dict_digest
from .contract import (
    PHASE16_CHECKPOINT_VERSION,
    PHASE16_TRAINING_VERSION,
    STARTING_CHECKPOINT,
    STARTING_CHECKPOINT_SHA256,
    STARTING_MODEL_STATE_DIGEST,
    ArmConfig,
    Phase16TrainingError,
    contract_digest,
)

REQUIRED_KEYS = (
    "phase16_checkpoint_version",
    "arm",
    "arm_digest",
    "model_state",
    "model_state_digest",
    "optimizer_state",
    "ema_state",
    "trainer_state",
    "collector_state",
    "clock",
)


class Phase16CheckpointError(Phase16TrainingError):
    """A Phase 16 checkpoint could not be written, read or resumed."""


def repository_root() -> Path:
    """The repo root, derived from this file rather than from the cwd.

    An arm runs under `nohup` from a launcher whose working directory is not
    guaranteed, and an export that resolved the starting checkpoint relative to
    the cwd would fail hours into a run rather than at import.
    """
    return Path(__file__).resolve().parents[3]


def file_sha256(path: "str | Path") -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu")
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(item) for item in value)
    return value


# ---------------------------------------------------------------------------
# The starting model
# ---------------------------------------------------------------------------


def load_starting_model(checkpoint_path=None, *, device: str = "mps", root: "str | Path" = "."):
    """The read-only P24 copy every arm starts from, digest-checked twice.

    Refuses anything whose file SHA-256 or model-state digest is not the
    accepted one: "wrong starting model" is detected here rather than after an
    hour of training.

    The copy is in the accepted *evaluation* format (it is the Phase 14 hour-24
    export Phase 15 froze), so it is read through the accepted model loader
    rather than the Phase 9 training-payload reader. Unlike Phase 15, which
    freezes it, this returns a **trainable** model: gradients on, training mode
    off until the trainer sets it.
    """
    path = Path(checkpoint_path or (Path(root) / STARTING_CHECKPOINT))
    if not path.is_file():
        raise Phase16CheckpointError(f"the starting checkpoint is missing at {path}")
    digest = file_sha256(path)
    if digest != STARTING_CHECKPOINT_SHA256:
        raise Phase16CheckpointError(
            f"{path} has SHA-256 {digest}, not the accepted "
            f"{STARTING_CHECKPOINT_SHA256}; a Phase 16 arm may not start from "
            "another checkpoint"
        )
    model, _metadata = load_checkpoint(
        path,
        device=torch.device(device),
        dtype=torch.float32,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
    )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    observed = state_dict_digest(model)
    if observed != STARTING_MODEL_STATE_DIGEST:
        raise Phase16CheckpointError(
            f"the loaded model-state digest {observed} != the accepted "
            f"{STARTING_MODEL_STATE_DIGEST}"
        )
    return model


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


def build_payload(
    *,
    config: ArmConfig,
    model,
    optimizer,
    ema,
    trainer_state: dict,
    collector_state: dict,
    clock: dict,
    diagnostics: "dict | None" = None,
) -> dict:
    """One `phase16_checkpoint_v1` payload, complete by construction."""
    payload = {
        "phase16_checkpoint_version": PHASE16_CHECKPOINT_VERSION,
        "training_version": PHASE16_TRAINING_VERSION,
        "contract_digest": contract_digest(),
        "arm": config.to_dict(),
        "arm_digest": config.digest(),
        "model_state": _to_cpu(model.state_dict()),
        "model_state_digest": state_dict_digest(model),
        "optimizer_state": _to_cpu(optimizer.state_dict()),
        "ema_state": (
            {
                "present": True,
                "decay": float(ema.decay),
                "updates": int(ema.updates),
                "state_dict": _to_cpu(ema.state_dict()),
            }
            if ema is not None
            else {"present": False, "statement": "this arm runs without an EMA"}
        ),
        "trainer_state": dict(trainer_state),
        "collector_state": dict(collector_state),
        "clock": dict(clock),
        "starting_checkpoint": {
            "path": STARTING_CHECKPOINT,
            "sha256": STARTING_CHECKPOINT_SHA256,
            "model_state_digest": STARTING_MODEL_STATE_DIGEST,
        },
        "diagnostics": dict(diagnostics or {}),
    }
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:  # pragma: no cover - the literal above covers every key
        raise Phase16CheckpointError(f"the payload is missing {missing}")
    return payload


def save(payload: dict, path: "str | Path") -> dict:
    """Write a payload atomically and return what landed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return {
        "path": str(destination),
        "sha256": file_sha256(destination),
        "bytes": destination.stat().st_size,
        "model_state_digest": payload["model_state_digest"],
        "arm_digest": payload["arm_digest"],
    }


def read(path: "str | Path") -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise Phase16CheckpointError(f"{path}: payload is not a mapping")
    if payload.get("phase16_checkpoint_version") != PHASE16_CHECKPOINT_VERSION:
        raise Phase16CheckpointError(
            f"{path} carries checkpoint version "
            f"{payload.get('phase16_checkpoint_version')!r}, not "
            f"{PHASE16_CHECKPOINT_VERSION!r}"
        )
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise Phase16CheckpointError(f"{path} is missing {missing}")
    return payload


def restore(payload: dict, *, config: ArmConfig, model, optimizer, ema=None) -> dict:
    """Load one payload into a live arm, refusing a different arm.

    The digest check is the whole point: three arms that differ only by flags
    would otherwise resume into each other and report a fourth, unnamed recipe
    under one of their names.
    """
    if payload["arm_digest"] != config.digest():
        raise Phase16CheckpointError(
            f"the checkpoint holds arm {payload['arm'].get('arm_id')!r} with digest "
            f"{payload['arm_digest'][:12]}, but this run is arm {config.arm_id!r} "
            f"with digest {config.digest()[:12]}"
        )
    model.load_state_dict(payload["model_state"])
    observed = state_dict_digest(model)
    if observed != payload["model_state_digest"]:
        raise Phase16CheckpointError(
            f"the restored model-state digest {observed} != the stored "
            f"{payload['model_state_digest']}"
        )
    optimizer.load_state_dict(payload["optimizer_state"])
    stored_ema = payload.get("ema_state") or {}
    if ema is not None:
        if not stored_ema.get("present"):
            raise Phase16CheckpointError(
                "this arm runs an EMA but the checkpoint stores none"
            )
        ema.load_state_dict(stored_ema["state_dict"], updates=stored_ema.get("updates", 0))
    return {
        "arm_id": config.arm_id,
        "model_state_digest": observed,
        "ema_restored": ema is not None,
        "clock": dict(payload.get("clock") or {}),
        "trainer_state": dict(payload.get("trainer_state") or {}),
        "collector_state": dict(payload.get("collector_state") or {}),
    }


# ---------------------------------------------------------------------------
# The evaluation export
# ---------------------------------------------------------------------------


def export_evaluation_weights(
    payload_or_path, export_path: "str | Path", *, use_ema: bool = False
) -> dict:
    """Export one arm state to the format the accepted `InferenceOwner` reads.

    The export is refused unless every tensor round-trips bitwise, so a scored
    checkpoint can never be weights that are not the checkpoint's.
    """
    payload = (
        payload_or_path
        if isinstance(payload_or_path, dict)
        else read(payload_or_path)
    )
    state = payload["model_state"]
    if use_ema:
        ema_state = payload.get("ema_state") or {}
        if not ema_state.get("present"):
            raise Phase16CheckpointError(
                "an EMA export was asked for but this checkpoint stores no EMA"
            )
        state = ema_state["state_dict"]

    # Rebuild the accepted C1 network, then load the requested state into it.
    source = repository_root() / STARTING_CHECKPOINT
    if not source.is_file():  # pragma: no cover - the arm already loaded it
        raise Phase16CheckpointError(f"the starting checkpoint is missing at {source}")
    model, _metadata = load_checkpoint(
        source,
        device=torch.device("cpu"),
        dtype=torch.float32,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
    )
    model.load_state_dict(state)
    model.eval()
    digest = state_dict_digest(model)
    parameters = int(sum(tensor.numel() for tensor in model.parameters()))

    export_path = Path(export_path)
    save_checkpoint(model, export_path)
    reloaded, _metadata = load_checkpoint(
        export_path,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
    )
    left = model.state_dict()
    right = reloaded.state_dict()
    bitwise = set(left) == set(right) and all(
        torch.equal(left[name], right[name]) for name in left
    )
    if not bitwise:
        raise Phase16CheckpointError(
            f"the evaluation export to {export_path} changed the weights; BLOCKED"
        )
    if state_dict_digest(reloaded) != digest:
        raise Phase16CheckpointError(
            f"the evaluation export to {export_path} changed the model-state digest"
        )
    del model, reloaded
    return {
        "export": str(export_path),
        "export_sha256": file_sha256(export_path),
        "model_state_digest": digest,
        "parameters": parameters,
        "source": "ema" if use_ema else "raw",
        "bitwise_state_dict_match": True,
    }


def write_identity(path: "str | Path", document: dict) -> dict:
    """Write a small JSON identity next to an exported weights file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    return {"path": str(target), "sha256": file_sha256(target)}


def checkpoint_semantics() -> dict:
    return {
        "checkpoint_version": PHASE16_CHECKPOINT_VERSION,
        "carries": list(REQUIRED_KEYS),
        "resume_identity": "the arm digest; a different arm is refused",
        "ema": "stored beside the raw state, never instead of it",
        "export": "stratego.model.checkpoint.save_checkpoint, bitwise-verified",
    }


__all__ = [
    "PHASE16_CHECKPOINT_VERSION",
    "Phase16CheckpointError",
    "REQUIRED_KEYS",
    "build_payload",
    "checkpoint_semantics",
    "export_evaluation_weights",
    "file_sha256",
    "load_starting_model",
    "read",
    "restore",
    "save",
    "write_identity",
]
