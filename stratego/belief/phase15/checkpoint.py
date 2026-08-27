"""Phase 15 Agent 1: the B18/B24 checkpoint format.

Specification source: `01_AGENT_1_BELIEF_HEAD_TRAINING.md` section 8.

What a belief checkpoint contains
---------------------------------
```text
copied final block            copied encoder norm
belief MLP                    calibration temperature
source policy identity + digests
corpus / config identity
```

and nothing else. There is no policy tensor and no value tensor in the
file, so a caller who wanted to read one could not: the guarantee is the
file's contents, not a convention.

Bound to its source, by refusal
-------------------------------
:func:`load_specialist` requires the frozen policy model the checkpoint was
built from, and refuses to return a specialist unless the loaded model's
state digest equals the one recorded at save time. A B18 file cannot be
attached to P24 by accident, and neither can be attached to a P18 whose
bytes have changed.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from .contract import Phase15Error, SPECIALIST_SOURCE
from .heads import BELIEF_ARCHITECTURE_VERSION, Phase15BeliefSpecialist

#: The checkpoint-format identity.
BELIEF_CHECKPOINT_VERSION = "phase15_belief_checkpoint_v1"

#: The tensor-name prefixes a belief checkpoint may contain. Anything else
#: is refused on save and on load.
ALLOWED_PREFIXES = ("block.", "encoder_norm.", "head.", "log_temperature")

#: Prefixes that would mean policy or value weights leaked into the file.
FORBIDDEN_PREFIXES = (
    "policy_",
    "value_",
    "belief_output",
    "input_projection",
    "position_embedding",
    "blocks.",
)


class Phase15CheckpointError(Phase15Error):
    """A belief checkpoint could not be written, read back or bound."""


def state_digest(state: dict) -> str:
    """sha256 over `(name, shape, float32 C-order bytes)` in sorted order.

    The accepted digest recipe, applied to a belief specialist's own state
    dict, so two files holding the same weights always agree.
    """
    hasher = hashlib.sha256()
    for name in sorted(state):
        array = state[name].detach().to("cpu", torch.float32).contiguous().numpy()
        hasher.update(name.encode())
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def check_contents(state: dict) -> None:
    """Refuse a state dict that is not exactly the belief specialist's."""
    for name in state:
        if not name.startswith(ALLOWED_PREFIXES):
            raise Phase15CheckpointError(
                f"tensor {name!r} is not part of a belief specialist; a belief "
                "checkpoint holds only the copied block, the encoder norm, the "
                "belief MLP and the calibration temperature"
            )
        for forbidden in FORBIDDEN_PREFIXES:
            if name.startswith(forbidden):  # pragma: no cover - unreachable above
                raise Phase15CheckpointError(
                    f"tensor {name!r} looks like a policy or value parameter"
                )


def save_specialist(
    model: Phase15BeliefSpecialist,
    path: "Path | str",
    *,
    source_identity: dict,
    corpus_identity: dict,
    training_record: dict,
    calibration_record: dict,
    overwrite: bool = False,
) -> dict:
    """Write one B18/B24 checkpoint. Returns its identity block.

    A run that would overwrite existing bytes is refused unless explicitly
    asked: section 9 requires that a repeat or reproducibility pass cannot
    overwrite the selected checkpoint.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise Phase15CheckpointError(
            f"{path} already exists; every run must use a unique output path so a "
            "repeat pass cannot overwrite the selected bytes"
        )
    state = {
        name: tensor.detach().to("cpu", torch.float32).clone()
        for name, tensor in model.state_dict().items()
    }
    check_contents(state)
    specialist_id = model.specialist_id
    expected_source = SPECIALIST_SOURCE.get(specialist_id)
    if expected_source and source_identity.get("source_id") != expected_source:
        raise Phase15CheckpointError(
            f"{specialist_id} must bind to {expected_source}, the identity names "
            f"{source_identity.get('source_id')!r}"
        )

    payload = {
        "checkpoint_format_version": BELIEF_CHECKPOINT_VERSION,
        "architecture_version": BELIEF_ARCHITECTURE_VERSION,
        "specialist_id": specialist_id,
        "state_dict": state,
        "state_digest": state_digest(state),
        "temperature": model.temperature,
        "parameters": model.parameter_counts(),
        "architecture": model.describe()["architecture"],
        "source_policy": {
            "source_id": source_identity["source_id"],
            "logical_identity": source_identity["logical_identity"],
            "hour": source_identity["hour"],
            "model_state_digest": source_identity["model_state_digest"],
            "phase15_copy_sha256": source_identity["phase15_copy_sha256"],
            "phase15_copy_path": source_identity["phase15_copy_path"],
            "original_snapshot_sha256": source_identity["original_snapshot_sha256"],
            "global_optimizer_step": source_identity["global_optimizer_step"],
        },
        "corpus": {
            "corpus_version": corpus_identity["corpus_version"],
            "corpus_digest": corpus_identity["corpus_digest"],
            "corpus_format_version": corpus_identity["corpus_format_version"],
        },
        "training": {
            key: training_record[key]
            for key in (
                "config",
                "best_epoch",
                "best_dev_ce",
                "epochs_run",
                "stopped_because",
                "time_to_best_seconds",
                "training_seconds",
                "train_positions",
            )
            if key in training_record
        },
        "calibration": calibration_record,
        "holds_policy_parameters": False,
        "holds_value_parameters": False,
        "note": (
            "a belief specialist. Its outputs are hidden-rank marginals only; it "
            "has no policy or value head, and the deployed move model is the "
            "untouched Phase 14 candidate this file names."
        ),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "bytes": int(path.stat().st_size),
        "state_digest": payload["state_digest"],
        "specialist_id": specialist_id,
        "temperature": payload["temperature"],
    }


def _file_sha256(path: "Path | str", *, chunk: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def read_payload(path: "Path | str") -> dict:
    """The checkpoint document, without building a model."""
    path = Path(path)
    if not path.is_file():
        raise Phase15CheckpointError(f"no belief checkpoint at {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_format_version") != BELIEF_CHECKPOINT_VERSION:
        raise Phase15CheckpointError(
            f"{path} is not a {BELIEF_CHECKPOINT_VERSION} document"
        )
    check_contents(payload["state_dict"])
    observed = state_digest(payload["state_dict"])
    if observed != payload["state_digest"]:
        raise Phase15CheckpointError(
            f"{path}: state digest {observed} != recorded {payload['state_digest']}"
        )
    return payload


def load_specialist(
    path: "Path | str", policy_model, *, device: str = "cpu"
) -> "tuple[Phase15BeliefSpecialist, dict]":
    """Load one specialist, bound to the policy model it was trained from.

    Refuses unless the supplied backbone's model-state digest equals the one
    recorded at save time — section 8's "belief checkpoint loads only with
    its recorded source identity".
    """
    from ...training.phase9_behavior import state_dict_digest

    payload = read_payload(path)
    observed = state_dict_digest(policy_model)
    expected = payload["source_policy"]["model_state_digest"]
    if observed != expected:
        raise Phase15CheckpointError(
            f"{Path(path).name} records source {payload['source_policy']['logical_identity']} "
            f"({expected[:16]}), the supplied backbone is {observed[:16]}"
        )
    model = Phase15BeliefSpecialist.from_policy(
        policy_model, specialist_id=payload["specialist_id"]
    )
    model.load_state_dict(payload["state_dict"])
    model.to(torch.device(device)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


__all__ = [
    "ALLOWED_PREFIXES",
    "BELIEF_CHECKPOINT_VERSION",
    "FORBIDDEN_PREFIXES",
    "Phase15CheckpointError",
    "check_contents",
    "load_specialist",
    "read_payload",
    "save_specialist",
    "state_digest",
]
