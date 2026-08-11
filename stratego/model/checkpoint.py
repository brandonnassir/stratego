"""Checkpoint format, compatibility validation and loading.

Specification source: Phase 5 single-agent instructions, section 4.4.

The rule this module exists to enforce
--------------------------------------
Weights are meaningless without the semantics they were trained under. A tensor
of the right shape can always be loaded; what makes it *correct* is that the
rules, the observation, the action encoding and the model contract are the same
ones the weights were fitted to. So every compatibility field is checked before
the state dict is touched, and every failure raises. There is no "load anyway"
switch, because the failure mode it would create -- a network silently playing
under the wrong observation semantics -- is invisible in every downstream
metric.

Layout
------
A checkpoint is a single `torch.save` payload holding plain dictionaries,
strings, integers and tensors -- no custom classes -- so it loads under
`weights_only=True` and cannot execute code from disk.

.. code-block:: text

    checkpoint_format_version   int, this module's format generation
    model_architecture_id       "integration_model_v1"
    model_contract_version      "model_contract_v1"
    rules_version               "stratego_project_v1"
    observation_version         "observation_v2_1_127ch"
    action_encoding_version     "source_destination_10000_v1"
    policy_action_frame         "absolute_engine_squares"
    model_configuration         dict, the IntegrationModelConfig fields
    state_dict                  dict[str, Tensor]
    training_iteration          int
    training_step               int
    creation_timestamp          ISO 8601 UTC string
    optimizer_state             optional, may be absent or None
    ema_state                   optional, may be absent or None
    training_metrics            optional, may be absent or None
    provenance                  dict: torch/python versions, weights digest, note
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..engine.constants import OBSERVATION_VERSION, RULES_VERSION
from .contract import (
    ACTION_ENCODING_VERSION,
    MODEL_CONTRACT_VERSION,
    POLICY_ACTION_FRAME,
)
from .integration_model import (
    FIXTURE_NOTE,
    MODEL_ARCHITECTURE_ID,
    IntegrationModel,
    IntegrationModelConfig,
)

#: Current checkpoint generation. A file claiming a *higher* version was written
#: by newer code whose semantics this code cannot know, so it is refused.
CHECKPOINT_FORMAT_VERSION = 1

#: Fields that must be present. `state_dict` is included: a metadata-only file is
#: not a checkpoint.
REQUIRED_FIELDS: tuple[str, ...] = (
    "checkpoint_format_version",
    "model_architecture_id",
    "model_contract_version",
    "rules_version",
    "observation_version",
    "action_encoding_version",
    "model_configuration",
    "state_dict",
    "training_iteration",
    "training_step",
    "creation_timestamp",
)

#: Fields that may be absent, or present and `None`.
OPTIONAL_FIELDS: tuple[str, ...] = (
    "optimizer_state",
    "ema_state",
    "training_metrics",
    "policy_action_frame",
    "provenance",
)


class CheckpointError(RuntimeError):
    """Base class for every checkpoint failure. Always loud, never repaired."""


class CheckpointFormatError(CheckpointError):
    """The file is unreadable, truncated, or structurally not a checkpoint."""


class CheckpointCompatibilityError(CheckpointError):
    """The file is readable but its semantics do not match this code."""


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def state_dict_digest(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Order-independent SHA-256 over parameter names, shapes, dtypes and bytes.

    Sorting the names first means the digest depends on the weights and not on
    dictionary insertion order, so it is stable across a save/reload cycle.
    """
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().to("cpu")
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_digest(path: "str | Path") -> str:
    """SHA-256 of the checkpoint file itself, for report evidence."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def build_checkpoint_payload(
    model: IntegrationModel,
    *,
    training_iteration: int = 0,
    training_step: int = 0,
    optimizer_state: "Mapping[str, Any] | None" = None,
    ema_state: "Mapping[str, Any] | None" = None,
    training_metrics: "Mapping[str, Any] | None" = None,
    creation_timestamp: str | None = None,
) -> dict:
    """Assemble the payload for `model` without writing anything.

    The state dict is moved to CPU float32 first so a checkpoint written from a
    Metal or float16 run reloads identically on any device.
    """
    if not isinstance(model, IntegrationModel):
        raise CheckpointError(
            f"expected an IntegrationModel, got {type(model).__name__}"
        )
    state_dict = {
        name: tensor.detach().to("cpu", torch.float32).clone()
        for name, tensor in model.state_dict().items()
    }
    timestamp = creation_timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_architecture_id": model.architecture_id,
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "rules_version": RULES_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "action_encoding_version": ACTION_ENCODING_VERSION,
        "policy_action_frame": POLICY_ACTION_FRAME,
        "model_configuration": model.config.to_dict(),
        "state_dict": state_dict,
        "training_iteration": int(training_iteration),
        "training_step": int(training_step),
        "creation_timestamp": timestamp,
        "optimizer_state": dict(optimizer_state) if optimizer_state is not None else None,
        "ema_state": dict(ema_state) if ema_state is not None else None,
        "training_metrics": dict(training_metrics) if training_metrics is not None else None,
        "provenance": {
            # `str(...)`: `torch.__version__` is a `TorchVersion` instance, and a
            # payload holding a non-primitive class cannot be read back under
            # `weights_only=True`.
            "torch_version": str(torch.__version__),
            "initialisation_seed": model.initialisation_seed,
            "parameter_count": model.parameter_count(),
            "state_dict_digest": state_dict_digest(state_dict),
            "note": FIXTURE_NOTE,
        },
    }


def save_checkpoint(model: IntegrationModel, path: "str | Path", **kwargs: Any) -> Path:
    """Write a checkpoint for `model` and return the path.

    Writes to a temporary sibling first and renames, so an interrupted save
    cannot leave a half-written file that a later load would have to guess about.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_checkpoint_payload(model, **kwargs)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _expected_versions() -> dict[str, str]:
    return {
        "model_architecture_id": MODEL_ARCHITECTURE_ID,
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "rules_version": RULES_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "action_encoding_version": ACTION_ENCODING_VERSION,
    }


def validate_checkpoint_payload(payload: Any, *, source: str = "<memory>") -> dict:
    """Check every compatibility field, then the weights. Returns the metadata.

    Order matters: semantics before shapes. Reporting "wrong observation version"
    is useful; reporting "unexpected key" for a checkpoint that was never
    compatible in the first place is not.
    """
    if not isinstance(payload, Mapping):
        raise CheckpointFormatError(
            f"{source}: expected a checkpoint mapping, got {type(payload).__name__}"
        )

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise CheckpointCompatibilityError(
            f"{source}: checkpoint is missing required metadata: {', '.join(sorted(missing))}"
        )
    unexpected = sorted(set(payload) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
    if unexpected:
        raise CheckpointCompatibilityError(
            f"{source}: checkpoint carries unknown field(s): {', '.join(unexpected)}"
        )

    format_version = payload["checkpoint_format_version"]
    if not isinstance(format_version, int) or isinstance(format_version, bool):
        raise CheckpointCompatibilityError(
            f"{source}: checkpoint_format_version must be an integer, got "
            f"{format_version!r}"
        )
    if format_version > CHECKPOINT_FORMAT_VERSION:
        raise CheckpointCompatibilityError(
            f"{source}: checkpoint format version {format_version} was written by newer "
            f"code; this build understands up to {CHECKPOINT_FORMAT_VERSION}. Refusing "
            "to guess at its semantics."
        )
    if format_version < 1:
        raise CheckpointCompatibilityError(
            f"{source}: unknown checkpoint format version {format_version}"
        )

    for field, expected in _expected_versions().items():
        actual = payload[field]
        if actual != expected:
            raise CheckpointCompatibilityError(
                f"{source}: checkpoint {field} is {actual!r} but this build requires "
                f"{expected!r}. The weights may load, but they would be interpreted "
                "under different semantics."
            )

    frame = payload.get("policy_action_frame", POLICY_ACTION_FRAME)
    if frame != POLICY_ACTION_FRAME:
        raise CheckpointCompatibilityError(
            f"{source}: checkpoint policy_action_frame is {frame!r}, this build uses "
            f"{POLICY_ACTION_FRAME!r}; the 10,000 policy indices would mean different moves"
        )

    for field in ("training_iteration", "training_step"):
        value = payload[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CheckpointCompatibilityError(
                f"{source}: {field} must be a non-negative integer, got {value!r}"
            )
    if not isinstance(payload["creation_timestamp"], str) or not payload["creation_timestamp"]:
        raise CheckpointCompatibilityError(f"{source}: creation_timestamp must be a string")

    configuration = payload["model_configuration"]
    if not isinstance(configuration, Mapping):
        raise CheckpointCompatibilityError(
            f"{source}: model_configuration must be a mapping, got "
            f"{type(configuration).__name__}"
        )
    try:
        config = IntegrationModelConfig.from_dict(configuration)
    except Exception as error:  # noqa: BLE001 -- re-raised as a checkpoint failure
        raise CheckpointCompatibilityError(
            f"{source}: model_configuration is not compatible with this build: {error}"
        ) from error

    state_dict = payload["state_dict"]
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise CheckpointFormatError(f"{source}: state_dict is missing or empty")

    reference = IntegrationModel(config).state_dict()
    stored = set(state_dict)
    expected_names = set(reference)
    absent = sorted(expected_names - stored)
    extra = sorted(stored - expected_names)
    if absent:
        raise CheckpointCompatibilityError(
            f"{source}: checkpoint is missing weights: {', '.join(absent[:8])}"
            + (" ..." if len(absent) > 8 else "")
        )
    if extra:
        raise CheckpointCompatibilityError(
            f"{source}: checkpoint carries unexpected weights: {', '.join(extra[:8])}"
            + (" ..." if len(extra) > 8 else "")
        )
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise CheckpointFormatError(
                f"{source}: weight {name!r} is {type(tensor).__name__}, expected a tensor"
            )
        if tuple(tensor.shape) != tuple(reference[name].shape):
            raise CheckpointCompatibilityError(
                f"{source}: weight {name!r} has shape {tuple(tensor.shape)}, the "
                f"configuration implies {tuple(reference[name].shape)}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise CheckpointFormatError(f"{source}: weight {name!r} contains a non-finite value")

    return checkpoint_metadata(payload)


def checkpoint_metadata(payload: Mapping[str, Any]) -> dict:
    """Everything except the weights, as plain JSON-serialisable data."""
    metadata = {field: payload[field] for field in REQUIRED_FIELDS if field != "state_dict"}
    metadata["model_configuration"] = dict(payload["model_configuration"])
    metadata["policy_action_frame"] = payload.get("policy_action_frame", POLICY_ACTION_FRAME)
    metadata["has_optimizer_state"] = payload.get("optimizer_state") is not None
    metadata["has_ema_state"] = payload.get("ema_state") is not None
    metadata["has_training_metrics"] = payload.get("training_metrics") is not None
    provenance = payload.get("provenance")
    metadata["provenance"] = dict(provenance) if isinstance(provenance, Mapping) else {}
    metadata["state_dict_digest"] = state_dict_digest(payload["state_dict"])
    metadata["parameter_tensor_count"] = len(payload["state_dict"])
    return metadata


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def read_checkpoint_payload(path: "str | Path") -> dict:
    """Read and deserialise a checkpoint file, with corruption reported clearly."""
    location = Path(path)
    if not location.exists():
        raise CheckpointFormatError(f"{location}: checkpoint file does not exist")
    if location.stat().st_size == 0:
        raise CheckpointFormatError(f"{location}: checkpoint file is empty")
    try:
        # `weights_only=True` refuses to unpickle arbitrary classes, so a
        # tampered file cannot execute code during a load.
        payload = torch.load(location, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 -- every read failure is one category
        raise CheckpointFormatError(
            f"{location}: checkpoint could not be read (corrupted, truncated or not a "
            f"checkpoint): {type(error).__name__}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise CheckpointFormatError(
            f"{location}: checkpoint holds {type(payload).__name__}, expected a dict"
        )
    return payload


def load_checkpoint(
    path: "str | Path",
    *,
    device: "torch.device | str" = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[IntegrationModel, dict]:
    """Validate a checkpoint file and rebuild the model it describes.

    Returns `(model, metadata)`. The model is in evaluation mode on `device` in
    `dtype`; the weights are always read as the stored float32 and cast after,
    so precision is a run-time choice rather than something baked into a file.
    """
    payload = read_checkpoint_payload(path)
    metadata = validate_checkpoint_payload(payload, source=str(path))

    config = IntegrationModelConfig.from_dict(payload["model_configuration"])
    model = IntegrationModel(config)
    # `strict=True` is the point of the exercise: after the explicit key check
    # above this should be impossible to fail, and if it ever does, it must be
    # an error rather than a partially-initialised network.
    model.load_state_dict(
        {name: tensor.to(torch.float32) for name, tensor in payload["state_dict"].items()},
        strict=True,
    )
    model = model.to(device=torch.device(device), dtype=dtype)
    model.eval()
    metadata["loaded_device"] = str(torch.device(device))
    metadata["loaded_dtype"] = str(dtype)
    metadata["checkpoint_path"] = str(path)
    metadata["checkpoint_file_digest"] = file_digest(path)
    return model, metadata


def payload_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialise a payload to bytes, for corruption tests that truncate a file."""
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    return buffer.getvalue()


def metadata_json(metadata: Mapping[str, Any]) -> str:
    """Stable JSON text of checkpoint metadata, for report artifacts."""
    return json.dumps(metadata, indent=2, sort_keys=True, default=str)


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "OPTIONAL_FIELDS",
    "REQUIRED_FIELDS",
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointFormatError",
    "build_checkpoint_payload",
    "checkpoint_metadata",
    "file_digest",
    "load_checkpoint",
    "metadata_json",
    "payload_bytes",
    "read_checkpoint_payload",
    "save_checkpoint",
    "state_dict_digest",
]
