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

Architectures are registered, not hard-coded
--------------------------------------------
Phase 5 had one network, so this module could name `IntegrationModel` directly.
Phase 6 adds the `stratego_transformer_v1` candidate family, and the choice at
that point is between loosening the architecture check to "any module" or
writing down which architectures exist. It writes them down: see
:func:`register_architecture`. An unregistered `model_architecture_id` is
refused exactly as a wrong rules version is -- the file may be perfectly
well-formed, but this build does not know what its weights mean.

Registration also carries a per-architecture configuration check, which is what
makes *shape*-compatible and *semantics*-compatible different things for a
family whose members can share tensor shapes. Two candidates that differ only in
head count have byte-identical state dicts; only the configuration separates
them, so the configuration is what is compared.

.. code-block:: text

    checkpoint_format_version   int, this module's format generation
    model_architecture_id       "integration_model_v1" | "stratego_transformer_v1"
    model_contract_version      "model_contract_v2"
    rules_version               "stratego_project_v1"
    observation_version         "observation_v2_1_127ch"
    action_encoding_version     "source_destination_10000_v1"
    policy_action_frame         "perspective_normalized_squares"
    engine_action_frame         "absolute_engine_squares"
    model_configuration         dict, the architecture's own configuration fields
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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..engine.constants import OBSERVATION_VERSION, RULES_VERSION
from .architecture_configs import (
    ARCHITECTURE_FAMILY,
    CANDIDATES,
    CandidateConfig,
    is_ladder_candidate,
)
from .base import StrategoModel
from .contract import (
    ACTION_ENCODING_VERSION,
    ENGINE_ACTION_FRAME,
    LEGACY_CONTRACT_V1,
    MODEL_CONTRACT_VERSION,
    POLICY_ACTION_FRAME,
)
from .integration_model import (
    FIXTURE_NOTE,
    MODEL_ARCHITECTURE_ID,
    IntegrationModel,
    IntegrationModelConfig,
)
from .production_model import ProductionModel

#: Current checkpoint generation. A file claiming a *higher* version was written
#: by newer code whose semantics this code cannot know, so it is refused.
CHECKPOINT_FORMAT_VERSION = 1

#: Fields that must be present. `state_dict` is included: a metadata-only file is
#: not a checkpoint.
#:
#: Both frame fields are **required** under `model_contract_v2`. Under v1
#: `policy_action_frame` was optional and defaulted to the running build's frame
#: when absent, which is exactly the silent reinterpretation this phase exists to
#: remove: a file that does not say which frame its 10,000 policy outputs are
#: indexed in cannot be loaded safely, so it is not loaded at all.
REQUIRED_FIELDS: tuple[str, ...] = (
    "checkpoint_format_version",
    "model_architecture_id",
    "model_contract_version",
    "rules_version",
    "observation_version",
    "action_encoding_version",
    "policy_action_frame",
    "engine_action_frame",
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
    "provenance",
)


class CheckpointError(RuntimeError):
    """Base class for every checkpoint failure. Always loud, never repaired."""


class CheckpointFormatError(CheckpointError):
    """The file is unreadable, truncated, or structurally not a checkpoint."""


class CheckpointCompatibilityError(CheckpointError):
    """The file is readable but its semantics do not match this code."""


# ---------------------------------------------------------------------------
# The architecture registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchitectureRegistration:
    """Everything this module needs in order to trust an architecture id.

    `build` is what turns a stored configuration back into a network, and is
    also what produces the *reference* state dict a checkpoint's weights are
    checked against -- so a registration cannot claim an architecture it cannot
    actually construct.
    """

    architecture_id: str
    model_class: type
    config_from_dict: Callable[[Mapping[str, Any]], Any]
    build: Callable[[Any], StrategoModel]
    #: Optional extra identity rule, raising on a configuration that is
    #: well-formed but not one this architecture is allowed to claim.
    check_configuration: "Callable[[Any], None] | None" = None
    description: str = ""


_REGISTRY: dict[str, ArchitectureRegistration] = {}


def register_architecture(registration: ArchitectureRegistration) -> ArchitectureRegistration:
    """Make an architecture loadable. Re-registering the same id is refused.

    Deliberately not idempotent: a silent overwrite would mean two modules
    disagreeing about what an id means, and the loser would be whichever
    imported first -- an import-order-dependent semantics change, which is the
    exact class of bug this whole module exists to prevent.
    """
    existing = _REGISTRY.get(registration.architecture_id)
    if existing is not None and existing is not registration:
        raise CheckpointError(
            f"architecture id {registration.architecture_id!r} is already registered to "
            f"{existing.model_class.__name__}"
        )
    _REGISTRY[registration.architecture_id] = registration
    return registration


def registered_architectures() -> tuple[str, ...]:
    """Every architecture id this build can load, in registration order."""
    return tuple(_REGISTRY)


def architecture_registration(architecture_id: Any) -> ArchitectureRegistration:
    """Look up a registration, or refuse the id with a listing of the known ones."""
    if not isinstance(architecture_id, str) or architecture_id not in _REGISTRY:
        raise CheckpointCompatibilityError(
            f"unknown model_architecture_id {architecture_id!r}; this build can load "
            f"{', '.join(registered_architectures())}. The weights may load, but they "
            "would be interpreted under different semantics."
        )
    return _REGISTRY[architecture_id]


def _check_candidate_configuration(config: CandidateConfig) -> None:
    """A checkpoint claiming a ladder candidate must carry that candidate's shape.

    Without this, a file could say `candidate_id: "C3"` while carrying C2's
    dimensions, and every later report would attribute C2's numbers to C3. The
    `CandidateConfig` validators already reject an *impossible* configuration;
    this rejects a *dishonest* one.
    """
    if not is_ladder_candidate(config.candidate_id):
        return
    expected = CANDIDATES[config.candidate_id]
    if config != expected:
        raise CheckpointCompatibilityError(
            f"checkpoint claims candidate {config.candidate_id!r} but its configuration is "
            f"{config.describe()}, while this build's {config.candidate_id} is "
            f"{expected.describe()}"
        )


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
    model: StrategoModel,
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
    if not isinstance(model, StrategoModel):
        raise CheckpointError(
            f"expected a StrategoModel, got {type(model).__name__}"
        )
    registration = architecture_registration(model.architecture_id)
    if not isinstance(model, registration.model_class):
        raise CheckpointError(
            f"model claims architecture id {model.architecture_id!r}, which is registered to "
            f"{registration.model_class.__name__}, but it is a {type(model).__name__}"
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
        "engine_action_frame": ENGINE_ACTION_FRAME,
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
            # Each architecture states its own status. The Phase 5 fixture says
            # it is a fixture; a Phase 6 candidate says it is untrained. Neither
            # claim should be written by this module on the model's behalf.
            "note": str(model.architecture_summary().get("note", "")),
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
    """The fields whose value must match this build exactly.

    `model_architecture_id` is deliberately absent: it is not a single expected
    string any more but a registry lookup, checked separately so the message can
    list what this build *does* know.
    """
    return {
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "rules_version": RULES_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "action_encoding_version": ACTION_ENCODING_VERSION,
    }


def _expected_frames() -> dict[str, str]:
    return {
        "policy_action_frame": POLICY_ACTION_FRAME,
        "engine_action_frame": ENGINE_ACTION_FRAME,
    }


def looks_like_contract_v1(payload: Mapping[str, Any]) -> bool:
    """Whether a payload advertises the retired `model_contract_v1` semantics.

    Recognition only. There is no v1 loading path and this never makes a file
    loadable; it exists so the rejection message can say *why* a perfectly
    well-formed Phase 5 checkpoint is being refused, instead of leaving someone
    to guess at a bare "field mismatch".
    """
    return (
        payload.get("model_contract_version") == LEGACY_CONTRACT_V1["model_contract_version"]
        or payload.get("policy_action_frame") == LEGACY_CONTRACT_V1["policy_action_frame"]
    )


def _legacy_note(payload: Mapping[str, Any]) -> str:
    """The explanatory tail appended to a v1 rejection, or an empty string."""
    if not looks_like_contract_v1(payload):
        return ""
    return (
        " This is a model_contract_v1 checkpoint: its policy head is indexed in "
        f"{LEGACY_CONTRACT_V1['policy_action_frame']}, while this build indexes the same "
        f"10,000 outputs in {POLICY_ACTION_FRAME}. The weights would load and the network "
        "would play the 180-degree-wrong move for blue, so the file is refused. Retrain or "
        "regenerate the checkpoint under model_contract_v2; there is no in-place conversion, "
        "because the mapping between the frames is not a permutation of the *weights*."
    )


def accepted_under_contract_v1(payload: Any) -> bool:
    """Whether a `model_contract_v1` build would have accepted this payload.

    A frozen replica of the v1 acceptance rule, kept for one purpose: proving
    that the v1/v2 incompatibility is symmetric. It is easy to show that v2
    refuses v1 files, because this build does it; showing that a v1 build would
    have refused *these* files needs the old rule written down, since the old
    build no longer exists. Deliberately a pure predicate over metadata -- it
    reads no weights and can load nothing.
    """
    if not isinstance(payload, Mapping):
        return False
    v1_frame_fields = ("policy_action_frame", "engine_action_frame")
    v1_required = tuple(field for field in REQUIRED_FIELDS if field not in v1_frame_fields)
    if any(field not in payload for field in v1_required):
        return False
    # v1 knew nothing about `engine_action_frame`, and refused unknown fields --
    # which is what makes a v2 file unloadable there even before the frame check.
    v1_known = set(v1_required) | {
        "optimizer_state",
        "ema_state",
        "training_metrics",
        "policy_action_frame",
        "provenance",
    }
    if set(payload) - v1_known:
        return False
    if payload.get("model_contract_version") != LEGACY_CONTRACT_V1["model_contract_version"]:
        return False
    # v1 defaulted a missing frame to its own, which is the hole v2 closed.
    frame = payload.get("policy_action_frame", LEGACY_CONTRACT_V1["policy_action_frame"])
    return frame == LEGACY_CONTRACT_V1["policy_action_frame"]


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
            + _legacy_note(payload)
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

    # Frames before every other semantic field, and long before any shape check.
    # A frame mismatch is the failure that would otherwise be *invisible*: the
    # tensors are the right shape, the weights load, and the network simply
    # plays mirrored moves for one colour. Naming it first means the error a
    # reader sees is the error that actually matters.
    for field, expected in _expected_frames().items():
        actual = payload[field]
        if actual != expected:
            raise CheckpointCompatibilityError(
                f"{source}: checkpoint {field} is {actual!r}, this build uses {expected!r}; "
                "the 10,000 policy indices would mean different moves."
                + _legacy_note(payload)
            )

    for field, expected in _expected_versions().items():
        actual = payload[field]
        if actual != expected:
            raise CheckpointCompatibilityError(
                f"{source}: checkpoint {field} is {actual!r} but this build requires "
                f"{expected!r}. The weights may load, but they would be interpreted "
                "under different semantics." + _legacy_note(payload)
            )

    for field in ("training_iteration", "training_step"):
        value = payload[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CheckpointCompatibilityError(
                f"{source}: {field} must be a non-negative integer, got {value!r}"
            )
    if not isinstance(payload["creation_timestamp"], str) or not payload["creation_timestamp"]:
        raise CheckpointCompatibilityError(f"{source}: creation_timestamp must be a string")

    # Which architecture, and therefore which configuration class and which
    # reference weights. An unregistered id is refused here rather than being
    # allowed to reach a shape comparison it might coincidentally survive.
    try:
        registration = architecture_registration(payload["model_architecture_id"])
    except CheckpointCompatibilityError as error:
        raise CheckpointCompatibilityError(f"{source}: {error}" + _legacy_note(payload)) from None

    configuration = payload["model_configuration"]
    if not isinstance(configuration, Mapping):
        raise CheckpointCompatibilityError(
            f"{source}: model_configuration must be a mapping, got "
            f"{type(configuration).__name__}"
        )
    try:
        config = registration.config_from_dict(configuration)
        if registration.check_configuration is not None:
            registration.check_configuration(config)
    except CheckpointCompatibilityError as error:
        raise CheckpointCompatibilityError(f"{source}: {error}") from None
    except Exception as error:  # noqa: BLE001 -- re-raised as a checkpoint failure
        raise CheckpointCompatibilityError(
            f"{source}: model_configuration is not compatible with this build: {error}"
        ) from error

    state_dict = payload["state_dict"]
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise CheckpointFormatError(f"{source}: state_dict is missing or empty")

    reference = registration.build(config).state_dict()
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


def check_expected_identity(
    metadata: Mapping[str, Any],
    *,
    architecture_id: str | None = None,
    configuration: Any = None,
    source: str = "<memory>",
) -> None:
    """Refuse a checkpoint that is valid but is not the one the caller wanted.

    The gate that matters for a *family*. Validation proves a file is
    self-consistent; it cannot know which candidate the caller is holding. Two
    candidates differing only in head count have identical tensor shapes, so
    `load_state_dict` would succeed and every number reported afterwards would
    be attributed to the wrong architecture. Configurations are compared, not
    shapes, because only the configuration can tell them apart.
    """
    if architecture_id is not None and metadata["model_architecture_id"] != architecture_id:
        raise CheckpointCompatibilityError(
            f"{source}: checkpoint is {metadata['model_architecture_id']!r} but the caller "
            f"expected {architecture_id!r}"
        )
    if configuration is None:
        return
    expected = configuration.to_dict() if hasattr(configuration, "to_dict") else dict(configuration)
    actual = dict(metadata["model_configuration"])
    if actual != expected:
        differing = sorted(
            key
            for key in set(expected) | set(actual)
            if expected.get(key) != actual.get(key)
        )
        detail = ", ".join(
            f"{key}: checkpoint {actual.get(key)!r} vs expected {expected.get(key)!r}"
            for key in differing
        )
        raise CheckpointCompatibilityError(
            f"{source}: checkpoint configuration does not match the expected one ({detail}). "
            "Tensor shapes may still be compatible; the weights would be a different "
            "architecture wearing the right shapes."
        )


def load_checkpoint(
    path: "str | Path",
    *,
    device: "torch.device | str" = "cpu",
    dtype: torch.dtype = torch.float32,
    expected_architecture_id: str | None = None,
    expected_configuration: Any = None,
) -> tuple[StrategoModel, dict]:
    """Validate a checkpoint file and rebuild the model it describes.

    Returns `(model, metadata)`. The model is in evaluation mode on `device` in
    `dtype`; the weights are always read as the stored float32 and cast after,
    so precision is a run-time choice rather than something baked into a file.

    Pass `expected_architecture_id` / `expected_configuration` when the caller
    already knows which architecture it is asking for -- Agent 3 reloading a
    specific candidate, say. Without them the file simply describes itself, and
    a self-consistent file always rebuilds the network it was written from.
    """
    payload = read_checkpoint_payload(path)
    metadata = validate_checkpoint_payload(payload, source=str(path))
    check_expected_identity(
        metadata,
        architecture_id=expected_architecture_id,
        configuration=expected_configuration,
        source=str(path),
    )

    registration = architecture_registration(payload["model_architecture_id"])
    config = registration.config_from_dict(payload["model_configuration"])
    model = registration.build(config)
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


def load_checkpoint_into(model: StrategoModel, path: "str | Path") -> dict:
    """Load weights into an *existing* model, refusing a different architecture.

    The identity check is not optional here, because a target exists: the
    caller is holding a specific candidate and asking for its weights, so a file
    describing a different one is a mistake even when every tensor shape agrees.
    """
    payload = read_checkpoint_payload(path)
    metadata = validate_checkpoint_payload(payload, source=str(path))
    check_expected_identity(
        metadata,
        architecture_id=model.architecture_id,
        configuration=model.config,
        source=str(path),
    )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.load_state_dict(
        {
            name: tensor.to(device=device, dtype=dtype)
            for name, tensor in payload["state_dict"].items()
        },
        strict=True,
    )
    metadata["checkpoint_path"] = str(path)
    metadata["checkpoint_file_digest"] = file_digest(path)
    return metadata


def payload_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialise a payload to bytes, for corruption tests that truncate a file."""
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    return buffer.getvalue()


def metadata_json(metadata: Mapping[str, Any]) -> str:
    """Stable JSON text of checkpoint metadata, for report artifacts."""
    return json.dumps(metadata, indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# The architectures this build knows
# ---------------------------------------------------------------------------
#
# Registered here rather than in each model module, so that importing this
# module is enough to make every shipped architecture loadable -- a checkpoint
# whose loadability depended on which model module the caller happened to import
# first would be a semantics change disguised as an import.

INTEGRATION_MODEL_V1 = register_architecture(
    ArchitectureRegistration(
        architecture_id=MODEL_ARCHITECTURE_ID,
        model_class=IntegrationModel,
        config_from_dict=IntegrationModelConfig.from_dict,
        build=IntegrationModel,
        description=FIXTURE_NOTE,
    )
)

STRATEGO_TRANSFORMER_V1 = register_architecture(
    ArchitectureRegistration(
        architecture_id=ARCHITECTURE_FAMILY,
        model_class=ProductionModel,
        config_from_dict=CandidateConfig.from_dict,
        build=ProductionModel,
        check_configuration=_check_candidate_configuration,
        description=(
            "Phase 6 candidate Transformer family. One implementation; C0-C6 differ only "
            "in width, blocks, heads and feed-forward width."
        ),
    )
)


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "INTEGRATION_MODEL_V1",
    "OPTIONAL_FIELDS",
    "REQUIRED_FIELDS",
    "STRATEGO_TRANSFORMER_V1",
    "ArchitectureRegistration",
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointFormatError",
    "architecture_registration",
    "check_expected_identity",
    "load_checkpoint_into",
    "register_architecture",
    "registered_architectures",
    "accepted_under_contract_v1",
    "build_checkpoint_payload",
    "checkpoint_metadata",
    "file_digest",
    "load_checkpoint",
    "looks_like_contract_v1",
    "metadata_json",
    "payload_bytes",
    "read_checkpoint_payload",
    "save_checkpoint",
    "state_dict_digest",
]
