"""Phase 8 Agent 4: `warmstart_checkpoint_v1` — save, validate, resume.

Specification sources:

- `04_AGENT_4_TRAINER_AND_RESUME.md` ("Checkpoint contents", "Interrupted
  checkpoint writes")
- `00_PHASE_8_SEQUENCE_AND_COMMON_CONTRACT.md` section 23 (checkpoint/resume
  contract)
- Agent 4 supplementary review instruction (corpus identity by digest, never
  by storage path)

The corpus is a set of digests, not a directory
-----------------------------------------------
A checkpoint names its training corpus by **version and accepted content /
metadata / commit-index digests** (:class:`CorpusIdentity`). The resolved
filesystem root is recorded only inside the non-semantic ``diagnostics``
block, and no compatibility check ever reads it: a pure relocation of the
corpus bytes with identical digests defines the *same* training corpus and
must leave every checkpoint resumable, while a corpus whose digests differ is
a different corpus no matter where it sits. Live verification against the
accepted digests is :func:`verify_corpus_identity`, and a mismatch there is a
stop condition — the corpus is never regenerated or repaired by this module.

Atomic writes
-------------
A save goes ``temporary file -> flush -> fsync -> reload-and-validate the
temporary bytes -> os.replace -> fsync directory``. A crash at any boundary
leaves either the previous complete checkpoint or the new complete one; the
``crash_hook`` argument exists so the tests can force a crash at each boundary
and prove that claim against the real filesystem sequence.

Every load is a validation
--------------------------
Structure, versions, the embedded model payload (through the frozen
:func:`stratego.model.checkpoint.validate_checkpoint_payload`), and a
whole-payload integrity digest are checked before any state is handed back.
Resume additionally requires exact train-config and corpus-identity equality.
The one sanctioned relaxation is :func:`load_model_for_evaluation`, which
validates the embedded model checkpoint and returns the network without
requiring the caller's run to be the checkpoint's run — an explicit
evaluation-only path, never a resume.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ..model.checkpoint import (
    CheckpointError,
    build_checkpoint_payload,
    validate_checkpoint_payload,
)
from ..model.production_model import ProductionModel, build_candidate_model
from .corpus_commit import audit_commit_integrity, corpus_content_digest
from .synthetic_corpus import _commit_index_digest, _metadata_digest
from .warmstart_contract import WARMSTART_EXAMPLE_VERSION
from .warmstart_dataset import (
    DATA_CURSOR_VERSION,
    TRAIN_ORDER_VERSION,
    DataCursor,
)
from .warmstart_seed import CORPUS_SPLITS, DECISION_SAMPLER_VERSION, SYNTHETIC_CORPUS_VERSION

#: The Phase 8 resumable-checkpoint schema. Any change to a field's meaning is
#: a new version after review, never an in-place edit.
WARMSTART_CHECKPOINT_VERSION = "warmstart_checkpoint_v1"

#: The Phase 8 trainer identity carried inside every checkpoint. Lives here
#: (not in the trainer module) so the checkpoint schema owns every version it
#: rejects mismatches against.
WARMSTART_TRAINER_VERSION = "warmstart_trainer_v1"

#: Every key a `warmstart_checkpoint_v1` payload must carry.
REQUIRED_KEYS = (
    "warmstart_checkpoint_version",
    "trainer_version",
    "model",
    "train_config",
    "train_config_digest",
    "corpus_identity",
    "example_version",
    "train_order_version",
    "data_cursor_version",
    "sampler_version",
    "optimizer_state",
    "scheduler_state",
    "global_step",
    "examples_consumed",
    "data_cursor",
    "best_validation",
    "validation_history",
    "rng",
    "diagnostics",
    "integrity_digest",
)


class WarmstartCheckpointError(RuntimeError):
    """Base class for every warm-start checkpoint failure. Always loud."""


class WarmstartCheckpointFormatError(WarmstartCheckpointError):
    """The file is unreadable, truncated, corrupted, or structurally wrong."""


class WarmstartCheckpointCompatibilityError(WarmstartCheckpointError):
    """The file is intact but does not belong to the caller's logical run."""


class WarmstartCorpusMismatchError(WarmstartCheckpointError):
    """The live corpus digests differ from the accepted identity: BLOCKED."""


# ---------------------------------------------------------------------------
# Corpus identity — version plus digests, never a path
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusIdentity:
    """What "the training corpus" means to a checkpoint.

    Deliberately holds no filesystem path: two roots with these digests hold
    the same corpus, and one root whose digests drifted does not hold this
    corpus at all.
    """

    corpus_version: str
    content_digest: str
    metadata_digest: str
    commit_index_digest: str

    def __post_init__(self) -> None:
        for name in ("content_digest", "metadata_digest", "commit_index_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise WarmstartCheckpointError(
                    f"{name} must be a 64-hex-character SHA-256, got {value!r}"
                )

    def to_dict(self) -> dict:
        return {
            "corpus_version": self.corpus_version,
            "content_digest": self.content_digest,
            "metadata_digest": self.metadata_digest,
            "commit_index_digest": self.commit_index_digest,
        }

    @staticmethod
    def from_dict(payload: dict) -> "CorpusIdentity":
        try:
            return CorpusIdentity(
                corpus_version=str(payload["corpus_version"]),
                content_digest=str(payload["content_digest"]),
                metadata_digest=str(payload["metadata_digest"]),
                commit_index_digest=str(payload["commit_index_digest"]),
            )
        except KeyError as error:
            raise WarmstartCheckpointError(
                f"corpus identity is missing field {error}"
            ) from None


def measure_corpus_identity(
    root: "str | Path", *, splits: "tuple[str, ...]" = CORPUS_SPLITS
) -> CorpusIdentity:
    """The live digests of the corpus at `root`, as an identity."""
    root = Path(root)
    return CorpusIdentity(
        corpus_version=SYNTHETIC_CORPUS_VERSION,
        content_digest=corpus_content_digest(root, splits),
        metadata_digest=_metadata_digest(root, splits),
        commit_index_digest=_commit_index_digest(root, splits),
    )


def verify_corpus_identity(
    root: "str | Path",
    expected: "CorpusIdentity | dict | None" = None,
    *,
    splits: "tuple[str, ...]" = CORPUS_SPLITS,
    check_payload_bytes: bool = True,
) -> CorpusIdentity:
    """Measure the corpus at `root` and require it to be `expected`.

    The three digests are journal- and metadata-derived; the trajectory
    payload *bytes* are pinned by the per-game digests the journals record,
    so `check_payload_bytes=True` (the default) additionally re-reads every
    payload and metadata record against its committed digest through the
    frozen `audit_commit_integrity`. Skipping that read is a test-only
    shortcut, never a production verification.

    With `expected=None` the measurement itself becomes the identity (the
    mini-corpus test path). With an expected identity, any difference raises
    :class:`WarmstartCorpusMismatchError` — the BLOCKED condition; the caller
    must never respond by regenerating or repairing corpus bytes.
    """
    observed = measure_corpus_identity(root, splits=splits)
    if check_payload_bytes:
        integrity = audit_commit_integrity(root, splits)
        violations = {
            name: len(entries)
            for name, entries in integrity.items()
            if isinstance(entries, list) and entries
        }
        if violations:
            raise WarmstartCorpusMismatchError(
                f"the corpus at {root} fails byte-level integrity against its own "
                f"commit journals ({violations}). This is a stop condition: never "
                "regenerate or repair the corpus."
            )
    if expected is None:
        return observed
    if isinstance(expected, dict):
        expected = CorpusIdentity.from_dict(expected)
    if observed != expected:
        differing = [
            f"{name}: observed {getattr(observed, name)!r}, accepted "
            f"{getattr(expected, name)!r}"
            for name in (
                "corpus_version",
                "content_digest",
                "metadata_digest",
                "commit_index_digest",
            )
            if getattr(observed, name) != getattr(expected, name)
        ]
        raise WarmstartCorpusMismatchError(
            f"the corpus at {root} is not the accepted corpus ({'; '.join(differing)}). "
            "This is a stop condition: correct the resolver/pointer configuration if "
            "it names the wrong location, and never regenerate or repair the corpus."
        )
    return observed


# ---------------------------------------------------------------------------
# Integrity digest
# ---------------------------------------------------------------------------


def _digest_tree(value, hasher, path: str) -> None:
    """Deterministic traversal hash over an arbitrary payload tree.

    Dictionaries are visited in sorted-key order, sequences in index order,
    tensors as shape/dtype/bytes; every node folds its path in, so moving a
    value between fields changes the digest even when the bytes agree.
    """
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            _digest_tree(value[key], hasher, f"{path}/{key}")
    elif isinstance(value, (list, tuple)):
        hasher.update(f"{path}#len={len(value)}".encode())
        for index, entry in enumerate(value):
            _digest_tree(entry, hasher, f"{path}[{index}]")
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().to("cpu")
        hasher.update(
            f"{path}|tensor|{tuple(tensor.shape)}|{tensor.dtype}".encode()
        )
        hasher.update(tensor.contiguous().numpy().tobytes())
    else:
        hasher.update(f"{path}|{type(value).__name__}|{value!r}".encode())


def payload_integrity_digest(payload: dict) -> str:
    """SHA-256 over every field except `integrity_digest` itself."""
    hasher = hashlib.sha256()
    _digest_tree(
        {key: value for key, value in payload.items() if key != "integrity_digest"},
        hasher,
        "",
    )
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------


def _state_tree_to_cpu(value):
    """A deep copy of an optimizer/scheduler state tree with tensors on CPU."""
    if isinstance(value, dict):
        return {key: _state_tree_to_cpu(entry) for key, entry in value.items()}
    if isinstance(value, list):
        return [_state_tree_to_cpu(entry) for entry in value]
    if isinstance(value, tuple):
        return tuple(_state_tree_to_cpu(entry) for entry in value)
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    return value


def collect_rng_state(device: str) -> dict:
    """Every RNG state the contract names, with unused streams stated as such.

    The trainer consumes no Python or NumPy randomness (every order is a pure
    function of the frozen cursor) and C1 has no dropout, so nothing draws
    from the torch generators either; the torch states are captured anyway so
    the checkpoint records the environment completely.
    """
    rng = {
        "python": "unused: no trainer stream draws from the Python RNG",
        "numpy": "unused: no trainer stream draws from the NumPy RNG",
        "torch_cpu": torch.get_rng_state(),
        "torch_mps": None,
        "note": (
            "train order, decision sampling and validation order are pure "
            "functions of frozen seeds and the data cursor"
        ),
    }
    if str(device).startswith("mps") and hasattr(torch, "mps"):
        try:
            rng["torch_mps"] = torch.mps.get_rng_state()
        except Exception:  # noqa: BLE001 - absence of an MPS generator is not a failure
            rng["torch_mps"] = None
    return rng


def build_warmstart_checkpoint_payload(
    *,
    model: ProductionModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_config: dict,
    train_config_digest: str,
    corpus_identity: CorpusIdentity,
    cursor: DataCursor,
    global_step: int,
    examples_consumed: int,
    best_validation: dict,
    validation_history: list,
    diagnostics: dict,
) -> dict:
    """Assemble one complete `warmstart_checkpoint_v1` payload in memory.

    `diagnostics` carries the resolved corpus root, device, topology, wall
    clock, source revision and software versions; it is recorded verbatim and
    never consulted by any compatibility check.
    """
    payload = {
        "warmstart_checkpoint_version": WARMSTART_CHECKPOINT_VERSION,
        "trainer_version": WARMSTART_TRAINER_VERSION,
        "model": build_checkpoint_payload(model, training_step=int(global_step)),
        "train_config": dict(train_config),
        "train_config_digest": str(train_config_digest),
        "corpus_identity": corpus_identity.to_dict(),
        "example_version": WARMSTART_EXAMPLE_VERSION,
        "train_order_version": TRAIN_ORDER_VERSION,
        "data_cursor_version": DATA_CURSOR_VERSION,
        "sampler_version": DECISION_SAMPLER_VERSION,
        "optimizer_state": _state_tree_to_cpu(optimizer.state_dict()),
        "scheduler_state": _state_tree_to_cpu(scheduler.state_dict()),
        "global_step": int(global_step),
        "examples_consumed": int(examples_consumed),
        "data_cursor": cursor.to_dict(),
        "best_validation": dict(best_validation),
        "validation_history": [dict(entry) for entry in validation_history],
        "rng": collect_rng_state(str(diagnostics.get("device", "cpu"))),
        "diagnostics": dict(diagnostics),
    }
    payload["integrity_digest"] = payload_integrity_digest(payload)
    return payload


# ---------------------------------------------------------------------------
# Atomic writing
# ---------------------------------------------------------------------------


def save_warmstart_checkpoint(
    payload: dict,
    path: "str | Path",
    *,
    fsync: bool = True,
    crash_hook=None,
) -> dict:
    """Write one checkpoint atomically and prove the written bytes load.

    ```text
    write .partial -> flush+fsync -> reload .partial and fully validate
        -> os.replace -> fsync directory
    ```

    `crash_hook(stage)` (tests only) is invoked at ``after_write``,
    ``after_validate`` and ``after_commit``; raising inside it simulates a
    crash at that exact boundary.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    started = time.perf_counter()
    with open(temporary, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())
    if crash_hook is not None:
        crash_hook("after_write")
    # Validate the bytes actually on disk, not the payload in memory: this is
    # the step that turns "rename is atomic" into "a renamed file is valid".
    reloaded = read_warmstart_payload(temporary)
    validate_warmstart_payload(reloaded, source=str(temporary))
    if crash_hook is not None:
        crash_hook("after_validate")
    os.replace(temporary, destination)
    if fsync:
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    if crash_hook is not None:
        crash_hook("after_commit")
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "seconds": time.perf_counter() - started,
        "integrity_digest": payload["integrity_digest"],
    }


# ---------------------------------------------------------------------------
# Reading and validation
# ---------------------------------------------------------------------------


def read_warmstart_payload(path: "str | Path") -> dict:
    """Deserialize a checkpoint file, reporting corruption as a format error."""
    location = Path(path)
    if not location.exists():
        raise WarmstartCheckpointFormatError(f"{location}: checkpoint does not exist")
    if location.stat().st_size == 0:
        raise WarmstartCheckpointFormatError(f"{location}: checkpoint file is empty")
    try:
        payload = torch.load(location, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - every unreadable file is one category
        raise WarmstartCheckpointFormatError(
            f"{location}: checkpoint could not be read (corrupted, truncated or not "
            f"a checkpoint): {type(error).__name__}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise WarmstartCheckpointFormatError(
            f"{location}: checkpoint holds {type(payload).__name__}, expected a dict"
        )
    return payload


def validate_warmstart_payload(payload: dict, *, source: str = "<memory>") -> dict:
    """Structure, versions, integrity digest and the embedded model payload.

    Returns JSON-safe metadata. Deliberately *not* a resume authorization:
    a payload can be perfectly valid and still belong to a different run —
    that separation is :func:`check_resume_identity`'s job.
    """
    if not isinstance(payload, dict):
        raise WarmstartCheckpointFormatError(
            f"{source}: expected a checkpoint mapping, got {type(payload).__name__}"
        )
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise WarmstartCheckpointFormatError(
            f"{source}: checkpoint is missing required field(s): {', '.join(missing)}"
        )
    unexpected = sorted(set(payload) - set(REQUIRED_KEYS))
    if unexpected:
        raise WarmstartCheckpointFormatError(
            f"{source}: checkpoint carries unknown field(s): {', '.join(unexpected)}"
        )

    version = payload["warmstart_checkpoint_version"]
    if version != WARMSTART_CHECKPOINT_VERSION:
        raise WarmstartCheckpointCompatibilityError(
            f"{source}: checkpoint version {version!r} is not "
            f"{WARMSTART_CHECKPOINT_VERSION!r}; refusing to guess at its semantics"
        )
    trainer = payload["trainer_version"]
    if trainer != WARMSTART_TRAINER_VERSION:
        raise WarmstartCheckpointCompatibilityError(
            f"{source}: trainer version {trainer!r} is not {WARMSTART_TRAINER_VERSION!r}"
        )
    fixed_versions = {
        "example_version": WARMSTART_EXAMPLE_VERSION,
        "train_order_version": TRAIN_ORDER_VERSION,
        "data_cursor_version": DATA_CURSOR_VERSION,
        "sampler_version": DECISION_SAMPLER_VERSION,
    }
    for field, expected in fixed_versions.items():
        if payload[field] != expected:
            raise WarmstartCheckpointCompatibilityError(
                f"{source}: {field} is {payload[field]!r}, this build uses {expected!r}"
            )

    recorded = payload["integrity_digest"]
    recomputed = payload_integrity_digest(payload)
    if recorded != recomputed:
        raise WarmstartCheckpointFormatError(
            f"{source}: integrity digest mismatch (recorded {recorded!r}, recomputed "
            f"{recomputed!r}); the file's content was altered after it was written"
        )

    for field in ("global_step", "examples_consumed"):
        value = payload[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise WarmstartCheckpointFormatError(
                f"{source}: {field} must be a non-negative integer, got {value!r}"
            )

    try:
        cursor = DataCursor.from_dict(payload["data_cursor"])
    except Exception as error:  # noqa: BLE001 - cursor problems are compatibility problems
        raise WarmstartCheckpointCompatibilityError(
            f"{source}: data cursor cannot be restored: {error}"
        ) from error

    try:
        model_metadata = validate_checkpoint_payload(payload["model"], source=source)
    except CheckpointError as error:
        raise WarmstartCheckpointCompatibilityError(
            f"{source}: embedded model checkpoint is invalid: {error}"
        ) from error

    identity = CorpusIdentity.from_dict(payload["corpus_identity"])
    if not isinstance(payload["optimizer_state"], dict) or not payload["optimizer_state"]:
        raise WarmstartCheckpointFormatError(f"{source}: optimizer_state is missing or empty")
    if not isinstance(payload["scheduler_state"], dict):
        raise WarmstartCheckpointFormatError(f"{source}: scheduler_state must be a mapping")
    if not isinstance(payload["validation_history"], list):
        raise WarmstartCheckpointFormatError(f"{source}: validation_history must be a list")

    return {
        "warmstart_checkpoint_version": WARMSTART_CHECKPOINT_VERSION,
        "trainer_version": WARMSTART_TRAINER_VERSION,
        "train_config": dict(payload["train_config"]),
        "train_config_digest": payload["train_config_digest"],
        "corpus_identity": identity.to_dict(),
        "global_step": payload["global_step"],
        "examples_consumed": payload["examples_consumed"],
        "data_cursor": cursor.to_dict(),
        "best_validation": dict(payload["best_validation"]),
        "validation_entries": len(payload["validation_history"]),
        "integrity_digest": payload["integrity_digest"],
        "model": model_metadata,
        "diagnostics": dict(payload["diagnostics"]),
    }


def check_resume_identity(
    payload: dict,
    *,
    expected_train_config: dict,
    expected_train_config_digest: str,
    expected_corpus_identity: CorpusIdentity,
    source: str = "<memory>",
) -> None:
    """Refuse a valid checkpoint that is not the caller's logical run.

    Config digests are compared first for the loud headline, then the full
    dictionaries so the message can name exactly which field drifted. Corpus
    identity is digest equality — a moved corpus with identical digests passes
    by construction, a regenerated one cannot.
    """
    if payload["train_config_digest"] != expected_train_config_digest:
        stored = dict(payload["train_config"])
        expected = dict(expected_train_config)
        differing = sorted(
            key
            for key in set(stored) | set(expected)
            if stored.get(key) != expected.get(key)
        )
        detail = ", ".join(
            f"{key}: checkpoint {stored.get(key)!r} vs run {expected.get(key)!r}"
            for key in differing
        ) or "digests differ but the serialized fields agree; the digest rule changed"
        raise WarmstartCheckpointCompatibilityError(
            f"{source}: checkpoint belongs to a different training configuration "
            f"({detail}); resuming would silently change the run's identity"
        )
    stored_identity = CorpusIdentity.from_dict(payload["corpus_identity"])
    if stored_identity != expected_corpus_identity:
        differing = [
            name
            for name in (
                "corpus_version",
                "content_digest",
                "metadata_digest",
                "commit_index_digest",
            )
            if getattr(stored_identity, name) != getattr(expected_corpus_identity, name)
        ]
        raise WarmstartCheckpointCompatibilityError(
            f"{source}: checkpoint was trained on a different corpus "
            f"(fields: {', '.join(differing)}); refusing to silently resume on it"
        )


def load_warmstart_checkpoint(
    path: "str | Path",
    *,
    expected_train_config: dict,
    expected_train_config_digest: str,
    expected_corpus_identity: CorpusIdentity,
    device: "torch.device | str" = "cpu",
) -> dict:
    """Validate, identity-check and materialize one resumable checkpoint.

    Returns every piece the trainer needs to continue the exact logical run:

    ```text
    model                the rebuilt network, train mode, float32 on `device`
    optimizer_state      CPU state tree for Optimizer.load_state_dict
    scheduler_state      state for the rebuilt scheduler
    cursor               the DataCursor of the next unserved batch
    global_step / examples_consumed / best_validation / validation_history
    metadata             JSON-safe summary including diagnostics
    ```
    """
    payload = read_warmstart_payload(path)
    metadata = validate_warmstart_payload(payload, source=str(path))
    check_resume_identity(
        payload,
        expected_train_config=expected_train_config,
        expected_train_config_digest=expected_train_config_digest,
        expected_corpus_identity=expected_corpus_identity,
        source=str(path),
    )
    model_payload = payload["model"]
    model = build_candidate_model(
        model_payload["model_configuration"]["candidate_id"],
        seed=int(model_payload["provenance"]["initialisation_seed"]),
        device="cpu",
    )
    model.load_state_dict(
        {
            name: tensor.to(torch.float32)
            for name, tensor in model_payload["state_dict"].items()
        },
        strict=True,
    )
    model = model.to(device=torch.device(device), dtype=torch.float32)
    model.train()

    rng = payload["rng"]
    if isinstance(rng.get("torch_cpu"), torch.Tensor):
        torch.set_rng_state(rng["torch_cpu"].to(torch.uint8).cpu())
    if (
        isinstance(rng.get("torch_mps"), torch.Tensor)
        and str(device).startswith("mps")
        and hasattr(torch, "mps")
    ):
        torch.mps.set_rng_state(rng["torch_mps"].to(torch.uint8).cpu())

    return {
        "model": model,
        "optimizer_state": payload["optimizer_state"],
        "scheduler_state": payload["scheduler_state"],
        "cursor": DataCursor.from_dict(payload["data_cursor"]),
        "global_step": int(payload["global_step"]),
        "examples_consumed": int(payload["examples_consumed"]),
        "best_validation": dict(payload["best_validation"]),
        "validation_history": [dict(entry) for entry in payload["validation_history"]],
        "metadata": metadata,
    }


def load_model_for_evaluation(
    path: "str | Path", *, device: "torch.device | str" = "cpu"
) -> tuple:
    """The explicit evaluation-only load path: `(model, metadata)`.

    Validates the whole file (structure, integrity digest, embedded model
    contract) but deliberately skips the train-config and corpus-identity
    resume checks: evaluating a compatible model does not require being the
    run that produced it. The model comes back in eval mode and can never be
    mistaken for a resumed trainer because no optimizer or cursor state is
    returned.
    """
    payload = read_warmstart_payload(path)
    metadata = validate_warmstart_payload(payload, source=str(path))
    model_payload = payload["model"]
    model = build_candidate_model(
        model_payload["model_configuration"]["candidate_id"],
        seed=int(model_payload["provenance"]["initialisation_seed"]),
        device="cpu",
    )
    model.load_state_dict(
        {
            name: tensor.to(torch.float32)
            for name, tensor in model_payload["state_dict"].items()
        },
        strict=True,
    )
    model = model.to(device=torch.device(device), dtype=torch.float32)
    model.eval()
    return model, metadata


__all__ = [
    "REQUIRED_KEYS",
    "WARMSTART_CHECKPOINT_VERSION",
    "WARMSTART_TRAINER_VERSION",
    "CorpusIdentity",
    "WarmstartCheckpointCompatibilityError",
    "WarmstartCheckpointError",
    "WarmstartCheckpointFormatError",
    "WarmstartCorpusMismatchError",
    "build_warmstart_checkpoint_payload",
    "check_resume_identity",
    "collect_rng_state",
    "load_model_for_evaluation",
    "load_warmstart_checkpoint",
    "measure_corpus_identity",
    "payload_integrity_digest",
    "read_warmstart_payload",
    "save_warmstart_checkpoint",
    "validate_warmstart_payload",
    "verify_corpus_identity",
]
