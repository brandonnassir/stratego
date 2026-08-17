"""Phase 9 Agent 5: `phase9_checkpoint_v1` and the immutable archive.

Specification sources:

- `05_AGENT_5_PPO_TRAINER_AND_RESUME.md` ("Checkpoint/resume", the rejection
  list, the CPU/MPS proofs)
- `00_PHASE_9_SEQUENCE_AND_COMMON_CONTRACT.md` ("Checkpoint minimum
  contents", "Historical league", "Rollout state machine")
- Agent 1's `phase9_contract.CHECKPOINT_REQUIRED_FIELDS`, which is the
  authority on what a Phase 9 checkpoint must carry. This module reads that
  tuple rather than restating it, so a field added to the contract becomes a
  field this format refuses to be written without.

One format, three uses
----------------------
A Phase 9 checkpoint is written for three different reasons, and they are
deliberately the same bytes:

```text
resume checkpoint   the crash-safe continuation point of a training run
behavior snapshot   the immutable weights one RL iteration collects from
archive member      the immutable weights a historical opponent plays with
```

Making them one format is what lets a resumed trainer prove it is continuing
the run that collected its rollout: the behavior snapshot is not a stripped
export whose provenance has to be trusted, it is a complete checkpoint whose
identity fields can be compared field by field.

Archive identity is namespace-qualified
---------------------------------------
`pilot_p9a|H005`, `pilot_p9b|H005` and `canonical|H005` are three different
objects that share a local archive number. They are stored under separate
namespace directories, they carry their namespace inside the payload, and
:func:`bind_archive_member` refuses to hand back a snapshot whose payload
disagrees with the identity asked for. `H000` is the exception the frozen
schedule already names: one Phase 8 file, bit-identical everywhere, so its
token is namespace-free.

Binding weights without touching accepted code
----------------------------------------------
Agent 3's :func:`phase9_behavior.load_behavior_snapshot` reads Phase 8
containers through `load_model_for_evaluation`. Rather than change an accepted
module or write Phase 9 weights into a Phase 8-shaped file with fabricated
Phase 8 fields, :func:`bind_archive_member` builds the model from the Phase 9
payload itself and hands it to the accepted loader through its existing
`model=` parameter. The loader still hashes the real file, so the logical
identity is still bound to real bytes — nothing is bypassed, only supplied.

Absolute paths appear only in `diagnostics` and never define identity.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..model.checkpoint import (
    CheckpointError,
    build_checkpoint_payload,
    validate_checkpoint_payload,
)
from ..model.contract import (
    MODEL_CONTRACT_VERSION,
    OBSERVATION_VERSION,
    RULES_VERSION,
)
from ..model.production_model import build_candidate_model
from ..setups.sampler import SAMPLER_VERSION
from .phase9_behavior import (
    DEFAULT_INFERENCE_BATCH_SHAPE,
    BehaviorSnapshot,
    file_sha256,
    load_behavior_snapshot,
)
from .phase9_contract import (
    CHECKPOINT_REQUIRED_FIELDS,
    HISTORICAL_ANCHOR_ID,
    PHASE9_ADVANTAGE_VERSION,
    PHASE9_CHECKPOINT_VERSION,
    PHASE9_POPULATION_VERSION,
    PHASE9_ROLLOUT_SCHEDULE_VERSION,
    PHASE9_ROLLOUT_STORE_VERSION,
    PHASE9_TRAIN_ORDER_VERSION,
    contract_digest,
)
from .phase9_schedule import (
    ANCHOR_POLICY_TOKEN,
    ARCHIVE_TOKEN_PREFIX,
    RUN_NAMESPACES,
    historical_policy_token,
)
from .phase9_seed import CANONICAL_PHASE9_SEEDS
from .phase9_targets import PHASE9_EXAMPLE_VERSION, example_contract_digest
from .warmstart_checkpoint import CorpusIdentity, payload_integrity_digest

#: The trainer identity this checkpoint format belongs to. Declared here for
#: the same reason Phase 8 declares its trainer version in its checkpoint
#: module: the format is what a resume validates against, so the version has to
#: be readable without importing the trainer.
PHASE9_TRAINER_VERSION = "phase9_trainer_v1"

#: Everything a `phase9_checkpoint_v1` file holds: the contract's required
#: fields, plus the version/identity envelope that makes a file self-describing
#: and the diagnostics block that is recorded and never consulted.
PHASE9_CHECKPOINT_KEYS = CHECKPOINT_REQUIRED_FIELDS + (
    "phase9_checkpoint_version",
    "trainer_version",
    "train_config",
    "train_config_digest",
    "contract_digest",
    "example_version",
    "example_contract_digest",
    "advantage_version",
    "train_order_version",
    "rollout_store_version",
    "rollout_schedule_version",
    "snapshot_role",
    "rng",
    "diagnostics",
    "integrity_digest",
)

#: What a checkpoint may have been written for. The role is recorded, never
#: inferred from a path: an archive member moved to another directory is still
#: an archive member.
SNAPSHOT_ROLES = ("resume", "behavior_snapshot", "archive_member")

#: Where archive members live. Namespaced, because the local archive number is
#: not unique across runs.
ARCHIVE_DIRECTORY = "checkpoints/phase9/archive"


class Phase9CheckpointError(RuntimeError):
    """A Phase 9 checkpoint is unusable. Never repaired, always raised."""


class Phase9CheckpointFormatError(Phase9CheckpointError):
    """The file is not a readable, structurally valid checkpoint."""


class Phase9CheckpointIdentityError(Phase9CheckpointError):
    """The file is valid but does not belong to the run asking to resume it."""


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------


def _state_tree_to_cpu(value):
    """A deep copy of an optimizer/scheduler state tree with tensors on CPU.

    The Phase 8 twin of this helper exists for the same reason: a checkpoint
    written from an MPS run must reload on any device, and only CPU tensors
    round-trip through `weights_only=True`.
    """
    if isinstance(value, dict):
        return {key: _state_tree_to_cpu(entry) for key, entry in value.items()}
    if isinstance(value, list):
        return [_state_tree_to_cpu(entry) for entry in value]
    if isinstance(value, tuple):
        return tuple(_state_tree_to_cpu(entry) for entry in value)
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    return value


def software_runtime_versions() -> dict:
    """The runtime the checkpoint was written under. Diagnostic, not identity."""
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "platform": platform.platform(),
    }


def rules_model_observation_versions() -> dict:
    """The three frozen Phase 8 versions a Phase 9 run may never change."""
    return {
        "rules_version": RULES_VERSION,
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "observation_version": OBSERVATION_VERSION,
    }


def build_phase9_checkpoint_payload(
    *,
    model,
    optimizer,
    scheduler,
    train_config: dict,
    train_config_digest: str,
    corpus_identity: CorpusIdentity,
    global_optimizer_step: int,
    rl_iteration: int,
    minibatch_cursor: dict,
    examples_consumed: int,
    behavior_snapshot_identity: str,
    behavior_checkpoint_sha256: str,
    rollout_iteration_identity: str,
    sealed_rollout_digest: str,
    kl_beta: float,
    kl_controller_state: dict,
    entropy_schedule_position: dict,
    active_historical_identities,
    historical_checkpoint_digests: dict,
    best_validation_score,
    best_checkpoint_identity,
    validation_history,
    wall_clock_counters: dict,
    diagnostics: dict,
    snapshot_role: str = "resume",
    population_version: str = PHASE9_POPULATION_VERSION,
    opponent_schedule_version: str = PHASE9_ROLLOUT_SCHEDULE_VERSION,
    setup_sampler_version: str = SAMPLER_VERSION,
) -> dict:
    """Assemble one complete `phase9_checkpoint_v1` payload in memory.

    Every field named by `CHECKPOINT_REQUIRED_FIELDS` is populated here; the
    completeness check runs on the assembled payload rather than on this
    signature, so a contract field with no argument fails loudly instead of
    being written as `None`.
    """
    if snapshot_role not in SNAPSHOT_ROLES:
        raise Phase9CheckpointError(
            f"unknown snapshot role {snapshot_role!r}; expected one of "
            f"{list(SNAPSHOT_ROLES)}"
        )
    payload = {
        "phase9_checkpoint_version": PHASE9_CHECKPOINT_VERSION,
        "trainer_version": PHASE9_TRAINER_VERSION,
        "snapshot_role": str(snapshot_role),
        "model_state": build_checkpoint_payload(
            model, training_step=int(global_optimizer_step)
        ),
        "optimizer_state": _state_tree_to_cpu(optimizer.state_dict()),
        "scheduler_state": _state_tree_to_cpu(scheduler.state_dict()),
        "global_optimizer_step": int(global_optimizer_step),
        "rl_iteration": int(rl_iteration),
        "minibatch_cursor": dict(minibatch_cursor),
        "examples_consumed": int(examples_consumed),
        "behavior_snapshot_identity": str(behavior_snapshot_identity),
        "behavior_checkpoint_sha256": str(behavior_checkpoint_sha256),
        "rollout_iteration_identity": str(rollout_iteration_identity),
        "sealed_rollout_digest": str(sealed_rollout_digest),
        "kl_beta": float(kl_beta),
        "kl_controller_state": dict(kl_controller_state),
        "entropy_schedule_position": dict(entropy_schedule_position),
        "population_version": str(population_version),
        "active_historical_identities": [
            str(identity) for identity in active_historical_identities
        ],
        "historical_checkpoint_digests": {
            str(key): str(value) for key, value in dict(historical_checkpoint_digests).items()
        },
        "opponent_schedule_version": str(opponent_schedule_version),
        "setup_sampler_version": str(setup_sampler_version),
        "best_validation_score": best_validation_score,
        "best_checkpoint_identity": best_checkpoint_identity,
        "validation_history": [dict(entry) for entry in validation_history],
        "phase9_seeds": dict(CANONICAL_PHASE9_SEEDS),
        "corpus_identities": corpus_identity.to_dict(),
        "rules_model_observation_versions": rules_model_observation_versions(),
        "wall_clock_counters": dict(wall_clock_counters),
        "software_runtime_versions": software_runtime_versions(),
        "train_config": dict(train_config),
        "train_config_digest": str(train_config_digest),
        "contract_digest": contract_digest(),
        "example_version": PHASE9_EXAMPLE_VERSION,
        "example_contract_digest": example_contract_digest(),
        "advantage_version": PHASE9_ADVANTAGE_VERSION,
        "train_order_version": PHASE9_TRAIN_ORDER_VERSION,
        "rollout_store_version": PHASE9_ROLLOUT_STORE_VERSION,
        "rollout_schedule_version": PHASE9_ROLLOUT_SCHEDULE_VERSION,
        "rng": _collect_rng_state(str(diagnostics.get("device", "cpu"))),
        "diagnostics": dict(diagnostics),
    }
    missing = [key for key in PHASE9_CHECKPOINT_KEYS if key not in payload and key != "integrity_digest"]
    if missing:
        raise Phase9CheckpointError(
            f"assembled payload is missing contract field(s): {', '.join(missing)}"
        )
    payload["integrity_digest"] = payload_integrity_digest(payload)
    return payload


def _collect_rng_state(device: str) -> dict:
    """Every RNG state, with the unused streams stated as unused.

    Phase 9 draws no training randomness at all: the epoch order is a pure
    function of `(namespace, iteration, epoch)` through the frozen train-order
    seed, and C1 has no dropout. The torch states are captured anyway so the
    file records the environment completely.
    """
    rng = {
        "python": "unused: the Phase 9 epoch order is a frozen-seed shuffle",
        "numpy": "unused: no trainer stream draws from the NumPy RNG",
        "torch_cpu": torch.get_rng_state(),
        "torch_mps": None,
        "note": (
            "minibatch order derives from train_order_seed(namespace, "
            "iteration, epoch); no global RNG cursor decides any batch"
        ),
    }
    if str(device).startswith("mps") and hasattr(torch, "mps"):
        try:
            rng["torch_mps"] = torch.mps.get_rng_state()
        except Exception:  # noqa: BLE001 - absence of an MPS generator is not a failure
            rng["torch_mps"] = None
    return rng


# ---------------------------------------------------------------------------
# Atomic writing
# ---------------------------------------------------------------------------


def save_phase9_checkpoint(
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

    The reload is what turns "rename is atomic" into "a renamed file is
    valid": the bytes actually on disk are validated, not the payload in
    memory. `crash_hook(stage)` (tests only) fires at `after_write`,
    `after_validate` and `after_commit`; raising inside it simulates a crash at
    that exact boundary.
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
    reloaded = read_phase9_payload(temporary)
    validate_phase9_payload(reloaded, source=str(temporary))
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
        "sha256": file_sha256(destination),
    }


# ---------------------------------------------------------------------------
# Reading and validation
# ---------------------------------------------------------------------------


def read_phase9_payload(path: "str | Path") -> dict:
    """Deserialize a checkpoint file, reporting corruption as a format error."""
    location = Path(path)
    if not location.exists():
        raise Phase9CheckpointFormatError(f"{location}: checkpoint does not exist")
    if location.stat().st_size == 0:
        raise Phase9CheckpointFormatError(f"{location}: checkpoint file is empty")
    try:
        payload = torch.load(location, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - every unreadable file is one category
        raise Phase9CheckpointFormatError(
            f"{location}: checkpoint could not be read (corrupted, truncated or "
            f"not a checkpoint): {type(error).__name__}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise Phase9CheckpointFormatError(
            f"{location}: checkpoint holds {type(payload).__name__}, expected a dict"
        )
    return payload


def validate_phase9_payload(payload: dict, *, source: str = "<memory>") -> dict:
    """Structure, versions, integrity digest and the embedded model payload.

    Deliberately *not* a resume authorization: a payload can be perfectly valid
    and still belong to a different run, a different rollout or a different
    behavior snapshot. That separation is
    :func:`check_phase9_resume_identity`'s job.
    """
    if not isinstance(payload, dict):
        raise Phase9CheckpointFormatError(
            f"{source}: expected a checkpoint mapping, got {type(payload).__name__}"
        )
    missing = [key for key in PHASE9_CHECKPOINT_KEYS if key not in payload]
    if missing:
        raise Phase9CheckpointFormatError(
            f"{source}: checkpoint is missing required field(s): {', '.join(missing)}"
        )
    unexpected = sorted(set(payload) - set(PHASE9_CHECKPOINT_KEYS))
    if unexpected:
        raise Phase9CheckpointFormatError(
            f"{source}: checkpoint carries unknown field(s): {', '.join(unexpected)}"
        )

    for field, expected in (
        ("phase9_checkpoint_version", PHASE9_CHECKPOINT_VERSION),
        ("trainer_version", PHASE9_TRAINER_VERSION),
        ("example_version", PHASE9_EXAMPLE_VERSION),
        ("advantage_version", PHASE9_ADVANTAGE_VERSION),
        ("train_order_version", PHASE9_TRAIN_ORDER_VERSION),
        ("rollout_store_version", PHASE9_ROLLOUT_STORE_VERSION),
        ("rollout_schedule_version", PHASE9_ROLLOUT_SCHEDULE_VERSION),
        ("contract_digest", contract_digest()),
        ("example_contract_digest", example_contract_digest()),
    ):
        if payload[field] != expected:
            raise Phase9CheckpointError(
                f"{source}: {field} is {payload[field]!r}, this build uses "
                f"{expected!r}; refusing to guess at the difference"
            )
    if payload["snapshot_role"] not in SNAPSHOT_ROLES:
        raise Phase9CheckpointFormatError(
            f"{source}: unknown snapshot role {payload['snapshot_role']!r}"
        )

    recorded = payload["integrity_digest"]
    recomputed = payload_integrity_digest(payload)
    if recorded != recomputed:
        raise Phase9CheckpointFormatError(
            f"{source}: integrity digest mismatch (recorded {recorded!r}, "
            f"recomputed {recomputed!r}); the file's content was altered after "
            "it was written"
        )

    for field in ("global_optimizer_step", "rl_iteration", "examples_consumed"):
        value = payload[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise Phase9CheckpointFormatError(
                f"{source}: {field} must be a non-negative integer, got {value!r}"
            )
    beta = payload["kl_beta"]
    if not isinstance(beta, float) or not np.isfinite(beta) or beta <= 0.0:
        raise Phase9CheckpointFormatError(
            f"{source}: kl_beta must be a finite positive float, got {beta!r}"
        )
    if not isinstance(payload["optimizer_state"], dict) or not payload["optimizer_state"]:
        raise Phase9CheckpointFormatError(
            f"{source}: optimizer_state is missing or empty"
        )
    if not isinstance(payload["scheduler_state"], dict):
        raise Phase9CheckpointFormatError(
            f"{source}: scheduler_state must be a mapping"
        )
    if not isinstance(payload["validation_history"], list):
        raise Phase9CheckpointFormatError(
            f"{source}: validation_history must be a list"
        )
    cursor = payload["minibatch_cursor"]
    if not isinstance(cursor, dict) or cursor.get("train_order_version") != (
        PHASE9_TRAIN_ORDER_VERSION
    ):
        raise Phase9CheckpointFormatError(
            f"{source}: minibatch cursor is missing or is not a "
            f"{PHASE9_TRAIN_ORDER_VERSION} cursor"
        )

    try:
        model_metadata = validate_checkpoint_payload(payload["model_state"], source=source)
    except CheckpointError as error:
        raise Phase9CheckpointError(
            f"{source}: embedded model checkpoint is invalid: {error}"
        ) from error
    identity = CorpusIdentity.from_dict(payload["corpus_identities"])

    return {
        "phase9_checkpoint_version": PHASE9_CHECKPOINT_VERSION,
        "trainer_version": PHASE9_TRAINER_VERSION,
        "snapshot_role": payload["snapshot_role"],
        "train_config": dict(payload["train_config"]),
        "train_config_digest": payload["train_config_digest"],
        "corpus_identities": identity.to_dict(),
        "global_optimizer_step": payload["global_optimizer_step"],
        "rl_iteration": payload["rl_iteration"],
        "examples_consumed": payload["examples_consumed"],
        "minibatch_cursor": dict(cursor),
        "behavior_snapshot_identity": payload["behavior_snapshot_identity"],
        "behavior_checkpoint_sha256": payload["behavior_checkpoint_sha256"],
        "rollout_iteration_identity": payload["rollout_iteration_identity"],
        "sealed_rollout_digest": payload["sealed_rollout_digest"],
        "kl_beta": payload["kl_beta"],
        "population_version": payload["population_version"],
        "best_validation_score": payload["best_validation_score"],
        "validation_entries": len(payload["validation_history"]),
        "integrity_digest": payload["integrity_digest"],
        "model_state": model_metadata,
        "diagnostics": dict(payload["diagnostics"]),
    }


# ---------------------------------------------------------------------------
# Resume authorization
# ---------------------------------------------------------------------------


def check_phase9_resume_identity(
    payload: dict,
    *,
    expected_train_config: "dict | None" = None,
    expected_train_config_digest: "str | None" = None,
    expected_corpus_identity: "CorpusIdentity | dict | None" = None,
    expected_sealed_rollout_digest: "str | None" = None,
    expected_rollout_identity: "str | None" = None,
    expected_behavior_checkpoint_sha256: "str | None" = None,
    expected_behavior_snapshot_identity: "str | None" = None,
    expected_population_version: "str | None" = None,
    expected_cursor: "dict | None" = None,
    source: str = "<memory>",
) -> dict:
    """Refuse to continue a run this checkpoint does not belong to.

    Every rejection the mission names, in one place, each with the question it
    answers:

    ```text
    optimizer/config drift    is this the same trainer configuration?
    corpus identity drift     is this the same accepted Phase 8 corpus?
    rollout digest drift      are these the same sealed bytes?
    rollout identity drift    is this the same logical iteration?
    behavior snapshot drift   were these games produced by these weights?
    population-version drift  is this the same league contract?
    cursor mismatch           is this the same position in the same order?
    ```

    Corpus identity is compared as version + digests. A relocated corpus with
    unchanged digests is the same corpus, so a path is never consulted.
    """
    problems: list[str] = []
    if expected_train_config_digest is not None:
        if payload["train_config_digest"] != expected_train_config_digest:
            problems.append(
                f"train config digest {payload['train_config_digest']!r} != "
                f"expected {expected_train_config_digest!r}"
            )
    if expected_train_config is not None:
        recorded = dict(payload["train_config"])
        differing = sorted(
            key
            for key in set(recorded) | set(expected_train_config)
            if recorded.get(key) != expected_train_config.get(key)
        )
        if differing:
            problems.append(f"train config differs in: {differing}")
    if expected_corpus_identity is not None:
        expected = (
            expected_corpus_identity
            if isinstance(expected_corpus_identity, CorpusIdentity)
            else CorpusIdentity.from_dict(expected_corpus_identity)
        )
        recorded = CorpusIdentity.from_dict(payload["corpus_identities"])
        if recorded != expected:
            problems.append(
                f"corpus identity {recorded.to_dict()} != expected {expected.to_dict()}"
            )
    for field, expected_value, label in (
        ("sealed_rollout_digest", expected_sealed_rollout_digest, "sealed rollout digest"),
        ("rollout_iteration_identity", expected_rollout_identity, "rollout identity"),
        (
            "behavior_checkpoint_sha256",
            expected_behavior_checkpoint_sha256,
            "behavior checkpoint SHA-256",
        ),
        (
            "behavior_snapshot_identity",
            expected_behavior_snapshot_identity,
            "behavior snapshot identity",
        ),
        ("population_version", expected_population_version, "population version"),
    ):
        if expected_value is not None and payload[field] != expected_value:
            problems.append(
                f"{label} {payload[field]!r} != expected {expected_value!r}"
            )
    if expected_cursor is not None:
        recorded = dict(payload["minibatch_cursor"])
        differing = sorted(
            key
            for key in set(recorded) | set(expected_cursor)
            if recorded.get(key) != expected_cursor.get(key)
        )
        if differing:
            problems.append(f"minibatch cursor differs in: {differing}")
    if problems:
        raise Phase9CheckpointIdentityError(
            f"{source}: this checkpoint does not belong to the run asking to "
            f"resume it ({'; '.join(problems)})"
        )
    return {
        "train_config_digest": payload["train_config_digest"],
        "sealed_rollout_digest": payload["sealed_rollout_digest"],
        "rollout_iteration_identity": payload["rollout_iteration_identity"],
        "behavior_snapshot_identity": payload["behavior_snapshot_identity"],
        "behavior_checkpoint_sha256": payload["behavior_checkpoint_sha256"],
        "population_version": payload["population_version"],
        "minibatch_cursor": dict(payload["minibatch_cursor"]),
    }


def model_from_payload(payload: dict, *, device: "str | torch.device" = "cpu"):
    """Rebuild the model a Phase 9 payload describes, on `device`, float32.

    `(candidate_id, initialisation_seed)` reconstructs the architecture and the
    state dict supplies the weights, exactly as the accepted evaluation loader
    does for a Phase 8 container.
    """
    model_payload = payload["model_state"]
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
    return model.to(device=torch.device(device), dtype=torch.float32)


def load_phase9_checkpoint(
    path: "str | Path",
    *,
    device: "str | torch.device" = "cpu",
    **identity_expectations,
) -> dict:
    """Read, validate, authorize and rebuild — in that order.

    Nothing is restored until every identity check has passed, so a rejected
    resume leaves the caller's process untouched.
    """
    payload = read_phase9_payload(path)
    metadata = validate_phase9_payload(payload, source=str(path))
    check_phase9_resume_identity(payload, source=str(path), **identity_expectations)
    return {
        "payload": payload,
        "metadata": metadata,
        "model": model_from_payload(payload, device=device),
        "optimizer_state": payload["optimizer_state"],
        "scheduler_state": payload["scheduler_state"],
        "minibatch_cursor": dict(payload["minibatch_cursor"]),
        "global_optimizer_step": int(payload["global_optimizer_step"]),
        "rl_iteration": int(payload["rl_iteration"]),
        "examples_consumed": int(payload["examples_consumed"]),
        "kl_beta": float(payload["kl_beta"]),
        "kl_controller_state": dict(payload["kl_controller_state"]),
        "entropy_schedule_position": dict(payload["entropy_schedule_position"]),
        "best_validation_score": payload["best_validation_score"],
        "best_checkpoint_identity": payload["best_checkpoint_identity"],
        "validation_history": [dict(entry) for entry in payload["validation_history"]],
        "wall_clock_counters": dict(payload["wall_clock_counters"]),
        "file_sha256": file_sha256(path),
    }


# ---------------------------------------------------------------------------
# The namespace-qualified immutable archive
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchiveMember:
    """One immutable archive object, named by namespace *and* local number.

    `qualified_identity` is what makes `pilot_p9a|H005` and `canonical|H005`
    different objects; `local_identity` is what the frozen schedule's active
    window talks about. Both are needed, and conflating them is precisely the
    mistake that would let one run's weights answer another run's schedule.
    """

    namespace: str
    local_identity: str
    policy_token: str
    path: str
    checkpoint_sha256: str
    state_dict_digest: str
    rl_iteration: int

    @property
    def qualified_identity(self) -> str:
        return qualified_archive_identity(self.namespace, self.local_identity)

    def to_dict(self) -> dict:
        return {
            "namespace": self.namespace,
            "local_identity": self.local_identity,
            "qualified_identity": self.qualified_identity,
            "policy_token": self.policy_token,
            "path": self.path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "state_dict_digest": self.state_dict_digest,
            "rl_iteration": int(self.rl_iteration),
        }


def qualified_archive_identity(namespace: str, local_identity: str) -> str:
    """`<namespace>|<H0nn>` — the identity an archive object actually has.

    `H000` keeps its namespace-free spelling because the frozen schedule
    already says so: it is one Phase 8 file, bit-identical in every run.
    """
    if local_identity == HISTORICAL_ANCHOR_ID:
        return HISTORICAL_ANCHOR_ID
    _require_namespace(namespace)
    return f"{namespace}|{local_identity}"


def _require_namespace(namespace: str) -> None:
    if namespace not in RUN_NAMESPACES:
        raise Phase9CheckpointError(
            f"unknown Phase 9 namespace {namespace!r}; expected one of "
            f"{list(RUN_NAMESPACES)}"
        )


def archive_directory(root: "str | Path", namespace: str) -> Path:
    """`<root>/<namespace>` — one directory per run, never a shared pool."""
    _require_namespace(namespace)
    return Path(root) / namespace


def archive_member_path(root: "str | Path", namespace: str, local_identity: str) -> Path:
    return archive_directory(root, namespace) / f"{local_identity}.pt"


def write_archive_member(
    payload: dict,
    root: "str | Path",
    *,
    namespace: str,
    local_identity: str,
    fsync: bool = True,
) -> ArchiveMember:
    """Write one archive member, or refuse because it already exists.

    "No archive checkpoint may be overwritten" is enforced here rather than by
    convention: an existing file is a stop condition, even when the bytes about
    to be written would be identical. A run that wants to re-derive an archive
    member has to say so by removing it, which is not something a training loop
    can do by accident.
    """
    if local_identity == HISTORICAL_ANCHOR_ID:
        raise Phase9CheckpointError(
            f"{HISTORICAL_ANCHOR_ID} is the accepted Phase 8 checkpoint and is "
            "never written by a Phase 9 run"
        )
    destination = archive_member_path(root, namespace, local_identity)
    if destination.exists():
        raise Phase9CheckpointError(
            f"archive member {qualified_archive_identity(namespace, local_identity)} "
            f"already exists at {destination}; archive checkpoints are immutable"
        )
    if payload["snapshot_role"] != "archive_member":
        raise Phase9CheckpointError(
            "an archive member must be written from a payload whose snapshot "
            f"role is 'archive_member', got {payload['snapshot_role']!r}"
        )
    recorded_namespace = payload["train_config"].get("namespace")
    if recorded_namespace != namespace:
        raise Phase9CheckpointError(
            f"payload belongs to namespace {recorded_namespace!r}, not {namespace!r}; "
            "an archive member may not be filed under another run's namespace"
        )
    written = save_phase9_checkpoint(payload, destination, fsync=fsync)
    return ArchiveMember(
        namespace=namespace,
        local_identity=local_identity,
        policy_token=historical_policy_token(namespace, local_identity),
        path=str(destination),
        checkpoint_sha256=written["sha256"],
        state_dict_digest=str(
            payload["model_state"]["provenance"]["state_dict_digest"]
        ),
        rl_iteration=int(payload["rl_iteration"]),
    )


def read_archive_member(
    root: "str | Path", *, namespace: str, local_identity: str
) -> ArchiveMember:
    """The recorded identity of an archive member already on disk."""
    path = archive_member_path(root, namespace, local_identity)
    payload = read_phase9_payload(path)
    validate_phase9_payload(payload, source=str(path))
    if payload["snapshot_role"] != "archive_member":
        raise Phase9CheckpointError(
            f"{path}: snapshot role is {payload['snapshot_role']!r}, not "
            "'archive_member'"
        )
    if payload["train_config"].get("namespace") != namespace:
        raise Phase9CheckpointError(
            f"{path}: payload names namespace "
            f"{payload['train_config'].get('namespace')!r}, not {namespace!r}"
        )
    return ArchiveMember(
        namespace=namespace,
        local_identity=local_identity,
        policy_token=historical_policy_token(namespace, local_identity),
        path=str(path),
        checkpoint_sha256=file_sha256(path),
        state_dict_digest=str(payload["model_state"]["provenance"]["state_dict_digest"]),
        rl_iteration=int(payload["rl_iteration"]),
    )


def bind_archive_member(
    member: ArchiveMember,
    *,
    device: str = "cpu",
    inference_batch_shape: int = DEFAULT_INFERENCE_BATCH_SHAPE,
    expected_sha256: "str | None" = None,
) -> BehaviorSnapshot:
    """Turn an archive member into a playable, frozen behavior snapshot.

    The model is built from the Phase 9 payload here and handed to the accepted
    Agent 3 loader through its `model=` parameter, so the file is still hashed
    and the logical identity is still bound to real bytes — the loader's
    binding check is supplied with weights, never bypassed.
    """
    payload = read_phase9_payload(member.path)
    validate_phase9_payload(payload, source=member.path)
    model = model_from_payload(payload, device=device)
    return load_behavior_snapshot(
        member.path,
        logical_identity=member.local_identity,
        policy_token=member.policy_token,
        device=device,
        inference_batch_shape=inference_batch_shape,
        expected_sha256=expected_sha256 or member.checkpoint_sha256,
        model=model,
        state_dict_digest_hint=None,
    )


def bind_behavior_snapshot(
    path: "str | Path",
    *,
    logical_identity: str,
    namespace: str,
    device: str = "cpu",
    inference_batch_shape: int = DEFAULT_INFERENCE_BATCH_SHAPE,
    expected_sha256: "str | None" = None,
) -> BehaviorSnapshot:
    """Bind a Phase 9 behavior snapshot file to its `B0nn` identity."""
    from .phase9_schedule import behavior_policy_token

    payload = read_phase9_payload(path)
    validate_phase9_payload(payload, source=str(path))
    if payload["behavior_snapshot_identity"] != logical_identity:
        raise Phase9CheckpointError(
            f"{path}: payload names behavior snapshot "
            f"{payload['behavior_snapshot_identity']!r}, not {logical_identity!r}"
        )
    model = model_from_payload(payload, device=device)
    return load_behavior_snapshot(
        path,
        logical_identity=logical_identity,
        policy_token=behavior_policy_token(namespace, int(payload["rl_iteration"])),
        device=device,
        inference_batch_shape=inference_batch_shape,
        expected_sha256=expected_sha256,
        model=model,
    )


def archive_manifest(root: "str | Path", namespace: str) -> dict:
    """Every archive member of one namespace, with its identities and digests."""
    directory = archive_directory(root, namespace)
    members = []
    if directory.exists():
        for path in sorted(directory.glob("H*.pt")):
            members.append(
                read_archive_member(
                    root, namespace=namespace, local_identity=path.stem
                ).to_dict()
            )
    return {
        "namespace": namespace,
        "archive_token_prefix": ARCHIVE_TOKEN_PREFIX,
        "anchor_identity": HISTORICAL_ANCHOR_ID,
        "anchor_policy_token": ANCHOR_POLICY_TOKEN,
        "directory": str(directory),
        "members": members,
        "identity_rule": (
            "an archive object is (namespace, local identity); the local number "
            "alone is not unique across runs"
        ),
    }


def checkpoint_semantics() -> dict:
    """The serializable statement of this format's frozen semantics."""
    return {
        "checkpoint_version": PHASE9_CHECKPOINT_VERSION,
        "trainer_version": PHASE9_TRAINER_VERSION,
        "required_fields": list(CHECKPOINT_REQUIRED_FIELDS),
        "payload_keys": list(PHASE9_CHECKPOINT_KEYS),
        "snapshot_roles": list(SNAPSHOT_ROLES),
        "atomic_write": (
            "write .partial -> fsync -> reload and fully validate the bytes on "
            "disk -> os.replace -> fsync the directory"
        ),
        "rejections": [
            "truncation or unreadable bytes",
            "integrity digest mismatch",
            "corpus identity drift",
            "sealed rollout digest drift",
            "rollout iteration identity drift",
            "behavior snapshot drift",
            "optimizer/train-config mismatch",
            "population-version mismatch",
            "minibatch cursor mismatch",
        ],
        "identity_rule": (
            "absolute paths are diagnostic only; identity is versions, logical "
            "identities and digests"
        ),
        "archive_rule": (
            "archive members are namespace-qualified and immutable; an existing "
            "file is never overwritten"
        ),
    }


__all__ = [
    "ARCHIVE_DIRECTORY",
    "PHASE9_CHECKPOINT_KEYS",
    "PHASE9_TRAINER_VERSION",
    "SNAPSHOT_ROLES",
    "ArchiveMember",
    "Phase9CheckpointError",
    "Phase9CheckpointFormatError",
    "Phase9CheckpointIdentityError",
    "archive_directory",
    "archive_manifest",
    "archive_member_path",
    "bind_archive_member",
    "bind_behavior_snapshot",
    "build_phase9_checkpoint_payload",
    "check_phase9_resume_identity",
    "checkpoint_semantics",
    "load_phase9_checkpoint",
    "model_from_payload",
    "qualified_archive_identity",
    "read_archive_member",
    "read_phase9_payload",
    "rules_model_observation_versions",
    "save_phase9_checkpoint",
    "software_runtime_versions",
    "validate_phase9_payload",
    "write_archive_member",
]
