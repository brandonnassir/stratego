"""Phase 14: hot resume checkpoints, the durable archive and candidate marks.

Specification source: `02_AGENT_2_FINAL_TRAINING_INTEGRATION.md` sections 9, 11
and 15, over the frozen `checkpoint_hierarchy` block.

Three cadences, one payload
---------------------------
A hot checkpoint (15 minutes, internal disk), a durable archive snapshot (2
hours, external volume) and a final-policy candidate (6 hours, a *mark* on an
archive snapshot) all carry the same complete payload. That is deliberate: an
archive snapshot that could not resume the run would be a snapshot whose
provenance nobody could reconstruct, and a candidate that was not also an
archive entry would be a fourth thing to keep consistent.

Validate before rotating
------------------------
:meth:`HotCheckpointRing.write` writes atomically, then *reads the file back
and validates it*, and only then removes anything older. A crash during the
write therefore costs the newest checkpoint and never the four good ones behind
it. The frozen retention — at least the most recent four valid hot checkpoints
— is a floor the ring enforces after validation, not before.

What a resume needs
-------------------
Everything on the frozen list: weights, optimizer state, the explicit statement
that no EMA exists, the optimizer step, RNG stream state, population schedule
state, the active historical pool with its categories, the archive cursor, the
trajectory/shard cursor, storage state, the ORIGINAL start and deadline, the
main/late schedule state, and candidate-evaluation scheduling state. A payload
missing any of them is refused at write time rather than discovered at resume.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch

from ..model.production_model import build_candidate_model
from .phase14_contract import (
    AGENT1C_CHECKPOINT,
    AGENT1C_SHA256,
    ANCHOR_SHA256,
    EMA_PRESENT,
    EMA_STATE_RECORD,
    HOT_CHECKPOINT_REQUIRED_FIELDS,
    HOT_CHECKPOINT_RETAIN,
    PHASE14_CONTRACT_VERSION,
    PHASE14_NAMESPACE,
    PHASE14_POOL_VERSION,
    PHASE14_TRAINER_VERSION,
    SELECTION_PACK_DIGEST,
    STARTING_CHECKPOINT,
    STARTING_CHECKPOINT_SHA256,
    STARTING_MODEL_STATE_DIGEST,
    contract_digest,
    file_sha256,
)
from .phase14_seed import ROOT_SEEDS, seed_contract_digest
from .phase9_behavior import state_dict_digest
from .phase9_checkpoint import (
    _state_tree_to_cpu,
    rules_model_observation_versions,
    software_runtime_versions,
)
from .warmstart_checkpoint import payload_integrity_digest

PHASE14_CHECKPOINT_VERSION = "phase14_checkpoint_v1"

SNAPSHOT_ROLE_HOT = "hot_resume"
SNAPSHOT_ROLE_ARCHIVE = "durable_archive"
SNAPSHOT_ROLE_BEHAVIOR = "behavior_snapshot"
SNAPSHOT_ROLES = (SNAPSHOT_ROLE_HOT, SNAPSHOT_ROLE_ARCHIVE, SNAPSHOT_ROLE_BEHAVIOR)

REQUIRED_KEYS = (
    "phase14_checkpoint_version",
    "snapshot_role",
    "namespace",
    "model_state",
    "model_state_digest",
    "optimizer_state",
    "ema_state",
    "trainer_state",
    "run_window",
    "schedule_state",
    "population_schedule_state",
    "active_historical_pool",
    "historical_archive_state",
    "shard_cursor",
    "storage_state",
    "candidate_evaluation_state",
    "rng",
    "upstream",
    "software_runtime_versions",
    "rules_model_observation_versions",
    "integrity_digest",
)


class Phase14CheckpointError(RuntimeError):
    """A Phase 14 checkpoint could not be written, read back or verified."""


# ---------------------------------------------------------------------------
# Bindings
# ---------------------------------------------------------------------------


def upstream_bindings() -> dict:
    """The frozen upstream identity every Phase 14 checkpoint carries.

    A checkpoint that cannot answer "which frozen upstream produced me, and
    under which contract" is not a Phase 14 checkpoint. Agent 1C appears here
    as an explicit *non*-parent, because "the policy did not start from Agent
    1C" is a claim the artifacts should be able to settle.
    """
    return {
        "parent_checkpoint": STARTING_CHECKPOINT,
        "parent_sha256": STARTING_CHECKPOINT_SHA256,
        "parent_model_state_digest": STARTING_MODEL_STATE_DIGEST,
        "anchors": dict(ANCHOR_SHA256),
        "agent1c_not_parent": {"checkpoint": AGENT1C_CHECKPOINT, "sha256": AGENT1C_SHA256},
        "phase14_contract_digest": contract_digest(),
        "phase14_seed_contract_digest": seed_contract_digest(),
        "phase14_pool_version": PHASE14_POOL_VERSION,
        "candidate_pack_digest": SELECTION_PACK_DIGEST,
        "root_seeds": dict(ROOT_SEEDS),
    }


def _rng_state(device: str) -> dict:
    """Every RNG state, with the unused streams stated as unused.

    Phase 14 draws no training randomness: the epoch order is a pure function
    of `(iteration, epoch)` through the frozen train-order seed, and the
    behavior draw is a pure function of `(game_id, ply)`. The torch states are
    captured anyway so the file records the environment completely.
    """
    rng = {
        "python": "unused: the Phase 14 epoch order is a frozen-seed shuffle",
        "numpy": "unused: no trainer stream draws from the NumPy RNG",
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_mps": None,
        "device": str(device),
        "note": (
            "minibatch order derives from train_order_seed(iteration, epoch) and "
            "action draws from action_sampling_uniform(game_id, ply); no global RNG "
            "cursor decides any batch or any move"
        ),
    }
    if str(device).startswith("mps") and hasattr(torch, "mps"):
        try:
            rng["torch_mps"] = torch.mps.get_rng_state().clone()
        except Exception:  # noqa: BLE001 - an unavailable MPS generator is recorded
            rng["torch_mps"] = None
    return rng


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


def build_payload(
    *,
    model,
    optimizer,
    snapshot_role: str,
    trainer_state: dict,
    run_window: dict,
    schedule_state: dict,
    population_schedule_state: dict,
    active_historical_pool: dict,
    historical_archive_state: dict,
    shard_cursor: dict,
    storage_state: dict,
    candidate_evaluation_state: dict,
    device: str,
    diagnostics: "dict | None" = None,
) -> dict:
    """One `phase14_checkpoint_v1` payload, complete by construction."""
    if snapshot_role not in SNAPSHOT_ROLES:
        raise Phase14CheckpointError(
            f"unknown snapshot role {snapshot_role!r}; expected one of {list(SNAPSHOT_ROLES)}"
        )
    for key in ("run_start_utc", "run_deadline_utc", "transition_utc"):
        if key not in run_window:
            raise Phase14CheckpointError(
                f"the run window is missing {key!r}; a checkpoint that cannot state "
                "the original deadline could be resumed into a fresh 168 hours"
            )
    summary = model.architecture_summary()
    payload = {
        "phase14_checkpoint_version": PHASE14_CHECKPOINT_VERSION,
        "contract_version": PHASE14_CONTRACT_VERSION,
        "trainer_version": PHASE14_TRAINER_VERSION,
        "snapshot_role": str(snapshot_role),
        "namespace": PHASE14_NAMESPACE,
        "model_state": {
            "model_configuration": {
                "candidate_id": "C1",
                "architecture_summary": summary,
                "parameters": int(sum(tensor.numel() for tensor in model.parameters())),
            },
            "provenance": {
                "initialisation_seed": 0,
                "parent": STARTING_CHECKPOINT_SHA256,
            },
            "state_dict": _state_tree_to_cpu(model.state_dict()),
        },
        "model_state_digest": state_dict_digest(model),
        "optimizer_state": _state_tree_to_cpu(optimizer.state_dict()),
        # Recorded as an explicit absence rather than omitted: "there is no EMA
        # in this system" and "somebody forgot the field" must not look alike.
        "ema_state": {"present": EMA_PRESENT, "statement": EMA_STATE_RECORD},
        "trainer_state": dict(trainer_state),
        "run_window": dict(run_window),
        "schedule_state": dict(schedule_state),
        "population_schedule_state": dict(population_schedule_state),
        "active_historical_pool": dict(active_historical_pool),
        "historical_archive_state": dict(historical_archive_state),
        "shard_cursor": dict(shard_cursor),
        "storage_state": dict(storage_state),
        "candidate_evaluation_state": dict(candidate_evaluation_state),
        "rng": _rng_state(device),
        "upstream": upstream_bindings(),
        "diagnostics": dict(diagnostics or {}),
        "software_runtime_versions": software_runtime_versions(),
        "rules_model_observation_versions": rules_model_observation_versions(),
    }
    missing = [key for key in HOT_CHECKPOINT_REQUIRED_FIELDS if not _covers(payload, key)]
    if missing:
        raise Phase14CheckpointError(
            f"the payload does not cover the frozen resume fields: {missing}"
        )
    payload["integrity_digest"] = payload_integrity_digest(payload)
    return payload


def _covers(payload: dict, field: str) -> bool:
    """Whether the payload answers one frozen resume-field requirement."""
    aliases = {
        "global_optimizer_step": ("trainer_state", "global_optimizer_step"),
        "run_start_utc": ("run_window", "run_start_utc"),
        "run_deadline_utc": ("run_window", "run_deadline_utc"),
    }
    if field in aliases:
        section, key = aliases[field]
        return key in payload.get(section, {})
    return field in payload


def save(payload: dict, path: "str | Path", *, fsync: bool = True) -> dict:
    """Write one payload atomically and return its file identity."""
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise Phase14CheckpointError(f"payload is missing required keys: {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".partial")
    os.close(handle)
    temporary = Path(temporary)
    try:
        with open(temporary, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            if fsync:
                os.fsync(stream.fileno())
        temporary.replace(path)
        if fsync:
            descriptor = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "model_state_digest": payload["model_state_digest"],
        "snapshot_role": payload["snapshot_role"],
        "global_optimizer_step": int(
            payload["trainer_state"].get("global_optimizer_step", 0)
        ),
    }


def read(path: "str | Path") -> dict:
    """Read and fully validate one Phase 14 checkpoint payload."""
    path = Path(path)
    if not path.exists():
        raise Phase14CheckpointError(f"no Phase 14 checkpoint at {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise Phase14CheckpointError(f"{path}: payload is not a mapping")
    if payload.get("phase14_checkpoint_version") != PHASE14_CHECKPOINT_VERSION:
        raise Phase14CheckpointError(
            f"{path}: names checkpoint version {payload.get('phase14_checkpoint_version')!r}"
        )
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise Phase14CheckpointError(f"{path}: missing required keys: {missing}")
    recorded = payload["integrity_digest"]
    recomputed = payload_integrity_digest(
        {key: value for key, value in payload.items() if key != "integrity_digest"}
    )
    if recorded != recomputed:
        raise Phase14CheckpointError(
            f"{path}: integrity digest {recorded} != recomputed {recomputed}"
        )
    expected = upstream_bindings()
    for key in ("parent_sha256", "phase14_contract_digest", "candidate_pack_digest"):
        if payload["upstream"].get(key) != expected[key]:
            raise Phase14CheckpointError(
                f"{path}: upstream {key} is {payload['upstream'].get(key)!r}, the live "
                f"frozen value is {expected[key]!r}"
            )
    return payload


def is_valid(path: "str | Path") -> bool:
    try:
        read(path)
    except Exception:  # noqa: BLE001 - "valid" here means "reads back and verifies"
        return False
    return True


def model_from_payload(payload: dict, *, device: "str | torch.device" = "cpu"):
    """Rebuild the model a Phase 14 payload describes, float32 on `device`."""
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
    model = model.to(device=torch.device(device), dtype=torch.float32)
    observed = state_dict_digest(model)
    if observed != payload["model_state_digest"]:
        raise Phase14CheckpointError(
            f"rebuilt model digest {observed} != recorded {payload['model_state_digest']}"
        )
    return model


# ---------------------------------------------------------------------------
# The hot ring
# ---------------------------------------------------------------------------


HOT_PREFIX = "hot_"
HOT_SUFFIX = ".pt"


class HotCheckpointRing:
    """The rotating set of hot resume checkpoints on fast internal storage.

    Writes are validated by reading them back before anything older is removed,
    and at least :data:`HOT_CHECKPOINT_RETAIN` valid files survive every
    rotation. A resume takes the newest file that validates — not the newest
    file — so a torn write costs one cadence rather than the run.
    """

    def __init__(self, directory, *, retain: int = HOT_CHECKPOINT_RETAIN) -> None:
        self.directory = Path(directory)
        self.retain = max(int(retain), HOT_CHECKPOINT_RETAIN)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _name(self, index: int, step: int) -> str:
        return f"{HOT_PREFIX}{index:06d}_step{step:09d}{HOT_SUFFIX}"

    def files(self) -> list:
        """Every hot file, newest first by write index."""
        found = sorted(
            (path for path in self.directory.glob(f"{HOT_PREFIX}*{HOT_SUFFIX}")),
            key=lambda path: path.name,
            reverse=True,
        )
        return list(found)

    def next_index(self) -> int:
        existing = self.files()
        if not existing:
            return 1
        return int(existing[0].name[len(HOT_PREFIX) : len(HOT_PREFIX) + 6]) + 1

    def write(self, payload: dict, *, fsync: bool = True) -> dict:
        """Write, validate, then prune. In that order, always."""
        index = self.next_index()
        step = int(payload["trainer_state"].get("global_optimizer_step", 0))
        path = self.directory / self._name(index, step)
        written = save(payload, path, fsync=fsync)
        try:
            read(path)
        except Phase14CheckpointError:
            path.unlink(missing_ok=True)
            raise
        written["index"] = index
        written["pruned"] = self.prune()
        return written

    def prune(self) -> list:
        """Remove hot files beyond the retention floor, oldest first.

        Only *valid* files count toward the floor: keeping four files of which
        two are unreadable would satisfy a count and not the requirement.
        """
        valid: list = []
        invalid: list = []
        for path in self.files():
            (valid if is_valid(path) else invalid).append(path)
        removed: list = []
        for path in valid[self.retain :]:
            path.unlink(missing_ok=True)
            removed.append(str(path))
        # An unreadable file older than the retained window is dead weight; one
        # inside the window is left alone so a human can look at it.
        for path in invalid:
            if len(valid) >= self.retain and path.name < valid[self.retain - 1].name:
                path.unlink(missing_ok=True)
                removed.append(str(path))
        return removed

    def latest_valid(self) -> "Path | None":
        for path in self.files():
            if is_valid(path):
                return path
        return None

    def load_latest(self) -> "tuple[Path, dict] | None":
        path = self.latest_valid()
        if path is None:
            return None
        return path, read(path)

    def status(self) -> dict:
        files = self.files()
        valid = [path for path in files if is_valid(path)]
        return {
            "directory": str(self.directory),
            "files": len(files),
            "valid": len(valid),
            "retain": self.retain,
            "latest": str(valid[0]) if valid else None,
        }


# ---------------------------------------------------------------------------
# The durable archive
# ---------------------------------------------------------------------------


ARCHIVE_PREFIX = "archive_"
CANDIDATE_MARK_SUFFIX = ".candidate.json"


def archive_snapshot_path(directory, position: int) -> Path:
    return Path(directory) / f"{ARCHIVE_PREFIX}{position:04d}{HOT_SUFFIX}"


def write_archive_snapshot(directory, payload: dict, *, position: int, fsync: bool = True) -> dict:
    """Write one durable snapshot and read it back before reporting success.

    The archive is the pool's ordering authority and the candidates' storage,
    so an entry that cannot be read is worse than an entry that does not exist:
    the write is verified here, and the caller only appends to the archive
    after this returns.
    """
    path = archive_snapshot_path(directory, position)
    if path.exists():
        raise Phase14CheckpointError(
            f"archive position {position} already exists at {path}; the archive is "
            "append-only and its order is its identity"
        )
    written = save(payload, path, fsync=fsync)
    read(path)
    written["position"] = int(position)
    return written


def candidate_mark_path(directory, hour: int) -> Path:
    return Path(directory) / f"candidate_h{hour:03d}{CANDIDATE_MARK_SUFFIX}"


def mark_candidate(
    directory,
    *,
    hour: int,
    snapshot_path: "str | Path",
    snapshot_sha256: str,
    model_state_digest: str,
    elapsed_seconds: float,
    written_utc: str,
    iteration: int,
    global_optimizer_step: int,
    archive_position: "int | None" = None,
) -> dict:
    """Mark one durable snapshot as a final-policy candidate.

    A mark, not a copy: the candidate *is* the archive snapshot (or, at hour 0,
    the accepted starting checkpoint). Copying would create a second set of
    bytes to keep consistent and would double the archive's disk footprint for
    no evidentiary gain.
    """
    path = Path(snapshot_path)
    record = {
        "artifact": "phase14_candidate_mark_v1",
        "hour": int(hour),
        "archive_position": None if archive_position is None else int(archive_position),
        "snapshot_path": str(path),
        "snapshot_sha256": str(snapshot_sha256),
        "model_state_digest": str(model_state_digest),
        "elapsed_seconds": float(elapsed_seconds),
        "written_utc": str(written_utc),
        "iteration": int(iteration),
        "global_optimizer_step": int(global_optimizer_step),
        "evaluation_status": "pending",
        "pack_digest": SELECTION_PACK_DIGEST,
        "note": (
            "a candidate, not the deployed final policy; selection happens after the "
            "run under the frozen selection rule"
        ),
    }
    mark = candidate_mark_path(directory, hour)
    mark.parent.mkdir(parents=True, exist_ok=True)
    mark.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def read_candidate_marks(directory) -> list:
    directory = Path(directory)
    if not directory.exists():
        return []
    marks = []
    for path in sorted(directory.glob(f"candidate_h*{CANDIDATE_MARK_SUFFIX}")):
        marks.append(json.loads(path.read_text()))
    return sorted(marks, key=lambda record: int(record["hour"]))


def archive_manifest(directory) -> dict:
    directory = Path(directory)
    snapshots = []
    for path in sorted(directory.glob(f"{ARCHIVE_PREFIX}*{HOT_SUFFIX}")):
        snapshots.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "position": int(path.stem[len(ARCHIVE_PREFIX) :]),
            }
        )
    return {
        "directory": str(directory),
        "snapshots": snapshots,
        "candidates": read_candidate_marks(directory),
    }


# ---------------------------------------------------------------------------
# Behavior snapshots
# ---------------------------------------------------------------------------


BEHAVIOR_PREFIX = "B"


def behavior_snapshot_path(directory, identity: str) -> Path:
    return Path(directory) / f"{identity}{HOT_SUFFIX}"


def write_behavior_snapshot(directory, payload: dict, *, identity: str, retain: int = 2) -> dict:
    """Freeze the learner's current weights as one iteration's behavior snapshot.

    Every current-policy decision of an iteration must come from *one*
    immutable set of weights, and that set has to be addressable by SHA-256 for
    the store's per-side identity to mean anything — so the snapshot is a file,
    written before collection starts, not the live trainer model.

    Only a couple are kept: the pool's opponents come from the durable archive,
    so an old behavior snapshot has no consumer once its iteration is committed.
    """
    directory = Path(directory)
    path = behavior_snapshot_path(directory, identity)
    written = save(payload, path)
    read(path)
    kept = sorted(directory.glob(f"{BEHAVIOR_PREFIX}*{HOT_SUFFIX}"), reverse=True)
    removed = []
    for old in kept[max(int(retain), 1) :]:
        old.unlink(missing_ok=True)
        removed.append(str(old))
    written["identity"] = identity
    written["pruned"] = removed
    return written


def load_any_model(path: "str | Path", *, device: str = "cpu"):
    """Build the model held by a Phase 14, Phase 9 or warmstart checkpoint.

    The pool spans three formats by construction — the P8 anchor is a warmstart
    checkpoint, the P9 anchor is the accepted Phase 9 payload, and every Phase
    14 snapshot is this module's format — so the resolver has to know all three
    rather than assume one.
    """
    path = Path(path)
    container = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(container, dict):
        raise Phase14CheckpointError(f"{path}: payload is not a mapping")
    if container.get("phase14_checkpoint_version") == PHASE14_CHECKPOINT_VERSION:
        return model_from_payload(read(path), device=device)
    if "phase9_checkpoint_version" in container:
        from .phase9_checkpoint import model_from_payload as phase9_model
        from .phase9_checkpoint import read_phase9_payload, validate_phase9_payload

        payload = read_phase9_payload(path)
        validate_phase9_payload(payload, source=str(path))
        return phase9_model(payload, device=device)
    from .warmstart_checkpoint import load_model_for_evaluation

    model, _metadata = load_model_for_evaluation(path, device=device)
    return model


class Phase14SnapshotResolver:
    """Binds identities to real weights, loading each file exactly once.

    Two identities can name the same bytes — at iteration 1 the P9 anchor and
    the learner's own behavior snapshot are the accepted Phase 9 checkpoint —
    and they stay two snapshots with two logical identities while the weights
    are loaded once. The file digest is recomputed on every bind regardless, so
    sharing can never smuggle in a different checkpoint.
    """

    def __init__(self, *, device: str = "cpu", inference_batch_shape: int = 64) -> None:
        self.device = device
        self.inference_batch_shape = int(inference_batch_shape)
        self._models: dict = {}
        self._digests: dict = {}
        self.load_count = 0

    def bind(
        self,
        path: "str | Path",
        *,
        logical_identity: str,
        policy_token: str,
        expected_sha256: "str | None" = None,
    ):
        from .phase9_behavior import load_behavior_snapshot

        key = str(Path(path).resolve())
        model = self._models.get(key)
        if model is None:
            model = load_any_model(path, device=self.device)
            self.load_count += 1
        snapshot = load_behavior_snapshot(
            path,
            logical_identity=logical_identity,
            policy_token=policy_token,
            device=self.device,
            inference_batch_shape=self.inference_batch_shape,
            expected_sha256=expected_sha256,
            model=model,
            state_dict_digest_hint=self._digests.get(key),
        )
        if key not in self._models:
            self._models[key] = snapshot.model
            self._digests[key] = snapshot.loaded_state_dict_digest
        return snapshot


def export_evaluation_weights(source: "str | Path", export_path: "str | Path") -> dict:
    """Export a Phase 14 *or* accepted Phase 9 checkpoint to the eval format.

    The accepted Phase 9 Agent 8 procedure, unchanged in substance: the source
    is opened read-only and the export is refused unless every tensor
    round-trips bitwise, so a candidate evaluation can never silently measure
    weights that are not the checkpoint's. Candidates are evaluated through the
    accepted `InferenceOwner`, which reads this format; producing it here is
    what lets the evaluator be the accepted machinery rather than a second
    greedy decision path.
    """
    from ..model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from ..model.checkpoint import load_checkpoint, save_checkpoint

    source = Path(source)
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    source_sha = file_sha256(source)

    container = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(container, dict):
        raise Phase14CheckpointError(f"{source}: payload is not a mapping")
    if container.get("phase14_checkpoint_version") == PHASE14_CHECKPOINT_VERSION:
        model = model_from_payload(read(source))
    else:
        from .phase9_checkpoint import model_from_payload as phase9_model
        from .phase9_checkpoint import read_phase9_payload

        model = phase9_model(read_phase9_payload(source))
    digest = state_dict_digest(model)
    parameters = int(sum(tensor.numel() for tensor in model.parameters()))
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
        raise Phase14CheckpointError(
            f"the evaluation export of {source} changed the weights; BLOCKED"
        )
    if state_dict_digest(reloaded) != digest:
        raise Phase14CheckpointError(
            f"the evaluation export of {source} changed the model-state digest"
        )
    if file_sha256(source) != source_sha:
        raise Phase14CheckpointError(f"{source} changed while it was being exported")
    del model, reloaded, container
    return {
        "source": str(source),
        "source_sha256": source_sha,
        "export": str(export_path),
        "export_sha256": file_sha256(export_path),
        "model_state_digest": digest,
        "parameters": parameters,
        "bitwise_state_dict_match": True,
    }


def checkpoint_semantics() -> dict:
    return {
        "checkpoint_version": PHASE14_CHECKPOINT_VERSION,
        "roles": list(SNAPSHOT_ROLES),
        "hot_retention": HOT_CHECKPOINT_RETAIN,
        "write_order": "atomic write -> read-back validation -> prune",
        "resume_selection": "the newest hot checkpoint that validates",
        "archive": "append-only; a position is never overwritten",
        "candidate": "a mark on an archive snapshot, never a copy",
        "required_fields": list(HOT_CHECKPOINT_REQUIRED_FIELDS),
        "ema": EMA_STATE_RECORD,
    }
