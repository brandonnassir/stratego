"""Optional Phase 10B: the checkpoint format, its namespace and its bindings.

Specification source: `OPTIONAL_PHASE_10B_SETUP_CONDITIONED_FINE_TUNING_AGENT.md`
section 13.

Namespace
---------
Everything Phase 10B writes lives under `checkpoints/phase10b/`. The accepted
`checkpoints/phase9/selfplay_c1_v1.pt` is opened read-only and never written,
which :func:`assert_phase9_untouched` re-proves from live bytes at every stage
boundary rather than trusting that no code path did it.

Bindings
--------
Each committed checkpoint carries its parent Phase 9 SHA and state digest, the
P10-D config digest, the Phase 10 utility and scaler digests, the Phase 10B
contract and seed digests, the iteration, the optimizer step, the RNG and
schedule identity, the active history identities, its own model-state digest
and its own file SHA-256. A checkpoint that cannot answer "which frozen
upstream produced me, and under which schedule position" is not a Phase 10B
checkpoint.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from ..model.production_model import build_candidate_model
from .phase10b_contract import (
    ACCEPTED_MODEL_T_COEFFICIENT_DIGEST,
    ACCEPTED_PHASE10_SYSTEM_DIGEST,
    ACCEPTED_PHASE9_CHECKPOINT,
    ACCEPTED_PHASE9_PARAMETERS,
    ACCEPTED_PHASE9_SHA256,
    ACCEPTED_PHASE9_STATE_DIGEST,
    ACCEPTED_SELECTOR_CONFIG_SHA256,
    ACCEPTED_TRAIT_SCALER_DIGEST,
    ANCHOR_IDENTITY,
    PHASE10B_NAMESPACE,
    PHASE10B_TRAINER_VERSION,
    SELECTED_CANDIDATE_ID,
    Phase10BContractError,
    contract_digest,
)
from .phase10b_seed import ROOT_SEEDS, seed_contract_digest
from .phase9_behavior import (
    DEFAULT_INFERENCE_BATCH_SHAPE,
    BehaviorSnapshot,
    file_sha256,
    load_behavior_snapshot,
    state_dict_digest,
)
from .phase9_checkpoint import (
    _state_tree_to_cpu,
    rules_model_observation_versions,
    software_runtime_versions,
)
from .warmstart_checkpoint import payload_integrity_digest

PHASE10B_CHECKPOINT_VERSION = "phase10b_checkpoint_v1"

#: Where every Phase 10B artifact lives. Never the Phase 9 or Phase 10 tree.
CHECKPOINT_DIRECTORY = "checkpoints/phase10b"
CANONICAL_CANDIDATE_PATH = "checkpoints/phase10b/setup_conditioned_c1_v1.pt"

SNAPSHOT_ROLES = ("resume", "behavior_snapshot", "archive_member", "candidate")

REQUIRED_KEYS = (
    "phase10b_checkpoint_version",
    "snapshot_role",
    "namespace",
    "rl_iteration",
    "global_optimizer_step",
    "examples_consumed",
    "behavior_snapshot_identity",
    "behavior_checkpoint_sha256",
    "sealed_rollout_digest",
    "model_state",
    "model_state_digest",
    "optimizer_state",
    "kl_beta",
    "kl_controller_state",
    "learning_rate",
    "entropy_coefficient",
    "active_history_identities",
    "history_checkpoint_digests",
    "upstream",
    "schedule_identity",
    "rng",
    "counters",
    "wall_clock",
    "diagnostics",
    "software_runtime_versions",
    "rules_model_observation_versions",
    "integrity_digest",
)


class Phase10BCheckpointError(RuntimeError):
    """A Phase 10B checkpoint could not be written, read back or verified."""


def upstream_bindings() -> dict:
    """The frozen upstream identity every Phase 10B checkpoint carries."""
    return {
        "parent_phase9_checkpoint": ACCEPTED_PHASE9_CHECKPOINT,
        "parent_phase9_sha256": ACCEPTED_PHASE9_SHA256,
        "parent_phase9_model_state_digest": ACCEPTED_PHASE9_STATE_DIGEST,
        "parent_phase9_parameters": ACCEPTED_PHASE9_PARAMETERS,
        "selector_candidate": SELECTED_CANDIDATE_ID,
        "selector_config_sha256": ACCEPTED_SELECTOR_CONFIG_SHA256,
        "utility_coefficient_digest": ACCEPTED_MODEL_T_COEFFICIENT_DIGEST,
        "trait_scaler_digest": ACCEPTED_TRAIT_SCALER_DIGEST,
        "phase10_system_digest": ACCEPTED_PHASE10_SYSTEM_DIGEST,
        "phase10b_contract_digest": contract_digest(),
        "phase10b_seed_contract_digest": seed_contract_digest(),
    }


def _rng_state(device: str) -> dict:
    payload = {
        "torch_cpu": torch.get_rng_state().clone(),
        "device": str(device),
    }
    if str(device).startswith("mps") and torch.backends.mps.is_available():
        try:
            payload["torch_mps"] = torch.mps.get_rng_state().clone()
        except Exception:  # noqa: BLE001 - an unavailable MPS RNG is recorded, not fatal
            payload["torch_mps"] = None
    return payload


def build_payload(
    *,
    model,
    optimizer,
    snapshot_role: str,
    rl_iteration: int,
    global_optimizer_step: int,
    examples_consumed: int,
    behavior_snapshot_identity: str,
    behavior_checkpoint_sha256: str,
    sealed_rollout_digest: str,
    kl_beta: float,
    kl_controller_state: dict,
    learning_rate: float,
    entropy_coefficient: float,
    active_history_identities,
    history_checkpoint_digests: dict,
    schedule_identity: dict,
    counters: dict,
    wall_clock: dict,
    device: str,
    diagnostics: "dict | None" = None,
) -> dict:
    """One `phase10b_checkpoint_v1` payload."""
    if snapshot_role not in SNAPSHOT_ROLES:
        raise Phase10BCheckpointError(
            f"unknown snapshot role {snapshot_role!r}; expected one of "
            f"{list(SNAPSHOT_ROLES)}"
        )
    parameters = int(sum(tensor.numel() for tensor in model.parameters()))
    if parameters != ACCEPTED_PHASE9_PARAMETERS:
        raise Phase10BCheckpointError(
            f"the model holds {parameters:,} parameters; Phase 10B fine-tunes C1 "
            f"with {ACCEPTED_PHASE9_PARAMETERS:,}"
        )
    summary = model.architecture_summary()
    payload = {
        "phase10b_checkpoint_version": PHASE10B_CHECKPOINT_VERSION,
        "trainer_version": PHASE10B_TRAINER_VERSION,
        "snapshot_role": str(snapshot_role),
        "namespace": PHASE10B_NAMESPACE,
        "rl_iteration": int(rl_iteration),
        "global_optimizer_step": int(global_optimizer_step),
        "examples_consumed": int(examples_consumed),
        "behavior_snapshot_identity": str(behavior_snapshot_identity),
        "behavior_checkpoint_sha256": str(behavior_checkpoint_sha256),
        "sealed_rollout_digest": str(sealed_rollout_digest),
        "model_state": {
            "model_configuration": {
                "candidate_id": "C1",
                "architecture_summary": summary,
                "parameters": parameters,
            },
            "provenance": {
                "initialisation_seed": 0,
                "parent": ACCEPTED_PHASE9_SHA256,
            },
            "state_dict": _state_tree_to_cpu(model.state_dict()),
        },
        "model_state_digest": state_dict_digest(model),
        "optimizer_state": _state_tree_to_cpu(optimizer.state_dict()),
        "kl_beta": float(kl_beta),
        "kl_controller_state": dict(kl_controller_state),
        "learning_rate": float(learning_rate),
        "entropy_coefficient": float(entropy_coefficient),
        "active_history_identities": list(active_history_identities),
        "history_checkpoint_digests": dict(history_checkpoint_digests),
        "upstream": upstream_bindings(),
        "schedule_identity": dict(schedule_identity),
        "rng": _rng_state(device),
        "counters": dict(counters),
        "wall_clock": dict(wall_clock),
        "diagnostics": dict(diagnostics or {}),
        "software_runtime_versions": software_runtime_versions(),
        "rules_model_observation_versions": rules_model_observation_versions(),
        "root_seeds": dict(ROOT_SEEDS),
    }
    payload["integrity_digest"] = payload_integrity_digest(payload)
    return payload


def save(payload: dict, path: "str | Path", *, fsync: bool = True) -> dict:
    """Write one payload atomically and return its file identity."""
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise Phase10BCheckpointError(f"payload is missing required keys: {missing}")
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
        "rl_iteration": int(payload["rl_iteration"]),
        "global_optimizer_step": int(payload["global_optimizer_step"]),
        "snapshot_role": payload["snapshot_role"],
    }


def read(path: "str | Path") -> dict:
    """Read and fully validate one Phase 10B checkpoint payload."""
    path = Path(path)
    if not path.exists():
        raise Phase10BCheckpointError(f"no Phase 10B checkpoint at {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise Phase10BCheckpointError(f"{path}: payload is not a mapping")
    if payload.get("phase10b_checkpoint_version") != PHASE10B_CHECKPOINT_VERSION:
        raise Phase10BCheckpointError(
            f"{path}: names checkpoint version "
            f"{payload.get('phase10b_checkpoint_version')!r}"
        )
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise Phase10BCheckpointError(f"{path}: missing required keys: {missing}")
    recorded = payload["integrity_digest"]
    recomputed = payload_integrity_digest(
        {key: value for key, value in payload.items() if key != "integrity_digest"}
    )
    if recorded != recomputed:
        raise Phase10BCheckpointError(
            f"{path}: integrity digest {recorded} != recomputed {recomputed}"
        )
    expected = upstream_bindings()
    for key in ("parent_phase9_sha256", "selector_config_sha256", "phase10b_contract_digest"):
        if payload["upstream"].get(key) != expected[key]:
            raise Phase10BCheckpointError(
                f"{path}: upstream {key} is {payload['upstream'].get(key)!r}, the "
                f"live frozen value is {expected[key]!r}"
            )
    return payload


def model_from_payload(payload: dict, *, device: "str | torch.device" = "cpu"):
    """Rebuild the model a Phase 10B payload describes, float32 on `device`."""
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
        raise Phase10BCheckpointError(
            f"rebuilt model digest {observed} != recorded {payload['model_state_digest']}"
        )
    return model


# ---------------------------------------------------------------------------
# Snapshot binding
# ---------------------------------------------------------------------------


@dataclass
class SnapshotResolver:
    """Loads snapshots once and shares identical weights across identities.

    At iteration 1 the learner snapshot and the anchor are the *same accepted
    Phase 9 file* under two logical identities. They stay two
    :class:`BehaviorSnapshot` objects because the store has to preserve that
    distinction, but the weights are loaded once.
    """

    device: str = "cpu"
    inference_batch_shape: int = DEFAULT_INFERENCE_BATCH_SHAPE

    def __post_init__(self) -> None:
        self._models: dict = {}
        self._digests: dict = {}
        self.load_count = 0

    def _load_model(self, path: Path):
        key = str(path.resolve())
        if key in self._models:
            return self._models[key], self._digests[key]
        payload = _read_any_payload(path)
        if payload["kind"] == "phase10b":
            model = model_from_payload(payload["payload"], device=self.device)
        else:
            from .phase9_checkpoint import model_from_payload as phase9_model

            model = phase9_model(payload["payload"], device=self.device)
        model.eval()
        model.requires_grad_(False)
        digest = state_dict_digest(model)
        self._models[key] = model
        self._digests[key] = digest
        self.load_count += 1
        return model, digest

    def resolve(
        self,
        path: "str | Path",
        *,
        logical_identity: str,
        policy_token: str,
        expected_sha256: "str | None" = None,
    ) -> BehaviorSnapshot:
        path = Path(path)
        model, digest = self._load_model(path)
        return load_behavior_snapshot(
            path,
            logical_identity=logical_identity,
            policy_token=policy_token,
            device=self.device,
            inference_batch_shape=self.inference_batch_shape,
            expected_sha256=expected_sha256,
            model=model,
            state_dict_digest_hint=digest,
        )


def _read_any_payload(path: Path) -> dict:
    """Read either a Phase 10B or an accepted Phase 9 checkpoint payload."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise Phase10BCheckpointError(f"{path}: payload is not a mapping")
    if payload.get("phase10b_checkpoint_version") == PHASE10B_CHECKPOINT_VERSION:
        return {"kind": "phase10b", "payload": read(path)}
    from .phase9_checkpoint import read_phase9_payload, validate_phase9_payload

    phase9 = read_phase9_payload(path)
    validate_phase9_payload(phase9, source=str(path))
    return {"kind": "phase9", "payload": phase9}


# ---------------------------------------------------------------------------
# Upstream preservation
# ---------------------------------------------------------------------------


def assert_phase9_untouched(repository_root: "str | Path") -> dict:
    """Re-prove from live bytes that the accepted Phase 9 checkpoint is intact.

    Called at every Phase 10B stage boundary. The invariant the plan states is
    absolute: the accepted move model before Phase 10B is byte-identical to the
    accepted move model after it.
    """
    path = Path(repository_root) / ACCEPTED_PHASE9_CHECKPOINT
    if not path.exists():
        raise Phase10BContractError(
            f"the accepted Phase 9 checkpoint is missing at {path}"
        )
    digest = file_sha256(path)
    if digest != ACCEPTED_PHASE9_SHA256:
        raise Phase10BContractError(
            f"the accepted Phase 9 checkpoint SHA-256 is {digest}, expected "
            f"{ACCEPTED_PHASE9_SHA256}; Phase 10B is BLOCKED"
        )
    return {"path": str(path), "sha256": digest, "unchanged": True}


def export_evaluation_weights(source: "str | Path", export_path: "str | Path") -> dict:
    """Export a Phase 10B *or* accepted Phase 9 checkpoint to the eval format.

    The accepted Phase 9 Agent 8 procedure, unchanged in substance: the source
    is opened read-only and the export is refused unless every tensor
    round-trips bitwise, so an evaluation can never silently measure weights
    that are not the checkpoint's.
    """
    from ..model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from ..model.checkpoint import load_checkpoint, save_checkpoint

    source = Path(source)
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    source_sha = file_sha256(source)

    container = _read_any_payload(source)
    if container["kind"] == "phase10b":
        model = model_from_payload(container["payload"])
    else:
        from .phase9_checkpoint import model_from_payload as phase9_model

        model = phase9_model(container["payload"])
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
        raise Phase10BCheckpointError(
            f"the evaluation export of {source} changed the weights; BLOCKED"
        )
    if state_dict_digest(reloaded) != digest:
        raise Phase10BCheckpointError(
            f"the evaluation export of {source} changed the model-state digest"
        )
    if file_sha256(source) != source_sha:
        raise Phase10BCheckpointError(f"{source} changed while it was being exported")
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


def archive_path(root: "str | Path", identity: str) -> Path:
    return Path(root) / "archive" / f"{identity}.pt"


def anchor_identity_record(repository_root: "str | Path") -> dict:
    """The anchor's real, addressable identity for the history manifest."""
    path = Path(repository_root) / ACCEPTED_PHASE9_CHECKPOINT
    return {
        "identity": ANCHOR_IDENTITY,
        "path": str(path),
        "sha256": file_sha256(path),
        "role": "the accepted Phase 9 move model; never evicted, never written",
    }


def manifest(directory: "str | Path") -> dict:
    """Every Phase 10B checkpoint on disk, with its identities and digests."""
    directory = Path(directory)
    members = []
    if directory.exists():
        for path in sorted(directory.rglob("*.pt")):
            try:
                payload = read(path)
            except Phase10BCheckpointError as error:
                members.append({"path": str(path), "error": str(error)})
                continue
            members.append(
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "snapshot_role": payload["snapshot_role"],
                    "rl_iteration": int(payload["rl_iteration"]),
                    "global_optimizer_step": int(payload["global_optimizer_step"]),
                    "model_state_digest": payload["model_state_digest"],
                }
            )
    return {
        "directory": str(directory),
        "checkpoint_version": PHASE10B_CHECKPOINT_VERSION,
        "anchor_identity": ANCHOR_IDENTITY,
        "members": members,
    }


def write_json(path: "str | Path", payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


__all__ = [
    "CANONICAL_CANDIDATE_PATH",
    "CHECKPOINT_DIRECTORY",
    "PHASE10B_CHECKPOINT_VERSION",
    "Phase10BCheckpointError",
    "SnapshotResolver",
    "anchor_identity_record",
    "archive_path",
    "assert_phase9_untouched",
    "export_evaluation_weights",
    "build_payload",
    "manifest",
    "model_from_payload",
    "read",
    "save",
    "upstream_bindings",
    "write_json",
]
