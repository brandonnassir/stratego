"""Phase 17 Agent 4: immutable paired evaluation exports.

Specification sources: Agent 4 instruction section 7, common contract sections
10 and 11.

An export is a *reading*, never an event in the run
---------------------------------------------------
Creating a candidate must not mutate training RNG, raw weights, EMA state or
timing counters. That is why nothing here calls into a live optimizer, samples
anything, or advances a counter: the EMA state dictionaries are cloned onto CPU
and written, and the run's own bookkeeping is passed in already computed. A
export that perturbed the run would make the 25 candidates a sequence of
slightly different experiments rather than 25 views of one.

EMA only, and the paired digest even when a lane ignores it
-----------------------------------------------------------
Raw weights generate training data; EMA weights are what evaluation sees. A
`move_only` lane does not consume the setup model, but the bundle still records
the setup EMA digest and marks `consumes_setup: false` -- so a later reader can
prove which setup policy was live when a move-only number was taken, instead of
inferring it from a timestamp.

Cadence is measured in active training time
--------------------------------------------
Not wall clock. A run that is paused, or whose host sleeps, must not emit a
burst of candidates on resume; and hour 0 is defined as *before the first
optimizer update*, not "the first thirty minutes". :func:`due_boundaries`
returns every 30-minute boundary the elapsed active-training time has crossed,
so a long iteration that spans two boundaries emits both rather than dropping
one.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import Phase17CheckpointError, file_sha256, json_digest
from .move_contract import (
    ACTION_ENCODING_VERSION,
    OBSERVATION_VERSION,
    RULES_VERSION,
    Phase17MoveError,
)

EXPORT_SCHEMA_VERSION = "phase17_paired_export_v1"

#: Contract section 11: hour 0 and every 30 minutes through hour 12.
EXPORT_INTERVAL_SECONDS = 30 * 60
EXPORT_HORIZON_SECONDS = 12 * 60 * 60
EXPECTED_CANDIDATES = EXPORT_HORIZON_SECONDS // EXPORT_INTERVAL_SECONDS + 1  # 25


class Phase17ExportError(Phase17MoveError):
    """A paired evaluation export could not be created as specified."""


def due_boundaries(previous_seconds: float, elapsed_seconds: float) -> list:
    """Every 30-minute boundary crossed in `(previous, elapsed]`, in order.

    Hour 0 is handled by the caller before the first optimizer update; this
    function covers boundary 1 onward. A single long iteration that spans two
    boundaries yields both: dropping one would silently shorten the cadence and
    the 25-candidate contract would quietly become 24.
    """
    first = int(previous_seconds // EXPORT_INTERVAL_SECONDS) + 1
    last = int(elapsed_seconds // EXPORT_INTERVAL_SECONDS)
    return [
        index
        for index in range(first, last + 1)
        if index * EXPORT_INTERVAL_SECONDS <= EXPORT_HORIZON_SECONDS
    ]


def candidate_id(run_id: str, index: int) -> str:
    """Immutable candidate name. Never a mutable `latest`."""
    return f"{run_id}-cand-{int(index):03d}"


@dataclass(frozen=True)
class PairedCandidate:
    """One written, immutable, digest-bound evaluation candidate."""

    candidate_id: str
    index: int
    path: str
    file_sha256: str
    manifest_digest: str
    move_ema_model_state_digest: str
    setup_ema_model_state_digest: str
    elapsed_active_training_seconds: float

    def document(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "index": int(self.index),
            "path": self.path,
            "file_sha256": self.file_sha256,
            "manifest_digest": self.manifest_digest,
            "move_ema_model_state_digest": self.move_ema_model_state_digest,
            "setup_ema_model_state_digest": self.setup_ema_model_state_digest,
            "elapsed_active_training_seconds": float(
                self.elapsed_active_training_seconds
            ),
        }


def _cpu_state(state: dict) -> dict:
    return {
        name: torch.as_tensor(tensor).detach().to("cpu").clone()
        for name, tensor in state.items()
    }


def build_manifest(
    *,
    run_id: str,
    index: int,
    move_ema_digest: str,
    setup_ema_digest: str,
    move_parameter_count: int,
    setup_parameter_count: int,
    start_identity: dict,
    parent_checkpoint: dict,
    config_digest: str,
    source_digest: str,
    elapsed_active_training_seconds: float,
    iteration: int,
    lanes: "dict | None" = None,
) -> dict:
    """Everything Agent 4 instruction section 7 requires a candidate to bind."""
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "work_package": "phase17",
        "run_id": run_id,
        "candidate_id": candidate_id(run_id, index),
        "candidate_index": int(index),
        "iteration": int(iteration),
        "elapsed_active_training_seconds": float(elapsed_active_training_seconds),
        "nominal_cadence_seconds": EXPORT_INTERVAL_SECONDS,
        "nominal_boundary_seconds": int(index) * EXPORT_INTERVAL_SECONDS,
        "weights": "EMA only; raw weights generate training data and are never exported",
        "move_ema_model_state_digest": move_ema_digest,
        "setup_ema_model_state_digest": setup_ema_digest,
        "move_parameter_count": int(move_parameter_count),
        "setup_parameter_count": int(setup_parameter_count),
        "identities": {
            "rules_version": RULES_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "action_encoding_version": ACTION_ENCODING_VERSION,
            "move_architecture": "phase9_selfplay_c1",
            "setup_architecture": "phase17_setup_model_v1 (4x128, 4 heads, FF512)",
        },
        "start_identity": dict(start_identity),
        "parent_checkpoint_identity": dict(parent_checkpoint),
        "config_digest": config_digest,
        "source_digest": source_digest,
        "lanes": dict(
            lanes
            or {
                "move_only": {"consumes_move": True, "consumes_setup": False},
                "joint_move_setup": {"consumes_move": True, "consumes_setup": True},
            }
        ),
    }


def write_paired_export(
    *,
    directory: "str | Path",
    manifest: dict,
    move_ema_state: dict,
    setup_ema_state: dict,
) -> PairedCandidate:
    """Write one immutable paired candidate; verify its own hashes afterwards.

    The bundle is a single file written through a temporary name and renamed,
    so a transfer that starts while the writer is running cannot pick up a
    partial candidate under its final name.
    """
    target_directory = Path(directory)
    target_directory.mkdir(parents=True, exist_ok=True)
    name = manifest["candidate_id"]
    target = target_directory / f"{name}.pt"
    if target.exists():
        raise Phase17ExportError(
            f"{target} already exists; every candidate is immutable and no "
            "candidate name is ever reused"
        )

    payload = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "manifest": manifest,
        "move_ema_state": _cpu_state(move_ema_state),
        "setup_ema_state": _cpu_state(setup_ema_state),
    }
    digest = json_digest(manifest)
    payload["manifest_digest"] = digest

    handle, temporary = tempfile.mkstemp(
        dir=str(target_directory), prefix=target.name + ".", suffix=".partial"
    )
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as sink:
            torch.save(payload, sink)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    reread = torch.load(target, map_location="cpu", weights_only=False)
    if reread["manifest_digest"] != digest:
        raise Phase17ExportError(f"{target} does not reproduce its manifest digest")
    if json_digest(reread["manifest"]) != digest:
        raise Phase17ExportError(f"{target} manifest re-digests differently")
    return PairedCandidate(
        candidate_id=name,
        index=int(manifest["candidate_index"]),
        path=str(target),
        file_sha256=file_sha256(target),
        manifest_digest=digest,
        move_ema_model_state_digest=manifest["move_ema_model_state_digest"],
        setup_ema_model_state_digest=manifest["setup_ema_model_state_digest"],
        elapsed_active_training_seconds=float(
            manifest["elapsed_active_training_seconds"]
        ),
    )


def verify_paired_export(path: "str | Path", *, expected_file_sha256: "str | None" = None) -> dict:
    """Re-verify a written candidate the way the remote evaluator will."""
    target = Path(path)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    manifest = payload["manifest"]
    observed = json_digest(manifest)
    if observed != payload["manifest_digest"]:
        raise Phase17ExportError(f"{target}: manifest digest {observed} != recorded")
    move_digest = _state_mapping_digest(payload["move_ema_state"])
    setup_digest = _state_mapping_digest(payload["setup_ema_state"])
    if move_digest != manifest["move_ema_model_state_digest"]:
        raise Phase17ExportError(
            f"{target}: move EMA digests to {move_digest}, not "
            f"{manifest['move_ema_model_state_digest']}"
        )
    if setup_digest != manifest["setup_ema_model_state_digest"]:
        raise Phase17ExportError(
            f"{target}: setup EMA digests to {setup_digest}, not "
            f"{manifest['setup_ema_model_state_digest']}"
        )
    file_digest = file_sha256(target)
    if expected_file_sha256 is not None and file_digest != expected_file_sha256:
        raise Phase17ExportError(
            f"{target}: file sha256 {file_digest}, not {expected_file_sha256}"
        )
    return {
        "candidate_id": manifest["candidate_id"],
        "manifest_digest": observed,
        "move_ema_model_state_digest": move_digest,
        "setup_ema_model_state_digest": setup_digest,
        "file_sha256": file_digest,
        "verified": True,
    }


def _state_mapping_digest(state: dict) -> str:
    """The accepted `state_dict_digest` walk over a plain mapping."""
    hasher = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        hasher.update(name.encode())
        array = torch.as_tensor(tensor).detach().to("cpu", torch.float32).contiguous().numpy()
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def export_schema() -> dict:
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "cadence": "hour 0 before the first optimizer update, then every 30 minutes of ACTIVE TRAINING time through hour 12",
        "expected_candidates": EXPECTED_CANDIDATES,
        "interval_seconds": EXPORT_INTERVAL_SECONDS,
        "horizon_seconds": EXPORT_HORIZON_SECONDS,
        "immutability": "one file per candidate, written to a temporary name and renamed; never a mutable 'latest'",
        "purity": "creation mutates no training RNG, no raw weights, no EMA state and no timing counter",
        "contents": [
            "manifest",
            "move_ema_state",
            "setup_ema_state",
            "manifest_digest",
        ],
        "agent_5_boundary": (
            "Agent 5 may wrap this bundle for its chosen transport but may not "
            "change its semantic identities"
        ),
    }


__all__ = [
    "EXPECTED_CANDIDATES",
    "EXPORT_HORIZON_SECONDS",
    "EXPORT_INTERVAL_SECONDS",
    "EXPORT_SCHEMA_VERSION",
    "PairedCandidate",
    "Phase17ExportError",
    "build_manifest",
    "candidate_id",
    "due_boundaries",
    "export_schema",
    "verify_paired_export",
    "write_paired_export",
]
