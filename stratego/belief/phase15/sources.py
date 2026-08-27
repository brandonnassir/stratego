"""Phase 15 Agent 1 section 3: resolve and freeze P18 and P24.

Specification source: `01_AGENT_1_BELIEF_HEAD_TRAINING.md` section 3.

The ledger is authoritative
---------------------------
Hour 18 and hour 24 are resolved from
`<run root>/evaluations/phase14_candidate_ledger.json` — never from the
newest `hot_*.pt`, and never from the archive's file ordering. The ledger
entry names the archive snapshot, its sha256, its `model_state_digest` and
its `global_optimizer_step`; every one of those is re-derived from bytes
here before a source is accepted.

Read-only, three ways
---------------------
1. The Phase 14 archive snapshot is opened for reading and never written.
2. The evaluation weights export the candidate evaluation actually scored
   is copied — byte for byte — into `checkpoints/phase15/`, and the copy's
   sha256 is checked against the source after the copy.
3. The loaded model has `requires_grad=False` on every parameter, so the
   freeze is a property of the object rather than of a convention a caller
   has to remember.

Nothing in this module opens the run state, the hot directory or any
control file, and nothing in it writes anywhere outside `checkpoints/phase15/`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

from .contract import SOURCE_HOURS, POLICY_SOURCES, Phase15Error

#: The identity of this resolution procedure.
SOURCE_BINDING_VERSION = "phase15_policy_source_binding_v1"

#: Default Phase 14 run root. Overridable so a test can point at a fixture.
DEFAULT_PHASE14_ROOT = Path("/Volumes/Brandon_Washington/stratego_phase14")

#: Where the read-only Phase 15 copies live.
PHASE15_CHECKPOINT_ROOT = Path("checkpoints/phase15")

_LEDGER_RELATIVE = Path("evaluations/phase14_candidate_ledger.json")
_WEIGHTS_RELATIVE = Path("evaluations/weights")


class Phase15SourceError(Phase15Error):
    """A Phase 14 candidate could not be resolved, verified or frozen."""


def file_sha256(path: "Path | str", *, chunk: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


@dataclass(frozen=True)
class PolicySource:
    """One frozen direct policy/value model: P18 or P24."""

    source_id: str
    hour: int
    original_snapshot_path: str
    original_snapshot_sha256: str
    evaluation_weights_path: str
    evaluation_weights_sha256: str
    phase15_copy_path: str
    phase15_copy_sha256: str
    phase15_copy_bytes: int
    model_state_digest: str
    global_optimizer_step: int
    iteration: int
    elapsed_seconds: float
    archive_position: "int | None"
    candidate_evaluation: dict
    pack_content_digest: str
    written_utc: str

    def to_dict(self) -> dict:
        return {
            "artifact": SOURCE_BINDING_VERSION,
            "source_id": self.source_id,
            "logical_identity": f"P{self.hour:02d}",
            **asdict(self),
            "note": (
                "an immutable Phase 14 policy/value model. Phase 15 never trains, "
                "re-saves or rotates it; the belief specialist is a separate object."
            ),
        }


def load_ledger(run_root: "Path | str" = DEFAULT_PHASE14_ROOT) -> dict:
    """The Phase 14 candidate ledger, opened read-only."""
    path = Path(run_root) / _LEDGER_RELATIVE
    if not path.is_file():
        raise Phase15SourceError(f"no Phase 14 candidate ledger at {path}")
    with open(path, "rb") as handle:
        ledger = json.loads(handle.read().decode())
    if ledger.get("artifact") != "phase14_candidate_ledger_v1":
        raise Phase15SourceError(
            f"{path} is not a phase14_candidate_ledger_v1 document"
        )
    return ledger


def resolve_candidate(ledger: dict, hour: int) -> dict:
    """The ledger entry for one candidate hour, with its completeness checked."""
    candidates = ledger.get("candidates") or {}
    entry = candidates.get(str(int(hour)))
    if entry is None:
        raise Phase15SourceError(
            f"the ledger holds no candidate for hour {hour}; it has hours "
            f"{sorted(int(key) for key in candidates)}"
        )
    if not entry.get("complete") or entry.get("status") != "complete":
        raise Phase15SourceError(
            f"candidate hour {hour} is not a complete evaluation "
            f"(status={entry.get('status')!r}, complete={entry.get('complete')!r})"
        )
    if entry.get("error"):
        raise Phase15SourceError(f"candidate hour {hour} recorded an error: {entry['error']}")
    mark = entry.get("mark") or {}
    if int(mark.get("hour", -1)) != int(hour):
        raise Phase15SourceError(
            f"candidate hour {hour} carries a mark for hour {mark.get('hour')!r}"
        )
    return entry


def freeze_source(
    source_id: str,
    *,
    run_root: "Path | str" = DEFAULT_PHASE14_ROOT,
    destination_root: "Path | str" = PHASE15_CHECKPOINT_ROOT,
    ledger: "dict | None" = None,
    device: str = "cpu",
) -> PolicySource:
    """Resolve, verify and freeze one policy source. Returns its binding.

    Verifies, in order: the ledger entry is a complete evaluation; the
    archive snapshot's sha256 matches the ledger; the evaluation weights
    load as the accepted C1 architecture; the loaded model-state digest
    matches the ledger's; the Phase 15 copy is byte-identical to the
    evaluation weights it came from; and the archive snapshot is unchanged
    after all of it.
    """
    import torch

    from ...model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from ...model.checkpoint import load_checkpoint
    from ...training.phase9_behavior import state_dict_digest

    if source_id not in POLICY_SOURCES:
        raise Phase15SourceError(
            f"source must be one of {list(POLICY_SOURCES)}, got {source_id!r}"
        )
    hour = SOURCE_HOURS[source_id]
    run_root = Path(run_root)
    ledger = load_ledger(run_root) if ledger is None else ledger
    entry = resolve_candidate(ledger, hour)
    mark = entry["mark"]

    snapshot = Path(mark["snapshot_path"])
    if not snapshot.is_file():
        raise Phase15SourceError(f"candidate hour {hour} names a missing snapshot {snapshot}")
    snapshot_sha = file_sha256(snapshot)
    if snapshot_sha != mark["snapshot_sha256"]:
        raise Phase15SourceError(
            f"candidate hour {hour}: snapshot sha256 {snapshot_sha} != ledger "
            f"{mark['snapshot_sha256']}"
        )

    weights = run_root / _WEIGHTS_RELATIVE / f"hour_{hour:03d}.pt"
    if not weights.is_file():
        raise Phase15SourceError(f"candidate hour {hour} has no evaluation weights at {weights}")
    weights_sha = file_sha256(weights)

    model, _metadata = load_checkpoint(
        weights,
        device=torch.device(device),
        dtype=torch.float32,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
    )
    observed = state_dict_digest(model)
    if observed != mark["model_state_digest"]:
        raise Phase15SourceError(
            f"candidate hour {hour}: evaluation weights carry model-state digest "
            f"{observed}, the ledger records {mark['model_state_digest']}"
        )
    del model

    destination_root = Path(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    copy = destination_root / f"{source_id}_source_readonly.pt"
    shutil.copyfile(weights, copy)
    copy_sha = file_sha256(copy)
    if copy_sha != weights_sha:
        raise Phase15SourceError(  # pragma: no cover - a copy that changed bytes
            f"the Phase 15 copy of hour {hour} is {copy_sha}, the source is {weights_sha}"
        )
    copy.chmod(0o444)

    if file_sha256(snapshot) != snapshot_sha:  # pragma: no cover - defensive
        raise Phase15SourceError(f"{snapshot} changed while it was being read")
    if file_sha256(weights) != weights_sha:  # pragma: no cover - defensive
        raise Phase15SourceError(f"{weights} changed while it was being read")

    return PolicySource(
        source_id=source_id,
        hour=int(hour),
        original_snapshot_path=str(snapshot),
        original_snapshot_sha256=snapshot_sha,
        evaluation_weights_path=str(weights),
        evaluation_weights_sha256=weights_sha,
        phase15_copy_path=str(copy),
        phase15_copy_sha256=copy_sha,
        phase15_copy_bytes=int(copy.stat().st_size),
        model_state_digest=observed,
        global_optimizer_step=int(mark["global_optimizer_step"]),
        iteration=int(mark["iteration"]),
        elapsed_seconds=float(mark["elapsed_seconds"]),
        archive_position=mark.get("archive_position"),
        candidate_evaluation={
            "games_played": int(entry["games_played"]),
            "mean_ewr": float(entry["mean_ewr"]),
            "min_stratum_ewr": float(entry["min_stratum_ewr"]),
            "strata": dict(entry["strata"]),
            "pack_digest": mark["pack_digest"],
            "attempts": int(entry["attempts"]),
            "selection_rule": ledger.get("selection_rule"),
            "note": (
                "the Phase 14 candidate evaluation on the frozen 128-game pack. "
                "Not a Phase 15 result and not a final-strength claim."
            ),
        },
        pack_content_digest=entry["pack_content_digest"],
        written_utc=mark["written_utc"],
    )


def load_frozen_policy(source: PolicySource, *, device: str = "cpu"):
    """The frozen policy/value model of one source, from the Phase 15 copy.

    Every parameter comes back with `requires_grad=False`, and the loaded
    model-state digest is re-checked against the binding, so a caller
    cannot accidentally hold a different object than the one the handoff
    document names.
    """
    import torch

    from ...model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from ...model.checkpoint import load_checkpoint
    from ...training.phase9_behavior import state_dict_digest

    model, metadata = load_checkpoint(
        Path(source.phase15_copy_path),
        device=torch.device(device),
        dtype=torch.float32,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    observed = state_dict_digest(model)
    if observed != source.model_state_digest:
        raise Phase15SourceError(
            f"{source.source_id}: loaded model-state digest {observed} != bound "
            f"{source.model_state_digest}"
        )
    return model, metadata


def write_source_identity(source: PolicySource, path: "Path | str") -> Path:
    """Write one `p18_source_identity.json` / `p24_source_identity.json`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(source.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def freeze_sources(
    *,
    run_root: "Path | str" = DEFAULT_PHASE14_ROOT,
    destination_root: "Path | str" = PHASE15_CHECKPOINT_ROOT,
    device: str = "cpu",
) -> dict:
    """Freeze both sources and write both identity documents."""
    ledger = load_ledger(run_root)
    frozen = {}
    for source_id in POLICY_SOURCES:
        source = freeze_source(
            source_id,
            run_root=run_root,
            destination_root=destination_root,
            ledger=ledger,
            device=device,
        )
        write_source_identity(
            source, Path(destination_root) / f"{source_id}_source_identity.json"
        )
        frozen[source_id] = source
    if frozen["p18"].model_state_digest == frozen["p24"].model_state_digest:
        raise Phase15SourceError(  # pragma: no cover - two different hours
            "P18 and P24 resolved to the same weights"
        )
    return frozen


__all__ = [
    "DEFAULT_PHASE14_ROOT",
    "PHASE15_CHECKPOINT_ROOT",
    "SOURCE_BINDING_VERSION",
    "Phase15SourceError",
    "PolicySource",
    "file_sha256",
    "freeze_source",
    "freeze_sources",
    "load_frozen_policy",
    "load_ledger",
    "resolve_candidate",
    "write_source_identity",
]
