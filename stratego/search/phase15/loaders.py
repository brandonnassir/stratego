"""Phase 15 Agent 2 section 6: digest-bound loading of every model.

Specification source: `02_AGENT_2_SEARCH_IMPLEMENTATION.md` sections 1 and 6.

Bound to bytes, not to labels
-----------------------------
Nothing here is loaded from a path alone. Every load re-derives the file's
sha256 and the loaded object's state digest and compares both against the
Agent 1 handoff (`phase15_search_handoff_v1`), which is itself checked for
its artifact name before a single tensor is read. A caller cannot end up
holding a different object than the one the handoff document names, and
section 1's "verify every handoff digest before implementation or
evaluation" is therefore a property of the loader rather than of a checklist.

Four objects, three kinds
-------------------------
```text
P18, P24            frozen Phase 14 policy/value models  (policy, value, rollouts, fallback)
B18, B24            Phase 15 belief specialists          (marginals only)
phase9 anchor       the accepted Phase 9 C1              (a match opponent, nothing else)
```

`load_specialist` refuses to attach a belief checkpoint to a backbone whose
model-state digest is not the one recorded at save time, so B18 can only
ever be built over P18's prefix and B24 over P24's — which is a statement
about the *frozen prefix the specialist owns a copy of*, not about which
move model it may be paired with. Cross-pairing (P18+B24, P24+B18) is
intentional and happens one level up, in the provider and the engine.

The anchor's export path is a Phase 15 path
-------------------------------------------
The accepted `load_frozen_c1` helper re-exports the accepted Phase 9
checkpoint into the export path it is given. Phase 12's own export
(`checkpoints/phase12/phase9_c1_readonly_copy.pt`) is a frozen Phase 12
artifact, so this module exports to `checkpoints/phase15/` instead and never
writes a byte inside another phase's directory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .contract import (
    HANDOFF_ARTIFACT,
    LEARNED_PROVIDERS,
    MOVE_MODELS,
    PROVIDER_BACKBONE,
    Phase15SearchError,
)

#: Where the Agent 1 handoff lives, relative to the repository root.
DEFAULT_HANDOFF_PATH = Path("reports/phase15/phase15_search_handoff_v1.json")

#: The Phase 15 export of the accepted Phase 9 C1, used as a match opponent.
PHASE9_ANCHOR_EXPORT = Path("checkpoints/phase15/phase9_anchor_readonly.pt")

#: The loader identity a report and the frozen candidate record.
LOADER_VERSION = "phase15_search_loader_v1"


class Phase15LoadError(Phase15SearchError):
    """A model could not be loaded, or did not match its bound digest."""


def file_sha256(path: "Path | str", *, chunk: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def load_handoff(path: "Path | str | None" = None, *, root: "Path | str" = ".") -> dict:
    """The Agent 1 handoff, with its artifact name and shape checked."""
    path = Path(root) / DEFAULT_HANDOFF_PATH if path is None else Path(path)
    if not path.is_file():
        raise Phase15LoadError(
            f"no Agent 1 handoff at {path}; Agent 2 may not run until Agent 1 has "
            f"delivered {HANDOFF_ARTIFACT}"
        )
    handoff = json.loads(path.read_text())
    if handoff.get("artifact") != HANDOFF_ARTIFACT:
        raise Phase15LoadError(f"{path} is not a {HANDOFF_ARTIFACT} document")
    for section, keys in (
        ("policy_models", MOVE_MODELS),
        ("belief_models", LEARNED_PROVIDERS),
    ):
        block = handoff.get(section) or {}
        missing = [key for key in keys if key not in block]
        if missing:
            raise Phase15LoadError(
                f"{path}: handoff section {section!r} is missing {missing}"
            )
    if "corpus" not in handoff:
        raise Phase15LoadError(f"{path}: handoff carries no corpus identity")
    return handoff


# ---------------------------------------------------------------------------
# Move models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedMoveModel:
    """One frozen Phase 14 policy/value model plus its verified identity."""

    move_model: str
    model: object
    identity: dict

    def describe(self) -> dict:
        return dict(self.identity)


def load_move_model(
    move_model: str,
    handoff: dict,
    *,
    root: "Path | str" = ".",
    device: str = "cpu",
) -> LoadedMoveModel:
    """P18 or P24, digest-checked against the handoff before it is returned."""
    import torch

    from ...model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from ...model.checkpoint import load_checkpoint
    from ...training.phase9_behavior import state_dict_digest

    if move_model not in MOVE_MODELS:
        raise Phase15LoadError(
            f"move model must be one of {list(MOVE_MODELS)}, got {move_model!r}"
        )
    record = handoff["policy_models"][move_model]
    path = Path(root) / record["checkpoint_path"]
    if not path.is_file():
        raise Phase15LoadError(f"{move_model}: no checkpoint at {path}")
    observed_sha = file_sha256(path)
    if observed_sha != record["checkpoint_sha256"]:
        raise Phase15LoadError(
            f"{move_model}: {path} has sha256 {observed_sha}, the handoff records "
            f"{record['checkpoint_sha256']}; refusing to load unbound bytes"
        )
    model, metadata = load_checkpoint(
        path,
        device=torch.device(device),
        dtype=torch.float32,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    observed_state = state_dict_digest(model)
    if observed_state != record["model_state_digest"]:
        raise Phase15LoadError(
            f"{move_model}: loaded model-state digest {observed_state} != handoff "
            f"{record['model_state_digest']}"
        )
    identity = {
        "move_model": move_model,
        "logical_identity": record["logical_identity"],
        "checkpoint_path": str(path),
        "checkpoint_sha256": observed_sha,
        "model_state_digest": observed_state,
        "global_optimizer_step": int(record["global_optimizer_step"]),
        "phase14_candidate_hour": int(record["phase14_candidate_hour"]),
        "phase14_archive_sha256": record["phase14_archive_sha256"],
        "architecture_id": getattr(model, "architecture_id", None),
        "role": "policy, value, rollout policy for both sides, direct fallback",
        "trained_by_phase15": False,
        "checkpoint_metadata_keys": sorted(metadata) if isinstance(metadata, dict) else [],
    }
    return LoadedMoveModel(move_model=move_model, model=model, identity=identity)


def load_move_models(
    handoff: dict, *, root: "Path | str" = ".", device: str = "cpu"
) -> "dict[str, LoadedMoveModel]":
    """Both frozen move models, keyed by `p18` / `p24`."""
    return {
        name: load_move_model(name, handoff, root=root, device=device)
        for name in MOVE_MODELS
    }


# ---------------------------------------------------------------------------
# Belief specialists
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedSpecialist:
    """One Phase 15 belief specialist over its own frozen prefix."""

    provider_id: str
    specialist: object
    backbone: str
    identity: dict

    def describe(self) -> dict:
        return dict(self.identity)


def load_belief_specialist(
    provider_id: str,
    handoff: dict,
    move_models: "dict[str, LoadedMoveModel]",
    *,
    root: "Path | str" = ".",
    device: str = "cpu",
) -> LoadedSpecialist:
    """B18 or B24, over the frozen prefix its checkpoint was built from.

    The backbone is the *prefix owner* named by `PROVIDER_BACKBONE`, never
    the move model the caller intends to pair the beliefs with;
    `load_specialist` enforces that with its own digest refusal.
    """
    from ...belief.phase15.checkpoint import load_specialist

    if provider_id not in LEARNED_PROVIDERS:
        raise Phase15LoadError(
            f"belief provider must be one of {list(LEARNED_PROVIDERS)}, got "
            f"{provider_id!r}"
        )
    record = handoff["belief_models"][provider_id]
    path = Path(root) / record["checkpoint_path"]
    if not path.is_file():
        raise Phase15LoadError(f"{provider_id}: no checkpoint at {path}")
    observed_sha = file_sha256(path)
    if observed_sha != record["checkpoint_sha256"]:
        raise Phase15LoadError(
            f"{provider_id}: {path} has sha256 {observed_sha}, the handoff records "
            f"{record['checkpoint_sha256']}; refusing to load unbound bytes"
        )
    backbone_name = PROVIDER_BACKBONE[provider_id]
    if record.get("bound_policy") != backbone_name:
        raise Phase15LoadError(
            f"{provider_id}: the handoff binds it to {record.get('bound_policy')!r}, "
            f"the contract binds it to {backbone_name!r}"
        )
    backbone = move_models[backbone_name]
    specialist, payload = load_specialist(path, backbone.model, device=device)
    if payload["state_digest"] != record["state_digest"]:
        raise Phase15LoadError(
            f"{provider_id}: state digest {payload['state_digest']} != handoff "
            f"{record['state_digest']}"
        )
    calibration = dict(record.get("calibration") or {})
    applied = float(calibration.get("applied_temperature", 1.0))
    if abs(float(specialist.temperature) - applied) > 1e-9:
        raise Phase15LoadError(
            f"{provider_id}: the checkpoint carries temperature "
            f"{specialist.temperature}, the handoff records an applied temperature "
            f"of {applied}"
        )
    identity = {
        "provider_id": provider_id,
        "checkpoint_path": str(path),
        "checkpoint_sha256": observed_sha,
        "state_digest": payload["state_digest"],
        "architecture_version": payload["architecture_version"],
        "prefix_backbone": backbone_name,
        "prefix_backbone_state_digest": backbone.identity["model_state_digest"],
        "applied_temperature": applied,
        "fitted_temperature": calibration.get("fitted_temperature"),
        "keep_calibrated": calibration.get("keep_calibrated"),
        "holds_policy_parameters": bool(payload.get("holds_policy_parameters")),
        "holds_value_parameters": bool(payload.get("holds_value_parameters")),
        "corpus": dict(payload.get("corpus") or {}),
        "role": "hidden-rank marginals and legal hidden-world sampling only",
    }
    if identity["holds_policy_parameters"] or identity["holds_value_parameters"]:
        raise Phase15LoadError(
            f"{provider_id}: the checkpoint claims to hold policy or value "
            "parameters; a belief specialist holds neither"
        )
    return LoadedSpecialist(
        provider_id=provider_id,
        specialist=specialist,
        backbone=backbone_name,
        identity=identity,
    )


def load_belief_specialists(
    handoff: dict,
    move_models: "dict[str, LoadedMoveModel]",
    *,
    root: "Path | str" = ".",
    device: str = "cpu",
) -> "dict[str, LoadedSpecialist]":
    """Both belief specialists, keyed by `b18` / `b24`."""
    return {
        name: load_belief_specialist(
            name, handoff, move_models, root=root, device=device
        )
        for name in LEARNED_PROVIDERS
    }


# ---------------------------------------------------------------------------
# The accepted Phase 9 anchor (a match opponent)
# ---------------------------------------------------------------------------


def load_phase9_anchor(*, root: "Path | str" = ".", device: str = "cpu"):
    """`(model, identity)`: the accepted Phase 9 C1, for the anchor seat.

    Exported into `checkpoints/phase15/` so no Phase 12 byte is rewritten.
    This model is an *opponent*: it never provides a Phase 15 policy, value,
    rollout or belief.

    Reuses an existing export rather than rewriting it. The accepted
    `load_frozen_c1` re-exports unconditionally, and it writes through a
    single `<path>.partial` temporary — so ten worker processes calling it at
    once on the same path race each other and all but one fail with a missing
    temporary file. The export is a pure function of the accepted checkpoint,
    so an export whose loaded state digest already equals the accepted digest
    is *the* export and can simply be loaded. :func:`ensure_phase9_anchor`
    creates it once, in the parent, before any pool exists.
    """
    import torch

    from ...model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from ...model.checkpoint import load_checkpoint
    from ...belief.phase11b.features import load_frozen_c1
    from ...training.phase11_contract import (
        ACCEPTED_BELIEF_HEAD_DIGEST,
        ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
    )
    from ...training.phase9_behavior import state_dict_digest

    root = Path(root)
    export = root / PHASE9_ANCHOR_EXPORT
    if export.is_file():
        from ...belief.phase11b.features import belief_head_digest

        try:
            model, _metadata = load_checkpoint(
                export,
                device=torch.device(device),
                dtype=torch.float32,
                expected_architecture_id=ARCHITECTURE_FAMILY,
                expected_configuration=candidate_config("C1"),
            )
        except Exception:  # noqa: BLE001 - a damaged export is re-created below
            model = None
        if model is not None:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            observed_state = state_dict_digest(model)
            observed_head = belief_head_digest(model)
            if (
                observed_state == ACCEPTED_PHASE9_MODEL_STATE_DIGEST
                and observed_head == ACCEPTED_BELIEF_HEAD_DIGEST
            ):
                return model, {
                    "model_state_digest": observed_state,
                    "belief_head_digest": observed_head,
                    "export_path": str(export),
                    "export_sha256": file_sha256(export),
                    "reused_existing_export": True,
                    "role": "match opponent only",
                }
            raise Phase15LoadError(
                f"{export} does not carry the accepted Phase 9 digests "
                f"({observed_state} / {observed_head}); refusing to use it as the "
                "anchor"
            )
    model, identity = load_frozen_c1(root, export, device=device)
    record = dict(identity)
    record["role"] = "match opponent only"
    record["export_path"] = str(export)
    record["export_sha256"] = file_sha256(export)
    record["reused_existing_export"] = False
    return model, record


def ensure_phase9_anchor(*, root: "Path | str" = ".") -> Path:
    """Create the anchor export if it is missing. Call before forking workers."""
    export = Path(root) / PHASE9_ANCHOR_EXPORT
    if not export.is_file():
        load_phase9_anchor(root=root, device="cpu")
    return export


# ---------------------------------------------------------------------------
# Everything at once
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase15Models:
    """Every model Agent 2 needs, loaded once and shared by every arm."""

    handoff: dict
    handoff_path: str
    move_models: dict
    specialists: dict
    anchor: object = None
    anchor_identity: dict = None

    def move(self, name: str):
        try:
            return self.move_models[name].model
        except KeyError:
            raise Phase15LoadError(f"no move model {name!r} was loaded") from None

    def identities(self) -> dict:
        return {
            "loader_version": LOADER_VERSION,
            "handoff_path": self.handoff_path,
            "handoff_artifact": self.handoff.get("artifact"),
            "corpus": dict(self.handoff.get("corpus") or {}),
            "move_models": {
                name: loaded.identity for name, loaded in self.move_models.items()
            },
            "belief_models": {
                name: loaded.identity for name, loaded in self.specialists.items()
            },
            "phase9_anchor": dict(self.anchor_identity or {}),
        }


def load_all(
    *,
    root: "Path | str" = ".",
    device: str = "cpu",
    handoff_path: "Path | str | None" = None,
    with_anchor: bool = True,
) -> Phase15Models:
    """Load and verify the whole Phase 15 stack in one call."""
    root = Path(root)
    path = Path(root) / DEFAULT_HANDOFF_PATH if handoff_path is None else Path(handoff_path)
    handoff = load_handoff(path, root=root)
    move_models = load_move_models(handoff, root=root, device=device)
    specialists = load_belief_specialists(handoff, move_models, root=root, device=device)
    anchor = anchor_identity = None
    if with_anchor:
        anchor, anchor_identity = load_phase9_anchor(root=root, device=device)
    return Phase15Models(
        handoff=handoff,
        handoff_path=str(path),
        move_models=move_models,
        specialists=specialists,
        anchor=anchor,
        anchor_identity=anchor_identity,
    )


__all__ = [
    "DEFAULT_HANDOFF_PATH",
    "LOADER_VERSION",
    "LoadedMoveModel",
    "LoadedSpecialist",
    "PHASE9_ANCHOR_EXPORT",
    "Phase15LoadError",
    "Phase15Models",
    "file_sha256",
    "load_all",
    "load_belief_specialist",
    "load_belief_specialists",
    "load_handoff",
    "load_move_model",
    "load_move_models",
    "load_phase9_anchor",
    "ensure_phase9_anchor",
]
