"""Phase 17 Agent 5: loading a paired candidate, identity-first.

Specification sources: Agent 5 instruction section 3, common contract
sections 10-11, Agent 4's frozen `phase17_paired_export_v1`.

Nothing is loaded until everything is proved
--------------------------------------------
`verify_bundle` recomputes the manifest digest, both model-state digests and
the file's own sha256 from the bytes on disk, and compares the manifest's
rules/observation/action/architecture claims against the versions this build
actually implements. Only then are weights allowed into a model. The ordering
is the point: a bundle that arrived truncated, or that belongs to another run,
or that was produced by a build with a different action encoding, must be
refused *before* it can produce a plausible-looking EWR.

The refusal list is closed
--------------------------
Partial, duplicate-conflicting, stale-attributed and incompatible bundles each
have their own refusal with their own message, because "evaluation result bound
to the wrong candidate or benchmark" is a production stop condition and the
operator needs to know which one fired.

Why the move weights are written back out to a checkpoint
----------------------------------------------------------
The accepted inference path is `InferenceOwner(checkpoint_path, ...)`, and it
carries the architecture check, the dtype rule, the legality cross-check and
the greedy tie-break that every accepted evaluation in this repository has
used. Rather than reimplement any of that around a bare state dict, the
candidate's EMA weights are written into a real checkpoint in a temporary
directory and loaded through that same path. The candidate seat is therefore
running the identical decision code as the opponents it plays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .contract import (
    CANDIDATE_DECISION_MODE,
    EVALUATION_DEVICE,
    EVALUATION_DTYPE,
    EXPECTED_EXPORT_SCHEMA,
    Phase17BundleError,
    file_sha256,
    json_digest,
    state_mapping_digest,
)

#: The architecture a Phase 17 move candidate must be.
MOVE_CANDIDATE_ID = "C1"
MOVE_PARAMETER_COUNT = 863959
SETUP_PARAMETER_COUNT = 802320


@dataclass(frozen=True)
class VerifiedBundle:
    """One candidate whose every identity claim reproduced from its bytes."""

    path: str
    file_sha256: str
    manifest: dict
    manifest_digest: str
    candidate_id: str
    candidate_index: int
    run_id: str
    iteration: int
    elapsed_active_training_seconds: float
    move_ema_model_state_digest: str
    setup_ema_model_state_digest: str
    config_digest: str
    source_digest: str

    def identity(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "candidate_index": int(self.candidate_index),
            "run_id": self.run_id,
            "iteration": int(self.iteration),
            "elapsed_active_training_seconds": float(
                self.elapsed_active_training_seconds
            ),
            "file_sha256": self.file_sha256,
            "manifest_digest": self.manifest_digest,
            "move_ema_model_state_digest": self.move_ema_model_state_digest,
            "setup_ema_model_state_digest": self.setup_ema_model_state_digest,
            "config_digest": self.config_digest,
            "source_digest": self.source_digest,
        }


def verify_bundle(
    path: "Path | str",
    *,
    expected_file_sha256: "str | None" = None,
    expected_run_id: "str | None" = None,
    expected_candidate_id: "str | None" = None,
) -> "tuple[VerifiedBundle, dict]":
    """Recompute every identity from the bytes, then return the payload.

    Returns `(verified, payload)`. The payload's two state dicts are the only
    thing a caller may build a model from.
    """
    import torch

    from ...model.contract import (
        ACTION_ENCODING_VERSION,
        MODEL_CONTRACT_VERSION,
        OBSERVATION_VERSION,
        RULES_VERSION,
    )

    target = Path(path)
    if not target.is_file():
        raise Phase17BundleError(f"no candidate bundle at {target}")
    if target.name.endswith(".partial"):
        raise Phase17BundleError(
            f"{target} is a staging name; only an atomically published bundle "
            "may be evaluated"
        )

    observed_file = file_sha256(target)
    if expected_file_sha256 is not None and observed_file != expected_file_sha256:
        raise Phase17BundleError(
            f"{target}: file sha256 {observed_file} != the published "
            f"{expected_file_sha256}; the transfer is partial or corrupt"
        )

    try:
        payload = torch.load(target, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - the reason is reported verbatim
        raise Phase17BundleError(f"{target} could not be read: {error}") from error

    schema = payload.get("schema_version")
    if schema != EXPECTED_EXPORT_SCHEMA:
        raise Phase17BundleError(
            f"{target} carries export schema {schema!r}; this evaluator reads "
            f"{EXPECTED_EXPORT_SCHEMA!r} only"
        )
    for required in ("manifest", "manifest_digest", "move_ema_state", "setup_ema_state"):
        if required not in payload:
            raise Phase17BundleError(f"{target} is partial: no {required!r}")

    manifest = payload["manifest"]
    observed_manifest = json_digest(manifest)
    if observed_manifest != payload["manifest_digest"]:
        raise Phase17BundleError(
            f"{target}: manifest re-digests to {observed_manifest}, the bundle "
            f"records {payload['manifest_digest']}"
        )

    move_digest = state_mapping_digest(payload["move_ema_state"])
    setup_digest = state_mapping_digest(payload["setup_ema_state"])
    if move_digest != manifest["move_ema_model_state_digest"]:
        raise Phase17BundleError(
            f"{target}: move EMA digests to {move_digest}, the manifest claims "
            f"{manifest['move_ema_model_state_digest']}"
        )
    if setup_digest != manifest["setup_ema_model_state_digest"]:
        raise Phase17BundleError(
            f"{target}: setup EMA digests to {setup_digest}, the manifest claims "
            f"{manifest['setup_ema_model_state_digest']}"
        )

    identities = manifest.get("identities") or {}
    for field, expected in (
        ("rules_version", RULES_VERSION),
        ("observation_version", OBSERVATION_VERSION),
        ("action_encoding_version", ACTION_ENCODING_VERSION),
    ):
        observed = identities.get(field)
        if observed != expected:
            raise Phase17BundleError(
                f"{target}: manifest {field}={observed!r}, this build implements "
                f"{expected!r}; the two machines are not running the same game"
            )
    if int(manifest.get("move_parameter_count", -1)) != MOVE_PARAMETER_COUNT:
        raise Phase17BundleError(
            f"{target}: move parameter count {manifest.get('move_parameter_count')} "
            f"!= the accepted {MOVE_PARAMETER_COUNT}"
        )
    if int(manifest.get("setup_parameter_count", -1)) != SETUP_PARAMETER_COUNT:
        raise Phase17BundleError(
            f"{target}: setup parameter count "
            f"{manifest.get('setup_parameter_count')} != the accepted "
            f"{SETUP_PARAMETER_COUNT}"
        )

    if expected_run_id is not None and manifest.get("run_id") != expected_run_id:
        raise Phase17BundleError(
            f"{target}: bundle belongs to run {manifest.get('run_id')!r}, this "
            f"evaluation is bound to {expected_run_id!r}"
        )
    if (
        expected_candidate_id is not None
        and manifest.get("candidate_id") != expected_candidate_id
    ):
        raise Phase17BundleError(
            f"{target}: bundle is {manifest.get('candidate_id')!r} but was "
            f"published as {expected_candidate_id!r}; refusing a stale or "
            "mis-attributed candidate"
        )

    verified = VerifiedBundle(
        path=str(target),
        file_sha256=observed_file,
        manifest=manifest,
        manifest_digest=observed_manifest,
        candidate_id=str(manifest["candidate_id"]),
        candidate_index=int(manifest["candidate_index"]),
        run_id=str(manifest["run_id"]),
        iteration=int(manifest["iteration"]),
        elapsed_active_training_seconds=float(
            manifest["elapsed_active_training_seconds"]
        ),
        move_ema_model_state_digest=move_digest,
        setup_ema_model_state_digest=setup_digest,
        config_digest=str(manifest.get("config_digest", "")),
        source_digest=str(manifest.get("source_digest", "")),
    )
    _ = MODEL_CONTRACT_VERSION  # named so a contract bump is a visible import
    return verified, payload


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def materialize_move_checkpoint(
    payload: dict, directory: "Path | str", *, verified: VerifiedBundle
) -> Path:
    """Write the candidate's move EMA into a real accepted checkpoint."""
    import torch

    from ...model.checkpoint import save_checkpoint
    from ...model.production_model import build_candidate_model
    from ...training.phase9_behavior import state_dict_digest

    model = build_candidate_model(MOVE_CANDIDATE_ID, device="cpu")
    model.load_state_dict(
        {
            name: torch.as_tensor(value).detach().to("cpu", torch.float32).clone()
            for name, value in payload["move_ema_state"].items()
        },
        strict=True,
    )
    model.eval()
    observed = state_dict_digest(model)
    if observed != verified.move_ema_model_state_digest:
        raise Phase17BundleError(
            f"the loaded move model digests to {observed}, the bundle claims "
            f"{verified.move_ema_model_state_digest}"
        )
    parameters = int(sum(p.numel() for p in model.parameters()))
    if parameters != MOVE_PARAMETER_COUNT:
        raise Phase17BundleError(
            f"the loaded move model has {parameters} parameters, not "
            f"{MOVE_PARAMETER_COUNT}"
        )
    target = Path(directory) / f"{verified.candidate_id}.move_ema.pt"
    save_checkpoint(model, target)
    return target


def build_move_owner(checkpoint: "Path | str", *, name: str):
    """The candidate's move seat owner, on the accepted inference path."""
    from ...evaluation.neural_worker import InferenceOwner
    from ...model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config

    return InferenceOwner(
        checkpoint,
        decision_mode=CANDIDATE_DECISION_MODE,
        device=EVALUATION_DEVICE,
        dtype=EVALUATION_DTYPE,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config(MOVE_CANDIDATE_ID),
        name=name,
    )


def build_setup_model(payload: dict, *, verified: VerifiedBundle):
    """The candidate's setup EMA network, digest-checked, eval mode, CPU."""
    import torch

    from ...training.phase17.setup_model import Phase17SetupModel

    model = Phase17SetupModel()
    model.load_state_dict(
        {
            name: torch.as_tensor(value).detach().to("cpu", torch.float32).clone()
            for name, value in payload["setup_ema_state"].items()
        },
        strict=True,
    )
    model.eval()
    observed = state_mapping_digest(model.state_dict())
    if observed != verified.setup_ema_model_state_digest:
        raise Phase17BundleError(
            f"the loaded setup model digests to {observed}, the bundle claims "
            f"{verified.setup_ema_model_state_digest}"
        )
    parameters = int(sum(p.numel() for p in model.parameters()))
    if parameters != SETUP_PARAMETER_COUNT:
        raise Phase17BundleError(
            f"the loaded setup model has {parameters} parameters, not "
            f"{SETUP_PARAMETER_COUNT}"
        )
    return model


__all__ = [
    "MOVE_CANDIDATE_ID",
    "MOVE_PARAMETER_COUNT",
    "SETUP_PARAMETER_COUNT",
    "VerifiedBundle",
    "build_move_owner",
    "build_setup_model",
    "materialize_move_checkpoint",
    "verify_bundle",
]
