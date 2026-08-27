"""Phase 15 Agent 2 section 16: the frozen engineering candidate.

Specification source: `02_AGENT_2_SEARCH_IMPLEMENTATION.md` sections 15, 16.

Human-readable names *and* the exact digests. The record binds the selected
complete system to bytes, not to labels: a reader can see "P24 + B18, MEDIUM"
and a loader can refuse anything whose sha256 or state digest differs. It
also carries what the selection is *not* — `scientific_validation_status:
not performed` and an explicit limitations list — because section 16 asks for
an engineering deliverable and a later reader must not be able to mistake it
for a strength claim.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .contract import (
    ACCEPTABLE_MOVE_SECONDS,
    BETA_DEFAULT,
    EPSILON_DEFAULT,
    INTEGRATION_VERSION,
    ORACLE_AVAILABLE_IN_PRODUCTION,
    PHASE15_SCORE_DEFINITION,
    PHASE15_SEARCH_VERSION,
    PHASE15_STATUS_MARKERS,
    PRODUCTION_PROVIDERS,
    Phase15SearchError,
    pairing as pairing_of,
    preset as preset_of,
)
from .player import CANDIDATE_ARTIFACT, PLAYER_VERSION

#: Where the frozen candidate lives.
DEFAULT_CANDIDATE_PATH = Path("checkpoints/phase15/phase15_search_candidate_v1.json")


class Phase15CandidateError(Phase15SearchError):
    """The engineering candidate could not be built or read back."""


def build_candidate_record(
    *,
    selected_pairing: str,
    selected_preset: str,
    maximum_strength_preset: str,
    models,
    time_caps: dict,
    latency: dict,
    match_manifest_digest: str,
    position_manifest_digest: str,
    gate: dict,
    stage_a: dict,
    stage_b: dict,
    stage_c: dict,
    system_matrix: dict,
    known_limitations: "list | None" = None,
    environment: "dict | None" = None,
    generated_utc: "str | None" = None,
) -> dict:
    """The `phase15_search_candidate_v1` document."""
    target = pairing_of(selected_pairing)
    if target.kind != "search":
        raise Phase15CandidateError(
            f"{selected_pairing!r} is not a deployable search pairing"
        )
    if target.provider not in PRODUCTION_PROVIDERS:
        raise Phase15CandidateError(  # pragma: no cover - Pairing refuses first
            f"{selected_pairing!r} names a non-production provider"
        )
    config = preset_of(selected_preset)
    maximum = preset_of(maximum_strength_preset)
    move = models.move_models[target.move_model].identity
    belief = models.specialists[target.provider].identity
    handoff = models.handoff

    record = {
        "artifact": CANDIDATE_ARTIFACT,
        **PHASE15_STATUS_MARKERS,
        "generated_utc": generated_utc
        or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "integration_version": INTEGRATION_VERSION,
        "player_version": PLAYER_VERSION,
        "selected_system": {
            "pairing_id": selected_pairing,
            "description": target.description,
            "move_model": target.move_model.upper(),
            "belief_model": target.provider.upper(),
        },
        "move_model": {
            "logical_identity": move["logical_identity"],
            "checkpoint_path": move["checkpoint_path"],
            "checkpoint_sha256": move["checkpoint_sha256"],
            "model_state_digest": move["model_state_digest"],
            "phase14_candidate_hour": move["phase14_candidate_hour"],
            "phase14_archive_sha256": move["phase14_archive_sha256"],
            "global_optimizer_step": move["global_optimizer_step"],
            "role": move["role"],
            "trained_by_phase15": False,
        },
        "belief_model": {
            "provider_id": belief["provider_id"],
            "checkpoint_path": belief["checkpoint_path"],
            "checkpoint_sha256": belief["checkpoint_sha256"],
            "state_digest": belief["state_digest"],
            "architecture_version": belief["architecture_version"],
            "prefix_backbone": belief["prefix_backbone"],
            "prefix_backbone_state_digest": belief["prefix_backbone_state_digest"],
            "role": belief["role"],
            "holds_policy_parameters": belief["holds_policy_parameters"],
            "holds_value_parameters": belief["holds_value_parameters"],
        },
        "belief_calibration": {
            "applied_temperature": belief["applied_temperature"],
            "fitted_temperature": belief["fitted_temperature"],
            "keep_calibrated": belief["keep_calibrated"],
            "source": "the temperature Agent 1 recorded; Phase 15 Agent 2 fitted none",
        },
        "search": {
            "search_version": PHASE15_SEARCH_VERSION,
            "score_definition": PHASE15_SCORE_DEFINITION,
            "selected_preset": config.preset_id,
            "worlds": config.worlds,
            "root_candidates": f"<= {config.max_root_candidates}",
            "rollout_depth": config.rollout_depth,
            "beta": config.beta,
            "epsilon": config.epsilon,
            "policy_regularization": (
                f"S(a) = Q(a) + {config.beta} * log(pi(a) + {config.epsilon})"
            ),
            "value_definition": "V = P(win) - P(loss), exact terminal results override",
            "deduplicate_worlds": config.deduplicate_worlds,
            "verify_world_public_surface": config.verify_world_public_surface,
            "production": True,
        },
        "maximum_strength": {
            "preset_id": maximum.preset_id,
            "worlds": maximum.worlds,
            "root_candidates": f"<= {maximum.max_root_candidates}",
            "rollout_depth": maximum.rollout_depth,
        },
        "latency": latency,
        "time_caps_seconds": dict(time_caps),
        "latency_ceiling_seconds": ACCEPTABLE_MOVE_SECONDS,
        "direct_fallback": {
            "identity": move["logical_identity"],
            "model_state_digest": move["model_state_digest"],
            "rule": (
                "on timeout, search error, non-finite score or an illegal result, "
                "play the same move model's direct legal move; never forfeit"
            ),
        },
        "evidence": {
            "match_manifest_digest": match_manifest_digest,
            "position_manifest_digest": position_manifest_digest,
            "corpus": dict(handoff.get("corpus") or {}),
            "handoff_artifact": handoff.get("artifact"),
            "gate": gate,
            "stage_a_decision_diagnostic": stage_a,
            "stage_b_match_comparison": stage_b,
            "stage_c_budget": stage_c,
            "system_matrix": system_matrix,
        },
        "known_limitations": list(known_limitations or []),
        "oracle_available_in_production": ORACLE_AVAILABLE_IN_PRODUCTION,
        "scientific_validation_status": "not performed",
        "environment": dict(environment or {}),
    }
    return record


def write_candidate(record: dict, path: "Path | str" = DEFAULT_CANDIDATE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def load_candidate(path: "Path | str" = DEFAULT_CANDIDATE_PATH) -> dict:
    """Read the frozen candidate back, with its structural claims re-checked."""
    path = Path(path)
    if not path.is_file():
        raise Phase15CandidateError(f"no Phase 15 candidate at {path}")
    record = json.loads(path.read_text())
    if record.get("artifact") != CANDIDATE_ARTIFACT:
        raise Phase15CandidateError(f"{path} is not a {CANDIDATE_ARTIFACT} document")
    if record.get("oracle_available_in_production") is not False:
        raise Phase15CandidateError(
            f"{path} does not record oracle_available_in_production = false"
        )
    if record.get("scientific_validation_status") != "not performed":
        raise Phase15CandidateError(
            f"{path} misstates its validation status as "
            f"{record.get('scientific_validation_status')!r}"
        )
    return record


def load_player_from_candidate(
    path: "Path | str" = DEFAULT_CANDIDATE_PATH,
    *,
    root: "Path | str" = ".",
    device: str = "cpu",
    mode: str = "selected_search",
):
    """`(player, record)`: the packaged player, rebuilt from the frozen bytes.

    The one production entry point. It loads only what the record names, and
    the loader refuses any file whose sha256 or state digest differs — so this
    can only ever produce the frozen stack.
    """
    from .loaders import load_all
    from .player import MODE_MAX_STRENGTH, MODE_SELECTED, Phase15SearchPlayer
    from .systems import build_engine

    record = load_candidate(path)
    models = load_all(root=root, device=device, with_anchor=False)
    pairing_id = record["selected_system"]["pairing_id"]

    move = models.move_models[pairing_of(pairing_id).move_model].identity
    if move["checkpoint_sha256"] != record["move_model"]["checkpoint_sha256"]:
        raise Phase15CandidateError(
            "the loaded move model does not match the candidate's checkpoint sha256"
        )
    if move["model_state_digest"] != record["move_model"]["model_state_digest"]:
        raise Phase15CandidateError(
            "the loaded move model does not match the candidate's state digest"
        )
    belief = models.specialists[pairing_of(pairing_id).provider].identity
    if belief["checkpoint_sha256"] != record["belief_model"]["checkpoint_sha256"]:
        raise Phase15CandidateError(
            "the loaded belief model does not match the candidate's checkpoint sha256"
        )
    if belief["state_digest"] != record["belief_model"]["state_digest"]:
        raise Phase15CandidateError(
            "the loaded belief model does not match the candidate's state digest"
        )

    systems = {
        MODE_SELECTED: build_engine(
            pairing_id, models, record["search"]["selected_preset"], device=device
        ),
        MODE_MAX_STRENGTH: build_engine(
            pairing_id, models, record["maximum_strength"]["preset_id"], device=device
        ),
    }
    player = Phase15SearchPlayer(
        systems,
        models,
        mode=mode,
        time_caps=dict(record["time_caps_seconds"]),
        device=device,
    )
    return player, record


__all__ = [
    "DEFAULT_CANDIDATE_PATH",
    "Phase15CandidateError",
    "build_candidate_record",
    "load_candidate",
    "load_player_from_candidate",
    "write_candidate",
]
