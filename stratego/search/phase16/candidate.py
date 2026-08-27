"""Phase 16 Agent 2 section 7: the frozen stochastic candidate.

`phase16_stochastic_candidate_v1` binds the selected `(tau, tau_r, top_p)`
to the exact frozen bytes it runs over — the Phase 15 P24/B24 digests and
the applied belief temperature, copied from and re-checked against
`phase15_search_candidate_v1` — plus the budgets, the idle-measured caps,
the Stage 1/2 headline numbers with their pack names, and the stated
limitations. A loader can refuse anything whose digest differs, and a
reader cannot mistake the record for a strength claim.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..phase15.candidate import (
    DEFAULT_CANDIDATE_PATH as PHASE15_CANDIDATE_PATH,
    load_candidate as load_phase15_candidate,
)
from .contract import (
    ACCEPTABLE_MOVE_SECONDS,
    CANDIDATE_ARTIFACT_16,
    PHASE16_STATUS_MARKERS,
    STOCHASTIC_PAIRING,
    STOCHASTIC_VERSION,
    VARIED_MODE_PRESETS,
    VARIED_MODES,
    Phase16StochasticError,
)
from .stochastic import Phase16VariedPlayer, StochasticArm, build_stochastic_bundle

#: Where the frozen Phase 16 candidate lives.
DEFAULT_CANDIDATE_PATH_16 = Path("checkpoints/phase16/phase16_stochastic_candidate_v1.json")


class Phase16CandidateError(Phase16StochasticError):
    """The stochastic candidate could not be built or read back."""


def build_candidate_record_16(
    *,
    arm: StochasticArm,
    time_caps: dict,
    idle_latency: dict,
    stage1: dict,
    stage2: dict,
    selection: dict,
    probe: "dict | None",
    seed_streams: dict,
    known_limitations: list,
    root: "Path | str" = ".",
    environment: "dict | None" = None,
    generated_utc: "str | None" = None,
    deviations: "list | None" = None,
) -> dict:
    """The `phase16_stochastic_candidate_v1` document."""
    phase15 = load_phase15_candidate(Path(root) / PHASE15_CANDIDATE_PATH)
    if phase15["selected_system"]["pairing_id"] != arm.pairing_id:
        raise Phase16CandidateError(
            f"the arm runs {arm.pairing_id!r} but the frozen Phase 15 candidate "
            f"selects {phase15['selected_system']['pairing_id']!r}"
        )
    missing = [mode for mode in VARIED_MODES if mode not in time_caps]
    if missing:
        raise Phase16CandidateError(f"time caps missing for modes {missing}")
    record = {
        "artifact": CANDIDATE_ARTIFACT_16,
        **PHASE16_STATUS_MARKERS,
        "generated_utc": generated_utc
        or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stochastic_version": STOCHASTIC_VERSION,
        "selected_configuration": {
            "arm_id": arm.arm_id,
            "tau": float(arm.tau),
            "tau_r": float(arm.tau_r),
            "top_p": float(arm.top_p),
            "pairing_id": arm.pairing_id,
            "move_sampling": (
                "argmax S (frozen)"
                if arm.tau == 0
                else f"a ~ softmax(S(a)/{arm.tau}) over the engine's candidate set"
            ),
            "rollout_sampling": (
                "greedy (frozen)"
                if arm.tau_r == 0
                else (
                    f"both sides sample the move model's legal distribution at "
                    f"tau_r={arm.tau_r}, nucleus top_p={arm.top_p}"
                )
            ),
            "stochastic_mode_viable": bool(selection.get("stochastic_mode_viable")),
        },
        "modes": {
            mode: {
                "preset_id": VARIED_MODE_PRESETS[mode],
                "time_cap_seconds": float(time_caps[mode]),
            }
            for mode in VARIED_MODES
        },
        "zero_temperature_identity": (
            "tau = 0 and tau_r = 0 replay the frozen phase15_search_candidate_v1 "
            "decisions bit-identically; the regression test in "
            "tests/search/phase16/ pins it"
        ),
        # The bound bytes, copied from and re-checked against the accepted
        # Phase 15 candidate.
        "move_model": dict(phase15["move_model"]),
        "belief_model": dict(phase15["belief_model"]),
        "belief_calibration": dict(phase15["belief_calibration"]),
        "search": dict(phase15["search"]),
        "phase15_candidate": {
            "artifact": phase15["artifact"],
            "path": str(Path(root) / PHASE15_CANDIDATE_PATH),
        },
        "seed_streams": dict(seed_streams),
        "idle_latency": dict(idle_latency),
        "latency_ceiling_seconds": ACCEPTABLE_MOVE_SECONDS,
        "evidence": {
            "stage1": stage1,
            "stage2": stage2,
            "selection": selection,
            "repeat_encounter_probe": probe,
        },
        "known_limitations": list(known_limitations),
        "deviations": list(deviations or []),
        "oracle_available_in_production": False,
        "scientific_validation_status": "not performed",
        "environment": dict(environment or {}),
    }
    return record


def write_candidate_16(record: dict, path: "Path | str" = DEFAULT_CANDIDATE_PATH_16) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def load_candidate_16(path: "Path | str" = DEFAULT_CANDIDATE_PATH_16) -> dict:
    path = Path(path)
    if not path.is_file():
        raise Phase16CandidateError(f"no Phase 16 stochastic candidate at {path}")
    record = json.loads(path.read_text())
    if record.get("artifact") != CANDIDATE_ARTIFACT_16:
        raise Phase16CandidateError(f"{path} is not a {CANDIDATE_ARTIFACT_16} document")
    if record.get("oracle_available_in_production") is not False:
        raise Phase16CandidateError(
            f"{path} does not record oracle_available_in_production = false"
        )
    if record.get("scientific_validation_status") != "not performed":
        raise Phase16CandidateError(
            f"{path} misstates its validation status as "
            f"{record.get('scientific_validation_status')!r}"
        )
    return record


def load_varied_player(
    path: "Path | str" = DEFAULT_CANDIDATE_PATH_16,
    *,
    root: "Path | str" = ".",
    device: str = "cpu",
) -> "tuple[Phase16VariedPlayer, dict]":
    """`(player, record)`: the varied player, rebuilt from the frozen bytes.

    Loads only what the record names; the Phase 15 loader refuses any file
    whose sha256 or state digest differs, and the digests here are re-checked
    against the loaded identities before an engine is built.
    """
    from ..phase15.loaders import load_all

    record = load_candidate_16(path)
    configuration = record["selected_configuration"]
    arm = StochasticArm(
        float(configuration["tau"]),
        float(configuration["tau_r"]),
        top_p=float(configuration["top_p"]),
        pairing_id=configuration["pairing_id"],
    )
    models = load_all(root=root, device=device, with_anchor=False)
    move_model = configuration["pairing_id"].split("_", 1)[0]
    provider = configuration["pairing_id"].split("_", 1)[1]
    move = models.move_models[move_model].identity
    if move["checkpoint_sha256"] != record["move_model"]["checkpoint_sha256"]:
        raise Phase16CandidateError(
            "the loaded move model does not match the candidate's checkpoint sha256"
        )
    if move["model_state_digest"] != record["move_model"]["model_state_digest"]:
        raise Phase16CandidateError(
            "the loaded move model does not match the candidate's state digest"
        )
    belief = models.specialists[provider].identity
    if belief["checkpoint_sha256"] != record["belief_model"]["checkpoint_sha256"]:
        raise Phase16CandidateError(
            "the loaded belief model does not match the candidate's checkpoint sha256"
        )
    if belief["state_digest"] != record["belief_model"]["state_digest"]:
        raise Phase16CandidateError(
            "the loaded belief model does not match the candidate's state digest"
        )
    if float(belief["applied_temperature"]) != float(
        record["belief_calibration"]["applied_temperature"]
    ):
        raise Phase16CandidateError(
            "the applied belief temperature differs from the frozen record"
        )
    bundles = {
        mode: build_stochastic_bundle(
            models, arm, record["modes"][mode]["preset_id"], device=device
        )
        for mode in VARIED_MODES
    }
    time_caps = {
        mode: float(record["modes"][mode]["time_cap_seconds"]) for mode in VARIED_MODES
    }
    player = Phase16VariedPlayer(
        arm, bundles, models, time_caps=time_caps, device=device
    )
    return player, record


__all__ = [
    "DEFAULT_CANDIDATE_PATH_16",
    "Phase16CandidateError",
    "build_candidate_record_16",
    "load_candidate_16",
    "load_varied_player",
    "write_candidate_16",
]
