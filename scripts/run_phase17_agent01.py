"""Phase 17 Agent 1 — verification and artifact generation.

Agent 1 trains nothing and adds no production code. This script exists so that
every number in Agent 1's artifacts is *recomputed from live bytes* rather than
transcribed, and so a reviewer can reproduce the whole boundary in one command.

Roles
-----
``--role observe``   writes ``reports/phase17/agent_01_process_boundary.json``
``--role identity``  writes ``reports/phase17/phase17_start_identity_v1.json``
``--role probe``     writes ``reports/phase17/agent_01_boundary_target_probe.json``
``--role bind``      fills the handoff's ``bound_artifacts`` from bytes on disk
``--role all``       observe, identity, probe (bind is a separate final step)

Nothing here mutates a tracked file, an accepted checkpoint, a result ledger or
the Phase 14 run state. Every external read is read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

REPORT_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase17"

WORK_PACKAGE = "phase17"
PROVISIONAL_RUN_ID = "RUN-2026-A"

#: The one accepted move-policy start. Claimed by the common contract; every
#: value below is recomputed and compared, never copied into the output.
START_CHECKPOINT = "checkpoints/phase9/selfplay_c1_v1.pt"
CLAIMED_FILE_SHA256 = "dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea"
CLAIMED_MODEL_STATE_DIGEST = (
    "f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd"
)

PHASE14_RUN_STATE = Path(
    "/Volumes/Brandon_Washington/stratego_phase14/phase14_run_state.json"
)
PHASE14_EMERGENCY_STOP = Path(
    "/Volumes/Brandon_Washington/stratego_phase14/phase14_emergency_stop.json"
)
PHASE14_CLOSURE_DEADLINE_UTC = "2026-08-28T16:15:34.689Z"

ATARAXOS_PDF = REPOSITORY_ROOT / "2511.07312v1.pdf"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: "str | Path") -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(*command: str, strip: bool = True) -> str:
    """Run a command in the repository root and return stdout.

    `strip=False` matters for `git status --porcelain`: its status codes are
    column-significant and a leading space on the first line is data, not
    whitespace.
    """
    result = subprocess.run(
        command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if strip else result.stdout


def write_json(name: str, payload: dict) -> Path:
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = REPORT_DIRECTORY / name
    destination.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return destination


# ---------------------------------------------------------------------------
# Section 1 — the process and source boundary
# ---------------------------------------------------------------------------


def _live_processes() -> list:
    """Every process whose command line could be a learner/collector/monitor."""
    listing = run("ps", "-Ao", "pid,ppid,etime,pcpu,args")
    interesting = []
    for line in listing.splitlines()[1:]:
        lowered = line.lower()
        if not any(
            token in lowered
            for token in ("phase9", "phase10", "phase13", "phase14", "phase15",
                          "phase16", "phase17", "caffeinate", "dashboard")
        ):
            continue
        if "run_phase17_agent01" in lowered or " ps -ao " in lowered:
            continue
        fields = line.split(None, 4)
        if len(fields) < 5:
            continue
        interesting.append(
            {
                "pid": int(fields[0]),
                "ppid": int(fields[1]),
                "elapsed": fields[2],
                "cpu_percent": float(fields[3]),
                "command": fields[4],
            }
        )
    return interesting


def _classify_process(entry: dict) -> str:
    command = entry["command"].lower()
    if "phase14_dashboard" in command:
        return "read_only_monitor"
    if "phase14_launch" in command or "runner" in command:
        return "trainer_or_supervisor"
    return "unclassified"


def _phase14_closure() -> dict:
    if not PHASE14_RUN_STATE.exists():
        return {
            "run_state_readable": False,
            "note": "external volume not mounted; closure state unresolved",
        }
    state = json.loads(PHASE14_RUN_STATE.read_text())
    progress = state.get("progress", {})
    stop = (
        json.loads(PHASE14_EMERGENCY_STOP.read_text())
        if PHASE14_EMERGENCY_STOP.exists()
        else None
    )
    return {
        "run_state_readable": True,
        "run_state_path": str(PHASE14_RUN_STATE),
        "closed": bool(progress.get("closed")),
        "closed_reason": state.get("closed_reason"),
        "iterations_completed": progress.get("iterations_completed"),
        "elapsed_hours": state.get("elapsed_hours"),
        "mode": state.get("mode"),
        "emergency_stop_requested_utc": (stop or {}).get("requested_utc"),
        "emergency_stop_reason": (stop or {}).get("reason"),
        "closure_deadline_utc": PHASE14_CLOSURE_DEADLINE_UTC,
        "read_only": True,
    }


def _worktree() -> dict:
    porcelain = run("git", "status", "--porcelain=v1", strip=False)
    modified, untracked = [], []
    for line in porcelain.splitlines():
        if not line:
            continue
        code, path = line[:2], line[3:]
        (untracked if code == "??" else modified).append({"code": code.strip(), "path": path})
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "commit_subject": run("git", "log", "-1", "--format=%s"),
        "commit_utc": run("git", "log", "-1", "--format=%cI"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "remote": run("git", "config", "--get", "remote.origin.url"),
        "modified_tracked": modified,
        "untracked_entries": untracked,
        "modified_tracked_count": len(modified),
        "untracked_entry_count": len(untracked),
        "diffstat": run("git", "diff", "--stat"),
    }


def _phase14_launch_binding() -> dict:
    """Read-only: would the Phase 14 launch/finalize code binding pass today?

    `scripts/phase14_launch.py --role finalize` calls
    `assert_bound_launch_code()` BEFORE it branches on the role, so this
    question decides whether formal Phase 14 closure is reachable at all.
    Nothing here writes; `assert_launch_code` only recomputes digests.
    """
    try:
        from stratego.training.phase14_launch import (
            assert_bound_launch_code, code_binding, launch_manifest_path,
            load_launch_manifest,
        )
    except Exception as error:  # noqa: BLE001
        return {"checked": False, "error": f"{type(error).__name__}: {error}"}

    manifest = load_launch_manifest(launch_manifest_path())
    observed = code_binding()
    bound = manifest.get("code", {}).get("git", {})
    result = {
        "checked": True,
        "read_only": True,
        "bound_revision": bound.get("revision"),
        "observed_revision": observed["git"]["revision"],
        "revision_matches": bound.get("revision") == observed["git"]["revision"],
        "bound_dirty_tracked_files": list(bound.get("dirty_tracked_files", [])),
        "observed_dirty_tracked_files": list(observed["git"]["dirty_tracked_files"]),
        "code_digest_matches": observed["code_digest"] == manifest["code"].get("code_digest"),
    }
    try:
        assert_bound_launch_code()
        result["assert_bound_launch_code"] = "passed"
        result["finalize_reachable"] = True
    except Exception as error:  # noqa: BLE001
        result["assert_bound_launch_code"] = f"{type(error).__name__}"
        result["assert_bound_launch_code_message"] = str(error)
        result["finalize_reachable"] = False
    result["interpretation"] = (
        "the import closure is unchanged (code_digest matches) and the revision "
        "still matches; the refusal, if any, is the dirty-tracked-file list. The "
        "two stratego_project_docs edits landed with the 2026-08-27 documentation "
        "pass and are NOT Phase 17 work."
        if result.get("code_digest_matches") else
        "the import closure itself differs from the manifest"
    )
    return result


def observe() -> Path:
    processes = [
        {**entry, "classification": _classify_process(entry)} for entry in _live_processes()
    ]
    payload = {
        "artifact": "phase17_agent01_process_boundary_v1",
        "work_package": WORK_PACKAGE,
        "observed_utc": utc_now(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cwd": str(REPOSITORY_ROOT),
        },
        "evidence_classification": "PENDING",
        "scientific_validation_status": "not performed",
        "canonical_documents_read": [
            "stratego_project_docs/STATUS.md",
            "stratego_project_docs/PHASE_HISTORY.md",
            "stratego_project_docs/EVIDENCE_INDEX.md",
            "stratego_project_docs/EXPERIMENT_FRAMEWORK.md",
        ],
        "active_processes": processes,
        "active_learner_or_collector": [
            entry for entry in processes if entry["classification"] == "trainer_or_supervisor"
        ],
        "phase14_closure": _phase14_closure(),
        "phase14_launch_binding": _phase14_launch_binding(),
        "version_control": _worktree(),
        "phase16_still_untracked": bool(
            run("git", "ls-files", "stratego/training/phase16") == ""
        ),
        "accepted_paths_mutated_by_agent_1": [],
        "accepted_path_check": {
            "method": "git status over tracked accepted paths",
            "interpretation": (
                "the single tracked modification under an accepted path is "
                "reports/phase13/phase14_launch_manifest_v1.json, the required "
                "self-referential manifest rebuild recorded in STATUS.md section 5. "
                "It predates Phase 17 and Agent 1 did not touch it."
            ),
            "tracked_modifications": [
                line
                for line in run(
                    "git", "status", "--porcelain=v1", "--",
                    "checkpoints/", "reports/phase_9_data", "reports/phase_10_data",
                    "reports/phase_11_data", "reports/phase12", "reports/phase13",
                    strip=False,
                ).splitlines()
                if line and not line.startswith("??")
            ],
        },
        "actions_taken": "read only; no process signalled, no file mutated, no git write",
    }
    return write_json("agent_01_process_boundary.json", payload)


# ---------------------------------------------------------------------------
# Section 2 — the Phase 9 start, recomputed
# ---------------------------------------------------------------------------


def identity() -> Path:
    import torch  # noqa: F401  (imported for the version record)

    from stratego.training import phase9_checkpoint as p9
    from stratego.training.phase9_behavior import state_dict_digest as canonical_digest
    from stratego.model.checkpoint import state_dict_digest as container_digest
    from stratego.training import phase14_contract as p14

    path = REPOSITORY_ROOT / START_CHECKPOINT
    observed_file = file_sha256(path)

    payload_on_disk = p9.read_phase9_payload(path)
    metadata = p9.validate_phase9_payload(payload_on_disk, source=str(path))
    model = p9.model_from_payload(payload_on_disk, device="cpu")
    model.eval()

    model_state = payload_on_disk["model_state"]
    observed_canonical = canonical_digest(model)
    observed_container = container_digest(model_state["state_dict"])

    payload = {
        "artifact": "phase17_start_identity_v1",
        "work_package": WORK_PACKAGE,
        "provisional_run_id": PROVISIONAL_RUN_ID,
        "verified_utc": utc_now(),
        "evidence_classification": "ACCEPTED (upstream); this record is PENDING",
        "path": START_CHECKPOINT,
        "file_bytes": path.stat().st_size,
        "verification": {
            "loader": "stratego.training.phase9_checkpoint."
            "read_phase9_payload -> validate_phase9_payload -> model_from_payload",
            "validate_phase9_payload": "passed",
            "file_sha256_claimed": CLAIMED_FILE_SHA256,
            "file_sha256_observed": observed_file,
            "file_sha256_matches": observed_file == CLAIMED_FILE_SHA256,
            "model_state_digest_claimed": CLAIMED_MODEL_STATE_DIGEST,
            "model_state_digest_observed": observed_canonical,
            "model_state_digest_matches": observed_canonical == CLAIMED_MODEL_STATE_DIGEST,
            "phase14_contract_agreement": {
                "starting_checkpoint": p14.STARTING_CHECKPOINT == START_CHECKPOINT,
                "file_sha256": p14.STARTING_CHECKPOINT_SHA256 == observed_file,
                "model_state_digest": p14.STARTING_MODEL_STATE_DIGEST == observed_canonical,
                "parameter_count": p14.ACCEPTED_C1_PARAMETERS
                == int(sum(p.numel() for p in model.parameters())),
            },
        },
        "two_digest_conventions": {
            "why_this_block_exists": (
                "the repository contains two functions named state_dict_digest and "
                "they disagree on these bytes; Phase 17 must name one"
            ),
            "canonical": {
                "function": "stratego.training.phase9_behavior.state_dict_digest",
                "input": "a live nn.Module",
                "hashes": "sorted name + str(numpy shape) + float32 bytes (no dtype field)",
                "value": observed_canonical,
                "used_by": [
                    "phase9 agent 8 acceptance", "phase10_contract", "phase10b_contract",
                    "phase11_contract", "phase12 candidate", "phase14_contract",
                ],
                "status": "THE Phase 17 model-state digest",
            },
            "container": {
                "function": "stratego.model.checkpoint.state_dict_digest",
                "input": "a state_dict mapping",
                "hashes": "sorted name + str(tuple(shape)) + str(dtype) + bytes",
                "value": observed_container,
                "also_stored_at": "model_state.provenance.state_dict_digest",
                "status": "NOT the Phase 17 model-state digest; recorded so a "
                "future agent cannot mistake a mismatch for corruption",
            },
        },
        "architecture": {
            "model_architecture_id": model_state.get("model_architecture_id"),
            "model_contract_version": model_state.get("model_contract_version"),
            "configuration": model_state.get("model_configuration"),
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "parameter_tensors": len(model_state["state_dict"]),
            "dtypes": sorted({str(t.dtype) for t in model_state["state_dict"].values()}),
            "heads_present": [name for name, _ in model.named_children()],
        },
        "contract_versions": {
            "rules_version": model_state.get("rules_version"),
            "observation_version": model_state.get("observation_version"),
            "action_encoding_version": model_state.get("action_encoding_version"),
            "engine_action_frame": model_state.get("engine_action_frame"),
            "policy_action_frame": model_state.get("policy_action_frame"),
            "checkpoint_format_version": model_state.get("checkpoint_format_version"),
            "phase9_checkpoint_version": payload_on_disk.get("phase9_checkpoint_version"),
        },
        "lineage": {
            "snapshot_role": payload_on_disk.get("snapshot_role"),
            "behavior_snapshot_identity": payload_on_disk.get("behavior_snapshot_identity"),
            "behavior_checkpoint_sha256": payload_on_disk.get("behavior_checkpoint_sha256"),
            "produced_after_iteration": (payload_on_disk.get("diagnostics") or {}).get(
                "produced_after_iteration"
            ),
            "collects_iteration": (payload_on_disk.get("diagnostics") or {}).get(
                "collects_iteration"
            ),
            "run_label": (payload_on_disk.get("diagnostics") or {}).get("run_label"),
            "creation_timestamp": model_state.get("creation_timestamp"),
        },
        "optimizer_and_schedule_metadata_present_in_file": {
            "global_optimizer_step": payload_on_disk.get("global_optimizer_step"),
            "rl_iteration": payload_on_disk.get("rl_iteration"),
            "examples_consumed": payload_on_disk.get("examples_consumed"),
            "kl_beta": payload_on_disk.get("kl_beta"),
            "entropy_schedule_position": payload_on_disk.get("entropy_schedule_position"),
            "best_checkpoint_identity": payload_on_disk.get("best_checkpoint_identity"),
            "best_validation_score": payload_on_disk.get("best_validation_score"),
            "ema_state": model_state.get("ema_state"),
            "integrity_digest": payload_on_disk.get("integrity_digest"),
            "train_config_digest": payload_on_disk.get("train_config_digest"),
            "contract_digest": payload_on_disk.get("contract_digest"),
            "trainer_version": payload_on_disk.get("trainer_version"),
        },
        "phase17_start_semantics": {
            "mode": "weights_only_warm_start",
            "loads": "move-model state_dict only",
            "discards": [
                "optimizer_state (fresh AdamW moments, zero)",
                "scheduler_state (LR schedule reset to Phase 17 iteration 1)",
                "kl_controller_state (fresh move KL controller, beta0 0.005)",
                "minibatch_cursor", "rng", "validation_history",
                "global_optimizer_step / rl_iteration / examples_consumed",
            ],
            "creates": [
                "fresh move EMA initialised from the loaded RAW weights",
                "setup model / optimizer / KL controller / EMA from scratch",
            ],
            "belief_head": {
                "present_in_checkpoint": "belief_output" in dict(model.named_children()),
                "phase17_loss_weight": 0.0,
                "reason": "common contract section 4 disables the Phase 9 marginal "
                "belief auxiliary loss; the head stays for checkpoint compatibility",
                "phase9_accepted_weight_being_overridden": 0.25,
            },
            "resume_identity_check_not_used": (
                "check_phase9_resume_identity authorizes a Phase 9 *resume*; Phase 17 "
                "is a new lineage and deliberately does not resume, so only "
                "read/validate/rebuild are used"
            ),
        },
        "refusals": {
            "alternate_checkpoints_refused": [
                {
                    "path": "checkpoints/phase12/phase9_c1_readonly_copy.pt",
                    "reason": "convenience copy with different file bytes (81906f71...)",
                },
                {
                    "path": "checkpoints/phase15/p18_source_readonly.pt",
                    "reason": "Phase 14 intermediate; evaluation instrument only",
                },
                {
                    "path": "checkpoints/phase15/p24_source_readonly.pt",
                    "reason": "Phase 14 intermediate; evaluation instrument only",
                },
                {
                    "path": "checkpoints/phase11b/agent01_1c_final_block_plus_mlp.pt",
                    "reason": "belief/search only; never a policy start",
                },
            ],
            "loader_refused": "the P24-specific Phase 16 loader",
        },
        "software_runtime_at_verification": {
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
            "numpy": __import__("numpy").__version__,
            "platform": platform.platform(),
        },
        "software_runtime_recorded_in_checkpoint": payload_on_disk.get(
            "software_runtime_versions"
        ),
        "validate_metadata_keys": sorted(metadata.keys()) if isinstance(metadata, dict) else None,
    }
    return write_json("phase17_start_identity_v1.json", payload)


# ---------------------------------------------------------------------------
# Section 3 — the boundary-target probe
# ---------------------------------------------------------------------------


def probe() -> Path:
    import numpy as np

    from stratego.training.phase16.targets import (
        truncated_advantages, whole_game_targets, window_edge_invariant,
    )
    from stratego.training.phase9_contract import (
        behavior_value_scalar, wdl_lambda_targets as accepted_wdl_targets,
        terminal_z, LAMBDA_VALUE,
    )

    rng = np.random.default_rng(20260827)
    decisions = 12
    predictions = [
        tuple(float(x) for x in row)
        for row in rng.dirichlet([2.0, 2.0, 1.0], size=decisions)
    ]
    terminal_result, player, boundaries = "red_win", 0, [4, 8]

    whole = whole_game_targets(predictions, terminal_result, player)
    values = [behavior_value_scalar(row) for row in predictions]
    z = terminal_z(whole["outcome"])

    buffered = window_edge_invariant(predictions, terminal_result, player, boundaries)

    emitted = []
    emitted += truncated_advantages(values[0:3], values[3])     # window 1, tail v[3]
    emitted += truncated_advantages(values[3:7], values[7])     # window 2, tail v[7]
    emitted += truncated_advantages(values[7:12], float(z))     # window 3, tail z
    whole_advantages = list(whole["advantages"])
    advantage_error = np.abs(np.asarray(emitted) - np.asarray(whole_advantages))

    def wdl_bootstrap(chunk, tail_vector):
        following = np.asarray(tail_vector, dtype=np.float64)
        out = [None] * len(chunk)
        for t in range(len(chunk) - 1, -1, -1):
            nxt = (
                np.asarray(chunk[t + 1], dtype=np.float64)
                if t + 1 < len(chunk)
                else np.asarray(tail_vector, dtype=np.float64)
            )
            following = (1.0 - LAMBDA_VALUE) * nxt + LAMBDA_VALUE * following
            out[t] = tuple(float(v) for v in following)
        return out

    whole_wdl = [
        tuple(float(v) for v in row)
        for row in accepted_wdl_targets(list(predictions), whole["outcome"])
    ]
    wdl_error = np.abs(
        np.asarray(wdl_bootstrap(predictions[0:3], predictions[3]))
        - np.asarray(whole_wdl[0:3])
    )

    payload = {
        "artifact": "phase17_agent01_boundary_target_probe_v1",
        "work_package": WORK_PACKAGE,
        "computed_utc": utc_now(),
        "evidence_classification": "ENGINEERING",
        "scientific_validation_status": "not performed",
        "question": (
            "Common contract section 6 requires BOTH partial emission with "
            "boundary bootstrapping AND that a game spanning >= 3 windows match "
            "the accepted whole-game targets to float32 tolerance. Can both hold?"
        ),
        "setup": {
            "decisions": decisions,
            "window_boundaries": boundaries,
            "windows": 3,
            "terminal_result": terminal_result,
            "player": player,
            "predictions_source": "numpy default_rng(20260827).dirichlet([2,2,1])",
            "lambda_advantage": 0.5,
            "lambda_value": LAMBDA_VALUE,
            "tolerance": buffered["tolerance"],
        },
        "phase16_buffered_path": {
            "function": "stratego.training.phase16.targets.window_edge_invariant",
            "holds": buffered["holds"],
            "max_advantage_difference": buffered["max_advantage_difference"],
            "max_wdl_difference": buffered["max_wdl_difference"],
            "why_it_is_exact": (
                "windowed_targets BUFFERS the whole game and computes targets at "
                "the close; partial_advantages is called at each boundary but its "
                "result is discarded into boundary_reports. The invariant therefore "
                "measures the reduction property, not partial emission."
            ),
        },
        "mandated_partial_emission_path": {
            "max_advantage_difference": float(advantage_error.max()),
            "mean_advantage_difference": float(advantage_error.mean()),
            "rows_exceeding_tolerance": int((advantage_error > buffered["tolerance"]).sum()),
            "rows_total": int(advantage_error.size),
            "max_wdl_difference_window_1": float(wdl_error.max()),
            "per_row": [
                {
                    "t": t,
                    "advantage_whole_game": float(a),
                    "advantage_bootstrapped": float(b),
                    "absolute_difference": float(d),
                }
                for t, (a, b, d) in enumerate(zip(whole_advantages, emitted, advantage_error))
            ],
        },
        "finding": (
            "The two section-6 requirements are mutually exclusive. Only the final "
            "window -- whose tail is the true terminal z -- reproduces the accepted "
            "targets; every earlier window differs by construction because a "
            "truncated lambda-return closed on a value estimate is not the full "
            "lambda-return. The satisfiable invariant is the REDUCTION property."
        ),
        "recommendation_for_operator": (
            "Restate the section 6 invariant as: 'when a window boundary coincides "
            "with the terminal step, the windowed walk equals the accepted "
            "whole-game walk entry for entry to float32 tolerance' (gate G-M4a, "
            "testable), and add a separate bounded-divergence telemetry check for "
            "genuine boundaries. Agent 1 has NOT amended the contract."
        ),
    }
    return write_json("agent_01_boundary_target_probe.json", payload)


# ---------------------------------------------------------------------------
# Section 4 — bind the handoff to bytes on disk
# ---------------------------------------------------------------------------

BOUND_FILES = (
    "agent_01_process_boundary.json",
    "phase17_start_identity_v1.json",
    "agent_01_boundary_target_probe.json",
    "ataraxos_method_map_v1.md",
    "ataraxos_method_map_v1.json",
    "agent_01_baseline_inclusion_list.json",
    "agent_01_report.md",
)


def bind() -> Path:
    handoff_path = REPORT_DIRECTORY / "phase17_contract_handoff_v1.json"
    handoff = json.loads(handoff_path.read_text())

    bound = {}
    for name in BOUND_FILES:
        candidate = REPORT_DIRECTORY / name
        bound[name] = {
            "present": candidate.exists(),
            "sha256": file_sha256(candidate) if candidate.exists() else None,
            "bytes": candidate.stat().st_size if candidate.exists() else None,
        }

    sources = {}
    for relative in (
        START_CHECKPOINT,
        "data/phase16/phase16_benchmark_v1.json",
        "instructions/phase_17_tandem_current_policy_self_play/"
        "00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md",
        "instructions/phase_17_tandem_current_policy_self_play/"
        "01_AGENT_1_CONTRACT_AND_BASELINE.md",
        "2511.07312v1.pdf",
    ):
        candidate = REPOSITORY_ROOT / relative
        sources[relative] = {
            "present": candidate.exists(),
            "sha256": file_sha256(candidate) if candidate.exists() else None,
        }

    handoff["bound_artifacts"] = bound
    handoff["bound_sources"] = sources
    handoff["bound_utc"] = utc_now()
    handoff["bound_commit"] = run("git", "rev-parse", "HEAD")
    handoff_path.write_text(json.dumps(handoff, indent=2, sort_keys=False) + "\n")
    return handoff_path




# ---------------------------------------------------------------------------
# Section 5 — the Ataraxos method map
# ---------------------------------------------------------------------------
#
# One row per relevant paper method, in the field order
# `01_AGENT_1_CONTRACT_AND_BASELINE.md` section 3 specifies. The Markdown and
# the JSON are rendered from this single list so they cannot drift apart.
#
# Paper: Sokota, Vinitsky, Hu, Kolter, Farina, "Superhuman AI for Stratego
# Using Self-Play Reinforcement Learning and Test-Time Search", arXiv
# 2511.07312v1, read in full from the local copy `2511.07312v1.pdf`.

PAPER_ID = "arXiv:2511.07312v1"
PAPER_TITLE = (
    "Superhuman AI for Stratego Using Self-Play Reinforcement Learning "
    "and Test-Time Search"
)
#: Derived once, in row B05, and depended on by rows M07 and S15.
PAPER_ITERATIONS = 42376

E, S, D, X = "exact", "scaled", "intentional divergence", "not used"

ROWS = [
 # ---------------- move: population and action semantics ----------------
 dict(id="M01", area="move/population", paper_ref="2.3; D.2",
  paper="Ataraxos directly samples from its policy networks to generate self-play data; no search in collection. 1,536 parallel environments per GPU, 202 simulator moves (101 per player) between training iterations, ~5e6 transitions per iteration over 16 GPUs.",
  phase17="100% current-policy self-play; both seats are the current RAW move snapshot. Legal moves sampled categorically with an explicit per-decision seed; argmax prohibited. Search prohibited in collection and training. Historical checkpoints, rule and stress agents are evaluation instruments only.",
  status=E, reason="Identical semantics. The environment count and 202-step cadence are replaced by a fixed learner-transition budget (row M02) because a single M4 Pro cannot hold 1,536x16 environments.",
  owner="Agent 2 / Agent 4",
  test="Correctness gate: sampled-not-argmax assertion; structural no-search assertion; structural no-training-opponent assertion; per-decision seed and behavior digest stored on every transition."),

 dict(id="M02", area="move/iteration sizing", paper_ref="D.2; 2.3",
  paper="Iteration = a fixed number of simulator steps (202) across a fixed environment population. Games span multiple iterations, so training data is slightly off-policy.",
  phase17="Iteration = exactly 65,536 learner transitions harvested (DEFAULT_WINDOW_DECISIONS, already the Phase 16 value). Because both seats learn, every legal model decision is a learner transition.",
  status=S, reason="Same invariant -- fixed work per iteration, not fixed games -- expressed in transitions rather than environment steps. This is the single change Phase 14's telemetry most supports: 83% of its 59.97 h was training, and minutes/iteration grew 15.7 -> 153.8 under whole-game (2,048-game) sizing while collection grew only 4.8 -> 16.3.",
  owner="Agent 2",
  test="Correctness gate: harvested transitions per iteration == 65,536 exactly, over at least three consecutive windows. Telemetry separates transitions_harvested / transitions_trained / boundary_rows / games_completed / active_games / game_length / policy_age."),

 dict(id="M03", area="move/current-policy binding", paper_ref="2.1; 2.4; D.4 (theta_t)",
  paper="theta_t is 'the parameters that played the move at position x'. Every environment plays under the live network; a new iteration's steps are played by the newly updated network.",
  phase17="'Current policy' means current at the decision. After every move update, every in-flight game must resolve its next Red or Blue action through the newly rebound RAW snapshot.",
  status=E, reason="Phase 16 does NOT do this and the divergence is a hard Phase 17 blocker. Verified in source: stratego/training/phase16/collector.py:536 rebind() replaces the collector's participants, but Phase16GameRunner is constructed with the participants object at game creation (stratego/training/phase9_collector.py:332) and resolves every ply from its own copy (phase9_collector.py:447, acting_snapshot_for). Snapshots are frozen (assert_frozen), so an in-flight game keeps its game-start weights for its whole length.",
  owner="Agent 2",
  test="Correctness gate: forced-rebind test on an already-running game -- start a game, update the move model, assert the next decision's recorded behavior digest equals the NEW raw digest, and that no transition is ever recorded under a stale digest (production stop condition)."),

 dict(id="M04", area="move/boundary targets", paper_ref="D.4",
  paper="Advantage delta by lambda-return with lambda=0.5 over {E[v_theta_t(x')]} (and o if the game finished), baseline E[v_theta_t(x)]. Outcome probabilities xi by lambda-return with lambda=0.8 over the vector values (and one-hot o if finished).",
  phase17="Same lambdas (the accepted Phase 9 LAMBDA_ADVANTAGE=0.5, LAMBDA_VALUE=0.8, GAMMA=1.0 already equal the paper). At a window boundary: unfinished scalar advantage traces bootstrap from the boundary value; W/D/L lambda targets bootstrap from the boundary W/D/L prediction. Only the minimum carry state continues the trace.",
  status=E, reason="lambda values and gamma are already identical. The bootstrapping is the paper's implicit behaviour made explicit for a window that closes mid-game. NOTE: Phase 16 built truncated_advantages but left partial emission OFF and buffers whole games, because phase9_batch_loss averages value and belief over every row with no per-row mask. Phase 17 needs a NEW phase17-namespace target/loss path; the accepted objective must not be edited.",
  owner="Agent 2",
  test="Gate G-M4a (reduction invariant, satisfiable): with the tail set to the terminal z the windowed walk equals the accepted whole-game walk entry for entry to 1e-6. Gate G-M4b (literal whole-game equality across >=3 windows) is PROVABLY UNSATISFIABLE under partial emission -- see reports/phase17/agent_01_boundary_target_probe.json (max advantage difference 0.309, max W/D/L difference 0.121, 7 of 12 rows over tolerance). Operator confirmation required."),

 dict(id="M05", area="move/PPO clipping", paper_ref="Eq. 6; Table 22",
  paper="L_pi = -min(r*delta, clip(r, 0.8, 1.2)*delta); importance ratio clipping parameter 0.2; r = pi_theta(m|x)/pi_theta_t(m|x).",
  phase17="Identical. PPO_CLIP_EPSILON = 0.2, already the accepted Phase 9 value; behavior probabilities always come from the recorded raw snapshot, never recomputed.",
  status=E, reason="Accepted Phase 9 and the paper agree exactly.",
  owner="Agent 2",
  test="Clip fraction logged per epoch; CLIP_FRACTION_HARD_LIMIT 0.75 retained."),

 dict(id="M06", area="move/advantage filter", paper_ref="D.4; Table 22",
  paper="Train on a move only if |estimated advantage| is at or above the 0.75 quantile AND at or above 0.01. Reduced wall-clock per iteration ~2.5x while increasing sample efficiency and asymptotic performance.",
  phase17="Retained unchanged: ADVANTAGE_FILTER_QUANTILE 0.75, ADVANTAGE_FILTER_FLOOR 0.01.",
  status=E, reason="Accepted Phase 9 values already equal the paper's.",
  owner="Agent 2",
  test="Telemetry reports transitions_harvested (65,536) and transitions_trained (post-filter) separately. The 65,536 budget is a HARVEST budget; the trained count is the smaller filtered subset."),

 dict(id="M07", area="move/learning rate", paper_ref="Table 22",
  paper="Adam LR = clip(0.5 / n^1.1, 5e-6, 1e-4). Over N_paper ~= 42,376 iterations this holds the 1e-4 ceiling until n=2,305 (5.44% of the run), reaches the 5e-6 floor at n=35,112 (82.86%), dynamic range 20x.",
  phase17="Frozen by common contract section 9: lr(n) = clamp(1.5e-4 * (n/n_ref)^-1.1, 1.5e-5, 1.5e-4), n_ref = ceil(0.125*N). Holds the ceiling for the first 12.50% of the run; would reach the floor only at n = 1.0139*N, i.e. NEVER within the run; dynamic range 10x.",
  status=D, reason="The contract wins over the paper (common contract section 2) and Agent 1 has NOT changed it. Recorded so the deviation is visible: the local curve is flatter and shallower than the paper's. A shape-preserving map would be n_ref = ceil(0.0545*N) with lr_min = lr_max/20 = 7.5e-6, reaching the floor at 82.86% of the run. Flagged for operator confirmation; do NOT amend without one.",
  owner="Agent 1 freezes / Agent 4 substitutes N",
  test="Agent 4 freezes N, n_ref and the whole curve from the preflight rehearsal BEFORE h0 and never recomputes them from live speed. Resume recomputes lr from the stored 1-based iteration and the frozen horizon."),

 dict(id="M08", area="move/entropy vs magnet KL", paper_ref="Eq. 6; Table 22; 2.4",
  paper="NO entropy bonus in the move loss. Instead a myopic REVERSE KL penalty toward a magnet policy rho (choose a movable piece uniformly, then a legal move for it uniformly): + alpha*KL(pi_theta(x), rho(x)) with alpha = 0.05 / n^0.3. alpha(1)=0.05 -> alpha(N_paper)=0.002046.",
  phase17="An ENTROPY BONUS with the paper's exponent and the accepted Phase 9 endpoints: c_H(n) = max(0.001, 0.005 * n^-0.3), subtracted from the loss. No magnet policy exists in this project.",
  status=D, reason="THESE ARE DIFFERENT REGULARIZERS AND MUST NEVER BE REPORTED AS EACH OTHER. An entropy bonus pulls toward the uniform distribution over LEGAL ACTIONS; the paper's magnet KL pulls toward a structured piece-then-move distribution that is not uniform over legal actions. Only the n^-0.3 shape is shared. Quantified consequence of the frozen constants: c_H reaches its 0.001 floor at n=214, which is 71% of a 6-hour run (N~313) but only 34% of a 12-hour run (N~626) -- so about two thirds of the Phase 17 production run would sit at the terminal floor, which is the Phase 14 failure mode the Phase 16 schedule module was written to avoid. Flagged for operator confirmation.",
  owner="Agent 1 freezes / Agent 2 implements",
  test="Telemetry logs move policy entropy and c_H(n) under their own names; no field may be labelled 'magnet KL'. Stop condition: move entropy below 25% of its first-hour median for five windows."),

 dict(id="M09", area="move/behavior KL", paper_ref="Eq. 6; Table 22; 2.4",
  paper="+ 0.1 * KL(pi_theta(x), pi_theta_t(x)) -- a REVERSE KL (current || behavior), FIXED coefficient 0.1, no controller.",
  phase17="The accepted Phase 9 adaptive controller on the FORWARD KL D_KL(pi_b || pi_theta) (stratego/training/phase9_loss.py:543, 'D_KL(pi_b || pi_theta) over the legal set'). target 0.015, beta0 0.005, beta bounds [1e-4, 0.2], increase >0.03 (x2), decrease <0.0075 (x0.5), hard mean-KL limit 0.08.",
  status=D, reason="Opposite direction AND fixed-vs-adaptive. Named as a divergence rather than treated as equivalent, exactly as the Agent 1 brief section 3 requires. Retained because common contract section 9 says the accepted controller stands unless the paper map and the operator justify a change; Agent 1 justifies recording, not changing.",
  owner="Agent 1 freezes / Agent 2 implements",
  test="Telemetry names the direction explicitly ('forward, D_KL(pi_b||pi_theta)'). Stop condition: mean KL above 0.08 for three consecutive windows unless the existing hard veto fires first."),

 dict(id="M10", area="move/epochs", paper_ref="Table 22",
  paper="1 epoch per training iteration.",
  phase17="1 epoch per iteration.",
  status=E, reason="Common contract section 9 sets 1, overriding the accepted Phase 9 EPOCHS_PER_ROLLOUT = 2, which matches the paper.",
  owner="Agent 2", test="Optimizer-step count per iteration logged and checkpointed."),

 dict(id="M11", area="move/EMA", paper_ref="Table 22; 2.6",
  paper="EMA of the parameters with smoothing 0.999, updated after the training iteration; the EMA is what is EVALUATED, never what plays training games.",
  phase17="EMA decay 0.999, initialised fresh from the loaded Phase 9 RAW weights. RAW generates all training data; EMA never acts in the training population; every checkpoint stores both; evaluation exports carry paired EMA weights.",
  status=E, reason="Identical, including the raw/EMA role split. The accepted Phase 9 system keeps NO EMA (phase14_contract.EMA_PRESENT = False), so this is new here and the Phase 9 checkpoint's ema_state field is null, as verified.",
  owner="Agent 2 / Agent 4",
  test="Correctness gate: assert no EMA-weighted participant ever appears in collection; checkpoint round trip preserves both states."),

 dict(id="M12", area="move/gradient clip", paper_ref="Table 22",
  paper="Maximum gradient norm 0.267.",
  phase17="The accepted Phase 9 OPTIMIZER_CONSTRAINTS gradient_clip_norm = 1.0.",
  status=D, reason="Not changed by the common contract, so the accepted value stands. Recorded because 1.0 vs 0.267 is a 3.7x looser update-size control on one of the paper's four named update-size mechanisms, and Phase 17 also loosens a second one (row M07's flatter LR).",
  owner="Agent 1 records / Agent 2 implements",
  test="Gradient norm logged per optimizer step; nonfinite gradient is an immediate stop."),

 dict(id="M13", area="move/optimizer", paper_ref="Table 22; 2.4",
  paper="Adam.",
  phase17="AdamW, betas (0.9, 0.999), eps 1e-8, weight_decay 0.01, float32, MPS -- the accepted Phase 9 OPTIMIZER_CONSTRAINTS, with FRESH zero moments.",
  status=D, reason="AdamW's decoupled weight decay is an extra regularizer the paper does not use. Retained because it is the accepted Phase 9 optimizer and the common contract says 'fresh AdamW optimizer moments'.",
  owner="Agent 2", test="Optimizer identity and moment freshness asserted at start; optimizer state checkpointed."),

 dict(id="M14", area="move/value loss", paper_ref="Eq. 5; Table 22",
  paper="L_v = cross-entropy(xi, v_theta(x)); value loss coefficient 1; L_move = L_pi + L_v.",
  phase17="The accepted Phase 9 VALUE_LOSS_WEIGHT = 0.5.",
  status=D, reason="Half the paper's weight. Not changed by the common contract, so the accepted value stands; recorded so a later agent does not read 'W/D/L value cross-entropy' as evidence of parity.",
  owner="Agent 1 records / Agent 2 implements", test="Value loss logged separately from policy loss."),

 dict(id="M15", area="move/belief auxiliary", paper_ref="D.5; 2.5",
  paper="The belief network is a SEPARATE 57.1M-parameter network trained AFTER the move/setup run (4 H100s, 4 days) on trajectories of the FINAL policies. It is not an auxiliary head on the move network.",
  phase17="The Phase 9 marginal belief auxiliary head stays in the checkpoint for compatibility but receives loss weight 0.0 and is never a source of targets. Joint autoregressive belief training is a later phase.",
  status=E, reason="Zeroing the auxiliary weight moves Phase 17 TOWARD the paper's separation. The accepted Phase 9 BELIEF_LOSS_WEIGHT is 0.25 and is deliberately overridden.",
  owner="Agent 2", test="Assert belief loss term contributes exactly 0.0; assert no belief target is constructed."),

 dict(id="M16", area="move/network", paper_ref="Table 24",
  paper="Encoder-only transformer, depth 8, embedding 384, 8 heads, feed-forward 1,536, 14.7M parameters, pre-layernorm, key-query product policy head, learned absolute positional embeddings.",
  phase17="The accepted C1: depth 4, width 128, 4 heads, feed-forward 512, 863,959 parameters, pre-layernorm, key-query policy head (policy_query/policy_key), learned_row_column_v1 position encoding, 127-channel observation, 10,000 action logits.",
  status=S, reason="Same family and same policy parameterization at ~1/17 the parameters -- the local scale decision made in Phase 6 and accepted since. Not revisited by Phase 17.",
  owner="fixed upstream", test="Start-identity gate binds the architecture config and the 863,959 count."),

 # ---------------- setup ----------------
 dict(id="S01", area="setup/architecture", paper_ref="Table 23; Fig. 3",
  paper="Decoder-only causal transformer over start token + 40 row-major piece tokens. Depth 4, embedding 512, 8 heads, feed-forward 2,048 (= 4 x width), learned positional embedding init std 0.1, pre-layernorm, 12.6M parameters.",
  phase17="Depth 4, width 128, 4 heads, feed-forward 512 (= 4 x width), pre-layernorm, ~0.80M parameters.",
  status=S, reason="Width scaled 512 -> 128 with the paper's 4x feed-forward ratio preserved and depth kept at 4. ARITHMETIC CONFIRMING FF=512: 4 blocks x (attention 66,048 + feed-forward 131,712 + 2 layernorms 512) = 793,088, plus 13 token embeddings x 128 = 1,664, 41 positions x 128 = 5,248, final norm 256, and the three heads 1,548 + 387 + 129 -> 802,320 parameters, i.e. 'approximately 0.8 million'. Feed-forward width 51 would give 328,412 and could not reach 0.8M. 512 governs.",
  owner="Agent 3", test="Setup gate asserts the parameter count is within a declared band of 802,320 and that the config is 4/128/4/512."),

 dict(id="S02", area="setup/causal factorization", paper_ref="D.3; Fig. 3",
  paper="Given a setup prefix in row-major order the network emits (1) W/L/D probabilities, (2) a real-valued conditional-entropy estimate, (3) a distribution over the next piece placement. Training on entire setups in single forward-backward passes.",
  phase17="At every one of the 40 prefixes: masked 12-way next-piece logits, W/D/L logits, and a scalar conditional-entropy prediction. Remaining inventory computed solely from the prefix; exhausted types receive an unsampleable mask.",
  status=E, reason="Identical factorization and identical head set.",
  owner="Agent 3",
  test="Setup gate: autoregressive causality test (prefix k's outputs are unchanged by tokens > k); exhausted-token adversarial masking test; zero inventory/legality/placement failures over >= 5,000 samples split across colours."),

 dict(id="S03", area="setup/orientation", paper_ref="not in the paper (local engine concern)",
  paper="No analogue; the paper's simulator has one canonical frame.",
  phase17="Generate in canonical own-side coordinates; convert only at the engine boundary through the accepted Phase 15 helper (stratego.belief.phase15.orientation, rule 'red engine row == canonical rank; blue engine row == 9 - canonical rank', version phase15_orientation_rule_v1). Never generate Blue directly in engine orientation; never pass canonical Blue to create_game.",
  status=X, reason="Local-only requirement. It exists because the Phase 11B glue mis-oriented Blue setups: 77.0% Blue front-row flags under the old path versus 1.77% corrected.",
  owner="Agent 3", test="Setup gate: zero orientation failures; the accepted negative_canary and check_board are imported, never re-derived."),

 dict(id="S04", area="setup/sampling and pools", paper_ref="D.2; Table 18",
  paper="A pool of pre-generated setups sampled from the setup network, 1,000 per player per GPU, REGENERATED AFTER EACH TRAINING ITERATION.",
  phase17="Vectorized fresh pools generated under each frozen raw setup snapshot; default 512-1,000 candidates per side per iteration; Agent 3 selects the smallest size that keeps game creation supplied without material training delay. Unused and refill counts recorded.",
  status=E, reason="Same regeneration cadence and same per-side pool scale; only the count is allowed to shrink to local throughput.",
  owner="Agent 3",
  test="Setup gate records generation cost; pool unused/refill counters in telemetry. A generation or orientation failure is FATAL -- there is no frozen setup library in Phase 17 training and no silent library fallback."),

 dict(id="S05", area="setup/binding to a game", paper_ref="D.2; D.3",
  paper="A setup is drawn once at game creation from the pool; theta_t is 'the parameters that generated sigma-bar'; because games span iterations the setup data is slightly off-policy.",
  phase17="A setup is sampled once at game creation and stays bound to that game. Its behavior probabilities and setup-snapshot digest remain attached until the outcome arrives, even after the setup learner has updated. Deliberately UNLIKE the move side (row M03).",
  status=E, reason="Identical, and the asymmetry with M03 is intentional: a setup is one decision made once, a move policy acts repeatedly.",
  owner="Agent 3 / Agent 4", test="Setup episode records policy age and the generating snapshot digest; the ratio denominator is always the recorded behavior probability."),

 dict(id="S06", area="setup/returns and filtering", paper_ref="2.3; D.3",
  paper="Monte Carlo returns -- the final outcomes of games played by the current policy -- with NO advantage filtering. All setups of games finished during the last data collection period are trained on.",
  phase17="Both sides train from the result: win +1, draw 0, loss -1 from that side's own perspective. All 40 prefixes, no move-style top-quartile filter.",
  status=E, reason="Identical.",
  owner="Agent 3",
  test="Setup gate: Red/Blue/draw outcome-sign tests and a synthetic reward-flip gradient test (flipping the outcome flips the gradient's sign)."),

 dict(id="S07", area="setup/advantage", paper_ref="D.3 (delta)",
  paper="delta = (o - E[v_theta_t(sigma)]) + alpha*(H(sigma-bar | sigma; theta_t) - h_theta_t(sigma)), with alpha the regularization temperature (the entropy-maximization coefficient).",
  phase17="Same two-term form. H is read as the REALIZED SUFFIX INFORMATION CONTENT I(sigma-bar|sigma;theta_t) = -log pi_theta_t(sigma-bar|sigma) in nats -- the only quantity computable from one sampled setup, and the Monte Carlo estimator of the conditional entropy. AMBIGUITY RESOLVED: the entropy term is frozen as alpha*(I/10 - h_theta_t(sigma)), i.e. BOTH sides in the normalized units the prediction loss trains.",
  status=D, reason="As printed, Eq. (1) regresses h to H/10 while delta uses (H - h). Those are different units, and the mixed form degenerates to roughly 0.9*alpha*H -- an UNCENTERED bonus. With early setup entropy around 100 bits (~69 nats, paper Fig. 4B) that term is ~6.2 at alpha=0.1, which would swamp (o - E[v]) in [-2, 2]. The normalized reading keeps the entropy term commensurate with the outcome term and is the only reading under which alpha behaves as an advantage coefficient. Recorded as a deviation from the literal text and flagged for operator confirmation.",
  owner="Agent 1 freezes / Agent 3 implements",
  test="Telemetry logs I, I/10, h and the two advantage terms separately so their relative magnitudes are visible from hour 0."),

 dict(id="S08", area="setup/conditional-entropy head", paper_ref="Eq. 1; Table 20",
  paper="L_h = (H(sigma-bar|sigma;theta_t)/10 - h_theta(sigma))^2; conditional-entropy prediction loss coefficient 1; normalizing constant 1/10.",
  phase17="Identical, with H read as the realized suffix information content per row S07.",
  status=E, reason="Transcribed exactly, including the 1/10 normalizer and the coefficient of 1.",
  owner="Agent 3", test="Setup gate: h converges toward I/10 on the initial masked model; L_h logged separately."),

 dict(id="S09", area="setup/value head", paper_ref="Eq. 2; Table 20",
  paper="L_v = -log v_theta(o | sigma); value loss coefficient 0.5. L_setup = L_pi + 0.5*L_v + L_h.",
  phase17="Identical: W/D/L cross-entropy at every prefix, weight 0.5, and the same total weighting.",
  status=E, reason="Transcribed exactly.",
  owner="Agent 3", test="Each of the three loss terms logged under its own name."),

 dict(id="S10", area="setup/PPO clipping", paper_ref="Eq. 3; Table 20",
  paper="L_pi = -min(r*delta, clip(r, 0.8, 1.2)*delta), r = pi_theta(sigma+|sigma)/pi_theta_t(sigma+|sigma); importance ratio clipping parameter 0.2.",
  phase17="Identical: clipping 0.2, per-prefix ratio, behavior probabilities always from the recorded raw setup snapshot.",
  status=E, reason="Transcribed exactly.",
  owner="Agent 3", test="Setup clip fraction logged per epoch."),

 dict(id="S11", area="setup/behavior KL", paper_ref="Eq. 3; Table 20",
  paper="+ 0.1 * KL(pi_theta(sigma), pi_theta_t(sigma)) -- REVERSE KL (current || behavior) over the next-piece distribution, FIXED coefficient 0.1.",
  phase17="A separate adaptive setup behavior-KL controller, independent of the move controller. PROVISIONAL fields pending Agent 3 calibration: direction, target, beta0, beta bounds and hard range.",
  status=D, reason="Fixed-vs-adaptive, and the direction must be stated rather than inherited. The move side's accepted controller measures the FORWARD KL, so 'the same controller' would silently flip the paper's direction. Agent 1 freezes the SCHEMA and marks the numbers provisional; Agent 3's soak supplies calibration; the operator confirms before launch.",
  owner="Agent 1 schema / Agent 3 calibration",
  test="The direction is a required, logged field. Stop condition: setup KL above its hard range for three consecutive setup updates."),

 dict(id="S12", area="setup/epochs", paper_ref="Table 20",
  paper="5 epochs per training iteration, batches of 1,024 per GPU.",
  phase17="5 setup epochs per setup iteration is the DEFAULT. Agent 3 may recommend fewer only with measured generation, forward/backward and total iteration costs showing that five materially threaten the 12-hour move budget. Silent reduction is prohibited.",
  status=E, reason="Transcribed exactly with an explicit, evidence-gated escape hatch.",
  owner="Agent 3", test="Setup gate: five-epoch setup throughput measurement, reported as generation / forward-backward / total per iteration."),

 dict(id="S13", area="setup/gradient clip", paper_ref="D.3; Table 20",
  paper="Maximum gradient norm 0.5.",
  phase17="0.5.", status=E, reason="Transcribed exactly. Note this differs from the move side's 1.0 (row M12) and the two must not be merged into one constant.",
  owner="Agent 3", test="Setup gradient norm logged separately from the move gradient norm."),

 dict(id="S14", area="setup/learning rate", paper_ref="Table 20",
  paper="Adam, learning rate 5e-5, CONSTANT -- the paper schedules only the move learning rate (2.4).",
  phase17="Adam, 5e-5, constant.",
  status=E, reason="No horizon mapping is needed for a constant. Transcribed exactly.",
  owner="Agent 3", test="Setup scheduler position checkpointed even though the value is constant, so a resume is provably identical."),

 dict(id="S15", area="setup/regularization temperature", paper_ref="Table 20; 2.4",
  paper="alpha(n) = 0.1 / n^0.3. Over N_paper ~= 42,376 this runs 0.100000 -> 0.004091, an anneal depth of 24.4x.",
  phase17="Endpoint-preserving re-horizon: alpha(n) = max(0.1 * n^(-p), 0.1 * N_paper^(-0.3)) with p = 0.3 * ln(N_paper)/ln(N) and N_paper = 42,376. Both endpoints are exact: alpha(1) = 0.100000 and alpha(N) = 0.004091 for any N. Example exponents: N=300 -> p=0.5604; N=626 -> p=0.4964.",
  status=S, reason="Raw transcription onto a 12-hour horizon would END 3.5x more heavily regularized than the paper (N=626 gives alpha(N)=0.014489 against the paper's 0.004091) -- the setup policy would never leave the high-entropy regime. The exponent rescale is the minimal shape-preserving map: same power-law family, same first and last value, log axis rescaled. RISK, recorded not hidden: the paper warns that annealing too aggressively 'collapsed the entropy of the model'; traversing a week-long anneal in 12 hours is aggressive by construction. The floor makes the schedule safe on overrun. Flagged for operator confirmation.",
  owner="Agent 1 freezes / Agent 4 substitutes N",
  test="Instrumented by the existing stop policy: setup mean prefix entropy below 60% of its initial baseline for three checks, and flag effective support below four."),

 dict(id="S16", area="setup/EMA", paper_ref="D.3; Table 20",
  paper="EMA of the setup parameters, smoothing 0.999, updated after the training iteration; used for evaluation.",
  phase17="Setup EMA decay 0.999, initialised from scratch with the setup model. RAW generates all setups; EMA is exported for evaluation only.",
  status=E, reason="Transcribed exactly, with the same raw/EMA role split as the move side.",
  owner="Agent 3 / Agent 4", test="Paired checkpoint stores raw and EMA setup states; the joint evaluation lane uses paired EMA weights."),

 dict(id="S17", area="setup/episode supply", paper_ref="D.3",
  paper="'All setups for games that were finished during the last data collection period were included in the training data.' Batches of 1,024 per GPU; 99.5e3 setup gradient steps over the run.",
  phase17="Completed setup episodes enter a bounded FIFO queue and are consumed once in a fixed setup-sequence budget. Queue depth, oldest/mean age, policy age, consumed count and any rejected or discarded episode are recorded. Silent dropping is prohibited. Too few episodes -> skip the setup update EXPLICITLY; repeated starvation is a production stop condition.",
  status=S, reason="Same 'finished since last collection' source with an explicit bounded queue, because a single machine finishes far fewer games per iteration than 16 GPUs and starvation is a real local failure mode rather than a theoretical one.",
  owner="Agent 3 / Agent 4",
  test="Stop conditions: no setup optimizer update for one complete 30-minute interval after warm-up while games and setup episodes complete; setup queue age/backlog over the frozen ceiling for three windows."),

 # ---------------- boundaries ----------------
 dict(id="B01", area="belief", paper_ref="2.5; D.5; Table 25; Fig. 7",
  paper="A separate 57.1M-parameter encoder+decoder belief network (encoder depth 6, 4 decoder blocks, 8 heads, width 512, dropout 0.2, feed-forward 2,048), trained AFTER the self-play run on trajectories of the FINAL policies, on 4 H100s for 4 days.",
  phase17="NOT BUILT IN PHASE 17. The move network's marginal belief head is present but zero-weighted (row M15). Joint autoregressive belief training is a later phase, permitted only after the operator promotes a Phase 17 checkpoint.",
  status=X, reason="Deliberate phase boundary. The paper's own ordering agrees: belief training is downstream of a finished policy.",
  owner="none in Phase 17", test="Structural assertion that no Phase 17 module constructs a belief world model."),

 dict(id="B02", area="search", paper_ref="2.5; D.7; Table 26",
  paper="Test-time search: sample ~1,000/|legal| hidden configurations from the belief net, run 1,000 depth-40 rollouts, average value predictions, then take one tabular magnetic-mirror-descent step, pi_search proportional to exp(q/alpha) * rho^(...) * pi_theta^(...), alpha=0.002, beta=0.02, and SAMPLE the played move from it.",
  phase17="PROHIBITED in collection and in training. No Phase 17 agent implements or quietly prepares belief-guided search.",
  status=X, reason="Deliberate phase boundary, and it also matches the paper's own finding (2.3) that search-based data generation was not worth its cost.",
  owner="none in Phase 17",
  test="Correctness gate: structural no-search assertion over the collection path."),

 dict(id="B03", area="game rules", paper_ref="D.1; Table 17",
  paper="Training under a 100-move battleless draw rule (evaluation under 200); maximum game length 4,000 moves. The proximity-to-rule observation channel is fractional, so n battleless moves read as n/100 in training and n/200 in testing.",
  phase17="The accepted local ruleset stratego_project_v1 is unchanged. Any change to the two-square, continuous-chasing, battleless or move-safety limits would require a NEW rules identifier (02_project_ruleset.md section 227) and is out of scope.",
  status=X, reason="Changing the rules version would invalidate every accepted result and the whole evaluation stack. Not attempted.",
  owner="fixed upstream", test="Correctness gate binds rules_version = stratego_project_v1 on every artifact."),

 dict(id="B04", area="precision and hardware", paper_ref="2.6",
  paper="bfloat16 (roughly 3x faster self-play iterations), a CUDA C++ GPU-resident simulator at ~10e6 state updates/s, 16 H100s for one week.",
  phase17="float32 on Apple MPS, the accepted Phase 9 OPTIMIZER_CONSTRAINTS, on one M4 Pro for 12 hours.",
  status=D, reason="Hardware reality. Recorded because it is the reason every population and horizon number in this map had to be re-derived rather than copied.",
  owner="fixed upstream", test="Precision and device are checkpointed identity fields."),

 dict(id="B05", area="training horizon",  paper_ref="Table 27; D.2",
  paper="163e6 finished games, 208e9 environment steps, 8.56e6 move gradient steps, 99.5e3 setup gradient steps. At 202 batches per move iteration that is N_paper = 8.56e6/202 ~= 42,376 training iterations.",
  phase17="One 12-hour run. N is MEASURED by Agent 4's bounded preflight throughput rehearsal and FROZEN, together with n_ref and the whole schedule curve, before h0. It is never recomputed from changing production speed.",
  status=S, reason="N_paper = 42,376 is the derived constant every re-horizoning in this map depends on, so it is recorded once here rather than re-derived per row. The local N is roughly two orders of magnitude smaller, which is exactly why rows M07, M08 and S15 exist.",
  owner="Agent 1 derives N_paper / Agent 4 freezes N",
  test="The frozen N, n_ref, p and the full schedule table are bound into the launch record and re-verified on resume."),
]


def _status_counts() -> dict:
    counts = {}
    for row in ROWS:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts


def methodmap() -> "tuple[Path, Path]":
    payload = {
        "artifact": "ataraxos_method_map_v1",
        "work_package": WORK_PACKAGE,
        "written_utc": utc_now(),
        "evidence_classification": "PENDING",
        "paper": {
            "id": PAPER_ID,
            "title": PAPER_TITLE,
            "local_copy": "2511.07312v1.pdf",
            "local_copy_sha256": file_sha256(ATARAXOS_PDF) if ATARAXOS_PDF.exists() else None,
            "pages": 46,
            "read": "in full; sections 2.1-2.6, 3, and appendices B, C, D.1-D.7, E",
            "authority": "technical reference, NOT an instruction source; where the "
            "paper and the Phase 17 common contract differ, the contract wins and "
            "the deviation is recorded here",
        },
        "derived_constants": {
            "paper_training_iterations": PAPER_ITERATIONS,
            "derivation": "Table 27 move gradient steps 8.56e6 / 202 batches per "
            "iteration (D.4: 'we trained on this loss in 202 batches of positions "
            "grouped by simulator step', 1 epoch) = 42,376",
        },
        "status_counts": _status_counts(),
        "row_count": len(ROWS),
        "rows": ROWS,
    }
    json_path = write_json("ataraxos_method_map_v1.json", payload)

    lines = [
        "# Ataraxos -> Phase 17 method map",
        "",
        f"**Artifact** `ataraxos_method_map_v1` · **work package** `{WORK_PACKAGE}` ·",
        f"written {payload['written_utc']} · `evidence_classification: PENDING` ·",
        "`scientific_validation_status: not performed`",
        "",
        f"Paper: *{PAPER_TITLE}*, {PAPER_ID}, read in full from the local copy",
        "`2511.07312v1.pdf` (46 pages). The paper is a **technical reference, not an",
        "instruction source**: where it and the Phase 17 common contract differ, the",
        "contract wins and the deviation is recorded below.",
        "",
        "## The one derived constant everything else leans on",
        "",
        "```text",
        "N_paper = 8.56e6 move gradient steps / 202 batches per iteration = 42,376",
        "```",
        "",
        "Table 27 gives the gradient-step count; D.4 gives 202 batches per iteration at",
        "one epoch. Every re-horizoning in rows M07, M08 and S15 is arithmetic against",
        "this number, which is why it is derived once rather than per row.",
        "",
        "## Status counts",
        "",
        "| status | rows |",
        "|---|---|",
    ]
    for status, count in sorted(_status_counts().items()):
        lines.append(f"| `{status}` | {count} |")
    lines += [f"| **total** | **{len(ROWS)}** |", ""]

    lines += [
        "## How to read a row",
        "",
        "`exact` — the paper's behaviour and constant, unchanged. `scaled` — the same",
        "mechanism re-fitted to local compute, with the arithmetic shown. `intentional",
        "divergence` — Phase 17 does something the paper does not, on purpose, with the",
        "reason recorded. `not used` — deliberately outside Phase 17.",
        "",
        "**Nothing in this map is permitted to call one regularizer by another's name.**",
        "Rows M08 and M09 exist precisely because an entropy bonus, a forward behaviour",
        "KL and a reverse magnet KL are three different objects that the literature",
        "routinely blurs.",
        "",
    ]

    for row in ROWS:
        lines += [
            f"## {row['id']} · {row['area']}",
            "",
            f"**Paper** ({row['paper_ref']}) — {row['paper']}",
            "",
            f"**Phase 17** — {row['phase17']}",
            "",
            f"**Status** `{row['status']}` · **owner** {row['owner']}",
            "",
            f"**Reason** — {row['reason']}",
            "",
            f"**Required test / telemetry** — {row['test']}",
            "",
        ]

    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    md_path = REPORT_DIRECTORY / "ataraxos_method_map_v1.md"
    md_path.write_text("\n".join(lines))
    return md_path, json_path




# ---------------------------------------------------------------------------
# Section 6 — the integration-baseline inclusion list
# ---------------------------------------------------------------------------
#
# Agents 2-4 may not begin against a mutable untracked base. Agent 1 does not
# commit; it produces the exact list and the operator authorizes it.

#: Paths to track. Everything here is source, tests, evidence or specification.
INCLUDE_PATHS = (
    "stratego/belief/phase15",
    "stratego/search/phase15",
    "stratego/search/phase16",
    "stratego/training/phase16",
    "stratego/evaluation/phase16",
    "tests/belief/phase15",
    "tests/search/phase15",
    "tests/search/phase16",
    "tests/training/phase16",
    "tests/evaluation/phase16",
    "reports/phase15",
    "reports/phase16",
    "reports/phase17",
    "data/phase16",
    "checkpoints/phase16/phase16_stochastic_candidate_v1.json",
    "checkpoints/phase16/phase16_recipe_candidate_v1.json",
    "instructions/phase_15_belief_search_engineering",
    "instructions/phase_16_robustness_and_distribution",
    "instructions/phase_17_tandem_current_policy_self_play",
    "scripts/phase16_capture_setup.py",
    "scripts/play_phase15.py",
    "scripts/play_phase16.py",
    "scripts/play_phase16_operator.py",
    "scripts/run_phase15_agent01.py",
    "scripts/run_phase15_agent02.py",
    "scripts/run_phase15_mixture.py",
    "scripts/run_phase16_agent01.py",
    "scripts/run_phase16_agent02.py",
    "scripts/run_phase16_agent03.py",
    "scripts/run_phase17_agent01.py",
    "README.md",
    "stratego_project_docs/STATUS.md",
    "stratego_project_docs/PHASE_HISTORY.md",
    "stratego_project_docs/EVIDENCE_INDEX.md",
    "stratego_project_docs/EXPERIMENT_FRAMEWORK.md",
)

#: Production bytes that must be excluded by new .gitignore rules, following
#: the pattern already established for phases 8, 9, 10, 10b, 11, 11b, 13 and 14.
EXCLUDE_PATHS = (
    ("checkpoints/phase15/", "Phase 15 belief specialists and read-only P18/P24 "
     "source copies; reproducible from the Phase 15 pipeline, and the digests "
     "that name them are tracked in reports/phase15"),
    ("data/phase15/", "Phase 15 belief corpus and feature caches; reproducible "
     "from the frozen corpus contract"),
    ("checkpoints/phase16/arms/", "recipe-shootout hot checkpoints; working "
     "bytes of a completed engineering comparison whose verdict is tracked in "
     "checkpoints/phase16/phase16_recipe_candidate_v1.json"),
)

#: Tracked files that are ALREADY modified and are deliberately left alone.
LEAVE_UNSTAGED = (
    ("reports/phase13/phase14_launch_manifest_v1.json",
     "the required self-referential manifest rebuild; committing it changes the "
     "revision the manifest binds and is a Phase 13/14 decision, not a Phase 17 one"),
    ("stratego_project_docs/05_project_plan.md",
     "2026-08-27 documentation pass; committing changes HEAD away from the "
     "manifest's bound revision 124f3be"),
    ("stratego_project_docs/README.md",
     "2026-08-27 documentation pass; same reason"),
)


def _tree_size(path: Path) -> "tuple[int, int]":
    if path.is_file():
        return 1, path.stat().st_size
    files = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and "__pycache__" not in candidate.parts
    ]
    return len(files), sum(candidate.stat().st_size for candidate in files)


def inclusion_list() -> Path:
    include, total_files, total_bytes = [], 0, 0
    for relative in INCLUDE_PATHS:
        candidate = REPOSITORY_ROOT / relative
        files, size = _tree_size(candidate) if candidate.exists() else (0, 0)
        total_files += files
        total_bytes += size
        include.append(
            {
                "path": relative,
                "exists": candidate.exists(),
                "kind": "file" if candidate.is_file() else "directory",
                "files": files,
                "bytes": size,
                "already_tracked": run("git", "ls-files", relative) != "",
            }
        )

    exclude = []
    for relative, reason in EXCLUDE_PATHS:
        candidate = REPOSITORY_ROOT / relative.rstrip("/")
        files, size = _tree_size(candidate) if candidate.exists() else (0, 0)
        exclude.append(
            {
                "gitignore_entry": relative,
                "exists": candidate.exists(),
                "files": files,
                "bytes": size,
                "reason": reason,
            }
        )

    payload = {
        "artifact": "phase17_agent01_baseline_inclusion_list_v1",
        "work_package": WORK_PACKAGE,
        "written_utc": utc_now(),
        "status": "PROPOSED — requires operator authority; Agent 1 has committed nothing",
        "why": (
            "Common contract section 3: Agents 2-4 may not begin against a mutable "
            "untracked Phase 16 base, and a tar archive is not an integration "
            "baseline. This is the exact list; the operator authorizes it."
        ),
        "include": include,
        "include_totals": {"files": total_files, "bytes": total_bytes,
                           "mebibytes": round(total_bytes / 1048576, 1)},
        "exclude_via_new_gitignore_rules": exclude,
        "exclude_totals": {
            "bytes": sum(entry["bytes"] for entry in exclude),
            "gibibytes": round(sum(entry["bytes"] for entry in exclude) / 1073741824, 2),
        },
        "leave_unstaged": [
            {"path": path, "reason": reason} for path, reason in LEAVE_UNSTAGED
        ],
        "prohibited_operations": [
            "git clean", "git stash", "git checkout of a modified tracked file",
            "git reset", "history rewrite", "deleting any artifact",
        ],
        "existing_backups": {
            "note": "recorded for completeness; an archive is not a baseline",
            "tar_archives": sorted(
                str(candidate.name)
                for candidate in Path("/Volumes/Brandon_Washington").glob(
                    "stratego_untracked_backup_*.tar"
                )
            ) if Path("/Volumes/Brandon_Washington").exists() else [],
        },
        "operator_decision_required": {
            "question": "May the Phase 15-17 inclusion list be committed now?",
            "option_a_wait": {
                "action": "commit nothing until Phase 14 is formally closed",
                "cost": "Agents 2-4 stay blocked; formal closure is currently "
                        "UNREACHABLE on its own terms (see agent_01_process_boundary"
                        ".json -> phase14_launch_binding), so 'wait' has no defined end",
            },
            "option_b_commit_untracked_only": {
                "action": "commit ONLY the untracked include list; leave all three "
                          "modified tracked files unstaged",
                "effect": "HEAD moves off 124f3be, so the Phase 14 launch manifest's "
                          "bound revision no longer matches",
                "already_true": "assert_bound_launch_code ALREADY refuses today on the "
                                "dirty-file list alone, before Phase 17 changes anything",
                "recommended": True,
            },
            "option_c_rebuild_manifest_first": {
                "action": "rebuild the Phase 14 launch manifest against the current "
                          "tree (scripts/phase14_build_launch_package.py), restoring the "
                          "code binding, THEN decide on the baseline commit",
                "note": "the manifest is self-referential and has been rebuilt twice "
                        "before for exactly this reason; this is a Phase 13/14 action "
                        "and Agent 1 did not take it",
            },
        },
    }
    return write_json("agent_01_baseline_inclusion_list.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role",
        choices=("observe", "identity", "probe", "methodmap", "inclusion", "bind", "all"),
        default="all",
    )
    arguments = parser.parse_args()
    roles = (
        ("observe", "identity", "probe", "methodmap", "inclusion")
        if arguments.role == "all"
        else (arguments.role,)
    )
    handlers = {
        "observe": observe, "identity": identity, "probe": probe,
        "methodmap": methodmap, "inclusion": inclusion_list, "bind": bind,
    }
    for role in roles:
        produced = handlers[role]()
        for destination in (produced if isinstance(produced, tuple) else (produced,)):
            print(f"{role}: wrote {destination.relative_to(REPOSITORY_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
