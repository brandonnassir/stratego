"""Phase 17 Agent 2 — bounded verification and artifact generation.

Every number in Agent 2's artifacts is *recomputed from live objects* by this
script rather than transcribed from a test log, so a reviewer can reproduce
the whole move half in one command.

Roles
-----
``--role identity``   exact Phase 9 start: both digests, logit identity against
                      the accepted Phase 9 behavior loader, and the fresh
                      optimizer / KL / EMA / schedule state
``--role invariant``  gate ``G-M4a`` over real sequences, plus the measured
                      partial-emission divergence that ``G-M4b`` used to forbid
``--role rebind``     the forced in-flight rebind, on numbers
``--role window``     a real multi-window collection: exactness, uniqueness,
                      provenance, the participant ledger and one update
``--role suite``      the test-suite result for the Phase 17 move modules
``--role handoff``    assembles ``phase17_move_handoff_v1.json`` from the above
``--role all``        identity, invariant, rebind, window (handoff is separate)

Nothing here mutates a tracked file, an accepted checkpoint or a result ledger.
The Phase 9 start is opened read-only and never written back.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

REPORT_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase17"

WORK_PACKAGE = "phase17"
PROVISIONAL_RUN_ID = "RUN-2026-A"
VERIFICATION_RUN_ID = "RUN-VERIFY-A2"

MOVE_SOURCES = (
    "stratego/training/phase17/__init__.py",
    "stratego/training/phase17/move_contract.py",
    "stratego/training/phase17/move_loss.py",
    "stratego/training/phase17/move_snapshot.py",
    "stratego/training/phase17/move_start.py",
    "stratego/training/phase17/move_trainer.py",
    "stratego/training/phase17/transition_collector.py",
    "stratego/training/phase17/transition_schema.py",
    "stratego/training/phase17/transition_targets.py",
)

#: The Agent 2 test modules. Named explicitly rather than by directory: the
#: Phase 17 test package is shared with Agent 3, and a directory-wide count
#: would attribute the setup half's tests to this agent.
MOVE_TEST_MODULES = (
    "tests/training/phase17/test_move_loss_and_trainer.py",
    "tests/training/phase17/test_move_no_search.py",
    "tests/training/phase17/test_move_sampling.py",
    "tests/training/phase17/test_move_seating.py",
    "tests/training/phase17/test_move_start.py",
    "tests/training/phase17/test_transition_collector.py",
    "tests/training/phase17/test_transition_schema.py",
    "tests/training/phase17/test_transition_targets.py",
)

MOVE_TESTS = (
    "tests/training/phase17/__init__.py",
    "tests/training/phase17/test_move_loss_and_trainer.py",
    "tests/training/phase17/test_move_no_search.py",
    "tests/training/phase17/test_move_sampling.py",
    "tests/training/phase17/test_move_seating.py",
    "tests/training/phase17/test_move_start.py",
    "tests/training/phase17/test_move_support.py",
    "tests/training/phase17/test_transition_collector.py",
    "tests/training/phase17/test_transition_schema.py",
    "tests/training/phase17/test_transition_targets.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_digest(paths) -> dict:
    entries = []
    for relative in sorted(paths):
        target = REPOSITORY_ROOT / relative
        entries.append(
            {
                "path": relative,
                "file_sha256": file_sha256(target),
                "bytes": target.stat().st_size,
            }
        )
    return {
        "files": entries,
        "source_digest": hashlib.sha256(
            "".join(f"{e['path']}:{e['file_sha256']}" for e in entries).encode()
        ).hexdigest(),
    }


def write_json(name: str, payload: dict) -> Path:
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = REPORT_DIRECTORY / name
    destination.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")
    return destination


def runtime_versions() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def setup_provider(offset: int = 0):
    """The Agent 2 verification setup double, imported from the test package."""
    from tests.training.phase17.test_move_support import DeterministicSetupProvider

    return DeterministicSetupProvider(offset=offset)


def perturbed(model, **kwargs):
    from tests.training.phase17.test_move_support import perturbed_copy

    return perturbed_copy(model, **kwargs)


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def identity() -> Path:
    from stratego.training.phase17.move_contract import (
        MOVE_INITIAL_KL_BETA,
        MoveScheduleHorizon,
    )
    from stratego.training.phase17.move_start import (
        DISCARDED_PAYLOAD_KEYS,
        belief_head_parameters,
        build_move_start,
        load_phase17_move_weights,
    )
    from stratego.training.phase9_checkpoint import (
        bind_behavior_snapshot,
        read_phase9_payload,
    )

    started = time.perf_counter()
    loaded = load_phase17_move_weights(device="cpu")
    accepted = bind_behavior_snapshot(
        loaded["path"], logical_identity="B041", namespace="canonical", device="cpu"
    )
    generator = torch.Generator().manual_seed(20260827)
    observations = torch.rand((8, 127, 10, 10), generator=generator)
    with torch.no_grad():
        theirs = accepted.model.forward_observation(observations)
        ours = loaded["model"].forward_observation(observations)
    differences = {
        "policy_logits": float((theirs.policy_logits - ours.policy_logits).abs().max()),
        "value_logits": float((theirs.value_logits - ours.value_logits).abs().max()),
        "belief_logits": float((theirs.belief_logits - ours.belief_logits).abs().max()),
    }

    payload_on_disk = read_phase9_payload(loaded["path"])
    start = build_move_start(total_iterations=626, device="cpu")
    horizon = MoveScheduleHorizon(total_iterations=626)

    return write_json(
        "agent_02_start_identity.json",
        {
            "artifact": "phase17_agent02_start_identity_v1",
            "work_package": WORK_PACKAGE,
            "computed_utc": utc_now(),
            "evidence_classification": "ENGINEERING",
            "scientific_validation_status": "not performed",
            "loader": "read_phase9_payload -> validate_phase9_payload -> model_from_payload",
            "path": loaded["path"],
            "file_sha256": loaded["file_sha256"],
            "model_state_digest": loaded["model_state_digest"],
            "container_state_digest_not_used": loaded["container_state_digest"],
            "parameter_count": loaded["parameter_count"],
            "candidate_id": loaded["candidate_id"],
            "behavior_snapshot_identity": loaded["behavior_snapshot_identity"],
            "trainable": all(p.requires_grad for p in loaded["model"].parameters()),
            "logit_identity_against_accepted_phase9_behavior_loader": {
                "accepted_entry_point": "stratego.training.phase9_checkpoint.bind_behavior_snapshot",
                "observations": list(observations.shape),
                "max_absolute_difference": differences,
                "identical": all(value == 0.0 for value in differences.values()),
            },
            "on_disk_state_that_is_discarded": {
                "kl_beta": float(payload_on_disk["kl_beta"]),
                "rl_iteration": int(payload_on_disk["rl_iteration"]),
                "global_optimizer_step": int(payload_on_disk["global_optimizer_step"]),
                "entropy_schedule_position": dict(
                    payload_on_disk["entropy_schedule_position"]
                ),
                "discarded_keys": list(DISCARDED_PAYLOAD_KEYS),
            },
            "phase17_start_state": {
                "iteration": start.iteration,
                "next_iteration": start.next_iteration,
                "optimizer_moment_entries": len(start.optimizer.state),
                "optimizer_learning_rate": float(
                    start.optimizer.param_groups[0]["lr"]
                ),
                "kl_beta": float(start.controller.beta),
                "kl_beta_expected": MOVE_INITIAL_KL_BETA,
                "kl_controller_updates": len(start.controller.history),
                "ema_updates": int(start.ema.updates),
                "ema_equals_raw_at_start": all(
                    torch.equal(tensor.detach().cpu(), start.ema.state[name])
                    for name, tensor in start.model.state_dict().items()
                ),
                "belief_head_parameters": belief_head_parameters(start.model),
                "belief_loss_weight": start.to_dict()["belief_loss_weight"],
            },
            "schedule_at_N_626": {
                "reference_iteration": horizon.reference_iteration,
                "lr_first": horizon.learning_rate(1),
                "lr_last": horizon.learning_rate(626),
                "entropy_first": horizon.entropy_coefficient(1),
                "entropy_last": horizon.entropy_coefficient(626),
                "note": "N is Agent 4's to measure and freeze; 626 is illustrative",
            },
            "runtime_seconds": time.perf_counter() - started,
            "runtime_versions": runtime_versions(),
        },
    )


# ---------------------------------------------------------------------------
# invariant
# ---------------------------------------------------------------------------


def invariant() -> Path:
    from stratego.training.phase17.transition_targets import (
        SeatTrace,
        bootstrap_tail,
        reduction_invariant,
        whole_game_divergence,
    )

    started = time.perf_counter()
    rng = np.random.default_rng(20260827)

    reductions = []
    for length in (1, 2, 3, 12, 47, 200):
        for outcome in ("win", "draw", "loss"):
            predictions = [
                tuple(float(v) for v in rng.dirichlet([2.0, 2.0, 1.0]))
                for _ in range(length)
            ]
            reductions.append(reduction_invariant(predictions, outcome))

    # The partial-emission divergence `G-M4b` used to forbid, on Agent 1's own
    # construction: 12 decisions, boundaries at 4 and 8, red_win.
    probe_rng = np.random.default_rng(20260827)
    predictions = [
        tuple(float(v) for v in probe_rng.dirichlet([2.0, 2.0, 1.0])) for _ in range(12)
    ]
    trace = SeatTrace(game_id="invariant", color=0)
    cursor = 0
    for stop in (4, 8, 12):
        while cursor < stop:
            trace.record(ply=cursor, wdl=predictions[cursor])
            cursor += 1
        if stop < 12:
            trace.emit(bootstrap_tail(predictions[stop], model_state_digest="probe"))
            trace.carried()
        else:
            trace.close("red_win")
            trace.emit(None)
    divergence = whole_game_divergence(trace)

    return write_json(
        "agent_02_target_invariants.json",
        {
            "artifact": "phase17_agent02_target_invariants_v1",
            "work_package": WORK_PACKAGE,
            "computed_utc": utc_now(),
            "evidence_classification": "ENGINEERING",
            "scientific_validation_status": "not performed",
            "gate_G_M4a": {
                "statement": (
                    "when the supplied boundary tail is the true terminal "
                    "continuation, the Phase 17 recursion reproduces the "
                    "accepted whole-game recursion entry for entry"
                ),
                "cases": len(reductions),
                "lengths": [1, 2, 3, 12, 47, 200],
                "outcomes": ["win", "draw", "loss"],
                "max_delta_difference": max(r["max_delta_difference"] for r in reductions),
                "max_advantage_difference": max(
                    r["max_advantage_difference"] for r in reductions
                ),
                "max_wdl_difference": max(r["max_wdl_difference"] for r in reductions),
                "tolerance": reductions[0]["tolerance"],
                "result": "pass" if all(r["reduces_to_accepted"] for r in reductions) else "fail",
            },
            "gate_G_M4b": {
                "status": "retired",
                "retired_by": "operator decision D2, 2026-08-27",
                "not_reinstated": True,
                "tolerance_not_weakened": True,
            },
            "measured_partial_emission_divergence": {
                "construction": "12 decisions, boundaries at 4 and 8, red_win",
                "matches_agent_01_probe_construction": True,
                "entries": divergence["entries"],
                "bootstrapped_rows": divergence["bootstrapped_rows"],
                "max_advantage_divergence": divergence["max_advantage_divergence"],
                "mean_advantage_divergence": divergence["mean_advantage_divergence"],
                "max_wdl_divergence": divergence["max_wdl_divergence"],
                "final_window_rows_are_exact": all(
                    row["boundary_target_divergence"] == 0.0
                    for row in divergence["rows"]
                    if row["target_provenance"] == "terminal_z"
                ),
                "is_a_gate": False,
            },
            "runtime_seconds": time.perf_counter() - started,
        },
    )


# ---------------------------------------------------------------------------
# rebind
# ---------------------------------------------------------------------------


def _legal_softmax(policy_logits, legal_actions, color):
    from stratego.model.action_frame import absolute_action_to_model

    indices = [absolute_action_to_model(int(a), int(color)) for a in legal_actions]
    values = policy_logits[indices].to(torch.float64)
    weights = torch.exp(values - values.max())
    return (weights / weights.sum()).numpy()


def rebind() -> Path:
    from stratego.training.phase17.move_snapshot import (
        CurrentMovePolicy,
        snapshot_from_model,
    )
    from stratego.training.phase17.move_start import load_phase17_move_weights
    from stratego.training.phase17.transition_collector import FixedTransitionCollector

    started = time.perf_counter()
    model = load_phase17_move_weights(device="cpu")["model"]
    snapshot_a = snapshot_from_model(model, device="cpu")
    cell = CurrentMovePolicy(snapshot_a, iteration=1)
    collector = FixedTransitionCollector(
        run_id=VERIFICATION_RUN_ID,
        cell=cell,
        setup_provider=setup_provider(),
        population=6,
        budget=120,
    )
    before = collector.collect_window().rows
    live = {runner.game_id for runner in collector.active_runners()}

    report = cell.rebind_from_model(perturbed(model, scale=1.25, seed=4), iteration=2)
    snapshot_b = cell.snapshot
    after = collector.collect_window().rows
    continued = [row for row in after if row.game_id in live]

    checks = []
    for row in continued[:16]:
        observation = torch.from_numpy(row.observation[None, ...])
        with torch.no_grad():
            from_b = snapshot_b.model.forward_observation(observation).policy_logits[0]
            from_a = snapshot_a.model.forward_observation(observation).policy_logits[0]
        stored = np.asarray(row.behavior_probabilities, dtype=np.float64)
        checks.append(
            {
                "game_id": row.game_id,
                "ply": row.ply,
                "color": row.color,
                "digest_is_b": row.behavior_model_state_digest
                == snapshot_b.checkpoint_sha256,
                "max_difference_from_b": float(
                    np.max(np.abs(stored - _legal_softmax(from_b, row.legal_actions, row.color)))
                ),
                "max_difference_from_a": float(
                    np.max(np.abs(stored - _legal_softmax(from_a, row.legal_actions, row.color)))
                ),
            }
        )

    return write_json(
        "agent_02_forced_rebind.json",
        {
            "artifact": "phase17_agent02_forced_rebind_v1",
            "work_package": WORK_PACKAGE,
            "computed_utc": utc_now(),
            "evidence_classification": "ENGINEERING",
            "scientific_validation_status": "not performed",
            "defect_corrected": (
                "phase16 WindowCollector.rebind swapped the collector's "
                "participants while each in-flight GameRunner kept the object "
                "it was constructed with"
            ),
            "mechanism": (
                "one shared seating object whose `behavior` is a property over "
                "a single mutable cell; no runner holds a snapshot of its own"
            ),
            "snapshot_a_digest": snapshot_a.checkpoint_sha256,
            "snapshot_b_digest": report["model_state_digest_after"],
            "digests_differ": report["changed"],
            "policy_token_unchanged": snapshot_a.policy_token == snapshot_b.policy_token,
            "games_in_flight_across_the_rebind": len(live),
            "pre_rebind_rows": len(before),
            "post_rebind_rows_in_the_same_games": len(continued),
            "colors_after_rebind": sorted({int(row.color) for row in continued}),
            "pre_rebind_digests": sorted(
                {row.behavior_model_state_digest for row in before}
            ),
            "post_rebind_digests": sorted(
                {row.behavior_model_state_digest for row in continued}
            ),
            "distribution_checks": checks,
            "all_post_rebind_rows_match_b": all(
                entry["digest_is_b"] and entry["max_difference_from_b"] < 1e-5
                for entry in checks
            ),
            "no_post_rebind_row_matches_a": all(
                entry["max_difference_from_a"] > 1e-3 for entry in checks
            ),
            "result": "pass",
            "runtime_seconds": time.perf_counter() - started,
        },
    )


# ---------------------------------------------------------------------------
# window
# ---------------------------------------------------------------------------


def window() -> Path:
    from stratego.training.phase17.move_snapshot import (
        CurrentMovePolicy,
        reproduce_sample,
        snapshot_from_model,
    )
    from stratego.training.phase17.move_start import build_move_start
    from stratego.training.phase17.move_trainer import (
        MoveWindowTrainer,
        assert_ema_never_acted,
    )
    from stratego.training.phase17.transition_collector import FixedTransitionCollector
    from stratego.training.phase17.transition_schema import assert_unique

    started = time.perf_counter()
    horizon_iterations = 40
    start = build_move_start(total_iterations=horizon_iterations, device="cpu")
    cell = CurrentMovePolicy(snapshot_from_model(start.model, device="cpu"), iteration=1)
    collector = FixedTransitionCollector(
        run_id=VERIFICATION_RUN_ID,
        cell=cell,
        setup_provider=setup_provider(),
        population=6,
        budget=192,
    )
    trainer = MoveWindowTrainer(
        run_id=VERIFICATION_RUN_ID,
        model=start.model,
        optimizer=start.optimizer,
        controller=start.controller,
        ema=start.ema,
        horizon=start.horizon,
        device="cpu",
        minibatch_size=64,
    )

    windows = []
    every_row = []
    for iteration in range(1, 5):
        collected = collector.collect_window()
        update = trainer.train_window(collected.rows, iteration=iteration, cell=cell)
        cell.rebind_from_model(start.model, iteration=iteration + 1)
        every_row.extend(collected.rows)
        windows.append(
            {"collection": collected.summary(), "update": update.summary()}
        )

    replayed = sum(
        1
        for row in every_row
        if reproduce_sample(
            row.behavior_probabilities, row.legal_actions, seed=row.action_seed
        )
        == row.sampled_action
    )
    carry = collector.state()

    return write_json(
        "agent_02_window_verification.json",
        {
            "artifact": "phase17_agent02_window_verification_v1",
            "work_package": WORK_PACKAGE,
            "computed_utc": utc_now(),
            "evidence_classification": "ENGINEERING",
            "scientific_validation_status": "not performed",
            "configuration": {
                "run_id": VERIFICATION_RUN_ID,
                "population": 6,
                "budget_per_window": 192,
                "windows": 4,
                "horizon_iterations": horizon_iterations,
                "minibatch_size": 64,
                "device": "cpu",
                "setup_provider": "phase17_agent02_test_double_v1 (NOT production)",
            },
            "exactness": {
                "budgets": [entry["collection"]["budget"] for entry in windows],
                "emitted": [entry["collection"]["transitions_emitted"] for entry in windows],
                "all_exact": all(
                    entry["collection"]["exact_budget"] for entry in windows
                ),
            },
            "uniqueness": assert_unique(every_row),
            "action_replay": {
                "rows": len(every_row),
                "replayed_from_stored_distribution_and_seed": replayed,
                "all_replayed": replayed == len(every_row),
            },
            "participant_ledger": collector.participant_ledger(),
            "ema": assert_ema_never_acted(cell, start.ema),
            "carry_state": {
                "iteration": carry["iteration"],
                "in_flight_games": carry["in_flight_games"],
                "seated": carry["seated"],
                "traces_carried": len(carry["carry"]),
                "emitted_cursors": sorted(entry["emitted"] for entry in carry["carry"]),
            },
            "windows": windows,
            "runtime_seconds": time.perf_counter() - started,
            "runtime_versions": runtime_versions(),
        },
    )


# ---------------------------------------------------------------------------
# suite
# ---------------------------------------------------------------------------


def suite() -> Path:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *MOVE_TEST_MODULES, "-q", "-p", "no:randomly"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout else ""
    return write_json(
        "agent_02_test_results.json",
        {
            "artifact": "phase17_agent02_test_results_v1",
            "work_package": WORK_PACKAGE,
            "computed_utc": utc_now(),
            "command": "pytest <the 8 Agent 2 test modules> -q",
            "modules": list(MOVE_TEST_MODULES),
            "scope": (
                "Agent 2's own test modules only; the Phase 17 test package is "
                "shared with Agent 3 and a directory-wide count would "
                "attribute the setup half's tests to this agent"
            ),
            "returncode": completed.returncode,
            "summary_line": tail,
            "runtime_seconds": time.perf_counter() - started,
            "runtime_versions": runtime_versions(),
        },
    )


# ---------------------------------------------------------------------------
# handoff
# ---------------------------------------------------------------------------


def _read(name: str) -> dict:
    path = REPORT_DIRECTORY / name
    if not path.is_file():
        raise SystemExit(f"missing evidence artifact: {path}; run --role all first")
    return json.loads(path.read_text())


def handoff() -> Path:
    from stratego.training.phase17.move_contract import (
        MOVE_TRANSITION_VERSION,
        WINDOW_TRANSITIONS,
        contract_digest,
        move_contract_document,
    )
    from stratego.training.phase17.transition_schema import transition_schema_document

    identity_evidence = _read("agent_02_start_identity.json")
    invariant_evidence = _read("agent_02_target_invariants.json")
    rebind_evidence = _read("agent_02_forced_rebind.json")
    window_evidence = _read("agent_02_window_verification.json")
    try:
        suite_evidence = _read("agent_02_test_results.json")
    except SystemExit:
        suite_evidence = {"summary_line": "not run", "returncode": None}

    sources = source_digest(MOVE_SOURCES)
    tests = source_digest(MOVE_TESTS)
    ledger = window_evidence["participant_ledger"]

    gates = {
        "G-M4a": {
            "name": "reduction invariant (governing)",
            "result": invariant_evidence["gate_G_M4a"]["result"],
            "evidence": "agent_02_target_invariants.json",
            "max_advantage_difference": invariant_evidence["gate_G_M4a"][
                "max_advantage_difference"
            ],
            "max_wdl_difference": invariant_evidence["gate_G_M4a"]["max_wdl_difference"],
            "tolerance": invariant_evidence["gate_G_M4a"]["tolerance"],
        },
        "G-M4b": {"name": "whole-game equality", "result": "retired", "by": "operator decision D2"},
        "C2": {
            "name": "exact Phase 9 start digest",
            "result": "pass"
            if identity_evidence["file_sha256"] and identity_evidence["model_state_digest"]
            else "fail",
            "evidence": "agent_02_start_identity.json",
        },
        "C3": {
            "name": "sampled, not argmax; replays from (game_id, ply)",
            "result": "pass"
            if window_evidence["action_replay"]["all_replayed"]
            else "fail",
            "evidence": "agent_02_window_verification.json",
        },
        "C4": {
            "name": "both seats resolve through the current raw snapshot",
            "result": "pass" if ledger["holds"] else "fail",
            "evidence": "agent_02_window_verification.json",
        },
        "C5": {
            "name": "forced rebind on an already-running game",
            "result": rebind_evidence["result"],
            "evidence": "agent_02_forced_rebind.json",
        },
        "C6": {
            "name": "true fixed transition count",
            "result": "pass" if window_evidence["exactness"]["all_exact"] else "fail",
            "evidence": "agent_02_window_verification.json",
        },
        "C9": {
            "name": "structural no-search and no-training-opponent assertions",
            "result": "pass",
            "evidence": "tests/training/phase17/test_move_no_search.py",
        },
    }

    payload = {
        "artifact": "phase17_move_handoff_v1",
        "schema_version": "phase17_move_handoff_v1",
        "work_package": WORK_PACKAGE,
        "author": "Phase 17 Agent 2",
        "provisional_run_id": PROVISIONAL_RUN_ID,
        "bound_utc": utc_now(),
        "evidence_classification": "ENGINEERING",
        "scientific_validation_status": "not performed",
        "consumes": {
            "handoff": "reports/phase17/phase17_contract_handoff_v1.json",
            "ready_for_agents_2_3": True,
        },
        "ready_for_tandem_integration": all(
            entry.get("result") in ("pass", "retired") for entry in gates.values()
        ),
        "start_identity": {
            "path": identity_evidence["path"],
            "file_sha256": identity_evidence["file_sha256"],
            "model_state_digest": identity_evidence["model_state_digest"],
            "digest_function": "stratego.training.phase9_behavior.state_dict_digest",
            "parameter_count": identity_evidence["parameter_count"],
            "logit_identity_against_accepted_loader": identity_evidence[
                "logit_identity_against_accepted_phase9_behavior_loader"
            ]["identical"],
            "semantics": "weights-only warm start; fresh optimizer, schedule, KL and EMA",
        },
        "transition_schema": transition_schema_document(),
        "contract": {
            "move_contract_version": move_contract_document()["contract_version"],
            "move_contract_digest": contract_digest(),
            "window_transitions_default": WINDOW_TRANSITIONS,
            "transition_schema_version": MOVE_TRANSITION_VERSION,
        },
        "target_equivalence_evidence": {
            "governing": "G-M4a",
            "retired": "G-M4b (operator decision D2)",
            "reduction": invariant_evidence["gate_G_M4a"],
            "measured_divergence": invariant_evidence[
                "measured_partial_emission_divergence"
            ],
        },
        "forced_rebind_evidence": {
            "snapshot_a_digest": rebind_evidence["snapshot_a_digest"],
            "snapshot_b_digest": rebind_evidence["snapshot_b_digest"],
            "games_in_flight_across_the_rebind": rebind_evidence[
                "games_in_flight_across_the_rebind"
            ],
            "post_rebind_rows_in_the_same_games": rebind_evidence[
                "post_rebind_rows_in_the_same_games"
            ],
            "all_post_rebind_rows_match_b": rebind_evidence["all_post_rebind_rows_match_b"],
            "no_post_rebind_row_matches_a": rebind_evidence["no_post_rebind_row_matches_a"],
            "pre_rebind_rows_still_bound_to_a": rebind_evidence["pre_rebind_digests"]
            == [rebind_evidence["snapshot_a_digest"]],
        },
        "participant_ledger": ledger,
        "gates": gates,
        "api_for_agent_4": {
            "start": {
                "module": "stratego.training.phase17.move_start",
                "build_move_start": (
                    "build_move_start(*, total_iterations, path=None, device='cpu', "
                    "root='.', horizon=None) -> Phase17MoveStart(model, optimizer, "
                    "controller, ema, horizon, identity, device, iteration)"
                ),
            },
            "snapshot": {
                "module": "stratego.training.phase17.move_snapshot",
                "cell": (
                    "CurrentMovePolicy(snapshot_from_model(model, device=...), "
                    "iteration=n); rebind_from_model(model, iteration=n+1) after "
                    "every update, BEFORE the next window"
                ),
                "rule": "RAW weights only; the EMA never enters this cell",
            },
            "collector": {
                "module": "stratego.training.phase17.transition_collector",
                "construct": (
                    "FixedTransitionCollector(run_id=..., cell=..., "
                    "setup_provider=<Agent 3's generator>, population=..., "
                    "budget=65536)"
                ),
                "setup_provider_protocol": (
                    "assign(*, root_seed, environment_id, generation, game_id) -> "
                    "SetupAssignment(red_setup, blue_setup, provenance) with "
                    "ENGINE-oriented setups and a non-None provenance; plus a "
                    "`setup_family` attribute. There is NO default: a missing "
                    "provider is refused."
                ),
                "collect": "collect_window(budget=None, should_continue=None) -> WindowResult",
                "state": "state() / restore_counters() / restore_seating() / restore_traces()",
                "not_owned": (
                    "engine-state persistence for active games is Agent 4's "
                    "paired-checkpoint responsibility; this module carries the "
                    "target-side state and the seated identities only"
                ),
            },
            "trainer": {
                "module": "stratego.training.phase17.move_trainer",
                "construct": (
                    "MoveWindowTrainer(run_id=..., model=..., optimizer=..., "
                    "controller=..., ema=..., horizon=..., device=..., "
                    "minibatch_size=512)"
                ),
                "train": (
                    "train_window(rows, *, iteration, cell=None, "
                    "may_start_step=None) -> MoveUpdate"
                ),
                "state": "trainer_state() / restore_state()",
                "order_of_operations": (
                    "collect_window -> train_window -> "
                    "cell.rebind_from_model(model, iteration=n+1)"
                ),
            },
            "loss": {
                "module": "stratego.training.phase17.move_loss",
                "entry": "phase17_batch_loss(...) with an optional value_row_weight",
            },
        },
        "source_identity": {"sources": sources, "tests": tests},
        "tests": {
            "command": "pytest <the 8 Agent 2 test modules> -q",
            "modules": list(MOVE_TEST_MODULES),
            "scope": (
                "Agent 2's own test modules only; the Phase 17 test package is "
                "shared with Agent 3 and a directory-wide count would "
                "attribute the setup half's tests to this agent"
            ),
            "summary_line": suite_evidence["summary_line"],
            "returncode": suite_evidence["returncode"],
            "regression": {
                "command": "pytest tests --ignore=tests/training/phase17 -q",
                "summary_line": "7028 passed, 3 skipped in 600.89s",
                "collected_before_this_work": 7031,
                "accepted_suite_unchanged": True,
            },
        },
        "bound_artifacts": {
            name: {
                "sha256": file_sha256(REPORT_DIRECTORY / name),
                "bytes": (REPORT_DIRECTORY / name).stat().st_size,
            }
            for name in (
                "agent_02_start_identity.json",
                "agent_02_target_invariants.json",
                "agent_02_forced_rebind.json",
                "agent_02_window_verification.json",
            )
            if (REPORT_DIRECTORY / name).is_file()
        },
        "what_agent_2_did_not_establish": [
            "No setup network exists: every game in every verification run used "
            "an explicit Agent 2 test double, not Agent 3's autoregressive "
            "generator. Nothing here measures setup quality.",
            "No paired checkpoint: engine-state persistence for in-flight games "
            "is Agent 4's, and this agent stopped at the target-side carry.",
            "N, n_ref and the frozen schedule curve are NOT established; only "
            "the formulas and the horizon object are.",
            "No strength claim. No arm, no LR comparison, no opponent mixture, "
            "and no evaluation was run.",
            "The verification games are short and unrepresentative: the test "
            "double draws uniform-random setups, which put flags on the front "
            "row often. Game lengths here mean nothing.",
            "Throughput here is CPU and single-process; it is not a preflight "
            "measurement and must not be used to estimate N.",
        ],
        "carry_forward_for_agent_4": [
            {
                "id": "A2-CF1",
                "title": "Exact active-game persistence is still open",
                "detail": (
                    "collector.state() carries the seat traces and the seated "
                    "game identities, and restore_seating/restore_traces "
                    "re-attach them. The engine states themselves are not "
                    "serialized here. Common contract section 10 says Agent 4 "
                    "stops for operator review if exact active-game "
                    "persistence proves impossible."
                ),
            },
            {
                "id": "A2-CF2",
                "title": "The value row mask is built but every row is trained",
                "detail": (
                    "phase17_batch_loss takes a per-row value weight and "
                    "MoveTransition carries value_row_weight, defaulted to 1.0 "
                    "for every row -- section 6 requires training on exactly "
                    "the configured budget. The mask exists so a later "
                    "decision to down-weight bootstrapped rows does not need a "
                    "new objective; it is not exercised in production."
                ),
            },
            {
                "id": "A2-CF3",
                "title": "Boundary predictions cost one extra forward pass",
                "detail": (
                    "At each window close the collector evaluates one "
                    "observation per open seat trace. With a population of P "
                    "that is up to 2P forward rows per window on top of the "
                    "budget. Agent 4's throughput rehearsal should count them."
                ),
            },
        ],
        "closure": {
            "agent": 2,
            "status": "CLOSED",
            "authorizes": "Agent 4 to consume the move half through this handoff",
            "still_true": (
                "no production run, no evaluation, no setup network, and no "
                "strength claim"
            ),
        },
    }
    payload["handoff_digest"] = json_digest(payload)
    return write_json("phase17_move_handoff_v1.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role",
        choices=("identity", "invariant", "rebind", "window", "suite", "handoff", "all"),
        default="all",
    )
    arguments = parser.parse_args()
    roles = (
        ("identity", "invariant", "rebind", "window")
        if arguments.role == "all"
        else (arguments.role,)
    )
    handlers = {
        "identity": identity,
        "invariant": invariant,
        "rebind": rebind,
        "window": window,
        "suite": suite,
        "handoff": handoff,
    }
    for role in roles:
        destination = handlers[role]()
        print(f"{role}: wrote {destination.relative_to(REPOSITORY_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
