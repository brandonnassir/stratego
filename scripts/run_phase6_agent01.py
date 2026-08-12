#!/usr/bin/env python3
"""Phase 6 Agent 1 acceptance harness: model contract v2 and normalized actions.

Runs every measurement the Phase 6 Agent 1 instructions require and writes

    reports/phase_6_data/agent_01_model_contract_v2.json

What this script is and is not
------------------------------
It validates the *migration*: that the model-facing action space can be moved
into the acting player's normalized frame without changing a single engine
identifier, without weakening hidden-information safety, and without disturbing
Phase 4 match semantics.

It is not architecture selection, not a benchmark and not training. The network
used throughout is the untrained Phase 5 integration fixture, whose weights are
bit-identical to the ones the accepted Phase 5 run used -- only the contract
metadata around them moved. Every playing-strength number produced here is
meaningless by construction and is recorded only to show that matches ran clean.

Usage::

    python scripts/run_phase6_agent01.py              # full acceptance run
    python scripts/run_phase6_agent01.py --quick      # fast smoke run
    python scripts/run_phase6_agent01.py --skip-pytest
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.engine.actions import decode_action, encode_action  # noqa: E402
from stratego.engine.constants import (  # noqa: E402
    ACTION_SPACE_SIZE,
    BLUE,
    IMPLEMENTATION_VERSION,
    OBSERVATION_VERSION,
    PLAYERS,
    RED,
    RULES_VERSION,
    TRAINING_RULES,
)
from stratego.engine.coordinates import square_from_name, square_name  # noqa: E402
from stratego.engine.legal_moves import legal_action_mask, legal_actions  # noqa: E402
from stratego.engine.observation import build_observation  # noqa: E402
from stratego.engine.permutation import permute_hidden_identities  # noqa: E402
from stratego.engine.random_play import play_random_game_to_ply  # noqa: E402
from stratego.evaluation.match_runner import (  # noqa: E402
    compare_results,
    replay_stored_match,
    reproduce_match,
    results_digest,
    run_schedule,
)
from stratego.evaluation.match_spec import build_paired_schedule  # noqa: E402
from stratego.evaluation.policy import build_policy_input  # noqa: E402
from stratego.evaluation.registry import policy_ref  # noqa: E402
from stratego.evaluation.reporting import write_json  # noqa: E402
from stratego.evaluation.setup_bank import SetupBank  # noqa: E402
from stratego.evaluation.statistics import summarize_matchup  # noqa: E402
from stratego.model.action_frame import (  # noqa: E402
    absolute_action_to_model,
    absolute_legal_actions_to_model,
    absolute_legal_mask_to_model,
    action_frame_summary,
    model_action_to_absolute,
    model_legal_actions_to_absolute,
    model_legal_mask_to_absolute,
)
from stratego.model.checkpoint import (  # noqa: E402
    CheckpointError,
    accepted_under_contract_v1,
    build_checkpoint_payload,
    file_digest,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_payload,
)
from stratego.model.contract import (  # noqa: E402
    ACTION_ENCODING_VERSION,
    ENGINE_ACTION_FRAME,
    LEGACY_CONTRACT_V1,
    MODEL_CONTRACT_VERSION,
    POLICY_ACTION_FRAME,
    TOKEN_SQUARE_FRAME,
    contract_summary,
)
from stratego.model.integration_model import build_integration_model  # noqa: E402
from stratego.model.policy_adapter import (  # noqa: E402
    DECISION_MODE_CATEGORICAL,
    DECISION_MODE_GREEDY,
    GreedyNeuralPolicy,
    SeededCategoricalNeuralPolicy,
    greedy_action,
)
from stratego.model.tokenization import tokenize_numpy_observation  # noqa: E402
from stratego.training.belief_targets import dense_belief_target  # noqa: E402
from tests.observation.test_perspective import (  # noqa: E402
    mirror_action,
    mirrored_games,
    play_mirrored,
)

SCHEMA_VERSION = "phase_6_agent_01_v1"
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_6_data"
CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / "checkpoints"
PHASE_5_ACCEPTANCE = (
    REPOSITORY_ROOT / "reports" / "phase_5_data" / "agent_01_phase5_acceptance.json"
)

#: The same seed the accepted Phase 5 fixture was built from, so the v2
#: checkpoint differs from the v1 one only in its recorded semantics.
MODEL_SEED = 20250501

#: The Phase 4 core ladder, unchanged.
CORE_BASELINES = (
    "random_legal",
    "basic_heuristic",
    "tactical_rule_based",
    "strategic_rule_based",
)

#: Measured before any Phase 6 edit, at the commit that accepted Phase 5.
#: Reproduce with: git stash && python -m pytest -q
PREEXISTING_SUITE = {
    "command": "python -m pytest -q",
    "commit": "8f4f5e39dea0ef6bd1e7fbf6f62c82a74d1e1628",
    "recorded_before_any_phase_6_edit": True,
    "passed": 2155,
    "failed": 0,
    "errors": 0,
    "skipped": 2,
    "seconds": 102.29,
    "note": "identical to the totals recorded by the accepted Phase 5 run",
}

SEEDS = {
    "model_initialisation": MODEL_SEED,
    "hidden_information": 90210,
    "symmetry": 0,
    "position_corpus": 0,
    "evaluation_policy_seed": 13,
    "bootstrap": 17,
}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001 - a missing git is not a Phase 6 failure
        return "unknown"


def environment() -> dict:
    return {
        "commit": git_commit(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_threads": torch.get_num_threads(),
    }


def position_corpus(count: int) -> list:
    """Real non-terminal positions across both colours and the whole game.

    Two seed families so the corpus is not one game sampled repeatedly, and an
    alternating ply parity so both colours are the acting player -- the
    conversion is per-player, and a single-colour corpus would pass with the
    player argument ignored.
    """
    plies = (0, 7, 18, 25, 40, 61, 75, 96, 120, 155, 190, 233, 260, 301, 44, 89)[:count]
    seeds = (0, 0, 0, 0, 0, 0, 0, 0, 37, 37, 37, 37, 37, 37, 91, 91)[:count]
    corpus = []
    for ply, seed in zip(plies, seeds):
        for attempt in range(seed, seed + 400):
            state = play_random_game_to_ply(attempt, ply, rules=TRAINING_RULES)
            if not state.terminal and state.total_moves == ply:
                corpus.append(state)
                break
    return corpus


# ---------------------------------------------------------------------------
# 1. Prerequisite: Phase 5 acceptance
# ---------------------------------------------------------------------------


def verify_prerequisites() -> dict:
    """Read the real Phase 5 acceptance artifact; do not take it on trust."""
    problems: list[str] = []
    if not PHASE_5_ACCEPTANCE.exists():
        return {
            "phase_5_accepted": False,
            "problems": [f"{PHASE_5_ACCEPTANCE} does not exist"],
        }
    payload = json.loads(PHASE_5_ACCEPTANCE.read_text())

    if payload.get("status") != "PASS":
        problems.append(f"Phase 5 status is {payload.get('status')!r}, expected PASS")
    gates_true = int(payload.get("gates_true", 0))
    gates_total = int(payload.get("gates_total", 0))
    if (gates_true, gates_total) != (22, 22):
        problems.append(f"Phase 5 gates are {gates_true}/{gates_total}, expected 22/22")
    if payload.get("quick_mode"):
        problems.append("the accepted Phase 5 run was a --quick run")

    contract = payload.get("model", {}).get("contract", {})
    recorded_contract = contract.get("model_contract_version") or payload.get(
        "frozen_contracts", {}
    ).get("model_contract", {}).get("actual")
    architecture = payload.get("model", {}).get("architecture_id")

    return {
        "phase_5_accepted": not problems,
        "phase_5_status": payload.get("status"),
        "phase_5_gates": f"{gates_true}/{gates_total}",
        "phase_5_commit": payload.get("environment", {}).get("commit"),
        "phase_5_quick_mode": bool(payload.get("quick_mode")),
        "phase_5_model_contract_version": recorded_contract,
        "phase_5_architecture_id": architecture,
        "phase_5_headline_numbers": payload.get("headline_numbers", {}),
        "integration_model_is_integration_only": _fixture_is_integration_only(),
        "preexisting_suite": PREEXISTING_SUITE,
        "problems": problems,
    }


def _fixture_is_integration_only() -> bool:
    """The Phase 5 network must still declare itself a fixture, not a candidate."""
    from stratego.model.integration_model import FIXTURE_NOTE, MODEL_ARCHITECTURE_ID

    return MODEL_ARCHITECTURE_ID == "integration_model_v1" and "integration" in FIXTURE_NOTE.lower()


# ---------------------------------------------------------------------------
# 2. Exhaustive action-frame audit
# ---------------------------------------------------------------------------


#: Geometry pinned by hand rather than by formula: boundaries, long scout runs,
#: lateral and vertical moves, and the first and last rows and columns.
PINNED_GEOMETRY = (
    ("a1", "a2", "single step, first square"),
    ("j10", "j9", "single step, last square"),
    ("a1", "b1", "lateral, first row"),
    ("j10", "i10", "lateral, last row"),
    ("a1", "a10", "full-column scout run"),
    ("a5", "j5", "full-row scout run"),
    ("a10", "a9", "first column, last row"),
    ("j1", "j2", "last column, first row"),
    ("e5", "e6", "centre, beside the lakes"),
    ("d4", "d7", "long scout past a lake column"),
    ("a1", "j10", "corner to corner"),
)


def audit_action_round_trips() -> dict:
    """All 10,000 identifiers, both players, both directions."""
    started = time.perf_counter()

    forward_cases = forward_mismatches = 0
    reverse_cases = reverse_mismatches = 0
    collisions = 0
    bijection_ok = True
    encoding_preserved = True

    for player in sorted(PLAYERS):
        image = []
        for action in range(ACTION_SPACE_SIZE):
            model_action = absolute_action_to_model(action, player)
            image.append(model_action)
            forward_cases += 1
            if model_action_to_absolute(model_action, player) != action:
                forward_mismatches += 1

            reverse_cases += 1
            if absolute_action_to_model(model_action_to_absolute(action, player), player) != action:
                reverse_mismatches += 1

            # The encoding rule itself must be untouched by the frame change.
            source, destination = decode_action(model_action)
            if encode_action(source, destination) != model_action:
                encoding_preserved = False

        distinct = len(set(image))
        collisions += ACTION_SPACE_SIZE - distinct
        if sorted(image) != list(range(ACTION_SPACE_SIZE)):
            bijection_ok = False

    pinned = []
    pinned_mismatches = 0
    for source_name, destination_name, note in PINNED_GEOMETRY:
        action = encode_action(square_from_name(source_name), square_from_name(destination_name))
        red = absolute_action_to_model(action, RED)
        blue = absolute_action_to_model(action, BLUE)
        blue_source, blue_destination = decode_action(blue)
        entry = {
            "absolute": f"{source_name}->{destination_name}",
            "absolute_action_id": action,
            "red_model_action_id": red,
            "blue_model_action_id": blue,
            "blue_reads_as": f"{square_name(blue_source)}->{square_name(blue_destination)}",
            "note": note,
        }
        if red != action:
            pinned_mismatches += 1
        if model_action_to_absolute(blue, BLUE) != action:
            pinned_mismatches += 1
        pinned.append(entry)

    # Red must be the identity and blue must not be, or the player argument is
    # being ignored somewhere.
    red_identity = all(
        absolute_action_to_model(action, RED) == action for action in range(ACTION_SPACE_SIZE)
    )
    blue_moved = sum(
        1 for action in range(ACTION_SPACE_SIZE) if absolute_action_to_model(action, BLUE) != action
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "forward_cases": forward_cases,
        "forward_mismatches": forward_mismatches,
        "reverse_cases": reverse_cases,
        "reverse_mismatches": reverse_mismatches,
        "collisions": collisions,
        "bijection_ok": bijection_ok,
        "action_encoding_preserved": encoding_preserved,
        "red_is_identity": red_identity,
        "blue_actions_moved": blue_moved,
        "pinned_geometry": pinned,
        "pinned_geometry_mismatches": pinned_mismatches,
        "frames": action_frame_summary(),
        "seconds": round(time.perf_counter() - started, 3),
    }


# ---------------------------------------------------------------------------
# 3. Legal-action and dense-mask equivalence over real positions
# ---------------------------------------------------------------------------


def audit_legal_products(corpus: list) -> dict:
    """List/mask equivalence in both frames, over real engine positions."""
    started = time.perf_counter()

    positions = 0
    comparisons = 0
    list_mask_mismatches = 0
    round_trip_mismatches = 0
    mask_round_trip_mismatches = 0
    dtype_changes = 0
    per_position = []
    acting_players = set()
    legal_counts = []

    for state in corpus:
        player = state.acting_player
        acting_players.add(player)
        absolute = legal_actions(state)
        mask = legal_action_mask(state, absolute)
        positions += 1
        legal_counts.append(len(absolute))

        model_actions = absolute_legal_actions_to_model(absolute, player)
        model_mask = absolute_legal_mask_to_model(mask, player)
        comparisons += len(absolute)

        # 3. the transformed list must equal the nonzero transformed-mask indices
        if list(model_actions) != sorted(int(a) for a in np.flatnonzero(model_mask)):
            list_mask_mismatches += 1
        if model_mask.dtype != mask.dtype:
            dtype_changes += 1

        # 4/5. back to absolute, exact equality with the engine's own set
        restored = model_legal_actions_to_absolute(model_actions, player)
        if set(restored) != set(absolute) or len(restored) != len(absolute):
            round_trip_mismatches += 1
        if not np.array_equal(model_legal_mask_to_absolute(model_mask, player), mask):
            mask_round_trip_mismatches += 1

        per_position.append(
            {
                "ply": state.total_moves,
                "acting_player": int(player),
                "legal_actions": len(absolute),
                "model_frame_differs": set(model_actions) != set(absolute),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "positions_tested": positions,
        "action_comparisons": comparisons,
        "list_mask_mismatches": list_mask_mismatches,
        "absolute_round_trip_mismatches": round_trip_mismatches,
        "mask_round_trip_mismatches": mask_round_trip_mismatches,
        "mask_dtype_changes": dtype_changes,
        "both_colours_covered": acting_players == set(PLAYERS),
        "legal_action_count": {
            "min": min(legal_counts) if legal_counts else 0,
            "max": max(legal_counts) if legal_counts else 0,
            "mean": round(statistics.fmean(legal_counts), 2) if legal_counts else 0.0,
            "note": "observed, not an assumed maximum; nothing here caps the count",
        },
        "per_position": per_position,
        "seconds": round(time.perf_counter() - started, 3),
    }


# ---------------------------------------------------------------------------
# 4. Colour symmetry
# ---------------------------------------------------------------------------


def audit_symmetry(model, trials: int) -> dict:
    """Mirrored, colour-swapped positions must be identical in the model frame.

    The pair construction is the accepted Phase 2 mirrored-game instrument, so
    both positions are genuinely reachable states produced by the frozen engine.
    """
    started = time.perf_counter()

    compared = 0
    observation_mismatches = 0
    mask_mismatches = 0
    legal_set_mismatches = 0
    normalized_action_mismatches = 0
    absolute_not_mirrored = 0
    v1_frame_disagreements = 0
    skipped_terminal = 0

    plies_cycle = (0, 1, 6, 14, 25, 40, 60, 90)
    seed = 0
    while compared < trials:
        plies = plies_cycle[compared % len(plies_cycle)]
        original, twin = mirrored_games(seed)
        play_mirrored(original, twin, plies, seed)
        seed += 1
        if original.terminal or twin.terminal:
            skipped_terminal += 1
            continue
        compared += 1

        left, right = original.acting_player, twin.acting_player
        left_observation = build_observation(original, left)
        right_observation = build_observation(twin, right)
        if not np.array_equal(left_observation, right_observation):
            observation_mismatches += 1

        left_actions = legal_actions(original)
        right_actions = legal_actions(twin)
        if absolute_legal_actions_to_model(left_actions, left) != absolute_legal_actions_to_model(
            right_actions, right
        ):
            legal_set_mismatches += 1
        if not np.array_equal(
            absolute_legal_mask_to_model(legal_action_mask(original, left_actions), left),
            absolute_legal_mask_to_model(legal_action_mask(twin, right_actions), right),
        ):
            mask_mismatches += 1

        with torch.no_grad():
            left_logits = model(tokenize_numpy_observation(left_observation)).policy_logits[0]
            right_logits = model(tokenize_numpy_observation(right_observation)).policy_logits[0]

        left_choice = greedy_action(left_logits, absolute_legal_actions_to_model(left_actions, left))
        right_choice = greedy_action(
            right_logits, absolute_legal_actions_to_model(right_actions, right)
        )
        if left_choice != right_choice:
            normalized_action_mismatches += 1

        left_absolute = model_action_to_absolute(left_choice, left)
        right_absolute = model_action_to_absolute(right_choice, right)
        if mirror_action(left_absolute) != right_absolute:
            absolute_not_mirrored += 1

        # Negative control: the retired v1 rule, selecting over absolute ids.
        if mirror_action(greedy_action(left_logits, left_actions)) != greedy_action(
            right_logits, right_actions
        ):
            v1_frame_disagreements += 1

    mismatches = (
        observation_mismatches
        + mask_mismatches
        + legal_set_mismatches
        + normalized_action_mismatches
        + absolute_not_mirrored
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "trials": compared,
        "skipped_terminal_pairs": skipped_terminal,
        "observation_mismatches": observation_mismatches,
        "normalized_mask_mismatches": mask_mismatches,
        "normalized_legal_set_mismatches": legal_set_mismatches,
        "normalized_action_mismatches": normalized_action_mismatches,
        "absolute_actions_not_mirrored": absolute_not_mirrored,
        "total_mismatches": mismatches,
        "v1_absolute_frame_disagreements": v1_frame_disagreements,
        "v1_control_proves_the_migration_changed_something": v1_frame_disagreements > 0,
        "construction": (
            "colour-swapped 180-degree-rotated twin games advanced through mirrored "
            "actions; the twin's first player is blue so the ply counts match and the "
            "acting colours are opposite"
        ),
        "seconds": round(time.perf_counter() - started, 3),
    }


# ---------------------------------------------------------------------------
# 5. Checkpoint compatibility across the v1/v2 boundary
# ---------------------------------------------------------------------------


def audit_checkpoint_compatibility(model, v2_path: Path) -> dict:
    """v1 files must fail here; v2 files must have failed there."""
    started = time.perf_counter()
    cases: dict[str, dict] = {}

    def record(name: str, *, expect_rejected: bool, action) -> None:
        try:
            action()
        except CheckpointError as error:
            cases[name] = {
                "expected_rejected": expect_rejected,
                "rejected": True,
                "error": type(error).__name__,
                "message": str(error)[:240],
                "as_expected": expect_rejected,
            }
            return
        cases[name] = {
            "expected_rejected": expect_rejected,
            "rejected": False,
            "error": None,
            "message": "ACCEPTED",
            "as_expected": not expect_rejected,
        }

    def v1_payload() -> dict:
        payload = build_checkpoint_payload(model)
        payload["model_contract_version"] = LEGACY_CONTRACT_V1["model_contract_version"]
        payload["policy_action_frame"] = LEGACY_CONTRACT_V1["policy_action_frame"]
        del payload["engine_action_frame"]
        return payload

    v1_file = CHECKPOINT_DIRECTORY / "integration_model_v1.pt"
    digest_before = file_digest(v1_file) if v1_file.exists() else None

    record("shipped_v1_file_on_disk", expect_rejected=True, action=lambda: load_checkpoint(v1_file))
    record(
        "synthetic_v1_payload",
        expect_rejected=True,
        action=lambda: validate_checkpoint_payload(v1_payload()),
    )
    for field, wrong in (
        ("policy_action_frame", "absolute_engine_squares"),
        ("policy_action_frame", "something_else_entirely"),
        ("engine_action_frame", "perspective_normalized_squares"),
        ("model_contract_version", "model_contract_v1"),
        ("model_contract_version", "model_contract_v3"),
    ):
        def mutated(field=field, wrong=wrong):
            payload = build_checkpoint_payload(model)
            payload[field] = wrong
            validate_checkpoint_payload(payload)

        record(f"wrong_{field}_{wrong}", expect_rejected=True, action=mutated)

    for field in ("policy_action_frame", "engine_action_frame"):
        def missing(field=field):
            payload = build_checkpoint_payload(model)
            del payload[field]
            validate_checkpoint_payload(payload)

        record(f"missing_{field}", expect_rejected=True, action=missing)

    record(
        "current_v2_payload",
        expect_rejected=False,
        action=lambda: validate_checkpoint_payload(build_checkpoint_payload(model)),
    )
    record("current_v2_file", expect_rejected=False, action=lambda: load_checkpoint(v2_path))

    digest_after = file_digest(v1_file) if v1_file.exists() else None
    _, v2_metadata = load_checkpoint(v2_path)

    return {
        "schema_version": SCHEMA_VERSION,
        "cases": cases,
        "cases_total": len(cases),
        "cases_as_expected": sum(1 for case in cases.values() if case["as_expected"]),
        "rejection_failures": sum(1 for case in cases.values() if not case["as_expected"]),
        "v1_file_unmodified_by_refused_load": digest_before == digest_after,
        "v1_file_digest": digest_before,
        "v2_would_be_refused_under_v1": not accepted_under_contract_v1(
            build_checkpoint_payload(model)
        ),
        "v1_rule_still_accepts_v1": accepted_under_contract_v1(v1_payload()),
        "v2_checkpoint": {
            "path": str(v2_path.relative_to(REPOSITORY_ROOT)),
            "file_digest": file_digest(v2_path),
            "state_dict_digest": v2_metadata["state_dict_digest"],
            "model_contract_version": v2_metadata["model_contract_version"],
            "policy_action_frame": v2_metadata["policy_action_frame"],
            "engine_action_frame": v2_metadata["engine_action_frame"],
        },
        "seconds": round(time.perf_counter() - started, 3),
    }


# ---------------------------------------------------------------------------
# 6. Hidden-information audit under v2
# ---------------------------------------------------------------------------


@dataclass
class HiddenCounters:
    trials: int = 0
    skipped_invalid: int = 0
    skipped_unchanged: int = 0
    skipped_too_few_hidden: int = 0
    observation_mismatch: int = 0
    normalized_legal_mismatch: int = 0
    normalized_mask_mismatch: int = 0
    policy_logits_mismatch: int = 0
    value_logits_mismatch: int = 0
    belief_logits_mismatch: int = 0
    model_action_mismatch: int = 0
    absolute_action_mismatch: int = 0
    diagnostics_mismatch: int = 0
    positive_control_belief_failures: int = 0
    positive_control_type_failures: int = 0


def audit_hidden_information(model, policy, target_trials: int, seed: int = 90210) -> dict:
    """At least `target_trials` valid permutation trials, expecting zero drift.

    Extends the Phase 5 audit with the two products v2 adds: the *normalized*
    legality set and mask, and the model-frame action recorded in diagnostics.
    A permutation of hidden identities must move none of them.
    """
    started = time.perf_counter()
    counters = HiddenCounters()
    rng = random.Random(seed)

    plies = (15, 30, 55, 85, 125, 180, 240)
    ply_histogram = {ply: 0 for ply in plies}
    hidden_piece_counts: list[int] = []
    source_games = 0
    game_seed = 0

    while counters.trials < target_trials:
        ply = plies[source_games % len(plies)]
        state = play_random_game_to_ply(game_seed, ply, rules=TRAINING_RULES)
        game_seed += 1
        if state.terminal or state.total_moves != ply:
            continue
        source_games += 1
        observer = state.acting_player

        observation = build_observation(state, observer)
        actions = legal_actions(state)
        model_actions = absolute_legal_actions_to_model(actions, observer)
        model_mask = absolute_legal_mask_to_model(legal_action_mask(state, actions), observer)
        with torch.no_grad():
            outputs = model(tokenize_numpy_observation(observation))
        decision = policy.decide_checked(
            build_policy_input(
                state,
                policy=policy.ref,
                policy_seed=SEEDS["evaluation_policy_seed"],
                requirements=policy.requirements,
                legal=actions,
            )
        )
        labels, mask = dense_belief_target(state, observer)
        types = _hidden_types(state, observer)

        for _ in range(20):
            if counters.trials >= target_trials:
                break
            twin, info = permute_hidden_identities(state, observer, rng)
            if info["hidden_pieces"] < 2:
                counters.skipped_too_few_hidden += 1
                break
            if not info["valid"]:
                counters.skipped_invalid += 1
                continue
            if not info["changed"]:
                counters.skipped_unchanged += 1
                continue

            counters.trials += 1
            ply_histogram[ply] += 1
            hidden_piece_counts.append(int(info["hidden_pieces"]))

            twin_observation = build_observation(twin, observer)
            if not np.array_equal(observation, twin_observation):
                counters.observation_mismatch += 1

            twin_actions = legal_actions(twin)
            twin_model_actions = absolute_legal_actions_to_model(twin_actions, observer)
            if twin_model_actions != model_actions:
                counters.normalized_legal_mismatch += 1
            if not np.array_equal(
                absolute_legal_mask_to_model(
                    legal_action_mask(twin, twin_actions), observer
                ),
                model_mask,
            ):
                counters.normalized_mask_mismatch += 1

            with torch.no_grad():
                twin_outputs = model(tokenize_numpy_observation(twin_observation))
            if not torch.equal(outputs.policy_logits, twin_outputs.policy_logits):
                counters.policy_logits_mismatch += 1
            if not torch.equal(outputs.value_logits, twin_outputs.value_logits):
                counters.value_logits_mismatch += 1
            if not torch.equal(outputs.belief_logits, twin_outputs.belief_logits):
                counters.belief_logits_mismatch += 1

            twin_decision = policy.decide_checked(
                build_policy_input(
                    twin,
                    policy=policy.ref,
                    policy_seed=SEEDS["evaluation_policy_seed"],
                    requirements=policy.requirements,
                    legal=twin_actions,
                )
            )
            if twin_decision.selected_action_id != decision.selected_action_id:
                counters.absolute_action_mismatch += 1
            if twin_decision.diagnostics.get("model_action_id") != decision.diagnostics.get(
                "model_action_id"
            ):
                counters.model_action_mismatch += 1
            if dict(twin_decision.diagnostics) != dict(decision.diagnostics):
                counters.diagnostics_mismatch += 1

            # Positive controls: the privileged products *must* have moved, or
            # the permutation did nothing and the trial proves nothing.
            twin_labels, twin_mask = dense_belief_target(twin, observer)
            if np.array_equal(labels, twin_labels) or not np.array_equal(mask, twin_mask):
                counters.positive_control_belief_failures += 1
            if _hidden_types(twin, observer) == types:
                counters.positive_control_type_failures += 1

    mismatch_fields = (
        "observation_mismatch",
        "normalized_legal_mismatch",
        "normalized_mask_mismatch",
        "policy_logits_mismatch",
        "value_logits_mismatch",
        "belief_logits_mismatch",
        "model_action_mismatch",
        "absolute_action_mismatch",
        "diagnostics_mismatch",
    )
    mismatches = {name: getattr(counters, name) for name in mismatch_fields}
    return {
        "schema_version": SCHEMA_VERSION,
        "trials": counters.trials,
        "target_trials": target_trials,
        "source_positions": source_games,
        "plies_sampled": list(plies),
        "ply_histogram": {str(key): value for key, value in ply_histogram.items()},
        "hidden_pieces_permuted": {
            "min": min(hidden_piece_counts) if hidden_piece_counts else 0,
            "max": max(hidden_piece_counts) if hidden_piece_counts else 0,
            "mean": (
                round(statistics.fmean(hidden_piece_counts), 3) if hidden_piece_counts else 0.0
            ),
        },
        "skipped": {
            "invalid": counters.skipped_invalid,
            "unchanged": counters.skipped_unchanged,
            "too_few_hidden_pieces": counters.skipped_too_few_hidden,
        },
        "mismatches": mismatches,
        "total_mismatches": sum(mismatches.values()),
        "positive_control_failures": (
            counters.positive_control_belief_failures + counters.positive_control_type_failures
        ),
        "positive_controls": {
            "belief_targets_changed_failures": counters.positive_control_belief_failures,
            "hidden_types_changed_failures": counters.positive_control_type_failures,
        },
        "seconds": round(time.perf_counter() - started, 3),
    }


def _hidden_types(state, observer: int) -> list[int]:
    return [
        record.true_type
        for record in state.pieces
        if record.owner != observer and record.alive and not record.known_to(observer)
    ]


# ---------------------------------------------------------------------------
# 7. Phase 4 integration regression
# ---------------------------------------------------------------------------


def run_evaluation_regression(checkpoint_path: Path, pair_ids, quick: bool) -> tuple[dict, list]:
    """A defensible neural evaluation subset in both modes and both colours."""
    started = time.perf_counter()
    bank = SetupBank.generate(max(pair_ids) + 1)

    all_rows = []
    per_mode = {}
    for policy_class, mode in (
        (GreedyNeuralPolicy, DECISION_MODE_GREEDY),
        (SeededCategoricalNeuralPolicy, DECISION_MODE_CATEGORICAL),
    ):
        policy = policy_class.from_checkpoint(checkpoint_path)
        policies = {policy.ref.token: policy}
        mode_rows = []
        per_opponent = {}

        for opponent_id in CORE_BASELINES:
            units = build_paired_schedule(policy.ref, policy_ref(opponent_id), pair_ids)
            specs = [spec for unit in units for spec in unit.matches]
            summary = run_schedule(specs, bank, policies=policies, worker_count=1)
            rows = list(summary.results)
            mode_rows.extend(rows)
            stats = summarize_matchup(rows, resamples=1000, seed=SEEDS["bootstrap"])
            per_opponent[opponent_id] = {
                "matches": len(rows),
                "paired_units": summary.paired_units_run,
                "wins": stats.counts.wins,
                "draws": stats.counts.draws,
                "losses": stats.counts.losses,
                "effective_win_rate": round(stats.effective_win_rate, 4),
                "policy_errors": summary.policy_errors,
                "illegal_actions": summary.illegal_policy_actions,
                "mean_plies": stats.plies["mean"],
                "results_digest": summary.results_digest,
            }

        # Determinism: rerun the entire mode and compare row by row.
        rerun_specs = [
            spec
            for opponent_id in CORE_BASELINES
            for unit in build_paired_schedule(policy.ref, policy_ref(opponent_id), pair_ids)
            for spec in unit.matches
        ]
        rerun = run_schedule(rerun_specs, bank, policies=policies, worker_count=1)
        rerun_problems = compare_results(mode_rows, rerun.results)

        # Replay: the stored absolute action history must still reconstruct.
        sample = mode_rows if quick else mode_rows[:: max(1, len(mode_rows) // 32)]
        replay_problems = []
        reproduction_problems = []
        for row in sample:
            replay_problems.extend(replay_stored_match(row))
            reproduction_problems.extend(
                compare_results([row], [reproduce_match(row, policies=policies)])
            )

        by_unit: dict[str, list] = {}
        for row in mode_rows:
            by_unit.setdefault(row.paired_unit_id, []).append(row)
        colour_swap_ok = all(
            len(rows) == 2
            and {row.candidate_color for row in rows} == {RED, BLUE}
            and rows[0].red_setup == rows[1].red_setup
            and rows[0].blue_setup == rows[1].blue_setup
            for rows in by_unit.values()
        )

        per_mode[mode] = {
            "policy_id": policy.policy_id,
            "policy_version": policy.policy_version,
            "stochastic": policy.stochastic,
            "matches": len(mode_rows),
            "paired_units": len(by_unit),
            "colours_played": sorted({int(row.candidate_color) for row in mode_rows}),
            "colour_swap_correct": colour_swap_ok,
            "per_opponent": per_opponent,
            "policy_errors": sum(1 for row in mode_rows if row.errored),
            "illegal_actions": sum(
                1 for row in mode_rows if row.policy_error_category == "illegal_action"
            ),
            "rerun_identical": rerun_problems == [],
            "rerun_differences": rerun_problems,
            "results_digest": results_digest(mode_rows),
            "rerun_digest": results_digest(rerun.results),
            "replayed_rows": len(sample),
            "replay_problems": replay_problems,
            "reproduction_problems": reproduction_problems,
            "checkpoint_identity": policy.describe()["checkpoint"],
        }
        all_rows.extend(mode_rows)

    total_plies = sum(row.plies for row in all_rows)
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "setup_pair_ids": len(list(pair_ids)),
            "matches": len(all_rows),
            "plies": total_plies,
            "modes": per_mode,
            "illegal_actions": sum(mode["illegal_actions"] for mode in per_mode.values()),
            "policy_errors": sum(mode["policy_errors"] for mode in per_mode.values()),
            "greedy_reproducible": per_mode[DECISION_MODE_GREEDY]["rerun_identical"],
            "seeded_categorical_reproducible": per_mode[DECISION_MODE_CATEGORICAL][
                "rerun_identical"
            ],
            "replay_problems": sum(
                len(mode["replay_problems"]) for mode in per_mode.values()
            ),
            "reproduction_problems": sum(
                len(mode["reproduction_problems"]) for mode in per_mode.values()
            ),
            "both_colours_played": all(
                mode["colours_played"] == [RED, BLUE] for mode in per_mode.values()
            ),
            "match_semantics_unchanged": all(
                mode["colour_swap_correct"] for mode in per_mode.values()
            ),
            "note": (
                "the network is untrained; win rates are recorded to show the matches "
                "ran clean and carry no information about playing strength"
            ),
            "seconds": round(time.perf_counter() - started, 3),
        },
        all_rows,
    )


# ---------------------------------------------------------------------------
# 8. Test suite
# ---------------------------------------------------------------------------


def run_pytest() -> dict:
    import re
    import xml.etree.ElementTree as ElementTree

    started = time.perf_counter()
    report = DATA_DIRECTORY / ".pytest_junit.xml"
    report.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", f"--junitxml={report}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    tail = process.stdout.strip().splitlines()[-1] if process.stdout.strip() else ""

    def count(pattern: str) -> int:
        match = re.search(rf"(\d+) {pattern}", tail)
        return int(match.group(1)) if match else 0

    per_module: dict[str, dict] = {}
    if report.exists():
        for case in ElementTree.parse(report).getroot().iter("testcase"):
            module = case.get("file") or case.get("classname", "").replace(".", "/") + ".py"
            entry = per_module.setdefault(module, {"passed": 0, "failed": 0, "skipped": 0})
            if case.find("failure") is not None or case.find("error") is not None:
                entry["failed"] += 1
            elif case.find("skipped") is not None:
                entry["skipped"] += 1
            else:
                entry["passed"] += 1
        # The XML is an intermediate, not a deliverable: everything worth keeping
        # is in the per-module breakdown below.
        report.unlink()

    return {
        "command": "python -m pytest -q",
        "exit_code": process.returncode,
        "passed": count("passed"),
        "failed": count("failed"),
        "errors": count("error"),
        "skipped": count("skipped"),
        "summary_line": tail,
        "per_module": per_module,
        "seconds": round(time.perf_counter() - started, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument("--skip-pytest", action="store_true", help="measurements only")
    parser.add_argument(
        "--hidden-trials", type=int, default=10_000, help="model-level permutation trials"
    )
    parser.add_argument(
        "--symmetry-trials", type=int, default=200, help="mirrored position pairs"
    )
    parser.add_argument(
        "--pair-ids",
        type=int,
        default=16,
        help="setup pairs per matchup; 16 gives 256 matches across both modes",
    )
    parser.add_argument("--corpus", type=int, default=16, help="legality corpus positions")
    parser.add_argument(
        "--data-directory", type=Path, default=DATA_DIRECTORY, help="artifact destination"
    )
    return parser.parse_args()


def main() -> int:
    options = parse_arguments()
    if options.quick:
        options.hidden_trials = min(options.hidden_trials, 400)
        options.symmetry_trials = min(options.symmetry_trials, 12)
        options.pair_ids = min(options.pair_ids, 2)
        options.corpus = min(options.corpus, 4)

    started = time.perf_counter()
    data_directory = Path(options.data_directory)
    data_directory.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pair_ids = range(options.pair_ids)

    print("Phase 6 Agent 1 harness - model contract v2")
    print(f"  commit          {git_commit()[:12]}")
    print(f"  contract        {MODEL_CONTRACT_VERSION}")
    print(f"  policy frame    {POLICY_ACTION_FRAME}")
    print(f"  engine frame    {ENGINE_ACTION_FRAME}")
    print()

    prerequisites = verify_prerequisites()
    print(
        f"[1/8] prerequisite: Phase 5 {prerequisites.get('phase_5_status')} "
        f"{prerequisites.get('phase_5_gates')}, accepted={prerequisites['phase_5_accepted']}"
    )
    if not prerequisites["phase_5_accepted"]:
        for problem in prerequisites["problems"]:
            print(f"      BLOCKED: {problem}")
        write_json(
            data_directory / "agent_01_model_contract_v2.json",
            {
                "agent": "agent_01",
                "phase": "phase_6",
                "status": "BLOCKED",
                "schema_version": SCHEMA_VERSION,
                "prerequisite_status": prerequisites,
            },
        )
        return 2

    model = build_integration_model(seed=MODEL_SEED)
    checkpoint_path = save_checkpoint(
        model,
        CHECKPOINT_DIRECTORY / "integration_model_v2.pt",
        training_iteration=0,
        training_step=0,
        training_metrics={"note": "untrained integration fixture, model_contract_v2"},
    )
    print(f"      checkpoint: {checkpoint_path.name}")

    round_trips = audit_action_round_trips()
    print(
        f"[2/8] action frame: {round_trips['forward_cases']:,} forward / "
        f"{round_trips['reverse_cases']:,} reverse cases, "
        f"{round_trips['forward_mismatches'] + round_trips['reverse_mismatches']} mismatches, "
        f"bijection={round_trips['bijection_ok']}"
    )

    corpus = position_corpus(options.corpus)
    legality = audit_legal_products(corpus)
    print(
        f"[3/8] legality: {legality['positions_tested']} positions, "
        f"{legality['action_comparisons']:,} action comparisons, "
        f"{legality['list_mask_mismatches'] + legality['absolute_round_trip_mismatches']} mismatches"
    )

    symmetry = audit_symmetry(model, options.symmetry_trials)
    print(
        f"[4/8] symmetry: {symmetry['trials']} mirrored pairs, "
        f"{symmetry['total_mismatches']} mismatches, "
        f"v1-frame control disagreed {symmetry['v1_absolute_frame_disagreements']} times"
    )

    compatibility = audit_checkpoint_compatibility(model, checkpoint_path)
    print(
        f"[5/8] checkpoints: {compatibility['cases_as_expected']}/{compatibility['cases_total']} "
        f"as expected, v2-refused-under-v1={compatibility['v2_would_be_refused_under_v1']}"
    )

    greedy_policy = GreedyNeuralPolicy.from_checkpoint(checkpoint_path)
    hidden = audit_hidden_information(
        model, greedy_policy, options.hidden_trials, seed=SEEDS["hidden_information"]
    )
    print(
        f"[6/8] hidden information: {hidden['trials']:,} trials, "
        f"{hidden['total_mismatches']} mismatches, "
        f"{hidden['positive_control_failures']} positive-control failures "
        f"({hidden['seconds']:.1f}s)"
    )

    evaluation, rows = run_evaluation_regression(checkpoint_path, pair_ids, options.quick)
    print(
        f"[7/8] evaluation: {evaluation['matches']} matches, {evaluation['plies']:,} plies, "
        f"{evaluation['illegal_actions']} illegal, {evaluation['policy_errors']} policy errors "
        f"({evaluation['seconds']:.1f}s)"
    )

    suite = (
        {"skipped_by_flag": True, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
        if options.skip_pytest
        else run_pytest()
    )
    if not options.skip_pytest:
        print(f"[8/8] suite: {suite['summary_line']}")
    else:
        print("[8/8] suite: skipped by --skip-pytest")

    gates = {
        "phase_5_accepted": prerequisites["phase_5_accepted"],
        "preexisting_suite_green": PREEXISTING_SUITE["failed"] == 0,
        "model_contract_v2_explicit": MODEL_CONTRACT_VERSION == "model_contract_v2",
        "engine_action_semantics_unchanged": (
            ACTION_ENCODING_VERSION == "source_destination_10000_v1"
            and ENGINE_ACTION_FRAME == "absolute_engine_squares"
            and round_trips["action_encoding_preserved"]
        ),
        "absolute_round_trips_clean": (
            round_trips["forward_cases"] == 20_000 and round_trips["forward_mismatches"] == 0
        ),
        "reverse_round_trips_clean": (
            round_trips["reverse_cases"] == 20_000 and round_trips["reverse_mismatches"] == 0
        ),
        "action_frame_is_a_bijection": (
            round_trips["bijection_ok"] and round_trips["collisions"] == 0
        ),
        "pinned_geometry_correct": round_trips["pinned_geometry_mismatches"] == 0,
        "legal_list_and_mask_exact": (
            legality["list_mask_mismatches"] == 0
            and legality["absolute_round_trip_mismatches"] == 0
            and legality["mask_round_trip_mismatches"] == 0
            and legality["both_colours_covered"]
        ),
        "symmetry_regression_passes": (
            symmetry["total_mismatches"] == 0
            and symmetry["trials"] >= (12 if options.quick else 100)
            and symmetry["v1_control_proves_the_migration_changed_something"]
        ),
        "v1_and_v2_checkpoints_fail_loudly": (
            compatibility["rejection_failures"] == 0
            and compatibility["v2_would_be_refused_under_v1"]
            and compatibility["v1_rule_still_accepts_v1"]
            and compatibility["v1_file_unmodified_by_refused_load"]
        ),
        "hidden_information_zero_mismatch": (
            hidden["trials"] >= options.hidden_trials and hidden["total_mismatches"] == 0
        ),
        "positive_controls_succeed": hidden["positive_control_failures"] == 0,
        "greedy_and_seeded_reproduce": (
            evaluation["greedy_reproducible"] and evaluation["seeded_categorical_reproducible"]
        ),
        "evaluation_regression_clean": (
            evaluation["illegal_actions"] == 0
            and evaluation["policy_errors"] == 0
            and evaluation["replay_problems"] == 0
            and evaluation["reproduction_problems"] == 0
            and evaluation["both_colours_played"]
            and evaluation["match_semantics_unchanged"]
        ),
        "full_suite_green": (
            options.skip_pytest is False
            and suite["failed"] == 0
            and suite["errors"] == 0
            and suite["passed"] >= PREEXISTING_SUITE["passed"]
        ),
    }
    status = "PASS" if all(gates.values()) else "FAIL"

    payload = {
        "agent": "agent_01",
        "phase": "phase_6",
        "status": status,
        "schema_version": SCHEMA_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **environment(),
        "quick_mode": options.quick,
        "prerequisite_status": prerequisites,
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "token_square_frame": TOKEN_SQUARE_FRAME,
        "policy_action_frame": POLICY_ACTION_FRAME,
        "engine_action_frame": ENGINE_ACTION_FRAME,
        "action_encoding_version": ACTION_ENCODING_VERSION,
        "contract_summary": contract_summary(),
        "frozen_contracts": {
            "rules_version": RULES_VERSION,
            "reference_engine": IMPLEMENTATION_VERSION,
            "observation_version": OBSERVATION_VERSION,
        },
        "action_round_trip_cases": round_trips["forward_cases"],
        "action_round_trip_mismatches": round_trips["forward_mismatches"],
        "reverse_round_trip_cases": round_trips["reverse_cases"],
        "reverse_round_trip_mismatches": round_trips["reverse_mismatches"],
        "action_frame_audit": round_trips,
        "legal_positions_tested": legality["positions_tested"],
        "legal_action_comparisons": legality["action_comparisons"],
        "legal_mask_mismatches": (
            legality["list_mask_mismatches"]
            + legality["mask_round_trip_mismatches"]
            + legality["absolute_round_trip_mismatches"]
        ),
        "legality_audit": legality,
        "symmetry_trials": symmetry["trials"],
        "symmetry_mismatches": symmetry["total_mismatches"],
        "symmetry_audit": symmetry,
        "hidden_information_trials": hidden["trials"],
        "hidden_information_mismatches": hidden["total_mismatches"],
        "positive_control_failures": hidden["positive_control_failures"],
        "hidden_information_audit": hidden,
        "checkpoint_compatibility_tests": compatibility["cases_total"],
        "checkpoint_rejection_failures": compatibility["rejection_failures"],
        "checkpoint_compatibility": compatibility,
        "evaluation_regression": evaluation,
        "test_total": suite["passed"] + suite["failed"] + suite["skipped"],
        "test_passed": suite["passed"],
        "test_failed": suite["failed"],
        "tests_before": PREEXISTING_SUITE,
        "tests_after": suite,
        "commands": [
            "python -m pytest -q",
            "python scripts/run_phase6_agent01.py",
        ],
        "durations": {
            "action_frame_seconds": round_trips["seconds"],
            "legality_seconds": legality["seconds"],
            "symmetry_seconds": symmetry["seconds"],
            "checkpoint_seconds": compatibility["seconds"],
            "hidden_information_seconds": hidden["seconds"],
            "evaluation_seconds": evaluation["seconds"],
            "suite_seconds": suite.get("seconds", 0),
        },
        "seeds": SEEDS,
        "files_created": FILES_CREATED,
        "files_modified": FILES_MODIFIED,
        "completion_gates": gates,
        "gates_total": len(gates),
        "gates_true": sum(1 for value in gates.values() if value),
        "problems": [name for name, value in gates.items() if not value],
        "total_seconds": round(time.perf_counter() - started, 2),
    }

    write_json(data_directory / "agent_01_model_contract_v2.json", payload)

    print()
    print(f"status                  {status}")
    print(f"gates true              {payload['gates_true']}/{payload['gates_total']}")
    for name, value in gates.items():
        if not value:
            print(f"  FAILED GATE           {name}")
    print(f"total seconds           {payload['total_seconds']}")
    print(f"written                 {data_directory}/agent_01_model_contract_v2.json")
    return 0 if status == "PASS" else 1


FILES_CREATED = [
    "stratego/model/action_frame.py",
    "tests/model/test_action_frame.py",
    "tests/model/test_symmetry.py",
    "scripts/run_phase6_agent01.py",
    "checkpoints/integration_model_v2.pt",
    "reports/phase_6_data/agent_01_model_contract_v2.json",
    "reports/phase_6_implementation_report.md",
]

FILES_MODIFIED = [
    "stratego/model/__init__.py",
    "stratego/model/contract.py",
    "stratego/model/checkpoint.py",
    "stratego/model/policy_adapter.py",
    "tests/model/conftest.py",
    "tests/model/test_contract.py",
    "tests/model/test_checkpoint.py",
    "tests/model/test_evaluation_integration.py",
    "scripts/run_phase5.py",
]


if __name__ == "__main__":
    raise SystemExit(main())
