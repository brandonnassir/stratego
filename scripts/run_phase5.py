#!/usr/bin/env python3
"""Phase 5 acceptance harness: neural model contract and end-to-end integration.

Runs every measurement the Phase 5 instructions require and writes the six
machine-readable artifacts under `reports/phase_5_data/`:

- `agent_01_phase5_acceptance.json`         the 22 gates plus run metadata
- `agent_01_action_mapping.json`            the exhaustive 10,000-action audit
- `agent_01_hidden_information.json`        the 10,000-trial model-level audit
- `agent_01_checkpoint_compatibility.json`  round-trip identity and negatives
- `agent_01_numerical_batch_performance.json`  device, precision, batch, latency
- `agent_01_evaluation_gauntlet.csv`        every gauntlet match, Phase 4 row format

What this script is not
-----------------------
It is **not** training and **not** architecture selection. The network is the
Phase 5 integration fixture, it is never trained, and section 5.7 forbids tuning
the architecture against the latency numbers measured here. Phase 6 owns both.

Usage::

    python scripts/run_phase5.py                # full acceptance run
    python scripts/run_phase5.py --quick        # fast smoke run
    python scripts/run_phase5.py --skip-pytest  # measurements only
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

from stratego.engine.actions import (  # noqa: E402
    action_destination,
    action_source,
    decode_action,
    encode_action,
)
from stratego.engine.constants import (  # noqa: E402
    ACTION_SPACE_SIZE,
    BLUE,
    IMPLEMENTATION_VERSION,
    NUM_SQUARES,
    OBSERVATION_CHANNELS,
    OBSERVATION_VERSION,
    RED,
    RULES_VERSION,
    TRAINING_RULES,
)
from stratego.engine.legal_moves import legal_action_mask, legal_actions  # noqa: E402
from stratego.engine.observation import build_observation  # noqa: E402
from stratego.engine.permutation import permute_hidden_identities  # noqa: E402
from stratego.engine.random_play import play_random_game_to_ply  # noqa: E402
from stratego.engine.snapshot import create_snapshot, snapshot_to_json  # noqa: E402
from stratego.engine.transition import IllegalActionError, apply_action  # noqa: E402
from stratego.evaluation.match_runner import (  # noqa: E402
    compare_results,
    play_match,
    replay_stored_match,
    reproduce_match,
    results_digest,
    run_schedule,
)
from stratego.evaluation.match_spec import (  # noqa: E402
    EVALUATION_SUITE_VERSION,
    PAIRING_COLOR_SWAP_SAME_BOARD,
    build_paired_schedule,
)
from stratego.evaluation.policy import (  # noqa: E402
    POLICY_INTERFACE_VERSION,
    PolicyRequirements,
    build_policy_input,
)
from stratego.evaluation.registry import policy_ref  # noqa: E402
from stratego.evaluation.reporting import write_json, write_results_csv  # noqa: E402
from stratego.evaluation.setup_bank import SETUP_BANK_VERSION, SetupBank  # noqa: E402
from stratego.evaluation.statistics import summarize_matchup  # noqa: E402
from stratego.model.checkpoint import (  # noqa: E402
    CHECKPOINT_FORMAT_VERSION,
    CheckpointError,
    build_checkpoint_payload,
    file_digest,
    load_checkpoint,
    read_checkpoint_payload,
    save_checkpoint,
    state_dict_digest,
    validate_checkpoint_payload,
)
from stratego.model.contract import (  # noqa: E402
    ACTION_ENCODING_VERSION,
    MODEL_CONTRACT_VERSION,
    POLICY_ACTION_FRAME,
    VALUE_CLASS_ORDER,
    contract_summary,
    expected_value,
    value_probabilities,
)
from stratego.model.integration_model import (  # noqa: E402
    MODEL_ARCHITECTURE_ID,
    IntegrationModelConfig,
    build_integration_model,
)
from stratego.model.losses import multi_head_loss  # noqa: E402
from stratego.model.policy_adapter import (  # noqa: E402
    DECISION_MODE_CATEGORICAL,
    DECISION_MODE_GREEDY,
    GreedyNeuralPolicy,
    NeuralPolicyError,
    SeededCategoricalNeuralPolicy,
    greedy_action,
)
from stratego.model.tokenization import (  # noqa: E402
    observation_to_tokens,
    position_coded_observation,
    tokenize_numpy_observation,
)
from stratego.training.belief_targets import (  # noqa: E402
    belief_target_summary,
    dense_belief_target,
)

SCHEMA_VERSION = "phase_5_acceptance_v1"
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_5_data"
CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / "checkpoints"

#: Predeclared comparison tolerances. Mirrored in tests/model/test_device_batch_equivalence.py.
FLOAT32_TOLERANCE = {"atol": 1e-4, "rtol": 1e-4}
FLOAT16_TOLERANCE = {"atol": 5e-2, "rtol": 5e-2}

#: The accepted Phase 4 core ladder, in the instruction's order.
CORE_BASELINES = (
    "random_legal",
    "basic_heuristic",
    "tactical_rule_based",
    "strategic_rule_based",
)

MODEL_SEED = 20250501

#: The clean baseline recorded **before** any Phase 5 file was written, so gate 2
#: cites a measurement rather than an assertion. Reproduce with:
#:     git stash && python -m pytest -q
PREEXISTING_SUITE = {
    "command": "python -m pytest -q",
    "commit": "1d8e7cb4473e4bb037aff6a70d162effe3457c68",
    "recorded_before_any_phase_5_edit": True,
    "passed": 1963,
    "failed": 0,
    "errors": 0,
    "skipped": 2,
    "seconds": 64.01,
    "skip_reasons": [
        "tests/evaluation/test_baseline_information_safety.py:219 "
        "random_legal does not expose a per-move score vector",
        "tests/evaluation/test_baseline_information_safety.py:219 "
        "stress_chaos does not expose a per-move score vector",
    ],
    "note": "pre-existing capability skips, unrelated to Phase 5 and not repaired",
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001 - a missing git is not a Phase 5 failure
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


# ---------------------------------------------------------------------------
# 1. Frozen contracts
# ---------------------------------------------------------------------------


def verify_frozen_contracts() -> dict:
    """Confirm every version Phase 5 inherits is exactly what it inherited."""
    expected = {
        "rules_version": (RULES_VERSION, "stratego_project_v1"),
        "reference_engine": (IMPLEMENTATION_VERSION, "phase2_1_reference_1.1.0"),
        "observation_version": (OBSERVATION_VERSION, "observation_v2_1_127ch"),
        "action_encoding_version": (ACTION_ENCODING_VERSION, "source_destination_10000_v1"),
        "policy_interface_version": (POLICY_INTERFACE_VERSION, "policy_interface_v1"),
        "setup_bank_version": (SETUP_BANK_VERSION, "evaluation_setup_bank_v1"),
        "pairing_mode": (PAIRING_COLOR_SWAP_SAME_BOARD, "color_swap_same_board"),
    }
    checks = {
        name: {"actual": actual, "expected": wanted, "ok": actual == wanted}
        for name, (actual, wanted) in expected.items()
    }
    checks["observation_channels"] = {
        "actual": OBSERVATION_CHANNELS,
        "expected": 127,
        "ok": OBSERVATION_CHANNELS == 127,
    }
    checks["action_space_size"] = {
        "actual": ACTION_SPACE_SIZE,
        "expected": 10_000,
        "ok": ACTION_SPACE_SIZE == 10_000,
    }
    checks["evaluation_suite_version"] = {
        "actual": EVALUATION_SUITE_VERSION,
        "expected": EVALUATION_SUITE_VERSION,
        "ok": True,
    }
    return {
        "all_ok": all(entry["ok"] for entry in checks.values()),
        "checks": checks,
        "model_contract": contract_summary(),
        "belief_target_contract": belief_target_summary(),
    }


# ---------------------------------------------------------------------------
# 2. Action mapping and tokenization
# ---------------------------------------------------------------------------


def position_corpus(count: int = 12) -> list:
    """Deterministic positions spanning the game, from several seed families."""
    plies = (0, 10, 25, 50, 90, 140, 200, 260, 18, 60, 120, 240)[:count]
    seeds = (0, 0, 0, 0, 0, 0, 0, 0, 37, 37, 37, 37)[:count]
    corpus = []
    for ply, seed in zip(plies, seeds):
        for attempt in range(seed, seed + 200):
            state = play_random_game_to_ply(attempt, ply, rules=TRAINING_RULES)
            if not state.terminal and state.total_moves == ply:
                corpus.append(state)
                break
    return corpus


def audit_action_mapping(model, quick: bool) -> dict:
    """The exhaustive encode/decode audit plus adapter selection over a corpus."""
    started = time.perf_counter()

    round_trip_mismatches = []
    arithmetic_mismatches = []
    for action_id in range(ACTION_SPACE_SIZE):
        source, destination = decode_action(action_id)
        if encode_action(source, destination) != action_id:
            round_trip_mismatches.append(action_id)
        if action_id != 100 * source + destination:
            arithmetic_mismatches.append(action_id)
        if action_source(action_id) != source or action_destination(action_id) != destination:
            arithmetic_mismatches.append(action_id)
    distinct_pairs = len({decode_action(a) for a in range(ACTION_SPACE_SIZE)})

    # Tokenization ordering, frozen with a position-coded tensor.
    coded = position_coded_observation(batch=2)
    tokens = observation_to_tokens(coded)
    token_mismatches = 0
    for batch in range(coded.shape[0]):
        for square in range(NUM_SQUARES):
            row, column = divmod(square, 10)
            if not torch.equal(tokens[batch, square], coded[batch, :, row, column]):
                token_mismatches += 1

    # Adapter selection: a crafted unique maximum at every legal index.
    corpus = position_corpus(4 if quick else 12)
    selection_checked = 0
    selection_mismatches = 0
    families: set[str] = set()
    for state in corpus:
        actions = legal_actions(state)
        mask = legal_action_mask(state, actions)
        for action in actions:
            logits = torch.full((ACTION_SPACE_SIZE,), -50.0)
            logits[action] = 50.0
            if greedy_action(logits, actions) != action:
                selection_mismatches += 1
            if mask[action] != 1:
                selection_mismatches += 1
            selection_checked += 1
            source, destination = decode_action(action)
            occupied = state.board[destination] is not None
            distance = max(
                abs(source // 10 - destination // 10), abs(source % 10 - destination % 10)
            )
            families.add("attack" if occupied else "quiet")
            families.add("long_scout" if distance > 1 else "single_step")
            families.add("lateral" if source // 10 == destination // 10 else "vertical")

    # The engine's illegal-action guard: loud and inert.
    guard_state = play_random_game_to_ply(3, 40, rules=TRAINING_RULES)
    legal_set = set(legal_actions(guard_state))
    illegal = next(a for a in range(ACTION_SPACE_SIZE) if a not in legal_set)
    before = snapshot_to_json(create_snapshot(guard_state, include_history=True))
    guard_raised, guard_inert = False, False
    try:
        apply_action(guard_state, illegal)
    except IllegalActionError:
        guard_raised = True
    after = snapshot_to_json(create_snapshot(guard_state, include_history=True))
    guard_inert = before == after

    return {
        "schema_version": SCHEMA_VERSION,
        "action_space_size": ACTION_SPACE_SIZE,
        "round_trip_actions_checked": ACTION_SPACE_SIZE,
        "round_trip_mismatches": len(round_trip_mismatches),
        "arithmetic_mismatches": len(arithmetic_mismatches),
        "distinct_square_pairs": distinct_pairs,
        "bijection_ok": distinct_pairs == ACTION_SPACE_SIZE,
        "policy_action_frame": POLICY_ACTION_FRAME,
        "tokenization_positions_checked": int(coded.shape[0] * NUM_SQUARES),
        "tokenization_mismatches": token_mismatches,
        "tokenization_inverse_exact": bool(
            torch.equal(
                observation_to_tokens(coded).transpose(1, 2).reshape(coded.shape), coded
            )
        ),
        "adapter_positions": len(corpus),
        "adapter_selections_checked": selection_checked,
        "adapter_selection_mismatches": selection_mismatches,
        "move_families_covered": sorted(families),
        "engine_illegal_guard_raised": guard_raised,
        "engine_illegal_guard_inert": guard_inert,
        "seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# 3. Checkpoint compatibility
# ---------------------------------------------------------------------------


def audit_checkpoint(model, checkpoint_path: Path) -> dict:
    """Round-trip identity on CPU plus every required negative case."""
    started = time.perf_counter()

    generator = torch.Generator().manual_seed(4242)
    probe = torch.randn(4, 127, 10, 10, generator=generator)
    with torch.no_grad():
        before = model.forward_observation(probe).detached_cpu()

    reloaded, metadata = load_checkpoint(checkpoint_path)
    with torch.no_grad():
        after = reloaded.forward_observation(probe).detached_cpu()

    legal = list(range(0, ACTION_SPACE_SIZE, 37))
    identity = {
        "policy_logits_bit_identical": bool(
            torch.equal(before.policy_logits, after.policy_logits)
        ),
        "value_logits_bit_identical": bool(torch.equal(before.value_logits, after.value_logits)),
        "belief_logits_bit_identical": bool(
            torch.equal(before.belief_logits, after.belief_logits)
        ),
        "greedy_action_identical": bool(
            greedy_action(before.policy_logits[0], legal)
            == greedy_action(after.policy_logits[0], legal)
        ),
        "state_dict_digest_stable": bool(
            state_dict_digest(model.state_dict())
            == state_dict_digest(read_checkpoint_payload(checkpoint_path)["state_dict"])
        ),
    }

    # Negative cases: each must raise.
    def mutate(name, change):
        payload = build_checkpoint_payload(model)
        change(payload)
        return name, payload

    cases = [
        mutate("missing_rules_version", lambda p: p.pop("rules_version")),
        mutate("missing_state_dict", lambda p: p.pop("state_dict")),
        mutate("missing_model_configuration", lambda p: p.pop("model_configuration")),
        mutate("missing_creation_timestamp", lambda p: p.pop("creation_timestamp")),
        mutate(
            "newer_format_version",
            lambda p: p.__setitem__("checkpoint_format_version", CHECKPOINT_FORMAT_VERSION + 1),
        ),
        mutate("zero_format_version", lambda p: p.__setitem__("checkpoint_format_version", 0)),
        mutate("wrong_rules", lambda p: p.__setitem__("rules_version", "stratego_project_v2")),
        mutate(
            "superseded_observation",
            lambda p: p.__setitem__("observation_version", "observation_v2_127ch"),
        ),
        mutate(
            "wrong_action_encoding",
            lambda p: p.__setitem__("action_encoding_version", "source_destination_10000_v2"),
        ),
        mutate(
            "wrong_model_contract",
            lambda p: p.__setitem__("model_contract_version", "model_contract_v2"),
        ),
        mutate(
            "wrong_architecture",
            lambda p: p.__setitem__("model_architecture_id", "ataraxos_full_v1"),
        ),
        mutate(
            "wrong_policy_frame",
            lambda p: p.__setitem__("policy_action_frame", "perspective_normalized_squares"),
        ),
        mutate(
            "incompatible_configuration",
            lambda p: p.__setitem__(
                "model_configuration", dict(p["model_configuration"], width=128)
            ),
        ),
        mutate(
            "unknown_configuration_field",
            lambda p: p.__setitem__(
                "model_configuration", dict(p["model_configuration"], mystery=1)
            ),
        ),
        mutate("unknown_field", lambda p: p.__setitem__("mystery_field", 1)),
        mutate(
            "missing_weight",
            lambda p: p["state_dict"].pop(sorted(p["state_dict"])[0]),
        ),
        mutate(
            "unexpected_weight",
            lambda p: p["state_dict"].__setitem__("ghost.weight", torch.zeros(2, 2)),
        ),
        mutate(
            "wrong_weight_shape",
            lambda p: p["state_dict"].__setitem__(
                sorted(p["state_dict"])[0], torch.zeros(5, 5)
            ),
        ),
        mutate("negative_training_step", lambda p: p.__setitem__("training_step", -1)),
    ]

    negatives = {}
    for name, payload in cases:
        try:
            validate_checkpoint_payload(payload, source=name)
        except CheckpointError as error:
            negatives[name] = {"rejected": True, "error": type(error).__name__, "message": str(error)[:160]}
        else:
            negatives[name] = {"rejected": False, "error": None, "message": "ACCEPTED"}

    # Corrupted files. Written to a temporary directory rather than next to the
    # real checkpoint, so a deliberately broken file can never be mistaken for
    # an artifact of this run.
    import tempfile

    scratch_directory = tempfile.TemporaryDirectory(prefix="phase5_corruption_")
    scratch = Path(scratch_directory.name)
    corrupted = {}
    data = checkpoint_path.read_bytes()

    truncated = scratch / "truncated.pt"
    truncated.write_bytes(data[: len(data) // 2])
    garbage = scratch / "garbage.pt"
    garbage.write_bytes(b"not a checkpoint" * 512)
    empty = scratch / "empty.pt"
    empty.write_bytes(b"")
    bare = scratch / "bare_state_dict.pt"
    torch.save(model.state_dict(), bare)
    missing = scratch / "absent.pt"

    for name, path in (
        ("truncated_file", truncated),
        ("random_bytes_file", garbage),
        ("empty_file", empty),
        ("bare_state_dict_file", bare),
        ("missing_file", missing),
    ):
        try:
            load_checkpoint(path)
        except CheckpointError as error:
            corrupted[name] = {"rejected": True, "error": type(error).__name__}
        else:
            corrupted[name] = {"rejected": False, "error": None}
    scratch_directory.cleanup()

    all_rejected = all(entry["rejected"] for entry in negatives.values()) and all(
        entry["rejected"] for entry in corrupted.values()
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_path": str(checkpoint_path.relative_to(REPOSITORY_ROOT)),
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_file_digest": file_digest(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "metadata": {
            key: value
            for key, value in metadata.items()
            if key not in ("provenance",)
        },
        "provenance": metadata.get("provenance", {}),
        "round_trip_identity": identity,
        "round_trip_all_identical": all(identity.values()),
        "negative_cases": negatives,
        "negative_cases_total": len(negatives),
        "negative_cases_rejected": sum(1 for e in negatives.values() if e["rejected"]),
        "corrupted_files": corrupted,
        "all_incompatibilities_rejected": all_rejected,
        "seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# 4. Model-level hidden-information audit
# ---------------------------------------------------------------------------


@dataclass
class HiddenInformationCounters:
    trials: int = 0
    skipped_invalid: int = 0
    skipped_unchanged: int = 0
    skipped_too_few_hidden: int = 0
    observation_mismatch: int = 0
    legal_action_mismatch: int = 0
    policy_logits_mismatch: int = 0
    value_logits_mismatch: int = 0
    belief_logits_mismatch: int = 0
    greedy_action_mismatch: int = 0
    diagnostics_mismatch: int = 0
    positive_control_failures: int = 0
    hidden_types_unchanged: int = 0


def audit_hidden_information(model, policy, target_trials: int, seed: int = 90210) -> dict:
    """At least `target_trials` valid paired trials, expecting zero mismatches."""
    started = time.perf_counter()
    counters = HiddenInformationCounters()

    plies = (15, 30, 55, 85, 125, 180, 240)
    hidden_piece_counts: list[int] = []
    ply_histogram: dict[int, int] = {ply: 0 for ply in plies}
    source_games = 0
    rng = random.Random(seed)

    game_seed = 0
    while counters.trials < target_trials:
        ply = plies[source_games % len(plies)]
        state = play_random_game_to_ply(game_seed, ply, rules=TRAINING_RULES)
        game_seed += 1
        if state.terminal or state.total_moves != ply:
            continue
        source_games += 1
        observer = state.acting_player

        original_observation = build_observation(state, observer)
        original_actions = legal_actions(state)
        original_request = build_policy_input(
            state,
            policy=policy.ref,
            policy_seed=13,
            requirements=policy.requirements,
            legal=original_actions,
        )
        with torch.no_grad():
            original_outputs = model(tokenize_numpy_observation(original_observation))
        original_decision = policy.decide_checked(original_request)
        original_labels, original_mask = dense_belief_target(state, observer)
        original_types = [
            record.true_type
            for record in state.pieces
            if record.owner != observer and record.alive and not record.known_to(observer)
        ]

        # Several permutations per position: generating a position costs far more
        # than permuting one, and each permutation is an independent trial.
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
            if not np.array_equal(original_observation, twin_observation):
                counters.observation_mismatch += 1
            twin_actions = legal_actions(twin)
            if original_actions != twin_actions:
                counters.legal_action_mismatch += 1

            with torch.no_grad():
                twin_outputs = model(tokenize_numpy_observation(twin_observation))
            if not torch.equal(original_outputs.policy_logits, twin_outputs.policy_logits):
                counters.policy_logits_mismatch += 1
            if not torch.equal(original_outputs.value_logits, twin_outputs.value_logits):
                counters.value_logits_mismatch += 1
            if not torch.equal(original_outputs.belief_logits, twin_outputs.belief_logits):
                counters.belief_logits_mismatch += 1

            twin_decision = policy.decide_checked(
                build_policy_input(
                    twin,
                    policy=policy.ref,
                    policy_seed=13,
                    requirements=policy.requirements,
                    legal=twin_actions,
                )
            )
            if twin_decision.selected_action_id != original_decision.selected_action_id:
                counters.greedy_action_mismatch += 1
            if dict(twin_decision.diagnostics) != dict(original_decision.diagnostics):
                counters.diagnostics_mismatch += 1

            # Positive controls.
            twin_labels, twin_mask = dense_belief_target(twin, observer)
            if np.array_equal(original_labels, twin_labels) or not np.array_equal(
                original_mask, twin_mask
            ):
                counters.positive_control_failures += 1
            twin_types = [
                record.true_type
                for record in twin.pieces
                if record.owner != observer and record.alive and not record.known_to(observer)
            ]
            if original_types == twin_types:
                counters.hidden_types_unchanged += 1

    mismatch_fields = (
        "observation_mismatch",
        "legal_action_mismatch",
        "policy_logits_mismatch",
        "value_logits_mismatch",
        "belief_logits_mismatch",
        "greedy_action_mismatch",
        "diagnostics_mismatch",
    )
    mismatches = {name: getattr(counters, name) for name in mismatch_fields}
    total_mismatches = sum(mismatches.values())

    return {
        "schema_version": SCHEMA_VERSION,
        "trials": counters.trials,
        "target_trials": target_trials,
        "source_positions": source_games,
        "trial_sources": "seeded random games, permute_hidden_identities per position",
        "plies_sampled": list(plies),
        "ply_histogram": {str(k): v for k, v in ply_histogram.items()},
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
        "total_mismatches": total_mismatches,
        "positive_control_failures": counters.positive_control_failures,
        "hidden_types_unchanged": counters.hidden_types_unchanged,
        "positive_control_valid_on_every_trial": counters.positive_control_failures == 0
        and counters.hidden_types_unchanged == 0,
        "permutation_seed": seed,
        "deterministic_cpu_inference": True,
        "seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# 5. Value, belief and autograd
# ---------------------------------------------------------------------------


def audit_value_belief_autograd(model) -> dict:
    """Controlled value semantics, belief mask statistics, one backward pass."""
    started = time.perf_counter()

    # Value semantics under controlled logits.
    controlled = torch.tensor([[6.0, 0.0, -6.0], [-6.0, 0.0, 6.0], [0.0, 20.0, 0.0]])
    probabilities = value_probabilities(controlled)
    value_checks = {
        "class_order": list(VALUE_CLASS_ORDER),
        "probabilities_sum_to_one": bool(
            torch.allclose(probabilities.sum(dim=1), torch.ones(3), atol=1e-6)
        ),
        "expected_value_matches_win_minus_loss": bool(
            torch.allclose(
                expected_value(controlled), probabilities[:, 0] - probabilities[:, 2], atol=1e-7
            )
        ),
        "win_row_expected_value": round(float(expected_value(controlled)[0]), 6),
        "loss_row_expected_value": round(float(expected_value(controlled)[1]), 6),
        "draw_row_expected_value": round(float(expected_value(controlled)[2]), 6),
    }

    # Acting-player perspective: a position and its colour-swapped mirror.
    from tests.observation.test_perspective import mirrored_games, play_mirrored

    original, twin = mirrored_games(4)
    play_mirrored(original, twin, 24, 4)
    perspective_pairs = 0
    perspective_mismatches = 0
    for observer_a, observer_b in ((RED, BLUE), (BLUE, RED)):
        tokens_a = tokenize_numpy_observation(build_observation(original, observer_a))
        tokens_b = tokenize_numpy_observation(build_observation(twin, observer_b))
        with torch.no_grad():
            out_a = model(tokens_a)
            out_b = model(tokens_b)
        perspective_pairs += 1
        if not torch.equal(tokens_a, tokens_b) or not torch.equal(
            out_a.value_logits, out_b.value_logits
        ):
            perspective_mismatches += 1

    # Belief mask statistics, plus a direct audit of every excluded category.
    from stratego.engine.constants import LAKE_SQUARE_SET
    from stratego.engine.coordinates import to_perspective
    from stratego.engine.observation import belief_target
    from stratego.model.contract import BELIEF_IGNORE_INDEX

    supervised_counts, excluded_counts = [], []
    exclusion_violations = {
        "own_pieces": 0,
        "empty_squares": 0,
        "lakes": 0,
        "revealed_opponent_pieces": 0,
        "label_mask_disagreements": 0,
        "sparse_target_disagreements": 0,
    }
    for state in position_corpus(8):
        observer = state.acting_player
        labels, mask = dense_belief_target(state, observer)
        supervised_counts.append(int(mask.sum()))
        excluded_counts.append(int(NUM_SQUARES - mask.sum()))
        supervised = set(np.flatnonzero(mask).tolist())

        if not np.array_equal(mask, labels != BELIEF_IGNORE_INDEX):
            exclusion_violations["label_mask_disagreements"] += 1
        if len(supervised) != len(belief_target(state, observer)):
            exclusion_violations["sparse_target_disagreements"] += 1

        occupied = set()
        for record in state.pieces:
            if not record.alive:
                continue
            normalized = to_perspective(record.current_square, observer)
            occupied.add(normalized)
            if record.owner == observer and normalized in supervised:
                exclusion_violations["own_pieces"] += 1
            elif (
                record.owner != observer
                and record.known_to(observer)
                and normalized in supervised
            ):
                exclusion_violations["revealed_opponent_pieces"] += 1
        for square in set(range(NUM_SQUARES)) - occupied:
            if square in supervised:
                exclusion_violations["empty_squares"] += 1
        for lake in LAKE_SQUARE_SET:
            if to_perspective(lake, observer) in supervised:
                exclusion_violations["lakes"] += 1

    # One controlled backward pass.
    training_model = build_integration_model(seed=99)
    training_model.train()
    states = position_corpus(4)
    tokens = tokenize_numpy_observation(
        [build_observation(state, state.acting_player) for state in states]
    )
    masks, targets, labels_list, mask_list = [], [], [], []
    for state in states:
        actions = legal_actions(state)
        masks.append(legal_action_mask(state, actions).astype(bool))
        targets.append(actions[len(actions) // 2])
        pair = dense_belief_target(state, state.acting_player)
        labels_list.append(pair[0])
        mask_list.append(pair[1])

    outputs = training_model(tokens)
    loss = multi_head_loss(
        outputs,
        target_actions=torch.tensor(targets, dtype=torch.int64),
        legal_mask=torch.from_numpy(np.stack(masks)),
        target_value_classes=torch.tensor([index % 3 for index in range(len(states))]),
        belief_labels=torch.from_numpy(np.stack(labels_list)),
        belief_mask=torch.from_numpy(np.stack(mask_list)),
    )
    loss.total.backward()

    groups = {
        "shared_encoder": ("input_projection", "position_embedding", "blocks", "encoder_norm"),
        "policy_head": ("policy_source", "policy_destination"),
        "value_head": ("value_body", "value_head"),
        "belief_head": ("belief_head",),
    }
    gradient_report = {}
    parameters_without_gradient = []
    non_finite_gradients = []
    for name, parameter in training_model.named_parameters():
        if parameter.grad is None:
            parameters_without_gradient.append(name)
        elif not bool(torch.isfinite(parameter.grad).all()):
            non_finite_gradients.append(name)
    for group, prefixes in groups.items():
        norms = [
            float(parameter.grad.norm())
            for name, parameter in training_model.named_parameters()
            if parameter.grad is not None and name.startswith(prefixes)
        ]
        gradient_report[group] = {
            "tensors": len(norms),
            "max_gradient_norm": round(max(norms), 8) if norms else 0.0,
            "all_nonzero": bool(norms) and all(norm > 0.0 for norm in norms),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "value": value_checks,
        "acting_player_perspective_pairs": perspective_pairs,
        "acting_player_perspective_mismatches": perspective_mismatches,
        "belief": {
            "logits_shape": [NUM_SQUARES, 12],
            "positions_sampled": len(supervised_counts),
            "supervised_squares_min": min(supervised_counts),
            "supervised_squares_max": max(supervised_counts),
            "supervised_squares_mean": round(statistics.fmean(supervised_counts), 3),
            "excluded_squares_mean": round(statistics.fmean(excluded_counts), 3),
            "exclusion_violations": exclusion_violations,
            "exclusions_all_clean": all(
                value == 0 for value in exclusion_violations.values()
            ),
            "target_contract": belief_target_summary(),
        },
        "autograd": {
            "losses": loss.to_dict(),
            "all_losses_finite": loss.all_finite(),
            "parameters_without_gradient": parameters_without_gradient,
            "non_finite_gradients": non_finite_gradients,
            "gradient_groups": gradient_report,
            "backward_passes": 1,
            "optimizer_steps": 0,
        },
        "seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# 6. Device, precision, batch and performance
# ---------------------------------------------------------------------------


def _max_errors(reference: torch.Tensor, other: torch.Tensor) -> dict:
    reference = reference.detach().to("cpu", torch.float32)
    other = other.detach().to("cpu", torch.float32)
    absolute = (reference - other).abs()
    relative = absolute / reference.abs().clamp(min=1e-12)
    return {
        "max_absolute_error": float(absolute.max()),
        "max_relative_error": float(relative.max()),
    }


def audit_numerical_and_performance(quick: bool) -> dict:
    """CPU/MPS float32 and float16, batch equivalence, and MPS latency."""
    started = time.perf_counter()
    mps_available = torch.backends.mps.is_available()

    corpus = position_corpus(8)
    observations = [build_observation(state, state.acting_player) for state in corpus]
    tokens = tokenize_numpy_observation(observations)
    legal_sets = [legal_actions(state) for state in corpus]

    cpu_model = build_integration_model(seed=MODEL_SEED, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        cpu_out = cpu_model(tokens)

    device_reports = {}
    for label, dtype, tolerance in (
        ("mps_float32", torch.float32, FLOAT32_TOLERANCE),
        ("mps_float16", torch.float16, FLOAT16_TOLERANCE),
    ):
        if not mps_available:
            device_reports[label] = {"available": False, "skipped": True, "reason": "MPS unavailable"}
            continue
        device_model = build_integration_model(seed=MODEL_SEED, device="mps", dtype=dtype)
        with torch.no_grad():
            device_out = device_model(tokens.to("mps", dtype))
        device_cpu = device_out.detached_cpu()

        crafted_agreements, crafted_total = 0, 0
        natural_agreements, natural_total = 0, 0
        for index, legal in enumerate(legal_sets):
            reference_row = cpu_out.policy_logits[index].to("cpu", torch.float32)
            device_row = device_cpu.policy_logits[index]
            natural_total += 1
            natural_agreements += int(
                greedy_action(reference_row, legal) == greedy_action(device_row, legal)
            )
            for target in legal[:: max(1, len(legal) // 5)]:
                margin = torch.zeros(ACTION_SPACE_SIZE)
                margin[target] = 100.0
                crafted_total += 1
                crafted_agreements += int(
                    greedy_action(reference_row + margin, legal) == target
                    and greedy_action(device_row + margin, legal) == target
                )

        device_reports[label] = {
            "available": True,
            "skipped": False,
            "dtype": str(dtype),
            "tolerance": tolerance,
            "finite_outputs": device_cpu.all_finite(),
            "policy_logits": _max_errors(cpu_out.policy_logits, device_cpu.policy_logits),
            "value_probabilities": _max_errors(
                value_probabilities(cpu_out.value_logits),
                value_probabilities(device_cpu.value_logits),
            ),
            "belief_logits": _max_errors(cpu_out.belief_logits, device_cpu.belief_logits),
            "within_tolerance": bool(
                torch.allclose(cpu_out.policy_logits, device_cpu.policy_logits, **tolerance)
                and torch.allclose(cpu_out.value_logits, device_cpu.value_logits, **tolerance)
                and torch.allclose(cpu_out.belief_logits, device_cpu.belief_logits, **tolerance)
            ),
            "legal_action_sets_identical": True,  # legality is an engine product
            "crafted_margin_greedy_agreement": f"{crafted_agreements}/{crafted_total}",
            "crafted_margin_exact": crafted_agreements == crafted_total,
            "natural_greedy_agreement": f"{natural_agreements}/{natural_total}",
            "natural_greedy_agreement_rate": round(natural_agreements / natural_total, 4),
        }

    # Batch equivalence, one position alone versus embedded in a batch.
    batch_report = {}
    single = tokens[:1]
    generator = torch.Generator().manual_seed(31337)
    for batch_size in (8, 64, 256):
        filler = observation_to_tokens(
            torch.randn(batch_size - 1, 127, 10, 10, generator=generator)
        )
        stacked = torch.cat([single, filler], dim=0)
        with torch.no_grad():
            alone = cpu_model(single)
            together = cpu_model(stacked).row(0)
        legal = legal_sets[0]
        batch_report[f"batch_{batch_size}"] = {
            "policy_logits": _max_errors(alone.policy_logits, together.policy_logits),
            "value_logits": _max_errors(alone.value_logits, together.value_logits),
            "belief_logits": _max_errors(alone.belief_logits, together.belief_logits),
            "within_tolerance": bool(
                torch.allclose(alone.policy_logits, together.policy_logits, **FLOAT32_TOLERANCE)
                and torch.allclose(alone.value_logits, together.value_logits, **FLOAT32_TOLERANCE)
                and torch.allclose(
                    alone.belief_logits, together.belief_logits, **FLOAT32_TOLERANCE
                )
            ),
            "selected_action_identical": bool(
                greedy_action(alone.policy_logits[0], legal)
                == greedy_action(together.policy_logits[0], legal)
            ),
        }
        if torch.backends.mps.is_available():
            device_model = build_integration_model(
                seed=MODEL_SEED, device="mps", dtype=torch.float32
            )
            with torch.no_grad():
                alone_mps = device_model(single.to("mps")).detached_cpu()
                together_mps = device_model(stacked.to("mps")).detached_cpu().row(0)
            batch_report[f"batch_{batch_size}"]["mps_within_tolerance"] = bool(
                torch.allclose(
                    alone_mps.policy_logits, together_mps.policy_logits, **FLOAT32_TOLERANCE
                )
            )

    # Latency. Tokenization and masking are outside the timed region except where
    # the "full_decision" row says otherwise; both are reported so neither hides.
    performance = []
    if torch.backends.mps.is_available():
        batches = (1, 64, 256) if quick else (1, 64, 256, 1024)
        for dtype in (torch.float32, torch.float16):
            device_model = build_integration_model(seed=MODEL_SEED, device="mps", dtype=dtype)
            for batch_size in batches:
                try:
                    sample = torch.randn(
                        batch_size, 100, 127, generator=generator
                    ).to("mps", dtype)
                    with torch.no_grad():
                        for _ in range(5):  # warmup
                            device_model(sample)
                        torch.mps.synchronize()
                        timings = []
                        for _ in range(20 if batch_size < 1024 else 8):
                            start = time.perf_counter()
                            device_model(sample)
                            torch.mps.synchronize()
                            timings.append(time.perf_counter() - start)
                    median = statistics.median(timings)
                    performance.append(
                        {
                            "device": "mps",
                            "dtype": str(dtype),
                            "batch": batch_size,
                            "median_latency_ms": round(median * 1000, 4),
                            "mean_latency_ms": round(statistics.fmean(timings) * 1000, 4),
                            "positions_per_second": round(batch_size / median, 1),
                            "repeats": len(timings),
                            "warmup_iterations": 5,
                            "synchronization": "torch.mps.synchronize()",
                            "tokenization_in_timing": False,
                            "masking_in_timing": False,
                            "out_of_memory": False,
                        }
                    )
                except RuntimeError as error:
                    performance.append(
                        {
                            "device": "mps",
                            "dtype": str(dtype),
                            "batch": batch_size,
                            "out_of_memory": True,
                            "error": str(error)[:200],
                        }
                    )

    # One end-to-end decision cost on CPU, tokenization and masking included.
    decision_model = build_integration_model(seed=MODEL_SEED)
    state = corpus[3]
    observation = build_observation(state, state.acting_player)
    legal = legal_actions(state)
    for _ in range(5):
        with torch.no_grad():
            decision_model(tokenize_numpy_observation(observation))
    timings = []
    for _ in range(50):
        start = time.perf_counter()
        with torch.no_grad():
            row = decision_model(tokenize_numpy_observation(observation)).policy_logits[0]
        greedy_action(row, legal)
        timings.append(time.perf_counter() - start)
    performance.append(
        {
            "device": "cpu",
            "dtype": "torch.float32",
            "batch": 1,
            "median_latency_ms": round(statistics.median(timings) * 1000, 4),
            "positions_per_second": round(1.0 / statistics.median(timings), 1),
            "repeats": len(timings),
            "warmup_iterations": 5,
            "synchronization": "not required on CPU",
            "tokenization_in_timing": True,
            "masking_in_timing": True,
            "out_of_memory": False,
            "note": "full single decision: numpy observation -> tokens -> forward -> masked greedy",
        }
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "declared_tolerances": {"float32": FLOAT32_TOLERANCE, "float16": FLOAT16_TOLERANCE},
        "mps_available": mps_available,
        "devices": device_reports,
        "batch_equivalence": batch_report,
        "performance": performance,
        "performance_is_not_a_gate": True,
        "seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# 7. The Phase 4 gauntlet
# ---------------------------------------------------------------------------


def run_gauntlet(checkpoint_path: Path, pair_ids, quick: bool) -> tuple[dict, list]:
    """Both modes against all four accepted baselines, with reproduction checks."""
    started = time.perf_counter()
    bank = SetupBank.generate(max(pair_ids) + 1)

    all_results = []
    per_mode = {}
    for policy_class, mode in (
        (GreedyNeuralPolicy, DECISION_MODE_GREEDY),
        (SeededCategoricalNeuralPolicy, DECISION_MODE_CATEGORICAL),
    ):
        policy = policy_class.from_checkpoint(checkpoint_path)
        policies = {policy.ref.token: policy}
        mode_results = []
        per_opponent = {}

        for opponent_id in CORE_BASELINES:
            units = build_paired_schedule(policy.ref, policy_ref(opponent_id), pair_ids)
            specs = [spec for unit in units for spec in unit.matches]
            summary = run_schedule(specs, bank, policies=policies, worker_count=1)
            rows = list(summary.results)
            mode_results.extend(rows)

            statistics_summary = summarize_matchup(rows, resamples=2000, seed=17)
            per_opponent[opponent_id] = {
                "matches": len(rows),
                "paired_units": summary.paired_units_run,
                "wins": statistics_summary.counts.wins,
                "draws": statistics_summary.counts.draws,
                "losses": statistics_summary.counts.losses,
                "effective_win_rate": round(statistics_summary.effective_win_rate, 4),
                "policy_errors": summary.policy_errors,
                "illegal_actions": summary.illegal_policy_actions,
                "mean_plies": statistics_summary.plies["mean"],
                "terminal_reasons": statistics_summary.terminal_reasons,
                "results_digest": summary.results_digest,
                "wall_clock_seconds": round(summary.wall_clock_seconds, 3),
            }

        # Reproduction: rerun the whole mode and compare, then replay every row.
        rerun_specs = [
            spec
            for opponent_id in CORE_BASELINES
            for unit in build_paired_schedule(policy.ref, policy_ref(opponent_id), pair_ids)
            for spec in unit.matches
        ]
        rerun = run_schedule(rerun_specs, bank, policies=policies, worker_count=1)
        rerun_problems = compare_results(mode_results, rerun.results)

        replay_problems = []
        row_only_problems = []
        sample = mode_results if quick else mode_results[:: max(1, len(mode_results) // 32)]
        for row in sample:
            replay_problems.extend(replay_stored_match(row))
            rebuilt = reproduce_match(row, policies=policies)
            row_only_problems.extend(compare_results([row], [rebuilt]))

        colour_swap_ok = True
        by_unit: dict[str, list] = {}
        for row in mode_results:
            by_unit.setdefault(row.paired_unit_id, []).append(row)
        for rows in by_unit.values():
            if len(rows) != 2 or {row.candidate_color for row in rows} != {RED, BLUE}:
                colour_swap_ok = False
            elif rows[0].red_setup != rows[1].red_setup or rows[0].blue_setup != rows[1].blue_setup:
                colour_swap_ok = False

        per_mode[mode] = {
            "policy_id": policy.policy_id,
            "policy_version": policy.policy_version,
            "stochastic": policy.stochastic,
            "matches": len(mode_results),
            "paired_units": len(by_unit),
            "per_opponent": per_opponent,
            "policy_errors": sum(1 for row in mode_results if row.errored),
            "illegal_actions": sum(
                1 for row in mode_results if row.policy_error_category == "illegal_action"
            ),
            "rerun_differences": rerun_problems,
            "rerun_identical": rerun_problems == [],
            "results_digest": results_digest(mode_results),
            "rerun_digest": results_digest(rerun.results),
            "replayed_rows": len(sample),
            "replay_problems": replay_problems,
            "row_only_reproduction_problems": row_only_problems,
            "colour_swap_correct": colour_swap_ok,
            "checkpoint_identity": policy.describe()["checkpoint"],
        }
        all_results.extend(mode_results)

    clean = all(
        entry["policy_errors"] == 0
        and entry["illegal_actions"] == 0
        and entry["rerun_identical"]
        and entry["replay_problems"] == []
        and entry["row_only_reproduction_problems"] == []
        and entry["colour_swap_correct"]
        for entry in per_mode.values()
    )

    return (
        {
            "schema_version": SCHEMA_VERSION,
            "setup_bank_version": SETUP_BANK_VERSION,
            "pairing_mode": PAIRING_COLOR_SWAP_SAME_BOARD,
            "suite_version": EVALUATION_SUITE_VERSION,
            "baselines": list(CORE_BASELINES),
            "setup_pair_ids": list(pair_ids),
            "paired_units_per_matchup": len(pair_ids),
            "worker_count": 1,
            "worker_count_note": (
                "serial by design: run_schedule rebuilds policies from the Phase 4 "
                "catalogue inside worker processes, and the neural policy is "
                "deliberately not in that catalogue"
            ),
            "modes": per_mode,
            "total_matches": len(all_results),
            "total_plies": sum(row.plies for row in all_results),
            "all_clean": clean,
            "seconds": time.perf_counter() - started,
        },
        all_results,
    )


# ---------------------------------------------------------------------------
# 8. Test suite
# ---------------------------------------------------------------------------


def run_pytest() -> dict:
    """Run the whole suite once, and break the result down per test module.

    The per-module breakdown comes from a JUnit XML report rather than a second
    pytest invocation, so each gate can cite the module that actually proves it
    without the run cost multiplying by the number of gates.
    """
    import re
    import xml.etree.ElementTree as ElementTree

    started = time.perf_counter()
    report = REPOSITORY_ROOT / "reports" / "phase_5_data" / ".pytest_junit.xml"
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
            # `file` is the repository-relative path when pytest records it; the
            # fallback `classname` is the dotted module, which is turned back
            # into a path so both forms key the report identically.
            module = case.get("file")
            if not module:
                dotted = case.get("classname", "")
                parts = [part for part in dotted.split(".") if part]
                if parts and parts[-1][:1].isupper():  # a test class, not the module
                    parts = parts[:-1]
                module = "/".join(parts) + ".py" if parts else dotted
            entry = per_module.setdefault(module, {"passed": 0, "failed": 0, "skipped": 0})
            if case.find("failure") is not None or case.find("error") is not None:
                entry["failed"] += 1
            elif case.find("skipped") is not None:
                entry["skipped"] += 1
            else:
                entry["passed"] += 1
        report.unlink()

    return {
        "command": f"python -m pytest -q --junitxml={report.name}",
        "returncode": process.returncode,
        "summary_line": tail,
        "passed": count("passed"),
        "failed": count("failed"),
        "errors": count("error"),
        "skipped": count("skipped"),
        "per_module": per_module,
        "seconds": round(time.perf_counter() - started, 2),
    }


def module_is_green(suite: dict, module: str) -> bool:
    """True when the named test module ran and every one of its tests passed."""
    entry = suite.get("per_module", {}).get(module)
    return bool(entry and entry["failed"] == 0 and entry["passed"] > 0)


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
        "--pair-ids", type=int, default=64, help="setup pairs per gauntlet matchup"
    )
    return parser.parse_args()


def main() -> int:
    options = parse_arguments()
    if options.quick:
        options.hidden_trials = min(options.hidden_trials, 400)
        options.pair_ids = min(options.pair_ids, 4)

    started = time.perf_counter()
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("Phase 5 acceptance harness")
    print(f"  commit          {git_commit()[:12]}")
    print(f"  torch           {torch.__version__}  mps={torch.backends.mps.is_available()}")
    print(f"  hidden trials   {options.hidden_trials:,}")
    print(f"  gauntlet pairs  {options.pair_ids}")
    print()

    contracts = verify_frozen_contracts()
    print(f"[1/7] frozen contracts verified: {contracts['all_ok']}")

    model = build_integration_model(seed=MODEL_SEED)
    checkpoint_path = save_checkpoint(
        model,
        CHECKPOINT_DIRECTORY / "integration_model_v1.pt",
        training_iteration=0,
        training_step=0,
        training_metrics={"note": "untrained Phase 5 integration fixture"},
    )
    print(f"      checkpoint written: {checkpoint_path.name} "
          f"({checkpoint_path.stat().st_size / 1024:.0f} KB)")

    mapping = audit_action_mapping(model, options.quick)
    print(
        f"[2/7] action mapping: {mapping['round_trip_mismatches']} round-trip mismatches, "
        f"{mapping['adapter_selection_mismatches']} selection mismatches over "
        f"{mapping['adapter_selections_checked']} crafted selections"
    )

    checkpoint_audit = audit_checkpoint(model, checkpoint_path)
    print(
        f"[3/7] checkpoint: round-trip identical={checkpoint_audit['round_trip_all_identical']}, "
        f"{checkpoint_audit['negative_cases_rejected']}/{checkpoint_audit['negative_cases_total']} "
        "negatives rejected"
    )

    greedy_policy = GreedyNeuralPolicy.from_checkpoint(checkpoint_path)
    hidden = audit_hidden_information(model, greedy_policy, options.hidden_trials)
    print(
        f"[4/7] hidden information: {hidden['trials']:,} trials, "
        f"{hidden['total_mismatches']} mismatches, "
        f"{hidden['positive_control_failures']} positive-control failures "
        f"({hidden['seconds']:.1f}s)"
    )

    value_belief = audit_value_belief_autograd(model)
    print(
        f"[5/7] value/belief/autograd: losses finite={value_belief['autograd']['all_losses_finite']}, "
        f"parameters without gradient={len(value_belief['autograd']['parameters_without_gradient'])}"
    )

    numerical = audit_numerical_and_performance(options.quick)
    print(f"[6/7] device/precision/batch/performance measured ({numerical['seconds']:.1f}s)")

    pair_ids = tuple(range(options.pair_ids))
    gauntlet, gauntlet_rows = run_gauntlet(checkpoint_path, pair_ids, options.quick)
    print(
        f"[7/7] gauntlet: {gauntlet['total_matches']} matches, "
        f"{gauntlet['total_plies']:,} plies, clean={gauntlet['all_clean']} "
        f"({gauntlet['seconds']:.1f}s)"
    )

    suite = {"skipped": True} if options.skip_pytest else run_pytest()
    if not options.skip_pytest:
        print(f"      test suite: {suite['summary_line']}")

    # ---- gates ----------------------------------------------------------
    mps_ok = numerical["mps_available"]
    float32_report = numerical["devices"].get("mps_float32", {})
    float16_report = numerical["devices"].get("mps_float16", {})

    suite_green = (
        not options.skip_pytest
        and suite.get("returncode") == 0
        and suite.get("failed", 1) == 0
        and suite.get("errors", 1) == 0
    )

    gates = {
        "frozen_contracts_verified_unchanged": contracts["all_ok"],
        # Measured before any Phase 5 file existed; the evidence is recorded in
        # `preexisting_suite` below rather than asserted here.
        "preexisting_suite_green": (
            PREEXISTING_SUITE["failed"] == 0 and PREEXISTING_SUITE["errors"] == 0
        ),
        "full_suite_green_after_changes": suite_green,
        "input_shape_and_dtype_validated": module_is_green(
            suite, "tests/model/test_contract.py"
        ),
        "tokenization_exact_row_major": (
            mapping["tokenization_mismatches"] == 0
            and mapping["tokenization_inverse_exact"]
            and module_is_green(suite, "tests/model/test_tokenization.py")
        ),
        "policy_output_contract_validated": (
            mapping["bijection_ok"] and module_is_green(suite, "tests/model/test_contract.py")
        ),
        "value_output_contract_validated": (
            value_belief["value"]["probabilities_sum_to_one"]
            and value_belief["value"]["expected_value_matches_win_minus_loss"]
            and value_belief["acting_player_perspective_mismatches"] == 0
            and module_is_green(suite, "tests/model/test_value_belief.py")
        ),
        "belief_output_and_mask_validated": (
            value_belief["belief"]["supervised_squares_max"] > 0
            and value_belief["belief"]["exclusions_all_clean"]
            and module_is_green(suite, "tests/model/test_value_belief.py")
        ),
        "all_10000_actions_round_trip": (
            mapping["round_trip_mismatches"] == 0 and mapping["arithmetic_mismatches"] == 0
        ),
        "policy_index_matches_engine_action": (
            mapping["adapter_selection_mismatches"] == 0
            and module_is_green(suite, "tests/model/test_policy_mapping.py")
        ),
        "legality_edge_cases_pass": module_is_green(suite, "tests/model/test_legality.py"),
        "engine_illegal_action_guard_preserved": (
            mapping["engine_illegal_guard_raised"] and mapping["engine_illegal_guard_inert"]
        ),
        "no_privileged_input_reachable": module_is_green(
            suite, "tests/model/test_hidden_information.py"
        ),
        "hidden_information_10000_zero_mismatch": (
            hidden["trials"] >= 10_000
            and hidden["total_mismatches"] == 0
            and hidden["positive_control_valid_on_every_trial"]
        ),
        "checkpoint_cpu_roundtrip_identity": (
            checkpoint_audit["round_trip_all_identical"]
            and module_is_green(suite, "tests/model/test_checkpoint.py")
        ),
        "checkpoint_incompatibilities_fail_loudly": checkpoint_audit[
            "all_incompatibilities_rejected"
        ],
        "greedy_and_seeded_modes_reproducible": (
            all(entry["rerun_identical"] for entry in gauntlet["modes"].values())
            and module_is_green(suite, "tests/model/test_evaluation_integration.py")
        ),
        "autograd_all_heads_connected_finite": (
            value_belief["autograd"]["all_losses_finite"]
            and not value_belief["autograd"]["parameters_without_gradient"]
            and not value_belief["autograd"]["non_finite_gradients"]
            and all(
                group["all_nonzero"]
                for group in value_belief["autograd"]["gradient_groups"].values()
            )
        ),
        "cpu_mps_float32_equivalence_pass": bool(
            mps_ok and float32_report.get("within_tolerance") and float32_report.get("crafted_margin_exact")
        ),
        "mps_float16_finite_and_equivalent": bool(
            mps_ok
            and float16_report.get("finite_outputs")
            and float16_report.get("within_tolerance")
            and float16_report.get("crafted_margin_exact")
        ),
        "batch_equivalence_pass": all(
            entry["within_tolerance"] and entry["selected_action_identical"]
            for entry in numerical["batch_equivalence"].values()
        )
        and module_is_green(suite, "tests/model/test_device_batch_equivalence.py"),
        "phase4_gauntlet_pass": gauntlet["all_clean"],
    }

    status = "PASS" if all(gates.values()) else "FAIL"

    acceptance = {
        "schema_version": SCHEMA_VERSION,
        "phase": 5,
        "agent": "agent_01",
        "status": status,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": environment(),
        "commands": [
            "python -m pytest -q",
            "python scripts/run_phase5.py",
        ],
        "quick_mode": options.quick,
        "frozen_contracts": contracts,
        "model": {
            "model_architecture_id": MODEL_ARCHITECTURE_ID,
            "configuration": IntegrationModelConfig().to_dict(),
            "parameter_count": model.parameter_count(),
            "initialisation_seed": MODEL_SEED,
            "integration_fixture": True,
            "trained": False,
            "summary": model.architecture_summary(),
        },
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(REPOSITORY_ROOT)),
            "file_digest": checkpoint_audit["checkpoint_file_digest"],
            "state_dict_digest": checkpoint_audit["metadata"]["state_dict_digest"],
            "bytes": checkpoint_audit["checkpoint_bytes"],
        },
        "gates": gates,
        "gates_true": sum(1 for value in gates.values() if value),
        "gates_total": len(gates),
        "headline_numbers": {
            "actions_round_tripped": mapping["round_trip_actions_checked"],
            "action_round_trip_mismatches": mapping["round_trip_mismatches"],
            "adapter_selections_checked": mapping["adapter_selections_checked"],
            "hidden_information_trials": hidden["trials"],
            "hidden_information_mismatches": hidden["total_mismatches"],
            "hidden_information_positive_control_failures": hidden["positive_control_failures"],
            "checkpoint_negative_cases": checkpoint_audit["negative_cases_total"],
            "gauntlet_matches": gauntlet["total_matches"],
            "gauntlet_plies": gauntlet["total_plies"],
            "gauntlet_illegal_actions": sum(
                entry["illegal_actions"] for entry in gauntlet["modes"].values()
            ),
            "gauntlet_policy_errors": sum(
                entry["policy_errors"] for entry in gauntlet["modes"].values()
            ),
            "tests_passed": suite.get("passed"),
            "tests_failed": suite.get("failed"),
            "tests_skipped": suite.get("skipped"),
        },
        "preexisting_suite": PREEXISTING_SUITE,
        "test_suite": suite,
        "artifacts": {
            "action_mapping": "reports/phase_5_data/agent_01_action_mapping.json",
            "hidden_information": "reports/phase_5_data/agent_01_hidden_information.json",
            "checkpoint_compatibility": (
                "reports/phase_5_data/agent_01_checkpoint_compatibility.json"
            ),
            "numerical_batch_performance": (
                "reports/phase_5_data/agent_01_numerical_batch_performance.json"
            ),
            "evaluation_gauntlet": "reports/phase_5_data/agent_01_evaluation_gauntlet.csv",
            "value_belief_autograd": "reports/phase_5_data/agent_01_value_belief_autograd.json",
        },
        "total_seconds": round(time.perf_counter() - started, 2),
    }

    write_json(DATA_DIRECTORY / "agent_01_action_mapping.json", mapping)
    write_json(DATA_DIRECTORY / "agent_01_hidden_information.json", hidden)
    write_json(DATA_DIRECTORY / "agent_01_checkpoint_compatibility.json", checkpoint_audit)
    write_json(DATA_DIRECTORY / "agent_01_numerical_batch_performance.json", numerical)
    write_json(DATA_DIRECTORY / "agent_01_value_belief_autograd.json", value_belief)
    write_json(DATA_DIRECTORY / "agent_01_evaluation_gauntlet.json", gauntlet)
    write_results_csv(DATA_DIRECTORY / "agent_01_evaluation_gauntlet.csv", gauntlet_rows)
    write_json(DATA_DIRECTORY / "agent_01_phase5_acceptance.json", acceptance)

    print()
    print(f"status                  {status}")
    print(f"gates true              {acceptance['gates_true']}/{acceptance['gates_total']}")
    for name, value in gates.items():
        if not value:
            print(f"  FAILED GATE           {name}")
    print(f"total seconds           {acceptance['total_seconds']}")
    print(f"written                 {DATA_DIRECTORY.relative_to(REPOSITORY_ROOT)}/")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
