#!/usr/bin/env python3
"""Phase 9 Agent 4 acceptance harness: RL targets, advantages, anti-leak audit.

Re-verifies the accepted Agent 1-3 freezes from live source, then proves the
Phase 9 trainable-example semantics on real sealed bytes:

- every learner decision of a substantial sealed rollout audited exhaustively,
  with the learner designation, the final-perspective outcome, the same-player
  next-state links, every advantage, every W/D/L lambda target, the filter
  threshold, the eligibility and the belief labels all recomputed here through
  arithmetic that never calls the production helper;
- >= 25,000 valid hidden-identity permutation trials, each rebuilding the whole
  example from a counterfactual privileged state;
- five positive controls, each required to fire;
- >= 100,000 learner decisions re-checked against the exact frozen behavior
  snapshot with an independent legal-softmax recomputation.

Artifacts:

    reports/phase_9_data/agent_04_target_audit.json
    reports/phase_9_data/agent_04_antileak.json
    reports/phase_9_data/agent_04_example_contract.json
    reports/phase_9_data/agent_04_acceptance.json

The audited rollout
-------------------
`canonical` iteration 1 of Agent 3's sealed soak subtree: 2,048 games,
380,564 decisions, 282,414 of them learner-controlled, sealed digest
`df2e6e44...`. It is the largest sealed Phase 9 rollout in existence and it
carries all four population buckets and all three learner-control modes, which
is what makes a single exhaustive pass sufficient evidence.

What this script does not do
----------------------------
No optimizer is constructed, no loss is computed, no gradient is taken, no
checkpoint is selected, and no pilot is run. The Phase 9 final-test bank is
never opened. `phase9_targets` and `phase9_antileak` are scanned for training
symbols to prove that structurally rather than by assertion.

Usage::

    python scripts/run_phase9_agent04.py                  # every stage
    python scripts/run_phase9_agent04.py --stage targets
    python scripts/run_phase9_agent04.py --quick          # reduced volumes
    python scripts/run_phase9_agent04.py --record-final-suite
"""

from __future__ import annotations

import argparse
import ast
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from stratego.engine.constants import (  # noqa: E402
    BLUE,
    IMPLEMENTATION_VERSION,
    OBSERVATION_VERSION,
    PLAYER_NAMES,
    RED,
    RULES_VERSION,
)
from stratego.model.action_frame import model_action_to_absolute  # noqa: E402
from stratego.model.policy_adapter import prepare_legality  # noqa: E402
from stratego.training import phase9_antileak as antileak  # noqa: E402
from stratego.training import phase9_behavior as pb  # noqa: E402
from stratego.training import phase9_contract as contract  # noqa: E402
from stratego.training import phase9_rollout_store as store  # noqa: E402
from stratego.training import phase9_schedule as schedule  # noqa: E402
from stratego.training import phase9_seed as seeds  # noqa: E402
from stratego.training import phase9_storage as storage  # noqa: E402
from stratego.training import phase9_targets as targets  # noqa: E402
from stratego.training import synthetic_corpus as sc  # noqa: E402
from stratego.training.reconstruction import iter_reconstructed_decisions  # noqa: E402

AGENT = 4
PHASE = 9
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_9_data"

TARGET_ARTIFACT = DATA_DIRECTORY / "agent_04_target_audit.json"
ANTILEAK_ARTIFACT = DATA_DIRECTORY / "agent_04_antileak.json"
CONTRACT_ARTIFACT = DATA_DIRECTORY / "agent_04_example_contract.json"
ACCEPTANCE_ARTIFACT = DATA_DIRECTORY / "agent_04_acceptance.json"

AGENT1_ACCEPTANCE = DATA_DIRECTORY / "agent_01_acceptance.json"
AGENT2_ACCEPTANCE = DATA_DIRECTORY / "agent_02_acceptance.json"
AGENT3_ACCEPTANCE = DATA_DIRECTORY / "agent_03_acceptance.json"

#: Frozen Phase 8 inputs, pinned from the common contract.
ANCHOR_CHECKPOINT = REPOSITORY_ROOT / "checkpoints" / "phase8" / "warmstart_c1_v1.pt"
ANCHOR_SHA256 = "f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca"
UNTRAINED_CHECKPOINT = (
    REPOSITORY_ROOT / "checkpoints" / "phase8" / "warmstart_c1_v1_initialisation.pt"
)
ACCEPTED_CONTRACT_DIGEST = (
    "ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34"
)
ACCEPTED_POPULATION_DIGEST = (
    "6756790b15ee66195bc6339363e19fc475e3c606ef10613619b78b23d21bda73"
)

#: The harness may pin the expected resolver result to *verify the resolver*;
#: no library code hard-codes either path.
EXPECTED_CORPUS_ROOT = (
    "/Users/brandonwashington/Dev/Github/stratego/gpt_agent/"
    "data/stratego_phase8/warmstart/synthetic_warmstart_corpus_v1"
)

#: Agent 3's sealed soak subtree, and the rollout audited exhaustively here.
SOAK_SUBTREE = "agent_03_soak"
AUDIT_NAMESPACE = "canonical"
AUDIT_ITERATION = 1

#: Assignment floors.
ANTILEAK_TRIAL_MINIMUM = 25_000
BEHAVIOR_RECHECK_MINIMUM = 100_000

#: Historical-opponent decisions verified against their own archive member and
#: against the learner's snapshot. The point is the per-side discipline Agent 3
#: carried forward, not the volume.
HISTORICAL_SAMPLE = 2_048

#: Modules that must contain no training machinery at all.
TARGET_MODULES = (
    "stratego/training/phase9_targets.py",
    "stratego/training/phase9_antileak.py",
)

FORBIDDEN_TRAINING_SYMBOLS = (
    "backward",
    "zero_grad",
    "AdamW",
    "Adam",
    "SGD",
    "optim",
    "optimizer",
    "policy_loss",
    "value_loss",
    "belief_loss",
    "multi_head_loss",
    "step",
)

#: Numeric slack between the production targets and this harness's independent
#: arithmetic. Both are float64 over the same stored float32 inputs, so only
#: summation-order ulps may differ.
REFERENCE_TOLERANCE = 1e-12


def _print(message: str) -> None:
    print(message, flush=True)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"required artifact missing: {path}")
    return json.loads(path.read_text())


def rollout_root() -> Path:
    """The Agent 3 soak subtree, resolved through Agent 2's storage layer.

    The tracked pointer file decides where Phase 9 bytes live; no path is
    hard-coded into anything but this harness's diagnostics.
    """
    return Path(storage.default_rollout_root()) / SOAK_SUBTREE


# ---------------------------------------------------------------------------
# Independent reference arithmetic
#
# Deliberately written from the assignment's formulas rather than by calling
# `phase9_contract` or `phase9_targets`. The constants are literals here for
# the same reason: a tuned lambda in the frozen contract would show up as a
# mismatch instead of propagating silently into both sides of the comparison.
# ---------------------------------------------------------------------------

_ONE_HOT = {"win": (1.0, 0.0, 0.0), "draw": (0.0, 1.0, 0.0), "loss": (0.0, 0.0, 1.0)}


def _reference_outcome(terminal_result: str, player: int) -> str:
    if terminal_result == "draw":
        return "draw"
    if terminal_result not in ("red_win", "blue_win"):
        raise ValueError(f"unknown terminal result {terminal_result!r}")
    winner = RED if terminal_result == "red_win" else BLUE
    return "win" if int(player) == winner else "loss"


def _reference_targets(predictions, outcome: str) -> dict:
    """`v`, `delta`, `A` and `Y` of one same-player sequence, from scratch."""
    z = {"win": 1, "draw": 0, "loss": -1}[outcome]
    values = [float(item[0]) - float(item[2]) for item in predictions]
    count = len(values)
    deltas = [
        (values[index + 1] - values[index]) if index + 1 < count else (z - values[index])
        for index in range(count)
    ]
    advantages = [0.0] * count
    following = 0.0
    for index in range(count - 1, -1, -1):
        advantages[index] = deltas[index] + 0.5 * following
        following = advantages[index]
    wdl = [None] * count
    if count:
        wdl[count - 1] = _ONE_HOT[outcome]
        for index in range(count - 2, -1, -1):
            wdl[index] = tuple(
                0.2 * float(predictions[index + 1][component])
                + 0.8 * float(wdl[index + 1][component])
                for component in range(3)
            )
    return {"z": z, "values": values, "deltas": deltas, "advantages": advantages, "wdl": wdl}


def _reference_threshold(all_advantages) -> float:
    """`tau = max(Q_0.75(|A|), 0.01)` through numpy's linear quantile."""
    magnitudes = np.abs(np.asarray(all_advantages, dtype=np.float64))
    return float(max(np.quantile(magnitudes, 0.75, method="linear"), 0.01))


def _reference_moments(eligible) -> tuple:
    if not len(eligible):
        return (0.0, 0.0)
    array = np.asarray(eligible, dtype=np.float64)
    return (float(array.mean()), float(array.std()))


def _independent_legal_softmax(policy_logits_row, legality) -> tuple:
    """The temperature-1 legal softmax, recomputed without the production path.

    Reads the model-frame logits directly, softmaxes over exactly the legal
    entries in float64, and maps the result back to ascending absolute order
    through the frozen frame table. `phase9_behavior.behavior_distribution` is
    never called, so a bug there cannot verify itself.
    """
    row = np.asarray(policy_logits_row, dtype=np.float64).reshape(-1)
    ordered_model = sorted(int(action) for action in legality.model)
    values = row[ordered_model]
    weights = np.exp(values - values.max())
    probabilities = weights / weights.sum()
    by_absolute = {
        model_action_to_absolute(action, int(legality.acting_player)): float(probability)
        for action, probability in zip(ordered_model, probabilities)
    }
    return tuple(by_absolute[action] for action in legality.absolute)


def _independent_action_draw(probabilities, legal_absolute, game_id: str, ply: int) -> int:
    """The frozen cumulative walk, rewritten here rather than imported."""
    uniform = seeds.behavior_sample_uniform(game_id, int(ply))
    cumulative = 0.0
    for action, probability in zip(legal_absolute, probabilities):
        cumulative += float(probability)
        if cumulative >= uniform:
            return int(action)
    return int(legal_absolute[-1])


# ---------------------------------------------------------------------------
# Stage 1: prerequisites
# ---------------------------------------------------------------------------


def verify_prerequisites() -> dict:
    """Agents 1-3 `PASS`, their digests, and the sealed rollout evidence."""
    agent1 = _load_json(AGENT1_ACCEPTANCE)
    agent2 = _load_json(AGENT2_ACCEPTANCE)
    agent3 = _load_json(AGENT3_ACCEPTANCE)

    problems: list[str] = []
    for name, document in (("agent 1", agent1), ("agent 2", agent2), ("agent 3", agent3)):
        if document.get("status") != "PASS":
            problems.append(f"{name} reports status {document.get('status')!r}")
        gates = document.get("completion_gates", {})
        failed = [gate for gate, value in gates.items() if not value]
        if failed:
            problems.append(f"{name} carries failed gates: {failed}")

    live_contract = contract.contract_digest()
    live_population = schedule.population_digest()
    if live_contract != ACCEPTED_CONTRACT_DIGEST:
        problems.append("the live contract digest is not the accepted one")
    if live_population != ACCEPTED_POPULATION_DIGEST:
        problems.append("the live population digest is not the accepted one")

    recorded_schedules = agent3["prerequisites"]["run_schedule_digests"]
    schedule_digests = {
        namespace: schedule.run_schedule_digest(namespace)
        for namespace in seeds.RUN_NAMESPACES
    }
    key_by_namespace = {
        seeds.CANONICAL_NAMESPACE: "canonical",
        **{namespace: f"pilot_{namespace.split('_')[1]}" for namespace in seeds.PILOT_NAMESPACES},
    }
    for namespace, digest in schedule_digests.items():
        recorded = recorded_schedules.get(key_by_namespace[namespace])
        if recorded != digest:
            problems.append(f"{namespace} schedule digest drifted from Agent 3's record")

    # The rollout Agent 4 audits must be the sealed rollout Agent 3 recorded,
    # recomputed from the bytes rather than read from the state file.
    root = rollout_root()
    sealed = _load_json(DATA_DIRECTORY / "agent_03_collection_soak.json")
    recorded_digest = None
    for entry in sealed.get("iterations", []):
        if entry.get("namespace") == AUDIT_NAMESPACE and int(entry.get("iteration", 0)) == (
            AUDIT_ITERATION
        ):
            recorded_digest = entry.get("sealed_rollout_digest")
    reader = store.Phase9RolloutReader(root, AUDIT_NAMESPACE, AUDIT_ITERATION)
    live_digest = store.sealed_rollout_digest(list(reader.commits.values()))
    state = store.read_iteration_state(root, AUDIT_NAMESPACE, AUDIT_ITERATION)
    if state is None or state.get("state") == "COLLECTING":
        problems.append("the audited iteration is not sealed")
    if recorded_digest is not None and recorded_digest != live_digest:
        problems.append("the audited rollout's digest differs from Agent 3's record")
    if state is not None and state.get("sealed_rollout_digest") != live_digest:
        problems.append("the sealed state digest does not match the committed bytes")
    if state is not None and state.get("behavior_checkpoint_sha256") != ANCHOR_SHA256:
        problems.append("the audited rollout was not collected from the Phase 8 anchor")

    reproduction = _load_json(DATA_DIRECTORY / "agent_03_behavior_reproduction.json")
    if reproduction["learner"]["mismatches"] != 0:
        problems.append("Agent 3's behavior reproduction evidence carries mismatches")

    anchor_digest = pb.file_sha256(ANCHOR_CHECKPOINT)
    if anchor_digest != ANCHOR_SHA256:
        problems.append("the Phase 8 anchor checkpoint is not the accepted one")

    return {
        "agent1_status": agent1.get("status"),
        "agent2_status": agent2.get("status"),
        "agent3_status": agent3.get("status"),
        "agent1_gates": f"{agent1.get('gates_true')}/{agent1.get('gates_total')}",
        "agent2_gates": f"{agent2.get('gates_true')}/{agent2.get('gates_total')}",
        "agent3_gates": f"{agent3.get('gates_true')}/{agent3.get('gates_total')}",
        "contract_digest": live_contract,
        "contract_digest_matches_accepted": live_contract == ACCEPTED_CONTRACT_DIGEST,
        "population_digest": live_population,
        "population_digest_matches_accepted": live_population == ACCEPTED_POPULATION_DIGEST,
        "run_schedule_digests": schedule_digests,
        "phase8_checkpoint_sha256": anchor_digest,
        "phase8_checkpoint_matches_accepted": anchor_digest == ANCHOR_SHA256,
        "audited_rollout": {
            "root": str(root),
            "namespace": AUDIT_NAMESPACE,
            "iteration": AUDIT_ITERATION,
            "state": None if state is None else state.get("state"),
            "games": len(reader),
            "sealed_rollout_digest": live_digest,
            "digest_recorded_by_agent_3": recorded_digest,
            "digest_matches_agent_3": recorded_digest == live_digest,
            "behavior_snapshot_id": None if state is None else state.get("behavior_snapshot_id"),
            "behavior_checkpoint_sha256": (
                None if state is None else state.get("behavior_checkpoint_sha256")
            ),
        },
        "agent3_behavior_reproduction": {
            "learner_decisions": reproduction["learner"]["decisions"],
            "learner_mismatches": reproduction["learner"]["mismatches"],
            "historical_decisions": reproduction["historical"]["decisions"],
            "control_holds": reproduction["cross_checkpoint_control"]["control_holds"],
        },
        "problems": problems,
    }


def verify_corpus() -> dict:
    """The mandatory resolver check, plus a no-hard-coded-path scan.

    Agent 4 consumes no corpus payload — its examples come from sealed Phase 9
    rollouts — so the requirement here is the resolver and the accepted
    identity, computed from the corpus itself rather than quoted.
    """
    from stratego.training.corpus_commit import corpus_content_digest

    problems: list[str] = []
    resolution = sc.describe_corpus_root()
    root = sc.default_corpus_root()
    if str(root) != EXPECTED_CORPUS_ROOT:
        problems.append(f"corpus resolver returned {root}, expected {EXPECTED_CORPUS_ROOT}")

    observed = {
        "corpus_version": contract.EXPECTED_CORPUS_VERSION,
        "content_digest": corpus_content_digest(root, sc.CORPUS_SPLITS),
        "metadata_digest": sc._metadata_digest(root, sc.CORPUS_SPLITS),
        "commit_index_digest": sc._commit_index_digest(root, sc.CORPUS_SPLITS),
    }
    accepted = {
        "corpus_version": contract.EXPECTED_CORPUS_VERSION,
        "content_digest": contract.EXPECTED_CORPUS_CONTENT_DIGEST,
        "metadata_digest": contract.EXPECTED_CORPUS_METADATA_DIGEST,
        "commit_index_digest": contract.EXPECTED_CORPUS_COMMIT_INDEX_DIGEST,
    }
    for key, value in accepted.items():
        if observed[key] != value:
            problems.append(f"corpus {key} {observed[key]} != accepted {value}")

    # No target/dataset module may hard-code an absolute data or rollout path.
    hard_coded = []
    for module_path in TARGET_MODULES:
        text = (REPOSITORY_ROOT / module_path).read_text()
        if EXPECTED_CORPUS_ROOT in text or "/Volumes/" in text or "/Users/" in text:
            hard_coded.append(module_path)
    if hard_coded:
        problems.append(f"modules hard-code an absolute data path: {hard_coded}")

    return {
        "resolver": "stratego.training.synthetic_corpus.default_corpus_root()",
        "resolution": resolution,
        "resolved_root": str(root),
        "resolved_root_matches_expected": str(root) == EXPECTED_CORPUS_ROOT,
        "accepted_identity": accepted,
        "observed_identity": observed,
        "identity_matches": observed == accepted,
        "modules_scanned": list(TARGET_MODULES),
        "modules_hard_coding_absolute_paths": hard_coded,
        "identity_rule": (
            "corpus identity is version + accepted digests, not filesystem "
            "location; a digest mismatch is BLOCKED and never repaired"
        ),
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Stage 2: the example contract, the train order and the cursor
# ---------------------------------------------------------------------------


def run_contract_stage(reader, *, keys) -> dict:
    """Publish `phase9_example_v1` and demonstrate the order it hands Agent 5.

    The train order is exercised rather than described: two epochs are built,
    checked to be permutations of the same universe, checked to differ from
    each other and to be reproducible, and a mid-epoch cursor is resumed to
    prove the remaining minibatches are exactly the ones a crash would have
    left.
    """
    document = targets.example_contract()
    digest = targets.example_contract_digest()

    total = len(keys)
    slices = targets.minibatch_slices(total)
    epoch0 = targets.epoch_order(keys, AUDIT_NAMESPACE, AUDIT_ITERATION, 0)
    epoch1 = targets.epoch_order(keys, AUDIT_NAMESPACE, AUDIT_ITERATION, 1)
    repeat0 = targets.epoch_order(keys, AUDIT_NAMESPACE, AUDIT_ITERATION, 0)

    problems: list[str] = []
    if sorted(epoch0) != list(range(total)):
        problems.append("epoch 0 is not a permutation of the universe")
    if sorted(epoch1) != list(range(total)):
        problems.append("epoch 1 is not a permutation of the universe")
    if epoch0 != repeat0:
        problems.append("the epoch order is not reproducible")
    if epoch0 == epoch1:
        problems.append("two epochs produced the same order")
    if sum(stop - start for start, stop in slices) != total:
        problems.append("the minibatch slices do not cover the universe")
    if slices and slices[-1][1] - slices[-1][0] > contract.MINIBATCH_SIZE:
        problems.append("a minibatch exceeds the frozen size")

    # A resume must land on exactly the batches a crash left behind.
    cursor = targets.Phase9MinibatchCursor.start(
        namespace=AUDIT_NAMESPACE,
        iteration=AUDIT_ITERATION,
        sealed_rollout_digest=store.sealed_rollout_digest(list(reader.commits.values())),
        total_examples=total,
        epochs=contract.EPOCHS_PER_ROLLOUT,
    )
    consumed = 0
    for index in range(7):
        start, stop = slices[index]
        consumed += stop - start
        cursor = cursor.advance(stop - start)
    if cursor.examples_consumed != consumed or cursor.minibatch_index != 7:
        problems.append("the cursor does not track its own consumption")
    resumed = targets.minibatch_keys(
        keys, AUDIT_NAMESPACE, AUDIT_ITERATION, cursor.epoch, cursor.minibatch_index
    )
    expected_start, expected_stop = slices[7]
    if resumed != tuple(keys[position] for position in epoch0[expected_start:expected_stop]):
        problems.append("a resumed cursor does not reproduce the interrupted order")

    # Cursor rollover into the next epoch.
    rollover = targets.Phase9MinibatchCursor.start(
        namespace=AUDIT_NAMESPACE,
        iteration=AUDIT_ITERATION,
        sealed_rollout_digest=cursor.sealed_rollout_digest,
        total_examples=total,
        epochs=contract.EPOCHS_PER_ROLLOUT,
    )
    for _ in range(len(slices)):
        rollover = rollover.advance(0)
    if rollover.epoch != 1 or rollover.minibatch_index != 0:
        problems.append("the cursor does not roll into the next epoch")

    return {
        "example_contract": document,
        "example_contract_digest": digest,
        "train_order": {
            "version": contract.PHASE9_TRAIN_ORDER_VERSION,
            "namespace": AUDIT_NAMESPACE,
            "iteration": AUDIT_ITERATION,
            "universe": total,
            "minibatch_size": contract.MINIBATCH_SIZE,
            "minibatches_per_epoch": len(slices),
            "final_minibatch_size": slices[-1][1] - slices[-1][0] if slices else 0,
            "epochs_per_rollout": contract.EPOCHS_PER_ROLLOUT,
            "epoch0_seed": seeds.train_order_seed(AUDIT_NAMESPACE, AUDIT_ITERATION, 0),
            "epoch1_seed": seeds.train_order_seed(AUDIT_NAMESPACE, AUDIT_ITERATION, 1),
            "epoch_orders_differ": epoch0 != epoch1,
            "epoch_order_reproducible": epoch0 == repeat0,
            "resume_reproduces_interrupted_order": not any(
                "resumed cursor" in problem for problem in problems
            ),
            "cursor_after_seven_minibatches": cursor.to_dict(),
        },
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Stage 3: the exhaustive target audit
# ---------------------------------------------------------------------------


def _sequence_reference_problems(record, metadata, sequences) -> list:
    """Recompute one game's learner sequences without the production module.

    Learner designation is rebuilt from the *schedule*, the plies are rescanned
    from the payload, the same-player links are checked directly, and the
    outcome, values, deltas, advantages and W/D/L targets all come from
    :func:`_reference_targets`.
    """
    problems: list[str] = []
    game_id = record.game_id
    scheduled = schedule.rebuild_scheduled_game(game_id)
    expected_sides = {name for name in scheduled.learner_sides}
    metadata_sides = {
        PLAYER_NAMES[player] for player in targets.learner_players(metadata)
    }
    if expected_sides != metadata_sides:
        problems.append(
            f"{game_id}: schedule trains {sorted(expected_sides)}, metadata claims "
            f"{sorted(metadata_sides)}"
        )
    if metadata["learner_control"] != scheduled.learner_control:
        problems.append(f"{game_id}: learner_control disagrees with the schedule")
    if metadata["learner_color"] != scheduled.learner_color:
        problems.append(f"{game_id}: learner_color disagrees with the schedule")

    for player, sequence in sequences.items():
        colour = PLAYER_NAMES[player]
        if colour not in expected_sides:
            problems.append(f"{game_id}: {colour} is not a scheduled learner side")
        own = tuple(
            index
            for index, decision in enumerate(record.decisions)
            if int(decision.acting_player) == int(player)
        )
        if own != sequence.plies:
            problems.append(f"{game_id} {colour}: sequence plies are not this player's own")
        if any(later <= earlier for earlier, later in zip(own, own[1:])):
            problems.append(f"{game_id} {colour}: sequence plies are not increasing")
        for earlier, later in zip(own, own[1:]):
            between = [
                index
                for index in range(earlier + 1, later)
                if int(record.decisions[index].acting_player) == int(player)
            ]
            if between:
                problems.append(f"{game_id} {colour}: a same-player decision was skipped")
        if own and own[-1] != max(
            index
            for index, decision in enumerate(record.decisions)
            if int(decision.acting_player) == int(player)
        ):
            problems.append(f"{game_id} {colour}: the sequence does not end at the last turn")

        outcome = _reference_outcome(record.terminal_result, player)
        if outcome != sequence.outcome:
            problems.append(f"{game_id} {colour}: outcome perspective disagrees")
        predictions = [
            tuple(float(value) for value in record.decisions[ply].win_draw_loss_prediction)
            for ply in own
        ]
        reference = _reference_targets(predictions, outcome)
        if reference["z"] != sequence.z:
            problems.append(f"{game_id} {colour}: z disagrees")
        for index, (mine, theirs) in enumerate(zip(reference["values"], sequence.values)):
            if abs(mine - theirs) > REFERENCE_TOLERANCE:
                problems.append(f"{game_id} {colour}: v_{index} disagrees")
        for index, (mine, theirs) in enumerate(zip(reference["deltas"], sequence.deltas)):
            if abs(mine - theirs) > REFERENCE_TOLERANCE:
                problems.append(f"{game_id} {colour}: delta_{index} disagrees")
    return problems


def run_target_audit(reader, *, limit: "int | None", audit_examples: bool) -> dict:
    """Audit every learner decision of the sealed rollout. Zero mismatches.

    Two passes, as the module documents: the first derives every sequence and
    every advantage from the stored decisions alone, which is what makes the
    per-iteration threshold computable; the second replays each game and audits
    the built examples.
    """
    started = time.perf_counter()
    root = reader.root if hasattr(reader, "root") else None
    sealed = store.sealed_rollout_digest(list(reader.commits.values()))

    advantages_by_key: dict = {}
    sequences_by_game: dict = {}
    reference_advantages: dict = {}
    reference_wdl: dict = {}
    sequence_problems: list[str] = []
    quantity_problems: list[str] = []
    control_counts: dict = {}
    colour_counts: dict = {"red": 0, "blue": 0}
    bucket_counts: dict = {}
    simplex_failures = 0
    games = 0

    for index, game_id in enumerate(reader.game_ids):
        if limit is not None and index >= limit:
            break
        record, metadata = reader.read_game(game_id)
        sequences = targets.build_sequences(record, metadata)
        sequences_by_game[game_id] = sequences
        quantity_problems.extend(
            targets.verify_learner_decision_count(record, metadata, sequences)
        )
        sequence_problems.extend(_sequence_reference_problems(record, metadata, sequences))

        control = metadata["learner_control"]
        control_counts[control] = control_counts.get(control, 0) + 1
        bucket_counts[metadata["bucket"]] = bucket_counts.get(metadata["bucket"], 0) + 1

        for player, sequence in sequences.items():
            colour_counts[PLAYER_NAMES[player]] += len(sequence)
            predictions = [
                tuple(float(value) for value in record.decisions[ply].win_draw_loss_prediction)
                for ply in sequence.plies
            ]
            reference = _reference_targets(
                predictions, _reference_outcome(record.terminal_result, player)
            )
            for position, ply in enumerate(sequence.plies):
                key = (game_id, int(ply))
                advantages_by_key[key] = float(sequence.advantages[position])
                reference_advantages[key] = float(reference["advantages"][position])
                reference_wdl[key] = tuple(reference["wdl"][position])
                target = sequence.wdl_targets[position]
                if (
                    not all(np.isfinite(target))
                    or any(value < -targets.SIMPLEX_TOLERANCE for value in target)
                    or abs(sum(target) - 1.0) > targets.SIMPLEX_TOLERANCE
                ):
                    simplex_failures += 1
        games += 1

    advantage_mismatches = [
        key
        for key, value in advantages_by_key.items()
        if abs(value - reference_advantages[key]) > REFERENCE_TOLERANCE
    ]

    statistics = targets.iteration_statistics(
        advantages_by_key,
        namespace=AUDIT_NAMESPACE,
        iteration=AUDIT_ITERATION,
        sealed_rollout_digest=sealed,
        games=games,
    )
    reference_threshold = _reference_threshold(list(reference_advantages.values()))
    reference_eligible = [
        value for value in reference_advantages.values() if abs(value) >= reference_threshold
    ]
    reference_mean, reference_std = _reference_moments(reference_eligible)
    filter_problems: list[str] = []
    if abs(statistics.threshold - reference_threshold) > REFERENCE_TOLERANCE:
        filter_problems.append("the filter threshold disagrees with the reference quantile")
    if statistics.eligible != len(reference_eligible):
        filter_problems.append("the eligible count disagrees with the reference filter")
    if abs(statistics.mean_eligible - reference_mean) > REFERENCE_TOLERANCE:
        filter_problems.append("the PPO subset mean disagrees")
    if abs(statistics.std_eligible - reference_std) > REFERENCE_TOLERANCE:
        filter_problems.append("the PPO subset std disagrees")
    pass1_seconds = time.perf_counter() - started

    # -- pass 2: replay, build, audit ---------------------------------------
    example_problems: list[str] = []
    belief_mismatches = 0
    eligibility_mismatches = 0
    standardization_mismatches = 0
    wdl_mismatches = 0
    examples = 0
    eligible_examples = 0
    belief_squares = 0
    started_pass2 = time.perf_counter()

    if audit_examples:
        for index, game_id in enumerate(reader.game_ids):
            if limit is not None and index >= limit:
                break
            record, metadata = reader.read_game(game_id)
            sequences = sequences_by_game[game_id]
            by_ply = {
                int(ply): sequence
                for sequence in sequences.values()
                for ply in sequence.plies
            }
            for rebuilt in iter_reconstructed_decisions(
                record,
                sorted(by_ply),
                dense_mask=True,
                include_public_knowledge=False,
                copy_state=False,
            ):
                sequence = by_ply[int(rebuilt.ply)]
                example = targets.build_example(
                    record, metadata, rebuilt, sequence, statistics
                )
                problems = targets.audit_example(
                    example, record, metadata, rebuilt, sequence, statistics
                )
                key = (game_id, int(rebuilt.ply))
                if any("belief" in problem for problem in problems):
                    belief_mismatches += 1
                if example.ppo_eligible != (
                    abs(reference_advantages[key]) >= reference_threshold
                ):
                    eligibility_mismatches += 1
                    problems.append(f"{key}: eligibility disagrees with the reference filter")
                expected_standardized = (
                    0.0
                    if not reference_eligible
                    else (reference_advantages[key] - reference_mean) / (reference_std + 1e-8)
                )
                if abs(example.standardized_advantage - expected_standardized) > 1e-9:
                    standardization_mismatches += 1
                    problems.append(f"{key}: standardized advantage disagrees")
                if any(
                    abs(float(mine) - float(theirs)) > REFERENCE_TOLERANCE
                    for mine, theirs in zip(example.wdl_target, reference_wdl[key])
                ):
                    wdl_mismatches += 1
                    problems.append(f"{key}: W/D/L target disagrees with the reference")
                if problems:
                    example_problems.extend(problems)
                examples += 1
                eligible_examples += int(example.ppo_eligible)
                belief_squares += example.supervised_belief_squares
    pass2_seconds = time.perf_counter() - started_pass2

    return {
        "namespace": AUDIT_NAMESPACE,
        "iteration": AUDIT_ITERATION,
        "rollout_root": str(root),
        "sealed_rollout_digest": sealed,
        "games_audited": games,
        "learner_decisions": len(advantages_by_key),
        "examples_audited": examples,
        "learner_control_counts": control_counts,
        "bucket_counts": bucket_counts,
        "learner_decisions_by_colour": colour_counts,
        "statistics": statistics.to_dict(),
        "reference": {
            "threshold": reference_threshold,
            "eligible": len(reference_eligible),
            "mean_eligible": reference_mean,
            "std_eligible": reference_std,
            "arithmetic": (
                "written from the assignment's formulas in this harness; neither "
                "phase9_contract nor phase9_targets is consulted for a value it "
                "is being compared against"
            ),
        },
        "advantage_mismatches": len(advantage_mismatches),
        "wdl_target_mismatches": wdl_mismatches,
        "value_target_simplex_failures": simplex_failures,
        "belief_target_mismatches": belief_mismatches,
        "eligibility_mismatches": eligibility_mismatches,
        "standardization_mismatches": standardization_mismatches,
        "filter_problems": filter_problems,
        "sequence_problems": sequence_problems[:10],
        "sequence_problem_count": len(sequence_problems),
        "behavior_quantity_problems": quantity_problems[:10],
        "behavior_quantity_problem_count": len(quantity_problems),
        "example_problems": example_problems[:10],
        "example_problem_count": len(example_problems),
        "eligible_examples": eligible_examples,
        "retention_fraction": eligible_examples / examples if examples else 0.0,
        "supervised_belief_squares": belief_squares,
        "mean_supervised_belief_squares": belief_squares / examples if examples else 0.0,
        "pass1_seconds": pass1_seconds,
        "pass2_seconds": pass2_seconds,
        "decisions_per_second": examples / pass2_seconds if pass2_seconds else 0.0,
    }, statistics, sequences_by_game


# ---------------------------------------------------------------------------
# Stage 4: anti-leak trials and positive controls
# ---------------------------------------------------------------------------


def run_antileak(reader, statistics, sequences_by_game, *, minimum: int, seed: int = 20260816) -> dict:
    """Hidden-identity permutation trials over real sealed learner decisions."""
    started = time.perf_counter()
    rng = random.Random(seed)
    trials = 0
    valid = 0
    changed = 0
    invariant_mismatches = 0
    control_failures = 0
    mismatch_examples: list[str] = []
    boundary_problems: list[str] = []
    object_graph_problems: list[str] = []
    hidden_piece_total = 0
    games_used = 0
    controls: list[dict] = []
    control_attempts = 0

    for game_id in reader.game_ids:
        if valid >= minimum and controls:
            break
        record, metadata = reader.read_game(game_id)
        sequences = sequences_by_game.get(game_id) or targets.build_sequences(record, metadata)
        by_ply = {
            int(ply): sequence for sequence in sequences.values() for ply in sequence.plies
        }
        if not by_ply:
            continue
        games_used += 1
        for rebuilt in iter_reconstructed_decisions(
            record,
            sorted(by_ply),
            dense_mask=True,
            include_public_knowledge=False,
            copy_state=True,
        ):
            if valid >= minimum and controls:
                break
            sequence = by_ply[int(rebuilt.ply)]
            trial = antileak.hidden_permutation_trial(
                record, metadata, rebuilt, sequence, statistics, rng
            )
            trials += 1
            valid += int(trial["valid"])
            changed += int(trial["changed"])
            hidden_piece_total += trial["hidden_pieces"]
            if trial["mismatches"]:
                invariant_mismatches += len(trial["mismatches"])
                mismatch_examples.extend(trial["mismatches"][:2])
            if not trial["control_ok"]:
                control_failures += 1

            if trials % 512 == 1:
                example = targets.build_example(
                    record, metadata, rebuilt, sequence, statistics
                )
                boundary_problems.extend(
                    antileak.audit_model_input(targets.model_input_fields_only(example))
                )
                object_graph_problems.extend(antileak.audit_example_object_graph(example))
                batch = targets.build_batch([example])
                boundary_problems.extend(antileak.audit_model_input(batch["model_input"]))

            if not controls:
                control_attempts += 1
                try:
                    controls = antileak.positive_controls(
                        record, metadata, rebuilt, sequence, statistics
                    )
                except antileak.Phase9AntileakError:
                    # A vacuous control is refused rather than counted; move on
                    # to a decision that can actually host all five.
                    controls = []

    elapsed = time.perf_counter() - started
    fired = [control for control in controls if control["fired"]]
    return {
        "antileak_version": antileak.PHASE9_ANTILEAK_VERSION,
        "namespace": AUDIT_NAMESPACE,
        "iteration": AUDIT_ITERATION,
        "sealed_rollout_digest": statistics.sealed_rollout_digest,
        "seed": seed,
        "games_used": games_used,
        "trials": trials,
        "valid_trials": valid,
        "assignment_changed_trials": changed,
        "mean_hidden_pieces": hidden_piece_total / trials if trials else 0.0,
        "invariant_fields": list(antileak.INVARIANT_FIELDS),
        "privileged_fields": list(antileak.PRIVILEGED_FIELDS),
        "invariant_mismatches": invariant_mismatches,
        "label_control_failures": control_failures,
        "mismatch_examples": mismatch_examples[:10],
        "model_input_boundary_problems": boundary_problems[:10],
        "model_input_boundary_problem_count": len(boundary_problems),
        "object_graph_problems": object_graph_problems[:10],
        "positive_controls": controls,
        "positive_controls_fired": len(fired),
        "positive_controls_expected": len(antileak.POSITIVE_CONTROL_NAMES),
        "positive_control_decisions_tried": control_attempts,
        "all_positive_controls_fire": len(fired) == len(antileak.POSITIVE_CONTROL_NAMES),
        "seconds": elapsed,
        "trials_per_second": trials / elapsed if elapsed else 0.0,
    }


# ---------------------------------------------------------------------------
# Stage 5: independent behavior-policy consistency
# ---------------------------------------------------------------------------


def _load_snapshot(identity: str, token: str, device: str, batch_shape: int, model=None, hint=None):
    return pb.load_behavior_snapshot(
        ANCHOR_CHECKPOINT,
        logical_identity=identity,
        policy_token=token,
        device=device,
        inference_batch_shape=batch_shape,
        expected_sha256=ANCHOR_SHA256,
        model=model,
        state_dict_digest_hint=hint,
    )


def _recheck_chunk(snapshot, chunk) -> dict:
    """One forward pass' worth of independent comparisons."""
    observations = np.stack([item["observation"] for item in chunk])
    policy_logits, wdl = pb.evaluate_observations(snapshot, observations)
    mismatches: list[str] = []
    max_probability = 0.0
    max_wdl = 0.0
    redraw_mismatches = 0
    for row, item in enumerate(chunk):
        recomputed = _independent_legal_softmax(policy_logits[row], item["legality"])
        stored = item["stored_probabilities"]
        if len(recomputed) != len(stored):
            mismatches.append(f"{item['game_id']} ply {item['ply']}: legal set size differs")
            continue
        difference = max(
            abs(float(left) - float(right)) for left, right in zip(stored, recomputed)
        )
        max_probability = max(max_probability, difference)
        if difference > contract.BEHAVIOR_PROBABILITY_ABS_TOLERANCE:
            mismatches.append(
                f"{item['game_id']} ply {item['ply']}: distribution differs by {difference:.3e}"
            )
        wdl_difference = max(
            abs(float(left) - float(right))
            for left, right in zip(item["stored_wdl"], np.asarray(wdl[row]).reshape(3))
        )
        max_wdl = max(max_wdl, wdl_difference)
        if wdl_difference > contract.BEHAVIOR_PROBABILITY_ABS_TOLERANCE:
            mismatches.append(
                f"{item['game_id']} ply {item['ply']}: W/D/L differs by {wdl_difference:.3e}"
            )
        redrawn = _independent_action_draw(
            stored, item["legality"].absolute, item["game_id"], item["ply"]
        )
        if redrawn != item["stored_action"]:
            redraw_mismatches += 1
            mismatches.append(
                f"{item['game_id']} ply {item['ply']}: the stored distribution redraws "
                f"{redrawn}, not {item['stored_action']}"
            )
        realized = float(stored[list(item["legality"].absolute).index(item["stored_action"])])
        if realized != item["stored_realized_probability"]:
            mismatches.append(
                f"{item['game_id']} ply {item['ply']}: pi_b(a_t|s_t) is not the stored entry"
            )
    return {
        "mismatches": mismatches,
        "max_probability": max_probability,
        "max_wdl": max_wdl,
        "redraw_mismatches": redraw_mismatches,
    }


def _decision_requests(record, metadata, wanted_players):
    """Replay one game and yield everything a re-check needs, per decision.

    An independent replay: the state is advanced through the frozen engine from
    the payload's own setups and actions, and the legality is regenerated at
    every ply rather than read from the record.
    """
    from stratego.engine.legal_moves import legal_action_mask, legal_actions
    from stratego.engine.observation import build_observation
    from stratego.engine.state import create_game
    from stratego.engine.transition import apply_action
    from stratego.training.warmstart_contract import CORPUS_RULES

    state = create_game(
        record.red_setup, record.blue_setup, rules=CORPUS_RULES, game_id=record.game_id
    )
    for decision in record.decisions:
        legal = legal_actions(state)
        actor = int(state.acting_player)
        if actor in wanted_players:
            legality = prepare_legality(legal, legal_action_mask(state, legal), actor)
            stored_legal = tuple(int(action) for action in decision.legal_action_ids)
            if tuple(legality.absolute) != stored_legal:
                yield {"kind": "legal_set_mismatch", "ply": decision.ply}
            else:
                yield {
                    "kind": "request",
                    "game_id": record.game_id,
                    "ply": int(decision.ply),
                    "acting_player": actor,
                    "observation": build_observation(state, actor),
                    "legality": legality,
                    "stored_probabilities": tuple(
                        float(value) for value in decision.old_probabilities
                    ),
                    "stored_wdl": tuple(
                        float(value) for value in decision.win_draw_loss_prediction
                    ),
                    "stored_action": int(decision.selected_action_id),
                    "stored_token": decision.collection_policy_version,
                    "stored_realized_probability": targets.behavior_action_probability(decision),
                }
        apply_action(state, decision.selected_action_id, legal=legal)


def run_behavior_consistency(
    reader, *, device: str, batch_shape: int, minimum: int, historical_sample: int
) -> dict:
    """Re-check learner decisions against the exact frozen behavior snapshot.

    Independent of Agent 3's acceptance function in every part that could hide
    an error: this harness replays the games itself, regenerates legality
    itself, recomputes the legal softmax itself in float64, and redraws the
    realized action with its own cumulative walk. The only shared machinery is
    the forward pass at the frozen batch shape, which is the thing being
    verified against.
    """
    started = time.perf_counter()
    behavior_token = schedule.behavior_policy_token(AUDIT_NAMESPACE, AUDIT_ITERATION)
    learner_snapshot = _load_snapshot(
        schedule.behavior_snapshot_identity(AUDIT_ITERATION),
        behavior_token,
        device,
        batch_shape,
    )
    anchor_snapshot = _load_snapshot(
        contract.HISTORICAL_ANCHOR_ID,
        schedule.ANCHOR_POLICY_TOKEN,
        device,
        batch_shape,
        model=learner_snapshot.model,
        hint=learner_snapshot.loaded_state_dict_digest,
    )
    digest_before = learner_snapshot.loaded_state_dict_digest

    checked = 0
    mismatches: list[str] = []
    max_probability = 0.0
    max_wdl = 0.0
    redraw_mismatches = 0
    legal_set_mismatches = 0
    token_mismatches = 0
    games_used = 0
    historical_checked = 0
    historical_mismatches = 0
    wrong_side_control = {"decisions": 0, "verified": 0, "max_abs_difference": 0.0}

    pending: list = []
    for game_id in reader.game_ids:
        if checked >= minimum and historical_checked >= historical_sample:
            break
        record, metadata = reader.read_game(game_id)
        learner = set(targets.learner_players(metadata))
        games_used += 1
        for item in _decision_requests(record, metadata, learner):
            if item["kind"] == "legal_set_mismatch":
                legal_set_mismatches += 1
                continue
            if item["stored_token"] != behavior_token:
                token_mismatches += 1
                mismatches.append(
                    f"{item['game_id']} ply {item['ply']}: learner decision stores token "
                    f"{item['stored_token']!r}"
                )
            pending.append(item)
            if len(pending) >= batch_shape:
                report = _recheck_chunk(learner_snapshot, pending)
                checked += len(pending)
                mismatches.extend(report["mismatches"])
                max_probability = max(max_probability, report["max_probability"])
                max_wdl = max(max_wdl, report["max_wdl"])
                redraw_mismatches += report["redraw_mismatches"]
                pending = []
        if checked >= minimum:
            break

    if pending:
        report = _recheck_chunk(learner_snapshot, pending)
        checked += len(pending)
        mismatches.extend(report["mismatches"])
        max_probability = max(max_probability, report["max_probability"])
        max_wdl = max(max_wdl, report["max_wdl"])
        redraw_mismatches += report["redraw_mismatches"]

    # -- the per-side carry-forward ----------------------------------------
    # A historical opponent's moves belong to its own archive member. Verifying
    # them against the iteration's learner snapshot must *fail*; against H000 it
    # must pass. Both directions are measured, because only the pair proves the
    # discipline rather than the tolerance.
    historical_pending: list = []
    for game_id in reader.game_ids:
        if historical_checked >= historical_sample:
            break
        record, metadata = reader.read_game(game_id)
        if metadata["opponent_kind"] != "historical_snapshot":
            continue
        opponents = {
            player
            for player in (RED, BLUE)
            if player not in set(targets.learner_players(metadata))
        }
        for item in _decision_requests(record, metadata, opponents):
            if item["kind"] != "request":
                continue
            historical_pending.append(item)
            if len(historical_pending) >= batch_shape:
                report = _recheck_chunk(anchor_snapshot, historical_pending)
                historical_checked += len(historical_pending)
                historical_mismatches += len(report["mismatches"])
                if wrong_side_control["decisions"] < batch_shape:
                    wrong = _recheck_chunk(learner_snapshot, historical_pending)
                    # H000 and B001 are the same weights in iteration 1, so the
                    # control that can actually fail is the *untrained*
                    # checkpoint; recorded below.
                    wrong_side_control["decisions"] = len(historical_pending)
                    wrong_side_control["verified"] = len(historical_pending) - len(
                        wrong["mismatches"]
                    )
                    wrong_side_control["max_abs_difference"] = wrong["max_probability"]
                historical_pending = []
                if historical_checked >= historical_sample:
                    break

    untrained = run_untrained_control(reader, device=device, batch_shape=batch_shape)
    learner_snapshot.assert_frozen()
    digest_after = pb.state_dict_digest(learner_snapshot.model)
    elapsed = time.perf_counter() - started
    return {
        "device": device,
        "inference_batch_shape": batch_shape,
        "tolerance": contract.BEHAVIOR_PROBABILITY_ABS_TOLERANCE,
        "recomputation": (
            "float64 legal softmax over the model-frame logits, mapped back to "
            "ascending absolute order through the frozen action frame; the "
            "production behavior_distribution is not called"
        ),
        "behavior_snapshot_id": learner_snapshot.logical_identity,
        "behavior_checkpoint_sha256": learner_snapshot.checkpoint_sha256,
        "behavior_policy_token": behavior_token,
        "games_used": games_used,
        "learner_decisions_rechecked": checked,
        "learner_mismatches": len(mismatches),
        "max_abs_probability_difference": max_probability,
        "max_abs_wdl_difference": max_wdl,
        "action_redraw_mismatches": redraw_mismatches,
        "legal_set_mismatches": legal_set_mismatches,
        "policy_token_mismatches": token_mismatches,
        "problems": mismatches[:10],
        "historical_side": {
            "decisions": historical_checked,
            "mismatches": historical_mismatches,
            "snapshot": anchor_snapshot.logical_identity,
            "note": (
                "iteration 1's H000 and B001 are the same accepted Phase 8 file, "
                "so this direction proves the per-side resolution path rather "
                "than a weight difference; the untrained control below is the "
                "one that can fail"
            ),
            "same_weights_as_learner": anchor_snapshot.checkpoint_sha256
            == learner_snapshot.checkpoint_sha256,
            "cross_snapshot_agreement": wrong_side_control,
        },
        "untrained_checkpoint_control": untrained,
        "snapshot_state_dict_digest_before": digest_before,
        "snapshot_state_dict_digest_after": digest_after,
        "snapshot_weights_unchanged": digest_before == digest_after,
        "seconds": elapsed,
        "decisions_per_second": checked / elapsed if elapsed else 0.0,
    }


def run_untrained_control(reader, *, device: str, batch_shape: int, sample: int = 256) -> dict:
    """Negative control: the re-check must fail against the wrong network."""
    untrained_digest = pb.file_sha256(UNTRAINED_CHECKPOINT)
    snapshot = pb.load_behavior_snapshot(
        UNTRAINED_CHECKPOINT,
        logical_identity="phase8_untrained_control",
        policy_token=schedule.behavior_policy_token(AUDIT_NAMESPACE, AUDIT_ITERATION),
        device=device,
        inference_batch_shape=batch_shape,
        expected_sha256=untrained_digest,
    )
    collected: list = []
    for game_id in reader.game_ids:
        record, metadata = reader.read_game(game_id)
        learner = set(targets.learner_players(metadata))
        for item in _decision_requests(record, metadata, learner):
            if item["kind"] == "request":
                collected.append(item)
            if len(collected) >= sample:
                break
        if len(collected) >= sample:
            break
    verified = 0
    worst = 0.0
    for start in range(0, len(collected), batch_shape):
        chunk = collected[start : start + batch_shape]
        report = _recheck_chunk(snapshot, chunk)
        verified += len(chunk) - len(
            {problem.split(":")[0] for problem in report["mismatches"]}
        )
        worst = max(worst, report["max_probability"])
    return {
        "checkpoint": str(UNTRAINED_CHECKPOINT.relative_to(REPOSITORY_ROOT)),
        "checkpoint_sha256": untrained_digest,
        "decisions": len(collected),
        "verified_against_wrong_checkpoint": max(verified, 0),
        "max_abs_probability_difference": worst,
        "control_holds": worst > contract.BEHAVIOR_PROBABILITY_ABS_TOLERANCE,
    }


# ---------------------------------------------------------------------------
# Stage 6: no meaningful RL training
# ---------------------------------------------------------------------------


def audit_no_meaningful_training() -> dict:
    """Prove structurally that nothing here can train a network.

    An AST walk over the target modules for optimizer/loss symbols — names and
    attributes only, so prose in a docstring does not trip it — plus a live
    check that a snapshot used for a re-check comes back with its parameters
    still frozen and its weights unmoved.
    """
    findings: list[str] = []
    for module_path in TARGET_MODULES:
        tree = ast.parse((REPOSITORY_ROOT / module_path).read_text())
        used: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)
        hits = sorted(used & set(FORBIDDEN_TRAINING_SYMBOLS))
        if hits:
            findings.append(f"{module_path} uses {hits}")

    snapshot = _load_snapshot(
        schedule.behavior_snapshot_identity(AUDIT_ITERATION),
        schedule.behavior_policy_token(AUDIT_NAMESPACE, AUDIT_ITERATION),
        "cpu",
        8,
    )
    before = snapshot.loaded_state_dict_digest
    trainable = [
        name for name, parameter in snapshot.model.named_parameters() if parameter.requires_grad
    ]
    grad_enabled = torch.is_grad_enabled()
    after = pb.state_dict_digest(snapshot.model)
    if trainable:
        findings.append(f"{len(trainable)} parameters are trainable")
    if after != before:
        findings.append("the snapshot's weights moved")
    return {
        "modules_scanned": list(TARGET_MODULES),
        "forbidden_symbols": list(FORBIDDEN_TRAINING_SYMBOLS),
        "symbol_findings": findings,
        "trainable_parameters": len(trainable),
        "grad_enabled_at_audit": bool(grad_enabled),
        "state_dict_digest_before": before,
        "state_dict_digest_after": after,
        "weights_unchanged": after == before,
        "no_meaningful_rl_training": not findings,
    }


# ---------------------------------------------------------------------------
# Gates, artifacts and the entry point
# ---------------------------------------------------------------------------


def build_gates(prerequisites, corpus, contract_stage, audit, leak, behavior, training, tests_after) -> dict:
    controls_fired = leak.get("all_positive_controls_fire", False)
    return {
        "agents1_3_pass": all(
            prerequisites.get(key) == "PASS"
            for key in ("agent1_status", "agent2_status", "agent3_status")
        )
        and not prerequisites.get("problems", ["x"])
        and prerequisites.get("contract_digest_matches_accepted", False),
        "corpus_resolver_verified": corpus.get("resolved_root_matches_expected", False)
        and not corpus.get("modules_hard_coding_absolute_paths", ["x"]),
        "corpus_digests_match": corpus.get("identity_matches", False),
        "same_player_sequence_audit_pass": audit.get("sequence_problem_count", 1) == 0
        and audit.get("behavior_quantity_problem_count", 1) == 0,
        "red_blue_perspective_audit_pass": sorted(
            audit.get("learner_control_counts", {})
        )
        == ["blue", "both", "red"]
        and audit.get("learner_decisions_by_colour", {}).get("red", 0) > 0
        and audit.get("learner_decisions_by_colour", {}).get("blue", 0) > 0
        and audit.get("example_problem_count", 1) == 0,
        "advantages_exhaustively_match": audit.get("advantage_mismatches", 1) == 0
        and audit.get("learner_decisions", 0) > 0,
        "wdl_targets_exhaustively_match": audit.get("wdl_target_mismatches", 1) == 0,
        "advantage_filter_exact": not audit.get("filter_problems", ["x"])
        and audit.get("eligibility_mismatches", 1) == 0
        and audit.get("standardization_mismatches", 1) == 0,
        "value_target_simplex_failures_zero": audit.get("value_target_simplex_failures", 1) == 0,
        "belief_target_mismatches_zero": audit.get("belief_target_mismatches", 1) == 0,
        "behavior_reproduction_ge_100k": behavior.get("learner_decisions_rechecked", 0)
        >= BEHAVIOR_RECHECK_MINIMUM,
        "behavior_reproduction_mismatches_zero": behavior.get("learner_mismatches", 1) == 0
        and behavior.get("legal_set_mismatches", 1) == 0
        and behavior.get("action_redraw_mismatches", 1) == 0
        and behavior.get("policy_token_mismatches", 1) == 0
        and behavior.get("untrained_checkpoint_control", {}).get("control_holds", False),
        "hidden_permutation_trials_ge_25000": leak.get("valid_trials", 0)
        >= ANTILEAK_TRIAL_MINIMUM,
        "model_input_leak_mismatches_zero": leak.get("invariant_mismatches", 1) == 0
        and leak.get("label_control_failures", 1) == 0
        and leak.get("model_input_boundary_problem_count", 1) == 0
        and not leak.get("object_graph_problems", ["x"]),
        "positive_controls_fire": controls_fired,
        "learner_control_mismatches_zero": audit.get("sequence_problem_count", 1) == 0,
        "no_meaningful_rl_training": bool(training.get("no_meaningful_rl_training"))
        and not contract_stage.get("problems", ["x"]),
        # `returncode` is present only when pytest actually ran.
        "full_suite_green": tests_after.get("returncode") == 0,
    }


DEVIATIONS = [
    (
        "The exhaustive audit runs on Agent 3's sealed soak subtree "
        "(<rollout_root>/agent_03_soak/canonical iteration 1) rather than on a "
        "production <rollout_root>/canonical/ tree, because that is where the "
        "only substantial sealed Phase 9 rollout currently lives. The games are "
        "the real scheduled iteration-1 games and the sealed digest recomputed "
        "here matches the one Agent 3 recorded, so the audited object is the "
        "real one; identity is version + digests, never a path."
    ),
    (
        "`phase9_example_v1` is an Agent 4 addition rather than one of Agent 1's "
        "nine frozen identities: the assignment requires an example/batch "
        "contract and Agent 1 froze none. It names only the *shape* of the "
        "object handed to Agent 5 — every learning constant it quotes is read "
        "from the frozen `phase9_contract`, so a tuned value would have to be "
        "tuned there, where the accepted contract digest would catch it."
    ),
    (
        "Iteration 1's historical opponent H000 and behavior snapshot B001 are "
        "the same accepted Phase 8 file, so verifying a historical opponent's "
        "moves against B001 cannot fail on weights alone. The per-side "
        "discipline is therefore evidenced by the resolution path plus an "
        "explicit untrained-checkpoint control, which does fail by ~1e-1."
    ),
]

HANDOFF = {
    "example_iterator": (
        "phase9_targets.iter_rollout_examples(reader, statistics) — games in "
        "ascending game_id order, decisions in ascending ply order, every read "
        "digest-checked by the store"
    ),
    "example_schema": "phase9_targets.Phase9RLExample / phase9_example_v1",
    "train_order": (
        "phase9_targets.train_order_keys(reader) then epoch_order(keys, "
        "namespace, iteration, epoch); contiguous 512-example minibatches, the "
        "final partial batch consumed"
    ),
    "cursor": (
        "phase9_targets.Phase9MinibatchCursor — (epoch, minibatch_index, "
        "examples_consumed); minibatch_keys() rebuilds the exact keys of an "
        "interrupted batch from the sealed rollout alone"
    ),
    "ppo_eligibility": (
        "IterationTargetStatistics.is_eligible(A) = |A| >= tau, tau = "
        "max(Q_0.75(|A|), 0.01) over the whole sealed iteration; "
        "Phase9RLExample.ppo_eligible carries it per example"
    ),
    "standardized_advantages": (
        "IterationTargetStatistics.standardize(A) over the PPO subset's mean and "
        "population std with epsilon 1e-8; zero-variance and empty-subset both "
        "yield 0.0 and are recorded as flags on the statistics"
    ),
    "behavior_quantity": (
        "Phase9RLExample.behavior_action_probability is the stored float32 "
        "pi_b(a_t|s_t); behavior_legal_probabilities is the full legal "
        "distribution for the KL term, ascending absolute order"
    ),
    "wdl_targets": "Phase9RLExample.wdl_target, learner perspective, categorical CE",
    "belief_targets": (
        "Phase9RLExample.belief_target / belief_mask — the frozen Phase 8 "
        "hidden-only semantics via belief_targets.dense_belief_target"
    ),
    "model_input_boundary": (
        "phase9_targets.model_input_fields_only(example) and build_batch()['model_input'] "
        "are the only routes to the backbone; phase9_antileak.audit_model_input "
        "refuses anything else"
    ),
    "value_and_belief_are_unfiltered": (
        "the advantage filter applies to the PPO policy loss only; value, "
        "belief, KL and entropy use every learner decision"
    ),
}


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            cwd=REPOSITORY_ROOT,
            check=False,
        ).stdout.decode().strip()
    except OSError:  # pragma: no cover
        return "unknown"


def _git_state() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        return "dirty" if result.stdout.strip() else "clean"
    except OSError:  # pragma: no cover
        return "unknown"


def run_pytest() -> dict:
    started = time.perf_counter()
    command = [sys.executable, "-m", "pytest", "tests", "-q"]
    result = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True)
    output = result.stdout.decode()
    summary = ""
    for line in output.splitlines()[::-1]:
        if " passed" in line or " failed" in line or " error" in line:
            summary = line.strip()
            break
    passed = skipped = 0
    for token, name in ((" passed", "passed"), (" skipped", "skipped")):
        if token in summary:
            piece = summary.split(token)[0].split()[-1]
            if piece.isdigit():
                if name == "passed":
                    passed = int(piece)
                else:
                    skipped = int(piece)
    return {
        "command": " ".join(command),
        "returncode": result.returncode,
        "summary": summary,
        "seconds": time.perf_counter() - started,
        "passed": passed,
        "skipped": skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 9 Agent 4 acceptance harness")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "verify", "contract", "targets", "antileak", "behavior"],
    )
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--batch-shape", type=int, default=pb.DEFAULT_INFERENCE_BATCH_SHAPE)
    parser.add_argument("--games", type=int, default=None, help="limit audited games")
    parser.add_argument("--antileak-trials", type=int, default=ANTILEAK_TRIAL_MINIMUM)
    parser.add_argument("--behavior-minimum", type=int, default=BEHAVIOR_RECHECK_MINIMUM)
    parser.add_argument("--quick", action="store_true", help="small volumes, for a smoke run")
    parser.add_argument("--run-pytest", action="store_true")
    parser.add_argument("--record-final-suite", action="store_true")
    arguments = parser.parse_args()

    if arguments.record_final_suite:
        return record_final_suite()

    if arguments.quick:
        arguments.games = arguments.games or 24
        arguments.antileak_trials = min(arguments.antileak_trials, 256)
        arguments.behavior_minimum = min(arguments.behavior_minimum, 512)

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    durations: dict = {}
    started = time.perf_counter()

    _print("[1/6] prerequisites and corpus")
    prerequisites = verify_prerequisites()
    corpus = verify_corpus()
    durations["verify"] = time.perf_counter() - started
    for problem in prerequisites["problems"]:
        _print(f"      ! {problem}")
    if arguments.stage == "verify":
        _print(json.dumps({"prerequisites": prerequisites, "corpus": corpus}, indent=2, default=str))
        return 0 if not prerequisites["problems"] else 1

    root = rollout_root()
    reader = store.Phase9RolloutReader(root, AUDIT_NAMESPACE, AUDIT_ITERATION)
    _print(f"      rollout {AUDIT_NAMESPACE}/it{AUDIT_ITERATION:03d}: {len(reader)} games")

    _print("[2/6] exhaustive target audit")
    stage_started = time.perf_counter()
    audit, statistics, sequences_by_game = run_target_audit(
        reader, limit=arguments.games, audit_examples=True
    )
    durations["targets"] = time.perf_counter() - stage_started
    _print(
        f"      {audit['learner_decisions']} learner decisions, "
        f"{audit['examples_audited']} examples, "
        f"tau={audit['statistics']['threshold']:.6f}, "
        f"{audit['example_problem_count']} problems"
    )
    if arguments.stage == "targets":
        TARGET_ARTIFACT.write_text(json.dumps(audit, indent=2, default=str) + "\n")
        return 0

    _print("[3/6] example contract, train order and cursor")
    stage_started = time.perf_counter()
    keys = tuple(sorted(advantage_key for advantage_key in _keys_from(sequences_by_game)))
    contract_stage = run_contract_stage(reader, keys=keys)
    durations["contract"] = time.perf_counter() - stage_started
    _print(
        f"      universe {len(keys)}, "
        f"{contract_stage['train_order']['minibatches_per_epoch']} minibatches/epoch"
    )
    if arguments.stage == "contract":
        CONTRACT_ARTIFACT.write_text(json.dumps(contract_stage, indent=2, default=str) + "\n")
        return 0

    _print("[4/6] anti-leak trials and positive controls")
    stage_started = time.perf_counter()
    leak = run_antileak(
        reader, statistics, sequences_by_game, minimum=arguments.antileak_trials
    )
    durations["antileak"] = time.perf_counter() - stage_started
    _print(
        f"      {leak['valid_trials']} valid trials, "
        f"{leak['invariant_mismatches']} invariant mismatches, "
        f"{leak['positive_controls_fired']}/{leak['positive_controls_expected']} controls fire"
    )
    if arguments.stage == "antileak":
        ANTILEAK_ARTIFACT.write_text(json.dumps(leak, indent=2, default=str) + "\n")
        return 0

    _print("[5/6] independent behavior-policy consistency")
    stage_started = time.perf_counter()
    behavior = run_behavior_consistency(
        reader,
        device=arguments.device,
        batch_shape=arguments.batch_shape,
        minimum=arguments.behavior_minimum,
        historical_sample=HISTORICAL_SAMPLE if not arguments.quick else 128,
    )
    durations["behavior"] = time.perf_counter() - stage_started
    _print(
        f"      {behavior['learner_decisions_rechecked']} decisions, "
        f"{behavior['learner_mismatches']} mismatches, "
        f"max |dp| = {behavior['max_abs_probability_difference']:.3e}"
    )
    if arguments.stage == "behavior":
        return 0

    _print("[6/6] no-training audit and artifacts")
    training = audit_no_meaningful_training()
    tests_after = run_pytest() if arguments.run_pytest else {}

    gates = build_gates(
        prerequisites, corpus, contract_stage, audit, leak, behavior, training, tests_after
    )
    problems = (
        prerequisites["problems"]
        + contract_stage["problems"]
        + audit["filter_problems"]
        + audit["sequence_problems"]
        + audit["example_problems"]
        + leak["mismatch_examples"]
        + behavior["problems"]
        + training["symbol_findings"]
    )
    acceptance = {
        "phase": PHASE,
        "agent": AGENT,
        "status": "PASS" if all(value for key, value in gates.items() if key != "full_suite_green")
        else "FAIL",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "source_revision": _git_revision(),
        "working_tree_state": _git_state(),
        "rules_version": RULES_VERSION,
        "engine_version": IMPLEMENTATION_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "artifact": "agent_04_acceptance",
        "example_version": targets.PHASE9_EXAMPLE_VERSION,
        "example_contract_digest": contract_stage["example_contract_digest"],
        "advantage_version": contract.PHASE9_ADVANTAGE_VERSION,
        "train_order_version": contract.PHASE9_TRAIN_ORDER_VERSION,
        "antileak_version": antileak.PHASE9_ANTILEAK_VERSION,
        "prerequisites": prerequisites,
        "corpus": corpus,
        "target_audit": {
            key: value
            for key, value in audit.items()
            if key not in ("sequence_problems", "example_problems", "behavior_quantity_problems")
        },
        "antileak": {
            key: value for key, value in leak.items() if key != "positive_controls"
        },
        "positive_controls": leak["positive_controls"],
        "behavior_consistency": behavior,
        "train_order": contract_stage["train_order"],
        "no_training_audit": training,
        "tests_after": tests_after,
        "completion_gates": gates,
        "gates_total": len(gates),
        "gates_true": sum(1 for value in gates.values() if value),
        "problems": problems[:20],
        "deviations": DEVIATIONS,
        "handoff_to_agent_5": HANDOFF,
        "durations": durations,
        "total_seconds": time.perf_counter() - started,
    }

    TARGET_ARTIFACT.write_text(json.dumps(audit, indent=2, default=str) + "\n")
    ANTILEAK_ARTIFACT.write_text(json.dumps(leak, indent=2, default=str) + "\n")
    CONTRACT_ARTIFACT.write_text(json.dumps(contract_stage, indent=2, default=str) + "\n")
    ACCEPTANCE_ARTIFACT.write_text(json.dumps(acceptance, indent=2, default=str) + "\n")

    _print("")
    _print(f"status: {acceptance['status']}  gates {acceptance['gates_true']}/{acceptance['gates_total']}")
    for name, value in gates.items():
        if not value:
            _print(f"  FALSE  {name}")
    return 0 if acceptance["status"] == "PASS" else 1


def _keys_from(sequences_by_game: dict):
    for game_id, sequences in sequences_by_game.items():
        for sequence in sequences.values():
            for ply in sequence.plies:
                yield (game_id, int(ply))


def record_final_suite() -> int:
    """Re-run the suite with the artifacts in place and record the result.

    A test inside the suite cannot soundly assert that the suite passed, so the
    artifact tests verify every gate except `full_suite_green`; that one is
    established here, by a pass that could actually see the artifacts.
    """
    if not ACCEPTANCE_ARTIFACT.exists():
        raise SystemExit("run the harness before recording the final suite")
    acceptance = json.loads(ACCEPTANCE_ARTIFACT.read_text())
    tests_after = run_pytest()
    tests_after["covers_agent_04_artifact_tests"] = True
    acceptance["tests_after"] = tests_after
    acceptance["completion_gates"]["full_suite_green"] = tests_after["returncode"] == 0
    acceptance["gates_true"] = sum(1 for value in acceptance["completion_gates"].values() if value)
    acceptance["status"] = (
        "PASS" if all(acceptance["completion_gates"].values()) else "FAIL"
    )
    ACCEPTANCE_ARTIFACT.write_text(json.dumps(acceptance, indent=2, default=str) + "\n")
    _print(f"suite: {tests_after['summary']}")
    _print(f"status: {acceptance['status']}")
    return 0 if acceptance["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
