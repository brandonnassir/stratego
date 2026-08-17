"""Phase 9 Agent 4: the information-security boundary of an RL example.

The property under test is the one the phase's observer-safety gate names: a
network, a mask, a PPO ratio, a KL term and a learner designation must all be
functions of public information alone, while the belief labels must be
functions of privileged truth. A paired hidden-identity permutation separates
those two claims — everything public must be bitwise unchanged, and the labels
must move exactly when the truth moves.

Every control here is a *positive* one. An anti-leak suite made only of "the
observation did not change" assertions would pass unchanged against a builder
that leaked, so each of the five failures the assignment names is planted and
required to be caught.
"""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

from stratego.engine.constants import opponent_of
from stratego.engine.observation import build_observation
from stratego.training import phase9_antileak as antileak
from stratego.training import phase9_collector as pc
from stratego.training import phase9_rollout_store as store
from stratego.training import phase9_targets as targets
from stratego.training.phase9_contract import (
    PHASE9_POPULATION_VERSION,
    PHASE9_ROLLOUT_SCHEDULE_VERSION,
)
from stratego.training.phase9_schedule import rebuild_scheduled_game
from stratego.training.phase9_seed import phase9_game_id
from stratego.training.reconstruction import iter_reconstructed_decisions

ANCHOR_CHECKPOINT = "checkpoints/phase8/warmstart_c1_v1.pt"
ANCHOR_SHA256 = "f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca"
CONTRACT_DIGEST = "ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34"

#: Trials are run at every sampled decision of one real game. Enough to cover
#: openings (many hidden pieces) and the late middlegame (few); the assignment's
#: 25,000-trial floor is the harness's job, not the suite's.
SAMPLED_PLIES = 32


@pytest.fixture(scope="module")
def game():
    """One real `current vs current` game, played once, with its statistics."""
    resolver = pc.SnapshotResolver(device="cpu", inference_batch_shape=4)
    behavior = resolver.resolve(
        ANCHOR_CHECKPOINT,
        logical_identity="B001",
        policy_token="phase9_behavior_v1|ns=canonical|B001",
        expected_sha256=ANCHOR_SHA256,
    )
    participants = pc.IterationParticipants(behavior=behavior, historical={})
    scheduled = rebuild_scheduled_game(phase9_game_id("canonical", 1, "current", 0))
    runner = pc.play_game(scheduled, participants)
    metadata = store.build_rollout_metadata(
        scheduled,
        runner.record,
        setup_provenance=runner.assignment.provenance,
        behavior_checkpoint_sha256=ANCHOR_SHA256,
        opponent_checkpoint_sha256=None,
        learner_decision_count=runner.learner_decision_count,
        population_version=PHASE9_POPULATION_VERSION,
        schedule_version=PHASE9_ROLLOUT_SCHEDULE_VERSION,
        contract_digest=CONTRACT_DIGEST,
    )
    record = runner.record
    sequences = targets.build_sequences(record, metadata)
    statistics = targets.iteration_statistics(
        {
            (record.game_id, ply): sequence.advantages[index]
            for sequence in sequences.values()
            for index, ply in enumerate(sequence.plies)
        },
        namespace="canonical",
        iteration=1,
        sealed_rollout_digest="digest",
        games=1,
    )
    return record, metadata, sequences, statistics


@pytest.fixture(scope="module")
def decisions(game):
    """`[(rebuilt, sequence), ...]` for the sampled plies, reconstructed once."""
    record, metadata, sequences, _statistics = game
    by_ply = {
        int(ply): sequence for sequence in sequences.values() for ply in sequence.plies
    }
    wanted = sorted(by_ply)[:SAMPLED_PLIES]
    return [
        (rebuilt, by_ply[rebuilt.ply])
        for rebuilt in iter_reconstructed_decisions(
            record, wanted, dense_mask=True, include_public_knowledge=False
        )
    ]


# ---------------------------------------------------------------------------
# Paired permutation trials
# ---------------------------------------------------------------------------


def test_no_public_field_moves_under_a_hidden_permutation(game, decisions):
    record, metadata, _sequences, statistics = game
    rng = random.Random(20260816)
    changed = 0
    for rebuilt, sequence in decisions:
        trial = antileak.hidden_permutation_trial(
            record, metadata, rebuilt, sequence, statistics, rng
        )
        assert trial["valid"]
        assert not trial["mismatches"], trial["mismatches"]
        assert trial["control_ok"]
        changed += int(trial["changed"])
    assert changed, "no trial actually reassigned an identity; nothing was proven"


def test_belief_labels_move_exactly_when_the_hidden_assignment_moves(game, decisions):
    record, metadata, _sequences, statistics = game
    rng = random.Random(7)
    for rebuilt, sequence in decisions:
        trial = antileak.hidden_permutation_trial(
            record, metadata, rebuilt, sequence, statistics, rng
        )
        assert trial["labels_differ"] == trial["truth_differs"]
        if trial["changed"]:
            assert trial["labels_differ"], "privileged labels ignored the truth"


def test_the_trial_catches_a_leaking_observation_builder(game, decisions):
    """Negative control: the comparison surface must be able to fail.

    The leaking builder writes the hidden types into the observation. Building
    an example around it and rerunning the comparison must report the
    observation as changed.
    """
    record, metadata, _sequences, statistics = game
    rebuilt, sequence = decisions[0]
    example = targets.build_example(record, metadata, rebuilt, sequence, statistics)
    leaked = replace(
        example,
        observation=antileak.leaking_observation_builder(
            rebuilt.state, int(rebuilt.acting_player)
        ),
    )
    assert not np.array_equal(leaked.observation, example.observation)
    assert targets.audit_example(leaked, record, metadata, rebuilt, sequence, statistics)


def test_a_permutation_does_not_change_the_legal_set_or_the_action_frame(game, decisions):
    record, metadata, _sequences, statistics = game
    rng = random.Random(11)
    rebuilt, _sequence = decisions[0]
    actor = int(rebuilt.acting_player)
    permuted, info = antileak.permute_hidden_identities(rebuilt.state, actor, rng)
    rebuilt_permuted = antileak.rebuild_from_state(permuted, rebuilt.ply)
    assert info["valid"]
    assert rebuilt_permuted.legal_action_ids == tuple(rebuilt.legal_action_ids)
    assert np.array_equal(
        rebuilt_permuted.observation, build_observation(rebuilt.state, actor)
    )


def test_hidden_truth_reads_only_unresolved_opponent_pieces(game, decisions):
    _record, _metadata, _sequences, _statistics = game
    rebuilt, _sequence = decisions[0]
    actor = int(rebuilt.acting_player)
    truth = antileak.hidden_truth(rebuilt.state, actor)
    expected = [
        piece.true_type
        for piece in rebuilt.state.pieces
        if piece.owner == opponent_of(actor) and piece.alive and not piece.known_to(actor)
    ]
    assert list(truth) == expected
    assert truth, "the sampled decision has no hidden opponent piece"


# ---------------------------------------------------------------------------
# The five positive controls
# ---------------------------------------------------------------------------


def _first_hostable(game, decisions):
    """The first sampled decision that can host all five controls."""
    record, metadata, _sequences, statistics = game
    for rebuilt, sequence in decisions:
        try:
            return rebuilt, sequence, antileak.positive_controls(
                record, metadata, rebuilt, sequence, statistics
            )
        except antileak.Phase9AntileakError:
            continue
    raise AssertionError("no sampled decision could host the positive controls")


def test_every_positive_control_fires(game, decisions):
    _rebuilt, _sequence, controls = _first_hostable(game, decisions)
    assert [control["control"] for control in controls] == list(
        antileak.POSITIVE_CONTROL_NAMES
    )
    for control in controls:
        assert control["fired"], f"{control['control']} did not fire"


def test_a_planted_identity_in_the_observation_is_caught(game, decisions):
    record, metadata, _sequences, statistics = game
    rebuilt, sequence = decisions[0]
    example = targets.build_example(record, metadata, rebuilt, sequence, statistics)
    planted = antileak.plant_identity_in_observation(example, rebuilt.state)
    problems = targets.audit_example(
        planted, record, metadata, rebuilt, sequence, statistics
    )
    assert any("observation" in problem for problem in problems)


def test_privileged_metadata_on_the_model_input_is_caught(game, decisions):
    record, metadata, _sequences, statistics = game
    rebuilt, sequence = decisions[0]
    example = targets.build_example(record, metadata, rebuilt, sequence, statistics)
    assert not antileak.audit_model_input(targets.model_input_fields_only(example))
    leaked = antileak.attach_privileged_metadata_to_model_input(example)
    assert antileak.audit_model_input(leaked)


def test_a_wrong_action_frame_is_caught(game, decisions):
    record, metadata, _sequences, statistics = game
    for rebuilt, sequence in decisions:
        example = targets.build_example(record, metadata, rebuilt, sequence, statistics)
        try:
            framed = antileak.use_wrong_action_frame(example)
        except antileak.Phase9AntileakError:
            continue
        assert targets.audit_example(
            framed, record, metadata, rebuilt, sequence, statistics
        )
        return
    raise AssertionError("no sampled decision could host the wrong-frame control")


def test_a_wrong_value_perspective_is_caught(game, decisions):
    record, metadata, _sequences, statistics = game
    for rebuilt, sequence in decisions:
        example = targets.build_example(record, metadata, rebuilt, sequence, statistics)
        try:
            valued = antileak.use_wrong_value_perspective(example, record)
        except antileak.Phase9AntileakError:
            continue
        assert targets.audit_example(
            valued, record, metadata, rebuilt, sequence, statistics
        )
        return
    raise AssertionError("no sampled decision could host the wrong-perspective control")


def test_a_wrong_learner_control_side_is_caught(game, decisions):
    record, metadata, _sequences, statistics = game
    rebuilt, sequence = decisions[0]
    example = targets.build_example(record, metadata, rebuilt, sequence, statistics)
    sided = antileak.use_wrong_learner_control_side(example)
    assert targets.audit_example(sided, record, metadata, rebuilt, sequence, statistics)


def test_a_vacuous_control_raises_instead_of_reporting_a_pass(game, decisions):
    """A control that plants nothing must never be counted as fired."""
    record, metadata, _sequences, statistics = game
    rebuilt, sequence = decisions[0]
    example = targets.build_example(record, metadata, rebuilt, sequence, statistics)
    class _NoHiddenState:
        pieces = ()

    with pytest.raises(antileak.Phase9AntileakError):
        antileak.plant_identity_in_observation(example, _NoHiddenState())


# ---------------------------------------------------------------------------
# The object-graph boundary
# ---------------------------------------------------------------------------


def test_an_example_carries_no_privileged_field(game, decisions):
    record, metadata, _sequences, statistics = game
    rebuilt, sequence = decisions[0]
    example = targets.build_example(record, metadata, rebuilt, sequence, statistics)
    assert not antileak.audit_example_object_graph(example)


def test_a_batch_model_input_holds_only_observations(game, decisions):
    record, metadata, _sequences, statistics = game
    examples = [
        targets.build_example(record, metadata, rebuilt, sequence, statistics)
        for rebuilt, sequence in decisions[:4]
    ]
    batch = targets.build_batch(examples)
    assert not antileak.audit_model_input(batch["model_input"])
    polluted = dict(batch["model_input"], belief_target=batch["loss_inputs"]["belief_target"])
    assert antileak.audit_model_input(polluted)


def test_a_model_input_of_the_wrong_dtype_is_caught(game, decisions):
    record, metadata, _sequences, statistics = game
    rebuilt, sequence = decisions[0]
    example = targets.build_example(record, metadata, rebuilt, sequence, statistics)
    assert antileak.audit_model_input(
        {"observation": example.observation.astype(np.float64)}
    )
