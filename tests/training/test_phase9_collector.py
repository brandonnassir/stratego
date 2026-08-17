"""Phase 9 Agent 3: the self-play collector.

What is checked here is the collection *boundary*, not the game loop: that an
iteration draws every current-policy move from one immutable snapshot, that a
historical opponent's moves are attributed to its own checkpoint and never to
the learner's, that the model never sees anything the acting player may not
legally know, and that a crashed collection resumes to byte-identical bytes.

The observer-safety test is a positive control by construction: it plants
privileged truth in an observation and requires the audit to find it. An audit
that merely rebuilt the observation and compared it to itself would pass
without ever being able to detect a leak.
"""

from __future__ import annotations

import numpy as np
import pytest

from stratego.engine.legal_moves import legal_actions
from stratego.engine.observation import build_observation
from stratego.engine.state import create_game
from stratego.engine.transition import apply_action
from stratego.training import phase9_collector as pc
from stratego.training import phase9_rollout_store as store
from stratego.training.phase9_contract import (
    PHASE9_POPULATION_VERSION,
    PHASE9_ROLLOUT_SCHEDULE_VERSION,
)
from stratego.training.phase9_schedule import rebuild_scheduled_game
from stratego.training.phase9_seed import phase9_game_id
from stratego.training.setup_source import training_setup_source
from stratego.training.warmstart_contract import CORPUS_RULES, EXPECTED_SETUP_PROFILE

ANCHOR_CHECKPOINT = "checkpoints/phase8/warmstart_c1_v1.pt"
ANCHOR_SHA256 = "f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca"
UNTRAINED_CHECKPOINT = "checkpoints/phase8/warmstart_c1_v1_initialisation.pt"
UNTRAINED_SHA256 = "01c907eeef86ec04121db55ccffb9365e8df27fdf05921b921947d4af365754c"
CONTRACT_DIGEST = "ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34"

COLLECT_KWARGS = dict(
    population_version=PHASE9_POPULATION_VERSION,
    schedule_version=PHASE9_ROLLOUT_SCHEDULE_VERSION,
    contract_digest=CONTRACT_DIGEST,
)


@pytest.fixture(scope="module")
def participants():
    """Iteration 1 of a fresh run: `B001` and `H000` are both the anchor file."""
    resolver = pc.SnapshotResolver(device="cpu", inference_batch_shape=2)
    behavior = resolver.resolve(
        ANCHOR_CHECKPOINT,
        logical_identity="B001",
        policy_token="phase9_behavior_v1|ns=canonical|B001",
        expected_sha256=ANCHOR_SHA256,
    )
    anchor = resolver.resolve(
        ANCHOR_CHECKPOINT,
        logical_identity="H000",
        policy_token="phase9_anchor_v1|H000",
        expected_sha256=ANCHOR_SHA256,
    )
    assert resolver.load_count == 1, "identical weights must be loaded once"
    return pc.IterationParticipants(behavior=behavior, historical={"H000": anchor})


def _scheduled(bucket, ordinal=0, iteration=1, namespace="canonical"):
    return rebuild_scheduled_game(phase9_game_id(namespace, iteration, bucket, ordinal))


def _position(root_seed=555, plies=3):
    source = training_setup_source(EXPECTED_SETUP_PROFILE)
    assignment = source.assign(root_seed=root_seed, environment_id=0, generation=0)
    state = create_game(
        assignment.red_setup, assignment.blue_setup, rules=CORPUS_RULES, game_id="probe"
    )
    for _ in range(plies):
        apply_action(state, legal_actions(state)[0])
    return state


# ---------------------------------------------------------------------------
# Who plays which side
# ---------------------------------------------------------------------------


def test_every_bucket_plays_and_records_its_scheduled_participants(participants):
    for bucket in ("current", "historical", "rule", "stress"):
        scheduled = _scheduled(bucket)
        runner = pc.play_game(scheduled, participants)
        record = runner.record

        tokens = {decision.collection_policy_version for decision in record.decisions}
        assert tokens <= {scheduled.red_policy_identity, scheduled.blue_policy_identity}
        assert record.collection_checkpoint_id == ANCHOR_SHA256
        # Learner accounting matches the frozen learner-control semantics.
        expected = sum(
            1
            for decision in record.decisions
            if ("red" if decision.acting_player == 0 else "blue") in scheduled.learner_sides
        )
        assert runner.learner_decision_count == expected
        if bucket == "current":
            assert scheduled.learner_control == "both"
            assert runner.learner_decision_count == len(record.decisions)
        else:
            assert scheduled.learner_control in ("red", "blue")
            assert runner.learner_decision_count < len(record.decisions)


def test_a_rule_side_is_never_a_learner_side(participants):
    for bucket in ("rule", "stress"):
        scheduled = _scheduled(bucket)
        runner = pc.play_game(scheduled, participants)
        opponent_colour = "red" if scheduled.learner_color == "blue" else "blue"
        assert opponent_colour not in scheduled.learner_sides
        # Its decisions are stored (state reconstruction needs them) as the
        # accepted one-hot with the neutral value, and carry its own token.
        opponent = 0 if opponent_colour == "red" else 1
        stored = [d for d in runner.record.decisions if d.acting_player == opponent]
        assert stored
        for decision in stored:
            assert sorted(decision.old_probabilities)[-1] == pytest.approx(1.0)
            assert decision.win_draw_loss_prediction == pytest.approx((1 / 3, 1 / 3, 1 / 3))
            assert "@" in decision.collection_policy_version


def test_a_historical_opponents_moves_are_attributed_to_its_own_snapshot(participants):
    scheduled = _scheduled("historical")
    learner = 0 if scheduled.learner_color == "red" else 1
    assert pc.acting_snapshot_for(scheduled, participants, learner).logical_identity == "B001"
    assert (
        pc.acting_snapshot_for(scheduled, participants, 1 - learner).logical_identity == "H000"
    )


def test_an_unbound_archive_identity_cannot_be_collected(participants):
    """No fabricated checkpoint may satisfy a pre-enumerated schedule."""
    empty = pc.IterationParticipants(behavior=participants.behavior, historical={})
    scheduled = _scheduled("historical")
    with pytest.raises(pc.Phase9CollectorError, match="real immutable checkpoint"):
        pc.play_game(scheduled, empty)


def test_a_historical_opponent_with_different_weights_changes_the_game(participants):
    """Proof the opponent's own checkpoint really drives its moves.

    `H000` and `B001` are the same file in iteration 1, so a collector that
    quietly used the learner for both sides would look correct. Swapping in
    genuinely different opponent weights has to change the game.
    """
    other = pc.SnapshotResolver(device="cpu", inference_batch_shape=2).resolve(
        UNTRAINED_CHECKPOINT,
        logical_identity="H000",
        policy_token="phase9_anchor_v1|H000",
        expected_sha256=UNTRAINED_SHA256,
    )
    scheduled = _scheduled("historical")
    baseline = pc.play_game(scheduled, participants).record
    swapped = pc.play_game(
        scheduled,
        pc.IterationParticipants(behavior=participants.behavior, historical={"H000": other}),
    ).record
    assert swapped.actions != baseline.actions


# ---------------------------------------------------------------------------
# Observer safety
# ---------------------------------------------------------------------------


def test_the_model_input_is_the_observer_safe_observation(participants):
    state = _position()
    observer = state.acting_player
    report = pc.observer_safety_probe(state, observer, build_observation(state, observer))
    assert report["safe"], report["problems"]
    assert report["hidden_opponent_pieces"] > 0
    assert report["permutation_applied"]
    assert report["entries_sensitive_to_hidden_truth"] == 0


def _leaking_builder(state, observer):
    """An observation builder that writes hidden opponent types into a channel."""
    leaked = np.array(build_observation(state, observer))
    for record in state.pieces:
        if record.owner != observer and record.alive and not record.known_to(observer):
            row, column = divmod(record.current_square, 10)
            leaked[0, row, column] = float(record.true_type) + 1.0
    return leaked


def test_the_probe_detects_a_builder_that_leaks_privileged_truth():
    """The positive control: a planted leak must actually be caught.

    The leaking builder is self-consistent — the input it produces matches the
    input it declares — so only the hidden-truth counterfactual can find it.
    This is the check that makes the boundary audit evidence rather than a
    restatement.
    """
    state = _position()
    observer = state.acting_player
    leaked = _leaking_builder(state, observer)

    report = pc.observer_safety_probe(state, observer, leaked, builder=_leaking_builder)
    assert not report["safe"]
    assert any("privileged truth" in problem for problem in report["problems"])
    assert report["entries_sensitive_to_hidden_truth"] > 0
    # The frozen builder passes the identical check on the identical position.
    assert pc.observer_safety_probe(state, observer, build_observation(state, observer))["safe"]


def test_the_probe_detects_a_hand_edited_model_input():
    """The other leak path: the builder is clean, the array handed on is not."""
    state = _position()
    observer = state.acting_player
    report = pc.observer_safety_probe(state, observer, _leaking_builder(state, observer))
    assert not report["safe"]
    assert any("observer-safe observation" in problem for problem in report["problems"])


def test_the_probe_detects_an_input_that_is_not_the_declared_observation():
    state = _position()
    observer = state.acting_player
    other = build_observation(state, 1 - observer)
    report = pc.observer_safety_probe(state, observer, other)
    assert not report["safe"]
    assert any("observer-safe observation" in problem for problem in report["problems"])


def test_collected_games_pass_the_boundary_audit_in_flight(participants):
    runner = pc.play_game(_scheduled("current"), participants, observer_probe_plies=6)
    assert len(runner.observer_probes) == 6
    assert all(probe["safe"] for probe in runner.observer_probes)


# ---------------------------------------------------------------------------
# Batching changes throughput, never identity
# ---------------------------------------------------------------------------


def test_batched_collection_reproduces_serial_collection_exactly(participants):
    ids = [
        phase9_game_id("canonical", 1, bucket, ordinal)
        for bucket, ordinal in (("current", 0), ("historical", 0), ("rule", 0))
    ]
    batched = {runner.game_id: runner for runner in pc.collect_games(ids, participants, games_in_flight=3)}
    assert len(batched) == len(ids)
    for game_id in ids:
        serial = pc.play_game(rebuild_scheduled_game(game_id), participants).record
        parallel = batched[game_id].record
        assert parallel.actions == serial.actions
        assert parallel.terminal_result == serial.terminal_result
        assert parallel.decisions == serial.decisions


# ---------------------------------------------------------------------------
# The iteration driver: resume and seal
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_iteration(monkeypatch):
    """Shrink `canonical` iteration 1 to four games, one per bucket.

    The driver's contract — reconcile, subtract, regenerate, seal — is about
    the *set* of scheduled ids, not how many there are, so a four-game
    iteration exercises it exactly and affordably.
    """
    ids = tuple(
        phase9_game_id("canonical", 1, bucket, 0)
        for bucket in ("current", "historical", "rule", "stress")
    )
    monkeypatch.setattr(store, "iteration_game_ids", lambda namespace, iteration: ids)
    return ids


def test_a_full_iteration_collects_and_seals(tmp_path, participants, tiny_iteration):
    summary = pc.collect_iteration(
        tmp_path, "canonical", 1, participants, games_in_flight=4, **COLLECT_KWARGS
    )
    assert summary["games_collected"] == len(tiny_iteration)
    assert summary["sealed"], summary["seal"]["problems"]
    assert summary["seal"]["behavior_snapshot_identities"] == ["B001"]
    assert set(summary["bucket_counts"]) == {"current", "historical", "rule", "stress"}
    assert store.read_iteration_state(tmp_path, "canonical", 1)["state"] == "SEALED"

    # Re-running a sealed iteration plays nothing and changes nothing.
    again = pc.collect_iteration(tmp_path, "canonical", 1, participants, **COLLECT_KWARGS)
    assert again["already_sealed"]
    assert again["games_collected"] == 0
    assert again["sealed_rollout_digest"] == summary["sealed_rollout_digest"]


def test_a_crashed_collection_resumes_to_the_same_sealed_digest(
    tmp_path, participants, tiny_iteration
):
    """The property the whole store exists to provide."""
    clean = tmp_path / "clean"
    crashed = tmp_path / "crashed"

    reference = pc.collect_iteration(
        clean, "canonical", 1, participants, games_in_flight=4, **COLLECT_KWARGS
    )
    assert reference["sealed"]

    class Boom(RuntimeError):
        pass

    calls = {"n": 0}

    def crash_hook(stage, _writer):
        if stage != "after_metadata":
            return
        calls["n"] += 1
        if calls["n"] == 2:  # one game committed, the next dies before its commit
            raise Boom("collection crash")

    with pytest.raises(Boom):
        pc.collect_iteration(
            crashed,
            "canonical",
            1,
            participants,
            games_in_flight=1,
            crash_hook=crash_hook,
            **COLLECT_KWARGS,
        )
    partial = store.Phase9RolloutReader(crashed, "canonical", 1)
    assert 0 < len(partial) < len(tiny_iteration)

    # A different worker topology on resume, as the contract explicitly allows.
    resumed = pc.collect_iteration(
        crashed, "canonical", 1, participants, games_in_flight=4, **COLLECT_KWARGS
    )
    assert resumed["sealed"], resumed["seal"]["problems"]
    assert resumed["games_already_committed"] == len(partial)
    assert resumed["sealed_rollout_digest"] == reference["sealed_rollout_digest"]

    # And byte-for-byte, not merely digest-for-digest.
    clean_reader = store.Phase9RolloutReader(clean, "canonical", 1)
    crashed_reader = store.Phase9RolloutReader(crashed, "canonical", 1)
    for game_id in clean_reader.game_ids:
        assert crashed_reader.read_payload(game_id) == clean_reader.read_payload(game_id)


def test_a_committed_game_is_never_regenerated(tmp_path, participants, tiny_iteration):
    pc.collect_iteration(
        tmp_path, "canonical", 1, participants, limit=2, seal=False, **COLLECT_KWARGS
    )
    first = store.Phase9RolloutReader(tmp_path, "canonical", 1)
    committed = {game_id: first.read_payload(game_id) for game_id in first.game_ids}

    summary = pc.collect_iteration(tmp_path, "canonical", 1, participants, **COLLECT_KWARGS)
    assert summary["games_collected"] == len(tiny_iteration) - len(committed)

    second = store.Phase9RolloutReader(tmp_path, "canonical", 1)
    for game_id, payload in committed.items():
        assert second.read_payload(game_id) == payload


def test_resuming_with_a_different_inference_shape_is_refused(
    tmp_path, participants, tiny_iteration
):
    """Two shapes agree to 1e-4 but not to the byte, so a resume may not switch."""
    pc.collect_iteration(
        tmp_path, "canonical", 1, participants, limit=1, seal=False, **COLLECT_KWARGS
    )
    other = pc.SnapshotResolver(device="cpu", inference_batch_shape=8).resolve(
        ANCHOR_CHECKPOINT,
        logical_identity="B001",
        policy_token="phase9_behavior_v1|ns=canonical|B001",
        expected_sha256=ANCHOR_SHA256,
    )
    mismatched = pc.IterationParticipants(
        behavior=other, historical=dict(participants.historical)
    )
    with pytest.raises(pc.Phase9CollectorError, match="would not converge"):
        pc.collect_iteration(tmp_path, "canonical", 1, mismatched, **COLLECT_KWARGS)


def test_resuming_under_a_different_behavior_snapshot_is_refused(
    tmp_path, participants, tiny_iteration
):
    """One iteration, one behavior identity — enforced before a byte is written."""
    pc.collect_iteration(
        tmp_path, "canonical", 1, participants, limit=1, seal=False, **COLLECT_KWARGS
    )
    other = pc.SnapshotResolver(device="cpu", inference_batch_shape=2).resolve(
        UNTRAINED_CHECKPOINT,
        logical_identity="B001",
        policy_token="phase9_behavior_v1|ns=canonical|B001",
        expected_sha256=UNTRAINED_SHA256,
    )
    with pytest.raises(pc.Phase9CollectorError, match="would not converge"):
        pc.collect_iteration(
            tmp_path,
            "canonical",
            1,
            pc.IterationParticipants(behavior=other, historical=dict(participants.historical)),
            **COLLECT_KWARGS,
        )


def test_no_optimizer_state_exists_anywhere_in_the_collection_path(participants):
    """The collector may not train, and the snapshot may not drift while it runs."""
    before = participants.behavior.loaded_state_dict_digest
    pc.play_game(_scheduled("current"), participants)
    participants.behavior.assert_frozen()
    from stratego.training.phase9_behavior import state_dict_digest

    assert state_dict_digest(participants.behavior.model) == before
