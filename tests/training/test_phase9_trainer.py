"""Phase 9 Agent 5: `phase9_trainer_v1` — ownership, order, damping, resume.

The trainer's job is to be *refusable*: almost every test here is a negative
control that produces one specific corruption and requires the trainer to stop
rather than train through it.
"""

from __future__ import annotations

import shutil

import pytest
import torch

from stratego.training import phase9_rollout_store as store
from stratego.training import phase9_targets as targets
from stratego.training import phase9_trainer as pt
from stratego.training.phase9_contract import (
    CLIP_FRACTION_HARD_LIMIT,
    ENTROPY_COEFFICIENT_END,
    ENTROPY_COEFFICIENT_START,
    EPOCHS_PER_ROLLOUT,
    KL_BETA_MAX,
    KL_BETA_MIN,
    KL_HARD_LIMIT,
    MINIBATCH_SIZE,
    PILOT_CANDIDATES,
)
from stratego.training.warmstart_checkpoint import CorpusIdentity

from .conftest import PHASE8_ANCHOR_PATH, PHASE8_ANCHOR_SHA256

ACCEPTED_CORPUS = CorpusIdentity(
    corpus_version="synthetic_warmstart_corpus_v1",
    content_digest="c95c3545b07f2341e7efbc83c79e6342510dd973038b0f72e7eae013cff87d0d",
    metadata_digest="1db0f02fe45b16f539f070b1e12d4fdd6f390fd0487180fe660af0f4d49c81bb",
    commit_index_digest="32e8e18d1ca57ee555ed848851284f5938d4989ceb6c864f83ca4b9286c15db1",
)


@pytest.fixture
def rollout_copy(phase9_mini_rollout, tmp_path):
    """A private copy, for tests that move the iteration's state machine."""
    root, namespace, iteration, behavior = phase9_mini_rollout
    destination = tmp_path / "rollout"
    shutil.copytree(root, destination)
    return destination, namespace, iteration, behavior


def unit_config(**overrides):
    parameters = {
        "namespace": "canonical",
        "minibatch_size": 64,
        "device": "cpu",
        "total_iterations": 8,
    }
    parameters.update(overrides)
    return pt.Phase9TrainConfig.for_unit_test(**parameters)


def build_trainer(config=None, workers=1, **kwargs):
    return pt.Phase9Trainer.from_phase8_checkpoint(
        PHASE8_ANCHOR_PATH,
        config or unit_config(),
        ACCEPTED_CORPUS,
        topology=pt.LoaderTopology(workers=workers, prefetch=1, record_cache_size=8),
        **kwargs,
    )


def bind(phase9_mini_rollout, **kwargs):
    root, namespace, iteration, behavior = phase9_mini_rollout
    parameters = {"require_full_schedule": False, "resuming": True}
    parameters.update(kwargs)
    return pt.bind_sealed_rollout(root, namespace, iteration, **parameters)


# ---------------------------------------------------------------------------
# Configuration: Agent 5 selects nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate", [entry["candidate_id"] for entry in PILOT_CANDIDATES])
def test_every_frozen_candidate_builds(candidate):
    config = pt.Phase9TrainConfig.for_candidate(candidate, device="cpu")
    entry = {item["candidate_id"]: item for item in PILOT_CANDIDATES}[candidate]
    assert config.learning_rate == entry["learning_rate"]
    assert config.initial_kl_beta == entry["initial_kl_beta"]
    assert config.minibatch_size == MINIBATCH_SIZE
    assert config.epochs_per_rollout == EPOCHS_PER_ROLLOUT
    assert config.selects_a_configuration


def test_a_learning_rate_outside_the_frozen_matrix_is_refused():
    with pytest.raises(pt.Phase9TrainerError, match="six frozen candidates"):
        pt.Phase9TrainConfig(
            namespace="canonical",
            candidate_id="invented",
            scope=pt.SCOPE_PILOT,
            learning_rate=2e-4,
            initial_kl_beta=0.005,
            total_iterations=8,
        )


def test_the_soak_configuration_is_neutral_and_selects_nothing():
    config = pt.Phase9TrainConfig.for_soak(namespace="canonical", total_iterations=4)
    neutral = {item["candidate_id"]: item for item in PILOT_CANDIDATES}[
        pt.SOAK_CANDIDATE_ID
    ]
    assert config.learning_rate == neutral["learning_rate"]
    assert config.initial_kl_beta == neutral["initial_kl_beta"]
    assert config.scope == pt.SCOPE_SOAK
    assert not config.selects_a_configuration


def test_frozen_optimizer_constants_may_not_be_overridden():
    with pytest.raises(pt.Phase9TrainerError, match="minibatch_size is frozen"):
        pt.Phase9TrainConfig.for_candidate("P9-A", device="cpu").__class__(
            namespace="canonical",
            candidate_id="P9-A",
            scope=pt.SCOPE_PILOT,
            learning_rate=1e-4,
            initial_kl_beta=0.005,
            total_iterations=8,
            minibatch_size=256,
        )


def test_an_initial_beta_outside_the_clamp_is_refused():
    with pytest.raises(pt.Phase9TrainerError, match="frozen clamp"):
        unit_config(initial_kl_beta=KL_BETA_MAX * 2)


def test_config_digest_is_stable_and_sensitive():
    a = pt.Phase9TrainConfig.for_candidate("P9-A", device="cpu")
    b = pt.Phase9TrainConfig.for_candidate("P9-A", device="cpu")
    c = pt.Phase9TrainConfig.for_candidate("P9-B", device="cpu")
    assert a.digest() == b.digest()
    assert a.digest() != c.digest()


# ---------------------------------------------------------------------------
# Iteration ownership
# ---------------------------------------------------------------------------


def test_binding_verifies_the_sealed_rollout(phase9_mini_rollout):
    rollout = bind(phase9_mini_rollout)
    assert rollout.behavior_snapshot_id == "B001"
    assert rollout.behavior_checkpoint_sha256 == PHASE8_ANCHOR_SHA256
    assert rollout.learner_decisions > 0
    assert rollout.verifications["digest_recomputed_from_commits"]
    assert rollout.verifications["learner_control_mismatches"] == 0
    assert rollout.verifications["version_mismatches"] == 0
    assert rollout.rollout_id == f"phase9_rollout_v1|ns=canonical|it=001"


def test_an_unsealed_iteration_is_refused(rollout_copy):
    root, namespace, iteration, _behavior = rollout_copy
    store.write_iteration_state(root, namespace, iteration + 1, "COLLECTING")
    with pytest.raises(pt.Phase9TrainerError, match="not one of"):
        pt.bind_sealed_rollout(
            root, namespace, iteration + 1, require_full_schedule=False
        )


def test_a_digest_that_disagrees_with_the_bytes_is_refused(rollout_copy):
    root, namespace, iteration, _behavior = rollout_copy
    store.write_iteration_state(
        root, namespace, iteration, "SEALED", sealed_rollout_digest="0" * 64
    )
    with pytest.raises(pt.Phase9TrainerError, match="recomputed sealed digest"):
        pt.bind_sealed_rollout(root, namespace, iteration, require_full_schedule=False)


def test_an_incomplete_schedule_is_refused_in_production_mode(phase9_mini_rollout):
    root, namespace, iteration, _behavior = phase9_mini_rollout
    with pytest.raises(pt.Phase9TrainerError, match="bucket counts"):
        pt.bind_sealed_rollout(
            root, namespace, iteration, require_full_schedule=True, resuming=True
        )


def test_a_foreign_behavior_snapshot_is_refused(phase9_mini_rollout):
    """The snapshot handed in must be the one that collected the iteration."""
    from stratego.training import phase9_behavior as pb

    other = pb.load_behavior_snapshot(
        "checkpoints/phase8/warmstart_c1_v1_initialisation.pt",
        logical_identity="B001",
        policy_token="phase9_behavior_v1|ns=canonical|B001",
        device="cpu",
        inference_batch_shape=4,
    )
    with pytest.raises(pt.Phase9TrainerError, match="was collected under"):
        bind(phase9_mini_rollout, behavior_snapshot=other)


def test_off_policy_weights_may_not_consume_the_rollout(phase9_mini_rollout):
    """The on-policy binding: these games must have come from these weights."""
    _root, _namespace, _iteration, behavior = phase9_mini_rollout
    with pytest.raises(pt.Phase9TrainerError, match="may not consume another policy"):
        bind(
            phase9_mini_rollout,
            behavior_snapshot=behavior,
            expected_model_state_digest="deadbeef" * 8,
        )


def test_matching_weights_pass_the_on_policy_binding(phase9_mini_rollout):
    _root, _namespace, _iteration, behavior = phase9_mini_rollout
    trainer = build_trainer()
    rollout = bind(
        phase9_mini_rollout,
        behavior_snapshot=behavior,
        expected_model_state_digest=trainer.model_state_digest(),
    )
    assert rollout.verifications["on_policy_state_dict_digest"] is not None
    trainer.close()


def test_a_learner_control_that_contradicts_the_schedule_is_refused(rollout_copy):
    root, namespace, iteration, _behavior = rollout_copy
    directory = store.metadata_directory(root, namespace, iteration)
    path = next(directory.glob("*.meta.jsonl"))
    lines = path.read_text().splitlines()
    import json

    record = json.loads(lines[0])
    record["learner_control"] = "both" if record["learner_control"] != "both" else "red"
    lines[0] = json.dumps(record, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    # The metadata digest now disagrees with the commit, which the reader
    # catches first; either refusal is the point — the payload is not trainable.
    with pytest.raises((pt.Phase9TrainerError, store.Phase9RolloutStoreError)):
        pt.bind_sealed_rollout(root, namespace, iteration, require_full_schedule=False)


def test_a_bound_iteration_moves_to_training_and_back_without_touching_bytes(rollout_copy):
    root, namespace, iteration, _behavior = rollout_copy
    rollout = pt.bind_sealed_rollout(
        root, namespace, iteration, require_full_schedule=False
    )
    before = store.sealed_rollout_digest(rollout.reader.commits)
    trainer = build_trainer()
    with trainer:
        trainer.bind_iteration(rollout)
        state = store.read_iteration_state(root, namespace, iteration)
        assert state["state"] == "TRAINING"
        assert state["training_complete"] is False
        trainer.train_iteration(updates=1)
        marked = trainer.mark_iteration_trained()
    assert marked["training_complete"] is True
    after = store.sealed_rollout_digest(
        store.Phase9RolloutReader(root, namespace, iteration).commits
    )
    assert after == before


def test_a_resumed_run_may_rebind_a_training_iteration(rollout_copy):
    root, namespace, iteration, _behavior = rollout_copy
    store.write_iteration_state(
        root,
        namespace,
        iteration,
        "TRAINING",
        sealed_rollout_digest=store.sealed_rollout_digest(
            store.Phase9RolloutReader(root, namespace, iteration).commits
        ),
    )
    with pytest.raises(pt.Phase9TrainerError, match="not one of"):
        pt.bind_sealed_rollout(root, namespace, iteration, require_full_schedule=False)
    rollout = pt.bind_sealed_rollout(
        root, namespace, iteration, require_full_schedule=False, resuming=True
    )
    assert rollout.learner_decisions > 0


# ---------------------------------------------------------------------------
# Train order and the opponent-gradient guarantee
# ---------------------------------------------------------------------------


def test_the_universe_is_exactly_agent_4s_train_order(phase9_mini_rollout):
    rollout = bind(phase9_mini_rollout)
    expected = targets.train_order_keys(rollout.reader)
    assert rollout.keys == expected
    assert len(set(rollout.keys)) == len(rollout.keys)


def test_no_opponent_decision_is_a_member_of_the_universe(phase9_mini_rollout):
    """Rule, stress and historical-opponent decisions get zero gradient because
    they are never in the batch universe at all."""
    rollout = bind(phase9_mini_rollout)
    by_game = {}
    for game_id, ply in rollout.keys:
        by_game.setdefault(game_id, []).append(ply)
    checked_one_sided = 0
    for game_id, plies in by_game.items():
        record, metadata = rollout.reader.read_game(game_id)
        learners = targets.learner_players(metadata)
        for ply in plies:
            assert int(record.decisions[ply].acting_player) in learners
        if metadata["learner_control"] != "both":
            checked_one_sided += 1
            opponent_plies = {
                index
                for index, decision in enumerate(record.decisions)
                if int(decision.acting_player) not in learners
            }
            assert opponent_plies and not opponent_plies.intersection(plies)
    assert checked_one_sided >= 1


def test_a_minibatch_containing_an_opponent_decision_is_refused(phase9_mini_rollout):
    rollout = bind(phase9_mini_rollout)
    one_sided = None
    for game_id in rollout.reader.game_ids:
        record, metadata = rollout.reader.read_game(game_id)
        if metadata["learner_control"] != "both":
            learners = targets.learner_players(metadata)
            opponent = next(
                index
                for index, decision in enumerate(record.decisions)
                if int(decision.acting_player) not in learners
            )
            one_sided = (game_id, opponent)
            break
    assert one_sided is not None
    trainer = build_trainer()
    with trainer:
        trainer.bind_iteration(rollout, mark_training=False)
        with pytest.raises(pt.Phase9TrainerError, match="outside the sealed"):
            trainer._verify_batch(
                [one_sided],
                {
                    "behavior_checkpoint_sha256": (rollout.behavior_checkpoint_sha256,),
                    "rollout_ids": (rollout.rollout_id,),
                },
                rollout,
            )
    assert trainer.counters["data_mismatches"] == 1


def test_building_a_batch_from_an_opponent_ply_is_refused(phase9_mini_rollout):
    rollout = bind(phase9_mini_rollout)
    for game_id in rollout.reader.game_ids:
        record, metadata = rollout.reader.read_game(game_id)
        if metadata["learner_control"] == "both":
            continue
        learners = targets.learner_players(metadata)
        opponent = next(
            index
            for index, decision in enumerate(record.decisions)
            if int(decision.acting_player) not in learners
        )
        with pytest.raises(pt.Phase9TrainerError, match="not learner decisions"):
            pt.examples_for_keys(record, metadata, rollout.statistics, [opponent])
        return
    raise AssertionError("the fixture contains no one-sided game")


def test_minibatch_keys_follow_the_frozen_shuffle(phase9_mini_rollout):
    rollout = bind(phase9_mini_rollout)
    config = unit_config()
    cursor = targets.Phase9MinibatchCursor.start(
        namespace=rollout.namespace,
        iteration=rollout.iteration,
        sealed_rollout_digest=rollout.sealed_rollout_digest,
        total_examples=rollout.learner_decisions,
        epochs=config.epochs_per_rollout,
        minibatch_size=config.minibatch_size,
    )
    expected = targets.minibatch_keys(
        rollout.keys, rollout.namespace, rollout.iteration, 0, 0, config.minibatch_size
    )
    packed = pt.build_minibatch(expected, rollout.reader, rollout.statistics)
    assert tuple(zip(packed["game_ids"], packed["decision_indices"])) == tuple(expected)
    assert cursor.minibatches_per_epoch == len(
        targets.minibatch_slices(rollout.learner_decisions, config.minibatch_size)
    )


def test_the_final_partial_minibatch_is_consumed(phase9_mini_rollout):
    rollout = bind(phase9_mini_rollout)
    config = unit_config()
    slices = targets.minibatch_slices(rollout.learner_decisions, config.minibatch_size)
    consumed = sum(stop - start for start, stop in slices)
    assert consumed == rollout.learner_decisions
    trainer = build_trainer(config)
    with trainer:
        trainer.bind_iteration(rollout, mark_training=False)
        rows = trainer.train_iteration()
    assert len(rows) == len(slices) * config.epochs_per_rollout
    assert trainer.examples_consumed == rollout.learner_decisions * config.epochs_per_rollout


@pytest.mark.parametrize("workers", [1, 2])
def test_one_trainer_consumes_two_successive_iterations(phase9_mini_rollout, workers):
    """The canonical run's actual shape: bind, train, bind again, train again.

    The second bind must drop the first iteration's exhausted data pipeline.
    Left in place, its prefetch queue is empty and the second iteration's first
    minibatch has nothing to pop — which is invisible to any single-iteration
    test.
    """
    trainer = build_trainer(workers=workers)
    with trainer:
        for expected_iteration in (1, 2):
            rollout = bind(phase9_mini_rollout)
            # The fixture holds one sealed iteration; re-binding it twice
            # exercises the trainer transition, which is what is under test.
            trainer.bind_iteration(rollout, mark_training=False)
            rows = trainer.train_iteration(updates=2)
            assert len(rows) == 2
            assert trainer.global_step == 2 * expected_iteration
        assert trainer.counters["data_mismatches"] == 0


def test_batch_identity_does_not_depend_on_worker_count(phase9_mini_rollout):
    digests = {}
    for workers in (1, 2):
        rollout = bind(phase9_mini_rollout)
        trainer = build_trainer(workers=workers)
        with trainer:
            trainer.bind_iteration(rollout, mark_training=False)
            rows = trainer.train_iteration(updates=3, capture_batch_digests=True)
        digests[workers] = [
            (row["batch_digest"], row["loss_total"]) for row in rows
        ]
    assert digests[1] == digests[2]


# ---------------------------------------------------------------------------
# Damping, entropy, hard limits
# ---------------------------------------------------------------------------


def close_epoch(controller, *, iteration, epoch, mean_kl, examples=512):
    controller.observe(mean_kl=mean_kl, examples=examples, clipped=0, ppo_examples=examples)
    return controller.update(iteration=iteration, epoch=epoch)


def test_kl_controller_follows_the_frozen_rules():
    controller = pt.KLController(beta=0.01)
    assert close_epoch(controller, iteration=1, epoch=0, mean_kl=0.05)["beta_after"] == 0.02
    assert close_epoch(controller, iteration=1, epoch=1, mean_kl=0.001)["beta_after"] == 0.01
    unchanged = close_epoch(controller, iteration=2, epoch=0, mean_kl=0.015)
    assert unchanged["beta_after"] == 0.01
    assert unchanged["direction"] == "unchanged"
    assert len(controller.history) == 3


def test_kl_controller_clamps_in_both_directions():
    high = pt.KLController(beta=KL_BETA_MAX)
    close_epoch(high, iteration=1, epoch=0, mean_kl=0.5)
    assert high.beta == KL_BETA_MAX
    low = pt.KLController(beta=KL_BETA_MIN)
    close_epoch(low, iteration=1, epoch=0, mean_kl=0.0)
    assert low.beta == KL_BETA_MIN


def test_the_epoch_mean_is_example_weighted_over_the_whole_epoch():
    controller = pt.KLController(beta=0.01)
    controller.observe(mean_kl=0.10, examples=400, clipped=10, ppo_examples=100)
    controller.observe(mean_kl=0.02, examples=100, clipped=0, ppo_examples=50)
    assert controller.epoch_mean_kl == pytest.approx((0.10 * 400 + 0.02 * 100) / 500)
    assert controller.epoch_clip_fraction == pytest.approx(10 / 150)
    controller.update(iteration=1, epoch=0)
    assert controller.epoch_examples == 0


def test_a_partial_epoch_survives_the_checkpoint():
    """The half-measured epoch is state, not a local variable.

    A run checkpointed mid-epoch and resumed must close that epoch on the
    whole epoch's KL. If the accumulator reset on reload, the resumed run
    would damp on the post-resume half alone — and nothing at the resume
    boundary would show it, because the difference only appears one epoch
    later.
    """
    controller = pt.KLController(beta=0.01)
    controller.observe(mean_kl=0.10, examples=400, clipped=10, ppo_examples=100)
    restored = pt.KLController.from_dict(controller.to_dict())
    assert restored.epoch_kl_sum == controller.epoch_kl_sum
    assert restored.epoch_examples == controller.epoch_examples
    assert restored.epoch_clipped == controller.epoch_clipped
    assert restored.epoch_ppo_examples == controller.epoch_ppo_examples

    controller.observe(mean_kl=0.02, examples=100, clipped=0, ppo_examples=50)
    restored.observe(mean_kl=0.02, examples=100, clipped=0, ppo_examples=50)
    assert (
        restored.update(iteration=1, epoch=0)["mean_epoch_kl"]
        == controller.update(iteration=1, epoch=0)["mean_epoch_kl"]
    )


def test_a_resume_across_an_epoch_boundary_damps_identically(phase9_mini_rollout, tmp_path):
    """The end-to-end version, over a real epoch boundary.

    The uninterrupted run measures the epoch in a single call; the split run
    checkpoints in the middle of it and closes it from another process's state.
    Both must close epoch 0 on the same example-weighted mean, which is only
    true if the half-measured epoch survived the checkpoint.
    """
    config = unit_config()
    rollout = bind(phase9_mini_rollout)
    per_epoch = len(
        targets.minibatch_slices(rollout.learner_decisions, config.minibatch_size)
    )
    assert per_epoch >= 3, "the fixture must have several minibatches per epoch"
    split = per_epoch // 2

    uninterrupted = build_trainer(config)
    with uninterrupted:
        uninterrupted.bind_iteration(rollout, mark_training=False)
        uninterrupted.train_iteration(updates=per_epoch)
        expected = [dict(entry) for entry in uninterrupted.controller.history]
    assert len(expected) == 1, "one epoch boundary should have been crossed"

    first = build_trainer(config)
    with first:
        first.bind_iteration(bind(phase9_mini_rollout), mark_training=False)
        first.train_iteration(updates=split)
        first.save_checkpoint(tmp_path / "mid_epoch.pt")
        assert first.controller.epoch_examples > 0, "the epoch must be half-measured"
        assert not first.controller.history, "the boundary must still be ahead"

    resumed = pt.Phase9Trainer.resume(
        tmp_path / "mid_epoch.pt",
        config=config,
        corpus_identity=ACCEPTED_CORPUS,
        topology=pt.LoaderTopology(workers=1, prefetch=1, record_cache_size=8),
    )
    resumed.rebind_iteration(bind(phase9_mini_rollout))
    with resumed:
        resumed.train_iteration(updates=per_epoch - split)
        actual = [dict(entry) for entry in resumed.controller.history]

    assert actual == expected
    assert resumed.controller.beta == uninterrupted.controller.beta
    # The claim is only meaningful if the pre-checkpoint half was counted: the
    # closed epoch must span every learner decision, not just the resumed part.
    assert expected[0]["epoch_examples"] == rollout.learner_decisions


def test_the_controller_fires_once_per_epoch(phase9_mini_rollout):
    rollout = bind(phase9_mini_rollout)
    trainer = build_trainer()
    with trainer:
        trainer.bind_iteration(rollout, mark_training=False)
        rows = trainer.train_iteration()
    assert len(trainer.controller.history) == EPOCHS_PER_ROLLOUT
    boundaries = [row for row in rows if "epoch_mean_kl" in row]
    assert len(boundaries) == EPOCHS_PER_ROLLOUT
    assert [entry["epoch"] for entry in trainer.controller.history] == [0, 1]


def test_the_kl_hard_limit_stops_the_run(phase9_mini_rollout):
    trainer = build_trainer()
    with pytest.raises(pt.Phase9TrainerError, match="exceeds the frozen hard limit"):
        trainer._check_hard_limits(
            iteration=1, epoch=0, mean_kl=KL_HARD_LIMIT + 0.01, clip_fraction=0.0
        )
    assert trainer.counters["kl_hard_limit_breaches"] == 1
    trainer.close()


def test_the_clip_fraction_hard_limit_stops_the_run():
    trainer = build_trainer()
    with pytest.raises(pt.Phase9TrainerError, match="clip fraction"):
        trainer._check_hard_limits(
            iteration=1,
            epoch=0,
            mean_kl=0.001,
            clip_fraction=CLIP_FRACTION_HARD_LIMIT + 0.01,
        )
    assert trainer.counters["clip_fraction_hard_limit_breaches"] == 1
    trainer.close()


def test_entropy_coefficient_is_constant_within_an_iteration(phase9_mini_rollout):
    rollout = bind(phase9_mini_rollout)
    trainer = build_trainer(unit_config(total_iterations=8))
    with trainer:
        trainer.bind_iteration(rollout, mark_training=False)
        rows = trainer.train_iteration(updates=3)
    coefficients = {row["entropy_coefficient"] for row in rows}
    assert len(coefficients) == 1
    assert coefficients.pop() == pytest.approx(ENTROPY_COEFFICIENT_START)


def test_entropy_schedule_traverses_the_frozen_endpoints():
    from stratego.training.phase9_contract import entropy_coefficient

    assert entropy_coefficient(1, 8) == pytest.approx(ENTROPY_COEFFICIENT_START)
    assert entropy_coefficient(8, 8) == pytest.approx(ENTROPY_COEFFICIENT_END)
    assert entropy_coefficient(1, 60) == pytest.approx(ENTROPY_COEFFICIENT_START)
    assert entropy_coefficient(60, 60) == pytest.approx(ENTROPY_COEFFICIENT_END)


def test_the_on_policy_start_has_near_zero_kl_and_unit_ratios(phase9_mini_rollout):
    """The strongest single check that the whole target/frame chain is right.

    At the start of an iteration the learner *is* the behavior snapshot, so
    `pi_theta` must reproduce the stored `pi_b` and both the KL and the PPO
    ratio must sit at their identity values. A frame mix-up anywhere would
    move these by orders of magnitude.
    """
    rollout = bind(phase9_mini_rollout)
    trainer = build_trainer()
    with trainer:
        trainer.bind_iteration(rollout, mark_training=False)
        rows = trainer.train_iteration(updates=1)
    assert abs(rows[0]["behavior_kl"]) < 1e-5
    assert rows[0]["ratio_mean"] == pytest.approx(1.0, abs=1e-3)
    assert rows[0]["clip_fraction"] == 0.0


# ---------------------------------------------------------------------------
# Checkpoint and resume
# ---------------------------------------------------------------------------


def test_resume_restores_the_exact_logical_state(phase9_mini_rollout, tmp_path):
    rollout = bind(phase9_mini_rollout)
    config = unit_config()
    donor = build_trainer(config)
    with donor:
        donor.bind_iteration(rollout, mark_training=False)
        donor.train_iteration(updates=2)
        donor.save_checkpoint(tmp_path / "ckpt.pt")
        frozen = donor.state_summary()
        donor_rows = donor.train_iteration(updates=2, capture_batch_digests=True)
        donor_parameters = donor.parameter_snapshot()

    resumed = pt.Phase9Trainer.resume(
        tmp_path / "ckpt.pt",
        config=config,
        corpus_identity=ACCEPTED_CORPUS,
        topology=pt.LoaderTopology(workers=1, prefetch=1, record_cache_size=8),
        expected_sealed_rollout_digest=rollout.sealed_rollout_digest,
        expected_behavior_checkpoint_sha256=rollout.behavior_checkpoint_sha256,
    )
    assert resumed.state_summary()["minibatch_cursor"] == frozen["minibatch_cursor"]
    assert resumed.state_summary()["kl_beta"] == frozen["kl_beta"]
    assert resumed.state_summary()["global_optimizer_step"] == frozen["global_optimizer_step"]
    assert (
        resumed.state_summary()["optimizer_state_structure"]
        == frozen["optimizer_state_structure"]
    )
    resumed.rebind_iteration(rollout)
    with resumed:
        resumed_rows = resumed.train_iteration(updates=2, capture_batch_digests=True)
        resumed_parameters = resumed.parameter_snapshot()

    assert [row["batch_digest"] for row in resumed_rows] == [
        row["batch_digest"] for row in donor_rows
    ]
    # CPU float32 with a fixed thread count is deterministic, so the split run
    # is bitwise identical to the uninterrupted continuation.
    for name, tensor in donor_parameters.items():
        assert torch.equal(resumed_parameters[name], tensor), name


def test_resume_refuses_a_checkpoint_from_another_rollout(phase9_mini_rollout, tmp_path):
    rollout = bind(phase9_mini_rollout)
    config = unit_config()
    trainer = build_trainer(config)
    with trainer:
        trainer.bind_iteration(rollout, mark_training=False)
        trainer.train_iteration(updates=1)
        trainer.save_checkpoint(tmp_path / "ckpt.pt")
    from stratego.training.phase9_checkpoint import Phase9CheckpointIdentityError

    with pytest.raises(Phase9CheckpointIdentityError, match="sealed rollout digest"):
        pt.Phase9Trainer.resume(
            tmp_path / "ckpt.pt",
            config=config,
            corpus_identity=ACCEPTED_CORPUS,
            expected_sealed_rollout_digest="0" * 64,
        )


def test_resume_refuses_a_different_train_config(phase9_mini_rollout, tmp_path):
    rollout = bind(phase9_mini_rollout)
    trainer = build_trainer(unit_config())
    with trainer:
        trainer.bind_iteration(rollout, mark_training=False)
        trainer.train_iteration(updates=1)
        trainer.save_checkpoint(tmp_path / "ckpt.pt")
    from stratego.training.phase9_checkpoint import Phase9CheckpointIdentityError

    with pytest.raises(Phase9CheckpointIdentityError, match="train config"):
        pt.Phase9Trainer.resume(
            tmp_path / "ckpt.pt",
            config=unit_config(learning_rate=3e-4),
            corpus_identity=ACCEPTED_CORPUS,
        )


def test_rebinding_a_different_rollout_is_refused(phase9_mini_rollout, tmp_path):
    rollout = bind(phase9_mini_rollout)
    config = unit_config()
    trainer = build_trainer(config)
    with trainer:
        trainer.bind_iteration(rollout, mark_training=False)
        trainer.train_iteration(updates=1)
        trainer.save_checkpoint(tmp_path / "ckpt.pt")
    resumed = pt.Phase9Trainer.resume(
        tmp_path / "ckpt.pt", config=config, corpus_identity=ACCEPTED_CORPUS
    )
    impostor = pt.SealedRollout(
        root=rollout.root,
        namespace=rollout.namespace,
        iteration=rollout.iteration,
        sealed_rollout_digest="9" * 64,
        behavior_snapshot_id=rollout.behavior_snapshot_id,
        behavior_checkpoint_sha256=rollout.behavior_checkpoint_sha256,
        games=rollout.games,
        keys=rollout.keys,
        statistics=rollout.statistics,
        reader=rollout.reader,
    )
    with pytest.raises(pt.Phase9TrainerError, match="sealed rollout digest"):
        resumed.rebind_iteration(impostor)
    resumed.close()


def test_a_checkpoint_records_the_bound_rollout_identity(phase9_mini_rollout, tmp_path):
    from stratego.training import phase9_checkpoint as pck

    rollout = bind(phase9_mini_rollout)
    trainer = build_trainer()
    with trainer:
        trainer.bind_iteration(rollout, mark_training=False)
        trainer.train_iteration(updates=1)
        trainer.save_checkpoint(tmp_path / "ckpt.pt")
    payload = pck.read_phase9_payload(tmp_path / "ckpt.pt")
    assert payload["sealed_rollout_digest"] == rollout.sealed_rollout_digest
    assert payload["rollout_iteration_identity"] == rollout.rollout_id
    assert payload["behavior_snapshot_identity"] == rollout.behavior_snapshot_id
    assert payload["snapshot_role"] == "resume"


def test_a_trained_run_freezes_a_bindable_behavior_snapshot(phase9_mini_rollout, tmp_path):
    """The next iteration's collector needs these weights as a frozen snapshot."""
    from stratego.training import phase9_checkpoint as pck

    rollout = bind(phase9_mini_rollout)
    trainer = build_trainer()
    with trainer:
        trainer.bind_iteration(rollout, mark_training=False)
        trainer.train_iteration(updates=2)
        written = trainer.save_behavior_snapshot(
            tmp_path / "B002.pt", logical_identity="B002", rl_iteration=2
        )
        trained_digest = trainer.model_state_digest()
    snapshot = pck.bind_behavior_snapshot(
        tmp_path / "B002.pt",
        logical_identity="B002",
        namespace="canonical",
        device="cpu",
        inference_batch_shape=4,
        expected_sha256=written["sha256"],
    )
    snapshot.assert_frozen()
    assert snapshot.policy_token == "phase9_behavior_v1|ns=canonical|B002"
    # The snapshot must be the *trained* weights, not the ones it started from.
    assert snapshot.loaded_state_dict_digest == trained_digest
    assert snapshot.checkpoint_sha256 != PHASE8_ANCHOR_SHA256

    # ...and it must refuse to answer to another iteration's identity.
    with pytest.raises(pck.Phase9CheckpointError, match="payload names behavior"):
        pck.bind_behavior_snapshot(
            tmp_path / "B002.pt",
            logical_identity="B003",
            namespace="canonical",
            device="cpu",
        )


def test_a_run_may_archive_an_immutable_namespace_local_member(phase9_mini_rollout, tmp_path):
    """Agent 6's pilots need a real pilot-local `H005`; this is that path."""
    from stratego.training import phase9_checkpoint as pck

    rollout = bind(phase9_mini_rollout)
    trainer = build_trainer()
    with trainer:
        trainer.bind_iteration(rollout, mark_training=False)
        trainer.train_iteration(updates=1)
        payload = trainer.checkpoint_payload(snapshot_role="archive_member")
    member = pck.write_archive_member(
        payload, tmp_path / "archive", namespace="canonical", local_identity="H005",
        fsync=False,
    )
    assert member.qualified_identity == "canonical|H005"
    snapshot = pck.bind_archive_member(member, device="cpu", inference_batch_shape=4)
    snapshot.assert_frozen()
    assert snapshot.checkpoint_sha256 == member.checkpoint_sha256


# ---------------------------------------------------------------------------
# Counters and semantics
# ---------------------------------------------------------------------------


def test_a_clean_run_leaves_every_counter_at_zero(phase9_mini_rollout):
    rollout = bind(phase9_mini_rollout)
    trainer = build_trainer()
    with trainer:
        trainer.bind_iteration(rollout, mark_training=False)
        rows = trainer.train_iteration(updates=4)
    assert all(value == 0 for value in trainer.counters.values()), trainer.counters
    assert all(row["loss_total"] == row["loss_total"] for row in rows)


def test_trainer_semantics_states_the_frozen_constraints():
    semantics = pt.trainer_semantics()
    assert semantics["optimizer_constraints"]["minibatch_size"] == MINIBATCH_SIZE
    assert semantics["optimizer_constraints"]["epochs_per_rollout"] == EPOCHS_PER_ROLLOUT
    assert semantics["kl_controller"]["hard_limits"]["mean_epoch_kl"] == KL_HARD_LIMIT
    assert "zero" in semantics["populations"]["opponent"]
    assert semantics["architecture"] == "C1"
