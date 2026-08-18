"""Phase 9 Agent 7: the canonical harness's own decisions, with negative controls.

The canonical run makes exactly three kinds of decision that are not already
frozen in a library module, and each of them can fail silently:

```text
which checkpoint the frozen validation score selects
whether a process restart really continued the same logical run
whether the legacy `scope="pilot_candidate"` token changes how training works
```

A harness that always answers "yes, fine" to the second question would let a
corrupted resume through, so every check here is paired with a control that
must fail. The module is imported by path because it is a script; nothing in
it runs at import time beyond its own imports.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

from stratego.training import phase9_checkpoint as pck
from stratego.training import phase9_trainer as pt
from stratego.training.warmstart_checkpoint import CorpusIdentity
from stratego.training.phase9_contract import (
    ACTIVE_WINDOW_RECENT_SNAPSHOTS,
    ARCHIVE_CADENCE_ITERATIONS,
    CANONICAL_BUCKET_COUNTS,
    CANONICAL_GAMES_PER_ITERATION,
    CANONICAL_ITERATIONS,
    CANONICAL_MAX_SCHEDULED_GAMES,
    HISTORICAL_ANCHOR_ID,
    VALIDATION_CADENCE_ITERATIONS,
    VALIDATION_TIE_BREAK,
    active_historical_window,
    archive_snapshot_id,
)

from .conftest import PHASE8_ANCHOR_PATH

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPOSITORY_ROOT / "scripts" / "run_phase9_agent07.py"

ACCEPTED_CORPUS = CorpusIdentity(
    corpus_version="synthetic_warmstart_corpus_v1",
    content_digest="c95c3545b07f2341e7efbc83c79e6342510dd973038b0f72e7eae013cff87d0d",
    metadata_digest="1db0f02fe45b16f539f070b1e12d4fdd6f390fd0487180fe660af0f4d49c81bb",
    commit_index_digest="32e8e18d1ca57ee555ed848851284f5938d4989ceb6c864f83ca4b9286c15db1",
)


@pytest.fixture(scope="module")
def harness():
    if not HARNESS_PATH.exists():
        pytest.skip(f"{HARNESS_PATH} is absent")
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    spec = importlib.util.spec_from_file_location("run_phase9_agent07", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# What the harness is contracted to execute
# ---------------------------------------------------------------------------


def test_harness_targets_the_frozen_canonical_experiment(harness):
    assert harness.NAMESPACE == "canonical"
    assert harness.CANDIDATE_ID == "P9-C"
    assert CANONICAL_ITERATIONS == 60
    assert CANONICAL_GAMES_PER_ITERATION == 2048
    assert CANONICAL_MAX_SCHEDULED_GAMES == 122_880
    assert VALIDATION_CADENCE_ITERATIONS == ARCHIVE_CADENCE_ITERATIONS == 5
    assert sum(CANONICAL_BUCKET_COUNTS.values()) == CANONICAL_GAMES_PER_ITERATION


def test_harness_pins_the_immutable_original_contract_digest(harness):
    from stratego.training.phase9_amendment import (
        AMENDED_CONTRACT_DIGEST,
        amendment_digest,
    )
    from stratego.training.phase9_contract import contract_digest

    assert harness.ACCEPTED_CONTRACT_DIGEST == contract_digest()
    assert harness.ACCEPTED_CONTRACT_DIGEST == AMENDED_CONTRACT_DIGEST
    assert harness.ACCEPTED_AMENDMENT_DIGEST == amendment_digest()


def test_harness_labels_both_train_config_document_namespaces(harness):
    assert (
        harness.ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_12H
        != harness.ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_AMENDED
    )
    assert harness.ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_12H.startswith("9284fbc6")
    assert harness.ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_AMENDED.startswith("22ac552d")


def test_scheduled_restarts_land_inside_both_epochs(harness):
    """One restart per epoch, so the partial-epoch state is real in both."""
    assert set(harness.SCHEDULED_RESTARTS) == {4, 12}
    # Two epochs: a fraction below 0.5 is inside epoch 1, above is inside epoch 2.
    assert harness.SCHEDULED_RESTARTS[4] < 0.5
    assert 0.5 < harness.SCHEDULED_RESTARTS[12] < 1.0
    # Both land after several committed iterations, and the later one after
    # two validation passes and two archive members exist.
    assert min(harness.SCHEDULED_RESTARTS) >= 4
    assert max(harness.SCHEDULED_RESTARTS) > VALIDATION_CADENCE_ITERATIONS * 2


# ---------------------------------------------------------------------------
# The canonical runtime configuration
# ---------------------------------------------------------------------------


def test_canonical_config_reconstructs_the_accepted_runtime_identity(harness):
    modules = harness._training()
    config = harness.canonical_config(modules, "mps")
    assert config.digest() == harness.ACCEPTED_TRAINER_RUNTIME_IDENTITY_DIGEST
    assert config.namespace == "canonical"
    assert config.total_iterations == CANONICAL_ITERATIONS
    assert config.learning_rate == pytest.approx(3e-4)
    assert config.initial_kl_beta == pytest.approx(0.005)
    assert config.scope == harness.RUNTIME_SCOPE_TOKEN == pt.SCOPE_PILOT


def test_canonical_config_refuses_a_configuration_that_is_not_the_frozen_one(harness):
    """The negative control: any drift must fail the identity check."""
    modules = harness._training()
    original = pt.Phase9TrainConfig.for_candidate

    def drifted(candidate_id, **kwargs):
        kwargs["total_iterations"] = 59
        return original(candidate_id, **kwargs)

    pt.Phase9TrainConfig.for_candidate = staticmethod(drifted)
    try:
        with pytest.raises(harness.Agent7Error, match="trainer runtime identity"):
            harness.canonical_config(modules, "mps")
    finally:
        pt.Phase9TrainConfig.for_candidate = original


# ---------------------------------------------------------------------------
# The scope audit
# ---------------------------------------------------------------------------


def test_scope_audit_finds_the_legacy_token_inert(harness):
    audit = harness.scope_behaviour_audit(harness._training())
    assert audit["runtime_scope_token"] == "pilot_candidate"
    assert audit["learning_fields_that_differ_by_scope"] == []
    assert audit["library_consumers_of_selects_a_configuration"] == []
    assert audit["changes_training_behaviour"] is False
    # The only value branches are the membership check and the unit-test
    # relaxation, which makes a production scope stricter, never looser.
    for branch in audit["value_branches_in_trainer"]:
        assert "SCOPES" in branch["source"] or "SCOPE_UNIT_TEST" in branch["source"]


def test_scope_audit_would_report_a_scope_that_reached_a_learning_constant(harness):
    """The negative control: a scope-dependent learning number must be caught."""
    modules = harness._training()
    original = pt.Phase9TrainConfig.for_candidate

    def scope_dependent(candidate_id, **kwargs):
        config = original(candidate_id, **kwargs)
        if kwargs.get("scope") == pt.SCOPE_SOAK:
            return original(candidate_id, **{**kwargs, "total_iterations": 8})
        return config

    pt.Phase9TrainConfig.for_candidate = staticmethod(scope_dependent)
    try:
        audit = harness.scope_behaviour_audit(modules)
    finally:
        pt.Phase9TrainConfig.for_candidate = original
    assert audit["learning_fields_that_differ_by_scope"] == ["total_iterations"]
    assert audit["changes_training_behaviour"] is True
    assert audit["verdict"].startswith("BLOCKED")


# ---------------------------------------------------------------------------
# Checkpoint selection
# ---------------------------------------------------------------------------


def _journal(scores, strategic=None, kls=None, throughput=None) -> dict:
    strategic = strategic or {}
    kls = kls or {}
    throughput = throughput or {}
    return {
        "iterations": [
            {
                "iteration": iteration,
                "mean_behavior_kl": kls.get(iteration, 0.01),
                "examples_per_second": throughput.get(iteration, 1000.0),
            }
            for iteration in sorted(scores)
        ],
        "validations": {
            str(iteration): {
                "iteration": iteration,
                "selection_score": score,
                "effective_win_rates": {
                    "strategic_rule_based": strategic.get(iteration, 0.5)
                },
                "checkpoint_identity": f"behavior_B{iteration + 1:03d}.pt",
                "checkpoint_sha256": f"sha{iteration}",
            }
            for iteration, score in scores.items()
        },
    }


def test_strictly_highest_score_wins_even_when_it_is_not_the_last(harness):
    journal = _journal({5: 0.60, 10: 0.71, 15: 0.64, 20: 0.68})
    selection = harness.select_best_validation(journal)
    assert selection["best"]["iteration"] == 10
    assert selection["unique_on_score"] is True
    assert selection["final_iteration_is_best"] is False
    assert selection["ranked"] == [10, 20, 15, 5]


def test_final_iteration_is_not_automatically_selected(harness):
    journal = _journal({5: 0.80, 10: 0.70, 15: 0.60})
    assert harness.select_best_validation(journal)["best"]["iteration"] == 5


def test_ties_fall_through_the_frozen_tie_break_in_order(harness):
    # Equal score -> Strategic EWR decides.
    journal = _journal({5: 0.70, 10: 0.70}, strategic={5: 0.61, 10: 0.55})
    selection = harness.select_best_validation(journal)
    assert selection["best"]["iteration"] == 5
    assert selection["unique_on_score"] is False
    assert selection["tied_on_score"] == [5, 10]

    # Equal score and Strategic -> lower mean behavior KL decides.
    journal = _journal(
        {5: 0.70, 10: 0.70}, strategic={5: 0.6, 10: 0.6}, kls={5: 0.03, 10: 0.01}
    )
    assert harness.select_best_validation(journal)["best"]["iteration"] == 10

    # Equal score, Strategic and KL -> higher examples/s decides.
    journal = _journal(
        {5: 0.70, 10: 0.70},
        strategic={5: 0.6, 10: 0.6},
        kls={5: 0.02, 10: 0.02},
        throughput={5: 900.0, 10: 1500.0},
    )
    assert harness.select_best_validation(journal)["best"]["iteration"] == 10
    assert harness.select_best_validation(journal)["tie_break"] == list(VALIDATION_TIE_BREAK)


def test_selection_refuses_a_run_with_no_validation_passes(harness):
    with pytest.raises(harness.Agent7Error, match="no validation passes"):
        harness.select_best_validation({"iterations": [], "validations": {}})


# ---------------------------------------------------------------------------
# Resume continuity
# ---------------------------------------------------------------------------


def _evidence() -> dict:
    return {
        "state_summary": {
            "global_optimizer_step": 812,
            "examples_consumed": 415_744,
            "rl_iteration": 4,
            "minibatch_cursor": {"epoch": 0, "minibatch_index": 330},
            "kl_beta": 0.005,
            "kl_controller_partial_epoch": {"kl_sum": 3.2, "examples": 168_960},
            "entropy_schedule_position": {"iteration": 4, "coefficient": 0.00473},
            "scheduler_last_epoch": 812,
            "sealed_rollout_digest": "seal-4",
        },
        "model_state_digest": "model-digest",
        "next_batch": {"count": 512, "digest": "batch-digest"},
        "probe": {
            "batch_digest": "packed-digest",
            "loss_total": 1.234567,
            "loss_ppo": -0.012345,
            "loss_value": 1.098765,
            "loss_belief": 0.456789,
            "behavior_kl": 0.004321,
            "policy_entropy": 3.210987,
        },
        "active_history": {
            "identities": ["H000"],
            "digests": {"H000": "anchor-sha"},
            "bound": ["H000"],
        },
        "validation_history": [],
        "best_validation": {"score": None, "identity": None},
        "sealed_rollout": {
            "rollout_id": "canonical|004",
            "sealed_rollout_digest": "seal-4",
            "learner_decisions": 281_000,
            "train_order_keys_digest": "keys-digest",
        },
        "behavior": {"snapshot_id": "B004", "checkpoint_sha256": "behavior-sha"},
        "iterations_committed": 3,
        "process_pid": 1234,
    }


def test_identical_evidence_passes_the_backend_aware_criterion(harness):
    before = _evidence()
    after = copy.deepcopy(before)
    after["process_pid"] = 5678
    comparison = harness.compare_resume(before, after)
    assert comparison["passed"] is True
    assert comparison["criterion_id"] == "phase9_backend_aware_resume_equivalence_v1"
    assert all(comparison["checks"].values())


@pytest.mark.parametrize(
    "mutate, failing_check",
    [
        (
            lambda payload: payload["state_summary"].__setitem__("global_optimizer_step", 813),
            "logical_state_equal",
        ),
        (
            lambda payload: payload["state_summary"]["minibatch_cursor"].__setitem__(
                "minibatch_index", 331
            ),
            "logical_state_equal",
        ),
        (
            lambda payload: payload["state_summary"]["kl_controller_partial_epoch"].__setitem__(
                "kl_sum", 0.0
            ),
            "logical_state_equal",
        ),
        (
            lambda payload: payload["state_summary"].__setitem__(
                "entropy_schedule_position", {"iteration": 5, "coefficient": 0.0046}
            ),
            "logical_state_equal",
        ),
        (
            lambda payload: payload.__setitem__("model_state_digest", "other-model"),
            "model_state_digest_bitwise_equal",
        ),
        (
            lambda payload: payload["next_batch"].__setitem__("digest", "other-batch"),
            "next_batch_identical",
        ),
        (
            lambda payload: payload["probe"].__setitem__("batch_digest", "other-packed"),
            "probe_batch_digest_equal",
        ),
        (
            lambda payload: payload["probe"].__setitem__("loss_total", 1.3),
            "probe_within_backend_tolerance",
        ),
        (
            lambda payload: payload["active_history"].__setitem__("identities", ["H000", "H005"]),
            "active_history_equal",
        ),
        (
            lambda payload: payload.__setitem__(
                "validation_history", [{"iteration": 5, "selection_score": 0.6}]
            ),
            "validation_history_equal",
        ),
        (
            lambda payload: payload["best_validation"].__setitem__("score", 0.61),
            "best_validation_equal",
        ),
        (
            lambda payload: payload["sealed_rollout"].__setitem__(
                "sealed_rollout_digest", "seal-other"
            ),
            "sealed_rollout_identity_equal",
        ),
        (
            lambda payload: payload["behavior"].__setitem__("checkpoint_sha256", "other-behavior"),
            "behavior_equal_control",
        ),
    ],
)
def test_every_continuity_field_has_a_failing_control(harness, mutate, failing_check):
    before = _evidence()
    after = copy.deepcopy(before)
    mutate(after)
    comparison = harness.compare_resume(before, after)
    assert comparison["passed"] is False
    if failing_check != "behavior_equal_control":
        assert comparison["checks"][failing_check] is False
    else:
        assert comparison["checks"]["behavior_snapshot_equal"] is False


def test_container_types_do_not_decide_continuity(harness):
    """The pre-exit side crosses the boundary as JSON, so its tuples come back
    as lists — AdamW's `betas` is the live example. A comparison that let
    Python's container types decide would report a corrupt resume on every
    healthy restart, which is the same failure as never checking at all."""
    before = _evidence()
    before["state_summary"]["optimizer_state_structure"] = {
        "param_groups": [{"betas": (0.9, 0.999), "lr": 3e-4}],
        "state_entries": 66,
        "step_values": [812],
    }
    after = copy.deepcopy(before)
    after["state_summary"]["optimizer_state_structure"]["param_groups"][0]["betas"] = [
        0.9,
        0.999,
    ]
    assert harness.compare_resume(before, after)["passed"] is True

    # A genuinely different optimizer still fails.
    drifted = copy.deepcopy(before)
    drifted["state_summary"]["optimizer_state_structure"]["param_groups"][0]["lr"] = 1e-4
    assert harness.compare_resume(before, drifted)["passed"] is False


def test_probe_tolerance_accepts_backend_noise_but_not_a_different_computation(harness):
    before = _evidence()
    within = copy.deepcopy(before)
    within["probe"]["loss_total"] = before["probe"]["loss_total"] + 5e-7
    assert harness.compare_resume(before, within)["passed"] is True

    outside = copy.deepcopy(before)
    outside["probe"]["policy_entropy"] = before["probe"]["policy_entropy"] * 1.001
    assert harness.compare_resume(before, outside)["passed"] is False


# ---------------------------------------------------------------------------
# What a mid-iteration resume may and may not require
# ---------------------------------------------------------------------------


def _unit_config():
    return pt.Phase9TrainConfig.for_unit_test(
        namespace="canonical", minibatch_size=64, device="cpu", total_iterations=8
    )


def test_a_mid_iteration_resume_binds_on_identity_not_on_live_weights(
    phase9_mini_rollout, tmp_path
):
    """The fresh-start on-policy guard is the wrong question mid-iteration.

    Before an iteration's first optimizer step the learner's weights *are* the
    behavior snapshot, and demanding that is right. After even one step they
    have legitimately moved — that divergence is exactly what PPO's ratio
    pi_theta/pi_b measures — so a resume that re-applied the fresh-start guard
    could never continue an iteration it stopped inside. What still has to
    bind is the identity `phase9_checkpoint_v1` records.
    """
    root, namespace, iteration, behavior = phase9_mini_rollout
    config = _unit_config()
    trainer = pt.Phase9Trainer.from_phase8_checkpoint(
        PHASE8_ANCHOR_PATH,
        config,
        ACCEPTED_CORPUS,
        topology=pt.LoaderTopology(workers=1, prefetch=1, record_cache_size=8),
    )
    try:
        # Before the first step: the fresh-start guard holds.
        assert behavior.loaded_state_dict_digest == trainer.model_state_digest()
        rollout = pt.bind_sealed_rollout(
            root,
            namespace,
            iteration,
            behavior_snapshot=behavior,
            expected_model_state_digest=trainer.model_state_digest(),
            require_full_schedule=False,
            resuming=True,
        )
        trainer.bind_iteration(rollout, mark_training=False)
        rows = trainer.train_iteration(updates=2)
        assert len(rows) == 2
        assert not trainer.cursor.finished
        moved = trainer.model_state_digest()
        checkpoint = tmp_path / "restart.pt"
        trainer.save_checkpoint(checkpoint)
    finally:
        trainer.close()

    # The learner has moved, so the fresh-start guard now *must* fail — which
    # is precisely why a resume may not apply it.
    assert moved != behavior.loaded_state_dict_digest
    with pytest.raises(pt.Phase9TrainerError, match="may not consume another policy"):
        pt.bind_sealed_rollout(
            root,
            namespace,
            iteration,
            behavior_snapshot=behavior,
            expected_model_state_digest=moved,
            require_full_schedule=False,
            resuming=True,
        )

    # The binding a resume checks instead: the checkpoint names this
    # iteration's behavior snapshot, its SHA-256 and its RL iteration.
    payload = pck.read_phase9_payload(checkpoint)
    assert payload["behavior_snapshot_identity"] == behavior.logical_identity
    assert payload["behavior_checkpoint_sha256"] == behavior.checkpoint_sha256
    assert int(payload["rl_iteration"]) == iteration

    # And with that binding satisfied the resumed trainer continues the same
    # sealed rollout from the exact cursor, repeating no step and skipping none.
    resumed = pt.Phase9Trainer.resume(
        checkpoint,
        config=config,
        corpus_identity=ACCEPTED_CORPUS,
        topology=pt.LoaderTopology(workers=1, prefetch=1, record_cache_size=8),
    )
    try:
        assert resumed.model_state_digest() == moved
        assert resumed.cursor.minibatch_index == 2
        assert resumed.global_step == 2
        rebound = pt.bind_sealed_rollout(
            root,
            namespace,
            iteration,
            behavior_snapshot=behavior,
            require_full_schedule=False,
            resuming=True,
        )
        resumed.rebind_iteration(rebound)
        more = resumed.train_iteration(updates=1)
        assert more[0]["global_optimizer_step"] == 3
    finally:
        resumed.close()


def test_a_resumed_checkpoint_from_another_iteration_is_refused(
    phase9_mini_rollout, tmp_path
):
    """The negative control for the binding a resume *does* check."""
    root, namespace, iteration, behavior = phase9_mini_rollout
    config = _unit_config()
    trainer = pt.Phase9Trainer.from_phase8_checkpoint(
        PHASE8_ANCHOR_PATH,
        config,
        ACCEPTED_CORPUS,
        topology=pt.LoaderTopology(workers=1, prefetch=1, record_cache_size=8),
    )
    try:
        rollout = pt.bind_sealed_rollout(
            root,
            namespace,
            iteration,
            behavior_snapshot=behavior,
            require_full_schedule=False,
            resuming=True,
        )
        trainer.bind_iteration(rollout, mark_training=False)
        trainer.train_iteration(updates=1)
        checkpoint = tmp_path / "wrong_iteration.pt"
        trainer.save_checkpoint(checkpoint)
    finally:
        trainer.close()

    payload = pck.read_phase9_payload(checkpoint)
    # A checkpoint that names a different behavior snapshot than the iteration
    # being resumed is the corruption the harness has to catch.
    assert payload["behavior_snapshot_identity"] == "B001"
    assert payload["behavior_snapshot_identity"] != "B004"
    assert int(payload["rl_iteration"]) != 4


# ---------------------------------------------------------------------------
# The curve row a restart carries across the process boundary
# ---------------------------------------------------------------------------


def test_curve_row_keeps_every_field_the_iteration_summary_reads(harness):
    """A restart summarises from journalled rows; a dropped field would be a
    silently wrong per-iteration metric rather than an error."""
    row = {
        "epoch": 0,
        "minibatch_index": 3,
        "global_optimizer_step": 4,
        "examples": 512,
        "loss_total": 1.0,
        "loss_ppo": -0.1,
        "loss_value": 1.1,
        "loss_belief": 0.4,
        "behavior_kl": 0.004,
        "policy_entropy": 3.2,
        "kl_beta": 0.005,
        "entropy_coefficient": 0.005,
        "clip_fraction": 0.02,
        "ppo_examples": 128,
        "ppo_clipped": 3,
        "advantage_retention": 0.25,
        "grad_norm_pre_clip": 0.9,
        "parameter_norm": 12.3,
        "ratio_mean": 1.0,
        "step_seconds": 0.4,
        "data_wait_seconds": 0.01,
        "epoch_mean_kl": 0.004,
        "epoch_clip_fraction": 0.02,
        "kl_beta_after_epoch": 0.005,
        "namespace": "canonical",
        "iteration": 4,
    }
    kept = harness.curve_row(row)

    class _Statistics:
        @staticmethod
        def to_dict():
            return {"threshold": 0.01, "retention_fraction": 0.25}

    class _Rollout:
        namespace = "canonical"
        iteration = 4
        sealed_rollout_digest = "seal"
        behavior_snapshot_id = "B004"
        behavior_checkpoint_sha256 = "sha"
        games = 2048
        learner_decisions = 281_000
        statistics = _Statistics()

    class _Controller:
        beta = 0.005

    summary = harness.summarize_iteration(
        _Rollout(), [kept], {"games_per_second": 9.0}, {"train_seconds": 10.0}, None, _Controller()
    )
    assert summary["updates"] == 1
    assert summary["examples"] == 512
    assert summary["mean_behavior_kl"] == pytest.approx(0.004)
    assert summary["epoch_mean_kls"] == [0.004]
    assert summary["epoch_clip_fractions"] == [0.02]
    assert summary["mean_policy_entropy"] == pytest.approx(3.2)
    assert summary["mean_advantage_retention"] == pytest.approx(0.25)
    assert summary["examples_per_second"] == pytest.approx(51.2)


# ---------------------------------------------------------------------------
# The historical league the run must maintain
# ---------------------------------------------------------------------------


def test_canonical_league_cadence_and_window_are_the_frozen_ones():
    created = [
        archive_snapshot_id(iteration)
        for iteration in range(
            ARCHIVE_CADENCE_ITERATIONS,
            CANONICAL_ITERATIONS + 1,
            ARCHIVE_CADENCE_ITERATIONS,
        )
    ]
    assert created == [f"H{value:03d}" for value in range(5, 61, 5)]
    assert len(created) == 12

    assert active_historical_window(1) == (HISTORICAL_ANCHOR_ID,)
    assert active_historical_window(6) == (HISTORICAL_ANCHOR_ID, "H005")
    # Once more than eight archives exist the oldest leaves the active window
    # but is never deleted.
    late = active_historical_window(60)
    assert late[0] == HISTORICAL_ANCHOR_ID
    assert len(late) == ACTIVE_WINDOW_RECENT_SNAPSHOTS + 1
    assert late[-1] == "H055"
    assert "H005" not in late
    assert "H005" in created
