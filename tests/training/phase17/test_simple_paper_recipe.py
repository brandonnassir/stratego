"""Operator decision D10 conformance, in one place.

Every other Phase 17 test module covers the mechanics of one component. This
one covers the eight things D10 changed, end to end on a real tandem iteration,
so a reader who wants to know whether the run in front of them is the
simplified paper-shaped recipe has a single file to read.

Where a claim is already asserted in detail elsewhere, this module asserts the
live-path version of it -- the value the runner actually produced, not the
value a unit can produce in isolation. The two failure modes it is aimed at are
a recipe change that never reached the runner, and a runner that reached a
different recipe than the one its telemetry claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from stratego.engine.constants import BLUE, PIECE_COUNTS, RED
from stratego.training.phase17.runner import (
    Phase17RunnerError,
    TandemConfig,
    TandemRunner,
)
from stratego.training.phase17.setup_contract import (
    PRODUCTION_RUN_ID,
    SETUP_BEHAVIOR_KL_COEFFICIENT,
    SETUP_RECIPE_VERSION,
    setup_alpha,
)
from stratego.training.phase17.setup_episode import attach_setup_episodes
from stratego.training.phase17.setup_model import build_setup_model
from stratego.training.phase17.setup_sampling import to_engine_setup
from stratego.training.phase17.supervisor import MODE_INTEGRATION


def tiny_config(run_id: str = "RUN-TEST-D10") -> TandemConfig:
    return TandemConfig(
        run_id=run_id,
        total_iterations=20,
        move_budget=400,
        population=8,
        pool_size_per_side=16,
        setup_minibatch_episodes=4,
        move_minibatch_size=64,
    )


@pytest.fixture(scope="module")
def tandem():
    """One runner advanced until at least one real setup update has happened."""
    runner = TandemRunner(tiny_config(), supervisor_mode=MODE_INTEGRATION)
    results = []
    while len(results) < 6 or not any(
        not result.setup_update.skipped for result in results
    ):
        results.append(runner.run_iteration())
    return runner, results


# -- 1. recipe and run identity ---------------------------------------------


def test_the_recipe_identity_reaches_every_artifact(tandem):
    runner, _ = tandem
    assert SETUP_RECIPE_VERSION == "phase17_simple_paper_tandem_v1"
    assert PRODUCTION_RUN_ID == "RUN-2026-B"
    assert runner.config.document()["recipe"] == SETUP_RECIPE_VERSION
    assert runner.identity_document()["recipe"] == SETUP_RECIPE_VERSION
    assert runner.setup_trainer.state_document()["recipe"] == SETUP_RECIPE_VERSION
    assert runner.setup_config.document()["recipe"] == SETUP_RECIPE_VERSION


# -- 2. production initialization --------------------------------------------


def test_production_cannot_be_handed_a_rehearsal_setup_model():
    """D10 section 3: no setup state from a rehearsal may enter production.

    Cheap by construction -- the refusal happens before the Phase 9 loader
    runs. The positive half of this claim (Phase 9 weights in, a from-scratch
    setup model out) costs a real load and is asserted in
    `test_runner_tandem.py::test_production_starts_from_phase_9_plus_a_freshly_random_setup_model`.
    """
    rehearsed = build_setup_model(device="cpu", seed=4242)
    with pytest.raises(Phase17RunnerError, match="build its own setup model"):
        TandemRunner(tiny_config(PRODUCTION_RUN_ID), setup_model=rehearsed)


# -- 3. the fixed behavior-KL coefficient ------------------------------------


def test_the_live_update_used_the_fixed_reverse_coefficient(tandem):
    runner, results = tandem
    real = [r for r in results if not r.setup_update.skipped]
    assert real
    for result in real:
        assert result.setup_update.behavior_kl_coefficient == SETUP_BEHAVIOR_KL_COEFFICIENT
        document = result.setup_update.document()
        assert document["behavior_kl_is_adaptive"] is False
        assert "beta" not in document
        assert "beta" not in str(sorted(document))
    assert runner.setup_config.kl_direction == "reverse_current_given_behavior"


def test_the_reverse_kl_is_not_the_move_controllers_forward_kl(tandem):
    """Two KLs, two directions, two names -- and only one of them adaptive."""
    runner, _ = tandem
    assert runner.setup_config.kl_direction == "reverse_current_given_behavior"
    # The move half keeps its accepted adaptive controller, untouched.
    assert hasattr(runner.move_trainer, "controller")
    assert hasattr(runner.move_trainer.controller, "beta")
    # The setup half has no controller at all.
    assert not hasattr(runner.setup_trainer, "controller")


# -- 4. alpha on the shared global iteration ---------------------------------


@pytest.mark.parametrize("iteration", [1, 2, 640])
def test_alpha_at_the_iterations_d10_names(iteration):
    assert setup_alpha(iteration) == pytest.approx(0.1 * iteration ** -0.3)


def test_the_runner_indexes_alpha_by_the_global_iteration(tandem):
    _, results = tandem
    for result in results:
        assert result.setup_update.setup_iteration == result.iteration
        assert result.setup_update.alpha == pytest.approx(setup_alpha(result.iteration))
    # Including across a skip: alpha keeps annealing whether or not the setup
    # optimizer fired.
    assert any(result.setup_update.skipped for result in results)


# -- 5. the printed paper advantage ------------------------------------------


def test_the_advantage_is_rebuilt_from_the_recorded_behavior_fields(
    completed_episodes,
):
    """Every quantity in `delta` belongs to the snapshot that drew the episode.

    Recomputing `E[v]` or `h` from the current network would turn the PPO ratio
    into a correction against itself, so the arithmetic is checked term by term
    against the recorded arrays rather than against a re-run.
    """
    from stratego.training.phase17.setup_learning import (
        expected_value_from_wdl,
        setup_advantage,
    )

    alpha = setup_alpha(7)
    for episode in completed_episodes[:4]:
        expected = expected_value_from_wdl(episode.prefix_wdl_predictions)
        outcome_term = float(episode.outcome) - expected
        entropy_term = alpha * (
            np.asarray(episode.suffix_information_content, dtype=np.float64)
            - np.asarray(
                episode.prefix_conditional_entropy_predictions, dtype=np.float64
            )
        )
        assert np.allclose(
            setup_advantage(episode, alpha), outcome_term + entropy_term, atol=1e-5
        )


def test_the_conditional_entropy_loss_still_targets_one_tenth_of_the_information(
    setup_model, config, completed_episodes
):
    """D10 keeps `L_h` normalized even though the advantage is not."""
    from stratego.training.phase17.setup_learning import build_batch, setup_batch_loss

    batch = build_batch(completed_episodes[:8], alpha=0.1)
    expected = (
        np.asarray(completed_episodes[0].suffix_information_content, dtype=np.float64)
        * 0.1
    )
    assert np.allclose(batch.normalized_information[0].numpy(), expected, atol=1e-5)
    _, terms = setup_batch_loss(setup_model, batch, config=config)
    assert float(terms["conditional_entropy_loss"].detach()) >= 0.0


# -- 6. every completed episode, once, for five epochs -----------------------


def test_the_live_update_trained_five_epochs_on_exactly_what_arrived(tandem):
    runner, results = tandem
    real = [r for r in results if not r.setup_update.skipped]
    assert real
    for result in real:
        update = result.setup_update
        assert len(update.epochs) == 5
        assert update.episodes_consumed == 2 * result.window.games_finished
        assert result.buffer_telemetry["depth"] == 0
    buffer = runner.setup_trainer.queue
    assert buffer.enqueued_count == buffer.consumed_count > 0
    assert buffer.rejected_count == 0
    assert not runner.enqueue_rejections


def test_no_quota_warm_up_or_age_rule_survives_on_the_active_path(tandem):
    runner, _ = tandem
    for retired in (
        "budget_policy",
        "warmed_up",
        "_setup_gate",
    ):
        assert not hasattr(runner, retired), retired
    for retired in ("setup_budget", "setup_warm_up_minimum", "setup_max_age_iterations"):
        assert not hasattr(runner.config, retired), retired
    buffer = runner.setup_trainer.queue
    for retired in ("capacity", "max_age_iterations", "consume_exact", "over_age"):
        assert not hasattr(buffer, retired), retired


# -- 7. a fresh pool from the current snapshot, every iteration ---------------


def test_every_setup_the_runner_produced_is_legal_inventory_correct_and_oriented(
    tandem,
):
    runner, _ = tandem
    inventory = {piece: count for piece, count in PIECE_COUNTS.items()}
    seen = 0
    for game_runner in runner.collector.slots:
        if game_runner is None:
            continue
        for color, setup in (
            (RED, tuple(game_runner.builder.red_setup)),
            (BLUE, tuple(game_runner.builder.blue_setup)),
        ):
            assert len(setup) == 40
            counts: dict = {}
            for piece in setup:
                counts[piece] = counts.get(piece, 0) + 1
            assert counts == inventory
            seen += 1
    assert seen == 2 * sum(1 for r in runner.collector.slots if r is not None)
    assert runner.provider.legality_failures == 0
    assert runner.provider.orientation_failures == 0
    assert runner.provider.fallback_attempts == 0


def test_orientation_goes_through_the_accepted_helper_only(red_samples):
    """Canonical own-side coordinates in, engine frame out, once."""
    sample = red_samples[0]
    assert sample.engine_setup == to_engine_setup(sample.canonical_setup, int(RED))
    assert sample.orientation_rule_version == "phase15_orientation_rule_v1"


def test_every_move_the_runner_sampled_was_legal_and_from_the_current_policy(tandem):
    runner, results = tandem
    known = {entry["model_state_digest"] for entry in runner.cell.digest_history()}
    for result in results:
        for row in result.window.rows:
            assert row.sampled_action in row.legal_actions
            # `legal_mask` is over the MODEL action space, which is why the
            # sampled action carries both encodings.
            assert bool(row.legal_mask[row.sampled_action_model])
            assert row.behavior_model_state_digest in known
    ledger = runner.collector.participant_ledger()
    assert ledger["holds"]
    assert not ledger["unknown_model_states"]


# -- 8. statistical stops are warnings ---------------------------------------


def test_the_only_stops_left_are_the_integrity_family(tandem):
    runner, _ = tandem
    document = runner.supervisor.document()
    stops = {
        code
        for code, entry in document["predicates"].items()
        if entry["severity"] == "stop"
    }
    assert stops == {"I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8"}
    assert set(document["warning_codes"]) == {
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
    }
    assert not runner.supervisor.should_stop


# -- the crash-correctness the buffer exists for ------------------------------


def test_a_checkpoint_taken_with_a_pending_outcome_neither_loses_nor_duplicates_it(
    tmp_path, red_samples, blue_samples
):
    """The one job D10 leaves the buffer.

    A checkpoint is normally taken between iterations, when the buffer is
    empty. This drives the case the buffer is persisted for anyway: an outcome
    that has arrived and has not yet been trained on when the process stops.
    """
    from stratego.training.phase17.checkpoint import (
        read_joint_checkpoint,
        write_joint_checkpoint,
    )

    config = tiny_config("RUN-TEST-D10-CRASH")
    runner = TandemRunner(config, supervisor_mode=MODE_INTEGRATION)
    runner.run_iteration()

    pending = attach_setup_episodes(
        red_samples[0], blue_samples[0], run_id=config.run_id, game_id="pending-game"
    ).complete("red_win")
    for episode in pending:
        runner._enqueue(episode)
    assert len(runner.setup_trainer.queue) == 2

    payload = runner.capture(
        checkpoint_generation=1,
        parent_checkpoint_identity={},
        config_digest="cfg",
        source_digest="src",
        run_digest="run",
        telemetry_position={
            "path": str(tmp_path / "t.jsonl"),
            "records": 0,
            "offset": 0,
            "last_record_digest": None,
        },
        next_export_boundary_seconds=1800.0,
    )
    write_joint_checkpoint(payload, tmp_path / "joint.pt")
    reread = read_joint_checkpoint(
        tmp_path / "joint.pt",
        run_id=config.run_id,
        config_digest="cfg",
        source_digest="src",
    )

    resumed = TandemRunner(config, supervisor_mode=MODE_INTEGRATION)
    report = resumed.restore(reread)
    assert report["completed_setup_buffer_depth"] == 2
    restored = list(resumed.setup_trainer.queue._queue)
    assert [(e.game_id, int(e.color), e.outcome) for e in restored] == [
        (e.game_id, int(e.color), e.outcome) for e in pending
    ]
    # And the next iteration consumes them exactly once.
    result = resumed.run_iteration()
    assert result.setup_update.episodes_consumed >= 2
    assert len(resumed.setup_trainer.queue) == 0


def test_the_production_run_id_refuses_a_foreign_run_checkpoint(tmp_path):
    """A `RUN-2026-A` rehearsal checkpoint cannot be resumed into production."""
    from stratego.training.phase17.checkpoint import (
        Phase17CheckpointError,
        read_joint_checkpoint,
        write_joint_checkpoint,
    )

    config = tiny_config("RUN-2026-A")
    runner = TandemRunner(config, supervisor_mode=MODE_INTEGRATION)
    runner.run_iteration()
    payload = runner.capture(
        checkpoint_generation=1,
        parent_checkpoint_identity={},
        config_digest="cfg",
        source_digest="src",
        run_digest="run",
        telemetry_position={
            "path": str(tmp_path / "t.jsonl"),
            "records": 0,
            "offset": 0,
            "last_record_digest": None,
        },
        next_export_boundary_seconds=1800.0,
    )
    write_joint_checkpoint(payload, tmp_path / "rehearsal.pt")
    with pytest.raises(Phase17CheckpointError, match="belongs to run"):
        read_joint_checkpoint(tmp_path / "rehearsal.pt", run_id=PRODUCTION_RUN_ID)


def test_the_production_entry_point_defaults_to_the_new_lineage():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    if str(root / "scripts") not in sys.path:
        sys.path.insert(0, str(root / "scripts"))
    import run_phase17_training

    frozen = run_phase17_training.load_frozen(
        root / "reports/phase17/agent_04_schedule.json",
        root / "reports/phase17/agent_04_throughput.json",
    )
    config = run_phase17_training.build_production_config(
        frozen, run_id=PRODUCTION_RUN_ID
    )
    assert config.run_id == PRODUCTION_RUN_ID
    assert config.is_production
    assert config.total_iterations == 640
    assert config.move_budget == 65536
    assert config.pool_size_per_side == 512
    assert config.recipe == SETUP_RECIPE_VERSION
