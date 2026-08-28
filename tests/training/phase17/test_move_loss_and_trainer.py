"""Phase 17 Agent 2: the per-row-maskable objective and the one-epoch update."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stratego.training.phase17.move_contract import (
    BELIEF_LOSS_WEIGHT,
    MOVE_EPOCHS_PER_ITERATION,
    MOVE_LR_MAX,
    MOVE_LR_MIN,
    PHASE9_ACCEPTED_BELIEF_LOSS_WEIGHT,
    MoveScheduleHorizon,
    Phase17MoveError,
    reference_iteration,
)
from stratego.training.phase17.move_loss import (
    Phase17LossError,
    assert_value_loss_reduces,
    loss_semantics,
    masked_soft_value_loss,
    phase17_batch_loss,
)
from stratego.training.phase17.move_snapshot import CurrentMovePolicy, snapshot_from_model
from stratego.training.phase17.move_start import (
    belief_head_parameters,
    build_move_start,
)
from stratego.training.phase17.move_trainer import (
    MoveWindowTrainer,
    Phase17TrainerError,
    assert_ema_never_acted,
    build_arrays,
    train_order,
    trainer_semantics,
    window_statistics,
)
from stratego.training.phase17.transition_collector import FixedTransitionCollector
from stratego.training.phase9_contract import KL_HARD_LIMIT
from stratego.training.phase9_loss import soft_value_loss

from .test_move_support import DeterministicSetupProvider, perturbed_copy

RUN = "RUN-TEST-A"


# ---------------------------------------------------------------------------
# The masked value term
# ---------------------------------------------------------------------------


def _value_batch(rows: int = 16, seed: int = 3):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, 3, generator=generator)
    targets = torch.softmax(torch.randn(rows, 3, generator=generator), dim=1)
    return logits, targets


def test_the_masked_value_term_reduces_to_the_accepted_one():
    logits, targets = _value_batch()
    report = assert_value_loss_reduces(logits, targets)
    assert report["reduces_to_accepted"] is True
    assert report["difference"] == 0.0
    assert report["phase17_masked"] == pytest.approx(
        float(soft_value_loss(logits, targets))
    )


def test_a_row_weight_of_zero_removes_that_row_from_the_value_term():
    logits, targets = _value_batch(rows=8, seed=11)
    keep = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    masked = float(masked_soft_value_loss(logits, targets, keep))
    subset = float(soft_value_loss(logits[::2], targets[::2]))
    assert masked == pytest.approx(subset, abs=1e-6)


def test_an_all_zero_weight_contributes_exactly_zero_and_keeps_the_graph():
    logits, targets = _value_batch(rows=4, seed=5)
    logits.requires_grad_(True)
    loss = masked_soft_value_loss(logits, targets, torch.zeros(4))
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) == 0.0


def test_a_wrong_shaped_row_weight_is_refused():
    logits, targets = _value_batch(rows=4)
    with pytest.raises(Phase17LossError, match="value row weight"):
        masked_soft_value_loss(logits, targets, torch.ones(3))


def test_a_negative_row_weight_is_refused():
    logits, targets = _value_batch(rows=4)
    with pytest.raises(Phase17LossError, match="negative or non-finite"):
        masked_soft_value_loss(logits, targets, torch.tensor([1.0, -1.0, 1.0, 1.0]))


def test_loss_semantics_keeps_the_three_regularizers_apart():
    semantics = loss_semantics()
    assert "FORWARD" in semantics["kl_direction"]
    assert "BONUS" in semantics["entropy_term"]
    assert semantics["belief_weight"] == 0.0
    assert semantics["phase9_accepted_belief_weight"] == 0.25
    assert semantics["accepted_objective_not_edited"] == (
        "stratego.training.phase9_loss.phase9_batch_loss"
    )


# ---------------------------------------------------------------------------
# A real window, trained
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def window():
    """One small real window of transitions, collected once and reused."""
    start = build_move_start(total_iterations=40, device="cpu")
    cell = CurrentMovePolicy(snapshot_from_model(start.model, device="cpu"), iteration=1)
    collector = FixedTransitionCollector(
        run_id=RUN,
        cell=cell,
        setup_provider=DeterministicSetupProvider(),
        population=6,
        budget=192,
    )
    result = collector.collect_window()
    return {"start": start, "cell": cell, "rows": result.rows, "result": result}


def _trainer(start, *, minibatch_size=64):
    return MoveWindowTrainer(
        run_id=RUN,
        model=start.model,
        optimizer=start.optimizer,
        controller=start.controller,
        ema=start.ema,
        horizon=start.horizon,
        device="cpu",
        minibatch_size=minibatch_size,
    )


def test_one_update_moves_the_raw_weights_and_the_ema_follows(window):
    start = build_move_start(total_iterations=40, device="cpu")
    cell = CurrentMovePolicy(snapshot_from_model(start.model, device="cpu"), iteration=1)
    trainer = _trainer(start)
    before = {name: t.detach().clone() for name, t in start.model.state_dict().items()}
    ema_before = start.ema.state_dict()

    update = trainer.train_window(window["rows"], iteration=1, cell=cell)
    summary = update.summary()

    assert summary["optimizer_steps"] > 0
    assert summary["raw_changed"] is True
    assert summary["epochs_per_iteration"] == MOVE_EPOCHS_PER_ITERATION
    assert len(update.epochs) == 1
    assert summary["transitions_harvested"] == len(window["rows"])
    assert 0 < summary["transitions_trained"] < summary["transitions_harvested"]
    assert (
        summary["boundary_bootstrapped_rows"] + summary["terminal_rows"]
        == summary["transitions_harvested"]
    )

    after = start.model.state_dict()
    assert any(
        not torch.equal(before[name], after[name].detach()) for name in before
    )
    ema_after = start.ema.state_dict()
    assert start.ema.updates == summary["optimizer_steps"]
    moved = [
        name
        for name in ema_before
        if ema_before[name].is_floating_point()
        and not torch.equal(ema_before[name], ema_after[name])
    ]
    assert moved, "the EMA did not follow the raw weights"
    # The EMA lags: at decay 0.999 it is far closer to where it started.
    name = moved[0]
    assert float((ema_after[name] - ema_before[name]).abs().max()) < float(
        (after[name].detach() - before[name]).abs().max()
    )


def test_the_ema_never_acted_in_the_training_population(window):
    start = build_move_start(total_iterations=40, device="cpu")
    cell = CurrentMovePolicy(snapshot_from_model(start.model, device="cpu"), iteration=1)
    trainer = _trainer(start)
    trainer.train_window(window["rows"], iteration=1, cell=cell)
    report = assert_ema_never_acted(cell, start.ema)
    assert report["holds"] is True
    assert report["ema_ever_acted"] is False
    assert report["ema_updates"] > 0


def test_the_belief_head_receives_no_gradient(window):
    """Stronger than a zero coefficient: the term is not in the graph at all."""
    start = build_move_start(total_iterations=40, device="cpu")
    cell = CurrentMovePolicy(snapshot_from_model(start.model, device="cpu"), iteration=1)
    trainer = _trainer(start, minibatch_size=32)
    names = set(belief_head_parameters(start.model))
    trainer.train_window(window["rows"][:64], iteration=1, cell=cell)
    for name, parameter in start.model.named_parameters():
        if name in names:
            assert parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
        elif name.startswith("policy_"):
            assert parameter.grad is not None


def test_the_belief_head_weights_do_not_move(window):
    start = build_move_start(total_iterations=40, device="cpu")
    cell = CurrentMovePolicy(snapshot_from_model(start.model, device="cpu"), iteration=1)
    trainer = _trainer(start, minibatch_size=32)
    names = belief_head_parameters(start.model)
    before = {
        name: parameter.detach().clone()
        for name, parameter in start.model.named_parameters()
        if name in names
    }
    trainer.train_window(window["rows"][:64], iteration=1, cell=cell)
    for name, parameter in start.model.named_parameters():
        if name in before:
            assert torch.equal(before[name], parameter.detach())


def test_the_belief_term_is_zero_in_the_telemetry(window):
    start = build_move_start(total_iterations=40, device="cpu")
    cell = CurrentMovePolicy(snapshot_from_model(start.model, device="cpu"), iteration=1)
    trainer = _trainer(start, minibatch_size=32)
    update = trainer.train_window(window["rows"][:64], iteration=1, cell=cell)
    assert update.means["mean_loss_belief"] == 0.0
    assert update.means["mean_belief_weight"] == 0.0
    assert update.summary()["belief_loss_weight"] == 0.0
    assert BELIEF_LOSS_WEIGHT == 0.0
    assert PHASE9_ACCEPTED_BELIEF_LOSS_WEIGHT == 0.25


def test_the_window_statistics_are_the_accepted_formulas(window):
    statistics = window_statistics(window["rows"], iteration=1)
    magnitudes = sorted(abs(float(row.advantage_target)) for row in window["rows"])
    from stratego.training.phase9_contract import advantage_filter_threshold

    assert statistics.threshold == pytest.approx(advantage_filter_threshold(magnitudes))
    assert statistics.retention_fraction == pytest.approx(
        statistics.eligible / statistics.rows
    )
    assert 0.2 <= statistics.retention_fraction <= 0.35


def test_the_batch_arrays_carry_only_the_accepted_model_input(window):
    arrays = build_arrays(window["rows"][:8])
    assert set(arrays) == {
        "observation",
        "legal_mask",
        "sampled_action_model",
        "behavior_action_probability",
        "behavior_probabilities",
        "standardized_advantage",
        "ppo_eligible",
        "wdl_target",
        "value_row_weight",
    }
    assert "belief_target" not in arrays
    assert arrays["observation"].shape == (8, 127, 10, 10)
    assert arrays["behavior_probabilities"].shape == (8, 10000)
    # the dense matrix carries the stored bytes, not a recomputation
    row = window["rows"][0]
    assert float(arrays["behavior_probabilities"][0].sum()) == pytest.approx(
        float(sum(row.behavior_probabilities)), abs=1e-5
    )


def test_the_train_order_is_deterministic_from_run_iteration_and_epoch(window):
    rows = window["rows"][:40]
    first = train_order(rows, run_id=RUN, iteration=3, epoch=1)
    assert first == train_order(rows, run_id=RUN, iteration=3, epoch=1)
    assert first != train_order(rows, run_id=RUN, iteration=4, epoch=1)
    assert first != train_order(rows, run_id="RUN-OTHER", iteration=3, epoch=1)
    assert sorted(first) == list(range(len(rows)))


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_an_iteration_outside_the_frozen_horizon_is_refused(window):
    start = build_move_start(total_iterations=40, device="cpu")
    trainer = _trainer(start)
    with pytest.raises(Phase17TrainerError, match="outside the frozen horizon"):
        trainer.train_window(window["rows"][:8], iteration=41)
    with pytest.raises(Phase17TrainerError, match="outside the frozen horizon"):
        trainer.train_window(window["rows"][:8], iteration=0)


def test_a_row_under_a_model_state_the_cell_never_held_is_refused(window):
    import copy

    start = build_move_start(total_iterations=40, device="cpu")
    cell = CurrentMovePolicy(snapshot_from_model(start.model, device="cpu"), iteration=1)
    trainer = _trainer(start)
    rows = [copy.copy(row) for row in window["rows"][:8]]
    rows[3].behavior_model_state_digest = "e" * 64
    with pytest.raises(Phase17TrainerError, match="never held"):
        trainer.train_window(rows, iteration=1, cell=cell)
    assert trainer.counters["stale_digest_refusals"] == 1


def test_an_empty_window_is_refused():
    start = build_move_start(total_iterations=40, device="cpu")
    with pytest.raises(Phase17TrainerError, match="no rows"):
        _trainer(start).train_window([], iteration=1)


def test_more_than_one_epoch_is_refused():
    start = build_move_start(total_iterations=40, device="cpu")
    with pytest.raises(Phase17TrainerError, match="1 epoch per iteration"):
        MoveWindowTrainer(
            run_id=RUN,
            model=start.model,
            optimizer=start.optimizer,
            controller=start.controller,
            ema=start.ema,
            horizon=start.horizon,
            device="cpu",
            epochs=2,
        )


def test_a_kl_above_the_hard_limit_vetoes_the_update(window, monkeypatch):
    start = build_move_start(total_iterations=40, device="cpu")
    cell = CurrentMovePolicy(snapshot_from_model(start.model, device="cpu"), iteration=1)
    trainer = _trainer(start, minibatch_size=32)
    real_observe = trainer.controller.observe

    def loud(*, mean_kl, examples, clipped, ppo_examples):
        real_observe(
            mean_kl=KL_HARD_LIMIT * 2.0,
            examples=examples,
            clipped=clipped,
            ppo_examples=ppo_examples,
        )

    monkeypatch.setattr(trainer.controller, "observe", loud)
    with pytest.raises(Phase17TrainerError, match="exceeds the hard limit"):
        trainer.train_window(window["rows"][:64], iteration=1, cell=cell)
    assert trainer.counters["kl_vetoes"] == 1


def test_a_nonfinite_target_is_refused(window):
    import copy

    start = build_move_start(total_iterations=40, device="cpu")
    trainer = _trainer(start)
    rows = [copy.copy(row) for row in window["rows"][:8]]
    rows[2].advantage_target = float("nan")
    with pytest.raises(Phase17TrainerError, match="non-finite advantage"):
        trainer.train_window(rows, iteration=1)


def test_the_trainer_state_round_trips(window):
    start = build_move_start(total_iterations=40, device="cpu")
    cell = CurrentMovePolicy(snapshot_from_model(start.model, device="cpu"), iteration=1)
    trainer = _trainer(start, minibatch_size=64)
    trainer.train_window(window["rows"], iteration=1, cell=cell)
    state = trainer.trainer_state()

    other = _trainer(build_move_start(total_iterations=40, device="cpu"))
    other.restore_state(state)
    assert other.global_step == trainer.global_step
    assert other.examples_consumed == trainer.examples_consumed
    assert other.controller.beta == pytest.approx(trainer.controller.beta)
    assert other.controller.history == trainer.controller.history

    state["run_id"] = "RUN-OTHER"
    with pytest.raises(Phase17TrainerError, match="belongs to run"):
        other.restore_state(state)


# ---------------------------------------------------------------------------
# The schedules
# ---------------------------------------------------------------------------


def test_the_reference_iteration_is_one_eighth_of_the_horizon():
    assert reference_iteration(626) == 79
    assert reference_iteration(8) == 1
    assert reference_iteration(1) == 1
    with pytest.raises(Phase17MoveError, match="horizon N"):
        reference_iteration(0)


def test_the_learning_rate_holds_its_ceiling_then_decays():
    horizon = MoveScheduleHorizon(total_iterations=626)
    assert horizon.reference_iteration == 79
    assert horizon.learning_rate(1) == pytest.approx(MOVE_LR_MAX)
    assert horizon.learning_rate(79) == pytest.approx(MOVE_LR_MAX)
    assert horizon.learning_rate(80) < MOVE_LR_MAX
    assert horizon.learning_rate(626) > MOVE_LR_MIN
    rates = [horizon.learning_rate(n) for n in range(1, 627)]
    assert all(later <= earlier for earlier, later in zip(rates, rates[1:]))
    assert min(rates) >= MOVE_LR_MIN


def test_the_entropy_bonus_anneals_to_its_floor():
    horizon = MoveScheduleHorizon(total_iterations=626)
    assert horizon.entropy_coefficient(1) == pytest.approx(0.005)
    assert horizon.entropy_coefficient(214) == pytest.approx(0.001, abs=1e-5)
    assert horizon.entropy_coefficient(626) == pytest.approx(0.001)
    values = [horizon.entropy_coefficient(n) for n in range(1, 627)]
    assert all(later <= earlier for earlier, later in zip(values, values[1:]))


def test_the_horizon_curve_covers_every_iteration_and_is_serializable():
    horizon = MoveScheduleHorizon(total_iterations=12)
    curve = horizon.curve()
    assert [row["iteration"] for row in curve] == list(range(1, 13))
    document = horizon.to_dict()
    assert document["reference_iteration"] == 2
    assert document["first"] == curve[0]
    assert document["last"] == curve[-1]
    assert "entropy bonus" in document["entropy_is"]


def test_a_schedule_index_of_zero_is_refused():
    horizon = MoveScheduleHorizon(total_iterations=10)
    with pytest.raises(Phase17MoveError, match="schedule index"):
        horizon.learning_rate(0)
    with pytest.raises(Phase17MoveError, match="schedule index"):
        horizon.entropy_coefficient(-1)


def test_trainer_semantics_lists_its_refusals():
    semantics = trainer_semantics()
    assert semantics["epochs_per_iteration"] == 1
    assert "never acts" in semantics["ema"]
    assert any("never held" in entry for entry in semantics["refusals"])


def test_the_ema_digest_uses_the_accepted_algorithm():
    """The EMA is a mapping, not a module, and must hash the same way anyway."""
    from stratego.training.phase17.move_trainer import state_mapping_digest
    from stratego.training.phase9_behavior import state_dict_digest

    start = build_move_start(total_iterations=40, device="cpu")
    assert state_mapping_digest(start.model.state_dict()) == state_dict_digest(
        start.model
    )
    # Before any update the EMA *is* the raw weights, and the check says so
    # rather than pretending otherwise: it holds because nothing has trained.
    cell = CurrentMovePolicy(snapshot_from_model(start.model, device="cpu"), iteration=1)
    report = assert_ema_never_acted(cell, start.ema)
    assert report["ema_updates"] == 0
    assert report["ema_ever_acted"] is True
    assert report["holds"] is True


def test_a_trainer_run_id_with_a_separator_is_refused():
    start = build_move_start(total_iterations=40, device="cpu")
    with pytest.raises(Phase17MoveError, match="run id"):
        MoveWindowTrainer(
            run_id="RUN:A",
            model=start.model,
            optimizer=start.optimizer,
            controller=start.controller,
            ema=start.ema,
            horizon=start.horizon,
            device="cpu",
        )
