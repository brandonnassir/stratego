"""Phase 16 Agent 3: the window collector, the learner, and their identities."""

import numpy as np
import pytest
import torch

from stratego.training.phase16 import contract as C
from stratego.training.phase16 import collector as COL
from stratego.training.phase16 import trainer as TR
from stratego.training.phase16.snapshots import (
    behavior_token,
    bind_anchor,
    participants_for,
    snapshot_from_model,
)


@pytest.fixture(scope="module")
def one_window(starting_model, library_source):
    """One small collected window, built once and shared by the readers.

    Module-scoped on purpose: collecting a window means playing real games to
    completion, and every test below reads the same window rather than paying
    for its own.
    """
    from stratego.training.phase16.collector import WindowCollector
    from stratego.training.phase16.population import HistoricalPool

    config = C.ARM_B.replace(
        arm_id="test_tiny",
        population=4,
        window_decisions=128,
        minibatch_size=64,
        device="cpu",
        collection_device="cpu",
    )
    participants = participants_for(
        starting_model,
        identity="CURRENT",
        device="cpu",
        historical=bind_anchor(starting_model, identity="P24", device="cpu"),
    )
    collector = WindowCollector(
        config, participants, setup_source=library_source, pool=HistoricalPool("P24")
    )
    return config, collector, collector.collect_window()


# ---------------------------------------------------------------------------
# The action-sampling stream
# ---------------------------------------------------------------------------


def test_the_action_stream_is_phase16s_own_and_deterministic():
    from stratego.training.phase14_seed import action_sampling_uniform as phase14

    identifier = C.game_id("test_tiny", 0, 0)
    first = COL.action_sampling_uniform(identifier, 7)
    assert first == COL.action_sampling_uniform(identifier, 7)
    assert first != COL.action_sampling_uniform(identifier, 8)
    assert 0.0 <= first < 1.0
    with pytest.raises(Exception):
        phase14(identifier, 7)  # a Phase 16 id is not a Phase 14 id


def test_the_cumulative_walk_selects_by_stored_mass():
    legal = (10, 20, 30)
    # a point mass on the middle action must always select it
    assert COL.select_action((0.0, 1.0, 0.0), legal, "g", 0) == 20
    with pytest.raises(Exception):
        COL.select_action((0.5, 0.5), legal, "g", 0)
    with pytest.raises(Exception):
        COL.select_action((0.3, 0.3, 0.4), (30, 20, 10), "g", 0)


# ---------------------------------------------------------------------------
# Harvesting
# ---------------------------------------------------------------------------


def test_a_window_emits_only_finished_games_with_exact_targets(one_window):
    _config, _collector, window = one_window
    assert window.rows, "the window floor guarantees at least one minibatch"
    assert window.games_finished >= 1
    finished = {row.game_id for row in window.rows}
    assert len(finished) == window.games_finished
    for row in window.rows:
        assert sum(row.wdl_target) == pytest.approx(1.0, abs=1e-5)
        assert all(value >= -1e-6 for value in row.wdl_target)


def test_the_last_decision_of_a_game_carries_the_one_hot_outcome(one_window):
    _config, _collector, window = one_window
    by_game: dict = {}
    for row in window.rows:
        key = (row.game_id, row.learner_side)
        if key not in by_game or row.decision_index > by_game[key].decision_index:
            by_game[key] = row
    assert by_game
    for row in by_game.values():
        assert sorted(row.wdl_target) == [0.0, 0.0, 1.0]


def test_a_harvested_row_carries_exactly_the_fields_the_objective_reads(one_window):
    _config, _collector, window = one_window
    row = window.rows[0]
    from stratego.model.contract import OBSERVATION_SHAPE, POLICY_LOGIT_COUNT

    assert row.observation.shape == tuple(OBSERVATION_SHAPE)
    assert row.observation.dtype == np.float32
    assert row.legal_mask.shape == (POLICY_LOGIT_COUNT,)
    assert row.legal_mask.dtype == np.bool_
    assert row.belief_target.shape == (100,) and row.belief_mask.shape == (100,)
    assert len(row.behavior_legal_actions) == len(row.behavior_legal_probabilities)
    assert row.sampled_action_abs in row.behavior_legal_actions
    assert 0.0 <= row.behavior_action_probability <= 1.0
    assert row.behavior_action_logprob == pytest.approx(
        float(np.log(max(row.behavior_action_probability, 1e-12)))
    )
    assert row.legal_mask[row.sampled_action_model]


def test_the_belief_label_never_reaches_the_backbone(one_window):
    """The one model input is the observation; the label rides in its own field."""
    _config, _collector, window = one_window
    arrays = TR.build_arrays(window.rows[:8])
    assert set(arrays) == {
        "observation",
        "legal_mask",
        "sampled_action_model",
        "behavior_action_probability",
        "behavior_probabilities",
        "standardized_advantage",
        "ppo_eligible",
        "wdl_target",
        "belief_target",
        "belief_mask",
    }
    assert arrays["observation"].shape[0] == 8


def test_plies_advanced_counts_every_ply_not_only_finished_games(one_window):
    _config, collector, window = one_window
    assert window.plies_advanced >= window.plies
    assert window.plies_advanced > 0
    assert window.plies_per_second > 0
    assert collector.plies_advanced == window.plies_advanced


def test_a_second_window_continues_the_same_population(tiny_collector):
    first = tiny_collector.collect_window()
    live_before = [runner for runner in tiny_collector.slots if runner is not None]
    ids_before = {runner.game_id for runner in live_before}
    second = tiny_collector.collect_window()
    assert second.iteration == first.iteration + 1
    # the games that were mid-play are the same objects, further along
    still = {
        runner.game_id for runner in tiny_collector.slots if runner is not None
    }
    assert ids_before & still or second.games_finished >= len(ids_before)
    assert tiny_collector.decisions_collected >= first.learner_decisions


def test_no_ply_is_counted_twice_across_a_window_boundary(tiny_collector):
    first = tiny_collector.collect_window()
    second = tiny_collector.collect_window()
    total = first.plies_advanced + second.plies_advanced
    assert total == tiny_collector.plies_advanced
    live = sum(
        int(runner.state.total_moves)
        for runner in tiny_collector.slots
        if runner is not None
    )
    finished = sum(length for length in first.game_lengths + second.game_lengths)
    assert total == live + finished


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_a_snapshot_of_the_live_model_is_frozen_and_digest_bound(starting_model):
    snapshot = snapshot_from_model(starting_model, identity="W0001", device="cpu")
    snapshot.assert_frozen()
    assert snapshot.policy_token == behavior_token("W0001")
    assert not snapshot.model.training
    assert all(not p.requires_grad for p in snapshot.model.parameters())
    # and it does not alias the live model
    assert all(p.requires_grad for p in starting_model.parameters())


def test_rebinding_updates_the_weights_but_not_the_identity(tiny_collector, starting_model):
    tiny_collector.collect_window()
    assert any(runner is not None for runner in tiny_collector.slots)
    replacement = participants_for(
        starting_model,
        identity="CURRENT",
        device="cpu",
        historical=bind_anchor(starting_model, identity="P24", device="cpu"),
    )
    report = tiny_collector.rebind(replacement)
    assert report["identity"] == "CURRENT"
    assert report["games_in_flight"] >= 1
    moved = participants_for(starting_model, identity="W0002", device="cpu")
    with pytest.raises(COL.Phase16CollectorError):
        tiny_collector.rebind(moved)


def test_the_scheduled_adapter_names_tokens_the_snapshots_carry():
    from stratego.training.phase16.population import draw_for_slot

    draw = draw_for_slot(C.ARM_B, slot=0, draw=0)
    scheduled = COL.ScheduledPhase16Game(draw, "CURRENT")
    assert scheduled.red_policy_identity == scheduled.blue_policy_identity
    assert scheduled.red_policy_identity == behavior_token("CURRENT")
    assert scheduled.opponent_kind == "current_policy"
    assert scheduled.historical_snapshot_identity is None


def test_collector_semantics_declares_no_search():
    semantics = COL.collector_semantics()
    assert semantics["search"].startswith("absent")
    assert "the game id scheme" in semantics["phase16_own"]
    imported = [
        line
        for line in open(COL.__file__).read().splitlines()
        if line.lstrip().startswith(("import ", "from "))
    ]
    assert not any("search" in line for line in imported)


# ---------------------------------------------------------------------------
# The learner
# ---------------------------------------------------------------------------


def test_the_window_filter_is_the_accepted_quantile_rule(one_window):
    _config, _collector, window = one_window
    statistics = TR.window_statistics(window.rows, iteration=1)
    magnitudes = sorted(abs(row.advantage) for row in window.rows)
    from stratego.training.phase9_contract import advantage_filter_threshold

    assert statistics.threshold == pytest.approx(advantage_filter_threshold(magnitudes))
    assert statistics.threshold >= 0.01
    # a Q75 filter retains roughly the top quarter
    assert 0.15 < statistics.retention_fraction < 0.40


def test_standardization_is_over_the_selected_subset_only(one_window):
    _config, _collector, window = one_window
    statistics = TR.window_statistics(window.rows, iteration=1)
    TR.apply_statistics(window.rows, statistics)
    eligible = [row for row in window.rows if row.ppo_eligible]
    assert len(eligible) == statistics.eligible
    values = np.asarray([row.standardized_advantage for row in eligible])
    assert float(values.mean()) == pytest.approx(0.0, abs=1e-4)
    assert all(row.ppo_eligible == (abs(row.advantage) >= statistics.threshold) for row in window.rows)


def test_the_training_order_is_deterministic_and_epoch_separated(one_window):
    _config, _collector, window = one_window
    rows = window.rows
    first = TR.train_order(rows, arm_id="b_damped", iteration=3, epoch=1)
    assert first == TR.train_order(rows, arm_id="b_damped", iteration=3, epoch=1)
    assert first != TR.train_order(rows, arm_id="b_damped", iteration=3, epoch=2)
    assert first != TR.train_order(rows, arm_id="a_control", iteration=3, epoch=1)
    assert sorted(first) == list(range(len(rows)))


def test_an_update_moves_the_weights_and_records_what_it_did(one_window, starting_model):
    """A real update on a real window, at a minibatch large enough to be one.

    The minibatch size matters here: the accepted KL veto is a *behaviour*
    limit, and an 8-example step at 1.5e-4 trips it on noise alone. This test
    is about the update landing, so it uses a size the production arm uses.
    """
    config, _collector, window = one_window
    import copy

    model = copy.deepcopy(starting_model)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    trainer = TR.WindowTrainer(config, model, device="cpu")
    update = trainer.train_window(list(window.rows), iteration=1)
    assert update.steps >= 1
    assert update.examples >= update.steps
    assert update.learning_rate == C.DEFAULT_LR_MAX
    assert update.entropy_coefficient == C.DEFAULT_ENTROPY_START
    assert len(update.epochs) == config.epochs
    after = model.state_dict()
    assert any(not torch.equal(before[name], after[name]) for name in before)
    assert trainer.counters["non_finite_losses"] == 0


def test_the_ema_tracks_the_weights_without_being_them(starting_model):
    import copy

    model = copy.deepcopy(starting_model)
    ema = TR.WeightEMA(model, decay=0.5)
    original = {name: tensor.clone() for name, tensor in ema.state.items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model)
    assert ema.updates == 1
    for name, tensor in model.state_dict().items():
        if tensor.is_floating_point():
            expected = original[name] * 0.5 + tensor.to("cpu") * 0.5
            assert torch.allclose(ema.state[name], expected)
        else:
            assert torch.equal(ema.state[name], tensor.to("cpu"))
    # the EMA is not the model
    assert not any(
        torch.equal(ema.state[name], model.state_dict()[name].to("cpu"))
        for name, value in model.state_dict().items()
        if value.is_floating_point()
    )


def test_an_ema_round_trips_through_its_state_dict(starting_model):
    import copy

    model = copy.deepcopy(starting_model)
    ema = TR.WeightEMA(model, decay=0.9)
    ema.update(model)
    payload = ema.state_dict()
    restored = TR.WeightEMA(model, decay=0.9)
    restored.load_state_dict(payload, updates=ema.updates)
    assert restored.updates == 1
    assert all(torch.equal(payload[name], restored.state[name]) for name in payload)


def test_an_arm_without_an_ema_builds_none(starting_model):
    trainer = TR.WindowTrainer(C.ARM_A.replace(device="cpu"), starting_model, device="cpu")
    assert trainer.ema is None
    assert trainer.trainer_state()["ema"] == {"present": False}


def test_the_hard_limits_are_the_accepted_ones(starting_model):
    from stratego.training.phase9_contract import (
        CLIP_FRACTION_HARD_LIMIT,
        KL_HARD_LIMIT,
    )

    trainer = TR.WindowTrainer(C.ARM_B.replace(device="cpu"), starting_model, device="cpu")
    with pytest.raises(TR.Phase16TrainerError):
        trainer._check_hard_limits(
            iteration=1, epoch=1, mean_kl=KL_HARD_LIMIT + 1e-6, clip_fraction=0.0
        )
    assert trainer.counters["kl_vetoes"] == 1
    with pytest.raises(TR.Phase16TrainerError):
        trainer._check_hard_limits(
            iteration=1, epoch=1, mean_kl=0.0, clip_fraction=CLIP_FRACTION_HARD_LIMIT + 1e-6
        )
    assert trainer.counters["clip_vetoes"] == 1
    # inside both limits, nothing is raised and nothing is counted
    trainer._check_hard_limits(iteration=1, epoch=1, mean_kl=0.01, clip_fraction=0.2)
    assert trainer.counters["kl_vetoes"] == 1


def test_trainer_semantics_names_the_accepted_objective():
    semantics = TR.trainer_semantics()
    assert semantics["objective"].startswith("stratego.training.phase9_loss")
    assert semantics["kl_controller"].endswith("unchanged")
    assert semantics["gradient_clip_norm"] == 1.0
    assert "never trains" in semantics["ema"]


def test_empty_windows_are_refused_rather_than_trained_on(starting_model):
    trainer = TR.WindowTrainer(C.ARM_B.replace(device="cpu"), starting_model, device="cpu")
    with pytest.raises(TR.Phase16TrainerError):
        trainer.train_window([], iteration=1)
    with pytest.raises(TR.Phase16TrainerError):
        TR.window_statistics([], iteration=1)
    with pytest.raises(TR.Phase16TrainerError):
        TR.build_arrays([])
