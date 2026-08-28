"""Phase 17: the setup update and its identities, under operator decision D10."""

import copy

import numpy as np
import pytest
import torch

from stratego.training.phase9_behavior import state_dict_digest
from stratego.training.phase17.setup_contract import (
    SETUP_BEHAVIOR_KL_COEFFICIENT,
    SETUP_CONDITIONAL_ENTROPY_NORMALIZER,
    SETUP_GRADIENT_CLIP_NORM,
    SETUP_PREFIXES,
    SETUP_RECIPE_VERSION,
    Phase17SetupError,
    SetupTrainingConfig,
    setup_alpha,
)
from stratego.training.phase17.setup_learning import (
    SetupTrainer,
    advantage_terms,
    build_batch,
    expected_value_from_wdl,
    setup_advantage,
    setup_batch_loss,
)
from stratego.training.phase17.setup_model import build_setup_model


# -- the alpha schedule -----------------------------------------------------


def test_alpha_is_the_papers_printed_schedule():
    """D10 section 4: `alpha(n) = 0.1 * n**-0.3`, transcribed, no floor."""
    assert setup_alpha(1) == pytest.approx(0.1)
    assert setup_alpha(2) == pytest.approx(0.1 * 2 ** -0.3)
    assert setup_alpha(640) == pytest.approx(0.1 * 640 ** -0.3)


def test_alpha_does_not_depend_on_the_run_length():
    """D3's re-horizoning is retired: nothing here knows what N is.

    The signature is the check. A one-argument alpha cannot be re-fitted to a
    horizon, so a 640-iteration run and a 42,376-iteration one read the same
    curve at the same `n` -- which is what D10 chose, accepting that a short
    run simply ends less annealed than the paper's.
    """
    import inspect

    assert list(inspect.signature(setup_alpha).parameters) == ["iteration"]


def test_alpha_is_monotone_and_never_floors():
    values = [setup_alpha(n) for n in range(1, 2001)]
    assert all(later < earlier for earlier, later in zip(values, values[1:]))
    # No floor: it keeps falling well past any horizon this run could reach.
    assert setup_alpha(100_000) < setup_alpha(2000)


def test_a_zero_or_negative_iteration_is_refused():
    with pytest.raises(Phase17SetupError, match="one-based"):
        setup_alpha(0)
    with pytest.raises(Phase17SetupError, match="one-based"):
        setup_alpha(-3)


def zero_kl(config):
    """The same config with the behavior-KL term switched off.

    Used only where a test needs to isolate another term. The coefficient is a
    contract constant, so this is an explicit local override rather than a
    knob the run has.
    """
    return config.replace(behavior_kl_coefficient=0.0)


# -- the advantage ----------------------------------------------------------


def test_expected_value_is_win_minus_loss():
    wdl = np.array([[0.6, 0.1, 0.3]], dtype=np.float32)
    assert expected_value_from_wdl(wdl)[0] == pytest.approx(0.3)


def test_the_advantage_is_the_papers_printed_form(completed_episodes):
    """delta = (o - E[v]) + alpha * (I - h), operator decision D10 section 4."""
    episode = completed_episodes[0]
    alpha = 0.1
    expected = expected_value_from_wdl(episode.prefix_wdl_predictions)
    manual = (float(episode.outcome) - expected) + alpha * (
        episode.suffix_information_content
        - episode.prefix_conditional_entropy_predictions
    )
    assert np.allclose(setup_advantage(episode, alpha), manual, atol=1e-5)


def test_the_information_enters_in_raw_nats_not_in_tenths(completed_episodes):
    """The unit asymmetry D10 kept deliberately.

    `I` is in nats and `h` predicts `I/10`, so the entropy term is close to
    `0.9 * alpha * I` rather than the `0.9 * alpha * (I/10)` of the retired
    D7-B form. Asserting the factor of ten explicitly, because "which units"
    is exactly the thing a later reader would otherwise have to infer.
    """
    episode = copy.deepcopy(completed_episodes[0])
    episode.prefix_conditional_entropy_predictions = np.zeros_like(
        episode.prefix_conditional_entropy_predictions
    )
    delta = setup_advantage(episode, 0.1)
    expected = expected_value_from_wdl(episode.prefix_wdl_predictions)
    entropy_term = delta - (float(episode.outcome) - expected)
    normalized_form = 0.1 * (
        episode.suffix_information_content * SETUP_CONDITIONAL_ENTROPY_NORMALIZER
    )
    assert np.allclose(entropy_term, 10.0 * normalized_form, atol=1e-4)


def test_the_advantage_reads_the_recorded_prediction_head(completed_episodes):
    """The paper's form is CENTERED: perturbing `h` must move the advantage.

    D7-B deliberately dropped `h` from the advantage because the centered
    residual converged to zero. D10 puts it back, and the reason it does not
    go inert this time is the unit asymmetry above: `L_h` drives `h` to `I/10`,
    not to `I`, so `(I - h)` settles at roughly `0.9 * I` rather than at zero.
    """
    episode = copy.deepcopy(completed_episodes[0])
    before = setup_advantage(episode, 0.1)
    episode.prefix_conditional_entropy_predictions = (
        episode.prefix_conditional_entropy_predictions + 5.0
    )
    after = setup_advantage(episode, 0.1)
    assert np.allclose(after, np.asarray(before) - 0.1 * 5.0, atol=1e-4)


def test_a_converged_head_leaves_nine_tenths_of_the_information(completed_episodes):
    """Setting `h` exactly to `I/10` is the converged state `L_h` trains toward."""
    episode = copy.deepcopy(completed_episodes[0])
    episode.prefix_conditional_entropy_predictions = (
        episode.suffix_information_content * SETUP_CONDITIONAL_ENTROPY_NORMALIZER
    ).astype(np.float32)
    terms = advantage_terms(episode, 0.1)
    assert terms["conditional_entropy_residual_abs_mean"] == pytest.approx(0.0, abs=1e-5)
    assert terms["entropy_term_mean"] == pytest.approx(
        0.9 * 0.1 * terms["information_nats_mean"], rel=1e-4
    )


def test_the_component_magnitudes_are_recorded(completed_episodes):
    """D10 section 4 asks for the magnitudes, not for a compensating scale.

    The recipe as printed puts an entropy term of order `0.9 * alpha * I` --
    several nats at alpha = 0.1 -- against an outcome term bounded by 2. That
    is what Agent 1 refused when it chose the normalized D7-B form, and D10
    chose it anyway. The number is asserted here so the report and the
    telemetry state the same thing.
    """
    terms = advantage_terms(completed_episodes[0], 0.1)
    assert terms["outcome_term_abs_mean"] <= 2.0
    assert terms["entropy_term_abs_mean"] > 2.0
    assert terms["entropy_to_outcome_abs_ratio"] > 1.0
    assert terms["alpha"] == 0.1


def test_the_entropy_head_is_still_trained(setup_model, config, completed_episodes):
    """`L_h` is retained for telemetry and paper alignment, so it must still
    reach the head's parameters."""
    model = build_setup_model(seed=606)
    batch = build_batch(completed_episodes[:8], alpha=0.1)
    _, terms = setup_batch_loss(model, batch, config=zero_kl(config))
    model.zero_grad(set_to_none=True)
    terms["conditional_entropy_loss"].backward()
    assert model.conditional_entropy_head.weight.grad.abs().max() > 0.0


def test_an_open_episode_can_never_enter_the_update(red_samples):
    from stratego.training.phase17.setup_episode import SetupEpisode

    episode = SetupEpisode.create(red_samples[0], run_id="RUN-TEST-A", game_id="g")
    with pytest.raises(Phase17SetupError, match="has no outcome"):
        setup_advantage(episode, 0.1)


def test_a_corrupt_mask_is_caught_when_the_batch_is_built(completed_episodes):
    episode = copy.deepcopy(completed_episodes[0])
    episode.inventory_masks[3, :] = True
    with pytest.raises(Phase17SetupError, match="disagrees with the mask"):
        build_batch([episode], alpha=0.1)


# -- the loss ---------------------------------------------------------------


def test_every_loss_term_is_returned_under_its_own_name(setup_model, config, completed_episodes):
    batch = build_batch(completed_episodes[:8], alpha=0.1)
    total, terms = setup_batch_loss(setup_model, batch, config=config)
    for name in (
        "policy_loss",
        "value_loss",
        "conditional_entropy_loss",
        "behavior_kl",
        "total_loss",
        "clip_fraction",
        "mean_prefix_entropy_nats",
    ):
        assert name in terms
        assert torch.isfinite(terms[name]), name
    assert torch.allclose(total, terms["total_loss"])


def test_the_total_is_exactly_the_weighted_sum(setup_model, config, completed_episodes):
    batch = build_batch(completed_episodes[:8], alpha=0.1)
    total, terms = setup_batch_loss(setup_model, batch, config=config)
    rebuilt = (
        terms["policy_loss"]
        + config.value_loss_weight * terms["value_loss"]
        + config.conditional_entropy_loss_weight * terms["conditional_entropy_loss"]
        + config.behavior_kl_coefficient * terms["behavior_kl"]
    )
    assert torch.allclose(total, rebuilt, atol=1e-6)


def test_the_ratio_is_one_when_the_model_is_still_the_behavior_snapshot(
    setup_model, config, completed_episodes
):
    """Behavior probabilities come from the recorded snapshot, never a re-run."""
    batch = build_batch(completed_episodes[:8], alpha=0.1)
    _, terms = setup_batch_loss(setup_model, batch, config=zero_kl(config))
    assert float(terms["ratio_mean"].detach()) == pytest.approx(1.0, abs=1e-4)
    assert float(terms["behavior_kl"].detach()) == pytest.approx(0.0, abs=1e-5)
    assert float(terms["clip_fraction"].detach()) == 0.0


def test_masked_types_hold_no_probability_in_the_loss_path(
    setup_model, config, completed_episodes
):
    batch = build_batch(completed_episodes[:4], alpha=0.1)
    outputs = setup_model(batch.sequence)
    from stratego.training.phase17.setup_contract import MASKED_LOGIT

    excluded = outputs["piece_logits"].to(torch.float32).masked_fill(~batch.masks, MASKED_LOGIT)
    probabilities = torch.softmax(excluded, dim=-1)
    assert float(probabilities[~batch.masks].max().detach()) == 0.0


def test_reversing_only_the_outcomes_reverses_the_policy_gradient(
    setup_model, config, completed_episodes
):
    """Section 7's reward-flip test.

    Set up so the assertion is exact. Two things are neutralised first:
    `alpha = 0`, because the entropy half of `delta` is outcome-independent
    and would contaminate the comparison; and a uniform W/D/L prediction, so
    `delta` reduces to the outcome itself.

    The model is the one that generated the episodes, so every ratio is 1 and
    the PPO clip is inactive. That matters: `-min(r*delta, clip(r)*delta)` is
    deliberately asymmetric in the sign of `delta`, so once the policy has
    moved off the behavior snapshot, flipping the outcomes reverses the
    gradient's direction without exactly negating it. The off-policy case is
    asserted separately below.
    """
    model = copy.deepcopy(setup_model)
    neutral = np.full((SETUP_PREFIXES, 3), 1.0 / 3.0, dtype=np.float32)

    def prepare(sign: int):
        episodes = []
        for episode in completed_episodes[:8]:
            clone = copy.deepcopy(episode)
            clone.prefix_wdl_predictions = neutral.copy()
            if clone.outcome == 0:
                clone.outcome = 1  # a draw carries no sign to flip
            clone.outcome = sign * clone.outcome
            episodes.append(clone)
        return episodes

    def policy_gradient(episodes, target):
        batch = build_batch(episodes, alpha=0.0)
        _, terms = setup_batch_loss(target, batch, config=zero_kl(config))
        target.zero_grad(set_to_none=True)
        terms["policy_loss"].backward()
        return target.piece_head.weight.grad.detach().clone()

    forward = policy_gradient(prepare(1), model)
    reversed_ = policy_gradient(prepare(-1), model)
    assert forward.abs().max() > 0.0
    assert torch.allclose(forward, -reversed_, atol=1e-6)


def test_the_outcome_enters_the_surrogate_linearly_even_off_policy(
    config, completed_episodes
):
    """The flip negates the UNCLIPPED surrogate exactly, at any policy age.

    Recorded because the clipped objective does not have this property and it
    is easy to assume it does. `-min(r*delta, clip(r)*delta)` is asymmetric in
    the sign of `delta`: once the policy has moved off the behavior snapshot,
    flipping the outcomes changes *which rows are gradient-active* rather than
    negating the gradient. Measured here at a 0.2 clip on a deliberately
    far-off-policy model, the flipped gradient has positive cosine similarity
    with the original -- so "reward flip reverses the gradient" is only true
    of the surrogate, and the gate reports it that way.
    """
    model = build_setup_model(seed=4242)
    neutral = np.full((SETUP_PREFIXES, 3), 1.0 / 3.0, dtype=np.float32)
    wide = config.replace(ppo_clip_epsilon=1e6)

    def gradient(sign: int, active_config):
        episodes = []
        for episode in completed_episodes[:8]:
            clone = copy.deepcopy(episode)
            clone.prefix_wdl_predictions = neutral.copy()
            clone.outcome = sign * (clone.outcome or 1)
            episodes.append(clone)
        batch = build_batch(episodes, alpha=0.0)
        _, terms = setup_batch_loss(model, batch, config=zero_kl(active_config))
        model.zero_grad(set_to_none=True)
        terms["policy_loss"].backward()
        return (
            model.piece_head.weight.grad.detach().clone(),
            float(terms["clip_fraction"].detach()),
        )

    forward, _ = gradient(1, wide)
    reversed_, _ = gradient(-1, wide)
    assert forward.abs().max() > 0.0
    assert torch.allclose(forward, -reversed_, atol=1e-6)

    # And the clip really is active here, so the contrast is not hypothetical.
    _, clip_fraction = gradient(1, config)
    assert clip_fraction > 0.5


def test_flipping_the_outcome_flips_the_outcome_term_exactly(completed_episodes):
    episode = copy.deepcopy(completed_episodes[0])
    assert episode.outcome != 0
    before = setup_advantage(episode, alpha=0.0)
    episode.outcome = -episode.outcome
    after = setup_advantage(episode, alpha=0.0)
    expected = -np.asarray(before) + 2 * (
        -expected_value_from_wdl(episode.prefix_wdl_predictions)
    )
    assert np.allclose(after, expected, atol=1e-5)


# -- the fixed behavior-KL coefficient --------------------------------------


def test_the_kl_direction_is_the_papers_reverse_kl(config):
    assert config.kl_direction == "reverse_current_given_behavior"


def test_a_flipped_kl_direction_is_refused():
    """The move controller measures the FORWARD KL; the two must never merge."""
    with pytest.raises(Phase17SetupError, match="silently flip"):
        SetupTrainingConfig(run_id="RUN-TEST-A", kl_direction="forward")


def test_the_coefficient_is_the_fixed_d10_value(config):
    assert SETUP_BEHAVIOR_KL_COEFFICIENT == 0.1
    assert config.behavior_kl_coefficient == 0.1


def test_there_is_no_adaptive_controller_left_on_the_active_path():
    """D10 section 1 retires the controller, not just its constants.

    Agent 4's soak sat at the controller's upper bound for 97.5% of its
    iterations. The decision reads that as evidence for deleting it, so the
    symbol has to be gone -- a dormant controller a config flag could revive
    is the second recipe D10 section 2 forbids.
    """
    import stratego.training.phase17.setup_contract as contract
    import stratego.training.phase17.setup_learning as learning

    assert not hasattr(learning, "SetupKLController")
    for retired in (
        "SETUP_KL_TARGET",
        "SETUP_KL_BETA_INITIAL",
        "SETUP_KL_BETA_BOUNDS",
        "SETUP_KL_HARD_LIMIT",
        "SETUP_KL_CONTROLLER_VERSION",
    ):
        assert not hasattr(contract, retired), retired
    for field in ("kl_target", "kl_beta_initial", "kl_beta_bounds", "kl_hard_limit"):
        assert not hasattr(config_for_test(), field), field


def config_for_test():
    return SetupTrainingConfig(run_id="RUN-TEST-A")


def test_the_coefficient_is_constant_across_every_epoch(config, completed_episodes):
    model = build_setup_model(seed=7)
    trainer = SetupTrainer(model, config)
    _load(trainer, completed_episodes[:16])
    result = trainer.update(global_iteration=1)
    coefficients = {epoch["behavior_kl_coefficient"] for epoch in result.epochs}
    assert coefficients == {0.1}
    assert result.behavior_kl_coefficient == 0.1
    assert result.document()["behavior_kl_is_adaptive"] is False


def test_the_kl_readings_are_telemetry_and_nothing_consumes_them(
    config, completed_episodes, setup_model
):
    """Both readings are reported; neither feeds a controller.

    The split still matters for reading a run: on the live path epoch 0 starts
    exactly on the behavior snapshot, so its KL is near zero by construction
    and the mean across epochs understates the drift. The final epoch is where
    the policy actually ended up. Measured here on the model that generated the
    episodes, which is what the tandem runner always has.
    """
    trainer = SetupTrainer(copy.deepcopy(setup_model), config)
    _load(trainer, completed_episodes[:16])
    result = trainer.update(global_iteration=1)
    assert len(result.per_epoch_kl) == 5
    assert result.final_epoch_kl == pytest.approx(result.per_epoch_kl[-1])
    # Epoch 0 begins on the snapshot, so its reading is near zero by
    # construction rather than by evidence -- orders of magnitude below the
    # drift the final epoch measures.
    assert result.per_epoch_kl[0] < 0.01 * result.per_epoch_kl[-1]
    assert result.mean_iteration_kl < result.final_epoch_kl
    assert result.behavior_kl_coefficient == 0.1


# -- the trainer ------------------------------------------------------------


def _load(trainer, episodes):
    for episode in episodes:
        trainer.queue.enqueue(episode)


def test_completed_outcomes_produce_steps_gradients_and_a_changed_digest(
    config, completed_episodes
):
    """The outcome-to-update path, end to end."""
    model = build_setup_model(seed=7)
    trainer = SetupTrainer(model, config)
    _load(trainer, completed_episodes[:16])
    result = trainer.update(global_iteration=1)
    assert result.skipped is False
    assert result.episodes_consumed == 16
    assert result.optimizer_steps == config.epochs_per_iteration * 2
    assert result.gradient_norm_mean > 0.0
    assert result.digest_before != result.digest_after
    assert len(result.epochs) == 5


def test_five_epochs_is_the_default(config):
    assert config.epochs_per_iteration == 5


def test_every_completed_episode_is_consumed_exactly_once(config, completed_episodes):
    """D10 section 4: all of them, both sides, once -- no quota, no leftovers."""
    model = build_setup_model(seed=7)
    trainer = SetupTrainer(model, config)
    _load(trainer, completed_episodes)
    result = trainer.update(global_iteration=1)
    assert result.episodes_consumed == len(completed_episodes)
    assert len(trainer.queue) == 0
    # And a second update in the same state finds nothing left to train on.
    again = trainer.update(global_iteration=2)
    assert again.skipped is True
    assert again.episodes_consumed == 0


def test_an_odd_sized_arrival_is_trained_on_whole_not_truncated(
    config, completed_episodes
):
    """A partial minibatch is trained, not dropped.

    The retired quota consumed a fixed 572 and skipped otherwise, so batch
    sizes were constant by construction. They are not any more: the batch is
    whatever arrived, and the tail shorter than one minibatch has to be a real
    optimizer step rather than a silently discarded remainder.
    """
    model = build_setup_model(seed=7)
    trainer = SetupTrainer(model, config)
    episodes = completed_episodes[:11]  # 8 + 3, against minibatch_episodes = 8
    _load(trainer, episodes)
    result = trainer.update(global_iteration=1)
    assert result.episodes_consumed == 11
    assert result.optimizer_steps == config.epochs_per_iteration * 2
    assert [epoch["minibatches"] for epoch in result.epochs] == [2] * 5


def test_an_empty_buffer_skips_explicitly_and_moves_nothing(config):
    model = build_setup_model(seed=7)
    trainer = SetupTrainer(model, config)
    before = state_dict_digest(model)
    result = trainer.update(global_iteration=4)
    assert result.skipped is True
    assert "no game completed" in result.skip_reason
    assert result.optimizer_steps == 0
    assert state_dict_digest(model) == before
    # A skip still advances the shared index, so alpha keeps annealing.
    assert result.alpha == pytest.approx(setup_alpha(4))


def test_alpha_follows_the_global_iteration_not_the_update_count(
    config, completed_episodes
):
    """Agent 4's carry-forward A4-CF6, settled by D10 section 4.

    The two indices diverge exactly when a setup update is skipped. Under D10
    alpha is a property of where the RUN is, so a trainer that skipped
    iterations 1-4 and updates at 5 uses alpha(5), not alpha(1).
    """
    model = build_setup_model(seed=7)
    trainer = SetupTrainer(model, config)
    for iteration in range(1, 5):
        assert trainer.update(global_iteration=iteration).skipped is True
    _load(trainer, completed_episodes[:8])
    result = trainer.update(global_iteration=5)
    assert result.skipped is False
    assert result.alpha == pytest.approx(setup_alpha(5))
    assert trainer.updates == 1 and trainer.skips == 4


def test_the_ema_updates_once_per_iteration_not_once_per_epoch(config, completed_episodes):
    model = build_setup_model(seed=7)
    trainer = SetupTrainer(model, config)
    _load(trainer, completed_episodes[:16])
    trainer.update(global_iteration=1)
    assert trainer.ema.updates == 1


def test_the_ema_is_not_the_raw_weights(config, completed_episodes):
    model = build_setup_model(seed=7)
    trainer = SetupTrainer(model, config)
    _load(trainer, completed_episodes[:16])
    trainer.update(global_iteration=1)
    assert state_dict_digest(trainer.ema.as_model()) != state_dict_digest(model)


def test_gradients_are_clipped_at_the_setup_norm_not_the_move_norm():
    assert SETUP_GRADIENT_CLIP_NORM == 0.5


def test_identical_data_order_and_state_reproduce_the_update(config, completed_episodes):
    """Section 7: identical data/order/checkpoint state reproduces the update."""
    digests = []
    for _ in range(2):
        model = build_setup_model(seed=31)
        trainer = SetupTrainer(model, config)
        _load(trainer, [copy.deepcopy(episode) for episode in completed_episodes[:16]])
        result = trainer.update(global_iteration=1)
        digests.append((result.digest_after, tuple(map(tuple, result.shuffle_orders))))
    assert digests[0] == digests[1]


def test_the_shuffle_order_is_recorded_and_process_stable(config):
    """A resume in a fresh process must reproduce the same epoch order."""
    from stratego.training.phase17.setup_contract import derive_shuffle_seed

    assert derive_shuffle_seed("RUN-TEST-A", 3, 1) == derive_shuffle_seed("RUN-TEST-A", 3, 1)
    assert derive_shuffle_seed("RUN-TEST-A", 3, 1) != derive_shuffle_seed("RUN-TEST-A", 3, 2)


def test_all_forty_prefixes_are_used_with_no_advantage_filter(completed_episodes):
    batch = build_batch(completed_episodes[:4], alpha=0.1)
    assert batch.advantage.shape == (4, SETUP_PREFIXES)
    assert batch.tokens.shape == (4, SETUP_PREFIXES)


def test_the_setup_state_round_trips_including_the_pending_buffer(
    config, completed_episodes
):
    """Common contract section 10: raw, EMA, optimizer, fixed KL, buffer."""
    model = build_setup_model(seed=55)
    trainer = SetupTrainer(model, config)
    _load(trainer, completed_episodes[:16])
    trainer.update(global_iteration=1)
    _load(trainer, completed_episodes[16:20])
    document = trainer.state_document()

    restored_model = build_setup_model(seed=999)
    restored = SetupTrainer(restored_model, config)
    restored.load_state_document(document)

    assert state_dict_digest(restored.model) == document["setup_raw_model_state_digest"]
    assert state_dict_digest(restored.ema.as_model()) == document["setup_ema_model_state_digest"]
    assert restored.setup_iteration == trainer.setup_iteration
    assert restored.optimizer_step_count == trainer.optimizer_step_count
    assert len(restored.queue) == len(trainer.queue) == 4


def test_the_checkpoint_calls_the_kl_a_fixed_coefficient(config):
    """D10 section 1: not beta, not a target, not controller state."""
    trainer = SetupTrainer(build_setup_model(seed=55), config)
    document = trainer.state_document()
    assert "setup_kl_controller_state" not in document
    assert document["setup_behavior_kl"] == {
        "direction": "reverse_current_given_behavior",
        "coefficient": 0.1,
        "adaptive": False,
    }
    assert document["recipe"] == SETUP_RECIPE_VERSION


def test_a_state_carrying_an_adaptive_controller_is_refused(config):
    trainer = SetupTrainer(build_setup_model(seed=55), config)
    document = trainer.state_document()
    document["setup_behavior_kl"] = {
        "direction": "reverse_current_given_behavior",
        "coefficient": 0.1,
        "adaptive": True,
        "beta_bounds": [0.001, 1.0],
    }
    with pytest.raises(Phase17SetupError, match="ADAPTIVE"):
        trainer.load_state_document(document)


def test_a_state_that_cannot_say_whether_it_is_adaptive_is_refused(config):
    """Fail closed. A `.get("adaptive")` here would read absence as "no"."""
    trainer = SetupTrainer(build_setup_model(seed=55), config)
    document = trainer.state_document()
    document["setup_behavior_kl"] = {
        "direction": "reverse_current_given_behavior",
        "coefficient": 0.1,
    }
    with pytest.raises(Phase17SetupError, match="does not declare"):
        trainer.load_state_document(document)


def test_a_state_from_a_different_coefficient_is_refused(config):
    trainer = SetupTrainer(build_setup_model(seed=55), config)
    document = trainer.state_document()
    document["setup_behavior_kl"] = dict(document["setup_behavior_kl"], coefficient=0.5)
    with pytest.raises(Phase17SetupError, match="coefficient"):
        trainer.load_state_document(document)


def test_a_state_from_the_retired_recipe_is_refused(config):
    """A D7-B checkpoint's advantages came from a different equation."""
    trainer = SetupTrainer(build_setup_model(seed=55), config)
    document = trainer.state_document()
    document["recipe"] = "phase17_setup_update_v2"
    with pytest.raises(Phase17SetupError, match="recipe"):
        trainer.load_state_document(document)


def test_a_state_from_another_run_is_refused(config, completed_episodes):
    model = build_setup_model(seed=55)
    trainer = SetupTrainer(model, config)
    document = trainer.state_document()
    document["run_id"] = "RUN-SOMEONE-ELSE"
    with pytest.raises(Phase17SetupError, match="belongs to run"):
        trainer.load_state_document(document)


def test_a_state_from_another_config_digest_is_refused(config):
    model = build_setup_model(seed=55)
    trainer = SetupTrainer(model, config)
    document = trainer.state_document()
    document["config_digest"] = "0" * 64
    with pytest.raises(Phase17SetupError, match="different setup config digest"):
        trainer.load_state_document(document)


def test_a_partial_state_is_refused(config):
    model = build_setup_model(seed=55)
    trainer = SetupTrainer(model, config)
    document = trainer.state_document()
    del document["setup_optimizer_state"]
    with pytest.raises(Phase17SetupError, match="missing"):
        trainer.load_state_document(document)


def test_a_resumed_trainer_continues_identically(config, completed_episodes):
    """The point of the round trip: the next update must match."""
    first_model = build_setup_model(seed=77)
    first = SetupTrainer(first_model, config)
    _load(first, [copy.deepcopy(e) for e in completed_episodes[:8]])
    first.update(global_iteration=1)
    _load(first, [copy.deepcopy(e) for e in completed_episodes[8:16]])
    document = first.state_document()
    continued = first.update(global_iteration=2)

    second_model = build_setup_model(seed=123)
    second = SetupTrainer(second_model, config)
    second.load_state_document(document)
    resumed = second.update(global_iteration=2)

    assert resumed.digest_after == continued.digest_after
    assert resumed.shuffle_orders == continued.shuffle_orders
    assert resumed.episodes_consumed == continued.episodes_consumed == 8
    assert resumed.epochs[0]["total_loss"] == pytest.approx(
        continued.epochs[0]["total_loss"], abs=1e-9
    )


def test_a_restored_ema_lands_on_the_models_device_not_the_payloads(config):
    """A CPU-serialized EMA restored onto a non-CPU model must move.

    Found by the D10 smoke, which is the first thing to resume on the
    production device. A paired checkpoint writes the EMA to CPU and
    `read_joint_checkpoint` loads it with `map_location="cpu"`, so a restore
    that kept the payload's device left a CPU shadow accumulating against MPS
    parameters. `SetupEMA.update` then raised on the FIRST setup update after
    the resume -- hours into a run, and invisible to a CPU-only rehearsal.

    Asserted on CPU too, where it cannot fail, so the intent survives on a host
    with no accelerator.
    """
    trainer = SetupTrainer(build_setup_model(device=config.device, seed=88), config)
    expected = trainer.ema.device
    document = trainer.state_document()
    assert all(
        tensor.device.type == "cpu" for tensor in document["setup_ema_state"].values()
    ), "the checkpoint must serialize the EMA to CPU"

    restored = SetupTrainer(build_setup_model(device=config.device, seed=99), config)
    restored.load_state_document(document)
    assert all(
        tensor.device == expected for tensor in restored.ema.shadow.values()
    )
    # And the next update runs rather than raising on a device mismatch.
    restored.ema.update(restored.model)
    assert restored.ema.updates == 1


def test_a_restored_ema_lands_on_the_accelerator_when_there_is_one(config):
    """The same claim where it can actually fail: the production device."""
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device on this host")
    accelerated = config.replace(device="mps")
    trainer = SetupTrainer(build_setup_model(device="mps", seed=88), accelerated)
    document = trainer.state_document()
    restored = SetupTrainer(build_setup_model(device="mps", seed=99), accelerated)
    restored.load_state_document(document)
    assert all(
        tensor.device.type == "mps" for tensor in restored.ema.shadow.values()
    )
    restored.ema.update(restored.model)


def test_a_resume_neither_loses_nor_duplicates_a_completed_outcome(
    config, completed_episodes
):
    """D10 section 7 makes this an integrity stop, so it is asserted directly."""
    trainer = SetupTrainer(build_setup_model(seed=77), config)
    _load(trainer, [copy.deepcopy(e) for e in completed_episodes[:6]])
    document = trainer.state_document()

    restored = SetupTrainer(build_setup_model(seed=123), config)
    restored.load_state_document(document)
    taken = restored.queue.consume_all(setup_iteration=1)
    keys = [(episode.game_id, episode.color) for episode in taken]
    assert len(keys) == 6
    assert len(set(keys)) == 6
    assert keys == [(e.game_id, e.color) for e in completed_episodes[:6]]
    assert [e.outcome for e in taken] == [e.outcome for e in completed_episodes[:6]]
