"""Phase 17 Agent 3 sections 5 and 7: the setup update and its identities."""

import copy

import numpy as np
import pytest
import torch

from stratego.training.phase9_behavior import state_dict_digest
from stratego.training.phase17.setup_contract import (
    ALPHA_FLOOR,
    N_PAPER,
    SETUP_CONDITIONAL_ENTROPY_NORMALIZER,
    SETUP_GRADIENT_CLIP_NORM,
    SETUP_PREFIXES,
    Phase17SetupError,
    SetupTrainingConfig,
    setup_alpha,
    setup_alpha_exponent,
)
from stratego.training.phase17.setup_episode import SetupEpisodeQueue
from stratego.training.phase17.setup_learning import (
    SetupKLController,
    SetupTrainer,
    advantage_terms,
    build_batch,
    expected_value_from_wdl,
    setup_advantage,
    setup_batch_loss,
)
from stratego.training.phase17.setup_model import build_setup_model


# -- the alpha schedule -----------------------------------------------------


def test_alpha_preserves_both_paper_endpoints_at_any_horizon():
    """Operator decision D3: alpha(1) = 0.1 and alpha(N) = the paper's endpoint."""
    for total in (300, 626, 1200):
        assert setup_alpha(1, total) == pytest.approx(0.1)
        assert setup_alpha(total, total) == pytest.approx(ALPHA_FLOOR, rel=1e-9)


def test_the_raw_transcription_would_end_far_more_regularized():
    """Why D3 exists: 0.1 * n**-0.3 on a 626-iteration run ends 3.5x too high."""
    raw = 0.1 * 626 ** -0.3
    assert raw / ALPHA_FLOOR == pytest.approx(3.54, abs=0.01)


def test_alpha_is_monotone_and_floored():
    total = 626
    values = [setup_alpha(n, total) for n in range(1, total + 1)]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(values, values[1:]))
    assert setup_alpha(total * 10, total) == pytest.approx(ALPHA_FLOOR)
    assert setup_alpha_exponent(N_PAPER) == pytest.approx(0.3)


def test_a_degenerate_horizon_is_refused():
    with pytest.raises(Phase17SetupError, match="at least 2 iterations"):
        setup_alpha(1, 1)
    with pytest.raises(Phase17SetupError, match="one-based"):
        setup_alpha(0, 100)


# -- the advantage ----------------------------------------------------------


def test_expected_value_is_win_minus_loss():
    wdl = np.array([[0.6, 0.1, 0.3]], dtype=np.float32)
    assert expected_value_from_wdl(wdl)[0] == pytest.approx(0.3)


def test_the_advantage_is_the_two_frozen_terms(completed_episodes):
    """delta = (o - E[v]) + 0.9 * alpha * (I/10), operator decision D7-B."""
    episode = completed_episodes[0]
    alpha = 0.1
    expected = expected_value_from_wdl(episode.prefix_wdl_predictions)
    manual = (float(episode.outcome) - expected) + 0.9 * alpha * (
        episode.suffix_information_content * SETUP_CONDITIONAL_ENTROPY_NORMALIZER
    )
    assert np.allclose(setup_advantage(episode, alpha), manual, atol=1e-5)


def test_the_entropy_bonus_does_not_depend_on_the_prediction_head(completed_episodes):
    """The whole point of D7-B: the bonus is uncentered.

    Under the retired v1 form the bonus was `alpha*(I/10 - h)`, so it vanished
    as `L_h` drove `h` to `I/10` -- measured falling to ~1/100th of the outcome
    term within ten iterations. Perturbing `h` must now leave the advantage
    untouched.
    """
    episode = copy.deepcopy(completed_episodes[0])
    before = setup_advantage(episode, 0.1)
    episode.prefix_conditional_entropy_predictions = (
        episode.prefix_conditional_entropy_predictions + 5.0
    )
    assert np.allclose(setup_advantage(episode, 0.1), before, atol=1e-6)


def test_a_converged_entropy_head_does_not_kill_the_bonus(completed_episodes):
    """The v1 failure mode, asserted absent.

    Setting `h` exactly equal to `I/10` is the converged state. Under v1 that
    made the entropy term identically zero; under D7-B the bonus is unchanged.
    """
    episode = copy.deepcopy(completed_episodes[0])
    episode.prefix_conditional_entropy_predictions = (
        episode.suffix_information_content * SETUP_CONDITIONAL_ENTROPY_NORMALIZER
    ).astype(np.float32)
    terms = advantage_terms(episode, 0.1)
    assert terms["conditional_entropy_residual_abs_mean"] == pytest.approx(0.0, abs=1e-5)
    assert terms["v1_centered_entropy_term_abs_mean"] == pytest.approx(0.0, abs=1e-6)
    assert terms["entropy_term_abs_mean"] > 0.1  # the D7-B bonus survives


def test_the_bonus_is_commensurate_with_the_outcome_term(completed_episodes):
    """D7-B satisfies both of D4's stated constraints.

    Agent 1 rejected the paper's literal `0.9*alpha*H` because in RAW NATS it
    is about 5.6 at alpha = 0.1, against an outcome term bounded by 2. The same
    uncentered shape in normalized `I/10` units is about 0.56 -- the same
    mechanism at a commensurate scale.
    """
    terms = advantage_terms(completed_episodes[0], 0.1)
    raw_nats_form = 0.9 * 0.1 * terms["information_mean"]
    assert raw_nats_form > 2.0  # what Agent 1 correctly refused
    assert 0.2 < terms["entropy_term_abs_mean"] < 2.0  # what D7-B produces
    assert terms["entropy_term_max_abs"] < 2.0


def test_the_entropy_head_is_still_trained(setup_model, config, completed_episodes):
    """`L_h` is retained for telemetry and paper alignment, so it must still
    reach the head's parameters."""
    model = build_setup_model(seed=606)
    batch = build_batch(completed_episodes[:8], alpha=0.1)
    _, terms = setup_batch_loss(model, batch, config=config, beta=0.0)
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
    total, terms = setup_batch_loss(setup_model, batch, config=config, beta=0.1)
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
    beta = 0.07
    total, terms = setup_batch_loss(setup_model, batch, config=config, beta=beta)
    rebuilt = (
        terms["policy_loss"]
        + config.value_loss_weight * terms["value_loss"]
        + config.conditional_entropy_loss_weight * terms["conditional_entropy_loss"]
        + beta * terms["behavior_kl"]
    )
    assert torch.allclose(total, rebuilt, atol=1e-6)


def test_the_ratio_is_one_when_the_model_is_still_the_behavior_snapshot(
    setup_model, config, completed_episodes
):
    """Behavior probabilities come from the recorded snapshot, never a re-run."""
    batch = build_batch(completed_episodes[:8], alpha=0.1)
    _, terms = setup_batch_loss(setup_model, batch, config=config, beta=0.0)
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
        _, terms = setup_batch_loss(target, batch, config=config, beta=0.0)
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
        _, terms = setup_batch_loss(model, batch, config=active_config, beta=0.0)
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


# -- the KL controller ------------------------------------------------------


def test_the_controller_direction_is_the_papers_reverse_kl(config):
    controller = SetupKLController.from_config(config)
    assert controller.direction == "reverse_current_given_behavior"
    assert controller.direction != "forward"


def test_the_d5_constants_are_the_operators_resolved_values(config):
    """Operator decision D5, 2026-08-27, from the measured soak."""
    from stratego.training.phase17.setup_contract import (
        SETUP_KL_BETA_BOUNDS,
        SETUP_KL_BETA_INITIAL,
        SETUP_KL_HARD_LIMIT,
        SETUP_KL_TARGET,
    )

    assert SETUP_KL_TARGET == 0.0018
    assert SETUP_KL_BETA_INITIAL == 0.1
    assert SETUP_KL_BETA_BOUNDS == (0.001, 1.0)
    assert SETUP_KL_HARD_LIMIT == 0.08
    assert (config.kl_target, config.kl_beta_initial) == (0.0018, 0.1)
    assert (config.kl_beta_bounds, config.kl_hard_limit) == ((0.001, 1.0), 0.08)


def test_the_retargeted_controller_has_room_to_move_in_both_directions(config):
    """The v1 failure: beta pinned at its lower bound for 100% of iterations.

    With the target at the measured scale and the lower bound a decade wider,
    beta must be able to fall from 0.1 without immediately hitting the floor,
    and to rise again.
    """
    controller = SetupKLController.from_config(config)
    assert controller.beta == 0.1
    steps_to_floor = 0
    while controller.beta > controller.beta_min and steps_to_floor < 100:
        controller.update(0.0)
        steps_to_floor += 1
    assert steps_to_floor > 8  # a decade of headroom, not one step
    controller.beta = 0.1
    controller.update(config.kl_target * 10)
    assert controller.beta > 0.1  # and it can still rise


def test_beta_rises_on_a_high_kl_and_falls_on_a_low_one(config):
    controller = SetupKLController.from_config(config)
    start = controller.beta
    controller.update(config.kl_target * 10)
    assert controller.beta > start
    for _ in range(20):
        controller.update(0.0)
    assert controller.beta == pytest.approx(config.kl_beta_bounds[0])
    assert controller.history[-1]["at_lower_bound"] is True


def test_the_hard_limit_streak_is_counted(config):
    controller = SetupKLController.from_config(config)
    controller.update(config.kl_hard_limit * 2)
    controller.update(config.kl_hard_limit * 2)
    assert controller.consecutive_over_hard_limit == 2
    controller.update(0.0)
    assert controller.consecutive_over_hard_limit == 0


def test_the_controller_round_trips(config):
    controller = SetupKLController.from_config(config)
    controller.update(0.05)
    restored = SetupKLController.from_state_document(controller.state_document())
    assert restored.beta == controller.beta
    assert restored.direction == controller.direction
    assert restored.history == controller.history


# -- the trainer ------------------------------------------------------------


def _load(trainer, episodes):
    for episode in episodes:
        trainer.queue.enqueue(episode)


def test_completed_outcomes_produce_steps_gradients_and_a_changed_digest(
    config, completed_episodes
):
    """Section 7: the outcome-to-update path, end to end."""
    model = build_setup_model(seed=7)
    trainer = SetupTrainer(model, config)
    _load(trainer, completed_episodes[:16])
    result = trainer.update(batch_episodes=16)
    assert result.skipped is False
    assert result.episodes_consumed == 16
    assert result.optimizer_steps == config.epochs_per_iteration * 2
    assert result.gradient_norm_mean > 0.0
    assert result.digest_before != result.digest_after
    assert len(result.epochs) == 5


def test_five_epochs_is_the_default(config):
    assert config.epochs_per_iteration == 5


def test_a_starved_queue_skips_explicitly(config, completed_episodes):
    model = build_setup_model(seed=7)
    trainer = SetupTrainer(model, config)
    _load(trainer, completed_episodes[:2])
    before = state_dict_digest(model)
    result = trainer.update(batch_episodes=16)
    assert result.skipped is True
    assert "queue held 2" in result.skip_reason
    assert result.optimizer_steps == 0
    assert state_dict_digest(model) == before


def test_the_ema_updates_once_per_iteration_not_once_per_epoch(config, completed_episodes):
    model = build_setup_model(seed=7)
    trainer = SetupTrainer(model, config)
    _load(trainer, completed_episodes[:16])
    trainer.update(batch_episodes=16)
    assert trainer.ema.updates == 1


def test_the_ema_is_not_the_raw_weights(config, completed_episodes):
    model = build_setup_model(seed=7)
    trainer = SetupTrainer(model, config)
    _load(trainer, completed_episodes[:16])
    trainer.update(batch_episodes=16)
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
        result = trainer.update(batch_episodes=16)
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


def test_the_setup_state_round_trips_including_queue_and_controller(
    config, completed_episodes
):
    """Section 7 / common contract section 10: raw, EMA, optimizer, KL, queue."""
    model = build_setup_model(seed=55)
    trainer = SetupTrainer(model, config)
    _load(trainer, completed_episodes[:16])
    trainer.update(batch_episodes=8)
    _load(trainer, completed_episodes[16:20])
    document = trainer.state_document()

    restored_model = build_setup_model(seed=999)
    restored = SetupTrainer(restored_model, config)
    restored.load_state_document(document)

    assert state_dict_digest(restored.model) == document["setup_raw_model_state_digest"]
    assert state_dict_digest(restored.ema.as_model()) == document["setup_ema_model_state_digest"]
    assert restored.setup_iteration == trainer.setup_iteration
    assert restored.optimizer_step_count == trainer.optimizer_step_count
    assert restored.controller.beta == trainer.controller.beta
    assert restored.controller.direction == trainer.controller.direction
    assert len(restored.queue) == len(trainer.queue)


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
    _load(first, [copy.deepcopy(e) for e in completed_episodes[:16]])
    first.update(batch_episodes=8)
    document = first.state_document()
    continued = first.update(batch_episodes=8)

    second_model = build_setup_model(seed=123)
    second = SetupTrainer(second_model, config)
    second.load_state_document(document)
    resumed = second.update(batch_episodes=8)

    assert resumed.digest_after == continued.digest_after
    assert resumed.shuffle_orders == continued.shuffle_orders
    assert resumed.epochs[0]["total_loss"] == pytest.approx(
        continued.epochs[0]["total_loss"], abs=1e-9
    )
