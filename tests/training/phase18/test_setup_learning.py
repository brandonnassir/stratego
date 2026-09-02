"""S12, S16, S17, S18, S25, S26, S27, S28, S29: the loss, the optimizer, the
step counts, clipping, the EMA and the checkpoint round trip."""

import copy

import numpy as np
import pytest
import torch

from stratego.training.phase18.reference_oracle import oracle_ema_closed_form, oracle_step_counts
from stratego.training.phase18.setup_buffer import SetupBuffer
from stratego.training.phase18.setup_contract import (
    ENTROPY_NORMALIZER,
    SETUP_BATCH_SIZE,
    SETUP_EPOCHS_PER_UPDATE,
    Phase18SetupConfigError,
    Phase18SetupError,
    SetupTrainingConfig,
)
from stratego.training.phase18.setup_learning import SetupEMA, SetupTrainer, setup_batch_loss
from stratego.training.phase18.setup_model import build_setup_model, state_dict_digest
from stratego.training.phase18.setup_sampling import generate_pool

from .conftest import NAMESPACE, RUN_ID


def _first_batch(buffer, size=48):
    buffer.process(alpha=0.1)
    return next(buffer.minibatches(size, seed=0))


# -- S12: the entropy head is trained toward I/10 -----------------------------------


def test_the_entropy_target_is_i_over_ten_and_the_head_converges_to_it_not_to_i(setup_model, filled_buffer, config):
    batch = _first_batch(filled_buffer)
    assert torch.allclose(batch.entropy_target * ENTROPY_NORMALIZER, torch.as_tensor(np.stack([
        filled_buffer.samples[filled_buffer.index_of(f)].suffix_information for f in batch.fingerprints
    ])), atol=1e-5)
    model = copy.deepcopy(setup_model)
    head_only = torch.optim.Adam(model.entropy_head.parameters(), lr=0.05)
    for _ in range(300):
        outputs = model(batch.sequence)
        loss = ((outputs["entropy_prediction"] - batch.entropy_target) ** 2).mean()
        head_only.zero_grad()
        loss.backward()
        head_only.step()
    with torch.no_grad():
        fitted = model(batch.sequence)["entropy_prediction"]
    target_mean = float(batch.entropy_target.mean())
    assert abs(float(fitted.mean()) - target_mean) < 0.15 * target_mean
    assert abs(float(fitted.mean()) - ENTROPY_NORMALIZER * target_mean) > 3.0, "it did not converge to I"


# -- S16: PPO ratio and clipping ---------------------------------------------------


def test_a_ratio_of_one_gives_minus_the_mean_advantage(setup_model, filled_buffer, config):
    batch = _first_batch(filled_buffer)
    total, terms = setup_batch_loss(setup_model, batch, config=config)
    assert float(terms["ratio_mean"]) == pytest.approx(1.0, abs=1e-5)
    assert float(terms["policy_loss"]) == pytest.approx(-float(batch.advantage.mean()), abs=1e-4)
    assert float(terms["clip_fraction"]) == 0.0


def test_a_positive_advantage_with_ratio_above_1_2_is_clipped_and_a_negative_below_0_8_is_clipped(setup_model, filled_buffer, config):
    batch = _first_batch(filled_buffer)
    with torch.no_grad():
        _, base = setup_batch_loss(setup_model, batch, config=config)
    # Shift the recorded behavior log-probs so the ratio is 1.5 everywhere.
    shifted = copy.copy(batch)
    shifted.behavior_selected_log_prob = batch.behavior_selected_log_prob - float(np.log(1.5))
    shifted.advantage = torch.ones_like(batch.advantage)
    _, up = setup_batch_loss(setup_model, shifted, config=config)
    assert float(up["policy_loss"]) == pytest.approx(-1.2, abs=1e-4)
    assert float(up["clip_fraction"]) == 1.0
    shifted.behavior_selected_log_prob = batch.behavior_selected_log_prob - float(np.log(0.5))
    shifted.advantage = -torch.ones_like(batch.advantage)
    _, down = setup_batch_loss(setup_model, shifted, config=config)
    assert float(down["policy_loss"]) == pytest.approx(0.8, abs=1e-4)
    # Unclipped direction: a positive advantage with ratio 0.5 is NOT clipped (min picks 0.5 * delta).
    shifted.advantage = torch.ones_like(batch.advantage)
    _, unclipped = setup_batch_loss(setup_model, shifted, config=config)
    assert float(unclipped["policy_loss"]) == pytest.approx(-0.5, abs=1e-4)


# -- S17: reverse KL direction and masking ---------------------------------------


def test_the_kl_is_current_given_behavior_and_illegal_types_contribute_exactly_zero(setup_model, filled_buffer, config):
    batch = _first_batch(filled_buffer)
    other = build_setup_model(seed=77)  # a different current policy
    _, terms = setup_batch_loss(other, batch, config=config)
    with torch.no_grad():
        logits = other(batch.sequence)["piece_logits"].masked_fill(~batch.masks, -1e9)
        log_current = torch.log_softmax(logits, dim=-1)
        p_current = log_current.exp().masked_fill(~batch.masks, 0.0)
        log_behavior = batch.behavior_log_probs
        p_behavior = log_behavior.exp().masked_fill(~batch.masks, 0.0)
        forward = (p_current * (log_current - log_behavior).masked_fill(~batch.masks, 0.0)).sum(-1).mean()
        reverse = (p_behavior * (log_behavior - log_current).masked_fill(~batch.masks, 0.0)).sum(-1).mean()
    assert float(terms["behavior_kl"]) == pytest.approx(float(forward), abs=1e-5)
    assert abs(float(forward) - float(reverse)) > 1e-4, "the fixture is asymmetric"
    # Masking: poison the behavior log-prob of illegal entries; the KL must not move.
    poisoned = copy.copy(batch)
    poisoned.behavior_log_probs = batch.behavior_log_probs.masked_fill(~batch.masks, 50.0)
    _, poisoned_terms = setup_batch_loss(other, poisoned, config=config)
    assert float(poisoned_terms["behavior_kl"]) == pytest.approx(float(terms["behavior_kl"]), abs=1e-7)
    assert float(terms["behavior_kl"]) >= 0.0


# -- S18: loss weights --------------------------------------------------------------


def test_the_four_coefficients_are_frozen_and_the_total_is_their_weighted_sum(setup_model, filled_buffer, config):
    document = config.document()["optimisation"]
    assert (document["policy_loss_weight"], document["value_loss_weight"]) == (1.0, 0.5)
    assert (document["entropy_prediction_loss_weight"], document["behavior_kl"]["coefficient"]) == (1.0, 0.1)
    assert document["behavior_kl"]["adaptive"] is False
    batch = _first_batch(filled_buffer)
    total, terms = setup_batch_loss(build_setup_model(seed=5), batch, config=config)
    expected = terms["policy_loss"] + 0.5 * terms["value_loss"] + 1.0 * terms["entropy_prediction_loss"] + 0.1 * terms["behavior_kl"]
    assert float(total) == pytest.approx(float(expected), abs=1e-6)


# -- S25: AdamW at zero decay is Adam --------------------------------------------------


def test_adam_and_adamw_at_zero_weight_decay_take_identical_steps(setup_model, filled_buffer, config):
    batch = _first_batch(filled_buffer)
    a = copy.deepcopy(setup_model)
    b = copy.deepcopy(setup_model)
    adam = torch.optim.Adam(a.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    adamw = torch.optim.AdamW(b.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    for _ in range(100):
        for model, optimizer in ((a, adam), (b, adamw)):
            total, _ = setup_batch_loss(model, batch, config=config)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
    assert state_dict_digest(a) == state_dict_digest(b)


def test_the_trainer_uses_adamw_with_zero_decay_and_the_config_refuses_a_nonzero_decay(setup_model, config):
    trainer = SetupTrainer(copy.deepcopy(setup_model), config, namespace=NAMESPACE, seed_index=1)
    assert type(trainer.optimizer).__name__ == "AdamW"
    for group in trainer.optimizer.param_groups:
        assert group["weight_decay"] == 0.0 and group["lr"] == 5e-5
        assert tuple(group["betas"]) == (0.9, 0.999) and group["eps"] == 1e-8
    with pytest.raises(Phase18SetupConfigError, match="weight decay"):
        SetupTrainingConfig(run_id="x", weight_decay=0.01)


# -- S26: batch size and epochs ---------------------------------------------------


@pytest.mark.parametrize("ready, batch_size, expected_steps", [(48, 48, 5), (48, 32, 10), (48, 16, 15)])
def test_one_optimizer_step_per_minibatch_per_epoch_and_one_ema_update(pool, outcomes_by_fingerprint, config, ready, batch_size, expected_steps):
    buffer = SetupBuffer(storage_duration=1)
    buffer.add_pool(pool.samples[:ready], period=1)
    for sample in pool.samples[:ready]:
        buffer.add_outcomes((sample.content_fingerprint, z) for z in outcomes_by_fingerprint[sample.content_fingerprint])
    trainer = SetupTrainer(build_setup_model(seed=1), config.replace(batch_size=batch_size), namespace=NAMESPACE, seed_index=1)
    result = trainer.update(buffer, global_iteration=1)
    assert result.optimizer_steps == expected_steps == oracle_step_counts(ready, batch_size, 5)["optimizer_steps"]
    assert result.epochs == SETUP_EPOCHS_PER_UPDATE == 5
    assert result.ema_updates == 1 and trainer.ema.updates == 1


def test_the_production_batch_is_1024_setups_which_is_one_step_per_epoch_at_1024_ready():
    assert SETUP_BATCH_SIZE == 1024
    assert oracle_step_counts(1024, 1024, 5) == {"minibatches_per_epoch": 1, "optimizer_steps": 5, "ema_updates": 1}
    assert oracle_step_counts(1500, 1024, 5)["optimizer_steps"] == 10


# -- S27: gradient clipping ------------------------------------------------------------


def test_the_post_clip_norm_never_exceeds_0_5_and_only_setup_parameters_are_stepped(setup_model, filled_buffer, config):
    trainer = SetupTrainer(copy.deepcopy(setup_model), config, namespace=NAMESPACE, seed_index=1)
    result = trainer.update(filled_buffer, global_iteration=1)
    assert result.pre_clip_grad_norms and all(n <= 0.5 + 1e-5 for n in result.post_clip_grad_norms)
    assert all(post <= pre + 1e-6 for pre, post in zip(result.pre_clip_grad_norms, result.post_clip_grad_norms))
    assert config.gradient_clip_norm == 0.5
    stepped = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert stepped == {id(p) for p in trainer.model.parameters()}


# -- S28: EMA and the raw/EMA split ----------------------------------------------------


def test_the_ema_follows_the_closed_form_and_is_updated_once_per_update(setup_model, filled_buffer, config):
    raw = copy.deepcopy(setup_model)
    ema = SetupEMA(raw, 0.999)
    shadow0 = {k: v.clone() for k, v in ema.state_dict().items()}
    for _ in range(7):
        ema.update(raw)
    for name, tensor in ema.state_dict().items():
        expected = oracle_ema_closed_form(shadow0[name].numpy(), raw.state_dict()[name].numpy(), 0.999, 7)
        assert np.allclose(tensor.numpy(), expected, atol=1e-7)
    trainer = SetupTrainer(copy.deepcopy(setup_model), config.replace(batch_size=8), namespace=NAMESPACE, seed_index=1)
    result = trainer.update(filled_buffer, global_iteration=1)
    assert result.optimizer_steps == 30 and result.ema_updates == 1


def test_the_raw_model_generates_and_the_ema_only_evaluates(setup_model, filled_buffer, config):
    trainer = SetupTrainer(copy.deepcopy(setup_model), config, namespace=NAMESPACE, seed_index=1)
    assert trainer.generation_actor is trainer.model
    trainer.update(filled_buffer, global_iteration=1)
    evaluation = trainer.evaluation_model()
    assert evaluation is not trainer.model
    assert state_dict_digest(evaluation) != state_dict_digest(trainer.model)
    assert not evaluation.training
    # The EMA object never appears in the optimizer's parameter groups.
    stepped = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert not any(id(t) in stepped for t in trainer.ema.shadow.values())


# -- S29: checkpoint round trip on CPU and on the production device ----------------------


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_save_reload_and_one_more_update(device, tmp_path, pool, outcomes_by_fingerprint, config):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available on this host")
    cfg = config.replace(device=device, batch_size=16)
    model = build_setup_model(device=device, seed=21)
    trainer = SetupTrainer(model, cfg, namespace=NAMESPACE, seed_index=1)
    generated = generate_pool(model, namespace=NAMESPACE, seed_index=1, snapshot_iteration=0, snapshot_digest=state_dict_digest(model), count=32, device=device)
    buffer = SetupBuffer(storage_duration=1, device=device)
    buffer.add_pool(generated.samples, period=1)
    for sample in generated.samples:
        buffer.add_outcomes((sample.content_fingerprint, z) for z in (1, -1, 0, 1))
    trainer.update(buffer, global_iteration=1)
    buffer.filter(1)

    manifest = trainer.save_checkpoint(tmp_path / "ckpt")
    assert {"raw", "optimizer", "ema"} <= set(manifest)
    assert manifest["raw"]["state_digest"] == state_dict_digest(trainer.model)
    restored, _ = SetupTrainer.load_checkpoint(tmp_path / "ckpt", cfg, namespace=NAMESPACE, seed_index=1, device=device)
    assert state_dict_digest(restored.model) == state_dict_digest(trainer.model)
    assert state_dict_digest(restored.ema.as_model()) == state_dict_digest(trainer.ema.as_model())
    assert restored.optimizer.state_dict()["state"].keys() == trainer.optimizer.state_dict()["state"].keys()
    assert restored.ema.device.type == torch.device(device).type
    assert restored.updates == 1 and restored.ema.updates == 1 and restored.optimizer_step_count == trainer.optimizer_step_count

    # One further update on the restored trainer, on the same device.
    again = generate_pool(restored.model, namespace=NAMESPACE, seed_index=1, snapshot_iteration=1, snapshot_digest=state_dict_digest(restored.model), count=32, device=device)
    buffer2 = SetupBuffer(storage_duration=1, device=device)
    buffer2.add_pool(again.samples, period=2)
    for sample in again.samples:
        buffer2.add_outcomes((sample.content_fingerprint, z) for z in (1, 1, -1, 0))
    result = restored.update(buffer2, global_iteration=2)
    assert result.optimizer_steps == 10 and restored.ema.updates == 2
    with pytest.raises(Phase18SetupError):
        SetupTrainer.load_checkpoint(tmp_path / "ckpt", cfg.replace(run_id="OTHER"), namespace=NAMESPACE, seed_index=1, device=device)


def test_a_tampered_checkpoint_file_is_refused(tmp_path, setup_model, filled_buffer, config):
    trainer = SetupTrainer(copy.deepcopy(setup_model), config, namespace=NAMESPACE, seed_index=1)
    trainer.save_checkpoint(tmp_path / "ckpt")
    (tmp_path / "ckpt" / "ema.pt").write_bytes(b"corrupt")
    with pytest.raises(Phase18SetupError, match="digest moved"):
        SetupTrainer.load_checkpoint(tmp_path / "ckpt", config, namespace=NAMESPACE, seed_index=1)


def test_a_non_finite_loss_is_fatal(setup_model, filled_buffer, config):
    trainer = SetupTrainer(copy.deepcopy(setup_model), config, namespace=NAMESPACE, seed_index=1)
    with torch.no_grad():
        trainer.model.piece_head.bias.fill_(float("nan"))
    with pytest.raises(Phase18SetupError, match="non-finite"):
        trainer.update(filled_buffer, global_iteration=1)
    assert trainer.non_finite_events == 1
