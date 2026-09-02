"""Part F: the implementation-independent canned parity oracle against the
production buffer, loss and trainer."""

import ast
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from stratego.engine.constants import FLAG
from stratego.training.phase18 import reference_oracle as oracle
from stratego.training.phase18.setup_buffer import SetupBuffer, softmax
from stratego.training.phase18.setup_contract import ENTROPY_NORMALIZER, START_TOKEN
from stratego.training.phase18.setup_learning import setup_batch_loss
from stratego.training.phase18.setup_model import build_setup_model

from .conftest import batch_to_numpy

COEFFICIENTS = {"clip_epsilon": 0.2, "policy_weight": 1.0, "value_weight": 0.5, "entropy_weight": 1.0, "kl_weight": 0.1}
ALPHA = 0.1


def test_the_oracle_never_imports_the_production_loss_or_buffer():
    source = Path(oracle.__file__).read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            text = ast.unparse(node)
            assert "setup_learning" not in text and "setup_buffer" not in text and "synthetic" not in text, text


@pytest.fixture(scope="module")
def canned(pool, outcomes_by_fingerprint):
    """Six pooled setups, their outcomes, and the production batch built from them."""
    samples = pool.samples[:6]
    buffer = SetupBuffer(storage_duration=1)
    buffer.add_pool(samples, period=1)
    for sample in samples:
        buffer.add_outcomes((sample.content_fingerprint, z) for z in outcomes_by_fingerprint[sample.content_fingerprint])
    processed = buffer.process(alpha=ALPHA)
    batch = next(buffer.minibatches(64, seed=0))
    by_fingerprint = {s.content_fingerprint: s for s in samples}
    return {"samples": samples, "buffer": buffer, "processed": processed, "batch": batch, "fixture": batch_to_numpy(batch), "by_fingerprint": by_fingerprint, "outcomes": outcomes_by_fingerprint}


# -- masks, alignment, information, expected value, aggregation, advantage --------------


def test_masks_and_information_and_expected_values_and_aggregation_and_advantage_match(canned):
    fixture, buffer = canned["fixture"], canned["buffer"]
    for row, fingerprint in enumerate(fixture["fingerprints"]):
        sample = canned["by_fingerprint"][fingerprint]
        tokens = fixture["tokens"][row]
        assert np.array_equal(oracle.oracle_legal_masks(tokens), fixture["masks"][row])
        assert fixture["sequence"][row, 0] == START_TOKEN and np.array_equal(fixture["sequence"][row, 1:], tokens)
        information = oracle.oracle_suffix_information(sample.behavior_selected_log_prob)
        assert np.allclose(information, sample.suffix_information, atol=1e-4)
        assert np.allclose(information / ENTROPY_NORMALIZER, fixture["entropy_target"][row], atol=1e-5)
        expected = oracle.oracle_expected_value(oracle.oracle_softmax(sample.wdl_logits))
        mean, count, z_bar = oracle.oracle_running_mean(canned["outcomes"][fingerprint])
        assert count == 4 and np.allclose(mean, fixture["value_target"][row], atol=1e-9)
        assert buffer.outcome_record(fingerprint)["z_bar"] == pytest.approx(z_bar)
        advantage = oracle.oracle_flat_advantage(z_bar, expected, information, sample.entropy_prediction, ALPHA)
        assert np.allclose(advantage, fixture["advantage"][row], atol=1e-4)
        recursion = oracle.oracle_published_recursion(mean, oracle.oracle_softmax(sample.wdl_logits), -sample.behavior_selected_log_prob.astype(np.float64), sample.entropy_prediction, td_lambda=1.0, gae_lambda=1.0, reg_temp=ALPHA, reg_norm=ENTROPY_NORMALIZER)
        assert np.allclose(recursion["advantage"], fixture["advantage"][row], atol=1e-4)


def test_i_minus_ten_h_in_the_oracle_is_the_published_denormalisation(canned):
    sample = canned["samples"][0]
    information = oracle.oracle_suffix_information(sample.behavior_selected_log_prob)
    residual = oracle.oracle_flat_advantage(0.0, np.zeros(40), information, sample.entropy_prediction, 1.0)
    assert np.allclose(residual, information - 10.0 * sample.entropy_prediction.astype(np.float64), atol=1e-9)


# -- the loss ----------------------------------------------------------------------


def test_every_loss_term_and_the_total_match_the_oracle_in_double_precision(canned, config, setup_model):
    double = copy.deepcopy(setup_model).to(torch.float64).eval()
    total, terms = setup_batch_loss(double, canned["batch"], config=config)
    expected = oracle.oracle_loss_from_model(setup_model, canned["fixture"], COEFFICIENTS)
    for name in ("policy_loss", "value_loss", "entropy_prediction_loss", "behavior_kl", "total_loss"):
        assert float(terms[name]) == pytest.approx(expected[name], rel=1e-9, abs=1e-9), name
    assert float(terms["clip_fraction"]) == pytest.approx(expected["clip_fraction"])


def test_the_oracle_ppo_clips_in_both_directions_and_the_kl_direction_is_current_given_behavior():
    logits = np.zeros((1, 40, 12))
    logits[0, :, 3] = 2.0
    masks = np.ones((1, 40, 12), dtype=bool)
    tokens = np.full((1, 40), 3)
    log_pi = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    behavior = np.zeros((1, 40, 12)) - np.log(12.0)  # uniform behavior
    fixture = dict(
        piece_logits=logits, wdl_logits=np.zeros((1, 40, 3)), entropy_prediction=np.zeros((1, 40)), tokens=tokens, masks=masks,
        behavior_log_probs=behavior, behavior_selected_log_prob=behavior[0, :, 3][None], advantage=np.ones((1, 40)),
        value_target=np.array([[0.0, 0.0, 1.0]]), entropy_target=np.zeros((1, 40)),
    )
    terms = oracle.oracle_loss_terms(**fixture, **COEFFICIENTS)
    ratio = float(np.exp(log_pi[0, 0, 3] + np.log(12.0)))
    assert ratio > 1.2 and terms["policy_loss"] == pytest.approx(-1.2)
    p = np.exp(log_pi[0, 0])
    forward = float((p * (log_pi[0, 0] - behavior[0, 0])).sum())
    reverse = float((np.exp(behavior[0, 0]) * (behavior[0, 0] - log_pi[0, 0])).sum())
    assert terms["behavior_kl"] == pytest.approx(forward) and abs(forward - reverse) > 1e-3


# -- gradients -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parameter_name, flat_index",
    [
        ("piece_head.bias", FLAG),
        ("piece_head.weight", 5),
        ("wdl_head.bias", 2),
        ("entropy_head.bias", 0),
        ("entropy_head.weight", 17),
        ("layers.0.attention.query.weight", 130),
        ("layers.3.feed_forward.2.bias", 9),
        ("token_embedding.weight", START_TOKEN * 128 + 4),
        ("positional_embedding.weight", 5 * 128 + 3),
        ("final_norm.weight", 40),
    ],
)
def test_production_gradients_match_central_finite_differences_of_the_oracle_loss(canned, config, setup_model, parameter_name, flat_index):
    double = copy.deepcopy(setup_model).to(torch.float64)
    double.train()
    total, _ = setup_batch_loss(double, canned["batch"], config=config)
    total.backward()
    autograd = float(dict(double.named_parameters())[parameter_name].grad.view(-1)[flat_index])
    numeric = oracle.oracle_finite_difference_gradient(setup_model, canned["fixture"], COEFFICIENTS, parameter_name, flat_index, epsilon=1e-4)
    assert autograd == pytest.approx(numeric, rel=1e-5, abs=1e-7), (parameter_name, flat_index)


# -- optimizer, clipping, EMA, steps ----------------------------------------------------


def test_torch_adamw_at_zero_decay_matches_the_oracle_adam_update():
    torch.manual_seed(0)
    parameter = torch.nn.Parameter(torch.randn(6, dtype=torch.float64))
    optimizer = torch.optim.AdamW([parameter], lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    values = parameter.detach().numpy().copy()
    state: dict = {}
    for step in range(5):
        gradient = np.array([0.3, -0.2, 0.05, 1.0, -0.7, 0.0]) * (step + 1)
        optimizer.zero_grad()
        parameter.grad = torch.as_tensor(gradient)
        optimizer.step()
        values, state = oracle.oracle_adam_step(values, gradient, state, lr=5e-5, betas=(0.9, 0.999), eps=1e-8)
        assert np.allclose(parameter.detach().numpy(), values, atol=1e-12)


def test_clip_grad_norm_scale_matches_the_oracle(setup_model, canned, config):
    model = copy.deepcopy(setup_model)
    total, _ = setup_batch_loss(model, canned["batch"], config=config)
    total.backward()
    norms = [float(p.grad.norm()) for p in model.parameters()]
    scale = oracle.oracle_clip_scale(norms, 0.5)
    before = [p.grad.clone() for p in model.parameters()]
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    for grad, prior in zip((p.grad for p in model.parameters()), before):
        assert torch.allclose(grad, prior * scale, atol=1e-6)


def test_step_counts_and_ema_cadence_follow_the_published_loop():
    assert oracle.oracle_step_counts(1024, 1024, 5) == {"minibatches_per_epoch": 1, "optimizer_steps": 5, "ema_updates": 1}
    assert oracle.oracle_step_counts(2048, 1024, 5)["optimizer_steps"] == 10
    assert oracle.oracle_ema_closed_form(1.0, 0.0, 0.999, 3) == pytest.approx(0.999 ** 3)
