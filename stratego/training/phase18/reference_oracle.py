"""Phase 18 Gate G2: the implementation-independent canned parity oracle.

This module recomputes every quantity of the published setup update from
first principles, in numpy float64, and NEVER imports or calls the production
loss, buffer or trainer (`setup_learning`, `setup_buffer`). The only
production objects it touches are the model's forward pass (to obtain logits
for a canned batch) and the contract's constants (so the two sides agree on
what the numbers are supposed to be). `tests/training/phase18/test_reference_oracle.py`
scans this file's imports to pin that boundary.

Everything here is a literal transcription of the published implementation
at commit `92db29e8ffc323b1b8a2804b5c3f84695d036b05`:

```text
oracle_published_recursion    arrangement/buffer.py process_data, lines 311-352
oracle_running_mean           arrangement/buffer.py add_rewards, lines 259-271
oracle_loss_terms             core/rl.py arr_train, lines 633-679
oracle_adam_step              torch.optim.Adam / AdamW at weight_decay 0
oracle_clip_scale             torch.nn.utils.clip_grad_norm_
oracle_ema_closed_form        networks/exponential_weighted_average.py
```
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from ...engine.constants import FLAG, NUM_PIECE_TYPES, PIECE_COUNTS, PIECES_PER_PLAYER
from .setup_contract import (
    CATEGORICAL_AGGREGATION,
    ENTROPY_NORMALIZER,
    FLAG_PERMITTED_FILES,
    MASKED_LOGIT,
    SETUP_PREFIXES,
    START_TOKEN,
)

_AGG = np.array(CATEGORICAL_AGGREGATION, dtype=np.float64)


# ---------------------------------------------------------------------------
# Masks and alignment (S01, S02, S04)
# ---------------------------------------------------------------------------


def oracle_inventory_mask(prefix_tokens) -> np.ndarray:
    """Recount the inventory from scratch: legal iff at least one piece of
    the type remains after the prefix."""
    remaining = {t: PIECE_COUNTS[t] for t in range(NUM_PIECE_TYPES)}
    for token in prefix_tokens:
        remaining[int(token)] -= 1
    return np.array([remaining[t] > 0 for t in range(NUM_PIECE_TYPES)], dtype=bool)


def oracle_handedness_mask(square: int) -> np.ndarray:
    """The Flag is legal only on the permitted files of the square's rank."""
    mask = np.ones(NUM_PIECE_TYPES, dtype=bool)
    if (int(square) % 10) not in FLAG_PERMITTED_FILES:
        mask[FLAG] = False
    return mask


def oracle_legal_masks(tokens) -> np.ndarray:
    """`[40, 12]`: inventory AND handedness at every prefix of one setup."""
    tokens = [int(t) for t in tokens]
    return np.stack([oracle_inventory_mask(tokens[:k]) & oracle_handedness_mask(k) for k in range(SETUP_PREFIXES)])


# ---------------------------------------------------------------------------
# Information, expected value, aggregation, advantage (S08, S09, S11-S15)
# ---------------------------------------------------------------------------


def oracle_suffix_information(selected_log_probs) -> np.ndarray:
    """`I_k = -sum_{j>=k} log pi_b(t_j)` written as an explicit double loop."""
    values = [float(v) for v in selected_log_probs]
    return np.array([-sum(values[j] for j in range(k, len(values))) for k in range(len(values))], dtype=np.float64)


def oracle_softmax(logits) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def oracle_expected_value(wdl_probabilities) -> np.ndarray:
    """`E[v] = p_win - p_loss` with the published (loss, draw, win) order."""
    p = np.asarray(wdl_probabilities, dtype=np.float64)
    return p[..., 2] - p[..., 0]


def oracle_running_mean(outcomes) -> tuple:
    """The published per-row running mean of one-hot outcomes, step by step.
    Returns `(mean_one_hot, count, z_bar)`."""
    mean = np.zeros(3, dtype=np.float64)
    count = 0
    for z in outcomes:
        one_hot = np.zeros(3, dtype=np.float64)
        one_hot[int(z) + 1] = 1.0
        mean = (count * mean + one_hot) / (count + 1)
        count += 1
    return mean, count, float(mean @ _AGG)


def oracle_flat_advantage(z_bar: float, expected_values, information, h_normalized, alpha: float) -> np.ndarray:
    """`delta_k = (z_bar - E[v_k]) + alpha * (I_k - 10 h_k)`."""
    e = np.asarray(expected_values, dtype=np.float64)
    i = np.asarray(information, dtype=np.float64)
    h = np.asarray(h_normalized, dtype=np.float64)
    return (z_bar - e) + alpha * (i - ENTROPY_NORMALIZER * h)


def oracle_published_recursion(
    reward_one_hot, values_probabilities, nll, ents_normalized, *, td_lambda: float, gae_lambda: float, reg_temp: float, reg_norm: float
) -> dict:
    """Literal transcription of `ArrangementBuffer.process_data` for one row.

    `reward_one_hot` [3] is the averaged one-hot outcome; `values_probabilities`
    [40, 3] the softmaxed behavior W/D/L predictions; `nll` [40] the chosen-token
    negative log-likelihoods; `ents_normalized` [40] the stored h. Returns the
    aggregated advantage, the value estimate and the renormalised entropy
    target, exactly as the buffer stores them.
    """
    rewards = np.asarray(reward_one_hot, dtype=np.float64)
    values = np.asarray(values_probabilities, dtype=np.float64)
    nll = np.asarray(nll, dtype=np.float64)
    ents = reg_norm * np.asarray(ents_normalized, dtype=np.float64)
    n = SETUP_PREFIXES
    adv_est = np.zeros((n, 3), dtype=np.float64)
    val_est = np.zeros((n, 3), dtype=np.float64)
    reg_val_est = np.zeros(n, dtype=np.float64)
    for step in range(n - 1, -1, -1):
        if step == n - 1:
            delta = rewards - values[step]
            td_trace = delta
            gae_trace = delta
        else:
            delta = values[step + 1] - values[step]
            td_trace = delta + td_lambda * td_trace
            gae_trace = delta + gae_lambda * gae_trace
        val_est[step] = td_trace + values[step]
        adv_est[step] = gae_trace
    adv = adv_est @ _AGG
    for step in range(n - 1, -1, -1):
        if step == n - 1:
            delta = nll[step] - ents[step]
            reg_td_trace = delta
            reg_gae_trace = delta
        else:
            delta = nll[step] + ents[step + 1] - ents[step]
            reg_td_trace = delta + td_lambda * reg_td_trace
            reg_gae_trace = delta + gae_lambda * reg_gae_trace
        reg_val_est[step] = reg_td_trace + ents[step]
        adv[step] += reg_temp * reg_gae_trace
    return {"advantage": adv, "value_estimate": val_est, "entropy_target": reg_val_est / reg_norm}


def oracle_alpha(iteration_one_based: int) -> float:
    """`power_schedule(0.1, step, 0.3, 1.0, 0.001)` with `step = n - 1`."""
    step = int(iteration_one_based) - 1
    x = 0.1 / ((step + 1) ** 0.3)
    x = max(x, 0.001)
    x = min(x, 1.0)
    return x


# ---------------------------------------------------------------------------
# The loss (S16, S17, S18)
# ---------------------------------------------------------------------------


def oracle_loss_terms(
    piece_logits,
    wdl_logits,
    entropy_prediction,
    tokens,
    masks,
    behavior_log_probs,
    behavior_selected_log_prob,
    advantage,
    value_target,
    entropy_target,
    *,
    clip_epsilon: float,
    policy_weight: float,
    value_weight: float,
    entropy_weight: float,
    kl_weight: float,
) -> dict:
    """Every term of `L_setup` for a `[B, 40, ...]` batch, in float64."""
    logits = np.where(np.asarray(masks, dtype=bool), np.asarray(piece_logits, dtype=np.float64), MASKED_LOGIT)
    log_pi = logits - logits.max(axis=-1, keepdims=True)
    log_pi = log_pi - np.log(np.exp(log_pi).sum(axis=-1, keepdims=True))
    pi = np.exp(log_pi)
    tokens = np.asarray(tokens, dtype=np.int64)
    b, k = tokens.shape
    selected = log_pi[np.arange(b)[:, None], np.arange(k)[None, :], tokens]
    ratio = np.exp(selected - np.asarray(behavior_selected_log_prob, dtype=np.float64))
    adv = np.asarray(advantage, dtype=np.float64)
    clipped = np.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy = -np.minimum(ratio * adv, clipped * adv).mean()

    wdl = np.asarray(wdl_logits, dtype=np.float64)
    log_wdl = wdl - wdl.max(axis=-1, keepdims=True)
    log_wdl = log_wdl - np.log(np.exp(log_wdl).sum(axis=-1, keepdims=True))
    target = np.asarray(value_target, dtype=np.float64)[:, None, :]
    value = -(target * log_wdl).sum(-1).mean()

    entropy = ((np.asarray(entropy_prediction, dtype=np.float64) - np.asarray(entropy_target, dtype=np.float64)) ** 2).mean()

    masks_f = np.asarray(masks, dtype=bool)
    surprise = np.where(masks_f, log_pi - np.asarray(behavior_log_probs, dtype=np.float64), 0.0)
    kl = (np.where(masks_f, pi, 0.0) * surprise).sum(-1).mean()

    total = policy_weight * policy + value_weight * value + entropy_weight * entropy + kl_weight * kl
    return {
        "policy_loss": float(policy),
        "value_loss": float(value),
        "entropy_prediction_loss": float(entropy),
        "behavior_kl": float(kl),
        "total_loss": float(total),
        "ratio": ratio,
        "clip_fraction": float((np.abs(ratio - 1.0) > clip_epsilon).mean()),
    }


def oracle_forward(model, sequence) -> dict:
    """The production model's forward pass on a canned `[B, 41]` sequence, as
    float64 numpy arrays. The model is deep-copied to double precision."""
    double = copy.deepcopy(model).to(torch.float64).eval()
    with torch.no_grad():
        outputs = double(torch.as_tensor(np.asarray(sequence, dtype=np.int64)))
    return {name: value.detach().cpu().numpy().astype(np.float64) for name, value in outputs.items()}


def oracle_loss_from_model(model, fixture: dict, coefficients: dict) -> dict:
    """The oracle's total loss for a canned fixture through the model forward."""
    outputs = oracle_forward(model, fixture["sequence"])
    return oracle_loss_terms(
        outputs["piece_logits"],
        outputs["wdl_logits"],
        outputs["entropy_prediction"],
        fixture["tokens"],
        fixture["masks"],
        fixture["behavior_log_probs"],
        fixture["behavior_selected_log_prob"],
        fixture["advantage"],
        fixture["value_target"],
        fixture["entropy_target"],
        **coefficients,
    )


def oracle_finite_difference_gradient(model, fixture: dict, coefficients: dict, parameter_name: str, flat_index: int, epsilon: float = 1e-4) -> float:
    """Central finite difference of the oracle loss with respect to one
    parameter entry, on a float64 copy of the model."""
    double = copy.deepcopy(model).to(torch.float64).eval()
    parameter = dict(double.named_parameters())[parameter_name]
    flat = parameter.data.view(-1)
    original = float(flat[flat_index])
    sequence = torch.as_tensor(np.asarray(fixture["sequence"], dtype=np.int64))

    def loss_at(value: float) -> float:
        with torch.no_grad():
            flat[flat_index] = value
            outputs = double(sequence)
        arrays = {name: out.detach().cpu().numpy().astype(np.float64) for name, out in outputs.items()}
        return oracle_loss_terms(
            arrays["piece_logits"], arrays["wdl_logits"], arrays["entropy_prediction"],
            fixture["tokens"], fixture["masks"], fixture["behavior_log_probs"],
            fixture["behavior_selected_log_prob"], fixture["advantage"], fixture["value_target"],
            fixture["entropy_target"], **coefficients,
        )["total_loss"]

    plus = loss_at(original + epsilon)
    minus = loss_at(original - epsilon)
    flat[flat_index] = original
    return (plus - minus) / (2.0 * epsilon)


# ---------------------------------------------------------------------------
# Optimizer, clipping, EMA, step counts (S25-S28)
# ---------------------------------------------------------------------------


def oracle_adam_step(parameter, gradient, state: dict, *, lr: float, betas: tuple, eps: float) -> tuple:
    """One Adam step (weight decay 0) in float64: the PyTorch reference update
    `p -= lr * m_hat / (sqrt(v_hat) + eps)`."""
    beta1, beta2 = betas
    step = state.get("step", 0) + 1
    m = beta1 * state.get("m", np.zeros_like(parameter)) + (1.0 - beta1) * gradient
    v = beta2 * state.get("v", np.zeros_like(parameter)) + (1.0 - beta2) * gradient * gradient
    m_hat = m / (1.0 - beta1 ** step)
    v_hat = v / (1.0 - beta2 ** step)
    updated = parameter - lr * m_hat / (np.sqrt(v_hat) + eps)
    return updated, {"step": step, "m": m, "v": v}


def oracle_clip_scale(gradient_norms, max_norm: float) -> float:
    """`clip_grad_norm_` scales every gradient by `min(1, max_norm / total)`;
    PyTorch uses `max_norm / (total + 1e-6)` clamped at 1."""
    total = math.sqrt(sum(float(n) ** 2 for n in gradient_norms))
    return min(1.0, max_norm / (total + 1e-6))


def oracle_ema_closed_form(shadow0, raw, decay: float, updates: int):
    """After `updates` EMA updates against a FIXED raw parameter:
    `decay^k * shadow0 + (1 - decay^k) * raw`."""
    factor = decay ** int(updates)
    return factor * np.asarray(shadow0, dtype=np.float64) + (1.0 - factor) * np.asarray(raw, dtype=np.float64)


def oracle_step_counts(ready: int, batch_size: int, epochs: int) -> dict:
    """The published `arr_train` loop: `ceil(ready / batch)` minibatches per
    epoch, one optimizer step each, `epochs` epochs, ONE EMA update."""
    per_epoch = -(-int(ready) // int(batch_size))
    return {"minibatches_per_epoch": per_epoch, "optimizer_steps": per_epoch * int(epochs), "ema_updates": 1}


def canned_sequence(tokens) -> np.ndarray:
    """`[B, 41]` start token + tokens, for the model forward."""
    array = np.asarray(tokens, dtype=np.int64)
    sequence = np.full((array.shape[0], PIECES_PER_PLAYER + 1), START_TOKEN, dtype=np.int64)
    sequence[:, 1:] = array
    return sequence


__all__ = [
    "canned_sequence",
    "oracle_adam_step",
    "oracle_alpha",
    "oracle_clip_scale",
    "oracle_ema_closed_form",
    "oracle_expected_value",
    "oracle_finite_difference_gradient",
    "oracle_flat_advantage",
    "oracle_forward",
    "oracle_handedness_mask",
    "oracle_inventory_mask",
    "oracle_legal_masks",
    "oracle_loss_from_model",
    "oracle_loss_terms",
    "oracle_published_recursion",
    "oracle_running_mean",
    "oracle_softmax",
    "oracle_step_counts",
    "oracle_suffix_information",
]
