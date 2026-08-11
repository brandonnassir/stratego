"""One controlled backward pass: are the encoder and all three heads connected?

Covers Phase 5 gate 18 (`autograd_all_heads_connected_finite`).

.. warning::

   This is a *connectivity smoke test*, not a training experiment. Phase 5
   authorises exactly one backward pass and forbids optimizer tuning, multi-step
   learning and hyperparameter search. Nothing here steps an optimizer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stratego.engine.observation import build_observation
from stratego.engine.legal_moves import legal_action_mask, legal_actions
from stratego.model.contract import ModelContractError
from stratego.model.integration_model import build_integration_model
from stratego.model.losses import (
    DEFAULT_BELIEF_WEIGHT,
    DEFAULT_VALUE_WEIGHT,
    belief_loss,
    multi_head_loss,
    policy_loss,
    value_loss,
)
from stratego.model.tokenization import tokenize_numpy_observation
from stratego.training.belief_targets import dense_belief_target

from ..helpers import nonterminal_state


def _training_batch(batch: int = 4):
    """A small real batch: engine observations, engine legality, engine targets.

    The value targets are arbitrary class labels -- there is no learning here, so
    a fixed rotation through WIN/DRAW/LOSS is enough to exercise the head.
    """
    states = [nonterminal_state(ply) for ply in (20, 40, 60, 80)[:batch]]
    observations = [build_observation(state, state.acting_player) for state in states]
    tokens = tokenize_numpy_observation(observations)

    masks, targets = [], []
    for state in states:
        actions = legal_actions(state)
        masks.append(legal_action_mask(state, actions).astype(bool))
        targets.append(actions[len(actions) // 2])

    labels, belief_mask = [], []
    for state in states:
        pair = dense_belief_target(state, state.acting_player)
        labels.append(pair[0])
        belief_mask.append(pair[1])

    return {
        "tokens": tokens,
        "legal_mask": torch.from_numpy(np.stack(masks)),
        "target_actions": torch.tensor(targets, dtype=torch.int64),
        "target_values": torch.tensor([index % 3 for index in range(len(states))]),
        "belief_labels": torch.from_numpy(np.stack(labels)),
        "belief_mask": torch.from_numpy(np.stack(belief_mask)),
    }


def test_one_backward_pass_is_finite_and_reaches_every_parameter():
    model = build_integration_model(seed=99)
    model.train()  # no dropout exists, but state the intent
    batch = _training_batch()

    outputs = model(batch["tokens"])
    loss = multi_head_loss(
        outputs,
        target_actions=batch["target_actions"],
        legal_mask=batch["legal_mask"],
        target_value_classes=batch["target_values"],
        belief_labels=batch["belief_labels"],
        belief_mask=batch["belief_mask"],
    )

    assert loss.all_finite()
    assert float(loss.total.detach()) > 0.0
    loss.total.backward()

    missing, non_finite = [], []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(parameter.grad).all()):
            non_finite.append(name)
    assert missing == []
    assert non_finite == []


def test_each_head_sends_gradient_into_its_own_parameters_and_the_shared_encoder():
    """Three separate backward passes, one per head, so attribution is unambiguous."""
    batch = _training_batch()
    shared_prefixes = ("input_projection", "position_embedding", "blocks", "encoder_norm")

    head_parameters = {
        "policy": ("policy_source", "policy_destination"),
        "value": ("value_body", "value_head"),
        "belief": ("belief_head",),
    }

    for head, owned in head_parameters.items():
        model = build_integration_model(seed=99)
        model.zero_grad(set_to_none=True)
        outputs = model(batch["tokens"])
        if head == "policy":
            loss = policy_loss(
                outputs.policy_logits, batch["target_actions"], batch["legal_mask"]
            )
        elif head == "value":
            loss = value_loss(outputs.value_logits, batch["target_values"])
        else:
            loss = belief_loss(
                outputs.belief_logits, batch["belief_labels"], batch["belief_mask"]
            )
        loss.backward()

        touched = {
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and bool((parameter.grad != 0).any())
        }
        # This head's own parameters moved ...
        assert any(name.startswith(prefix) for prefix in owned for name in touched), head
        # ... and so did the shared encoder, which is the connectivity claim.
        assert any(
            name.startswith(prefix) for prefix in shared_prefixes for name in touched
        ), head
        # ... while the other heads stayed exactly at zero.
        for other, other_prefixes in head_parameters.items():
            if other == head:
                continue
            for name in touched:
                assert not name.startswith(other_prefixes), f"{head} leaked into {other}"


def test_the_components_combine_with_the_declared_weights():
    model = build_integration_model(seed=99)
    batch = _training_batch()
    outputs = model(batch["tokens"])
    loss = multi_head_loss(
        outputs,
        target_actions=batch["target_actions"],
        legal_mask=batch["legal_mask"],
        target_value_classes=batch["target_values"],
        belief_labels=batch["belief_labels"],
        belief_mask=batch["belief_mask"],
        value_weight=0.5,
        belief_weight=0.25,
    )
    manual = loss.policy + 0.5 * loss.value + 0.25 * loss.belief
    assert torch.allclose(loss.total, manual, atol=1e-6)
    assert loss.to_dict()["value_weight"] == 0.5
    assert loss.to_dict()["belief_weight"] == 0.25


def test_the_default_weights_are_the_documented_placeholders():
    assert DEFAULT_VALUE_WEIGHT == 1.0
    assert DEFAULT_BELIEF_WEIGHT == 1.0


def test_the_masked_policy_loss_never_produces_a_non_finite_gradient():
    """A row where almost every action is illegal is the numerically hard case."""
    logits = torch.randn(2, 10_000, generator=torch.Generator().manual_seed(4))
    logits.requires_grad_(True)
    mask = torch.zeros(2, 10_000, dtype=torch.bool)
    mask[0, 42] = True  # exactly one legal action
    mask[1, [1, 2, 3]] = True
    targets = torch.tensor([42, 2])

    loss = policy_loss(logits, targets, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert bool(torch.isfinite(logits.grad).all())
    # No gradient escaped onto an illegal action.
    assert bool((logits.grad[~mask] == 0).all())


def test_an_illegal_target_action_is_refused():
    logits = torch.zeros(1, 10_000, requires_grad=True)
    mask = torch.zeros(1, 10_000, dtype=torch.bool)
    mask[0, 5] = True
    with pytest.raises(ModelContractError, match="illegal"):
        policy_loss(logits, torch.tensor([6]), mask)


def test_a_legality_mask_with_an_empty_row_is_refused():
    logits = torch.zeros(2, 10_000)
    mask = torch.zeros(2, 10_000, dtype=torch.bool)
    mask[0, 1] = True  # row 1 has nothing legal at all
    with pytest.raises(ModelContractError, match="no legal action"):
        policy_loss(logits, torch.tensor([1, 1]), mask)


def test_inference_does_not_build_a_graph(model):
    """Ordinary evaluation must not accumulate autograd state."""
    from .conftest import deterministic_observation

    with torch.no_grad():
        outputs = model.forward_observation(deterministic_observation(seed=6))
    assert outputs.policy_logits.grad_fn is None
    assert outputs.value_logits.grad_fn is None
    assert outputs.belief_logits.grad_fn is None
