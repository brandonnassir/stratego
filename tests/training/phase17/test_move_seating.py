"""Phase 17 Agent 2: the live current-policy cell and its refusals (check C4)."""

from __future__ import annotations

import pytest
import torch

from stratego.training.phase17.move_contract import (
    CURRENT_POLICY_IDENTITY,
    CURRENT_POLICY_TOKEN,
    START_MODEL_STATE_DIGEST,
)
from stratego.training.phase17.move_snapshot import (
    CurrentMovePolicy,
    Phase17Seating,
    Phase17SeatingError,
    RefusingRulePolicies,
    freeze_model,
    seating_semantics,
    snapshot_from_model,
)
from stratego.training.phase17.move_start import load_phase17_move_weights
from stratego.training.phase9_behavior import BehaviorSnapshot, state_dict_digest

from .test_move_support import perturbed_copy


@pytest.fixture(scope="module")
def model():
    return load_phase17_move_weights(device="cpu")["model"]


def test_a_snapshot_is_a_frozen_copy_not_an_alias(model):
    snapshot = snapshot_from_model(model, device="cpu")
    assert snapshot.checkpoint_sha256 == START_MODEL_STATE_DIGEST
    assert snapshot.model is not model
    assert not any(p.requires_grad for p in snapshot.model.parameters())
    assert not snapshot.model.training
    snapshot.assert_frozen()

    # Mutating the live model must not move the snapshot: the PPO denominator
    # would otherwise stop meaning what it says.
    with torch.no_grad():
        next(iter(model.parameters())).add_(1.0)
    try:
        assert state_dict_digest(snapshot.model) == START_MODEL_STATE_DIGEST
        snapshot.assert_frozen()
    finally:
        with torch.no_grad():
            next(iter(model.parameters())).sub_(1.0)


def test_the_seating_resolves_through_the_cell_every_time(model):
    """The Phase 16 defect, checked directly: no reader caches the snapshot."""
    first = snapshot_from_model(model, device="cpu")
    cell = CurrentMovePolicy(first, iteration=1)
    seating = Phase17Seating(cell)
    assert seating.behavior is first

    second = snapshot_from_model(perturbed_copy(model), device="cpu")
    report = cell.rebind(second, iteration=2)
    assert report["changed"] is True
    assert report["model_state_digest_before"] == first.checkpoint_sha256
    assert seating.behavior is second
    assert cell.iteration == 2
    assert cell.rebinds == 1
    assert cell.known_digests() == (first.checkpoint_sha256, second.checkpoint_sha256)


def test_the_token_is_stable_while_the_digest_moves(model):
    cell = CurrentMovePolicy(snapshot_from_model(model, device="cpu"), iteration=1)
    before = cell.digest
    cell.rebind_from_model(perturbed_copy(model), iteration=2)
    assert cell.snapshot.policy_token == CURRENT_POLICY_TOKEN
    assert cell.snapshot.logical_identity == CURRENT_POLICY_IDENTITY
    assert cell.digest != before


def test_a_backwards_rebind_is_refused(model):
    cell = CurrentMovePolicy(snapshot_from_model(model, device="cpu"), iteration=5)
    with pytest.raises(Phase17SeatingError, match="rebind backwards"):
        cell.rebind_from_model(perturbed_copy(model), iteration=4)


def test_a_snapshot_with_another_token_is_refused(model):
    frozen = freeze_model(model, device="cpu")
    digest = state_dict_digest(frozen)
    foreign = BehaviorSnapshot(
        logical_identity="B041",
        policy_token="phase9_behavior_v1|ns=canonical|B041",
        checkpoint_path="<test>",
        checkpoint_sha256=digest,
        device="cpu",
        model=frozen,
        loaded_state_dict_digest=digest,
    )
    with pytest.raises(Phase17SeatingError, match="current-policy token"):
        CurrentMovePolicy(foreign, iteration=1)


def test_a_trainable_snapshot_is_refused(model):
    frozen = freeze_model(model, device="cpu")
    for parameter in frozen.parameters():
        parameter.requires_grad_(True)
    digest = state_dict_digest(frozen)
    thawed = BehaviorSnapshot(
        logical_identity=CURRENT_POLICY_IDENTITY,
        policy_token=CURRENT_POLICY_TOKEN,
        checkpoint_path="<test>",
        checkpoint_sha256=digest,
        device="cpu",
        model=frozen,
        loaded_state_dict_digest=digest,
    )
    with pytest.raises(Phase17SeatingError, match="trainable parameters"):
        CurrentMovePolicy(thawed, iteration=1)


def test_the_rule_and_historical_seats_refuse_rather_than_return(model):
    cell = CurrentMovePolicy(snapshot_from_model(model, device="cpu"), iteration=1)
    seating = Phase17Seating(cell)
    assert isinstance(seating.rules, RefusingRulePolicies)
    assert seating.historical == {}
    with pytest.raises(Phase17SeatingError, match="100% current-policy"):
        seating.rules.get("strategic_rule_based")
    with pytest.raises(Phase17SeatingError, match="evaluation instrument"):
        seating.historical_snapshot("H035")


def test_seating_needs_a_cell():
    with pytest.raises(Phase17SeatingError, match="CurrentMovePolicy"):
        Phase17Seating(object())


def test_seating_semantics_names_the_defect_it_fixes():
    semantics = seating_semantics()
    assert "stale" in semantics["resolution"]
    assert "WindowCollector.rebind" in semantics["phase16_defect_fixed"]
    assert semantics["weights"].startswith("RAW")
