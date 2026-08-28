"""Phase 17 Agent 2: the exact Phase 9 start (gate G-C check C2)."""

from __future__ import annotations

import shutil

import pytest
import torch

from stratego.training.phase17.move_contract import (
    MOVE_EMA_DECAY,
    MOVE_INITIAL_KL_BETA,
    START_FILE_SHA256,
    START_MODEL_STATE_DIGEST,
    START_PARAMETER_COUNT,
    MoveScheduleHorizon,
)
from stratego.training.phase17.move_start import (
    DISCARDED_PAYLOAD_KEYS,
    Phase17MoveStartError,
    belief_head_parameters,
    build_move_start,
    load_phase17_move_weights,
    start_semantics,
)
from stratego.training.phase9_behavior import state_dict_digest
from stratego.training.phase9_checkpoint import (
    bind_behavior_snapshot,
    read_phase9_payload,
)

START_PATH = "checkpoints/phase9/selfplay_c1_v1.pt"


@pytest.fixture(scope="module")
def loaded():
    return load_phase17_move_weights(device="cpu")


def test_both_start_digests_reproduce_from_the_bytes(loaded):
    assert loaded["file_sha256"] == START_FILE_SHA256
    assert loaded["model_state_digest"] == START_MODEL_STATE_DIGEST
    assert loaded["parameter_count"] == START_PARAMETER_COUNT
    assert loaded["candidate_id"] == "C1"
    assert loaded["behavior_snapshot_identity"] == "B041"


def test_the_digest_is_recomputed_from_the_live_module(loaded):
    assert state_dict_digest(loaded["model"]) == START_MODEL_STATE_DIGEST
    # The other function with the same name disagrees, and the loader says so
    # rather than leaving a future reader to mistake it for corruption.
    assert loaded["container_state_digest"] != loaded["model_state_digest"]


def test_logits_are_identical_to_the_accepted_phase9_behavior_loader(loaded):
    """The start is the same weights the accepted collection path would bind.

    Two independent accepted entry points -- `bind_behavior_snapshot`, which is
    how Phase 9 seats B041 for collection, and the Phase 17 loader -- must put
    the same numbers on the same fixed observations. Digest equality alone
    would not prove the *model* was rebuilt identically.
    """
    snapshot = bind_behavior_snapshot(
        START_PATH, logical_identity="B041", namespace="canonical", device="cpu"
    )
    generator = torch.Generator().manual_seed(20260827)
    observations = torch.rand((4, 127, 10, 10), generator=generator)
    with torch.no_grad():
        accepted = snapshot.model.forward_observation(observations)
        ours = loaded["model"].forward_observation(observations)
    assert torch.equal(accepted.policy_logits, ours.policy_logits)
    assert torch.equal(accepted.value_logits, ours.value_logits)
    assert torch.equal(accepted.belief_logits, ours.belief_logits)


def test_the_model_comes_back_trainable(loaded):
    assert all(parameter.requires_grad for parameter in loaded["model"].parameters())


def test_a_checkpoint_with_other_bytes_is_refused(tmp_path):
    copy = tmp_path / "not_the_start.pt"
    shutil.copyfile(START_PATH, copy)
    with copy.open("ab") as handle:
        handle.write(b"\x00")
    with pytest.raises(Phase17MoveStartError, match="not the accepted"):
        load_phase17_move_weights(copy, device="cpu")


def test_a_missing_checkpoint_is_refused(tmp_path):
    with pytest.raises(Phase17MoveStartError, match="missing"):
        load_phase17_move_weights(tmp_path / "absent.pt", device="cpu")


def test_a_wrong_expected_model_state_digest_is_refused():
    with pytest.raises(Phase17MoveStartError, match="model-state digest"):
        load_phase17_move_weights(device="cpu", expected_model_state_digest="0" * 64)


def test_an_unsupported_device_is_refused():
    with pytest.raises(Phase17MoveStartError, match="unsupported device"):
        load_phase17_move_weights(device="cuda")


def test_the_start_is_a_new_lineage_not_a_resume():
    """Every optimizer, schedule and controller value is Phase 17's own."""
    payload = read_phase9_payload(START_PATH)
    start = build_move_start(total_iterations=200, device="cpu")

    # The file carries a beta already at its 0.2 ceiling and iteration 41.
    assert float(payload["kl_beta"]) == pytest.approx(0.2)
    assert int(payload["rl_iteration"]) == 41
    assert int(payload["global_optimizer_step"]) > 0

    assert float(start.controller.beta) == pytest.approx(MOVE_INITIAL_KL_BETA)
    assert start.controller.history == []
    assert start.iteration == 0
    assert start.next_iteration == 1
    assert start.optimizer.state == {}
    assert start.optimizer.param_groups[0]["lr"] == pytest.approx(
        start.horizon.learning_rate(1)
    )
    for key in DISCARDED_PAYLOAD_KEYS:
        assert key in start.to_dict()["discarded_from_phase9_payload"]


def test_the_ema_starts_equal_to_the_raw_weights():
    start = build_move_start(total_iterations=200, device="cpu")
    assert start.ema.updates == 0
    assert start.ema.decay == pytest.approx(MOVE_EMA_DECAY)
    raw = start.model.state_dict()
    stored = start.ema.state_dict()
    assert set(raw) == set(stored)
    for name, tensor in raw.items():
        assert torch.equal(tensor.detach().cpu(), stored[name])


def test_the_belief_head_is_present_and_carries_no_weight():
    start = build_move_start(total_iterations=200, device="cpu")
    names = belief_head_parameters(start.model)
    assert names == ["belief_output.weight", "belief_output.bias"]
    assert start.to_dict()["belief_loss_weight"] == 0.0


def test_a_horizon_that_disagrees_with_the_request_is_refused():
    with pytest.raises(Phase17MoveStartError, match="horizon covers"):
        build_move_start(
            total_iterations=200,
            device="cpu",
            horizon=MoveScheduleHorizon(total_iterations=199),
        )


def test_start_semantics_names_the_loader_and_the_digest_function():
    semantics = start_semantics()
    assert "read_phase9_payload" in semantics["loader"]
    assert semantics["digest_function"].endswith("phase9_behavior.state_dict_digest")
    assert semantics["belief_loss_weight"] == 0.0
    assert "not used" in semantics["resume_identity_check"]
