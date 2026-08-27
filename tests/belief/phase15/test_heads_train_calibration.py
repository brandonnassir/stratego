"""Phase 15 Agent 1 sections 8-10: the specialists, the recipe, the temperature.

Section 8 asks for four proofs, and three of them are about what *cannot*
happen: the source policy must be bit-identical before and after training,
no policy or value parameter may take a gradient, and a belief checkpoint
must refuse a backbone that is not the one it was trained on. These tests
are those proofs, plus the recipe binding and the calibration invariants.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stratego.belief.phase15 import calibration as CAL
from stratego.belief.phase15 import checkpoint as CK
from stratego.belief.phase15 import contract as C
from stratego.belief.phase15 import heads as H
from stratego.belief.phase15 import train as T
from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
from stratego.model.checkpoint import load_checkpoint
from stratego.training.phase9_behavior import state_dict_digest

SOURCE = "checkpoints/phase15/p18_source_readonly.pt"


def _backbone():
    from pathlib import Path

    if not Path(SOURCE).is_file():
        pytest.skip("the P18 read-only copy has not been created yet")
    model, _metadata = load_checkpoint(
        SOURCE,
        device=torch.device("cpu"),
        dtype=torch.float32,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@pytest.fixture(scope="module")
def backbone():
    return _backbone()


@pytest.fixture()
def specialist(backbone):
    return H.Phase15BeliefSpecialist.from_policy(backbone, specialist_id="b18")


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


def test_the_mlp_is_the_declared_shape():
    layers = [
        module for module in H.BeliefMLP().body if isinstance(module, torch.nn.Linear)
    ]
    assert [layer.in_features for layer in layers] == [128, 512, 512]
    assert [layer.out_features for layer in layers] == [512, 512, 12]
    assert C.MLP_ACTIVATION == "gelu"
    assert any(isinstance(module, torch.nn.GELU) for module in H.BeliefMLP().body)


def test_a_specialist_holds_no_policy_or_value_parameter(specialist):
    names = list(specialist.state_dict())
    assert names
    for name in names:
        assert name.startswith(("block.", "encoder_norm.", "head.", "log_temperature"))
        assert "policy" not in name
        assert "value" not in name
        assert "belief_output" not in name


def test_the_specialist_copies_rather_than_shares_the_source_block(
    backbone, specialist
):
    source = list(backbone.blocks)[-1]
    for (name, copied), (_name, original) in zip(
        specialist.block.named_parameters(), source.named_parameters()
    ):
        assert copied is not original
        assert torch.equal(copied.detach(), original.detach()), name
    with torch.no_grad():
        next(specialist.block.parameters()).add_(1.0)
    assert not torch.equal(
        next(specialist.block.parameters()).detach(),
        next(source.parameters()).detach(),
    )


def test_the_frozen_prefix_is_every_block_but_the_last(backbone):
    assert H.TRAINABLE_BLOCKS == 1
    assert len(list(backbone.blocks)) == 4


def test_temperature_starts_at_one_and_must_stay_positive(specialist):
    assert specialist.temperature == pytest.approx(1.0)
    specialist.set_temperature(2.5)
    assert specialist.temperature == pytest.approx(2.5)
    with pytest.raises(H.Phase15HeadError):
        specialist.set_temperature(0.0)
    with pytest.raises(H.Phase15HeadError):
        specialist.set_temperature(-1.0)


def test_calibrated_logits_never_reorder_a_row(specialist):
    specialist.set_temperature(3.25)
    logits = torch.randn(64, 12)
    scaled = specialist.calibrated_logits(logits)
    assert torch.equal(logits.argmax(dim=1), scaled.argmax(dim=1))


# ---------------------------------------------------------------------------
# Gradient isolation
# ---------------------------------------------------------------------------


def test_the_optimizer_groups_hold_only_specialist_tensors(specialist):
    groups = H.trainable_parameter_groups(
        specialist, head_lr=1e-3, block_lr=1e-4
    )
    owned = {id(tensor) for tensor in specialist.parameters()}
    for group in groups:
        for tensor in group["params"]:
            assert id(tensor) in owned
    assert [group["lr"] for group in groups] == [1e-4, 1e-3]


def test_a_backward_pass_leaves_the_source_untouched(backbone, specialist):
    before = state_dict_digest(backbone)
    tokens = torch.randn(4, 100, 128)
    gather = (torch.zeros(4, dtype=torch.long), torch.arange(4))
    loss = torch.nn.functional.cross_entropy(
        specialist(tokens, gather), torch.zeros(4, dtype=torch.long)
    )
    loss.backward()
    assert T.assert_no_source_gradients(backbone) == {
        "policy_value_parameters_with_gradient": 0,
        "policy_value_parameters_requiring_grad": 0,
        "checked_parameters": len(list(backbone.parameters())),
    }
    assert state_dict_digest(backbone) == before


def test_the_gradient_check_fires_when_a_source_parameter_is_unfrozen(backbone):
    parameter = next(backbone.parameters())
    parameter.requires_grad_(True)
    try:
        with pytest.raises(T.Phase15TrainError):
            T.assert_no_source_gradients(backbone)
    finally:
        parameter.requires_grad_(False)


# ---------------------------------------------------------------------------
# The recipe
# ---------------------------------------------------------------------------


def test_the_train_config_defaults_are_the_declared_recipe():
    config = T.TrainConfig(specialist_id="b18")
    assert config.head_learning_rate == C.RECIPE["head_learning_rate"]
    assert config.block_learning_rate == C.RECIPE["final_block_learning_rate"]
    assert config.weight_decay == C.RECIPE["weight_decay"]
    assert config.batch_size == C.RECIPE["batch_size"] == 256
    assert config.epochs == C.RECIPE["max_epochs"] == 12
    assert config.patience == C.RECIPE["early_stop_patience"] == 3
    assert config.optimizer == "adamw"
    assert config.schedule == "cosine"
    assert config.batch_size_changed_from is None


def test_both_specialists_share_one_recipe():
    left = T.TrainConfig(specialist_id="b18").to_dict()
    right = T.TrainConfig(specialist_id="b24").to_dict()
    left.pop("specialist_id")
    right.pop("specialist_id")
    assert left == right


def test_an_unknown_optimizer_is_refused(specialist):
    config = T.TrainConfig(specialist_id="b18", optimizer="sgd")
    with pytest.raises(T.Phase15TrainError):
        T._optimizer(specialist, config)


def test_sample_batches_index_exactly_the_pieces_of_their_positions():
    data = {
        "samples": 4,
        "piece_offset": np.array([0, 2, 5, 6, 9], dtype=np.int64),
        "perspective_square": np.arange(9, dtype=np.int64),
        "true_rank": np.arange(9, dtype=np.int64) % 12,
    }
    rows = np.array([0, 2, 3], dtype=np.int64)
    batches = list(T.sample_batches(data, rows, 3))
    block, token_rows, squares, labels = batches[0]
    assert list(block) == [0, 2, 3]
    # Two pieces from row 0, one from row 2, three from row 3.
    assert list(token_rows) == [0, 0, 1, 2, 2, 2]
    assert list(squares) == [0, 1, 5, 6, 7, 8]
    assert list(labels) == [0, 1, 5, 6, 7, 8]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _overconfident(rows: int = 4000, scale: float = 3.0):
    rng = np.random.default_rng(11)
    true_rank = rng.integers(0, 12, size=rows)
    base = rng.normal(size=(rows, 12))
    base[np.arange(rows), true_rank] += 1.5
    return base * scale, true_rank


def test_a_fitted_temperature_lowers_the_calibration_split_nll():
    logits, true_rank = _overconfident()
    fit = CAL.fit_temperature(logits, true_rank)
    assert fit["temperature"] > 1.0
    assert fit["calibration_nll_fitted"] < fit["calibration_nll_raw"]
    assert fit["top1_labels_changed"] == 0


def test_temperature_scaling_never_moves_a_top_one_label():
    logits, _true = _overconfident()
    raw = logits.argmax(axis=1)
    for temperature in (0.1, 0.5, 1.0, 2.0, 17.0):
        scaled = CAL.scaled_probabilities(logits, temperature).argmax(axis=1)
        assert np.array_equal(raw, scaled)


def test_scaled_probabilities_are_a_simplex():
    logits, _true = _overconfident(rows=256)
    probabilities = CAL.scaled_probabilities(logits, 2.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities > 0).all()


def test_a_non_positive_temperature_is_refused():
    logits, true_rank = _overconfident(rows=64)
    with pytest.raises(CAL.Phase15CalibrationError):
        CAL.scaled_probabilities(logits, 0.0)
    with pytest.raises(CAL.Phase15CalibrationError):
        CAL.negative_log_likelihood(logits, true_rank, -1.0)


def test_an_already_calibrated_model_gets_a_temperature_near_one():
    # Labels drawn *from* the model's own softmax, so the logits are exactly
    # calibrated by construction and the optimum is T = 1.
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(20000, 12)) * 1.5
    probabilities = CAL.scaled_probabilities(logits, 1.0)
    true_rank = np.array(
        [rng.choice(12, p=row) for row in probabilities], dtype=np.int64
    )
    fit = CAL.fit_temperature(logits, true_rank)
    assert 0.9 < fit["temperature"] < 1.1


def test_an_underconfident_model_gets_a_temperature_below_one():
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(20000, 12)) * 1.5
    probabilities = CAL.scaled_probabilities(logits, 1.0)
    true_rank = np.array(
        [rng.choice(12, p=row) for row in probabilities], dtype=np.int64
    )
    # Flatten the logits: the fitter must sharpen them back.
    fit = CAL.fit_temperature(logits * 0.4, true_rank)
    assert fit["temperature"] < 0.6
    assert fit["calibration_nll_fitted"] < fit["calibration_nll_raw"]


def test_the_keep_rule_needs_both_improvements():
    raw = {"nll": 2.0, "expected_calibration_error": 0.10,
           "maximum_calibration_error": 0.2, "top1": 0.3}
    better = {"nll": 1.9, "expected_calibration_error": 0.05,
              "maximum_calibration_error": 0.1, "top1": 0.3}
    assert CAL.decide(raw, better)["keep_calibrated"] is True
    worse_nll = {**better, "nll": 2.1}
    assert CAL.decide(raw, worse_nll)["keep_calibrated"] is False
    worse_ece = {**better, "expected_calibration_error": 0.2}
    assert CAL.decide(raw, worse_ece)["keep_calibrated"] is False
    moved_top1 = {**better, "top1": 0.31}
    assert CAL.decide(raw, moved_top1)["keep_calibrated"] is False


def test_the_fitter_refuses_mismatched_shapes():
    with pytest.raises(CAL.Phase15CalibrationError):
        CAL.fit_temperature(np.zeros((4, 12)), np.zeros(3, dtype=np.int64))
    with pytest.raises(CAL.Phase15CalibrationError):
        CAL.fit_temperature(np.zeros((0, 12)), np.zeros(0, dtype=np.int64))


# ---------------------------------------------------------------------------
# The checkpoint
# ---------------------------------------------------------------------------


def _identity(source_id="p18"):
    return {
        "source_id": source_id,
        "logical_identity": "P18" if source_id == "p18" else "P24",
        "hour": 18 if source_id == "p18" else 24,
        "model_state_digest": "a" * 64,
        "phase15_copy_sha256": "b" * 64,
        "phase15_copy_path": SOURCE,
        "original_snapshot_sha256": "c" * 64,
        "global_optimizer_step": 92718,
    }


def _corpus():
    return {
        "corpus_version": C.CORPUS_VERSION,
        "corpus_digest": "d" * 64,
        "corpus_format_version": "phase15_belief_corpus_store_v1",
    }


def test_a_checkpoint_holds_only_belief_tensors(tmp_path, specialist):
    path = tmp_path / "b18.pt"
    CK.save_specialist(
        specialist,
        path,
        source_identity=_identity(),
        corpus_identity=_corpus(),
        training_record={"best_epoch": 3},
        calibration_record={"temperature": 1.0},
    )
    payload = CK.read_payload(path)
    assert payload["holds_policy_parameters"] is False
    assert payload["holds_value_parameters"] is False
    for name in payload["state_dict"]:
        assert name.startswith(CK.ALLOWED_PREFIXES)


def test_saving_over_an_existing_path_is_refused(tmp_path, specialist):
    path = tmp_path / "b18.pt"
    arguments = {
        "source_identity": _identity(),
        "corpus_identity": _corpus(),
        "training_record": {},
        "calibration_record": {},
    }
    CK.save_specialist(specialist, path, **arguments)
    with pytest.raises(CK.Phase15CheckpointError):
        CK.save_specialist(specialist, path, **arguments)
    CK.save_specialist(specialist, path, overwrite=True, **arguments)


def test_a_specialist_cannot_be_bound_to_the_wrong_source(tmp_path, specialist):
    with pytest.raises(CK.Phase15CheckpointError):
        CK.save_specialist(
            specialist,
            tmp_path / "b18.pt",
            source_identity=_identity("p24"),
            corpus_identity=_corpus(),
            training_record={},
            calibration_record={},
        )


def test_a_checkpoint_refuses_a_backbone_it_was_not_trained_on(
    tmp_path, backbone, specialist
):
    path = tmp_path / "b18.pt"
    CK.save_specialist(
        specialist,
        path,
        source_identity=_identity(),
        corpus_identity=_corpus(),
        training_record={},
        calibration_record={},
    )
    # `_identity` records a fabricated digest, so the real backbone is the
    # wrong one — which is exactly the refusal section 8 asks for.
    with pytest.raises(CK.Phase15CheckpointError):
        CK.load_specialist(path, backbone)


def test_a_checkpoint_loads_with_its_recorded_source(tmp_path, backbone, specialist):
    identity = {**_identity(), "model_state_digest": state_dict_digest(backbone)}
    path = tmp_path / "b18.pt"
    specialist.set_temperature(1.75)
    CK.save_specialist(
        specialist,
        path,
        source_identity=identity,
        corpus_identity=_corpus(),
        training_record={},
        calibration_record={},
    )
    loaded, payload = CK.load_specialist(path, backbone)
    assert loaded.temperature == pytest.approx(1.75)
    assert payload["specialist_id"] == "b18"
    assert not any(tensor.requires_grad for tensor in loaded.parameters())
    assert CK.state_digest(loaded.state_dict()) == payload["state_digest"]


def test_a_forbidden_tensor_name_is_refused():
    with pytest.raises(CK.Phase15CheckpointError):
        CK.check_contents({"policy_head.weight": torch.zeros(2)})
    with pytest.raises(CK.Phase15CheckpointError):
        CK.check_contents({"belief_output.weight": torch.zeros(2)})
