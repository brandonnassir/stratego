"""Checkpoint identity on reload, and every incompatibility failing loudly.

Covers Phase 5 gates 15 (`checkpoint_cpu_roundtrip_identity`) and 16
(`checkpoint_incompatibilities_fail_loudly`).
"""

from __future__ import annotations

import copy

import pytest
import torch

from stratego.model.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    REQUIRED_FIELDS,
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointFormatError,
    build_checkpoint_payload,
    load_checkpoint,
    read_checkpoint_payload,
    save_checkpoint,
    state_dict_digest,
    validate_checkpoint_payload,
)
from stratego.model.integration_model import (
    IntegrationModel,
    IntegrationModelConfig,
    build_integration_model,
)
from stratego.model.policy_adapter import greedy_action

from .conftest import deterministic_observation


def _payload(model=None) -> dict:
    return build_checkpoint_payload(model or build_integration_model(seed=1))


# ---------------------------------------------------------------------------
# Round-trip identity
# ---------------------------------------------------------------------------


def test_save_destroy_reload_reproduces_every_head_bit_for_bit(tmp_path, model):
    observation = deterministic_observation(seed=4, batch=3)
    before = model.forward_observation(observation).detached_cpu()

    path = save_checkpoint(model, tmp_path / "round_trip.pt", training_iteration=3, training_step=9)
    reloaded, metadata = load_checkpoint(path)
    after = reloaded.forward_observation(observation).detached_cpu()

    assert torch.equal(before.policy_logits, after.policy_logits)
    assert torch.equal(before.value_logits, after.value_logits)
    assert torch.equal(before.belief_logits, after.belief_logits)
    assert metadata["training_iteration"] == 3
    assert metadata["training_step"] == 9


def test_the_reloaded_model_selects_the_same_greedy_action(tmp_path, model):
    observation = deterministic_observation(seed=5)
    legal = list(range(0, 10_000, 37))

    path = save_checkpoint(model, tmp_path / "greedy.pt")
    reloaded, _ = load_checkpoint(path)

    before = greedy_action(model.forward_observation(observation).policy_logits[0], legal)
    after = greedy_action(reloaded.forward_observation(observation).policy_logits[0], legal)
    assert before == after


def test_the_weights_digest_survives_a_round_trip(tmp_path, model):
    path = save_checkpoint(model, tmp_path / "digest.pt")
    payload = read_checkpoint_payload(path)
    assert state_dict_digest(payload["state_dict"]) == state_dict_digest(model.state_dict())


def test_the_stored_metadata_names_every_frozen_contract(tmp_path, model):
    path = save_checkpoint(model, tmp_path / "metadata.pt")
    _, metadata = load_checkpoint(path)
    assert metadata["model_architecture_id"] == "integration_model_v1"
    assert metadata["model_contract_version"] == "model_contract_v1"
    assert metadata["rules_version"] == "stratego_project_v1"
    assert metadata["observation_version"] == "observation_v2_1_127ch"
    assert metadata["action_encoding_version"] == "source_destination_10000_v1"
    assert metadata["policy_action_frame"] == "absolute_engine_squares"
    assert metadata["checkpoint_file_digest"]
    assert metadata["creation_timestamp"]


def test_optional_state_is_carried_when_present_and_absent_otherwise(tmp_path, model):
    plain = save_checkpoint(model, tmp_path / "plain.pt")
    _, metadata = load_checkpoint(plain)
    assert metadata["has_optimizer_state"] is False
    assert metadata["has_ema_state"] is False

    rich = save_checkpoint(
        model,
        tmp_path / "rich.pt",
        optimizer_state={"step": 4},
        ema_state={"decay": 0.999},
        training_metrics={"loss": 1.25},
    )
    _, metadata = load_checkpoint(rich)
    assert metadata["has_optimizer_state"] is True
    assert metadata["has_ema_state"] is True
    assert metadata["has_training_metrics"] is True


def test_a_checkpoint_written_from_float16_reloads_as_float32(tmp_path, model):
    """Precision is a run-time choice; a file always stores float32 weights."""
    half = build_integration_model(seed=model.initialisation_seed, dtype=torch.float16)
    path = save_checkpoint(half, tmp_path / "half.pt")
    payload = read_checkpoint_payload(path)
    assert all(tensor.dtype == torch.float32 for tensor in payload["state_dict"].values())


# ---------------------------------------------------------------------------
# Incompatibility: every one of these must raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", [name for name in REQUIRED_FIELDS])
def test_missing_required_metadata_is_refused(field):
    payload = _payload()
    del payload[field]
    with pytest.raises(CheckpointError, match="missing"):
        validate_checkpoint_payload(payload)


def test_an_unknown_field_is_refused():
    payload = _payload()
    payload["mystery_field"] = 1
    with pytest.raises(CheckpointCompatibilityError, match="unknown field"):
        validate_checkpoint_payload(payload)


def test_a_newer_format_version_is_refused_rather_than_guessed():
    payload = _payload()
    payload["checkpoint_format_version"] = CHECKPOINT_FORMAT_VERSION + 1
    with pytest.raises(CheckpointCompatibilityError, match="newer code"):
        validate_checkpoint_payload(payload)


def test_a_nonsense_format_version_is_refused():
    for value in (0, -1, "1", 1.0):
        payload = _payload()
        payload["checkpoint_format_version"] = value
        with pytest.raises(CheckpointCompatibilityError):
            validate_checkpoint_payload(payload)


@pytest.mark.parametrize(
    "field, wrong",
    [
        ("rules_version", "stratego_project_v2"),
        ("observation_version", "observation_v2_127ch"),  # the superseded one
        ("observation_version", "observation_v3_200ch"),
        ("action_encoding_version", "source_destination_10000_v2"),
        ("model_contract_version", "model_contract_v2"),
        ("model_architecture_id", "ataraxos_full_v1"),
        ("policy_action_frame", "perspective_normalized_squares"),
    ],
)
def test_wrong_semantics_are_refused_even_though_the_weights_would_load(field, wrong):
    """The point of the whole module: shape-compatible is not semantics-compatible."""
    payload = _payload()
    payload[field] = wrong
    with pytest.raises(CheckpointCompatibilityError):
        validate_checkpoint_payload(payload)


def test_an_incompatible_configuration_is_refused():
    payload = _payload()
    payload["model_configuration"] = dict(payload["model_configuration"], width=128)
    with pytest.raises(CheckpointCompatibilityError, match="shape"):
        validate_checkpoint_payload(payload)


def test_an_unknown_configuration_field_is_refused():
    payload = _payload()
    payload["model_configuration"] = dict(payload["model_configuration"], mystery=3)
    with pytest.raises(CheckpointCompatibilityError):
        validate_checkpoint_payload(payload)


def test_a_configuration_that_contradicts_the_contract_is_refused():
    payload = _payload()
    payload["model_configuration"] = dict(payload["model_configuration"], belief_types=13)
    with pytest.raises(CheckpointCompatibilityError):
        validate_checkpoint_payload(payload)


def test_missing_weights_are_refused():
    payload = _payload()
    name = sorted(payload["state_dict"])[0]
    del payload["state_dict"][name]
    with pytest.raises(CheckpointCompatibilityError, match="missing weights"):
        validate_checkpoint_payload(payload)


def test_unexpected_weights_are_refused():
    payload = _payload()
    payload["state_dict"]["ghost_layer.weight"] = torch.zeros(3, 3)
    with pytest.raises(CheckpointCompatibilityError, match="unexpected weights"):
        validate_checkpoint_payload(payload)


def test_a_wrongly_shaped_weight_is_refused():
    payload = _payload()
    name = sorted(payload["state_dict"])[0]
    payload["state_dict"][name] = torch.zeros(5, 5)
    with pytest.raises(CheckpointCompatibilityError, match="shape"):
        validate_checkpoint_payload(payload)


def test_a_non_finite_weight_is_refused():
    payload = _payload()
    name = sorted(payload["state_dict"])[0]
    corrupted = payload["state_dict"][name].clone()
    corrupted.view(-1)[0] = float("nan")
    payload["state_dict"][name] = corrupted
    with pytest.raises(CheckpointFormatError, match="non-finite"):
        validate_checkpoint_payload(payload)


def test_an_empty_state_dict_is_refused():
    payload = _payload()
    payload["state_dict"] = {}
    with pytest.raises(CheckpointError):
        validate_checkpoint_payload(payload)


def test_a_negative_training_counter_is_refused():
    payload = _payload()
    payload["training_step"] = -1
    with pytest.raises(CheckpointCompatibilityError):
        validate_checkpoint_payload(payload)


# ---------------------------------------------------------------------------
# Corrupted files
# ---------------------------------------------------------------------------


def test_a_truncated_file_is_refused(tmp_path, model):
    path = save_checkpoint(model, tmp_path / "truncated.pt")
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(CheckpointFormatError, match="could not be read"):
        load_checkpoint(path)


def test_a_file_of_random_bytes_is_refused(tmp_path):
    path = tmp_path / "garbage.pt"
    path.write_bytes(b"this is definitely not a checkpoint" * 100)
    with pytest.raises(CheckpointFormatError):
        load_checkpoint(path)


def test_an_empty_file_is_refused(tmp_path):
    path = tmp_path / "empty.pt"
    path.write_bytes(b"")
    with pytest.raises(CheckpointFormatError, match="empty"):
        load_checkpoint(path)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(CheckpointFormatError, match="does not exist"):
        load_checkpoint(tmp_path / "nothing_here.pt")


def test_a_file_holding_a_bare_state_dict_is_refused(tmp_path, model):
    """The common mistake: `torch.save(model.state_dict(), path)`."""
    path = tmp_path / "bare.pt"
    torch.save(model.state_dict(), path)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


def test_a_tampered_file_still_validates_its_metadata(tmp_path, model):
    """Editing a stored version on disk is caught at load, not ignored."""
    path = save_checkpoint(model, tmp_path / "tampered.pt")
    payload = read_checkpoint_payload(path)
    payload["observation_version"] = "observation_v2_127ch"
    torch.save(payload, path)
    with pytest.raises(CheckpointCompatibilityError, match="observation_version"):
        load_checkpoint(path)


def test_a_valid_payload_passes_all_of_the_above():
    """The negative tests would be vacuous if the positive case did not pass."""
    metadata = validate_checkpoint_payload(_payload())
    assert metadata["model_architecture_id"] == "integration_model_v1"
    assert metadata["parameter_tensor_count"] > 0


def test_a_deep_copied_payload_is_unchanged_by_validation():
    payload = _payload()
    original = copy.deepcopy(payload["model_configuration"])
    validate_checkpoint_payload(payload)
    assert payload["model_configuration"] == original


def test_the_configuration_round_trips_through_its_dictionary_form():
    config = IntegrationModelConfig()
    assert IntegrationModelConfig.from_dict(config.to_dict()) == config
    assert IntegrationModel(config).config == config
