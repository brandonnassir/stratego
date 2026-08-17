"""Phase 9 Agent 5: `phase9_checkpoint_v1`, its rejections, and the archive.

Each rejection the mission lists gets a test that produces exactly that
corruption and no other, so a passing suite means the checkpoint refuses for
the stated reason rather than by accident.
"""

from __future__ import annotations

import json

import pytest
import torch

from stratego.model.production_model import build_candidate_model
from stratego.training import phase9_checkpoint as pck
from stratego.training.phase9_contract import (
    CHECKPOINT_REQUIRED_FIELDS,
    HISTORICAL_ANCHOR_ID,
    PHASE9_CHECKPOINT_VERSION,
    PHASE9_POPULATION_VERSION,
)
from stratego.training.phase9_targets import PHASE9_EXAMPLE_VERSION
from stratego.training.warmstart_checkpoint import CorpusIdentity
from stratego.training.warmstart_seed import CANONICAL_C1_INIT_SEED

ACCEPTED_CORPUS = CorpusIdentity(
    corpus_version="synthetic_warmstart_corpus_v1",
    content_digest="c95c3545b07f2341e7efbc83c79e6342510dd973038b0f72e7eae013cff87d0d",
    metadata_digest="1db0f02fe45b16f539f070b1e12d4fdd6f390fd0487180fe660af0f4d49c81bb",
    commit_index_digest="32e8e18d1ca57ee555ed848851284f5938d4989ceb6c864f83ca4b9286c15db1",
)

ROLLOUT_DIGEST = "a" * 64
BEHAVIOR_SHA = "b" * 64


@pytest.fixture(scope="module")
def model():
    return build_candidate_model("C1", seed=CANONICAL_C1_INIT_SEED, device="cpu")


@pytest.fixture(scope="module")
def parts(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # One step so the optimizer state is non-empty, which is what a real
    # checkpoint carries and what the validator insists on.
    loss = sum(parameter.sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    return optimizer, scheduler


def make_payload(model, parts, *, namespace="canonical", role="resume", **overrides):
    optimizer, scheduler = parts
    cursor = {
        "train_order_version": "phase9_train_order_v1",
        "namespace": namespace,
        "iteration": 1,
        "sealed_rollout_digest": ROLLOUT_DIGEST,
        "epoch": 0,
        "minibatch_index": 3,
        "examples_consumed": 1536,
        "total_examples": 4096,
        "minibatch_size": 512,
        "epochs": 2,
    }
    fields = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_config": {"namespace": namespace, "learning_rate": 3e-4},
        "train_config_digest": "c" * 64,
        "corpus_identity": ACCEPTED_CORPUS,
        "global_optimizer_step": 3,
        "rl_iteration": 1,
        "minibatch_cursor": cursor,
        "examples_consumed": 1536,
        "behavior_snapshot_identity": "B001",
        "behavior_checkpoint_sha256": BEHAVIOR_SHA,
        "rollout_iteration_identity": f"phase9_rollout_v1|ns={namespace}|it=001",
        "sealed_rollout_digest": ROLLOUT_DIGEST,
        "kl_beta": 0.005,
        "kl_controller_state": {"beta": 0.005, "target": 0.015, "history": []},
        "entropy_schedule_position": {"iteration": 1, "total_iterations": 8},
        "active_historical_identities": ["H000"],
        "historical_checkpoint_digests": {"H000": BEHAVIOR_SHA},
        "best_validation_score": None,
        "best_checkpoint_identity": None,
        "validation_history": [],
        "wall_clock_counters": {"train_seconds": 1.0},
        "diagnostics": {"device": "cpu"},
        "snapshot_role": role,
    }
    fields.update(overrides)
    return pck.build_phase9_checkpoint_payload(**fields)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_payload_carries_every_contract_field(model, parts):
    payload = make_payload(model, parts)
    for field in CHECKPOINT_REQUIRED_FIELDS:
        assert field in payload, field
    assert payload["phase9_checkpoint_version"] == PHASE9_CHECKPOINT_VERSION
    assert payload["example_version"] == PHASE9_EXAMPLE_VERSION
    assert payload["population_version"] == PHASE9_POPULATION_VERSION


def test_round_trip_saves_reads_and_validates(model, parts, tmp_path):
    payload = make_payload(model, parts)
    written = pck.save_phase9_checkpoint(payload, tmp_path / "ckpt.pt", fsync=False)
    assert written["bytes"] > 0
    reloaded = pck.read_phase9_payload(tmp_path / "ckpt.pt")
    metadata = pck.validate_phase9_payload(reloaded, source="reloaded")
    assert metadata["global_optimizer_step"] == 3
    assert metadata["sealed_rollout_digest"] == ROLLOUT_DIGEST
    assert metadata["minibatch_cursor"]["minibatch_index"] == 3


def test_load_restores_model_optimizer_and_controller(model, parts, tmp_path):
    payload = make_payload(model, parts)
    pck.save_phase9_checkpoint(payload, tmp_path / "ckpt.pt", fsync=False)
    restored = pck.load_phase9_checkpoint(tmp_path / "ckpt.pt")
    assert restored["global_optimizer_step"] == 3
    assert restored["kl_beta"] == pytest.approx(0.005)
    assert restored["minibatch_cursor"]["epoch"] == 0
    for name, tensor in model.state_dict().items():
        assert torch.equal(restored["model"].state_dict()[name].cpu(), tensor.cpu())


def test_diagnostics_hold_paths_and_identity_does_not(model, parts):
    payload = make_payload(model, parts, diagnostics={"device": "cpu", "root": "/tmp/x"})
    identity_text = json.dumps(
        {
            key: value
            for key, value in payload.items()
            if key
            not in ("model_state", "optimizer_state", "scheduler_state", "rng", "diagnostics")
        },
        default=str,
    )
    assert "/tmp/x" not in identity_text
    assert payload["diagnostics"]["root"] == "/tmp/x"


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def test_truncation_is_rejected(model, parts, tmp_path):
    payload = make_payload(model, parts)
    path = tmp_path / "ckpt.pt"
    pck.save_phase9_checkpoint(payload, path, fsync=False)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(pck.Phase9CheckpointFormatError, match="could not be read"):
        pck.read_phase9_payload(path)


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.pt"
    path.write_bytes(b"")
    with pytest.raises(pck.Phase9CheckpointFormatError, match="empty"):
        pck.read_phase9_payload(path)


def test_integrity_digest_mismatch_is_rejected(model, parts):
    payload = make_payload(model, parts)
    payload["global_optimizer_step"] = 9999
    with pytest.raises(pck.Phase9CheckpointFormatError, match="integrity digest"):
        pck.validate_phase9_payload(payload)


def test_missing_field_is_rejected(model, parts):
    payload = make_payload(model, parts)
    del payload["kl_controller_state"]
    with pytest.raises(pck.Phase9CheckpointFormatError, match="missing required"):
        pck.validate_phase9_payload(payload)


def test_unknown_field_is_rejected(model, parts):
    payload = make_payload(model, parts)
    payload["something_new"] = 1
    with pytest.raises(pck.Phase9CheckpointFormatError, match="unknown field"):
        pck.validate_phase9_payload(payload)


def test_foreign_contract_digest_is_rejected(model, parts):
    payload = make_payload(model, parts)
    payload["contract_digest"] = "0" * 64
    payload["integrity_digest"] = pck.payload_integrity_digest(payload)
    with pytest.raises(pck.Phase9CheckpointError, match="contract_digest"):
        pck.validate_phase9_payload(payload)


def test_corpus_identity_drift_is_rejected(model, parts):
    payload = make_payload(model, parts)
    drifted = CorpusIdentity(
        corpus_version="synthetic_warmstart_corpus_v1",
        content_digest="d" * 64,
        metadata_digest=ACCEPTED_CORPUS.metadata_digest,
        commit_index_digest=ACCEPTED_CORPUS.commit_index_digest,
    )
    with pytest.raises(pck.Phase9CheckpointIdentityError, match="corpus identity"):
        pck.check_phase9_resume_identity(payload, expected_corpus_identity=drifted)


def test_rollout_digest_drift_is_rejected(model, parts):
    payload = make_payload(model, parts)
    with pytest.raises(pck.Phase9CheckpointIdentityError, match="sealed rollout digest"):
        pck.check_phase9_resume_identity(
            payload, expected_sealed_rollout_digest="e" * 64
        )


def test_rollout_identity_drift_is_rejected(model, parts):
    payload = make_payload(model, parts)
    with pytest.raises(pck.Phase9CheckpointIdentityError, match="rollout identity"):
        pck.check_phase9_resume_identity(
            payload, expected_rollout_identity="phase9_rollout_v1|ns=canonical|it=002"
        )


def test_behavior_snapshot_drift_is_rejected(model, parts):
    payload = make_payload(model, parts)
    with pytest.raises(pck.Phase9CheckpointIdentityError, match="behavior checkpoint"):
        pck.check_phase9_resume_identity(
            payload, expected_behavior_checkpoint_sha256="f" * 64
        )
    with pytest.raises(pck.Phase9CheckpointIdentityError, match="behavior snapshot"):
        pck.check_phase9_resume_identity(
            payload, expected_behavior_snapshot_identity="B002"
        )


def test_train_config_mismatch_is_rejected(model, parts):
    payload = make_payload(model, parts)
    with pytest.raises(pck.Phase9CheckpointIdentityError, match="train config digest"):
        pck.check_phase9_resume_identity(payload, expected_train_config_digest="0" * 64)
    with pytest.raises(pck.Phase9CheckpointIdentityError, match="train config differs"):
        pck.check_phase9_resume_identity(
            payload,
            expected_train_config={"namespace": "canonical", "learning_rate": 6e-4},
        )


def test_population_version_mismatch_is_rejected(model, parts):
    payload = make_payload(model, parts)
    with pytest.raises(pck.Phase9CheckpointIdentityError, match="population version"):
        pck.check_phase9_resume_identity(
            payload, expected_population_version="phase9_population_v2"
        )


def test_cursor_mismatch_is_rejected(model, parts):
    payload = make_payload(model, parts)
    cursor = dict(payload["minibatch_cursor"])
    cursor["minibatch_index"] = 4
    with pytest.raises(pck.Phase9CheckpointIdentityError, match="cursor differs"):
        pck.check_phase9_resume_identity(payload, expected_cursor=cursor)


def test_matching_expectations_authorize_the_resume(model, parts):
    payload = make_payload(model, parts)
    authorized = pck.check_phase9_resume_identity(
        payload,
        expected_corpus_identity=ACCEPTED_CORPUS,
        expected_sealed_rollout_digest=ROLLOUT_DIGEST,
        expected_behavior_checkpoint_sha256=BEHAVIOR_SHA,
        expected_behavior_snapshot_identity="B001",
        expected_population_version=PHASE9_POPULATION_VERSION,
        expected_cursor=payload["minibatch_cursor"],
    )
    assert authorized["sealed_rollout_digest"] == ROLLOUT_DIGEST


# ---------------------------------------------------------------------------
# Atomic writing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["after_write", "after_validate"])
def test_a_crash_before_commit_leaves_no_destination_file(model, parts, tmp_path, stage):
    payload = make_payload(model, parts)
    path = tmp_path / "ckpt.pt"

    def crash(reached):
        if reached == stage:
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        pck.save_phase9_checkpoint(payload, path, fsync=False, crash_hook=crash)
    assert not path.exists()
    assert path.with_suffix(".pt.partial").exists()


def test_a_crash_after_commit_leaves_a_valid_file(model, parts, tmp_path):
    payload = make_payload(model, parts)
    path = tmp_path / "ckpt.pt"

    def crash(reached):
        if reached == "after_commit":
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        pck.save_phase9_checkpoint(payload, path, fsync=False, crash_hook=crash)
    assert path.exists()
    pck.validate_phase9_payload(pck.read_phase9_payload(path), source=str(path))


def test_an_overwrite_never_leaves_a_half_written_destination(model, parts, tmp_path):
    path = tmp_path / "ckpt.pt"
    pck.save_phase9_checkpoint(make_payload(model, parts), path, fsync=False)
    first = pck.read_phase9_payload(path)["global_optimizer_step"]

    def crash(reached):
        if reached == "after_write":
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError):
        pck.save_phase9_checkpoint(
            make_payload(model, parts, global_optimizer_step=77),
            path,
            fsync=False,
            crash_hook=crash,
        )
    assert pck.read_phase9_payload(path)["global_optimizer_step"] == first


# ---------------------------------------------------------------------------
# The namespace-qualified immutable archive
# ---------------------------------------------------------------------------


def test_archive_identity_is_namespace_qualified():
    """The one claim the supplementary instruction singles out."""
    a = pck.qualified_archive_identity("pilot_p9a", "H005")
    b = pck.qualified_archive_identity("pilot_p9b", "H005")
    c = pck.qualified_archive_identity("canonical", "H005")
    assert a == "pilot_p9a|H005"
    assert len({a, b, c}) == 3
    # The anchor is the documented exception: one file, one name, no namespace.
    assert pck.qualified_archive_identity("pilot_p9a", HISTORICAL_ANCHOR_ID) == (
        HISTORICAL_ANCHOR_ID
    )


def test_same_local_number_in_two_namespaces_is_two_objects(model, parts, tmp_path):
    root = tmp_path / "archive"
    members = {}
    for namespace in ("pilot_p9a", "canonical"):
        payload = make_payload(
            model,
            parts,
            namespace=namespace,
            role="archive_member",
            train_config={"namespace": namespace, "learning_rate": 3e-4},
        )
        members[namespace] = pck.write_archive_member(
            payload, root, namespace=namespace, local_identity="H005", fsync=False
        )
    assert members["pilot_p9a"].path != members["canonical"].path
    assert members["pilot_p9a"].qualified_identity != members["canonical"].qualified_identity
    assert members["pilot_p9a"].local_identity == members["canonical"].local_identity
    assert members["pilot_p9a"].policy_token != members["canonical"].policy_token


def test_an_archive_member_is_never_overwritten(model, parts, tmp_path):
    root = tmp_path / "archive"
    payload = make_payload(model, parts, namespace="pilot_p9a", role="archive_member",
                           train_config={"namespace": "pilot_p9a"})
    pck.write_archive_member(
        payload, root, namespace="pilot_p9a", local_identity="H005", fsync=False
    )
    with pytest.raises(pck.Phase9CheckpointError, match="immutable"):
        pck.write_archive_member(
            payload, root, namespace="pilot_p9a", local_identity="H005", fsync=False
        )


def test_the_phase8_anchor_is_never_written_by_a_phase9_run(model, parts, tmp_path):
    payload = make_payload(model, parts, namespace="pilot_p9a", role="archive_member",
                           train_config={"namespace": "pilot_p9a"})
    with pytest.raises(pck.Phase9CheckpointError, match="accepted Phase 8 checkpoint"):
        pck.write_archive_member(
            payload,
            tmp_path,
            namespace="pilot_p9a",
            local_identity=HISTORICAL_ANCHOR_ID,
            fsync=False,
        )


def test_a_member_may_not_be_filed_under_another_runs_namespace(model, parts, tmp_path):
    payload = make_payload(model, parts, namespace="pilot_p9a", role="archive_member",
                           train_config={"namespace": "pilot_p9a"})
    with pytest.raises(pck.Phase9CheckpointError, match="another run's namespace"):
        pck.write_archive_member(
            payload, tmp_path, namespace="canonical", local_identity="H005", fsync=False
        )


def test_a_resume_checkpoint_is_not_an_archive_member(model, parts, tmp_path):
    payload = make_payload(model, parts, namespace="pilot_p9a", role="resume",
                           train_config={"namespace": "pilot_p9a"})
    with pytest.raises(pck.Phase9CheckpointError, match="snapshot role"):
        pck.write_archive_member(
            payload, tmp_path, namespace="pilot_p9a", local_identity="H005", fsync=False
        )


def test_an_archive_member_binds_to_a_frozen_playable_snapshot(model, parts, tmp_path):
    root = tmp_path / "archive"
    payload = make_payload(model, parts, namespace="pilot_p9a", role="archive_member",
                           train_config={"namespace": "pilot_p9a"})
    member = pck.write_archive_member(
        payload, root, namespace="pilot_p9a", local_identity="H005", fsync=False
    )
    snapshot = pck.bind_archive_member(member, device="cpu", inference_batch_shape=4)
    assert snapshot.logical_identity == "H005"
    assert snapshot.policy_token == "phase9_archive_v1|ns=pilot_p9a|H005"
    assert snapshot.checkpoint_sha256 == member.checkpoint_sha256
    snapshot.assert_frozen()
    for name, tensor in model.state_dict().items():
        assert torch.equal(snapshot.model.state_dict()[name].cpu(), tensor.cpu())


def test_binding_a_member_to_the_wrong_digest_is_refused(model, parts, tmp_path):
    from stratego.training.phase9_behavior import Phase9BehaviorError

    root = tmp_path / "archive"
    payload = make_payload(model, parts, namespace="pilot_p9a", role="archive_member",
                           train_config={"namespace": "pilot_p9a"})
    member = pck.write_archive_member(
        payload, root, namespace="pilot_p9a", local_identity="H005", fsync=False
    )
    with pytest.raises(Phase9BehaviorError, match="bound to"):
        pck.bind_archive_member(member, expected_sha256="9" * 64)


def test_archive_manifest_lists_members_with_identities(model, parts, tmp_path):
    root = tmp_path / "archive"
    for local in ("H005", "H010"):
        payload = make_payload(model, parts, namespace="pilot_p9a", role="archive_member",
                               train_config={"namespace": "pilot_p9a"})
        pck.write_archive_member(
            payload, root, namespace="pilot_p9a", local_identity=local, fsync=False
        )
    manifest = pck.archive_manifest(root, "pilot_p9a")
    assert [entry["local_identity"] for entry in manifest["members"]] == ["H005", "H010"]
    assert all(
        entry["qualified_identity"].startswith("pilot_p9a|") for entry in manifest["members"]
    )


def test_unknown_namespace_is_refused(tmp_path):
    with pytest.raises(pck.Phase9CheckpointError, match="unknown Phase 9 namespace"):
        pck.archive_directory(tmp_path, "not_a_namespace")


def test_checkpoint_semantics_lists_every_rejection():
    semantics = pck.checkpoint_semantics()
    text = " ".join(semantics["rejections"])
    for expected in (
        "truncation",
        "integrity digest",
        "corpus identity",
        "sealed rollout digest",
        "behavior snapshot",
        "population-version",
        "cursor",
    ):
        assert expected in text
    assert semantics["required_fields"] == list(CHECKPOINT_REQUIRED_FIELDS)
