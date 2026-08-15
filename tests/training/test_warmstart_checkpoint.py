"""Phase 8 Agent 4: `warmstart_checkpoint_v1` — atomicity, rejection, identity.

Three claims are proven here against the real filesystem and real payload
bytes: an interrupted write can never leave an acceptable-but-partial file, a
corrupted or tampered file is refused, and a checkpoint belongs to exactly
one logical run — identified by train-config digest and corpus digests, never
by where the corpus bytes happen to live.
"""

from __future__ import annotations

import shutil

import pytest
import torch

from stratego.model.production_model import build_candidate_model
from stratego.training.warmstart_checkpoint import (
    WARMSTART_CHECKPOINT_VERSION,
    CorpusIdentity,
    WarmstartCheckpointCompatibilityError,
    WarmstartCheckpointFormatError,
    WarmstartCorpusMismatchError,
    build_warmstart_checkpoint_payload,
    load_model_for_evaluation,
    load_warmstart_checkpoint,
    measure_corpus_identity,
    payload_integrity_digest,
    read_warmstart_payload,
    save_warmstart_checkpoint,
    validate_warmstart_payload,
    verify_corpus_identity,
)
from stratego.training.warmstart_dataset import DataCursor


@pytest.fixture(scope="module")
def mini_identity(warmstart_mini_corpus):
    root, _game_ids = warmstart_mini_corpus
    return root, measure_corpus_identity(root)


def fake_train_config(**overrides) -> dict:
    config = {
        "trainer_version": "warmstart_trainer_v1",
        "candidate_id": "unittest_checkpoint",
        "model_candidate": "C0",
        "learning_rate": 1e-3,
        "batch_size": 8,
        "split": "train",
        "order": "shuffle",
    }
    config.update(overrides)
    return config


def config_digest(config: dict) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.fixture()
def small_payload(mini_identity):
    _root, identity = mini_identity
    model = build_candidate_model("C0", seed=7, device="cpu")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    probe = sum(parameter.sum() for parameter in model.parameters())
    probe.backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    config = fake_train_config()
    payload = build_warmstart_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_config=config,
        train_config_digest=config_digest(config),
        corpus_identity=identity,
        cursor=DataCursor(split="train", batch_size=8),
        global_step=1,
        examples_consumed=8,
        best_validation={"score": None, "step": None},
        validation_history=[],
        diagnostics={"device": "cpu", "resolved_corpus_root": "/anywhere/at/all"},
    )
    return payload, config, identity, model


def retampered(payload: dict, mutate) -> dict:
    """A tampered copy whose integrity digest is *recomputed*, so the check
    under test fires instead of the integrity check."""
    copy = {key: value for key, value in payload.items()}
    mutate(copy)
    copy["integrity_digest"] = payload_integrity_digest(copy)
    return copy


class TestAtomicWrites:
    def test_save_validate_reload_roundtrip(self, small_payload, tmp_path):
        payload, _config, _identity, model = small_payload
        path = tmp_path / "checkpoint.pt"
        report = save_warmstart_checkpoint(payload, path)
        assert report["bytes"] > 0
        reloaded = read_warmstart_payload(path)
        metadata = validate_warmstart_payload(reloaded, source=str(path))
        assert metadata["global_step"] == 1
        assert metadata["warmstart_checkpoint_version"] == WARMSTART_CHECKPOINT_VERSION
        # No .partial residue after a clean commit.
        assert not path.with_suffix(path.suffix + ".partial").exists()
        stored = reloaded["model"]["state_dict"]
        for name, tensor in model.state_dict().items():
            assert torch.equal(stored[name], tensor.detach().cpu())

    @pytest.mark.parametrize("stage", ["after_write", "after_validate"])
    def test_crash_before_commit_preserves_previous_checkpoint(
        self, small_payload, tmp_path, stage
    ):
        payload, _config, _identity, _model = small_payload
        path = tmp_path / "checkpoint.pt"
        save_warmstart_checkpoint(payload, path)
        newer = retampered(payload, lambda copy: copy.update(global_step=2))

        class SimulatedCrash(RuntimeError):
            pass

        def crash_hook(reached):
            if reached == stage:
                raise SimulatedCrash(reached)

        with pytest.raises(SimulatedCrash):
            save_warmstart_checkpoint(newer, path, crash_hook=crash_hook)
        # The destination is still the previous complete checkpoint...
        survivor = read_warmstart_payload(path)
        validate_warmstart_payload(survivor, source=str(path))
        assert survivor["global_step"] == 1
        # A crash after the atomic replace instead leaves the new complete
        # checkpoint; that direction is covered by the test below.

    def test_crash_after_commit_leaves_new_complete_checkpoint(
        self, small_payload, tmp_path
    ):
        payload, _config, _identity, _model = small_payload
        path = tmp_path / "checkpoint.pt"
        save_warmstart_checkpoint(payload, path)
        newer = retampered(payload, lambda copy: copy.update(global_step=2))

        class SimulatedCrash(RuntimeError):
            pass

        def crash_hook(reached):
            if reached == "after_commit":
                raise SimulatedCrash(reached)

        with pytest.raises(SimulatedCrash):
            save_warmstart_checkpoint(newer, path, crash_hook=crash_hook)
        survivor = read_warmstart_payload(path)
        validate_warmstart_payload(survivor, source=str(path))
        assert survivor["global_step"] == 2

    def test_missing_and_empty_files_are_format_errors(self, tmp_path):
        with pytest.raises(WarmstartCheckpointFormatError, match="does not exist"):
            read_warmstart_payload(tmp_path / "absent.pt")
        empty = tmp_path / "empty.pt"
        empty.touch()
        with pytest.raises(WarmstartCheckpointFormatError, match="empty"):
            read_warmstart_payload(empty)


class TestCorruptionRejection:
    def test_truncated_files_are_rejected_at_every_cut(self, small_payload, tmp_path):
        payload, _config, _identity, _model = small_payload
        path = tmp_path / "checkpoint.pt"
        save_warmstart_checkpoint(payload, path)
        blob = path.read_bytes()
        for fraction in (0.05, 0.25, 0.5, 0.9, 0.999):
            cut = tmp_path / f"truncated_{fraction}.pt"
            cut.write_bytes(blob[: int(len(blob) * fraction)])
            with pytest.raises(WarmstartCheckpointFormatError):
                validate_warmstart_payload(read_warmstart_payload(cut), source=str(cut))

    def test_content_tampering_fails_the_integrity_digest(
        self, small_payload, tmp_path
    ):
        payload, _config, _identity, _model = small_payload
        path = tmp_path / "checkpoint.pt"
        save_warmstart_checkpoint(payload, path)
        loaded = read_warmstart_payload(path)
        # A plausible-looking edit with a stale digest: the exact shape of a
        # partially-flipped file that still deserializes.
        loaded["examples_consumed"] = 800000
        rewritten = tmp_path / "tampered.pt"
        torch.save(loaded, rewritten)
        with pytest.raises(WarmstartCheckpointFormatError, match="integrity digest"):
            validate_warmstart_payload(
                read_warmstart_payload(rewritten), source=str(rewritten)
            )

    def test_tensor_tampering_fails_the_integrity_digest(self, small_payload, tmp_path):
        payload, _config, _identity, _model = small_payload
        loaded = {key: value for key, value in payload.items()}
        name = next(iter(loaded["model"]["state_dict"]))
        loaded["model"]["state_dict"][name] = (
            loaded["model"]["state_dict"][name].clone() + 1.0
        )
        rewritten = tmp_path / "tensor_tampered.pt"
        torch.save(loaded, rewritten)
        with pytest.raises(WarmstartCheckpointFormatError, match="integrity digest"):
            validate_warmstart_payload(
                read_warmstart_payload(rewritten), source=str(rewritten)
            )

    def test_missing_and_unknown_fields_are_rejected(self, small_payload):
        payload, _config, _identity, _model = small_payload
        amputated = {key: value for key, value in payload.items() if key != "data_cursor"}
        with pytest.raises(WarmstartCheckpointFormatError, match="missing"):
            validate_warmstart_payload(amputated)
        extended = retampered(payload, lambda copy: copy.update(surprise=1))
        with pytest.raises(WarmstartCheckpointFormatError, match="unknown"):
            validate_warmstart_payload(extended)


class TestCompatibilityRejection:
    def test_wrong_checkpoint_version_is_rejected(self, small_payload):
        payload, _config, _identity, _model = small_payload
        tampered = retampered(
            payload,
            lambda copy: copy.update(warmstart_checkpoint_version="warmstart_checkpoint_v0"),
        )
        with pytest.raises(WarmstartCheckpointCompatibilityError, match="version"):
            validate_warmstart_payload(tampered)

    def test_wrong_trainer_version_is_rejected(self, small_payload):
        payload, _config, _identity, _model = small_payload
        tampered = retampered(
            payload, lambda copy: copy.update(trainer_version="warmstart_trainer_v0")
        )
        with pytest.raises(WarmstartCheckpointCompatibilityError, match="trainer"):
            validate_warmstart_payload(tampered)

    def test_wrong_example_version_is_rejected(self, small_payload):
        payload, _config, _identity, _model = small_payload
        tampered = retampered(
            payload, lambda copy: copy.update(example_version="warmstart_example_v0")
        )
        with pytest.raises(WarmstartCheckpointCompatibilityError, match="example_version"):
            validate_warmstart_payload(tampered)

    def test_model_contract_mismatch_is_rejected(self, small_payload):
        payload, _config, _identity, _model = small_payload

        def mutate(copy):
            model_payload = dict(copy["model"])
            model_payload["observation_version"] = "observation_v1_64ch"
            copy["model"] = model_payload

        tampered = retampered(payload, mutate)
        with pytest.raises(
            WarmstartCheckpointCompatibilityError, match="observation_version"
        ):
            validate_warmstart_payload(tampered)

    def test_dishonest_candidate_configuration_is_rejected(self, small_payload):
        payload, _config, _identity, _model = small_payload

        def mutate(copy):
            model_payload = dict(copy["model"])
            configuration = dict(model_payload["model_configuration"])
            configuration["heads"] = 8
            model_payload["model_configuration"] = configuration
            copy["model"] = model_payload

        tampered = retampered(payload, mutate)
        with pytest.raises(WarmstartCheckpointCompatibilityError):
            validate_warmstart_payload(tampered)

    def test_resume_refuses_a_different_train_config(
        self, small_payload, tmp_path
    ):
        payload, config, identity, _model = small_payload
        path = tmp_path / "checkpoint.pt"
        save_warmstart_checkpoint(payload, path)
        other = fake_train_config(learning_rate=3e-4)
        with pytest.raises(
            WarmstartCheckpointCompatibilityError, match="learning_rate"
        ):
            load_warmstart_checkpoint(
                path,
                expected_train_config=other,
                expected_train_config_digest=config_digest(other),
                expected_corpus_identity=identity,
            )

    def test_resume_refuses_a_different_corpus_digest(self, small_payload, tmp_path):
        payload, config, identity, _model = small_payload
        path = tmp_path / "checkpoint.pt"
        save_warmstart_checkpoint(payload, path)
        drifted = CorpusIdentity(
            corpus_version=identity.corpus_version,
            content_digest="0" * 64,
            metadata_digest=identity.metadata_digest,
            commit_index_digest=identity.commit_index_digest,
        )
        with pytest.raises(WarmstartCheckpointCompatibilityError, match="corpus"):
            load_warmstart_checkpoint(
                path,
                expected_train_config=config,
                expected_train_config_digest=config_digest(config),
                expected_corpus_identity=drifted,
            )

    def test_evaluation_only_load_ignores_run_identity(self, small_payload, tmp_path):
        payload, config, identity, model = small_payload
        path = tmp_path / "checkpoint.pt"
        save_warmstart_checkpoint(payload, path)
        # The full resume path refuses a foreign run...
        other = fake_train_config(candidate_id="unittest_other_run")
        with pytest.raises(WarmstartCheckpointCompatibilityError):
            load_warmstart_checkpoint(
                path,
                expected_train_config=other,
                expected_train_config_digest=config_digest(other),
                expected_corpus_identity=identity,
            )
        # ...while the explicit evaluation-only path loads the compatible model.
        evaluated, metadata = load_model_for_evaluation(path)
        assert not evaluated.training
        assert metadata["global_step"] == 1
        for name, tensor in model.state_dict().items():
            assert torch.equal(evaluated.state_dict()[name], tensor)


class TestCorpusIdentity:
    def test_identity_holds_digests_and_no_path(self, mini_identity):
        _root, identity = mini_identity
        payload = identity.to_dict()
        assert set(payload) == {
            "corpus_version",
            "content_digest",
            "metadata_digest",
            "commit_index_digest",
        }
        assert CorpusIdentity.from_dict(payload) == identity

    def test_pure_relocation_preserves_identity(self, mini_identity, tmp_path):
        root, identity = mini_identity
        relocated = tmp_path / "relocated_corpus"
        shutil.copytree(root, relocated)
        moved = measure_corpus_identity(relocated)
        assert moved == identity
        assert verify_corpus_identity(relocated, identity) == identity

    def test_relocated_checkpoint_still_resumes(self, small_payload, tmp_path, mini_identity):
        """A pure relocation with identical digests defines the same corpus,
        so a checkpoint written before the move resumes after it."""
        payload, config, identity, _model = small_payload
        root, _ = mini_identity
        path = tmp_path / "checkpoint.pt"
        save_warmstart_checkpoint(payload, path)
        relocated = tmp_path / "relocated_for_resume"
        shutil.copytree(root, relocated)
        restored = load_warmstart_checkpoint(
            path,
            expected_train_config=config,
            expected_train_config_digest=config_digest(config),
            expected_corpus_identity=measure_corpus_identity(relocated),
        )
        assert restored["global_step"] == 1

    def test_payload_byte_drift_is_a_stop_condition(self, mini_identity, tmp_path):
        # Flipping a byte inside a shard leaves every journal digest intact, so
        # only the byte-level pass can catch it — which is why verification
        # re-reads payloads against their committed digests by default.
        root, identity = mini_identity
        drifted = tmp_path / "drifted_corpus"
        shutil.copytree(root, drifted)
        shard = next(drifted.rglob("*.stgshard"))
        blob = bytearray(shard.read_bytes())
        blob[len(blob) // 2] ^= 0xFF
        shard.write_bytes(bytes(blob))
        with pytest.raises(WarmstartCorpusMismatchError, match="stop condition"):
            verify_corpus_identity(drifted, identity)

    def test_journal_drift_is_a_stop_condition(self, mini_identity, tmp_path):
        # Rewriting a committed digest inside a journal changes the measured
        # content digest, so the identity comparison itself refuses it.
        root, identity = mini_identity
        drifted = tmp_path / "journal_drifted_corpus"
        shutil.copytree(root, drifted)
        journal = next(drifted.rglob("*.commit.jsonl"))
        text = journal.read_text()
        import re

        tampered = re.sub(
            r'"trajectory_sha256":"[0-9a-f]{8}',
            lambda match: match.group(0)[:-8] + "deadbeef",
            text,
            count=1,
        )
        assert tampered != text
        journal.write_text(tampered)
        with pytest.raises(WarmstartCorpusMismatchError, match="stop condition"):
            verify_corpus_identity(drifted, identity)

    def test_verify_without_expectation_returns_measurement(self, mini_identity):
        root, identity = mini_identity
        assert verify_corpus_identity(root, None) == identity
