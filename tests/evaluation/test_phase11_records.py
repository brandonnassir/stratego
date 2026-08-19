"""Phase 11 Agent 2: prediction storage, digests and path independence."""

import json

import numpy as np
import pytest

from stratego.evaluation.phase11_records import (
    PREDICTION_STORE_VERSION,
    PUBLIC_SHARD_ARRAYS,
    Phase11GameRecorder,
    Phase11StoreError,
    TRUTH_SHARD_ARRAYS,
    game_file_stem,
    iter_records,
    manifest_digest,
    model_identity,
    prediction_identity,
    read_public_shard,
    read_truth_shard,
    shard_digest,
    store_root,
    write_public_shard,
    write_truth_shard,
)
from stratego.training.phase11_contract import (
    PREDICTION_RECORD_FIELDS,
    PREDICTION_RECORD_VERSION,
    PRIVILEGED_RECORD_FIELDS,
)
from stratego.training.phase11_seed import phase11_game_id, phase11_case_id

CASE_ID = phase11_case_id("phase11_validation_bank_v1", "basic_rule", "p10d", 3)
GAME_ID = phase11_game_id(CASE_ID, 0)

MODEL_ID = model_identity("a" * 64, "b" * 64)


def make_recorder(decisions=3, per_decision=2):
    recorder = Phase11GameRecorder(
        {
            "bank_version": "phase11_validation_bank_v1",
            "case_id": CASE_ID,
            "game_id": GAME_ID,
            "game_index": 0,
            "observer_color": "red",
            "opponent_stratum": "basic_rule",
            "opponent_setup_source": "p10d",
        }
    )
    rng = np.random.default_rng(4)
    for decision in range(decisions):
        events = [
            {
                "piece_slot": slot,
                "piece_square": 60 + slot,
                "perspective_square": 39 - slot,
                "piece_moved": bool(slot % 2),
                "legal_rank_mask": (
                    (1,) * 10 + (0, 0) if slot % 2 else (1,) * 12
                ),
                "belief_logits": rng.normal(size=12).astype(np.float32),
            }
            for slot in range(per_decision)
        ]
        recorder.record_decision(
            decision_index=decision * 2,
            public_state_identity=f"{decision:064x}",
            observation_sha256=f"{decision + 100:064x}",
            remaining_counts=(1, 8, 5, 4, 4, 4, 3, 2, 1, 1, 1, 6),
            events=events,
        )
    recorder.action_history = tuple(range(decisions * 2))
    return recorder


def test_the_public_shard_carries_exactly_the_frozen_arrays(tmp_path):
    recorder = make_recorder()
    entry = write_public_shard(tmp_path, recorder)
    arrays = read_public_shard(tmp_path, GAME_ID)
    assert tuple(arrays) == PUBLIC_SHARD_ARRAYS
    assert entry["public_shard_digest"] == shard_digest(arrays, PUBLIC_SHARD_ARRAYS)
    assert entry["decisions"] == 3
    assert entry["events"] == 6
    assert entry["store_version"] == PREDICTION_STORE_VERSION


def test_the_public_shard_holds_no_truth_field(tmp_path):
    write_public_shard(tmp_path, make_recorder())
    arrays = read_public_shard(tmp_path, GAME_ID)
    for name in arrays:
        assert "true" not in name and "truth" not in name
    assert set(TRUTH_SHARD_ARRAYS) & set(arrays) == set()


def test_the_truth_shard_is_a_separate_file(tmp_path):
    write_public_shard(tmp_path, make_recorder())
    write_truth_shard(tmp_path, GAME_ID, np.arange(6, dtype=np.int8))
    public = tmp_path / "public" / f"{game_file_stem(GAME_ID)}.npz"
    truth = tmp_path / "truth" / f"{game_file_stem(GAME_ID)}.npz"
    assert public.exists() and truth.exists() and public != truth
    assert list(read_truth_shard(tmp_path, GAME_ID)) == list(range(6))


def test_a_missing_shard_is_an_error_not_an_empty_result(tmp_path):
    with pytest.raises(Phase11StoreError):
        read_public_shard(tmp_path, GAME_ID)
    with pytest.raises(Phase11StoreError):
        read_truth_shard(tmp_path, GAME_ID)


def test_the_csr_offsets_index_the_events(tmp_path):
    write_public_shard(tmp_path, make_recorder(decisions=4, per_decision=3))
    arrays = read_public_shard(tmp_path, GAME_ID)
    offsets = arrays["event_offset"]
    assert offsets[0] == 0
    assert offsets[-1] == arrays["piece_slot"].size == 12
    assert list(np.diff(offsets)) == [3, 3, 3, 3]


def test_a_decision_with_no_hidden_targets_is_recorded_and_counted(tmp_path):
    recorder = make_recorder(decisions=1)
    recorder.record_decision(
        decision_index=99,
        public_state_identity="f" * 64,
        observation_sha256="e" * 64,
        remaining_counts=(0,) * 12,
        events=[],
    )
    assert recorder.empty_decisions == 1
    entry = write_public_shard(tmp_path, recorder)
    assert entry["decisions"] == 2
    assert entry["empty_decisions"] == 1
    arrays = read_public_shard(tmp_path, GAME_ID)
    assert arrays["event_offset"][-1] == arrays["event_offset"][-2]


def test_records_round_trip_to_the_frozen_schema(tmp_path):
    recorder = make_recorder()
    entry = write_public_shard(tmp_path, recorder)
    arrays = read_public_shard(tmp_path, GAME_ID)
    truth = np.array([0, 1, 2, 3, 4, 5], dtype=np.int8)
    write_truth_shard(tmp_path, GAME_ID, truth)
    records = list(iter_records(entry, arrays, truth, model_id=MODEL_ID))
    assert len(records) == 6
    for record in records:
        assert tuple(record) == PREDICTION_RECORD_FIELDS
        assert record["record_version"] == PREDICTION_RECORD_VERSION
        assert record["model_identity"] == MODEL_ID
        assert len(record["learned_probabilities"]) == 12
        assert sum(record["learned_probabilities"]) == pytest.approx(1.0, abs=1e-12)
        assert sum(record["baseline_probabilities"]) == pytest.approx(1.0, abs=1e-12)
        assert record["prediction_id"].startswith(GAME_ID)


def test_records_without_a_truth_shard_carry_no_label(tmp_path):
    recorder = make_recorder()
    entry = write_public_shard(tmp_path, recorder)
    arrays = read_public_shard(tmp_path, GAME_ID)
    records = list(iter_records(entry, arrays, None, model_id=MODEL_ID))
    assert all(record["true_rank_index"] is None for record in records)
    assert PRIVILEGED_RECORD_FIELDS == ("true_rank_index",)


def test_the_moved_mask_reaches_the_baseline_vector(tmp_path):
    recorder = make_recorder()
    entry = write_public_shard(tmp_path, recorder)
    arrays = read_public_shard(tmp_path, GAME_ID)
    records = list(iter_records(entry, arrays, None, model_id=MODEL_ID))
    moved = [record for record in records if record["piece_moved"]]
    assert moved
    for record in moved:
        assert record["baseline_probabilities"][10] == 0.0
        assert record["baseline_probabilities"][11] == 0.0


def test_prediction_identity_covers_the_stored_content():
    logits = np.zeros(12, dtype=np.float32)
    mask = np.ones(12, dtype=np.uint8)
    counts = np.ones(12, dtype=np.int16)
    base = prediction_identity("p", "s", MODEL_ID, logits, mask, counts)
    changed = logits.copy()
    changed[0] = 1.0
    assert prediction_identity("p", "s", MODEL_ID, changed, mask, counts) != base
    assert prediction_identity("p", "t", MODEL_ID, logits, mask, counts) != base
    assert prediction_identity("p", "s", "other", logits, mask, counts) != base


def test_the_shard_digest_is_content_not_path(tmp_path):
    recorder = make_recorder()
    first = write_public_shard(tmp_path / "here", recorder)
    second = write_public_shard(tmp_path / "elsewhere", recorder)
    assert first["public_shard_digest"] == second["public_shard_digest"]


def test_the_manifest_digest_ignores_paths_and_timings():
    manifest = {"games": 4, "store_root": "/a", "written_at": 1, "duration_seconds": 2.5}
    other = {"games": 4, "store_root": "/b", "written_at": 9, "duration_seconds": 9.9}
    assert manifest_digest(manifest) == manifest_digest(other)
    assert manifest_digest({**manifest, "games": 5}) != manifest_digest(manifest)


def test_the_store_pointer_blocks_on_an_absent_volume(tmp_path):
    (tmp_path / "data").mkdir()
    pointer = tmp_path / "data" / "phase11_prediction_root.txt"
    pointer.write_text("/Volumes/definitely-not-mounted-1234/phase11\n")
    with pytest.raises(Phase11StoreError) as error:
        store_root(tmp_path)
    assert "BLOCKED" in str(error.value)


def test_the_store_root_defaults_inside_the_repository(tmp_path):
    assert store_root(tmp_path).is_relative_to(tmp_path)


def test_a_ragged_event_is_refused(tmp_path):
    recorder = make_recorder(decisions=0)
    with pytest.raises(Phase11StoreError):
        recorder.record_decision(
            decision_index=0,
            public_state_identity="0" * 64,
            observation_sha256="0" * 64,
            remaining_counts=(1,) * 11,
            events=[],
        )
