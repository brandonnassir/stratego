"""Phase 9 Agent 3: the crash-safe rollout store.

The store makes exactly one promise: a game is visible if and only if its
payload, its metadata and its commit record all exist and verify. Everything
here tests that promise from the outside — what survives a crash, what a
resume regenerates, what a seal refuses — rather than the order of writes that
happens to implement it.

The negative controls matter more than the positive ones. A store that sealed
an iteration with a missing game, an unscheduled game, an orphan record or two
behavior identities in it would hand Agent 4 a rollout whose importance ratios
are quietly wrong, and every one of those failures is silent without a test
that plants it.
"""

from __future__ import annotations

import dataclasses
import json
import shutil

import pytest

from stratego.training import phase9_collector as pc
from stratego.training import phase9_rollout_store as store
from stratego.training.phase9_contract import (
    PHASE9_POPULATION_VERSION,
    PHASE9_ROLLOUT_SCHEDULE_VERSION,
)
from stratego.training.phase9_schedule import rebuild_scheduled_game
from stratego.training.phase9_seed import phase9_game_id

ANCHOR_CHECKPOINT = "checkpoints/phase8/warmstart_c1_v1.pt"
ANCHOR_SHA256 = "f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca"
CONTRACT_DIGEST = "ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34"

#: Three real games, one per opponent kind that can appear at iteration 1.
SAMPLE_GAMES = (
    ("current", 0),
    ("historical", 0),
    ("rule", 0),
)


@pytest.fixture(scope="module")
def participants():
    resolver = pc.SnapshotResolver(device="cpu", inference_batch_shape=1)
    behavior = resolver.resolve(
        ANCHOR_CHECKPOINT,
        logical_identity="B001",
        policy_token="phase9_behavior_v1|ns=canonical|B001",
        expected_sha256=ANCHOR_SHA256,
    )
    anchor = resolver.resolve(
        ANCHOR_CHECKPOINT,
        logical_identity="H000",
        policy_token="phase9_anchor_v1|H000",
        expected_sha256=ANCHOR_SHA256,
    )
    return pc.IterationParticipants(behavior=behavior, historical={"H000": anchor})


@pytest.fixture(scope="module")
def collected(participants):
    """`[(record, metadata), ...]` for three real games, played once."""
    games = []
    for bucket, ordinal in SAMPLE_GAMES:
        scheduled = rebuild_scheduled_game(phase9_game_id("canonical", 1, bucket, ordinal))
        runner = pc.play_game(scheduled, participants)
        opponent_digest = (
            ANCHOR_SHA256 if scheduled.opponent_kind == "historical_snapshot" else None
        )
        metadata = store.build_rollout_metadata(
            scheduled,
            runner.record,
            setup_provenance=runner.assignment.provenance,
            behavior_checkpoint_sha256=ANCHOR_SHA256,
            opponent_checkpoint_sha256=opponent_digest,
            learner_decision_count=runner.learner_decision_count,
            population_version=PHASE9_POPULATION_VERSION,
            schedule_version=PHASE9_ROLLOUT_SCHEDULE_VERSION,
            contract_digest=CONTRACT_DIGEST,
        )
        games.append((runner.record, metadata))
    return games


def _write(root, games, *, worker_id=0, crash_hook=None, target_bytes=None):
    writer = store.Phase9RolloutWriter(
        root,
        namespace="canonical",
        iteration=1,
        worker_id=worker_id,
        crash_hook=crash_hook,
        **({} if target_bytes is None else {"target_bytes": target_bytes}),
    )
    try:
        for record, metadata in games:
            writer.write_game(record, metadata)
    finally:
        writer.close()
    return writer


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_carries_every_frozen_field(collected):
    _record, metadata = collected[0]
    assert tuple(metadata) == store.METADATA_FIELDS
    assert not store.validate_rollout_metadata(metadata, _record)


def test_historical_metadata_records_the_opponents_own_checkpoint(collected):
    """The distinction the whole per-side identity story rests on."""
    record, metadata = next(
        pair for pair in collected if pair[1]["opponent_kind"] == "historical_snapshot"
    )
    assert metadata["behavior_snapshot_id"] == "B001"
    assert metadata["opponent_identity"] == "phase9_anchor_v1|H000"
    assert metadata["opponent_checkpoint_sha256"] == ANCHOR_SHA256
    # The payload's game-level digest names the *current* snapshot only.
    assert record.collection_checkpoint_id == metadata["behavior_checkpoint_sha256"]


def test_a_historical_game_without_an_opponent_digest_is_refused(collected):
    _record, metadata = next(
        pair for pair in collected if pair[1]["opponent_kind"] == "historical_snapshot"
    )
    broken = dict(metadata, opponent_checkpoint_sha256=None)
    problems = store.validate_rollout_metadata(broken)
    assert any("unaddressable" in problem for problem in problems)


def test_a_rule_opponent_may_not_claim_a_checkpoint(collected):
    _record, metadata = next(
        pair for pair in collected if pair[1]["opponent_kind"] == "rule_policy"
    )
    broken = dict(metadata, opponent_checkpoint_sha256=ANCHOR_SHA256)
    assert any(
        "claims a checkpoint" in problem for problem in store.validate_rollout_metadata(broken)
    )


def test_metadata_disagreeing_with_its_game_id_is_refused(collected):
    _record, metadata = collected[0]
    assert store.validate_rollout_metadata(dict(metadata, bucket="stress"))
    assert store.validate_rollout_metadata(dict(metadata, iteration=7))


def test_metadata_disagreeing_with_its_payload_is_refused(collected):
    record, metadata = collected[0]
    assert store.validate_rollout_metadata(dict(metadata, final_ply=1), record)
    assert store.validate_rollout_metadata(dict(metadata, terminal_result="draw"), record) or (
        record.terminal_result == "draw"
    )


# ---------------------------------------------------------------------------
# The commit protocol
# ---------------------------------------------------------------------------


def test_committed_games_read_back_by_random_access(tmp_path, collected):
    _write(tmp_path, collected)
    reader = store.Phase9RolloutReader(tmp_path, "canonical", 1)
    assert len(reader) == len(collected)
    for record, metadata in collected:
        stored_record, stored_metadata = reader.read_game(record.game_id)
        assert stored_record.actions == record.actions
        assert stored_record.decisions == record.decisions
        assert stored_metadata == metadata
    assert reader.orphans() == {"metadata_without_commit": [], "commit_without_metadata": []}


def test_a_shard_rollover_does_not_break_random_access(tmp_path, collected):
    # A tiny target forces a rollover between every game.
    _write(tmp_path, collected, target_bytes=1)
    reader = store.Phase9RolloutReader(tmp_path, "canonical", 1)
    assert len({commit.shard_name for commit in reader.commits.values()}) > 1
    for record, _metadata in collected:
        assert reader.read_game(record.game_id)[0].actions == record.actions


def test_a_second_writer_may_not_reopen_a_file_set(tmp_path, collected):
    _write(tmp_path, collected)
    with pytest.raises(store.Phase9RolloutStoreError, match="fresh worker id"):
        store.Phase9RolloutWriter(tmp_path, namespace="canonical", iteration=1, worker_id=0)
    assert store.next_worker_id(tmp_path, "canonical", 1) == 1


@pytest.mark.parametrize("stage", store.CRASH_STAGES)
def test_recovery_exposes_committed_games_and_nothing_else(tmp_path, collected, stage):
    """Every critical interruption point, recovered by the truncation rule."""

    class Boom(RuntimeError):
        pass

    calls = {"n": 0}

    def crash_hook(name, _writer):
        if name != stage:
            return
        calls["n"] += 1
        if calls["n"] == 2:  # let one game commit first, then die on the next
            raise Boom(stage)

    with pytest.raises(Boom):
        _write(tmp_path, collected, crash_hook=crash_hook, target_bytes=1)

    before = store.Phase9RolloutReader(tmp_path, "canonical", 1)
    survived = set(before.commits)
    report = store.reconcile_iteration(tmp_path, "canonical", 1)
    after = store.Phase9RolloutReader(tmp_path, "canonical", 1)

    assert set(after.commits) == survived, "reconciliation removed a committed game"
    assert after.orphans() == {"metadata_without_commit": [], "commit_without_metadata": []}
    assert not report["duplicate_game_ids"]
    for game_id in after.game_ids:
        after.read_game(game_id)  # decodes and digest-checks


def test_resume_regenerates_only_the_missing_games(tmp_path, collected):
    _write(tmp_path, collected[:1])
    store.reconcile_iteration(tmp_path, "canonical", 1)
    pending = store.pending_game_ids(tmp_path, "canonical", 1)
    assert collected[0][0].game_id not in pending
    assert len(pending) == 2048 - 1

    _write(tmp_path, collected[1:], worker_id=1)
    reader = store.Phase9RolloutReader(tmp_path, "canonical", 1)
    assert len(reader) == 3
    remaining = store.pending_game_ids(tmp_path, "canonical", 1)
    assert not set(pair[0].game_id for pair in collected) & set(remaining)


def test_a_foreign_committed_game_is_reported_rather_than_ignored(tmp_path, collected):
    record, metadata = collected[0]
    foreign_id = phase9_game_id("pilot_p9a", 1, "current", 0)
    record = dataclasses.replace(record, game_id=foreign_id)
    metadata = dict(
        metadata, game_id=foreign_id, namespace="canonical", ordinal=0, bucket="current"
    )
    writer = store.Phase9RolloutWriter(
        tmp_path, namespace="canonical", iteration=1, worker_id=0
    )
    try:
        # The writer itself rejects it, because the id names another namespace.
        with pytest.raises(store.Phase9RolloutStoreError):
            writer.write_game(record, metadata)
    finally:
        writer.close()


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_the_sealed_digest_is_location_independent(tmp_path, collected):
    """Moving verified bytes must not change rollout identity."""
    first = tmp_path / "volume_a"
    second = tmp_path / "volume_b"
    _write(first, collected)
    original = store.sealed_rollout_digest(
        store.Phase9RolloutReader(first, "canonical", 1).commits
    )
    shutil.copytree(first, second)
    relocated = store.sealed_rollout_digest(
        store.Phase9RolloutReader(second, "canonical", 1).commits
    )
    assert relocated == original


def test_the_sealed_digest_is_independent_of_worker_partitioning(tmp_path, collected):
    one = tmp_path / "one_worker"
    many = tmp_path / "three_workers"
    _write(one, collected)
    for index, game in enumerate(collected):
        _write(many, [game], worker_id=index)
    assert store.sealed_rollout_digest(
        store.Phase9RolloutReader(many, "canonical", 1).commits
    ) == store.sealed_rollout_digest(
        store.Phase9RolloutReader(one, "canonical", 1).commits
    )


def test_the_sealed_digest_changes_when_a_game_changes(tmp_path, collected):
    root = tmp_path / "a"
    _write(root, collected)
    commits = list(store.Phase9RolloutReader(root, "canonical", 1).commits.values())
    original = store.sealed_rollout_digest(commits)
    tampered = [dataclasses.replace(commits[0], payload_sha256="0" * 64)] + commits[1:]
    assert store.sealed_rollout_digest(tampered) != original


# ---------------------------------------------------------------------------
# The seal rule
# ---------------------------------------------------------------------------


def _seal(root):
    return store.seal_iteration(
        root, "canonical", 1, expected_behavior_checkpoint=ANCHOR_SHA256
    )


def test_an_incomplete_iteration_does_not_seal(tmp_path, collected):
    _write(tmp_path, collected)
    summary = _seal(tmp_path)
    assert not summary["sealed"]
    assert summary["missing_games"] == 2048 - 3
    assert store.read_iteration_state(tmp_path, "canonical", 1)["state"] == "COLLECTING"


def test_a_complete_iteration_seals_and_becomes_immutable(tmp_path, participants, collected):
    """A whole iteration, kept affordable by shrinking the scheduled set."""
    import stratego.training.phase9_rollout_store as module

    scheduled = tuple(record.game_id for record, _metadata in collected)
    original = module.iteration_game_ids
    module.iteration_game_ids = lambda namespace, iteration: scheduled
    try:
        _write(tmp_path, collected)
        summary = _seal(tmp_path)
        assert summary["sealed"], summary["problems"]
        assert summary["committed_games"] == len(collected)
        assert summary["behavior_snapshot_identities"] == ["B001"]
        assert summary["behavior_checkpoint_digests"] == [ANCHOR_SHA256]

        state = store.read_iteration_state(tmp_path, "canonical", 1)
        assert state["state"] == "SEALED"
        assert state["sealed_rollout_digest"] == summary["sealed_rollout_digest"]
        manifest = json.loads(
            (store.iteration_directory(tmp_path, "canonical", 1) / "manifest.json").read_text()
        )
        assert manifest["sealed_rollout_digest"] == summary["sealed_rollout_digest"]

        # Sealed rollouts are immutable: no writer may open one.
        with pytest.raises(store.Phase9RolloutStoreError, match="immutable"):
            store.Phase9RolloutWriter(
                tmp_path, namespace="canonical", iteration=1, worker_id=9
            )
        with pytest.raises(store.Phase9RolloutStoreError, match="immutable"):
            store.write_iteration_state(tmp_path, "canonical", 1, "COLLECTING")
    finally:
        module.iteration_game_ids = original


def test_two_behavior_identities_in_one_iteration_block_the_seal(
    tmp_path, participants, collected
):
    """The single hardest failure the seal rule exists to catch."""
    import stratego.training.phase9_rollout_store as module

    record, metadata = collected[0]
    other = "1" * 64
    mixed_record = dataclasses.replace(record, collection_checkpoint_id=other)
    mixed_metadata = dict(
        metadata, behavior_snapshot_id="B002", behavior_checkpoint_sha256=other
    )
    games = [(mixed_record, mixed_metadata)] + list(collected[1:])
    scheduled = tuple(pair[0].game_id for pair in games)
    original = module.iteration_game_ids
    module.iteration_game_ids = lambda namespace, iteration: scheduled
    try:
        _write(tmp_path, games)
        summary = _seal(tmp_path)
        assert not summary["sealed"]
        assert any("mixes behavior identities" in problem for problem in summary["problems"])
        assert any("mixes behavior checkpoints" in problem for problem in summary["problems"])
    finally:
        module.iteration_game_ids = original


def test_an_unscheduled_game_blocks_the_seal(tmp_path, collected):
    import stratego.training.phase9_rollout_store as module

    scheduled = (collected[0][0].game_id,)
    original = module.iteration_game_ids
    module.iteration_game_ids = lambda namespace, iteration: scheduled
    try:
        _write(tmp_path, collected)
        summary = _seal(tmp_path)
        assert not summary["sealed"]
        assert summary["unscheduled_games"] == 2
    finally:
        module.iteration_game_ids = original


def test_reconciliation_removes_an_uncommitted_metadata_tail(tmp_path, collected):
    """An orphan sidecar line is uncommitted work, so it is cut, not accepted."""
    _write(tmp_path, collected)
    path = store.metadata_directory(tmp_path, "canonical", 1) / f"w00{store.METADATA_SUFFIX}"
    orphan = dict(collected[0][1], game_id=phase9_game_id("canonical", 1, "current", 5))
    with path.open("ab") as handle:
        handle.write((json.dumps(orphan, sort_keys=True, separators=(",", ":")) + "\n").encode())
    assert store.Phase9RolloutReader(tmp_path, "canonical", 1).orphans()[
        "metadata_without_commit"
    ] == [orphan["game_id"]]

    store.reconcile_iteration(tmp_path, "canonical", 1)
    reader = store.Phase9RolloutReader(tmp_path, "canonical", 1)
    assert reader.orphans() == {"metadata_without_commit": [], "commit_without_metadata": []}
    assert set(reader.commits) == {pair[0].game_id for pair in collected}


def test_a_commit_whose_metadata_vanished_blocks_the_seal(tmp_path, collected):
    """The orphan that truncation cannot repair: a claim with nothing behind it.

    Reconciliation only ever shortens files, so a metadata sidecar that lost
    its tail leaves commits pointing at records that are not there. Sealing has
    to refuse that rather than seal a rollout Agent 4 cannot read.
    """
    import stratego.training.phase9_rollout_store as module

    scheduled = tuple(pair[0].game_id for pair in collected)
    original = module.iteration_game_ids
    module.iteration_game_ids = lambda namespace, iteration: scheduled
    try:
        _write(tmp_path, collected)
        path = (
            store.metadata_directory(tmp_path, "canonical", 1) / f"w00{store.METADATA_SUFFIX}"
        )
        kept = path.read_text().splitlines(keepends=True)[0]
        path.write_text(kept)

        reader = store.Phase9RolloutReader(tmp_path, "canonical", 1)
        assert len(reader.orphans()["commit_without_metadata"]) == 2
        summary = _seal(tmp_path)
        assert not summary["sealed"]
        assert summary["orphan_records"] == 2
    finally:
        module.iteration_game_ids = original


def test_sealing_under_the_wrong_expected_checkpoint_is_refused(tmp_path, collected):
    import stratego.training.phase9_rollout_store as module

    scheduled = tuple(pair[0].game_id for pair in collected)
    original = module.iteration_game_ids
    module.iteration_game_ids = lambda namespace, iteration: scheduled
    try:
        _write(tmp_path, collected)
        summary = store.seal_iteration(
            tmp_path, "canonical", 1, expected_behavior_checkpoint="2" * 64
        )
        assert not summary["sealed"]
        assert any("not the expected" in problem for problem in summary["problems"])
    finally:
        module.iteration_game_ids = original


# ---------------------------------------------------------------------------
# Journal robustness
# ---------------------------------------------------------------------------


def test_a_torn_journal_line_contributes_nothing(tmp_path, collected):
    _write(tmp_path, collected)
    path = store.journal_directory(tmp_path, "canonical", 1) / f"w00{store.JOURNAL_SUFFIX}"
    with path.open("ab") as handle:
        handle.write(b'{"commit_version": "phase9_rollout')  # no newline: interrupted
    commits, valid = store.read_journal(path)
    assert len(commits) == len(collected)
    assert valid < path.stat().st_size
    store.reconcile_iteration(tmp_path, "canonical", 1)
    assert path.stat().st_size == valid


def test_a_foreign_commit_protocol_is_refused(tmp_path, collected):
    _write(tmp_path, collected)
    path = store.journal_directory(tmp_path, "canonical", 1) / f"w00{store.JOURNAL_SUFFIX}"
    line = json.loads(path.read_text().splitlines()[0])
    line["commit_version"] = "some_other_store_v9"
    with path.open("ab") as handle:
        handle.write((json.dumps(line, sort_keys=True) + "\n").encode())
    with pytest.raises(store.Phase9RolloutStoreError, match="commit protocol"):
        store.read_journal(path)


def test_a_corrupted_payload_is_detected_on_read(tmp_path, collected):
    _write(tmp_path, collected)
    reader = store.Phase9RolloutReader(tmp_path, "canonical", 1)
    game_id = reader.game_ids[0]
    commit = reader.commits[game_id]
    shard = (
        store.shards_directory(tmp_path, "canonical", 1)
        / f"{commit.shard_name}{store.SHARD_SUFFIX}"
    )
    raw = bytearray(shard.read_bytes())
    raw[-1] ^= 0xFF
    shard.write_bytes(bytes(raw))
    with pytest.raises(store.Phase9RolloutStoreError):
        store.Phase9RolloutReader(tmp_path, "canonical", 1).read_payload(
            reader.game_ids[-1]
        )


def test_sealing_does_not_drop_the_recorded_collection_conditions(tmp_path, collected):
    """The device and batch shape a rollout was collected under must survive.

    They are the conditions its committed bytes were produced under. A reader
    of a SEALED rollout needs them exactly as much as a resuming collector
    does, so a later transition may add facts but not lose these.
    """
    import stratego.training.phase9_rollout_store as module

    scheduled = tuple(pair[0].game_id for pair in collected)
    original = module.iteration_game_ids
    module.iteration_game_ids = lambda namespace, iteration: scheduled
    try:
        store.write_iteration_state(
            tmp_path,
            "canonical",
            1,
            "COLLECTING",
            behavior_snapshot_id="B001",
            behavior_checkpoint_sha256=ANCHOR_SHA256,
            inference_device="cpu",
            inference_batch_shape=1,
            collector_version="phase9_collector_v1",
        )
        _write(tmp_path, collected)
        summary = _seal(tmp_path)
        assert summary["sealed"], summary["problems"]

        sealed = store.read_iteration_state(tmp_path, "canonical", 1)
        assert sealed["state"] == "SEALED"
        for key in store.STATE_CARRY_FORWARD_KEYS:
            assert sealed.get(key) is not None, key
        assert sealed["inference_device"] == "cpu"
        assert sealed["inference_batch_shape"] == 1
        assert sealed["collector_version"] == "phase9_collector_v1"
        # And the history records both transitions, not just the last one.
        assert [entry["state"] for entry in sealed["history"]] == ["COLLECTING", "SEALED"]
    finally:
        module.iteration_game_ids = original
