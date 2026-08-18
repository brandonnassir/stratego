"""The Phase 10 outcome store: a game is visible only once it is committed.

These tests exercise the commit protocol itself rather than the games that go
through it, so they build structurally real records — real train-split setups
resolved from real scheduled game ids, with fabricated outcome halves — and
put them through the writer at every interruption point the instruction names:

```text
before payload / after payload / after metadata / before commit / after commit
between games / at shard rollover
```

The property under test is the same one every time: after
:func:`reconcile_corpus`, the store holds exactly the games whose commit line
was written, and nothing else — no orphan payload, no orphan metadata line, no
torn journal tail.
"""

import json
from pathlib import Path

import pytest

from stratego.training import phase10_outcome_store as store
from stratego.training.phase10_schedule import (
    CORPUS_SPLIT,
    CORPUS_VERSION,
    OUTCOME_RECORD_FIELDS,
    enumerate_schedule,
    resolve_side,
)
from stratego.training.phase10_seed import corpus_match_seed


class InjectedCrash(RuntimeError):
    """A deliberately injected interruption."""


@pytest.fixture(scope="module")
def library():
    from stratego.setups.sampler import load_library_index

    return load_library_index()


@pytest.fixture(scope="module")
def records(library):
    """Six structurally real records, sharing one fabricated outcome shape.

    The setups and their provenance are genuine — resolved from genuine
    scheduled game ids through the frozen sampler — because the store's
    section validation and the reconstruction audit both read them. The
    outcome half is fabricated on purpose: no test here should need a neural
    forward pass to prove that a truncation is exact.
    """
    from stratego.setups.traits import TRAIT_SCHEMA_VERSION

    schedule = enumerate_schedule()
    built = []
    for position, index in enumerate((0, 3_000, 6_000, 9_000, 12_000, 16_383)):
        game = schedule[index]
        sides = {}
        for color in ("red", "blue"):
            sampled, attempt, seed = resolve_side(game.game_id, color, index=library)
            sides[color] = (sampled, attempt, seed)
        setup = {
            "corpus_version": CORPUS_VERSION,
            "record_version": store.OUTCOME_RECORD_VERSION,
            "game_id": game.game_id,
            "red_family": game.red_family,
            "blue_family": game.blue_family,
            "ordinal": game.ordinal,
            "split": CORPUS_SPLIT,
            "match_seed": corpus_match_seed(game.game_id),
            "red_setup_draw_seed": sides["red"][2],
            "blue_setup_draw_seed": sides["blue"][2],
            "red_setup_attempt": sides["red"][1],
            "blue_setup_attempt": sides["blue"][1],
            "red_base_setup_id": sides["red"][0].base_setup_id,
            "blue_base_setup_id": sides["blue"][0].base_setup_id,
            "red_provenance": dict(sides["red"][0].provenance),
            "blue_provenance": dict(sides["blue"][0].provenance),
            "red_final_fingerprint": sides["red"][0].provenance["final_setup_fingerprint"],
            "blue_final_fingerprint": sides["blue"][0].provenance["final_setup_fingerprint"],
            "red_trait_identity": {"trait_schema_version": TRAIT_SCHEMA_VERSION, "base_trait_digest": "0" * 64, "final_trait_digest": "1" * 64},
            "blue_trait_identity": {"trait_schema_version": TRAIT_SCHEMA_VERSION, "base_trait_digest": "2" * 64, "final_trait_digest": "3" * 64},
            "trait_schema_version": TRAIT_SCHEMA_VERSION,
            "library_content_digest": library.content_digest,
            "corpus_contract_digest": "a" * 64,
            "outcome_schedule_digest": "b" * 64,
            "contract_bundle_digest": "c" * 64,
        }
        token = ("red_win", "draw", "red_loss")[position % 3]
        outcome = {
            "result": token,
            "winner": {"red_win": "red", "draw": None, "red_loss": "blue"}[token],
            "red_score": {"red_win": 1.0, "draw": 0.5, "red_loss": 0.0}[token],
            "plies": 100 + position,
            "decisions": 100 + position,
            "terminal_reason": "flag_capture",
            "move_policy_identity": "test_policy@1",
            "move_checkpoint_sha256": "d" * 64,
            "move_model_state_digest": "e" * 64,
        }
        built.append(store.build_stored_record(setup, outcome))
    return built


def write_all(root, records, *, crash_stage=None, victim=None, target_bytes=None):
    """Write `records` into a fresh store, optionally crashing partway."""
    written = {"count": 0}

    def hook(stage, _writer):
        if stage == crash_stage and written["count"] == victim:
            raise InjectedCrash(f"{stage}@{victim}")

    writer = store.OutcomeWriter(
        root,
        segment=0,
        worker_id=0,
        target_bytes=target_bytes or store.DEFAULT_OUTCOME_SHARD_BYTES,
        crash_hook=hook if crash_stage else None,
    )
    crashed = False
    try:
        for record in records:
            writer.write_record(record)
            written["count"] += 1
    except InjectedCrash:
        # A real crash never reaches `close`, so neither does this one: the
        # handles are abandoned exactly as a killed process abandons them.
        crashed = True
    else:
        writer.close()
    return crashed, written["count"]


# ---------------------------------------------------------------------------
# Record shape
# ---------------------------------------------------------------------------


def test_frozen_schema_is_a_strict_subset_of_the_stored_record():
    frozen = {name for name, _text in OUTCOME_RECORD_FIELDS}
    assert frozen == set(store.FROZEN_RECORD_FIELDS)
    assert frozen < set(store.ASSEMBLED_RECORD_FIELDS)
    assert set(store.ADDITIONAL_RECORD_FIELDS) == set(store.ASSEMBLED_RECORD_FIELDS) - frozen


def test_setup_and_outcome_halves_are_disjoint_closed_sets():
    assert not set(store.SETUP_SECTION_FIELDS) & set(store.OUTCOME_SECTION_FIELDS)
    assert not set(store.DERIVED_RECORD_FIELDS) & set(store.SETUP_SECTION_FIELDS)
    assert not set(store.DERIVED_RECORD_FIELDS) & set(store.OUTCOME_SECTION_FIELDS)


def test_an_outcome_field_cannot_hide_in_the_setup_half(records):
    setup = dict(records[0]["setup"])
    setup["result"] = "red_win"
    with pytest.raises(store.OutcomeStoreError, match="unexpected"):
        store.build_stored_record(setup, records[0]["outcome"])


def test_a_missing_setup_field_is_refused(records):
    setup = dict(records[0]["setup"])
    del setup["red_provenance"]
    with pytest.raises(store.OutcomeStoreError, match="missing"):
        store.build_stored_record(setup, records[0]["outcome"])


def test_a_score_that_contradicts_its_result_is_refused(records):
    outcome = dict(records[0]["outcome"])
    outcome["result"] = "draw"
    outcome["red_score"] = 1.0
    with pytest.raises(store.OutcomeStoreError, match="frozen target"):
        store.build_stored_record(records[0]["setup"], outcome)


def test_an_unknown_result_token_is_refused(records):
    outcome = dict(records[0]["outcome"])
    outcome["result"] = "red_almost_win"
    with pytest.raises(store.OutcomeStoreError, match="unknown result"):
        store.build_stored_record(records[0]["setup"], outcome)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_a_clean_write_reads_back_every_record(tmp_path, records):
    crashed, count = write_all(tmp_path, records)
    assert not crashed and count == len(records)

    reader = store.OutcomeReader(tmp_path)
    assert len(reader) == len(records)
    assert reader.game_ids == tuple(sorted(r["setup"]["game_id"] for r in records))
    for record in records:
        assembled = reader.record(record["setup"]["game_id"])
        assert set(assembled) == set(store.ASSEMBLED_RECORD_FIELDS)
        assert assembled["red_provenance"] == record["setup"]["red_provenance"]
        assert assembled["result"] == record["outcome"]["result"]
    assert store.audit_store_integrity(tmp_path)["all_pass"]


def test_canonical_order_is_sorted_game_id_not_write_order(tmp_path, records):
    write_all(tmp_path, list(reversed(records)))
    reader = store.OutcomeReader(tmp_path)
    assert list(reader.game_ids) == sorted(reader.game_ids)


def test_the_three_digests_describe_their_own_bytes(tmp_path, records):
    write_all(tmp_path, records)
    reader = store.OutcomeReader(tmp_path)
    for game_id in reader.game_ids:
        commit = reader.commit(game_id)
        assembled = reader.record(game_id)
        assert assembled["payload_digest"] == store.payload_digest(reader.payload(game_id))
        assert assembled["metadata_digest"] == store.metadata_digest(reader.metadata(game_id))
        assert assembled["commit_digest"] == store.commit_digest(commit.to_dict())


def test_a_tampered_payload_is_detected(tmp_path, records):
    write_all(tmp_path, records)
    reader = store.OutcomeReader(tmp_path)
    shard = reader.shard_paths()[0]
    raw = bytearray(shard.read_bytes())
    raw[-1] ^= 0xFF
    shard.write_bytes(bytes(raw))
    with pytest.raises(store.OutcomeStoreError):
        for _ in store.OutcomeReader(tmp_path).iter_records():
            pass


def test_a_tampered_metadata_line_is_detected(tmp_path, records):
    write_all(tmp_path, records)
    path = next(store.metadata_directory(tmp_path).glob("*" + store.METADATA_SUFFIX))
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["plies"] = first["plies"] + 1
    lines[0] = store.canonical_json(first)
    path.write_text("\n".join(lines) + "\n")
    audit = store.audit_store_integrity(tmp_path)
    assert not audit["all_pass"]
    assert not audit["checks"]["metadata_digests_match"]


# ---------------------------------------------------------------------------
# Crash injection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stage,keeps_victim",
    [
        ("before_payload", False),
        ("after_payload", False),
        ("after_metadata", False),
        ("before_commit_flush", False),
        ("after_commit", True),
    ],
)
def test_recovery_exposes_exactly_the_committed_games(tmp_path, records, stage, keeps_victim):
    victim = 3
    crashed, _count = write_all(tmp_path, records, crash_stage=stage, victim=victim)
    assert crashed

    store.reconcile_corpus(tmp_path)
    reader = store.OutcomeReader(tmp_path)
    expected = victim + 1 if keeps_victim else victim
    assert len(reader) == expected
    assert set(reader.game_ids) == {r["setup"]["game_id"] for r in records[:expected]}
    assert store.audit_store_integrity(tmp_path)["all_pass"]


def test_recovery_at_shard_rollover_keeps_the_committed_prefix(tmp_path, records):
    crashed, _ = write_all(
        tmp_path, records, crash_stage="shard_rollover", victim=3, target_bytes=1
    )
    assert crashed
    store.reconcile_corpus(tmp_path)
    reader = store.OutcomeReader(tmp_path)
    assert len(reader) == 3
    # A one-byte rollover target gives every record its own shard, so the
    # three survivors sit in three distinct ones. The store also holds the
    # shard opened at construction, which rolled over before record 0 landed
    # and is therefore empty rather than uncommitted.
    assert len({reader.commit(game_id).shard_name for game_id in reader.game_ids}) == 3
    assert len(reader.shard_paths()) == 4
    assert store.audit_store_integrity(tmp_path)["all_pass"]


def test_a_shard_written_entirely_after_the_last_commit_is_removed(tmp_path, records):
    """The uncommitted-whole-shard branch of recovery, exercised directly."""
    crashed, _ = write_all(
        tmp_path, records, crash_stage="after_payload", victim=3, target_bytes=1
    )
    assert crashed
    before = len(list(store.records_directory(tmp_path).glob("*" + store.RECORD_SUFFIX)))
    recovery = store.reconcile_corpus(tmp_path)
    after = len(list(store.records_directory(tmp_path).glob("*" + store.RECORD_SUFFIX)))
    assert recovery["shards_removed"]
    assert after < before
    assert len(store.OutcomeReader(tmp_path)) == 3
    assert store.audit_store_integrity(tmp_path)["all_pass"]


def test_reconciliation_is_idempotent(tmp_path, records):
    write_all(tmp_path, records, crash_stage="after_metadata", victim=2)
    first = store.reconcile_corpus(tmp_path)
    second = store.reconcile_corpus(tmp_path)
    assert first["committed_count"] == second["committed_count"]
    assert second["bytes_discarded"] == 0
    assert not second["shards_removed"]


def test_a_torn_journal_line_contributes_nothing(tmp_path, records):
    write_all(tmp_path, records)
    path = next(store.journal_directory(tmp_path).glob("*" + store.JOURNAL_SUFFIX))
    raw = path.read_bytes()
    path.write_bytes(raw + b'{"commit_version":"phase10_out')  # no newline
    commits, valid = store.read_journal(path)
    assert len(commits) == len(records)
    assert valid == len(raw)
    store.reconcile_corpus(tmp_path)
    assert len(store.OutcomeReader(tmp_path)) == len(records)


def test_a_journal_written_by_another_protocol_is_refused(tmp_path, records):
    write_all(tmp_path, records)
    path = next(store.journal_directory(tmp_path).glob("*" + store.JOURNAL_SUFFIX))
    lines = path.read_text().splitlines()
    entry = json.loads(lines[0])
    entry["commit_version"] = "some_other_commit_v9"
    path.write_text(store.canonical_json(entry) + "\n")
    with pytest.raises(store.OutcomeStoreError, match="commit protocol"):
        store.read_journal(path)


def test_a_resumed_run_opens_a_fresh_segment(tmp_path, records):
    write_all(tmp_path, records[:3])
    assert store.next_segment(tmp_path) == 1
    with pytest.raises(store.OutcomeStoreError, match="already exists"):
        store.OutcomeWriter(tmp_path, segment=0, worker_id=0)

    writer = store.OutcomeWriter(tmp_path, segment=1, worker_id=0)
    for record in records[3:]:
        writer.write_record(record)
    writer.close()
    assert len(store.OutcomeReader(tmp_path)) == len(records)


# ---------------------------------------------------------------------------
# Identity and sealing
# ---------------------------------------------------------------------------


def test_content_digest_is_independent_of_partitioning(tmp_path, records):
    one = tmp_path / "one"
    many = tmp_path / "many"
    write_all(one, records)

    many.mkdir()
    for worker_id, chunk in enumerate([records[0::3], records[1::3], records[2::3]]):
        writer = store.OutcomeWriter(many, segment=0, worker_id=worker_id)
        for record in chunk:
            writer.write_record(record)
        writer.close()

    assert store.corpus_content_digest(one) == store.corpus_content_digest(many)
    assert store.OutcomeReader(one).game_ids == store.OutcomeReader(many).game_ids


def test_content_digest_is_independent_of_the_path(tmp_path, records):
    import shutil

    original = tmp_path / "original"
    copy = tmp_path / "copied" / "elsewhere"
    write_all(original, records)
    copy.parent.mkdir(parents=True)
    shutil.copytree(original, copy)
    assert store.corpus_content_digest(original) == store.corpus_content_digest(copy)


def test_sealing_freezes_the_corpus(tmp_path, records):
    write_all(tmp_path, records)
    assert store.read_state(tmp_path) == store.STATE_COLLECTING

    seal = store.seal_corpus(tmp_path, expected_games=len(records))
    assert store.read_state(tmp_path) == store.STATE_SEALED
    assert seal["content_digest"] == store.corpus_content_digest(tmp_path)
    assert store.verify_seal(tmp_path)["all_pass"]


def test_a_sealed_corpus_refuses_every_mutation(tmp_path, records):
    write_all(tmp_path, records)
    store.seal_corpus(tmp_path, expected_games=len(records))

    with pytest.raises(store.OutcomeStoreError, match="SEALED"):
        store.OutcomeWriter(tmp_path, segment=1, worker_id=0)
    with pytest.raises(store.OutcomeStoreError, match="SEALED"):
        store.reconcile_corpus(tmp_path)
    with pytest.raises(store.OutcomeStoreError, match="already SEALED"):
        store.seal_corpus(tmp_path, expected_games=len(records))


def test_sealing_the_wrong_number_of_games_is_refused(tmp_path, records):
    write_all(tmp_path, records)
    with pytest.raises(store.OutcomeStoreError, match="committed games, expected"):
        store.seal_corpus(tmp_path, expected_games=len(records) + 1)
    assert store.read_state(tmp_path) == store.STATE_COLLECTING


def test_verify_seal_detects_a_changed_corpus(tmp_path, records):
    write_all(tmp_path, records)
    store.seal_corpus(tmp_path, expected_games=len(records))
    seal_file = store.seal_path(tmp_path)
    payload = json.loads(seal_file.read_text())
    payload["content_digest"] = "f" * 64
    seal_file.write_text(json.dumps(payload))
    verification = store.verify_seal(tmp_path)
    assert not verification["all_pass"]
    assert not verification["checks"]["content_digest_matches"]


def test_storage_summary_reports_real_bytes(tmp_path, records):
    write_all(tmp_path, records)
    summary = store.storage_summary(tmp_path)
    assert summary["committed_games"] == len(records)
    assert summary["total_bytes"] > 0
    assert 0.0 < summary["compression_ratio"] < 1.0
    assert summary["bytes_per_game"] == pytest.approx(summary["total_bytes"] / len(records))
