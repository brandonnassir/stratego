"""Phase 10 Agent 6's soak machinery: identity, store, and per-game audit.

Everything here is hermetic — stores live in tmp_path, no game is played and
no model is loaded. Records are real production draws (through the frozen
`learned_setup_source_v1` under the selected P10-D configuration) carrying
fabricated outcome fields, which is exactly the boundary the store and the
audit own: the *game* half of a record is the match runner's business, and
the collector tests already cover it.
"""

from __future__ import annotations

import json

import pytest

from stratego.training import phase10_soak as soak
from stratego.training.phase10_seed import (
    PHASE10_MASTER_SEED,
    derive_phase10_seed,
)
from stratego.training.phase10_selector import Phase10SelectorError


@pytest.fixture(scope="module")
def source():
    return soak.build_soak_source()


@pytest.fixture(scope="module")
def identity():
    return soak.soak_identity_block(
        {
            "source_sha256": (
                "dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea"
            ),
            "model_state_digest": (
                "f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd"
            ),
        }
    )


def _record(source, identity, ordinal: int, **outcome) -> dict:
    """One valid soak record: real draws, fabricated outcome fields."""
    game_id = soak.soak_game_id(ordinal)
    sides = soak.draw_soak_sides(source, game_id)
    token = outcome.get("result", "red_win")
    record = {
        "soak_version": soak.SOAK_VERSION,
        "record_version": soak.SOAK_RECORD_VERSION,
        "game_id": game_id,
        "ordinal": ordinal,
        "split": soak.SOAK_SPLIT,
        "selector_config_sha256": identity["selector_config_sha256"],
        "candidate_id": identity["candidate_id"],
        "selector_identity": identity["selector_identity"],
        "match_seed": soak.soak_match_seed(game_id),
        "result": token,
        "winner": None if token == "draw" else ("red" if token == "red_win" else "blue"),
        "red_score": soak.SOAK_RESULT_TARGETS[token],
        "plies": outcome.get("plies", 240),
        "decisions": outcome.get("decisions", 240),
        "terminal_reason": outcome.get("terminal_reason", "flag_captured"),
        "move_policy_identity": soak.soak_policy_ref().token,
        "move_checkpoint_sha256": identity["phase9_checkpoint_sha256"],
        "move_model_state_digest": identity["phase9_model_state_digest"],
        "library_content_digest": identity["library_content_digest"],
        "contract_bundle_digest": identity["contract_bundle_digest"],
    }
    for color in ("red", "blue"):
        draw = sides[color]["draw"]
        record[f"{color}_selector_request"] = dict(sides[color]["request"])
        record[f"{color}_selector_provenance"] = draw.selector_provenance()
        record[f"{color}_setup_provenance"] = dict(draw.setup_provenance)
        record[f"{color}_base_setup_id"] = draw.base_setup_id
        record[f"{color}_family"] = draw.family_id
        record[f"{color}_final_fingerprint"] = draw.final_setup_fingerprint
    return soak.validate_soak_record(record)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_game_id_round_trips_and_rejects_foreign_ids():
    game_id = soak.soak_game_id(42)
    assert game_id == f"phase10_soak_v1|ms={PHASE10_MASTER_SEED}|g=00042"
    parsed = soak.parse_soak_game_id(game_id)
    assert parsed["ordinal"] == 42
    for bad in (
        "phase10_outcome_v1|ms=2026081801|rf=F00|bf=F01|g=07",
        "phase10_soak_v1|ms=999|g=00001",
        "phase10_soak_v1|ms=2026081801|g=1",
        "",
        None,
    ):
        with pytest.raises(soak.Phase10SoakError):
            soak.parse_soak_game_id(bad)
    with pytest.raises(soak.Phase10SoakError):
        soak.soak_game_id(-1)
    with pytest.raises(soak.Phase10SoakError):
        soak.soak_game_id(100_000)


def test_schedule_is_canonical_and_distinct():
    ids = soak.soak_game_ids(64)
    assert len(ids) == len(set(ids)) == 64
    # Zero-padded ordinals make lexicographic order the ordinal order, so the
    # store's canonical sorted order is also the schedule order.
    assert tuple(sorted(ids)) == ids


def test_seed_streams_are_deterministic_and_namespace_separated():
    game_id = soak.soak_game_id(7)
    red = soak.soak_selector_seed(game_id, "red")
    assert red == soak.soak_selector_seed(game_id, "red")
    assert red != soak.soak_selector_seed(game_id, "blue")
    assert soak.soak_match_seed(game_id) not in (
        red,
        soak.soak_selector_seed(game_id, "blue"),
    )
    # The same identity parts under a frozen Agent 1 domain give an unrelated
    # stream: the payload's first token differs, which is the namespace wall.
    assert soak.derive_soak_seed("soak_match", game_id) != derive_phase10_seed(
        "corpus_match", game_id
    )
    with pytest.raises(soak.Phase10SoakError):
        soak.derive_soak_seed("corpus_match", game_id)
    with pytest.raises(soak.Phase10SoakError):
        soak.soak_selector_seed(game_id, "green")


def test_collision_audit_is_clean_on_a_slice():
    audit = soak.soak_seed_collision_audit(64)
    assert audit["no_collisions"], audit["findings"][:4]
    assert audit["streams"]["soak_selector"]["count"] == 128
    assert audit["streams"]["soak_match"]["count"] == 64
    assert audit["total_seeds"] == audit["distinct_seeds"]


def test_match_spec_binds_the_selected_configuration():
    game_id = soak.soak_game_id(3)
    spec = soak.soak_match_spec(game_id)
    assert spec.root_seed == soak.soak_match_seed(game_id)
    assert spec.suite_version == soak.SOAK_VERSION
    assert soak.selected_selector_identity() in spec.setup_bank_version
    assert spec.match_id == soak.soak_match_spec(game_id).match_id
    assert spec.candidate.token == spec.opponent.token


def test_hidden_input_positive_control_fires():
    control = soak.hidden_input_positive_control()
    assert control["all_rejected"]
    with pytest.raises(Phase10SelectorError):
        soak.SelectorRequest.from_payload(
            {"split": "train", "color": "red", "selector_seed": 1, "opponent_family": "F00"}
        )


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


def test_writer_commits_and_reader_reads_back(tmp_path, source, identity):
    records = [_record(source, identity, ordinal) for ordinal in range(3)]
    writer = soak.SoakWriter(tmp_path, segment=0, worker_id=0)
    for record in records:
        writer.write_record(record)
    stats = writer.close()
    assert stats["games_written"] == 3
    reader = soak.SoakReader(tmp_path)
    assert len(reader) == 3
    assert reader.game_ids == tuple(sorted(record["game_id"] for record in records))
    for record in records:
        assert reader.record(record["game_id"]) == record
    assert reader.storage_summary()["bytes_per_game"] > 0
    assert not reader.duplicate_committed_ids


def test_writer_refuses_an_existing_file_set(tmp_path, source, identity):
    writer = soak.SoakWriter(tmp_path, segment=0, worker_id=0)
    writer.write_record(_record(source, identity, 0))
    writer.close()
    with pytest.raises(soak.Phase10SoakError):
        soak.SoakWriter(tmp_path, segment=0, worker_id=0)
    # A fresh segment is the resume path.
    assert soak.next_soak_segment(tmp_path) == 1
    soak.SoakWriter(tmp_path, segment=1, worker_id=0).close()


def test_torn_tails_are_invisible_and_reconcile_truncates(tmp_path, source, identity):
    writer = soak.SoakWriter(tmp_path, segment=0, worker_id=0)
    writer.write_record(_record(source, identity, 0))
    writer.write_record(_record(source, identity, 1))
    writer.close()
    path = writer.journal_path
    intact = path.read_bytes()

    # A torn (newline-less) tail is an interrupted write, by definition.
    path.write_bytes(intact + b'{"commit_version": "phase10_soak_commit_v1", "ga')
    envelopes, valid = soak.read_soak_journal(path)
    assert len(envelopes) == 2 and valid == len(intact)

    # A complete line whose payload digest does not match its record is not a
    # commit either — nor is anything after it.
    tampered = _record(source, identity, 2)
    envelope = {
        "commit_version": soak.SOAK_COMMIT_VERSION,
        "game_id": tampered["game_id"],
        "payload_sha256": "0" * 64,
        "committed_unix": 0.0,
        "record": tampered,
    }
    path.write_bytes(intact + (soak.canonical_json(envelope) + "\n").encode())
    envelopes, valid = soak.read_soak_journal(path)
    assert len(envelopes) == 2 and valid == len(intact)

    recovery = soak.reconcile_soak(tmp_path)
    assert recovery["committed_count"] == 2
    assert recovery["bytes_discarded"] > 0
    assert path.read_bytes() == intact
    # Reconciling a clean store is a no-op.
    assert soak.reconcile_soak(tmp_path)["bytes_discarded"] == 0


def test_duplicate_commits_across_file_sets_are_flagged(tmp_path, source, identity):
    record = _record(source, identity, 5)
    for worker_id in (0, 1):
        writer = soak.SoakWriter(tmp_path, segment=0, worker_id=worker_id)
        writer.write_record(record)
        writer.close()
    reader = soak.SoakReader(tmp_path)
    assert reader.duplicate_committed_ids == [record["game_id"]]
    with pytest.raises(soak.Phase10SoakError):
        soak.seal_soak(tmp_path, expected_games=1)


def test_content_digest_is_partition_independent(tmp_path, source, identity):
    records = [_record(source, identity, ordinal) for ordinal in range(4)]
    one = tmp_path / "one_worker"
    two = tmp_path / "two_workers"
    writer = soak.SoakWriter(one, segment=0, worker_id=0)
    for record in records:
        writer.write_record(record)
    writer.close()
    for worker_id, bucket in enumerate((records[::2], records[1::2])):
        writer = soak.SoakWriter(two, segment=0, worker_id=worker_id)
        for record in bucket:
            writer.write_record(record)
        writer.close()
    assert soak.soak_content_digest(one) == soak.soak_content_digest(two)


def test_seal_freezes_and_refuses_mutation(tmp_path, source, identity):
    writer = soak.SoakWriter(tmp_path, segment=0, worker_id=0)
    writer.write_record(_record(source, identity, 0))
    writer.close()
    with pytest.raises(soak.Phase10SoakError):
        soak.seal_soak(tmp_path, expected_games=2)  # wrong size refuses
    seal = soak.seal_soak(tmp_path, expected_games=1)
    verification = soak.verify_soak_seal(tmp_path)
    assert verification["all_pass"]
    assert verification["seal"]["content_digest"] == seal["content_digest"]
    with pytest.raises(soak.Phase10SoakError):
        soak.SoakWriter(tmp_path, segment=1, worker_id=0)
    with pytest.raises(soak.Phase10SoakError):
        soak.reconcile_soak(tmp_path)
    with pytest.raises(soak.Phase10SoakError):
        soak.seal_soak(tmp_path, expected_games=1)


def test_record_validation_closes_both_field_sets(source, identity):
    record = _record(source, identity, 0)
    grown = dict(record)
    grown["opponent_family"] = "F00"
    with pytest.raises(soak.Phase10SoakError):
        soak.validate_soak_record(grown)
    shrunk = dict(record)
    del shrunk["red_setup_provenance"]
    with pytest.raises(soak.Phase10SoakError):
        soak.validate_soak_record(shrunk)
    wrong_score = dict(record)
    wrong_score["red_score"] = 0.0  # contradicts result red_win
    with pytest.raises(soak.Phase10SoakError):
        soak.validate_soak_record(wrong_score)
    injected = dict(record)
    injected["red_selector_request"] = {
        "split": "train", "color": "red", "selector_seed": 1, "opponent_family": "F00",
    }
    with pytest.raises(soak.Phase10SoakError):
        soak.validate_soak_record(injected)


# ---------------------------------------------------------------------------
# The per-game audit
# ---------------------------------------------------------------------------


def test_verify_soak_game_passes_a_faithful_record(source, identity):
    scheduled = set(soak.soak_game_ids(8))
    for ordinal in range(3):
        record = _record(source, identity, ordinal)
        verdict = soak.verify_soak_game(record, source, scheduled=scheduled)
        assert verdict["ok"], verdict["findings"]
        assert all(value == 0 for value in verdict["counters"].values())


def test_verify_soak_game_catches_tampering(source, identity):
    scheduled = set(soak.soak_game_ids(8))
    record = _record(source, identity, 1)

    swapped_base = json.loads(json.dumps(record))
    other = _record(source, identity, 2)
    swapped_base["red_base_setup_id"] = other["red_base_setup_id"]
    verdict = soak.verify_soak_game(swapped_base, source, scheduled=scheduled)
    assert not verdict["ok"]
    assert verdict["counters"]["determinism_mismatches"] > 0

    wrong_seed = json.loads(json.dumps(record))
    wrong_seed["red_selector_request"]["selector_seed"] += 1
    verdict = soak.verify_soak_game(wrong_seed, source, scheduled=scheduled)
    assert verdict["counters"]["seed_derivation_mismatches"] > 0

    wrong_identity = json.loads(json.dumps(record))
    wrong_identity["candidate_id"] = "P10-E"
    verdict = soak.verify_soak_game(wrong_identity, source, scheduled=scheduled)
    assert verdict["counters"]["selector_identity_mismatches"] > 0

    unscheduled = _record(source, identity, 7)
    verdict = soak.verify_soak_game(unscheduled, source, scheduled=set(soak.soak_game_ids(4)))
    assert verdict["counters"]["unscheduled_game_ids"] > 0


def test_audit_aggregates_and_merge_agrees(source, identity):
    scheduled = set(soak.soak_game_ids(8))
    records = [_record(source, identity, ordinal) for ordinal in range(4)]
    whole = soak.audit_soak_records(records, source=source, scheduled=scheduled)
    assert whole["games_audited"] == 4
    assert all(value == 0 for value in whole["counters"].values())
    assert whole["finding_count"] == 0
    assert sum(whole["family_counts"]["red"].values()) == 4
    assert whole["distinct_final_fingerprints"] == len(set(whole["final_fingerprints"]))

    halves = [
        soak.audit_soak_records(records[:2], source=source, scheduled=scheduled),
        soak.audit_soak_records(records[2:], source=source, scheduled=scheduled),
    ]
    # The shards travel through JSON in the harness; the merge must not care.
    merged = soak.merge_soak_audits(json.loads(json.dumps(halves)))
    assert merged["games_audited"] == whole["games_audited"]
    assert merged["counters"] == whole["counters"]
    assert merged["family_counts"] == whole["family_counts"]
    assert merged["base_ids"] == whole["base_ids"]
    assert merged["result_counts"] == whole["result_counts"]
    assert merged["distinct_final_fingerprints"] == whole["distinct_final_fingerprints"]

    diagnostics = soak.soak_diagnostics(merged, isolation=frozenset())
    assert diagnostics["total_sides"] == 8
    assert diagnostics["phase9_fingerprint_landings"]["landings"] == 0
    for color in ("red", "blue"):
        per_color = diagnostics["per_color"][color]
        assert per_color["draws"] == 4
        assert 0.0 <= per_color["neutral_branch_rate"] <= 1.0
        assert per_color["empirical_vs_exact"]["sampling_noise_expectation"] > 0


def test_storage_resolution_prefers_environment_then_pointer(tmp_path, monkeypatch):
    monkeypatch.setenv(soak.PHASE10_SOAK_ROOT_ENV, str(tmp_path / "env_root"))
    assert soak.default_soak_root() == tmp_path / "env_root"
    description = soak.describe_soak_root()
    assert description["source"] == "environment"
    monkeypatch.delenv(soak.PHASE10_SOAK_ROOT_ENV)
    description = soak.describe_soak_root()
    assert description["source"] in ("pointer_file", "repository_default")
    health = soak.probe_volume_health(tmp_path / "probe_root")
    assert health["usable"] and health["write_probe"]["ok"]
