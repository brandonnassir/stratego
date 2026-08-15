"""Phase 8 Agent 2: the synthetic corpus schedule, determinism and audits.

Specification sources:

- `02_AGENT_2_SYNTHETIC_CORPUS.md` (mission, determinism, split isolation,
  ordered matchup audit, trajectory correctness, setup provenance, PASS gates)
- `00_PHASE_8_SEQUENCE_AND_COMMON_CONTRACT.md` sections 11-13

Crash and resume behaviour lives in `test_corpus_resume.py`; everything here is
about the corpus a completed run is supposed to contain.
"""

from __future__ import annotations

import json

import pytest

from dataclasses import replace

from stratego.engine.legal_moves import legal_actions
from stratego.engine.setup import serialize_setup
from stratego.engine.state import create_game
from stratego.evaluation.registry import ALL_POLICY_IDS
from stratego.training import rule_population as rp
from stratego.training import synthetic_corpus as sc
from stratego.training.corpus_commit import (
    CorpusCommitError,
    CorpusReader,
    corpus_content_digest,
)
from stratego.training.warmstart_contract import (
    SCHEDULE_TOTALS,
    ordered_matchup_cells,
)
from stratego.training.warmstart_seed import (
    CORPUS_SPLITS,
    GAMES_PER_CELL,
    blue_policy_seed,
    parse_synthetic_game_id,
    red_policy_seed,
    setup_root_seed,
    synthetic_game_id,
)

#: A handful of ordered cells that between them cover a policy-weighted teacher,
#: a zero-weight teacher and a stress teacher, on both sides.
SAMPLE_CELLS = (
    ("tactical_rule_based@1.0.0", "basic_heuristic@1.0.0"),
    ("basic_heuristic@1.0.0", "stress_scout_rush@1.0.0"),
    ("stress_chaos@1.0.0", "strategic_rule_based@1.1.0"),
)


def sample_game_ids(split: str = "train", per_cell: int = 1) -> tuple:
    return tuple(
        synthetic_game_id(split, red, blue, ordinal)
        for red, blue in SAMPLE_CELLS
        for ordinal in range(per_cell)
    )


@pytest.fixture(scope="module")
def mini_corpus(tmp_path_factory):
    """One small committed corpus, generated once and reused read-only."""
    root = tmp_path_factory.mktemp("mini_corpus")
    game_ids = sample_game_ids("train", 2) + sample_game_ids("validation", 1)
    sc.generate_corpus(root, worker_count=1, chunks_per_worker=1, game_ids=game_ids)
    return root, game_ids


# ---------------------------------------------------------------------------
# The frozen schedule
# ---------------------------------------------------------------------------


def test_schedule_matches_the_frozen_totals():
    summary = sc.schedule_summary()
    assert summary["cells"] == 100
    assert summary["per_split"] == {"train": 20000, "validation": 4000, "test": 4000}
    assert summary["total"] == SCHEDULE_TOTALS["total"] == 28000


def test_every_scheduled_identity_is_unique_and_parses_back():
    ids = sc.all_scheduled_game_ids()
    assert len(ids) == 28000
    assert len(set(ids)) == 28000
    counts = {}
    for game_id in ids:
        identity = parse_synthetic_game_id(game_id)
        key = (identity["split"], identity["red_token"], identity["blue_token"])
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 3 * 100
    for (split, _red, _blue), count in counts.items():
        assert count == GAMES_PER_CELL[split]


def test_the_schedule_covers_exactly_the_frozen_roster():
    tokens = {
        token
        for cell in ordered_matchup_cells()
        for token in (cell["red_token"], cell["blue_token"])
    }
    assert len(tokens) == len(ALL_POLICY_IDS) == 10
    assert {token.split("@")[0] for token in tokens} == set(ALL_POLICY_IDS)


def test_partition_is_a_permutation_of_its_input():
    ids = sc.all_scheduled_game_ids(("validation",))[:250]
    for chunks in (1, 3, 7, 64):
        buckets = sc.partition_games(ids, chunks)
        flat = [game_id for bucket in buckets for game_id in bucket]
        assert sorted(flat) == sorted(ids)
        assert len(buckets) == min(chunks, len(ids))


# ---------------------------------------------------------------------------
# Determinism: content is a pure function of the game id
# ---------------------------------------------------------------------------


def test_same_game_id_reproduces_setups_actions_and_result():
    game_id = sample_game_ids("train")[0]
    first = rp.play_corpus_game(game_id)
    second = rp.play_corpus_game(game_id)
    assert first.record.red_setup == second.record.red_setup
    assert first.record.blue_setup == second.record.blue_setup
    assert first.record.actions == second.record.actions
    assert first.record.terminal_result == second.record.terminal_result
    assert first.record.terminal_reason == second.record.terminal_reason
    assert first.metadata == second.metadata


def test_different_ordinals_are_different_games():
    red, blue = SAMPLE_CELLS[0]
    first = rp.play_corpus_game(synthetic_game_id("train", red, blue, 0))
    second = rp.play_corpus_game(synthetic_game_id("train", red, blue, 1))
    assert first.record.actions != second.record.actions


def test_per_game_seeds_are_domain_separated():
    game_id = sample_game_ids("train")[0]
    seeds = {
        setup_root_seed(game_id),
        red_policy_seed(game_id),
        blue_policy_seed(game_id),
    }
    assert len(seeds) == 3
    game = rp.play_corpus_game(game_id)
    assert game.metadata["setup_root_seed"] == setup_root_seed(game_id)
    assert game.metadata["red_policy_seed"] == red_policy_seed(game_id)
    assert game.metadata["blue_policy_seed"] == blue_policy_seed(game_id)


def test_a_source_for_the_wrong_split_is_refused():
    game_id = sample_game_ids("validation")[0]
    from stratego.training.warmstart_contract import corpus_setup_source

    with pytest.raises(rp.RulePopulationError, match="setup source"):
        rp.play_corpus_game(game_id, setup_source=corpus_setup_source("train"))


def test_worker_and_enumeration_order_do_not_change_the_corpus(tmp_path):
    game_ids = sample_game_ids("train", 2)
    serial = tmp_path / "serial"
    shuffled = tmp_path / "shuffled"
    sc.generate_corpus(serial, worker_count=1, chunks_per_worker=1, game_ids=game_ids)
    sc.generate_corpus(
        shuffled, worker_count=1, chunks_per_worker=5, game_ids=tuple(reversed(game_ids))
    )
    assert corpus_content_digest(serial, CORPUS_SPLITS) == corpus_content_digest(
        shuffled, CORPUS_SPLITS
    )
    left = CorpusReader(serial, CORPUS_SPLITS)
    right = CorpusReader(shuffled, CORPUS_SPLITS)
    assert left.game_ids() == right.game_ids()
    for game_id in left.game_ids():
        assert left.record(game_id).actions == right.record(game_id).actions
        assert left.metadata(game_id) == right.metadata(game_id)


def test_an_isolated_rebuild_reproduces_a_persisted_game(mini_corpus):
    root, _ = mini_corpus
    reader = CorpusReader(root, CORPUS_SPLITS)
    for game_id in reader.game_ids()[:3]:
        stored_record, stored_metadata = reader.game(game_id)
        rebuilt = rp.play_corpus_game(game_id)
        assert rebuilt.record.actions == stored_record.actions
        assert rebuilt.record.red_setup == stored_record.red_setup
        assert rebuilt.record.blue_setup == stored_record.blue_setup
        assert rebuilt.record.terminal_result == stored_record.terminal_result
        assert rebuilt.metadata == stored_metadata


# ---------------------------------------------------------------------------
# What a stored decision means
# ---------------------------------------------------------------------------


def test_a_stored_decision_is_the_realized_rule_agent_action():
    game_id = sample_game_ids("train")[0]
    game = rp.play_corpus_game(game_id)
    for decision in game.record.decisions:
        chosen = decision.legal_action_ids.index(decision.selected_action_id)
        assert decision.old_probabilities[chosen] == pytest.approx(1.0)
        assert sum(decision.old_probabilities) == pytest.approx(1.0)
        assert (
            sum(1 for value in decision.old_probabilities if value) == 1
        ), "a realized decision is one-hot"
        assert decision.win_draw_loss_prediction == pytest.approx(
            rp.NEUTRAL_VALUE_PREDICTION, abs=1e-6
        )


def test_every_decision_names_the_rule_policy_that_made_it():
    game_id = synthetic_game_id(
        "train", "tactical_rule_based@1.0.0", "basic_heuristic@1.0.0", 3
    )
    game = rp.play_corpus_game(game_id)
    tokens = {decision.collection_policy_version for decision in game.record.decisions}
    assert tokens <= {"tactical_rule_based@1.0.0", "basic_heuristic@1.0.0"}
    for decision in game.record.decisions:
        expected = (
            "tactical_rule_based@1.0.0"
            if decision.acting_player == 0
            else "basic_heuristic@1.0.0"
        )
        assert decision.collection_policy_version == expected
    assert game.metadata["red_policy_weight"] == 1.0
    assert game.metadata["blue_policy_weight"] == 0.5


def test_zero_weight_teachers_still_produce_full_games():
    game_id = synthetic_game_id("train", "random_legal@1.0.0", "stress_chaos@1.0.0", 0)
    game = rp.play_corpus_game(game_id)
    assert game.metadata["red_policy_weight"] == 0.0
    assert game.metadata["blue_policy_weight"] == 0.0
    assert game.total_decisions > 0
    assert game.record.terminal_result in ("red_win", "blue_win", "draw")


def test_the_live_population_still_matches_the_frozen_roster():
    assert rp.verify_live_population() == []
    assert len(rp.roster_digest()) == 64


def test_an_unknown_teacher_token_is_refused():
    with pytest.raises(rp.RulePopulationError, match="teacher roster"):
        rp.teacher_by_token("contract_first_legal@1.0.0")


# ---------------------------------------------------------------------------
# Split isolation
# ---------------------------------------------------------------------------


def test_each_split_samples_only_its_own_setup_bases():
    seen: dict = {}
    for split in CORPUS_SPLITS:
        bases = set()
        for game_id in sample_game_ids(split, 2):
            game = rp.play_corpus_game(game_id)
            provenance = game.metadata["setup_provenance"]
            assert provenance["split"] == split
            for side in ("red", "blue"):
                assert provenance[side]["split"] == split
                bases.add(provenance[side]["base_setup_id"])
        seen[split] = bases
    assert not seen["train"] & seen["validation"]
    assert not seen["train"] & seen["test"]
    assert not seen["validation"] & seen["test"]


def test_split_isolation_audit_flags_a_shared_base(mini_corpus):
    root, _ = mini_corpus
    audit = sc.audit_corpus(root, worker_count=1, observation_plies=1)
    assert all(count == 0 for count in audit["split_isolation"]["base_id_overlaps"].values())
    committed_ids = CorpusReader(root, CORPUS_SPLITS).game_ids()
    assert committed_ids
    contaminated = sc._audit_split_isolation(
        {"train": ["F01:001"], "validation": ["F01:001"]},
        committed_ids,
        ("train", "validation"),
    )
    assert contaminated["problems"]
    assert contaminated["base_id_overlaps"]["train|validation"] == 1


# ---------------------------------------------------------------------------
# Persistence, replay and provenance
# ---------------------------------------------------------------------------


def test_a_committed_game_reads_back_identically(mini_corpus):
    root, game_ids = mini_corpus
    reader = CorpusReader(root, CORPUS_SPLITS)
    assert set(reader.game_ids()) == set(game_ids)
    for game_id in reader.game_ids():
        record, metadata = reader.game(game_id)
        assert record.game_id == game_id == metadata["synthetic_game_id"]
        assert metadata["final_ply"] == record.final_ply
        assert serialize_setup(record.red_setup) == metadata["red_setup"]
        assert rp.validate_game_metadata(metadata, record) == []


def test_the_replay_audit_is_clean_and_checks_every_decision(mini_corpus):
    root, _ = mini_corpus
    audit = sc.audit_corpus(root, worker_count=1, observation_plies=4)
    assert audit["problems"] == []
    assert audit["audited_games"] == audit["committed_games"]
    reader = CorpusReader(root, CORPUS_SPLITS)
    expected = sum(len(reader.record(game_id).decisions) for game_id in reader.game_ids())
    assert audit["replayed_decisions"] == expected
    assert audit["observation_cross_checks"] > 0
    assert audit["full_provenance_rebuilds"] == audit["committed_games"]


def test_the_replay_audit_catches_a_tampered_action(mini_corpus):
    """A record whose action list was edited must not replay clean.

    This is the regression for a real gap: the first implementation replayed the
    *decision* records and never looked at the record's own action list, so an
    edited action list produced a clean replay and only the structural validator
    noticed.
    """
    root, _ = mini_corpus
    reader = CorpusReader(root, CORPUS_SPLITS)
    record = reader.record(reader.game_ids()[0])
    opening = create_game(
        record.red_setup, record.blue_setup, rules=record.rules(), game_id=record.game_id
    )
    actions = list(record.actions)
    actions[0] = next(
        action for action in legal_actions(opening) if action != actions[0]
    )
    assert sc.replay_game(replace(record, actions=tuple(actions)))["problems"]

    # And the same for an action list that is simply the wrong length.
    assert sc.replay_game(replace(record, actions=record.actions[:-1]))["problems"]


def test_illegal_actions_are_counted_not_inferred(mini_corpus):
    """`0 illegal actions` is a PASS gate, so the audit has to count them.

    A record whose stored action is legal nowhere in the position must raise the
    counter, not merely add a sentence to the problem list.
    """
    root, _ = mini_corpus
    reader = CorpusReader(root, CORPUS_SPLITS)
    record = reader.record(reader.game_ids()[0])
    clean = sc.replay_game(record)
    assert clean["illegal_actions"] == 0
    assert clean["legal_set_mismatches"] == 0

    decisions = list(record.decisions)
    first = decisions[0]
    outside = next(
        action for action in range(10000) if action not in first.legal_action_ids
    )
    decisions[0] = replace(
        first,
        legal_action_ids=tuple(sorted(set(first.legal_action_ids) | {outside})),
        old_probabilities=first.old_probabilities + (0.0,),
    )
    actions = list(record.actions)
    actions[0] = outside
    tampered = replace(record, actions=tuple(actions), decisions=tuple(decisions))
    result = sc.replay_game(tampered)
    assert result["illegal_actions"] + result["legal_set_mismatches"] >= 1
    assert result["problems"]


def test_the_replay_audit_catches_a_tampered_result(mini_corpus):
    root, _ = mini_corpus
    reader = CorpusReader(root, CORPUS_SPLITS)
    record = reader.record(reader.game_ids()[0])
    other = "red_win" if record.terminal_result != "red_win" else "blue_win"
    assert sc.replay_game(replace(record, terminal_result=other))["problems"]


def test_provenance_rebuilds_both_setups_for_every_game(mini_corpus):
    root, _ = mini_corpus
    reader = CorpusReader(root, CORPUS_SPLITS)
    for game_id in reader.game_ids():
        record, metadata = reader.game(game_id)
        assert sc.audit_provenance(metadata, record) == []
        assert sc.setup_fingerprints(metadata, record) == []


def test_a_metadata_record_that_disagrees_with_its_trajectory_is_rejected(mini_corpus):
    root, _ = mini_corpus
    reader = CorpusReader(root, CORPUS_SPLITS)
    game_id = reader.game_ids()[0]
    record, metadata = reader.game(game_id)
    tampered = dict(metadata, terminal_result="red_win" if metadata["terminal_result"] != "red_win" else "draw")
    assert rp.validate_game_metadata(tampered, record)
    tampered = dict(metadata, red_policy_weight=0.25)
    assert rp.validate_game_metadata(tampered, record)
    tampered = dict(metadata, setup_root_seed=1)
    assert rp.validate_game_metadata(tampered, record)
    tampered = dict(metadata)
    del tampered["cell_index"]
    assert rp.validate_game_metadata(tampered, record)


def test_writing_a_game_into_the_wrong_split_is_refused(tmp_path):
    from stratego.training.corpus_commit import CorpusWriter

    game = rp.play_corpus_game(sample_game_ids("train")[0])
    writer = CorpusWriter(tmp_path, split="validation", segment=0, worker_id=0)
    try:
        with pytest.raises(CorpusCommitError, match="split"):
            writer.write_game(game)
    finally:
        writer.close()


def test_a_second_writer_may_not_reopen_a_file_set(tmp_path):
    from stratego.training.corpus_commit import CorpusWriter

    writer = CorpusWriter(tmp_path, split="train", segment=0, worker_id=0)
    try:
        with pytest.raises(CorpusCommitError, match="already exists"):
            CorpusWriter(tmp_path, split="train", segment=0, worker_id=0)
    finally:
        writer.close()


# ---------------------------------------------------------------------------
# Matchup accounting and manifest
# ---------------------------------------------------------------------------


def test_matchup_rows_cover_every_cell_and_split(mini_corpus):
    root, game_ids = mini_corpus
    audit = sc.audit_corpus(root, worker_count=1, observation_plies=0)
    rows = sc.matchup_rows(audit)
    assert len(rows) == 300
    assert set(rows[0]) == set(sc.MATCHUP_CSV_COLUMNS)
    populated = [row for row in rows if row["games"]]
    assert sum(row["games"] for row in populated) == len(game_ids)
    for row in populated:
        assert row["red_wins"] + row["blue_wins"] + row["draws"] == row["games"]
        assert row["expected_games"] == GAMES_PER_CELL[row["corpus_split"]]
        assert row["selected_decisions"] <= row["total_decisions"]
        assert row["min_plies"] <= row["mean_plies"] <= row["max_plies"]


def test_the_decision_sampler_caps_selected_decisions_per_game(mini_corpus):
    root, _ = mini_corpus
    from stratego.training.warmstart_seed import (
        MAX_DECISIONS_PER_GAME,
        selected_decision_indices,
    )

    reader = CorpusReader(root, CORPUS_SPLITS)
    for game_id in reader.game_ids():
        total = len(reader.record(game_id).decisions)
        selected = selected_decision_indices(game_id, total)
        assert len(selected) == min(total, MAX_DECISIONS_PER_GAME)
        assert list(selected) == sorted(set(selected))
        assert all(0 <= index < total for index in selected)


def test_finalization_writes_manifests_a_reader_can_verify(mini_corpus, tmp_path):
    root = tmp_path / "finalized"
    game_ids = sample_game_ids("train", 1)
    sc.generate_corpus(root, worker_count=1, chunks_per_worker=1, game_ids=game_ids)
    result = sc.finalize_corpus(root, worker_count=1, observation_plies=2)
    assert result["shard_manifests"] >= 1
    manifest = result["manifest"]
    assert manifest["corpus_version"] == "synthetic_warmstart_corpus_v1"
    assert manifest["content_digest"] == corpus_content_digest(root, CORPUS_SPLITS)
    assert manifest["policy_roster_digest"] == rp.roster_digest()
    assert manifest["free_bytes"] > 0
    assert manifest["storage"]["total_bytes"] > 0

    from stratego.training.shard_writer import directory_summary

    summary = directory_summary(root / "train" / "shards", decode=True)
    assert summary["ok"], summary["problems"]
    assert summary["record_count"] == len(game_ids)
    assert summary["duplicate_game_ids"] == []


def test_completion_gates_reject_an_incomplete_corpus(mini_corpus):
    root, _ = mini_corpus
    audit = sc.audit_corpus(root, worker_count=1, observation_plies=0)
    gates = sc.completion_gates(audit)
    assert gates["replay_and_target_audit_clean"] is True
    assert gates["zero_orphan_trajectories"] is True
    assert gates["zero_duplicate_committed_ids"] is True
    assert gates["upstream_unchanged"] is True
    assert gates["live_population_unchanged"] is True
    # The mini corpus is deliberately a few games, so the schedule gates must
    # fail: a gate set that passed here would pass on an empty corpus too.
    assert gates["scheduled_games_exact"] is False
    assert gates["all_cells_exact"] is False


def test_the_corpus_root_redirect_resolves_in_order(tmp_path, monkeypatch):
    """Environment override, then the pointer file, then the repository default.

    The corpus may legitimately live off the repository volume, so every
    consumer has to be able to find it from one recorded fact rather than by
    assuming a path.
    """
    pointer = sc.repository_root() / sc.CORPUS_ROOT_POINTER
    original = pointer.read_text() if pointer.exists() else None
    monkeypatch.delenv(sc.CORPUS_ROOT_ENV, raising=False)
    try:
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(str(tmp_path / "from_pointer") + "\n")
        assert sc.default_corpus_root() == tmp_path / "from_pointer"
        assert sc.describe_corpus_root()["source"] == "pointer_file"

        monkeypatch.setenv(sc.CORPUS_ROOT_ENV, str(tmp_path / "from_env"))
        assert sc.default_corpus_root() == tmp_path / "from_env"
        assert sc.describe_corpus_root()["source"] == "environment"

        monkeypatch.delenv(sc.CORPUS_ROOT_ENV)
        pointer.unlink()
        assert sc.default_corpus_root() == sc.repository_root() / sc.DEFAULT_CORPUS_ROOT
        assert sc.describe_corpus_root()["source"] == "repository_default"
    finally:
        if original is None:
            pointer.unlink(missing_ok=True)
        else:
            pointer.write_text(original)


def test_reporting_paths_survive_a_corpus_outside_the_repository(tmp_path):
    """A corpus on another volume must still be reportable.

    Regression: the acceptance runner recorded the corpus manifest's location
    with `Path.relative_to(REPOSITORY_ROOT)`, which raises for any root outside
    the repository — so the first finalize run after the corpus was relocated
    crashed while assembling its artifacts, after all the expensive work had
    already succeeded.
    """
    inside = sc.repository_root() / "reports" / "phase_8_data"
    assert sc.repository_relative(inside) == "reports/phase_8_data"

    outside = tmp_path / "elsewhere" / "manifest.json"
    outside.parent.mkdir(parents=True)
    outside.write_text("{}")
    assert sc.repository_relative(outside) == str(outside.resolve())


def test_relocating_a_corpus_preserves_every_digest(tmp_path):
    """Moving the bytes must not change the corpus identity.

    The digests are built from game ids and payload/metadata digests, never
    from shard filenames or paths, so a corpus that moves volumes stays the
    same corpus. This is the property the relocation tool verifies before it
    deletes anything.
    """
    source = tmp_path / "source"
    sc.generate_corpus(
        source, worker_count=1, chunks_per_worker=1, game_ids=sample_game_ids("train", 1)
    )
    before = (
        corpus_content_digest(source, CORPUS_SPLITS),
        sc._metadata_digest(source, CORPUS_SPLITS),
        sc._commit_index_digest(source, CORPUS_SPLITS),
    )

    import shutil

    destination = tmp_path / "elsewhere" / "corpus"
    destination.parent.mkdir(parents=True)
    shutil.copytree(source, destination)
    after = (
        corpus_content_digest(destination, CORPUS_SPLITS),
        sc._metadata_digest(destination, CORPUS_SPLITS),
        sc._commit_index_digest(destination, CORPUS_SPLITS),
    )
    assert before == after

    moved = CorpusReader(destination, CORPUS_SPLITS)
    original = CorpusReader(source, CORPUS_SPLITS)
    assert moved.game_ids() == original.game_ids()
    for game_id in moved.game_ids():
        assert moved.record(game_id).actions == original.record(game_id).actions
        assert moved.metadata(game_id) == original.metadata(game_id)
    audit = sc.audit_corpus(destination, worker_count=1, observation_plies=2)
    assert audit["problems"] == []


def test_the_audit_json_round_trips(mini_corpus):
    root, _ = mini_corpus
    audit = sc.audit_corpus(root, worker_count=1, observation_plies=1)
    assert json.loads(json.dumps(audit)) == json.loads(json.dumps(audit))
