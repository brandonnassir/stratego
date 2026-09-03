"""Stage 6B: the live trajectory store commits, finalises, refuses to be
appended to, enumerates a frozen universe and rebuilds the accepted examples."""

import json

import numpy as np
import pytest

from stratego.training.phase18 import g3_live_store as ls
from stratego.training.phase18.g3_contract import Phase18G3Error
from stratego.training.warmstart_dataset import WarmstartDataset
from stratego.training.warmstart_examples import examples_for_game
from stratego.training.warmstart_seed import selected_decision_indices

NAMESPACE = "phase18_g3_live_store_test_v1"


@pytest.fixture(scope="module")
def train_games(warmstart_mini_corpus):
    root, game_ids = warmstart_mini_corpus
    dataset = WarmstartDataset(root, require_complete_split=False)
    games = []
    for game_id in game_ids:
        record, metadata = dataset.game(game_id)
        if metadata["corpus_split"] == "train":
            games.append((record, dict(metadata)))
    assert len(games) == 3
    return games


def _write_period(root, period, games):
    writer = ls.LivePeriodWriter(root, period=period, namespace=NAMESPACE, lineage="candidate", run_id="G3-TEST")
    commits = [writer.write(record, metadata) for record, metadata in games]
    summary = writer.close()
    return commits, summary


def test_a_period_commits_finalises_and_verifies(tmp_path, train_games):
    commits, summary = _write_period(tmp_path, 1, train_games)
    assert summary["games"] == 3 and summary["period"] == 1
    reader = ls.LiveRecordReader(tmp_path)
    assert reader.periods() == (1,)
    assert reader.verify_period(1)["verified"]
    journal = reader.commits(1)
    assert list(journal) == [record.game_id for record, _ in train_games]
    for (record, _metadata), commit in zip(train_games, commits):
        assert journal[record.game_id] == commit
        assert commit.total_decisions == len(record.decisions)
        assert reader.period_of(record.game_id) == 1
        rebuilt = reader.record(1, record.game_id)
        assert rebuilt.actions == record.actions and rebuilt.game_id == record.game_id


def test_the_selected_decisions_follow_the_accepted_sampler_shape(train_games):
    for record, _ in train_games:
        total = len(record.decisions)
        live = ls.live_selected_decision_indices(NAMESPACE, record.game_id, total)
        accepted = selected_decision_indices(record.game_id, total)
        assert len(live) == len(accepted)
        assert list(live) == sorted(set(live))
        if total <= 64:
            assert live == accepted
        else:
            # Same bins, different (namespace-derived) draws inside them.
            from stratego.training.warmstart_seed import decision_bin_bounds

            for index, (low, high) in zip(live, decision_bin_bounds(total)):
                assert low <= index < high
    assert ls.live_selected_decision_indices(NAMESPACE, "x", 0) == ()
    assert ls.live_selected_decision_indices("other_namespace", train_games[0][0].game_id, 500) != ls.live_selected_decision_indices(NAMESPACE, train_games[0][0].game_id, 500)


def test_examples_reproduce_the_accepted_builder_exactly(tmp_path, train_games):
    _write_period(tmp_path, 1, train_games)
    reader = ls.LiveRecordReader(tmp_path)
    universe = reader.universe([1])
    assert len(universe) == sum(len(c.selected_decisions) for c in reader.commits(1).values())
    keys = universe[::7][:12]
    built = reader.examples(keys)
    by_game = {}
    for record, metadata in train_games:
        by_game[record.game_id] = (record, metadata)
    for (game_id, index), example in zip(keys, built):
        record, metadata = by_game[game_id]
        direct = next(examples_for_game(record, metadata, (index,)))
        assert example.key == direct.key == (game_id, index)
        assert np.array_equal(example.observation, direct.observation)
        assert np.array_equal(example.belief_target, direct.belief_target)
        assert example.policy_action_model == direct.policy_action_model
        assert example.corpus_split == "train"
    assert ls.universe_digest(universe) == ls.universe_digest(ls.LiveRecordReader(tmp_path).universe([1]))


def test_a_period_is_never_appended_to_and_later_periods_can_be_discarded(tmp_path, train_games):
    _write_period(tmp_path, 1, train_games[:1])
    with pytest.raises(Phase18G3Error, match="never appended"):
        ls.LivePeriodWriter(tmp_path, period=1, namespace=NAMESPACE, lineage="candidate", run_id="G3-TEST")
    _write_period(tmp_path, 2, train_games[1:2])
    _write_period(tmp_path, 3, train_games[2:3])
    assert ls.available_periods(tmp_path) == (1, 2, 3)
    renamed = ls.discard_periods_after(tmp_path, 1)
    assert len(renamed) == 8 and ls.available_periods(tmp_path) == (1,)
    assert all(".orphaned" in entry["to"] for entry in renamed)
    # The orphaned period is unreadable, the kept one intact.
    reader = ls.LiveRecordReader(tmp_path)
    with pytest.raises(Phase18G3Error, match="not finalised"):
        reader.commits(2)
    assert reader.verify_period(1)["verified"]


def test_an_unfinalised_period_is_invisible_and_a_wrong_split_is_refused(tmp_path, train_games):
    writer = ls.LivePeriodWriter(tmp_path, period=1, namespace=NAMESPACE, lineage="control", run_id="G3-TEST")
    record, metadata = train_games[0]
    writer.write(record, metadata)
    assert ls.available_periods(tmp_path) == ()
    with pytest.raises(Phase18G3Error, match="training split"):
        writer.write(record, dict(metadata, corpus_split="validation"))
    writer.close()
    assert ls.available_periods(tmp_path) == (1,)
    done = json.loads((tmp_path / "period_0001.done.json").read_text())
    assert done["games"] == 1 and set(done["files"]) == {"period_0001.records", "period_0001.meta.jsonl", "period_0001.journal.jsonl"}


# ---------------------------------------------------------------------------
# The period-1 gate correction: the lineage-neutral semantic comparison
# ---------------------------------------------------------------------------


def _two_lineage_stores(root, train_games, *, mutate_control=None):
    """Both lineages commit the same games; the control's metadata may be mutated."""
    stores = {}
    writers = {
        lineage: ls.LivePeriodWriter(root / lineage, period=1, namespace=NAMESPACE, lineage=lineage, run_id="G3-TEST")
        for lineage in ("candidate", "control")
    }
    for index, (record, metadata) in enumerate(train_games):
        for lineage, writer in writers.items():
            stamped = dict(metadata, lineage=lineage)
            if lineage == "control" and mutate_control is not None:
                stamped = mutate_control(index, stamped)
            writer.write(record, stamped)
    for lineage, writer in writers.items():
        stores[lineage] = writer.close()
    return stores


def test_the_semantic_comparison_passes_when_only_the_lineage_stamp_differs(tmp_path, train_games):
    summaries = _two_lineage_stores(tmp_path, train_games)
    # The raw digests hash the metadata line, which carries the lineage: unequal by construction.
    assert summaries["candidate"]["commit_digest"] != summaries["control"]["commit_digest"]
    assert summaries["candidate"]["games"] == summaries["control"]["games"] == 3
    result = ls.compare_live_periods(tmp_path / "candidate", tmp_path / "control", 1, lineage_a="candidate", lineage_b="control")
    assert result["matched"], result["problems"]
    assert result["problems"] == []
    assert result["raw_commit_digest"] == {
        "candidate": summaries["candidate"]["commit_digest"],
        "control": summaries["control"]["commit_digest"],
    }
    assert result["raw_commit_digest_equal"] is False
    assert "lineage stamp" in result["raw_commit_digest_note"]
    assert all(result["file_integrity"][lineage]["verified"] for lineage in ("candidate", "control"))
    semantic = result["semantic"]
    assert semantic["matched"] and all(semantic["checks"].values()), semantic["checks"]
    assert semantic["commits"] == {"candidate": 3, "control": 3}
    assert semantic["selected_examples"]["candidate"] == semantic["selected_examples"]["control"] == summaries["candidate"]["selected_examples"]
    assert semantic["lineage_neutral_commit_digest"]["candidate"] == semantic["lineage_neutral_commit_digest"]["control"]
    assert semantic["metadata_field_differences"] == []
    assert "'lineage'" in semantic["permitted_difference"]
    # The neutral metadata drops exactly one field, and the stamps read as expected.
    reader_a, reader_b = ls.LiveRecordReader(tmp_path / "candidate"), ls.LiveRecordReader(tmp_path / "control")
    for game_id in reader_a.commits(1):
        a, b = reader_a.metadata(1, game_id), reader_b.metadata(1, game_id)
        assert a != b and (a["lineage"], b["lineage"]) == ("candidate", "control")
        assert ls.lineage_neutral_metadata(a) == ls.lineage_neutral_metadata(b)
        assert set(a) - set(ls.lineage_neutral_metadata(a)) == {"lineage"}
        assert ls.lineage_neutral_metadata_sha256(a) == ls.lineage_neutral_metadata_sha256(b)


def test_the_semantic_comparison_fails_on_a_non_lineage_metadata_field_on_disk(tmp_path, train_games):
    def bump_blue_seed(index, metadata):
        return dict(metadata, blue_policy_seed=int(metadata["blue_policy_seed"]) + 1) if index == 1 else metadata

    _two_lineage_stores(tmp_path, train_games, mutate_control=bump_blue_seed)
    result = ls.compare_live_periods(tmp_path / "candidate", tmp_path / "control", 1, lineage_a="candidate", lineage_b="control")
    assert not result["matched"]
    assert all(result["file_integrity"][lineage]["verified"] for lineage in ("candidate", "control"))
    semantic = result["semantic"]
    assert semantic["checks"]["neutral_metadata"] is False
    assert semantic["checks"]["trajectory_digests"] and semantic["checks"]["game_ids_and_order"] and semantic["checks"]["lineage_stamps"]
    assert semantic["checks"]["lineage_neutral_commit_digest"] is False
    assert semantic["metadata_field_differences"] == [{"game_id": train_games[1][0].game_id, "fields": ["blue_policy_seed"]}]
    assert any("blue_policy_seed" in problem for problem in result["problems"])


def test_the_semantic_comparison_fails_on_every_non_lineage_difference(tmp_path, train_games):
    import copy

    _two_lineage_stores(tmp_path, train_games)
    entries_a = ls.semantic_period_entries(ls.LiveRecordReader(tmp_path / "candidate"), 1)
    entries_b = ls.semantic_period_entries(ls.LiveRecordReader(tmp_path / "control"), 1)
    baseline = ls.compare_period_semantics(entries_a, entries_b, lineage_a="candidate", lineage_b="control")
    assert baseline["matched"] and baseline["problems"] == []

    def mutated(change):
        entries = copy.deepcopy(entries_b)
        change(entries)
        return ls.compare_period_semantics(entries_a, entries, lineage_a="candidate", lineage_b="control")

    # A changed non-lineage metadata field.
    result = mutated(lambda e: e[0]["neutral_metadata"].__setitem__("period_completed", 99))
    assert not result["matched"] and result["checks"]["neutral_metadata"] is False
    assert result["metadata_field_differences"][0]["fields"] == ["period_completed"]
    # An added metadata field.
    result = mutated(lambda e: e[2]["neutral_metadata"].__setitem__("extra", 1))
    assert not result["matched"] and result["metadata_field_differences"][0]["fields"] == ["extra"]
    # A removed metadata field.
    result = mutated(lambda e: e[2]["neutral_metadata"].pop("final_ply"))
    assert not result["matched"] and result["metadata_field_differences"][0]["fields"] == ["final_ply"]
    # A different trajectory under the same game id.
    result = mutated(lambda e: e[1].__setitem__("trajectory_sha256", "0" * 64))
    assert not result["matched"] and result["checks"]["trajectory_digests"] is False
    assert any("trajectory digests differ" in p for p in result["problems"])
    # A different selected-decision list.
    result = mutated(lambda e: e[1].__setitem__("selected_decisions", tuple(e[1]["selected_decisions"][:-1])))
    assert not result["matched"] and result["checks"]["selected_decisions"] is False
    # The same games in a different order.
    result = mutated(lambda e: e.reverse())
    assert not result["matched"] and result["checks"]["game_ids_and_order"] is False
    assert any("different order" in p for p in result["problems"])
    # A missing commit.
    result = mutated(lambda e: e.pop())
    assert not result["matched"] and result["checks"]["commit_count"] is False and result["checks"]["game_ids_and_order"] is False
    # A different game id.
    result = mutated(lambda e: e[0].__setitem__("game_id", "other"))
    assert not result["matched"] and result["checks"]["game_ids_and_order"] is False
    # A wrong or missing lineage stamp is not a permitted difference.
    result = mutated(lambda e: e[0].__setitem__("lineage", "candidate"))
    assert not result["matched"] and result["checks"]["lineage_stamps"] is False
    result = mutated(lambda e: e[0].__setitem__("lineage", None))
    assert not result["matched"] and result["checks"]["lineage_stamps"] is False
    # Two stores under the same label are not a two-lineage comparison.
    result = ls.compare_period_semantics(entries_a, entries_a, lineage_a="candidate", lineage_b="candidate")
    assert not result["matched"] and result["checks"]["lineage_labels_differ"] is False
    # Every other check in the baseline is still true (no check is vacuous).
    assert set(baseline["checks"]) == {
        "lineage_stamps",
        "lineage_labels_differ",
        "commit_count",
        "game_ids_and_order",
        "trajectory_digests",
        "selected_decisions",
        "neutral_metadata",
        "lineage_neutral_commit_digest",
    }


def test_the_raw_digest_still_audits_the_files(tmp_path, train_games):
    """The raw commit digest keeps its file-integrity job: a tampered store fails
    the comparison through `verify_period`, whatever the semantics say."""
    _two_lineage_stores(tmp_path, train_games)
    path = tmp_path / "control" / "period_0001.meta.jsonl"
    lines = path.read_text().splitlines()
    lines[0] = lines[0].replace('"lineage":"control"', '"lineage":"candidate"')
    path.write_text("\n".join(lines) + "\n")
    result = ls.compare_live_periods(tmp_path / "candidate", tmp_path / "control", 1, lineage_a="candidate", lineage_b="control")
    assert not result["matched"]
    assert result["file_integrity"]["control"]["verified"] is False
    assert any("do not verify" in p for p in result["problems"])
