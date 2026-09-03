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
