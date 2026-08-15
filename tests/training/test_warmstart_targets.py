"""Phase 8 Agent 3: target semantics, baselines, and dataset determinism.

The belief tests exercise both observers thoroughly; the baseline tests pin
the frozen `warmstart_eval_v1` arithmetic; the dataset tests prove the
deterministic universe, the frozen shuffle, the exact resume cursor and
worker-count independence on a real committed mini corpus.
"""

from __future__ import annotations

import numpy as np
import pytest

from stratego.engine.constants import BLUE, PIECE_COUNTS, RED, opponent_of
from stratego.engine.coordinates import to_perspective
from stratego.engine.observation import belief_target
from stratego.model.contract import BELIEF_IGNORE_INDEX
from stratego.training.belief_targets import PIECE_TYPE_INDEX, dense_belief_target
from stratego.training.corpus_commit import CorpusReader
from stratego.training.reconstruction import reconstruct_state
from stratego.training import warmstart_baselines as wb
from stratego.training import warmstart_dataset as wd
from stratego.training.warmstart_seed import (
    CORPUS_SPLITS,
    selected_decision_indices,
    train_order_seed,
)


@pytest.fixture(scope="module")
def mini_reader(warmstart_mini_corpus):
    root, game_ids = warmstart_mini_corpus
    return CorpusReader(root, CORPUS_SPLITS), game_ids


def _sampled_states(reader, game_ids, per_game=4):
    """Reconstructed mid-game states across the mini corpus."""
    states = []
    for game_id in game_ids:
        record, metadata = reader.game(game_id)
        total = len(record.decisions)
        for ply in sorted({total // 5, total // 2, (3 * total) // 4, total - 1}):
            state, _replayed = reconstruct_state(record, ply)
            states.append((record, metadata, ply, state))
            if len(states) % per_game == 0:
                break
    return states


# ---------------------------------------------------------------------------
# Belief targets: hidden-only semantics for both observers
# ---------------------------------------------------------------------------


def test_belief_supervision_covers_exactly_the_unresolved_opponents(mini_reader):
    reader, game_ids = mini_reader
    checked = 0
    for _record, _metadata, _ply, state in _sampled_states(reader, game_ids):
        for observer in (RED, BLUE):
            labels, mask = dense_belief_target(state, observer)
            expected = {}
            for piece in state.pieces:
                if piece.owner == observer or not piece.alive:
                    continue
                if piece.known_to(observer):
                    continue
                expected[to_perspective(piece.current_square, observer)] = piece.true_type
            assert set(np.flatnonzero(mask)) == set(expected)
            for square, true_type in expected.items():
                assert labels[square] == true_type
            assert np.all(labels[~mask] == BELIEF_IGNORE_INDEX)
            assert np.array_equal(mask, labels != BELIEF_IGNORE_INDEX)
            checked += 1
    assert checked >= 20


def test_excluded_squares_are_own_known_empty_or_lake(mini_reader):
    reader, game_ids = mini_reader
    for _record, _metadata, _ply, state in _sampled_states(reader, game_ids)[:6]:
        for observer in (RED, BLUE):
            _labels, mask = dense_belief_target(state, observer)
            occupied = {}
            for piece in state.pieces:
                if piece.alive:
                    occupied[to_perspective(piece.current_square, observer)] = piece
            for square in np.flatnonzero(~mask):
                piece = occupied.get(int(square))
                if piece is None:
                    continue  # empty or lake: nothing to supervise
                assert piece.owner == observer or piece.known_to(observer)


def test_revealed_opponent_pieces_leave_the_supervised_set(mini_reader):
    """Across a game, some opponent piece becomes known and stops being a target."""
    reader, game_ids = mini_reader
    for game_id in game_ids:
        record, _metadata = reader.game(game_id)
        early, _ = reconstruct_state(record, 0)
        late, _ = reconstruct_state(record, len(record.decisions) - 1)
        for observer in (RED, BLUE):
            opponent = opponent_of(observer)
            early_hidden = {
                piece.piece_id
                for piece in early.pieces_of(opponent)
                if piece.alive and not piece.known_to(observer)
            }
            late_known_or_gone = {
                piece.piece_id
                for piece in late.pieces_of(opponent)
                if not piece.alive or piece.known_to(observer)
            }
            resolved = early_hidden & late_known_or_gone
            if resolved:
                _labels, late_mask = dense_belief_target(late, observer)
                late_squares = {
                    to_perspective(piece.current_square, observer)
                    for piece in late.pieces_of(opponent)
                    if piece.alive and piece.piece_id in resolved
                }
                assert not late_squares & set(np.flatnonzero(late_mask))
                return
    pytest.fail("no opponent piece was ever resolved in the mini corpus")


def test_dense_targets_agree_with_the_sparse_engine_authority(mini_reader):
    reader, game_ids = mini_reader
    for _record, _metadata, _ply, state in _sampled_states(reader, game_ids)[:8]:
        for observer in (RED, BLUE):
            labels, mask = dense_belief_target(state, observer)
            sparse = belief_target(state, observer)
            assert int(mask.sum()) == len(sparse)
            for entry in sparse:
                square = to_perspective(int(entry["square"]), observer)
                assert mask[square]
                assert labels[square] == PIECE_TYPE_INDEX[entry["true_type"]]


# ---------------------------------------------------------------------------
# Belief baseline: observable marginal equals the hidden composition
# ---------------------------------------------------------------------------


def test_observable_unresolved_counts_equal_the_hidden_composition(mini_reader):
    """`U_T` from public knowledge must equal the true hidden-type counts.

    Every capture reveals both combatants, so `initial - known` (observable)
    and `count of hidden true types` (privileged) are the same number; the
    baseline leans on that identity and this pins it.
    """
    reader, game_ids = mini_reader
    from stratego.engine.observation import build_observation

    for _record, _metadata, _ply, state in _sampled_states(reader, game_ids):
        for observer in (RED, BLUE):
            observation = build_observation(state, observer)
            observable = wb.unresolved_counts_from_observation(observation)
            hidden = np.zeros(12, dtype=np.int64)
            for piece in state.pieces:
                if (
                    piece.owner != observer
                    and piece.alive
                    and not piece.known_to(observer)
                ):
                    hidden[piece.true_type] += 1
            assert np.array_equal(observable, hidden)


def test_belief_marginal_statistics_are_exact_on_a_hand_case():
    counts = np.zeros(12, dtype=np.int64)
    counts[3] = 3
    counts[7] = 1
    marginal = wb.belief_marginal(counts)
    assert marginal[3] == pytest.approx(0.75)
    assert marginal[7] == pytest.approx(0.25)
    stats = wb.belief_marginal_statistics(counts, [3, 3, 7])
    assert stats["pieces"] == 3
    assert stats["top1_hits"] == 2  # predicted type is 3 (the argmax)
    assert stats["cross_entropy_sum"] == pytest.approx(
        -2 * np.log(0.75) - np.log(0.25)
    )
    with pytest.raises(wb.WarmstartBaselineError):
        wb.belief_marginal(np.zeros(12, dtype=np.int64))


def test_the_marginal_ties_break_toward_the_lowest_type_index():
    counts = np.zeros(12, dtype=np.int64)
    counts[2] = 2
    counts[5] = 2
    stats = wb.belief_marginal_statistics(counts, [5])
    assert stats["top1_hits"] == 0  # the tie resolves to type 2, not 5


# ---------------------------------------------------------------------------
# Value and policy baselines
# ---------------------------------------------------------------------------


def test_the_value_prior_is_the_train_frequency_vector():
    prior = wb.fit_value_prior([6, 3, 1])
    assert prior == (0.6, 0.3, 0.1)
    with pytest.raises(wb.WarmstartBaselineError):
        wb.fit_value_prior([0, 0, 0])


def test_value_prior_metrics_match_hand_arithmetic():
    prior = (0.5, 0.25, 0.25)
    metrics = wb.value_prior_metrics([2, 1, 1], prior)
    expected_ce = -(2 * np.log(0.5) + np.log(0.25) + np.log(0.25)) / 4
    assert metrics["cross_entropy"] == pytest.approx(expected_ce)
    assert metrics["predicted_class"] == "WIN"
    assert metrics["accuracy"] == pytest.approx(0.5)
    # Brier of a constant prediction, averaged over the four one-hot targets.
    brier_win = (0.5 - 1) ** 2 + 0.25**2 + 0.25**2
    brier_other = 0.5**2 + (0.25 - 1) ** 2 + 0.25**2
    assert metrics["brier"] == pytest.approx((2 * brier_win + 2 * brier_other) / 4)


def test_uniform_policy_metrics_weight_by_the_frozen_supervision_weights():
    metrics = wb.uniform_policy_metrics([4, 8], [1.0, 0.5])
    assert metrics["cross_entropy"] == pytest.approx(
        (np.log(4) + 0.5 * np.log(8)) / 1.5
    )
    assert metrics["expected_top1_accuracy"] == pytest.approx(
        (1.0 / 4 + 0.5 / 8) / 1.5
    )
    with pytest.raises(wb.WarmstartBaselineError):
        wb.uniform_policy_metrics([4], [0.0])


def test_the_game_bootstrap_is_seeded_and_reproducible():
    rng = np.random.default_rng(11)
    numerators = rng.uniform(1.0, 2.0, size=40)
    denominators = np.full(40, 2.0)
    first = wb.bootstrap_ratio_interval(numerators, denominators, seed=99, replicates=500)
    second = wb.bootstrap_ratio_interval(numerators, denominators, seed=99, replicates=500)
    other = wb.bootstrap_ratio_interval(numerators, denominators, seed=100, replicates=500)
    assert (first["lower"], first["upper"]) == (second["lower"], second["upper"])
    assert (first["lower"], first["upper"]) != (other["lower"], other["upper"])
    assert first["lower"] <= first["point"] <= first["upper"]


def test_initial_type_counts_are_the_engine_inventory():
    assert wb.INITIAL_TYPE_COUNTS.sum() == 40
    assert list(wb.INITIAL_TYPE_COUNTS) == [
        PIECE_COUNTS[piece_type] for piece_type in range(12)
    ]


# ---------------------------------------------------------------------------
# Universe, shuffle, cursor, worker independence
# ---------------------------------------------------------------------------


def test_the_universe_is_deterministic_and_schedule_ordered(mini_reader):
    reader, game_ids = mini_reader
    universe = wd.selected_example_universe(reader, "train", require_complete=False)
    again = wd.selected_example_universe(reader, "train", require_complete=False)
    assert universe == again
    assert wd.universe_digest(universe) == wd.universe_digest(again)
    per_game = {}
    for game_id, index in universe:
        per_game.setdefault(game_id, []).append(index)
    train_ids = [game_id for game_id in game_ids if "|split=train|" in game_id]
    assert set(per_game) == set(train_ids)
    for game_id, indices in per_game.items():
        total = reader.commits[game_id].total_decisions
        assert tuple(indices) == selected_decision_indices(game_id, total)


def test_an_incomplete_split_fails_loudly_under_the_production_default(mini_reader):
    """The mini corpus misses scheduled games, so strict enumeration refuses.

    Training on a silently shrunken universe would change the run's identity;
    the production default must raise rather than skip.
    """
    reader, _game_ids = mini_reader
    with pytest.raises(wd.WarmstartDatasetError):
        wd.selected_example_universe(reader, "train")


def test_epoch_orders_are_frozen_functions_of_the_epoch():
    first = wd.epoch_order(1000, 0)
    again = wd.epoch_order(1000, 0)
    second = wd.epoch_order(1000, 1)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(1000))
    identity = wd.epoch_order(10, 0, wd.ORDER_SEQUENTIAL)
    assert identity.tolist() == list(range(10))
    # The shuffle stream is the frozen Agent 1 derivation.
    expected = np.random.default_rng(train_order_seed(0)).permutation(1000)
    assert np.array_equal(first, expected)


def _mini_dataset(root):
    return wd.WarmstartDataset(root, record_cache_size=16, require_complete_split=False)


def _train_cursor(batch_size=16):
    return wd.DataCursor(split="train", batch_size=batch_size)


def _collect(dataset, cursor, batches, workers=1):
    collected = []
    for batch, cursor_after, _stats in wd.iter_batches(
        dataset, cursor, batches=batches, workers=workers
    ):
        arrays, metadata = wd.arrays_from_examples(
            dataset.examples(batch.keys)
        )
        collected.append((batch.keys, wd.batch_digest(arrays, metadata), cursor_after))
    return collected


def test_the_cursor_resumes_at_the_exact_next_batch(warmstart_mini_corpus):
    root, _game_ids = warmstart_mini_corpus
    dataset = _mini_dataset(root)
    cursor = _train_cursor()
    full = _collect(dataset, cursor, 5)
    resumed = _collect(dataset, full[1][2], 3)
    assert [entry[0] for entry in resumed] == [entry[0] for entry in full[2:5]]
    assert [entry[1] for entry in resumed] == [entry[1] for entry in full[2:5]]


def test_batches_never_span_an_epoch_boundary(warmstart_mini_corpus):
    root, _game_ids = warmstart_mini_corpus
    dataset = _mini_dataset(root)
    universe = dataset.universe("train")
    batch_size = 32
    cursor = _train_cursor(batch_size)
    per_epoch = -(-len(universe) // batch_size)
    plans = wd.plan_batches(universe, cursor, per_epoch + 1)
    epoch_keys = [keys for _index, keys, _after in plans[:per_epoch]]
    assert sum(len(keys) for keys in epoch_keys) == len(universe)
    assert sorted(key for keys in epoch_keys for key in keys) == sorted(universe)
    assert plans[per_epoch - 1][2].epoch == 1
    assert plans[per_epoch - 1][2].position == 0
    assert len(plans[per_epoch][1]) == min(batch_size, len(universe))


def test_worker_count_and_prefetch_do_not_change_the_batches(warmstart_mini_corpus):
    root, _game_ids = warmstart_mini_corpus
    reference = None
    for workers, prefetch in ((1, 2), (2, 1), (2, 3)):
        dataset = _mini_dataset(root)
        digests = []
        for batch, _cursor, _stats in wd.iter_batches(
            dataset, _train_cursor(), batches=4, workers=workers, prefetch=prefetch
        ):
            arrays, metadata = wd.arrays_from_examples(dataset.examples(batch.keys))
            digests.append(wd.batch_digest(arrays, metadata))
        if reference is None:
            reference = digests
        else:
            assert digests == reference, f"workers={workers} prefetch={prefetch}"


def test_cursor_serialization_round_trips_and_rejects_version_drift():
    cursor = wd.DataCursor(split="train", batch_size=64, epoch=3, position=1280)
    assert wd.DataCursor.from_dict(cursor.to_dict()) == cursor
    stale = dict(cursor.to_dict(), example_version="warmstart_example_v0")
    with pytest.raises(wd.WarmstartDatasetError):
        wd.DataCursor.from_dict(stale)


def test_sequential_iteration_covers_a_split_in_universe_order(warmstart_mini_corpus):
    root, _game_ids = warmstart_mini_corpus
    dataset = _mini_dataset(root)
    universe = dataset.universe("train")
    seen = []
    for batch in dataset.iter_sequential("train", batch_size=48):
        seen.extend(batch.keys)
    assert tuple(seen) == universe
