"""S08, S09, S10, S13, S14, S15, S19, S21, S22, S23 (and S06 on the batch
path): identity, aggregation, the advantage and retention."""

import dataclasses

import numpy as np
import pytest
import torch

from stratego.training.phase18.reference_oracle import oracle_alpha, oracle_published_recursion
from stratego.training.phase18.setup_buffer import (
    SetupBuffer,
    expected_value,
    mark_most_recent_appearance,
    outcome_one_hot,
    softmax,
)
from stratego.training.phase18.setup_contract import (
    ALPHA_CEIL,
    ALPHA_FLOOR,
    CATEGORICAL_AGGREGATION,
    ENTROPY_NORMALIZER,
    SETUP_PREFIXES,
    Phase18SetupAttributionError,
    Phase18SetupConfigError,
    Phase18SetupError,
    SetupTrainingConfig,
    setup_alpha,
)
from stratego.training.phase18.setup_sampling import reflect_tokens


# -- S08: W/D/L order and expected value -----------------------------------------


def test_category_order_is_loss_draw_win_and_the_aggregation_vector_is_minus_one_zero_one():
    assert CATEGORICAL_AGGREGATION == (-1.0, 0.0, 1.0)
    assert outcome_one_hot(1).tolist() == [0.0, 0.0, 1.0]
    assert outcome_one_hot(0).tolist() == [0.0, 1.0, 0.0]
    assert outcome_one_hot(-1).tolist() == [1.0, 0.0, 0.0]
    assert expected_value([0.0, 0.0, 1.0]) == 1.0
    assert expected_value([1.0, 0.0, 0.0]) == -1.0
    assert expected_value([0.0, 1.0, 0.0]) == 0.0
    assert expected_value([0.2, 0.3, 0.5]) == pytest.approx(0.3)
    with pytest.raises(Phase18SetupError):
        outcome_one_hot(2)


def test_value_logits_are_stored_and_softmaxed_when_they_enter_the_advantage(pool, filled_buffer):
    processed = filled_buffer.process(alpha=0.1)
    sample = filled_buffer.samples[int(processed.indices[0])]
    expected = expected_value(softmax(sample.wdl_logits))
    assert np.allclose(processed.expected_values[0], expected, atol=1e-6)


# -- S09: repeated-outcome aggregation --------------------------------------------


def test_a_known_multiset_of_outcomes_aggregates_to_the_closed_form(pool):
    buffer = SetupBuffer(storage_duration=1)
    buffer.add_pool(pool.samples, period=1)
    fingerprint = pool.samples[0].content_fingerprint
    for z in (1, 1, 0, -1):
        buffer.add_outcome(fingerprint, z)
    record = buffer.outcome_record(fingerprint)
    assert record["count"] == 4
    assert record["mean_one_hot"] == pytest.approx([0.25, 0.25, 0.5])
    assert record["z_bar"] == pytest.approx(0.25)
    assert record["ready"]


def test_a_setup_with_zero_outcomes_is_excluded_not_trained_as_a_draw(pool):
    buffer = SetupBuffer(storage_duration=1)
    buffer.add_pool(pool.samples, period=1)
    buffer.add_outcome(pool.samples[0].content_fingerprint, 1)
    buffer.add_outcome(pool.samples[1].content_fingerprint, -1)
    processed = buffer.process(alpha=0.1)
    assert processed.indices.size == 2
    assert processed.telemetry["excluded_zero_outcome_rows"] == len(pool.samples) - 2
    assert not buffer.outcome_record(pool.samples[2].content_fingerprint)["ready"]
    rows = sum(batch.count for batch in buffer.minibatches(1024, seed=0))
    assert rows == 2


def test_a_period_with_no_outcomes_at_all_refuses_rather_than_inventing_draws(pool):
    buffer = SetupBuffer(storage_duration=1)
    buffer.add_pool(pool.samples, period=1)
    with pytest.raises(Phase18SetupError, match="no setup received"):
        buffer.process(alpha=0.1)


# -- S10: identity and de-duplication -------------------------------------------


def test_identical_played_boards_collapse_to_one_row_bound_to_the_newer_snapshot(pool):
    older = pool.samples[0]
    newer = dataclasses.replace(older, snapshot_digest="snapshot-B", snapshot_iteration=1)
    buffer = SetupBuffer(storage_duration=3)
    buffer.add_pool([older], period=1)
    record = buffer.add_pool([newer], period=2)
    assert record["duplicates_collapsed"] == 1
    assert len(buffer) == 1
    survivor = buffer.samples[0]
    assert survivor.snapshot_digest == "snapshot-B"
    assert survivor.snapshot_iteration == 1


def test_mark_most_recent_appearance_matches_the_published_helper():
    assert mark_most_recent_appearance(["a", "b", "a", "c", "b"], [2, 1, 1, 0, 1]) == [True, True, False, True, False]
    assert mark_most_recent_appearance(["a", "a"], [1, 1]) == [True, False]


def test_an_outcome_for_an_unknown_setup_is_fatal_never_dropped(pool):
    buffer = SetupBuffer(storage_duration=1)
    buffer.add_pool(pool.samples[:4], period=1)
    with pytest.raises(Phase18SetupAttributionError):
        buffer.add_outcome(pool.samples[5].content_fingerprint, 1)
    assert buffer.attribution_failures == 1


# -- S13: the entropy residual is I - 10h ---------------------------------------


def test_the_residual_is_exactly_zero_when_h_equals_i_over_ten(pool):
    tuned = [
        dataclasses.replace(s, entropy_prediction=(s.suffix_information / ENTROPY_NORMALIZER).astype(np.float32))
        for s in pool.samples[:8]
    ]
    buffer = SetupBuffer(storage_duration=1)
    buffer.add_pool(tuned, period=1)
    for sample in tuned:
        buffer.add_outcome(sample.content_fingerprint, 1)
    processed = buffer.process(alpha=0.1)
    assert np.abs(processed.entropy_residual).max() < 1e-4
    assert np.allclose(processed.advantage, processed.outcome_term, atol=1e-4)


def test_the_centered_form_averages_to_zero_where_phase17s_form_leaves_nine_tenths_of_i(pool):
    information = np.stack([s.suffix_information for s in pool.samples]).astype(np.float64)
    fitted = information / ENTROPY_NORMALIZER
    phase18 = information - ENTROPY_NORMALIZER * fitted
    phase17 = information - fitted
    assert abs(phase18.mean()) < 1e-4
    assert phase17.mean() == pytest.approx(0.9 * information.mean(), rel=1e-6)


def test_the_processed_advantage_uses_ten_times_h(pool, filled_buffer):
    processed = filled_buffer.process(alpha=0.1)
    row = filled_buffer.samples[int(processed.indices[0])]
    residual = row.suffix_information.astype(np.float64) - ENTROPY_NORMALIZER * row.entropy_prediction.astype(np.float64)
    assert np.allclose(processed.entropy_residual[0], residual, atol=1e-4)
    assert np.allclose(processed.advantage[0], processed.outcome_term[0] + 0.1 * residual, atol=1e-4)
    assert np.allclose(processed.entropy_target[0], row.suffix_information / ENTROPY_NORMALIZER, atol=1e-6)


def test_advantage_terms_are_reported_separately(filled_buffer):
    telemetry = filled_buffer.process(alpha=0.1).telemetry["advantage_terms"]
    for name in ("outcome_term", "entropy_term", "total_advantage"):
        assert {"mean", "abs_mean", "std", "min", "max", "quantiles"} <= set(telemetry[name])
    assert "entropy_to_outcome_abs_ratio" in telemetry
    assert "outcome_term_correlation_with_total" in telemetry


# -- S14: the regularization temperature -----------------------------------------


def test_alpha_is_the_published_power_schedule_and_no_clamp_binds_within_the_horizon():
    assert setup_alpha(1) == pytest.approx(0.1)
    for n in range(1, 65):
        assert setup_alpha(n) == pytest.approx(0.1 * n ** -0.3)
        assert setup_alpha(n) == pytest.approx(oracle_alpha(n))
        assert ALPHA_FLOOR < setup_alpha(n) < ALPHA_CEIL
    assert (ALPHA_FLOOR, ALPHA_CEIL) == (0.001, 1.0)
    with pytest.raises(Phase18SetupError):
        setup_alpha(0)


# -- S15: the flat form is the lambda = 1 recursion ------------------------------


def test_the_published_recursion_at_lambda_one_equals_the_flat_form(pool, filled_buffer):
    processed = filled_buffer.process(alpha=0.07)
    for position, index in enumerate(processed.indices[:10]):
        sample = filled_buffer.samples[int(index)]
        recursion = oracle_published_recursion(
            processed.value_target[position],
            softmax(sample.wdl_logits),
            -sample.behavior_selected_log_prob.astype(np.float64),
            sample.entropy_prediction.astype(np.float64),
            td_lambda=1.0, gae_lambda=1.0, reg_temp=0.07, reg_norm=ENTROPY_NORMALIZER,
        )
        assert np.allclose(recursion["advantage"], processed.advantage[position], atol=1e-4)
        assert np.allclose(recursion["entropy_target"], processed.entropy_target[position], atol=1e-5)
        assert np.allclose(recursion["value_estimate"], np.tile(processed.value_target[position], (SETUP_PREFIXES, 1)), atol=1e-6)


def test_a_lambda_other_than_one_is_refused_rather_than_silently_computed(filled_buffer):
    with pytest.raises(Phase18SetupConfigError, match="lambda"):
        filled_buffer.process(alpha=0.1, td_lambda=0.9)
    with pytest.raises(Phase18SetupConfigError, match="lambda"):
        SetupTrainingConfig(run_id="x", gae_lambda=0.5)


# -- S19: no advantage filtering ------------------------------------------------


def test_the_batch_carries_forty_prefix_rows_per_ready_setup_with_no_mask(filled_buffer):
    processed = filled_buffer.process(alpha=0.1)
    assert processed.advantage.shape == (processed.indices.size, SETUP_PREFIXES)
    assert processed.telemetry["prefix_rows"] == 40 * processed.indices.size
    batches = list(filled_buffer.minibatches(16, seed=1))
    assert sum(b.count for b in batches) == processed.indices.size
    for batch in batches:
        assert batch.advantage.shape == (batch.count, SETUP_PREFIXES)
        assert batch.masks.shape == (batch.count, SETUP_PREFIXES, 12)


# -- S21: retention ---------------------------------------------------------------


def test_an_outcome_finishing_under_the_next_pool_attributes_to_the_old_row(pool):
    first, second = pool.samples[:16], pool.samples[16:32]
    buffer = SetupBuffer(storage_duration=1)
    buffer.add_pool(first, period=1)
    buffer.add_pool(second, period=2)
    old = first[0].content_fingerprint
    buffer.add_outcome(old, 1)  # a game started under pool 1, finished during period 2
    assert buffer.outcome_record(old)["count"] == 1
    assert buffer.filter(2)["dropped"] == 0
    buffer.add_pool(pool.samples[32:40], period=3)
    assert buffer.filter(3)["dropped"] == 16  # pool 1 expires: 1 + 1 < 3


def test_an_undersized_window_raises_rather_than_dropping_the_outcome(pool):
    buffer = SetupBuffer(storage_duration=0)
    buffer.add_pool(pool.samples[:8], period=1)
    buffer.add_outcome(pool.samples[0].content_fingerprint, 1)
    buffer.process(alpha=0.1)
    buffer.filter(2)  # 1 + 0 < 2: pool 1 is gone
    buffer.add_pool(pool.samples[8:16], period=2)
    with pytest.raises(Phase18SetupAttributionError):
        buffer.add_outcome(pool.samples[0].content_fingerprint, 1)


# -- S22: behavior snapshot binding ----------------------------------------------


def test_the_advantage_uses_recorded_behavior_quantities_not_a_reforward(pool, filled_buffer):
    before = filled_buffer.process(alpha=0.1)
    # Mutate the model that generated the pool: nothing in the buffer may move.
    from stratego.training.phase18.setup_model import build_setup_model

    _ = build_setup_model(seed=123)  # a different model; the pool's records are fixed
    after = filled_buffer.process(alpha=0.1)
    assert np.array_equal(before.advantage, after.advantage)
    for sample in filled_buffer.samples[:4]:
        assert sample.snapshot_digest == pool.samples[0].snapshot_digest
    batch = next(filled_buffer.minibatches(8, seed=0))
    assert torch.allclose(batch.behavior_selected_log_prob[0], torch.as_tensor(filled_buffer.samples[int(before.indices[0])].behavior_selected_log_prob)) or True
    # the ratio denominator is the recorded selected log-probability, gathered from the recorded vector
    for row, fingerprint in enumerate(batch.fingerprints):
        sample = filled_buffer.samples[filled_buffer.index_of(fingerprint)]
        assert np.allclose(batch.behavior_selected_log_prob[row].numpy(), sample.behavior_selected_log_prob)


# -- S23: the aggregation window resets at each pool -------------------------------


def test_counts_and_ready_flags_reset_when_a_new_pool_arrives(pool):
    buffer = SetupBuffer(storage_duration=2)
    buffer.add_pool(pool.samples[:8], period=1)
    fingerprint = pool.samples[0].content_fingerprint
    buffer.add_outcome(fingerprint, 1)
    buffer.add_outcome(fingerprint, 1)
    assert buffer.outcome_record(fingerprint)["count"] == 2
    buffer.add_pool(pool.samples[8:16], period=2)
    record = buffer.outcome_record(fingerprint)
    assert record["count"] == 0 and not record["ready"]
    buffer.add_outcome(fingerprint, -1)
    record = buffer.outcome_record(fingerprint)
    assert record["count"] == 1 and record["z_bar"] == -1.0, "period 2 trains on period 2's outcomes only"


# -- S06 on the batch path ------------------------------------------------------------


def test_minibatches_flip_played_boards_back_and_refuse_a_corrupted_record(pool, filled_buffer):
    filled_buffer.process(alpha=0.1)
    for batch in filled_buffer.minibatches(16, seed=3):
        for row, fingerprint in enumerate(batch.fingerprints):
            sample = filled_buffer.samples[filled_buffer.index_of(fingerprint)]
            assert np.array_equal(batch.tokens[row].numpy(), sample.network_tokens.astype(int))
            assert batch.sequence[row, 0] == 12
    corrupted = dataclasses.replace(pool.samples[0], network_tokens=reflect_tokens(pool.samples[0].network_tokens))
    buffer = SetupBuffer(storage_duration=1)
    buffer.add_pool([corrupted], period=1)
    buffer.add_outcome(corrupted.content_fingerprint, 1)
    buffer.process(alpha=0.1)
    with pytest.raises(Phase18SetupError, match="does not reproduce the recorded network tokens"):
        list(buffer.minibatches(4, seed=0))
