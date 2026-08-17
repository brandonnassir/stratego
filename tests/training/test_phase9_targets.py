"""Phase 9 Agent 4: same-player targets, the advantage filter, and examples.

What is checked here is the *semantics* a PPO update depends on, not the code
paths that produce them. Every target is recomputed with reference arithmetic
written from the assignment's formulas — `delta`, `A`, `Y`, `tau`, the
standardization — so a tuned constant in the frozen contract would show up as a
disagreement rather than propagate into both sides of a comparison.

Every property has a negative control beside it: a sequence that inserted an
opponent decision, a target built from the wrong perspective, an eligibility
mask taken from the wrong threshold and a stored quantity that drifted from the
payload are all constructed and required to be caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from stratego.engine.constants import BLUE, RED
from stratego.model.contract import BELIEF_IGNORE_INDEX
from stratego.training import phase9_collector as pc
from stratego.training import phase9_contract as contract
from stratego.training import phase9_rollout_store as store
from stratego.training import phase9_targets as targets
from stratego.training.phase9_contract import (
    PHASE9_POPULATION_VERSION,
    PHASE9_ROLLOUT_SCHEDULE_VERSION,
)
from stratego.training.phase9_schedule import rebuild_scheduled_game
from stratego.training.phase9_seed import phase9_game_id, train_order_seed
from stratego.training.reconstruction import iter_reconstructed_decisions

ANCHOR_CHECKPOINT = "checkpoints/phase8/warmstart_c1_v1.pt"
ANCHOR_SHA256 = "f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca"
CONTRACT_DIGEST = "ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34"

#: One game per learner-control mode: `current` trains both colours, and the
#: asymmetric buckets train exactly one, with the colour fixed by the frozen
#: schedule rather than chosen here.
SAMPLE_GAMES = (("current", 0), ("historical", 0), ("historical", 1), ("rule", 0))


@pytest.fixture(scope="module")
def participants():
    resolver = pc.SnapshotResolver(device="cpu", inference_batch_shape=4)
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
def games(participants):
    """`[(record, metadata), ...]` for four real games, played once."""
    played = []
    for bucket, ordinal in SAMPLE_GAMES:
        scheduled = rebuild_scheduled_game(phase9_game_id("canonical", 1, bucket, ordinal))
        runner = pc.play_game(scheduled, participants)
        metadata = store.build_rollout_metadata(
            scheduled,
            runner.record,
            setup_provenance=runner.assignment.provenance,
            behavior_checkpoint_sha256=ANCHOR_SHA256,
            opponent_checkpoint_sha256=(
                ANCHOR_SHA256 if scheduled.opponent_kind == "historical_snapshot" else None
            ),
            learner_decision_count=runner.learner_decision_count,
            population_version=PHASE9_POPULATION_VERSION,
            schedule_version=PHASE9_ROLLOUT_SCHEDULE_VERSION,
            contract_digest=CONTRACT_DIGEST,
        )
        played.append((runner.record, metadata))
    return played


@pytest.fixture(scope="module")
def both_sided(games):
    """The `current vs current` game: the only one that trains both colours."""
    for record, metadata in games:
        if metadata["learner_control"] == "both":
            return record, metadata
    raise AssertionError("the sample contains no both-sided game")


@pytest.fixture(scope="module")
def one_sided(games):
    for record, metadata in games:
        if metadata["learner_control"] in ("red", "blue"):
            return record, metadata
    raise AssertionError("the sample contains no one-sided game")


# ---------------------------------------------------------------------------
# Reference arithmetic — written from the assignment, not from the module
# ---------------------------------------------------------------------------

_ONE_HOT = {"win": (1.0, 0.0, 0.0), "draw": (0.0, 1.0, 0.0), "loss": (0.0, 0.0, 1.0)}


def reference_targets(predictions, outcome):
    z = {"win": 1, "draw": 0, "loss": -1}[outcome]
    values = [float(item[0]) - float(item[2]) for item in predictions]
    deltas = [
        (values[index + 1] - values[index]) if index + 1 < len(values) else (z - values[index])
        for index in range(len(values))
    ]
    advantages = [0.0] * len(values)
    following = 0.0
    for index in reversed(range(len(values))):
        advantages[index] = deltas[index] + 0.5 * following
        following = advantages[index]
    wdl = [None] * len(values)
    wdl[-1] = _ONE_HOT[outcome]
    for index in reversed(range(len(values) - 1)):
        wdl[index] = tuple(
            0.2 * float(predictions[index + 1][component]) + 0.8 * float(wdl[index + 1][component])
            for component in range(3)
        )
    return values, deltas, advantages, wdl


# ---------------------------------------------------------------------------
# Same-player extraction
# ---------------------------------------------------------------------------


def test_learner_players_follows_learner_control():
    assert targets.learner_players({"learner_control": "both", "learner_color": None}) == (
        RED,
        BLUE,
    )
    assert targets.learner_players({"learner_control": "red", "learner_color": "red"}) == (RED,)
    assert targets.learner_players({"learner_control": "blue", "learner_color": "blue"}) == (BLUE,)


@pytest.mark.parametrize(
    "metadata",
    [
        {"learner_control": "both", "learner_color": "red"},
        {"learner_control": "red", "learner_color": "blue"},
        {"learner_control": "neither", "learner_color": None},
    ],
)
def test_a_contradictory_learner_designation_is_refused(metadata):
    """The two fields answer the same question and may never disagree."""
    with pytest.raises(targets.Phase9TargetError):
        targets.learner_players(metadata)


def test_a_sequence_holds_only_that_players_own_decisions(both_sided):
    record, metadata = both_sided
    sequences = targets.build_sequences(record, metadata)
    assert set(sequences) == {RED, BLUE}
    for player, sequence in sequences.items():
        assert all(
            record.decisions[ply].acting_player == player for ply in sequence.plies
        ), "an opponent decision entered a learner sequence"
        assert list(sequence.plies) == sorted(sequence.plies), "game order was not preserved"


def test_the_two_sequences_of_a_both_sided_game_partition_the_decisions(both_sided):
    record, metadata = both_sided
    sequences = targets.build_sequences(record, metadata)
    plies = [ply for sequence in sequences.values() for ply in sequence.plies]
    assert sorted(plies) == list(range(len(record.decisions)))
    assert len(set(plies)) == len(plies), "a ply landed in two sequences"


def test_an_asymmetric_game_trains_only_the_current_policy_side(one_sided):
    record, metadata = one_sided
    sequences = targets.build_sequences(record, metadata)
    assert len(sequences) == 1
    (player,) = sequences
    assert targets.learner_side_name(player) == metadata["learner_control"]
    opponent_plies = {
        ply
        for ply, decision in enumerate(record.decisions)
        if decision.acting_player != player
    }
    assert opponent_plies, "the opponent made no move; the test proves nothing"
    assert not opponent_plies & set(sequences[player].plies)


def test_consecutive_entries_are_consecutive_turns_of_one_player(games):
    """Nothing of that player's is skipped between two sequence entries."""
    for record, metadata in games:
        for player, sequence in targets.build_sequences(record, metadata).items():
            for earlier, later in zip(sequence.plies, sequence.plies[1:]):
                between = [
                    ply
                    for ply in range(earlier + 1, later)
                    if record.decisions[ply].acting_player == player
                ]
                assert not between, f"{record.game_id}: skipped {between}"


def test_the_sequence_ends_at_that_players_last_turn(games):
    """Terminal may arrive before that player's next turn; that is the last entry."""
    for record, metadata in games:
        for player, sequence in targets.build_sequences(record, metadata).items():
            own = [
                ply
                for ply, decision in enumerate(record.decisions)
                if decision.acting_player == player
            ]
            assert sequence.plies[-1] == own[-1]


def test_the_stored_learner_quantity_matches_the_payload(games):
    for record, metadata in games:
        sequences = targets.build_sequences(record, metadata)
        assert not targets.verify_learner_decision_count(record, metadata, sequences)


def test_a_drifted_learner_count_is_caught(both_sided):
    """Negative control: the collector's bookkeeping is checked, not trusted."""
    record, metadata = both_sided
    sequences = targets.build_sequences(record, metadata)
    drifted = dict(metadata, learner_decision_count=metadata["learner_decision_count"] + 1)
    assert targets.verify_learner_decision_count(record, drifted, sequences)


# ---------------------------------------------------------------------------
# Scalar value, advantage and W/D/L target
# ---------------------------------------------------------------------------


def test_the_scalar_behavior_value_is_win_minus_loss(games):
    for record, metadata in games:
        for player, sequence in targets.build_sequences(record, metadata).items():
            for prediction, value in zip(sequence.predictions, sequence.values):
                assert value == pytest.approx(prediction[0] - prediction[2], abs=1e-12)


def test_stored_behavior_wdl_is_a_normalized_simplex(games):
    for record, metadata in games:
        for sequence in targets.build_sequences(record, metadata).values():
            for prediction in sequence.predictions:
                assert all(np.isfinite(prediction))
                assert sum(prediction) == pytest.approx(1.0, abs=targets.SIMPLEX_TOLERANCE)


def test_advantages_match_independent_reference_arithmetic(games):
    for record, metadata in games:
        for player, sequence in targets.build_sequences(record, metadata).items():
            outcome = targets.terminal_outcome(record.terminal_result, player)
            values, deltas, advantages, _ = reference_targets(sequence.predictions, outcome)
            assert sequence.values == pytest.approx(values, abs=1e-12)
            assert sequence.deltas == pytest.approx(deltas, abs=1e-12)
            assert sequence.advantages == pytest.approx(advantages, abs=1e-12)


def test_the_final_delta_uses_the_terminal_outcome(games):
    """The game ends before that player's next turn: `delta = z - v`."""
    for record, metadata in games:
        for player, sequence in targets.build_sequences(record, metadata).items():
            assert sequence.deltas[-1] == pytest.approx(
                sequence.z - sequence.values[-1], abs=1e-12
            )


def test_wdl_targets_match_independent_reference_arithmetic(games):
    for record, metadata in games:
        for player, sequence in targets.build_sequences(record, metadata).items():
            outcome = targets.terminal_outcome(record.terminal_result, player)
            _values, _deltas, _advantages, wdl = reference_targets(
                sequence.predictions, outcome
            )
            for mine, theirs in zip(sequence.wdl_targets, wdl):
                assert mine == pytest.approx(theirs, abs=1e-12)


def test_every_wdl_target_is_a_finite_nonnegative_simplex(games):
    for record, metadata in games:
        for sequence in targets.build_sequences(record, metadata).values():
            for target in sequence.wdl_targets:
                assert all(np.isfinite(target))
                assert min(target) >= -targets.SIMPLEX_TOLERANCE
                assert sum(target) == pytest.approx(1.0, abs=targets.SIMPLEX_TOLERANCE)


def test_the_terminal_target_is_the_one_hot_outcome(games):
    for record, metadata in games:
        for player, sequence in targets.build_sequences(record, metadata).items():
            outcome = targets.terminal_outcome(record.terminal_result, player)
            assert sequence.wdl_targets[-1] == pytest.approx(_ONE_HOT[outcome], abs=0.0)


def test_the_two_colours_of_a_decided_game_hold_opposite_perspectives(both_sided):
    record, metadata = both_sided
    if record.terminal_result == "draw":
        pytest.skip("a drawn game has the same outcome from both perspectives")
    sequences = targets.build_sequences(record, metadata)
    assert sequences[RED].z == -sequences[BLUE].z


def test_a_target_built_from_the_wrong_perspective_is_visible(both_sided):
    """Negative control: a reversed perspective is still a valid simplex."""
    record, metadata = both_sided
    if record.terminal_result == "draw":
        pytest.skip("a drawn game has the same outcome from both perspectives")
    sequences = targets.build_sequences(record, metadata)
    red = sequences[RED]
    wrong = targets.build_sequence(record, metadata, BLUE)
    assert red.wdl_targets[-1] != wrong.wdl_targets[-1]
    # Still a normalized simplex — which is exactly why only a recomputation
    # from the learner's own perspective can notice the reversal.
    assert all(
        sum(target) == pytest.approx(1.0, abs=targets.SIMPLEX_TOLERANCE)
        for target in wrong.wdl_targets
    )


def test_no_sign_flip_is_applied_at_an_opponent_turn(both_sided):
    """Consecutive deltas are differences of the *same* player's own values."""
    record, metadata = both_sided
    sequence = targets.build_sequences(record, metadata)[RED]
    for index in range(len(sequence) - 1):
        assert sequence.deltas[index] == pytest.approx(
            sequence.values[index + 1] - sequence.values[index], abs=1e-12
        )


def test_an_unknown_terminal_result_is_refused():
    with pytest.raises(targets.Phase9TargetError):
        targets.terminal_outcome("abandoned", RED)


# ---------------------------------------------------------------------------
# The per-iteration advantage filter
# ---------------------------------------------------------------------------


def _statistics(values):
    return targets.iteration_statistics(
        {("g", index): value for index, value in enumerate(values)},
        namespace="canonical",
        iteration=1,
        sealed_rollout_digest="digest",
        games=1,
    )


def test_the_threshold_is_the_linear_q75_of_absolute_advantages():
    values = [0.9, -0.4, 0.15, -0.02, 0.6, -0.85, 0.33, 0.07]
    statistics = _statistics(values)
    expected = float(np.quantile(np.abs(values), 0.75, method="linear"))
    assert statistics.threshold == pytest.approx(expected, abs=1e-12)
    assert statistics.threshold >= contract.ADVANTAGE_FILTER_FLOOR


def test_the_floor_binds_when_every_advantage_is_tiny():
    statistics = _statistics([0.001, -0.002, 0.0005, 0.0])
    assert statistics.threshold == contract.ADVANTAGE_FILTER_FLOOR
    assert statistics.eligible == 0
    assert statistics.no_eligible


def test_eligibility_is_exactly_absolute_advantage_at_or_above_tau():
    values = [0.9, -0.4, 0.15, -0.02, 0.6, -0.85, 0.33, 0.07]
    statistics = _statistics(values)
    for value in values:
        assert statistics.is_eligible(value) == (abs(value) >= statistics.threshold)


def test_standardization_uses_only_the_eligible_subset():
    values = [0.9, -0.4, 0.15, -0.02, 0.6, -0.85, 0.33, 0.07]
    statistics = _statistics(values)
    eligible = [value for value in values if abs(value) >= statistics.threshold]
    assert statistics.mean_eligible == pytest.approx(float(np.mean(eligible)), abs=1e-12)
    assert statistics.std_eligible == pytest.approx(float(np.std(eligible)), abs=1e-12)
    for value in eligible:
        assert statistics.standardize(value) == pytest.approx(
            (value - np.mean(eligible)) / (np.std(eligible) + 1e-8), abs=1e-9
        )


def test_standardizing_over_everything_would_give_a_different_answer():
    """Negative control: the "eligible subset only" rule is load-bearing."""
    values = [0.9, -0.4, 0.15, -0.02, 0.6, -0.85, 0.33, 0.07]
    statistics = _statistics(values)
    whole = (values[0] - np.mean(values)) / (np.std(values) + 1e-8)
    assert statistics.standardize(values[0]) != pytest.approx(whole, abs=1e-6)


def test_zero_variance_is_handled_explicitly():
    """Every eligible advantage identical: standardized to 0, flagged, no NaN."""
    statistics = _statistics([0.5, 0.5, 0.5, 0.5])
    assert statistics.zero_variance
    assert statistics.std_eligible == 0.0
    assert statistics.standardize(0.5) == 0.0
    assert np.isfinite(statistics.standardize(0.5))


def test_an_empty_eligible_subset_is_handled_explicitly():
    statistics = _statistics([0.0, 0.0, 0.0])
    assert statistics.no_eligible
    assert statistics.standardize(0.0) == 0.0
    assert statistics.retention_fraction == 0.0


def test_filtering_an_iteration_with_no_learner_decisions_is_refused():
    with pytest.raises(targets.Phase9TargetError):
        _statistics([])


def test_value_and_belief_populations_are_not_filtered(games):
    """The filter touches PPO eligibility only; every example still exists."""
    record, metadata = games[0]
    sequences = targets.build_sequences(record, metadata)
    advantages = {
        (record.game_id, ply): sequence.advantages[index]
        for sequence in sequences.values()
        for index, ply in enumerate(sequence.plies)
    }
    statistics = targets.iteration_statistics(
        advantages,
        namespace="canonical",
        iteration=1,
        sealed_rollout_digest="digest",
        games=1,
    )
    built = list(targets.examples_for_game(record, metadata, statistics, sequences))
    assert len(built) == len(advantages)
    assert 0 < sum(example.ppo_eligible for example in built) < len(built)


# ---------------------------------------------------------------------------
# The example
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built(games):
    """`(record, metadata, statistics, examples)` for one game, built once."""
    record, metadata = games[0]
    sequences = targets.build_sequences(record, metadata)
    advantages = {
        (record.game_id, ply): sequence.advantages[index]
        for sequence in sequences.values()
        for index, ply in enumerate(sequence.plies)
    }
    statistics = targets.iteration_statistics(
        advantages,
        namespace="canonical",
        iteration=1,
        sealed_rollout_digest="digest",
        games=1,
    )
    examples = list(targets.examples_for_game(record, metadata, statistics, sequences))
    return record, metadata, statistics, sequences, examples


def test_every_learner_decision_becomes_exactly_one_example(built):
    record, metadata, _statistics, sequences, examples = built
    expected = sum(len(sequence) for sequence in sequences.values())
    assert len(examples) == expected
    assert len({example.key for example in examples}) == expected


def test_examples_arrive_in_ascending_ply_order(built):
    *_ignored, examples = built
    indices = [example.decision_index for example in examples]
    assert indices == sorted(indices)


def test_the_behavior_probability_is_the_stored_realized_entry(built):
    record, _metadata, _statistics, _sequences, examples = built
    for example in examples:
        decision = record.decisions[example.decision_index]
        index = list(decision.legal_action_ids).index(decision.selected_action_id)
        assert example.behavior_action_probability == float(decision.old_probabilities[index])
        assert example.behavior_action_logprob == pytest.approx(
            float(np.log(max(example.behavior_action_probability, 1e-12))), abs=1e-12
        )


def test_the_behavior_distribution_is_carried_in_stored_absolute_order(built):
    record, *_ignored, examples = built
    for example in examples:
        decision = record.decisions[example.decision_index]
        assert example.behavior_legal_actions == tuple(decision.legal_action_ids)
        assert example.behavior_legal_probabilities == tuple(decision.old_probabilities)
        assert list(example.behavior_legal_actions) == sorted(example.behavior_legal_actions)


def test_a_zero_probability_realized_action_still_floors_its_log():
    """The PPO ratio's denominator never becomes an infinity."""
    assert targets.behavior_action_logprob(0.0) == pytest.approx(float(np.log(1e-12)))


def test_the_legal_mask_is_the_model_frame_of_the_engine_legal_set(built):
    record, *_ignored, examples = built
    for example in examples:
        decision = record.decisions[example.decision_index]
        assert int(example.legal_mask.sum()) == len(decision.legal_action_ids)
        assert example.legal_mask[example.sampled_action_model]


def test_belief_labels_are_hidden_only_and_masked_consistently(built):
    *_ignored, examples = built
    for example in examples:
        assert np.array_equal(
            example.belief_mask, example.belief_target != BELIEF_IGNORE_INDEX
        )
        assert example.belief_target.dtype == np.int64
        assert example.belief_mask.dtype == bool
    assert any(example.supervised_belief_squares > 0 for example in examples)


def test_every_built_example_audits_clean(built):
    record, metadata, statistics, sequences, _examples = built
    by_ply = {
        int(ply): sequence for sequence in sequences.values() for ply in sequence.plies
    }
    for rebuilt in iter_reconstructed_decisions(
        record, sorted(by_ply)[:24], dense_mask=True, include_public_knowledge=False
    ):
        sequence = by_ply[rebuilt.ply]
        example = targets.build_example(record, metadata, rebuilt, sequence, statistics)
        assert not targets.audit_example(
            example, record, metadata, rebuilt, sequence, statistics
        )


def test_the_audit_catches_a_target_that_drifted(built):
    """Negative control: an audit that cannot fail proves nothing."""
    record, metadata, statistics, sequences, _examples = built
    by_ply = {
        int(ply): sequence for sequence in sequences.values() for ply in sequence.plies
    }
    rebuilt = next(
        iter_reconstructed_decisions(
            record, [sorted(by_ply)[0]], dense_mask=True, include_public_knowledge=False
        )
    )
    sequence = by_ply[rebuilt.ply]
    example = targets.build_example(record, metadata, rebuilt, sequence, statistics)
    from dataclasses import replace

    for corrupted in (
        replace(example, advantage=example.advantage + 1.0),
        replace(example, ppo_eligible=not example.ppo_eligible),
        replace(example, standardized_advantage=example.standardized_advantage + 1.0),
        replace(example, behavior_action_probability=0.5 * example.behavior_action_probability),
        replace(example, belief_mask=np.zeros_like(example.belief_mask)),
    ):
        assert targets.audit_example(
            corrupted, record, metadata, rebuilt, sequence, statistics
        )


def test_building_an_opponent_decision_as_a_learner_step_is_refused(one_sided):
    """The opponent's decisions stay in the trajectory and out of the loss."""
    record, metadata = one_sided
    sequences = targets.build_sequences(record, metadata)
    (player,) = sequences
    opponent_ply = next(
        ply
        for ply, decision in enumerate(record.decisions)
        if decision.acting_player != player
    )
    rebuilt = next(
        iter_reconstructed_decisions(
            record, [opponent_ply], dense_mask=True, include_public_knowledge=False
        )
    )
    with pytest.raises(targets.Phase9TargetError):
        targets.build_example(record, metadata, rebuilt, sequences[player], None)


# ---------------------------------------------------------------------------
# The batch boundary
# ---------------------------------------------------------------------------


def test_only_the_observation_reaches_the_backbone(built):
    *_ignored, examples = built
    payload = targets.model_input_fields_only(examples[0])
    assert set(payload) == {"observation"}
    batch = targets.build_batch(examples[:4])
    assert set(batch["model_input"]) == {"observation"}
    assert batch["model_input"]["observation"].shape == (4, 127, 10, 10)
    assert "belief_target" in batch["loss_inputs"]
    assert "belief_target" not in batch["model_input"]


def test_a_batch_keeps_every_loss_input_aligned(built):
    *_ignored, examples = built
    batch = targets.build_batch(examples[:8])
    assert batch["size"] == 8
    assert batch["loss_inputs"]["advantage"].shape == (8,)
    assert batch["loss_inputs"]["wdl_target"].shape == (8, 3)
    assert batch["loss_inputs"]["belief_target"].shape == (8, 100)
    assert batch["identity"]["game_id"][0] == examples[0].game_id


def test_an_empty_batch_is_refused():
    with pytest.raises(targets.Phase9TargetError):
        targets.build_batch([])


# ---------------------------------------------------------------------------
# The rollout-to-example iterator handed to Agent 5
# ---------------------------------------------------------------------------


@pytest.fixture
def written(tmp_path, games):
    """A small rollout on disk, read back through the store's own reader."""
    writer = store.Phase9RolloutWriter(tmp_path, namespace="canonical", iteration=1, worker_id=0)
    try:
        for record, metadata in games:
            writer.write_game(record, metadata)
    finally:
        writer.close()
    return store.Phase9RolloutReader(tmp_path, "canonical", 1)


def test_the_iterator_walks_the_rollout_deterministically(written, games):
    """Games ascending by id, decisions ascending by ply, nothing else."""
    advantages, sequences_by_game, problems = targets.collect_iteration_advantages(written)
    assert not problems
    statistics = targets.iteration_statistics(
        advantages,
        namespace="canonical",
        iteration=1,
        sealed_rollout_digest="digest",
        games=len(written),
    )
    keys = [
        (example.game_id, example.decision_index)
        for example in targets.iter_rollout_examples(
            written, statistics, sequences_by_game=sequences_by_game
        )
    ]
    assert len(keys) == len(advantages)
    assert keys == sorted(keys), "the example stream is not in (game_id, ply) order"
    assert keys == [
        (example.game_id, example.decision_index)
        for example in targets.iter_rollout_examples(written, statistics)
    ], "two passes over the same rollout produced different streams"


def test_the_train_order_universe_is_every_learner_decision(written):
    advantages, _sequences, _problems = targets.collect_iteration_advantages(written)
    keys = targets.train_order_keys(written)
    assert keys == tuple(sorted(advantages))
    assert len(set(keys)) == len(keys)


def test_the_iterator_carries_the_rollout_identity_onto_every_example(written):
    advantages, sequences_by_game, _problems = targets.collect_iteration_advantages(written)
    statistics = targets.iteration_statistics(
        advantages,
        namespace="canonical",
        iteration=1,
        sealed_rollout_digest="sealed-digest",
        games=len(written),
    )
    for example in targets.iter_rollout_examples(
        written, statistics, sequences_by_game=sequences_by_game
    ):
        assert example.rollout_id == targets.rollout_identity("canonical", 1)
        assert example.sealed_rollout_digest == "sealed-digest"
        assert example.behavior_checkpoint_sha256 == ANCHOR_SHA256


# ---------------------------------------------------------------------------
# Train order and the resumable cursor
# ---------------------------------------------------------------------------


KEYS = tuple(("g%02d" % (index // 10), index % 10) for index in range(1300))


def test_the_epoch_order_is_a_reproducible_permutation():
    first = targets.epoch_order(KEYS, "canonical", 1, 0)
    again = targets.epoch_order(KEYS, "canonical", 1, 0)
    assert first == again
    assert sorted(first) == list(range(len(KEYS)))


def test_each_epoch_draws_its_own_order():
    assert targets.epoch_order(KEYS, "canonical", 1, 0) != targets.epoch_order(
        KEYS, "canonical", 1, 1
    )


def test_the_order_stream_is_domain_separated_by_run_and_iteration():
    assert targets.epoch_order(KEYS, "canonical", 1, 0) != targets.epoch_order(
        KEYS, "pilot_p9a", 1, 0
    )
    assert targets.epoch_order(KEYS, "canonical", 1, 0) != targets.epoch_order(
        KEYS, "canonical", 2, 0
    )
    assert train_order_seed("canonical", 1, 0) != train_order_seed("pilot_p9a", 1, 0)


def test_the_final_partial_minibatch_is_consumed():
    slices = targets.minibatch_slices(len(KEYS))
    assert sum(stop - start for start, stop in slices) == len(KEYS)
    assert slices[-1][1] - slices[-1][0] == len(KEYS) % contract.MINIBATCH_SIZE
    assert slices[-1][1] == len(KEYS)


def test_a_cursor_resumes_the_exact_interrupted_minibatch():
    cursor = targets.Phase9MinibatchCursor.start(
        namespace="canonical",
        iteration=1,
        sealed_rollout_digest="digest",
        total_examples=len(KEYS),
        epochs=2,
    )
    consumed = []
    for _ in range(2):
        start, stop = targets.minibatch_slices(len(KEYS))[cursor.minibatch_index]
        consumed.append(
            targets.minibatch_keys(KEYS, "canonical", 1, cursor.epoch, cursor.minibatch_index)
        )
        cursor = cursor.advance(stop - start)
    resumed = targets.minibatch_keys(
        KEYS, "canonical", 1, cursor.epoch, cursor.minibatch_index
    )
    order = targets.epoch_order(KEYS, "canonical", 1, 0)
    start, stop = targets.minibatch_slices(len(KEYS))[2]
    assert resumed == tuple(KEYS[position] for position in order[start:stop])
    assert cursor.examples_consumed == 2 * contract.MINIBATCH_SIZE
    assert set(consumed[0]) & set(resumed) == set()


def test_the_cursor_rolls_over_into_the_next_epoch():
    cursor = targets.Phase9MinibatchCursor.start(
        namespace="canonical",
        iteration=1,
        sealed_rollout_digest="digest",
        total_examples=len(KEYS),
        epochs=2,
    )
    for _ in range(cursor.minibatches_per_epoch):
        cursor = cursor.advance(1)
    assert cursor.epoch == 1
    assert cursor.minibatch_index == 0
    assert not cursor.finished
    for _ in range(cursor.minibatches_per_epoch):
        cursor = cursor.advance(1)
    assert cursor.finished


def test_an_epoch_covers_every_example_exactly_once():
    seen: list = []
    for index in range(len(targets.minibatch_slices(len(KEYS)))):
        seen.extend(targets.minibatch_keys(KEYS, "canonical", 1, 0, index))
    assert sorted(seen) == sorted(KEYS)


def test_a_minibatch_outside_the_epoch_is_refused():
    with pytest.raises(targets.Phase9TargetError):
        targets.minibatch_keys(KEYS, "canonical", 1, 0, 999)


# ---------------------------------------------------------------------------
# The published contract
# ---------------------------------------------------------------------------


def test_the_example_contract_quotes_the_frozen_constants():
    document = targets.example_contract()
    assert document["constants"]["gamma"] == contract.GAMMA == 1.0
    assert document["constants"]["lambda_A"] == contract.LAMBDA_ADVANTAGE == 0.5
    assert document["constants"]["lambda_V"] == contract.LAMBDA_VALUE == 0.8
    assert document["constants"]["filter_quantile"] == contract.ADVANTAGE_FILTER_QUANTILE
    assert document["constants"]["filter_floor"] == contract.ADVANTAGE_FILTER_FLOOR
    assert document["constants"]["minibatch_size"] == contract.MINIBATCH_SIZE


def test_the_contract_names_one_model_input_and_marks_belief_a_target():
    document = targets.example_contract()
    assert document["model_input_fields"] == ["observation"]
    assert document["fields"]["observation"]["role"] == "model_input"
    assert document["fields"]["belief_target"]["role"] == "loss_input"
    assert document["fields"]["legal_mask"]["role"] == "masking"


def test_the_contract_digest_is_stable_and_content_addressed():
    first = targets.example_contract_digest()
    assert first == targets.example_contract_digest()
    assert len(first) == 64


def test_the_example_schema_carries_no_privileged_field():
    example_fields = set(targets.Phase9RLExample.__dataclass_fields__)
    assert not example_fields & set(targets.FORBIDDEN_EXAMPLE_FIELDS)
    assert set(targets.MODEL_INPUT_FIELDS) <= example_fields
