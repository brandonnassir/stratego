"""Compact trajectory storage tests (Phase 3 Agent 3).

These run at a small deterministic scale so they stay part of the ordinary
pytest run. The full-scale gate -- at least 1,000,000 reconstructed historical
decisions and the snapshot-interval benchmark -- lives in
`scripts/run_phase3_agent03.py`.

What is asserted here is the storage contract: a record round-trips exactly, it
never carries an observation tensor, its decisions are sparse and internally
consistent, and the codec refuses anything it does not fully understand rather
than dropping a field.
"""

import numpy as np
import pytest

from stratego.engine.constants import (
    IMPLEMENTATION_VERSION,
    OBSERVATION_SHAPE,
    OBSERVATION_VERSION,
    PIECES_PER_PLAYER,
    RULES_VERSION,
    TRAINING_RULES,
)
from stratego.engine.replay import rebuild_final_state
from stratego.engine.snapshot import create_snapshot, restore_snapshot
from stratego.engine.state import state_fingerprint
from stratego.training.batch_simulation import BatchSimulator
from stratego.training.serialization import ByteReader, ByteWriter, CodecError, StringTable
from stratego.training.trajectory import (
    DEFAULT_SNAPSHOT_INTERVAL,
    SUPPORTED_SNAPSHOT_INTERVALS,
    SYNTHETIC_POLICY_VERSION,
    TRAJECTORY_VERSION,
    DecisionRecord,
    GameRecord,
    GameTrajectoryBuilder,
    TrajectoryError,
    collect_games,
    decode_game_record,
    decode_game_record_compressed,
    decode_snapshot,
    encode_game_record,
    encode_game_record_compressed,
    encode_snapshot,
    select_action_from_policy,
    setup_id,
    synthetic_policy,
    synthetic_value,
    validate_decision_record,
    validate_game_record,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def collect(
    games: int = 2,
    *,
    root_seed: int = 4001,
    environments: int = 4,
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
) -> "list[GameRecord]":
    """A few complete games from a deterministically seeded batch."""
    simulator = BatchSimulator(environments, root_seed=root_seed, rules=TRAINING_RULES)
    return list(
        collect_games(simulator, games=games, snapshot_interval=snapshot_interval)
    )


@pytest.fixture(scope="module")
def records() -> "list[GameRecord]":
    return collect(3)


@pytest.fixture(scope="module")
def record(records) -> GameRecord:
    return records[0]


# ---------------------------------------------------------------------------
# Byte-level primitives
# ---------------------------------------------------------------------------


def test_varints_round_trip():
    writer = ByteWriter()
    values = [0, 1, 127, 128, 300, 9_999, 2**31, 2**48]
    for value in values:
        writer.uvarint(value)
    reader = ByteReader(writer.to_bytes())
    assert [reader.uvarint() for _ in values] == values
    reader.expect_exhausted()


def test_signed_varints_round_trip():
    writer = ByteWriter()
    values = [0, -1, 1, -128, 128, -9_999, 9_999]
    for value in values:
        writer.svarint(value)
    reader = ByteReader(writer.to_bytes())
    assert [reader.svarint() for _ in values] == values


def test_uvarint_rejects_negative():
    with pytest.raises(CodecError):
        ByteWriter().uvarint(-1)


def test_ascending_sequence_rejects_unsorted():
    with pytest.raises(CodecError):
        ByteWriter().ascending_uvarints([5, 3])


def test_ascending_sequence_round_trips_as_deltas():
    values = tuple(range(0, 9_000, 37))
    writer = ByteWriter()
    writer.ascending_uvarints(values)
    assert ByteReader(writer.to_bytes()).ascending_uvarints() == values


def test_string_table_interns_and_resolves():
    table = StringTable()
    first = table.intern("own_piece")
    assert table.intern("own_piece") == first
    assert table.intern(None) == 0
    assert table.resolve(first) == "own_piece"
    assert table.resolve(0) is None


def test_reader_reports_trailing_bytes():
    reader = ByteReader(b"\x01\x02")
    reader.uvarint()
    with pytest.raises(CodecError):
        reader.expect_exhausted()


# ---------------------------------------------------------------------------
# Snapshot codec
# ---------------------------------------------------------------------------


def test_snapshot_round_trips_to_an_identical_state(record):
    context = record.context()
    for entry in record.snapshots:
        decoded = decode_snapshot(entry.payload, context)
        restored = restore_snapshot(decoded)
        assert restored.total_moves == entry.ply
        # Rebuild the same ply the long way and compare the full fingerprint.
        expected = rebuild_final_state(
            _record_truncated_to(record, entry.ply).to_replay_record()
        )
        assert state_fingerprint(restored, include_history=False) == state_fingerprint(
            expected, include_history=False
        )


def _record_truncated_to(record: GameRecord, ply: int) -> GameRecord:
    """A copy of `record` whose action list stops at `ply`, for replay checks."""
    from dataclasses import replace

    return replace(
        record,
        actions=record.actions[:ply],
        final_ply=ply,
        decisions=record.decisions[:ply],
    )


def test_snapshot_codec_rejects_a_foreign_game(record):
    from dataclasses import replace as dataclass_replace

    context = record.context()
    other = dataclass_replace(context, game_id="not-this-game")
    snapshot = restore_snapshot(decode_snapshot(record.snapshots[0].payload, context))
    with pytest.raises(TrajectoryError, match="does not belong"):
        encode_snapshot(create_snapshot(snapshot), other)


def test_snapshot_codec_rejects_an_unknown_field(record):
    context = record.context()
    snapshot = create_snapshot(
        restore_snapshot(decode_snapshot(record.snapshots[0].payload, context))
    )
    snapshot["something_new"] = 1
    with pytest.raises(TrajectoryError, match="no longer matches"):
        encode_snapshot(snapshot, context)


def test_snapshot_codec_rejects_a_history_bearing_snapshot(record):
    """`include_history=True` adds fields the compact codec does not carry."""
    context = record.context()
    state = restore_snapshot(decode_snapshot(record.snapshots[0].payload, context))
    with pytest.raises(TrajectoryError, match="no longer matches"):
        encode_snapshot(create_snapshot(state, include_history=True), context)


def test_snapshot_codec_rejects_a_setup_disagreement(record):
    from dataclasses import replace as dataclass_replace

    context = record.context()
    state = restore_snapshot(decode_snapshot(record.snapshots[0].payload, context))
    swapped = dataclass_replace(context, red_setup=record.blue_setup)
    with pytest.raises(TrajectoryError, match="true type|starting square"):
        encode_snapshot(create_snapshot(state), swapped)


# ---------------------------------------------------------------------------
# Game record codec
# ---------------------------------------------------------------------------


def test_game_record_round_trips_exactly(records):
    for record in records:
        assert decode_game_record(encode_game_record(record)) == record


def test_compressed_round_trip_matches(records):
    for record in records:
        assert decode_game_record_compressed(encode_game_record_compressed(record)) == record


def test_compression_reduces_size(record):
    assert len(encode_game_record_compressed(record)) < len(encode_game_record(record))


def test_decoder_rejects_foreign_bytes():
    with pytest.raises(CodecError, match="bad magic"):
        decode_game_record(b"NOPE" + b"\x01")


def test_decoder_rejects_a_future_format_version(record):
    payload = bytearray(encode_game_record(record))
    payload[4] = 99
    with pytest.raises(CodecError, match="unsupported trajectory format"):
        decode_game_record(bytes(payload))


def test_record_is_versioned(record):
    assert record.trajectory_version == TRAJECTORY_VERSION
    assert record.rules_version == RULES_VERSION
    assert record.observation_version == OBSERVATION_VERSION
    assert record.implementation_version == IMPLEMENTATION_VERSION


def test_record_carries_the_required_game_fields(record):
    assert record.game_id
    assert record.environment_id >= 0
    assert record.generation >= 0
    assert len(record.red_setup) == PIECES_PER_PLAYER
    assert len(record.blue_setup) == PIECES_PER_PLAYER
    assert record.first_player in ("red", "blue")
    assert record.setup_family
    assert record.setup_id == setup_id(record.red_setup, record.blue_setup)
    assert record.terminal_result in ("red_win", "blue_win", "draw")
    assert record.terminal_reason
    assert record.final_ply == len(record.actions)
    assert record.collection_policy_version == SYNTHETIC_POLICY_VERSION
    assert record.snapshot_interval == DEFAULT_SNAPSHOT_INTERVAL


def test_record_agrees_with_the_frozen_replay_schema(record):
    """The compact record and the frozen replay record describe one game."""
    replay = record.to_replay_record()
    final = rebuild_final_state(replay)
    assert final.terminal
    assert final.total_moves == record.final_ply
    assert final.terminal_reason == record.terminal_reason


def test_record_validates_clean(records):
    for record in records:
        assert validate_game_record(record) == []


# ---------------------------------------------------------------------------
# Storage principle: no dense tensors, no dense masks
# ---------------------------------------------------------------------------


def test_no_observation_tensor_is_stored(record):
    """A decision must not carry anything the size of an observation."""
    observation_floats = int(np.prod(OBSERVATION_SHAPE))
    for decision in record.decisions:
        assert len(decision.legal_action_ids) < observation_floats
        assert len(decision.old_probabilities) == len(decision.legal_action_ids)
        for value in vars(decision).values():
            assert not isinstance(value, np.ndarray)


def test_decision_storage_is_sparse_not_dense(record):
    """Legal sets are far smaller than the 10,000-entry action space."""
    for decision in record.decisions:
        assert 0 < len(decision.legal_action_ids) < 1_000


def test_bytes_per_decision_stay_far_below_a_dense_vector(record):
    """A dense 10,000 `float32` vector is 40,000 bytes; a record is not."""
    encoded = len(encode_game_record(record))
    assert encoded / max(len(record.decisions), 1) < 40_000 / 10


def test_snapshot_cadence_is_configurable():
    for interval in SUPPORTED_SNAPSHOT_INTERVALS:
        record = collect(1, root_seed=77, snapshot_interval=interval)[0]
        assert record.snapshot_interval == interval
        assert [entry.ply for entry in record.snapshots] == list(
            range(0, record.final_ply + 1, interval)
        )[: len(record.snapshots)]
        assert all(entry.ply % interval == 0 for entry in record.snapshots)
        assert validate_game_record(record) == []


def test_shorter_interval_costs_more_snapshot_bytes():
    dense = collect(1, root_seed=99, snapshot_interval=16)[0]
    sparse = collect(1, root_seed=99, snapshot_interval=64)[0]
    assert dense.final_ply == sparse.final_ply
    assert len(dense.snapshots) > len(sparse.snapshots)
    assert dense.snapshot_bytes > sparse.snapshot_bytes


# ---------------------------------------------------------------------------
# Sparse decision-storage checks
# ---------------------------------------------------------------------------


def test_legal_ids_are_unique_and_ascending(record):
    for decision in record.decisions:
        ids = decision.legal_action_ids
        assert len(set(ids)) == len(ids)
        assert list(ids) == sorted(ids)


def test_probabilities_match_the_legal_set_one_for_one(record):
    for decision in record.decisions:
        assert len(decision.old_probabilities) == len(decision.legal_action_ids)
        assert all(np.isfinite(decision.old_probabilities))
        assert all(value >= 0.0 for value in decision.old_probabilities)
        assert sum(decision.old_probabilities) == pytest.approx(1.0, abs=1e-4)


def test_selected_action_is_legal_and_matches_the_action_list(record):
    for index, decision in enumerate(record.decisions):
        assert decision.selected_action_id in decision.legal_action_ids
        assert decision.selected_action_id == record.actions[index]
        assert decision.legal_action_ids[decision.selected_action_index] == (
            decision.selected_action_id
        )


def test_value_prediction_has_three_normalised_finite_entries(record):
    for decision in record.decisions:
        value = decision.win_draw_loss_prediction
        assert len(value) == 3
        assert all(np.isfinite(value))
        assert sum(value) == pytest.approx(1.0, abs=1e-4)


def test_collection_policy_version_is_preserved_through_the_codec(record):
    decoded = decode_game_record(encode_game_record(record))
    for original, rebuilt in zip(record.decisions, decoded.decisions):
        assert rebuilt.collection_policy_version == original.collection_policy_version
        assert rebuilt.collection_policy_version == SYNTHETIC_POLICY_VERSION


def test_snapshot_reference_points_at_or_before_the_decision(record):
    for decision in record.decisions:
        entry = record.snapshots[decision.snapshot_reference]
        assert entry.ply <= decision.ply
        assert decision.snapshot_reference == record.snapshot_index_for_ply(decision.ply)


def test_probabilities_are_stored_at_float32_so_the_record_is_exact(record):
    """A stored probability decodes to exactly the value the record holds."""
    decoded = decode_game_record(encode_game_record(record))
    for original, rebuilt in zip(record.decisions, decoded.decisions):
        assert rebuilt.old_probabilities == original.old_probabilities
        assert rebuilt.win_draw_loss_prediction == original.win_draw_loss_prediction


# ---------------------------------------------------------------------------
# Validation surface
# ---------------------------------------------------------------------------


def _decision(**overrides) -> DecisionRecord:
    base = dict(
        game_id="g",
        ply=0,
        acting_player=0,
        selected_action_id=101,
        legal_action_ids=(101, 202),
        old_probabilities=(0.25, 0.75),
        win_draw_loss_prediction=(0.2, 0.3, 0.5),
        collection_policy_version="policy_v1",
        snapshot_reference=0,
    )
    base.update(overrides)
    return DecisionRecord(**base)


def test_validation_accepts_a_well_formed_decision():
    assert validate_decision_record(_decision()) == []


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"legal_action_ids": (202, 101)}, "ascending"),
        ({"legal_action_ids": (101, 101)}, "duplicate"),
        ({"old_probabilities": (1.0,)}, "probabilities for"),
        ({"old_probabilities": (0.5, 0.9)}, "sum to"),
        ({"old_probabilities": (-0.1, 1.1)}, "negative"),
        ({"old_probabilities": (float("nan"), 1.0)}, "non-finite"),
        ({"selected_action_id": 999}, "is not legal"),
        ({"collection_policy_version": ""}, "policy version"),
        ({"win_draw_loss_prediction": (0.5, 0.5)}, "value prediction has"),
        ({"win_draw_loss_prediction": (0.5, 0.5, 0.5)}, "value probabilities sum"),
        ({"acting_player": 7}, "acting player"),
        ({"legal_action_ids": (101, 99_999)}, "outside the action space"),
    ],
)
def test_validation_rejects_broken_decisions(overrides, expected):
    problems = validate_decision_record(_decision(**overrides))
    assert any(expected in problem for problem in problems), problems


def test_game_validation_catches_a_tampered_action_list(record):
    from dataclasses import replace as dataclass_replace

    tampered = dataclass_replace(
        record, actions=(record.actions[0] + 1,) + record.actions[1:]
    )
    assert any("disagrees with" in problem for problem in validate_game_record(tampered))


def test_game_validation_catches_a_missing_first_snapshot(record):
    from dataclasses import replace as dataclass_replace

    tampered = dataclass_replace(record, snapshots=record.snapshots[1:])
    assert any("ply 0" in problem for problem in validate_game_record(tampered))


# ---------------------------------------------------------------------------
# Deterministic synthetic model outputs
# ---------------------------------------------------------------------------


def test_synthetic_policy_is_deterministic_and_normalised():
    legal = (11, 22, 33, 44)
    first = synthetic_policy("game", 7, legal)
    assert first == synthetic_policy("game", 7, legal)
    assert len(first) == len(legal)
    assert sum(first) == pytest.approx(1.0)
    assert all(value > 0.0 for value in first)


def test_synthetic_policy_depends_on_the_position():
    legal = (11, 22, 33)
    assert synthetic_policy("game", 7, legal) != synthetic_policy("game", 8, legal)
    assert synthetic_policy("a", 7, legal) != synthetic_policy("b", 7, legal)


def test_synthetic_policy_rejects_an_empty_legal_set():
    with pytest.raises(TrajectoryError):
        synthetic_policy("game", 0, ())


def test_synthetic_value_is_deterministic_and_normalised():
    value = synthetic_value("game", 3)
    assert value == synthetic_value("game", 3)
    assert len(value) == 3
    assert sum(value) == pytest.approx(1.0)


def test_action_selection_is_deterministic_and_legal():
    legal = (11, 22, 33, 44)
    probabilities = synthetic_policy("game", 5, legal)
    chosen = select_action_from_policy("game", 5, legal, probabilities)
    assert chosen == select_action_from_policy("game", 5, legal, probabilities)
    assert chosen in legal


def test_collection_is_reproducible_from_the_seeds():
    first = collect(2, root_seed=515)
    second = collect(2, root_seed=515)
    assert [record.actions for record in first] == [record.actions for record in second]
    assert [encode_game_record(record) for record in first] == [
        encode_game_record(record) for record in second
    ]


# ---------------------------------------------------------------------------
# Builder contract
# ---------------------------------------------------------------------------


def test_builder_rejects_out_of_order_plies():
    simulator = BatchSimulator(1, root_seed=808)
    builder = GameTrajectoryBuilder(
        game_id=simulator.game_id(0),
        environment_id=simulator.environment_id(0),
        generation=simulator.generation(0),
        red_setup=simulator.setups(0)[0],
        blue_setup=simulator.setups(0)[1],
        rules=simulator.rules,
    )
    state = simulator.game_state(0)
    legal = simulator.legal_actions(0)
    probabilities = synthetic_policy(state.game_id, 0, legal)
    builder.record_decision(
        state,
        legal_action_ids=legal,
        probabilities=probabilities,
        win_draw_loss_prediction=synthetic_value(state.game_id, 0),
        selected_action_id=legal[0],
    )
    with pytest.raises(TrajectoryError, match="in order"):
        builder.record_decision(
            state,
            legal_action_ids=legal,
            probabilities=probabilities,
            win_draw_loss_prediction=synthetic_value(state.game_id, 0),
            selected_action_id=legal[0],
        )


def test_builder_refuses_to_finish_a_live_game():
    simulator = BatchSimulator(1, root_seed=909)
    builder = GameTrajectoryBuilder(
        game_id=simulator.game_id(0),
        environment_id=simulator.environment_id(0),
        generation=simulator.generation(0),
        red_setup=simulator.setups(0)[0],
        blue_setup=simulator.setups(0)[1],
        rules=simulator.rules,
    )
    with pytest.raises(TrajectoryError, match="not terminal"):
        builder.finish(simulator.game_state(0))


def test_builder_rejects_a_non_positive_interval():
    simulator = BatchSimulator(1, root_seed=1010)
    with pytest.raises(TrajectoryError, match="snapshot_interval"):
        GameTrajectoryBuilder(
            game_id=simulator.game_id(0),
            environment_id=simulator.environment_id(0),
            generation=simulator.generation(0),
            red_setup=simulator.setups(0)[0],
            blue_setup=simulator.setups(0)[1],
            rules=simulator.rules,
            snapshot_interval=0,
        )
