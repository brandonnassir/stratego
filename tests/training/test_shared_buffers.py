"""Shared-memory buffer contract: layout, round trip and isolation helpers.

The round-trip test is the one that matters most: a *separate process* writes
known arrays into its own slot range and the coordinator must see exactly those
bytes, with no serialisation transform anywhere in between.
"""

import multiprocessing as mp
import pickle

import numpy as np
import pytest

from stratego.engine.constants import (
    ACTION_SPACE_SIZE,
    NOT_TERMINAL,
    OBSERVATION_SHAPE,
    TERMINAL_REASONS,
)
from stratego.training.shared_buffers import (
    COORDINATOR_WRITTEN_FIELDS,
    FIELD_SPECS,
    NO_TERMINAL_REASON,
    SHARED_BUFFER_VERSION,
    STATUS_ACTIVE,
    WORKER_WRITTEN_FIELDS,
    SharedBufferDescriptor,
    SharedBufferError,
    SharedEnvironmentBuffers,
    buffer_nbytes,
    plan_layout,
    terminal_reason_code,
    terminal_reason_name,
)


@pytest.fixture
def buffers():
    created = SharedEnvironmentBuffers.create(8)
    try:
        yield created
    finally:
        created.close()
        created.unlink()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_required_payload_has_the_specified_shapes_and_dtypes(buffers):
    """The payload Agent 2's instructions require, at the sizes they require."""
    assert buffers.observations.shape == (8,) + OBSERVATION_SHAPE
    assert buffers.observations.dtype == np.float32
    assert buffers.legal_mask.shape == (8, ACTION_SPACE_SIZE)
    assert buffers.legal_mask.dtype == np.uint8
    assert buffers.acting_player.shape == (8,)
    assert buffers.environment_id.shape == (8,)
    assert buffers.generation.shape == (8,)
    assert buffers.actions.shape == (8,)
    assert buffers.terminal.shape == (8,)


def test_bulk_arrays_are_contiguous_so_they_can_be_wrapped_without_a_copy(buffers):
    assert buffers.observations.flags["C_CONTIGUOUS"]
    assert buffers.legal_mask.flags["C_CONTIGUOUS"]
    # Views into shared memory own no data of their own.
    assert not buffers.observations.flags["OWNDATA"]


def test_fields_do_not_overlap_and_are_cache_line_aligned():
    offsets, total = plan_layout(64)
    ordered = sorted(offsets.items(), key=lambda item: item[1])
    for (name, offset), (_, next_offset) in zip(ordered, ordered[1:]):
        spec = next(spec for spec in FIELD_SPECS if spec.name == name)
        assert offset % 64 == 0
        assert offset + spec.nbytes(64) <= next_offset
    last_name, last_offset = ordered[-1]
    last = next(spec for spec in FIELD_SPECS if spec.name == last_name)
    assert last_offset + last.nbytes(64) <= total
    assert buffer_nbytes(64) == total


def test_every_field_has_exactly_one_writer():
    """The buffers carry no lock, so single-writer discipline is the contract."""
    assert set(WORKER_WRITTEN_FIELDS) & set(COORDINATOR_WRITTEN_FIELDS) == set()
    assert set(WORKER_WRITTEN_FIELDS) | set(COORDINATOR_WRITTEN_FIELDS) == {
        spec.name for spec in FIELD_SPECS
    }
    # The coordinator only ever writes what the workers consume: the action to
    # apply, the reset request, and -- added for Agent 5 -- the model decision a
    # recording worker folds into the slot's trajectory.
    assert set(COORDINATOR_WRITTEN_FIELDS) == {
        "actions",
        "reset_request",
        "policy_probabilities",
        "value_prediction",
        "decision_valid",
    }


def test_create_initialises_every_field_to_its_fill_value(buffers):
    for spec in FIELD_SPECS:
        assert np.all(buffers[spec.name] == spec.fill), spec.name


def test_descriptor_is_small_and_picklable(buffers):
    payload = pickle.dumps(buffers.descriptor)
    # The descriptor is the only buffer-related object that crosses a pipe. It
    # must stay a handle, never a copy of the data.
    assert len(payload) < 1024
    assert pickle.loads(payload) == buffers.descriptor


def test_unknown_field_is_rejected(buffers):
    with pytest.raises(KeyError):
        buffers["belief_targets"]
    with pytest.raises(AttributeError):
        buffers.belief_targets


def test_view_returns_views_not_copies(buffers):
    view = buffers.view(2, 5)
    view["observations"][0, 0, 0, 0] = 7.5
    assert buffers.observations[2, 0, 0, 0] == 7.5
    view["generation"][2] = 4
    assert buffers.generation[4] == 4
    assert buffers.generation[1] == -1


def test_view_outside_the_batch_is_rejected(buffers):
    with pytest.raises(ValueError):
        buffers.view(0, 9)
    with pytest.raises(ValueError):
        buffers.view(-1, 3)


def test_attach_rejects_a_foreign_version(buffers):
    descriptor = SharedBufferDescriptor(
        name=buffers.descriptor.name,
        num_environments=8,
        nbytes=buffers.descriptor.nbytes,
        offsets=buffers.descriptor.offsets,
        version="shared_buffers_v0",
    )
    with pytest.raises(SharedBufferError):
        SharedEnvironmentBuffers.attach(descriptor)


def test_attach_reports_a_missing_block():
    descriptor = SharedBufferDescriptor(
        name="psm_stratego_absent",
        num_environments=1,
        nbytes=64,
        offsets=(),
        version=SHARED_BUFFER_VERSION,
    )
    with pytest.raises(SharedBufferError):
        SharedEnvironmentBuffers.attach(descriptor)


def test_only_the_owner_may_unlink(buffers):
    attached = SharedEnvironmentBuffers.attach(buffers.descriptor)
    try:
        with pytest.raises(SharedBufferError):
            attached.unlink()
    finally:
        attached.close()


# ---------------------------------------------------------------------------
# Cross-process round trip
# ---------------------------------------------------------------------------


def known_observation(slot: int) -> np.ndarray:
    """A distinctive pattern that no zero-fill or partial write could produce."""
    values = np.arange(int(np.prod(OBSERVATION_SHAPE)), dtype=np.float32)
    return ((values * (slot + 1)) % 977.0).reshape(OBSERVATION_SHAPE).astype(np.float32)


def known_mask(slot: int) -> np.ndarray:
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.uint8)
    mask[(slot + 1) :: 7] = 1
    mask[slot] = 1
    return mask


def write_known_rows(descriptor: SharedBufferDescriptor, start: int, stop: int) -> None:
    """Runs in a *separate process*: fill one slot range with known values."""
    attached = SharedEnvironmentBuffers.attach(descriptor)
    try:
        view = attached.view(start, stop)
        for local, slot in enumerate(range(start, stop)):
            view["observations"][local] = known_observation(slot)
            view["legal_mask"][local] = known_mask(slot)
            view["legal_count"][local] = int(known_mask(slot).sum())
            view["acting_player"][local] = slot % 2
            view["environment_id"][local] = slot
            view["generation"][local] = slot * 3
            view["ply"][local] = slot * 11
            view["status"][local] = STATUS_ACTIVE
            view["worker_id"][local] = start
            view["publish_sequence"][local] += 1
            view["last_result_red"][local] = -1.0
    finally:
        view = None
        attached.close()


def test_shared_buffer_round_trip_across_processes():
    """Known arrays written by two worker processes arrive byte-identical."""
    coordinator = SharedEnvironmentBuffers.create(16)
    context = mp.get_context("spawn")
    try:
        processes = [
            context.Process(
                target=write_known_rows, args=(coordinator.descriptor, start, start + 8)
            )
            for start in (0, 8)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=120)
            assert process.exitcode == 0

        for slot in range(16):
            # Exact equality, not approximate: nothing may transform the bytes.
            assert np.array_equal(coordinator.observations[slot], known_observation(slot))
            assert np.array_equal(coordinator.legal_mask[slot], known_mask(slot))
            assert coordinator.environment_id[slot] == slot
            assert coordinator.generation[slot] == slot * 3
            assert coordinator.ply[slot] == slot * 11
            assert coordinator.acting_player[slot] == slot % 2
            assert coordinator.legal_count[slot] == int(known_mask(slot).sum())
            assert coordinator.last_result_red[slot] == -1.0
        assert coordinator.worker_id.tolist() == [0] * 8 + [8] * 8
        assert coordinator.publish_sequence.tolist() == [1] * 16
        # Coordinator-owned fields were never touched by a worker.
        assert np.all(coordinator.actions == -1)
        assert np.all(coordinator.reset_request == 0)
    finally:
        coordinator.close()
        coordinator.unlink()


def test_coordinator_writes_are_visible_to_an_attached_process():
    """The action path: coordinator writes, another process reads the same bytes."""
    coordinator = SharedEnvironmentBuffers.create(4)
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    try:
        coordinator.actions[:] = [11, 22, 33, 44]
        coordinator.reset_request[:] = [0, 1, 0, 1]
        process = context.Process(
            target=read_back_rows, args=(coordinator.descriptor, child)
        )
        process.start()
        child.close()
        assert parent.recv() == {
            "actions": [11, 22, 33, 44],
            "reset_request": [0, 1, 0, 1],
        }
        process.join(timeout=120)
        assert process.exitcode == 0
    finally:
        parent.close()
        coordinator.close()
        coordinator.unlink()


def read_back_rows(descriptor: SharedBufferDescriptor, connection) -> None:
    """Runs in a separate process: report what the coordinator wrote."""
    attached = SharedEnvironmentBuffers.attach(descriptor)
    try:
        connection.send(
            {
                "actions": attached.actions.tolist(),
                "reset_request": attached.reset_request.tolist(),
            }
        )
    finally:
        attached.close()
        connection.close()


# ---------------------------------------------------------------------------
# Isolation and staleness helpers
# ---------------------------------------------------------------------------


def test_stale_slots_reports_exactly_the_slots_that_were_not_republished(buffers):
    expected = np.ones(8, dtype=np.int64)
    buffers.publish_sequence[:] = 1
    assert buffers.stale_slots(expected).tolist() == []
    buffers.publish_sequence[3] = 0
    buffers.publish_sequence[6] = 2
    assert buffers.stale_slots(expected).tolist() == [3, 6]


def test_snapshot_rows_is_an_independent_copy(buffers):
    buffers.observations[2, 0, 0, 0] = 5.0
    before = buffers.snapshot_rows([2, 3])
    buffers.observations[2, 0, 0, 0] = 9.0
    assert before["observations"][0, 0, 0, 0] == 5.0
    assert buffers.rows_equal(before, [2, 3]) == ["observations"]
    buffers.observations[2, 0, 0, 0] = 5.0
    assert buffers.rows_equal(before, [2, 3]) == []


# ---------------------------------------------------------------------------
# Terminal reason codes
# ---------------------------------------------------------------------------


def test_terminal_reason_codes_round_trip_for_every_frozen_reason():
    for reason in TERMINAL_REASONS:
        assert terminal_reason_name(terminal_reason_code(reason)) == reason
    assert terminal_reason_name(NO_TERMINAL_REASON) == NOT_TERMINAL


def test_unknown_terminal_reason_fails_loudly():
    with pytest.raises(SharedBufferError):
        terminal_reason_code("resigned")
    with pytest.raises(SharedBufferError):
        terminal_reason_name(99)
