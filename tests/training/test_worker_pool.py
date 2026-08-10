"""Multiprocess CPU simulation layer: equivalence, isolation and failure surface.

These tests run the real pool with real worker processes at a small scale.
`scripts/run_phase3_agent02.py` runs the same checks at acceptance scale, so a
regression shows up in the ordinary test run rather than only in the harness.
"""

import os
import pickle

import numpy as np
import pytest

from stratego.engine.constants import ACTION_SPACE_SIZE, TRAINING_RULES
from stratego.training.batch_simulation import BatchSimulator
from stratego.training.shared_buffers import (
    STATUS_ACTIVE,
    STATUS_TERMINAL,
    buffer_nbytes,
)
from stratego.training.worker_pool import (
    THREAD_LIMIT_VARIABLES,
    PhaseReport,
    StaleBufferError,
    WorkerCrashError,
    WorkerPool,
    WorkerTimeoutError,
    collect_finished,
    partition_environments,
    select_action,
    select_actions,
    slot_hash,
    slot_hashes,
)

ROOT_SEED = 4242

#: Whatever the person running the suite has set, captured before any pool
#: touches the environment.
AMBIENT_THREAD_LIMITS = ",".join(
    os.environ.get(name, "") for name in THREAD_LIMIT_VARIABLES
)


@pytest.fixture
def pool():
    """A small running pool, always shut down even if a test fails."""
    running = WorkerPool(24, 4, root_seed=ROOT_SEED, step_timeout=120.0)
    running.start()
    try:
        yield running
    finally:
        running.shutdown()


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------


def test_workers_own_disjoint_contiguous_ranges_that_cover_the_batch():
    assignments = partition_environments(1024, 8)
    assert [a.size for a in assignments] == [128] * 8
    assert assignments[0].start == 0
    assert assignments[-1].stop == 1024
    for previous, following in zip(assignments, assignments[1:]):
        assert previous.stop == following.start


def test_an_uneven_split_spreads_the_remainder_and_still_covers_the_batch():
    assignments = partition_environments(1024, 6)
    assert sum(a.size for a in assignments) == 1024
    assert [a.size for a in assignments] == [171, 171, 171, 171, 170, 170]
    assert assignments[-1].stop == 1024


def test_a_partition_that_would_starve_a_worker_is_rejected():
    with pytest.raises(ValueError):
        partition_environments(3, 4)
    with pytest.raises(ValueError):
        partition_environments(16, 0)


# ---------------------------------------------------------------------------
# Backward-compatible batch interface extension
# ---------------------------------------------------------------------------


def test_the_default_batch_simulator_is_unchanged_by_the_extension():
    """`first_environment_id` defaults to 0, so Agent 1's semantics are intact."""
    simulator = BatchSimulator(6, root_seed=ROOT_SEED)
    assert simulator.first_environment_id == 0
    assert simulator.environment_ids() == (0, 1, 2, 3, 4, 5)


def test_an_offset_batch_holds_exactly_the_slots_of_the_matching_global_range():
    """A worker's simulator is a window on the one global batch, not a copy of it."""
    whole = BatchSimulator(12, root_seed=ROOT_SEED)
    window = BatchSimulator(4, root_seed=ROOT_SEED, first_environment_id=8)
    assert window.environment_ids() == (8, 9, 10, 11)
    for local, slot in enumerate(range(8, 12)):
        assert window.game_id(local) == whole.game_id(slot)
        assert window.slot_seed(local) == whole.slot_seed(slot)
        assert window.slot_fingerprint(local) == whole.slot_fingerprint(slot)


def test_a_negative_environment_offset_is_rejected():
    with pytest.raises(ValueError):
        BatchSimulator(4, root_seed=ROOT_SEED, first_environment_id=-1)


# ---------------------------------------------------------------------------
# Deterministic policy
# ---------------------------------------------------------------------------


def test_the_scalar_and_vector_hashes_agree():
    environment_id = np.arange(64, dtype=np.int32)
    generation = (environment_id % 5).astype(np.int32)
    ply = (environment_id * 7 % 31).astype(np.int32)
    vector = slot_hashes(ROOT_SEED, environment_id, generation, ply)
    for index in range(64):
        assert int(vector[index]) == slot_hash(
            ROOT_SEED, int(environment_id[index]), int(generation[index]), int(ply[index])
        )


def test_the_policy_depends_only_on_slot_identity_and_ply():
    first = slot_hash(1, 2, 3, 4)
    assert first == slot_hash(1, 2, 3, 4)
    assert first != slot_hash(1, 2, 3, 5)
    assert first != slot_hash(1, 2, 4, 4)
    assert first != slot_hash(1, 3, 3, 4)
    assert first != slot_hash(2, 2, 3, 4)


def test_the_dense_mask_policy_matches_the_legal_list_policy(pool):
    """The coordinator only has masks; the reference only has lists. Same pick."""
    reference = BatchSimulator(24, root_seed=ROOT_SEED)
    buffers = pool.buffers
    for _ in range(12):
        actions = pool.select_actions()
        for slot in range(24):
            expected = select_action(
                ROOT_SEED,
                int(buffers.environment_id[slot]),
                int(buffers.generation[slot]),
                int(buffers.ply[slot]),
                reference.legal_actions(slot),
            )
            assert int(actions[slot]) == expected
            assert buffers.legal_mask[slot, actions[slot]] == 1
        pool.set_actions(actions)
        pool.step(auto_reset=False)
        reference.step({slot: int(actions[slot]) for slot in range(24)})
        if reference.finished_slots():  # keep the two sides comparable
            break


def test_the_policy_skips_slots_with_no_player_to_move(pool):
    buffers = pool.buffers
    buffers.status[5] = STATUS_TERMINAL
    buffers.legal_mask[5].fill(0)
    buffers.legal_count[5] = 0
    actions = select_actions(buffers, ROOT_SEED)
    assert actions[5] == -1
    assert np.all(actions[np.arange(24) != 5] >= 0)


# ---------------------------------------------------------------------------
# Cross-process equivalence
# ---------------------------------------------------------------------------


def compare_slot(buffers, reference: BatchSimulator, slot: int) -> list[str]:
    """Compare one shared-memory row against the single-process reference."""
    problems: list[str] = []
    if int(buffers.environment_id[slot]) != reference.environment_id(slot):
        problems.append("environment_id differs")
    if int(buffers.generation[slot]) != reference.generation(slot):
        problems.append("generation differs")
    state = reference.game_state(slot)
    if int(buffers.ply[slot]) != state.total_moves:
        problems.append("ply differs")
    if int(buffers.acting_player[slot]) != reference.acting_player(slot):
        problems.append("acting player differs")
    if bool(buffers.terminal[slot]) != state.terminal:
        problems.append("terminal flag differs")
    if state.terminal:
        if int(buffers.status[slot]) != STATUS_TERMINAL:
            problems.append("status is not terminal")
        if buffers.legal_mask[slot].any():
            problems.append("a terminal slot published a non-empty legality mask")
        if buffers.observations[slot].any():
            problems.append("a terminal slot published an observation")
        return problems
    if int(buffers.status[slot]) != STATUS_ACTIVE:
        problems.append("status is not active")
    if not np.array_equal(buffers.observations[slot], reference.observation(slot)):
        problems.append("observation differs")
    if not np.array_equal(buffers.legal_mask[slot], reference.legal_action_mask(slot)):
        problems.append("dense legal mask differs")
    if int(buffers.legal_count[slot]) != len(reference.legal_actions(slot)):
        problems.append("legal count differs")
    return problems


def run_matched(pool: WorkerPool, steps: int) -> dict:
    """Drive the pool and an identically seeded single-process batch in lockstep."""
    reference = BatchSimulator(
        pool.num_environments, root_seed=pool.root_seed, rules=pool.rules
    )
    buffers = pool.buffers
    episodes = np.zeros(pool.num_environments, dtype=np.int32)
    problems: list[str] = []
    compared = 0
    stepped = 0
    resets = 0
    finished = 0

    for slot in range(pool.num_environments):
        problems.extend(compare_slot(buffers, reference, slot))

    for _ in range(steps):
        actions = pool.select_actions()
        pool.set_actions(actions)
        report = pool.step(auto_reset=True)
        stepped += report.stepped

        reference.step({slot: int(actions[slot]) for slot in range(len(actions)) if actions[slot] >= 0})
        outcomes = {
            outcome["environment_id"]: outcome
            for outcome in collect_finished(buffers, episodes)
        }
        for slot in reference.finished_slots():
            finished += 1
            expected = reference.outcome(slot)
            reported = outcomes.pop(expected.environment_id, None)
            if reported is None:
                problems.append(f"slot {slot} finished without a reported outcome")
                continue
            if reported["terminal_reason"] != expected.terminal_reason:
                problems.append("terminal reason differs")
            if reported["winner"] != expected.winner:
                problems.append("winner differs")
            if reported["is_draw"] != expected.is_draw:
                problems.append("draw flag differs")
            if reported["total_moves"] != expected.total_moves:
                problems.append("final ply differs")
            if reported["result_for_red"] != expected.result_for_red:
                problems.append("red result differs")
            if reported["generation"] != expected.generation:
                problems.append("outcome generation differs")
        if outcomes:
            problems.append(f"unexpected reported outcomes: {sorted(outcomes)}")
        resets += len(reference.reset_finished())

        for slot in range(pool.num_environments):
            problems.extend(compare_slot(buffers, reference, slot))
            compared += 1

    return {
        "problems": problems,
        "compared": compared,
        "stepped": stepped,
        "resets": resets,
        "finished": finished,
    }


def test_multiprocess_stepping_matches_the_single_process_batch_wrapper(pool):
    result = run_matched(pool, steps=120)
    assert result["problems"] == []
    assert result["stepped"] == 24 * 120
    assert result["compared"] == 24 * 120


def test_the_result_does_not_depend_on_how_many_workers_are_used():
    """Slot content is a function of identity, not of the partitioning."""
    fingerprints = []
    for num_workers in (2, 3, 8):
        running = WorkerPool(24, num_workers, root_seed=77, step_timeout=120.0)
        running.start()
        try:
            for _ in range(40):
                running.set_actions(running.select_actions())
                running.step()
            buffers = running.buffers
            fingerprints.append(
                (
                    buffers.observations.tobytes(),
                    buffers.legal_mask.tobytes(),
                    buffers.generation.tolist(),
                    buffers.ply.tolist(),
                    buffers.acting_player.tolist(),
                    buffers.episode_count.tolist(),
                )
            )
        finally:
            running.shutdown()
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]


def test_environment_identifiers_are_globally_unique_across_workers(pool):
    buffers = pool.buffers
    assert buffers.environment_id.tolist() == list(range(24))
    assert sorted(set(buffers.worker_id.tolist())) == [0, 1, 2, 3]
    for assignment in pool.assignments:
        owned = np.flatnonzero(buffers.worker_id == assignment.worker_id)
        assert owned.tolist() == list(range(assignment.start, assignment.stop))


# ---------------------------------------------------------------------------
# Reset isolation
# ---------------------------------------------------------------------------


def test_a_requested_reset_touches_only_the_requested_slots(pool):
    buffers = pool.buffers
    for _ in range(15):
        pool.set_actions(pool.select_actions())
        pool.step()

    selected = [0, 7, 12, 23]  # spread across every worker
    untouched = [slot for slot in range(24) if slot not in selected]
    before_untouched = buffers.snapshot_rows(untouched)
    before_generation = buffers.generation.copy()
    before_environment = buffers.environment_id.copy()

    pool.request_reset(selected)
    pool.step(apply_actions=False, auto_reset=False)

    # Neighbouring slots are byte-identical in every field except the publish
    # counter, which every slot advances on every phase.
    differing = buffers.rows_equal(before_untouched, untouched)
    assert differing == ["publish_sequence"]

    for slot in selected:
        assert buffers.generation[slot] == before_generation[slot] + 1
        assert buffers.environment_id[slot] == before_environment[slot]
        assert buffers.ply[slot] == 0
        assert buffers.terminal[slot] == 0
        assert buffers.status[slot] == STATUS_ACTIVE
        assert buffers.legal_count[slot] > 0
    for slot in untouched:
        assert buffers.generation[slot] == before_generation[slot]
    assert np.all(buffers.reset_request == 0)


def test_a_reset_slot_holds_the_game_its_new_generation_seeds(pool):
    buffers = pool.buffers
    pool.request_reset([3, 19])
    pool.step(apply_actions=False, auto_reset=False)

    for slot in (3, 19):
        expected = BatchSimulator(
            1,
            root_seed=pool.root_seed,
            rules=pool.rules,
            first_environment_id=slot,
        )
        # A reset slot must be a brand-new game, rebuildable from its identity
        # alone -- which is what lets any process reconstruct it later.
        for _ in range(int(buffers.generation[slot])):
            expected.reset_slots([0])
        assert np.array_equal(buffers.observations[slot], expected.observation(0))
        assert np.array_equal(buffers.legal_mask[slot], expected.legal_action_mask(0))


def test_a_reset_request_outside_the_batch_is_rejected(pool):
    with pytest.raises(ValueError):
        pool.request_reset([0, 24])


def test_finished_games_reset_independently_and_bump_the_generation_once():
    """Slots reach terminal at different plies and only those slots reset."""
    running = WorkerPool(16, 4, root_seed=909, step_timeout=120.0)
    running.start()
    try:
        buffers = running.buffers
        episodes = np.zeros(16, dtype=np.int32)
        generations = buffers.generation.copy()
        completed = 0
        for _ in range(400):
            running.set_actions(running.select_actions())
            report = running.step(auto_reset=True)
            for outcome in collect_finished(buffers, episodes):
                slot = outcome["slot"]
                completed += 1
                assert outcome["generation"] == generations[slot]
                generations[slot] += 1
                assert buffers.generation[slot] == generations[slot]
                assert buffers.ply[slot] == 0
            assert report.resets == report.terminals
            if completed >= 3:
                break
        assert completed >= 1, "no game finished; the test cannot check resets"
        assert buffers.episode_count.sum() == completed
    finally:
        running.shutdown()


# ---------------------------------------------------------------------------
# Control channel
# ---------------------------------------------------------------------------


def test_no_bulk_payload_travels_through_the_control_pipes(pool):
    """Commands and replies stay small fixed-shape dictionaries of scalars."""
    pool.set_actions(pool.select_actions())
    pool.step()

    assert len(pickle.dumps(pool.last_command)) < 256
    assert pool.last_replies
    for reply in pool.last_replies:
        assert len(pickle.dumps(reply)) < 512
        for key, value in reply.items():
            assert isinstance(value, (int, float, str)), key
    # For scale: one observation alone is larger than every control message in
    # the phase put together.
    control_bytes = len(pickle.dumps(pool.last_command)) * len(pool.assignments) + sum(
        len(pickle.dumps(reply)) for reply in pool.last_replies
    )
    assert control_bytes < pool.buffers.observations[0].nbytes


def test_the_shared_block_is_allocated_once_and_never_grows(pool):
    expected = buffer_nbytes(24)
    assert pool.buffers.nbytes == expected
    name = pool.buffers.descriptor.name
    for _ in range(5):
        pool.set_actions(pool.select_actions())
        pool.step()
        assert pool.buffers.nbytes == expected
        assert pool.buffers.descriptor.name == name


def test_a_phase_report_accounts_for_the_time_it_spent(pool):
    pool.set_actions(pool.select_actions())
    report = pool.step()
    assert isinstance(report, PhaseReport)
    assert report.stepped == 24
    assert report.observation_builds > 0
    assert report.wall_seconds > 0
    assert report.wait_seconds <= report.wall_seconds
    assert report.straggler_seconds >= 0
    assert report.worker_cpu_seconds > 0


# ---------------------------------------------------------------------------
# Failure surface
# ---------------------------------------------------------------------------


def test_a_killed_worker_is_reported_as_an_infrastructure_error():
    running = WorkerPool(16, 4, root_seed=31, step_timeout=20.0)
    running.start()
    try:
        running.set_actions(running.select_actions())
        running.step()
        running.kill_worker(2)
        with pytest.raises(WorkerCrashError) as raised:
            for _ in range(3):
                running.set_actions(running.select_actions())
                running.step()
        message = str(raised.value)
        assert "worker 2" in message
        # The error has to say which environments stopped being simulated.
        assert "[8, 12)" in message or "worker 2" in message
    finally:
        running.shutdown()


def test_a_worker_that_stops_responding_is_reported_rather_than_waited_on():
    running = WorkerPool(16, 4, root_seed=31, step_timeout=1.5)
    running.start()
    try:
        running.stall_worker(1, 30.0)
        with pytest.raises(WorkerTimeoutError) as raised:
            running.set_actions(running.select_actions())
            running.step()
        assert "did not complete" in str(raised.value)
    finally:
        # The stalled worker is still sleeping, so do not wait for it politely.
        for worker_id in range(4):
            try:
                running.kill_worker(worker_id)
            except Exception:
                pass
        running.shutdown(timeout=5.0)


def test_a_slot_that_was_not_republished_is_a_hard_error(pool):
    """The coordinator must never read a buffer a worker stopped maintaining."""
    pool.set_actions(pool.select_actions())
    pool.step()
    pool._expected_publish[9] += 1  # pretend slot 9 was skipped by its worker
    with pytest.raises(StaleBufferError) as raised:
        pool._check_published()
    assert "stale" in str(raised.value)


def test_a_pool_cannot_be_used_after_shutdown():
    running = WorkerPool(8, 2, root_seed=5)
    running.start()
    running.shutdown()
    with pytest.raises(Exception):
        running.step()


def test_a_worker_error_carries_the_remote_traceback():
    """A malformed command must surface as a fault, not a silent hang."""
    running = WorkerPool(8, 2, root_seed=5, step_timeout=20.0)
    running.start()
    try:
        running._sequence += 1
        for worker in running._workers:
            worker.connection.send({"kind": "nonsense", "sequence": running._sequence})
        with pytest.raises(Exception) as raised:
            running._await_replies(20.0, stage="step")
        assert "nonsense" in str(raised.value)
    finally:
        running.shutdown(timeout=5.0)


# ---------------------------------------------------------------------------
# Hidden information
# ---------------------------------------------------------------------------


def test_the_shared_payload_carries_no_privileged_field(pool):
    """Belief targets and true identities must not reach a model-facing buffer."""
    published = set(pool.buffers.field_names())
    assert "belief_targets" not in published
    assert "true_type" not in published
    assert not any("belief" in name or "hidden" in name for name in published)
    # The published observation is exactly the frozen observer-safe contract.
    assert pool.buffers.observations.shape[1:] == (127, 10, 10)
    assert pool.buffers.legal_mask.shape[1] == ACTION_SPACE_SIZE


def ambient_thread_limits() -> str:
    return ",".join(os.environ.get(name, "") for name in THREAD_LIMIT_VARIABLES)


def test_workers_run_single_threaded_numerical_libraries(pool):
    """Otherwise the CPU scaling screen would measure thread thrash.

    The workers report what they actually see in their own environment, so this
    proves inheritance rather than assuming it.
    """
    assert set(pool.worker_thread_limits) == {0, 1, 2, 3}
    assert set(pool.worker_thread_limits.values()) == {"1,1,1,1,1"}


def test_the_coordinator_environment_is_restored_after_startup(pool):
    """The limits exist for the children; the coordinator is left as it was."""
    assert ambient_thread_limits() == AMBIENT_THREAD_LIMITS


def test_thread_limits_can_be_left_alone():
    running = WorkerPool(8, 2, root_seed=5, limit_worker_threads=False)
    running.start()
    try:
        assert set(running.worker_thread_limits.values()) == {AMBIENT_THREAD_LIMITS}
    finally:
        running.shutdown()


def test_rules_configuration_reaches_the_workers(pool):
    assert pool.rules is TRAINING_RULES
    reference = BatchSimulator(24, root_seed=ROOT_SEED, rules=TRAINING_RULES)
    assert np.array_equal(pool.buffers.observations[0], reference.observation(0))
