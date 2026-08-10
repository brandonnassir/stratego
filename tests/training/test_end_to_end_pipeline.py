"""The integrated pipeline: real workers, a real model, real shared memory.

These run the whole cycle at a small scale. `scripts/run_phase3_agent05.py`
runs the same checks at acceptance scale, so a regression surfaces in an
ordinary test run rather than only after a multi-hour harness.

Metal is used when it is available and the central processing unit stands in
when it is not, because what is under test here is the *pipeline*, not the
device. The acceptance harness refuses to substitute the processor; that
distinction is deliberate.
"""

import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stratego.engine.constants import ACTION_SPACE_SIZE  # noqa: E402
from stratego.training.coordinator import (  # noqa: E402
    CoordinatorConfig,
    SelfPlayCoordinator,
)
from stratego.training.end_to_end_benchmark import (  # noqa: E402
    measure_simulation_pipeline,
    reference_game,
    run_integrated_gate,
    run_reconstruction_gate,
)
from stratego.training.shared_buffers import (  # noqa: E402
    COORDINATOR_WRITTEN_FIELDS,
    STATUS_ACTIVE,
    WORKER_WRITTEN_FIELDS,
)
from stratego.training.worker_pool import (  # noqa: E402
    DEFAULT_COLLECTION_POLICY_VERSION,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@pytest.fixture(scope="module")
def small_config():
    return CoordinatorConfig(
        num_environments=16,
        num_workers=2,
        inference_batch_size=8,
        root_seed=90_210,
        precision="float32" if DEVICE == "cpu" else "float16",
        detailed_timing=False,
    )


# ---------------------------------------------------------------------------
# The cycle
# ---------------------------------------------------------------------------


def test_every_sampled_action_is_legal_in_the_published_mask(small_config):
    """The engine is the only authority on legality; sampling must respect it."""
    coordinator = SelfPlayCoordinator(small_config, device=DEVICE)
    coordinator.start()
    try:
        buffers = coordinator.pool.buffers
        for _ in range(12):
            masks = buffers.legal_mask.copy()
            statuses = buffers.status.copy()
            coordinator.step()
            actions = coordinator.last_actions
            for slot in range(small_config.num_environments):
                if statuses[slot] != STATUS_ACTIVE:
                    continue
                action = int(actions[slot])
                assert 0 <= action < ACTION_SPACE_SIZE
                assert masks[slot, action] == 1, (
                    f"slot {slot}: sampled action {action} was not in the "
                    f"published legal mask"
                )
    finally:
        coordinator.shutdown()


def test_inference_is_chunked_by_the_batch_size(small_config):
    """`inference_batch_size` splits the ready rows; it does not cap them."""
    coordinator = SelfPlayCoordinator(small_config, device=DEVICE)
    coordinator.start()
    try:
        metrics = coordinator.step()
        expected = -(-metrics.positions // small_config.inference_batch_size)
        assert metrics.chunks == expected
        # Every active environment gets a decision, not just one batch of them.
        assert metrics.positions == small_config.num_environments
        assert metrics.transitions == small_config.num_environments
    finally:
        coordinator.shutdown()


def test_a_batch_larger_than_the_batch_is_one_underfilled_dispatch():
    config = CoordinatorConfig(
        num_environments=8,
        num_workers=2,
        inference_batch_size=1024,
        root_seed=1234,
        precision="float32" if DEVICE == "cpu" else "float16",
        detailed_timing=False,
    )
    coordinator = SelfPlayCoordinator(config, device=DEVICE)
    coordinator.start()
    try:
        metrics = coordinator.step()
        assert metrics.chunks == 1
        assert metrics.positions == 8
    finally:
        coordinator.shutdown()


def test_games_finish_and_slots_reset_independently(small_config):
    """A finished game must not disturb any other slot."""
    coordinator = SelfPlayCoordinator(small_config, device=DEVICE)
    coordinator.start()
    try:
        buffers = coordinator.pool.buffers
        seen_reset = False
        for _ in range(400):
            generations_before = buffers.generation.copy()
            metrics = coordinator.step()
            if metrics.resets:
                seen_reset = True
                changed = np.flatnonzero(buffers.generation != generations_before)
                assert changed.size == metrics.resets
                # Each reset slot advances by exactly one generation.
                assert np.all(
                    buffers.generation[changed] == generations_before[changed] + 1
                )
                # Every reset slot immediately holds a fresh, playable game.
                assert np.all(buffers.ply[changed] == 0)
                assert np.all(buffers.legal_count[changed] > 0)
                break
        assert seen_reset, "no game finished in 400 steps; raise the step budget"
    finally:
        coordinator.shutdown()


def test_terminal_outcomes_are_counted(small_config):
    coordinator = SelfPlayCoordinator(small_config, device=DEVICE)
    coordinator.start()
    try:
        for _ in range(400):
            coordinator.step()
            if coordinator.games_finished:
                break
        assert coordinator.games_finished > 0
        assert sum(coordinator.terminal_reason_counts.values()) == (
            coordinator.games_finished
        )
        assert "not_terminal" not in coordinator.terminal_reason_counts
    finally:
        coordinator.shutdown()


# ---------------------------------------------------------------------------
# Hidden information and device ownership
# ---------------------------------------------------------------------------


def test_no_privileged_field_reaches_the_shared_transport():
    """Belief targets and the true board stay inside the worker."""
    published = set(WORKER_WRITTEN_FIELDS) | set(COORDINATOR_WRITTEN_FIELDS)
    for forbidden in (
        "belief_targets",
        "belief_target",
        "true_board",
        "piece_identities",
        "red_setup",
        "blue_setup",
    ):
        assert forbidden not in published


def test_published_observation_is_the_acting_players_perspective_only(small_config):
    """The transport carries one perspective, and it is the mover's."""
    from stratego.engine.observation import build_observation

    coordinator = SelfPlayCoordinator(small_config, device=DEVICE)
    coordinator.start()
    try:
        buffers = coordinator.pool.buffers
        for slot in range(small_config.num_environments):
            reference = reference_game(small_config.root_seed, slot, 0)
            expected = build_observation(reference, reference.acting_player)
            assert np.array_equal(buffers.observations[slot], expected)
            assert int(buffers.acting_player[slot]) == reference.acting_player
    finally:
        coordinator.shutdown()


def test_a_simulation_worker_never_imports_torch():
    """Importing the worker layer must not pull the model or Metal in with it."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import stratego.training.worker_pool; "
            "print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False", (
        "stratego.training.worker_pool pulled PyTorch into its import graph; a "
        "simulation worker must stay free of the device"
    )


def test_the_control_pipe_stays_small_with_recording_enabled():
    """Recording adds counters, never bulk payload, to the per-phase reply."""
    import pickle

    config = CoordinatorConfig(
        num_environments=16,
        num_workers=2,
        inference_batch_size=16,
        root_seed=555,
        precision="float32" if DEVICE == "cpu" else "float16",
        record_trajectories=True,
        detailed_timing=False,
    )
    coordinator = SelfPlayCoordinator(config, device=DEVICE)
    coordinator.start()
    try:
        coordinator.step()
        assert coordinator.pool.last_replies
        for reply in coordinator.pool.last_replies:
            assert len(pickle.dumps(reply)) < 1024
    finally:
        coordinator.shutdown()


# ---------------------------------------------------------------------------
# Recording and reconstruction
# ---------------------------------------------------------------------------


def test_recorded_decisions_reconstruct_exactly():
    """A real recorded game must rebuild through Agent 3's path bit for bit."""
    result = run_reconstruction_gate(
        num_environments=32,
        num_workers=2,
        inference_batch_size=32,
        target_decisions=300,
        max_global_steps=2000,
        max_concurrent_verifications=4,
        precision="float32" if DEVICE == "cpu" else "float16",
        device=DEVICE,
    )
    assert result["decisions_reconstructed"] >= 300
    assert result["reconstruction_mismatches"] == 0, result["mismatch_details"][:3]
    assert result["games_joined_late"] == 0
    assert result["decisions_recorded"] > 0
    assert result["record_bytes"] > 0


def test_stored_decisions_carry_the_collection_policy_version():
    """A training consumer has to be able to tell this corpus from Agent 3's."""
    from stratego.training.trajectory import (
        SYNTHETIC_POLICY_VERSION,
        decode_game_record,
    )

    result = run_reconstruction_gate(
        num_environments=32,
        num_workers=2,
        inference_batch_size=32,
        target_decisions=120,
        max_global_steps=2000,
        max_concurrent_verifications=4,
        precision="float32" if DEVICE == "cpu" else "float16",
        device=DEVICE,
    )
    assert result["collection_policy_version"] == DEFAULT_COLLECTION_POLICY_VERSION
    assert result["collection_policy_version"] != SYNTHETIC_POLICY_VERSION


def test_recorded_probabilities_cover_the_legal_set(small_config):
    """Every stored probability row is a distribution over that slot's legal set.

    The legal count has to be read *before* the step. `pool.step()` republishes
    every slot, so afterwards `legal_count` describes the position the game has
    moved on to, while `policy_probabilities` still describes the one the
    decision was taken in. The worker reads both before it advances anything,
    which is the ordering this checks against.
    """
    config = CoordinatorConfig(
        num_environments=16,
        num_workers=2,
        inference_batch_size=16,
        root_seed=4321,
        precision="float32" if DEVICE == "cpu" else "float16",
        record_trajectories=True,
        detailed_timing=False,
    )
    coordinator = SelfPlayCoordinator(config, device=DEVICE)
    coordinator.start()
    try:
        buffers = coordinator.pool.buffers
        for _ in range(5):
            counts_before = buffers.legal_count.copy()
            coordinator.step()
            for slot in range(config.num_environments):
                if not buffers.decision_valid[slot]:
                    continue
                count = int(counts_before[slot])
                row = buffers.policy_probabilities[slot, :count]
                assert count > 0
                assert np.all(row >= 0.0)
                assert row.sum() == pytest.approx(1.0, abs=1e-3)
                # Nothing may spill past the legal prefix.
                tail = buffers.policy_probabilities[slot, count:]
                assert np.all(tail == 0.0)
                values = buffers.value_prediction[slot]
                assert values.sum() == pytest.approx(1.0, abs=1e-3)
    finally:
        coordinator.shutdown()


# ---------------------------------------------------------------------------
# Integrated differential
# ---------------------------------------------------------------------------


def test_integrated_gate_finds_no_mismatch():
    """The whole chain, checked against independently built engine games."""
    report = run_integrated_gate(
        num_environments=16,
        num_workers=2,
        inference_batch_size=16,
        target_environment_steps=600,
        precision="float32" if DEVICE == "cpu" else "float16",
        device=DEVICE,
    )
    assert report.environment_steps >= 600
    assert report.mismatches == 0, report.mismatch_details[:5]
    assert report.row_comparisons > 0
    assert report.action_legality_checks > 0
    # Identity never repeats: a slot's generations are distinct trajectories.
    assert report.distinct_trajectory_keys == 16 + report.resets_observed


# ---------------------------------------------------------------------------
# Simulation-only pipeline (the R numerator)
# ---------------------------------------------------------------------------


def test_simulation_pipeline_runs_without_a_model():
    """The numerator measurement must not need the device at all."""
    result = measure_simulation_pipeline(
        num_environments=32, num_workers=2, seconds=1.5
    )
    assert result["positions_per_second"] > 0
    assert result["positions"] > 0
    assert result["transitions"] > 0
    assert result["global_steps"] > 0


def test_simulation_pipeline_records_when_asked():
    result = measure_simulation_pipeline(
        num_environments=32,
        num_workers=2,
        seconds=1.5,
        record_trajectories=True,
    )
    assert result["decisions_recorded"] > 0
