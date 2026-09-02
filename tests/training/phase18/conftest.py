"""Shared fixtures for the Phase 18 setup-parity tests.

Everything is small and on CPU unless a test is explicitly about the
production device. The one expensive shared object, a 48-setup pool drawn
from a fixed-seed model, is session-scoped.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = "phase18_g2_test_fixture_v1"
RUN_ID = "G2-TEST"


@pytest.fixture(scope="session", autouse=True)
def _threads():
    torch.set_num_threads(4)


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return REPOSITORY_ROOT


@pytest.fixture(scope="session")
def setup_model():
    from stratego.training.phase18.setup_contract import model_seed
    from stratego.training.phase18.setup_model import build_setup_model

    return build_setup_model(device="cpu", seed=model_seed(NAMESPACE, 1))


@pytest.fixture(scope="session")
def model_digest(setup_model):
    from stratego.training.phase18.setup_model import state_dict_digest

    return state_dict_digest(setup_model)


@pytest.fixture
def config():
    from stratego.training.phase18.setup_contract import SetupTrainingConfig

    return SetupTrainingConfig(run_id=RUN_ID, device="cpu")


@pytest.fixture(scope="session")
def pool(setup_model, model_digest):
    """48 setups under one frozen snapshot: forced handedness, seeded reflection."""
    from stratego.training.phase18.setup_sampling import generate_pool

    return generate_pool(
        setup_model,
        namespace=NAMESPACE,
        seed_index=1,
        snapshot_iteration=0,
        snapshot_digest=model_digest,
        count=48,
    )


@pytest.fixture(scope="session")
def outcomes_by_fingerprint(pool):
    """A deterministic multiset of outcomes per pooled setup: 4 per row, with a
    mix of wins, draws and losses, so the running mean is exercised."""
    pattern = [(1, 1, 0, -1), (1, -1, -1, -1), (0, 0, 1, 1), (1, 1, 1, 1), (-1, 0, -1, 1)]
    return {
        sample.content_fingerprint: pattern[index % len(pattern)]
        for index, sample in enumerate(pool.samples)
    }


@pytest.fixture
def filled_buffer(pool, outcomes_by_fingerprint):
    from stratego.training.phase18.setup_buffer import SetupBuffer

    buffer = SetupBuffer(storage_duration=1, device="cpu")
    buffer.add_pool(pool.samples, period=1)
    for fingerprint, outcomes in outcomes_by_fingerprint.items():
        buffer.add_outcomes((fingerprint, z) for z in outcomes)
    return buffer


def batch_to_numpy(batch) -> dict:
    """A `SetupBatch` as float64 numpy arrays, the oracle's input form."""
    return {
        "sequence": batch.sequence.cpu().numpy().astype(np.int64),
        "tokens": batch.tokens.cpu().numpy().astype(np.int64),
        "masks": batch.masks.cpu().numpy().astype(bool),
        "behavior_log_probs": batch.behavior_log_probs.cpu().numpy().astype(np.float64),
        "behavior_selected_log_prob": batch.behavior_selected_log_prob.cpu().numpy().astype(np.float64),
        "advantage": batch.advantage.cpu().numpy().astype(np.float64),
        "value_target": batch.value_target.cpu().numpy().astype(np.float64),
        "entropy_target": batch.entropy_target.cpu().numpy().astype(np.float64),
        "fingerprints": list(batch.fingerprints),
    }
