"""Shared fixtures for the Phase 17 Agent 3 setup tests.

Everything here is small and on CPU. The tests are about mechanics --
masking, orientation, causality, outcome signs, queue discipline, resume
identity -- and mechanics do not need a production pool to be wrong in a
visible way. The one shared expensive object, the setup model, is
session-scoped and built from a fixed seed.
"""

from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "RUN-TEST-A"


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return REPOSITORY_ROOT


@pytest.fixture(scope="session")
def setup_model():
    from stratego.training.phase17.setup_model import build_setup_model

    return build_setup_model(device="cpu", seed=1717)


@pytest.fixture(scope="session")
def model_digest(setup_model):
    from stratego.training.phase9_behavior import state_dict_digest

    return state_dict_digest(setup_model)


@pytest.fixture
def config():
    from stratego.training.phase17.setup_contract import SetupTrainingConfig

    return SetupTrainingConfig(
        run_id=RUN_ID, total_iterations=626, device="cpu", minibatch_episodes=8
    )


@pytest.fixture(scope="session")
def red_samples(setup_model, model_digest):
    from stratego.training.phase17.setup_sampling import generate_setups

    return generate_setups(
        setup_model,
        run_id=RUN_ID,
        game_ids=[f"game-{index}" for index in range(24)],
        color=0,
        model_state_digest=model_digest,
        snapshot_iteration=0,
    )


@pytest.fixture(scope="session")
def blue_samples(setup_model, model_digest):
    from stratego.training.phase17.setup_sampling import generate_setups

    return generate_setups(
        setup_model,
        run_id=RUN_ID,
        game_ids=[f"game-{index}" for index in range(24)],
        color=1,
        model_state_digest=model_digest,
        snapshot_iteration=0,
    )


@pytest.fixture
def completed_episodes(red_samples, blue_samples):
    """24 games' worth of episodes, with a mix of Red wins, Blue wins and draws."""
    from stratego.training.phase17.setup_episode import attach_setup_episodes

    results = ["red_win", "blue_win", "draw"]
    episodes = []
    for index in range(len(red_samples)):
        pair = attach_setup_episodes(
            red_samples[index], blue_samples[index], run_id=RUN_ID, game_id=f"game-{index}"
        )
        episodes.extend(pair.complete(results[index % len(results)]))
    return episodes
