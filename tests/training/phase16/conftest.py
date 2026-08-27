"""Shared fixtures for the Phase 16 Agent 3 training-loop tests.

Everything here is *small*: a 4-game population on a 128-decision window on
CPU. The tests are about mechanics -- identity, targets, schedules, resume --
and mechanics do not need a production window to be wrong in a visible way. The
one genuinely expensive object, the starting C1 model, is session-scoped and
skips cleanly when the read-only copy is absent.
"""

from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return REPOSITORY_ROOT


@pytest.fixture(scope="session")
def starting_model():
    """The read-only P24 copy, digest-checked, on CPU."""
    from stratego.training.phase16.checkpoint import (
        STARTING_CHECKPOINT,
        load_starting_model,
    )

    if not (REPOSITORY_ROOT / STARTING_CHECKPOINT).is_file():
        pytest.skip("the read-only P24 copy is not present")
    return load_starting_model(device="cpu", root=REPOSITORY_ROOT)


@pytest.fixture(scope="session")
def library_source():
    """The library setup mixture, built once."""
    from stratego.training.phase16.setups import build_setup_source

    return build_setup_source("library", root=REPOSITORY_ROOT)


@pytest.fixture(scope="session")
def expanded_source():
    """The expanded setup mixture, built once; skips without an adversarial pack."""
    from stratego.training.phase16.setups import Phase16SetupError, build_setup_source

    try:
        return build_setup_source("expanded", root=REPOSITORY_ROOT)
    except Phase16SetupError as error:
        pytest.skip(str(error))


@pytest.fixture
def tiny_config():
    """A 4-game, 128-decision arm on CPU."""
    from stratego.training.phase16.contract import ARM_B

    return ARM_B.replace(
        arm_id="test_tiny",
        population=4,
        window_decisions=128,
        minibatch_size=8,
        device="cpu",
        collection_device="cpu",
    )


@pytest.fixture
def tiny_collector(tiny_config, starting_model, library_source):
    """A seated 4-game population on the tiny config."""
    from stratego.training.phase16.collector import WindowCollector
    from stratego.training.phase16.population import HistoricalPool
    from stratego.training.phase16.snapshots import bind_anchor, participants_for

    historical = bind_anchor(starting_model, identity="P24", device="cpu")
    participants = participants_for(
        starting_model, identity="CURRENT", device="cpu", historical=historical
    )
    return WindowCollector(
        tiny_config,
        participants,
        setup_source=library_source,
        pool=HistoricalPool("P24"),
    )
