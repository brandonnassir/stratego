"""Shared fixtures for the Phase 16 Agent 2 stochastic-search tests.

The accepted Phase 15 fixtures are re-exported unchanged: the hermetic
`fake_models` (random weights, fast, tests mechanics for any weights) and
the bound `models` (the real frozen stack, digest-checked, skipping cleanly
when the handoff is absent). Phase 16 adds only what is new here.
"""

from tests.search.phase15.conftest import (  # noqa: F401
    REPOSITORY_ROOT,
    fake_models,
    handoff_available,
    midgame_state,
    models,
    repository_root,
    setup_sources,
)

import pytest


@pytest.fixture(scope="session")
def phase15_position_manifest():
    """The frozen Phase 15 Stage A position manifest, if present."""
    import json

    path = REPOSITORY_ROOT / "reports/phase15/agent_02_position_manifest.json"
    if not path.is_file():
        pytest.skip("the Phase 15 position manifest is not present")
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def phase15_stage_a_rows():
    """The frozen Phase 15 Stage A decisions CSV, if present."""
    import csv

    path = REPOSITORY_ROOT / "reports/phase15/agent_02_decisions.csv"
    if not path.is_file():
        pytest.skip("the Phase 15 Stage A decisions CSV is not present")
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))
