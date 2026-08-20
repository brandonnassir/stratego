"""Shared fixtures for the Phase 12 search tests.

The engine mechanics do not depend on trained weights, so the fixtures use
a randomly initialized C1 (`stratego_transformer_v1`) rather than loading
the accepted checkpoint: legality, boundaries, determinism and value
plumbing must hold for any weights, and the tests stay fast and hermetic.
The runner script exercises the real accepted checkpoint.
"""

import pytest

from stratego.belief.phase11b.interface import Phase11BPublicState
from stratego.engine.observation import build_observation
from stratego.evaluation.phase11_public_state import build_public_state_document
from stratego.evaluation.policy import build_public_view
from stratego.model.production_model import ProductionModel

from tests.helpers import nonterminal_state


@pytest.fixture(scope="session")
def random_c1():
    model = ProductionModel("C1")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@pytest.fixture(scope="session")
def midgame_state():
    """A deterministic non-terminal position with plenty of hidden pieces."""
    return nonterminal_state(24, first_seed=3)


def public_state_for(state):
    """The Phase 11B public-state pair for the acting player of `state`."""
    observer = state.acting_player
    observation = build_observation(state, observer)
    document = build_public_state_document(
        build_public_view(state, observer), observation
    )
    return Phase11BPublicState(document, observation)


@pytest.fixture(scope="session")
def midgame_public(midgame_state):
    return public_state_for(midgame_state)
