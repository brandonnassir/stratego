"""Phase 17 Agent 2: shared builders for the move-half tests.

No test functions live here. What lives here is the small amount of scaffolding
several test modules need -- a deterministic setup provider, a tiny stand-in
model, and the trace/prediction fixtures -- kept in one place so a change to
the collector's injection points shows up as one edit rather than five.

The setup provider is deliberately a *test double* and says so. Phase 17
production setups come from Agent 3's autoregressive network through Agent 4's
runner; the collector takes an explicit provider and has no default, so nothing
here can leak into a production path by being importable.
"""

from __future__ import annotations

import numpy as np
import torch

from stratego.engine.random_play import make_random_setups
from stratego.training.setup_source import SetupAssignment

TEST_SETUP_FAMILY = "phase17_agent02_test_double_v1"


class DeterministicSetupProvider:
    """Legal engine-order setups from the accepted uniform generator.

    A TEST DOUBLE. It exists so the move half can be exercised end to end
    before Agent 3's setup network lands; it is not a fallback and no
    production path constructs it.
    """

    setup_family = TEST_SETUP_FAMILY

    def __init__(self, *, offset: int = 0) -> None:
        self.offset = int(offset)
        self.assignments = 0

    def describe(self) -> dict:
        return {
            "source_id": self.setup_family,
            "kind": "test_double",
            "produces_provenance": True,
        }

    def assign(self, *, root_seed: int, environment_id: int, generation: int, game_id: str = "") -> SetupAssignment:
        self.assignments += 1
        red, blue = make_random_setups(int(root_seed) + self.offset)
        return SetupAssignment(
            red_setup=red,
            blue_setup=blue,
            provenance={
                "source_id": self.setup_family,
                "game_id": str(game_id),
                "root_seed": int(root_seed),
                "environment_id": int(environment_id),
                "generation": int(generation),
            },
        )


class PerturbableC1(torch.nn.Module):
    """Not used for weights -- see `perturbed_copy` for how snapshot B is made."""


def perturbed_copy(model, *, scale: float = 0.75, seed: int = 17):
    """A copy of `model` with deliberately different policy logits.

    Used by the forced-rebind test: snapshot B must produce a *visibly*
    different legal distribution and a different model-state digest, so that
    "the decision came from B" is a claim about numbers rather than metadata.
    """
    import copy

    clone = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for name, parameter in clone.named_parameters():
            if not name.startswith(("policy_", "value_", "belief_")):
                continue
            noise = torch.randn(
                parameter.shape, generator=generator, dtype=parameter.dtype
            )
            parameter.add_(noise * float(scale))
    return clone


def dirichlet_predictions(count: int, *, seed: int = 20260827) -> list:
    """`count` W/D/L predictions on the simplex, the probe's own construction."""
    rng = np.random.default_rng(int(seed))
    return [tuple(float(v) for v in rng.dirichlet([2.0, 2.0, 1.0])) for _ in range(count)]
