"""Shared fixtures for the Phase 5 model tests.

The fixtures are session-scoped because building the fixture network and writing
a checkpoint are the two slowest things these tests do, and neither is mutated
by any test: every test either reads weights or builds its own model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from stratego.model.checkpoint import save_checkpoint
from stratego.model.integration_model import IntegrationModel, build_integration_model
from stratego.model.policy_adapter import (
    GreedyNeuralPolicy,
    NeuralCheckpointPolicy,
    SeededCategoricalNeuralPolicy,
)

TEST_SEED = 20250501


@pytest.fixture(scope="session")
def repository_root() -> Path:
    """The repository root, for the tests that read a real shipped artifact."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def model() -> IntegrationModel:
    """The deterministic CPU float32 fixture network."""
    return build_integration_model(seed=TEST_SEED)


@pytest.fixture(scope="session")
def checkpoint_path(tmp_path_factory, model) -> "object":
    """A real checkpoint on disk, written once for the whole session."""
    directory = tmp_path_factory.mktemp("phase5_checkpoint")
    return save_checkpoint(
        model,
        directory / "integration_model_v1.pt",
        training_iteration=0,
        training_step=0,
        training_metrics={"note": "untrained integration fixture"},
    )


@pytest.fixture(scope="session")
def greedy_policy(checkpoint_path) -> GreedyNeuralPolicy:
    return GreedyNeuralPolicy.from_checkpoint(checkpoint_path)


@pytest.fixture(scope="session")
def sampling_policy(checkpoint_path) -> SeededCategoricalNeuralPolicy:
    return SeededCategoricalNeuralPolicy.from_checkpoint(checkpoint_path)


def deterministic_observation(seed: int = 0, batch: int = 1) -> torch.Tensor:
    """A reproducible pseudo-observation in the canonical input shape.

    Not a real board -- tests that need engine semantics build a real position.
    This exists for the pure tensor-contract tests, where only shape, dtype and
    determinism matter.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(batch, 127, 10, 10, generator=generator)


def crafted_policy_logits(
    winner: int, *, background: float = -50.0, peak: float = 50.0
) -> torch.Tensor:
    """`[10000]` logits with a unique maximum at `winner`.

    Used wherever a test needs to know exactly which action a correct
    implementation must pick, with no possibility of a tie.
    """
    logits = torch.full((10000,), background, dtype=torch.float32)
    logits[winner] = peak
    return logits


class StubOutputPolicy(NeuralCheckpointPolicy):
    """A policy whose forward pass is replaced by a caller-supplied logit row.

    Lets the legality, tie-break and numerical tests drive the *real* decision
    path -- requirement checks, legality cross-check, selection, diagnostics --
    without needing a model that happens to produce the desired logits.
    """

    policy_id = "integration_model_v2_stub"

    def __init__(self, model, policy_logits: torch.Tensor, *, mode: str, **kwargs):
        super().__init__(model, **kwargs)
        self.policy_logits = policy_logits
        # Instance attribute shadows the class attribute, so one stub class can
        # exercise both selection modes.
        self.decision_mode = mode
        self.calls = 0

    def evaluate(self, observation: np.ndarray):
        from stratego.model.contract import ModelOutputs

        self.calls += 1
        row = self.policy_logits
        if row.dim() == 1:
            row = row[None, :]
        return ModelOutputs(
            policy_logits=row,
            value_logits=torch.zeros(1, 3),
            belief_logits=torch.zeros(1, 100, 12),
        )
