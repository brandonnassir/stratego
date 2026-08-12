"""The interface every Stratego network implements, whatever its architecture.

Phase 5 had exactly one network, so `checkpoint.py` and `policy_adapter.py`
could name `IntegrationModel` directly. Phase 6 adds a whole candidate family,
and the choice at that point is between loosening those checks to "anything with
a forward method" or writing the interface down. This module writes it down.

The rule it encodes: a checkpoint, a policy adapter and an evaluator care about
the *boundary* -- tokens in, validated :class:`~stratego.model.contract.ModelOutputs`
out, plus enough identity to know what the weights mean -- and never about which
architecture is behind it. Anything satisfying this base class can be saved,
loaded, and handed to `NeuralCheckpointPolicy` without a second decision path.

Registration, not inheritance, is what makes a model *loadable*: see
:func:`stratego.model.checkpoint.register_architecture`. Subclassing this only
promises the interface.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar, Protocol, runtime_checkable

import torch
from torch import nn

from .contract import ModelOutputs
from .tokenization import observation_to_tokens


@runtime_checkable
class ModelConfiguration(Protocol):
    """What every architecture's configuration object must provide.

    Only two things, because only two things are needed to write a checkpoint
    that can be validated later: a serializable form, and a stable identity.
    """

    def to_dict(self) -> dict: ...


class StrategoModel(nn.Module, abc.ABC):
    """Base class for every network that may cross the model boundary.

    Subclasses must set :attr:`architecture_id`, hold a serializable `config`,
    and return validated :class:`ModelOutputs` from :meth:`forward`. Everything
    downstream -- checkpoints, the policy adapter, Phase 4 evaluation -- is
    written against this and nothing more.
    """

    #: The checkpoint's `model_architecture_id`. Must be registered with
    #: :mod:`stratego.model.checkpoint` before a checkpoint can be written.
    architecture_id: ClassVar[str] = ""

    #: True only for scaffolding networks that must never be mistaken for a
    #: production candidate (the Phase 5 fixture sets it).
    is_integration_fixture: ClassVar[bool] = False

    config: Any

    @property
    @abc.abstractmethod
    def initialisation_seed(self) -> int:
        """The explicit seed the weights were initialised from."""

    @abc.abstractmethod
    def reset_parameters(self, *, seed: int | None = None) -> None:
        """Re-initialise deterministically from an explicit seed."""

    @abc.abstractmethod
    def architecture_summary(self) -> dict:
        """Serializable description, carried into checkpoints and reports."""

    @abc.abstractmethod
    def forward(self, tokens: torch.Tensor) -> ModelOutputs:
        """`[B, 100, 127]` tokens -> validated policy / value / belief heads."""

    def parameter_count(self) -> int:
        """Every parameter, trainable or not."""
        return sum(parameter.numel() for parameter in self.parameters())

    def trainable_parameter_count(self) -> int:
        """Parameters that carry a gradient. Equal to :meth:`parameter_count`
        for every network in this repository -- nothing is frozen -- but the two
        are reported separately rather than assumed identical."""
        return sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )

    def forward_observation(self, observation: torch.Tensor) -> ModelOutputs:
        """Convenience path from the canonical `[B, 127, 10, 10]` input."""
        return self.forward(observation_to_tokens(observation))


__all__ = ["ModelConfiguration", "StrategoModel"]
