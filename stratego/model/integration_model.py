"""`integration_model_v1` -- the deliberately small Phase 5 integration fixture.

.. warning::

   **This is an integration fixture, not the final model.** It exists to prove
   that the frozen engine/observation/action system can drive a real PyTorch
   module through a real checkpoint into the real Phase 4 evaluator. It is never
   trained, its weights carry no strategic meaning, and it is **not** the
   production/Ataraxos architecture. Phase 6 owns the architecture decision and
   is free to ignore every shape below. Only the *boundary* (`contract.py`) is
   meant to survive.

Specification sources:

- Phase 5 single-agent instructions, section 4.2 (width 64, two Transformer
  encoder blocks, four heads, modest feed-forward, shared encoder, three heads)
- `05_project_plan.md` section 5 (100 board tokens, source-query/destination-key
  policy head, win/draw/loss value head, shared per-square belief head)

Why it is written out rather than assembled from `nn.TransformerEncoder`
-----------------------------------------------------------------------
Two reasons, both about evidence rather than taste. The blocks below contain no
dropout at all, so there is no train/eval divergence to reason about when a test
claims two runs are bit-identical. And Phase 5 explicitly forbids adopting the
Phase 3 benchmark probe as the model design, so the fixture is an independent
module rather than a re-export of `stratego.training.representative_model`.

Shapes
------
.. code-block:: text

    tokens  [B, 100, 127]
        -> input projection + learned position embedding -> [B, 100, 64]
        -> 2 x (pre-norm self-attention + pre-norm feed-forward)
        -> final norm                                       [B, 100, 64]
        |-- policy: source query . destination key -> [B, 100, 100] -> [B, 10000]
        |-- value:  mean over tokens -> MLP           -> [B, 3]
        |-- belief: per token linear                  -> [B, 100, 12]

The policy head's `(100, 100)` matrix is flattened row-major, so entry
`(source, destination)` lands at index `100 * source + destination`. That is the
engine's action encoding exactly, which is why no remapping table exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch
from torch import nn

from .base import StrategoModel
from .contract import (
    BELIEF_TYPE_COUNT,
    MODEL_CONTRACT_VERSION,
    TOKEN_COUNT,
    TOKEN_FEATURES,
    VALUE_CLASS_COUNT,
    ModelContractError,
    ModelOutputs,
    validate_token_batch,
)
from .tokenization import observation_to_tokens

#: Identifier stored in every checkpoint. A different architecture must use a
#: different identifier; loading weights across identifiers is refused.
MODEL_ARCHITECTURE_ID = "integration_model_v1"

#: Machine-readable statement that this network is integration scaffolding.
IS_INTEGRATION_FIXTURE = True

#: Repeated in `architecture_summary()` so it reaches every report and JSON file.
FIXTURE_NOTE = (
    "integration_model_v1 is a Phase 5 integration fixture. It is untrained, its "
    "playing strength is meaningless, and it is not the final/production/Ataraxos "
    "model. Phase 6 selects the real architecture."
)


@dataclass(frozen=True)
class IntegrationModelConfig:
    """Shape of the fixture. Defaults are the Phase 5 instruction's defaults.

    These are a *default*, not a playing-strength choice: nothing in Phase 5
    tunes them, and section 5.7 forbids tuning them against the performance
    measurements.
    """

    num_tokens: int = TOKEN_COUNT
    input_features: int = TOKEN_FEATURES
    width: int = 64
    num_blocks: int = 2
    num_heads: int = 4
    feedforward_width: int = 256
    value_classes: int = VALUE_CLASS_COUNT
    belief_types: int = BELIEF_TYPE_COUNT

    def __post_init__(self) -> None:
        if self.width % self.num_heads:
            raise ModelContractError(
                f"width {self.width} must divide evenly across {self.num_heads} heads"
            )
        for name in ("num_tokens", "input_features", "width", "num_blocks", "num_heads",
                     "feedforward_width", "value_classes", "belief_types"):
            if getattr(self, name) < 1:
                raise ModelContractError(f"{name} must be positive, got {getattr(self, name)}")
        # The contract fixes these three; a config that disagrees would produce
        # tensors the boundary validators reject, so it is refused up front.
        if self.num_tokens != TOKEN_COUNT:
            raise ModelContractError(f"num_tokens must be {TOKEN_COUNT}, got {self.num_tokens}")
        if self.input_features != TOKEN_FEATURES:
            raise ModelContractError(
                f"input_features must be {TOKEN_FEATURES}, got {self.input_features}"
            )
        if self.value_classes != VALUE_CLASS_COUNT:
            raise ModelContractError(
                f"value_classes must be {VALUE_CLASS_COUNT}, got {self.value_classes}"
            )
        if self.belief_types != BELIEF_TYPE_COUNT:
            raise ModelContractError(
                f"belief_types must be {BELIEF_TYPE_COUNT}, got {self.belief_types}"
            )

    def to_dict(self) -> dict:
        return {
            "num_tokens": self.num_tokens,
            "input_features": self.input_features,
            "width": self.width,
            "num_blocks": self.num_blocks,
            "num_heads": self.num_heads,
            "feedforward_width": self.feedforward_width,
            "value_classes": self.value_classes,
            "belief_types": self.belief_types,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "IntegrationModelConfig":
        known = IntegrationModelConfig().to_dict()
        unexpected = sorted(set(payload) - set(known))
        if unexpected:
            raise ModelContractError(
                f"unknown model configuration field(s): {', '.join(unexpected)}"
            )
        merged = {key: int(payload.get(key, value)) for key, value in known.items()}
        return IntegrationModelConfig(**merged)

    def replace(self, **changes: int) -> "IntegrationModelConfig":
        return replace(self, **changes)


class _EncoderBlock(nn.Module):
    """Pre-norm self-attention followed by a pre-norm feed-forward. No dropout.

    Pre-norm (normalize *before* each sublayer, add the residual after) is the
    usual choice for small Transformers because it trains without a warmup
    schedule. Here it mostly just keeps activations well-scaled at initialisation
    so an untrained forward pass stays comfortably finite in float16.
    """

    def __init__(self, config: IntegrationModelConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.width,
            num_heads=config.num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(config.width)
        self.feedforward = nn.Sequential(
            nn.Linear(config.width, config.feedforward_width),
            nn.GELU(),
            nn.Linear(config.feedforward_width, config.width),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normed = self.attention_norm(hidden)
        # Every one of the 100 board squares may attend to every other; there is
        # no causal or padding mask, because a board position is not a sequence.
        attended, _ = self.attention(normed, normed, normed, need_weights=False)
        hidden = hidden + attended
        return hidden + self.feedforward(self.feedforward_norm(hidden))


class IntegrationModel(StrategoModel):
    """The Phase 5 fixture network. Untrained, and intended to stay untrained.

    Implements :class:`~stratego.model.base.StrategoModel` so that the Phase 6
    checkpoint registry and policy adapter can be written against an interface
    rather than against this class. The network itself is unchanged: the Phase 5
    fixture's weights, shapes and architecture id are all exactly as accepted.
    """

    is_integration_fixture = IS_INTEGRATION_FIXTURE
    architecture_id = MODEL_ARCHITECTURE_ID

    def __init__(self, config: "IntegrationModelConfig | None" = None, *, seed: int = 20250501):
        super().__init__()
        self.config = config or IntegrationModelConfig()
        self._init_seed = int(seed)
        cfg = self.config

        self.input_projection = nn.Linear(cfg.input_features, cfg.width)
        # One learned vector per board square. The board has a fixed geometry, so
        # a learned embedding is simpler and cheaper than a sinusoidal scheme.
        self.position_embedding = nn.Parameter(torch.zeros(1, cfg.num_tokens, cfg.width))
        self.blocks = nn.ModuleList(_EncoderBlock(cfg) for _ in range(cfg.num_blocks))
        self.encoder_norm = nn.LayerNorm(cfg.width)

        # Policy: each square produces a "move from here" query and a "move to
        # here" key; their dot product scores one (source, destination) pair.
        self.policy_source = nn.Linear(cfg.width, cfg.width)
        self.policy_destination = nn.Linear(cfg.width, cfg.width)
        self._policy_scale = 1.0 / math.sqrt(cfg.width)

        # Value: pooled position -> WIN / DRAW / LOSS.
        self.value_body = nn.Linear(cfg.width, cfg.width)
        self.value_head = nn.Linear(cfg.width, cfg.value_classes)

        # Belief: one 12-way distribution per square, sharing the encoder rather
        # than running a second network.
        self.belief_head = nn.Linear(cfg.width, cfg.belief_types)

        self.reset_parameters(seed=self._init_seed)
        # Deterministic inference is the default. There is no dropout to disable,
        # but `eval()` also makes the intent explicit to anyone reading a trace.
        self.eval()

    # -- construction ------------------------------------------------------

    def reset_parameters(self, *, seed: int | None = None) -> None:
        """Deterministic initialisation from an explicit CPU generator.

        Using an explicit `torch.Generator` rather than the global RNG means two
        processes produce identical weights without having to agree about global
        seeding, which is what makes the save/reload and cross-device gates
        meaningful.
        """
        if seed is not None:
            self._init_seed = int(seed)
        generator = torch.Generator(device="cpu").manual_seed(self._init_seed)
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    bound = 1.0 / math.sqrt(module.in_features)
                    module.weight.uniform_(-bound, bound, generator=generator)
                    if module.bias is not None:
                        module.bias.zero_()
                elif isinstance(module, nn.LayerNorm):
                    module.weight.fill_(1.0)
                    module.bias.zero_()
                elif isinstance(module, nn.MultiheadAttention):
                    bound = 1.0 / math.sqrt(module.embed_dim)
                    module.in_proj_weight.uniform_(-bound, bound, generator=generator)
                    if module.in_proj_bias is not None:
                        module.in_proj_bias.zero_()
                    module.out_proj.weight.uniform_(-bound, bound, generator=generator)
                    if module.out_proj.bias is not None:
                        module.out_proj.bias.zero_()
            self.position_embedding.normal_(mean=0.0, std=0.02, generator=generator)

    @property
    def initialisation_seed(self) -> int:
        return self._init_seed

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def architecture_summary(self) -> dict:
        """Serialisable description, carried into checkpoints and reports."""
        summary = dict(self.config.to_dict())
        summary.update(
            {
                "model_architecture_id": self.architecture_id,
                "model_contract_version": MODEL_CONTRACT_VERSION,
                "integration_fixture": IS_INTEGRATION_FIXTURE,
                "parameter_count": self.parameter_count(),
                "initialisation_seed": self._init_seed,
                "trained": False,
                "policy_head": (
                    "source query . destination key over 100x100, flattened "
                    "row-major to action_id = 100 * source + destination"
                ),
                "value_head": "mean-pooled tokens -> 3 logits (WIN, DRAW, LOSS)",
                "belief_head": "per-square linear -> 12 opponent piece-type logits",
                "note": FIXTURE_NOTE,
            }
        )
        return summary

    # -- forward -----------------------------------------------------------

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """`[B, 100, 127]` -> `[B, 100, width]` shared encoder state."""
        validate_token_batch(tokens)
        if tokens.dtype != self.input_projection.weight.dtype:
            # Casting here rather than raising keeps float16 inference callable
            # with a float32 input, which is how the precision gate feeds it.
            tokens = tokens.to(self.input_projection.weight.dtype)
        hidden = self.input_projection(tokens) + self.position_embedding
        for block in self.blocks:
            hidden = block(hidden)
        return self.encoder_norm(hidden)

    def forward(self, tokens: torch.Tensor) -> ModelOutputs:
        """One forward pass over tokenized observations. Returns validated heads."""
        hidden = self.encode(tokens)
        batch = hidden.shape[0]

        source = self.policy_source(hidden)
        destination = self.policy_destination(hidden)
        # (B, 100, W) x (B, W, 100) -> (B, 100, 100), row = source, column =
        # destination. `reshape` then flattens row-major into 10,000 entries.
        scores = torch.matmul(source, destination.transpose(1, 2)) * self._policy_scale
        policy_logits = scores.reshape(batch, self.config.num_tokens * self.config.num_tokens)

        pooled = hidden.mean(dim=1)
        value_logits = self.value_head(torch.nn.functional.gelu(self.value_body(pooled)))

        belief_logits = self.belief_head(hidden)

        return ModelOutputs.validated(policy_logits, value_logits, belief_logits, batch=batch)

    def forward_observation(self, observation: torch.Tensor) -> ModelOutputs:
        """Convenience path from the canonical `[B, 127, 10, 10]` input."""
        return self.forward(observation_to_tokens(observation))


def build_integration_model(
    config: "IntegrationModelConfig | None" = None,
    *,
    seed: int = 20250501,
    device: "torch.device | str" = "cpu",
    dtype: torch.dtype = torch.float32,
) -> IntegrationModel:
    """Build on CPU with fixed weights, then move and cast.

    Building on CPU first is what makes CPU and Metal comparisons meaningful:
    both devices then start from bit-identical float32 weights, so any
    difference measured later comes from the kernels, not from initialisation.
    """
    model = IntegrationModel(config, seed=seed)
    model = model.to(device=torch.device(device), dtype=dtype)
    model.eval()
    return model


__all__ = [
    "FIXTURE_NOTE",
    "IS_INTEGRATION_FIXTURE",
    "MODEL_ARCHITECTURE_ID",
    "IntegrationModel",
    "IntegrationModelConfig",
    "build_integration_model",
]
