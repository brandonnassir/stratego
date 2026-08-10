"""Temporary *representative* compact Transformer for the Phase 3 benchmark.

.. warning::

   **This is a benchmark probe, not the frozen model design.** It exists only
   to measure Apple Metal Performance Shaders inference cost, legality
   application cost and action-sampling cost at the shapes Phase 4 will
   actually use. It is never trained for playing strength, its weights carry
   no meaning, and the final architecture is selected later. Nothing here is
   a frozen contract.

Specification sources:

- `05_project_plan.md` section 5 (compact Transformer direction: 100 board
  tokens, width near 128, 4 layers, 4 heads, feedforward near 512, a
  source-query / destination-key policy head, a win/draw/loss value head and a
  shared belief head rather than a separate large belief Transformer)
- `06_observation_v2_127ch.md` sections 3 and 14 (the `(127, 10, 10)`
  observation and the *separate* 10,000-entry legality mask)
- `03_game_engine_spec.md` section 18 (only the coordinator touches Metal)

Model inputs
------------
The only tensors this module accepts are:

1. the approved `observation_v2_1_127ch` tensor, tokenised to `(B, 100, 127)`;
2. the engine's legality information, either as the dense `(B, 10000)` mask or
   as the compact padded legal-action-identifier form in :class:`CompactLegality`.

Privileged belief targets are training *labels* and never enter this module.
The belief head predicts them; it is not given them.

Action encoding
---------------
`action_id = 100 * source + destination`. The policy head scores a logical
`(B, 100, 100)` source-destination matrix whose row-major flattening is exactly
that identifier, so no permutation is needed anywhere in the path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from ..engine.constants import (
    ACTION_SPACE_SIZE,
    NUM_PIECE_TYPES,
    NUM_SQUARES,
    OBSERVATION_CHANNELS,
)

# Bumped whenever the probe's shapes or head structure change. It is *not* an
# engine or observation version and must never be treated as one.
REPRESENTATIVE_MODEL_VERSION = "representative_benchmark_probe_0.1.0"

#: Machine-readable statement that this network is throw-away benchmark scaffolding.
IS_BENCHMARK_PROBE = True

#: Win / draw / loss.
VALUE_CLASSES = 3


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepresentativeConfig:
    """Shape of the representative probe.

    Defaults are the Phase 1/2 planning target in `05_project_plan.md` section
    5. They are deliberately *not* tuned: the point is to measure the cost of
    the planned shape, not to search for a better one.
    """

    num_tokens: int = NUM_SQUARES
    input_features: int = OBSERVATION_CHANNELS
    width: int = 128
    num_layers: int = 4
    num_heads: int = 4
    feedforward_width: int = 512
    value_classes: int = VALUE_CLASSES
    belief_types: int = NUM_PIECE_TYPES

    def summary(self) -> dict:
        return {
            "benchmark_probe": IS_BENCHMARK_PROBE,
            "version": REPRESENTATIVE_MODEL_VERSION,
            "tokens": self.num_tokens,
            "input_features_per_token": self.input_features,
            "embedding_width": self.width,
            "transformer_layers": self.num_layers,
            "attention_heads": self.num_heads,
            "feedforward_width": self.feedforward_width,
            "policy_head": "source-query / destination-key over 100x100 -> 10000 logits",
            "value_head": f"{self.value_classes} logits (win, draw, loss)",
            "belief_head": (
                f"shared per-square placeholder -> ({self.num_tokens}, "
                f"{self.belief_types}) opponent-type logits"
            ),
            "action_encoding": "action_id = 100 * source + destination",
        }


@dataclass
class ModelOutputs:
    """One forward pass.

    `policy_logits` is the flattened source-destination matrix, `value_logits`
    is win/draw/loss and `belief_logits` is the shared-head placeholder.
    """

    policy_logits: torch.Tensor  # (B, 10000)
    value_logits: torch.Tensor  # (B, 3)
    belief_logits: torch.Tensor  # (B, 100, 12)

    def all_finite(self) -> bool:
        return bool(
            torch.isfinite(self.policy_logits).all()
            and torch.isfinite(self.value_logits).all()
            and torch.isfinite(self.belief_logits).all()
        )


# ---------------------------------------------------------------------------
# Observation tokenisation
# ---------------------------------------------------------------------------


def observation_to_tokens(observation: np.ndarray) -> np.ndarray:
    """`(127, 10, 10)` or `(B, 127, 10, 10)` -> `(…, 100, 127)` tokens.

    One token per board square, 127 raw features per token. This is a pure
    layout change: no channel is dropped, combined or reordered, so the
    approved observation contract is untouched.
    """
    array = np.asarray(observation)
    if array.ndim == 3:
        array = array[None, ...]
        squeeze = True
    elif array.ndim == 4:
        squeeze = False
    else:
        raise ValueError(
            f"expected (127, 10, 10) or (B, 127, 10, 10), got shape {array.shape}"
        )
    batch, channels = array.shape[0], array.shape[1]
    if channels != OBSERVATION_CHANNELS:
        raise ValueError(
            f"expected {OBSERVATION_CHANNELS} observation channels, got {channels}"
        )
    tokens = array.reshape(batch, channels, NUM_SQUARES).transpose(0, 2, 1)
    tokens = np.ascontiguousarray(tokens, dtype=np.float32)
    return tokens[0] if squeeze else tokens


# ---------------------------------------------------------------------------
# The probe network
# ---------------------------------------------------------------------------


class RepresentativeTransformer(nn.Module):
    """Compact encoder + policy / value / belief probes.

    Untrained and *intended* to stay untrained. Weights come from a fixed seed
    so a benchmark run is reproducible, not because the values mean anything.
    """

    is_benchmark_probe = IS_BENCHMARK_PROBE

    def __init__(self, config: RepresentativeConfig | None = None, *, seed: int = 0):
        super().__init__()
        self.config = config or RepresentativeConfig()
        cfg = self.config
        if cfg.width % cfg.num_heads:
            raise ValueError("embedding width must divide evenly across heads")

        generator = torch.Generator().manual_seed(seed)
        self._init_seed = seed

        self.input_projection = nn.Linear(cfg.input_features, cfg.width)
        self.position_embedding = nn.Parameter(torch.empty(1, cfg.num_tokens, cfg.width))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.width,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.feedforward_width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers, enable_nested_tensor=False
        )
        self.encoder_norm = nn.LayerNorm(cfg.width)

        # Policy probe: source queries against destination keys.
        self.policy_source = nn.Linear(cfg.width, cfg.width)
        self.policy_destination = nn.Linear(cfg.width, cfg.width)
        self._policy_scale = 1.0 / math.sqrt(cfg.width)

        # Value probe: pooled encoder state -> win / draw / loss.
        self.value_body = nn.Linear(cfg.width, cfg.width)
        self.value_head = nn.Linear(cfg.width, cfg.value_classes)

        # Belief probe: deliberately a lightweight shared head, *not* the
        # paper's separate large belief Transformer.
        self.belief_head = nn.Linear(cfg.width, cfg.belief_types)

        self._reset_parameters(generator)
        self.eval()

    # -- construction ------------------------------------------------------

    def _reset_parameters(self, generator: torch.Generator) -> None:
        """Deterministic weights from `generator`, on CPU, before any device move."""
        with torch.no_grad():
            for parameter in self.parameters():
                if parameter.dim() >= 2:
                    bound = 1.0 / math.sqrt(parameter.shape[-1])
                    parameter.uniform_(-bound, bound, generator=generator)
                else:
                    parameter.zero_()
            self.position_embedding.normal_(mean=0.0, std=0.02, generator=generator)
            for module in self.modules():
                if isinstance(module, nn.LayerNorm):
                    module.weight.fill_(1.0)
                    module.bias.zero_()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def architecture_summary(self) -> dict:
        summary = dict(self.config.summary())
        summary["parameter_count"] = self.parameter_count()
        summary["initialisation_seed"] = self._init_seed
        summary["trained"] = False
        summary["note"] = (
            "Representative benchmark probe only. Not the frozen model design; "
            "not trained for playing strength."
        )
        return summary

    # -- forward -----------------------------------------------------------

    def forward(self, tokens: torch.Tensor) -> ModelOutputs:
        """`tokens` is `(B, 100, 127)` in this module's parameter dtype."""
        if tokens.dim() != 3:
            raise ValueError(f"expected (B, 100, 127) tokens, got {tuple(tokens.shape)}")
        batch, num_tokens, features = tokens.shape
        cfg = self.config
        if num_tokens != cfg.num_tokens or features != cfg.input_features:
            raise ValueError(
                f"expected (B, {cfg.num_tokens}, {cfg.input_features}) tokens, "
                f"got {tuple(tokens.shape)}"
            )

        hidden = self.input_projection(tokens) + self.position_embedding
        hidden = self.encoder(hidden)
        hidden = self.encoder_norm(hidden)

        source = self.policy_source(hidden)
        destination = self.policy_destination(hidden)
        scores = torch.matmul(source, destination.transpose(1, 2)) * self._policy_scale
        policy_logits = scores.reshape(batch, cfg.num_tokens * cfg.num_tokens)

        pooled = hidden.mean(dim=1)
        value_logits = self.value_head(torch.nn.functional.gelu(self.value_body(pooled)))

        belief_logits = self.belief_head(hidden)

        return ModelOutputs(
            policy_logits=policy_logits,
            value_logits=value_logits,
            belief_logits=belief_logits,
        )


def build_representative_model(
    config: RepresentativeConfig | None = None,
    *,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> RepresentativeTransformer:
    """Build the probe on CPU with fixed weights, then move/cast it.

    Building on CPU first keeps the weights identical across devices and
    precisions, which is what makes the cross-device comparisons meaningful.
    """
    model = RepresentativeTransformer(config, seed=seed)
    model = model.to(device=torch.device(device), dtype=dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Legality application and action sampling
# ---------------------------------------------------------------------------
#
# Two representations of the same engine-supplied legal set are benchmarked:
#
# - *dense*: the engine's `(B, 10000)` mask, applied straight to the policy
#   logits;
# - *compact*: padded legal-action identifiers, `(B, capacity)`, gathered out of
#   the policy logits.
#
# Both must produce identical normalised probabilities over the legal set and
# both must only ever return actions the engine declared legal. The engine
# stays the sole authority on legality (`06_...` section 14); nothing here
# infers it.

#: Sampling and normalisation run in float32 even when the model runs in a
#: reduced precision. A 10,000-way categorical draw taken in float16 is not
#: numerically trustworthy, and the cast is inside every timed region, so a
#: reduced-precision result is never flattered by skipping it.
SAMPLING_DTYPE = torch.float32


@dataclass(frozen=True)
class CompactLegality:
    """Padded legal-action identifiers for a batch.

    `action_ids` is `(B, capacity)` int64 with arbitrary values in the padded
    tail; `valid` is the `(B, capacity)` bool mask saying which entries are
    real. `counts` is kept for reporting only.
    """

    action_ids: torch.Tensor
    valid: torch.Tensor
    counts: torch.Tensor

    @property
    def capacity(self) -> int:
        return int(self.action_ids.shape[1])

    def to(self, device: torch.device | str) -> "CompactLegality":
        device = torch.device(device)
        return CompactLegality(
            action_ids=self.action_ids.to(device),
            valid=self.valid.to(device),
            counts=self.counts.to(device),
        )

    def nbytes(self) -> int:
        return (
            self.action_ids.numel() * self.action_ids.element_size()
            + self.valid.numel() * self.valid.element_size()
        )


def build_compact_legality(
    legal_action_lists: Sequence[Sequence[int]],
    *,
    capacity: int | None = None,
) -> CompactLegality:
    """Pad `legal_action_lists` into the compact representation.

    Raises loudly when a row does not fit `capacity` rather than silently
    truncating a legal move away, which would be a correctness failure
    disguised as a performance win.
    """
    counts = [len(row) for row in legal_action_lists]
    if not counts:
        raise ValueError("compact legality needs at least one row")
    needed = max(counts)
    if capacity is None:
        capacity = max(needed, 1)
    if needed > capacity:
        raise ValueError(
            f"legal-action capacity {capacity} is too small for a state with "
            f"{needed} legal actions"
        )
    batch = len(legal_action_lists)
    ids = np.zeros((batch, capacity), dtype=np.int64)
    valid = np.zeros((batch, capacity), dtype=bool)
    for row, actions in enumerate(legal_action_lists):
        count = counts[row]
        if count:
            ids[row, :count] = np.asarray(actions, dtype=np.int64)
            valid[row, :count] = True
    return CompactLegality(
        action_ids=torch.from_numpy(ids),
        valid=torch.from_numpy(valid),
        counts=torch.tensor(counts, dtype=torch.int64),
    )


def dense_mask_to_bool(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Engine `(B, 10000)` uint8 mask -> bool tensor."""
    tensor = mask if isinstance(mask, torch.Tensor) else torch.from_numpy(np.asarray(mask))
    if tensor.dim() != 2 or tensor.shape[1] != ACTION_SPACE_SIZE:
        raise ValueError(
            f"expected (B, {ACTION_SPACE_SIZE}) legality mask, got {tuple(tensor.shape)}"
        )
    return tensor.to(torch.bool)


def _gumbel_noise(
    shape: tuple[int, ...],
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    uniform = torch.rand(shape, device=device, dtype=SAMPLING_DTYPE, generator=generator)
    # `torch.rand` draws from [0, 1), so *both* ends have to be excluded.
    #
    # The upper clamp keeps `1 - u` away from 0, where the inner `log` would
    # diverge. The lower clamp is what stops `u == 0` -- which `torch.rand` does
    # return, at a rate around 2**-24 -- from making `log1p(-u)` exactly 0 and
    # the outer `log` therefore `-inf`, which would give a noise of `+inf`.
    #
    # A `+inf` noise is not merely a large draw: added to the `-inf` that
    # `apply_dense_legality` writes at every illegal entry it produces `NaN`,
    # and `argmax` ranks `NaN` above every finite value, so the sample lands on
    # an action the engine declared illegal. At roughly one draw in 17 million
    # this is invisible in a short benchmark and near-certain in a sustained
    # self-play run; it was found by the frozen engine refusing the action.
    #
    # Clamping to [1e-7, 1 - 1e-7] bounds the noise to about +/-16.1, far inside
    # float32, and moves probability mass of 1e-7 at each tail.
    uniform = uniform.clamp(min=1e-7, max=1.0 - 1e-7)
    return -torch.log(-torch.log1p(-uniform))


def apply_dense_legality(
    policy_logits: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    """Illegal entries become `-inf` in float32."""
    return policy_logits.to(SAMPLING_DTYPE).masked_fill(~legal_mask, float("-inf"))


def dense_legal_probabilities(
    policy_logits: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    """Softmax over the legal set only; illegal entries are exactly zero."""
    masked = apply_dense_legality(policy_logits, legal_mask)
    return torch.softmax(masked, dim=1)


def sample_dense(
    policy_logits: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one legal `action_id` per row from the dense mask.

    Gumbel-max is exactly a categorical draw from the masked softmax and is a
    single fused-friendly kernel on every backend, including Metal.
    """
    masked = apply_dense_legality(policy_logits, legal_mask)
    noise = _gumbel_noise(tuple(masked.shape), masked.device, generator)
    return torch.argmax(masked + noise, dim=1)


def compact_gathered_logits(
    policy_logits: torch.Tensor, legality: CompactLegality
) -> torch.Tensor:
    """`(B, capacity)` float32 logits for the legal actions, `-inf` on padding."""
    gathered = policy_logits.to(SAMPLING_DTYPE).gather(1, legality.action_ids)
    return gathered.masked_fill(~legality.valid, float("-inf"))


def compact_legal_probabilities(
    policy_logits: torch.Tensor, legality: CompactLegality
) -> torch.Tensor:
    """Softmax over the padded legal set; padding is exactly zero."""
    return torch.softmax(compact_gathered_logits(policy_logits, legality), dim=1)


def sample_compact(
    policy_logits: torch.Tensor,
    legality: CompactLegality,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one legal `action_id` per row from the compact representation."""
    gathered = compact_gathered_logits(policy_logits, legality)
    noise = _gumbel_noise(tuple(gathered.shape), gathered.device, generator)
    chosen = torch.argmax(gathered + noise, dim=1, keepdim=True)
    return legality.action_ids.gather(1, chosen).squeeze(1)


def scatter_compact_probabilities(
    compact_probabilities: torch.Tensor, legality: CompactLegality
) -> torch.Tensor:
    """Compact `(B, capacity)` probabilities back onto the `(B, 10000)` space.

    Used only to prove the two legality paths agree; the compact path never
    needs this at run time.
    """
    batch = compact_probabilities.shape[0]
    dense = torch.zeros(
        (batch, ACTION_SPACE_SIZE),
        dtype=compact_probabilities.dtype,
        device=compact_probabilities.device,
    )
    dense.scatter_add_(
        1, legality.action_ids, compact_probabilities * legality.valid.to(compact_probabilities.dtype)
    )
    return dense
