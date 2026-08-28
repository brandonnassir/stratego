"""Phase 17 Agent 3: the decoder-only causal setup network.

Specification sources:

- `03_AGENT_3_AUTOREGRESSIVE_SETUP_NETWORK.md` section 2
- `00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md` section 7
- `reports/phase17/ataraxos_method_map_v1.md` rows S01, S02, S08, S09

Shape
-----
```text
tokens   [B, 41]   start token, then the 40 canonical row-major placements
outputs  [B, 40]   read at prefixes 0..39; prefix k has seen k placements
```

Position 40's outputs are never read: after the fortieth placement there is
nothing left to predict. The positional table still carries 41 rows because
the sequence the model is trained on is 41 long, and truncating the table
would make a 41-token forward pass silently impossible.

Why the attention is hand-rolled
--------------------------------
`nn.MultiheadAttention` reaches the same 66,048 parameters, but the causal
mask would then be an argument passed from outside the module. Section 2
makes causality a *required test*, and the cheapest way to keep that test
honest is for the mask to be constructed inside `forward` from the sequence
length alone, with no caller able to weaken it.

What may not enter
------------------
The only inputs are the canonical prefix and its position. No enemy setup,
family label, opponent identity, terminal outcome or future token reaches
inference; the inventory mask is *derived* from the prefix rather than
supplied, which is why :func:`inventory_mask_from_prefix` lives in
`setup_sampling` and takes nothing but the tokens.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ...engine.constants import NUM_PIECE_TYPES
from .setup_contract import (
    POSITIONAL_INIT_STD,
    SETUP_BLOCKS,
    SETUP_FEED_FORWARD_WIDTH,
    SETUP_HEADS,
    SETUP_MODEL_VERSION,
    SETUP_PARAMETER_TARGET,
    SETUP_PARAMETER_TOLERANCE,
    SETUP_PREFIXES,
    SETUP_SEQUENCE_LENGTH,
    SETUP_VOCABULARY,
    SETUP_WIDTH,
    Phase17SetupError,
)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a mask built from the input's own length."""

    def __init__(self, width: int = SETUP_WIDTH, heads: int = SETUP_HEADS) -> None:
        super().__init__()
        if width % heads:
            raise Phase17SetupError(f"width {width} is not divisible by {heads} heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        shape = (batch, length, self.heads, self.head_width)
        query = self.query(hidden).view(shape).transpose(1, 2)
        key = self.key(hidden).view(shape).transpose(1, 2)
        value = self.value(hidden).view(shape).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_width)
        # Built here, from `length`, so no caller can pass a weaker mask.
        causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch, length, width)
        return self.out(attended)


class DecoderBlock(nn.Module):
    """Pre-layernorm block: `x + attn(norm(x))`, then `x + ff(norm(x))`."""

    def __init__(
        self,
        width: int = SETUP_WIDTH,
        heads: int = SETUP_HEADS,
        feed_forward_width: int = SETUP_FEED_FORWARD_WIDTH,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(width, heads)
        self.feed_forward_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, feed_forward_width),
            nn.GELU(),
            nn.Linear(feed_forward_width, width),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.feed_forward(self.feed_forward_norm(hidden))


class SetupModelOutput(dict):
    """`{'piece_logits', 'wdl_logits', 'conditional_entropy'}`, all at 40 prefixes."""


class Phase17SetupModel(nn.Module):
    """The Phase 17 setup policy: 4 x 128 x 4 heads x 512 feed-forward.

    Three heads at every prefix, exactly as the paper's D.3 factorization:
    next-piece logits, W/D/L logits, and a scalar conditional-entropy
    prediction. The conditional-entropy scalar is in the *normalized* units
    `I/10` the paper's Eq. (1) regresses -- see method map row S07 and
    operator decision D4. It is not a nats-valued entropy and must never be
    compared against one.
    """

    version = SETUP_MODEL_VERSION

    def __init__(
        self,
        blocks: int = SETUP_BLOCKS,
        width: int = SETUP_WIDTH,
        heads: int = SETUP_HEADS,
        feed_forward_width: int = SETUP_FEED_FORWARD_WIDTH,
        vocabulary: int = SETUP_VOCABULARY,
        sequence_length: int = SETUP_SEQUENCE_LENGTH,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.width = width
        self.heads = heads
        self.feed_forward_width = feed_forward_width
        self.vocabulary = vocabulary
        self.sequence_length = sequence_length

        self.token_embedding = nn.Embedding(vocabulary, width)
        self.positional_embedding = nn.Embedding(sequence_length, width)
        self.layers = nn.ModuleList(
            DecoderBlock(width, heads, feed_forward_width) for _ in range(blocks)
        )
        self.final_norm = nn.LayerNorm(width)
        self.piece_head = nn.Linear(width, NUM_PIECE_TYPES)
        self.wdl_head = nn.Linear(width, 3)
        self.conditional_entropy_head = nn.Linear(width, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Paper Table 23: learned positional embeddings initialised at std 0.1."""
        nn.init.normal_(self.positional_embedding.weight, mean=0.0, std=POSITIONAL_INIT_STD)

    # -- forward ---------------------------------------------------------

    def forward(self, tokens: torch.Tensor) -> SetupModelOutput:
        """Run the causal stack over `[B, L]` tokens and read every prefix.

        `tokens[:, 0]` must be the start token; `tokens[:, 1 + k]` is the piece
        placed at canonical square `k`. The returned tensors are indexed by
        prefix: entry `k` is the model's output having seen `k` placements.
        """
        if tokens.dim() != 2:
            raise Phase17SetupError(f"expected [batch, length] tokens, got {tuple(tokens.shape)}")
        length = tokens.shape[1]
        if not 1 <= length <= self.sequence_length:
            raise Phase17SetupError(
                f"sequence length must be in 1..{self.sequence_length}, got {length}"
            )
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.positional_embedding(positions)[None]
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.final_norm(hidden)
        # Prefix k reads position k. Position 40 predicts nothing.
        readable = hidden[:, : min(length, SETUP_PREFIXES)]
        return SetupModelOutput(
            piece_logits=self.piece_head(readable),
            wdl_logits=self.wdl_head(readable),
            conditional_entropy=self.conditional_entropy_head(readable).squeeze(-1),
        )

    # -- identity --------------------------------------------------------

    def architecture(self) -> dict:
        return {
            "setup_model_version": self.version,
            "blocks": self.blocks,
            "width": self.width,
            "heads": self.heads,
            "feed_forward_width": self.feed_forward_width,
            "normalization": "pre_layernorm",
            "vocabulary": self.vocabulary,
            "sequence_length": self.sequence_length,
            "prefixes": SETUP_PREFIXES,
            "parameter_count": count_parameters(self),
            "parameter_target": SETUP_PARAMETER_TARGET,
        }


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def assert_architecture(model: Phase17SetupModel) -> dict:
    """Refuse a model outside Agent 1's frozen parameter tolerance.

    Section 2: *fail the architecture gate if it falls outside Agent 1's
    frozen tolerance rather than quietly changing widths*. The count is fully
    determined by 4/128/4/512 and the three heads, so the tolerance is zero
    and a mismatch always means the shape moved.
    """
    architecture = model.architecture()
    count = architecture["parameter_count"]
    low = SETUP_PARAMETER_TARGET - SETUP_PARAMETER_TOLERANCE
    high = SETUP_PARAMETER_TARGET + SETUP_PARAMETER_TOLERANCE
    if not low <= count <= high:
        raise Phase17SetupError(
            f"setup model has {count} trainable parameters, outside the frozen band "
            f"[{low}, {high}] around {SETUP_PARAMETER_TARGET}; change the contract with "
            "the operator rather than the widths"
        )
    expected = (SETUP_BLOCKS, SETUP_WIDTH, SETUP_HEADS, SETUP_FEED_FORWARD_WIDTH)
    actual = (model.blocks, model.width, model.heads, model.feed_forward_width)
    if actual != expected:
        raise Phase17SetupError(
            f"setup architecture is {actual}, contract freezes {expected}"
        )
    return architecture


def build_setup_model(device: str = "cpu", seed: int | None = None) -> Phase17SetupModel:
    """A fresh setup model on `device`, optionally from a fixed seed.

    Common contract section 4: the setup model is initialised from scratch --
    there is no setup lineage to carry forward and no library to warm-start
    from.
    """
    if seed is not None:
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(int(seed))
        try:
            model = Phase17SetupModel()
        finally:
            torch.random.set_rng_state(generator_state)
    else:
        model = Phase17SetupModel()
    assert_architecture(model)
    return model.to(device)


__all__ = [
    "CausalSelfAttention",
    "DecoderBlock",
    "Phase17SetupModel",
    "SetupModelOutput",
    "assert_architecture",
    "build_setup_model",
    "count_parameters",
]
