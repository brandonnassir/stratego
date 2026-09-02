"""Phase 18: the decoder-only causal setup network (S01, S03, S30).

The architecture is the frozen 4 x 128 x 4-head x 512 pre-layernorm decoder
with exactly 802,320 trainable parameters -- the Phase 17 shape, rebuilt in the
Phase 18 namespace so the Phase 17 module is neither edited nor imported by
the training path. A test pins the two state-dict shapes to each other.

Shape
-----
```text
tokens   [B, 41]   start token, then the 40 canonical row-major placements
outputs  [B, 40]   read at prefixes 0..39; prefix k has seen k placements
```

The published `ArrangementTransformer.forward` prepends its start token
inside `forward` and truncates to 40 positions; here the caller supplies the
start token and position 40 is never read. The alignment is identical (S01):
position `t` predicts placement `t`.

Heads (S03, S08)
----------------
```text
piece_logits        [B, 40, 12]   12 live piece types; masking is applied by
                                  the caller from the prefix (S02, S04)
wdl_logits          [B, 40, 3]    order (loss, draw, win) = (0, 1, 2)
entropy_prediction  [B, 40]       the NORMALIZED suffix entropy, h ~ I/10
```

The published head is 14-way with `lake` and `empty` permanently masked; the
12-way head is that softmax restricted to the live classes, which changes the
parameter count by 514 and nothing else (S03).

Why the attention is hand-rolled
--------------------------------
The causal mask is built inside `forward` from the sequence length alone, so
no caller can weaken it and the causality test is honest.
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
    Phase18SetupError,
)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a mask built from the input's own length."""

    def __init__(self, width: int = SETUP_WIDTH, heads: int = SETUP_HEADS) -> None:
        super().__init__()
        if width % heads:
            raise Phase18SetupError(f"width {width} is not divisible by {heads} heads")
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
    """`{'piece_logits', 'wdl_logits', 'entropy_prediction'}` at 40 prefixes."""


class Phase18SetupModel(nn.Module):
    """The Phase 18 setup policy: 4 x 128 x 4 heads x 512 feed-forward."""

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
        self.entropy_head = nn.Linear(width, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Paper Table 23 / published `pos_emb_std`: positional init std 0.1."""
        nn.init.normal_(self.positional_embedding.weight, mean=0.0, std=POSITIONAL_INIT_STD)

    def forward(self, tokens: torch.Tensor) -> SetupModelOutput:
        """Run the causal stack over `[B, L]` tokens and read every prefix.

        `tokens[:, 0]` must be the start token; `tokens[:, 1 + k]` is the piece
        placed at canonical square `k`. Entry `k` of every output is the
        model's prediction having seen exactly `k` placements.
        """
        if tokens.dim() != 2:
            raise Phase18SetupError(f"expected [batch, length] tokens, got {tuple(tokens.shape)}")
        length = tokens.shape[1]
        if not 1 <= length <= self.sequence_length:
            raise Phase18SetupError(
                f"sequence length must be in 1..{self.sequence_length}, got {length}"
            )
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.positional_embedding(positions)[None]
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.final_norm(hidden)
        readable = hidden[:, : min(length, SETUP_PREFIXES)]
        return SetupModelOutput(
            piece_logits=self.piece_head(readable),
            wdl_logits=self.wdl_head(readable),
            entropy_prediction=self.entropy_head(readable).squeeze(-1),
        )

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
            "piece_classes": NUM_PIECE_TYPES,
            "wdl_classes": 3,
            "parameter_count": count_parameters(self),
            "parameter_target": SETUP_PARAMETER_TARGET,
            "positional_init_std": POSITIONAL_INIT_STD,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def assert_architecture(model: Phase18SetupModel) -> dict:
    """Refuse a model outside the frozen parameter band (tolerance zero)."""
    architecture = model.architecture()
    count = architecture["parameter_count"]
    low = SETUP_PARAMETER_TARGET - SETUP_PARAMETER_TOLERANCE
    high = SETUP_PARAMETER_TARGET + SETUP_PARAMETER_TOLERANCE
    if not low <= count <= high:
        raise Phase18SetupError(
            f"setup model has {count} trainable parameters, outside the frozen band "
            f"[{low}, {high}] around {SETUP_PARAMETER_TARGET}"
        )
    expected = (SETUP_BLOCKS, SETUP_WIDTH, SETUP_HEADS, SETUP_FEED_FORWARD_WIDTH)
    actual = (model.blocks, model.width, model.heads, model.feed_forward_width)
    if actual != expected:
        raise Phase18SetupError(f"setup architecture is {actual}, contract freezes {expected}")
    return architecture


def build_setup_model(device: str = "cpu", seed: int | None = None) -> Phase18SetupModel:
    """A fresh setup model on `device`, from a fixed seed when one is given.

    The seed drives `torch.manual_seed` around construction only, and the
    global RNG state is restored afterwards, so building a model never moves
    any other stream.
    """
    if seed is not None:
        state = torch.random.get_rng_state()
        torch.manual_seed(int(seed))
        try:
            model = Phase18SetupModel()
        finally:
            torch.random.set_rng_state(state)
    else:
        model = Phase18SetupModel()
    assert_architecture(model)
    return model.to(device)


def state_dict_digest(model: nn.Module) -> str:
    """The accepted parameter digest (`phase9_behavior.state_dict_digest`)."""
    from ..phase9_behavior import state_dict_digest as accepted_digest

    return accepted_digest(model)


__all__ = [
    "CausalSelfAttention",
    "DecoderBlock",
    "Phase18SetupModel",
    "SetupModelOutput",
    "assert_architecture",
    "build_setup_model",
    "count_parameters",
    "state_dict_digest",
]
