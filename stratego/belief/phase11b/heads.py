"""Phase 11B Agent 1 candidates: what gets attached to the frozen encoder.

Specification source: `01_AGENT_1_ATTACHED_BELIEF_HEAD.md` (Experiments 1A,
1B and the optional 1C).

```text
1A   frozen C1 feature [128] -> the existing 128->12 linear -> 12 logits
1B   frozen C1 feature [128] -> 512 -> 512 -> 12 logits
1C   penultimate tokens [100, 128] -> last C1 block -> norm -> 1B head
```

1A starts from the accepted weights on purpose
-----------------------------------------------
Agent 1's question is whether the Phase 11 weakness was "mainly insufficient
dedicated belief optimization or an undersized belief output head". Starting
1A from the accepted `belief_output` weights and optimizing *only* belief
cross-entropy answers the first half directly: whatever 1A gains over the
accepted head is what dedicated belief optimization was worth at fixed
capacity. A fresh initialization would have answered a different question.

One architecture per experiment, no sweep
------------------------------------------
Width 512 (the top of the instructed 256-512 family), GELU (the accepted
architecture family's activation), no dropout, no normalization beyond the
`encoder_norm` the frozen features already carry. These are choices, not
search results, and the report says so.
"""

from __future__ import annotations

import torch
from torch import nn

from .contract import C1_FEATURE_WIDTH, RANK_COUNT
from .features import LAYER_FINAL, LAYER_PENULTIMATE

#: The candidate identities. Used as leaderboard keys and file stems.
CANDIDATE_1A = "agent01_1a_existing_linear_head"
CANDIDATE_1B = "agent01_1b_attached_mlp_head"
CANDIDATE_1C = "agent01_1c_final_block_plus_mlp"

#: The one architecture chosen for 1B, and reused as 1C's head.
MLP_HIDDEN_WIDTHS = (512, 512)
ACTIVATION = "gelu"


class ExistingBeliefHead(nn.Module):
    """Experiment 1A: the accepted 128->12 mapping, and nothing else."""

    candidate_id = CANDIDATE_1A
    architecture = "linear(128->12)"
    feature_layer = LAYER_FINAL

    def __init__(self) -> None:
        super().__init__()
        self.belief_output = nn.Linear(C1_FEATURE_WIDTH, RANK_COUNT)

    @classmethod
    def from_accepted(cls, model) -> "ExistingBeliefHead":
        """Initialized from the accepted Phase 9 belief head's weights."""
        head = cls()
        with torch.no_grad():
            head.belief_output.weight.copy_(model.belief_output.weight)
            head.belief_output.bias.copy_(model.belief_output.bias)
        return head

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.belief_output(features)

    def belief_logits(self, token_features: torch.Tensor) -> torch.Tensor:
        """`[100, 128]` frozen features of one position -> `[100, 12]`."""
        return self.forward(token_features)


class AttachedBeliefMLP(nn.Module):
    """Experiment 1B: one modest nonlinear head on the frozen feature."""

    candidate_id = CANDIDATE_1B
    architecture = "mlp(128->512->512->12, gelu)"
    feature_layer = LAYER_FINAL

    def __init__(self, widths=MLP_HIDDEN_WIDTHS) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = C1_FEATURE_WIDTH
        for hidden in widths:
            layers.append(nn.Linear(width, hidden))
            layers.append(nn.GELU())
            width = hidden
        layers.append(nn.Linear(width, RANK_COUNT))
        self.body = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.body(features)

    def belief_logits(self, token_features: torch.Tensor) -> torch.Tensor:
        """`[100, 128]` frozen features of one position -> `[100, 12]`."""
        return self.forward(token_features)


class FinalBlockBeliefModel(nn.Module):
    """Experiment 1C: the last C1 block, unfrozen, plus the 1B head.

    Consumes the *penultimate* cache — the last block's input — so the
    frozen prefix is still computed exactly once. The block and the encoder
    norm are deep copies: the accepted model is not mutated.
    """

    candidate_id = CANDIDATE_1C
    architecture = "c1_last_block + " + AttachedBeliefMLP.architecture
    feature_layer = LAYER_PENULTIMATE

    def __init__(self, block: nn.Module, encoder_norm: nn.Module) -> None:
        super().__init__()
        import copy

        self.block = copy.deepcopy(block)
        self.encoder_norm = copy.deepcopy(encoder_norm)
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        self.head = AttachedBeliefMLP()

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """`[B, 100, 128]` penultimate tokens -> `[B, 100, 128]` features."""
        return self.encoder_norm(self.block(tokens))

    def forward(self, tokens: torch.Tensor, gather: torch.Tensor) -> torch.Tensor:
        """Encode a batch of positions and read the supervised tokens.

        `gather` is `[B, 100]` boolean or a `(row, square)` index pair; the
        trainer passes the index pair, which keeps the loss exactly per
        supervised piece rather than per padded square.
        """
        features = self.encode(tokens)
        return self.head(features[gather[0], gather[1]])

    def belief_logits(self, token_features: torch.Tensor) -> torch.Tensor:
        """`[100, 128]` penultimate tokens of one position -> `[100, 12]`.

        Runs the unfrozen block itself, which is why 1C declares the
        penultimate layer: its input is the frozen prefix's output.
        """
        return self.head(self.encode(token_features.unsqueeze(0))[0])


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    return int(
        sum(
            tensor.numel()
            for tensor in module.parameters()
            if tensor.requires_grad or not trainable_only
        )
    )


def build_candidate(candidate_id: str, frozen_model=None):
    """The candidate named by its leaderboard id."""
    if candidate_id == CANDIDATE_1A:
        if frozen_model is None:
            raise ValueError("1A initializes from the accepted head; pass frozen_model")
        return ExistingBeliefHead.from_accepted(frozen_model)
    if candidate_id == CANDIDATE_1B:
        return AttachedBeliefMLP()
    if candidate_id == CANDIDATE_1C:
        if frozen_model is None:
            raise ValueError("1C copies the accepted final block; pass frozen_model")
        from .features import final_block

        block, norm = final_block(frozen_model)
        return FinalBlockBeliefModel(block, norm)
    raise ValueError(f"unknown Phase 11B Agent 1 candidate {candidate_id!r}")


__all__ = [
    "ACTIVATION",
    "CANDIDATE_1A",
    "CANDIDATE_1B",
    "CANDIDATE_1C",
    "MLP_HIDDEN_WIDTHS",
    "AttachedBeliefMLP",
    "ExistingBeliefHead",
    "FinalBlockBeliefModel",
    "build_candidate",
    "parameter_count",
]
