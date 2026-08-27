"""Phase 15 Agent 1 section 8: B18 and B24.

Specification source: `01_AGENT_1_BELIEF_HEAD_TRAINING.md` section 8.

```text
frozen  P18/P24 prefix   first three C1 transformer blocks
trained copy             final C1 block + encoder norm
fresh   belief MLP       128 -> 512 -> 512 -> 12, GELU
```

The Phase 11B 1C pattern, bound independently
----------------------------------------------
This is architecturally the successful Phase 11B Agent 1C candidate — a
deep copy of the last encoder block, a deep copy of the encoder norm, and a
two-hidden-layer GELU MLP on top. It is re-expressed in this namespace
rather than imported because the two specialists must bind to *different*
backbones and carry their own calibration temperature and source identity,
and because a Phase 15 checkpoint must be loadable without the Phase 11B
corpus module that lives beside `phase11b/heads.py`.

The deployed move model is untouched
------------------------------------
`P18` and `P24` remain the objects the search will call for policy and
value. A :class:`Phase15BeliefSpecialist` holds *copies*: mutating one
cannot change the source, and the source's parameters are never handed to
an optimizer. The specialist's own policy and value heads do not exist at
all — there is nothing to accidentally read.

Why the last block is trainable
-------------------------------
The frozen prefix's output is a constant of the corpus, so it is cached
once (see :mod:`.features`) and every epoch is then just the last block and
the MLP. Unfreezing the last block is what made 1C the strongest Phase 11B
candidate; keeping the first three frozen is what keeps the cache valid.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from .contract import (
    C1_FEATURE_WIDTH,
    MLP_ACTIVATION,
    MLP_HIDDEN_WIDTHS,
    RANK_COUNT,
    Phase15Error,
)

#: The architecture identity a checkpoint records.
BELIEF_ARCHITECTURE_VERSION = "phase15_belief_specialist_v1"

#: How many trailing encoder blocks the specialist owns a copy of.
TRAINABLE_BLOCKS = 1


class Phase15HeadError(Phase15Error):
    """A belief specialist could not be built or bound."""


class BeliefMLP(nn.Module):
    """`128 -> 512 -> 512 -> 12`, GELU. Freshly initialized, always."""

    architecture = (
        f"mlp({C1_FEATURE_WIDTH}->"
        + "->".join(str(width) for width in MLP_HIDDEN_WIDTHS)
        + f"->{RANK_COUNT}, {MLP_ACTIVATION})"
    )

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


class Phase15BeliefSpecialist(nn.Module):
    """B18 or B24: a trainable tail on a frozen policy backbone.

    Consumes the *penultimate* representation — the input to the source
    model's last encoder block — so the frozen three-block prefix is
    computed exactly once per position and cached.
    """

    architecture_version = BELIEF_ARCHITECTURE_VERSION

    def __init__(self, block: nn.Module, encoder_norm: nn.Module, *, specialist_id: str) -> None:
        super().__init__()
        self.specialist_id = str(specialist_id)
        self.block = copy.deepcopy(block)
        self.encoder_norm = copy.deepcopy(encoder_norm)
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        self.head = BeliefMLP()
        # A positive scalar temperature, fitted after training on the
        # calibration split. Stored as its log so any optimizer or load
        # path keeps it positive by construction; 1.0 means uncalibrated.
        self.register_buffer("log_temperature", torch.zeros((), dtype=torch.float32))

    # -- construction ------------------------------------------------------

    @classmethod
    def from_policy(cls, policy_model, *, specialist_id: str) -> "Phase15BeliefSpecialist":
        """Bind a fresh specialist to one frozen policy backbone."""
        blocks = list(policy_model.blocks)
        if len(blocks) <= TRAINABLE_BLOCKS:  # pragma: no cover - C1 has four
            raise Phase15HeadError(
                f"the backbone has {len(blocks)} blocks; at least "
                f"{TRAINABLE_BLOCKS + 1} are needed to keep a frozen prefix"
            )
        return cls(blocks[-1], policy_model.encoder_norm, specialist_id=specialist_id)

    # -- temperature -------------------------------------------------------

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature))

    def set_temperature(self, value: float) -> None:
        value = float(value)
        if not value > 0.0:
            raise Phase15HeadError(f"temperature must be positive, got {value}")
        with torch.no_grad():
            self.log_temperature.fill_(float(torch.log(torch.tensor(value))))

    # -- forward -----------------------------------------------------------

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """`[B, 100, 128]` penultimate tokens -> `[B, 100, 128]` features."""
        return self.encoder_norm(self.block(tokens))

    def forward(self, tokens: torch.Tensor, gather) -> torch.Tensor:
        """Encode a batch of positions and read the supervised tokens.

        `gather` is a `(row, square)` index pair, so the loss is exactly per
        supervised hidden piece rather than per padded square. Returns raw
        logits: the calibration temperature is applied by
        :meth:`calibrated_logits`, never inside the training loss.
        """
        features = self.encode(tokens)
        return self.head(features[gather[0], gather[1]])

    def belief_logits(self, token_features: torch.Tensor) -> torch.Tensor:
        """`[100, 128]` penultimate tokens of one position -> `[100, 12]`."""
        return self.head(self.encode(token_features.unsqueeze(0))[0])

    def calibrated_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Logits divided by the fitted temperature.

        Temperature scaling is a strictly positive rescale, so it cannot
        reorder a row: the top-1 label is identical before and after.
        """
        return logits / torch.exp(self.log_temperature)

    # -- identity ----------------------------------------------------------

    def parameter_counts(self) -> dict:
        def count(module: nn.Module) -> int:
            return int(sum(tensor.numel() for tensor in module.parameters()))

        return {
            "block": count(self.block),
            "encoder_norm": count(self.encoder_norm),
            "head": count(self.head),
            "total": count(self),
            "trainable": int(
                sum(
                    tensor.numel()
                    for tensor in self.parameters()
                    if tensor.requires_grad
                )
            ),
        }

    def describe(self) -> dict:
        return {
            "architecture_version": BELIEF_ARCHITECTURE_VERSION,
            "specialist_id": self.specialist_id,
            "architecture": (
                f"c1_last_block + encoder_norm + {BeliefMLP.architecture}"
            ),
            "frozen_prefix_blocks": "first three C1 transformer blocks of the source",
            "trainable_blocks": TRAINABLE_BLOCKS,
            "holds_policy_parameters": False,
            "holds_value_parameters": False,
            "temperature": self.temperature,
            "parameters": self.parameter_counts(),
        }


def trainable_parameter_groups(
    model: Phase15BeliefSpecialist, *, head_lr: float, block_lr: float
) -> list:
    """The two optimizer groups of section 9's recipe.

    Every tensor in both groups belongs to the specialist. No policy or
    value parameter can enter, because the specialist holds none — which is
    a stronger guarantee than filtering them out would be.
    """
    block = list(model.block.parameters()) + list(model.encoder_norm.parameters())
    head = list(model.head.parameters())
    return [
        {"params": block, "lr": float(block_lr), "name": "final_block"},
        {"params": head, "lr": float(head_lr), "name": "belief_head"},
    ]


__all__ = [
    "BELIEF_ARCHITECTURE_VERSION",
    "TRAINABLE_BLOCKS",
    "BeliefMLP",
    "Phase15BeliefSpecialist",
    "Phase15HeadError",
    "trainable_parameter_groups",
]
