"""Phase 11B Agent 2: a dedicated spatial belief specialist on raw pixels.

Specification source: `02_AGENT_2_RAW_OBSERVATION_CNN.md` ("Model",
"Required Interface").

```text
public 127 x 10 x 10 observation
    -> 3x3 spatial projection to width 160
    -> 8 residual 3x3 convolution blocks
    -> per-square representation
    -> 12 rank logits per square
```

What this model is for
----------------------
Agent 1 measured that the frozen C1 feature is close to exhausted: a 216x
larger head bought `0.0036` `R_CE`, and unfreezing one encoder block bought
as much again. This model bypasses C1's learned compression entirely and
learns its own representation from the same public observation the policy
sees. The comparison is therefore *representation against representation*,
not head against head.

One configuration, declared not searched
-----------------------------------------
Width 160, 8 residual blocks, a 128-wide 1x1 read-out. That is
3,897,004 parameters, mid-band of the instructed 3-5M range, and it is the
only architecture Agent 2 trains: `02_AGENT_2` forbids an architecture
sweep, so these are choices, and the report records them as choices.

Given the band, depth was preferred to width. Each 3x3 convolution grows
the receptive field by one square in every direction, so the stem plus 8
two-convolution blocks — 17 convolutional layers — reaches 35x35: every
square sees the whole 10x10 board with margin, which a belief about a
*bomb behind a wall of scouts* needs and which a two- or three-layer tower
of the same parameter count could not express. The alternative in the band,
width 192 with 6 blocks, is 13 layers and 4.2M parameters; one had to be
chosen without a sweep and depth is the property this task argues for.

Public input only
-----------------
:meth:`RawObservationBeliefCNN.forward` takes exactly one argument, the
127-channel public observation, and the module holds no other input path.
A true rank cannot reach it: the corpus stores labels in a different
directory, the loader hands them over only when asked by name, and the
model's signature has nowhere to put them.

Read-out geometry
-----------------
The accepted `observation_to_tokens` flattens `(10, 10)` row-major into the
100 token indices, and the corpus stores each hidden piece's
`perspective_square` in exactly that index space. So a `[B, 12, 10, 10]`
convolution output reshaped to `[B, 12, 100]` and transposed is square-major
in the corpus's own coordinates, and :meth:`per_square_logits` is that
reshape and nothing else.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .contract import NUM_SQUARES, OBSERVATION_SHAPE, RANK_COUNT, Phase11BError
from .interface import Phase11BBeliefModel, Phase11BPublicState

#: The candidate identity. Leaderboard key, checkpoint stem, seed part.
CANDIDATE_2 = "agent02_raw_observation_cnn"

#: The one declared configuration.
RAW_CNN_WIDTH = 160
RAW_CNN_BLOCKS = 8
RAW_CNN_READOUT_WIDTH = 128
RAW_CNN_ACTIVATION = "relu"
RAW_CNN_NORMALIZATION = "batchnorm2d"

#: The architecture-family identity a checkpoint carries.
RAW_CNN_VERSION = "phase11b_raw_observation_cnn_v1"


class Phase11BRawCNNError(Phase11BError):
    """A raw-observation CNN was built or driven outside its contract."""


class ResidualBlock(nn.Module):
    """`conv3x3 -> norm -> relu -> [drop] -> conv3x3 -> norm -> (+ input) -> relu`.

    The standard pre-activation-free residual block of the board-game
    convolution family. Convolutions carry no bias because the
    normalization immediately after them has one.

    `dropout` is channel dropout (`Dropout2d`) on the block's hidden
    activation, and it is `0.0` by default so that the declared Agent 2
    architecture is exactly the one the instruction describes. It carries no
    parameters either way, so a checkpoint of one setting loads into the
    other.
    """

    def __init__(self, width: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(float(dropout)) if dropout else nn.Identity()

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self.activation(self.norm1(self.conv1(planes))))
        hidden = self.norm2(self.conv2(hidden))
        return self.activation(hidden + planes)


class RawObservationBeliefCNN(nn.Module):
    """The Agent 2 candidate: public observation in, 12 rank logits per square."""

    candidate_id = CANDIDATE_2
    architecture_version = RAW_CNN_VERSION

    def __init__(
        self,
        *,
        width: int = RAW_CNN_WIDTH,
        blocks: int = RAW_CNN_BLOCKS,
        readout_width: int = RAW_CNN_READOUT_WIDTH,
        block_dropout: float = 0.0,
        readout_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels = OBSERVATION_SHAPE[0]
        self.width = int(width)
        self.blocks_count = int(blocks)
        self.readout_width = int(readout_width)
        self.block_dropout = float(block_dropout)
        self.readout_dropout = float(readout_dropout)
        self.stem = nn.Sequential(
            nn.Conv2d(channels, self.width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(self.width, dropout=self.block_dropout)
                for _ in range(self.blocks_count)
            ]
        )
        # The dropout module is *inserted* rather than always present, so a
        # zero-dropout model keeps the positional `state_dict` keys of the
        # architecture as first declared and the two runs' checkpoints stay
        # mutually loadable at their own settings.
        readout: list[nn.Module] = []
        if self.readout_dropout:
            readout.append(nn.Dropout2d(self.readout_dropout))
        readout += [
            nn.Conv2d(self.width, self.readout_width, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.readout_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.readout_width, RANK_COUNT, kernel_size=1),
        ]
        self.readout = nn.Sequential(*readout)

    @property
    def architecture(self) -> str:
        drop = (
            f", dropout {self.block_dropout:g}/{self.readout_dropout:g}"
            if (self.block_dropout or self.readout_dropout)
            else ""
        )
        return (
            f"raw_cnn(127 -> conv3x3 {self.width} -> {self.blocks_count} x residual3x3 "
            f"-> 1x1 {self.readout_width} -> {RANK_COUNT}/square, "
            f"{RAW_CNN_ACTIVATION}, {RAW_CNN_NORMALIZATION}{drop})"
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """`[B, 127, 10, 10]` public observations -> `[B, 12, 10, 10]` logits."""
        if observations.dim() != 4 or tuple(observations.shape[1:]) != OBSERVATION_SHAPE:
            raise Phase11BRawCNNError(
                f"observations must be [B, {', '.join(str(dim) for dim in OBSERVATION_SHAPE)}], "
                f"got {tuple(observations.shape)}"
            )
        return self.readout(self.blocks(self.stem(observations)))

    def per_square_logits(self, observations: torch.Tensor) -> torch.Tensor:
        """`[B, 127, 10, 10]` -> `[B, 100, 12]` in accepted token order.

        Row-major, matching `observation_to_tokens`, so square index `s` is
        the corpus's `perspective_square` `s` and nothing has to be
        translated at the call site.
        """
        planes = self.forward(observations)
        return planes.reshape(planes.shape[0], RANK_COUNT, NUM_SQUARES).transpose(1, 2)

    def logits_at(
        self, observations: torch.Tensor, rows: torch.Tensor, squares: torch.Tensor
    ) -> torch.Tensor:
        """`[K, 12]` — the logits of `K` supervised pieces of this batch.

        `rows` indexes into the batch and `squares` into the 100 tokens, so
        the loss is exactly per hidden piece rather than per padded square.
        """
        return self.per_square_logits(observations)[rows, squares]


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    return int(
        sum(
            tensor.numel()
            for tensor in module.parameters()
            if tensor.requires_grad or not trainable_only
        )
    )


def parameter_breakdown(model: RawObservationBeliefCNN) -> dict:
    """Where the parameters are, so the count can be checked by hand."""
    return {
        "stem": parameter_count(model.stem),
        "residual_tower": parameter_count(model.blocks),
        "readout": parameter_count(model.readout),
        "total": parameter_count(model),
        "width": model.width,
        "blocks": model.blocks_count,
        "readout_width": model.readout_width,
        "block_dropout": model.block_dropout,
        "readout_dropout": model.readout_dropout,
        "conv3x3_layers": 1 + 2 * model.blocks_count,
        "receptive_field_width_squares": 1 + 2 * (1 + 2 * model.blocks_count),
    }


def build_raw_cnn(*, seed: "int | None" = None, **kwargs) -> RawObservationBeliefCNN:
    """The declared Agent 2 architecture, initialized from a named stream."""
    if seed is not None:
        torch.manual_seed(int(seed) % (2**63))
    return RawObservationBeliefCNN(**kwargs)


def load_raw_cnn(path, *, map_location: str = "cpu") -> tuple:
    """`(model, payload)` — a saved candidate, rebuilt from its own record.

    Dropout carries no parameters but a read-out dropout module shifts the
    positional `state_dict` keys of `nn.Sequential`, so a checkpoint has to
    be rebuilt at the shape it was saved with rather than at the default.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    shape = payload.get("parameter_breakdown", {})
    model = RawObservationBeliefCNN(
        width=int(shape.get("width", RAW_CNN_WIDTH)),
        blocks=int(shape.get("blocks", RAW_CNN_BLOCKS)),
        readout_width=int(shape.get("readout_width", RAW_CNN_READOUT_WIDTH)),
        block_dropout=float(shape.get("block_dropout", 0.0)),
        readout_dropout=float(shape.get("readout_dropout", 0.0)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


# ---------------------------------------------------------------------------
# The required belief interface
# ---------------------------------------------------------------------------


class _PerSquareReadout(nn.Module):
    """Identity read-out: this candidate already emits 12 logits per square.

    `Phase11BBeliefModel.predict_marginals` reads
    `head.belief_logits(features)[squares]`. Agent 1's heads consume a
    `[100, 128]` C1 feature there; Agent 2 has no C1 stage at all and its
    "feature" *is* the `[100, 12]` logit field, so the head is the identity
    and the accepted marginal/sampling path above it is reused verbatim.
    """

    #: Agent 2 reads no C1 cache layer. Declared so the attribute exists.
    feature_layer = None

    def belief_logits(self, per_square_logits: torch.Tensor) -> torch.Tensor:
        return per_square_logits


class RawObservationBeliefModel(Phase11BBeliefModel):
    """The Agent 2 CNN behind the shared Phase 11B belief interface.

    Subclasses the Agent 1 interface rather than reimplementing it: only
    the two lines that produce a per-square logit field differ, and
    `sample_worlds` — the part that has to go through the **accepted,
    unmodified** Phase 11 sampler — is inherited unchanged.
    """

    def __init__(
        self,
        model: RawObservationBeliefCNN,
        *,
        candidate_id: str = CANDIDATE_2,
        device: str = "cpu",
    ) -> None:
        super().__init__(model, _PerSquareReadout(), candidate_id=candidate_id, device=device)
        self.cnn = model.to(torch.device(device)).eval()

    def _features(self, state: Phase11BPublicState) -> torch.Tensor:
        """`[100, 12]` — this candidate's logit field for one public state.

        The only input is `state.observation`, the same public tensor the
        corpus stores; nothing else about the position reaches the model.
        """
        batch = torch.from_numpy(np.array(state.observation, dtype=np.float32, copy=True))
        with torch.no_grad():
            logits = self.cnn(batch.unsqueeze(0).to(self.device))
        return logits[0].reshape(RANK_COUNT, NUM_SQUARES).transpose(0, 1)

    def describe(self) -> dict:
        return {
            **super().describe(),
            "architecture": self.cnn.architecture,
            "architecture_version": RAW_CNN_VERSION,
            "consumes_c1_features": False,
            "consumes_public_observation": True,
            "parameters": parameter_count(self.cnn),
        }


__all__ = [
    "CANDIDATE_2",
    "RAW_CNN_ACTIVATION",
    "RAW_CNN_BLOCKS",
    "RAW_CNN_NORMALIZATION",
    "RAW_CNN_READOUT_WIDTH",
    "RAW_CNN_VERSION",
    "RAW_CNN_WIDTH",
    "Phase11BRawCNNError",
    "RawObservationBeliefCNN",
    "RawObservationBeliefModel",
    "ResidualBlock",
    "build_raw_cnn",
    "load_raw_cnn",
    "parameter_breakdown",
    "parameter_count",
]
