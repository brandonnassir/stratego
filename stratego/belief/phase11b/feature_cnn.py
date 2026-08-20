"""Phase 11B Agent 3: a belief CNN over the frozen C1 per-square field.

Specification source: `03_AGENT_3_C1_FEATURE_CNN.md` ("Model", "Training",
"Required Interface").

```text
public observation
      ->
frozen C1                      (accepted Phase 9 weights, never updated)
      ->
per-square C1 field [100, 128] (ProductionModel.encode, all tokens)
      ->
residual belief CNN
      ->
12 rank logits per unresolved square
```

Deliberately Agent 2's tower, on a different representation
------------------------------------------------------------
`03_AGENT_3` asks a comparative question — is the final C1 representation
still carrying belief-relevant information, or was the tiny Phase 11 head
the extraction bottleneck? — and it wants that read against Agent 2's
raw-observation CNN. A comparison like that is only clean if the *specialist
is held fixed and the representation is the thing that changes*. So this
model is Agent 2's declared architecture with one difference: its input is
the frozen C1 field's 128 channels instead of the observation's 127.

Concretely, :class:`ResidualBlock` is imported from `raw_cnn` rather than
re-declared, the width is 160, the depth is 8 residual blocks, and the
read-out is the same 1x1 pair. Agent 2 is 3,897,004 parameters; this is
3,898,444 — the 1,440 difference is the stem's one extra input channel and
nothing else. **No sweep**: width, depth and read-out width were not chosen
here at all, they were inherited so that the difference between the two
reports is the representation.

Why a convolution tower on top of an encoder that already attends globally
---------------------------------------------------------------------------
C1's six transformer blocks give every square global context, so a further
tower is not adding reach. What it adds is *belief-specific* nonlinear
capacity over a representation trained for policy, value and belief
together: Agent 1 measured that a 216x larger per-piece head bought only
0.0036 `R_CE`, but a per-piece head cannot combine one square's feature with
its neighbours'. This model can. If that still buys nothing, the honest
reading is that the seam itself is the limit rather than the head on it.

The specialist never sees the raw observation
----------------------------------------------
`03_AGENT_3`: "Do not feed raw observation into the specialist. That is
Agent 4's experiment." :meth:`C1FeatureBeliefCNN.forward` takes exactly one
argument, the `[B, 100, 128]` frozen field, and the module holds no other
input path. The 127-channel observation reaches it only through the frozen
encoder, which is what makes this an experiment about C1's representation.

C1 stays frozen
---------------
The encoder is loaded through `features.load_frozen_c1`, which checks the
accepted state and belief-head digests and sets `requires_grad=False` on
every parameter, and the trained field cache means C1 is not even *called*
during training. No optimizer in this module is ever handed a C1 parameter:
:func:`build_feature_cnn` returns the specialist alone, and that is the only
object the trainer sees.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .contract import C1_FEATURE_WIDTH, NUM_SQUARES, RANK_COUNT, Phase11BError
from .features import LAYER_FINAL, encode_batch
from .feature_seam import SEAM_ID, field_to_planes
from .interface import Phase11BBeliefModel
from .raw_cnn import ResidualBlock, parameter_count
from .raw_train import _synchronize

#: The candidate identity. Leaderboard key, checkpoint stem, seed part.
CANDIDATE_3 = "agent03_c1_feature_cnn"

#: The one declared configuration — Agent 2's, inherited unchanged so the
#: two candidates differ in representation rather than in capacity.
FEATURE_CNN_WIDTH = 160
FEATURE_CNN_BLOCKS = 8
FEATURE_CNN_READOUT_WIDTH = 128
FEATURE_CNN_ACTIVATION = "relu"
FEATURE_CNN_NORMALIZATION = "batchnorm2d"

#: The architecture-family identity a checkpoint carries.
FEATURE_CNN_VERSION = "phase11b_c1_feature_cnn_v1"


class Phase11BFeatureCNNError(Phase11BError):
    """A C1-feature CNN was built or driven outside its contract."""


class C1FeatureBeliefCNN(nn.Module):
    """The Agent 3 candidate: frozen C1 field in, 12 rank logits per square.

    `feature_layer` is the hook the shared Phase 11B belief interface reads
    to decide which frozen tensor to compute for a live position. It is
    `LAYER_FINAL` — the accepted `encode` output — which is exactly the
    tensor this model was trained on, so the training path and the
    deployed path cannot drift apart.
    """

    candidate_id = CANDIDATE_3
    architecture_version = FEATURE_CNN_VERSION
    seam_id = SEAM_ID
    feature_layer = LAYER_FINAL

    def __init__(
        self,
        *,
        width: int = FEATURE_CNN_WIDTH,
        blocks: int = FEATURE_CNN_BLOCKS,
        readout_width: int = FEATURE_CNN_READOUT_WIDTH,
        block_dropout: float = 0.0,
        readout_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.blocks_count = int(blocks)
        self.readout_width = int(readout_width)
        self.block_dropout = float(block_dropout)
        self.readout_dropout = float(readout_dropout)
        self.stem = nn.Sequential(
            nn.Conv2d(C1_FEATURE_WIDTH, self.width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(self.width, dropout=self.block_dropout)
                for _ in range(self.blocks_count)
            ]
        )
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
            f"c1_field({C1_FEATURE_WIDTH}) -> conv3x3 {self.width} -> "
            f"{self.blocks_count} x residual3x3 -> 1x1 {self.readout_width} -> "
            f"{RANK_COUNT}/square, {FEATURE_CNN_ACTIVATION}, "
            f"{FEATURE_CNN_NORMALIZATION}{drop}"
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """`[B, 100, 128]` frozen C1 field -> `[B, 12, 10, 10]` logits.

        The input signature *is* the seam: this model consumes precisely
        what `ProductionModel.encode` emits, in the accepted token order,
        and has no second argument a raw observation could arrive in.
        """
        if field.dim() != 3 or tuple(field.shape[1:]) != (NUM_SQUARES, C1_FEATURE_WIDTH):
            raise Phase11BFeatureCNNError(
                f"field must be [B, {NUM_SQUARES}, {C1_FEATURE_WIDTH}], "
                f"got {tuple(field.shape)}"
            )
        return self.readout(self.blocks(self.stem(field_to_planes(field))))

    def per_square_logits(self, field: torch.Tensor) -> torch.Tensor:
        """`[B, 100, 128]` -> `[B, 100, 12]` in accepted token order.

        Row-major, matching `observation_to_tokens`, so square index `s` is
        the corpus's `perspective_square` `s`.
        """
        planes = self.forward(field)
        return planes.reshape(planes.shape[0], RANK_COUNT, NUM_SQUARES).transpose(1, 2)

    def logits_at(
        self, field: torch.Tensor, rows: torch.Tensor, squares: torch.Tensor
    ) -> torch.Tensor:
        """`[K, 12]` — the logits of `K` supervised pieces of this batch.

        The same `(rows, squares)` gather Agent 2's trainer produces, so the
        loss is exactly per hidden piece and in the corpus's own order.
        """
        return self.per_square_logits(field)[rows, squares]

    def belief_logits(self, token_features: torch.Tensor) -> torch.Tensor:
        """`[100, 128]` frozen field of one position -> `[100, 12]`.

        The hook `Phase11BBeliefModel.predict_marginals` calls. Its shape
        contract is Agent 1's, so the accepted marginal and world-sampling
        path above it is inherited rather than reimplemented.
        """
        return self.per_square_logits(token_features.unsqueeze(0))[0]


def parameter_breakdown(model: C1FeatureBeliefCNN) -> dict:
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
        "input_channels": C1_FEATURE_WIDTH,
        "conv3x3_layers": 1 + 2 * model.blocks_count,
        "frozen_c1_parameters": 863_959,
        "trainable_c1_parameters": 0,
    }


def build_feature_cnn(*, seed: "int | None" = None, **kwargs) -> C1FeatureBeliefCNN:
    """The declared Agent 3 architecture, initialized from a named stream.

    Returns the specialist **alone**. C1 is not part of this object and is
    therefore not part of any optimizer built from its parameters — the
    freeze is structural, not a convention the trainer has to remember.
    """
    if seed is not None:
        torch.manual_seed(int(seed) % (2**63))
    return C1FeatureBeliefCNN(**kwargs)


def load_feature_cnn(path, *, map_location: str = "cpu") -> tuple:
    """`(model, payload)` — a saved candidate, rebuilt from its own record."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    shape = payload.get("parameter_breakdown", {})
    model = C1FeatureBeliefCNN(
        width=int(shape.get("width", FEATURE_CNN_WIDTH)),
        blocks=int(shape.get("blocks", FEATURE_CNN_BLOCKS)),
        readout_width=int(shape.get("readout_width", FEATURE_CNN_READOUT_WIDTH)),
        block_dropout=float(shape.get("block_dropout", 0.0)),
        readout_dropout=float(shape.get("readout_dropout", 0.0)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


# ---------------------------------------------------------------------------
# Training over the cached field
# ---------------------------------------------------------------------------


def feature_split_view(data: dict, field) -> dict:
    """A stored split whose model input is the frozen C1 field.

    Agent 2's trainer is representation-agnostic by construction: it stages
    `data["observations"]` as one tensor, indexes it by sample row, and
    hands batches to `model.logits_at`. Nothing in it knows or cares whether
    those rows are 127-channel observations or 128-wide C1 fields.

    So Agent 3 does not fork the trainer. It hands the same trainer a *view*
    of the split whose input array is the cached field, which means Agent 2
    and Agent 3 share not only an architecture but an optimizer, a shuffling
    scheme, a probe schedule and a checkpoint-selection rule — and the
    difference between their two numbers is the representation.

    Every other key is passed through unchanged, so the labels, the piece
    offsets, the strata and the baseline arrays are literally the same
    objects the corpus loader produced.
    """
    field = np.asarray(field)
    samples = int(data["samples"])
    if field.shape != (samples, NUM_SQUARES, C1_FEATURE_WIDTH):
        raise Phase11BFeatureCNNError(
            f"field is {field.shape}, expected {(samples, NUM_SQUARES, C1_FEATURE_WIDTH)}"
        )
    view = dict(data)
    view["observations"] = field
    view["model_input"] = "frozen_c1_field"
    view["public_observations"] = data["observations"]
    return view


# ---------------------------------------------------------------------------
# The required belief interface
# ---------------------------------------------------------------------------


class C1FeatureBeliefModel(Phase11BBeliefModel):
    """The Agent 3 CNN behind the shared Phase 11B belief interface.

    Subclasses Agent 1's interface rather than reimplementing it. The
    encoder slot holds the frozen C1 and the head slot holds the belief CNN,
    so `predict_marginals` computes the seam once per position and
    `sample_worlds` — the part that must go through the **accepted,
    unmodified** Phase 11 sampler — is inherited code, not a fork.
    """

    def __init__(
        self,
        frozen_c1,
        model: C1FeatureBeliefCNN,
        *,
        candidate_id: str = CANDIDATE_3,
        device: str = "cpu",
    ) -> None:
        target = torch.device(device)
        super().__init__(
            frozen_c1.to(target), model.to(target), candidate_id=candidate_id, device=device
        )
        self.cnn = self.head

    def describe(self) -> dict:
        return {
            **super().describe(),
            "architecture": self.cnn.architecture,
            "architecture_version": FEATURE_CNN_VERSION,
            "seam_id": SEAM_ID,
            "consumes_c1_features": True,
            "consumes_public_observation": False,
            "c1_frozen": True,
            "parameters": parameter_count(self.cnn),
        }


@torch.no_grad()
def inference_cost(
    model: C1FeatureBeliefCNN,
    frozen_c1,
    field: np.ndarray,
    observations: np.ndarray,
    data: dict,
    *,
    device: str = "cpu",
    positions: int = 256,
    repeats: int = 10,
) -> dict:
    """Latency of both honest readings of this candidate's cost.

    A C1-feature candidate has two defensible prices and reporting only one
    would be a rhetorical choice:

    ``specialist`` is the belief CNN alone over an already-computed field —
    the added cost inside a search that is *already* running C1 for its
    policy, which is the situation this project is actually in.

    ``end_to_end`` is the frozen encode plus the specialist from the raw
    public observation — the cost of a belief query in isolation, and the
    number that is comparable to Agent 2's, which has no C1 stage.
    """
    target = torch.device(device)
    model = model.to(target).eval()
    frozen_c1 = frozen_c1.to(target).eval()
    rows = min(int(positions), int(data["samples"]))
    per_position = float(int(data["pieces"]) / max(int(data["samples"]), 1))
    batch_field = torch.from_numpy(np.array(field[:rows], dtype=np.float32, copy=True)).to(
        target
    )
    batch_observations = np.array(observations[:rows], dtype=np.float32, copy=True)

    def timed(call, warmup: int = 3) -> float:
        for _ in range(warmup):
            call()
        _synchronize(target)
        started = time.perf_counter()
        for _ in range(int(repeats)):
            call()
        _synchronize(target)
        return (time.perf_counter() - started) / int(repeats)

    specialist_batch = timed(lambda: model(batch_field))
    specialist_single = timed(lambda: model(batch_field[:1]))
    encode_batched = timed(lambda: encode_batch(frozen_c1, batch_observations, LAYER_FINAL))
    encode_single = timed(
        lambda: encode_batch(frozen_c1, batch_observations[:1], LAYER_FINAL)
    )

    def block(name: str, batched: float, single: float) -> dict:
        return {
            "milliseconds_per_decision_batched": round(batched / rows * 1e3, 4),
            "milliseconds_per_decision_single": round(single * 1e3, 4),
            "microseconds_per_piece_batched": round(batched / rows / per_position * 1e6, 4),
            "what": name,
        }

    return {
        "device": str(device),
        "batch_positions": rows,
        "repeats": int(repeats),
        "hidden_pieces_per_decision": round(per_position, 3),
        "specialist": block(
            "the belief CNN alone, over an already-computed C1 field",
            specialist_batch,
            specialist_single,
        ),
        "frozen_c1_encode": block(
            "the frozen C1 encode alone, which a search already pays for its policy",
            encode_batched,
            encode_single,
        ),
        "end_to_end": block(
            "public observation -> frozen C1 -> belief CNN",
            specialist_batch + encode_batched,
            specialist_single + encode_single,
        ),
    }


__all__ = [
    "CANDIDATE_3",
    "C1FeatureBeliefCNN",
    "C1FeatureBeliefModel",
    "FEATURE_CNN_ACTIVATION",
    "FEATURE_CNN_BLOCKS",
    "FEATURE_CNN_NORMALIZATION",
    "FEATURE_CNN_READOUT_WIDTH",
    "FEATURE_CNN_VERSION",
    "FEATURE_CNN_WIDTH",
    "Phase11BFeatureCNNError",
    "build_feature_cnn",
    "feature_split_view",
    "inference_cost",
    "load_feature_cnn",
    "parameter_breakdown",
]
