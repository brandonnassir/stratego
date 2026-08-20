"""Phase 11B Agent 4: a fusion CNN over the raw observation *and* the C1 field.

Specification source: `04_AGENT_4_HYBRID_RAW_C1_CNN.md` ("Inputs", "Model",
"Engineering Question", "Required Interface").

```text
raw 127 x 10 x 10 public observation -> conv3x3 80 --\
                                                      concat -> 160
frozen per-square C1 field [100, 128] -> conv3x3 80 --/
                                                      ->
                                              8 x residual3x3 160
                                                      ->
                                              1x1 128 -> 12 logits/square
```

The tower is held fixed so the *input* is the variable
--------------------------------------------------------
Agent 2 asked what a specialist learns from raw pixels; Agent 3 asked what
one learns from C1's final representation; Agent 4 asks whether the two are
complementary. That question is only answerable if the three candidates
differ in their inputs and in nothing else, so the residual tower and the
read-out are Agent 2's, imported rather than re-declared, at Agent 2's width
and Agent 2's depth. What changes is the stem:

```text
Agent 2   conv3x3(127 -> 160)                       3,897,004 parameters
Agent 3   conv3x3(128 -> 160)                       3,898,444 parameters
Agent 4   conv3x3(127 -> 80) || conv3x3(128 -> 80)  3,897,724 parameters
```

The three counts are within 1,440 of each other — a spread of 0.04% — so a
difference between the three reports is a difference of representation and
not of capacity. That is the whole design.

Why the branches are 80 and 80
-------------------------------
They are not a tuned split. `04_AGENT_4` forbids a branch-width sweep, and
the fused width had to be 160 for the tower to be Agent 2's, so the only
free choice was how to divide 160 between the two branches. It is divided
evenly, because the experiment asks whether the two sources are
*complementary* and an uneven split would prejudge which one carries more.
Each branch's projection is deliberately "small" in the instruction's sense
— one 3x3 convolution, no depth of its own — so the fusion happens early and
the shared tower, not a private per-branch stack, does the work.

Concatenation rather than addition or gating
---------------------------------------------
`04_AGENT_4` names "concatenate/fuse" and forbids a fusion-method sweep.
Concatenation is the choice, for a structural reason: summing two
projections would force the two representations into one shared 160-channel
basis before a single nonlinearity has seen them together, and gating would
add a learned mixing rule this experiment has no budget to validate.
Concatenation lets the first residual block's 3x3 convolution learn the
mixture itself, per channel and per neighbourhood, which is the weakest
assumption of the three.

No hidden truth enters either branch
-------------------------------------
`04_AGENT_4`: "No hidden truth may enter either branch."
:meth:`HybridBeliefCNN.forward` takes exactly one tensor, whose 255 channels
are the 127 public observation planes followed by the 128 frozen C1 planes,
and :meth:`forward_parts` takes exactly those two objects. Both are
functions of the public observation alone: the corpus keeps true ranks in a
different directory, the loader hands them over only when asked by name, and
neither entry point has an argument a label could arrive in.

C1 stays frozen
---------------
Agent 3's frozen seam is reused exactly — the same `encode` output, the same
`LAYER_FINAL` token, the same `field_to_planes` layout, the same cache files
on disk. C1 is not called during training at all, and no optimizer built
from this module's parameters can reach it: :func:`build_hybrid_cnn` returns
the specialist alone.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .contract import (
    C1_FEATURE_WIDTH,
    NUM_SQUARES,
    OBSERVATION_SHAPE,
    RANK_COUNT,
    Phase11BError,
)
from .features import LAYER_FINAL, encode_batch
from .feature_seam import BOARD_COLUMNS, BOARD_ROWS, SEAM_ID, field_to_planes
from .interface import Phase11BBeliefModel, Phase11BPublicState
from .raw_cnn import ResidualBlock, _PerSquareReadout, parameter_count
from .raw_train import _synchronize

#: The candidate identity. Leaderboard key, checkpoint stem, seed part.
CANDIDATE_4 = "agent04_hybrid_raw_c1_cnn"

#: The one declared configuration. `HYBRID_*_BRANCH_WIDTH` sum to the
#: inherited tower width, which is what fixes them.
HYBRID_RAW_BRANCH_WIDTH = 80
HYBRID_C1_BRANCH_WIDTH = 80
HYBRID_WIDTH = HYBRID_RAW_BRANCH_WIDTH + HYBRID_C1_BRANCH_WIDTH
HYBRID_BLOCKS = 8
HYBRID_READOUT_WIDTH = 128
HYBRID_FUSION = "concatenate"
HYBRID_ACTIVATION = "relu"
HYBRID_NORMALIZATION = "batchnorm2d"

#: The architecture-family identity a checkpoint carries.
HYBRID_CNN_VERSION = "phase11b_hybrid_raw_c1_cnn_v1"

#: The channel layout of the one tensor the shared trainer stages: the 127
#: public observation planes, then the 128 frozen C1 planes.
RAW_CHANNELS = OBSERVATION_SHAPE[0]
FUSED_CHANNELS = RAW_CHANNELS + C1_FEATURE_WIDTH
FUSED_SHAPE = (FUSED_CHANNELS, BOARD_ROWS, BOARD_COLUMNS)


class Phase11BHybridCNNError(Phase11BError):
    """A hybrid CNN was built or driven outside its contract."""


class HybridBeliefCNN(nn.Module):
    """The Agent 4 candidate: two public branches in, 12 rank logits per square.

    `feature_layer` is the hook the deployed interface reads to decide which
    frozen tensor to compute for a live position. It is `LAYER_FINAL` — the
    accepted `encode` output, which is exactly the tensor the cached field
    holds — so the training path and the deployed path cannot drift apart.
    """

    candidate_id = CANDIDATE_4
    architecture_version = HYBRID_CNN_VERSION
    seam_id = SEAM_ID
    feature_layer = LAYER_FINAL
    fusion = HYBRID_FUSION

    def __init__(
        self,
        *,
        raw_branch_width: int = HYBRID_RAW_BRANCH_WIDTH,
        c1_branch_width: int = HYBRID_C1_BRANCH_WIDTH,
        blocks: int = HYBRID_BLOCKS,
        readout_width: int = HYBRID_READOUT_WIDTH,
        block_dropout: float = 0.0,
        readout_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.raw_branch_width = int(raw_branch_width)
        self.c1_branch_width = int(c1_branch_width)
        self.width = self.raw_branch_width + self.c1_branch_width
        self.blocks_count = int(blocks)
        self.readout_width = int(readout_width)
        self.block_dropout = float(block_dropout)
        self.readout_dropout = float(readout_dropout)

        # The two "small spatial projections" of the instruction's diagram.
        # One 3x3 each: enough to be spatial, shallow enough that the fusion
        # is early and the shared tower does the work.
        self.raw_branch = nn.Sequential(
            nn.Conv2d(
                RAW_CHANNELS, self.raw_branch_width, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.raw_branch_width),
            nn.ReLU(inplace=True),
        )
        self.c1_branch = nn.Sequential(
            nn.Conv2d(
                C1_FEATURE_WIDTH, self.c1_branch_width, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.c1_branch_width),
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
            f"hybrid(raw {RAW_CHANNELS} -> conv3x3 {self.raw_branch_width} || "
            f"c1_field({C1_FEATURE_WIDTH}) -> conv3x3 {self.c1_branch_width}) -> "
            f"{HYBRID_FUSION} {self.width} -> {self.blocks_count} x residual3x3 -> "
            f"1x1 {self.readout_width} -> {RANK_COUNT}/square, "
            f"{HYBRID_ACTIVATION}, {HYBRID_NORMALIZATION}{drop}"
        )

    # -- the two entry points ------------------------------------------------

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """`[B, 255, 10, 10]` staged input -> `[B, 12, 10, 10]` logits.

        The 255 channels are the 127 public observation planes followed by
        the 128 frozen C1 planes, which is the layout
        :func:`fused_observations` writes. Taking one tensor is what lets
        Agent 4 reuse Agent 2's trainer unchanged; :meth:`forward_parts` is
        the same computation with the two halves handed over separately,
        and the split here is the only place the layout is interpreted.
        """
        if fused.dim() != 4 or tuple(fused.shape[1:]) != FUSED_SHAPE:
            raise Phase11BHybridCNNError(
                f"fused input must be [B, {', '.join(str(dim) for dim in FUSED_SHAPE)}], "
                f"got {tuple(fused.shape)}"
            )
        return self.forward_planes(fused[:, :RAW_CHANNELS], fused[:, RAW_CHANNELS:])

    def forward_parts(self, observations: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        """`[B, 127, 10, 10]` and `[B, 100, 128]` -> `[B, 12, 10, 10]` logits.

        The deployed shape of the model: a live position arrives as a public
        observation, the frozen encoder turns it into a field, and the two
        go in side by side. `field_to_planes` is Agent 3's, so the C1 branch
        sees the same layout in deployment that it saw in training.
        """
        if observations.dim() != 4 or tuple(observations.shape[1:]) != OBSERVATION_SHAPE:
            raise Phase11BHybridCNNError(
                f"observations must be [B, {', '.join(str(d) for d in OBSERVATION_SHAPE)}], "
                f"got {tuple(observations.shape)}"
            )
        return self.forward_planes(observations, field_to_planes(field))

    def forward_planes(
        self, observations: torch.Tensor, field_planes: torch.Tensor
    ) -> torch.Tensor:
        """The fusion itself. Both inputs are `[B, C, 10, 10]` board planes."""
        if field_planes.dim() != 4 or field_planes.shape[1] != C1_FEATURE_WIDTH:
            raise Phase11BHybridCNNError(
                f"C1 planes must be [B, {C1_FEATURE_WIDTH}, {BOARD_ROWS}, "
                f"{BOARD_COLUMNS}], got {tuple(field_planes.shape)}"
            )
        fused = torch.cat(
            (self.raw_branch(observations), self.c1_branch(field_planes)), dim=1
        )
        return self.readout(self.blocks(fused))

    # -- read-out geometry, shared with Agents 2 and 3 -----------------------

    def per_square_logits(self, fused: torch.Tensor) -> torch.Tensor:
        """`[B, 255, 10, 10]` -> `[B, 100, 12]` in accepted token order."""
        planes = self.forward(fused)
        return planes.reshape(planes.shape[0], RANK_COUNT, NUM_SQUARES).transpose(1, 2)

    def logits_at(
        self, fused: torch.Tensor, rows: torch.Tensor, squares: torch.Tensor
    ) -> torch.Tensor:
        """`[K, 12]` — the logits of `K` supervised pieces of this batch.

        The same signature Agent 2's trainer calls, so Agent 4 trains
        through the identical optimizer, shuffle, probe schedule and
        checkpoint-selection rule.
        """
        return self.per_square_logits(fused)[rows, squares]

    def per_square_logits_from_parts(
        self, observations: torch.Tensor, field: torch.Tensor
    ) -> torch.Tensor:
        """`[B, 100, 12]` from a live observation and its frozen field."""
        planes = self.forward_parts(observations, field)
        return planes.reshape(planes.shape[0], RANK_COUNT, NUM_SQUARES).transpose(1, 2)


def parameter_breakdown(model: HybridBeliefCNN) -> dict:
    """Where the parameters are, so the count can be checked by hand."""
    return {
        "raw_branch": parameter_count(model.raw_branch),
        "c1_branch": parameter_count(model.c1_branch),
        "residual_tower": parameter_count(model.blocks),
        "readout": parameter_count(model.readout),
        "total": parameter_count(model),
        "raw_branch_width": model.raw_branch_width,
        "c1_branch_width": model.c1_branch_width,
        "width": model.width,
        "blocks": model.blocks_count,
        "readout_width": model.readout_width,
        "block_dropout": model.block_dropout,
        "readout_dropout": model.readout_dropout,
        "fusion": HYBRID_FUSION,
        "raw_input_channels": RAW_CHANNELS,
        "c1_input_channels": C1_FEATURE_WIDTH,
        "conv3x3_layers": 2 + 2 * model.blocks_count,
        "frozen_c1_parameters": 863_959,
        "trainable_c1_parameters": 0,
    }


def build_hybrid_cnn(*, seed: "int | None" = None, **kwargs) -> HybridBeliefCNN:
    """The declared Agent 4 architecture, initialized from a named stream.

    Returns the specialist **alone**. C1 is not part of this object and is
    therefore not part of any optimizer built from its parameters — the
    freeze is structural, not a convention the trainer has to remember.
    """
    if seed is not None:
        torch.manual_seed(int(seed) % (2**63))
    return HybridBeliefCNN(**kwargs)


def load_hybrid_cnn(path, *, map_location: str = "cpu") -> tuple:
    """`(model, payload)` — a saved candidate, rebuilt from its own record."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    shape = payload.get("parameter_breakdown", {})
    model = HybridBeliefCNN(
        raw_branch_width=int(shape.get("raw_branch_width", HYBRID_RAW_BRANCH_WIDTH)),
        c1_branch_width=int(shape.get("c1_branch_width", HYBRID_C1_BRANCH_WIDTH)),
        blocks=int(shape.get("blocks", HYBRID_BLOCKS)),
        readout_width=int(shape.get("readout_width", HYBRID_READOUT_WIDTH)),
        block_dropout=float(shape.get("block_dropout", 0.0)),
        readout_dropout=float(shape.get("readout_dropout", 0.0)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


# ---------------------------------------------------------------------------
# The staged fused input
# ---------------------------------------------------------------------------

#: The fused-cache format identity. A change to the channel order, the
#: layout or the dtype is a new version, never a silent edit.
FUSED_CACHE_VERSION = "phase11b_hybrid_fused_input_cache_v1"


def fused_cache_path(root: "Path | str", split: str):
    """Where one split's fused input lives."""
    from pathlib import Path as _Path

    return _Path(root) / f"hybrid_input_{split}.npy"


def fuse_arrays(observations: np.ndarray, field: np.ndarray) -> np.ndarray:
    """`[B, 127, 10, 10]` and `[B, 100, 128]` -> `[B, 255, 10, 10]`.

    The numpy statement of what :meth:`HybridBeliefCNN.forward` splits back
    apart, and of what the cache on disk holds: the public observation
    planes first, the frozen C1 planes second, in Agent 3's
    `field_to_planes` layout. The two halves are written by the same two
    lines that a test re-derives them with, so the layout has one
    definition rather than two that could drift.
    """
    observations = np.asarray(observations, dtype=np.float32)
    field = np.asarray(field, dtype=np.float32)
    if observations.shape[1:] != OBSERVATION_SHAPE:
        raise Phase11BHybridCNNError(
            f"observations are {observations.shape}, expected [B, {OBSERVATION_SHAPE}]"
        )
    if field.shape[1:] != (NUM_SQUARES, C1_FEATURE_WIDTH):
        raise Phase11BHybridCNNError(
            f"field is {field.shape}, expected [B, {NUM_SQUARES}, {C1_FEATURE_WIDTH}]"
        )
    if observations.shape[0] != field.shape[0]:
        raise Phase11BHybridCNNError(
            f"{observations.shape[0]} observations against {field.shape[0]} fields"
        )
    planes = (
        field.transpose(0, 2, 1)
        .reshape(field.shape[0], C1_FEATURE_WIDTH, BOARD_ROWS, BOARD_COLUMNS)
    )
    return np.concatenate((observations, planes), axis=1)


def build_fused_cache(
    data: dict, field, path, *, batch_size: int = 512, progress=None
) -> dict:
    """Write one split's fused input, and describe it.

    Streamed into a `.npy` memory map rather than assembled in RAM: the
    training split is 26,898 positions x 255 channels x 100 squares, which
    is 2.7 GB, and there is no reason for that array and the two arrays it
    came from to be resident at once.

    Nothing new is computed here. The observation half is the corpus's own
    stored bytes and the C1 half is Agent 3's cache, unchanged — this is a
    re-layout so that Agent 2's single-tensor trainer can be reused without
    forking it, not a third representation.
    """
    from pathlib import Path as _Path

    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(data["samples"])
    observations = data["observations"]
    cache = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(samples, *FUSED_SHAPE)
    )
    started = time.perf_counter()
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        cache[start:stop] = fuse_arrays(
            np.asarray(observations[start:stop]), np.asarray(field[start:stop])
        )
        if progress is not None:
            progress(stop, samples, time.perf_counter() - started)
    cache.flush()
    seconds = round(time.perf_counter() - started, 3)
    digest = fused_digest(cache)
    del cache
    return {
        "cache_version": FUSED_CACHE_VERSION,
        "seam_id": SEAM_ID,
        "layer_token": LAYER_FINAL,
        "split": data.get("split"),
        "path": str(path),
        "shape": [samples, *FUSED_SHAPE],
        "dtype": "float32",
        "channel_layout": {
            "public_observation": [0, RAW_CHANNELS],
            "frozen_c1_field": [RAW_CHANNELS, FUSED_CHANNELS],
        },
        "bytes": int(path.stat().st_size),
        "digest": digest,
        "seconds": seconds,
        "positions_per_second": round(samples / max(seconds, 1e-9), 1),
        "batch_size": int(batch_size),
        "derived_from": [
            "the common corpus public observations",
            "Agent 3's frozen C1 field cache",
        ],
        "contains_labels": False,
    }


def load_fused_cache(path, *, expected_samples: "int | None" = None):
    """A fused cache, memory-mapped read-only."""
    from pathlib import Path as _Path

    path = _Path(path)
    if not path.exists():
        raise Phase11BHybridCNNError(f"fused cache {path} does not exist")
    cache = np.load(path, mmap_mode="r")
    if cache.ndim != 4 or tuple(cache.shape[1:]) != FUSED_SHAPE:
        raise Phase11BHybridCNNError(
            f"{path} is {cache.shape}, expected [N, {FUSED_SHAPE}]"
        )
    if expected_samples is not None and cache.shape[0] != int(expected_samples):
        raise Phase11BHybridCNNError(
            f"{path} holds {cache.shape[0]} positions, expected {expected_samples}"
        )
    return cache


def fused_digest(cache: np.ndarray, *, chunk: int = 1024) -> str:
    """Content identity of a fused cache, streamed rather than materialized."""
    import hashlib

    hasher = hashlib.sha256()
    hasher.update(FUSED_CACHE_VERSION.encode())
    hasher.update(SEAM_ID.encode())
    hasher.update(str(tuple(cache.shape)).encode())
    for start in range(0, cache.shape[0], chunk):
        block = np.ascontiguousarray(cache[start : start + chunk], dtype=np.float32)
        hasher.update(block.tobytes())
    return hasher.hexdigest()


def verify_fused_cache(
    data: dict, field, cache: np.ndarray, *, rows: int = 64, seed: int = 20260819
) -> dict:
    """Re-derive a random sample of the fused cache from its two sources.

    The cache is only legitimate if it is exactly the public observation and
    the frozen C1 field side by side and nothing else. That is a claim about
    what the model is fed, so it is measured: a random sample of positions is
    re-fused from the corpus bytes and Agent 3's cache and compared to what
    is stored, and each half is compared separately so a report can say that
    *neither* branch was altered on its way into the tower.
    """
    generator = np.random.default_rng(int(seed))
    samples = int(data["samples"])
    picked = np.sort(
        generator.choice(samples, size=min(int(rows), samples), replace=False)
    )
    stored = np.asarray(cache[picked], dtype=np.float32)
    observations = np.asarray(data["observations"])[picked]
    fields = np.asarray(field[picked], dtype=np.float32)
    rebuilt = fuse_arrays(observations, fields)
    raw_difference = float(np.abs(stored[:, :RAW_CHANNELS] - observations).max())
    c1_difference = float(np.abs(rebuilt[:, RAW_CHANNELS:] - stored[:, RAW_CHANNELS:]).max())
    difference = float(np.abs(rebuilt - stored).max())
    return {
        "rows_checked": int(picked.size),
        "max_absolute_difference": difference,
        "raw_half_max_absolute_difference": raw_difference,
        "c1_half_max_absolute_difference": c1_difference,
        "raw_half_is_the_corpus_observation": bool(raw_difference == 0.0),
        "c1_half_is_agent3s_field": bool(c1_difference == 0.0),
        "bit_identical": bool(difference == 0.0),
        "inputs": [
            "the common corpus public observations",
            "Agent 3's frozen C1 field cache",
        ],
    }


def hybrid_split_view(data: dict, fused) -> dict:
    """A stored split whose model input is the fused two-branch tensor.

    Agent 2's trainer stages `data["observations"]` as one tensor, indexes it
    by sample row and hands batches to `model.logits_at`; nothing in it knows
    what those channels mean. Agent 3 exploited that to swap in the C1 field.
    Agent 4 exploits it once more, with the two representations side by side,
    which is why all three candidates share not only a tower but an
    optimizer, a shuffle, a probe schedule and a checkpoint-selection rule.

    Every other key is passed through unchanged, so the labels, the piece
    offsets, the strata and the baseline arrays are literally the same
    objects the corpus loader produced.
    """
    fused = np.asarray(fused)
    samples = int(data["samples"])
    if fused.shape != (samples, *FUSED_SHAPE):
        raise Phase11BHybridCNNError(
            f"fused input is {fused.shape}, expected {(samples, *FUSED_SHAPE)}"
        )
    view = dict(data)
    view["observations"] = fused
    view["model_input"] = "public_observation_and_frozen_c1_field"
    view["public_observations"] = data["observations"]
    return view


# ---------------------------------------------------------------------------
# The required belief interface
# ---------------------------------------------------------------------------


class HybridBeliefModel(Phase11BBeliefModel):
    """The Agent 4 CNN behind the shared Phase 11B belief interface.

    Subclasses Agent 1's interface rather than reimplementing it. The
    encoder slot holds the frozen C1, so a live position is encoded through
    the accepted seam exactly as Agent 3 encodes it, and `sample_worlds` —
    the part that must go through the **accepted, unmodified** Phase 11
    sampler — is inherited code, not a fork.

    The head slot holds Agent 2's identity read-out, because this candidate
    already emits 12 logits per square: `_features` runs the whole hybrid
    forward pass and returns the `[100, 12]` field, and
    `predict_marginals` gathers the unresolved pieces' squares out of it.
    """

    def __init__(
        self,
        frozen_c1,
        model: HybridBeliefCNN,
        *,
        candidate_id: str = CANDIDATE_4,
        device: str = "cpu",
    ) -> None:
        target = torch.device(device)
        super().__init__(
            frozen_c1.to(target), _PerSquareReadout(), candidate_id=candidate_id, device=device
        )
        self.cnn = model.to(target).eval()

    def _features(self, state: Phase11BPublicState) -> torch.Tensor:
        """`[100, 12]` — this candidate's logit field for one public state.

        The only input is `state.observation`. The C1 branch's input is
        derived from it here, through the same frozen encoder and the same
        `feature_layer` the cached field was built from, so the deployed
        path and the trained path read the identical tensor.
        """
        observation = np.array(state.observation, dtype=np.float32, copy=True)[None]
        field = encode_batch(self.encoder, observation, self.cnn.feature_layer)
        with torch.no_grad():
            logits = self.cnn.per_square_logits_from_parts(
                torch.from_numpy(observation).to(self.device), field.to(self.device)
            )
        return logits[0]

    def describe(self) -> dict:
        return {
            **super().describe(),
            "architecture": self.cnn.architecture,
            "architecture_version": HYBRID_CNN_VERSION,
            "seam_id": SEAM_ID,
            "fusion": HYBRID_FUSION,
            "consumes_c1_features": True,
            "consumes_public_observation": True,
            "c1_frozen": True,
            "parameters": parameter_count(self.cnn),
        }


@torch.no_grad()
def inference_cost(
    model: HybridBeliefCNN,
    frozen_c1,
    fused: np.ndarray,
    observations: np.ndarray,
    data: dict,
    *,
    device: str = "cpu",
    positions: int = 256,
    repeats: int = 10,
) -> dict:
    """Latency of both honest readings of this candidate's cost.

    Agent 3's two readings, for the same reason. ``specialist`` is the
    fusion CNN alone over an already-computed field — the added cost inside
    a search that is *already* running C1 for its policy. ``end_to_end`` is
    the frozen encode plus the specialist from the raw public observation —
    the cost of a belief query in isolation, and the number comparable to
    Agent 2's, which has no C1 stage.
    """
    target = torch.device(device)
    model = model.to(target).eval()
    frozen_c1 = frozen_c1.to(target).eval()
    rows = min(int(positions), int(data["samples"]))
    per_position = float(int(data["pieces"]) / max(int(data["samples"]), 1))
    batch_fused = torch.from_numpy(np.array(fused[:rows], dtype=np.float32, copy=True)).to(
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

    specialist_batch = timed(lambda: model(batch_fused))
    specialist_single = timed(lambda: model(batch_fused[:1]))
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
            "the fusion CNN alone, over an already-computed C1 field",
            specialist_batch,
            specialist_single,
        ),
        "frozen_c1_encode": block(
            "the frozen C1 encode alone, which a search already pays for its policy",
            encode_batched,
            encode_single,
        ),
        "end_to_end": block(
            "public observation -> frozen C1 -> fusion CNN",
            specialist_batch + encode_batched,
            specialist_single + encode_single,
        ),
    }


__all__ = [
    "CANDIDATE_4",
    "FUSED_CACHE_VERSION",
    "FUSED_CHANNELS",
    "FUSED_SHAPE",
    "HYBRID_ACTIVATION",
    "HYBRID_BLOCKS",
    "HYBRID_C1_BRANCH_WIDTH",
    "HYBRID_CNN_VERSION",
    "HYBRID_FUSION",
    "HYBRID_NORMALIZATION",
    "HYBRID_RAW_BRANCH_WIDTH",
    "HYBRID_READOUT_WIDTH",
    "HYBRID_WIDTH",
    "HybridBeliefCNN",
    "HybridBeliefModel",
    "Phase11BHybridCNNError",
    "RAW_CHANNELS",
    "build_fused_cache",
    "build_hybrid_cnn",
    "fuse_arrays",
    "fused_cache_path",
    "fused_digest",
    "hybrid_split_view",
    "inference_cost",
    "load_fused_cache",
    "load_hybrid_cnn",
    "parameter_breakdown",
    "parameter_count",
    "verify_fused_cache",
]
