"""Phase 11B Agent 1: the frozen C1 representation, cached once.

Specification source: `01_AGENT_1_ATTACHED_BELIEF_HEAD.md` (Experiments 1A
and 1B — "Freeze every shared C1 parameter", "Keep C1 completely frozen").

Why caching is the whole experiment design
-------------------------------------------
1A and 1B differ only in what is attached to a **frozen** encoder. A frozen
encoder's output is a constant of the corpus, so running the 864k-parameter
transformer once per epoch would be recomputing a constant. Caching it turns
both experiments into training a small head over a fixed matrix, which is
what lets Agent 1 answer its question in minutes rather than hours — and it
guarantees 1A and 1B see *identical* features, so any difference between
them is the head and nothing else.

Two cache layers, one for each kind of experiment
--------------------------------------------------
```text
final         encoder output at the supervised tokens   [M, 128]
penultimate   input to the last encoder block, all tokens   [N, 100, 128]
```

`final` is what 1A and 1B consume: one row per supervised piece, the exact
tensor the accepted `belief_output` linear layer reads. `penultimate` is
what 1C consumes, because unfreezing the last block makes that block's
*input* the constant.

The Phase 9 weights are opened read-only
-----------------------------------------
The accepted checkpoint is never written. :func:`load_frozen_c1` exports it
through the accepted `export_evaluation_weights` into a Phase 11B path and
loads *that*, and every returned parameter has `requires_grad=False`, so
the freeze is a property of the object rather than of a convention the
caller has to remember.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from .contract import C1_FEATURE_WIDTH, NUM_SQUARES, Phase11BError

#: The cache-format identity.
FEATURE_CACHE_VERSION = "phase11b_c1_feature_cache_v1"

LAYER_FINAL = "final"
LAYER_PENULTIMATE = "penultimate"
CACHE_LAYERS = (LAYER_FINAL, LAYER_PENULTIMATE)

#: The accepted Phase 9 export, relative to the repository root. Read-only.
ACCEPTED_PHASE9_CHECKPOINT = "checkpoints/phase9/selfplay_c1_v1.pt"

#: Where Phase 11B keeps its own copy of the accepted weights.
PHASE11B_EXPORT = "checkpoints/phase11b/phase9_c1_readonly_copy.pt"


class Phase11BFeatureError(Phase11BError):
    """A frozen-feature cache could not be built or verified."""


def belief_head_digest(model) -> str:
    """The accepted belief-head digest recipe, applied to a live model.

    sha256 over `(name, str(shape), float32 C-order bytes)` of the
    `belief_output.*` tensors in sorted name order — the frozen Phase 11
    Agent 1 recipe, so a Phase 11B model can be compared to the accepted
    head without importing Phase 11's harness.
    """
    import torch

    state = model.state_dict()
    hasher = hashlib.sha256()
    for name in sorted(name for name in state if name.startswith("belief_output.")):
        tensor = state[name]
        array = tensor.detach().to("cpu", torch.float32).contiguous().numpy()
        hasher.update(name.encode())
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def load_frozen_c1(
    repository_root: "Path | str",
    export_path: "Path | str | None" = None,
    *,
    device: str = "cpu",
):
    """A frozen read-only copy of the accepted Phase 9 C1 model.

    Returns `(model, identity)`. The identity carries the accepted digests
    so a report can state exactly which weights produced a feature cache.
    """
    import torch

    from ...model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from ...model.checkpoint import load_checkpoint
    from ...training.phase10_collector import export_evaluation_weights
    from ...training.phase11_contract import (
        ACCEPTED_BELIEF_HEAD_DIGEST,
        ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
    )
    from ...training.phase9_behavior import state_dict_digest

    root = Path(repository_root)
    source = root / ACCEPTED_PHASE9_CHECKPOINT
    export_path = root / PHASE11B_EXPORT if export_path is None else Path(export_path)
    export = export_evaluation_weights(source, export_path)
    model, metadata = load_checkpoint(
        export_path,
        device=torch.device(device),
        dtype=torch.float32,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    observed_state = state_dict_digest(model)
    observed_head = belief_head_digest(model)
    if observed_state != ACCEPTED_PHASE9_MODEL_STATE_DIGEST:
        raise Phase11BFeatureError(
            f"loaded Phase 9 state digest {observed_state} != accepted"
        )
    if observed_head != ACCEPTED_BELIEF_HEAD_DIGEST:
        raise Phase11BFeatureError(
            f"loaded belief-head digest {observed_head} != accepted"
        )
    identity = {
        "source_checkpoint": ACCEPTED_PHASE9_CHECKPOINT,
        "source_sha256": export["source_sha256"] if "source_sha256" in export else None,
        "export_path": str(export_path),
        "model_state_digest": observed_state,
        "belief_head_digest": observed_head,
        "parameters": int(sum(tensor.numel() for tensor in model.parameters())),
        "trainable_parameters": int(
            sum(tensor.numel() for tensor in model.parameters() if tensor.requires_grad)
        ),
        "device": str(device),
        "dtype": "float32",
        "architecture": metadata.get("model_architecture_id"),
        "candidate_id": "C1",
    }
    return model, identity


def encode_batch(model, observations: np.ndarray, layer: str):
    """`[B, 127, 10, 10]` numpy -> the requested layer's tensor.

    The single definition of "the frozen prefix". Both the cache builder and
    the live belief interface call it, so the two cannot drift apart.

    `final` reruns the accepted `encode`; `penultimate` stops one block
    short, which is exactly `encode` minus its last two lines.
    """
    import torch

    from ...model.tokenization import observation_to_tokens

    # `copy=True`: the corpus observations are a read-only memmap, and torch
    # refuses to wrap a non-writable buffer without a warning.
    tensor = torch.from_numpy(np.array(observations, dtype=np.float32, copy=True))
    tokens = observation_to_tokens(tensor.to(next(model.parameters()).device))
    if layer == LAYER_FINAL:
        return model.encode(tokens)
    if layer != LAYER_PENULTIMATE:  # pragma: no cover - guarded by callers
        raise Phase11BFeatureError(f"unknown cache layer {layer!r}")
    hidden = model.input_projection(tokens) + model.position_embedding().unsqueeze(0)
    for block in list(model.blocks)[:-1]:
        hidden = block(hidden)
    return hidden


def final_block(model):
    """`(last encoder block, encoder norm)` — the 1C trainable prefix."""
    return list(model.blocks)[-1], model.encoder_norm


def encode_split(
    model,
    data: dict,
    *,
    layer: str = LAYER_FINAL,
    batch_size: int = 256,
    progress=None,
) -> tuple:
    """Cache the frozen features of one stored split.

    For `final` the result is `[M, 128]` — one row per supervised hidden
    piece, gathered at that piece's perspective-normalized token, in the
    corpus's own piece order, so `features[i]` and `true_rank[i]` are the
    same piece by construction.

    For `penultimate` the result is `[N, 100, 128]`.
    """
    import torch

    if layer not in CACHE_LAYERS:
        raise Phase11BFeatureError(f"unknown cache layer {layer!r}")
    samples = int(data["samples"])
    offsets = np.asarray(data["piece_offset"], dtype=np.int64)
    squares = np.asarray(data["perspective_square"], dtype=np.int64)
    observations = data["observations"]

    rows = int(data["pieces"]) if layer == LAYER_FINAL else samples
    width = C1_FEATURE_WIDTH
    shape = (rows, width) if layer == LAYER_FINAL else (rows, NUM_SQUARES, width)
    cache = np.empty(shape, dtype=np.float32)

    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, samples, batch_size):
            stop = min(start + batch_size, samples)
            encoded = encode_batch(model, np.asarray(observations[start:stop]), layer)
            if layer == LAYER_PENULTIMATE:
                cache[start:stop] = encoded.detach().cpu().numpy()
            else:
                block = encoded.detach().cpu().numpy()
                for position in range(start, stop):
                    lo, hi = int(offsets[position]), int(offsets[position + 1])
                    if hi > lo:
                        cache[lo:hi] = block[position - start, squares[lo:hi]]
            if progress is not None:
                progress(stop, samples, time.perf_counter() - started)
    return cache, round(time.perf_counter() - started, 3)


def cache_digest(cache: np.ndarray) -> str:
    """Content identity of a feature cache."""
    hasher = hashlib.sha256()
    hasher.update(FEATURE_CACHE_VERSION.encode())
    hasher.update(str(cache.shape).encode())
    hasher.update(np.ascontiguousarray(cache, dtype=np.float32).tobytes())
    return hasher.hexdigest()


__all__ = [
    "ACCEPTED_PHASE9_CHECKPOINT",
    "CACHE_LAYERS",
    "FEATURE_CACHE_VERSION",
    "LAYER_FINAL",
    "LAYER_PENULTIMATE",
    "PHASE11B_EXPORT",
    "Phase11BFeatureError",
    "belief_head_digest",
    "cache_digest",
    "encode_batch",
    "encode_split",
    "final_block",
    "load_frozen_c1",
]
