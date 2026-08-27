"""Phase 15 Agent 1: the frozen prefix, cached once per backbone.

Specification source: `01_AGENT_1_BELIEF_HEAD_TRAINING.md` section 8.

Why the cache is part of the design
------------------------------------
A specialist's frozen prefix is the source policy's first three encoder
blocks. Frozen means its output is a *constant of the corpus*, so running
it once per epoch would be recomputing a constant twelve times. Caching it
turns each epoch into the last block plus the MLP, and it guarantees that
B18 and B24 each see exactly the representation their own backbone
produces — which is the whole reason the two specialists are separate
objects rather than one head trained on a shared feature.

One definition of "the frozen prefix"
--------------------------------------
:func:`encode_prefix` is the only place the split between frozen and
trainable is expressed. The cache builder and the live belief provider both
call it, so the two cannot drift apart: a provider that computed the prefix
differently from the trainer would silently evaluate a different model.

The cache is derived, not evidence
-----------------------------------
It is a pure function of `(source model state digest, corpus split bytes)`,
both of which the manifest already binds, so the cache file itself carries
no identity a report needs. It lives under `checkpoints/phase15/` and may
be deleted and rebuilt at any time.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from .contract import C1_FEATURE_WIDTH, NUM_SQUARES, Phase15Error

#: The cache-format identity.
FEATURE_CACHE_VERSION = "phase15_frozen_prefix_cache_v1"

#: How many trailing blocks the specialist owns; the prefix is the rest.
from .heads import TRAINABLE_BLOCKS  # noqa: E402


class Phase15FeatureError(Phase15Error):
    """A frozen-prefix cache could not be built or verified."""


def encode_prefix(model, observations: np.ndarray):
    """`[B, 127, 10, 10]` numpy -> `[B, 100, 128]` penultimate tokens.

    The accepted `encode` minus its trailing block and encoder norm — which
    is exactly the part a Phase 15 specialist owns a trainable copy of.
    """
    import torch

    from ...model.tokenization import observation_to_tokens

    # `copy=True`: the corpus observations are a read-only memmap, and torch
    # refuses to wrap a non-writable buffer without a warning.
    tensor = torch.from_numpy(np.array(observations, dtype=np.float32, copy=True))
    tokens = observation_to_tokens(tensor.to(next(model.parameters()).device))
    hidden = model.input_projection(tokens) + model.position_embedding().unsqueeze(0)
    blocks = list(model.blocks)
    for block in blocks[: len(blocks) - TRAINABLE_BLOCKS]:
        hidden = block(hidden)
    return hidden


def cache_path(root: "Path | str", specialist_id: str, split: str) -> Path:
    return Path(root) / f"{specialist_id}_prefix_{split}.npy"


def build_cache(
    model,
    data: dict,
    path: "Path | str",
    *,
    batch_size: int = 512,
    device: str = "cpu",
    progress=None,
) -> dict:
    """Cache one split's frozen prefix to a `[N, 100, 128]` float32 file.

    Written through a memmap so a 120,000-position split never has to fit
    in memory, and returned as a read-only memmap so the trainer cannot
    write to it by accident.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(data["samples"])
    observations = data["observations"]
    cache = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(samples, NUM_SQUARES, C1_FEATURE_WIDTH)
    )
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, samples, batch_size):
            stop = min(start + batch_size, samples)
            encoded = encode_prefix(model, np.asarray(observations[start:stop]))
            cache[start:stop] = encoded.detach().to("cpu", torch.float32).numpy()
            if progress is not None:
                progress(stop, samples, time.perf_counter() - started)
    cache.flush()
    del cache
    return {
        "cache_version": FEATURE_CACHE_VERSION,
        "path": str(path),
        "split": data.get("split"),
        "samples": samples,
        "shape": [samples, NUM_SQUARES, C1_FEATURE_WIDTH],
        "dtype": "float32",
        "bytes": int(path.stat().st_size),
        "device": device,
        "seconds": round(time.perf_counter() - started, 3),
    }


def load_cache(path: "Path | str") -> np.ndarray:
    """A cached split's prefix, read-only."""
    path = Path(path)
    if not path.is_file():
        raise Phase15FeatureError(f"no frozen-prefix cache at {path}")
    return np.load(path, mmap_mode="r")


def cache_digest(cache: np.ndarray, *, rows: int = 256) -> str:
    """A cheap content fingerprint of a cache: shape plus a strided sample.

    Digesting six gigabytes on every load would cost more than rebuilding
    the cache. This is enough to catch a truncated or mismatched file, and
    the cache is derived data whose true identity is the source digest and
    the corpus digest the manifest already binds.
    """
    hasher = hashlib.sha256()
    hasher.update(FEATURE_CACHE_VERSION.encode())
    hasher.update(str(cache.shape).encode())
    count = int(cache.shape[0])
    if count:
        step = max(1, count // int(rows))
        for index in range(0, count, step):
            hasher.update(
                np.ascontiguousarray(cache[index], dtype=np.float32).tobytes()
            )
    return hasher.hexdigest()


__all__ = [
    "FEATURE_CACHE_VERSION",
    "Phase15FeatureError",
    "build_cache",
    "cache_digest",
    "cache_path",
    "encode_prefix",
    "load_cache",
]
