"""Phase 11B Agent 3: the frozen C1 seam, and its per-square cache.

Specification source: `03_AGENT_3_C1_FEATURE_CNN.md` ("Frozen C1 Feature
Seam" — "deliberately select the richest spatial/token-level representation
immediately before the task heads that can be mapped back to board cells").

Which tensor, exactly
---------------------
`ProductionModel.forward` is three heads over one shared tensor::

    hidden = self.encode(tokens)            # [B, 100, 128]
    policy_logits = ... query/key over hidden
    value_logits  = ... over hidden.mean(dim=1)
    belief_logits = self.belief_output(hidden)

`hidden` — the return of :meth:`ProductionModel.encode`, which is
`encoder_norm(block_N(... block_1(input_projection(tokens) + position)))` —
is therefore *literally* the representation immediately before the task
heads. It is one 128-vector per board square, token `i` being row-major
square `i`, so it maps back to board cells exactly and with no pooling
anywhere in the path.

Why not one of the other candidates
------------------------------------
```text
pooled value feature   hidden.mean(dim=1)   [128]        global, not per-square
belief_output          [B, 100, 12]         already compressed 128 -> 12
penultimate            last block's input   [B, 100, 128]  one block short of the heads
policy query / key     [B, 100, 128]        a head's own projection, not shared
```

`03_AGENT_3` asks for the *richest* per-square tensor before the heads and
warns against "an unnecessarily pooled or compressed global vector", which
rules out the value pooling and the 12-way belief logits. The penultimate
tensor is per-square and 128-wide, but it is one encoder block short of
what the heads actually read, and the question Agent 3 asks — *does the
final C1 representation retain belief-relevant information* — is a question
about what the heads see. So the seam is `encode`'s output.

No new frozen-prefix code
-------------------------
Agent 1's :func:`features.encode_batch` already returns exactly this tensor
for `LAYER_FINAL`; Agent 1's *cache* then gathers it at the supervised
squares, which is what a per-piece head needs and what a spatial CNN cannot
use. So Agent 3 reuses the accepted seam call verbatim and keeps all 100
tokens. `features.py` is not modified: Agent 1's and Agent 2's artifacts
stay byte-for-byte what they were.

Derivable from public inputs alone
-----------------------------------
`03_AGENT_3`: "Any cache must be derivable from the common public
observations plus the accepted frozen C1." Every row here is
`encode(observation_to_tokens(public observation))` under weights whose
accepted digests are checked at load time. No label, no privileged array
and no Agent 3 parameter takes part, so the cache is a pure function of two
things that already exist, and :func:`verify_field_cache` re-derives a
sample of it to say so as a measurement.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from .contract import C1_FEATURE_WIDTH, NUM_SQUARES, Phase11BError
from .features import LAYER_FINAL, encode_batch

#: The cache-format identity. A change to the layer, the layout or the
#: dtype is a new version, never a silent edit.
FIELD_CACHE_VERSION = "phase11b_c1_field_cache_v1"

#: The seam identity, carried by the checkpoint and the report.
SEAM_ID = "c1_encoder_output_all_tokens"

#: The board geometry of the cached field, restated so a stored array can be
#: checked without importing the engine.
BOARD_ROWS = 10
BOARD_COLUMNS = 10

#: Exactly which tensor Agent 3 reads, in a form a report can print and a
#: test can assert against.
SEAM_DESCRIPTION = {
    "seam_id": SEAM_ID,
    "module": "stratego.model.production_model",
    "class": "ProductionModel",
    "tensor": "ProductionModel.encode(tokens)",
    "definition": (
        "encoder_norm(block_6(... block_1(input_projection(tokens) + "
        "position_embedding())))"
    ),
    "shape": ["batch", NUM_SQUARES, C1_FEATURE_WIDTH],
    "layer_token": LAYER_FINAL,
    "is_per_square": True,
    "is_pooled": False,
    "consumed_by_heads": ["policy_head", "value_head", "belief_head"],
    "position_in_c1": "the shared tensor immediately before all three task heads",
    "square_mapping": (
        "token i is row-major normalized square i (row i // 10, column i % 10), "
        "the accepted observation_to_tokens order, so the field maps back to "
        "board cells with a transpose and a reshape and nothing else"
    ),
    "alternatives_rejected": {
        "hidden.mean(dim=1)": "pooled to one global 128-vector; not per-square",
        "belief_output(hidden)": "already compressed 128 -> 12 by the accepted head",
        "penultimate block input": "per-square but one encoder block short of the heads",
        "policy_query / policy_key": "a single head's own projection, not the shared tensor",
    },
    "c1_frozen": True,
    "gradients_reaching_c1": False,
    "derivable_from": ["common public observation", "accepted frozen C1 weights"],
}


class Phase11BSeamError(Phase11BError):
    """A frozen C1 field cache could not be built, loaded or verified."""


def field_to_planes(field):
    """`[B, 100, 128]` C1 field -> `[B, 128, 10, 10]` convolution planes.

    The accepted `tokens_to_observation` operation applied to a 128-wide
    token field: transpose the channel axis forward, then reshape the 100
    squares row-major back into `(10, 10)`. `planes[b, c, r, k]` is
    therefore channel `c` of square `r * 10 + k`, which is the corpus's own
    `perspective_square` index — nothing has to be translated at the call
    site, and a `[B, 12, 10, 10]` output reshapes back the same way.

    `contiguous()` for the reason the accepted `observation_to_tokens` calls
    it, plus one this seam learns the hard way: the reshape after a
    transpose is representable as a *view* here, so without it the planes
    reach the convolution non-contiguous and MPS's convolution and
    batch-norm backward kernels fail outright on the strided gradient. One
    copy of a `[B, 128, 10, 10]` block, and every downstream kernel stays on
    its fast path.
    """
    import torch

    if not isinstance(field, torch.Tensor):  # pragma: no cover - defensive
        field = torch.as_tensor(field)
    if field.dim() != 3 or tuple(field.shape[1:]) != (NUM_SQUARES, C1_FEATURE_WIDTH):
        raise Phase11BSeamError(
            f"field must be [B, {NUM_SQUARES}, {C1_FEATURE_WIDTH}], got {tuple(field.shape)}"
        )
    return (
        field.transpose(1, 2)
        .reshape(field.shape[0], C1_FEATURE_WIDTH, BOARD_ROWS, BOARD_COLUMNS)
        .contiguous()
    )


def field_cache_path(root: "Path | str", split: str) -> Path:
    """Where one split's field cache lives."""
    return Path(root) / f"c1_field_{split}.npy"


def build_field_cache(
    model,
    data: dict,
    path: "Path | str",
    *,
    batch_size: int = 256,
    progress=None,
) -> dict:
    """Cache the frozen C1 field of one stored split, and describe it.

    Written straight into a `.npy` memory map rather than assembled in RAM:
    the training split is 26,898 positions x 100 squares x 128 channels,
    which is 1.4 GB, and there is no reason for both that array and the
    observations it came from to be resident at once.

    The returned block is the "feature-cache metadata" `03_AGENT_3`
    requires: what it is, what produced it, how big it is, what it hashes
    to, and what it cost.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(data["samples"])
    observations = data["observations"]
    cache = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(samples, NUM_SQUARES, C1_FEATURE_WIDTH),
    )
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, samples, batch_size):
            stop = min(start + batch_size, samples)
            encoded = encode_batch(
                model, np.asarray(observations[start:stop]), LAYER_FINAL
            )
            block = encoded.detach().cpu().numpy()
            if block.shape != (stop - start, NUM_SQUARES, C1_FEATURE_WIDTH):
                raise Phase11BSeamError(
                    f"the seam produced {block.shape}, expected "
                    f"{(stop - start, NUM_SQUARES, C1_FEATURE_WIDTH)}"
                )
            cache[start:stop] = block
            if progress is not None:
                progress(stop, samples, time.perf_counter() - started)
    cache.flush()
    seconds = round(time.perf_counter() - started, 3)
    digest = field_digest(cache)
    del cache
    return {
        "cache_version": FIELD_CACHE_VERSION,
        "seam_id": SEAM_ID,
        "layer_token": LAYER_FINAL,
        "split": data.get("split"),
        "path": str(path),
        "shape": [samples, NUM_SQUARES, C1_FEATURE_WIDTH],
        "dtype": "float32",
        "bytes": int(path.stat().st_size),
        "digest": digest,
        "seconds": seconds,
        "positions_per_second": round(samples / max(seconds, 1e-9), 1),
        "batch_size": int(batch_size),
        "derived_from": "public observations + accepted frozen C1",
        "contains_labels": False,
    }


def load_field_cache(path: "Path | str", *, expected_samples: "int | None" = None):
    """A field cache, memory-mapped read-only."""
    path = Path(path)
    if not path.exists():
        raise Phase11BSeamError(f"field cache {path} does not exist")
    cache = np.load(path, mmap_mode="r")
    if cache.ndim != 3 or cache.shape[1:] != (NUM_SQUARES, C1_FEATURE_WIDTH):
        raise Phase11BSeamError(
            f"{path} is {cache.shape}, expected [N, {NUM_SQUARES}, {C1_FEATURE_WIDTH}]"
        )
    if expected_samples is not None and cache.shape[0] != int(expected_samples):
        raise Phase11BSeamError(
            f"{path} holds {cache.shape[0]} positions, expected {expected_samples}"
        )
    return cache


def field_digest(cache: np.ndarray, *, chunk: int = 2048) -> str:
    """Content identity of a field cache, streamed rather than materialized."""
    hasher = hashlib.sha256()
    hasher.update(FIELD_CACHE_VERSION.encode())
    hasher.update(SEAM_ID.encode())
    hasher.update(str(tuple(cache.shape)).encode())
    for start in range(0, cache.shape[0], chunk):
        block = np.ascontiguousarray(cache[start : start + chunk], dtype=np.float32)
        hasher.update(block.tobytes())
    return hasher.hexdigest()


def verify_field_cache(
    model, data: dict, cache: np.ndarray, *, rows: int = 64, seed: int = 20260819
) -> dict:
    """Re-derive a random sample of the cache from the public observations.

    The cache is only legitimate if it is a function of the public
    observation and the frozen C1 and of nothing else. That is a claim about
    reproducibility, so it is measured: a random sample of positions is
    re-encoded and compared to what is stored.
    """
    import torch

    generator = np.random.default_rng(int(seed))
    samples = int(data["samples"])
    picked = np.sort(
        generator.choice(samples, size=min(int(rows), samples), replace=False)
    )
    observations = np.asarray(data["observations"])[picked]
    with torch.no_grad():
        recomputed = (
            encode_batch(model, observations, LAYER_FINAL).detach().cpu().numpy()
        )
    stored = np.asarray(cache[picked], dtype=np.float32)
    difference = float(np.abs(recomputed - stored).max())
    return {
        "rows_checked": int(picked.size),
        "max_absolute_difference": difference,
        "bit_identical": bool(difference == 0.0),
        "inputs": ["public observation", "accepted frozen C1 weights"],
    }


__all__ = [
    "BOARD_COLUMNS",
    "BOARD_ROWS",
    "FIELD_CACHE_VERSION",
    "Phase11BSeamError",
    "SEAM_DESCRIPTION",
    "SEAM_ID",
    "build_field_cache",
    "field_cache_path",
    "field_digest",
    "field_to_planes",
    "load_field_cache",
    "verify_field_cache",
]
