"""Phase 18 Stage 6B: exact state capture of the accepted `SetupBuffer` for
the joint bundle (design section 3: "the SetupBuffer's rows, counts, means,
ready flags and period").

The accepted buffer module is left byte-identical (the G2 raw-confirmation
driver pins the setup implementation's digests); this module reads and
rebuilds the buffer's rows and counters from outside. The transient processed
rows are deliberately not captured: `process` is re-run from the restored
outcomes before any minibatch is drawn, exactly as after a fresh outcome.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .g3_contract import Phase18G3Error
from .setup_buffer import SetupBuffer
from .setup_contract import SETUP_BUFFER_VERSION
from .setup_sampling import SampledSetup

_SAMPLE_ARRAY_FIELDS = (
    "network_tokens",
    "legal_masks",
    "behavior_log_probs",
    "behavior_selected_log_prob",
    "suffix_information",
    "wdl_logits",
    "entropy_prediction",
)


def sample_document(sample: SampledSetup) -> dict:
    """A `SampledSetup` as plain scalars, tuples and numpy arrays."""
    return {
        "index": int(sample.index),
        "lane": int(sample.lane),
        "root_seed": int(sample.root_seed),
        "reflection_seed": int(sample.reflection_seed),
        "reflected": bool(sample.reflected),
        "played_canonical": tuple(int(v) for v in sample.played_canonical),
        "engine_setup": tuple(int(v) for v in sample.engine_setup),
        "snapshot_digest": str(sample.snapshot_digest),
        "snapshot_iteration": int(sample.snapshot_iteration),
        **{name: np.array(getattr(sample, name), copy=True) for name in _SAMPLE_ARRAY_FIELDS},
    }


def sample_from_document(document: dict) -> SampledSetup:
    return SampledSetup(
        index=int(document["index"]),
        lane=int(document["lane"]),
        root_seed=int(document["root_seed"]),
        reflection_seed=int(document["reflection_seed"]),
        reflected=bool(document["reflected"]),
        network_tokens=np.asarray(document["network_tokens"], dtype=np.int8),
        played_canonical=tuple(int(v) for v in document["played_canonical"]),
        engine_setup=tuple(int(v) for v in document["engine_setup"]),
        legal_masks=np.asarray(document["legal_masks"], dtype=bool),
        behavior_log_probs=np.asarray(document["behavior_log_probs"], dtype=np.float32),
        behavior_selected_log_prob=np.asarray(document["behavior_selected_log_prob"], dtype=np.float32),
        suffix_information=np.asarray(document["suffix_information"], dtype=np.float32),
        wdl_logits=np.asarray(document["wdl_logits"], dtype=np.float32),
        entropy_prediction=np.asarray(document["entropy_prediction"], dtype=np.float32),
        snapshot_digest=str(document["snapshot_digest"]),
        snapshot_iteration=int(document["snapshot_iteration"]),
    )


def sample_digest(document: dict) -> str:
    hasher = hashlib.sha256()
    for key in ("index", "lane", "root_seed", "reflection_seed", "reflected", "played_canonical", "engine_setup", "snapshot_digest", "snapshot_iteration"):
        hasher.update(f"{key}={document[key]!r};".encode())
    for name in _SAMPLE_ARRAY_FIELDS:
        array = np.ascontiguousarray(document[name])
        hasher.update(f"{name}|{array.dtype}|{array.shape}".encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def capture_buffer_state(buffer: SetupBuffer) -> dict:
    """Every row, counter and flag of the buffer, as a document a bundle can carry."""
    return {
        "buffer_version": buffer.version,
        "storage_duration": int(buffer.storage_duration),
        "need_pool": bool(buffer.need_pool),
        "samples": [sample_document(sample) for sample in buffer._samples],
        "period_added": buffer._period_added.astype(np.int64).tolist(),
        "counts": buffer._counts.astype(np.int64).tolist(),
        "mean_one_hot": buffer._mean_one_hot.astype(np.float64).tolist(),
        "ready": buffer._ready.astype(bool).tolist(),
        "counters": {
            "duplicates_collapsed_total": int(buffer.duplicates_collapsed_total),
            "attribution_failures": int(buffer.attribution_failures),
            "pools_added": int(buffer.pools_added),
            "outcomes_added": int(buffer.outcomes_added),
        },
    }


def restore_buffer_state(payload: dict, *, device: str = "cpu") -> SetupBuffer:
    """Rebuild a buffer from `capture_buffer_state`, refusing a foreign version."""
    if payload.get("buffer_version") != SETUP_BUFFER_VERSION:
        raise Phase18G3Error(
            f"buffer state under version {payload.get('buffer_version')!r}, not {SETUP_BUFFER_VERSION!r}"
        )
    buffer = SetupBuffer(storage_duration=int(payload["storage_duration"]), device=device)
    buffer._samples = [sample_from_document(entry) for entry in payload["samples"]]
    rows = len(buffer._samples)
    buffer._period_added = np.asarray(payload["period_added"], dtype=np.int64).reshape(rows)
    buffer._counts = np.asarray(payload["counts"], dtype=np.int64).reshape(rows)
    buffer._mean_one_hot = np.asarray(payload["mean_one_hot"], dtype=np.float64).reshape(rows, 3)
    buffer._ready = np.asarray(payload["ready"], dtype=bool).reshape(rows)
    buffer._lookup = {sample.content_fingerprint: index for index, sample in enumerate(buffer._samples)}
    if len(buffer._lookup) != rows:
        raise Phase18G3Error("restored buffer rows repeat a fingerprint")
    buffer._processed = None
    buffer.need_pool = bool(payload["need_pool"])
    counters = payload["counters"]
    buffer.duplicates_collapsed_total = int(counters["duplicates_collapsed_total"])
    buffer.attribution_failures = int(counters["attribution_failures"])
    buffer.pools_added = int(counters["pools_added"])
    buffer.outcomes_added = int(counters["outcomes_added"])
    return buffer


def buffer_state_digest(buffer: SetupBuffer) -> str:
    """sha256 over the captured state, arrays as exact bytes."""
    hasher = hashlib.sha256()
    state = capture_buffer_state(buffer)
    for key in ("buffer_version", "storage_duration", "need_pool", "counters"):
        hasher.update(repr(state[key]).encode())
    for name in ("period_added", "counts", "ready"):
        hasher.update(np.asarray(state[name]).tobytes())
    hasher.update(np.asarray(state["mean_one_hot"], dtype=np.float64).tobytes())
    for entry in state["samples"]:
        hasher.update(sample_digest(entry).encode())
    return hasher.hexdigest()


__all__ = [
    "buffer_state_digest",
    "capture_buffer_state",
    "restore_buffer_state",
    "sample_digest",
    "sample_document",
    "sample_from_document",
]
