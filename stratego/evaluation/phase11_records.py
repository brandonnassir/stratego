"""Phase 11 Agent 2: primitive prediction storage and its manifest.

Specification sources:

- `02_AGENT_2_BELIEF_EVALUATOR_BASELINES_VALIDATION.md` section 2
  ("Prediction recorder"), Deliverables ("Storage can be external; identity
  must be path-independent")
- Agent 1's `phase11_belief_contract_v1`, section `prediction_record`

Two files per game, and that is the point
-----------------------------------------
Every game writes a **public shard** during play and a **truth shard**
afterwards, from a separate replay that runs after every learned and
baseline vector already exists. The split is not tidiness: it is the
public/privileged boundary made physical. A reader that never opens
`*_truth.npz` has provably never seen a hidden rank, and the scoring path
is the only thing in Phase 11 that opens one.

Primitives, not summaries
-------------------------
The public shard stores the head's raw float32 logit rows rather than the
probability vectors. The frozen learned 12-vector is the float64 softmax of
exactly those rows, so storing logits is strictly more primitive: the audit
path recomputes the probabilities instead of trusting them, and the baseline
is recomputed from the stored counts and masks instead of being read back.
Nothing in `agent_02_predictive_metrics.json` is recoverable only from a
summary.

```text
decisions   decision_index, public_state_identity, observation_sha256,
            remaining_counts, event_offset          (CSR into the events)
game        action_history                          (public: every move is seen)
events      piece_slot, piece_square, perspective_square, piece_moved,
            legal_rank_mask, belief_logits
truth       true_rank_index                          (privileged shard)
```

Identity is path-independent: `shard_digest` hashes content, and the
manifest records logical game identity, so relocating the store to an
external volume changes nothing an audit compares.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..training.phase11_contract import (
    PREDICTION_RECORD_FIELDS,
    PREDICTION_RECORD_VERSION,
    PRIVILEGED_RECORD_FIELDS,
    Phase11ContractError,
    RANK_COUNT,
    progress_bucket,
)
from ..training.phase11_seed import phase11_prediction_id

#: The storage format identity, recorded in every manifest.
PREDICTION_STORE_VERSION = "phase11_prediction_store_v1"

#: The tracked pointer naming the prediction-store root, so the bytes can
#: live on an external volume while the repository records where.
STORE_POINTER_RELATIVE_PATH = "data/phase11_prediction_root.txt"

#: The frozen array names of a public shard, in digest order.
PUBLIC_SHARD_ARRAYS = (
    "decision_index",
    "public_state_identity",
    "observation_sha256",
    "remaining_counts",
    "event_offset",
    "piece_slot",
    "piece_square",
    "perspective_square",
    "piece_moved",
    "legal_rank_mask",
    "belief_logits",
    "action_history",
)

#: The frozen array names of a truth shard.
TRUTH_SHARD_ARRAYS = ("true_rank_index",)


class Phase11StoreError(Phase11ContractError):
    """A prediction shard could not be written, read or verified."""


def _hex_rows(digests: "list[str]") -> np.ndarray:
    """`[n, 32] uint8` from n 64-hex-digit SHA-256 strings."""
    if not digests:
        return np.zeros((0, 32), dtype=np.uint8)
    return np.stack(
        [np.frombuffer(bytes.fromhex(value), dtype=np.uint8) for value in digests]
    )


def _hex_of(row) -> str:
    return bytes(np.asarray(row, dtype=np.uint8)).hex()


# ---------------------------------------------------------------------------
# Building one game's public shard
# ---------------------------------------------------------------------------


class Phase11GameRecorder:
    """Accumulates one game's public prediction rows. Holds no truth.

    The recorder is handed only public products — the document identity,
    the public counts and masks, and the head's logit rows — so a hidden
    rank has no field to arrive in.
    """

    def __init__(self, game_meta: dict) -> None:
        self.meta = dict(game_meta)
        self.decision_index: list[int] = []
        self.state_identity: list[str] = []
        self.observation_sha: list[str] = []
        self.remaining_counts: list[tuple[int, ...]] = []
        self.event_offset: list[int] = [0]
        self.piece_slot: list[int] = []
        self.piece_square: list[int] = []
        self.perspective_square: list[int] = []
        self.piece_moved: list[int] = []
        self.legal_rank_mask: list[tuple[int, ...]] = []
        self.belief_logits: list[np.ndarray] = []
        self.empty_decisions = 0
        #: The game's absolute action ids. Public information — both
        #: players watch every move — and what lets the audits replay a
        #: game from its shard without the match runner.
        self.action_history: tuple[int, ...] = ()

    def record_decision(
        self,
        *,
        decision_index: int,
        public_state_identity: str,
        observation_sha256: str,
        remaining_counts,
        events: "list[dict]",
    ) -> None:
        """One observer decision and its hidden-piece events, in slot order."""
        self.decision_index.append(int(decision_index))
        self.state_identity.append(str(public_state_identity))
        self.observation_sha.append(str(observation_sha256))
        counts = tuple(int(value) for value in remaining_counts)
        if len(counts) != RANK_COUNT:
            raise Phase11StoreError(f"remaining_counts has length {len(counts)}")
        self.remaining_counts.append(counts)
        if not events:
            self.empty_decisions += 1
        for event in sorted(events, key=lambda item: int(item["piece_slot"])):
            self.piece_slot.append(int(event["piece_slot"]))
            self.piece_square.append(int(event["piece_square"]))
            self.perspective_square.append(int(event["perspective_square"]))
            self.piece_moved.append(1 if event["piece_moved"] else 0)
            mask = tuple(int(value) for value in event["legal_rank_mask"])
            if len(mask) != RANK_COUNT:
                raise Phase11StoreError(f"legal_rank_mask has length {len(mask)}")
            self.legal_rank_mask.append(mask)
            row = np.asarray(event["belief_logits"], dtype=np.float32)
            if row.shape != (RANK_COUNT,):
                raise Phase11StoreError(f"belief logits have shape {row.shape}")
            self.belief_logits.append(row)
        self.event_offset.append(len(self.piece_slot))

    @property
    def decisions(self) -> int:
        return len(self.decision_index)

    @property
    def events(self) -> int:
        return len(self.piece_slot)

    def arrays(self) -> dict:
        """The public shard's arrays, in the frozen order."""
        events = self.events
        payload = {
            "decision_index": np.asarray(self.decision_index, dtype=np.int32),
            "public_state_identity": _hex_rows(self.state_identity),
            "observation_sha256": _hex_rows(self.observation_sha),
            "remaining_counts": np.asarray(
                self.remaining_counts, dtype=np.int16
            ).reshape(self.decisions, RANK_COUNT),
            "event_offset": np.asarray(self.event_offset, dtype=np.int32),
            "piece_slot": np.asarray(self.piece_slot, dtype=np.int16),
            "piece_square": np.asarray(self.piece_square, dtype=np.int16),
            "perspective_square": np.asarray(self.perspective_square, dtype=np.int16),
            "piece_moved": np.asarray(self.piece_moved, dtype=np.uint8),
            "legal_rank_mask": np.asarray(
                self.legal_rank_mask, dtype=np.uint8
            ).reshape(events, RANK_COUNT),
            "belief_logits": (
                np.stack(self.belief_logits)
                if self.belief_logits
                else np.zeros((0, RANK_COUNT), dtype=np.float32)
            ).astype(np.float32),
            "action_history": np.asarray(self.action_history, dtype=np.int32),
        }
        if tuple(payload) != PUBLIC_SHARD_ARRAYS:
            raise Phase11StoreError("public shard arrays drifted from the frozen order")
        return payload


# ---------------------------------------------------------------------------
# Shard files
# ---------------------------------------------------------------------------


def shard_digest(arrays: dict, names: "tuple[str, ...]") -> str:
    """Content digest of a shard: names, dtypes, shapes and bytes, in order."""
    hasher = hashlib.sha256()
    hasher.update(PREDICTION_STORE_VERSION.encode())
    for name in names:
        array = np.ascontiguousarray(arrays[name])
        hasher.update(name.encode())
        hasher.update(str(array.dtype).encode())
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def public_shard_path(root: "str | Path", game_id: str) -> Path:
    return Path(root) / "public" / f"{game_file_stem(game_id)}.npz"


def truth_shard_path(root: "str | Path", game_id: str) -> Path:
    return Path(root) / "truth" / f"{game_file_stem(game_id)}.npz"


def game_file_stem(game_id: str) -> str:
    """A filesystem-safe stem. A *path*, therefore never a logical identity."""
    return hashlib.sha256(game_id.encode()).hexdigest()[:32]


def write_public_shard(root: "str | Path", recorder: Phase11GameRecorder) -> dict:
    """Write one game's public shard and return its manifest entry."""
    arrays = recorder.arrays()
    path = public_shard_path(root, recorder.meta["game_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    entry = dict(recorder.meta)
    entry.update(
        {
            "record_version": PREDICTION_RECORD_VERSION,
            "store_version": PREDICTION_STORE_VERSION,
            "decisions": recorder.decisions,
            "events": recorder.events,
            "empty_decisions": recorder.empty_decisions,
            "public_shard_digest": shard_digest(arrays, PUBLIC_SHARD_ARRAYS),
        }
    )
    return entry


def read_public_shard(root: "str | Path", game_id: str) -> dict:
    path = public_shard_path(root, game_id)
    if not path.exists():
        raise Phase11StoreError(f"the public shard of {game_id} is missing at {path}")
    with np.load(path) as handle:
        return {name: handle[name] for name in PUBLIC_SHARD_ARRAYS}


def write_truth_shard(root: "str | Path", game_id: str, true_rank_index) -> dict:
    """Write one game's privileged shard. Called only by the truth pass."""
    arrays = {"true_rank_index": np.asarray(true_rank_index, dtype=np.int8)}
    path = truth_shard_path(root, game_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {
        "game_id": game_id,
        "events": int(arrays["true_rank_index"].size),
        "truth_shard_digest": shard_digest(arrays, TRUTH_SHARD_ARRAYS),
        "privileged_fields": list(PRIVILEGED_RECORD_FIELDS),
    }


def read_truth_shard(root: "str | Path", game_id: str) -> np.ndarray:
    path = truth_shard_path(root, game_id)
    if not path.exists():
        raise Phase11StoreError(f"the truth shard of {game_id} is missing at {path}")
    with np.load(path) as handle:
        return handle["true_rank_index"]


# ---------------------------------------------------------------------------
# The logical record
# ---------------------------------------------------------------------------


def model_identity(model_state_digest: str, belief_head_digest: str) -> str:
    """The frozen model identity every prediction record carries."""
    return (
        f"selfplay_c1_v1|state={model_state_digest}|belief={belief_head_digest}"
    )


def prediction_identity(
    prediction_id: str,
    public_state_identity: str,
    model_id: str,
    logits: np.ndarray,
    mask: np.ndarray,
    counts: np.ndarray,
) -> str:
    """Content identity of one prediction record. Public content only."""
    hasher = hashlib.sha256()
    hasher.update(PREDICTION_RECORD_VERSION.encode())
    hasher.update(prediction_id.encode())
    hasher.update(public_state_identity.encode())
    hasher.update(model_id.encode())
    hasher.update(np.ascontiguousarray(logits, dtype=np.float32).tobytes())
    hasher.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    hasher.update(np.ascontiguousarray(counts, dtype=np.int16).tobytes())
    return hasher.hexdigest()


def iter_records(
    manifest_entry: dict,
    arrays: dict,
    truth: "np.ndarray | None" = None,
    *,
    model_id: str,
):
    """Yield the frozen logical prediction records of one game.

    `truth` is optional and separate: without it the records carry
    `true_rank_index=None`, which is exactly what the production side of
    the boundary produces.
    """
    from .phase11_belief import softmax_float64
    from .phase11_baselines import remaining_count_distribution

    game_id = manifest_entry["game_id"]
    offsets = arrays["event_offset"]
    for decision, start in enumerate(offsets[:-1]):
        stop = int(offsets[decision + 1])
        index = int(arrays["decision_index"][decision])
        state_identity = _hex_of(arrays["public_state_identity"][decision])
        counts = arrays["remaining_counts"][decision]
        for position in range(int(start), stop):
            slot = int(arrays["piece_slot"][position])
            mask = arrays["legal_rank_mask"][position]
            logits = arrays["belief_logits"][position]
            record = {
                "record_version": PREDICTION_RECORD_VERSION,
                "bank_version": manifest_entry["bank_version"],
                "case_id": manifest_entry["case_id"],
                "game_id": game_id,
                "prediction_id": phase11_prediction_id(game_id, index, slot),
                "decision_index": index,
                "observer_color": manifest_entry["observer_color"],
                "opponent_stratum": manifest_entry["opponent_stratum"],
                "opponent_setup_source": manifest_entry["opponent_setup_source"],
                "public_state_identity": state_identity,
                "observation_sha256": _hex_of(arrays["observation_sha256"][decision]),
                "piece_slot": slot,
                "piece_square": int(arrays["piece_square"][position]),
                "piece_moved": bool(arrays["piece_moved"][position]),
                "progress_bucket": progress_bucket(index),
                "legal_rank_mask": [int(value) for value in mask],
                "remaining_counts": [int(value) for value in counts],
                "learned_probabilities": softmax_float64(logits).tolist(),
                "baseline_probabilities": remaining_count_distribution(
                    counts, mask
                ).tolist(),
                "true_rank_index": (
                    None if truth is None else int(truth[position])
                ),
                "model_identity": model_id,
                "prediction_identity": prediction_identity(
                    phase11_prediction_id(game_id, index, slot),
                    state_identity,
                    model_id,
                    logits,
                    mask,
                    counts,
                ),
            }
            if tuple(record) != PREDICTION_RECORD_FIELDS:
                raise Phase11StoreError(
                    "the prediction record fields drifted from the frozen schema"
                )
            yield record


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def manifest_digest(manifest: dict) -> str:
    """Digest over the manifest's logical content — never over a path."""
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in ("store_root", "written_at", "duration_seconds")
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_manifest(root: "str | Path", manifest: dict) -> Path:
    path = Path(root) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return path


def read_manifest(root: "str | Path") -> dict:
    path = Path(root) / "manifest.json"
    if not path.exists():
        raise Phase11StoreError(f"no prediction manifest at {path}")
    return json.loads(path.read_text())


def store_root(repository_root: "str | Path") -> Path:
    """The prediction-store root, from the tracked pointer if one exists.

    A pointer naming an absent volume is BLOCKED, never silently replaced
    by an internal path — the storage-semantics rule Agent 1 froze.
    """
    base = Path(repository_root)
    pointer = base / STORE_POINTER_RELATIVE_PATH
    if pointer.exists():
        named = Path(pointer.read_text().strip()).expanduser()
        if not named.parent.exists():
            raise Phase11StoreError(
                f"the prediction-store pointer names {named}, whose volume is "
                "not present; this is BLOCKED, not a reason to use another path"
            )
        return named
    return base / "data" / "phase11" / "agent02" / "validation_predictions"


__all__ = [
    "PREDICTION_RECORD_VERSION",
    "PREDICTION_STORE_VERSION",
    "PUBLIC_SHARD_ARRAYS",
    "Phase11GameRecorder",
    "Phase11StoreError",
    "STORE_POINTER_RELATIVE_PATH",
    "TRUTH_SHARD_ARRAYS",
    "game_file_stem",
    "iter_records",
    "manifest_digest",
    "model_identity",
    "prediction_identity",
    "public_shard_path",
    "read_manifest",
    "read_public_shard",
    "read_truth_shard",
    "shard_digest",
    "store_root",
    "truth_shard_path",
    "write_manifest",
    "write_public_shard",
    "write_truth_shard",
]
