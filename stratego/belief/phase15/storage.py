"""Phase 15 corpus storage: public bytes here, privileged labels there.

Specification source: `01_AGENT_1_BELIEF_HEAD_TRAINING.md` section 7.

The separation is a directory, and the loader honours it
--------------------------------------------------------
A split is written as two directories. `public/` holds everything a model
may consume — the 127x10x10 observations, the public-state identities, the
remaining inventory, the legal-rank masks, the cell labels. `privileged/`
holds exactly one array: the true rank of every hidden piece.
:func:`load_split` returns the public half and takes `labels=False` by
default, so reaching the labels is an act, not an accident.

Ragged pieces, CSR offsets
--------------------------
A decision has between 1 and 40 hidden pieces, so the piece-level arrays
are one flat run indexed by `piece_offset[i]:piece_offset[i + 1]`. No
padding, no ragged object arrays, and one `int64` offset vector that makes
every per-sample slice a view.

Wall clock is outside the identity
-----------------------------------
:func:`corpus_digest` hashes every stored array in a frozen order together
with the manifest's logical fields; durations are attached to the manifest
*after* the digest is taken. This is the Phase 11 `manifest_digest` defect
not repeated.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ...setups.families import FAMILY_KEYS
from .contract import (
    CORPUS_COLORS,
    CORPUS_SPLITS,
    CORPUS_VERSION,
    OBSERVATION_SHAPE,
    OPPONENTS,
    POLICY_SOURCES,
    PRIVILEGED_DIRECTORY,
    PUBLIC_DIRECTORY,
    RANK_COUNT,
    SETUP_SOURCES,
    Phase15Error,
)

#: The stored-format identity. A change to any array name, dtype or order
#: is a new version, never a silent edit.
CORPUS_FORMAT_VERSION = "phase15_belief_corpus_store_v1"

#: The public per-sample arrays, in digest order: `name -> dtype`.
PUBLIC_SAMPLE_ARRAYS = {
    "game_ordinal": "int32",
    "observer_model": "int8",
    "opponent": "int8",
    "setup_source": "int8",
    "observer_family": "int8",
    "opponent_family": "int8",
    "observer_color": "int8",
    "decision_index": "int32",
    "total_moves": "int32",
    "piece_offset": "int64",
    "remaining_counts": "int16",
    "target_mask": "bool",
}

#: The public per-piece arrays, in digest order.
PUBLIC_PIECE_ARRAYS = {
    "piece_slot": "int16",
    "piece_square": "int16",
    "perspective_square": "int16",
    "piece_moved": "bool",
    "legal_rank_mask": "bool",
}

#: The privileged array. One name, one dtype, one directory.
PRIVILEGED_ARRAYS = {"true_rank": "int8"}

#: Text sidecars in `public/`. Identities, one per line.
GAME_IDS_FILE = "game_ids.txt"
STATE_IDENTITIES_FILE = "public_state_identities.txt"
OBSERVATIONS_FILE = "observations.f32"

_PRIVILEGED_NOTE = (
    "Phase 15 privileged belief labels: the true rank of every hidden\n"
    "opponent piece of phase15_belief_corpus_v1. These are SUPERVISED\n"
    "TARGETS ONLY. They must never enter a model-input path, a search\n"
    "prior, a setup-selection feature or any public metadata. The public\n"
    f"half of this split lives in ../{PUBLIC_DIRECTORY}/ and is the only\n"
    "half a model may read.\n"
)

_OBSERVER_INDEX = {name: index for index, name in enumerate(POLICY_SOURCES)}
_OPPONENT_INDEX = {name: index for index, name in enumerate(OPPONENTS)}
_SOURCE_INDEX = {name: index for index, name in enumerate(SETUP_SOURCES)}
_COLOR_INDEX = {name: index for index, name in enumerate(CORPUS_COLORS)}
_FAMILY_INDEX = {name: index for index, name in enumerate(FAMILY_KEYS)}


class Phase15StorageError(Phase15Error):
    """A corpus split could not be written, read back or verified."""


def split_root(root: "Path | str", split: str) -> Path:
    if split not in CORPUS_SPLITS:
        raise Phase15StorageError(f"unknown split {split!r}")
    return Path(root) / split


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


class SplitWriter:
    """Streams one split to disk: observations first, labels last.

    Observations are appended to a raw binary file as they are produced, so
    a 120,000-position split never has to fit in memory at once. Everything
    else is small enough to accumulate and write in one go.
    """

    def __init__(self, root: "Path | str", split: str) -> None:
        if split not in CORPUS_SPLITS:
            raise Phase15StorageError(f"unknown split {split!r}")
        self.split = split
        self.root = split_root(root, split)
        self.public = self.root / PUBLIC_DIRECTORY
        self.privileged = self.root / PRIVILEGED_DIRECTORY
        self.public.mkdir(parents=True, exist_ok=True)
        self.privileged.mkdir(parents=True, exist_ok=True)
        self._observations = open(self.public / OBSERVATIONS_FILE, "wb")
        self._sample: dict[str, list] = {name: [] for name in PUBLIC_SAMPLE_ARRAYS}
        self._piece: dict[str, list] = {name: [] for name in PUBLIC_PIECE_ARRAYS}
        self._true_rank: list[int] = []
        self._game_ids: list[str] = []
        self._identities: list[str] = []
        self._games: list[dict] = []
        self._offset = 0
        self._sample["piece_offset"].append(0)
        self.samples = 0
        self.pieces = 0
        self.closed = False

    def add_game(self, plan, result, eligible: int, samples: "list[dict]") -> int:
        """Append one game's selected samples. Returns the split's new size."""
        if self.closed:  # pragma: no cover - defensive
            raise Phase15StorageError("this split writer is closed")
        ordinal = len(self._games)
        self._games.append(
            {
                "game_ordinal": ordinal,
                "game_id": plan.game_id,
                "observer_model": plan.observer_model,
                "opponent": plan.opponent,
                "setup_source": plan.setup_source,
                "observer_color": plan.observer_color,
                "observer_family_key": plan.observer_family_key,
                "opponent_family_key": plan.opponent_family_key,
                "observer_base_setup_id": plan.observer_base_setup_id,
                "opponent_base_setup_id": plan.opponent_base_setup_id,
                "observer_setup_branch": plan.observer_setup_branch,
                "opponent_setup_branch": plan.opponent_setup_branch,
                "plies": int(result.plies),
                "eligible_decisions": int(eligible),
                "samples": len(samples),
                "winner": result.winner,
                "draw": bool(result.draw),
            }
        )
        self._game_ids.append(plan.game_id)
        for sample in samples:
            observation = np.ascontiguousarray(sample["observation"], dtype=np.float32)
            if observation.shape != OBSERVATION_SHAPE:  # pragma: no cover
                raise Phase15StorageError(f"observation shape {observation.shape}")
            self._observations.write(observation.tobytes())
            self._sample["game_ordinal"].append(ordinal)
            self._sample["observer_model"].append(_OBSERVER_INDEX[sample["observer_model"]])
            self._sample["opponent"].append(_OPPONENT_INDEX[sample["opponent"]])
            self._sample["setup_source"].append(_SOURCE_INDEX[sample["setup_source"]])
            self._sample["observer_family"].append(
                _FAMILY_INDEX[sample["observer_family_key"]]
            )
            self._sample["opponent_family"].append(
                _FAMILY_INDEX[sample["opponent_family_key"]]
            )
            self._sample["observer_color"].append(_COLOR_INDEX[sample["observer_color"]])
            self._sample["decision_index"].append(sample["decision_index"])
            self._sample["total_moves"].append(sample["total_moves"])
            self._sample["remaining_counts"].append(sample["remaining_counts"])
            self._sample["target_mask"].append(sample["target_mask"])
            self._identities.append(sample["public_state_identity"])
            for piece in sample["pieces"]:
                self._piece["piece_slot"].append(piece["piece_slot"])
                self._piece["piece_square"].append(piece["piece_square"])
                self._piece["perspective_square"].append(piece["perspective_square"])
                self._piece["piece_moved"].append(piece["piece_moved"])
                self._piece["legal_rank_mask"].append(piece["legal_rank_mask"])
                self._true_rank.append(piece["true_rank"])
            self._offset += len(sample["pieces"])
            self._sample["piece_offset"].append(self._offset)
            self.samples += 1
            self.pieces += len(sample["pieces"])
        return self.samples

    def close(self) -> dict:
        """Write every array and return this split's manifest block."""
        if self.closed:  # pragma: no cover - defensive
            raise Phase15StorageError("this split writer is already closed")
        self._observations.close()
        self.closed = True

        for name, dtype in PUBLIC_SAMPLE_ARRAYS.items():
            np.save(self.public / f"{name}.npy", np.asarray(self._sample[name], dtype=dtype))
        for name, dtype in PUBLIC_PIECE_ARRAYS.items():
            array = np.asarray(self._piece[name], dtype=dtype)
            if name == "legal_rank_mask" and array.size == 0:  # pragma: no cover
                array = array.reshape(0, RANK_COUNT)
            np.save(self.public / f"{name}.npy", array)
        np.save(
            self.privileged / "true_rank.npy", np.asarray(self._true_rank, dtype="int8")
        )
        (self.privileged / "README.txt").write_text(_PRIVILEGED_NOTE)
        (self.public / GAME_IDS_FILE).write_text("\n".join(self._game_ids) + "\n")
        (self.public / STATE_IDENTITIES_FILE).write_text(
            "\n".join(self._identities) + "\n"
        )
        (self.public / "games.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self._games)
        )
        return {
            "split": self.split,
            "games": len(self._games),
            "samples": int(self.samples),
            "pieces": int(self.pieces),
            "hidden_pieces_per_sample": (
                round(self.pieces / self.samples, 4) if self.samples else 0.0
            ),
            "observation_bytes": int(
                (self.public / OBSERVATIONS_FILE).stat().st_size
            ),
        }


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def load_split(
    root: "Path | str", split: str, *, labels: bool = False, mmap: bool = True
) -> dict:
    """One stored split. The privileged labels only when explicitly asked."""
    root = split_root(root, split)
    public = root / PUBLIC_DIRECTORY
    if not public.is_dir():
        raise Phase15StorageError(f"no public half of split {split!r} at {public}")
    mode = "r" if mmap else None
    data: dict = {"split": split, "root": str(root)}
    for name in PUBLIC_SAMPLE_ARRAYS:
        data[name] = np.load(public / f"{name}.npy", mmap_mode=mode)
    for name in PUBLIC_PIECE_ARRAYS:
        data[name] = np.load(public / f"{name}.npy", mmap_mode=mode)
    data["samples"] = int(data["game_ordinal"].shape[0])
    data["pieces"] = int(data["piece_slot"].shape[0])
    data["games"] = int(data["game_ordinal"].max()) + 1 if data["samples"] else 0
    count = data["samples"]
    data["observations"] = np.memmap(
        public / OBSERVATIONS_FILE,
        dtype=np.float32,
        mode="r",
        shape=(count, *OBSERVATION_SHAPE),
    )
    data["game_ids"] = (public / GAME_IDS_FILE).read_text().split()
    data["public_state_identities"] = (public / STATE_IDENTITIES_FILE).read_text().split()
    if labels:
        data["true_rank"] = np.load(
            root / PRIVILEGED_DIRECTORY / "true_rank.npy", mmap_mode=mode
        )
    return data


def label_names() -> dict:
    """The integer codes the stored label arrays use, for a manifest."""
    return {
        "observer_model": list(POLICY_SOURCES),
        "opponent": list(OPPONENTS),
        "setup_source": list(SETUP_SOURCES),
        "observer_color": list(CORPUS_COLORS),
        "family": list(FAMILY_KEYS),
    }


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def file_sha256(path: "Path | str", *, chunk: int = 1 << 22) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def split_digest(root: "Path | str", split: str) -> dict:
    """`relative path -> sha256` of every stored file of one split."""
    base = split_root(root, split)
    digests = {}
    for path in sorted(base.rglob("*")):
        if path.is_file():
            digests[str(path.relative_to(base))] = file_sha256(path)
    return digests


def corpus_digest(manifest: dict) -> str:
    """The content identity of a whole corpus.

    Hashes the logical fields and every per-split file digest in a frozen
    order. Durations are deliberately not part of the payload.
    """
    payload = {
        "corpus_version": manifest["corpus_version"],
        "corpus_format_version": manifest["corpus_format_version"],
        "run_version": manifest["run_version"],
        "identity_version": manifest["identity_version"],
        "seeds": manifest["seeds"],
        "splits": {
            split: {
                key: block[key]
                for key in ("split", "games", "samples", "pieces", "file_digests")
            }
            for split, block in sorted(manifest["splits"].items())
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_manifest(root: "Path | str", manifest: dict) -> Path:
    path = Path(root).parent / f"{CORPUS_VERSION}_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def read_manifest(root: "Path | str") -> dict:
    path = Path(root).parent / f"{CORPUS_VERSION}_manifest.json"
    if not path.is_file():
        raise Phase15StorageError(f"no corpus manifest at {path}")
    return json.loads(path.read_text())


__all__ = [
    "CORPUS_FORMAT_VERSION",
    "OBSERVATIONS_FILE",
    "PRIVILEGED_ARRAYS",
    "PUBLIC_PIECE_ARRAYS",
    "PUBLIC_SAMPLE_ARRAYS",
    "Phase15StorageError",
    "SplitWriter",
    "corpus_digest",
    "file_sha256",
    "label_names",
    "load_split",
    "read_manifest",
    "split_digest",
    "split_root",
    "write_manifest",
]
