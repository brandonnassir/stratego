"""Phase 11B corpus storage: public bytes here, privileged labels there.

Specification sources:

- `00_PHASE_11B_OVERVIEW.md` ("Canonical Sample Contents" — "Public inputs
  and privileged labels must be stored separately")
- `01_AGENT_1_ATTACHED_BELIEF_HEAD.md` ("Data Boundary", "Required
  Artifacts" — "Preserve the common corpus ... so later agents can reuse it
  byte-for-byte")

The separation is a directory, and the loader honours it
--------------------------------------------------------
A split is written as two directories. `public/` holds everything a model
may consume — the 127x10x10 observations, the public-state identities, the
remaining inventory, the legal-rank masks, the slice keys. `privileged/`
holds exactly one array: the true rank of every hidden piece.
:func:`load_split` returns the public half and takes `labels=False` by
default, so reaching the labels is an act, not an accident.

Byte-for-byte reuse
-------------------
:func:`corpus_digest` hashes every stored array in a frozen order together
with the manifest's logical fields. Agents 2-5 recompute it and compare;
equality is the whole guarantee that four experiments were scored on one
corpus. Wall-clock durations live *outside* the digested manifest — the
Phase 11 `manifest_digest` defect, not repeated here.

Ragged pieces, CSR offsets
--------------------------
A decision has between 1 and 40 hidden pieces, so the piece-level arrays
are one flat run indexed by `piece_offset[i]:piece_offset[i + 1]`. No
padding, no ragged object arrays, and one `int64` offset vector that makes
every per-sample slice a view.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .contract import (
    CORPUS_COLORS,
    CORPUS_SOURCES,
    CORPUS_SPLITS,
    CORPUS_STRATA,
    CORPUS_VERSION,
    OBSERVATION_SHAPE,
    PRIVILEGED_DIRECTORY,
    PUBLIC_DIRECTORY,
    Phase11BError,
)

#: The stored-format identity. A change to any array name, dtype or order
#: is a new version, never a silent edit.
CORPUS_FORMAT_VERSION = "phase11b_corpus_store_v1"

#: The public arrays, in digest order: `name -> dtype`.
PUBLIC_SAMPLE_ARRAYS = {
    "game_ordinal": "int32",
    "stratum": "int8",
    "setup_source": "int8",
    "observer_color": "int8",
    "decision_index": "int32",
    "total_moves": "int32",
    "piece_offset": "int64",
    "remaining_counts": "int16",
    "target_mask": "bool",
}

PUBLIC_PIECE_ARRAYS = {
    "piece_slot": "int16",
    "piece_square": "int16",
    "perspective_square": "int16",
    "piece_moved": "bool",
    "legal_rank_mask": "bool",
}

#: The privileged array. One name, one dtype, one directory.
PRIVILEGED_ARRAYS = {"true_rank": "int8"}

_PRIVILEGED_NOTE = (
    "Phase 11B privileged belief labels: the true rank of every hidden\n"
    "opponent piece of the common corpus. These are SUPERVISED TARGETS ONLY.\n"
    "They must never enter a model-input path. The public half of this split\n"
    f"lives in ../{PUBLIC_DIRECTORY}/ and is the only half a model may read.\n"
)

_STRATUM_INDEX = {name: index for index, name in enumerate(CORPUS_STRATA)}
_SOURCE_INDEX = {name: index for index, name in enumerate(CORPUS_SOURCES)}
_COLOR_INDEX = {name: index for index, name in enumerate(CORPUS_COLORS)}


class Phase11BStorageError(Phase11BError):
    """A corpus split could not be written, read back or verified."""


def split_root(root: "Path | str", split: str) -> Path:
    if split not in CORPUS_SPLITS:
        raise Phase11BStorageError(f"unknown split {split!r}")
    return Path(root) / split


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


class SplitWriter:
    """Streams one split to disk: observations first, labels last.

    Observations are appended to a raw binary file as they are produced, so
    a 2,048-game corpus never has to fit in memory at once. Everything else
    is small enough to accumulate and write in one go.
    """

    def __init__(self, root: "Path | str", split: str) -> None:
        self.split = split
        self.root = split_root(root, split)
        self.public = self.root / PUBLIC_DIRECTORY
        self.privileged = self.root / PRIVILEGED_DIRECTORY
        self.public.mkdir(parents=True, exist_ok=True)
        self.privileged.mkdir(parents=True, exist_ok=True)
        self._observations = open(self.public / "observations.f32", "wb")
        self._identities: list[str] = []
        self._games: list[dict] = []
        self._sample: dict[str, list] = {name: [] for name in PUBLIC_SAMPLE_ARRAYS}
        self._piece: dict[str, list] = {name: [] for name in PUBLIC_PIECE_ARRAYS}
        self._true_rank: list[int] = []
        self._sample["piece_offset"].append(0)
        self.samples = 0
        self.pieces = 0

    def add_game(self, plan, result, decisions: "list[dict]", samples: "list[dict]") -> None:
        """Append one game's public record and its selected samples."""
        ordinal = len(self._games)
        eligible = sum(1 for row in decisions if row["unresolved"] > 0)
        self._games.append(
            {
                "game_ordinal": ordinal,
                "game_id": plan.game_id,
                "split": plan.split,
                "stratum": plan.stratum,
                "setup_source": plan.setup_source,
                "observer_color": plan.observer_color,
                "match_seed": int(plan.match_seed),
                "plies": int(result.plies),
                "terminal_reason": result.terminal_reason,
                "observer_result": result.candidate_result,
                "observer_decisions": len(decisions),
                "eligible_decisions": eligible,
                "sampled_decisions": len(samples),
                "replay_digest": result.replay_digest,
            }
        )
        for sample in samples:
            self._add_sample(ordinal, sample)

    def _add_sample(self, ordinal: int, sample: dict) -> None:
        observation = np.ascontiguousarray(sample["observation"], dtype=np.float32)
        if observation.shape != OBSERVATION_SHAPE:  # pragma: no cover - checked upstream
            raise Phase11BStorageError(f"observation shape {observation.shape} is not frozen")
        self._observations.write(observation.tobytes())
        self._identities.append(sample["public_state_identity"])

        self._sample["game_ordinal"].append(ordinal)
        self._sample["stratum"].append(_STRATUM_INDEX[sample["stratum"]])
        self._sample["setup_source"].append(_SOURCE_INDEX[sample["setup_source"]])
        self._sample["observer_color"].append(_COLOR_INDEX[sample["observer_color"]])
        self._sample["decision_index"].append(int(sample["decision_index"]))
        self._sample["total_moves"].append(int(sample["total_moves"]))
        self._sample["remaining_counts"].append(list(sample["remaining_counts"]))
        self._sample["target_mask"].append(np.asarray(sample["target_mask"], dtype=bool))

        for piece in sample["pieces"]:
            self._piece["piece_slot"].append(int(piece["piece_slot"]))
            self._piece["piece_square"].append(int(piece["piece_square"]))
            self._piece["perspective_square"].append(int(piece["perspective_square"]))
            self._piece["piece_moved"].append(bool(piece["piece_moved"]))
            self._piece["legal_rank_mask"].append(
                [bool(value) for value in piece["legal_rank_mask"]]
            )
            self._true_rank.append(int(piece["true_rank"]))
        self.pieces += len(sample["pieces"])
        self._sample["piece_offset"].append(self.pieces)
        self.samples += 1

    def close(self) -> dict:
        """Write every remaining array and return the split's manifest."""
        self._observations.close()
        samples = self.samples
        expected = samples * int(np.prod(OBSERVATION_SHAPE)) * 4
        observed = (self.public / "observations.f32").stat().st_size
        if observed != expected:
            raise Phase11BStorageError(
                f"observations.f32 is {observed} bytes, expected {expected}"
            )

        arrays = {
            name: np.asarray(self._sample[name], dtype=dtype)
            for name, dtype in PUBLIC_SAMPLE_ARRAYS.items()
        }
        if samples == 0:  # pragma: no cover - a split always has samples
            raise Phase11BStorageError(f"{self.split} produced no samples")
        for name, array in arrays.items():
            length = samples + 1 if name == "piece_offset" else samples
            if array.shape[0] != length:  # pragma: no cover - accumulation invariant
                raise Phase11BStorageError(f"{name} has {array.shape[0]} rows, expected {length}")
        np.savez(self.public / "samples.npz", **arrays)

        pieces = {
            name: np.asarray(self._piece[name], dtype=dtype)
            for name, dtype in PUBLIC_PIECE_ARRAYS.items()
        }
        for name, array in pieces.items():
            if array.shape[0] != self.pieces:  # pragma: no cover - accumulation invariant
                raise Phase11BStorageError(f"{name} has {array.shape[0]} rows, expected {self.pieces}")
        np.savez(self.public / "pieces.npz", **pieces)

        (self.public / "identities.txt").write_text("\n".join(self._identities) + "\n")
        (self.public / "games.jsonl").write_text(
            "".join(json.dumps(game, sort_keys=True, separators=(",", ":")) + "\n" for game in self._games)
        )
        np.savez(
            self.privileged / "labels.npz",
            true_rank=np.asarray(self._true_rank, dtype=np.int8),
        )
        (self.privileged / "README.txt").write_text(_PRIVILEGED_NOTE)

        specification = CORPUS_SPLITS[self.split]
        games = self._games
        return {
            "split": self.split,
            "corpus_version": CORPUS_VERSION,
            "corpus_format_version": CORPUS_FORMAT_VERSION,
            "games": len(games),
            "games_expected": int(specification["games"]),
            "complete": len(games) == int(specification["games"]),
            "decisions_per_game": int(specification["decisions_per_game"]),
            "library_split": specification["library_split"],
            "samples": samples,
            "hidden_pieces": self.pieces,
            "observer_decisions": sum(game["observer_decisions"] for game in games),
            "eligible_decisions": sum(game["eligible_decisions"] for game in games),
            "zero_sample_games": sum(1 for game in games if game["sampled_decisions"] == 0),
            "observation_shape": list(OBSERVATION_SHAPE),
            "observation_dtype": "float32",
            "strata": {
                name: sum(1 for game in games if game["stratum"] == name)
                for name in CORPUS_STRATA
            },
            "sources": {
                name: sum(1 for game in games if game["setup_source"] == name)
                for name in CORPUS_SOURCES
            },
            "observer_colors": {
                name: sum(1 for game in games if game["observer_color"] == name)
                for name in CORPUS_COLORS
            },
            "outcomes": {
                name: sum(1 for game in games if game["observer_result"] == name)
                for name in ("win", "draw", "loss")
            },
        }


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def load_split(root: "Path | str", split: str, *, labels: bool = False) -> dict:
    """The public half of one stored split; labels only when asked by name.

    Observations are memory-mapped, so a caller that touches a batch at a
    time never pages in the whole 1.3 GB.
    """
    base = split_root(root, split)
    public = base / PUBLIC_DIRECTORY
    with np.load(public / "samples.npz") as handle:
        sample_arrays = {name: handle[name] for name in PUBLIC_SAMPLE_ARRAYS}
    with np.load(public / "pieces.npz") as handle:
        piece_arrays = {name: handle[name] for name in PUBLIC_PIECE_ARRAYS}
    count = int(sample_arrays["game_ordinal"].shape[0])
    observations = np.memmap(
        public / "observations.f32",
        dtype=np.float32,
        mode="r",
        shape=(count, *OBSERVATION_SHAPE),
    )
    identities = (public / "identities.txt").read_text().split("\n")[:-1]
    if len(identities) != count:
        raise Phase11BStorageError(
            f"{split}: {len(identities)} identities for {count} samples"
        )
    games = [
        json.loads(line)
        for line in (public / "games.jsonl").read_text().splitlines()
        if line
    ]
    data = {
        "split": split,
        "root": base,
        "samples": count,
        "pieces": int(piece_arrays["piece_slot"].shape[0]),
        "observations": observations,
        "identities": identities,
        "games": games,
        **sample_arrays,
        **piece_arrays,
    }
    if labels:
        with np.load(base / PRIVILEGED_DIRECTORY / "labels.npz") as handle:
            data["true_rank"] = handle["true_rank"]
        if data["true_rank"].shape[0] != data["pieces"]:
            raise Phase11BStorageError(
                f"{split}: {data['true_rank'].shape[0]} labels for {data['pieces']} pieces"
            )
    return data


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

#: The manifest fields that are logical corpus content. Paths, timestamps
#: and durations are deliberately absent: two runs of the same generator
#: must agree on this digest.
MANIFEST_CONTENT_FIELDS = (
    "complete",
    "corpus_format_version",
    "corpus_version",
    "decisions_per_game",
    "eligible_decisions",
    "games",
    "games_expected",
    "hidden_pieces",
    "library_split",
    "observation_dtype",
    "observation_shape",
    "observer_decisions",
    "samples",
    "split",
    "zero_sample_games",
)


def split_digest(root: "Path | str", split: str) -> dict:
    """Per-file SHA-256 of one stored split, in a frozen order."""
    base = split_root(root, split)
    files = (
        f"{PUBLIC_DIRECTORY}/observations.f32",
        f"{PUBLIC_DIRECTORY}/samples.npz",
        f"{PUBLIC_DIRECTORY}/pieces.npz",
        f"{PUBLIC_DIRECTORY}/identities.txt",
        f"{PUBLIC_DIRECTORY}/games.jsonl",
        f"{PRIVILEGED_DIRECTORY}/labels.npz",
    )
    digests = {}
    for name in files:
        path = base / name
        if not path.exists():
            raise Phase11BStorageError(f"{split}: {name} is missing")
        hasher = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 22), b""):
                hasher.update(block)
        digests[name] = hasher.hexdigest()
    return digests


def corpus_digest(manifest: dict) -> str:
    """A content-only identity of the whole common corpus.

    Hashes the two splits' file digests and their logical manifest fields
    and nothing else, so two independent generations of the corpus produce
    the same string and Agents 2-5 can prove byte-for-byte reuse.
    """
    payload = {
        "corpus_version": manifest["corpus_version"],
        "corpus_format_version": manifest["corpus_format_version"],
        "splits": {
            split: {
                "content": {
                    field: manifest["splits"][split][field]
                    for field in MANIFEST_CONTENT_FIELDS
                },
                "files": manifest["splits"][split]["file_digests"],
            }
            for split in sorted(manifest["splits"])
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_manifest(root: "Path | str", manifest: dict) -> Path:
    path = Path(root) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return path


def read_manifest(root: "Path | str") -> dict:
    path = Path(root) / "manifest.json"
    if not path.exists():
        raise Phase11BStorageError(f"no corpus manifest at {path}")
    return json.loads(path.read_text())


__all__ = [
    "CORPUS_FORMAT_VERSION",
    "MANIFEST_CONTENT_FIELDS",
    "PRIVILEGED_ARRAYS",
    "PUBLIC_PIECE_ARRAYS",
    "PUBLIC_SAMPLE_ARRAYS",
    "Phase11BStorageError",
    "SplitWriter",
    "corpus_digest",
    "load_split",
    "read_manifest",
    "split_digest",
    "split_root",
    "write_manifest",
]
