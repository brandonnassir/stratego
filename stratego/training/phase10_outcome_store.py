"""Phase 10 Agent 2: the crash-safe commit store of the setup-outcome corpus.

Specification sources:

- `02_AGENT_2_SETUP_OUTCOME_CORPUS.md` ("Outcome record", "Crash-safe store")
- `00_PHASE_10_SEQUENCE_AND_COMMON_CONTRACT.md` ("Controlled setup-outcome
  corpus", "Storage/path semantics")
- `stratego/training/corpus_commit.py` — the accepted Phase 8 commit protocol
  this module reproduces in rigor for a different payload

Why a second store rather than the Phase 8 one
----------------------------------------------
Phase 8's store holds `trajectory_v1` game records: tens of thousands of
decisions per game, a binary codec, and a metadata sidecar whose job is to
let the trainer index a corpus it cannot hold in memory. Phase 10 stores
something categorically smaller — one *outcome* per game: who won, in how
many plies, from which two fully described setups. Encoding that through the
trajectory codec would mean inventing a fake decision list, and the Phase 8
metadata validator would have nothing true to say about it.

What is reused is the part that matters: the **commit protocol**. The rule is
one sentence, and it is Phase 8's sentence — *a game becomes visible only
when its commit record exists* — with the same write order, the same
byte-count-carrying journal, and the same truncation-based recovery.

Layout
------
One *file set* belongs to one (segment, worker) and is the only thing that
process appends to::

    <root>/records/seg0000_w00_s0000.stgout    outcome payload frames
    <root>/metadata/seg0000_w00.meta.jsonl     one JSON line per game
    <root>/journal/seg0000_w00.commit.jsonl    one JSON line per commit

Every commit record carries the two file sizes *after* its own writes::

    shard_name  shard_bytes_after  metadata_bytes_after

which is what makes recovery a truncation instead of a rewrite. A shard rolls
over only *between* games, at a committed boundary, so a closed shard never
contains an uncommitted record.

Pre-game and post-game are separated in the bytes
-------------------------------------------------
The instruction is explicit that setup descriptors must stay clearly apart
from outcome fields, so the stored payload is literally two named objects —
`setup` and `outcome` — and the flat 27-field record Agent 1 froze is
*assembled* on read from those two plus the three digests that cannot live
inside the bytes they describe. A record therefore cannot quietly grow an
outcome-shaped field into its setup half.

State
-----
```text
COLLECTING -> SEALED
```

A sealed corpus is immutable: :class:`OutcomeWriter` refuses to open under a
seal, and :func:`reconcile_corpus` refuses to truncate one. The seal records
the content digest of every committed record in canonical game-id order, so
"the same corpus" is a computable claim rather than a filename.

Nothing here resolves a path: :mod:`stratego.training.phase10_storage` owns
that, and identity never contains one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path

from .phase10_schedule import (
    CORPUS_VERSION,
    OUTCOME_RECORD_FIELDS,
    RESULT_TARGETS,
    Phase10ScheduleError,
    parse_phase10_game_id,
)
from .serialization import DEFAULT_COMPRESSION_LEVEL, compress, decompress

#: Version of the commit protocol: the journal fields, the write order and the
#: truncation-based recovery. A change to any of the three is a new version.
OUTCOME_COMMIT_VERSION = "phase10_outcome_commit_v1"

#: Version of one stored record's own layout (`setup` / `outcome` halves).
OUTCOME_RECORD_VERSION = "phase10_outcome_record_v1"

RECORDS_DIRECTORY = "records"
METADATA_DIRECTORY = "metadata"
JOURNAL_DIRECTORY = "journal"

RECORD_SUFFIX = ".stgout"
METADATA_SUFFIX = ".meta.jsonl"
JOURNAL_SUFFIX = ".commit.jsonl"

STATE_FILENAME = "corpus_state.json"
SEAL_FILENAME = "corpus_seal.json"

STATE_COLLECTING = "COLLECTING"
STATE_SEALED = "SEALED"
STATES = (STATE_COLLECTING, STATE_SEALED)

#: Container magic. Distinct from Phase 8's `STGOSHRD` because the frames hold
#: outcome payloads, not `trajectory_v1` records, and a reader must not be able
#: to mistake one for the other.
RECORD_MAGIC = b"STGOOUTC"
RECORD_FORMAT_VERSION = 1

#: Default shard rollover size.
DEFAULT_OUTCOME_SHARD_BYTES = 8 * 1024 * 1024

_HEADER = struct.Struct("<II")
_LENGTH = struct.Struct("<I")

#: Every field of a commit record. A line missing any of them is not a commit.
COMMIT_FIELDS = (
    "commit_version",
    "game_id",
    "file_set",
    "shard_name",
    "record_index",
    "shard_bytes_after",
    "metadata_bytes_after",
    "payload_sha256",
    "metadata_sha256",
    "result",
    "plies",
    "committed_unix",
)

#: The two halves of a stored payload, and the keys each one owns. Both are
#: closed sets: an unexpected key on either side is a rejected record, which
#: is how "clearly separated" is enforced rather than merely intended.
SETUP_SECTION_FIELDS = (
    "corpus_version",
    "record_version",
    "game_id",
    "red_family",
    "blue_family",
    "ordinal",
    "split",
    "match_seed",
    "red_setup_draw_seed",
    "blue_setup_draw_seed",
    "red_setup_attempt",
    "blue_setup_attempt",
    "red_base_setup_id",
    "blue_base_setup_id",
    "red_provenance",
    "blue_provenance",
    "red_final_fingerprint",
    "blue_final_fingerprint",
    "red_trait_identity",
    "blue_trait_identity",
    "trait_schema_version",
    "library_content_digest",
    "corpus_contract_digest",
    "outcome_schedule_digest",
    "contract_bundle_digest",
)

OUTCOME_SECTION_FIELDS = (
    "result",
    "winner",
    "red_score",
    "plies",
    "decisions",
    "terminal_reason",
    "move_policy_identity",
    "move_checkpoint_sha256",
    "move_model_state_digest",
)

#: The three digests a record carries but cannot contain: each names bytes
#: that only exist once the record itself has been written.
DERIVED_RECORD_FIELDS = ("payload_digest", "metadata_digest", "commit_digest")

#: Agent 1's frozen schema, as a set, for the subset assertion below.
FROZEN_RECORD_FIELDS = tuple(name for name, _text in OUTCOME_RECORD_FIELDS)

#: The complete assembled record. Agent 1 froze 27 required fields; Agent 2's
#: instruction requires "at minimum" those, and separately requires a trait
#: identity per side, the final setup fingerprints, a record version and the
#: contract/schedule digests. Those seven are the whole of the difference, and
#: the assertion below is what keeps the frozen 27 a genuine subset rather
#: than a claim.
ASSEMBLED_RECORD_FIELDS = (
    SETUP_SECTION_FIELDS + OUTCOME_SECTION_FIELDS + DERIVED_RECORD_FIELDS
)
assert set(FROZEN_RECORD_FIELDS) <= set(ASSEMBLED_RECORD_FIELDS), sorted(
    set(FROZEN_RECORD_FIELDS) - set(ASSEMBLED_RECORD_FIELDS)
)

#: Fields present in a stored record beyond Agent 1's frozen 27.
ADDITIONAL_RECORD_FIELDS = tuple(
    sorted(set(ASSEMBLED_RECORD_FIELDS) - set(FROZEN_RECORD_FIELDS))
)

_FILE_SET_PATTERN = re.compile(r"^seg(?P<segment>\d{4})_w(?P<worker>\d{2})$")
_SHARD_PATTERN = re.compile(
    r"^seg(?P<segment>\d{4})_w(?P<worker>\d{2})_s(?P<shard>\d{4})$"
)


class OutcomeStoreError(RuntimeError):
    """The outcome store could not be written, read back or reconciled."""


# ---------------------------------------------------------------------------
# Canonical bytes and digests
# ---------------------------------------------------------------------------


def canonical_json(payload: dict) -> str:
    """The one canonical text form used for every digest in this module."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def encode_payload(record: dict, level: int = DEFAULT_COMPRESSION_LEVEL) -> "tuple[bytes, bytes]":
    """`(raw, compressed)` bytes of one outcome record's stored payload."""
    raw = canonical_json(record).encode()
    return raw, compress(raw, level)


def decode_payload(payload: bytes) -> dict:
    """The stored record inside one payload frame."""
    return json.loads(decompress(payload).decode())


def payload_digest(payload: bytes) -> str:
    """SHA-256 of one stored payload, exactly as written."""
    return hashlib.sha256(payload).hexdigest()


def metadata_line(metadata: dict) -> str:
    """The canonical one-line JSON form of a metadata record."""
    return canonical_json(metadata) + "\n"


def metadata_digest(metadata: dict) -> str:
    """SHA-256 over the canonical metadata line, without the newline."""
    return hashlib.sha256(metadata_line(metadata)[:-1].encode()).hexdigest()


def commit_digest(commit: dict) -> str:
    """SHA-256 over the canonical commit line, without the newline."""
    return hashlib.sha256(canonical_json(commit).encode()).hexdigest()


def file_set_name(segment: int, worker_id: int) -> str:
    return f"seg{int(segment):04d}_w{int(worker_id):02d}"


def shard_name(segment: int, worker_id: int, shard_index: int) -> str:
    return f"{file_set_name(segment, worker_id)}_s{int(shard_index):04d}"


def records_directory(root: "str | Path") -> Path:
    return Path(root) / RECORDS_DIRECTORY


def metadata_directory(root: "str | Path") -> Path:
    return Path(root) / METADATA_DIRECTORY


def journal_directory(root: "str | Path") -> Path:
    return Path(root) / JOURNAL_DIRECTORY


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


# ---------------------------------------------------------------------------
# Store state
# ---------------------------------------------------------------------------


def state_path(root: "str | Path") -> Path:
    return Path(root) / STATE_FILENAME


def seal_path(root: "str | Path") -> Path:
    return Path(root) / SEAL_FILENAME


def read_state(root: "str | Path") -> str:
    """`COLLECTING` for a store that does not exist yet; whatever it says once it does."""
    path = state_path(root)
    if not path.exists():
        return STATE_COLLECTING
    payload = json.loads(path.read_text())
    state = str(payload.get("state"))
    if state not in STATES:
        raise OutcomeStoreError(f"{path}: unknown corpus state {state!r}")
    return state


def write_state(root: "str | Path", state: str, **extra) -> dict:
    if state not in STATES:
        raise OutcomeStoreError(f"unknown corpus state {state!r}")
    payload = {
        "corpus_version": CORPUS_VERSION,
        "commit_version": OUTCOME_COMMIT_VERSION,
        "state": state,
        "updated_unix": time.time(),
        **extra,
    }
    path = state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def require_collecting(root: "str | Path") -> None:
    """Refuse any mutation of a sealed corpus. A seal is not advisory."""
    if read_state(root) == STATE_SEALED:
        raise OutcomeStoreError(
            f"{root} is SEALED; a sealed corpus is immutable and must not be "
            "appended to, truncated or re-collected"
        )


# ---------------------------------------------------------------------------
# The append-only journal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommitRecord:
    """One committed game, as the journal stores it."""

    game_id: str
    file_set: str
    shard_name: str
    record_index: int
    shard_bytes_after: int
    metadata_bytes_after: int
    payload_sha256: str
    metadata_sha256: str
    result: str
    plies: int
    committed_unix: float

    def to_dict(self) -> dict:
        return {
            "commit_version": OUTCOME_COMMIT_VERSION,
            "game_id": self.game_id,
            "file_set": self.file_set,
            "shard_name": self.shard_name,
            "record_index": self.record_index,
            "shard_bytes_after": self.shard_bytes_after,
            "metadata_bytes_after": self.metadata_bytes_after,
            "payload_sha256": self.payload_sha256,
            "metadata_sha256": self.metadata_sha256,
            "result": self.result,
            "plies": self.plies,
            "committed_unix": self.committed_unix,
        }

    @property
    def digest(self) -> str:
        """The commit identity: a digest over the committed line's own bytes.

        `committed_unix` is inside it deliberately. Two runs that collect the
        same game produce the same *payload* digest and different *commit*
        digests, which is exactly the distinction the balance audit needs to
        tell "the same game" from "the same commit".
        """
        return commit_digest(self.to_dict())

    @staticmethod
    def from_dict(payload: dict) -> "CommitRecord":
        return CommitRecord(
            game_id=str(payload["game_id"]),
            file_set=str(payload["file_set"]),
            shard_name=str(payload["shard_name"]),
            record_index=int(payload["record_index"]),
            shard_bytes_after=int(payload["shard_bytes_after"]),
            metadata_bytes_after=int(payload["metadata_bytes_after"]),
            payload_sha256=str(payload["payload_sha256"]),
            metadata_sha256=str(payload["metadata_sha256"]),
            result=str(payload["result"]),
            plies=int(payload["plies"]),
            committed_unix=float(payload["committed_unix"]),
        )


def read_journal(path: "str | Path") -> tuple:
    """`(commits, valid_bytes)` for one journal file.

    Only newline-terminated, fully parseable lines carrying every commit field
    are accepted, so a process killed mid-line contributes nothing.
    `valid_bytes` is where the journal would be truncated to remove the torn
    tail.
    """
    path = Path(path)
    if not path.exists():
        return ([], 0)
    raw = path.read_bytes()
    commits: list[CommitRecord] = []
    offset = 0
    valid = 0
    while True:
        index = raw.find(b"\n", offset)
        if index < 0:
            # A tail without a newline is an interrupted write, by definition.
            break
        line = raw[offset:index]
        offset = index + 1
        text = line.strip()
        if not text:
            valid = offset
            continue
        try:
            payload = json.loads(text.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            break
        if not isinstance(payload, dict) or any(key not in payload for key in COMMIT_FIELDS):
            break
        if payload["commit_version"] != OUTCOME_COMMIT_VERSION:
            raise OutcomeStoreError(
                f"{path}: commit protocol {payload['commit_version']!r} is not "
                f"{OUTCOME_COMMIT_VERSION!r}"
            )
        commits.append(CommitRecord.from_dict(payload))
        valid = offset
    return (commits, valid)


def read_metadata_file(path: "str | Path") -> tuple:
    """`(records, valid_bytes)` for one metadata sidecar, same tail rule."""
    path = Path(path)
    if not path.exists():
        return ([], 0)
    raw = path.read_bytes()
    records: list[dict] = []
    offset = 0
    valid = 0
    while True:
        index = raw.find(b"\n", offset)
        if index < 0:
            break
        line = raw[offset:index]
        offset = index + 1
        text = line.strip()
        if not text:
            valid = offset
            continue
        try:
            payload = json.loads(text.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            break
        if not isinstance(payload, dict) or "game_id" not in payload:
            break
        records.append(payload)
        valid = offset
    return (records, valid)


# ---------------------------------------------------------------------------
# Building one record
# ---------------------------------------------------------------------------


def build_stored_record(setup_section: dict, outcome_section: dict) -> dict:
    """The two-halved payload of one game, validated field by field.

    Both halves are closed sets, so an outcome field cannot drift into the
    setup half (or the reverse) without this raising. That is the mechanical
    form of "keep pre-game setup descriptors clearly separated from post-game
    outcome fields".
    """
    for label, section, expected in (
        ("setup", setup_section, SETUP_SECTION_FIELDS),
        ("outcome", outcome_section, OUTCOME_SECTION_FIELDS),
    ):
        missing = [name for name in expected if name not in section]
        extra = [name for name in section if name not in expected]
        if missing or extra:
            raise OutcomeStoreError(
                f"{label} section is malformed: missing={missing} unexpected={extra}"
            )
    if setup_section["corpus_version"] != CORPUS_VERSION:
        raise OutcomeStoreError(
            f"record names corpus {setup_section['corpus_version']!r}, not {CORPUS_VERSION!r}"
        )
    if setup_section["record_version"] != OUTCOME_RECORD_VERSION:
        raise OutcomeStoreError(
            f"record names layout {setup_section['record_version']!r}, not "
            f"{OUTCOME_RECORD_VERSION!r}"
        )
    result = outcome_section["result"]
    if result not in RESULT_TARGETS:
        raise OutcomeStoreError(f"unknown result token {result!r}")
    if float(outcome_section["red_score"]) != RESULT_TARGETS[result]:
        raise OutcomeStoreError(
            f"result {result!r} carries red_score {outcome_section['red_score']!r}, "
            f"not the frozen target {RESULT_TARGETS[result]}"
        )
    try:
        parse_phase10_game_id(setup_section["game_id"])
    except Phase10ScheduleError as error:  # pragma: no cover - defensive
        raise OutcomeStoreError(str(error)) from error
    return {"setup": dict(setup_section), "outcome": dict(outcome_section)}


def build_metadata(record: dict, *, payload_sha256: str) -> dict:
    """The compact index line: enough to audit without decompressing anything.

    Deliberately not a copy of the record. It carries identity, the balance
    audit's dimensions and the payload digest that binds it to the bytes.
    """
    setup = record["setup"]
    outcome = record["outcome"]
    return {
        "corpus_version": setup["corpus_version"],
        "record_version": setup["record_version"],
        "game_id": setup["game_id"],
        "red_family": setup["red_family"],
        "blue_family": setup["blue_family"],
        "ordinal": setup["ordinal"],
        "split": setup["split"],
        "red_base_setup_id": setup["red_base_setup_id"],
        "blue_base_setup_id": setup["blue_base_setup_id"],
        "red_final_fingerprint": setup["red_final_fingerprint"],
        "blue_final_fingerprint": setup["blue_final_fingerprint"],
        "result": outcome["result"],
        "red_score": outcome["red_score"],
        "plies": outcome["plies"],
        "terminal_reason": outcome["terminal_reason"],
        "move_policy_identity": outcome["move_policy_identity"],
        "move_model_state_digest": outcome["move_model_state_digest"],
        "payload_sha256": payload_sha256,
    }


def assemble_record(record: dict, metadata: dict, commit: CommitRecord) -> dict:
    """The flat record Agent 3 reads: both halves plus the three digests."""
    assembled = dict(record["setup"])
    assembled.update(record["outcome"])
    assembled["payload_digest"] = commit.payload_sha256
    assembled["metadata_digest"] = metadata_digest(metadata)
    assembled["commit_digest"] = commit.digest
    return assembled


# ---------------------------------------------------------------------------
# The writer
# ---------------------------------------------------------------------------


class OutcomeWriter:
    """The append-only writer of one (segment, worker) file set.

    Not shared and not thread-safe: one instance belongs to one process, which
    is the only thing that ever appends to these three files.
    """

    def __init__(
        self,
        root: "str | Path",
        *,
        segment: int,
        worker_id: int,
        target_bytes: int = DEFAULT_OUTCOME_SHARD_BYTES,
        compression_level: int = DEFAULT_COMPRESSION_LEVEL,
        fsync_on_commit: bool = False,
        crash_hook=None,
    ) -> None:
        self.root = Path(root)
        self.segment = int(segment)
        self.worker_id = int(worker_id)
        self.target_bytes = int(target_bytes)
        self.compression_level = int(compression_level)
        self.fsync_on_commit = bool(fsync_on_commit)
        self.crash_hook = crash_hook

        require_collecting(self.root)

        self.name = file_set_name(self.segment, self.worker_id)
        self.records_directory = records_directory(self.root)
        self.metadata_path = metadata_directory(self.root) / f"{self.name}{METADATA_SUFFIX}"
        self.journal_path = journal_directory(self.root) / f"{self.name}{JOURNAL_SUFFIX}"
        for directory in (
            self.records_directory,
            self.metadata_path.parent,
            self.journal_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        if self.metadata_path.exists() or self.journal_path.exists():
            raise OutcomeStoreError(
                f"file set {self.name} already exists; a resumed run must open a "
                "fresh segment rather than append to a reconciled one"
            )

        self.commits: list[CommitRecord] = []
        self.games_written = 0
        self.uncompressed_bytes = 0
        self.compressed_bytes = 0
        self.write_seconds = 0.0
        self.verify_seconds = 0.0

        self._shard_index = -1
        self._shard_path: Path | None = None
        self._shard_handle = None
        self._shard_records = 0
        self._metadata_handle = open(self.metadata_path, "ab")
        self._journal_handle = open(self.journal_path, "ab")
        self._open_shard()

    # -- lifecycle ---------------------------------------------------------

    def _hook(self, stage: str) -> None:
        if self.crash_hook is not None:
            self.crash_hook(stage, self)

    def _open_shard(self) -> None:
        self._shard_index += 1
        name = shard_name(self.segment, self.worker_id, self._shard_index)
        self._shard_path = self.records_directory / f"{name}{RECORD_SUFFIX}"
        header = {
            "corpus_version": CORPUS_VERSION,
            "record_version": OUTCOME_RECORD_VERSION,
            "commit_version": OUTCOME_COMMIT_VERSION,
            "file_set": self.name,
            "worker_id": self.worker_id,
            "shard_index": self._shard_index,
            "compressed": True,
            "compression": "zlib",
            "compression_level": self.compression_level,
            "opened_unix": time.time(),
        }
        blob = json.dumps(header, sort_keys=True).encode()
        self._shard_handle = self._shard_path.open("wb")
        preamble = RECORD_MAGIC + _HEADER.pack(RECORD_FORMAT_VERSION, len(blob)) + blob
        self._shard_handle.write(preamble)
        self._shard_handle.flush()
        self._shard_records = 0

    def _roll_shard_if_needed(self) -> None:
        """Roll over between games only, so a shard closes on a committed boundary."""
        if self._shard_path is None or self._shard_handle is None:
            return
        if self._shard_handle.tell() < self.target_bytes:
            return
        self._hook("shard_rollover")
        self._close_shard()
        self._open_shard()

    def _close_shard(self) -> None:
        if self._shard_handle is None:
            return
        self._shard_handle.flush()
        os.fsync(self._shard_handle.fileno())
        self._shard_handle.close()
        self._shard_handle = None

    @property
    def current_shard_name(self) -> str:
        return shard_name(self.segment, self.worker_id, self._shard_index)

    # -- the commit protocol ------------------------------------------------

    def write_record(self, record: dict) -> CommitRecord:
        """Persist and commit one outcome record, in the frozen order.

        ```text
        1. encode + compress + decode-verify the payload
        2. build and check the metadata line against the record
        3. append the payload frame, flush
        4. append the metadata line, flush
        5. append the commit line, flush
        ```

        A failure or interruption anywhere before step 5 leaves the game
        uncommitted and therefore invisible; :func:`reconcile_corpus` removes
        the partial bytes on the next open.
        """
        setup = record.get("setup")
        outcome = record.get("outcome")
        if set(record) != {"setup", "outcome"} or not isinstance(setup, dict):
            raise OutcomeStoreError(
                "a stored record is exactly a 'setup' half and an 'outcome' half"
            )
        # Re-validates both closed sections even when the caller already built
        # the record through `build_stored_record`: the writer is the last place
        # a malformed record can still be refused.
        record = build_stored_record(setup, outcome)

        raw, payload = encode_payload(record, self.compression_level)
        digest = payload_digest(payload)
        metadata = build_metadata(record, payload_sha256=digest)

        started = time.perf_counter()
        # "Verifies" means the bytes decode back to this record, not that they
        # are the right length. Doing it before the write is what lets the
        # commit rule promise that a visible record is a readable one.
        if decode_payload(payload) != record:
            raise OutcomeStoreError(
                f"game {setup['game_id']} failed pre-commit verification: the stored "
                "payload does not decode back to the record it came from"
            )
        if metadata["game_id"] != setup["game_id"]:
            raise OutcomeStoreError(
                f"metadata names {metadata['game_id']!r}, record names {setup['game_id']!r}"
            )
        self.verify_seconds += time.perf_counter() - started

        self._roll_shard_if_needed()
        self._hook("before_payload")

        line = metadata_line(metadata)
        write_started = time.perf_counter()
        frame = _LENGTH.pack(len(payload)) + payload
        self._shard_handle.write(frame)
        self._shard_handle.flush()
        shard_bytes_after = self._shard_handle.tell()
        record_index = self._shard_records
        self._shard_records += 1
        self._hook("after_payload")

        self._metadata_handle.write(line.encode())
        self._metadata_handle.flush()
        metadata_bytes_after = self._metadata_handle.tell()
        self._hook("after_metadata")

        commit = CommitRecord(
            game_id=setup["game_id"],
            file_set=self.name,
            shard_name=self.current_shard_name,
            record_index=record_index,
            shard_bytes_after=shard_bytes_after,
            metadata_bytes_after=metadata_bytes_after,
            payload_sha256=digest,
            metadata_sha256=metadata_digest(metadata),
            result=outcome["result"],
            plies=int(outcome["plies"]),
            committed_unix=time.time(),
        )
        self._hook("before_commit_flush")
        self._journal_handle.write((canonical_json(commit.to_dict()) + "\n").encode())
        self._journal_handle.flush()
        if self.fsync_on_commit:
            os.fsync(self._shard_handle.fileno())
            os.fsync(self._metadata_handle.fileno())
            os.fsync(self._journal_handle.fileno())
        self.write_seconds += time.perf_counter() - write_started
        self._hook("after_commit")

        self.commits.append(commit)
        self.games_written += 1
        self.uncompressed_bytes += len(raw)
        self.compressed_bytes += len(payload)
        return commit

    def close(self) -> dict:
        """Flush and fsync everything this writer owns. Idempotent."""
        self._close_shard()
        for handle in (self._metadata_handle, self._journal_handle):
            if handle is None or handle.closed:
                continue
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        self._metadata_handle = None
        self._journal_handle = None
        return self.stats()

    def stats(self) -> dict:
        return {
            "file_set": self.name,
            "segment": self.segment,
            "worker_id": self.worker_id,
            "games_written": self.games_written,
            "shards_opened": self._shard_index + 1,
            "uncompressed_bytes": self.uncompressed_bytes,
            "compressed_bytes": self.compressed_bytes,
            "compression_ratio": (
                self.compressed_bytes / self.uncompressed_bytes
                if self.uncompressed_bytes
                else 0.0
            ),
            "verify_seconds": self.verify_seconds,
            "write_seconds": self.write_seconds,
        }


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def _file_sets(root: "str | Path") -> list:
    """Every (segment, worker) file set that exists."""
    names = set()
    for path in journal_directory(root).glob(f"*{JOURNAL_SUFFIX}"):
        names.add(path.name[: -len(JOURNAL_SUFFIX)])
    for path in metadata_directory(root).glob(f"*{METADATA_SUFFIX}"):
        names.add(path.name[: -len(METADATA_SUFFIX)])
    for path in records_directory(root).glob(f"*{RECORD_SUFFIX}"):
        match = _SHARD_PATTERN.match(path.stem)
        if match is not None:
            names.add(file_set_name(int(match["segment"]), int(match["worker"])))
    resolved = []
    for name in sorted(names):
        match = _FILE_SET_PATTERN.match(name)
        if match is None:
            raise OutcomeStoreError(f"unrecognized outcome file set name: {name!r}")
        resolved.append((int(match["segment"]), int(match["worker"]), name))
    return resolved


def _truncate(path: Path, size: int) -> int:
    """Cut a file back to `size`, returning the bytes removed."""
    current = _file_size(path)
    if current <= size:
        return 0
    with path.open("r+b") as handle:
        handle.truncate(size)
        handle.flush()
        os.fsync(handle.fileno())
    return current - size


def reconcile_file_set(root: "str | Path", name: str) -> dict:
    """Cut one file set back to its last commit and report what was discarded.

    This is the only function that ever removes corpus bytes. It removes
    exactly the bytes no commit record claims, which by the write order can
    only be work that was interrupted before it became visible.
    """
    root = Path(root)
    journal_path = journal_directory(root) / f"{name}{JOURNAL_SUFFIX}"
    metadata_path = metadata_directory(root) / f"{name}{METADATA_SUFFIX}"
    shard_directory = records_directory(root)

    commits, journal_valid_bytes = read_journal(journal_path)
    report = {
        "file_set": name,
        "committed_games": len(commits),
        "journal_bytes_discarded": 0,
        "metadata_bytes_discarded": 0,
        "shard_bytes_discarded": 0,
        "shards_removed": [],
        "torn_journal_tail": False,
    }

    if journal_path.exists():
        removed = _truncate(journal_path, journal_valid_bytes)
        report["journal_bytes_discarded"] = removed
        report["torn_journal_tail"] = removed > 0

    last = commits[-1] if commits else None
    metadata_keep = last.metadata_bytes_after if last is not None else 0
    if metadata_path.exists():
        report["metadata_bytes_discarded"] = _truncate(metadata_path, metadata_keep)

    match = _FILE_SET_PATTERN.match(name)
    if match is None:  # pragma: no cover - callers pass validated names
        raise OutcomeStoreError(f"unrecognized outcome file set name: {name!r}")
    segment, worker = int(match["segment"]), int(match["worker"])
    last_shard_index = -1
    if last is not None:
        shard_match = _SHARD_PATTERN.match(last.shard_name)
        if shard_match is None:
            raise OutcomeStoreError(
                f"commit for {last.game_id} names shard {last.shard_name!r}, which is "
                "not an outcome shard name"
            )
        last_shard_index = int(shard_match["shard"])
        shard_path = shard_directory / f"{last.shard_name}{RECORD_SUFFIX}"
        if not shard_path.exists():
            raise OutcomeStoreError(
                f"committed game {last.game_id} names missing shard {shard_path}"
            )
        report["shard_bytes_discarded"] += _truncate(shard_path, last.shard_bytes_after)

    for path in sorted(shard_directory.glob(f"{name}_s*{RECORD_SUFFIX}")):
        shard_match = _SHARD_PATTERN.match(path.stem)
        if shard_match is None:  # pragma: no cover - glob already constrains it
            continue
        if (int(shard_match["segment"]), int(shard_match["worker"])) != (segment, worker):
            continue  # pragma: no cover - glob already constrains it
        if int(shard_match["shard"]) <= last_shard_index:
            continue
        # Every record in this shard was written after the last commit, so the
        # whole file is uncommitted work.
        report["shard_bytes_discarded"] += _file_size(path)
        report["shards_removed"].append(path.name)
        path.unlink()

    return report


def reconcile_corpus(root: "str | Path") -> dict:
    """Reconcile every file set before collection resumes.

    Returns the committed index plus a per-file-set account of what was
    discarded. After this call the store contains committed games and nothing
    else, which is the precondition every later audit assumes.
    """
    root = Path(root)
    require_collecting(root)
    reports = []
    committed: dict[str, CommitRecord] = {}
    duplicates: list[str] = []
    for _segment, _worker, name in _file_sets(root):
        reports.append(reconcile_file_set(root, name))
        commits, _ = read_journal(journal_directory(root) / f"{name}{JOURNAL_SUFFIX}")
        for commit in commits:
            if commit.game_id in committed:
                duplicates.append(commit.game_id)
            committed[commit.game_id] = commit
    return {
        "commit_version": OUTCOME_COMMIT_VERSION,
        "committed": committed,
        "committed_count": len(committed),
        "duplicate_committed_ids": sorted(set(duplicates)),
        "file_sets": reports,
        "shards_removed": [name for report in reports for name in report["shards_removed"]],
        "bytes_discarded": sum(
            report["journal_bytes_discarded"]
            + report["metadata_bytes_discarded"]
            + report["shard_bytes_discarded"]
            for report in reports
        ),
    }


def next_segment(root: "str | Path") -> int:
    """The first segment number no file set has used.

    A resumed run always writes a fresh segment. Appending to a reconciled file
    set would work, but a segment boundary makes "these bytes were written by
    that attempt" readable straight off the filenames.
    """
    highest = -1
    for segment, _worker, _name in _file_sets(root):
        highest = max(highest, segment)
    return highest + 1


# ---------------------------------------------------------------------------
# Reading a committed corpus
# ---------------------------------------------------------------------------


def _shard_frame_offsets(path: Path) -> list:
    """Byte offset and length of every frame in one shard, in written order.

    Walks the length prefixes by seeking rather than reading the payloads, so
    the cost of indexing a shard is its frame count and not its size. That
    matters because a full-corpus audit indexes every shard once and then
    reads 16,384 individual payloads out of them.
    """
    size = path.stat().st_size
    frames = []
    with path.open("rb") as handle:
        magic = handle.read(len(RECORD_MAGIC))
        if magic != RECORD_MAGIC:
            raise OutcomeStoreError(f"{path}: not an outcome shard")
        version, header_bytes = _HEADER.unpack(handle.read(_HEADER.size))
        if version != RECORD_FORMAT_VERSION:
            raise OutcomeStoreError(f"{path}: shard format version {version}")
        cursor = len(RECORD_MAGIC) + _HEADER.size + header_bytes
        handle.seek(cursor)
        while cursor < size:
            prefix = handle.read(_LENGTH.size)
            if len(prefix) != _LENGTH.size:
                raise OutcomeStoreError(f"{path}: truncated frame length prefix")
            (length,) = _LENGTH.unpack(prefix)
            cursor += _LENGTH.size
            frames.append((cursor, length))
            cursor += length
            handle.seek(cursor)
    if cursor != size:
        raise OutcomeStoreError(f"{path}: trailing bytes after the last frame")
    return frames


class OutcomeReader:
    """Read-only access to a committed corpus, in canonical game-id order.

    The canonical order is `sorted(game_id)` and nothing else: it does not
    depend on which worker wrote a record, in which segment, or in what
    arrival order, which is what makes two differently partitioned runs of the
    same schedule the same corpus.
    """

    def __init__(self, root: "str | Path") -> None:
        self.root = Path(root)
        self.state = read_state(self.root)
        self._commits: dict[str, CommitRecord] = {}
        self._metadata: dict[str, dict] = {}
        self._frame_cache: dict[str, list] = {}
        duplicates: list[str] = []
        for _segment, _worker, name in _file_sets(self.root):
            commits, _ = read_journal(journal_directory(self.root) / f"{name}{JOURNAL_SUFFIX}")
            for commit in commits:
                if commit.game_id in self._commits:
                    duplicates.append(commit.game_id)
                self._commits[commit.game_id] = commit
            records, _ = read_metadata_file(
                metadata_directory(self.root) / f"{name}{METADATA_SUFFIX}"
            )
            for entry in records:
                self._metadata[str(entry["game_id"])] = entry
        self.duplicate_committed_ids = sorted(set(duplicates))
        self.game_ids = tuple(sorted(self._commits))

    def __len__(self) -> int:
        return len(self._commits)

    def commit(self, game_id: str) -> CommitRecord:
        try:
            return self._commits[game_id]
        except KeyError as error:
            raise OutcomeStoreError(f"{game_id} is not committed in {self.root}") from error

    def metadata(self, game_id: str) -> dict:
        commit = self.commit(game_id)
        entry = self._metadata.get(game_id)
        if entry is None:
            raise OutcomeStoreError(f"{game_id} is committed but has no metadata line")
        if metadata_digest(entry) != commit.metadata_sha256:
            raise OutcomeStoreError(
                f"{game_id}: metadata digest disagrees with its commit record"
            )
        return entry

    def _frames(self, shard: str) -> list:
        """The frame table of one shard, indexed once and then remembered."""
        table = self._frame_cache.get(shard)
        if table is None:
            table = _shard_frame_offsets(records_directory(self.root) / f"{shard}{RECORD_SUFFIX}")
            self._frame_cache[shard] = table
        return table

    def payload(self, game_id: str) -> bytes:
        commit = self.commit(game_id)
        path = records_directory(self.root) / f"{commit.shard_name}{RECORD_SUFFIX}"
        frames = self._frames(commit.shard_name)
        if commit.record_index >= len(frames):
            raise OutcomeStoreError(
                f"{game_id}: commit names frame {commit.record_index} of "
                f"{commit.shard_name}, which holds {len(frames)}"
            )
        offset, length = frames[commit.record_index]
        with path.open("rb") as handle:
            handle.seek(offset)
            payload = handle.read(length)
        if len(payload) != length:
            raise OutcomeStoreError(f"{game_id}: stored frame is shorter than its length prefix")
        if payload_digest(payload) != commit.payload_sha256:
            raise OutcomeStoreError(f"{game_id}: stored payload digest disagrees with its commit")
        return payload

    def stored(self, game_id: str) -> dict:
        """The two-halved stored record."""
        return decode_payload(self.payload(game_id))

    def record(self, game_id: str) -> dict:
        """The flat assembled record: both halves plus the three digests."""
        return assemble_record(self.stored(game_id), self.metadata(game_id), self.commit(game_id))

    def iter_records(self):
        for game_id in self.game_ids:
            yield self.record(game_id)

    def shard_paths(self) -> list:
        return sorted(records_directory(self.root).glob(f"*{RECORD_SUFFIX}"))


# ---------------------------------------------------------------------------
# Integrity, content identity and sealing
# ---------------------------------------------------------------------------


def corpus_content_digest(root: "str | Path") -> str:
    """SHA-256 over every committed payload digest, in canonical game-id order.

    Path-independent by construction: the same records copied to another
    volume, or rewritten by a differently partitioned run, produce the same
    value.
    """
    reader = OutcomeReader(root)
    digest = hashlib.sha256()
    digest.update(f"{CORPUS_VERSION}|{OUTCOME_RECORD_VERSION}|{len(reader)}".encode())
    for game_id in reader.game_ids:
        digest.update(f"|{game_id}|{reader.commit(game_id).payload_sha256}".encode())
    return digest.hexdigest()


def audit_store_integrity(root: "str | Path") -> dict:
    """Every committed record read back and checked against its own digests."""
    root = Path(root)
    reader = OutcomeReader(root)
    payload_mismatches: list[str] = []
    metadata_mismatches: list[str] = []
    orphan_metadata: list[str] = []
    section_violations: list[str] = []
    field_violations: list[str] = []

    for game_id in reader.game_ids:
        commit = reader.commit(game_id)
        try:
            stored = reader.stored(game_id)
        except OutcomeStoreError as error:
            payload_mismatches.append(f"{game_id}: {error}")
            continue
        try:
            metadata = reader.metadata(game_id)
        except OutcomeStoreError as error:
            metadata_mismatches.append(f"{game_id}: {error}")
            continue
        if set(stored) != {"setup", "outcome"}:
            section_violations.append(f"{game_id}: stored halves are {sorted(stored)}")
            continue
        if set(stored["setup"]) != set(SETUP_SECTION_FIELDS):
            section_violations.append(f"{game_id}: setup half has the wrong fields")
        if set(stored["outcome"]) != set(OUTCOME_SECTION_FIELDS):
            section_violations.append(f"{game_id}: outcome half has the wrong fields")
        assembled = assemble_record(stored, metadata, commit)
        if set(assembled) != set(ASSEMBLED_RECORD_FIELDS):
            field_violations.append(f"{game_id}: assembled record has the wrong fields")
        if not set(FROZEN_RECORD_FIELDS) <= set(assembled):
            field_violations.append(f"{game_id}: assembled record misses a frozen field")

    committed = set(reader.game_ids)
    for game_id in reader._metadata:  # noqa: SLF001 - the reader is this module's own
        if game_id not in committed:
            orphan_metadata.append(game_id)

    checks = {
        "payload_digests_match": not payload_mismatches,
        "metadata_digests_match": not metadata_mismatches,
        "no_orphan_metadata": not orphan_metadata,
        "sections_well_formed": not section_violations,
        "assembled_fields_exact": not field_violations,
        "no_duplicate_commit_ids": not reader.duplicate_committed_ids,
    }
    return {
        "commit_version": OUTCOME_COMMIT_VERSION,
        "record_version": OUTCOME_RECORD_VERSION,
        "committed_games": len(reader),
        "state": reader.state,
        "payload_mismatches": payload_mismatches[:32],
        "metadata_mismatches": metadata_mismatches[:32],
        "orphan_metadata": sorted(orphan_metadata)[:32],
        "section_violations": section_violations[:32],
        "field_violations": field_violations[:32],
        "duplicate_committed_ids": reader.duplicate_committed_ids[:32],
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def storage_summary(root: "str | Path") -> dict:
    """Byte-level diagnostics. Paths are diagnostic; identity never is."""
    root = Path(root)
    reader = OutcomeReader(root)
    shard_bytes = sum(_file_size(path) for path in reader.shard_paths())
    metadata_bytes = sum(
        _file_size(path) for path in metadata_directory(root).glob(f"*{METADATA_SUFFIX}")
    )
    journal_bytes = sum(
        _file_size(path) for path in journal_directory(root).glob(f"*{JOURNAL_SUFFIX}")
    )
    uncompressed = 0
    compressed = 0
    for game_id in reader.game_ids:
        payload = reader.payload(game_id)
        compressed += len(payload)
        uncompressed += len(decompress(payload))
    total = shard_bytes + metadata_bytes + journal_bytes
    games = max(len(reader), 1)
    return {
        "committed_games": len(reader),
        "shard_count": len(reader.shard_paths()),
        "record_bytes": shard_bytes,
        "metadata_bytes": metadata_bytes,
        "journal_bytes": journal_bytes,
        "total_bytes": total,
        "bytes_per_game": total / games,
        "payload_uncompressed_bytes": uncompressed,
        "payload_compressed_bytes": compressed,
        "compression_ratio": compressed / uncompressed if uncompressed else 0.0,
    }


def seal_corpus(root: "str | Path", *, expected_games: int, extra: "dict | None" = None) -> dict:
    """Move the store `COLLECTING -> SEALED` and freeze its content identity.

    Sealing is refused unless the corpus is exactly the expected size, carries
    no duplicate commit, and reads back clean — a seal is a claim about the
    bytes, so it is computed from them rather than asserted over them.
    """
    root = Path(root)
    state = read_state(root)
    if state == STATE_SEALED:
        raise OutcomeStoreError(f"{root} is already SEALED")
    integrity = audit_store_integrity(root)
    if not integrity["all_pass"]:
        raise OutcomeStoreError(f"refusing to seal {root}: integrity audit failed")
    if integrity["committed_games"] != int(expected_games):
        raise OutcomeStoreError(
            f"refusing to seal {root}: {integrity['committed_games']} committed games, "
            f"expected {expected_games}"
        )
    digest = corpus_content_digest(root)
    seal = {
        "corpus_version": CORPUS_VERSION,
        "record_version": OUTCOME_RECORD_VERSION,
        "commit_version": OUTCOME_COMMIT_VERSION,
        "committed_games": integrity["committed_games"],
        "content_digest": digest,
        "sealed_unix": time.time(),
        **(extra or {}),
    }
    seal_path(root).write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    write_state(root, STATE_SEALED, content_digest=digest, committed_games=integrity["committed_games"])
    return seal


def read_seal(root: "str | Path") -> dict:
    path = seal_path(root)
    if not path.exists():
        raise OutcomeStoreError(f"{root} carries no seal")
    return json.loads(path.read_text())


def verify_seal(root: "str | Path") -> dict:
    """Recompute a sealed corpus's content digest and compare it to its seal."""
    seal = read_seal(root)
    observed = corpus_content_digest(root)
    reader = OutcomeReader(root)
    checks = {
        "state_is_sealed": read_state(root) == STATE_SEALED,
        "content_digest_matches": observed == seal["content_digest"],
        "committed_games_match": len(reader) == int(seal["committed_games"]),
        "corpus_version_matches": seal["corpus_version"] == CORPUS_VERSION,
    }
    return {
        "seal": seal,
        "observed_content_digest": observed,
        "observed_committed_games": len(reader),
        "checks": checks,
        "all_pass": all(checks.values()),
    }


__all__ = [
    "ADDITIONAL_RECORD_FIELDS",
    "ASSEMBLED_RECORD_FIELDS",
    "COMMIT_FIELDS",
    "DEFAULT_OUTCOME_SHARD_BYTES",
    "FROZEN_RECORD_FIELDS",
    "OUTCOME_COMMIT_VERSION",
    "OUTCOME_RECORD_VERSION",
    "OUTCOME_SECTION_FIELDS",
    "SETUP_SECTION_FIELDS",
    "STATE_COLLECTING",
    "STATE_SEALED",
    "CommitRecord",
    "OutcomeReader",
    "OutcomeStoreError",
    "OutcomeWriter",
    "assemble_record",
    "audit_store_integrity",
    "build_metadata",
    "build_stored_record",
    "canonical_json",
    "commit_digest",
    "corpus_content_digest",
    "decode_payload",
    "encode_payload",
    "file_set_name",
    "metadata_digest",
    "metadata_line",
    "next_segment",
    "payload_digest",
    "read_journal",
    "read_metadata_file",
    "read_seal",
    "read_state",
    "reconcile_corpus",
    "reconcile_file_set",
    "require_collecting",
    "seal_corpus",
    "shard_name",
    "storage_summary",
    "verify_seal",
    "write_state",
]
