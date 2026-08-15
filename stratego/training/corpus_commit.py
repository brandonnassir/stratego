"""Phase 8 Agent 2: the crash-safe commit store of the synthetic corpus.

Specification sources:

- `02_AGENT_2_SYNTHETIC_CORPUS.md` ("Crash-safe commit design", "Required
  injected-crash tests", "Corpus finalization")
- `00_PHASE_8_SEQUENCE_AND_COMMON_CONTRACT.md` section 24 (corpus crash/restart
  requirement)

The problem this closes
-----------------------
Phase 7 left one theoretical window open: a crash between the provenance
sidecar write and the trajectory write leaves two files disagreeing about which
games exist. Phase 8's corpus is static and is read thousands of times by the
trainer, so "probably consistent" is not good enough — a half-written game must
be *invisible*, not merely rare.

The rule is one sentence: **a game becomes visible only when its commit record
exists.** The commit record is written last, after the trajectory bytes and the
metadata line are both on the filesystem and both verify.

Write-ahead layout
------------------
One *file set* belongs to one (split, segment, worker) and is the only thing
that process appends to::

    <root>/<split>/shards/seg0000_w00_s0000.stgshard    trajectory payloads
    <root>/<split>/metadata/seg0000_w00.meta.jsonl      one JSON line per game
    <root>/<split>/journal/seg0000_w00.commit.jsonl     one JSON line per commit

The shard container is byte-identical to the accepted Phase 6B format
(:mod:`stratego.training.shard_writer`): magic, format version, JSON header,
then length-prefixed payloads. Existing readers work unchanged, and at
finalization each shard also gets the Phase 6B sibling manifest, so
`verify_shard` accepts a finalized corpus.

Every commit record carries the two file sizes *after* its own writes:

```text
shard_name  shard_bytes_after  metadata_bytes_after
```

which is what makes recovery a truncation instead of a rewrite. A shard rolls
over only *between* games, at a committed boundary, so a closed shard never
contains an uncommitted record.

Recovery
--------
:func:`reconcile_corpus` runs before any generation and is the only thing that
may modify already-written bytes:

```text
for every file set:
    drop a torn tail line from the journal
    truncate the metadata file to the last commit's metadata_bytes_after
    truncate the last committed shard to its shard_bytes_after
    remove shards written entirely after the last commit
```

After it returns there are, by construction, zero orphan trajectory records and
zero orphan metadata records: every byte that survives belongs to a committed
game. Uncommitted work is discarded rather than repaired, which is safe because
a corpus game's content is a pure function of its identifier — regenerating it
reproduces the same bytes.

Nothing here changes `trajectory_v1`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .rule_population import validate_game_metadata
from .serialization import DEFAULT_COMPRESSION_LEVEL, compress, decompress
from .shard_writer import (
    MANIFEST_SUFFIX,
    SHARD_FORMAT_VERSION,
    SHARD_MAGIC,
    SHARD_SUFFIX,
    ShardError,
    _HEADER,
    _LENGTH,
    iter_shard_payloads,
    read_shard_header,
)
from .trajectory import (
    TRAJECTORY_VERSION,
    GameRecord,
    decode_game_record,
    encode_game_record,
    validate_game_record,
)

#: Version of the commit protocol: the journal fields, the ordering rule and the
#: truncation-based recovery. A change to any of the three is a new version.
CORPUS_COMMIT_VERSION = "warmstart_corpus_commit_v1"

SHARDS_DIRECTORY = "shards"
METADATA_DIRECTORY = "metadata"
JOURNAL_DIRECTORY = "journal"

METADATA_SUFFIX = ".meta.jsonl"
JOURNAL_SUFFIX = ".commit.jsonl"

#: Default shard rollover size. Small enough that the finalized corpus is a
#: handful of files per split rather than one huge one, large enough that the
#: per-shard header is noise.
DEFAULT_CORPUS_SHARD_BYTES = 32 * 1024 * 1024

#: Every field of a commit record. A line missing any of them is not a commit.
COMMIT_FIELDS = (
    "commit_version",
    "game_id",
    "split",
    "shard_name",
    "record_index",
    "shard_bytes_after",
    "metadata_bytes_after",
    "trajectory_sha256",
    "metadata_sha256",
    "final_ply",
    "total_decisions",
    "committed_unix",
)

_FILE_SET_PATTERN = re.compile(r"^seg(?P<segment>\d{4})_w(?P<worker>\d{2})$")
_SHARD_PATTERN = re.compile(
    r"^seg(?P<segment>\d{4})_w(?P<worker>\d{2})_s(?P<shard>\d{4})$"
)


class CorpusCommitError(RuntimeError):
    """The corpus store could not be written, read back or reconciled."""


# ---------------------------------------------------------------------------
# Digests and paths
# ---------------------------------------------------------------------------


def payload_digest(payload: bytes) -> str:
    """SHA-256 of one stored trajectory payload, exactly as written."""
    return hashlib.sha256(payload).hexdigest()


def metadata_line(metadata: dict) -> str:
    """The canonical one-line JSON form of a metadata record."""
    return json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"


def metadata_digest(metadata: dict) -> str:
    """SHA-256 over the canonical metadata line, without the newline."""
    return hashlib.sha256(metadata_line(metadata)[:-1].encode()).hexdigest()


def file_set_name(segment: int, worker_id: int) -> str:
    return f"seg{int(segment):04d}_w{int(worker_id):02d}"


def shard_name(segment: int, worker_id: int, shard_index: int) -> str:
    return f"{file_set_name(segment, worker_id)}_s{int(shard_index):04d}"


def split_root(root: "str | Path", split: str) -> Path:
    return Path(root) / split


def shards_directory(root: "str | Path", split: str) -> Path:
    return split_root(root, split) / SHARDS_DIRECTORY


def metadata_directory(root: "str | Path", split: str) -> Path:
    return split_root(root, split) / METADATA_DIRECTORY


def journal_directory(root: "str | Path", split: str) -> Path:
    return split_root(root, split) / JOURNAL_DIRECTORY


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


# ---------------------------------------------------------------------------
# The append-only journal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommitRecord:
    """One committed game, as the journal stores it."""

    game_id: str
    split: str
    shard_name: str
    record_index: int
    shard_bytes_after: int
    metadata_bytes_after: int
    trajectory_sha256: str
    metadata_sha256: str
    final_ply: int
    total_decisions: int
    committed_unix: float

    def to_dict(self) -> dict:
        return {
            "commit_version": CORPUS_COMMIT_VERSION,
            "game_id": self.game_id,
            "split": self.split,
            "shard_name": self.shard_name,
            "record_index": self.record_index,
            "shard_bytes_after": self.shard_bytes_after,
            "metadata_bytes_after": self.metadata_bytes_after,
            "trajectory_sha256": self.trajectory_sha256,
            "metadata_sha256": self.metadata_sha256,
            "final_ply": self.final_ply,
            "total_decisions": self.total_decisions,
            "committed_unix": self.committed_unix,
        }

    @staticmethod
    def from_dict(payload: dict) -> "CommitRecord":
        return CommitRecord(
            game_id=str(payload["game_id"]),
            split=str(payload["split"]),
            shard_name=str(payload["shard_name"]),
            record_index=int(payload["record_index"]),
            shard_bytes_after=int(payload["shard_bytes_after"]),
            metadata_bytes_after=int(payload["metadata_bytes_after"]),
            trajectory_sha256=str(payload["trajectory_sha256"]),
            metadata_sha256=str(payload["metadata_sha256"]),
            final_ply=int(payload["final_ply"]),
            total_decisions=int(payload["total_decisions"]),
            committed_unix=float(payload["committed_unix"]),
        )


def read_journal(path: "str | Path") -> tuple:
    """`(commits, valid_bytes)` for one journal file.

    Only newline-terminated, fully parseable lines carrying every commit field
    are accepted, so a process killed mid-line contributes nothing. `valid_bytes`
    is where the journal would be truncated to remove the torn tail.
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
        if not isinstance(payload, dict) or any(
            key not in payload for key in COMMIT_FIELDS
        ):
            break
        if payload["commit_version"] != CORPUS_COMMIT_VERSION:
            raise CorpusCommitError(
                f"{path}: commit protocol {payload['commit_version']!r} is not "
                f"{CORPUS_COMMIT_VERSION!r}"
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
        if not isinstance(payload, dict) or "synthetic_game_id" not in payload:
            break
        records.append(payload)
        valid = offset
    return (records, valid)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

#: Named points a test may interrupt a write at. The writer calls the injected
#: hook at each one; a hook that raises simulates a process dying there.
CRASH_STAGES = (
    "before_trajectory",
    "after_trajectory",
    "after_metadata",
    "before_commit_flush",
    "after_commit",
    "shard_rollover",
)


class CorpusWriter:
    """The append-only writer of one (split, segment, worker) file set.

    Not shared and not thread-safe: one instance belongs to one process, which
    is the only thing that ever appends to these three files.
    """

    def __init__(
        self,
        root: "str | Path",
        *,
        split: str,
        segment: int,
        worker_id: int,
        target_bytes: int = DEFAULT_CORPUS_SHARD_BYTES,
        compression_level: int = DEFAULT_COMPRESSION_LEVEL,
        fsync_on_commit: bool = False,
        crash_hook=None,
    ) -> None:
        self.root = Path(root)
        self.split = str(split)
        self.segment = int(segment)
        self.worker_id = int(worker_id)
        self.target_bytes = int(target_bytes)
        self.compression_level = int(compression_level)
        self.fsync_on_commit = bool(fsync_on_commit)
        self.crash_hook = crash_hook

        self.name = file_set_name(self.segment, self.worker_id)
        self.shards_directory = shards_directory(self.root, self.split)
        self.metadata_path = (
            metadata_directory(self.root, self.split) / f"{self.name}{METADATA_SUFFIX}"
        )
        self.journal_path = (
            journal_directory(self.root, self.split) / f"{self.name}{JOURNAL_SUFFIX}"
        )
        for directory in (
            self.shards_directory,
            self.metadata_path.parent,
            self.journal_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        if self.metadata_path.exists() or self.journal_path.exists():
            raise CorpusCommitError(
                f"file set {self.name} of split {self.split!r} already exists; a "
                "resumed run must open a fresh segment rather than append to a "
                "reconciled one"
            )

        self.commits: list[CommitRecord] = []
        self.games_written = 0
        self.uncompressed_bytes = 0
        self.compressed_bytes = 0
        self.encode_seconds = 0.0
        self.compress_seconds = 0.0
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
        self._shard_path = self.shards_directory / f"{name}{SHARD_SUFFIX}"
        header = {
            "run_id": self.name,
            "worker_id": self.worker_id,
            "shard_index": self._shard_index,
            "trajectory_version": TRAJECTORY_VERSION,
            "collection_policy_version": CORPUS_COMMIT_VERSION,
            "corpus_split": self.split,
            "compressed": True,
            "compression": "zlib",
            "compression_level": self.compression_level,
            "opened_unix": time.time(),
        }
        blob = json.dumps(header, sort_keys=True).encode()
        self._shard_handle = self._shard_path.open("wb")
        preamble = SHARD_MAGIC + _HEADER.pack(SHARD_FORMAT_VERSION, len(blob)) + blob
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

    def write_game(self, game) -> CommitRecord:
        """Persist and commit one game, in the frozen order.

        ```text
        1. encode + compress + decode-verify the trajectory payload
        2. validate the metadata against the record
        3. append the trajectory bytes, flush
        4. append the metadata line, flush
        5. append the commit line, flush
        ```

        A failure or interruption anywhere before step 5 leaves the game
        uncommitted and therefore invisible; :func:`reconcile_corpus` removes
        the partial bytes on the next open.
        """
        record = game.record
        metadata = game.metadata
        if record.game_id != metadata["synthetic_game_id"]:
            raise CorpusCommitError(
                f"record {record.game_id!r} does not match metadata "
                f"{metadata['synthetic_game_id']!r}"
            )
        if metadata["corpus_split"] != self.split:
            raise CorpusCommitError(
                f"game {record.game_id} belongs to split "
                f"{metadata['corpus_split']!r}, not {self.split!r}"
            )

        started = time.perf_counter()
        raw = encode_game_record(record)
        self.encode_seconds += time.perf_counter() - started

        started = time.perf_counter()
        payload = compress(raw, self.compression_level)
        self.compress_seconds += time.perf_counter() - started

        # "Verifies" means the bytes decode back to this game, not that they are
        # the right length. Doing it before the write is what lets the commit
        # rule promise that a visible record is a readable one.
        started = time.perf_counter()
        problems = _verify_payload(payload, record)
        problems.extend(validate_game_metadata(metadata, record))
        self.verify_seconds += time.perf_counter() - started
        if problems:
            raise CorpusCommitError(
                f"game {record.game_id} failed pre-commit verification: {problems}"
            )

        self._roll_shard_if_needed()
        self._hook("before_trajectory")

        line = metadata_line(metadata)
        write_started = time.perf_counter()
        frame = _LENGTH.pack(len(payload)) + payload
        self._shard_handle.write(frame)
        self._shard_handle.flush()
        shard_bytes_after = self._shard_handle.tell()
        record_index = self._shard_records
        self._shard_records += 1
        self._hook("after_trajectory")

        self._metadata_handle.write(line.encode())
        self._metadata_handle.flush()
        metadata_bytes_after = self._metadata_handle.tell()
        self._hook("after_metadata")

        commit = CommitRecord(
            game_id=record.game_id,
            split=self.split,
            shard_name=self.current_shard_name,
            record_index=record_index,
            shard_bytes_after=shard_bytes_after,
            metadata_bytes_after=metadata_bytes_after,
            trajectory_sha256=payload_digest(payload),
            metadata_sha256=metadata_digest(metadata),
            final_ply=int(record.final_ply),
            total_decisions=len(record.decisions),
            committed_unix=time.time(),
        )
        self._hook("before_commit_flush")
        self._journal_handle.write(
            (json.dumps(commit.to_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
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
            "split": self.split,
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
            "encode_seconds": self.encode_seconds,
            "compress_seconds": self.compress_seconds,
            "verify_seconds": self.verify_seconds,
            "write_seconds": self.write_seconds,
        }


def _verify_payload(payload: bytes, record: GameRecord) -> list:
    """Every disagreement between stored bytes and the record they came from."""
    try:
        decoded = decode_game_record(decompress(payload))
    except Exception as error:  # noqa: BLE001 - an undecodable payload is a finding
        return [f"payload does not decode: {type(error).__name__}: {error}"]
    problems = validate_game_record(decoded)
    if decoded.game_id != record.game_id:
        problems.append("decoded payload carries a different game id")
    if decoded.actions != record.actions:
        problems.append("decoded payload carries a different action sequence")
    if decoded.terminal_result != record.terminal_result:
        problems.append("decoded payload carries a different result")
    if decoded.terminal_reason != record.terminal_reason:
        problems.append("decoded payload carries a different terminal reason")
    if (decoded.red_setup, decoded.blue_setup) != (record.red_setup, record.blue_setup):
        problems.append("decoded payload carries different setups")
    return problems


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def _file_sets(root: "str | Path", split: str) -> list:
    """Every (segment, worker) file set that exists for one split."""
    names = set()
    for path in journal_directory(root, split).glob(f"*{JOURNAL_SUFFIX}"):
        names.add(path.name[: -len(JOURNAL_SUFFIX)])
    for path in metadata_directory(root, split).glob(f"*{METADATA_SUFFIX}"):
        names.add(path.name[: -len(METADATA_SUFFIX)])
    for path in shards_directory(root, split).glob(f"*{SHARD_SUFFIX}"):
        match = _SHARD_PATTERN.match(path.stem)
        if match is not None:
            names.add(file_set_name(int(match["segment"]), int(match["worker"])))
    resolved = []
    for name in sorted(names):
        match = _FILE_SET_PATTERN.match(name)
        if match is None:
            raise CorpusCommitError(f"unrecognized corpus file set name: {name!r}")
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


def reconcile_file_set(root: "str | Path", split: str, name: str) -> dict:
    """Cut one file set back to its last commit and report what was discarded.

    This is the only function that ever removes corpus bytes. It removes exactly
    the bytes no commit record claims, which by the write order can only be work
    that was interrupted before it became visible.
    """
    root = Path(root)
    journal_path = journal_directory(root, split) / f"{name}{JOURNAL_SUFFIX}"
    metadata_path = metadata_directory(root, split) / f"{name}{METADATA_SUFFIX}"
    shard_directory = shards_directory(root, split)

    commits, journal_valid_bytes = read_journal(journal_path)
    report = {
        "file_set": name,
        "split": split,
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
        raise CorpusCommitError(f"unrecognized corpus file set name: {name!r}")
    segment, worker = int(match["segment"]), int(match["worker"])
    last_shard_index = -1
    if last is not None:
        shard_match = _SHARD_PATTERN.match(last.shard_name)
        if shard_match is None:
            raise CorpusCommitError(
                f"commit for {last.game_id} names shard {last.shard_name!r}, which is "
                "not a corpus shard name"
            )
        last_shard_index = int(shard_match["shard"])
        shard_path = shard_directory / f"{last.shard_name}{SHARD_SUFFIX}"
        if not shard_path.exists():
            raise CorpusCommitError(
                f"committed game {last.game_id} names missing shard {shard_path}"
            )
        report["shard_bytes_discarded"] += _truncate(shard_path, last.shard_bytes_after)

    for path in sorted(shard_directory.glob(f"{name}_s*{SHARD_SUFFIX}")):
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
        manifest = path.with_suffix(MANIFEST_SUFFIX)
        if manifest.exists():
            manifest.unlink()

    return report


def reconcile_corpus(root: "str | Path", splits: "tuple[str, ...]") -> dict:
    """Reconcile every file set of every split before generation resumes.

    Returns the committed index plus a per-file-set account of what was
    discarded. After this call the on-disk corpus contains committed games and
    nothing else, which is the precondition every later audit assumes.
    """
    root = Path(root)
    reports = []
    committed: dict[str, CommitRecord] = {}
    duplicates: list[str] = []
    for split in splits:
        for _segment, _worker, name in _file_sets(root, split):
            report = reconcile_file_set(root, split, name)
            reports.append(report)
            commits, _ = read_journal(
                journal_directory(root, split) / f"{name}{JOURNAL_SUFFIX}"
            )
            for commit in commits:
                if commit.game_id in committed:
                    duplicates.append(commit.game_id)
                committed[commit.game_id] = commit
    return {
        "commit_version": CORPUS_COMMIT_VERSION,
        "committed": committed,
        "committed_count": len(committed),
        "duplicate_committed_ids": sorted(set(duplicates)),
        "file_sets": reports,
        "shards_removed": [
            name for report in reports for name in report["shards_removed"]
        ],
        "bytes_discarded": sum(
            report["journal_bytes_discarded"]
            + report["metadata_bytes_discarded"]
            + report["shard_bytes_discarded"]
            for report in reports
        ),
    }


def next_segment(root: "str | Path", splits: "tuple[str, ...]") -> int:
    """The first segment number no split has used.

    A resumed run always writes a fresh segment. Appending to a reconciled file
    set would work, but a segment boundary makes "these bytes were written by
    that attempt" readable straight off the filenames.
    """
    highest = -1
    for split in splits:
        for segment, _worker, _name in _file_sets(root, split):
            highest = max(highest, segment)
    return highest + 1


# ---------------------------------------------------------------------------
# Reading a committed corpus
# ---------------------------------------------------------------------------


class CorpusReader:
    """Random access to the committed games of one corpus root.

    The commit journals are the index: a game the journals do not name does not
    exist, whatever bytes happen to be on disk. That is the same rule the
    trainer relies on, so a reader and a resume can never disagree about the
    corpus contents.
    """

    def __init__(self, root: "str | Path", splits: "tuple[str, ...]") -> None:
        self.root = Path(root)
        self.splits = tuple(splits)
        self.commits: dict[str, CommitRecord] = {}
        self.by_split: dict[str, list] = {split: [] for split in self.splits}
        self._metadata: dict[str, dict] = {}
        self._shard_offsets: dict[tuple, list] = {}
        for split in self.splits:
            for _segment, _worker, name in _file_sets(self.root, split):
                commits, _ = read_journal(
                    journal_directory(self.root, split) / f"{name}{JOURNAL_SUFFIX}"
                )
                for commit in commits:
                    if commit.game_id in self.commits:
                        raise CorpusCommitError(
                            f"duplicate committed game id {commit.game_id!r}"
                        )
                    self.commits[commit.game_id] = commit
                    self.by_split[split].append(commit.game_id)
                records, _ = read_metadata_file(
                    metadata_directory(self.root, split) / f"{name}{METADATA_SUFFIX}"
                )
                for record in records:
                    game_id = str(record["synthetic_game_id"])
                    if game_id in self._metadata:
                        raise CorpusCommitError(
                            f"duplicate metadata record for {game_id!r}"
                        )
                    self._metadata[game_id] = record
        for split in self.splits:
            self.by_split[split].sort()

    def __len__(self) -> int:
        return len(self.commits)

    def game_ids(self, split: "str | None" = None) -> tuple:
        if split is None:
            return tuple(sorted(self.commits))
        return tuple(self.by_split[split])

    def metadata(self, game_id: str) -> dict:
        """The committed metadata record of one game."""
        try:
            return self._metadata[game_id]
        except KeyError:
            raise CorpusCommitError(
                f"no committed metadata for game {game_id!r}"
            ) from None

    def payload(self, game_id: str) -> bytes:
        """The stored trajectory payload of one committed game."""
        commit = self.commits.get(game_id)
        if commit is None:
            raise CorpusCommitError(f"game {game_id!r} is not committed")
        path = (
            shards_directory(self.root, commit.split)
            / f"{commit.shard_name}{SHARD_SUFFIX}"
        )
        offsets = self._shard_offsets.get((commit.split, commit.shard_name))
        if offsets is None:
            offsets = _shard_frame_offsets(path)
            self._shard_offsets[(commit.split, commit.shard_name)] = offsets
        if not 0 <= commit.record_index < len(offsets):
            raise CorpusCommitError(
                f"game {game_id!r} names record {commit.record_index} of "
                f"{commit.shard_name}, which holds {len(offsets)} records"
            )
        offset, size = offsets[commit.record_index]
        with path.open("rb") as handle:
            handle.seek(offset)
            payload = handle.read(size)
        if len(payload) != size:  # pragma: no cover - truncation is reconciled away
            raise CorpusCommitError(f"game {game_id!r}: short read from {path}")
        return payload

    def record(self, game_id: str) -> GameRecord:
        """The decoded `trajectory_v1` record of one committed game."""
        return decode_game_record(decompress(self.payload(game_id)))

    def game(self, game_id: str) -> tuple:
        """`(record, metadata)` for one committed game."""
        return (self.record(game_id), self.metadata(game_id))

    def iter_games(self, split: "str | None" = None):
        """Every committed game of a split, ascending by game id."""
        for game_id in self.game_ids(split):
            yield self.game(game_id)

    def shard_paths(self, split: str) -> list:
        return sorted(shards_directory(self.root, split).glob(f"*{SHARD_SUFFIX}"))


def _shard_frame_offsets(path: Path) -> list:
    """`(payload_offset, payload_size)` of every frame in one shard."""
    if not path.exists():
        raise CorpusCommitError(f"missing shard file {path}")
    offsets = []
    file_bytes = path.stat().st_size
    with path.open("rb") as handle:
        magic = handle.read(len(SHARD_MAGIC))
        if magic != SHARD_MAGIC:
            raise ShardError(f"{path} is not a shard file")
        version, length = _HEADER.unpack(handle.read(_HEADER.size))
        if version != SHARD_FORMAT_VERSION:
            raise ShardError(f"{path} has shard format version {version}")
        handle.read(length)
        while True:
            raw_length = handle.read(_LENGTH.size)
            if len(raw_length) < _LENGTH.size:
                return offsets
            (size,) = _LENGTH.unpack(raw_length)
            offset = handle.tell()
            if offset + size > file_bytes:
                # A frame the file does not fully contain is an interrupted
                # write; reconciliation removes it, and it is never indexed.
                return offsets
            offsets.append((offset, size))
            handle.seek(size, os.SEEK_CUR)


# ---------------------------------------------------------------------------
# Auditing a committed corpus
# ---------------------------------------------------------------------------


def audit_commit_integrity(root: "str | Path", splits: "tuple[str, ...]") -> dict:
    """Reconcile the three identity sets and count every kind of orphan.

    ```text
    committed ids   the commit journals
    metadata ids    the metadata sidecars
    payload ids     the trajectory bytes actually in the shards
    ```

    A finalized corpus has all three equal, with every payload digest matching
    the digest its commit recorded.
    """
    root = Path(root)
    committed: dict[str, CommitRecord] = {}
    duplicate_committed: list[str] = []
    metadata_ids: list[str] = []
    payload_ids: list[str] = []
    digest_mismatches: list[str] = []
    metadata_digest_mismatches: list[str] = []
    decode_failures: list[str] = []
    split_of_payload: dict[str, str] = {}

    for split in splits:
        for _segment, _worker, name in _file_sets(root, split):
            commits, _ = read_journal(
                journal_directory(root, split) / f"{name}{JOURNAL_SUFFIX}"
            )
            for commit in commits:
                if commit.game_id in committed:
                    duplicate_committed.append(commit.game_id)
                committed[commit.game_id] = commit
            records, _ = read_metadata_file(
                metadata_directory(root, split) / f"{name}{METADATA_SUFFIX}"
            )
            for record in records:
                game_id = str(record["synthetic_game_id"])
                metadata_ids.append(game_id)
                commit = committed.get(game_id)
                if commit is not None and metadata_digest(record) != commit.metadata_sha256:
                    metadata_digest_mismatches.append(game_id)
        for path in sorted(shards_directory(root, split).glob(f"*{SHARD_SUFFIX}")):
            for payload in iter_shard_payloads(path):
                try:
                    record = decode_game_record(decompress(payload))
                except Exception as error:  # noqa: BLE001 - a bad payload is a finding
                    decode_failures.append(f"{path.name}: {type(error).__name__}: {error}")
                    continue
                payload_ids.append(record.game_id)
                split_of_payload[record.game_id] = split
                commit = committed.get(record.game_id)
                if commit is not None and payload_digest(payload) != commit.trajectory_sha256:
                    digest_mismatches.append(record.game_id)

    committed_ids = set(committed)
    metadata_set = set(metadata_ids)
    payload_set = set(payload_ids)
    return {
        "committed_count": len(committed),
        "metadata_count": len(metadata_ids),
        "payload_count": len(payload_ids),
        "duplicate_committed_ids": sorted(set(duplicate_committed)),
        "duplicate_metadata_ids": sorted(
            {name for name in metadata_ids if metadata_ids.count(name) > 1}
        )
        if len(metadata_ids) != len(metadata_set)
        else [],
        "duplicate_payload_ids": sorted(
            {name for name in payload_ids if payload_ids.count(name) > 1}
        )
        if len(payload_ids) != len(payload_set)
        else [],
        "orphan_trajectory_records": sorted(payload_set - committed_ids),
        "orphan_metadata_records": sorted(metadata_set - committed_ids),
        "missing_trajectory_payloads": sorted(committed_ids - payload_set),
        "missing_metadata_records": sorted(committed_ids - metadata_set),
        "trajectory_digest_mismatches": sorted(set(digest_mismatches)),
        "metadata_digest_mismatches": sorted(set(metadata_digest_mismatches)),
        "payload_decode_failures": decode_failures,
        "split_placement_violations": sorted(
            game_id
            for game_id, split in split_of_payload.items()
            if game_id in committed and committed[game_id].split != split
        ),
    }


def corpus_content_digest(root: "str | Path", splits: "tuple[str, ...]") -> str:
    """SHA-256 over the committed corpus, independent of how it was written.

    Built from `game_id | trajectory digest | metadata digest` sorted by game
    id, so shard filenames, worker counts, segment numbers and arrival order
    cannot appear in it. Two runs that commit the same logical games agree here
    even if one of them crashed and resumed.
    """
    reader_commits: list[CommitRecord] = []
    for split in splits:
        for _segment, _worker, name in _file_sets(root, split):
            commits, _ = read_journal(
                journal_directory(root, split) / f"{name}{JOURNAL_SUFFIX}"
            )
            reader_commits.extend(commits)
    hasher = hashlib.sha256()
    for commit in sorted(reader_commits, key=lambda entry: entry.game_id):
        hasher.update(
            f"{commit.game_id}|{commit.trajectory_sha256}|{commit.metadata_sha256}\n".encode()
        )
    return hasher.hexdigest()


def write_shard_manifests(root: "str | Path", splits: "tuple[str, ...]") -> list:
    """Write the Phase 6B sibling manifest of every finalized shard.

    Only at finalization, and only from the bytes actually on disk after
    reconciliation, so a manifest can never describe a shard that was later
    truncated. With the manifests present, `shard_writer.verify_shard` and
    `directory_summary` accept the corpus unchanged.
    """
    manifests = []
    for split in splits:
        for path in sorted(shards_directory(root, split).glob(f"*{SHARD_SUFFIX}")):
            header = read_shard_header(path)
            game_ids = []
            records = 0
            uncompressed = 0
            compressed = 0
            for payload in iter_shard_payloads(path):
                body = decompress(payload)
                game_ids.append(decode_game_record(body).game_id)
                records += 1
                compressed += len(payload)
                uncompressed += len(body)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {
                "run_id": header["run_id"],
                "worker_id": header["worker_id"],
                "shard_index": header["shard_index"],
                "data_file": path.name,
                "format_version": SHARD_FORMAT_VERSION,
                "trajectory_version": TRAJECTORY_VERSION,
                "collection_policy_version": header["collection_policy_version"],
                "corpus_split": split,
                "compressed": True,
                "compression": "zlib",
                "records": records,
                "uncompressed_bytes": uncompressed,
                "compressed_bytes": compressed,
                "file_bytes": path.stat().st_size,
                "sha256": digest,
                "game_ids": game_ids,
                "seconds_open": 0.0,
                "closed_unix": time.time(),
            }
            path.with_suffix(MANIFEST_SUFFIX).write_text(
                json.dumps(manifest, sort_keys=True) + "\n"
            )
            manifests.append(manifest)
    return manifests


def storage_summary(root: "str | Path", splits: "tuple[str, ...]") -> dict:
    """Bytes on disk by kind, per split and overall."""
    root = Path(root)
    per_split = {}
    for split in splits:
        shard_bytes = sum(
            path.stat().st_size
            for path in shards_directory(root, split).glob(f"*{SHARD_SUFFIX}")
        )
        manifest_bytes = sum(
            path.stat().st_size
            for path in shards_directory(root, split).glob(f"*{MANIFEST_SUFFIX}")
        )
        metadata_bytes = sum(
            path.stat().st_size
            for path in metadata_directory(root, split).glob(f"*{METADATA_SUFFIX}")
        )
        journal_bytes = sum(
            path.stat().st_size
            for path in journal_directory(root, split).glob(f"*{JOURNAL_SUFFIX}")
        )
        per_split[split] = {
            "shard_bytes": shard_bytes,
            "shard_manifest_bytes": manifest_bytes,
            "metadata_bytes": metadata_bytes,
            "journal_bytes": journal_bytes,
            "total_bytes": shard_bytes + manifest_bytes + metadata_bytes + journal_bytes,
            "shard_files": len(list(shards_directory(root, split).glob(f"*{SHARD_SUFFIX}"))),
        }
    return {
        "per_split": per_split,
        "total_bytes": sum(entry["total_bytes"] for entry in per_split.values()),
        "shard_bytes": sum(entry["shard_bytes"] for entry in per_split.values()),
        "metadata_bytes": sum(entry["metadata_bytes"] for entry in per_split.values()),
        "journal_bytes": sum(entry["journal_bytes"] for entry in per_split.values()),
    }


__all__ = [
    "COMMIT_FIELDS",
    "CORPUS_COMMIT_VERSION",
    "CRASH_STAGES",
    "DEFAULT_CORPUS_SHARD_BYTES",
    "JOURNAL_DIRECTORY",
    "JOURNAL_SUFFIX",
    "METADATA_DIRECTORY",
    "METADATA_SUFFIX",
    "SHARDS_DIRECTORY",
    "CommitRecord",
    "CorpusCommitError",
    "CorpusReader",
    "CorpusWriter",
    "audit_commit_integrity",
    "corpus_content_digest",
    "journal_directory",
    "metadata_digest",
    "metadata_directory",
    "metadata_line",
    "next_segment",
    "payload_digest",
    "read_journal",
    "read_metadata_file",
    "reconcile_corpus",
    "reconcile_file_set",
    "shard_name",
    "shards_directory",
    "split_root",
    "storage_summary",
    "write_shard_manifests",
]
