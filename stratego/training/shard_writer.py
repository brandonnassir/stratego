"""Phase 6B: durable shard files for collected trajectories.

Specification source: the Phase 6B production-recording follow-up.

What this is
------------
A minimal append-only container that groups already-encoded `trajectory_v1`
records into files on disk. It is a *container*, not a record format and not a
compression format: each element is exactly the payload
:func:`stratego.training.trajectory.encode_game_record` produces, optionally run
through the repository's existing `zlib` level-6 helper. Nothing about
`trajectory_v1` changes, and a reader that can decode a record today can decode
every record this writes.

Why per-worker files
--------------------
The collection pool is ten pure NumPy/engine processes and one Metal
coordinator. Each worker seals its own games, so each worker writes its own
shards and nothing large crosses a pipe. That has one property worth stating
plainly, because Phase 6B gates on it: **a write backlog is structurally
impossible.** The worker compresses and writes inside `finalise_recording`, so
the bytes are on the filesystem before the call returns and there is no queue to
grow. The cost is write latency charged to the worker phase, which is measured
rather than assumed -- the workers are idle roughly 91% of each step, so there is
room for it.

The file layout
---------------
Each shard is one `.stgshard` data file plus one `.json` manifest written when
the shard closes::

    magic            b"STGOSHRD"                8 bytes
    format version   uint32                     little endian
    header length    uint32
    header           UTF-8 JSON                 run/worker/shard identity
    then, repeated:
        payload length   uint32
        payload          bytes                  one encoded game record

A record is never split across shards. The manifest carries the record count,
both byte totals, the SHA-256 of the data file and every game id in the shard,
so a shard can be checked without decoding it and located without scanning.

Crash behaviour
---------------
The data file is flushed per record and the manifest is written last, so a shard
without a manifest is an interrupted shard: its complete records are still
readable, and :func:`read_shard` reports the truncation rather than raising.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

from .serialization import DEFAULT_COMPRESSION_LEVEL, compress, decompress
from .trajectory import (
    TRAJECTORY_VERSION,
    decode_game_record,
    encode_game_record,
    validate_game_record,
)

SHARD_MAGIC = b"STGOSHRD"
SHARD_FORMAT_VERSION = 1
SHARD_SUFFIX = ".stgshard"
MANIFEST_SUFFIX = ".json"

#: Default rollover size. Chosen so a 168-hour run at the measured compressed
#: rate produces a few thousand files rather than a few hundred thousand, while
#: still rolling often enough that an interrupted shard loses little.
DEFAULT_SHARD_TARGET_BYTES = 128 * 1024 * 1024

_LENGTH = struct.Struct("<I")
_HEADER = struct.Struct("<II")


class ShardError(RuntimeError):
    """A shard could not be written or read back."""


@dataclass
class ShardStats:
    """Everything Phase 6B has to report about the persistence path."""

    shards_opened: int = 0
    shards_closed: int = 0
    records_written: int = 0
    uncompressed_bytes: int = 0
    compressed_bytes: int = 0
    bytes_written: int = 0
    encode_seconds: float = 0.0
    compress_seconds: float = 0.0
    write_seconds: float = 0.0
    flush_seconds: float = 0.0
    write_errors: int = 0
    error_details: list = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        """Compressed over uncompressed. 1.0 when compression is off."""
        return (
            self.compressed_bytes / self.uncompressed_bytes
            if self.uncompressed_bytes
            else 0.0
        )

    @property
    def container_overhead_bytes(self) -> int:
        """What the container costs on top of the payloads themselves."""
        return self.bytes_written - self.compressed_bytes

    def as_dict(self) -> dict:
        return {
            "shards_opened": self.shards_opened,
            "shards_closed": self.shards_closed,
            "records_written": self.records_written,
            "uncompressed_bytes": self.uncompressed_bytes,
            "compressed_bytes": self.compressed_bytes,
            "bytes_written": self.bytes_written,
            "container_overhead_bytes": self.container_overhead_bytes,
            "compression_ratio": self.compression_ratio,
            "encode_seconds": self.encode_seconds,
            "compress_seconds": self.compress_seconds,
            "write_seconds": self.write_seconds,
            "flush_seconds": self.flush_seconds,
            "write_errors": self.write_errors,
            "error_details": list(self.error_details),
            # Synchronous by construction: the bytes are on the filesystem
            # before `write` returns, so nothing can queue up behind it.
            "pending_records": 0,
            "pending_bytes": 0,
            "backlog_is_structurally_impossible": True,
        }


class ShardWriter:
    """Append-only shard writer for one worker.

    Not thread-safe and not shared: one instance belongs to one worker process,
    which is the only thing that ever calls it.
    """

    def __init__(
        self,
        directory: "str | Path",
        *,
        worker_id: int,
        run_id: str,
        compress_records: bool = True,
        compression_level: int = DEFAULT_COMPRESSION_LEVEL,
        target_bytes: int = DEFAULT_SHARD_TARGET_BYTES,
        collection_policy_version: str = "",
        fsync_on_close: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.worker_id = int(worker_id)
        self.run_id = str(run_id)
        self.compress_records = bool(compress_records)
        self.compression_level = int(compression_level)
        self.target_bytes = int(target_bytes)
        self.collection_policy_version = collection_policy_version
        self.fsync_on_close = bool(fsync_on_close)
        self.stats = ShardStats()

        self.directory.mkdir(parents=True, exist_ok=True)
        self._handle = None
        self._path: Path | None = None
        self._shard_index = -1
        self._shard_bytes = 0
        self._shard_records = 0
        self._shard_uncompressed = 0
        self._shard_compressed = 0
        self._shard_digest = None
        self._shard_game_ids: list[str] = []
        self._shard_started = 0.0
        self.closed_shards: list[dict] = []

    # -- shard lifecycle ----------------------------------------------------

    def _shard_name(self, index: int) -> str:
        return f"{self.run_id}_w{self.worker_id:02d}_s{index:06d}"

    def _open_shard(self) -> None:
        self._shard_index += 1
        name = self._shard_name(self._shard_index)
        self._path = self.directory / f"{name}{SHARD_SUFFIX}"
        header = {
            "run_id": self.run_id,
            "worker_id": self.worker_id,
            "shard_index": self._shard_index,
            "trajectory_version": TRAJECTORY_VERSION,
            "collection_policy_version": self.collection_policy_version,
            "compressed": self.compress_records,
            "compression": "zlib" if self.compress_records else "none",
            "compression_level": (
                self.compression_level if self.compress_records else 0
            ),
            "opened_unix": time.time(),
        }
        blob = json.dumps(header, sort_keys=True).encode()
        self._handle = self._path.open("wb")
        preamble = SHARD_MAGIC + _HEADER.pack(SHARD_FORMAT_VERSION, len(blob)) + blob
        self._handle.write(preamble)
        # Counted into `bytes_written` so the stat equals the bytes actually on
        # disk; a storage projection built from it must not silently omit the
        # per-shard header.
        self.stats.bytes_written += len(preamble)
        self._shard_bytes = len(preamble)
        self._shard_records = 0
        self._shard_uncompressed = 0
        self._shard_compressed = 0
        self._shard_digest = hashlib.sha256(preamble)
        self._shard_game_ids = []
        self._shard_started = time.perf_counter()
        self.stats.shards_opened += 1

    def _close_shard(self) -> None:
        """Flush, fsync and write the manifest. The manifest is written last."""
        if self._handle is None or self._path is None:
            return
        started = time.perf_counter()
        self._handle.flush()
        if self.fsync_on_close:
            os.fsync(self._handle.fileno())
        self._handle.close()
        self.stats.flush_seconds += time.perf_counter() - started
        self._handle = None

        manifest = {
            "run_id": self.run_id,
            "worker_id": self.worker_id,
            "shard_index": self._shard_index,
            "data_file": self._path.name,
            "format_version": SHARD_FORMAT_VERSION,
            "trajectory_version": TRAJECTORY_VERSION,
            "collection_policy_version": self.collection_policy_version,
            "compressed": self.compress_records,
            "compression": "zlib" if self.compress_records else "none",
            "records": self._shard_records,
            "uncompressed_bytes": self._shard_uncompressed,
            "compressed_bytes": self._shard_compressed,
            "file_bytes": self._shard_bytes,
            "sha256": self._shard_digest.hexdigest(),
            "game_ids": list(self._shard_game_ids),
            "seconds_open": time.perf_counter() - self._shard_started,
            "closed_unix": time.time(),
        }
        manifest_path = self._path.with_suffix(MANIFEST_SUFFIX)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        self.closed_shards.append(manifest)
        self.stats.shards_closed += 1
        self._path = None

    # -- writing ------------------------------------------------------------

    def write(self, record) -> dict:
        """Encode, optionally compress, and append one game record.

        Returns the per-record accounting. Raises :class:`ShardError` only if the
        filesystem refuses the write; the caller decides whether that ends the
        run, because a collection process that cannot persist is not doing its
        job even though the model is still healthy.
        """
        encode_started = time.perf_counter()
        raw = encode_game_record(record)
        encode_seconds = time.perf_counter() - encode_started
        self.stats.encode_seconds += encode_seconds

        compress_seconds = 0.0
        if self.compress_records:
            compress_started = time.perf_counter()
            payload = compress(raw, self.compression_level)
            compress_seconds = time.perf_counter() - compress_started
            self.stats.compress_seconds += compress_seconds
        else:
            payload = raw

        if self._handle is None:
            self._open_shard()

        frame = _LENGTH.pack(len(payload)) + payload
        write_started = time.perf_counter()
        try:
            self._handle.write(frame)
            self._handle.flush()
        except (OSError, ValueError) as error:
            # OSError is the filesystem refusing the bytes; ValueError is
            # CPython's "I/O operation on closed file", which is the same
            # operational condition -- the worker can no longer persist.
            self.stats.write_errors += 1
            detail = f"{type(error).__name__}: {error}"
            if len(self.stats.error_details) < 20:
                self.stats.error_details.append(detail)
            raise ShardError(f"could not write to {self._path}: {detail}") from error
        write_seconds = time.perf_counter() - write_started
        self.stats.write_seconds += write_seconds

        self._shard_digest.update(frame)
        self._shard_bytes += len(frame)
        self._shard_records += 1
        self._shard_uncompressed += len(raw)
        self._shard_compressed += len(payload)
        self._shard_game_ids.append(record.game_id)

        self.stats.records_written += 1
        self.stats.uncompressed_bytes += len(raw)
        self.stats.compressed_bytes += len(payload)
        self.stats.bytes_written += len(frame)

        if self._shard_bytes >= self.target_bytes:
            self._close_shard()

        return {
            "uncompressed_bytes": len(raw),
            "compressed_bytes": len(payload),
            "frame_bytes": len(frame),
            "encode_seconds": encode_seconds,
            "compress_seconds": compress_seconds,
            "write_seconds": write_seconds,
        }

    def close(self) -> dict:
        """Finish the open shard. Safe to call more than once."""
        self._close_shard()
        return self.stats.as_dict()


# ---------------------------------------------------------------------------
# Reading back
# ---------------------------------------------------------------------------


def _file_sha256(path: Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    """The file's digest, read in chunks so a shard never loads whole."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def read_shard_header(path: "str | Path") -> dict:
    """The shard's header without reading any record."""
    path = Path(path)
    with path.open("rb") as handle:
        magic = handle.read(len(SHARD_MAGIC))
        if magic != SHARD_MAGIC:
            raise ShardError(f"{path} is not a shard file")
        version, length = _HEADER.unpack(handle.read(_HEADER.size))
        if version != SHARD_FORMAT_VERSION:
            raise ShardError(f"{path} has shard format version {version}")
        return json.loads(handle.read(length).decode())


def iter_shard_payloads(path: "str | Path"):
    """Yield each stored payload, stopping cleanly at a truncated tail."""
    path = Path(path)
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
            if not raw_length:
                return
            if len(raw_length) < _LENGTH.size:
                return
            (size,) = _LENGTH.unpack(raw_length)
            payload = handle.read(size)
            if len(payload) < size:
                # An interrupted write. The complete records before it are still
                # good, which is the whole point of framing each one.
                return
            yield payload


def read_shard(
    path: "str | Path", *, decode: bool = True, keep_records: bool = False
) -> dict:
    """Read one shard back and report what it contains.

    With `decode`, every record is decoded and structurally validated, which is
    what makes a persisted shard evidence rather than a file of the right size.

    **Records are decoded, validated and dropped** unless `keep_records` is set.
    This is not an optimisation, it is a correctness property of the verifier:
    the first Phase 6B soak wrote 9.2 GiB of shards, and the original
    implementation of this function retained every decoded record of every shard
    at once, which drove a 48 GB machine 28 GiB into swap and wedged the
    verifying process for three quarters of an hour. A verifier must be able to
    check a corpus larger than memory; `keep_records` exists for tests and small
    inspections only.
    """
    path = Path(path)
    header = read_shard_header(path)
    compressed = bool(header.get("compressed", False))
    records: list = []
    record_count = 0
    payload_bytes = 0
    decoded_errors: list[str] = []
    for payload in iter_shard_payloads(path):
        payload_bytes += len(payload)
        if not decode:
            record_count += 1
            continue
        try:
            body = decompress(payload) if compressed else payload
            record = decode_game_record(body)
            validate_game_record(record)
            record_count += 1
            if keep_records:
                records.append(record)
        except Exception as error:  # noqa: BLE001 - a bad record is a result
            decoded_errors.append(f"{type(error).__name__}: {error}")
    return {
        "path": str(path),
        "header": header,
        "record_count": record_count,
        "payload_bytes": payload_bytes,
        "file_bytes": path.stat().st_size,
        "records": records,
        "decode_errors": decoded_errors,
    }


def verify_shard(
    path: "str | Path", *, decode: bool = True, keep_records: bool = False
) -> dict:
    """Check a shard against its manifest, and decode it if asked.

    Three independent things have to agree: the manifest's record count, the
    SHA-256 of the bytes on disk, and -- when `decode` is set -- the codec's
    willingness to rebuild every record it contains. Decoded records are dropped
    unless `keep_records` is set; see :func:`read_shard` for why.
    """
    path = Path(path)
    manifest_path = path.with_suffix(MANIFEST_SUFFIX)
    problems: list[str] = []
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        problems.append("no manifest; the shard was not closed cleanly")

    digest = _file_sha256(path)
    result = read_shard(path, decode=decode, keep_records=keep_records)
    if manifest is not None:
        if manifest["sha256"] != digest:
            problems.append("sha256 does not match the manifest")
        if manifest["records"] != result["record_count"]:
            problems.append(
                f"manifest claims {manifest['records']} records, "
                f"read {result['record_count']}"
            )
        if manifest["file_bytes"] != result["file_bytes"]:
            problems.append("file size does not match the manifest")
    problems.extend(result["decode_errors"])
    return {
        "path": str(path),
        "sha256": digest,
        "manifest": manifest,
        "record_count": result["record_count"],
        "file_bytes": result["file_bytes"],
        "decoded": decode,
        "problems": problems,
        "ok": not problems,
        "records": result["records"],
    }


def shard_paths(directory: "str | Path") -> list:
    """Every shard data file in a directory, in a stable order."""
    return sorted(Path(directory).glob(f"*{SHARD_SUFFIX}"))


def directory_summary(
    directory: "str | Path", *, decode: bool = False, progress=None
) -> dict:
    """Roll every shard in a directory up into one report.

    Shards are verified one at a time and their decoded records are never
    retained, so verifying a corpus costs one shard of memory regardless of how
    many shards the directory holds. `progress`, when given, is called with each
    shard's verification result as it completes -- a multi-gigabyte verification
    should be visible while it runs, not a silent stall.
    """
    paths = shard_paths(directory)
    shards = []
    game_ids: list[str] = []
    for path in paths:
        shard = verify_shard(path, decode=decode, keep_records=False)
        shard.pop("records", None)
        shards.append(shard)
        if progress is not None:
            progress(shard)
    for shard in shards:
        if shard["manifest"]:
            game_ids.extend(shard["manifest"]["game_ids"])
    return {
        "directory": str(directory),
        "shard_count": len(shards),
        "record_count": sum(shard["record_count"] for shard in shards),
        "file_bytes": sum(shard["file_bytes"] for shard in shards),
        "unclosed_shards": sum(1 for shard in shards if shard["manifest"] is None),
        "problem_shards": [shard["path"] for shard in shards if not shard["ok"]],
        "problems": [problem for shard in shards for problem in shard["problems"]],
        "game_ids": game_ids,
        "duplicate_game_ids": sorted(
            {name for name in game_ids if game_ids.count(name) > 1}
        )
        if len(game_ids) != len(set(game_ids))
        else [],
        "ok": all(shard["ok"] for shard in shards),
    }


__all__ = [
    "DEFAULT_SHARD_TARGET_BYTES",
    "MANIFEST_SUFFIX",
    "SHARD_FORMAT_VERSION",
    "SHARD_MAGIC",
    "SHARD_SUFFIX",
    "ShardError",
    "ShardStats",
    "ShardWriter",
    "directory_summary",
    "iter_shard_payloads",
    "read_shard",
    "read_shard_header",
    "shard_paths",
    "verify_shard",
]
