"""Phase 9 Agent 3: the crash-safe per-iteration rollout store.

Specification sources:

- `03_AGENT_3_SELFPLAY_COLLECTOR_AND_ROLLOUT_STORE.md` ("Rollout store",
  "Iteration state machine", "Crash/recovery")
- `phase9_contract.rollout_store_schema()`, frozen by Agent 1: the metadata
  field list, the commit rule, the seal rule and the sealed-rollout digest
- the accepted Phase 8 `warmstart_corpus_commit_v1` protocol, reused in shape

What is reused and what is new
------------------------------
The commit protocol is the accepted Phase 8 one, deliberately: append the
payload, then the metadata line, then the commit line, each flushed in that
order, and treat "has a commit line" as the definition of visible. Recovery is
the same truncation rule — cut every file back to what the last commit record
claims, because by the write order those are the only bytes that can be
uncommitted work. That protocol survived the Phase 8 corpus and there is no
reason to invent a second one.

What is new is Phase 9 identity. A file set is keyed by `(namespace,
iteration)` rather than by corpus split, the metadata sidecar carries the
frozen `phase9_rollout_store_v1` fields, and an iteration carries a persisted
state machine. `trajectory_v1` is reused with no field changing meaning; every
per-game fact it has no slot for lives in the sidecar.

Identity is never a path
------------------------
The sealed rollout digest is the SHA-256 over the sorted committed
`(game_id, payload_sha256, metadata_sha256)` triples. No directory name, no
shard name, no worker id and no mount point enters it. Copy a sealed
iteration to another volume and it is the same rollout — which is the whole
reason `phase9_storage` is a separate module from anything that computes an
identity here.
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

from .corpus_commit import metadata_line, payload_digest
from .phase9_contract import (
    PHASE9_ROLLOUT_STORE_VERSION,
    ROLLOUT_STATES,
    bucket_counts,
)
from .phase9_schedule import Phase9ScheduleError, iteration_game_ids
from .phase9_seed import PHASE9_ROLLOUT_VERSION, parse_phase9_game_id
from .phase9_storage import namespace_rollout_directory
from .serialization import DEFAULT_COMPRESSION_LEVEL, compress, decompress
from .shard_writer import (
    SHARD_FORMAT_VERSION,
    SHARD_MAGIC,
    SHARD_SUFFIX,
)
from .trajectory import (
    TRAJECTORY_VERSION,
    GameRecord,
    decode_game_record,
    encode_game_record,
    validate_game_record,
)

#: The commit protocol version. Agent 1 froze the name; this module is its
#: implementation, so a change to the journal fields, the write order or the
#: truncation rule is a new contract version, never a silent edit.
PHASE9_COMMIT_VERSION = PHASE9_ROLLOUT_STORE_VERSION

SHARDS_DIRECTORY = "shards"
METADATA_DIRECTORY = "metadata"
JOURNAL_DIRECTORY = "journal"
STATE_FILENAME = "state.json"
MANIFEST_FILENAME = "manifest.json"

METADATA_SUFFIX = ".meta.jsonl"
JOURNAL_SUFFIX = ".commit.jsonl"

#: Shard rollover size. Small enough that one iteration is a handful of files
#: and a rollover is actually exercised during a real collection, large enough
#: that the per-shard header is noise.
DEFAULT_ROLLOUT_SHARD_BYTES = 32 * 1024 * 1024

_LENGTH = struct.Struct("<I")
_HEADER = struct.Struct("<II")

#: Every field of a Phase 9 commit record. A line missing any of them is not a
#: commit, so a process killed mid-line contributes nothing.
COMMIT_FIELDS = (
    "commit_version",
    "phase9_game_id",
    "namespace",
    "iteration",
    "file_set",
    "shard_name",
    "record_index",
    "shard_bytes_after",
    "metadata_bytes_after",
    "payload_sha256",
    "metadata_sha256",
    "final_ply",
    "total_decisions",
    "learner_decision_count",
    "committed_unix",
)

#: The frozen `phase9_rollout_store_v1` metadata fields, in emission order.
#: Taken verbatim from `rollout_store_schema()["metadata_fields"]`, with the
#: schema's parenthetical notes resolved to real key names.
METADATA_FIELDS = (
    "game_id",
    "rollout_version",
    "store_version",
    "namespace",
    "iteration",
    "bucket",
    "ordinal",
    "learner_control",
    "learner_color",
    "red_policy_token",
    "blue_policy_token",
    "behavior_snapshot_id",
    "behavior_checkpoint_sha256",
    "opponent_kind",
    "opponent_identity",
    "opponent_checkpoint_sha256",
    "setup_root_seed",
    "red_policy_seed",
    "blue_policy_seed",
    "setup_provenance",
    "terminal_result",
    "terminal_reason",
    "total_decisions",
    "learner_decision_count",
    "final_ply",
    "trajectory_version",
    "population_version",
    "schedule_version",
    "contract_digest",
)

_FILE_SET_PATTERN = re.compile(r"^w(?P<worker>\d{2})$")
_SHARD_PATTERN = re.compile(r"^w(?P<worker>\d{2})_s(?P<shard>\d{4})$")

#: Named points a test may interrupt a write at. The writer calls the injected
#: hook at each; a hook that raises simulates a process dying there.
CRASH_STAGES = (
    "before_payload",
    "after_payload",
    "after_metadata",
    "before_commit_flush",
    "after_commit",
    "shard_rollover",
    "between_games",
)


class Phase9RolloutStoreError(RuntimeError):
    """The rollout store could not be written, read back or reconciled."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def iteration_directory(root, namespace: str, iteration: int) -> Path:
    """Where one iteration's bytes live. A convention, never an identity."""
    return namespace_rollout_directory(root, namespace, iteration)


def shards_directory(root, namespace: str, iteration: int) -> Path:
    return iteration_directory(root, namespace, iteration) / SHARDS_DIRECTORY


def metadata_directory(root, namespace: str, iteration: int) -> Path:
    return iteration_directory(root, namespace, iteration) / METADATA_DIRECTORY


def journal_directory(root, namespace: str, iteration: int) -> Path:
    return iteration_directory(root, namespace, iteration) / JOURNAL_DIRECTORY


def file_set_name(worker_id: int) -> str:
    return f"w{int(worker_id):02d}"


def shard_name(worker_id: int, shard_index: int) -> str:
    return f"{file_set_name(worker_id)}_s{int(shard_index):04d}"


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def metadata_digest(metadata: dict) -> str:
    """SHA-256 over the canonical metadata line, without the newline."""
    return hashlib.sha256(metadata_line(metadata)[:-1].encode()).hexdigest()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def build_rollout_metadata(
    scheduled,
    record: GameRecord,
    *,
    setup_provenance: dict,
    behavior_checkpoint_sha256: str,
    opponent_checkpoint_sha256: "str | None",
    learner_decision_count: int,
    population_version: str,
    schedule_version: str,
    contract_digest: str,
) -> dict:
    """The `phase9_rollout_store_v1` sidecar of one collected game.

    Everything `trajectory_v1` has no field for, and nothing it does: the
    payload stays the replay authority. `opponent_checkpoint_sha256` is the
    field that keeps a historical opponent's real weights addressable —
    `GameRecord.collection_checkpoint_id` names the iteration's current
    snapshot and cannot answer for the other side of the board.
    """
    metadata = {
        "game_id": scheduled.phase9_game_id,
        "rollout_version": PHASE9_ROLLOUT_VERSION,
        "store_version": PHASE9_COMMIT_VERSION,
        "namespace": scheduled.run_namespace,
        "iteration": int(scheduled.rl_iteration),
        "bucket": scheduled.bucket,
        "ordinal": int(scheduled.game_ordinal),
        "learner_control": scheduled.learner_control,
        "learner_color": scheduled.learner_color,
        "red_policy_token": scheduled.red_policy_identity,
        "blue_policy_token": scheduled.blue_policy_identity,
        "behavior_snapshot_id": scheduled.behavior_snapshot_identity,
        "behavior_checkpoint_sha256": str(behavior_checkpoint_sha256),
        "opponent_kind": scheduled.opponent_kind,
        "opponent_identity": scheduled.opponent_identity,
        "opponent_checkpoint_sha256": (
            None if opponent_checkpoint_sha256 is None else str(opponent_checkpoint_sha256)
        ),
        "setup_root_seed": int(scheduled.setup_root_seed),
        "red_policy_seed": scheduled.red_policy_seed,
        "blue_policy_seed": scheduled.blue_policy_seed,
        "setup_provenance": setup_provenance,
        "terminal_result": record.terminal_result,
        "terminal_reason": record.terminal_reason,
        "total_decisions": len(record.decisions),
        "learner_decision_count": int(learner_decision_count),
        "final_ply": int(record.final_ply),
        "trajectory_version": record.trajectory_version,
        "population_version": str(population_version),
        "schedule_version": str(schedule_version),
        "contract_digest": str(contract_digest),
    }
    missing = [field for field in METADATA_FIELDS if field not in metadata]
    if missing:  # pragma: no cover - the literal above covers every field
        raise Phase9RolloutStoreError(f"metadata is missing fields: {missing}")
    return {field: metadata[field] for field in METADATA_FIELDS}


def validate_rollout_metadata(metadata: dict, record: "GameRecord | None" = None) -> list:
    """Every disagreement between a sidecar, its schedule and its payload."""
    problems: list[str] = []
    missing = [field for field in METADATA_FIELDS if field not in metadata]
    if missing:
        return [f"metadata is missing fields: {missing}"]
    extra = sorted(set(metadata) - set(METADATA_FIELDS))
    if extra:
        problems.append(f"metadata carries unknown fields: {extra}")

    try:
        fields = parse_phase9_game_id(metadata["game_id"])
    except Exception as error:  # noqa: BLE001 - a malformed id is a finding
        return problems + [f"game id is not a Phase 9 rollout id: {error}"]

    for key, expected in (
        ("namespace", fields["namespace"]),
        ("iteration", fields["iteration"]),
        ("bucket", fields["bucket"]),
        ("ordinal", fields["ordinal"]),
        ("rollout_version", fields["rollout_version"]),
    ):
        if metadata[key] != expected:
            problems.append(
                f"metadata {key}={metadata[key]!r} disagrees with the game id ({expected!r})"
            )
    if metadata["store_version"] != PHASE9_COMMIT_VERSION:
        problems.append(f"store version {metadata['store_version']!r} is not this store's")
    if metadata["trajectory_version"] != TRAJECTORY_VERSION:
        problems.append(f"trajectory version {metadata['trajectory_version']!r}")
    if metadata["learner_control"] not in ("red", "blue", "both"):
        problems.append(f"learner_control {metadata['learner_control']!r}")
    if metadata["opponent_kind"] in ("historical_snapshot",) and not metadata[
        "opponent_checkpoint_sha256"
    ]:
        problems.append(
            "a historical matchup stored no opponent_checkpoint_sha256; the "
            "opponent's real weights would be unaddressable"
        )
    if metadata["opponent_kind"] in ("rule_policy", "stress_policy") and metadata[
        "opponent_checkpoint_sha256"
    ]:
        problems.append("a rule/stress opponent claims a checkpoint digest")
    if not metadata["behavior_checkpoint_sha256"]:
        problems.append("missing behavior_checkpoint_sha256")

    if record is not None:
        if record.game_id != metadata["game_id"]:
            problems.append("payload game id does not match the metadata game id")
        if record.terminal_result != metadata["terminal_result"]:
            problems.append("payload terminal result does not match the metadata")
        if record.terminal_reason != metadata["terminal_reason"]:
            problems.append("payload terminal reason does not match the metadata")
        if int(record.final_ply) != int(metadata["final_ply"]):
            problems.append("payload final ply does not match the metadata")
        if len(record.decisions) != int(metadata["total_decisions"]):
            problems.append("payload decision count does not match the metadata")
        if record.collection_checkpoint_id != metadata["behavior_checkpoint_sha256"]:
            problems.append(
                "payload collection_checkpoint_id is not the metadata behavior digest"
            )
        if int(record.root_seed) != int(metadata["setup_root_seed"]):
            problems.append("payload root seed does not match the metadata")
    return problems


# ---------------------------------------------------------------------------
# The append-only journal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase9CommitRecord:
    """One committed rollout game, as the journal stores it."""

    phase9_game_id: str
    namespace: str
    iteration: int
    file_set: str
    shard_name: str
    record_index: int
    shard_bytes_after: int
    metadata_bytes_after: int
    payload_sha256: str
    metadata_sha256: str
    final_ply: int
    total_decisions: int
    learner_decision_count: int
    committed_unix: float

    def to_dict(self) -> dict:
        return {
            "commit_version": PHASE9_COMMIT_VERSION,
            "phase9_game_id": self.phase9_game_id,
            "namespace": self.namespace,
            "iteration": self.iteration,
            "file_set": self.file_set,
            "shard_name": self.shard_name,
            "record_index": self.record_index,
            "shard_bytes_after": self.shard_bytes_after,
            "metadata_bytes_after": self.metadata_bytes_after,
            "payload_sha256": self.payload_sha256,
            "metadata_sha256": self.metadata_sha256,
            "final_ply": self.final_ply,
            "total_decisions": self.total_decisions,
            "learner_decision_count": self.learner_decision_count,
            "committed_unix": self.committed_unix,
        }

    @staticmethod
    def from_dict(payload: dict) -> "Phase9CommitRecord":
        return Phase9CommitRecord(
            phase9_game_id=str(payload["phase9_game_id"]),
            namespace=str(payload["namespace"]),
            iteration=int(payload["iteration"]),
            file_set=str(payload["file_set"]),
            shard_name=str(payload["shard_name"]),
            record_index=int(payload["record_index"]),
            shard_bytes_after=int(payload["shard_bytes_after"]),
            metadata_bytes_after=int(payload["metadata_bytes_after"]),
            payload_sha256=str(payload["payload_sha256"]),
            metadata_sha256=str(payload["metadata_sha256"]),
            final_ply=int(payload["final_ply"]),
            total_decisions=int(payload["total_decisions"]),
            learner_decision_count=int(payload["learner_decision_count"]),
            committed_unix=float(payload["committed_unix"]),
        )


def read_journal(path) -> tuple:
    """`(commits, valid_bytes)` for one journal file.

    Only newline-terminated, fully parseable lines carrying every commit field
    count. `valid_bytes` is where the journal would be truncated to remove an
    interrupted tail.
    """
    path = Path(path)
    if not path.exists():
        return ([], 0)
    raw = path.read_bytes()
    commits: list[Phase9CommitRecord] = []
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
        if not isinstance(payload, dict) or any(key not in payload for key in COMMIT_FIELDS):
            break
        if payload["commit_version"] != PHASE9_COMMIT_VERSION:
            raise Phase9RolloutStoreError(
                f"{path}: commit protocol {payload['commit_version']!r} is not "
                f"{PHASE9_COMMIT_VERSION!r}"
            )
        commits.append(Phase9CommitRecord.from_dict(payload))
        valid = offset
    return (commits, valid)


def read_metadata_file(path) -> tuple:
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
# Writing
# ---------------------------------------------------------------------------


class Phase9RolloutWriter:
    """The append-only writer of one worker's file set in one iteration.

    Not shared and not thread-safe: one instance belongs to one process, which
    is the only thing that ever appends to these three files. Worker ids
    partition the *files*, never the logical games — which is why a resume may
    use a different worker count and still converge to the same digest.
    """

    def __init__(
        self,
        root,
        *,
        namespace: str,
        iteration: int,
        worker_id: int,
        target_bytes: int = DEFAULT_ROLLOUT_SHARD_BYTES,
        compression_level: int = DEFAULT_COMPRESSION_LEVEL,
        fsync_on_commit: bool = False,
        crash_hook=None,
    ) -> None:
        self.root = Path(root)
        self.namespace = str(namespace)
        self.iteration = int(iteration)
        self.worker_id = int(worker_id)
        self.target_bytes = int(target_bytes)
        self.compression_level = int(compression_level)
        self.fsync_on_commit = bool(fsync_on_commit)
        self.crash_hook = crash_hook

        self.name = file_set_name(self.worker_id)
        self.shards_directory = shards_directory(self.root, self.namespace, self.iteration)
        self.metadata_path = (
            metadata_directory(self.root, self.namespace, self.iteration)
            / f"{self.name}{METADATA_SUFFIX}"
        )
        self.journal_path = (
            journal_directory(self.root, self.namespace, self.iteration)
            / f"{self.name}{JOURNAL_SUFFIX}"
        )
        for directory in (
            self.shards_directory,
            self.metadata_path.parent,
            self.journal_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        state = read_iteration_state(self.root, self.namespace, self.iteration)
        if state is not None and state["state"] != "COLLECTING":
            raise Phase9RolloutStoreError(
                f"{self.namespace} iteration {self.iteration} is {state['state']}; "
                "a sealed rollout is immutable and cannot be appended to"
            )

        # A resumed run opens a *fresh* worker id rather than appending to a
        # reconciled file set: appending would make the truncation rule depend
        # on how many times the run had crashed.
        if self.metadata_path.exists() or self.journal_path.exists():
            raise Phase9RolloutStoreError(
                f"file set {self.name} of {self.namespace} iteration {self.iteration} "
                "already exists; a resumed run must open a fresh worker id"
            )

        self.commits: list[Phase9CommitRecord] = []
        self.games_written = 0
        self.uncompressed_bytes = 0
        self.compressed_bytes = 0
        self.metadata_bytes = 0
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
        name = shard_name(self.worker_id, self._shard_index)
        self._shard_path = self.shards_directory / f"{name}{SHARD_SUFFIX}"
        header = {
            "run_id": f"{self.namespace}|it={self.iteration:03d}|{self.name}",
            "worker_id": self.worker_id,
            "shard_index": self._shard_index,
            "trajectory_version": TRAJECTORY_VERSION,
            "collection_policy_version": PHASE9_COMMIT_VERSION,
            "namespace": self.namespace,
            "iteration": self.iteration,
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
        """Roll between games only, so a shard closes on a committed boundary."""
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
        return shard_name(self.worker_id, self._shard_index)

    # -- the commit protocol ------------------------------------------------

    def write_game(self, record: GameRecord, metadata: dict) -> Phase9CommitRecord:
        """Persist and commit one game, in the frozen order.

        ```text
        1. encode + compress + decode-verify the payload
        2. validate the metadata against the record
        3. append the payload bytes, flush
        4. append the metadata line, flush
        5. append the commit line, flush
        ```

        An interruption anywhere before step 5 leaves the game uncommitted and
        therefore invisible; :func:`reconcile_iteration` removes the partial
        bytes on the next open.
        """
        if self.games_written:
            # The quiet moment between two games: everything on disk is
            # consistent and nothing is half-written. A process killed here
            # must lose nothing at all, which is a different claim from
            # surviving a kill mid-write.
            self._hook("between_games")
        if record.game_id != metadata["game_id"]:
            raise Phase9RolloutStoreError(
                f"record {record.game_id!r} does not match metadata {metadata['game_id']!r}"
            )
        if metadata["namespace"] != self.namespace or int(metadata["iteration"]) != self.iteration:
            raise Phase9RolloutStoreError(
                f"game {record.game_id} belongs to {metadata['namespace']} iteration "
                f"{metadata['iteration']}, not {self.namespace} iteration {self.iteration}"
            )

        started = time.perf_counter()
        raw = encode_game_record(record)
        self.encode_seconds += time.perf_counter() - started

        started = time.perf_counter()
        payload = compress(raw, self.compression_level)
        self.compress_seconds += time.perf_counter() - started

        started = time.perf_counter()
        problems = _verify_payload(payload, record)
        problems.extend(validate_rollout_metadata(metadata, record))
        self.verify_seconds += time.perf_counter() - started
        if problems:
            raise Phase9RolloutStoreError(
                f"game {record.game_id} failed pre-commit verification: {problems}"
            )

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

        commit = Phase9CommitRecord(
            phase9_game_id=record.game_id,
            namespace=self.namespace,
            iteration=self.iteration,
            file_set=self.name,
            shard_name=self.current_shard_name,
            record_index=record_index,
            shard_bytes_after=shard_bytes_after,
            metadata_bytes_after=metadata_bytes_after,
            payload_sha256=payload_digest(payload),
            metadata_sha256=metadata_digest(metadata),
            final_ply=int(record.final_ply),
            total_decisions=len(record.decisions),
            learner_decision_count=int(metadata["learner_decision_count"]),
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
        self.metadata_bytes += len(line.encode())
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
            "namespace": self.namespace,
            "iteration": self.iteration,
            "worker_id": self.worker_id,
            "games_written": self.games_written,
            "shards_opened": self._shard_index + 1,
            "uncompressed_bytes": self.uncompressed_bytes,
            "compressed_bytes": self.compressed_bytes,
            "metadata_bytes": self.metadata_bytes,
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
    if decoded.collection_checkpoint_id != record.collection_checkpoint_id:
        problems.append("decoded payload carries a different behavior checkpoint id")
    if len(decoded.decisions) != len(record.decisions):
        problems.append("decoded payload carries a different decision count")
    return problems


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def _file_sets(root, namespace: str, iteration: int) -> list:
    """Every worker file set that exists for one iteration."""
    names = set()
    for path in journal_directory(root, namespace, iteration).glob(f"*{JOURNAL_SUFFIX}"):
        names.add(path.name[: -len(JOURNAL_SUFFIX)])
    for path in metadata_directory(root, namespace, iteration).glob(f"*{METADATA_SUFFIX}"):
        names.add(path.name[: -len(METADATA_SUFFIX)])
    for path in shards_directory(root, namespace, iteration).glob(f"*{SHARD_SUFFIX}"):
        match = _SHARD_PATTERN.match(path.stem)
        if match is not None:
            names.add(file_set_name(int(match["worker"])))
    resolved = []
    for name in sorted(names):
        match = _FILE_SET_PATTERN.match(name)
        if match is None:
            raise Phase9RolloutStoreError(f"unrecognized rollout file set name: {name!r}")
        resolved.append((int(match["worker"]), name))
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


def reconcile_file_set(root, namespace: str, iteration: int, name: str) -> dict:
    """Cut one file set back to its last commit and report what was discarded.

    The only function that ever removes rollout bytes. It removes exactly the
    bytes no commit record claims, which by the write order can only be work
    interrupted before it became visible.
    """
    root = Path(root)
    journal_path = journal_directory(root, namespace, iteration) / f"{name}{JOURNAL_SUFFIX}"
    metadata_path = metadata_directory(root, namespace, iteration) / f"{name}{METADATA_SUFFIX}"
    shard_directory = shards_directory(root, namespace, iteration)

    commits, journal_valid_bytes = read_journal(journal_path)
    report = {
        "file_set": name,
        "namespace": namespace,
        "iteration": int(iteration),
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
        raise Phase9RolloutStoreError(f"unrecognized rollout file set name: {name!r}")
    worker = int(match["worker"])
    last_shard_index = -1
    if last is not None:
        shard_match = _SHARD_PATTERN.match(last.shard_name)
        if shard_match is None:
            raise Phase9RolloutStoreError(
                f"commit for {last.phase9_game_id} names shard {last.shard_name!r}, "
                "which is not a rollout shard name"
            )
        last_shard_index = int(shard_match["shard"])
        shard_path = shard_directory / f"{last.shard_name}{SHARD_SUFFIX}"
        if not shard_path.exists():
            raise Phase9RolloutStoreError(
                f"committed game {last.phase9_game_id} names missing shard {shard_path}"
            )
        report["shard_bytes_discarded"] += _truncate(shard_path, last.shard_bytes_after)

    for path in sorted(shard_directory.glob(f"{name}_s*{SHARD_SUFFIX}")):
        shard_match = _SHARD_PATTERN.match(path.stem)
        if shard_match is None:  # pragma: no cover - the glob already constrains it
            continue
        if int(shard_match["worker"]) != worker:  # pragma: no cover - ditto
            continue
        if int(shard_match["shard"]) <= last_shard_index:
            continue
        # Every record in this shard was written after the last commit, so the
        # whole file is uncommitted work.
        report["shard_bytes_discarded"] += _file_size(path)
        report["shards_removed"].append(path.name)
        path.unlink()

    return report


def reconcile_iteration(root, namespace: str, iteration: int) -> dict:
    """Reconcile every file set of one iteration before collection resumes.

    After this call the iteration directory holds committed games and nothing
    else, which is the precondition every later audit assumes.
    """
    root = Path(root)
    reports = []
    committed: dict[str, Phase9CommitRecord] = {}
    duplicates: list[str] = []
    for _worker, name in _file_sets(root, namespace, iteration):
        report = reconcile_file_set(root, namespace, iteration, name)
        reports.append(report)
        commits, _ = read_journal(
            journal_directory(root, namespace, iteration) / f"{name}{JOURNAL_SUFFIX}"
        )
        for commit in commits:
            if commit.phase9_game_id in committed:
                duplicates.append(commit.phase9_game_id)
                continue
            committed[commit.phase9_game_id] = commit
    return {
        "namespace": namespace,
        "iteration": int(iteration),
        "file_sets": reports,
        "committed": committed,
        "committed_game_ids": tuple(sorted(committed)),
        "duplicate_game_ids": sorted(set(duplicates)),
        "bytes_discarded": sum(
            report["journal_bytes_discarded"]
            + report["metadata_bytes_discarded"]
            + report["shard_bytes_discarded"]
            for report in reports
        ),
    }


def next_worker_id(root, namespace: str, iteration: int) -> int:
    """The lowest worker id no file set has claimed in this iteration."""
    used = {worker for worker, _name in _file_sets(root, namespace, iteration)}
    candidate = 0
    while candidate in used:
        candidate += 1
    if candidate > 99:
        raise Phase9RolloutStoreError(
            f"{namespace} iteration {iteration} has exhausted the two-digit worker "
            "id space; the file-set naming would collide"
        )
    return candidate


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _shard_frame_offsets(path: Path) -> list:
    """Byte offset and length of every frame in one shard, in write order."""
    with path.open("rb") as handle:
        magic = handle.read(len(SHARD_MAGIC))
        if magic != SHARD_MAGIC:
            raise Phase9RolloutStoreError(f"{path}: not a stgshard file")
        version, header_length = _HEADER.unpack(handle.read(_HEADER.size))
        if version != SHARD_FORMAT_VERSION:
            raise Phase9RolloutStoreError(f"{path}: shard format version {version}")
        handle.read(header_length)
        offsets = []
        while True:
            raw = handle.read(_LENGTH.size)
            if len(raw) < _LENGTH.size:
                break
            (length,) = _LENGTH.unpack(raw)
            offsets.append((handle.tell(), length))
            handle.seek(length, os.SEEK_CUR)
    return offsets


class Phase9RolloutReader:
    """Random-access reader over one iteration's committed games.

    Agent 4 gets this: a sealed rollout it can walk in any order, reconstruct
    from, and audit without trusting a single number the collector reported.
    """

    def __init__(self, root, namespace: str, iteration: int) -> None:
        self.root = Path(root)
        self.namespace = str(namespace)
        self.iteration = int(iteration)
        self.commits: dict[str, Phase9CommitRecord] = {}
        self.metadata: dict[str, dict] = {}
        for _worker, name in _file_sets(self.root, self.namespace, self.iteration):
            commits, _ = read_journal(
                journal_directory(self.root, self.namespace, self.iteration)
                / f"{name}{JOURNAL_SUFFIX}"
            )
            for commit in commits:
                self.commits[commit.phase9_game_id] = commit
            records, _ = read_metadata_file(
                metadata_directory(self.root, self.namespace, self.iteration)
                / f"{name}{METADATA_SUFFIX}"
            )
            for record in records:
                self.metadata[str(record["game_id"])] = record
        self._offsets: dict[str, list] = {}

    @property
    def game_ids(self) -> tuple:
        return tuple(sorted(self.commits))

    def __len__(self) -> int:
        return len(self.commits)

    def _frame(self, commit: Phase9CommitRecord) -> bytes:
        path = (
            shards_directory(self.root, self.namespace, self.iteration)
            / f"{commit.shard_name}{SHARD_SUFFIX}"
        )
        if commit.shard_name not in self._offsets:
            self._offsets[commit.shard_name] = _shard_frame_offsets(path)
        offsets = self._offsets[commit.shard_name]
        if not 0 <= commit.record_index < len(offsets):
            raise Phase9RolloutStoreError(
                f"{commit.phase9_game_id}: record index {commit.record_index} is "
                f"outside shard {commit.shard_name} ({len(offsets)} frames)"
            )
        offset, length = offsets[commit.record_index]
        with path.open("rb") as handle:
            handle.seek(offset)
            payload = handle.read(length)
        if len(payload) != length:
            raise Phase9RolloutStoreError(f"{commit.phase9_game_id}: truncated frame")
        return payload

    def read_payload(self, game_id: str) -> bytes:
        commit = self.commits.get(game_id)
        if commit is None:
            raise Phase9RolloutStoreError(
                f"{game_id} is not committed in {self.namespace} iteration {self.iteration}"
            )
        payload = self._frame(commit)
        digest = payload_digest(payload)
        if digest != commit.payload_sha256:
            raise Phase9RolloutStoreError(
                f"{game_id}: stored payload digest {digest} != committed "
                f"{commit.payload_sha256}"
            )
        return payload

    def read_game(self, game_id: str) -> tuple:
        """`(record, metadata)` for one committed game, digest-checked."""
        record = decode_game_record(decompress(self.read_payload(game_id)))
        metadata = self.metadata.get(game_id)
        if metadata is None:
            raise Phase9RolloutStoreError(f"{game_id} has a commit but no metadata record")
        digest = metadata_digest(metadata)
        commit = self.commits[game_id]
        if digest != commit.metadata_sha256:
            raise Phase9RolloutStoreError(
                f"{game_id}: metadata digest {digest} != committed {commit.metadata_sha256}"
            )
        return record, metadata

    def iter_games(self):
        for game_id in self.game_ids:
            yield self.read_game(game_id)

    def orphans(self) -> dict:
        """Records on either side of the commit boundary with no counterpart."""
        return {
            "metadata_without_commit": sorted(set(self.metadata) - set(self.commits)),
            "commit_without_metadata": sorted(set(self.commits) - set(self.metadata)),
        }


# ---------------------------------------------------------------------------
# Digest and the iteration state machine
# ---------------------------------------------------------------------------


def sealed_rollout_digest(commits) -> str:
    """SHA-256 over the sorted committed `(id, payload, metadata)` triples.

    The frozen identity of a sealed rollout. Location, shard names, worker
    ids, record order and wall-clock times appear nowhere, so the same bytes
    on another volume, written by another worker count, are the same rollout.
    """
    triples = sorted(
        (commit.phase9_game_id, commit.payload_sha256, commit.metadata_sha256)
        for commit in (commits.values() if isinstance(commits, dict) else commits)
    )
    hasher = hashlib.sha256()
    hasher.update(PHASE9_COMMIT_VERSION.encode())
    for game_id, payload, metadata in triples:
        hasher.update(f"{game_id}|{payload}|{metadata}\n".encode())
    return hasher.hexdigest()


def state_path(root, namespace: str, iteration: int) -> Path:
    return iteration_directory(root, namespace, iteration) / STATE_FILENAME


def read_iteration_state(root, namespace: str, iteration: int) -> "dict | None":
    path = state_path(root, namespace, iteration)
    if not path.exists():
        return None
    return json.loads(path.read_text())


#: State-document keys a later transition must not silently drop. The device
#: and batch shape are the conditions the committed bytes were produced under;
#: a reader of a SEALED rollout needs them as much as a resuming collector
#: does, so a transition carries them forward instead of resetting them.
STATE_CARRY_FORWARD_KEYS = (
    "behavior_snapshot_id",
    "behavior_checkpoint_sha256",
    "inference_device",
    "inference_batch_shape",
    "collector_version",
)


def write_iteration_state(
    root, namespace: str, iteration: int, state: str, **extra
) -> dict:
    """Persist one state-machine transition, appending to its own history.

    The lifecycle is only meaningful if it is durable across a crash, so it
    lives on disk beside the bytes rather than in the collector's memory. The
    document accumulates: a transition may add facts and may override them,
    but it cannot lose the ones in :data:`STATE_CARRY_FORWARD_KEYS`.
    """
    if state not in ROLLOUT_STATES:
        raise Phase9RolloutStoreError(
            f"unknown rollout state {state!r}; expected one of {list(ROLLOUT_STATES)}"
        )
    path = state_path(root, namespace, iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = read_iteration_state(root, namespace, iteration)
    history = list(previous["history"]) if previous else []
    if previous is not None and previous["state"] == "SEALED" and state == "COLLECTING":
        raise Phase9RolloutStoreError(
            f"{namespace} iteration {iteration} is SEALED; sealed rollouts are immutable"
        )
    history.append({"state": state, "unix": time.time()})
    carried = {
        key: previous[key]
        for key in STATE_CARRY_FORWARD_KEYS
        if previous is not None and previous.get(key) is not None and key not in extra
    }
    document = {
        "store_version": PHASE9_COMMIT_VERSION,
        "namespace": str(namespace),
        "iteration": int(iteration),
        "state": state,
        "states": list(ROLLOUT_STATES),
        "history": history,
        **carried,
        **extra,
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def seal_iteration(
    root,
    namespace: str,
    iteration: int,
    *,
    expected_behavior_checkpoint: str,
    manifest_extra: "dict | None" = None,
) -> dict:
    """Run every seal precondition, then move `COLLECTING -> SEALED`.

    A rollout seals only when the exact scheduled game set is present with no
    duplicate, no unscheduled game and no orphan record; every payload decodes
    and validates; every metadata sidecar agrees with its payload; and one
    behavior identity covers the whole iteration. Anything else leaves the
    iteration COLLECTING and returns the reasons.
    """
    root = Path(root)
    reconciled = reconcile_iteration(root, namespace, iteration)
    reader = Phase9RolloutReader(root, namespace, iteration)
    scheduled = set(iteration_game_ids(namespace, iteration))
    committed = set(reader.commits)

    problems: list[str] = []
    if reconciled["duplicate_game_ids"]:
        problems.append(f"duplicate committed ids: {reconciled['duplicate_game_ids'][:5]}")
    missing = sorted(scheduled - committed)
    if missing:
        problems.append(f"{len(missing)} scheduled games are not committed (e.g. {missing[:3]})")
    unscheduled = sorted(committed - scheduled)
    if unscheduled:
        problems.append(f"{len(unscheduled)} committed games are not scheduled (e.g. {unscheduled[:3]})")
    orphans = reader.orphans()
    if orphans["metadata_without_commit"] or orphans["commit_without_metadata"]:
        problems.append(f"orphan records: {orphans}")

    behavior_identities = set()
    behavior_digests = set()
    decoded = 0
    learner_decisions = 0
    total_decisions = 0
    for game_id in reader.game_ids:
        try:
            record, metadata = reader.read_game(game_id)
        except Exception as error:  # noqa: BLE001 - an unreadable game blocks sealing
            problems.append(f"{game_id}: {type(error).__name__}: {error}")
            continue
        decoded += 1
        record_problems = validate_game_record(record)
        record_problems.extend(validate_rollout_metadata(metadata, record))
        if record_problems:
            problems.append(f"{game_id}: {record_problems[:3]}")
        behavior_identities.add(metadata["behavior_snapshot_id"])
        behavior_digests.add(metadata["behavior_checkpoint_sha256"])
        learner_decisions += int(metadata["learner_decision_count"])
        total_decisions += int(metadata["total_decisions"])

    if len(behavior_identities) > 1:
        problems.append(f"iteration mixes behavior identities: {sorted(behavior_identities)}")
    if len(behavior_digests) > 1:
        problems.append(f"iteration mixes behavior checkpoints: {sorted(behavior_digests)}")
    if behavior_digests and expected_behavior_checkpoint not in behavior_digests:
        problems.append(
            f"iteration was collected under {sorted(behavior_digests)}, not the "
            f"expected {expected_behavior_checkpoint}"
        )

    digest = sealed_rollout_digest(reader.commits)
    summary = {
        "namespace": namespace,
        "iteration": int(iteration),
        "scheduled_games": len(scheduled),
        "committed_games": len(committed),
        "decoded_games": decoded,
        "duplicate_game_ids": len(reconciled["duplicate_game_ids"]),
        "unscheduled_games": len(unscheduled),
        "missing_games": len(missing),
        "orphan_records": len(orphans["metadata_without_commit"])
        + len(orphans["commit_without_metadata"]),
        "behavior_snapshot_identities": sorted(behavior_identities),
        "behavior_checkpoint_digests": sorted(behavior_digests),
        "total_decisions": total_decisions,
        "learner_decision_count": learner_decisions,
        "sealed_rollout_digest": digest,
        "sealed": not problems,
        "problems": problems,
    }
    if problems:
        write_iteration_state(
            root, namespace, iteration, "COLLECTING", seal_attempt=summary
        )
        return summary

    write_iteration_state(
        root,
        namespace,
        iteration,
        "SEALED",
        sealed_rollout_digest=digest,
        committed_games=len(committed),
        behavior_snapshot_id=sorted(behavior_identities)[0] if behavior_identities else None,
        behavior_checkpoint_sha256=expected_behavior_checkpoint,
    )
    manifest = dict(manifest_extra or {})
    manifest.update(summary)
    (iteration_directory(root, namespace, iteration) / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    return summary


def pending_game_ids(root, namespace: str, iteration: int) -> tuple:
    """`scheduled - committed`, in schedule order.

    The whole resume rule: a committed game is never regenerated, a missing one
    always is, and nothing about how the previous attempt was partitioned
    enters the answer.
    """
    reader = Phase9RolloutReader(Path(root), namespace, iteration)
    committed = set(reader.commits)
    scheduled = iteration_game_ids(namespace, iteration)
    foreign = committed - set(scheduled)
    if foreign:
        raise Phase9ScheduleError(
            f"{namespace} iteration {iteration} has {len(foreign)} committed games "
            f"that it never scheduled (e.g. {sorted(foreign)[:3]})"
        )
    return tuple(game_id for game_id in scheduled if game_id not in committed)


def iteration_storage_summary(root, namespace: str, iteration: int) -> dict:
    """Measured on-disk cost of one iteration. Diagnostic, never identity."""
    directory = iteration_directory(root, namespace, iteration)
    shard_bytes = sum(
        _file_size(path) for path in shards_directory(root, namespace, iteration).glob(f"*{SHARD_SUFFIX}")
    )
    metadata_bytes = sum(
        _file_size(path)
        for path in metadata_directory(root, namespace, iteration).glob(f"*{METADATA_SUFFIX}")
    )
    journal_bytes = sum(
        _file_size(path)
        for path in journal_directory(root, namespace, iteration).glob(f"*{JOURNAL_SUFFIX}")
    )
    total = shard_bytes + metadata_bytes + journal_bytes
    reader = Phase9RolloutReader(root, namespace, iteration)
    games = len(reader)
    decisions = sum(commit.total_decisions for commit in reader.commits.values())
    return {
        "directory": str(directory),
        "shard_bytes": shard_bytes,
        "metadata_bytes": metadata_bytes,
        "journal_bytes": journal_bytes,
        "total_bytes": total,
        "committed_games": games,
        "total_decisions": decisions,
        "bytes_per_game": total / games if games else 0.0,
        "bytes_per_decision": total / decisions if decisions else 0.0,
    }


def expected_iteration_games(namespace: str) -> int:
    return sum(bucket_counts(namespace).values())


__all__ = [
    "COMMIT_FIELDS",
    "CRASH_STAGES",
    "DEFAULT_ROLLOUT_SHARD_BYTES",
    "METADATA_FIELDS",
    "PHASE9_COMMIT_VERSION",
    "Phase9CommitRecord",
    "Phase9RolloutReader",
    "Phase9RolloutStoreError",
    "Phase9RolloutWriter",
    "build_rollout_metadata",
    "expected_iteration_games",
    "iteration_directory",
    "iteration_storage_summary",
    "metadata_digest",
    "next_worker_id",
    "pending_game_ids",
    "read_iteration_state",
    "read_journal",
    "read_metadata_file",
    "reconcile_file_set",
    "reconcile_iteration",
    "seal_iteration",
    "sealed_rollout_digest",
    "STATE_CARRY_FORWARD_KEYS",
    "state_path",
    "validate_rollout_metadata",
    "write_iteration_state",
]
