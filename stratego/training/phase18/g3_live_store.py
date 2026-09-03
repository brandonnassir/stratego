"""Phase 18 Stage 6B: the per-period live trajectory store of one lineage.

Every collector game a lineage completes is committed here as a
`trajectory_v1` record plus a metadata sidecar, one append-only file set per
period, so the C1 live stream can be rebuilt from the frozen Phase 8 example
builder (`warmstart_examples.examples_for_game`) exactly as the canonical
corpus is, and so a restarted lineage reads back precisely the games the
bundle knew about.

```text
<root>/period_0007.records         G3LV magic, JSON header, length-prefixed
                                    zlib frames of encoded GameRecords
<root>/period_0007.meta.jsonl      one metadata line per game, frame order
<root>/period_0007.journal.jsonl   one commit line per game, written AFTER
                                    the frame and the metadata line are flushed
<root>/period_0007.done.json       written by close(): counts and the sha256
                                    of the three files; a period without it
                                    is not readable
```

The commit journal is the index, as in the accepted corpus: a game the
journal does not name does not exist. The journal also carries the game's
selected decision indices, so the live example universe is enumerated without
decoding a single payload.

Two digests, two purposes (the period-1 gate correction)
--------------------------------------------------------
`commit_digest` (the *raw* digest) hashes every commit's game id, trajectory
digest and metadata-line digest. The metadata line carries `lineage`, so the
raw digests of the candidate and the control can NEVER be equal even when
both lineages committed byte-identical games; the raw digest is a file-
integrity audit of one lineage's own store (resume, bundle, verify_period),
not a cross-lineage comparison. The cross-lineage fairness comparison is the
*semantic* one below (`compare_live_periods`): equal commit counts and
order, equal game ids, equal trajectory digests, equal selected-decision
lists and equal metadata after removing exactly the `lineage` field, which
must read "candidate" in the candidate store and "control" in the control
store. No other metadata field is ever ignored.

Why not `corpus_commit.CorpusWriter`
------------------------------------
Its pre-commit verification requires a *synthetic* corpus game id whose seeds
re-derive from the Phase 8 master seed. A live game's identity is the pilot's
`(run, seed, period, slot, draw)`, so the accepted writer would refuse it by
design. The frame layout, the commit-after-flush order and the journal-as-
index rule are transcribed from it; the example builder is imported unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from ..trajectory import (
    GameRecord,
    decode_game_record_compressed,
    encode_game_record_compressed,
    validate_game_record,
)
from ..warmstart_examples import examples_for_game
from ..warmstart_seed import MAX_DECISIONS_PER_GAME, decision_bin_bounds
from .g3_contract import G3_LIVE_STORE_VERSION, Phase18G3Error
from .setup_contract import file_sha256, stream_seed

_MAGIC = b"G3LV"
_HEADER = struct.Struct("<I")
_LENGTH = struct.Struct("<I")

RECORDS_SUFFIX = ".records"
METADATA_SUFFIX = ".meta.jsonl"
JOURNAL_SUFFIX = ".journal.jsonl"
DONE_SUFFIX = ".done.json"


def period_name(period: int) -> str:
    if int(period) < 1:
        raise Phase18G3Error("periods are one-based")
    return f"period_{int(period):04d}"


def live_selected_decision_indices(namespace: str, game_id: str, total_decisions: int) -> tuple:
    """`warmstart_decision_sampler_v1`'s rule with the pilot's own bin seeds.

    ```text
    T <= 0      ()
    T <= 64     every decision
    T >  64     one index per bin: lo + (seed(namespace, game, bin) % (hi - lo))
    ```

    The accepted sampler seeds each bin from the synthetic game id through the
    Phase 8 master seed; a live game has no such id, so its bins draw from the
    pilot namespace through `derive_stream_seed`. Bins, widths and the modulo
    draw are the accepted ones (`decision_bin_bounds` is imported unchanged).
    """
    total = int(total_decisions)
    if total < 0:
        raise Phase18G3Error(f"total_decisions must be >= 0, got {total}")
    if total == 0:
        return ()
    if total <= MAX_DECISIONS_PER_GAME:
        return tuple(range(total))
    selected = []
    for bin_index, (low, high) in enumerate(decision_bin_bounds(total)):
        draw = stream_seed(namespace, "live_decision_sampler", game_id, bin_index) % (high - low)
        selected.append(low + draw)
    return tuple(selected)


# ---------------------------------------------------------------------------
# Writing one period
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveCommit:
    """One committed live game, as the journal stores it."""

    game_id: str
    period: int
    record_index: int
    offset: int
    size: int
    trajectory_sha256: str
    metadata_sha256: str
    final_ply: int
    total_decisions: int
    selected_decisions: tuple

    def to_dict(self) -> dict:
        return {
            "store_version": G3_LIVE_STORE_VERSION,
            "game_id": self.game_id,
            "period": int(self.period),
            "record_index": int(self.record_index),
            "offset": int(self.offset),
            "size": int(self.size),
            "trajectory_sha256": self.trajectory_sha256,
            "metadata_sha256": self.metadata_sha256,
            "final_ply": int(self.final_ply),
            "total_decisions": int(self.total_decisions),
            "selected_decisions": [int(index) for index in self.selected_decisions],
        }

    @staticmethod
    def from_dict(payload: dict) -> "LiveCommit":
        if payload.get("store_version") != G3_LIVE_STORE_VERSION:
            raise Phase18G3Error(f"live commit under store version {payload.get('store_version')!r}")
        return LiveCommit(
            game_id=str(payload["game_id"]),
            period=int(payload["period"]),
            record_index=int(payload["record_index"]),
            offset=int(payload["offset"]),
            size=int(payload["size"]),
            trajectory_sha256=str(payload["trajectory_sha256"]),
            metadata_sha256=str(payload["metadata_sha256"]),
            final_ply=int(payload["final_ply"]),
            total_decisions=int(payload["total_decisions"]),
            selected_decisions=tuple(int(index) for index in payload["selected_decisions"]),
        )


class LivePeriodWriter:
    """The append-only writer of one period's file set. One process, one period."""

    def __init__(self, root, *, period: int, namespace: str, lineage: str, run_id: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.period = int(period)
        self.namespace = namespace
        self.name = period_name(period)
        self.records_path = self.root / f"{self.name}{RECORDS_SUFFIX}"
        self.metadata_path = self.root / f"{self.name}{METADATA_SUFFIX}"
        self.journal_path = self.root / f"{self.name}{JOURNAL_SUFFIX}"
        self.done_path = self.root / f"{self.name}{DONE_SUFFIX}"
        for path in (self.records_path, self.metadata_path, self.journal_path, self.done_path):
            if path.exists():
                raise Phase18G3Error(
                    f"{path} already exists; a period's live file set is never appended to or "
                    "overwritten (discard it explicitly on a resume)"
                )
        header = {
            "store_version": G3_LIVE_STORE_VERSION,
            "period": self.period,
            "namespace": namespace,
            "lineage": lineage,
            "run_id": run_id,
            "opened_unix": time.time(),
        }
        blob = json.dumps(header, sort_keys=True).encode()
        self._records = self.records_path.open("wb")
        self._records.write(_MAGIC + _HEADER.pack(len(blob)) + blob)
        self._records.flush()
        self._metadata = self.metadata_path.open("ab")
        self._journal = self.journal_path.open("ab")
        self.commits: list = []
        self.compressed_bytes = 0
        self.closed = False

    def write(self, record: GameRecord, metadata: dict) -> LiveCommit:
        """Commit one finished game: frame, metadata line, then the journal line."""
        if self.closed:
            raise Phase18G3Error("the live period writer is closed")
        if metadata.get("synthetic_game_id") != record.game_id:
            raise Phase18G3Error(
                f"metadata names {metadata.get('synthetic_game_id')!r}, record is {record.game_id!r}"
            )
        if metadata.get("corpus_split") != "train":
            raise Phase18G3Error("live examples enter only the C1 training split (common contract 9.2)")
        problems = validate_game_record(record)
        if problems:
            raise Phase18G3Error(f"{record.game_id}: the sealed trajectory is invalid: {problems}")
        payload = encode_game_record_compressed(record)
        decoded = decode_game_record_compressed(payload)
        if decoded.game_id != record.game_id or decoded.actions != record.actions:
            raise Phase18G3Error(f"{record.game_id}: the stored payload does not decode back to the game")
        line = json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"

        offset = self._records.tell()
        self._records.write(_LENGTH.pack(len(payload)) + payload)
        self._records.flush()
        self._metadata.write(line.encode())
        self._metadata.flush()
        commit = LiveCommit(
            game_id=record.game_id,
            period=self.period,
            record_index=len(self.commits),
            offset=offset + _LENGTH.size,
            size=len(payload),
            trajectory_sha256=hashlib.sha256(payload).hexdigest(),
            metadata_sha256=hashlib.sha256(line.encode()).hexdigest(),
            final_ply=int(record.final_ply),
            total_decisions=len(record.decisions),
            selected_decisions=live_selected_decision_indices(self.namespace, record.game_id, len(record.decisions)),
        )
        self._journal.write((json.dumps(commit.to_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode())
        self._journal.flush()
        self.commits.append(commit)
        self.compressed_bytes += len(payload)
        return commit

    def close(self) -> dict:
        """Flush, fsync and finalise the period. Idempotent."""
        if self.closed:
            return json.loads(self.done_path.read_text())
        for handle in (self._records, self._metadata, self._journal):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        summary = {
            "store_version": G3_LIVE_STORE_VERSION,
            "period": self.period,
            "games": len(self.commits),
            "selected_examples": int(sum(len(commit.selected_decisions) for commit in self.commits)),
            "compressed_bytes": int(self.compressed_bytes),
            "files": {
                path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
                for path in (self.records_path, self.metadata_path, self.journal_path)
            },
            "commit_digest": commits_digest(self.commits),
            "closed_unix": time.time(),
        }
        temporary = self.done_path.with_suffix(".partial")
        temporary.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
        os.replace(temporary, self.done_path)
        self.closed = True
        return summary


def commits_digest(commits) -> str:
    """Order-dependent digest over the committed game ids and payload digests."""
    hasher = hashlib.sha256()
    for commit in commits:
        hasher.update(f"{commit.game_id}|{commit.trajectory_sha256}|{commit.metadata_sha256}\n".encode())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Reading finalised periods
# ---------------------------------------------------------------------------


def available_periods(root) -> tuple:
    """Every finalised period under `root`, ascending."""
    root = Path(root)
    if not root.exists():
        return ()
    periods = []
    for path in root.glob(f"period_*{DONE_SUFFIX}"):
        periods.append(int(path.name[len("period_") : len("period_") + 4]))
    return tuple(sorted(periods))


def discard_periods_after(root, period: int, *, destination=None) -> list:
    """Move every period file set newer than `period` out of the way.

    A restarted lineage re-plays the period after its bundle, so any live
    files a crashed process wrote for later periods must not be readable.
    Nothing is deleted: with `destination` the files are moved there under
    their own names (the resume archive); without it they are renamed in place
    with an `.orphaned` marker. The moves are returned for the resume record.
    """
    root = Path(root)
    renamed = []
    if not root.exists():
        return renamed
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if destination is not None:
        destination = Path(destination)
    for path in sorted(root.glob("period_*")):
        if ".orphaned" in path.name:
            continue
        stem = path.name[len("period_") : len("period_") + 4]
        if not stem.isdigit() or int(stem) <= int(period):
            continue
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
            target = destination / path.name
            if target.exists():
                raise Phase18G3Error(f"{target} already exists in the archive")
        else:
            target = path.with_name(f"{path.name}.orphaned-{stamp}")
        os.replace(path, target)
        renamed.append({"from": str(path), "to": str(target)})
    return renamed


class LiveRecordReader:
    """Random access to the committed live games of finalised periods."""

    def __init__(self, root, *, record_cache_size: int = 256) -> None:
        self.root = Path(root)
        self.record_cache_size = int(record_cache_size)
        self._commits: dict = {}
        self._metadata: dict = {}
        self._period_of: dict = {}
        self._records: OrderedDict = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    # -- periods -----------------------------------------------------------

    def periods(self) -> tuple:
        return available_periods(self.root)

    def summary(self, period: int) -> dict:
        path = self.root / f"{period_name(period)}{DONE_SUFFIX}"
        if not path.exists():
            raise Phase18G3Error(f"live period {period} is not finalised under {self.root}")
        return json.loads(path.read_text())

    def commits(self, period: int) -> "OrderedDict[str, LiveCommit]":
        """The journal of one finalised period, in commit order."""
        period = int(period)
        loaded = self._commits.get(period)
        if loaded is None:
            self.summary(period)
            loaded = OrderedDict()
            journal = self.root / f"{period_name(period)}{JOURNAL_SUFFIX}"
            with journal.open("rb") as handle:
                for raw in handle:
                    raw = raw.strip()
                    if not raw:
                        continue
                    commit = LiveCommit.from_dict(json.loads(raw))
                    if commit.game_id in loaded:
                        raise Phase18G3Error(f"duplicate live commit {commit.game_id!r} in period {period}")
                    loaded[commit.game_id] = commit
                    self._period_of[commit.game_id] = period
            self._commits[period] = loaded
        return loaded

    def period_of(self, game_id: str) -> int:
        """The finalised period a committed game belongs to.

        Unknown ids trigger one rescan of the finalised periods, which is how a
        loader worker learns about periods the parent finalised after the
        worker started.
        """
        period = self._period_of.get(game_id)
        if period is None:
            for candidate in self.periods():
                if candidate not in self._commits:
                    self.commits(candidate)
            period = self._period_of.get(game_id)
        if period is None:
            raise Phase18G3Error(f"live game {game_id!r} is not committed in any finalised period")
        return int(period)

    def verify_period(self, period: int) -> dict:
        """Re-hash the three files against the finalisation summary."""
        summary = self.summary(period)
        problems = []
        for name, entry in summary["files"].items():
            path = self.root / name
            if not path.exists():
                problems.append(f"{name} is missing")
                continue
            observed = file_sha256(path)
            if observed != entry["sha256"]:
                problems.append(f"{name} digests to {observed}, recorded {entry['sha256']}")
        commits = list(self.commits(period).values())
        if commits_digest(commits) != summary["commit_digest"]:
            problems.append("the journal does not reproduce the recorded commit digest")
        if len(commits) != int(summary["games"]):
            problems.append(f"{len(commits)} commits for {summary['games']} recorded games")
        return {"period": int(period), "problems": problems, "verified": not problems}

    # -- games -------------------------------------------------------------

    def metadata(self, period: int, game_id: str) -> dict:
        period = int(period)
        loaded = self._metadata.get(period)
        if loaded is None:
            commits = self.commits(period)
            loaded = {}
            path = self.root / f"{period_name(period)}{METADATA_SUFFIX}"
            with path.open("rb") as handle:
                for raw in handle:
                    raw = raw.strip()
                    if not raw:
                        continue
                    entry = json.loads(raw)
                    identifier = str(entry["synthetic_game_id"])
                    if identifier in commits:
                        loaded[identifier] = entry
            self._metadata[period] = loaded
        try:
            return loaded[game_id]
        except KeyError:
            raise Phase18G3Error(f"no committed metadata for live game {game_id!r} in period {period}") from None

    def record(self, period: int, game_id: str) -> GameRecord:
        key = (int(period), game_id)
        cached = self._records.get(key)
        if cached is not None:
            self._records.move_to_end(key)
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        commit = self.commits(period).get(game_id)
        if commit is None:
            raise Phase18G3Error(f"live game {game_id!r} is not committed in period {period}")
        path = self.root / f"{period_name(period)}{RECORDS_SUFFIX}"
        with path.open("rb") as handle:
            handle.seek(commit.offset)
            payload = handle.read(commit.size)
        if len(payload) != commit.size:
            raise Phase18G3Error(f"{game_id}: short read from {path}")
        if hashlib.sha256(payload).hexdigest() != commit.trajectory_sha256:
            raise Phase18G3Error(f"{game_id}: the stored payload does not reproduce its committed digest")
        decoded = decode_game_record_compressed(payload)
        self._records[key] = decoded
        while len(self._records) > self.record_cache_size:
            self._records.popitem(last=False)
        return decoded

    def examples(self, keys) -> list:
        """Examples for `(game_id, decision_index)` keys, in key order.

        Grouped by game so each record is decoded once and reconstructed in one
        ascending pass through the accepted `examples_for_game`; the results are
        placed back into the caller's order. The key shape is the accepted
        corpus key shape, so a mixed batch's identities hash through the
        trainer's own `keys_digest`.
        """
        by_game: "OrderedDict[str, list]" = OrderedDict()
        for slot, (game_id, index) in enumerate(keys):
            by_game.setdefault(game_id, []).append((int(index), slot))
        results: list = [None] * len(keys)
        for game_id, wanted in by_game.items():
            period = self.period_of(game_id)
            record = self.record(period, game_id)
            metadata = self.metadata(period, game_id)
            plies = tuple(sorted(index for index, _slot in wanted))
            slots = {index: slot for index, slot in wanted}
            produced = 0
            for example in examples_for_game(record, metadata, plies):
                results[slots[example.decision_index]] = example
                produced += 1
            if produced != len(wanted):
                raise Phase18G3Error(f"{game_id}: {produced} examples for {len(wanted)} requested plies")
        return results

    # -- the universe --------------------------------------------------------

    def universe(self, periods) -> tuple:
        """Every `(game_id, decision_index)` of `periods`, frozen order.

        Periods ascending, games in commit order, decisions ascending: the
        order is a pure function of the committed journals, so both lineages
        and every restart enumerate the same universe from the same files.
        """
        keys: list = []
        for period in sorted(int(p) for p in periods):
            for game_id, commit in self.commits(period).items():
                for index in commit.selected_decisions:
                    keys.append((game_id, int(index)))
        return tuple(keys)


def universe_digest(universe) -> str:
    hasher = hashlib.sha256()
    for game_id, index in universe:
        hasher.update(f"{game_id}|{index}\n".encode())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# The lineage-neutral semantic comparison of two lineages' periods
# ---------------------------------------------------------------------------

#: The ONE metadata field the two lineages are permitted to differ in.
LINEAGE_METADATA_FIELD = "lineage"


def lineage_neutral_metadata(metadata: dict) -> dict:
    """The metadata document with exactly the `lineage` field removed."""
    return {key: value for key, value in metadata.items() if key != LINEAGE_METADATA_FIELD}


def lineage_neutral_metadata_sha256(metadata: dict) -> str:
    line = json.dumps(lineage_neutral_metadata(metadata), sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(line.encode()).hexdigest()


def semantic_period_entries(reader: "LiveRecordReader", period: int) -> list:
    """One lineage-neutral entry per commit of a finalised period, in commit order.

    Each entry carries what the fairness comparison reads: the game id, the
    trajectory digest, the selected decisions, the lineage the store stamped,
    the neutral metadata and its digest. Nothing is decoded.
    """
    entries = []
    for game_id, commit in reader.commits(period).items():
        metadata = reader.metadata(period, game_id)
        entries.append(
            {
                "game_id": game_id,
                "record_index": int(commit.record_index),
                "trajectory_sha256": commit.trajectory_sha256,
                "metadata_sha256": commit.metadata_sha256,
                "selected_decisions": tuple(int(index) for index in commit.selected_decisions),
                "final_ply": int(commit.final_ply),
                "total_decisions": int(commit.total_decisions),
                "lineage": metadata.get(LINEAGE_METADATA_FIELD),
                "neutral_metadata": lineage_neutral_metadata(metadata),
                "neutral_metadata_sha256": lineage_neutral_metadata_sha256(metadata),
            }
        )
    return entries


def lineage_neutral_commit_digest(entries) -> str:
    """`commits_digest` with the metadata digest taken over the neutral metadata."""
    hasher = hashlib.sha256()
    for entry in entries:
        hasher.update(f"{entry['game_id']}|{entry['trajectory_sha256']}|{entry['neutral_metadata_sha256']}\n".encode())
    return hasher.hexdigest()


def compare_period_semantics(entries_a, entries_b, *, lineage_a: str, lineage_b: str, max_reported: int = 8) -> dict:
    """Are two lineages' periods the same games, apart from the lineage stamp?

    Required, in order: every entry of `entries_a` is stamped `lineage_a` and
    every entry of `entries_b` `lineage_b`; equal commit counts; equal game
    ids in the same order; equal trajectory digests; equal selected-decision
    lists; equal neutral metadata (every field other than `lineage`). The
    first differences of each kind are reported by game and field. A missing
    lineage stamp, a wrong one, or ANY other metadata difference fails.
    """
    entries_a = list(entries_a)
    entries_b = list(entries_b)
    problems: list = []
    checks: dict = {}

    def note(problem: str) -> None:
        if len(problems) < max_reported:
            problems.append(problem)

    stamped = {}
    for label, entries, expected in ((lineage_a, entries_a, lineage_a), (lineage_b, entries_b, lineage_b)):
        wrong = [e["game_id"] for e in entries if e["lineage"] != expected]
        stamped[label] = not wrong
        for game_id in wrong[:2]:
            note(f"{game_id} in the {label} store is stamped lineage={next(e['lineage'] for e in entries if e['game_id'] == game_id)!r}, not {expected!r}")
    checks["lineage_stamps"] = all(stamped.values())
    checks["lineage_labels_differ"] = lineage_a != lineage_b
    if lineage_a == lineage_b:
        note(f"both stores are labelled {lineage_a!r}; the comparison is between two lineages")

    checks["commit_count"] = len(entries_a) == len(entries_b)
    if not checks["commit_count"]:
        note(f"{len(entries_a)} commits in {lineage_a}, {len(entries_b)} in {lineage_b}")
    ids_a = [e["game_id"] for e in entries_a]
    ids_b = [e["game_id"] for e in entries_b]
    checks["game_ids_and_order"] = ids_a == ids_b
    if not checks["game_ids_and_order"]:
        if sorted(ids_a) == sorted(ids_b):
            note("the same game ids were committed in a different order")
        else:
            only_a = sorted(set(ids_a) - set(ids_b))[:3]
            only_b = sorted(set(ids_b) - set(ids_a))[:3]
            note(f"game ids differ (only in {lineage_a}: {only_a}; only in {lineage_b}: {only_b})")

    trajectories = selections = metadata_same = True
    field_differences: list = []
    for a, b in zip(entries_a, entries_b):
        if a["game_id"] != b["game_id"]:
            continue
        if a["trajectory_sha256"] != b["trajectory_sha256"]:
            trajectories = False
            note(f"{a['game_id']}: trajectory digests differ")
        if tuple(a["selected_decisions"]) != tuple(b["selected_decisions"]):
            selections = False
            note(f"{a['game_id']}: selected decisions differ")
        if a["neutral_metadata"] != b["neutral_metadata"]:
            metadata_same = False
            keys = sorted(set(a["neutral_metadata"]) | set(b["neutral_metadata"]))
            differing = [k for k in keys if a["neutral_metadata"].get(k, "<absent>") != b["neutral_metadata"].get(k, "<absent>")]
            field_differences.append({"game_id": a["game_id"], "fields": differing})
            note(f"{a['game_id']}: metadata differs in non-lineage field(s) {differing}")
    checks["trajectory_digests"] = trajectories
    checks["selected_decisions"] = selections
    checks["neutral_metadata"] = metadata_same

    neutral_a = lineage_neutral_commit_digest(entries_a)
    neutral_b = lineage_neutral_commit_digest(entries_b)
    checks["lineage_neutral_commit_digest"] = neutral_a == neutral_b
    matched = all(checks.values()) and not problems
    return {
        "matched": matched,
        "checks": checks,
        "commits": {lineage_a: len(entries_a), lineage_b: len(entries_b)},
        "selected_examples": {
            lineage_a: int(sum(len(e["selected_decisions"]) for e in entries_a)),
            lineage_b: int(sum(len(e["selected_decisions"]) for e in entries_b)),
        },
        "lineage_neutral_commit_digest": {lineage_a: neutral_a, lineage_b: neutral_b},
        "permitted_difference": f"metadata field {LINEAGE_METADATA_FIELD!r} only: {lineage_a!r} versus {lineage_b!r}",
        "metadata_field_differences": field_differences[:max_reported],
        "problems": problems,
    }


def compare_live_periods(root_a, root_b, period: int, *, lineage_a: str, lineage_b: str) -> dict:
    """The fairness comparison of one period between two lineages' stores.

    Each store is first re-hashed against its own finalisation summary (the
    raw digest's job: file integrity); then the semantic comparison runs.
    Both raw commit digests are reported as audit information. They differ
    whenever at least one game was committed, because each metadata line
    carries the lineage; that inequality is expected and is NOT a condition.
    """
    reader_a = LiveRecordReader(root_a)
    reader_b = LiveRecordReader(root_b)
    integrity = {lineage_a: reader_a.verify_period(period), lineage_b: reader_b.verify_period(period)}
    raw = {lineage_a: reader_a.summary(period)["commit_digest"], lineage_b: reader_b.summary(period)["commit_digest"]}
    semantic = compare_period_semantics(
        semantic_period_entries(reader_a, period),
        semantic_period_entries(reader_b, period),
        lineage_a=lineage_a,
        lineage_b=lineage_b,
    )
    problems = list(semantic["problems"])
    for label, verification in integrity.items():
        if not verification["verified"]:
            problems.append(f"{label}: the live period {period} files do not verify: {verification['problems']}")
    games = semantic["commits"][lineage_a]
    return {
        "period": int(period),
        "matched": semantic["matched"] and all(v["verified"] for v in integrity.values()),
        "file_integrity": integrity,
        "raw_commit_digest": raw,
        "raw_commit_digest_equal": raw[lineage_a] == raw[lineage_b],
        "raw_commit_digest_note": (
            "audit only; unequal by construction when any game was committed, because the hashed "
            "metadata line carries the lineage stamp"
            if games
            else "audit only; equal here because no game was committed in this period"
        ),
        "semantic": semantic,
        "problems": problems,
    }


__all__ = [
    "DONE_SUFFIX",
    "JOURNAL_SUFFIX",
    "LiveCommit",
    "LivePeriodWriter",
    "LINEAGE_METADATA_FIELD",
    "LiveRecordReader",
    "METADATA_SUFFIX",
    "RECORDS_SUFFIX",
    "available_periods",
    "commits_digest",
    "compare_live_periods",
    "compare_period_semantics",
    "discard_periods_after",
    "lineage_neutral_commit_digest",
    "lineage_neutral_metadata",
    "lineage_neutral_metadata_sha256",
    "live_selected_decision_indices",
    "period_name",
    "semantic_period_entries",
    "universe_digest",
]
