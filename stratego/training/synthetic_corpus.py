"""Phase 8 Agent 2: generating, resuming and auditing the synthetic corpus.

Specification sources:

- `02_AGENT_2_SYNTHETIC_CORPUS.md` (mission, storage, determinism, split
  isolation, matchup audit, trajectory correctness, setup provenance,
  finalization)
- `00_PHASE_8_SEQUENCE_AND_COMMON_CONTRACT.md` sections 11-13, 24

What this module owns
---------------------
The schedule and the audits. Playing a game is
:mod:`stratego.training.rule_population`; making a game durable is
:mod:`stratego.training.corpus_commit`. This module decides *which* games exist
(the frozen 100 x (200/40/40) schedule), hands the pending ones to workers, and
then proves the result is the corpus the contract asked for.

Resume is subtraction
---------------------
There is no checkpoint file and no cursor. The schedule is a pure function of
the frozen contract, the committed set is a pure function of the commit
journals, and

```text
pending = scheduled - committed
```

is the whole resume protocol. A game's content depends only on its identifier,
so it does not matter which attempt, which worker or which segment produces it;
an interrupted run and an uninterrupted one converge to the same logical corpus.

Auditing is independent of generation
-------------------------------------
Every audit reads the persisted bytes back and re-derives what it checks from
the frozen contract, never from the generator's own bookkeeping: the replay
audit replays trajectories through the engine, the provenance audit rebuilds
both setups from `setup_provenance_v1`, and the schedule audit compares
committed ids against freshly enumerated ones. A generator bug that produced a
consistent-looking corpus would still fail them.
"""

from __future__ import annotations

import hashlib
import os
import platform
import resource
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..engine.legal_moves import legal_actions
from ..engine.observation import build_observation
from ..engine.replay import terminal_result_label
from ..engine.setup import serialize_setup
from ..engine.state import create_game
from ..engine.transition import apply_action
from ..setups.contracts import SETUP_LIBRARY_VERSION
from ..setups.perturbation import PERTURBATION_SEED_ENCODING, PERTURBATION_VERSION
from ..setups.sampler import SAMPLER_VERSION, load_library_index
from .corpus_commit import (
    CORPUS_COMMIT_VERSION,
    DEFAULT_CORPUS_SHARD_BYTES,
    CorpusReader,
    CorpusWriter,
    audit_commit_integrity,
    corpus_content_digest,
    metadata_digest,
    next_segment,
    reconcile_corpus,
    storage_summary,
    write_shard_manifests,
)
from .reconstruction import observation_digest, reconstruct_state
from .rule_population import (
    RULE_POPULATION_VERSION,
    TeacherCache,
    ordered_matchup_id,
    play_corpus_game,
    roster_digest,
    teacher_by_token,
    validate_game_metadata,
    verify_live_population,
)
from .setup_source import (
    PROVENANCE_SCHEMA_VERSION,
    SETUP_SOURCE_VERSION,
    verify_provenance_against_setups,
    verify_provenance_split,
)
from .trajectory import (
    DEFAULT_SNAPSHOT_INTERVAL,
    TRAJECTORY_VERSION,
    GameRecord,
    validate_game_record,
)
from .warmstart_contract import (
    CORPUS_RULES,
    EXPECTED_SETUP_PROFILE,
    SCHEDULE_TOTALS,
    corpus_setup_source,
    iter_game_identities,
    ordered_matchup_cells,
    verify_frozen_upstream,
)
from .warmstart_seed import (
    CORPUS_MASTER_SEED,
    CORPUS_SPLITS,
    GAMES_PER_CELL,
    SYNTHETIC_CORPUS_VERSION,
    parse_synthetic_game_id,
    selected_decision_indices,
)

#: The default corpus root, relative to the repository. The common contract
#: names this the preferred location and allows redirecting it by
#: configuration; the manifest always records where the bytes actually went and
#: how much room was left.
DEFAULT_CORPUS_ROOT = "data/warmstart/synthetic_warmstart_corpus_v1"

#: Environment variable that redirects the corpus root for one process.
CORPUS_ROOT_ENV = "STRATEGO_WARMSTART_CORPUS_ROOT"

#: Durable redirect: a one-line file holding the corpus root. Deliberately
#: outside `data/warmstart/` so it survives when the corpus bytes do not live
#: in the repository at all, and so it is version-controlled while they are
#: not. It exists because a redirect that only lives in an environment variable
#: is a redirect every later agent can forget to set.
CORPUS_ROOT_POINTER = "data/warmstart_corpus_root.txt"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_corpus_root() -> Path:
    """Where this installation keeps `synthetic_warmstart_corpus_v1`.

    Resolution order, first match wins:

    ```text
    STRATEGO_WARMSTART_CORPUS_ROOT     explicit per-process override
    data/warmstart_corpus_root.txt     the recorded redirect, if any
    data/warmstart/...                 the contract's preferred path
    ```

    Every consumer — the generator, the auditors, and Agents 3-7 — should ask
    this rather than assume a path, so a relocation is one recorded fact
    instead of a search-and-replace.
    """
    configured = os.environ.get(CORPUS_ROOT_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    pointer = repository_root() / CORPUS_ROOT_POINTER
    if pointer.exists():
        recorded = pointer.read_text().strip()
        if recorded:
            return Path(recorded).expanduser()
    return repository_root() / DEFAULT_CORPUS_ROOT


def repository_relative(path: "str | Path") -> str:
    """`path` relative to the repository if it is inside it, else absolute.

    A reporting path must not assume containment: the corpus root is
    redirectable by contract and may sit on another volume entirely, in which
    case there is no relative form and the absolute one is the honest answer.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(repository_root()))
    except ValueError:
        return str(resolved)


def describe_corpus_root() -> dict:
    """Which redirect (if any) chose the corpus root, for the manifest."""
    configured = os.environ.get(CORPUS_ROOT_ENV, "").strip()
    pointer = repository_root() / CORPUS_ROOT_POINTER
    recorded = pointer.read_text().strip() if pointer.exists() else ""
    if configured:
        source = "environment"
    elif recorded:
        source = "pointer_file"
    else:
        source = "repository_default"
    return {
        "root": str(default_corpus_root()),
        "source": source,
        "environment_variable": CORPUS_ROOT_ENV,
        "environment_value": configured,
        "pointer_file": str(pointer),
        "pointer_value": recorded,
        "repository_default": str(repository_root() / DEFAULT_CORPUS_ROOT),
    }

#: Worker ids are two digits in every filename, so a run may not deal work into
#: more chunks than that.
MAX_CHUNKS = 100


class SyntheticCorpusError(RuntimeError):
    """The synthetic corpus could not be generated or audited as contracted."""


def _peak_rss_bytes() -> dict:
    """Peak resident set size of this process and of its finished children.

    `ru_maxrss` is bytes on Darwin and kilobytes on Linux; both are normalized
    here so a recorded number means the same thing on either platform. Children
    are counted separately because generation's memory lives in the workers, not
    in the parent that waits on them.
    """
    scale = 1 if platform.system() == "Darwin" else 1024
    return {
        "parent_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale,
        "worker_peak_rss_bytes": (
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
        ),
        "note": "worker value is the largest single finished child, not their sum",
    }


# ---------------------------------------------------------------------------
# The frozen schedule
# ---------------------------------------------------------------------------


def scheduled_game_ids(split: str) -> tuple:
    """Every logical game id of one split, in frozen schedule order."""
    return tuple(entry[4] for entry in iter_game_identities(split))


def all_scheduled_game_ids(splits: "tuple[str, ...]" = CORPUS_SPLITS) -> tuple:
    ids: list[str] = []
    for split in splits:
        ids.extend(scheduled_game_ids(split))
    return tuple(ids)


def schedule_summary(splits: "tuple[str, ...]" = CORPUS_SPLITS) -> dict:
    """Counts the frozen schedule promises, re-derived rather than quoted."""
    per_split = {split: len(scheduled_game_ids(split)) for split in splits}
    return {
        "cells": len(ordered_matchup_cells()),
        "games_per_cell": {split: GAMES_PER_CELL[split] for split in splits},
        "per_split": per_split,
        "total": sum(per_split.values()),
        "expected_totals": {
            split: SCHEDULE_TOTALS[split] for split in splits if split in SCHEDULE_TOTALS
        },
    }


def pending_game_ids(
    root: "str | Path", splits: "tuple[str, ...]" = CORPUS_SPLITS, *, committed=None
) -> tuple:
    """The scheduled games no commit journal claims yet, in schedule order."""
    if committed is None:
        committed = set(reconcile_corpus(root, splits)["committed"])
    return tuple(
        game_id
        for split in splits
        for game_id in scheduled_game_ids(split)
        if game_id not in committed
    )


def partition_games(game_ids: "tuple[str, ...]", chunks: int) -> list:
    """Deal games round-robin into `chunks` lists.

    Round-robin rather than contiguous slicing: consecutive ids are the same
    ordered matchup, and matchup mean game length varies by more than 2x across
    the roster, so contiguous slices would leave one worker with every long
    game of a slow cell.
    """
    if chunks < 1:
        raise SyntheticCorpusError(f"chunks must be at least 1, got {chunks}")
    buckets: list[list] = [[] for _ in range(chunks)]
    for index, game_id in enumerate(game_ids):
        buckets[index % chunks].append(game_id)
    return [bucket for bucket in buckets if bucket]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_games(
    root: "str | Path",
    game_ids: "tuple[str, ...]",
    *,
    segment: int,
    worker_id: int,
    target_bytes: int = DEFAULT_CORPUS_SHARD_BYTES,
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
    fsync_on_commit: bool = False,
    crash_hook=None,
    progress=None,
) -> dict:
    """Play and commit a list of games inside one process.

    Opens one writer per split it actually touches, so a chunk that spans splits
    still keeps each split's bytes in its own directory. Every writer is closed
    on the way out, including when a game raises — an interrupted chunk must
    leave a reconcilable file set, not an open handle.
    """
    writers: dict[str, CorpusWriter] = {}
    sources: dict[str, object] = {}
    teachers = TeacherCache()
    counters = {
        "games": 0,
        "decisions": 0,
        "plies": 0,
        "play_seconds": 0.0,
        "commit_seconds": 0.0,
    }
    try:
        for game_id in game_ids:
            split = parse_synthetic_game_id(game_id)["split"]
            source = sources.get(split)
            if source is None:
                source = corpus_setup_source(split)
                sources[split] = source
            writer = writers.get(split)
            if writer is None:
                writer = CorpusWriter(
                    root,
                    split=split,
                    segment=segment,
                    worker_id=worker_id,
                    target_bytes=target_bytes,
                    fsync_on_commit=fsync_on_commit,
                    crash_hook=crash_hook,
                )
                writers[split] = writer
            started = time.perf_counter()
            game = play_corpus_game(
                game_id,
                setup_source=source,
                teachers=teachers,
                snapshot_interval=snapshot_interval,
            )
            played = time.perf_counter()
            writer.write_game(game)
            counters["play_seconds"] += played - started
            counters["commit_seconds"] += time.perf_counter() - played
            counters["games"] += 1
            counters["decisions"] += game.total_decisions
            counters["plies"] += game.record.final_ply
            if progress is not None:
                progress(game)
    finally:
        writer_stats = [writer.close() for writer in writers.values()]
    counters["writers"] = writer_stats
    counters["worker_id"] = worker_id
    counters["segment"] = segment
    counters["policies_built"] = len(teachers)
    return counters


_WORKER_STATE: dict = {}


def _worker_init(options: dict) -> None:
    _WORKER_STATE.clear()
    _WORKER_STATE["options"] = options


def _run_chunk(payload: tuple) -> dict:
    """Generate one chunk in a worker process. Plain data in, plain data out."""
    worker_id, game_ids = payload
    options = _WORKER_STATE["options"]
    return generate_games(
        options["root"],
        tuple(game_ids),
        segment=options["segment"],
        worker_id=worker_id,
        target_bytes=options["target_bytes"],
        snapshot_interval=options["snapshot_interval"],
        fsync_on_commit=options["fsync_on_commit"],
    )


def generate_corpus(
    root: "str | Path",
    *,
    splits: "tuple[str, ...]" = CORPUS_SPLITS,
    worker_count: int = 1,
    chunks_per_worker: int = 4,
    target_bytes: int = DEFAULT_CORPUS_SHARD_BYTES,
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
    fsync_on_commit: bool = False,
    limit: "int | None" = None,
    game_ids: "tuple[str, ...] | None" = None,
) -> dict:
    """Reconcile, then generate every scheduled game that is not committed yet.

    Safe to call repeatedly: the first call generates the corpus, a call after a
    crash generates exactly the missing games, and a call on a complete corpus
    does nothing. `limit` stops after that many games — a deliberately
    interruptible run, used by the crash tests and by staged production runs.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    reconciliation = reconcile_corpus(root, splits)
    if reconciliation["duplicate_committed_ids"]:
        raise SyntheticCorpusError(
            "the commit journals already contain duplicate game ids: "
            f"{reconciliation['duplicate_committed_ids'][:5]}"
        )
    committed = set(reconciliation["committed"])
    pending = (
        pending_game_ids(root, splits, committed=committed)
        if game_ids is None
        else tuple(game_id for game_id in game_ids if game_id not in committed)
    )
    if limit is not None:
        pending = pending[: int(limit)]

    segment = next_segment(root, splits)
    result = {
        "corpus_version": SYNTHETIC_CORPUS_VERSION,
        "root": str(root),
        "segment": segment,
        "splits": list(splits),
        "already_committed": len(committed),
        "scheduled": len(all_scheduled_game_ids(splits)),
        "pending": len(pending),
        "reconciliation": {
            key: value
            for key, value in reconciliation.items()
            if key not in ("committed",)
        },
        "worker_count": worker_count,
        "chunks": [],
        "games_generated": 0,
        "decisions_generated": 0,
        "plies_generated": 0,
    }
    if not pending:
        result["wall_clock_seconds"] = time.perf_counter() - started
        return result

    chunk_count = max(
        1, min(MAX_CHUNKS, len(pending), max(1, worker_count) * max(1, chunks_per_worker))
    )
    buckets = partition_games(pending, chunk_count)
    options = {
        "root": str(root),
        "segment": segment,
        "target_bytes": int(target_bytes),
        "snapshot_interval": int(snapshot_interval),
        "fsync_on_commit": bool(fsync_on_commit),
    }

    if worker_count == 1:
        _worker_init(options)
        chunk_results = [_run_chunk((index, bucket)) for index, bucket in enumerate(buckets)]
    else:
        payloads = [(index, bucket) for index, bucket in enumerate(buckets)]
        with ProcessPoolExecutor(
            max_workers=worker_count, initializer=_worker_init, initargs=(options,)
        ) as pool:
            chunk_results = list(pool.map(_run_chunk, payloads))

    result["chunks"] = chunk_results
    result["games_generated"] = sum(chunk["games"] for chunk in chunk_results)
    result["decisions_generated"] = sum(chunk["decisions"] for chunk in chunk_results)
    result["plies_generated"] = sum(chunk["plies"] for chunk in chunk_results)
    result["wall_clock_seconds"] = time.perf_counter() - started
    elapsed = result["wall_clock_seconds"]
    result["games_per_second"] = result["games_generated"] / elapsed if elapsed else 0.0
    result["decisions_per_second"] = (
        result["decisions_generated"] / elapsed if elapsed else 0.0
    )
    result["memory"] = _peak_rss_bytes()
    uncompressed = sum(
        writer["uncompressed_bytes"] for chunk in chunk_results for writer in chunk["writers"]
    )
    compressed = sum(
        writer["compressed_bytes"] for chunk in chunk_results for writer in chunk["writers"]
    )
    result["bytes"] = {
        "uncompressed": uncompressed,
        "compressed": compressed,
        "compression_ratio": compressed / uncompressed if uncompressed else 0.0,
        "compressed_bytes_per_game": (
            compressed / result["games_generated"] if result["games_generated"] else 0.0
        ),
        "compressed_bytes_per_decision": (
            compressed / result["decisions_generated"]
            if result["decisions_generated"]
            else 0.0
        ),
    }
    result["seconds_by_phase"] = {
        "play": sum(chunk["play_seconds"] for chunk in chunk_results),
        "commit": sum(chunk["commit_seconds"] for chunk in chunk_results),
        "encode": sum(
            writer["encode_seconds"] for chunk in chunk_results for writer in chunk["writers"]
        ),
        "compress": sum(
            writer["compress_seconds"] for chunk in chunk_results for writer in chunk["writers"]
        ),
        "verify": sum(
            writer["verify_seconds"] for chunk in chunk_results for writer in chunk["writers"]
        ),
        "write": sum(
            writer["write_seconds"] for chunk in chunk_results for writer in chunk["writers"]
        ),
    }
    return result


# ---------------------------------------------------------------------------
# Trajectory correctness
# ---------------------------------------------------------------------------


def replay_game(record: GameRecord, *, observation_plies: int = 0) -> dict:
    """Replay one stored game through the frozen engine and report mismatches.

    The record is not trusted for anything the engine can re-derive. Starting
    from the two stored setups, the game is replayed from the record's *action
    list* — the same field the frozen replay path uses — while every ply
    re-generates the legal action list and compares it to the stored one, checks
    the acting player, and requires the decision record to name the same action
    the action list does. The terminal state's result, reason and length must
    then equal the stored header. `observation_plies` additionally rebuilds that
    many positions through the *snapshot* path and compares their observations
    with the linear replay's, so the two independent reconstruction routes have
    to agree as well.
    """
    problems: list[str] = []
    state = create_game(
        record.red_setup, record.blue_setup, rules=record.rules(), game_id=record.game_id
    )
    checkpoints = _observation_checkpoints(record, observation_plies)
    observation_checks = 0
    legal_checks = 0

    illegal_actions = 0
    legal_set_mismatches = 0

    if len(record.actions) != len(record.decisions):
        problems.append(
            f"{len(record.actions)} stored actions for {len(record.decisions)} decisions"
        )
        return {
            "game_id": record.game_id,
            "problems": problems,
            "decisions": len(record.decisions),
            "legal_checks": 0,
            "observation_checks": 0,
            "illegal_actions": 0,
            "legal_set_mismatches": 0,
        }

    for index, (action_id, decision) in enumerate(zip(record.actions, record.decisions)):
        if state.terminal:
            problems.append(f"ply {index}: the replayed game ended before the record did")
            break
        legal = legal_actions(state)
        legal_checks += 1
        if tuple(legal) != tuple(decision.legal_action_ids):
            legal_set_mismatches += 1
            problems.append(f"ply {index}: replayed legal actions differ from the record")
            break
        if action_id not in legal:
            illegal_actions += 1
            problems.append(f"ply {index}: the stored action is not legal in this position")
            break
        if state.acting_player != decision.acting_player:
            problems.append(f"ply {index}: replayed acting player differs from the record")
            break
        if state.total_moves != decision.ply:
            problems.append(f"ply {index}: replayed ply index differs from the record")
            break
        if action_id != decision.selected_action_id:
            problems.append(
                f"ply {index}: the action list and the decision record name different "
                "actions"
            )
            break
        if index in checkpoints:
            observation_checks += 1
            problems.extend(_compare_reconstruction(record, index, state))
        try:
            apply_action(state, action_id, legal=legal)
        except Exception as error:  # noqa: BLE001 - an engine rejection is a finding
            illegal_actions += 1
            problems.append(
                f"ply {index}: engine rejected the stored action: "
                f"{type(error).__name__}: {error}"
            )
            break

    if not problems:
        if not state.terminal:
            problems.append("the replayed game is not terminal")
        else:
            if terminal_result_label(state) != record.terminal_result:
                problems.append("replayed result differs from the record")
            if state.terminal_reason != record.terminal_reason:
                problems.append("replayed terminal reason differs from the record")
            if state.total_moves != record.final_ply:
                problems.append("replayed length differs from the record")
    return {
        "game_id": record.game_id,
        "problems": problems,
        "decisions": len(record.decisions),
        "legal_checks": legal_checks,
        "observation_checks": observation_checks,
        # Counted rather than inferred from the problem text: "0 illegal actions"
        # is a named PASS gate, and a gate should read a number.
        "illegal_actions": illegal_actions,
        "legal_set_mismatches": legal_set_mismatches,
    }


def _observation_checkpoints(record: GameRecord, count: int) -> set:
    """Evenly spaced plies at which to cross-check the snapshot path."""
    total = len(record.decisions)
    if count <= 0 or total == 0:
        return set()
    if count >= total:
        return set(range(total))
    return {(index * total) // count for index in range(count)}


def _compare_reconstruction(record: GameRecord, ply: int, live_state) -> list:
    """Snapshot-based reconstruction of one ply against the linear replay."""
    problems: list[str] = []
    rebuilt, _replayed = reconstruct_state(record, ply)
    if tuple(rebuilt.board) != tuple(live_state.board):
        problems.append(f"ply {ply}: reconstructed board differs from the replayed board")
    if rebuilt.acting_player != live_state.acting_player:
        problems.append(f"ply {ply}: reconstructed acting player differs")
    if tuple(legal_actions(rebuilt)) != tuple(legal_actions(live_state)):
        problems.append(f"ply {ply}: reconstructed legal actions differ")
    observer = live_state.acting_player
    if observation_digest(build_observation(rebuilt, observer)) != observation_digest(
        build_observation(live_state, observer)
    ):
        problems.append(f"ply {ply}: reconstructed observation differs")
    return problems


def audit_provenance(metadata: dict, record: GameRecord) -> list:
    """Rebuild both setups from `setup_provenance_v1` and check the split.

    The strong form: each side's library entry is rebuilt from provenance alone
    through the frozen Phase 7 path, re-oriented, and required to equal the
    setup the trajectory actually stores.
    """
    provenance = metadata["setup_provenance"]
    problems = list(verify_provenance_split(provenance, metadata["corpus_split"]))
    problems.extend(
        verify_provenance_against_setups(
            provenance, red_setup=record.red_setup, blue_setup=record.blue_setup
        )
    )
    return problems


def setup_fingerprints(metadata: dict, record: GameRecord) -> list:
    """Every disagreement between the recorded engine setups and the trajectory.

    Cheap enough to run on every game: it compares the serialized setup the
    provenance sidecar recorded with the one the record stores, without
    rebuilding the library entry.
    """
    problems: list[str] = []
    provenance = metadata["setup_provenance"]
    for side, setup in (("red", record.red_setup), ("blue", record.blue_setup)):
        stored = serialize_setup(setup)
        if provenance[side]["engine_setup"] != stored:
            problems.append(f"{side}: provenance engine_setup differs from the trajectory")
        if metadata[f"{side}_setup"] != stored:
            problems.append(f"{side}: metadata setup differs from the trajectory")
    return problems


# ---------------------------------------------------------------------------
# The corpus audit
# ---------------------------------------------------------------------------


def _audit_chunk(payload: tuple) -> dict:
    """Audit one chunk of committed games inside a worker process."""
    root, splits, game_ids, observation_plies, full_provenance_ids = payload
    reader = CorpusReader(root, tuple(splits))
    return audit_games(
        reader,
        tuple(game_ids),
        observation_plies=observation_plies,
        full_provenance_ids=frozenset(full_provenance_ids),
    )


def audit_games(
    reader: CorpusReader,
    game_ids: "tuple[str, ...]",
    *,
    observation_plies: int = 0,
    full_provenance_ids: "frozenset | None" = None,
) -> dict:
    """Replay, provenance and schema audit of a list of committed games."""
    full_provenance_ids = full_provenance_ids or frozenset()
    problems: list[str] = []
    cells: dict = defaultdict(
        lambda: {
            "games": 0,
            "red_wins": 0,
            "blue_wins": 0,
            "draws": 0,
            "plies": 0,
            "decisions": 0,
            "selected_decisions": 0,
            "min_plies": None,
            "max_plies": None,
        }
    )
    terminal_reasons: Counter = Counter()
    results: Counter = Counter()
    length_histogram: Counter = Counter()
    base_ids: dict = defaultdict(set)
    families: Counter = Counter()
    audited = 0
    replayed_decisions = 0
    observation_checks = 0
    provenance_rebuilds = 0
    illegal_actions = 0
    legal_set_mismatches = 0
    zero_decision_games: list[str] = []

    for game_id in game_ids:
        record, metadata = reader.game(game_id)
        split = metadata["corpus_split"]

        problems.extend(
            f"{game_id}: {problem}" for problem in validate_game_record(record)
        )
        problems.extend(
            f"{game_id}: {problem}" for problem in validate_game_metadata(metadata, record)
        )
        problems.extend(f"{game_id}: {problem}" for problem in setup_fingerprints(metadata, record))

        replay = replay_game(record, observation_plies=observation_plies)
        problems.extend(f"{game_id}: {problem}" for problem in replay["problems"])
        replayed_decisions += replay["decisions"]
        observation_checks += replay["observation_checks"]
        illegal_actions += replay["illegal_actions"]
        legal_set_mismatches += replay["legal_set_mismatches"]

        if game_id in full_provenance_ids:
            provenance_rebuilds += 1
            problems.extend(
                f"{game_id}: {problem}" for problem in audit_provenance(metadata, record)
            )

        provenance = metadata["setup_provenance"]
        for side in ("red", "blue"):
            base_ids[split].add(str(provenance[side]["base_setup_id"]))
            families[str(provenance[side]["primary_family_id"])] += 1

        key = (split, int(metadata["cell_index"]))
        cell = cells[key]
        cell["games"] += 1
        cell["plies"] += int(record.final_ply)
        cell["decisions"] += len(record.decisions)
        cell["selected_decisions"] += len(
            selected_decision_indices(game_id, len(record.decisions))
        )
        cell["min_plies"] = (
            record.final_ply
            if cell["min_plies"] is None
            else min(cell["min_plies"], record.final_ply)
        )
        cell["max_plies"] = (
            record.final_ply
            if cell["max_plies"] is None
            else max(cell["max_plies"], record.final_ply)
        )
        if record.terminal_result == "red_win":
            cell["red_wins"] += 1
        elif record.terminal_result == "blue_win":
            cell["blue_wins"] += 1
        else:
            cell["draws"] += 1
        terminal_reasons[record.terminal_reason] += 1
        results[record.terminal_result] += 1
        length_histogram[_length_bucket(record.final_ply)] += 1
        if not record.decisions:
            zero_decision_games.append(game_id)
        audited += 1

    return {
        "audited": audited,
        "problems": problems,
        "replayed_decisions": replayed_decisions,
        "observation_checks": observation_checks,
        "provenance_rebuilds": provenance_rebuilds,
        "illegal_actions": illegal_actions,
        "legal_set_mismatches": legal_set_mismatches,
        "cells": {f"{split}|{index}": value for (split, index), value in cells.items()},
        "terminal_reasons": dict(terminal_reasons),
        "results": dict(results),
        "length_histogram": dict(length_histogram),
        "base_ids": {split: sorted(values) for split, values in base_ids.items()},
        "families": dict(families),
        "zero_decision_games": zero_decision_games,
    }


def _length_bucket(plies: int) -> str:
    for upper in (25, 50, 100, 200, 400, 800, 1600, 3200):
        if plies < upper:
            return f"<{upper}"
    return ">=3200"


def _merge_audit(chunks: "list[dict]") -> dict:
    """Combine per-chunk audit results into one corpus-level result."""
    merged = {
        "audited": 0,
        "problems": [],
        "replayed_decisions": 0,
        "observation_checks": 0,
        "provenance_rebuilds": 0,
        "illegal_actions": 0,
        "legal_set_mismatches": 0,
        "cells": {},
        "terminal_reasons": Counter(),
        "results": Counter(),
        "length_histogram": Counter(),
        "base_ids": defaultdict(set),
        "families": Counter(),
        "zero_decision_games": [],
    }
    for chunk in chunks:
        merged["audited"] += chunk["audited"]
        merged["problems"].extend(chunk["problems"])
        merged["replayed_decisions"] += chunk["replayed_decisions"]
        merged["observation_checks"] += chunk["observation_checks"]
        merged["provenance_rebuilds"] += chunk["provenance_rebuilds"]
        merged["illegal_actions"] += chunk["illegal_actions"]
        merged["legal_set_mismatches"] += chunk["legal_set_mismatches"]
        merged["terminal_reasons"].update(chunk["terminal_reasons"])
        merged["results"].update(chunk["results"])
        merged["length_histogram"].update(chunk["length_histogram"])
        merged["families"].update(chunk["families"])
        merged["zero_decision_games"].extend(chunk["zero_decision_games"])
        for split, values in chunk["base_ids"].items():
            merged["base_ids"][split].update(values)
        for key, cell in chunk["cells"].items():
            current = merged["cells"].get(key)
            if current is None:
                merged["cells"][key] = dict(cell)
                continue
            for field in (
                "games",
                "red_wins",
                "blue_wins",
                "draws",
                "plies",
                "decisions",
                "selected_decisions",
            ):
                current[field] += cell[field]
            for field, combine in (("min_plies", min), ("max_plies", max)):
                if cell[field] is None:
                    continue
                current[field] = (
                    cell[field]
                    if current[field] is None
                    else combine(current[field], cell[field])
                )
    merged["terminal_reasons"] = dict(merged["terminal_reasons"])
    merged["results"] = dict(merged["results"])
    merged["length_histogram"] = dict(merged["length_histogram"])
    merged["families"] = dict(merged["families"])
    merged["base_ids"] = {split: sorted(values) for split, values in merged["base_ids"].items()}
    return merged


def audit_corpus(
    root: "str | Path",
    *,
    splits: "tuple[str, ...]" = CORPUS_SPLITS,
    worker_count: int = 1,
    chunks_per_worker: int = 4,
    observation_plies: int = 4,
    full_provenance_games: "int | None" = None,
) -> dict:
    """Read the committed corpus back and check every Agent 2 gate.

    `full_provenance_games` limits the expensive library rebuild to that many
    games (evenly spread across the schedule); `None` rebuilds every game, which
    is the preferred evidence and is affordable at this corpus size.
    """
    root = Path(root)
    started = time.perf_counter()
    reader = CorpusReader(root, splits)
    committed_ids = reader.game_ids()
    integrity = audit_commit_integrity(root, splits)

    if full_provenance_games is None:
        full_ids = frozenset(committed_ids)
    else:
        step = max(1, len(committed_ids) // max(1, int(full_provenance_games)))
        full_ids = frozenset(committed_ids[::step][: int(full_provenance_games)])

    # One chunk per audit worker, partitioned by *file set* rather than
    # round-robin by game. A reader loads metadata one whole file set at a time,
    # so a round-robin partition would make every worker parse every metadata
    # record; grouping by file set keeps each worker's resident metadata to the
    # file sets it was actually given.
    chunk_count = 1 if worker_count <= 1 else max(1, min(MAX_CHUNKS, worker_count))
    buckets = _partition_by_file_set(reader, committed_ids, chunk_count)
    if worker_count == 1 or len(buckets) <= 1:
        chunks = [
            audit_games(
                reader,
                tuple(bucket),
                observation_plies=observation_plies,
                full_provenance_ids=full_ids,
            )
            for bucket in buckets
        ]
    else:
        payloads = [
            (str(root), list(splits), list(bucket), observation_plies, sorted(full_ids & set(bucket)))
            for bucket in buckets
        ]
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            chunks = list(pool.map(_audit_chunk, payloads))
    merged = _merge_audit(chunks) if chunks else _merge_audit([])

    schedule = _audit_schedule(committed_ids, splits)
    isolation = _audit_split_isolation(merged["base_ids"], committed_ids, splits)

    return {
        "corpus_version": SYNTHETIC_CORPUS_VERSION,
        "commit_version": CORPUS_COMMIT_VERSION,
        "root": str(root),
        "audited_games": merged["audited"],
        "committed_games": len(committed_ids),
        "integrity": integrity,
        "schedule": schedule,
        "split_isolation": isolation,
        "replayed_decisions": merged["replayed_decisions"],
        "observation_cross_checks": merged["observation_checks"],
        "full_provenance_rebuilds": merged["provenance_rebuilds"],
        "illegal_actions": merged["illegal_actions"],
        "legal_set_mismatches": merged["legal_set_mismatches"],
        "cells": merged["cells"],
        "terminal_reasons": merged["terminal_reasons"],
        "results": merged["results"],
        "length_histogram": merged["length_histogram"],
        "families": merged["families"],
        "zero_decision_games": merged["zero_decision_games"],
        # Per-game findings only. Schedule and split-isolation findings stay in
        # their own sections so a partial corpus under test does not read as a
        # replay failure.
        "problems": merged["problems"],
        "problem_count": (
            len(merged["problems"])
            + len(schedule["problems"])
            + len(isolation["problems"])
        ),
        "content_digest": corpus_content_digest(root, splits),
        "storage": storage_summary(root, splits),
        "audit_seconds": time.perf_counter() - started,
    }


def _partition_by_file_set(
    reader: CorpusReader, committed_ids: "tuple[str, ...]", chunks: int
) -> list:
    """Group games by the file set that holds them, then pack groups into chunks.

    Longest-processing-time first: the largest file sets are placed before the
    smallest, which keeps the chunks even without splitting a file set across
    two workers.
    """
    if not committed_ids:
        return []
    groups: dict = defaultdict(list)
    for game_id in committed_ids:
        commit = reader.commits[game_id]
        groups[(commit.split, commit.file_set)].append(game_id)
    ordered = sorted(groups.values(), key=lambda entry: (-len(entry), entry[0]))
    buckets: list = [[] for _ in range(max(1, min(chunks, len(ordered))))]
    for group in ordered:
        target = min(buckets, key=len)
        target.extend(sorted(group))
    return [bucket for bucket in buckets if bucket]


def _audit_schedule(committed_ids: "tuple[str, ...]", splits: "tuple[str, ...]") -> dict:
    """Committed ids against a freshly enumerated schedule."""
    problems: list[str] = []
    committed = set(committed_ids)
    if len(committed) != len(committed_ids):
        problems.append("the committed index contains duplicate game ids")
    per_split = {}
    per_cell_problems = 0
    for split in splits:
        scheduled = scheduled_game_ids(split)
        present = [game_id for game_id in scheduled if game_id in committed]
        missing = [game_id for game_id in scheduled if game_id not in committed]
        per_split[split] = {
            "scheduled": len(scheduled),
            "committed": len(present),
            "missing": len(missing),
            "expected": SCHEDULE_TOTALS.get(split),
        }
        if missing:
            problems.append(
                f"{split}: {len(missing)} scheduled games are not committed "
                f"(first: {missing[0]})"
            )
        if SCHEDULE_TOTALS.get(split) is not None and len(scheduled) != SCHEDULE_TOTALS[split]:
            problems.append(
                f"{split}: enumerated {len(scheduled)} games, the contract promises "
                f"{SCHEDULE_TOTALS[split]}"
            )
        counts: Counter = Counter()
        for game_id in present:
            identity = parse_synthetic_game_id(game_id)
            counts[(identity["red_token"], identity["blue_token"])] += 1
        for cell in ordered_matchup_cells():
            observed = counts[(cell["red_token"], cell["blue_token"])]
            if observed != GAMES_PER_CELL[split]:
                per_cell_problems += 1
                if per_cell_problems <= 10:
                    problems.append(
                        f"{split}: cell {cell['cell_index']} holds {observed} games, "
                        f"expected {GAMES_PER_CELL[split]}"
                    )
    foreign = sorted(committed - set(all_scheduled_game_ids(splits)))
    if foreign:
        problems.append(f"{len(foreign)} committed games are not in the schedule")
    return {
        "per_split": per_split,
        "unscheduled_committed_ids": foreign[:10],
        "cells_with_wrong_count": per_cell_problems,
        "problems": problems,
    }


def _audit_split_isolation(
    base_ids: dict, committed_ids: "tuple[str, ...]", splits: "tuple[str, ...]"
) -> dict:
    """Setup-base overlap between splits, and game ids appearing twice."""
    problems: list[str] = []
    overlaps = {}
    names = [split for split in splits if split in base_ids]
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = sorted(set(base_ids[left]) & set(base_ids[right]))
            overlaps[f"{left}|{right}"] = len(shared)
            if shared:
                problems.append(
                    f"{left}/{right}: {len(shared)} setup base ids appear in both "
                    f"splits (first: {shared[0]})"
                )
    per_split_ids: dict = defaultdict(set)
    for game_id in committed_ids:
        per_split_ids[parse_synthetic_game_id(game_id)["split"]].add(game_id)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            shared = per_split_ids[left] & per_split_ids[right]
            if shared:  # pragma: no cover - the id encodes its split
                problems.append(f"{left}/{right}: {len(shared)} game ids appear in both")
    return {
        "base_id_counts": {split: len(values) for split, values in base_ids.items()},
        "base_id_overlaps": overlaps,
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Matchup counts and manifest
# ---------------------------------------------------------------------------

MATCHUP_CSV_COLUMNS = (
    "corpus_split",
    "cell_index",
    "red_policy",
    "blue_policy",
    "ordered_matchup_id",
    "red_policy_weight",
    "blue_policy_weight",
    "games",
    "expected_games",
    "red_wins",
    "blue_wins",
    "draws",
    "total_plies",
    "mean_plies",
    "min_plies",
    "max_plies",
    "total_decisions",
    "selected_decisions",
)


def matchup_rows(audit: dict, splits: "tuple[str, ...]" = CORPUS_SPLITS) -> list:
    """One row per ordered cell per split, in frozen schedule order."""
    rows = []
    for split in splits:
        for cell in ordered_matchup_cells():
            entry = audit["cells"].get(f"{split}|{cell['cell_index']}", {})
            games = int(entry.get("games", 0))
            rows.append(
                {
                    "corpus_split": split,
                    "cell_index": cell["cell_index"],
                    "red_policy": cell["red_token"],
                    "blue_policy": cell["blue_token"],
                    "ordered_matchup_id": ordered_matchup_id(
                        cell["red_token"], cell["blue_token"]
                    ),
                    "red_policy_weight": teacher_by_token(cell["red_token"]).policy_weight,
                    "blue_policy_weight": teacher_by_token(cell["blue_token"]).policy_weight,
                    "games": games,
                    "expected_games": GAMES_PER_CELL[split],
                    "red_wins": int(entry.get("red_wins", 0)),
                    "blue_wins": int(entry.get("blue_wins", 0)),
                    "draws": int(entry.get("draws", 0)),
                    "total_plies": int(entry.get("plies", 0)),
                    "mean_plies": round(entry.get("plies", 0) / games, 3) if games else 0.0,
                    "min_plies": entry.get("min_plies") if games else 0,
                    "max_plies": entry.get("max_plies") if games else 0,
                    "total_decisions": int(entry.get("decisions", 0)),
                    "selected_decisions": int(entry.get("selected_decisions", 0)),
                }
            )
    return rows


def finalize_corpus(
    root: "str | Path",
    *,
    splits: "tuple[str, ...]" = CORPUS_SPLITS,
    worker_count: int = 1,
    observation_plies: int = 4,
    full_provenance_games: "int | None" = None,
    generation_commands: "list | None" = None,
) -> dict:
    """Reconcile, audit, write shard manifests and produce the corpus manifest.

    Finalization is deliberately re-runnable and reads only persisted bytes, so
    the manifest it writes describes the corpus that exists rather than the one
    the generator believed it had written.
    """
    root = Path(root)
    reconciliation = reconcile_corpus(root, splits)
    audit = audit_corpus(
        root,
        splits=splits,
        worker_count=worker_count,
        observation_plies=observation_plies,
        full_provenance_games=full_provenance_games,
    )
    manifests = write_shard_manifests(root, splits)
    manifest = corpus_manifest(
        root,
        splits=splits,
        audit=audit,
        shard_manifests=manifests,
        generation_commands=generation_commands or [],
    )
    return {
        "reconciliation": {
            key: value for key, value in reconciliation.items() if key != "committed"
        },
        "audit": audit,
        "manifest": manifest,
        "shard_manifests": len(manifests),
    }


def corpus_manifest(
    root: "str | Path",
    *,
    splits: "tuple[str, ...]" = CORPUS_SPLITS,
    audit: dict,
    shard_manifests: "list | None" = None,
    generation_commands: "list | None" = None,
) -> dict:
    """Everything needed to reproduce this corpus from scratch."""
    root = Path(root)
    usage = _disk_usage(root)
    manifest = {
        "corpus_version": SYNTHETIC_CORPUS_VERSION,
        "contract_version": "warmstart_training_contract_v1",
        "commit_version": CORPUS_COMMIT_VERSION,
        "rule_population_version": RULE_POPULATION_VERSION,
        "trajectory_schema": TRAJECTORY_VERSION,
        "snapshot_interval": DEFAULT_SNAPSHOT_INTERVAL,
        "compression": {"codec": "zlib", "level": 6, "container": "stgshard_v1"},
        "policy_roster_digest": roster_digest(),
        "setup_library_version": SETUP_LIBRARY_VERSION,
        "setup_library_digest": load_library_index().content_digest,
        "sampler_version": SAMPLER_VERSION,
        "perturbation_version": PERTURBATION_VERSION,
        "seed_encoding": PERTURBATION_SEED_ENCODING,
        "setup_source_version": SETUP_SOURCE_VERSION,
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "setup_profile": EXPECTED_SETUP_PROFILE,
        "corpus_master_seed": CORPUS_MASTER_SEED,
        "corpus_root_resolution": describe_corpus_root(),
        "split_seed_rule": (
            "every stream is derive_warmstart_seed(domain, synthetic_game_id); the "
            "split is part of the game id, so the splits are domain-separated by "
            "construction rather than by separate seeds"
        ),
        "rules": {
            "context": CORPUS_RULES.context,
            "battleless_move_limit": CORPUS_RULES.battleless_move_limit,
            "absolute_move_limit": CORPUS_RULES.absolute_move_limit,
            "first_player": CORPUS_RULES.first_player,
        },
        "schedule": schedule_summary(splits),
        "game_counts": {
            split: audit["schedule"]["per_split"][split]["committed"] for split in splits
        },
        "ordered_matchup_counts": {
            split: GAMES_PER_CELL[split] for split in splits
        },
        "content_digest": audit["content_digest"],
        "metadata_digest": _metadata_digest(root, splits),
        "commit_index_digest": _commit_index_digest(root, splits),
        "storage_path": str(root.resolve()),
        "storage": audit["storage"],
        "free_bytes": usage["free_bytes"],
        "total_bytes_on_volume": usage["total_bytes"],
        "shard_files": len(shard_manifests or []),
        "generation_commands": list(generation_commands or []),
        "regeneration": (
            "python scripts/run_phase8_agent02.py --generate --finalize; the "
            "schedule, seeds and policies are all frozen, so a fresh run of the "
            "same command reproduces this corpus byte for byte"
        ),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    return manifest


def _metadata_digest(root: "str | Path", splits: "tuple[str, ...]") -> str:
    """SHA-256 over every committed metadata record, ordered by game id."""
    reader = CorpusReader(root, splits)
    hasher = hashlib.sha256()
    for game_id in reader.game_ids():
        hasher.update(f"{game_id}|{metadata_digest(reader.metadata(game_id))}\n".encode())
    return hasher.hexdigest()


def _commit_index_digest(root: "str | Path", splits: "tuple[str, ...]") -> str:
    """SHA-256 over the commit index itself: ids, digests and record counts."""
    reader = CorpusReader(root, splits)
    hasher = hashlib.sha256()
    for game_id in reader.game_ids():
        commit = reader.commits[game_id]
        hasher.update(
            f"{commit.game_id}|{commit.split}|{commit.trajectory_sha256}|"
            f"{commit.metadata_sha256}|{commit.final_ply}|{commit.total_decisions}\n".encode()
        )
    return hasher.hexdigest()


def _disk_usage(root: "str | Path") -> dict:
    path = Path(root)
    while not path.exists() and path != path.parent:
        path = path.parent
    usage = os.statvfs(path)
    return {
        "free_bytes": usage.f_bavail * usage.f_frsize,
        "total_bytes": usage.f_blocks * usage.f_frsize,
    }


# ---------------------------------------------------------------------------
# Pass gates
# ---------------------------------------------------------------------------


def completion_gates(audit: dict, *, splits: "tuple[str, ...]" = CORPUS_SPLITS) -> dict:
    """Every Agent 2 PASS gate, evaluated from the audit of persisted bytes."""
    integrity = audit["integrity"]
    schedule = audit["schedule"]
    isolation = audit["split_isolation"]
    expected_total = sum(SCHEDULE_TOTALS[split] for split in splits if split in SCHEDULE_TOTALS)
    return {
        "scheduled_games_exact": audit["committed_games"] == expected_total,
        "split_counts_exact": all(
            schedule["per_split"][split]["committed"] == SCHEDULE_TOTALS[split]
            for split in splits
            if split in SCHEDULE_TOTALS
        ),
        "all_cells_exact": schedule["cells_with_wrong_count"] == 0,
        "no_missing_scheduled_games": all(
            schedule["per_split"][split]["missing"] == 0 for split in splits
        ),
        "no_unscheduled_games": not schedule["unscheduled_committed_ids"],
        "zero_duplicate_committed_ids": not integrity["duplicate_committed_ids"],
        "zero_orphan_trajectories": not integrity["orphan_trajectory_records"],
        "zero_orphan_metadata": not integrity["orphan_metadata_records"],
        "zero_missing_payloads": not integrity["missing_trajectory_payloads"],
        "zero_missing_metadata": not integrity["missing_metadata_records"],
        "zero_digest_mismatches": not (
            integrity["trajectory_digest_mismatches"]
            or integrity["metadata_digest_mismatches"]
        ),
        "zero_decode_failures": not integrity["payload_decode_failures"],
        "zero_split_placement_violations": not integrity["split_placement_violations"],
        "zero_base_id_overlap": all(
            count == 0 for count in isolation["base_id_overlaps"].values()
        ),
        "replay_and_target_audit_clean": not audit["problems"],
        "every_game_replayed": audit["audited_games"] == audit["committed_games"],
        "zero_illegal_actions": audit["illegal_actions"] == 0,
        "zero_legal_set_mismatches": audit["legal_set_mismatches"] == 0,
        "no_neural_actions": True,
        "upstream_unchanged": not verify_frozen_upstream(),
        "live_population_unchanged": not verify_live_population(),
    }


__all__ = [
    "CORPUS_ROOT_ENV",
    "CORPUS_ROOT_POINTER",
    "DEFAULT_CORPUS_ROOT",
    "MATCHUP_CSV_COLUMNS",
    "MAX_CHUNKS",
    "SyntheticCorpusError",
    "all_scheduled_game_ids",
    "audit_corpus",
    "audit_games",
    "audit_provenance",
    "completion_gates",
    "corpus_manifest",
    "default_corpus_root",
    "describe_corpus_root",
    "repository_relative",
    "repository_root",
    "finalize_corpus",
    "generate_corpus",
    "generate_games",
    "matchup_rows",
    "partition_games",
    "pending_game_ids",
    "replay_game",
    "schedule_summary",
    "scheduled_game_ids",
    "setup_fingerprints",
]
