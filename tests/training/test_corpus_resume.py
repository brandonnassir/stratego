"""Phase 8 Agent 2: crash, restart and reconciliation of the synthetic corpus.

Specification sources:

- `02_AGENT_2_SYNTHETIC_CORPUS.md` ("Crash-safe commit design", "Required
  injected-crash tests")
- `00_PHASE_8_SEQUENCE_AND_COMMON_CONTRACT.md` section 24

The property under test throughout is a single sentence: **a game is visible
only if it is committed.** Every test here interrupts a run somewhere, resumes
it, and requires the finished corpus to be the one an uninterrupted run would
have produced — same ids, same bytes, no duplicates, no orphans.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from stratego.training import synthetic_corpus as sc
from stratego.training.corpus_commit import (
    CRASH_STAGES,
    JOURNAL_SUFFIX,
    METADATA_SUFFIX,
    CorpusCommitError,
    CorpusReader,
    audit_commit_integrity,
    corpus_content_digest,
    journal_directory,
    metadata_directory,
    read_journal,
    reconcile_corpus,
    shards_directory,
)
from stratego.training.warmstart_seed import CORPUS_SPLITS, synthetic_game_id

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Six games from three ordered cells. Small enough to run repeatedly, varied
#: enough that a shard rollover can be provoked with a small target size.
CRASH_CELLS = (
    ("tactical_rule_based@1.0.0", "basic_heuristic@1.0.0"),
    ("basic_heuristic@1.0.0", "stress_scout_rush@1.0.0"),
)


def crash_game_ids(count: int = 6) -> tuple:
    ids = tuple(
        synthetic_game_id("train", red, blue, ordinal)
        for ordinal in range(count)
        for red, blue in CRASH_CELLS
    )
    return ids[:count]


class Interrupt(RuntimeError):
    """Stands in for the process dying at a named point of the commit protocol."""


def crash_after(stage: str, games: int):
    """A hook that raises the moment the run reaches `stage` of game `games`."""
    seen = {"commits": 0}

    def hook(current_stage: str, _writer) -> None:
        if current_stage == "after_commit":
            seen["commits"] += 1
            return
        if current_stage == stage and seen["commits"] == games:
            raise Interrupt(f"interrupted at {stage} of game {games}")

    return hook


def clean_corpus(root: Path, game_ids: tuple, **kwargs) -> str:
    sc.generate_corpus(root, worker_count=1, chunks_per_worker=1, game_ids=game_ids, **kwargs)
    return corpus_content_digest(root, CORPUS_SPLITS)


# ---------------------------------------------------------------------------
# Injected crashes at every stage of the commit protocol
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stage",
    [stage for stage in CRASH_STAGES if stage not in ("after_commit", "shard_rollover")],
)
def test_a_crash_at_any_commit_stage_resumes_to_the_same_corpus(tmp_path, stage):
    game_ids = crash_game_ids(6)
    expected = clean_corpus(tmp_path / "clean", game_ids)

    root = tmp_path / "crashed"
    with pytest.raises(Interrupt):
        sc.generate_games(
            root,
            game_ids,
            segment=0,
            worker_id=0,
            crash_hook=crash_after(stage, games=3),
        )

    integrity = audit_commit_integrity(root, CORPUS_SPLITS)
    assert integrity["committed_count"] == 3
    # Before reconciliation an interrupted stage may have left a payload or a
    # metadata line with no commit. Neither may ever be visible.
    reader = CorpusReader(root, CORPUS_SPLITS)
    assert len(reader) == 3
    assert set(reader.game_ids()) <= set(game_ids)

    sc.generate_corpus(root, worker_count=1, chunks_per_worker=1, game_ids=game_ids)
    assert corpus_content_digest(root, CORPUS_SPLITS) == expected

    final = audit_commit_integrity(root, CORPUS_SPLITS)
    assert final["committed_count"] == len(game_ids)
    assert final["duplicate_committed_ids"] == []
    assert final["orphan_trajectory_records"] == []
    assert final["orphan_metadata_records"] == []
    assert final["missing_trajectory_payloads"] == []
    assert final["missing_metadata_records"] == []
    assert final["trajectory_digest_mismatches"] == []
    assert final["metadata_digest_mismatches"] == []


def test_a_crash_at_a_shard_rollover_resumes_to_the_same_corpus(tmp_path):
    game_ids = crash_game_ids(6)
    expected = clean_corpus(tmp_path / "clean", game_ids, target_bytes=4096)

    root = tmp_path / "crashed"
    rollovers = {"count": 0}

    def hook(stage: str, _writer) -> None:
        if stage == "shard_rollover":
            rollovers["count"] += 1
            if rollovers["count"] == 2:
                raise Interrupt("interrupted at a shard rollover")

    with pytest.raises(Interrupt):
        sc.generate_games(
            root, game_ids, segment=0, worker_id=0, target_bytes=4096, crash_hook=hook
        )
    assert rollovers["count"] == 2

    sc.generate_corpus(
        root, worker_count=1, chunks_per_worker=1, game_ids=game_ids, target_bytes=4096
    )
    assert corpus_content_digest(root, CORPUS_SPLITS) == expected
    integrity = audit_commit_integrity(root, CORPUS_SPLITS)
    assert integrity["orphan_trajectory_records"] == []
    assert integrity["committed_count"] == len(game_ids)
    assert len(list(shards_directory(root, "train").glob("*.stgshard"))) > 1


def test_reconciliation_discards_uncommitted_bytes(tmp_path):
    game_ids = crash_game_ids(4)
    root = tmp_path / "crashed"
    with pytest.raises(Interrupt):
        sc.generate_games(
            root,
            game_ids,
            segment=0,
            worker_id=0,
            crash_hook=crash_after("after_metadata", games=2),
        )
    metadata_path = next(metadata_directory(root, "train").glob(f"*{METADATA_SUFFIX}"))
    shard_path = next(shards_directory(root, "train").glob("*.stgshard"))
    before = (metadata_path.stat().st_size, shard_path.stat().st_size)

    report = reconcile_corpus(root, CORPUS_SPLITS)
    after = (metadata_path.stat().st_size, shard_path.stat().st_size)
    assert after[0] < before[0], "the uncommitted metadata line must be removed"
    assert after[1] < before[1], "the uncommitted trajectory payload must be removed"
    assert report["committed_count"] == 2
    assert report["bytes_discarded"] > 0
    assert audit_commit_integrity(root, CORPUS_SPLITS)["orphan_metadata_records"] == []


def test_a_torn_journal_line_is_dropped(tmp_path):
    game_ids = crash_game_ids(3)
    root = tmp_path / "torn"
    sc.generate_corpus(root, worker_count=1, chunks_per_worker=1, game_ids=game_ids)
    journal_path = next(journal_directory(root, "train").glob(f"*{JOURNAL_SUFFIX}"))
    raw = journal_path.read_bytes()
    # A process killed mid-line leaves a prefix with no newline.
    journal_path.write_bytes(raw + raw.splitlines()[-1][:40])

    commits, valid_bytes = read_journal(journal_path)
    assert len(commits) == 3
    assert valid_bytes == len(raw)
    assert len(CorpusReader(root, CORPUS_SPLITS)) == 3
    reconcile_corpus(root, CORPUS_SPLITS)
    assert journal_path.stat().st_size == len(raw)


def test_a_commit_from_another_protocol_version_is_refused(tmp_path):
    root = tmp_path / "foreign"
    sc.generate_corpus(
        root, worker_count=1, chunks_per_worker=1, game_ids=crash_game_ids(1)
    )
    journal_path = next(journal_directory(root, "train").glob(f"*{JOURNAL_SUFFIX}"))
    line = json.loads(journal_path.read_text().splitlines()[0])
    line["commit_version"] = "warmstart_corpus_commit_v99"
    journal_path.write_text(json.dumps(line) + "\n")
    with pytest.raises(CorpusCommitError, match="commit protocol"):
        read_journal(journal_path)


# ---------------------------------------------------------------------------
# Process restart
# ---------------------------------------------------------------------------

_CHILD_SCRIPT = """
import os, sys
sys.path.insert(0, {repository!r})
from stratego.training import synthetic_corpus as sc

def hook(stage, writer):
    if stage == "after_trajectory" and writer.games_written == {kill_after}:
        # No unwinding, no flush, no close: the process is simply gone.
        os._exit(9)

sc.generate_games({root!r}, {game_ids!r}, segment=0, worker_id=0, crash_hook=hook)
"""


def test_a_killed_process_resumes_to_the_same_corpus(tmp_path):
    game_ids = crash_game_ids(5)
    expected = clean_corpus(tmp_path / "clean", game_ids)

    root = tmp_path / "killed"
    script = _CHILD_SCRIPT.format(
        repository=str(REPOSITORY_ROOT),
        root=str(root),
        game_ids=list(game_ids),
        kill_after=2,
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert completed.returncode == 9, completed.stderr

    # The killed process left an uncommitted trajectory payload behind; it must
    # be invisible before reconciliation and gone after it.
    assert len(CorpusReader(root, CORPUS_SPLITS)) == 2
    sc.generate_corpus(root, worker_count=1, chunks_per_worker=1, game_ids=game_ids)
    assert corpus_content_digest(root, CORPUS_SPLITS) == expected

    integrity = audit_commit_integrity(root, CORPUS_SPLITS)
    assert integrity["committed_count"] == len(game_ids)
    assert integrity["duplicate_committed_ids"] == []
    assert integrity["orphan_trajectory_records"] == []
    assert integrity["orphan_metadata_records"] == []


# ---------------------------------------------------------------------------
# Resume boundaries, worker counts and repeated runs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("first_batch", [1, 3, 5])
def test_the_resume_boundary_does_not_change_the_corpus(tmp_path, first_batch):
    game_ids = crash_game_ids(6)
    expected = clean_corpus(tmp_path / "clean", game_ids)

    root = tmp_path / f"staged_{first_batch}"
    sc.generate_corpus(
        root, worker_count=1, chunks_per_worker=1, game_ids=game_ids, limit=first_batch
    )
    assert len(CorpusReader(root, CORPUS_SPLITS)) == first_batch
    sc.generate_corpus(root, worker_count=1, chunks_per_worker=1, game_ids=game_ids)
    assert corpus_content_digest(root, CORPUS_SPLITS) == expected
    assert len(CorpusReader(root, CORPUS_SPLITS)) == len(game_ids)


def test_regenerating_a_complete_corpus_is_a_no_op(tmp_path):
    game_ids = crash_game_ids(3)
    root = tmp_path / "complete"
    sc.generate_corpus(root, worker_count=1, chunks_per_worker=1, game_ids=game_ids)
    digest = corpus_content_digest(root, CORPUS_SPLITS)
    again = sc.generate_corpus(
        root, worker_count=1, chunks_per_worker=1, game_ids=game_ids
    )
    assert again["pending"] == 0
    assert again["games_generated"] == 0
    assert corpus_content_digest(root, CORPUS_SPLITS) == digest
    assert audit_commit_integrity(root, CORPUS_SPLITS)["duplicate_committed_ids"] == []


def test_a_resumed_run_never_regenerates_a_committed_game(tmp_path):
    game_ids = crash_game_ids(6)
    root = tmp_path / "resumed"
    sc.generate_corpus(
        root, worker_count=1, chunks_per_worker=1, game_ids=game_ids, limit=4
    )
    committed_before = dict(reconcile_corpus(root, CORPUS_SPLITS)["committed"])
    result = sc.generate_corpus(
        root, worker_count=1, chunks_per_worker=1, game_ids=game_ids
    )
    assert result["already_committed"] == 4
    assert result["pending"] == 2
    assert result["games_generated"] == 2
    committed_after = dict(reconcile_corpus(root, CORPUS_SPLITS)["committed"])
    for game_id, commit in committed_before.items():
        assert committed_after[game_id].trajectory_sha256 == commit.trajectory_sha256
        assert committed_after[game_id].shard_name == commit.shard_name


def test_parallel_and_serial_runs_agree(tmp_path):
    game_ids = crash_game_ids(6)
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    clean_corpus(serial, game_ids)
    sc.generate_corpus(parallel, worker_count=3, chunks_per_worker=2, game_ids=game_ids)
    assert corpus_content_digest(parallel, CORPUS_SPLITS) == corpus_content_digest(
        serial, CORPUS_SPLITS
    )
    left = CorpusReader(serial, CORPUS_SPLITS)
    right = CorpusReader(parallel, CORPUS_SPLITS)
    assert left.game_ids() == right.game_ids()
    for game_id in left.game_ids():
        assert left.record(game_id).actions == right.record(game_id).actions
    integrity = audit_commit_integrity(parallel, CORPUS_SPLITS)
    assert integrity["duplicate_committed_ids"] == []
    assert integrity["orphan_trajectory_records"] == []


def test_a_crashed_and_a_clean_corpus_finalize_identically(tmp_path):
    game_ids = crash_game_ids(6)
    clean = tmp_path / "clean"
    crashed = tmp_path / "crashed"
    clean_corpus(clean, game_ids)

    with pytest.raises(Interrupt):
        sc.generate_games(
            crashed,
            game_ids,
            segment=0,
            worker_id=0,
            crash_hook=crash_after("before_commit_flush", games=2),
        )
    sc.generate_corpus(crashed, worker_count=2, chunks_per_worker=1, game_ids=game_ids)

    clean_result = sc.finalize_corpus(clean, worker_count=1, observation_plies=2)
    crashed_result = sc.finalize_corpus(crashed, worker_count=1, observation_plies=2)
    assert clean_result["audit"]["problems"] == []
    assert crashed_result["audit"]["problems"] == []
    assert (
        crashed_result["manifest"]["content_digest"]
        == clean_result["manifest"]["content_digest"]
    )
    assert (
        crashed_result["manifest"]["metadata_digest"]
        == clean_result["manifest"]["metadata_digest"]
    )
    assert (
        crashed_result["manifest"]["commit_index_digest"]
        == clean_result["manifest"]["commit_index_digest"]
    )
