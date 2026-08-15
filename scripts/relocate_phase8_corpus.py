#!/usr/bin/env python3
"""Relocate the accepted Phase 8 corpus without regenerating it.

The Phase 8 common contract allows the corpus root to be redirected by
configuration and requires the manifest to record where the bytes actually
live. This moves an already-accepted `synthetic_warmstart_corpus_v1` to a new
root and proves the move changed nothing.

Copy, verify, then remove — in that order. The source is deleted only after the
destination reproduces the accepted content, metadata and commit-index digests
and the full set of committed game identities, so an interrupted relocation
leaves the corpus intact at the old root rather than half-present at both.

Usage::

    python scripts/relocate_phase8_corpus.py --destination /Volumes/Disk/path
    python scripts/relocate_phase8_corpus.py --destination ... --keep-source
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.training import synthetic_corpus as sc  # noqa: E402
from stratego.training.corpus_commit import (  # noqa: E402
    CorpusReader,
    audit_commit_integrity,
    corpus_content_digest,
)
from stratego.training.warmstart_seed import CORPUS_SPLITS  # noqa: E402


def digests(root: Path) -> dict:
    """The three identity digests plus the committed-id set of one root."""
    reader = CorpusReader(root, CORPUS_SPLITS)
    return {
        "content_digest": corpus_content_digest(root, CORPUS_SPLITS),
        "metadata_digest": sc._metadata_digest(root, CORPUS_SPLITS),
        "commit_index_digest": sc._commit_index_digest(root, CORPUS_SPLITS),
        "committed_games": len(reader),
        "game_ids_digest": _game_ids_digest(reader),
        "per_split": {split: len(reader.game_ids(split)) for split in CORPUS_SPLITS},
    }


def _game_ids_digest(reader: CorpusReader) -> str:
    import hashlib

    hasher = hashlib.sha256()
    for game_id in reader.game_ids():
        hasher.update(f"{game_id}\n".encode())
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None, help="defaults to the resolved root")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--keep-source", action="store_true")
    parser.add_argument(
        "--no-pointer",
        action="store_true",
        help="do not record the new root in the pointer file",
    )
    arguments = parser.parse_args()

    source = Path(arguments.source) if arguments.source else sc.default_corpus_root()
    destination = Path(arguments.destination).expanduser()
    if not source.exists():
        print(f"no corpus at {source}")
        return 2
    if destination.exists() and any(destination.iterdir()):
        print(f"destination {destination} already exists and is not empty")
        return 2

    print(f"source      {source}")
    print(f"destination {destination}")

    started = time.perf_counter()
    before = digests(source)
    print(f"source digests read in {time.perf_counter() - started:.1f}s: "
          f"{before['committed_games']} committed games")

    started = time.perf_counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    copy_seconds = time.perf_counter() - started
    copied_bytes = sum(
        path.stat().st_size for path in destination.rglob("*") if path.is_file()
    )
    print(f"copied {copied_bytes / 1e6:.1f} MB in {copy_seconds:.1f}s")

    started = time.perf_counter()
    after = digests(destination)
    integrity = audit_commit_integrity(destination, CORPUS_SPLITS)
    verify_seconds = time.perf_counter() - started

    identical = all(before[key] == after[key] for key in before)
    clean = not any(
        integrity[key]
        for key in (
            "duplicate_committed_ids",
            "orphan_trajectory_records",
            "orphan_metadata_records",
            "missing_trajectory_payloads",
            "missing_metadata_records",
            "trajectory_digest_mismatches",
            "metadata_digest_mismatches",
            "payload_decode_failures",
            "split_placement_violations",
        )
    )
    print(f"verified in {verify_seconds:.1f}s: digests identical={identical} integrity_clean={clean}")
    for key in before:
        if before[key] != after[key]:
            print(f"  MISMATCH {key}: {before[key]!r} -> {after[key]!r}")

    if not (identical and clean):
        print("relocation NOT confirmed; the source has been left untouched")
        return 1

    if not arguments.no_pointer:
        pointer = REPOSITORY_ROOT / sc.CORPUS_ROOT_POINTER
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(str(destination) + "\n")
        print(f"recorded redirect in {pointer}")

    removed = False
    if not arguments.keep_source:
        shutil.rmtree(source)
        removed = True
        print(f"removed source {source}")

    report = {
        "source": str(source),
        "destination": str(destination),
        "copied_bytes": copied_bytes,
        "copy_seconds": round(copy_seconds, 3),
        "verify_seconds": round(verify_seconds, 3),
        "digests_before": before,
        "digests_after": after,
        "digests_identical": identical,
        "integrity_clean": clean,
        "source_removed": removed,
        "pointer_written": not arguments.no_pointer,
    }
    print(json.dumps({key: report[key] for key in ("digests_identical", "integrity_clean", "source_removed")}))
    (REPOSITORY_ROOT / "reports" / "phase_8_data" / "agent_02_relocation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
