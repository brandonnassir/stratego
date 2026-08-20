"""Phase 11B Part 0 driver: play the corpus and write it once.

Specification source: `01_AGENT_1_ATTACHED_BELIEF_HEAD.md` ("Part 0 — Build
the Common Phase 11B Corpus").

One function, two splits
------------------------
:func:`build_corpus` plays both splits with one pair of inference owners
and writes both stores, then digests the result. It is the only writer of
`data/phase11b/common_corpus_v1`, and it refuses to overwrite an existing
corpus unless asked — Agents 2-5 are promised these exact bytes.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from .contract import CORPUS_SPLITS, CORPUS_VERSION, PHASE11B_STATUS_MARKERS
from .corpus import (
    CORPUS_RUN_VERSION,
    Phase11BCorpusError,
    Phase11BSetupSources,
    corpus_plans,
    play_corpus_game,
    privileged_extract,
    select_decisions,
)
from .seeds import CANONICAL_PHASE11B_SEEDS, PHASE11B_IDENTITY_VERSION
from .storage import (
    CORPUS_FORMAT_VERSION,
    SplitWriter,
    corpus_digest,
    split_digest,
    write_manifest,
)


def build_split(
    root: "Path | str",
    split: str,
    owners: dict,
    sources: "Phase11BSetupSources | None" = None,
    *,
    limit: "int | None" = None,
    progress=None,
) -> tuple:
    """Play and store one split. Returns `(manifest block, seconds)`."""
    if split not in CORPUS_SPLITS:
        raise Phase11BCorpusError(f"unknown split {split!r}")
    per_game = int(CORPUS_SPLITS[split]["decisions_per_game"])
    plans = corpus_plans(split, sources, limit=limit)
    writer = SplitWriter(root, split)
    started = time.perf_counter()
    for position, plan in enumerate(plans):
        result, decisions = play_corpus_game(plan, owners)
        selected = select_decisions(decisions, per_game)
        extracted = privileged_extract(plan, result, selected)
        writer.add_game(plan, result, decisions, extracted.samples)
        if progress is not None and (position + 1) % 128 == 0:
            progress(split, position + 1, len(plans), writer.samples, time.perf_counter() - started)
    block = writer.close()
    block["file_digests"] = split_digest(root, split)
    return block, round(time.perf_counter() - started, 3)


def build_corpus(
    root: "Path | str",
    owners: dict,
    *,
    limits: "dict | None" = None,
    overwrite: bool = False,
    progress=None,
) -> dict:
    """Build both splits and write the corpus manifest.

    `limits` maps a split to a game count, for the throughput pilot only. A
    limited corpus is marked incomplete in its manifest and its digest
    differs from the full corpus's, so a pilot can never be mistaken for
    the artifact Agents 2-5 reuse.
    """
    root = Path(root)
    if root.exists():
        if not overwrite:
            raise Phase11BCorpusError(
                f"{root} already holds a corpus; pass overwrite=True to replace it"
            )
        shutil.rmtree(root)
    root.mkdir(parents=True)

    sources = Phase11BSetupSources()
    limits = dict(limits or {})
    splits: dict = {}
    durations: dict = {}
    for split in CORPUS_SPLITS:
        block, seconds = build_split(
            root, split, owners, sources, limit=limits.get(split), progress=progress
        )
        splits[split] = block
        durations[split] = seconds

    manifest = {
        "corpus_version": CORPUS_VERSION,
        "corpus_format_version": CORPUS_FORMAT_VERSION,
        "run_version": CORPUS_RUN_VERSION,
        "identity_version": PHASE11B_IDENTITY_VERSION,
        "seeds": dict(CANONICAL_PHASE11B_SEEDS),
        "splits": splits,
        **PHASE11B_STATUS_MARKERS,
    }
    manifest["corpus_digest"] = corpus_digest(manifest)
    # Durations are attached after the digest, so the corpus identity never
    # embeds a wall clock (the Phase 11 `manifest_digest` defect).
    manifest["generation_seconds"] = durations
    write_manifest(root, manifest)
    return manifest


__all__ = ["build_corpus", "build_split"]
