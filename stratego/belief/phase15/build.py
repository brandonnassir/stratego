"""Phase 15 Agent 1: the corpus driver.

Specification source: `01_AGENT_1_BELIEF_HEAD_TRAINING.md` sections 5-7.

Position-budgeted, not game-budgeted
------------------------------------
:func:`build_split` consumes plans in cycle order until the split's
*position* budget is met, then stops at the next game boundary. The plan
cycle is a deterministically permuted realisation of the exact section 6
mixture, so a run that stops early stops on a balanced prefix.

Parallel, but written in plan order
-----------------------------------
Games are played in worker processes — a game is a pure function of its
plan, so a worker needs nothing but the plan and its own frozen model
copies. The driver consumes completed batches **in submission order** and
writes them through a single :class:`~.storage.SplitWriter`, so the stored
bytes are identical to what a single process would have written, and the
corpus digest does not depend on the worker count.

CPU, deliberately
-----------------
A pilot measured the observer's single-request forward pass at roughly
2x faster on CPU than on MPS: at batch size one the Metal dispatch cost
dominates the 864k-parameter model. The corpus is therefore generated on
CPU, with one torch thread per worker so eight workers do not fight over
the same ten performance cores.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from .contract import (
    CORPUS_SPLITS,
    CORPUS_VERSION,
    DECISIONS_PER_GAME,
    LIBRARY_SPLIT,
    POSITION_FLOOR,
    POSITION_TARGET,
    PHASE15_STATUS_MARKERS,
    Phase15Error,
)
from .corpus import (
    CORPUS_RUN_VERSION,
    DECISION_MODE,
    iter_plans,
    play_corpus_game,
    privileged_extract,
    select_decisions,
)
from .seeds import CANONICAL_PHASE15_SEEDS, PHASE15_IDENTITY_VERSION
from .setups import Phase15SetupSources
from .storage import (
    CORPUS_FORMAT_VERSION,
    SplitWriter,
    corpus_digest,
    label_names,
    split_digest,
    write_manifest,
)

#: How many games one worker task plays. Large enough to amortise the task
#: hand-off, small enough that the driver never holds much in memory.
BATCH_GAMES = 24

#: Worker-process state, built once by :func:`_initialise_worker`.
_WORKER: dict = {}


class Phase15BuildError(Phase15Error):
    """A corpus split could not be generated."""


def _initialise_worker(checkpoint_paths: dict, threads: int = 1) -> None:
    """Build one worker's frozen inference owners and setup sources."""
    import torch

    from ...evaluation.neural_worker import InferenceOwner
    from ...model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config

    torch.set_num_threads(int(threads))
    owners = {
        source_id: InferenceOwner(
            path,
            decision_mode=DECISION_MODE,
            device="cpu",
            dtype="float32",
            expected_architecture_id=ARCHITECTURE_FAMILY,
            expected_configuration=candidate_config("C1"),
            name=f"phase15_{source_id}",
        )
        for source_id, path in checkpoint_paths.items()
    }
    _WORKER["owners"] = owners
    _WORKER["sources"] = Phase15SetupSources()


def _play_batch(plans: "list") -> "list[tuple]":
    """Play one batch of plans. Returns `(plan, result, eligible, samples)`."""
    owners = _WORKER["owners"]
    played = []
    for plan in plans:
        result, decisions = play_corpus_game(plan, owners)
        eligible = sum(1 for row in decisions if row["unresolved"] > 0)
        selected = select_decisions(decisions, DECISIONS_PER_GAME)
        extracted = privileged_extract(plan, result, selected, eligible)
        played.append((plan, result, eligible, extracted.samples))
    return played


def _batches(plans_iterator, size: int, count: int) -> "list[list]":
    """Take `count` batches of `size` plans, stopping if the stream ends."""
    batches = []
    for _ in range(count):
        batch = []
        for _ in range(size):
            try:
                batch.append(next(plans_iterator))
            except StopIteration:  # pragma: no cover - the stream is unbounded
                break
        if not batch:
            break
        batches.append(batch)
    return batches


def build_split(
    root: "Path | str",
    split: str,
    checkpoint_paths: dict,
    *,
    target_positions: "int | None" = None,
    workers: int = 8,
    progress=None,
) -> tuple:
    """Play and store one split until its position budget is met.

    Returns `(manifest block, seconds)`. The block records the achieved
    position count, which is what section 6 asks a report to state — the
    intended game count is not evidence of anything.
    """
    from concurrent.futures import ProcessPoolExecutor

    if split not in CORPUS_SPLITS:
        raise Phase15BuildError(f"unknown split {split!r}")
    target = int(POSITION_TARGET[split] if target_positions is None else target_positions)
    if target < 1:
        raise Phase15BuildError(f"target must be positive, got {target}")

    sources = Phase15SetupSources()
    plans = iter_plans(split, sources)
    writer = SplitWriter(root, split)
    started = time.perf_counter()
    workers = max(1, int(workers))

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialise_worker,
        initargs=(dict(checkpoint_paths), 1),
    ) as pool:
        while writer.samples < target:
            batches = _batches(plans, BATCH_GAMES, workers)
            if not batches:  # pragma: no cover - the stream is unbounded
                break
            futures = [pool.submit(_play_batch, batch) for batch in batches]
            for future in futures:
                played = future.result()
                if writer.samples >= target:
                    # The budget was met by an earlier future of this round.
                    # The remaining results are discarded rather than stored,
                    # so the split stops on the *first* game that meets the
                    # budget however many workers happened to be in flight.
                    break
                for plan, result, eligible, samples in played:
                    writer.add_game(plan, result, eligible, samples)
                    if writer.samples >= target:
                        break
                if progress is not None:
                    progress(
                        split,
                        writer.samples,
                        target,
                        time.perf_counter() - started,
                    )

    block = writer.close()
    block["target_positions"] = target
    block["floor_positions"] = int(POSITION_FLOOR[split])
    block["met_target"] = bool(block["samples"] >= target)
    block["library_split"] = LIBRARY_SPLIT[split]
    block["file_digests"] = split_digest(root, split)
    return block, round(time.perf_counter() - started, 3)


def build_corpus(
    root: "Path | str",
    checkpoint_paths: dict,
    sources_identity: dict,
    orientation_evidence: dict,
    *,
    targets: "dict | None" = None,
    workers: int = 8,
    overwrite: bool = False,
    progress=None,
) -> dict:
    """Build every split and write the corpus manifest.

    Refuses to overwrite an existing corpus unless asked: the trained
    specialists, the calibration temperatures and the search handoff are
    all bound to these exact bytes.
    """
    import shutil

    root = Path(root)
    if root.exists():
        if not overwrite:
            raise Phase15BuildError(
                f"{root} already holds a corpus; pass overwrite=True to replace it"
            )
        shutil.rmtree(root)
    root.mkdir(parents=True)

    if not orientation_evidence.get("passed"):
        raise Phase15BuildError(
            "the section 4 orientation gate has not passed; no corpus generation "
            "may begin"
        )

    targets = dict(targets or {})
    splits: dict = {}
    durations: dict = {}
    for split in CORPUS_SPLITS:
        block, seconds = build_split(
            root,
            split,
            checkpoint_paths,
            target_positions=targets.get(split),
            workers=workers,
            progress=progress,
        )
        splits[split] = block
        durations[split] = seconds

    manifest = {
        "corpus_version": CORPUS_VERSION,
        "corpus_format_version": CORPUS_FORMAT_VERSION,
        "run_version": CORPUS_RUN_VERSION,
        "identity_version": PHASE15_IDENTITY_VERSION,
        "seeds": dict(CANONICAL_PHASE15_SEEDS),
        "decision_mode": DECISION_MODE,
        "decisions_per_game_cap": DECISIONS_PER_GAME,
        "termination_cap": {
            "rules": "EVALUATION_RULES, unchanged",
            "battleless_move_limit": 200,
            "absolute_move_limit": 4000,
            "trajectory_retirement": (
                "not used: each game plays to its accepted termination, because "
                "evenly spaced sampling is defined over the complete eligible list"
            ),
        },
        "label_codes": label_names(),
        "policy_sources": sources_identity,
        "orientation": orientation_evidence,
        "splits": splits,
        **PHASE15_STATUS_MARKERS,
    }
    manifest["corpus_digest"] = corpus_digest(manifest)
    # Durations are attached after the digest, so the corpus identity never
    # embeds a wall clock.
    manifest["generation_seconds"] = durations
    manifest["workers"] = int(workers)
    manifest["host_cpus"] = os.cpu_count()
    write_manifest(root, manifest)
    return manifest


__all__ = ["BATCH_GAMES", "Phase15BuildError", "build_corpus", "build_split"]
