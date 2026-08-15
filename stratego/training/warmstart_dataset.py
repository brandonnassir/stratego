"""Phase 8 Agent 3: the deterministic streaming dataset over the corpus.

Specification sources:

- `03_AGENT_3_TRAINING_EXAMPLES_AND_TARGETS.md` ("Mission", "Deterministic
  shuffle / cursor", "Dataset throughput")
- `00_PHASE_8_SEQUENCE_AND_COMMON_CONTRACT.md` sections 14-15, 23 (decision
  sampler, example schema, data-cursor resume)

Identity before machinery
-------------------------
The dataset's whole contract is that every quantity is a pure function of
frozen identities:

```text
universe   = f(frozen schedule, commit journals, decision sampler)
order      = f(universe length, train_order_seed(epoch))          [shuffle]
           = identity                                             [sequential]
batch b    = order[b*B : (b+1)*B]
```

Workers, prefetch depth and completion timing appear nowhere in those
equations, so they cannot change which logical batch is which. The parallel
loader assigns *batches* to workers and yields results strictly in submission
order; a worker that finishes early waits in the reassembly queue rather than
jumping the line.

The model-input boundary
------------------------
:class:`WarmstartBatch` separates the one tensor the network may consume
(`observations`) from everything else (`targets`, identity metadata). The
model forward call receives `batch.model_input()` — a bare tensor with no
Python references back to records, metadata, labels or identities — which is
what the object-graph regression in
:mod:`tests.information_security.test_warmstart_target_boundary` proves.
"""

from __future__ import annotations

import hashlib
import resource
import statistics
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ..model.contract import OBSERVATION_SHAPE
from .corpus_commit import CorpusReader
from .synthetic_corpus import default_corpus_root
from .warmstart_contract import WARMSTART_EXAMPLE_VERSION, iter_game_identities
from .warmstart_examples import WarmstartExample, examples_for_game
from .warmstart_seed import (
    CORPUS_SPLITS,
    DECISION_SAMPLER_VERSION,
    selected_decision_indices,
    train_order_seed,
)

#: The deterministic train-order contract: one permutation per epoch drawn from
#: the frozen `train_order` stream, batches as contiguous slices, never crossing
#: an epoch boundary. A change to any of the three is a new version.
TRAIN_ORDER_VERSION = "warmstart_train_order_v1"

#: The resume-cursor schema carried inside `warmstart_checkpoint_v1`.
DATA_CURSOR_VERSION = "warmstart_data_cursor_v1"

#: The frozen Phase 8 batch size (common contract section 19).
DEFAULT_BATCH_SIZE = 256

#: Decoded records kept per process. Selected decisions cluster 62-to-a-game,
#: so even a shuffled epoch revisits games; the cache turns some of those
#: revisits into hits without holding more than ~tens of MB.
DEFAULT_RECORD_CACHE = 512

ORDER_SHUFFLE = "shuffle"
ORDER_SEQUENTIAL = "sequential"


class WarmstartDatasetError(RuntimeError):
    """The dataset could not enumerate, order or reconstruct as contracted."""


# ---------------------------------------------------------------------------
# The selected-example universe
# ---------------------------------------------------------------------------


def selected_example_universe(
    reader: CorpusReader, split: str, *, require_complete: bool = True
) -> tuple:
    """Every `(game_id, decision_index)` of one split, frozen order.

    Games follow the frozen schedule order (cell-major, then ordinal); inside
    a game the sampler's indices are ascending. The decision count comes from
    the commit journal, so enumerating the universe never decodes a payload,
    and the same manifest always yields the same ordered universe.

    `require_complete=True` (the production default) refuses a corpus missing
    any scheduled game: training on a silently shrunken universe would change
    the run's identity. Tests over deliberately partial mini corpora pass
    `False`, which keeps the schedule order and simply skips absent games.
    """
    if split not in CORPUS_SPLITS:
        raise WarmstartDatasetError(f"unknown corpus split {split!r}")
    keys: list = []
    for _cell, _red, _blue, _ordinal, game_id in iter_game_identities(split):
        commit = reader.commits.get(game_id)
        if commit is None:
            if require_complete:
                raise WarmstartDatasetError(
                    f"scheduled game {game_id!r} is not committed; the corpus is "
                    "not the finalized one"
                )
            continue
        for index in selected_decision_indices(game_id, commit.total_decisions):
            keys.append((game_id, index))
    return tuple(keys)


def universe_digest(universe: tuple) -> str:
    """SHA-256 over the ordered `(game_id, decision_index)` universe."""
    hasher = hashlib.sha256()
    for game_id, index in universe:
        hasher.update(f"{game_id}|{index}\n".encode())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Deterministic order and the resume cursor
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _shuffle_permutation(size: int, epoch: int) -> np.ndarray:
    permutation = np.random.default_rng(train_order_seed(epoch)).permutation(size)
    permutation.setflags(write=False)
    return permutation


def epoch_order(size: int, epoch: int, order: str = ORDER_SHUFFLE) -> np.ndarray:
    """The example order of one epoch: a frozen shuffle or the identity."""
    if size < 0:
        raise WarmstartDatasetError(f"universe size must be >= 0, got {size}")
    if order == ORDER_SHUFFLE:
        return _shuffle_permutation(int(size), int(epoch))
    if order == ORDER_SEQUENTIAL:
        identity = np.arange(int(size), dtype=np.int64)
        identity.setflags(write=False)
        return identity
    raise WarmstartDatasetError(f"unknown order kind {order!r}")


@dataclass(frozen=True)
class DataCursor:
    """The exact position of a training run inside the example stream.

    `position` indexes the epoch's order array and names the *next* example to
    serve. Restoring a cursor therefore reproduces the exact next batch, which
    is the property Agent 4's checkpoint/resume proof leans on.
    """

    split: str
    batch_size: int = DEFAULT_BATCH_SIZE
    epoch: int = 0
    position: int = 0
    order: str = ORDER_SHUFFLE
    cursor_version: str = DATA_CURSOR_VERSION
    example_version: str = WARMSTART_EXAMPLE_VERSION
    order_version: str = TRAIN_ORDER_VERSION
    sampler_version: str = DECISION_SAMPLER_VERSION

    def to_dict(self) -> dict:
        return {
            "cursor_version": self.cursor_version,
            "example_version": self.example_version,
            "order_version": self.order_version,
            "sampler_version": self.sampler_version,
            "split": self.split,
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "position": self.position,
            "order": self.order,
        }

    @staticmethod
    def from_dict(payload: dict) -> "DataCursor":
        cursor = DataCursor(
            split=str(payload["split"]),
            batch_size=int(payload["batch_size"]),
            epoch=int(payload["epoch"]),
            position=int(payload["position"]),
            order=str(payload["order"]),
        )
        for field in ("cursor_version", "example_version", "order_version", "sampler_version"):
            if payload.get(field) != getattr(cursor, field):
                raise WarmstartDatasetError(
                    f"cursor {field} {payload.get(field)!r} does not match the live "
                    f"{getattr(cursor, field)!r}; refusing to resume across versions"
                )
        return cursor


def plan_batch(universe: tuple, cursor: DataCursor) -> tuple:
    """`(keys, cursor_after)` of the batch the cursor points at.

    Batches are contiguous slices of the epoch order and never span an epoch
    boundary: the final short slice of an epoch is its own batch, after which
    the next epoch begins at position zero with its own frozen permutation.
    """
    size = len(universe)
    if size == 0:
        raise WarmstartDatasetError("the universe is empty")
    if not 0 <= cursor.position < size:
        raise WarmstartDatasetError(
            f"cursor position {cursor.position} is outside the universe of {size}"
        )
    order = epoch_order(size, cursor.epoch, cursor.order)
    stop = min(cursor.position + cursor.batch_size, size)
    keys = tuple(universe[index] for index in order[cursor.position : stop])
    if stop >= size:
        cursor_after = replace(cursor, epoch=cursor.epoch + 1, position=0)
    else:
        cursor_after = replace(cursor, position=stop)
    return keys, cursor_after


def plan_batches(universe: tuple, cursor: DataCursor, count: int) -> list:
    """The next `count` batch plans: `[(batch_index, keys, cursor_after), ...]`."""
    plans = []
    current = cursor
    for batch_index in range(int(count)):
        keys, current_after = plan_batch(universe, current)
        plans.append((batch_index, keys, current_after))
        current = current_after
    return plans


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------


class WarmstartDataset:
    """Random access from `(game_id, decision_index)` keys to examples.

    Opens the corpus through :func:`default_corpus_root` — the production
    resolver — unless a root is passed explicitly (tests pass their own mini
    corpora). Decoded records are kept in a bounded LRU per instance.
    """

    def __init__(
        self,
        root: "str | Path | None" = None,
        splits: "tuple[str, ...]" = CORPUS_SPLITS,
        *,
        record_cache_size: int = DEFAULT_RECORD_CACHE,
        require_complete_split: bool = True,
    ) -> None:
        self.root = Path(root) if root is not None else default_corpus_root()
        self.splits = tuple(splits)
        self.reader = CorpusReader(self.root, self.splits)
        self.record_cache_size = int(record_cache_size)
        self.require_complete_split = bool(require_complete_split)
        self._records: OrderedDict = OrderedDict()
        self._universes: dict = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.decode_seconds = 0.0

    # -- corpus access -----------------------------------------------------

    def game(self, game_id: str) -> tuple:
        """`(record, metadata)`, LRU-cached per instance."""
        cached = self._records.get(game_id)
        if cached is not None:
            self._records.move_to_end(game_id)
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        started = time.perf_counter()
        entry = (self.reader.record(game_id), self.reader.metadata(game_id))
        self.decode_seconds += time.perf_counter() - started
        self._records[game_id] = entry
        while len(self._records) > self.record_cache_size:
            self._records.popitem(last=False)
        return entry

    def universe(self, split: str) -> tuple:
        universe = self._universes.get(split)
        if universe is None:
            universe = selected_example_universe(
                self.reader, split, require_complete=self.require_complete_split
            )
            self._universes[split] = universe
        return universe

    # -- example construction ----------------------------------------------

    def examples(self, keys: "tuple[tuple, ...]") -> list:
        """The examples of `keys`, in exactly the order the keys were given.

        Keys are grouped by game so each record is decoded once and its plies
        reconstructed in one ascending sequential pass; the results are then
        placed back into the caller's order. Grouping is a pure performance
        move — order comes from the keys alone.
        """
        by_game: "OrderedDict[str, list]" = OrderedDict()
        for slot, (game_id, index) in enumerate(keys):
            by_game.setdefault(game_id, []).append((int(index), slot))
        results: list = [None] * len(keys)
        for game_id, wanted in by_game.items():
            record, metadata = self.game(game_id)
            plies = tuple(sorted(index for index, _slot in wanted))
            slots = {index: slot for index, slot in wanted}
            produced = 0
            for example in examples_for_game(record, metadata, plies):
                results[slots[example.decision_index]] = example
                produced += 1
            if produced != len(wanted):
                raise WarmstartDatasetError(
                    f"{game_id}: {produced} examples for {len(wanted)} requested plies"
                )
        return results

    def batch_arrays(self, keys: "tuple[tuple, ...]") -> tuple:
        """One batch as plain numpy arrays plus construction statistics.

        Arrays rather than tensors so a worker process can hand the batch to
        the parent through a pickle without dragging torch across the pipe.
        """
        started = time.perf_counter()
        decode_before = self.decode_seconds
        hits_before, misses_before = self.cache_hits, self.cache_misses
        examples = self.examples(keys)
        arrays, metadata = arrays_from_examples(examples)
        stats = {
            "examples": len(examples),
            "build_seconds": time.perf_counter() - started,
            "decode_seconds": self.decode_seconds - decode_before,
            "cache_hits": self.cache_hits - hits_before,
            "cache_misses": self.cache_misses - misses_before,
            "games": len({example.game_id for example in examples}),
        }
        return arrays, metadata, stats

    def iter_sequential(self, split: str, *, batch_size: int = DEFAULT_BATCH_SIZE, limit=None):
        """Universe-order batches of one split, for validation/structural passes."""
        cursor = DataCursor(split=split, batch_size=batch_size, order=ORDER_SEQUENTIAL)
        universe = self.universe(split)
        produced = 0
        while cursor.epoch == 0 and (limit is None or produced < limit):
            keys, cursor = plan_batch(universe, cursor)
            arrays, metadata, _stats = self.batch_arrays(keys)
            yield batch_from_arrays(arrays, metadata)
            produced += 1


# ---------------------------------------------------------------------------
# Batches at the model boundary
# ---------------------------------------------------------------------------


def arrays_from_examples(examples: "list[WarmstartExample]") -> tuple:
    """`(arrays, metadata)` of one batch, in the examples' given order.

    Every array is a fresh stack/copy, so the arrays alias none of the
    examples' own buffers and the batch can outlive them.
    """
    if not examples:
        raise WarmstartDatasetError("a batch needs at least one example")
    arrays = {
        "observation": np.stack([example.observation for example in examples]),
        "legal_mask": np.stack([example.legal_mask for example in examples]),
        "acting_player": np.array(
            [example.acting_player for example in examples], dtype=np.int8
        ),
        "policy_action_abs": np.array(
            [example.policy_action_abs for example in examples], dtype=np.int32
        ),
        "policy_action_model": np.array(
            [example.policy_action_model for example in examples], dtype=np.int64
        ),
        "policy_weight": np.array(
            [example.policy_weight for example in examples], dtype=np.float32
        ),
        "value_target": np.array(
            [example.value_target for example in examples], dtype=np.int64
        ),
        "belief_target": np.stack([example.belief_target for example in examples]),
        "belief_mask": np.stack([example.belief_mask for example in examples]),
    }
    metadata = {
        "keys": tuple((example.game_id, example.decision_index) for example in examples),
        "source_policy_ids": tuple(example.source_policy_id for example in examples),
        "corpus_splits": tuple(example.corpus_split for example in examples),
    }
    return arrays, metadata


@dataclass(frozen=True)
class WarmstartTargets:
    """Loss inputs of one batch. Never handed to the model forward call."""

    legal_mask: torch.Tensor  # bool  [B, 10000], model frame
    policy_action_model: torch.Tensor  # int64 [B]
    policy_weight: torch.Tensor  # float32 [B]
    value_target: torch.Tensor  # int64 [B]
    belief_target: torch.Tensor  # int64 [B, 100]
    belief_mask: torch.Tensor  # bool  [B, 100]
    policy_action_abs: torch.Tensor  # int32 [B]
    acting_player: torch.Tensor  # int8  [B]


@dataclass(frozen=True)
class WarmstartBatch:
    """One training batch with the model input isolated from every target.

    `model_input()` returns the observation tensor and nothing else. The
    tensor is created from a fresh numpy stack, so it holds no reference to
    records, metadata, targets or identities.
    """

    observations: torch.Tensor  # float32 [B, 127, 10, 10]
    targets: WarmstartTargets
    keys: tuple
    source_policy_ids: tuple
    corpus_splits: tuple

    @property
    def batch_size(self) -> int:
        return int(self.observations.shape[0])

    def model_input(self) -> torch.Tensor:
        return self.observations


def batch_from_arrays(arrays: dict, metadata: dict) -> WarmstartBatch:
    """Assemble the torch-side batch from a worker's numpy arrays."""
    observations = torch.from_numpy(np.ascontiguousarray(arrays["observation"]))
    if tuple(observations.shape[1:]) != OBSERVATION_SHAPE:
        raise WarmstartDatasetError(
            f"batch observations have shape {tuple(observations.shape)}"
        )
    targets = WarmstartTargets(
        legal_mask=torch.from_numpy(arrays["legal_mask"]),
        policy_action_model=torch.from_numpy(arrays["policy_action_model"]),
        policy_weight=torch.from_numpy(arrays["policy_weight"]),
        value_target=torch.from_numpy(arrays["value_target"]),
        belief_target=torch.from_numpy(arrays["belief_target"]),
        belief_mask=torch.from_numpy(arrays["belief_mask"]),
        policy_action_abs=torch.from_numpy(arrays["policy_action_abs"]),
        acting_player=torch.from_numpy(arrays["acting_player"]),
    )
    return WarmstartBatch(
        observations=observations,
        targets=targets,
        keys=tuple(metadata["keys"]),
        source_policy_ids=tuple(metadata["source_policy_ids"]),
        corpus_splits=tuple(metadata["corpus_splits"]),
    )


def batch_digest(arrays: dict, metadata: dict) -> str:
    """SHA-256 over a batch's identities and exact tensor bytes.

    The determinism evidence: two loaders that produce the same digests
    produced the same logical batches with the same numerical content.
    """
    hasher = hashlib.sha256()
    for game_id, index in metadata["keys"]:
        hasher.update(f"{game_id}|{index}\n".encode())
    for name in sorted(arrays):
        hasher.update(name.encode())
        hasher.update(np.ascontiguousarray(arrays[name]).tobytes())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# The parallel loader
# ---------------------------------------------------------------------------

_WORKER: dict = {}


def _loader_init(options: dict) -> None:
    _WORKER.clear()
    _WORKER["dataset"] = WarmstartDataset(
        root=options["root"],
        splits=tuple(options["splits"]),
        record_cache_size=options["record_cache_size"],
    )


def _loader_task(payload: tuple) -> tuple:
    batch_index, keys = payload
    dataset = _WORKER["dataset"]
    arrays, metadata, stats = dataset.batch_arrays(tuple(keys))
    return batch_index, arrays, metadata, stats


def iter_batches(
    dataset: WarmstartDataset,
    cursor: DataCursor,
    *,
    batches: int,
    workers: int = 1,
    prefetch: int = 2,
):
    """Yield `(batch, cursor_after, stats)` strictly in logical batch order.

    The plans are computed up front from the cursor alone; workers receive
    `(batch_index, keys)` payloads and the parent yields results in submission
    order, so worker count, scheduling and prefetch depth cannot reorder or
    change a batch. `workers=1` builds in-process, which is bit-identical to
    the pooled path because both run the same `batch_arrays`.
    """
    plans = plan_batches(dataset.universe(cursor.split), cursor, batches)
    if workers <= 1:
        for _index, keys, cursor_after in plans:
            arrays, metadata, stats = dataset.batch_arrays(keys)
            yield batch_from_arrays(arrays, metadata), cursor_after, stats
        return

    options = {
        "root": str(dataset.root),
        "splits": tuple(dataset.splits),
        "record_cache_size": dataset.record_cache_size,
    }
    with ProcessPoolExecutor(
        max_workers=int(workers), initializer=_loader_init, initargs=(options,)
    ) as pool:
        pending = []
        plan_iter = iter(plans)
        for _ in range(max(1, int(workers) * max(1, int(prefetch)))):
            plan = next(plan_iter, None)
            if plan is None:
                break
            index, keys, cursor_after = plan
            pending.append((pool.submit(_loader_task, (index, keys)), cursor_after))
        while pending:
            future, cursor_after = pending.pop(0)
            batch_index, arrays, metadata, stats = future.result()
            plan = next(plan_iter, None)
            if plan is not None:
                index, keys, next_cursor = plan
                pending.append((pool.submit(_loader_task, (index, keys)), next_cursor))
            stats = dict(stats, batch_index=batch_index)
            yield batch_from_arrays(arrays, metadata), cursor_after, stats


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------


def _percentile(values: list, fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return float(ordered[index])


def snapshot_cost_probe(
    dataset: WarmstartDataset, split: str, *, samples: int = 512
) -> dict:
    """Random-access reconstruction cost: seek one decision, cold, per sample.

    Sequential batch construction amortizes replay across a game's plies; this
    probe measures the un-amortized quantity the snapshot interval controls —
    restore the nearest snapshot, replay forward, build one decision — over an
    evenly spread sample of the split's universe.
    """
    from .reconstruction import reconstruct_decision

    universe = dataset.universe(split)
    step = max(1, len(universe) // max(1, int(samples)))
    sample = universe[::step][: int(samples)]
    replayed: list = []
    seconds: list = []
    for game_id, index in sample:
        record, _metadata = dataset.game(game_id)
        started = time.perf_counter()
        rebuilt = reconstruct_decision(
            record, index, dense_mask=True, include_public_knowledge=False
        )
        seconds.append(time.perf_counter() - started)
        replayed.append(rebuilt.replayed_actions)
    return {
        "samples": len(sample),
        "snapshot_interval": 32,
        "replayed_actions_mean": statistics.fmean(replayed) if replayed else 0.0,
        "replayed_actions_max": max(replayed) if replayed else 0,
        "seek_seconds_p50": _percentile(seconds, 0.50),
        "seek_seconds_p95": _percentile(seconds, 0.95),
        "seek_seconds_mean": statistics.fmean(seconds) if seconds else 0.0,
    }


def benchmark_dataset(
    root: "str | Path | None" = None,
    *,
    split: str = "train",
    worker_counts: "tuple[int, ...]" = (1, 2, 4, 8, 10),
    batches: int = 20,
    batch_size: int = DEFAULT_BATCH_SIZE,
    record_cache_size: int = DEFAULT_RECORD_CACHE,
) -> dict:
    """Measure reconstruction feeding at several worker counts.

    Every configuration serves the *same* logical batches (the same cursor),
    so the per-configuration digests double as worker-independence evidence:
    identical digests mean identical batches, whatever the parallelism.
    """
    configurations = []
    reference_digests: "list[str] | None" = None
    for workers in worker_counts:
        dataset = WarmstartDataset(
            root, record_cache_size=record_cache_size
        )
        cursor = DataCursor(split=split, batch_size=batch_size)
        usage_before = resource.getrusage(resource.RUSAGE_SELF)
        children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        started = time.perf_counter()
        digests: list = []
        build_seconds: list = []
        arrival_gaps: list = []
        examples = 0
        games = 0
        decode_seconds = 0.0
        last_arrival = started
        plans = plan_batches(dataset.universe(split), cursor, batches)
        if workers <= 1:
            for _index, keys, _cursor_after in plans:
                arrays, metadata, stats = dataset.batch_arrays(keys)
                now = time.perf_counter()
                arrival_gaps.append(now - last_arrival)
                last_arrival = now
                digests.append(batch_digest(arrays, metadata))
                build_seconds.append(stats["build_seconds"])
                examples += stats["examples"]
                games += stats["games"]
                decode_seconds += stats["decode_seconds"]
        else:
            options = {
                "root": str(dataset.root),
                "splits": tuple(dataset.splits),
                "record_cache_size": record_cache_size,
            }
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_loader_init, initargs=(options,)
            ) as pool:
                results = pool.map(
                    _loader_task,
                    [(index, keys) for index, keys, _after in plans],
                    chunksize=1,
                )
                for _batch_index, arrays, metadata, stats in results:
                    now = time.perf_counter()
                    arrival_gaps.append(now - last_arrival)
                    last_arrival = now
                    digests.append(batch_digest(arrays, metadata))
                    build_seconds.append(stats["build_seconds"])
                    examples += stats["examples"]
                    games += stats["games"]
                    decode_seconds += stats["decode_seconds"]
        wall = time.perf_counter() - started
        usage_after = resource.getrusage(resource.RUSAGE_SELF)
        children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_seconds = (
            (usage_after.ru_utime - usage_before.ru_utime)
            + (usage_after.ru_stime - usage_before.ru_stime)
            + (children_after.ru_utime - children_before.ru_utime)
            + (children_after.ru_stime - children_before.ru_stime)
        )
        if reference_digests is None:
            reference_digests = digests
        configurations.append(
            {
                "workers": workers,
                "batches": len(digests),
                "examples": examples,
                "wall_seconds": wall,
                "examples_per_second": examples / wall if wall else 0.0,
                "batch_build_seconds_p50": _percentile(build_seconds, 0.50),
                "batch_build_seconds_p95": _percentile(build_seconds, 0.95),
                "batch_arrival_seconds_p50": _percentile(arrival_gaps, 0.50),
                "batch_arrival_seconds_p95": _percentile(arrival_gaps, 0.95),
                "mean_batch_build_seconds": (
                    statistics.fmean(build_seconds) if build_seconds else 0.0
                ),
                "decode_seconds": decode_seconds,
                "games_touched": games,
                "cpu_utilization": cpu_seconds / wall if wall else 0.0,
                "parent_peak_rss_bytes": usage_after.ru_maxrss,
                "worker_peak_rss_bytes": children_after.ru_maxrss,
                "digests_match_reference": digests == reference_digests,
                "first_batch_digest": digests[0] if digests else None,
            }
        )
    probe_dataset = WarmstartDataset(root, record_cache_size=record_cache_size)
    return {
        "split": split,
        "batch_size": batch_size,
        "batches_per_configuration": batches,
        "record_cache_size": record_cache_size,
        "configurations": configurations,
        "all_configurations_identical": all(
            entry["digests_match_reference"] for entry in configurations
        ),
        "snapshot_cost": snapshot_cost_probe(probe_dataset, split),
    }


__all__ = [
    "DATA_CURSOR_VERSION",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_RECORD_CACHE",
    "ORDER_SEQUENTIAL",
    "ORDER_SHUFFLE",
    "TRAIN_ORDER_VERSION",
    "DataCursor",
    "WarmstartBatch",
    "WarmstartDataset",
    "WarmstartDatasetError",
    "WarmstartTargets",
    "arrays_from_examples",
    "batch_digest",
    "batch_from_arrays",
    "benchmark_dataset",
    "epoch_order",
    "iter_batches",
    "plan_batch",
    "plan_batches",
    "selected_example_universe",
    "snapshot_cost_probe",
    "universe_digest",
]
