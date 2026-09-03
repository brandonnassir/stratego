"""Phase 18 Stage 6B: K supervised C1 updates per period from the
canonical/live mixture (G3-ENG-02, the C1 half of the joint period loop).

The accepted Phase 8 trainer is reused unchanged for everything that touches
the model: `WarmstartTrainer.train_updates` still builds the batch, runs the
frozen loss, clips, steps AdamW and the warm-up scheduler, keeps the counters
and refuses a non-finite loss. The single thing the joint loop needs that the
accepted trainer does not have is a batch source that mixes two streams, so
the subclass below replaces only the trainer's batch pipeline:

```text
per update u of period p (both lineages, identical rule and seeds):
    live_count       = min(live_per_batch, |retained live universe|)
    canonical_count  = batch_size - live_count            (canonical fills a
                                                           short live stream)
    canonical keys   = the next canonical_count keys of the frozen Phase 8
                       train order, through the accepted DataCursor / plan_batch
    live keys        = a seeded draw without replacement of live_count keys
                       from the retained live universe, seed =
                       derive_stream_seed(namespace, 'c1_live_draw', seed, p, u)
    batch            = arrays_from_examples(canonical examples + live examples)
```

Plans are pure functions of the cursor, the universe and the seed; workers
receive keys and return arrays; results are consumed strictly in plan order,
so worker count and prefetch depth cannot change a batch (the accepted
`_BatchPipeline` argument, transcribed). The cursor stored in the C1 checkpoint
advances by the canonical half only, which is what makes a resumed lineage
serve the exact next canonical keys.

Mixture telemetry (the period-1 gate correction)
-------------------------------------------------
The accepted trainer's metric row keeps only the cache counters of the batch
statistics, so the first pilot recorded `live_rows_served = 0` for a period
that served 8,192 live rows. Every row now carries the mixture the pipeline
actually served (`canonical_examples`, `live_examples`, `examples`, the live
draw seed and the plan index), checked at the step against the trainer's own
batch size and against the plan; the period record carries
`canonical_rows_served` and `live_rows_served`, and a period whose served
counts differ from its planned counts fails before anything is written. The
telemetry is read from the statistics the pipeline already returned with the
arrays: no key, batch, loss, optimizer or scheduler path changes.
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

from ..warmstart_checkpoint import CorpusIdentity, load_warmstart_checkpoint
from ..warmstart_dataset import DataCursor, WarmstartDataset, arrays_from_examples, plan_batch
from ..warmstart_trainer import LoaderTopology, WarmstartTrainConfig, WarmstartTrainer
from .g3_contract import Phase18G3Error, PilotConfig, live_draw_seed
from .g3_live_store import LiveRecordReader, universe_digest


@dataclass(frozen=True)
class MixturePlan:
    """One planned mixed batch: its keys, and the cursor after its canonical half."""

    index: int
    period: int
    update: int
    canonical_keys: tuple
    live_keys: tuple
    cursor_after: DataCursor
    live_seed: int

    @property
    def batch_size(self) -> int:
        return len(self.canonical_keys) + len(self.live_keys)


def plan_mixture_batches(
    canonical_universe: tuple,
    cursor: DataCursor,
    live_universe: tuple,
    *,
    period: int,
    batches: int,
    batch_size: int,
    live_per_batch: int,
    namespace: str,
    seed_index: int,
) -> list:
    """The `batches` mixed plans of one period, from the cursor and the seeds alone."""
    if int(batches) < 1:
        raise Phase18G3Error("a period plans at least one batch")
    if live_per_batch < 0 or live_per_batch > batch_size:
        raise Phase18G3Error("live_per_batch must be in 0..batch_size")
    plans = []
    current = cursor
    live_size = len(live_universe)
    for update in range(1, int(batches) + 1):
        live_count = min(int(live_per_batch), live_size)
        canonical_count = int(batch_size) - live_count
        keys, after = plan_batch(canonical_universe, replace(current, batch_size=canonical_count))
        after = replace(after, batch_size=current.batch_size)
        seed = live_draw_seed(namespace, seed_index, period, update)
        if live_count:
            rng = np.random.default_rng(seed % (2**32))
            chosen = rng.choice(live_size, size=live_count, replace=False)
            live_keys = tuple(live_universe[int(position)] for position in chosen)
        else:
            live_keys = ()
        plans.append(
            MixturePlan(
                index=update - 1,
                period=int(period),
                update=update,
                canonical_keys=tuple(keys),
                live_keys=live_keys,
                cursor_after=after,
                live_seed=int(seed),
            )
        )
        current = after
    return plans


# ---------------------------------------------------------------------------
# Building one mixed batch (in-process, or in a loader worker)
# ---------------------------------------------------------------------------


def build_mixed_batch(dataset: WarmstartDataset, live: LiveRecordReader, canonical_keys, live_keys) -> tuple:
    started = time.perf_counter()
    hits, misses = dataset.cache_hits, dataset.cache_misses
    canonical = dataset.examples(tuple(canonical_keys)) if canonical_keys else []
    live_examples = live.examples(tuple(live_keys)) if live_keys else []
    examples = list(canonical) + list(live_examples)
    arrays, metadata = arrays_from_examples(examples)
    stats = {
        "examples": len(examples),
        "canonical_examples": len(canonical),
        "live_examples": len(live_examples),
        "build_seconds": time.perf_counter() - started,
        "cache_hits": dataset.cache_hits - hits,
        "cache_misses": dataset.cache_misses - misses,
        "games": len({example.game_id for example in examples}),
    }
    return arrays, metadata, stats


_WORKER: dict = {}


def _mixture_worker_init(options: dict) -> None:
    _WORKER.clear()
    _WORKER["dataset"] = WarmstartDataset(
        root=options["root"],
        splits=tuple(options["splits"]),
        record_cache_size=int(options["record_cache_size"]),
        require_complete_split=bool(options["require_complete_split"]),
    )
    _WORKER["live"] = LiveRecordReader(options["live_root"], record_cache_size=int(options["record_cache_size"]))


def _mixture_task(payload: tuple) -> tuple:
    index, canonical_keys, live_keys = payload
    arrays, metadata, stats = build_mixed_batch(_WORKER["dataset"], _WORKER["live"], canonical_keys, live_keys)
    return index, arrays, metadata, stats


class MixturePipeline:
    """Planned mixed batches in, byte-identical batches out, strictly in order.

    `workers=1` builds in-process through the same `build_mixed_batch`, which
    is the bit-identical reference path; more workers only change arrival
    times. A period's plans must be fully consumed before the next period's
    are scheduled, so a plan can never be built against a stale live universe.
    """

    def __init__(
        self,
        dataset: WarmstartDataset,
        *,
        live_root,
        workers: int,
        prefetch: int,
        record_cache_size: int,
        require_complete_split: bool,
    ) -> None:
        self.dataset = dataset
        self.live = LiveRecordReader(live_root, record_cache_size=int(record_cache_size))
        self.workers = int(workers)
        self.prefetch = max(1, int(prefetch))
        self._plans: deque = deque()
        self._pending: deque = deque()
        self._pool = None
        self.served = 0
        #: The statistics of every batch served in the current period, in
        #: service order (reset by `schedule`); the trainer folds them into rows.
        self.period_stats: list = []
        if self.workers > 1:
            options = {
                "root": str(dataset.root),
                "splits": tuple(dataset.splits),
                "record_cache_size": int(record_cache_size),
                "require_complete_split": bool(require_complete_split),
                "live_root": str(live_root),
            }
            self._pool = ProcessPoolExecutor(max_workers=self.workers, initializer=_mixture_worker_init, initargs=(options,))

    def schedule(self, plans) -> None:
        if self._plans or self._pending:
            raise Phase18G3Error(
                f"{self.remaining()} planned batches of the previous period were never consumed"
            )
        self._plans = deque(plans)
        self.period_stats = []
        self._fill()

    def remaining(self) -> int:
        return len(self._plans) + len(self._pending)

    def _fill(self) -> None:
        if self._pool is None:
            return
        target = self.workers * self.prefetch
        while len(self._pending) < target and self._plans:
            plan = self._plans.popleft()
            future = self._pool.submit(_mixture_task, (plan.index, plan.canonical_keys, plan.live_keys))
            self._pending.append((future, plan))

    def next(self) -> tuple:
        """`(arrays, metadata, stats, cursor_after, wait_seconds)`, the trainer's contract."""
        if self._pool is None:
            if not self._plans:
                raise Phase18G3Error("no planned batch is left in this period")
            plan = self._plans.popleft()
            started = time.perf_counter()
            arrays, metadata, stats = build_mixed_batch(self.dataset, self.live, plan.canonical_keys, plan.live_keys)
            waited = time.perf_counter() - started
        else:
            if not self._pending:
                raise Phase18G3Error("no planned batch is left in this period")
            future, plan = self._pending.popleft()
            started = time.perf_counter()
            index, arrays, metadata, stats = future.result()
            waited = time.perf_counter() - started
            if index != plan.index:
                raise Phase18G3Error(f"batch {index} arrived for plan {plan.index}; the order broke")
            self._fill()
        self.served += 1
        stats = dict(stats, plan_index=plan.index, period=plan.period, update=plan.update, live_seed=plan.live_seed)
        self.period_stats.append(stats)
        return arrays, metadata, stats, plan.cursor_after, waited

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None
        self._plans.clear()
        self._pending.clear()


# ---------------------------------------------------------------------------
# The trainer
# ---------------------------------------------------------------------------


class JointC1Trainer(WarmstartTrainer):
    """The accepted Phase 8 trainer with the mixed canonical/live batch source.

    Nothing about the loss, the optimizer, the schedule, the counters, the
    validation cadence or the checkpoint format changes; `_ensure_pipeline`
    is the one override, and `begin_period` is the one addition.
    """

    def __init__(
        self,
        config: WarmstartTrainConfig,
        corpus_identity: CorpusIdentity,
        *,
        pilot: PilotConfig,
        live_root,
        root=None,
        require_complete_split: bool = True,
        value_prior=None,
        run_label: str = "",
        fsync_checkpoints: bool = True,
        _restored=None,
    ) -> None:
        if pilot.c1_train_config != config:
            raise Phase18G3Error("the pilot configuration names a different C1 train configuration")
        super().__init__(
            config,
            corpus_identity,
            root=root,
            topology=LoaderTopology(
                workers=pilot.loader_workers,
                prefetch=pilot.loader_prefetch,
                record_cache_size=pilot.record_cache_size,
            ),
            require_complete_split=require_complete_split,
            value_prior=value_prior,
            run_label=run_label,
            fsync_checkpoints=fsync_checkpoints,
            _restored=_restored,
        )
        self.pilot = pilot
        self.live_root = live_root
        self.require_complete_split = bool(require_complete_split)
        if _restored is None:
            # The cursor advances by the canonical half only.
            self.cursor = replace(self.cursor, batch_size=int(pilot.canonical_per_batch))
        elif int(self.cursor.batch_size) != int(pilot.canonical_per_batch):
            raise Phase18G3Error(
                f"the restored cursor serves {self.cursor.batch_size} canonical rows per batch, the "
                f"pilot {pilot.canonical_per_batch}"
            )
        self.period_record: dict | None = None
        self._period_plans: list = []

    @classmethod
    def resume(
        cls,
        checkpoint_path,
        *,
        config: WarmstartTrainConfig,
        corpus_identity: CorpusIdentity,
        pilot: PilotConfig,
        live_root,
        root=None,
        require_complete_split: bool = True,
        value_prior=None,
        run_label: str = "",
        fsync_checkpoints: bool = True,
    ) -> "JointC1Trainer":
        restored = load_warmstart_checkpoint(
            checkpoint_path,
            expected_train_config=config.identity(),
            expected_train_config_digest=config.digest(),
            expected_corpus_identity=corpus_identity,
            device=config.device,
        )
        return cls(
            config,
            corpus_identity,
            pilot=pilot,
            live_root=live_root,
            root=root,
            require_complete_split=require_complete_split,
            value_prior=value_prior,
            run_label=run_label,
            fsync_checkpoints=fsync_checkpoints,
            _restored=restored,
        )

    def _ensure_pipeline(self):
        if self._pipeline is None:
            self._pipeline = MixturePipeline(
                self.dataset,
                live_root=self.live_root,
                workers=self.topology.workers,
                prefetch=self.topology.prefetch,
                record_cache_size=self.topology.record_cache_size,
                require_complete_split=self.require_complete_split,
            )
        return self._pipeline

    def begin_period(self, *, period: int, live_universe: tuple, updates: int) -> dict:
        """Plan and schedule the period's `updates` mixed batches."""
        pilot = self.pilot
        plans = plan_mixture_batches(
            self.dataset.universe(self.cursor.split),
            self.cursor,
            tuple(live_universe),
            period=int(period),
            batches=int(updates),
            batch_size=int(self.config.batch_size),
            live_per_batch=int(pilot.live_per_batch),
            namespace=pilot.namespace,
            seed_index=pilot.seed_index,
        )
        self._ensure_pipeline().schedule(plans)
        self._period_plans = list(plans)
        self.period_record = {
            "period": int(period),
            "updates_planned": len(plans),
            "live_universe_size": len(live_universe),
            "live_universe_digest": universe_digest(live_universe),
            "live_rows_planned": int(sum(len(plan.live_keys) for plan in plans)),
            "canonical_rows_planned": int(sum(len(plan.canonical_keys) for plan in plans)),
            "cursor_before": self.cursor.to_dict(),
            "cursor_after_planned": plans[-1].cursor_after.to_dict(),
            "live_seeds": [plan.live_seed for plan in plans],
        }
        return dict(self.period_record)

    def train_period(self, *, period: int, live_universe: tuple, updates: int) -> tuple:
        """`begin_period` then exactly `updates` accepted training steps."""
        record = self.begin_period(period=period, live_universe=live_universe, updates=updates)
        step_before = self.global_step
        rows = self.train_updates(int(updates))
        pipeline = self._ensure_pipeline()
        if pipeline.remaining():
            raise Phase18G3Error(f"{pipeline.remaining()} planned batches were not consumed in period {period}")
        if self.global_step - step_before != int(updates):
            raise Phase18G3Error(f"period {period}: {self.global_step - step_before} updates ran, {updates} planned")
        if self.cursor.to_dict() != record["cursor_after_planned"]:
            raise Phase18G3Error(f"period {period}: the cursor did not land where the plan said")
        record["updates_completed"] = int(updates)
        record["global_step_after"] = int(self.global_step)
        record["keys_digests"] = [row["keys_digest"] for row in rows]
        record["canonical_rows_served"] = int(sum(int(row["canonical_examples"]) for row in rows))
        record["live_rows_served"] = int(sum(int(row["live_examples"]) for row in rows))
        record["examples_served"] = int(sum(int(row["examples"]) for row in rows))
        served_equals_planned = (
            record["canonical_rows_served"] == int(record["canonical_rows_planned"])
            and record["live_rows_served"] == int(record["live_rows_planned"])
            and record["examples_served"] == record["canonical_rows_served"] + record["live_rows_served"]
        )
        if not served_equals_planned:
            raise Phase18G3Error(
                f"period {period}: the pipeline served {record['canonical_rows_served']} canonical and "
                f"{record['live_rows_served']} live rows, the plan was {record['canonical_rows_planned']} and "
                f"{record['live_rows_planned']}; the period is not the planned mixture"
            )
        record["served_equals_planned"] = True
        return rows, record

    #: Row fields folded in from the pipeline's batch statistics.
    MIXTURE_ROW_FIELDS = ("canonical_examples", "live_examples", "examples", "live_seed", "plan_index")

    def _fold_mixture_stats(self, row: dict, batch, stats: dict) -> None:
        """Record the served mixture on the row; refuse a batch that is not its plan.

        Runs at the step (the accepted trainer's `on_step` hook), so a
        mis-served batch stops the period at once rather than after K updates.
        """
        canonical = int(stats["canonical_examples"])
        live = int(stats["live_examples"])
        total = int(stats["examples"])
        index = int(stats["plan_index"])
        where = f"period {stats.get('period')} update {stats.get('update')}"
        if canonical + live != total or int(batch.batch_size) != total:
            raise Phase18G3Error(
                f"{where}: the pipeline reports {canonical} canonical + {live} live = {total} examples "
                f"but the trainer consumed a batch of {batch.batch_size}"
            )
        plan = self._period_plans[index] if index < len(self._period_plans) else None
        if plan is None or plan.index != index:
            raise Phase18G3Error(f"{where}: no scheduled plan with index {index}")
        if canonical != len(plan.canonical_keys) or live != len(plan.live_keys) or int(stats["live_seed"]) != plan.live_seed:
            raise Phase18G3Error(
                f"{where}: served {canonical} canonical + {live} live examples (live seed {stats['live_seed']}), "
                f"planned {len(plan.canonical_keys)} + {len(plan.live_keys)} (live seed {plan.live_seed})"
            )
        row["canonical_examples"] = canonical
        row["live_examples"] = live
        row["examples"] = total
        row["live_seed"] = int(stats["live_seed"])
        row["plan_index"] = index

    def train_updates(self, updates: int, *, on_step=None, **keywords) -> list:
        """The accepted loop; the served mixture is folded into each row at the step."""
        pipeline = self._ensure_pipeline()
        served_before = pipeline.served
        stats_before = len(pipeline.period_stats)

        def at_step(row, batch):
            # `next()` appended this batch's statistics before the trainer consumed it.
            self._fold_mixture_stats(row, batch, pipeline.period_stats[-1])
            if on_step is not None:
                on_step(row, batch)

        rows = super().train_updates(updates, on_step=at_step, **keywords)
        if pipeline.served - served_before != len(rows):
            raise Phase18G3Error("the pipeline served a different number of batches than rows recorded")
        if len(pipeline.period_stats) - stats_before != len(rows):
            raise Phase18G3Error("the pipeline recorded a different number of batch statistics than rows")
        for row in rows:
            missing = [name for name in self.MIXTURE_ROW_FIELDS if name not in row]
            if missing:
                raise Phase18G3Error(f"a C1 row is missing the mixture telemetry {missing}")
        return rows


__all__ = ["JointC1Trainer", "MixturePipeline", "MixturePlan", "build_mixed_batch", "plan_mixture_batches"]
