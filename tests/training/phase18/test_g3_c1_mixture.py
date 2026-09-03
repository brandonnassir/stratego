"""Stage 6B: the canonical/live mixture through the accepted C1 trainer is a
pure function of the cursor, the live universe and the seeds; it resumes
exactly; and the parallel loader reproduces the serial reference."""

import pytest
import torch

from stratego.training.phase18 import g3_c1 as c1
from stratego.training.phase18 import g3_live_store as ls
from stratego.training.phase18.g3_contract import Phase18G3Error, PilotConfig
from stratego.training.warmstart_checkpoint import verify_corpus_identity
from stratego.training.warmstart_dataset import WarmstartDataset
from stratego.training.warmstart_trainer import unit_test_config

NAMESPACE = "phase18_g3_c1_mixture_test_v1"
UNIFORM_PRIOR = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


@pytest.fixture(scope="module")
def corpus(warmstart_mini_corpus):
    root, game_ids = warmstart_mini_corpus
    identity = verify_corpus_identity(root, None, check_payload_bytes=False)
    return root, identity, game_ids


@pytest.fixture(scope="module")
def live_root(tmp_path_factory, corpus):
    root, _identity, game_ids = corpus
    live = tmp_path_factory.mktemp("live")
    dataset = WarmstartDataset(root, require_complete_split=False)
    writer = ls.LivePeriodWriter(live, period=1, namespace=NAMESPACE, lineage="candidate", run_id="G3-C1-TEST")
    for game_id in game_ids:
        record, metadata = dataset.game(game_id)
        if metadata["corpus_split"] == "train":
            writer.write(record, dict(metadata))
    writer.close()
    return live


def pilot(lineage="candidate", **overrides) -> PilotConfig:
    fields = dict(
        run_id="G3-C1-TEST",
        namespace=NAMESPACE,
        seed_index=1,
        lineage=lineage,
        c1_train_config=unit_test_config(batch_size=8),
        canonical_per_batch=4,
        live_per_batch=4,
        periods=2,
        c1_updates_per_period=3,
        slots=2,
        pool_size=4,
        plies_per_period=20,
        schedule_cells=2,
        buffer_storage_periods=201,
        live_retention_periods=2,
        bundle_cadence_periods=1,
        threads=1,
    )
    fields.update(overrides)
    return PilotConfig(**fields)


def trainer_for(config, corpus, live_root):
    root, identity, _ = corpus
    return c1.JointC1Trainer(
        config.c1_train_config,
        identity,
        pilot=config,
        live_root=live_root,
        root=root,
        require_complete_split=False,
        value_prior=UNIFORM_PRIOR,
    )


def test_plans_are_pure_functions_of_cursor_universe_and_seed(corpus, live_root):
    torch.set_num_threads(1)
    config = pilot()
    reader = ls.LiveRecordReader(live_root)
    universe = reader.universe([1])
    trainer = trainer_for(config, corpus, live_root)
    canonical = trainer.dataset.universe("train")
    plans = c1.plan_mixture_batches(canonical, trainer.cursor, universe, period=1, batches=3, batch_size=8, live_per_batch=4, namespace=NAMESPACE, seed_index=1)
    again = c1.plan_mixture_batches(canonical, trainer.cursor, universe, period=1, batches=3, batch_size=8, live_per_batch=4, namespace=NAMESPACE, seed_index=1)
    assert [(p.canonical_keys, p.live_keys, p.cursor_after) for p in plans] == [(p.canonical_keys, p.live_keys, p.cursor_after) for p in again]
    assert all(len(p.canonical_keys) == 4 and len(p.live_keys) == 4 for p in plans)
    assert all(key in universe for p in plans for key in p.live_keys)
    assert plans[0].cursor_after.position == 4 and plans[0].cursor_after.batch_size == 4
    other = c1.plan_mixture_batches(canonical, trainer.cursor, universe, period=2, batches=3, batch_size=8, live_per_batch=4, namespace=NAMESPACE, seed_index=1)
    assert [p.live_keys for p in other] != [p.live_keys for p in plans]
    # An empty live universe is filled by the canonical stream.
    empty = c1.plan_mixture_batches(canonical, trainer.cursor, (), period=1, batches=1, batch_size=8, live_per_batch=4, namespace=NAMESPACE, seed_index=1)
    assert len(empty[0].canonical_keys) == 8 and empty[0].live_keys == ()
    trainer.close()


def test_a_period_of_mixed_updates_is_deterministic_and_records_its_keys(corpus, live_root):
    torch.set_num_threads(1)
    config = pilot()
    universe = ls.LiveRecordReader(live_root).universe([1])
    digests = []
    for _ in range(2):
        trainer = trainer_for(config, corpus, live_root)
        rows, record = trainer.train_period(period=1, live_universe=universe, updates=3)
        assert len(rows) == 3 and record["updates_completed"] == 3 and record["live_rows_planned"] == 12
        assert record["live_universe_digest"] == ls.universe_digest(universe)
        assert trainer.global_step == 3 and trainer.cursor.position == 12 and trainer.cursor.batch_size == 4
        # Every row records the mixture the pipeline served, and the served totals equal the plan.
        for index, row in enumerate(rows):
            assert (row["canonical_examples"], row["live_examples"], row["examples"]) == (4, 4, 8)
            assert row["batch_size"] == 8 and row["plan_index"] == index
        assert [row["live_seed"] for row in rows] == record["live_seeds"]
        assert record["canonical_rows_served"] == record["canonical_rows_planned"] == 12
        assert record["live_rows_served"] == record["live_rows_planned"] == 12
        assert record["examples_served"] == 24 and record["served_equals_planned"] is True
        digests.append((record["keys_digests"], c1.c1_digest(trainer) if hasattr(c1, "c1_digest") else None))
        trainer.close()
    assert digests[0][0] == digests[1][0]


def test_the_production_mixture_is_128_canonical_and_128_live_for_64_updates(corpus, live_root):
    """With the frozen production values (batch 256, 128 canonical + 128 live,
    64 updates per period) every planned batch is the 128:128 mixture and the
    period plans 8,192 canonical and 8,192 live rows."""
    from dataclasses import replace

    from stratego.training.phase18.g3_contract import C1_UPDATES_PER_PERIOD, CANONICAL_PER_BATCH, LIVE_PER_BATCH

    assert (C1_UPDATES_PER_PERIOD, CANONICAL_PER_BATCH, LIVE_PER_BATCH) == (64, 128, 128)
    torch.set_num_threads(1)
    config = pilot()
    trainer = trainer_for(config, corpus, live_root)
    universe = ls.LiveRecordReader(live_root).universe([1])
    assert len(universe) >= LIVE_PER_BATCH
    # The accepted planner never spans an epoch boundary, so the canonical
    # universe must hold a period's worth of rows for every batch to be full,
    # as the 42,376-game production corpus does; a synthetic universe stands in.
    canonical = tuple((f"game_{index // 64:05d}", index % 64) for index in range(4 * 8192))
    cursor = replace(trainer.cursor, batch_size=CANONICAL_PER_BATCH)
    plans = c1.plan_mixture_batches(
        canonical, cursor, universe, period=1, batches=C1_UPDATES_PER_PERIOD, batch_size=CANONICAL_PER_BATCH + LIVE_PER_BATCH,
        live_per_batch=LIVE_PER_BATCH, namespace=NAMESPACE, seed_index=1,
    )
    assert len(plans) == 64
    assert all(len(plan.canonical_keys) == 128 and len(plan.live_keys) == 128 and plan.batch_size == 256 for plan in plans)
    assert all(len(set(plan.live_keys)) == 128 and set(plan.live_keys) <= set(universe) for plan in plans)
    assert all(len(set(plan.canonical_keys)) == 128 and set(plan.canonical_keys) <= set(canonical) for plan in plans)
    assert sum(len(plan.canonical_keys) for plan in plans) == 8192
    assert sum(len(plan.live_keys) for plan in plans) == 8192
    assert plans[-1].cursor_after.batch_size == 128 and plans[-1].cursor_after.position == 8192
    trainer.close()

    # One real update at the production composition through the accepted
    # trainer: the row records 128 + 128 = 256, the period record 128 / 128 served.
    production = pilot(c1_train_config=unit_test_config(batch_size=256), canonical_per_batch=128, live_per_batch=128, c1_updates_per_period=1)
    trainer = trainer_for(production, corpus, live_root)
    rows, record = trainer.train_period(period=1, live_universe=universe, updates=1)
    assert len(rows) == 1
    assert (rows[0]["canonical_examples"], rows[0]["live_examples"], rows[0]["examples"], rows[0]["batch_size"]) == (128, 128, 256, 256)
    assert record["canonical_rows_served"] == record["canonical_rows_planned"] == 128
    assert record["live_rows_served"] == record["live_rows_planned"] == 128
    assert record["served_equals_planned"] is True
    assert trainer.cursor.batch_size == 128 and trainer.cursor.position == 128
    trainer.close()


def test_a_period_whose_served_mixture_differs_from_the_plan_fails_at_the_step(monkeypatch, corpus, live_root):
    """The served-equals-planned check is not telemetry: a pipeline that serves
    one live example short stops the period at that update."""
    torch.set_num_threads(1)
    original = c1.build_mixed_batch
    calls = []

    def one_live_short(dataset, live, canonical_keys, live_keys):
        calls.append(len(live_keys))
        return original(dataset, live, canonical_keys, tuple(live_keys)[:-1])

    monkeypatch.setattr(c1, "build_mixed_batch", one_live_short)
    config = pilot()
    universe = ls.LiveRecordReader(live_root).universe([1])
    trainer = trainer_for(config, corpus, live_root)
    with pytest.raises(Phase18G3Error, match="planned 4 \\+ 4"):
        trainer.train_period(period=1, live_universe=universe, updates=3)
    assert calls == [4], "the first mis-served batch stopped the period"
    assert trainer.global_step == 1
    trainer.close()


def test_resume_serves_the_exact_next_batches(tmp_path, corpus, live_root):
    torch.set_num_threads(1)
    from stratego.training.phase18.setup_model import state_dict_digest

    config = pilot()
    universe = ls.LiveRecordReader(live_root).universe([1])
    uninterrupted = trainer_for(config, corpus, live_root)
    uninterrupted.train_period(period=1, live_universe=universe, updates=3)
    checkpoint = tmp_path / "c1.pt"
    uninterrupted.save_checkpoint(checkpoint)
    _rows, second = uninterrupted.train_period(period=2, live_universe=universe, updates=3)
    uninterrupted.close()

    root, identity, _ = corpus
    resumed = c1.JointC1Trainer.resume(
        checkpoint, config=config.c1_train_config, corpus_identity=identity, pilot=config, live_root=live_root, root=root, require_complete_split=False, value_prior=UNIFORM_PRIOR
    )
    assert resumed.global_step == 3 and resumed.cursor.batch_size == 4
    _rows, second_resumed = resumed.train_period(period=2, live_universe=universe, updates=3)
    resumed.close()
    assert second_resumed["keys_digests"] == second["keys_digests"]
    assert second_resumed["cursor_after_planned"] == second["cursor_after_planned"]
    assert state_dict_digest(resumed.model) == state_dict_digest(uninterrupted.model)


def test_the_parallel_loader_reproduces_the_serial_batches(corpus, live_root):
    torch.set_num_threads(1)
    universe = ls.LiveRecordReader(live_root).universe([1])
    digests = {}
    for workers in (1, 2):
        config = pilot(loader_workers=workers, loader_prefetch=1)
        trainer = trainer_for(config, corpus, live_root)
        trainer.begin_period(period=1, live_universe=universe, updates=3)
        rows = trainer.train_updates(3, capture_batch_digests=True)
        digests[workers] = [row["batch_digest"] for row in rows]
        trainer.close()
    assert digests[1] == digests[2]


def test_a_period_must_consume_every_planned_batch(corpus, live_root):
    torch.set_num_threads(1)
    config = pilot()
    universe = ls.LiveRecordReader(live_root).universe([1])
    trainer = trainer_for(config, corpus, live_root)
    trainer.begin_period(period=1, live_universe=universe, updates=3)
    trainer.train_updates(1)
    with pytest.raises(Phase18G3Error, match="never consumed"):
        trainer.begin_period(period=2, live_universe=universe, updates=3)
    trainer.close()
