"""Stage 6B: the teacher-schedule collector plays from the pool, attributes
exactly two outcomes per finished game, commits every finished trajectory,
keeps the G4 accounting identity, and persists its unfinished games exactly
through the Phase 17 codec."""

import copy

import pytest
import torch

from stratego.engine.constants import BLUE, RED
from stratego.training.phase18 import g3_collector as gc
from stratego.training.phase18.g3_contract import Phase18G3Error, PilotConfig
from stratego.training.phase18.g3_buffer_state import buffer_state_digest, capture_buffer_state, restore_buffer_state
from stratego.training.phase18.g3_live_store import LiveRecordReader
from stratego.training.phase18.setup_buffer import SetupBuffer
from stratego.training.phase18.setup_model import build_setup_model, state_dict_digest
from stratego.training.phase18.setup_sampling import generate_pool
from stratego.training.warmstart_trainer import unit_test_config

NAMESPACE = "phase18_g3_collector_test_v1"


def tiny_config(lineage="candidate", **overrides) -> PilotConfig:
    fields = dict(
        run_id="G3-COLLECTOR-TEST",
        namespace=NAMESPACE,
        seed_index=1,
        lineage=lineage,
        c1_train_config=unit_test_config(batch_size=8),
        canonical_per_batch=4,
        live_per_batch=4,
        periods=3,
        c1_updates_per_period=2,
        slots=3,
        pool_size=8,
        plies_per_period=64,
        schedule_cells=4,
        cell_indices=(11, 12, 21, 22),
        buffer_storage_periods=101,
        live_retention_periods=2,
        bundle_cadence_periods=1,
        threads=1,
    )
    fields.update(overrides)
    return PilotConfig(**fields)


@pytest.fixture(scope="module")
def pool():
    torch.set_num_threads(1)
    model = build_setup_model(device="cpu", seed=11)
    digest = state_dict_digest(model)
    generation = generate_pool(model, namespace=NAMESPACE, seed_index=1, snapshot_iteration=0, snapshot_digest=digest, count=8)
    return generation.samples, digest


def _fresh(tmp_path, config, pool):
    samples, digest = pool
    buffer = SetupBuffer(storage_duration=config.buffer_storage_periods)
    buffer.add_pool(samples, period=1)
    collector = gc.PeriodCollector(config, buffer, live_root=tmp_path / "live")
    collector.begin_period(1, samples, snapshot_digest=digest)
    return buffer, collector


def test_one_period_plays_from_the_pool_and_reconciles(tmp_path, pool):
    config = tiny_config()
    samples, digest = pool
    buffer, collector = _fresh(tmp_path, config, pool)
    accounting = collector.run_period()
    document = collector.end_period()
    assert accounting.plies_advanced == config.slots * config.plies_per_period
    assert document["started"] == document["completed"] + document["in_flight_at_end"] + document["failed"]
    assert document["outcomes_attributed"] == 2 * document["completed"]
    assert document["live"]["games"] == document["completed"]
    assert buffer.outcomes_added == 2 * document["completed"]
    # Every started game took the next schedule cell in cyclic order and a pool row per lane.
    assert collector.cell_cursor == document["started"]
    reader = LiveRecordReader(tmp_path / "live")
    for game_id in document["completed_game_ids_digest"] and reader.commits(1):
        metadata = reader.metadata(1, game_id)
        assert metadata["corpus_split"] == "train" and metadata["lineage"] == "candidate"
        assert metadata["setup_provenance"]["red"]["content_fingerprint"] in {s.content_fingerprint for s in samples if s.lane == RED}
        assert metadata["setup_provenance"]["blue"]["content_fingerprint"] in {s.content_fingerprint for s in samples if s.lane == BLUE}
        assert metadata["cell_index"] in (11, 12, 21, 22)
        record = reader.record(1, game_id)
        assert record.rules_context == "training" and record.battleless_move_limit == 100


def test_outcomes_attribute_to_the_played_rows_from_the_owners_view(tmp_path, pool):
    config = tiny_config(slots=2, plies_per_period=400)
    samples, digest = pool
    buffer, collector = _fresh(tmp_path, config, pool)
    collector.run_period()
    document = collector.end_period()
    assert document["completed"] >= 1
    reader = LiveRecordReader(tmp_path / "live")
    for game_id in reader.commits(1):
        metadata = reader.metadata(1, game_id)
        red = buffer.outcome_record(metadata["setup_provenance"]["red"]["content_fingerprint"])
        blue = buffer.outcome_record(metadata["setup_provenance"]["blue"]["content_fingerprint"])
        assert red["count"] >= 1 and blue["count"] >= 1 and red["ready"] and blue["ready"]
    # The recorded outcome pairs are +1/-1 or 0/0 per game.
    outcomes = collector_outcomes(document, reader)
    for red_z, blue_z in outcomes:
        assert (red_z, blue_z) in ((1, -1), (-1, 1), (0, 0))


def collector_outcomes(document, reader):
    pairs = []
    for game_id in reader.commits(1):
        result = reader.metadata(1, game_id)["terminal_result"]
        pairs.append({"red_win": (1, -1), "blue_win": (-1, 1), "draw": (0, 0)}[result])
    return pairs


def test_both_lineages_produce_identical_period_one_games(tmp_path, pool):
    documents = {}
    for lineage in ("candidate", "control"):
        config = tiny_config(lineage=lineage)
        buffer, collector = _fresh(tmp_path / lineage, config, pool)
        collector.run_period()
        documents[lineage] = collector.end_period()
    a, b = documents["candidate"], documents["control"]
    for key in ("started", "completed", "completed_game_ids_digest", "outcome_records_digest", "plies_advanced"):
        assert a[key] == b[key], key
    assert a["live"]["commit_digest"] == b["live"]["commit_digest"]


def test_capture_and_restore_reproduce_the_next_period_exactly(tmp_path, pool):
    config = tiny_config()
    samples, digest = pool
    # Uninterrupted: periods 1 and 2 in one collector.
    buffer_a, control = _fresh(tmp_path / "a", config, pool)
    control.run_period()
    first = control.end_period()
    assert first["in_flight_at_end"] > 0, "the tiny configuration must leave a game unfinished"
    captured = control.capture()
    buffer_state = capture_buffer_state(buffer_a)
    buffer_a.add_pool(samples, period=2)
    control.begin_period(2, samples, snapshot_digest=digest)
    control.run_period()
    second_control = control.end_period()

    # Restarted: a fresh collector restored from the capture, then period 2.
    buffer_b = restore_buffer_state(buffer_state)
    assert buffer_state_digest(buffer_b) == buffer_state_digest(restore_buffer_state(buffer_state))
    restarted = gc.PeriodCollector(config, buffer_b, live_root=tmp_path / "b" / "live")
    restore = restarted.restore(copy.deepcopy(captured))
    assert restore["games_restored"] == first["in_flight_at_end"]
    for slot, runner in enumerate(restarted.slots):
        original = control_slots_after_first(control, captured, slot)
        if runner is None:
            assert original is None
        else:
            assert runner.game_id == original
    buffer_b.add_pool(samples, period=2)
    restarted.begin_period(2, samples, snapshot_digest=digest)
    restarted.run_period()
    second_restarted = restarted.end_period()
    for key in ("started", "completed", "in_flight_at_end", "plies_advanced", "outcomes_attributed", "cross_period_attributions", "completed_game_ids_digest", "outcome_records_digest"):
        assert second_control[key] == second_restarted[key], key
    assert second_control["live"]["commit_digest"] == second_restarted["live"]["commit_digest"]
    assert buffer_state_digest(buffer_a) == buffer_state_digest(buffer_b)
    assert restarted.telemetry() == control.telemetry()


def control_slots_after_first(control, captured, slot):
    ids = {entry["identity"]["slot"]: entry["identity"]["game_id"] for entry in __import__("pickle").loads(__import__("zlib").decompress(captured["active_games_blob"]))}
    return ids.get(slot)


def test_an_unattributable_outcome_is_fatal(tmp_path, pool):
    config = tiny_config(slots=1, plies_per_period=4000)
    samples, digest = pool
    buffer = SetupBuffer(storage_duration=config.buffer_storage_periods)
    buffer.add_pool(samples, period=1)
    collector = gc.PeriodCollector(config, buffer, live_root=tmp_path / "live")
    collector.begin_period(1, samples, snapshot_digest=digest)
    # Drop every blue row from the buffer behind the collector's back: the first
    # finished game's red outcome attributes, its blue outcome cannot.
    buffer.filter(10_000)
    buffer.add_pool([s for s in samples if s.lane == RED], period=10_001)
    with pytest.raises(Phase18G3Error, match="could not be attributed"):
        collector.run_period()
    assert collector.attribution_failures == 1


def test_the_reserved_and_test_bases_are_refused():
    from stratego.training.phase18.g3_contract import assert_base_index_is_evaluable

    assert assert_base_index_is_evaluable(400) == 400 and assert_base_index_is_evaluable(409) == 409
    for index in (410, 449):
        with pytest.raises(Phase18G3Error, match="reserved"):
            assert_base_index_is_evaluable(index)
    with pytest.raises(Phase18G3Error, match="sealed test"):
        assert_base_index_is_evaluable(450)
    with pytest.raises(Phase18G3Error, match="not an evaluation base"):
        assert_base_index_is_evaluable(399)
