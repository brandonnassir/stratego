"""Phase 6B: durable shard persistence, the memory verdict, and recycling.

The six-hour soak is not run here. What is testable without it is everything the
soak's conclusions rest on:

- a shard round-trips: what is written decodes back to the same games, and the
  manifest, the digest and the file agree;
- the container is a container -- it wraps the existing `trajectory_v1` record
  codec and adds four bytes per record, rather than being a second format;
- the default pipeline is untouched: with no output directory configured no
  worker performs file I/O and every counter means what it meant in Phase 6;
- a write backlog is structurally impossible, which is the claim the
  "unbounded write backlog = 0" gate rests on;
- the A/B/C/D memory classification returns the outcome the evidence supports,
  including the case where a real trend is still mitigable by recycling;
- recycling gives every segment its own seed and run id, which is what stops a
  restart from replaying the same games or colliding shard names.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stratego.training import phase6b_recording as p6b
from stratego.training import phase6b_recycle as recycle
from stratego.training import shard_writer as sw
from stratego.training.coordinator import ACTION_FRAME_NORMALIZED
from stratego.training.trajectory import (
    TRAJECTORY_VERSION,
    decode_game_record,
    encode_game_record,
)
from stratego.training.worker_pool import RecordingConfig

from .test_trajectory import collect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def records():
    """Six complete games from the same deterministic helper Phase 3 uses."""
    return collect(6, environments=6)


# ---------------------------------------------------------------------------
# The container
# ---------------------------------------------------------------------------


class TestShardRoundTrip:
    def test_written_records_decode_back_to_the_same_games(self, tmp_path, records):
        writer = sw.ShardWriter(tmp_path, worker_id=0, run_id="t")
        for record in records:
            writer.write(record)
        writer.close()

        paths = sw.shard_paths(tmp_path)
        assert len(paths) == 1
        result = sw.read_shard(paths[0], decode=True, keep_records=True)
        assert result["record_count"] == len(records)
        assert not result["decode_errors"]
        assert [rebuilt.game_id for rebuilt in result["records"]] == [
            record.game_id for record in records
        ]

    def test_the_payloads_are_the_existing_record_codec(self, tmp_path, records):
        """A container, not a second format: unwrap a frame and decode it directly."""
        writer = sw.ShardWriter(tmp_path, worker_id=0, run_id="t", compress_records=False)
        writer.write(records[0])
        writer.close()
        payloads = list(sw.iter_shard_payloads(sw.shard_paths(tmp_path)[0]))
        assert len(payloads) == 1
        # Uncompressed, the stored bytes are exactly what encode_game_record gives.
        assert payloads[0] == encode_game_record(records[0])
        assert decode_game_record(payloads[0]).game_id == records[0].game_id

    def test_bytes_written_equals_the_bytes_on_disk(self, tmp_path, records):
        writer = sw.ShardWriter(tmp_path, worker_id=0, run_id="t", compress_records=False)
        for record in records:
            writer.write(record)
        writer.close()
        stats = writer.stats
        on_disk = sum(path.stat().st_size for path in sw.shard_paths(tmp_path))
        assert stats.bytes_written == on_disk
        # Overhead is one preamble per shard plus four length bytes per record.
        assert stats.container_overhead_bytes > 4 * len(records)
        assert stats.container_overhead_bytes == on_disk - stats.compressed_bytes

    def test_compression_actually_compresses(self, tmp_path, records):
        writer = sw.ShardWriter(tmp_path, worker_id=0, run_id="t", compress_records=True)
        for record in records:
            writer.write(record)
        writer.close()
        assert 0.0 < writer.stats.compression_ratio < 1.0
        assert writer.stats.compressed_bytes < writer.stats.uncompressed_bytes

    def test_uncompressed_mode_has_ratio_one(self, tmp_path, records):
        writer = sw.ShardWriter(tmp_path, worker_id=0, run_id="t", compress_records=False)
        for record in records:
            writer.write(record)
        writer.close()
        assert writer.stats.compression_ratio == pytest.approx(1.0)

    def test_the_header_names_the_trajectory_version(self, tmp_path, records):
        writer = sw.ShardWriter(tmp_path, worker_id=3, run_id="abc")
        writer.write(records[0])
        writer.close()
        header = sw.read_shard_header(sw.shard_paths(tmp_path)[0])
        assert header["trajectory_version"] == TRAJECTORY_VERSION
        assert header["worker_id"] == 3
        assert header["run_id"] == "abc"
        assert header["compression"] == "zlib"


class TestShardRollover:
    def test_a_full_shard_rolls_over(self, tmp_path, records):
        writer = sw.ShardWriter(tmp_path, worker_id=0, run_id="t", target_bytes=1)
        for record in records:
            writer.write(record)
        writer.close()
        # target_bytes=1 closes after every record.
        assert len(sw.shard_paths(tmp_path)) == len(records)
        assert writer.stats.shards_closed == len(records)

    def test_a_record_is_never_split_across_shards(self, tmp_path, records):
        writer = sw.ShardWriter(tmp_path, worker_id=0, run_id="t", target_bytes=1)
        for record in records:
            writer.write(record)
        writer.close()
        for path in sw.shard_paths(tmp_path):
            result = sw.read_shard(path, decode=True)
            assert result["record_count"] == 1
            assert not result["decode_errors"]

    def test_shard_names_are_ordered_and_unique(self, tmp_path, records):
        writer = sw.ShardWriter(tmp_path, worker_id=7, run_id="run", target_bytes=1)
        for record in records:
            writer.write(record)
        writer.close()
        names = [path.name for path in sw.shard_paths(tmp_path)]
        assert len(set(names)) == len(names)
        assert names == sorted(names)
        assert all(name.startswith("run_w07_s") for name in names)


class TestManifest:
    def test_a_closed_shard_has_a_manifest_that_matches_it(self, tmp_path, records):
        writer = sw.ShardWriter(tmp_path, worker_id=0, run_id="t")
        for record in records:
            writer.write(record)
        writer.close()
        path = sw.shard_paths(tmp_path)[0]
        verified = sw.verify_shard(path, decode=True, keep_records=True)
        assert verified["ok"], verified["problems"]
        assert verified["manifest"]["records"] == len(records)
        assert verified["manifest"]["game_ids"] == [r.game_id for r in records]
        assert verified["manifest"]["sha256"] == verified["sha256"]

    def test_a_tampered_shard_fails_verification(self, tmp_path, records):
        writer = sw.ShardWriter(tmp_path, worker_id=0, run_id="t")
        writer.write(records[0])
        writer.close()
        path = sw.shard_paths(tmp_path)[0]
        blob = bytearray(path.read_bytes())
        blob[-1] ^= 0xFF
        path.write_bytes(bytes(blob))
        verified = sw.verify_shard(path, decode=True)
        assert not verified["ok"]
        assert any("sha256" in problem for problem in verified["problems"])

    def test_an_unclosed_shard_is_reported_not_raised(self, tmp_path, records):
        writer = sw.ShardWriter(tmp_path, worker_id=0, run_id="t")
        writer.write(records[0])
        writer._handle.flush()  # simulate a crash: bytes on disk, no manifest
        path = sw.shard_paths(tmp_path)[0]
        verified = sw.verify_shard(path, decode=True)
        assert verified["manifest"] is None
        assert any("manifest" in problem for problem in verified["problems"])
        # The record itself is still readable.
        assert verified["record_count"] == 1

    def test_a_truncated_tail_stops_cleanly(self, tmp_path, records):
        writer = sw.ShardWriter(tmp_path, worker_id=0, run_id="t")
        for record in records:
            writer.write(record)
        writer.close()
        path = sw.shard_paths(tmp_path)[0]
        blob = path.read_bytes()
        path.write_bytes(blob[: len(blob) - 40])
        result = sw.read_shard(path, decode=True)
        # The complete records before the truncation survive.
        assert 0 < result["record_count"] < len(records)
        assert not result["decode_errors"]


class TestDirectorySummary:
    def test_it_aggregates_every_shard(self, tmp_path, records):
        for worker in range(3):
            writer = sw.ShardWriter(tmp_path, worker_id=worker, run_id="t")
            for record in records[:2]:
                writer.write(record)
            writer.close()
        summary = sw.directory_summary(tmp_path, decode=True)
        assert summary["shard_count"] == 3
        assert summary["record_count"] == 6
        assert summary["unclosed_shards"] == 0

    def test_it_notices_duplicate_game_ids(self, tmp_path, records):
        for worker in range(2):
            writer = sw.ShardWriter(tmp_path, worker_id=worker, run_id="t")
            writer.write(records[0])  # the same game twice
            writer.close()
        summary = sw.directory_summary(tmp_path, decode=False)
        assert summary["duplicate_game_ids"] == [records[0].game_id]


class TestVerifierNeverRetainsRecords:
    """Regression: the first 6B soak's verifier held every decoded record of a
    9.2 GiB corpus in memory at once and drove the machine 28 GiB into swap.
    Decoded records must be validated and dropped unless explicitly kept."""

    def test_read_shard_drops_records_by_default(self, tmp_path, records):
        writer = sw.ShardWriter(tmp_path, worker_id=0, run_id="t")
        for record in records:
            writer.write(record)
        writer.close()
        result = sw.read_shard(sw.shard_paths(tmp_path)[0], decode=True)
        assert result["record_count"] == len(records)
        assert result["records"] == []

    def test_verify_shard_drops_records_by_default(self, tmp_path, records):
        writer = sw.ShardWriter(tmp_path, worker_id=0, run_id="t")
        for record in records:
            writer.write(record)
        writer.close()
        verified = sw.verify_shard(sw.shard_paths(tmp_path)[0], decode=True)
        assert verified["ok"]
        assert verified["record_count"] == len(records)
        assert verified["records"] == []

    def test_directory_summary_holds_no_record_objects(self, tmp_path, records):
        for worker in range(3):
            writer = sw.ShardWriter(tmp_path, worker_id=worker, run_id="t")
            for record in records:
                writer.write(record)
            writer.close()
        summary = sw.directory_summary(tmp_path, decode=True)
        assert summary["record_count"] == 3 * len(records)
        text = repr(summary)
        assert "GameRecord" not in text

    def test_directory_summary_reports_progress_per_shard(self, tmp_path, records):
        for worker in range(2):
            writer = sw.ShardWriter(tmp_path, worker_id=worker, run_id="t")
            writer.write(records[0])
            writer.close()
        seen = []
        sw.directory_summary(tmp_path, decode=True, progress=seen.append)
        assert len(seen) == 2
        assert all(shard["record_count"] == 1 for shard in seen)


class TestMemoryWatchdog:
    """Regression: a run that drives the machine into swap does not fail, it
    wedges in uninterruptible page-in waits. The watchdog aborts before that."""

    SYSTEM = {"total_bytes": 48 * 1024**3, "available_bytes": 20 * 1024**3}

    def test_quiet_when_the_machine_is_healthy(self):
        assert p6b.check_memory_watchdog(
            swap_start_bytes=0, swap_now_bytes=10**9, system=self.SYSTEM
        ) is None

    def test_a_preexisting_swap_baseline_is_not_a_trip(self):
        baseline = 25 * 1024**2
        assert p6b.check_memory_watchdog(
            swap_start_bytes=baseline, swap_now_bytes=baseline, system=self.SYSTEM
        ) is None

    def test_swap_growth_trips_it(self):
        message = p6b.check_memory_watchdog(
            swap_start_bytes=0,
            swap_now_bytes=p6b.SWAP_GROWTH_LIMIT_BYTES + 1,
            system=self.SYSTEM,
        )
        assert message is not None and "swap grew" in message

    def test_low_system_availability_trips_it(self):
        message = p6b.check_memory_watchdog(
            swap_start_bytes=0,
            swap_now_bytes=0,
            system={"total_bytes": 48 * 1024**3, "available_bytes": 1 * 1024**3},
        )
        assert message is not None and "available memory" in message


class TestBacklogIsImpossible:
    def test_pending_work_is_always_zero(self, tmp_path, records):
        writer = sw.ShardWriter(tmp_path, worker_id=0, run_id="t")
        for record in records:
            writer.write(record)
            assert writer.stats.as_dict()["pending_records"] == 0
            assert writer.stats.as_dict()["pending_bytes"] == 0
        writer.close()
        assert writer.stats.as_dict()["backlog_is_structurally_impossible"] is True

    def test_the_bytes_are_on_disk_before_write_returns(self, tmp_path, records):
        writer = sw.ShardWriter(tmp_path, worker_id=0, run_id="t")
        writer.write(records[0])
        # No close, no flush by the test: the file already holds the record.
        path = sw.shard_paths(tmp_path)[0]
        assert path.stat().st_size > 0
        assert len(list(sw.iter_shard_payloads(path))) == 1

    def test_a_write_failure_is_loud(self, tmp_path, records):
        writer = sw.ShardWriter(tmp_path, worker_id=0, run_id="t")
        writer.write(records[0])
        writer._handle.close()  # the next write cannot succeed
        with pytest.raises(sw.ShardError):
            writer.write(records[1])
        assert writer.stats.write_errors == 1


# ---------------------------------------------------------------------------
# The default pipeline is untouched
# ---------------------------------------------------------------------------


class TestDefaultPathUnchanged:
    def test_recording_config_defaults_to_no_persistence(self):
        config = RecordingConfig()
        assert config.output_directory is None
        assert config.shard_target_bytes > 0

    def test_a_pool_without_an_output_directory_builds_no_writer(self):
        from stratego.training.worker_pool import _WorkerRuntime

        assert "output_directory" in RecordingConfig.__dataclass_fields__
        # The runtime only constructs a writer when both recording and an output
        # directory are set; this asserts the guard rather than starting a pool.
        source = Path(_WorkerRuntime.__init__.__code__.co_filename).read_text()
        assert "if self.recording.enabled and self.recording.output_directory:" in source

    def test_persistence_fields_reach_the_coordinator_config(self):
        from stratego.training.coordinator import CoordinatorConfig

        config = CoordinatorConfig(
            num_environments=4, num_workers=1, inference_batch_size=4
        )
        assert config.trajectory_output_directory is None
        assert config.compress_trajectories is False
        assert "trajectory_output_directory" in config.as_dict()


# ---------------------------------------------------------------------------
# The soak configuration keeps Phase 6 frozen
# ---------------------------------------------------------------------------


class TestRecordingConfiguration:
    def test_it_preserves_the_frozen_phase_6_topology(self, tmp_path):
        config = p6b.recording_configuration(output_directory=str(tmp_path), run_id="t")
        assert config.num_workers == 10
        assert config.num_environments == 1536
        assert config.inference_batch_size == 2048
        assert config.precision == "float16"
        assert config.legality == "dense"
        assert config.snapshot_interval == 32
        assert config.action_frame == ACTION_FRAME_NORMALIZED

    def test_it_turns_on_durable_compressed_persistence(self, tmp_path):
        config = p6b.recording_configuration(output_directory=str(tmp_path), run_id="t")
        assert config.record_trajectories is True
        assert config.trajectory_output_directory == str(tmp_path)
        assert config.compress_trajectories is True

    def test_a_soak_without_an_output_directory_is_refused(self, tmp_path):
        from dataclasses import replace

        config = replace(
            p6b.recording_configuration(output_directory=str(tmp_path), run_id="t"),
            trajectory_output_directory=None,
        )
        with pytest.raises(p6b.RecordingSoakError, match="persist"):
            p6b.run_recording_soak("C1", config, seconds=1.0)

    def test_a_soak_that_does_not_record_is_refused(self, tmp_path):
        from dataclasses import replace

        config = replace(
            p6b.recording_configuration(output_directory=str(tmp_path), run_id="t"),
            record_trajectories=False,
        )
        with pytest.raises(p6b.RecordingSoakError, match="record"):
            p6b.run_recording_soak("C1", config, seconds=1.0)


# ---------------------------------------------------------------------------
# The memory verdict
# ---------------------------------------------------------------------------


SYSTEM_BYTES = 48 * 1024**3


def series(mib_per_hour: float, *, samples: int = 20, jitter=None):
    per_second = mib_per_hour * 2**20 / 3600.0
    rows = []
    for index in range(samples):
        elapsed = 60.0 * index
        value = 9e9 + per_second * elapsed
        if jitter:
            value += jitter[index % len(jitter)]
        rows.append(
            {
                "in_measured_window": True,
                "elapsed_seconds": elapsed,
                "total_rss_bytes": value,
                "per_worker_rss_bytes": {0: value / 10},
            }
        )
    return rows


class TestMemoryClassification:
    def test_a_flat_series_is_outcome_a(self):
        verdict = p6b.classify_memory_outcome(
            series(0.0), total_system_bytes=SYSTEM_BYTES
        )
        assert verdict["outcome"] == "A"

    def test_a_small_slope_below_the_threshold_is_outcome_a(self):
        verdict = p6b.classify_memory_outcome(
            series(p6b.PLATEAU_SLOPE_MIB_PER_HOUR / 2), total_system_bytes=SYSTEM_BYTES
        )
        assert verdict["outcome"] == "A"

    def test_noise_without_a_trend_is_outcome_a(self):
        # A large apparent slope but no line: scatter, not growth.
        noisy = series(0.0, jitter=[+4e8, -4e8, +2e8, -2e8])
        verdict = p6b.classify_memory_outcome(noisy, total_system_bytes=SYSTEM_BYTES)
        assert verdict["r_squared"] < p6b.GROWTH_R_SQUARED_FLOOR
        assert verdict["outcome"] == "A"

    def test_a_real_trend_that_recycling_can_bound_is_outcome_c(self):
        verdict = p6b.classify_memory_outcome(
            series(191.0), total_system_bytes=SYSTEM_BYTES
        )
        assert verdict["outcome"] == "C"
        assert verdict["required_restart_interval_hours"] > p6b.MINIMUM_PRACTICAL_RESTART_HOURS

    def test_growth_too_fast_for_any_practical_interval_is_outcome_d(self):
        verdict = p6b.classify_memory_outcome(
            series(20_000.0), total_system_bytes=SYSTEM_BYTES
        )
        assert verdict["outcome"] == "D"
        assert verdict["required_restart_interval_hours"] < p6b.MINIMUM_PRACTICAL_RESTART_HOURS

    def test_a_large_projection_alone_does_not_mean_blocked(self):
        """The C/D split is about whether recycling can bound it, not about size.

        This is the correction that matters: a 168-hour projection far exceeding
        system memory is exactly the situation recycling exists for, so it must
        not be classified BLOCKED on magnitude alone.
        """
        verdict = p6b.classify_memory_outcome(
            series(500.0), total_system_bytes=SYSTEM_BYTES
        )
        assert verdict["projected_168h_bytes"] > SYSTEM_BYTES
        assert verdict["outcome"] == "C"

    def test_it_reports_the_thresholds_it_used(self):
        verdict = p6b.classify_memory_outcome(
            series(0.0), total_system_bytes=SYSTEM_BYTES
        )
        thresholds = verdict["thresholds"]
        assert thresholds["plateau_slope_mib_per_hour"] == p6b.PLATEAU_SLOPE_MIB_PER_HOUR
        assert thresholds["growth_r_squared_floor"] == p6b.GROWTH_R_SQUARED_FLOOR
        assert thresholds["minimum_practical_restart_hours"] == (
            p6b.MINIMUM_PRACTICAL_RESTART_HOURS
        )

    def test_it_needs_enough_settled_samples(self):
        with pytest.raises(p6b.RecordingSoakError, match="four settled"):
            p6b.classify_memory_outcome(
                series(0.0, samples=2), total_system_bytes=SYSTEM_BYTES
            )

    def test_per_worker_slopes_are_reported(self):
        verdict = p6b.classify_memory_outcome(
            series(100.0), total_system_bytes=SYSTEM_BYTES
        )
        assert verdict["per_worker_slope_bytes_per_hour"]


class TestRestartInterval:
    def test_the_interval_comes_from_the_measured_slope(self):
        verdict = p6b.classify_memory_outcome(
            series(200.0), total_system_bytes=SYSTEM_BYTES
        )
        budget = 12 * 1024**3
        interval = p6b.recommended_restart_interval_hours(verdict, budget_bytes=budget)
        assert interval["required"] is True
        # 12 GiB at 200 MiB/hour is a little over 61 hours.
        assert interval["interval_hours"] == pytest.approx(
            budget / verdict["slope_bytes_per_hour"]
        )
        assert 55 < interval["interval_hours"] < 70

    def test_no_positive_slope_needs_no_recycling(self):
        verdict = p6b.classify_memory_outcome(
            series(0.0), total_system_bytes=SYSTEM_BYTES
        )
        interval = p6b.recommended_restart_interval_hours(
            verdict, budget_bytes=12 * 1024**3
        )
        assert interval["required"] is False


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------


class TestStorageProjection:
    def test_the_week_is_604800_seconds_of_the_measured_disk_rate(self):
        steady = {
            "window_seconds": 3600.0,
            "write_throughput_bytes_per_second": 1_000_000.0,
            "produced_gib_per_hour": 5.0,
            "compression_ratio": 0.69,
            "bytes_per_decision_written": 130.0,
        }
        projection = p6b.storage_projection_from_disk(
            steady, volume_total_bytes=1000 * 1024**3, volume_free_bytes=700 * 1024**3
        )
        assert projection["extrapolated"]["bytes_per_168_hours"] == pytest.approx(
            1_000_000.0 * 604800
        )
        assert p6b.FINAL_RUN_SECONDS == 604800.0

    def test_it_reserves_headroom_for_the_open_shards(self):
        steady = {
            "window_seconds": 3600.0,
            "write_throughput_bytes_per_second": 1.0,
            "produced_gib_per_hour": 1.0,
            "compression_ratio": 0.7,
            "bytes_per_decision_written": 1.0,
        }
        projection = p6b.storage_projection_from_disk(
            steady, volume_total_bytes=10**12, volume_free_bytes=10**12,
            shard_target_bytes=128 * 1024**2,
        )
        # One open shard per worker is the only transient the writer needs.
        assert projection["extrapolated"]["shard_headroom_bytes"] == (
            128 * 1024**2 * 10
        )

    def test_it_reports_against_total_and_against_free(self):
        steady = {
            "window_seconds": 3600.0,
            "write_throughput_bytes_per_second": 1_000_000.0,
            "produced_gib_per_hour": 5.0,
            "compression_ratio": 0.69,
            "bytes_per_decision_written": 130.0,
        }
        projection = p6b.storage_projection_from_disk(
            steady, volume_total_bytes=1000 * 1024**3, volume_free_bytes=100 * 1024**3
        )
        assert projection["volume"]["fits_in_total"] is True
        assert projection["volume"]["fits_in_free_now"] is False


# ---------------------------------------------------------------------------
# Recycling
# ---------------------------------------------------------------------------


class TestSegmentIdentity:
    def test_every_segment_gets_its_own_seed(self):
        seeds = [recycle.segment_root_seed(60_006, index) for index in range(20)]
        assert len(set(seeds)) == len(seeds)
        assert seeds[0] == 60_006

    def test_the_seed_stride_is_wide_enough_to_never_overlap(self):
        # Two adjacent segments must not be able to reach the same
        # (root_seed, environment, generation) identity.
        assert recycle.SEGMENT_SEED_STRIDE > 1_000_000

    def test_every_segment_gets_its_own_run_id(self):
        ids = [recycle.segment_run_id("run", index) for index in range(20)]
        assert len(set(ids)) == len(ids)

    def test_seeds_are_reproducible_from_the_base(self):
        assert recycle.segment_root_seed(1, 3) == recycle.segment_root_seed(1, 3)


class TestRecycleSummaryValidation:
    @staticmethod
    def supervisor(tmp_path):
        return recycle.RecyclingSupervisor(
            output_directory=tmp_path / "out",
            state_directory=tmp_path / "state",
            base_run_id="t",
        )

    @staticmethod
    def segment_state(index: int, **overrides) -> dict:
        state = {
            "segment": index,
            "run_id": recycle.segment_run_id("t", index),
            "root_seed": recycle.segment_root_seed(60_006, index),
            "configuration_digest": "deadbeef",
            "status": "ok",
            "exit_code": 0,
            "seconds": 100.0,
            "supervisor_wall_seconds": 101.0,
            "startup_shutdown_overhead_seconds": 1.0,
            "positions": 1000,
            "games": 10,
            "decisions_recorded": 1000,
            "records_persisted": 0,
            "bytes_produced": 100,
            "bytes_written": 90,
            "compressed_bytes": 90,
            "shards_closed": 0,
            "rss_at_start_bytes": 200 * 2**20,
            "rss_at_end_bytes": 1500 * 2**20,
            "rss_growth_bytes": 1300 * 2**20,
            "verified_decisions": 10,
            "reconstruction_mismatches": 0,
            "write_errors": 0,
            "failures": {
                "illegal_actions": 0, "action_frame_errors": 0, "worker_errors": 0,
                "model_errors": 0, "nonfinite_outputs": 0, "other_errors": 0,
            },
        }
        state.update(overrides)
        return state

    def test_a_clean_recycled_run_validates(self, tmp_path):
        supervisor = self.supervisor(tmp_path)
        supervisor.segments = [self.segment_state(index) for index in range(3)]
        report = supervisor.summarize(elapsed_wall_seconds=303.0)
        assert report["ok"], report["problems"]
        assert report["segments_run"] == 3
        assert report["rss_returns_to_baseline"] is True

    def test_restart_overhead_is_inside_the_wall_clock(self, tmp_path):
        supervisor = self.supervisor(tmp_path)
        supervisor.segments = [self.segment_state(index) for index in range(3)]
        report = supervisor.summarize(elapsed_wall_seconds=303.0)
        assert report["restart_overhead_seconds"] == pytest.approx(3.0)
        assert report["collection_seconds"] == pytest.approx(300.0)
        assert report["restart_overhead_seconds"] + report["collection_seconds"] == (
            pytest.approx(report["elapsed_wall_seconds"])
        )

    def test_a_shared_seed_is_caught(self, tmp_path):
        supervisor = self.supervisor(tmp_path)
        supervisor.segments = [
            self.segment_state(0),
            self.segment_state(1, root_seed=recycle.segment_root_seed(60_006, 0)),
        ]
        report = supervisor.summarize(elapsed_wall_seconds=200.0)
        assert not report["ok"]
        assert any("root seed" in problem for problem in report["problems"])

    def test_a_shared_run_id_is_caught(self, tmp_path):
        supervisor = self.supervisor(tmp_path)
        supervisor.segments = [
            self.segment_state(0),
            self.segment_state(1, run_id=recycle.segment_run_id("t", 0)),
        ]
        report = supervisor.summarize(elapsed_wall_seconds=200.0)
        assert not report["ok"]
        assert any("run id" in problem for problem in report["problems"])

    def test_a_configuration_change_between_segments_is_caught(self, tmp_path):
        supervisor = self.supervisor(tmp_path)
        supervisor.segments = [
            self.segment_state(0),
            self.segment_state(1, configuration_digest="different"),
        ]
        report = supervisor.summarize(elapsed_wall_seconds=200.0)
        assert not report["ok"]
        assert any("digest" in problem for problem in report["problems"])

    def test_a_failed_correctness_counter_is_caught(self, tmp_path):
        supervisor = self.supervisor(tmp_path)
        bad = self.segment_state(1)
        bad["failures"]["illegal_actions"] = 1
        supervisor.segments = [self.segment_state(0), bad]
        report = supervisor.summarize(elapsed_wall_seconds=200.0)
        assert not report["ok"]
        assert any("illegal_actions" in problem for problem in report["problems"])

    def test_memory_not_returning_to_baseline_is_caught(self, tmp_path):
        supervisor = self.supervisor(tmp_path)
        supervisor.segments = [
            self.segment_state(0),
            self.segment_state(1, rss_at_start_bytes=600 * 2**20),
        ]
        report = supervisor.summarize(elapsed_wall_seconds=200.0)
        assert report["rss_returns_to_baseline"] is False
        assert not report["ok"]

    def test_run_totals_accumulate_across_segments(self, tmp_path):
        supervisor = self.supervisor(tmp_path)
        supervisor.segments = [self.segment_state(index) for index in range(4)]
        report = supervisor.summarize(elapsed_wall_seconds=404.0)
        assert report["totals"]["positions"] == 4000
        assert report["totals"]["games"] == 40

    def test_no_segments_is_an_error(self, tmp_path):
        with pytest.raises(recycle.RecycleError, match="no segments"):
            self.supervisor(tmp_path).summarize(elapsed_wall_seconds=1.0)


# ---------------------------------------------------------------------------
# Published artifacts
# ---------------------------------------------------------------------------

DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "reports" / "phase_6_data"


@pytest.mark.skipif(
    not (DATA_DIRECTORY / "agent_06b_recording_soak.json").exists(),
    reason="Phase 6B has not produced its artifacts yet",
)
class TestPublishedArtifacts:
    """The artifacts must be internally consistent with their own verdict.

    Phase 6B may legitimately be BLOCKED -- the first soak was -- so these tests
    do not assert success. They assert that whatever status the artifact
    records, the evidence required for that status is present: a PASS must
    carry every hard gate true, and a BLOCKED must carry the failure it blocked
    on. The strict PASS-shaped assertions arm themselves automatically the day
    a passing soak writes PASS artifacts.
    """

    @staticmethod
    def soak() -> dict:
        return json.loads((DATA_DIRECTORY / "agent_06b_recording_soak.json").read_text())

    def test_the_artifact_names_its_own_verdict(self):
        soak = self.soak()
        assert soak["phase_6b_recommendation"] in ("PASS", "BLOCKED", "FAIL")
        assert "completion_gates" in soak

    def test_disk_persistence_was_actually_exercised(self):
        soak = self.soak()
        assert soak["shard_verification"]["record_count"] > 0
        assert soak["shard_verification"]["shard_count"] > 0

    def test_compression_was_exercised(self):
        assert 0.0 < self.soak()["steady_state"]["compression_ratio"] < 0.95

    def test_a_pass_requires_every_hard_gate_and_the_full_duration(self):
        soak = self.soak()
        if soak["phase_6b_recommendation"] != "PASS":
            pytest.skip("Phase 6B is not claiming PASS")
        assert soak["soak"]["total_seconds"] >= p6b.MINIMUM_SOAK_SECONDS
        assert soak["memory_verdict"]["outcome"] in ("A", "B", "C")
        gates = soak["completion_gates"]
        for name in (
            "illegal_actions_zero", "action_frame_mismatches_zero",
            "reconstruction_mismatches_zero", "worker_failures_zero",
            "model_mps_failures_zero", "nonfinite_outputs_zero",
            "write_errors_zero", "write_backlog_bounded", "no_swap_growth",
            "shards_all_decode", "no_duplicate_games", "no_unclosed_shards",
        ):
            assert gates[name] is True, name

    def test_a_blocked_verdict_records_what_blocked_it(self):
        soak = self.soak()
        if soak["phase_6b_recommendation"] != "BLOCKED":
            pytest.skip("Phase 6B is not BLOCKED")
        assert soak.get("blocking_findings"), "BLOCKED without recorded findings"
        # The soak evidence must still be present and honest.
        assert soak["soak"]["status"] in ("ok", "error")
        assert soak["sample_count"] > 0

    def test_the_storage_projection_uses_the_measured_disk_rate(self):
        storage = json.loads(
            (DATA_DIRECTORY / "agent_06b_storage_validation.json").read_text()
        )
        steady = self.soak()["steady_state"]
        measured = storage["projection"]["measured"]
        assert measured["write_throughput_bytes_per_second"] == pytest.approx(
            steady["write_throughput_bytes_per_second"]
        )


@pytest.mark.skipif(
    not (DATA_DIRECTORY / "agent_06b_anomaly_fix_validation.json").exists(),
    reason="the Gate 1 fix validation has not been run yet",
)
class TestAnomalyFixValidationArtifact:
    """The Gate 1 acceptance artifact: the formerly failing sequence passes."""

    @staticmethod
    def payload() -> dict:
        return json.loads(
            (DATA_DIRECTORY / "agent_06b_anomaly_fix_validation.json").read_text()
        )

    def test_the_exact_identity_is_the_soak_abort_identity(self):
        identity = self.payload()["failing_identity"]
        assert identity["root_seed"] == 60006
        assert identity["environment_id"] == 112
        assert identity["generation"] == 98
        assert identity["game_id"] == "batch60006-env000112-gen000098"

    def test_every_stage_confirms_the_fix(self):
        payload = self.payload()
        assert payload["formerly_failing_sequence_now_passes"] is True
        for name, stage in payload["stages"].items():
            assert stage.get("correct", stage.get("skipped")) is True, name

    def test_the_engine_state_is_terminal_with_the_authorized_semantics(self):
        engine = self.payload()["stages"]["engine_state"]
        assert engine["terminal"] is True
        assert engine["terminal_reason"] == "opponent_no_legal_move"
        assert engine["winner_is_blue"] is True
        assert engine["legal_actions_for_acting_player"] == 0
        assert engine["game_end_events"] == 1

    def test_the_probability_is_the_exact_reciprocal(self):
        assert self.payload()["probability_one_in_exact"] == 548_340


@pytest.mark.skipif(
    not (DATA_DIRECTORY / "agent_06b_final_soak.json").exists(),
    reason="the final recycled soak has not been run yet",
)
class TestFinalSoakArtifacts:
    """The Gate 2 artifacts: internally consistent with their own verdict."""

    @staticmethod
    def payload() -> dict:
        return json.loads((DATA_DIRECTORY / "agent_06b_final_soak.json").read_text())

    def test_the_artifact_names_its_own_verdict(self):
        payload = self.payload()
        assert payload["phase_6b_final_recommendation"] in ("PASS", "FAIL")
        assert "completion_gates" in payload

    def test_the_soak_is_recycled_not_continuous(self):
        payload = self.payload()
        assert payload["supervisor_summary"]["segments_run"] >= 2
        seeds = payload["supervisor_summary"]["segment_root_seeds"]
        assert len(set(seeds)) == len(seeds)

    def test_a_pass_requires_every_hard_gate_and_the_duration(self):
        payload = self.payload()
        if payload["phase_6b_final_recommendation"] != "PASS":
            pytest.skip("the final soak is not claiming PASS")
        gates = payload["completion_gates"]
        for name, value in gates.items():
            assert value is True, name
        assert payload["supervisor_summary"]["elapsed_wall_seconds"] >= 4 * 3600.0
        assert payload["supervisor_summary"]["segments_run"] >= 4
        assert payload["shard_verification"]["record_count"] > 0
        assert payload["shard_verification"]["verifier_peak_rss_mib"] < 512.0

    def test_stillborn_accounting_is_reconciled(self):
        payload = self.payload()
        persisted = payload["persisted_stillborn_records"]
        assert len(persisted) == payload["stillborn_counted_by_workers"]
        for entry in persisted:
            assert entry["final_ply"] == 0
            assert entry["decisions"] == 0
            assert entry["terminal_reason"] in (
                "opponent_no_legal_move",
                "both_no_legal_move_draw",
            )

    def test_the_storage_projection_artifact_matches_the_soak(self):
        storage = json.loads(
            (DATA_DIRECTORY / "agent_06b_final_storage_validation.json").read_text()
        )
        payload = self.payload()
        assert storage["run_id"] == payload["run_id"]
        assert storage["configuration_digest"] == payload["configuration_digest"]
        extrapolated = storage["projection"]["extrapolated_168h"]
        assert extrapolated["gib"] > 0

    def test_the_recycling_artifact_matches_the_soak(self):
        recycling = json.loads(
            (DATA_DIRECTORY / "agent_06b_recycling_validation.json").read_text()
        )
        payload = self.payload()
        assert recycling["run_id"] == payload["run_id"]
        summary = recycling["recycling"]
        assert summary["segments_run"] == payload["supervisor_summary"]["segments_run"]
        assert len(summary["rss_at_segment_start_bytes"]) == summary["segments_run"]
