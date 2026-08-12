"""Phase 6 Agent 6: the soak harness, the projection and the selection rule.

The soak itself is an hour long and is not run here. What is testable without
running it is everything that decides what the soak measures and what its
numbers are then allowed to mean:

- the soak configuration really is Agent 4's production topology, in the
  normalized frame, with recording on;
- the sampling statistics say what they claim about growth and drift;
- the gates are wired to the counters they name, and a fault in any one of them
  turns the corresponding gate false;
- the projection is arithmetic on 604,800 seconds and keeps its measured inputs
  separate from its extrapolated outputs;
- the selection rule cannot see a playing-strength quantity, and adding one
  changes nothing.

A short soak *is* run, once, against the real pipeline, so the harness is not
only unit-tested against its own dataclasses. It uses a tiny topology and a
handful of seconds, which is enough to drive the whole chain -- real candidate,
normalized frame, engine validation, production recording, in-worker
reconstruction and the finiteness probe -- without being a benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stratego.model.architecture_configs import (
    ARCHITECTURE_FAMILY,
    FAMILY_INITIALIZATION_SEED,
    candidate_config,
)
from stratego.model.contract import MODEL_CONTRACT_VERSION
from stratego.training import phase6_soak as soak
from stratego.training.coordinator import ACTION_FRAME_NORMALIZED
from stratego.training.phase6_pipeline_benchmark import FORBIDDEN_INPUT_SUBSTRINGS
from stratego.training.trajectory import TRAJECTORY_VERSION

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_6_data"


# ---------------------------------------------------------------------------
# Fixtures: the measured frontier, as Agent 6 assembles it
# ---------------------------------------------------------------------------


def frontier_row(
    candidate_id: str,
    *,
    parameters: int,
    recording: float,
    collection: float | None = None,
    training: float = 1000.0,
    float16: float = 10000.0,
    float32: float = 8000.0,
    gib_per_hour: float = 5.0,
    stable: bool = True,
    **extra,
) -> dict:
    row = {
        "candidate_id": candidate_id,
        "parameters": parameters,
        "standalone_float32_positions_per_second": float32,
        "standalone_float16_positions_per_second": float16,
        "training_examples_per_second": training,
        "collection_positions_per_second": collection
        if collection is not None
        else recording * 1.25,
        "recording_positions_per_second": recording,
        "gib_per_hour": gib_per_hour,
        "process_rss_bytes": 4 * 1024**3,
        "metal_memory_bytes": 3 * 1024**3,
        "checkpoint_bytes": parameters * 4,
        "numerically_stable_float16": stable,
        "bottleneck_ratio": 5.0,
    }
    row.update(extra)
    return row


@pytest.fixture
def measured_frontier() -> list[dict]:
    """The real Agents 2/3/4 numbers for C0, C1, C2 and C3."""
    return [
        frontier_row(
            "C0",
            parameters=123_223,
            recording=12_689.281717751446,
            collection=17_451.020938018,
            training=9084.3,
            float16=27_156.32,
            float32=25_363.42,
            gib_per_hour=7.303320575671577,
        ),
        frontier_row(
            "C1",
            parameters=863_959,
            recording=9420.104770206057,
            collection=11_874.79916506647,
            training=3045.75,
            float16=14_919.31,
            float32=12_304.37,
            gib_per_hour=5.651265370275841,
        ),
        frontier_row(
            "C2",
            parameters=1_922_519,
            recording=6486.0,
            collection=7495.0,
            training=1854.28,
            float16=8635.93,
            float32=6785.31,
            gib_per_hour=3.89,
        ),
        frontier_row(
            "C3",
            parameters=2_812_247,
            recording=4886.0,
            collection=5496.0,
            training=1293.69,
            float16=6071.22,
            float32=4811.58,
            gib_per_hour=2.97,
        ),
    ]


def sample_row(index: int, **overrides) -> dict:
    """One time-series row shaped like the ones `run_soak` emits."""
    row = {
        "candidate_id": "C1",
        "sample_index": index,
        "elapsed_seconds": 60.0 * (index + 1),
        "global_step": 400 * (index + 1),
        "in_measured_window": True,
        "positions": 600_000 * (index + 1),
        "window_positions": 600_000,
        "positions_per_second": 10_000.0,
        "cumulative_positions_per_second": 10_000.0,
        "games": 1000 * (index + 1),
        "window_games": 1000,
        "games_per_second": 16.6,
        "mean_game_length": 500.0,
        "terminal_reason_counts": {"flag_capture": 900},
        "worker_failures": 0,
        "model_failures": 0,
        "nonfinite_outputs": 0,
        "illegal_actions": 0,
        "action_frame_errors": 0,
        "workers_alive": 10,
        "sampled_legality_checks": 600_000 * (index + 1),
        "verified_games": 10 * (index + 1),
        "verified_decisions": 5000 * (index + 1),
        "reconstruction_mismatches": 0,
        "decisions_recorded": 600_000 * (index + 1),
        "trajectory_bytes": 110_000_000 * (index + 1),
        "window_trajectory_bytes": 110_000_000,
        "gib_per_hour": 6.1,
        "bytes_per_decision": 187.0,
        "snapshot_count": 100 * (index + 1),
        "coordinator_rss_bytes": 4 * 1024**3,
        "worker_rss_bytes": 3 * 1024**3,
        "total_rss_bytes": 7 * 1024**3,
        "worker_processes": 10,
        "shared_memory_bytes": 90 * 1024**2,
        "metal_current_allocated_bytes": 600_000,
        "metal_driver_allocated_bytes": 3 * 1024**3,
        "swap_used_bytes": 0,
        "probe_rows_checked": 512,
        "probe_logits_checked": 512 * (10_000 + 3 + 1200),
        "cumulative_probe_seconds": 0.05 * (index + 1),
    }
    row.update(overrides)
    return row


class _Marker:
    """Stands in for the warmup baseline `run_soak` records."""

    def __init__(self, **fields):
        for name, value in fields.items():
            setattr(self, name, value)


def summarize(samples, *, failures=None, totals=None, requested=None, status="ok"):
    if requested is None:
        requested = samples[-1]["elapsed_seconds"] if samples else 0.0
    warmup = _Marker(
        elapsed=0.0, step=0, positions=0, games=0, record_bytes=0, decisions=0,
        verified_decisions=0,
    )
    return soak.summarize_soak(
        candidate_id="C1",
        config=soak.soak_configuration("C1"),
        samples=samples,
        warmup_marker=warmup,
        warmup_steps=0,
        total_seconds=samples[-1]["elapsed_seconds"] if samples else 0.0,
        totals=totals or {"total_verified_decisions": 5000, "total_verified_games": 10},
        failures=failures or {
            "worker_errors": 0, "model_errors": 0, "nonfinite_outputs": 0,
            "illegal_actions": 0, "action_frame_errors": 0, "other_errors": 0,
        },
        parameters=863_959,
        status=status,
        error_text=None,
        error_category=None,
        probe_seconds=1.0,
        probe_logits=10_000,
        probe_rows_checked=512,
        worker_liveness_checks=600,
        swap_start={"swap_used_bytes": 0},
        swap_end={"swap_used_bytes": 0},
        metal_start={},
        metal_end={},
        requested_seconds=requested,
    )


# ---------------------------------------------------------------------------
# The soak configuration is the topology the instruction names
# ---------------------------------------------------------------------------


class TestSoakConfiguration:
    def test_it_is_agent_4s_best_defensible_production_topology(self):
        config = soak.soak_configuration("C1")
        assert config.num_workers == 10
        assert config.num_environments == 1536
        assert config.inference_batch_size == 2048
        assert config.precision == "float16"
        assert config.legality == "dense"
        assert config.snapshot_interval == 32

    def test_it_records_production_trajectories(self):
        config = soak.soak_configuration("C1")
        assert config.record_trajectories is True
        assert TRAJECTORY_VERSION == "trajectory_v1"

    def test_the_model_acts_in_the_normalized_frame(self):
        config = soak.soak_configuration("C1")
        assert config.action_frame == ACTION_FRAME_NORMALIZED
        assert MODEL_CONTRACT_VERSION == "model_contract_v2"

    def test_every_sampled_action_is_checked_against_its_own_mask(self):
        assert soak.soak_configuration("C1").verify_sampled_legality is True

    def test_detailed_timing_is_off_because_a_soak_measures_production(self):
        assert soak.soak_configuration("C1").detailed_timing is False

    def test_verification_is_configured_to_outlast_the_hour(self):
        config = soak.soak_configuration("C1")
        # A budget the run can exhaust would stop checking reconstruction partway
        # through, which is the opposite of what a soak is for.
        assert config.verify_target_decisions >= 1_000_000
        assert config.max_concurrent_verifications == 1

    def test_no_game_is_retained_so_nothing_accumulates_in_a_worker(self):
        assert soak.soak_configuration("C1").retain_games == 0

    def test_a_non_recording_configuration_is_refused(self):
        config = soak.soak_configuration("C1")
        object.__setattr__(config, "record_trajectories", False)
        with pytest.raises(soak.SoakError, match="production-recording"):
            soak.run_soak("C1", config, seconds=1.0)


# ---------------------------------------------------------------------------
# The statistics
# ---------------------------------------------------------------------------


class TestGrowthStatistics:
    def test_a_flat_series_has_no_growth(self):
        report = soak.growth_report("rss", [0.0, 60.0, 120.0, 180.0], [100.0] * 4)
        assert report["relative_change"] == 0.0
        assert report["slope_per_hour"] == 0.0
        assert report["within_tolerance"] is True

    def test_a_rising_series_is_caught(self):
        elapsed = [60.0 * index for index in range(10)]
        values = [100.0 + 10.0 * index for index in range(10)]
        report = soak.growth_report("rss", elapsed, values)
        assert report["relative_change"] > soak.MEMORY_GROWTH_TOLERANCE
        assert report["within_tolerance"] is False
        assert report["slope_per_hour"] == pytest.approx(600.0)
        assert report["r_squared"] == pytest.approx(1.0)

    def test_noise_smaller_than_the_tolerance_is_not_growth(self):
        elapsed = [60.0 * index for index in range(20)]
        values = [100.0 + (1.0 if index % 2 else -1.0) for index in range(20)]
        report = soak.growth_report("rss", elapsed, values)
        assert report["within_tolerance"] is True

    def test_half_over_half_compares_the_two_halves(self):
        halves = soak.half_over_half([10.0, 10.0, 20.0, 20.0])
        assert halves["first_half_mean"] == 10.0
        assert halves["second_half_mean"] == 20.0
        assert halves["relative_change"] == pytest.approx(1.0)

    def test_a_single_sample_cannot_show_a_trend(self):
        report = soak.growth_report("rss", [0.0], [100.0])
        assert report["slope_per_hour"] == 0.0
        assert report["within_tolerance"] is True


# ---------------------------------------------------------------------------
# The gates are wired to the counters they name
# ---------------------------------------------------------------------------


class TestSoakGates:
    def test_a_clean_run_passes_every_gate(self):
        summary = summarize([sample_row(index) for index in range(10)])
        assert summary["passed"] is True
        assert summary["gates_true"] == summary["gates_total"]

    @pytest.mark.parametrize(
        "counter,gate",
        [
            ("illegal_actions", "illegal_actions_zero"),
            ("action_frame_errors", "action_frame_mismatches_zero"),
            ("worker_errors", "worker_failures_zero"),
            ("model_errors", "model_mps_failures_zero"),
            ("nonfinite_outputs", "nonfinite_production_outputs_zero"),
            ("other_errors", "other_failures_zero"),
        ],
    )
    def test_each_failure_counter_turns_its_own_gate_false(self, counter, gate):
        failures = {
            "worker_errors": 0, "model_errors": 0, "nonfinite_outputs": 0,
            "illegal_actions": 0, "action_frame_errors": 0, "other_errors": 0,
        }
        failures[counter] = 1
        summary = summarize([sample_row(index) for index in range(10)], failures=failures)
        assert summary["completion_gates"][gate] is False
        assert summary["passed"] is False

    def test_a_reconstruction_mismatch_fails_its_gate(self):
        summary = summarize(
            [sample_row(index) for index in range(10)],
            totals={
                "total_reconstruction_mismatches": 1,
                "total_verified_decisions": 5000,
                "total_verified_games": 10,
            },
        )
        assert summary["completion_gates"]["reconstruction_mismatches_zero"] is False
        assert summary["passed"] is False

    def test_any_swap_fails_the_swap_gate(self):
        samples = [sample_row(index) for index in range(10)]
        samples[4]["swap_used_bytes"] = 1024
        summary = summarize(samples)
        assert summary["completion_gates"]["swap_zero"] is False
        assert summary["memory"]["swap_used_bytes_max"] == 1024

    def test_persistent_memory_growth_fails_its_gate(self):
        samples = [
            sample_row(index, coordinator_rss_bytes=4 * 1024**3 + index * 300 * 1024**2)
            for index in range(10)
        ]
        summary = summarize(samples)
        assert summary["completion_gates"]["no_unexplained_memory_growth"] is False
        assert summary["memory_growth"]["coordinator_rss_bytes"]["within_tolerance"] is False

    def test_a_short_run_does_not_count_as_a_completed_soak(self):
        samples = [sample_row(index) for index in range(3)]
        summary = summarize(samples, requested=3600.0)
        assert summary["completion_gates"]["soak_completed_continuously"] is False

    def test_verification_must_still_be_running_at_the_end(self):
        # Every sample reports the same cumulative verified count, which is what a
        # budget exhausted early in the run looks like.
        samples = [sample_row(index, verified_decisions=5000) for index in range(10)]
        summary = summarize(samples)
        assert summary["completion_gates"]["reconstruction_ran_throughout"] is False

    def test_the_whole_run_and_the_steady_window_are_reported_separately(self):
        samples = [sample_row(index) for index in range(10)]
        summary = summarize(samples)
        assert summary["whole_run"]["positions"] == samples[-1]["positions"]
        assert summary["steady_state"]["positions_per_second"] > 0
        assert "window_seconds" in summary["steady_state"]


class TestDriftReporting:
    def test_a_steady_run_reports_no_meaningful_drift(self):
        summary = summarize([sample_row(index) for index in range(10)])
        assert summary["throughput_drift"]["small_and_stable"] is True
        assert abs(summary["throughput_drift"]["relative_drift_per_hour"]) < 1e-9

    def test_a_decaying_run_reports_drift(self):
        samples = [
            sample_row(index, positions_per_second=10_000.0 - 400.0 * index)
            for index in range(10)
        ]
        summary = summarize(samples)
        assert summary["throughput_drift"]["relative_drift_per_hour"] < 0
        assert summary["throughput_drift"]["small_and_stable"] is False


# ---------------------------------------------------------------------------
# The 168-hour projection
# ---------------------------------------------------------------------------


class TestWeeklyProjection:
    def test_the_week_is_exactly_604800_seconds(self):
        assert soak.FINAL_RUN_SECONDS == 604800.0
        assert soak.FINAL_RUN_HOURS == 168.0
        assert soak.FINAL_RUN_HOURS * 3600 == soak.FINAL_RUN_SECONDS

    def test_positions_are_throughput_times_the_week(self):
        projection = soak.weekly_projection(
            candidate_id="C1",
            positions_per_second=9000.0,
            games_per_second=17.0,
            bytes_per_second=1_700_000.0,
            bytes_per_decision=187.0,
            checkpoint_bytes=3_473_613,
            training_examples_per_second=3045.75,
            measurement_source="test",
            measured_seconds=3400.0,
        )
        assert projection["extrapolated"]["positions"] == pytest.approx(9000.0 * 604800)
        assert projection["extrapolated"]["games"] == pytest.approx(17.0 * 604800)

    def test_measured_inputs_and_extrapolations_are_in_separate_blocks(self):
        projection = soak.weekly_projection(
            candidate_id="C1", positions_per_second=9000.0, games_per_second=17.0,
            bytes_per_second=1_700_000.0, bytes_per_decision=187.0,
            checkpoint_bytes=3_473_613, training_examples_per_second=3045.75,
            measurement_source="soak", measured_seconds=3400.0,
        )
        assert set(projection) >= {"measured", "extrapolated"}
        assert "positions" not in projection["measured"]
        assert (
            projection["measured"]["recording_inclusive_positions_per_second"] == 9000.0
        )

    def test_checkpoint_storage_is_given_for_several_frequencies(self):
        projection = soak.weekly_projection(
            candidate_id="C1", positions_per_second=9000.0, games_per_second=17.0,
            bytes_per_second=1_700_000.0, bytes_per_decision=187.0,
            checkpoint_bytes=1_000_000, training_examples_per_second=3045.75,
            measurement_source="soak", measured_seconds=3400.0,
        )
        schedules = projection["extrapolated"]["checkpoint_storage"]
        assert schedules["hourly"]["checkpoints_retained"] == 168
        assert schedules["daily"]["checkpoints_retained"] == 7
        assert schedules["hourly"]["bytes"] == 168_000_000

    def test_it_refuses_to_project_from_a_non_positive_rate(self):
        with pytest.raises(ValueError, match="positive"):
            soak.weekly_projection(
                candidate_id="C1", positions_per_second=0.0, games_per_second=0.0,
                bytes_per_second=0.0, bytes_per_decision=0.0, checkpoint_bytes=1,
                training_examples_per_second=1.0, measurement_source="", measured_seconds=1.0,
            )

    def test_it_says_out_loud_that_this_is_not_a_learning_claim(self):
        projection = soak.weekly_projection(
            candidate_id="C1", positions_per_second=9000.0, games_per_second=17.0,
            bytes_per_second=1_700_000.0, bytes_per_decision=187.0,
            checkpoint_bytes=1, training_examples_per_second=1.0,
            measurement_source="soak", measured_seconds=3400.0,
        )
        assert "not" in projection["extrapolation_is_not_a_learning_claim"].lower()
        assert "strong" in projection["extrapolation_is_not_a_learning_claim"].lower()


class TestStorageAnalysis:
    def test_it_compares_against_the_users_declared_capacity(self):
        analysis = soak.storage_analysis(
            candidate_id="C1",
            trajectory_bytes_168h=900 * 1000**3,
            checkpoint_bytes=3_473_613,
            measured_bytes_per_decision=187.0,
        )
        assert analysis["declared_capacity"]["internal_free_gb"] == pytest.approx(150.0)
        assert analysis["declared_capacity"]["external_free_gb"] == pytest.approx(1000.0)
        assert analysis["uncompressed"]["fits_internal"] is False
        assert analysis["uncompressed"]["fits_external"] is True

    def test_an_overflowing_week_does_not_fit_either_volume(self):
        analysis = soak.storage_analysis(
            candidate_id="C0",
            trajectory_bytes_168h=1_300 * 1000**3,
            checkpoint_bytes=504_965,
            measured_bytes_per_decision=186.0,
        )
        assert analysis["uncompressed"]["fits_external"] is False
        assert analysis["uncompressed"]["fraction_of_external"] > 1.0

    def test_compression_is_only_reported_when_it_was_measured(self):
        without = soak.storage_analysis(
            candidate_id="C1", trajectory_bytes_168h=10**12,
            checkpoint_bytes=1, measured_bytes_per_decision=187.0,
        )
        assert "compressed" not in without

        with_ratio = soak.storage_analysis(
            candidate_id="C1", trajectory_bytes_168h=10**12,
            checkpoint_bytes=1, measured_bytes_per_decision=187.0,
            compression_ratio=0.685, compression_source="Agent 4 probe",
        )
        assert with_ratio["compressed"]["is_measured_not_assumed"] is True
        assert with_ratio["compressed"]["ratio"] == 0.685
        assert with_ratio["compressed"]["source"] == "Agent 4 probe"


# ---------------------------------------------------------------------------
# The selection rule
# ---------------------------------------------------------------------------


class TestSelectionCannotSeeStrength:
    def test_no_input_key_is_strength_shaped(self):
        for key in soak.SELECTION_INPUT_KEYS:
            for forbidden in FORBIDDEN_INPUT_SUBSTRINGS:
                assert forbidden not in key, f"{key} contains {forbidden!r}"

    def test_selection_inputs_is_the_complete_view(self, measured_frontier):
        seen = soak.selection_inputs(measured_frontier[0])
        assert set(seen) == set(soak.SELECTION_INPUT_KEYS)

    def test_adding_a_win_rate_to_every_row_changes_nothing(self, measured_frontier):
        before = soak.select_architectures(measured_frontier)
        poisoned = [
            {**row, "win_rate": 0.99 if row["candidate_id"] == "C3" else 0.01,
             "elo": 3000 if row["candidate_id"] == "C3" else 0,
             "match_score": row["parameters"]}
            for row in measured_frontier
        ]
        after = soak.select_architectures(poisoned)
        assert after["primary_id"] == before["primary_id"]
        assert after["fallback_id"] == before["fallback_id"]

    def test_the_rule_records_that_strength_is_not_an_input(self, measured_frontier):
        selection = soak.select_architectures(measured_frontier)
        assert "strength" in selection["rule"]["strength_is_not_an_input"]

    def test_the_knee_does_not_move_between_the_two_throughput_sources(
        self, measured_frontier
    ):
        """Agent 4 published two recording rates; the choice must not depend on it.

        The headline rows are 30-second windows from a cold pool; the storage run
        is warmed and sustained. Agent 6 reads the sustained one, and this asserts
        the decision would have been identical either way.
        """
        sustained = {"C0": 11_705.76, "C1": 8954.21, "C2": 6231.30, "C3": 4735.33}
        from_storage_run = [
            {**row, "recording_positions_per_second": sustained[row["candidate_id"]]}
            for row in measured_frontier
        ]
        headline = soak.select_architectures(measured_frontier)
        warmed = soak.select_architectures(from_storage_run)
        assert warmed["primary_id"] == headline["primary_id"] == "C1"
        assert warmed["fallback_id"] == headline["fallback_id"] == "C0"


class TestKneeSelection:
    def test_the_measured_frontier_puts_the_knee_at_c1(self, measured_frontier):
        selection = soak.select_architectures(measured_frontier)
        assert selection["primary_id"] == "C1"
        assert selection["fallback_id"] == "C0"

    def test_the_choice_is_robust_across_the_plausible_floor_range(
        self, measured_frontier, monkeypatch
    ):
        """The declared floor of 0.5 is not load-bearing.

        On the measured ladder the C1 -> C2 step scores 0.33x the best step, so
        any floor above ~0.35 stops at C1. The declared 0.5 sits in the middle of
        that range rather than at its edge, which is the point: the knee is a
        property of the measurements, not of the constant.
        """
        for floor in (0.4, 0.5, 0.6, 0.75, 0.9, 1.0):
            monkeypatch.setattr(soak, "KNEE_EFFICIENCY_FLOOR", floor)
            selection = soak.select_architectures(measured_frontier)
            assert selection["primary_id"] == "C1", floor

    def test_the_floor_at_which_the_knee_would_move_is_far_below_the_declared_one(
        self, measured_frontier, monkeypatch
    ):
        declared = soak.KNEE_EFFICIENCY_FLOOR
        monkeypatch.setattr(soak, "KNEE_EFFICIENCY_FLOOR", 0.30)
        assert soak.select_architectures(measured_frontier)["primary_id"] == "C2"
        assert declared > 0.35

    def test_it_does_not_simply_pick_the_largest(self, measured_frontier):
        selection = soak.select_architectures(measured_frontier)
        largest = max(measured_frontier, key=lambda row: row["parameters"])
        assert selection["primary_id"] != largest["candidate_id"]

    def test_it_does_not_simply_pick_the_fastest(self, measured_frontier):
        selection = soak.select_architectures(measured_frontier)
        fastest = max(
            measured_frontier, key=lambda row: row["recording_positions_per_second"]
        )
        assert selection["primary_id"] != fastest["candidate_id"]

    def test_a_uniform_ladder_walks_to_the_top(self):
        # Every rung equally efficient: there is no knee below the ceiling, and
        # the rule should not invent one.
        rows = [
            frontier_row(f"U{index}", parameters=100_000 * 2**index,
                         recording=10_000.0 * 0.8**index)
            for index in range(4)
        ]
        selection = soak.select_architectures(rows)
        assert selection["primary_id"] == "U3"
        assert selection["fallback_id"] == "U2"

    def test_a_cliff_after_the_first_rung_stops_there(self):
        rows = [
            frontier_row("S0", parameters=100_000, recording=10_000.0),
            frontier_row("S1", parameters=700_000, recording=9_000.0),
            frontier_row("S2", parameters=730_000, recording=3_000.0),
        ]
        selection = soak.select_architectures(rows)
        assert selection["primary_id"] == "S1"
        assert selection["fallback_id"] == "S0"

    def test_a_numerically_unstable_candidate_is_not_selectable(self, measured_frontier):
        rows = [
            {**row, "numerically_stable_float16": row["candidate_id"] != "C1"}
            for row in measured_frontier
        ]
        selection = soak.select_architectures(rows)
        assert "C1" not in selection["ordered_candidate_ids"]
        assert "C1" in selection["excluded_candidate_ids"]

    def test_a_candidate_below_the_throughput_floor_is_excluded(self, measured_frontier):
        selection = soak.select_architectures(
            measured_frontier, minimum_recording_positions=7000.0
        )
        assert selection["ordered_candidate_ids"] == ["C0", "C1"]

    def test_fewer_than_two_viable_candidates_is_an_error(self, measured_frontier):
        with pytest.raises(soak.SoakError, match="at least two"):
            soak.select_architectures(
                measured_frontier, minimum_recording_positions=20_000.0
            )

    def test_primary_and_fallback_are_never_the_same(self, measured_frontier):
        selection = soak.select_architectures(measured_frontier)
        assert selection["primary_id"] != selection["fallback_id"]

    def test_the_fallback_is_smaller_and_faster_than_the_primary(self, measured_frontier):
        selection = soak.select_architectures(measured_frontier)
        rows = {row["candidate_id"]: row for row in measured_frontier}
        primary = rows[selection["primary_id"]]
        fallback = rows[selection["fallback_id"]]
        assert fallback["parameters"] < primary["parameters"]
        assert (
            fallback["recording_positions_per_second"]
            > primary["recording_positions_per_second"]
        )


class TestNeighborTradeoffs:
    def test_every_neighbouring_pair_is_quantified(self, measured_frontier):
        steps = soak.neighbor_tradeoffs(measured_frontier)
        assert [(step["from"], step["to"]) for step in steps] == [
            ("C0", "C1"), ("C1", "C2"), ("C2", "C3")
        ]

    def test_it_reports_the_six_axes_the_instruction_names(self, measured_frontier):
        step = soak.neighbor_tradeoffs(measured_frontier)[0]
        for key in (
            "parameters_change",
            "standalone_float16_inference_change",
            "training_step_change",
            "recording_change",
            "memory_change",
            "storage_rate_change",
        ):
            assert key in step

    def test_percentages_are_relative_changes(self, measured_frontier):
        step = soak.neighbor_tradeoffs(measured_frontier)[0]
        assert step["parameters_change"] == pytest.approx(863_959 / 123_223 - 1)
        assert step["parameter_ratio"] == pytest.approx(863_959 / 123_223)

    def test_the_first_step_is_the_most_efficient_on_the_measured_ladder(
        self, measured_frontier
    ):
        scores = [
            step["capacity_per_recording_throughput_given_up"]
            for step in soak.neighbor_tradeoffs(measured_frontier)
        ]
        assert scores[0] > scores[1] > scores[2]
        assert scores[0] > 2 * scores[1]


class TestArchitectureRecord:
    @pytest.mark.parametrize("candidate_id", ["C0", "C1", "C2", "C3"])
    def test_it_states_the_exact_frozen_configuration(self, candidate_id):
        record = soak.architecture_record(candidate_id)
        configuration = candidate_config(candidate_id)
        assert record["width"] == configuration.width
        assert record["blocks"] == configuration.blocks
        assert record["heads"] == configuration.heads
        assert record["feed_forward_width"] == configuration.feed_forward_width
        assert record["configuration_digest"] == configuration.digest()
        assert record["model_contract_version"] == MODEL_CONTRACT_VERSION
        assert record["architecture_family"] == ARCHITECTURE_FAMILY
        assert record["initialization_seed"] == FAMILY_INITIALIZATION_SEED

    def test_it_names_every_head_the_instruction_asks_for(self):
        record = soak.architecture_record("C1")
        assert record["policy_head"]
        assert record["value_head"]
        assert record["belief_head"]
        assert record["position_encoding"] == "learned_row_column_v1"

    def test_the_record_is_a_frozen_configuration_not_an_idea(self):
        record = soak.architecture_record("C0")
        rebuilt = candidate_config("C0").from_dict(record["configuration"])
        assert rebuilt == candidate_config("C0")


# ---------------------------------------------------------------------------
# One real, short run through the actual pipeline
# ---------------------------------------------------------------------------


def test_a_short_soak_runs_the_real_pipeline_and_passes_its_correctness_gates():
    """Not an hour, but the same code path, the same model and the same engine."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("the soak path requires Metal")
    config = soak.soak_configuration(
        "C0", workers=2, environments=64, inference_batch_size=64
    )
    result = soak.run_soak(
        "C0", config, seconds=6.0, sample_seconds=2.0, warmup_steps=2,
        probe_rows=16, device=torch.device("mps"),
    )
    assert result["status"] == "ok", result["error"]
    assert result["failures"]["illegal_actions"] == 0
    assert result["failures"]["action_frame_errors"] == 0
    assert result["failures"]["worker_errors"] == 0
    assert result["failures"]["model_errors"] == 0
    assert result["failures"]["nonfinite_outputs"] == 0
    assert result["correctness"]["reconstruction_mismatches"] == 0
    assert result["correctness"]["finiteness_probe_rows"] > 0
    assert result["sample_count"] >= 2
    assert result["whole_run"]["positions"] > 0
    # The probe really did look at all three heads on real published rows.
    assert result["correctness"]["finiteness_probe_logits"] > 0


# ---------------------------------------------------------------------------
# The artifacts, once they exist
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (DATA_DIRECTORY / "agent_06_architecture_decision.json").exists(),
    reason="Agent 6 has not produced its artifacts yet",
)
class TestPublishedArtifacts:
    @staticmethod
    def decision() -> dict:
        return json.loads(
            (DATA_DIRECTORY / "agent_06_architecture_decision.json").read_text()
        )

    def test_the_decision_names_every_required_field(self):
        decision = self.decision()
        for field in (
            "agent", "status", "all_prerequisites", "finalists",
            "primary_architecture", "fallback_architecture", "selection_method",
            "neighbor_tradeoffs", "soak_result", "weekly_projection",
            "storage_analysis", "parallel_evaluation_ready", "backend_statement",
            "full_suite", "completion_gates", "phase_6_recommendation",
        ):
            assert field in decision, field

    def test_the_comparison_keeps_all_four_measured_candidates(self):
        decision = self.decision()
        ids = {row["candidate_id"] for row in decision["finalists"]["rows"]}
        assert ids == {"C0", "C1", "C2", "C3"}

    def test_the_primary_and_fallback_are_exact_configurations(self):
        decision = self.decision()
        for role in ("primary_architecture", "fallback_architecture"):
            record = decision[role]
            rebuilt = candidate_config(record["candidate_id"])
            assert record["configuration_digest"] == rebuilt.digest()
            assert record["width"] == rebuilt.width
            assert record["blocks"] == rebuilt.blocks

    def test_the_fallback_is_the_smaller_of_the_two(self):
        decision = self.decision()
        assert (
            decision["fallback_architecture"]["parameters"]
            < decision["primary_architecture"]["parameters"]
        )

    def test_the_soak_ran_for_about_an_hour(self):
        decision = self.decision()
        assert decision["soak_result"]["seconds"] >= 3500.0

    def test_every_hard_soak_gate_is_true(self):
        gates = self.decision()["soak_result"]["completion_gates"]
        for name in (
            "illegal_actions_zero", "action_frame_mismatches_zero",
            "reconstruction_mismatches_zero", "worker_failures_zero",
            "model_mps_failures_zero", "nonfinite_production_outputs_zero",
            "swap_zero", "no_unexplained_memory_growth",
        ):
            assert gates[name] is True, name

    def test_the_projection_uses_the_soaks_own_sustained_throughput(self):
        decision = self.decision()
        measured = decision["weekly_projection"]["measured"]
        steady = decision["soak_result"]["steady_state"]
        assert measured["recording_inclusive_positions_per_second"] == pytest.approx(
            steady["positions_per_second"]
        )
        assert decision["weekly_projection"]["extrapolated"][
            "positions"
        ] == pytest.approx(steady["positions_per_second"] * 604800)

    def test_the_headline_numbers_are_in_the_timeseries_too(self):
        import csv

        path = DATA_DIRECTORY / "agent_06_soak_timeseries.csv"
        rows = list(csv.DictReader(path.open()))
        decision = self.decision()
        assert len(rows) == decision["soak_result"]["correctness"].get(
            "sample_count", len(rows)
        )
        assert all(int(row["illegal_actions"]) == 0 for row in rows)
        assert all(int(row["swap_used_bytes"]) == 0 for row in rows)
        assert all(int(row["reconstruction_mismatches"]) == 0 for row in rows)

    def test_no_strength_field_appears_in_the_selection_method(self):
        selection = self.decision()["selection_method"]
        serialized = json.dumps(selection).lower()
        # The rule documents that strength is excluded, so the words appear in
        # prose; what must not appear is a numeric field carrying one.
        for row in selection["step_scores"]:
            for key in row:
                assert not any(
                    forbidden in key for forbidden in ("win_rate", "elo", "score_")
                ), key
