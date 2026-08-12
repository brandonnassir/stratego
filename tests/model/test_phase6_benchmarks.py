"""Tests for the Phase 6 Agent 3 benchmark machinery.

These tests are about the *instrument*, not about the numbers it produces. A
benchmark that silently ran on CPU, silently ran in float32 while labelled
float16, quietly dropped the rows where a candidate failed, or let a
playing-strength figure reach the classification would still produce a complete
and plausible-looking report -- which is exactly why each of those is asserted
here rather than assumed.

Every measurement claim in the Agent 3 report rests on one of these properties.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from stratego.model import benchmark_helpers as helpers
from stratego.model.architecture_configs import (
    CANDIDATE_IDS,
    candidate_config,
    config_digests,
)
from stratego.model.contract import ModelOutputs
from stratego.model.production_model import build_candidate_model

metal = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal / MPS is not available on this host"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus() -> helpers.BenchmarkCorpus:
    """A small real-position corpus.

    `environments=16` rather than the harness default: with 128 environments a
    128-position corpus would be filled by a single sampling round, every row
    would sit at the same ply, and the acting colour would therefore be constant
    -- the exact degenerate corpus `build_benchmark_corpus` refuses to return.
    """
    return helpers.build_benchmark_corpus(positions=128, environments=16)


@pytest.fixture(scope="module")
def synthetic_summaries() -> list[dict]:
    """Fixed candidate summaries, so classification can be tested without measuring.

    Deliberately handwritten: a classification test that fed real measurements
    in would be testing this host's hardware rather than the rule.
    """
    return [
        {
            "candidate_id": "C0",
            "parameters": 123_223,
            "best_float32_positions_per_second": 40_000.0,
            "best_float16_positions_per_second": 45_000.0,
            "representative_training_examples_per_second": 8_000.0,
            "max_stable_inference_batch": 2048,
            "max_stable_training_batch": 256,
            "peak_metal_fraction": 0.05,
            "numerically_stable_float32": True,
            "numerically_stable_float16": True,
        },
        {
            "candidate_id": "C1",
            "parameters": 863_959,
            "best_float32_positions_per_second": 20_000.0,
            "best_float16_positions_per_second": 22_000.0,
            "representative_training_examples_per_second": 4_000.0,
            "max_stable_inference_batch": 2048,
            "max_stable_training_batch": 256,
            "peak_metal_fraction": 0.10,
            "numerically_stable_float32": True,
            "numerically_stable_float16": True,
        },
        {
            # Larger than C1 and faster on every axis: C1 is dominated by it.
            "candidate_id": "C2",
            "parameters": 1_922_519,
            "best_float32_positions_per_second": 25_000.0,
            "best_float16_positions_per_second": 26_000.0,
            "representative_training_examples_per_second": 5_000.0,
            "max_stable_inference_batch": 2048,
            "max_stable_training_batch": 256,
            "peak_metal_fraction": 0.15,
            "numerically_stable_float32": True,
            "numerically_stable_float16": True,
        },
        {
            # Below the practical throughput floor.
            "candidate_id": "C6",
            "parameters": 14_702_807,
            "best_float32_positions_per_second": 900.0,
            "best_float16_positions_per_second": 1_100.0,
            "representative_training_examples_per_second": 300.0,
            "max_stable_inference_batch": 2048,
            "max_stable_training_batch": 256,
            "peak_metal_fraction": 0.40,
            "numerically_stable_float32": True,
            "numerically_stable_float16": True,
        },
    ]


def _harness_module():
    """Import `scripts/run_phase6_agent03.py` by path.

    The scripts directory is not a package, and the CSV renderer is the piece of
    the harness whose behaviour the report depends on ("unavailable, not zero"),
    so it is worth reaching for rather than reimplementing in the test.
    """
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_phase6_agent03.py"
    spec = importlib.util.spec_from_file_location("run_phase6_agent03", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The benchmark measures the family Agent 2 accepted
# ---------------------------------------------------------------------------


class TestConfigurationReproduction:
    def test_every_candidate_digest_reproduces(self):
        report = helpers.reproduce_candidate_configs()
        assert report["all_reproduced"] is True
        assert set(report["candidates"]) == set(CANDIDATE_IDS)
        for candidate_id, entry in report["candidates"].items():
            assert entry["digest_matches"] is True
            assert entry["config_digest"] == config_digests()[candidate_id]

    def test_parameter_counts_match_agent_02_record(self):
        """The recorded artifact, not a recomputation of it.

        This is the check that would fail if an architecture were edited to
        improve its benchmark results, which the Agent 3 instructions forbid.
        """
        path = (
            Path(__file__).resolve().parents[2]
            / "reports"
            / "phase_6_data"
            / "agent_02_architecture_family.json"
        )
        recorded = json.loads(path.read_text())["parameter_counts"]
        rebuilt = helpers.reproduce_candidate_configs()["candidates"]
        for candidate_id in CANDIDATE_IDS:
            assert rebuilt[candidate_id]["trainable_parameters"] == recorded[candidate_id]

    def test_a_changed_configuration_is_visible_as_a_changed_digest(self):
        """The negative control: the digest has to actually be sensitive."""
        c2 = candidate_config("C2")
        altered = type(c2).from_dict({**c2.to_dict(), "blocks": c2.blocks + 1})
        assert altered.digest() != c2.digest()


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


class TestCorpus:
    def test_corpus_is_deterministic(self):
        first = helpers.build_benchmark_corpus(positions=64, environments=16)
        second = helpers.build_benchmark_corpus(positions=64, environments=16)
        assert first.digest == second.digest
        assert np.array_equal(first.observations, second.observations)
        assert np.array_equal(first.normalized_masks, second.normalized_masks)
        assert np.array_equal(first.policy_targets, second.policy_targets)
        assert np.array_equal(first.value_targets, second.value_targets)
        assert first.absolute_legal_lists == second.absolute_legal_lists

    def test_a_different_seed_gives_a_different_corpus(self):
        other = helpers.build_benchmark_corpus(positions=64, environments=16, seed=7)
        base = helpers.build_benchmark_corpus(positions=64, environments=16)
        assert other.digest != base.digest

    def test_digest_covers_the_targets_not_only_the_positions(self):
        base = helpers.build_benchmark_corpus(positions=64, environments=16)
        retargeted = helpers.build_benchmark_corpus(
            positions=64, environments=16, target_seed=helpers.TARGET_SEED + 1
        )
        assert np.array_equal(retargeted.observations, base.observations)
        assert retargeted.digest != base.digest

    def test_both_acting_colours_are_present(self, corpus):
        counts = corpus.stats()["acting_player_counts"]
        assert counts["red"] > 0
        assert counts["blue"] > 0

    def test_a_single_colour_corpus_is_refused(self):
        """An even stride samples only even plies, so red always acts.

        That corpus would look healthy and would never exercise the blue branch
        of the perspective transform, so construction must refuse it rather than
        return it.
        """
        with pytest.raises(helpers.BenchmarkError, match="only one acting colour"):
            helpers.build_benchmark_corpus(positions=32, environments=32, stride=16)

    def test_policy_targets_are_legal_in_the_normalized_frame(self, corpus):
        report = helpers.verify_policy_targets_legal(corpus)
        assert report["all_targets_legal"] is True
        assert report["targets_illegal_in_normalized_list"] == 0
        assert report["targets_illegal_in_normalized_mask"] == 0
        assert report["policy_target_frame"] == "perspective_normalized_squares"

    def test_the_two_normalized_legality_products_agree(self, corpus):
        report = helpers.verify_legality_frames_agree(corpus)
        assert report["normalized_list_vs_mask_mismatches"] == 0
        assert report["normalized_to_absolute_set_mismatches"] == 0

    def test_normalized_legality_is_not_the_absolute_mask_for_blue(self, corpus):
        """The conversion must actually do something.

        Without this, a corpus that never converted anything would satisfy every
        other assertion above, because the two frames coincide for red.
        """
        blue = [
            index
            for index in range(corpus.size)
            if int(corpus.acting_players[index]) == 1
        ]
        assert blue, "the corpus fixture must contain at least one blue-acting position"
        moved = sum(
            1
            for index in blue
            if set(corpus.normalized_legal_lists[index])
            != set(corpus.absolute_legal_lists[index])
        )
        assert moved == len(blue)

    def test_red_normalized_legality_is_the_absolute_legality(self, corpus):
        red = [index for index in range(corpus.size) if int(corpus.acting_players[index]) == 0]
        assert red
        for index in red:
            assert set(corpus.normalized_legal_lists[index]) == set(
                corpus.absolute_legal_lists[index]
            )

    def test_belief_targets_are_masked_and_never_an_input(self, corpus):
        """Belief labels are supervision, and only on hidden opponent squares."""
        assert corpus.belief_labels.shape == (corpus.size, 100)
        assert corpus.belief_masks.shape == (corpus.size, 100)
        assert np.array_equal(corpus.belief_masks, corpus.belief_labels != -100)
        supervised = corpus.belief_labels[corpus.belief_masks]
        assert supervised.min() >= 0
        assert supervised.max() < 12
        # The observation tensor is the only thing a model receives, and it has
        # the contract's channel count -- no target has been concatenated onto it.
        assert corpus.observations.shape[1:] == (127, 10, 10)

    def test_host_batches_are_deterministic_slices(self, corpus):
        first = helpers.make_host_batch(corpus, 32)
        second = helpers.make_host_batch(corpus, 32)
        assert np.array_equal(first.observations, second.observations)
        assert np.array_equal(first.indices, second.indices)

    def test_host_batches_wrap_around_rather_than_running_out(self, corpus):
        batch = helpers.make_host_batch(corpus, corpus.size + 8)
        assert batch.batch_size == corpus.size + 8
        assert int(batch.indices[corpus.size]) == 0


# ---------------------------------------------------------------------------
# Device and precision labelling
# ---------------------------------------------------------------------------


def _cpu_outputs(batch: int = 2) -> ModelOutputs:
    model = build_candidate_model("C0", device="cpu", dtype=torch.float32)
    tokens = torch.zeros(batch, 100, 127, dtype=torch.float32)
    with torch.no_grad():
        return model(tokens)


class TestExecutionLabels:
    def test_cpu_results_cannot_be_labelled_mps(self):
        outputs = _cpu_outputs()
        with pytest.raises(helpers.BenchmarkIntegrityError, match="outputs are on cpu"):
            helpers.verify_execution_labels(
                outputs,
                requested_device=torch.device("mps"),
                requested_precision="float32",
            )

    def test_float32_results_cannot_be_labelled_float16(self):
        outputs = _cpu_outputs()
        with pytest.raises(helpers.BenchmarkIntegrityError, match="mislabel its dtype"):
            helpers.verify_execution_labels(
                outputs,
                requested_device=torch.device("cpu"),
                requested_precision="float16",
            )

    def test_a_truthful_label_is_accepted_and_read_off_the_tensor(self):
        outputs = _cpu_outputs()
        labels = helpers.verify_execution_labels(
            outputs, requested_device=torch.device("cpu"), requested_precision="float32"
        )
        assert labels["observed_device_type"] == "cpu"
        assert labels["observed_precision"] == "float32"

    @metal
    def test_an_mps_row_records_mps_and_its_real_dtype(self, corpus):
        device = helpers.require_mps()
        for precision in helpers.PRECISIONS:
            model = build_candidate_model(
                "C0", device=device, dtype=helpers.resolve_dtype(precision)
            )
            row = helpers.run_inference_point(
                model=model,
                candidate_id="C0",
                config_digest=config_digests()["C0"],
                parameters=model.parameter_count(),
                corpus=corpus,
                batch=8,
                precision=precision,
                boundary=helpers.BOUNDARY_A,
                device=device,
            )
            assert row["status"] == "ok"
            assert row["observed_device"].startswith("mps")
            assert row["observed_precision"] == precision
            assert row["requested_precision"] == precision

    @metal
    def test_a_cpu_model_under_an_mps_request_never_yields_an_mps_row(self, corpus):
        """Two independent barriers stop CPU work being reported as Metal work.

        The first is PyTorch itself: the tokens are placed on the requested
        device, so a CPU model cannot consume them and the attempt becomes an
        error row whose observed device is never `mps`.
        """
        device = helpers.require_mps()
        cpu_model = build_candidate_model("C0", device="cpu", dtype=torch.float32)
        row = helpers.run_inference_point(
            model=cpu_model,
            candidate_id="C0",
            config_digest=config_digests()["C0"],
            parameters=cpu_model.parameter_count(),
            corpus=corpus,
            batch=8,
            precision="float32",
            boundary=helpers.BOUNDARY_A,
            device=device,
        )
        assert row["status"] != "ok"
        assert row["observed_device"] is None
        assert row["positions_per_second"] is None

    @metal
    def test_cpu_outputs_under_an_mps_request_raise_rather_than_becoming_a_row(
        self, corpus, monkeypatch
    ):
        """The second barrier, with PyTorch's own check removed.

        If some future code path did manage to produce CPU results while `mps`
        was requested, the integrity check must stop the run outright. A row
        that lies about where it ran is worse than a missing row, because every
        comparison downstream would inherit it silently -- so this is the one
        failure the harness raises on instead of recording.
        """
        device = helpers.require_mps()
        cpu_model = build_candidate_model("C0", device="cpu", dtype=torch.float32)
        tokens = torch.zeros(8, 100, 127, dtype=torch.float32)

        def cpu_only(**_kwargs):
            def run():
                with torch.no_grad():
                    return cpu_model(tokens)

            return run, run

        monkeypatch.setattr(helpers, "make_boundary_operation", cpu_only)
        with pytest.raises(helpers.BenchmarkIntegrityError, match="outputs are on cpu"):
            helpers.run_inference_point(
                model=cpu_model,
                candidate_id="C0",
                config_digest=config_digests()["C0"],
                parameters=cpu_model.parameter_count(),
                corpus=corpus,
                batch=8,
                precision="float32",
                boundary=helpers.BOUNDARY_A,
                device=device,
            )

    @metal
    def test_a_float32_path_cannot_be_recorded_as_float16(self, corpus, monkeypatch):
        """The same barrier for precision, which is the other silent failure."""
        device = helpers.require_mps()
        model = build_candidate_model("C0", device=device, dtype=torch.float32)
        tokens = torch.zeros(8, 100, 127, dtype=torch.float32, device=device)

        def float32_only(**_kwargs):
            def run():
                with torch.no_grad():
                    return model(tokens)

            return run, run

        monkeypatch.setattr(helpers, "make_boundary_operation", float32_only)
        with pytest.raises(helpers.BenchmarkIntegrityError, match="mislabel its dtype"):
            helpers.run_inference_point(
                model=model,
                candidate_id="C0",
                config_digest=config_digests()["C0"],
                parameters=model.parameter_count(),
                corpus=corpus,
                batch=8,
                precision="float16",
                boundary=helpers.BOUNDARY_A,
                device=device,
            )

    def test_resolve_dtype_rejects_an_unknown_precision(self):
        with pytest.raises(helpers.BenchmarkError, match="unknown precision"):
            helpers.resolve_dtype("bfloat16")


# ---------------------------------------------------------------------------
# Timing and synchronisation
# ---------------------------------------------------------------------------


class TestTiming:
    def test_summary_reports_median_p95_and_mean(self):
        summary = helpers.summarise_samples([0.001, 0.002, 0.003, 0.004])
        assert summary["measurement_iterations"] == 4
        assert summary["median_latency_ms"] == pytest.approx(2.5)
        assert summary["mean_latency_ms"] == pytest.approx(2.5)
        assert summary["p95_latency_ms"] >= summary["median_latency_ms"]

    def test_empty_samples_do_not_become_zero_latency(self):
        summary = helpers.summarise_samples([])
        assert summary["median_latency_ms"] is None

    @metal
    def test_timed_region_leaves_no_queued_work(self):
        """MPS dispatch is asynchronous; the timed region must close the queue.

        The instrument: run a genuinely heavy operation through `timed_samples`,
        then synchronise again immediately afterwards and time *that*. If the
        helper synchronised inside its timed region there is nothing left to
        wait for and the trailing synchronise is nearly free. If it did not, the
        trailing synchronise absorbs the real work and the samples were measuring
        queue submission.
        """
        import time

        device = helpers.require_mps()
        left = torch.randn(2048, 2048, device=device)

        def heavy() -> None:
            result = left
            for _ in range(6):
                result = result @ left
            return result

        samples = helpers.timed_samples(
            heavy, device=device, warmup=2, minimum=5, maximum=5, target_seconds=0.0
        )
        start = time.perf_counter()
        torch.mps.synchronize()
        trailing = time.perf_counter() - start

        assert len(samples) == 5
        assert min(samples) > 0
        # The trailing drain must be a small fraction of one sample's work.
        assert trailing < 0.25 * min(samples)

    @metal
    def test_samples_account_for_the_wall_clock_time_spent(self):
        """Sum of samples ~= wall time of the measurement phase.

        A missing trailing synchronise shows up here as samples that add up to
        far less than the time the call actually took.
        """
        import time

        device = helpers.require_mps()
        left = torch.randn(1024, 1024, device=device)

        def heavy() -> None:
            result = left
            for _ in range(4):
                result = result @ left
            return result

        start = time.perf_counter()
        samples = helpers.timed_samples(
            heavy, device=device, warmup=0, minimum=6, maximum=6, target_seconds=0.0
        )
        torch.mps.synchronize()
        wall = time.perf_counter() - start
        assert sum(samples) >= 0.7 * wall


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------


class TestBoundaries:
    def test_every_boundary_declares_what_it_contains(self):
        assert set(helpers.BOUNDARIES) == set(helpers.BOUNDARY_CONTENTS)
        for boundary in helpers.BOUNDARIES:
            assert len(helpers.BOUNDARY_CONTENTS[boundary]) > 40

    def test_an_unknown_boundary_is_refused(self, corpus):
        model = build_candidate_model("C0")
        with pytest.raises(helpers.BenchmarkError, match="unknown timing boundary"):
            helpers.make_boundary_operation(
                boundary="D_everything",
                model=model,
                host=helpers.make_host_batch(corpus, 4),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

    @metal
    def test_boundary_cost_is_monotone(self, corpus):
        """C includes B includes A, so their medians should order that way.

        Asserted loosely (with a tolerance) because these are real timings, but
        an inversion would mean a boundary is not doing the work it claims.
        """
        device = helpers.require_mps()
        model = build_candidate_model("C0", device=device, dtype=torch.float32)
        host = helpers.make_host_batch(corpus, 64)
        medians = {}
        for boundary in helpers.BOUNDARIES:
            row = helpers.run_inference_point(
                model=model,
                candidate_id="C0",
                config_digest=config_digests()["C0"],
                parameters=model.parameter_count(),
                corpus=corpus,
                batch=64,
                precision="float32",
                boundary=boundary,
                device=device,
                host=host,
            )
            assert row["status"] == "ok"
            medians[boundary] = row["median_latency_ms"]
        assert medians[helpers.BOUNDARY_B] >= 0.8 * medians[helpers.BOUNDARY_A]
        assert medians[helpers.BOUNDARY_C] >= 0.8 * medians[helpers.BOUNDARY_B]

    @metal
    def test_selected_actions_are_legal_in_both_frames(self, corpus):
        device = helpers.require_mps()
        model = build_candidate_model("C0", device=device, dtype=torch.float32)
        report = helpers.selection_validity(
            model=model,
            host=helpers.make_host_batch(corpus, 64),
            device=device,
            dtype=torch.float32,
            corpus=corpus,
        )
        assert report["illegal_normalized_selections"] == 0
        assert report["illegal_absolute_actions"] == 0
        assert report["selections"] == 64


# ---------------------------------------------------------------------------
# Failure rows are retained
# ---------------------------------------------------------------------------


class TestFailureRowsAreKept:
    def test_out_of_memory_becomes_a_row_not_an_exception(self, corpus, monkeypatch):
        def explode(**_kwargs):
            raise RuntimeError("MPS backend out of memory (MPS allocated: 40.00 GB)")

        monkeypatch.setattr(helpers, "make_boundary_operation", explode)
        row = helpers.run_inference_point(
            model=build_candidate_model("C0"),
            candidate_id="C0",
            config_digest=config_digests()["C0"],
            parameters=123_223,
            corpus=corpus,
            batch=4096,
            precision="float32",
            boundary=helpers.BOUNDARY_A,
            device=torch.device("cpu"),
        )
        assert row["status"] == "oom"
        assert row["oom"] is True
        assert "out of memory" in row["error"]
        assert row["batch"] == 4096
        assert row["candidate_id"] == "C0"

    def test_a_non_memory_failure_is_recorded_as_an_error_row(self, corpus, monkeypatch):
        def explode(**_kwargs):
            raise RuntimeError("some unsupported Metal operation")

        monkeypatch.setattr(helpers, "make_boundary_operation", explode)
        row = helpers.run_inference_point(
            model=build_candidate_model("C0"),
            candidate_id="C0",
            config_digest=config_digests()["C0"],
            parameters=123_223,
            corpus=corpus,
            batch=64,
            precision="float32",
            boundary=helpers.BOUNDARY_A,
            device=torch.device("cpu"),
        )
        assert row["status"] == "error"
        assert row["oom"] is False
        assert "unsupported Metal operation" in row["error"]

    def test_training_out_of_memory_becomes_a_row(self, corpus, monkeypatch):
        def explode(*_args, **_kwargs):
            raise RuntimeError("MPS backend out of memory")

        monkeypatch.setattr(helpers, "build_candidate_model", explode)
        row = helpers.run_training_point(
            candidate_id="C6",
            config=candidate_config("C6"),
            config_digest=config_digests()["C6"],
            parameters=14_702_807,
            corpus=corpus,
            batch=256,
            precision="float16",
            device=torch.device("cpu"),
        )
        assert row["status"] == "oom"
        assert row["oom"] is True
        assert row["optimizer_step"] is False
        assert row["parameter_update"] is False

    def test_failed_rows_survive_the_csv_writer(self, tmp_path, corpus, monkeypatch):
        module = _harness_module()

        def explode(**_kwargs):
            raise RuntimeError("MPS backend out of memory")

        monkeypatch.setattr(helpers, "make_boundary_operation", explode)
        row = helpers.run_inference_point(
            model=build_candidate_model("C0"),
            candidate_id="C0",
            config_digest=config_digests()["C0"],
            parameters=123_223,
            corpus=corpus,
            batch=4096,
            precision="float32",
            boundary=helpers.BOUNDARY_A,
            device=torch.device("cpu"),
        )
        path = tmp_path / "inference.csv"
        written = module.write_csv(path, module.INFERENCE_COLUMNS, [row])
        assert written == 1
        text = path.read_text()
        assert "oom" in text
        assert "4096" in text


class TestMemoryReporting:
    def test_unavailable_memory_renders_as_unavailable_not_zero(self):
        module = _harness_module()
        assert module.render(None, "metal_driver_bytes") == "unavailable"
        assert module.render(None, "peak_memory_if_available") == "unavailable"
        assert module.render(None, "memory_fraction_of_recommended") == "unavailable"
        # A non-memory column stays empty rather than claiming a measurement.
        assert module.render(None, "median_latency_ms") == ""
        assert module.render(0, "metal_driver_bytes") == 0

    def test_snapshot_reports_none_rather_than_zero_for_missing_apis(self):
        snapshot = helpers.metal_memory_snapshot()
        for key in ("metal_allocated_bytes", "metal_driver_bytes"):
            assert snapshot[key] is None or isinstance(snapshot[key], int)
        assert isinstance(snapshot["process_rss_bytes"], int)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassification:
    def test_classification_is_deterministic(self, synthetic_summaries):
        first = helpers.classify_candidates(synthetic_summaries)
        second = helpers.classify_candidates(synthetic_summaries)
        assert first["verdicts"] == second["verdicts"]
        assert first["reasons"] == second["reasons"]

    def test_classification_does_not_depend_on_input_order(self, synthetic_summaries):
        shuffled = list(synthetic_summaries)
        random.Random(11).shuffle(shuffled)
        assert (
            helpers.classify_candidates(shuffled)["verdicts"]
            == helpers.classify_candidates(synthetic_summaries)["verdicts"]
        )
        assert (
            helpers.classify_candidates(list(reversed(synthetic_summaries)))["verdicts"]
            == helpers.classify_candidates(synthetic_summaries)["verdicts"]
        )

    def test_the_declared_rule_produces_the_declared_verdicts(self, synthetic_summaries):
        verdicts = helpers.classify_candidates(synthetic_summaries)["verdicts"]
        assert verdicts["C0"] == "ADVANCE"
        assert verdicts["C2"] == "ADVANCE"
        # C2 is larger than C1 and at least as fast on every axis.
        assert verdicts["C1"] == "DOMINATED"
        # C6 sits below the declared practical throughput floor.
        assert verdicts["C6"] == "IMPRACTICAL"

    def test_a_strength_field_cannot_change_a_verdict(self, synthetic_summaries):
        """The positive control for "no strength-based selection".

        Injecting a decisive-looking win rate that would reverse the ordering
        must change nothing at all.
        """
        polluted = []
        for index, summary in enumerate(synthetic_summaries):
            polluted.append(
                {
                    **summary,
                    "win_rate": 1.0 - index * 0.25,
                    "elo": 2000 - index * 300,
                    "gauntlet_score": 0.9 - index * 0.2,
                    "match_results": {"wins": 100 - index},
                }
            )
        assert (
            helpers.classify_candidates(polluted)["verdicts"]
            == helpers.classify_candidates(synthetic_summaries)["verdicts"]
        )

    def test_classification_inputs_are_restricted_to_the_declared_keys(
        self, synthetic_summaries
    ):
        projected = helpers.classification_inputs(
            {**synthetic_summaries[0], "win_rate": 0.99}
        )
        assert set(projected) == set(helpers.CLASSIFICATION_INPUT_KEYS)
        assert "win_rate" not in projected

    def test_a_strength_shaped_input_key_is_refused(self, monkeypatch):
        """The guard has to bite, or it is decoration.

        Adding a strength field to the allowlist in some later phase must fail
        loudly rather than quietly become an input.
        """
        monkeypatch.setattr(
            helpers,
            "CLASSIFICATION_INPUT_KEYS",
            helpers.CLASSIFICATION_INPUT_KEYS + ("win_rate",),
        )
        with pytest.raises(helpers.BenchmarkError, match="playing-strength"):
            helpers.classification_inputs({"candidate_id": "C0", "win_rate": 1.0})

    def test_a_failed_float32_numerical_check_makes_a_candidate_impractical(
        self, synthetic_summaries
    ):
        broken = [
            {**summary, "numerically_stable_float32": False}
            if summary["candidate_id"] == "C0"
            else summary
            for summary in synthetic_summaries
        ]
        verdicts = helpers.classify_candidates(broken)["verdicts"]
        assert verdicts["C0"] == "IMPRACTICAL"

    def test_float16_failure_alone_does_not_eliminate_a_candidate(
        self, synthetic_summaries
    ):
        """A sound float32 path is enough to advance.

        Pure float16 backward failing is information for Agent 4, not grounds
        for removing an otherwise viable architecture.
        """
        half_broken = [
            {**summary, "numerically_stable_float16": False}
            for summary in synthetic_summaries
        ]
        result = helpers.classify_candidates(half_broken)
        assert result["verdicts"]["C0"] == "ADVANCE"
        assert "float32 only" in result["reasons"]["C0"]

    def test_a_candidate_that_cannot_reach_the_minimum_batch_is_impractical(
        self, synthetic_summaries
    ):
        limited = [
            {**summary, "max_stable_inference_batch": 64}
            if summary["candidate_id"] == "C0"
            else summary
            for summary in synthetic_summaries
        ]
        assert helpers.classify_candidates(limited)["verdicts"]["C0"] == "IMPRACTICAL"

    def test_excessive_memory_makes_a_candidate_impractical(self, synthetic_summaries):
        hungry = [
            {**summary, "peak_metal_fraction": 0.95}
            if summary["candidate_id"] == "C0"
            else summary
            for summary in synthetic_summaries
        ]
        assert helpers.classify_candidates(hungry)["verdicts"]["C0"] == "IMPRACTICAL"

    def test_every_candidate_receives_exactly_one_verdict(self, synthetic_summaries):
        result = helpers.classify_candidates(synthetic_summaries)
        ids = set(summary["candidate_id"] for summary in synthetic_summaries)
        assert set(result["verdicts"]) == ids
        assert (
            len(result["advance_ids"])
            + len(result["dominated_ids"])
            + len(result["impractical_ids"])
            == len(ids)
        )
        assert not set(result["advance_ids"]) & set(result["dominated_ids"])

    def test_rules_are_serialisable_and_declare_the_thresholds(self):
        rules = helpers.classification_rules()
        assert json.loads(json.dumps(rules))
        assert rules["thresholds"]["min_viable_inference_batch"] == (
            helpers.MIN_VIABLE_INFERENCE_BATCH
        )
        assert rules["thresholds"]["min_viable_positions_per_second"] == (
            helpers.MIN_VIABLE_POSITIONS_PER_SECOND
        )
        assert "classification_input_keys" in rules


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


class TestSummaries:
    def test_only_stable_rows_contribute_to_the_frontier_figures(self):
        """A candidate must not look fast because its failures were dropped."""
        rows = [
            {
                "candidate_id": "C0",
                "boundary": helpers.BOUNDARY_A,
                "precision": "float32",
                "batch": 256,
                "status": "ok",
                "positions_per_second": 10_000.0,
                "memory_fraction_of_recommended": 0.1,
                "oom": False,
            },
            {
                # An OOM row carrying a nonsense throughput must be ignored.
                "candidate_id": "C0",
                "boundary": helpers.BOUNDARY_A,
                "precision": "float32",
                "batch": 2048,
                "status": "oom",
                "positions_per_second": 999_999.0,
                "memory_fraction_of_recommended": 0.9,
                "oom": True,
            },
        ]
        summary = helpers.summarise_candidate(
            candidate_id="C0",
            parameters=123_223,
            inference_rows=rows,
            training_rows=[],
            numerical=None,
        )
        assert summary["best_float32_positions_per_second"] == 10_000.0
        assert summary["max_stable_inference_batch"] == 256
        assert summary["inference_oom_rows"] == 1

    def test_boundary_a_is_what_defines_throughput(self):
        rows = [
            {
                "candidate_id": "C0",
                "boundary": helpers.BOUNDARY_C,
                "precision": "float32",
                "batch": 512,
                "status": "ok",
                "positions_per_second": 50_000.0,
                "memory_fraction_of_recommended": 0.1,
                "oom": False,
            }
        ]
        summary = helpers.summarise_candidate(
            candidate_id="C0",
            parameters=1,
            inference_rows=rows,
            training_rows=[],
            numerical=None,
        )
        assert summary["best_float32_positions_per_second"] is None


# ---------------------------------------------------------------------------
# Method disclosure
# ---------------------------------------------------------------------------


class TestMethodSummary:
    def test_method_summary_is_serialisable_and_complete(self):
        summary = helpers.benchmark_method_summary()
        assert json.loads(json.dumps(summary, default=str))
        for key in (
            "benchmark_version",
            "boundaries",
            "warmup_iterations",
            "synchronization",
            "tolerances",
            "memory_apis",
            "float16_training_policy",
            "target_generation",
        ):
            assert key in summary

    def test_tolerances_are_declared_for_both_comparisons_and_every_head(self):
        for key in ("mps_float32", "mps_float16"):
            limits = helpers.TOLERANCES[key]
            for head in (
                "policy_logits_max_abs",
                "value_probabilities_max_abs",
                "belief_logits_max_abs",
            ):
                assert limits[head] > 0
        # float16 is genuinely looser than float32; a single shared number would
        # mean one of the two comparisons was not thought about.
        assert (
            helpers.TOLERANCES["mps_float16"]["policy_logits_max_abs"]
            > helpers.TOLERANCES["mps_float32"]["policy_logits_max_abs"]
        )

    def test_target_generation_is_documented(self):
        targets = helpers.benchmark_method_summary()["target_generation"]
        assert set(targets) == {"policy", "value", "belief"}
        assert "normalized legal list" in targets["policy"]
        assert "never a model input" in targets["belief"]

    def test_float16_policy_states_no_autocast_and_no_loss_scaling(self):
        policy = helpers.benchmark_method_summary()["float16_training_policy"]
        assert "no autocast" in policy
        assert "no loss scaling" in policy


# ---------------------------------------------------------------------------
# The training step is a measurement, not training
# ---------------------------------------------------------------------------


class TestTrainingStep:
    @metal
    def test_one_step_produces_finite_losses_and_connected_gradients(self, corpus):
        device = helpers.require_mps()
        row = helpers.run_training_point(
            candidate_id="C0",
            config=candidate_config("C0"),
            config_digest=config_digests()["C0"],
            parameters=123_223,
            corpus=corpus,
            batch=32,
            precision="float32",
            device=device,
        )
        assert row["status"] == "ok"
        assert row["finite_loss"] is True
        assert row["finite_gradients"] is True
        for group in (
            "shared_encoder_gradient",
            "policy_head_gradient",
            "value_head_gradient",
            "belief_head_gradient",
        ):
            assert row[group] > 0, f"{group} received no gradient signal"

    @metal
    def test_no_optimizer_step_and_no_parameter_update(self, corpus):
        """Phase 6 authorises a backward pass only to measure compute.

        Proved by comparing the weights before and after, not by asserting the
        flag the row happens to carry.
        """
        device = helpers.require_mps()
        before = {
            name: parameter.detach().clone()
            for name, parameter in build_candidate_model(
                "C0", device="cpu", dtype=torch.float32
            ).named_parameters()
        }
        row = helpers.run_training_point(
            candidate_id="C0",
            config=candidate_config("C0"),
            config_digest=config_digests()["C0"],
            parameters=123_223,
            corpus=corpus,
            batch=32,
            precision="float32",
            device=device,
        )
        assert row["optimizer_step"] is False
        assert row["parameter_update"] is False
        # A freshly built candidate must be bit-identical to the one built
        # before the step: nothing the benchmark did can have persisted.
        after = build_candidate_model("C0", device="cpu", dtype=torch.float32)
        for name, parameter in after.named_parameters():
            assert torch.equal(parameter.detach(), before[name])

    @metal
    def test_float16_training_row_really_ran_in_float16(self, corpus):
        device = helpers.require_mps()
        row = helpers.run_training_point(
            candidate_id="C0",
            config=candidate_config("C0"),
            config_digest=config_digests()["C0"],
            parameters=123_223,
            corpus=corpus,
            batch=32,
            precision="float16",
            device=device,
        )
        assert row["precision"] == "float16"
        if row["status"] == "ok":
            assert row["observed_precision"] == "float16"
            assert row["observed_device"].startswith("mps")


# ---------------------------------------------------------------------------
# Numerical comparison
# ---------------------------------------------------------------------------


class TestNumericalComparison:
    @metal
    def test_cpu_and_mps_float32_agree_within_the_declared_tolerance(self, corpus):
        report = helpers.numerical_comparison(
            candidate_id="C0",
            config=candidate_config("C0"),
            corpus=corpus,
            device=helpers.require_mps(),
            positions=64,
        )
        entry = report["comparisons"]["mps_float32"]
        assert entry["status"] == "ok"
        assert entry["within_tolerance"] is True
        assert entry["finite_outputs"] is True
        assert entry["illegal_absolute_actions"] == 0

    @metal
    def test_the_crafted_margin_actually_dominates_on_the_reference(self, corpus):
        """Without this, crafted-margin agreement could pass vacuously."""
        report = helpers.numerical_comparison(
            candidate_id="C0",
            config=candidate_config("C0"),
            corpus=corpus,
            device=helpers.require_mps(),
            positions=64,
        )
        assert report["crafted_margin_effective_on_reference"] is True
        assert report["comparisons"]["mps_float32"]["crafted_margin_passes"] is True
        assert report["comparisons"]["mps_float16"]["crafted_margin_passes"] is True

    @metal
    def test_float16_is_reported_with_both_absolute_and_relative_error(self, corpus):
        report = helpers.numerical_comparison(
            candidate_id="C0",
            config=candidate_config("C0"),
            corpus=corpus,
            device=helpers.require_mps(),
            positions=64,
        )
        heads = report["comparisons"]["mps_float16"]["heads"]
        for head in ("policy_logits", "value_probabilities", "belief_logits"):
            entry = heads[head]
            assert entry["max_absolute_error"] >= 0
            assert entry["mean_absolute_error"] >= 0
            assert "meaningful_max_relative_error" in entry
            assert "max_relative_error_unfiltered" in entry
            assert entry["relative_error_floor"] == helpers.RELATIVE_ERROR_FLOOR

    def test_relative_error_floor_is_declared_and_positive(self):
        assert helpers.RELATIVE_ERROR_FLOOR > 0
