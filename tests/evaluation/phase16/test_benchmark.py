"""The canonical benchmark pack: construction, manifest, freeze discipline."""

import json

import pytest

from stratego.evaluation.phase16.benchmark import (
    BENCHMARK_CELLS,
    FAMILY_ANY,
    benchmark_board_plan,
    benchmark_plans,
    build_benchmark_manifest,
    load_benchmark_manifest,
    materialize_benchmark,
    quick_subset_ids,
    requested_family,
)
from stratego.evaluation.phase16.contract import (
    MATCH_FAMILY_KEYS,
    Phase16MeasurementError,
    QUICK_SUBSET_ORDINAL,
)


@pytest.fixture(scope="module")
def small_plans(setup_sources):
    """Ordinals 0-1 of the first four cells — enough to pin the mechanics."""
    plans = []
    for cell_index, (opponent, source, color) in enumerate(BENCHMARK_CELLS[:4]):
        for ordinal in (0, 1):
            plans.append(
                benchmark_board_plan(
                    opponent, source, color, ordinal, setup_sources, cell_index=cell_index
                )
            )
    return plans


class TestCells:
    def test_sixty_cells(self):
        assert len(BENCHMARK_CELLS) == 60
        assert len(set(BENCHMARK_CELLS)) == 60

    def test_requested_family_cycles_targeted_only(self):
        assert requested_family("neutral_v1", 5, 0) == FAMILY_ANY
        keys = {
            requested_family("targeted_family", cell, ordinal)
            for cell in range(60)
            for ordinal in range(2)
        }
        assert keys == set(MATCH_FAMILY_KEYS)


class TestBoards:
    def test_deterministic(self, setup_sources):
        first = benchmark_board_plan("p18", "neutral_v1", "red", 0, setup_sources)
        second = benchmark_board_plan("p18", "neutral_v1", "red", 0, setup_sources)
        assert first.red_setup == second.red_setup
        assert first.blue_setup == second.blue_setup
        assert first.match_seed == second.match_seed

    def test_gated(self, small_plans):
        for plan in small_plans:
            assert plan.orientation["paired_mirror"] is True

    def test_distinct_across_ordinals(self, small_plans):
        by_cell: dict = {}
        for plan in small_plans:
            by_cell.setdefault(plan.cell_index, []).append(plan)
        for plans in by_cell.values():
            assert plans[0].board_id != plans[1].board_id

    def test_unknown_cell_refused(self, setup_sources):
        with pytest.raises(Phase16MeasurementError):
            benchmark_board_plan("p18", "nope", "red", 0, setup_sources)


class TestManifest:
    def test_round_trip_and_verify(self, tmp_path, setup_sources, small_plans):
        manifest = build_benchmark_manifest(
            small_plans, generated_utc="t", sources=setup_sources
        )
        assert manifest["board_count"] == len(small_plans)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(manifest, sort_keys=True))
        loaded = load_benchmark_manifest(path)
        rebuilt = materialize_benchmark(loaded, sources=setup_sources, verify=True)
        assert [plan.board_id for plan in rebuilt] == [
            plan.board_id for plan in small_plans
        ]

    def test_tamper_refused(self, tmp_path, setup_sources, small_plans):
        manifest = build_benchmark_manifest(small_plans, generated_utc="t")
        manifest["boards"][0]["red_setup"][0] = 11
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(manifest, sort_keys=True))
        with pytest.raises(Phase16MeasurementError):
            load_benchmark_manifest(path)

    def test_setup_tamper_refused_on_materialize(self, tmp_path, setup_sources, small_plans):
        from stratego.evaluation.phase16.benchmark import manifest_digest

        manifest = build_benchmark_manifest(small_plans, generated_utc="t")
        manifest["boards"][0]["red_setup"] = list(
            reversed(manifest["boards"][0]["red_setup"])
        )
        manifest["manifest_digest"] = manifest_digest(manifest)  # digest re-forged
        with pytest.raises(Phase16MeasurementError):
            materialize_benchmark(manifest, sources=setup_sources, verify=True)

    def test_quick_subset(self, small_plans):
        manifest = build_benchmark_manifest(small_plans, generated_utc="t")
        subset = quick_subset_ids(manifest)
        assert len(subset) == len(small_plans) // 2
        for board in subset:
            assert f"|g={QUICK_SUBSET_ORDINAL:03d}" in board


class TestFrozenArtifact:
    """The delivered manifest, when present: shape and digest discipline."""

    def test_delivered_manifest(self, repository_root):
        try:
            manifest = load_benchmark_manifest(root=repository_root)
        except Phase16MeasurementError:
            pytest.skip("the frozen benchmark manifest has not been generated yet")
        assert manifest["board_count"] == 120
        assert manifest["balance"]["by_color"] == {"blue": 60, "red": 60}
        assert set(manifest["balance"]["by_opponent"].values()) == {12}
        assert set(manifest["balance"]["by_setup_source"].values()) == {40}
        assert len(quick_subset_ids(manifest)) == 60
