"""The scoring runner (hermetic paths) and the operator log."""

import importlib.util
import json
from pathlib import Path

import pytest

from stratego.evaluation.phase16.analysis import (
    analyse_baseline,
    paired_delta_by,
    predeclared_reading,
)
from stratego.evaluation.phase16.contract import (
    ARM_ADVERSARIAL_BOTH,
    ARM_ADVERSARIAL_OPPONENT,
    ARM_CONTROL,
    OPERATOR_LOG_SCHEMA,
    Phase16MeasurementError,
)
from stratego.evaluation.phase16.operator_log import (
    OperatorGameLogger,
    harvest_operator_setups,
    operator_series_summary,
    read_log,
)
from stratego.evaluation.phase16.runner import (
    Task16,
    load_results,
    normalize_seat_spec,
    resolve_subset,
    run_pack16,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class TestSeatSpecs:
    def test_pairing_ids_accepted(self):
        assert normalize_seat_spec("p24_b24") == "p24_b24"
        assert normalize_seat_spec("p18_direct") == "p18_direct"
        assert normalize_seat_spec("p24_remaining_count") == "p24_remaining_count"

    def test_oracle_refused_by_name(self):
        with pytest.raises(Phase16MeasurementError):
            normalize_seat_spec("p24_oracle")
        with pytest.raises(Phase16MeasurementError):
            normalize_seat_spec("p18_oracle")

    def test_unknown_refused(self):
        with pytest.raises(Exception):
            normalize_seat_spec("p24_b25")

    def test_provider_spec(self):
        spec = normalize_seat_spec(
            {"factory": "some.module:build", "kwargs": {"a": 1}, "arm_id": "agent2"}
        )
        assert spec == ("some.module:build", json.dumps({"a": 1}, sort_keys=True), "agent2")
        task = Task16(spec, "TINY", "board")
        assert task.arm_id == "agent2"

    def test_provider_spec_requires_factory_and_arm(self):
        with pytest.raises(Phase16MeasurementError):
            normalize_seat_spec({"factory": "no_colon", "arm_id": "x"})
        with pytest.raises(Phase16MeasurementError):
            normalize_seat_spec({"factory": "a:b"})


@pytest.fixture(scope="module")
def manifest():
    from stratego.evaluation.phase16.benchmark import load_benchmark_manifest

    try:
        return load_benchmark_manifest(root=REPOSITORY_ROOT)
    except Phase16MeasurementError:
        pytest.skip("the frozen benchmark manifest has not been generated yet")


@pytest.fixture(scope="module")
def parse_grid():
    spec = importlib.util.spec_from_file_location(
        "phase16_capture_setup",
        REPOSITORY_ROOT / "scripts/phase16_capture_setup.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_grid


class TestSubsets:
    def test_full(self, manifest):
        assert len(resolve_subset(manifest, None)) == 120

    def test_quick60(self, manifest):
        subset = resolve_subset(manifest, "quick60")
        assert len(subset) == 60
        assert set(subset) <= set(resolve_subset(manifest, None))

    def test_unknown_board_refused(self, manifest):
        with pytest.raises(Phase16MeasurementError):
            resolve_subset(manifest, ["not_a_board"])


class TestResume:
    def _fake_result(self, arm, preset, board):
        return {
            "row": {
                "arm_id": arm,
                "preset_id": preset,
                "board_id": board,
                "outcome": "win",
                "effective_score": 1.0,
            },
            "move_seconds": [],
            "fallback_reasons": {},
        }

    def test_finished_tasks_are_not_replayed(self, tmp_path):
        """A fully finished pack resumes without loading a single model."""
        out = tmp_path / "pack.jsonl"
        tasks = [Task16("p24_b24", "TINY", f"board_{index}") for index in range(3)]
        with out.open("w") as handle:
            for task in tasks:
                handle.write(
                    json.dumps(self._fake_result(*task.key)) + "\n"
                )
        results = run_pack16(tasks, root=str(tmp_path), workers=4, out_path=out)
        assert [entry["row"]["board_id"] for entry in results] == [
            task.board_id for task in tasks
        ]

    def test_load_results_keys(self, tmp_path):
        out = tmp_path / "pack.jsonl"
        out.write_text(json.dumps(self._fake_result("a", "TINY", "b")) + "\n")
        finished = load_results(out)
        assert ("a", "TINY", "b") in finished


class TestAnalysis:
    def _rows(self, arm, scores, family="scout_screen", preset="TINY"):
        return [
            {
                "arm_id": "p24_b24",
                "preset_id": preset,
                "setup_source": arm,
                "requested_family": family,
                "opponent": "p18",
                "ordinal": index,
                "board_id": f"{arm}|{index}",
                "outcome": (
                    "win" if score == 1.0 else "loss" if score == 0.0 else "draw"
                ),
                "effective_score": score,
            }
            for index, score in enumerate(scores)
        ]

    def test_paired_delta(self):
        left = self._rows("x", [1.0, 0.0, 1.0, 1.0])
        right = self._rows("y", [1.0, 1.0, 0.0, 1.0])
        delta = paired_delta_by(left, right)
        assert delta["pairs"] == 4
        assert delta["delta"] == 0.0
        assert delta["wins"] == 1 and delta["losses"] == 1 and delta["ties"] == 2

    def test_predeclared_reading(self):
        assert predeclared_reading(0.15) == "confirms_distribution_hypothesis"
        assert predeclared_reading(0.04) == "weakens_distribution_hypothesis"
        assert predeclared_reading(0.07) == "between_predeclared_thresholds"
        assert predeclared_reading(None) == "not_measured"

    def test_analyse_baseline(self):
        rows = (
            self._rows(ARM_CONTROL, [1.0, 1.0, 1.0, 1.0])
            + self._rows(ARM_ADVERSARIAL_OPPONENT, [1.0, 0.0, 0.0, 1.0])
            + self._rows(ARM_ADVERSARIAL_BOTH, [0.5, 1.0, 0.0, 1.0])
        )
        report = analyse_baseline(rows)
        assert report["arms"][ARM_CONTROL]["ewr"] == 1.0
        primary = report["paired"]["adversarial_opponent_minus_control"]
        assert primary["overall"]["delta"] == -0.5
        assert primary["drop"] == 0.5
        assert report["reading"] == "confirms_distribution_hypothesis"

    def test_mixed_presets_refused(self):
        rows = self._rows(ARM_CONTROL, [1.0]) + self._rows(
            ARM_CONTROL, [1.0], preset="MEDIUM"
        )
        with pytest.raises(Phase16MeasurementError):
            analyse_baseline(rows)


class TestOperatorLog:
    def _logged_game(self, tmp_path, library_document, *, index=1, winner="blue"):
        from stratego.evaluation.phase16.adversarial import library_entry

        setup = library_entry(library_document, "bombed_corner_flag", 1)
        logger = OperatorGameLogger(
            seats={"red": "human", "blue": "maximum_strength"},
            script="test",
            series="rebaseline_v1",
            game_index=index,
        )
        logger.set_setup(
            "red", canonical=setup["canonical_setup"], source="operator_entered"
        )
        logger.set_setup(
            "blue",
            canonical=library_entry(library_document, "miner_wall", 0)["canonical_setup"],
            source="phase14_learned",
            family_key="balanced_conventional",
        )
        logger.record_move("red", 100, 12.5)
        logger.record_move("blue", 200, 0.9)
        logger.finish(
            result="win", winner=winner, terminal_reason="flag_capture", plies=2
        )
        return logger.append(tmp_path / "log.jsonl")

    def test_schema_line(self, tmp_path, library_document):
        path = self._logged_game(tmp_path, library_document)
        games = read_log(path)
        assert len(games) == 1
        game = games[0]
        assert game["schema"] == OPERATOR_LOG_SCHEMA
        assert game["operator_color"] == "red"
        assert game["actions"] == [100, 200]
        assert game["move_seconds"]["red"] == [12.5]
        assert game["result"]["winner"] == "blue"
        assert len(game["setups"]["red"]["canonical"]) == 40

    def test_append_requires_finish_and_setups(self):
        logger = OperatorGameLogger(
            seats={"red": "human", "blue": "p24_direct"}, script="test"
        )
        with pytest.raises(Phase16MeasurementError):
            logger.append("nowhere.jsonl")

    def test_series_summary(self, tmp_path, library_document):
        self._logged_game(tmp_path, library_document, index=1, winner="blue")
        self._logged_game(tmp_path, library_document, index=2, winner="red")
        games = read_log(tmp_path / "log.jsonl")
        summary = operator_series_summary(games, "rebaseline_v1")
        assert summary["games"] == 2
        assert summary["machine_ewr"] == 0.5  # machine is blue: one win, one loss
        assert summary["running_machine_ewr"] == [1.0, 0.5]

    def test_harvest_from_log(self, tmp_path, library_document):
        from stratego.evaluation.phase16.adversarial import save_library

        self._logged_game(tmp_path, library_document)
        library_path = tmp_path / "library.json"
        save_library(library_document, library_path)
        report = harvest_operator_setups(
            log_path=tmp_path / "log.jsonl", library_path=library_path, root="."
        )
        assert report["operator_setups_found"] == 1
        assert len(report["appended"]) == 1
        # A second harvest of the same log: dedup, nothing appended.
        report = harvest_operator_setups(
            log_path=tmp_path / "log.jsonl", library_path=library_path, root="."
        )
        assert report["appended"] == []


class TestCaptureGrid:
    def _grid_text(self, canonical):
        from stratego.engine.constants import PIECE_TYPE_CODES

        lines = []
        for rank in range(4):
            lines.append(
                " ".join(PIECE_TYPE_CODES[canonical[rank * 10 + file]] for file in range(10))
            )
        return "\n".join(lines)

    def test_round_trip(self, parse_grid, library_document):
        from stratego.evaluation.phase16.adversarial import library_entry

        canonical = tuple(
            library_entry(library_document, "spy_shadow", 2)["canonical_setup"]
        )
        assert parse_grid(self._grid_text(canonical)) == canonical

    def test_front_first_flips(self, parse_grid, library_document):
        from stratego.evaluation.phase16.adversarial import library_entry

        canonical = tuple(
            library_entry(library_document, "spy_shadow", 2)["canonical_setup"]
        )
        text = self._grid_text(canonical)
        flipped = "\n".join(reversed(text.splitlines()))
        assert parse_grid(flipped, front_first=True) == canonical

    def test_bad_inventory_refused(self, parse_grid):
        grid = "\n".join(" ".join(["B"] * 10) for _ in range(4))
        with pytest.raises(Phase16MeasurementError):
            parse_grid(grid)

    def test_bad_token_refused(self, parse_grid, library_document):
        from stratego.evaluation.phase16.adversarial import library_entry

        canonical = tuple(
            library_entry(library_document, "spy_shadow", 2)["canonical_setup"]
        )
        text = self._grid_text(canonical).replace("S", "Z", 1)
        with pytest.raises(Phase16MeasurementError):
            parse_grid(text)
