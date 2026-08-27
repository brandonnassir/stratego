"""Phase 16 Agent 3: window-edge targets, including section 2.2's invariant."""

import numpy as np
import pytest

from stratego.training.phase16 import targets as T
from stratego.training.phase16.contract import Phase16TrainingError


def simplex_predictions(count: int, seed: int = 11) -> list:
    """`count` valid W/D/L predictions from one seed."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(count):
        draw = rng.random(3)
        draw = draw / draw.sum()
        rows.append(tuple(float(value) for value in draw))
    return rows


# ---------------------------------------------------------------------------
# The recursions
# ---------------------------------------------------------------------------


def test_the_windowed_walk_reduces_to_the_accepted_walk_at_a_terminal_tail():
    from stratego.training.phase9_contract import advantages as accepted

    for seed in range(5):
        values = list(np.random.default_rng(seed).uniform(-1, 1, 23))
        for z in (-1, 0, 1):
            assert T.truncated_advantages(values, float(z)) == pytest.approx(
                accepted(values, z)
            )
            report = T.assert_reduces_to_accepted(values, z)
            assert report["reduces_to_accepted"]
            assert report["max_advantage_difference"] == 0.0


def test_a_boundary_value_closes_the_final_delta():
    values = [0.1, 0.4, -0.2]
    boundary = 0.8
    deltas = T.boundary_deltas(values, boundary)
    assert deltas == pytest.approx([0.3, -0.6, 1.0])
    # A_t = delta_t + 0.5 * A_{t+1}, A beyond the boundary = 0
    assert T.truncated_advantages(values, boundary) == pytest.approx(
        [0.3 + 0.5 * (-0.6 + 0.5 * 1.0), -0.6 + 0.5 * 1.0, 1.0]
    )


def test_empty_sequences_are_empty_not_an_error():
    assert T.truncated_advantages([], 0.0) == []
    assert T.boundary_deltas([], 0.0) == []


# ---------------------------------------------------------------------------
# The live track
# ---------------------------------------------------------------------------


def test_a_track_accumulates_values_and_closes_on_the_result():
    predictions = simplex_predictions(6)
    track = T.LearnerTrack(game_id="g", player=0)
    for ply, prediction in enumerate(predictions):
        track.record(ply=ply * 2, prediction=prediction, row_index=ply)
    assert len(track) == 6
    assert track.pending == 6
    assert track.values[0] == pytest.approx(predictions[0][0] - predictions[0][2])
    with pytest.raises(Phase16TrainingError):
        _ = track.z
    track.close("red_win")
    assert track.outcome == "win" and track.z == 1
    with pytest.raises(Phase16TrainingError):
        track.record(ply=99, prediction=predictions[0], row_index=99)
    with pytest.raises(Phase16TrainingError):
        track.close("red_win")


def test_a_track_refuses_a_prediction_that_is_not_on_the_simplex():
    track = T.LearnerTrack(game_id="g", player=0)
    for bad in ((0.5, 0.5, 0.5), (0.5, -0.1, 0.6), (1.0, 0.0)):
        with pytest.raises(Exception):
            track.record(ply=0, prediction=bad, row_index=0)


def test_blue_and_red_read_the_same_result_from_opposite_sides():
    predictions = simplex_predictions(4)
    red, blue = T.LearnerTrack(game_id="g", player=0), T.LearnerTrack(game_id="g", player=1)
    for ply, prediction in enumerate(predictions):
        red.record(ply=ply, prediction=prediction, row_index=ply)
        blue.record(ply=ply, prediction=prediction, row_index=ply)
    red.close("red_win")
    blue.close("red_win")
    assert (red.outcome, blue.outcome) == ("win", "loss")
    assert (red.z, blue.z) == (1, -1)


def test_exact_targets_need_a_finished_track_and_a_bootstrap_needs_an_open_one():
    track = T.LearnerTrack(game_id="g", player=0)
    for ply, prediction in enumerate(simplex_predictions(5)):
        track.record(ply=ply, prediction=prediction, row_index=ply)
    partial = T.partial_advantages(track)
    # the carry-over rule: the last decision is the boundary and is not emitted
    assert partial["entries"] == 4
    assert partial["boundary_value"] == pytest.approx(track.values[-1])
    with pytest.raises(Phase16TrainingError):
        T.track_targets(track)
    track.close("draw")
    exact = T.track_targets(track)
    assert exact["entries"] == 5
    assert exact["wdl_targets"][-1] == (0.0, 1.0, 0.0)
    with pytest.raises(Phase16TrainingError):
        T.partial_advantages(track)


def test_a_one_decision_track_has_nothing_to_bootstrap():
    track = T.LearnerTrack(game_id="g", player=0)
    track.record(ply=0, prediction=(0.5, 0.3, 0.2), row_index=0)
    assert T.partial_advantages(track)["entries"] == 0


# ---------------------------------------------------------------------------
# Section 2.2's required invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("result", ["red_win", "blue_win", "draw"])
@pytest.mark.parametrize("player", [0, 1])
def test_windowed_targets_equal_whole_game_targets_across_three_windows(result, player):
    """The invariant section 2.2 requires, on every outcome and both colours."""
    predictions = simplex_predictions(41, seed=5)
    report = T.window_edge_invariant(predictions, result, player, [9, 20, 33])
    assert report["windows"] == 4 >= 3
    assert report["holds"]
    assert report["max_advantage_difference"] <= T.INVARIANT_TOLERANCE
    assert report["max_wdl_difference"] <= T.INVARIANT_TOLERANCE


def test_the_invariant_holds_for_many_boundary_layouts():
    predictions = simplex_predictions(60, seed=17)
    rng = np.random.default_rng(3)
    for _ in range(12):
        boundaries = sorted(rng.choice(range(1, 60), size=4, replace=False).tolist())
        report = T.window_edge_invariant(predictions, "blue_win", 1, boundaries)
        assert report["holds"], boundaries
        assert report["windows"] >= 3


def test_the_invariant_refuses_fewer_than_three_windows():
    predictions = simplex_predictions(12)
    with pytest.raises(Phase16TrainingError):
        T.window_edge_invariant(predictions, "draw", 0, [6])


def test_windowed_targets_are_bit_identical_not_merely_close():
    """"To float32 tolerance" is the ceiling; the buffer path should be exact."""
    predictions = simplex_predictions(37, seed=23)
    whole = T.whole_game_targets(predictions, "red_win", 0)
    windowed = T.windowed_targets(predictions, "red_win", 0, [4, 15, 27])
    assert windowed["advantages"] == whole["advantages"]
    assert windowed["wdl_targets"] == whole["wdl_targets"]


def test_every_windowed_wdl_target_is_a_distribution():
    from stratego.training.phase9_targets import validate_wdl_target

    predictions = simplex_predictions(25, seed=31)
    windowed = T.windowed_targets(predictions, "draw", 1, [5, 11, 19])
    for index, target in enumerate(windowed["wdl_targets"]):
        validate_wdl_target(target, where=f"entry {index}")


def test_boundary_reports_are_produced_at_each_interior_window():
    predictions = simplex_predictions(30, seed=13)
    windowed = T.windowed_targets(predictions, "red_win", 0, [7, 14, 21])
    assert len(windowed["boundary_reports"]) == 3
    assert [report["entries"] for report in windowed["boundary_reports"]] == [6, 13, 20]


def test_semantics_names_the_production_emission_rule():
    semantics = T.targets_semantics()
    assert semantics["lambda_A"] == 0.5 and semantics["lambda_V"] == 0.8
    assert "whole games only" in semantics["production_emission"]
    assert "never bootstrapped" in semantics["wdl"]
