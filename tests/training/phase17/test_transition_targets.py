"""Phase 17 Agent 2: boundary-bootstrapped targets, and gate `G-M4a`."""

from __future__ import annotations

import numpy as np
import pytest

from stratego.training.phase17.move_contract import (
    PROVENANCE_BOOTSTRAP,
    PROVENANCE_TERMINAL,
    TARGET_TOLERANCE,
)
from stratego.training.phase17.transition_targets import (
    BoundaryTail,
    Phase17TargetError,
    SeatTrace,
    advantages_with_tail,
    bootstrap_tail,
    deltas_with_tail,
    reduction_invariant,
    segment_targets,
    targets_semantics,
    terminal_tail,
    wdl_targets_with_tail,
    whole_game_divergence,
)
from stratego.training.phase9_contract import (
    advantages as accepted_advantages,
    behavior_value_scalar,
    temporal_deltas as accepted_deltas,
    terminal_z,
    wdl_lambda_targets as accepted_wdl_targets,
)

from .test_move_support import dirichlet_predictions

OUTCOMES = ("win", "draw", "loss")


# ---------------------------------------------------------------------------
# G-M4a: the governing reduction invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", OUTCOMES)
@pytest.mark.parametrize("length", [1, 2, 3, 12, 47])
def test_the_tailed_recursion_reduces_to_the_accepted_one(outcome, length):
    """Gate `G-M4a`. When the tail is the true terminal continuation, the
    Phase 17 walk reproduces the accepted whole-game walk entry for entry."""
    predictions = dirichlet_predictions(length, seed=1000 + length)
    report = reduction_invariant(predictions, outcome)
    assert report["gate"] == "G-M4a"
    assert report["reduces_to_accepted"] is True
    assert report["max_delta_difference"] <= TARGET_TOLERANCE
    assert report["max_advantage_difference"] <= TARGET_TOLERANCE
    assert report["max_wdl_difference"] <= TARGET_TOLERANCE
    assert report["decisions"] == length


@pytest.mark.parametrize("outcome", OUTCOMES)
def test_the_terminal_tail_is_exactly_the_accepted_walk(outcome):
    predictions = dirichlet_predictions(20, seed=77)
    values = [behavior_value_scalar(row) for row in predictions]
    tail = terminal_tail(outcome)

    assert deltas_with_tail(values, tail.value) == pytest.approx(
        accepted_deltas(list(values), terminal_z(outcome))
    )
    assert advantages_with_tail(values, tail.value, tail.advantage) == pytest.approx(
        accepted_advantages(list(values), terminal_z(outcome))
    )
    ours = wdl_targets_with_tail(predictions, tail.wdl_prediction, tail.wdl_target)
    theirs = accepted_wdl_targets(list(predictions), outcome)
    assert np.allclose(np.asarray(ours), np.asarray(theirs), atol=0.0, rtol=0.0)
    assert tail.provenance == PROVENANCE_TERMINAL


def test_an_empty_sequence_is_refused_by_the_invariant():
    with pytest.raises(Phase17TargetError, match="at least one decision"):
        reduction_invariant([], "win")


def test_an_unknown_outcome_is_refused():
    with pytest.raises(Phase17TargetError, match="unknown terminal outcome"):
        terminal_tail("resign")


# ---------------------------------------------------------------------------
# The bootstrap
# ---------------------------------------------------------------------------


def test_the_bootstrap_reads_one_stored_prediction_two_ways():
    """The scalar tail and the W/D/L tail are the same stored prediction.

    Not two independently produced numbers: `value` is the accepted
    `behavior_value_scalar` of the very prediction that closes the W/D/L walk.
    """
    prediction = (0.55, 0.2, 0.25)
    tail = bootstrap_tail(prediction, model_state_digest="abc123")
    assert tail.value == pytest.approx(behavior_value_scalar(prediction))
    assert tail.wdl_prediction == pytest.approx(prediction)
    assert tail.wdl_target == pytest.approx(prediction)
    assert tail.advantage == 0.0
    assert tail.provenance == PROVENANCE_BOOTSTRAP


def test_a_bootstrap_without_a_named_snapshot_is_refused():
    with pytest.raises(Phase17TargetError, match="must name the raw snapshot"):
        BoundaryTail(
            kind="bootstrap",
            value=0.1,
            advantage=0.0,
            wdl_prediction=(0.4, 0.3, 0.3),
            wdl_target=(0.4, 0.3, 0.3),
        )


def test_a_bootstrapped_wdl_target_is_the_boundary_prediction_itself():
    """`Y_last = (1-lV)*P_b + lV*P_b = P_b`, which is what section 6 asks for."""
    predictions = dirichlet_predictions(5, seed=9)
    boundary = (0.3, 0.45, 0.25)
    targets = wdl_targets_with_tail(predictions, boundary, boundary)
    assert targets[-1] == pytest.approx(boundary)


def test_an_unfinished_suffix_changes_only_through_the_supplied_bootstrap():
    """Two bootstraps that differ change the segment only via the tail.

    Every delta inside the segment is an exact `v_{t+1} - v_t` and is untouched
    by the tail; the tail enters through the last delta and the lambda walk
    back from it.
    """
    predictions = dirichlet_predictions(6, seed=31)
    values = [behavior_value_scalar(row) for row in predictions]
    low = advantages_with_tail(values, -0.9, 0.0)
    high = advantages_with_tail(values, 0.9, 0.0)
    difference = np.asarray(high) - np.asarray(low)
    # A_t responds to the tail with weight lambda_A**(last - t): monotone,
    # strictly decreasing backwards, and never zero.
    assert difference[-1] == pytest.approx(1.8)
    assert np.all(difference > 0)
    assert np.all(np.diff(difference) > 0)
    assert difference[-2] == pytest.approx(1.8 * 0.5)

    # The deltas inside the segment are unchanged by the tail.
    assert deltas_with_tail(values, -0.9)[:-1] == pytest.approx(
        deltas_with_tail(values, 0.9)[:-1]
    )


def test_a_tail_that_is_not_a_simplex_point_is_refused():
    with pytest.raises(Phase17TargetError, match="sums to"):
        bootstrap_tail((0.5, 0.5, 0.5), model_state_digest="abc")
    with pytest.raises(Phase17TargetError, match="wdl_prediction sums to"):
        BoundaryTail(
            kind="terminal",
            value=1.0,
            advantage=0.0,
            wdl_prediction=(0.5, 0.5, 0.5),
            wdl_target=(1.0, 0.0, 0.0),
        )
    with pytest.raises(Phase17TargetError, match="not a simplex point"):
        BoundaryTail(
            kind="terminal",
            value=1.0,
            advantage=0.0,
            wdl_prediction=(1.0, 0.0, 0.0),
            wdl_target=(-0.5, 1.0, 0.5),
        )


def test_the_tail_round_trips_through_its_dict():
    tail = bootstrap_tail((0.2, 0.3, 0.5), model_state_digest="digest0")
    assert BoundaryTail.from_dict(tail.to_dict()) == tail


# ---------------------------------------------------------------------------
# Perspective
# ---------------------------------------------------------------------------


def test_swapping_perspective_transforms_outcomes_and_targets():
    """Red and Blue read the same sealed result with opposite signs."""
    predictions = dirichlet_predictions(8, seed=5)
    red = SeatTrace(game_id="g", color=0)
    blue = SeatTrace(game_id="g", color=1)
    for index, row in enumerate(predictions):
        red.record(ply=2 * index, wdl=row)
        blue.record(ply=2 * index + 1, wdl=row)
    assert red.close("red_win") == "win"
    assert blue.close("red_win") == "loss"

    red_rows = red.emit(None)["rows"]
    blue_rows = blue.emit(None)["rows"]
    assert terminal_tail("win").value == -terminal_tail("loss").value
    # Identical stored values, opposite terminal z: the final advantage differs
    # by exactly the difference in z.
    assert red_rows[-1]["advantage_target"] - blue_rows[-1]["advantage_target"] == (
        pytest.approx(2.0)
    )
    assert red_rows[-1]["wdl_target"] == (1.0, 0.0, 0.0)
    assert blue_rows[-1]["wdl_target"] == (0.0, 0.0, 1.0)


def test_a_draw_reads_the_same_from_both_seats():
    predictions = dirichlet_predictions(4, seed=6)
    traces = []
    for color in (0, 1):
        trace = SeatTrace(game_id="g", color=color)
        for index, row in enumerate(predictions):
            trace.record(ply=index, wdl=row)
        assert trace.close("draw") == "draw"
        traces.append(trace.emit(None)["rows"])
    assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# The carry state
# ---------------------------------------------------------------------------


def _play_windowed(predictions, boundaries, terminal_result="red_win"):
    """Feed a trace in windows, emitting at each boundary. Returns every row."""
    trace = SeatTrace(game_id="g", color=0)
    rows: list = []
    cursor = 0
    for stop in [*boundaries, len(predictions)]:
        while cursor < stop:
            trace.record(ply=cursor, wdl=predictions[cursor])
            cursor += 1
        if stop < len(predictions):
            tail = bootstrap_tail(predictions[stop], model_state_digest="snapshot")
            rows.extend(trace.emit(tail)["rows"])
            trace.carried()
        else:
            trace.close(terminal_result)
            rows.extend(trace.emit(None)["rows"])
    return trace, rows


def test_every_collected_transition_is_emitted_exactly_once():
    predictions = dirichlet_predictions(12, seed=20260827)
    trace, rows = _play_windowed(predictions, [4, 8])
    assert len(rows) == 12
    assert [row["index"] for row in rows] == list(range(12))
    assert trace.pending == 0
    assert trace.emit(None)["entries"] == 0


def test_the_bootstrap_age_counts_the_boundaries_the_trace_crossed():
    predictions = dirichlet_predictions(12, seed=20260827)
    _trace, rows = _play_windowed(predictions, [4, 8])
    assert [row["bootstrap_age_windows"] for row in rows] == [0] * 4 + [1] * 4 + [2] * 4


def test_provenance_and_boundary_status_are_stamped_per_row():
    predictions = dirichlet_predictions(12, seed=20260827)
    _trace, rows = _play_windowed(predictions, [4, 8])
    provenance = [row["target_provenance"] for row in rows]
    assert provenance == [PROVENANCE_BOOTSTRAP] * 8 + [PROVENANCE_TERMINAL] * 4
    status = [row["boundary_status"] for row in rows]
    assert status[3] == "window_boundary"
    assert status[7] == "window_boundary"
    assert status[11] == "terminal"
    assert status[0] == status[1] == status[2] == "interior"


def test_the_final_window_reproduces_the_accepted_targets_for_its_own_rows():
    """The tail of the last window IS the terminal z, so those rows are exact."""
    predictions = dirichlet_predictions(12, seed=20260827)
    _trace, rows = _play_windowed(predictions, [4, 8])
    values = [behavior_value_scalar(row) for row in predictions]
    whole = accepted_advantages(list(values), terminal_z("win"))
    for index in range(8, 12):
        assert rows[index]["advantage_target"] == pytest.approx(whole[index])


def test_the_measured_divergence_is_real_and_is_reported_not_gated():
    """Operator decision D2, checked: `G-M4b` cannot be reinstated silently.

    If this ever came out at zero, either the bootstrap stopped bootstrapping
    or partial emission stopped being partial.
    """
    predictions = dirichlet_predictions(12, seed=20260827)
    trace, _rows = _play_windowed(predictions, [4, 8])
    report = whole_game_divergence(trace)
    assert report["entries"] == 12
    assert report["bootstrapped_rows"] == 8
    assert report["max_advantage_divergence"] > 0.05
    assert report["max_wdl_divergence"] > 0.05
    # The final window's rows are exact, and say so.
    exact = [row for row in report["rows"] if row["target_provenance"] == PROVENANCE_TERMINAL]
    assert len(exact) == 4
    assert all(row["boundary_target_divergence"] == pytest.approx(0.0) for row in exact)
    assert all(row["boundary_wdl_divergence"] == pytest.approx(0.0) for row in exact)


def test_divergence_needs_a_finished_trace():
    trace = SeatTrace(game_id="g", color=0)
    trace.record(ply=0, wdl=(0.4, 0.3, 0.3))
    with pytest.raises(Phase17TargetError, match="finished trace"):
        whole_game_divergence(trace)


def test_an_unfinished_trace_refuses_to_emit_without_a_stored_boundary():
    trace = SeatTrace(game_id="g", color=0)
    trace.record(ply=0, wdl=(0.4, 0.3, 0.3))
    with pytest.raises(Phase17TargetError, match="stored boundary prediction"):
        trace.emit(None)


def test_a_decision_after_close_is_refused():
    trace = SeatTrace(game_id="g", color=0)
    trace.record(ply=0, wdl=(0.4, 0.3, 0.3))
    trace.close("red_win")
    with pytest.raises(Phase17TargetError, match="after the trace closed"):
        trace.record(ply=1, wdl=(0.4, 0.3, 0.3))


def test_the_carry_state_round_trips_without_duplicating_or_omitting():
    predictions = dirichlet_predictions(9, seed=44)
    trace = SeatTrace(game_id="g", color=1)
    for index in range(4):
        trace.record(ply=index, wdl=predictions[index])
    first = trace.emit(bootstrap_tail(predictions[4], model_state_digest="s"))["rows"]
    trace.carried()

    restored = SeatTrace.from_dict(trace.to_dict())
    assert restored.to_dict() == trace.to_dict()
    assert restored.emitted == 4
    assert restored.pending == 0
    assert restored.windows_spanned == 1

    for index in range(4, 9):
        restored.record(ply=index, wdl=predictions[index])
    restored.close("blue_win")
    second = restored.emit(None)["rows"]
    assert [row["index"] for row in first + second] == list(range(9))
    assert len(first + second) == 9


def test_segment_targets_refuses_mismatched_inputs():
    with pytest.raises(Phase17TargetError, match="values for"):
        segment_targets([0.1, 0.2], [(0.3, 0.4, 0.3)], terminal_tail("win"))


def test_targets_semantics_names_the_retired_invariant():
    semantics = targets_semantics()
    assert semantics["governing_invariant"].startswith("G-M4a")
    assert "G-M4b" in semantics["retired_invariant"]
    assert "never a gate" in semantics["divergence"]
    assert "STORED" in semantics["bootstrap_source"]
