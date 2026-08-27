"""Phase 16 Agent 3: window-edge targets.

Specification source: `03_AGENT_3_TRAINING_LOOP_V2.md` section 2.2.

The problem a window creates
---------------------------
Phase 9's targets are whole-game backward recursions: the advantage recursion
`A_t = delta_t + lambda_A * A_{t+1}` and the value recursion
`Y_t = (1 - lambda_V) * P_{t+1} + lambda_V * Y_{t+1}` both start at the
terminal step and walk backwards. A window collector stops mid-game, so
"the terminal step" is not always available when a batch has to be built.

Two rules, and why they differ
------------------------------
```text
advantage   TD(lambda_A) over the track's stored values; the tail is closed by
            a value -- the terminal z when the game ended, otherwise v at the
            boundary -- and A beyond the boundary is 0
W/D/L       lambda_V blending toward the final *outcome*, and an outcome is
            not a thing a bootstrap can invent; buffered per game until the
            game finishes
```

The asymmetry is not an oversight. An advantage is a difference of values and
degrades gracefully when the future is replaced by an estimate of itself. A
W/D/L target is a distribution anchored on a one-hot result, and anchoring it
on a prediction instead would train the value head toward its own output.

What the production path uses
-----------------------------
Because :func:`~stratego.training.phase9_loss.phase9_batch_loss` averages the
value and belief terms over *every* row -- it has no per-row loss mask, and
this phase does not rewrite the accepted objective -- a row whose W/D/L target
is not yet knowable cannot be in a batch at all. So the collector buffers a
game whole and emits it when it finishes, and :func:`track_targets` is then
called with the terminal `z`, in which case it reduces **exactly** to the
accepted whole-game functions. :func:`truncated_advantages` with a boundary
value is the partial-emission path: built, tested, and off by default. Section
7's `known_limitations` says so rather than leaving it to be discovered.

The `deltas` walk is the accepted one
-------------------------------------
`temporal_deltas` appends `z - values[-1]` as the final delta, so passing the
terminal z as the tail value reproduces it entry for entry. That is the whole
of the compatibility claim, and :func:`assert_reduces_to_accepted` proves it
on real numbers rather than asserting it in a docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..phase9_contract import (
    LAMBDA_ADVANTAGE,
    LAMBDA_VALUE,
    advantages as accepted_advantages,
    behavior_value_scalar,
    temporal_deltas as accepted_deltas,
    terminal_z,
    wdl_lambda_targets as accepted_wdl_targets,
)
from ..phase9_targets import (
    terminal_outcome,
    validate_behavior_wdl,
    validate_wdl_target,
)
from .contract import PHASE16_TARGETS_VERSION, Phase16TrainingError

#: float32 round-trip tolerance for the window/whole-game invariant.
INVARIANT_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# The two recursions, with an explicit tail
# ---------------------------------------------------------------------------


def boundary_deltas(values: "list[float]", tail_value: float) -> list:
    """`delta_t = v_{t+1} - v_t`, with the final delta closed by `tail_value`.

    With `tail_value = z` this is the accepted
    :func:`~stratego.training.phase9_contract.temporal_deltas` entry for entry;
    with `tail_value = v_boundary` it is the same walk over a window whose
    future has been replaced by one value estimate.
    """
    if not values:
        return []
    deltas = [float(values[t + 1]) - float(values[t]) for t in range(len(values) - 1)]
    deltas.append(float(tail_value) - float(values[-1]))
    return deltas


def truncated_advantages(
    values: "list[float]", tail_value: float, *, lambda_a: float = LAMBDA_ADVANTAGE
) -> list:
    """`A_t = delta_t + lambda_A * A_{t+1}`, with A beyond the tail = 0."""
    deltas = boundary_deltas(values, tail_value)
    result = [0.0] * len(deltas)
    following = 0.0
    for t in range(len(deltas) - 1, -1, -1):
        result[t] = deltas[t] + float(lambda_a) * following
        following = result[t]
    return result


def assert_reduces_to_accepted(values: "list[float]", z: int) -> dict:
    """Prove the windowed walk *is* the accepted walk when the tail is `z`.

    A mechanical check rather than a comment: if either recursion is ever
    edited, this stops agreeing and the invariant test fails.
    """
    ours = truncated_advantages(list(values), float(z))
    theirs = accepted_advantages(list(values), int(z))
    our_deltas = boundary_deltas(list(values), float(z))
    their_deltas = accepted_deltas(list(values), int(z))
    if len(ours) != len(theirs) or any(
        abs(a - b) > INVARIANT_TOLERANCE for a, b in zip(ours, theirs)
    ):
        raise Phase16TrainingError(
            "the Phase 16 advantage walk no longer reduces to the accepted one"
        )
    if len(our_deltas) != len(their_deltas) or any(
        abs(a - b) > INVARIANT_TOLERANCE for a, b in zip(our_deltas, their_deltas)
    ):
        raise Phase16TrainingError(
            "the Phase 16 delta walk no longer reduces to the accepted one"
        )
    return {
        "entries": len(ours),
        "max_advantage_difference": max(
            (abs(a - b) for a, b in zip(ours, theirs)), default=0.0
        ),
        "max_delta_difference": max(
            (abs(a - b) for a, b in zip(our_deltas, their_deltas)), default=0.0
        ),
        "reduces_to_accepted": True,
    }


# ---------------------------------------------------------------------------
# One learner colour's live track
# ---------------------------------------------------------------------------


@dataclass
class LearnerTrack:
    """One (game, colour) pair's stored values, accumulated as it is played.

    Holds no observation and no engine state: this is the cheap half of the
    two-pass split the accepted targets module already draws, kept cheap so a
    window boundary costs a backward walk over floats rather than a replay.
    """

    game_id: str
    player: int
    plies: list = field(default_factory=list)
    predictions: list = field(default_factory=list)
    values: list = field(default_factory=list)
    row_indices: list = field(default_factory=list)
    #: how many entries have already been emitted into a training window
    emitted: int = 0
    finished: bool = False
    outcome: "str | None" = None

    def __len__(self) -> int:
        return len(self.plies)

    def record(self, *, ply: int, prediction, row_index: int) -> None:
        """Append one stored learner decision."""
        if self.finished:
            raise Phase16TrainingError(
                f"{self.game_id}: a decision arrived after the track was closed"
            )
        values = tuple(float(value) for value in prediction)
        validate_behavior_wdl(values, where=f"{self.game_id} ply {ply}")
        self.plies.append(int(ply))
        self.predictions.append(values)
        self.values.append(behavior_value_scalar(values))
        self.row_indices.append(int(row_index))

    def close(self, terminal_result: str) -> None:
        """Close the track on the game's terminal result."""
        if self.finished:
            raise Phase16TrainingError(f"{self.game_id}: the track is already closed")
        self.outcome = terminal_outcome(terminal_result, self.player)
        self.finished = True

    @property
    def z(self) -> int:
        if not self.finished:
            raise Phase16TrainingError(f"{self.game_id}: the track has no terminal z yet")
        return terminal_z(self.outcome)

    @property
    def pending(self) -> int:
        """Entries collected but not yet emitted into a window."""
        return len(self.plies) - self.emitted


def track_targets(track: LearnerTrack) -> dict:
    """Both target families over a **finished** track: the exact whole-game math.

    Returns advantages and W/D/L targets over every entry of the track, which
    for a finished game are the accepted Phase 9 values entry for entry.
    """
    if not track.finished:
        raise Phase16TrainingError(
            f"{track.game_id}: exact targets need a finished track; use "
            "partial_advantages for a boundary emission"
        )
    if not track.plies:
        return {"advantages": (), "wdl_targets": (), "entries": 0}
    z = track.z
    advantages = truncated_advantages(list(track.values), float(z))
    targets = accepted_wdl_targets(list(track.predictions), track.outcome)
    for ply, target in zip(track.plies, targets):
        validate_wdl_target(target, where=f"{track.game_id} ply {ply}")
    return {
        "advantages": tuple(advantages),
        "wdl_targets": tuple(tuple(float(v) for v in row) for row in targets),
        "entries": len(track.plies),
        "outcome": track.outcome,
        "z": int(z),
    }


def partial_advantages(track: LearnerTrack) -> dict:
    """Advantages for an **unfinished** track, bootstrapped at the boundary.

    The carry-over rule: the track's last stored decision is *not* emitted --
    it is the boundary, and its own stored `v` closes the tail of the entries
    before it. Every emitted delta is therefore an exact `v_{t+1} - v_t`, and
    only the lambda tail is truncated. Emitting the boundary decision itself
    would require a value the window does not have.
    """
    if track.finished:
        raise Phase16TrainingError(
            f"{track.game_id}: a finished track uses track_targets, not a bootstrap"
        )
    if len(track.plies) < 2:
        return {"advantages": (), "entries": 0, "boundary_value": None}
    emittable = list(track.values[:-1])
    boundary_value = float(track.values[-1])
    return {
        "advantages": tuple(truncated_advantages(emittable, boundary_value)),
        "entries": len(emittable),
        "boundary_value": boundary_value,
        "boundary_ply": int(track.plies[-1]),
    }


# ---------------------------------------------------------------------------
# The required invariant (section 2.2)
# ---------------------------------------------------------------------------


def whole_game_targets(predictions, terminal_result: str, player: int) -> dict:
    """The accepted whole-game targets of one colour, computed in one call."""
    outcome = terminal_outcome(terminal_result, player)
    values = [behavior_value_scalar(row) for row in predictions]
    return {
        "advantages": tuple(accepted_advantages(values, terminal_z(outcome))),
        "wdl_targets": tuple(
            tuple(float(v) for v in row)
            for row in accepted_wdl_targets(list(predictions), outcome)
        ),
        "outcome": outcome,
    }


def windowed_targets(predictions, terminal_result: str, player: int, boundaries) -> dict:
    """The same targets built through the window machinery, across boundaries.

    `boundaries` are the entry counts at which windows closed. The track is fed
    in those chunks, exactly as the collector feeds it, and the targets are
    taken at the close -- which for a buffered game is the window in which it
    finished.
    """
    ordered = sorted({int(value) for value in boundaries if 0 < int(value) < len(predictions)})
    track = LearnerTrack(game_id="invariant", player=int(player))
    cursor = 0
    boundary_reports = []
    for stop in [*ordered, len(predictions)]:
        while cursor < stop:
            track.record(ply=cursor, prediction=predictions[cursor], row_index=cursor)
            cursor += 1
        if stop < len(predictions):
            boundary_reports.append(partial_advantages(track))
    track.close(terminal_result)
    exact = track_targets(track)
    return {
        "advantages": exact["advantages"],
        "wdl_targets": exact["wdl_targets"],
        "windows": len(ordered) + 1,
        "boundary_reports": boundary_reports,
    }


def window_edge_invariant(predictions, terminal_result: str, player: int, boundaries) -> dict:
    """Section 2.2's required invariant, as a computation that returns evidence.

    For a finished game whose stored values are identical, targets computed
    windowed (spanning >= 3 windows) equal targets computed whole-game to
    float32 tolerance.
    """
    ordered = sorted({int(v) for v in boundaries if 0 < int(v) < len(predictions)})
    if len(ordered) + 1 < 3:
        raise Phase16TrainingError(
            "the invariant is specified over at least three windows; supply at "
            "least two interior boundaries"
        )
    whole = whole_game_targets(predictions, terminal_result, player)
    windowed = windowed_targets(predictions, terminal_result, player, ordered)

    advantage_error = float(
        np.max(
            np.abs(
                np.asarray(windowed["advantages"], dtype=np.float64)
                - np.asarray(whole["advantages"], dtype=np.float64)
            )
        )
    )
    wdl_error = float(
        np.max(
            np.abs(
                np.asarray(windowed["wdl_targets"], dtype=np.float64)
                - np.asarray(whole["wdl_targets"], dtype=np.float64)
            )
        )
    )
    holds = advantage_error <= INVARIANT_TOLERANCE and wdl_error <= INVARIANT_TOLERANCE
    return {
        "targets_version": PHASE16_TARGETS_VERSION,
        "decisions": len(predictions),
        "windows": windowed["windows"],
        "boundaries": ordered,
        "max_advantage_difference": advantage_error,
        "max_wdl_difference": wdl_error,
        "tolerance": INVARIANT_TOLERANCE,
        "holds": bool(holds),
        "reduction_check": assert_reduces_to_accepted(
            [behavior_value_scalar(row) for row in predictions],
            terminal_z(whole["outcome"]),
        ),
    }


def targets_semantics() -> dict:
    return {
        "targets_version": PHASE16_TARGETS_VERSION,
        "lambda_A": LAMBDA_ADVANTAGE,
        "lambda_V": LAMBDA_VALUE,
        "advantage": (
            "A_t = delta_t + lambda_A * A_{t+1}; the tail is closed by the "
            "terminal z (finished) or by v at the boundary (partial); A beyond "
            "the tail is 0"
        ),
        "wdl": (
            "the accepted lambda_V recursion toward the one-hot outcome; "
            "buffered per game and never bootstrapped"
        ),
        "production_emission": (
            "whole games only, so every emitted row carries an exact W/D/L "
            "target and the accepted objective needs no per-row loss mask"
        ),
        "reduces_to_accepted": "truncated_advantages(values, z) == phase9 advantages(values, z)",
    }


__all__ = [
    "INVARIANT_TOLERANCE",
    "LearnerTrack",
    "assert_reduces_to_accepted",
    "boundary_deltas",
    "partial_advantages",
    "targets_semantics",
    "track_targets",
    "truncated_advantages",
    "whole_game_targets",
    "window_edge_invariant",
    "windowed_targets",
]
