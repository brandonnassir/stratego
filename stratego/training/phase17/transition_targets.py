"""Phase 17 Agent 2: boundary-bootstrapped targets and the carry state.

Specification sources: common contract section 6 (as amended by operator
decision D2 on 2026-08-27), Agent 2 instruction section 5,
`reports/phase17/agent_01_boundary_target_probe.json`.

What changed, and what did not
------------------------------
Phase 16 buffered a whole game and emitted it when it finished, so every
emitted row carried an exact whole-game target. Phase 17 emits **every**
transition of the current window and closes the open traces on a value
estimate. That is not a weaker version of the accepted recursion; it is the
same recursion with a different tail.

```text
advantage   A_t = delta_t + lambda_A * A_{t+1}
            delta_t = v_{t+1} - v_t inside the segment
            delta_last = tail_value - v_last
            A beyond the tail = tail_advantage
W/D/L       Y_t = (1 - lambda_V) * P_{t+1} + lambda_V * Y_{t+1}
            Y_last = (1 - lambda_V) * tail_prediction + lambda_V * tail_target
```

Both are *one* function each, with the tail supplied. Substituting the true
terminal continuation recovers the accepted whole-game walk entry for entry:

```text
terminal    tail_value = z,  tail_advantage = 0
            tail_prediction = tail_target = one_hot(outcome)
                 =>  Y_last = (1-lV)*onehot + lV*onehot = onehot
```

That substitution is gate `G-M4a`, and :func:`reduction_invariant` computes it
rather than asserting it.

Why `G-M4b` could not have held
-------------------------------
Operator decision D2 retired the requirement that a game split over three or
more windows reproduce the whole-game targets. It is impossible alongside
partial emission: a lambda-return truncated at a boundary and closed on a
value estimate equals the full lambda-return only when the estimate happens to
equal the continuation it replaces. Agent 1 measured the gap at 0.309 in
advantage and 0.121 in W/D/L on a three-window synthetic track. This module
therefore *measures* the divergence per row and reports it as telemetry --
see :func:`whole_game_divergence` -- and the tolerance is never weakened to
make a retired invariant pass.

The bootstrap is a stored prediction
------------------------------------
Common contract section 6.2: an unfinished trace takes its tail from the
**stored** boundary value and stored boundary W/D/L prediction -- never from a
later-known outcome and never from a prediction recomputed at training time.
:class:`BoundaryTail` is that stored object; it rides in the carry state and in
the checkpoint, so a resumed window closes its traces on the same numbers the
uninterrupted one would.
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
    Phase9TargetError,
    terminal_outcome,
    validate_behavior_wdl,
)
from .move_contract import (
    BOUNDARY_INTERIOR,
    BOUNDARY_TERMINAL,
    BOUNDARY_WINDOW,
    MOVE_TARGETS_VERSION,
    PROVENANCE_BOOTSTRAP,
    PROVENANCE_TERMINAL,
    TARGET_TOLERANCE,
    Phase17MoveError,
)

TAIL_TERMINAL = "terminal"
TAIL_BOOTSTRAP = "bootstrap"

_ONE_HOT = {
    "win": (1.0, 0.0, 0.0),
    "draw": (0.0, 1.0, 0.0),
    "loss": (0.0, 0.0, 1.0),
}


class Phase17TargetError(Phase17MoveError):
    """A Phase 17 target could not be built as specified."""


# ---------------------------------------------------------------------------
# The tail
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryTail:
    """Everything a segment needs to close, and where it came from.

    `kind` is load-bearing: it is what stamps `target_provenance` onto every
    row of the segment, so a later reader can separate exact targets from
    bootstrapped ones without re-deriving anything.
    """

    kind: str
    value: float
    advantage: float
    wdl_prediction: tuple
    wdl_target: tuple
    #: the raw snapshot that produced a bootstrap prediction; None for terminal
    model_state_digest: "str | None" = None
    #: the terminal outcome from this seat's perspective; None for a bootstrap
    outcome: "str | None" = None

    def __post_init__(self) -> None:
        if self.kind not in (TAIL_TERMINAL, TAIL_BOOTSTRAP):
            raise Phase17TargetError(f"unknown boundary tail kind: {self.kind!r}")
        for name in ("wdl_prediction", "wdl_target"):
            row = np.asarray(getattr(self, name), dtype=np.float64)
            if row.shape != (3,) or not np.isfinite(row).all() or bool((row < 0).any()):
                raise Phase17TargetError(f"the tail's {name} is not a simplex point")
            if abs(float(row.sum()) - 1.0) > 1e-4:
                raise Phase17TargetError(
                    f"the tail's {name} sums to {float(row.sum())!r}"
                )
        if not np.isfinite([self.value, self.advantage]).all():
            raise Phase17TargetError("a non-finite boundary tail value")
        if self.kind == TAIL_BOOTSTRAP and not self.model_state_digest:
            raise Phase17TargetError(
                "a bootstrapped tail must name the raw snapshot that predicted it"
            )

    @property
    def provenance(self) -> str:
        return PROVENANCE_TERMINAL if self.kind == TAIL_TERMINAL else PROVENANCE_BOOTSTRAP

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "value": float(self.value),
            "advantage": float(self.advantage),
            "wdl_prediction": [float(v) for v in self.wdl_prediction],
            "wdl_target": [float(v) for v in self.wdl_target],
            "model_state_digest": self.model_state_digest,
            "outcome": self.outcome,
            "provenance": self.provenance,
        }

    @staticmethod
    def from_dict(payload: dict) -> "BoundaryTail":
        return BoundaryTail(
            kind=str(payload["kind"]),
            value=float(payload["value"]),
            advantage=float(payload["advantage"]),
            wdl_prediction=tuple(float(v) for v in payload["wdl_prediction"]),
            wdl_target=tuple(float(v) for v in payload["wdl_target"]),
            model_state_digest=payload.get("model_state_digest"),
            outcome=payload.get("outcome"),
        )


def terminal_tail(outcome: str) -> BoundaryTail:
    """The tail that reproduces the accepted whole-game recursion exactly.

    `tail_value = z`, `tail_advantage = 0`, and both W/D/L slots the one-hot
    outcome -- under which `Y_last = (1-lV)*Z + lV*Z = Z`, which is the
    accepted terminal target.
    """
    if outcome not in _ONE_HOT:
        raise Phase17TargetError(f"unknown terminal outcome: {outcome!r}")
    one_hot = _ONE_HOT[outcome]
    return BoundaryTail(
        kind=TAIL_TERMINAL,
        value=float(terminal_z(outcome)),
        advantage=0.0,
        wdl_prediction=one_hot,
        wdl_target=one_hot,
        outcome=outcome,
    )


def bootstrap_tail(wdl, *, model_state_digest: str) -> BoundaryTail:
    """The tail of an unfinished trace: the stored boundary W/D/L prediction.

    The scalar bootstrap is `P(W) - P(L)` of that same stored prediction --
    the accepted `behavior_value_scalar`, so the estimate that closes the
    advantage walk and the one that closes the W/D/L walk are the same
    prediction read two ways, never two independently produced numbers.

    `A` beyond the boundary is 0: the trace is truncated there, and inventing a
    continuation advantage would be inventing data.
    """
    row = tuple(float(v) for v in np.asarray(wdl, dtype=np.float64).reshape(3))
    try:
        validate_behavior_wdl(row, where="boundary bootstrap prediction")
    except Phase9TargetError as error:
        raise Phase17TargetError(str(error)) from error
    return BoundaryTail(
        kind=TAIL_BOOTSTRAP,
        value=float(behavior_value_scalar(row)),
        advantage=0.0,
        wdl_prediction=row,
        wdl_target=row,
        model_state_digest=str(model_state_digest),
    )


# ---------------------------------------------------------------------------
# The two recursions, with an explicit tail
# ---------------------------------------------------------------------------


def deltas_with_tail(values, tail_value: float) -> list:
    """`delta_t = v_{t+1} - v_t`, with the final delta closed by `tail_value`."""
    ordered = [float(value) for value in values]
    if not ordered:
        return []
    deltas = [ordered[t + 1] - ordered[t] for t in range(len(ordered) - 1)]
    deltas.append(float(tail_value) - ordered[-1])
    return deltas


def advantages_with_tail(values, tail_value: float, tail_advantage: float = 0.0) -> list:
    """`A_t = delta_t + lambda_A * A_{t+1}`, with `A` beyond the tail supplied."""
    deltas = deltas_with_tail(values, tail_value)
    result = [0.0] * len(deltas)
    following = float(tail_advantage)
    for t in range(len(deltas) - 1, -1, -1):
        result[t] = deltas[t] + LAMBDA_ADVANTAGE * following
        following = result[t]
    return result


def wdl_targets_with_tail(predictions, tail_prediction, tail_target) -> list:
    """`Y_t = (1 - lambda_V) * P_{t+1} + lambda_V * Y_{t+1}`, tail supplied.

    The accepted recursion sets the terminal target to the one-hot outcome
    directly. Written with a tail, that is the same formula with both tail
    slots equal to the one-hot -- which is why :func:`terminal_tail` reduces
    exactly instead of approximately.
    """
    rows = [tuple(float(v) for v in row) for row in predictions]
    if not rows:
        return []
    prediction = tuple(float(v) for v in tail_prediction)
    following = tuple(float(v) for v in tail_target)
    targets: list = [None] * len(rows)
    targets[-1] = tuple(
        (1.0 - LAMBDA_VALUE) * prediction[k] + LAMBDA_VALUE * following[k]
        for k in range(3)
    )
    for t in range(len(rows) - 2, -1, -1):
        nxt = rows[t + 1]
        nxt_target = targets[t + 1]
        targets[t] = tuple(
            (1.0 - LAMBDA_VALUE) * nxt[k] + LAMBDA_VALUE * nxt_target[k]
            for k in range(3)
        )
    return targets


def segment_targets(values, predictions, tail: BoundaryTail) -> dict:
    """Both target families over one emitted segment, closed by `tail`."""
    if len(values) != len(predictions):
        raise Phase17TargetError(
            f"{len(values)} values for {len(predictions)} W/D/L predictions"
        )
    if not values:
        return {"advantages": (), "wdl_targets": (), "entries": 0}
    return {
        "advantages": tuple(advantages_with_tail(values, tail.value, tail.advantage)),
        "wdl_targets": tuple(
            wdl_targets_with_tail(predictions, tail.wdl_prediction, tail.wdl_target)
        ),
        "entries": len(values),
        "provenance": tail.provenance,
        "tail": tail.to_dict(),
    }


# ---------------------------------------------------------------------------
# The governing invariant (G-M4a)
# ---------------------------------------------------------------------------


def reduction_invariant(predictions, outcome: str) -> dict:
    """Gate `G-M4a`, computed: the tailed walk IS the accepted walk at terminal.

    A mechanical check rather than a comment. If either recursion is ever
    edited, this stops agreeing and the invariant test fails.
    """
    rows = [tuple(float(v) for v in row) for row in predictions]
    if not rows:
        raise Phase17TargetError("the reduction invariant needs at least one decision")
    if outcome not in _ONE_HOT:
        raise Phase17TargetError(f"unknown terminal outcome: {outcome!r}")
    values = [behavior_value_scalar(row) for row in rows]
    z = terminal_z(outcome)
    tail = terminal_tail(outcome)

    ours_deltas = deltas_with_tail(values, tail.value)
    theirs_deltas = accepted_deltas(list(values), z)
    ours_advantage = advantages_with_tail(values, tail.value, tail.advantage)
    theirs_advantage = accepted_advantages(list(values), z)
    ours_wdl = wdl_targets_with_tail(rows, tail.wdl_prediction, tail.wdl_target)
    theirs_wdl = accepted_wdl_targets(list(rows), outcome)

    def worst(a, b) -> float:
        left = np.asarray(a, dtype=np.float64)
        right = np.asarray(b, dtype=np.float64)
        if left.shape != right.shape:
            raise Phase17TargetError(
                f"the tailed walk produced {left.shape} entries and the accepted "
                f"walk {right.shape}"
            )
        return float(np.max(np.abs(left - right))) if left.size else 0.0

    delta_error = worst(ours_deltas, theirs_deltas)
    advantage_error = worst(ours_advantage, theirs_advantage)
    wdl_error = worst(ours_wdl, theirs_wdl)
    holds = max(delta_error, advantage_error, wdl_error) <= TARGET_TOLERANCE
    if not holds:
        raise Phase17TargetError(
            "the Phase 17 tailed recursion no longer reduces to the accepted "
            f"whole-game recursion (delta {delta_error:.3e}, advantage "
            f"{advantage_error:.3e}, wdl {wdl_error:.3e})"
        )
    return {
        "gate": "G-M4a",
        "targets_version": MOVE_TARGETS_VERSION,
        "decisions": len(rows),
        "outcome": outcome,
        "z": int(z),
        "max_delta_difference": delta_error,
        "max_advantage_difference": advantage_error,
        "max_wdl_difference": wdl_error,
        "tolerance": TARGET_TOLERANCE,
        "reduces_to_accepted": True,
        "retired": "G-M4b, by operator decision D2 (2026-08-27)",
    }


# ---------------------------------------------------------------------------
# One seat's live trace: the carry state
# ---------------------------------------------------------------------------


@dataclass
class SeatTrace:
    """One `(game, colour)` pair's stored decisions, carried across windows.

    Holds no observation and no engine state: a window boundary costs a
    backward walk over floats, never a replay. The whole sequence is kept --
    not only the un-emitted tail -- because the divergence telemetry compares
    an already-emitted row against the target its eventual terminal outcome
    would have given it, and that comparison needs the entries before it.
    """

    game_id: str
    color: int
    plies: list = field(default_factory=list)
    predictions: list = field(default_factory=list)
    values: list = field(default_factory=list)
    #: the advantage/W/D/L target each entry was actually emitted with
    emitted_advantages: list = field(default_factory=list)
    emitted_wdl_targets: list = field(default_factory=list)
    emitted_provenance: list = field(default_factory=list)
    #: how many entries have already been emitted into a training window
    emitted: int = 0
    #: how many window boundaries this trace has been carried across
    windows_spanned: int = 0
    closed: bool = False
    outcome: "str | None" = None

    def __len__(self) -> int:
        return len(self.plies)

    @property
    def pending(self) -> int:
        """Entries collected but not yet emitted into a window."""
        return len(self.plies) - self.emitted

    def record(self, *, ply: int, wdl) -> int:
        """Append one stored decision. Returns its index in the trace."""
        if self.closed:
            raise Phase17TargetError(
                f"{self.game_id}/{self.color}: a decision arrived after the trace closed"
            )
        row = tuple(float(v) for v in np.asarray(wdl, dtype=np.float64).reshape(3))
        try:
            validate_behavior_wdl(row, where=f"{self.game_id} ply {ply}")
        except Phase9TargetError as error:
            raise Phase17TargetError(str(error)) from error
        self.plies.append(int(ply))
        self.predictions.append(row)
        self.values.append(behavior_value_scalar(row))
        return len(self.plies) - 1

    def close(self, terminal_result: str) -> str:
        """Close the trace on the game's terminal result, from this seat's view."""
        if self.closed:
            raise Phase17TargetError(f"{self.game_id}/{self.color}: already closed")
        self.outcome = terminal_outcome(terminal_result, int(self.color))
        self.closed = True
        return self.outcome

    def tail_for(self, boundary: "BoundaryTail | None") -> BoundaryTail:
        """The tail this trace closes its pending segment on."""
        if self.closed:
            if self.outcome is None:  # pragma: no cover - close() always sets it
                raise Phase17TargetError(f"{self.game_id}: closed with no outcome")
            return terminal_tail(self.outcome)
        if boundary is None:
            raise Phase17TargetError(
                f"{self.game_id}/{self.color}: an unfinished trace needs a stored "
                "boundary prediction; Phase 17 never bootstraps from a "
                "recomputed one"
            )
        return boundary

    def emit(self, boundary: "BoundaryTail | None") -> dict:
        """Close the pending segment and hand back its per-entry targets.

        Advances the emission cursor, so a transition can never be emitted
        twice: the next window starts where this one stopped.
        """
        count = self.pending
        if count == 0:
            return {"entries": 0, "rows": [], "tail": None}
        tail = self.tail_for(boundary)
        start = self.emitted
        built = segment_targets(
            self.values[start:], self.predictions[start:], tail
        )
        rows = []
        for offset in range(count):
            index = start + offset
            advantage = float(built["advantages"][offset])
            wdl_target = tuple(float(v) for v in built["wdl_targets"][offset])
            last = offset == count - 1
            rows.append(
                {
                    "index": index,
                    "ply": int(self.plies[index]),
                    "advantage_target": advantage,
                    "wdl_target": wdl_target,
                    "target_provenance": tail.provenance,
                    "boundary_status": (
                        BOUNDARY_TERMINAL
                        if last and tail.kind == TAIL_TERMINAL
                        else BOUNDARY_WINDOW
                        if last
                        else BOUNDARY_INTERIOR
                    ),
                    "bootstrap_age_windows": int(self.windows_spanned),
                }
            )
            self.emitted_advantages.append(advantage)
            self.emitted_wdl_targets.append(wdl_target)
            self.emitted_provenance.append(tail.provenance)
        self.emitted = len(self.plies)
        return {"entries": count, "rows": rows, "tail": tail.to_dict()}

    def carried(self) -> None:
        """Record that this trace survived a window boundary still open."""
        if self.closed:
            raise Phase17TargetError(
                f"{self.game_id}/{self.color}: a closed trace is not carried"
            )
        self.windows_spanned += 1

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "color": int(self.color),
            "plies": [int(p) for p in self.plies],
            "predictions": [[float(v) for v in row] for row in self.predictions],
            "values": [float(v) for v in self.values],
            "emitted_advantages": [float(v) for v in self.emitted_advantages],
            "emitted_wdl_targets": [
                [float(v) for v in row] for row in self.emitted_wdl_targets
            ],
            "emitted_provenance": list(self.emitted_provenance),
            "emitted": int(self.emitted),
            "windows_spanned": int(self.windows_spanned),
            "closed": bool(self.closed),
            "outcome": self.outcome,
        }

    @staticmethod
    def from_dict(payload: dict) -> "SeatTrace":
        trace = SeatTrace(game_id=str(payload["game_id"]), color=int(payload["color"]))
        trace.plies = [int(p) for p in payload["plies"]]
        trace.predictions = [tuple(float(v) for v in row) for row in payload["predictions"]]
        trace.values = [float(v) for v in payload["values"]]
        trace.emitted_advantages = [float(v) for v in payload["emitted_advantages"]]
        trace.emitted_wdl_targets = [
            tuple(float(v) for v in row) for row in payload["emitted_wdl_targets"]
        ]
        trace.emitted_provenance = list(payload["emitted_provenance"])
        trace.emitted = int(payload["emitted"])
        trace.windows_spanned = int(payload["windows_spanned"])
        trace.closed = bool(payload["closed"])
        trace.outcome = payload["outcome"]
        return trace


# ---------------------------------------------------------------------------
# Divergence telemetry (never a gate)
# ---------------------------------------------------------------------------


def whole_game_divergence(trace: SeatTrace) -> dict:
    """Per-row: emitted target vs the target the terminal outcome would give.

    Operator decision D2 retired the requirement that these be equal. This is
    the measurement that replaced it, and it is reported, never enforced:
    a bootstrapped row's difference from its eventual whole-game target is a
    property of TD(lambda) truncation, not a defect.
    """
    if not trace.closed or trace.outcome is None:
        raise Phase17TargetError(
            f"{trace.game_id}/{trace.color}: divergence needs a finished trace"
        )
    if len(trace.emitted_advantages) != len(trace.plies):
        raise Phase17TargetError(
            f"{trace.game_id}/{trace.color}: {len(trace.emitted_advantages)} "
            f"emitted targets for {len(trace.plies)} decisions"
        )
    if not trace.plies:
        return {"entries": 0, "rows": []}
    z = terminal_z(trace.outcome)
    whole_advantages = accepted_advantages(list(trace.values), z)
    whole_wdl = accepted_wdl_targets(list(trace.predictions), trace.outcome)
    rows = []
    for index, ply in enumerate(trace.plies):
        advantage_difference = float(
            trace.emitted_advantages[index] - whole_advantages[index]
        )
        wdl_difference = float(
            np.max(
                np.abs(
                    np.asarray(trace.emitted_wdl_targets[index], dtype=np.float64)
                    - np.asarray(whole_wdl[index], dtype=np.float64)
                )
            )
        )
        rows.append(
            {
                "index": index,
                "ply": int(ply),
                "target_provenance": trace.emitted_provenance[index],
                "advantage_emitted": float(trace.emitted_advantages[index]),
                "advantage_whole_game": float(whole_advantages[index]),
                "boundary_target_divergence": advantage_difference,
                "boundary_wdl_divergence": wdl_difference,
            }
        )
    magnitudes = [abs(row["boundary_target_divergence"]) for row in rows]
    return {
        "game_id": trace.game_id,
        "color": int(trace.color),
        "outcome": trace.outcome,
        "entries": len(rows),
        "windows_spanned": int(trace.windows_spanned),
        "max_advantage_divergence": max(magnitudes),
        "mean_advantage_divergence": float(np.mean(magnitudes)),
        "max_wdl_divergence": max(row["boundary_wdl_divergence"] for row in rows),
        "bootstrapped_rows": sum(
            1 for row in rows if row["target_provenance"] == PROVENANCE_BOOTSTRAP
        ),
        "rows": rows,
    }


def targets_semantics() -> dict:
    return {
        "targets_version": MOVE_TARGETS_VERSION,
        "lambda_A": LAMBDA_ADVANTAGE,
        "lambda_V": LAMBDA_VALUE,
        "advantage": (
            "A_t = delta_t + lambda_A * A_{t+1}; the tail is closed by the "
            "terminal z (finished) or the stored boundary value (unfinished), "
            "and A beyond the tail is 0"
        ),
        "wdl": (
            "Y_t = (1-lambda_V) P_{t+1} + lambda_V Y_{t+1}; the tail is the "
            "one-hot outcome (finished) or the stored boundary W/D/L "
            "prediction in both slots (unfinished)"
        ),
        "bootstrap_source": (
            "the STORED boundary prediction; never a later-known outcome and "
            "never a prediction recomputed at training time"
        ),
        "governing_invariant": "G-M4a, reduction to the accepted whole-game walk",
        "retired_invariant": "G-M4b, by operator decision D2 (2026-08-27)",
        "divergence": "measured per row and reported; never a gate",
        "emission": "every collected transition, exactly once",
    }


__all__ = [
    "TAIL_BOOTSTRAP",
    "TAIL_TERMINAL",
    "BoundaryTail",
    "Phase17TargetError",
    "SeatTrace",
    "advantages_with_tail",
    "bootstrap_tail",
    "deltas_with_tail",
    "reduction_invariant",
    "segment_targets",
    "targets_semantics",
    "terminal_tail",
    "wdl_targets_with_tail",
    "whole_game_divergence",
]
