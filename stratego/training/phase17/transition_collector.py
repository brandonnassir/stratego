"""Phase 17 Agent 2: the true fixed-transition window collector.

Specification sources: common contract sections 5 and 6, Agent 2 instruction
sections 3, 4, 5 and 7.

Three things are different from Phase 16, and only three
--------------------------------------------------------
```text
1  both seats are the current RAW move policy, resolved per decision through
   the live cell -- there is no participant mixture and no per-runner copy
2  the window closes on an exact transition count, not on "close to a budget";
   every collected transition is emitted for training in the same window
3  an open trace closes on a STORED boundary prediction taken at the boundary,
   for both seats, rather than waiting for a terminal outcome
```

Everything else -- the engine, the observation, legality, the trajectory
builder, the sealed-record validation, the batched lockstep loop -- is the
accepted Phase 9 machinery, imported and subclassed rather than rewritten.

Why the pending request is cleared at a boundary
------------------------------------------------
`GameRunner.advance()` builds a `NeuralRequest` that captures the acting
snapshot. If a window ended while a request was in flight and the model then
updated, applying that request in the next window would record a decision under
the *previous* digest -- the exact defect this phase exists to fix, reappearing
one ply later. So the boundary drops any un-applied request. The engine state
is untouched, and the next window's `advance()` rebuilds the request against
the current cell. `apply_neural` additionally refuses a request whose snapshot
is not the cell's current object, which makes a stale application impossible
rather than merely unlikely.

Exactly the budget
------------------
The batch that would overshoot the budget is truncated, not skipped: the games
left over simply keep their place and are decided first in the next window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ...engine.constants import PLAYER_NAMES, PLAYERS
from ...engine.observation import build_observation
from ...engine.transition import apply_action
from ...model.action_frame import absolute_action_to_model
from ..phase9_behavior import behavior_distribution, evaluate_observations
from ..phase9_collector import GameRunner
from ..phase9_targets import behavior_action_logprob
from ..serialization import to_float32
from ..trajectory import DEFAULT_SNAPSHOT_INTERVAL
from .move_contract import (
    CURRENT_POLICY_IDENTITY,
    CURRENT_POLICY_TOKEN,
    DOMAIN_SETUP_DRAW,
    MOVE_COLLECTOR_VERSION,
    WINDOW_TRANSITIONS,
    Phase17MoveError,
    derive_move_seed,
    game_id as make_game_id,
    parse_game_id,
    require_run_id,
)
from .move_snapshot import (
    CurrentMovePolicy,
    Phase17Seating,
    sample_legal_action,
)
from .transition_schema import MoveTransition, validate_transition
from .transition_targets import (
    BoundaryTail,
    SeatTrace,
    bootstrap_tail,
    whole_game_divergence,
)


class Phase17CollectorError(Phase17MoveError):
    """A Phase 17 window could not be collected as specified."""


# ---------------------------------------------------------------------------
# The scheduled-game adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduledPhase17Game:
    """A Phase 17 game in the shape the accepted Phase 9 runner reads.

    `learner_color is None` is what makes `acting_snapshot_for` return the
    current behavior snapshot for *both* seats -- the accepted resolution
    point, told the truth about a population in which both sides are learners,
    rather than a fork of it.
    """

    game_id: str
    setup_root_seed: int

    @property
    def phase9_game_id(self) -> str:
        return self.game_id

    @property
    def rollout_game_id(self) -> str:
        return self.game_id

    @property
    def behavior_snapshot_identity(self) -> str:
        return CURRENT_POLICY_IDENTITY

    @property
    def behavior_policy_token(self) -> str:
        return CURRENT_POLICY_TOKEN

    @property
    def learner_control(self) -> str:
        return "both_seats"

    @property
    def learner_color(self) -> None:
        """Both seats are learners; the accepted resolver reads `None` as such."""
        return None

    @property
    def learner_sides(self) -> tuple:
        return tuple(PLAYER_NAMES[player] for player in PLAYERS)

    @property
    def opponent_kind(self) -> str:
        return "current_policy"

    @property
    def opponent_identity(self) -> str:
        return CURRENT_POLICY_IDENTITY

    @property
    def historical_snapshot_identity(self) -> None:
        return None

    @property
    def red_policy_identity(self) -> str:
        return CURRENT_POLICY_TOKEN

    @property
    def blue_policy_identity(self) -> str:
        return CURRENT_POLICY_TOKEN

    @property
    def red_policy_seed(self) -> None:
        return None

    @property
    def blue_policy_seed(self) -> None:
        return None


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------


class Phase17GameRunner(GameRunner):
    """The accepted Phase 9 runner, harvesting `phase17_move_transition_v1` rows.

    One override. `apply_neural` samples the action from Phase 17's stream,
    takes the transition while the live state is still in hand, and refuses a
    request whose snapshot is not the current one. Nothing about how a game is
    played, stored or validated changes.
    """

    def __init__(self, *args, run_id: str, cell: CurrentMovePolicy, **kwargs) -> None:
        self.run_id = require_run_id(run_id)
        self.cell = cell
        self.rows: list = []
        self.traces: dict = {}
        self.row_index_by_key: dict = {}
        self.rule_decision_count = 0
        super().__init__(*args, **kwargs)

    # -- traces ------------------------------------------------------------

    def trace_for(self, player: int) -> SeatTrace:
        trace = self.traces.get(int(player))
        if trace is None:
            trace = SeatTrace(game_id=self.game_id, color=int(player))
            self.traces[int(player)] = trace
        return trace

    def open_traces(self) -> list:
        return [trace for trace in self.traces.values() if not trace.closed]

    # -- the one override --------------------------------------------------

    def apply_neural(self, policy_logits_row, wdl_row) -> None:
        request = self.pending
        if request is None:  # pragma: no cover - the caller pairs these
            raise Phase17CollectorError(f"{self.game_id}: no pending neural decision")
        if request.snapshot is not self.cell.snapshot:
            raise Phase17CollectorError(
                f"{self.game_id} ply {request.ply}: the pending decision was "
                f"prepared under model state {request.snapshot.checkpoint_sha256} "
                f"but the current raw policy is {self.cell.digest}; a decision "
                "may never be recorded under a stale snapshot"
            )
        legality = request.legality
        actor = int(legality.acting_player)
        probabilities = behavior_distribution(policy_logits_row, legality)
        legal = tuple(int(action) for action in legality.absolute)
        draw = sample_legal_action(probabilities, legal, self.game_id, request.ply)
        selected = int(draw["action"])
        if selected not in legal:  # pragma: no cover - selection is an index
            raise Phase17CollectorError(
                f"{self.game_id} ply {request.ply}: sampled action {selected} is not legal"
            )
        expected_token = self._side_token(actor)
        if request.snapshot.policy_token != expected_token:
            raise Phase17CollectorError(
                f"{self.game_id} ply {request.ply}: acting snapshot token "
                f"{request.snapshot.policy_token!r} is not the scheduled "
                f"{expected_token!r}"
            )
        wdl = tuple(to_float32(float(v)) for v in np.asarray(wdl_row).reshape(3))
        stored = tuple(to_float32(value) for value in probabilities)
        probability = float(stored[int(draw["index"])])

        index = self.trace_for(actor).record(ply=int(request.ply), wdl=wdl)
        row = MoveTransition(
            run_id=self.run_id,
            iteration=int(self.cell.iteration),
            window_index=int(self.cell.iteration),
            game_id=self.game_id,
            ply=int(request.ply),
            color=actor,
            perspective_player=actor,
            observation=np.ascontiguousarray(request.observation, dtype=np.float32),
            legal_mask=np.asarray(legality.model_mask, dtype=bool),
            legal_actions=legal,
            behavior_probabilities=stored,
            sampled_action=selected,
            sampled_action_index=int(draw["index"]),
            sampled_action_model=int(absolute_action_to_model(selected, actor)),
            behavior_action_probability=probability,
            behavior_action_logprob=behavior_action_logprob(probability),
            action_seed=int(draw["seed"]),
            behavior_model_state_digest=request.snapshot.checkpoint_sha256,
            behavior_snapshot_iteration=int(self.cell.iteration),
            stored_value_scalar=float(wdl[0]) - float(wdl[2]),
            stored_wdl=wdl,
        )
        self.rows.append(row)
        self.row_index_by_key[(actor, index)] = len(self.rows) - 1

        self.builder.record_decision(
            self.state,
            legal_action_ids=legal,
            probabilities=stored,
            win_draw_loss_prediction=wdl,
            selected_action_id=selected,
            collection_policy_version=expected_token,
        )
        self.neural_decision_count += 1
        self.learner_decision_count += 1
        self.learner_neural_decision_count += 1
        self.pending = None
        apply_action(self.state, selected, legal=list(legal))

    def _rule_decision(self, legal, mask) -> None:  # pragma: no cover - unreachable
        self.rule_decision_count += 1
        raise Phase17CollectorError(
            f"{self.game_id}: a rule/stress decision was reached in a "
            "100% current-policy population"
        )

    # -- boundary ----------------------------------------------------------

    def drop_pending(self) -> bool:
        """Discard an un-applied request so it cannot be applied under new weights."""
        if self.pending is None:
            return False
        self.pending = None
        return True

    def boundary_predictions(self, cell: CurrentMovePolicy) -> dict:
        """One stored bootstrap prediction per open trace, at the live state.

        The observation is built from each seat's own perspective at the
        boundary state -- the accepted `build_observation`, which takes an
        explicit observer -- and evaluated under the raw snapshot that has just
        finished generating this window's decisions. The result is *stored*; a
        resume reads it back rather than recomputing it.
        """
        open_traces = [
            trace for trace in self.traces.values() if not trace.closed and trace.pending
        ]
        if not open_traces or self.state.terminal:
            return {}
        snapshot = cell.snapshot
        observations = np.stack(
            [
                np.ascontiguousarray(
                    build_observation(self.state, int(trace.color)), dtype=np.float32
                )
                for trace in open_traces
            ]
        )
        _logits, wdl = evaluate_observations(snapshot, observations)
        tails = {}
        for position, trace in enumerate(open_traces):
            row = tuple(
                to_float32(float(value))
                for value in np.asarray(wdl[position]).reshape(3)
            )
            tails[int(trace.color)] = bootstrap_tail(
                row, model_state_digest=snapshot.checkpoint_sha256
            )
        return tails

    def close_traces(self) -> None:
        """Close every open trace on the sealed terminal result."""
        if self.record is None:
            raise Phase17CollectorError(f"{self.game_id}: the game has not finished")
        for trace in self.traces.values():
            if not trace.closed:
                trace.close(self.record.terminal_result)


# ---------------------------------------------------------------------------
# One window's result
# ---------------------------------------------------------------------------


@dataclass
class WindowResult:
    """One closed window: its rows, and everything the telemetry row needs."""

    iteration: int
    budget: int
    rows: list = field(default_factory=list)
    transitions_harvested: int = 0
    games_started: int = 0
    games_finished: int = 0
    active_games: int = 0
    boundary_rows: int = 0
    terminal_rows: int = 0
    carried_traces: int = 0
    plies_advanced: int = 0
    game_lengths: list = field(default_factory=list)
    terminal_results: dict = field(default_factory=dict)
    terminal_reasons: dict = field(default_factory=dict)
    divergence: list = field(default_factory=list)
    participant_digests: dict = field(default_factory=dict)
    rule_decisions: int = 0
    dropped_pending: int = 0
    sealed_at_boundary: int = 0
    seconds: float = 0.0
    stopped_early: bool = False

    @property
    def transitions_per_second(self) -> float:
        return self.transitions_harvested / self.seconds if self.seconds > 0 else 0.0

    def divergence_summary(self) -> dict:
        """Boundary-target divergence over the games that finished this window."""
        rows = [entry for report in self.divergence for entry in report["rows"]]
        bootstrapped = [
            entry
            for entry in rows
            if entry["target_provenance"] == "boundary_bootstrap"
        ]
        magnitudes = [abs(entry["boundary_target_divergence"]) for entry in bootstrapped]
        wdl = [entry["boundary_wdl_divergence"] for entry in bootstrapped]
        ages = [report["windows_spanned"] for report in self.divergence]
        return {
            "traces_closed": len(self.divergence),
            "rows_compared": len(rows),
            "bootstrapped_rows": len(bootstrapped),
            "max_advantage_divergence": max(magnitudes) if magnitudes else 0.0,
            "mean_advantage_divergence": (
                float(np.mean(magnitudes)) if magnitudes else 0.0
            ),
            "max_wdl_divergence": max(wdl) if wdl else 0.0,
            "mean_wdl_divergence": float(np.mean(wdl)) if wdl else 0.0,
            "max_windows_spanned": max(ages) if ages else 0,
            "is_a_gate": False,
        }

    def summary(self) -> dict:
        return {
            "collector_version": MOVE_COLLECTOR_VERSION,
            "iteration": int(self.iteration),
            "budget": int(self.budget),
            "transitions_harvested": int(self.transitions_harvested),
            "transitions_emitted": len(self.rows),
            "exact_budget": len(self.rows) == int(self.budget),
            "boundary_rows": int(self.boundary_rows),
            "terminal_rows": int(self.terminal_rows),
            "carried_traces": int(self.carried_traces),
            "games_started": int(self.games_started),
            "games_finished": int(self.games_finished),
            "active_games": int(self.active_games),
            "plies_advanced": int(self.plies_advanced),
            "mean_game_length": (
                float(np.mean(self.game_lengths)) if self.game_lengths else 0.0
            ),
            "terminal_results": dict(self.terminal_results),
            "terminal_reasons": dict(self.terminal_reasons),
            "participant_digests": dict(self.participant_digests),
            "rule_decisions": int(self.rule_decisions),
            "dropped_pending_requests": int(self.dropped_pending),
            "sealed_at_boundary": int(self.sealed_at_boundary),
            "seconds": float(self.seconds),
            "transitions_per_second": self.transitions_per_second,
            "stopped_early": bool(self.stopped_early),
            "boundary_target_divergence": self.divergence_summary(),
        }


# ---------------------------------------------------------------------------
# The window collector
# ---------------------------------------------------------------------------


class FixedTransitionCollector:
    """A persistent population advanced until exactly the budget is collected.

    The population is created once and outlives every window: a game that does
    not finish keeps its place, and only games that *end* are replaced. Unlike
    Phase 16 the rows do not wait for the game -- every transition collected in
    a window is emitted in that window, with its trace closed on the terminal
    outcome if the game ended and on the stored boundary prediction otherwise.
    """

    def __init__(
        self,
        *,
        run_id: str,
        cell: CurrentMovePolicy,
        setup_provider,
        population: int,
        budget: int = WINDOW_TRANSITIONS,
        snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
    ) -> None:
        self.run_id = require_run_id(run_id)
        if not isinstance(cell, CurrentMovePolicy):
            raise Phase17CollectorError(
                f"the collector needs a CurrentMovePolicy, got {type(cell).__name__}"
            )
        if setup_provider is None:
            raise Phase17CollectorError(
                "a Phase 17 collector needs an explicit setup provider; there is "
                "no frozen setup library in Phase 17 training and no silent "
                "library fallback"
            )
        if not isinstance(population, int) or isinstance(population, bool) or population < 1:
            raise Phase17CollectorError(
                f"population must be an int >= 1, got {population!r}"
            )
        self.cell = cell
        self.seating = Phase17Seating(cell)
        self.setup_provider = setup_provider
        self.budget = int(budget)
        self.snapshot_interval = int(snapshot_interval)
        self.slots: list = [None] * int(population)
        self.draw_counts: list = [0] * int(population)
        self.ply_marks: list = [0] * int(population)
        self.iteration = 0
        self.games_completed = 0
        self.transitions_collected = 0
        self.transitions_emitted = 0
        self.plies_advanced = 0
        #: every model-state digest any stored transition was produced under
        self.observed_digests: dict = {}

    # -- population --------------------------------------------------------

    def _start(self, slot: int) -> Phase17GameRunner:
        draw = self.draw_counts[slot]
        identifier = make_game_id(self.run_id, slot, draw)
        self.draw_counts[slot] += 1
        scheduled = ScheduledPhase17Game(
            game_id=identifier,
            setup_root_seed=derive_move_seed(DOMAIN_SETUP_DRAW, identifier),
        )
        return Phase17GameRunner(
            scheduled,
            self.seating,
            run_id=self.run_id,
            cell=self.cell,
            setup_source=self.setup_provider,
            behavior_checkpoint_sha256=self.cell.digest,
            snapshot_interval=self.snapshot_interval,
            observer_probe_plies=0,
        )

    def fill(self) -> int:
        """Seat a fresh game in every empty slot. Returns how many were started."""
        started = 0
        for slot in range(len(self.slots)):
            if self.slots[slot] is None:
                self.slots[slot] = self._start(slot)
                self.ply_marks[slot] = 0
                started += 1
        return started

    def active_runners(self) -> list:
        return [runner for runner in self.slots if runner is not None]

    def in_flight_transitions(self) -> int:
        return sum(len(runner.rows) for runner in self.active_runners())

    # -- one window --------------------------------------------------------

    def collect_window(
        self,
        *,
        budget: "int | None" = None,
        should_continue=None,
    ) -> WindowResult:
        """Advance the population until exactly `budget` transitions exist.

        Both seats are learners, so every legal model decision is a learner
        transition and the budget counts decisions directly.
        """
        self.iteration += 1
        target = int(self.budget if budget is None else budget)
        if target < 1:
            raise Phase17CollectorError(f"a window budget must be >= 1, got {target}")
        result = WindowResult(iteration=self.iteration, budget=target)
        started = time.perf_counter()
        result.games_started += self.fill()

        collected = 0
        finished_runners: list = []
        while collected < target:
            if should_continue is not None and not should_continue():
                result.stopped_early = True
                break
            active = self.active_runners()
            if not active:  # pragma: no cover - fill() always seats the slots
                break

            pending = []
            finished = []
            for runner in active:
                request = runner.advance()
                if request is None:
                    finished.append(runner)
                else:
                    pending.append(request)

            if pending:
                collected += self._evaluate(pending, room=target - collected)
            for runner in finished:
                self._retire(runner, result, finished_runners)
            if finished:
                result.games_started += self.fill()
            if not pending and not finished:  # pragma: no cover - defensive
                raise Phase17CollectorError(
                    "the population made no progress; neither a decision nor a "
                    "finished game in one pass"
                )

        result.transitions_harvested = collected
        self._close_window(result, finished_runners)
        result.seconds = time.perf_counter() - started
        self.transitions_collected += collected
        return result

    def _evaluate(self, pending, *, room: int) -> int:
        """One lockstep batch of neural decisions, capped by the room left.

        The cap is what makes the window *exact*: a batch that would overshoot
        is truncated, and the games left over keep their pending state and are
        decided first in the next window.
        """
        if room <= 0:  # pragma: no cover - the caller checks
            return 0
        applied = 0
        grouped: dict = {}
        for request in pending:
            grouped.setdefault(request.snapshot.checkpoint_sha256, []).append(request)
        if len(grouped) > 1:
            raise Phase17CollectorError(
                "one Phase 17 window resolved more than one acting model state: "
                f"{sorted(grouped)}"
            )
        for requests in grouped.values():
            shape = int(requests[0].snapshot.inference_batch_shape)
            cursor = 0
            while cursor < len(requests) and applied < room:
                batch = requests[cursor : cursor + shape]
                batch = batch[: room - applied]
                observations = np.stack([request.observation for request in batch])
                policy_logits, wdl = evaluate_observations(
                    batch[0].snapshot, observations
                )
                for position, request in enumerate(batch):
                    request.runner.apply_neural(policy_logits[position], wdl[position])
                applied += len(batch)
                cursor += len(batch)
        return applied

    def _seal_terminal_slots(self, result: WindowResult, finished: list) -> None:
        """Seal a game whose last decision ended it just as the budget ran out.

        The budget can be reached by the very action that terminates a game, in
        which case the loop exits before the runner is advanced again and the
        slot still holds a finished position. Such a trace must close on the
        real outcome -- bootstrapping a terminal state would invent a
        continuation that does not exist -- so the seal happens here, at the
        window close, before anything is emitted.
        """
        for runner in list(self.slots):
            if runner is None or runner.record is not None:
                continue
            if runner.pending is not None or not runner.state.terminal:
                continue
            if runner.advance() is not None:  # pragma: no cover - terminal is terminal
                raise Phase17CollectorError(
                    f"{runner.game_id}: a terminal state produced a pending decision"
                )
            result.sealed_at_boundary += 1
            self._retire(runner, result, finished)

    def _retire(self, runner: Phase17GameRunner, result: WindowResult, finished: list) -> None:
        """Close one finished game's traces and free its slot.

        The rows are *not* emitted here: emission happens once, at the window
        close, so a finished game and a still-running one are treated alike.
        """
        runner.close_traces()
        record = runner.record
        result.games_finished += 1
        result.game_lengths.append(int(record.final_ply))
        result.terminal_results[record.terminal_result] = (
            result.terminal_results.get(record.terminal_result, 0) + 1
        )
        result.terminal_reasons[record.terminal_reason] = (
            result.terminal_reasons.get(record.terminal_reason, 0) + 1
        )
        self.games_completed += 1
        for slot, seated in enumerate(self.slots):
            if seated is runner:
                result.plies_advanced += int(record.final_ply) - self.ply_marks[slot]
                self.ply_marks[slot] = 0
                self.slots[slot] = None
                break
        finished.append(runner)

    def _close_window(self, result: WindowResult, finished_runners: list) -> None:
        """Emit every collected transition, and carry what is still open."""
        self._seal_terminal_slots(result, finished_runners)
        for runner in finished_runners:
            self._emit(runner, result, tails={})

        for slot, runner in enumerate(self.slots):
            if runner is None:
                continue
            if runner.drop_pending():
                result.dropped_pending += 1
            total = int(runner.state.total_moves)
            result.plies_advanced += total - self.ply_marks[slot]
            self.ply_marks[slot] = total
            tails = runner.boundary_predictions(self.cell)
            self._emit(runner, result, tails=tails)
            for trace in runner.open_traces():
                trace.carried()
                result.carried_traces += 1

        result.active_games = len(self.active_runners())
        result.rows.sort(key=lambda row: (row.game_id, row.color, row.ply))
        self.transitions_emitted += len(result.rows)
        self.plies_advanced += result.plies_advanced
        if len(result.rows) != result.transitions_harvested:
            raise Phase17CollectorError(
                f"window {result.iteration} harvested "
                f"{result.transitions_harvested} transitions but emitted "
                f"{len(result.rows)}; partial emission must emit every one"
            )

    def _emit(self, runner: Phase17GameRunner, result: WindowResult, *, tails: dict) -> None:
        """Fill one runner's pending rows with their targets and hand them over."""
        for color, trace in sorted(runner.traces.items()):
            if not trace.pending:
                continue
            tail: "BoundaryTail | None" = tails.get(int(color))
            emission = trace.emit(tail)
            for entry in emission["rows"]:
                row = runner.rows[runner.row_index_by_key[(int(color), entry["index"])]]
                row.advantage_target = float(entry["advantage_target"])
                row.wdl_target = tuple(float(v) for v in entry["wdl_target"])
                row.target_provenance = entry["target_provenance"]
                row.boundary_status = entry["boundary_status"]
                row.bootstrap_age_windows = int(entry["bootstrap_age_windows"])
                row.iteration = int(result.iteration)
                row.window_index = int(result.iteration)
                row.policy_age_iterations = max(
                    0, int(result.iteration) - int(row.behavior_snapshot_iteration)
                )
                validate_transition(row, where=f"{row.game_id} ply {row.ply}")
                self.observed_digests[row.behavior_model_state_digest] = (
                    self.observed_digests.get(row.behavior_model_state_digest, 0) + 1
                )
                result.participant_digests[row.behavior_model_state_digest] = (
                    result.participant_digests.get(row.behavior_model_state_digest, 0) + 1
                )
                if row.target_provenance == "boundary_bootstrap":
                    result.boundary_rows += 1
                else:
                    result.terminal_rows += 1
                result.rows.append(row)
        result.rule_decisions += int(runner.rule_decision_count)

        if runner.record is not None:
            for trace in runner.traces.values():
                if trace.closed and len(trace.emitted_advantages) == len(trace.plies):
                    report = whole_game_divergence(trace)
                    result.divergence.append(report)
                    for entry in report["rows"]:
                        index = runner.row_index_by_key.get((int(trace.color), entry["index"]))
                        if index is None:  # pragma: no cover - keys are complete
                            continue
                        row = runner.rows[index]
                        row.boundary_target_divergence = float(
                            entry["boundary_target_divergence"]
                        )
                        row.boundary_wdl_divergence = float(
                            entry["boundary_wdl_divergence"]
                        )

    # -- the runtime participant ledger -----------------------------------

    def participant_ledger(self) -> dict:
        """Proof that only the current raw policy acted during collection.

        Every stored transition's `behavior_model_state_digest` must be one the
        live cell has actually held. A digest from anywhere else means a
        non-current participant produced a training decision -- immediate stop
        condition I5.
        """
        known = set(self.cell.known_digests())
        unknown = {
            digest: count
            for digest, count in self.observed_digests.items()
            if digest not in known
        }
        rule_decisions = sum(
            runner.rule_decision_count for runner in self.active_runners()
        )
        return {
            "collector_version": MOVE_COLLECTOR_VERSION,
            "policy_token": CURRENT_POLICY_TOKEN,
            "seats": {"red": CURRENT_POLICY_TOKEN, "blue": CURRENT_POLICY_TOKEN},
            "distinct_acting_model_states": len(self.observed_digests),
            "transitions_by_model_state": dict(self.observed_digests),
            "cell_digest_history": self.cell.digest_history(),
            "unknown_model_states": unknown,
            "rule_or_stress_decisions": int(rule_decisions),
            "historical_participants": 0,
            "search_participants": 0,
            "holds": not unknown and rule_decisions == 0,
        }

    # -- persistence -------------------------------------------------------

    def state(self) -> dict:
        """The window-boundary carry state, with no rows and no observations."""
        traces = []
        seated = []
        for slot, runner in enumerate(self.slots):
            if runner is None:
                continue
            seated.append(
                {
                    "slot": int(slot),
                    "game_id": runner.game_id,
                    "draw": int(parse_game_id(runner.game_id)["draw"]),
                    "total_moves": int(runner.state.total_moves),
                }
            )
            for color, trace in sorted(runner.traces.items()):
                traces.append({"slot": int(slot), **trace.to_dict()})
        return {
            "collector_version": MOVE_COLLECTOR_VERSION,
            "run_id": self.run_id,
            "iteration": int(self.iteration),
            "population": len(self.slots),
            "budget": int(self.budget),
            "draw_counts": list(self.draw_counts),
            "ply_marks": list(self.ply_marks),
            "games_completed": int(self.games_completed),
            "transitions_collected": int(self.transitions_collected),
            "transitions_emitted": int(self.transitions_emitted),
            "plies_advanced": int(self.plies_advanced),
            "observed_digests": dict(self.observed_digests),
            "current_policy": self.cell.to_dict(),
            "in_flight_games": len(self.active_runners()),
            "seated": seated,
            "carry": traces,
            "note": (
                "engine states are NOT here: common contract section 10 makes "
                "exact active-game persistence Agent 4's paired-checkpoint "
                "responsibility. This is the move half's carry state."
            ),
        }

    def restore_counters(self, payload: dict) -> None:
        """Restore the counters a resumed collector continues from."""
        if payload.get("run_id") != self.run_id:
            raise Phase17CollectorError(
                f"carry state belongs to run {payload.get('run_id')!r}, not "
                f"{self.run_id!r}"
            )
        if int(payload["population"]) != len(self.slots):
            raise Phase17CollectorError(
                f"carry state has {payload['population']} slots, this collector "
                f"has {len(self.slots)}"
            )
        self.iteration = int(payload["iteration"])
        self.draw_counts = [int(value) for value in payload["draw_counts"]]
        self.ply_marks = [int(value) for value in payload["ply_marks"]]
        self.games_completed = int(payload["games_completed"])
        self.transitions_collected = int(payload["transitions_collected"])
        self.transitions_emitted = int(payload["transitions_emitted"])
        self.plies_advanced = int(payload["plies_advanced"])
        self.observed_digests = {
            str(key): int(value) for key, value in payload["observed_digests"].items()
        }

    def restore_seating(self, payload: dict) -> int:
        """Point the empty slots at the game ids the carry state names.

        Sets each slot's draw counter so the next :meth:`fill` seats the same
        logical game the interrupted run held. It does **not** replay the
        moves: common contract section 10 makes exact active-game persistence
        Agent 4's paired-checkpoint responsibility, and this is the move half's
        half of that -- the identities, so the traces have something to attach
        to and so a resumed draw sequence does not skip a game.
        """
        seated = payload.get("seated")
        if not seated:
            raise Phase17CollectorError(
                "the carry state names no seated games; a resumed population "
                "cannot be reconstructed from trace data alone"
            )
        for entry in seated:
            slot = int(entry["slot"])
            if not 0 <= slot < len(self.slots):
                raise Phase17CollectorError(
                    f"carry state names slot {slot}, outside 0..{len(self.slots) - 1}"
                )
            if self.slots[slot] is not None:
                raise Phase17CollectorError(
                    f"slot {slot} is already seated; restore into an empty population"
                )
            self.draw_counts[slot] = int(entry["draw"])
        return len(seated)

    def restore_traces(self, payload: dict) -> int:
        """Re-attach the carried seat traces to the seated runners.

        Returns how many traces were restored. The engine states themselves are
        Agent 4's to persist; this restores the target-side carry so a resumed
        window neither duplicates nor omits a transition.
        """
        restored = 0
        by_slot: dict = {}
        for entry in payload.get("carry", []):
            by_slot.setdefault(int(entry["slot"]), []).append(entry)
        for slot, entries in by_slot.items():
            runner = self.slots[slot] if slot < len(self.slots) else None
            if runner is None:
                raise Phase17CollectorError(
                    f"carry state names slot {slot}, which has no seated game"
                )
            for entry in entries:
                trace = SeatTrace.from_dict(entry)
                if trace.game_id != runner.game_id:
                    raise Phase17CollectorError(
                        f"slot {slot} carries {trace.game_id!r} but is seated with "
                        f"{runner.game_id!r}"
                    )
                runner.traces[int(trace.color)] = trace
                restored += 1
        return restored


def collector_semantics() -> dict:
    return {
        "collector_version": MOVE_COLLECTOR_VERSION,
        "population": "persistent; only finished games are replaced",
        "seats": "Red and Blue are the same current raw move snapshot",
        "resolution": "per decision, through the live cell; no per-runner copy",
        "budget": "exactly the configured transition count per window",
        "emission": "every transition collected in a window is emitted in it",
        "boundary": (
            "an un-applied neural request is dropped so it cannot be applied "
            "under new weights; the engine state is untouched"
        ),
        "bootstrap": "one stored boundary prediction per open trace, both seats",
        "search": "not imported and not reachable",
        "rule_and_stress": "structurally refused; `_rule_decision` raises",
    }


__all__ = [
    "FixedTransitionCollector",
    "Phase17CollectorError",
    "Phase17GameRunner",
    "ScheduledPhase17Game",
    "WindowResult",
    "collector_semantics",
]
