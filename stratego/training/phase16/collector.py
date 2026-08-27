"""Phase 16 Agent 3: the window collector.

Specification source: `03_AGENT_3_TRAINING_LOOP_V2.md` section 2.1.

What a Phase 16 iteration is
----------------------------
A fixed budget of **learner decisions**, not a fixed number of games. A
persistent population of games is advanced until the budget is met; training
then runs on what the window produced and the *same* games continue afterwards.
Iteration wall-time stops depending on game length, which is the failure that
took Phase 14's iterations from 24 to 138 minutes.

Reuse, not reimplementation
---------------------------
The game loop, the batching topology, the observer-safety boundary and the
trajectory builder are the **accepted Phase 9** ones, imported and subclassed.
Three things are Phase 16's own and all three are required:

1. the action-sampling stream, which descends from the Phase 16 roots;
2. the game-id scheme, so a Phase 16 game can never be mistaken for a Phase 9,
   10B or 14 rollout;
3. example harvesting *at collection time* -- the observation, the model-frame
   legal mask and the belief label are taken from the live state as the
   decision is made, so no window ever has to replay a game to build a batch.

Point 3 is the one real departure from the accepted two-pass design, and it is
forced: Phase 9's pass 2 replays a *sealed* rollout, and a window has no sealed
rollout. Everything the replay would have cross-checked is instead checked
here at the moment it is produced, against the same engine products.

Nothing here optimizes anything: there is no optimizer, no loss, no gradient
and no PPO in this module. There is also no search -- not as an option, not
behind a flag.

Buffer per game, emit on finish
-------------------------------
A game's harvested rows stay in a per-game buffer until the game ends, then the
whole game is emitted with exact whole-game targets. That is what section 2.2's
"buffer per game until then" asks for, and it is also what the accepted
objective requires: `phase9_batch_loss` averages its value and belief terms
over every row and has no per-row loss mask, so a row whose W/D/L target is not
yet knowable cannot be in a batch at all. The boundary-bootstrapped advantage
path exists in `targets.py` and is tested; it is not the production path, and
`known_limitations` says so.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ...engine.legal_moves import legal_action_mask, legal_actions
from ...engine.observation import build_observation
from ...engine.transition import apply_action
from ...model.action_frame import absolute_action_to_model
from ..belief_targets import dense_belief_target
from ..phase9_behavior import (
    Phase9BehaviorError,
    behavior_distribution,
    evaluate_observations,
)
from ..phase9_collector import (
    GameRunner,
    IterationParticipants,
    Phase9CollectorError,
    acting_snapshot_for,
)
from ..phase9_targets import (
    behavior_action_logprob,
    behavior_action_probability as stored_action_probability,
)
from ..serialization import to_float32
from ..trajectory import DEFAULT_SNAPSHOT_INTERVAL
from .contract import (
    DOMAIN_ACTION_SAMPLING,
    ArmConfig,
    Phase16TrainingError,
    derive_train_seed,
    uniform_from_seed,
)
from .population import (
    KIND_CURRENT,
    KIND_HANDCRAFTED,
    KIND_HISTORICAL,
    HistoricalPool,
    SlotDraw,
    draw_for_slot,
    player_index,
)
from .snapshots import behavior_token
from .targets import LearnerTrack, track_targets

PHASE16_COLLECTOR_IMPL = "phase16_window_collector_v1"


class Phase16CollectorError(Phase16TrainingError):
    """A Phase 16 window could not be collected as specified."""


# ---------------------------------------------------------------------------
# The Phase 16 action-sampling stream
# ---------------------------------------------------------------------------


def action_sampling_uniform(game_id: str, ply: int) -> float:
    """The uniform that chooses one Phase 16 action, from its own domain."""
    return uniform_from_seed(derive_train_seed(DOMAIN_ACTION_SAMPLING, game_id, int(ply)))


def select_action(probabilities, legal_absolute, game_id: str, ply: int) -> int:
    """The frozen cumulative-walk draw, on the Phase 16 uniform.

    Identical in rule to the accepted Phase 9 sampler -- walk ascending,
    accumulate, take the first action whose cumulative mass reaches the
    uniform, and let a float32 tail shortfall take the last legal action.
    """
    actions = tuple(int(action) for action in legal_absolute)
    if len(actions) != len(probabilities):
        raise Phase9BehaviorError(
            f"{len(probabilities)} probabilities for {len(actions)} legal actions"
        )
    if list(actions) != sorted(actions):
        raise Phase9BehaviorError("the legal action list is not ascending")
    uniform = action_sampling_uniform(game_id, ply)
    cumulative = 0.0
    for action, probability in zip(actions, probabilities):
        cumulative += float(probability)
        if cumulative >= uniform:
            return int(action)
    return int(actions[-1])


# ---------------------------------------------------------------------------
# One harvested decision
# ---------------------------------------------------------------------------


@dataclass
class HarvestedRow:
    """One learner decision, complete except for its targets.

    Field names match `Phase9RLExample` wherever the objective reads them, so
    the batch builder below is the accepted collation with two fields it fills
    itself rather than a second example schema.
    """

    observation: np.ndarray
    legal_mask: np.ndarray
    sampled_action_model: int
    sampled_action_abs: int
    behavior_action_probability: float
    behavior_action_logprob: float
    behavior_legal_actions: tuple
    behavior_legal_probabilities: tuple
    belief_target: np.ndarray
    belief_mask: np.ndarray
    game_id: str
    decision_index: int
    learner_side: int
    #: filled when the game finishes and the window closes
    advantage: float = 0.0
    standardized_advantage: float = 0.0
    ppo_eligible: bool = False
    wdl_target: tuple = (0.0, 1.0, 0.0)

    def nbytes(self) -> int:
        return int(
            self.observation.nbytes
            + self.legal_mask.nbytes
            + self.belief_target.nbytes
            + self.belief_mask.nbytes
        )


# ---------------------------------------------------------------------------
# The scheduled-game adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduledPhase16Game:
    """A `SlotDraw` in the shape the accepted Phase 9 runner reads.

    The accepted runner resolves "whose network is this move?" through
    attribute names it fixed in Phase 9. Rather than fork the runner to rename
    them, the Phase 16 draw is presented under those names; every value is the
    draw's own.
    """

    draw: SlotDraw
    behavior_identity: str

    @property
    def behavior_policy_token(self) -> str:
        """The token the learner's decisions are stored under."""
        return behavior_token(self.behavior_identity)

    @property
    def opponent_policy_token(self) -> str:
        """The token the opponent's decisions are stored under.

        A neural opponent is named by the same `phase16_behavior_v1|<identity>`
        scheme its bound snapshot carries; a handcrafted one is already an
        `id@version` token from the frozen Phase 4 registry.
        """
        if self.draw.opponent_kind == KIND_CURRENT:
            return self.behavior_policy_token
        if self.draw.opponent_kind == KIND_HISTORICAL:
            return behavior_token(self.draw.opponent_identity)
        return self.draw.opponent_identity

    @property
    def phase9_game_id(self) -> str:
        return self.draw.game_id

    @property
    def rollout_game_id(self) -> str:
        return self.draw.game_id

    @property
    def setup_root_seed(self) -> int:
        return 0

    @property
    def behavior_snapshot_identity(self) -> str:
        return self.behavior_identity

    @property
    def learner_control(self) -> str:
        return self.draw.learner_control

    @property
    def learner_color(self) -> "str | None":
        return self.draw.learner_color

    @property
    def learner_sides(self) -> tuple:
        return self.draw.learner_sides

    @property
    def opponent_kind(self) -> str:
        return {
            KIND_CURRENT: "current_policy",
            KIND_HISTORICAL: "historical_snapshot",
            KIND_HANDCRAFTED: "rule_policy",
        }[self.draw.opponent_kind]

    @property
    def opponent_identity(self) -> str:
        return self.draw.opponent_identity

    @property
    def historical_snapshot_identity(self) -> "str | None":
        return (
            self.draw.opponent_identity
            if self.draw.opponent_kind == KIND_HISTORICAL
            else None
        )

    @property
    def red_policy_identity(self) -> str:
        if self.draw.learner_color in (None, "red"):
            return self.behavior_policy_token
        return self.opponent_policy_token

    @property
    def blue_policy_identity(self) -> str:
        if self.draw.learner_color in (None, "blue"):
            return self.behavior_policy_token
        return self.opponent_policy_token

    @property
    def red_policy_seed(self) -> "int | None":
        return self.draw.red_policy_seed

    @property
    def blue_policy_seed(self) -> "int | None":
        return self.draw.blue_policy_seed


# ---------------------------------------------------------------------------
# One game in flight
# ---------------------------------------------------------------------------


class Phase16GameRunner(GameRunner):
    """The accepted Phase 9 runner, on the Phase 16 stream, harvesting rows.

    Two overrides. `apply_neural` swaps the action-sampling uniform and takes
    the training row while the live state is still in hand; nothing else about
    how a game is played, stored or validated changes.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.rows: list = []
        self.tracks: dict = {}
        super().__init__(*args, **kwargs)

    def _track_for(self, player: int) -> LearnerTrack:
        track = self.tracks.get(player)
        if track is None:
            track = LearnerTrack(game_id=self.game_id, player=int(player))
            self.tracks[player] = track
        return track

    def apply_neural(self, policy_logits_row, wdl_row) -> None:
        request = self.pending
        if request is None:  # pragma: no cover - the caller pairs these
            raise Phase16CollectorError(f"{self.game_id}: no pending neural decision")
        legality = request.legality
        actor = int(legality.acting_player)
        probabilities = behavior_distribution(policy_logits_row, legality)
        selected = select_action(
            probabilities, legality.absolute, self.game_id, request.ply
        )
        if selected not in legality.absolute:  # pragma: no cover - selection is an index
            raise Phase16CollectorError(
                f"{self.game_id} ply {request.ply}: selected action {selected} is not legal"
            )
        wdl = tuple(to_float32(float(v)) for v in np.asarray(wdl_row).reshape(3))
        expected_token = self._side_token(actor)
        if request.snapshot.policy_token != expected_token:
            raise Phase16CollectorError(
                f"{self.game_id} ply {request.ply}: acting snapshot token "
                f"{request.snapshot.policy_token!r} is not the scheduled "
                f"{expected_token!r}"
            )

        learner = self._is_learner(actor)
        if learner:
            self._harvest(request, probabilities, selected, wdl)

        self.builder.record_decision(
            self.state,
            legal_action_ids=tuple(legality.absolute),
            probabilities=probabilities,
            win_draw_loss_prediction=wdl,
            selected_action_id=selected,
            collection_policy_version=expected_token,
        )
        self.neural_decision_count += 1
        if learner:
            self.learner_decision_count += 1
            self.learner_neural_decision_count += 1
        self.pending = None
        apply_action(self.state, selected, legal=list(legality.absolute))

    def _harvest(self, request, probabilities, selected: int, wdl) -> None:
        """Take one training row from the live state, before the action lands.

        The belief label is privileged truth and is built *after* the public
        observation already exists, exactly as the accepted example builder
        orders them; the observation the model will see is the one the decision
        was made from, byte for byte.
        """
        legality = request.legality
        actor = int(legality.acting_player)
        labels, mask = dense_belief_target(self.state, actor)
        stored = tuple(to_float32(value) for value in probabilities)
        legal = tuple(int(action) for action in legality.absolute)
        probability = float(stored[legal.index(int(selected))])
        row = HarvestedRow(
            observation=np.ascontiguousarray(request.observation, dtype=np.float32),
            legal_mask=np.asarray(legality.model_mask, dtype=bool),
            sampled_action_model=int(absolute_action_to_model(int(selected), actor)),
            sampled_action_abs=int(selected),
            behavior_action_probability=probability,
            behavior_action_logprob=behavior_action_logprob(probability),
            behavior_legal_actions=legal,
            behavior_legal_probabilities=stored,
            belief_target=labels,
            belief_mask=mask,
            game_id=self.game_id,
            decision_index=int(request.ply),
            learner_side=actor,
        )
        self.rows.append(row)
        self._track_for(actor).record(
            ply=int(request.ply), prediction=wdl, row_index=len(self.rows) - 1
        )

    def finalize(self) -> list:
        """Close every track on the terminal result and fill exact targets.

        Returns the game's rows, each carrying the whole-game advantage and
        W/D/L target of the accepted recursions -- the standardization and the
        eligibility flag are per-window statistics and are applied later.
        """
        if self.record is None:
            raise Phase16CollectorError(f"{self.game_id}: the game has not finished")
        for track in self.tracks.values():
            track.close(self.record.terminal_result)
            exact = track_targets(track)
            for position, row_index in enumerate(track.row_indices):
                row = self.rows[row_index]
                row.advantage = float(exact["advantages"][position])
                row.wdl_target = tuple(
                    float(value) for value in exact["wdl_targets"][position]
                )
        return list(self.rows)


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


@dataclass
class WindowResult:
    """One closed window: its rows and everything the telemetry row needs."""

    iteration: int
    rows: list = field(default_factory=list)
    games_finished: int = 0
    games_started: int = 0
    learner_decisions: int = 0
    neural_decisions: int = 0
    plies: int = 0
    seconds: float = 0.0
    plies_advanced: int = 0
    terminal_results: dict = field(default_factory=dict)
    terminal_reasons: dict = field(default_factory=dict)
    game_lengths: list = field(default_factory=list)
    draws: list = field(default_factory=list)
    in_flight_rows: int = 0
    stopped_early: bool = False

    @property
    def plies_per_second(self) -> float:
        """Throughput over every ply the window advanced, finished or not.

        Counting only finished games would report a window that pushed 96 long
        games halfway as having done almost nothing, which is the opposite of
        the truth and would make the collection-throughput gate meaningless.
        """
        return self.plies_advanced / self.seconds if self.seconds > 0 else 0.0

    def summary(self) -> dict:
        lengths = np.asarray(self.game_lengths, dtype=np.float64) if self.game_lengths else None
        return {
            "iteration": int(self.iteration),
            "rows": len(self.rows),
            "games_finished": int(self.games_finished),
            "games_started": int(self.games_started),
            "learner_decisions": int(self.learner_decisions),
            "neural_decisions": int(self.neural_decisions),
            "plies": int(self.plies),
            "plies_advanced": int(self.plies_advanced),
            "seconds": round(float(self.seconds), 3),
            "plies_per_second": round(self.plies_per_second, 2),
            "terminal_results": dict(sorted(self.terminal_results.items())),
            "terminal_reasons": dict(sorted(self.terminal_reasons.items())),
            "game_length": {
                "mean": float(lengths.mean()),
                "p50": float(np.percentile(lengths, 50)),
                "p90": float(np.percentile(lengths, 90)),
                "max": float(lengths.max()),
            }
            if lengths is not None and lengths.size
            else {},
            "in_flight_rows": int(self.in_flight_rows),
            "stopped_early": bool(self.stopped_early),
        }


class WindowCollector:
    """A persistent population of games, advanced one window at a time.

    The population is created once and outlives every window: a game that does
    not finish simply keeps its place, and only games that *end* are replaced.
    That is the whole of the structural change, and it is why an iteration's
    wall-time no longer tracks how long the current games happen to be.
    """

    def __init__(
        self,
        config: ArmConfig,
        participants: IterationParticipants,
        *,
        setup_source,
        pool: "HistoricalPool | None" = None,
        snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
    ) -> None:
        self.config = config
        self.participants = participants
        self.setup_source = setup_source
        self.pool = pool or HistoricalPool()
        self.snapshot_interval = int(snapshot_interval)
        self.slots: list = [None] * int(config.population)
        self.draw_counts: list = [0] * int(config.population)
        #: plies of each slot's current game already counted in an earlier window
        self.ply_marks: list = [0] * int(config.population)
        self.iteration = 0
        self.games_completed = 0
        self.decisions_collected = 0
        self.plies_advanced = 0

    # -- population --------------------------------------------------------

    def _behavior_identity(self) -> str:
        return self.participants.behavior.logical_identity

    def _start(self, slot: int) -> Phase16GameRunner:
        draw = draw_for_slot(
            self.config,
            slot=slot,
            draw=self.draw_counts[slot],
            pool=self.pool,
        )
        self.draw_counts[slot] += 1
        scheduled = ScheduledPhase16Game(draw, self._behavior_identity())
        return Phase16GameRunner(
            scheduled,
            self.participants,
            setup_source=self.setup_source,
            behavior_checkpoint_sha256=self.participants.behavior.checkpoint_sha256,
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

    def in_flight_rows(self) -> int:
        return sum(len(runner.rows) for runner in self.slots if runner is not None)

    def rebind(self, participants: IterationParticipants) -> dict:
        """Point the population at updated weights under the same identity.

        A window collector continues the same games after an update, so a
        game's decisions legitimately come from several sets of weights. PPO
        does not mind: the ratio's denominator is the per-decision stored
        probability and every harvested row carries its own. What would break
        is the acting-token check, which compares the scheduled policy token to
        the snapshot's -- so the *logical identity* may not move while games are
        in flight, and a rotation that changes it is refused until the
        population is drained.
        """
        previous = self._behavior_identity()
        live = [slot for slot, runner in enumerate(self.slots) if runner is not None]
        if live and participants.behavior.logical_identity != previous:
            raise Phase16CollectorError(
                f"cannot rebind {previous!r} to "
                f"{participants.behavior.logical_identity!r} with {len(live)} games "
                "in flight; drain the population first"
            )
        before = self.participants.behavior.checkpoint_sha256
        self.participants = participants
        return {
            "identity": previous,
            "state_digest_before": before,
            "state_digest_after": participants.behavior.checkpoint_sha256,
            "games_in_flight": len(live),
        }

    # -- one window --------------------------------------------------------

    def collect_window(
        self,
        *,
        budget: "int | None" = None,
        min_rows: "int | None" = None,
        should_continue=None,
    ) -> WindowResult:
        """Advance the population until `budget` learner decisions are collected.

        The budget counts learner decisions *harvested in this window*, which
        is the quantity the trainer consumes, not games and not plies. Rows are
        emitted per finished game, so a window's row count is close to but not
        equal to its budget: it is short by whatever the games still in flight
        are holding, and long by the earlier windows' carry-over.

        `min_rows` keeps the very first window honest. A population seated from
        scratch has finished nothing, so a budget-sized first window can emit
        zero rows and there would be nothing to train on. Collection then
        continues past the budget until one minibatch exists. At steady state
        the condition is already met when the budget is and this costs nothing.
        """
        self.iteration += 1
        budget = int(self.config.window_decisions if budget is None else budget)
        floor = int(self.config.minibatch_size if min_rows is None else min_rows)
        result = WindowResult(iteration=self.iteration)
        started = time.perf_counter()
        result.games_started += self.fill()

        collected = 0
        while collected < budget or len(result.rows) < floor:
            if should_continue is not None and not should_continue():
                result.stopped_early = True
                break
            active = [runner for runner in self.slots if runner is not None]
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
                collected += self._evaluate(pending)

            for runner in finished:
                self._retire(runner, result)
            if finished:
                result.games_started += self.fill()

        # The population keeps its place; only the counters close. Every ply a
        # still-live game advanced in this window is counted here and marked,
        # so no ply is counted twice and none is lost at a boundary.
        for slot, runner in enumerate(self.slots):
            if runner is None:
                continue
            total = int(runner.state.total_moves)
            result.plies_advanced += total - self.ply_marks[slot]
            self.ply_marks[slot] = total
        result.learner_decisions = collected
        result.seconds = time.perf_counter() - started
        result.in_flight_rows = self.in_flight_rows()
        self.decisions_collected += collected
        self.plies_advanced += result.plies_advanced
        return result

    def _evaluate(self, pending) -> int:
        """Run one lockstep batch of neural decisions. Returns learner rows added."""
        from ..phase9_collector import _drain_batches

        before = 0
        for request in pending:
            before += len(request.runner.rows)
        for batch in _drain_batches(pending):
            observations = np.stack([request.observation for request in batch])
            policy_logits, wdl = evaluate_observations(batch[0].snapshot, observations)
            for row, request in enumerate(batch):
                request.runner.apply_neural(policy_logits[row], wdl[row])
        after = 0
        for request in pending:
            after += len(request.runner.rows)
        return after - before

    def _retire(self, runner: Phase16GameRunner, result: WindowResult) -> None:
        """Emit one finished game's rows and free its slot."""
        rows = runner.finalize()
        record = runner.record
        result.rows.extend(rows)
        result.games_finished += 1
        result.neural_decisions += runner.neural_decision_count
        result.plies += int(record.final_ply)
        result.game_lengths.append(int(record.final_ply))
        result.terminal_results[record.terminal_result] = (
            result.terminal_results.get(record.terminal_result, 0) + 1
        )
        result.terminal_reasons[record.terminal_reason] = (
            result.terminal_reasons.get(record.terminal_reason, 0) + 1
        )
        result.draws.append(runner.scheduled.draw)
        self.games_completed += 1
        for slot, seated in enumerate(self.slots):
            if seated is runner:
                result.plies_advanced += int(record.final_ply) - self.ply_marks[slot]
                self.ply_marks[slot] = 0
                self.slots[slot] = None
                break

    def drain(self, *, should_continue=None) -> WindowResult:
        """Play every in-flight game to its end without seating replacements.

        The only way to reach a state where the behavior snapshot can be
        rebound. Costs one partial window; a 6-hour arm pays it once per
        snapshot rotation, which is why rotation is a cadence and not a habit.
        """
        result = WindowResult(iteration=self.iteration)
        started = time.perf_counter()
        while any(runner is not None for runner in self.slots):
            if should_continue is not None and not should_continue():
                result.stopped_early = True
                break
            active = [runner for runner in self.slots if runner is not None]
            pending = []
            finished = []
            for runner in active:
                request = runner.advance()
                if request is None:
                    finished.append(runner)
                else:
                    pending.append(request)
            if pending:
                result.learner_decisions += self._evaluate(pending)
            for runner in finished:
                self._retire(runner, result)
        result.seconds = time.perf_counter() - started
        result.in_flight_rows = self.in_flight_rows()
        return result

    def state(self) -> dict:
        """The population's resume state. Carries no rows and no observations."""
        return {
            "collector_version": PHASE16_COLLECTOR_IMPL,
            "iteration": int(self.iteration),
            "population": len(self.slots),
            "draw_counts": list(self.draw_counts),
            "games_completed": int(self.games_completed),
            "decisions_collected": int(self.decisions_collected),
            "plies_advanced": int(self.plies_advanced),
            "behavior_snapshot_id": self._behavior_identity(),
            "behavior_checkpoint_sha256": self.participants.behavior.checkpoint_sha256,
            "in_flight": sum(1 for runner in self.slots if runner is not None),
            "pool": self.pool.to_dict(),
        }


def collector_semantics() -> dict:
    return {
        "collector_version": PHASE16_COLLECTOR_IMPL,
        "unit": "one window = a fixed budget of learner decisions",
        "runner": "stratego.training.phase9_collector.GameRunner, subclassed",
        "distribution": "stratego.training.phase9_behavior.behavior_distribution",
        "phase16_own": [
            "the action-sampling stream",
            "the game id scheme",
            "example harvesting at collection time",
        ],
        "emission": "whole games, on finish, with exact whole-game targets",
        "search": "absent; no module under stratego.search is imported",
    }


__all__ = [
    "HarvestedRow",
    "PHASE16_COLLECTOR_IMPL",
    "Phase16CollectorError",
    "Phase16GameRunner",
    "ScheduledPhase16Game",
    "WindowCollector",
    "WindowResult",
    "action_sampling_uniform",
    "collector_semantics",
    "select_action",
]
