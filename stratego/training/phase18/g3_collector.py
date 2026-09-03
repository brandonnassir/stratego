"""Phase 18 Stage 6B: the asynchronous pool-driven teacher-schedule collector
(G3-ENG-01, design v1 section 3.2 as retained by v2 section 2.3).

```text
slots      S concurrent game slots, advanced in fixed slot order
period     every slot advances T plies; a slot whose game ends immediately
           starts a new game from the CURRENT pool, so games started in
           period t may finish in t + 1 or later
cells      each new game takes the next cell of the frozen 100-cell teacher
           schedule in cyclic order (game starts are exactly uniform over cells)
pool       the k-th game started in a period takes red row k mod n and blue
           row (perm_p[k mod n] + k div n) mod n of the period's lanes, where
           perm_p is the period's seeded pairing permutation and n = pool / 2
outcomes   every completed game attributes exactly two outcomes, one per pool
           row from its owner's perspective, through SetupBuffer.add_outcome;
           an unattributable outcome is fatal (gate G3)
live       every completed game's trajectory is committed to the lineage's
           live record store with its metadata (gate G4 accounting)
rules      TRAINING_RULES (CORPUS_RULES), stamped on every record
seeds      per-game teacher seeds from (namespace, seed_index, colour, period,
           slot, ordinal) through derive_stream_seed; the lineage id enters
           no seed, which is what makes the two lineages' period 1 identical
```

The game runner reuses the accepted Phase 8 corpus decision loop
(`rule_population.play_corpus_game`) one ply at a time, and the Phase 17
`capture_active_game` / `restore_active_game` codec for exact persistence of
an unfinished game. Nothing about how a teacher decides, how a trajectory is
recorded or how an engine state is snapshotted is changed.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
import zlib
from dataclasses import dataclass, field

import numpy as np

from ...engine.constants import BLUE, PLAYER_NAMES, PLAYERS, RED
from ...engine.legal_moves import legal_actions
from ...engine.state import create_game
from ...engine.transition import apply_action
from ...evaluation.policy import PolicyContractError, build_policy_input
from ..rule_population import NEUTRAL_VALUE_PREDICTION, TeacherCache, teacher_by_token
from ..trajectory import GameTrajectoryBuilder, validate_game_record
from ..warmstart_contract import ordered_matchup_cells
from .g3_contract import (
    COLLECTOR_RULES,
    G3_COLLECTION_POLICY_VERSION,
    G3_HARNESS_VERSION,
    Phase18G3AccountingError,
    Phase18G3Error,
    PilotConfig,
    collector_policy_seed,
    pairing_seed,
)
from .g3_live_store import LivePeriodWriter
from .setup_buffer import SetupBuffer
from .setup_sampling import SampledSetup

#: The `trajectory_v1` setup-family label of a pool-drawn setup.
G3_SETUP_FAMILY = "phase18_g3_setup_pool_v1"

_TERMINAL_OUTCOME = {"red_win": {RED: 1, BLUE: -1}, "blue_win": {RED: -1, BLUE: 1}, "draw": {RED: 0, BLUE: 0}}


def live_game_id(run_id: str, seed_index: int, period: int, slot: int, draw: int) -> str:
    """The lineage-free identity of one collector game."""
    return f"phase18_g3|run={run_id}|seed={int(seed_index)}|p={int(period):04d}|slot={int(slot):05d}|d={int(draw):05d}"


def schedule_cells(count: int, indices=None) -> list:
    """The collector's cells in cyclic order: the named `indices`, or the first
    `count` cells of the frozen 100-cell schedule. Production names none."""
    cells = ordered_matchup_cells()
    if indices is not None:
        chosen = [cells[int(index)] for index in indices]
        if len(chosen) != int(count):
            raise Phase18G3Error("the named cells do not match schedule_cells")
        return chosen
    if not 1 <= int(count) <= len(cells):
        raise Phase18G3Error(f"schedule_cells must be in 1..{len(cells)}")
    return cells[: int(count)]


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameIdentity:
    """Everything that names one collector game and re-seats it exactly."""

    game_id: str
    period_started: int
    slot: int
    draw: int
    game_ordinal: int
    cell_index: int
    red_token: str
    blue_token: str
    red_seed: int
    blue_seed: int
    red_fingerprint: str
    blue_fingerprint: str
    red_class: str
    blue_class: str
    red_pool_index: int
    blue_pool_index: int
    pool_period: int
    pool_snapshot_digest: str
    red_reflected: bool
    blue_reflected: bool
    red_setup: tuple
    blue_setup: tuple

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "period_started": int(self.period_started),
            "slot": int(self.slot),
            "draw": int(self.draw),
            "game_ordinal": int(self.game_ordinal),
            "cell_index": int(self.cell_index),
            "red_token": self.red_token,
            "blue_token": self.blue_token,
            "red_seed": int(self.red_seed),
            "blue_seed": int(self.blue_seed),
            "red_fingerprint": self.red_fingerprint,
            "blue_fingerprint": self.blue_fingerprint,
            "red_class": self.red_class,
            "blue_class": self.blue_class,
            "red_pool_index": int(self.red_pool_index),
            "blue_pool_index": int(self.blue_pool_index),
            "pool_period": int(self.pool_period),
            "pool_snapshot_digest": self.pool_snapshot_digest,
            "red_reflected": bool(self.red_reflected),
            "blue_reflected": bool(self.blue_reflected),
            "red_setup": [int(v) for v in self.red_setup],
            "blue_setup": [int(v) for v in self.blue_setup],
        }

    @staticmethod
    def from_dict(payload: dict) -> "GameIdentity":
        return GameIdentity(
            game_id=str(payload["game_id"]),
            period_started=int(payload["period_started"]),
            slot=int(payload["slot"]),
            draw=int(payload["draw"]),
            game_ordinal=int(payload["game_ordinal"]),
            cell_index=int(payload["cell_index"]),
            red_token=str(payload["red_token"]),
            blue_token=str(payload["blue_token"]),
            red_seed=int(payload["red_seed"]),
            blue_seed=int(payload["blue_seed"]),
            red_fingerprint=str(payload["red_fingerprint"]),
            blue_fingerprint=str(payload["blue_fingerprint"]),
            red_class=str(payload["red_class"]),
            blue_class=str(payload["blue_class"]),
            red_pool_index=int(payload["red_pool_index"]),
            blue_pool_index=int(payload["blue_pool_index"]),
            pool_period=int(payload["pool_period"]),
            pool_snapshot_digest=str(payload["pool_snapshot_digest"]),
            red_reflected=bool(payload["red_reflected"]),
            blue_reflected=bool(payload["blue_reflected"]),
            red_setup=tuple(int(v) for v in payload["red_setup"]),
            blue_setup=tuple(int(v) for v in payload["blue_setup"]),
        )


class TeacherGameRunner:
    """One collector game, played one ply at a time under the frozen teachers.

    The decision loop is `rule_population.play_corpus_game`'s, split at the
    ply boundary. The attributes `traces`, `rows`, `row_index_by_key`,
    `pending` and the four decision counters exist so the accepted Phase 17
    `capture_active_game` / `restore_active_game` codec persists this runner
    verbatim; a teacher game has no neural decision and no seat trace.
    """

    def __init__(self, identity: GameIdentity, teachers: TeacherCache, *, snapshot_interval: int) -> None:
        self.identity = identity
        self.game_id = identity.game_id
        self.teachers = teachers
        self.policies = {RED: teachers.get(identity.red_token), BLUE: teachers.get(identity.blue_token)}
        self.seeds = {RED: int(identity.red_seed), BLUE: int(identity.blue_seed)}
        self.state = create_game(identity.red_setup, identity.blue_setup, rules=COLLECTOR_RULES, game_id=self.game_id)
        self.builder = GameTrajectoryBuilder(
            game_id=self.game_id,
            environment_id=0,
            generation=int(identity.pool_period),
            red_setup=identity.red_setup,
            blue_setup=identity.blue_setup,
            rules=COLLECTOR_RULES,
            root_seed=int(identity.red_seed),
            slot_seed=int(identity.slot),
            snapshot_interval=int(snapshot_interval),
            collection_policy_version=G3_COLLECTION_POLICY_VERSION,
            collection_checkpoint_id=identity.pool_snapshot_digest,
            setup_family=G3_SETUP_FAMILY,
        )
        self.record = None
        # Phase 17 codec fields (no neural decisions, no traces in a teacher game).
        self.traces: dict = {}
        self.rows: list = []
        self.row_index_by_key: dict = {}
        self.pending = None
        self.learner_decision_count = 0
        self.neural_decision_count = 0
        self.learner_neural_decision_count = 0
        self.rule_decision_count = 0
        self.period_completed = None

    @property
    def finished(self) -> bool:
        return self.record is not None

    def step(self) -> bool:
        """Play one ply. Returns True when the game is terminal afterwards."""
        if self.record is not None:
            raise Phase18G3Error(f"{self.game_id}: stepping a finished game")
        state = self.state
        if state.terminal:
            return True
        actor = state.acting_player
        policy = self.policies[actor]
        legal = legal_actions(state)
        request = build_policy_input(
            state,
            policy=policy.ref,
            policy_seed=self.seeds[actor],
            requirements=policy.requirements,
            game_id=self.game_id,
            legal=legal,
        )
        try:
            result = policy.decide_checked(request)
        except PolicyContractError as error:
            raise Phase18G3Error(
                f"game {self.game_id}: policy {policy.ref.token} violated its contract at ply {request.ply}: {error}"
            ) from error
        selected = int(result.selected_action_id)
        self.builder.record_decision(
            state,
            legal_action_ids=legal,
            probabilities=tuple(1.0 if action == selected else 0.0 for action in legal),
            win_draw_loss_prediction=NEUTRAL_VALUE_PREDICTION,
            selected_action_id=selected,
            collection_policy_version=policy.ref.token,
        )
        self.rule_decision_count += 1
        apply_action(state, selected, legal=legal)
        return bool(state.terminal)

    def finish(self):
        if not self.state.terminal:
            raise Phase18G3Error(f"{self.game_id}: finishing a game that is not terminal")
        record = self.builder.finish(self.state)
        problems = validate_game_record(record)
        if problems:
            raise Phase18G3Error(f"game {self.game_id}: the sealed trajectory is invalid: {problems}")
        self.record = record
        return record

    def outcomes(self) -> dict:
        """`{RED: z, BLUE: z}` from each owner's perspective; +1 win, 0 draw, -1 loss."""
        if self.record is None:
            raise Phase18G3Error(f"{self.game_id}: no outcome before the game finished")
        try:
            return dict(_TERMINAL_OUTCOME[self.record.terminal_result])
        except KeyError:
            raise Phase18G3Error(f"{self.game_id}: unknown terminal result {self.record.terminal_result!r}") from None

    # -- persistence through the Phase 17 codec -------------------------------

    def capture(self) -> dict:
        from ..phase17.checkpoint import capture_active_game

        return {
            "identity": self.identity.to_dict(),
            "rule_decision_count": int(self.rule_decision_count),
            "phase17": capture_active_game(self, slot=int(self.identity.slot), draw=int(self.identity.draw)),
        }

    @classmethod
    def restore(cls, payload: dict, teachers: TeacherCache, *, snapshot_interval: int) -> "TeacherGameRunner":
        from ..phase17.checkpoint import restore_active_game

        identity = GameIdentity.from_dict(payload["identity"])
        runner = cls(identity, teachers, snapshot_interval=snapshot_interval)
        restore_active_game(runner, payload["phase17"])
        runner.rule_decision_count = int(payload["rule_decision_count"])
        if runner.state.game_id != identity.game_id:
            raise Phase18G3Error(f"restored state names {runner.state.game_id!r}, expected {identity.game_id!r}")
        if list(runner.state.action_history) != list(runner.builder._actions):
            raise Phase18G3Error(f"{identity.game_id}: the restored builder and engine disagree on the action history")
        return runner


# ---------------------------------------------------------------------------
# Pool lanes and pairing
# ---------------------------------------------------------------------------


class PoolLanes:
    """One period's pool split into its red and blue lanes, with the pairing rule."""

    def __init__(self, samples, *, period: int, namespace: str, seed_index: int, snapshot_digest: str) -> None:
        self.period = int(period)
        self.snapshot_digest = snapshot_digest
        self.red = [sample for sample in samples if sample.lane == RED]
        self.blue = [sample for sample in samples if sample.lane == BLUE]
        if not self.red or not self.blue:
            raise Phase18G3Error("a pool needs at least one setup in each lane")
        if len(self.red) != len(self.blue):
            raise Phase18G3Error(f"lanes are unequal: {len(self.red)} red, {len(self.blue)} blue")
        for sample in samples:
            if not isinstance(sample, SampledSetup):
                raise Phase18G3Error("pools hold SampledSetup rows only")
        n = len(self.red)
        self.permutation = np.random.default_rng(pairing_seed(namespace, seed_index, period) % (2**32)).permutation(n)
        self.uses = 0

    def __len__(self) -> int:
        return len(self.red)

    def pair(self, k: int) -> tuple:
        """`(red_index, blue_index)` of the k-th game started in the period."""
        n = len(self.red)
        red_index = int(k % n)
        blue_index = int((int(self.permutation[k % n]) + k // n) % n)
        return red_index, blue_index


# ---------------------------------------------------------------------------
# The collector
# ---------------------------------------------------------------------------


@dataclass
class PeriodAccounting:
    """Gate G4: started = completed + in_flight + failed, per period."""

    period: int
    started: int = 0
    completed: int = 0
    failed: int = 0
    in_flight_at_end: int = 0
    plies_advanced: int = 0
    outcomes_attributed: int = 0
    cross_period_attributions: int = 0
    immediately_terminal_games: int = 0
    completed_game_ids: list = field(default_factory=list)
    outcome_records: list = field(default_factory=list)
    completed_lengths: list = field(default_factory=list)
    terminal_results: dict = field(default_factory=dict)
    terminal_reasons: dict = field(default_factory=dict)
    completions_per_cell: dict = field(default_factory=dict)
    seconds: float = 0.0

    def document(self) -> dict:
        return {
            "period": int(self.period),
            "started": int(self.started),
            "completed": int(self.completed),
            "failed": int(self.failed),
            "in_flight_at_end": int(self.in_flight_at_end),
            "plies_advanced": int(self.plies_advanced),
            "outcomes_attributed": int(self.outcomes_attributed),
            "cross_period_attributions": int(self.cross_period_attributions),
            "immediately_terminal_games": int(self.immediately_terminal_games),
            "completed_games": len(self.completed_game_ids),
            "completed_game_ids_digest": hashlib.sha256("\n".join(sorted(self.completed_game_ids)).encode()).hexdigest(),
            "outcome_records_digest": hashlib.sha256(
                json.dumps(self.outcome_records, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "mean_completed_length": (
                float(np.mean(self.completed_lengths)) if self.completed_lengths else None
            ),
            "terminal_results": dict(self.terminal_results),
            "terminal_reasons": dict(self.terminal_reasons),
            "completions_per_cell": {str(k): v for k, v in sorted(self.completions_per_cell.items())},
            "seconds": round(float(self.seconds), 3),
        }


class PeriodCollector:
    """The persistent slot population of one lineage."""

    def __init__(self, config: PilotConfig, buffer: SetupBuffer, *, live_root) -> None:
        self.config = config
        self.buffer = buffer
        self.live_root = live_root
        self.teachers = TeacherCache()
        self.cells = schedule_cells(config.schedule_cells, config.cell_indices)
        self.slots: list = [None] * int(config.slots)
        self.draw_counts: list = [0] * int(config.slots)
        self.cell_cursor = 0
        self.game_ordinal = 0
        self.periods_completed = 0
        self.games_started_total = 0
        self.games_completed_total = 0
        self.outcomes_total = 0
        self.attribution_failures = 0
        self.lanes: "PoolLanes | None" = None
        self.current: "PeriodAccounting | None" = None
        self._writer: "LivePeriodWriter | None" = None
        self._period_games_started = 0

    # -- one period --------------------------------------------------------

    def begin_period(self, period: int, samples, *, snapshot_digest: str) -> None:
        if self.current is not None:
            raise Phase18G3Error("the previous period was not ended")
        if int(period) != self.periods_completed + 1:
            raise Phase18G3Error(f"period {period} does not follow {self.periods_completed}")
        self.lanes = PoolLanes(
            samples,
            period=int(period),
            namespace=self.config.namespace,
            seed_index=self.config.seed_index,
            snapshot_digest=snapshot_digest,
        )
        self.current = PeriodAccounting(period=int(period))
        self._period_games_started = 0
        self._writer = LivePeriodWriter(
            self.live_root,
            period=int(period),
            namespace=self.config.namespace,
            lineage=self.config.lineage,
            run_id=self.config.run_id,
        )

    def _start(self, slot: int) -> TeacherGameRunner:
        assert self.lanes is not None and self.current is not None
        period = self.current.period
        draw = self.draw_counts[slot]
        self.draw_counts[slot] += 1
        k = self._period_games_started
        self._period_games_started += 1
        red_index, blue_index = self.lanes.pair(k)
        red, blue = self.lanes.red[red_index], self.lanes.blue[blue_index]
        cell = self.cells[self.cell_cursor % len(self.cells)]
        self.cell_cursor += 1
        identity = GameIdentity(
            game_id=live_game_id(self.config.run_id, self.config.seed_index, period, slot, draw),
            period_started=period,
            slot=int(slot),
            draw=int(draw),
            game_ordinal=int(self.game_ordinal),
            cell_index=int(cell["cell_index"]),
            red_token=cell["red_token"],
            blue_token=cell["blue_token"],
            red_seed=collector_policy_seed(self.config.namespace, self.config.seed_index, "red", period, slot, draw),
            blue_seed=collector_policy_seed(self.config.namespace, self.config.seed_index, "blue", period, slot, draw),
            red_fingerprint=red.content_fingerprint,
            blue_fingerprint=blue.content_fingerprint,
            red_class=red.class_fingerprint,
            blue_class=blue.class_fingerprint,
            red_pool_index=int(red.index),
            blue_pool_index=int(blue.index),
            pool_period=int(self.lanes.period),
            pool_snapshot_digest=self.lanes.snapshot_digest,
            red_reflected=bool(red.reflected),
            blue_reflected=bool(blue.reflected),
            red_setup=tuple(int(v) for v in red.engine_setup),
            blue_setup=tuple(int(v) for v in blue.engine_setup),
        )
        self.game_ordinal += 1
        self.games_started_total += 1
        self.current.started += 1
        return TeacherGameRunner(identity, self.teachers, snapshot_interval=self.config.snapshot_interval)

    def _complete(self, runner: TeacherGameRunner, slot: int) -> None:
        assert self.current is not None and self._writer is not None
        record = runner.finish()
        period = self.current.period
        runner.period_completed = period
        outcomes = runner.outcomes()
        identity = runner.identity
        for colour, fingerprint in ((RED, identity.red_fingerprint), (BLUE, identity.blue_fingerprint)):
            try:
                self.buffer.add_outcome(fingerprint, outcomes[colour])
            except Exception as error:  # Phase18SetupAttributionError, re-raised as the G3 gate
                self.attribution_failures += 1
                raise Phase18G3Error(
                    f"{identity.game_id}: outcome for the {PLAYER_NAMES[colour]} setup could not be "
                    f"attributed (pool period {identity.pool_period}, current {period}): {error}"
                ) from error
            self.current.outcomes_attributed += 1
            self.current.outcome_records.append([fingerprint, int(outcomes[colour])])
            self.outcomes_total += 1
        if identity.pool_period != period:
            self.current.cross_period_attributions += 2
        metadata = build_live_metadata(runner, period_completed=period, config=self.config)
        self._writer.write(record, metadata)
        self.current.completed += 1
        self.games_completed_total += 1
        self.current.completed_game_ids.append(identity.game_id)
        self.current.completed_lengths.append(int(record.final_ply))
        self.current.terminal_results[record.terminal_result] = self.current.terminal_results.get(record.terminal_result, 0) + 1
        self.current.terminal_reasons[record.terminal_reason] = self.current.terminal_reasons.get(record.terminal_reason, 0) + 1
        self.current.completions_per_cell[identity.cell_index] = self.current.completions_per_cell.get(identity.cell_index, 0) + 1
        if record.final_ply == 0:
            self.current.immediately_terminal_games += 1
        self.slots[slot] = None

    def run_period(self) -> PeriodAccounting:
        """Advance every slot exactly `plies_per_period` plies, in slot order."""
        if self.current is None or self.lanes is None:
            raise Phase18G3Error("begin_period must precede run_period")
        started = time.perf_counter()
        budget = int(self.config.plies_per_period)
        for slot in range(len(self.slots)):
            plies_left = budget
            starts_this_slot = 0
            while plies_left > 0:
                runner = self.slots[slot]
                if runner is None:
                    if starts_this_slot > budget:
                        raise Phase18G3Error(
                            f"slot {slot} started more games than plies in one period; every pool "
                            "setup appears to be terminal at creation"
                        )
                    runner = self._start(slot)
                    starts_this_slot += 1
                    self.slots[slot] = runner
                    if runner.state.terminal:
                        # Terminal at creation: the mobility rule decided it (engine 1.2.0).
                        self._complete(runner, slot)
                        continue
                terminal = runner.step()
                plies_left -= 1
                self.current.plies_advanced += 1
                if terminal:
                    self._complete(runner, slot)
        self.current.seconds = time.perf_counter() - started
        return self.current

    def end_period(self) -> dict:
        """Finalise the live file set and check the accounting identity."""
        if self.current is None or self._writer is None:
            raise Phase18G3Error("no period is open")
        accounting = self.current
        accounting.in_flight_at_end = sum(1 for runner in self.slots if runner is not None)
        live_summary = self._writer.close()
        self._writer = None
        if accounting.started != accounting.completed + accounting.in_flight_at_end - self._carried_in() + accounting.failed:
            raise Phase18G3AccountingError(
                f"period {accounting.period}: started {accounting.started} != completed "
                f"{accounting.completed} + in-flight delta + failed {accounting.failed}"
            )
        if live_summary["games"] != accounting.completed:
            raise Phase18G3AccountingError(
                f"period {accounting.period}: {live_summary['games']} live games committed for "
                f"{accounting.completed} completions"
            )
        if accounting.outcomes_attributed != 2 * accounting.completed:
            raise Phase18G3AccountingError(
                f"period {accounting.period}: {accounting.outcomes_attributed} outcomes for {accounting.completed} games"
            )
        self.periods_completed += 1
        self._carried = accounting.in_flight_at_end
        document = accounting.document() | {"live": live_summary}
        self.current = None
        self.lanes = None
        return document

    _carried = 0

    def _carried_in(self) -> int:
        """Games in flight when the period began (started in earlier periods)."""
        return int(self._carried)

    # -- persistence -------------------------------------------------------

    def capture(self) -> dict:
        """The population between periods, for the joint bundle."""
        if self.current is not None:
            raise Phase18G3Error("the collector is captured between periods only")
        active = []
        for slot, runner in enumerate(self.slots):
            if runner is None:
                continue
            active.append(runner.capture())
        return {
            "harness_version": G3_HARNESS_VERSION,
            "slots": int(len(self.slots)),
            "draw_counts": [int(v) for v in self.draw_counts],
            "cell_cursor": int(self.cell_cursor),
            "game_ordinal": int(self.game_ordinal),
            "periods_completed": int(self.periods_completed),
            "games_started_total": int(self.games_started_total),
            "games_completed_total": int(self.games_completed_total),
            "outcomes_total": int(self.outcomes_total),
            "attribution_failures": int(self.attribution_failures),
            "carried": int(self._carried),
            "active_games": len(active),
            # Pickled, not JSON: the Phase 17 payload carries encoded snapshot bytes.
            "active_games_blob": zlib.compress(pickle.dumps(active, protocol=4), 6),
            "active_game_ids": [entry["identity"]["game_id"] for entry in active],
        }

    def restore(self, payload: dict) -> dict:
        if payload.get("harness_version") != G3_HARNESS_VERSION:
            raise Phase18G3Error(f"collector state under harness {payload.get('harness_version')!r}")
        if int(payload["slots"]) != len(self.slots):
            raise Phase18G3Error(f"checkpoint has {payload['slots']} slots, the configuration {len(self.slots)}")
        self.draw_counts = [int(v) for v in payload["draw_counts"]]
        self.cell_cursor = int(payload["cell_cursor"])
        self.game_ordinal = int(payload["game_ordinal"])
        self.periods_completed = int(payload["periods_completed"])
        self.games_started_total = int(payload["games_started_total"])
        self.games_completed_total = int(payload["games_completed_total"])
        self.outcomes_total = int(payload["outcomes_total"])
        self.attribution_failures = int(payload["attribution_failures"])
        self._carried = int(payload["carried"])
        active = pickle.loads(zlib.decompress(payload["active_games_blob"]))
        if len(active) != int(payload["active_games"]):
            raise Phase18G3Error("the active-game blob does not hold the recorded number of games")
        self.slots = [None] * len(self.slots)
        restored = 0
        for entry in active:
            runner = TeacherGameRunner.restore(entry, self.teachers, snapshot_interval=self.config.snapshot_interval)
            slot = int(runner.identity.slot)
            if self.slots[slot] is not None:
                raise Phase18G3Error(f"two active games claim slot {slot}")
            self.slots[slot] = runner
            restored += 1
        if [entry["identity"]["game_id"] for entry in active] != list(payload["active_game_ids"]):
            raise Phase18G3Error("the restored active games do not match the recorded ids")
        return {"games_restored": restored, "periods_completed": self.periods_completed}

    def telemetry(self) -> dict:
        return {
            "periods_completed": int(self.periods_completed),
            "games_started_total": int(self.games_started_total),
            "games_completed_total": int(self.games_completed_total),
            "outcomes_total": int(self.outcomes_total),
            "in_flight": sum(1 for runner in self.slots if runner is not None),
            "attribution_failures": int(self.attribution_failures),
            "cell_cursor": int(self.cell_cursor),
            "schedule_cells": len(self.cells),
        }


def build_live_metadata(runner: TeacherGameRunner, *, period_completed: int, config: PilotConfig) -> dict:
    """The live sidecar: the Phase 8 fields the example builder reads, plus the
    pilot's identity and the exact pool provenance of both setups."""
    identity = runner.identity
    record = runner.record
    if record is None:
        raise Phase18G3Error(f"{identity.game_id}: metadata before the game finished")
    red = teacher_by_token(identity.red_token)
    blue = teacher_by_token(identity.blue_token)
    return {
        "corpus_version": G3_COLLECTION_POLICY_VERSION,
        "corpus_split": "train",
        "synthetic_game_id": identity.game_id,
        "run_id": config.run_id,
        "lineage": config.lineage,
        "seed_index": int(config.seed_index),
        "period_started": int(identity.period_started),
        "period_completed": int(period_completed),
        "slot": int(identity.slot),
        "draw": int(identity.draw),
        "game_ordinal": int(identity.game_ordinal),
        "cell_index": int(identity.cell_index),
        "ordered_matchup_id": f"{identity.red_token}>{identity.blue_token}",
        "red_policy_id": red.policy_id,
        "red_policy_version": red.policy_version,
        "red_policy_seed": int(identity.red_seed),
        "red_policy_weight": float(red.policy_weight),
        "blue_policy_id": blue.policy_id,
        "blue_policy_version": blue.policy_version,
        "blue_policy_seed": int(identity.blue_seed),
        "blue_policy_weight": float(blue.policy_weight),
        "setup_provenance": {
            "source": G3_SETUP_FAMILY,
            "pool_period": int(identity.pool_period),
            "pool_snapshot_digest": identity.pool_snapshot_digest,
            "red": {
                "primary_family_id": G3_SETUP_FAMILY,
                "content_fingerprint": identity.red_fingerprint,
                "class_fingerprint": identity.red_class,
                "pool_index": int(identity.red_pool_index),
                "reflected": bool(identity.red_reflected),
            },
            "blue": {
                "primary_family_id": G3_SETUP_FAMILY,
                "content_fingerprint": identity.blue_fingerprint,
                "class_fingerprint": identity.blue_class,
                "pool_index": int(identity.blue_pool_index),
                "reflected": bool(identity.blue_reflected),
            },
        },
        "red_setup": [int(v) for v in record.red_setup],
        "blue_setup": [int(v) for v in record.blue_setup],
        "setup_family": record.setup_family,
        "setup_id": record.setup_id,
        "trajectory_version": record.trajectory_version,
        "snapshot_interval": int(record.snapshot_interval),
        "first_player": record.first_player,
        "rules_context": record.rules_context,
        "terminal_result": record.terminal_result,
        "terminal_reason": record.terminal_reason,
        "final_ply": int(record.final_ply),
        "total_decisions": len(record.decisions),
        "rule_decision_count": int(runner.rule_decision_count),
        "collection_policy_version": G3_COLLECTION_POLICY_VERSION,
    }


__all__ = [
    "G3_SETUP_FAMILY",
    "GameIdentity",
    "PeriodAccounting",
    "PeriodCollector",
    "PoolLanes",
    "TeacherGameRunner",
    "build_live_metadata",
    "live_game_id",
    "schedule_cells",
]
