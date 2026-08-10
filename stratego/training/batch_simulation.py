"""Single-process batch wrapper around the frozen Phase 2.1 reference engine.

Specification sources:

- `03_game_engine_spec.md` section 16 (batch simulation interface)
- `03_game_engine_spec.md` section 17 (environment identifier and generation)
- `03_game_engine_spec.md` sections 10, 19 (atomicity, loud failure)

The wrapper owns `N` independent reference-engine games ("slots") and exposes
them with the bulk-synchronous semantics Phase 3 is built on:

```text
read observations / legality for every active slot
-> choose one action per active slot
-> apply every chosen action
-> collect terminal results
-> reset the finished slots the caller selects
-> next batch step
```

It contains no simulation logic of its own. A slot is nothing more than a
`GameState` the wrapper happens to hold, so batching cannot alter behaviour.

Slot identity
-------------
`environment_id` is fixed for the lifetime of a slot. `generation` starts at `0`
and increments by exactly one each time the slot is reset into a new game, so
`(environment_id, generation)` identifies exactly one game and no trajectory
record can span a reset boundary.

Determinism
-----------
Every game is built from `derive_slot_seed(root_seed, environment_id,
generation)`. The whole batch is therefore reproducible from `root_seed` alone,
*and* any single slot generation is reproducible without replaying its
neighbours -- which is what lets Agent 2 rebuild an arbitrary slot inside an
arbitrary worker process.

Illegal actions
---------------
:meth:`BatchSimulator.step` validates every submitted action before it applies
any of them, so one illegal action aborts the whole batch step with no slot
mutated at all. That is strictly stronger than the engine's own per-state
atomicity guarantee and it is what makes a rejected batch step inert.
"""

import hashlib
from bisect import bisect_left
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..engine.constants import (
    ACTION_SPACE_SIZE,
    NOT_TERMINAL,
    OBSERVATION_SHAPE,
    PLAYERS,
    TRAINING_RULES,
    RulesConfig,
)
from ..engine.events import (
    filter_events_for_observer,
    public_board_view,
    public_setup_view,
)
from ..engine.legal_moves import legal_action_mask, legal_actions
from ..engine.observation import OBSERVATION_DTYPE, belief_target, build_observation
from ..engine.random_play import make_random_setups
from ..engine.replay import ReplayRecord, build_replay_record
from ..engine.snapshot import create_snapshot
from ..engine.state import GameState, create_game, render_board, state_fingerprint
from ..engine.transition import IllegalActionError, TerminalStateError, apply_action

BATCH_INTERFACE_VERSION = "batch_simulation_v1"

# `acting_players` reports this for a finished slot: a terminal state has no
# player to move, and `GameState.acting_player` still names the last mover.
NO_ACTING_PLAYER = -1

# A dense action vector uses any negative entry to mean "do not step this slot".
SKIP_ACTION = -1


class BatchSimulationError(ValueError):
    """Base class for every batch-level rejection."""


class UnknownEnvironmentError(BatchSimulationError):
    """Raised when a slot index does not exist in this batch."""


class BatchIllegalActionError(BatchSimulationError, IllegalActionError):
    """Raised when a submitted action is not legal in its slot.

    Also an :class:`~stratego.engine.transition.IllegalActionError`, so callers
    that already handle the single-game error handle the batch error too.
    """


class BatchTerminalStateError(BatchSimulationError, TerminalStateError):
    """Raised when a step is submitted for a finished slot."""


def derive_slot_seed(root_seed: int, environment_id: int, generation: int) -> int:
    """Deterministic per-generation seed for one environment slot.

    A hash rather than arithmetic mixing, so neighbouring slots and consecutive
    generations get unrelated setups instead of correlated ones. The value only
    depends on the three arguments, so any process can rebuild any slot.
    """
    payload = f"{int(root_seed)}:{int(environment_id)}:{int(generation)}".encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"stratego-slot").digest()
    # Keep the seed non-negative; `random.Random` accepts arbitrarily large ints.
    return int.from_bytes(digest, "big") >> 1


def slot_game_id(root_seed: int, environment_id: int, generation: int) -> str:
    """Stable game identifier for one `(environment_id, generation)` game.

    The identifier is part of the state fingerprint, so it must be derived the
    same way by anything that rebuilds a slot for differential comparison.
    """
    return f"batch{int(root_seed)}-env{int(environment_id):06d}-gen{int(generation):06d}"


@dataclass
class EnvironmentSlot:
    """One persistent environment slot and the game currently living in it."""

    environment_id: int
    generation: int
    seed: int
    game_id: str
    red_setup: tuple[int, ...]
    blue_setup: tuple[int, ...]
    state: GameState
    # Legal actions for the current state, generated at most once per state.
    # Invalidated by every transition and by every reset.
    cached_legal_actions: list[int] | None = field(default=None, repr=False)

    @property
    def trajectory_key(self) -> tuple[int, int]:
        """`(environment_id, generation)`; identifies one game."""
        return (self.environment_id, self.generation)


@dataclass(frozen=True)
class SlotOutcome:
    """Result/reason view of one slot, terminal or not."""

    environment_id: int
    generation: int
    game_id: str
    terminal: bool
    terminal_reason: str
    winner: int | None
    is_draw: bool
    total_moves: int
    battleless_moves: int
    result_for_red: float | None
    result_for_blue: float | None

    @property
    def trajectory_key(self) -> tuple[int, int]:
        return (self.environment_id, self.generation)


@dataclass(frozen=True)
class BatchStepResult:
    """What one batch step did.

    `events` holds the derived engine events each stepped slot produced, keyed by
    slot index, which is what a trajectory recorder needs. `outcomes` carries the
    terminal result of every slot that finished *during this step*.
    """

    stepped: tuple[int, ...]
    actions: dict[int, int]
    events: dict[int, tuple[dict, ...]]
    newly_terminal: tuple[int, ...]
    outcomes: dict[int, SlotOutcome]


class BatchSimulator:
    """`N` independent frozen-reference games with bulk-synchronous access.

    Correctness first: the implementation loops over individual states and makes
    no attempt to vectorise the engine.
    """

    def __init__(
        self,
        num_environments: int,
        *,
        root_seed: int = 0,
        rules: RulesConfig = TRAINING_RULES,
        first_environment_id: int = 0,
    ) -> None:
        """`num_environments` slots seeded from `root_seed`.

        `first_environment_id` offsets the `environment_id` of the first slot;
        it defaults to `0`, so an existing single-process caller is unaffected.
        Agent 2 gives each simulation worker a simulator over a disjoint
        `environment_id` range while keeping one global `root_seed`, which is
        what makes `derive_slot_seed(root_seed, environment_id, generation)`
        unique across the whole run and independent of which process owns the
        slot.
        """
        if num_environments < 1:
            raise ValueError("a batch needs at least one environment")
        if first_environment_id < 0:
            raise ValueError("first_environment_id must not be negative")
        self.num_environments = int(num_environments)
        self.root_seed = int(root_seed)
        self.rules = rules
        self.first_environment_id = int(first_environment_id)
        self._slots: list[EnvironmentSlot] = [
            self._build_slot(self.first_environment_id + offset, 0)
            for offset in range(self.num_environments)
        ]

    # -- construction ------------------------------------------------------

    def _build_slot(self, environment_id: int, generation: int) -> EnvironmentSlot:
        seed = derive_slot_seed(self.root_seed, environment_id, generation)
        red_setup, blue_setup = make_random_setups(seed)
        game_id = slot_game_id(self.root_seed, environment_id, generation)
        state = create_game(red_setup, blue_setup, rules=self.rules, game_id=game_id)
        return EnvironmentSlot(
            environment_id=environment_id,
            generation=generation,
            seed=seed,
            game_id=game_id,
            red_setup=red_setup,
            blue_setup=blue_setup,
            state=state,
        )

    # -- slot addressing ---------------------------------------------------

    def __len__(self) -> int:
        return self.num_environments

    def _slot(self, slot: int) -> EnvironmentSlot:
        index = int(slot)
        if not 0 <= index < self.num_environments:
            raise UnknownEnvironmentError(
                f"slot {slot!r} is outside this batch of {self.num_environments}"
            )
        return self._slots[index]

    def _slot_list(self, slots: "Iterable[int] | None") -> tuple[int, ...]:
        """Normalise a slot selection; `None` means every active slot."""
        if slots is None:
            return self.active_slots()
        selected = tuple(int(slot) for slot in slots)
        for slot in selected:
            self._slot(slot)
        return selected

    def environment_ids(self) -> tuple[int, ...]:
        """Fixed `environment_id` of every slot, in slot order."""
        return tuple(slot.environment_id for slot in self._slots)

    def generations(self) -> tuple[int, ...]:
        """Current `generation` of every slot, in slot order."""
        return tuple(slot.generation for slot in self._slots)

    def generation(self, slot: int) -> int:
        return self._slot(slot).generation

    def environment_id(self, slot: int) -> int:
        return self._slot(slot).environment_id

    def trajectory_key(self, slot: int) -> tuple[int, int]:
        """`(environment_id, generation)` of the game currently in `slot`."""
        return self._slot(slot).trajectory_key

    def game_id(self, slot: int) -> str:
        return self._slot(slot).game_id

    def slot_seed(self, slot: int) -> int:
        return self._slot(slot).seed

    def setups(self, slot: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """`(red_setup, blue_setup)` of the game currently in `slot`."""
        record = self._slot(slot)
        return record.red_setup, record.blue_setup

    def game_state(self, slot: int) -> GameState:
        """The live privileged state. Callers must not mutate it."""
        return self._slot(slot).state

    def active_slots(self) -> tuple[int, ...]:
        return tuple(
            index for index, slot in enumerate(self._slots) if not slot.state.terminal
        )

    def finished_slots(self) -> tuple[int, ...]:
        return tuple(
            index for index, slot in enumerate(self._slots) if slot.state.terminal
        )

    def is_terminal(self, slot: int) -> bool:
        return self._slot(slot).state.terminal

    @property
    def num_active(self) -> int:
        return sum(1 for slot in self._slots if not slot.state.terminal)

    # -- model-facing reads ------------------------------------------------

    def acting_players(self, slots: "Iterable[int] | None" = None) -> np.ndarray:
        """Acting player per selected slot, `NO_ACTING_PLAYER` when finished."""
        selected = self._slot_list(slots)
        players = np.empty(len(selected), dtype=np.int8)
        for position, slot in enumerate(selected):
            state = self._slots[slot].state
            players[position] = (
                NO_ACTING_PLAYER if state.terminal else state.acting_player
            )
        return players

    def acting_player(self, slot: int) -> int:
        state = self._slot(slot).state
        return NO_ACTING_PLAYER if state.terminal else state.acting_player

    def observation(self, slot: int, observer: int | None = None) -> np.ndarray:
        """`observation_v2_1_127ch` for `observer`, defaulting to the mover."""
        return build_observation(self._slot(slot).state, observer)

    def observations(self, slots: "Iterable[int] | None" = None) -> np.ndarray:
        """Stacked `(n, 127, 10, 10)` acting-player observations.

        The stacked layout is the one Agent 2 copies into shared memory.
        """
        selected = self._slot_list(slots)
        stacked = np.empty((len(selected),) + OBSERVATION_SHAPE, dtype=OBSERVATION_DTYPE)
        for position, slot in enumerate(selected):
            stacked[position] = build_observation(self._slots[slot].state)
        return stacked

    def legal_actions(self, slot: int) -> list[int]:
        """Ascending legal actions for the mover; empty for a finished slot."""
        return list(self._legal_actions(self._slot(slot)))

    def legal_action_lists(
        self, slots: "Iterable[int] | None" = None
    ) -> list[list[int]]:
        selected = self._slot_list(slots)
        return [list(self._legal_actions(self._slots[slot])) for slot in selected]

    def legal_action_mask(self, slot: int) -> np.ndarray:
        """Dense 10,000-entry `uint8` mask for one slot."""
        record = self._slot(slot)
        return legal_action_mask(record.state, self._legal_actions(record))

    def legal_action_masks(self, slots: "Iterable[int] | None" = None) -> np.ndarray:
        """Stacked `(n, 10000)` `uint8` legality masks."""
        selected = self._slot_list(slots)
        masks = np.zeros((len(selected), ACTION_SPACE_SIZE), dtype=np.uint8)
        for position, slot in enumerate(selected):
            record = self._slots[slot]
            masks[position] = legal_action_mask(record.state, self._legal_actions(record))
        return masks

    def _legal_actions(self, record: EnvironmentSlot) -> list[int]:
        """Cached legal-action list for one slot.

        The list is a pure function of the state, so caching it cannot change
        behaviour; it only avoids regenerating the same list for the observation
        read, the mask read and the step validation of a single batch step.
        """
        if record.cached_legal_actions is None:
            record.cached_legal_actions = legal_actions(record.state)
        return record.cached_legal_actions

    # -- stepping ----------------------------------------------------------

    def step(self, actions: "Mapping[int, int] | Sequence[int]") -> BatchStepResult:
        """Apply at most one action per slot.

        `actions` is either a `{slot: action_id}` mapping, or a dense sequence of
        exactly `num_environments` entries in which any negative entry means
        "leave this slot alone".

        Every action is validated before any is applied. If any is illegal, or
        addresses a finished or unknown slot, nothing in the batch is mutated and
        the corresponding error is raised.
        """
        submitted = self._normalise_actions(actions)

        # -- validation pass: touch nothing ------------------------------
        for slot, action_id in submitted:
            record = self._slot(slot)
            if record.state.terminal:
                raise BatchTerminalStateError(
                    f"slot {slot} (environment {record.environment_id}, generation "
                    f"{record.generation}) is terminal ({record.state.terminal_reason}); "
                    "reset it before stepping"
                )
            legal = self._legal_actions(record)
            position = bisect_left(legal, action_id)
            if position >= len(legal) or legal[position] != action_id:
                raise BatchIllegalActionError(
                    f"action {action_id} is not legal in slot {slot} "
                    f"(environment {record.environment_id}, generation "
                    f"{record.generation}, game {record.game_id}); "
                    "no slot in the batch was modified"
                )

        # -- application pass: every action is known good ----------------
        events: dict[int, tuple[dict, ...]] = {}
        newly_terminal: list[int] = []
        outcomes: dict[int, SlotOutcome] = {}
        for slot, action_id in submitted:
            record = self._slots[slot]
            generated = apply_action(
                record.state, action_id, legal=record.cached_legal_actions
            )
            record.cached_legal_actions = None
            events[slot] = tuple(generated)
            if record.state.terminal:
                newly_terminal.append(slot)
                outcomes[slot] = self.outcome(slot)

        return BatchStepResult(
            stepped=tuple(slot for slot, _ in submitted),
            actions={slot: action_id for slot, action_id in submitted},
            events=events,
            newly_terminal=tuple(newly_terminal),
            outcomes=outcomes,
        )

    def _normalise_actions(
        self, actions: "Mapping[int, int] | Sequence[int]"
    ) -> list[tuple[int, int]]:
        """Convert either accepted action form into a sorted `(slot, action)` list."""
        if isinstance(actions, Mapping):
            submitted = [(int(slot), int(action)) for slot, action in actions.items()]
        else:
            dense = list(actions)
            if len(dense) != self.num_environments:
                raise ValueError(
                    f"a dense action vector must hold exactly {self.num_environments} "
                    f"entries, got {len(dense)}; use a {{slot: action}} mapping to step "
                    "a subset"
                )
            submitted = [
                (slot, int(action))
                for slot, action in enumerate(dense)
                if int(action) >= 0
            ]
        # Slot order makes the application sequence deterministic. Slots are
        # independent, so the order cannot change any individual result.
        submitted.sort()
        return submitted

    # -- terminal results --------------------------------------------------

    def outcome(self, slot: int) -> SlotOutcome:
        """Result/reason view of one slot; non-terminal slots report `not_terminal`."""
        record = self._slot(slot)
        state = record.state
        terminal = state.terminal
        return SlotOutcome(
            environment_id=record.environment_id,
            generation=record.generation,
            game_id=record.game_id,
            terminal=terminal,
            terminal_reason=state.terminal_reason if terminal else NOT_TERMINAL,
            winner=state.winner,
            is_draw=state.is_draw,
            total_moves=state.total_moves,
            battleless_moves=state.battleless_moves,
            result_for_red=state.result_for(PLAYERS[0]) if terminal else None,
            result_for_blue=state.result_for(PLAYERS[1]) if terminal else None,
        )

    def outcomes(self, slots: "Iterable[int] | None" = None) -> dict[int, SlotOutcome]:
        selected = (
            tuple(range(self.num_environments)) if slots is None else self._slot_list(slots)
        )
        return {slot: self.outcome(slot) for slot in selected}

    def finished_outcomes(self) -> dict[int, SlotOutcome]:
        return {slot: self.outcome(slot) for slot in self.finished_slots()}

    # -- independent reset -------------------------------------------------

    def reset_slots(self, slots: Iterable[int]) -> tuple[int, ...]:
        """Reset the selected slots into brand-new games.

        Returns the new `generation` of each reset slot, ordered by slot index.
        Slots that were not selected are not touched in any way: each reset
        replaces one slot's `EnvironmentSlot` with a freshly constructed one and
        reads nothing from the rest of the batch.

        Unknown slot indices are rejected before any slot is reset.
        """
        selected = sorted({int(slot) for slot in slots})
        for slot in selected:
            self._slot(slot)

        new_generations: list[int] = []
        for slot in selected:
            record = self._slots[slot]
            self._slots[slot] = self._build_slot(
                record.environment_id, record.generation + 1
            )
            new_generations.append(self._slots[slot].generation)
        return tuple(new_generations)

    def reset_finished(self) -> tuple[int, ...]:
        """Reset every finished slot. Returns the slots that were reset."""
        finished = self.finished_slots()
        self.reset_slots(finished)
        return finished

    # -- comparison --------------------------------------------------------

    def slot_fingerprint(self, slot: int, include_history: bool = True) -> tuple:
        """Canonical value covering slot identity plus the whole game state.

        Two slots with equal fingerprints are indistinguishable to every other
        part of the engine and to the batch layer.
        """
        record = self._slot(slot)
        return (
            record.environment_id,
            record.generation,
            record.seed,
            record.game_id,
            record.red_setup,
            record.blue_setup,
            state_fingerprint(record.state, include_history=include_history),
        )

    def batch_fingerprint(self, include_history: bool = True) -> tuple:
        return tuple(
            self.slot_fingerprint(slot, include_history=include_history)
            for slot in range(self.num_environments)
        )

    # -- privileged and serialisable extras --------------------------------
    #
    # Required by Agent 3 (compact trajectories) and by the browser service.
    # `belief_targets` is a *training target* and must never be fed to the
    # policy encoder; it is deliberately not part of `observations`.

    def belief_targets(self, slot: int, observer: int | None = None) -> list[dict]:
        """Ground-truth identities of the opponent pieces hidden from `observer`."""
        return belief_target(self._slot(slot).state, observer)

    def snapshot(self, slot: int, include_history: bool = False) -> dict:
        """Compact restorable snapshot of one slot's state."""
        return create_snapshot(self._slot(slot).state, include_history=include_history)

    def replay_record(self, slot: int, seeds: dict | None = None) -> ReplayRecord:
        """Replay record for the game currently in `slot`.

        The record carries the slot seed and identity, so the game can be
        regenerated from the record alone rather than only replayed.
        """
        record = self._slot(slot)
        payload = {
            "batch_interface_version": BATCH_INTERFACE_VERSION,
            "root_seed": self.root_seed,
            "environment_id": record.environment_id,
            "generation": record.generation,
            "slot_seed": record.seed,
        }
        payload.update(seeds or {})
        return build_replay_record(
            record.state, record.red_setup, record.blue_setup, seeds=payload
        )

    def public_board(self, slot: int, observer: int) -> dict:
        """Browser-safe board view for `observer`."""
        return public_board_view(self._slot(slot).state, observer)

    def public_setup(self, slot: int, observer: int) -> dict:
        return public_setup_view(self._slot(slot).state, observer)

    def public_events(self, slot: int, observer: int) -> list[dict]:
        """Observer-filtered public event stream of the current game."""
        return filter_events_for_observer(self._slot(slot).state.events, observer)

    def render(self, slot: int, observer: int | None = None) -> str:
        """Text board rendering for manual inspection."""
        return render_board(self._slot(slot).state, observer)


__all__ = [
    "BATCH_INTERFACE_VERSION",
    "NO_ACTING_PLAYER",
    "SKIP_ACTION",
    "BatchIllegalActionError",
    "BatchSimulationError",
    "BatchSimulator",
    "BatchStepResult",
    "BatchTerminalStateError",
    "EnvironmentSlot",
    "SlotOutcome",
    "UnknownEnvironmentError",
    "derive_slot_seed",
    "slot_game_id",
]
