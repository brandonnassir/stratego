"""Multiprocess CPU simulation layer over the frozen Phase 2.1 reference engine.

Specification sources:

- `03_game_engine_spec.md` section 16 (bulk-synchronous batch interface)
- `03_game_engine_spec.md` section 19 (loud failure, infrastructure recovery)
- `00_PHASE_3_SEQUENCE_AND_COMMON_CONTRACT.md` (approved architecture)

Architecture::

    coordinator process
        |
    persistent shared-memory arrays   (stratego.training.shared_buffers)
        |
    CPU simulation workers            (one BatchSimulator each)

Each worker owns a fixed, disjoint, contiguous range of environment slots for
the life of the pool, and slot `s` always holds `environment_id == s`. A worker
therefore reads and writes only its own rows, which is what makes the buffers
lock-free.

One bulk-synchronous step::

    coordinator writes actions / reset requests into shared memory
    -> coordinator releases the phase to every worker
    -> each worker reads its own actions, advances its games, resets what it
       must, republishes its own observations and legality
    -> each worker signals completion
    -> coordinator reads the new observations

What travels through the control pipes
--------------------------------------
Only small fixed-shape dictionaries: a command name, a sequence number, two
boolean flags, and a reply of counters and timings. No observation, legality
mask, action vector or game state is ever pickled. The bulk payload lives in
shared memory for the whole run and is allocated exactly once.

Failure surface
---------------
A worker that exits is detected through its process sentinel, a worker that
raises reports the exception and its traceback before exiting, and a worker that
stops responding is detected by the phase timeout. All three raise, and none of
them lets the coordinator continue reading buffers a worker has stopped
maintaining -- `publish_sequence` is checked after every phase for exactly that
reason. Production recovery (restarting a worker and rebuilding its slots) is
deliberately not implemented here; Agent 2 only has to make failure detectable.

Determinism
-----------
No process-local randomness exists. Every game is built from
`derive_slot_seed(root_seed, environment_id, generation)`, so a slot's content
depends on its identity and not on which worker holds it, how many workers there
are, or the order in which phases complete. The benchmark policy in this module
is likewise a pure function of `(root_seed, environment_id, generation, ply)`.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import signal
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.connection import wait as wait_for_objects

import numpy as np

from ..engine.constants import PLAYERS, RulesConfig, TRAINING_RULES
from .batch_simulation import BatchSimulator
from .reconstruction import (
    compare_digests,
    digest_live_decision,
    digest_reconstructed_decision,
    iter_reconstructed_decisions,
)
from .shared_buffers import (
    NO_ACTING_PLAYER,
    NO_WINNER,
    POLICY_CAPACITY,
    SKIP_ACTION,
    STATUS_ACTIVE,
    STATUS_TERMINAL,
    SharedBufferDescriptor,
    SharedEnvironmentBuffers,
    max_resident_bytes,
    terminal_reason_code,
    terminal_reason_name,
)
from .trajectory import (
    DEFAULT_SNAPSHOT_INTERVAL,
    GameTrajectoryBuilder,
    builder_for_slot,
    decode_game_record,
    decode_game_record_compressed,
    encode_game_record,
    encode_game_record_compressed,
)

WORKER_POOL_VERSION = "worker_pool_v1"

#: Collection policy identifier stamped into every decision recorded through the
#: end-to-end pipeline. It is deliberately distinct from Agent 3's
#: `synthetic_hash_policy_v1`: these decisions come from the coordinator's
#: representative model on Metal, not from the synthetic hash policy, and a
#: training consumer must be able to tell the two corpora apart.
DEFAULT_COLLECTION_POLICY_VERSION = "end_to_end_representative_probe_v1"

#: Environment variables that make numerical libraries spawn their own thread
#: pools. A simulation worker is single-threaded Python; letting NumPy's backend
#: open one pool per worker would oversubscribe the machine and make the CPU
#: scaling screen measure thread thrash instead of worker scaling.
THREAD_LIMIT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_MASK64 = 0xFFFFFFFFFFFFFFFF
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15
_MIX_A = 0xBF58476D1CE4E5B9
_MIX_B = 0x94D049BB133111EB


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WorkerPoolError(RuntimeError):
    """Base class for every infrastructure failure of the simulation pool."""


class WorkerCrashError(WorkerPoolError):
    """A worker process exited without completing the phase."""


class WorkerTimeoutError(WorkerPoolError):
    """A worker did not complete the phase within the timeout."""


class WorkerFaultError(WorkerPoolError):
    """A worker raised. Carries the remote exception and traceback."""

    def __init__(self, message: str, *, worker_id: int, remote_traceback: str) -> None:
        super().__init__(message)
        self.worker_id = worker_id
        self.remote_traceback = remote_traceback


class StaleBufferError(WorkerPoolError):
    """A phase completed without every slot being republished.

    Reading on would mean feeding the model an observation no worker refreshed,
    so this is a hard error rather than a warning.
    """


# ---------------------------------------------------------------------------
# Deterministic benchmark policy
# ---------------------------------------------------------------------------
#
# A cheap, seeded, uniform choice among the legal actions. It exists so the CPU
# scaling benchmark and the cross-process equivalence run can select actions
# without a model. Two forms are provided and are required to agree:
#
# - `select_action` picks from an explicit legal-action list (reference side);
# - `select_actions` picks from the dense shared masks for the whole batch at
#   once (coordinator side, which is all a coordinator has).


def _splitmix64(value: int) -> int:
    """One SplitMix64 round on a Python integer."""
    value = (value + _GOLDEN_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _MIX_A) & _MASK64
    value = ((value ^ (value >> 27)) * _MIX_B) & _MASK64
    return value ^ (value >> 31)


def _splitmix64_array(value: np.ndarray) -> np.ndarray:
    """The same round, elementwise, on a `uint64` array."""
    value = value + np.uint64(_GOLDEN_GAMMA)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(_MIX_A)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(_MIX_B)
    return value ^ (value >> np.uint64(31))


def slot_hash(root_seed: int, environment_id: int, generation: int, ply: int) -> int:
    """Deterministic 64-bit value for one decision point.

    Depends on nothing but its arguments, so the coordinator, a worker and an
    independently stepped reference game all derive the same value.
    """
    digest = _splitmix64(int(root_seed) & _MASK64)
    for value in (environment_id, generation, ply):
        digest = _splitmix64(digest ^ (int(value) & _MASK64))
    return digest


def slot_hashes(
    root_seed: int,
    environment_id: np.ndarray,
    generation: np.ndarray,
    ply: np.ndarray,
) -> np.ndarray:
    """Vectorised :func:`slot_hash` over a whole batch."""
    digest = np.full(
        environment_id.shape, _splitmix64(int(root_seed) & _MASK64), dtype=np.uint64
    )
    for value in (environment_id, generation, ply):
        digest = _splitmix64_array(digest ^ value.astype(np.uint64))
    return digest


def select_action(
    root_seed: int,
    environment_id: int,
    generation: int,
    ply: int,
    legal: "list[int] | np.ndarray",
) -> int:
    """Deterministic choice from an explicit legal-action list."""
    count = len(legal)
    if count == 0:
        return SKIP_ACTION
    return int(legal[slot_hash(root_seed, environment_id, generation, ply) % count])


def select_actions(
    buffers: SharedEnvironmentBuffers, root_seed: int, out: "np.ndarray | None" = None
) -> np.ndarray:
    """Deterministic choice for every active slot, straight from the dense masks.

    This is the coordinator's real cost of the dense-legality transport: it has
    10,000 bits per environment and no legal-action list, exactly as a model
    would. Two things make it as cheap as that transport allows:

    - one `flatnonzero` over the whole mask buffer rather than one per row, with
      the per-row start of the flat index recovered by `searchsorted` because
      the result is sorted;
    - a zero-copy `bool` reinterpretation of the `uint8` mask, which is the same
      bytes but takes NumPy's boolean `nonzero` path -- roughly four times
      faster than scanning the identical memory as `uint8`. The stored dtype
      stays `uint8` so the published contract is unchanged.
    """
    masks = buffers.legal_mask
    num_environments, action_space = masks.shape
    if out is None:
        out = np.empty(num_environments, dtype=np.int32)
    out.fill(SKIP_ACTION)

    counts = buffers.legal_count.astype(np.int64)
    active = np.flatnonzero((buffers.status == STATUS_ACTIVE) & (counts > 0))
    if active.size == 0:
        return out

    flat = np.flatnonzero(masks.view(np.bool_))
    rows = np.arange(num_environments, dtype=np.int64)
    starts = np.searchsorted(flat, rows * action_space)

    offsets = slot_hashes(
        root_seed, buffers.environment_id, buffers.generation, buffers.ply
    )[active] % counts[active].astype(np.uint64)
    picked = flat[starts[active] + offsets.astype(np.int64)]
    out[active] = (picked - active * action_space).astype(np.int32)
    return out


def offer_to_reservoir(
    rng: random.Random, retained: list, item, *, capacity: int, seen: int
) -> None:
    """Standard reservoir sampling: keep `capacity` of `seen` items, uniformly.

    Used for the sample of encoded records a run hands back. The obvious
    alternative -- keep the first `capacity` and drop the rest -- looks harmless
    and is not: a pool starts with every slot at ply 0, so the first games to
    seal are the shortest games of the whole run. A caller asking for six sample
    records to reconstruct, or to measure bytes on, would get six ten-ply games
    when a production game runs to about five hundred plies.

    `seen` is the total number of items offered so far, this one included.
    """
    if capacity <= 0:
        return
    if len(retained) < capacity:
        retained.append(item)
        return
    index = rng.randrange(seen)
    if index < capacity:
        retained[index] = item


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerAssignment:
    """The half-open slot range one worker owns for the life of the pool."""

    worker_id: int
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start

    def as_dict(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "start": self.start,
            "stop": self.stop,
            "size": self.size,
        }


def partition_environments(
    num_environments: int, num_workers: int
) -> tuple[WorkerAssignment, ...]:
    """Split slots into contiguous, disjoint, near-equal worker ranges."""
    if num_workers < 1:
        raise ValueError("a pool needs at least one worker")
    if num_environments < num_workers:
        raise ValueError(
            f"{num_environments} environments cannot be split across "
            f"{num_workers} workers; every worker must own at least one slot"
        )
    base, remainder = divmod(num_environments, num_workers)
    assignments: list[WorkerAssignment] = []
    cursor = 0
    for worker_id in range(num_workers):
        size = base + (1 if worker_id < remainder else 0)
        assignments.append(WorkerAssignment(worker_id, cursor, cursor + size))
        cursor += size
    assert cursor == num_environments
    return tuple(assignments)


@dataclass(frozen=True)
class RecordingConfig:
    """How a worker turns the coordinator's decisions into trajectory records.

    Recording happens *inside* the worker because the coordinator holds no game
    object and must not be sent one. The policy and value that belong in a
    record arrive through the coordinator-written shared fields, so enabling
    this adds no traffic to the control pipes.

    `verify_target_decisions` is a per-worker budget. A game selected for
    verification carries live digests through its whole life, which is far more
    expensive than recording alone, so only enough games to meet the budget are
    selected. Verification round-trips the record through the codec before
    reconstructing it, which is strictly stronger than reconstructing the
    in-memory object.
    """

    enabled: bool = False
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL
    collection_policy_version: str = DEFAULT_COLLECTION_POLICY_VERSION
    collection_checkpoint_id: str | None = None
    #: Encode every sealed record. This is what a real collection run pays, and
    #: the byte total is what a storage estimate is built from. The payload is
    #: dropped immediately afterwards unless `retain_games` asks for it.
    encode_records: bool = True
    compress_records: bool = False
    verify_target_decisions: int = 0
    #: How many games one worker may carry live digests for at the same time.
    #: Digesting costs roughly 35x what plain recording costs, so this is what
    #: keeps a verification budget from being spent all at once across the whole
    #: slot range.
    max_concurrent_verifications: int = 2
    #: Encoded payloads handed back at shutdown so a caller can inspect real
    #: records without keeping the whole run on disk.
    retain_games: int = 0
    #: Compare the shared `legal_count` against the simulator's own legal-action
    #: list on every recorded decision. Cheap, and it is the check that would
    #: catch a coordinator reading a stale or misaligned row.
    check_shared_legality: bool = True
    #: Phase 6B: where sealed records are *persisted*. `None` keeps the accepted
    #: Phase 3/6 behaviour exactly -- records are encoded, measured and dropped,
    #: and no worker touches the filesystem. When set, each worker appends its
    #: own sealed records to its own shard files under this directory; see
    #: `stratego.training.shard_writer` for why the writer is per-worker and
    #: synchronous.
    output_directory: str | None = None
    #: Rollover size for one shard file. Ignored when `output_directory` is None.
    shard_target_bytes: int = 128 * 1024 * 1024
    #: Stamped into shard names so two runs writing to one directory cannot
    #: collide and so a shard can be traced back to the run that produced it.
    run_id: str = "run"


class _WorkerRuntime:
    """The worker process's own state. Never touched by the coordinator."""

    def __init__(
        self,
        descriptor: SharedBufferDescriptor,
        assignment: WorkerAssignment,
        root_seed: int,
        rules: RulesConfig,
        recording: RecordingConfig | None = None,
    ) -> None:
        self.assignment = assignment
        self.root_seed = int(root_seed)
        self.buffers = SharedEnvironmentBuffers.attach(descriptor)
        self.view = self.buffers.view(assignment.start, assignment.stop)
        # `first_environment_id` makes this worker's environment identifiers the
        # global slot indices, so identity does not depend on the partitioning.
        self.simulator = BatchSimulator(
            assignment.size,
            root_seed=root_seed,
            rules=rules,
            first_environment_id=assignment.start,
        )
        self.observation_builds = 0
        self.transitions = 0
        self.terminals = 0
        self.resets = 0
        # Games that were terminal at creation (`phase2_1_reference_1.2.0`): a
        # legal random setup can strand the first player at ply 0. Counted so a
        # run can report them; their outcomes and records flow through the
        # ordinary sealing paths.
        self.stillborn_games = 0

        # -- recording state --------------------------------------------------
        self.recording = recording or RecordingConfig()
        # slot -> (identity key, builder or None). The key is what makes a reset
        # start a new record instead of appending to the finished game's.
        self._builders: dict[int, tuple[tuple[int, int], GameTrajectoryBuilder | None]] = {}
        # slot -> live digests, only for games selected for verification.
        self._live_digests: dict[int, list] = {}
        self.retained_records: list[bytes] = []
        # Games sealed so far, and the generator that decides which of them the
        # retained sample keeps. Seeded from the pool's root seed and this
        # worker's identity, so the sample is reproducible for a given run
        # without depending on global randomness.
        self.games_sealed = 0
        self._retention_rng = random.Random(
            (int(root_seed) << 16) ^ (assignment.worker_id + 1)
        )
        self.decisions_recorded = 0
        self.games_recorded = 0
        self.record_bytes = 0
        self.snapshot_bytes = 0
        self.snapshot_count = 0
        self.verified_games = 0
        self.verified_decisions = 0
        self.reconstruction_mismatches = 0
        self.mismatch_details: list[dict] = []
        self.recording_seconds = 0.0
        self.verification_seconds = 0.0
        # Phase 6B persistence. Absent unless an output directory is configured,
        # so the accepted encode-and-drop path is untouched by default and a
        # worker still performs no file I/O.
        self.shard_writer = None
        if self.recording.enabled and self.recording.output_directory:
            from .shard_writer import ShardWriter

            self.shard_writer = ShardWriter(
                self.recording.output_directory,
                worker_id=assignment.worker_id,
                run_id=self.recording.run_id,
                compress_records=self.recording.compress_records,
                target_bytes=self.recording.shard_target_bytes,
                collection_policy_version=self.recording.collection_policy_version,
            )
        # A game that was already in progress when recording began cannot be
        # recorded from ply 0, so it is skipped and counted rather than stored
        # as a partial trajectory. Enabling recording at pool start keeps this 0.
        self.games_joined_late = 0

    # -- publishing ----------------------------------------------------------

    def publish(self) -> None:
        """Write every owned slot's current observation/legality/metadata."""
        view = self.view
        simulator = self.simulator
        worker_id = self.assignment.worker_id
        for local in range(simulator.num_environments):
            state = simulator.game_state(local)
            if state.terminal:
                # A terminal slot has no player to move, so publishing an
                # observation for it would mean publishing a perspective that
                # does not exist. Zero it instead and say so in `status`.
                view["observations"][local].fill(0.0)
                view["legal_mask"][local].fill(0)
                view["legal_count"][local] = 0
                view["acting_player"][local] = NO_ACTING_PLAYER
                view["terminal"][local] = 1
                view["status"][local] = STATUS_TERMINAL
            else:
                view["observations"][local] = simulator.observation(local)
                view["legal_mask"][local] = simulator.legal_action_mask(local)
                legal_count = len(simulator.legal_actions(local))
                if legal_count == 0:
                    # Phase 6B, first soak, t=8,981s: the coordinator sampled
                    # from a published all-zero mask -- an active slot with no
                    # legal action, which the engine's transition-time mobility
                    # check should make impossible. The reader-side detection
                    # saw only the symptom; this writer-side check captures the
                    # contradiction with the state still in hand, so one
                    # occurrence is a complete diagnosis rather than a mystery.
                    from ..engine.legal_moves import has_legal_action

                    raise WorkerPoolError(
                        "publish would mark an active slot with zero legal "
                        f"actions: slot {self.assignment.start + local}, game "
                        f"{state.game_id!r}, generation "
                        f"{simulator.generation(local)}, ply {state.total_moves}, "
                        f"acting_player {state.acting_player}, terminal "
                        f"{state.terminal} ({state.terminal_reason!r}), "
                        f"has_legal_action(acting)="
                        f"{has_legal_action(state, state.acting_player)}, "
                        f"battleless {state.battleless_moves}, last actions "
                        f"{list(state.action_history[-6:])}"
                    )
                view["legal_count"][local] = legal_count
                view["acting_player"][local] = state.acting_player
                view["terminal"][local] = 0
                view["status"][local] = STATUS_ACTIVE
                self.observation_builds += 1
            view["environment_id"][local] = simulator.environment_id(local)
            view["generation"][local] = simulator.generation(local)
            view["ply"][local] = state.total_moves
            view["battleless_moves"][local] = state.battleless_moves
            view["worker_id"][local] = worker_id
            view["publish_sequence"][local] += 1

    def record_outcome(self, local: int) -> None:
        """Persist the result of a game that has just finished.

        Kept in its own `last_*` fields so an immediate reset cannot erase the
        outcome before the coordinator has read it.
        """
        outcome = self.simulator.outcome(local)
        view = self.view
        view["episode_count"][local] += 1
        view["last_terminal_reason"][local] = terminal_reason_code(
            outcome.terminal_reason
        )
        view["last_winner"][local] = (
            NO_WINNER if outcome.winner is None else outcome.winner
        )
        view["last_is_draw"][local] = 1 if outcome.is_draw else 0
        view["last_total_moves"][local] = outcome.total_moves
        view["last_generation"][local] = outcome.generation
        view["last_result_red"][local] = outcome.result_for_red or 0.0
        view["last_result_blue"][local] = outcome.result_for_blue or 0.0

    # -- trajectory recording ------------------------------------------------

    def _wants_verification(self) -> bool:
        """Whether to carry live digests through the game starting right now.

        The budget only advances when a game *ends*, so gating on it alone would
        select every slot in flight at once and overshoot by the batch width.
        Capping the number of games being digested concurrently keeps the cost
        of the correctness gate proportional to what it actually verifies.
        """
        if self.verified_decisions >= self.recording.verify_target_decisions:
            return False
        return len(self._live_digests) < self.recording.max_concurrent_verifications

    def record_decisions(self) -> None:
        """Fold the coordinator's decisions into each slot's trajectory.

        Called at the top of a phase, before any action is applied, because a
        decision belongs to the position it was taken in. Every slot the
        coordinator marked `decision_valid` is recorded against the live state
        the coordinator saw.
        """
        started = time.perf_counter()
        view = self.view
        valid = view["decision_valid"]
        config = self.recording
        for local in np.flatnonzero(valid).tolist():
            state = self.simulator.game_state(local)
            if state.terminal:
                # The coordinator should never mark a terminal slot; if it does,
                # that is a bug worth surfacing rather than a record to write.
                raise WorkerPoolError(
                    f"slot {self.assignment.start + local} is terminal but the "
                    f"coordinator marked it decision_valid"
                )

            key = (
                self.simulator.environment_id(local),
                self.simulator.generation(local),
            )
            entry = self._builders.get(local)
            if entry is None or entry[0] != key:
                if state.total_moves != 0:
                    # Recording began mid-game. A partial trajectory is not a
                    # trajectory, so this game is skipped for its whole life.
                    self._builders[local] = (key, None)
                    self.games_joined_late += 1
                    continue
                builder = builder_for_slot(
                    self.simulator,
                    local,
                    snapshot_interval=config.snapshot_interval,
                    collection_policy_version=config.collection_policy_version,
                    collection_checkpoint_id=config.collection_checkpoint_id,
                )
                self._builders[local] = (key, builder)
                if self._wants_verification():
                    self._live_digests[local] = []
                entry = self._builders[local]

            builder = entry[1]
            if builder is None:
                continue

            legal = self.simulator.legal_actions(local)
            count = len(legal)
            if config.check_shared_legality and count != int(view["legal_count"][local]):
                raise WorkerPoolError(
                    f"slot {self.assignment.start + local}: shared legal_count "
                    f"{int(view['legal_count'][local])} does not match the "
                    f"engine's {count} legal actions"
                )
            if count > POLICY_CAPACITY:
                raise WorkerPoolError(
                    f"slot {self.assignment.start + local} has {count} legal "
                    f"actions, above the POLICY_CAPACITY of {POLICY_CAPACITY}"
                )

            value = view["value_prediction"][local]
            decision = builder.record_decision(
                state,
                legal_action_ids=legal,
                probabilities=view["policy_probabilities"][local, :count].tolist(),
                win_draw_loss_prediction=(
                    float(value[0]),
                    float(value[1]),
                    float(value[2]),
                ),
                selected_action_id=int(view["actions"][local]),
            )
            self.decisions_recorded += 1

            digests = self._live_digests.get(local)
            if digests is not None:
                digests.append(
                    digest_live_decision(
                        state,
                        decision,
                        environment_id=key[0],
                        generation=key[1],
                        dense_mask=True,
                        legal_action_ids=legal,
                    )
                )
        self.recording_seconds += time.perf_counter() - started

    def _record_stillborn(self, local: int) -> None:
        """Count and seal a game that was terminal at creation.

        `phase2_1_reference_1.2.0` can create a slot whose first player is
        stranded: the game is decided before any decision exists. It is a
        completed game all the same -- its outcome feeds `episode_count` and
        the `last_*` fields exactly like any transition-terminal game, and when
        recording is enabled its zero-decision record is sealed through the
        ordinary path so the game is persisted with its setups, winner and
        terminal reason rather than silently vanishing at the reset.
        """
        self.record_outcome(local)
        self.stillborn_games += 1
        if not self.recording.enabled:
            return
        key = (
            self.simulator.environment_id(local),
            self.simulator.generation(local),
        )
        self._builders[local] = (
            key,
            builder_for_slot(
                self.simulator,
                local,
                snapshot_interval=self.recording.snapshot_interval,
                collection_policy_version=self.recording.collection_policy_version,
                collection_checkpoint_id=self.recording.collection_checkpoint_id,
            ),
        )
        self.finalise_recording(local)

    def finalise_recording(self, local: int) -> None:
        """Seal, optionally verify, and discard the record of a finished game."""
        entry = self._builders.pop(local, None)
        digests = self._live_digests.pop(local, None)
        if entry is None or entry[1] is None:
            return
        started = time.perf_counter()
        config = self.recording
        record = entry[1].finish(self.simulator.game_state(local))
        self.games_recorded += 1
        self.games_sealed += 1
        self.snapshot_bytes += record.snapshot_bytes
        self.snapshot_count += len(record.snapshots)

        payload: bytes | None = None
        if self.shard_writer is not None:
            # Phase 6B: the writer owns the encode and the compress, so the
            # record is serialised once and the produced/compressed/persisted
            # byte totals all come from the same pass. `record_bytes` keeps
            # meaning *uncompressed bytes produced*, so it stays comparable to
            # every Phase 3/4/6 storage figure; what landed on disk is reported
            # separately from the writer's own stats.
            accounting = self.shard_writer.write(record)
            self.record_bytes += accounting["uncompressed_bytes"]
            if config.retain_games or digests:
                # Only re-serialise when something actually needs the bytes: the
                # retention reservoir, or a verification that has to round-trip
                # the record through the codec.
                payload = (
                    encode_game_record_compressed(record)
                    if config.compress_records
                    else encode_game_record(record)
                )
                offer_to_reservoir(
                    self._retention_rng,
                    self.retained_records,
                    payload,
                    capacity=config.retain_games,
                    seen=self.games_sealed,
                )
        elif config.encode_records:
            payload = (
                encode_game_record_compressed(record)
                if config.compress_records
                else encode_game_record(record)
            )
            self.record_bytes += len(payload)
            offer_to_reservoir(
                self._retention_rng,
                self.retained_records,
                payload,
                capacity=config.retain_games,
                seen=self.games_sealed,
            )
        self.recording_seconds += time.perf_counter() - started

        if digests:
            self._verify_record(record, payload, digests)

    def _verify_record(self, record, payload: bytes | None, digests: list) -> None:
        """Reconstruct a whole game through Agent 3's path and compare digests.

        The record is round-tripped through the codec first when an encoded
        payload exists, so what gets reconstructed is what storage would hand
        back rather than the object that is still in memory.
        """
        started = time.perf_counter()
        if payload is not None:
            record = (
                decode_game_record_compressed(payload)
                if self.recording.compress_records
                else decode_game_record(payload)
            )
        by_ply = {digest.ply: digest for digest in digests}
        rebuilt_iterator = iter_reconstructed_decisions(
            record, sorted(by_ply), dense_mask=True, copy_state=False
        )
        for rebuilt in rebuilt_iterator:
            live = by_ply[rebuilt.ply]
            decision = record.decision_at(rebuilt.ply)
            mismatches = compare_digests(
                live, digest_reconstructed_decision(rebuilt, decision)
            )
            self.verified_decisions += 1
            if mismatches:
                self.reconstruction_mismatches += len(mismatches)
                if len(self.mismatch_details) < 20:
                    self.mismatch_details.append(
                        {
                            "game_id": record.game_id,
                            "environment_id": record.environment_id,
                            "generation": record.generation,
                            "ply": rebuilt.ply,
                            "mismatches": [
                                {"category": category, "field": field}
                                for category, field in mismatches
                            ],
                        }
                    )
            if decision.collection_policy_version != (
                self.recording.collection_policy_version
            ):
                self.reconstruction_mismatches += 1
                if len(self.mismatch_details) < 20:
                    self.mismatch_details.append(
                        {
                            "game_id": record.game_id,
                            "ply": rebuilt.ply,
                            "mismatches": [
                                {
                                    "category": "collection_policy_version",
                                    "field": decision.collection_policy_version,
                                }
                            ],
                        }
                    )
        self.verified_games += 1
        self.verification_seconds += time.perf_counter() - started

    def recording_counters(self) -> dict:
        # Absent entirely when recording is off, so a pool that does not record
        # sends exactly the reply Agent 2 specified. These are all scalars: the
        # reply stays a small fixed-shape dictionary either way.
        if not self.recording.enabled:
            return {}
        persistence = {}
        if self.shard_writer is not None:
            stats = self.shard_writer.stats
            persistence = {
                "total_persisted_bytes": stats.bytes_written,
                "total_compressed_bytes": stats.compressed_bytes,
                "total_shards_opened": stats.shards_opened,
                "total_shards_closed": stats.shards_closed,
                "total_records_persisted": stats.records_written,
                "total_encode_seconds": stats.encode_seconds,
                "total_compress_seconds": stats.compress_seconds,
                "total_write_seconds": stats.write_seconds,
                "total_flush_seconds": stats.flush_seconds,
                "total_write_errors": stats.write_errors,
                # Synchronous per-worker writes: the bytes are on the filesystem
                # before the sealing call returns, so there is no queue.
                "total_pending_records": 0,
                "total_pending_bytes": 0,
            }
        return {
            "total_decisions_recorded": self.decisions_recorded,
            "total_games_recorded": self.games_recorded,
            "total_record_bytes": self.record_bytes,
            **persistence,
            "total_snapshot_bytes": self.snapshot_bytes,
            "total_snapshot_count": self.snapshot_count,
            "total_verified_games": self.verified_games,
            "total_verified_decisions": self.verified_decisions,
            "total_reconstruction_mismatches": self.reconstruction_mismatches,
            "total_games_joined_late": self.games_joined_late,
            "recording_seconds": self.recording_seconds,
            "verification_seconds": self.verification_seconds,
        }

    # -- commands ------------------------------------------------------------

    def counters(self, observation_builds_before: int) -> dict:
        """Per-phase counters plus the run totals.

        The per-phase values are what a throughput measurement needs; the
        cumulative `total_*` values let the coordinator reconcile a whole run
        without summing every phase.
        """
        return {
            "observation_builds": self.observation_builds - observation_builds_before,
            "total_observation_builds": self.observation_builds,
            "total_transitions": self.transitions,
            "total_terminals": self.terminals,
            "total_resets": self.resets,
            "total_stillborn_games": self.stillborn_games,
            **self.recording_counters(),
        }

    def step(self, apply_actions: bool, auto_reset: bool) -> dict:
        """One bulk-synchronous phase for this worker's slots."""
        observation_builds_before = self.observation_builds
        stepped = 0
        terminals = 0
        if apply_actions:
            # A decision describes the position it was taken in, so it has to be
            # stored before the action that leaves that position is applied.
            if self.recording.enabled:
                self.record_decisions()
            # The dense action view is exactly `size` entries, which is the form
            # `BatchSimulator.step` accepts; a negative entry skips the slot.
            result = self.simulator.step(self.view["actions"])
            stepped = len(result.stepped)
            self.transitions += stepped
            for local in result.newly_terminal:
                self.record_outcome(local)
                if self.recording.enabled:
                    # Sealed on the terminal state, before any reset can move
                    # the slot on to its next game.
                    self.finalise_recording(local)
            terminals = len(result.newly_terminal)
            self.terminals += terminals

        requested = np.flatnonzero(self.view["reset_request"]).tolist()
        to_reset = set(requested)
        if auto_reset:
            to_reset.update(self.simulator.finished_slots())
        if to_reset:
            # A game terminal at creation reaches its reset having never been
            # stepped: no decision exists and `record_outcome` never ran for
            # it. Seal its outcome and (when recording) its zero-decision
            # record now, before the reset replaces the state. Ply 0 is what
            # identifies it -- a game that ended through a transition was
            # sealed in the `newly_terminal` loop above, in an earlier step's
            # loop, or at startup it cannot be (a stepped game has moves).
            for local in sorted(to_reset):
                state = self.simulator.game_state(local)
                if state.terminal and state.total_moves == 0:
                    self._record_stillborn(local)
            self.simulator.reset_slots(sorted(to_reset))
            self.resets += len(to_reset)
            if self.recording.enabled:
                # A reset ends whatever the slot was holding. Dropping the
                # builder here means an abandoned game cannot leak into the next
                # generation's record; the identity key would catch it anyway,
                # but this also releases the memory immediately.
                for local in to_reset:
                    self._builders.pop(local, None)
                    self._live_digests.pop(local, None)

        self.publish()
        return {
            "stepped": stepped,
            "terminals": terminals,
            "resets": len(to_reset),
            **self.counters(observation_builds_before),
        }

    def close(self) -> None:
        # The open shard is finished first, so its manifest exists and the run
        # ends with no shard left unclosed. This runs in the worker's `finally`,
        # so it happens on a clean shutdown and on a fault alike.
        if self.shard_writer is not None:
            try:
                self.shard_writer.close()
            except Exception:  # pragma: no cover - shutdown must not mask a fault
                pass
        # Every view has to be dropped before the mapping, or the exported
        # buffers keep the shared block alive.
        self.view = {}
        self.buffers.close()


def _worker_main(
    connection: Connection,
    descriptor: SharedBufferDescriptor,
    assignment: WorkerAssignment,
    root_seed: int,
    rules: RulesConfig,
    recording: RecordingConfig | None = None,
) -> None:
    """Entry point of a simulation worker process."""
    runtime: _WorkerRuntime | None = None
    started = time.perf_counter()
    try:
        runtime = _WorkerRuntime(descriptor, assignment, root_seed, rules, recording)
        runtime.publish()
        connection.send(
            {
                "kind": "ready",
                "worker_id": assignment.worker_id,
                "sequence": 0,
                "pid": os.getpid(),
                # Reported so the coordinator can prove the children really
                # inherited the single-thread numerical-library settings rather
                # than assuming they did.
                "thread_limits": ",".join(
                    os.environ.get(name, "") for name in THREAD_LIMIT_VARIABLES
                ),
                "busy_seconds": time.perf_counter() - started,
                "process_seconds": time.process_time(),
                "max_rss_bytes": max_resident_bytes(),
                "stepped": 0,
                "terminals": 0,
                "resets": 0,
                **runtime.counters(0),
            }
        )

        while True:
            command = connection.recv()
            kind = command["kind"]
            if kind == "shutdown":
                # Finish the open shard *before* reporting, so the manifest for
                # the last shard exists and `total_shards_closed` in this reply
                # is the final number rather than one short. `close` is
                # idempotent, so the `finally` below is still safe.
                if runtime.shard_writer is not None:
                    runtime.shard_writer.close()
                connection.send(
                    {
                        "kind": "shutdown_ack",
                        "worker_id": assignment.worker_id,
                        "sequence": command["sequence"],
                        "process_seconds": time.process_time(),
                        "max_rss_bytes": max_resident_bytes(),
                        # The only phase in which a worker may return bulk data,
                        # and only the handful of records `retain_games` asked
                        # for. Everything else recorded this run was encoded,
                        # counted and dropped.
                        "retained_records": tuple(runtime.retained_records),
                        "mismatch_details": tuple(runtime.mismatch_details),
                        **runtime.counters(runtime.observation_builds),
                    }
                )
                break

            phase_started = time.perf_counter()
            builds_before = runtime.observation_builds
            if kind == "step":
                counters = runtime.step(
                    apply_actions=command["apply_actions"],
                    auto_reset=command["auto_reset"],
                )
            elif kind == "publish":
                runtime.publish()
                counters = {
                    "stepped": 0,
                    "terminals": 0,
                    "resets": 0,
                    **runtime.counters(builds_before),
                }
            elif kind == "stall":
                # Test-only fault injection: lets the hang-detection path be
                # exercised without killing a process.
                time.sleep(float(command["seconds"]))
                counters = {
                    "stepped": 0,
                    "terminals": 0,
                    "resets": 0,
                    **runtime.counters(builds_before),
                }
            else:
                raise WorkerPoolError(f"unknown worker command {kind!r}")

            connection.send(
                {
                    "kind": "phase_complete",
                    "worker_id": assignment.worker_id,
                    "sequence": command["sequence"],
                    "busy_seconds": time.perf_counter() - phase_started,
                    "process_seconds": time.process_time(),
                    "max_rss_bytes": max_resident_bytes(),
                    **counters,
                }
            )
    except BaseException as error:  # noqa: BLE001 - the coordinator must see it
        try:
            connection.send(
                {
                    "kind": "error",
                    "worker_id": assignment.worker_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:  # pragma: no cover - pipe already gone
            pass
    finally:
        if runtime is not None:
            runtime.close()
        try:
            connection.close()
        except Exception:  # pragma: no cover - defensive
            pass


# ---------------------------------------------------------------------------
# Coordinator side
# ---------------------------------------------------------------------------


@dataclass
class _Worker:
    assignment: WorkerAssignment
    process: mp.process.BaseProcess
    connection: Connection
    process_seconds: float = 0.0
    max_rss_bytes: int = 0
    pid: int = -1


@dataclass(frozen=True)
class PhaseReport:
    """What one bulk-synchronous phase did and where its time went."""

    sequence: int
    stepped: int
    terminals: int
    resets: int
    observation_builds: int
    wall_seconds: float
    dispatch_seconds: float
    wait_seconds: float
    straggler_seconds: float
    worker_busy_seconds: float
    worker_cpu_seconds: float


class WorkerPool:
    """Coordinator side of the multiprocess CPU simulation layer.

    Owns the shared buffers and the worker processes, and drives the phases.
    It never simulates anything itself and never receives a game object.
    """

    def __init__(
        self,
        num_environments: int,
        num_workers: int,
        *,
        root_seed: int = 0,
        rules: RulesConfig = TRAINING_RULES,
        step_timeout: float = 120.0,
        start_timeout: float = 300.0,
        start_method: str = "spawn",
        limit_worker_threads: bool = True,
        recording: RecordingConfig | None = None,
    ) -> None:
        self.num_environments = int(num_environments)
        self.num_workers = int(num_workers)
        self.root_seed = int(root_seed)
        self.rules = rules
        self.step_timeout = float(step_timeout)
        self.start_timeout = float(start_timeout)
        self.start_method = start_method
        self.limit_worker_threads = limit_worker_threads
        self.recording = recording or RecordingConfig()

        self.assignments = partition_environments(num_environments, num_workers)
        self.buffers: SharedEnvironmentBuffers | None = None
        self._workers: list[_Worker] = []
        self._sequence = 0
        self._expected_publish = np.zeros(self.num_environments, dtype=np.int64)
        self._started = False
        self._closed = False
        self._pending_resets = False
        self.startup_seconds = 0.0
        # The most recent control-channel traffic, kept so a test or the
        # acceptance harness can measure how little of it there is. These are
        # the only objects that ever cross a pipe.
        self.last_command: dict | None = None
        self.last_replies: tuple[dict, ...] = ()
        # Filled at startup from what the workers actually see in their own
        # environment, in `THREAD_LIMIT_VARIABLES` order.
        self.worker_thread_limits: dict[int, str] = {}

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Allocate the buffers, spawn the workers and wait for the first publish."""
        if self._started:
            raise WorkerPoolError("this pool has already been started")
        started = time.perf_counter()
        self.buffers = SharedEnvironmentBuffers.create(self.num_environments)
        context = mp.get_context(self.start_method)

        previous_environment = self._apply_thread_limits()
        try:
            for assignment in self.assignments:
                parent_connection, child_connection = context.Pipe(duplex=True)
                process = context.Process(
                    target=_worker_main,
                    args=(
                        child_connection,
                        self.buffers.descriptor,
                        assignment,
                        self.root_seed,
                        self.rules,
                        self.recording,
                    ),
                    name=f"stratego-worker-{assignment.worker_id}",
                    daemon=True,
                )
                process.start()
                # The coordinator must drop its copy of the child's end, or a
                # dead worker never produces an end-of-file and a crash looks
                # like a hang.
                child_connection.close()
                self._workers.append(
                    _Worker(
                        assignment=assignment,
                        process=process,
                        connection=parent_connection,
                    )
                )
            self._started = True
            replies = self._await_replies(self.start_timeout, stage="startup")
        finally:
            self._restore_thread_limits(previous_environment)

        for reply in replies.values():
            worker = self._workers[reply["worker_id"]]
            worker.pid = reply["pid"]
        self.worker_thread_limits = {
            reply["worker_id"]: reply["thread_limits"] for reply in replies.values()
        }
        self._expected_publish.fill(1)
        self._check_published()
        self.startup_seconds = time.perf_counter() - started

    def _apply_thread_limits(self) -> dict[str, str | None]:
        """Pin numerical-library thread pools to one thread in the children.

        A spawned child inherits the parent's environment, and the variables
        have to be set before the child imports NumPy, so they are set here and
        restored once every worker has reported that it is up.
        """
        if not self.limit_worker_threads:
            return {}
        previous = {name: os.environ.get(name) for name in THREAD_LIMIT_VARIABLES}
        for name in THREAD_LIMIT_VARIABLES:
            os.environ[name] = "1"
        return previous

    @staticmethod
    def _restore_thread_limits(previous: dict[str, str | None]) -> None:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def shutdown(self, *, timeout: float = 30.0) -> dict:
        """Stop the workers and release the shared block. Safe to call twice."""
        if self._closed:
            return {}
        self._closed = True
        totals = {
            "worker_cpu_seconds": 0.0,
            "worker_max_rss_bytes": 0,
            "total_observation_builds": 0,
            "total_transitions": 0,
            "total_terminals": 0,
            "total_resets": 0,
            "total_stillborn_games": 0,
            "total_decisions_recorded": 0,
            "total_games_recorded": 0,
            "total_record_bytes": 0,
            "total_snapshot_bytes": 0,
            "total_snapshot_count": 0,
            "total_verified_games": 0,
            "total_verified_decisions": 0,
            "total_reconstruction_mismatches": 0,
            "total_games_joined_late": 0,
            "recording_seconds": 0.0,
            "verification_seconds": 0.0,
            # Phase 6B persistence; stay 0 when no output directory is set.
            "total_persisted_bytes": 0,
            "total_compressed_bytes": 0,
            "total_shards_opened": 0,
            "total_shards_closed": 0,
            "total_records_persisted": 0,
            "total_encode_seconds": 0.0,
            "total_compress_seconds": 0.0,
            "total_write_seconds": 0.0,
            "total_flush_seconds": 0.0,
            "total_write_errors": 0,
            "total_pending_records": 0,
            "total_pending_bytes": 0,
        }
        retained: list[bytes] = []
        mismatch_details: list[dict] = []
        if self._started:
            self._sequence += 1
            for worker in self._workers:
                try:
                    worker.connection.send(
                        {"kind": "shutdown", "sequence": self._sequence}
                    )
                except (OSError, BrokenPipeError):  # pragma: no cover - already gone
                    pass
            deadline = time.monotonic() + timeout
            for worker in self._workers:
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    if worker.connection.poll(remaining):
                        reply = worker.connection.recv()
                        if reply.get("kind") == "shutdown_ack":
                            totals["worker_cpu_seconds"] += reply["process_seconds"]
                            totals["worker_max_rss_bytes"] += reply["max_rss_bytes"]
                            for key in tuple(totals):
                                if key in ("worker_cpu_seconds", "worker_max_rss_bytes"):
                                    continue
                                totals[key] += reply.get(key, 0)
                            retained.extend(reply.get("retained_records", ()))
                            mismatch_details.extend(reply.get("mismatch_details", ()))
                except (EOFError, OSError):  # pragma: no cover - already gone
                    pass

            for worker in self._workers:
                worker.process.join(timeout=max(0.0, deadline - time.monotonic()))
                if worker.process.is_alive():  # pragma: no cover - defensive
                    worker.process.terminate()
                    worker.process.join(timeout=5.0)
                try:
                    worker.connection.close()
                except Exception:  # pragma: no cover - defensive
                    pass

        if self.buffers is not None:
            self.buffers.close()
            self.buffers.unlink()
            self.buffers = None
        totals["retained_records"] = tuple(retained)
        totals["mismatch_details"] = tuple(mismatch_details)
        return totals

    def __enter__(self) -> "WorkerPool":
        self.start()
        return self

    def __exit__(self, *exception) -> None:
        self.shutdown()

    # -- phases ------------------------------------------------------------

    def step(self, *, apply_actions: bool = True, auto_reset: bool = True) -> PhaseReport:
        """Run one bulk-synchronous phase across every worker.

        Workers read the shared `actions` and `reset_request` rows they own,
        advance and reset their games, and republish their slots.
        """
        self._require_running()
        started = time.perf_counter()
        self._sequence += 1
        command = {
            "kind": "step",
            "sequence": self._sequence,
            "apply_actions": bool(apply_actions),
            "auto_reset": bool(auto_reset),
        }
        dispatch_started = time.perf_counter()
        self._dispatch(command)
        dispatch_seconds = time.perf_counter() - dispatch_started

        wait_started = time.perf_counter()
        replies = self._await_replies(self.step_timeout, stage="step")
        wait_seconds = time.perf_counter() - wait_started

        self._expected_publish += 1
        self._check_published()
        if self._pending_resets:
            # The coordinator owns `reset_request`, so it is also the only
            # writer that may clear it.
            self.buffers.reset_request.fill(0)
            self._pending_resets = False

        arrivals = [reply["_arrival"] for reply in replies.values()]
        self.last_command = command
        self.last_replies = tuple(
            {key: value for key, value in reply.items() if key != "_arrival"}
            for reply in replies.values()
        )
        return PhaseReport(
            sequence=self._sequence,
            stepped=sum(reply["stepped"] for reply in replies.values()),
            terminals=sum(reply["terminals"] for reply in replies.values()),
            resets=sum(reply["resets"] for reply in replies.values()),
            observation_builds=sum(
                reply["observation_builds"] for reply in replies.values()
            ),
            wall_seconds=time.perf_counter() - started,
            dispatch_seconds=dispatch_seconds,
            wait_seconds=wait_seconds,
            straggler_seconds=max(arrivals) - min(arrivals),
            worker_busy_seconds=sum(reply["busy_seconds"] for reply in replies.values()),
            worker_cpu_seconds=self._worker_cpu_delta(replies),
        )

    def publish(self) -> PhaseReport:
        """Ask every worker to republish its slots without stepping anything."""
        return self.step(apply_actions=False, auto_reset=False)

    def request_reset(self, slots) -> None:
        """Mark slots for reset in the next phase.

        The request travels in shared memory, not through the control pipes, so
        resetting 2,048 slots costs the same as resetting one.
        """
        self._require_running()
        index = np.asarray(list(slots), dtype=np.int64)
        if index.size and (index.min() < 0 or index.max() >= self.num_environments):
            raise ValueError("reset request refers to a slot outside the batch")
        self.buffers.reset_request[index] = 1
        self._pending_resets = True

    def set_actions(self, actions: np.ndarray) -> None:
        """Write the dense action vector the next phase will apply."""
        self._require_running()
        self.buffers.actions[:] = actions

    def clear_actions(self) -> None:
        self._require_running()
        self.buffers.actions.fill(SKIP_ACTION)

    def clear_decisions(self) -> None:
        """Retract every decision mark before a phase writes the new ones.

        Only `decision_valid` is cleared. The probability and value rows are
        left alone deliberately: a worker reads them only where the mark is set,
        so zeroing 2,048 x 128 floats every step would be pure cost.
        """
        self._require_running()
        self.buffers.decision_valid.fill(0)

    def recording_totals(self) -> dict:
        """Cumulative recording counters as of the most recent phase."""
        keys = (
            "total_decisions_recorded",
            "total_games_recorded",
            "total_stillborn_games",
            "total_record_bytes",
            "total_snapshot_bytes",
            "total_snapshot_count",
            "total_verified_games",
            "total_verified_decisions",
            "total_reconstruction_mismatches",
            "total_games_joined_late",
            "recording_seconds",
            "verification_seconds",
            # Phase 6B persistence. Absent from every reply unless an output
            # directory is configured, in which case `sum` over a missing key
            # already yields 0.
            "total_persisted_bytes",
            "total_compressed_bytes",
            "total_shards_opened",
            "total_shards_closed",
            "total_records_persisted",
            "total_encode_seconds",
            "total_compress_seconds",
            "total_write_seconds",
            "total_flush_seconds",
            "total_write_errors",
            "total_pending_records",
            "total_pending_bytes",
        )
        return {
            key: sum(reply.get(key, 0) for reply in self.last_replies) for key in keys
        }

    def select_actions(self, out: "np.ndarray | None" = None) -> np.ndarray:
        """Deterministic benchmark policy over the current shared masks."""
        self._require_running()
        return select_actions(self.buffers, self.root_seed, out)

    # -- fault injection (tests and the acceptance harness) ------------------

    def kill_worker(self, worker_id: int, *, sig: int = signal.SIGKILL) -> int:
        """Terminate one worker outright. Returns its process identifier."""
        self._require_running()
        worker = self._workers[worker_id]
        pid = worker.process.pid
        os.kill(pid, sig)
        worker.process.join(timeout=5.0)
        return pid

    def stall_worker(self, worker_id: int, seconds: float) -> None:
        """Make one worker sleep through the next phase. Test-only."""
        self._require_running()
        self._sequence += 1
        self._workers[worker_id].connection.send(
            {"kind": "stall", "sequence": self._sequence, "seconds": float(seconds)}
        )

    # -- worker information -------------------------------------------------

    def worker_pids(self) -> tuple[int, ...]:
        return tuple(worker.pid for worker in self._workers)

    def worker_liveness(self) -> tuple[bool, ...]:
        """Whether each worker process is still running, in worker-id order."""
        return tuple(worker.process.is_alive() for worker in self._workers)

    def worker_cpu_seconds(self) -> float:
        return sum(worker.process_seconds for worker in self._workers)

    def worker_max_rss_bytes(self) -> int:
        return sum(worker.max_rss_bytes for worker in self._workers)

    def assignments_as_dicts(self) -> list[dict]:
        return [assignment.as_dict() for assignment in self.assignments]

    # -- internals ---------------------------------------------------------

    def _require_running(self) -> None:
        if not self._started or self._closed or self.buffers is None:
            raise WorkerPoolError("this pool is not running")

    def _dispatch(self, command: dict) -> None:
        """Send one command to every worker, reporting a dead pipe as a crash."""
        failed: list[_Worker] = []
        for worker in self._workers:
            try:
                worker.connection.send(command)
            except (BrokenPipeError, EOFError, OSError):
                failed.append(worker)
        if failed:
            details = ", ".join(
                f"worker {worker.assignment.worker_id} (pid {worker.process.pid}, "
                f"exit code {worker.process.exitcode})"
                for worker in failed
            )
            raise WorkerCrashError(
                f"could not dispatch {command['kind']!r} to {details}; the control "
                "channel is gone, so those workers are no longer simulating"
            )

    def _worker_cpu_delta(self, replies: dict) -> float:
        delta = 0.0
        for reply in replies.values():
            worker = self._workers[reply["worker_id"]]
            delta += reply["process_seconds"] - worker.process_seconds
            worker.process_seconds = reply["process_seconds"]
            worker.max_rss_bytes = max(worker.max_rss_bytes, reply["max_rss_bytes"])
        return delta

    def _check_published(self) -> None:
        """Prove every slot was refreshed by the worker that owns it."""
        stale = self.buffers.stale_slots(self._expected_publish)
        if stale.size:
            owners = sorted({int(self.buffers.worker_id[slot]) for slot in stale[:32]})
            raise StaleBufferError(
                f"{stale.size} environment slots were not republished after phase "
                f"{self._sequence} (first: {stale[:8].tolist()}, owning workers "
                f"{owners}); the coordinator would be reading stale observations"
            )

    def _await_replies(self, timeout: float, *, stage: str) -> dict:
        """Collect one reply per worker, or raise a clear infrastructure error.

        Watches the control pipes and the process sentinels together, so an
        exited worker is noticed immediately instead of at the phase timeout.
        """
        pending = {worker.connection: worker for worker in self._workers}
        sentinels = {worker.process.sentinel: worker for worker in self._workers}
        replies: dict[int, dict] = {}
        deadline = time.monotonic() + timeout

        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                slow = sorted(worker.assignment.worker_id for worker in pending.values())
                raise WorkerTimeoutError(
                    f"workers {slow} did not complete the {stage} phase within "
                    f"{timeout:.1f}s; the pool is not making progress and the "
                    "shared buffers must not be read"
                )
            ready = wait_for_objects(
                list(pending) + list(sentinels), timeout=min(remaining, 1.0)
            )
            crashed: list[_Worker] = []

            for obj in ready:
                if isinstance(obj, Connection):
                    worker = pending.get(obj)
                    if worker is None:
                        continue
                    try:
                        reply = obj.recv()
                    except (EOFError, OSError):
                        del pending[obj]
                        crashed.append(worker)
                        continue
                    del pending[obj]
                    if reply.get("kind") == "error":
                        raise WorkerFaultError(
                            f"worker {reply['worker_id']} raised "
                            f"{reply['error_type']}: {reply['message']}",
                            worker_id=reply["worker_id"],
                            remote_traceback=reply["traceback"],
                        )
                    reply["_arrival"] = time.perf_counter()
                    replies[reply["worker_id"]] = reply
                else:
                    worker = sentinels.pop(obj, None)
                    if worker is None:
                        continue
                    if worker.connection in pending:
                        # The pipe may still hold the worker's last message, so
                        # give the connection path one more chance before
                        # declaring a crash.
                        if not worker.connection.poll(0):
                            del pending[worker.connection]
                            crashed.append(worker)

            if crashed:
                details = ", ".join(
                    f"worker {worker.assignment.worker_id} (pid {worker.process.pid}, "
                    f"exit code {worker.process.exitcode}) owning slots "
                    f"[{worker.assignment.start}, {worker.assignment.stop})"
                    for worker in crashed
                )
                raise WorkerCrashError(
                    f"simulation worker process failure during the {stage} phase: "
                    f"{details}. The environments those workers own are no longer "
                    "being simulated and their shared-memory slots are stale."
                )

        return replies


# ---------------------------------------------------------------------------
# Coordinator-side reading helpers
# ---------------------------------------------------------------------------


def collect_finished(
    buffers: SharedEnvironmentBuffers, previous_episode_count: np.ndarray
) -> list[dict]:
    """Outcome of every slot that finished a game since the last call.

    Reads only the `last_*` fields, which a worker never overwrites until the
    slot finishes another game, so an immediate reset cannot lose an outcome.
    """
    finished = np.flatnonzero(buffers.episode_count > previous_episode_count)
    outcomes: list[dict] = []
    for slot in finished:
        winner = int(buffers.last_winner[slot])
        outcomes.append(
            {
                "slot": int(slot),
                "environment_id": int(buffers.environment_id[slot]),
                "generation": int(buffers.last_generation[slot]),
                "terminal_reason": terminal_reason_name(
                    int(buffers.last_terminal_reason[slot])
                ),
                "winner": None if winner == NO_WINNER else winner,
                "is_draw": bool(buffers.last_is_draw[slot]),
                "total_moves": int(buffers.last_total_moves[slot]),
                "result_for_red": float(buffers.last_result_red[slot]),
                "result_for_blue": float(buffers.last_result_blue[slot]),
            }
        )
    previous_episode_count[:] = buffers.episode_count
    return outcomes


def result_for_player(outcome: dict, player: int) -> float:
    """Result of a collected outcome from one player's point of view."""
    return outcome["result_for_red" if player == PLAYERS[0] else "result_for_blue"]


__all__ = [
    "THREAD_LIMIT_VARIABLES",
    "WORKER_POOL_VERSION",
    "PhaseReport",
    "StaleBufferError",
    "WorkerAssignment",
    "WorkerCrashError",
    "WorkerFaultError",
    "WorkerPool",
    "WorkerPoolError",
    "WorkerTimeoutError",
    "collect_finished",
    "offer_to_reservoir",
    "partition_environments",
    "result_for_player",
    "select_action",
    "select_actions",
    "slot_hash",
    "slot_hashes",
]
