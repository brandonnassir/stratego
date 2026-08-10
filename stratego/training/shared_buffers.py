"""Persistent preallocated shared-memory buffers for Phase 3 CPU simulation.

Specification sources:

- `03_game_engine_spec.md` section 16 (batch simulation interface)
- `03_game_engine_spec.md` section 17 (environment identifier and generation)
- `00_PHASE_3_SEQUENCE_AND_COMMON_CONTRACT.md` (persistent preallocated buffers)

Every environment slot has a fixed row in every buffer, so the coordinator and
the simulation workers exchange observations, legality, actions and control
metadata by writing into memory both already hold rather than by sending
objects through a queue. Nothing here serialises a game state, an observation or
a legality mask.

Layout
------
One :class:`~multiprocessing.shared_memory.SharedMemory` block holds every
field. Fields are stored field-major and 64-byte aligned, so `observations` is a
genuine `(N, 127, 10, 10)` C-contiguous `float32` array and `legal_mask` a
genuine `(N, 10000)` `uint8` array; a later agent can wrap either with
`torch.from_numpy` without a copy.

Writer discipline
-----------------
The buffers carry no lock. They do not need one because every field has exactly
one writer:

- `WRITER_WORKER` fields are written only by the worker that owns the slot, and
  a slot belongs to exactly one worker for the life of the pool;
- `WRITER_COORDINATOR` fields are written only by the coordinator.

Within a bulk-synchronous step the two sides never write at the same time: the
coordinator writes actions and reset requests, hands the phase to the workers,
and does not touch the buffers again until every worker has reported completion.
`publish_sequence` makes a violation of that discipline detectable rather than
silent -- see :meth:`SharedEnvironmentBuffers.stale_slots`.

Decision transport
------------------
`policy_probabilities`, `value_prediction` and `decision_valid` were added for
Agent 5 and are written by the coordinator alongside `actions`. They exist
because trajectory records must be built by the worker that owns the slot --
the coordinator deliberately holds no game object -- while the policy and value
that belong in a record come from the coordinator's model. Carrying them down
the same shared block keeps recording free of any extra round trip. They are a
pure addition: a pool that does not record trajectories never reads them.

Hidden information
------------------
Only observer-safe data is published here. Privileged values -- the true board,
piece identities and `belief_targets` -- stay inside the worker process, which
is what keeps the model-facing transport free of hidden information. The
coordinator-written decision fields are model *outputs* over the already-public
legal-action list, so they add no privileged information to the transport.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

from ..engine.constants import (
    ACTION_SPACE_SIZE,
    NOT_TERMINAL,
    OBSERVATION_SHAPE,
    TERMINAL_REASONS,
)
from ..engine.observation import OBSERVATION_DTYPE

SHARED_BUFFER_VERSION = "shared_buffers_v1"

# Slot lifecycle, `status`.
STATUS_EMPTY = 0
STATUS_ACTIVE = 1
STATUS_TERMINAL = 2

# `acting_player` for a slot with no player to move, matching
# `batch_simulation.NO_ACTING_PLAYER`.
NO_ACTING_PLAYER = -1

# `actions` entry meaning "do not step this slot", matching
# `batch_simulation.SKIP_ACTION`.
SKIP_ACTION = -1

# Capacity of a `policy_probabilities` row. The coordinator writes one
# probability per entry of the slot's *ascending* legal-action list, which is
# the same order `BatchSimulator.legal_actions` returns and the same order
# `numpy.flatnonzero(legal_mask)` produces, so neither side has to transmit the
# identifiers themselves. Agent 3 observed a maximum of 62 legal actions across
# 1,000,162 decisions and Agent 4 priced its compact legality path at 128; the
# same headroom is used here. A slot with more legal actions than this raises
# rather than silently truncating the distribution.
POLICY_CAPACITY = 128

# Entries in a `value_prediction` row: the win/draw/loss triple that
# `trajectory.GameTrajectoryBuilder.record_decision` already expects.
VALUE_CLASSES = 3

# `last_terminal_reason` before a slot has ever finished a game.
NO_TERMINAL_REASON = -1

# `winner` for a drawn or unfinished game.
NO_WINNER = -1

WRITER_WORKER = "worker"
WRITER_COORDINATOR = "coordinator"

# Terminal reasons travel as their index in the frozen `TERMINAL_REASONS` tuple
# so the buffer stays a fixed-width numeric array. The mapping is part of the
# frozen engine contract, not something this module defines.
TERMINAL_REASON_CODES = {reason: code for code, reason in enumerate(TERMINAL_REASONS)}
TERMINAL_REASON_BY_CODE = dict(enumerate(TERMINAL_REASONS))
NOT_TERMINAL_CODE = TERMINAL_REASON_CODES[NOT_TERMINAL]

# Fields are aligned to a cache line so that two workers writing adjacent
# scalar fields never share a cache line boundary by accident.
_ALIGNMENT = 64


class SharedBufferError(RuntimeError):
    """Raised when a shared-memory buffer cannot be created, attached or used."""


@dataclass(frozen=True)
class FieldSpec:
    """One shared field: a fixed row per environment slot."""

    name: str
    item_shape: tuple[int, ...]
    dtype: str
    fill: float
    writer: str
    description: str

    def shape(self, num_environments: int) -> tuple[int, ...]:
        return (num_environments,) + self.item_shape

    def nbytes(self, num_environments: int) -> int:
        itemsize = np.dtype(self.dtype).itemsize
        count = int(np.prod(self.item_shape, dtype=np.int64)) if self.item_shape else 1
        return num_environments * count * itemsize


#: The shared payload. `observations`, `legal_mask`, `acting_player`,
#: `environment_id`, `generation` and `actions` are the fields Agent 2's
#: instructions require; the rest is the terminal/status metadata the
#: coordinator needs in order to apply a reset policy and record outcomes
#: without asking a worker for an object.
FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(
        "observations",
        OBSERVATION_SHAPE,
        "float32",
        0.0,
        WRITER_WORKER,
        "observation_v2_1_127ch from the acting player's perspective; zero when "
        "the slot has no player to move",
    ),
    FieldSpec(
        "legal_mask",
        (ACTION_SPACE_SIZE,),
        "uint8",
        0,
        WRITER_WORKER,
        "dense 10,000-entry legality mask for the acting player",
    ),
    FieldSpec(
        "legal_count",
        (),
        "int32",
        0,
        WRITER_WORKER,
        "number of set entries in legal_mask",
    ),
    FieldSpec(
        "acting_player",
        (),
        "int8",
        NO_ACTING_PLAYER,
        WRITER_WORKER,
        "player to move, -1 when the slot is terminal",
    ),
    FieldSpec(
        "environment_id",
        (),
        "int32",
        -1,
        WRITER_WORKER,
        "fixed environment identifier of the slot",
    ),
    FieldSpec(
        "generation",
        (),
        "int32",
        -1,
        WRITER_WORKER,
        "generation of the game currently in the slot; +1 per reset",
    ),
    FieldSpec(
        "ply",
        (),
        "int32",
        0,
        WRITER_WORKER,
        "GameState.total_moves of the current game",
    ),
    FieldSpec(
        "battleless_moves",
        (),
        "int32",
        0,
        WRITER_WORKER,
        "GameState.battleless_moves of the current game",
    ),
    FieldSpec(
        "terminal",
        (),
        "uint8",
        0,
        WRITER_WORKER,
        "1 when the current game has finished and has not been reset",
    ),
    FieldSpec(
        "status",
        (),
        "int8",
        STATUS_EMPTY,
        WRITER_WORKER,
        "slot lifecycle: 0 empty, 1 active, 2 terminal",
    ),
    FieldSpec(
        "worker_id",
        (),
        "int16",
        -1,
        WRITER_WORKER,
        "worker that owns the slot; fixed for the life of the pool",
    ),
    FieldSpec(
        "publish_sequence",
        (),
        "int64",
        0,
        WRITER_WORKER,
        "incremented every time the owning worker republishes the slot; lets the "
        "coordinator prove it is not reading a stale buffer",
    ),
    FieldSpec(
        "episode_count",
        (),
        "int32",
        0,
        WRITER_WORKER,
        "monotonic count of games completed in this slot",
    ),
    FieldSpec(
        "last_terminal_reason",
        (),
        "int8",
        NO_TERMINAL_REASON,
        WRITER_WORKER,
        "TERMINAL_REASON_CODES value of the most recently finished game, -1 if none",
    ),
    FieldSpec(
        "last_winner",
        (),
        "int8",
        NO_WINNER,
        WRITER_WORKER,
        "winner of the most recently finished game, -1 for a draw or none",
    ),
    FieldSpec(
        "last_is_draw",
        (),
        "uint8",
        0,
        WRITER_WORKER,
        "1 when the most recently finished game was a draw",
    ),
    FieldSpec(
        "last_total_moves",
        (),
        "int32",
        0,
        WRITER_WORKER,
        "final ply of the most recently finished game",
    ),
    FieldSpec(
        "last_generation",
        (),
        "int32",
        -1,
        WRITER_WORKER,
        "generation of the most recently finished game",
    ),
    FieldSpec(
        "last_result_red",
        (),
        "float32",
        0.0,
        WRITER_WORKER,
        "result_for(RED) of the most recently finished game",
    ),
    FieldSpec(
        "last_result_blue",
        (),
        "float32",
        0.0,
        WRITER_WORKER,
        "result_for(BLUE) of the most recently finished game",
    ),
    FieldSpec(
        "actions",
        (),
        "int32",
        SKIP_ACTION,
        WRITER_COORDINATOR,
        "action identifier to apply to the slot; negative means leave it alone",
    ),
    FieldSpec(
        "reset_request",
        (),
        "uint8",
        0,
        WRITER_COORDINATOR,
        "1 asks the owning worker to reset the slot in the next phase",
    ),
    # -- decision transport, added for Agent 5 -------------------------------
    # Trajectory records have to be built inside the worker that owns the slot,
    # because the coordinator deliberately holds no game object. But the policy
    # and value that belong in a record are produced by the coordinator's model.
    # These three fields carry that decision back down the same shared block the
    # actions travel on, so recording costs no extra round trip and no game
    # state is ever serialised.
    FieldSpec(
        "policy_probabilities",
        (POLICY_CAPACITY,),
        "float32",
        0.0,
        WRITER_COORDINATOR,
        "probabilities over the slot's ascending legal-action list, zero-padded "
        "to POLICY_CAPACITY; only the first legal_count entries are meaningful",
    ),
    FieldSpec(
        "value_prediction",
        (VALUE_CLASSES,),
        "float32",
        0.0,
        WRITER_COORDINATOR,
        "win/draw/loss prediction for the acting player of the slot",
    ),
    FieldSpec(
        "decision_valid",
        (),
        "uint8",
        0,
        WRITER_COORDINATOR,
        "1 when the coordinator wrote a policy/value decision for this slot in "
        "the current phase; a recording worker stores a decision only when set",
    ),
)

FIELD_SPECS_BY_NAME = {spec.name: spec for spec in FIELD_SPECS}

WORKER_WRITTEN_FIELDS = tuple(
    spec.name for spec in FIELD_SPECS if spec.writer == WRITER_WORKER
)
COORDINATOR_WRITTEN_FIELDS = tuple(
    spec.name for spec in FIELD_SPECS if spec.writer == WRITER_COORDINATOR
)


@dataclass(frozen=True)
class SharedBufferDescriptor:
    """Everything a worker needs in order to attach to an existing block.

    Small and picklable on purpose: this is the one thing about the buffers that
    travels through a control channel, and it carries no game data.
    """

    name: str
    num_environments: int
    nbytes: int
    offsets: tuple[tuple[str, int], ...]
    version: str = SHARED_BUFFER_VERSION

    def offset_map(self) -> dict[str, int]:
        return dict(self.offsets)


def _aligned(offset: int) -> int:
    remainder = offset % _ALIGNMENT
    return offset if remainder == 0 else offset + _ALIGNMENT - remainder


def plan_layout(num_environments: int) -> tuple[dict[str, int], int]:
    """Byte offset of every field plus the total block size."""
    offsets: dict[str, int] = {}
    cursor = 0
    for spec in FIELD_SPECS:
        cursor = _aligned(cursor)
        offsets[spec.name] = cursor
        cursor += spec.nbytes(num_environments)
    return offsets, _aligned(cursor)


def buffer_nbytes(num_environments: int) -> int:
    """Total shared-memory bytes one pool of `num_environments` slots needs."""
    return plan_layout(num_environments)[1]


def _open_shared_memory(name: str) -> shared_memory.SharedMemory:
    """Attach without registering with the resource tracker where possible.

    A worker attaches to a block the coordinator owns. Registering it in the
    worker makes the resource tracker warn about, and on some versions unlink,
    a block the worker does not own. `track` exists from Python 3.13.
    """
    try:
        return shared_memory.SharedMemory(name=name, create=False, track=False)
    except TypeError:  # pragma: no cover - Python < 3.13
        return shared_memory.SharedMemory(name=name, create=False)


class SharedEnvironmentBuffers:
    """Persistent shared-memory view of `num_environments` environment slots.

    Create it once in the coordinator, attach to it in every worker, and keep it
    for the life of the run. Reallocating per step would defeat the point.
    """

    def __init__(
        self,
        descriptor: SharedBufferDescriptor,
        block: shared_memory.SharedMemory,
        *,
        owner: bool,
    ) -> None:
        self.descriptor = descriptor
        self.num_environments = descriptor.num_environments
        self._block = block
        self._owner = owner
        self._closed = False
        self._arrays: dict[str, np.ndarray] = {}
        offsets = descriptor.offset_map()
        for spec in FIELD_SPECS:
            self._arrays[spec.name] = np.ndarray(
                spec.shape(self.num_environments),
                dtype=np.dtype(spec.dtype),
                buffer=block.buf,
                offset=offsets[spec.name],
            )

    # -- construction ------------------------------------------------------

    @classmethod
    def create(cls, num_environments: int) -> "SharedEnvironmentBuffers":
        """Allocate a new block and initialise every field to its fill value."""
        if num_environments < 1:
            raise ValueError("a shared buffer needs at least one environment slot")
        offsets, total = plan_layout(num_environments)
        try:
            block = shared_memory.SharedMemory(create=True, size=total)
        except OSError as error:  # pragma: no cover - platform capability
            raise SharedBufferError(
                f"could not allocate {total} bytes of shared memory for "
                f"{num_environments} environments: {error}"
            ) from error
        descriptor = SharedBufferDescriptor(
            name=block.name,
            num_environments=num_environments,
            nbytes=total,
            offsets=tuple(sorted(offsets.items())),
        )
        buffers = cls(descriptor, block, owner=True)
        buffers.reset_fields()
        return buffers

    @classmethod
    def attach(cls, descriptor: SharedBufferDescriptor) -> "SharedEnvironmentBuffers":
        """Attach to a block another process created. Does not initialise it."""
        if descriptor.version != SHARED_BUFFER_VERSION:
            raise SharedBufferError(
                f"shared buffer version {descriptor.version!r} does not match "
                f"{SHARED_BUFFER_VERSION!r}"
            )
        try:
            block = _open_shared_memory(descriptor.name)
        except FileNotFoundError as error:
            raise SharedBufferError(
                f"shared memory block {descriptor.name!r} does not exist; the "
                "coordinator may have shut down"
            ) from error
        if block.size < descriptor.nbytes:  # pragma: no cover - defensive
            block.close()
            raise SharedBufferError(
                f"shared memory block {descriptor.name!r} is {block.size} bytes, "
                f"expected at least {descriptor.nbytes}"
            )
        return cls(descriptor, block, owner=False)

    def reset_fields(self) -> None:
        """Write every field's fill value. Coordinator-side setup only."""
        for spec in FIELD_SPECS:
            self._arrays[spec.name].fill(spec.fill)

    # -- access ------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_environments

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self._arrays[name]
        except KeyError:
            raise KeyError(f"unknown shared field {name!r}") from None

    def __getattr__(self, name: str) -> np.ndarray:
        # Only reached for names that are not real attributes, so the explicit
        # attributes set in __init__ keep working normally.
        arrays = self.__dict__.get("_arrays")
        if arrays is not None and name in arrays:
            return arrays[name]
        raise AttributeError(name)

    def field_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in FIELD_SPECS)

    def view(self, start: int, stop: int) -> dict[str, np.ndarray]:
        """Every field restricted to the half-open slot range `[start, stop)`.

        The returned arrays are views into shared memory, not copies, so a
        worker writes its own rows through them and touches nothing else.
        """
        if not 0 <= start <= stop <= self.num_environments:
            raise ValueError(
                f"slot range [{start}, {stop}) is outside a batch of "
                f"{self.num_environments}"
            )
        return {name: array[start:stop] for name, array in self._arrays.items()}

    # -- reporting ---------------------------------------------------------

    def shapes(self) -> dict[str, list[int]]:
        return {
            spec.name: list(spec.shape(self.num_environments)) for spec in FIELD_SPECS
        }

    def dtypes(self) -> dict[str, str]:
        return {spec.name: spec.dtype for spec in FIELD_SPECS}

    def field_documentation(self) -> list[dict]:
        """Documented dtype/shape/writer of every field, for the data file."""
        return [
            {
                "name": spec.name,
                "shape": list(spec.shape(self.num_environments)),
                "dtype": spec.dtype,
                "writer": spec.writer,
                "bytes": spec.nbytes(self.num_environments),
                "description": spec.description,
            }
            for spec in FIELD_SPECS
        ]

    @property
    def nbytes(self) -> int:
        return self.descriptor.nbytes

    def stale_slots(self, expected: np.ndarray) -> np.ndarray:
        """Slots whose `publish_sequence` is not `expected`.

        The coordinator calls this after a phase. A non-empty result means a
        worker did not republish a slot it owns, which is exactly the
        stale-buffer failure the architecture must not paper over.
        """
        return np.flatnonzero(self._arrays["publish_sequence"] != expected)

    def snapshot_rows(self, slots: "np.ndarray | list[int]") -> dict[str, np.ndarray]:
        """Independent copies of the selected rows of every field.

        Used by the reset-isolation checks: a copy taken before a phase can be
        compared byte for byte with the shared rows after it.
        """
        index = np.asarray(slots, dtype=np.int64)
        return {name: array[index].copy() for name, array in self._arrays.items()}

    def rows_equal(self, other: dict[str, np.ndarray], slots) -> list[str]:
        """Names of the fields whose selected rows differ from `other`."""
        index = np.asarray(slots, dtype=np.int64)
        return [
            name
            for name, array in self._arrays.items()
            if not np.array_equal(array[index], other[name])
        ]

    # -- teardown ----------------------------------------------------------

    def close(self) -> None:
        """Drop this process's mapping. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        # Every numpy view has to go before the mapping does, otherwise
        # `SharedMemory.close` raises BufferError on the exported buffers.
        self._arrays.clear()
        try:
            self._block.close()
        except BufferError:  # pragma: no cover - defensive
            pass

    def unlink(self) -> None:
        """Destroy the block. Only the creating process may call this."""
        if not self._owner:
            raise SharedBufferError("only the owning process may unlink a shared block")
        try:
            self._block.unlink()
        except FileNotFoundError:  # pragma: no cover - already gone
            pass

    def __enter__(self) -> "SharedEnvironmentBuffers":
        return self

    def __exit__(self, *exception) -> None:
        self.close()
        if self._owner:
            self.unlink()


def terminal_reason_code(reason: str) -> int:
    """Frozen-tuple index of a terminal reason string."""
    try:
        return TERMINAL_REASON_CODES[reason]
    except KeyError:
        raise SharedBufferError(f"unknown terminal reason {reason!r}") from None


def terminal_reason_name(code: int) -> str:
    """Terminal reason string for a code, or `not_terminal` for the unset value."""
    if int(code) == NO_TERMINAL_REASON:
        return NOT_TERMINAL
    try:
        return TERMINAL_REASON_BY_CODE[int(code)]
    except KeyError:
        raise SharedBufferError(f"unknown terminal reason code {code!r}") from None


def max_resident_bytes() -> int:
    """Peak resident set size of the calling process, in bytes."""
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    return usage if sys.platform == "darwin" else usage * 1024


__all__ = [
    "COORDINATOR_WRITTEN_FIELDS",
    "FIELD_SPECS",
    "FIELD_SPECS_BY_NAME",
    "NOT_TERMINAL_CODE",
    "NO_ACTING_PLAYER",
    "NO_TERMINAL_REASON",
    "NO_WINNER",
    "SHARED_BUFFER_VERSION",
    "SKIP_ACTION",
    "STATUS_ACTIVE",
    "STATUS_EMPTY",
    "STATUS_TERMINAL",
    "TERMINAL_REASON_BY_CODE",
    "TERMINAL_REASON_CODES",
    "WORKER_WRITTEN_FIELDS",
    "WRITER_COORDINATOR",
    "WRITER_WORKER",
    "FieldSpec",
    "SharedBufferDescriptor",
    "SharedBufferError",
    "SharedEnvironmentBuffers",
    "buffer_nbytes",
    "max_resident_bytes",
    "plan_layout",
    "terminal_reason_code",
    "terminal_reason_name",
]
