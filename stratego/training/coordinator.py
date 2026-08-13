"""Bulk-synchronous self-play coordinator: CPU workers, one Metal consumer.

Specification sources:

- `03_game_engine_spec.md` section 16 (bulk-synchronous batch interface)
- `03_game_engine_spec.md` section 18 (only the coordinator owns the device)
- `00_PHASE_3_SEQUENCE_AND_COMMON_CONTRACT.md` (approved architecture)
- `05_AGENT_5_END_TO_END_DECISION.md` (required cycle)

One global step::

    workers build observations and legality into shared memory
    -> barrier: every worker reports its phase complete
    -> coordinator runs the model over the ready rows, in inference-batch chunks
    -> coordinator applies legality and samples one legal action per row
    -> actions, policy and value written back to shared memory
    -> workers advance their environments
    -> finished games are sealed and their slots independently reset
    -> next global step

Device ownership
----------------
This module is the *only* place that imports the model or touches Metal. A
simulation worker never imports it, which is why the worker pool it drives can
stay a pure NumPy/engine process. Importing this module requires PyTorch;
`stratego.training.worker_pool` does not.

Why the coordinator always knows the compact legal set
------------------------------------------------------
A stored decision carries one probability per *legal* action, in the ascending
order the engine produces. The coordinator therefore has to know that ordering
whenever it is recording, regardless of which legality representation the model
uses. Building it is a vectorised pass over the dense masks -- see
:func:`compact_legality_from_masks` -- and it is skipped entirely when recording
is off, so a pure throughput run does not pay for it.

Determinism
-----------
Sampling uses a seeded generator on the device. Two runs with the same seed,
configuration and worker count draw the same actions. Changing the worker count
changes nothing about what a slot contains, because a slot's game is a function
of `(root_seed, environment_id, generation)` alone -- but it does change the
sampling order, so action draws are reproducible per configuration rather than
across configurations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..engine.constants import ACTION_SPACE_SIZE, RulesConfig, TRAINING_RULES
from .representative_model import (
    CompactLegality,
    RepresentativeConfig,
    build_representative_model,
    compact_legal_probabilities,
    dense_legal_probabilities,
    dense_mask_to_bool,
    sample_compact,
    sample_dense,
)
from .shared_buffers import (
    POLICY_CAPACITY,
    SKIP_ACTION,
    STATUS_ACTIVE,
    VALUE_CLASSES,
    terminal_reason_name,
)
from .worker_pool import (
    DEFAULT_COLLECTION_POLICY_VERSION,
    RecordingConfig,
    WorkerPool,
)

COORDINATOR_VERSION = "coordinator_v1"

#: Precisions the coordinator will run the encoder in. Sampling and
#: normalisation always run in float32 regardless -- see
#: `representative_model.SAMPLING_DTYPE`.
PRECISIONS = ("float32", "float16", "bfloat16")

DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

LEGALITY_MODES = ("dense", "compact")

#: Which frame the *model* indexes its 10,000 policy logits in.
#:
#: `absolute_engine_squares` is the Phase 3 behaviour and stays the default, so
#: every accepted Phase 3 measurement and test means exactly what it meant
#: before. `perspective_normalized_squares` is `model_contract_v2`: the model
#: reads normalized tokens *and* emits a normalized action, and the coordinator
#: converts both legality and the selected action at the device boundary. The
#: engine is absolute in both cases and is never told about the model's frame.
ACTION_FRAME_ABSOLUTE = "absolute_engine_squares"
ACTION_FRAME_NORMALIZED = "perspective_normalized_squares"
ACTION_FRAMES = (ACTION_FRAME_ABSOLUTE, ACTION_FRAME_NORMALIZED)


class CoordinatorError(RuntimeError):
    """Raised when the end-to-end pipeline cannot run or has lost an invariant."""


class ActionFrameMismatchError(CoordinatorError):
    """A batch could not be converted between the engine and model frames.

    Separate from the generic coordinator error because a frame failure has one
    specific consequence -- the move that reaches the engine is not the move the
    network chose -- and the benchmark counts it as its own category.
    """


# ---------------------------------------------------------------------------
# Legality
# ---------------------------------------------------------------------------


def compact_legality_from_masks(
    masks: np.ndarray, *, capacity: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised `(B, 10000)` dense masks -> padded ascending legal identifiers.

    Returns `(action_ids, valid, counts)`. `numpy.nonzero` walks a 2-D array in
    row-major order, so the identifiers land in exactly the ascending order
    `BatchSimulator.legal_actions` returns -- which is what lets a worker line
    the probabilities up with its own legal list without transmitting either.

    This is the whole-batch replacement for
    `representative_model.build_compact_legality`, which loops in Python and is
    fine for a 2,048-row benchmark pool built once but not for a per-step cost.
    """
    if masks.ndim != 2 or masks.shape[1] != ACTION_SPACE_SIZE:
        raise ValueError(
            f"expected (B, {ACTION_SPACE_SIZE}) masks, got {tuple(masks.shape)}"
        )
    # Zero-copy reinterpretation; NumPy's boolean `nonzero` is several times
    # faster than scanning the identical bytes as uint8 (Agent 2's finding).
    boolean = masks.view(np.bool_) if masks.dtype == np.uint8 else masks.astype(bool)
    rows, columns = np.nonzero(boolean)
    counts = np.count_nonzero(boolean, axis=1)
    if counts.size and int(counts.max()) > capacity:
        raise CoordinatorError(
            f"a position has {int(counts.max())} legal actions, above the "
            f"compact capacity of {capacity}"
        )

    starts = np.zeros(counts.size, dtype=np.int64)
    if counts.size > 1:
        np.cumsum(counts[:-1], out=starts[1:])
    positions = np.arange(rows.size, dtype=np.int64) - np.repeat(starts, counts)

    action_ids = np.zeros((masks.shape[0], capacity), dtype=np.int64)
    valid = np.zeros((masks.shape[0], capacity), dtype=bool)
    action_ids[rows, positions] = columns
    valid[rows, positions] = True
    return action_ids, valid, counts.astype(np.int64)


# ---------------------------------------------------------------------------
# Action-frame conversion at the device boundary
# ---------------------------------------------------------------------------


class NormalizedActionFrame:
    """Whole-batch absolute <-> perspective-normalized conversion, on the device.

    The conversion itself is *not* defined here. Both tables are built by asking
    :mod:`stratego.model.action_frame` for every one of the 10,000 identifiers,
    per player, so this class holds no coordinate geometry of its own and cannot
    become a second, competing convention -- which is the whole reason Agent 1
    put the transform in one module.

    What this class adds is shape: Agent 1's helpers convert one action, one
    list or one mask at a time, and a global step converts up to 2,048 dense
    10,000-entry masks. Doing that per row in Python would cost more than the
    forward pass, so the tables are moved to the device once and applied as two
    `index_select` calls per acting colour.

    The permutation is per *player*, so rows are split by their published acting
    player and each group is converted with its own table. A player whose table
    is the identity (red, under the frozen convention) is skipped entirely
    rather than copied through a no-op gather.
    """

    def __init__(self, device: torch.device) -> None:
        # Imported here rather than at module scope: building the tables costs
        # 20,000 engine calls, and a run in the absolute frame must not pay for
        # a conversion it never performs.
        from ..engine.constants import PLAYERS
        from ..model.action_frame import absolute_action_to_model, model_action_to_absolute

        identity = np.arange(ACTION_SPACE_SIZE, dtype=np.int64)
        self.device = device
        self.players = tuple(int(player) for player in PLAYERS)
        self.to_model_host: dict[int, np.ndarray] = {}
        self.to_absolute_host: dict[int, np.ndarray] = {}
        self.is_identity: dict[int, bool] = {}

        for player in self.players:
            forward = np.fromiter(
                (absolute_action_to_model(action, player) for action in range(ACTION_SPACE_SIZE)),
                dtype=np.int64,
                count=ACTION_SPACE_SIZE,
            )
            inverse = np.fromiter(
                (model_action_to_absolute(action, player) for action in range(ACTION_SPACE_SIZE)),
                dtype=np.int64,
                count=ACTION_SPACE_SIZE,
            )
            # Agent 1 proves this at its own import; it is re-proved here because
            # a non-bijective table would silently make two distinct engine moves
            # share one policy logit, and the pipeline would never notice.
            if not np.array_equal(inverse[forward], identity):
                raise ActionFrameMismatchError(
                    f"absolute -> model -> absolute is not the identity for player {player}"
                )
            self.to_model_host[player] = forward
            self.to_absolute_host[player] = inverse
            self.is_identity[player] = bool(np.array_equal(forward, identity))

        self.to_model = {
            player: torch.from_numpy(table).to(device)
            for player, table in self.to_model_host.items()
        }
        self.to_absolute = {
            player: torch.from_numpy(table).to(device)
            for player, table in self.to_absolute_host.items()
        }

    # -- row assignment -----------------------------------------------------

    def split_rows(self, acting_players: np.ndarray) -> dict[int, np.ndarray]:
        """Group row indices by acting player, refusing a row with no player.

        Every row handed to the model is a position somebody is to move in, so
        a `NO_ACTING_PLAYER` row here means the coordinator is about to ask the
        network for a move in a game that has none.
        """
        assignment: dict[int, np.ndarray] = {}
        assigned = 0
        for player in self.players:
            index = np.flatnonzero(acting_players == player)
            assigned += int(index.size)
            if index.size:
                assignment[player] = index
        if assigned != int(acting_players.size):
            raise ActionFrameMismatchError(
                f"{int(acting_players.size) - assigned} of {int(acting_players.size)} rows "
                f"published an acting player outside {self.players}; a normalized action "
                "cannot be converted without knowing whose perspective it is in"
            )
        return assignment

    def _rows(self, index: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(index)).to(self.device)

    # -- conversions --------------------------------------------------------

    def masks_to_model(
        self, masks: torch.Tensor, assignment: dict[int, np.ndarray]
    ) -> torch.Tensor:
        """Dense absolute legality `(B, 10000)` -> the model's normalized frame.

        `normalized[m] = absolute[to_absolute[m]]` -- the gather spelling of
        Agent 1's scatter definition `normalized[to_model[a]] = absolute[a]`.
        Written this way because a gather is one kernel on Metal and because the
        result does not depend on the transform happening to be an involution.
        Converts in place: `masks` is already a fresh device copy of shared
        memory, and a second 20 MB allocation per chunk is pure cost.
        """
        for player, index in assignment.items():
            if self.is_identity[player]:
                continue
            rows = self._rows(index)
            converted = masks.index_select(0, rows).index_select(1, self.to_absolute[player])
            masks[rows] = converted
        return masks

    def actions_to_absolute(
        self, model_actions: torch.Tensor, assignment: dict[int, np.ndarray]
    ) -> torch.Tensor:
        """Selected normalized identifiers -> the absolute actions the engine gets."""
        absolute = model_actions.clone()
        for player, index in assignment.items():
            if self.is_identity[player]:
                continue
            rows = self._rows(index)
            absolute[rows] = self.to_absolute[player].index_select(
                0, model_actions.index_select(0, rows)
            )
        return absolute

    def action_ids_to_model(
        self, action_ids: torch.Tensor, assignment: dict[int, np.ndarray]
    ) -> torch.Tensor:
        """`(B, capacity)` absolute identifiers -> normalized identifiers.

        Used for the two places a *set* of engine actions has to be read in the
        model's frame: gathering one stored probability per legal action, and
        the compact-legality sampling path. The ascending absolute order is
        deliberately preserved rather than re-sorted -- it is the order the
        worker's own legal-action list is in, and the record depends on it.
        """
        model_ids = action_ids.clone()
        for player, index in assignment.items():
            if self.is_identity[player]:
                continue
            rows = self._rows(index)
            picked = action_ids.index_select(0, rows)
            model_ids[rows] = self.to_model[player].index_select(
                0, picked.reshape(-1)
            ).reshape(picked.shape)
        return model_ids

    def summary(self) -> dict:
        return {
            "engine_action_frame": ACTION_FRAME_ABSOLUTE,
            "model_action_frame": ACTION_FRAME_NORMALIZED,
            "action_space_size": ACTION_SPACE_SIZE,
            "identity_players": sorted(
                player for player, flag in self.is_identity.items() if flag
            ),
            "implementation": "stratego.model.action_frame",
        }


# ---------------------------------------------------------------------------
# Configuration and metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoordinatorConfig:
    """One end-to-end pipeline configuration."""

    num_environments: int
    num_workers: int
    inference_batch_size: int
    precision: str = "float16"
    legality: str = "dense"
    #: Frame the model's policy logits are indexed in. Defaults to the Phase 3
    #: behaviour; `model_contract_v2` candidates use `ACTION_FRAME_NORMALIZED`.
    action_frame: str = ACTION_FRAME_ABSOLUTE
    compact_capacity: int = POLICY_CAPACITY
    root_seed: int = 0
    sampling_seed: int = 12345
    model_seed: int = 0
    rules: RulesConfig = TRAINING_RULES
    record_trajectories: bool = False
    snapshot_interval: int = 32
    #: Phase 6B durable persistence. `None` is the accepted Phase 3/6 behaviour:
    #: sealed records are encoded, measured and dropped, and no worker touches
    #: the filesystem. Setting a directory turns on per-worker shard writing.
    trajectory_output_directory: str | None = None
    #: Compress persisted records with the repository's existing zlib helper.
    #: Only meaningful alongside `trajectory_output_directory`.
    compress_trajectories: bool = False
    #: Shard rollover size, and the run tag stamped into shard filenames.
    shard_target_bytes: int = 128 * 1024 * 1024
    run_id: str = "run"
    collection_policy_version: str = DEFAULT_COLLECTION_POLICY_VERSION
    collection_checkpoint_id: str | None = None
    verify_target_decisions: int = 0
    max_concurrent_verifications: int = 2
    retain_games: int = 0
    #: Synchronise the device between the encoder and the sampling stage so the
    #: two can be timed apart. Costs a little throughput; the benchmark measures
    #: how much rather than assuming it is free.
    detailed_timing: bool = True
    #: Check every sampled action against the mask it was drawn from before the
    #: workers see it. One gather per step; it names a sampling fault at the
    #: point it happens instead of letting it surface as a worker fault.
    verify_sampled_legality: bool = True
    step_timeout: float = 300.0

    def __post_init__(self) -> None:
        if self.precision not in DTYPE_BY_NAME:
            raise ValueError(f"unknown precision {self.precision!r}")
        if self.legality not in LEGALITY_MODES:
            raise ValueError(f"unknown legality mode {self.legality!r}")
        if self.action_frame not in ACTION_FRAMES:
            raise ValueError(f"unknown model action frame {self.action_frame!r}")
        if self.compact_capacity > POLICY_CAPACITY:
            raise ValueError(
                f"compact capacity {self.compact_capacity} exceeds the shared "
                f"policy row width of {POLICY_CAPACITY}"
            )
        if self.inference_batch_size < 1:
            raise ValueError("inference batch size must be positive")

    @property
    def label(self) -> str:
        return (
            f"w{self.num_workers}_e{self.num_environments}"
            f"_b{self.inference_batch_size}_{self.precision}_{self.legality}"
        )

    def as_dict(self) -> dict:
        return {
            "num_workers": self.num_workers,
            "num_environments": self.num_environments,
            "inference_batch_size": self.inference_batch_size,
            "precision": self.precision,
            "legality": self.legality,
            "action_frame": self.action_frame,
            "compact_capacity": self.compact_capacity,
            "record_trajectories": self.record_trajectories,
            "snapshot_interval": self.snapshot_interval,
            "detailed_timing": self.detailed_timing,
            "trajectory_output_directory": self.trajectory_output_directory,
            "compress_trajectories": self.compress_trajectories,
            "shard_target_bytes": self.shard_target_bytes,
            "run_id": self.run_id,
            "label": self.label,
        }


@dataclass
class StepMetrics:
    """Where one global step's wall time went."""

    step: int = 0
    positions: int = 0
    transitions: int = 0
    terminals: int = 0
    resets: int = 0
    chunks: int = 0
    wall_seconds: float = 0.0
    observation_seconds: float = 0.0
    legality_seconds: float = 0.0
    transfer_seconds: float = 0.0
    #: Absolute <-> normalized conversion of the dense masks, the compact
    #: identifiers and the selected actions. Exactly zero in the absolute frame.
    frame_seconds: float = 0.0
    inference_seconds: float = 0.0
    sampling_seconds: float = 0.0
    writeback_seconds: float = 0.0
    worker_seconds: float = 0.0
    barrier_seconds: float = 0.0
    straggler_seconds: float = 0.0
    worker_busy_seconds: float = 0.0

    @property
    def coordinator_seconds(self) -> float:
        return (
            self.observation_seconds
            + self.legality_seconds
            + self.transfer_seconds
            + self.frame_seconds
            + self.inference_seconds
            + self.sampling_seconds
            + self.writeback_seconds
        )


@dataclass
class RunTotals:
    """Accumulated totals over a measurement block."""

    steps: int = 0
    positions: int = 0
    transitions: int = 0
    terminals: int = 0
    resets: int = 0
    chunks: int = 0
    wall_seconds: float = 0.0
    observation_seconds: float = 0.0
    legality_seconds: float = 0.0
    transfer_seconds: float = 0.0
    frame_seconds: float = 0.0
    inference_seconds: float = 0.0
    sampling_seconds: float = 0.0
    writeback_seconds: float = 0.0
    worker_seconds: float = 0.0
    barrier_seconds: float = 0.0
    straggler_seconds: float = 0.0
    worker_busy_seconds: float = 0.0
    step_latencies: list[float] = field(default_factory=list)
    terminal_reason_counts: dict = field(default_factory=dict)
    games: int = 0

    @property
    def coordinator_seconds(self) -> float:
        """Everything the coordinator did itself, excluding the worker phase."""
        return (
            self.observation_seconds
            + self.legality_seconds
            + self.transfer_seconds
            + self.frame_seconds
            + self.inference_seconds
            + self.sampling_seconds
            + self.writeback_seconds
        )

    def add(self, metrics: StepMetrics) -> None:
        self.steps += 1
        self.positions += metrics.positions
        self.transitions += metrics.transitions
        self.terminals += metrics.terminals
        self.resets += metrics.resets
        self.chunks += metrics.chunks
        self.wall_seconds += metrics.wall_seconds
        self.observation_seconds += metrics.observation_seconds
        self.legality_seconds += metrics.legality_seconds
        self.transfer_seconds += metrics.transfer_seconds
        self.frame_seconds += metrics.frame_seconds
        self.inference_seconds += metrics.inference_seconds
        self.sampling_seconds += metrics.sampling_seconds
        self.writeback_seconds += metrics.writeback_seconds
        self.worker_seconds += metrics.worker_seconds
        self.barrier_seconds += metrics.barrier_seconds
        self.straggler_seconds += metrics.straggler_seconds
        self.worker_busy_seconds += metrics.worker_busy_seconds
        self.step_latencies.append(metrics.wall_seconds)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def resolve_device(requested: str | None = None) -> torch.device:
    """Pick the device, defaulting to Metal and refusing to fake it."""
    if requested is not None:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    raise CoordinatorError(
        "Metal is not available. The end-to-end decision must not be taken on "
        "central-processing-unit inference numbers; pass an explicit device only "
        "for tests."
    )


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":  # pragma: no cover - not this platform
        torch.cuda.synchronize()


def mps_memory_bytes() -> dict:
    """Metal allocator counters, empty when Metal is not in use."""
    if not torch.backends.mps.is_available():
        return {}
    try:
        return {
            "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        }
    except Exception:  # pragma: no cover - allocator counters are best effort
        return {}


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class SelfPlayCoordinator:
    """Drives the bulk-synchronous cycle and owns the only model instance."""

    def __init__(
        self,
        config: CoordinatorConfig,
        *,
        device: torch.device | str | None = None,
        model_config: RepresentativeConfig | None = None,
        model: torch.nn.Module | None = None,
        model_label: str | None = None,
    ) -> None:
        self.config = config
        self.device = (
            device if isinstance(device, torch.device) else resolve_device(device)
        )
        self.dtype = DTYPE_BY_NAME[config.precision]

        recording = RecordingConfig(
            enabled=config.record_trajectories,
            snapshot_interval=config.snapshot_interval,
            collection_policy_version=config.collection_policy_version,
            collection_checkpoint_id=config.collection_checkpoint_id,
            verify_target_decisions=config.verify_target_decisions,
            max_concurrent_verifications=config.max_concurrent_verifications,
            retain_games=config.retain_games,
            compress_records=config.compress_trajectories,
            output_directory=config.trajectory_output_directory,
            shard_target_bytes=config.shard_target_bytes,
            run_id=config.run_id,
        )
        self.pool = WorkerPool(
            config.num_environments,
            config.num_workers,
            root_seed=config.root_seed,
            rules=config.rules,
            step_timeout=config.step_timeout,
            recording=recording,
        )

        # Either the Phase 3 representative probe (the default, so every Phase 3
        # measurement still means what it meant) or a real network supplied by
        # the caller. An injected model is moved and cast here rather than by the
        # caller, so the coordinator remains the only owner of a device model.
        if model is None:
            self.model = build_representative_model(
                model_config,
                seed=config.model_seed,
                device=self.device,
                dtype=self.dtype,
            )
            self.model_label = model_label or "representative_benchmark_probe"
        else:
            if model_config is not None:
                raise CoordinatorError(
                    "pass either a built model or a RepresentativeConfig, not both"
                )
            self.model = model.to(device=self.device, dtype=self.dtype)
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            self.model_label = model_label or type(model).__name__

        self.frame = (
            NormalizedActionFrame(self.device)
            if config.action_frame == ACTION_FRAME_NORMALIZED
            else None
        )
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(config.sampling_seed)

        self.step_index = 0
        self.totals = RunTotals()
        self.games_finished = 0
        self.total_game_moves = 0
        self.terminal_reason_counts: dict[str, int] = {}
        self.errors: list[str] = []
        self._started = False

        # Reusable host-side staging so a step does not allocate per chunk.
        self._actions = np.full(config.num_environments, SKIP_ACTION, dtype=np.int32)
        self._model_actions = np.full(config.num_environments, SKIP_ACTION, dtype=np.int64)
        self._counted_episodes = np.zeros(config.num_environments, dtype=np.int64)

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self.pool.start()
        self._started = True

    def shutdown(self) -> dict:
        totals = self.pool.shutdown() if self._started else {}
        self._started = False
        return totals

    def __enter__(self) -> "SelfPlayCoordinator":
        self.start()
        return self

    def __exit__(self, *exception) -> None:
        self.shutdown()

    # -- one global step ----------------------------------------------------

    def step(self) -> StepMetrics:
        """Run one complete global step of the required cycle."""
        if not self._started:
            raise CoordinatorError("coordinator has not been started")
        config = self.config
        buffers = self.pool.buffers
        metrics = StepMetrics(step=self.step_index)
        step_started = time.perf_counter()

        # -- ready state: which rows the workers published as needing a move --
        gather_started = time.perf_counter()
        status = buffers.status
        active = np.flatnonzero(status == STATUS_ACTIVE)
        all_active = active.size == config.num_environments
        metrics.positions = int(active.size)
        if active.size == 0:
            raise CoordinatorError("no active environment in the batch")

        observations = buffers.observations
        # `(N, 127, 10, 10)` -> `(N, 127, 100)` is a pure view; the transpose to
        # token-major happens on the device, where it costs a device copy rather
        # than a 50 KB-per-row host copy.
        flat = observations.reshape(config.num_environments, observations.shape[1], -1)
        if all_active:
            observation_rows = flat
            mask_rows = buffers.legal_mask
            acting_rows = buffers.acting_player
        else:
            observation_rows = flat[active]
            mask_rows = buffers.legal_mask[active]
            acting_rows = buffers.acting_player[active]
        metrics.observation_seconds = time.perf_counter() - gather_started

        # -- compact legal set, when a stored decision will need it ------------
        legality_started = time.perf_counter()
        compact_ids = compact_valid = None
        need_compact = config.record_trajectories or config.legality == "compact"
        if need_compact:
            compact_ids, compact_valid, _ = compact_legality_from_masks(
                mask_rows, capacity=config.compact_capacity
            )
        metrics.legality_seconds = time.perf_counter() - legality_started

        # -- inference in chunks of the configured batch size ------------------
        chunk_size = min(config.inference_batch_size, int(active.size))
        sampled = np.empty(active.size, dtype=np.int64)
        model_actions = np.empty(active.size, dtype=np.int64)
        values = np.empty((active.size, VALUE_CLASSES), dtype=np.float32)
        probabilities = (
            np.zeros((active.size, config.compact_capacity), dtype=np.float32)
            if config.record_trajectories
            else None
        )

        for start in range(0, int(active.size), chunk_size):
            stop = min(start + chunk_size, int(active.size))
            metrics.chunks += 1
            self._run_chunk(
                metrics,
                observation_rows[start:stop],
                mask_rows[start:stop],
                acting_rows[start:stop],
                None if compact_ids is None else compact_ids[start:stop],
                None if compact_valid is None else compact_valid[start:stop],
                sampled[start:stop],
                model_actions[start:stop],
                values[start:stop],
                None if probabilities is None else probabilities[start:stop],
            )

        # -- write the decision back into shared memory ------------------------
        writeback_started = time.perf_counter()
        self._actions.fill(SKIP_ACTION)
        self._actions[active] = sampled
        self._model_actions.fill(SKIP_ACTION)
        self._model_actions[active] = model_actions

        if config.verify_sampled_legality:
            # The engine is the backstop and will refuse an illegal action, but
            # it refuses it inside a worker, which surfaces as an opaque worker
            # fault a whole phase later. Checking the sampled actions against the
            # very masks they were drawn from costs one gather and names the
            # cause on the spot. This is what caught the `+inf` Gumbel draw.
            chosen = mask_rows[np.arange(active.size), sampled]
            if not chosen.all():
                offenders = np.flatnonzero(chosen == 0)[:5]
                details = ", ".join(
                    f"slot {int(active[position])} action {int(sampled[position])} "
                    f"(legal_count {int(buffers.legal_count[active[position]])})"
                    for position in offenders
                )
                raise CoordinatorError(
                    f"sampling returned {int((chosen == 0).sum())} action(s) the "
                    f"published legality mask forbids: {details}"
                )
        self.pool.set_actions(self._actions)
        self.pool.clear_decisions()
        if config.record_trajectories:
            capacity = config.compact_capacity
            if all_active:
                buffers.policy_probabilities[:, :capacity] = probabilities
                buffers.value_prediction[:] = values
                buffers.decision_valid[:] = 1
            else:
                buffers.policy_probabilities[active, :capacity] = probabilities
                buffers.value_prediction[active] = values
                buffers.decision_valid[active] = 1
        metrics.writeback_seconds = time.perf_counter() - writeback_started

        # -- workers advance, finalise and reset -------------------------------
        worker_started = time.perf_counter()
        report = self.pool.step(apply_actions=True, auto_reset=True)
        metrics.worker_seconds = time.perf_counter() - worker_started
        metrics.transitions = report.stepped
        metrics.terminals = report.terminals
        metrics.resets = report.resets
        metrics.barrier_seconds = report.wait_seconds
        metrics.straggler_seconds = report.straggler_seconds
        metrics.worker_busy_seconds = report.worker_busy_seconds

        if report.terminals:
            self._collect_finished()

        metrics.wall_seconds = time.perf_counter() - step_started
        self.step_index += 1
        self.totals.add(metrics)
        return metrics

    def _run_chunk(
        self,
        metrics: StepMetrics,
        observation_rows: np.ndarray,
        mask_rows: np.ndarray,
        acting_rows: np.ndarray,
        compact_ids: np.ndarray | None,
        compact_valid: np.ndarray | None,
        sampled_out: np.ndarray,
        model_actions_out: np.ndarray,
        values_out: np.ndarray,
        probabilities_out: np.ndarray | None,
    ) -> None:
        """One MPS dispatch: transfer, apply the frame, encode, apply legality, sample.

        In the absolute frame this is the accepted Phase 3 path unchanged. In the
        normalized frame two conversions bracket the network: the engine's dense
        legality is permuted into the model's frame *before* it is applied to the
        logits, and the selected identifier is permuted back *before* anything
        outside this method sees it. Both are timed into `frame_seconds`, so the
        cost of `model_contract_v2` is a measured number rather than an assumption.
        """
        config = self.config
        device = self.device
        detailed = config.detailed_timing
        frame = self.frame

        transfer_started = time.perf_counter()
        # `(B, 127, 100)` contiguous host block -> device, then token-major.
        tokens = torch.from_numpy(np.ascontiguousarray(observation_rows)).to(
            device, non_blocking=True
        )
        tokens = tokens.transpose(1, 2).contiguous().to(self.dtype)

        # `uint8` and `bool` are both one byte, so the reinterpretation is
        # zero-copy and the 10,000 bytes per row go straight from the shared
        # block to the device without a host-side conversion pass.
        dense_device = (
            dense_mask_to_bool(mask_rows.view(np.bool_)).to(device, non_blocking=True)
            if config.legality == "dense"
            else None
        )
        compact_ids_device = (
            torch.from_numpy(compact_ids).to(device, non_blocking=True)
            if compact_ids is not None
            else None
        )
        compact_valid_device = (
            torch.from_numpy(compact_valid).to(device, non_blocking=True)
            if compact_valid is not None
            else None
        )
        if detailed:
            synchronize(device)
        metrics.transfer_seconds += time.perf_counter() - transfer_started

        # -- engine frame -> model frame ---------------------------------------
        assignment: dict[int, np.ndarray] = {}
        model_ids_device = compact_ids_device
        if frame is not None:
            frame_started = time.perf_counter()
            assignment = frame.split_rows(acting_rows)
            if dense_device is not None:
                dense_device = frame.masks_to_model(dense_device, assignment)
            if compact_ids_device is not None:
                model_ids_device = frame.action_ids_to_model(
                    compact_ids_device, assignment
                )
            if detailed:
                synchronize(device)
            metrics.frame_seconds += time.perf_counter() - frame_started

        legality_tensor: torch.Tensor | CompactLegality
        if config.legality == "dense":
            legality_tensor = dense_device
        else:
            legality_tensor = CompactLegality(
                action_ids=model_ids_device,
                valid=compact_valid_device,
                counts=torch.empty(0, dtype=torch.int64),
            )

        inference_started = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(tokens)
        if detailed:
            synchronize(device)
        metrics.inference_seconds += time.perf_counter() - inference_started

        sampling_started = time.perf_counter()
        with torch.no_grad():
            if config.legality == "dense":
                actions = sample_dense(
                    outputs.policy_logits,
                    legality_tensor,
                    generator=self.generator,
                )
            else:
                actions = sample_compact(
                    outputs.policy_logits,
                    legality_tensor,
                    generator=self.generator,
                )
            value_probabilities = torch.softmax(
                outputs.value_logits.to(torch.float32), dim=1
            )

            if probabilities_out is not None:
                if config.legality == "compact":
                    legal_probabilities = compact_legal_probabilities(
                        outputs.policy_logits, legality_tensor
                    )
                else:
                    # The dense path still owes the record one probability per
                    # legal action, so the compact identifiers built above are
                    # used to gather them out of the dense distribution. Under
                    # the normalized frame the gather is by the *model*
                    # identifiers, while the record keeps the ascending absolute
                    # order the worker's own legal list is in.
                    dense_probabilities = dense_legal_probabilities(
                        outputs.policy_logits, legality_tensor
                    )
                    legal_probabilities = dense_probabilities.gather(1, model_ids_device)
                    legal_probabilities = legal_probabilities * compact_valid_device.to(
                        legal_probabilities.dtype
                    )
                probabilities_out[:] = legal_probabilities.to("cpu").numpy()
        metrics.sampling_seconds += time.perf_counter() - sampling_started

        # -- model frame -> engine frame ---------------------------------------
        if frame is None:
            engine_actions = actions
        else:
            frame_started = time.perf_counter()
            engine_actions = frame.actions_to_absolute(actions, assignment)
            if detailed:
                # Without this the conversion is timed as a dispatch and its
                # device work is charged to whichever later statement happens to
                # synchronise first, which is exactly the accounting the
                # `frame_conversion_fraction` column exists to avoid.
                synchronize(device)
            metrics.frame_seconds += time.perf_counter() - frame_started

        readback_started = time.perf_counter()
        # Reading the actions back is what actually forces the device to finish,
        # so the step always ends synchronised. Every device-to-host copy in the
        # chunk is inside this one timed region -- a readback left between two
        # timers would be charged to neither while absorbing all the pending
        # device work, and the reported fractions would not sum.
        sampled_out[:] = engine_actions.to("cpu").numpy()
        values_out[:] = value_probabilities.to("cpu").numpy()
        if frame is None:
            model_actions_out[:] = -1
        else:
            model_actions_out[:] = actions.to("cpu").numpy()
        metrics.sampling_seconds += time.perf_counter() - readback_started

    # -- finished games -----------------------------------------------------

    def _collect_finished(self) -> None:
        """Tally the games that finished in the phase just completed.

        Read straight out of the `last_*` shared fields, which a reset does not
        overwrite, so an outcome cannot be lost to an immediate reset and no
        round trip to a worker is needed. `episode_count` is monotonic per slot,
        so the difference against what has already been counted is how many
        games that slot finished since the last look.
        """
        buffers = self.pool.buffers
        episodes = buffers.episode_count.astype(np.int64)
        newly = episodes - self._counted_episodes
        for slot in np.flatnonzero(newly > 0).tolist():
            reason = terminal_reason_name(int(buffers.last_terminal_reason[slot]))
            count = int(newly[slot])
            self.terminal_reason_counts[reason] = (
                self.terminal_reason_counts.get(reason, 0) + count
            )
            self.games_finished += count
            # A slot finishes at most one game per phase, so `last_total_moves`
            # is the length of the game the increment refers to. Accumulated
            # here because the mean game length is a reporting requirement and
            # the `last_*` fields are the only place the length survives a reset.
            self.total_game_moves += count * int(buffers.last_total_moves[slot])
        self._counted_episodes = episodes

    # -- convenience --------------------------------------------------------

    def run_steps(self, steps: int) -> RunTotals:
        for _ in range(steps):
            self.step()
        return self.totals

    @property
    def last_actions(self) -> np.ndarray:
        """The dense *absolute* action vector written for the most recent step."""
        return self._actions

    @property
    def last_model_actions(self) -> np.ndarray:
        """The normalized identifiers the model actually selected, per slot.

        All `SKIP_ACTION` in the absolute frame, where the model's selection and
        the engine's action are the same identifier and there is nothing to
        distinguish. In the normalized frame this is what the correctness gate
        checks the inverse conversion against.
        """
        return self._model_actions

    def infer_batch(
        self,
        observation_rows: np.ndarray,
        mask_rows: np.ndarray,
        acting_rows: np.ndarray,
        *,
        compact_ids: np.ndarray | None = None,
        compact_valid: np.ndarray | None = None,
        record_probabilities: bool = False,
    ) -> dict:
        """Run exactly one inference chunk and return what it produced.

        The same `_run_chunk` a global step calls, on a caller-supplied batch and
        with no worker phase around it. This is what makes the denominator of the
        bottleneck ratio a measurement of *this* pipeline's inference stage --
        transfer, frame conversion, forward, legality, sampling and readback --
        rather than of a separately written benchmark that resembles it.
        """
        rows = int(observation_rows.shape[0])
        metrics = StepMetrics()
        sampled = np.empty(rows, dtype=np.int64)
        model_actions = np.empty(rows, dtype=np.int64)
        values = np.empty((rows, VALUE_CLASSES), dtype=np.float32)
        probabilities = (
            np.zeros((rows, self.config.compact_capacity), dtype=np.float32)
            if record_probabilities
            else None
        )
        self._run_chunk(
            metrics,
            observation_rows,
            mask_rows,
            acting_rows,
            compact_ids,
            compact_valid,
            sampled,
            model_actions,
            values,
            probabilities,
        )
        return {
            "absolute_actions": sampled,
            "model_actions": model_actions,
            "value_probabilities": values,
            "legal_probabilities": probabilities,
            "metrics": metrics,
        }

    def snapshot(self) -> dict:
        """A cheap, allocation-free view of run-level counters."""
        totals = self.totals
        return {
            "steps": totals.steps,
            "positions": totals.positions,
            "transitions": totals.transitions,
            "games": self.games_finished,
            "resets": totals.resets,
            "wall_seconds": totals.wall_seconds,
            "terminal_reason_counts": dict(self.terminal_reason_counts),
        }


__all__ = [
    "ACTION_FRAMES",
    "ACTION_FRAME_ABSOLUTE",
    "ACTION_FRAME_NORMALIZED",
    "COORDINATOR_VERSION",
    "DTYPE_BY_NAME",
    "LEGALITY_MODES",
    "PRECISIONS",
    "ActionFrameMismatchError",
    "CoordinatorConfig",
    "CoordinatorError",
    "NormalizedActionFrame",
    "RunTotals",
    "SelfPlayCoordinator",
    "StepMetrics",
    "compact_legality_from_masks",
    "mps_memory_bytes",
    "resolve_device",
    "synchronize",
]
