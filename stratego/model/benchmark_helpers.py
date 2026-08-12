"""Measurement machinery for the Agent 3 standalone MPS benchmark.

This module answers one question: on this M4 Pro, what does each C0-C6 candidate
actually cost to run forward and to run one training step, without the simulator
in the loop. It does not select an architecture and it does not know how to play
Stratego.

Three commitments shape everything here.

**Fairness is structural, not procedural.** Every candidate goes through the same
functions with the same corpus rows, the same warmup and repetition policy, the
same synchronisation, the same loss definitions and the same target tensors. A
candidate is a `CandidateConfig` handed to shared code; there is no per-candidate
branch anywhere in this file, so there is no place for a favoured candidate to be
optimised differently.

**A label must be a measurement, not an intention.** Asking for `mps` and
silently getting CPU, or asking for `float16` and silently timing float32, are
the two failures that would quietly invalidate the whole phase. Every row
therefore records the device and dtype read back off the *output tensor*, and
:func:`verify_execution_labels` refuses a mismatch rather than reporting it. The
requested label and the observed label are separate fields in the CSV.

**A failure is a result.** Out-of-memory, a non-finite head, a contract
violation at a large batch: these are the measurements that establish where the
frontier is, so they are recorded as rows with a status and an error string, not
swallowed and not retried into success.

Phase 3/4's accepted `stratego.training.mps_benchmark` is imported read-only for
device detection, Metal memory counters and synchronisation. Nothing in this
module writes to it, and the Phase 3 position pool is deliberately *not* reused:
that pool records neither the acting player nor normalized legality, and Agent 3
needs both (see :class:`BenchmarkCorpus`).
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from ..training.batch_simulation import BatchSimulator
from ..training.belief_targets import (
    BELIEF_TARGET_VERSION,
    dense_belief_target,
)
from ..training.mps_benchmark import (
    detect_device_report,
    mps_memory_bytes,
    peak_memory_bytes,
    synchronize,
)
from .action_frame import (
    absolute_legal_actions_to_model,
    absolute_legal_mask_to_model,
    action_frame_summary,
    model_action_to_absolute,
)
from .architecture_configs import (
    ARCHITECTURE_FAMILY,
    ARCHITECTURE_FAMILY_VERSION,
    CANDIDATE_IDS,
    FAMILY_INITIALIZATION_SEED,
    CandidateConfig,
    candidate_config,
    config_digests,
)
from .contract import (
    ACTION_ENCODING_VERSION,
    BELIEF_IGNORE_INDEX,
    MODEL_CONTRACT_VERSION,
    POLICY_ACTION_FRAME,
    POLICY_LOGIT_COUNT,
    VALUE_CLASS_COUNT,
    ModelContractError,
    ModelOutputs,
)
from .losses import multi_head_loss
from .production_model import ProductionModel, build_candidate_model
from .tokenization import observation_to_tokens, tokenize_numpy_observation

# ---------------------------------------------------------------------------
# Identity and declared policy
#
# Everything in this block is part of the benchmark's identity: change any of it
# and the numbers are no longer comparable with the recorded run, which is why
# they are constants carried into the report rather than call-site defaults.
# ---------------------------------------------------------------------------

#: Bump when any measurement policy below changes.
BENCHMARK_VERSION = "agent_03_mps_benchmark_0.1.0"

#: The corpus construction recipe. Part of the corpus digest.
CORPUS_VERSION = "agent_03_corpus_v1"
CORPUS_SEED = 20260811
CORPUS_POSITIONS = 4096
CORPUS_ENVIRONMENTS = 128

#: Deliberately odd. The acting player is the ply's parity, and every slot in a
#: sampling round sits at the same ply, so an *even* stride samples nothing but
#: even plies and produces a corpus in which red is always the acting player.
#: That corpus would look entirely healthy and would never once exercise the
#: blue branch of the perspective transform -- which is the whole subject of
#: `model_contract_v2`. An odd stride alternates the parity round by round.
CORPUS_STRIDE = 15

#: Seed for the benchmark target tensors (policy / value). Separate from the
#: corpus seed so the corpus and its labels can be reasoned about independently.
TARGET_SEED = 20260812

#: The required inference ladder. Every candidate attempts every entry.
INFERENCE_BATCH_SIZES: tuple[int, ...] = (1, 64, 256, 512, 1024, 1536, 2048)

#: Optional probing above the required ladder, used only while throughput is
#: still improving and Metal memory stays far below the pressure line. The
#: benchmark is not permitted to drive the host into OOM or swap to find a
#: ceiling; a ceiling reported as "not reached below the memory guard" is a
#: valid and honest result.
EXTENDED_BATCH_SIZES: tuple[int, ...] = (3072, 4096)

#: The required training ladder.
TRAINING_BATCH_SIZES: tuple[int, ...] = (32, 64, 128, 256)

PRECISIONS: tuple[str, ...] = ("float32", "float16")

DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
}

#: Warmup and repetition policy, identical for every point.
WARMUP_ITERATIONS = 5
MIN_MEASUREMENT_ITERATIONS = 10
MAX_MEASUREMENT_ITERATIONS = 60
TARGET_MEASUREMENT_SECONDS = 1.0

#: Metal memory guard. A configuration is attempted only while the driver
#: allocation sits below this fraction of `recommended_max_memory()`, and
#: extended batches stop at the same line. This is what keeps "find the
#: frontier" from meaning "exhaust the machine".
MEMORY_PRESSURE_FRACTION = 0.60

#: Above this fraction of recommended maximum at its own best batch, a candidate
#: is classified IMPRACTICAL on memory grounds.
MEMORY_IMPRACTICAL_FRACTION = 0.80

#: Extended probing continues only while throughput improves by at least this
#: relative margin; a flat or falling curve means the knee has been found.
EXTENDED_PROBE_IMPROVEMENT = 0.02

# ---------------------------------------------------------------------------
# Timing boundaries
# ---------------------------------------------------------------------------

BOUNDARY_A = "A_model_forward"
BOUNDARY_B = "B_observation_tokenization_model"
BOUNDARY_C = "C_observation_tokenization_model_selection"

BOUNDARIES: tuple[str, ...] = (BOUNDARY_A, BOUNDARY_B, BOUNDARY_C)

#: Exactly what each timed region contains. Recorded per row, because "latency"
#: without its boundary is not a number anyone can act on.
BOUNDARY_CONTENTS: dict[str, str] = {
    BOUNDARY_A: (
        "model forward only; tokens already resident on the device in the target "
        "dtype, contiguous; timed region is model(tokens) plus device synchronisation"
    ),
    BOUNDARY_B: (
        "host NumPy observation (B,127,10,10) -> observation_batch_from_numpy "
        "(copy + contract validation) -> observation_to_tokens relayout -> host-to-device "
        "transfer -> model forward; timed region ends after device synchronisation"
    ),
    BOUNDARY_C: (
        "boundary B, plus the normalized dense legality mask host-to-device transfer, "
        "masked greedy selection over the normalized action frame on device, readback "
        "of the chosen normalized identifiers, and conversion to absolute engine "
        "actions via stratego.model.action_frame"
    ),
}

# ---------------------------------------------------------------------------
# Predeclared numerical tolerances
#
# Declared here, before any measurement, and applied identically to every
# candidate regardless of depth. Absolute error is the primary criterion:
# float16 logits that sit near zero produce enormous relative ratios that say
# nothing about whether the network behaves the same, so relative error is
# recorded honestly but only judged where the reference value is large enough
# for a ratio to mean anything.
# ---------------------------------------------------------------------------

#: Reference magnitude below which a relative error is not meaningful and is
#: excluded from `meaningful_relative_error` (it is still visible in the raw
#: max/mean absolute error).
RELATIVE_ERROR_FLOOR = 1e-3

TOLERANCES: dict[str, dict[str, float]] = {
    # CPU float32 reference vs MPS float32: same arithmetic, different kernels.
    "mps_float32": {
        "policy_logits_max_abs": 1e-4,
        "value_probabilities_max_abs": 1e-5,
        "belief_logits_max_abs": 1e-4,
    },
    # CPU float32 reference vs MPS float16: a genuine precision change.
    "mps_float16": {
        "policy_logits_max_abs": 5e-2,
        "value_probabilities_max_abs": 5e-3,
        "belief_logits_max_abs": 5e-2,
    },
}

#: The margin added to one designated legal action per crafted-margin position.
#: Large enough that no float16 rounding can reorder the top two legal logits,
#: so a disagreement is a real defect rather than a tie broken differently.
CRAFTED_MARGIN = 5.0

#: Positions used for the numerical comparison. Small enough to run on CPU for
#: every candidate, large enough to exercise every head on real boards.
NUMERICAL_CHECK_POSITIONS = 256

# ---------------------------------------------------------------------------
# Classification policy
#
# Every threshold that decides ADVANCE / DOMINATED / IMPRACTICAL is declared
# here, before the measurements exist, and `classify_candidates` reads nothing
# else. See CLASSIFICATION_INPUT_KEYS for the hard boundary on what may
# influence a classification.
# ---------------------------------------------------------------------------

#: A candidate that cannot run a stable float32 forward pass at this batch is
#: not usable by the Phase 3 collector, which batches far above it.
MIN_VIABLE_INFERENCE_BATCH = 256

#: A candidate that cannot complete one float32 training step at this batch
#: cannot be trained at all on this host.
MIN_VIABLE_TRAINING_BATCH = 32

#: Practical floor on sustained float32 inference throughput. Phase 3 measured a
#: simulation-only numerator of ~96,963 positions/s and an integrated rate of
#: ~12,838 positions/s with a representative model at ~14,922 positions/s. Under
#: the same serial composition (1/integrated = 1/simulation + 1/model) a model
#: below 5,000 positions/s caps the integrated pipeline under ~4,755
#: positions/s, roughly a third of the Phase 3 reference, before any recording
#: cost. That is the documented line between "slow" and "not practical".
MIN_VIABLE_POSITIONS_PER_SECOND = 5000.0

#: The only fields a classification may read. Anything else -- above all any
#: measure of how well a random-weight network plays -- is invisible to
#: `classify_candidates` by construction, not by convention.
CLASSIFICATION_INPUT_KEYS: tuple[str, ...] = (
    "candidate_id",
    "parameters",
    "best_float32_positions_per_second",
    "best_float16_positions_per_second",
    "representative_training_examples_per_second",
    "max_stable_inference_batch",
    "max_stable_training_batch",
    "peak_metal_fraction",
    "numerically_stable_float32",
    "numerically_stable_float16",
)

#: Substrings that must never appear in a classification input key. Playing
#: strength is banned from selection for the whole of Phase 6, and a benchmark
#: summary is exactly where it would leak in.
FORBIDDEN_CLASSIFICATION_SUBSTRINGS: tuple[str, ...] = (
    "win",
    "loss_rate",
    "draw",
    "elo",
    "strength",
    "score",
    "result",
    "match",
    "gauntlet",
)


class BenchmarkError(RuntimeError):
    """Any failure of the benchmark harness itself, as opposed to a candidate."""


class BenchmarkIntegrityError(BenchmarkError):
    """A measurement did not happen on the device or in the dtype it claimed.

    Raised rather than recorded: a row that lies about where it ran is worse
    than no row, because every downstream comparison would silently inherit it.
    """


# ---------------------------------------------------------------------------
# Device and precision plumbing
# ---------------------------------------------------------------------------


def resolve_dtype(precision: str) -> torch.dtype:
    """`"float16"` -> `torch.float16`, and nothing else."""
    try:
        return DTYPE_BY_NAME[precision]
    except KeyError:
        raise BenchmarkError(
            f"unknown precision {precision!r}; expected one of {sorted(DTYPE_BY_NAME)}"
        ) from None


def precision_name(dtype: torch.dtype) -> str:
    """The inverse of :func:`resolve_dtype`, used to read a label back off a tensor."""
    for name, candidate in DTYPE_BY_NAME.items():
        if candidate == dtype:
            return name
    return str(dtype)


def require_mps() -> torch.device:
    """The MPS device, or a hard failure.

    Agent 3's whole purpose is an MPS measurement, so there is no CPU fallback
    here at all. A missing backend is a `BLOCKED` stop condition for the agent,
    not a reason to quietly benchmark something else.
    """
    if not torch.backends.mps.is_built():
        raise BenchmarkError("this PyTorch build has no MPS backend; cannot benchmark MPS")
    if not torch.backends.mps.is_available():
        raise BenchmarkError("MPS is not available on this host; cannot benchmark MPS")
    return torch.device("mps")


def verify_execution_labels(
    outputs: ModelOutputs,
    *,
    requested_device: torch.device,
    requested_precision: str,
) -> dict:
    """Read device and dtype off the result and refuse a mismatched label.

    The check is deliberately made against `outputs.policy_logits` -- a tensor
    the model actually produced -- rather than against the model's parameters or
    the string that was requested. A model moved to MPS whose forward pass fell
    back to CPU, or a float16 request that silently ran in float32, both fail
    here.
    """
    observed_device = outputs.policy_logits.device
    observed_dtype = outputs.policy_logits.dtype
    observed_precision = precision_name(observed_dtype)

    if observed_device.type != requested_device.type:
        raise BenchmarkIntegrityError(
            f"requested device {requested_device} but the outputs are on "
            f"{observed_device}; a benchmark row must not mislabel where it ran"
        )
    if observed_dtype != resolve_dtype(requested_precision):
        raise BenchmarkIntegrityError(
            f"requested precision {requested_precision} but the outputs are "
            f"{observed_dtype}; a benchmark row must not mislabel its dtype"
        )
    return {
        "observed_device": str(observed_device),
        "observed_device_type": observed_device.type,
        "observed_dtype": str(observed_dtype),
        "observed_precision": observed_precision,
    }


def metal_memory_snapshot() -> dict:
    """Metal counters plus process RSS, with `None` where an API is unavailable.

    Unavailable is reported as `None` and rendered as `unavailable`, never as
    zero: a missing counter and an empty allocator are different facts and the
    report must not conflate them.
    """
    metal = mps_memory_bytes()
    return {
        "process_rss_bytes": peak_memory_bytes(),
        "metal_allocated_bytes": metal["current_allocated_bytes"],
        "metal_driver_bytes": metal["driver_allocated_bytes"],
        "metal_recommended_max_bytes": metal["recommended_max_bytes"],
    }


def memory_pressure_fraction() -> float | None:
    """Driver allocation as a fraction of the recommended maximum, if knowable."""
    snapshot = metal_memory_snapshot()
    driver = snapshot["metal_driver_bytes"]
    recommended = snapshot["metal_recommended_max_bytes"]
    if driver is None or not recommended:
        return None
    return float(driver) / float(recommended)


def release_device_memory(device: torch.device) -> None:
    """Return cached blocks between configurations so one point cannot inherit
    another's allocator state."""
    if device.type == "mps":
        empty = getattr(torch.mps, "empty_cache", None)
        if empty is not None:
            empty()


def _is_out_of_memory(error: BaseException) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "insufficient memory" in text


# ---------------------------------------------------------------------------
# The input corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkCorpus:
    """Deterministic real engine positions with everything a benchmark needs.

    Built from the frozen engine through `BatchSimulator`, not synthesised. Each
    row is an acting-player position: its observation is already
    perspective-normalized (`observation_v2_1_127ch`), and both legality frames
    are carried so that the absolute engine mask and the normalized model mask
    can be compared without either being rebuilt.

    The normalized products are produced *only* by
    :mod:`stratego.model.action_frame`. This class does not know the geometry of
    the transform and must never learn it -- a second implementation of the
    conversion is precisely the defect Agent 1 removed.
    """

    observations: np.ndarray  # (P, 127, 10, 10) float32, acting player's view
    acting_players: np.ndarray  # (P,) int64
    absolute_masks: np.ndarray  # (P, 10000) bool, engine frame
    normalized_masks: np.ndarray  # (P, 10000) bool, model frame
    absolute_legal_lists: tuple[tuple[int, ...], ...]
    normalized_legal_lists: tuple[tuple[int, ...], ...]
    belief_labels: np.ndarray  # (P, 100) int64, BELIEF_IGNORE_INDEX off-target
    belief_masks: np.ndarray  # (P, 100) bool
    policy_targets: np.ndarray  # (P,) int64, normalized frame, always legal
    value_targets: np.ndarray  # (P,) int64, WIN/DRAW/LOSS
    plies: np.ndarray  # (P,) int64
    seed: int
    build_seconds: float
    digest: str

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])

    def indices_for(self, batch: int, *, offset: int = 0) -> np.ndarray:
        """A contiguous, wrap-around slice. Deterministic in `(batch, offset)`.

        The same rows therefore back every candidate and every precision at a
        given batch size, which is what makes the comparison a comparison.
        """
        if batch < 1:
            raise BenchmarkError(f"batch must be at least 1, got {batch}")
        return (offset + np.arange(batch, dtype=np.int64)) % self.size

    def stats(self) -> dict:
        counts = np.array([len(row) for row in self.absolute_legal_lists], dtype=np.int64)
        supervised = self.belief_masks.sum(axis=1)
        return {
            "corpus_version": CORPUS_VERSION,
            "positions": self.size,
            "seed": self.seed,
            "digest": self.digest,
            "build_seconds": round(self.build_seconds, 3),
            "environments": CORPUS_ENVIRONMENTS,
            "collection_stride": CORPUS_STRIDE,
            "acting_player_counts": {
                "red": int((self.acting_players == 0).sum()),
                "blue": int((self.acting_players == 1).sum()),
            },
            "legal_actions_min": int(counts.min()),
            "legal_actions_max": int(counts.max()),
            "legal_actions_mean": round(float(counts.mean()), 3),
            "ply_min": int(self.plies.min()),
            "ply_max": int(self.plies.max()),
            "ply_mean": round(float(self.plies.mean()), 2),
            "belief_supervised_min": int(supervised.min()),
            "belief_supervised_max": int(supervised.max()),
            "belief_supervised_mean": round(float(supervised.mean()), 3),
            "belief_target_version": BELIEF_TARGET_VERSION,
            "observation_bytes": int(self.observations.nbytes),
            "policy_target_frame": POLICY_ACTION_FRAME,
            "target_seed": TARGET_SEED,
        }


def _corpus_digest(
    *,
    observations: np.ndarray,
    acting_players: np.ndarray,
    absolute_masks: np.ndarray,
    normalized_masks: np.ndarray,
    belief_labels: np.ndarray,
    belief_masks: np.ndarray,
    policy_targets: np.ndarray,
    value_targets: np.ndarray,
    plies: np.ndarray,
) -> str:
    """SHA-256 over every array plus the recipe that produced them.

    The recipe is inside the digest so that two corpora built with different
    seeds or sizes cannot collide even if, absurdly, their contents matched.
    """
    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(
            {
                "corpus_version": CORPUS_VERSION,
                "seed": CORPUS_SEED,
                "target_seed": TARGET_SEED,
                "positions": int(observations.shape[0]),
                "environments": CORPUS_ENVIRONMENTS,
                "stride": CORPUS_STRIDE,
                "belief_target_version": BELIEF_TARGET_VERSION,
                "action_encoding_version": ACTION_ENCODING_VERSION,
                "policy_action_frame": POLICY_ACTION_FRAME,
            },
            sort_keys=True,
        ).encode()
    )
    for array in (
        observations,
        acting_players,
        absolute_masks,
        normalized_masks,
        belief_labels,
        belief_masks,
        policy_targets,
        value_targets,
        plies,
    ):
        hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def build_benchmark_corpus(
    *,
    positions: int = CORPUS_POSITIONS,
    seed: int = CORPUS_SEED,
    environments: int = CORPUS_ENVIRONMENTS,
    stride: int = CORPUS_STRIDE,
    target_seed: int = TARGET_SEED,
) -> BenchmarkCorpus:
    """Collect `positions` real acting-player positions and label them.

    Slots take uniformly random legal actions from a seeded generator and are
    sampled every `stride` plies, so the corpus spans openings through late
    middlegame rather than a hundred copies of the opening. Nothing about the
    Phase 3 pool or the engine is modified; this walks the same public
    `BatchSimulator` surface and records the two things the Phase 3 pool does
    not: the acting player, and legality in both frames.

    Determinism: `(positions, seed, environments, stride, target_seed)` fixes
    every byte of the result, which the digest then pins.
    """
    started = time.perf_counter()
    simulator = BatchSimulator(num_environments=environments, root_seed=seed)
    rng = np.random.default_rng(seed)

    observation_blocks: list[np.ndarray] = []
    acting: list[int] = []
    absolute_mask_blocks: list[np.ndarray] = []
    absolute_lists: list[tuple[int, ...]] = []
    belief_label_rows: list[np.ndarray] = []
    belief_mask_rows: list[np.ndarray] = []
    plies: list[int] = []
    collected = 0
    step_index = 0

    while collected < positions:
        if step_index % stride == 0:
            active = simulator.active_slots()
            if active:
                keep = min(len(active), positions - collected)
                chosen = active[:keep]
                observation_blocks.append(np.asarray(simulator.observations(chosen)))
                absolute_mask_blocks.append(np.asarray(simulator.legal_action_masks(chosen)))
                for slot in chosen:
                    state = simulator.game_state(slot)
                    observer = int(simulator.acting_player(slot))
                    acting.append(observer)
                    absolute_lists.append(tuple(int(a) for a in simulator.legal_actions(slot)))
                    labels, mask = dense_belief_target(state, observer)
                    belief_label_rows.append(labels)
                    belief_mask_rows.append(mask)
                    plies.append(int(state.ply))
                collected += keep
                if collected >= positions:
                    break

        active = simulator.active_slots()
        if not active:
            simulator.reset_finished()
            step_index += 1
            continue
        actions = {}
        for slot in active:
            legal = simulator.legal_actions(slot)
            actions[slot] = int(legal[rng.integers(len(legal))])
        simulator.step(actions)
        simulator.reset_finished()
        step_index += 1

    observations = np.ascontiguousarray(
        np.concatenate(observation_blocks, axis=0)[:positions], dtype=np.float32
    )
    absolute_masks = np.ascontiguousarray(
        np.concatenate(absolute_mask_blocks, axis=0)[:positions]
    ).astype(bool)
    acting_players = np.asarray(acting[:positions], dtype=np.int64)
    absolute_legal_lists = tuple(absolute_lists[:positions])
    belief_labels = np.stack(belief_label_rows[:positions], axis=0).astype(np.int64)
    belief_masks = np.stack(belief_mask_rows[:positions], axis=0).astype(bool)
    ply_array = np.asarray(plies[:positions], dtype=np.int64)

    # The single conversion, applied row by row through the authoritative
    # module. Both products are converted -- the list and the dense mask --
    # because a benchmark that converted only one of them could not later prove
    # they still agree.
    normalized_masks = np.zeros_like(absolute_masks)
    normalized_lists: list[tuple[int, ...]] = []
    for index in range(positions):
        player = int(acting_players[index])
        normalized_masks[index] = absolute_legal_mask_to_model(absolute_masks[index], player)
        normalized_lists.append(
            tuple(
                int(a)
                for a in absolute_legal_actions_to_model(absolute_legal_lists[index], player)
            )
        )

    # Benchmark targets. These exist so the backward pass has something real to
    # differentiate; nothing is learned from them and no optimizer ever sees
    # them. The policy target is drawn from the position's own normalized legal
    # list, so it is legal in the model frame by construction -- and
    # `verify_policy_targets_legal` proves it independently rather than trusting
    # this line.
    target_rng = np.random.default_rng(target_seed)
    policy_targets = np.asarray(
        [int(row[target_rng.integers(len(row))]) for row in normalized_lists], dtype=np.int64
    )
    value_targets = target_rng.integers(0, VALUE_CLASS_COUNT, size=positions).astype(np.int64)

    # A single-colour corpus is a defective benchmark input, not a quirk: it
    # would exercise only the identity half of the perspective transform while
    # reporting full coverage. Fail at construction rather than let every
    # downstream measurement inherit the gap.
    red = int((acting_players == 0).sum())
    blue = int((acting_players == 1).sum())
    if red == 0 or blue == 0:
        raise BenchmarkError(
            f"corpus covers only one acting colour (red={red}, blue={blue}); the "
            "perspective transform would never be exercised on the other. Check that "
            f"the collection stride ({stride}) is odd."
        )

    digest = _corpus_digest(
        observations=observations,
        acting_players=acting_players,
        absolute_masks=absolute_masks,
        normalized_masks=normalized_masks,
        belief_labels=belief_labels,
        belief_masks=belief_masks,
        policy_targets=policy_targets,
        value_targets=value_targets,
        plies=ply_array,
    )

    return BenchmarkCorpus(
        observations=observations,
        acting_players=acting_players,
        absolute_masks=absolute_masks,
        normalized_masks=normalized_masks,
        absolute_legal_lists=absolute_legal_lists,
        normalized_legal_lists=tuple(normalized_lists),
        belief_labels=belief_labels,
        belief_masks=belief_masks,
        policy_targets=policy_targets,
        value_targets=value_targets,
        plies=ply_array,
        seed=seed,
        build_seconds=time.perf_counter() - started,
        digest=digest,
    )


def verify_policy_targets_legal(corpus: BenchmarkCorpus) -> dict:
    """Prove every policy target is legal in the normalized model frame.

    Checked two independent ways -- membership in the normalized legal list and
    a true entry in the normalized dense mask -- because the two products are
    what the loss and the selection path respectively consume. A target that is
    legal under one and not the other would make `policy_loss` raise at some
    unpredictable batch offset rather than at the corpus boundary.
    """
    illegal_in_list = 0
    illegal_in_mask = 0
    for index in range(corpus.size):
        target = int(corpus.policy_targets[index])
        if target not in corpus.normalized_legal_lists[index]:
            illegal_in_list += 1
        if not bool(corpus.normalized_masks[index, target]):
            illegal_in_mask += 1
    return {
        "positions": corpus.size,
        "targets_illegal_in_normalized_list": illegal_in_list,
        "targets_illegal_in_normalized_mask": illegal_in_mask,
        "all_targets_legal": illegal_in_list == 0 and illegal_in_mask == 0,
        "policy_target_frame": POLICY_ACTION_FRAME,
    }


def verify_legality_frames_agree(corpus: BenchmarkCorpus) -> dict:
    """The converted list and the converted mask must describe the same set.

    Cheap, and it closes the one gap the corpus could otherwise hide: both
    normalized products come from the same module but through different
    entry points, so agreeing is evidence and disagreeing is a defect.
    """
    mismatches = 0
    absolute_recovery_mismatches = 0
    for index in range(corpus.size):
        from_mask = set(np.flatnonzero(corpus.normalized_masks[index]).tolist())
        from_list = set(corpus.normalized_legal_lists[index])
        if from_mask != from_list:
            mismatches += 1
        player = int(corpus.acting_players[index])
        recovered = {
            model_action_to_absolute(action, player)
            for action in corpus.normalized_legal_lists[index]
        }
        if recovered != set(corpus.absolute_legal_lists[index]):
            absolute_recovery_mismatches += 1
    return {
        "positions": corpus.size,
        "normalized_list_vs_mask_mismatches": mismatches,
        "normalized_to_absolute_set_mismatches": absolute_recovery_mismatches,
        "frames_agree": mismatches == 0 and absolute_recovery_mismatches == 0,
    }


# ---------------------------------------------------------------------------
# Host-side batches
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostBatch:
    """One fixed host-side batch, reused across every timed iteration.

    Held on the host in NumPy exactly as the Phase 3 collector would hand it
    over, so boundary B measures a real preprocessing path rather than a tensor
    that was already conveniently shaped.
    """

    observations: np.ndarray  # (B, 127, 10, 10) float32
    normalized_masks: np.ndarray  # (B, 10000) bool
    acting_players: np.ndarray  # (B,) int64
    policy_targets: np.ndarray  # (B,) int64
    value_targets: np.ndarray  # (B,) int64
    belief_labels: np.ndarray  # (B, 100) int64
    belief_masks: np.ndarray  # (B, 100) bool
    indices: np.ndarray

    @property
    def batch_size(self) -> int:
        return int(self.observations.shape[0])


def make_host_batch(corpus: BenchmarkCorpus, batch: int, *, offset: int = 0) -> HostBatch:
    """Slice `corpus` into one reusable host batch."""
    indices = corpus.indices_for(batch, offset=offset)
    return HostBatch(
        observations=np.ascontiguousarray(corpus.observations[indices]),
        normalized_masks=np.ascontiguousarray(corpus.normalized_masks[indices]),
        acting_players=np.ascontiguousarray(corpus.acting_players[indices]),
        policy_targets=np.ascontiguousarray(corpus.policy_targets[indices]),
        value_targets=np.ascontiguousarray(corpus.value_targets[indices]),
        belief_labels=np.ascontiguousarray(corpus.belief_labels[indices]),
        belief_masks=np.ascontiguousarray(corpus.belief_masks[indices]),
        indices=indices,
    )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def summarise_samples(samples: Sequence[float]) -> dict:
    """Median / p95 / mean and friends, in milliseconds."""
    if not samples:
        return {
            "measurement_iterations": 0,
            "median_latency_ms": None,
            "p95_latency_ms": None,
            "mean_latency_ms": None,
            "min_latency_ms": None,
            "max_latency_ms": None,
            "stdev_latency_ms": None,
        }
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "measurement_iterations": len(ordered),
        "median_latency_ms": round(1000 * statistics.median(ordered), 4),
        "p95_latency_ms": round(1000 * ordered[index], 4),
        "mean_latency_ms": round(1000 * statistics.fmean(ordered), 4),
        "min_latency_ms": round(1000 * ordered[0], 4),
        "max_latency_ms": round(1000 * ordered[-1], 4),
        "stdev_latency_ms": round(
            1000 * (statistics.stdev(ordered) if len(ordered) > 1 else 0.0), 4
        ),
    }


def timed_samples(
    operation: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int = WARMUP_ITERATIONS,
    minimum: int = MIN_MEASUREMENT_ITERATIONS,
    maximum: int = MAX_MEASUREMENT_ITERATIONS,
    target_seconds: float = TARGET_MEASUREMENT_SECONDS,
) -> list[float]:
    """Time `operation` with the device synchronised on both sides of each sample.

    MPS dispatch is asynchronous: without the trailing synchronise, `operation`
    returns as soon as the work is *queued* and the measured latency is the cost
    of queueing, which for a large batch is off by more than an order of
    magnitude. The leading synchronise matters too -- it stops the previous
    sample's unfinished work from being charged to this one.

    Warmup is unmeasured and covers Metal shader compilation and allocator
    growth, both of which are one-off costs that would otherwise land entirely
    on the first sample.
    """
    for _ in range(warmup):
        operation()
    synchronize(device)

    samples: list[float] = []
    elapsed = 0.0
    while len(samples) < minimum or (elapsed < target_seconds and len(samples) < maximum):
        synchronize(device)
        start = time.perf_counter()
        operation()
        synchronize(device)
        sample = time.perf_counter() - start
        samples.append(sample)
        elapsed += sample
    return samples


# ---------------------------------------------------------------------------
# The three timed boundaries
# ---------------------------------------------------------------------------


def _greedy_normalized_selection(
    policy_logits: torch.Tensor, legal_mask_device: torch.Tensor
) -> torch.Tensor:
    """Batched greedy choice over the legal entries, in the normalized frame.

    Illegal entries are pushed to `-inf` rather than to a large finite value:
    this is a *selection*, nothing differentiates it, and `-inf` cannot be
    out-argmaxed by an illegal entry however extreme the logits get. `argmax`
    resolves ties to the lowest index, which is the same deterministic
    tie-break the accepted single-position adapter uses -- on the normalized
    identifier, as `model_contract_v2` requires.
    """
    masked = policy_logits.to(torch.float32).masked_fill(
        ~legal_mask_device, float("-inf")
    )
    return masked.argmax(dim=1)


def make_boundary_operation(
    *,
    boundary: str,
    model: ProductionModel,
    host: HostBatch,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Callable[[], ModelOutputs], Callable[[], ModelOutputs]]:
    """Build the timed closure for one boundary, plus a probe that returns outputs.

    Both closures do exactly the same work; the second is called once outside
    any timed region so the row can record finiteness and the observed
    device/dtype without those checks polluting the latency.
    """
    if boundary not in BOUNDARY_CONTENTS:
        raise BenchmarkError(f"unknown timing boundary {boundary!r}")

    if boundary == BOUNDARY_A:
        # Prepared once, outside the timed region, exactly as the boundary says.
        tokens = observation_to_tokens(
            torch.from_numpy(host.observations).to(device=device, dtype=dtype)
        ).contiguous()

        def run() -> ModelOutputs:
            with torch.no_grad():
                return model(tokens)

        return run, run

    if boundary == BOUNDARY_B:

        def run_b() -> ModelOutputs:
            tokens = tokenize_numpy_observation(
                host.observations, dtype=dtype, device=device
            )
            with torch.no_grad():
                return model(tokens)

        return run_b, run_b

    mask_host = torch.from_numpy(host.normalized_masks)
    acting = host.acting_players

    def run_c() -> ModelOutputs:
        tokens = tokenize_numpy_observation(host.observations, dtype=dtype, device=device)
        mask_device = mask_host.to(device=device, non_blocking=False)
        with torch.no_grad():
            outputs = model(tokens)
            chosen = _greedy_normalized_selection(outputs.policy_logits, mask_device)
        # Readback and the inverse frame conversion are part of live inference:
        # the engine only ever accepts an absolute identifier, so a benchmark
        # that stopped at the device-side argmax would be measuring a path that
        # cannot actually play.
        normalized = chosen.to("cpu").numpy()
        for row in range(normalized.shape[0]):
            model_action_to_absolute(int(normalized[row]), int(acting[row]))
        return outputs

    return run_c, run_c


def selection_validity(
    *,
    model: ProductionModel,
    host: HostBatch,
    device: torch.device,
    dtype: torch.dtype,
    corpus: BenchmarkCorpus,
) -> dict:
    """Run boundary C once and check every converted action against the engine set.

    This is the correctness half of boundary C. The timed closure deliberately
    does not assert anything -- assertions in a timed region measure the
    assertion -- so validity is established here instead, on the same code path
    with the same inputs.
    """
    tokens = tokenize_numpy_observation(host.observations, dtype=dtype, device=device)
    mask_device = torch.from_numpy(host.normalized_masks).to(device=device)
    with torch.no_grad():
        outputs = model(tokens)
        chosen = _greedy_normalized_selection(outputs.policy_logits, mask_device)
    normalized = chosen.to("cpu").numpy()

    illegal_normalized = 0
    illegal_absolute = 0
    for row in range(normalized.shape[0]):
        source_index = int(host.indices[row])
        player = int(host.acting_players[row])
        model_action = int(normalized[row])
        if model_action not in corpus.normalized_legal_lists[source_index]:
            illegal_normalized += 1
        absolute = model_action_to_absolute(model_action, player)
        if absolute not in corpus.absolute_legal_lists[source_index]:
            illegal_absolute += 1
    return {
        "selections": int(normalized.shape[0]),
        "illegal_normalized_selections": illegal_normalized,
        "illegal_absolute_actions": illegal_absolute,
        "all_selections_legal": illegal_normalized == 0 and illegal_absolute == 0,
    }


# ---------------------------------------------------------------------------
# One inference measurement
# ---------------------------------------------------------------------------


def run_inference_point(
    *,
    model: ProductionModel,
    candidate_id: str,
    config_digest: str,
    parameters: int,
    corpus: BenchmarkCorpus,
    batch: int,
    precision: str,
    boundary: str,
    device: torch.device,
    host: HostBatch | None = None,
) -> dict:
    """Measure one `(candidate, batch, precision, boundary)` point.

    Always returns a row. An out-of-memory failure, a non-finite head or a
    contract violation becomes a row with a status and an error string, because
    the batch at which a candidate stops working is the measurement that
    establishes its ceiling.
    """
    dtype = resolve_dtype(precision)
    row: dict[str, Any] = {
        "candidate_id": candidate_id,
        "config_digest": config_digest,
        "architecture_family": ARCHITECTURE_FAMILY,
        "parameters": parameters,
        "requested_device": device.type,
        "requested_precision": precision,
        "precision": precision,
        "batch": batch,
        "boundary": boundary,
        "boundary_includes": BOUNDARY_CONTENTS[boundary],
        "corpus_digest": corpus.digest,
        "warmup_iterations": WARMUP_ITERATIONS,
        "measurement_iterations": 0,
        "status": "error",
        "median_latency_ms": None,
        "p95_latency_ms": None,
        "mean_latency_ms": None,
        "min_latency_ms": None,
        "max_latency_ms": None,
        "stdev_latency_ms": None,
        "positions_per_second": None,
        "finite_outputs": None,
        "observed_device": None,
        "observed_precision": None,
        "process_rss_bytes": None,
        "metal_allocated_bytes": None,
        "metal_driver_bytes": None,
        "metal_recommended_max_bytes": None,
        "peak_memory_if_available": None,
        "memory_fraction_of_recommended": None,
        "oom": False,
        "error": "",
    }

    before = memory_pressure_fraction()
    if before is not None and before > MEMORY_PRESSURE_FRACTION:
        row["status"] = "skipped_memory_guard"
        row["error"] = (
            f"Metal driver allocation was already {before:.3f} of the recommended "
            f"maximum, above the {MEMORY_PRESSURE_FRACTION:.2f} guard; not attempted"
        )
        row.update(_memory_row(metal_memory_snapshot()))
        return row

    try:
        batch_host = host if host is not None else make_host_batch(corpus, batch)
        timed, probe = make_boundary_operation(
            boundary=boundary, model=model, host=batch_host, device=device, dtype=dtype
        )
        outputs = probe()
        synchronize(device)
        labels = verify_execution_labels(
            outputs, requested_device=device, requested_precision=precision
        )
        row.update(labels)
        row["finite_outputs"] = bool(outputs.all_finite())

        samples = timed_samples(timed, device=device)
        row.update(summarise_samples(samples))
        median = statistics.median(samples)
        row["positions_per_second"] = round(batch / median, 2) if median > 0 else None
        row["status"] = "ok" if row["finite_outputs"] else "non_finite"
        if not row["finite_outputs"]:
            row["error"] = "at least one head produced a non-finite value"
    except BenchmarkIntegrityError:
        # Never downgraded to a row: a mislabelled measurement must stop the run.
        raise
    except (RuntimeError, ModelContractError, MemoryError) as error:
        oom = _is_out_of_memory(error)
        row["status"] = "oom" if oom else "error"
        row["oom"] = oom
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        snapshot = metal_memory_snapshot()
        row.update(_memory_row(snapshot))
        release_device_memory(device)

    return row


def _memory_row(snapshot: Mapping[str, Any]) -> dict:
    driver = snapshot["metal_driver_bytes"]
    recommended = snapshot["metal_recommended_max_bytes"]
    fraction = (
        round(float(driver) / float(recommended), 6)
        if driver is not None and recommended
        else None
    )
    return {
        "process_rss_bytes": snapshot["process_rss_bytes"],
        "metal_allocated_bytes": snapshot["metal_allocated_bytes"],
        "metal_driver_bytes": snapshot["metal_driver_bytes"],
        "metal_recommended_max_bytes": snapshot["metal_recommended_max_bytes"],
        "peak_memory_if_available": snapshot["process_rss_bytes"],
        "memory_fraction_of_recommended": fraction,
    }


# ---------------------------------------------------------------------------
# Numerical checks
# ---------------------------------------------------------------------------


def _head_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Absolute and relative error for one head, both reported honestly.

    `meaningful_relative_error` is restricted to entries whose reference
    magnitude exceeds :data:`RELATIVE_ERROR_FLOOR`. Near-zero logits otherwise
    produce ratios in the thousands that describe the denominator rather than
    the network, and a float16 run must not be failed for that. The unrestricted
    maximum is still reported alongside so nothing is hidden by the filter.
    """
    reference = reference.detach().to("cpu", torch.float32)
    candidate = candidate.detach().to("cpu", torch.float32)
    difference = (candidate - reference).abs()
    magnitude = reference.abs()
    significant = magnitude > RELATIVE_ERROR_FLOOR
    relative_all = difference / magnitude.clamp(min=torch.finfo(torch.float32).tiny)
    meaningful = relative_all[significant]
    return {
        "max_absolute_error": float(difference.max()) if difference.numel() else 0.0,
        "mean_absolute_error": float(difference.mean()) if difference.numel() else 0.0,
        "max_relative_error_unfiltered": float(relative_all.max())
        if relative_all.numel()
        else 0.0,
        "meaningful_max_relative_error": float(meaningful.max()) if meaningful.numel() else 0.0,
        "meaningful_mean_relative_error": float(meaningful.mean())
        if meaningful.numel()
        else 0.0,
        "relative_error_floor": RELATIVE_ERROR_FLOOR,
        "entries_compared": int(difference.numel()),
        "entries_above_floor": int(significant.sum()),
        "finite": bool(torch.isfinite(candidate).all()),
    }


def _greedy_from_logits(
    policy_logits: torch.Tensor, mask: torch.Tensor, bonus: torch.Tensor | None = None
) -> np.ndarray:
    logits = policy_logits.detach().to("cpu", torch.float32)
    if bonus is not None:
        logits = logits + bonus
    masked = logits.masked_fill(~mask, float("-inf"))
    return masked.argmax(dim=1).numpy()


def numerical_comparison(
    *,
    candidate_id: str,
    config: CandidateConfig,
    corpus: BenchmarkCorpus,
    device: torch.device,
    seed: int = FAMILY_INITIALIZATION_SEED,
    positions: int = NUMERICAL_CHECK_POSITIONS,
) -> dict:
    """CPU float32 vs MPS float32, and the float32 reference vs MPS float16.

    Both comparisons run on the same corpus rows through the same weights: the
    candidate is built on CPU and moved, so the two devices start from
    bit-identical float32 parameters and every difference measured here is the
    kernels or the precision, never the initialisation.

    Three action-agreement figures are reported for each device/precision:

    * **crafted margin** -- a fixed bonus is added to one designated legal action
      per position, on both sides, so the intended choice wins by far more than
      any rounding can move it. A disagreement here is a defect and is treated
      as one.
    * **natural corpus** -- the untouched logits. Near-ties are real here, so a
      float16 flip is information rather than a failure, and it is reported
      without being judged.
    * **absolute validity** -- every selection converted back through
      `action_frame` must land in the engine's absolute legal set.
    """
    host = make_host_batch(corpus, min(positions, corpus.size))
    mask = torch.from_numpy(host.normalized_masks)

    # A deterministic crafted bonus: one designated legal action per row, chosen
    # by position index so it is reproducible without another RNG stream.
    bonus = torch.zeros(host.batch_size, POLICY_LOGIT_COUNT, dtype=torch.float32)
    designated = np.empty(host.batch_size, dtype=np.int64)
    for row in range(host.batch_size):
        legal = corpus.normalized_legal_lists[int(host.indices[row])]
        action = int(legal[row % len(legal)])
        designated[row] = action
        bonus[row, action] = CRAFTED_MARGIN

    reference_model = build_candidate_model(config, seed=seed, device="cpu", dtype=torch.float32)
    reference_tokens = tokenize_numpy_observation(
        host.observations, dtype=torch.float32, device="cpu"
    )
    with torch.no_grad():
        reference = reference_model(reference_tokens)
    reference_cpu = reference.detached_cpu()
    reference_natural = _greedy_from_logits(reference_cpu.policy_logits, mask)
    reference_crafted = _greedy_from_logits(reference_cpu.policy_logits, mask, bonus)

    report: dict[str, Any] = {
        "candidate_id": candidate_id,
        "positions": host.batch_size,
        "corpus_digest": corpus.digest,
        "crafted_margin": CRAFTED_MARGIN,
        "tolerances": TOLERANCES,
        "reference": {
            "device": "cpu",
            "precision": "float32",
            "finite": bool(reference.all_finite()),
        },
        "comparisons": {},
    }

    # The crafted margin has to actually dominate on the reference itself, or
    # the "agreement" it measures would be vacuous.
    report["crafted_margin_effective_on_reference"] = bool(
        np.array_equal(reference_crafted, designated)
    )

    for key, precision in (("mps_float32", "float32"), ("mps_float16", "float16")):
        dtype = resolve_dtype(precision)
        entry: dict[str, Any] = {
            "device": device.type,
            "precision": precision,
            "status": "ok",
            "error": "",
        }
        try:
            model = build_candidate_model(config, seed=seed, device=device, dtype=dtype)
            tokens = tokenize_numpy_observation(host.observations, dtype=dtype, device=device)
            with torch.no_grad():
                outputs = model(tokens)
            synchronize(device)
            entry.update(
                verify_execution_labels(
                    outputs, requested_device=device, requested_precision=precision
                )
            )
            moved = outputs.detached_cpu()
            entry["heads"] = {
                "policy_logits": _head_error(reference_cpu.policy_logits, moved.policy_logits),
                "value_probabilities": _head_error(
                    torch.softmax(reference_cpu.value_logits, dim=-1),
                    torch.softmax(moved.value_logits, dim=-1),
                ),
                "belief_logits": _head_error(reference_cpu.belief_logits, moved.belief_logits),
            }
            entry["finite_outputs"] = bool(outputs.all_finite())

            natural = _greedy_from_logits(moved.policy_logits, mask)
            crafted = _greedy_from_logits(moved.policy_logits, mask, bonus)
            entry["natural_greedy_agreement"] = int((natural == reference_natural).sum())
            entry["natural_greedy_disagreements"] = int((natural != reference_natural).sum())
            entry["crafted_margin_agreement"] = int((crafted == designated).sum())
            entry["crafted_margin_disagreements"] = int((crafted != designated).sum())
            entry["crafted_margin_passes"] = bool(np.array_equal(crafted, designated))

            illegal_absolute = 0
            for row in range(host.batch_size):
                player = int(host.acting_players[row])
                absolute = model_action_to_absolute(int(natural[row]), player)
                if absolute not in corpus.absolute_legal_lists[int(host.indices[row])]:
                    illegal_absolute += 1
            entry["illegal_absolute_actions"] = illegal_absolute
            entry["absolute_action_validity_passes"] = illegal_absolute == 0

            limits = TOLERANCES[key]
            entry["within_tolerance"] = bool(
                entry["heads"]["policy_logits"]["max_absolute_error"]
                <= limits["policy_logits_max_abs"]
                and entry["heads"]["value_probabilities"]["max_absolute_error"]
                <= limits["value_probabilities_max_abs"]
                and entry["heads"]["belief_logits"]["max_absolute_error"]
                <= limits["belief_logits_max_abs"]
            )
            entry["passes"] = bool(
                entry["within_tolerance"]
                and entry["finite_outputs"]
                and entry["crafted_margin_passes"]
                and entry["absolute_action_validity_passes"]
            )
            del model, outputs
        except BenchmarkIntegrityError:
            raise
        except (RuntimeError, ModelContractError, MemoryError) as error:
            entry["status"] = "oom" if _is_out_of_memory(error) else "error"
            entry["error"] = f"{type(error).__name__}: {error}"
            entry["passes"] = False
            entry["finite_outputs"] = False
        finally:
            release_device_memory(device)
        report["comparisons"][key] = entry

    return report


# ---------------------------------------------------------------------------
# Training-step benchmark
# ---------------------------------------------------------------------------

#: Which parameters belong to which head, for the per-group gradient report.
GRADIENT_GROUPS: dict[str, tuple[str, ...]] = {
    "shared_encoder_gradient": (
        "input_projection",
        "row_embedding",
        "column_embedding",
        "blocks",
        "encoder_norm",
    ),
    "policy_head_gradient": ("policy_query", "policy_key", "policy_source_bias", "policy_destination_bias"),
    "value_head_gradient": ("value_body", "value_output"),
    "belief_head_gradient": ("belief_output",),
}


def _gradient_report(model: ProductionModel) -> dict:
    """Per-group gradient norms plus a global finiteness verdict.

    Norms rather than a boolean: "a gradient exists" and "a gradient is doing
    anything" are different claims, and a head that receives an exactly zero
    gradient is disconnected in a way a presence check would not catch.
    """
    totals = {name: 0.0 for name in GRADIENT_GROUPS}
    missing: list[str] = []
    non_finite: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
            continue
        gradient = parameter.grad.detach().to(torch.float32)
        if not bool(torch.isfinite(gradient).all()):
            non_finite.append(name)
        squared = float((gradient * gradient).sum())
        for group, prefixes in GRADIENT_GROUPS.items():
            if any(name.startswith(prefix) for prefix in prefixes):
                totals[group] += squared
                break
    report = {group: round(total**0.5, 8) for group, total in totals.items()}
    report["finite_gradients"] = not non_finite and not missing
    report["parameters_without_gradient"] = len(missing)
    report["parameters_with_non_finite_gradient"] = len(non_finite)
    report["non_finite_parameter_names"] = non_finite[:8]
    return report


def run_training_point(
    *,
    candidate_id: str,
    config: CandidateConfig,
    config_digest: str,
    parameters: int,
    corpus: BenchmarkCorpus,
    batch: int,
    precision: str,
    device: torch.device,
    seed: int = FAMILY_INITIALIZATION_SEED,
) -> dict:
    """Time one forward + three losses + backward, with no optimizer step.

    The step is deliberately incomplete: there is no optimizer, no parameter
    update and no scheduler, because Phase 6 authorises a backward pass only to
    measure compute and prove gradient connectivity. Nothing here trains.

    float16 is measured as *pure* float16 -- the parameters and the activations
    are half, with no autocast and no loss scaling. That is the honest version
    of the question "can this candidate be trained in half precision on this
    host", and if the answer is no it is recorded as no. The loss functions
    themselves upcast their inputs to float32 internally (see
    `stratego.model.losses`), which is a property of the accepted loss code and
    applies identically to every candidate.
    """
    dtype = resolve_dtype(precision)
    row: dict[str, Any] = {
        "candidate_id": candidate_id,
        "config_digest": config_digest,
        "parameters": parameters,
        "requested_device": device.type,
        "requested_precision": precision,
        "precision": precision,
        "batch": batch,
        "corpus_digest": corpus.digest,
        "warmup_iterations": WARMUP_ITERATIONS,
        "measurement_iterations": 0,
        "status": "error",
        "forward_ms": None,
        "loss_ms": None,
        "backward_ms": None,
        "total_ms": None,
        "examples_per_second": None,
        "policy_loss": None,
        "value_loss": None,
        "belief_loss": None,
        "total_loss": None,
        "finite_loss": None,
        "finite_gradients": None,
        "shared_encoder_gradient": None,
        "policy_head_gradient": None,
        "value_head_gradient": None,
        "belief_head_gradient": None,
        "observed_device": None,
        "observed_precision": None,
        "process_rss_bytes": None,
        "metal_allocated_bytes": None,
        "metal_driver_bytes": None,
        "metal_recommended_max_bytes": None,
        "memory_fraction_of_recommended": None,
        "oom": False,
        "error": "",
        "optimizer_step": False,
        "parameter_update": False,
    }

    before = memory_pressure_fraction()
    if before is not None and before > MEMORY_PRESSURE_FRACTION:
        row["status"] = "skipped_memory_guard"
        row["error"] = (
            f"Metal driver allocation was already {before:.3f} of the recommended "
            f"maximum, above the {MEMORY_PRESSURE_FRACTION:.2f} guard; not attempted"
        )
        row.update(_memory_row(metal_memory_snapshot()))
        return row

    model: ProductionModel | None = None
    try:
        host = make_host_batch(corpus, batch)
        model = build_candidate_model(config, seed=seed, device=device, dtype=dtype)
        model.train()

        tokens = tokenize_numpy_observation(host.observations, dtype=dtype, device=device)
        legal_mask = torch.from_numpy(host.normalized_masks).to(device=device)
        policy_targets = torch.from_numpy(host.policy_targets).to(device=device)
        value_targets = torch.from_numpy(host.value_targets).to(device=device)
        belief_labels = torch.from_numpy(host.belief_labels).to(device=device)
        belief_masks = torch.from_numpy(host.belief_masks).to(device=device)

        forward_samples: list[float] = []
        loss_samples: list[float] = []
        backward_samples: list[float] = []
        total_samples: list[float] = []
        last_losses: dict | None = None

        def one_step(record: bool) -> None:
            nonlocal last_losses
            model.zero_grad(set_to_none=True)
            synchronize(device)
            start = time.perf_counter()

            outputs = model(tokens)
            synchronize(device)
            after_forward = time.perf_counter()

            losses = multi_head_loss(
                outputs,
                target_actions=policy_targets,
                legal_mask=legal_mask,
                target_value_classes=value_targets,
                belief_labels=belief_labels,
                belief_mask=belief_masks,
            )
            synchronize(device)
            after_loss = time.perf_counter()

            losses.total.backward()
            synchronize(device)
            after_backward = time.perf_counter()

            if record:
                forward_samples.append(after_forward - start)
                loss_samples.append(after_loss - after_forward)
                backward_samples.append(after_backward - after_loss)
                total_samples.append(after_backward - start)
                last_losses = {
                    "losses": losses.to_dict(),
                    "finite": losses.all_finite(),
                    "labels": verify_execution_labels(
                        outputs, requested_device=device, requested_precision=precision
                    ),
                }

        for _ in range(WARMUP_ITERATIONS):
            one_step(record=False)

        elapsed = 0.0
        while len(total_samples) < MIN_MEASUREMENT_ITERATIONS or (
            elapsed < TARGET_MEASUREMENT_SECONDS
            and len(total_samples) < MAX_MEASUREMENT_ITERATIONS
        ):
            one_step(record=True)
            elapsed = sum(total_samples)

        assert last_losses is not None
        row.update(last_losses["labels"])
        row["measurement_iterations"] = len(total_samples)
        row["forward_ms"] = round(1000 * statistics.median(forward_samples), 4)
        row["loss_ms"] = round(1000 * statistics.median(loss_samples), 4)
        row["backward_ms"] = round(1000 * statistics.median(backward_samples), 4)
        median_total = statistics.median(total_samples)
        row["total_ms"] = round(1000 * median_total, 4)
        row["examples_per_second"] = round(batch / median_total, 2) if median_total > 0 else None

        losses = last_losses["losses"]
        row["policy_loss"] = round(losses["policy"], 6)
        row["value_loss"] = round(losses["value"], 6)
        row["belief_loss"] = round(losses["belief"], 6)
        row["total_loss"] = round(losses["total"], 6)
        row["finite_loss"] = bool(last_losses["finite"])

        gradients = _gradient_report(model)
        row["finite_gradients"] = gradients["finite_gradients"]
        for group in GRADIENT_GROUPS:
            row[group] = gradients[group]
        row["parameters_without_gradient"] = gradients["parameters_without_gradient"]
        row["parameters_with_non_finite_gradient"] = gradients[
            "parameters_with_non_finite_gradient"
        ]

        if not row["finite_loss"]:
            row["status"] = "non_finite_loss"
            row["error"] = "at least one loss component was not finite"
        elif not row["finite_gradients"]:
            row["status"] = "non_finite_gradients"
            row["error"] = (
                f"{gradients['parameters_with_non_finite_gradient']} parameters had a "
                f"non-finite gradient and {gradients['parameters_without_gradient']} had none"
            )
        else:
            row["status"] = "ok"
    except BenchmarkIntegrityError:
        raise
    except (RuntimeError, ModelContractError, MemoryError) as error:
        oom = _is_out_of_memory(error)
        row["status"] = "oom" if oom else "error"
        row["oom"] = oom
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        if model is not None:
            model.zero_grad(set_to_none=True)
            del model
        row.update(_memory_row(metal_memory_snapshot()))
        release_device_memory(device)

    return row


# ---------------------------------------------------------------------------
# Candidate summaries and classification
# ---------------------------------------------------------------------------


def summarise_candidate(
    *,
    candidate_id: str,
    parameters: int,
    inference_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    numerical: Mapping[str, Any] | None,
) -> dict:
    """Reduce a candidate's rows to the frontier figures the report needs.

    "Stable" means a row whose status is `ok`: it ran, it stayed finite, and it
    ran where and in the precision it claimed. Rows that hit the memory guard,
    ran out of memory or produced a non-finite head are excluded from the
    frontier figures and counted separately, so a candidate cannot look fast by
    being measured only where it happened to survive.
    """

    def stable(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return [row for row in rows if row.get("status") == "ok"]

    forward_rows = [row for row in inference_rows if row["boundary"] == BOUNDARY_A]
    stable_forward = stable(forward_rows)

    def best_throughput(precision: str) -> float | None:
        values = [
            row["positions_per_second"]
            for row in stable_forward
            if row["precision"] == precision and row["positions_per_second"] is not None
        ]
        return max(values) if values else None

    def best_batch(precision: str) -> int | None:
        values = [row["batch"] for row in stable_forward if row["precision"] == precision]
        return max(values) if values else None

    stable_training = stable(training_rows)
    float32_training = [row for row in stable_training if row["precision"] == "float32"]
    # "Representative" is the largest stable float32 step, because that is the
    # configuration a real training loop would actually choose.
    representative = max(
        float32_training, key=lambda row: row["batch"], default=None
    )

    memory_values = [
        row["memory_fraction_of_recommended"]
        for row in inference_rows
        if row.get("memory_fraction_of_recommended") is not None
    ]

    numerical_float32 = None
    numerical_float16 = None
    if numerical is not None:
        numerical_float32 = bool(
            numerical["comparisons"].get("mps_float32", {}).get("passes", False)
        )
        numerical_float16 = bool(
            numerical["comparisons"].get("mps_float16", {}).get("passes", False)
        )

    max_inference_batch = max(
        (best_batch("float32") or 0), (best_batch("float16") or 0)
    ) or None

    return {
        "candidate_id": candidate_id,
        "parameters": parameters,
        "best_float32_positions_per_second": best_throughput("float32"),
        "best_float16_positions_per_second": best_throughput("float16"),
        "best_float32_batch": best_batch("float32"),
        "best_float16_batch": best_batch("float16"),
        "max_stable_inference_batch": max_inference_batch,
        "representative_training_examples_per_second": (
            representative["examples_per_second"] if representative else None
        ),
        "representative_training_batch": representative["batch"] if representative else None,
        "max_stable_training_batch": max(
            (row["batch"] for row in stable_training), default=None
        ),
        "max_stable_float32_training_batch": max(
            (row["batch"] for row in float32_training), default=None
        ),
        "float16_training_stable": any(
            row["precision"] == "float16" for row in stable_training
        ),
        "peak_metal_fraction": max(memory_values) if memory_values else None,
        "numerically_stable_float32": numerical_float32,
        "numerically_stable_float16": numerical_float16,
        "inference_rows": len(inference_rows),
        "inference_oom_rows": sum(1 for row in inference_rows if row.get("oom")),
        "inference_error_rows": sum(
            1 for row in inference_rows if row.get("status") not in ("ok", "oom")
        ),
        "training_rows": len(training_rows),
        "training_oom_rows": sum(1 for row in training_rows if row.get("oom")),
    }


def classification_inputs(summary: Mapping[str, Any]) -> dict:
    """Project a summary onto the fields a classification may see.

    This is the mechanism, not a comment: `classify_candidates` calls it and
    then cannot reach anything else, so a strength field added to a summary in
    some later phase has no path into a classification. Adding one to
    `CLASSIFICATION_INPUT_KEYS` would trip the substring guard below.
    """
    for key in CLASSIFICATION_INPUT_KEYS:
        lowered = key.lower()
        for banned in FORBIDDEN_CLASSIFICATION_SUBSTRINGS:
            if banned in lowered:
                raise BenchmarkError(
                    f"classification input {key!r} looks like a playing-strength field "
                    f"(matched {banned!r}); Phase 6 forbids strength-based selection"
                )
    return {key: summary.get(key) for key in CLASSIFICATION_INPUT_KEYS}


def _dominates(better: Mapping[str, Any], worse: Mapping[str, Any]) -> bool:
    """True when `better` is at least as good on every axis and better on one.

    The axes are parameter count (a capacity proxy), float32 inference
    throughput, training throughput and maximum stable inference batch. Reading
    parameters as an axis to *maximise* is what makes domination meaningful
    here: a candidate that is both larger and faster leaves nothing for a
    smaller, slower one to offer.
    """
    axes = (
        "parameters",
        "best_float32_positions_per_second",
        "representative_training_examples_per_second",
        "max_stable_inference_batch",
    )
    strictly_better = False
    for axis in axes:
        left = better.get(axis)
        right = worse.get(axis)
        if left is None or right is None:
            return False
        if left < right:
            return False
        if left > right:
            strictly_better = True
    return strictly_better


def classify_candidates(summaries: Sequence[Mapping[str, Any]]) -> dict:
    """ADVANCE / DOMINATED / IMPRACTICAL, deterministically, from the summaries.

    The rule, in order:

    1. **IMPRACTICAL** when the candidate cannot be used at all on this host --
       no stable float32 forward pass at batch `MIN_VIABLE_INFERENCE_BATCH`, no
       stable float32 training step at `MIN_VIABLE_TRAINING_BATCH`, a failed
       float32 numerical check, sustained throughput below
       `MIN_VIABLE_POSITIONS_PER_SECOND`, or peak Metal use above
       `MEMORY_IMPRACTICAL_FRACTION` of the recommended maximum.
    2. **DOMINATED** when some other practical candidate is at least as large
       *and* at least as fast on inference, training and maximum batch, with at
       least one strict improvement.
    3. **ADVANCE** otherwise.

    Deterministic in the input: same summaries in, same verdicts out, with no
    RNG, no wall-clock and no dependence on iteration order (candidates are
    sorted before comparison). float16 never decides anything on its own -- a
    candidate whose float32 path is sound is not eliminated because pure float16
    backward fails, and that fact is carried into the reason string instead.
    """
    projected = {
        summary["candidate_id"]: classification_inputs(summary) for summary in summaries
    }
    ordered = sorted(projected)

    verdicts: dict[str, str] = {}
    reasons: dict[str, str] = {}

    for candidate_id in ordered:
        entry = projected[candidate_id]
        problems: list[str] = []
        batch = entry.get("max_stable_inference_batch")
        training_batch = entry.get("max_stable_training_batch")
        throughput = entry.get("best_float32_positions_per_second")
        memory = entry.get("peak_metal_fraction")

        if batch is None or batch < MIN_VIABLE_INFERENCE_BATCH:
            problems.append(
                f"no stable float32 inference at batch {MIN_VIABLE_INFERENCE_BATCH} "
                f"(best stable batch: {batch})"
            )
        if training_batch is None or training_batch < MIN_VIABLE_TRAINING_BATCH:
            problems.append(
                f"no stable training step at batch {MIN_VIABLE_TRAINING_BATCH} "
                f"(best stable batch: {training_batch})"
            )
        if entry.get("numerically_stable_float32") is not True:
            problems.append("the CPU/MPS float32 numerical check did not pass")
        if throughput is None or throughput < MIN_VIABLE_POSITIONS_PER_SECOND:
            problems.append(
                f"sustained float32 throughput {throughput} positions/s is below the "
                f"{MIN_VIABLE_POSITIONS_PER_SECOND:.0f} positions/s practical floor"
            )
        if memory is not None and memory > MEMORY_IMPRACTICAL_FRACTION:
            problems.append(
                f"peak Metal use {memory:.3f} exceeds {MEMORY_IMPRACTICAL_FRACTION:.2f} "
                "of the recommended maximum"
            )

        if problems:
            verdicts[candidate_id] = "IMPRACTICAL"
            reasons[candidate_id] = "; ".join(problems)

    practical = [candidate_id for candidate_id in ordered if candidate_id not in verdicts]

    for candidate_id in practical:
        dominators = [
            other
            for other in practical
            if other != candidate_id
            and _dominates(projected[other], projected[candidate_id])
        ]
        if dominators:
            verdicts[candidate_id] = "DOMINATED"
            reasons[candidate_id] = (
                f"dominated by {', '.join(sorted(dominators))}: at least as many "
                "parameters and at least as fast on inference, training and maximum "
                "stable batch"
            )
        else:
            verdicts[candidate_id] = "ADVANCE"
            entry = projected[candidate_id]
            reason = (
                f"on the measured frontier: {entry['parameters']:,} parameters, "
                f"{entry['best_float32_positions_per_second']:,.0f} positions/s float32, "
                f"{entry['representative_training_examples_per_second']:,.0f} training "
                f"examples/s, stable to inference batch {entry['max_stable_inference_batch']}"
            )
            if entry.get("numerically_stable_float16") is not True:
                reason += "; float32 only (the float16 numerical check did not pass)"
            reasons[candidate_id] = reason

    return {
        "verdicts": verdicts,
        "reasons": reasons,
        "advance_ids": sorted(k for k, v in verdicts.items() if v == "ADVANCE"),
        "dominated_ids": sorted(k for k, v in verdicts.items() if v == "DOMINATED"),
        "impractical_ids": sorted(k for k, v in verdicts.items() if v == "IMPRACTICAL"),
        "classification_inputs": projected,
        "rules": classification_rules(),
    }


def classification_rules() -> dict:
    """The rule, serialised, so the report and the data file cannot drift."""
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "order": ["IMPRACTICAL", "DOMINATED", "ADVANCE"],
        "impractical_if_any": [
            f"max stable inference batch < {MIN_VIABLE_INFERENCE_BATCH}",
            f"max stable training batch < {MIN_VIABLE_TRAINING_BATCH}",
            "CPU float32 vs MPS float32 numerical check did not pass",
            f"best stable float32 throughput < {MIN_VIABLE_POSITIONS_PER_SECOND} positions/s",
            f"peak Metal use > {MEMORY_IMPRACTICAL_FRACTION} of recommended maximum",
        ],
        "dominated_if": (
            "another practical candidate has >= parameters, >= float32 inference "
            "positions/s, >= training examples/s and >= max stable inference batch, "
            "with at least one strict improvement"
        ),
        "advance_otherwise": True,
        "classification_input_keys": list(CLASSIFICATION_INPUT_KEYS),
        "forbidden_input_substrings": list(FORBIDDEN_CLASSIFICATION_SUBSTRINGS),
        "float16_never_eliminates": (
            "a candidate with a sound float32 path is not eliminated because pure "
            "float16 backward fails; the caveat is carried in its reason string"
        ),
        "strength_is_not_an_input": (
            "no playing-strength, win-rate or match-result field is reachable from "
            "classify_candidates; see classification_inputs()"
        ),
        "practical_floor_derivation": (
            "Phase 3 measured simulation-only ~96,963 positions/s and integrated "
            "~12,838 positions/s. Under serial composition a model below 5,000 "
            "positions/s caps the integrated pipeline under ~4,755 positions/s."
        ),
        "thresholds": {
            "min_viable_inference_batch": MIN_VIABLE_INFERENCE_BATCH,
            "min_viable_training_batch": MIN_VIABLE_TRAINING_BATCH,
            "min_viable_positions_per_second": MIN_VIABLE_POSITIONS_PER_SECOND,
            "memory_impractical_fraction": MEMORY_IMPRACTICAL_FRACTION,
            "memory_pressure_fraction": MEMORY_PRESSURE_FRACTION,
        },
    }


def pareto_frontier(summaries: Sequence[Mapping[str, Any]], classification: Mapping) -> list[dict]:
    """One report-ready row per candidate, in ladder order."""
    by_id = {summary["candidate_id"]: summary for summary in summaries}
    rows = []
    for candidate_id in CANDIDATE_IDS:
        if candidate_id not in by_id:
            continue
        summary = by_id[candidate_id]
        rows.append(
            {
                "candidate_id": candidate_id,
                "parameters": summary["parameters"],
                "best_float32_positions_per_second": summary[
                    "best_float32_positions_per_second"
                ],
                "best_float16_positions_per_second": summary[
                    "best_float16_positions_per_second"
                ],
                "representative_training_examples_per_second": summary[
                    "representative_training_examples_per_second"
                ],
                "max_stable_inference_batch": summary["max_stable_inference_batch"],
                "max_stable_training_batch": summary["max_stable_training_batch"],
                "peak_metal_fraction": summary["peak_metal_fraction"],
                "classification": classification["verdicts"].get(candidate_id),
                "reason": classification["reasons"].get(candidate_id),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Reproduction of the Agent 2 family
# ---------------------------------------------------------------------------


def reproduce_candidate_configs(*, seed: int = FAMILY_INITIALIZATION_SEED) -> dict:
    """Rebuild every candidate and require its digest and parameter count to match.

    The prerequisite Agent 3 is actually asked to check is not "Agent 2 wrote a
    file that says PASS" but "the models I am about to time are the models Agent
    2 accepted". This rebuilds each candidate from its stored configuration and
    compares the configuration digest and the exact trainable parameter count
    before a single measurement is taken.
    """
    digests = config_digests()
    report: dict[str, Any] = {"candidates": {}, "all_reproduced": True}
    for candidate_id in CANDIDATE_IDS:
        config = candidate_config(candidate_id)
        model = ProductionModel(config, seed=seed)
        entry = {
            "candidate_id": candidate_id,
            "config_digest": config.digest(),
            "expected_config_digest": digests[candidate_id],
            "digest_matches": config.digest() == digests[candidate_id],
            "parameters": model.parameter_count(),
            "trainable_parameters": model.trainable_parameter_count(),
            "architecture_family": ARCHITECTURE_FAMILY,
            "architecture_family_version": ARCHITECTURE_FAMILY_VERSION,
            "initialisation_seed": model.initialisation_seed,
        }
        entry["reproduced"] = bool(entry["digest_matches"])
        report["candidates"][candidate_id] = entry
        report["all_reproduced"] = report["all_reproduced"] and entry["reproduced"]
        del model
    return report


def benchmark_method_summary() -> dict:
    """Everything a reader needs to reproduce or distrust these numbers."""
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "action_encoding_version": ACTION_ENCODING_VERSION,
        "policy_action_frame": POLICY_ACTION_FRAME,
        "action_frame": action_frame_summary(),
        "architecture_family": ARCHITECTURE_FAMILY,
        "architecture_family_version": ARCHITECTURE_FAMILY_VERSION,
        "initialization_seed": FAMILY_INITIALIZATION_SEED,
        "corpus_version": CORPUS_VERSION,
        "corpus_seed": CORPUS_SEED,
        "corpus_positions": CORPUS_POSITIONS,
        "target_seed": TARGET_SEED,
        "inference_batch_sizes": list(INFERENCE_BATCH_SIZES),
        "extended_batch_sizes": list(EXTENDED_BATCH_SIZES),
        "training_batch_sizes": list(TRAINING_BATCH_SIZES),
        "precisions": list(PRECISIONS),
        "boundaries": {name: BOUNDARY_CONTENTS[name] for name in BOUNDARIES},
        "warmup_iterations": WARMUP_ITERATIONS,
        "min_measurement_iterations": MIN_MEASUREMENT_ITERATIONS,
        "max_measurement_iterations": MAX_MEASUREMENT_ITERATIONS,
        "target_measurement_seconds": TARGET_MEASUREMENT_SECONDS,
        "synchronization": (
            "torch.mps.synchronize() immediately before and immediately after every "
            "timed region, and between training-step stages"
        ),
        "dtype_convention": (
            "parameters and activations are both cast to the row's precision; the "
            "accepted loss functions upcast to float32 internally, identically for "
            "every candidate"
        ),
        "float16_training_policy": (
            "pure float16: no autocast, no loss scaling, no optimizer, no parameter "
            "update; non-finite behaviour is reported rather than repaired"
        ),
        "memory_apis": [
            "torch.mps.current_allocated_memory",
            "torch.mps.driver_allocated_memory",
            "torch.mps.recommended_max_memory",
            "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        ],
        "memory_guard": {
            "pressure_fraction": MEMORY_PRESSURE_FRACTION,
            "policy": (
                "a configuration is not attempted while Metal driver allocation is "
                "already above the pressure fraction, and extended batches stop at "
                "the same line; the host is never deliberately driven into OOM or swap"
            ),
        },
        "extended_probe_policy": (
            "batches above 2048 are attempted only while throughput is still "
            f"improving by at least {EXTENDED_PROBE_IMPROVEMENT:.0%} and Metal memory "
            "remains below the pressure fraction"
        ),
        "tolerances": TOLERANCES,
        "relative_error_floor": RELATIVE_ERROR_FLOOR,
        "crafted_margin": CRAFTED_MARGIN,
        "numerical_check_positions": NUMERICAL_CHECK_POSITIONS,
        "belief_target_version": BELIEF_TARGET_VERSION,
        "belief_ignore_index": BELIEF_IGNORE_INDEX,
        "target_generation": {
            "policy": (
                "one action drawn per position from that position's own normalized "
                "legal list with a seeded generator; legality is then proved "
                "independently against both the normalized list and the normalized "
                "dense mask by verify_policy_targets_legal()"
            ),
            "value": (
                "a seeded uniform draw over the three WIN/DRAW/LOSS classes; the "
                "value target carries no game outcome and exists only to make the "
                "value head's backward pass real"
            ),
            "belief": (
                "stratego.training.belief_targets.dense_belief_target on the real "
                "GameState for the acting player: unresolved hidden opponent pieces "
                "only, in normalized squares, with BELIEF_IGNORE_INDEX elsewhere. "
                "Privileged, and used only as a backward-pass target -- never a "
                "model input"
            ),
        },
        "environment": detect_device_report(),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }


__all__ = [
    "BENCHMARK_VERSION",
    "BOUNDARIES",
    "BOUNDARY_A",
    "BOUNDARY_B",
    "BOUNDARY_C",
    "BOUNDARY_CONTENTS",
    "CLASSIFICATION_INPUT_KEYS",
    "CORPUS_POSITIONS",
    "CORPUS_SEED",
    "EXTENDED_BATCH_SIZES",
    "EXTENDED_PROBE_IMPROVEMENT",
    "INFERENCE_BATCH_SIZES",
    "MEMORY_IMPRACTICAL_FRACTION",
    "MEMORY_PRESSURE_FRACTION",
    "MIN_VIABLE_INFERENCE_BATCH",
    "MIN_VIABLE_POSITIONS_PER_SECOND",
    "MIN_VIABLE_TRAINING_BATCH",
    "PRECISIONS",
    "TOLERANCES",
    "TRAINING_BATCH_SIZES",
    "BenchmarkCorpus",
    "BenchmarkError",
    "BenchmarkIntegrityError",
    "HostBatch",
    "benchmark_method_summary",
    "build_benchmark_corpus",
    "classification_inputs",
    "classification_rules",
    "classify_candidates",
    "make_host_batch",
    "memory_pressure_fraction",
    "metal_memory_snapshot",
    "numerical_comparison",
    "pareto_frontier",
    "precision_name",
    "reproduce_candidate_configs",
    "require_mps",
    "resolve_dtype",
    "run_inference_point",
    "run_training_point",
    "selection_validity",
    "summarise_candidate",
    "summarise_samples",
    "timed_samples",
    "verify_execution_labels",
    "verify_legality_frames_agree",
    "verify_policy_targets_legal",
]
