"""Checkpoint-aware parallel neural evaluation: one MPS owner, many CPU games.

Specification sources:

- Phase 6 Agent 5 instructions ("MPS ownership requirement", "Observer-safe
  payload", "Deterministic batching", "Greedy reproducibility sweep")
- `stratego/evaluation/policy.py` -- `policy_interface_v1`, unchanged
- `stratego/evaluation/match_runner.py` -- `play_match`, unchanged
- `stratego/model/policy_adapter.py` -- the decision rules this module reuses

The limitation this removes
---------------------------
Phase 5 ran its neural gauntlet at `worker_count=1`. Not because a game is
serial, but because :func:`stratego.evaluation.match_runner.run_schedule`
rebuilds policies from the Phase 4 catalogue inside each worker process, and the
neural policy is deliberately not in that catalogue. Registering it there would
change what every audit that enumerates "all policies" means, and handing each
worker its own checkpoint would put one Metal context per process on a machine
with one GPU.

The topology here does neither::

    N CPU game workers            pure engine/NumPy processes, no torch import
      -> observer-safe request    identity, seed, observation, legal actions
      -> 1 long-lived owner       the only process holding Metal
         -> checkpoint loaded once
         -> forward pass
         -> deterministic selection in the normalized frame
      -> absolute action back
      -> the worker's engine validates and applies it

The checkpoint is loaded once per owner, and an owner outlives a whole sweep:
the 1/2/4/8/shuffled runs of the greedy gate share a single loaded model.

Why decisions cannot depend on the worker count
-----------------------------------------------
Two independent reasons, and both are needed.

*The game inputs* were already worker-count-independent: `MatchSpec` fixes the
setups, the colour assignment and both policy seeds before dispatch, and
`derive_decision_seed(policy_seed, ply)` fixes each decision's stream from the
ply alone. Nothing in that chain can see a worker index, a shard boundary or a
clock. That is Phase 4's guarantee and this module does not touch it.

*The model input* is the part that is new, and it is where a batching design can
quietly break reproducibility. Under the default `single_request` batch policy
the tensor the network sees for a decision is built from that decision's request
and nothing else, so the logits -- and therefore the action -- are a pure
function of the request. Worker count, chunking, arrival timing and schedule
order change only *when* a request is served, never *what* is computed for it.

The alternative `arrival_batched` policy exists to measure what batching buys
and costs. Its batch membership depends on which workers happened to be waiting,
so it is a performance instrument only: no gate in this project is allowed to
run under it, and :func:`compare_batch_policies` is provided to measure whether
it agrees with the deterministic path rather than to assume that it does. Phase
6's common contract is explicit that approximate float batch equivalence does
not guarantee identical actions in near-tie positions.

Request ordering is canonical either way. The owner sorts everything it has
drained by `(match_id, ply, acting_player, worker_index)` before serving it, so
the sequence of forward passes for a given set of pending requests is fixed by
identity rather than by arrival.

No torch in a game worker
-------------------------
This module is imported by the worker processes, so it must not import torch at
module scope, and it does not. :class:`InferenceOwner` imports torch and
:mod:`stratego.model` lazily, inside the parent process that already owns Metal.

That is necessary but not sufficient, and the gap is worth stating because it is
easy to trip over. `spawn` re-imports the *parent's* `__main__` module in every
child, so a launcher script that imports torch at module scope puts torch in
every game worker even though nothing in this package asked for it. The property
is therefore measured rather than assumed: every worker reports what it actually
imported (:func:`worker_module_report`), :attr:`NeuralRunSummary.workers_importing_torch`
counts the offenders, and the acceptance harness keeps its own torch imports
inside functions so the count is zero. A launcher that gets this wrong produces
a run whose data file says so.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import queue as queue_module
import random
import sys
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..engine.constants import ACTION_SPACE_SIZE, OBSERVATION_SHAPE, PLAYERS, RulesConfig
from .match_runner import (
    ERROR_ILLEGAL_ACTION,
    MATCH_RESULT_SCHEMA_VERSION,
    MATCH_RUNNER_VERSION,
    ON_POLICY_ERROR_MODES,
    ON_POLICY_ERROR_RAISE,
    MatchResult,
    _rules_payload,
    play_match,
    resolve_policies,
    results_digest,
)
from .match_spec import MatchSpec, rules_token, schedule_digest, shard_schedule
from .policy import (
    Policy,
    PolicyContractError,
    PolicyInput,
    PolicyRef,
    PolicyRequirements,
    PolicyResult,
)
from .setup_bank import SetupBank

NEURAL_WORKER_VERSION = "neural_worker_v1"

#: Selection modes, spelled here so a game worker can name one without importing
#: :mod:`stratego.model.policy_adapter` (which imports torch). A test asserts
#: these three constants equal the adapter's, so drift is loud rather than
#: silent.
DECISION_MODE_GREEDY = "greedy"
DECISION_MODE_CATEGORICAL = "seeded_categorical"
DECISION_MODES = (DECISION_MODE_GREEDY, DECISION_MODE_CATEGORICAL)
NEURAL_DECISION_RULE_VERSION = "0.2.0"

#: Batch policies. Only the first may be used for a reproducibility gate; see
#: the module docstring.
BATCH_POLICY_SINGLE = "single_request"
BATCH_POLICY_ARRIVAL = "arrival_batched"
BATCH_POLICIES = (BATCH_POLICY_SINGLE, BATCH_POLICY_ARRIVAL)

#: Message kinds on the shared request queue.
KIND_REQUEST = "inference_request"
KIND_WORKER_DONE = "worker_done"

#: How long a worker waits for one answer before declaring the owner gone, and
#: how long the parent waits for a worker's rows once it has said it finished.
DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_RESULT_TIMEOUT = 120.0
#: How long a silently dead worker is given to have its final message flushed
#: through the queue's feeder thread before the run is declared broken.
WORKER_DEATH_GRACE_SECONDS = 5.0


class NeuralEvaluationError(RuntimeError):
    """Raised when a parallel neural run cannot be completed or trusted."""


class RemoteInferenceError(PolicyContractError):
    """A decision could not be obtained from the inference owner.

    Subclasses :class:`~stratego.evaluation.policy.PolicyContractError` so the
    Phase 4 runner classifies it like every other policy failure: the match is
    aborted or quarantined and *no* action is submitted. Nothing in this module
    ever answers a failed inference with a random, first or previous legal move.
    """


# ---------------------------------------------------------------------------
# Policy identity
# ---------------------------------------------------------------------------
#
# A neural evaluation policy is "these weights, this selection rule, this
# arithmetic precision". All three change decisions, so all three are in the
# identity -- and therefore in `match_id`. None of these identifiers is added to
# the Phase 4 catalogue: `stratego.evaluation.registry` still enumerates exactly
# the ladder and stress opponents it did in Phase 4.

_MODE_TOKENS = {DECISION_MODE_GREEDY: "greedy", DECISION_MODE_CATEGORICAL: "sampled"}


def neural_policy_ref(
    candidate_id: str,
    *,
    decision_mode: str = DECISION_MODE_GREEDY,
    dtype_name: str = "float32",
    rule_version: str = NEURAL_DECISION_RULE_VERSION,
) -> PolicyRef:
    """The `id@version` a Phase 6 candidate checkpoint plays under.

    Precision is part of the version because it is part of the decision rule: a
    float16 forward pass can flip a near-tie, and a result recorded under one
    precision must not be attributed to the other.
    """
    if decision_mode not in DECISION_MODES:
        raise NeuralEvaluationError(f"unknown decision mode {decision_mode!r}")
    return PolicyRef(
        policy_id=f"phase6_{candidate_id.lower()}_{_MODE_TOKENS[decision_mode]}",
        policy_version=f"{rule_version}+{dtype_name}",
    )


# ---------------------------------------------------------------------------
# The observer-safe payload
# ---------------------------------------------------------------------------


#: Exactly the fields an inference request may carry. The information-safety
#: test asserts the dataclass has these and nothing else, so widening the
#: payload cannot happen without editing this tuple deliberately.
REQUEST_FIELDS: tuple[str, ...] = (
    "request_id",
    "match_id",
    "paired_unit_id",
    "ply",
    "acting_player",
    "decision_seed",
    "observation",
    "legal_actions",
    "legal_action_mask",
)


@dataclass(frozen=True)
class InferenceRequest:
    """One decision, as the neural policy is allowed to see it.

    Every field is either an identifier the policy already knew, the per-ply
    decision seed Phase 4 derived, or an observer-safe engine product. There is
    no `GameState`, no `PieceRecord`, no hidden identity, no belief target, no
    opponent setup and no replay object anywhere in the graph -- and because
    `observation` is copied out of engine-owned memory on construction, the
    payload does not even alias one.
    """

    request_id: str
    match_id: str
    paired_unit_id: str
    ply: int
    acting_player: int
    decision_seed: int
    #: `(127, 10, 10)` float32, `observation_v2_1_127ch`, acting player's view.
    observation: np.ndarray
    #: Absolute engine action identifiers, ascending.
    legal_actions: tuple[int, ...]
    #: The engine's dense `uint8` legality mask, in absolute squares.
    legal_action_mask: np.ndarray

    @staticmethod
    def request_id_for(match_id: str, ply: int) -> str:
        """Identity of a decision: one match plays one ply exactly once."""
        return f"{match_id}#{int(ply)}"

    @staticmethod
    def from_policy_input(request: PolicyInput) -> "InferenceRequest":
        """Project a `PolicyInput` down to the transportable payload.

        The arrays are copied rather than referenced: an engine observation is
        handed out read-only as a view of engine-owned memory, and a request
        that travelled to another process must not be distinguishable from one
        that did not.
        """
        observation = request.require_observation()
        mask = request.require_legal_action_mask()
        return InferenceRequest(
            request_id=InferenceRequest.request_id_for(request.match_id, request.ply),
            match_id=request.match_id,
            paired_unit_id=request.paired_unit_id,
            ply=int(request.ply),
            acting_player=int(request.acting_player),
            decision_seed=int(request.decision_seed),
            observation=np.array(observation, dtype=np.float32, copy=True),
            legal_actions=tuple(int(action) for action in request.legal_actions),
            legal_action_mask=np.array(mask, dtype=np.uint8, copy=True),
        )

    @property
    def sort_key(self) -> tuple:
        """Canonical ordering key. Identity only -- never arrival or worker."""
        return (self.match_id, self.ply, self.acting_player, self.request_id)

    def identity(self) -> dict:
        """The non-array part of the payload, for logs and audits."""
        return {
            "request_id": self.request_id,
            "match_id": self.match_id,
            "paired_unit_id": self.paired_unit_id,
            "ply": self.ply,
            "acting_player": self.acting_player,
            "decision_seed": self.decision_seed,
            "legal_action_count": len(self.legal_actions),
        }

    def digest(self) -> str:
        """Content digest of one request, for the deterministic-ordering tests."""
        hasher = hashlib.sha256()
        hasher.update(self.request_id.encode())
        hasher.update(str(self.acting_player).encode())
        hasher.update(str(self.decision_seed).encode())
        hasher.update(np.ascontiguousarray(self.observation, dtype=np.float32).tobytes())
        hasher.update(np.asarray(self.legal_actions, dtype=np.int64).tobytes())
        return hasher.hexdigest()


@dataclass(frozen=True)
class InferenceResponse:
    """One decision the owner is willing to stand behind."""

    request_id: str
    decision_seed: int
    absolute_action_id: int
    model_action_id: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    batch_size: int = 1
    batch_index: int = 0


@dataclass(frozen=True)
class InferenceFailure:
    """A refusal. Carries no action, by design.

    Returned instead of an :class:`InferenceResponse` whenever the owner cannot
    produce a trustworthy decision -- a malformed request, a non-finite logit on
    a legal action, a conversion that landed outside the legal set, or a fault
    in the owner itself. The worker turns it into a raised
    :class:`RemoteInferenceError`; nothing substitutes a move.
    """

    request_id: str
    error_type: str
    message: str
    fatal: bool = False


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class InferenceChannel(ABC):
    """How a game worker reaches the inference owner."""

    #: For diagnostics and the topology record in the data file.
    transport: ClassVar[str] = "abstract"

    @abstractmethod
    def infer(self, request: InferenceRequest) -> "InferenceResponse | InferenceFailure":
        """Answer one request. Must not raise for a *refused* decision."""

    def stats(self) -> dict:
        return {}


class LocalInferenceChannel(InferenceChannel):
    """In-process channel: the owner is called directly.

    Used by the `worker_count=1` path, which is both the fast path for small
    runs and the serial reference the parallel sweep is compared against. It
    exercises the same owner code as the multiprocess path, so a difference
    between one worker and eight can only come from the transport.
    """

    transport = "in_process"

    def __init__(self, owner: "InferenceOwner"):
        self.owner = owner
        self.requests = 0

    def infer(self, request: InferenceRequest) -> "InferenceResponse | InferenceFailure":
        self.requests += 1
        return self.owner.serve(request)

    def stats(self) -> dict:
        return {"transport": self.transport, "requests": self.requests}


class QueueInferenceChannel(InferenceChannel):
    """Cross-process channel: one shared request queue, one private reply queue.

    The reply queue is per worker rather than shared, so an answer cannot be
    delivered to the wrong game even transiently. The worker also checks the
    returned `request_id` and `decision_seed` against what it asked for, which
    turns a crossed wire into a loud failure instead of a plausible move.
    """

    transport = "spawned_processes"

    def __init__(
        self,
        worker_index: int,
        request_queue,
        response_queue,
        *,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ):
        self.worker_index = int(worker_index)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.timeout = float(timeout)
        self.requests = 0
        self.wait_seconds = 0.0

    def infer(self, request: InferenceRequest) -> "InferenceResponse | InferenceFailure":
        self.requests += 1
        self.request_queue.put(
            {
                "kind": KIND_REQUEST,
                "worker_index": self.worker_index,
                # Wall clock: `perf_counter` has no cross-process meaning, and
                # this number is only ever used as a queue-wait measurement.
                "enqueued_at": time.time(),
                "request": request,
            }
        )
        started = time.perf_counter()
        try:
            message = self.response_queue.get(timeout=self.timeout)
        except queue_module.Empty as error:
            raise RemoteInferenceError(
                f"the inference owner did not answer request {request.request_id!r} "
                f"within {self.timeout:g}s; refusing to continue this match"
            ) from error
        finally:
            self.wait_seconds += time.perf_counter() - started
        return message

    def stats(self) -> dict:
        return {
            "transport": self.transport,
            "requests": self.requests,
            "wait_seconds": round(self.wait_seconds, 6),
        }


# ---------------------------------------------------------------------------
# The worker-side policy
# ---------------------------------------------------------------------------


class RemoteNeuralPolicy(Policy):
    """A `policy_interface_v1` policy whose forward pass happens elsewhere.

    Holds no model, no weights and no torch import. It declares exactly the two
    observer-safe products the neural policy needs, ships them, and validates
    the answer before letting it become a `PolicyResult` -- after which the Phase
    4 runner re-checks legality and the engine checks it again.
    """

    policy_id: ClassVar[str] = "remote_neural_unbound"
    policy_version: ClassVar[str] = NEURAL_DECISION_RULE_VERSION
    requirements: ClassVar[PolicyRequirements] = PolicyRequirements(
        observation=True,
        legal_action_mask=True,
        public_view=False,
    )
    description: ClassVar[str] = (
        "Phase 6 checkpoint policy served by a long-lived inference owner; the "
        "decision rules are stratego.model.policy_adapter's, unchanged."
    )

    def __init__(
        self,
        ref: PolicyRef,
        channel: InferenceChannel,
        *,
        decision_mode: str = DECISION_MODE_GREEDY,
    ):
        if decision_mode not in DECISION_MODES:
            raise NeuralEvaluationError(f"unknown decision mode {decision_mode!r}")
        # Instance attributes shadow the class attributes, so one class can carry
        # any candidate/mode/precision identity without a subclass per checkpoint.
        self.policy_id = ref.policy_id
        self.policy_version = ref.policy_version
        self.decision_mode = decision_mode
        self.stochastic = decision_mode == DECISION_MODE_CATEGORICAL
        self.channel = channel
        self.decisions = 0

    def decide(self, request: PolicyInput) -> PolicyResult:
        payload = InferenceRequest.from_policy_input(request)
        answer = self.channel.infer(payload)

        if isinstance(answer, InferenceFailure):
            raise RemoteInferenceError(
                f"the inference owner refused decision {payload.request_id!r} "
                f"({answer.error_type}): {answer.message}"
            )
        if not isinstance(answer, InferenceResponse):
            raise RemoteInferenceError(
                f"the inference owner returned {type(answer).__name__} for "
                f"{payload.request_id!r}, expected an InferenceResponse"
            )
        if answer.request_id != payload.request_id:
            raise RemoteInferenceError(
                f"the inference owner answered {answer.request_id!r} while this "
                f"worker asked about {payload.request_id!r}"
            )
        if answer.decision_seed != request.decision_seed:
            raise RemoteInferenceError(
                f"decision {payload.request_id!r} came back under seed "
                f"{answer.decision_seed} but Phase 4 derived {request.decision_seed}"
            )

        self.decisions += 1
        diagnostics = dict(answer.diagnostics)
        diagnostics.update(
            {
                "transport": self.channel.transport,
                "inference_batch_size": answer.batch_size,
            }
        )
        return self.result(request, answer.absolute_action_id, diagnostics)

    def describe(self) -> dict:
        description = super().describe()
        description.update(
            {
                "decision_mode": self.decision_mode,
                "transport": self.channel.transport,
                "neural_worker_version": NEURAL_WORKER_VERSION,
                "holds_model_weights": False,
            }
        )
        return description


def worker_module_report() -> dict:
    """What the calling process has imported, for the MPS-ownership test.

    A game worker must be a pure engine/NumPy process. This is the observation
    the test makes, taken inside the worker and shipped back with its rows.
    """
    return {
        "pid": os.getpid(),
        "torch_imported": "torch" in sys.modules,
        "model_modules_imported": sorted(
            name
            for name in sys.modules
            if name == "stratego.model" or name.startswith("stratego.model.")
        ),
        # Zero unless something in this process built an owner, which a game
        # worker must never do.
        "checkpoint_loads": checkpoint_load_count(),
    }


# ---------------------------------------------------------------------------
# The inference owner
# ---------------------------------------------------------------------------
#
# Everything below this line runs in the process that owns Metal. torch and
# `stratego.model` are imported inside `InferenceOwner`, never at module scope,
# because this module is also imported by every game worker.

#: Process-wide count of checkpoint loads performed by an owner. The gate is
#: "one per long-lived owner", and a counter that lives outside the owner is the
#: only way to notice an owner being rebuilt in a loop.
_CHECKPOINT_LOADS = 0


def checkpoint_load_count() -> int:
    """How many checkpoints :class:`InferenceOwner` has loaded in this process."""
    return _CHECKPOINT_LOADS


def reset_checkpoint_load_count() -> None:
    """Zero the counter. For tests and for one measured run at a time."""
    global _CHECKPOINT_LOADS
    _CHECKPOINT_LOADS = 0


class InferenceOwner:
    """The single long-lived holder of the model, the device and the weights.

    One owner serves an entire sweep: the greedy 1/2/4/8/shuffled runs share one
    loaded checkpoint, which is what the "checkpoint loads per long-lived
    inference owner = 1" requirement means in practice.

    The decision rules are not reimplemented here. `prepare_legality` and
    `select_action` are imported from :mod:`stratego.model.policy_adapter`, so
    the frame conversion, the legality cross-check, the greedy tie-break, the
    categorical sampler and the "converted back to an illegal action" refusal
    are the same code the serial Phase 5 adapter runs.
    """

    def __init__(
        self,
        checkpoint_path: "str | Path",
        *,
        decision_mode: str = DECISION_MODE_GREEDY,
        device: str = "mps",
        dtype: str = "float32",
        expected_architecture_id: str | None = None,
        expected_configuration: Any = None,
        batch_policy: str = BATCH_POLICY_SINGLE,
        max_batch_size: int = 1,
        name: str = "inference_owner",
    ):
        global _CHECKPOINT_LOADS

        if decision_mode not in DECISION_MODES:
            raise NeuralEvaluationError(f"unknown decision mode {decision_mode!r}")
        if batch_policy not in BATCH_POLICIES:
            raise NeuralEvaluationError(
                f"unknown batch policy {batch_policy!r}; known policies are "
                f"{', '.join(BATCH_POLICIES)}"
            )
        if batch_policy == BATCH_POLICY_SINGLE and max_batch_size != 1:
            raise NeuralEvaluationError(
                "the single_request batch policy serves exactly one request per "
                f"forward pass; max_batch_size={max_batch_size} contradicts it"
            )
        if max_batch_size < 1:
            raise NeuralEvaluationError(f"max_batch_size must be at least 1, got {max_batch_size}")

        # Lazy, and deliberately so: importing this module must not import torch
        # into a game worker. See the module docstring.
        import torch

        from ..model import policy_adapter as adapter
        from ..model.checkpoint import load_checkpoint
        from ..model.contract import MODEL_CONTRACT_VERSION, POLICY_ACTION_FRAME
        from ..model.tokenization import observation_batch_from_numpy, observation_to_tokens

        self._torch = torch
        self._adapter = adapter
        self._observation_batch_from_numpy = observation_batch_from_numpy
        self._observation_to_tokens = observation_to_tokens
        self._contract_version = MODEL_CONTRACT_VERSION
        self._policy_action_frame = POLICY_ACTION_FRAME

        if adapter.DECISION_MODE_GREEDY != DECISION_MODE_GREEDY or (
            adapter.DECISION_MODE_CATEGORICAL != DECISION_MODE_CATEGORICAL
        ):  # pragma: no cover - guarded by a test
            raise NeuralEvaluationError(
                "the decision-mode constants in stratego.evaluation.neural_worker have "
                "drifted from stratego.model.policy_adapter"
            )

        self.name = str(name)
        self.checkpoint_path = Path(checkpoint_path)
        self.decision_mode = decision_mode
        self.batch_policy = batch_policy
        self.max_batch_size = int(max_batch_size)
        self.device = torch.device(device)
        self.dtype_name = str(dtype)
        self.dtype = getattr(torch, self.dtype_name)

        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise NeuralEvaluationError(
                "device 'mps' was requested but Metal is not available; refusing to "
                "silently fall back to the CPU"
            )

        started = time.perf_counter()
        model, metadata = load_checkpoint(
            self.checkpoint_path,
            device=self.device,
            dtype=self.dtype,
            expected_architecture_id=expected_architecture_id,
            expected_configuration=expected_configuration,
        )
        self.checkpoint_load_seconds = time.perf_counter() - started
        self.checkpoint_load_count = 1
        _CHECKPOINT_LOADS += 1

        self.model = model
        self.metadata = dict(metadata)
        self.closed = False

        # Fault injection for the "inference coordinator failure" case. A test
        # sets it; nothing in a normal run does.
        self.fault_hook = None

        self._aborted = False
        self._fault: str | None = None
        self._requests_served = 0
        self._failures_returned = 0
        self._batches_served = 0
        self._inference_seconds = 0.0
        self._serve_seconds = 0.0
        self._batch_sizes: dict[int, int] = {}

    # -- identity ----------------------------------------------------------

    def identity(self) -> dict:
        """Everything a report needs to say which weights produced a result."""
        return {
            "owner_name": self.name,
            "neural_worker_version": NEURAL_WORKER_VERSION,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_file_digest": self.metadata.get("checkpoint_file_digest"),
            "state_dict_digest": self.metadata.get("state_dict_digest"),
            "model_architecture_id": self.metadata.get("model_architecture_id"),
            "model_configuration": self.metadata.get("model_configuration"),
            "model_contract_version": self.metadata.get("model_contract_version"),
            "policy_action_frame": self.metadata.get("policy_action_frame"),
            "engine_action_frame": self.metadata.get("engine_action_frame"),
            "rules_version": self.metadata.get("rules_version"),
            "observation_version": self.metadata.get("observation_version"),
            "action_encoding_version": self.metadata.get("action_encoding_version"),
            "parameter_count": self.model.parameter_count(),
            "device": str(self.device),
            "dtype": self.dtype_name,
            "decision_mode": self.decision_mode,
            "batch_policy": self.batch_policy,
            "max_batch_size": self.max_batch_size,
            "checkpoint_load_count": self.checkpoint_load_count,
            "checkpoint_load_seconds": round(self.checkpoint_load_seconds, 6),
        }

    def stats(self) -> dict:
        """Cumulative counters. The orchestrator diffs these around a run."""
        return {
            "requests_served": self._requests_served,
            "failures_returned": self._failures_returned,
            "batches_served": self._batches_served,
            # `inference_seconds` is the forward pass alone; `serve_seconds` is
            # everything the owner does for a decision -- request validation,
            # the legality cross-check, both frame conversions, the forward,
            # selection and diagnostics. The gap between them is the owner's
            # CPU-side cost, which is single-threaded and therefore on the
            # critical path of every decision in the run.
            "inference_seconds": self._inference_seconds,
            "serve_seconds": self._serve_seconds,
            "batch_sizes": dict(self._batch_sizes),
            "aborted": self._aborted,
            "fault": self._fault,
        }

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def fault(self) -> str | None:
        return self._fault

    # -- serving -----------------------------------------------------------

    def serve(self, request: InferenceRequest) -> "InferenceResponse | InferenceFailure":
        """Answer exactly one request in its own forward pass."""
        return self.serve_batch([request])[0]

    def serve_batch(
        self, requests: "Sequence[InferenceRequest]"
    ) -> list["InferenceResponse | InferenceFailure"]:
        """Answer a group of requests from one forward pass.

        Never raises for a bad *request*: an unusable request produces an
        :class:`InferenceFailure` for that request alone and the rest of the
        batch is still served. A failure of the *owner* -- the device, the
        model, the fault hook -- aborts the owner, fails every request in flight
        and every request afterwards, and is reported through `fault`.
        """
        if self.closed:
            raise NeuralEvaluationError(f"inference owner {self.name!r} is closed")

        pending = list(requests)
        if not pending:
            return []
        entered = time.perf_counter()
        try:
            return self._serve_batch(pending)
        finally:
            self._serve_seconds += time.perf_counter() - entered

    def _serve_batch(
        self, pending: "list[InferenceRequest]"
    ) -> list["InferenceResponse | InferenceFailure"]:
        if self._aborted:
            return [
                InferenceFailure(
                    request.request_id,
                    "OwnerAborted",
                    f"the inference owner aborted earlier: {self._fault}",
                    fatal=True,
                )
                for request in pending
            ]

        responses: list[Any] = [None] * len(pending)
        prepared: list[tuple[int, InferenceRequest, Any]] = []
        for position, request in enumerate(pending):
            try:
                self._validate_request(request)
                # Legality first, before any kernel runs: exactly the order the
                # serial adapter uses.
                legality = self._adapter.prepare_legality(
                    request.legal_actions, request.legal_action_mask, request.acting_player
                )
            except Exception as error:  # noqa: BLE001 -- reported per request
                responses[position] = InferenceFailure(
                    getattr(request, "request_id", "<unknown>"),
                    type(error).__name__,
                    str(error),
                )
                self._failures_returned += 1
                continue
            prepared.append((position, request, legality))

        if prepared:
            try:
                if self.fault_hook is not None:
                    self.fault_hook([request for _, request, _ in prepared])
                outputs, elapsed = self._forward([request for _, request, _ in prepared])
            except Exception as error:  # noqa: BLE001 -- an owner-level fault
                self._aborted = True
                self._fault = f"{type(error).__name__}: {error}"
                for position, request, _ in prepared:
                    responses[position] = InferenceFailure(
                        request.request_id, type(error).__name__, str(error), fatal=True
                    )
                    self._failures_returned += 1
                return responses

            self._batches_served += 1
            self._inference_seconds += elapsed
            self._batch_sizes[len(prepared)] = self._batch_sizes.get(len(prepared), 0) + 1

            for row, (position, request, legality) in enumerate(prepared):
                try:
                    responses[position] = self._select(outputs, row, request, legality, len(prepared))
                    self._requests_served += 1
                except Exception as error:  # noqa: BLE001 -- reported per request
                    responses[position] = InferenceFailure(
                        request.request_id, type(error).__name__, str(error)
                    )
                    self._failures_returned += 1

        return responses

    def _select(self, outputs, row: int, request, legality, batch_size: int) -> InferenceResponse:
        # One stream per decision, built from the Phase 4 decision seed alone --
        # never from a global generator advanced in arrival order.
        rng = (
            random.Random(request.decision_seed)
            if self.decision_mode == DECISION_MODE_CATEGORICAL
            else None
        )
        selection = self._adapter.select_action(
            outputs.policy_logits[row],
            legality,
            decision_mode=self.decision_mode,
            rng=rng,
        )
        diagnostics = {
            "mode": self.decision_mode,
            "legal_action_count": selection.legal_action_count,
            "model_architecture_id": self.model.architecture_id,
            "model_contract_version": self._contract_version,
            "policy_action_frame": self._policy_action_frame,
            "source_square": selection.source_square,
            "destination_square": selection.destination_square,
            "model_action_id": selection.model_action_id,
            "selected_logit": selection.selected_logit,
        }
        diagnostics.update(self._adapter.value_diagnostics(outputs.value_logits, row))
        return InferenceResponse(
            request_id=request.request_id,
            decision_seed=request.decision_seed,
            absolute_action_id=selection.absolute_action_id,
            model_action_id=selection.model_action_id,
            diagnostics=diagnostics,
            batch_size=batch_size,
            batch_index=row,
        )

    def probe_policy_logits(self, requests: "Sequence[InferenceRequest]") -> list:
        """Raw policy-logit rows for `requests`, as detached CPU float32 tensors.

        A measurement surface, not a decision path: it exists so a caller can ask
        whether this architecture on this device gives the *same* row for a
        position evaluated alone and inside a batch. That question has to be
        answered by measurement rather than assumption -- Phase 6's common
        contract is explicit that approximate float batch equivalence does not
        guarantee identical actions in near-tie positions -- and the answer is
        what tells a later agent whether batching may be enabled deliberately.
        """
        if self.closed:
            raise NeuralEvaluationError(f"inference owner {self.name!r} is closed")
        for request in requests:
            self._validate_request(request)
        outputs, _ = self._forward(list(requests))
        return [
            outputs.policy_logits[row].detach().to("cpu", self._torch.float32).clone()
            for row in range(len(requests))
        ]

    def _forward(self, requests: "Sequence[InferenceRequest]"):
        torch = self._torch
        batch = self._observation_batch_from_numpy(
            [request.observation for request in requests],
            dtype=self.dtype,
            device=self.device,
        )
        started = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(self._observation_to_tokens(batch))
        if self.device.type == "mps":
            torch.mps.synchronize()
        return outputs, time.perf_counter() - started

    def _validate_request(self, request: Any) -> None:
        """Everything the owner refuses to guess about."""
        if not isinstance(request, InferenceRequest):
            raise NeuralEvaluationError(
                f"expected an InferenceRequest, got {type(request).__name__}"
            )
        if not isinstance(request.request_id, str) or not request.request_id:
            raise NeuralEvaluationError("the request carries no request_id")
        if request.acting_player not in PLAYERS:
            raise NeuralEvaluationError(
                f"request {request.request_id!r} names acting player "
                f"{request.acting_player!r}, which is not a player"
            )
        if not isinstance(request.decision_seed, (int, np.integer)) or request.decision_seed < 0:
            raise NeuralEvaluationError(
                f"request {request.request_id!r} carries decision seed "
                f"{request.decision_seed!r}"
            )
        observation = np.asarray(request.observation)
        if observation.shape != OBSERVATION_SHAPE:
            raise NeuralEvaluationError(
                f"request {request.request_id!r} carries an observation of shape "
                f"{observation.shape}, expected {OBSERVATION_SHAPE}"
            )
        if not np.isfinite(observation).all():
            raise NeuralEvaluationError(
                f"request {request.request_id!r} carries a non-finite observation"
            )
        mask = np.asarray(request.legal_action_mask)
        if mask.shape != (ACTION_SPACE_SIZE,):
            raise NeuralEvaluationError(
                f"request {request.request_id!r} carries a legality mask of shape "
                f"{mask.shape}, expected ({ACTION_SPACE_SIZE},)"
            )
        if not request.legal_actions:
            raise NeuralEvaluationError(
                f"request {request.request_id!r} carries an empty legal-action list"
            )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Release the model and the Metal allocator. Idempotent."""
        if self.closed:
            return
        self.closed = True
        self.model = None
        if self.device.type == "mps":
            try:
                self._torch.mps.empty_cache()
            except Exception:  # pragma: no cover - allocator cleanup is best effort
                pass

    def __enter__(self) -> "InferenceOwner":
        return self

    def __exit__(self, *exception) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Running a schedule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeuralRunSummary:
    """What one execution of a neural schedule produced.

    Field names match :class:`~stratego.evaluation.match_runner.RunSummary`
    wherever they mean the same thing, so a sweep can compare a neural run
    against a baseline run without a translation layer.
    """

    results: tuple[MatchResult, ...]
    schedule_digest: str
    results_digest: str
    worker_count: int
    chunk_count: int
    matches_run: int
    paired_units_run: int
    policy_errors: int
    illegal_policy_actions: int
    wall_clock_seconds: float
    decisions: int
    inference: Mapping[str, Any]
    workers: tuple[Mapping[str, Any], ...]
    transport: str
    batch_policy: str
    runner_version: str = MATCH_RUNNER_VERSION
    schema_version: str = MATCH_RESULT_SCHEMA_VERSION
    neural_worker_version: str = NEURAL_WORKER_VERSION

    @property
    def plies(self) -> int:
        return sum(row.plies for row in self.results)

    @property
    def replay_digests(self) -> dict[str, str]:
        return {row.match_id: row.replay_digest for row in self.results}

    @property
    def workers_importing_torch(self) -> int:
        """Separate game-worker processes that ended up importing torch.

        Must be zero. It is not zero automatically: see the module docstring on
        `spawn` re-importing the launcher's `__main__`. The `worker_count=1`
        path is excluded because there the "worker" *is* the owner's process,
        which necessarily holds torch; that run is the serial reference, not a
        claim about process separation.
        """
        return sum(
            1
            for report in self.workers
            if not report.get("in_owner_process")
            and report.get("modules", {}).get("torch_imported")
        )

    @property
    def worker_checkpoint_loads(self) -> int:
        """Checkpoint loads performed inside game workers. Must be zero."""
        return sum(int(report.get("checkpoint_loads", 0)) for report in self.workers)

    def summary_dict(self) -> dict:
        return {
            "runner_version": self.runner_version,
            "neural_worker_version": self.neural_worker_version,
            "match_result_schema_version": self.schema_version,
            "schedule_digest": self.schedule_digest,
            "results_digest": self.results_digest,
            "worker_count": self.worker_count,
            "chunk_count": self.chunk_count,
            "transport": self.transport,
            "batch_policy": self.batch_policy,
            "matches_run": self.matches_run,
            "paired_units_run": self.paired_units_run,
            "policy_errors": self.policy_errors,
            "illegal_policy_actions": self.illegal_policy_actions,
            "total_plies": self.plies,
            "decisions": self.decisions,
            "wall_clock_seconds": self.wall_clock_seconds,
            "workers_importing_torch": self.workers_importing_torch,
            "worker_checkpoint_loads": self.worker_checkpoint_loads,
            "inference": dict(self.inference),
        }


def run_neural_schedule(
    matches: "Sequence[MatchSpec]",
    bank: SetupBank,
    owner: InferenceOwner,
    *,
    policy_ref: "PolicyRef | None" = None,
    worker_count: int = 1,
    chunks_per_worker: int = 4,
    record_actions: bool = True,
    on_policy_error: str = ON_POLICY_ERROR_RAISE,
    verify_invariants: bool = False,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    result_timeout: float = DEFAULT_RESULT_TIMEOUT,
) -> NeuralRunSummary:
    """Run a schedule whose candidate is `owner`'s checkpoint.

    `owner` is passed in already loaded and is *not* closed here: one owner is
    meant to serve a whole sweep, which is what makes "one checkpoint load per
    long-lived inference owner" observable rather than asserted.

    `worker_count=1` runs in this process against the same owner through
    :class:`LocalInferenceChannel`. Higher counts spawn that many pure-engine
    processes, which reach the owner over queues. Under the default
    `single_request` batch policy the two paths compute bit-identical model
    inputs, so they are required to produce identical games.
    """
    if worker_count < 1:
        raise NeuralEvaluationError(f"worker_count must be at least 1, got {worker_count}")
    if chunks_per_worker < 1:
        raise NeuralEvaluationError(
            f"chunks_per_worker must be at least 1, got {chunks_per_worker}"
        )
    if on_policy_error not in ON_POLICY_ERROR_MODES:
        raise NeuralEvaluationError(f"unknown on_policy_error mode {on_policy_error!r}")

    specs = tuple(matches)
    if not specs:
        raise NeuralEvaluationError("run_neural_schedule was given an empty schedule")

    rules_set = {rules_token(spec.rules) for spec in specs}
    if len(rules_set) > 1:
        raise NeuralEvaluationError(
            "a single run must use one rules configuration; this schedule mixes "
            f"{len(rules_set)}"
        )

    reference = policy_ref if policy_ref is not None else _infer_policy_ref(specs)
    _require_reference_present(specs, reference)

    before = owner.stats()
    started = time.perf_counter()
    if worker_count == 1:
        results, chunk_count, worker_reports, service = _run_locally(
            specs,
            bank=bank,
            owner=owner,
            reference=reference,
            record_actions=record_actions,
            on_policy_error=on_policy_error,
            verify_invariants=verify_invariants,
        )
        transport = LocalInferenceChannel.transport
    else:
        results, chunk_count, worker_reports, service = _run_across_processes(
            specs,
            bank=bank,
            owner=owner,
            reference=reference,
            worker_count=worker_count,
            chunks_per_worker=chunks_per_worker,
            record_actions=record_actions,
            on_policy_error=on_policy_error,
            verify_invariants=verify_invariants,
            request_timeout=request_timeout,
            result_timeout=result_timeout,
        )
        transport = QueueInferenceChannel.transport
    elapsed = time.perf_counter() - started
    after = owner.stats()

    if owner.aborted:
        raise NeuralEvaluationError(
            f"the inference owner {owner.name!r} aborted during the run: {owner.fault}"
        )

    ordered = tuple(sorted(results, key=lambda row: row.match_id))
    inference = _inference_delta(before, after)
    inference.update(service)
    inference["checkpoint_load_count"] = owner.checkpoint_load_count
    inference["checkpoint_loads_in_process"] = checkpoint_load_count()

    return NeuralRunSummary(
        results=ordered,
        schedule_digest=schedule_digest(specs),
        results_digest=results_digest(ordered),
        worker_count=worker_count,
        chunk_count=chunk_count,
        matches_run=len(ordered),
        paired_units_run=len({row.paired_unit_id for row in ordered}),
        policy_errors=sum(1 for row in ordered if row.errored),
        illegal_policy_actions=sum(
            1 for row in ordered if row.policy_error_category == ERROR_ILLEGAL_ACTION
        ),
        wall_clock_seconds=elapsed,
        decisions=int(inference.get("requests_served", 0)),
        inference=inference,
        workers=tuple(worker_reports),
        transport=transport,
        batch_policy=owner.batch_policy,
    )


def _infer_policy_ref(specs: "Sequence[MatchSpec]") -> PolicyRef:
    """The one policy in this schedule the owner is expected to play.

    A schedule may only contain one neural side; anything else would need a
    second owner and is refused rather than guessed at.
    """
    refs = {spec.candidate for spec in specs} | {spec.opponent for spec in specs}
    served = sorted(
        {ref for ref in refs if ref.policy_id.startswith("phase6_")},
        key=lambda ref: ref.token,
    )
    if len(served) != 1:
        raise NeuralEvaluationError(
            "could not infer which policy the inference owner plays "
            f"({[ref.token for ref in served]}); pass policy_ref explicitly"
        )
    return served[0]


def _require_reference_present(specs: "Sequence[MatchSpec]", reference: PolicyRef) -> None:
    missing = [
        spec.match_id
        for spec in specs
        if reference not in (spec.candidate, spec.opponent)
    ]
    if missing:
        raise NeuralEvaluationError(
            f"{len(missing)} match(es) in this schedule do not name {reference.token}; "
            f"the first is {missing[0]}"
        )


def _inference_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict:
    sizes = {
        size: after["batch_sizes"].get(size, 0) - before["batch_sizes"].get(size, 0)
        for size in after["batch_sizes"]
    }
    sizes = {size: count for size, count in sizes.items() if count > 0}
    total_rows = sum(size * count for size, count in sizes.items())
    batches = sum(sizes.values())
    return {
        "requests_served": after["requests_served"] - before["requests_served"],
        "failures_returned": after["failures_returned"] - before["failures_returned"],
        "batches_served": batches,
        "inference_seconds": round(after["inference_seconds"] - before["inference_seconds"], 6),
        "serve_seconds": round(after["serve_seconds"] - before["serve_seconds"], 6),
        "batch_size_histogram": {str(size): count for size, count in sorted(sizes.items())},
        "max_batch_size_seen": max(sizes) if sizes else 0,
        "mean_batch_size": round(total_rows / batches, 4) if batches else 0.0,
    }


# -- the in-process path ----------------------------------------------------


def _run_locally(
    specs: "Sequence[MatchSpec]",
    *,
    bank: SetupBank,
    owner: InferenceOwner,
    reference: PolicyRef,
    record_actions: bool,
    on_policy_error: str,
    verify_invariants: bool,
):
    channel = LocalInferenceChannel(owner)
    policy = RemoteNeuralPolicy(reference, channel, decision_mode=owner.decision_mode)
    cache: dict[str, Policy] = {reference.token: policy}
    results = []
    for spec in specs:
        results.append(
            play_match(
                spec,
                bank=bank,
                policies=_policies_for(spec, cache),
                record_actions=record_actions,
                on_policy_error=on_policy_error,
                verify_invariants=verify_invariants,
            )
        )
    report = {
        "worker_index": 0,
        "status": "ok",
        "matches": len(results),
        "decisions": policy.decisions,
        "channel": channel.stats(),
        "modules": worker_module_report(),
        "checkpoint_loads": 0,
        # The serial reference plays inside the owner's own process, so its
        # module report describes the owner, not a game worker.
        "in_owner_process": True,
    }
    service = {
        # One dispatch per decision: the in-process channel never queues.
        "service_dispatches": policy.decisions,
        "queue_wait_seconds": 0.0,
        "queue_wait_mean_seconds": 0.0,
        "queue_wait_max_seconds": 0.0,
    }
    return results, 1, [report], service


def _policies_for(spec: MatchSpec, cache: "dict[str, Policy]") -> dict[str, Policy]:
    """The two policies this match needs, built at most once per process."""
    needed = (spec.candidate, spec.opponent)
    if any(ref.token not in cache for ref in needed):
        cache.update(resolve_policies(spec, cache))
    return {ref.token: cache[ref.token] for ref in needed}


# -- the multiprocess path --------------------------------------------------


def _neural_worker_main(
    worker_index: int,
    spec_payloads: list,
    bank_payload: dict,
    options: dict,
    request_queue,
    response_queue,
    result_queue,
) -> None:
    """One pure-engine game worker. Imports no torch and loads no checkpoint."""
    report: dict[str, Any] = {
        "worker_index": worker_index,
        "status": "ok",
        "in_owner_process": False,
    }
    channel = None
    policy = None
    try:
        bank = SetupBank.from_dict(bank_payload)
        rules = RulesConfig(**options["rules_payload"])
        reference = PolicyRef.from_dict(options["policy_ref"])
        channel = QueueInferenceChannel(
            worker_index,
            request_queue,
            response_queue,
            timeout=options["request_timeout"],
        )
        policy = RemoteNeuralPolicy(
            reference, channel, decision_mode=options["decision_mode"]
        )
        cache: dict[str, Policy] = {reference.token: policy}

        rows = []
        for entry in spec_payloads:
            spec = MatchSpec.from_dict(entry, rules=rules)
            rows.append(
                play_match(
                    spec,
                    bank=bank,
                    policies=_policies_for(spec, cache),
                    record_actions=options["record_actions"],
                    on_policy_error=options["on_policy_error"],
                    verify_invariants=options["verify_invariants"],
                ).to_dict()
            )
        report["rows"] = rows
        report["matches"] = len(rows)
    except BaseException as error:  # noqa: BLE001 -- reported to the parent verbatim
        report["status"] = "error"
        report["rows"] = []
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        report["decisions"] = policy.decisions if policy is not None else 0
        report["channel"] = channel.stats() if channel is not None else {}
        report["modules"] = worker_module_report()
        report["checkpoint_loads"] = checkpoint_load_count()
        # Rows first, then the "I am finished" marker: the parent uses the
        # marker to decide the service loop is over, and must not conclude that
        # before the rows are on their way.
        result_queue.put(report)
        request_queue.put({"kind": KIND_WORKER_DONE, "worker_index": worker_index})


def _run_across_processes(
    specs: "Sequence[MatchSpec]",
    *,
    bank: SetupBank,
    owner: InferenceOwner,
    reference: PolicyRef,
    worker_count: int,
    chunks_per_worker: int,
    record_actions: bool,
    on_policy_error: str,
    verify_invariants: bool,
    request_timeout: float,
    result_timeout: float,
):
    # Chunking is for load balance only. Every chunk is a set of already-built
    # match identities, so which worker draws which chunk cannot reach a game.
    chunk_count = min(len(specs), max(worker_count * chunks_per_worker, worker_count))
    chunks = [chunk for chunk in shard_schedule(specs, chunk_count) if chunk]
    assignments: list[list] = [[] for _ in range(worker_count)]
    for index, chunk in enumerate(chunks):
        assignments[index % worker_count].extend(spec.to_dict() for spec in chunk)

    options = {
        "record_actions": record_actions,
        "on_policy_error": on_policy_error,
        "verify_invariants": verify_invariants,
        "rules_payload": _rules_payload(specs[0].rules),
        "policy_ref": reference.to_dict(),
        "decision_mode": owner.decision_mode,
        "request_timeout": request_timeout,
    }

    # `spawn` is macOS's default and the only start method safe next to a live
    # Metal context: a forked child would inherit the parent's device state.
    context = multiprocessing.get_context("spawn")
    request_queue = context.Queue()
    result_queue = context.Queue()
    response_queues = [context.Queue() for _ in range(worker_count)]
    bank_payload = bank.to_dict()

    processes = []
    for index in range(worker_count):
        process = context.Process(
            target=_neural_worker_main,
            args=(
                index,
                assignments[index],
                bank_payload,
                options,
                request_queue,
                response_queues[index],
                result_queue,
            ),
            daemon=True,
        )
        process.start()
        processes.append(process)

    try:
        service = _serve_requests(
            owner,
            request_queue=request_queue,
            response_queues=response_queues,
            processes=processes,
        )
        reports = []
        for _ in range(worker_count):
            try:
                reports.append(result_queue.get(timeout=result_timeout))
            except queue_module.Empty as error:
                raise NeuralEvaluationError(
                    f"a game worker finished without returning its rows within "
                    f"{result_timeout:g}s"
                ) from error
    finally:
        for process in processes:
            process.join(timeout=result_timeout)
            if process.is_alive():  # pragma: no cover - a hung worker
                process.terminate()
                process.join(timeout=5)
        for shared in (request_queue, result_queue, *response_queues):
            shared.close()
            shared.join_thread()

    failed = [report for report in reports if report["status"] != "ok"]
    if failed:
        first = failed[0]
        raise NeuralEvaluationError(
            f"{len(failed)} of {worker_count} game workers failed; worker "
            f"{first['worker_index']} reported {first['error']}\n{first.get('traceback', '')}"
        )

    results = [
        MatchResult.from_dict(row)
        for report in sorted(reports, key=lambda entry: entry["worker_index"])
        for row in report["rows"]
    ]
    for report in reports:
        report.pop("rows", None)
    return results, len(chunks), reports, service


def _serve_requests(owner: InferenceOwner, *, request_queue, response_queues, processes) -> dict:
    """The owner's service loop. Runs in the parent, which is the only process
    holding Metal.

    Ordering is canonical: whatever has been drained is sorted by request
    identity before being served, so the sequence of forward passes for a given
    set of pending requests does not depend on which worker happened to arrive
    first. Under `single_request` this ordering is a reproducibility *audit*
    rather than a requirement -- each request is its own batch, so the answer
    could not depend on the order anyway -- and under `arrival_batched` it is
    what makes a batch's contents order-canonical once its membership is fixed.
    """
    active = set(range(len(response_queues)))
    drain_limit = max(owner.max_batch_size, len(response_queues))
    waits: list[float] = []
    dispatches = 0
    dead_since: dict[int, float] = {}

    while active:
        pending: list[dict] = []
        try:
            _absorb(request_queue.get(timeout=0.25), pending, active)
        except queue_module.Empty:
            _check_liveness(processes, active, dead_since)
            continue
        while len(pending) < drain_limit:
            try:
                _absorb(request_queue.get_nowait(), pending, active)
            except queue_module.Empty:
                break
        if not pending:
            continue

        now = time.time()
        waits.extend(max(0.0, now - item["enqueued_at"]) for item in pending)
        ordered = sorted(
            pending,
            key=lambda item: (item["request"].sort_key, item["worker_index"]),
        )
        if owner.batch_policy == BATCH_POLICY_SINGLE:
            groups = [[item] for item in ordered]
        else:
            groups = [
                ordered[start : start + owner.max_batch_size]
                for start in range(0, len(ordered), owner.max_batch_size)
            ]
        for group in groups:
            answers = owner.serve_batch([item["request"] for item in group])
            dispatches += 1
            for item, answer in zip(group, answers):
                response_queues[item["worker_index"]].put(answer)

    return {
        "service_dispatches": dispatches,
        "queue_wait_seconds": round(sum(waits), 6),
        "queue_wait_mean_seconds": round(sum(waits) / len(waits), 6) if waits else 0.0,
        "queue_wait_max_seconds": round(max(waits), 6) if waits else 0.0,
    }


def _absorb(message: dict, pending: list, active: set) -> None:
    kind = message.get("kind")
    if kind == KIND_WORKER_DONE:
        active.discard(message["worker_index"])
    elif kind == KIND_REQUEST:
        pending.append(message)
    else:  # pragma: no cover - defensive
        raise NeuralEvaluationError(f"unknown message on the request queue: {kind!r}")


def _check_liveness(processes, active: set, dead_since: dict) -> None:
    """Fail loudly when a worker dies without saying it finished.

    A short grace period first: a worker's last message travels through the
    queue's feeder thread, so "the process object is dead" briefly precedes "the
    message has arrived".
    """
    now = time.monotonic()
    for index in sorted(active):
        if processes[index].is_alive():
            dead_since.pop(index, None)
            continue
        first_seen = dead_since.setdefault(index, now)
        if now - first_seen > WORKER_DEATH_GRACE_SECONDS:
            raise NeuralEvaluationError(
                f"game worker {index} exited with code {processes[index].exitcode} "
                "without returning its results"
            )


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------


def field_level_mismatches(
    runs: "Mapping[str, NeuralRunSummary]", *, baseline: str
) -> list[str]:
    """Every field-level difference between the baseline run and each other run.

    Uses :func:`stratego.evaluation.match_runner.compare_results`, so the fields
    compared are exactly the Phase 4 reproducible ones: identity, setups, seeds,
    winner, terminal reason, ply count, action history and replay digest.
    """
    from .match_runner import compare_results

    reference = runs[baseline]
    problems: list[str] = []
    for label, run in runs.items():
        if label == baseline:
            continue
        problems.extend(
            f"{baseline} vs {label}: {problem}"
            for problem in compare_results(reference.results, run.results)
        )
    return problems


def sweep_digests(runs: "Mapping[str, NeuralRunSummary]") -> dict:
    """The distinct-digest counts the reproducibility gate is stated in."""
    results = {label: run.results_digest for label, run in runs.items()}
    replays = {
        label: hashlib.sha256(
            "\n".join(f"{match_id}:{digest}" for match_id, digest in sorted(run.replay_digests.items())).encode()
        ).hexdigest()
        for label, run in runs.items()
    }
    return {
        "results_digests": results,
        "distinct_results_digests": len(set(results.values())),
        "replay_digest_set_digests": replays,
        "distinct_replay_digest_sets": len(set(replays.values())),
    }


def compare_batch_policies(
    first: "NeuralRunSummary", second: "NeuralRunSummary"
) -> dict:
    """How two runs of one schedule differ. Used to *measure* batching, not to
    assume it is safe."""
    from .match_runner import compare_results

    differences = compare_results(first.results, second.results)
    changed = sorted(
        {problem.split(":")[0].replace("match ", "").strip() for problem in differences}
    )
    return {
        "identical": not differences,
        "field_differences": len(differences),
        "matches_changed": len(changed),
        "matches_compared": first.matches_run,
        "first_results_digest": first.results_digest,
        "second_results_digest": second.results_digest,
        "examples": differences[:5],
    }


__all__ = [
    "BATCH_POLICIES",
    "BATCH_POLICY_ARRIVAL",
    "BATCH_POLICY_SINGLE",
    "DECISION_MODES",
    "DECISION_MODE_CATEGORICAL",
    "DECISION_MODE_GREEDY",
    "DEFAULT_REQUEST_TIMEOUT",
    "NEURAL_DECISION_RULE_VERSION",
    "NEURAL_WORKER_VERSION",
    "REQUEST_FIELDS",
    "InferenceChannel",
    "InferenceFailure",
    "InferenceOwner",
    "InferenceRequest",
    "InferenceResponse",
    "LocalInferenceChannel",
    "NeuralEvaluationError",
    "NeuralRunSummary",
    "QueueInferenceChannel",
    "RemoteInferenceError",
    "RemoteNeuralPolicy",
    "checkpoint_load_count",
    "compare_batch_policies",
    "field_level_mismatches",
    "neural_policy_ref",
    "reset_checkpoint_load_count",
    "run_neural_schedule",
    "sweep_digests",
    "worker_module_report",
]
