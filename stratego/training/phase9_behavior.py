"""Phase 9 Agent 3: immutable behavior snapshots and the neural decision path.

Specification sources:

- `03_AGENT_3_SELFPLAY_COLLECTOR_AND_ROLLOUT_STORE.md` ("Behavior snapshot",
  "Neural action selection", "Behavior reproduction audit")
- `00_PHASE_9_SEQUENCE_AND_COMMON_CONTRACT.md` ("Behavior-policy semantics")
- `phase9_contract.behavior_policy_semantics()`, frozen by Agent 1: the
  temperature-1 legal softmax, the cumulative-walk selection rule, the storage
  representation, and the `1e-4` reconstruction tolerance.

What a behavior snapshot is
---------------------------
One immutable set of weights that produced games. Two kinds exist and they are
deliberately *not* interchangeable:

```text
B0nn   the learner frozen at the start of RL iteration nn
H0nn   an archive member, the opponent in a historical matchup
```

Both are :class:`BehaviorSnapshot`. Each carries its own logical identity, its
own policy token and — the part that matters — its own checkpoint SHA-256. A
historical opponent's decisions were produced by *its* checkpoint, never by the
iteration's current learner, so :func:`reproduce_decision` is always asked for
the acting side's snapshot rather than the game's. `GameRecord.
collection_checkpoint_id` names the iteration's current snapshot and answers a
different question; using it to verify a historical action would verify the
wrong network.

Why the inference batch shape is frozen
---------------------------------------
The collector batches decisions from many in-flight games into one forward
pass, and a resumed run will not group them the same way. Measured on this
machine with the accepted Phase 8 C1 checkpoint:

```text
variable batch shape   policy logits bitwise stable, value logits differ by
                       ~9e-8 (WDL softmax ~3e-8) between batch 1 and batch 8
fixed batch shape      bitwise identical for a given row at any position,
                       with any neighbours, on both CPU and MPS
```

A ~1e-8 drift is far inside the `1e-4` reproduction tolerance, but it is *not*
inside float32 storage rounding: it changes stored bytes, which changes payload
digests, which would break "a resumed run converges to the same sealed rollout
digest". So every forward pass is padded to a fixed row count carried by the
snapshot itself, and the rollout store records that count alongside the device
and refuses a resume that would change either. The shape is a parameter rather
than a hidden constant precisely so the invariant is enforced and visible in
the artifacts instead of assumed; :data:`DEFAULT_INFERENCE_BATCH_SHAPE` is what
production collection uses.

Why the realized action is drawn from the *stored* probabilities
---------------------------------------------------------------
The frozen rule walks legal actions accumulating behavior probabilities and
takes the first whose cumulative mass reaches `behavior_sample_uniform(game_id,
ply)`. Walking the float64 pre-rounding values would leave a ulp-wide window
per decision in which a verifier reading the stored float32 record would select
a different action — at ~2.5M decisions that is a real expected mismatch, not a
theoretical one. Rounding to float32 *first* and walking the stored values
makes the sealed record self-verifying by construction: the audit reproduces
the realized action from the bytes alone, and the checkpoint recomputation
separately proves the distribution those bytes claim.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..engine.constants import ACTION_SPACE_SIZE
from ..model.action_frame import model_action_to_absolute
from ..model.contract import value_probabilities
from ..model.policy_adapter import LegalityProducts, usable_logits
from ..model.tokenization import observation_batch_from_numpy, observation_to_tokens
from .phase9_contract import (
    BEHAVIOR_PROBABILITY_ABS_TOLERANCE,
    BEHAVIOR_TEMPERATURE,
)
from .phase9_seed import behavior_sample_uniform
from .serialization import to_float32
from .warmstart_checkpoint import load_model_for_evaluation

#: Rows in every production forward pass. Short batches are padded with zeros
#: and the padding rows are discarded, so a decision's logits are a function of
#: its own observation alone. 64 is the measured knee on this machine (MPS:
#: 7.5k rows/s at 64, 12.1k at 256 — the larger shape wastes more on partly
#: filled tails than it wins).
DEFAULT_INFERENCE_BATCH_SHAPE = 64

#: Observation shape the model contract fixes; used to build padding rows.
OBSERVATION_SHAPE = (127, 10, 10)

#: The only devices this collector will run inference on. Recorded in the
#: rollout manifest: two devices agree to `1e-4` but not to the last float32
#: bit, so a resume converges byte-for-byte on the device that wrote the bytes.
SUPPORTED_DEVICES = ("cpu", "mps")


class Phase9BehaviorError(RuntimeError):
    """Raised when a behavior snapshot or a neural decision is untrustworthy."""


# ---------------------------------------------------------------------------
# Snapshot identity
# ---------------------------------------------------------------------------


def file_sha256(path: "str | Path", *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    """SHA-256 of a checkpoint file exactly as it sits on disk."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_digest(model) -> str:
    """A digest over the live parameters, independent of the file on disk.

    The file digest proves which bytes were loaded; this proves the weights in
    memory did not move afterwards. Both are needed: an optimizer step would
    leave the file untouched.
    """
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        hasher.update(name.encode())
        array = tensor.detach().to("cpu", torch.float32).contiguous().numpy()
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


@dataclass
class BehaviorSnapshot:
    """One immutable set of weights, with the identity that names it.

    `logical_identity` is what the schedule talks about (`B001`, `H000`);
    `checkpoint_sha256` is what actually produced a move. The two are stored
    separately everywhere in Phase 9 precisely because they answer different
    questions — a pilot `H005` and a canonical `H005` are the same logical
    archive slot and different weights.
    """

    logical_identity: str
    policy_token: str
    checkpoint_path: str
    checkpoint_sha256: str
    device: str
    inference_batch_shape: int = DEFAULT_INFERENCE_BATCH_SHAPE
    model: object = field(repr=False, default=None)
    loaded_state_dict_digest: str = ""

    def identity(self) -> dict:
        return {
            "logical_identity": self.logical_identity,
            "policy_token": self.policy_token,
            "checkpoint_sha256": self.checkpoint_sha256,
            "state_dict_digest": self.loaded_state_dict_digest,
            "device": self.device,
            "temperature": BEHAVIOR_TEMPERATURE,
            "inference_batch_shape": int(self.inference_batch_shape),
        }

    def assert_frozen(self) -> None:
        """Refuse to continue if these weights are no longer the ones loaded.

        Called at collection start and again at seal time. Covers both ways a
        snapshot can stop being immutable: a parameter left trainable (an
        optimizer could mutate it) and a state dict that has actually changed.
        """
        trainable = [
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        ]
        if trainable:
            raise Phase9BehaviorError(
                f"behavior snapshot {self.logical_identity} has {len(trainable)} "
                f"trainable parameters (first: {trainable[0]}); a collection "
                "snapshot must be frozen against optimizer mutation"
            )
        if self.model.training:
            raise Phase9BehaviorError(
                f"behavior snapshot {self.logical_identity} is in training mode"
            )
        current = state_dict_digest(self.model)
        if current != self.loaded_state_dict_digest:
            raise Phase9BehaviorError(
                f"behavior snapshot {self.logical_identity} changed since it was "
                f"loaded: state dict digest {current} != {self.loaded_state_dict_digest}"
            )


def load_behavior_snapshot(
    checkpoint_path: "str | Path",
    *,
    logical_identity: str,
    policy_token: str,
    device: str = "cpu",
    inference_batch_shape: int = DEFAULT_INFERENCE_BATCH_SHAPE,
    expected_sha256: "str | None" = None,
    model=None,
    state_dict_digest_hint: "str | None" = None,
) -> BehaviorSnapshot:
    """Load one immutable snapshot and bind its logical identity to real bytes.

    `expected_sha256` is the binding check the mission requires before a
    historical archive member may be used: a logical identity is only allowed
    to name a checkpoint whose actual digest matches. Never invent one — if an
    archive member does not exist yet, it has no digest and cannot be
    collected against.

    `model` shares weights already loaded from the *same file* — `H000` and a
    fresh run's `B001` are one checkpoint under two logical identities, and
    loading it twice would waste memory to produce bit-identical parameters.
    The file digest is still recomputed, so sharing can never smuggle in a
    different checkpoint; `state_dict_digest_hint` only skips re-hashing
    parameters already hashed this session.
    """
    if device not in SUPPORTED_DEVICES:
        raise Phase9BehaviorError(
            f"unsupported inference device {device!r}; expected one of {list(SUPPORTED_DEVICES)}"
        )
    if not isinstance(inference_batch_shape, int) or inference_batch_shape < 1:
        raise Phase9BehaviorError(
            f"inference batch shape must be a positive int, got {inference_batch_shape!r}"
        )
    path = Path(checkpoint_path)
    if not path.exists():
        raise Phase9BehaviorError(
            f"behavior snapshot {logical_identity} names missing checkpoint {path}; "
            "a scheduled archive identity with no real immutable checkpoint cannot "
            "be collected against"
        )
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise Phase9BehaviorError(
            f"checkpoint {path} has SHA-256 {digest}, but {logical_identity} is "
            f"bound to {expected_sha256}"
        )
    if model is None:
        model, _metadata = load_model_for_evaluation(path, device=device)
        state_dict_digest_hint = None
    model.eval()
    model.requires_grad_(False)
    snapshot = BehaviorSnapshot(
        logical_identity=str(logical_identity),
        policy_token=str(policy_token),
        checkpoint_path=str(path),
        checkpoint_sha256=digest,
        device=device,
        inference_batch_shape=int(inference_batch_shape),
        model=model,
        loaded_state_dict_digest=state_dict_digest_hint or state_dict_digest(model),
    )
    snapshot.assert_frozen()
    return snapshot


# ---------------------------------------------------------------------------
# Inference at a frozen batch shape
# ---------------------------------------------------------------------------


def evaluate_observations(snapshot: BehaviorSnapshot, observations) -> tuple:
    """`(policy_logits, wdl)` for up to the snapshot's fixed batch shape.

    Always pads to that shape, so a row's outputs do not depend on how many
    other decisions happened to be in flight. Returns CPU float32 tensors:
    every selection and storage decision downstream is made on the CPU, which
    is the contract's bit-stable reference side.
    """
    shape = int(snapshot.inference_batch_shape)
    rows = np.asarray(observations, dtype=np.float32)
    if rows.ndim == 3:
        rows = rows[None, ...]
    if rows.ndim != 4 or rows.shape[1:] != OBSERVATION_SHAPE:
        raise Phase9BehaviorError(
            f"expected a batch of {OBSERVATION_SHAPE} observations, got shape {rows.shape}"
        )
    count = rows.shape[0]
    if not 1 <= count <= shape:
        raise Phase9BehaviorError(f"batch of {count} rows is outside 1..{shape}")
    padded = np.zeros((shape,) + OBSERVATION_SHAPE, dtype=np.float32)
    padded[:count] = rows
    batch = observation_batch_from_numpy(
        padded, dtype=torch.float32, device=snapshot.device
    )
    with torch.no_grad():
        outputs = snapshot.model(observation_to_tokens(batch))
    policy_logits = outputs.policy_logits[:count].detach().to("cpu", torch.float32)
    wdl = value_probabilities(outputs.value_logits)[:count].detach().to("cpu", torch.float32)
    return policy_logits, wdl


# ---------------------------------------------------------------------------
# The frozen behavior distribution and selection rule
# ---------------------------------------------------------------------------


def behavior_distribution(policy_logits_row, legality: LegalityProducts) -> tuple:
    """The stored temperature-1 legal softmax, in ascending *absolute* order.

    The network thinks in perspective-normalized squares; `trajectory_v1`
    stores `legal_action_ids` ascending in the engine's absolute frame and
    requires one probability per entry in that order. This function is the
    single place the two orders are reconciled, so nothing downstream has to
    know that blue's frames differ at all.

    Entries come back already rounded to float32: these are the bytes that will
    be stored, and the selection walks exactly them.
    """
    if BEHAVIOR_TEMPERATURE != 1.0:  # pragma: no cover - frozen at 1.0
        raise Phase9BehaviorError("the frozen behavior temperature is 1.0")
    ordered_model = tuple(sorted(legality.model))
    values = usable_logits(policy_logits_row, ordered_model).to(torch.float64)
    weights = torch.exp(values - values.max())
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise Phase9BehaviorError(
            f"the masked behavior distribution has no usable mass (sum={total!r})"
        )
    probabilities = (weights / total).tolist()

    by_absolute = {}
    for model_action, probability in zip(ordered_model, probabilities):
        absolute = model_action_to_absolute(int(model_action), legality.acting_player)
        by_absolute[absolute] = to_float32(probability)
    if set(by_absolute) != set(legality.absolute):
        raise Phase9BehaviorError(  # pragma: no cover - the frame map is a bijection
            "the model->absolute action map did not cover the engine's legal set"
        )
    return tuple(by_absolute[action] for action in legality.absolute)


def select_behavior_action(
    probabilities, legal_absolute, game_id: str, ply: int
) -> int:
    """The frozen cumulative-walk draw over the stored distribution.

    Walk ascending, accumulate, take the first action whose cumulative mass
    reaches `behavior_sample_uniform(game_id, ply)`; a float32 tail shortfall
    takes the last legal action. Pure: the same identity always draws the same
    move, with no RNG cursor anywhere.
    """
    actions = tuple(int(action) for action in legal_absolute)
    if len(actions) != len(probabilities):
        raise Phase9BehaviorError(
            f"{len(probabilities)} probabilities for {len(actions)} legal actions"
        )
    if list(actions) != sorted(actions):
        raise Phase9BehaviorError("the legal action list is not ascending")
    uniform = behavior_sample_uniform(game_id, ply)
    cumulative = 0.0
    for action, probability in zip(actions, probabilities):
        cumulative += float(probability)
        if cumulative >= uniform:
            return int(action)
    return int(actions[-1])


@dataclass(frozen=True)
class BehaviorDecision:
    """One neural decision, exactly as it will be stored and later verified."""

    game_id: str
    ply: int
    acting_player: int
    legal_action_ids: tuple
    probabilities: tuple
    win_draw_loss: tuple
    selected_action_id: int
    policy_token: str
    checkpoint_sha256: str
    snapshot_identity: str


def build_decision(
    snapshot: BehaviorSnapshot,
    *,
    game_id: str,
    ply: int,
    legality: LegalityProducts,
    policy_logits_row,
    wdl_row,
) -> BehaviorDecision:
    """Assemble one stored neural decision from one forward-pass row."""
    probabilities = behavior_distribution(policy_logits_row, legality)
    selected = select_behavior_action(probabilities, legality.absolute, game_id, ply)
    if selected not in legality.absolute:  # pragma: no cover - selection is an index
        raise Phase9BehaviorError(
            f"{game_id} ply {ply}: selected action {selected} is not legal"
        )
    wdl = tuple(to_float32(float(value)) for value in np.asarray(wdl_row).reshape(3))
    return BehaviorDecision(
        game_id=str(game_id),
        ply=int(ply),
        acting_player=int(legality.acting_player),
        legal_action_ids=tuple(legality.absolute),
        probabilities=probabilities,
        win_draw_loss=wdl,
        selected_action_id=int(selected),
        policy_token=snapshot.policy_token,
        checkpoint_sha256=snapshot.checkpoint_sha256,
        snapshot_identity=snapshot.logical_identity,
    )


# ---------------------------------------------------------------------------
# The reproduction audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReproductionRequest:
    """One stored decision to be re-derived, with everything the audit needs."""

    game_id: str
    ply: int
    acting_player: int
    observation: object
    legality: LegalityProducts
    stored_probabilities: tuple
    stored_wdl: tuple
    stored_action: int
    stored_policy_token: str
    stored_checkpoint_sha256: str


def reproduce_decisions(
    snapshot: BehaviorSnapshot,
    requests,
    *,
    tolerance: float = BEHAVIOR_PROBABILITY_ABS_TOLERANCE,
) -> list:
    """Re-derive many stored decisions of one acting snapshot, batched.

    Same comparison as :func:`reproduce_decision` on every request — this only
    shares the forward passes, which is what makes a hundred thousand-decision
    audit affordable. Batching cannot change a verdict: the batch is padded to
    the snapshot's fixed shape, so a row's logits are a function of its own
    observation.
    """
    pending = list(requests)
    reports: list[dict] = []
    shape = int(snapshot.inference_batch_shape)
    for start in range(0, len(pending), shape):
        chunk = pending[start : start + shape]
        live = [item for item in chunk if item.stored_checkpoint_sha256 == snapshot.checkpoint_sha256]
        for item in chunk:
            if item.stored_checkpoint_sha256 != snapshot.checkpoint_sha256:
                reports.append(
                    {
                        "game_id": item.game_id,
                        "ply": item.ply,
                        "verified": False,
                        "max_abs_difference": None,
                        "problems": [
                            f"stored checkpoint {item.stored_checkpoint_sha256} is not "
                            f"the acting snapshot {snapshot.checkpoint_sha256}"
                        ],
                    }
                )
        if not live:
            continue
        observations = np.stack([np.asarray(item.observation) for item in live])
        policy_logits, wdl = evaluate_observations(snapshot, observations)
        for row, item in enumerate(live):
            reports.append(
                _compare_decision(
                    snapshot,
                    item,
                    policy_logits[row],
                    wdl[row],
                    tolerance=tolerance,
                )
            )
    return reports


def reproduce_decision(
    snapshot: BehaviorSnapshot,
    *,
    game_id: str,
    ply: int,
    acting_player: int,
    observation,
    legality: LegalityProducts,
    stored_probabilities,
    stored_wdl,
    stored_action: int,
    stored_policy_token: str,
    stored_checkpoint_sha256: str,
    tolerance: float = BEHAVIOR_PROBABILITY_ABS_TOLERANCE,
) -> dict:
    """Independently re-derive one stored decision from the acting checkpoint.

    Everything the audit is required to check happens here: the acting side's
    identity, the legal set, the action frame, the distribution, the realized
    action's legality and its stored probability, the WDL output, and the
    snapshot identity. The caller supplies the snapshot of *the side that
    acted*; handing in the iteration's current learner for a historical
    opponent's decision is exactly the mistake this signature is shaped to
    make visible.
    """
    return reproduce_decisions(
        snapshot,
        [
            ReproductionRequest(
                game_id=game_id,
                ply=ply,
                acting_player=acting_player,
                observation=observation,
                legality=legality,
                stored_probabilities=tuple(stored_probabilities),
                stored_wdl=tuple(stored_wdl),
                stored_action=int(stored_action),
                stored_policy_token=str(stored_policy_token),
                stored_checkpoint_sha256=str(stored_checkpoint_sha256),
            )
        ],
        tolerance=tolerance,
    )[0]


def _compare_decision(
    snapshot: BehaviorSnapshot,
    item: ReproductionRequest,
    policy_logits_row,
    wdl_row,
    *,
    tolerance: float,
) -> dict:
    """The whole comparison for one decision, given its recomputed row."""
    game_id = item.game_id
    ply = item.ply
    legality = item.legality
    stored_probabilities = item.stored_probabilities
    stored_action = item.stored_action
    problems: list[str] = []
    acting_player = item.acting_player
    stored_policy_token = item.stored_policy_token
    stored_wdl = item.stored_wdl
    if stored_policy_token != snapshot.policy_token:
        problems.append(
            f"stored policy token {stored_policy_token!r} is not {snapshot.policy_token!r}"
        )
    if int(acting_player) != int(legality.acting_player):
        problems.append(
            f"acting player {acting_player} disagrees with the legality frame "
            f"{legality.acting_player}"
        )
    if tuple(legality.absolute) != tuple(sorted(legality.absolute)):
        problems.append("the recomputed legal set is not ascending")

    recomputed = behavior_distribution(policy_logits_row, legality)
    if len(recomputed) != len(stored_probabilities):
        problems.append(
            f"{len(stored_probabilities)} stored probabilities for "
            f"{len(recomputed)} legal actions"
        )
        max_difference = None
    else:
        differences = [
            abs(float(stored) - float(fresh))
            for stored, fresh in zip(stored_probabilities, recomputed)
        ]
        max_difference = max(differences) if differences else 0.0
        if max_difference > tolerance:
            problems.append(
                f"behavior distribution differs by {max_difference:.3e} > {tolerance:.1e}"
            )

    recomputed_action = select_behavior_action(
        stored_probabilities, legality.absolute, game_id, ply
    )
    if recomputed_action != int(stored_action):
        problems.append(
            f"the stored distribution redraws action {recomputed_action}, not the "
            f"stored {stored_action}"
        )
    if int(stored_action) not in legality.absolute:
        problems.append(f"stored action {stored_action} is not in the legal set")
    if not 0 <= int(stored_action) < ACTION_SPACE_SIZE:
        problems.append(f"stored action {stored_action} is outside the action space")

    wdl_difference = max(
        abs(float(stored) - float(fresh))
        for stored, fresh in zip(stored_wdl, np.asarray(wdl_row).reshape(3))
    )
    if wdl_difference > tolerance:
        problems.append(f"WDL output differs by {wdl_difference:.3e} > {tolerance:.1e}")

    return {
        "game_id": game_id,
        "ply": ply,
        "verified": not problems,
        "max_abs_difference": max_difference,
        "wdl_max_abs_difference": wdl_difference,
        "snapshot_identity": snapshot.logical_identity,
        "problems": problems,
    }


__all__ = [
    "DEFAULT_INFERENCE_BATCH_SHAPE",
    "OBSERVATION_SHAPE",
    "SUPPORTED_DEVICES",
    "BehaviorDecision",
    "BehaviorSnapshot",
    "Phase9BehaviorError",
    "behavior_distribution",
    "build_decision",
    "evaluate_observations",
    "file_sha256",
    "load_behavior_snapshot",
    "ReproductionRequest",
    "reproduce_decision",
    "reproduce_decisions",
    "select_behavior_action",
    "state_dict_digest",
]
