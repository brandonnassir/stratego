"""Phase 17 Agent 2: the live current-policy cell, seating, and action sampling.

Specification sources: common contract section 5, Agent 2 instruction sections
3 and 4.

The Phase 16 defect this module exists to fix
---------------------------------------------
`stratego.training.phase16.collector.WindowCollector.rebind` reassigns the
*collector's* `participants` reference, but every game already in flight was
constructed with the previous `IterationParticipants` object and keeps it in
`GameRunner.participants`. The accepted runner resolves "whose network is this
move?" through `acting_snapshot_for(self.scheduled, self.participants, actor)`
on every ply -- so the resolution *is* dynamic, and it resolves against a stale
object. An in-flight game therefore keeps playing under the weights it was
created with, for as long as it lives.

Common contract section 5 calls that a hard Phase 17 blocker: "`current
policy` means current at the decision, not current when the game began."

The fix is structural, not procedural
-------------------------------------
:class:`CurrentMovePolicy` is a single mutable cell. :class:`Phase17Seating`
exposes `behavior` as a **property** that reads the cell, and *one* seating
object is shared by the whole population. There is no per-runner copy that
could go stale, so "propagate the rebind to every runner" is not a step anyone
can forget: there is nothing to propagate.

The token and the digest are different questions
------------------------------------------------
The accepted runner checks that the acting snapshot's `policy_token` equals the
scheduled side's token. A rebind must therefore not move the token -- so the
token names the seat role (`the current raw move policy`) and is constant for
the run, while `checkpoint_sha256` carries the model-state digest and moves
every iteration. Each stored transition records the digest, which is what makes
"was this decision taken under the current policy?" answerable from the data.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from ..phase9_behavior import (
    BehaviorSnapshot,
    Phase9BehaviorError,
    state_dict_digest,
)
from .move_contract import (
    CURRENT_POLICY_IDENTITY,
    CURRENT_POLICY_TOKEN,
    DOMAIN_ACTION_SAMPLING,
    Phase17MoveError,
    derive_move_seed,
    uniform_from_seed,
)

PHASE17_SNAPSHOT_VERSION = "phase17_current_raw_move_snapshot_v1"

DEFAULT_INFERENCE_BATCH_SHAPE = 64


class Phase17SeatingError(Phase17MoveError):
    """A Phase 17 seat or snapshot was used outside the current-policy contract."""


# ---------------------------------------------------------------------------
# One frozen copy of the live raw weights
# ---------------------------------------------------------------------------


def freeze_model(model, *, device: str):
    """A detached, frozen, eval-mode copy of one live model on `device`.

    A *copy*, never an alias: an aliased "snapshot" would be mutated by the
    next optimizer step and the PPO ratio's denominator would stop meaning
    what it says.
    """
    frozen = copy.deepcopy(model).to(torch.device(device))
    frozen.eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return frozen


def snapshot_from_model(
    model,
    *,
    device: str,
    inference_batch_shape: int = DEFAULT_INFERENCE_BATCH_SHAPE,
    provenance: str = "live raw move weights, copied",
) -> BehaviorSnapshot:
    """One immutable behavior snapshot of the current RAW move weights.

    RAW, never EMA: common contract section 10 makes the EMA evaluation-only,
    and a run whose behavior policy were the EMA would store a PPO denominator
    no set of weights in the run ever actually produced.
    """
    frozen = freeze_model(model, device=device)
    digest = state_dict_digest(frozen)
    snapshot = BehaviorSnapshot(
        logical_identity=CURRENT_POLICY_IDENTITY,
        policy_token=CURRENT_POLICY_TOKEN,
        checkpoint_path=f"<{PHASE17_SNAPSHOT_VERSION}: {provenance}>",
        checkpoint_sha256=digest,
        device=str(device),
        inference_batch_shape=int(inference_batch_shape),
        model=frozen,
        loaded_state_dict_digest=digest,
    )
    snapshot.assert_frozen()
    return snapshot


# ---------------------------------------------------------------------------
# The live cell
# ---------------------------------------------------------------------------


class CurrentMovePolicy:
    """The single mutable cell every in-flight decision resolves through.

    Holds one frozen raw snapshot and the iteration that produced it. `rebind`
    replaces both atomically: a reader either sees the whole old snapshot or
    the whole new one, never a half-swapped pair, because the swap is one
    attribute assignment of an already-built object.
    """

    def __init__(self, snapshot: BehaviorSnapshot, *, iteration: int = 0) -> None:
        self._assert_snapshot(snapshot)
        self._snapshot = snapshot
        self._iteration = int(iteration)
        self._rebinds = 0
        self._digest_history = [
            {"iteration": int(iteration), "model_state_digest": snapshot.checkpoint_sha256}
        ]

    @staticmethod
    def _assert_snapshot(snapshot: BehaviorSnapshot) -> None:
        if not isinstance(snapshot, BehaviorSnapshot):
            raise Phase17SeatingError(
                f"the current move policy must be a BehaviorSnapshot, got "
                f"{type(snapshot).__name__}"
            )
        if snapshot.policy_token != CURRENT_POLICY_TOKEN:
            raise Phase17SeatingError(
                f"a Phase 17 training snapshot must carry the current-policy "
                f"token {CURRENT_POLICY_TOKEN!r}, got {snapshot.policy_token!r}"
            )
        try:
            snapshot.assert_frozen()
        except Phase9BehaviorError as error:
            raise Phase17SeatingError(str(error)) from error

    @property
    def snapshot(self) -> BehaviorSnapshot:
        return self._snapshot

    @property
    def iteration(self) -> int:
        """The iteration whose weights this snapshot holds."""
        return self._iteration

    @property
    def digest(self) -> str:
        return self._snapshot.checkpoint_sha256

    @property
    def rebinds(self) -> int:
        return self._rebinds

    def rebind(self, snapshot: BehaviorSnapshot, *, iteration: int) -> dict:
        """Point the whole population at new weights, in one assignment."""
        self._assert_snapshot(snapshot)
        if int(iteration) < self._iteration:
            raise Phase17SeatingError(
                f"cannot rebind backwards from iteration {self._iteration} to "
                f"{iteration}"
            )
        before = self._snapshot.checkpoint_sha256
        self._snapshot = snapshot
        self._iteration = int(iteration)
        self._rebinds += 1
        entry = {
            "iteration": int(iteration),
            "model_state_digest": snapshot.checkpoint_sha256,
        }
        self._digest_history.append(entry)
        return {
            "rebind": self._rebinds,
            "iteration": int(iteration),
            "model_state_digest_before": before,
            "model_state_digest_after": snapshot.checkpoint_sha256,
            "changed": before != snapshot.checkpoint_sha256,
        }

    def rebind_from_model(self, model, *, iteration: int, device: "str | None" = None) -> dict:
        """Freeze the live raw model and rebind to the copy."""
        return self.rebind(
            snapshot_from_model(
                model,
                device=device or self._snapshot.device,
                inference_batch_shape=self._snapshot.inference_batch_shape,
            ),
            iteration=iteration,
        )

    def known_digests(self) -> tuple:
        """Every model-state digest this cell has ever held, oldest first."""
        return tuple(entry["model_state_digest"] for entry in self._digest_history)

    def digest_history(self) -> list:
        return [dict(entry) for entry in self._digest_history]

    def to_dict(self) -> dict:
        return {
            "snapshot_version": PHASE17_SNAPSHOT_VERSION,
            "policy_token": CURRENT_POLICY_TOKEN,
            "logical_identity": CURRENT_POLICY_IDENTITY,
            "iteration": self._iteration,
            "model_state_digest": self.digest,
            "rebinds": self._rebinds,
            "device": self._snapshot.device,
            "inference_batch_shape": int(self._snapshot.inference_batch_shape),
            "weights": "RAW; the EMA never acts in the training population",
        }


# ---------------------------------------------------------------------------
# Seating
# ---------------------------------------------------------------------------


class RefusingRulePolicies:
    """The `rules` slot of the accepted participants, wired to refuse.

    Phase 17 training has no rule, stress or handcrafted seat, so this object
    is only ever reached if something has gone structurally wrong. It raises
    rather than returning a policy, which turns immediate stop condition I5
    ("a non-current training opponent entering collection") into an exception
    at the moment of the attempt.
    """

    def get(self, policy_id: str):
        raise Phase17SeatingError(
            f"Phase 17 training is 100% current-policy self-play; a rule/stress "
            f"decision was requested for policy {policy_id!r}"
        )


class Phase17Seating:
    """The accepted `IterationParticipants` shape, backed by the live cell.

    Duck-typed rather than subclassed: `acting_snapshot_for` reads exactly
    `participants.behavior` and `participants.historical_snapshot(...)`, and
    `GameRunner._rule_decision` reads `participants.rules`. Making `behavior` a
    property is the entire fix -- every ply of every in-flight game resolves it
    afresh, so there is no stale copy anywhere in the population.
    """

    def __init__(self, cell: CurrentMovePolicy) -> None:
        if not isinstance(cell, CurrentMovePolicy):
            raise Phase17SeatingError(
                f"seating needs a CurrentMovePolicy, got {type(cell).__name__}"
            )
        self.cell = cell
        self.historical: dict = {}
        self.rules = RefusingRulePolicies()

    @property
    def behavior(self) -> BehaviorSnapshot:
        """Resolved per decision, from the cell. Never cached by a runner."""
        return self.cell.snapshot

    def historical_snapshot(self, identity: str) -> BehaviorSnapshot:
        raise Phase17SeatingError(
            f"Phase 17 training has no historical participants; {identity!r} is "
            "an evaluation instrument only"
        )


# ---------------------------------------------------------------------------
# Action sampling
# ---------------------------------------------------------------------------


def action_seed(game_id: str, ply: int) -> int:
    """The explicit, stable per-decision seed. Reproducible from `(game, ply)`."""
    return derive_move_seed(DOMAIN_ACTION_SAMPLING, str(game_id), int(ply))


def action_sampling_uniform(game_id: str, ply: int) -> float:
    """The uniform that chooses one Phase 17 action, from its own domain."""
    return uniform_from_seed(action_seed(game_id, ply))


def sample_legal_action(probabilities, legal_absolute, game_id: str, ply: int) -> dict:
    """The frozen cumulative-walk draw over the stored legal distribution.

    Identical in *rule* to the accepted Phase 9 sampler -- walk ascending,
    accumulate, take the first action whose cumulative mass reaches the
    uniform, and let a float32 tail shortfall take the last legal action -- on
    Phase 17's own uniform stream.

    Argmax is prohibited (common contract section 5). Nothing in this module
    can produce one: the walk never inspects which probability is largest.
    """
    actions = tuple(int(action) for action in legal_absolute)
    values = tuple(float(value) for value in probabilities)
    if len(actions) != len(values):
        raise Phase17SeatingError(
            f"{len(values)} probabilities for {len(actions)} legal actions"
        )
    if not actions:
        raise Phase17SeatingError("cannot sample from an empty legal set")
    if list(actions) != sorted(actions):
        raise Phase17SeatingError(
            "the legal action list is not ascending; the sampler walks it in order"
        )
    seed = action_seed(game_id, ply)
    uniform = uniform_from_seed(seed)
    cumulative = 0.0
    for index, (action, probability) in enumerate(zip(actions, values)):
        cumulative += probability
        if cumulative >= uniform:
            return {
                "action": int(action),
                "index": int(index),
                "seed": int(seed),
                "uniform": float(uniform),
                "selection": "categorical_cumulative_walk",
            }
    return {
        "action": int(actions[-1]),
        "index": len(actions) - 1,
        "seed": int(seed),
        "uniform": float(uniform),
        "selection": "categorical_cumulative_walk_float32_tail",
    }


def reproduce_sample(probabilities, legal_absolute, *, seed: int) -> int:
    """Replay one draw from the stored distribution and the stored seed alone.

    The stored row must reproduce its own action without the game, the model
    or the ply: this is the function that makes `action_seed` a claim the data
    can be checked against.
    """
    actions = tuple(int(action) for action in legal_absolute)
    values = tuple(float(value) for value in probabilities)
    if len(actions) != len(values):
        raise Phase17SeatingError(
            f"{len(values)} probabilities for {len(actions)} legal actions"
        )
    uniform = uniform_from_seed(int(seed))
    cumulative = 0.0
    for action, probability in zip(actions, values):
        cumulative += probability
        if cumulative >= uniform:
            return int(action)
    return int(actions[-1])


def seating_semantics() -> dict:
    return {
        "snapshot_version": PHASE17_SNAPSHOT_VERSION,
        "population": "Red and Blue are the same current raw move snapshot",
        "resolution": (
            "per decision, through CurrentMovePolicy; a runner holds no "
            "snapshot of its own, so an in-flight game cannot keep stale weights"
        ),
        "phase16_defect_fixed": (
            "WindowCollector.rebind swapped the collector's participants while "
            "each in-flight GameRunner kept the object it was constructed with"
        ),
        "token_vs_digest": (
            "policy_token names the seat role and is constant for the run; "
            "checkpoint_sha256 is the model-state digest and moves every rebind"
        ),
        "weights": "RAW only; the EMA never acts",
        "selection": "categorical sample over the legal set; argmax prohibited",
        "seed": "derive_move_seed(action_sampling, game_id, ply)",
    }


__all__ = [
    "CurrentMovePolicy",
    "DEFAULT_INFERENCE_BATCH_SHAPE",
    "PHASE17_SNAPSHOT_VERSION",
    "Phase17Seating",
    "Phase17SeatingError",
    "RefusingRulePolicies",
    "action_sampling_uniform",
    "action_seed",
    "freeze_model",
    "reproduce_sample",
    "sample_legal_action",
    "seating_semantics",
    "snapshot_from_model",
]
