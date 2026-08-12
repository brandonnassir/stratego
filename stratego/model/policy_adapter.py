"""The reusable checkpoint-backed policy: model logits in, one legal action out.

Specification sources:

- Phase 5 single-agent instructions, sections 3, 4.3 and 5.2
- `stratego/evaluation/policy.py` (`policy_interface_v1`, the frozen Phase 4
  contract this adapter implements)

The path, in order
------------------
.. code-block:: text

    PolicyInput (observation + legal action product, nothing privileged)
        -> [1, 127, 10, 10] float tensor            normalized observation
        -> tokenization -> [1, 100, 127]
        -> model -> policy / value / belief logits  normalized action frame
        -> engine legality, converted to that frame
        -> greedy or seeded categorical choice      normalized action
        -> model_action_to_absolute                 absolute engine action
        -> PolicyResult -> the engine validates it independently

Where the frame change happens
------------------------------
Under `model_contract_v2` the model's 10,000 logits are indexed in the acting
player's *normalized* squares while the engine's legality products are in
absolute squares, so exactly two conversions bracket the decision: the engine's
legal actions and dense mask are converted into the model frame before anything
is scored, and the single chosen identifier is converted back before it leaves
this module. Nothing in between knows about absolute squares, and nothing
outside `stratego.model.action_frame` performs the conversion.

Selection is therefore done entirely in the model frame, including the
tie-break. "Lowest identifier among the maximal logits" now means the lowest
*normalized* identifier, which for blue is a different move than v1 would have
picked -- and that is the intended consequence: the tie-break, like everything
else the network sees, must not depend on which colour it is playing.

Two rules dominate the design
-----------------------------
**The engine is the legality authority, always.** The model may score an illegal
index arbitrarily high; that is allowed and expected. Selection happens over the
engine's own legal-action product, and the adapter additionally cross-checks the
dense mask against the legal-action tuple, because two disagreeing legality
products means something upstream is broken and a decision taken under either
one would be untrustworthy.

**A failure is a failure.** Every invalid input, non-finite usable logit, empty
legality product or shape mismatch raises. Nothing here falls back to a random
move, to the first legal move, or to "the best finite one". Phase 5 forbids
substituting a legal action after a policy/model failure, and a substituted move
is exactly the kind of bug that leaves no trace in any result table.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch

from ..engine.actions import decode_action
from ..engine.constants import ACTION_SPACE_SIZE
from ..evaluation.policy import (
    Policy,
    PolicyContractError,
    PolicyInput,
    PolicyRequirements,
    PolicyResult,
)
from .action_frame import (
    absolute_legal_actions_to_model,
    absolute_legal_mask_to_model,
    model_action_to_absolute,
)
from .checkpoint import load_checkpoint
from .contract import (
    MODEL_CONTRACT_VERSION,
    POLICY_ACTION_FRAME,
    ModelOutputs,
    expected_value,
    numpy_observation_is_canonical,
    value_probabilities,
)
from .base import StrategoModel
from .tokenization import observation_batch_from_numpy, observation_to_tokens

#: Selection modes. Both are reproducible; only the second consumes randomness.
DECISION_MODE_GREEDY = "greedy"
DECISION_MODE_CATEGORICAL = "seeded_categorical"

#: Version of the *adapter behaviour* (selection rule, tie-break, seeding), not
#: of the weights. A change here changes decisions, so it changes the policy
#: identity a match was recorded under. `0.2.0` is the move to the normalized
#: action frame: the same weights on the same position now select a different
#: move for blue, so results recorded under `0.1.0` cannot be reproduced here
#: and must not be attributed to the same policy.
NEURAL_POLICY_VERSION = "0.2.0"


class NeuralPolicyError(PolicyContractError):
    """Raised when the model path cannot produce a trustworthy decision.

    Subclasses `PolicyContractError` so the Phase 4 runner classifies it with
    every other policy failure instead of treating it as an unknown exception.
    """


# ---------------------------------------------------------------------------
# Pure selection helpers
# ---------------------------------------------------------------------------
#
# These take raw logits and a legal-action list, so the legality and numerical
# edge cases can be tested directly with crafted tensors, with no engine, no
# model and no match runner in the way.


def legal_actions_from_mask(mask: np.ndarray) -> tuple[int, ...]:
    """The action identifiers a dense `uint8` mask marks legal, ascending."""
    array = np.asarray(mask)
    if array.ndim != 1 or array.shape[0] != ACTION_SPACE_SIZE:
        raise NeuralPolicyError(
            f"legality mask must be a flat length-{ACTION_SPACE_SIZE} array, got "
            f"shape {array.shape}"
        )
    if array.dtype == bool:
        flags = array
    else:
        if not np.isin(array, (0, 1)).all():
            raise NeuralPolicyError("legality mask holds a value other than 0 or 1")
        flags = array.astype(bool)
    return tuple(int(index) for index in np.flatnonzero(flags))


def validate_legality(
    legal_actions: "Sequence[int]", mask: "np.ndarray | None" = None
) -> tuple[int, ...]:
    """Check the engine's legality product(s) and return the ascending id tuple.

    When both products are present they must describe the same set. They are
    built from the same list inside the engine, so a disagreement means the
    `PolicyInput` was assembled or mutated by something that should not have.
    """
    actions = tuple(int(action) for action in legal_actions)
    if not actions:
        raise NeuralPolicyError(
            "the legality product is empty; a non-terminal position always has at "
            "least one legal action, so this decision cannot be made"
        )
    if any(not 0 <= action < ACTION_SPACE_SIZE for action in actions):
        raise NeuralPolicyError("a legal action identifier is outside 0..9999")
    if len(set(actions)) != len(actions):
        raise NeuralPolicyError("the legal-action list contains a duplicate")

    ordered = tuple(sorted(actions))
    if mask is not None:
        from_mask = legal_actions_from_mask(mask)
        if from_mask != ordered:
            raise NeuralPolicyError(
                f"the dense legality mask marks {len(from_mask)} actions but the "
                f"legal-action list holds {len(ordered)}; the two engine legality "
                "products disagree"
            )
    return ordered


def usable_logits(policy_logits: torch.Tensor, legal_actions: "Sequence[int]") -> torch.Tensor:
    """The legal entries of one row of policy logits, as finite float32.

    Casting to float32 before the finiteness check is what makes float16 model
    output usable: a float16 logit of 70,000 is `inf` in its own dtype, and
    widening does not resurrect it, so such a row is still rejected. What
    widening does fix is comparing float16 values without double rounding.
    """
    if policy_logits.dim() == 2:
        if policy_logits.shape[0] != 1:
            raise NeuralPolicyError(
                f"expected a single-row policy output, got batch {policy_logits.shape[0]}"
            )
        policy_logits = policy_logits[0]
    if policy_logits.dim() != 1 or policy_logits.shape[0] != ACTION_SPACE_SIZE:
        raise NeuralPolicyError(
            f"policy logits must hold {ACTION_SPACE_SIZE} entries, got "
            f"shape {tuple(policy_logits.shape)}"
        )
    if not policy_logits.is_floating_point():
        raise NeuralPolicyError(f"policy logits must be floating point, got {policy_logits.dtype}")

    index = torch.as_tensor(list(legal_actions), dtype=torch.int64, device=policy_logits.device)
    selected = policy_logits.detach().index_select(0, index).to(torch.float32)
    if not bool(torch.isfinite(selected).all()):
        offenders = [
            int(legal_actions[position])
            for position in torch.nonzero(~torch.isfinite(selected)).flatten().tolist()[:5]
        ]
        raise NeuralPolicyError(
            "the model produced a non-finite logit on a legal action "
            f"(first offending action ids: {offenders}); refusing to choose a move "
            "from an untrustworthy distribution"
        )
    return selected


def greedy_action(policy_logits: torch.Tensor, legal_actions: "Sequence[int]") -> int:
    """Highest-scoring legal action; ties go to the lowest action identifier.

    The tie-break is defined rather than left to `argmax` so the choice does not
    depend on which device or kernel produced the logits. `legal_actions` is
    sorted ascending by :func:`validate_legality`, and `torch.argmax` returns the
    first maximal position, so "first" is "lowest action id" by construction --
    but the assertion below states it rather than trusting it.
    """
    ordered = validate_legality(legal_actions)
    values = usable_logits(policy_logits, ordered)
    best = float(values.max())
    for action, value in zip(ordered, values.tolist()):
        if value == best:
            return int(action)
    raise NeuralPolicyError("no legal action matched the maximum logit")  # pragma: no cover


def categorical_action(
    policy_logits: torch.Tensor,
    legal_actions: "Sequence[int]",
    rng: random.Random,
) -> tuple[int, list[float]]:
    """One seeded categorical draw over the legal actions. Returns `(action, p)`.

    Deliberately *not* Gumbel-max. Phase 3 lost a run to a Gumbel sampler whose
    uniform draw could be exactly zero, producing `+inf` noise, then `NaN` after
    adding the `-inf` illegal fill, and `argmax` ranks `NaN` first -- so the
    sampler chose an action the engine had declared illegal. Here the draw is a
    single `rng.random()` walked along an explicit cumulative sum in float64, so
    every intermediate is finite by construction and the result is an index into
    the legal list, which cannot name an illegal action at all.

    The generator is passed in and drawn from exactly once. Callers must create
    the decision stream once per decision, never once per draw.
    """
    ordered = validate_legality(legal_actions)
    # float64 for the cumulative sum, and the move to the CPU happens *before*
    # the widening rather than in one `.to(...)` call: Metal has no float64
    # dtype at all, and a combined move-and-cast is performed on the source
    # device, so it raises on an MPS tensor. Nothing numerical changes for a CPU
    # model -- the values are already exact float32 by this point, and a device
    # copy is a copy, not a rounding.
    values = usable_logits(policy_logits, ordered).to("cpu").to(torch.float64)
    # Subtracting the maximum before exponentiating keeps `exp` in range; it does
    # not change the distribution.
    weights = torch.exp(values - values.max())
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise NeuralPolicyError(
            "the masked policy distribution has no usable probability mass "
            f"(sum={total!r}); refusing to sample"
        )
    probabilities = (weights / total).tolist()

    draw = rng.random()  # the single draw for this decision
    cumulative = 0.0
    for position, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return int(ordered[position]), probabilities
    # Only reachable when floating-point rounding leaves the cumulative sum a few
    # ulps below the draw. Falling to the last legal action keeps the result inside
    # the legal set, which is the property that matters.
    return int(ordered[-1]), probabilities


# ---------------------------------------------------------------------------
# The two halves of one decision
# ---------------------------------------------------------------------------
#
# A decision is "check the engine's legality products and put them in the model
# frame", then a forward pass, then "select in the model frame and convert the
# single choice back". Both halves are pure functions of their arguments, which
# is what lets the serial adapter below and Phase 6's remote inference owner run
# *the same* legality, frame-conversion, tie-break and sampling rules instead of
# two implementations that agree until they do not.


@dataclass(frozen=True)
class LegalityProducts:
    """The engine's legality for one decision, in both frames.

    `absolute` is what the engine will accept; `model` and `model_mask` are the
    same set expressed in the acting player's normalized squares, which is the
    only frame anything downstream of here is allowed to think in.
    """

    acting_player: int
    absolute: tuple[int, ...]
    model: tuple[int, ...]
    model_mask: np.ndarray


@dataclass(frozen=True)
class ActionSelection:
    """One chosen move, named in both frames, with its decode already done."""

    absolute_action_id: int
    model_action_id: int
    source_square: int
    destination_square: int
    selected_logit: float
    legal_action_count: int


def prepare_legality(
    legal_actions: "Sequence[int]", mask: np.ndarray, acting_player: int
) -> LegalityProducts:
    """Cross-check the engine's two legality products and convert them.

    The absolute products are compared in their own frame first, so a
    disagreement is reported as what it is rather than as a confusing mismatch
    between two converted objects. The mask is then converted independently of
    the list and the two are compared again: a conversion that dropped or
    collided an entry would otherwise be invisible, since a permuted mask still
    has the right shape and the right number of ones.
    """
    absolute = validate_legality(legal_actions, mask)
    model = absolute_legal_actions_to_model(absolute, acting_player)
    model_mask = absolute_legal_mask_to_model(mask, acting_player)
    validate_legality(model, model_mask)
    return LegalityProducts(
        acting_player=int(acting_player),
        absolute=absolute,
        model=model,
        model_mask=model_mask,
    )


def select_action(
    policy_logits: torch.Tensor,
    legality: LegalityProducts,
    *,
    decision_mode: str,
    rng: "random.Random | None" = None,
) -> ActionSelection:
    """Choose one action from one row of normalized policy logits.

    `rng` is required for -- and only used by -- the seeded categorical mode, and
    must be a stream created once for this decision. The chosen normalized
    identifier is converted back exactly once, here, and is then checked against
    the engine's own absolute legal set rather than trusted.
    """
    if decision_mode == DECISION_MODE_GREEDY:
        selected = greedy_action(policy_logits, legality.model)
    elif decision_mode == DECISION_MODE_CATEGORICAL:
        if rng is None:
            raise NeuralPolicyError(
                f"decision mode {decision_mode!r} needs a per-decision random stream"
            )
        selected = categorical_action(policy_logits, legality.model, rng)[0]
    else:
        raise NeuralPolicyError(f"unknown decision mode {decision_mode!r}")

    absolute_selected = model_action_to_absolute(selected, legality.acting_player)
    if absolute_selected not in legality.absolute:
        raise NeuralPolicyError(  # pragma: no cover - unreachable via a bijection
            f"the normalized action {selected} converted to absolute action "
            f"{absolute_selected}, which the engine did not declare legal; refusing "
            "to submit it"
        )
    source, destination = decode_action(absolute_selected)
    return ActionSelection(
        absolute_action_id=int(absolute_selected),
        model_action_id=int(selected),
        source_square=int(source),
        destination_square=int(destination),
        selected_logit=float(policy_logits[selected].to(torch.float32)),
        legal_action_count=len(legality.model),
    )


def value_diagnostics(value_logits: torch.Tensor, row: int = 0) -> dict:
    """The WIN/DRAW/LOSS diagnostics a decision carries, for one batch row."""
    probabilities = value_probabilities(value_logits)[row]
    return {
        "value_win": float(probabilities[0]),
        "value_draw": float(probabilities[1]),
        "value_loss": float(probabilities[2]),
        "expected_value": float(expected_value(value_logits)[row]),
    }


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


class NeuralCheckpointPolicy(Policy):
    """A loaded checkpoint driving decisions through `policy_interface_v1`.

    Requests exactly two observer-safe products -- the 127-channel observation
    and the dense legality mask -- and nothing else. It has no `GameState`, no
    piece records, no belief target and no replay, by construction: the only
    things it ever reads are the fields of a :class:`PolicyInput`.

    Subclasses fix `policy_id` and `decision_mode`; the two shipped below are the
    greedy and seeded-categorical modes the Phase 5 gauntlet exercises.
    """

    policy_version: ClassVar[str] = NEURAL_POLICY_VERSION
    requirements: ClassVar[PolicyRequirements] = PolicyRequirements(
        observation=True,
        legal_action_mask=True,
        public_view=False,
    )
    decision_mode: ClassVar[str] = DECISION_MODE_GREEDY

    def __init__(
        self,
        model: StrategoModel,
        *,
        metadata: "Mapping[str, Any] | None" = None,
        device: "torch.device | str" = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        # Any architecture implementing the model boundary, not one named class.
        # Phase 6 candidates must reach Phase 4 evaluation through *this*
        # decision path; a second adapter would be a second set of legality and
        # frame-conversion rules to keep in step.
        if not isinstance(model, StrategoModel):
            raise NeuralPolicyError(
                f"expected a StrategoModel, got {type(model).__name__}"
            )
        self.device = torch.device(device)
        self.dtype = dtype
        # `Module.to` moves parameters in place and returns the same object, so a
        # caller that hands the same model to two policies at different
        # precisions gets one model at the last precision requested. Load a
        # checkpoint per policy (as `from_checkpoint` does) when that matters.
        self.model = model.to(device=self.device, dtype=dtype)
        self.model.eval()
        self.metadata = dict(metadata or {})

    @classmethod
    def from_checkpoint(
        cls,
        path,
        *,
        device: "torch.device | str" = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "NeuralCheckpointPolicy":
        """Build the policy from a validated checkpoint file on disk."""
        model, metadata = load_checkpoint(path, device=device, dtype=dtype)
        return cls(model, metadata=metadata, device=device, dtype=dtype)

    # -- inference ---------------------------------------------------------

    def evaluate(self, observation: np.ndarray) -> ModelOutputs:
        """One forward pass for one observation. No gradients, evaluation mode."""
        if not numpy_observation_is_canonical(observation):
            raise NeuralPolicyError(
                f"expected a (127, 10, 10) observation, got shape {np.shape(observation)}"
            )
        batch = observation_batch_from_numpy(
            observation, dtype=self.dtype, device=self.device
        )
        with torch.no_grad():
            return self.model(observation_to_tokens(batch))

    # -- decision ----------------------------------------------------------

    def decide(self, request: PolicyInput) -> PolicyResult:
        observation = request.require_observation()
        mask = request.require_legal_action_mask()
        # Legality first, in both frames, before a single kernel runs: a
        # malformed request should fail without ever reaching the device.
        legality = prepare_legality(request.legal_actions, mask, request.acting_player)

        outputs = self.evaluate(observation)
        policy_logits = outputs.policy_logits[0]

        # One stream per decision, created once and drawn from once.
        rng = (
            request.random_stream()
            if self.decision_mode == DECISION_MODE_CATEGORICAL
            else None
        )
        selection = select_action(
            policy_logits, legality, decision_mode=self.decision_mode, rng=rng
        )

        diagnostics: dict[str, Any] = {
            "mode": self.decision_mode,
            "legal_action_count": selection.legal_action_count,
            "model_architecture_id": self.model.architecture_id,
            "model_contract_version": MODEL_CONTRACT_VERSION,
            "policy_action_frame": POLICY_ACTION_FRAME,
            # Absolute squares: these describe the move the engine will apply
            # and the replay will record.
            "source_square": selection.source_square,
            "destination_square": selection.destination_square,
            "model_action_id": selection.model_action_id,
            "selected_logit": selection.selected_logit,
        }
        diagnostics.update(value_diagnostics(outputs.value_logits))
        return self.result(request, selection.absolute_action_id, diagnostics)

    # -- description -------------------------------------------------------

    def describe(self) -> dict:
        description = super().describe()
        description.update(
            {
                "decision_mode": self.decision_mode,
                "model_contract_version": MODEL_CONTRACT_VERSION,
                "policy_action_frame": POLICY_ACTION_FRAME,
                "device": str(self.device),
                "dtype": str(self.dtype),
                "model_architecture_id": self.model.architecture_id,
                "parameter_count": self.model.parameter_count(),
                "checkpoint": {
                    key: self.metadata.get(key)
                    for key in (
                        "checkpoint_path",
                        "checkpoint_file_digest",
                        "state_dict_digest",
                        "model_contract_version",
                        "rules_version",
                        "observation_version",
                        "action_encoding_version",
                        "training_iteration",
                        "training_step",
                        "creation_timestamp",
                    )
                    if key in self.metadata
                },
            }
        )
        return description


class GreedyNeuralPolicy(NeuralCheckpointPolicy):
    """Deterministic: the highest-scoring legal action, ties to the lowest id.

    The identifier says `v2` because the decision rule changed frames, not
    because the network did: these weights are still the Phase 5 fixture. A
    Phase 5 result row recorded under `integration_model_v1_greedy` describes a
    genuinely different policy and must not be compared with this one.
    """

    policy_id = "integration_model_v2_greedy"
    decision_mode = DECISION_MODE_GREEDY
    stochastic = False
    description = (
        "Integration fixture under model_contract_v2, greedy over the masked "
        "normalized policy logits. Untrained; playing strength is not meaningful."
    )


class SeededCategoricalNeuralPolicy(NeuralCheckpointPolicy):
    """Stochastic: one seeded categorical draw from the masked policy softmax."""

    policy_id = "integration_model_v2_sampled"
    decision_mode = DECISION_MODE_CATEGORICAL
    stochastic = True
    description = (
        "Integration fixture under model_contract_v2, seeded categorical draw over "
        "the masked normalized policy logits. Untrained; playing strength is not "
        "meaningful."
    )


#: The two shipped modes, for scripts that need to iterate over both.
NEURAL_POLICY_CLASSES: tuple[type[NeuralCheckpointPolicy], ...] = (
    GreedyNeuralPolicy,
    SeededCategoricalNeuralPolicy,
)


def build_neural_policy(
    checkpoint_path,
    *,
    mode: str = DECISION_MODE_GREEDY,
    device: "torch.device | str" = "cpu",
    dtype: torch.dtype = torch.float32,
) -> NeuralCheckpointPolicy:
    """Load a checkpoint into the policy class for `mode`."""
    for policy_class in NEURAL_POLICY_CLASSES:
        if policy_class.decision_mode == mode:
            return policy_class.from_checkpoint(checkpoint_path, device=device, dtype=dtype)
    known = ", ".join(policy_class.decision_mode for policy_class in NEURAL_POLICY_CLASSES)
    raise NeuralPolicyError(f"unknown decision mode {mode!r}; known modes are {known}")


__all__ = [
    "DECISION_MODE_CATEGORICAL",
    "DECISION_MODE_GREEDY",
    "NEURAL_POLICY_CLASSES",
    "NEURAL_POLICY_VERSION",
    "ActionSelection",
    "GreedyNeuralPolicy",
    "LegalityProducts",
    "NeuralCheckpointPolicy",
    "NeuralPolicyError",
    "SeededCategoricalNeuralPolicy",
    "build_neural_policy",
    "categorical_action",
    "greedy_action",
    "legal_actions_from_mask",
    "prepare_legality",
    "select_action",
    "usable_logits",
    "validate_legality",
    "value_diagnostics",
]
