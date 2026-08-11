"""The reusable checkpoint-backed policy: model logits in, one legal action out.

Specification sources:

- Phase 5 single-agent instructions, sections 3, 4.3 and 5.2
- `stratego/evaluation/policy.py` (`policy_interface_v1`, the frozen Phase 4
  contract this adapter implements)

The path, in order
------------------
.. code-block:: text

    PolicyInput (observation + legal action product, nothing privileged)
        -> [1, 127, 10, 10] float tensor
        -> tokenization -> [1, 100, 127]
        -> model -> policy / value / belief logits
        -> authoritative engine legality mask
        -> greedy or seeded categorical choice
        -> PolicyResult -> the engine validates it independently

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
from .checkpoint import load_checkpoint
from .contract import (
    ModelOutputs,
    expected_value,
    numpy_observation_is_canonical,
    value_probabilities,
)
from .integration_model import IntegrationModel
from .tokenization import observation_batch_from_numpy, observation_to_tokens

#: Selection modes. Both are reproducible; only the second consumes randomness.
DECISION_MODE_GREEDY = "greedy"
DECISION_MODE_CATEGORICAL = "seeded_categorical"

#: Version of the *adapter behaviour* (selection rule, tie-break, seeding), not
#: of the weights. A change here changes decisions, so it changes the policy
#: identity a match was recorded under.
NEURAL_POLICY_VERSION = "0.1.0"


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
    values = usable_logits(policy_logits, ordered).to(torch.float64)
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
        model: IntegrationModel,
        *,
        metadata: "Mapping[str, Any] | None" = None,
        device: "torch.device | str" = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if not isinstance(model, IntegrationModel):
            raise NeuralPolicyError(
                f"expected an IntegrationModel, got {type(model).__name__}"
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
        legal = validate_legality(request.legal_actions, mask)

        outputs = self.evaluate(observation)
        policy_logits = outputs.policy_logits[0]

        diagnostics: dict[str, Any] = {
            "mode": self.decision_mode,
            "legal_action_count": len(legal),
            "model_architecture_id": self.model.architecture_id,
        }

        if self.decision_mode == DECISION_MODE_GREEDY:
            selected = greedy_action(policy_logits, legal)
        elif self.decision_mode == DECISION_MODE_CATEGORICAL:
            # One stream per decision, created once and drawn from once.
            selected = categorical_action(policy_logits, legal, request.random_stream())[0]
        else:  # pragma: no cover - guarded by the subclass definitions
            raise NeuralPolicyError(f"unknown decision mode {self.decision_mode!r}")

        source, destination = decode_action(selected)
        probabilities = value_probabilities(outputs.value_logits)[0]
        diagnostics.update(
            {
                "source_square": source,
                "destination_square": destination,
                "selected_logit": float(policy_logits[selected].to(torch.float32)),
                "value_win": float(probabilities[0]),
                "value_draw": float(probabilities[1]),
                "value_loss": float(probabilities[2]),
                "expected_value": float(expected_value(outputs.value_logits)[0]),
            }
        )
        return self.result(request, selected, diagnostics)

    # -- description -------------------------------------------------------

    def describe(self) -> dict:
        description = super().describe()
        description.update(
            {
                "decision_mode": self.decision_mode,
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
    """Deterministic: the highest-scoring legal action, ties to the lowest id."""

    policy_id = "integration_model_v1_greedy"
    decision_mode = DECISION_MODE_GREEDY
    stochastic = False
    description = (
        "Phase 5 integration fixture, greedy over the masked policy logits. "
        "Untrained; playing strength is not meaningful."
    )


class SeededCategoricalNeuralPolicy(NeuralCheckpointPolicy):
    """Stochastic: one seeded categorical draw from the masked policy softmax."""

    policy_id = "integration_model_v1_sampled"
    decision_mode = DECISION_MODE_CATEGORICAL
    stochastic = True
    description = (
        "Phase 5 integration fixture, seeded categorical draw over the masked "
        "policy logits. Untrained; playing strength is not meaningful."
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
    "GreedyNeuralPolicy",
    "NeuralCheckpointPolicy",
    "NeuralPolicyError",
    "SeededCategoricalNeuralPolicy",
    "build_neural_policy",
    "categorical_action",
    "greedy_action",
    "legal_actions_from_mask",
    "usable_logits",
    "validate_legality",
]
