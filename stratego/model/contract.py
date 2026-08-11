"""The neural model contract: versions, shapes and boundary validation.

Specification sources:

- Phase 5 single-agent instructions, sections 3 and 4.1
- `06_observation_v2_127ch.md` section 3 (the `(127, 10, 10)` tensor, perspective
  normalization, and the legality mask as a *separate* model input)
- `03_game_engine_spec.md` section 8 (`action_id = 100 * source + destination`)

What this module is for
-----------------------
Everything downstream of the engine agrees on four tensor shapes and one
outcome ordering. Writing them down once, with loud validators, is what stops a
later model from quietly changing the meaning of a tensor that an older
checkpoint was saved under.

Nothing here imports `GameState`, `PieceRecord` or `belief_target`. The whole
`stratego.model` package is deliberately free of privileged engine products:
the only engine imports are frozen constants and the pure action encoding, so
an object-graph audit cannot find a path from a model object to a hidden piece
type. Privileged belief *targets* are built in
:mod:`stratego.training.belief_targets`, which the model package never imports.

The four shapes
---------------
==============  =====================  ==============================================
Name            Shape                  Meaning
==============  =====================  ==============================================
observation     ``(B, 127, 10, 10)``   canonical model input, perspective-normalized
tokens          ``(B, 100, 127)``      pure row-major relayout of the observation
policy logits   ``(B, 10000)``         ``index = 100 * source + destination``
value logits    ``(B, 3)``             WIN, DRAW, LOSS for the acting player
belief logits   ``(B, 100, 12)``       per-square opponent piece-type logits
==============  =====================  ==============================================

Two frames meet here, on purpose
--------------------------------
The observation is **perspective-normalized** (blue's board is rotated 180
degrees, see `06_observation_v2_127ch.md` section 3), while the action space is
in **absolute engine squares** (`actions.py`). Phase 5 keeps the literal reading
of the instruction: policy logit ``a`` is the engine move ``decode_action(a)``,
with no remapping anywhere in the adapter, so the engine's legality mask applies
to the logits with zero transformation. The consequence is that a single network
playing blue reads a normalized board and must emit absolute-frame actions.
That asymmetry is recorded as :data:`POLICY_ACTION_FRAME` and flagged for Phase
6 rather than silently resolved here; the engine already ships
:func:`stratego.engine.actions.action_to_perspective` if Phase 6 chooses the
other convention, which would require a new
:data:`MODEL_CONTRACT_VERSION`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..engine.constants import (
    ACTION_SPACE_SIZE,
    BOARD_COLUMNS,
    BOARD_ROWS,
    NUM_PIECE_TYPES,
    NUM_SQUARES,
    OBSERVATION_CHANNELS,
    OBSERVATION_VERSION,
    RULES_VERSION,
)

# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

#: The Phase 5 model-boundary contract. Bumped only when a shape, an ordering or
#: a frame convention changes -- never for a weight change or a new architecture.
MODEL_CONTRACT_VERSION = "model_contract_v1"

#: The action encoding this contract is written against. The engine owns the
#: encoding itself; this constant only records which one a checkpoint assumes.
ACTION_ENCODING_VERSION = "source_destination_10000_v1"

#: Which frame the 10,000 policy logits live in. `absolute_engine_squares` means
#: logit `a` is exactly `decode_action(a)` in absolute engine squares, so no
#: remapping table exists anywhere between the model and `apply_action`.
POLICY_ACTION_FRAME = "absolute_engine_squares"

#: The frame the 100 input tokens live in. Deliberately *different* from
#: `POLICY_ACTION_FRAME`; see the module docstring.
TOKEN_SQUARE_FRAME = "perspective_normalized_squares"

# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

OBSERVATION_SHAPE: tuple[int, int, int] = (OBSERVATION_CHANNELS, BOARD_ROWS, BOARD_COLUMNS)
TOKEN_COUNT: int = NUM_SQUARES
TOKEN_FEATURES: int = OBSERVATION_CHANNELS
POLICY_LOGIT_COUNT: int = ACTION_SPACE_SIZE
BELIEF_TYPE_COUNT: int = NUM_PIECE_TYPES

#: Win / draw / loss, from the acting (model) player's perspective, in this order.
VALUE_CLASS_ORDER: tuple[str, str, str] = ("WIN", "DRAW", "LOSS")
VALUE_CLASS_COUNT: int = len(VALUE_CLASS_ORDER)
VALUE_WIN_INDEX = 0
VALUE_DRAW_INDEX = 1
VALUE_LOSS_INDEX = 2

#: Belief label used for a square that carries no target. Chosen to be the value
#: `torch.nn.functional.cross_entropy(ignore_index=...)` skips, so an unmasked
#: square can never contribute a gradient even if a caller forgets the mask.
BELIEF_IGNORE_INDEX = -100


class ModelContractError(ValueError):
    """Raised when a tensor crossing the model boundary violates the contract.

    Deliberately loud. Phase 5 forbids repairing a malformed model product or
    substituting a legal action after a failure, so every validator here raises
    rather than coercing.
    """


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------


def _describe(tensor: Any) -> str:
    """Shape/dtype text for an error message, for tensors and arrays alike."""
    shape = tuple(getattr(tensor, "shape", ()))
    dtype = getattr(tensor, "dtype", type(tensor).__name__)
    return f"shape {shape}, dtype {dtype}"


def _require_floating(tensor: torch.Tensor, name: str) -> None:
    """Dtype *family* check: float16/32/64 all pass, integers and bools do not.

    The contract fixes the shapes, not the precision -- float16 on Metal is an
    explicitly required Phase 5 configuration.
    """
    if not tensor.is_floating_point():
        raise ModelContractError(
            f"{name} must be a floating-point tensor, got dtype {tensor.dtype}"
        )


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ModelContractError(f"{name} contains a non-finite value")


def validate_observation_batch(observation: torch.Tensor, *, name: str = "observation") -> int:
    """Enforce the canonical `[B, 127, 10, 10]` model input; return `B`.

    Requires rank 4, the exact channel/row/column dimensions, a floating dtype
    and a non-empty batch. Finiteness is checked because a `NaN` observation
    would silently poison every head at once.
    """
    if not isinstance(observation, torch.Tensor):
        raise ModelContractError(
            f"{name} must be a torch.Tensor, got {type(observation).__name__}"
        )
    if observation.dim() != 4:
        raise ModelContractError(
            f"{name} must have rank 4 as (B, {OBSERVATION_CHANNELS}, {BOARD_ROWS}, "
            f"{BOARD_COLUMNS}), got {_describe(observation)}"
        )
    batch, channels, rows, columns = observation.shape
    if (channels, rows, columns) != OBSERVATION_SHAPE:
        raise ModelContractError(
            f"{name} must be (B, {OBSERVATION_CHANNELS}, {BOARD_ROWS}, {BOARD_COLUMNS}), "
            f"got {_describe(observation)}"
        )
    if batch < 1:
        raise ModelContractError(f"{name} has an empty batch dimension")
    _require_floating(observation, name)
    _require_finite(observation, name)
    return int(batch)


def validate_token_batch(tokens: torch.Tensor, *, name: str = "tokens") -> int:
    """Enforce the `[B, 100, 127]` tokenized input; return `B`."""
    if not isinstance(tokens, torch.Tensor):
        raise ModelContractError(f"{name} must be a torch.Tensor, got {type(tokens).__name__}")
    if tokens.dim() != 3:
        raise ModelContractError(
            f"{name} must have rank 3 as (B, {TOKEN_COUNT}, {TOKEN_FEATURES}), "
            f"got {_describe(tokens)}"
        )
    batch, count, features = tokens.shape
    if (count, features) != (TOKEN_COUNT, TOKEN_FEATURES):
        raise ModelContractError(
            f"{name} must be (B, {TOKEN_COUNT}, {TOKEN_FEATURES}), got {_describe(tokens)}"
        )
    if batch < 1:
        raise ModelContractError(f"{name} has an empty batch dimension")
    _require_floating(tokens, name)
    return int(batch)


def validate_policy_logits(
    logits: torch.Tensor, *, batch: int | None = None, require_finite: bool = False
) -> int:
    """Enforce `[B, 10000]` policy logits; return `B`.

    `require_finite` is off by default on purpose: a model is *allowed* to score
    an illegal index arbitrarily, and the adapter is what decides whether the
    usable (legal) entries are finite. Turning it on is for tests that state a
    stronger expectation.
    """
    if logits.dim() != 2 or logits.shape[1] != POLICY_LOGIT_COUNT:
        raise ModelContractError(
            f"policy logits must be (B, {POLICY_LOGIT_COUNT}), got {_describe(logits)}"
        )
    _require_floating(logits, "policy logits")
    if require_finite:
        _require_finite(logits, "policy logits")
    return _check_batch(int(logits.shape[0]), batch, "policy logits")


def validate_value_logits(
    logits: torch.Tensor, *, batch: int | None = None, require_finite: bool = True
) -> int:
    """Enforce `[B, 3]` WIN/DRAW/LOSS logits; return `B`."""
    if logits.dim() != 2 or logits.shape[1] != VALUE_CLASS_COUNT:
        raise ModelContractError(
            f"value logits must be (B, {VALUE_CLASS_COUNT}) in the order "
            f"{', '.join(VALUE_CLASS_ORDER)}, got {_describe(logits)}"
        )
    _require_floating(logits, "value logits")
    if require_finite:
        _require_finite(logits, "value logits")
    return _check_batch(int(logits.shape[0]), batch, "value logits")


def validate_belief_logits(
    logits: torch.Tensor, *, batch: int | None = None, require_finite: bool = True
) -> int:
    """Enforce `[B, 100, 12]` per-square belief logits; return `B`."""
    if logits.dim() != 3 or logits.shape[1:] != (TOKEN_COUNT, BELIEF_TYPE_COUNT):
        raise ModelContractError(
            f"belief logits must be (B, {TOKEN_COUNT}, {BELIEF_TYPE_COUNT}), "
            f"got {_describe(logits)}"
        )
    _require_floating(logits, "belief logits")
    if require_finite:
        _require_finite(logits, "belief logits")
    return _check_batch(int(logits.shape[0]), batch, "belief logits")


def _check_batch(actual: int, expected: int | None, name: str) -> int:
    if expected is not None and actual != expected:
        raise ModelContractError(
            f"{name} has batch {actual}, expected {expected}; every head must "
            "agree with the input batch"
        )
    return actual


# ---------------------------------------------------------------------------
# The validated output structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelOutputs:
    """One forward pass through the three heads, with its shapes checked.

    Construct through :meth:`validated` (or :func:`build_model_outputs`) so no
    consumer ever sees an unchecked head. The dataclass is frozen because a
    result that is mutated after validation is a result that was not validated.
    """

    policy_logits: torch.Tensor  # (B, 10000)
    value_logits: torch.Tensor  # (B, 3)
    belief_logits: torch.Tensor  # (B, 100, 12)

    @property
    def batch_size(self) -> int:
        return int(self.policy_logits.shape[0])

    @classmethod
    def validated(
        cls,
        policy_logits: torch.Tensor,
        value_logits: torch.Tensor,
        belief_logits: torch.Tensor,
        *,
        batch: int | None = None,
        require_finite_policy: bool = False,
    ) -> "ModelOutputs":
        """Validate all three heads against each other and return the structure."""
        resolved = validate_policy_logits(
            policy_logits, batch=batch, require_finite=require_finite_policy
        )
        validate_value_logits(value_logits, batch=resolved)
        validate_belief_logits(belief_logits, batch=resolved)
        for name, tensor in (
            ("value logits", value_logits),
            ("belief logits", belief_logits),
        ):
            if tensor.device != policy_logits.device:
                raise ModelContractError(
                    f"{name} are on {tensor.device} but policy logits are on "
                    f"{policy_logits.device}; one forward pass must not straddle devices"
                )
        return cls(policy_logits, value_logits, belief_logits)

    def all_finite(self) -> bool:
        """True when every head is finite. Policy logits may legitimately not be."""
        return bool(
            torch.isfinite(self.policy_logits).all()
            and torch.isfinite(self.value_logits).all()
            and torch.isfinite(self.belief_logits).all()
        )

    def row(self, index: int) -> "ModelOutputs":
        """One batch row, kept as a batch of 1. Used by the batch-equivalence gate."""
        return ModelOutputs(
            self.policy_logits[index : index + 1],
            self.value_logits[index : index + 1],
            self.belief_logits[index : index + 1],
        )

    def detached_cpu(self) -> "ModelOutputs":
        """A detached float32 CPU copy, for comparison and serialisation."""
        return ModelOutputs(
            self.policy_logits.detach().to("cpu", torch.float32),
            self.value_logits.detach().to("cpu", torch.float32),
            self.belief_logits.detach().to("cpu", torch.float32),
        )


def build_model_outputs(
    policy_logits: torch.Tensor,
    value_logits: torch.Tensor,
    belief_logits: torch.Tensor,
    **kwargs: Any,
) -> ModelOutputs:
    """Functional spelling of :meth:`ModelOutputs.validated`."""
    return ModelOutputs.validated(policy_logits, value_logits, belief_logits, **kwargs)


# ---------------------------------------------------------------------------
# Value semantics
# ---------------------------------------------------------------------------


def value_probabilities(value_logits: torch.Tensor) -> torch.Tensor:
    """Softmax over WIN/DRAW/LOSS. Rows sum to one and stay in class order."""
    validate_value_logits(value_logits)
    return torch.softmax(value_logits.to(torch.float32), dim=-1)


def expected_value(value_logits: torch.Tensor) -> torch.Tensor:
    """`E[v] = P(WIN) - P(LOSS)`, from the acting player's perspective.

    The contract deliberately keeps three classes rather than one scalar head:
    a draw is a distinct Stratego outcome (the battleless and absolute move
    limits both produce one), and collapsing it into a scalar would throw away
    the model's ability to say "this is drawish" instead of "this is even".
    """
    probabilities = value_probabilities(value_logits)
    return probabilities[..., VALUE_WIN_INDEX] - probabilities[..., VALUE_LOSS_INDEX]


# ---------------------------------------------------------------------------
# Machine-readable description
# ---------------------------------------------------------------------------


def contract_summary() -> dict:
    """Serialisable statement of the contract, for reports and checkpoints."""
    return {
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "rules_version": RULES_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "action_encoding_version": ACTION_ENCODING_VERSION,
        "policy_action_frame": POLICY_ACTION_FRAME,
        "token_square_frame": TOKEN_SQUARE_FRAME,
        "observation_shape": list(OBSERVATION_SHAPE),
        "token_shape": [TOKEN_COUNT, TOKEN_FEATURES],
        "policy_logits": [POLICY_LOGIT_COUNT],
        "value_logits": [VALUE_CLASS_COUNT],
        "value_class_order": list(VALUE_CLASS_ORDER),
        "belief_logits": [TOKEN_COUNT, BELIEF_TYPE_COUNT],
        "belief_ignore_index": BELIEF_IGNORE_INDEX,
        "action_index_rule": "action_id = 100 * source + destination",
    }


def numpy_observation_is_canonical(observation: np.ndarray) -> bool:
    """True when a NumPy observation has the frozen `(127, 10, 10)` single shape.

    Used by the adapter to give a clearer message than a reshape failure when a
    `PolicyInput` carries something unexpected.
    """
    return tuple(np.shape(observation)) == OBSERVATION_SHAPE


__all__ = [
    "ACTION_ENCODING_VERSION",
    "BELIEF_IGNORE_INDEX",
    "BELIEF_TYPE_COUNT",
    "MODEL_CONTRACT_VERSION",
    "OBSERVATION_SHAPE",
    "POLICY_ACTION_FRAME",
    "POLICY_LOGIT_COUNT",
    "TOKEN_COUNT",
    "TOKEN_FEATURES",
    "TOKEN_SQUARE_FRAME",
    "VALUE_CLASS_COUNT",
    "VALUE_CLASS_ORDER",
    "VALUE_DRAW_INDEX",
    "VALUE_LOSS_INDEX",
    "VALUE_WIN_INDEX",
    "ModelContractError",
    "ModelOutputs",
    "build_model_outputs",
    "contract_summary",
    "expected_value",
    "numpy_observation_is_canonical",
    "validate_belief_logits",
    "validate_observation_batch",
    "validate_policy_logits",
    "validate_token_batch",
    "validate_value_logits",
    "value_probabilities",
]
