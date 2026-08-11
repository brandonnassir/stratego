"""Phase 5 neural model contract and end-to-end integration layer.

Public surface, in the order the data flows:

- :mod:`~stratego.model.contract` -- versions, shapes, validated outputs, value
  semantics;
- :mod:`~stratego.model.tokenization` -- the single `[B,127,10,10] ->
  [B,100,127]` relayout;
- :mod:`~stratego.model.integration_model` -- `integration_model_v1`, the small
  Phase 5 fixture network (**not** the final architecture);
- :mod:`~stratego.model.checkpoint` -- checkpoint format and compatibility
  validation;
- :mod:`~stratego.model.losses` -- placeholder multi-head losses for the
  autograd connectivity check;
- :mod:`~stratego.model.policy_adapter` -- the reusable checkpoint-backed
  `policy_interface_v1` policy.

No module in this package imports `GameState`, `PieceRecord` or
:func:`stratego.engine.observation.belief_target`. Privileged belief *targets*
are built in :mod:`stratego.training.belief_targets`, on the training side of
the boundary.
"""

from .checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointFormatError,
    load_checkpoint,
    save_checkpoint,
    state_dict_digest,
)
from .contract import (
    ACTION_ENCODING_VERSION,
    BELIEF_IGNORE_INDEX,
    BELIEF_TYPE_COUNT,
    MODEL_CONTRACT_VERSION,
    POLICY_ACTION_FRAME,
    POLICY_LOGIT_COUNT,
    TOKEN_COUNT,
    TOKEN_FEATURES,
    VALUE_CLASS_ORDER,
    ModelContractError,
    ModelOutputs,
    contract_summary,
    expected_value,
    value_probabilities,
)
from .integration_model import (
    MODEL_ARCHITECTURE_ID,
    IntegrationModel,
    IntegrationModelConfig,
    build_integration_model,
)
from .losses import MultiHeadLoss, belief_loss, multi_head_loss, policy_loss, value_loss
from .policy_adapter import (
    DECISION_MODE_CATEGORICAL,
    DECISION_MODE_GREEDY,
    NEURAL_POLICY_CLASSES,
    GreedyNeuralPolicy,
    NeuralCheckpointPolicy,
    NeuralPolicyError,
    SeededCategoricalNeuralPolicy,
    build_neural_policy,
    categorical_action,
    greedy_action,
)
from .tokenization import (
    observation_to_tokens,
    tokenize_numpy_observation,
    tokens_to_observation,
)

__all__ = [
    "ACTION_ENCODING_VERSION",
    "BELIEF_IGNORE_INDEX",
    "BELIEF_TYPE_COUNT",
    "CHECKPOINT_FORMAT_VERSION",
    "DECISION_MODE_CATEGORICAL",
    "DECISION_MODE_GREEDY",
    "MODEL_ARCHITECTURE_ID",
    "MODEL_CONTRACT_VERSION",
    "NEURAL_POLICY_CLASSES",
    "POLICY_ACTION_FRAME",
    "POLICY_LOGIT_COUNT",
    "TOKEN_COUNT",
    "TOKEN_FEATURES",
    "VALUE_CLASS_ORDER",
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointFormatError",
    "GreedyNeuralPolicy",
    "IntegrationModel",
    "IntegrationModelConfig",
    "ModelContractError",
    "ModelOutputs",
    "MultiHeadLoss",
    "NeuralCheckpointPolicy",
    "NeuralPolicyError",
    "SeededCategoricalNeuralPolicy",
    "belief_loss",
    "build_integration_model",
    "build_neural_policy",
    "categorical_action",
    "contract_summary",
    "expected_value",
    "greedy_action",
    "load_checkpoint",
    "multi_head_loss",
    "observation_to_tokens",
    "policy_loss",
    "save_checkpoint",
    "state_dict_digest",
    "tokenize_numpy_observation",
    "tokens_to_observation",
    "value_probabilities",
    "value_loss",
]
