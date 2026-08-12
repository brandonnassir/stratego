"""Neural model contract (`model_contract_v2`) and end-to-end integration layer.

Public surface, in the order the data flows:

- :mod:`~stratego.model.contract` -- versions, shapes, frames, validated outputs,
  value semantics;
- :mod:`~stratego.model.action_frame` -- the *only* conversion between absolute
  engine actions and the normalized model action frame;
- :mod:`~stratego.model.tokenization` -- the single `[B,127,10,10] ->
  [B,100,127]` relayout;
- :mod:`~stratego.model.base` -- the interface every network implements;
- :mod:`~stratego.model.integration_model` -- `integration_model_v1`, the small
  Phase 5 fixture network (**not** the final architecture);
- :mod:`~stratego.model.architecture_configs` -- the Phase 6 C0-C6 candidate
  ladder, as serializable configurations;
- :mod:`~stratego.model.production_model` -- `stratego_transformer_v1`, the one
  configurable candidate network the ladder describes;
- :mod:`~stratego.model.checkpoint` -- checkpoint format, the architecture
  registry, and compatibility validation;
- :mod:`~stratego.model.losses` -- placeholder multi-head losses for the
  autograd connectivity check;
- :mod:`~stratego.model.policy_adapter` -- the reusable checkpoint-backed
  `policy_interface_v1` policy.

No module in this package imports `GameState`, `PieceRecord` or
:func:`stratego.engine.observation.belief_target`. Privileged belief *targets*
are built in :mod:`stratego.training.belief_targets`, on the training side of
the boundary.
"""

from .action_frame import (
    ActionFrameError,
    absolute_action_to_model,
    absolute_legal_actions_to_model,
    absolute_legal_mask_to_model,
    action_frame_summary,
    model_action_to_absolute,
    model_legal_actions_to_absolute,
    model_legal_mask_to_absolute,
)
from .architecture_configs import (
    ARCHITECTURE_FAMILY,
    ARCHITECTURE_FAMILY_VERSION,
    CANDIDATE_IDS,
    CANDIDATE_ROLES,
    CANDIDATES,
    FAMILY_INITIALIZATION_SEED,
    ArchitectureConfigError,
    CandidateConfig,
    architecture_family_digest,
    candidate_config,
    candidate_configs,
    candidate_table,
    config_digests,
    family_summary,
)
from .base import StrategoModel
from .checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    ArchitectureRegistration,
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointFormatError,
    accepted_under_contract_v1,
    architecture_registration,
    check_expected_identity,
    load_checkpoint,
    load_checkpoint_into,
    register_architecture,
    registered_architectures,
    save_checkpoint,
    state_dict_digest,
)
from .contract import (
    ACTION_ENCODING_VERSION,
    BELIEF_IGNORE_INDEX,
    BELIEF_TYPE_COUNT,
    ENGINE_ACTION_FRAME,
    MODEL_ACTION_FRAME,
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
from .production_model import (
    ProductionModel,
    benchmark_observation_batch,
    benchmark_token_batch,
    build_all_candidates,
    build_candidate_model,
    validate_candidate_outputs,
)
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
    "ARCHITECTURE_FAMILY",
    "ARCHITECTURE_FAMILY_VERSION",
    "BELIEF_IGNORE_INDEX",
    "BELIEF_TYPE_COUNT",
    "CANDIDATES",
    "CANDIDATE_IDS",
    "CANDIDATE_ROLES",
    "CHECKPOINT_FORMAT_VERSION",
    "DECISION_MODE_CATEGORICAL",
    "DECISION_MODE_GREEDY",
    "ENGINE_ACTION_FRAME",
    "FAMILY_INITIALIZATION_SEED",
    "MODEL_ACTION_FRAME",
    "MODEL_ARCHITECTURE_ID",
    "MODEL_CONTRACT_VERSION",
    "NEURAL_POLICY_CLASSES",
    "POLICY_ACTION_FRAME",
    "POLICY_LOGIT_COUNT",
    "TOKEN_COUNT",
    "TOKEN_FEATURES",
    "VALUE_CLASS_ORDER",
    "ActionFrameError",
    "ArchitectureConfigError",
    "ArchitectureRegistration",
    "CandidateConfig",
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
    "ProductionModel",
    "SeededCategoricalNeuralPolicy",
    "StrategoModel",
    "absolute_action_to_model",
    "absolute_legal_actions_to_model",
    "absolute_legal_mask_to_model",
    "accepted_under_contract_v1",
    "action_frame_summary",
    "architecture_family_digest",
    "architecture_registration",
    "belief_loss",
    "benchmark_observation_batch",
    "benchmark_token_batch",
    "build_all_candidates",
    "build_candidate_model",
    "build_integration_model",
    "build_neural_policy",
    "candidate_config",
    "candidate_configs",
    "candidate_table",
    "categorical_action",
    "check_expected_identity",
    "config_digests",
    "contract_summary",
    "expected_value",
    "family_summary",
    "greedy_action",
    "load_checkpoint",
    "load_checkpoint_into",
    "model_action_to_absolute",
    "model_legal_actions_to_absolute",
    "model_legal_mask_to_absolute",
    "multi_head_loss",
    "observation_to_tokens",
    "policy_loss",
    "register_architecture",
    "registered_architectures",
    "save_checkpoint",
    "state_dict_digest",
    "tokenize_numpy_observation",
    "tokens_to_observation",
    "validate_candidate_outputs",
    "value_probabilities",
    "value_loss",
]
